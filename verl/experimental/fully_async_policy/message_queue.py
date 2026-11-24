# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def payload_size(sample):
    """Serialized bytes, excluding Ray framing."""
    return len(sample) if isinstance(sample, (bytes, bytearray, memoryview)) else 0


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.queue = deque(maxlen=self.max_queue_size)
        self._queue_costs = deque()
        self._completed_costs = {}
        self.generated_tokens = dict(trajectories=0, prompt_tokens=0, response_tokens=0, total_tokens=0)
        self.evicted_tokens = dict(self.generated_tokens)
        self.evicted_payload_bytes = 0
        self.completed_commit_calls = 0

        self.val_queue = deque()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0
        self.queue_payload_bytes = 0
        self.peak_queue_payload_bytes = 0
        self.total_payload_bytes = 0
        self.full_enqueue_attempts = 0
        self.batch_get_calls = 0
        self.unmeasured_payloads = 0
        self._reservations = set()
        self._committed_reservations = {}
        self.reservation_attempts = 0
        self.reservation_refusals = 0
        self.payload_commit_calls = 0
        self.duplicate_commit_calls = 0

        print(f"[MessageQueue] initialized with max_queue_size={max_queue_size}")

    def _append(self, sample, cost=None):
        size = payload_size(sample)
        self.queue.append(sample)
        self._queue_costs.append(cost)
        if cost is not None:
            for key, value in cost.items():
                self.generated_tokens[key] += value
        self.total_produced += 1
        self.queue_payload_bytes += size
        self.total_payload_bytes += size
        self.peak_queue_payload_bytes = max(self.peak_queue_payload_bytes, self.queue_payload_bytes)
        if sample is not None and not isinstance(sample, (bytes, bytearray, memoryview)):
            self.unmeasured_payloads += 1
        self._consumer_condition.notify_all()

    def _popleft(self):
        sample = self.queue.popleft()
        self._queue_costs.popleft()
        size = payload_size(sample)
        self.queue_payload_bytes -= size
        return sample

    async def commit_completed_sample(self, token: str, sample: Any, cost: dict) -> bool:
        """Atomically retain a complete group, evicting the oldest unconsumed group.

        Laminar's experience buffer evicts ready backlog rather than blocking
        generation. Here one serialized item is a whole GRPO prompt group, so
        eviction cannot split a group. Already consumed items are never touched.
        Only completed data uses this API; end markers remain lossless.
        """
        if not isinstance(token, str) or not token or len(token) > 128:
            raise ValueError("bounded nonempty commit token required")
        if not isinstance(sample, bytes):
            raise ValueError("completed group must be serialized data, not an end marker")
        if (
            set(cost) != set(self.generated_tokens)
            or any(type(v) is not int or v < 0 for v in cost.values())
            or cost["trajectories"] <= 0
            or cost["total_tokens"] != cost["prompt_tokens"] + cost["response_tokens"]
        ):
            raise ValueError("invalid complete-group token accounting")
        async with self._lock:
            self.completed_commit_calls += 1
            if token in self._committed_reservations:
                if self._committed_reservations[token] != sample or self._completed_costs.get(token) != cost:
                    raise ValueError("commit token reused with different completed data")
                self.duplicate_commit_calls += 1
                return True
            if not self.running:
                raise RuntimeError("MessageQueue is shut down")
            if self._reservations:
                raise RuntimeError("cannot mix completed eviction with lossless reservations")
            if any(item is None for item in self.queue):
                raise RuntimeError("cannot enqueue data after an end marker")
            if len(self.queue) >= self.max_queue_size:
                old_cost = self._queue_costs[0]
                if old_cost is None:
                    raise RuntimeError("cannot evict an unaccounted legacy result")
                old = self._popleft()
                self.evicted_payload_bytes += payload_size(old)
                for key, value in old_cost.items():
                    self.evicted_tokens[key] += value
                self.dropped_samples += 1
            self._append(sample, dict(cost))
            # Client settlement releases this transient idempotency record even
            # if cancellation arrives after the remote commit has taken effect.
            self._committed_reservations[token] = sample
            self._completed_costs[token] = dict(cost)
            return True

    async def try_put_sample(self, sample: Any) -> bool:
        """Lossless nonblocking enqueue; do not occupy a Ray actor slot while full."""
        async with self._lock:
            if not self.running:
                raise RuntimeError("MessageQueue is shut down")
            if len(self.queue) + len(self._reservations) >= self.max_queue_size:
                self.full_enqueue_attempts += 1
                return False
            self._append(sample)
            return True

    async def try_reserve_slot(self, token: str) -> bool:
        """Lightweight capacity polling; no completed payload crosses Ray yet."""
        if not isinstance(token, str) or not token or len(token) > 128:
            raise ValueError("bounded nonempty reservation token required")
        async with self._lock:
            if not self.running:
                raise RuntimeError("MessageQueue is shut down")
            self.reservation_attempts += 1
            if token in self._reservations or token in self._committed_reservations:
                return True
            if len(self.queue) + len(self._reservations) >= self.max_queue_size:
                self.reservation_refusals += 1
                return False
            self._reservations.add(token)
            return True

    async def commit_reserved_sample(self, token: str, sample: Any) -> bool:
        async with self._lock:
            self.payload_commit_calls += 1
            if token in self._committed_reservations:
                if self._committed_reservations[token] != sample:
                    raise ValueError("reservation already committed with a different payload")
                self.duplicate_commit_calls += 1
                return True
            if not self.running:
                raise RuntimeError("MessageQueue is shut down")
            if token not in self._reservations:
                raise RuntimeError("missing completed-result reservation")
            self._reservations.remove(token)
            self._append(sample)
            # Retain idempotency until the owning client acknowledges/settles.
            self._committed_reservations[token] = sample
            return True

    async def release_reservation(self, token: str):
        async with self._lock:
            reserved = token in self._reservations
            committed = token in self._committed_reservations
            self._reservations.discard(token)
            self._committed_reservations.pop(token, None)
            self._completed_costs.pop(token, None)
            # A committed queue item belongs to the consumer and is never removed.
            return {"released_uncommitted": reserved, "committed": committed}

    async def put_sample(self, sample: Any) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        async with self._lock:
            if not self.running:
                raise RuntimeError("MessageQueue is shut down")
            # If queue is full, remove the oldest sample (rarely happens)
            is_drop = False
            if len(self.queue) + len(self._reservations) >= self.max_queue_size:
                if not self.queue:
                    raise RuntimeError("all queue slots are reserved")
                self._popleft()
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self._append(sample)

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            if is_drop:
                return False
            return True

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None, 0

            # Get one sample
            data = self._popleft()
            self.total_consumed += 1
            return data, len(self.queue)

    async def get_samples(self, max_items: int = 32, max_bytes: int = 64 << 20):
        """Wait for one result, then drain a bounded FIFO batch without waiting.

        max_bytes bounds serialized bytes.
        One individually oversized result is returned alone for forward progress.
        Stop at the first end marker; never consume data beyond it.
        """
        if type(max_items) is not int or max_items <= 0 or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("positive integer batch bounds required")
        async with self._lock:
            while not self.queue and self.running:
                await self._consumer_condition.wait()
            self.batch_get_calls += 1
            if not self.queue:
                return [(None, 0)]
            result, total = [], 0
            while self.queue and len(result) < max_items:
                size = payload_size(self.queue[0])
                if result and total + size > max_bytes:
                    break
                sample = self._popleft()
                total += size
                self.total_consumed += 1
                result.append((sample, len(self.queue)))
                if sample is None:
                    break
            return result

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "running": self.running,
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "max_queue_size": self.max_queue_size,
                "queue_payload_bytes": self.queue_payload_bytes,
                "peak_queue_payload_bytes": self.peak_queue_payload_bytes,
                "total_payload_bytes": self.total_payload_bytes,
                "full_enqueue_attempts": self.full_enqueue_attempts,
                "batch_get_calls": self.batch_get_calls,
                "unmeasured_payloads": self.unmeasured_payloads,
                "reserved_slots": len(self._reservations),
                "committed_awaiting_client_settlement": len(self._committed_reservations),
                "reservation_attempts": self.reservation_attempts,
                "reservation_refusals": self.reservation_refusals,
                "payload_commit_calls": self.payload_commit_calls,
                "duplicate_commit_calls": self.duplicate_commit_calls,
                "completed_commit_calls": self.completed_commit_calls,
                "generated_tokens": dict(self.generated_tokens),
                "evicted_tokens": dict(self.evicted_tokens),
                "evicted_payload_bytes": self.evicted_payload_bytes,
            }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            self._queue_costs.clear()
            self.queue_payload_bytes = 0
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor
        self.transport_statistics = dict(
            enqueue_wait_seconds=0.0,
            payload_bytes_sent=0,
            payload_bytes_received=0,
            get_rpc_calls=0,
            reservation_rpc_attempts=0,
            reservation_refusals=0,
            payload_commit_attempts=0,
            commit_attempt_payload_bytes=0,
            delivery_cancelled_or_failed=0,
            reservation_cleanup_failures=0,
            reservation_cleanup_cancel_requests=0,
        )

    def get_transport_statistics(self):
        return dict(self.transport_statistics)

    async def put_sample(self, sample: Any) -> bool:
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(sample)
        return await asyncio.wrap_future(future.future())

    async def put_completed_sample(self, sample: Any, cost: dict) -> bool:
        """Bounded, nonblocking completed backlog; preserve whole prompt groups."""
        return await self._put_sample_settled(sample, cost=cost)

    async def put_sample_wait(self, sample: Any) -> bool:
        """Apply lossless backpressure, including for end markers."""
        return await self._put_sample_settled(sample)

    async def _put_sample_settled(self, sample: Any, *, cost=None) -> bool:
        """Share cancellation-safe ownership across both delivery policies."""
        size = payload_size(sample)
        start = time.perf_counter()
        token, pending, primary_error = uuid.uuid4().hex, None, None
        try:
            while cost is None:
                self.transport_statistics["reservation_rpc_attempts"] += 1
                future = self.queue_actor.try_reserve_slot.remote(token)
                pending = asyncio.wrap_future(future.future())
                if await asyncio.shield(pending):
                    break
                self.transport_statistics["reservation_refusals"] += 1
                await asyncio.sleep(0.1)
            self.transport_statistics["payload_commit_attempts"] += 1
            self.transport_statistics["commit_attempt_payload_bytes"] += size
            future = (
                self.queue_actor.commit_reserved_sample.remote(token, sample)
                if cost is None
                else self.queue_actor.commit_completed_sample.remote(token, sample, cost)
            )
            pending = asyncio.wrap_future(future.future())
            acknowledged = await asyncio.shield(pending)
            if not acknowledged:
                raise RuntimeError("reserved result commit was not acknowledged")
            # These bytes count acknowledged enqueues, not NIC traffic/Ray framing.
            self.transport_statistics["payload_bytes_sent"] += size
            return True
        except BaseException as error:
            primary_error = error
            self.transport_statistics["delivery_cancelled_or_failed"] += 1
            raise
        finally:
            cleanup = asyncio.create_task(self._settle_reservation(token, pending))
            cancelled, cleanup_error = False, None
            try:
                while True:
                    try:
                        await asyncio.shield(cleanup)
                        break
                    except asyncio.CancelledError:
                        if cleanup.cancelled():
                            raise
                        cancelled = True
                        self.transport_statistics["reservation_cleanup_cancel_requests"] += 1
            except BaseException as error:
                cleanup_error = error
            finally:
                self.transport_statistics["enqueue_wait_seconds"] += time.perf_counter() - start
            if cleanup_error is not None:
                self.transport_statistics["reservation_cleanup_failures"] += 1
                logger.error(
                    "Result reservation cleanup failed; remote capacity requires inspection",
                    exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
                )
                if primary_error is None:
                    raise cleanup_error
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note("Reservation cleanup also failed: " + repr(cleanup_error))
            if cancelled and primary_error is None:
                raise asyncio.CancelledError

    async def _settle_reservation(self, token, pending):
        async def settle():
            try:
                if pending is not None:
                    await pending
            except Exception:
                # A failed RPC may still have applied its mutation. Release below
                # is idempotent and never removes an already committed queue item.
                pass
            future = self.queue_actor.release_reservation.remote(token)
            return await asyncio.wrap_future(future.future())

        await asyncio.wait_for(settle(), timeout=60.0)

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        result = await asyncio.wrap_future(future.future())
        self.transport_statistics["get_rpc_calls"] += 1
        if result is None:
            return None
        sample, queue_len = result
        return self._measure_received(sample), queue_len

    def _measure_received(self, sample):
        self.transport_statistics["payload_bytes_received"] += payload_size(sample)
        return sample

    async def get_samples(self, max_items: int = 32, max_bytes: int = 64 << 20):
        future = self.queue_actor.get_samples.remote(max_items=max_items, max_bytes=max_bytes)
        results = await asyncio.wrap_future(future.future())
        self.transport_statistics["get_rpc_calls"] += 1
        return [(self._measure_received(sample), queue_len) for sample, queue_len in results]

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        sample, queue_len = ray.get(self.queue_actor.get_sample.remote())
        self.transport_statistics["get_rpc_calls"] += 1
        return self._measure_received(sample), queue_len

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())

    def get_queue_size_sync(self) -> int:
        """Get current queue length (sync, blocking).

        Used by callers running outside an event loop (e.g. synchronous
        post-processing hooks) that need an immediate snapshot of the queue size.
        """
        return ray.get(self.queue_actor.get_queue_size.remote())
