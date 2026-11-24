"""Bounded snapshot buckets with independent export and TP-column fetch lanes.

Only descriptors cross Ray. CUDA storage remains owned until the corresponding
NCCL/IPC acknowledgement. The legacy path is retained with bucket_bytes=0.
"""

import asyncio
import math
import time

import torch

from verl.checkpoint_engine.gpu_snapshot import tensor_dtype
from verl.checkpoint_engine.gpu_snapshot_pool import _VISION_PREFIX_RE, GPUSnapshotPool, settle
from verl.checkpoint_engine.gpu_snapshot_transport import DirectedNCCL, GPUSnapshotWorker


def byte_layout(tensors):
    entries, end = [], 0
    names = set()
    for name, shape, dtype in tensors:
        if name in names or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid or duplicate bucket tensor")
        names.add(name)
        offset = (end + 15) // 16 * 16
        size = math.prod(shape) * torch.empty((), dtype=tensor_dtype(dtype)).element_size()
        entries.append((name, tuple(shape), dtype, offset, size))
        end = offset + size
    return entries, end


def tensor_buckets(tensors, capacity):
    group, size = [], 0
    for item in tensors:
        _, nbytes = byte_layout([item])
        needed = (size + 15) // 16 * 16 + nbytes
        if group and needed > capacity:
            yield group
            group, size = [], 0
        group.append(item)
        size = (size + 15) // 16 * 16 + nbytes
    if group:
        yield group


def views(buffer, entries):
    return [
        (name, buffer[offset : offset + size].view(tensor_dtype(dtype)).reshape(shape))
        for name, shape, dtype, offset, size in entries
    ]


class BucketSnapshotPool(GPUSnapshotPool):
    def __init__(self, replicas, config):
        super().__init__(replicas, config)
        self.fetch_locks = [asyncio.Lock() for _ in range(self.tp_size)]
        self.fetch_timings = {}

    async def initialize(self):
        await super().initialize()
        try:
            ids = await settle([self.call(tp, "unique_id") for tp in range(self.tp_size)])
            await settle(
                [
                    self.call(i, "initialize_fetch", len(self.replicas), ids[i % self.tp_size], i // self.tp_size)
                    for i in range(len(self.workers))
                ]
            )
        except BaseException as error:
            self.fatal_error = f"fetch communicator initialization failed: {error}"
            raise

    async def prepare_bucket(self, version, tensors):
        self.require_open()
        if self.staging is None or version != self.staging.version or self.ingest_task is not None:
            raise ValueError("bucket does not match an idle staging version")
        entries, size = byte_layout(tensors)
        if not entries or size > self.config.max_tensor_bytes:
            raise ValueError("empty or oversized source bucket")
        columns = {}
        for tp, required in enumerate(self.required):
            selected = [entry for entry in entries if entry[0] in required]
            if selected:
                columns[tp] = selected
        for name, *_ in entries:
            if name in self.exported_names:
                raise ValueError(f"duplicate exported source tensor {name}")
            if not any(name in required for required in self.required):
                if name not in self.ignored_source_names and not _VISION_PREFIX_RE.match(name):
                    raise ValueError(f"unexpected exported source tensor {name}")
        self.exported_names.update(entry[0] for entry in entries)
        if not columns:
            return []
        started = time.perf_counter()
        await self.io_lock.acquire()
        self.metrics["export_lock_wait_s"] = self.metrics.get("export_lock_wait_s", 0) + time.perf_counter() - started
        try:
            self.require_open()
            await settle([self.call(i, "allocate", "input", (size,), "torch.uint8") for i in columns])
        except BaseException as error:
            self.fatal_error = f"bucket allocation failed: {error}"
            self.io_lock.release()
            raise
        self.seen.update(entry[0] for selected in columns.values() for entry in selected)

        async def ingest():
            try:
                start = time.perf_counter()
                await settle([self.call(i, "receive_input") for i in columns])
                self.metrics["source_transfer_s"] = (
                    self.metrics.get("source_transfer_s", 0) + time.perf_counter() - start
                )
                start = time.perf_counter()
                await settle([self.convert_and_store(i, selected) for i, selected in columns.items()])
                self.metrics["conversion_and_distribution_wall_s"] = (
                    self.metrics.get("conversion_and_distribution_wall_s", 0) + time.perf_counter() - start
                )
                self.metrics["source_buckets"] = self.metrics.get("source_buckets", 0) + 1
            except BaseException as error:
                self.fatal_error = f"bucket ingestion failed: {error}"
                raise
            finally:
                self.io_lock.release()

        self.ingest_task = asyncio.create_task(ingest())
        return [i + 1 for i in columns]

    async def convert_and_store(self, tp_rank, entries=None, is_last=False):
        start = time.perf_counter()
        outputs = await self.call(tp_rank, "convert_bucket", entries or [], is_last)
        self.metrics["conversion_s"] = self.metrics.get("conversion_s", 0) + time.perf_counter() - start
        expected = {name: (shape, dtype) for name, shape, dtype in self.plan.tensors[tp_rank]}
        output_names = set()
        for output in outputs:
            key = (tp_rank, output["name"])
            if key in self.completed or output["name"] in output_names:
                raise ValueError(f"duplicate converted tensor {key}")
            if expected.get(output["name"]) != (tuple(output["shape"]), output["dtype"]):
                raise ValueError(f"converted layout mismatch {key}")
            output_names.add(output["name"])
        start = time.perf_counter()
        chunks = [c for c in self.plan.chunks if c.tp_rank == tp_rank and c.name in output_names]
        for owner in sorted({c.owner for c in chunks}):
            selected = [c for c in chunks if c.owner == owner]
            if owner == tp_rank:
                await self.call(owner, "store_chunks", self.staging.slot, selected, None)
            else:
                await settle(
                    [
                        self.call(tp_rank, "send_outputs", selected, owner + 1),
                        self.call(owner, "store_chunks", self.staging.slot, selected, tp_rank + 1),
                    ]
                )
        self.completed.update((tp_rank, name) for name in output_names)
        await self.call(tp_rank, "release_outputs")
        self.metrics["shard_distribution_s"] = self.metrics.get("shard_distribution_s", 0) + time.perf_counter() - start

    async def fetch_bucket(self, snapshot, worker_index, tensors, key):
        self.validate(snapshot)
        if not 0 <= worker_index < len(self.workers) or not self.slots.versions[snapshot.version]["readers"]:
            raise ValueError("bucket fetch requires a pinned version and valid target")
        tp = worker_index % self.tp_size
        expected = {name: (tuple(shape), dtype) for name, shape, dtype in self.plan.tensors[tp]}
        entries, size = byte_layout(tensors)
        if not entries or size > self.config.max_tensor_bytes:
            raise ValueError("empty or oversized fetch bucket")
        if any(expected.get(name) != (tuple(shape), dtype) for name, shape, dtype in tensors):
            raise ValueError("runtime bucket does not belong to TP rank")
        start = time.perf_counter()
        async with self.fetch_locks[tp]:
            self.validate(snapshot)
            locked = time.perf_counter()
            try:
                await self.call(worker_index, "allocate", key, (size,), "torch.uint8")
                names = {entry[0] for entry in entries}
                chunks = [c for c in self.plan.chunks if c.tp_rank == tp and c.name in names]
                for owner in sorted({c.owner for c in chunks}):
                    selected = [c for c in chunks if c.owner == owner]
                    if owner == worker_index:
                        await self.call(worker_index, "receive_chunks", key, snapshot.slot, selected, entries, None)
                    else:
                        await settle(
                            [
                                self.call(owner, "send_chunks", snapshot.slot, selected, worker_index // self.tp_size),
                                self.call(
                                    worker_index,
                                    "receive_chunks",
                                    key,
                                    snapshot.slot,
                                    selected,
                                    entries,
                                    owner // self.tp_size,
                                ),
                            ]
                        )
            except BaseException as error:
                self.fatal_error = f"bucket fetch failed: {error}"
                raise
            metric = self.fetch_timings.setdefault(
                (snapshot.version, worker_index), {"lock_wait_s": 0, "transfer_s": 0, "buckets": 0}
            )
            metric["lock_wait_s"] += locked - start
            metric["transfer_s"] += time.perf_counter() - locked
            metric["buckets"] += 1

    def status(self):
        result = super().status()
        result["fetch_timings"] = {f"{v}:{rank}": dict(m) for (v, rank), m in self.fetch_timings.items()}
        return result


class BucketSnapshotWorker(GPUSnapshotWorker):
    def __init__(self, adapter, config):
        super().__init__(adapter, config)
        self.fetch_transport = DirectedNCCL(config.operation_timeout_s)

    async def initialize_fetch(self, world_size, unique_id, rank):
        await self.fetch_transport.initialize(world_size, unique_id, rank)

    async def close(self):
        if self.transport.failed or self.fetch_transport.failed:
            raise RuntimeError("snapshot transport failed; retain storage and restart")
        await self.fetch_transport.close()
        await super().close()

    async def convert_bucket(self, entries, is_last=False):
        from verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer import rebuild_ipc_tensor, serialize_ipc_tensor

        payload = serialize_ipc_tensor(self.buffers["input"]) if entries else None
        result = await self.server.convert_gpu_snapshot_bucket.remote(entries, payload, is_last)
        self.outputs = {entry["name"]: rebuild_ipc_tensor(entry["payload"], self.transport.device) for entry in result}
        self._release_buffer("input")
        return [{k: v for k, v in entry.items() if k != "payload"} for entry in result]

    async def store_chunks(self, slot, chunks, source):
        occupied = sum(t.nbytes for values in self.slots.values() for t in values.values())
        required = sum(c.numel * torch.empty((), dtype=tensor_dtype(c.dtype)).element_size() for c in chunks)
        if occupied + required > self.config.max_bytes_per_gpu or any(c.key in self.slots[slot] for c in chunks):
            raise ValueError("duplicate chunk or storage budget exceeded")
        operations = []
        for chunk in chunks:
            if source is None:
                tensor = self.outputs[chunk.name].reshape(-1)[chunk.offset : chunk.offset + chunk.numel].clone()
            else:
                tensor = torch.empty(chunk.numel, dtype=tensor_dtype(chunk.dtype), device="cuda")
                operations.append((tensor, source, True))
            self.slots[slot][chunk.key] = tensor
        if operations:
            await self.transport.transfer_many(operations)
        else:
            torch.cuda.current_stream().synchronize()

    async def send_outputs(self, chunks, destination):
        await self.transport.transfer_many(
            [(self.outputs[c.name].reshape(-1)[c.offset : c.offset + c.numel], destination, False) for c in chunks]
        )

    async def send_chunks(self, slot, chunks, destination):
        await self.fetch_transport.transfer_many([(self.slots[slot][c.key], destination, False) for c in chunks])

    async def receive_chunks(self, key, slot, chunks, entries, source):
        tensors = dict(views(self.buffers[key], entries))
        operations = []
        for chunk in chunks:
            target = tensors[chunk.name].reshape(-1)[chunk.offset : chunk.offset + chunk.numel]
            if source is None:
                target.copy_(self.slots[slot][chunk.key])
            else:
                operations.append((target, source, True))
        if operations:
            await self.fetch_transport.transfer_many(operations)
        else:
            torch.cuda.current_stream().synchronize()

    async def install(self, publisher, snapshot, worker_index, tensors):
        async def weights():
            for number, group in enumerate(tensor_buckets(tensors, self.config.bucket_bytes)):
                key = f"install:{snapshot.version}:bucket:{number}"
                entries, _ = byte_layout(group)
                await publisher.gpu_fetch_bucket.remote(snapshot, worker_index, group, key)
                try:
                    for item in views(self.buffers[key], entries):
                        yield item
                        del item  # Do not retain the previous bucket during the next fetch.
                finally:
                    self._release_buffer(key)

        iterator = weights()
        try:
            await self.adapter.update_weights(
                iterator,
                global_steps=snapshot.version,
                wire_format=getattr(self.adapter, "snapshot_wire_format", "runtime_tensors"),
            )
        except BaseException as error:
            self.transport.failed = f"bucket installation failed: {error}"
            self.buffer_released.set()
            raise
        finally:
            await iterator.aclose()


async def export_buckets(worker, version, publisher, weights, dtype, config):
    """All FSDP ranks drain identical collectives even after a source failure."""
    source = torch.distributed.get_rank() == 0
    failure = None
    buffer, descriptors, used = None, [], 0
    metrics = {"buckets": 0, "source_bytes": 0, "send_and_convert_s": 0}

    async def flush():
        nonlocal buffer, descriptors, used
        if not descriptors:
            return
        start = time.perf_counter()
        destinations = (await settle([publisher.prepare_gpu_bucket.remote(version, descriptors)]))[0]
        if destinations:
            await worker._gpu_snapshot_sender.transfer_many([(buffer[:used], d, False) for d in destinations])
            await settle([publisher.finish_gpu_input.remote()])
        metrics["send_and_convert_s"] += time.perf_counter() - start
        metrics["buckets"] += 1
        metrics["source_bytes"] += used
        buffer, descriptors, used = None, [], 0

    started = time.perf_counter()
    try:
        for name, tensor in weights:
            if not source or failure is not None:
                continue
            try:
                tensor = tensor.detach()
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                tensor = tensor.contiguous()
                if tensor.device.type != "cuda" or tensor.nbytes > config.max_tensor_bytes:
                    raise ValueError("source tensor must fit GPU working budget")
                offset = (used + 15) // 16 * 16
                if descriptors and offset + tensor.nbytes > config.bucket_bytes:
                    await flush()
                    offset = 0
                if buffer is None:
                    buffer = torch.empty(
                        max(config.bucket_bytes, tensor.nbytes), dtype=torch.uint8, device=tensor.device
                    )
                buffer[offset : offset + tensor.nbytes].copy_(tensor.reshape(-1).view(torch.uint8))
                used = offset + tensor.nbytes
                descriptors.append((name, tuple(tensor.shape), str(tensor.dtype)))
            except BaseException as error:
                failure = error
        if source and failure is None:
            await flush()
    except BaseException as error:
        failure = failure or error
    if failure is not None:
        if source:
            await settle([publisher.fail_gpu_snapshot.remote(f"bucket source export failed: {failure}")])
        raise failure
    metrics["source_export_s"] = time.perf_counter() - started
    return metrics if source else None
