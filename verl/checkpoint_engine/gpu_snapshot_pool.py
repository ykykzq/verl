"""CPU metadata coordinator for immutable, TP-column-sharded GPU snapshots."""

import asyncio
import logging
import math
import re
import time

import torch

from verl.checkpoint_engine.gpu_snapshot import GPUWeightSnapshot, SnapshotSlots, plan_layout, tensor_dtype

_VISION_PREFIX_RE = re.compile(r"^(model\.)?(visual|vision_tower|vision_model)\.")
logger = logging.getLogger(__name__)


async def settle(calls):
    """Do not release owners when cancellation leaves a remote/native call alive."""
    task = asyncio.ensure_future(asyncio.gather(*calls, return_exceptions=True))
    cancelled = False
    while True:
        try:
            results = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
    for result in results:
        if isinstance(result, BaseException):
            raise result
    if cancelled:
        raise asyncio.CancelledError
    return results


class GPUSnapshotPool:
    def __init__(self, replicas, config):
        self.replicas = sorted(replicas, key=lambda r: r.replica_rank)
        sizes = {len(replica.workers) for replica in self.replicas}
        if len(sizes) != 1:
            raise ValueError("nccl_gpu requires equal TP sizes across rollout replicas")
        self.tp_size = sizes.pop()
        self.workers = [worker for replica in self.replicas for worker in replica.workers]
        self.config = config
        self.slots = SnapshotSlots(config.slots)
        self.io_lock = asyncio.Lock()
        self.reservation_lock = asyncio.Lock()
        self.shutdown_lock = asyncio.Lock()
        self.install_tasks = set()
        self.closing = False
        self.quiesced = False
        self.closed = False
        self.initialized = False
        self.plan = None
        self.fatal_error = None
        self.staging = None
        self.ingest_task = None
        self.seen = set()
        self.exported_names = set()
        self.completed = set()
        self.metrics = {}
        self.publisher = None

    async def call(self, index, method, *args):
        results = await settle([self.workers[index].gpu_snapshot_rpc.remote(method, *args)])
        return results[0]

    def healthy(self):
        if self.fatal_error:
            raise RuntimeError(f"GPU snapshot pool requires worker restart: {self.fatal_error}")

    def require_open(self):
        self.healthy()
        if self.closing or self.closed:
            raise RuntimeError("GPU snapshot pool is closing or closed")

    async def describe(self):
        self.require_open()
        descriptions = await settle([self.call(i, "describe") for i in range(len(self.workers))])
        self.plan = plan_layout(
            [d["layout"] for d in descriptions], len(self.replicas), self.tp_size, self.config.chunk_bytes
        )
        required = [sorted(d["required_names"]) for d in descriptions]
        for i in range(len(self.workers)):
            if required[i] != required[i % self.tp_size]:
                raise ValueError("RTP source tensor requirements differ across replicas")
        self.required = [set(names) for names in required[: self.tp_size]]
        self.ignored_source_names = set().union(*(d.get("ignored_source_names", []) for d in descriptions))
        peak = max(self.plan.owner_bytes) * self.config.slots
        if peak > self.config.max_bytes_per_gpu:
            raise ValueError(
                f"TP GPU snapshots require {peak} bytes/GPU for {self.config.slots} slots; "
                f"max_bytes_per_gpu={self.config.max_bytes_per_gpu}"
            )
        for layout in self.plan.tensors:
            for name, shape, dtype in layout:
                size = math.prod(shape) * torch.empty((), dtype=tensor_dtype(dtype)).element_size()
                if size > self.config.max_tensor_bytes:
                    raise ValueError(f"runtime tensor {name} requires {size} bytes, above max_tensor_bytes")
        self.unique_id = await self.call(0, "unique_id")
        return {
            "world_size": len(self.workers) + 1,
            "unique_id": self.unique_id,
            "timeout_s": self.config.operation_timeout_s,
        }

    async def initialize(self):
        self.require_open()
        if self.initialized:
            raise RuntimeError("GPU snapshot pool is already initialized")
        try:
            await settle(
                [
                    self.call(i, "initialize", len(self.workers) + 1, self.unique_id, i + 1)
                    for i in range(len(self.workers))
                ]
            )
        except BaseException as error:
            self.fatal_error = f"GPU snapshot initialization failed: {type(error).__name__}: {error}"
            raise
        self.initialized = True

    async def begin(self, version):
        self.require_open()
        async with self.reservation_lock:
            self.require_open()
            if self.staging is not None:
                raise RuntimeError("a GPU snapshot export is already staging")
            deadline = time.monotonic() + self.config.reserve_timeout_s
            while True:
                try:
                    slot = self.slots.reserve(version)
                    break
                except BufferError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"GPU snapshot slots are busy: {self.slots.versions}") from None
                    await asyncio.sleep(0.05)
                    self.require_open()
            self.staging = GPUWeightSnapshot(version, slot, self.plan.layout_hash)
            self.seen = set()
            self.exported_names = set()
            self.completed = set()
            self.started = time.perf_counter()
            self.metrics = {name: value for name, value in self.metrics.items() if name.startswith("last_install_")}
            self.metrics["export_version"] = version
            try:
                await settle([self.call(i, "reserve", slot) for i in range(len(self.workers))])
                await settle([self.call(i, "begin_conversion", version) for i in range(self.tp_size)])
            except BaseException as error:
                self.fatal_error = f"snapshot begin failed: {error}"
                raise
            return {"slot": slot, "version": version}

    async def prepare_input(self, version, name, shape, dtype):
        self.require_open()
        if self.staging is None or version != self.staging.version:
            raise ValueError("input does not match the staging GPU version")
        if self.ingest_task is not None:
            raise RuntimeError("previous GPU input has not been acknowledged")
        if name in self.exported_names:
            raise ValueError(f"duplicate exported source tensor {name}")
        destinations = [i for i, names in enumerate(self.required) if name in names]
        if not destinations:
            if name not in self.ignored_source_names and not _VISION_PREFIX_RE.match(name):
                raise ValueError(f"unexpected exported source tensor {name}")
            self.exported_names.add(name)
            return []
        self.exported_names.add(name)
        await self.io_lock.acquire()
        try:
            self.require_open()
        except BaseException:
            self.io_lock.release()
            raise
        try:
            await settle([self.call(i, "allocate", "input", shape, dtype) for i in destinations])
        except BaseException as error:
            self.fatal_error = f"GPU snapshot input allocation failed: {type(error).__name__}: {error}"
            self.io_lock.release()
            raise
        self.seen.add(name)

        async def receive_and_convert():
            try:
                started = time.perf_counter()
                await settle([self.call(i, "receive_input") for i in destinations])
                self.metrics["source_transfer_s"] = (
                    self.metrics.get("source_transfer_s", 0) + time.perf_counter() - started
                )
                await self._convert_columns_and_store(destinations, name)
            except BaseException as error:
                self.fatal_error = f"GPU snapshot input {name} failed: {type(error).__name__}: {error}"
                raise
            finally:
                self.io_lock.release()

        self.ingest_task = asyncio.create_task(receive_and_convert())
        return [i + 1 for i in destinations]

    async def finish_input(self):
        if self.ingest_task is None:
            raise RuntimeError("no GPU input is pending")
        await settle([self.ingest_task])
        self.ingest_task = None

    async def _convert_columns_and_store(self, tp_ranks, name=None, is_last=False):
        """Measure group wall time while the caller holds the shared I/O lock."""
        started = time.perf_counter()
        try:
            # Columns own disjoint workers; preserve transfer order within each column.
            await settle([self.convert_and_store(i, name, is_last) for i in tp_ranks])
        finally:
            self.metrics["conversion_and_distribution_wall_s"] = (
                self.metrics.get("conversion_and_distribution_wall_s", 0) + time.perf_counter() - started
            )

    async def convert_and_store(self, tp_rank, name=None, is_last=False):
        """Conversion/distribution metrics sum column durations, which may overlap."""
        started = time.perf_counter()
        outputs = await self.call(tp_rank, "convert", name, is_last)
        self.metrics["conversion_s"] = self.metrics.get("conversion_s", 0) + time.perf_counter() - started
        expected = {name: (shape, dtype) for name, shape, dtype in self.plan.tensors[tp_rank]}
        started = time.perf_counter()
        for output in outputs:
            key = (tp_rank, output["name"])
            if key in getattr(self, "completed", set()):
                raise ValueError(f"duplicate converted runtime tensor {key}")
            if expected.get(output["name"]) != (tuple(output["shape"]), output["dtype"]):
                raise ValueError(f"converted runtime layout mismatch for {key}")
            for chunk in self.plan.chunks:
                if chunk.tp_rank != tp_rank or chunk.name != output["name"]:
                    continue
                if chunk.owner == tp_rank:
                    await self.call(tp_rank, "store_chunk", self.staging.slot, chunk, None)
                else:
                    await settle(
                        [
                            self.call(tp_rank, "send_output", chunk, chunk.owner + 1),
                            self.call(chunk.owner, "store_chunk", self.staging.slot, chunk, tp_rank + 1),
                        ]
                    )
            self.completed.add(key)
        await self.call(tp_rank, "release_outputs")
        self.metrics["shard_distribution_s"] = (
            self.metrics.get("shard_distribution_s", 0) + time.perf_counter() - started
        )

    async def seal(self, version):
        self.require_open()
        if self.staging is None or self.staging.version != version or self.ingest_task is not None:
            raise ValueError("GPU snapshot is not ready to seal")
        async with self.io_lock:
            self.require_open()
            try:
                await self._convert_columns_and_store(range(self.tp_size), is_last=True)
                expected = {(i, name) for i, layout in enumerate(self.plan.tensors) for name, _, _ in layout}
                if self.completed != expected:
                    raise ValueError(f"GPU snapshot is incomplete: {len(expected - self.completed)} missing tensors")
                await settle([self.call(i, "finish_conversion") for i in range(self.tp_size)])
                snapshot = self.staging
                self.slots.commit(version)
                self.staging = None
                self.metrics["export_s"] = time.perf_counter() - self.started
                logger.info("GPU snapshot sealed version=%s metrics=%s", version, self.metrics)
                return snapshot
            except BaseException as error:
                self.fatal_error = f"GPU snapshot seal failed: {error}"
                raise

    def validate(self, snapshot):
        self.healthy()
        state = self.slots.versions.get(snapshot.version)
        if (
            snapshot.layout_hash != self.plan.layout_hash
            or not state
            or not state["ready"]
            or state["slot"] != snapshot.slot
        ):
            raise ValueError("GPU snapshot manifest is stale, incomplete or has the wrong layout")

    async def install(self, replica, snapshot):
        self.require_open()
        self.validate(snapshot)
        self.slots.pin(snapshot.version)
        completed = asyncio.get_running_loop().create_future()
        self.install_tasks.add(completed)
        started = time.perf_counter()
        try:
            index = next(i for i, r in enumerate(self.replicas) if r.replica_rank == replica.replica_rank)
            await settle(
                [
                    self.call(
                        index * self.tp_size + tp,
                        "install",
                        self.publisher,
                        snapshot,
                        index * self.tp_size + tp,
                        self.plan.tensors[tp],
                    )
                    for tp in range(self.tp_size)
                ]
            )
        except BaseException as error:
            # Remote/IPC failures do not prove that a GPU consumer released its source storage.
            self.fatal_error = f"GPU snapshot install failed: {type(error).__name__}: {error}"
            raise
        finally:
            self.slots.unpin(snapshot.version)
            completed.set_result(None)
            self.install_tasks.discard(completed)
            self.metrics["last_install_s"] = time.perf_counter() - started
            self.metrics["last_install_version"] = snapshot.version
            self.metrics["last_install_replica_rank"] = replica.replica_rank

    async def fetch_tensor(self, snapshot, worker_index, name, key):
        async with self.io_lock:
            self.validate(snapshot)
            if not 0 <= worker_index < len(self.workers) or not self.slots.versions[snapshot.version]["readers"]:
                raise ValueError("GPU snapshot fetch requires a pinned version and a valid target")
            tp = worker_index % self.tp_size
            descriptor = next((item for item in self.plan.tensors[tp] if item[0] == name), None)
            if descriptor is None:
                raise ValueError("runtime tensor does not belong to this TP rank")
            try:
                await self.call(worker_index, "allocate", key, descriptor[1], descriptor[2])
                for chunk in self.plan.chunks:
                    if chunk.tp_rank != tp or chunk.name != name:
                        continue
                    if chunk.owner == worker_index:
                        await self.call(worker_index, "receive_chunk", key, snapshot.slot, chunk, None)
                    else:
                        await settle(
                            [
                                self.call(chunk.owner, "send_chunk", snapshot.slot, chunk, worker_index + 1),
                                self.call(worker_index, "receive_chunk", key, snapshot.slot, chunk, chunk.owner + 1),
                            ]
                        )
            except BaseException as error:
                self.fatal_error = f"GPU snapshot fetch failed: {error}"
                raise

    async def quiesce(self):
        """Stop admission and settle owners without entering collective communicator teardown."""
        async with self.shutdown_lock:
            if self.closed or self.quiesced:
                return
            self.healthy()
            self.closing = True
            try:
                async with self.reservation_lock:
                    await settle(list(self.install_tasks))
                    if self.ingest_task is not None:
                        await self.finish_input()
                    async with self.io_lock:
                        self.healthy()
                        if self.staging is not None:
                            await settle([self.call(i, "finish_conversion") for i in range(self.tp_size)])
                            self.slots.abort(self.staging.version)
                            self.staging = None
                        self.quiesced = True
            except BaseException as error:
                self.fatal_error = f"GPU snapshot quiesce failed: {type(error).__name__}: {error}"
                raise

    async def close(self):
        """Close healthy storage peers; the source rank must close concurrently."""
        await self.quiesce()
        async with self.shutdown_lock:
            if self.closed:
                return
            self.healthy()
            try:
                await settle([self.call(i, "close") for i in range(len(self.workers))])
            except BaseException as error:
                self.fatal_error = f"GPU snapshot close failed: {type(error).__name__}: {error}"
                raise
            self.slots = SnapshotSlots(self.config.slots)
            self.initialized = False
            self.closed = True

    def status(self):
        return {
            "fatal_error": self.fatal_error,
            "closing": self.closing,
            "quiesced": self.quiesced,
            "closed": self.closed,
            "initialized": self.initialized,
            "slots": {version: dict(state) for version, state in self.slots.versions.items()},
            "latest_version": self.slots.latest,
            "staging_version": self.staging.version if self.staging is not None else None,
            "ingest_pending": self.ingest_task is not None,
            "required_source_tensors": len(set().union(*getattr(self, "required", []))),
            "seen_source_tensors": len(self.seen),
            "required_runtime_tensors": sum(len(layout) for layout in self.plan.tensors) if self.plan else 0,
            "completed_runtime_tensors": len(self.completed),
            "owner_bytes_per_version": self.plan.owner_bytes if self.plan else (),
            "metrics": dict(self.metrics),
        }
