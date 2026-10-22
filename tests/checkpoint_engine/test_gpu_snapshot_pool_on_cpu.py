"""Metadata-only RPC harness; CPU tensors travel exclusively through the fake fabric."""

import asyncio
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from verl.checkpoint_engine.gpu_snapshot_pool import GPUSnapshotPool
from verl.checkpoint_engine.gpu_snapshot_transport import DirectedNCCL, GPUSnapshotWorker


def assert_metadata(value):
    assert not isinstance(value, torch.Tensor), "tensor payload entered an RPC"
    if is_dataclass(value):
        for field in fields(value):
            assert_metadata(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            assert_metadata(item)
    elif isinstance(value, list | tuple):
        for item in value:
            assert_metadata(item)


class Worker:
    def __init__(self, harness, index):
        self.harness, self.index = harness, index
        self.slots, self.buffers, self.outputs, self.installed = {}, {}, {}, {}
        self.ignored_source_names = []
        self.gates, self.failures = {}, {}
        self.entered = defaultdict(asyncio.Event)
        self.gpu_snapshot_rpc = SimpleNamespace(remote=self.remote)

    def remote(self, method, *args):
        assert_metadata(args)
        task = asyncio.create_task(self.dispatch(method, *args))
        self.harness.tasks.add(task)
        task.add_done_callback(self.harness.tasks.discard)
        # Ray caller cancellation does not stop the remote actor's operation.
        return asyncio.shield(task)

    async def dispatch(self, method, *args):
        self.entered[method].set()
        if method in self.gates:
            await self.gates[method].wait()
        if method in self.failures:
            raise self.failures[method]
        result = await getattr(self, method)(*args)
        assert_metadata(result)
        return result

    async def describe(self):
        return {
            "layout": [{"name": "layer", "shape": [6], "dtype": "torch.float32"}],
            "required_names": ["source"],
            "ignored_source_names": self.ignored_source_names,
        }

    async def unique_id(self):
        return (1, 2, 3)

    async def initialize(self, world_size, unique_id, rank):
        assert (world_size, unique_id, rank) == (5, (1, 2, 3), self.index + 1)

    async def reserve(self, slot):
        self.slots[slot] = {}

    async def begin_conversion(self, version):
        self.outputs.clear()

    async def allocate(self, key, shape, dtype):
        assert key not in self.buffers
        self.buffers[key] = torch.empty(shape, dtype=getattr(torch, dtype.removeprefix("torch.")))

    async def receive_input(self):
        self.buffers["input"].copy_(await self.harness.receive(0, self.index + 1, "input"))

    async def convert(self, name=None, is_last=False):
        if is_last:
            return []
        offset = (self.index % 2) * 6
        self.outputs["layer"] = self.buffers.pop("input")[offset : offset + 6].clone()
        return [{"name": "layer", "shape": [6], "dtype": "torch.float32"}]

    async def release_outputs(self):
        self.outputs.clear()

    async def finish_conversion(self):
        pass

    async def close(self):
        self.slots.clear()
        self.buffers.clear()
        self.outputs.clear()

    async def store_chunk(self, slot, chunk, source):
        if source is None:
            value = self.outputs[chunk.name][chunk.offset : chunk.offset + chunk.numel].clone()
        else:
            value = await self.harness.receive(source, self.index + 1, ("output", chunk.key))
        self.slots[slot][chunk.key] = value

    async def send_output(self, chunk, destination):
        value = self.outputs[chunk.name][chunk.offset : chunk.offset + chunk.numel]
        self.harness.send(self.index + 1, destination, ("output", chunk.key), value)

    async def send_chunk(self, slot, chunk, destination):
        self.harness.send(self.index + 1, destination, ("stored", chunk.key), self.slots[slot][chunk.key])

    async def receive_chunk(self, key, slot, chunk, source):
        value = (
            self.slots[slot][chunk.key]
            if source is None
            else await self.harness.receive(source, self.index + 1, ("stored", chunk.key))
        )
        self.buffers[key][chunk.offset : chunk.offset + chunk.numel].copy_(value)

    async def install(self, publisher, snapshot, worker_index, tensors):
        assert publisher == "publisher-handle"
        for name, _shape, _dtype in tensors:
            key = f"install:{snapshot.version}:{name}"
            await self.harness.pool.fetch_tensor(snapshot, worker_index, name, key)
            self.installed[snapshot.version, name] = self.buffers.pop(key)


class Harness:
    def __init__(self):
        self.tasks = set()
        self.queues = defaultdict(asyncio.Queue)
        self.workers = [Worker(self, index) for index in range(4)]
        self.replicas = [
            SimpleNamespace(replica_rank=0, workers=self.workers[:2]),
            SimpleNamespace(replica_rank=1, workers=self.workers[2:]),
        ]
        self.config = SimpleNamespace(
            slots=2,
            chunk_bytes=8,
            max_bytes_per_gpu=1024,
            max_tensor_bytes=1024,
            max_conversion_bytes=1024,
            operation_timeout_s=0.2,
            reserve_timeout_s=0.02,
        )
        self.pool = GPUSnapshotPool(self.replicas, self.config)
        self.pool.publisher = "publisher-handle"

    def send(self, source, destination, tag, value):
        self.queues[source, destination, tag].put_nowait(value.clone())

    async def receive(self, source, destination, tag):
        return await self.queues[source, destination, tag].get()

    async def start(self):
        assert (await self.pool.describe())["world_size"] == 5
        await self.pool.initialize()

    async def export(self, version):
        await self.pool.begin(version)
        await self.input(version)
        return await self.pool.seal(version)

    async def input(self, version):
        destinations = await self.pool.prepare_input(version, "source", [12], "torch.float32")
        assert destinations == [1, 2]
        for destination in destinations:
            self.send(0, destination, "input", torch.arange(12, dtype=torch.float32) + version * 100)
        await self.pool.finish_input()

    async def cleanup(self):
        for worker in self.workers:
            for gate in worker.gates.values():
                gate.set()
        if self.pool.ingest_task is not None:
            await asyncio.gather(self.pool.ingest_task, return_exceptions=True)
        while self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_receive_convert_scatter_and_install_two_versions_without_tensor_rpcs():
    harness = Harness()
    try:
        await harness.start()
        for version in (1, 2):
            snapshot = await harness.export(version)
            for chunk in harness.pool.plan.chunks:
                assert chunk.owner % 2 == chunk.tp_rank
                torch.testing.assert_close(
                    harness.workers[chunk.owner].slots[snapshot.slot][chunk.key],
                    torch.arange(chunk.offset, chunk.offset + chunk.numel, dtype=torch.float32)
                    + chunk.tp_rank * 6
                    + version * 100,
                )
            await harness.pool.install(harness.replicas[1], snapshot)
            assert harness.pool.slots.versions[version]["readers"] == 0
            torch.testing.assert_close(
                harness.workers[2].installed[version, "layer"],
                torch.arange(100 * version, 100 * version + 6, dtype=torch.float32),
            )
            torch.testing.assert_close(
                harness.workers[3].installed[version, "layer"],
                torch.arange(100 * version + 6, 100 * version + 12, dtype=torch.float32),
            )
        assert harness.pool.slots.latest == 2
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
async def test_unknown_nonvision_source_is_rejected_before_allocation():
    harness = Harness()
    try:
        await harness.start()
        await harness.pool.begin(1)
        with pytest.raises(ValueError, match="unexpected.*soruce"):
            await harness.pool.prepare_input(1, "soruce", [12], "torch.float32")
        assert not any(worker.buffers for worker in harness.workers)
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    ["visual.projection", "model.visual.projection", "vision_tower.layer", "model.vision_model.layer", "alias"],
)
async def test_only_known_vision_and_explicit_nonowned_sources_are_skipped_once(name):
    harness = Harness()
    try:
        for worker in harness.workers:
            worker.ignored_source_names = ["alias"]
        await harness.start()
        await harness.pool.begin(1)
        assert await harness.pool.prepare_input(1, name, [12], "torch.float32") == []
        assert harness.pool.status()["seen_source_tensors"] == 0
        with pytest.raises(ValueError, match="duplicate"):
            await harness.pool.prepare_input(1, name, [12], "torch.float32")
        await harness.input(1)
        assert harness.pool.status()["seen_source_tensors"] == 1
        with pytest.raises(ValueError, match="duplicate"):
            await harness.pool.prepare_input(1, "source", [12], "torch.float32")
        await harness.pool.seal(1)
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
async def test_partial_allocation_failure_poison_prevents_retry_over_live_buffers():
    harness = Harness()
    try:
        await harness.start()
        await harness.pool.begin(1)
        harness.workers[1].failures["allocate"] = RuntimeError("allocation failed")
        with pytest.raises(RuntimeError, match="allocation failed"):
            await harness.pool.prepare_input(1, "source", [12], "torch.float32")
        assert "input" in harness.workers[0].buffers
        assert not harness.pool.io_lock.locked()
        with pytest.raises(RuntimeError, match="restart"):
            await harness.pool.prepare_input(1, "source", [12], "torch.float32")
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
async def test_prepare_cancellation_keeps_io_lock_until_remote_allocation_settles():
    harness = Harness()
    prepare = None
    try:
        await harness.start()
        await harness.pool.begin(1)
        harness.workers[0].gates["allocate"] = gate = asyncio.Event()
        prepare = asyncio.create_task(harness.pool.prepare_input(1, "source", [12], "torch.float32"))
        await harness.workers[0].entered["allocate"].wait()
        prepare.cancel()
        await asyncio.sleep(0)
        assert not prepare.done()
        assert harness.pool.io_lock.locked()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await prepare
        assert "input" in harness.workers[0].buffers
        with pytest.raises(RuntimeError, match="restart"):
            harness.pool.healthy()
    finally:
        await harness.cleanup()
        if prepare is not None:
            await asyncio.gather(prepare, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_single_rpc_retains_io_lock_until_remote_fetch_finishes():
    harness = Harness()
    fetching = None
    try:
        await harness.start()
        snapshot = await harness.export(1)
        harness.pool.slots.pin(1)
        worker = harness.workers[0]
        worker.gates["receive_chunk"] = gate = asyncio.Event()
        fetching = asyncio.create_task(harness.pool.fetch_tensor(snapshot, 0, "layer", "held"))
        await worker.entered["receive_chunk"].wait()
        fetching.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fetching), timeout=0.01)
        assert harness.pool.io_lock.locked()
        assert "held" in worker.buffers
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await fetching
        assert not harness.pool.io_lock.locked()
        with pytest.raises(RuntimeError, match="restart"):
            harness.pool.healthy()
    finally:
        await harness.cleanup()
        if fetching is not None:
            await asyncio.gather(fetching, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_install_waits_for_peer_and_poison_prevents_snapshot_reuse():
    harness = Harness()
    installation = None
    try:
        await harness.start()
        snapshot = await harness.export(1)
        harness.workers[2].failures["install"] = RuntimeError("unknown IPC completion")
        harness.workers[3].gates["install"] = gate = asyncio.Event()
        installation = asyncio.create_task(harness.pool.install(harness.replicas[1], snapshot))
        await harness.workers[3].entered["install"].wait()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(installation), timeout=0.01)
        assert harness.pool.slots.versions[1]["readers"] == 1
        gate.set()
        with pytest.raises(RuntimeError, match="unknown IPC completion"):
            await installation
        assert harness.pool.slots.versions[1]["readers"] == 0
        with pytest.raises(RuntimeError, match="restart"):
            await harness.pool.begin(2)
        assert harness.pool.slots.latest == 1
        assert any(worker.slots[snapshot.slot] for worker in harness.workers)
    finally:
        await harness.cleanup()
        if installation is not None:
            await asyncio.gather(installation, return_exceptions=True)


def cpu_storage_worker(monkeypatch, update_weights):
    worker = GPUSnapshotWorker.__new__(GPUSnapshotWorker)
    worker.config = SimpleNamespace(max_tensor_bytes=16, operation_timeout_s=0.05)
    worker.transport = SimpleNamespace(failed=None)
    worker.buffers = {"install:1:layer": torch.ones(4)}
    worker.buffer_released = asyncio.Event()
    worker.adapter = SimpleNamespace(update_weights=update_weights)
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("device", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    return worker


async def already_fetched(*args):
    assert_metadata(args)


@pytest.mark.asyncio
async def test_worker_allocation_waits_for_install_ack_before_using_shared_buffer_budget(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def update(weights, **kwargs):
        async for _name, _tensor in weights:
            entered.set()
            await release.wait()

    worker = cpu_storage_worker(monkeypatch, update)
    publisher = SimpleNamespace(gpu_fetch_tensor=SimpleNamespace(remote=already_fetched))
    installing = asyncio.create_task(
        worker.install(publisher, SimpleNamespace(version=1), 0, [("layer", (4,), "torch.float32")])
    )
    allocating = None
    try:
        await entered.wait()
        allocating = asyncio.create_task(worker.allocate("input", (4,), "torch.float32"))
        await asyncio.sleep(0)
        assert not allocating.done()
        assert sum(tensor.nbytes for tensor in worker.buffers.values()) == 16
        release.set()
        await installing
        await allocating
        assert set(worker.buffers) == {"input"}
    finally:
        release.set()
        await asyncio.gather(installing, return_exceptions=True)
        if allocating is not None:
            await asyncio.gather(allocating, return_exceptions=True)


@pytest.mark.asyncio
async def test_worker_buffer_budget_timeout_preserves_live_buffers(monkeypatch):
    worker = cpu_storage_worker(monkeypatch, None)
    with pytest.raises(TimeoutError, match="buffer"):
        await worker.allocate("input", (4,), "torch.float32")
    assert set(worker.buffers) == {"install:1:layer"}


@pytest.mark.asyncio
async def test_worker_install_failure_retains_ipc_source_and_rejects_reuse(monkeypatch):
    async def update(weights, **kwargs):
        async for _name, _tensor in weights:
            raise RuntimeError("IPC completion is unknown")

    worker = cpu_storage_worker(monkeypatch, update)
    publisher = SimpleNamespace(gpu_fetch_tensor=SimpleNamespace(remote=already_fetched))
    with pytest.raises(RuntimeError, match="IPC completion"):
        await worker.install(publisher, SimpleNamespace(version=1), 0, [("layer", (4,), "torch.float32")])
    assert worker.transport.failed
    assert "install:1:layer" in worker.buffers
    with pytest.raises(RuntimeError, match="restart"):
        await worker.allocate("input", (1,), "torch.float32")


@pytest.mark.asyncio
async def test_pinned_old_snapshot_blocks_eviction_until_install_completes():
    harness = Harness()
    installing = None
    try:
        await harness.start()
        first = await harness.export(1)
        second = await harness.export(2)
        for worker in harness.workers[2:]:
            worker.gates["install"] = asyncio.Event()
        installing = asyncio.create_task(harness.pool.install(harness.replicas[1], first))
        await harness.workers[2].entered["install"].wait()
        with pytest.raises(TimeoutError, match="slots are busy"):
            await harness.pool.begin(3)
        assert harness.pool.slots.latest == 2
        assert harness.pool.slots.versions[1]["readers"] == 1
        for worker in harness.workers[2:]:
            worker.gates["install"].set()
        await installing
        third = await harness.export(3)
        assert third.slot == first.slot
        assert third.slot != second.slot
        with pytest.raises(ValueError, match="stale"):
            harness.pool.validate(first)
        harness.pool.validate(second)
    finally:
        await harness.cleanup()
        if installing is not None:
            await asyncio.gather(installing, return_exceptions=True)


@pytest.mark.asyncio
async def test_fetch_rejects_unpinned_stale_and_wrong_tp_requests_before_allocation():
    harness = Harness()
    try:
        await harness.start()
        snapshot = await harness.export(1)
        with pytest.raises(ValueError, match="pinned"):
            await harness.pool.fetch_tensor(snapshot, 0, "layer", "invalid")
        harness.pool.slots.pin(1)
        for changed in (replace(snapshot, slot=99), replace(snapshot, layout_hash="wrong")):
            with pytest.raises(ValueError, match="stale"):
                await harness.pool.fetch_tensor(changed, 0, "layer", "invalid")
        with pytest.raises(ValueError, match="target"):
            await harness.pool.fetch_tensor(snapshot, 4, "layer", "invalid")
        with pytest.raises(ValueError, match="TP rank"):
            await harness.pool.fetch_tensor(snapshot, 0, "not-a-runtime-key", "invalid")
        assert all("invalid" not in worker.buffers for worker in harness.workers)
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
async def test_conversion_failure_keeps_staging_storage_and_blocks_commit():
    harness = Harness()
    try:
        await harness.start()
        await harness.pool.begin(1)
        harness.workers[1].failures["convert"] = RuntimeError("conversion failed")
        with pytest.raises(RuntimeError, match="conversion failed"):
            await harness.input(1)
        assert harness.pool.slots.latest is None
        assert harness.pool.slots.versions[1]["ready"] is False
        assert harness.pool.staging.version == 1
        assert any(worker.slots[0] for worker in harness.workers)
        assert not harness.pool.io_lock.locked()
        with pytest.raises(RuntimeError, match="restart"):
            await harness.pool.seal(1)
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
async def test_pool_status_reports_source_and_runtime_completion_without_payloads():
    harness = Harness()
    try:
        await harness.start()
        await harness.pool.begin(1)
        pending = harness.pool.status()
        assert pending["staging_version"] == 1
        assert pending["required_source_tensors"] == 1
        assert pending["seen_source_tensors"] == 0
        assert pending["required_runtime_tensors"] == 2
        assert pending["completed_runtime_tensors"] == 0
        await harness.input(1)
        staged = harness.pool.status()
        assert staged["seen_source_tensors"] == 1
        assert staged["completed_runtime_tensors"] == 2
        await harness.pool.seal(1)
        completed = harness.pool.status()
        assert completed["latest_version"] == 1
        assert completed["staging_version"] is None
        assert_metadata(completed)
        completed["slots"][1]["readers"] = 99
        assert harness.pool.slots.versions[1]["readers"] == 0
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_native_failure_retains_source_owner_until_worker_restart(monkeypatch, cancel):
    transport = DirectedNCCL.__new__(DirectedNCCL)
    transport.timeout_s = 0.03
    transport.device = 0
    transport.failed = None
    transport.retained = []
    transport.tasks = set()
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    started, release = Event(), Event()
    owner = torch.ones(4)

    def blocked_native(value):
        started.set()
        release.wait(timeout=2)
        return value.sum().item()

    running = asyncio.create_task(transport._native(blocked_native, owner))
    try:
        await asyncio.to_thread(started.wait, 1)
        if cancel:
            running.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await running
        assert transport.failed
        assert transport.retained[0][0] is owner
        assert any(not task.done() for task in transport.tasks)
        with pytest.raises(RuntimeError, match="restart"):
            await transport._native(lambda: None)
    finally:
        release.set()
        await asyncio.gather(running, *transport.tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_pool_shutdown_quiesces_install_before_closing_and_rejects_new_export():
    harness = Harness()
    installing = stopping = None
    try:
        await harness.start()
        snapshot = await harness.export(1)
        harness.workers[2].gates["install"] = gate = asyncio.Event()
        installing = asyncio.create_task(harness.pool.install(harness.replicas[1], snapshot))
        await harness.workers[2].entered["install"].wait()
        stopping = asyncio.create_task(harness.pool.quiesce())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="closing|closed"):
            await asyncio.wait_for(harness.pool.begin(2), timeout=0.2)
        assert not stopping.done()
        assert all(not worker.entered["close"].is_set() for worker in harness.workers)
        gate.set()
        await installing
        await stopping
        assert harness.pool.slots.versions[1]["readers"] == 0
        await harness.pool.close()
        assert all(not worker.slots and not worker.buffers for worker in harness.workers)
        assert harness.pool.status()["closed"] is True
        await harness.pool.quiesce()
        await harness.pool.close()
    finally:
        await harness.cleanup()
        await asyncio.gather(*(task for task in (installing, stopping) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_quiesce_aborts_settled_partial_conversion_without_comm_close():
    harness = Harness()
    try:
        await harness.start()
        await harness.pool.begin(1)
        await harness.input(1)
        await harness.pool.quiesce()
        assert harness.pool.staging is None
        assert harness.pool.slots.latest is None
        assert harness.pool.slots.versions == {}
        assert all(worker.entered["finish_conversion"].is_set() for worker in harness.workers[:2])
        assert all(not worker.entered["close"].is_set() for worker in harness.workers)
        await harness.pool.close()
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
async def test_failed_pool_shutdown_preserves_workers_and_snapshot_storage():
    harness = Harness()
    try:
        await harness.start()
        snapshot = await harness.export(1)
        harness.pool.fatal_error = "unknown native completion"
        for method in (harness.pool.quiesce, harness.pool.close):
            with pytest.raises(RuntimeError, match="restart"):
                await method()
        assert any(worker.slots[snapshot.slot] for worker in harness.workers)
        assert all(not worker.entered["close"].is_set() for worker in harness.workers)
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
async def test_directed_transport_close_destroys_only_healthy_comm(monkeypatch):
    transport = DirectedNCCL.__new__(DirectedNCCL)
    transport.timeout_s = 0.1
    transport.device = 0
    transport.failed = None
    transport.retained = []
    transport.tasks = set()
    destroyed = []
    communicator = SimpleNamespace(destroy=lambda: destroyed.append(True))
    transport.comm = communicator
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    with pytest.raises(RuntimeError, match="initialized"):
        await transport.initialize(2, (), 0)
    await transport.close()
    assert destroyed == [True]
    assert transport.comm is None
    await transport.close()
    transport.comm = communicator
    transport.failed = "unknown native completion"
    with pytest.raises(RuntimeError, match="restart"):
        await transport.close()
    assert transport.comm is communicator
    assert destroyed == [True]


@pytest.mark.asyncio
async def test_worker_close_returns_cached_memory_only_after_healthy_transport_close(monkeypatch):
    worker = GPUSnapshotWorker.__new__(GPUSnapshotWorker)
    worker.slots = {0: {"chunk": torch.ones(4)}}
    worker.buffers = {"input": torch.ones(4)}
    worker.outputs = {"output": torch.ones(4)}
    worker.buffer_released = asyncio.Event()
    events = []

    async def close():
        events.append("transport-close")
        if worker.transport.failed:
            raise RuntimeError("restart required")

    def empty_cache():
        assert not worker.slots and not worker.buffers and not worker.outputs
        events.append("empty-cache")

    worker.transport = SimpleNamespace(close=close, device=3, failed="unknown native completion")
    monkeypatch.setattr(torch.cuda, "device", lambda device: events.append(("device", device)) or nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    with pytest.raises(RuntimeError, match="restart"):
        await worker.close()
    assert worker.slots and worker.buffers and worker.outputs
    assert events == ["transport-close"]
    worker.transport.failed = None
    await worker.close()
    assert events == ["transport-close", "transport-close", ("device", 3), "empty-cache"]


@pytest.mark.asyncio
async def test_quiesce_waits_for_install_only_not_its_callers_remaining_work():
    harness = Harness()
    parent = stopping = None
    release_parent = asyncio.Event()
    try:
        await harness.start()
        snapshot = await harness.export(1)
        harness.workers[2].gates["install"] = gate = asyncio.Event()

        async def caller():
            await harness.pool.install(harness.replicas[1], snapshot)
            await release_parent.wait()

        parent = asyncio.create_task(caller())
        await harness.workers[2].entered["install"].wait()
        stopping = asyncio.create_task(harness.pool.quiesce())
        await asyncio.sleep(0)
        gate.set()
        await asyncio.wait_for(asyncio.shield(stopping), 0.2)
        assert not parent.done()
    finally:
        release_parent.set()
        await harness.cleanup()
        await asyncio.gather(*(task for task in (parent, stopping) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_partial_initialization_poison_prevents_collective_close_and_reuse():
    harness = Harness()
    try:
        await harness.pool.describe()
        harness.workers[1].failures["initialize"] = RuntimeError("communicator initialization failed")
        with pytest.raises(RuntimeError, match="initialization failed"):
            await harness.pool.initialize()
        assert harness.pool.status()["initialized"] is False
        for operation in (harness.pool.initialize, harness.pool.close):
            with pytest.raises(RuntimeError, match="restart"):
                await operation()
        assert all(not worker.entered["close"].is_set() for worker in harness.workers)
    finally:
        await harness.cleanup()


@pytest.mark.asyncio
async def test_stage_timings_reset_each_export_and_install_keeps_its_version():
    harness = Harness()
    try:
        await harness.start()
        snapshot = await harness.export(1)
        await harness.pool.install(harness.replicas[1], snapshot)
        harness.pool.metrics["source_transfer_s"] = 1000
        harness.pool.metrics["conversion_s"] = 1000
        harness.pool.metrics["shard_distribution_s"] = 1000
        await harness.export(2)
        metrics = harness.pool.status()["metrics"]
        assert metrics["export_version"] == 2
        assert metrics["last_install_version"] == 1
        assert metrics["last_install_replica_rank"] == 1
        assert all(metrics[name] < 1000 for name in ("source_transfer_s", "conversion_s", "shard_distribution_s"))
        assert harness.pool.status()["initialized"] is True
        await harness.pool.close()
        assert harness.pool.status()["initialized"] is False
    finally:
        await harness.cleanup()
