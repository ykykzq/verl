import asyncio
import sys
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from tests.checkpoint_engine import test_temporary_nccl_group_on_cpu
from tests.checkpoint_engine.test_gpu_snapshot_bucket_on_cpu import BucketWorker
from tests.checkpoint_engine.test_gpu_snapshot_pool_on_cpu import Harness
from verl.checkpoint_engine.gpu_snapshot_bucket import byte_layout, views
from verl.checkpoint_engine.gpu_snapshot_pipeline import PipelineSnapshotPool, export_pipeline, select_converters

nccl_engine = test_temporary_nccl_group_on_cpu.nccl


@pytest.fixture(autouse=True)
def cpu_topology(nccl_engine, monkeypatch):
    # Reuse the production topology builder with only the CUDA imports stubbed.
    # Do not require a CuPy wheel for these metadata/CPU-storage tests.
    module = sys.modules[type(nccl_engine.engine).__module__]
    monkeypatch.setitem(sys.modules, "verl.checkpoint_engine.nccl_checkpoint_engine", module)


class PipelineWorker(BucketWorker):
    async def node_id(self):
        return "a" if self.index < 2 else "b"

    async def describe(self):
        return {
            "layout": [dict(name=f"w{i}", shape=[6], dtype="torch.float32") for i in range(4)],
            "required_names": [f"w{i}" for i in range(4)],
        }

    async def initialize(self, size, uid, rank):
        assert size == 4 and rank == self.index

    async def initialize_input(self, size, uid, rank):
        assert size == 4 and rank == self.index

    async def receive_pipeline_input(self, key):
        self.buffers[key].copy_(await self.harness.receive(-1, self.index, key))

    async def convert_pipeline_bucket(self, key, entries, output_key, last):
        self.output_batches = getattr(self, "output_batches", {})
        assert output_key not in self.output_batches and len(self.output_batches) < 2
        output = {} if last else {n: t.chunk(2)[self.index % 2].clone() for n, t in views(self.buffers[key], entries)}
        self.output_batches[output_key] = output
        if key is not None:
            self.buffers.pop(key)
        return [dict(name=n, shape=list(t.shape), dtype=str(t.dtype)) for n, t in output.items()]

    async def send_pipeline_outputs(self, chunks, destination, key):
        for c in chunks:
            value = self.output_batches[key][c.name][c.offset : c.offset + c.numel]
            self.harness.send(self.index, destination, ("pipeline", c.key), value)

    async def store_pipeline_chunks(self, slot, chunks, source, key):
        for c in chunks:
            value = (
                self.output_batches[key][c.name][c.offset : c.offset + c.numel].clone()
                if source is None
                else await self.harness.receive(source, self.index, ("pipeline", c.key))
            )
            self.slots[slot][c.key] = value

    async def release_pipeline_outputs(self, key):
        self.output_batches.pop(key)


async def make_harness():
    h = Harness()
    h.config.bucket_bytes = 64
    h.workers = [PipelineWorker(h, i) for i in range(4)]
    h.replicas = [SimpleNamespace(replica_rank=i, workers=h.workers[i * 2 : i * 2 + 2]) for i in range(2)]
    h.pool = PipelineSnapshotPool(h.replicas, h.config)
    h.pool.publisher = "publisher-handle"
    description = await h.pool.describe(["b", "b", "c"])
    assert description["actor_ranks"] == [0, 1, -1]
    assert h.pool.converter_workers == [2, 3]
    await h.pool.initialize()
    await h.pool.begin(1)
    return h


async def feed(h, i):
    tensors = [(f"w{i}", (12,), "torch.float32")]
    result = await h.pool.prepare_bucket(1, tensors)
    entries, size = byte_layout(tensors)
    buffer = torch.empty(size, dtype=torch.uint8)
    dict(views(buffer, entries))[f"w{i}"].copy_(torch.arange(12) + i * 100)
    for worker in h.pool.converter_workers:
        h.send(-1, worker, f"pipeline-input:{result['bucket'] % 2}", buffer)
    await h.pool.finish_input()


def test_converter_selection_prefers_trainer_node_not_replica_zero():
    assert select_converters(["b"] * 4, ["a"] * 4 + ["b"] * 4 + ["a"] * 4, 4) == [4, 5, 6, 7]
    with pytest.raises(ValueError):
        select_converters([], ["a"], 1)


def test_pipeline_config_requires_two_bounded_input_slots():
    from verl.workers.config.rollout import GPUSnapshotConfig

    assert GPUSnapshotConfig().pipeline_depth == 0
    for kwargs in [
        dict(pipeline_depth=1),
        dict(pipeline_depth=True),
        dict(pipeline_depth=2),
        dict(pipeline_depth=2, bucket_bytes=3 << 30),
    ]:
        with pytest.raises(ValueError):
            GPUSnapshotConfig(**kwargs)
    assert GPUSnapshotConfig(pipeline_depth=2, bucket_bytes=512 << 20).pipeline_depth == 2


@pytest.mark.asyncio
async def test_server_keeps_two_distinct_output_owners_until_each_ack():
    from unittest.mock import patch

    from tests.workers.rollout.rollout_rtp_llm.test_gpu_snapshot_server_on_cpu import MODULE, server

    instance = server()
    with patch(f"{MODULE}.set_expandable_segments", create=True):
        await instance.begin_gpu_snapshot(1, 4096)
    with (
        patch(f"{MODULE}.rebuild_ipc_tensor", return_value=SimpleNamespace(device=torch.device("cuda:0"))),
        patch(f"{MODULE}.serialize_ipc_tensor", return_value=b"ipc"),
        patch(f"{MODULE}.get_device_id", return_value=0),
    ):
        await instance.convert_gpu_snapshot("source", b"input", output_key="a")
        first = instance._gpu_snapshot["output_batches"]["a"]
        await instance.convert_gpu_snapshot("source", b"input", output_key="b")
        assert instance._gpu_snapshot["output_batches"]["a"] is first
        for key in ["a", "c"]:
            with pytest.raises(RuntimeError, match="slots"):
                await instance.convert_gpu_snapshot("source", b"input", output_key=key)
        with pytest.raises(RuntimeError, match="release"):
            await instance.close_gpu_snapshot()
        await instance.release_gpu_snapshot_outputs("a")
        assert set(instance._gpu_snapshot["output_batches"]) == {"b"}
        await instance.convert_gpu_snapshot("source", b"input", output_key="a")
    await instance.release_gpu_snapshot_outputs("a")
    await instance.release_gpu_snapshot_outputs("b")
    await instance.close_gpu_snapshot()


@pytest.mark.asyncio
async def test_pipeline_bounded_overlap_and_byte_exact_storage():
    h = await make_harness()
    tasks = []
    try:
        convert_gate, distribution_gate = asyncio.Event(), asyncio.Event()
        for w in h.workers[2:]:
            w.gates["convert_pipeline_bucket"] = convert_gate
        for w in h.workers:
            w.gates["store_pipeline_chunks"] = distribution_gate
        await feed(h, 0)
        await asyncio.wait_for(h.workers[2].entered["convert_pipeline_bucket"].wait(), 1)
        # Receiving bucket 1 must complete while bucket 0 conversion is blocked.
        await asyncio.wait_for(feed(h, 1), 1)
        assert h.pool.input_live == 2
        third = asyncio.create_task(feed(h, 2))
        tasks.append(third)
        await asyncio.sleep(0.01)
        assert not third.done(), "third input must respect two-slot backpressure"
        convert_gate.set()
        await asyncio.wait_for(third, 1)
        await asyncio.wait_for(h.workers[0].entered["store_pipeline_chunks"].wait(), 1)
        for _ in range(100):
            if h.pool.output_live == 2:
                break
            await asyncio.sleep(0.001)
        assert h.pool.output_live == 2
        assert all(w.output_batches for w in h.workers[2:]), "DMA owners released too soon"
        fourth = asyncio.create_task(feed(h, 3))
        tasks.append(fourth)
        await asyncio.sleep(0.01)
        distribution_gate.set()
        await asyncio.wait_for(fourth, 1)
        snap = await asyncio.wait_for(h.pool.seal(1), 1)
        assert h.pool.input_live == h.pool.output_live == 0
        assert h.pool.metrics["input_high_water"] == h.pool.metrics["output_high_water"] == 2
        for c in h.pool.plan.chunks:
            value = h.workers[c.owner].slots[snap.slot][c.key]
            expected = torch.arange(c.offset, c.offset + c.numel) + c.tp_rank * 6 + int(c.name[1:]) * 100
            torch.testing.assert_close(value, expected.float())
        assert all(not w.output_batches for w in h.workers[2:])
        await h.pool.close()
    finally:
        await h.cleanup()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_drain_waits_for_distribution_before_returning():
    h = await make_harness()
    task = None
    try:
        gate = asyncio.Event()
        for w in h.workers:
            w.gates["store_pipeline_chunks"] = gate
        await feed(h, 0)
        await asyncio.wait_for(h.workers[0].entered["store_pipeline_chunks"].wait(), 1)
        task = asyncio.create_task(h.pool.drain())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert all(w.output_batches for w in h.workers[2:])
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await h.cleanup()
        await asyncio.gather(*list(h.pool.jobs), return_exceptions=True)


@pytest.mark.asyncio
async def test_pipeline_source_failure_drains_fsdp_generator(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    consumed = []

    async def failed(_):
        pass

    def weights():
        for i in range(5):
            consumed.append(i)
            yield str(i), torch.ones(2)

    publisher = SimpleNamespace(fail_gpu_snapshot=SimpleNamespace(remote=failed))
    with pytest.raises(ValueError, match="source tensor"):
        await export_pipeline(
            None, 1, publisher, weights(), torch.float32, SimpleNamespace(max_tensor_bytes=100, bucket_bytes=16)
        )
    assert consumed == list(range(5))


@pytest.mark.asyncio
async def test_quiesce_closes_admission_before_waiting_for_pipeline():
    h = await make_harness()
    stopping = None
    try:
        gate = asyncio.Event()
        for w in h.workers[2:]:
            w.gates["convert_pipeline_bucket"] = gate
        await feed(h, 0)
        await h.workers[2].entered["convert_pipeline_bucket"].wait()
        stopping = asyncio.create_task(h.pool.quiesce())
        await asyncio.sleep(0)
        assert h.pool.closing, "shutdown must fence new exports before waiting for CUDA owners"
        with pytest.raises(RuntimeError, match="closing"):
            await h.pool.prepare_bucket(1, [("w1", (12,), "torch.float32")])
        assert not stopping.done()
        assert all(not w.entered["finish_conversion"].is_set() for w in h.workers)
        gate.set()
        await asyncio.wait_for(stopping, 1)
        assert h.pool.quiesced and h.pool.staging is None
        assert h.pool.input_live == h.pool.output_live == 0
        await h.pool.close()
    finally:
        await h.cleanup()
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_input_allocation_poison_retains_partial_owners():
    h = await make_harness()
    try:
        h.workers[3].failures["allocate"] = RuntimeError("allocation failed")
        with pytest.raises(RuntimeError, match="allocation failed"):
            await h.pool.prepare_bucket(1, [("w0", (12,), "torch.float32")])
        assert h.workers[2].buffers, "successful peers must retain partially allocated input storage"
        assert h.pool.fatal_error, "a partial allocation cannot be retried in place"
        with pytest.raises(RuntimeError, match="restart"):
            await h.pool.close()
        assert all(not w.entered["close"].is_set() for w in h.workers)
    finally:
        await h.cleanup()


@pytest.mark.asyncio
async def test_quiesce_waits_for_admitted_allocation_to_join_pipeline():
    h = await make_harness()
    feeding = stopping = None
    try:
        h.workers[2].gates["allocate"] = allocation_gate = asyncio.Event()
        for w in h.workers[2:]:
            w.gates["convert_pipeline_bucket"] = asyncio.Event()
        feeding = asyncio.create_task(feed(h, 0))
        await asyncio.wait_for(h.workers[2].entered["allocate"].wait(), 1)
        stopping = asyncio.create_task(h.pool.quiesce())
        await asyncio.sleep(0)
        assert h.pool.closing and not stopping.done()
        assert not h.pool.jobs, "allocation has not handed ownership to stage tasks yet"
        allocation_gate.set()
        await asyncio.wait_for(feeding, 1)
        assert not stopping.done()
        assert all(not w.entered["finish_conversion"].is_set() for w in h.workers)
        for w in h.workers[2:]:
            w.gates["convert_pipeline_bucket"].set()
        await asyncio.wait_for(stopping, 1)
        assert h.pool.quiesced and not h.pool.jobs
        await h.pool.close()
    finally:
        await h.cleanup()
        await asyncio.gather(*(t for t in (feeding, stopping) if t is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_distribution_failure_preserves_outputs_and_blocks_seal_and_close():
    h = await make_harness()
    try:
        h.workers[0].failures["store_pipeline_chunks"] = RuntimeError("distribution failed")
        await feed(h, 0)
        with pytest.raises(RuntimeError, match="distribution failed"):
            await h.pool.drain()
        assert h.pool.fatal_error
        assert h.workers[2].output_batches, "failed DMA must keep its source owners"
        for operation in (h.pool.seal(1), h.pool.close()):
            with pytest.raises(RuntimeError, match="restart|distribution failed"):
                await operation
        assert all(not w.entered["close"].is_set() for w in h.workers)
    finally:
        await h.cleanup()
        await asyncio.gather(*list(h.pool.jobs), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["broadcast", "transfer_many"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_nccl_pipeline_waits_for_device_completion_and_retains_on_cancel(monkeypatch, operation, cancel):
    from verl.checkpoint_engine.gpu_snapshot_transport import DirectedNCCL

    enqueued, completed = Event(), Event()
    events, calls = [], []
    producer_stream = object()

    class CUDAEvent:
        def __init__(self):
            events.append(self)

        def record(self, stream):
            self.stream = stream
            if len(events) == 2:
                enqueued.set()

        def query(self):
            return completed.is_set()

    transport = DirectedNCCL.__new__(DirectedNCCL)
    transport.timeout_s = 5
    transport.device = 0
    transport.failed = None
    transport.retained = []
    transport.tasks = set()
    transport.stream = SimpleNamespace(cuda_stream=123, wait_event=lambda event: calls.append(("wait", event)))
    transport.comm = SimpleNamespace(
        broadcast=lambda *args: calls.append(("broadcast", args)),
        send=lambda *args: calls.append(("send", args)),
        recv=lambda *args: calls.append(("recv", args)),
        check_async_error=lambda: None,
    )
    nccl = SimpleNamespace(
        NCCL_UINT8=1,
        groupStart=lambda: calls.append(("start",)),
        groupEnd=lambda: calls.append(("end",)),
    )
    monkeypatch.setitem(sys.modules, "cupy.cuda", SimpleNamespace(nccl=nccl))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: producer_stream)
    monkeypatch.setattr(torch.cuda, "Event", CUDAEvent)
    owners = [
        SimpleNamespace(device=torch.device("cuda"), is_contiguous=lambda: True, data_ptr=lambda: 42, nbytes=32)
        for _ in range(2)
    ]
    payload = owners[0] if operation == "broadcast" else [(owners[0], 1, False), (owners[1], 2, True)]
    running = asyncio.create_task(getattr(transport, operation)(payload))
    try:
        assert await asyncio.to_thread(enqueued.wait, 1)
        assert events[0].stream is producer_stream
        assert events[1].stream is transport.stream
        assert calls[0] == ("wait", events[0])
        assert not running.done(), "NCCL enqueue cannot acknowledge buffer reuse"
        if cancel:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert transport.failed and transport.retained[0][0] is payload
            assert any(not task.done() for task in transport.tasks)
            with pytest.raises(RuntimeError, match="restart"):
                await transport.close()
        else:
            completed.set()
            await asyncio.wait_for(running, 1)
            assert not transport.retained and not transport.tasks and not transport.failed
    finally:
        completed.set()
        await asyncio.gather(running, *transport.tasks, return_exceptions=True)
