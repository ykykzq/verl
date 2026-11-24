import asyncio
import weakref
from types import SimpleNamespace

import pytest
import torch

from tests.checkpoint_engine.test_gpu_snapshot_pool_on_cpu import Harness, Worker
from verl.checkpoint_engine.gpu_snapshot_bucket import BucketSnapshotPool, byte_layout, tensor_buckets, views


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_format", [None, "named_tensors"], ids=["rtp", "vllm"])
async def test_install_releases_previous_bucket_before_next_fetch(wire_format):
    from verl.checkpoint_engine.gpu_snapshot_bucket import BucketSnapshotWorker

    worker = BucketSnapshotWorker.__new__(BucketSnapshotWorker)
    worker.config = SimpleNamespace(bucket_bytes=16)
    worker.transport = SimpleNamespace(failed=None)
    worker.buffers = {}
    worker.buffer_released = asyncio.Event()
    previous = None

    async def fetch(snapshot, worker_index, group, key):
        nonlocal previous
        assert previous is None or previous() is None, "previous bucket still retained"
        entries, size = byte_layout(group)
        worker.buffers[key] = torch.empty(size, dtype=torch.uint8)

    async def update(iterator, **kwargs):
        nonlocal previous
        assert kwargs["wire_format"] == (wire_format or "runtime_tensors")
        # Match RayIpcWeightSender: clear consumer references before anext.
        while True:
            try:
                item = await anext(iterator)
            except StopAsyncIteration:
                break
            previous = weakref.ref(item[1])
            del item

    worker.adapter = SimpleNamespace(update_weights=update)
    if wire_format:
        worker.adapter.snapshot_wire_format = wire_format
    publisher = SimpleNamespace(gpu_fetch_bucket=SimpleNamespace(remote=fetch))
    await worker.install(publisher, SimpleNamespace(version=1), 0, [(str(i), (4,), "torch.float32") for i in range(3)])
    assert previous() is None and not worker.buffers


@pytest.mark.asyncio
async def test_source_failure_still_drains_all_fsdp_collectives(monkeypatch):
    from verl.checkpoint_engine.gpu_snapshot_bucket import export_buckets

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    consumed, errors = [], []

    def weights():
        for i in range(5):
            consumed.append(i)
            yield str(i), torch.ones(2)

    async def failed(message):
        errors.append(message)

    publisher = SimpleNamespace(fail_gpu_snapshot=SimpleNamespace(remote=failed))
    with pytest.raises(ValueError, match="source tensor"):
        await export_buckets(
            None, 1, publisher, weights(), torch.float32, SimpleNamespace(max_tensor_bytes=100, bucket_bytes=16)
        )
    assert consumed == list(range(5)) and len(errors) == 1


class BucketWorker(Worker):
    async def initialize_fetch(self, size, uid, rank):
        assert size == 2 and rank == self.index // 2

    async def convert_bucket(self, entries, is_last=False):
        return await self.convert(entries[0][0] if entries else None, is_last)

    async def store_chunks(self, slot, chunks, source):
        for chunk in chunks:
            await self.store_chunk(slot, chunk, source)

    async def send_outputs(self, chunks, destination):
        for chunk in chunks:
            await self.send_output(chunk, destination)

    async def send_chunks(self, slot, chunks, destination):
        for chunk in chunks:
            await self.send_chunk(slot, chunk, destination * 2 + self.index % 2 + 1)

    async def receive_chunks(self, key, slot, chunks, entries, source):
        tensor_views = dict(views(self.buffers[key], entries))
        for chunk in chunks:
            value = (
                self.slots[slot][chunk.key]
                if source is None
                else await self.harness.receive(source * 2 + self.index % 2 + 1, self.index + 1, ("stored", chunk.key))
            )
            tensor_views[chunk.name].reshape(-1)[chunk.offset : chunk.offset + chunk.numel].copy_(value)


def make_harness():
    h = Harness()
    h.workers = [BucketWorker(h, i) for i in range(4)]
    h.replicas = [SimpleNamespace(replica_rank=i, workers=h.workers[i * 2 : i * 2 + 2]) for i in range(2)]
    h.pool = BucketSnapshotPool(h.replicas, h.pool.config)
    h.pool.publisher = "publisher-handle"
    return h


def test_mixed_dtype_bucket_roundtrip_and_oversize():
    tensors = [("a", (3,), "torch.int64"), ("b", (7,), "torch.bfloat16"), ("c", (23,), "torch.uint8")]
    entries, size = byte_layout(tensors)
    buffer = torch.empty(size, dtype=torch.uint8)
    values = dict(views(buffer, entries))
    for index, (_, tensor) in enumerate(values.items()):
        tensor.fill_(index + 1)
    assert all(torch.all(t == i + 1) for i, t in enumerate(values.values()))
    assert [len(b) for b in tensor_buckets(tensors, 16)] == [1, 1, 1]
    assert all(entry[3] % 16 == 0 for entry in entries)
    with pytest.raises(ValueError, match="duplicate"):
        byte_layout([tensors[0], tensors[0]])


@pytest.mark.asyncio
async def test_fetch_does_not_wait_on_export_lock_and_columns_can_overlap():
    h = make_harness()
    tasks = []
    try:
        await h.start()
        # Existing per-tensor export also exercises the bucket converter final flush.
        snap = await h.export(1)
        h.pool.slots.pin(1)
        await h.pool.io_lock.acquire()
        for worker in h.workers[2:]:
            worker.gates["receive_chunks"] = asyncio.Event()
        tensors = [("layer", (6,), "torch.float32")]
        tasks = [asyncio.create_task(h.pool.fetch_bucket(snap, i, tensors, "fetch")) for i in (2, 3)]
        await asyncio.wait_for(asyncio.gather(*(w.entered["receive_chunks"].wait() for w in h.workers[2:])), 1)
        assert all(lock.locked() for lock in h.pool.fetch_locks)
        for w in h.workers[2:]:
            w.gates["receive_chunks"].set()
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
        assert h.pool.io_lock.locked()
        entries, _ = byte_layout(tensors)
        for i in (2, 3):
            got = dict(views(h.workers[i].buffers["fetch"], entries))["layer"]
            expected = torch.arange(6, dtype=torch.float32) + (i % 2) * 6 + 100
            assert torch.equal(got, expected)
        h.pool.slots.unpin(1)
    finally:
        if h.pool.io_lock.locked():
            h.pool.io_lock.release()
        await h.cleanup()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_fetch_cancel_waits_for_remote_and_poison_preserves_buffer():
    h = make_harness()
    task = None
    try:
        await h.start()
        snap = await h.export(1)
        h.pool.slots.pin(1)
        h.workers[2].gates["receive_chunks"] = gate = asyncio.Event()
        task = asyncio.create_task(h.pool.fetch_bucket(snap, 2, [("layer", (6,), "torch.float32")], "held"))
        await h.workers[2].entered["receive_chunks"].wait()
        task.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.01)
        assert h.pool.fetch_locks[0].locked()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "held" in h.workers[2].buffers
        assert h.pool.fatal_error
    finally:
        await h.cleanup()
        if task:
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_invalid_fetch_rejected_before_allocation():
    h = make_harness()
    try:
        await h.start()
        snap = await h.export(1)
        with pytest.raises(ValueError, match="pinned"):
            await h.pool.fetch_bucket(snap, 2, [("layer", (6,), "torch.float32")], "bad")
        h.pool.slots.pin(1)
        with pytest.raises(ValueError, match="TP rank"):
            await h.pool.fetch_bucket(snap, 2, [("layer", (7,), "torch.float32")], "bad")
        assert "bad" not in h.workers[2].buffers
    finally:
        await h.cleanup()
