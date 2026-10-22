import asyncio
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine.gpu_snapshot import GPUWeightSnapshot, SnapshotSlots
from verl.checkpoint_engine.weight_publication import SnapshotCheckpointEngineManager


@pytest.mark.asyncio
async def test_gpu_version_is_pinned_during_drain_and_only_one_replica_installs():
    manager = SnapshotCheckpointEngineManager.__new__(SnapshotCheckpointEngineManager)
    manager._weight_update_lock = asyncio.Lock()
    manager.relay = None
    slots = SnapshotSlots(2)
    first = GPUWeightSnapshot(1, slots.reserve(1), "layout")
    slots.commit(1)
    manager.gpu_pool = SimpleNamespace(slots=slots, validate=lambda snapshot: None)
    manager._replica = lambda rank: rank
    entered, release = asyncio.Event(), asyncio.Event()
    active = []

    async def update(replicas, version, recovery=False):
        assert slots.versions[version]["readers"] == 1
        assert not active
        active.extend(replicas)
        entered.set()
        await release.wait()
        active.clear()
        return {"version": version}

    manager._update_selected_replicas = update
    task = asyncio.create_task(manager.update_replica_from_snapshot(first, 0))
    await entered.wait()
    second = GPUWeightSnapshot(2, slots.reserve(2), "layout")
    slots.commit(2)
    with pytest.raises(BufferError):
        slots.reserve(3)
    next_task = asyncio.create_task(manager.update_replica_from_snapshot(second, 1))
    await asyncio.sleep(0)
    assert slots.versions[2]["readers"] == 0
    assert manager._snapshot == first
    release.set()
    assert await task == {"version": 1}
    assert await next_task == {"version": 2}
    assert slots.versions[1]["readers"] == slots.versions[2]["readers"] == 0
    assert slots.reserve(3) == first.slot


@pytest.mark.asyncio
async def test_gpu_transfer_uses_pool_without_cpu_snapshot_replay():
    manager = SnapshotCheckpointEngineManager.__new__(SnapshotCheckpointEngineManager)
    snapshot = GPUWeightSnapshot(4, 0, "layout")
    manager._snapshot = snapshot
    calls = []

    async def install(replica, value):
        calls.append((replica, value))

    manager.gpu_pool = SimpleNamespace(install=install)
    assert await manager._transfer_replica_weights("replica2", 4) == {}
    assert calls == [("replica2", snapshot)]


@pytest.mark.asyncio
async def test_checkpoint_worker_exposes_healthy_gpu_close_rpc():
    from verl.checkpoint_engine.base import CheckpointEngineWorker

    closed = []

    async def close():
        closed.append(True)

    worker = SimpleNamespace(
        rollout_config=SimpleNamespace(weight_publication=SimpleNamespace(transport="nccl_gpu")),
        _gpu_snapshot_worker=SimpleNamespace(close=close),
    )
    await CheckpointEngineWorker.gpu_snapshot_rpc.__wrapped__(worker, "close")
    assert closed == [True]
