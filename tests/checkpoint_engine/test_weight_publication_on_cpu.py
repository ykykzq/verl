# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import asyncio
from types import SimpleNamespace

import pytest
import torch

from tests.checkpoint_engine.test_fine_grained_updates_on_cpu import (
    RouterHandle,
    WorkerGroup,
    environment,  # noqa: F401
)
from verl.checkpoint_engine import base


def snapshot_manager(env, monkeypatch, max_retries=1):
    from verl.checkpoint_engine.weight_publication import SnapshotCheckpointEngineManager

    class SnapshotWorkers(WorkerGroup):
        def update_weights_from_snapshot(self, snapshot):
            env.snapshots.append(snapshot)
            return self.update_weights(global_steps=snapshot.version)

    env.snapshots = []
    monkeypatch.setattr(
        base,
        "RayWorkerGroup",
        lambda worker_handles, ray_cls_with_init: SnapshotWorkers(env, worker_handles[0].replica),
    )
    original = env.manager(max_retries=max_retries)
    return SnapshotCheckpointEngineManager(
        checkpoint_config=original.config,
        fine_grained_config=original.fine_grained_config,
        replicas=env.replicas,
        load_balancer_handle=RouterHandle(env.router),
    )


@pytest.mark.asyncio
async def test_snapshot_update_never_calls_training_workers(environment, monkeypatch):  # noqa: F811
    env = environment
    manager = snapshot_manager(env, monkeypatch)
    snapshot = SimpleNamespace(version=8)
    result = await manager.update_replica_from_snapshot(snapshot, 0)
    assert result["updated"] == 1
    assert env.replicas[0].version == 8
    assert env.replicas[1].version == 7
    assert env.snapshots == [snapshot]
    assert not any(name == "trainer" for name, _ in env.events)
    assert not env.topologies
    assert env.serving_during_transfer == [{"s1"}]


@pytest.mark.asyncio
async def test_snapshot_retry_reuses_original_manifest(environment, monkeypatch):  # noqa: F811
    env = environment
    manager = snapshot_manager(env, monkeypatch)
    env.failures[("s0", "transfer")] = 1
    snapshot = SimpleNamespace(version=8)
    result = await manager.update_replica_from_snapshot(snapshot, 0)
    assert result["replicas"][0]["attempts"] == 2
    assert env.snapshots == [snapshot, snapshot]
    assert (await manager.get_replica_states())[0]["weight_version"] == 8


@pytest.mark.asyncio
async def test_snapshot_failure_quarantines_and_explicit_recovery_works(environment, monkeypatch):  # noqa: F811
    env = environment
    manager = snapshot_manager(env, monkeypatch, max_retries=0)
    env.failures[("s0", "transfer")] = 1
    result = await manager.update_replica_from_snapshot(SimpleNamespace(version=8), 0)
    assert result["quarantined"] == 1
    assert not env.replicas[0].generation_open
    assert env.replicas[1].generation_open
    result = await manager.update_replica_from_snapshot(SimpleNamespace(version=9), 0, recovery=True)
    assert result["updated"] == 1
    assert env.replicas[0].version == 9
    assert env.replicas[0].generation_open


@pytest.mark.asyncio
async def test_snapshot_cancellation_finishes_isolation_before_return(environment, monkeypatch):  # noqa: F811
    env = environment
    manager = snapshot_manager(env, monkeypatch)
    env.transfer_started = asyncio.Event()
    env.transfer_continue = asyncio.Event()
    updating = asyncio.create_task(manager.update_replica_from_snapshot(SimpleNamespace(version=8), 0))
    await env.transfer_started.wait()
    updating.cancel()
    await asyncio.sleep(0)
    assert not updating.done()
    env.transfer_continue.set()
    with pytest.raises(asyncio.CancelledError):
        await updating
    assert (await manager.get_replica_states())[0]["lifecycle_state"] == "QUARANTINED"
    assert not env.replicas[0].generation_open


@pytest.mark.asyncio
async def test_checkpoint_worker_consumes_snapshot_without_receive_collective(monkeypatch):
    import verl.checkpoint_engine.weight_snapshot as snapshots

    async def weights(snapshot):
        assert snapshot.version == 8
        yield "embedding", torch.tensor([1.0, 2.0])
        yield "head", torch.tensor([3.0])

    class Adapter:
        async def update_weights(self, weights, global_steps, wire_format):
            self.version = global_steps
            self.format = wire_format
            self.weights = {name: tensor.tolist() async for name, tensor in weights}

    worker = base.CheckpointEngineWorker.__new__(base.CheckpointEngineWorker)
    worker.server_adapter = Adapter()
    monkeypatch.setattr(snapshots, "iter_snapshot_weights", weights)
    monkeypatch.setattr(base, "get_device_id", lambda: "cpu", raising=False)
    await base.CheckpointEngineWorker.update_weights_from_snapshot.__wrapped__(worker, SimpleNamespace(version=8))
    assert worker.server_adapter.weights == {"embedding": [1.0, 2.0], "head": [3.0]}
    assert worker.server_adapter.version == 8
    assert worker.server_adapter.format == "named_tensors"
