# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

from types import SimpleNamespace

import pytest
import torch

from verl.checkpoint_engine import base
from verl.workers import engine_workers


@pytest.mark.asyncio
async def test_checkpoint_worker_installs_only_its_local_relay_snapshot(monkeypatch):
    import verl.checkpoint_engine.weight_snapshot as snapshots

    local = SimpleNamespace(version=4, node="local")
    remote = SimpleNamespace(version=4, node="remote")

    async def weights(snapshot):
        assert snapshot is local
        yield "weight", torch.tensor([1.0, 2.0])

    class Adapter:
        async def update_weights(self, weights, global_steps, wire_format):
            self.version = global_steps
            self.format = wire_format
            self.weights = {name: tensor.tolist() async for name, tensor in weights}

    worker = base.CheckpointEngineWorker.__new__(base.CheckpointEngineWorker)
    worker.server_adapter = Adapter()
    monkeypatch.setattr(base.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "local"))
    monkeypatch.setattr(base, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(snapshots, "iter_snapshot_weights", weights)
    await base.CheckpointEngineWorker.update_weights_from_relay.__wrapped__(worker, {"local": local, "remote": remote})
    assert worker.server_adapter.weights == {"weight": [1.0, 2.0]}
    assert worker.server_adapter.version == 4
    assert worker.server_adapter.format == "named_tensors"
    assert base.CheckpointEngineWorker.get_weight_relay_node_id.__wrapped__(worker) == "local"


@pytest.mark.asyncio
async def test_checkpoint_worker_rejects_missing_local_or_mixed_version_manifests(monkeypatch):
    worker = base.CheckpointEngineWorker.__new__(base.CheckpointEngineWorker)
    monkeypatch.setattr(base.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "local"))
    with pytest.raises(ValueError, match="local"):
        await base.CheckpointEngineWorker.update_weights_from_relay.__wrapped__(
            worker, {"remote": SimpleNamespace(version=4)}
        )
    with pytest.raises(ValueError, match="version"):
        await base.CheckpointEngineWorker.update_weights_from_relay.__wrapped__(
            worker, {"local": SimpleNamespace(version=4), "remote": SimpleNamespace(version=5)}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("rank, expected", [(0, "source-node"), (1, None)])
async def test_only_snapshot_export_rank_reports_source_node(monkeypatch, rank, expected):
    worker = engine_workers.ActorRolloutRefWorker.__new__(engine_workers.ActorRolloutRefWorker)
    worker.role = "actor"
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(base.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "source-node"))
    result = await engine_workers.ActorRolloutRefWorker.get_weight_publication_node_id.__wrapped__(worker)
    assert result == expected
