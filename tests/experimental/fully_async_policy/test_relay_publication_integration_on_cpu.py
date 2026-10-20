# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
from types import SimpleNamespace

import pytest

from tests.experimental.fully_async_policy.test_joint_publication_integration_on_cpu import Trainer, config


@pytest.mark.asyncio
@pytest.mark.parametrize("source_nodes", [[None, "export-node"], [None, None], ["a", "b"]])
async def test_trainer_routes_rdma_master_to_the_unique_snapshot_export_source(monkeypatch, source_nodes):
    from verl.checkpoint_engine import weight_publication

    created = []
    monkeypatch.setattr(
        weight_publication,
        "WeightPublisherActor",
        SimpleNamespace(remote=lambda **kwargs: created.append(kwargs) or "publisher"),
    )
    instance = Trainer.__new__(Trainer)
    instance.config = config()
    instance.config.actor_rollout_ref.rollout.weight_publication.transport = "mooncake_rdma"
    instance.config.actor_rollout_ref.rollout.weight_publication.relay = {
        "device_name": "mlx5_1",
        "max_staging_bytes": 8192,
        "pipeline_depth": 2,
    }
    instance.actor_rollout_wg = SimpleNamespace(
        get_weight_publication_node_id=lambda: [asyncio.sleep(0, result=node) for node in source_nodes]
    )
    instance.rollouter = SimpleNamespace(
        get_replicas=SimpleNamespace(remote=lambda: asyncio.sleep(0, result=["replica"])),
        get_load_balancer=SimpleNamespace(remote=lambda: asyncio.sleep(0, result="router")),
    )
    if len([node for node in source_nodes if node is not None]) != 1:
        with pytest.raises(RuntimeError, match="source"):
            await instance._setup_checkpoint_manager()
        assert not created
        return
    await instance._setup_checkpoint_manager()
    assert created[0]["source_node_id"] == "export-node"
    assert created[0]["relay_config"].device_name == "mlx5_1"
    assert created[0]["relay_config"].pipeline_depth == 2
