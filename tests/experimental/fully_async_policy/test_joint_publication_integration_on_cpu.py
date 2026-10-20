# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy import fully_async_trainer as trainer_module

Trainer = trainer_module.FullyAsyncTrainer.__ray_metadata__.modified_class


def config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "hybrid_engine": False,
                "actor": {"strategy": "fsdp2"},
                "model": {"lora": {"rank": 0, "merge": False}},
                "rollout": {
                    "name": "rtp_llm",
                    "nnodes": 1,
                    "enable_prefix_caching": True,
                    "weight_publication": {
                        "enabled": True,
                        "joint_scheduling": {"enabled": True, "kv_capacity_tokens": 16384, "max_batch_size": 8},
                    },
                    "fine_grained_weight_update": {"enabled": True},
                    "trajectory_migration": {"enabled": True, "allow_cross_version_recompute": True},
                    "checkpoint_engine": {"_target_": "verl.workers.config.CheckpointEngineConfig"},
                },
            },
            "async_training": {
                "use_trainer_do_validate": False,
                "use_dynamic_resource_scheduling": False,
                "partial_rollout": True,
            },
            "rollout": {"nnodes": 1},
        }
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("trajectory_migration.enabled", False, "trajectory_migration"),
        ("trajectory_migration.allow_cross_version_recompute", False, "allow_cross_version_recompute"),
        ("enable_prefix_caching", False, "enable_prefix_caching"),
    ],
)
def test_trainer_rejects_incomplete_joint_configuration_before_allocating_workers(field, value, match):
    cfg = config()
    OmegaConf.update(cfg, f"actor_rollout_ref.rollout.{field}", value)
    with pytest.raises(ValueError, match=match):
        trainer_module.validate_weight_publication_config(cfg)


def test_joint_updates_require_partial_rollout_to_preserve_interrupted_trajectories():
    cfg = config()
    cfg.async_training.partial_rollout = False
    with pytest.raises(ValueError, match="partial_rollout"):
        trainer_module.validate_weight_publication_config(cfg)
    cfg.async_training.partial_rollout = True
    trainer_module.validate_weight_publication_config(cfg)


@pytest.mark.asyncio
async def test_trainer_passes_validated_joint_policy_to_publisher(monkeypatch):
    from verl.checkpoint_engine import weight_publication

    created = []

    def create(**kwargs):
        created.append(kwargs)
        return "publisher"

    monkeypatch.setattr(weight_publication, "WeightPublisherActor", SimpleNamespace(remote=create))
    instance = Trainer.__new__(Trainer)
    instance.config = config()
    instance.rollouter = SimpleNamespace(
        get_replicas=SimpleNamespace(remote=lambda: asyncio.sleep(0, result=["replica"])),
        get_load_balancer=SimpleNamespace(remote=lambda: asyncio.sleep(0, result="router")),
    )
    await instance._setup_checkpoint_manager()
    joint = created[0]["joint_scheduling_config"]
    assert joint.enabled and joint.kv_capacity_tokens == 16384 and joint.max_batch_size == 8
    assert instance.weight_publisher == "publisher"
