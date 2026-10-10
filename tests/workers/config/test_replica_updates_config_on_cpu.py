# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
from omegaconf import OmegaConf

from verl.workers import config as worker_config
from verl.workers.config import RolloutConfig, TrajectoryMigrationConfig


def test_replica_update_features_are_opt_in():
    config = RolloutConfig(name="rtp_llm")
    assert config.fine_grained_weight_update.enabled is False
    assert config.trajectory_migration.allow_cross_version_recompute is False


@pytest.mark.parametrize("container", [dict, OmegaConf.create])
def test_rollout_normalizes_nested_update_configuration(container):
    config = RolloutConfig(
        name="rtp_llm",
        fine_grained_weight_update=container({"enabled": True, "max_retries": 0, "retry_backoff_s": 0}),
        trajectory_migration=container({"enabled": True, "allow_cross_version_recompute": True}),
    )
    assert isinstance(config.fine_grained_weight_update, worker_config.FineGrainedWeightUpdateConfig)
    assert config.fine_grained_weight_update.max_retries == 0
    assert config.fine_grained_weight_update.retry_backoff_s == 0
    assert config.trajectory_migration.allow_cross_version_recompute is True


@pytest.mark.parametrize("field", ["max_retries", "retry_backoff_s"])
def test_negative_retry_configuration_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        RolloutConfig(name="rtp_llm", fine_grained_weight_update={field: -1})


def test_wrong_nested_update_configuration_type_is_rejected():
    with pytest.raises(TypeError, match="fine_grained_weight_update must be"):
        RolloutConfig(name="rtp_llm", fine_grained_weight_update="enabled")


def test_same_version_migration_keeps_existing_defaults():
    migration = TrajectoryMigrationConfig()
    assert migration.allow_cross_version_recompute is False
    assert migration.kv_transfer_backend == "remote_prefix"
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        RolloutConfig(name="rtp_llm", trajectory_migration={"enabled": True}, enable_prefix_caching=False)


def test_cross_version_recompute_allows_prefix_cache_to_be_disabled():
    config = RolloutConfig(
        name="rtp_llm",
        trajectory_migration={"enabled": True, "allow_cross_version_recompute": True},
        enable_prefix_caching=False,
    )
    assert config.trajectory_migration.allow_cross_version_recompute is True


def test_fine_grained_updates_require_rtp_llm():
    with pytest.raises(ValueError, match="rtp_llm"):
        RolloutConfig(name="vllm", fine_grained_weight_update={"enabled": True})
