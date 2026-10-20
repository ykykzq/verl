# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from verl.workers.config import RolloutConfig


def joint_rollout_config():
    return {
        "name": "rtp_llm",
        "fine_grained_weight_update": {"enabled": True},
        "weight_publication": {
            "enabled": True,
            "joint_scheduling": {"enabled": True, "kv_capacity_tokens": 16384, "max_batch_size": 8},
        },
        "trajectory_migration": {"enabled": True, "allow_cross_version_recompute": True},
    }


def test_joint_scheduling_defaults_preserve_legacy_publication():
    assert not RolloutConfig(name="rtp_llm").weight_publication.joint_scheduling.enabled


@pytest.mark.parametrize("container", [dict, OmegaConf.create])
def test_joint_scheduling_normalizes_nested_configuration(container):
    values = joint_rollout_config()
    values["weight_publication"] = container(values["weight_publication"])
    config = RolloutConfig(**values)
    joint = config.weight_publication.joint_scheduling
    assert joint.enabled and joint.kv_capacity_tokens == 16384 and joint.max_batch_size == 8
    assert joint.min_decode_tokens == 64
    assert joint.max_defer_s == 30


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kv_capacity_tokens", 0),
        ("max_batch_size", 0),
        ("kv_block_size_tokens", 0),
        ("max_batch_size", True),
        ("min_decode_tokens", -1),
        ("poll_interval_s", 0),
        ("poll_interval_s", float("inf")),
        ("update_cooldown_s", float("nan")),
        ("min_decode_window_s", -1),
        ("max_defer_s", 0),
        ("evacuation_timeout_s", -1),
        ("staleness_version_gap", 0),
        ("tail_tokens", 0),
        ("tail_kl_threshold", float("nan")),
        ("probe_timeout_s", 0),
        ("max_concurrent_probes", 0),
        ("overload_threshold", 1.1),
        ("overload_threshold", 0),
    ],
)
def test_joint_scheduling_rejects_invalid_limits(field, value):
    values = joint_rollout_config()
    values["weight_publication"]["joint_scheduling"][field] = value
    with pytest.raises(ValueError, match=field):
        RolloutConfig(**values)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("weight_publication", "enabled", False, "weight_publication"),
        ("fine_grained_weight_update", "enabled", False, "fine_grained"),
        ("trajectory_migration", "enabled", False, "trajectory_migration"),
        ("trajectory_migration", "allow_cross_version_recompute", False, "allow_cross_version_recompute"),
    ],
)
def test_joint_scheduling_requires_complete_update_and_continuation_path(section, field, value, error):
    values = deepcopy(joint_rollout_config())
    values[section][field] = value
    with pytest.raises(ValueError, match=error):
        RolloutConfig(**values)


def test_joint_scheduling_rejects_wrong_nested_type():
    with pytest.raises(TypeError, match="joint_scheduling"):
        RolloutConfig(name="rtp_llm", weight_publication={"joint_scheduling": "enabled"})


def test_joint_scheduling_requires_prefix_caching_for_same_version_evacuation():
    values = joint_rollout_config()
    values["enable_prefix_caching"] = False
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        RolloutConfig(**values)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), 0, -1, True, "unbounded"])
def test_joint_migration_transfer_timeout_is_finite_and_positive(timeout):
    values = joint_rollout_config()
    values["trajectory_migration"]["transfer_timeout_s"] = timeout
    with pytest.raises(ValueError, match="transfer_timeout_s"):
        RolloutConfig(**values)
