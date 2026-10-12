# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import pytest
from omegaconf import OmegaConf

from verl.workers.config import RolloutConfig


def test_publication_is_opt_in():
    assert not RolloutConfig(name="rtp_llm").weight_publication.enabled


@pytest.mark.parametrize("container", [dict, OmegaConf.create])
def test_publication_normalizes_configuration(container):
    config = RolloutConfig(
        name="rtp_llm",
        fine_grained_weight_update={"enabled": True},
        weight_publication=container({"enabled": True, "wait_timeout_s": 30}),
    )
    assert config.weight_publication.enabled
    assert config.weight_publication.wait_timeout_s == 30


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_publication_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="wait_timeout_s"):
        RolloutConfig(name="rtp_llm", weight_publication={"wait_timeout_s": timeout})


def test_publication_requires_fine_grained_updates():
    with pytest.raises(ValueError, match="fine_grained"):
        RolloutConfig(name="rtp_llm", weight_publication={"enabled": True})


def test_publication_rejects_wrong_nested_type():
    with pytest.raises(TypeError, match="weight_publication"):
        RolloutConfig(name="rtp_llm", weight_publication="enabled")
