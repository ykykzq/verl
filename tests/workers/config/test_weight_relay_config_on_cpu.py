# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
from omegaconf import OmegaConf

from verl.workers.config import WeightPublicationConfig


def test_default_publication_uses_ray_without_a_relay_dependency():
    config = WeightPublicationConfig()
    assert config.transport == "ray"
    assert config.relay.pipeline_depth == 2
    assert config.relay.max_staging_bytes == 4 << 30


@pytest.mark.parametrize("container", [dict, OmegaConf.create])
def test_relay_configuration_normalizes_nested_settings(container):
    config = WeightPublicationConfig(
        enabled=True,
        transport="mooncake_rdma",
        relay=container({"device_name": "mlx5_0", "max_staging_bytes": 8192, "pipeline_depth": 3}),
    )
    assert config.relay.device_name == "mlx5_0"
    assert config.relay.max_staging_bytes == 8192
    assert config.relay.pipeline_depth == 3


@pytest.mark.parametrize("field", ["max_staging_bytes", "pipeline_depth"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2", float("inf"), float("nan")])
def test_relay_rejects_invalid_resource_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        WeightPublicationConfig(relay={field: value})


@pytest.mark.parametrize("transport", ["tcp", "rdma", "", None])
def test_publication_rejects_unknown_transport(transport):
    with pytest.raises(ValueError, match="transport"):
        WeightPublicationConfig(enabled=True, transport=transport)


def test_rdma_transport_requires_publication_and_typed_relay_settings():
    with pytest.raises(ValueError, match="enabled"):
        WeightPublicationConfig(transport="mooncake_rdma")
    with pytest.raises(TypeError, match="relay"):
        WeightPublicationConfig(relay="rdma")
    with pytest.raises(ValueError, match="device_name"):
        WeightPublicationConfig(relay={"device_name": ["mlx5_0"]})
