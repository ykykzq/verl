# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Opt-in hardware acceptance: VERL_RUN_RDMA_RELAY_TEST=1 RAY_ADDRESS=auto pytest ..."""

import asyncio
import os
from types import SimpleNamespace

import pytest
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.checkpoint_engine.relay_distribution import RelayDistribution
from verl.checkpoint_engine.weight_snapshot import export_weight_snapshot, iter_snapshot_weights


def expected_weights(version):
    return (
        ("matrix", (torch.arange(24, dtype=torch.bfloat16).reshape(6, 4) + version).T),
        ("scalar", torch.tensor(version + 0.5, dtype=torch.float64)),
        ("indices", torch.arange(5, dtype=torch.int64) + version),
        ("mask", torch.tensor([True, False, True])),
        ("empty", torch.empty((0, 3), dtype=torch.float32)),
        ("half", torch.arange(7, dtype=torch.float16) + version),
    )


@ray.remote(num_cpus=0, max_restarts=0)
class SnapshotSource:
    def export(self, version):
        return export_weight_snapshot(expected_weights(version), version, bucket_size=32, is_master=True)


@ray.remote(num_cpus=0, max_restarts=0)
class RelayConsumer:
    def __init__(self, logical_node=None):
        self.node_id = logical_node or ray.get_runtime_context().get_node_id()

    def get_weight_relay_node_id(self):
        return self.node_id

    async def verify(self, snapshot, version):
        expected = dict(expected_weights(version))
        actual = {name: tensor async for name, tensor in iter_snapshot_weights(snapshot)}
        assert snapshot.version == version
        assert actual.keys() == expected.keys()
        for name, tensor in actual.items():
            reference = expected[name]
            assert tensor.device.type == "cpu"
            assert tensor.dtype == reference.dtype
            assert tensor.shape == reference.shape
            assert torch.equal(tensor, reference), name
        return {"node_id": self.node_id, "version": version, "tensor_count": len(actual)}


async def assert_no_registered_storage(relays, timeout=30):
    async def check():
        while True:
            statuses = await asyncio.gather(*(relay.get_status.remote() for relay in relays))
            if all(status["registered_bytes"] == status["registered_buckets"] == 0 for status in statuses):
                return statuses
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(check(), timeout)


@pytest.mark.skipif(
    os.environ.get("VERL_RUN_RDMA_RELAY_TEST") != "1",
    reason="requires explicit opt-in and a multi-node Ray cluster with working Mooncake RDMA",
)
@pytest.mark.asyncio
async def test_real_rdma_relay_multinode_round_trip_and_registered_memory_cleanup():
    connected_here = not ray.is_initialized()
    actors, relays = [], []
    distribution = None
    try:
        if connected_here:
            ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=False)
        nodes = sorted(node["NodeID"] for node in ray.nodes() if node["Alive"])
        assert len(nodes) >= 2, "RDMA acceptance requires at least two live Ray nodes"
        nodes = nodes[:3]
        source = SnapshotSource.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[0], soft=False)
        ).remote()
        actors.append(source)
        consumers = []
        for node in nodes:
            consumer = RelayConsumer.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)
            ).remote()
            consumers.append(consumer)
            actors.append(consumer)
        replicas = [SimpleNamespace(replica_rank=rank, workers=[worker]) for rank, worker in enumerate(consumers)]
        distribution = RelayDistribution(
            replicas,
            nodes[0],
            {
                "device_name": os.environ.get("VERL_RDMA_DEVICE_NAME", ""),
                "max_staging_bytes": 1 << 20,
                "pipeline_depth": 2,
            },
        )
        snapshot = await source.export.remote(7)
        assert len(snapshot.buckets) > 1
        distribution.pin(snapshot)
        timeout = float(os.environ.get("VERL_RDMA_TEST_TIMEOUT_S", "180"))
        for rank, (node, worker) in enumerate(zip(nodes, consumers, strict=True)):
            manifests = await asyncio.wait_for(distribution.snapshots_for(rank, 7), timeout)
            assert set(manifests) == {node}
            result = await worker.verify.remote(manifests[node], 7)
            assert result == {"node_id": node, "version": 7, "tensor_count": 6}
        relays = list(distribution.relays.values())
        statuses = await assert_no_registered_storage(relays, timeout)
        assert {status["node_id"] for status in statuses} == set(nodes)
        assert not distribution.get_status()["node_errors"]
        assert distribution.get_status()["fatal_error"] is None
        distribution.unpin(snapshot)
        await distribution.close()
        assert distribution.closed and not distribution.relays
        distribution = None
    finally:
        if distribution is not None:
            relays = list(distribution.relays.values())
            await distribution.close()
        for actor in [*relays, *actors]:
            ray.kill(actor, no_restart=True)
        if connected_here and ray.is_initialized():
            ray.shutdown()
