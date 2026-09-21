# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import asyncio
from types import SimpleNamespace

from verl.workers.rollout.router import GlobalRequestLoadBalancer


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _metadata(version="v1"):
    return {
        "model_id": "model-a",
        "weight_version": version,
        "kv_transfer_backends": ["remote_prefix"],
        "kv_transfer_domain": "kvcm-a",
        "kv_transfer_namespace": f"kvcm-a:{version}",
    }


def _config():
    return {
        "enabled": True,
        "scorer_class": "verl.workers.rollout.trajectory_scheduler.LoadAwareTrajectoryReplicaScorer",
        "scorer_kwargs": {"load_weight": 1.0, "affinity_weight": 0.0},
        "gate_classes": [
            "verl.workers.rollout.trajectory_scheduler.DifferentReplicaGate",
            "verl.workers.rollout.trajectory_scheduler.SameModelGate",
            "verl.workers.rollout.trajectory_scheduler.SameWeightVersionGate",
            "verl.workers.rollout.trajectory_scheduler.KVTransferCapabilityGate",
        ],
        "kv_transfer_backend": "remote_prefix",
    }


def test_plan_and_commit_move_sticky_trajectory_to_scored_replica():
    router = GlobalRequestLoadBalancer(
        servers={"s0": None, "s1": None},
        trajectory_migration_config=_config(),
        server_metadata={"s0": _metadata(), "s1": _metadata()},
    )
    router._request_id_to_server["trajectory-1"] = "s0"
    router._inflight_requests["s0"] = 2

    decision = asyncio.run(
        router.plan_trajectory_migration(
            request_id="trajectory-1",
            source_server_id="s0",
            trajectory={"request_id": "trajectory-1", "generated_tokens": 64},
        )
    )

    assert decision["migrate"] is True
    assert decision["target_server_id"] == "s1"
    router.commit_trajectory_migration("trajectory-1", decision["decision_id"])
    server_id, _ = router.acquire_server("trajectory-1")
    assert server_id == "s1"


def test_weight_version_gate_rejects_movement():
    router = GlobalRequestLoadBalancer(
        servers={"s0": None, "s1": None},
        trajectory_migration_config=_config(),
        server_metadata={"s0": _metadata("v1"), "s1": _metadata("v2")},
    )
    router._inflight_requests["s0"] = 2

    decision = asyncio.run(
        router.plan_trajectory_migration(
            request_id="trajectory-1",
            source_server_id="s0",
            trajectory={"request_id": "trajectory-1"},
        )
    )

    assert decision["migrate"] is False
    assert decision["rejected_by"].endswith("SameWeightVersionGate")


def test_kv_namespace_gate_rejects_movement():
    source = _metadata()
    target = _metadata()
    target["kv_transfer_namespace"] = "another-weight-namespace"
    router = GlobalRequestLoadBalancer(
        servers={"s0": None, "s1": None},
        trajectory_migration_config=_config(),
        server_metadata={"s0": source, "s1": target},
    )
    router._inflight_requests["s0"] = 2

    decision = asyncio.run(
        router.plan_trajectory_migration(
            request_id="trajectory-1",
            source_server_id="s0",
            trajectory={"request_id": "trajectory-1"},
        )
    )

    assert decision["migrate"] is False
    assert decision["rejected_by"].endswith("KVTransferCapabilityGate")


def test_clear_sticky_cache_cancels_pending_migration_reservations():
    router = GlobalRequestLoadBalancer(
        servers={"s0": None, "s1": None},
        trajectory_migration_config=_config(),
        server_metadata={"s0": _metadata(), "s1": _metadata()},
    )
    router._inflight_requests["s0"] = 2
    decision = asyncio.run(
        router.plan_trajectory_migration(
            request_id="trajectory-1",
            source_server_id="s0",
            trajectory={"request_id": "trajectory-1"},
        )
    )
    assert decision["migrate"] is True

    router.clear_sticky_cache()

    assert router._pending_migrations == {}
    assert router._migration_reservations == {"s0": 0, "s1": 0}


def test_abort_trajectory_targets_only_its_sticky_replica():
    aborted = []
    server = SimpleNamespace(
        abort_request=_RemoteMethod(lambda request_id: aborted.append(request_id) or {"aborted": True})
    )
    router = GlobalRequestLoadBalancer(servers={"s0": server})
    router._request_id_to_server["trajectory-1"] = "s0"

    result = asyncio.run(router.abort_trajectory("trajectory-1"))

    assert result == {"aborted": True, "server_id": "s0"}
    assert aborted == ["trajectory-1"]
