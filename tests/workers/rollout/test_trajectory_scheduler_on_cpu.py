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

from verl.workers.rollout.router import GlobalRequestLoadBalancer
from verl.workers.rollout.trajectory_scheduler import GateResult, ReplicaSnapshot, TrajectoryScheduler


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


class _RejectAllGate:
    def evaluate(self, context):
        return GateResult(False, "custom placement policy rejected target")


def _choose(source_metadata, target_metadata, **overrides):
    config = {**_config(), **overrides}
    return TrajectoryScheduler(config).choose(
        {"request_id": "trajectory-1"},
        "s0",
        [ReplicaSnapshot("s0", 2, source_metadata), ReplicaSnapshot("s1", 0, target_metadata)],
    )


def test_enabled_cross_version_recompute_does_not_require_kv_capabilities():
    target, diagnostics = _choose(
        {"model_id": "model-a", "weight_version": 7},
        {"model_id": "model-a", "weight_version": 8},
        allow_cross_version_recompute=True,
    )

    assert target is not None
    assert target.server_id == "s1"
    assert diagnostics["mode"] == "recompute"
    assert diagnostics["source_version"] == 7
    assert diagnostics["target_version"] == 8


def test_same_version_keeps_remote_prefix_mode_with_recompute_enabled():
    target, diagnostics = _choose(_metadata(7), _metadata(7), allow_cross_version_recompute=True)

    assert target is not None
    assert diagnostics["mode"] == "remote_prefix"
    assert diagnostics["source_version"] == diagnostics["target_version"] == 7


def test_disabled_cross_version_recompute_cannot_be_bypassed_by_custom_gate_chain():
    target, diagnostics = _choose(
        _metadata(7),
        _metadata(8),
        gate_classes=["verl.workers.rollout.trajectory_scheduler.DifferentReplicaGate"],
    )

    assert target is None
    assert diagnostics["rejected_by"].endswith("SameWeightVersionGate")


@pytest.mark.parametrize(
    "source_overrides,target_overrides,gate",
    [
        ({"weight_version": None}, {}, "SameWeightVersionGate"),
        ({}, {"weight_version": None}, "SameWeightVersionGate"),
        ({}, {"model_id": "another-model"}, "SameModelGate"),
        ({"model_id": None}, {"model_id": None}, "SameModelGate"),
    ],
)
def test_recompute_enforces_identity_and_known_versions_with_custom_gates(source_overrides, target_overrides, gate):
    target, diagnostics = _choose(
        {**_metadata(7), **source_overrides},
        {**_metadata(8), **target_overrides},
        allow_cross_version_recompute=True,
        gate_classes=["verl.workers.rollout.trajectory_scheduler.DifferentReplicaGate"],
    )

    assert target is None
    assert diagnostics["rejected_by"].endswith(gate)


def test_recompute_preserves_custom_migration_gate(monkeypatch):
    from verl.workers.rollout import trajectory_scheduler

    original_loader = trajectory_scheduler.load_class_from_fqn
    monkeypatch.setattr(
        trajectory_scheduler,
        "load_class_from_fqn",
        lambda path, label: _RejectAllGate if path == "custom.RejectAllGate" else original_loader(path, label),
    )

    target, diagnostics = _choose(
        _metadata(7),
        _metadata(8),
        allow_cross_version_recompute=True,
        gate_classes=["custom.RejectAllGate"],
    )

    assert target is None
    assert diagnostics["reason"] == "custom placement policy rejected target"


def test_updated_single_replica_can_recompute_its_older_trajectory():
    scheduler = TrajectoryScheduler({**_config(), "allow_cross_version_recompute": True})
    replica = ReplicaSnapshot("s0", 0, _metadata(1))

    target, diagnostics = scheduler.choose(
        {"request_id": "trajectory-1"}, "s0", [replica], source_metadata={"weight_version": 0}
    )

    assert target is replica
    assert diagnostics["mode"] == "recompute"
    assert diagnostics["source_version"] == 0
    assert diagnostics["target_version"] == 1
    assert replica.metadata["weight_version"] == 1


@pytest.mark.parametrize("lifecycle", ["UPDATING", "QUARANTINED"])
def test_recompute_never_selects_isolated_replica(lifecycle):
    scheduler = TrajectoryScheduler({**_config(), "allow_cross_version_recompute": True})
    source = ReplicaSnapshot("s0", 2, _metadata(0), lifecycle_state="SERVING")
    isolated = ReplicaSnapshot("s1", 0, _metadata(1), lifecycle_state=lifecycle)

    target, _ = scheduler.choose({"request_id": "trajectory-1"}, "s0", [source, isolated])

    assert target is None


@pytest.mark.parametrize("same_replica", [False, True])
def test_required_resume_ignores_load_threshold_but_discretionary_migration_does_not(same_replica):
    scheduler = TrajectoryScheduler({**_config(), "allow_cross_version_recompute": True, "min_score_improvement": 2.0})
    source = ReplicaSnapshot("s0", 0, _metadata(0), lifecycle_state="QUARANTINED")
    available = ReplicaSnapshot("s0" if same_replica else "s1", 5, _metadata(1))
    replicas = [available] if same_replica else [source, available]

    target, _ = scheduler.choose(
        {"request_id": "trajectory-1", "migration_required": True},
        "s0",
        replicas,
        source_metadata={"weight_version": 0},
    )
    discretionary_target, _ = scheduler.choose(
        {"request_id": "trajectory-1"}, "s0", replicas, source_metadata={"weight_version": 0}
    )

    assert target is available
    assert discretionary_target is None


def test_required_resume_can_return_to_unchanged_source_without_a_transfer():
    source = ReplicaSnapshot("s0", 2, _metadata(7))
    isolated = ReplicaSnapshot("s1", 0, _metadata(8), lifecycle_state="QUARANTINED")
    scheduler = TrajectoryScheduler({**_config(), "allow_cross_version_recompute": True})

    target, diagnostics = scheduler.choose(
        {"request_id": "trajectory-1", "migration_required": True},
        "s0",
        [source, isolated],
        source_metadata={"weight_version": 7},
    )

    assert target is source
    assert diagnostics["resume_source"] is True
    assert "mode" not in diagnostics
    assert diagnostics["source_version"] == diagnostics["target_version"] == 7
