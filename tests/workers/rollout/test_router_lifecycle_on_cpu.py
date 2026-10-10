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


def _metadata(version=1):
    return {
        "model_id": "model-a",
        "weight_version": version,
        "kv_transfer_backends": ["remote_prefix"],
        "kv_transfer_domain": "kvcm-a",
        "kv_transfer_namespace": f"kvcm-a:{version}",
    }


def _router(**kwargs):
    return GlobalRequestLoadBalancer(
        {"s0": None, "s1": None},
        server_metadata={"s0": _metadata(), "s1": _metadata()},
        trajectory_migration_config={
            "enabled": True,
            "scorer_kwargs": {"affinity_weight": 0.0},
        },
        **kwargs,
    )


def _plan(router):
    return router.plan_trajectory_migration("trajectory", "s0", {"request_id": "trajectory"})


def test_pin_captures_version_and_prevents_update_but_not_serving():
    router = _router()
    pinned = router.pin_replica("s0", reason="comparison")
    assert pinned["pin"] == {"weight_version": 1, "reason": "comparison"}
    assert pinned["lifecycle_state"] == "SERVING"
    with pytest.raises(RuntimeError, match="pinned"):
        router.begin_replica_update("s0", 2)
    router.begin_replica_update("s1", 2)
    assert router.acquire_server("request")[0] == "s0"
    pinned["pin"]["reason"] = "changed outside router"
    assert router.get_replica_states()["s0"]["pin"]["reason"] == "comparison"
    assert router.unpin_replica("s0")["pin"] is None
    assert router.begin_replica_update("s0", 2)["desired_weight_version"] == 2


def test_pin_rejects_a_different_version_and_updating_replica():
    router = _router()
    with pytest.raises(ValueError, match="version"):
        router.pin_replica("s0", weight_version=2)
    router.begin_replica_update("s0", 2)
    with pytest.raises(RuntimeError, match="UPDATING"):
        router.pin_replica("s0")


def test_commit_publishes_only_the_requested_version_and_returns_detached_state():
    router = _router()
    state = router.begin_replica_update("s0", 2)
    assert state["weight_version"] == 1
    assert state["lifecycle_state"] == "UPDATING"
    state["weight_version"] = 99
    with pytest.raises(ValueError, match="version"):
        router.commit_replica_update("s0", 3)
    committed = router.commit_replica_update("s0", 2, metadata=_metadata(99))
    assert committed == {
        "lifecycle_state": "SERVING",
        "weight_version": 2,
        "desired_weight_version": None,
        "pin": None,
        "last_update_error": None,
    }
    assert router.get_status()["server_metadata"]["s0"]["weight_version"] == 2
    assert router.get_replica_states()["s0"] == committed


def test_failed_update_is_quarantined_until_explicit_recovery():
    router = _router()
    router.begin_replica_update("s0", 2)
    failed = router.fail_replica_update("s0", "partial stream")
    assert failed["lifecycle_state"] == "QUARANTINED"
    assert failed["weight_version"] == 1
    assert failed["desired_weight_version"] is None
    assert failed["last_update_error"] == "partial stream"
    with pytest.raises(RuntimeError, match="QUARANTINED"):
        router.begin_replica_update("s0", 3)
    recovering = router.begin_replica_recovery("s0", 3)
    assert recovering["lifecycle_state"] == "UPDATING"
    assert recovering["weight_version"] == 1
    assert recovering["desired_weight_version"] == 3
    recovered = router.commit_replica_update("s0", 3)
    assert recovered["last_update_error"] is None
    assert recovered["weight_version"] == 3


@pytest.mark.parametrize(
    "operation,args",
    [
        ("commit_replica_update", ("s0", 2)),
        ("fail_replica_update", ("s0", "failed")),
        ("begin_replica_recovery", ("s0", 2)),
    ],
)
def test_invalid_lifecycle_transitions_fail_closed(operation, args):
    router = _router()
    with pytest.raises(RuntimeError, match="SERVING"):
        getattr(router, operation)(*args)


@pytest.mark.parametrize("full_determinism", [False, True])
@pytest.mark.parametrize("quarantine", [False, True])
def test_isolated_replica_is_excluded_from_all_routes(full_determinism, quarantine):
    router = _router(full_determinism=full_determinism)
    router._request_id_to_server["sticky"] = "s0"
    router.begin_replica_update("s0", 2)
    if quarantine:
        router.fail_replica_update("s0", "stream failed")
    for request_id in ["sticky", "new-1", "new-2", "new-3"]:
        assert router.acquire_server(request_id)[0] == "s1"
    assert router.get_all_servers() == ["s0", "s1"]
    status = router.get_status()
    assert status["active_servers"] == 1
    assert status["replica_states"]["s0"]["weight_version"] == 1
    router.begin_replica_update("s1", 2)
    with pytest.raises(RuntimeError, match="No available servers"):
        router.acquire_server("no-replica")


@pytest.mark.parametrize("isolated", ["s0", "s1"])
def test_update_invalidates_reservations_involving_source_or_target(isolated):
    router = _router()
    decision = asyncio.run(_plan(router))
    assert decision["migrate"] is True
    router.begin_replica_update(isolated, 2)
    assert router.get_status()["pending_migrations"] == {}
    assert router._migration_reservations["s1"] == 0
    with pytest.raises(RuntimeError, match="stale"):
        router.commit_trajectory_migration("trajectory", decision["decision_id"])


def test_nonserving_replicas_cannot_be_migration_targets():
    router = _router()
    router.begin_replica_update("s1", 2)
    assert asyncio.run(_plan(router))["migrate"] is False
    router.fail_replica_update("s1", "failed")
    assert asyncio.run(_plan(router))["migrate"] is False


def test_metadata_discovers_legacy_versions_but_cannot_overwrite_pins_or_commits():
    router = _router()
    router.update_server_metadata("s0", _metadata(2))
    assert router.get_replica_states()["s0"]["weight_version"] == 2
    router.pin_replica("s0")
    router.update_server_metadata("s0", _metadata(3))
    assert router.get_status()["server_metadata"]["s0"]["weight_version"] == 2
    router.unpin_replica("s0")
    router.begin_replica_update("s0", 3)
    router.update_server_metadata("s0", _metadata(3))
    assert router.get_replica_states()["s0"]["weight_version"] == 2
    router.commit_replica_update("s0", 3, metadata=_metadata(3))
    router.update_server_metadata("s0", _metadata(2))
    assert router.get_status()["server_metadata"]["s0"] == _metadata(3)


@pytest.mark.parametrize("action", ["update", "commit", "pin"])
def test_capability_refresh_cannot_revert_lifecycle_changes_while_awaiting(action):
    async def scenario():
        router = _router()
        started = asyncio.Event()
        resume = asyncio.Event()

        async def capabilities():
            started.set()
            await resume.wait()
            return _metadata(1)

        router.add_servers(
            {"s1": SimpleNamespace(get_trajectory_migration_capabilities=SimpleNamespace(remote=capabilities))}
        )
        planning = asyncio.create_task(_plan(router))
        await started.wait()
        if action == "pin":
            router.update_server_metadata("s1", _metadata(2))
            router.pin_replica("s1")
        else:
            router.begin_replica_update("s1", 2)
            if action == "commit":
                router.commit_replica_update("s1", 2, metadata=_metadata(2))
        resume.set()
        decision = await planning
        assert decision["migrate"] is False
        if action != "update":
            assert router.get_status()["server_metadata"]["s1"] == _metadata(2)
        else:
            assert router.get_replica_states()["s1"]["lifecycle_state"] == "UPDATING"

    asyncio.run(scenario())


def test_remove_and_reregister_starts_with_clean_lifecycle():
    router = _router()
    router.pin_replica("s0")
    router.remove_servers(["s0"])
    assert "s0" not in router.get_replica_states()
    router.add_servers({"s0": None})
    state = router.get_replica_states()["s0"]
    assert state["lifecycle_state"] == "SERVING"
    assert state["pin"] is None
    assert state["weight_version"] is None


def test_old_segment_version_is_used_only_in_the_source_snapshot():
    router = GlobalRequestLoadBalancer(
        {"s0": None, "s1": None},
        server_metadata={"s0": _metadata(2), "s1": _metadata(2)},
        trajectory_migration_config={
            "enabled": True,
            "allow_cross_version_recompute": True,
            "scorer_kwargs": {"affinity_weight": 0.0},
        },
    )
    decision = asyncio.run(
        router.plan_trajectory_migration("trajectory", "s0", {"request_id": "trajectory"}, source_metadata=_metadata(1))
    )
    assert router.get_replica_states()["s0"]["weight_version"] == 2
    assert decision["migrate"] is True
    assert decision["mode"] == "recompute"
    assert decision["source_version"] == 1
    assert decision["target_version"] == 2
    assert decision["source_metadata"]["weight_version"] == 1


def test_update_requires_an_explicit_version_and_cannot_restart_in_progress():
    router = _router()
    with pytest.raises(ValueError, match="version"):
        router.begin_replica_update("s0", None)
    router.begin_replica_update("s0", 2)
    with pytest.raises(RuntimeError, match="UPDATING"):
        router.begin_replica_update("s0", 3)


@pytest.mark.parametrize("lifecycle", ["SERVING", "UPDATING", "QUARANTINED"])
def test_same_replica_recomputes_old_tokens_only_after_update_commit(lifecycle):
    router = GlobalRequestLoadBalancer(
        {"s0": None},
        server_metadata={"s0": _metadata(1)},
        trajectory_migration_config={"enabled": True, "allow_cross_version_recompute": True},
    )
    router.begin_replica_update("s0", 2)
    router.commit_replica_update("s0", 2, metadata=_metadata(2))
    if lifecycle != "SERVING":
        router.begin_replica_update("s0", 3)
        if lifecycle == "QUARANTINED":
            router.fail_replica_update("s0", "transfer failed")
    decision = asyncio.run(
        router.plan_trajectory_migration("trajectory", "s0", {"request_id": "trajectory"}, source_metadata=_metadata(1))
    )
    assert decision["migrate"] is (lifecycle == "SERVING")
    if lifecycle == "SERVING":
        assert decision["target_server_id"] == "s0"
        assert decision["mode"] == "recompute"
        assert decision["source_version"] == 1
        assert decision["target_version"] == 2
        assert decision["source_metadata"]["weight_version"] == 1
        assert decision["target_metadata"]["weight_version"] == 2
        router.commit_trajectory_migration("trajectory", decision["decision_id"])
        assert router.acquire_server("trajectory")[0] == "s0"
