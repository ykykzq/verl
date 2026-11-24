"""Version-only migration preserves lifecycle, capacity and decode-time fences."""

import asyncio
from types import SimpleNamespace

import pytest

from tests.workers.rollout.test_joint_scheduler_on_cpu import assign, metadata, plan, progress, router
from verl.workers.config.rollout import JointSchedulingConfig
from verl.workers.rollout.joint_scheduler import JointReplicaScheduler


def test_explicit_mode_and_legacy_default():
    assert JointSchedulingConfig().staleness_mode == "version_kl"
    assert JointSchedulingConfig(staleness_mode="version").staleness_mode == "version"
    for invalid in (None, True, "kl", "VERSION", []):
        with pytest.raises(ValueError, match="staleness_mode"):
            JointSchedulingConfig(staleness_mode=invalid)


@pytest.mark.parametrize("target_version,expected", [(1, False), (2, True), (5, True)])
def test_version_gap_drives_recompute_without_probe(target_version, expected):
    async def run():
        r, _ = router(staleness_mode="version", min_decode_tokens=0, max_defer_s=120)
        assign(r, "x", "a")
        r.prepare_weight_update(target_version)
        r.begin_replica_update("b", target_version)
        r.commit_replica_update("b", target_version, metadata(target_version))
        await progress(r, "x")  # Servers are None: an attempted probe would fail.
        assert not r._joint.probes and not r._joint.probe_resources
        assert not r._joint.trajectories["x"].staleness
        decision = await plan(r, "x")
        assert bool(decision["migrate"]) == expected
        if expected:
            assert decision["target_server_id"] == "b"
            assert decision["mode"] == "recompute"

    asyncio.run(run())


def scheduler(mode="version"):
    s = JointReplicaScheduler(
        dict(
            enabled=True,
            kv_capacity_tokens=128,
            max_batch_size=2,
            staleness_mode=mode,
            min_decode_tokens=0,
            max_defer_s=120,
        )
    )
    s.clock = lambda: 100.0
    s.register("x", 16, 48)
    s.acquire("x", {"a": {"lifecycle_state": "SERVING", "weight_version": 0}})
    s.trajectories["x"].weight_version = 0
    s.trajectories["x"].generated_tokens = 8
    return s


def snapshot(sid, version, lifecycle="SERVING", fingerprint="layout"):
    return SimpleNamespace(server_id=sid, metadata=metadata(version, fingerprint), lifecycle_state=lifecycle)


@pytest.mark.parametrize("kind", ["updating", "quarantined", "layout", "full", "sequence", "evacuating"])
def test_version_only_still_rejects_unsafe_targets(kind):
    s = scheduler()
    target = snapshot("b", 2)
    if kind in ("updating", "quarantined"):
        target.lifecycle_state = kind.upper()
    elif kind == "layout":
        target.metadata = metadata(2, "other")
    elif kind == "full":
        for rid in ("y", "z"):
            s.register(rid, 16, 48)
            s.acquire(rid, {"b": {"lifecycle_state": "SERVING", "weight_version": 2}})
    elif kind == "sequence":
        s.engine_limits["b"] = {"max_sequence_tokens": 32}
    elif kind == "evacuating":
        s.evacuating["b"] = 99.0
    assert s.target("x", "a", metadata(0), [snapshot("a", 0), target], False)[0] is None


def test_published_but_uninstalled_version_does_not_trigger_migration():
    s = scheduler()
    s.latest_version = 10
    assert s.target("x", "a", metadata(0), [snapshot("a", 0), snapshot("b", 1)], False)[0] is None
    assert not s.version_stale(None, 10)
    assert not s.version_stale(0, None)


def test_version_only_allows_newer_installed_version_with_pending_publication():
    s = scheduler()
    s.trajectories["x"].staleness = dict(source_version=0, target_version=2, kl=0.0)
    s.pending_since["b"] = -19.0
    assert s.target("x", "a", metadata(0), [snapshot("a", 0), snapshot("b", 3)], False) == (
        "b",
        "version-gap staleness",
    )
    s.maybe_probe("x", [1] * 24, {}, {}, {})
    assert not s.probes and not s.probe_resources and not s.probe_errors


def test_no_token_floor_but_two_second_window_and_required_resume_remain():
    s = scheduler()
    rec = s.trajectories["x"]
    rec.moved_at = 99.0
    rec.moved_tokens = rec.generated_tokens
    replicas = [snapshot("a", 0), snapshot("b", 2)]
    assert s.protected(rec)
    assert s.target("x", "a", metadata(0), replicas, False)[0] is None
    assert s.target("x", "a", metadata(0), replicas, True) == ("a", "resume source")
    s.clock = lambda: 101.0
    assert not s.protected(rec)
    assert s.target("x", "a", metadata(0), replicas, False)[0] == "b"


def test_max_defer_prioritizes_oldest_but_preserves_migration_fence():
    s = scheduler()
    states = {sid: dict(lifecycle_state="SERVING", weight_version=0) for sid in ("a", "b")}
    metas = {sid: metadata(0) for sid in states}
    s.pending_since.update(a=-21.0, b=-20.0)
    assert s.select_update(3, states, metas, {}) == {"server_id": "a", "reason": "maximum deferral reached"}
    pending = {"x": dict(source_server_id="a", target_server_id="b")}
    assert s.select_update(3, states, metas, pending)["server_id"] is None


def test_legacy_mode_still_requires_kl_gate():
    s = scheduler(mode="version_kl")
    replicas = [snapshot("a", 0), snapshot("b", 2)]
    assert s.target("x", "a", metadata(0), replicas, False)[0] is None
    s.trajectories["x"].staleness = dict(source_version=0, target_version=2, kl=0.2)
    assert s.target("x", "a", metadata(0), replicas, False)[0] == "b"
