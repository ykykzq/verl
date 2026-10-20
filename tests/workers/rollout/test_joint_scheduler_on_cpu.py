# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import asyncio
from types import SimpleNamespace

import pytest

from verl.workers.rollout.router import GlobalRequestLoadBalancer


def metadata(version=0, fingerprint="layout"):
    return dict(
        model_id="model",
        weight_version=version,
        kv_transfer_backends=["remote_prefix"],
        kv_transfer_domain="domain",
        kv_transfer_namespace=f"cache-{version}",
        rollout_config_fingerprint=fingerprint,
    )


def router(**settings):
    config = dict(
        enabled=True,
        kv_capacity_tokens=1024,
        max_batch_size=8,
        kv_block_size_tokens=16,
        update_cooldown_s=5.0,
        min_decode_window_s=2.0,
        min_decode_tokens=4,
        max_defer_s=30.0,
        evacuation_timeout_s=5.0,
        sparse_request_threshold=1,
        staleness_version_gap=2,
        tail_tokens=2,
        tail_kl_threshold=0.1,
        probe_interval_s=5.0,
        probe_timeout_s=1.0,
        max_concurrent_probes=1,
        overload_threshold=0.9,
    )
    config.update(settings)
    instance = GlobalRequestLoadBalancer(
        {"a": None, "b": None},
        server_metadata={"a": metadata(), "b": metadata()},
        trajectory_migration_config={"enabled": True, "allow_cross_version_recompute": True},
        joint_scheduling_config=config,
    )
    clock = [100.0]
    instance._joint.clock = lambda: clock[0]
    return instance, clock


def assign(r, request, server, budget=64):
    r.register_trajectory(request, 16, budget - 16)
    r._request_id_to_server[request] = server
    assert r.acquire_server(request)[0] == server
    r.release_server(server)


async def progress(r, request, generated=8, version=0):
    rec = r._joint.trajectories[request]
    return await r.report_trajectory_progress(request, rec.server_id, generated, version, [1] * (16 + generated))


async def plan(r, request, required=False):
    rec = r._joint.trajectories[request]
    return await r.plan_trajectory_migration(
        request,
        rec.server_id,
        dict(
            request_id=request,
            prefix_tokens=16 + rec.generated_tokens,
            generated_tokens=rec.generated_tokens,
            migration_required=required,
        ),
        source_metadata={"weight_version": rec.weight_version},
    )


def test_publication_closes_new_admissions_but_not_checkpoint_continuations():
    r, _ = router()
    assign(r, "old", "a")
    r.prepare_weight_update(1)
    assert r.acquire_server("old")[0] == "a"
    r.register_trajectory("new", 16, 48)
    with pytest.raises(RuntimeError, match="No available servers"):
        r.acquire_server("new")
    r.begin_replica_update("b", 1)
    r.commit_replica_update("b", 1, metadata(1))
    assert r.acquire_server("new")[0] == "b"


def test_checkpoint_gaps_still_consume_capacity_and_cannot_be_evicted_by_lru():
    r, _ = router(kv_capacity_tokens=64, max_batch_size=1)
    assign(r, "one", "a")
    assign(r, "two", "b")
    r.clear_sticky_cache()
    r.register_trajectory("three", 16, 48)
    with pytest.raises(RuntimeError, match="No available servers"):
        r.acquire_server("three")
    assert r.acquire_server("one")[0] == "a"
    r.finish_trajectory("one")
    assert r.acquire_server("three")[0] == "a"


def test_pending_and_committed_transfers_reserve_capacity_until_completion():
    async def run():
        r, _ = router(kv_capacity_tokens=128, max_batch_size=2)
        assign(r, "x", "a")
        assign(r, "y", "a")
        assign(r, "z", "b")
        await progress(r, "x")
        await progress(r, "y")
        r.prepare_weight_update(1)
        r._joint.evacuating["a"] = 100.0
        first = await plan(r, "x")
        assert first["migrate"] and first["target_server_id"] == "b"
        assert not (await plan(r, "y"))["migrate"]
        r.commit_trajectory_migration("x", first["decision_id"])
        assert not (await plan(r, "y"))["migrate"]
        r.finish_trajectory("x")
        assert (await plan(r, "y"))["migrate"]

    asyncio.run(run())


def test_cancelled_migration_keeps_source_and_releases_target_capacity():
    async def run():
        r, _ = router()
        assign(r, "x", "a")
        await progress(r, "x")
        r.prepare_weight_update(1)
        r._joint.evacuating["a"] = 100.0
        decision = await plan(r, "x")
        r.cancel_trajectory_migration("x", decision["decision_id"])
        assert r._joint.trajectories["x"].server_id == "a"
        assert not r._joint.reservations
        assert r.acquire_server("x")[0] == "a"

    asyncio.run(run())


def test_low_occupancy_replica_evacuates_then_updates_before_busy_replica():
    async def run():
        r, _ = router()
        for name, server in [("x", "a"), ("y", "b"), ("z", "b")]:
            assign(r, name, server)
            await progress(r, name)
        r.prepare_weight_update(1)
        assert r.select_replica_for_update(1)["server_id"] is None
        assert "a" in r._joint.evacuating
        decision = await plan(r, "x")
        assert decision["mode"] == "remote_prefix"
        r.commit_trajectory_migration("x", decision["decision_id"])
        assert r.select_replica_for_update(1)["server_id"] == "a"

    asyncio.run(run())


def test_publications_do_not_reset_deferral_deadline_and_updates_honor_cooldown():
    r, clock = router(kv_capacity_tokens=64, max_batch_size=1)
    assign(r, "x", "a")
    assign(r, "y", "b")
    r.prepare_weight_update(1)
    clock[0] += 31
    r.prepare_weight_update(2)
    assert r.select_replica_for_update(2)["server_id"] == "a"
    r.begin_replica_update("a", 2)
    r.commit_replica_update("a", 2, metadata(2))
    r.prepare_weight_update(3)
    assert r.select_replica_for_update(3)["server_id"] != "a"


def test_migration_window_and_incompatible_layout_prevent_movement():
    async def run():
        r, clock = router()
        assign(r, "x", "a")
        await progress(r, "x")
        r.prepare_weight_update(1)
        r._joint.evacuating["a"] = clock[0]
        r.update_server_metadata("b", metadata(fingerprint="other"))
        assert not (await plan(r, "x"))["migrate"]
        r.update_server_metadata("b", metadata())
        decision = await plan(r, "x")
        r.commit_trajectory_migration("x", decision["decision_id"])
        r._joint.evacuating = {"b": clock[0]}
        assert not (await plan(r, "x"))["migrate"]
        clock[0] += 3
        assert not (await plan(r, "x"))["migrate"]
        await progress(r, "x", generated=12)
        assert (await plan(r, "x"))["migrate"]

    asyncio.run(run())


def test_update_survivor_recomputes_in_place_never_downgrades():
    async def run():
        r, clock = router()
        assign(r, "x", "a")
        await progress(r, "x")
        r.prepare_weight_update(2)
        r.begin_replica_update("a", 2)
        r.commit_replica_update("a", 2, metadata(2))
        decision = await plan(r, "x", required=True)
        assert decision["target_server_id"] == "a"
        assert decision["mode"] == "recompute"
        r.commit_trajectory_migration("x", decision["decision_id"])
        await progress(r, "x", generated=16, version=2)
        clock[0] += 10
        assert not (await plan(r, "x"))["migrate"]

    asyncio.run(run())


def test_unregistered_requests_cannot_bypass_joint_admission():
    r, _ = router()
    with pytest.raises(ValueError, match="register"):
        r.acquire_server("unknown")
    with pytest.raises(ValueError, match="capacity"):
        r.register_trajectory("large", 1000, 1000)


def test_tail_probe_is_nonblocking_and_high_kl_moves_to_new_version():
    async def run():
        r, clock = router()
        assign(r, "x", "a")
        r.prepare_weight_update(2)
        r.begin_replica_update("b", 2)
        r.commit_replica_update("b", 2, metadata(2))
        gate = asyncio.Event()

        async def score(prefix_ids, tail_tokens, expected_weight_version):
            await gate.wait()
            return dict(
                weight_version=expected_weight_version,
                log_probs=[-1.0 if expected_weight_version == 0 else -4.0] * tail_tokens,
            )

        for sid in ("a", "b"):
            r._servers[sid] = SimpleNamespace(score_trajectory_tail=SimpleNamespace(remote=score))
        await asyncio.wait_for(progress(r, "x"), 0.1)
        assert len(r._joint.probes) == 1
        assert not (await plan(r, "x"))["migrate"]
        gate.set()
        await asyncio.gather(*r._joint.probes.values())
        decision = await plan(r, "x")
        assert decision["migrate"] and decision["target_server_id"] == "b"
        assert decision["mode"] == "recompute"
        r.finish_trajectory("x")

    asyncio.run(run())


def test_selected_update_replica_cannot_receive_a_migration_before_begin_rpc():
    async def run():
        r, _ = router()
        assign(r, "x", "a")
        await progress(r, "x")
        r.prepare_weight_update(1)
        assert r.select_replica_for_update(1)["server_id"] == "b"
        r._joint.evacuating["a"] = 100.0
        assert not (await plan(r, "x"))["migrate"]

    asyncio.run(run())


@pytest.mark.parametrize("removed", [False, True])
def test_unavailable_source_recovers_on_same_version_without_waiting_for_training(removed):
    async def run():
        r, _ = router()
        assign(r, "x", "a")
        await progress(r, "x")
        if removed:
            r.remove_servers(["a"])
        else:
            r.begin_replica_update("a", 1)
            r.fail_replica_update("a", "failed")
        decision = await plan(r, "x", required=True)
        assert decision["migrate"]
        assert decision["target_server_id"] == "b"
        assert decision["mode"] == "recompute"

    asyncio.run(run())


def test_token_floor_cannot_override_maximum_deferral_forever():
    async def run():
        r, clock = router()
        assign(r, "x", "a")
        await progress(r, "x")
        r.prepare_weight_update(1)
        r._joint.evacuating["a"] = clock[0]
        decision = await plan(r, "x")
        r.commit_trajectory_migration("x", decision["decision_id"])
        r.begin_replica_update("a", 1)
        r.commit_replica_update("a", 1, metadata(1))
        clock[0] += 1000
        assert r.select_replica_for_update(1)["server_id"] == "b"

    asyncio.run(run())


def test_finished_trajectory_cannot_leak_a_probe_cancelled_before_first_execution():
    async def run():
        r, _ = router()
        assign(r, "x", "a")
        r.prepare_weight_update(2)
        r.begin_replica_update("b", 2)
        r.commit_replica_update("b", 2, metadata(2))
        await progress(r, "x")
        r.finish_trajectory("x")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not r._joint.probes
        assert not r._joint.probe_resources

    asyncio.run(run())


def test_failed_probe_retains_both_reservations_until_other_rpc_settles():
    async def run():
        r, _ = router(probe_timeout_s=0.01)
        assign(r, "x", "a")
        r.prepare_weight_update(2)
        r.begin_replica_update("b", 2)
        r.commit_replica_update("b", 2, metadata(2))
        started, finish = asyncio.Event(), asyncio.Event()

        async def score(prefix_ids, tail_tokens, expected_weight_version):
            if expected_weight_version == 0:
                raise RuntimeError("source failed")
            started.set()
            await finish.wait()
            return dict(weight_version=2, log_probs=[-2.0] * tail_tokens)

        for sid in ("a", "b"):
            r._servers[sid] = SimpleNamespace(score_trajectory_tail=SimpleNamespace(remote=score))
        await progress(r, "x")
        await started.wait()
        await asyncio.sleep(0.02)
        assert r._joint.probe_resources
        r.finish_trajectory("x")
        assert r._joint.probe_resources
        finish.set()
        await asyncio.gather(*r._joint.probes.values(), return_exceptions=True)
        assert not r._joint.probe_resources

    asyncio.run(run())


def test_actual_backend_block_size_must_fit_configured_accounting():
    r, _ = router()
    with pytest.raises(ValueError, match="block accounting"):
        r.update_server_metadata("a", {**metadata(), "kv_block_size_tokens": 1024})


def test_actual_engine_batch_limit_restricts_admission_and_probe_slots():
    r, _ = router()
    for sid in ("a", "b"):
        r.update_server_metadata(sid, {**metadata(), "max_batch_size": 1})
    assign(r, "x", "a")
    assign(r, "y", "b")
    r.register_trajectory("z", 16, 48)
    with pytest.raises(RuntimeError, match="No available servers"):
        r.acquire_server("z")


def test_oversized_sequence_is_rejected_instead_of_waiting_forever():
    r, _ = router()
    for sid in ("a", "b"):
        r.update_server_metadata(sid, {**metadata(), "max_sequence_tokens": 32})
    with pytest.raises(ValueError, match="sequence"):
        r.register_trajectory("x", 16, 64)


def test_overload_relief_does_not_change_version_or_bounce_back():
    async def run():
        r, _ = router(kv_capacity_tokens=256, max_batch_size=4)
        for i in range(4):
            assign(r, str(i), "a")
            await progress(r, str(i))
        d = await plan(r, "0")
        assert d["target_server_id"] == "b" and d["mode"] == "remote_prefix"
        r.commit_trajectory_migration("0", d["decision_id"])
        assert not (await plan(r, "0"))["migrate"]

    asyncio.run(run())


def test_overload_relief_uses_effective_engine_batch_limit():
    async def run():
        r, _ = router(max_batch_size=8)
        r.update_server_metadata("a", {**metadata(), "max_batch_size": 3})
        for i in range(3):
            assign(r, str(i), "a")
            await progress(r, str(i))
        decision = await plan(r, "0")
        assert decision["migrate"] and decision["target_server_id"] == "b"
        assert decision["mode"] == "remote_prefix"

    asyncio.run(run())


def test_probe_version_change_invalidates_result():
    async def run():
        r, _ = router()
        assign(r, "x", "a")
        r.prepare_weight_update(2)
        r.begin_replica_update("b", 2)
        r.commit_replica_update("b", 2, metadata(2))
        entered, release = asyncio.Event(), asyncio.Event()

        async def score(prefix_ids, tail_tokens, expected_weight_version):
            entered.set()
            await release.wait()
            return dict(
                weight_version=expected_weight_version,
                log_probs=[-1.0 if expected_weight_version == 0 else -5.0] * tail_tokens,
            )

        for sid in ("a", "b"):
            r._servers[sid] = SimpleNamespace(score_trajectory_tail=SimpleNamespace(remote=score))
        await progress(r, "x")
        await entered.wait()
        r.begin_replica_update("b", 3)
        r.commit_replica_update("b", 3, metadata(3))
        release.set()
        await asyncio.gather(*r._joint.probes.values())
        assert not r._joint.trajectories["x"].staleness

    asyncio.run(run())
