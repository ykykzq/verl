# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest

from tests.workers.rollout.test_joint_scheduler_on_cpu import assign, metadata, plan, progress, router


def server(usage):
    async def snapshot():
        return usage

    return SimpleNamespace(get_laminar_kv_usage=SimpleNamespace(remote=snapshot))


@pytest.mark.asyncio
@pytest.mark.parametrize("occupancy,repack", [(0.2, True), (0.51, False)])
async def test_laminar_fences_new_rollouts_and_preserves_existing_versions(occupancy, repack):
    r, _ = router(policy="laminar")

    r._servers.update(a=server(occupancy), b=server(0.6))
    assign(r, "old-a", "a")
    assign(r, "old-b", "b")
    await progress(r, "old-a")

    r.prepare_weight_update(1)
    r.register_trajectory("new", 16, 48)
    with pytest.raises(RuntimeError, match="No available servers"):
        r.acquire_server("new")

    decision = await r.select_laminar_replica_for_update(1)
    assert decision["server_id"] is None
    migration = await plan(r, "old-a")
    if repack:
        assert migration["target_server_id"] == "b" and migration["mode"] == "remote_prefix"
        r.commit_trajectory_migration("old-a", migration["decision_id"])
    else:
        assert not migration["migrate"]
        assert r.acquire_server("old-a")[0] == "a"
        r.release_server("a")
        r.finish_trajectory("old-a")
    decision = await r.select_laminar_replica_for_update(1)
    assert decision["server_id"] == "a"
    r.begin_replica_update("a", 1)
    r.commit_replica_update("a", 1, metadata(1))
    assert r.acquire_server("new")[0] == "a"
    assert r.acquire_server("old-b")[0] == "b"


@pytest.mark.asyncio
async def test_laminar_best_fit_accounts_for_all_sources_assigned_to_a_destination():
    r, _ = router(policy="laminar")
    r._servers.update(a=server(0.1), b=server(0.61))
    for sid, utilization in (("c", 0.16), ("d", 0.55)):
        r.add_servers({sid: server(utilization)})
        r.update_server_metadata(sid, metadata())
    for sid in "abcd":
        assign(r, f"on-{sid}", sid)
    r.prepare_weight_update(1)
    result = await r.select_laminar_replica_for_update(1)
    assert result["sources"] == {"a": "b", "c": "b"}
    assert set(r._joint.evacuating) == {"a", "c"}


@pytest.mark.asyncio
async def test_laminar_installs_independent_idle_replica_while_repack_drains():
    r, _ = router(policy="laminar")
    r._servers.update(a=server(0.1), b=server(0.7))
    r.add_servers({"c": server(0.7)})
    r.update_server_metadata("c", metadata())
    for sid in "abc":
        assign(r, f"on-{sid}", sid)

    r.prepare_weight_update(1)
    decision = await r.select_laminar_replica_for_update(1)
    assert decision["sources"] == {"a": "b"}

    r.finish_trajectory("on-b")
    r.finish_trajectory("on-c")
    decision = await r.select_laminar_replica_for_update(1)
    assert decision["server_id"] == "c"
    r.begin_replica_update("c", 1)
    r.commit_replica_update("c", 1, metadata(1))
    r.register_trajectory("new", 16, 48)
    assert r.acquire_server("new")[0] == "c"
    assert r._joint.repack_targets == {"a": "b"}


@pytest.mark.asyncio
async def test_laminar_releases_destination_when_source_install_fails():
    r, _ = router(policy="laminar")
    r._servers.update(a=server(0.1), b=server(0.7))
    assign(r, "on-a", "a")
    assign(r, "on-b", "b")
    r.prepare_weight_update(1)
    decision = await r.select_laminar_replica_for_update(1)
    assert decision["sources"] == {"a": "b"}

    r.finish_trajectory("on-a")
    r.finish_trajectory("on-b")
    decision = await r.select_laminar_replica_for_update(1)
    assert decision["server_id"] == "a"
    r.begin_replica_update("a", 1)
    r.fail_replica_update("a", "test install failure")

    decision = await r.select_laminar_replica_for_update(1)
    assert decision["server_id"] == "b"
