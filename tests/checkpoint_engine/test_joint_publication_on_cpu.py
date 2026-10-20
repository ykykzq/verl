# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine.weight_publication import WeightPublisher
from verl.checkpoint_engine.weight_scheduler import WeightPublicationScheduler


class RemoteMethod:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class JointManager:
    def __init__(self):
        self.states = {rank: {"lifecycle_state": "SERVING", "weight_version": 0, "pin": None} for rank in (3, 7)}
        self.replicas = [SimpleNamespace(replica_rank=rank, server_address=f"server-{rank}") for rank in self.states]
        self.prepared = []
        self.selections = []
        self.deferred = True
        self.transfers = []
        self.release = asyncio.Event()
        self.release.set()
        self.started = asyncio.Queue()
        self.load_balancer = SimpleNamespace(
            prepare_weight_update=RemoteMethod(self.prepare),
            select_replica_for_update=RemoteMethod(self.select),
        )

    def prepare(self, version):
        self.prepared.append(version)
        return {"version": version}

    def select(self, version):
        assert version in self.prepared
        self.selections.append(version)
        candidates = [
            rank
            for rank, state in self.states.items()
            if state["lifecycle_state"] == "SERVING" and not state["pin"] and state["weight_version"] < version
        ]
        server_id = None if self.deferred or not candidates else f"server-{max(candidates)}"
        return {"server_id": server_id, "reason": "cooldown" if self.deferred else "empty"}

    async def get_replica_states(self):
        return {rank: dict(state) for rank, state in self.states.items()}

    async def update_replica_from_snapshot(self, snapshot, rank, recovery=False):
        self.states[rank]["lifecycle_state"] = "UPDATING"
        self.transfers.append((rank, snapshot.version))
        await self.started.put(self.transfers[-1])
        await self.release.wait()
        self.states[rank].update(lifecycle_state="SERVING", weight_version=snapshot.version)
        return {"replicas": {rank: {"status": "updated"}}}

    async def pin_replica(self, rank, reason):
        self.states[rank]["pin"] = {"reason": reason}

    async def unpin_replica(self, rank):
        self.states[rank]["pin"] = None


def scheduler_for(manager):
    return WeightPublicationScheduler(manager, joint_scheduling_config={"enabled": True, "poll_interval_s": 0.005})


@pytest.mark.asyncio
async def test_publication_prepares_all_replicas_without_waiting_for_drain():
    manager = JointManager()
    scheduler = scheduler_for(manager)
    try:
        result = await asyncio.wait_for(scheduler.publish(SimpleNamespace(version=1)), 0.1)
        assert result["accepted_version"] == 1
        assert manager.prepared == [1]
        with pytest.raises(asyncio.TimeoutError):
            await scheduler.wait_for_version(1, timeout_s=0.02)
        assert not manager.transfers
        assert len(manager.selections) < 20
        manager.deferred = False
        await scheduler.wait_for_version(1, timeout_s=1)
        assert manager.transfers == [(7, 1), (3, 1)]
    finally:
        await scheduler.close(drain=False)


@pytest.mark.asyncio
async def test_new_publication_coalesces_while_replicas_are_deferred():
    manager = JointManager()
    scheduler = scheduler_for(manager)
    try:
        await scheduler.publish(SimpleNamespace(version=1))
        await scheduler.publish(SimpleNamespace(version=2))
        await scheduler.publish(SimpleNamespace(version=4))
        manager.deferred = False
        await scheduler.wait_for_version(4, timeout_s=1)
        assert manager.prepared == [1, 2, 4]
        assert manager.transfers == [(7, 4), (3, 4)]
    finally:
        await scheduler.close(drain=False)


@pytest.mark.asyncio
async def test_new_publication_is_prepared_during_active_transfer():
    manager = JointManager()
    manager.deferred = False
    manager.release.clear()
    scheduler = scheduler_for(manager)
    try:
        await scheduler.publish(SimpleNamespace(version=1))
        assert await asyncio.wait_for(manager.started.get(), 1) == (7, 1)
        await asyncio.wait_for(scheduler.publish(SimpleNamespace(version=2)), 0.1)
        assert manager.prepared == [1, 2]
        manager.release.set()
        await scheduler.wait_for_version(2, timeout_s=1)
        assert manager.transfers == [(7, 1), (7, 2), (3, 2)]
    finally:
        manager.release.set()
        await scheduler.close(drain=False)


@pytest.mark.asyncio
async def test_draining_close_waits_for_deferred_replica_and_cancel_close_does_not():
    manager = JointManager()
    scheduler = scheduler_for(manager)
    await scheduler.publish(SimpleNamespace(version=1))
    close_task = asyncio.create_task(scheduler.close())
    try:
        await asyncio.sleep(0.02)
        assert not close_task.done()
        manager.deferred = False
        await asyncio.wait_for(close_task, 1)
        assert manager.transfers == [(7, 1), (3, 1)]
    finally:
        manager.deferred = False
        await close_task

    other = scheduler_for(JointManager())
    await other.publish(SimpleNamespace(version=2))
    await asyncio.wait_for(other.close(drain=False), 0.1)
    assert (await other.get_status())["closed"]


@pytest.mark.asyncio
async def test_selection_failure_surfaces_as_publication_health_failure():
    manager = JointManager()

    def invalid_selection(version):
        return {"server_id": "not-a-replica"}

    manager.load_balancer.select_replica_for_update = RemoteMethod(invalid_selection)
    scheduler = scheduler_for(manager)
    await scheduler.publish(SimpleNamespace(version=1))
    with pytest.raises(RuntimeError, match="unknown.*replica"):
        await scheduler.wait_for_version(1, timeout_s=1)
    with pytest.raises(RuntimeError, match="unknown.*replica"):
        await scheduler.close(drain=False)


@pytest.mark.asyncio
async def test_cancelled_publication_settles_preparation_and_keeps_snapshot_available():
    manager = JointManager()
    manager.deferred = False
    prepared = asyncio.Event()
    release_prepare = asyncio.Event()

    async def prepare(version):
        manager.prepare(version)
        prepared.set()
        await release_prepare.wait()

    manager.load_balancer.prepare_weight_update = SimpleNamespace(remote=prepare)
    scheduler = scheduler_for(manager)
    publishing = asyncio.create_task(scheduler.publish(SimpleNamespace(version=2)))
    await prepared.wait()
    publishing.cancel()
    await asyncio.sleep(0)
    assert not publishing.done()
    release_prepare.set()
    with pytest.raises(asyncio.CancelledError):
        await publishing
    try:
        await scheduler.wait_for_version(2, timeout_s=1)
        assert manager.transfers == [(7, 2), (3, 2)]
    finally:
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_preparation_failure_cannot_leave_a_silently_healthy_scheduler(synchronous):
    manager = JointManager()

    def prepare(version):
        raise RuntimeError("router unavailable")

    manager.load_balancer.prepare_weight_update = (
        SimpleNamespace(remote=prepare) if synchronous else RemoteMethod(prepare)
    )
    scheduler = scheduler_for(manager)
    with pytest.raises(RuntimeError, match="router unavailable"):
        await scheduler.publish(SimpleNamespace(version=1))
    assert not manager.transfers
    assert (await scheduler.get_status())["fatal_error"] == "router unavailable"
    with pytest.raises(RuntimeError, match="router unavailable"):
        await scheduler.close(drain=False)


@pytest.mark.asyncio
async def test_pins_remain_excluded_and_unpin_wakes_deferred_publication():
    manager = JointManager()
    manager.deferred = False
    scheduler = scheduler_for(manager)
    try:
        await scheduler.pin_replica(7, "keep reference")
        await scheduler.publish(SimpleNamespace(version=2))
        await scheduler.wait_for_version(2, timeout_s=1)
        assert manager.transfers == [(3, 2)]
        await scheduler.unpin_replica(7)
        await scheduler.wait_for_version(2, timeout_s=1)
        assert manager.transfers == [(3, 2), (7, 2)]
    finally:
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("quarantined", [False, True])
async def test_joint_health_fails_when_only_pinned_stale_replicas_can_serve(quarantined):
    manager = JointManager()
    for state in manager.states.values():
        state["pin"] = {"reason": "keep old weights"}
    if quarantined:
        manager.states[7].update(lifecycle_state="QUARANTINED", pin=None)
    scheduler = scheduler_for(manager)
    publisher = WeightPublisher.__new__(WeightPublisher)
    publisher.scheduler = scheduler
    publisher.last_commit_metrics = {}
    try:
        await scheduler.publish(SimpleNamespace(version=1))

        async def wait_idle():
            while not (await scheduler.get_status())["idle"]:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_idle(), 1)
        with pytest.raises(RuntimeError, match="unpin.*recover"):
            await asyncio.wait_for(publisher.watch_health(), timeout=0.1)
        assert manager.transfers == []
    finally:
        await scheduler.close(drain=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["cooldown", "fresh", "updating", "legacy_pinned"])
async def test_health_preserves_deferred_progress_and_legacy_admission(scenario):
    manager = JointManager()
    if scenario != "cooldown":
        for state in manager.states.values():
            state["pin"] = {"reason": "reference"}
    if scenario == "fresh":
        manager.states[3]["weight_version"] = 1
    elif scenario == "updating":
        manager.states[3].update(lifecycle_state="UPDATING", pin=None)
    scheduler = WeightPublicationScheduler(manager) if scenario == "legacy_pinned" else scheduler_for(manager)
    publisher = WeightPublisher.__new__(WeightPublisher)
    publisher.scheduler = scheduler
    publisher.last_commit_metrics = {}
    health = None
    try:
        await scheduler.publish(SimpleNamespace(version=1))
        health = asyncio.create_task(publisher.watch_health())
        await asyncio.sleep(0.02)
        assert not health.done()
        if scenario == "cooldown":
            assert not (await scheduler.get_status())["idle"]
    finally:
        if health is not None:
            health.cancel()
            await asyncio.gather(health, return_exceptions=True)
        await scheduler.close(drain=False)
