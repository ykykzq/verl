# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import asyncio
import gc
import weakref
from dataclasses import dataclass

import pytest

from verl.checkpoint_engine.weight_scheduler import WeightPublicationScheduler


@dataclass(frozen=True)
class Snapshot:
    version: int
    buckets: tuple = ()
    tensor_count: int = 0
    total_bytes: int = 0


class Manager:
    """Model observable replica state, with deterministic transfer barriers."""

    def __init__(self, versions=(0, 0)):
        self.states = {
            rank: {"lifecycle_state": "SERVING", "weight_version": version, "pin": None, "last_update_error": None}
            for rank, version in enumerate(versions)
        }
        self.started = asyncio.Queue()
        self.release = asyncio.Event()
        self.release.set()
        self.failed_ranks = set()
        self.fatal = False
        self.cancelled = asyncio.Event()
        self.transfers = []
        self.active = 0

    async def get_replica_states(self):
        return {rank: dict(state) for rank, state in self.states.items()}

    async def pin_replica(self, rank, reason):
        assert not self.active
        self.states[rank]["pin"] = {"reason": reason}
        return dict(self.states[rank])

    async def unpin_replica(self, rank):
        assert not self.active
        self.states[rank]["pin"] = None
        return dict(self.states[rank])

    async def update_replica_from_snapshot(self, snapshot, replica_rank, recovery=False):
        assert not self.active
        state = self.states[replica_rank]
        if recovery and state["lifecycle_state"] != "QUARANTINED":
            raise RuntimeError("recovery requires QUARANTINED")
        self.active += 1
        assert recovery or (state["lifecycle_state"] == "SERVING" and not state["pin"])
        state["lifecycle_state"] = "UPDATING"
        self.transfers.append((replica_rank, snapshot.version, recovery))
        await self.started.put(self.transfers[-1])
        try:
            await self.release.wait()
            if replica_rank in self.failed_ranks:
                state.update(lifecycle_state="QUARANTINED", last_update_error="install failed")
                if self.fatal:
                    raise RuntimeError("fatal install failure")
                status = "quarantined"
            else:
                state.update(lifecycle_state="SERVING", weight_version=snapshot.version, last_update_error=None)
                status = "updated"
            entry = {"status": status, "version": state["weight_version"], "attempts": 1}
            return {
                "replicas": {replica_rank: entry},
                "updated": int(status == "updated"),
                "pinned": 0,
                "quarantined": int(status == "quarantined"),
                "retried": 0,
            }
        except asyncio.CancelledError:
            state.update(lifecycle_state="QUARANTINED", last_update_error="cancelled")
            self.cancelled.set()
            raise
        finally:
            self.active -= 1


async def next_transfer(manager):
    return await asyncio.wait_for(manager.started.get(), 1)


@pytest.mark.asyncio
async def test_publish_returns_during_transfer_and_coalesces_pending_versions():
    manager = Manager()
    manager.release.clear()
    commits = []

    async def on_commit(version, result):
        commits.append((version, result["updated"]))

    scheduler = WeightPublicationScheduler(manager, on_commit=on_commit)
    first, intermediate, latest = Snapshot(1), Snapshot(2), Snapshot(3)
    intermediate_ref = weakref.ref(intermediate)
    try:
        await scheduler.publish(first)
        assert await next_transfer(manager) == (0, 1, False)
        await asyncio.wait_for(scheduler.publish(intermediate), 0.2)
        await asyncio.wait_for(scheduler.publish(latest), 0.2)
        del intermediate
        gc.collect()
        assert intermediate_ref() is None
        assert commits == []
        status = await scheduler.get_status()
        assert (status["latest_version"], status["active_version"]) == (3, 1)
        manager.release.set()
        await scheduler.wait_for_version(3, timeout_s=1)
        assert manager.transfers == [(0, 1, False), (1, 3, False), (0, 3, False)]
        assert commits == [(1, 1), (3, 1)]
    finally:
        manager.release.set()
        await scheduler.close()


@pytest.mark.asyncio
async def test_unpin_catches_up_without_another_publication():
    manager = Manager()
    scheduler = WeightPublicationScheduler(manager)
    try:
        await scheduler.pin_replica(0, "reference")
        await scheduler.publish(Snapshot(4))
        await scheduler.wait_for_version(4, timeout_s=1)
        assert [state["weight_version"] for state in manager.states.values()] == [0, 4]
        await scheduler.unpin_replica(0)
        await scheduler.wait_for_version(4, timeout_s=1)
        assert [state["weight_version"] for state in manager.states.values()] == [4, 4]
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_quarantine_isolated_and_explicit_recovery_uses_retained_latest():
    manager = Manager()
    manager.failed_ranks.add(0)
    scheduler = WeightPublicationScheduler(manager)
    try:
        await scheduler.publish(Snapshot(5))
        await scheduler.wait_for_version(5, timeout_s=1)
        assert manager.states[0]["lifecycle_state"] == "QUARANTINED"
        await scheduler.publish(Snapshot(6))
        await scheduler.wait_for_version(6, timeout_s=1)
        manager.failed_ranks.clear()
        result = await scheduler.recover_replica(0)
        assert result["updated"] == 1
        assert manager.transfers == [(0, 5, False), (1, 5, False), (1, 6, False), (0, 6, True)]
        await scheduler.wait_for_version(6, timeout_s=1)
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_barrier_waits_for_serving_commit_and_times_out_without_cancelling_transfer():
    manager = Manager((0,))
    manager.release.clear()
    scheduler = WeightPublicationScheduler(manager)
    try:
        await scheduler.publish(Snapshot(1))
        await next_transfer(manager)
        with pytest.raises(TimeoutError):
            await scheduler.wait_for_version(1, timeout_s=0.01)
        assert manager.active == 1
        manager.release.set()
        await scheduler.wait_for_version(1, timeout_s=1)
    finally:
        manager.release.set()
        await scheduler.close()


@pytest.mark.asyncio
async def test_all_quarantined_barrier_fails_actionably_instead_of_waiting_for_timeout():
    manager = Manager((0,))
    manager.failed_ranks.add(0)
    scheduler = WeightPublicationScheduler(manager)
    try:
        await scheduler.publish(Snapshot(1))
        with pytest.raises(RuntimeError, match="(?i)(serving|quarantin|unavailable)"):
            await scheduler.wait_for_version(1, timeout_s=1)
    finally:
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_failure", [False, True])
async def test_background_fatal_errors_are_observed_by_publish_status_and_wait(callback_failure):
    manager = Manager((0,))
    if not callback_failure:
        manager.failed_ranks.add(0)
        manager.fatal = True

    async def on_commit(version, result):
        raise RuntimeError("commit callback failed")

    scheduler = WeightPublicationScheduler(manager, on_commit=on_commit if callback_failure else None)
    await scheduler.publish(Snapshot(1))
    try:
        with pytest.raises(RuntimeError, match="(fatal install|commit callback)"):
            await scheduler.wait_for_version(1, timeout_s=1)
        with pytest.raises(RuntimeError, match="(fatal install|commit callback)"):
            await scheduler.publish(Snapshot(2))
        status = await scheduler.get_status()
        assert ("fatal install" if not callback_failure else "commit callback") in status["fatal_error"]
    finally:
        with pytest.raises(RuntimeError, match="(fatal install|commit callback)"):
            await scheduler.close()


@pytest.mark.asyncio
async def test_cancel_close_awaits_transfer_cleanup_and_releases_snapshots():
    manager = Manager((0,))
    manager.release.clear()
    scheduler = WeightPublicationScheduler(manager)
    snapshot = Snapshot(1)
    snapshot_ref = weakref.ref(snapshot)
    await scheduler.publish(snapshot)
    await next_transfer(manager)
    del snapshot
    await scheduler.close(drain=False)
    gc.collect()
    assert manager.cancelled.is_set()
    assert manager.active == 0
    assert snapshot_ref() is None
    with pytest.raises(RuntimeError, match="closed"):
        await scheduler.publish(Snapshot(2))


@pytest.mark.asyncio
async def test_drain_close_finishes_transfer_and_rejects_new_work():
    manager = Manager((0,))
    manager.release.clear()
    scheduler = WeightPublicationScheduler(manager)
    await scheduler.publish(Snapshot(1))
    await next_transfer(manager)
    closing = asyncio.create_task(scheduler.close())
    await asyncio.sleep(0)
    assert not closing.done()
    with pytest.raises(RuntimeError, match="closed|closing"):
        await scheduler.publish(Snapshot(2))
    manager.release.set()
    await asyncio.wait_for(closing, 1)
    assert manager.states[0]["weight_version"] == 1


@pytest.mark.asyncio
async def test_publication_versions_are_nonnegative_and_strictly_monotonic():
    manager = Manager((0,))
    scheduler = WeightPublicationScheduler(manager)
    try:
        for version in (-1, True, 1.5):
            with pytest.raises(ValueError):
                await scheduler.publish(Snapshot(version))
        await scheduler.publish(Snapshot(2))
        for snapshot in (Snapshot(1), Snapshot(2), Snapshot(2, buckets=("different",))):
            with pytest.raises(ValueError):
                await scheduler.publish(snapshot)
        await scheduler.wait_for_version(2, timeout_s=1)
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_cancelling_close_settles_an_active_recovery_rpc():
    manager = Manager((0,))
    scheduler = WeightPublicationScheduler(manager)
    await scheduler.publish(Snapshot(1))
    await scheduler.wait_for_version(1, timeout_s=1)
    await next_transfer(manager)
    manager.states[0]["lifecycle_state"] = "QUARANTINED"
    manager.release.clear()
    recovery = asyncio.create_task(scheduler.recover_replica(0))
    assert await next_transfer(manager) == (0, 1, True)
    await scheduler.close(drain=False)
    with pytest.raises(RuntimeError, match="closed|cancelled"):
        await asyncio.wait_for(recovery, 0.1)


@pytest.mark.asyncio
async def test_startup_accepts_the_rtp_initial_version_sentinel():
    manager = Manager(("initial",))
    scheduler = WeightPublicationScheduler(manager)
    try:
        await scheduler.publish(Snapshot(0))
        await scheduler.wait_for_version(0, timeout_s=1)
        assert manager.states[0]["weight_version"] == 0
        assert (await scheduler.get_status())["fatal_error"] is None
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_close_before_worker_starts_settles_queued_operations():
    manager = Manager((0,))
    scheduler = WeightPublicationScheduler(manager)
    pin = asyncio.create_task(scheduler.pin_replica(0, "reference"))
    closing = asyncio.create_task(scheduler.close(drain=False))
    await asyncio.wait_for(closing, 1)
    with pytest.raises(RuntimeError, match="closed"):
        await asyncio.wait_for(pin, 0.1)


@pytest.mark.asyncio
async def test_pin_serializes_after_active_commit_and_prevents_future_updates():
    manager = Manager()
    manager.release.clear()
    scheduler = WeightPublicationScheduler(manager)
    try:
        await scheduler.publish(Snapshot(1))
        await next_transfer(manager)
        pin = asyncio.create_task(scheduler.pin_replica(0, "reference"))
        await asyncio.sleep(0)
        assert not pin.done()
        await scheduler.publish(Snapshot(2))
        manager.release.set()
        await asyncio.wait_for(pin, 1)
        await scheduler.wait_for_version(2, timeout_s=1)
        assert manager.states[0]["weight_version"] == 1
        assert manager.states[1]["weight_version"] == 2
        assert manager.transfers == [(0, 1, False), (1, 2, False)]
    finally:
        manager.release.set()
        await scheduler.close()


@pytest.mark.asyncio
async def test_barrier_waits_for_commit_callback_and_recovery_does_not_repeat_it():
    manager = Manager((0,))
    entered = asyncio.Event()
    release = asyncio.Event()
    commits = []

    async def on_commit(version, result):
        entered.set()
        await release.wait()
        commits.append(version)

    scheduler = WeightPublicationScheduler(manager, on_commit=on_commit)
    try:
        await scheduler.publish(Snapshot(1))
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(TimeoutError):
            await scheduler.wait_for_version(1, timeout_s=0.01)
        release.set()
        await scheduler.wait_for_version(1, timeout_s=1)
        manager.states[0]["lifecycle_state"] = "QUARANTINED"
        await scheduler.recover_replica(0)
        assert commits == [1]
    finally:
        release.set()
        await scheduler.close()


@pytest.mark.asyncio
async def test_all_pinned_old_replicas_require_unpin_before_publication_can_serve():
    manager = Manager((0,))
    scheduler = WeightPublicationScheduler(manager)
    try:
        await scheduler.pin_replica(0, "reference")
        await scheduler.publish(Snapshot(1))
        with pytest.raises(RuntimeError, match="unpin"):
            await scheduler.wait_for_version(1, timeout_s=1)
        await scheduler.unpin_replica(0)
        await scheduler.wait_for_version(1, timeout_s=1)
    finally:
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pinned", [False, True])
async def test_rejected_recovery_of_serving_replica_does_not_poison_publication(pinned):
    manager = Manager()
    scheduler = WeightPublicationScheduler(manager)
    try:
        await scheduler.publish(Snapshot(1))
        await scheduler.wait_for_version(1, timeout_s=1)
        if pinned:
            await scheduler.pin_replica(0, "reference")
        with pytest.raises(RuntimeError, match="QUARANTINED"):
            await scheduler.recover_replica(0)
        assert (await scheduler.get_status())["fatal_error"] is None
        await scheduler.publish(Snapshot(2))
        await scheduler.wait_for_version(2, timeout_s=1)
        assert manager.states[1]["weight_version"] == 2
        assert manager.states[0]["weight_version"] == (1 if pinned else 2)
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_genuine_recovery_transfer_failure_remains_fatal():
    manager = Manager()
    manager.failed_ranks.add(0)
    scheduler = WeightPublicationScheduler(manager)
    await scheduler.publish(Snapshot(1))
    await scheduler.wait_for_version(1, timeout_s=1)
    manager.fatal = True
    try:
        with pytest.raises(RuntimeError, match="fatal install"):
            await scheduler.recover_replica(0)
        assert "fatal install" in (await scheduler.get_status())["fatal_error"]
        with pytest.raises(RuntimeError, match="fatal install"):
            await scheduler.publish(Snapshot(2))
    finally:
        with pytest.raises(RuntimeError, match="fatal install"):
            await scheduler.close()
