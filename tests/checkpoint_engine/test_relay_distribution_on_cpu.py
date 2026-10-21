# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
from types import SimpleNamespace

import pytest
import ray

from verl.checkpoint_engine.weight_snapshot import WeightSnapshot


class Remote:
    def __init__(self, fn):
        self.remote = fn


class Relay:
    def __init__(self, node, events, blocked=None, fail=False):
        self.node, self.events, self.blocked, self.fail = node, events, blocked, fail
        self.live = set()
        self.failed_sources = set()
        self.dead_on_release = False
        self.load_bucket = Remote(self.load)
        self.pull_bucket = Remote(self.pull)
        self.release_bucket = Remote(self.release)
        self.close = Remote(self.finish)

    async def load(self, ref):
        bucket_id = f"{self.node}-{ref.hex()}"
        self.live.add(bucket_id)
        self.events.append(("load", self.node, bucket_id))
        return dict(bucket_id=bucket_id, node_id=self.node)

    async def pull(self, descriptor):
        self.events.append(("pull", self.node, descriptor["node_id"]))
        if self.blocked:
            await self.blocked.wait()
        if self.fail or descriptor["node_id"] in self.failed_sources:
            raise RuntimeError("RDMA read failed")
        bucket_id = f"{self.node}-{descriptor['bucket_id']}"
        self.live.add(bucket_id)
        return dict(descriptor=dict(bucket_id=bucket_id, node_id=self.node), ref=ray.ObjectRef.from_random())

    async def release(self, bucket_id):
        if self.dead_on_release:
            self.live.clear()
            raise ray.exceptions.ActorDiedError()
        self.live.remove(bucket_id)
        self.events.append(("release", self.node, bucket_id))

    async def finish(self):
        if self.dead_on_release:
            raise ray.exceptions.ActorDiedError()
        assert not self.live


def snapshot(version, buckets=2):
    return WeightSnapshot(version, tuple(ray.ObjectRef.from_random() for _ in range(buckets)), buckets, buckets * 8)


def distribution(monkeypatch, *, blocked=None, fail=False):
    from verl.checkpoint_engine import relay_distribution as module

    events = []
    relays = {
        "source": Relay("source", events),
        "a": Relay("a", events),
        "b": Relay("b", events, blocked=blocked, fail=fail),
        "c": Relay("c", events),
    }

    def worker(node):
        async def node_id():
            return node

        return SimpleNamespace(get_weight_relay_node_id=Remote(node_id))

    replicas = [SimpleNamespace(replica_rank=i, workers=[worker(n)]) for i, n in enumerate(("a", "b", "c"))]
    monkeypatch.setattr(module, "create_relay", lambda node_id, config: relays[node_id])
    monkeypatch.setattr(module, "destroy_relay", lambda handle: None, raising=False)
    obj = module.RelayDistribution(replicas, "source", dict(pipeline_depth=2, max_staging_bytes=1024, device_name=""))
    return obj, relays, events


def test_rejects_bucket_that_exceeds_concurrent_staging_budget(monkeypatch):
    dist, _, events = distribution(monkeypatch)
    # 1024 bytes / pipeline_depth 2: a 513-byte bucket cannot reserve a slot.
    oversized = WeightSnapshot(1, (ray.ObjectRef.from_random(),), 1, 513, 513)
    with pytest.raises(ValueError, match="staging"):
        dist.publish(oversized)
    assert not events


@pytest.mark.asyncio
async def test_fast_node_is_ready_before_downstream_and_source_stays_registered(monkeypatch):
    unblock = asyncio.Event()
    dist, relays, events = distribution(monkeypatch, blocked=unblock)
    s = snapshot(1)
    dist.publish(s)
    dist.pin(s)
    local = await asyncio.wait_for(dist.snapshots_for(0, 1), 2)
    assert local["a"].version == 1 and len(local["a"].buckets) == 2
    assert not relays["c"].live
    assert relays["source"].live
    closing = asyncio.create_task(dist.close())
    await asyncio.sleep(0)
    assert not closing.done() and relays["source"].live
    unblock.set()
    await closing
    assert all(not relay.live for relay in relays.values())
    assert len([e for e in events if e[0] == "load"]) == 2


@pytest.mark.asyncio
async def test_failed_middle_relay_does_not_block_downstream(monkeypatch):
    dist, relays, events = distribution(monkeypatch, fail=True)
    s = snapshot(3)
    dist.pin(s)
    try:
        with pytest.raises(RuntimeError, match="RDMA read failed"):
            await asyncio.wait_for(dist.snapshots_for(1, 3), 2)
        local = await asyncio.wait_for(dist.snapshots_for(2, 3), 2)
        assert local["c"].version == 3
        assert ("pull", "c", "a") in events
    finally:
        await dist.close()
    assert all(not relay.live for relay in relays.values())


@pytest.mark.asyncio
async def test_active_version_is_retained_and_pending_publications_coalesce(monkeypatch):
    dist, _, events = distribution(monkeypatch)
    old = snapshot(1)
    dist.pin(old)
    try:
        await asyncio.wait_for(dist.snapshots_for(0, 1), 2)
        dist.publish(snapshot(2))
        dist.publish(snapshot(3))
        await asyncio.sleep(0)
        assert dist.get_status()["active_version"] == 1
        dist.unpin(old)
        await asyncio.wait_for(dist.snapshots_for(0, 3), 2)
        assert dist.get_status()["active_version"] == 3
        assert len([e for e in events if e[0] == "load"]) == 4
    finally:
        await dist.close()


@pytest.mark.asyncio
async def test_cancelling_waiter_does_not_cancel_publication(monkeypatch):
    unblock = asyncio.Event()
    dist, _, _ = distribution(monkeypatch, blocked=unblock)
    dist.publish(snapshot(1))
    waiter = asyncio.create_task(dist.snapshots_for(1, 1))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    unblock.set()
    try:
        assert (await asyncio.wait_for(dist.snapshots_for(1, 1), 2))["b"].version == 1
    finally:
        await dist.close()


@pytest.mark.asyncio
async def test_dead_predecessor_falls_back_to_master_without_losing_healthy_receivers(monkeypatch):
    dist, relays, events = distribution(monkeypatch)
    relays["b"].failed_sources.add("a")
    relays["c"].failed_sources.add("a")
    dist.publish(snapshot(4))
    try:
        assert (await asyncio.wait_for(dist.snapshots_for(1, 4), 2))["b"].version == 4
        assert (await asyncio.wait_for(dist.snapshots_for(2, 4), 2))["c"].version == 4
        assert ("pull", "b", "source") in events
        assert not dist.get_status()["node_errors"]
    finally:
        await dist.close()


@pytest.mark.asyncio
async def test_transient_failure_can_retry_same_pinned_version(monkeypatch):
    dist, relays, _ = distribution(monkeypatch, fail=True)
    s = snapshot(1)
    dist.pin(s)
    try:
        with pytest.raises(RuntimeError, match="RDMA read failed"):
            await dist.snapshots_for(1, 1)
        relays["b"].fail = False
        await dist.retry_failed(1, 1)
        assert (await asyncio.wait_for(dist.snapshots_for(1, 1), 2))["b"].version == 1
    finally:
        dist.unpin(s)
        await dist.close()


@pytest.mark.asyncio
async def test_dead_actor_cleanup_does_not_poison_healthy_nodes(monkeypatch):
    dist, relays, _ = distribution(monkeypatch)
    relays["a"].dead_on_release = True
    dist.publish(snapshot(1))
    try:
        assert (await asyncio.wait_for(dist.snapshots_for(2, 1), 2))["c"].version == 1
        await dist.current.task
        assert dist.get_status()["fatal_error"] is None
        assert "a" in dist.get_status()["node_errors"]
        assert "a" not in dist.get_status()["ready_nodes"]
    finally:
        await dist.close()
