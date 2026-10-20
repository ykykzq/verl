# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
from types import SimpleNamespace

import pytest

from tests.checkpoint_engine.test_fine_grained_updates_on_cpu import (
    RouterHandle,
    WorkerGroup,
    environment,  # noqa: F401
)
from verl.checkpoint_engine import base
from verl.checkpoint_engine.weight_publication import SnapshotCheckpointEngineManager


def manager(env, monkeypatch, fail=False):
    class LocalRelay:
        def __init__(self):
            self.active = None
            self.started, self.ready = asyncio.Event(), asyncio.Event()

        def pin(self, snapshot):
            assert self.active is None
            self.active = snapshot

        def unpin(self, snapshot):
            assert self.active is snapshot
            self.active = None

        async def retry_failed(self, rank, version):
            pass

        async def snapshots_for(self, rank, version):
            self.started.set()
            await self.ready.wait()
            if fail:
                raise RuntimeError("node RDMA failed")
            env.events.append((f"s{rank}", "local_snapshot_ready"))
            return {"node": self.active}

    class LocalWorkers(WorkerGroup):
        def update_weights_from_relay(self, snapshot_by_node):
            assert set(snapshot_by_node) == {"node"}
            return self.update_weights(snapshot_by_node["node"].version)

    monkeypatch.setattr(
        base,
        "RayWorkerGroup",
        lambda worker_handles, ray_cls_with_init: LocalWorkers(env, worker_handles[0].replica),
    )
    original = env.manager(max_retries=0)
    relay = LocalRelay()
    instance = SnapshotCheckpointEngineManager(
        original.config, original.fine_grained_config, env.replicas, RouterHandle(env.router), relay=relay
    )
    return instance, relay


@pytest.mark.asyncio
async def test_relay_snapshot_is_ready_before_replica_drain(environment, monkeypatch):  # noqa: F811
    m, relay = manager(environment, monkeypatch)
    task = asyncio.create_task(m.update_replica_from_snapshot(SimpleNamespace(version=8), 0))
    await asyncio.wait_for(relay.started.wait(), 2)
    assert environment.replicas[0].generation_open
    assert environment.replicas[1].generation_open
    relay.ready.set()
    result = await task
    assert result["updated"] == 1 and relay.active is None
    assert environment.events.index(("s0", "local_snapshot_ready")) < environment.events.index(("s0", "abort"))
    assert environment.replicas[0].version == 8
    assert environment.replicas[1].version == 7


@pytest.mark.asyncio
async def test_relay_failure_uses_replica_quarantine_and_releases_lease(environment, monkeypatch):  # noqa: F811
    m, relay = manager(environment, monkeypatch, fail=True)
    relay.ready.set()
    result = await m.update_replica_from_snapshot(SimpleNamespace(version=8), 0)
    assert result["quarantined"] == 1 and relay.active is None
    assert environment.replicas[1].generation_open
    assert not any(action == "transfer" for _, action in environment.events)


@pytest.mark.asyncio
async def test_publisher_admission_does_not_wait_for_relay_distribution(monkeypatch):
    from verl.checkpoint_engine.weight_publication import WeightPublisher

    events = []

    class Scheduler:
        async def publish(self, snapshot):
            events.append("admit")
            return {"accepted_version": snapshot.version}

    class Relay:
        def publish(self, snapshot):
            events.append(("stage", snapshot.version))

    publisher = WeightPublisher.__new__(WeightPublisher)
    publisher.scheduler, publisher.relay = Scheduler(), Relay()
    assert await publisher.publish(SimpleNamespace(version=4)) == {"accepted_version": 4}
    assert events == ["admit", ("stage", 4)]
