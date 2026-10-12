# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cross-process snapshot ownership and independent publication, without GPUs."""

import asyncio
import copy

import pytest
import ray
import torch

from verl.checkpoint_engine.weight_publication import WeightPublisher
from verl.checkpoint_engine.weight_scheduler import WeightPublicationScheduler
from verl.checkpoint_engine.weight_snapshot import export_weight_snapshot, iter_snapshot_weights


@ray.remote(num_cpus=0)
class Signal:
    def __init__(self):
        self.event = asyncio.Event()

    async def wait(self):
        await self.event.wait()

    async def release(self):
        self.event.set()


@ray.remote(num_cpus=0)
class TrainingSource:
    def __init__(self):
        self.weight = torch.tensor([1.0, 2.0])

    def export(self, version):
        return export_weight_snapshot([("weight", self.weight)], version, bucket_size=8, is_master=True)

    def mutate(self):
        self.weight.add_(100)
        return self.weight.tolist()

    def hold(self, gate):
        ray.get(gate.wait.remote())


class CPUReplicaInstaller:
    """Replace GPU installation only; use real Ray buckets and scheduler."""

    def __init__(self):
        self.states = {
            rank: {"weight_version": "initial", "lifecycle_state": "SERVING", "pin": None} for rank in range(2)
        }
        self.installed = {}
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_replica_states(self):
        return copy.deepcopy(self.states)

    async def update_replica_from_snapshot(self, snapshot, rank, recovery=False):
        self.states[rank]["lifecycle_state"] = "UPDATING"
        self.started.set()
        await self.release.wait()
        self.installed[rank] = {name: tensor.tolist() async for name, tensor in iter_snapshot_weights(snapshot)}
        self.states[rank].update(weight_version=snapshot.version, lifecycle_state="SERVING")
        return {"replicas": {rank: {"status": "updated", "version": snapshot.version}}, "updated": 1}


@ray.remote(num_cpus=0)
class PublicationProbe(WeightPublisher):
    def __init__(self):
        self.manager = CPUReplicaInstaller()
        self.notifications = []
        self.last_commit_metrics = {}
        self.scheduler = WeightPublicationScheduler(self.manager, on_commit=self._on_commit)

    async def _on_commit(self, version, result):
        self.notifications.append(version)

    async def wait_until_started(self):
        await self.manager.started.wait()

    async def release_installation(self):
        self.manager.release.set()

    async def installed(self):
        return self.manager.installed, self.notifications


@pytest.fixture(scope="module")
def ray_runtime():
    ray.init(address="local", num_cpus=2, include_dashboard=False, object_store_memory=100_000_000)
    yield
    ray.shutdown()


def test_snapshot_remains_immutable_and_publishes_while_training_actor_is_busy(ray_runtime):
    source, publisher, gate = TrainingSource.remote(), PublicationProbe.remote(), Signal.remote()
    try:
        snapshot = ray.get(source.export.remote(1), timeout=30)
        assert ray.get(publisher.publish.remote(snapshot), timeout=10)["accepted_version"] == 1
        ray.get(publisher.wait_until_started.remote(), timeout=10)

        # Training proceeds while installation is held on a different actor.
        assert ray.get(source.mutate.remote(), timeout=10) == [101.0, 102.0]
        busy_training = source.hold.remote(gate)
        del snapshot
        ray.get(publisher.release_installation.remote(), timeout=10)
        status = ray.get(publisher.wait_for_version.remote(1, 10), timeout=15)
        assert [state["weight_version"] for state in status["replicas"].values()] == [1, 1]
        assert ray.wait([busy_training], timeout=0)[0] == []
        installed, notifications = ray.get(publisher.installed.remote(), timeout=10)
        assert installed == {0: {"weight": [1.0, 2.0]}, 1: {"weight": [1.0, 2.0]}}
        assert notifications == [1]
        ray.get(gate.release.remote(), timeout=10)
        ray.get(busy_training, timeout=10)
        ray.get(publisher.close.remote(), timeout=10)
    finally:
        for actor in (source, publisher, gate):
            ray.kill(actor)
