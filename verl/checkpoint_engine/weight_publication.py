# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Independent weight publication using immutable Ray object-store snapshots."""

import asyncio

import ray

from verl.checkpoint_engine import base


class SnapshotCheckpointEngineManager(base.CheckpointEngineManager):
    """Reuse replica isolation and retries with a trainer-independent source."""

    def __init__(self, checkpoint_config, fine_grained_config, replicas, load_balancer_handle):
        super().__init__(
            config=checkpoint_config,
            actor_wg=None,
            replicas=replicas,
            fine_grained_config=fine_grained_config,
            load_balancer_handle=load_balancer_handle,
        )
        self._require_fine_grained()
        self._snapshot = None

    async def update_replica_from_snapshot(self, snapshot, replica_rank: int, recovery: bool = False) -> dict:
        if type(snapshot.version) is not int or snapshot.version < 0:
            raise ValueError("snapshot version must be a nonnegative integer")
        async with self._weight_update_lock:
            self._snapshot = snapshot
            try:
                return await self._update_selected_replicas(
                    [self._replica(replica_rank)], snapshot.version, recovery=recovery
                )
            finally:
                self._snapshot = None

    async def _transfer_replica_weights(self, replica, global_steps: int) -> dict:
        if self._snapshot is None or self._snapshot.version != global_steps:
            raise RuntimeError("replica update requires a matching immutable snapshot")
        rollout = base.RayWorkerGroup(
            worker_handles=replica.workers,
            ray_cls_with_init=base.RayClassWithInitArgs(cls=base._worker_cls),
        )
        await self._run_worker_calls(lambda: rollout.update_weights_from_snapshot(snapshot=self._snapshot))
        return {}


class WeightPublisher:
    """CPU coordinator. Its background loop never calls training workers."""

    def __init__(self, checkpoint_config, fine_grained_config, replicas, load_balancer_handle, rollouter):
        from verl.checkpoint_engine.weight_scheduler import WeightPublicationScheduler

        if not replicas:
            raise ValueError("weight publication requires standalone rollout replicas")
        manager = SnapshotCheckpointEngineManager(
            checkpoint_config, fine_grained_config, replicas, load_balancer_handle
        )
        self.rollouter = rollouter
        self.last_commit_metrics = {}
        self.scheduler = WeightPublicationScheduler(manager, on_commit=self._on_commit)

    async def _on_commit(self, version: int, result: dict):
        metrics = await self.rollouter.reset_staleness.remote()
        self.last_commit_metrics = {"version": version, "metrics": metrics}

    async def publish(self, snapshot) -> dict:
        return await self.scheduler.publish(snapshot)

    async def get_status(self) -> dict:
        status = await self.scheduler.get_status()
        status["last_commit"] = self.last_commit_metrics
        return status

    async def wait_for_version(self, version: int, timeout_s: float = 600.0) -> dict:
        return await self.scheduler.wait_for_version(version, timeout_s)

    async def pin_replica(self, replica_rank: int, reason: str) -> dict:
        return await self.scheduler.pin_replica(replica_rank, reason)

    async def unpin_replica(self, replica_rank: int) -> dict:
        return await self.scheduler.unpin_replica(replica_rank)

    async def recover_replica(self, replica_rank: int) -> dict:
        return await self.scheduler.recover_replica(replica_rank)

    async def watch_health(self):
        """Make background failure visible even while the trainer awaits data."""
        while True:
            status = await self.scheduler.get_status()
            if status.get("closed"):
                return
            if status.get("fatal_error"):
                raise RuntimeError(f"weight publisher failed: {status['fatal_error']}")
            states = status.get("replicas", {})
            if (
                status.get("latest_version") is not None
                and states
                and all(state["lifecycle_state"] == "QUARANTINED" for state in states.values())
            ):
                raise RuntimeError("all rollout replicas are quarantined; weight publisher recovery is required")
            await asyncio.sleep(0.5)

    async def close(self, drain: bool = True):
        return await self.scheduler.close(drain=drain)


WeightPublisherActor = ray.remote(num_cpus=0, max_restarts=0)(WeightPublisher)
