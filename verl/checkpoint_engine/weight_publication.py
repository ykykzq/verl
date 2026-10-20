# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Independent immutable weight publication with optional CPU RDMA relays."""

import asyncio

import ray

from verl.checkpoint_engine import base


class SnapshotCheckpointEngineManager(base.CheckpointEngineManager):
    """Reuse replica isolation and retries with a trainer-independent source."""

    def __init__(self, checkpoint_config, fine_grained_config, replicas, load_balancer_handle, relay=None):
        super().__init__(
            config=checkpoint_config,
            actor_wg=None,
            replicas=replicas,
            fine_grained_config=fine_grained_config,
            load_balancer_handle=load_balancer_handle,
        )
        self._require_fine_grained()
        self._snapshot = None
        self.relay = relay
        self._local_snapshots = None

    async def update_replica_from_snapshot(self, snapshot, replica_rank: int, recovery: bool = False) -> dict:
        if type(snapshot.version) is not int or snapshot.version < 0:
            raise ValueError("snapshot version must be a nonnegative integer")
        async with self._weight_update_lock:
            self._snapshot = snapshot
            pinned = False
            try:
                if self.relay is not None:
                    self.relay.pin(snapshot)
                    pinned = True
                return await self._update_selected_replicas(
                    [self._replica(replica_rank)], snapshot.version, recovery=recovery
                )
            finally:
                if pinned:
                    self.relay.unpin(snapshot)
                self._snapshot = None
                self._local_snapshots = None

    async def _prepare_replica_weights(self, replica, global_steps):
        if self.relay is not None:
            await self.relay.retry_failed(replica.replica_rank, global_steps)
            self._local_snapshots = await self.relay.snapshots_for(replica.replica_rank, global_steps)

    async def _transfer_replica_weights(self, replica, global_steps: int) -> dict:
        if self._snapshot is None or self._snapshot.version != global_steps:
            raise RuntimeError("replica update requires a matching immutable snapshot")
        rollout = base.RayWorkerGroup(
            worker_handles=replica.workers,
            ray_cls_with_init=base.RayClassWithInitArgs(cls=base._worker_cls),
        )
        if self.relay is not None:
            if not self._local_snapshots or any(s.version != global_steps for s in self._local_snapshots.values()):
                raise RuntimeError("replica update requires complete, matching node-local relay snapshots")
            await self._run_worker_calls(
                lambda: rollout.update_weights_from_relay(snapshot_by_node=self._local_snapshots)
            )
        else:
            await self._run_worker_calls(lambda: rollout.update_weights_from_snapshot(snapshot=self._snapshot))
        return {}


class WeightPublisher:
    """CPU coordinator. Its background loop never calls training workers."""

    def __init__(
        self,
        checkpoint_config,
        fine_grained_config,
        replicas,
        load_balancer_handle,
        rollouter,
        joint_scheduling_config=None,
        relay_config=None,
        source_node_id=None,
    ):
        from verl.checkpoint_engine.weight_scheduler import WeightPublicationScheduler

        if not replicas:
            raise ValueError("weight publication requires standalone rollout replicas")
        self.relay = None
        if relay_config is not None:
            from verl.checkpoint_engine.relay_distribution import RelayDistribution

            self.relay = RelayDistribution(replicas, source_node_id, relay_config)
        manager = SnapshotCheckpointEngineManager(
            checkpoint_config, fine_grained_config, replicas, load_balancer_handle, relay=self.relay
        )
        self.rollouter = rollouter
        self.last_commit_metrics = {}
        self.scheduler = WeightPublicationScheduler(
            manager, on_commit=self._on_commit, joint_scheduling_config=joint_scheduling_config
        )

    async def _on_commit(self, version: int, result: dict):
        metrics = await self.rollouter.reset_staleness.remote()
        self.last_commit_metrics = {"version": version, "metrics": metrics}

    async def publish(self, snapshot) -> dict:
        result = await self.scheduler.publish(snapshot)
        if getattr(self, "relay", None) is not None:
            self.relay.publish(snapshot)
        return result

    async def get_status(self) -> dict:
        status = await self.scheduler.get_status()
        status["last_commit"] = self.last_commit_metrics
        if getattr(self, "relay", None) is not None:
            status["relay"] = self.relay.get_status()
            status["fatal_error"] = status["fatal_error"] or status["relay"]["fatal_error"]
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
            status = await self.get_status()
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
            latest = status.get("latest_version")
            if getattr(self.scheduler, "_joint_enabled", False) and status.get("idle") and latest is not None:
                serving_latest = any(
                    state["lifecycle_state"] == "SERVING" and self.scheduler._version(state) >= latest
                    for state in states.values()
                )
                can_update = any(
                    state["lifecycle_state"] == "UPDATING"
                    or (state["lifecycle_state"] == "SERVING" and not state.get("pin"))
                    for state in states.values()
                )
                if not serving_latest and not can_update:
                    raise RuntimeError(
                        f"no serving replica can reach weight version {latest}; "
                        "joint scheduling has closed new admission; unpin or recover a rollout replica"
                    )
            await asyncio.sleep(0.5)

    async def close(self, drain: bool = True):
        try:
            return await self.scheduler.close(drain=drain)
        finally:
            if getattr(self, "relay", None) is not None:
                await self.relay.close()


WeightPublisherActor = ray.remote(num_cpus=0, max_restarts=0)(WeightPublisher)
