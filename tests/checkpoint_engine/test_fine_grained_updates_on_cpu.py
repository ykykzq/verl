# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import inspect
from collections import Counter
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine import base
from verl.workers.config import CheckpointEngineConfig
from verl.workers.rollout.router import GlobalRequestLoadBalancer


class RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args, **kwargs):
        result = self.fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result


class RouterHandle:
    def __init__(self, router):
        self.router = router

    def __getattr__(self, name):
        return RemoteMethod(getattr(self.router, name))


class Replica:
    def __init__(self, rank, environment):
        self.replica_rank = rank
        self.server_address = f"s{rank}"
        self.environment = environment
        self.workers = [SimpleNamespace(replica=self, index=i) for i in range(2)]
        self.version = 7
        self.generation_open = True
        self.kv_open = True
        self.weights = ("old",)
        self.server_handle = SimpleNamespace(
            abort_weight_update_from_ipc=RemoteMethod(self.abort_weight_update),
            get_trajectory_migration_capabilities=RemoteMethod(self.capabilities),
        )
        self.servers = [self.server_handle]

    async def event(self, action):
        self.environment.events.append((self.server_address, action))
        key = (self.server_address, action)
        if self.environment.failures[key]:
            self.environment.failures[key] -= 1
            raise RuntimeError(f"{self.server_address} {action} failed")

    async def abort_all_requests(self, reject_request=False):
        assert reject_request is True
        self.generation_open = False
        await self.event("abort")

    async def release_kv_cache(self):
        assert not self.generation_open
        self.kv_open = False
        await self.event("release_kv")

    async def resume_kv_cache(self):
        await self.event("resume_kv")
        self.kv_open = True

    async def resume_generation(self):
        # Even a partially successful resume must be closed again on failure.
        self.generation_open = True
        await self.event("resume_generation")

    async def abort_weight_update(self, round_id=None):
        self.generation_open = False
        await self.event("abort_weight")

    async def capabilities(self):
        await self.event("capabilities")
        return {"model_id": "model", "weight_version": self.version}


class WorkerGroup:
    def __init__(self, environment, replica=None):
        self.environment = environment
        self.replica = replica
        self.world_size = 2
        self.name = replica.server_address if replica else "trainer"

    def execute_checkpoint_engine(self, method, **kwargs):
        assert len(method) == self.world_size

        async def execute(rank, action):
            if action in ("prepare", "prepare_temporary"):
                self.environment.prepare_modes.append(action)
                action = "prepare"
            self.environment.events.append((self.name, f"{action}:{rank}"))
            key = (self.name, action)
            if self.environment.failures[key]:
                self.environment.failures[key] -= 1
                raise RuntimeError(f"{self.name} {action} failed")
            await asyncio.sleep(0)
            return {"worker": self.name, "rank": rank}

        return [asyncio.create_task(execute(rank, action)) for rank, action in enumerate(method)]

    def update_weights(self, global_steps=None, mode=None):
        async def transfer():
            self.environment.events.append((self.name, "transfer"))
            if self.replica:
                self.replica.weights = ("partial",)
                states = self.environment.router.get_replica_states()
                assert states[self.name]["lifecycle_state"] == "UPDATING"
                assert states[self.name]["weight_version"] == 7
                self.environment.serving_during_transfer.append(
                    {sid for sid, state in states.items() if state["lifecycle_state"] == "SERVING"}
                )
            if self.environment.transfer_started:
                self.environment.transfer_started.set()
                await self.environment.transfer_continue.wait()
            key = (self.name, "transfer")
            if self.environment.failures[key]:
                self.environment.failures[key] -= 1
                raise RuntimeError(f"{self.name} transfer failed")
            if self.replica:
                self.replica.weights = ("embedding", "attention", "head")
                self.replica.version = global_steps
            return {"bytes_sent": 100} if self.replica is None else None

        return [asyncio.create_task(transfer())]


@pytest.fixture
def environment(monkeypatch):
    env = SimpleNamespace(
        events=[],
        failures=Counter(),
        serving_during_transfer=[],
        topologies=[],
        prepare_modes=[],
        transfer_started=None,
        transfer_continue=None,
    )
    env.replicas = [Replica(i, env) for i in range(2)]
    env.router = GlobalRequestLoadBalancer(
        servers={r.server_address: r.server_handle for r in env.replicas},
        server_metadata={r.server_address: {"weight_version": 7, "model_id": "model"} for r in env.replicas},
    )
    env.actor = WorkerGroup(env)

    class Backend:
        def prepare_temporary(self):
            raise AssertionError("the worker group dispatches prepare_temporary on remote engines")

        @classmethod
        def build_topology(cls, actor_size, rollout_size, metadata):
            env.topologies.append(metadata)
            assert actor_size == 2 and rollout_size == 2
            return {"rank": [0, 1]}, {"rank": [2, 3]}

    monkeypatch.setattr(base.CheckpointEngineRegistry, "get", lambda backend: Backend)

    def rollout_group(worker_handles, ray_cls_with_init):
        replica = worker_handles[0].replica
        assert worker_handles == replica.workers
        return WorkerGroup(env, replica)

    monkeypatch.setattr(base, "RayWorkerGroup", rollout_group)

    def manager(**kwargs):
        return base.CheckpointEngineManager(
            CheckpointEngineConfig(backend="test"),
            env.actor,
            env.replicas,
            fine_grained_config=SimpleNamespace(
                enabled=True,
                max_retries=kwargs.get("max_retries", 1),
                retry_backoff_s=0,
                continue_on_failure=kwargs.get("continue_on_failure", True),
            ),
            load_balancer_handle=RouterHandle(env.router),
        )

    env.manager = manager
    return env


@pytest.mark.asyncio
async def test_updates_only_selected_replica_and_keeps_other_serving(environment):
    env = environment
    manager = env.manager()
    result = await manager.update_replica_weights(0, 8)
    assert result["replicas"][0]["status"] == "updated"
    assert result["replicas"][0]["attempts"] == 1
    assert result["updated"] == 1
    assert env.serving_during_transfer == [{"s1"}]
    assert all(name != "s1" for name, action in env.events)
    assert {m["worker"] for m in env.topologies[0]} == {"trainer", "s0"}
    assert env.replicas[0].weights == ("embedding", "attention", "head")
    states = await manager.get_replica_states()
    assert states[0]["weight_version"] == 8
    assert states[1]["weight_version"] == 7
    assert env.replicas[0].generation_open


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transfer", "resume_kv", "capabilities"])
async def test_bulk_resume_preserves_quarantine(environment, failure):
    env = environment
    manager = env.manager(max_retries=0)
    env.failures[("s0", failure)] = 1
    await manager.update_replica_weights(0, 8)
    assert not env.replicas[0].generation_open
    env.events.clear()

    await manager.resume_generation_replicas()

    assert not env.replicas[0].generation_open
    assert env.replicas[1].generation_open
    assert env.events == [("s1", "resume_generation")]


@pytest.mark.asyncio
async def test_bulk_resume_excludes_updating_replica(environment):
    env = environment
    manager = env.manager()
    env.router.begin_replica_update("s0", 8)
    env.replicas[0].generation_open = False

    await manager.resume_generation_replicas()

    assert not env.replicas[0].generation_open
    assert env.events == [("s1", "resume_generation")]


@pytest.mark.asyncio
async def test_rolling_update_skips_pins_then_updates_in_order(environment):
    env = environment
    manager = env.manager()
    await manager.pin_replica(0, "keep reference policy")
    result = await manager.update_weights(8)
    assert result["replicas"][0]["status"] == "pinned"
    assert result["replicas"][0]["attempts"] == 0
    assert result["pinned"] == 1
    assert not any(name == "s0" for name, _ in env.events)
    await manager.unpin_replica(0)
    env.events.clear()
    # This test's transport expects the old published version before transfer.
    env.router.update_server_metadata("s1", {"weight_version": 8})
    result = await manager.update_weights(8, replica_ranks=[0])
    assert result["updated"] == 1


@pytest.mark.asyncio
async def test_all_replicas_transfer_sequentially(environment):
    env = environment
    result = await env.manager().update_weights(8)
    assert result["updated"] == 2
    assert env.events.index(("s0", "resume_generation")) < env.events.index(("s1", "abort"))
    assert env.serving_during_transfer == [{"s1"}, {"s0"}]
    assert env.prepare_modes == ["prepare_temporary"] * 8


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepare", "init_process_group", "transfer", "finalize"])
async def test_failed_attempt_rebuilds_group_and_finalizes_both_sides(environment, phase):
    env = environment
    env.failures[("s0", phase)] = 1
    result = await env.manager().update_replica_weights(0, 8)
    assert result["replicas"][0]["attempts"] == 2
    assert result["replicas"][0]["status"] == "updated"
    assert env.events.count(("trainer", "prepare:0")) == 2
    assert env.events.count(("trainer", "finalize:0")) == 2
    assert env.events.count(("s0", "finalize:0")) == 2
    assert env.replicas[0].weights == ("embedding", "attention", "head")
    if phase in ("transfer", "finalize"):
        assert env.events.count(("trainer", "transfer")) == 2
        assert env.events.count(("s0", "transfer")) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["transfer", "finalize", "resume_kv", "resume_generation", "capabilities"])
async def test_final_failure_quarantines_and_keeps_generation_closed(environment, phase):
    env = environment
    env.failures[("s0", phase)] = 10
    manager = env.manager()
    result = await manager.update_weights(8)
    assert result["replicas"][0]["status"] == "quarantined"
    assert result["replicas"][0]["attempts"] == 2
    assert result["replicas"][0]["version"] == 7
    assert result["quarantined"] == 1
    assert result["updated"] == 1
    assert not env.replicas[0].generation_open
    assert env.router.acquire_server("new-request")[0] == "s1"
    assert (await manager.get_replica_states())[0]["last_update_error"]


@pytest.mark.asyncio
async def test_fail_fast_isolates_before_raising_and_manual_recovery_rejoins(environment):
    env = environment
    env.failures[("s0", "transfer")] = 1
    manager = env.manager(max_retries=0, continue_on_failure=False)
    with pytest.raises(RuntimeError, match="transfer failed"):
        await manager.update_weights(8)
    assert env.router.get_replica_states()["s0"]["lifecycle_state"] == "QUARANTINED"
    assert not env.replicas[0].generation_open
    assert not any(name == "s1" for name, _ in env.events)
    with pytest.raises(RuntimeError, match="[Qq]uarantin|recovery|recover"):
        await manager.update_replica_weights(0, 8)
    result = await manager.recover_replica(0, 8)
    assert result["replicas"][0]["status"] == "updated"
    state = (await manager.get_replica_states())[0]
    assert state["lifecycle_state"] == "SERVING"
    assert state["last_update_error"] is None
    assert env.replicas[0].generation_open


@pytest.mark.asyncio
async def test_concurrent_calls_do_not_overlap_trainer_process_groups(environment):
    env = environment
    env.transfer_started, env.transfer_continue = asyncio.Event(), asyncio.Event()
    manager = env.manager()
    first = asyncio.create_task(manager.update_replica_weights(0, 8))
    await env.transfer_started.wait()
    second = asyncio.create_task(manager.update_replica_weights(1, 8))
    await asyncio.sleep(0)
    assert env.router.get_replica_states()["s1"]["lifecycle_state"] == "SERVING"
    assert not any(name == "s1" for name, _ in env.events)
    env.transfer_continue.set()
    await asyncio.gather(first, second)
    assert env.events.index(("s0", "resume_generation")) < env.events.index(("s1", "prepare:0"))


@pytest.mark.asyncio
async def test_cancellation_finishes_cleanup_and_quarantines(environment):
    env = environment
    env.transfer_started, env.transfer_continue = asyncio.Event(), asyncio.Event()
    manager = env.manager()
    task = asyncio.create_task(manager.update_replica_weights(0, 8))
    await env.transfer_started.wait()
    task.cancel()
    env.transfer_continue.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ("trainer", "finalize:0") in env.events
    assert ("s0", "finalize:0") in env.events
    assert env.router.get_replica_states()["s0"]["lifecycle_state"] == "QUARANTINED"
    assert not env.replicas[0].generation_open


def test_enabled_requires_router_before_any_worker_action(environment):
    env = environment
    with pytest.raises(ValueError, match="[Rr]outer|load_balancer"):
        base.CheckpointEngineManager(
            CheckpointEngineConfig(backend="test"),
            env.actor,
            env.replicas,
            fine_grained_config=SimpleNamespace(enabled=True),
        )
    assert env.events == []


@pytest.mark.asyncio
async def test_disabled_rejects_selective_update_but_preserves_naive_path(monkeypatch):
    calls = []
    actor = SimpleNamespace(update_weights=lambda **kwargs: calls.append(kwargs) or [])
    manager = base.CheckpointEngineManager(CheckpointEngineConfig(backend="naive"), actor, [])
    monkeypatch.setattr(base.ray, "get", lambda refs: refs)
    assert await manager.update_weights(9) == {}
    assert calls == [{"global_steps": 9, "mode": "naive"}]
    with pytest.raises(ValueError, match="enabled|disabled|fine.grained"):
        await manager.update_weights(10, replica_ranks=[0])
    assert len(calls) == 1


def test_backend_without_temporary_group_support_fails_closed(environment, monkeypatch):
    monkeypatch.setattr(base.CheckpointEngineRegistry, "get", lambda backend: base.ColocatedCheckpointEngine)
    with pytest.raises(ValueError, match="temporary"):
        environment.manager()
    assert environment.events == []


@pytest.mark.asyncio
async def test_cancel_during_begin_acknowledgment_does_not_strand_updating_replica(environment):
    env = environment
    manager = env.manager()
    began, acknowledgment = asyncio.Event(), asyncio.Event()

    async def begin(server_id, version):
        state = env.router.begin_replica_update(server_id, version)
        began.set()
        await acknowledgment.wait()
        return state

    manager.load_balancer.begin_replica_update = RemoteMethod(begin)
    task = asyncio.create_task(manager.update_replica_weights(0, 8))
    await began.wait()
    task.cancel()
    acknowledgment.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert env.router.get_replica_states()["s0"]["lifecycle_state"] == "QUARANTINED"
    assert not env.replicas[0].generation_open


@pytest.mark.asyncio
async def test_cancel_during_commit_acknowledgment_keeps_published_replica_serving(environment):
    env = environment
    manager = env.manager()
    committed, acknowledgment = asyncio.Event(), asyncio.Event()

    async def commit(server_id, version, metadata=None):
        state = env.router.commit_replica_update(server_id, version, metadata)
        committed.set()
        await acknowledgment.wait()
        return state

    manager.load_balancer.commit_replica_update = RemoteMethod(commit)
    task = asyncio.create_task(manager.update_replica_weights(0, 8))
    await committed.wait()
    task.cancel()
    acknowledgment.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert env.router.get_replica_states()["s0"]["lifecycle_state"] == "SERVING"
    assert env.replicas[0].generation_open


@pytest.mark.asyncio
async def test_cleanup_rpc_error_still_marks_quarantined(environment):
    env = environment
    env.failures[("s0", "transfer")] = 1
    manager = env.manager(max_retries=0)

    def broken_dispatch():
        raise RuntimeError("IPC cleanup dispatch failed")

    env.replicas[0].server_handle.abort_weight_update_from_ipc.remote = broken_dispatch
    result = await manager.update_replica_weights(0, 8)
    assert result["replicas"][0]["status"] == "quarantined"
    assert "cleanup" in result["replicas"][0]["error"]
    assert env.router.get_replica_states()["s0"]["lifecycle_state"] == "QUARANTINED"
    assert not env.replicas[0].generation_open


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_fully_async_setup_wires_router_only_for_enabled_updates(environment, enabled):
    from omegaconf import OmegaConf

    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    env = environment
    router_handle = RouterHandle(env.router)
    rollouter = SimpleNamespace(get_replicas=RemoteMethod(lambda: env.replicas))
    if enabled:
        rollouter_class = FullyAsyncRollouter.__ray_metadata__.modified_class
        rollouter_instance = SimpleNamespace(llm_server_manager=SimpleNamespace(global_load_balancer=router_handle))
        rollouter.get_load_balancer = RemoteMethod(lambda: rollouter_class.get_load_balancer(rollouter_instance))
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "checkpoint_engine": {
                            "_target_": "verl.workers.config.CheckpointEngineConfig",
                            "backend": "test",
                        },
                        "fine_grained_weight_update": {
                            "_target_": "verl.workers.config.FineGrainedWeightUpdateConfig",
                            "enabled": enabled,
                            "retry_backoff_s": 0,
                        },
                    }
                },
            }
        ),
        actor_wg=env.actor,
        rollouter=rollouter,
    )
    trainer_class = FullyAsyncTrainer.__ray_metadata__.modified_class
    await trainer_class._setup_checkpoint_manager(trainer)
    assert trainer.checkpoint_manager.fine_grained_config.enabled is enabled
    if enabled:
        await trainer.checkpoint_manager.pin_replica(0, "reference")
        result = await trainer.checkpoint_manager.update_weights(8)
        assert result["pinned"] == 1
        assert result["updated"] == 1
        assert env.router.get_replica_states()["s0"]["pin"]["weight_version"] == 7
    else:
        assert trainer.checkpoint_manager.load_balancer is None
