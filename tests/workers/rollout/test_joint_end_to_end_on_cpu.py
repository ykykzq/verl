# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Real publication, routing and continuation, replacing GPU work only."""

import asyncio
import hashlib
import inspect
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.checkpoint_engine.weight_scheduler import WeightPublicationScheduler
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.router import GlobalRequestLoadBalancer


class Remote:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        async def invoke():
            result = self.function(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        # Like Ray, fire-and-forget release RPCs execute even without an await.
        return asyncio.create_task(invoke())


class Handle:
    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, name):
        return Remote(getattr(self.instance, name))


def digest(prefix):
    return hashlib.sha256(b"".join(int(token).to_bytes(8, "little", signed=True) for token in prefix)).hexdigest()


class CPUBackend:
    def __init__(self, server_id):
        self.server_id = server_id
        self.version = "initial"
        self.accepted = {}
        self.generated = []
        self.migrations = []

    def get_trajectory_migration_capabilities(self):
        return {
            "model_id": "cpu-model",
            "weight_version": self.version,
            "kv_transfer_backends": ["remote_prefix"],
            "kv_transfer_domain": "cpu-shared-cache",
            "kv_transfer_namespace": f"cache-{self.version}",
            "rollout_config_fingerprint": "cpu-layout",
        }

    def prepare_trajectory_migration(self, trajectory_state, target_metadata, backend, timeout_s):
        assert backend == "remote_prefix"
        assert target_metadata["weight_version"] == self.version
        prefix = trajectory_state["prompt_ids"] + trajectory_state["generated_token_ids"]
        return {
            **self.get_trajectory_migration_capabilities(),
            "backend": backend,
            "request_id": trajectory_state["request_id"],
            "prefix_tokens": len(prefix),
            "prefix_digest": digest(prefix),
        }

    def accept_trajectory_migration(self, ticket, trajectory_state):
        prefix = trajectory_state["prompt_ids"] + trajectory_state["generated_token_ids"]
        assert ticket["prefix_digest"] == digest(prefix)
        version = ticket.get("target_version", ticket.get("weight_version"))
        if version != self.version:
            return {"accepted": False, "reason": "version changed"}
        self.accepted[ticket["request_id"]] = ticket
        self.migrations.append((ticket["request_id"], ticket["backend"], list(prefix)))
        return {"accepted": True}

    def discard_trajectory_migration(self, request_id, prefix_digest):
        self.accepted.pop(request_id, None)
        return {"discarded": True}

    def abort_request(self, request_id):
        return {"aborted": True}

    def generate(
        self, request_id, prompt_ids, sampling_params, migration_ticket=None, expected_source_version=None, **kwargs
    ):
        if migration_ticket is not None:
            assert self.accepted.pop(request_id) == migration_ticket
        elif expected_source_version is not None and expected_source_version != self.version:
            return TokenOutput(
                token_ids=[],
                log_probs=None,
                stop_reason="aborted",
                extra_fields={"global_steps": self.version, "migration_rejected": True},
            )
        self.generated.append((request_id, self.version, list(prompt_ids)))
        count = sampling_params["max_tokens"]
        offset = len(prompt_ids) - 2
        extra = {"global_steps": self.version, "migration_checkpoint": True}
        if migration_ticket is not None and migration_ticket["backend"] == "recompute":
            extra["forced_prefill_tokens"] = len(prompt_ids)
        return TokenOutput(
            token_ids=list(range(10 + offset, 10 + offset + count)),
            log_probs=[-0.5 - 0.1 * self.version] * count,
            stop_reason="length",
            extra_fields=extra,
        )


class CPUInstaller:
    def __init__(self, router, backends, router_handle):
        self.router = router
        self.backends = backends
        self.load_balancer = router_handle
        self.replicas = [SimpleNamespace(replica_rank=rank, server_address=sid) for rank, sid in enumerate(backends)]
        self.transfers = []
        self.first_update_started = asyncio.Event()
        self.release_first_update = asyncio.Event()

    async def get_replica_states(self):
        states = self.router.get_replica_states()
        return {replica.replica_rank: states[replica.server_address] for replica in self.replicas}

    async def update_replica_from_snapshot(self, snapshot, rank, recovery=False):
        sid = self.replicas[rank].server_address
        self.router.begin_replica_update(sid, snapshot.version)
        self.transfers.append((sid, snapshot.version, self.router._joint.usage(sid)))
        if snapshot.version == 1 and not self.first_update_started.is_set():
            self.first_update_started.set()
            await self.release_first_update.wait()
        backend = self.backends[sid]
        backend.version = snapshot.version
        backend.accepted.clear()
        self.router.commit_replica_update(sid, snapshot.version, backend.get_trajectory_migration_capabilities())
        return {"replicas": {rank: {"status": "updated", "version": snapshot.version}}}


async def eventually(predicate, timeout=3):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout)


@pytest.mark.asyncio
async def test_publication_evacuates_checkpoints_then_recomputes_survivors_and_admits_only_fresh_requests():
    joint = {
        "enabled": True,
        "kv_capacity_tokens": 128,
        "max_batch_size": 8,
        "kv_block_size_tokens": 1,
        "update_cooldown_s": 0,
        "min_decode_window_s": 0,
        "min_decode_tokens": 0,
        "max_defer_s": 1,
        "evacuation_timeout_s": 0.05,
        "poll_interval_s": 0.005,
    }
    migration = {"enabled": True, "checkpoint_tokens": 2, "allow_cross_version_recompute": True}
    backends = {sid: CPUBackend(sid) for sid in ("s0", "s1")}
    router = GlobalRequestLoadBalancer(
        servers={sid: Handle(backend) for sid, backend in backends.items()},
        server_metadata={sid: backend.get_trajectory_migration_capabilities() for sid, backend in backends.items()},
        trajectory_migration_config=migration,
        joint_scheduling_config=joint,
    )
    clock = [100.0]
    router._joint.clock = lambda: clock[0]
    handle = Handle(router)
    evacuate = asyncio.Event()
    resume = asyncio.Event()
    reached = set()
    decisions = []

    async def report(**kwargs):
        result = await router.report_trajectory_progress(**kwargs)
        rid, generated = kwargs["request_id"], kwargs["generated_tokens"]
        reached.add((rid, generated))
        if rid == "b" and generated == 2:
            await evacuate.wait()
        elif (rid in ("a", "c") and generated == 2) or (rid == "b" and generated == 4):
            await resume.wait()
        return result

    handle.report_trajectory_progress = Remote(report)

    async def plan(**kwargs):
        decision = await router.plan_trajectory_migration(**kwargs)
        decisions.append((kwargs["request_id"], {k: v for k, v in decision.items() if not k.endswith("server")}))
        return decision

    handle.plan_trajectory_migration = Remote(plan)
    manager = CPUInstaller(router, backends, handle)
    scheduler = WeightPublicationScheduler(manager, joint_scheduling_config=joint)
    client = FullyAsyncLLMServerClient(
        OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "name": "rtp_llm",
                        "response_length": 6,
                        "fine_grained_weight_update": {"enabled": True},
                        "weight_publication": {"enabled": True, "joint_scheduling": joint},
                        "trajectory_migration": migration,
                    }
                }
            }
        ),
        handle,
    )
    tasks = {}
    try:
        await scheduler.publish(SimpleNamespace(version=0))
        await scheduler.wait_for_version(0, timeout_s=1)
        for rid in ("a", "b", "c"):
            tasks[rid] = asyncio.create_task(client.generate(rid, prompt_ids=[1, 2], sampling_params={"max_tokens": 6}))
            await eventually(lambda rid=rid: (rid, 2) in reached)
        assert router._joint.usage("s0") == (16, 2)
        assert router._joint.usage("s1") == (8, 1)
        assert router.get_total_inflight() == 0

        await scheduler.publish(SimpleNamespace(version=1))
        await eventually(lambda: "s1" in router._joint.evacuating)
        tasks["new"] = asyncio.create_task(client.generate("new", prompt_ids=[1, 2], sampling_params={"max_tokens": 2}))
        await eventually(lambda: "new" in router._joint.trajectories)
        assert router._joint.trajectories["new"].server_id is None
        assert not any(rid == "new" for backend in backends.values() for rid, _, _ in backend.generated)

        evacuate.set()
        await eventually(lambda: ("b", 4) in reached)
        await asyncio.wait_for(manager.first_update_started.wait(), 1)
        assert manager.transfers[-1] == ("s1", 1, (0, 0))
        assert backends["s0"].migrations[0] == ("b", "remote_prefix", [1, 2, 10, 11])
        assert router._joint.usage("s0") == (24, 3)
        assert not tasks["new"].done()

        manager.release_first_update.set()
        await eventually(lambda: router.get_replica_states()["s1"]["weight_version"] == 1)
        await eventually(lambda: "s0" in router._joint.evacuating)
        clock[0] += 0.1
        await scheduler.wait_for_version(1, timeout_s=2)
        assert manager.transfers[-1] == ("s0", 1, (24, 3))
        resume.set()
        _, pending = await asyncio.wait(tasks.values(), timeout=5)
        assert not pending, (router.get_status(), decisions[-10:], {sid: b.generated for sid, b in backends.items()})
        outputs = await asyncio.gather(*tasks.values())
        by_id = dict(zip(tasks, outputs, strict=True))
        for rid in ("a", "b", "c"):
            output = by_id[rid]
            assert output.token_ids == [10, 11, 12, 13, 14, 15]
            segments = output.extra_fields["trajectory_segments"]
            assert segments[0]["weight_version"] == 0
            assert segments[-1]["weight_version"] == 1
            assert output.extra_fields["forced_prefill_tokens"] > 0
            old_count = 4 if rid == "b" else 2
            assert output.log_probs == [-0.5] * old_count + [-0.6] * (6 - old_count)
        assert by_id["b"].extra_fields["trajectory_migration_counts"] == {"remote_prefix": 1, "recompute": 1}
        assert by_id["new"].extra_fields["min_global_steps"] == 1
        assert by_id["new"].extra_fields["max_global_steps"] == 1
        assert router._joint.trajectories == {}
        assert router._joint.reservations == {}
        assert router._pending_migrations == {}
        assert router.get_total_inflight() == 0
        assert all(router._joint.usage(sid) == (0, 0) for sid in backends)
        assert all(not backend.accepted for backend in backends.values())
    finally:
        evacuate.set()
        resume.set()
        manager.release_first_update.set()
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        await scheduler.close(drain=False)
