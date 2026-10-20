# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exercise cancellation across an actual queued Ray generation RPC."""

import asyncio

import pytest
import ray
from omegaconf import OmegaConf

from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput


@ray.remote(num_cpus=0)
class DelayedEnqueueServer:
    def __init__(self):
        self.entered = asyncio.Event()
        self.enqueue_allowed = asyncio.Event()
        self.abort_missed = asyncio.Event()
        self.stop = asyncio.Event()
        self.completed = False
        self.active = False

    async def generate(self, **kwargs):
        self.entered.set()
        await self.enqueue_allowed.wait()
        self.active = True
        await self.stop.wait()
        self.active = False
        self.completed = True
        return TokenOutput(token_ids=[], stop_reason="aborted", extra_fields={"global_steps": 1})

    async def abort_request(self, request_id):
        if not self.active:
            self.abort_missed.set()
            return {"aborted": False, "reason": "request not found"}
        self.stop.set()
        return {"aborted": True}

    async def wait_entered(self):
        await self.entered.wait()

    async def wait_abort_missed(self):
        await self.abort_missed.wait()

    def allow_enqueue(self):
        self.enqueue_allowed.set()

    def state(self):
        return {"active": self.active, "completed": self.completed}


@ray.remote(num_cpus=0)
class LifecycleRouter:
    def __init__(self, server):
        self.server = server
        self.registered = set()
        self.finished_after_backend = False

    def register_trajectory(self, request_id, prompt_tokens, max_new_tokens):
        self.registered.add(request_id)

    def require_acquire_fields(self):
        return []

    def require_release_fields(self):
        return []

    def acquire_server(self, request_id):
        assert request_id in self.registered
        return "s0", self.server

    def release_server(self, server_id):
        pass

    async def abort_trajectory(self, request_id):
        return await self.server.abort_request.remote(request_id)

    async def finish_trajectory(self, request_id):
        state = await self.server.state.remote()
        assert state == {"active": False, "completed": True}
        self.finished_after_backend = True
        self.registered.remove(request_id)

    def state(self):
        return {"registered": sorted(self.registered), "finished_after_backend": self.finished_after_backend}


@pytest.fixture(scope="module")
def ray_runtime():
    ray.init(address="local", num_cpus=1, include_dashboard=False, object_store_memory=100_000_000)
    yield
    ray.shutdown()


def test_cancelled_wait_does_not_release_before_delayed_enqueue_is_stopped(ray_runtime):
    server = DelayedEnqueueServer.remote()
    router = LifecycleRouter.remote(server)
    settings = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "rtp_llm",
                    "response_length": 8,
                    "fine_grained_weight_update": {"enabled": True},
                    "weight_publication": {"joint_scheduling": {"enabled": True}},
                    "trajectory_migration": {"enabled": True, "checkpoint_tokens": 2, "transfer_timeout_s": 5},
                }
            }
        }
    )

    async def run():
        client = FullyAsyncLLMServerClient(settings, router)
        task = asyncio.create_task(client.generate("t", prompt_ids=[1], sampling_params={}))
        await asyncio.wait_for(server.wait_entered.remote(), timeout=20)
        task.cancel()
        await asyncio.wait_for(server.wait_abort_missed.remote(), timeout=5)
        assert await router.state.remote() == {"registered": ["t"], "finished_after_backend": False}
        assert not task.done()
        task.cancel()
        await server.allow_enqueue.remote()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        assert await router.state.remote() == {"registered": [], "finished_after_backend": True}
        assert client._joint_generation_calls == {}

    try:
        asyncio.run(run())
    finally:
        ray.kill(router)
        ray.kill(server)
