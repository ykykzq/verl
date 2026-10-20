# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import asyncio
import inspect
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout import llm_server, router
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerManager
from verl.workers.rollout.replica import TokenOutput


class Remote:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args, **kwargs):
        result = self.fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result


class Registry:
    def __init__(self):
        self.events = []
        self.active = {}
        self.register_trajectory = Remote(self.register)
        self.report_trajectory_progress = Remote(self.report)
        self.finish_trajectory = Remote(self.finish)
        self.abort_trajectory = Remote(self.abort)
        self.plan_trajectory_migration = Remote(self.plan)
        self.commit_trajectory_migration = Remote(lambda **kwargs: self.events.append(("commit", kwargs["request_id"])))

    def register(self, request_id, prompt_tokens, max_new_tokens):
        assert request_id not in self.active
        self.active[request_id] = (prompt_tokens, max_new_tokens)
        self.events.append(("register", request_id, prompt_tokens, max_new_tokens))

    def report(self, **kwargs):
        assert kwargs["request_id"] in self.active
        self.events.append(("report", kwargs))
        return {}

    def finish(self, request_id):
        self.active.pop(request_id)
        self.events.append(("finish", request_id))

    def abort(self, request_id):
        self.events.append(("abort", request_id))
        return {"aborted": True}

    def plan(self, **kwargs):
        required = kwargs["trajectory"].get("migration_required", False)
        if not required:
            assert self.events[-1][0] == "report"
        self.events.append(("plan", kwargs["request_id"]))
        if required:
            return {"migrate": True, "resume_source": True, "decision_id": "resume"}
        return {"migrate": False}


def config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "rtp_llm",
                    "response_length": 8,
                    "fine_grained_weight_update": {"enabled": True},
                    "weight_publication": {"joint_scheduling": {"enabled": True}},
                    "trajectory_migration": {"enabled": True, "checkpoint_tokens": 2},
                }
            }
        }
    )


def segment(tokens, version, checkpoint=False, stop_reason="completed"):
    return TokenOutput(
        token_ids=tokens,
        log_probs=[-0.5] * len(tokens),
        stop_reason="length" if checkpoint else stop_reason,
        extra_fields={
            "global_steps": version,
            "migration_checkpoint": checkpoint,
            llm_server._ROUTE_SERVER_ID_FIELD: "s0",
        },
    )


@pytest.mark.asyncio
async def test_registration_survives_checkpoints_and_reports_before_migration(monkeypatch):
    registry = Registry()
    responses = iter([segment([10, 11], 1, checkpoint=True), segment([12], 2)])

    async def generate(self, request_id, **kwargs):
        assert registry.active == {"t": (2, 8)}
        return next(responses)

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    client = FullyAsyncLLMServerClient(config(), registry)
    output = await client.generate("t", prompt_ids=[1, 2], sampling_params={})

    assert [event[0] for event in registry.events] == ["register", "report", "plan", "report", "finish"]
    assert registry.events[1][1] == {
        "request_id": "t",
        "server_id": "s0",
        "generated_tokens": 2,
        "weight_version": 1,
        "prefix_ids": [1, 2, 10, 11],
    }
    assert registry.events[3][1]["generated_tokens"] == 3
    assert registry.events[3][1]["prefix_ids"] == [1, 2, 10, 11, 12]
    assert registry.active == {}
    assert output.log_probs == [-0.5, -0.5, -0.5]
    assert output.extra_fields["trajectory_segments"] == [
        {"start": 0, "end": 2, "weight_version": 1, "server_id": "s0"},
        {"start": 2, "end": 3, "weight_version": 2, "server_id": "s0"},
    ]


@pytest.mark.asyncio
async def test_partial_abort_reports_prefix_and_finishes_on_error(monkeypatch):
    registry = Registry()
    calls = 0

    async def generate(self, request_id, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return segment([10], 1, stop_reason="aborted")
        raise RuntimeError("backend failed")

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    client = FullyAsyncLLMServerClient(config(), registry)
    with pytest.raises(RuntimeError, match="backend failed"):
        await client.generate("t", prompt_ids=[1], sampling_params={"max_new_tokens": 4})
    assert registry.events[0] == ("register", "t", 1, 4)
    assert registry.events[1][1]["prefix_ids"] == [1, 10]
    assert registry.events[-1] == ("finish", "t")
    assert registry.active == {}


@pytest.mark.asyncio
async def test_cancel_during_registration_settles_before_releasing(monkeypatch):
    registry = Registry()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def register(**kwargs):
        entered.set()
        await release.wait()
        registry.register(**kwargs)

    async def generate(*args, **kwargs):
        pytest.fail("cancelled registration must not start generation")

    registry.register_trajectory = Remote(register)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    client = FullyAsyncLLMServerClient(config(), registry)
    task = asyncio.create_task(client.generate("t", prompt_ids=[1], sampling_params={}))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert registry.events == [("register", "t", 1, 8), ("finish", "t")]
    assert registry.active == {}


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_finish_rpc(monkeypatch):
    registry = Registry()
    decoding = asyncio.Event()
    finishing = asyncio.Event()
    release = asyncio.Event()

    async def generate(*args, **kwargs):
        decoding.set()
        await asyncio.Event().wait()

    async def finish(request_id):
        finishing.set()
        await release.wait()
        registry.finish(request_id)

    registry.finish_trajectory = Remote(finish)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    client = FullyAsyncLLMServerClient(config(), registry)
    task = asyncio.create_task(client.generate("t", prompt_ids=[1], sampling_params={}))
    await decoding.wait()
    task.cancel()
    await finishing.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert registry.active == {}


@pytest.mark.asyncio
async def test_cancel_waits_for_backend_abort_before_releasing_capacity(monkeypatch):
    registry = Registry()
    decoding = asyncio.Event()
    aborting = asyncio.Event()
    release = asyncio.Event()

    async def generate(*args, **kwargs):
        decoding.set()
        await asyncio.Event().wait()

    async def abort(request_id):
        aborting.set()
        await release.wait()
        registry.events.append(("abort", request_id))
        return {"aborted": True}

    registry.abort_trajectory = Remote(abort)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    client = FullyAsyncLLMServerClient(config(), registry)
    task = asyncio.create_task(client.generate("t", prompt_ids=[1], sampling_params={}))
    await decoding.wait()
    task.cancel()
    await asyncio.wait_for(aborting.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    assert registry.active == {"t": (1, 8)}
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert registry.events[-2:] == [("abort", "t"), ("finish", "t")]
    assert registry.active == {}


@pytest.mark.asyncio
async def test_failed_backend_abort_preserves_reservation(monkeypatch):
    registry = Registry()

    async def generate(*args, **kwargs):
        raise RuntimeError("generation failure")

    def abort(request_id):
        raise RuntimeError("abort failed")

    registry.abort_trajectory = Remote(abort)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    with pytest.raises(RuntimeError, match="generation failure"):
        await FullyAsyncLLMServerClient(config(), registry).generate("t", prompt_ids=[1], sampling_params={})
    assert registry.active == {"t": (1, 8)}
    assert registry.events == [("register", "t", 1, 8)]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [{"aborted": False, "reason": "abort timed out"}, {}, None])
async def test_uncertain_abort_response_preserves_reservation(monkeypatch, result):
    registry = Registry()

    async def generate(*args, **kwargs):
        raise RuntimeError("generation failure")

    registry.abort_trajectory = Remote(lambda request_id: result)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    with pytest.raises(RuntimeError, match="generation failure"):
        await FullyAsyncLLMServerClient(config(), registry).generate("t", prompt_ids=[1], sampling_params={})
    assert registry.active == {"t": (1, 8)}
    assert registry.events == [("register", "t", 1, 8)]


@pytest.mark.asyncio
async def test_unresponsive_abort_has_finite_cleanup_and_retains_capacity(monkeypatch):
    registry = Registry()
    decoding = asyncio.Event()

    async def generate(*args, **kwargs):
        decoding.set()
        await asyncio.Event().wait()

    async def abort(request_id):
        await asyncio.Event().wait()

    registry.abort_trajectory = Remote(abort)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    settings = config()
    settings.actor_rollout_ref.rollout.trajectory_migration.transfer_timeout_s = 0.02
    client = FullyAsyncLLMServerClient(settings, registry)
    task = asyncio.create_task(client.generate("t", prompt_ids=[1], sampling_params={}))
    await decoding.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert registry.active == {"t": (1, 8)}
    assert registry.events == [("register", "t", 1, 8)]


@pytest.mark.asyncio
async def test_quarantined_source_replans_without_rewriting_prefix_version(monkeypatch):
    registry = Registry()
    calls = 0

    async def generate(self, request_id, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return segment([10, 11], 1, checkpoint=True)
        if calls == 2:
            raise RuntimeError("Joint trajectory continuation requires migration")
        assert kwargs["prompt_ids"] == [1, 10, 11]
        return segment([12], 3)

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    output = await FullyAsyncLLMServerClient(config(), registry).generate("t", prompt_ids=[1], sampling_params={})
    assert calls == 3
    assert [entry[1]["weight_version"] for entry in registry.events if entry[0] == "report"] == [1, 3]
    assert output.extra_fields["min_global_steps"] == 1
    assert output.extra_fields["max_global_steps"] == 3
    assert [entry["weight_version"] for entry in output.extra_fields["trajectory_segments"]] == [1, 3]
    assert registry.active == {}


@pytest.mark.asyncio
async def test_failed_registration_does_not_release_an_existing_owner():
    registry = Registry()
    registry.active["t"] = (2, 6)

    def register(**kwargs):
        raise ValueError("trajectory already registered")

    registry.register_trajectory = Remote(register)
    with pytest.raises(ValueError, match="already registered"):
        await FullyAsyncLLMServerClient(config(), registry).generate("t", prompt_ids=[1], sampling_params={})
    assert registry.active == {"t": (2, 6)}
    assert registry.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", [False, True])
async def test_recompute_ticket_preserves_explicit_recovery_authorization(recovery):
    registry = Registry()
    tickets = []

    def accept(ticket, trajectory_state):
        tickets.append(ticket)
        return {"accepted": True}

    target = SimpleNamespace(accept_trajectory_migration=Remote(accept))
    decision = {
        "migrate": True,
        "decision_id": "recovery",
        "mode": "recompute",
        "source_server_id": "s0",
        "target_server_id": "s1",
        "source_metadata": {"model_id": "model"},
        "target_metadata": {"weight_version": 1},
        "source_version": 1,
        "target_version": 1,
        "target_server": target,
    }
    if recovery:
        decision["recovery_recompute"] = True
    registry.plan_trajectory_migration = Remote(lambda **kwargs: decision)
    client = FullyAsyncLLMServerClient(config(), registry)
    result = await client._try_migrate_trajectory(
        "t", "s0", [1], TokenOutput(token_ids=[10]), 1, 1, {"max_tokens": 2}, migration_required=True
    )
    assert result["migration_ticket"] == tickets[0]
    if recovery:
        assert tickets[0]["recovery_recompute"] is True
    else:
        assert "recovery_recompute" not in tickets[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [-1, 1.5, float("inf"), True])
async def test_invalid_budget_is_rejected_before_registration(budget):
    registry = Registry()
    client = FullyAsyncLLMServerClient(config(), registry)
    with pytest.raises(ValueError, match="nonnegative integer"):
        await client.generate("t", prompt_ids=[1], sampling_params={"max_tokens": budget})
    assert registry.events == []


@pytest.mark.asyncio
async def test_zero_budget_finishes_without_admission_or_decode(monkeypatch):
    registry = Registry()

    async def generate(*args, **kwargs):
        pytest.fail("zero token budget must not call backend")

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    output = await FullyAsyncLLMServerClient(config(), registry).generate(
        "t", prompt_ids=[1], sampling_params={"max_tokens": 0}
    )
    assert output.token_ids == []
    assert output.stop_reason == "length"
    assert registry.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_manager_forwards_joint_configuration_only_when_enabled(monkeypatch, enabled):
    passed = {}

    def create(**kwargs):
        passed.update(kwargs)
        return "router"

    monkeypatch.setattr(router, "get_router_handle", create)
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.rollout_config = OmegaConf.create(
        {"weight_publication": {"joint_scheduling": {"enabled": enabled, "max_batch_size": 32}}}
    )
    manager.server_addresses = ["s0"]
    manager.server_handles = [None]
    manager._load_balancer_cls = None
    await manager._init_global_load_balancer()
    if enabled:
        assert passed["joint_scheduling_config"] == {"enabled": True, "max_batch_size": 32}
    else:
        assert "joint_scheduling_config" not in passed
