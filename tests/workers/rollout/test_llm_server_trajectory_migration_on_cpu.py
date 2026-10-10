# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from verl import DataProto
from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _TransferServer:
    def __init__(self, role, *, accepted=True):
        self.calls = []
        self.accepted = accepted
        if role == "source":
            self.prepare_trajectory_migration = _RemoteMethod(self._prepare)
        else:
            self.accept_trajectory_migration = _RemoteMethod(self._accept)
            self.discard_trajectory_migration = _RemoteMethod(self._discard)

    def _prepare(self, **kwargs):
        self.calls.append(kwargs)
        return {"backend": "remote_prefix", "prefix_digest": "digest", "ticket": "ready"}

    def _accept(self, **kwargs):
        self.calls.append(kwargs)
        return {"accepted": self.accepted, "reason": "rejected for test"}

    def _discard(self, **kwargs):
        self.calls.append(kwargs)
        return {"discarded": True}


class _LoadBalancer:
    def __init__(self, source, target):
        self.plan_trajectory_migration = _RemoteMethod(self._plan)
        self.commit_trajectory_migration = _RemoteMethod(self._commit)
        self.cancel_trajectory_migration = _RemoteMethod(self._cancel)
        self.source = source
        self.target = target
        self.commits = []

    def _plan(self, **kwargs):
        return {
            "migrate": True,
            "decision_id": "decision-1",
            "source_server_id": "s0",
            "target_server_id": "s1",
            "source_server": self.source,
            "target_server": self.target,
            "target_metadata": {"weight_version": 7},
            "mode": "remote_prefix",
            "source_version": 7,
            "target_version": 7,
        }

    def _commit(self, **kwargs):
        self.commits.append(kwargs)
        return {"migrated": True}

    def _cancel(self, **kwargs):
        raise AssertionError(f"migration should not be cancelled: {kwargs}")


class _RollbackLoadBalancer(_LoadBalancer):
    def __init__(self, source, target):
        super().__init__(source, target)
        self.cancels = []

    def _cancel(self, **kwargs):
        self.cancels.append(kwargs)
        return {"cancelled": True}


@pytest.mark.asyncio
async def test_chunk_checkpoint_transfers_trajectory_and_continues(monkeypatch):
    segments = iter(
        [
            TokenOutput(
                token_ids=[10, 11],
                stop_reason="length",
                extra_fields={
                    "global_steps": 7,
                    "migration_checkpoint": True,
                    llm_server._ROUTE_SERVER_ID_FIELD: "s0",
                },
            ),
            TokenOutput(
                token_ids=[12],
                stop_reason="completed",
                extra_fields={
                    "global_steps": 7,
                    "migration_checkpoint": False,
                    llm_server._ROUTE_SERVER_ID_FIELD: "s1",
                },
            ),
        ]
    )
    prompts = []
    budgets = []

    async def fake_generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        prompts.append(list(prompt_ids))
        budgets.append(sampling_params["max_tokens"])
        return next(segments)

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake_generate)
    source = _TransferServer("source")
    target = _TransferServer("target")
    load_balancer = _LoadBalancer(source, target)
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                name="rtp_llm",
                response_length=8,
                trajectory_migration=SimpleNamespace(
                    enabled=True,
                    checkpoint_tokens=2,
                    transfer_timeout_s=1.0,
                    kv_transfer_backend="remote_prefix",
                ),
            )
        )
    )
    client = FullyAsyncLLMServerClient(config=config, load_balancer_handle=load_balancer)

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    assert prompts == [[1, 2], [1, 2, 10, 11]]
    assert budgets == [2, 2]
    assert output.token_ids == [10, 11, 12]
    assert load_balancer.commits == [{"request_id": "trajectory-1", "decision_id": "decision-1"}]
    transferred = source.calls[0]["trajectory_state"]
    assert transferred == {
        "request_id": "trajectory-1",
        "prompt_ids": [1, 2],
        "generated_token_ids": [10, 11],
        "checkpoint_index": 1,
        "sampling_params": {"max_tokens": 6},
    }
    assert target.calls[0]["trajectory_state"] == transferred
    assert output.extra_fields["trajectory_migrations"][0]["target_server_id"] == "s1"
    assert output.extra_fields["trajectory_migration_counts"] == {"remote_prefix": 1, "recompute": 0}
    assert output.extra_fields["trajectory_migration_replans"] == 0


@pytest.mark.asyncio
async def test_rejected_transfer_discards_target_state_and_releases_reservation():
    source = _TransferServer("source")
    target = _TransferServer("target", accepted=False)
    load_balancer = _RollbackLoadBalancer(source, target)
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                name="rtp_llm",
                trajectory_migration=SimpleNamespace(
                    enabled=True,
                    transfer_timeout_s=1.0,
                    kv_transfer_backend="remote_prefix",
                ),
            )
        )
    )
    client = FullyAsyncLLMServerClient(config=config, load_balancer_handle=load_balancer)

    result = await client._try_migrate_trajectory(
        request_id="trajectory-1",
        source_server_id="s0",
        prompt_ids=[1, 2],
        final_output=TokenOutput(token_ids=[10, 11]),
        checkpoint_index=1,
        source_weight_version=7,
        sampling_params={"max_tokens": 6},
    )

    assert result is None
    assert target.calls[-1] == {"request_id": "trajectory-1", "prefix_digest": "digest"}
    assert load_balancer.cancels == [{"request_id": "trajectory-1", "decision_id": "decision-1"}]


def _migration_config():
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                name="rtp_llm",
                response_length=8,
                trajectory_migration=SimpleNamespace(
                    enabled=True,
                    allow_cross_version_recompute=True,
                    checkpoint_tokens=2,
                    transfer_timeout_s=1.0,
                    kv_transfer_backend="remote_prefix",
                ),
            )
        )
    )


class _RecomputeLoadBalancer(_RollbackLoadBalancer):
    def _plan(self, **kwargs):
        return {
            **super()._plan(**kwargs),
            "mode": "recompute",
            "source_version": 7,
            "target_version": 8,
            "source_metadata": {"model_id": "model-a", "weight_version": 7},
            "target_metadata": {"model_id": "model-a", "weight_version": 8},
        }


@pytest.mark.asyncio
async def test_recompute_transfers_tokens_without_source_prepare_and_carries_ticket_to_enqueue(monkeypatch):
    calls = []

    async def generate_segment(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        calls.append({"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params), **kwargs})
        if len(calls) == 1:
            return TokenOutput(
                token_ids=[10, 11],
                stop_reason="length",
                extra_fields={"global_steps": 7, "migration_checkpoint": True, "_verl_route_server_id": "s0"},
            )
        return TokenOutput(
            token_ids=[12],
            stop_reason="completed",
            extra_fields={"global_steps": 8, "_verl_route_server_id": "s1"},
        )

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    source = _TransferServer("source")
    target = _TransferServer("target")
    load_balancer = _RecomputeLoadBalancer(source, target)
    client = FullyAsyncLLMServerClient(config=_migration_config(), load_balancer_handle=load_balancer)

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    assert source.calls == []
    expected_digest = hashlib.sha256(b"".join(token.to_bytes(8, "little", signed=True) for token in [1, 2, 10, 11]))
    ticket = target.calls[0]["ticket"]
    assert ticket == {
        "backend": "recompute",
        "mode": "recompute",
        "model_id": "model-a",
        "request_id": "trajectory-1",
        "prefix_tokens": 4,
        "prefix_digest": expected_digest.hexdigest(),
        "source_version": 7,
        "target_version": 8,
    }
    assert target.calls[0]["trajectory_state"] == {
        "request_id": "trajectory-1",
        "prompt_ids": [1, 2],
        "generated_token_ids": [10, 11],
        "checkpoint_index": 1,
        "sampling_params": {"max_tokens": 6},
    }
    assert calls[1]["migration_ticket"] == ticket
    assert calls[1]["prompt_ids"] == [1, 2, 10, 11]
    assert output.extra_fields["trajectory_migrations"] == [
        {
            "source_server_id": "s0",
            "target_server_id": "s1",
            "checkpoint_index": 1,
            "prefix_tokens": 4,
            "backend": "recompute",
            "mode": "recompute",
            "source_version": 7,
            "target_version": 8,
        }
    ]
    assert output.extra_fields["min_global_steps"] == 7
    assert output.extra_fields["max_global_steps"] == 8
    assert output.extra_fields["trajectory_migration_counts"] == {"remote_prefix": 0, "recompute": 1}
    assert output.extra_fields["trajectory_migration_replans"] == 0
    assert load_balancer.commits == [{"request_id": "trajectory-1", "decision_id": "decision-1"}]


@pytest.mark.asyncio
async def test_recompute_target_rejection_never_commits_and_cancels_reservation():
    source = _TransferServer("source")
    target = _TransferServer("target", accepted=False)
    load_balancer = _RecomputeLoadBalancer(source, target)
    client = FullyAsyncLLMServerClient(config=_migration_config(), load_balancer_handle=load_balancer)

    result = await client._try_migrate_trajectory(
        request_id="trajectory-1",
        source_server_id="s0",
        prompt_ids=[1, 2],
        final_output=TokenOutput(token_ids=[10, 11]),
        checkpoint_index=1,
        source_weight_version=7,
        sampling_params={"max_tokens": 6},
    )

    assert result is None
    assert source.calls == []
    assert load_balancer.commits == []
    assert target.calls[-1]["prefix_digest"] == target.calls[0]["ticket"]["prefix_digest"]
    assert load_balancer.cancels == [{"request_id": "trajectory-1", "decision_id": "decision-1"}]


@pytest.mark.asyncio
async def test_cancellation_after_acceptance_discards_ticket_and_reservation():
    source = _TransferServer("source")
    target = _TransferServer("target")
    load_balancer = _RecomputeLoadBalancer(source, target)

    def cancel_commit(**kwargs):
        raise asyncio.CancelledError

    load_balancer.commit_trajectory_migration = _RemoteMethod(cancel_commit)
    client = FullyAsyncLLMServerClient(config=_migration_config(), load_balancer_handle=load_balancer)

    with pytest.raises(asyncio.CancelledError):
        await client._try_migrate_trajectory(
            request_id="trajectory-1",
            source_server_id="s0",
            prompt_ids=[1, 2],
            final_output=TokenOutput(token_ids=[10, 11]),
            checkpoint_index=1,
            source_weight_version=7,
            sampling_params={"max_tokens": 6},
        )

    assert source.calls == []
    assert target.calls[-1]["prefix_digest"] == target.calls[0]["ticket"]["prefix_digest"]
    assert load_balancer.cancels == [{"request_id": "trajectory-1", "decision_id": "decision-1"}]


@pytest.mark.asyncio
async def test_initial_metadata_is_collected_for_fine_grained_updates_without_migration(monkeypatch):
    metadata = {"model_id": "model-a", "weight_version": 7, "replica_rank": 0}
    handle = SimpleNamespace(get_trajectory_migration_capabilities=_RemoteMethod(lambda: metadata))

    class Replica:
        def __init__(self, **kwargs):
            self._server_handle = handle
            self._server_address = "s0"

        async def init_standalone(self):
            pass

    manager = object.__new__(llm_server.LLMServerManager)
    manager.start_rank = 0
    manager.worker_group = None
    manager.rollout_replica_class = Replica
    manager.model_config = SimpleNamespace()
    manager.rollout_config = SimpleNamespace(
        name="rtp_llm",
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        n_gpus_per_node=1,
        nnodes=1,
        trajectory_migration=SimpleNamespace(enabled=False),
        fine_grained_weight_update=SimpleNamespace(enabled=True),
        prometheus=SimpleNamespace(enable=False),
        disable_log_stats=True,
    )
    monkeypatch.setattr(llm_server.RLInsightLogger, "enabled", lambda: False)

    await manager._initialize_llm_servers()

    assert manager.server_metadata == {"s0": metadata}


@pytest.mark.asyncio
async def test_stale_acceptance_replans_and_waits_for_an_eligible_target(monkeypatch):
    calls = []
    plans = []
    sleeps = []
    source = _TransferServer("source")
    target = _TransferServer("target")
    load_balancer = _RecomputeLoadBalancer(source, target)

    def plan(**kwargs):
        plans.append(kwargs)
        if len(plans) == 2:
            return {"migrate": False, "reason": "target is updating"}
        decision = load_balancer._plan(**kwargs)
        if len(plans) > 2:
            decision["target_version"] = 9
            decision["target_metadata"]["weight_version"] = 9
        return decision

    async def generate_segment(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return TokenOutput(
                token_ids=[10, 11],
                stop_reason="length",
                extra_fields={"global_steps": 7, "migration_checkpoint": True, "_verl_route_server_id": "s0"},
            )
        if len(calls) == 2:
            return TokenOutput(
                token_ids=[],
                stop_reason="aborted",
                extra_fields={"global_steps": 9, "migration_rejected": True, "_verl_route_server_id": "s1"},
            )
        return TokenOutput(
            token_ids=[12],
            stop_reason="completed",
            extra_fields={"global_steps": 9, "_verl_route_server_id": "s1"},
        )

    async def sleep(seconds):
        sleeps.append(seconds)

    load_balancer.plan_trajectory_migration = _RemoteMethod(plan)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    monkeypatch.setattr(llm_server.asyncio, "sleep", sleep)
    client = FullyAsyncLLMServerClient(config=_migration_config(), load_balancer_handle=load_balancer)

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    assert len(plans) == 3
    assert "migration_required" not in plans[0]["trajectory"]
    assert all(plan["trajectory"]["migration_required"] for plan in plans[1:])
    assert all(plan["source_server_id"] == "s0" and plan["source_metadata"]["weight_version"] == 7 for plan in plans)
    assert all(plan["trajectory"]["checkpoint_index"] == 1 for plan in plans)
    assert calls[1]["migration_ticket"]["target_version"] == 8
    assert calls[2]["migration_ticket"]["target_version"] == 9
    assert sleeps == [1]
    assert output.token_ids == [10, 11, 12]
    migrations = output.extra_fields["trajectory_migrations"]
    assert [(entry["source_version"], entry["target_version"]) for entry in migrations] == [(7, 9)]
    assert output.extra_fields["trajectory_migration_counts"] == {"remote_prefix": 0, "recompute": 1}
    assert output.extra_fields["trajectory_migration_replans"] == 1


@pytest.mark.asyncio
async def test_aborted_continuation_guards_source_version_and_replans_after_router_reroutes(monkeypatch):
    calls = []
    plans = []
    source = _TransferServer("source")
    target = _TransferServer("target")
    load_balancer = _RecomputeLoadBalancer(source, target)

    def plan(**kwargs):
        plans.append(kwargs)
        return load_balancer._plan(**kwargs)

    async def generate_segment(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return TokenOutput(
                token_ids=[10],
                stop_reason="aborted",
                extra_fields={"global_steps": 7, "_verl_route_server_id": "s0"},
            )
        if len(calls) == 2:
            return TokenOutput(
                token_ids=[],
                stop_reason="aborted",
                extra_fields={"global_steps": 8, "migration_rejected": True, "_verl_route_server_id": "s1"},
            )
        return TokenOutput(
            token_ids=[11],
            stop_reason="completed",
            extra_fields={"global_steps": 8, "_verl_route_server_id": "s1"},
        )

    async def sleep(seconds):
        pass

    load_balancer.plan_trajectory_migration = _RemoteMethod(plan)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    monkeypatch.setattr(llm_server.asyncio, "sleep", sleep)
    client = FullyAsyncLLMServerClient(config=_migration_config(), load_balancer_handle=load_balancer)

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    assert calls[1]["expected_source_version"] == 7
    assert "migration_ticket" not in calls[1]
    assert calls[2]["migration_ticket"]["source_version"] == 7
    assert calls[2]["migration_ticket"]["target_version"] == 8
    assert plans[0]["source_server_id"] == "s0"
    assert plans[0]["source_metadata"]["weight_version"] == 7
    assert output.token_ids == [10, 11]
    assert output.extra_fields["trajectory_migrations"][0]["mode"] == "recompute"


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_migration_counters_are_zero_without_movement_and_absent_when_disabled(monkeypatch, enabled):
    async def generate_segment(self, request_id, **kwargs):
        return TokenOutput(token_ids=[10], stop_reason="completed", extra_fields={"global_steps": 0})

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    config = _migration_config()
    config.actor_rollout_ref.rollout.trajectory_migration.enabled = enabled
    client = FullyAsyncLLMServerClient(config=config)

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    if enabled:
        assert output.extra_fields["trajectory_migration_counts"] == {"remote_prefix": 0, "recompute": 0}
        assert output.extra_fields["trajectory_migration_replans"] == 0
        assert output.extra_fields["forced_prefill_tokens"] == 0
    else:
        assert "trajectory_migration_counts" not in output.extra_fields
        assert "trajectory_migration_replans" not in output.extra_fields
        assert "forced_prefill_tokens" not in output.extra_fields


@pytest.mark.asyncio
async def test_forced_prefill_tokens_are_aggregated_across_recompute_segments(monkeypatch):
    segments = iter(
        [
            TokenOutput(
                token_ids=[10, 11],
                stop_reason="length",
                extra_fields={"global_steps": 7, "migration_checkpoint": True, "_verl_route_server_id": "s0"},
            ),
            TokenOutput(
                token_ids=[12, 13],
                stop_reason="length",
                extra_fields={
                    "global_steps": 8,
                    "migration_checkpoint": True,
                    "_verl_route_server_id": "s1",
                    "forced_prefill_tokens": 4,
                },
            ),
            TokenOutput(
                token_ids=[14],
                stop_reason="completed",
                extra_fields={"global_steps": 9, "_verl_route_server_id": "s1", "forced_prefill_tokens": 6},
            ),
        ]
    )

    async def generate_segment(self, request_id, **kwargs):
        return next(segments)

    source = _TransferServer("source")
    target = _TransferServer("target")
    load_balancer = _RecomputeLoadBalancer(source, target)

    def plan(**kwargs):
        decision = load_balancer._plan(**kwargs)
        source_version = kwargs["source_metadata"]["weight_version"]
        decision["source_version"] = source_version
        decision["target_version"] = source_version + 1
        decision["source_metadata"]["weight_version"] = source_version
        decision["target_metadata"]["weight_version"] = source_version + 1
        return decision

    load_balancer.plan_trajectory_migration = _RemoteMethod(plan)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    client = FullyAsyncLLMServerClient(config=_migration_config(), load_balancer_handle=load_balancer)

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    assert output.extra_fields["forced_prefill_tokens"] == 10
    assert output.extra_fields["trajectory_migration_counts"] == {"remote_prefix": 0, "recompute": 2}

    async def ordinary_segment(self, request_id, **kwargs):
        return TokenOutput(token_ids=[15], stop_reason="completed", extra_fields={"global_steps": 9})

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", ordinary_segment)
    ordinary = await client.generate(request_id="trajectory-2", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})
    batches = []
    for generated in (ordinary, output):
        fields = {}
        for key, value in generated.extra_fields.items():
            fields[key] = np.empty(1, dtype=object)
            fields[key][0] = value
        batches.append(DataProto.from_dict(non_tensors=fields))
    combined = DataProto.concat(batches)
    assert combined.non_tensor_batch["forced_prefill_tokens"].tolist() == [0, 10]


@pytest.mark.asyncio
async def test_rejected_target_can_resume_unchanged_source_without_transferring_kv(monkeypatch):
    calls = []
    plans = []
    source = _TransferServer("source")
    target = _TransferServer("target")
    load_balancer = _RecomputeLoadBalancer(source, target)

    def plan(**kwargs):
        plans.append(kwargs)
        decision = load_balancer._plan(**kwargs)
        if len(plans) > 1:
            decision.pop("mode")
            decision.update(resume_source=True, target_server_id="s0", target_server=source, target_version=7)
            decision["target_metadata"]["weight_version"] = 7
        return decision

    async def generate_segment(self, request_id, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return TokenOutput(
                token_ids=[10, 11],
                stop_reason="length",
                extra_fields={"global_steps": 7, "migration_checkpoint": True, "_verl_route_server_id": "s0"},
            )
        if len(calls) == 2:
            return TokenOutput(
                token_ids=[],
                stop_reason="aborted",
                extra_fields={"global_steps": 7, "migration_rejected": True, "_verl_route_server_id": "s0"},
            )
        return TokenOutput(
            token_ids=[12], stop_reason="completed", extra_fields={"global_steps": 7, "_verl_route_server_id": "s0"}
        )

    load_balancer.plan_trajectory_migration = _RemoteMethod(plan)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    client = FullyAsyncLLMServerClient(config=_migration_config(), load_balancer_handle=load_balancer)

    output = await asyncio.wait_for(
        client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8}), timeout=0.2
    )

    assert len(plans) == len(load_balancer.commits) == 2
    assert source.calls == []
    assert "migration_ticket" not in calls[2]
    assert calls[2]["expected_source_version"] == 7
    assert output.token_ids == [10, 11, 12]
    assert output.extra_fields["trajectory_migrations"] == []


@pytest.mark.asyncio
async def test_legacy_partial_resume_does_not_guard_versions_when_new_features_are_disabled(monkeypatch):
    calls = []

    async def generate_segment(self, request_id, **kwargs):
        calls.append(kwargs)
        return TokenOutput(
            token_ids=[len(calls)],
            stop_reason="aborted" if len(calls) == 1 else "completed",
            extra_fields={"global_steps": 6 + len(calls), "_verl_route_server_id": "s0"},
        )

    async def sleep(seconds):
        pass

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    monkeypatch.setattr(llm_server.asyncio, "sleep", sleep)
    config = _migration_config()
    config.actor_rollout_ref.rollout.trajectory_migration.allow_cross_version_recompute = False
    client = FullyAsyncLLMServerClient(config=config)

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    assert output.token_ids == [1, 2]
    assert "expected_source_version" not in calls[1]


@pytest.mark.asyncio
async def test_stale_legacy_ticket_resumes_without_replanning_when_new_features_are_disabled(monkeypatch):
    calls = []
    plans = []
    source = _TransferServer("source")
    target = _TransferServer("target")
    load_balancer = _LoadBalancer(source, target)

    def plan(**kwargs):
        plans.append(kwargs)
        return load_balancer._plan(**kwargs)

    async def generate_segment(self, request_id, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return TokenOutput(
                token_ids=[10, 11],
                stop_reason="length",
                extra_fields={"global_steps": 7, "migration_checkpoint": True, "_verl_route_server_id": "s0"},
            )
        if len(calls) == 2:
            return TokenOutput(
                token_ids=[],
                stop_reason="aborted",
                extra_fields={"global_steps": 8, "migration_rejected": True, "_verl_route_server_id": "s1"},
            )
        return TokenOutput(
            token_ids=[12], stop_reason="completed", extra_fields={"global_steps": 8, "_verl_route_server_id": "s1"}
        )

    load_balancer.plan_trajectory_migration = _RemoteMethod(plan)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    config = _migration_config()
    config.actor_rollout_ref.rollout.trajectory_migration.allow_cross_version_recompute = False
    client = FullyAsyncLLMServerClient(config=config, load_balancer_handle=load_balancer)

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    assert len(plans) == 1
    assert "expected_source_version" not in calls[2]
    assert "migration_ticket" not in calls[2]
    assert output.extra_fields["trajectory_migrations"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["UPDATING", "QUARANTINED", None])
async def test_fine_grained_standalone_acquisition_observes_replica_lifecycle(monkeypatch, state):
    attempts = []
    sleeps = []

    async def acquire(self, request_id, **kwargs):
        attempts.append(request_id)
        if len(attempts) == 1:
            raise RuntimeError("No available servers in load balancer")
        return "s0", "server-handle"

    async def sleep(seconds):
        sleeps.append(seconds)

    config = _migration_config()
    config.actor_rollout_ref.rollout.fine_grained_weight_update = SimpleNamespace(enabled=True)
    states = {} if state is None else {"s0": {"lifecycle_state": state}}
    load_balancer = SimpleNamespace(get_replica_states=_RemoteMethod(lambda: states))
    monkeypatch.setattr(llm_server.LLMServerClient, "_acquire_server", acquire)
    monkeypatch.setattr(llm_server.asyncio, "sleep", sleep)
    client = FullyAsyncLLMServerClient(config=config, load_balancer_handle=load_balancer)

    if state == "UPDATING":
        assert await client._acquire_server("trajectory-1") == ("s0", "server-handle")
        assert sleeps == [1]
    else:
        reason = "quarantined.*recover" if state == "QUARANTINED" else "No available servers"
        with pytest.raises(RuntimeError, match=reason):
            await client._acquire_server("trajectory-1")
        assert sleeps == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "versions,expected_min,expected_max",
    [([8, 7], 7, 8), ([7, 9, 8], 7, 9), ([None, 0], 0, 0), ([None, None], None, None)],
)
async def test_version_extrema_follow_token_producing_segments(monkeypatch, versions, expected_min, expected_max):
    pending_versions = iter(enumerate(versions))

    async def generate_segment(self, request_id, **kwargs):
        index, version = next(pending_versions)
        return TokenOutput(
            token_ids=[10 + index],
            stop_reason="completed" if index == len(versions) - 1 else "aborted",
            extra_fields={"global_steps": version, "_verl_route_server_id": "s0"},
        )

    async def sleep(seconds):
        pass

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    monkeypatch.setattr(llm_server.asyncio, "sleep", sleep)
    client = FullyAsyncLLMServerClient(config=_migration_config())

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    assert output.extra_fields["min_global_steps"] == expected_min
    assert output.extra_fields["max_global_steps"] == expected_max
    assert output.extra_fields["global_steps"] == versions[-1]


@pytest.mark.asyncio
async def test_empty_segments_do_not_change_trajectory_versions(monkeypatch):
    segments = iter(
        [
            TokenOutput(token_ids=[10], stop_reason="aborted", extra_fields={"global_steps": 7}),
            TokenOutput(token_ids=[], stop_reason="aborted", extra_fields={"global_steps": 99}),
            TokenOutput(token_ids=[11], stop_reason="aborted", extra_fields={"global_steps": 8}),
            TokenOutput(token_ids=[], stop_reason="completed", extra_fields={"global_steps": 100}),
        ]
    )

    async def generate_segment(self, request_id, **kwargs):
        return next(segments)

    async def sleep(seconds):
        pass

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    monkeypatch.setattr(llm_server.asyncio, "sleep", sleep)
    client = FullyAsyncLLMServerClient(config=_migration_config())

    output = await client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8})

    assert output.extra_fields["global_steps"] == 8
    assert output.extra_fields["min_global_steps"] == 7
    assert output.extra_fields["max_global_steps"] == 8


@pytest.mark.asyncio
async def test_required_replan_reports_all_quarantined_replicas(monkeypatch):
    calls = []
    plans = []
    load_balancer = _RecomputeLoadBalancer(_TransferServer("source"), _TransferServer("target"))

    def plan(**kwargs):
        plans.append(kwargs)
        return load_balancer._plan(**kwargs) if len(plans) == 1 else {"migrate": False}

    async def generate_segment(self, request_id, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return TokenOutput(
                token_ids=[10],
                stop_reason="length",
                extra_fields={"global_steps": 7, "migration_checkpoint": True, "_verl_route_server_id": "s0"},
            )
        return TokenOutput(
            token_ids=[],
            stop_reason="aborted",
            extra_fields={"global_steps": 8, "migration_rejected": True, "_verl_route_server_id": "s1"},
        )

    load_balancer.plan_trajectory_migration = _RemoteMethod(plan)
    load_balancer.get_replica_states = _RemoteMethod(lambda: {"s0": {"lifecycle_state": "QUARANTINED"}})
    config = _migration_config()
    config.actor_rollout_ref.rollout.fine_grained_weight_update = SimpleNamespace(enabled=True)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate_segment)
    client = FullyAsyncLLMServerClient(config=config, load_balancer_handle=load_balancer)

    with pytest.raises(RuntimeError, match="quarantined.*recover"):
        await asyncio.wait_for(
            client.generate(request_id="trajectory-1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8}),
            timeout=0.2,
        )
