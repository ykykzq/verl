# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest

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
