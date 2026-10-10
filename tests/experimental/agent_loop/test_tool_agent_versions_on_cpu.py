# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.workers.rollout.replica import TokenOutput


async def _generate_turns(outputs):
    pending = iter(outputs)

    async def generate(**kwargs):
        return next(pending)

    async def merge(prompt_ids, response_ids, response_mask, response_logprobs, **kwargs):
        return SimpleNamespace(token_ids=prompt_ids + response_ids), response_mask + [1] * len(response_ids), None

    loop = SimpleNamespace(
        tool_parser=SimpleNamespace(stop_token_ids=[]),
        server_manager=SimpleNamespace(generate=generate),
        ct_merge_assistant_token=merge,
        response_length=128,
        max_assistant_turns=1,
    )
    data = AgentData([], None, None, None, None, {}, "trajectory", {})
    data.prompt_ids = [1, 2]
    for _ in outputs:
        await ToolAgentLoop._handle_generating_state(loop, data, {})
    return data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "versions,expected",
    [
        ([(8, 8, 8), (7, 7, 7)], (7, 8, 7)),
        ([(7, 7, 7), (8, 9, 9), (8, 8, 8)], (7, 9, 8)),
        ([(8, 8, 8), (0, 0, 0)], (0, 8, 0)),
        ([(None, None, None), (0, 0, 0)], (0, 0, 0)),
    ],
)
async def test_tool_turns_preserve_true_version_extrema(versions, expected):
    outputs = [
        TokenOutput(
            token_ids=[10 + index],
            num_preempted=1,
            extra_fields={
                "min_global_steps": low,
                "max_global_steps": high,
                "global_steps": last,
                "spec_num_draft_tokens": 2,
            },
        )
        for index, (low, high, last) in enumerate(versions)
    ]

    data = await _generate_turns(outputs)

    assert tuple(data.extra_fields[key] for key in ("min_global_steps", "max_global_steps", "global_steps")) == expected
    assert data.metrics["num_preempted"] == len(outputs)
    assert data.extra_fields["spec_num_draft_tokens"] == 2 * len(outputs)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_first", [False, True])
async def test_empty_tool_turn_does_not_contribute_weight_versions(empty_first):
    produced = TokenOutput(
        token_ids=[10], extra_fields={"min_global_steps": 7, "max_global_steps": 8, "global_steps": 8}
    )
    empty = TokenOutput(token_ids=[], extra_fields={"min_global_steps": 99, "max_global_steps": 99, "global_steps": 99})

    data = await _generate_turns([empty, produced] if empty_first else [produced, empty])

    assert tuple(data.extra_fields[key] for key in ("min_global_steps", "max_global_steps", "global_steps")) == (
        7,
        8,
        8,
    )


@pytest.mark.asyncio
async def test_tool_turns_aggregate_migration_history_and_metrics_without_mutating_outputs():
    first_history = [{"mode": "remote_prefix", "source_version": 8, "target_version": 8}]
    second_history = [{"mode": "recompute", "source_version": 8, "target_version": 7}]
    first = TokenOutput(
        token_ids=[10],
        extra_fields={
            "trajectory_migrations": first_history,
            "trajectory_migration_counts": {"remote_prefix": 1, "recompute": 0},
            "trajectory_migration_replans": 1,
            "forced_prefill_tokens": 0,
        },
    )
    second = TokenOutput(
        token_ids=[11],
        extra_fields={
            "trajectory_migrations": second_history,
            "trajectory_migration_counts": {"remote_prefix": 0, "recompute": 1},
            "trajectory_migration_replans": 2,
            "forced_prefill_tokens": 32,
        },
    )

    data = await _generate_turns([first, second])

    assert data.extra_fields["trajectory_migrations"] == first_history + second_history
    assert data.extra_fields["trajectory_migration_counts"] == {"remote_prefix": 1, "recompute": 1}
    assert data.extra_fields["trajectory_migration_replans"] == 3
    assert data.extra_fields["forced_prefill_tokens"] == 32
    assert first.extra_fields["trajectory_migrations"] == [
        {"mode": "remote_prefix", "source_version": 8, "target_version": 8}
    ]
    assert first.extra_fields["trajectory_migration_counts"] == {"remote_prefix": 1, "recompute": 0}
