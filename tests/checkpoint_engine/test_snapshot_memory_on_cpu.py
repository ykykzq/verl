# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from verl.checkpoint_engine.snapshot_memory import fixed_converter_workers, snapshot_memory_role
from verl.workers.config.rollout import GPUSnapshotConfig


def config(**changes):
    return GPUSnapshotConfig(
        slots=2,
        max_bytes_per_gpu=12 << 30,
        max_conversion_bytes=32 << 30,
        max_tensor_bytes=6 << 30,
        bucket_bytes=512 << 20,
        pipeline_depth=2,
        **changes,
    )


def test_only_fixed_converter_reserves_conversion_working_set():
    legacy = config()
    role = replace(legacy, memory_planning="role", converter_replica_rank=1)
    plans = [snapshot_memory_role(role, i, 512 << 20) for i in range(3)]
    assert [p.total_bytes for p in plans] == [int(x * 2**30) for x in (18.5, 82.5, 18.5)]
    assert plans[1].total_bytes == snapshot_memory_role(legacy, 1, 512 << 20).total_bytes
    assert plans[1].conversion_input_bytes == 12 << 30
    assert plans[1].conversion_workspace_bytes == 32 << 30
    assert plans[1].conversion_output_bytes == 20 << 30
    assert fixed_converter_workers(role, [0, 1, 2], 4) == [4, 5, 6, 7]
    with pytest.raises(ValueError, match="absent"):
        fixed_converter_workers(role, [0, 2], 4)


def test_invalid_or_unreserved_converter_fails_closed():
    for kw in (
        {"memory_planning": "role"},
        {"memory_planning": "dynamic"},
        {"memory_planning": "role", "converter_replica_rank": -1},
        {"memory_planning": "role", "converter_replica_rank": True},
    ):
        with pytest.raises(ValueError):
            config(**kw)


@pytest.mark.asyncio
async def test_capacity_uses_slowest_tp_rank_and_cannot_change():
    from verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server import RTPLLMHttpServer

    server = RTPLLMHttpServer.__new__(RTPLLMHttpServer)
    server.engine = None
    server._replica_capacity = None
    server.config = SimpleNamespace(tensor_model_parallel_size=4)
    plans = [{"tp_rank": i, "capacity": n} for i, n in enumerate((144, 143, 142, 144))]
    with patch.object(server, "_max_concurrency", return_value=144):
        assert await server.freeze_replica_capacity(plans) == 142
        with pytest.raises(RuntimeError, match="immutable"):
            await server.freeze_replica_capacity(plans)


@pytest.mark.asyncio
async def test_nonconverter_rejects_conversion_before_allocating():
    from tests.workers.rollout.rollout_rtp_llm.test_gpu_snapshot_server_on_cpu import server

    instance = server()
    instance._snapshot_memory_role = snapshot_memory_role(
        config(memory_planning="role", converter_replica_rank=1), 0, 512 << 20
    )
    with pytest.raises(RuntimeError, match="without reserved"):
        await instance.begin_gpu_snapshot(1, 32 << 30)
    assert instance._gpu_snapshot is None


def test_capacity_is_not_a_kv_layout_compatibility_key():
    from verl.workers.rollout.joint_scheduler import JointReplicaScheduler

    common = dict(
        model_id="qwen25",
        rollout_config_fingerprint="same-tp4-layout",
        weight_version=7,
        kv_transfer_domain="test",
        kv_transfer_namespace="v7",
        kv_transfer_backends=["remote_prefix"],
    )
    source = dict(common, max_batch_size=51, kv_capacity_tokens=500000)
    target = dict(common, max_batch_size=144, kv_capacity_tokens=1474560)
    assert JointReplicaScheduler.compatible(source, target, True)
    assert not JointReplicaScheduler.compatible(source, dict(target, weight_version=8), True)
    assert not JointReplicaScheduler.compatible(source, dict(target, rollout_config_fingerprint="wrong-layout"), True)
