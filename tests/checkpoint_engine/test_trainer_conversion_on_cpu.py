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
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.checkpoint_engine.test_gpu_snapshot_bucket_on_cpu import BucketWorker
from tests.checkpoint_engine.test_gpu_snapshot_pool_on_cpu import Harness
from verl.checkpoint_engine.conversion_layout import merge_source_layouts, plan_source_buckets
from verl.checkpoint_engine.snapshot_memory import snapshot_memory_role
from verl.checkpoint_engine.trainer_conversion import TrainerConversionPool, validate_trainer_tp_mapping
from verl.workers.config.rollout import GPUSnapshotConfig


def test_explicit_nonidentity_trainer_tp_mapping():
    assert validate_trainer_tp_mapping([2, 0], 2, 4) == [2, 0]
    for value in ([0, 0], [4, 0], [0], [True, 0]):
        with pytest.raises(ValueError):
            validate_trainer_tp_mapping(value, 2, 4)


def test_trainer_conversion_has_no_reserved_rollout_converter():
    config = GPUSnapshotConfig(
        memory_planning="role",
        converter_replica_rank=None,
        conversion_location="trainer",
        trainer_tp_ranks=[2, 0, 3, 1],
        cache_conversion_plan=True,
        pipeline_depth=2,
        bucket_bytes=512 << 20,
        max_bytes_per_gpu=12 << 30,
        max_conversion_bytes=32 << 30,
        max_tensor_bytes=6 << 30,
    )
    for replica in range(3):
        role = snapshot_memory_role(config, replica, 512 << 20)
        assert not role.converter and role.total_bytes == int(18.5 * 2**30)
    with pytest.raises(ValueError):
        replace(config, cache_conversion_plan=False)


def test_source_layout_checks_rank_order_and_bucket_alignment():
    def layout(rank):
        return dict(
            rank=rank,
            world_size=2,
            export="full_tensor_all_gather_on_all_trainer_ranks",
            parameters=[
                dict(
                    name=n,
                    shape=[5],
                    dtype="torch.bfloat16",
                    aliases=[n],
                    fsdp_shard=dict(rank=rank, placements=["Shard(0)"]),
                )
                for n in ["a", "b", "c"]
            ],
        )

    values = [layout(1), layout(0)]
    sources = merge_source_layouts(values)
    assert sources[0]["producer_ranks"] == [0, 1]
    assert [v["rank"] for v in sources[0]["fsdp_shards"]] == [0, 1]
    buckets = plan_source_buckets(sources, 32, 64)
    assert [[(e["name"], e["offset"]) for e in b] for b in buckets] == [[("a", 0), ("b", 16)], [("c", 0)]]
    values[0]["parameters"][0]["shape"] = [6]
    with pytest.raises(ValueError, match="layout differs"):
        merge_source_layouts(values)


class Storage(BucketWorker):
    async def node_id(self):
        return "node-a" if self.index < 2 else "node-b"

    async def initialize(self, size, uid, rank):
        assert size == 6 and rank == self.index + 2


async def harness():
    h = Harness()
    h.config.bucket_bytes = 64
    h.config.trainer_tp_ranks = [1, 0]
    h.workers = [Storage(h, i) for i in range(4)]
    h.replicas = [SimpleNamespace(replica_rank=i, workers=h.workers[i * 2 : i * 2 + 2]) for i in range(2)]
    h.pool = TrainerConversionPool(h.replicas, h.config)
    h.pool.publisher = "publisher-handle"
    await h.pool.describe(["node-a", "node-b"])
    assert h.pool.converter_workers == []
    await h.pool.initialize()
    return h


async def send_column(h, version, trainer_rank, value):
    output = [{"name": "layer", "shape": [6], "dtype": "torch.float32"}]
    result = await h.pool.prepare_trainer_outputs(version, trainer_rank, 0, output)
    assert result["destination_offset"] == 2
    for chunk in result["chunks"]:
        h.send(trainer_rank, chunk.owner + 1, ("output", chunk.key), value[chunk.offset : chunk.offset + chunk.numel])
    await h.pool.finish_trainer_outputs(trainer_rank, 0)
    await h.pool.finish_trainer_round(version, trainer_rank, ["source"], {"conversion_s": 1})


@pytest.mark.asyncio
async def test_trainer_snapshot_is_invisible_until_all_tp_outputs_complete():
    h = await harness()
    await h.pool.begin(1)
    await send_column(h, 1, 1, torch.arange(6, dtype=torch.float32))
    with pytest.raises(ValueError, match="not complete"):
        h.pool.slots.pin(1)
    assert not h.pool.slots.versions[1]["ready"]
    await send_column(h, 1, 0, torch.arange(6, dtype=torch.float32) + 6)
    first = await h.pool.seal(1)
    h.pool.validate(first)
    h.pool.slots.pin(1)
    await h.pool.begin(2)
    h.pool.validate(first)
    await send_column(h, 2, 0, torch.arange(6, dtype=torch.float32) + 106)
    h.pool.validate(first)
    assert not h.pool.slots.versions[2]["ready"]
    await send_column(h, 2, 1, torch.arange(6, dtype=torch.float32) + 100)
    second = await h.pool.seal(2)
    assert first.slot != second.slot
    h.pool.validate(first)
    h.pool.validate(second)
    h.pool.slots.unpin(1)
    await h.pool.close()


@pytest.mark.asyncio
async def test_slow_receiver_holds_credit_and_cancel_waits_for_dma():
    h = await harness()
    await h.pool.begin(1)
    gate = asyncio.Event()
    for worker in h.workers:
        worker.gates["store_chunks"] = gate
    result = await h.pool.prepare_trainer_outputs(1, 1, 0, [{"name": "layer", "shape": [6], "dtype": "torch.float32"}])
    with pytest.raises(RuntimeError, match="backpressure"):
        await h.pool.prepare_trainer_outputs(1, 1, 0, [{"name": "layer", "shape": [6], "dtype": "torch.float32"}])
    waiter = asyncio.create_task(h.pool.finish_trainer_outputs(1, 0))
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()
    for chunk in result["chunks"]:
        h.send(1, chunk.owner + 1, ("output", chunk.key), torch.ones(chunk.numel))
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await h.pool.drain()
    assert h.pool.reserved_outputs  # failed caller cannot silently reuse the round
    h.pool.fatal_error = "cancelled trainer"
    with pytest.raises(RuntimeError, match="restart"):
        await h.pool.begin(2)


@pytest.mark.asyncio
async def test_conversion_plan_methods_cross_checkpoint_worker_rpc_boundary():
    import inspect
    from unittest.mock import AsyncMock

    from verl.checkpoint_engine.base import CheckpointEngineWorker

    storage = SimpleNamespace(
        prepare_conversion_plan=AsyncMock(return_value={"plan_hash": "validated"}),
        export_conversion_plan=AsyncMock(return_value={"payload": b"static", "plan_hash": "validated"}),
    )
    worker = SimpleNamespace(
        rollout_config=SimpleNamespace(weight_publication=SimpleNamespace(transport="nccl_gpu")),
        _gpu_snapshot_worker=storage,
    )
    rpc = inspect.unwrap(CheckpointEngineWorker.gpu_snapshot_rpc)
    assert await rpc(worker, "prepare_conversion_plan", ["source"], {"tp_rank": 0}) == {"plan_hash": "validated"}
    storage.prepare_conversion_plan.assert_awaited_once_with(["source"], {"tp_rank": 0})
    assert (await rpc(worker, "export_conversion_plan"))["payload"] == b"static"
    with pytest.raises(ValueError, match="unsupported"):
        await rpc(worker, "_private")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["preflight", "new_round", "missing_plan"])
async def test_trainer_initialization_failure_still_drains_fsdp_generator(monkeypatch, failure_at):
    from verl.checkpoint_engine.trainer_conversion import export_trainer_converted

    consumed, reported = [], []

    def fail(*args, **kwargs):
        raise RuntimeError("injected initialization failure")

    def weights():
        for i in range(4):
            consumed.append(i)
            yield f"w{i}", torch.ones(2)

    async def report(error):
        reported.append(error)

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    for name in ("reset_peak_memory_stats", "empty_cache", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved", "memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", fail if failure_at == "preflight" else lambda: (8 << 30, 16 << 30))
    plan = SimpleNamespace(
        buffer_budgets={"conversion_peak_bytes": 32, "largest_source_tensor_bytes": 8}, new_round=fail
    )
    worker = SimpleNamespace(
        _trainer_conversion_plan=None if failure_at == "missing_plan" else plan,
        _gpu_snapshot_sender=SimpleNamespace(failed=None),
    )
    with pytest.raises(RuntimeError, match="initialization failure|plan is missing"):
        await export_trainer_converted(
            worker,
            1,
            SimpleNamespace(fail_gpu_snapshot=SimpleNamespace(remote=report)),
            weights(),
            torch.float32,
            SimpleNamespace(trainer_tp_ranks=[0], max_conversion_bytes=1024),
        )
    assert consumed == [0, 1, 2, 3]
    assert len(reported) == 1
