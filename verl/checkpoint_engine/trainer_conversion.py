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
"""Convert full FSDP exports on explicit trainer ranks into immutable TP snapshots.

The validated full_tensor export remains unchanged. Every rank advances that
same generator in order; a converted bucket's device transfer settles before
advancing to the next FSDP collective. No GPU payload enters the Ray store.
"""

import time

import torch

from verl.checkpoint_engine.gpu_snapshot_pipeline import PipelineSnapshotPool
from verl.checkpoint_engine.gpu_snapshot_pool import GPUSnapshotPool, settle


def validate_trainer_tp_mapping(mapping, tp_size, trainer_size):
    if (
        len(mapping) != tp_size
        or len(set(mapping)) != tp_size
        or any(type(rank) is not int or not 0 <= rank < trainer_size for rank in mapping)
    ):
        raise ValueError("trainer_tp_ranks must map each target TP rank to one distinct trainer rank")
    return list(mapping)


class TrainerConversionPool(PipelineSnapshotPool):
    async def describe(self, source_nodes):
        await GPUSnapshotPool.describe(self)
        self.trainer_size = len(source_nodes)
        self.trainer_tp_ranks = validate_trainer_tp_mapping(
            self.config.trainer_tp_ranks, self.tp_size, self.trainer_size
        )
        worker_nodes = await settle([self.call(i, "node_id") for i in range(len(self.workers))])
        self.converter_workers = []  # No rollout GPU may run conversion in this mode.
        self.topology = {
            "conversion_location": "trainer",
            "source_nodes": source_nodes,
            "worker_nodes": worker_nodes,
            "trainer_tp_ranks": self.trainer_tp_ranks,
            "distribution_rank_offset": self.trainer_size,
        }
        self.input_world = self.trainer_size + len(self.workers)
        self.input_uid = self.unique_id
        self.transfers = {}
        self.reserved_outputs = set()
        return dict(
            world_size=self.input_world,
            unique_id=self.unique_id,
            timeout_s=self.config.operation_timeout_s,
            actor_ranks=list(range(self.trainer_size)),
            pipeline=True,
            conversion_location="trainer",
        )

    async def export_trainer_plans(self):
        if not getattr(self, "conversion_plans", None):
            raise RuntimeError("trainer conversion plans must be validated first")
        bundles = await settle([self.call(tp, "export_conversion_plan") for tp in range(self.tp_size)])
        for tp, bundle in enumerate(bundles):
            if bundle["plan_hash"] != self.conversion_plans[tp]["plan_hash"]:
                raise ValueError("exported trainer conversion plan differs from validated plan")
        return bundles

    async def initialize(self):
        self.require_open()
        if self.initialized:
            raise RuntimeError("trainer snapshot pool already initialized")
        try:
            await settle(
                [
                    self.call(i, "initialize", self.input_world, self.unique_id, self.trainer_size + i)
                    for i in range(len(self.workers))
                ]
            )
            ids = await settle([self.call(tp, "unique_id") for tp in range(self.tp_size)])
            await settle(
                [
                    self.call(i, "initialize_fetch", len(self.replicas), ids[i % self.tp_size], i // self.tp_size)
                    for i in range(len(self.workers))
                ]
            )
        except BaseException as error:
            self.fatal_error = f"trainer snapshot transport initialization failed: {error}"
            raise
        self.initialized = True

    async def begin(self, version):
        self.require_open()
        if self.transfers or self.reserved_outputs:
            raise RuntimeError("previous trainer output transfers have not settled")
        result = await super().begin(version)
        self.next_sequence = [0] * self.tp_size
        self.finished_trainers = set()
        return result

    def _target_tp(self, trainer_rank):
        if trainer_rank not in self.trainer_tp_ranks:
            raise ValueError("trainer rank has no assigned target TP shard")
        return self.trainer_tp_ranks.index(trainer_rank)

    async def prepare_trainer_outputs(self, version, trainer_rank, sequence, outputs):
        self.require_open()
        tp = self._target_tp(trainer_rank)
        if self.staging is None or self.staging.version != version or sequence != self.next_sequence[tp]:
            raise ValueError("trainer output version or collective sequence mismatch")
        if any(key[0] == tp for key in self.transfers):
            raise RuntimeError("trainer output backpressure: previous transfer still owns its tensors")
        expected = {name: (shape, dtype) for name, shape, dtype in self.plan.tensors[tp]}
        names = set()
        for item in outputs:
            name = item["name"]
            if name in names or (tp, name) in self.completed or (tp, name) in self.reserved_outputs:
                raise ValueError("duplicate trainer runtime tensor")
            if expected.get(name) != (tuple(item["shape"]), item["dtype"]):
                raise ValueError(f"trainer runtime layout mismatch: {name}")
            names.add(name)
        if not names:
            raise ValueError("empty trainer output transfer")
        chunks = [c for c in self.plan.chunks if c.tp_rank == tp and c.name in names]
        self.reserved_outputs.update((tp, name) for name in names)
        key = (tp, sequence)
        job = self.track(
            self.stage(
                "trainer_runtime_transfer",
                sequence,
                [
                    self.call(
                        owner, "store_chunks", self.staging.slot, [c for c in chunks if c.owner == owner], trainer_rank
                    )
                    for owner in sorted({c.owner for c in chunks})
                ],
            )
        )
        self.transfers[key] = (job, names)
        return {"chunks": chunks, "destination_offset": self.trainer_size}

    async def finish_trainer_outputs(self, trainer_rank, sequence):
        tp = self._target_tp(trainer_rank)
        key = (tp, sequence)
        job, names = self.transfers[key]
        await settle([job])
        self.completed.update((tp, name) for name in names)
        self.reserved_outputs.difference_update((tp, name) for name in names)
        del self.transfers[key]
        self.next_sequence[tp] += 1

    async def finish_trainer_round(self, version, trainer_rank, seen, metrics):
        tp = self._target_tp(trainer_rank)
        if self.staging is None or self.staging.version != version or trainer_rank in self.finished_trainers:
            raise ValueError("trainer conversion round version or completion mismatch")
        if any(key[0] == tp for key in self.transfers) or set(seen) != self.required[tp]:
            raise ValueError("trainer conversion is incomplete")
        self.finished_trainers.add(trainer_rank)
        self.metrics.setdefault("trainer_conversion", {})[trainer_rank] = metrics

    async def _convert_columns_and_store(self, tp_ranks, name=None, is_last=False):
        await self.drain()
        if not is_last or self.transfers or self.finished_trainers != set(self.trainer_tp_ranks):
            raise RuntimeError("cannot expose an incomplete trainer-converted snapshot")


async def initialize_trainer_plan(worker, bundles, config):
    rank = torch.distributed.get_rank()
    mapping = validate_trainer_tp_mapping(config.trainer_tp_ranks, len(bundles), torch.distributed.get_world_size())
    worker._trainer_conversion_plan = None
    if rank not in mapping:
        return None
    tp = mapping.index(rank)
    bundle = bundles[tp]
    started = time.perf_counter()
    if bundle.get("kind") == "hf_identity_tp1_v1":
        from verl.checkpoint_engine.hf_snapshot_plan import HFSnapshotPlan

        plan = HFSnapshotPlan.restore(bundle["payload"], bundle["plan_hash"])
    else:
        from rtp_llm.model_loader.runtime_snapshot import ConversionPlan

        plan = ConversionPlan.restore(bundle["payload"], bundle["plan_hash"])
    if plan.manifest["transport"]["tp_rank"] != tp:
        raise ValueError("FSDP rank was assigned the wrong target TP plan")
    if plan.buffer_budgets["conversion_peak_bytes"] > config.max_conversion_bytes:
        raise MemoryError("static conversion plan exceeds configured trainer workspace")
    worker._trainer_conversion_plan = plan
    return {
        "trainer_rank": rank,
        "target_tp_rank": tp,
        "plan_hash": plan.plan_hash,
        "restore_s": time.perf_counter() - started,
        "buffer_budgets": plan.buffer_budgets,
    }


async def export_trainer_converted(worker, version, publisher, weights, dtype, config):
    rank = torch.distributed.get_rank()
    plan = getattr(worker, "_trainer_conversion_plan", None)
    assigned = rank in config.trainer_tp_ranks
    sender = getattr(worker, "_gpu_snapshot_sender", None)
    round_ = None
    failure = None
    if assigned != (plan is not None):
        failure = RuntimeError("trainer conversion plan is missing; no rollout fallback is allowed")
    elif assigned and sender is None:
        failure = RuntimeError("trainer snapshot sender is missing")
    seen = set()
    sequence = 0
    started = time.perf_counter()
    metrics = {
        "parameter_aggregation_s": 0.0,
        "conversion_s": 0.0,
        "runtime_transfer_s": 0.0,
        "source_bytes": 0,
        "runtime_bytes": 0,
        "rule_construction_s": 0.0,
        "wall_start": time.time(),
        "stage_events": [],
    }

    async def send(outputs):
        nonlocal sequence
        if not outputs:
            return
        wall = time.time()
        start = time.perf_counter()
        description = [{"name": n, "shape": list(t.shape), "dtype": str(t.dtype)} for n, t in outputs]
        result = (await settle([publisher.prepare_trainer_outputs.remote(version, rank, sequence, description)]))[0]
        values = dict(outputs)
        operations = [
            (values[c.name].reshape(-1)[c.offset : c.offset + c.numel], result["destination_offset"] + c.owner, False)
            for owner in sorted({c.owner for c in result["chunks"]})
            for c in result["chunks"]
            if c.owner == owner
        ]
        await sender.transfer_many(operations)
        await settle([publisher.finish_trainer_outputs.remote(rank, sequence)])
        elapsed = time.perf_counter() - start
        metrics["runtime_transfer_s"] += elapsed
        metrics["runtime_bytes"] += sum(t.nbytes for _, t in outputs)
        metrics["stage_events"].append({"stage": "trainer_runtime_transfer", "start": wall, "duration": elapsed})
        sequence += 1

    try:
        try:
            torch.cuda.reset_peak_memory_stats()
            if failure is None and plan is not None:
                torch.cuda.empty_cache()
                free, _total = torch.cuda.mem_get_info()
                needed = (
                    plan.buffer_budgets["conversion_peak_bytes"]
                    + plan.buffer_budgets["largest_source_tensor_bytes"]
                    + (2 << 30)
                )
                metrics["preflight_free_bytes"] = free
                metrics["preflight_required_bytes"] = needed
                if free < needed:
                    raise MemoryError(
                        f"trainer conversion requires {needed} free bytes including safety, available {free}"
                    )
                round_ = plan.new_round(
                    version, torch.device("cuda", torch.cuda.current_device()), config.max_conversion_bytes
                )
        except BaseException as error:
            # Even a preflight/allocation error on one rank must not skip the
            # FSDP collectives still being entered by its peers below.
            failure = failure or error
        iterator = iter(weights)
        while True:
            before = time.perf_counter()
            try:
                name, tensor = next(iterator)
            except StopIteration:
                break
            try:
                if failure is not None:
                    continue  # Every FSDP rank must still drain all collectives in order.
                torch.cuda.synchronize()
                metrics["parameter_aggregation_s"] += time.perf_counter() - before
                if plan is None:
                    continue
                metrics["source_bytes"] += tensor.nbytes
                if name not in plan.required_names:
                    continue
                value = tensor.detach().to(dtype=dtype if tensor.is_floating_point() else tensor.dtype)
                start = time.perf_counter()
                outputs = round_.add([(name, value)])
                elapsed = time.perf_counter() - start
                metrics["conversion_s"] += elapsed
                metrics["stage_events"].append(
                    {"stage": "trainer_conversion", "start": time.time() - elapsed, "duration": elapsed}
                )
                seen.add(name)
                await send(outputs)
                del outputs, value
            except BaseException as error:
                failure = error
            finally:
                del tensor
        if failure is None and round_ is not None:
            await send(round_.add([], is_last=True))
    except BaseException as error:
        failure = failure or error
    finally:
        metrics.update(
            allocated_peak_bytes=torch.cuda.max_memory_allocated(),
            reserved_peak_bytes=torch.cuda.max_memory_reserved(),
            allocated_end_bytes=torch.cuda.memory_allocated(),
            reserved_end_bytes=torch.cuda.memory_reserved(),
            total_s=time.perf_counter() - started,
        )
        if round_ is not None:
            if sender.failed:
                worker._failed_conversion_round = round_  # Native DMA may still own outputs.
            else:
                round_.close()
    if failure is not None:
        await settle([publisher.fail_gpu_snapshot.remote(f"trainer rank {rank} conversion failed: {failure}")])
        raise failure
    if plan is not None:
        await settle([publisher.finish_trainer_round.remote(version, rank, sorted(seen), metrics)])
    return metrics if rank == 0 else None
