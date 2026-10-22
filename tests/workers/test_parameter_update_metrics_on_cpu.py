# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise parameter evidence through the production engine/worker metric path."""

import json
import math
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric, reduce_metrics
from verl.utils.metric.utils import AggregationType
from verl.workers.engine.base import BaseEngine
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine_workers import TrainingWorker


class _ProbeEngine(BaseEngine):
    def __init__(self, rank=0, group=None, enabled=True):
        self.rank = rank
        self.group = group
        self.module = torch.nn.Linear(rank + 2, 1, bias=False, dtype=torch.float64)
        torch.nn.init.constant_(self.module.weight, 10.0)
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.125)
        self.optimizer_config = SimpleNamespace(clip_grad=1000.0, verify_parameter_updates=enabled)
        self._qat_enabled = False
        self.raw_evidence = []
        self.injected_metrics = []
        self.scheduler_steps = 0

    def train_mode(self, **kwargs):
        return nullcontext()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        scale = data["scale"].sum() * (self.rank + 1)
        loss = self.module.weight.sum() * scale
        loss.backward()
        micro_metric = Metric(aggregation="sum")
        micro_metric.extend([float(self.rank + 1), float(self.rank + 3)])
        metrics = {"micro_sum": micro_metric}
        if self.group is not None:
            metrics["legacy_micro"] = [float(self.rank + 1), float(self.rank + 3)]
        return {"loss": [loss.detach().item()], "metrics": metrics, "model_output": {}}

    def optimizer_step(self):
        # Use the actual FSDP optimizer/probe integration with a small CPU module.
        grad_norm = FSDPEngine.optimizer_step(self)
        self.raw_evidence.append(dict(self.last_update_evidence))
        return grad_norm

    def train_batch(self, data, loss_function):
        output = super().train_batch(data, loss_function)
        self.injected_metrics.append(dict(output["metrics"]))
        return output

    def lr_scheduler_step(self):
        self.scheduler_steps += 1
        return self.optimizer.param_groups[0]["lr"]

    def is_mp_src_rank_with_outputs(self):
        return True

    def get_data_parallel_group(self):
        return self.group

    def get_data_parallel_rank(self):
        return self.rank

    def get_data_parallel_size(self):
        return dist.get_world_size(self.group) if self.group is not None else 1


def _run_worker(rank=0, group=None, enabled=True):
    engine = _ProbeEngine(rank, group, enabled)
    worker = SimpleNamespace(
        engine=engine,
        engine_config=SimpleNamespace(
            forward_only=False,
            use_dynamic_bsz=False,
            max_token_len_per_gpu=128,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        ),
        model_config={},
        device_name="cpu",
        flops_counter=None,
        loss_fn=lambda: None,
        profiler=SimpleNamespace(rank=rank, check_enable=lambda: False, step=lambda: None),
    )
    worker.train_batch = partial(TrainingWorker.train_batch, worker)
    worker._postprocess_output = partial(TrainingWorker._postprocess_output, worker)
    data = TensorDict({"scale": torch.tensor([4.0, 1.0, 2.0], dtype=torch.float64)}, batch_size=[3])
    tu.assign_non_tensor(data, num_mini_batch=3)
    memory = SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    with patch("verl.workers.engine_workers.get_torch_device", return_value=memory):
        result = TrainingWorker.train_mini_batch(worker, data)

    metrics = tu.get(result, "metrics")
    reduced = reduce_metrics(dict(metrics))
    size = engine.get_data_parallel_size()
    assert engine.scheduler_steps == 1
    assert metrics["lr"] == [0.125]
    assert metrics["grad_norm"] == pytest.approx([scale * (rank + 1) * math.sqrt(rank + 2) for scale in (4, 1, 2)])
    assert reduced["micro_sum"] == pytest.approx(12.0 if group is None else size + 3.0)
    if group is not None:
        assert metrics["legacy_micro"] == [value for _ in range(3) for r in range(size) for value in (r + 1.0, r + 3.0)]
        assert reduced["legacy_micro"] == pytest.approx((size + 3.0) / 2)

    evidence_keys = {"parameter_delta_max", "parameter_probe_elements_per_rank_max"}
    if enabled:
        assert reduced["parameter_delta_max"] == pytest.approx(0.5 * size)
        assert reduced["parameter_probe_elements_per_rank_max"] == size + 1.0
        for raw, injected in zip(engine.raw_evidence, engine.injected_metrics, strict=True):
            assert set(raw) == evidence_keys
            assert all(type(value) is float for value in raw.values())
            for key, value in raw.items():
                assert isinstance(injected[key], Metric)
                assert injected[key].aggregation is AggregationType.MAX
                assert injected[key].aggregate() == value
        if group is None:
            assert isinstance(metrics["parameter_delta_max"], Metric)
        else:
            assert metrics["parameter_delta_max"] == pytest.approx([0.5 * size, 0.125 * size, 0.25 * size])
    else:
        assert evidence_keys.isdisjoint(metrics)
        assert engine.raw_evidence == [{}, {}, {}]
    # The final logging boundary must contain only finite numeric values.
    assert all(math.isfinite(value) for value in reduced.values())
    json.dumps({key: float(value) for key, value in reduced.items()}, allow_nan=False)


@pytest.mark.parametrize("enabled", [True, False])
def test_parameter_evidence_without_dp_group(enabled):
    _run_worker(enabled=enabled)


def _distributed_worker(rank, size, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=size, timeout=timedelta(seconds=60)
    )
    try:
        for enabled in (True, False):
            _run_worker(rank, dist.group.WORLD, enabled)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_parameter_evidence_through_real_dp_gather(world_size, tmp_path):
    mp.spawn(_distributed_worker, args=(world_size, str(tmp_path / "rdzv")), nprocs=world_size, join=True)
