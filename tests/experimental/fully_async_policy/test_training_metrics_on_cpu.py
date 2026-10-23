# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise metric aggregation through the trainer's real step and queue-stop flow."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy import fully_async_trainer as trainer_module
from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator

Trainer = trainer_module.FullyAsyncTrainer.__ray_metadata__.modified_class


class RemoteMethod:
    def __init__(self, method):
        self.method = method

    def remote(self, *args, **kwargs):
        return self.method(*args, **kwargs)


def make_trainer(monkeypatch, updates, sync_interval, test_freq, publication):
    instance = Trainer.__new__(Trainer)
    instance.config = OmegaConf.create(
        {
            "global_profiler": {"steps": None},
            "actor_rollout_ref": {"rollout": {"temperature": 1.0, "weight_publication": {"transport": "ray"}}},
            "trainer": {"test_freq": test_freq, "save_freq": -1, "esi_redundant_time": 0},
            "async_training": {"use_trainer_do_validate": False},
        }
    )
    instance.global_steps = 0
    instance.epoch = 0
    instance.current_param_version = 0
    instance.local_trigger_step = 1
    instance.trigger_parameter_sync_step = sync_interval
    instance.last_ckpt_version = 0
    instance.stale_trajectory_processed = 0
    instance.required_samples = 4
    instance.staleness_threshold = 1
    instance.dynamic_schedule_enabled = False
    instance.dynamic_resource_controller = None
    instance.dynamic_schedule_ctx = SimpleNamespace(step_required_samples=4)
    instance.only_hybrid = False
    instance.weight_publication_enabled = publication
    instance._last_published_version = 0
    instance.metrics_aggregator = MetricsAggregator(total_gpus=2, hybrid_gpus=1, standalone_gpus=1)
    instance.message_queue_client = SimpleNamespace(get_queue_size_sync=lambda: 100 - instance.global_steps)
    instance.progress_bar = SimpleNamespace(update=lambda count: None, close=lambda: None)

    logged, published, validated = [], [], []
    instance.logger = SimpleNamespace(log=lambda *, data, step: logged.append((step, data.copy())))

    async def next_batch():
        if instance.global_steps > updates:
            return 0, None
        return 0, SimpleNamespace(meta_info={"trajectory_param_versions": [instance.current_param_version]})

    instance._get_samples_from_queue = next_batch
    for method in (
        "_fit_compute_reward",
        "_fit_compute_log_prob",
        "_fit_compute_ref_log_prob",
        "_fit_compute_critic",
        "_fit_compute_advantage",
        "_fit_update_critic",
    ):
        setattr(instance, method, lambda batch: batch)

    def update_actor(batch):
        instance.metrics["actor/update_sum"] = instance.global_steps * 10
        return batch

    instance._fit_update_actor = update_actor
    instance._fit_start_profile = lambda **kwargs: None
    instance._fit_stop_profile = lambda **kwargs: None
    instance._fit_dump_data = lambda batch: None
    instance._fit_collect_metrics = lambda batch: instance.metrics.update(
        {"timing_s/step": instance.timing_raw["step"]}
    )

    async def validate():
        validated.append(instance.current_param_version)
        return {"val/reward": 1.0}

    async def update_weights(global_steps):
        published.append(global_steps)

    def reset_staleness():
        result = Future()
        result.set_result({"dynamic_resource/rollout_resource_utilization": 0.5})
        return SimpleNamespace(future=lambda: result)

    instance.rollouter = SimpleNamespace(
        do_validate=RemoteMethod(validate),
        get_total_produced_samples=RemoteMethod(lambda: updates * 4),
        get_completed_steps=RemoteMethod(lambda: instance.current_param_version),
        reset_staleness=RemoteMethod(reset_staleness),
    )
    instance.checkpoint_manager = SimpleNamespace(update_weights=update_weights)
    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)

    def export(global_steps):
        return [asyncio.sleep(0, result=SimpleNamespace(version=global_steps))]

    async def publish(snapshot):
        published.append(snapshot.version)

    instance.actor_rollout_wg = SimpleNamespace(export_weight_snapshot=export)
    instance.weight_publisher = SimpleNamespace(
        get_status=RemoteMethod(lambda: asyncio.sleep(0, result={})),
        publish=RemoteMethod(publish),
    )
    instance.wait_for_weight_publication = lambda: asyncio.sleep(0)
    return instance, logged, published, validated


@pytest.mark.asyncio
@pytest.mark.parametrize("publication", [False, True], ids=["legacy_sync", "snapshot_publication"])
@pytest.mark.parametrize("test_freq", [-1, 0, 1, 2])
@pytest.mark.parametrize(
    "updates,sync_interval,expected_metrics,expected_published,expected_local_step",
    [
        (0, 1, [], [], 1),
        (1, 1, [(1, 1, 10, 1)], [1], 1),
        (2, 1, [(1, 1, 10, 1), (2, 2, 20, 2)], [1, 2], 1),
        (1, 2, [(1, 1, 10, 0)], [], 2),
        (2, 2, [(1, 2, 30, 1)], [1], 1),
        (3, 2, [(1, 2, 30, 1), (2, 3, 30, 1)], [1], 2),
    ],
)
async def test_training_metrics_include_each_update_once_through_queue_stop(
    monkeypatch,
    publication,
    test_freq,
    updates,
    sync_interval,
    expected_metrics,
    expected_published,
    expected_local_step,
):
    instance, logged, published, validated = make_trainer(monkeypatch, updates, sync_interval, test_freq, publication)

    await instance._fit_loop()

    training_logs = [(version, data) for version, data in logged if "training/global_step" in data]
    assert [
        (step, data["training/global_step"], data["actor/update_sum"], data["training/param_version"])
        for step, data in training_logs
    ] == expected_metrics
    assert [data["dynamic_resource/mq_size"] for _, data in training_logs] == [
        99 - global_step for _, global_step, _, _ in expected_metrics
    ]
    assert published == expected_published
    assert validated == [version for version in expected_published if test_freq > 0 and version % test_freq == 0]
    assert instance.current_param_version == (expected_published[-1] if expected_published else 0)
    assert instance.local_trigger_step == expected_local_step
    assert instance.metrics_aggregator.step_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("publication", [False, True], ids=["legacy_sync", "snapshot_publication"])
async def test_final_partial_cycle_does_not_overwrite_the_previous_log_row(monkeypatch, publication):
    instance, _, published, _ = make_trainer(monkeypatch, 3, 2, -1, publication)
    rows = {}

    def log(*, data, step):
        # Tracking forwards explicit steps to W&B, which merges data within a step.
        rows.setdefault(step, {}).update(data)

    instance.logger = SimpleNamespace(log=log)

    await instance._fit_loop()

    assert list(rows) == [1, 2]
    assert [
        (step, data["training/global_step"], data["training/param_version"], data["actor/update_sum"])
        for step, data in rows.items()
    ] == [(1, 2, 1, 30), (2, 3, 1, 30)]
    assert [data["fully_async/count/current_param_version"] for data in rows.values()] == [0, 1]
    assert published == [1]
    assert instance.current_param_version == 1


@pytest.mark.asyncio
async def test_sync_cycle_combines_current_training_metrics_with_rollout_utilization(monkeypatch):
    instance, logged, _, _ = make_trainer(monkeypatch, updates=1, sync_interval=1, test_freq=-1, publication=False)

    await instance._fit_loop()

    training_logs = [data for _, data in logged if "training/global_step" in data]
    assert len(training_logs) == 1
    assert training_logs[0]["dynamic_resource/rollout_resource_utilization"] == 0.5
    assert training_logs[0]["dynamic_resource/resource_utilization"] == 0.25
