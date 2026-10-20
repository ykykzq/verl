# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Exercise rollout task supervision with CPU generation and real asyncio tasks."""

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import torch

from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.protocol import DataProto

Rollouter = FullyAsyncRollouter.__ray_metadata__.modified_class
Runner = FullyAsyncTaskRunner.__ray_metadata__.modified_class


class CPURollouter(Rollouter):
    """Keep production supervision and sample processing, replacing GPU/dataset setup."""

    def __init__(self, failure="generation", cleanup_failure=False, queue_client=None):
        self._init_async_objects()
        self.failure = failure
        self.cleanup_failure = cleanup_failure
        self.pending_queue = asyncio.Queue(maxsize=2)
        self.active_tasks = set()
        self.max_concurrent_samples = 2
        self._active_count_history = []
        self.staleness_samples = 0
        self.total_generated_samples = 0
        self._step_generated_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        self.started = asyncio.Event()
        self.settled = asyncio.Event()
        self.finished = asyncio.Event()
        self.monitor_settled = asyncio.Event()
        self.queue_client = queue_client
        self.message_queue_client = SimpleNamespace(put_sample=self._put_sample)
        self.async_rollout_manager = SimpleNamespace(generate_sequences_single=self._generate)

    async def _put_sample(self, sample):
        if sample is None:
            self.finished.set()
            if self.cleanup_failure:
                raise RuntimeError("finish signal failed")
        if self.queue_client is not None:
            return await self.queue_client.put_sample(sample)
        return True

    async def _feed_samples(self):
        if self.failure == "feed":
            raise RuntimeError("feed failed")
        for index in range(2):
            batch = DataProto.from_dict({"input_ids": torch.tensor([[index]])})
            await self.pending_queue.put(RolloutSample(batch, str(index), 0, {}))
        await self.pending_queue.put(None)

    async def _generate(self, batch):
        if self.failure is None:
            return batch
        if batch.batch["input_ids"].item() == 0 and self.failure == "generation":
            await self.started.wait()
            raise TypeError("generation RPC rejected session_id")
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.settled.set()

    async def _async_monitor_loop(self):
        try:
            if self.failure == "monitor":
                await self.started.wait()
                raise RuntimeError("monitor failed")
            await self.finished.wait()
        finally:
            self.monitor_settled.set()

    async def _should_pause_generation(self):
        return False

    async def get_statistics(self):
        return {}


async def _settle_leftovers(instance):
    # Ensure a regression cannot leave test-owned tasks alive in the next test.
    tasks = list(instance.active_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,exception,message",
    [
        ("generation", TypeError, "generation RPC rejected session_id"),
        ("feed", RuntimeError, "feed failed"),
        ("monitor", RuntimeError, "monitor failed"),
    ],
)
async def test_fit_propagates_failure_after_settling_owned_tasks(failure, exception, message):
    instance = CPURollouter(failure)
    try:
        with pytest.raises(exception, match=message):
            await asyncio.wait_for(instance.fit(), timeout=2)
        assert instance.active_tasks == set()
        assert instance.feed_task is None
        assert instance.processor_task is None
        assert not instance.running
        assert instance.finished.is_set()
        assert instance.monitor_settled.is_set()
        if instance.started.is_set():
            assert instance.settled.is_set()
    finally:
        await _settle_leftovers(instance)


@pytest.mark.asyncio
async def test_finish_signal_failure_does_not_replace_generation_failure():
    instance = CPURollouter(cleanup_failure=True)
    try:
        with pytest.raises(TypeError, match="generation RPC rejected session_id"):
            await asyncio.wait_for(instance.fit(), timeout=2)
        assert not instance.running
        assert not instance.active_tasks
    finally:
        await _settle_leftovers(instance)


@pytest.mark.asyncio
async def test_successful_generation_still_finishes_and_delivers_samples():
    instance = CPURollouter(failure=None)
    await asyncio.wait_for(instance.fit(), timeout=2)
    assert instance.total_generated_samples == 2
    assert instance.finished.is_set()
    assert not instance.running
    assert not instance.active_tasks


@pytest.mark.asyncio
async def test_cancelling_fit_settles_generation_and_monitor():
    instance = CPURollouter(failure="blocked")
    fit_task = asyncio.create_task(instance.fit())
    try:
        await asyncio.wait_for(instance.started.wait(), timeout=2)
        fit_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await fit_task
        assert not instance.active_tasks
        assert instance.settled.is_set()
        assert instance.monitor_settled.is_set()
        assert not instance.running
    finally:
        await _settle_leftovers(instance)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["monitor", "cancel"])
async def test_failure_cleanup_finishes_when_interrupted_again(interrupt):
    instance = CPURollouter()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()
    monitor_failed = asyncio.Event()
    generate = instance._generate

    async def slow_cleanup(batch):
        try:
            return await generate(batch)
        finally:
            if batch.batch["input_ids"].item() == 1:
                cleanup_started.set()
                await cleanup_release.wait()
                cleanup_finished.set()

    async def failed_monitor():
        await cleanup_started.wait()
        monitor_failed.set()
        raise RuntimeError("monitor failed during cleanup")

    instance.async_rollout_manager.generate_sequences_single = slow_cleanup
    if interrupt == "monitor":
        instance._async_monitor_loop = failed_monitor
    fit_task = asyncio.create_task(instance.fit())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        if interrupt == "cancel":
            fit_task.cancel()
        else:
            await asyncio.wait_for(monitor_failed.wait(), timeout=2)
        # No exit is allowed until the blocked cleanup is released.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fit_task), timeout=0.05)
        cleanup_release.set()
        expected = asyncio.CancelledError if interrupt == "cancel" else RuntimeError
        with pytest.raises(expected):
            await asyncio.wait_for(fit_task, timeout=2)
        assert cleanup_finished.is_set()
        assert instance.finished.is_set()
        assert instance.feed_task is None
        assert instance.processor_task is None
        assert not instance.active_tasks
        assert not instance.running
    finally:
        cleanup_release.set()
        await asyncio.gather(fit_task, return_exceptions=True)
        await _settle_leftovers(instance)


@ray.remote(num_cpus=0)
class CPUTrainer:
    def __init__(self, queue_client):
        self.queue_client = queue_client

    async def fit(self):
        while True:
            sample, _queue_size = await self.queue_client.get_sample()
            if sample is None:
                return

    def get_weight_publisher(self):
        return None

    def shutdown_weight_publisher(self):
        pass


def _run_failure_probe():
    ray.init(address="local", num_cpus=2, include_dashboard=False, object_store_memory=100_000_000)
    try:
        queue_client = MessageQueueClient(MessageQueue.options(num_cpus=0).remote(None, 10))
        runner = Runner()
        runner.components = {
            "rollouter": ray.remote(num_cpus=0)(CPURollouter).remote(queue_client=queue_client),
            "trainer": CPUTrainer.remote(queue_client),
            "message_queue_client": queue_client,
        }
        runner._initialize_components = lambda config: None
        runner.run(None)
    finally:
        ray.shutdown()


def test_generation_failure_exits_ray_driver_with_nonzero_status():
    result = subprocess.run(
        [sys.executable, "-m", "tests.experimental.fully_async_policy.test_rollouter_failure_on_cpu"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert "generation RPC rejected session_id" in output, output
    assert result.returncode != 0, output
    assert "Component failed with error" in output, output


if __name__ == "__main__":
    _run_failure_probe()
