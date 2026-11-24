# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Regression coverage for publication-driven continuous rollout admission."""

import asyncio
from types import SimpleNamespace

import pytest
import ray
import torch
from omegaconf import OmegaConf

from tests.experimental.fully_async_policy.test_rollouter_failure_on_cpu import CPURollouter
from tests.workers.rollout.test_joint_scheduler_on_cpu import assign, plan, progress, router
from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.protocol import DataProto

Rollouter = FullyAsyncRollouter.__ray_metadata__.modified_class
Queue = MessageQueue.__ray_metadata__.modified_class


def queue_client(queue, **overrides):
    loop = asyncio.get_running_loop()

    def actor_method(method):
        def remote(*args, **kwargs):
            future = asyncio.run_coroutine_threadsafe(method(*args, **kwargs), loop)
            return SimpleNamespace(future=lambda: future)

        return SimpleNamespace(remote=remote)

    methods = (
        "get_sample",
        "get_samples",
        "get_statistics",
        "commit_completed_sample",
        "try_reserve_slot",
        "commit_reserved_sample",
        "release_reservation",
        "shutdown",
    )
    return MessageQueueClient(
        SimpleNamespace(**{name: actor_method(overrides.get(name, getattr(queue, name))) for name in methods})
    )


def rollouter(joint=True):
    r = Rollouter.__new__(Rollouter)
    r.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"weight_publication": {"joint_scheduling": {"enabled": joint, "completed_queue_size": 128}}}
            },
            "async_training": {"trigger_parameter_sync_step": 1},
        }
    )
    r._init_async_objects()
    r.paused = False
    r.required_samples = 2
    r.staleness_threshold = 0.5
    r.total_rollout_steps = 6
    r.concurrent_samples_per_replica = 4
    r.llm_server_manager = SimpleNamespace(get_active_server_count=lambda: 3, get_replicas=lambda: [1, 2, 3])
    r._active_count_history = []
    r.active_tasks = set()
    r.staleness_samples = 0
    r.pending_queue = asyncio.Queue()
    return r


@pytest.mark.asyncio
async def test_joint_fills_bounded_queue_without_trainer_or_version_wakeup():
    r = rollouter()
    rollout = r.config.actor_rollout_ref.rollout
    rollout.n = 16
    rollout.max_concurrent_requests_per_replica = 32
    rollout.weight_publication.joint_scheduling.completed_queue_size = 2
    r.total_generated_samples = r._step_generated_samples = r.processed_sample_count = r.dropped_stale_samples = 0
    await r.set_max_required_samples()
    assert r.max_concurrent_samples == 6 and r.max_queue_size == 2
    queue = Queue(None, r.max_queue_size)
    r.message_queue_client = queue_client(queue)
    r.get_statistics = queue.get_statistics

    async def generate(batch):
        await asyncio.sleep(0)
        return batch

    r.async_rollout_manager = SimpleNamespace(generate_sequences_single=generate)
    for idx in range(40):
        batch = DataProto.from_dict(
            tensors={"responses": torch.full((16, 3), idx), "attention_mask": torch.ones((16, 5), dtype=torch.int64)}
        )
        r.pending_queue.put_nowait(RolloutSample(batch, str(idx), 0, {}))
    r.pending_queue.put_nowait(None)
    await asyncio.wait_for(r._processor_worker(), 3)

    assert not r.active_tasks and not queue._committed_reservations
    assert r.total_generated_samples == queue.total_produced == 40
    assert queue.dropped_samples == 38 and queue.evicted_tokens["trajectories"] == 38 * 16
    retained = [ray.cloudpickle.loads(payload) for payload, _ in await queue.get_samples()]
    assert [sample.sample_id for sample in retained] == ["38", "39"]
    assert all(len(sample.full_batch) == 16 for sample in retained)


@pytest.mark.asyncio
async def test_cancel_after_remote_commit_preserves_result_and_settles_ownership():
    queue = Queue(None, 2)
    committed = asyncio.Event()
    release = asyncio.Event()

    async def commit(*args):
        result = await queue.commit_completed_sample(*args)
        committed.set()
        await release.wait()
        return result

    client = queue_client(queue, commit_completed_sample=commit)
    cost = dict(trajectories=16, prompt_tokens=32, response_tokens=48, total_tokens=80)
    delivery = asyncio.create_task(client.put_completed_sample(b"group", cost))
    await asyncio.wait_for(committed.wait(), 1)
    delivery.cancel()
    await asyncio.sleep(0)
    delivery.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(delivery, 1)
    assert [sample for sample, _ in await queue.get_samples()] == [b"group"]
    assert not queue._committed_reservations and not queue._completed_costs


@pytest.mark.asyncio
async def test_joint_processor_continues_beyond_step_budget_without_reset():
    r = rollouter()
    await r.set_max_required_samples()
    assert r.max_required_samples == 3
    assert r.max_concurrent_samples == 12
    assert r.max_queue_size == 128
    r._update_max_concurrent_samples()
    assert r.max_concurrent_samples == 12
    completed = []

    async def process(sample, *, on_generated):
        await asyncio.sleep(0)
        completed.append(sample.sample_id)
        on_generated()
        return True

    r._process_single_sample_streaming = process
    r.max_concurrent_samples = 2
    for i in range(6):
        await r.pending_queue.put(SimpleNamespace(sample_id=str(i)))
    await r.pending_queue.put(None)
    await asyncio.wait_for(r._processor_worker(), 2)
    assert len(completed) == 6 and not r.paused
    assert r._delivery_acked_groups > r.max_required_samples


@pytest.mark.asyncio
async def test_legacy_rollouter_keeps_step_and_queue_gates():
    r = rollouter(joint=False)
    await r.set_max_required_samples()
    assert r.max_concurrent_samples == r.max_queue_size == 3

    async def stats():
        return {"queue_size": 0}

    r.message_queue_client = SimpleNamespace(get_statistics=stats)
    r.staleness_samples = 3
    assert await r._should_pause_generation()


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_count", [1, 2], ids=["final_drain", "concurrency_limit"])
async def test_generation_wait_does_not_hold_publication_reset_lock(sample_count):
    r = rollouter()
    await r.set_max_required_samples()
    r.max_concurrent_samples = 1
    started = asyncio.Event()
    release = asyncio.Event()

    async def process(sample, *, on_generated):
        started.set()
        await release.wait()
        on_generated()
        return True

    r._process_single_sample_streaming = process
    for i in range(sample_count):
        await r.pending_queue.put(SimpleNamespace(sample_id=str(i)))
    await r.pending_queue.put(None)
    task = asyncio.create_task(r._processor_worker())
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(r.lock.acquire(), 0.2)
        r.lock.release()
        release.set()
        await asyncio.wait_for(task, 2)
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, *r.active_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_completed_buffer_preserves_fifo_and_finish_marker_when_full():
    q = Queue(None, 2)
    client = queue_client(q)
    await client.put_sample_wait("one")
    await client.put_sample_wait("two")
    end = asyncio.create_task(client.put_sample_wait(None))
    await asyncio.sleep(0.02)
    assert not end.done() and q.dropped_samples == 0
    assert (await q.get_sample())[0] == "one"
    await asyncio.wait_for(end, 1)
    assert (await q.get_sample())[0] == "two"
    assert (await q.get_sample())[0] is None
    assert q.total_produced == 3 and q.dropped_samples == 0
    await q.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        await client.put_sample_wait("after shutdown")


@pytest.mark.asyncio
async def test_completed_buffer_wait_is_cancellable_without_eviction():
    q = Queue(None, 1)
    client = queue_client(q)
    await client.put_sample_wait("keep")
    blocked = asyncio.create_task(client.put_sample_wait("pending"))
    await asyncio.sleep(0.02)
    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked
    assert (await q.get_sample())[0] == "keep" and q.dropped_samples == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["normal_finish", "cancel_finish"])
async def test_rollouter_finish_with_full_buffer_preserves_samples_and_remains_cancellable(cancel):
    q = Queue(None, 2)
    client = queue_client(q)
    finishing = asyncio.Event()

    async def put(sample):
        if sample is None:
            finishing.set()
        return await client.put_sample_wait(sample)

    instance = CPURollouter(failure=None)
    instance.paused = False
    instance.running = True
    instance.config = rollouter(joint=False).config
    instance.config.async_training.sample_budget_enabled = False
    instance.message_queue_client = SimpleNamespace(put_sample_wait=put)
    task = asyncio.create_task(instance._streaming_generation_main())
    try:
        await asyncio.wait_for(finishing.wait(), 1)
        assert len(q.queue) == 2 and not task.done()
        if cancel:
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.5)
            assert task in done, "cancel must not wait for a stopped consumer to drain the finish marker"
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(q.queue) == 2
        else:
            assert (await q.get_sample())[0] is not None
            assert (await q.get_sample())[0] is not None
            await asyncio.wait_for(task, 1)
            assert (await q.get_sample())[0] is None
        assert q.dropped_samples == 0
        assert instance.total_generated_samples == 2
        assert not instance.active_tasks and not instance.running
    finally:
        # Release a regressed shielded finish wait without hanging pytest teardown.
        await q.shutdown()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_disabled_deferral_does_not_force_update_or_expire_decode_protection():
    r, clock = router(max_defer_s=None, evacuation_timeout_s=1000)
    assign(r, "x", "a")
    assign(r, "y", "b")
    r.prepare_weight_update(1)
    clock[0] += 300
    assert r.select_replica_for_update(1)["reason"] == "evacuating a"
    rec = r._joint.trajectories["x"]
    rec.moved_at = 100
    rec.moved_tokens = rec.generated_tokens
    assert r._joint.protected(rec)
    rec.generated_tokens += r._joint.config.min_decode_tokens
    assert not r._joint.protected(rec)


@pytest.mark.asyncio
async def test_disabled_deadline_keeps_same_version_migration_destination_eligible():
    r, clock = router(max_defer_s=None)
    assign(r, "x", "a")
    await progress(r, "x")
    r.prepare_weight_update(1)
    r._joint.evacuating["a"] = clock[0]
    clock[0] += 300
    decision = await plan(r, "x")
    assert decision["migrate"] and decision["target_server_id"] == "b"
    assert decision["mode"] == "remote_prefix"
    r.register_trajectory("new", 16, 48)
    assert r.acquire_server("new")[0] == "b"
    r.begin_replica_update("a", 1)
    r.commit_replica_update(
        "a", 1, {**r._server_metadata["a"], "weight_version": 1, "kv_transfer_namespace": "cache-1"}
    )
    r.register_trajectory("fresh", 16, 48)
    assert r.acquire_server("fresh")[0] == "a"
