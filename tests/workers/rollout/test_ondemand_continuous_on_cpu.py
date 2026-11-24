"""Continuous decode and generation-fenced migration regression tests."""

import asyncio
from types import SimpleNamespace as NS

import pytest
from omegaconf import OmegaConf

from tests.experimental.fully_async_policy.test_continuous_joint_rollout_on_cpu import rollouter
from tests.workers.rollout.test_joint_scheduler_on_cpu import assign, progress, router
from verl.workers.rollout import llm_server
from verl.workers.rollout.replica import TokenOutput


class Remote:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *a, **kw):
        return await self.fn(*a, **kw)


@pytest.mark.asyncio
async def test_nonjoint_continuous_supply_has_no_step_budget_gate():
    r = rollouter(joint=False)
    r.config.async_training.sample_budget_enabled = False
    r.config.async_training.completed_queue_size = 7
    r.config.trainer = {"total_training_steps": 2}
    await r.set_max_required_samples()
    assert r.max_concurrent_samples == 12 and r.max_queue_size == 7 and r.total_train_steps == 2
    r.staleness_samples = 1000
    assert not await r._should_pause_generation()
    r._update_max_concurrent_samples()
    assert r.max_concurrent_samples == 12


@pytest.mark.asyncio
async def test_preview_does_not_pin_or_move_a_trajectory():
    r, clock = router(max_defer_s=None)
    assign(r, "x", "a")
    assign(r, "y", "b")
    await progress(r, "x")
    r.prepare_weight_update(1)
    r._joint.evacuating["a"] = clock[0]
    before = dict(r._request_id_to_server)
    decision = await r.plan_trajectory_migration(
        request_id="x",
        source_server_id="a",
        trajectory={
            "request_id": "x",
            "prompt_tokens": 16,
            "generated_tokens": 16,
            "prefix_tokens": 32,
            "checkpoint_index": 1,
        },
        source_metadata={"weight_version": 0},
        preview_only=True,
    )
    assert decision["migrate"]
    assert r._request_id_to_server == before
    assert not r._pending_migrations and not r._joint.reservations
    assert sum(r._migration_reservations.values()) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("selected,eos_race", [(False, False), (True, False), (True, True)])
async def test_continuous_segments_pause_only_when_selected(monkeypatch, selected, eos_race):
    config = NS(
        actor_rollout_ref=NS(
            rollout=NS(
                name="rtp_llm",
                response_length=500,
                weight_publication=NS(joint_scheduling=NS(enabled=True)),
                trajectory_migration=NS(
                    enabled=True,
                    checkpoint_mode="on_demand",
                    checkpoint_tokens=128,
                    progress_poll_interval_s=0.002,
                    strict_remote_hit=False,
                ),
            )
        )
    )
    c = llm_server.FullyAsyncLLMServerClient(config=config)
    paused = asyncio.Event()
    closed = asyncio.Event()
    budgets = []
    aborts = []
    migrations = []
    reports = []
    previews = []

    async def get_progress(rid):
        return {
            "found": True,
            "generated_tokens": 160,
            "prefix_ids": [1, 2] + [7] * 160,
            "generation_id": 91,
            "weight_version": 0,
        }

    async def abort(rid, expected_generation_id=None, preserve_kv=False):
        assert preserve_kv
        assert expected_generation_id == 91
        aborts.append(rid)
        paused.set()
        await closed.wait()
        return {"aborted": True}

    server = NS(get_request_progress=Remote(get_progress), abort_request=Remote(abort))

    async def plan(**kw):
        assert kw["preview_only"] is True
        previews.append(kw)
        return {"migrate": selected, "target_server_id": "s1"}

    async def report(**kw):
        reports.append(kw)

    c._load_balancer = NS(plan_trajectory_migration=Remote(plan), report_trajectory_progress=Remote(report))

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
        budgets.append(sampling_params["max_tokens"])
        if len(budgets) == 1:
            self._generation_routes[request_id] = ("s0", server)
            try:
                if selected:
                    await paused.wait()
                else:
                    await asyncio.sleep(0.02)
            finally:
                closed.set()
                self._generation_routes.pop(request_id, None)
            return TokenOutput(
                token_ids=[7] * 160,
                stop_reason="completed",
                extra_fields={"global_steps": 0, "reached_eos": eos_race, llm_server._ROUTE_SERVER_ID_FIELD: "s0"},
            )
        assert closed.is_set() and kw["migration_ticket"] == {"ticket": "ready"}
        return TokenOutput(
            token_ids=[8] * 5,
            stop_reason="completed",
            remote_reuse_len=160,
            extra_fields={"global_steps": 0, llm_server._ROUTE_SERVER_ID_FIELD: "s1", "remote_migration_hit": True},
        )

    async def migrate(**kw):
        assert closed.is_set()
        migrations.append(kw)
        return {"mode": "remote_prefix", "migration_ticket": {"ticket": "ready"}}

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    c._try_migrate_trajectory = migrate
    out = await asyncio.wait_for(
        c._generate_segments("request", prompt_ids=[1, 2], sampling_params={"max_tokens": 500}), 1
    )
    assert budgets == ([500, 340] if selected and not eos_race else [500])
    assert len(aborts) == int(selected)
    assert len(migrations) == int(selected and not eos_race)
    assert len(out.token_ids) == (165 if selected and not eos_race else 160)
    assert previews and reports


@pytest.mark.asyncio
async def test_on_demand_cancellation_waits_for_owned_generation(monkeypatch):
    config = NS(
        actor_rollout_ref=NS(
            rollout=NS(
                name="rtp_llm",
                weight_publication=NS(joint_scheduling=NS(enabled=True)),
                trajectory_migration=NS(enabled=True, checkpoint_mode="on_demand", progress_poll_interval_s=0.002),
            )
        )
    )
    c = llm_server.FullyAsyncLLMServerClient(config=config)
    started = asyncio.Event()
    settled = asyncio.Event()

    async def generate(self, **kw):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    task = asyncio.create_task(
        c._generate_on_demand(
            "x", original_prompt_tokens=1, generated_before=0, prompt_ids=[1], sampling_params={"max_tokens": 20}
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert settled.is_set()


@pytest.mark.asyncio
async def test_stop_generation_settles_workers_and_fit_returns_successfully():
    from tests.experimental.fully_async_policy.test_rollouter_failure_on_cpu import CPURollouter

    r = CPURollouter(failure="blocked")
    task = asyncio.create_task(r.fit())
    await asyncio.wait_for(r.started.wait(), 1)
    await asyncio.wait_for(r.stop_generation(), 1)
    await asyncio.wait_for(task, 1)
    assert r.settled.is_set() and not r.active_tasks and not r.running


@pytest.mark.asyncio
async def test_native_progress_and_abort_are_generation_fenced():
    from tests.workers.rollout.rollout_rtp_llm.test_async_server_kv_cache import TestRTPLLMKVCacheTransition

    started = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()

    async def native():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()

    task = asyncio.create_task(native())
    await started.wait()
    other = asyncio.create_task(asyncio.Event().wait())
    server, _ = TestRTPLLMKVCacheTransition._server(
        generation_allowed=True, abort_requested=False, inflight={"x": task, "other": other}
    )
    ids = [4, 5]
    server._request_progress = {
        "x": {"token_ids": ids, "prompt_ids": [1, 2, 3], "weight_version": 7, "generation_id": 92}
    }
    server._accepted_migrations["x"] = {"ticket": "next"}
    try:
        snap = await server.get_request_progress("x")
        ids.append(6)
        assert snap["prefix_ids"] == [1, 2, 3, 4, 5]
        refused = await server.abort_request("x", expected_generation_id=91)
        assert not refused["aborted"] and not task.done() and "x" in server._accepted_migrations
        abort = asyncio.create_task(server.abort_request("x", expected_generation_id=92))
        await closing.wait()
        assert not abort.done() and not other.done()
        release.set()
        assert (await abort)["aborted"] and task.cancelled() and not other.done()
    finally:
        release.set()
        task.cancel()
        other.cancel()
        await asyncio.gather(task, other, return_exceptions=True)


@pytest.mark.asyncio
async def test_explicit_training_step_limit_is_independent_of_producer_budget():
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    t = cls.__new__(cls)
    t.message_queue_client = object()
    t.rollouter = object()
    t.global_steps = 0
    t.config = OmegaConf.create({"trainer": {"total_training_steps": 3}})
    t.progress_bar = NS(close=lambda: None)
    t.metrics_aggregator = NS(step_count=0)
    t._fit_save_checkpoint = lambda **kw: None
    completed = []

    async def step():
        completed.append(t.global_steps)
        t.global_steps += 1

    t.fit_step = step
    await t._fit_loop()
    assert completed == [1, 2, 3]


def test_native_capacity_caps_configured_1024_request_ceiling():
    r, _ = router(kv_capacity_tokens=1024 * 10240, max_batch_size=1024)
    r._joint.validate_capacity_metadata(
        {"a": {"max_batch_size": 8, "max_sequence_tokens": 10240, "kv_capacity_tokens": 64}}
    )
    assign(r, "one", "a", budget=64)
    r.register_trajectory("two", 16, 48)
    assert r._joint.batch_limit("a") == 8 and r._joint.kv_limit("a") == 64
    assert not r._joint.fits("a", "two")


@pytest.mark.asyncio
async def test_selected_native_checkpoint_settles_without_cancelling_other_requests():
    from tests.workers.rollout.rollout_rtp_llm.test_async_server_kv_cache import TestRTPLLMKVCacheTransition

    requested = asyncio.Event()
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def native():
        await requested.wait()
        await asyncio.sleep(0.02)
        finished.set()

    task = asyncio.create_task(native())
    other = asyncio.create_task(asyncio.Event().wait())
    server, _ = TestRTPLLMKVCacheTransition._server(
        generation_allowed=True, abort_requested=False, inflight={"x": task, "other": other}
    )
    calls = []

    def checkpoint(gid):
        calls.append(gid)
        loop.call_soon_threadsafe(requested.set)
        return True

    server.engine = NS(request_kv_checkpoint=checkpoint)
    server._request_progress = {"x": {"generation_id": 93, "token_ids": [2], "prompt_ids": [1], "weight_version": 0}}
    try:
        assert not (await server.abort_request("x", expected_generation_id=92, preserve_kv=True))["aborted"]
        assert not calls and not task.done()
        result = await server.abort_request("x", expected_generation_id=93, preserve_kv=True)
        assert result["aborted"] and result["preserve_kv"] and calls == [93]
        assert finished.is_set() and task.done() and not task.cancelled() and not other.done()
    finally:
        task.cancel()
        other.cancel()
        await asyncio.gather(task, other, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["eventually_ready", "write_failed", "never_registered"])
async def test_native_prefix_registration_is_bounded_and_write_errors_are_not_retried(status):
    from tests.workers.rollout.rollout_rtp_llm.test_async_server_kv_cache import TestRTPLLMKVCacheTransition

    server, _ = TestRTPLLMKVCacheTransition._server(generation_allowed=True, abort_requested=False, inflight={})
    server.config = NS()
    caps = {
        "model_id": "test",
        "weight_version": 0,
        "kv_transfer_domain": "d",
        "kv_transfer_namespace": "n",
        "kv_block_size_tokens": 16,
    }
    server.get_trajectory_migration_capabilities = lambda: dict(caps)
    server._remote_prefix_migration_enabled = lambda: True
    server._cache_key_salt = lambda _: 0
    server.global_steps = 0
    server.replica_rank = 0
    calls = []

    def wait(tokens, salt, timeout):
        calls.append(timeout)
        if status == "write_failed":
            return {"status": "write_failed"}
        if status == "eventually_ready" and len(calls) > 1:
            return {"status": "success", "error_code": 0, "write_ids": [1], "prefix_tokens": 16}
        return {"status": "unavailable"}

    server.engine = NS(wait_remote_prefix_status=wait)
    args = ({"request_id": "x", "prompt_ids": [1] * 17, "generated_token_ids": [2]}, caps, "remote_prefix", 0.03)
    if status == "eventually_ready":
        ticket = await server.prepare_trajectory_migration(*args)
        assert ticket["remote_write_status"]["write_ids"] == [1] and len(calls) == 2
    else:
        with pytest.raises(asyncio.TimeoutError if status == "never_registered" else RuntimeError):
            await server.prepare_trajectory_migration(*args)
        if status == "write_failed":
            assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["unregistered", "rejected", "cancelled", "timeout"])
async def test_selected_checkpoint_unsettled_paths_cannot_export(scenario):
    from tests.workers.rollout.rollout_rtp_llm.test_async_server_kv_cache import TestRTPLLMKVCacheTransition

    task = asyncio.create_task(asyncio.Event().wait())
    loop = asyncio.get_running_loop()
    server, _ = TestRTPLLMKVCacheTransition._server(
        generation_allowed=True, abort_requested=False, inflight={"x": task}
    )
    server._request_progress = {} if scenario == "unregistered" else {"x": {"generation_id": 9}}
    calls = []

    def checkpoint(gid):
        calls.append(gid)
        if scenario == "cancelled":
            loop.call_soon_threadsafe(task.cancel)
        return scenario != "rejected"

    server.engine = NS(request_kv_checkpoint=checkpoint)
    try:
        if scenario in ("unregistered", "rejected"):
            result = await server.abort_request("x", preserve_kv=True, timeout_s=0.02)
            assert not result["aborted"] and not task.done()
            assert calls == ([] if scenario == "unregistered" else [9])
        else:
            with pytest.raises(RuntimeError, match="cancelled|deadline"):
                await server.abort_request("x", preserve_kv=True, timeout_s=0.02)
            if scenario == "timeout":
                assert not task.cancelled()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
