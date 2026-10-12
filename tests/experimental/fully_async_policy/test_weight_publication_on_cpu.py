# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy import fully_async_trainer as trainer_module

Trainer = trainer_module.FullyAsyncTrainer.__ray_metadata__.modified_class


class RemoteMethod:
    def __init__(self, method):
        self.method = method

    def remote(self, *args, **kwargs):
        return self.method(*args, **kwargs)


def config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "hybrid_engine": False,
                "actor": {"strategy": "fsdp2"},
                "model": {"lora": {"rank": 0, "merge": False}},
                "rollout": {
                    "name": "rtp_llm",
                    "nnodes": 1,
                    "weight_publication": {"enabled": True, "wait_timeout_s": 20},
                    "fine_grained_weight_update": {"enabled": True},
                    "checkpoint_engine": {"_target_": "verl.workers.config.CheckpointEngineConfig"},
                },
            },
            "async_training": {"use_trainer_do_validate": False, "use_dynamic_resource_scheduling": False},
            "rollout": {"nnodes": 1},
            "trainer": {"test_freq": 1},
            "global_profiler": {"steps": None},
        }
    )


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("actor_rollout_ref.actor.strategy", "megatron", "FSDP"),
        ("actor_rollout_ref.hybrid_engine", True, "hybrid"),
        ("async_training.use_trainer_do_validate", True, "use_trainer_do_validate"),
        ("async_training.use_dynamic_resource_scheduling", True, "dynamic"),
        ("actor_rollout_ref.model.lora.rank", 8, "LoRA"),
        ("rollout.nnodes", 0, "standalone"),
        ("actor_rollout_ref.rollout.name", "vllm", "rtp_llm"),
        ("actor_rollout_ref.rollout.fine_grained_weight_update.enabled", False, "fine_grained"),
    ],
)
def test_publication_rejects_unsupported_experiment(path, value, match):
    cfg = config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError, match=match):
        trainer_module.validate_weight_publication_config(cfg)


def test_full_weight_and_merged_lora_publication_are_supported():
    cfg = config()
    trainer_module.validate_weight_publication_config(cfg)
    cfg.actor_rollout_ref.model.lora.rank = 8
    cfg.actor_rollout_ref.model.lora.merge = True
    trainer_module.validate_weight_publication_config(cfg)
    cfg.actor_rollout_ref.rollout.weight_publication.enabled = False
    cfg.actor_rollout_ref.actor.strategy = "megatron"
    trainer_module.validate_weight_publication_config(cfg)


def trainer(export, publish, status=None):
    instance = Trainer.__new__(Trainer)
    instance.config = config()
    instance.weight_publication_enabled = True
    instance.weight_publisher = SimpleNamespace(
        publish=RemoteMethod(publish),
        get_status=RemoteMethod(status or (lambda: asyncio.sleep(0, result={"fatal_error": None}))),
    )
    instance.actor_rollout_wg = SimpleNamespace(export_weight_snapshot=export)
    instance.current_param_version = 3
    instance.local_trigger_step = 1
    instance.timing_raw = {}
    instance.logger = SimpleNamespace(log=lambda **kwargs: None)
    return instance


@pytest.mark.asyncio
async def test_training_exports_once_and_returns_after_publication_admission():
    calls = []
    snapshot = SimpleNamespace(version=3)

    async def export_rank(value):
        await asyncio.sleep(0)
        return value

    def export(global_steps):
        calls.append(global_steps)
        return [export_rank(snapshot), export_rank(None)]

    async def publish(value):
        assert value is snapshot
        return {"accepted": True}

    instance = trainer(export, publish)
    timing = await asyncio.wait_for(instance._fit_update_weights(), timeout=1)
    assert isinstance(timing, dict)
    assert calls == [3]
    assert await instance._fit_update_weights() is None
    assert calls == [3]


@pytest.mark.asyncio
async def test_failed_export_settles_all_ranks_before_returning_error():
    release = asyncio.Event()
    settled = asyncio.Event()

    async def failed_rank():
        raise RuntimeError("export failed")

    async def other_rank():
        await release.wait()
        settled.set()

    async def publish(snapshot):
        pytest.fail("partial export must not be published")

    instance = trainer(lambda **kwargs: [failed_rank(), other_rank()], publish)
    task = asyncio.create_task(instance._fit_update_weights())
    await asyncio.sleep(0.02)
    assert not task.done()
    release.set()
    with pytest.raises(RuntimeError, match="export failed"):
        await task
    assert settled.is_set()


@pytest.mark.asyncio
async def test_unhealthy_publisher_rejects_before_export():
    async def status():
        return {"fatal_error": "transfer crashed"}

    async def publish(snapshot):
        pytest.fail("unhealthy publisher must not admit snapshots")

    instance = trainer(lambda **kwargs: pytest.fail("export must not start"), publish, status)
    with pytest.raises(RuntimeError, match="transfer crashed"):
        await instance._fit_update_weights()


@pytest.mark.asyncio
async def test_validation_waits_for_version_before_using_rollouter():
    release = asyncio.Event()
    validated = asyncio.Event()

    async def wait(version=None, timeout_s=None):
        assert version is None or version == 3
        await release.wait()

    async def validate():
        validated.set()
        return {}

    instance = trainer(None, None)
    instance.wait_for_weight_publication = wait
    instance.rollouter = SimpleNamespace(do_validate=RemoteMethod(validate))
    task = asyncio.create_task(instance._fit_validate())
    await asyncio.sleep(0.02)
    assert not validated.is_set()
    release.set()
    await task
    assert validated.is_set()


@pytest.mark.asyncio
async def test_shutdown_terminates_actor_even_when_drain_fails(monkeypatch):
    async def close(drain):
        assert drain
        raise RuntimeError("drain failed")

    instance = trainer(None, None)
    handle = instance.weight_publisher
    handle.close = RemoteMethod(close)
    killed = []
    monkeypatch.setattr(trainer_module.ray, "kill", lambda actor: killed.append(actor))
    with pytest.raises(RuntimeError, match="drain failed"):
        await instance.shutdown_weight_publisher()
    assert killed == [handle]
    assert instance.weight_publisher is None


@pytest.mark.asyncio
@pytest.mark.parametrize("versions", [[], [3, 3], [2]])
async def test_malformed_export_is_never_published(versions):
    async def result(version):
        return SimpleNamespace(version=version)

    async def publish(snapshot):
        pytest.fail("invalid export must not publish")

    instance = trainer(lambda **kwargs: [result(version) for version in versions], publish)
    with pytest.raises(RuntimeError, match="manifest|version"):
        await instance._fit_update_weights()


@pytest.mark.asyncio
async def test_setup_owns_only_publisher_and_reuses_existing_coordinator(monkeypatch):
    from verl.checkpoint_engine import weight_publication

    instances = []
    handle = object()

    def create(**kwargs):
        assert kwargs["replicas"] == ["standalone"]
        assert kwargs["load_balancer_handle"] == "router"
        assert "actor_wg" not in kwargs
        instances.append(kwargs)
        return handle

    monkeypatch.setattr(weight_publication, "WeightPublisherActor", SimpleNamespace(remote=create))
    monkeypatch.setattr(trainer_module, "CheckpointEngineManager", lambda **kwargs: pytest.fail("legacy manager"))
    instance = Trainer.__new__(Trainer)
    instance.config = config()
    instance.rollouter = SimpleNamespace(
        get_replicas=RemoteMethod(lambda: asyncio.sleep(0, result=["standalone"])),
        get_load_balancer=RemoteMethod(lambda: asyncio.sleep(0, result="router")),
    )
    instance.checkpoint_manager = None
    await instance._setup_checkpoint_manager()
    await instance._setup_checkpoint_manager()
    assert instance.get_weight_publisher() is handle
    assert instance.checkpoint_manager is None
    assert len(instances) == 1


@pytest.mark.asyncio
async def test_fit_preserves_training_error_when_publication_shutdown_also_fails():
    instance = Trainer.__new__(Trainer)

    async def fail_training():
        raise ValueError("training failed")

    async def fail_shutdown(**kwargs):
        raise RuntimeError("shutdown failed")

    instance._fit_loop = fail_training
    instance.shutdown_weight_publisher = fail_shutdown
    with pytest.raises(ValueError, match="training failed"):
        await instance.fit()


def test_task_runner_shuts_down_publisher_if_initialization_fails():
    from verl.experimental.fully_async_policy import fully_async_main as main_module

    runner_class = main_module.FullyAsyncTaskRunner.__ray_metadata__.modified_class
    instance = runner_class.__new__(runner_class)
    shutdown = []
    instance.components = {
        "trainer": SimpleNamespace(shutdown_weight_publisher=RemoteMethod(lambda: shutdown.append(True)))
    }
    instance._initialize_components = lambda cfg: (_ for _ in ()).throw(ValueError("startup failed"))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(main_module.ray, "get", lambda value: value)
        with pytest.raises(ValueError, match="startup failed"):
            instance.run(config())
    assert shutdown == [True]


def test_startup_waits_for_publication_before_validation(monkeypatch):
    import verl.utils
    from verl.experimental.fully_async_policy import fully_async_main as main_module

    runner_class = main_module.FullyAsyncTaskRunner.__ray_metadata__.modified_class
    instance = runner_class.__new__(runner_class)
    calls = []
    trainer_handle = SimpleNamespace(
        **{
            name: RemoteMethod(lambda *args, _name=name, **kwargs: calls.append(_name))
            for name in [
                "set_rollouter",
                "set_total_train_steps",
                "set_message_queue_client",
                "load_checkpoint",
                "_fit_update_weights",
                "wait_for_weight_publication",
                "_fit_validate",
            ]
        }
    )
    rollouter_handle = SimpleNamespace(
        **{
            name: RemoteMethod(lambda *args, _name=name, **kwargs: calls.append(_name) or 1)
            for name in ["get_total_train_steps", "get_max_queue_size", "set_message_queue_client", "load_checkpoint"]
        }
    )
    instance.components = {"trainer": trainer_handle, "rollouter": rollouter_handle}
    instance._create_trainer = lambda cfg: None
    instance._setup_hybrid_worker_group = lambda cfg: None
    instance._create_rollouter = lambda cfg: None
    cfg = config()
    cfg.actor_rollout_ref.model.path = "test-model"
    cfg.data = {}
    monkeypatch.setattr(main_module, "copy_to_local", lambda *args, **kwargs: "test-model")
    monkeypatch.setattr(verl.utils, "hf_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(verl.utils, "hf_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "create_role_worker_mapping", lambda cfg: ({}, None))
    monkeypatch.setattr(main_module.ray, "get", lambda value: value)
    monkeypatch.setattr(main_module, "MessageQueue", SimpleNamespace(remote=lambda *args: object()))
    monkeypatch.setattr(main_module, "MessageQueueClient", lambda queue: queue)
    instance._initialize_components(cfg)
    assert calls.index("_fit_update_weights") < calls.index("wait_for_weight_publication")
    assert calls.index("wait_for_weight_publication") < calls.index("_fit_validate")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_cancelled_export_settles_all_worker_calls_before_returning(cancel_count):
    release = asyncio.Event()
    started = asyncio.Event()
    settled = asyncio.Event()

    async def export_rank():
        started.set()
        await release.wait()
        settled.set()
        return SimpleNamespace(version=3)

    async def publish(snapshot):
        pytest.fail("cancelled export must not publish")

    instance = trainer(lambda **kwargs: [export_rank()], publish)
    task = asyncio.create_task(instance._fit_update_weights())
    await started.wait()
    for _ in range(cancel_count):
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert settled.is_set()


@pytest.mark.asyncio
async def test_fit_closes_without_killing_actor_until_main_monitor_finishes(monkeypatch):
    closed = asyncio.Event()
    kills = []
    instance = trainer(None, None)
    handle = instance.weight_publisher

    async def close(drain):
        assert drain
        closed.set()

    async def finish_training():
        return

    handle.close = RemoteMethod(close)
    instance._fit_loop = finish_training
    monkeypatch.setattr(trainer_module.ray, "kill", lambda actor: kills.append(actor))
    await instance.fit()
    assert closed.is_set()
    assert kills == []
    assert instance.get_weight_publisher() is handle
    await instance.shutdown_weight_publisher()
    assert kills == [handle]


def test_main_observes_publisher_failure_while_training_and_rollout_are_pending(monkeypatch):
    from verl.experimental.fully_async_policy import fully_async_main as main_module

    runner_class = main_module.FullyAsyncTaskRunner.__ray_metadata__.modified_class
    instance = runner_class.__new__(runner_class)
    training, rollout, health = object(), object(), object()
    canceled = []
    publisher = SimpleNamespace(watch_health=RemoteMethod(lambda: health))

    async def clear():
        return

    instance.components = {
        "trainer": SimpleNamespace(
            fit=RemoteMethod(lambda: training), get_weight_publisher=RemoteMethod(lambda: publisher)
        ),
        "rollouter": SimpleNamespace(fit=RemoteMethod(lambda: rollout)),
        "message_queue_client": SimpleNamespace(clear_queue=clear),
    }

    def wait(futures, **kwargs):
        assert health in futures
        return [health], [training, rollout]

    def get(future):
        if future is health:
            raise RuntimeError("publication background failed")
        return future

    monkeypatch.setattr(main_module.ray, "wait", wait)
    monkeypatch.setattr(main_module.ray, "get", get)
    monkeypatch.setattr(main_module.ray, "cancel", lambda future: canceled.append(future))
    with pytest.raises(RuntimeError, match="publication background failed"):
        instance._run_training_loop()
    assert training in canceled and rollout in canceled


@pytest.mark.asyncio
async def test_publication_preserves_rollouter_profiling_boundaries():
    calls = []

    async def mark(name):
        calls.append(name)

    async def publish(snapshot):
        calls.append("publish")

    async def result():
        return SimpleNamespace(version=3)

    instance = trainer(lambda **kwargs: [result()], publish)
    instance.config.global_profiler.steps = [3, 4]
    instance.rollouter = SimpleNamespace(
        _stop_profiling=RemoteMethod(lambda: mark("stop")),
        _start_profiling=RemoteMethod(lambda: mark("start")),
    )
    await instance._fit_update_weights()
    assert calls == ["stop", "publish", "start"]
