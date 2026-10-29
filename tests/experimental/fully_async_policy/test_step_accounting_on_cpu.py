# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Keep optimizer-step counters consistent across full synchronization-cycle resumes."""

import asyncio
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.experimental.fully_async_policy.test_training_metrics_on_cpu import RemoteMethod, make_trainer
from verl.utils.profiler.torch_profile import build_trace_basename


class CompletedRef:
    def __init__(self):
        self.result = Future()
        self.result.set_result(None)

    def future(self):
        return self.result

    def __await__(self):
        return asyncio.wrap_future(self.result).__await__()


@pytest.mark.asyncio
@pytest.mark.parametrize("publication", [False, True], ids=["legacy_sync", "snapshot_publication"])
@pytest.mark.parametrize(
    "resume_mode,resume_version,sync_interval,expected_logs,expected_profiles,expected_checkpoints",
    [
        ("disable", None, 1, [(1, 1, 1), (2, 2, 2)], [2], [1, 2]),
        ("auto", None, 1, [(1, 1, 1), (2, 2, 2)], [2], [1, 2]),
        ("disable", None, 3, [(1, 3, 1), (2, 6, 2)], [4, 5, 6], [1, 2]),
        ("auto", None, 3, [(1, 3, 1), (2, 6, 2)], [4, 5, 6], [1, 2]),
        ("auto", 5, 1, [(6, 6, 6), (7, 7, 7)], [7], [6, 7]),
        ("resume_path", 5, 1, [(6, 6, 6), (7, 7, 7)], [7], [6, 7]),
        ("auto", 5, 3, [(6, 18, 6), (7, 21, 7)], [19, 20, 21], [6, 7]),
        ("resume_path", 5, 3, [(6, 18, 6), (7, 21, 7)], [19, 20, 21], [6, 7]),
    ],
    ids=["fresh-1", "empty-auto-1", "fresh-3", "empty-auto-3", "auto-1", "path-1", "auto-3", "path-3"],
)
async def test_fresh_and_resumed_cycles_keep_metrics_profiles_and_checkpoint_steps_aligned(
    monkeypatch,
    tmp_path,
    publication,
    resume_mode,
    resume_version,
    sync_interval,
    expected_logs,
    expected_profiles,
    expected_checkpoints,
):
    instance, logged, published, _ = make_trainer(
        monkeypatch, updates=2 * sync_interval, sync_interval=sync_interval, test_freq=-1, publication=publication
    )
    instance.config.trainer.update(
        {
            "save_freq": 1,
            "default_local_dir": str(tmp_path),
            "default_hdfs_dir": None,
            "resume_mode": resume_mode,
            "del_local_ckpt_after_load": False,
        }
    )
    instance.config.global_profiler.steps = [expected_checkpoints[-1]]
    instance.use_reference_policy = False
    instance.use_critic = True
    # Exercise the real inherited profiler gates and worker-facing step arguments.
    del instance._fit_start_profile
    del instance._fit_stop_profile

    actor_saves, critic_saves, loads, profiles, rollouter_saves = [], [], [], [], []

    def save_worker(records, local_path, remote_path, version, **kwargs):
        Path(local_path).mkdir(parents=True, exist_ok=True)
        records.append((Path(local_path).relative_to(tmp_path).as_posix(), remote_path, version))

    def start_profile(*, profile_step, role):
        basename = build_trace_basename(
            rank=0, role=role, save_file_prefix="actor", topology={}, profile_step=profile_step
        )
        profiles.append((instance.current_param_version, profile_step, basename))

    instance.actor_rollout_wg.save_checkpoint = lambda *args, **kwargs: save_worker(actor_saves, *args, **kwargs)
    instance.actor_rollout_wg.load_checkpoint = lambda path, **kwargs: loads.append(Path(path).name)
    instance.actor_rollout_wg.start_profile = start_profile
    instance.actor_rollout_wg.stop_profile = lambda **kwargs: None
    instance.critic_wg = SimpleNamespace(
        save_checkpoint=lambda *args, **kwargs: save_worker(critic_saves, *args, **kwargs),
        load_checkpoint=lambda path, **kwargs: loads.append(Path(path).name),
        start_profile=lambda **kwargs: None,
        stop_profile=lambda **kwargs: None,
    )
    instance.rollouter.save_checkpoint = RemoteMethod(lambda path: rollouter_saves.append(Path(path).name))
    instance.rollouter._start_profiling = RemoteMethod(CompletedRef)
    instance.rollouter._stop_profiling = RemoteMethod(CompletedRef)

    # Queue length is independent of the counter under test, as it is in production.
    batches = iter(range(2 * sync_interval))

    async def next_batch():
        if next(batches, None) is None:
            return 0, None
        return 0, SimpleNamespace(meta_info={"trajectory_param_versions": [instance.current_param_version]})

    instance._get_samples_from_queue = next_batch
    if resume_version is not None:
        resume_path = tmp_path / f"global_step_{resume_version}"
        resume_path.mkdir()
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(resume_version))
        instance.config.trainer.resume_from_path = str(resume_path)

    loaded_version = await instance.load_checkpoint()
    await instance._fit_loop()

    assert loaded_version == (resume_version or 0)
    assert loads == (["actor", "critic"] if resume_version is not None else [])
    assert [
        (step, data["training/global_step"], data["training/param_version"])
        for step, data in logged
        if "training/global_step" in data
    ] == expected_logs
    assert instance.global_steps == expected_logs[-1][1] + 1
    assert instance.current_param_version == expected_checkpoints[-1]
    assert instance.local_trigger_step == 1
    assert published == expected_checkpoints

    assert [step for _, step, _ in profiles] == expected_profiles
    assert all(version == expected_checkpoints[-1] - 1 for version, _, _ in profiles)
    assert all(name.startswith(f"actor_train_step{step}_rank0_") for _, step, name in profiles)
    assert actor_saves == [(f"global_step_{version}/actor", None, version) for version in expected_checkpoints]
    assert critic_saves == [(f"global_step_{version}/critic", None, version) for version in expected_checkpoints]
    assert rollouter_saves == [f"global_step_{version}" for version in expected_checkpoints]
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == str(expected_checkpoints[-1])
