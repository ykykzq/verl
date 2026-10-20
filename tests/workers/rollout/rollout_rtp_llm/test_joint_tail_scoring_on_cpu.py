# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import itertools
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server import RTPLLMHttpServer


class TestTailScoring(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = RTPLLMHttpServer.__new__(RTPLLMHttpServer)
        self.server.global_steps = 7
        self.server.replica_rank = 0
        self.server.config = SimpleNamespace(
            max_model_len=32,
            prompt_length=16,
            response_length=16,
            dtype="bfloat16",
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            enable_prefix_caching=True,
            trajectory_migration=SimpleNamespace(enabled=True, kv_transfer_backend="remote_prefix"),
        )
        self.server.model_config = SimpleNamespace(
            local_path="/model",
            hf_config=SimpleNamespace(architectures=["ModelForCausalLM"], layer_types=["full_attention"]),
        )
        self.server._request_counter = itertools.count(1)
        self.server._generation_allowed = asyncio.Event()
        self.server._generation_allowed.set()
        self.server._request_admission_lock = asyncio.Lock()
        self.server._inflight = {}
        self.server._accepted_migrations = {}
        self.server._abort_requested = False
        self.server._rejecting = False
        self.inputs = []
        self.closed = asyncio.Event()
        self.started = asyncio.Event()
        self.release = None
        self.loss = torch.tensor([0.1, 0.2, 0.3, 0.4])

        async def enqueue(generate_input):
            self.inputs.append(generate_input)

            async def stream():
                try:
                    self.started.set()
                    if self.release is not None:
                        await self.release.wait()
                    yield SimpleNamespace(generate_outputs=[SimpleNamespace(loss=self.loss)])
                finally:
                    self.closed.set()

            return stream()

        self.server.visitor = SimpleNamespace(enqueue=enqueue)
        modules = patch.dict(
            sys.modules,
            {
                "rtp_llm.config.generate_config": SimpleNamespace(GenerateConfig=SimpleNamespace),
                "rtp_llm.utils.base_model_datatypes": SimpleNamespace(GenerateInput=SimpleNamespace),
            },
        )
        modules.start()
        self.addCleanup(modules.stop)

    async def test_scores_tail_tokens_with_full_context_and_raw_logprobs(self):
        result = await self.server.score_trajectory_tail([10, 20, 30, 40, 50], 2, 7)
        self.assertEqual(result["weight_version"], 7)
        torch.testing.assert_close(torch.tensor(result["log_probs"]), torch.tensor([-0.3, -0.4]))
        self.assertEqual(self.inputs[0].token_ids.tolist(), [10, 20, 30, 40, 50])
        config = self.inputs[0].generate_config
        self.assertEqual(config.calculate_loss, 2)
        self.assertEqual(config.max_new_tokens, 1)
        self.assertFalse(config.is_streaming)
        self.assertFalse(config.reuse_cache)
        self.assertFalse(config.enable_remote_cache)
        self.assertFalse(config.can_use_pd_separation)
        self.assertTrue(self.closed.is_set())
        self.assertEqual(self.server._inflight, {})

    async def test_rejects_wrong_version_without_enqueuing(self):
        with self.assertRaisesRegex(RuntimeError, "version"):
            await self.server.score_trajectory_tail([10, 20], 1, 6)
        self.assertEqual(self.inputs, [])

    async def test_rechecks_version_after_admission_lock_wait(self):
        await self.server._request_admission_lock.acquire()
        task = asyncio.create_task(self.server.score_trajectory_tail([10, 20], 1, 7))
        await asyncio.sleep(0)
        self.server.global_steps = 8
        self.server._request_admission_lock.release()
        with self.assertRaisesRegex(RuntimeError, "version"):
            await task
        self.assertEqual(self.inputs, [])

    async def test_rejects_paused_or_rejecting_replica_without_waiting(self):
        for state in ("paused", "rejecting", "aborting"):
            with self.subTest(state=state):
                self.server._generation_allowed.set()
                self.server._rejecting = state == "rejecting"
                self.server._abort_requested = state == "aborting"
                if state == "paused":
                    self.server._generation_allowed.clear()
                with self.assertRaisesRegex(RuntimeError, "scoring"):
                    await asyncio.wait_for(self.server.score_trajectory_tail([10, 20], 1, 7), timeout=1)
        self.assertEqual(self.inputs, [])

    async def test_abort_drains_scoring_and_clears_registration(self):
        self.release = asyncio.Event()
        task = asyncio.create_task(self.server.score_trajectory_tail([10, 20, 30, 40, 50], 2, 7))
        await self.started.wait()
        self.assertEqual(len(self.server._inflight), 1)
        await self.server.abort_all_requests(reject_request=True)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.closed.is_set())
        self.assertEqual(self.server._inflight, {})

    async def test_caller_cancellation_closes_stream_and_removes_registration(self):
        self.release = asyncio.Event()
        task = asyncio.create_task(self.server.score_trajectory_tail([10, 20, 30, 40, 50], 2, 7))
        await self.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.closed.is_set())
        self.assertEqual(self.server._inflight, {})

    async def test_rejects_result_if_version_changed_during_scoring(self):
        self.release = asyncio.Event()
        task = asyncio.create_task(self.server.score_trajectory_tail([10, 20, 30, 40, 50], 2, 7))
        await self.started.wait()
        self.server.global_steps = 8
        self.release.set()
        with self.assertRaisesRegex(RuntimeError, "version"):
            await task
        self.assertEqual(self.server._inflight, {})

    async def test_missing_partial_or_nonfinite_loss_is_not_a_score(self):
        for loss in (None, torch.tensor([0.1]), torch.tensor([0.1, 0.2, float("nan"), 0.4])):
            with self.subTest(loss=loss):
                self.loss = loss
                with self.assertRaisesRegex(RuntimeError, "loss"):
                    await self.server.score_trajectory_tail([10, 20, 30, 40, 50], 2, 7)
                self.assertEqual(self.server._inflight, {})

    async def test_invalid_tail_or_oversize_prefix_rejected_before_enqueue(self):
        for prefix, tail in (([10], 1), ([10, 20], 0), ([10, 20], 2), (list(range(32)), 2)):
            with self.subTest(prefix=prefix, tail=tail):
                with self.assertRaises(ValueError):
                    await self.server.score_trajectory_tail(prefix, tail, 7)
        self.assertEqual(self.inputs, [])

    async def test_multiple_scores_use_distinct_drain_registrations(self):
        self.release = asyncio.Event()
        first = asyncio.create_task(self.server.score_trajectory_tail([10, 20, 30, 40, 50], 2, 7))
        second = asyncio.create_task(self.server.score_trajectory_tail([10, 20, 30, 40, 50], 2, 7))
        while len(self.inputs) != 2:
            await asyncio.sleep(0)
        self.assertEqual(len(self.server._inflight), 2)
        self.release.set()
        await asyncio.gather(first, second)
        self.assertEqual(self.server._inflight, {})

    async def test_backend_timeout_closes_stream_and_does_not_return_partial_score(self):
        self.release = asyncio.Event()
        with patch.dict(os.environ, {"VERL_RTP_LLM_TAIL_SCORE_TIMEOUT_S": "0.01"}):
            with self.assertRaises(asyncio.TimeoutError):
                await self.server.score_trajectory_tail([10, 20, 30, 40, 50], 2, 7)
        self.assertTrue(self.closed.is_set())
        self.assertEqual(self.server._inflight, {})

    def test_config_fingerprint_distinguishes_cache_layouts_not_replica_identity(self):
        first = self.server.get_trajectory_migration_capabilities()["rollout_config_fingerprint"]
        self.server.replica_rank = 12
        self.assertEqual(first, self.server.get_trajectory_migration_capabilities()["rollout_config_fingerprint"])
        self.server.config.dtype = "float16"
        self.assertNotEqual(first, self.server.get_trajectory_migration_capabilities()["rollout_config_fingerprint"])
        self.server.config.dtype = "bfloat16"
        self.server.model_config.hf_config.layer_types = ["linear_attention"]
        self.assertNotEqual(first, self.server.get_trajectory_migration_capabilities()["rollout_config_fingerprint"])

    def test_config_fingerprint_normalizes_dtype_alias(self):
        first = self.server.get_trajectory_migration_capabilities()["rollout_config_fingerprint"]
        self.server.config.dtype = "bf16"
        self.assertEqual(first, self.server.get_trajectory_migration_capabilities()["rollout_config_fingerprint"])

    def test_capabilities_report_hybrid_block_size_and_sequence_limit(self):
        regular = self.server.get_trajectory_migration_capabilities()
        self.assertEqual(regular["kv_block_size_tokens"], 16)
        self.assertEqual(regular["max_sequence_tokens"], 32)
        self.server.model_config.hf_config.layer_types = ["linear_attention", "full_attention"]
        hybrid = self.server.get_trajectory_migration_capabilities()
        self.assertEqual(hybrid["kv_block_size_tokens"], 1024)

    def test_capabilities_prefer_effective_engine_block_and_concurrency_settings(self):
        self.server._cfg = SimpleNamespace(
            kv_cache_config=SimpleNamespace(seq_size_per_block=64),
            concurrency_config=SimpleNamespace(concurrency_limit=3),
        )
        capabilities = self.server.get_trajectory_migration_capabilities()
        self.assertEqual(capabilities["kv_block_size_tokens"], 64)
        self.assertEqual(capabilities["max_batch_size"], 3)

    def test_config_fingerprint_tracks_runtime_cache_quantization_and_ssm_layout(self):
        self.server._cfg = SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                seq_size_per_block=16,
                kernel_seq_size_per_block=16,
                fp8_kv_cache=False,
                linear_step=1,
                ssm_state_dtype="float32",
            ),
            quantization_config=SimpleNamespace(int8_mode=0, quantization="none"),
        )
        first = self.server.get_trajectory_migration_capabilities()["rollout_config_fingerprint"]
        for field, value in (("fp8_kv_cache", True), ("ssm_state_dtype", "float16"), ("kernel_seq_size_per_block", 64)):
            with self.subTest(field=field):
                previous = getattr(self.server._cfg.kv_cache_config, field)
                setattr(self.server._cfg.kv_cache_config, field, value)
                self.assertNotEqual(
                    first, self.server.get_trajectory_migration_capabilities()["rollout_config_fingerprint"]
                )
                setattr(self.server._cfg.kv_cache_config, field, previous)
        self.server._cfg.quantization_config.quantization = "fp8"
        self.assertNotEqual(first, self.server.get_trajectory_migration_capabilities()["rollout_config_fingerprint"])
