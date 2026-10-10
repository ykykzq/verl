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
import itertools
import sys
import unittest
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server import RTPLLMHttpServer


class TestRTPLLMKVCacheTransition(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _server(*, generation_allowed: bool, abort_requested: bool, inflight: dict | None = None):
        server = RTPLLMHttpServer.__new__(RTPLLMHttpServer)
        server._generation_allowed = asyncio.Event()
        if generation_allowed:
            server._generation_allowed.set()
        server._abort_requested = abort_requested
        server._rejecting = False
        server._request_admission_lock = asyncio.Lock()
        server._inflight = {} if inflight is None else inflight
        server._accepted_migrations = {}
        clear_kv_cache = Mock()
        server.engine = SimpleNamespace(
            clear_kv_cache=clear_kv_cache,
            onflight_request_num=Mock(return_value=0),
            set_cache_key_salt=Mock(),
        )
        return server, clear_kv_cache

    async def test_clear_requires_generation_to_be_paused(self):
        server, clear_kv_cache = self._server(generation_allowed=True, abort_requested=False)

        with self.assertRaisesRegex(RuntimeError, "paused"):
            await server.clear_kv_cache()

        clear_kv_cache.assert_not_called()

    async def test_clear_rejects_registered_requests(self):
        server, clear_kv_cache = self._server(
            generation_allowed=False,
            abort_requested=True,
            inflight={"request-1": object()},
        )

        with self.assertRaisesRegex(RuntimeError, "in flight"):
            await server.clear_kv_cache()

        clear_kv_cache.assert_not_called()

    async def test_clear_calls_binding_after_drain(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)

        await server.clear_kv_cache()

        clear_kv_cache.assert_called_once_with()

    async def test_clear_waits_for_engine_requests_to_drain(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)
        server.engine.onflight_request_num.side_effect = [2, 1, 0]

        await server.clear_kv_cache()

        self.assertEqual(server.engine.onflight_request_num.call_count, 3)
        clear_kv_cache.assert_called_once_with()

    async def test_clear_waits_for_cache_references_after_rpc_drain(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)
        server.global_steps = 6
        clear_kv_cache.side_effect = [
            RuntimeError("clear_kv_cache refused: active/resident cache resources remain"),
            RuntimeError("clear_kv_cache refused while requests are in flight: 1"),
            None,
        ]
        await server.clear_kv_cache(interval_s=0)
        self.assertEqual(clear_kv_cache.call_count, 3)
        self.assertFalse(server._generation_allowed.is_set())
        self.assertEqual(server.global_steps, 6)

    async def test_clear_cache_resource_timeout_fails_closed(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)
        server.global_steps = 6
        clear_kv_cache.side_effect = RuntimeError("clear_kv_cache refused: active/resident cache resources remain")
        with self.assertRaisesRegex(RuntimeError, "cache resources did not drain"):
            await server.clear_kv_cache(timeout_s=0, interval_s=0)
        clear_kv_cache.assert_called_once_with()
        self.assertFalse(server._generation_allowed.is_set())
        self.assertEqual(server.global_steps, 6)

    async def test_clear_does_not_retry_unrelated_errors(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)
        clear_kv_cache.side_effect = RuntimeError("binding unavailable")
        with self.assertRaisesRegex(RuntimeError, "binding unavailable"):
            await server.clear_kv_cache()
        clear_kv_cache.assert_called_once_with()
        self.assertFalse(server._generation_allowed.is_set())

    async def test_clear_cache_wait_propagates_cancellation(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)
        clear_kv_cache.side_effect = RuntimeError("clear_kv_cache refused: active/resident cache resources remain")
        task = asyncio.create_task(server.clear_kv_cache(interval_s=60))
        while not clear_kv_cache.called:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(server._generation_allowed.is_set())
        self.assertFalse(server._request_admission_lock.locked())

    async def test_engine_drain_timeout_fails_closed(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)
        server.engine.onflight_request_num.return_value = 1

        with self.assertRaisesRegex(RuntimeError, "still has 1 request"):
            await server._await_engine_drain(timeout_s=0, interval_s=0)

        clear_kv_cache.assert_not_called()

    async def test_release_hook_uses_the_same_transition(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)

        await server.release_kv_cache()

        clear_kv_cache.assert_called_once_with()

    async def test_abort_accepts_reject_request_and_resets_it_on_plain_abort(self):
        server, _ = self._server(generation_allowed=True, abort_requested=False)

        await server.abort_all_requests(reject_request=True)

        self.assertTrue(server._rejecting)
        self.assertTrue(server._abort_requested)
        self.assertFalse(server._generation_allowed.is_set())

        await server.abort_all_requests()

        self.assertFalse(server._rejecting)

    async def test_abort_request_only_cancels_matching_task(self):
        matching = asyncio.create_task(asyncio.Event().wait())
        other = asyncio.create_task(asyncio.Event().wait())
        server, _ = self._server(
            generation_allowed=True,
            abort_requested=False,
            inflight={"request-1": matching, "request-2": other},
        )

        result = await server.abort_request("request-1")

        self.assertTrue(result["aborted"])
        self.assertTrue(matching.cancelled())
        self.assertFalse(other.done())
        other.cancel()
        with suppress(asyncio.CancelledError):
            await other

    async def test_transfer_ticket_waits_for_remote_cache_and_covers_prefix(self):
        server, _ = self._server(generation_allowed=True, abort_requested=False)
        server.replica_rank = 0
        server.global_steps = 9
        server.config = SimpleNamespace(
            trajectory_migration=SimpleNamespace(enabled=True, kv_transfer_backend="remote_prefix"),
            max_num_seqs=8,
            enable_prefix_caching=True,
        )
        server.model_config = SimpleNamespace(
            local_path="/model",
            hf_config=SimpleNamespace(architectures=["ModelForCausalLM"]),
        )
        wait_remote_cache_idle = Mock(return_value=True)
        server.engine.wait_remote_cache_idle = wait_remote_cache_idle
        state = {
            "request_id": "trajectory-1",
            "prompt_ids": [1, 2],
            "generated_token_ids": [3, 4],
        }
        target = server.get_trajectory_migration_capabilities()

        ticket = await server.prepare_trajectory_migration(state, target, "remote_prefix", timeout_s=2.0)
        accepted = await server.accept_trajectory_migration(ticket, state)

        wait_remote_cache_idle.assert_called_once_with(2000)
        self.assertEqual(ticket["prefix_tokens"], 4)
        self.assertTrue(accepted["accepted"])
        self.assertIn("trajectory-1", server._accepted_migrations)

    async def test_weight_version_updates_cache_key_namespace_before_publication(self):
        server, _ = self._server(generation_allowed=False, abort_requested=True)
        server.replica_rank = 0
        server.global_steps = None
        server.config = SimpleNamespace(
            trajectory_migration=SimpleNamespace(
                enabled=True,
                kv_transfer_backend="remote_prefix",
                backend_options={"instance_group": "training-job"},
            ),
            max_num_seqs=8,
            enable_prefix_caching=True,
        )
        server.model_config = SimpleNamespace(
            local_path="/model",
            hf_config=SimpleNamespace(architectures=["ModelForCausalLM"]),
        )

        expected_salt = server._cache_key_salt(10)
        await server.set_global_steps(10)

        server.engine.set_cache_key_salt.assert_called_once_with(expected_salt)
        self.assertEqual(server.global_steps, 10)
        self.assertEqual(server.get_trajectory_migration_capabilities()["kv_transfer_namespace"], expected_salt)


class TestRTPLLMTrajectoryAdmission(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server, _ = TestRTPLLMKVCacheTransition._server(generation_allowed=True, abort_requested=False)
        self.server.replica_rank = 1
        self.server.global_steps = 10
        self.server._request_counter = itertools.count(1)
        self.server._weight_update_lock = asyncio.Lock()
        self.server._weight_update = None
        self.server._weight_sync_error = None
        self.server.eos_token_id = 99
        self.server.config = SimpleNamespace(
            trajectory_migration=SimpleNamespace(
                enabled=True, kv_transfer_backend="remote_prefix", allow_cross_version_recompute=True
            ),
            max_num_seqs=8,
            enable_prefix_caching=True,
            response_length=8,
            prompt_length=16,
            max_model_len=24,
        )
        self.server.model_config = SimpleNamespace(
            local_path="/model", hf_config=SimpleNamespace(architectures=["ModelForCausalLM"])
        )
        self.state = {"request_id": "trajectory-1", "prompt_ids": [1, 2], "generated_token_ids": [3, 4]}
        self.ticket = {
            "backend": "recompute",
            "mode": "recompute",
            "model_id": "/model|ModelForCausalLM",
            "request_id": "trajectory-1",
            "prefix_tokens": 4,
            "prefix_digest": "73e200e2b048c86d4e8c86b86bf62bbda84c7384e34e250b01aa30ab29d234a4",
            "source_version": 9,
            "target_version": 10,
        }
        self.inputs = []

        async def enqueue(generate_input):
            self.inputs.append(generate_input)

            async def stream():
                yield SimpleNamespace(generate_outputs=[SimpleNamespace(output_ids=torch.tensor([5]), aux_info=None)])

            return stream()

        self.server.visitor = SimpleNamespace(enqueue=enqueue)
        self.rtp_types = patch.dict(
            sys.modules,
            {
                "rtp_llm.config.generate_config": SimpleNamespace(GenerateConfig=SimpleNamespace),
                "rtp_llm.utils.base_model_datatypes": SimpleNamespace(GenerateInput=SimpleNamespace),
            },
        )
        self.rtp_types.start()
        self.addCleanup(self.rtp_types.stop)

    def remote_ticket(self):
        capabilities = self.server.get_trajectory_migration_capabilities()
        return {
            key: value
            for key, value in {
                **self.ticket,
                **capabilities,
                "backend": "remote_prefix",
            }.items()
            if key not in {"mode", "source_version", "target_version"}
        }

    async def test_recompute_accepts_without_remote_cache_support(self):
        self.server.config.enable_prefix_caching = False
        accepted = await self.server.accept_trajectory_migration(self.ticket, self.state)
        self.assertTrue(accepted["accepted"], accepted)

    async def test_recompute_ticket_rejects_invalid_identity_and_versions(self):
        for field, value, reason in [
            ("model_id", "another-model", "model"),
            ("request_id", "another-request", "request"),
            ("prefix_tokens", 3, "prefix"),
            ("prefix_digest", "another-prefix", "prefix"),
            ("target_version", 11, "version"),
            ("source_version", None, "version"),
            ("source_version", 10, "version"),
            ("mode", "remote_prefix", "mode"),
        ]:
            with self.subTest(field=field, value=value):
                accepted = await self.server.accept_trajectory_migration({**self.ticket, field: value}, self.state)
                self.assertFalse(accepted["accepted"])
                self.assertIn(reason, accepted["reason"])
                self.assertEqual(self.server._accepted_migrations, {})

    async def test_recompute_requires_explicit_enable(self):
        self.server.config.trajectory_migration.allow_cross_version_recompute = False
        accepted = await self.server.accept_trajectory_migration(self.ticket, self.state)
        self.assertFalse(accepted["accepted"])

    async def test_recompute_requires_explicit_mode_to_force_cache_disable(self):
        ticket = {key: value for key, value in self.ticket.items() if key != "mode"}
        accepted = await self.server.accept_trajectory_migration(ticket, self.state)
        self.assertFalse(accepted["accepted"], accepted)

    async def test_recompute_rejects_engine_override_of_request_cache_switches(self):
        for override in ["1", "true", "TRUE"]:
            with (
                self.subTest(override=override),
                patch.dict("os.environ", {"RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES": override}),
            ):
                accepted = await self.server.accept_trajectory_migration(self.ticket, self.state)
                self.assertFalse(accepted["accepted"], accepted)
                self.assertIn("cache", accepted["reason"])

    async def test_recompute_rechecks_engine_cache_override_before_enqueue(self):
        await self.server.accept_trajectory_migration(self.ticket, self.state)
        with patch.dict("os.environ", {"RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES": "1"}):
            result = await self.server.generate(
                [1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1", migration_ticket=self.ticket
            )
        self.assertEqual(result.stop_reason, "aborted")
        self.assertTrue(result.extra_fields["migration_rejected"])
        self.assertEqual(self.inputs, [])

    async def test_closed_admission_rejects_all_migration_modes(self):
        for ticket in [self.ticket, self.remote_ticket()]:
            for state in ["paused", "aborting", "rejecting"]:
                with self.subTest(backend=ticket["backend"], state=state):
                    self.server._generation_allowed.set()
                    self.server._abort_requested = state == "aborting"
                    self.server._rejecting = state == "rejecting"
                    if state == "paused":
                        self.server._generation_allowed.clear()
                    accepted = await self.server.accept_trajectory_migration(ticket, self.state)
                    self.assertFalse(accepted["accepted"], accepted)
                    self.assertIn("admission", accepted["reason"])

    async def test_recompute_forces_prefill_only_for_the_migrated_request(self):
        accepted = await self.server.accept_trajectory_migration(self.ticket, self.state)
        self.assertTrue(accepted["accepted"], accepted)
        result = await self.server.generate(
            [1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1", migration_ticket=self.ticket
        )
        normal = await self.server.generate([6, 7], {"max_tokens": 1}, "ordinary-request")
        self.assertEqual(result.token_ids, [5])
        self.assertEqual(normal.token_ids, [5])
        self.assertEqual(result.extra_fields["forced_prefill_tokens"], 4)
        self.assertNotIn("forced_prefill_tokens", normal.extra_fields)
        self.assertFalse(self.inputs[0].generate_config.reuse_cache)
        self.assertFalse(self.inputs[0].generate_config.enable_remote_cache)
        self.assertTrue(self.inputs[1].generate_config.reuse_cache)
        self.assertTrue(self.inputs[1].generate_config.enable_remote_cache)
        self.assertEqual(self.server._accepted_migrations, {})

    async def test_remote_prefix_preserves_cache_reuse(self):
        ticket = self.remote_ticket()
        accepted = await self.server.accept_trajectory_migration(ticket, self.state)
        self.assertTrue(accepted["accepted"], accepted)
        result = await self.server.generate([1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1", migration_ticket=ticket)
        self.assertNotIn("forced_prefill_tokens", result.extra_fields)
        self.assertTrue(self.inputs[0].generate_config.reuse_cache)
        self.assertTrue(self.inputs[0].generate_config.enable_remote_cache)

    async def test_aborted_recompute_counts_prefill_only_after_tokens_were_produced(self):
        for produced_tokens in [[], [5]]:
            with self.subTest(produced_tokens=produced_tokens):
                await self.server.accept_trajectory_migration(self.ticket, self.state)

                async def enqueue(generate_input, output_tokens=produced_tokens):
                    async def stream():
                        yield SimpleNamespace(
                            generate_outputs=[
                                SimpleNamespace(
                                    output_ids=torch.tensor(output_tokens, dtype=torch.int32), aux_info=None
                                )
                            ]
                        )
                        raise RuntimeError("stream interrupted")

                    return stream()

                self.server.visitor.enqueue = enqueue
                result = await self.server.generate(
                    [1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1", migration_ticket=self.ticket
                )
                self.assertEqual(result.stop_reason, "aborted")
                self.assertEqual(result.token_ids, produced_tokens)
                if produced_tokens:
                    self.assertEqual(result.extra_fields["forced_prefill_tokens"], 4)
                else:
                    self.assertNotIn("forced_prefill_tokens", result.extra_fields)

    async def test_recompute_does_not_report_first_token_latency_as_prefill_duration(self):
        await self.server.accept_trajectory_migration(self.ticket, self.state)

        async def enqueue(generate_input):
            async def stream():
                for token in [5, 6]:
                    yield SimpleNamespace(
                        generate_outputs=[
                            SimpleNamespace(
                                output_ids=torch.tensor([token]),
                                aux_info=SimpleNamespace(
                                    softmax_probs=[],
                                    input_len=4,
                                    prefix_len=0,
                                    reuse_len=0,
                                    first_token_cost_time=125.5,
                                    wait_time=20.0,
                                    cost_time=180.0,
                                ),
                            )
                        ]
                    )

            return stream()

        self.server.visitor.enqueue = enqueue
        result = await self.server.generate(
            [1, 2, 3, 4], {"max_tokens": 2}, "trajectory-1", migration_ticket=self.ticket
        )
        self.assertEqual(result.extra_fields["forced_prefill_tokens"], 4)
        self.assertNotIn("forced_prefill_duration_s", result.extra_fields)

    async def test_version_change_after_acceptance_aborts_before_enqueue(self):
        for ticket in [self.ticket, self.remote_ticket()]:
            with self.subTest(backend=ticket["backend"]):
                self.server.global_steps = 10
                accepted = await self.server.accept_trajectory_migration(ticket, self.state)
                self.assertTrue(accepted["accepted"], accepted)
                self.server.global_steps = 11
                result = await self.server.generate(
                    [1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1", migration_ticket=ticket
                )
                self.assertEqual(result.stop_reason, "aborted")
                self.assertTrue(result.extra_fields["migration_rejected"])
                self.assertEqual(result.token_ids, [])
                self.assertNotIn("forced_prefill_tokens", result.extra_fields)
                self.assertEqual(self.inputs, [])
                self.assertEqual(self.server._accepted_migrations, {})

    async def test_enqueue_revalidates_under_admission_lock(self):
        ticket = self.remote_ticket()
        await self.server.accept_trajectory_migration(ticket, self.state)
        await self.server._request_admission_lock.acquire()
        task = asyncio.create_task(self.server.generate([1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1"))
        await asyncio.sleep(0)
        self.server.global_steps = 11
        self.server._request_admission_lock.release()
        result = await task
        self.assertEqual(result.stop_reason, "aborted")
        self.assertTrue(result.extra_fields["migration_rejected"])
        self.assertEqual(self.inputs, [])

    async def test_enqueue_revalidates_ticket_and_prefix_identity(self):
        for changed_ticket, request_id, prefix in [
            ({**self.ticket, "model_id": "another-model"}, "trajectory-1", [1, 2, 3, 4]),
            (self.ticket, "another-request", [1, 2, 3, 4]),
            (self.ticket, "trajectory-1", [1, 2, 3]),
            (self.ticket, "trajectory-1", [1, 2, 3, 99]),
        ]:
            with self.subTest(ticket=changed_ticket, request_id=request_id, prefix=prefix):
                accepted = await self.server.accept_trajectory_migration(self.ticket, self.state)
                self.assertTrue(accepted["accepted"], accepted)
                result = await self.server.generate(
                    prefix, {"max_tokens": 1}, request_id, migration_ticket=changed_ticket
                )
                self.assertEqual(result.stop_reason, "aborted")
                self.assertTrue(result.extra_fields["migration_rejected"])
                self.assertEqual(self.inputs, [])

    async def test_abort_clears_acceptance_and_old_ticket_cannot_generate_after_resume(self):
        ticket = self.remote_ticket()
        await self.server.accept_trajectory_migration(ticket, self.state)
        await self.server.abort_all_requests(reject_request=True)
        self.assertEqual(self.server._accepted_migrations, {})
        await self.server.resume_generation()
        result = await self.server.generate([1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1", migration_ticket=ticket)
        self.assertEqual(result.stop_reason, "aborted")
        self.assertTrue(result.extra_fields["migration_rejected"])
        self.assertEqual(self.inputs, [])

    async def test_ticketless_continuation_cannot_cross_weight_versions(self):
        for source_version in [9, "initial"]:
            with self.subTest(source_version=source_version):
                result = await self.server.generate(
                    [1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1", expected_source_version=source_version
                )
                self.assertEqual(result.stop_reason, "aborted")
                self.assertTrue(result.extra_fields["migration_rejected"])
                self.assertEqual(self.inputs, [])

    async def test_ticketless_continuation_allows_same_weight_version(self):
        result = await self.server.generate([1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1", expected_source_version=10)
        self.assertEqual(result.token_ids, [5])
        self.assertTrue(self.inputs[0].generate_config.reuse_cache)

    async def test_valid_recompute_ticket_allows_source_version_to_differ(self):
        await self.server.accept_trajectory_migration(self.ticket, self.state)
        result = await self.server.generate(
            [1, 2, 3, 4],
            {"max_tokens": 1},
            "trajectory-1",
            migration_ticket=self.ticket,
            expected_source_version=9,
        )
        self.assertEqual(result.token_ids, [5])
        self.assertFalse(self.inputs[0].generate_config.reuse_cache)

    async def test_segment_reports_weight_version_captured_at_admission(self):
        generated = asyncio.Event()
        finish_stream = asyncio.Event()

        async def enqueue(generate_input):
            async def stream():
                yield SimpleNamespace(generate_outputs=[SimpleNamespace(output_ids=torch.tensor([5]), aux_info=None)])
                generated.set()
                await finish_stream.wait()

            return stream()

        self.server.visitor.enqueue = enqueue
        task = asyncio.create_task(self.server.generate([1, 2, 3, 4], {"max_tokens": 1}, "trajectory-1"))
        await generated.wait()
        await self.server.set_global_steps(11)
        finish_stream.set()
        result = await task
        self.assertEqual(result.token_ids, [5])
        self.assertEqual(result.extra_fields["global_steps"], 10)


class TestRTPLLMWeightRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_manager_can_abort_sender_owned_round_and_retry_full_stream(self):
        server, _ = TestRTPLLMKVCacheTransition._server(generation_allowed=False, abort_requested=True)
        server._weight_update_lock = asyncio.Lock()
        server._weight_update = None
        server._weight_sync_error = None
        buffer = torch.tensor([1.0, 2.0]).view(torch.uint8)
        applied = []
        aborts = []

        def update(weights, is_last):
            applied.append(([name for name, _ in weights], is_last))
            if len(applied) == 2:
                raise RuntimeError("partial weight stream failed")

        server.weight_manager = SimpleNamespace(
            update_from_hf_tensors=update, abort_hf_update=lambda: aborts.append(True)
        )
        device = SimpleNamespace(synchronize=lambda: None, ipc_collect=lambda: None)
        entry = {"name": "first.weight", "shape": (2,), "dtype": "float32", "offset": 0, "handle": None}
        module = "verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server"
        with (
            patch(f"{module}.rebuild_ipc_tensor", return_value=buffer),
            patch(f"{module}.get_torch_device", return_value=device),
            patch(f"{module}.get_device_id", return_value=0),
        ):
            await server.begin_weight_update_from_ipc("sender-round-1", {"method": "cuda_ipc", "payload": b"handle"})
            await server.apply_weight_bucket_from_ipc("sender-round-1", 0, [entry], False)
            with self.assertRaisesRegex(RuntimeError, "partial weight stream failed"):
                await server.apply_weight_bucket_from_ipc(
                    "sender-round-1", 1, [{**entry, "name": "second.weight"}], True
                )
            await server.abort_weight_update_from_ipc()
            self.assertEqual(aborts, [True])
            self.assertIsNone(server._weight_update)
            self.assertFalse(server._generation_allowed.is_set())
            with self.assertRaisesRegex(RuntimeError, "cannot resume generation"):
                await server.resume_generation()
            await server.begin_weight_update_from_ipc("sender-round-2", {"method": "cuda_ipc", "payload": b"handle"})
            await server.apply_weight_bucket_from_ipc("sender-round-2", 0, [entry], False)
            await server.apply_weight_bucket_from_ipc("sender-round-2", 1, [{**entry, "name": "second.weight"}], True)
            result = await server.finish_weight_update_from_ipc("sender-round-2")
            await server.resume_generation()
        self.assertEqual(result["tensor_count"], 2)
        self.assertEqual(applied, [(["first.weight"], False), (["second.weight"], True)] * 2)
        self.assertTrue(server._generation_allowed.is_set())
        self.assertIsNone(server._weight_sync_error)
