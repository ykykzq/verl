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
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

from verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer import RayIpcWeightSender
from verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server import RTPLLMHttpServer


class _DeviceApi:
    def synchronize(self):
        pass

    def ipc_collect(self):
        pass

    def empty_cache(self):
        pass


class _RemoteMethod:
    def __init__(self, method):
        self.remote = method


class TestRayIpcWeightSender(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _sender(failure_at=None):
        calls = []
        failure = RuntimeError(f"{failure_at} failed")

        async def begin(**kwargs):
            calls.append("begin")
            if failure_at == "begin":
                raise failure

        async def apply(**kwargs):
            calls.append("apply")
            if failure_at == "bucket":
                raise failure

        async def finish(**kwargs):
            calls.append("finish")

        async def abort(**kwargs):
            calls.append("abort")

        sender = RayIpcWeightSender(
            SimpleNamespace(
                begin_weight_update_from_ipc=_RemoteMethod(begin),
                apply_weight_bucket_from_ipc=_RemoteMethod(apply),
                finish_weight_update_from_ipc=_RemoteMethod(finish),
                abort_weight_update_from_ipc=_RemoteMethod(abort),
            ),
            bucket_size_mb=1,
        )
        sender.bucket_size = 8

        def init_buffer():
            sender.buffer = torch.empty(sender.bucket_size, dtype=torch.uint8)
            if failure_at == "buffer":
                raise failure
            return {"method": "cuda_ipc", "payload": b"handle"}

        return sender, init_buffer, calls, failure

    async def test_failed_ipc_drains_collective_iterator_before_returning(self):
        for failure_at in ["buffer", "begin", "bucket"]:
            with self.subTest(failure_at=failure_at):
                sender, init_buffer, calls, failure = self._sender(failure_at)
                received = []
                all_collectives_complete = asyncio.Event()

                async def weights(received=received, complete=all_collectives_complete):
                    for index in range(4):
                        received.append(index)
                        yield f"weight.{index}", torch.tensor([1.0, 2.0])
                    complete.set()

                trainer = asyncio.create_task(all_collectives_complete.wait())
                try:
                    with (
                        patch.object(sender, "_init_buffer", side_effect=init_buffer),
                        patch(
                            "verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer.get_torch_device",
                            return_value=_DeviceApi(),
                        ),
                        self.assertRaises(RuntimeError) as raised,
                    ):
                        await sender.async_send_weights(weights())
                    self.assertIs(raised.exception, failure)
                    self.assertEqual(received, [0, 1, 2, 3])
                    self.assertTrue(all_collectives_complete.is_set())
                    await trainer
                    self.assertIsNone(sender.buffer)
                    self.assertNotIn("finish", calls)
                    self.assertEqual(calls.count("apply"), int(failure_at == "bucket"))
                finally:
                    trainer.cancel()
                    await asyncio.gather(trainer, return_exceptions=True)

    async def test_failed_weight_iterator_is_not_restarted_or_drained(self):
        sender, init_buffer, calls, _ = self._sender()
        failure = RuntimeError("collective iterator failed")

        class Weights:
            iterations = 0
            next_calls = 0

            def __aiter__(self):
                self.iterations += 1
                return self

            async def __anext__(self):
                self.next_calls += 1
                if self.next_calls == 1:
                    return "weight.0", torch.tensor([1.0, 2.0])
                raise failure

        weights = Weights()
        with (
            patch.object(sender, "_init_buffer", side_effect=init_buffer),
            patch(
                "verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer.get_torch_device", return_value=_DeviceApi()
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            await sender.async_send_weights(weights)
        self.assertIs(raised.exception, failure)
        self.assertEqual(weights.iterations, 1)
        self.assertEqual(weights.next_calls, 2)
        self.assertEqual(calls, ["begin", "abort"])
        self.assertIsNone(sender.buffer)

    async def test_cancellation_during_collective_wait_preserves_remaining_drain(self):
        sender, init_buffer, calls, _ = self._sender()
        waiting = asyncio.Event()
        release_collective = asyncio.Event()
        all_collectives_complete = asyncio.Event()
        received = []

        async def weights():
            for index in range(4):
                if index == 1:
                    waiting.set()
                    await release_collective.wait()
                received.append(index)
                yield f"weight.{index}", torch.tensor([1.0, 2.0])
            all_collectives_complete.set()

        with (
            patch.object(sender, "_init_buffer", side_effect=init_buffer),
            patch(
                "verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer.get_torch_device", return_value=_DeviceApi()
            ),
        ):
            task = asyncio.create_task(sender.async_send_weights(weights()))
            await waiting.wait()
            task.cancel()
            release_collective.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(received, [0, 1, 2, 3])
        self.assertTrue(all_collectives_complete.is_set())
        self.assertEqual(calls, ["begin", "abort"])
        self.assertIsNone(sender.buffer)

    async def test_cancellation_during_ipc_wait_drains_without_sending_more_buckets(self):
        sender, init_buffer, calls, _ = self._sender()
        waiting = asyncio.Event()
        received = []

        async def blocked_apply(**kwargs):
            calls.append("apply")
            waiting.set()
            await asyncio.Event().wait()

        sender.server_handle.apply_weight_bucket_from_ipc.remote = blocked_apply

        async def weights():
            for index in range(4):
                received.append(index)
                yield f"weight.{index}", torch.tensor([1.0, 2.0])

        with (
            patch.object(sender, "_init_buffer", side_effect=init_buffer),
            patch(
                "verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer.get_torch_device", return_value=_DeviceApi()
            ),
        ):
            task = asyncio.create_task(sender.async_send_weights(weights()))
            await waiting.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(received, [0, 1, 2, 3])
        self.assertEqual(calls, ["begin", "apply", "abort"])
        self.assertIsNone(sender.buffer)

    async def test_drain_failure_preserves_original_ipc_error_and_cleans_resources(self):
        sender, init_buffer, calls, failure = self._sender("bucket")
        drained = []

        async def weights():
            for index in range(3):
                drained.append(index)
                yield f"weight.{index}", torch.tensor([1.0, 2.0])
            raise RuntimeError("later collective failure")

        with (
            patch.object(sender, "_init_buffer", side_effect=init_buffer),
            patch(
                "verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer.get_torch_device", return_value=_DeviceApi()
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            await sender.async_send_weights(weights())
        self.assertIs(raised.exception, failure)
        self.assertEqual(drained, [0, 1, 2])
        self.assertEqual(calls, ["begin", "apply", "abort"])
        self.assertIsNone(sender.buffer)

    async def test_cleanup_failure_does_not_replace_the_ipc_error(self):
        sender, init_buffer, _, failure = self._sender("bucket")
        cleanup = sender._cleanup

        def failing_cleanup():
            cleanup()
            raise RuntimeError("cleanup failed")

        with (
            patch.object(sender, "_init_buffer", side_effect=init_buffer),
            patch.object(sender, "_cleanup", side_effect=failing_cleanup),
            patch(
                "verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer.get_torch_device", return_value=_DeviceApi()
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            await sender.async_send_weights([("weight.0", torch.tensor([1.0, 2.0]))])
        self.assertIs(raised.exception, failure)
        self.assertIsNone(sender.buffer)

    async def test_successful_sender_drains_and_commits_every_bucket(self):
        sender, init_buffer, calls, _ = self._sender()
        with (
            patch.object(sender, "_init_buffer", side_effect=init_buffer),
            patch(
                "verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer.get_torch_device", return_value=_DeviceApi()
            ),
        ):
            await sender.async_send_weights([(f"weight.{index}", torch.tensor([1.0, 2.0])) for index in range(4)])
        self.assertEqual(calls, ["begin", "apply", "apply", "apply", "apply", "finish"])
        self.assertIsNone(sender.buffer)

    async def test_sender_uses_metadata_only_and_aborts_failed_round(self):
        begin = AsyncMock()
        apply = AsyncMock(side_effect=RuntimeError("receiver failed"))
        finish = AsyncMock()
        abort = AsyncMock()
        server = SimpleNamespace(
            begin_weight_update_from_ipc=_RemoteMethod(begin),
            apply_weight_bucket_from_ipc=_RemoteMethod(apply),
            finish_weight_update_from_ipc=_RemoteMethod(finish),
            abort_weight_update_from_ipc=_RemoteMethod(abort),
        )
        sender = RayIpcWeightSender(server, bucket_size_mb=1, use_shm=False)
        sender.buffer = torch.empty(sender.bucket_size, dtype=torch.uint8)

        with (
            patch.object(sender, "_init_buffer", return_value={"method": "cuda_ipc", "payload": b"handle"}),
            patch.object(sender, "_cleanup") as cleanup,
            patch(
                "verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer.get_torch_device",
                return_value=_DeviceApi(),
            ),
            self.assertRaisesRegex(RuntimeError, "receiver failed"),
        ):
            await sender.async_send_weights([("model.weight", torch.tensor([1.0, 2.0]))])

        descriptor = begin.call_args.kwargs["descriptor"]
        entries = apply.call_args.kwargs["entries"]
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in descriptor.values()))
        self.assertFalse(any(isinstance(value, torch.Tensor) for entry in entries for value in entry.values()))
        finish.assert_not_awaited()
        abort.assert_awaited_once()
        cleanup.assert_called_once_with()


class TestRayIpcWeightUpdate(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _server():
        server = RTPLLMHttpServer.__new__(RTPLLMHttpServer)
        server._weight_update_lock = asyncio.Lock()
        source = torch.tensor([1.0, 2.0], dtype=torch.float32)
        buffer = torch.empty(source.nbytes, dtype=torch.uint8)
        buffer.view(dtype=source.dtype).copy_(source)
        server._weight_update = {
            "round_id": "round-1",
            "method": "cuda_ipc",
            "buffer": buffer,
            "shm": None,
            "next_sequence": 0,
            "names": set(),
            "saw_last": False,
            "tensor_count": 0,
        }
        server._weight_sync_error = "weight update round-1 has not committed"
        server.weight_manager = SimpleNamespace(
            update_from_hf_tensors=Mock(),
            abort_hf_update=Mock(),
        )
        return server

    @staticmethod
    def _entry(name: str = "model.weight"):
        return {
            "name": name,
            "shape": (2,),
            "dtype": "float32",
            "offset": 0,
            "handle": None,
        }

    async def test_begin_requires_generation_to_be_paused(self):
        server = RTPLLMHttpServer.__new__(RTPLLMHttpServer)
        server._weight_update_lock = asyncio.Lock()
        server._request_admission_lock = asyncio.Lock()
        server._weight_update = None
        server._generation_allowed = asyncio.Event()
        server._generation_allowed.set()
        server._abort_requested = False
        server._inflight = {}

        with self.assertRaisesRegex(RuntimeError, "paused"):
            await server.begin_weight_update_from_ipc("round-1", {"method": "invalid"})

    async def test_bucket_ack_follows_weight_application_and_commit(self):
        server = self._server()
        with (
            patch(
                "verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server.get_torch_device",
                return_value=_DeviceApi(),
            ),
            patch("verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server.get_device_id", return_value=0),
        ):
            ack = await server.apply_weight_bucket_from_ipc(
                round_id="round-1",
                sequence=0,
                entries=[self._entry()],
                is_last=True,
            )
            result = await server.finish_weight_update_from_ipc("round-1")

        self.assertEqual(ack["sequence"], 0)
        self.assertEqual(result["bucket_count"], 1)
        self.assertEqual(result["tensor_count"], 1)
        server.weight_manager.update_from_hf_tensors.assert_called_once()
        weights = server.weight_manager.update_from_hf_tensors.call_args.args[0]
        torch.testing.assert_close(weights[0][1], torch.tensor([1.0, 2.0]))
        self.assertTrue(server.weight_manager.update_from_hf_tensors.call_args.kwargs["is_last"])
        self.assertIsNone(server._weight_update)
        self.assertIsNone(server._weight_sync_error)

    async def test_begin_rejects_weight_update_while_generation_is_open(self):
        server = RTPLLMHttpServer.__new__(RTPLLMHttpServer)
        server._request_admission_lock = asyncio.Lock()
        server._weight_update_lock = asyncio.Lock()
        server._generation_allowed = asyncio.Event()
        server._generation_allowed.set()
        server._abort_requested = False
        server._inflight = {}
        server._weight_update = None

        with self.assertRaisesRegex(RuntimeError, "generation to be paused"):
            await server.begin_weight_update_from_ipc("round-1", {"method": "cuda_ipc", "payload": b"unused"})

    async def test_out_of_order_bucket_is_rejected_before_application(self):
        server = self._server()

        with self.assertRaisesRegex(RuntimeError, "expected bucket 0, got 1"):
            await server.apply_weight_bucket_from_ipc(
                round_id="round-1",
                sequence=1,
                entries=[self._entry()],
                is_last=False,
            )

        server.weight_manager.update_from_hf_tensors.assert_not_called()

    async def test_abort_discards_buffered_sources_and_keeps_generation_closed(self):
        server = self._server()
        server._request_admission_lock = asyncio.Lock()
        server._generation_allowed = asyncio.Event()
        server._abort_requested = True

        with patch(
            "verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server.get_torch_device",
            return_value=_DeviceApi(),
        ):
            await server.abort_weight_update_from_ipc("round-1")

        server.weight_manager.abort_hf_update.assert_called_once_with()
        self.assertIsNone(server._weight_update)
        with self.assertRaisesRegex(RuntimeError, "cannot resume generation"):
            await server.resume_generation()
