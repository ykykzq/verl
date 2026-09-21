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
        with patch(
            "verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server.get_torch_device",
            return_value=_DeviceApi(),
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
