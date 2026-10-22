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
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer import RayIpcWeightSender
from verl.workers.rollout.rtp_llm_rollout.rtp_llm_rollout import ServerAdapter


class _FailingCastTensor:
    dtype = torch.float32

    def __init__(self, failure):
        self.failure = failure
        self.cast_count = 0

    def detach(self):
        return self

    def is_floating_point(self):
        return True

    def to(self, dtype):
        self.cast_count += 1
        raise self.failure


class _WeightSource:
    def __init__(self, tensors, failure_at=None, failure=None):
        self.tensors = tensors
        self.failure_at = failure_at
        self.failure = failure
        self.iterations = 0
        self.next_calls = 0
        self.received = []
        self.completed = asyncio.Event()

    def __aiter__(self):
        self.iterations += 1
        return self

    async def __anext__(self):
        index = self.next_calls
        self.next_calls += 1
        if index == self.failure_at:
            raise self.failure
        if index == len(self.tensors):
            self.completed.set()
            raise StopAsyncIteration
        self.received.append(index)
        return f"weight.{index}", self.tensors[index]


class TestRTPWeightAdapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.adapter = ServerAdapter.__new__(ServerAdapter)
        self.adapter._engine_dtype = torch.bfloat16
        self.adapter._has_server = True
        self.adapter.replica_rank = 0
        self.adapter.rollout_rank = 0
        self.adapter.use_shm = False
        self.adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1))
        self.adapter.server_handle = SimpleNamespace(
            begin_weight_update_from_ipc=SimpleNamespace(remote=AsyncMock()),
            apply_weight_bucket_from_ipc=SimpleNamespace(remote=AsyncMock()),
            finish_weight_update_from_ipc=SimpleNamespace(remote=AsyncMock()),
            abort_weight_update_from_ipc=SimpleNamespace(remote=AsyncMock()),
            set_global_steps=SimpleNamespace(remote=AsyncMock()),
        )
        self.failure = torch.OutOfMemoryError("dtype conversion failed")
        self.bad_tensor = _FailingCastTensor(self.failure)
        device = SimpleNamespace(synchronize=lambda: None, ipc_collect=lambda: None, empty_cache=lambda: None)
        for patcher in [
            patch.object(RayIpcWeightSender, "_init_buffer", return_value={"method": "cuda_ipc", "payload": b"handle"}),
            patch("verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer.get_torch_device", return_value=device),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_conversion_failure_drains_remaining_collectives_before_returning(self):
        source = _WeightSource([self.bad_tensor, self.bad_tensor, self.bad_tensor])
        with self.assertRaises(torch.OutOfMemoryError) as raised:
            await self.adapter.update_weights(source, global_steps=12)
        self.assertIs(raised.exception, self.failure)
        self.assertEqual(source.received, [0, 1, 2])
        self.assertTrue(source.completed.is_set())
        self.assertEqual(source.iterations, 1)
        self.assertEqual(self.bad_tensor.cast_count, 1)
        self.adapter.server_handle.set_global_steps.remote.assert_not_awaited()

    async def test_runtime_format_preserves_keys_dtype_and_vision_like_names(self):
        captured = []

        async def send(sender, weights):
            async for entry in weights:
                captured.append(entry)

        values = [
            ("vision.weight", torch.ones(2, dtype=torch.float32)),
            ('[0,"kernel"]', torch.ones(2, dtype=torch.int8)),
        ]
        with patch.object(RayIpcWeightSender, "async_send_weights", send):
            await self.adapter.update_weights(values, wire_format="runtime_tensors")
        self.assertEqual([name for name, _ in captured], ["vision.weight", '[0,"kernel"]'])
        self.assertEqual([tensor.dtype for _, tensor in captured], [torch.float32, torch.int8])

    async def test_runtime_adapter_drops_source_and_view_before_next_fetch(self):
        refs = []

        async def weights():
            for index in range(3):
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs), "adapter retains the previous GPU storage")
                value = torch.ones(2)
                refs.append(weakref.ref(value))
                yield str(index), value
                del value

        async def send(sender, prepared):
            while True:
                try:
                    item = await anext(prepared)
                except StopAsyncIteration:
                    break
                refs.append(weakref.ref(item[1]))
                del item

        with patch.object(RayIpcWeightSender, "async_send_weights", send):
            await self.adapter.update_weights(weights(), wire_format="runtime_tensors")

    async def test_runtime_preparation_failure_does_not_drain_snapshot_inputs(self):
        failure = RuntimeError("detach failed")

        class InvalidTensor:
            def detach(self):
                raise failure

        source = _WeightSource([InvalidTensor(), InvalidTensor()])

        async def send(sender, prepared):
            async for _ in prepared:
                pass

        with (
            patch.object(RayIpcWeightSender, "async_send_weights", send),
            self.assertRaisesRegex(RuntimeError, "detach"),
        ):
            await self.adapter.update_weights(source, wire_format="runtime_tensors")
        self.assertEqual(source.received, [0])

    async def test_runtime_adapter_retains_unacknowledged_sender_and_rejects_reuse(self):
        async def send(sender, weights):
            sender.quarantined = True
            sender.buffer = torch.ones(2)
            raise RuntimeError("abort acknowledgement missing")

        with (
            patch.object(RayIpcWeightSender, "async_send_weights", send),
            self.assertRaisesRegex(RuntimeError, "abort"),
        ):
            await self.adapter.update_weights([], wire_format="runtime_tensors")
        self.assertIsNotNone(self.adapter._failed_runtime_senders[0].buffer)
        with self.assertRaisesRegex(RuntimeError, "restart"):
            await self.adapter.update_weights([], wire_format="runtime_tensors")

    async def test_nonleader_drains_raw_weights_without_casting(self):
        self.adapter._has_server = False
        source = _WeightSource([self.bad_tensor, self.bad_tensor, self.bad_tensor])
        await self.adapter.update_weights(source, global_steps=12)
        self.assertEqual(source.received, [0, 1, 2])
        self.assertTrue(source.completed.is_set())
        self.assertEqual(self.bad_tensor.cast_count, 0)
        self.adapter.server_handle.begin_weight_update_from_ipc.remote.assert_not_awaited()

    async def test_network_iterator_failure_is_not_swallowed_or_restarted(self):
        failure = RuntimeError("collective receive failed")
        source = _WeightSource([], failure_at=0, failure=failure)
        with self.assertRaises(RuntimeError) as raised:
            await self.adapter.update_weights(source, global_steps=12)
        self.assertIs(raised.exception, failure)
        self.assertEqual(source.iterations, 1)
        self.assertEqual(source.next_calls, 1)
        self.adapter.server_handle.set_global_steps.remote.assert_not_awaited()

    async def test_secondary_network_failure_preserves_original_conversion_error(self):
        source = _WeightSource(
            [self.bad_tensor, self.bad_tensor], failure_at=2, failure=RuntimeError("collective receive failed")
        )
        with self.assertRaises(torch.OutOfMemoryError) as raised:
            await self.adapter.update_weights(source, global_steps=12)
        self.assertIs(raised.exception, self.failure)
        self.assertEqual(source.received, [0, 1])
        self.assertEqual(source.iterations, 1)
        self.assertEqual(source.next_calls, 3)
        self.assertEqual(self.bad_tensor.cast_count, 1)
