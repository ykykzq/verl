"""CPU control-plane tests; GPU payload movement is covered on cluster."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer import RayIpcWeightSender
from verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server import RTPLLMHttpServer

MODULE = "verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server"


class Converter:
    required_names = frozenset({"source"})
    ignored_source_names = frozenset({"nonowned_alias"})
    expected_runtime_layout = [{"name": '[null,"out"]', "shape": [2], "dtype": "torch.float32"}]

    def __init__(self):
        self.closed = False
        self.finished = False

    def add(self, values, is_last=False):
        self.finished = is_last
        if not values:
            return []
        return [('[null,"out"]', SimpleNamespace(shape=(2,), dtype=torch.float32, device=torch.device("cuda:0")))]

    def close(self):
        self.closed = True


def server():
    instance = RTPLLMHttpServer.__new__(RTPLLMHttpServer)
    instance._gpu_snapshot_lock = asyncio.Lock()
    instance._gpu_snapshot = None
    instance._generation_allowed = asyncio.Event()
    instance._generation_allowed.set()
    instance._weight_sync_error = None
    instance.config = SimpleNamespace(tensor_model_parallel_size=1)
    instance.weight_manager = SimpleNamespace(
        get_runtime_weight_layout=lambda: Converter.expected_runtime_layout,
        create_runtime_snapshot_converter=lambda limit: Converter(),
    )
    return instance


class SnapshotServerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        allocator = patch(f"{MODULE}.set_expandable_segments", create=True)
        self.allocator = allocator.start()
        self.addCleanup(allocator.stop)

    async def test_snapshot_disables_expandable_allocator_before_factory(self):
        instance = server()

        def make_converter(limit):
            self.allocator.assert_called_once_with(False)
            return Converter()

        instance.weight_manager.create_runtime_snapshot_converter = make_converter
        await instance.begin_gpu_snapshot(7, 4096)
        await instance.close_gpu_snapshot()

    async def test_describe_and_conversion_do_not_pause_or_touch_live_weights(self):
        instance = server()
        metadata = await instance.describe_gpu_snapshot()
        self.assertEqual(
            metadata,
            {
                "layout": Converter.expected_runtime_layout,
                "required_names": ["source"],
                "ignored_source_names": ["nonowned_alias"],
            },
        )
        await instance.begin_gpu_snapshot(7, 4096)
        with (
            patch(f"{MODULE}.rebuild_ipc_tensor", return_value=SimpleNamespace(device=torch.device("cuda:0"))),
            patch(f"{MODULE}.serialize_ipc_tensor", return_value=b"gpu-handle"),
            patch(f"{MODULE}.get_device_id", return_value=0),
        ):
            output = await instance.convert_gpu_snapshot("source", b"source-handle", is_last=True)
        self.assertEqual(
            output, [{"name": '[null,"out"]', "shape": [2], "dtype": "torch.float32", "payload": b"gpu-handle"}]
        )
        self.assertTrue(instance._generation_allowed.is_set())
        self.assertIsNone(instance._weight_sync_error)
        with self.assertRaisesRegex(RuntimeError, "output|release"):
            await instance.close_gpu_snapshot()
        await instance.release_gpu_snapshot_outputs()
        await instance.close_gpu_snapshot()
        self.assertIsNone(instance._gpu_snapshot)

    async def test_new_conversion_cannot_release_previous_dma_outputs(self):
        instance = server()
        await instance.begin_gpu_snapshot(7, 4096)
        with (
            patch(f"{MODULE}.rebuild_ipc_tensor", return_value=SimpleNamespace(device=torch.device("cuda:0"))),
            patch(f"{MODULE}.serialize_ipc_tensor", return_value=b"gpu-handle"),
            patch(f"{MODULE}.get_device_id", return_value=0),
        ):
            await instance.convert_gpu_snapshot("source", b"source-handle")
            with self.assertRaisesRegex(RuntimeError, "output|release"):
                await instance.convert_gpu_snapshot(None, None, is_last=True)
        await instance.release_gpu_snapshot_outputs()
        await instance.convert_gpu_snapshot(None, None, is_last=True)
        await instance.close_gpu_snapshot()

    async def test_rejects_cpu_payload_and_overlapping_round(self):
        instance = server()
        await instance.begin_gpu_snapshot(7, 4096)
        with self.assertRaisesRegex(RuntimeError, "active"):
            await instance.begin_gpu_snapshot(8, 4096)
        with (
            patch(f"{MODULE}.rebuild_ipc_tensor", return_value=torch.ones(2)),
            patch(f"{MODULE}.get_device_id", return_value=0),
            self.assertRaisesRegex(ValueError, "CUDA|cuda"),
        ):
            await instance.convert_gpu_snapshot("source", b"invalid")
        await instance.close_gpu_snapshot()

    async def test_cancel_waits_for_native_conversion_and_keeps_outputs_owned(self):
        instance = server()
        entered = threading.Event()
        release = threading.Event()
        converter = Converter()
        original = converter.add

        def convert(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release conversion")
            return original(*args, **kwargs)

        converter.add = convert
        instance.weight_manager.create_runtime_snapshot_converter = lambda limit: converter
        await instance.begin_gpu_snapshot(7, 4096)
        with (
            patch(f"{MODULE}.rebuild_ipc_tensor", return_value=SimpleNamespace(device=torch.device("cuda:0"))),
            patch(f"{MODULE}.serialize_ipc_tensor", return_value=b"gpu-handle"),
            patch(f"{MODULE}.get_device_id", return_value=0),
        ):
            task = asyncio.create_task(instance.convert_gpu_snapshot("source", b"source-handle"))
            await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertTrue(instance._gpu_snapshot_lock.locked())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        with self.assertRaisesRegex(RuntimeError, "output|release"):
            await instance.close_gpu_snapshot()
        await instance.release_gpu_snapshot_outputs()
        await instance.close_gpu_snapshot()
        self.assertTrue(converter.closed)

    async def test_runtime_install_rejects_shared_memory_before_mapping(self):
        instance = server()
        instance._weight_update_lock = asyncio.Lock()
        instance._request_admission_lock = asyncio.Lock()
        instance._generation_allowed.clear()
        instance._abort_requested = True
        instance._inflight = {}
        instance._weight_update = None
        with (
            patch(f"{MODULE}.rebuild_shared_memory") as rebuild,
            patch(f"{MODULE}.get_device_id", return_value=0),
            self.assertRaisesRegex(ValueError, "runtime|CUDA"),
        ):
            await instance.begin_weight_update_from_ipc("round", {"method": "shm", "wire_format": "runtime_tensors"})
        rebuild.assert_not_called()


class SnapshotMemoryTest(unittest.TestCase):
    def test_each_rank_reserves_storage_conversion_input_and_ipc_bucket(self):
        instance = server()
        instance.config = SimpleNamespace(
            tensor_model_parallel_size=4,
            gpu_memory_utilization=0.9,
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=512),
            weight_publication=SimpleNamespace(
                transport="nccl_gpu",
                gpu_snapshot=SimpleNamespace(
                    max_bytes_per_gpu=8 << 30, max_conversion_bytes=8 << 30, max_tensor_bytes=4 << 30
                ),
            ),
        )
        instance.model_config = SimpleNamespace(local_path="/unused")
        device = SimpleNamespace(get_device_properties=lambda rank: SimpleNamespace(total_memory=48 << 30))
        with (
            patch(f"{MODULE}.get_torch_device", return_value=device),
            patch(f"{MODULE}._checkpoint_size_mib", return_value=40 * 1024),
        ):
            actual = instance._kv_cache_mem_mib()
        self.assertEqual(actual, int(0.9 * 49152) - 10240 - int(0.05 * 49152) - 20 * 1024 - 512)

    def test_runtime_sender_rejects_shared_memory(self):
        with self.assertRaisesRegex(ValueError, "runtime|CUDA"):
            RayIpcWeightSender(None, use_shm=True, wire_format="runtime_tensors")


if __name__ == "__main__":
    unittest.main()
