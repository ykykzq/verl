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
"""Bucketed rtp-llm weight transfer with Ray RPC control and IPC data."""

from __future__ import annotations

import asyncio
import gc
import logging
import pickle
import uuid
from multiprocessing import shared_memory
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device, is_support_ipc
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__file__)


def serialize_ipc_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize only a tensor's IPC descriptor, never its storage bytes."""
    return pickle.dumps(reduce_tensor(tensor), protocol=pickle.HIGHEST_PROTOCOL)


def rebuild_ipc_tensor(payload: bytes, device_id: int | None = None) -> torch.Tensor:
    """Rebuild a CUDA IPC tensor in the receiving process."""
    func, args = pickle.loads(payload)
    args = list(args)
    if device_id is not None:
        # The sender and server can have different CUDA_VISIBLE_DEVICES mappings.
        args[6] = device_id
    tensor = func(*args)
    if tensor.numel() > 0 and tensor.data_ptr() == 0:
        raise RuntimeError(
            "Rebuilt IPC tensor has a null data pointer. Disable expandable segments "
            "on the sender, align the two torch versions, or use shared memory."
        )
    return tensor


def _assert_ipc_exportable(buffer: torch.Tensor) -> None:
    addr = buffer.data_ptr()
    segment = next(
        (
            item
            for item in get_torch_device().memory_snapshot()
            if item["address"] <= addr < item["address"] + item["total_size"]
        ),
        None,
    )
    if segment is not None and segment.get("is_expandable"):
        raise RuntimeError(
            "Weight-transfer buffer was allocated in an expandable segment, whose IPC "
            "handle may not be readable by the rtp-llm torch build."
        )


def rebuild_shared_memory(name: str, size: int) -> tuple[torch.Tensor, shared_memory.SharedMemory]:
    shm = shared_memory.SharedMemory(name=name)
    return torch.frombuffer(shm.buf[:size], dtype=torch.uint8), shm


class RayIpcWeightSender:
    """Send tensor buckets while using Ray actor calls as the control channel."""

    def __init__(self, server_handle: Any, bucket_size_mb: int = 512, use_shm: bool = False):
        self.server_handle = server_handle
        self.bucket_size = int(bucket_size_mb) << 20
        self.bucket_size_mb = bucket_size_mb
        self.use_shm = use_shm
        self.buffer: torch.Tensor | None = None
        self.shm: shared_memory.SharedMemory | None = None

    async def async_send_weights(self, weights) -> None:
        round_id = uuid.uuid4().hex
        descriptor = self._init_buffer()
        began = False
        try:
            await self.server_handle.begin_weight_update_from_ipc.remote(
                round_id=round_id,
                descriptor=descriptor,
            )
            began = True

            sequence = 0
            offset = 0
            entries: list[dict[str, Any]] = []
            async for name, weight in ensure_async_iterator(weights):
                alignment = weight.element_size()
                offset = (offset + alignment - 1) // alignment * alignment
                if offset + weight.nbytes > self.bucket_size and entries:
                    await self._send_bucket(round_id, sequence, entries, is_last=False)
                    sequence += 1
                    entries = []
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    if self.use_shm:
                        raise ValueError(
                            f"Weight {name} ({weight.nbytes} bytes) exceeds the "
                            f"{self.bucket_size_mb} MiB shared-memory bucket"
                        )
                    await self._send_bucket(
                        round_id,
                        sequence,
                        [self._entry(name, weight, handle=serialize_ipc_tensor(weight))],
                        is_last=False,
                    )
                    sequence += 1
                    continue

                assert self.buffer is not None
                self.buffer[offset : offset + weight.nbytes].view(dtype=weight.dtype).view(weight.shape).copy_(
                    weight, non_blocking=True
                )
                entries.append(self._entry(name, weight, offset=offset))
                offset += weight.nbytes

            await self._send_bucket(round_id, sequence, entries, is_last=True)
            await self.server_handle.finish_weight_update_from_ipc.remote(round_id=round_id)
        except BaseException:
            if began:
                try:
                    await asyncio.shield(self.server_handle.abort_weight_update_from_ipc.remote(round_id=round_id))
                except BaseException:
                    logger.warning("Failed to abort rtp-llm weight-update round %s", round_id, exc_info=True)
            raise
        finally:
            self._cleanup()

    async def _send_bucket(
        self,
        round_id: str,
        sequence: int,
        entries: list[dict[str, Any]],
        *,
        is_last: bool,
    ) -> None:
        get_torch_device().synchronize()
        await self.server_handle.apply_weight_bucket_from_ipc.remote(
            round_id=round_id,
            sequence=sequence,
            entries=entries,
            is_last=is_last,
        )

    def _init_buffer(self) -> dict[str, Any]:
        if self.use_shm:
            # macOS caps POSIX shared-memory names at 31 characters.
            name = f"vrw_{uuid.uuid4().hex[:20]}"
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.bucket_size)
            self.buffer = torch.frombuffer(self.shm.buf, dtype=torch.uint8)
            return {"method": "shm", "name": name, "size": self.bucket_size}

        self.buffer = torch.empty(
            self.bucket_size,
            dtype=torch.uint8,
            device=f"{get_device_name()}:{get_device_id()}",
        )
        _assert_ipc_exportable(self.buffer)
        return {"method": "cuda_ipc", "payload": serialize_ipc_tensor(self.buffer)}

    @staticmethod
    def _entry(
        name: str,
        weight: torch.Tensor,
        *,
        offset: int = 0,
        handle: bytes | None = None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "shape": tuple(weight.shape),
            "dtype": str(weight.dtype).removeprefix("torch."),
            "offset": offset,
            "handle": handle,
        }

    def _cleanup(self) -> None:
        self.buffer = None
        gc.collect()
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
        if is_support_ipc():
            get_torch_device().ipc_collect()
        get_torch_device().empty_cache()
