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
"""Client-side adapter for the rtp-llm rollout server.

Runs inside each ``CheckpointEngineWorker`` and pushes freshly trained weights into
the co-located ``RTPLLMHttpServer`` actor over device IPC handles.
"""

import logging
import os
import re
import time
from typing import AsyncGenerator, Generator, Optional

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.device import get_torch_device, is_support_ipc, set_expandable_segments
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import ensure_async_iterator
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_VISION_PREFIX_RE = re.compile(r"^(model\.)?(visual|vision_tower|vision_model)\.")

_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class ServerAdapter(BaseRollout):
    """Weight-sync client for the rtp-llm rollout server.

    rtp-llm re-shards every incoming tensor internally, so only one rank per replica
    transmits. Every rank must still drain ``weights`` because the checkpoint engine
    backs it with collectives.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)
        self.server_handle: Optional[ray.actor.ActorHandle] = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.replica_rank = rank // rollout_world_size if replica_rank == -1 else replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size
        self._has_server = self.rollout_rank == 0
        self.is_leader_rank = self._has_server
        # rtp-llm keeps weights resident; only the kv cache is reclaimable.
        self.sleep_level = 1
        # The engine's weights live at its activation dtype and are updated by in-place
        # copy, so a mismatched dtype from the trainer is rejected outright.
        self._engine_dtype = _TORCH_DTYPES.get(str(self.config.dtype), torch.bfloat16)

        # Expandable-segment memory is exported over the virtual-memory IPC path, and its
        # handle can only be rebuilt by a torch build carrying that same path. The engine runs
        # an older torch that lacks it, so it decodes the payload as a legacy handle, silently
        # gets a null pointer, and aborts once it writes through it. This process only forwards
        # weights, so giving up expandable segments here costs nothing. Done in __init__
        # because the setting only affects allocations made after it.
        set_expandable_segments(False)

        local_rank = self.rollout_rank % local_world_size
        job_id = ray.get_runtime_context().get_job_id()
        self.zmq_handle = f"ipc:///tmp/rtp-llm-weights-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"
        self.use_shm = not is_support_ipc()
        if self.use_shm:
            logger.warning(
                "Device IPC unsupported; falling back to shared memory for weight transfer, "
                "which adds a host round trip per bucket."
            )

    def _ensure_server_handle(self) -> bool:
        if not self._has_server:
            return False
        if self.server_handle is None:
            prefix = self.config.get("name", "rtp_llm")
            self.server_handle = ray.get_actor(f"{prefix}_server_{self.replica_rank}_{self.node_rank}")
        return True

    async def resume(self, tags: list[str]):
        if self.config.free_cache_engine and self._ensure_server_handle():
            await self.server_handle.wake_up.remote(tags=tags)

    async def release(self):
        if self.config.free_cache_engine and self._ensure_server_handle():
            await self.server_handle.sleep.remote()

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        wire_format: str = "named_tensors",
        **kwargs,
    ):
        assert wire_format == "named_tensors", (
            f"rtp-llm rollout only consumes full named tensors; got wire_format={wire_format!r}"
        )
        start_time = time.time()
        num_sent = 0
        num_skipped = 0

        async def prepared() -> AsyncGenerator[tuple[str, torch.Tensor], None]:
            nonlocal num_sent, num_skipped
            async for name, param in ensure_async_iterator(weights):
                # The engine runs text-only (vit_separation=REMOTE), so it owns no vision
                # tower receptors while the trainer's Qwen3.5 checkpoint still carries them.
                if _VISION_PREFIX_RE.match(name):
                    num_skipped += 1
                    continue
                payload = param.detach()
                if payload.is_floating_point() and payload.dtype != self._engine_dtype:
                    payload = payload.to(self._engine_dtype)
                num_sent += 1
                yield name, payload

        if not self._ensure_server_handle():
            # The checkpoint engine backs `weights` with collectives, so every rank must
            # drain it even when it has no server to feed.
            async for _ in prepared():
                pass
            return

        # Start the receiver first: the sender binds the socket and blocks on the ack.
        future = self.server_handle.update_weights_from_ipc.remote(
            zmq_handle=self.zmq_handle,
            use_shm=self.use_shm,
        )
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=self.config.checkpoint_engine.update_weights_bucket_megabytes,
            use_shm=self.use_shm,
        )
        await sender.async_send_weights(prepared())
        await future

        await self.server_handle.clear_kv_cache.remote()
        if global_steps is not None:
            await self.server_handle.set_global_steps.remote(global_steps)

        get_torch_device().empty_cache()

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(
                f"rtp-llm update_weights done: sent {num_sent} tensors "
                f"(skipped {num_skipped} vision) in {time.time() - start_time:.2f}s"
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        raise NotImplementedError(
            "rtp-llm rollout only supports async server mode; use RTPLLMReplica with LLMServerClient."
        )
