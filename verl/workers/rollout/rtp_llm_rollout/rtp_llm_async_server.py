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
"""rtp-llm rollout server for verl's async (server-mode) rollout.

One ``RTPLLMHttpServer`` Ray actor owns one rtp-llm engine. Generation is
token-in / token-out over the engine's loopback gRPC channel; weights are pushed
in-process through ``WeightManager.update_from_tensor``.
"""

import asyncio
import glob
import itertools
import logging
import math
import os
import sys
from typing import Any, Optional

import ray
import torch
from fastapi import FastAPI
from omegaconf import DictConfig
from ray.actor import ActorHandle

from verl.plugin.platform import get_platform
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_resource_name,
    get_torch_device,
    get_visible_devices_keyword,
)
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings, run_uvicorn

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# rtp-llm derives its listener ports as start_port + rank_id * worker_info_port_num
# and consumes 9 of them; give each replica a padded block so replicas never overlap.
_PORTS_PER_WORKER = 9
_PORT_STRIDE = 16

# Architectures that rtp-llm registers under a name that differs from the HF string.
_ARCH_TO_MODEL_TYPE = {
    "Qwen3_5ForConditionalGeneration": "qwen35_dense",
    "Qwen3_5MoeForConditionalGeneration": "qwen35_moe",
    "Qwen3NextForCausalLM": "qwen3_next",
    "Qwen3ForCausalLM": "qwen_3",
    "Qwen3MoeForCausalLM": "qwen_3_moe",
    "Qwen2ForCausalLM": "qwen_2",
}


def _resolve_model_type(hf_config, local_path: str) -> str:
    architectures = getattr(hf_config, "architectures", None) or []
    for arch in architectures:
        if arch in _ARCH_TO_MODEL_TYPE:
            return _ARCH_TO_MODEL_TYPE[arch]

    import json

    from rtp_llm.model_factory_register import ModelDict

    with open(os.path.join(local_path, "config.json")) as f:
        config_json = json.load(f)
    model_type = ModelDict.get_ft_model_type_by_config(config_json)
    if model_type is None:
        raise ValueError(
            f"rtp-llm has no model registered for architectures={architectures!r}. "
            f"Add a mapping in _ARCH_TO_MODEL_TYPE or register the model in rtp_llm."
        )
    return model_type


def _checkpoint_size_mib(local_path: str) -> int:
    patterns = ("*.safetensors", "*.bin", "*.pt")
    total = 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(local_path, pattern)):
            total += os.path.getsize(path)
        if total:
            break
    return total // (1024 * 1024)


class RTPLLMHttpServer:
    """Owns a single-GPU rtp-llm engine and serves verl's token-in/token-out contract."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices
        os.environ["VERL_REPLICA_RANK"] = str(replica_rank)

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)

        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        elif self.config.max_model_len > max_position_embeddings:
            raise ValueError(
                f"max_model_len ({self.config.max_model_len}) exceeds max_position_embeddings "
                f"({max_position_embeddings})"
            )

        if self.config.tensor_model_parallel_size != 1:
            raise NotImplementedError(
                "rtp-llm rollout currently runs one engine per GPU; set "
                "actor_rollout_ref.rollout.tensor_model_parallel_size=1."
            )

        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        # Weight version, stamped onto every response so the trainer can measure staleness.
        self.global_steps = None

        # Cleared during weight sync so new requests park instead of racing the update.
        self._generation_allowed = asyncio.Event()
        self._generation_allowed.set()
        self._abort_requested = False
        self._inflight: dict[str, asyncio.Future] = {}
        self._request_counter = itertools.count(1)

        self.engine = None
        self.visitor = None
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        logger.info(
            f"RTPLLMHttpServer replica_rank={replica_rank} node_rank={node_rank} devices={cuda_visible_devices}"
        )

    # ------------------------------------------------------------------ launch

    def get_server_address(self):
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    def _start_port(self) -> int:
        base = int(os.environ.get("VERL_RTP_LLM_START_PORT", "31000"))
        index = self.replica_rank * self.nnodes + self.node_rank
        return base + index * _PORT_STRIDE

    def _kv_cache_mem_mib(self) -> int:
        """Translate verl's gpu_memory_utilization fraction into rtp-llm's absolute cap.

        rtp-llm has no utilization fraction: in auto mode it claims *all* free HBM
        minus a runtime reserve, which would collide with the trainer sharing the box.
        """
        total_mib = get_torch_device().get_device_properties(0).total_memory // (1024 * 1024)
        weights_mib = _checkpoint_size_mib(self.model_config.local_path)
        reserve_mib = max(2048, int(0.05 * total_mib))
        budget_mib = int(self.config.gpu_memory_utilization * total_mib)
        kv_mib = budget_mib - weights_mib - reserve_mib
        if kv_mib < 1024:
            raise ValueError(
                f"gpu_memory_utilization={self.config.gpu_memory_utilization} leaves only {kv_mib} MiB for the "
                f"kv cache (total={total_mib} MiB, weights={weights_mib} MiB, runtime reserve={reserve_mib} MiB). "
                f"Raise gpu_memory_utilization."
            )
        logger.info(
            f"rtp-llm memory plan: total={total_mib} MiB weights={weights_mib} MiB "
            f"reserve={reserve_mib} MiB kv_cache={kv_mib} MiB"
        )
        return kv_mib

    def _seq_size_per_block(self) -> int:
        """Pick the KV page size.

        Hybrid models (Qwen3.5) mix linear- and full-attention layers, and rtp-llm requires the
        full-attention block to be at least as large as the linear-attention state (~1.1 MiB here).
        aiter only ships paged-prefill kernels for page sizes 1, 16 and 1024, so 1024 is the only
        viable choice; plain attention models keep ROCm's default of 16.
        """
        hf_config = self.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        layer_types = getattr(text_config, "layer_types", None) or []
        return 1024 if any("linear" in str(t) for t in layer_types) else 16

    def _engine_max_seq_len(self) -> int:
        """Longest sequence this rollout can produce.

        max_model_len falls back to the model's max_position_embeddings (262144 here),
        but verl never asks for more than prompt+response, and planning for the larger
        figure both oversizes the engine and collapses the derived concurrency.
        """
        return min(self.config.max_model_len, self.config.prompt_length + self.config.response_length)

    def _max_concurrency(self) -> int:
        """Cap in-flight requests at what the kv cache can hold at full length.

        rtp-llm admits up to ``concurrency_limit`` streams and never preempts, so
        admitting more than the block pool can grow to deadlocks every in-flight
        stream until its timeout fires. One block spans ``group_layer_num`` (the
        full-attention layers; linear-attention blocks are padded to the same
        stride) times the per-layer K+V stride.
        """
        configured = max(1, self.config.max_num_seqs)
        hf_config = self.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        try:
            kv_heads = int(text_config.num_key_value_heads)
            head_dim = int(text_config.head_dim)
            layer_types = list(getattr(text_config, "layer_types", None) or [])
            full_layers = sum(1 for t in layer_types if "full" in str(t)) or int(text_config.num_hidden_layers)
        except (AttributeError, TypeError, ValueError):
            logger.warning("rtp-llm: cannot derive kv cache capacity, using max_num_seqs=%d", configured)
            return configured

        tokens_per_block = self._seq_size_per_block()
        dtype_size = 2  # bf16/fp16 kv cache
        block_bytes = full_layers * 2 * kv_heads * head_dim * dtype_size * tokens_per_block
        total_blocks = (self._kv_cache_mem_mib() * 1024 * 1024) // block_bytes
        # The allocator holds back ~5% of blocks; keep a wider margin so partial-rollout
        # resubmissions after a weight sync still find room.
        usable_blocks = int(total_blocks * 0.9)
        blocks_per_request = math.ceil(self._engine_max_seq_len() / tokens_per_block)
        capacity = max(1, usable_blocks // blocks_per_request)

        if capacity < configured:
            logger.warning(
                "rtp-llm: limiting concurrency to %d (kv cache holds %d blocks, %d per request); "
                "max_num_seqs=%d would exhaust the block pool",
                capacity,
                total_blocks,
                blocks_per_request,
                configured,
            )
        return min(configured, capacity)

    def _build_config(self):
        from rtp_llm.server.server_args.server_args import setup_args

        local_path = self.model_config.local_path
        model_type = _resolve_model_type(self.model_config.hf_config, local_path)
        start_port = self._start_port()
        max_concurrency = self._max_concurrency()
        max_context_batch_size = max(1, min(8, max_concurrency))

        argv = [
            "--checkpoint_path",
            local_path,
            "--tokenizer_path",
            local_path,
            "--model_type",
            model_type,
            "--max_seq_len",
            str(self._engine_max_seq_len()),
            "--tp_size",
            "1",
            "--dp_size",
            "1",
            "--world_size",
            "1",
            "--world_rank",
            "0",
            "--local_world_size",
            "1",
            "--start_port",
            str(start_port),
            "--worker_info_port_num",
            str(_PORTS_PER_WORKER),
            "--concurrency_limit",
            str(max_concurrency),
            "--max_context_batch_size",
            str(max_context_batch_size),
            "--max_batch_tokens_size",
            str(self.config.max_num_batched_tokens),
            "--kv_cache_mem_mb",
            str(self._kv_cache_mem_mib()),
            "--reuse_cache",
            "True" if self.config.enable_prefix_caching else "False",
            "--frontend_server_count",
            "0",
            "--seq_size_per_block",
            str(self._seq_size_per_block()),
            # The hand-written asm paged-attention kernel has no variant for the
            # head_dim=256 / page=1024 V layout that hybrid models need.
            "--use_triton_pa",
            "1",
        ]
        if self.config.dtype in ("bfloat16", "bf16"):
            argv += ["--act_type", "bf16"]
        elif self.config.dtype in ("float16", "fp16", "half"):
            argv += ["--act_type", "fp16"]

        cfg = setup_args(argv)
        cfg.server_config.ip = "127.0.0.1"
        # Qwen3.5 shares its config with the VL variant, so rtp-llm flags it multimodal and
        # would spin up a local ViT process on the rollout GPU. Keep the ViT remote (i.e. absent)
        # for text-only RL rollout.
        from rtp_llm.ops import VitSeparation

        cfg.vit_config.vit_separation = VitSeparation.VIT_SEPARATION_REMOTE
        return cfg

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        os.environ.setdefault("LOG_PATH", f"/tmp/rtp_llm_logs/replica_{self.replica_rank}_{self.node_rank}")
        os.makedirs(os.environ["LOG_PATH"], exist_ok=True)

        # Booting loads the checkpoint and warms up kernels; keep the actor's event loop free.
        await asyncio.to_thread(self._boot_engine)

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok", "global_steps": self.global_steps}

        self._server_port, self._server_task = await run_uvicorn(app, None, self._server_address)
        logger.info(f"rtp-llm replica {self.replica_rank} ready, control port {self._server_port}")

    def _boot_engine(self):
        # rtp-llm re-parses sys.argv in several places (setup_args, setup_and_configure_server).
        # Inside a Ray worker sys.argv carries Ray's own flags, which argparse rejects.
        saved_argv = sys.argv
        sys.argv = saved_argv[:1]
        try:
            self._boot_engine_impl()
        finally:
            sys.argv = saved_argv

    def _boot_engine_impl(self):
        from rtp_llm.config.server_config_setup import (
            set_parallelism_config,
            setup_and_configure_server,
            setup_cuda_device_and_accl_env,
        )
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.server.backend_manager import BackendManager
        from rtp_llm.server.backend_rpc_server_visitor import create_backend_rpc_server_visitor
        from rtp_llm.utils.concurrency_controller import ConcurrencyController, set_global_controller

        cfg = self._build_config()
        setup_and_configure_server(cfg)
        set_parallelism_config(cfg.parallelism_config, 0, cfg.ffn_disaggregate_config, cfg.prefill_cp_config)
        local_rank = cfg.parallelism_config.local_rank
        cfg.server_config.set_local_rank(local_rank)
        cfg.distribute_config.set_local_rank(local_rank)
        setup_cuda_device_and_accl_env(local_rank)
        set_global_controller(ConcurrencyController(cfg.concurrency_config.concurrency_limit))

        self.backend_manager = BackendManager(cfg)
        self.backend_manager.start()
        self.engine = self.backend_manager.engine
        self.weight_manager = self.engine.model.weight_manager

        rtp_model_config = ModelFactory.create_model_config(
            model_args=cfg.model_args,
            lora_config=cfg.lora_config,
            kv_cache_config=cfg.kv_cache_config,
            profiling_debug_logging_config=cfg.profiling_debug_logging_config,
            generate_env_config=cfg.generate_env_config,
            embedding_config=cfg.embedding_config,
            quantization_config=cfg.quantization_config,
            render_config=cfg.render_config,
            eplb_config=cfg.eplb_config,
            vit_config=cfg.vit_config,
        )
        self.visitor = create_backend_rpc_server_visitor(cfg, rtp_model_config, source_role="frontend")
        self.eos_token_id = self.engine.model.model_config.special_tokens.eos_token_id
        self._cfg = cfg

    # -------------------------------------------------------------- generation

    def _build_generate_config(self, sampling_params: dict[str, Any], max_tokens: int):
        from rtp_llm.config.generate_config import GenerateConfig

        top_k = sampling_params.get("top_k", 0)
        # verl encodes "disabled" as -1; rtp-llm uses 0.
        top_k = 0 if top_k is None or top_k < 0 else top_k
        want_logprobs = bool(sampling_params.get("logprobs", False))

        return GenerateConfig(
            max_new_tokens=max_tokens,
            min_new_tokens=0,
            do_sample=sampling_params.get("temperature", 1.0) > 0.0,
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            top_k=top_k,
            repetition_penalty=sampling_params.get("repetition_penalty", 1.0),
            num_beams=1,
            num_return_sequences=0,
            # Streaming is mandatory: verl's partial rollout needs the tokens produced
            # so far when a request is aborted, and softmax_probs only arrive per step.
            is_streaming=True,
            return_output_ids=True,
            return_softmax_probs=want_logprobs,
            aux_info=True,
            return_logits=False,
            return_hidden_states=False,
            ignore_eos=False,
            # Weights change every sync, so KV computed under older weights must not be reused.
            reuse_cache=False,
            # An unset timeout falls back to rtp-llm's 2h max_rpc_timeout_ms, so a stall would
            # only surface after two hours; bound it so failures are visible quickly.
            timeout_ms=int(float(os.environ.get("VERL_RTP_LLM_GENERATE_TIMEOUT_S", "3600")) * 1000),
        )

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        if image_data or video_data or audio_data:
            raise NotImplementedError("rtp-llm rollout does not support multimodal inputs yet.")

        from rtp_llm.utils.base_model_datatypes import GenerateInput

        await self._generation_allowed.wait()

        requested = sampling_params.get("max_tokens") or sampling_params.get("max_new_tokens")
        max_tokens = requested if requested else self.config.response_length
        max_tokens = max(1, min(max_tokens, self._engine_max_seq_len() - len(prompt_ids)))

        generate_config = self._build_generate_config(sampling_params, max_tokens)
        generate_input = GenerateInput(
            request_id=next(self._request_counter),
            token_ids=torch.tensor(prompt_ids, dtype=torch.int32),
            mm_inputs=[],
            generate_config=generate_config,
        )

        token_ids: list[int] = []
        probs: list[float] = []

        async def consume():
            stream = await self.visitor.enqueue(generate_input)
            try:
                async for outputs in stream:
                    output = outputs.generate_outputs[0]
                    # rtp-llm streams deltas, not cumulative ids.
                    if output.output_ids is not None:
                        token_ids.extend(output.output_ids.reshape(-1).tolist())
                    if output.aux_info is not None and output.aux_info.softmax_probs:
                        probs.extend(output.aux_info.softmax_probs)
            finally:
                await stream.aclose()

        # Consume in a task so abort_all_requests can cancel it: a request still queued
        # for kv cache blocks never yields, so a cooperative flag would not reach it.
        task = asyncio.ensure_future(consume())
        self._inflight[request_id] = task
        aborted = False
        try:
            await task
        except asyncio.CancelledError:
            aborted = True
        except Exception as e:
            # A stalled or timed-out request must not take the rollouter down; hand back
            # whatever was generated and let verl resume it as a partial rollout.
            aborted = True
            logger.warning(f"rtp-llm generate {request_id} failed after {len(token_ids)} tokens: {e}")
        finally:
            self._inflight.pop(request_id, None)

        # A completed request must expose its generated response. An aborted request may
        # legitimately have no token yet and will be resumed by the fully-async client.
        assert aborted or len(token_ids) > 0, f"rtp-llm returned no output token ids for {request_id}"

        # Trailing EOS is part of the response for verl's length accounting, matching vLLM.
        log_probs = None
        if generate_config.return_softmax_probs:
            log_probs = [math.log(p) if p > 0.0 else -float("inf") for p in probs[: len(token_ids)]]
            assert log_probs is not None and len(log_probs) == len(token_ids), (
                f"rtp-llm log_probs/token_ids length mismatch for {request_id}: "
                f"{len(log_probs)} vs {len(token_ids)}"
            )

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason="aborted" if aborted else None,
            extra_fields={"global_steps": self.global_steps},
        )

    # ----------------------------------------------------------------- control

    async def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps

    async def abort_all_requests(self):
        """Stop in-flight generation and hold off new requests until resume_generation().

        Cancelling each consumer task propagates a gRPC cancel, which frees the stream's
        kv cache blocks even when it was still queued and never produced a token.
        """
        self._generation_allowed.clear()
        self._abort_requested = True
        tasks = list(self._inflight.values())
        for task in tasks:
            task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=120)
            if pending:
                logger.warning(f"abort_all_requests: {len(pending)} requests did not wind down in 120s")

    async def resume_generation(self):
        self._abort_requested = False
        self._generation_allowed.set()

    async def clear_kv_cache(self):
        # reuse_cache is off for RL, so there is no cross-request prefix cache to invalidate.
        pass

    async def release_kv_cache(self):
        pass

    async def resume_kv_cache(self):
        pass

    async def wake_up(self, tags: Optional[list[str]] = None):
        if self.rollout_mode != RolloutMode.STANDALONE:
            raise ValueError(f"rtp-llm rollout only supports standalone mode, got {self.rollout_mode}")

    async def sleep(self):
        if self.rollout_mode != RolloutMode.STANDALONE:
            raise ValueError(f"rtp-llm rollout only supports standalone mode, got {self.rollout_mode}")

    async def update_weights_from_ipc(self, zmq_handle: str, use_shm: bool = False):
        """Pull a weight-sync round from the co-located CheckpointEngineWorker.

        Each bucket arrives as views into a shared buffer that the sender reuses for the
        next bucket, so the weight manager must not retain them past the callback.
        """
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        device = torch.device(get_device_name(), get_device_id())
        receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=device, use_shm=use_shm)

        def on_bucket_received(weights: list[tuple[str, torch.Tensor]], is_last: bool):
            self.weight_manager.update_from_hf_tensors(weights, is_last=is_last)

        await asyncio.to_thread(receiver.receive_weights, on_bucket_received)

    async def start_profile(self, **kwargs):
        pass

    async def stop_profile(self):
        pass


class RTPLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ) -> None:
        if is_reward_model or is_teacher_model:
            raise NotImplementedError("RTPLLMReplica does not support reward/teacher models yet.")
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = ray.remote(RTPLLMHttpServer)

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # Co-locate each engine with its CheckpointEngineWorker so weight-sync IPC handles
        # stay on the GPU that created them.
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_node_ids = [info[0] for info in worker_infos]
        worker_visible_devices = [info[1] for info in worker_infos]

        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            lo = node_rank * gpus_per_replica_node
            hi = (node_rank + 1) * gpus_per_replica_node
            node_visible_devices = ",".join(worker_visible_devices[lo:hi])
            node_id = worker_node_ids[lo]
            name = f"rtp_llm_server_{self.replica_rank}_{node_rank}{self.name_suffix}"

            env_vars = {
                **{var: "1" for var in get_platform().ray_noset_envvars()},
                **get_platform().rollout_env_vars(),
            }
            for var in ("VERL_RTP_LLM_START_PORT", "RTP_LLM_LOG_LEVEL", "LOG_LEVEL", "PYTHONPATH"):
                if value := os.environ.get(var):
                    env_vars[var] = value

            runtime_env = {"env_vars": env_vars}
            # rtp-llm's Qwen3.5 kernels require triton >= 3.6 while the FSDP trainer's
            # flash-linear-attention requires triton 3.5.x, so the engine gets its own
            # interpreter rather than sharing the trainer's.
            if conda_env := os.environ.get("VERL_RTP_LLM_CONDA_ENV"):
                runtime_env["conda"] = conda_env

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env=runtime_env,
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=self.workers[lo:hi],
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_visible_devices,
            )
            self.servers.append(server)

        await asyncio.gather(*[server.launch_server.remote() for server in self.servers])

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
