# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING, DictConfig, OmegaConf

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config.disaggregation import DisaggregationConfig
from verl.workers.config.model import MtpConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "PrometheusConfig",
    "RolloutConfig",
    "CheckpointEngineConfig",
    "FineGrainedWeightUpdateConfig",
    "JointSchedulingConfig",
    "WeightRelayConfig",
    "GPUSnapshotConfig",
    "WeightPublicationConfig",
    "TrajectoryMigrationConfig",
]


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    function_tool_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    default_agent_loop: str = "single_turn_agent"
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)
    # Fully qualified class name for custom AgentLoopManager (e.g., "mypackage.module.MyManager").
    # Security: This class will be dynamically imported via importlib. Only use trusted class paths.
    agent_loop_manager_class: Optional[str] = None


@dataclass
class TraceConfig(BaseConfig):
    project_name: Optional[str] = None
    experiment_name: Optional[str] = None
    backend: Optional[str] = None
    token2text: bool = False
    max_samples_per_step_per_worker: Optional[int] = None

    def __post_init__(self):
        if self.max_samples_per_step_per_worker is not None and self.max_samples_per_step_per_worker < 0:
            raise ValueError("`max_samples_per_step_per_worker` must be a non-negative integer or null.")


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class PrometheusConfig(BaseConfig):
    """
    Configuration for Prometheus server
    """

    # whether enable prometheus on server mode rollout
    enable: bool = False
    # Port number that Prometheus listens on, default is 9090
    port: int = 9090
    # Path to Prometheus configuration file
    file: str = "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"
    # Specify served_model_name to avoid displaying overly long model paths in Grafana
    served_model_name: Optional[str] = None


@dataclass
class CheckpointEngineConfig(BaseConfig):
    """
    Configuration for checkpoint engine to update weights from trainer to rollout
    """

    _mutable_fields = {"backend"}

    # Backend for checkpoint engine: naive, nccl, nixl, hccl
    backend: Optional[str] = "naive"
    # Bucket size in MB to transfer multiple weights at one time
    update_weights_bucket_megabytes: int = 2048
    # Additional keyword arguments for checkpoint engine
    engine_kwargs: dict = field(default_factory=dict)
    # If set, this Python module is imported on every worker process before the
    # backend is instantiated, allowing custom backends to register themselves
    # in CheckpointEngineRegistry.
    custom_backend_module: Optional[str] = None


@dataclass
class FineGrainedWeightUpdateConfig(BaseConfig):
    """Serial, independently recoverable weight updates for rollout replicas."""

    enabled: bool = False
    max_retries: int = 1
    retry_backoff_s: float = 1.0
    continue_on_failure: bool = True

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError("fine_grained_weight_update.max_retries must be non-negative")
        if self.retry_backoff_s < 0:
            raise ValueError("fine_grained_weight_update.retry_backoff_s must be non-negative")


@dataclass
class JointSchedulingConfig(BaseConfig):
    """Weight-centered admission, migration and independently paced updates."""

    enabled: bool = False
    kv_capacity_tokens: int = 0
    max_batch_size: int = 0
    kv_block_size_tokens: int = 16
    poll_interval_s: float = 0.1
    update_cooldown_s: float = 5.0
    min_decode_window_s: float = 2.0
    min_decode_tokens: int = 64
    max_defer_s: float = 30.0
    evacuation_timeout_s: float = 5.0
    sparse_request_threshold: int = 1
    staleness_version_gap: int = 2
    tail_tokens: int = 32
    tail_kl_threshold: float = 0.1
    probe_interval_s: float = 5.0
    probe_timeout_s: float = 2.0
    max_concurrent_probes: int = 1
    overload_threshold: float = 0.9

    def __post_init__(self):
        integer_minima = {
            "kv_capacity_tokens": 1 if self.enabled else 0,
            "max_batch_size": 1 if self.enabled else 0,
            "kv_block_size_tokens": 1,
            "min_decode_tokens": 0,
            "sparse_request_threshold": 0,
            "staleness_version_gap": 1,
            "tail_tokens": 1,
            "max_concurrent_probes": 1,
        }
        for name, minimum in integer_minima.items():
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"joint_scheduling.{name} must be an integer >= {minimum}")
        positive = {"poll_interval_s", "max_defer_s", "probe_timeout_s"}
        for name in (
            "poll_interval_s",
            "update_cooldown_s",
            "min_decode_window_s",
            "max_defer_s",
            "evacuation_timeout_s",
            "tail_kl_threshold",
            "probe_interval_s",
            "probe_timeout_s",
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name in positive and value == 0)
            ):
                constraint = "positive" if name in positive else "non-negative"
                raise ValueError(f"joint_scheduling.{name} must be finite and {constraint}")
        if (
            type(self.overload_threshold) not in (int, float)
            or not math.isfinite(self.overload_threshold)
            or not 0 < self.overload_threshold <= 1
        ):
            raise ValueError("joint_scheduling.overload_threshold must be in (0, 1]")


@dataclass
class WeightRelayConfig(BaseConfig):
    """CPU staging limits for the Mooncake RDMA weight relay."""

    device_name: str = ""
    max_staging_bytes: int = 4 << 30
    pipeline_depth: int = 2

    def __post_init__(self):
        if not isinstance(self.device_name, str):
            raise ValueError("weight_publication.relay.device_name must be a string")
        for name in ("max_staging_bytes", "pipeline_depth"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"weight_publication.relay.{name} must be a positive integer")


@dataclass
class GPUSnapshotConfig(BaseConfig):
    """Per-GPU persistent storage and temporary working memory limits."""

    slots: int = 2
    max_bytes_per_gpu: int = 10 << 30
    max_conversion_bytes: int = 8 << 30
    max_tensor_bytes: int = 4 << 30
    chunk_bytes: int = 256 << 20
    operation_timeout_s: float = 120.0
    reserve_timeout_s: float = 600.0

    def __post_init__(self):
        for name in ("slots", "max_bytes_per_gpu", "max_conversion_bytes", "max_tensor_bytes", "chunk_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"weight_publication.gpu_snapshot.{name} must be a positive integer")
        if self.slots < 2:
            raise ValueError("GPU publication requires at least two version slots")
        if self.chunk_bytes > min(self.max_bytes_per_gpu, self.max_tensor_bytes):
            raise ValueError("GPU chunk_bytes exceeds storage or temporary tensor capacity")
        for name in ("operation_timeout_s", "reserve_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"weight_publication.gpu_snapshot.{name} must be finite and positive")


@dataclass
class WeightPublicationConfig(BaseConfig):
    """Publish immutable snapshots independently of replica installation."""

    enabled: bool = False
    wait_timeout_s: float = 600.0
    transport: str = "ray"
    relay: WeightRelayConfig = field(default_factory=WeightRelayConfig)
    gpu_snapshot: GPUSnapshotConfig = field(default_factory=GPUSnapshotConfig)
    joint_scheduling: JointSchedulingConfig = field(default_factory=JointSchedulingConfig)

    def __post_init__(self):
        if not math.isfinite(self.wait_timeout_s) or self.wait_timeout_s <= 0:
            raise ValueError("weight_publication.wait_timeout_s must be finite and positive")
        if self.transport not in ("ray", "mooncake_rdma", "nccl_gpu"):
            raise ValueError("weight_publication.transport must be 'ray', 'mooncake_rdma' or 'nccl_gpu'")
        if self.transport != "ray" and not self.enabled:
            raise ValueError(f"{self.transport} transport requires weight_publication.enabled=True")
        if isinstance(self.gpu_snapshot, (dict, DictConfig)):
            values = self.gpu_snapshot
            if isinstance(values, DictConfig):
                values = OmegaConf.to_container(values, resolve=True)
            object.__setattr__(self, "gpu_snapshot", GPUSnapshotConfig(**values))
        elif not isinstance(self.gpu_snapshot, GPUSnapshotConfig):
            raise TypeError("weight_publication.gpu_snapshot must be dict, DictConfig, or GPUSnapshotConfig")
        if isinstance(self.relay, (dict, DictConfig)):
            relay = self.relay
            if isinstance(relay, DictConfig):
                relay = OmegaConf.to_container(relay, resolve=True)
            object.__setattr__(self, "relay", WeightRelayConfig(**relay))
        elif not isinstance(self.relay, WeightRelayConfig):
            raise TypeError("weight_publication.relay must be dict, DictConfig, or WeightRelayConfig")
        if isinstance(self.joint_scheduling, (dict, DictConfig)):
            joint = self.joint_scheduling
            if isinstance(joint, DictConfig):
                joint = OmegaConf.to_container(joint, resolve=True)
            object.__setattr__(self, "joint_scheduling", JointSchedulingConfig(**joint))
        elif not isinstance(self.joint_scheduling, JointSchedulingConfig):
            raise TypeError("weight_publication.joint_scheduling must be dict, DictConfig, or JointSchedulingConfig")
        if self.joint_scheduling.enabled and not self.enabled:
            raise ValueError("joint_scheduling requires weight_publication.enabled=True")


@dataclass
class TrajectoryMigrationConfig(BaseConfig):
    """Dynamic scheduling of an in-progress trajectory across rollout replicas."""

    enabled: bool = False
    allow_cross_version_recompute: bool = False
    checkpoint_tokens: int = 64
    scorer_class: str = "verl.workers.rollout.trajectory_scheduler.LoadAwareTrajectoryReplicaScorer"
    scorer_kwargs: dict = field(default_factory=dict)
    gate_classes: list[str] = field(
        default_factory=lambda: [
            "verl.workers.rollout.trajectory_scheduler.DifferentReplicaGate",
            "verl.workers.rollout.trajectory_scheduler.SameModelGate",
            "verl.workers.rollout.trajectory_scheduler.SameWeightVersionGate",
            "verl.workers.rollout.trajectory_scheduler.KVTransferCapabilityGate",
        ]
    )
    gate_kwargs: dict = field(default_factory=dict)
    min_score_improvement: float = 0.0
    kv_transfer_backend: str = "remote_prefix"
    transfer_timeout_s: float = 30.0
    storage_preflight: bool = True
    strict_remote_hit: bool = False
    retry_backoff_s: float = 1.0
    max_retry_backoff_s: float = 30.0
    backend_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.checkpoint_tokens <= 0:
            raise ValueError("trajectory_migration.checkpoint_tokens must be positive")
        if (
            type(self.transfer_timeout_s) not in (int, float)
            or not math.isfinite(self.transfer_timeout_s)
            or self.transfer_timeout_s <= 0
        ):
            raise ValueError("trajectory_migration.transfer_timeout_s must be finite and positive")
        for name in ("retry_backoff_s", "max_retry_backoff_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"trajectory_migration.{name} must be finite and non-negative")
        if self.max_retry_backoff_s < self.retry_backoff_s:
            raise ValueError("trajectory_migration.max_retry_backoff_s must be >= retry_backoff_s")


@dataclass
class RolloutConfig(BaseConfig):
    verify_weight_updates: bool = False
    _mutable_fields = {
        "max_model_len",
        "load_format",
        "engine_kwargs",
        "prompt_length",
        "response_length",
        "expert_parallel_size",
        "moe_tensor_parallel_size",
        "full_determinism",
        "max_num_seqs",
    }

    name: Optional[str] = MISSING
    mode: str = "async"
    nnodes: int = 0
    n_gpus_per_node: int = 8

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1
    repetition_penalty: float = 1.0

    # Whether to enable full determinism for reproducibility.
    full_determinism: bool = False

    # Random seed for rollout. Used as the seed for vLLM sampling and
    # enable_full_determinism() when full_determinism is True.
    seed: int = 42

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    standalone_gpu_memory_utilization: Optional[float] = None
    ignore_eos: bool = False
    enforce_eager: bool = False
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    moe_tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    logprobs_mode: Optional[str] = "processed_logprobs"
    scheduling_policy: Optional[str] = "fcfs"

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # Ray max_concurrency of the rollout server actor.
    ray_actor_max_concurrency: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    # Sampler top-k log-probs per generated token for score centering; 0 disables.
    topk_log_probs: int = 0

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    # Use Prometheus to collect and monitor rollout statistics
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)

    # Extension point for custom configurations
    custom: Optional[dict] = None

    # Fully qualified class name for a custom CheckpointEngineManager. When set, the trainer
    # loads this class instead of the built-in CheckpointEngineManager.
    checkpoint_manager_class: Optional[str] = None

    # Checkpoint Engine config for update weights from trainer to rollout
    checkpoint_engine: CheckpointEngineConfig = field(default_factory=CheckpointEngineConfig)

    fine_grained_weight_update: FineGrainedWeightUpdateConfig = field(default_factory=FineGrainedWeightUpdateConfig)
    weight_publication: WeightPublicationConfig = field(default_factory=WeightPublicationConfig)

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None

    skip_tokenizer_init: bool = True

    quantization: Optional[str] = None

    quantization_config_file: Optional[str] = None

    enable_rollout_routing_replay: bool = False
    moe_load_balance_metrics_interval: int = 0

    enable_sleep_mode: bool = True

    mtp: MtpConfig = field(default_factory=MtpConfig)

    qat: Optional[dict] = None

    disaggregation: DisaggregationConfig = field(default_factory=DisaggregationConfig)

    router_config_path: Optional[str] = None

    trajectory_migration: TrajectoryMigrationConfig = field(default_factory=TrajectoryMigrationConfig)

    def __post_init__(self):
        """Validate the rollout config"""
        if self.name == "rtp_llm":
            if type(self.tensor_model_parallel_size) is not int or self.tensor_model_parallel_size < 1:
                raise ValueError("rtp_llm tensor_model_parallel_size must be a positive integer")
            for field_name in ("data_parallel_size", "pipeline_model_parallel_size", "expert_parallel_size"):
                if getattr(self, field_name) != 1:
                    raise ValueError(f"rtp_llm requires {field_name}=1; use independent rollout replicas for DP")
        # Deprecation warning for mode field - only async mode is supported
        if self.mode == "sync":
            raise ValueError(
                "Rollout mode 'sync' has been removed. Please set "
                "`actor_rollout_ref.rollout.mode=async` or remove the mode setting entirely."
            )
        if self.mode != "async":
            warnings.warn(
                f"Unknown rollout mode '{self.mode}'. Only 'async' mode is supported. "
                "The 'mode' field is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.name != "trtllm" and self.expert_parallel_size > 1:
            assert self.expert_parallel_size == (self.tensor_model_parallel_size * self.data_parallel_size), (
                "expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size"
            )

        if self.moe_tensor_parallel_size is not None and self.moe_tensor_parallel_size > 1:
            assert self.name == "trtllm", "moe_tensor_parallel_size is only supported for trtllm"

        if self.name == "trtllm":
            # If either expert_parallel_size or moe_tensor_parallel_size is at default 1,
            # convert to None so TensorRT-LLM treats it as unspecified.
            # When both unspecified: moe_ep_size=1, moe_tp_size=moe_world_size (no EP, all TP).
            # When only one set: the other is auto-derived from tensor_model_parallel_size.
            if self.expert_parallel_size is not None and self.expert_parallel_size == 1:
                self.expert_parallel_size = None
            if self.moe_tensor_parallel_size is not None and self.moe_tensor_parallel_size == 1:
                self.moe_tensor_parallel_size = None
            if self.expert_parallel_size is not None and self.moe_tensor_parallel_size is not None:
                assert self.moe_tensor_parallel_size * self.expert_parallel_size == self.tensor_model_parallel_size, (
                    "moe_tensor_parallel_size * expert_parallel_size must equal tensor_model_parallel_size "
                    f"(got {self.moe_tensor_parallel_size} * {self.expert_parallel_size} = "
                    f"{self.moe_tensor_parallel_size * self.expert_parallel_size}, "
                    f"tensor_model_parallel_size={self.tensor_model_parallel_size})"
                )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm" or self.name == "sglang" or self.name == "trtllm":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )

        # Hydra passes this as dict/DictConfig; coerce to dataclass so
        # downstream .enabled etc. work. BaseConfig is frozen, hence object.__setattr__.
        if isinstance(self.disaggregation, dict):
            object.__setattr__(self, "disaggregation", DisaggregationConfig(**self.disaggregation))
        elif not isinstance(self.disaggregation, DisaggregationConfig):
            if not isinstance(self.disaggregation, DictConfig):
                raise TypeError(
                    f"rollout.disaggregation must be dict, DictConfig, or DisaggregationConfig; "
                    f"got {type(self.disaggregation).__name__}."
                )
            object.__setattr__(
                self,
                "disaggregation",
                DisaggregationConfig(**OmegaConf.to_container(self.disaggregation, resolve=True)),
            )

        if self.disaggregation.enabled and self.name not in ("sglang", "vllm"):
            raise ValueError(
                f"rollout.disaggregation.enabled=True requires rollout.name in ('sglang', 'vllm'); got {self.name!r}."
            )

        if isinstance(self.fine_grained_weight_update, dict):
            object.__setattr__(
                self,
                "fine_grained_weight_update",
                FineGrainedWeightUpdateConfig(**self.fine_grained_weight_update),
            )
        elif not isinstance(self.fine_grained_weight_update, FineGrainedWeightUpdateConfig):
            if not isinstance(self.fine_grained_weight_update, DictConfig):
                raise TypeError(
                    "rollout.fine_grained_weight_update must be dict, DictConfig, or FineGrainedWeightUpdateConfig; "
                    f"got {type(self.fine_grained_weight_update).__name__}."
                )
            object.__setattr__(
                self,
                "fine_grained_weight_update",
                FineGrainedWeightUpdateConfig(**OmegaConf.to_container(self.fine_grained_weight_update, resolve=True)),
            )

        if self.fine_grained_weight_update.enabled and self.name != "rtp_llm":
            raise ValueError("fine-grained weight updates currently support rollout.name=rtp_llm only")

        if isinstance(self.weight_publication, (dict, DictConfig)):
            publication = self.weight_publication
            if isinstance(publication, DictConfig):
                publication = OmegaConf.to_container(publication, resolve=True)
            object.__setattr__(self, "weight_publication", WeightPublicationConfig(**publication))
        elif not isinstance(self.weight_publication, WeightPublicationConfig):
            raise TypeError("rollout.weight_publication must be dict, DictConfig, or WeightPublicationConfig")
        if self.weight_publication.enabled:
            if self.name != "rtp_llm":
                raise ValueError("weight_publication requires rollout.name=rtp_llm")
            if not self.fine_grained_weight_update.enabled:
                raise ValueError("weight_publication requires fine_grained_weight_update.enabled=True")

        if isinstance(self.trajectory_migration, dict):
            object.__setattr__(
                self,
                "trajectory_migration",
                TrajectoryMigrationConfig(**self.trajectory_migration),
            )
        elif not isinstance(self.trajectory_migration, TrajectoryMigrationConfig):
            if not isinstance(self.trajectory_migration, DictConfig):
                raise TypeError(
                    "rollout.trajectory_migration must be dict, DictConfig, or TrajectoryMigrationConfig; "
                    f"got {type(self.trajectory_migration).__name__}."
                )
            object.__setattr__(
                self,
                "trajectory_migration",
                TrajectoryMigrationConfig(**OmegaConf.to_container(self.trajectory_migration, resolve=True)),
            )

        if self.trajectory_migration.enabled:
            if self.name != "rtp_llm":
                raise ValueError("trajectory migration currently supports rollout.name=rtp_llm only")
            if self.trajectory_migration.kv_transfer_backend != "remote_prefix":
                raise ValueError("rtp_llm trajectory migration currently requires kv_transfer_backend=remote_prefix")
            if not self.enable_prefix_caching and not self.trajectory_migration.allow_cross_version_recompute:
                raise ValueError("rtp_llm trajectory migration requires rollout.enable_prefix_caching=True")

        if self.weight_publication.joint_scheduling.enabled:
            if not self.trajectory_migration.enabled:
                raise ValueError("joint_scheduling requires trajectory_migration.enabled=True")
            if not self.trajectory_migration.allow_cross_version_recompute:
                raise ValueError("joint_scheduling requires trajectory_migration.allow_cross_version_recompute=True")
            if not self.enable_prefix_caching:
                raise ValueError("joint_scheduling requires enable_prefix_caching=True for same-version migration")

        if self.topk_log_probs:
            if self.name != "vllm":
                raise ValueError("rollout.topk_log_probs is supported by the vLLM rollout only.")
            if (
                not self.calculate_log_probs
                or self.top_p != 1.0
                or self.top_k != -1
                or self.logprobs_mode != "processed_logprobs"
            ):
                raise ValueError(
                    "rollout.topk_log_probs needs calculate_log_probs=True, top_p=1.0, top_k=-1 and "
                    "logprobs_mode='processed_logprobs' so the returned head is the sampling distribution."
                )
            vllm_kwargs = self.engine_kwargs.setdefault("vllm", {})
            vllm_kwargs["max_logprobs"] = max(vllm_kwargs.get("max_logprobs", 0), self.topk_log_probs)
