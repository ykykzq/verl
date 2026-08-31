#!/usr/bin/env bash
# GRPO | Qwen3.5-9B | MATH-500 | fully-async (separated Trainer / Rollouter)
# Rollouter: rtp-llm (replaces vLLM)
# Hardware: 8x AMD Instinct MI308X (gfx942), ROCm 7.2 -> 4 GPUs trainer + 4 GPUs rollouter
set -xeuo pipefail

project_name='GRPO-Qwen3.5-9B'
exp_name='qwen35-9b-math500-fully-async-4t-4r-rtpllm-len16k'

WS=${WS:-/home/admin/workspace/aop_lab}
MODEL_PATH=${MODEL_PATH:-"${WS}/app_source/lh_workspace/models/Qwen3.5-9B"}
DATA_DIR=${DATA_DIR:-"${WS}/app_data/data/math500"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${WS}/app_data/ckpts/${project_name}/${exp_name}"}
LOG_DIR=${LOG_DIR:-"${WS}/app_data/logs/${project_name}"}
mkdir -p "${LOG_DIR}" "${CKPTS_DIR}"

# ---------------- ROCm / MI308X runtime ----------------
# Let Ray manage per-worker device assignment; do not pin visibility globally
# (a global pin leaks into workers and conflicts with Ray's per-worker HIP set).
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES XPU_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Weight sync hands rtp-llm raw device IPC handles; legacy mode is required for
# cross-process HIP IPC between the trainer worker and the engine actor.
export HSA_ENABLE_IPC_MODE_LEGACY=1
export MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1
export NCCL_DEBUG=ERROR
# The trainer ranks JIT-compile Qwen3.5 linear-attention (TileLang) kernels on
# the first forward, which can take minutes and previously tripped the NCCL
# watchdog, aborting the process group. Extend the collective/heartbeat timeouts
# and relax the monitor so one-time compilation cannot tear down the PG.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export VERL_NCCL_TIMEOUT=3600
export ROCPROFILER_LOG_LEVEL=fatal
export TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=1

# ---------------- rtp-llm rollouter ----------------
export RTP_LLM_LOG_LEVEL=${RTP_LLM_LOG_LEVEL:-WARNING}
# Base port for the engine's gRPC / HTTP listeners; each replica offsets from here.
export VERL_RTP_LLM_START_PORT=${VERL_RTP_LLM_START_PORT:-31000}
# Import rtp_llm from the source tree: it carries the in-process weight-update hook and
# resolves the freshly built .so through bazel-bin.
RTP_LLM_HOME=${RTP_LLM_HOME:-"${WS}/app_source/lh_workspace/rtp-llm"}
export PYTHONPATH="${RTP_LLM_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
# The engine runs in its own interpreter: rtp-llm's Qwen3.5 kernels need triton>=3.6 and
# its .so is linked against torch 2.9.1, while the FSDP trainer needs triton 3.5.x for
# flash-linear-attention. Driver/trainer stay in `lh`, engines run in this clone.
export VERL_RTP_LLM_CONDA_ENV=${VERL_RTP_LLM_CONDA_ENV:-lh_rtp}

# ---------------- resource split ----------------
NNODES=1
n_gpus_rollout=4                 # 4 cards = Rollouter (4 data-parallel rtp-llm replicas)
n_gpus_training=4                # 4 cards = FSDP2 Trainer
gen_tp=1                         # TP=1 => 4 independent replicas => 4x rollout bandwidth
fsdp_size=4
sp_size=1

# ---------------- sequence lengths ----------------
max_prompt_length=1024
# Qwen3.5 runs with thinking enabled by default; even 8192 saturated the cap
# (clip_ratio up to 0.67), so give the reasoning chain ~2x room to finish.
max_response_length=$((1024 * 16))

# ---------------- algorithm ----------------
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# ---------------- batching ----------------
n_resp_per_prompt=8
train_prompt_bsz=0               # unused in fully-async
gen_prompt_bsz=1                 # streaming, one sample at a time
train_prompt_mini_bsz=24         # global mini-batch (divisible by 4 GPUs)
total_rollout_steps=$((500 * 4)) # 4 epochs over MATH-500
test_freq=5
save_freq=10

# ---------------- fully-async knobs ----------------
staleness_threshold=0.5
trigger_parameter_sync_step=2
require_batches=1
partial_rollout=True

# memory: Qwen3.5's 248320-vocab lm_head costs ~1MB of logits per token, so
# token-packed micro-batches OOM on the log-prob pass. Use micro-batch 1 plus
# chunked entropy, matching verl's own qwen3_5 FSDP examples.
use_dynamic_bsz=False
micro_bsz_per_gpu=1

# ---------------- overlong handling (DAPO soft penalty) ----------------
# Replaces the hard truncation-to-zero-reward with a graded penalty so that
# over-length responses still carry gradient signal.
enable_overlong_buffer=True
overlong_buffer_len=4096
overlong_penalty_factor=1.0

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.bypass_mode=False \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_fused_kernels=True \
    ++actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=rtp_llm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    critic.strategy=fsdp2 \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    trainer.val_before_train=False \
    trainer.test_freq="${test_freq}" \
    trainer.save_freq="${save_freq}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.total_epochs=4 \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.concurrent_samples_per_replica=16 \
    "$@" 2>&1 | tee "${LOG_DIR}/${exp_name}-$(date +%Y%m%d_%H%M%S).log"
