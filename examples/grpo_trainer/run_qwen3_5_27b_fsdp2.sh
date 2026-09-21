#!/usr/bin/env bash
# GRPO | Qwen3.5-27B | MATH-500 | fully async | 8x NVIDIA GB200
# Fixed resource split: 4 GPUs for FSDP2 training + 4 GPUs for RTP-LLM rollout.
set -xeuo pipefail

PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3.5-27B}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35-27b-math500-gb200-4t-4r-rtpllm}

VERL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-27B"}
DATA_DIR=${DATA_DIR:-"${RAY_DATA_HOME}/data/math500"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
LOG_DIR=${LOG_DIR:-"${RAY_DATA_HOME}/logs/${PROJECT_NAME}"}
RTP_LLM_HOME=${RTP_LLM_HOME:-"${VERL_ROOT}/../rtp-llm"}
mkdir -p "${LOG_DIR}" "${CKPTS_DIR}"

# Ray assigns disjoint devices to the trainer and rollout placement groups.
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES XPU_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}
export VERL_NCCL_TIMEOUT=${VERL_NCCL_TIMEOUT:-3600}
export TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

export RTP_LLM_LOG_LEVEL=${RTP_LLM_LOG_LEVEL:-WARNING}
export VERL_RTP_LLM_START_PORT=${VERL_RTP_LLM_START_PORT:-31000}
export PYTHONPATH="${RTP_LLM_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
# Set VERL_RTP_LLM_CONDA_ENV when RTP-LLM and the trainer use different
# torch/triton builds. The Ray server actor will enter that environment only.

NNODES=1
TRAIN_GPUS=4
ROLLOUT_GPUS=4
FSDP_SIZE=4
GEN_TP=1
SP_SIZE=1

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_SEQUENCE_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

N_RESPONSES=${N_RESPONSES:-8}
TRAIN_MINI_BATCH_SIZE=${TRAIN_MINI_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=1
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-2000}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
CONCURRENT_SAMPLES_PER_REPLICA=${CONCURRENT_SAMPLES_PER_REPLICA:-8}

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=rtp_llm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${N_RESPONSES} \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_SEQUENCE_LENGTH} \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    critic.strategy=fsdp2 \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${TRAIN_GPUS} \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.total_epochs=4 \
    rollout.nnodes=${NNODES} \
    rollout.n_gpus_per_node=${ROLLOUT_GPUS} \
    rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS} \
    async_training.staleness_threshold=0.5 \
    async_training.trigger_parameter_sync_step=2 \
    async_training.require_batches=1 \
    async_training.partial_rollout=True \
    async_training.concurrent_samples_per_replica=${CONCURRENT_SAMPLES_PER_REPLICA} \
    "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}-$(date +%Y%m%d_%H%M%S).log"
