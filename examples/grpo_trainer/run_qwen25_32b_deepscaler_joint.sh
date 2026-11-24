#!/usr/bin/env bash
# V5 D policy: 4 FSDP2 trainers, 3 TP4 replicas, trainer-side conversion.
# Requires the matching RTP wheel with request_kv_checkpoint and a working KVCM store.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${KVCM_SERVER_ADDRESS:?Set the KVCM metadata endpoint (host:port)}"
: "${KVCM_INSTANCE_GROUP:?Set the protocol-matched Mooncake storage group}"
export RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
export MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-32B-Instruct"}
export PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen25-32B-DeepScaleR}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25-trainer-conversion-version-only}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export N_RESPONSES=${N_RESPONSES:-4}
export TRAIN_MINI_BATCH_SIZE=${TRAIN_MINI_BATCH_SIZE:-8}
export TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-38421}
export CONCURRENT_SAMPLES_PER_REPLICA=${CONCURRENT_SAMPLES_PER_REPLICA:-256}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}

# Budgets reproduce the handoff configuration; measure memory fit on the target hardware.
exec bash "${SCRIPT_DIR}/run_qwen3_5_27b_fsdp2_deepscaler.sh" \
    trainer.nnodes=1 trainer.n_gpus_per_node=4 \
    rollout.nnodes=3 rollout.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.ray_actor_max_concurrency=4096 \
    actor_rollout_ref.rollout.weight_publication.enabled=True \
    actor_rollout_ref.rollout.weight_publication.transport=nccl_gpu \
    actor_rollout_ref.rollout.weight_publication.gpu_snapshot.bucket_bytes=536870912 \
    actor_rollout_ref.rollout.weight_publication.gpu_snapshot.pipeline_depth=2 \
    actor_rollout_ref.rollout.weight_publication.gpu_snapshot.conversion_location=trainer \
    actor_rollout_ref.rollout.weight_publication.gpu_snapshot.cache_conversion_plan=True \
    actor_rollout_ref.rollout.weight_publication.gpu_snapshot.memory_planning=role \
    actor_rollout_ref.rollout.weight_publication.gpu_snapshot.converter_replica_rank=null \
    'actor_rollout_ref.rollout.weight_publication.gpu_snapshot.trainer_tp_ranks=[0,1,2,3]' \
    actor_rollout_ref.rollout.weight_publication.gpu_snapshot.max_bytes_per_gpu="${SNAPSHOT_STORAGE_BYTES:-12884901888}" \
    actor_rollout_ref.rollout.weight_publication.gpu_snapshot.max_tensor_bytes="${SNAPSHOT_TENSOR_BYTES:-6442450944}" \
    actor_rollout_ref.rollout.weight_publication.gpu_snapshot.max_conversion_bytes="${SNAPSHOT_CONVERSION_BYTES:-34359738368}" \
    actor_rollout_ref.rollout.trajectory_migration.checkpoint_mode=on_demand \
    actor_rollout_ref.rollout.trajectory_migration.progress_poll_interval_s=1.0 \
    ++actor_rollout_ref.rollout.trajectory_migration.strict_remote_hit=False \
    +actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_server_address="${KVCM_SERVER_ADDRESS}" \
    +actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_instance_group="${KVCM_INSTANCE_GROUP}" \
    '+actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_model_sdk_config="[{\"type\":\"mooncake\",\"location\":\"*\",\"put_replica_num\":1}]"' \
    +actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_put_timeout_ms=10000 \
    +actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_get_timeout_ms=10000 \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.enabled=True \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.kv_capacity_tokens="${JOINT_KV_CAPACITY_TOKENS:-10485760}" \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.kv_block_size_tokens=16 \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.max_batch_size=1024 \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.staleness_mode=version \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.staleness_version_gap=2 \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.max_defer_s=120.0 \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.min_decode_tokens=0 \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.min_decode_window_s=2.0 \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.update_cooldown_s=5.0 \
    actor_rollout_ref.rollout.weight_publication.joint_scheduling.evacuation_timeout_s=5.0 \
    async_training.sample_budget_enabled=False \
    async_training.completed_queue_size=128 \
    async_training.trigger_parameter_sync_step=1 \
    trainer.total_training_steps=10 trainer.save_freq=-1 trainer.resume_mode=disable \
    trainer.val_before_train=True trainer.test_freq=10 \
    data.val_max_samples=500 data.val_batch_size=500 data.seed=42 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    ++trainer.learning_acceptance=True \
    ++trainer.learning_acceptance_max_zero_batches=32 \
    ++actor_rollout_ref.actor.optim.verify_parameter_updates=True \
    ++actor_rollout_ref.rollout.verify_weight_updates=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    "$@"
