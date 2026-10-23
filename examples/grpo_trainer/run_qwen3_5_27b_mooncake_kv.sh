#!/usr/bin/env bash
# OSS MODEL_PATH / CKPTS_DIR remain independent of the KV storage backend.
set -euo pipefail
: "${KVCM_SERVER_ADDRESS:?Set the KVCM metadata RPC host:port, not the Mooncake master address}"
: "${KVCM_INSTANCE_GROUP:?Set an instance group whose storage is configured as Mooncake}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/run_qwen3_5_27b_fsdp2.sh" \
    actor_rollout_ref.rollout.trajectory_migration.enabled=True \
    actor_rollout_ref.rollout.trajectory_migration.kv_transfer_backend=remote_prefix \
    "actor_rollout_ref.rollout.trajectory_migration.checkpoint_tokens=${MIGRATION_CHECKPOINT_TOKENS:-512}" \
    ++actor_rollout_ref.rollout.trajectory_migration.storage_preflight=True \
    "++actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_server_address='${KVCM_SERVER_ADDRESS}'" \
    "++actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_instance_group='${KVCM_INSTANCE_GROUP}'" \
    "++actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_model_sdk_config='[{\"type\":\"mooncake\"}]'" \
    "$@"
