#!/usr/bin/env bash
# DeepScaleR data with the existing FSDP2 + RTP-LLM fully-async recipe.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
export DATA_DIR=${DATA_DIR:-"${RAY_DATA_HOME}/data/deepscaler"}
export PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3.5-27B-DeepScaleR}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35-27b-deepscaler-rtpllm}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}

# Keep rewards answer-based by default; callers can opt into the length penalty.
exec bash "${SCRIPT_DIR}/run_qwen3_5_27b_fsdp2.sh" \
    reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    "$@"
