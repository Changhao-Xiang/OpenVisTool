#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
export RUN_NAME="${RUN_NAME:-qwen3vl_8b_openvistool_42k}"
export NNODES="${NNODES:-1}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TP="${TP:-2}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
exec "${SCRIPT_DIR}/train_megatron.sh"
