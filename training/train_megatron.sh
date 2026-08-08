#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${SWIFT_HOME:?Set SWIFT_HOME to an ms-swift checkout (experiments used version 4.2.0.dev0)}"
: "${DATA_ROOT:?Set DATA_ROOT to the root of an OpenVisTool-42K snapshot}"
: "${MODEL:?Set MODEL to a local model path or Hugging Face model ID}"

RUN_NAME="${RUN_NAME:-openvistool_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}}"

dataset_paths=(
    "${DATA_ROOT}/data/chart.jsonl"
    "${DATA_ROOT}/data/gui_grounding.jsonl"
    "${DATA_ROOT}/data/table.jsonl"
    "${DATA_ROOT}/data/web_to_html.jsonl"
    "${DATA_ROOT}/data/visual_search.jsonl"
)

for path in "${dataset_paths[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "error: missing dataset shard: ${path}" >&2
        exit 1
    fi
done

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TP="${TP:-2}"
PP="${PP:-1}"
CP="${CP:-1}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
TRAIN_ITERS="${TRAIN_ITERS:-}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-7}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.03}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
REPORT_TO="${REPORT_TO:-tensorboard}"

total_gpus=$((NNODES * NPROC_PER_NODE))
parallel_product=$((TP * PP * CP))
if (( parallel_product < 1 || total_gpus % parallel_product != 0 )); then
    echo "error: total_gpus=${total_gpus} must be divisible by TP*PP*CP=${parallel_product}" >&2
    exit 1
fi

data_parallel_size=$((total_gpus / parallel_product))
physical_batch=$((data_parallel_size * MICRO_BATCH_SIZE))
if (( physical_batch < 1 || GLOBAL_BATCH_SIZE % physical_batch != 0 )); then
    echo "error: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by DP*MICRO_BATCH_SIZE=${physical_batch}" >&2
    exit 1
fi

gradient_accumulation_steps=$((GLOBAL_BATCH_SIZE / physical_batch))
read -r -a report_to_args <<< "${REPORT_TO}"
if [[ -n "${TRAIN_ITERS}" ]]; then
    train_duration_args=(--train_iters "${TRAIN_ITERS}")
else
    train_duration_args=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
fi

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-8192}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_DIR}"

echo "model=${MODEL}"
echo "data_root=${DATA_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "gpus=${total_gpus} tp=${TP} pp=${PP} cp=${CP} dp=${data_parallel_size}"
echo "micro_batch=${MICRO_BATCH_SIZE} grad_accum=${gradient_accumulation_steps} global_batch=${GLOBAL_BATCH_SIZE}"

# Image paths in the public JSONL are relative to the snapshot root.
cd "${DATA_ROOT}"

torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${SWIFT_HOME}/swift/cli/_megatron/sft.py" \
    --model "${MODEL}" \
    --dataset "${dataset_paths[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --truncation_strategy delete \
    --padding_free true \
    --packing false \
    --dataset_shuffle true \
    --split_dataset_ratio 0.0 \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    "${train_duration_args[@]}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --pipeline_model_parallel_size "${PP}" \
    --tensor_model_parallel_size "${TP}" \
    --context_parallel_size "${CP}" \
    --sequence_parallel true \
    --lr "${LR}" \
    --min_lr "${MIN_LR}" \
    --lr_decay_style cosine \
    --lr_warmup_fraction "${LR_WARMUP_FRACTION}" \
    --weight_decay 0.1 \
    --weight_decay_incr_style constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_eps 1e-8 \
    --clip_grad 1.0 \
    --seed 42 \
    --data_seed 42 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --recompute_modules core_attn \
    --finetune true \
    --tuner_type full \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --gradient_accumulation_fusion false \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --torch_dtype bfloat16 \
    --save_strategy epoch \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_safetensors true \
    --no_save_optim true \
    --no_save_rng true \
    --logging_steps 1 \
    --report_to "${report_to_args[@]}" \
    --wandb_exp_name "${RUN_NAME}"
