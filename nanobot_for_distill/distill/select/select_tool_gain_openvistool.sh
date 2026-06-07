#!/bin/bash
# Batch-run select_tool_gain.py over the Chart / GUI / VisualSearch sub-domains
# and collect the per-dataset tool-gain selections under one output dir.
#
# No-tool avg@k is read from the .progress.jsonl produced by filter_difficulty.py
# (field `avg_k`) — it covers every sample, unlike the *_difficulty_filtered.jsonl
# which only keeps the difficult ones and stores scores inconsistently.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTDIR="$ROOT/dataset/OpenVisTool_toolgain_only/tool-gain-report"
GAIN_THRESHOLD=0.1
mkdir -p "$OUTDIR"

# domain<TAB>difficulty_filtered_jsonl[<TAB>notool_progress_override]
#   notool      = ${DATASET%.jsonl}.progress.jsonl   (score field: avg_k)
#                 — overridable via the optional 3rd field when the difficulty
#                   file and the no-tool progress file have different prefixes
#                   (e.g. CoSyn-400K: tool-gain ran on a range-filtered subset).
#   with-prefix = ${DATASET%.jsonl}_tool_gain_prefix.jsonl
DATASETS=(
  "Chart	dataset/Chart/ChartVerse-SFT-600K/ChartVerse-SFT-100K_difficulty_filtered_qwen35_9b.jsonl"
  "GUI	dataset/GUI/AgentNet_processed/ubuntu_click_highres_difficulty_filtered_qwen35_9b.jsonl"
  "GUI	dataset/GUI/GUI-Actor/uground_hard_b_landscape_70k_difficulty_filtered_qwen35_9b.jsonl"
  "GUI	dataset/GUI/OS-Atlas-data/os_atlas_linux_highres_small_bbox_difficulty_filtered.jsonl"
  "GUI	dataset/GUI/OS-Atlas-data/os_atlas_macos_highres_small_bbox_difficulty_filtered.jsonl"
  "GUI	dataset/GUI/OS-Atlas-data/os_atlas_windows_highres_small_bbox_difficulty_filtered.jsonl"
  "VisualSearch	dataset/VisualSearch/DeepEyesV2_RL_highres_nonchart_difficulty_filtered_qwen35_9b.jsonl"
  "VisualSearch	dataset/VisualSearch/Vero-600k-visual-search_difficulty_filtered_qwen35_9b.jsonl"
  "Table	dataset/Table/CoSyn-400K/table_difficulty_range_0.25_0.75.jsonl	dataset/Table/CoSyn-400K/table_difficulty_filtered_qwen35_9b.progress.jsonl"
  "Table	dataset/Table/TABLET-Small/TABLET-Small_difficulty_filtered_qwen35_9b.jsonl"
  "VinciCoder	dataset/Code/VinciCoder-1.6M-SFT/vincicoder_bucket_selected_difficulty_filtered_qwen35_9b.jsonl"
)

for entry in "${DATASETS[@]}"; do
  IFS=$'\t' read -r DOMAIN DATASET NOTOOL_OVERRIDE <<<"$entry"
  NOTOOL=${NOTOOL_OVERRIDE:-${DATASET%.jsonl}.progress.jsonl}
  WITH_PREFIX=${DATASET%.jsonl}_tool_gain_prefix.jsonl
  BASE=$(basename "$DATASET" .jsonl)
  OUTPUT="$OUTDIR/${DOMAIN}__${BASE}_tool_gain_selected.jsonl"
  REPORT=${OUTPUT%.jsonl}_report.json

  echo "================ [$DOMAIN] $BASE ================"
  if [[ ! -f "$ROOT/$NOTOOL" ]]; then
    echo "SKIP: no-tool progress file missing: $NOTOOL"
    continue
  fi
  if [[ ! -f "$ROOT/$WITH_PREFIX" ]]; then
    echo "SKIP: with-prefix file missing: $WITH_PREFIX"
    continue
  fi

  python "$ROOT/distill/select/select_tool_gain.py" \
      --notool-jsonl "$ROOT/$NOTOOL" \
      --notool-score-field avg_k \
      --with-prefix-jsonl "$ROOT/$WITH_PREFIX" \
      --output "$OUTPUT" \
      --report "$REPORT" \
      --gain-threshold "$GAIN_THRESHOLD"
done
