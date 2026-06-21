#!/bin/bash
# Build the NO-TOOL CoT-SFT counterpart of OpenVisTool for the Table &
# VisualSearch domains. Same prompts as OpenVisTool (both) -- we reuse the exact
# correct_and_gain index ids -- but the traces come from the no-tool teacher
# rollout (workspaces/qwen35plus_notool/<name>) and carry no tool calls
# (--tools none). This isolates the "tool helps" variable for the ablation:
# tool-trace SFT vs plain CoT SFT on identical prompts.
#
# NOTE: prompts are matched to OpenVisTool, but these no-tool traces are NOT
# re-filtered for answer correctness -- the no-tool teacher may answer wrong on
# some. If you want both arms correctness-clean, run filter_acc_toolcall on
# these sessions first and swap the index files below.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WS="$ROOT/workspaces/qwen35plus_notool"
OUTDIR="$ROOT/dataset/OpenVisTool_notool"
SWIFTDIR="$OUTDIR/swift"
WORKERS=16
mkdir -p "$SWIFTDIR"

# domain <TAB> swift_basename <TAB> index_file(rel ROOT) <TAB> sessions_dir(rel WS)
JOBS=(
  "Table	Table__table_difficulty_range_0_0.5	dataset/OpenVisTool/index/Table__table_difficulty_range_0_0.5_correct_and_gain.jsonl	Table__CoSyn-400K"
  "Table	Table__TABLET-Small_difficulty_filtered_qwen35_9b	dataset/OpenVisTool/index/Table__TABLET-Small_difficulty_filtered_qwen35_9b_correct_and_gain.jsonl	Table__TABLET-Small"
  "VisualSearch	VisualSearch__DeepEyesV2_RL_highres_nonchart_difficulty_filtered_qwen35_9b	dataset/OpenVisTool/index/VisualSearch__DeepEyesV2_RL_highres_nonchart_difficulty_filtered_qwen35_9b_correct_and_gain.jsonl	VisualSearch__DeepEyesV2_RL"
  "VisualSearch	VisualSearch__Vero-600k-visual-search_difficulty_filtered_qwen35_9b	dataset/OpenVisTool/index/VisualSearch__Vero-600k-visual-search_difficulty_filtered_qwen35_9b_correct_and_gain.jsonl	VisualSearch__Vero-600k"
  "Chart	Chart__ChartVerse-SFT-100K_difficulty_filtered_qwen35_9b	dataset/OpenVisTool/index/Chart__ChartVerse-SFT-100K_difficulty_filtered_qwen35_9b_correct_and_gain.jsonl	Chart__ChartVerse"
  "GUI	GUI__ubuntu_click_highres_difficulty_filtered_qwen35_9b	dataset/OpenVisTool/index/GUI__ubuntu_click_highres_difficulty_filtered_qwen35_9b_correct_and_gain.jsonl	GUI__ubuntu_click"
  "GUI	GUI__uground_hard_b_landscape_70k_difficulty_filtered_qwen35_9b	dataset/OpenVisTool/index/GUI__uground_hard_b_landscape_70k_difficulty_filtered_qwen35_9b_correct_and_gain.jsonl	GUI__uground"
  "GUI	GUI__os_atlas_linux_highres_small_bbox_difficulty_filtered	dataset/OpenVisTool/index/GUI__os_atlas_linux_highres_small_bbox_difficulty_filtered_correct_and_gain.jsonl	GUI__os_atlas_linux"
  "GUI	GUI__os_atlas_macos_highres_small_bbox_difficulty_filtered	dataset/OpenVisTool/index/GUI__os_atlas_macos_highres_small_bbox_difficulty_filtered_correct_and_gain.jsonl	GUI__os_atlas_macos"
  "GUI	GUI__os_atlas_windows_highres_small_bbox_difficulty_filtered	dataset/OpenVisTool/index/GUI__os_atlas_windows_highres_small_bbox_difficulty_filtered_correct_and_gain.jsonl	GUI__os_atlas_windows"
  "VinciCoder	VinciCoder__vincicoder_bucket_selected_difficulty_filtered_qwen35_9b	dataset/OpenVisTool/index/VinciCoder__vincicoder_bucket_selected_difficulty_filtered_qwen35_9b_correct_and_gain.jsonl	VinciCoder"
)

declare -A DOMAIN_FILES

for entry in "${JOBS[@]}"; do
  IFS=$'\t' read -r DOMAIN BASE INDEXREL SESSREL <<<"$entry"
  INDEX="$ROOT/$INDEXREL"
  SWIFT="$SWIFTDIR/${BASE}_notool_swift.jsonl"
  SESSIONS="$WS/$SESSREL"

  echo "================ $BASE ================"
  if [[ ! -f "$INDEX" ]]; then echo "SKIP: index missing: $INDEX"; continue; fi
  if [[ ! -d "$SESSIONS" ]]; then echo "SKIP: sessions dir missing: $SESSIONS"; continue; fi

  python "$ROOT/distill/utils/convert_to_swift.py" \
      --sessions-dir "$SESSIONS" \
      --index-file "$INDEX" \
      --output "$SWIFT" \
      --tools none \
      --workers "$WORKERS"

  DOMAIN_FILES[$DOMAIN]+=" $SWIFT"
done

declare -A DOMAIN_LABEL=( [VisualSearch]=VisualSearch [Table]=Table [Chart]=Chart [GUI]=GUI-Grounding [VinciCoder]=VinciCoder )
echo "================ merge per domain ================"
for DOMAIN in "${!DOMAIN_FILES[@]}"; do
  FILES=${DOMAIN_FILES[$DOMAIN]}
  [[ -z "${FILES// }" ]] && continue
  LINES=$(cat $FILES | wc -l)
  KK=$(( (LINES + 500) / 1000 ))
  MERGED="$OUTDIR/${DOMAIN_LABEL[$DOMAIN]}_notool_${KK}k.jsonl"
  cat $FILES > "$MERGED"
  printf '%-40s lines=%s\n' "$(basename "$MERGED")" "$LINES"
done
