#!/bin/bash
# Build the "acc + tool-gain" ablation dataset: for each dataset, intersect the
# correctness-filtered index with the tool-gain-selected index, convert the
# surviving sessions to ms-swift format, then merge per domain.
#
# Companion variants for the ablation:
#   dataset/OpenVisTool_acc_only                 -- correctness only
#   dataset/OpenVisTool_toolgain_only  -- tool-gain only
#   dataset/OpenVisTool  -- this script (both)
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TGDIR="$ROOT/dataset/OpenVisTool_toolgain_only/tool-gain-report"
WS="$ROOT/workspaces/qwen35plus"
OUTDIR="$ROOT/dataset/OpenVisTool"
INDEXDIR="$OUTDIR/index"
SWIFTDIR="$OUTDIR/swift"
WORKERS=16
mkdir -p "$INDEXDIR" "$SWIFTDIR"

# Full enabled-tool list from workspaces/qwen35plus/config.json (enabledTools).
TOOLS='read_file,write_file,edit_file,list_dir,exec,crop,locate_in_crop,draw_bbox,draw_line,draw_circle,in_range_color,rotate,flip,enhance_contrast,adjust_brightness,detect_edges,grayscale,connected_components,find_contours,hough_lines,hough_circles,template_match,computer_use,render_html'

# domain <TAB> tool_gain_basename <TAB> correctness_index(rel ROOT) <TAB> sessions_dir(rel WS)
JOBS=(
  "Chart	Chart__ChartVerse-SFT-100K_difficulty_filtered_qwen35_9b	workspaces/qwen35plus/ChartVerse-SFT-filtered_index.jsonl	ChartVerse-SFT-qwen35-9b_filtered"
  "GUI	GUI__ubuntu_click_highres_difficulty_filtered_qwen35_9b	workspaces/qwen35plus/AgentNet_ubuntu_filtered_index.jsonl	AgentNet_ubuntu"
  "GUI	GUI__uground_hard_b_landscape_70k_difficulty_filtered_qwen35_9b	workspaces/qwen35plus/GUI-Actor_filtered_index.jsonl	GUI-Actor"
  "GUI	GUI__os_atlas_linux_highres_small_bbox_difficulty_filtered	workspaces/qwen35plus/OS-Atlas-data/linux_filtered_index.jsonl	OS-Atlas-data/linux"
  "GUI	GUI__os_atlas_macos_highres_small_bbox_difficulty_filtered	workspaces/qwen35plus/OS-Atlas-data/macos_filtered_index.jsonl	OS-Atlas-data/macos"
  "GUI	GUI__os_atlas_windows_highres_small_bbox_difficulty_filtered	workspaces/qwen35plus/OS-Atlas-data/windows_filtered_index.jsonl	OS-Atlas-data/windows"
  "VisualSearch	VisualSearch__DeepEyesV2_RL_highres_nonchart_difficulty_filtered_qwen35_9b	workspaces/qwen35plus/DeepEyesV2_RL_filtered_index.jsonl	DeepEyesV2_RL"
  "VisualSearch	VisualSearch__Vero-600k-visual-search_difficulty_filtered_qwen35_9b	workspaces/qwen35plus/Vero-600k-visual-search_filtered_index.jsonl	Vero-600k-visual-search"
  "Table	Table__table_difficulty_range_0.25_0.75	workspaces/qwen35plus/CoSyn-400K_filtered_index.jsonl	CoSyn-400K"
  "Table	Table__TABLET-Small_difficulty_filtered_qwen35_9b	workspaces/qwen35plus/TABLET-Small_filtered_index.jsonl	TABLET-Small"
  "VinciCoder	VinciCoder__vincicoder_bucket_selected_difficulty_filtered_qwen35_9b	workspaces/qwen35plus/VinciCoder-filtered-index-12k.jsonl	VinciCoder"
)

# domain -> space-separated list of per-dataset swift files
declare -A DOMAIN_FILES

for entry in "${JOBS[@]}"; do
  IFS=$'\t' read -r DOMAIN TGBASE CORRECTNESS SESSREL <<<"$entry"
  TG_INDEX="$TGDIR/${TGBASE}_tool_gain_selected.jsonl"
  BOTH_INDEX="$INDEXDIR/${TGBASE}_correct_and_gain.jsonl"
  SWIFT="$SWIFTDIR/${TGBASE}_swift.jsonl"
  SESSIONS="$WS/$SESSREL"

  echo "================ $TGBASE ================"
  if [[ ! -f "$TG_INDEX" ]]; then echo "SKIP: tool-gain index missing: $TG_INDEX"; continue; fi
  if [[ ! -f "$ROOT/$CORRECTNESS" ]]; then echo "SKIP: correctness index missing: $CORRECTNESS"; continue; fi
  if [[ ! -d "$SESSIONS" ]]; then echo "SKIP: sessions dir missing: $SESSIONS"; continue; fi

  # Step 1: correctness ∩ tool-gain
  python "$ROOT/distill/select/select_correct_and_gain.py" \
      --correctness-index "$ROOT/$CORRECTNESS" \
      --tool-gain-index "$TG_INDEX" \
      --output "$BOTH_INDEX" \
      --report "${BOTH_INDEX%.jsonl}_report.json"

  # Step 2: convert the intersected sessions to ms-swift format
  python "$ROOT/distill/utils/convert_to_swift.py" \
      --sessions-dir "$SESSIONS" \
      --index-file "$BOTH_INDEX" \
      --output "$SWIFT" \
      --tools "$TOOLS" \
      --workers "$WORKERS"

  DOMAIN_FILES[$DOMAIN]+=" $SWIFT"
done

# Step 3: merge per domain. Domain label mirrors dataset/OpenVisTool_acc_only names.
declare -A DOMAIN_LABEL=( [Chart]=Chart [GUI]=GUI-Grounding [VisualSearch]=VisualSearch [Table]=Table [VinciCoder]=VinciCoder )
echo "================ merge per domain ================"
for DOMAIN in "${!DOMAIN_FILES[@]}"; do
  FILES=${DOMAIN_FILES[$DOMAIN]}
  [[ -z "${FILES// }" ]] && continue
  LINES=$(cat $FILES | wc -l)
  KK=$(( (LINES + 500) / 1000 ))
  MERGED="$OUTDIR/${DOMAIN_LABEL[$DOMAIN]}_${KK}k.jsonl"
  cat $FILES > "$MERGED"
  printf '%-40s lines=%s\n' "$(basename "$MERGED")" "$LINES"
done
