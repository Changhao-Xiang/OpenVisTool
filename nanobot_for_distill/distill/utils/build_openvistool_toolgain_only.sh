#!/bin/bash
# Build the "tool-gain only" ablation dataset: for each dataset, take the
# tool-gain-selected index from select_tool_gain_openvistool.sh, convert the
# matching sessions to ms-swift format, then merge per domain.
#
# Companion variants for the ablation:
#   dataset/OpenVisTool_acc_only       -- correctness only
#   dataset/OpenVisTool_toolgain_only  -- this script (tool-gain only)
#   dataset/OpenVisTool                -- correctness ∩ tool-gain (both)
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TGDIR="$ROOT/dataset/OpenVisTool_toolgain_only/tool-gain-report"
WS="$ROOT/workspaces/qwen35plus"
OUTDIR="$ROOT/dataset/OpenVisTool_toolgain_only"
SWIFTDIR="$OUTDIR/swift"
WORKERS=16
mkdir -p "$SWIFTDIR"

# Full enabled-tool list from workspaces/qwen35plus/config.json (enabledTools).
TOOLS='read_file,write_file,edit_file,list_dir,exec,crop,locate_in_crop,draw_bbox,draw_line,draw_circle,in_range_color,rotate,flip,enhance_contrast,adjust_brightness,detect_edges,grayscale,connected_components,find_contours,hough_lines,hough_circles,template_match,computer_use,render_html'

# domain<TAB>tool_gain_basename<TAB>sessions_dir(rel WS)
JOBS=(
  "Chart	Chart__ChartVerse-SFT-100K_difficulty_filtered_qwen35_9b	ChartVerse-SFT-qwen35-9b_filtered"
  "GUI	GUI__ubuntu_click_highres_difficulty_filtered_qwen35_9b	AgentNet_ubuntu"
  "GUI	GUI__uground_hard_b_landscape_70k_difficulty_filtered_qwen35_9b	GUI-Actor"
  "GUI	GUI__os_atlas_linux_highres_small_bbox_difficulty_filtered	OS-Atlas-data/linux"
  "GUI	GUI__os_atlas_macos_highres_small_bbox_difficulty_filtered	OS-Atlas-data/macos"
  "GUI	GUI__os_atlas_windows_highres_small_bbox_difficulty_filtered	OS-Atlas-data/windows"
  "VisualSearch	VisualSearch__DeepEyesV2_RL_highres_nonchart_difficulty_filtered_qwen35_9b	DeepEyesV2_RL"
  "VisualSearch	VisualSearch__Vero-600k-visual-search_difficulty_filtered_qwen35_9b	Vero-600k-visual-search"
  "Table	Table__table_difficulty_range_0_0.5	CoSyn-400K"
  "Table	Table__TABLET-Small_difficulty_filtered_qwen35_9b	TABLET-Small"
  "VinciCoder	VinciCoder__vincicoder_bucket_selected_difficulty_filtered_qwen35_9b	VinciCoder"
)

declare -A DOMAIN_FILES

for entry in "${JOBS[@]}"; do
  IFS=$'\t' read -r DOMAIN TGBASE SESSREL <<<"$entry"
  TG_INDEX="$TGDIR/${TGBASE}_tool_gain_selected.jsonl"
  SWIFT="$SWIFTDIR/${TGBASE}_swift.jsonl"
  SESSIONS="$WS/$SESSREL"

  echo "================ $TGBASE ================"
  if [[ ! -f "$TG_INDEX" ]]; then echo "SKIP: tool-gain index missing: $TG_INDEX"; continue; fi
  if [[ ! -d "$SESSIONS" ]]; then echo "SKIP: sessions dir missing: $SESSIONS"; continue; fi

  python "$ROOT/distill/utils/convert_to_swift.py" \
      --sessions-dir "$SESSIONS" \
      --index-file "$TG_INDEX" \
      --output "$SWIFT" \
      --tools "$TOOLS" \
      --workers "$WORKERS"

  DOMAIN_FILES[$DOMAIN]+=" $SWIFT"
done

# Merge per domain. Label mirrors dataset/OpenVisTool naming.
declare -A DOMAIN_LABEL=( [Chart]=Chart [GUI]=GUI-Grounding [VisualSearch]=VisualSearch [Table]=Table [VinciCoder]=VinciCoder )
echo "================ merge per domain ================"
for DOMAIN in "${!DOMAIN_FILES[@]}"; do
  FILES=${DOMAIN_FILES[$DOMAIN]}
  [[ -z "${FILES// }" ]] && continue
  LINES=$(cat $FILES | wc -l)
  KK=$(( (LINES + 500) / 1000 ))
  MERGED="$OUTDIR/${DOMAIN_LABEL[$DOMAIN]}_toolgain_only_${KK}k.jsonl"
  cat $FILES > "$MERGED"
  printf '%-40s lines=%s\n' "$(basename "$MERGED")" "$LINES"
done
