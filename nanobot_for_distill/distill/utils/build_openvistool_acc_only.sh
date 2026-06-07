#!/bin/bash
# Build the "acc only" ablation dataset: for each dataset, convert the
# correctness-filtered index directly to ms-swift format, then merge per domain.
# No tool-gain filtering; counterpart to build_openvistool_both.sh.
#
# Companion variants for the ablation:
#   dataset/OpenVisTool_acc_only       -- this script (correctness only)
#   dataset/OpenVisTool_toolgain_only  -- tool-gain only
#   dataset/OpenVisTool                -- correctness ∩ tool-gain (both)
#
# Note: only the *new* datasets are listed in JOBS by default; the original
# Chart / GUI / VisualSearch / Web2html acc_only files were built ad-hoc and
# are preserved as-is to avoid count-suffix churn. Re-add entries here when you
# want to fully reproduce those files.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WS="$ROOT/workspaces/qwen35plus"
OUTDIR="$ROOT/dataset/OpenVisTool_acc_only"
SWIFTDIR="$OUTDIR/swift"
WORKERS=16
mkdir -p "$SWIFTDIR"

# Full enabled-tool list from workspaces/qwen35plus/config.json (enabledTools).
TOOLS='read_file,write_file,edit_file,list_dir,exec,crop,locate_in_crop,draw_bbox,draw_line,draw_circle,in_range_color,rotate,flip,enhance_contrast,adjust_brightness,detect_edges,grayscale,connected_components,find_contours,hough_lines,hough_circles,template_match,computer_use,render_html'

# domain<TAB>basename<TAB>correctness_index(rel ROOT)<TAB>sessions_dir(rel WS)
JOBS=(
  "Table	CoSyn-400K	workspaces/qwen35plus/CoSyn-400K_filtered_index.jsonl	CoSyn-400K"
  "Table	TABLET-Small	workspaces/qwen35plus/TABLET-Small_filtered_index.jsonl	TABLET-Small"
  "VinciCoder	VinciCoder	workspaces/qwen35plus/VinciCoder-filtered-index-12k.jsonl	VinciCoder"
)

declare -A DOMAIN_FILES

for entry in "${JOBS[@]}"; do
  IFS=$'\t' read -r DOMAIN BASE CORRECTNESS SESSREL <<<"$entry"
  SWIFT="$SWIFTDIR/${DOMAIN}__${BASE}_swift.jsonl"
  SESSIONS="$WS/$SESSREL"

  echo "================ $BASE ================"
  if [[ ! -f "$ROOT/$CORRECTNESS" ]]; then echo "SKIP: correctness index missing: $CORRECTNESS"; continue; fi
  if [[ ! -d "$SESSIONS" ]]; then echo "SKIP: sessions dir missing: $SESSIONS"; continue; fi

  python "$ROOT/distill/utils/convert_to_swift.py" \
      --sessions-dir "$SESSIONS" \
      --index-file "$ROOT/$CORRECTNESS" \
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
  MERGED="$OUTDIR/${DOMAIN_LABEL[$DOMAIN]}_acc_only_${KK}k.jsonl"
  cat $FILES > "$MERGED"
  printf '%-40s lines=%s\n' "$(basename "$MERGED")" "$LINES"
done
