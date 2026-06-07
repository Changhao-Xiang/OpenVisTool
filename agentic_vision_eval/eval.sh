#!/usr/bin/env bash
# Run eval.py via uv (uv only manages the venv; the script is executed from
# source, so code changes take effect on the next run with no rebuild).
#
# Usage:
#   ./eval.sh --model-config configs/qwen35_9b_config.json
#   ./eval.sh --model-config configs/qwen35_9b_config.json -n 5
#   ./eval.sh --resume workspace/qwen35_9b/run_20260425_xxx
#   WORKERS=16 MAX_STEPS=30 ./eval.sh --model-config configs/xxx.json
#   AVG_K=4 ./eval.sh --model-config configs/xxx.json   # avg@4 metric

set -e

# --- find uv (PATH may not include ~/.local/bin in non-interactive shells) ---
UV="${UV_BIN:-$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")}"
if [ ! -x "$UV" ]; then
    echo "error: uv not found (tried PATH and $HOME/.local/bin/uv)" >&2
    exit 1
fi

# --- auto-unset proxies if any base_url in any json config is internal ---
_is_internal() {
    [[ "$1" =~ //(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.) ]]
}

INTERNAL=0
for cfg in configs/*.json; do
    [ -f "$cfg" ] || continue
    while IFS= read -r url; do
        if _is_internal "$url"; then
            echo "▶ detected internal endpoint ($url) in $cfg — unsetting proxies" >&2
            INTERNAL=1
            break 2
        fi
    done < <(grep -oE '"base_url"\s*:\s*"[^"]+"' "$cfg" | sed -E 's/.*"([^"]+)"$/\1/')
done

if [ "$INTERNAL" = "1" ]; then
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
fi

# --- run ---
EXTRA=()
[ -n "${WORKERS:-}" ]   && EXTRA+=(--workers "$WORKERS")
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$MAX_STEPS")
[ -n "${AVG_K:-}" ]     && EXTRA+=(--avg-k "$AVG_K")

# Run from source — uv only ensures the venv exists & is in sync.
exec "$UV" run python -u eval.py "${EXTRA[@]}" "$@"
