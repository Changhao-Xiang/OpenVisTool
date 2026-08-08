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
#   DASHBOARD=0 ./eval.sh --model-config configs/xxx.json # hide per-task dashboard table

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
for var in TEST_MODEL_BASE_URL JUDGE_BASE_URL EXTRACTOR_BASE_URL; do
    url="${!var:-}"
    if [ -n "$url" ] && _is_internal "$url"; then
        echo "▶ detected internal endpoint ($url) in $var — unsetting proxies" >&2
        INTERNAL=1
        break
    fi
done
for cfg in configs/*.json; do
    [ "$INTERNAL" = "0" ] || break
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
if [ -n "${DASHBOARD:-}" ]; then
    case "${DASHBOARD,,}" in
        0|false|no|off) EXTRA+=(--no-dashboard) ;;
        1|true|yes|on)  EXTRA+=(--dashboard) ;;
        *) echo "error: DASHBOARD must be one of 0/1/false/true/no/yes/off/on" >&2; exit 1 ;;
    esac
fi

# Prefer the activated shared virtualenv/conda environment. Without one,
# retain the standalone locked evaluation environment for evaluation-only use.
ACTIVE_ENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"
if [ -n "$ACTIVE_ENV" ] && [ -x "$ACTIVE_ENV/bin/python" ]; then
    exec "$ACTIVE_ENV/bin/python" -u eval.py "${EXTRA[@]}" "$@"
fi
exec "$UV" run --locked python -u eval.py "${EXTRA[@]}" "$@"
