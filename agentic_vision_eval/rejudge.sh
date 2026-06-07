#!/usr/bin/env bash
# Run rejudge.py via uv (same as eval.sh — env-only management).
#
# Usage:
#   ./rejudge.sh workspace/qwen35_9b/run_20260425_xxx
#   ./rejudge.sh workspace/.../run_xxx --workers 16

set -e

UV="${UV_BIN:-$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")}"
if [ ! -x "$UV" ]; then
    echo "error: uv not found (tried PATH and $HOME/.local/bin/uv)" >&2
    exit 1
fi

# Internal-IP detection mirrors eval.sh (rejudge talks to judge / extractor).
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

exec "$UV" run python -u rejudge.py "$@"
