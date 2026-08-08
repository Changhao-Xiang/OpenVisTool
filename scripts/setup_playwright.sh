#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    echo "error: environment not found at ${VENV_PATH}; run scripts/setup_env.sh first" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required; install it with: python -m pip install -U uv" >&2
    exit 1
fi

uv pip install --python "${VENV_PATH}/bin/python" "playwright>=1.49.0"
"${VENV_PATH}/bin/python" -m playwright install chromium

echo "Playwright and Chromium are ready for Web-to-HTML rendering."
