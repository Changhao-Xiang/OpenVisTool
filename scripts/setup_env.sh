#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PATH="${VENV_PATH:-${VIRTUAL_ENV:-${CONDA_PREFIX:-}}}"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required; install it with: python -m pip install -U uv" >&2
    exit 1
fi

if [[ -z "${ENV_PATH}" || ! -x "${ENV_PATH}/bin/python" ]]; then
    echo "error: activate the target conda/virtual environment, or set VENV_PATH" >&2
    exit 1
fi

# Keep this bootstrap machine-independent. GPU training and serving packages
# must be installed from their upstream instructions for the local CUDA stack.
uv pip install \
    --python "${ENV_PATH}/bin/python" \
    -e "${REPO_ROOT}" \
    -r "${REPO_ROOT}/evaluation/pyproject.toml"

"${ENV_PATH}/bin/python" - <<'PY'
import httpx
import nanobot
import openai

print(f"OpenVisTool evaluation dependencies are ready (openai={openai.__version__}).")
PY

echo "Environment: ${ENV_PATH}"
echo "Install the training and vLLM backends separately using the links in README.md."
