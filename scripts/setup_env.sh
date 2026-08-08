#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${SWIFT_HOME:?Set SWIFT_HOME to an ms-swift checkout (experiments used version 4.2.0.dev0)}"

if [[ ! -f "${SWIFT_HOME}/setup.py" || ! -f "${SWIFT_HOME}/requirements/megatron.txt" ]]; then
    echo "error: SWIFT_HOME is not an ms-swift source checkout: ${SWIFT_HOME}" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required; install it with: python -m pip install -U uv" >&2
    exit 1
fi

VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    uv venv --python "${PYTHON_VERSION}" "${VENV_PATH}"
fi

# Megatron's compiled CUDA stack is intentionally validated rather than
# resolved from unconstrained PyPI packages. Reuse a known-good training
# environment (recommended), or install the versions documented in README.md
# before running this script.
if ! "${VENV_PATH}/bin/python" - <<'PY'
import importlib
import sys

required = ("torch", "transformer_engine", "flash_attn")
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")

if missing:
    print("error: the target environment is not training-ready:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(1)
PY
then
    echo "error: provision the CUDA training stack first, or point VENV_PATH at a known-good conda environment" >&2
    echo 'example: conda activate swift-fa3 && VENV_PATH="$CONDA_PREFIX" SWIFT_HOME=/path/to/ms-swift bash scripts/setup_env.sh' >&2
    exit 1
fi

# Resolve the construction package, evaluation harness, ms-swift, and
# Megatron requirements together so incompatible dependency sets fail early.
uv pip install \
    --python "${VENV_PATH}/bin/python" \
    -e "${REPO_ROOT}" \
    -e "${SWIFT_HOME}" \
    -r "${REPO_ROOT}/evaluation/pyproject.toml" \
    -r "${SWIFT_HOME}/requirements/megatron.txt"

if ! check_output="$(uv pip check --python "${VENV_PATH}/bin/python" 2>&1)"; then
    # NVIDIA's transformer-engine-cu12 wheel reports a nonstandard platform
    # tag in some otherwise working CUDA 12 environments. Its real imports are
    # validated above and below; do not hide any other dependency problem.
    unexpected="$({ printf '%s\n' "$check_output" | sed \
        -e '/^Using Python .* environment at:/d' \
        -e '/^Checked [0-9][0-9]* packages in /d' \
        -e '/^Found 1 incompatibility$/d' \
        -e '/^The package `transformer-engine-cu12` was built for a different platform$/d'; } || true)"
    if [[ -n "$unexpected" ]]; then
        printf '%s\n' "$check_output" >&2
        exit 1
    fi
    echo "warning: ignoring transformer-engine-cu12 platform-tag warning; runtime imports succeeded" >&2
else
    printf '%s\n' "$check_output"
fi

# Import the real entry point so missing binary/runtime dependencies are caught
# during setup instead of after torchrun has launched multiple workers.
"${VENV_PATH}/bin/python" -c "from swift.megatron import megatron_sft_main"

echo "Environment ready: ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"
echo "For Web-to-HTML rendering, run: VENV_PATH=${VENV_PATH} bash scripts/setup_playwright.sh"
