# OpenVisTool-Bench evaluation

This directory contains the offline agent harness and the five scoring modes
used for OpenVisTool-Bench. The benchmark itself is distributed separately on
Hugging Face.

## Setup

The repository-level bootstrap installs OpenVisTool and the evaluation client
in the currently active environment. It does not alter CUDA packages:

```bash
conda create -n openvistool python=3.12 pip -y
conda activate openvistool
python -m pip install -U uv
bash scripts/setup_env.sh
cd evaluation
ln -s /path/to/OpenVisTool-Bench dataset
cp configs/model.example.json configs/model.json
cp configs/general_judge.example.json configs/general_judge.json
```

Keep `openvistool` activated for `eval.sh`, which uses that environment
directly. With no active environment it falls back to the locked
evaluation-only `uv` environment.

For an evaluation-only locked environment, run `uv sync --locked` inside
`evaluation/`. Web-to-HTML evaluation additionally requires Playwright and
Chromium; from the repository root run
`VENV_PATH="$CONDA_PREFIX" bash scripts/setup_playwright.sh` for the shared environment,
or `uv sync --locked --extra web && uv run playwright install chromium` for
the evaluation-only environment.

The JSON templates use `${ENVIRONMENT_VARIABLE}` placeholders. The config
loader expands them at runtime and fails if a required variable is missing.
This keeps credentials out of files and frozen run metadata.

```bash
export TEST_MODEL_ID=/path/to/model
export TEST_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export TEST_MODEL_API_KEY=EMPTY
export JUDGE_MODEL=your-judge-model
export JUDGE_BASE_URL=https://your-openai-compatible-endpoint/v1
export JUDGE_API_KEY=...
```

For local inference, vLLM is the default serving path. Install it using the
[official GPU instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html).
It can live in the shared environment when dependency versions are compatible,
or in a separate serving environment because the harness only uses its HTTP
endpoint. For Qwen3/Qwen3.5, enable the native tool-call parser and keep
`TEST_MODEL_ID` equal to the served name:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/checkpoint \
  --served-model-name openvistool_model --tensor-parallel-size 4 \
  --max-model-len 65536 --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml

export TEST_MODEL_ID=openvistool_model
export TEST_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export TEST_MODEL_API_KEY=EMPTY
```

## Run the five domains

Chart, Table, and Visual Search use answer extraction followed by an LLM judge:

```bash
./eval.sh --general-config configs/general_judge.json \
  --model-config configs/model.json --bench dataset/data/chart.jsonl
./eval.sh --general-config configs/general_judge.json \
  --model-config configs/model.json --bench dataset/data/table.jsonl
./eval.sh --general-config configs/general_judge.json \
  --model-config configs/model.json --bench dataset/data/visual_search.jsonl
```

GUI Grounding is scored deterministically by point-in-bounding-box and does
not require a judge endpoint:

```bash
cp configs/general_gui.example.json configs/general_gui.json
./eval.sh --general-config configs/general_gui.json \
  --model-config configs/model.json
```

Web-to-HTML renders the submitted page at the three target viewports and uses
the configured VLM judge:

```bash
cp configs/general_vision2web.example.json configs/general_vision2web.json
./eval.sh --general-config configs/general_vision2web.json \
  --model-config configs/model.json
```

For comparable visual scores, keep the locked Playwright version, its matching
Chromium build, installed fonts, viewport settings, and device scale factor
fixed across runs. Record the container image or OS/font manifest with the
reported result.

Use `--resume <run_dir>` to resume interrupted jobs and `rejudge.sh` to score
saved trajectories again without rerunning the agent. See `eval.py --help` and
`rejudge.py --help` for all overrides.
