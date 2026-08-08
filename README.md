# OpenVisTool

OpenVisTool is a data and training pipeline for tool-augmented visual agents.
It rolls out a teacher agent on multimodal tasks, filters trajectories by
answer correctness and measurable tool gain, converts the retained traces to
SFT format, trains open multimodal models, and evaluates them on a five-domain
benchmark.

[[Hugging Face collection](https://huggingface.co/collections/LockOnN/openvistool)]
[[arXiv (coming soon)](https://arxiv.org/abs/XXXX.XXXXX)]

This source release accompanies two separately distributed datasets:

- [**OpenVisTool-42K**](https://huggingface.co/datasets/LockOnN/OpenVisTool-42K):
  42,048 SFT trajectories with interleaved reasoning, tool calls, tool
  observations, and final answers.
- [**OpenVisTool-Bench**](https://huggingface.co/datasets/LockOnN/OpenVisTool-Bench):
  559 evaluation examples spanning Chart, Table, Visual Search, GUI Grounding,
  and Web-to-HTML.

## Repository layout

```text
distill/       teacher rollout, filtering, tool-gain selection, SFT conversion
nanobot/       lightweight agent loop and visual tool implementations
training/      reproducible ms-swift Megatron SFT launchers
evaluation/    OpenVisTool-Bench agent harness and scoring
configs/       credential-free construction templates
```

Large datasets, model checkpoints, generated workspaces, ms-swift, and
flash-attention are intentionally not vendored.

This is a construction-focused nanobot subset. Personal-assistant chat
channels, messaging bridges, heartbeat/cron services, and bundled end-user
skills are intentionally outside the public release surface. The released
root configuration accepts only the documented construction fields and rejects
unsupported extra keys.

## Installation

Python 3.11 or newer and an ms-swift source checkout are required. Megatron
also requires a CUDA-matched PyTorch, Transformer Engine, and FlashAttention
stack. The most reliable setup is to reuse a training environment that has
already run Megatron successfully, then install construction and evaluation
dependencies into that same environment:

```bash
python -m pip install -U uv
git clone https://github.com/modelscope/ms-swift.git /path/to/ms-swift
conda activate swift-fa3  # or another verified Megatron environment
VENV_PATH="$CONDA_PREFIX" SWIFT_HOME=/path/to/ms-swift bash scripts/setup_env.sh
```

The verified H800 environment used Python 3.12, PyTorch `2.10.0+cu128`,
Transformers `5.3.0`, Transformer Engine `2.13.0`, FlashAttention `2.8.3`,
Megatron-Core `0.16.1`, and mcore-bridge `1.1.1`. The experiments used
ms-swift `4.2.0.dev0`. `scripts/setup_env.sh` preserves a compatible installed
GPU stack and validates the Megatron entry point before reporting success.

To bootstrap a fresh conda environment, install a CUDA-matched PyTorch build
first, then follow the Transformer Engine and FlashAttention commands in the
ms-swift Megatron quick start before running `scripts/setup_env.sh`.

### Web-to-HTML rendering

The regular environment does not install a browser. Web-to-HTML construction
and evaluation additionally require Playwright and its matching Chromium
runtime:

```bash
VENV_PATH="${CONDA_PREFIX:-.venv}" bash scripts/setup_playwright.sh
```

## 1. Construct trajectories

Data construction lives under `distill/`:

- `distill/run.py` runs the teacher agent and saves one isolated session per
  input item;
- `distill/filter/` contains the difficulty, correctness, trajectory-quality,
  and tool-gain evaluators;
- `distill/select/` selects and intersects the retained example indices;
- `distill/utils/convert_to_swift.py` converts selected sessions to ms-swift
  agent SFT JSONL.

The rollout configuration template is `configs/distill.example.json`. See
[distill/README.md](distill/README.md) for the script interfaces and example
commands.

Copy the credential-free config and point its `custom` provider at an
OpenAI-compatible teacher endpoint. Keep the local config untracked.

```bash
cp configs/distill.example.json configs/distill.local.json
python distill/run.py /path/to/input.jsonl \
  --query-field query \
  --media-field images \
  --media-dir /path/to/images \
  --config configs/distill.local.json \
  --custom-instructions-override \
    distill/run/task_specific_toolcall_instructions/chart.md \
  --num-workers 8
```

### Simulated sandbox paths

Each rollout uses a separate workspace on the host. The agent does not receive
that host path: input media and generated artifacts are exposed through
simulated absolute paths such as `/mnt/data/image.png` and
`/mnt/data/result.json`. The harness maps these virtual paths to files inside
the current session workspace when tools execute, and converts real workspace
paths back to `/mnt/data/...` before they are shown to the model or persisted
in a trajectory. As a result, released traces use stable sandbox-style paths
without embedding source-machine workspace locations.

Teacher-generated code is untrusted. Run rollouts inside a disposable,
network-restricted container with only the task media mounted. The released
shell tools pass an environment allowlist to subprocesses and do not expose
API keys, tokens, passwords, or cookies; do not weaken that boundary in a
credential-bearing host process.

## 2. Train models

Training entry points live under `training/`. `training/train_megatron.sh` is
the shared ms-swift Megatron launcher, while `training/train_qwen35_4b.sh`,
`training/train_qwen35_9b.sh`, `training/train_qwen35_27b.sh`, and
`training/train_qwen3vl_8b.sh` provide model-specific defaults. See
[training/README.md](training/README.md) for setup and launcher usage.

The paper experiments use
[modelscope/ms-swift](https://github.com/modelscope/ms-swift) version
`4.2.0.dev0`. Clone and install ms-swift, download the public dataset, then
launch one of the provided wrappers:

```bash
git clone https://github.com/modelscope/ms-swift.git /path/to/ms-swift

python -m pip install -U huggingface_hub
hf download LockOnN/OpenVisTool-42K \
  --repo-type dataset --local-dir /path/to/OpenVisTool-42K

export SWIFT_HOME=/path/to/ms-swift
export DATA_ROOT=/path/to/OpenVisTool-42K
bash training/train_qwen35_9b.sh
```

## 3. Evaluate on OpenVisTool-Bench

Evaluation lives under `evaluation/`. `evaluation/eval.py` and
`evaluation/eval.sh` run or resume benchmark jobs, `evaluation/rejudge.py` and
`evaluation/rejudge.sh` rescore saved trajectories, and `evaluation/harness/`
contains the agents, tools, configuration loader, and domain scorers. See
[evaluation/README.md](evaluation/README.md) for the five domain-specific
commands.

```bash
python -m pip install -U huggingface_hub
hf download LockOnN/OpenVisTool-Bench \
  --repo-type dataset --local-dir /path/to/OpenVisTool-Bench

cd evaluation
ln -s /path/to/OpenVisTool-Bench dataset
cp configs/model.example.json configs/model.json
cp configs/general_judge.example.json configs/general_judge.json
```

Set the endpoint variables listed in
[evaluation/README.md](evaluation/README.md), then run:

```bash
./eval.sh --general-config configs/general_judge.json \
  --model-config configs/model.json --bench dataset/data/chart.jsonl
```

For local evaluation, serve the checkpoint through vLLM. Qwen3/Qwen3.5 tool
calling requires `--enable-auto-tool-choice --tool-call-parser qwen3_xml`, and
the configured model ID must match `--served-model-name`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/checkpoint \
  --served-model-name openvistool_model --tensor-parallel-size 4 \
  --max-model-len 65536 --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml

export TEST_MODEL_ID=openvistool_model
export TEST_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export TEST_MODEL_API_KEY=EMPTY
```

The same harness supports deterministic GUI grounding and rendered
Web-to-HTML scoring. Domain-specific commands and resume/rejudge behavior are
documented in the evaluation README.

## Dataset schema

An OpenVisTool-42K row contains:

- `id`: stable release identifier;
- `domain`: one of the five training domains;
- `tools`: serialized OpenAI function schemas used by the trajectory;
- `messages`: ordered system/user/assistant/tool-call/tool-response turns;
- `images`: paths relative to the downloaded dataset snapshot root.

Paths under `/mnt/data` inside messages describe the per-sample agent sandbox;
they are not host filesystem paths. No source host paths or credentials are
part of the public data.

## Responsible release and licensing

Code in this repository is released under the MIT License. Dataset and
benchmark provenance, licensing, and source-specific terms are documented in
their corresponding Hugging Face dataset cards.

## Acknowledgements

OpenVisTool is built on [HKUDS/nanobot](https://github.com/HKUDS/nanobot); we
thank its authors and contributors for releasing their work. Training uses
[modelscope/ms-swift](https://github.com/modelscope/ms-swift). Third-party
dataset acknowledgements are maintained in the Hugging Face dataset cards.
