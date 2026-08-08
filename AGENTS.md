# OpenVisTool repository guide

## Project background

OpenVisTool is a data construction, training, and evaluation pipeline for
tool-augmented visual agents. It rolls out a multimodal teacher agent, filters
the resulting trajectories by answer correctness and tool gain, converts the
selected traces to supervised fine-tuning data, trains open multimodal models,
and evaluates them on OpenVisTool-Bench.

The large datasets and model weights are distributed separately through the
[OpenVisTool Hugging Face collection](https://huggingface.co/collections/LockOnN/openvistool).
The repository contains code and configuration templates rather than dataset
media, checkpoints, or third-party training frameworks.

## Pipeline overview

```text
input tasks and media
  -> no-tool difficulty estimation
  -> teacher-agent tool-use rollout
  -> correctness and trajectory-quality filtering
  -> tool-gain estimation and selection
  -> ms-swift SFT conversion and training
  -> OpenVisTool-Bench evaluation
```

## Code framework

- `distill/`: trajectory construction, filtering, selection, conversion, and
  data-analysis utilities.
- `nanobot/`: the lightweight agent runtime used by data construction,
  including providers, tool registration, session handling, workspace path
  mapping, and visual tools.
- `training/`: parameterized Megatron SFT launchers for ms-swift. The ms-swift
  source tree is an external dependency and is not vendored here.
- `evaluation/`: a standalone OpenVisTool-Bench agent harness with domain
  scoring for Chart, Table, Visual Search, GUI Grounding, and Web-to-HTML.
- `configs/`: credential-free example configuration for trajectory rollout.

The root Python package requires Python 3.11 or newer and covers `distill/`
and `nanobot/`. `scripts/setup_env.sh` can install the root package, evaluation
harness, and an external ms-swift checkout into one environment. The evaluation
harness also retains its own `evaluation/pyproject.toml` and
`evaluation/uv.lock` for standalone use. Playwright and Chromium are an
optional Web-to-HTML setup handled by `scripts/setup_playwright.sh`.

Generated sessions and task files are isolated in per-item workspaces. Paths
shown to the agent use the virtual `/mnt/data` prefix and are translated to
real workspace paths by the path-mapping layer.
