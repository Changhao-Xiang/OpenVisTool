# Training

The released launcher reproduces the Megatron SFT setup used for the main
OpenVisTool-42K experiments without vendoring a full training framework.

1. Clone [modelscope/ms-swift](https://github.com/modelscope/ms-swift).
   The experiments used ms-swift version `4.2.0.dev0`.
2. Activate a CUDA environment that has already run Megatron successfully.
   The verified H800 stack is documented in the repository README.
3. Install all OpenVisTool components into that environment with
   `VENV_PATH="$CONDA_PREFIX" SWIFT_HOME=/path/to/ms-swift bash scripts/setup_env.sh`.
4. Download OpenVisTool-42K and keep its `data/` and `images/` directories
   together.
5. Set `SWIFT_HOME` and `DATA_ROOT`, then run one model wrapper.

```bash
git clone https://github.com/modelscope/ms-swift.git /path/to/ms-swift

export SWIFT_HOME=/path/to/ms-swift
export DATA_ROOT=/path/to/OpenVisTool-42K
NPROC_PER_NODE=4 bash training/train_qwen35_9b.sh  # four-GPU host
```

The wrappers encode the tensor-parallel and micro-batch settings used in the
paper. All cluster-dependent values can be overridden with environment
variables. The 27B default is four 8-GPU nodes; set `NNODES`, `NODE_RANK`,
`MASTER_ADDR`, and `MASTER_PORT` in the launcher environment.
Set `TRAIN_ITERS=1` for a one-step smoke test; when set, it overrides
`NUM_TRAIN_EPOCHS` without changing the normal three-epoch default.

Weights & Biases is opt-in. To enable it, provide the credential through the
normal W&B environment and set `REPORT_TO="tensorboard wandb"`. No credential
is stored by these scripts.

The shared training arguments are defined in `train_megatron.sh`; each model
wrapper records its model, topology, tensor-parallel, and micro-batch defaults.
