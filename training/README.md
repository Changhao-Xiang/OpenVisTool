# Training

The released launcher reproduces the Megatron SFT setup used for the main
OpenVisTool-42K experiments without vendoring a full training framework.

1. Create and activate the `openvistool` environment documented in the
   repository README, then run `bash scripts/setup_env.sh` from the repository
   root.
2. Following the official [PyTorch](https://pytorch.org/get-started/locally/)
   and [ms-swift](https://github.com/modelscope/ms-swift) instructions, install
   the training stack for the machine's CUDA version in that environment.
   Transformer Engine and FlashAttention installation links and the versions
   used by our smoke test are listed in the root README.
3. Keep an ms-swift source checkout available as `SWIFT_HOME`. The experiments
   used commit `bd9cbb08838e8709f6b363ee8d425a4a987c6477`
   (`4.2.0.dev0`).
4. Download OpenVisTool-42K and keep its `data/` and `images/` directories
   together.
5. Set `SWIFT_HOME` and `DATA_ROOT`, then run one model wrapper.

```bash
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
Dataset builder metadata defaults to a node-local `/tmp` cache to avoid file
locking problems on network file systems; set `HF_DATASETS_CACHE` to override it.

Weights & Biases is opt-in. To enable it, provide the credential through the
normal W&B environment and set `REPORT_TO="tensorboard wandb"`. No credential
is stored by these scripts.

The shared training arguments are defined in `train_megatron.sh`; each model
wrapper records its model, topology, tensor-parallel, and micro-batch defaults.
