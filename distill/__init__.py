"""Helpers for trajectory distillation workflows."""

from .config import build_agent_from_config, load_runtime_config
from .dataset import PreparedItem, load_jsonl_records, prepare_items

__all__ = [
    "PreparedItem",
    "build_agent_from_config",
    "load_jsonl_records",
    "load_runtime_config",
    "prepare_items",
    "run_batch",
]


def __getattr__(name: str):
    """Lazily expose the batch runner without pre-importing ``distill.run``.

    Keeping the executable module unloaded avoids the ``python -m distill.run``
    double-import warning while preserving the small convenience API.
    """
    if name == "run_batch":
        from .run import run_batch

        return run_batch
    raise AttributeError(name)
