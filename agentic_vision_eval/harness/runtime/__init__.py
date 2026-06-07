"""Run-time bookkeeping — sandboxes, result aggregation, the live dashboard.

- ``sandbox``    per-task/-sample sandbox dirs + tool registry construction
- ``results``    walk task_*/result.json -> results.jsonl + aggregate stats
- ``dashboard``  rich.Live per-task progress table (no-op when not a TTY)
"""

from .sandbox import (
    build_registry, setup_sandbox, setup_sample_sandbox, WORKSPACE_TOOL_CLASSES,
)
from .results import aggregate_results, failed_ids, print_summary
from .dashboard import EvalDashboard

__all__ = [
    "build_registry", "setup_sandbox", "setup_sample_sandbox",
    "WORKSPACE_TOOL_CLASSES",
    "aggregate_results", "failed_ids", "print_summary",
    "EvalDashboard",
]
