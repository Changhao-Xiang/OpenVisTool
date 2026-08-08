"""Per-task sandbox setup + tool registry construction."""

from __future__ import annotations

import shutil
from pathlib import Path

from harness.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from harness.tools.registry import ToolRegistry
from harness.tools.shell import ExecTool
from harness.tools.vision import (
    AdjustBrightnessTool,
    ComputerUseTool,
    ConnectedComponentsTool,
    CropTool,
    DetectEdgesTool,
    DrawBBoxTool,
    DrawCircleTool,
    DrawLineTool,
    EnhanceContrastTool,
    FindContoursTool,
    FlipTool,
    GrayscaleTool,
    HoughCirclesTool,
    HoughLinesTool,
    InRangeColorTool,
    RenderHtmlTool,
    RotateTool,
    TemplateMatchTool,
)

from ..core.benchmark import task_key

# Tools that take an explicit `workspace=` parameter at construction time.
# `ExecTool` is registered separately because its constructor takes
# `working_dir` plus the exec timeout.
WORKSPACE_TOOL_CLASSES = [
    ReadFileTool, WriteFileTool, EditFileTool, ListDirTool,
    CropTool, RotateTool, FlipTool,
    DrawBBoxTool, DrawLineTool, DrawCircleTool,
    EnhanceContrastTool, AdjustBrightnessTool, DetectEdgesTool, GrayscaleTool,
    InRangeColorTool,
    RenderHtmlTool,
    ConnectedComponentsTool, FindContoursTool,
    HoughLinesTool, HoughCirclesTool, TemplateMatchTool,
]


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if src.is_dir():
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _dataset_root_from_item(item: dict) -> Path | None:
    image_paths = item.get("image_paths") or []
    images = item.get("images") or []
    if not image_paths or not images:
        return None

    src_abs = Path(image_paths[0]).resolve()
    rel_parts = Path(images[0]).parts
    if len(src_abs.parts) <= len(rel_parts):
        return None
    return src_abs.parents[len(rel_parts) - 1]


def _link_vision2web_assets(sandbox: Path, item: dict) -> None:
    """Expose Vision2Web resources/workflow at sandbox root when present."""
    root = _dataset_root_from_item(item)
    if root is None:
        return

    resources_rel = item.get("resources_dir")
    if resources_rel:
        resources_src = root / resources_rel
        if resources_src.exists():
            _link_or_copy(resources_src.resolve(), sandbox / "resources")

    workflow_rel = item.get("workflow")
    if workflow_rel:
        workflow_src = root / workflow_rel
        if workflow_src.exists():
            _link_or_copy(workflow_src.resolve(), sandbox / "workflow.json")


def _link_sample_assets(task_sandbox: Path, sample: Path) -> None:
    for name in ("resources", "workflow.json"):
        src = task_sandbox / name
        if src.exists():
            _link_or_copy(src.resolve(), sample / name)


def setup_sandbox(run_dir: Path, item: dict) -> tuple[Path, list[str]]:
    """Create the per-task sandbox dir and link in input images.

    Returns (sandbox_path, [resolved_image_paths]).
    """
    sandbox = (run_dir / f"task_{task_key(item)}").resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for src in item.get("image_paths", []):
        src_abs = Path(src).resolve()
        dst = sandbox / src_abs.name
        if not dst.exists():
            try:
                dst.symlink_to(src_abs)
            except OSError:
                shutil.copy2(src_abs, dst)
        paths.append(str(dst))
    _link_vision2web_assets(sandbox, item)
    return sandbox, paths


def setup_sample_sandbox(task_sandbox: Path, item: dict,
                         sample_idx: int) -> tuple[Path, list[str]]:
    """Create a per-sample subdir `sample_<j>/` under the task sandbox and
    link in the input images. Used for avg@k > 1 so each rollout has an
    isolated working directory for tool artifacts.

    Returns (sample_sandbox_path, [resolved_image_paths]).
    """
    sample = (task_sandbox / f"sample_{sample_idx}").resolve()
    sample.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for src in item.get("image_paths", []):
        src_abs = Path(src).resolve()
        dst = sample / src_abs.name
        if not dst.exists():
            try:
                dst.symlink_to(src_abs)
            except OSError:
                shutil.copy2(src_abs, dst)
        paths.append(str(dst))
    _link_sample_assets(task_sandbox, sample)
    return sample, paths


def build_registry(sandbox: Path, exec_timeout: int,
                   include_computer_use: bool = False,
                   allowed_tools: list[str] | None = None) -> ToolRegistry:
    """Build a ToolRegistry rooted at `sandbox`.

    All workspace-aware tools share the same sandbox dir; ExecTool is
    confined to it via `restrict_to_workspace=True`.

    When `allowed_tools` is given, only the named tools are registered
    (computer_use is still controlled by `include_computer_use`, since it is
    the GUI grounding answer channel rather than an inspection tool).
    """
    allow = set(allowed_tools) if allowed_tools is not None else None
    reg = ToolRegistry()
    for cls in WORKSPACE_TOOL_CLASSES:
        tool = cls(workspace=sandbox, allowed_dir=sandbox)
        if allow is None or tool.name in allow:
            reg.register(tool)
    if include_computer_use:
        reg.register(ComputerUseTool())
    if allow is None or "exec" in allow:
        reg.register(ExecTool(
            working_dir=str(sandbox),
            restrict_to_workspace=True,
            timeout=exec_timeout,
        ))
    return reg
