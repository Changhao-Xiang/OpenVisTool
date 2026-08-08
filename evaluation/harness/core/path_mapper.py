"""Virtual path mapping between the real per-sample sandbox and a stable
``/mnt/data`` virtual root that the model sees.

Used by the agent loop to:
- Translate virtual paths in tool arguments to real paths before execution.
- Translate real paths in tool results back to virtual paths before they
  reach the model.

Design goal: the ``tools/`` package stays untouched. All translation happens
at the ``ToolRegistry.execute`` boundary inside ``utils/agent.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

VIRTUAL_ROOT = "/mnt/data"

# Tool argument keys whose value is a path the model expressed in virtual
# space and that needs to be rewritten to a real sandbox path.
_PATH_ARG_KEYS = frozenset({
    "path",
    "image_path",
    "template_path",
    "working_dir",
})

# Tool text arguments that may embed virtual paths inside code or replacement
# text. These need to be rewritten before the underlying tool writes them to
# disk, otherwise a later subprocess will try to open the non-existent
# ``/mnt/data`` path on the host filesystem.
_TEXT_ARG_KEYS_BY_TOOL = {
    "write_file": frozenset({"content"}),
    "edit_file": frozenset({"old_text", "new_text"}),
}


class PathMapper:
    """Bidirectional mapping between a real sandbox dir and ``/mnt/data``."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = Path(workspace).resolve()

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def virtual_root(self) -> str:
        return VIRTUAL_ROOT

    # ------------------------------------------------------------------
    # Real <-> Virtual single-path conversion
    # ------------------------------------------------------------------

    def to_virtual(self, real_path: str | Path) -> str:
        """Real absolute path under workspace -> virtual path. Other paths
        are returned unchanged (as a string).

        Note: we deliberately do NOT call ``Path.resolve()`` here, because
        sandbox image files are symlinks pointing at the dataset source —
        resolving would jump out of the sandbox and the prefix match would
        fail. We only normalize ``..`` segments via ``os.path.normpath``.
        """
        try:
            rp = os.path.normpath(str(real_path))
        except (TypeError, ValueError):
            return str(real_path)

        ws = str(self._workspace)
        if rp == ws:
            return VIRTUAL_ROOT
        if rp.startswith(ws + "/"):
            return VIRTUAL_ROOT + rp[len(ws):]
        return str(real_path)

    def to_real(self, virtual_path: str) -> str:
        """Virtual path (``/mnt/data/...`` or relative) -> real path under
        the sandbox. Absolute paths outside ``/mnt/data`` are returned
        unchanged so the underlying tool's ``allowed_dir`` check still
        applies."""
        if not virtual_path:
            return virtual_path
        if virtual_path == VIRTUAL_ROOT:
            return str(self._workspace)
        if virtual_path.startswith(VIRTUAL_ROOT + "/"):
            rel = virtual_path[len(VIRTUAL_ROOT) + 1:]
            return str(self._workspace / rel) if rel else str(self._workspace)
        # Relative path -> treat as relative to workspace.
        if not virtual_path.startswith("/"):
            return str(self._workspace / virtual_path)
        return virtual_path

    # ------------------------------------------------------------------
    # Bulk text rewriting
    # ------------------------------------------------------------------

    def rewrite_text(self, text: str) -> str:
        """Replace every occurrence of the real workspace prefix with
        ``/mnt/data``. Used on tool results before showing them to the model."""
        if not text:
            return text
        return text.replace(str(self._workspace), VIRTUAL_ROOT)

    def unrewrite_text(self, text: str) -> str:
        """Reverse of :meth:`rewrite_text`. Used on the ``exec`` command
        string, which may embed virtual paths anywhere."""
        if not text:
            return text
        return text.replace(VIRTUAL_ROOT, str(self._workspace))

    # ------------------------------------------------------------------
    # Tool argument / result rewriting (used by the agent loop)
    # ------------------------------------------------------------------

    def rewrite_tool_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Return a new args dict where path-like keys are translated from
        virtual to real, and embedded virtual paths in executable/text payloads
        are rewritten before they reach the underlying tool."""
        if not isinstance(args, dict):
            return args
        rewritten = dict(args)
        for key in _PATH_ARG_KEYS:
            if key in rewritten and isinstance(rewritten[key], str):
                rewritten[key] = self.to_real(rewritten[key])
        if tool_name == "exec" and isinstance(rewritten.get("command"), str):
            rewritten["command"] = self.unrewrite_text(rewritten["command"])
        for key in _TEXT_ARG_KEYS_BY_TOOL.get(tool_name, ()):
            if isinstance(rewritten.get(key), str):
                rewritten[key] = self.unrewrite_text(rewritten[key])
        return rewritten

    def rewrite_tool_result(
        self, result: str | list[dict[str, Any]]
    ) -> str | list[dict[str, Any]]:
        """Return the tool result with all real workspace paths rewritten to
        the virtual root. Image content blocks are passed through unchanged."""
        if isinstance(result, str):
            return self.rewrite_text(result)
        if isinstance(result, list):
            out: list[dict[str, Any]] = []
            for block in result:
                if isinstance(block, dict) and block.get("type") == "text":
                    out.append({**block, "text": self.rewrite_text(block.get("text", ""))})
                else:
                    out.append(block)
            return out
        return result
