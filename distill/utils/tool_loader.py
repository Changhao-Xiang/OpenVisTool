"""Auto-discover nanobot Tool subclasses and extract OpenAI-format schemas.

Replaces the static ``tool_registry.json`` approach: tool definitions are
read directly from the Python classes in ``nanobot.agent.tools``, so they
never drift out of sync.

Usage in convert_to_messages.py::

    from distill.utils.tool_loader import load_tools_and_pools

    registry = load_tools_and_pools()   # {"tools": {name: schema}, "pools": {...}}
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from typing import Any

from nanobot.agent.tools.base import Tool

# ---------------------------------------------------------------------------
# Pool definitions (moved from tool_registry.json)
# ---------------------------------------------------------------------------
DEFAULT_POOLS: dict[str, list[str]] = {
    "default": ["read_file", "write_file", "edit_file", "list_dir", "exec"],
    "filesystem": ["read_file", "write_file", "edit_file", "list_dir"],
    "web": ["web_search", "web_fetch"],
    "vision": [
        "draw_bbox",
        "draw_line",
        "draw_circle",
        "crop",
        "rotate",
        "flip",
        "enhance_contrast",
        "adjust_brightness",
        "detect_edges",
        "grayscale",
        "connected_components",
        "find_contours",
        "hough_lines",
        "hough_circles",
        "template_match",
        "computer_use",
    ],
    "full": [],  # resolved dynamically to all discovered tools
}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def discover_tool_classes() -> list[type[Tool]]:
    """Walk ``nanobot.agent.tools`` and collect all concrete Tool subclasses."""
    import nanobot.agent.tools as tools_pkg

    # Import every sub-module so that Tool subclasses are registered.
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        tools_pkg.__path__, prefix=tools_pkg.__name__ + "."
    ):
        try:
            importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001
            print(f"[tool_loader] WARN: failed to import {modname}: {exc}", file=sys.stderr)

    # Collect all concrete (non-abstract) subclasses, deduplicate by identity.
    seen: dict[int, type[Tool]] = {}

    def _collect(cls: type[Tool]) -> None:
        for sub in cls.__subclasses__():
            if id(sub) in seen:
                continue
            if not inspect.isabstract(sub):
                seen[id(sub)] = sub
            _collect(sub)

    _collect(Tool)
    return sorted(seen.values(), key=lambda c: c.__name__)


def _instantiate_tool(cls: type[Tool], **overrides: Any) -> Tool | None:
    """Instantiate a Tool subclass with dummy args (for schema extraction only).

    Extra *overrides* are forwarded to the constructor when it accepts
    matching parameter names.
    """
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in overrides:
            kwargs[name] = overrides[name]
        elif param.default is inspect.Parameter.empty:
            kwargs[name] = None
    try:
        return cls(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[tool_loader] WARN: cannot instantiate {cls.__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def discover_tool_schemas() -> dict[str, dict[str, Any]]:
    """Return ``{tool_name: openai_function_schema}`` for all discoverable tools."""
    schemas: dict[str, dict[str, Any]] = {}
    for cls in discover_tool_classes():
        tool = _instantiate_tool(cls)
        if tool is None:
            continue
        try:
            schema = tool.to_schema()
            schemas[tool.name] = schema
        except Exception as exc:  # noqa: BLE001
            print(
                f"[tool_loader] WARN: to_schema() failed for {cls.__name__}: {exc}",
                file=sys.stderr,
            )
    return schemas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_tools_and_pools(
    extra_pools: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Discover tools and return ``{"tools": {name: schema}, "pools": {...}}``.

    Drop-in replacement for the old ``load_tool_registry()`` that read from
    ``tool_registry.json``.
    """
    tools = discover_tool_schemas()

    pools = {k: list(v) for k, v in DEFAULT_POOLS.items()}
    pools["full"] = sorted(tools.keys())

    if extra_pools:
        pools.update(extra_pools)

    return {"tools": tools, "pools": pools}
