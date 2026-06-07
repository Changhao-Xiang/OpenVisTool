"""Encoding & serialization helpers for eval I/O.

- encode_image:   file path → base64 data URI (for sending to LLM)
- build_content:  benchmark item → multimodal content blocks
- strip_base64:   recursively replace base64 image bodies with byte-count
                  placeholders so traces stay human-readable on disk
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .path_mapper import PathMapper


def encode_image(path: str) -> str:
    """Read an image file and return a base64 data URI."""
    ext = Path(path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext or 'png'}"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def build_content(item: dict, script_image_paths: list[str] | None = None,
                  workspace: Path | None = None) -> list[dict]:
    """Build multimodal content blocks (images first, then text query) for a bench item.

    If `script_image_paths` is provided, prepend a `[Image file paths — use these
    in scripts]` section listing those paths so the model knows the exact paths
    to reference when writing tool/shell scripts (e.g. inside the sandbox).
    When `workspace` is provided, those paths are virtualised to ``/mnt/data``
    before being shown to the model.
    """
    blocks: list[dict] = [
        {"type": "image_url", "image_url": {"url": encode_image(p)}}
        for p in item["image_paths"]
    ]
    if script_image_paths:
        if workspace is not None:
            mapper = PathMapper(workspace)
            shown = [mapper.to_virtual(p) for p in script_image_paths]
        else:
            shown = list(script_image_paths)
        paths_info = "\n".join(f"- {p}" for p in shown)
        text = f"[Image file paths — use these in scripts]\n{paths_info}\n\n{item['query']}"
    else:
        text = item["query"]
    blocks.append({"type": "text", "text": text})
    return blocks


def strip_base64(obj: Any) -> Any:
    """Recursively walk dicts/lists and replace any base64 image_url body
    with a `<NN bytes omitted>` placeholder so traces stay readable."""
    if isinstance(obj, dict):
        if obj.get("type") == "image_url" and isinstance(obj.get("image_url"), dict):
            url = obj["image_url"].get("url", "")
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                header, _, b64 = url.partition(",")
                return {**obj, "image_url": {"url": f"{header},<{len(b64)} bytes omitted>"}}
        return {k: strip_base64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_base64(x) for x in obj]
    return obj
