"""Shared utilities for vision tools."""

from __future__ import annotations

import base64
import io
import re
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont

from nanobot.agent.tools.filesystem import _resolve_path

IMAGE_PATH_DESC = "Image path."

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in _FONT_PATHS:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def image_to_content_blocks(
    img: Image.Image,
    text: str | None = None,
    save_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert a PIL Image to OpenAI-compatible multimodal content blocks."""
    buf = io.BytesIO()
    fmt = "PNG" if img.mode == "RGBA" else "JPEG"
    ext = ".png" if fmt == "PNG" else ".jpg"
    img.save(buf, format=fmt, quality=92)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    mime = f"image/{fmt.lower()}"

    if save_path:
        if save_path.suffix != ext:
            save_path = save_path.with_suffix(ext)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(raw)
        text = f"Edited image saved to: {save_path.resolve()}"

    blocks: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
    if text:
        blocks.append({"type": "text", "text": text})
    return blocks


def load_image(
    image_path: str,
    workspace: Path | None,
    allowed_dir: Path | None,
) -> tuple[Image.Image, Path]:
    file_path = _resolve_path(image_path, workspace, allowed_dir)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {image_path}")
    with open(file_path, "rb") as f:
        img = Image.open(f)
        img.load()
    return img, file_path


def resolve_coord(coord: int | float, size: int) -> int:
    """Convert a 0-1000 relative coordinate to image-space pixels."""
    return round(int(coord) * size // 1000)


_COORD_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def coerce_number_array(value: Any, expected_len: int) -> list[float] | None:
    """Normalize a value to a ``list[float]`` of exactly ``expected_len`` items.

    Teacher models sometimes serialize coordinate arrays as strings
    (``"[100, 200]"``) or as singly-nested arrays (``[[100, 200]]``) even when
    the JSON schema asks for a plain array. Accept those forms so downstream
    tools are robust to model drift.

    Returns ``None`` when the value cannot be coerced.
    """
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]

    if isinstance(value, (list, tuple)) and len(value) == expected_len:
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return coerce_number_array(parsed, expected_len)
        numbers = _COORD_NUMBER_RE.findall(s)
        if len(numbers) == expected_len:
            try:
                return [float(n) for n in numbers]
            except ValueError:
                return None

    return None


def coerce_int_array(value: Any, expected_len: int) -> list[int] | None:
    """Like :func:`coerce_number_array` but returns ``list[int]``."""
    nums = coerce_number_array(value, expected_len)
    if nums is None:
        return None
    try:
        return [int(round(n)) for n in nums]
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    """Coerce a number/string to ``int``; returns ``None`` on failure."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(round(float(s)))
        except ValueError:
            return None
    return None


def resolve_box(box: list[int], width: int, height: int) -> list[int]:
    """Convert [x1, y1, x2, y2] from 0-1000 relative coords to image-space pixels."""
    return [
        resolve_coord(box[0], width),
        resolve_coord(box[1], height),
        resolve_coord(box[2], width),
        resolve_coord(box[3], height),
    ]


def resolve_point(point: list[int], width: int, height: int) -> list[int]:
    """Convert [x, y] from 0-1000 relative coords to image-space pixels."""
    return [
        resolve_coord(point[0], width),
        resolve_coord(point[1], height),
    ]


# Backward-compatible aliases
def rel_to_abs(coord: int, size: int) -> int:
    return resolve_coord(coord, size)


def rel_box_to_abs(box: list[int], width: int, height: int) -> list[int]:
    return resolve_box(box, width, height)


def rel_point_to_abs(point: list[int], width: int, height: int) -> list[int]:
    return resolve_point(point, width, height)


def save_path(workspace: Path | None, source_stem: str, kind: str) -> Path:
    """Build output path in the agent workspace root."""
    if workspace is None:
        raise ValueError("Vision tools require a workspace to save outputs.")
    workspace.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_stem)[:64] or "image"
    return workspace / f"{safe}_{kind}_{uuid.uuid4().hex[:8]}.png"
