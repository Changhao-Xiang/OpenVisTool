"""Color range masking tools for vision tasks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.vision._common import (
    IMAGE_PATH_DESC,
    image_to_content_blocks,
    load_image,
    resolve_box,
    save_path,
)

_COLORSPACES = {"hsv", "bgr"}


class InRangeColorTool(Tool):
    """Create a mask for pixels inside a color range."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "in_range_color"

    @property
    def description(self) -> str:
        return (
            "Create a mask for pixels within a specified color range. Supports HSV "
            "or BGR color spaces. Returns the masked image when pixels match."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "colorspace": {
                    "type": "string",
                    "enum": ["hsv", "bgr"],
                    "description": "'hsv' or 'bgr'. Default hsv. HSV uses OpenCV ranges: H 0-179, S/V 0-255.",
                },
                "lower": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Lower bound [h/b, s/g, v/r].",
                },
                "upper": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Upper bound [h/b, s/g, v/r].",
                },
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional ROI [x1, y1, x2, y2] in 0-1000 relative coords.",
                },
                "min_area": {
                    "type": "number",
                    "description": "Minimum component area as ROI fraction in [0,1] or pixels if >1. Default 1 pixel.",
                },
                "max_components": {
                    "type": "integer",
                    "description": "Maximum connected component bboxes to return. Default 50.",
                },
            },
            "required": ["image_path", "lower", "upper"],
        }

    async def execute(
        self,
        image_path: str,
        lower: list[float | int],
        upper: list[float | int],
        colorspace: str = "hsv",
        region: list[int] | None = None,
        min_area: float | int | None = None,
        max_components: int = 50,
        color_space: str | None = None,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            del kwargs
            colorspace = (color_space or colorspace).lower()
            if colorspace not in _COLORSPACES:
                return "Error: colorspace must be 'hsv' or 'bgr'"
            if len(lower) != 3 or len(upper) != 3:
                return "Error: lower and upper must each contain exactly 3 numbers"

            low = _normalize_bound(lower, colorspace)
            high = _normalize_bound(upper, colorspace)

            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            full_w, full_h = img.size
            off_x, off_y = 0, 0
            if region and len(region) == 4:
                x1, y1, x2, y2 = resolve_box(region, full_w, full_h)
                if x2 <= x1 or y2 <= y1:
                    return "Error: region is empty after conversion to image coordinates"
                img = img.crop((x1, y1, x2, y2))
                off_x, off_y = x1, y1

            rgb = np.asarray(img)
            if colorspace == "hsv":
                converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                mask = _hsv_in_range(converted, low, high)
            else:
                converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                mask = cv2.inRange(converted, low, high)

            total_px = int(mask.size)
            matched_px = int(np.count_nonzero(mask))

            if matched_px == 0:
                return "No pixels matched the specified color range."

            bboxes = _component_bboxes(
                mask,
                full_w=full_w,
                full_h=full_h,
                off_x=off_x,
                off_y=off_y,
                min_area=min_area,
                total_px=total_px,
                max_components=max_components,
            )
            masked_rgb = rgb.copy()
            masked_rgb[mask == 0] = 0
            masked_img = Image.fromarray(masked_rgb)
            out_path = save_path(self._workspace, file_path.stem, "in_range_color")
            bbox_payload = {
                "description": "Bounding boxes for connected components within the specified color range.",
                "colorspace": colorspace,
                "range": {"lower": low.tolist(), "upper": high.tolist()},
                "bboxes": bboxes,
            }
            blocks = [
                {
                    "type": "text",
                    "text": f"{json.dumps(bbox_payload, ensure_ascii=False)}\n",
                }
            ]
            blocks.extend(image_to_content_blocks(masked_img, save_path=out_path))
            return blocks
        except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in in_range_color: {e}"


def _normalize_bound(bound: list[float | int], colorspace: str) -> np.ndarray:
    values = np.array([round(float(v)) for v in bound], dtype=np.int16)
    if colorspace == "hsv":
        limits = np.array([179, 255, 255], dtype=np.int16)
    else:
        limits = np.array([255, 255, 255], dtype=np.int16)
    return np.clip(values, 0, limits).astype(np.uint8)


def _rel_coord(value: int | float, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(1000, round(float(value) * 1000 / size)))


def _component_bboxes(
    mask: np.ndarray,
    *,
    full_w: int,
    full_h: int,
    off_x: int,
    off_y: int,
    min_area: float | int | None,
    total_px: int,
    max_components: int,
) -> list[list[int]]:
    min_area_px = _min_area_px(min_area, total_px)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[tuple[int, int, int, int, int]] = []

    for label_id in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label_id]]
        if area < min_area_px:
            continue
        components.append((x + off_x, y + off_y, w, h, area))

    components.sort(key=lambda item: (item[0], item[1]))
    limit = max(1, int(max_components))
    return [
        [
            _rel_coord(x, full_w),
            _rel_coord(y, full_h),
            _rel_coord(x + w, full_w),
            _rel_coord(y + h, full_h),
        ]
        for x, y, w, h, _area in components[:limit]
    ]


def _min_area_px(min_area: float | int | None, total_px: int) -> int:
    if min_area is None:
        return 1
    value = float(min_area)
    if value < 0:
        raise ValueError("min_area must be non-negative")
    if value <= 1:
        return max(1, int(math.ceil(value * total_px)))
    return int(math.ceil(value))


def _hsv_in_range(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if int(lower[0]) <= int(upper[0]):
        return cv2.inRange(hsv, lower, upper)

    # Hue wraps around zero for colors such as red.
    high_hue = np.array([179, upper[1], upper[2]], dtype=np.uint8)
    low_hue = np.array([0, lower[1], lower[2]], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lower, high_hue), cv2.inRange(hsv, low_hue, upper))
