"""Color analysis tools for vision tasks.

Provides tools for sampling colors, finding dominant colors,
and segmenting connected regions by color match.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label as scipy_label

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.vision._common import (
    IMAGE_PATH_DESC,
    load_image,
    resolve_box,
    resolve_point,
)

# ---------------------------------------------------------------------------
# sample_color
# ---------------------------------------------------------------------------


class SampleColorTool(Tool):
    """Sample the RGB color at a single point."""

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "sample_color"

    @property
    def description(self) -> str:
        return "Return the RGB value at a queried point."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "point": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[x, y] in 0-1000 relative coords.",
                },
            },
            "required": ["image_path", "point"],
        }

    async def execute(
        self,
        image_path: str,
        point: list[int] | None = None,
        xy: list[int] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            pt = point if point is not None else xy
            if pt is None:
                return "Error: 'point' is required"
            img, _ = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            w, h = img.size
            x, y = resolve_point(pt, w, h)
            if not (0 <= x < w and 0 <= y < h):
                return "Error: point is outside image bounds"
            r, g, b = img.getpixel((x, y))[:3]
            return json.dumps({"rgb": [r, g, b]})
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in sample_color: {e}"


# ---------------------------------------------------------------------------
# color_clusters
# ---------------------------------------------------------------------------


class ColorClustersTool(Tool):
    """Find the top-K dominant colors in a region by area."""

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "color_clusters"

    @property
    def description(self) -> str:
        return (
            "Return the top-K dominant colors in a region, ranked by relative area. "
            "Near-identical shades are merged via 8-step per-channel quantization. "
            "Useful for chart series/legends and pie-chart sectors."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "ROI [x1, y1, x2, y2] in 0-1000 relative coords. Defaults to full image.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of top colors to return. Default 5.",
                },
            },
            "required": ["image_path"],
        }

    async def execute(
        self,
        image_path: str,
        region: list[int] | None = None,
        k: int = 5,
        **kwargs: Any,
    ) -> str:
        try:
            img, _ = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")

            if region and len(region) == 4:
                x1, y1, x2, y2 = resolve_box(region, img.size[0], img.size[1])
                img = img.crop((x1, y1, x2, y2))

            pixels = np.asarray(img).reshape(-1, 3)
            quantized = (pixels // 8 * 8).astype(np.uint8)
            counts: Counter = Counter(map(tuple, quantized.tolist()))
            total = len(pixels)
            top = counts.most_common(k)
            colors = [{"rgb": list(c), "percentage": f"{n / total * 100:.1f}%"} for c, n in top]
            return json.dumps({"colors": colors})
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in color_clusters: {e}"


# ---------------------------------------------------------------------------
# color_segments
# ---------------------------------------------------------------------------


class ColorSegmentsTool(Tool):
    """Find and measure color-matched connected regions in an image."""

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "color_segments"

    @property
    def description(self) -> str:
        return (
            "Find connected regions matching an RGB color, returning segment count " "and relative coverage."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "rgb": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Target color [R, G, B], 0-255.",
                },
                "tol": {
                    "type": "integer",
                    "description": "RGB distance tolerance. Default 30.",
                },
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional ROI [x1, y1, x2, y2] in 0-1000 relative coords.",
                },
                "min_area": {
                    "type": "number",
                    "description": (
                        "Drop segments smaller than this fraction of the ROI area in [0, 1]. "
                        "Example: 0.01 = 1% of the region. Default matches a very small "
                        "minimum area relative to the ROI size."
                    ),
                },
            },
            "required": ["image_path", "rgb"],
        }

    async def execute(
        self,
        image_path: str,
        rgb: list[int],
        tol: int = 30,
        region: list[int] | None = None,
        min_area: float | int | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if len(rgb) != 3:
                return "Error: rgb must be [R, G, B]"
            tr, tg, tb = rgb

            img, _ = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")

            if region and len(region) == 4:
                x1, y1, x2, y2 = resolve_box(region, img.size[0], img.size[1])
                roi = img.crop((x1, y1, x2, y2))
                off_x, off_y = x1, y1
            else:
                roi = img
                off_x, off_y = 0, 0

            # Vectorised colour matching
            arr = np.asarray(roi, dtype=np.float32)
            target = np.array([tr, tg, tb], dtype=np.float32)
            dist = np.sqrt(np.sum((arr - target) ** 2, axis=2))
            mask = dist <= tol

            total_matched = int(mask.sum())
            if total_matched == 0:
                return (
                    f"No pixels matching RGB({tr},{tg},{tb}) within tolerance {tol} "
                    f"were found{f' in region {region}' if region else ''}."
                )

            # Connected component labeling
            labels, num_raw = scipy_label(mask)

            # Per-segment stats, filtered by min_area
            total_px = roi.size[0] * roi.size[1]
            default_ratio = min(1.0, 10 / total_px) if total_px else 1.0
            min_area_ratio = default_ratio if min_area is None else float(min_area)
            if not 0 <= min_area_ratio <= 1:
                return "Error: min_area must be between 0 and 1"
            min_area_px = int(np.ceil(min_area_ratio * total_px))

            segments: list[dict[str, Any]] = []
            for seg_id in range(1, num_raw + 1):
                seg_mask = labels == seg_id
                count = int(seg_mask.sum())
                if count < min_area_px:
                    continue

                ys, xs = np.where(seg_mask)
                min_x = int(xs.min()) + off_x
                min_y = int(ys.min()) + off_y
                segments.append(
                    {
                        "id": len(segments) + 1,
                        "area_ratio": f"{count / total_px * 100:.2f}%",
                        "match_ratio": f"{count / total_matched * 100:.2f}%",
                        "_sort_key": [min_x, min_y],
                    }
                )

            # Sort left-to-right, then top-to-bottom
            segments.sort(key=lambda s: (s["_sort_key"][0], s["_sort_key"][1]))
            for i, seg in enumerate(segments):
                seg["id"] = i + 1
                seg.pop("_sort_key", None)

            results: dict[str, Any] = {
                "color": f"RGB({tr},{tg},{tb})",
                "tolerance": tol,
                "area_ratio": f"{total_matched / total_px * 100:.2f}%",
                "segment_count": len(segments),
                "segments": segments,
            }
            return json.dumps(results, ensure_ascii=False)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in color_segments: {e}"
