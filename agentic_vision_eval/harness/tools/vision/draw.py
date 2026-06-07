"""Drawing and annotation tools for vision tasks.

Three focused tools:
- ``draw_bbox``: rectangular bounding boxes
- ``draw_line``: line segments
- ``draw_circle``: circles (outlined or filled, useful as point markers too)

Each tool accepts a list of shapes so multiple annotations can be drawn on
the same image in one call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import ImageDraw, ImageFont

from harness.tools.base import Tool
from harness.tools.vision._common import (
    IMAGE_PATH_DESC,
    coerce_int,
    coerce_int_array,
    get_font,
    image_to_content_blocks,
    load_image,
    resolve_box,
    resolve_coord,
    resolve_point,
    save_path,
)

_DEFAULT_COLOR = "red"
_DEFAULT_LINE_WIDTH = 2


def _draw_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    pos: tuple[int, int],
    color: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    draw.text(pos, label, fill=color, font=font)


class _DrawToolBase(Tool):
    """Common plumbing for the three draw tools."""

    _kind: str = ""
    _shape_singular: str = "shape"
    _shape_plural: str = "shapes"

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    async def _run(
        self,
        image_path: str,
        items: list[dict],
        renderer,
    ) -> str | list[dict[str, Any]]:
        try:
            if not items:
                return f"Error: at least one {self._shape_singular} is required"
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            font = get_font(14)
            iw, ih = img.size

            for item in items:
                if not isinstance(item, dict):
                    continue
                color = item.get("color", _DEFAULT_COLOR)
                label = item.get("label")
                renderer(draw, item, color, label, font, iw, ih)

            out_path = save_path(self._workspace, file_path.stem, self._kind)
            return image_to_content_blocks(img, save_path=out_path)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Error in {self.name}: {e}"


class DrawBBoxTool(_DrawToolBase):
    """Draw one or more rectangular bounding boxes on an image."""

    _kind = "bbox"
    _shape_singular = "bounding box"
    _shape_plural = "bounding boxes"

    @property
    def name(self) -> str:
        return "draw_bbox"

    @property
    def description(self) -> str:
        return "Draw one or more rectangular bounding boxes on an image."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "boxes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "box": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "[x1, y1, x2, y2] in 0-1000 relative coords.",
                            },
                            "color": {"type": "string", "description": "Color name or hex. Default red."},
                            "label": {"type": "string", "description": "Optional text drawn above the box."},
                        },
                        "required": ["box"],
                    },
                    "description": "Boxes to draw.",
                },
            },
            "required": ["image_path", "boxes"],
        }

    async def execute(
        self,
        image_path: str,
        boxes: list[dict],
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        return await self._run(image_path, boxes, self._render_box)

    @staticmethod
    def _render_box(
        draw: ImageDraw.ImageDraw,
        item: dict,
        color: str,
        label: str | None,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        iw: int,
        ih: int,
    ) -> None:
        box = coerce_int_array(item.get("box"), 4)
        if box is None:
            return
        x1, y1, x2, y2 = resolve_box(box, iw, ih)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=_DEFAULT_LINE_WIDTH)
        if label:
            _draw_label(draw, label, (x1, max(0, y1 - 16)), color, font)


class DrawLineTool(_DrawToolBase):
    """Draw one or more line segments on an image."""

    _kind = "line"
    _shape_singular = "line"
    _shape_plural = "lines"

    @property
    def name(self) -> str:
        return "draw_line"

    @property
    def description(self) -> str:
        return (
            "Draw one or more line segments on an image. "
            "For a full horizontal/vertical guide, use the image edges as endpoints "
            "(e.g. [0, y, 1000, y] on the 0-1000 relative scale)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "lines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "[x1, y1, x2, y2] in 0-1000 relative coords.",
                            },
                            "color": {"type": "string", "description": "Color name or hex. Default red."},
                            "label": {
                                "type": "string",
                                "description": "Optional text drawn at the midpoint.",
                            },
                        },
                        "required": ["line"],
                    },
                    "description": "Lines to draw",
                },
            },
            "required": ["image_path", "lines"],
        }

    async def execute(
        self,
        image_path: str,
        lines: list[dict],
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        return await self._run(image_path, lines, self._render_line)

    @staticmethod
    def _render_line(
        draw: ImageDraw.ImageDraw,
        item: dict,
        color: str,
        label: str | None,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        iw: int,
        ih: int,
    ) -> None:
        line = coerce_int_array(item.get("line"), 4)
        if line is None:
            return
        x1, y1, x2, y2 = resolve_box(line, iw, ih)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=_DEFAULT_LINE_WIDTH)
        if label:
            _draw_label(draw, label, ((x1 + x2) // 2, (y1 + y2) // 2 - 16), color, font)


class DrawCircleTool(_DrawToolBase):
    """Draw one or more circles on an image (outlined or filled)."""

    _kind = "circle"
    _shape_singular = "circle"
    _shape_plural = "circles"

    @property
    def name(self) -> str:
        return "draw_circle"

    @property
    def description(self) -> str:
        return (
            "Draw one or more circles on an image (outlined or filled). "
            "Set `fill: true` to use a circle as a solid point marker."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        radius_hint = "Radius on the 0-1000 scale of min(width, height); e.g. 100 ≈ 10%."
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "circles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "center": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "[cx, cy] in 0-1000 relative coords.",
                            },
                            "radius": {"type": "integer", "description": radius_hint},
                            "color": {"type": "string", "description": "Color name or hex. Default red."},
                            "fill": {
                                "type": "boolean",
                                "description": "Fill the circle as a solid disc. Default false.",
                            },
                            "label": {
                                "type": "string",
                                "description": "Optional text drawn beside the circle.",
                            },
                        },
                        "required": ["center", "radius"],
                    },
                    "description": "Circles to draw.",
                },
            },
            "required": ["image_path", "circles"],
        }

    async def execute(
        self,
        image_path: str,
        circles: list[dict],
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        return await self._run(image_path, circles, self._render_circle)

    @staticmethod
    def _render_circle(
        draw: ImageDraw.ImageDraw,
        item: dict,
        color: str,
        label: str | None,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        iw: int,
        ih: int,
    ) -> None:
        center = coerce_int_array(item.get("center"), 2)
        radius = coerce_int(item.get("radius"))
        if center is None or radius is None:
            return
        cx, cy = resolve_point(center, iw, ih)
        r = max(1, resolve_coord(radius, min(iw, ih)))
        bbox = [cx - r, cy - r, cx + r, cy + r]
        fill = bool(item.get("fill", False))
        if fill:
            draw.ellipse(bbox, fill=color)
        else:
            draw.ellipse(bbox, outline=color, width=_DEFAULT_LINE_WIDTH)
        if label:
            _draw_label(draw, label, (cx + r + 4, cy - 8), color, font)
