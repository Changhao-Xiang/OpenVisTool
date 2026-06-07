"""Geometric transform tools for vision tasks.

Provides tools for cropping, rotating, flipping, and other spatial transformations.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.vision._common import (
    IMAGE_PATH_DESC,
    image_to_content_blocks,
    load_image,
    resolve_box,
    save_path,
    write_crop_metadata,
)


def _detect_bg_color(img: Image.Image) -> tuple[int, int, int]:
    """Detect background color by sampling border pixels."""
    w, h = img.size
    pixels: list[tuple[int, int, int]] = []
    step = max(1, w // 200)
    for x in range(0, w, step):
        pixels.append(img.getpixel((x, 0))[:3])
        pixels.append(img.getpixel((x, h - 1))[:3])
    step = max(1, h // 200)
    for y in range(0, h, step):
        pixels.append(img.getpixel((0, y))[:3])
        pixels.append(img.getpixel((w - 1, y))[:3])
    return Counter(pixels).most_common(1)[0][0]


# Map user-facing names to PIL transpose codes
_FLIP_MODES = {
    "horizontal": Image.Transpose.FLIP_LEFT_RIGHT,
    "vertical": Image.Transpose.FLIP_TOP_BOTTOM,
}


class CropTool(Tool):
    """Crop a rectangular region from an image."""

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "crop"

    @property
    def description(self) -> str:
        return "Crop a rectangular region from an image."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "crop": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Crop rectangle [x1, y1, x2, y2] in 0-1000 relative coords.",
                },
            },
            "required": ["image_path", "crop"],
        }

    async def execute(
        self,
        image_path: str,
        crop: list[int],
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            if len(crop) != 4:
                return "Error: crop must be [x1, y1, x2, y2]"
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            iw, ih = img.size
            x1, y1, x2, y2 = resolve_box(crop, iw, ih)
            cropped = img.crop((x1, y1, x2, y2))
            # crop always yields RGB → image_to_content_blocks saves JPEG (.jpg);
            # pin the suffix here so the sidecar matches the file the model sees.
            out_path = save_path(self._workspace, file_path.stem, "crop").with_suffix(".jpg")
            blocks = image_to_content_blocks(cropped, save_path=out_path)
            # Record the crop transform so locate_in_crop can map a click on the
            # cropped image back to the original image's 0-1000 space.
            write_crop_metadata(out_path, file_path, crop, (iw, ih))
            return blocks
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in crop: {e}"


class RotateTool(Tool):
    """Rotate an image by a given angle in degrees."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "rotate"

    @property
    def description(self) -> str:
        return (
            "Rotate an image counter-clockwise by `angle` degrees. "
            "The canvas auto-expands; new corners are filled with the auto-detected "
            "background color."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "angle": {"type": "number", "description": "Degrees, counter-clockwise."},
            },
            "required": ["image_path", "angle"],
        }

    async def execute(
        self,
        image_path: str,
        angle: float,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            bg = _detect_bg_color(img)
            img = img.rotate(angle, expand=True, fillcolor=bg)
            out_path = save_path(self._workspace, file_path.stem, "rotate")
            return image_to_content_blocks(img, save_path=out_path)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in rotate: {e}"


class FlipTool(Tool):
    """Flip an image horizontally or vertically."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "flip"

    @property
    def description(self) -> str:
        return "Flip an image along the horizontal or vertical axis."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "direction": {
                    "type": "string",
                    "enum": ["horizontal", "vertical"],
                    "description": "'horizontal' = left-right, 'vertical' = top-bottom.",
                },
            },
            "required": ["image_path", "direction"],
        }

    async def execute(
        self,
        image_path: str,
        direction: str,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            mode = _FLIP_MODES.get(direction)
            if mode is None:
                return f"Error: direction must be 'horizontal' or 'vertical', got '{direction}'"
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB").transpose(mode)
            out_path = save_path(self._workspace, file_path.stem, "flip")
            return image_to_content_blocks(img, save_path=out_path)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in flip: {e}"
