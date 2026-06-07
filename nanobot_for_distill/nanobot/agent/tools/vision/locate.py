"""locate_in_crop: map a click predicted on a cropped image back to the original.

GUI-grounding rationale: a model that emits the final click directly in the
original image's 0-1000 space gets no precision gain from cropping — it still
has to predict the target's location relative to the *whole* screenshot. This
tool lets the model instead predict the click on the zoomed-in ``crop`` output
(where the target is large and coordinates are coarse), and returns the exactly
equivalent point in the original image's 0-1000 space. Predicting on a crop that
covers a fraction ``f`` of the original amplifies coordinate precision by ``1/f``.

The inverse mapping is reconstructed purely from the earlier ``crop`` tool args
(recorded by ``CropTool`` as a ``.cropmeta.json`` sidecar), composing across
chained crops back to the original image. The returned coordinate is what the
model should then emit as the ``computer_use`` ``coordinate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import _resolve_path
from nanobot.agent.tools.vision._common import (
    IMAGE_PATH_DESC,
    coerce_number_array,
    map_crop_point_to_original,
)


class LocateInCropTool(Tool):
    """Map a 0-1000 click on a cropped image to the original image's 0-1000 space."""

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "locate_in_crop"

    @property
    def description(self) -> str:
        return (
            "Convert a click you predicted on a cropped image into the equivalent "
            "coordinate on the ORIGINAL image (both in 0-1000 space). Use this after "
            "`crop` has zoomed into the target: predict the click on the cropped image "
            "(where the target is large and easy to localize), then call this to get the "
            "original-image coordinate. The returned [x, y] is what you must emit as the "
            "final `computer_use` `coordinate` — do not re-estimate it on the full image "
            "yourself. If `image_path` is the original (uncropped) image, the coordinate "
            "is returned unchanged."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": (
                        IMAGE_PATH_DESC
                        + " The cropped image you predicted the click on (a `crop` output)."
                    ),
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[x, y] click position in the CROPPED image's 0-1000 space "
                        "(the target's center as you see it in that crop)."
                    ),
                },
            },
            "required": ["image_path", "coordinate"],
        }

    async def execute(
        self,
        image_path: str,
        coordinate: Any,
        **kwargs: Any,
    ) -> str:
        try:
            point = coerce_number_array(coordinate, 2)
            if point is None:
                return "Error: coordinate must be [x, y] in the cropped image's 0-1000 space."
            x, y = point

            file_path = _resolve_path(image_path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {image_path}"

            mapped = map_crop_point_to_original(file_path, x, y)
            if mapped is None:
                # No crop metadata: image_path is the original image, so the
                # coordinate is already in original 0-1000 space.
                ox, oy = max(0, min(1000, int(round(x)))), max(0, min(1000, int(round(y))))
                return (
                    f"Original-image coordinate (0-1000): [{ox}, {oy}]. "
                    "(image_path is the original image; coordinate returned unchanged.) "
                    "Emit this as the `computer_use` `coordinate`."
                )

            ox, oy = mapped
            return (
                f"Original-image coordinate (0-1000): [{ox}, {oy}]. "
                "Emit this exact value as the `computer_use` `coordinate`."
            )
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in locate_in_crop: {e}"
