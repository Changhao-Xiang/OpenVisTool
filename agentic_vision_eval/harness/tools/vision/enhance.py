"""Image enhancement tools for vision tasks.

Provides tools for contrast enhancement, edge detection, and grayscale/binarization
to help the model better analyse low-quality or complex images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from harness.tools.base import Tool
from harness.tools.vision._common import (
    IMAGE_PATH_DESC,
    image_to_content_blocks,
    load_image,
    save_path,
)


class EnhanceContrastTool(Tool):
    """Enhance image contrast with histogram equalization."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "enhance_contrast"

    @property
    def description(self) -> str:
        return (
            "Enhance contrast with OpenCV histogram equalization. "
            "Default lab mode applies CLAHE to the LAB lightness channel to preserve colors. "
            "Use grayscale mode for structure/text, hsv to equalize brightness, "
            "or bgr to equalize selected channels."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "color_mode": {
                    "type": "string",
                    "enum": ["lab", "grayscale", "bgr", "hsv"],
                    "description": "Histogram equalization mode. Default lab, using CLAHE on LAB lightness.",
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "BGR channel indices to equalize when color_mode='bgr'. " "Default [0, 1, 2]."
                    ),
                },
            },
            "required": ["image_path"],
        }

    async def execute(
        self,
        image_path: str,
        color_mode: str = "lab",
        channels: list[int] | None = None,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            del kwargs
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            result = _equalize_histogram(cv_img, color_mode=color_mode, channels=channels)
            img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            out_path = save_path(self._workspace, file_path.stem, "contrast")
            return image_to_content_blocks(img, save_path=out_path)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in enhance_contrast: {e}"


class AdjustBrightnessTool(Tool):
    """Adjust image brightness and contrast with linear scaling."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "adjust_brightness"

    @property
    def description(self) -> str:
        return (
            "Adjust image brightness and contrast with OpenCV-style linear scaling: "
            "output = saturate_uint8(abs(image * alpha + beta)). "
            "Increase beta to brighten, decrease beta to darken, and change alpha for contrast."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "alpha": {
                    "type": "number",
                    "description": "Positive scaling factor. Default 1.0.",
                },
                "beta": {
                    "type": "number",
                    "description": "Offset added after scaling. Default 0.",
                },
            },
            "required": ["image_path"],
        }

    async def execute(
        self,
        image_path: str,
        alpha: float = 1.0,
        beta: float = 0.0,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            del kwargs
            alpha = float(alpha)
            beta = float(beta)
            if alpha <= 0:
                return "Error: alpha must be positive"

            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float32)
            scaled = cv2.convertScaleAbs(arr * alpha + beta)
            out = Image.fromarray(scaled, mode="RGB")
            out_path = save_path(self._workspace, file_path.stem, "brightness")
            return image_to_content_blocks(out, save_path=out_path)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in adjust_brightness: {e}"


def _equalize_histogram(
    cv_img: np.ndarray,
    color_mode: str = "lab",
    channels: list[int] | None = None,
) -> np.ndarray:
    """Apply OpenCV histogram equalization using VTC-Bench's equalize modes."""
    color_mode = (color_mode or "lab").lower()
    if color_mode == "lab":
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    if color_mode == "grayscale":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    if color_mode == "bgr":
        selected_channels = channels if channels is not None else [0, 1, 2]
        normalized_channels = _normalize_bgr_channels(selected_channels)
        result = cv_img.copy()
        for channel in normalized_channels:
            result[:, :, channel] = cv2.equalizeHist(cv_img[:, :, channel])
        return result

    if color_mode == "hsv":
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    raise ValueError("color_mode must be 'lab', 'grayscale', 'bgr', or 'hsv'")


def _normalize_bgr_channels(channels: list[int]) -> list[int]:
    normalized: list[int] = []
    for channel in channels:
        try:
            channel_int = int(channel)
        except (TypeError, ValueError) as exc:
            raise ValueError("channels must contain BGR channel indices 0, 1, or 2") from exc
        if channel_int not in (0, 1, 2):
            raise ValueError("channels must contain BGR channel indices 0, 1, or 2")
        if channel_int not in normalized:
            normalized.append(channel_int)
    if not normalized:
        raise ValueError("channels must include at least one BGR channel index")
    return normalized


class DetectEdgesTool(Tool):
    """Detect edges in an image using Sobel operator."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "detect_edges"

    @property
    def description(self) -> str:
        return (
            "Run a Sobel edge detector and return a grayscale edge map. "
            "Useful for revealing object outlines, text boundaries, and structural shapes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
            },
            "required": ["image_path"],
        }

    async def execute(self, image_path: str, **kwargs: Any) -> str | list[dict[str, Any]]:
        try:
            from scipy.ndimage import sobel

            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            gray = np.asarray(img.convert("L"), dtype=np.float64)
            gx = sobel(gray, axis=1)
            gy = sobel(gray, axis=0)
            magnitude = np.sqrt(gx**2 + gy**2)
            if magnitude.max() > 0:
                magnitude = magnitude / magnitude.max() * 255
            edge_img = Image.fromarray(magnitude.astype(np.uint8), mode="L")
            out_path = save_path(self._workspace, file_path.stem, "edges")
            return image_to_content_blocks(edge_img, save_path=out_path)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in detect_edges: {e}"


class GrayscaleTool(Tool):
    """Convert an image to grayscale or black-and-white binary."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "grayscale"

    @property
    def description(self) -> str:
        return (
            "Convert an image to grayscale, or binarize it to separate foreground from "
            "background. Binarization is especially useful for documents and charts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "binary": {
                    "type": "boolean",
                    "description": "Binarize to black-and-white. Default false.",
                },
                "threshold": {
                    "type": "integer",
                    "description": "Threshold 0-255 (only used when binary=true). Omit for automatic Otsu.",
                },
            },
            "required": ["image_path"],
        }

    async def execute(
        self,
        image_path: str,
        binary: bool = False,
        threshold: int | None = None,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            gray = img.convert("L")

            if binary:
                arr = np.asarray(gray, dtype=np.uint8)
                if threshold is None:
                    threshold = _otsu_threshold(arr)
                else:
                    threshold = max(0, min(255, threshold))
                out = Image.fromarray(((arr > threshold) * 255).astype(np.uint8), mode="L")
            else:
                out = gray

            out_path = save_path(self._workspace, file_path.stem, "gray")
            return image_to_content_blocks(out, save_path=out_path)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in grayscale: {e}"


def _otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu's optimal binarization threshold."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)

    sum_bg, weight_bg = 0.0, 0
    best_thresh, best_var = 0, 0.0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var = var
            best_thresh = t

    return best_thresh
