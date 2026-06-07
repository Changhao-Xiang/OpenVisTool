"""Structural feature extraction tools for vision tasks."""

from __future__ import annotations

import json
import math
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
    resolve_box,
    resolve_coord,
    save_path,
)


def _rel_coord(value: int | float, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(1000, round(float(value) * 1000 / size)))


def _rel_point(x: int | float, y: int | float, width: int, height: int) -> list[int]:
    return [_rel_coord(x, width), _rel_coord(y, height)]


def _rel_box(
    x: int | float, y: int | float, w: int | float, h: int | float, width: int, height: int
) -> list[int]:
    return [
        _rel_coord(x, width),
        _rel_coord(y, height),
        _rel_coord(float(x) + float(w), width),
        _rel_coord(float(y) + float(h), height),
    ]


def _crop_region(img: Image.Image, region: list[int] | None) -> tuple[Image.Image, int, int]:
    if region and len(region) == 4:
        x1, y1, x2, y2 = resolve_box(region, img.size[0], img.size[1])
        x1 = max(0, min(img.size[0], x1))
        x2 = max(0, min(img.size[0], x2))
        y1 = max(0, min(img.size[1], y1))
        y2 = max(0, min(img.size[1], y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("region is empty after clipping to image bounds")
        return img.crop((x1, y1, x2, y2)), x1, y1
    return img, 0, 0


def _mask_from_image(
    arr_rgb: np.ndarray,
    *,
    threshold: int | None,
    foreground: str,
) -> tuple[np.ndarray, int | None]:
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    if foreground == "nonzero":
        return (gray > 0).astype(np.uint8) * 255, threshold
    if threshold is None:
        threshold_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(threshold_value)
    else:
        threshold = max(0, min(255, int(threshold)))
    if foreground == "light":
        mask = gray > threshold
    else:
        mask = gray <= threshold
    return mask.astype(np.uint8) * 255, threshold


def _area_min_px(min_area: float | int | None, total_px: int) -> int:
    if min_area is None:
        return 1
    value = float(min_area)
    if value < 0:
        raise ValueError("min_area must be non-negative")
    if value <= 1:
        return max(1, int(math.ceil(value * total_px)))
    return int(math.ceil(value))


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _draw_line_label(cv2, image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        image,
        text,
        (max(0, x), max(12, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _image_blocks_with_payload(
    img: Image.Image,
    payload: dict[str, Any],
    *,
    workspace: Path | None,
    source_stem: str,
    kind: str,
) -> list[dict[str, Any]]:
    out_path = save_path(workspace, source_stem, kind)
    blocks = image_to_content_blocks(img, save_path=out_path)
    blocks.append({"type": "text", "text": _json(payload)})
    return blocks


class ConnectedComponentsTool(Tool):
    """Find connected foreground components and return their geometry."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "connected_components"

    @property
    def description(self) -> str:
        return (
            "Find connected foreground regions in a thresholded image and return each "
            "component's bounding box, centroid, and area."
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
                    "description": "Optional ROI [x1, y1, x2, y2] in 0-1000 relative coords.",
                },
                "foreground": {
                    "type": "string",
                    "enum": ["dark", "light", "nonzero"],
                    "description": "Which pixels are foreground after grayscale thresholding. Default dark.",
                },
                "threshold": {
                    "type": "integer",
                    "description": "Grayscale threshold 0-255. Omit for Otsu.",
                },
                "min_area": {
                    "type": "number",
                    "description": "Minimum component area as ROI fraction in [0,1] or pixels if >1.",
                },
                "connectivity": {
                    "type": "integer",
                    "enum": [4, 8],
                    "description": "Pixel connectivity. Default 8.",
                },
                "max_results": {"type": "integer", "description": "Maximum components returned. Default 20."},
            },
            "required": ["image_path"],
        }

    async def execute(
        self,
        image_path: str,
        region: list[int] | None = None,
        foreground: str = "dark",
        threshold: int | None = None,
        min_area: float | int | None = None,
        connectivity: int = 8,
        max_results: int = 20,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            if foreground not in {"dark", "light", "nonzero"}:
                return "Error: foreground must be 'dark', 'light', or 'nonzero'"
            if connectivity not in {4, 8}:
                return "Error: connectivity must be 4 or 8"
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            full_w, full_h = img.size
            roi, off_x, off_y = _crop_region(img, region)
            arr = np.asarray(roi)
            mask, used_threshold = _mask_from_image(arr, threshold=threshold, foreground=foreground)
            total_px = roi.size[0] * roi.size[1]
            min_area_px = _area_min_px(min_area, total_px)

            count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity)
            components: list[dict[str, Any]] = []
            for label_id in range(1, count):
                x, y, w, h, area = [int(v) for v in stats[label_id]]
                if area < min_area_px:
                    continue
                cx, cy = centroids[label_id]
                abs_x = x + off_x
                abs_y = y + off_y
                components.append(
                    {
                        "id": int(label_id),
                        "bbox": _rel_box(abs_x, abs_y, w, h, full_w, full_h),
                        "centroid": _rel_point(float(cx) + off_x, float(cy) + off_y, full_w, full_h),
                        "area_px": area,
                        "area_ratio": f"{area / total_px * 100:.2f}%",
                    }
                )

            components.sort(key=lambda item: item["area_px"], reverse=True)
            max_results = max(1, int(max_results))
            components = components[:max_results]
            payload = {
                "threshold": used_threshold,
                "foreground": foreground,
                "component_count": len(components),
                "components": components,
            }

            draw_img = np.array(img)
            for component in components:
                x1, y1, x2, y2 = resolve_box(component["bbox"], full_w, full_h)
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                _draw_line_label(cv2, draw_img, str(component["id"]), x1, y1 - 4, (255, 0, 0))
            return _image_blocks_with_payload(
                Image.fromarray(draw_img),
                payload,
                workspace=self._workspace,
                source_stem=file_path.stem,
                kind="components",
            )
        except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in connected_components: {e}"


class FindContoursTool(Tool):
    """Find contours and return shape statistics."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "find_contours"

    @property
    def description(self) -> str:
        return (
            "Find contours in a thresholded image and return bounding boxes, area, "
            "perimeter, and polygon approximations."
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
                    "description": "Optional ROI [x1, y1, x2, y2] in 0-1000 relative coords.",
                },
                "foreground": {
                    "type": "string",
                    "enum": ["dark", "light", "nonzero"],
                    "description": "Which pixels are foreground. Default dark.",
                },
                "threshold": {"type": "integer", "description": "Grayscale threshold 0-255. Omit for Otsu."},
                "min_area": {
                    "type": "number",
                    "description": "Minimum contour area as ROI fraction in [0,1] or pixels if >1.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["external", "list", "tree"],
                    "description": "Contour retrieval mode. Default external.",
                },
                "epsilon_ratio": {
                    "type": "number",
                    "description": "Polygon approximation ratio of perimeter. Default 0.01.",
                },
                "max_results": {"type": "integer", "description": "Maximum contours returned. Default 20."},
            },
            "required": ["image_path"],
        }

    async def execute(
        self,
        image_path: str,
        region: list[int] | None = None,
        foreground: str = "dark",
        threshold: int | None = None,
        min_area: float | int | None = None,
        mode: str = "external",
        epsilon_ratio: float = 0.01,
        max_results: int = 20,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            mode_map = {
                "external": cv2.RETR_EXTERNAL,
                "list": cv2.RETR_LIST,
                "tree": cv2.RETR_TREE,
            }
            if mode not in mode_map:
                return "Error: mode must be 'external', 'list', or 'tree'"
            if foreground not in {"dark", "light", "nonzero"}:
                return "Error: foreground must be 'dark', 'light', or 'nonzero'"
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            full_w, full_h = img.size
            roi, off_x, off_y = _crop_region(img, region)
            arr = np.asarray(roi)
            mask, used_threshold = _mask_from_image(arr, threshold=threshold, foreground=foreground)
            total_px = roi.size[0] * roi.size[1]
            min_area_px = _area_min_px(min_area, total_px)

            contours, _hierarchy = cv2.findContours(mask, mode_map[mode], cv2.CHAIN_APPROX_SIMPLE)
            records: list[dict[str, Any]] = []
            for idx, contour in enumerate(contours):
                area = float(cv2.contourArea(contour))
                if area < min_area_px:
                    continue
                x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
                perimeter = float(cv2.arcLength(contour, True))
                epsilon = max(0.0, float(epsilon_ratio)) * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = [
                    _rel_point(int(pt[0][0]) + off_x, int(pt[0][1]) + off_y, full_w, full_h)
                    for pt in approx[:50]
                ]
                records.append(
                    {
                        "id": int(idx + 1),
                        "bbox": _rel_box(x + off_x, y + off_y, w, h, full_w, full_h),
                        "area_px": round(area, 2),
                        "area_ratio": f"{area / total_px * 100:.2f}%",
                        "perimeter_px": round(perimeter, 2),
                        "vertex_count": int(len(approx)),
                        "vertices": vertices,
                    }
                )

            records.sort(key=lambda item: item["area_px"], reverse=True)
            max_results = max(1, int(max_results))
            records = records[:max_results]
            payload = {
                "threshold": used_threshold,
                "foreground": foreground,
                "contour_count": len(records),
                "contours": records,
            }

            draw_img = np.array(img)
            for record in records:
                x1, y1, x2, y2 = resolve_box(record["bbox"], full_w, full_h)
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                _draw_line_label(cv2, draw_img, str(record["id"]), x1, y1 - 4, (255, 0, 0))
            return _image_blocks_with_payload(
                Image.fromarray(draw_img),
                payload,
                workspace=self._workspace,
                source_stem=file_path.stem,
                kind="contours",
            )
        except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in find_contours: {e}"


class HoughLinesTool(Tool):
    """Detect line segments using the probabilistic Hough transform."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "hough_lines"

    @property
    def description(self) -> str:
        return "Detect straight line segments, useful for table grids, chart axes, and GUI dividers."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional ROI [x1, y1, x2, y2] in 0-1000 relative coords.",
                },
                "canny_threshold1": {"type": "integer", "description": "Lower Canny threshold. Default 50."},
                "canny_threshold2": {"type": "integer", "description": "Upper Canny threshold. Default 150."},
                "threshold": {"type": "integer", "description": "Hough vote threshold. Default 50."},
                "min_line_length": {
                    "type": "integer",
                    "description": "Minimum line length on the 0-1000 scale of max(width, height). Default 80.",
                },
                "max_line_gap": {
                    "type": "integer",
                    "description": "Maximum gap on the 0-1000 scale of max(width, height). Default 20.",
                },
                "max_results": {"type": "integer", "description": "Maximum lines returned. Default 20."},
            },
            "required": ["image_path"],
        }

    async def execute(
        self,
        image_path: str,
        region: list[int] | None = None,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        threshold: int = 50,
        min_line_length: int = 80,
        max_line_gap: int = 20,
        max_results: int = 20,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            full_w, full_h = img.size
            roi, off_x, off_y = _crop_region(img, region)
            arr = np.asarray(roi)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, int(canny_threshold1), int(canny_threshold2))
            scale_size = max(roi.size)
            min_len_px = max(1, resolve_coord(min_line_length, scale_size))
            gap_px = max(0, resolve_coord(max_line_gap, scale_size))
            raw = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=max(1, int(threshold)),
                minLineLength=min_len_px,
                maxLineGap=gap_px,
            )
            records: list[dict[str, Any]] = []
            if raw is not None:
                for idx, line in enumerate(raw[:, 0, :]):
                    x1, y1, x2, y2 = [int(v) for v in line]
                    length = math.hypot(x2 - x1, y2 - y1)
                    records.append(
                        {
                            "id": int(idx + 1),
                            "line": [
                                *_rel_point(x1 + off_x, y1 + off_y, full_w, full_h),
                                *_rel_point(x2 + off_x, y2 + off_y, full_w, full_h),
                            ],
                            "length_px": round(length, 2),
                            "angle_deg": round(math.degrees(math.atan2(y2 - y1, x2 - x1)), 2),
                        }
                    )
            records.sort(key=lambda item: item["length_px"], reverse=True)
            max_results = max(1, int(max_results))
            records = records[:max_results]
            payload = {"line_count": len(records), "lines": records}

            draw_img = np.array(img)
            for record in records:
                x1, y1, x2, y2 = resolve_box(record["line"], full_w, full_h)
                cv2.line(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                _draw_line_label(cv2, draw_img, str(record["id"]), x1, y1 - 4, (255, 0, 0))
            return _image_blocks_with_payload(
                Image.fromarray(draw_img),
                payload,
                workspace=self._workspace,
                source_stem=file_path.stem,
                kind="hough_lines",
            )
        except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in hough_lines: {e}"


class HoughCirclesTool(Tool):
    """Detect circles using the Hough gradient transform."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "hough_circles"

    @property
    def description(self) -> str:
        return "Detect circles, useful for scatter/bubble charts, radio buttons, and circular targets."

    @property
    def parameters(self) -> dict[str, Any]:
        radius_hint = "Radius on the 0-1000 scale of min(width, height)."
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional ROI [x1, y1, x2, y2] in 0-1000 relative coords.",
                },
                "dp": {"type": "number", "description": "Inverse accumulator resolution. Default 1.2."},
                "min_dist": {
                    "type": "integer",
                    "description": "Minimum center distance on the 0-1000 scale. Default 40.",
                },
                "param1": {"type": "integer", "description": "Upper Canny threshold. Default 100."},
                "param2": {
                    "type": "integer",
                    "description": "Accumulator threshold; lower detects more circles. Default 30.",
                },
                "min_radius": {"type": "integer", "description": radius_hint},
                "max_radius": {"type": "integer", "description": radius_hint},
                "max_results": {"type": "integer", "description": "Maximum circles returned. Default 20."},
            },
            "required": ["image_path"],
        }

    async def execute(
        self,
        image_path: str,
        region: list[int] | None = None,
        dp: float = 1.2,
        min_dist: int = 40,
        param1: int = 100,
        param2: int = 30,
        min_radius: int = 0,
        max_radius: int = 0,
        max_results: int = 20,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            full_w, full_h = img.size
            roi, off_x, off_y = _crop_region(img, region)
            arr = np.asarray(roi)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 5)
            scale_size = min(roi.size)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=max(0.1, float(dp)),
                minDist=max(1, resolve_coord(min_dist, scale_size)),
                param1=max(1, int(param1)),
                param2=max(1, int(param2)),
                minRadius=max(0, resolve_coord(min_radius, scale_size)),
                maxRadius=max(0, resolve_coord(max_radius, scale_size)),
            )
            records: list[dict[str, Any]] = []
            if circles is not None:
                rounded = np.round(circles[0, :]).astype(int)
                for idx, (x, y, r) in enumerate(rounded):
                    records.append(
                        {
                            "id": int(idx + 1),
                            "center": _rel_point(x + off_x, y + off_y, full_w, full_h),
                            "radius": _rel_coord(r, min(full_w, full_h)),
                            "radius_px": int(r),
                        }
                    )
            records.sort(key=lambda item: item["radius_px"], reverse=True)
            max_results = max(1, int(max_results))
            records = records[:max_results]
            payload = {"circle_count": len(records), "circles": records}

            draw_img = np.array(img)
            for record in records:
                cx, cy = record["center"]
                abs_x = resolve_coord(cx, full_w)
                abs_y = resolve_coord(cy, full_h)
                abs_r = resolve_coord(record["radius"], min(full_w, full_h))
                cv2.circle(draw_img, (abs_x, abs_y), abs_r, (255, 0, 0), 2)
                _draw_line_label(cv2, draw_img, str(record["id"]), abs_x + abs_r + 2, abs_y, (255, 0, 0))
            return _image_blocks_with_payload(
                Image.fromarray(draw_img),
                payload,
                workspace=self._workspace,
                source_stem=file_path.stem,
                kind="hough_circles",
            )
        except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in hough_circles: {e}"


class TemplateMatchTool(Tool):
    """Locate regions matching a template image or template crop."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "template_match"

    @property
    def description(self) -> str:
        return (
            "Find areas in an image that match a template image or a template crop from "
            "the same image. Useful for GUI icons and visual search."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": IMAGE_PATH_DESC},
                "template_path": {
                    "type": "string",
                    "description": "Path to template image. Omit when using template_region.",
                },
                "template_region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Template crop [x1, y1, x2, y2] in 0-1000 coords from image_path.",
                },
                "search_region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional search ROI [x1, y1, x2, y2] in 0-1000 relative coords.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum normalized match score in [0,1]. Default 0.8.",
                },
                "max_results": {"type": "integer", "description": "Maximum matches returned. Default 10."},
            },
            "required": ["image_path"],
        }

    async def execute(
        self,
        image_path: str,
        template_path: str | None = None,
        template_region: list[int] | None = None,
        search_region: list[int] | None = None,
        threshold: float = 0.8,
        max_results: int = 10,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            img, file_path = load_image(image_path, self._workspace, self._allowed_dir)
            img = img.convert("RGB")
            full_w, full_h = img.size
            search_img, off_x, off_y = _crop_region(img, search_region)

            if template_path and template_region:
                return "Error: use either template_path or template_region, not both"
            if template_path:
                template_img, _ = load_image(template_path, self._workspace, self._allowed_dir)
                template_img = template_img.convert("RGB")
            elif template_region and len(template_region) == 4:
                template_img, _tx, _ty = _crop_region(img, template_region)
            else:
                return "Error: provide template_path or template_region"

            search_arr = np.asarray(search_img)
            template_arr = np.asarray(template_img)
            sh, sw = search_arr.shape[:2]
            th, tw = template_arr.shape[:2]
            if th > sh or tw > sw:
                return "Error: template is larger than the search region"

            search_gray = cv2.cvtColor(search_arr, cv2.COLOR_RGB2GRAY)
            template_gray = cv2.cvtColor(template_arr, cv2.COLOR_RGB2GRAY)
            scores = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            threshold = max(0.0, min(1.0, float(threshold)))
            ys, xs = np.where(scores >= threshold)
            candidates = sorted(
                (
                    (float(scores[y, x]), int(x), int(y))
                    for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
                ),
                reverse=True,
            )

            records: list[dict[str, Any]] = []
            for score, x, y in candidates:
                abs_x = x + off_x
                abs_y = y + off_y
                candidate_box = [abs_x, abs_y, abs_x + tw, abs_y + th]
                overlaps = False
                for record in records:
                    rx1, ry1, rx2, ry2 = resolve_box(record["bbox"], full_w, full_h)
                    ix1 = max(candidate_box[0], rx1)
                    iy1 = max(candidate_box[1], ry1)
                    ix2 = min(candidate_box[2], rx2)
                    iy2 = min(candidate_box[3], ry2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter > 0.3 * tw * th:
                        overlaps = True
                        break
                if overlaps:
                    continue
                records.append(
                    {
                        "id": len(records) + 1,
                        "bbox": _rel_box(abs_x, abs_y, tw, th, full_w, full_h),
                        "center": _rel_point(abs_x + tw / 2, abs_y + th / 2, full_w, full_h),
                        "score": round(score, 4),
                    }
                )
                if len(records) >= max(1, int(max_results)):
                    break

            payload = {"match_count": len(records), "matches": records}

            draw_img = np.array(img)
            for record in records:
                x1, y1, x2, y2 = resolve_box(record["bbox"], full_w, full_h)
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                _draw_line_label(cv2, draw_img, f"{record['id']}:{record['score']}", x1, y1 - 4, (255, 0, 0))
            return _image_blocks_with_payload(
                Image.fromarray(draw_img),
                payload,
                workspace=self._workspace,
                source_stem=file_path.stem,
                kind="template_match",
            )
        except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error in template_match: {e}"
