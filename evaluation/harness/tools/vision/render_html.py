"""Render HTML source to a screenshot via headless Chromium.

Lets the agent execute its own draft HTML and feed the rendered screenshot back
into the next reasoning step (so it can compare with the reference image and
iterate). Mirrors the behaviour of ``distill/filter/evaluate_html_vlm_judge.py``
so rollout-time rendering matches the offline VLM judge.
"""

from __future__ import annotations

import asyncio
import base64
import re
import threading
from pathlib import Path
from typing import Any

from harness.tools.base import Tool
from harness.tools.filesystem import _resolve_path
from harness.tools.vision._common import save_path

_RICK_SRC_RE = re.compile(r"(?P<prefix>\bsrc\s*=\s*[\"'])rick\.jpg(?P<suffix>[\"'])", re.IGNORECASE)
_RICK_URL_RE = re.compile(r"url\(\s*([\"']?)rick\.jpg\1\s*\)", re.IGNORECASE)
_PLACEHOLDER_IMAGE_DATA_URL = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MDAiIGhlaWdodD0i"
    "NjAwIiB2aWV3Qm94PSIwIDAgODAwIDYwMCI+PHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSI2MDAiIGZp"
    "bGw9IiNkOWQ5ZDkiLz48dGV4dCB4PSI0MDAiIHk9IjMwMCIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMt"
    "c2VyaWYiIGZvbnQtc2l6ZT0iNDgiIGZpbGw9IiM3Nzc3NzciIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PS"
    "IuMzVlbSI+cmljay5qcGc8L3RleHQ+PC9zdmc+"
)

_DEFAULT_VIEWPORT = (1280, 720)
_DEFAULT_TIMEOUT_MS = 10000
_DEFAULT_WAIT_MS = 500
_MAX_VIEWPORT = 4096


def _prepare_html(html_text: str) -> str:
    html_text = _RICK_SRC_RE.sub(rf"\g<prefix>{_PLACEHOLDER_IMAGE_DATA_URL}\g<suffix>", html_text)
    return _RICK_URL_RE.sub(f"url('{_PLACEHOLDER_IMAGE_DATA_URL}')", html_text)


async def _render_async(
    html_text: str,
    viewport: tuple[int, int],
    timeout_ms: int,
    wait_ms: int,
) -> bytes:
    from playwright.async_api import async_playwright  # pyright: ignore[reportMissingImports]

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
            await page.set_content(html_text, wait_until="networkidle", timeout=timeout_ms)
            if wait_ms > 0:
                await page.wait_for_timeout(wait_ms)
            return await page.screenshot(full_page=True, timeout=timeout_ms)
        finally:
            await browser.close()


def _render_sync(
    html_text: str,
    viewport: tuple[int, int],
    timeout_ms: int,
    wait_ms: int,
) -> bytes:
    """Run the async renderer in a fresh thread so an outer asyncio loop is fine."""
    result: list[bytes] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(_render_async(html_text, viewport, timeout_ms, wait_ms)))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


class RenderHtmlTool(Tool):
    """Render an HTML string to a screenshot using headless Chromium."""

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "render_html"

    @property
    def description(self) -> str:
        return (
            "Render an HTML file to a full-page screenshot using headless Chromium. "
            "Pass the path of the HTML file you wrote with `write_file` (and patched "
            "with `edit_file`); the tool reads the file from disk and renders it. "
            "Use it to visually verify your HTML against a reference design before "
            "submitting the final answer."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the HTML file to render. The tool reads this file from "
                        "disk — do NOT pass raw HTML. Use the same path you passed to "
                        "`write_file` / `edit_file`."
                    ),
                },
                "viewport_width": {
                    "type": "integer",
                    "description": (
                        f"Viewport width in pixels. Defaults to {_DEFAULT_VIEWPORT[0]}; "
                        "set this to match the reference image width when possible."
                    ),
                },
                "viewport_height": {
                    "type": "integer",
                    "description": (
                        f"Viewport height in pixels. Defaults to {_DEFAULT_VIEWPORT[1]}; "
                        "the screenshot itself is always captured full-page."
                    ),
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        try:
            if not isinstance(path, str) or not path.strip():
                return "Error: path must be a non-empty string"

            try:
                html_path = _resolve_path(path, self._workspace, self._allowed_dir)
            except PermissionError as exc:
                return f"Error: {exc}"
            if not html_path.exists():
                return f"Error: HTML file not found: {path}"
            if not html_path.is_file():
                return f"Error: not a file: {path}"

            try:
                html = html_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                return f"Error reading HTML file as utf-8: {exc}"

            if not html.strip():
                return f"Error: HTML file is empty: {path}"

            width = viewport_width if viewport_width is not None else _DEFAULT_VIEWPORT[0]
            height = viewport_height if viewport_height is not None else _DEFAULT_VIEWPORT[1]
            if width <= 0 or height <= 0:
                return "Error: viewport_width and viewport_height must be positive"
            if width > _MAX_VIEWPORT or height > _MAX_VIEWPORT:
                return f"Error: viewport dimensions must be <= {_MAX_VIEWPORT}"

            prepared = _prepare_html(html)
            try:
                png_bytes = _render_sync(
                    prepared,
                    (width, height),
                    _DEFAULT_TIMEOUT_MS,
                    _DEFAULT_WAIT_MS,
                )
            except ImportError:
                return (
                    "Error: playwright is not installed. Install it in the nanobot env "
                    "and run `python -m playwright install chromium`."
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the model
                return f"Error rendering HTML: {exc}"

            out_path = save_path(self._workspace, html_path.stem, "render")
            if out_path.suffix != ".png":
                out_path = out_path.with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(png_bytes)
            b64 = base64.b64encode(png_bytes).decode()
            return [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": f"Rendered image saved to: {out_path.resolve()}"},
            ]
        except (ValueError, PermissionError) as e:
            return f"Error: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Error in render_html: {e}"
