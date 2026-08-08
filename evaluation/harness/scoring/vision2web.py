"""Vision2Web evaluation: HTML resolution, headless rendering, and the
component-level visual-fidelity scoring that turns a generated webpage into a
0-100 score.

Pipeline per sample:
  1. ``resolve_submitted_html``       — pull the submitted HTML (final-turn code
                                        block, else the last HTML file the model
                                        wrote/edited in the agent loop).
  2. ``render_html_to_viewport_images`` — render that HTML once per device
                                        viewport via headless Chromium.
  3. ``judge_vision2web_viewports``   — judge each rendered viewport against its
                                        prototype screenshot, average the scores.

Helpers (``DEVICES``, ``vision2web_viewports``, ``vision2web_ref_images``,
``is_vision2web_*``) live here so eval.py and rejudge.py share one definition
instead of duplicating the viewport/reference bookkeeping.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

# Canonical device order used everywhere a Vision2Web sample is scored.
DEVICES: tuple[str, ...] = ("desktop", "tablet", "mobile")

DEFAULT_VIEWPORTS: dict[str, dict[str, int]] = {
    "desktop": {"width": 1920, "height": 1080},
    "tablet": {"width": 1024, "height": 768},
    "mobile": {"width": 375, "height": 812},
}

_PLAYWRIGHT = None
_BROWSER = None
_BROWSER_LOCK: asyncio.Lock | None = None
_RENDER_SEMAPHORE: asyncio.Semaphore | None = None


# ----------------------------------------------------------------------------
# Dataset predicates
# ----------------------------------------------------------------------------

def is_vision2web_source(source_dataset: str | None) -> bool:
    """True for the Vision2Web static-webpage benchmark (continuous 0-100 score).

    Used both to route eval.py to the code-gen scoring path and to tell the
    result aggregator / dashboard that scores are continuous rather than 0/1.
    """
    return (source_dataset or "") == "Vision2Web-webpage"


def is_vision2web_item(item: dict) -> bool:
    return is_vision2web_source(item.get("source_dataset"))


# ----------------------------------------------------------------------------
# Per-sample viewport + reference bookkeeping
# ----------------------------------------------------------------------------

def vision2web_viewports(sample_dir: str | Path) -> dict[str, dict[str, int]]:
    """Per-device render viewports from the task's ``workflow.json``, falling
    back to ``DEFAULT_VIEWPORTS`` for any device the workflow omits."""
    workflow_path = Path(sample_dir) / "workflow.json"
    viewports: dict[str, dict[str, int]] = {}
    if workflow_path.exists():
        try:
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        except Exception:
            workflow = None
        if isinstance(workflow, list):
            for entry in workflow:
                if not isinstance(entry, dict):
                    continue
                device = str(entry.get("summary") or "").strip().lower()
                resolution = entry.get("resolution") or {}
                if device and {"width", "height"}.issubset(resolution):
                    viewports[device] = {"width": int(resolution["width"]),
                                         "height": int(resolution["height"])}
    for device, viewport in DEFAULT_VIEWPORTS.items():
        viewports.setdefault(device, viewport)
    return viewports


def vision2web_ref_images(item: dict, sample_images: list[str]) -> dict[str, str]:
    """Map ``desktop``/``tablet``/``mobile`` -> the resolved prototype image
    path in the sandbox, derived from the benchmark item's image list."""
    refs: dict[str, str] = {}
    for rel, resolved in zip(item.get("images", []), sample_images):
        name = Path(rel).stem.lower()
        if name in DEVICES:
            refs[name] = resolved
    return refs


def _render_concurrency() -> int:
    raw = os.environ.get("HTML_RENDER_WORKERS") or os.environ.get("HTML_RENDER_CONCURRENCY")
    if not raw:
        return 4
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def _get_browser_lock() -> asyncio.Lock:
    global _BROWSER_LOCK
    if _BROWSER_LOCK is None:
        _BROWSER_LOCK = asyncio.Lock()
    return _BROWSER_LOCK


def _get_render_semaphore() -> asyncio.Semaphore:
    global _RENDER_SEMAPHORE
    if _RENDER_SEMAPHORE is None:
        _RENDER_SEMAPHORE = asyncio.Semaphore(_render_concurrency())
    return _RENDER_SEMAPHORE


async def _get_browser():
    """Return a process-wide Chromium browser, relaunching if needed."""
    global _PLAYWRIGHT, _BROWSER
    if _BROWSER is not None and getattr(_BROWSER, "is_connected", lambda: True)():
        return _BROWSER

    async with _get_browser_lock():
        if _BROWSER is not None and getattr(_BROWSER, "is_connected", lambda: True)():
            return _BROWSER
        from playwright.async_api import async_playwright

        if _PLAYWRIGHT is None:
            _PLAYWRIGHT = await async_playwright().start()
        _BROWSER = await _PLAYWRIGHT.chromium.launch(headless=True)
        return _BROWSER


async def _discard_browser() -> None:
    """Drop the cached browser after a render failure so the next call relaunches."""
    global _BROWSER, _PLAYWRIGHT
    async with _get_browser_lock():
        browser = _BROWSER
        playwright = _PLAYWRIGHT
        _BROWSER = None
        _PLAYWRIGHT = None
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass


async def _discard_browser_if_disconnected(browser) -> None:
    try:
        connected = browser.is_connected()
    except Exception:
        connected = False
    if not connected:
        await _discard_browser()


# ----------------------------------------------------------------------------
# HTML extraction / fallback
# ----------------------------------------------------------------------------

def extract_html(response: str) -> str:
    """Extract HTML from an agent response.

    Handles three common patterns:
      1. ```html ... ```  (fenced code block with language tag)
      2. ``` ... ```      (fenced code block without tag, content starts with '<')
      3. Raw HTML         (response itself starts with '<')
    Falls back to the full response if nothing matches.
    """
    if not (response or "").strip():
        return ""

    match = re.search(r"```html\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if candidate.startswith("<"):
            return candidate

    stripped = response.strip()
    if stripped.startswith("<"):
        return stripped

    return response.strip()


# Structural tags that mark a string as real HTML (vs. prose like "The page has
# been created..."). extract_html falls back to the whole response, so we must
# distinguish actual markup from a natural-language final turn.
_HTML_TAG_RE = re.compile(r"<\s*(?:!doctype\s+html|html|head|body|main|section|div)\b", re.I)

# Tool calls that touch the submitted HTML file, in the agentic write→render→
# edit workflow used by openvistool-trained models.
_HTML_EDIT_TOOLS = ("write_file", "edit_file", "render_html")


def looks_like_html(text: str | None) -> bool:
    """True if *text* contains an HTML structural tag (not just prose)."""
    return bool(_HTML_TAG_RE.search(text or ""))


def _iter_tool_calls(turns):
    """Yield (tool_name, args_dict) for every tool call across *turns*, in order.

    Tolerates both the in-memory turn shape and the trace.jsonl shape
    (tool_calls live under turn["message"], each with function.name +
    function.arguments where arguments is a JSON string).
    """
    for turn in (turns or []):
        if not isinstance(turn, dict):
            continue
        holder = turn.get("message") if isinstance(turn.get("message"), dict) else turn
        for tc in (holder.get("tool_calls") or []):
            fn = (tc or {}).get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
            yield name, args


def last_edited_html_path(turns) -> str | None:
    """Virtual path of the LAST .html file the model wrote/edited/rendered."""
    last = None
    for name, args in _iter_tool_calls(turns):
        if name in _HTML_EDIT_TOOLS:
            p = args.get("path")
            if isinstance(p, str) and p.strip().lower().endswith(".html"):
                last = p.strip()
    return last


def resolve_submitted_html(pred, turns, sample_dir, path_mapper=None):
    """Resolve the HTML to evaluate for a code-gen sample.

    1. ``extract_html(pred)`` if it looks like real HTML (final-turn code block).
    2. Otherwise fall back to the last HTML file the model wrote/edited in the
       agent loop, read from the sandbox. This matches openvistool-trained
       models that iterate via render_html and may end on a plain-text turn.

    Returns ``(html, source)`` where source is ``"pred"``, ``"file:<name>"``,
    or ``"none"`` (nothing usable found — caller renders a blank as before).
    """
    html = extract_html(pred)
    if looks_like_html(html):
        return html, "pred"

    vpath = last_edited_html_path(turns)
    if vpath:
        candidates = []
        if path_mapper is not None:
            try:
                candidates.append(Path(path_mapper.to_real(vpath)))
            except Exception:
                pass
        candidates.append(Path(sample_dir) / Path(vpath).name)
        seen = set()
        for fp in candidates:
            key = str(fp)
            if key in seen:
                continue
            seen.add(key)
            try:
                if fp.is_file():
                    txt = fp.read_text(encoding="utf-8", errors="replace")
                    if looks_like_html(txt):
                        return txt, f"file:{fp.name}"
            except OSError:
                pass

    return html, "none"


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

async def render_html_to_viewport_images(
    html_content: str,
    output_dir: str | Path,
    viewports: dict[str, dict[str, int]],
    html_filename: str = "generated.html",
) -> dict[str, str]:
    """Render HTML once per named viewport and save full-page screenshots.

    Returns a mapping such as {"desktop": "/.../generated_desktop.png"}.
    The HTML file is written inside *output_dir* so relative references like
    ``resources/foo.png`` resolve against the task sandbox.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / html_filename
    html_path.write_text(html_content, encoding="utf-8")

    outputs: dict[str, str] = {}
    async with _get_render_semaphore():
        browser = await _get_browser()
        try:
            for name, viewport in viewports.items():
                context = await browser.new_context(
                    viewport={
                        "width": int(viewport["width"]),
                        "height": int(viewport["height"]),
                    },
                    device_scale_factor=1,
                )
                try:
                    page = await context.new_page()
                    await page.goto(html_path.as_uri(), wait_until="domcontentloaded")
                    await page.wait_for_timeout(500)
                    screenshot_path = out_dir / f"generated_{name}.png"
                    await page.screenshot(path=str(screenshot_path), full_page=True)
                    outputs[name] = str(screenshot_path)
                finally:
                    await context.close()
        except Exception:
            await _discard_browser_if_disconnected(browser)
            raise

    return outputs


# ----------------------------------------------------------------------------
# Scoring (shared by eval.py and rejudge.py)
# ----------------------------------------------------------------------------

async def judge_vision2web_viewports(
    judge, ref_by_device: dict[str, str], gen_by_device: dict[str, str],
    devices: tuple[str, ...] = DEVICES,
) -> dict | None:
    """Judge each rendered viewport against its prototype and average the scores.

    ``judge`` is a :class:`~harness.scoring.judge.Judge`. ``ref_by_device`` maps
    device -> prototype image path; ``gen_by_device`` maps device -> rendered
    screenshot path. A viewport is scored only when both images are present.

    Returns ``{"scores", "avg", "details", "raw"}`` (avg is 0-100), or ``None``
    when no viewport could be scored (caller treats it as a 0 / skip).
    """
    viewport_scores: dict[str, float] = {}
    details: dict[str, dict] = {}
    raws: list[str] = []
    for device in devices:
        ref = ref_by_device.get(device)
        gen = gen_by_device.get(device)
        if not ref or not gen:
            continue
        try:
            score, raw, components = await judge.judge_vision2web_static(ref, gen, device)
        except Exception as e:  # noqa: BLE001 — one bad viewport must not abort the rest
            score, raw, components = 0.0, f"<judge error: {e}>", []
        viewport_scores[device] = score
        details[device] = {
            "score": score,
            "reference": ref,
            "generated": gen,
            "components": components,
            "judge_raw": raw,
        }
        raws.append(raw)

    if not viewport_scores:
        return None
    avg = round(sum(viewport_scores.values()) / len(viewport_scores), 4)
    return {"scores": viewport_scores, "avg": avg, "details": details,
            "raw": " || ".join(raws)}
