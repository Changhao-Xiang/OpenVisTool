"""Evaluate screenshot-to-HTML predictions with rules first, then a VLM judge.

For VinciCoder-style tasks, the reference answer is not a canonical string:
many different HTML/CSS implementations can render the same page. This script
therefore uses deterministic rules only to reject obvious failures, renders the
remaining prediction HTML, and asks a VLM judge to score visual consistency with
the reference screenshot from 0 to 100.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import html
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>")
_TOOL_CALL_RE = re.compile(r"<tool_call>[\s\S]*?</tool_call>")
_FENCED_CODE_RE = re.compile(r"```(?:html)?\s*([\s\S]*?)```", re.IGNORECASE)
_SCORE_RE = re.compile(r"(?<!\d)(100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)(?!\d)")
_TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&.+-]*")
_RICK_SRC_RE = re.compile(r"(?P<prefix>\bsrc\s*=\s*[\"'])rick\.jpg(?P<suffix>[\"'])", re.IGNORECASE)
_RICK_URL_RE = re.compile(r"url\(\s*([\"']?)rick\.jpg\1\s*\)", re.IGNORECASE)
_DEFAULT_QUERY_FALLBACKS = ("query", "question", "problem")
_DEFAULT_MEDIA_FALLBACKS = ("images", "image", "screenshot", "screenshots")
_PLACEHOLDER_IMAGE_DATA_URL = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MDAiIGhlaWdodD0i"
    "NjAwIiB2aWV3Qm94PSIwIDAgODAwIDYwMCI+PHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSI2MDAiIGZp"
    "bGw9IiNkOWQ5ZDkiLz48dGV4dCB4PSI0MDAiIHk9IjMwMCIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMt"
    "c2VyaWYiIGZvbnQtc2l6ZT0iNDgiIGZpbGw9IiM3Nzc3NzciIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PS"
    "IuMzVlbSI+cmljay5qcGc8L3RleHQ+PC9zdmc+"
)
_STRUCTURAL_TAGS = {
    "a",
    "article",
    "aside",
    "button",
    "div",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "header",
    "img",
    "input",
    "li",
    "main",
    "nav",
    "p",
    "section",
    "span",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_BOILERPLATE_TEXT = {
    "html",
    "head",
    "body",
    "style",
    "script",
    "doctype",
    "charset",
    "viewport",
    "width",
    "height",
    "margin",
    "padding",
    "display",
    "flex",
    "color",
    "background",
    "font",
    "family",
    "arial",
    "sans",
    "serif",
    "rick",
    "jpg",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            items.append(item)
    return items


def _get_text_field(item: dict[str, Any], field: str, fallbacks: tuple[str, ...] = ()) -> str:
    for candidate in (field, *fallbacks):
        value = item.get(candidate)
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        text = text.strip()
        if text:
            return text
    return ""


def _resolve_media_paths(item: dict[str, Any], media_field: str, media_dir: Path) -> list[Path]:
    raw_value = item.get(media_field)
    if raw_value is None:
        for fallback in _DEFAULT_MEDIA_FALLBACKS:
            raw_value = item.get(fallback)
            if raw_value is not None:
                break
    if raw_value is None:
        return []

    raw_paths = raw_value if isinstance(raw_value, list) else [raw_value]
    resolved: list[Path] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path.strip()).expanduser()
        if not path.is_absolute():
            path = (media_dir / path).resolve()
        if path.is_file():
            resolved.append(path)
    return resolved


def _load_records(path: Path) -> list[dict[str, Any]] | None:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return None


def _extract_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []

    events = records[1:] if records[0].get("_type") == "metadata" else records
    last_user_idx = -1
    for idx, record in enumerate(events):
        if record.get("role") == "user":
            last_user_idx = idx
    return events[last_user_idx:] if last_user_idx >= 0 else events


def _parse_session(session_path: Path) -> tuple[str | None, bool]:
    records = _load_records(session_path)
    if not records:
        return None, False

    last_visible = None
    tool_called = False
    for record in _extract_events(records):
        if record.get("role") != "assistant":
            continue
        if record.get("tool_calls"):
            tool_called = True
        content = str(record.get("content", "") or "")
        visible = _TOOL_CALL_RE.sub("", _THINK_RE.sub("", content)).strip()
        if visible:
            last_visible = visible
    return last_visible, tool_called


def _find_session_trace(sessions_dir: Path, item_id: Any, sample_index: int | None) -> Path | None:
    suffixes = [f"_s{sample_index}"] if sample_index is not None else [""]
    if sample_index is None:
        suffixes.extend([f"_s{idx}" for idx in range(10)])

    for suffix in suffixes:
        session_dir = sessions_dir / f"session_{item_id}{suffix}"
        nested = session_dir / "sessions"
        candidates = sorted(nested.glob("*.jsonl")) if nested.is_dir() else []
        if candidates:
            return candidates[0]
        candidates = sorted(session_dir.glob("*.jsonl")) if session_dir.is_dir() else []
        if candidates:
            return candidates[0]
    return None


class _HtmlStatsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.attrs: list[tuple[str, str, str]] = []
        self.text_parts: list[str] = []
        self.style_parts: list[str] = []
        self.errors: list[str] = []
        self._stack: list[str] = []
        self._raw_text_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.tags.append(tag)
        self._stack.append(tag)
        if tag in {"script", "style"}:
            self._raw_text_tag = tag
        for key, value in attrs:
            self.attrs.append((tag, key.lower(), value or ""))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.tags.append(tag)
        for key, value in attrs:
            self.attrs.append((tag, key.lower(), value or ""))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._raw_text_tag == tag:
            self._raw_text_tag = None
        if tag in self._stack:
            while self._stack:
                open_tag = self._stack.pop()
                if open_tag == tag:
                    break

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._raw_text_tag == "style":
            self.style_parts.append(text)
        elif self._raw_text_tag is None:
            self.text_parts.append(html.unescape(text))

    def error(self, message: str) -> None:
        self.errors.append(message)


@dataclass(frozen=True)
class HtmlStats:
    tags: Counter[str]
    structural_tags: Counter[str]
    attrs: list[tuple[str, str, str]]
    text_tokens: set[str]
    total_tags: int
    has_style: bool
    has_css_like_content: bool
    has_rick_image: bool


def _extract_html(text: str) -> str:
    text = _TOOL_CALL_RE.sub("", _THINK_RE.sub("", text or "")).strip()
    fenced = _FENCED_CODE_RE.findall(text)
    if fenced:
        html_blocks = [block.strip() for block in fenced if "<" in block and ">" in block]
        if html_blocks:
            return max(html_blocks, key=len)

    lower = text.lower()
    html_start = lower.find("<!doctype html")
    if html_start < 0:
        html_start = lower.find("<html")
    if html_start < 0:
        html_start = lower.find("<body")
    if html_start < 0:
        html_start = lower.find("<div")
    return text[html_start:].strip() if html_start >= 0 else text


def _text_tokens(text: str) -> set[str]:
    tokens = {token.lower() for token in _TEXT_TOKEN_RE.findall(text)}
    return {token for token in tokens if len(token) >= 3 and token not in _BOILERPLATE_TEXT}


def _parse_html_stats(html_text: str) -> tuple[HtmlStats | None, list[str]]:
    parser = _HtmlStatsParser()
    errors: list[str] = []
    try:
        parser.feed(html_text)
        parser.close()
    except Exception as exc:
        errors.append(f"html_parse_error:{exc}")
        return None, errors

    tag_counter = Counter(parser.tags)
    structural = Counter({tag: count for tag, count in tag_counter.items() if tag in _STRUCTURAL_TAGS})
    text = " ".join(parser.text_parts)
    styles = [value for tag, key, value in parser.attrs if key == "style"]
    style_text = " ".join(parser.style_parts)
    has_style = bool(tag_counter["style"] or styles)
    has_css_like_content = has_style or bool(re.search(r"\{[^{}]+:[^{}]+;?[^{}]*\}", style_text))
    has_rick_image = any(
        tag == "img" and key == "src" and Path(value).name == "rick.jpg" for tag, key, value in parser.attrs
    )

    stats = HtmlStats(
        tags=tag_counter,
        structural_tags=structural,
        attrs=parser.attrs,
        text_tokens=_text_tokens(text),
        total_tags=len(parser.tags),
        has_style=has_style,
        has_css_like_content=has_css_like_content,
        has_rick_image=has_rick_image,
    )
    return stats, errors


def _counter_jaccard(lhs: Counter[str], rhs: Counter[str]) -> float:
    keys = set(lhs) | set(rhs)
    if not keys:
        return 1.0
    intersection = sum(min(lhs[key], rhs[key]) for key in keys)
    union = sum(max(lhs[key], rhs[key]) for key in keys)
    return intersection / union if union else 1.0


def _set_recall(pred: set[str], ref: set[str]) -> float:
    if not ref:
        return 1.0
    return len(pred & ref) / len(ref)


def rule_check_html(
    prediction: str,
    reference_html: str,
    *,
    min_tag_ratio: float,
    min_structural_jaccard: float,
    min_text_recall: float,
    require_css: bool,
    require_rick_image: bool,
) -> dict[str, Any]:
    html_text = _extract_html(prediction)
    result: dict[str, Any] = {
        "passed": False,
        "errors": [],
        "html_length": len(html_text),
        "metrics": {},
        "extracted_html": html_text,
    }
    if not html_text:
        result["errors"].append("empty_prediction")
        return result
    if "<" not in html_text or ">" not in html_text:
        result["errors"].append("no_html_tags")
        return result

    pred_stats, pred_errors = _parse_html_stats(html_text)
    ref_stats, ref_errors = _parse_html_stats(reference_html)
    result["errors"].extend(pred_errors)
    if ref_errors:
        result["errors"].extend(f"reference_{error}" for error in ref_errors)
    if pred_stats is None or ref_stats is None:
        return result

    tag_ratio = pred_stats.total_tags / max(ref_stats.total_tags, 1)
    structural_jaccard = _counter_jaccard(pred_stats.structural_tags, ref_stats.structural_tags)
    text_recall = _set_recall(pred_stats.text_tokens, ref_stats.text_tokens)
    metrics = {
        "pred_total_tags": pred_stats.total_tags,
        "ref_total_tags": ref_stats.total_tags,
        "tag_ratio": round(tag_ratio, 4),
        "structural_jaccard": round(structural_jaccard, 4),
        "text_recall": round(text_recall, 4),
        "pred_has_css": pred_stats.has_css_like_content,
        "ref_has_css": ref_stats.has_css_like_content,
        "pred_has_rick_image": pred_stats.has_rick_image,
        "ref_has_rick_image": ref_stats.has_rick_image,
    }
    result["metrics"] = metrics

    if pred_stats.total_tags < 5:
        result["errors"].append("too_few_dom_nodes")
    if tag_ratio < min_tag_ratio:
        result["errors"].append("dom_too_small")
    if structural_jaccard < min_structural_jaccard:
        result["errors"].append("major_dom_structure_mismatch")
    if text_recall < min_text_recall:
        result["errors"].append("low_reference_text_coverage")
    if require_css and ref_stats.has_css_like_content and not pred_stats.has_css_like_content:
        result["errors"].append("missing_css")
    if require_rick_image and ref_stats.has_rick_image and not pred_stats.has_rick_image:
        result["errors"].append("missing_required_rick_image")

    result["passed"] = not result["errors"]
    return result


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def _image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{_guess_mime_type(path)};base64,{encoded}"


def _prepare_html_for_render(html_text: str) -> str:
    html_text = _RICK_SRC_RE.sub(rf"\g<prefix>{_PLACEHOLDER_IMAGE_DATA_URL}\g<suffix>", html_text)
    return _RICK_URL_RE.sub(f"url('{_PLACEHOLDER_IMAGE_DATA_URL}')", html_text)


class HtmlRenderer:
    """Thread-safe Playwright renderer with per-thread browser reuse.

    Each worker thread launches one chromium instance on first use and keeps
    it alive for the rest of the process. Per-render cost drops from
    chromium-launch (~3s) + page work to just new_page + page work, which is
    the only viable approach when filtering tens of thousands of sessions.
    """

    def __init__(self, *, timeout_ms: int, wait_ms: int) -> None:
        self.timeout_ms = timeout_ms
        self.wait_ms = wait_ms
        self._tls = threading.local()
        self._browsers: list[Any] = []
        self._playwrights: list[Any] = []
        self._lock = threading.Lock()
        atexit.register(self._shutdown)

    def _get_browser(self):
        browser = getattr(self._tls, "browser", None)
        if browser is not None:
            return browser
        try:
            from playwright.sync_api import sync_playwright  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "playwright is required for rendering. Install it in the nanobot env and run "
                "`python -m playwright install chromium`."
            ) from exc

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
        self._tls.playwright = pw
        self._tls.browser = browser
        with self._lock:
            self._browsers.append(browser)
            self._playwrights.append(pw)
        return browser

    def _shutdown(self) -> None:
        with self._lock:
            for browser in self._browsers:
                try:
                    browser.close()
                except Exception:
                    pass
            for pw in self._playwrights:
                try:
                    pw.stop()
                except Exception:
                    pass
            self._browsers.clear()
            self._playwrights.clear()

    def _discard_browser(self) -> None:
        # Called after a render failure: the chromium process may have died
        # (TargetClosedError, OOM, …). Drop the cached browser so the next
        # render on this thread re-launches a fresh one.
        browser = getattr(self._tls, "browser", None)
        pw = getattr(self._tls, "playwright", None)
        self._tls.browser = None
        self._tls.playwright = None
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
            with self._lock:
                try:
                    self._browsers.remove(browser)
                except ValueError:
                    pass
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass
            with self._lock:
                try:
                    self._playwrights.remove(pw)
                except ValueError:
                    pass

    def render(self, html_text: str, output_path: Path, viewport: tuple[int, int]) -> None:
        html_text = _prepare_html_for_render(html_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        browser = self._get_browser()
        try:
            context = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]})
        except Exception:
            self._discard_browser()
            raise
        try:
            page = context.new_page()
            try:
                page.set_content(html_text, wait_until="networkidle", timeout=self.timeout_ms)
            except Exception:
                # networkidle can stall on pages with always-pending resources;
                # fall back to "load" so the screenshot still happens.
                page.set_content(html_text, wait_until="load", timeout=self.timeout_ms)
            if self.wait_ms > 0:
                page.wait_for_timeout(self.wait_ms)
            page.screenshot(path=str(output_path), full_page=True, timeout=self.timeout_ms)
        except Exception:
            self._discard_browser()
            raise
        finally:
            try:
                context.close()
            except Exception:
                pass


class VlmHtmlJudge:
    _SYSTEM = (
        "You are a visual consistency judge. Compare the rendered image with the reference image "
        "and score their consistency from 0 to 100, where 0 means completely different and "
        "100 means completely identical."
    )

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._thread_local = threading.local()

    def _get_client(self) -> OpenAI:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=self.base_url,
            )
            self._thread_local.client = client
        return client

    @staticmethod
    def _parse_score(content: str) -> tuple[float | None, str]:
        text = content.strip()
        if not text:
            return None, ""
        match = _SCORE_RE.search(text)
        if match:
            return float(match.group(1)), text
        return None, text

    def judge(
        self,
        *,
        reference_image: Path,
        rendered_image: Path,
        retries: int,
    ) -> tuple[float, str, str]:
        user_text = (
            "Compare the two images for visual consistency. Image 1 is the reference image. "
            "Image 2 is the rendered image. Output only `SCORE: 0-100`."
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(reference_image)}},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(rendered_image)}},
        ]

        client = self._get_client()
        last_reply = ""
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._SYSTEM},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                last_reply = str(response.choices[0].message.content or "")
                score, reason = self._parse_score(last_reply)
                if score is not None:
                    return max(0.0, min(100.0, score)), reason, last_reply
            except Exception as exc:
                last_reply = f"judge_error:{exc}"
            if attempt < retries - 1:
                time.sleep(2**attempt)
        return 0.0, last_reply, last_reply


def _reference_viewport(reference_image: Path, default_viewport: tuple[int, int]) -> tuple[int, int]:
    try:
        with Image.open(reference_image) as image:
            width, height = image.size
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return default_viewport


def _sample_output_stem(item_id: Any, index: int, sample_index: int | None) -> str:
    sample = f"_s{sample_index}" if sample_index is not None else ""
    return f"item_{item_id if item_id is not None else index}{sample}"


def _process_one(
    item: dict[str, Any],
    *,
    index: int,
    sample_index: int | None,
    dataset_path: Path,
    sessions_dir: Path | None,
    media_dir: Path,
    render_dir: Path,
    query_field: str,
    answer_field: str,
    prediction_field: str,
    media_field: str,
    renderer: HtmlRenderer,
    judge: VlmHtmlJudge,
    judge_retries: int,
    score_threshold: float,
    min_tag_ratio: float,
    min_structural_jaccard: float,
    min_text_recall: float,
    require_css: bool,
    require_rick_image: bool,
    keep_rendered_images: bool,
    default_viewport: tuple[int, int],
) -> dict[str, Any]:
    item_id = item.get("id", index)
    query = _get_text_field(item, query_field, _DEFAULT_QUERY_FALLBACKS)
    reference_html = _get_text_field(item, answer_field)
    reference_images = _resolve_media_paths(item, media_field, media_dir)

    predicted = ""
    trace_path = None
    tool_called = False
    trace_missing = False
    if sessions_dir is not None:
        trace_path = _find_session_trace(sessions_dir, item_id, sample_index)
        if trace_path is None:
            trace_missing = True
        else:
            predicted, tool_called = _parse_session(trace_path)
            predicted = predicted or ""
    else:
        predicted = _get_text_field(item, prediction_field)

    result: dict[str, Any] = {
        "id": item_id,
        "index": index,
        "sample_index": sample_index,
        "query": query,
        "trace_path": str(trace_path) if trace_path else "",
        "trace_missing": trace_missing,
        "tool_called": tool_called,
        "reference_image": str(reference_images[0]) if reference_images else "",
        "rendered_image": "",
        "rule_passed": False,
        "rule_errors": [],
        "rule_metrics": {},
        "score": 0.0,
        "correct": False,
        "judge_reason": "",
        "judge_raw": "",
    }
    if trace_missing:
        result["rule_errors"] = ["missing_session_trace"]
        return result
    if not predicted:
        result["rule_errors"] = ["empty_prediction"]
        return result
    if not reference_html:
        result["rule_errors"] = ["missing_reference_html"]
        return result
    if not reference_images:
        result["rule_errors"] = ["missing_reference_image"]
        return result

    rule = rule_check_html(
        predicted,
        reference_html,
        min_tag_ratio=min_tag_ratio,
        min_structural_jaccard=min_structural_jaccard,
        min_text_recall=min_text_recall,
        require_css=require_css,
        require_rick_image=require_rick_image,
    )
    result["rule_passed"] = bool(rule["passed"])
    result["rule_errors"] = rule["errors"]
    result["rule_metrics"] = rule["metrics"]
    if not rule["passed"]:
        return result

    html_text = rule["extracted_html"]
    stem = _sample_output_stem(item_id, index, sample_index)
    rendered_path = (
        render_dir / f"{dataset_path.stem}_{stem}.png"
        if keep_rendered_images
        else Path(tempfile.gettempdir()) / f"{dataset_path.stem}_{stem}_{threading.get_ident()}.png"
    )
    result["rendered_image"] = str(rendered_path) if keep_rendered_images else ""
    viewport = _reference_viewport(reference_images[0], default_viewport)
    try:
        renderer.render(html_text, rendered_path, viewport)
    except Exception as exc:
        result["rule_passed"] = False
        result["rule_errors"] = [*result["rule_errors"], f"render_error:{exc}"]
        return result

    score, reason, raw = judge.judge(
        query=query,
        reference_image=reference_images[0],
        rendered_image=rendered_path,
        retries=judge_retries,
    )
    if not keep_rendered_images:
        rendered_path.unlink(missing_ok=True)
    result["score"] = round(score, 4)
    result["correct"] = score >= score_threshold
    result["judge_reason"] = reason
    result["judge_raw"] = raw
    return result


def _parse_sample_indices(value: str) -> list[int | None]:
    raw = value.strip()
    if not raw:
        return [None]
    if raw.lower() in {"none", "default"}:
        return [None]
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_viewport(value: str) -> tuple[int, int]:
    raw = value.lower().replace("x", ",")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--default-viewport must look like 1280x720")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("--default-viewport values must be positive")
    return width, height


def evaluate_dataset(
    *,
    dataset_path: Path,
    sessions_dir: Path | None,
    media_dir: Path,
    render_dir: Path,
    output_path: Path,
    sample_indices: list[int | None],
    query_field: str,
    answer_field: str,
    prediction_field: str,
    media_field: str,
    judge: VlmHtmlJudge,
    renderer: HtmlRenderer,
    judge_retries: int,
    score_threshold: float,
    min_tag_ratio: float,
    min_structural_jaccard: float,
    min_text_recall: float,
    require_css: bool,
    require_rick_image: bool,
    keep_rendered_images: bool,
    default_viewport: tuple[int, int],
    num_workers: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    items = _load_jsonl(dataset_path)
    if limit is not None:
        items = items[:limit]

    pending: list[tuple[dict[str, Any], int, int | None]] = []
    for index, item in enumerate(items):
        for sample_index in sample_indices:
            pending.append((item, index, sample_index))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if keep_rendered_images:
        render_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any] | None] = [None] * len(pending)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                item,
                index=index,
                sample_index=sample_index,
                dataset_path=dataset_path,
                sessions_dir=sessions_dir,
                media_dir=media_dir,
                render_dir=render_dir,
                query_field=query_field,
                answer_field=answer_field,
                prediction_field=prediction_field,
                media_field=media_field,
                renderer=renderer,
                judge=judge,
                judge_retries=judge_retries,
                score_threshold=score_threshold,
                min_tag_ratio=min_tag_ratio,
                min_structural_jaccard=min_structural_jaccard,
                min_text_recall=min_text_recall,
                require_css=require_css,
                require_rick_image=require_rick_image,
                keep_rendered_images=keep_rendered_images,
                default_viewport=default_viewport,
            ): out_idx
            for out_idx, (item, index, sample_index) in enumerate(pending)
        }
        with tqdm(total=len(futures), desc="Evaluating HTML") as progress:
            for future in as_completed(futures):
                out_idx = futures[future]
                result = future.result()
                results[out_idx] = result
                with output_path.open("a", encoding="utf-8") as file_obj:
                    file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")
                progress.update(1)

    return [result for result in results if result is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate screenshot-to-HTML predictions with rule checks and VLM visual scoring."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--sessions-dir", default="", help="Rollout sessions dir. If empty, use --prediction-field."
    )
    parser.add_argument("--media-dir", default="", help="Defaults to <dataset_dir>/images.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--render-dir", default="", help="Defaults to <output_dir>/rendered_html.")
    parser.add_argument(
        "--keep-rendered-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to keep rendered prediction screenshots on disk after judging.",
    )
    parser.add_argument(
        "--sample-indices", default="none", help="Comma-separated rollout sample indices, e.g. 0,1,2."
    )
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--media-field", default="images")
    parser.add_argument("--judge-model", default="gemini-3.0-flash-preview")
    parser.add_argument("--judge-api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--judge-max-tokens", type=int, default=16384)
    parser.add_argument("--judge-temperature", type=float, default=0.6)
    parser.add_argument("--score-threshold", type=float, default=80.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--render-timeout-ms", type=int, default=10000)
    parser.add_argument("--render-wait-ms", type=int, default=500)
    parser.add_argument("--default-viewport", default="1280x720")
    parser.add_argument("--min-tag-ratio", type=float, default=0.35)
    parser.add_argument("--min-structural-jaccard", type=float, default=0.25)
    parser.add_argument("--min-text-recall", type=float, default=0.25)
    parser.add_argument("--no-require-css", action="store_true")
    parser.add_argument("--no-require-rick-image", action="store_true")
    args = parser.parse_args()

    if not args.judge_api_base:
        raise EnvironmentError("--judge-api-base or OPENAI_API_BASE is required")

    dataset_path = Path(args.dataset).resolve()
    sessions_dir = Path(args.sessions_dir).resolve() if args.sessions_dir else None
    media_dir = (
        Path(args.media_dir).resolve() if args.media_dir else (dataset_path.parent / "images").resolve()
    )
    output_path = Path(args.output).resolve()
    render_dir = (
        Path(args.render_dir).resolve()
        if args.render_dir
        else (output_path.parent / "rendered_html").resolve()
    )
    if output_path.exists():
        output_path.unlink()

    sample_indices = _parse_sample_indices(args.sample_indices)
    default_viewport = _parse_viewport(args.default_viewport)
    renderer = HtmlRenderer(timeout_ms=args.render_timeout_ms, wait_ms=args.render_wait_ms)
    judge = VlmHtmlJudge(
        model=args.judge_model,
        base_url=args.judge_api_base,
        max_tokens=args.judge_max_tokens,
        temperature=args.judge_temperature,
    )

    print(
        f"Dataset: {dataset_path}\nMedia dir: {media_dir}\nSessions dir: {sessions_dir or '-'}\n"
        f"Judge: {args.judge_model}\nOutput: {output_path}\nRender dir: {render_dir}\n"
        f"Keep rendered images: {args.keep_rendered_images}\n"
        f"Score threshold: {args.score_threshold}, workers={args.num_workers}"
    )

    results = evaluate_dataset(
        dataset_path=dataset_path,
        sessions_dir=sessions_dir,
        media_dir=media_dir,
        render_dir=render_dir,
        output_path=output_path,
        sample_indices=sample_indices,
        query_field=args.query_field,
        answer_field=args.answer_field,
        prediction_field=args.prediction_field,
        media_field=args.media_field,
        judge=judge,
        renderer=renderer,
        judge_retries=args.judge_retries,
        score_threshold=args.score_threshold,
        min_tag_ratio=args.min_tag_ratio,
        min_structural_jaccard=args.min_structural_jaccard,
        min_text_recall=args.min_text_recall,
        require_css=not args.no_require_css,
        require_rick_image=not args.no_require_rick_image,
        keep_rendered_images=args.keep_rendered_images,
        default_viewport=default_viewport,
        num_workers=args.num_workers,
        limit=args.limit,
    )

    total = len(results)
    rule_passed = sum(1 for result in results if result["rule_passed"])
    judged = sum(1 for result in results if result["rendered_image"] and result["judge_raw"])
    correct = sum(1 for result in results if result["correct"])
    avg_score = sum(float(result["score"]) for result in results) / total if total else 0.0
    print(
        f"\nTotal: {total}, Rule passed: {rule_passed}, Judged: {judged}, "
        f"Correct@{args.score_threshold:g}: {correct}, Avg score: {avg_score:.2f}"
        if total
        else "No results."
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
