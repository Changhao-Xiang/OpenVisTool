"""Paired tool-trajectory ablation on agentic-vision evaluation traces.

For every test item this script evaluates one frozen base model under three
conditions: no trajectory, a trajectory from the correctness+instructive
checkpoint, and a trajectory from the correctness-only checkpoint.  Each
condition gets the same k rollout seeds and the output is one resumable JSONL
record per item.

The trajectory prefix intentionally matches filter_tool_gain_with_prefix.py:
tool-call arguments and tool responses are retained, while assistant reasoning
and the trained model's final natural-language answer are omitted.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import random
import re
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.filter.evaluate_answer_rule import check_match
from distill.filter.evaluate_html_vlm_judge import HtmlRenderer

DEFAULT_RETRIES = 3
DEFAULT_PREFIX_CHAR_LIMIT = 10_000
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
VIRTUAL_ROOT = "/mnt/data"
VIRTUAL_IMAGE_RE = re.compile(
    r"/mnt/data/[^\s\"'<>]+?\.(?:jpg|jpeg|png|bmp|gif|webp|tif|tiff)",
    re.IGNORECASE,
)
ABSOLUTE_IMAGE_RE = re.compile(
    r"/(?:[^\s\"'<>/]+/)+[^\s\"'<>]+?\.(?:jpg|jpeg|png|bmp|gif|webp|tif|tiff)",
    re.IGNORECASE,
)
BARE_IMAGE_RE = re.compile(
    r"(?<![/\w])([A-Za-z0-9._-]+\.(?:jpg|jpeg|png|bmp|gif|webp|tif|tiff))",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)

CONDITIONS = ("no_tool", "correctness_instructive", "correctness_only")
TOOL_TRAJECTORY_CONDITIONS = ("correctness_instructive", "correctness_only")
PREFIX_HEADER = (
    "Below are intermediate tool observations produced while solving the task. "
    "Treat them as given facts. Do not call any tools; answer the question directly."
)
PREFIX_FOOTER = "Based on the observations above, give the final answer to the original question."
GUI_PREFIX_HEADER = (
    "Below are intermediate tool observations produced while locating the requested UI element. "
    "Treat them as given evidence. No external tools are available now; synthesize the final click "
    "from these observations and follow the system output protocol exactly."
)
GUI_PREFIX_FOOTER = (
    "Based on the observations above, return exactly one `computer_use` left_click call for the "
    "original request, using normalized 0-1000 coordinates."
)

VISION2WEB_SYSTEM_PROMPT = """You are a senior front-end engineer with expertise in static responsive web development.

Build one production-quality static webpage that visually matches the provided desktop, tablet, and mobile prototype screenshots. Use the prototypes as the primary reference for layout, typography, spacing, colors, images, navigation, content sections, and responsive behavior.

## Assets
- Match the visual content with relative asset paths under `resources/` whenever possible.
- Do not use `desktop.jpg`, `tablet.jpg`, or `mobile.jpg` as page content images.
- Do not require a backend, build step, CDN, remote URL, or external API.

## Final Answer Protocol
- Return one complete HTML document with inline CSS and minimal inline JavaScript if needed.
- Start with `<!DOCTYPE html>` and end with `</html>`.
- Do not include markdown fences, explanations, paths, or deployment notes.
"""

VISION2WEB_JUDGE_PROMPT = """You are a senior QA automation engineer for visual website development.

Compare two images:
1. Prototype image: the target webpage design for a specific device viewport.
2. Actual page image: a screenshot rendered from the submitted implementation.

Evaluate visual fidelity using logical UI components/blocks. Segment the page into meaningful blocks such as header/navigation, hero, content cards, product/list sections, forms, media areas, footer, and any other visually distinct functional sections. Do not make components too tiny.

For each component, assign one score from this set only: 0, 0.25, 0.5, 0.75, 1.

Scoring rubric:
- 1.0: Perfect match. Position, layout, spacing, alignment, size, text, fonts, colors, icons, images, and media match the prototype with no visible differences.
- 0.75: Minor imperfections. Mostly accurate with small alignment, spacing, typography, color, or media differences.
- 0.5: Partial match. Component is recognizable but has noticeable mismatches.
- 0.25: Poor match. Component is present but strongly misaligned, incomplete, or visually inconsistent.
- 0.0: No match. Component is missing, unrelated, or completely misplaced.

Focus on the current device viewport: {device}. Penalize responsive layout mistakes visible in this viewport. If the actual page is blank, broken, or mostly unrelated, all component scores should be 0.

Output only a JSON array with objects containing `name`, `score`, and `reason`.
"""

DEFAULT_VIEWPORTS = {
    "desktop": (1920, 1080),
    "tablet": (1024, 768),
    "mobile": (375, 812),
}


@dataclass(frozen=True)
class ToolObservation:
    name: str
    call: str
    response: str
    images: tuple[Path, ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object in {path} line {line_no}")
        records.append(record)
    return records


def iter_jsonl_allow_partial_tail(path: Path) -> Iterable[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_no == len(lines):
                return
            raise ValueError(f"Invalid JSON in {path} line {line_no}")
        if isinstance(record, dict):
            yield record


def task_key(item: dict[str, Any]) -> str:
    source = str(item.get("source_dataset") or "dataset")
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_") or "dataset"
    return f"{safe_source}_{item['id']}"


def item_key(domain: str, item: dict[str, Any], index: int) -> str:
    return f"{domain}:{item.get('source_dataset', 'dataset')}:{item.get('id', index)}"


def trace_path(run_dir: Path, item: dict[str, Any]) -> Path:
    path = run_dir / f"task_{task_key(item)}" / "trace.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing agentic trace: {path}")
    return path


def truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def stringify_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [stringify_result(part) for part in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text", ""))
        if value.get("type") == "image_url":
            return "[image output]"
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def contains_image_part(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            return True
        return any(contains_image_part(part) for part in value.values())
    if isinstance(value, list):
        return any(contains_image_part(part) for part in value)
    return False


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for part in value.values():
            yield from iter_strings(part)
    elif isinstance(value, list):
        for part in value:
            yield from iter_strings(part)


def resolve_trace_image(raw_path: str, task_dir: Path) -> Path | None:
    raw_path = raw_path.strip().strip("\"'").rstrip(".,);]")
    if not raw_path:
        return None
    if raw_path.startswith(VIRTUAL_ROOT):
        candidate = task_dir / raw_path[len(VIRTUAL_ROOT) :].lstrip("/")
    else:
        path = Path(raw_path).expanduser()
        candidate = path if path.is_absolute() else task_dir / path
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
        return candidate
    return None


def result_images(result: Any, task_dir: Path) -> tuple[Path, ...]:
    if not contains_image_part(result):
        return ()
    candidates: list[str] = []
    for text in iter_strings(result):
        candidates.extend(VIRTUAL_IMAGE_RE.findall(text))
        candidates.extend(ABSOLUTE_IMAGE_RE.findall(text))
        candidates.extend(BARE_IMAGE_RE.findall(text))
    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = resolve_trace_image(candidate, task_dir)
        if path is None or str(path) in seen:
            continue
        seen.add(str(path))
        resolved.append(path)
    return tuple(resolved)


def extract_agentic_prefix(
    path: Path,
    *,
    char_limit: int,
    excluded_tools: set[str],
) -> list[ToolObservation]:
    observations: list[ToolObservation] = []
    for turn in load_jsonl(path):
        message = turn.get("message") if isinstance(turn.get("message"), dict) else {}
        calls: dict[str, tuple[str, Any]] = {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            raw_args = function.get("arguments") or ""
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = raw_args
            calls[str(call.get("id") or "")] = (name, args)

        for result in turn.get("tool_results") or []:
            call_id = str(result.get("tool_call_id") or "")
            fallback_name, fallback_args = calls.get(call_id, ("", {}))
            name = str(result.get("name") or fallback_name)
            if name in excluded_tools:
                continue
            args = result.get("args", fallback_args)
            if "args_raw" in result and not args:
                args = result["args_raw"]
            args_text = (
                args
                if isinstance(args, str)
                else json.dumps(args, ensure_ascii=False, sort_keys=True)
            )
            response_value = result.get("result")
            response = stringify_result(response_value).strip()
            if not response and not args_text:
                continue
            call_text = f"{name}({args_text})" if name else ""
            observations.append(
                ToolObservation(
                    name=name,
                    call=truncate(call_text, char_limit),
                    response=truncate(response, char_limit),
                    images=result_images(response_value, path.parent),
                )
            )
    return observations


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{guess_mime_type(path)};base64,{encoded}"


def response_to_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif getattr(part, "type", None) == "text":
                parts.append(str(getattr(part, "text", "")))
        return "".join(parts).strip()
    return str(content or "").strip()


def resolve_item_images(dataset_path: Path, item: dict[str, Any]) -> list[Path]:
    source = str(item.get("source_dataset") or "")
    images_root = dataset_path.parent / "images"
    if source:
        images_root /= source
    resolved: list[Path] = []
    for raw in item.get("images") or []:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = images_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing input image: {path}")
        resolved.append(path)
    return resolved


def build_content(
    query: str,
    original_images: list[Path],
    observations: list[ToolObservation] | None,
    *,
    scorer: str = "judge",
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
        for path in original_images
    ]
    if observations is None:
        content.append({"type": "text", "text": query})
        return content

    content.append({"type": "text", "text": f"[Question]\n{query}"})
    header = GUI_PREFIX_HEADER if scorer == "bbox" else PREFIX_HEADER
    footer = GUI_PREFIX_FOOTER if scorer == "bbox" else PREFIX_FOOTER
    content.append({"type": "text", "text": header})
    for index, observation in enumerate(observations, 1):
        lines = [f"[tool_call #{index}]", observation.call, f"[tool_response #{index}]", observation.response]
        content.append({"type": "text", "text": "\n".join(line for line in lines if line)})
        for path in observation.images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    content.append({"type": "text", "text": footer})
    return content


class SeededRolloutClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        retries: int,
        data_parallel_size: int = 0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.retries = retries
        self.data_parallel_size = data_parallel_size
        self._local = threading.local()
        self._rank_lock = threading.Lock()
        self._next_rank = 0

    def client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            default_headers = None
            if self.data_parallel_size > 0:
                with self._rank_lock:
                    rank = self._next_rank % self.data_parallel_size
                    self._next_rank += 1
                default_headers = {"X-data-parallel-rank": str(rank)}
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=self.base_url,
                timeout=3600.0,
                default_headers=default_headers,
            )
            self._local.client = client
        return client

    def rollout(self, content: list[dict[str, Any]], system_prompt: str, seed: int) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    seed=seed,
                    extra_body={"top_k": self.top_k},
                )
                return response_to_text(response)
            except Exception as exc:  # noqa: BLE001 - endpoint errors are retried uniformly
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise last_error


class StrictAnswerJudge:
    """Answer-equivalence judge that never turns endpoint failures into NO."""

    SYSTEM_PROMPT = (
        "You are an answer judge. Decide whether the prediction correctly answers the question "
        "using the ground truth as reference. Respond with exactly one word only: YES or NO."
    )
    YES_RE = re.compile(r"\bYES\b", re.IGNORECASE)
    NO_RE = re.compile(r"\bNO\b", re.IGNORECASE)

    def __init__(self, *, model: str, base_url: str, api_key: str = "EMPTY") -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._local = threading.local()

    def client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=3600.0,
            )
            self._local.client = client
        return client

    @classmethod
    def parse(cls, content: str) -> bool | None:
        if not content:
            return None
        head = re.split(r"[\s.,:;!?]+", content.strip(), maxsplit=1)[0].upper()
        if head == "YES":
            return True
        if head == "NO":
            return False
        for line in reversed(content.splitlines()):
            if not line.strip():
                continue
            yes = bool(cls.YES_RE.search(line))
            no = bool(cls.NO_RE.search(line))
            if yes != no:
                return yes
            break
        return None

    def judge(self, prediction: str, answer: str, *, query: str, retries: int) -> bool:
        if not prediction or not answer:
            return False
        user = "\n\n".join(
            (f"question:\n{query}", f"ground_truth:\n{answer}", f"prediction:\n{prediction}")
        )
        last_error = "unparseable response"
        for attempt in range(retries):
            try:
                response = self.client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=16384,
                    temperature=1.0,
                    top_p=0.95,
                    presence_penalty=1.5,
                    extra_body={"top_k": 20},
                )
                verdict = self.parse(response_to_text(response))
                if verdict is not None:
                    return verdict
                last_error = "unparseable judge response"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(f"Judge failed after {retries} attempts: {last_error}")


def preflight_endpoint(
    *,
    base_url: str,
    model: str,
    label: str,
    api_key: str = "EMPTY",
    retries: int = 3,
) -> None:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=30.0,
            )
            available = {entry.id for entry in client.models.list().data}
            if model not in available:
                raise RuntimeError(f"model {model!r} not listed; available={sorted(available)}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"{label} endpoint preflight failed: {last_error}")


def strip_reasoning(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    return UNCLOSED_THINK_RE.sub("", text).strip()


def extract_html(text: str) -> str:
    text = strip_reasoning(text)
    fenced = re.findall(r"```(?:html)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidates = [part.strip() for part in fenced if "<" in part and ">" in part]
    if candidates:
        return max(candidates, key=len)
    lower = text.lower()
    starts = [lower.find(marker) for marker in ("<!doctype html", "<html", "<body", "<div")]
    starts = [value for value in starts if value >= 0]
    return text[min(starts) :].strip() if starts else ""


def parse_component_score(raw: str) -> tuple[float | None, list[dict[str, Any]]]:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return None, []
    try:
        values = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, []
    if not isinstance(values, list):
        return None, []
    valid = {0.0, 0.25, 0.5, 0.75, 1.0}
    components: list[dict[str, Any]] = []
    scores: list[float] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            score = float(value.get("score"))
        except (TypeError, ValueError):
            continue
        score = min(valid, key=lambda candidate: abs(candidate - score))
        components.append(
            {
                "name": str(value.get("name", "component")),
                "score": score,
                "reason": str(value.get("reason", "")),
            }
        )
        scores.append(score)
    return (sum(scores) / len(scores), components) if scores else (None, components)


class Vision2WebScorer:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        temperature: float | None,
        max_tokens: int,
        retries: int,
        render_timeout_ms: int,
        render_wait_ms: int,
        render_slots: int,
        max_tokens_parameter: str = "max_tokens",
    ) -> None:
        if max_tokens_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"Unsupported token parameter: {max_tokens_parameter}")
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_parameter = max_tokens_parameter
        self.retries = retries
        self.renderer = HtmlRenderer(timeout_ms=render_timeout_ms, wait_ms=render_wait_ms)
        self.render_semaphore = threading.BoundedSemaphore(render_slots)
        self._local = threading.local()

    def client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=self.base_url,
                timeout=3600.0,
            )
            self._local.client = client
        return client

    def judge_viewport(self, reference: Path, generated: Path, device: str) -> tuple[float, dict[str, Any]]:
        content = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(reference)}},
            {"type": "image_url", "image_url": {"url": image_to_data_url(generated)}},
            {"type": "text", "text": f"Compare the prototype and actual page for the {device} viewport. Return only the JSON array."},
        ]
        last_raw = ""
        for attempt in range(self.retries):
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": VISION2WEB_JUDGE_PROMPT.format(device=device)},
                        {"role": "user", "content": content},
                    ],
                    self.max_tokens_parameter: self.max_tokens,
                }
                if self.temperature is not None:
                    request["temperature"] = self.temperature
                response = self.client().chat.completions.create(**request)
                last_raw = response_to_text(response)
                score, components = parse_component_score(last_raw)
                if score is not None:
                    return score * 100.0, {
                        "score": round(score * 100.0, 4),
                        "components": components,
                        "judge_raw": last_raw[:500],
                    }
            except Exception as exc:  # noqa: BLE001
                last_raw = f"judge_error:{exc}"
            if attempt < self.retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(
            f"Vision2Web judge failed for {device} after {self.retries} attempts: {last_raw[:500]}"
        )

    @staticmethod
    def viewports(asset_root: Path) -> dict[str, tuple[int, int]]:
        viewports = dict(DEFAULT_VIEWPORTS)
        workflow = asset_root / "workflow.json"
        if workflow.is_file():
            try:
                entries = json.loads(workflow.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                entries = []
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    device = str(entry.get("summary") or "").strip().lower()
                    resolution = entry.get("resolution") or {}
                    if device in viewports and {"width", "height"} <= set(resolution):
                        viewports[device] = (int(resolution["width"]), int(resolution["height"]))
        return viewports

    @staticmethod
    def add_base_href(html_text: str, asset_root: Path) -> str:
        base = f'<base href="{html.escape(asset_root.resolve().as_uri(), quote=True)}/">'
        head = re.search(r"<head\b[^>]*>", html_text, re.IGNORECASE)
        if head:
            return html_text[: head.end()] + base + html_text[head.end() :]
        return base + html_text

    def score(
        self,
        prediction: str,
        *,
        item: dict[str, Any],
        dataset_path: Path,
        original_images: list[Path],
    ) -> tuple[float, str, dict[str, Any]]:
        html_text = extract_html(prediction)
        if not html_text:
            return 0.0, "vision2web_no_html", {"viewports": {}}
        images_root = dataset_path.parent / "images" / str(item.get("source_dataset") or "")
        task_name = str(item.get("task_name") or Path(str(item.get("images", [""])[0])).parts[0])
        asset_root = (images_root / task_name).resolve()
        refs = {path.stem.lower(): path for path in original_images if path.stem.lower() in DEFAULT_VIEWPORTS}
        viewports = self.viewports(asset_root)
        details: dict[str, Any] = {}
        scores: list[float] = []
        with tempfile.TemporaryDirectory(prefix="openvistool_v2w_") as temp_dir:
            render_html = self.add_base_href(html_text, asset_root)
            generated: dict[str, Path] = {}
            with self.render_semaphore:
                for device, viewport in viewports.items():
                    output = Path(temp_dir) / f"generated_{device}.png"
                    try:
                        self.renderer.render(render_html, output, viewport)
                    except Exception as exc:  # noqa: BLE001
                        details[device] = {"score": 0.0, "render_error": str(exc)}
                        continue
                    generated[device] = output
            for device in DEFAULT_VIEWPORTS:
                if device not in refs or device not in generated:
                    continue
                score, detail = self.judge_viewport(refs[device], generated[device], device)
                scores.append(score)
                details[device] = detail
        raw_score = sum(scores) / len(scores) if scores else 0.0
        return raw_score / 100.0, "vision2web_component_judge", {
            "raw_score_0_100": round(raw_score, 4),
            "viewports": details,
        }


def prefix_stats(observations: list[ToolObservation], answer: str) -> dict[str, Any]:
    text = "\n".join(f"{obs.call}\n{obs.response}" for obs in observations)
    normalized_answer = re.sub(r"\s+", " ", answer.strip().lower())
    normalized_text = re.sub(r"\s+", " ", text.lower())
    answer_overlap = bool(len(normalized_answer) >= 3 and normalized_answer in normalized_text)
    return {
        "tool_responses": len(observations),
        "images": sum(len(obs.images) for obs in observations),
        "text_chars": len(text),
        "tool_counts": dict(sorted(Counter(obs.name for obs in observations).items())),
        "answer_literal_in_prefix": answer_overlap,
        "answer_channel_tools": sum(
            obs.name in {"computer_use", "write_file", "edit_file"} for obs in observations
        ),
    }


def condition_order(
    conditions: tuple[str, ...], index: int, rollout_index: int
) -> tuple[str, ...]:
    shift = (index + rollout_index) % len(conditions)
    return conditions[shift:] + conditions[:shift]


def process_item(
    item: dict[str, Any],
    *,
    index: int,
    args: argparse.Namespace,
    dataset_path: Path,
    rollout_client: SeededRolloutClient,
    answer_judge: StrictAnswerJudge | None,
    html_scorer: Vision2WebScorer | None,
    system_prompt: str,
    excluded_tools: set[str],
) -> dict[str, Any]:
    query = str(item.get("query") or item.get("problem") or "").strip()
    answer = str(item.get("answer") or "").strip()
    if not query or not answer:
        raise ValueError(f"Item {index} is missing query or answer")
    original_images = resolve_item_images(dataset_path, item)
    prefix_ic = extract_agentic_prefix(
        trace_path(Path(args.ic_run_dir), item),
        char_limit=args.per_response_char_limit,
        excluded_tools=excluded_tools,
    )
    prefix_acc = extract_agentic_prefix(
        trace_path(Path(args.acc_run_dir), item),
        char_limit=args.per_response_char_limit,
        excluded_tools=excluded_tools,
    )
    prefixes = {
        "no_tool": None,
        "correctness_instructive": prefix_ic,
        "correctness_only": prefix_acc,
    }
    active_conditions = args.conditions
    contents = {
        condition: build_content(
            query,
            original_images,
            prefix if prefix else None,
            scorer=args.scorer,
        )
        for condition, prefix in prefixes.items()
        if condition in active_conditions
    }
    state: dict[str, dict[str, Any]] = {
        condition: {"predictions": [], "scores": [], "methods": [], "score_details": []}
        for condition in active_conditions
    }

    for rollout_index in range(args.k):
        seed = args.seed + index * args.k + rollout_index
        for condition in condition_order(active_conditions, index, rollout_index):
            prediction = rollout_client.rollout(contents[condition], system_prompt, seed)
            if args.scorer == "judge":
                assert answer_judge is not None
                score = float(
                    answer_judge.judge(
                        prediction,
                        answer,
                        query=query,
                        retries=args.judge_retries,
                    )
                )
                method = "judge_llm"
                detail: dict[str, Any] = {}
            elif args.scorer in {"rule", "bbox"}:
                image_size = None
                match_mode = "generic"
                if args.scorer == "bbox":
                    match_mode = "bbox"
                    image_size = item.get("img_size")
                    if not image_size and original_images:
                        with Image.open(original_images[0]) as image:
                            image_size = list(image.size)
                matched, method = check_match(
                    prediction,
                    answer,
                    tolerance=args.tolerance,
                    mode=match_mode,
                    image_size=tuple(image_size) if image_size else None,
                )
                score = float(matched)
                detail = {}
            else:
                assert html_scorer is not None
                score, method, detail = html_scorer.score(
                    prediction,
                    item=item,
                    dataset_path=dataset_path,
                    original_images=original_images,
                )
            state[condition]["predictions"].append(prediction)
            state[condition]["scores"].append(round(score, 6))
            state[condition]["methods"].append(method)
            state[condition]["score_details"].append(detail)

    for condition in active_conditions:
        scores = state[condition]["scores"]
        state[condition]["avg_k"] = round(sum(scores) / len(scores), 6)
        state[condition]["correct_count"] = sum(score >= 0.5 for score in scores) if args.scorer != "vision2web" else None

    gains: dict[str, float] = {}
    if "no_tool" in state:
        no_tool = state["no_tool"]["avg_k"]
        for condition in TOOL_TRAJECTORY_CONDITIONS:
            if condition in state:
                gains[condition] = round(state[condition]["avg_k"] - no_tool, 6)
    if all(condition in state for condition in TOOL_TRAJECTORY_CONDITIONS):
        gains["paired_delta"] = round(
            state["correctness_instructive"]["avg_k"]
            - state["correctness_only"]["avg_k"],
            6,
        )
    return {
        "key": item_key(args.domain, item, index),
        "domain": args.domain,
        "index": index,
        "id": item.get("id", index),
        "source_dataset": item.get("source_dataset"),
        "query": query,
        "answer": answer,
        "scorer": args.scorer,
        "score_scale": "0_to_1",
        "conditions": state,
        "gains": gains,
        "prefix": {
            **(
                {"correctness_instructive": prefix_stats(prefix_ic, answer)}
                if "correctness_instructive" in active_conditions
                else {}
            ),
            **(
                {"correctness_only": prefix_stats(prefix_acc, answer)}
                if "correctness_only" in active_conditions
                else {}
            ),
            "excluded_tools": sorted(excluded_tools),
        },
        "traces": {
            "correctness_instructive": str(trace_path(Path(args.ic_run_dir), item)),
            "correctness_only": str(trace_path(Path(args.acc_run_dir), item)),
        },
        "trajectory_sources": {
            "correctness_instructive": {
                "checkpoint": args.ic_checkpoint_label,
                "run_dir": args.ic_run_dir,
            },
            "correctness_only": {
                "checkpoint": args.acc_checkpoint_label,
                "run_dir": args.acc_run_dir,
            },
        },
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def metric_vector(records: list[dict[str, Any]], metric: str) -> list[float]:
    if metric.startswith("condition:"):
        condition = metric.split(":", 1)[1]
        return [float(record["conditions"][condition]["avg_k"]) for record in records]
    return [float(record["gains"][metric]) for record in records]


def bootstrap_ci(records: list[dict[str, Any]], metric: str, *, samples: int, seed: int) -> list[float]:
    if not records:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled = [records[rng.randrange(len(records))] for _ in records]
        estimates.append(mean(metric_vector(sampled, metric)))
    return [round(percentile(estimates, 0.025), 6), round(percentile(estimates, 0.975), 6)]


def summarize_group(records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    candidate_metrics = {
        "avg_no_tool": "condition:no_tool",
        "avg_correctness_instructive": "condition:correctness_instructive",
        "avg_correctness_only": "condition:correctness_only",
        "gain_correctness_instructive": "correctness_instructive",
        "gain_correctness_only": "correctness_only",
        "paired_delta": "paired_delta",
    }
    common_conditions = set.intersection(*(set(record["conditions"]) for record in records))
    common_gains = set.intersection(*(set(record["gains"]) for record in records))
    metrics: dict[str, str] = {}
    for name, metric in candidate_metrics.items():
        available = (
            metric.split(":", 1)[1] in common_conditions
            if metric.startswith("condition:")
            else metric in common_gains
        )
        if available:
            metrics[name] = metric
    output: dict[str, Any] = {"n": len(records)}
    for offset, (name, metric) in enumerate(metrics.items()):
        output[name] = round(mean(metric_vector(records, metric)), 6)
        output[f"{name}_ci95"] = bootstrap_ci(
            records,
            metric,
            samples=bootstrap_samples,
            seed=seed + offset,
        )
    if "paired_delta" in common_gains:
        deltas = metric_vector(records, "paired_delta")
        output["delta_positive_rate"] = round(mean(value > 0 for value in deltas), 6)
        output["delta_negative_rate"] = round(mean(value < 0 for value in deltas), 6)
        output["delta_tie_rate"] = round(mean(value == 0 for value in deltas), 6)
    output["prefix_audit"] = {}
    for condition in TOOL_TRAJECTORY_CONDITIONS:
        if all(condition in record["prefix"] for record in records):
            audits = [record["prefix"][condition] for record in records]
            output["prefix_audit"][condition] = {
                "mean_tool_responses": round(mean(audit["tool_responses"] for audit in audits), 4),
                "mean_images": round(mean(audit["images"] for audit in audits), 4),
                "mean_text_chars": round(mean(audit["text_chars"] for audit in audits), 4),
                "answer_literal_rate": round(mean(audit["answer_literal_in_prefix"] for audit in audits), 4),
                "answer_channel_tools_mean": round(mean(audit["answer_channel_tools"] for audit in audits), 4),
            }
    return output


def stratified_bootstrap(records: list[dict[str, Any]], *, samples: int, seed: int) -> dict[str, list[float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["domain"])].append(record)
    rng = random.Random(seed)
    micro_values: list[float] = []
    macro_values: list[float] = []
    for _ in range(samples):
        domain_means: list[float] = []
        pooled: list[float] = []
        for domain_records in groups.values():
            sampled = [domain_records[rng.randrange(len(domain_records))] for _ in domain_records]
            deltas = metric_vector(sampled, "paired_delta")
            domain_means.append(mean(deltas))
            pooled.extend(deltas)
        micro_values.append(mean(pooled))
        macro_values.append(mean(domain_means))
    return {
        "micro_ci95": [round(percentile(micro_values, 0.025), 6), round(percentile(micro_values, 0.975), 6)],
        "macro_ci95": [round(percentile(macro_values, 0.025), 6), round(percentile(macro_values, 0.975), 6)],
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall_micro"]
    pair_only = "avg_no_tool" not in overall
    if pair_only:
        lines = [
            "# Instructive Trajectory Pair-Only Repeat",
            "",
            "All scores are normalized to [0, 1]. Delta is correctness+instructive minus correctness-only on paired items and rollout seeds.",
            "",
            "| Domain | N | Corr. + instr. | Corr. only | Delta | 95% CI(delta) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for domain, values in summary["domains"].items():
            ci = values["paired_delta_ci95"]
            lines.append(
                f"| {domain} | {values['n']} | {values['avg_correctness_instructive']:.4f} | "
                f"{values['avg_correctness_only']:.4f} | {values['paired_delta']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )
        ci = overall["paired_delta_ci95"]
        lines.append(
            f"| **Overall micro** | **{overall['n']}** | "
            f"**{overall['avg_correctness_instructive']:.4f}** | "
            f"**{overall['avg_correctness_only']:.4f}** | "
            f"**{overall['paired_delta']:+.4f}** | **[{ci[0]:+.4f}, {ci[1]:+.4f}]** |"
        )
    else:
        lines = [
            "# Instructive Trajectory Ablation",
            "",
            "All scores are normalized to [0, 1]. Gains are relative to the paired no-tool avg@k baseline.",
            "",
            "| Domain | N | No tool | Corr. + instr. | Corr. only | Gain C+I | Gain C | Delta | 95% CI(delta) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for domain, values in summary["domains"].items():
            ci = values["paired_delta_ci95"]
            lines.append(
                f"| {domain} | {values['n']} | {values['avg_no_tool']:.4f} | "
                f"{values['avg_correctness_instructive']:.4f} | {values['avg_correctness_only']:.4f} | "
                f"{values['gain_correctness_instructive']:+.4f} | {values['gain_correctness_only']:+.4f} | "
                f"{values['paired_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )
        ci = overall["paired_delta_ci95"]
        lines.append(
            f"| **Overall micro** | **{overall['n']}** | **{overall['avg_no_tool']:.4f}** | "
            f"**{overall['avg_correctness_instructive']:.4f}** | **{overall['avg_correctness_only']:.4f}** | "
            f"**{overall['gain_correctness_instructive']:+.4f}** | **{overall['gain_correctness_only']:+.4f}** | "
            f"**{overall['paired_delta']:+.4f}** | **[{ci[0]:+.4f}, {ci[1]:+.4f}]** |"
        )
    lines.extend(
        [
            "",
            "## Primary Estimand",
            "",
            f"- Micro paired delta: {summary['overall_micro']['paired_delta']:+.6f}; stratified 95% CI "
            f"{summary['stratified_bootstrap']['micro_ci95']}.",
            f"- Macro paired delta: {summary['overall_macro']['paired_delta']:+.6f}; stratified 95% CI "
            f"{summary['stratified_bootstrap']['macro_ci95']}.",
            "- A positive delta means the correctness+instructive model's trajectory helps the same frozen base model more than the correctness-only model's trajectory.",
            "",
            "## Interpretation Guardrails",
            "",
            "- The prefix excludes assistant reasoning and the trained model's final natural-language answer, but retains tool names, arguments, responses, and tool-produced images to match the data-filtering metric.",
            "- GUI `computer_use` arguments and Vision2Web file-edit arguments are answer-bearing channels. Use the prefix audit in `summary.json` and, when needed, rerun with `--exclude-tools` as a robustness check.",
            "- The causal comparison is paired by item and checkpoint training step; it does not by itself separate trajectory quality from differences in the trained models' direct task competence.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_summary(args: argparse.Namespace) -> None:
    inputs: list[Path] = []
    for raw in args.inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            inputs.extend(sorted(path.glob("*.jsonl")))
        else:
            inputs.append(path)
    records: list[dict[str, Any]] = []
    for path in inputs:
        records.extend(load_jsonl(path))
    if not records:
        raise ValueError("No ablation records found")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["domain"])].append(record)
    domains = {
        domain: summarize_group(values, bootstrap_samples=args.bootstrap_samples, seed=args.seed + index * 100)
        for index, (domain, values) in enumerate(sorted(groups.items()))
    }
    overall_micro = summarize_group(records, bootstrap_samples=args.bootstrap_samples, seed=args.seed + 10_000)
    metric_names = (
        "avg_no_tool",
        "avg_correctness_instructive",
        "avg_correctness_only",
        "gain_correctness_instructive",
        "gain_correctness_only",
        "paired_delta",
    )
    macro_metrics = {
        metric: round(mean(domain[metric] for domain in domains.values()), 6)
        for metric in metric_names
        if all(metric in domain for domain in domains.values())
    }
    summary = {
        "n": len(records),
        "domains": domains,
        "overall_micro": overall_micro,
        "overall_macro": {"n_domains": len(domains), **macro_metrics},
        "stratified_bootstrap": stratified_bootstrap(
            records,
            samples=args.bootstrap_samples,
            seed=args.seed + 20_000,
        ),
        "inputs": [str(path) for path in inputs],
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(render_summary_markdown(summary), encoding="utf-8")
    print(f"Summary JSON: {output}")
    print(f"Summary Markdown: {markdown}")
    print(f"Primary micro delta: {overall_micro['paired_delta']:+.6f}")


def run_evaluate(args: argparse.Namespace) -> None:
    if args.k <= 0 or args.num_workers <= 0:
        raise ValueError("--k and --num-workers must be positive")
    if args.rollout_data_parallel_size < 0:
        raise ValueError("--rollout-data-parallel-size must be non-negative")
    conditions = tuple(name.strip() for name in args.conditions.split(",") if name.strip())
    invalid_conditions = set(conditions) - set(CONDITIONS)
    if not conditions:
        raise ValueError("--conditions must select at least one condition")
    if invalid_conditions:
        raise ValueError(f"Unknown --conditions: {sorted(invalid_conditions)}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("--conditions must not contain duplicates")
    args.conditions = conditions
    if args.judge_general_config:
        general_config = json.loads(
            Path(args.judge_general_config).expanduser().resolve().read_text(encoding="utf-8")
        )
        judge_config = general_config.get("judge") or {}
        args.judge_model = str(judge_config.get("model") or args.judge_model)
        args.judge_api_base = str(judge_config.get("base_url") or args.judge_api_base)
        args.judge_api_key = str(judge_config.get("api_key") or args.judge_api_key)
    dataset_path = Path(args.dataset).expanduser().resolve()
    ic_run_dir = Path(args.ic_run_dir).expanduser().resolve()
    acc_run_dir = Path(args.acc_run_dir).expanduser().resolve()
    for path in (dataset_path, ic_run_dir, acc_run_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    args.ic_run_dir = str(ic_run_dir)
    args.acc_run_dir = str(acc_run_dir)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_records = list(iter_jsonl_allow_partial_tail(output)) if output.is_file() else []
    for record in existing_records:
        if tuple(record.get("conditions", {})) != conditions:
            raise ValueError(
                f"Existing output has different conditions: {tuple(record.get('conditions', {}))}"
            )
    processed = {str(record.get("key")) for record in existing_records}
    items = load_jsonl(dataset_path)
    indexed_items = list(enumerate(items))
    if args.sample_range:
        parts = [int(part) if part else None for part in args.sample_range.split(":")]
        indexed_items = indexed_items[slice(*parts)]
    pending = [
        (index, item)
        for index, item in indexed_items
        if item_key(args.domain, item, index) not in processed
    ]
    excluded_tools = {name.strip() for name in args.exclude_tools.split(",") if name.strip()}
    preflight_endpoint(
        base_url=args.rollout_api_base,
        model=args.model,
        label="rollout",
    )
    if args.scorer in {"judge", "vision2web"}:
        if not args.judge_model or not args.judge_api_base:
            raise ValueError(
                f"--judge-model and --judge-api-base are required for --scorer {args.scorer}"
            )
        preflight_endpoint(
            base_url=args.judge_api_base,
            model=args.judge_model,
            label="judge",
            api_key=args.judge_api_key,
        )
    system_prompt = (
        VISION2WEB_SYSTEM_PROMPT
        if args.scorer == "vision2web"
        else Path(args.system_prompt_md).read_text(encoding="utf-8").strip()
        if args.system_prompt_md
        else "You are a helpful assistant."
    )
    rollout_client = SeededRolloutClient(
        model=args.model,
        base_url=args.rollout_api_base,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        retries=args.rollout_retries,
        data_parallel_size=args.rollout_data_parallel_size,
    )
    answer_judge = (
        StrictAnswerJudge(
            model=args.judge_model,
            base_url=args.judge_api_base,
            api_key=args.judge_api_key,
        )
        if args.scorer == "judge"
        else None
    )
    html_scorer = (
        Vision2WebScorer(
            model=args.judge_model,
            base_url=args.judge_api_base,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            retries=args.judge_retries,
            render_timeout_ms=args.render_timeout_ms,
            render_wait_ms=args.render_wait_ms,
            render_slots=args.render_slots,
        )
        if args.scorer == "vision2web"
        else None
    )
    print(
        f"Domain={args.domain} scorer={args.scorer} items={len(indexed_items)} "
        f"resumed={len(indexed_items) - len(pending)} pending={len(pending)}\n"
        f"Base={args.model} @ {args.rollout_api_base}; k={args.k}; workers={args.num_workers}; "
        f"rollout_dp={args.rollout_data_parallel_size or 'auto'}\n"
        f"Conditions={','.join(args.conditions)}\n"
        f"IC traces={ic_run_dir}\nACC traces={acc_run_dir}\nOutput={output}"
    )
    if not pending:
        return
    failures = 0
    with output.open("a", encoding="utf-8", buffering=1) as output_file, ThreadPoolExecutor(
        max_workers=min(args.num_workers, len(pending))
    ) as executor:
        futures = {
            executor.submit(
                process_item,
                item,
                index=index,
                args=args,
                dataset_path=dataset_path,
                rollout_client=rollout_client,
                answer_judge=answer_judge,
                html_scorer=html_scorer,
                system_prompt=system_prompt,
                excluded_tools=excluded_tools,
            ): (index, item)
            for index, item in pending
        }
        with tqdm(total=len(futures), desc=f"Ablation {args.domain}", unit="item") as progress:
            for future in as_completed(futures):
                index, item = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    print(f"[error] index={index} id={item.get('id')}: {type(exc).__name__}: {exc}")
                else:
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                progress.update(1)
                progress.set_postfix(failures=failures)
    if failures:
        raise RuntimeError(f"{failures} items failed; rerun the same command to resume")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="Run one domain's paired ablation")
    evaluate.add_argument("--domain", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--ic-run-dir", required=True)
    evaluate.add_argument("--acc-run-dir", required=True)
    evaluate.add_argument("--ic-checkpoint-label", default="")
    evaluate.add_argument("--acc-checkpoint-label", default="")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--scorer",
        "--accuracy-backend",
        dest="scorer",
        choices=("rule", "judge", "bbox", "vision2web"),
        required=True,
    )
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--rollout-api-base", required=True)
    evaluate.add_argument(
        "--rollout-data-parallel-size",
        type=int,
        default=0,
        help=(
            "If positive, distribute worker-local rollout clients round-robin with "
            "vLLM's X-data-parallel-rank header. Zero leaves routing to the server."
        ),
    )
    evaluate.add_argument("--judge-model", default="")
    evaluate.add_argument("--judge-api-base", default="")
    evaluate.add_argument("--judge-api-key", default="EMPTY")
    evaluate.add_argument(
        "--judge-general-config",
        default="",
        help="Load judge model, base_url, and api_key from a general_config.json file.",
    )
    evaluate.add_argument("--system-prompt-md", default="")
    evaluate.add_argument("--k", type=int, default=4)
    evaluate.add_argument("--temperature", type=float, default=0.7)
    evaluate.add_argument("--top-p", type=float, default=0.95)
    evaluate.add_argument("--top-k", type=int, default=20)
    evaluate.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Relative numeric tolerance for --accuracy-backend rule.",
    )
    evaluate.add_argument("--max-tokens", type=int, default=16384)
    evaluate.add_argument("--num-workers", type=int, default=32)
    evaluate.add_argument("--seed", type=int, default=20260722)
    evaluate.add_argument("--sample-range", default="")
    evaluate.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated subset of no_tool,correctness_instructive,correctness_only",
    )
    evaluate.add_argument("--per-response-char-limit", type=int, default=DEFAULT_PREFIX_CHAR_LIMIT)
    evaluate.add_argument("--exclude-tools", default="")
    evaluate.add_argument("--rollout-retries", type=int, default=DEFAULT_RETRIES)
    evaluate.add_argument("--judge-retries", type=int, default=DEFAULT_RETRIES)
    evaluate.add_argument("--judge-temperature", type=float, default=0.6)
    evaluate.add_argument("--judge-max-tokens", type=int, default=16384)
    evaluate.add_argument("--render-timeout-ms", type=int, default=30_000)
    evaluate.add_argument("--render-wait-ms", type=int, default=500)
    evaluate.add_argument("--render-slots", type=int, default=4)

    summarize = subparsers.add_parser("summarize", help="Aggregate domain JSONL files")
    summarize.add_argument("--inputs", nargs="+", required=True)
    summarize.add_argument("--output", required=True)
    summarize.add_argument("--bootstrap-samples", type=int, default=10_000)
    summarize.add_argument("--seed", type=int, default=20260722)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        run_evaluate(args)
    else:
        run_summary(args)


if __name__ == "__main__":
    main()
