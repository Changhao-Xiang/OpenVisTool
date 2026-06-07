"""Compute avg@k with teacher tool_response prefix.

Given a dataset JSONL and a directory of teacher sessions produced by
`distill/run.py`, this script:

1. For each dataset item, loads the teacher session trace and extracts every
   `role == "tool"` response (text content + any images the tool produced).
   Tool-call metadata (tool_name, arguments) is intentionally dropped.
2. Builds a single user message containing the original query, the original
   images, and every teacher tool_response (text + image_url). No system
   tool_call protocol is used — the base model sees the tool outputs as
   observations and is asked to answer directly.
3. Performs k no-tool rollouts through an OpenAI-compatible API, then scores
   each rollout with the chosen accuracy backend (rule / judge / html_vlm) to
   obtain avg@k_with_prefix.
4. Writes per-item metrics to output JSONL. No filtering happens here; this
   script only reports the metric.

Example:
    python distill/filter/filter_tool_gain_with_prefix.py \
        --dataset dataset/Chart/.../foo.jsonl \
        --sessions-dir workspaces/qwen35plus/ChartVerse-SFT \
        --model Qwen3-VL-30B-A3B-Instruct \
        --rollout-api-base https://YOUR_ROLLOUT_ENDPOINT/v1 \
        --k 4 --accuracy-backend judge \
        --judge-model Qwen3.5-27B \
        --judge-api-base https://YOUR_JUDGE_ENDPOINT/v1
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.filter.evaluate_answer_judge import JudgeLLM
from distill.filter.evaluate_answer_rule import check_match
from distill.filter.evaluate_html_vlm_judge import (
    HtmlRenderer,
    VlmHtmlJudge,
    _parse_viewport,
)
from distill.filter.filter_difficulty import evaluate_html_prediction

DEFAULT_QUERY_FALLBACKS = ("query", "problem")
DEFAULT_MEDIA_FALLBACKS = ("images",)
DEFAULT_RETRIES = 3
DEFAULT_TOOL_RESULT_MAX_CHARS = 4000
SANDBOX_ROOT = "/mnt/data"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
SANDBOX_PATH_RE = re.compile(
    r"/mnt/data/[^\s\"'<>]+?\.(?:jpg|jpeg|png|bmp|gif|webp|tif|tiff)",
    re.IGNORECASE,
)
# Bare filename fallback for tool_responses that mention an image without the
# /mnt/data/ prefix (e.g. read_file's "Image loaded: 0000004.jpg ...").
BARE_IMAGE_NAME_RE = re.compile(
    r"(?<![/\w])([A-Za-z0-9._-]+\.(?:jpg|jpeg|png|bmp|gif|webp|tif|tiff))",
    re.IGNORECASE,
)
# Marker nanobot inserts whenever a tool message carries an image payload.
IMAGE_VIEWED_MARKER = "[image viewed *"


# ---------------------------------------------------------------------------
# Dataset / IO (copied minimal helpers from filter_difficulty.py)
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _iter_jsonl_records_allow_partial_tail(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            if line_no == len(lines):
                break
            raise ValueError(f"Invalid JSON in {path} line {line_no}")
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object in {path} line {line_no}")
        records.append(record)
    return records


def parse_sample_range(expr: str) -> slice:
    raw_expr = expr.strip()
    if not raw_expr:
        return slice(None)
    if ":" not in raw_expr:
        try:
            index = int(raw_expr)
        except ValueError as exc:
            raise ValueError(f"Invalid --sample-range: {expr!r}") from exc
        return slice(index, index + 1)
    parts = raw_expr.split(":")
    if len(parts) > 3:
        raise ValueError("--sample-range must follow Python slice syntax")
    parsed: list[int | None] = []
    for part in parts:
        if not part:
            parsed.append(None)
            continue
        try:
            parsed.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid --sample-range component: {part!r}") from exc
    return slice(*parsed)


def get_text_field(item: dict[str, Any], field: str, fallbacks: tuple[str, ...] = ()) -> str:
    for candidate in (field, *fallbacks):
        value = item.get(candidate)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            return value
    raise ValueError(f"Missing non-empty text field '{field}'")


def resolve_media_paths(item: dict[str, Any], media_field: str, media_dir: Path) -> list[Path]:
    raw_value = item.get(media_field)
    if raw_value is None:
        for fallback in DEFAULT_MEDIA_FALLBACKS:
            raw_value = item.get(fallback)
            if raw_value is not None:
                break
    if raw_value is None:
        return []
    raw_paths = raw_value if isinstance(raw_value, list) else [raw_value]
    resolved: list[Path] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str):
            raise ValueError(f"Field '{media_field}' must be a string or list of strings")
        path = Path(raw_path.strip()).expanduser()
        if not path.is_absolute():
            path = (media_dir / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")
        resolved.append(path)
    return resolved


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{_guess_mime_type(path)};base64,{encoded}"


def load_system_prompt_markdown(path_str: str) -> tuple[str, Path | None]:
    raw_path = path_str.strip()
    if not raw_path:
        return "", None
    prompt_path = Path(raw_path).expanduser().resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"System prompt markdown file not found: {prompt_path}")
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"System prompt markdown file is empty: {prompt_path}")
    return system_prompt, prompt_path


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
            elif hasattr(part, "type") and getattr(part, "type") == "text":
                parts.append(str(getattr(part, "text", "")))
        return "".join(parts).strip()
    return str(content or "").strip()


def get_item_key(item: dict[str, Any], index: int) -> str:
    item_id = item.get("id")
    if item_id is not None:
        return f"id:{item_id}"
    return f"index:{index}"


# ---------------------------------------------------------------------------
# Teacher prefix extraction
# ---------------------------------------------------------------------------


def _find_session_trace(sessions_dir: Path, item_id: Any, suffix: str) -> Path | None:
    session_dir = sessions_dir / f"session_{item_id}{suffix}"
    if not session_dir.is_dir():
        return None
    files = list(session_dir.glob("*.jsonl"))
    return files[0] if files else None


_PATH_ARG_KEYS = ("path", "file_path", "image", "image_path", "filename", "file")


def _resolve_sandbox_image(sandbox_path: str, session_dir: Path) -> Path | None:
    """Map a teacher-supplied image reference to a real file in ``session_dir``.

    Accepts: ``/mnt/data/foo.jpg`` (absolute sandbox path), ``foo.jpg`` (bare
    filename — models sometimes pass relative paths to ``read_file``), or
    nested paths like ``/mnt/data/sub/foo.jpg``. Returns ``None`` if the
    resolved file does not exist or is not an image.
    """
    if not sandbox_path:
        return None
    raw = sandbox_path.strip().strip('"').strip("'")
    if not raw:
        return None
    if raw.startswith(SANDBOX_ROOT):
        relative = raw[len(SANDBOX_ROOT) :].lstrip("/")
    else:
        # Treat anything else as relative to the sandbox root.
        relative = raw.lstrip("/")
    candidate = (session_dir / relative).resolve()
    if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
        return candidate
    return None


def _stringify_tool_content(content: Any) -> str:
    """Tool messages from nanobot are usually strings; be defensive anyway."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(json.dumps(part, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def extract_teacher_prefix(
    session_path: Path,
    *,
    per_response_char_limit: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> list[tuple[str, str, list[Path]]]:
    """Return per-tool_response (call_str, response_text, images) tuples in order.

    - call_str: ``"<tool_name>(<arguments_json>)"`` recovered by matching each
      tool message's ``tool_call_id`` against the preceding assistant
      ``tool_calls``. Empty string if the call could not be matched.
    - response_text: the raw tool message content, truncated.
    - images: every ``/mnt/data/*.{ext}`` path mentioned in either the call
      arguments or the response, in encounter order, **without deduplication**.
      Matched against the session directory; unresolvable paths are skipped.
    """
    session_dir = session_path.parent
    records = _iter_jsonl_records_allow_partial_tail(session_path)

    # Build tool_call_id -> "name(args)" lookup from assistant messages.
    call_index: dict[str, tuple[str, str]] = {}
    for record in records:
        if record.get("role") != "assistant":
            continue
        for tc in record.get("tool_calls") or []:
            tc_id = tc.get("id")
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or ""
            if isinstance(args, (dict, list)):
                args = json.dumps(args, ensure_ascii=False)
            if tc_id:
                call_index[tc_id] = (name, str(args))

    prefix: list[tuple[str, str, list[Path]]] = []
    for record in records:
        if record.get("role") != "tool":
            continue
        response = _stringify_tool_content(record.get("content"))
        if not response:
            continue

        name, args = call_index.get(record.get("tool_call_id"), ("", ""))
        call_str = f"{name}({args})" if name else ""

        # nanobot tags any tool message that carried an image payload with the
        # literal "[image viewed * N]" marker. Only try to resurrect images
        # when the marker is present. Prefer absolute /mnt/data/ paths; fall
        # back to bare filenames (models sometimes pass relative paths to
        # read_file).
        images: list[Path] = []
        if IMAGE_VIEWED_MARKER in response:
            seen: set[str] = set()
            for match in SANDBOX_PATH_RE.findall(response):
                resolved = _resolve_sandbox_image(match, session_dir)
                if resolved is None:
                    continue
                key = str(resolved)
                if key in seen:
                    continue
                seen.add(key)
                images.append(resolved)
            if not images:
                for match in BARE_IMAGE_NAME_RE.findall(response):
                    resolved = _resolve_sandbox_image(match, session_dir)
                    if resolved is None:
                        continue
                    key = str(resolved)
                    if key in seen:
                        continue
                    seen.add(key)
                    images.append(resolved)

        prefix.append(
            (
                _truncate(call_str, per_response_char_limit),
                _truncate(response, per_response_char_limit),
                images,
            )
        )

    return prefix


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

PREFIX_HEADER = (
    "Below are intermediate tool observations produced while solving the task. "
    "Treat them as given facts. Do not call any tools; answer the question directly."
)
PREFIX_FOOTER = "Based on the observations above, give the final answer to the original question."


def build_message_content(
    query: str,
    original_images: list[Path],
    tool_responses: list[tuple[str, str, list[Path]]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    for image_path in original_images:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
    content.append({"type": "text", "text": f"[Question]\n{query}"})

    if tool_responses:
        content.append({"type": "text", "text": PREFIX_HEADER})
        for idx, (call_str, response_text, images) in enumerate(tool_responses, 1):
            block_lines = [f"[tool_call #{idx}]"]
            if call_str:
                block_lines.append(call_str)
            block_lines.append(f"[tool_response #{idx}]")
            block_lines.append(response_text)
            content.append({"type": "text", "text": "\n".join(block_lines)})
            for image_path in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    }
                )
        content.append({"type": "text", "text": PREFIX_FOOTER})
    return content


# ---------------------------------------------------------------------------
# Rollout / evaluation (same shape as filter_difficulty.py)
# ---------------------------------------------------------------------------


class VllmRolloutClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str = "",
        base_url: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE")
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

    def rollout(self, content: list[dict[str, Any]], retries: int = DEFAULT_RETRIES) -> str:
        last_error: Exception | None = None
        client = self._get_client()
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response_to_text(response)
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(2**attempt)
        if last_error is not None:
            raise last_error
        return ""


def evaluate_prediction(
    prediction: str,
    answer: str,
    *,
    accuracy_backend: str,
    query: str,
    tolerance: float,
    match_mode: str,
    judge: JudgeLLM | None,
    judge_retries: int,
) -> tuple[bool, str]:
    if accuracy_backend == "rule":
        return check_match(prediction, answer, tolerance=tolerance, mode=match_mode)
    if accuracy_backend == "judge":
        if judge is None:
            raise ValueError("Judge backend requested but JudgeLLM is not initialized")
        return judge.judge(prediction, answer, question=query, retries=judge_retries), "judge_llm"
    raise ValueError(f"Unsupported accuracy backend: {accuracy_backend}")


def process_one_item(
    item: dict[str, Any],
    *,
    index: int,
    client: VllmRolloutClient,
    judge: JudgeLLM | None,
    html_judge: VlmHtmlJudge | None,
    html_renderer: HtmlRenderer | None,
    media_dir: Path,
    sessions_dir: Path,
    sample_suffix: str,
    query_field: str,
    answer_field: str,
    media_field: str,
    k: int,
    accuracy_backend: str,
    tolerance: float,
    match_mode: str,
    judge_retries: int,
    per_response_char_limit: int,
    html_score_threshold: float,
    html_render_dir: Path,
    dataset_stem: str,
    keep_rendered_images: bool,
    html_default_viewport: tuple[int, int],
    html_min_tag_ratio: float,
    html_min_structural_jaccard: float,
    html_min_text_recall: float,
    html_require_css: bool,
    html_require_rick_image: bool,
) -> dict[str, Any]:
    query = get_text_field(item, query_field, DEFAULT_QUERY_FALLBACKS)
    answer = get_text_field(item, answer_field)
    original_images = resolve_media_paths(item, media_field, media_dir)

    trace_path = _find_session_trace(sessions_dir, item.get("id"), sample_suffix)
    if trace_path is None:
        raise FileNotFoundError(
            f"Teacher session trace not found for id={item.get('id')!r} "
            f"under {sessions_dir} (suffix={sample_suffix!r})"
        )

    tool_responses = extract_teacher_prefix(trace_path, per_response_char_limit=per_response_char_limit)

    # Teacher never invoked a tool: there is no prefix to evaluate, and any
    # "gain" measured against the no-tool baseline would be noise. Mark the
    # sample as invalid (avg_k_with_prefix=None) so select_tool_gain.py skips
    # it, and skip rollout to save API calls.
    if not tool_responses:
        return {
            "key": get_item_key(item, index),
            "index": index,
            "id": item.get("id", index),
            "avg_k_with_prefix": None,
            "correct_count": 0,
            "corrects": [],
            "methods": [],
            "predictions": [],
            "prefix_tool_responses": 0,
            "prefix_images": 0,
            "prefix_text_chars": 0,
            "trace_path": str(trace_path),
            "skipped_reason": "no_teacher_tool_calls",
        }

    content = build_message_content(query, original_images, tool_responses)

    predictions: list[str] = []
    corrects: list[int] = []
    methods: list[str] = []
    scores: list[float] = []
    rule_results: list[dict[str, Any]] = []
    rendered_images: list[str] = []
    for rollout_index in range(k):
        prediction = client.rollout(content)
        if accuracy_backend == "html_vlm":
            if html_judge is None or html_renderer is None:
                raise ValueError("HTML VLM backend requested but judge/renderer is not initialized")
            if not original_images:
                raise ValueError("HTML VLM backend requires at least one reference image")
            is_correct, method, score, rule_payload, rendered_image = evaluate_html_prediction(
                prediction,
                answer,
                query=query,
                reference_image=original_images[0],
                renderer=html_renderer,
                html_judge=html_judge,
                judge_retries=judge_retries,
                score_threshold=html_score_threshold,
                render_dir=html_render_dir,
                dataset_stem=dataset_stem,
                item_id=item.get("id", index),
                index=index,
                rollout_index=rollout_index,
                keep_rendered_images=keep_rendered_images,
                default_viewport=html_default_viewport,
                min_tag_ratio=html_min_tag_ratio,
                min_structural_jaccard=html_min_structural_jaccard,
                min_text_recall=html_min_text_recall,
                require_css=html_require_css,
                require_rick_image=html_require_rick_image,
            )
            scores.append(score)
            rule_results.append(rule_payload)
            rendered_images.append(rendered_image)
        else:
            is_correct, method = evaluate_prediction(
                prediction,
                answer,
                accuracy_backend=accuracy_backend,
                query=query,
                tolerance=tolerance,
                match_mode=match_mode,
                judge=judge,
                judge_retries=judge_retries,
            )
        predictions.append(prediction)
        corrects.append(int(is_correct))
        methods.append(method)

    avg_k = sum(corrects) / k if k else 0.0
    result: dict[str, Any] = {
        "key": get_item_key(item, index),
        "index": index,
        "id": item.get("id", index),
        "avg_k_with_prefix": round(avg_k, 4),
        "correct_count": sum(corrects),
        "corrects": corrects,
        "methods": methods,
        "predictions": predictions,
        "prefix_tool_responses": len(tool_responses),
        "prefix_images": sum(len(imgs) for _, _, imgs in tool_responses),
        "prefix_text_chars": sum(len(call_str) + len(resp) for call_str, resp, _ in tool_responses),
        "trace_path": str(trace_path),
    }
    if scores:
        result["scores"] = scores
    if rule_results:
        result["rule_results"] = rule_results
    if rendered_images:
        result["rendered_images"] = rendered_images
    return result


# ---------------------------------------------------------------------------
# Progress / output
# ---------------------------------------------------------------------------


def load_processed_keys(progress_path: Path) -> set[str]:
    if not progress_path.is_file():
        return set()
    keys: set[str] = set()
    for record in _iter_jsonl_records_allow_partial_tail(progress_path):
        key = str(record.get("key", "")).strip()
        if key:
            keys.add(key)
    return keys


def infer_output_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}_tool_gain_prefix.jsonl").resolve()


def infer_progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(".progress.jsonl").resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute avg@k when the base model is given the teacher's tool_response "
            "prefix (no tool_call metadata)."
        )
    )
    p.add_argument("--dataset", "--input", dest="dataset", required=True)
    p.add_argument("--sessions-dir", required=True, help="Root directory of teacher sessions.")
    p.add_argument(
        "--sample-suffix",
        default="",
        help="Suffix on session dir names, e.g. '_s0' when teacher was run with --sample-index 0.",
    )
    p.add_argument("--media-dir", default="", help="Dataset image directory.")
    p.add_argument("--output", default="", help="Output JSONL.")
    p.add_argument("--model", required=True, help="Base model name exposed by the OpenAI-compatible API.")
    p.add_argument(
        "--rollout-api-base",
        default=os.environ.get("OPENAI_API_BASE", ""),
        help="OpenAI-compatible API base URL for rollout model. Defaults to OPENAI_API_BASE.",
    )
    p.add_argument("--query-field", default="query")
    p.add_argument("--answer-field", default="answer")
    p.add_argument("--media-field", default="images")
    p.add_argument("--k", type=int, default=4)
    p.add_argument(
        "--accuracy-backend",
        choices=("rule", "judge", "html_vlm"),
        default="rule",
        help="How to judge rollout correctness. Use html_vlm for screenshot-to-HTML tasks.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Rule-match tolerance. In generic mode it is relative error for non-zero numeric refs; in point mode it is the normalized coordinate tolerance. Ignored in bbox mode.",
    )
    p.add_argument(
        "--match-mode",
        choices=("generic", "point", "bbox"),
        default="generic",
        help="Ground-truth format for rule matching. Used only when --accuracy-backend=rule.",
    )
    p.add_argument("--judge-model", default="gemini-3.0-flash-preview", help="JudgeLLM model name.")
    p.add_argument(
        "--judge-api-base",
        default="",
        help="OpenAI-compatible API base URL for judge model. Defaults to --rollout-api-base.",
    )
    p.add_argument("--judge-retries", type=int, default=3, help="Retry count for JudgeLLM calls.")
    p.add_argument("--judge-max-tokens", type=int, default=16384)
    p.add_argument("--judge-temperature", type=float, default=0.6)
    p.add_argument(
        "--html-score-threshold",
        type=float,
        default=80.0,
        help="For --accuracy-backend=html_vlm, VLM score >= this value counts as correct.",
    )
    p.add_argument(
        "--html-render-dir",
        default="",
        help="Directory for rendered prediction screenshots. Defaults to <output_dir>/rendered_html_tool_gain_prefix.",
    )
    p.add_argument(
        "--keep-rendered-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For html_vlm, keep rendered prediction screenshots on disk after judging.",
    )
    p.add_argument("--html-render-timeout-ms", type=int, default=10000)
    p.add_argument("--html-render-wait-ms", type=int, default=500)
    p.add_argument("--html-default-viewport", default="1280x720")
    p.add_argument("--html-min-tag-ratio", type=float, default=0.35)
    p.add_argument("--html-min-structural-jaccard", type=float, default=0.25)
    p.add_argument("--html-min-text-recall", type=float, default=0.25)
    p.add_argument("--html-no-require-css", action="store_true")
    p.add_argument("--html-no-require-rick-image", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--system-prompt-md", "--system-prompt-file", dest="system_prompt_md", default="")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--sample-range", default="")
    p.add_argument(
        "--per-response-char-limit",
        type=int,
        default=DEFAULT_TOOL_RESULT_MAX_CHARS,
        help="Truncate each tool_response text to this many chars (<=0 disables).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be > 0")
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required")
    if not args.rollout_api_base:
        raise EnvironmentError("--rollout-api-base or OPENAI_API_BASE is required")
    if args.accuracy_backend in {"judge", "html_vlm"} and not (args.judge_api_base or args.rollout_api_base):
        raise EnvironmentError("--judge-api-base or --rollout-api-base is required for judge backend")

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    sessions_dir = Path(args.sessions_dir).resolve()
    if not sessions_dir.is_dir():
        raise FileNotFoundError(f"Sessions dir not found: {sessions_dir}")
    media_dir = (
        Path(args.media_dir).resolve() if args.media_dir else (dataset_path.parent / "images").resolve()
    )
    if not media_dir.is_dir():
        raise FileNotFoundError(f"Media directory not found: {media_dir}")

    output_path = Path(args.output).resolve() if args.output else infer_output_path(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = infer_progress_path(output_path)
    processed_keys = load_processed_keys(progress_path)

    html_render_dir = (
        Path(args.html_render_dir).resolve()
        if args.html_render_dir
        else (output_path.parent / "rendered_html_tool_gain_prefix").resolve()
    )

    system_prompt, system_prompt_path = load_system_prompt_markdown(args.system_prompt_md)

    items = load_jsonl(dataset_path)
    sample_slice = parse_sample_range(args.sample_range)
    selected_pairs = list(enumerate(items))[sample_slice]
    pending_pairs = [
        (index, item) for index, item in selected_pairs if get_item_key(item, index) not in processed_keys
    ]
    skipped = len(selected_pairs) - len(pending_pairs)

    client = VllmRolloutClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_prompt=system_prompt,
        base_url=args.rollout_api_base,
    )
    judge_api_base = args.judge_api_base or args.rollout_api_base
    judge = (
        JudgeLLM(model=args.judge_model, base_url=judge_api_base)
        if args.accuracy_backend == "judge"
        else None
    )
    html_judge = (
        VlmHtmlJudge(
            model=args.judge_model,
            base_url=judge_api_base,
            max_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
        )
        if args.accuracy_backend == "html_vlm"
        else None
    )
    html_renderer = (
        HtmlRenderer(timeout_ms=args.html_render_timeout_ms, wait_ms=args.html_render_wait_ms)
        if args.accuracy_backend == "html_vlm"
        else None
    )
    html_default_viewport = _parse_viewport(args.html_default_viewport)
    if args.accuracy_backend == "html_vlm" and args.keep_rendered_images:
        html_render_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Dataset: {dataset_path}\n"
        f"Sessions dir: {sessions_dir} (suffix={args.sample_suffix!r})\n"
        f"Media dir: {media_dir}\n"
        f"Model: {args.model} (k={args.k}, workers={args.num_workers})\n"
        f"Rollout api base: {args.rollout_api_base}\n"
        f"Accuracy backend: {args.accuracy_backend}"
        + (
            f" ({args.judge_model} @ {judge_api_base})"
            if args.accuracy_backend in {"judge", "html_vlm"}
            else f" ({args.match_mode})"
        )
        + "\n"
        + (
            f"HTML score threshold: {args.html_score_threshold}, render dir: {html_render_dir}, "
            f"keep rendered: {args.keep_rendered_images}\n"
            if args.accuracy_backend == "html_vlm"
            else ""
        )
        + f"System prompt md: {system_prompt_path or '-'}\n"
        f"Output: {output_path}\n"
        f"Progress: {progress_path}\n"
        f"Selected: {len(selected_pairs)} / {len(items)}; resumed: {skipped}; pending: {len(pending_pairs)}"
    )

    failed = 0
    processed = 0
    skipped_no_tool = 0
    mean_sum = 0.0
    mean_count = 0

    if not pending_pairs:
        print("Nothing to do.")
        return

    with (
        output_path.open("a", encoding="utf-8", buffering=1) as output_file,
        progress_path.open("a", encoding="utf-8", buffering=1) as progress_file,
        ThreadPoolExecutor(max_workers=min(args.num_workers, len(pending_pairs))) as executor,
    ):
        futures = {
            executor.submit(
                process_one_item,
                item,
                index=index,
                client=client,
                judge=judge,
                html_judge=html_judge,
                html_renderer=html_renderer,
                media_dir=media_dir,
                sessions_dir=sessions_dir,
                sample_suffix=args.sample_suffix,
                query_field=args.query_field,
                answer_field=args.answer_field,
                media_field=args.media_field,
                k=args.k,
                accuracy_backend=args.accuracy_backend,
                tolerance=args.tolerance,
                match_mode=args.match_mode,
                judge_retries=args.judge_retries,
                per_response_char_limit=args.per_response_char_limit,
                html_score_threshold=args.html_score_threshold,
                html_render_dir=html_render_dir,
                dataset_stem=dataset_path.stem,
                keep_rendered_images=args.keep_rendered_images,
                html_default_viewport=html_default_viewport,
                html_min_tag_ratio=args.html_min_tag_ratio,
                html_min_structural_jaccard=args.html_min_structural_jaccard,
                html_min_text_recall=args.html_min_text_recall,
                html_require_css=not args.html_no_require_css,
                html_require_rick_image=not args.html_no_require_rick_image,
            ): index
            for index, item in pending_pairs
        }

        with tqdm(total=len(futures), desc="Prefix rollout", unit="item") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed += 1
                    if result.get("skipped_reason") == "no_teacher_tool_calls":
                        skipped_no_tool += 1
                    elif result.get("avg_k_with_prefix") is not None:
                        mean_sum += result["avg_k_with_prefix"]
                        mean_count += 1
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                    progress_record: dict[str, Any] = {
                        "key": result["key"],
                        "index": result["index"],
                        "id": result["id"],
                        "avg_k_with_prefix": result["avg_k_with_prefix"],
                    }
                    if result.get("skipped_reason"):
                        progress_record["skipped_reason"] = result["skipped_reason"]
                    if result.get("scores"):
                        progress_record["score_avg"] = round(
                            sum(result["scores"]) / len(result["scores"]), 4
                        )
                    progress_file.write(json.dumps(progress_record, ensure_ascii=False) + "\n")
                    progress_file.flush()
                except Exception as exc:
                    failed += 1
                    tqdm.write(f"[WARN] item #{futures[future]} failed: {exc}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix(failed=failed, refresh=False)

    mean_avg = mean_sum / mean_count if mean_count else 0.0
    print(
        f"\nProcessed: {processed}\nFailed: {failed}\n"
        f"Skipped (teacher had no tool_calls): {skipped_no_tool}\n"
        f"Mean avg@k_with_prefix (over {mean_count} scored items): {mean_avg:.4f}\n"
        f"Output: {output_path}\nProgress: {progress_path}"
    )


if __name__ == "__main__":
    main()
