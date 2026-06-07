"""Filter distilled sessions: accuracy check + tool-call quality rules.

Two-stage pipeline that outputs an index file of passing session IDs:

  Stage 1 — Accuracy:
    Keep only sessions where the answer is correct AND has tool calls.
    Accuracy can be judged either by deterministic rules or by JudgeLLM.

  Stage 2 — Tool-call quality:
    Remove sessions matching ANY of:
      a. Model calls exactly one low-value tool (`read_file`, `sample_color`,
         or `write_file`) then directly answers (no real tool usage)
      b. Total tool-call rounds exceed --max-tool-rounds (default 100)
      c. Image files were overwritten during the session (duplicate save paths)

Output: a JSONL index file where each line is:
  {"id": <int>, "method": "<match_method>", "tool_calls": <int>}

Usage:
    python distill/utils/filter_distill_sessions.py \
        --sessions-dir workspaces/qwen35/ChartVerse-SFT-600K \
        --dataset dataset/ChartVerse-SFT-600K/ChartVerse-SFT-600K.jsonl \
        --output workspaces/qwen35/filtered_index.jsonl \
        [--tolerance 0.02] [--max-tool-rounds 100] [--limit 60000]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from distill.filter.evaluate_answer_judge import JudgeLLM
from distill.filter.evaluate_answer_rule import check_match
from distill.filter.evaluate_html_vlm_judge import (
    HtmlRenderer,
    VlmHtmlJudge,
    _parse_viewport,
    _reference_viewport,
    _sample_output_stem,
    rule_check_html,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Session trace parsing
# ---------------------------------------------------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
DEFAULT_MEDIA_FALLBACKS = ("images", "image", "screenshot", "screenshots")
SAVEFIG_RE = re.compile(
    r"""\.(?:savefig|save|imwrite)\s*\(\s*""" r"""(?:f?['"])([^'"]+\.(?:png|jpg|jpeg|gif|bmp|webp))['"]""",
    re.IGNORECASE,
)
SAVED_TO_RE = re.compile(r"Saved to: (\S+)")
IMAGE_VIEWED_RE = re.compile(r"^\[image viewed \* \d+\]", re.IGNORECASE)
HOST_PATH_RE = re.compile(r"/mnt/afs[A-Za-z_]*/\S+")


def parse_trace(jsonl_path: str) -> list[dict]:
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def extract_events(messages: list[dict]) -> list[dict]:
    if not messages:
        return []

    events = messages[1:] if messages[0].get("_type") == "metadata" else messages
    last_user_idx = -1
    for idx, record in enumerate(events):
        if record.get("role") == "user":
            last_user_idx = idx
    return events[last_user_idx:] if last_user_idx >= 0 else events


def extract_media_dir(messages: list[dict]) -> Path | None:
    if not messages:
        return None
    metadata = messages[0]
    if metadata.get("_type") != "metadata":
        return None
    meta_block = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    media_dir_str = str(meta_block.get("media_dir", "") or "").strip()
    return Path(media_dir_str).resolve() if media_dir_str else None


def infer_media_dir(dataset_path: Path) -> Path:
    return (dataset_path.parent / "images").resolve()


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
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path.strip()).expanduser()
        if not path.is_absolute():
            path = (media_dir / path).resolve()
        if path.is_file():
            resolved.append(path)
    return resolved


_COORD_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _coerce_coordinate_pair(coord: Any) -> list[float] | None:
    """Normalize a ``coordinate`` value to ``[x, y]`` floats.

    The teacher model sometimes emits the coordinate as a JSON array
    (``[x, y]``), sometimes as a string literal that *contains* a JSON array
    (``"[17, 500]"``), and occasionally as comma- or whitespace-separated
    numbers (``"17, 500"``). Handle all of these before giving up.
    """
    # Unwrap a single-element list like [[x, y]] that occasionally shows up.
    if isinstance(coord, (list, tuple)) and len(coord) == 1 and isinstance(coord[0], (list, tuple)):
        coord = coord[0]

    if isinstance(coord, (list, tuple)) and len(coord) == 2:
        try:
            return [float(coord[0]), float(coord[1])]
        except (TypeError, ValueError):
            return None

    if isinstance(coord, str):
        s = coord.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return _coerce_coordinate_pair(parsed)
        numbers = _COORD_NUMBER_RE.findall(s)
        if len(numbers) == 2:
            try:
                return [float(numbers[0]), float(numbers[1])]
            except ValueError:
                return None

    return None


def _extract_computer_use_coordinate(args: Any) -> list[float] | None:
    """Pull a 2-number coordinate out of a computer_use arguments blob.

    The teacher model emits `computer_use` in the Qwen3-VL schema, where
    click-style actions carry `coordinate: [x, y]` in the normalized 0-1000
    space. We deliberately ignore the `action` field — if a coordinate is
    present at all, we treat it as the grounding answer.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return _coerce_coordinate_pair(args.get("coordinate"))


def scan_session(
    messages: list[dict],
) -> tuple[str | None, bool, list[dict], dict | None, list[float] | None, bool]:
    """Return (last_assistant_content, has_non_terminator_tool_calls, tool_calls,
    computer_use_args, computer_use_coordinate, computer_use_is_last).

    ``has_non_terminator_tool_calls`` is True only when the trace contains at
    least one tool_call *other than* ``computer_use`` — `computer_use` is a
    format-only terminator and a trace that contains only it should be
    considered as having no real tool usage.

    ``computer_use_is_last`` is True only when ``computer_use`` appears in the
    very last assistant turn that contains tool_calls. Anything else (e.g. a
    `computer_use` followed by another assistant turn with more tool_calls)
    is anomalous and should be flagged.
    """
    last_assistant = None
    has_non_terminator_tool_calls = False
    tool_calls: list[dict] = []
    computer_use_args: dict | None = None
    computer_use_coordinate: list[float] | None = None
    last_assistant_with_tc_has_computer_use = False

    for msg in extract_events(messages):
        if msg.get("role") == "assistant":
            if msg.get("content"):
                last_assistant = msg["content"]
            turn_tool_calls = msg.get("tool_calls") or []
            if turn_tool_calls:
                last_assistant_with_tc_has_computer_use = any(
                    (tc.get("function") or {}).get("name") == "computer_use" for tc in turn_tool_calls
                )
            for tc in turn_tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"name": name, "args": args})
                if name == "computer_use":
                    # Keep the last computer_use seen; by construction the loop
                    # short-circuits on the first one, so there should be at
                    # most one per trace.
                    computer_use_args = args if isinstance(args, dict) else None
                    computer_use_coordinate = _extract_computer_use_coordinate(args)
                else:
                    has_non_terminator_tool_calls = True

    computer_use_is_last = computer_use_args is not None and last_assistant_with_tc_has_computer_use

    return (
        last_assistant,
        has_non_terminator_tool_calls,
        tool_calls,
        computer_use_args,
        computer_use_coordinate,
        computer_use_is_last,
    )


# ---------------------------------------------------------------------------
# Tool-call quality filters
# ---------------------------------------------------------------------------

SINGLE_LOW_VALUE_TOOLS = {"read_file", "sample_color", "write_file", "list_dir", "edit_file"}


def get_single_low_value_tool_reason(tool_calls: list[dict]) -> str | None:
    if len(tool_calls) != 1:
        return None
    tool_name = tool_calls[0]["name"]
    if tool_name not in SINGLE_LOW_VALUE_TOOLS:
        return None
    return f"single_{tool_name}"


def check_too_many_rounds(tool_calls: list[dict], max_rounds: int) -> bool:
    return len(tool_calls) > max_rounds


def check_image_overwrite(messages: list[dict], tool_calls: list[dict]) -> bool:
    paths: list[str] = []

    for msg in extract_events(messages):
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            for m in SAVED_TO_RE.finditer(content):
                p = m.group(1)
                if Path(p).suffix.lower() in IMG_EXTS:
                    paths.append(p)

    for tc in tool_calls:
        if tc["name"] == "write_file":
            wpath = tc["args"].get("path", "")
            if Path(wpath).suffix.lower() in IMG_EXTS:
                paths.append(wpath)
            content = tc["args"].get("content", "")
            for m in SAVEFIG_RE.finditer(content):
                paths.append(m.group(1))
        elif tc["name"] == "exec":
            cmd = tc["args"].get("command", "")
            for m in SAVEFIG_RE.finditer(cmd):
                paths.append(m.group(1))

    cnt = Counter(paths)
    return any(v > 1 for v in cnt.values())


def _session_root(session_path: Path) -> Path:
    if session_path.parent.name == "sessions" and session_path.parent.parent.name.startswith("session_"):
        return session_path.parent.parent
    return session_path.parent


def _normalize_path_string(raw_path: str) -> str:
    return raw_path.strip().strip('"').strip("'")


def _candidate_paths(raw_path: str, session_path: Path, media_dir: Path | None = None) -> list[Path]:
    raw_path = _normalize_path_string(raw_path)
    if not raw_path:
        return []

    path = Path(raw_path)
    session_root = _session_root(session_path)
    candidates: list[Path] = []

    def _add(candidate: Path | None) -> None:
        if candidate is None:
            return
        candidate = candidate.resolve()
        if candidate not in candidates:
            candidates.append(candidate)

    if path.is_absolute():
        _add(path)
    else:
        _add(session_path.parent / path)
        _add(session_root / path)
        _add(PROJECT_ROOT / path)

    if media_dir is not None:
        _add(media_dir / path)
        _add(media_dir / path.name)

    _add(session_root / path.name)

    parts = path.parts
    for idx, part in enumerate(parts):
        if part.startswith("session_"):
            suffix = parts[idx + 1 :]
            _add(session_root.joinpath(*suffix))
            break

    return candidates


def _resolve_existing_path(raw_path: str, session_path: Path, media_dir: Path | None = None) -> Path | None:
    for candidate in _candidate_paths(raw_path, session_path, media_dir):
        if candidate.is_file():
            return candidate
    return None


def _scan_host_paths_in_object(value: Any, hits: list[str]) -> None:
    if isinstance(value, str):
        hits.extend(HOST_PATH_RE.findall(value))
    elif isinstance(value, list):
        for item in value:
            _scan_host_paths_in_object(item, hits)
    elif isinstance(value, dict):
        for item in value.values():
            _scan_host_paths_in_object(item, hits)


def check_host_path_hit(messages: list[dict]) -> bool:
    hits: list[str] = []
    if messages and messages[0].get("_type") == "metadata":
        meta_block = messages[0].get("metadata") if isinstance(messages[0].get("metadata"), dict) else {}
        system_prompt = str(meta_block.get("system_prompt", "") or "")
        hits.extend(HOST_PATH_RE.findall(system_prompt))

    for record in extract_events(messages):
        role = record.get("role")
        if role == "user":
            _scan_host_paths_in_object(record.get("content"), hits)
        elif role == "assistant":
            hits.extend(HOST_PATH_RE.findall(str(record.get("reasoning_content", "") or "")))
            hits.extend(HOST_PATH_RE.findall(str(record.get("content", "") or "")))
            _scan_host_paths_in_object(record.get("tool_calls") or [], hits)
        elif role == "tool":
            hits.extend(HOST_PATH_RE.findall(str(record.get("content", "") or "")))

        if hits:
            return True

    return False


def _collect_read_file_images(
    messages: list[dict],
    *,
    jsonl_path: Path,
    media_dir: Path | None,
) -> dict[str, Path | None]:
    tool_images: dict[str, Path | None] = {}
    for record in extract_events(messages):
        if record.get("role") != "assistant":
            continue
        for tc in record.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tool_call_id = str(tc.get("id", "") or "")
            function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            if function.get("name") != "read_file":
                continue
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"path": arguments}
            if not isinstance(arguments, dict):
                continue
            raw_path = str(arguments.get("path", "")).strip()
            if Path(raw_path).suffix.lower() not in IMG_EXTS:
                continue
            tool_images[tool_call_id] = _resolve_existing_path(raw_path, jsonl_path, media_dir)
    return tool_images


def check_missing_images(messages: list[dict], jsonl_path: Path) -> bool:
    media_dir = extract_media_dir(messages)
    read_file_images = _collect_read_file_images(messages, jsonl_path=jsonl_path, media_dir=media_dir)

    for record in extract_events(messages):
        role = record.get("role")
        if role == "user":
            content = record.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image":
                    continue
                raw_path = str(block.get("image", "") or "")
                if _resolve_existing_path(raw_path, jsonl_path, media_dir) is None:
                    return True
            continue

        if role != "tool":
            continue

        text = str(record.get("content", "") or "")
        tool_call_id = str(record.get("tool_call_id", "") or "")
        if (
            IMAGE_VIEWED_RE.search(text)
            and tool_call_id in read_file_images
            and read_file_images[tool_call_id] is None
        ):
            return True

        for match in SAVED_TO_RE.finditer(text):
            if _resolve_existing_path(match.group(1), jsonl_path, media_dir) is None:
                return True

    return False


def quality_filter(
    messages: list[dict],
    tool_calls: list[dict],
    jsonl_path: Path,
    max_rounds: int,
    *,
    reject_host_paths: bool,
    reject_missing_images: bool,
) -> str | None:
    """Return rejection reason or None if session passes."""
    if reason := get_single_low_value_tool_reason(tool_calls):
        return reason
    if check_too_many_rounds(tool_calls, max_rounds):
        return "too_many_rounds"
    if check_image_overwrite(messages, tool_calls):
        return "image_overwrite"
    if reject_host_paths and check_host_path_hit(messages):
        return "host_path_hit"
    if reject_missing_images and check_missing_images(messages, jsonl_path):
        return "missing_image"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_reference_items(
    dataset_path: Path,
    needed_ids: set[int],
    *,
    answer_field: str,
    question_field: str,
    media_field: str,
    media_dir: Path,
    require_media: bool,
) -> dict[int, dict[str, Any]]:
    items = {}
    missing_answer = 0
    missing_question = 0
    missing_media = 0
    sample_keys: list[str] | None = None
    with open(dataset_path) as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if sample_keys is None:
                sample_keys = list(item.keys())
            item_id = item["id"]
            if item_id in needed_ids:
                answer = str(item.get(answer_field, "")).strip()
                question = str(item.get(question_field, "")).strip()
                media_paths = resolve_media_paths(item, media_field, media_dir)
                if answer_field not in item or not answer:
                    missing_answer += 1
                if question_field not in item or not question:
                    missing_question += 1
                if require_media and not media_paths:
                    missing_media += 1
                items[item_id] = {
                    "answer": answer,
                    "question": question,
                    "media_paths": media_paths,
                    "index": index,
                }
                if len(items) == len(needed_ids):
                    break

    total = len(items)
    if total and missing_answer:
        print(
            f"[warn] answer-field '{answer_field}' missing/empty in {missing_answer}/{total} items. "
            f"Available keys in dataset: {sample_keys}. "
            f"Pass --answer-field <correct_key> to fix.",
            file=sys.stderr,
        )
    if total and missing_question:
        print(
            f"[warn] question-field '{question_field}' missing/empty in {missing_question}/{total} items. "
            f"Available keys in dataset: {sample_keys}. "
            f"Judge backend will run without question context for those items — "
            f"pass --question-field <correct_key> to fix.",
            file=sys.stderr,
        )
    if require_media and total and missing_media:
        print(
            f"[warn] media-field '{media_field}' missing/unresolved in {missing_media}/{total} items. "
            f"Available keys in dataset: {sample_keys}. "
            f"Required when --accuracy-backend=html_vlm.",
            file=sys.stderr,
        )
    return items


def evaluate_html_prediction(
    prediction: str,
    reference_html: str,
    *,
    reference_image: Path,
    renderer: HtmlRenderer,
    html_judge: VlmHtmlJudge,
    judge_retries: int,
    score_threshold: float,
    render_dir: Path,
    dataset_stem: str,
    item_id: Any,
    index: int,
    keep_rendered_images: bool,
    default_viewport: tuple[int, int],
    min_tag_ratio: float,
    min_structural_jaccard: float,
    min_text_recall: float,
    require_css: bool,
    require_rick_image: bool,
) -> tuple[bool, str]:
    rule = rule_check_html(
        prediction,
        reference_html,
        min_tag_ratio=min_tag_ratio,
        min_structural_jaccard=min_structural_jaccard,
        min_text_recall=min_text_recall,
        require_css=require_css,
        require_rick_image=require_rick_image,
    )
    if not rule["passed"]:
        return False, "html_rule:" + ",".join(rule["errors"] or ["failed"])

    stem = _sample_output_stem(item_id, index, None)
    rendered_path = (
        render_dir / f"{dataset_stem}_{stem}.png"
        if keep_rendered_images
        else Path(tempfile.gettempdir()) / f"{dataset_stem}_{stem}_{threading.get_ident()}.png"
    )
    viewport = _reference_viewport(reference_image, default_viewport)
    try:
        renderer.render(str(rule["extracted_html"]), rendered_path, viewport)
        score, _reason, _raw = html_judge.judge(
            reference_image=reference_image,
            rendered_image=rendered_path,
            retries=judge_retries,
        )
    except Exception as exc:
        # Render or judge failure (chromium crash, VLM RPC error, …). Reject
        # the sample rather than killing the whole run — one bad HTML must
        # not abort 18k sessions.
        return False, f"html_render_error:{type(exc).__name__}"
    finally:
        if not keep_rendered_images:
            rendered_path.unlink(missing_ok=True)

    return score >= score_threshold, f"html_vlm_score:{score:.2f}"


def process_one_session(
    sid: int,
    jsonl_path: Path,
    ref_item: dict[str, Any] | None,
    *,
    accuracy_backend: str,
    match_mode: str,
    tolerance: float,
    max_tool_rounds: int,
    reject_host_paths: bool,
    reject_missing_images: bool,
    judge: JudgeLLM | None,
    html_judge: VlmHtmlJudge | None,
    html_renderer: HtmlRenderer | None,
    judge_retries: int,
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
) -> dict:
    try:
        messages = parse_trace(str(jsonl_path))
    except Exception:
        return {"sid": sid, "passed": False, "stage": "stage1", "reason": "parse_error"}

    (
        content,
        has_tool_calls,
        tool_calls,
        computer_use_args,
        computer_use_coordinate,
        computer_use_is_last,
    ) = scan_session(messages)
    has_computer_use = computer_use_args is not None

    if has_computer_use and not computer_use_is_last:
        print(
            f"[warn] sid={sid}: computer_use tool_call is not in the last "
            "assistant turn with tool_calls; the trace likely went past the "
            "terminator.",
            file=sys.stderr,
        )

    # computer_use terminates a GUI-grounding trace as a tool_call whose
    # arguments carry the answer (coordinate / text). If it's present but
    # no coordinate can be pulled out of it, the trace is an invalid/failed
    # grounding sample — reject with a dedicated reason.
    if has_computer_use and computer_use_coordinate is None:
        return {
            "sid": sid,
            "passed": False,
            "stage": "stage1",
            "reason": "computer_use_no_coordinate",
        }

    # For non-computer_use traces we still require a final assistant text.
    if not has_computer_use and content is None:
        return {"sid": sid, "passed": False, "stage": "stage1", "reason": "no_response"}
    if ref_item is None:
        return {"sid": sid, "passed": False, "stage": "stage1", "reason": "no_ref_answer"}
    if not has_tool_calls:
        return {"sid": sid, "passed": False, "stage": "stage1", "reason": "no_tool_calls"}

    ref_answer = ref_item.get("answer", "")
    question = ref_item.get("question", "")
    if not ref_answer:
        return {"sid": sid, "passed": False, "stage": "stage1", "reason": "empty_ref_answer"}

    if computer_use_coordinate is not None:
        if match_mode not in ("point", "bbox"):
            print(
                f"[warn] sid={sid}: computer_use coordinate extracted but "
                f"--match-mode={match_mode!r} is not 'point'/'bbox'; "
                "falling back to point-mode comparison.",
                file=sys.stderr,
            )
        effective_mode = match_mode if match_mode in ("point", "bbox") else "point"
        coord_text = f"[{computer_use_coordinate[0]}, {computer_use_coordinate[1]}]"
        is_match, method = check_match(coord_text, ref_answer, tolerance=tolerance, mode=effective_mode)
    elif accuracy_backend == "rule":
        is_match, method = check_match(content, ref_answer, tolerance=tolerance, mode=match_mode)
    elif accuracy_backend == "judge":
        if judge is None:
            return {"sid": sid, "passed": False, "stage": "stage1", "reason": "judge_not_initialized"}
        is_match = judge.judge(content, ref_answer, question=question, retries=judge_retries)
        method = "judge_llm"
    elif accuracy_backend == "html_vlm":
        if html_judge is None or html_renderer is None:
            return {"sid": sid, "passed": False, "stage": "stage1", "reason": "html_judge_not_initialized"}
        media_paths = ref_item.get("media_paths") or []
        if not media_paths:
            return {"sid": sid, "passed": False, "stage": "stage1", "reason": "missing_reference_image"}
        is_match, method = evaluate_html_prediction(
            content,
            ref_answer,
            reference_image=media_paths[0],
            renderer=html_renderer,
            html_judge=html_judge,
            judge_retries=judge_retries,
            score_threshold=html_score_threshold,
            render_dir=html_render_dir,
            dataset_stem=dataset_stem,
            item_id=sid,
            index=int(ref_item.get("index", sid)),
            keep_rendered_images=keep_rendered_images,
            default_viewport=html_default_viewport,
            min_tag_ratio=html_min_tag_ratio,
            min_structural_jaccard=html_min_structural_jaccard,
            min_text_recall=html_min_text_recall,
            require_css=html_require_css,
            require_rick_image=html_require_rick_image,
        )
    else:
        raise ValueError(f"Unsupported accuracy backend: {accuracy_backend}")

    if not is_match:
        return {"sid": sid, "passed": False, "stage": "stage1", "reason": "answer_wrong"}

    reason = quality_filter(
        messages,
        tool_calls,
        jsonl_path,
        max_tool_rounds,
        reject_host_paths=reject_host_paths,
        reject_missing_images=reject_missing_images,
    )
    if reason:
        return {"sid": sid, "passed": False, "stage": "stage2", "reason": reason}

    return {
        "sid": sid,
        "passed": True,
        "result": {"id": sid, "method": method, "tool_calls": len(tool_calls)},
    }


def main():
    parser = argparse.ArgumentParser(description="Filter sessions: accuracy + tool-call quality")
    parser.add_argument("--sessions-dir", required=True)
    parser.add_argument("--dataset", required=True, help="Original dataset JSONL with 'id' and 'answer'")
    parser.add_argument("--output", required=True, help="Output index JSONL")
    parser.add_argument(
        "--accuracy-backend",
        choices=("rule", "judge", "html_vlm"),
        default="rule",
        help="How to judge answer correctness in stage 1.",
    )
    parser.add_argument("--answer-field", default="answer", help="Dataset field containing reference answer.")
    parser.add_argument(
        "--question-field",
        default="question",
        help="Dataset field containing question text, used by judge backend.",
    )
    parser.add_argument(
        "--media-dir",
        default="",
        help="Directory that stores dataset images. Defaults to <dataset_dir>/images.",
    )
    parser.add_argument(
        "--media-field", default="images", help="Dataset field containing reference image path(s)."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Rule-match tolerance. In generic mode it is relative error for non-zero numeric refs; in point mode it is the normalized coordinate tolerance. Ignored in bbox mode.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("generic", "point", "bbox"),
        default="generic",
        help="Ground-truth format for rule matching. Used only when --accuracy-backend=rule.",
    )
    parser.add_argument("--judge-model", default="gemini-3.0-flash-preview", help="JudgeLLM model name.")
    parser.add_argument(
        "--judge-api-base",
        default="",
        help="OpenAI-compatible API base URL for judge/html_vlm model. Defaults to OPENAI_API_BASE.",
    )
    parser.add_argument("--judge-retries", type=int, default=3, help="Retry count for JudgeLLM calls.")
    parser.add_argument("--judge-max-tokens", type=int, default=16384)
    parser.add_argument("--judge-temperature", type=float, default=0.6)
    parser.add_argument(
        "--html-score-threshold",
        type=float,
        default=80.0,
        help="For --accuracy-backend=html_vlm, VLM score >= this value counts as correct.",
    )
    parser.add_argument(
        "--html-render-dir",
        default="",
        help="Directory for rendered prediction screenshots. Defaults to <output_dir>/rendered_html_acc.",
    )
    parser.add_argument(
        "--keep-rendered-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For html_vlm, keep rendered prediction screenshots on disk after judging.",
    )
    parser.add_argument("--html-render-timeout-ms", type=int, default=10000)
    parser.add_argument("--html-render-wait-ms", type=int, default=500)
    parser.add_argument("--html-default-viewport", default="1280x720")
    parser.add_argument("--html-min-tag-ratio", type=float, default=0.35)
    parser.add_argument("--html-min-structural-jaccard", type=float, default=0.25)
    parser.add_argument("--html-min-text-recall", type=float, default=0.25)
    parser.add_argument("--html-no-require-css", action="store_true")
    parser.add_argument("--html-no-require-rick-image", action="store_true")
    parser.add_argument("--max-tool-rounds", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N dataset rows")
    parser.add_argument(
        "--start-id",
        "--start_id",
        dest="start_id",
        type=int,
        default=None,
        help="Only process sessions with id >= this value.",
    )
    parser.add_argument(
        "--reject-host-paths",
        action="store_true",
        help="Reject sessions leaking host-absolute paths (recommended before conversion).",
    )
    parser.add_argument(
        "--reject-missing-images",
        action="store_true",
        help="Reject sessions referencing images that cannot be resolved on disk.",
    )
    parser.add_argument(
        "--workers",
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=1,
        help="Number of worker threads for session filtering",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        raise ValueError("--workers/--num-workers must be >= 1")
    if args.start_id is not None and args.start_id < 0:
        raise ValueError("--start-id must be >= 0")

    sessions_dir = Path(args.sessions_dir)
    dataset_path = Path(args.dataset).resolve()
    media_dir = Path(args.media_dir).resolve() if args.media_dir else infer_media_dir(dataset_path)
    output_path = Path(args.output)
    html_render_dir = (
        Path(args.html_render_dir).resolve()
        if args.html_render_dir
        else (output_path.resolve().parent / "rendered_html_acc").resolve()
    )
    html_default_viewport = _parse_viewport(args.html_default_viewport)

    # Discover sessions — one iterdir() pass, resolve jsonl path immediately.
    # Handles both plain session_<id> and k-sample session_<id>_s<N> dirs.
    # When multiple variants exist for the same id, keep the first one found
    # (plain session_<id> is preferred because it sorts lexicographically first).
    session_jsonls: dict[int, Path] = {}
    for d in sorted(sessions_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("session_"):
            continue
        rest = d.name[len("session_") :]
        numeric = rest.split("_")[0]
        if not numeric.isdigit():
            continue
        sid = int(numeric)
        if sid in session_jsonls:
            continue  # already have a trace for this id
        # Look for jsonl in sessions/ subdir first, then directly in d
        nested = d / "sessions"
        if nested.is_dir():
            candidates = sorted(p for p in nested.glob("*.jsonl") if p.is_file())
        else:
            candidates = sorted(p for p in d.glob("*.jsonl") if p.is_file())
        if candidates:
            session_jsonls[sid] = candidates[0]

    # Apply --limit (first N rows by dataset order)
    if args.limit is not None:
        limit_ids: set[int] = set()
        with open(args.dataset) as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                line = line.strip()
                if line:
                    limit_ids.add(json.loads(line)["id"])
        session_jsonls = {sid: p for sid, p in session_jsonls.items() if sid in limit_ids}

    if args.start_id is not None:
        session_jsonls = {sid: p for sid, p in session_jsonls.items() if sid >= args.start_id}
        print(f"Start id: {args.start_id}")

    total_sessions = len(session_jsonls)
    print(f"Sessions on disk: {total_sessions}")

    # Resume support: skip ids already recorded in the output index.
    resumed_results: list[dict] = []
    resumed_ids: set[int] = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if "id" in rec and rec["id"] not in resumed_ids:
                        resumed_results.append(rec)
                        resumed_ids.add(rec["id"])
            print(f"[resume] Found existing index with {len(resumed_ids)} entries — skipping those ids.")
        except Exception as e:
            print(f"[resume] Failed to read existing index ({e}); overwriting from scratch.")
            resumed_results = []
            resumed_ids = set()

    # Load reference answers
    ref_items = load_reference_items(
        dataset_path,
        set(session_jsonls.keys()),
        answer_field=args.answer_field,
        question_field=args.question_field,
        media_field=args.media_field,
        media_dir=media_dir,
        require_media=args.accuracy_backend == "html_vlm",
    )
    print(f"Reference items loaded: {len(ref_items)}")

    judge_api_base = args.judge_api_base or os.environ.get("OPENAI_API_BASE")
    judge = None
    if args.accuracy_backend == "judge":
        judge = JudgeLLM(model=args.judge_model, base_url=judge_api_base)
        print(f"Accuracy backend: judge ({judge.model})")
    elif args.accuracy_backend == "html_vlm":
        if not judge_api_base:
            raise EnvironmentError("--judge-api-base or OPENAI_API_BASE is required for html_vlm backend")
        html_judge = VlmHtmlJudge(
            model=args.judge_model,
            base_url=judge_api_base,
            max_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
        )
        html_renderer = HtmlRenderer(timeout_ms=args.html_render_timeout_ms, wait_ms=args.html_render_wait_ms)
        if args.keep_rendered_images:
            html_render_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Accuracy backend: html_vlm ({html_judge.model}), "
            f"score_threshold={args.html_score_threshold}, render_dir={html_render_dir}, "
            f"keep_rendered={args.keep_rendered_images}"
        )
    else:
        html_judge = None
        html_renderer = None
        print(f"Accuracy backend: rule ({args.match_mode})")
    if args.accuracy_backend == "judge":
        html_judge = None
        html_renderer = None

    # Counters
    stage1 = Counter()  # accuracy stage
    stage2 = Counter()  # quality stage
    passed = len(resumed_results)
    results: list[dict] = list(resumed_results)

    sorted_ids = [sid for sid in sorted(session_jsonls.keys()) if sid not in resumed_ids]
    if resumed_ids:
        print(f"To process: {len(sorted_ids)} (resumed {len(resumed_ids)})")
    worker_count = min(args.num_workers, len(sorted_ids)) if sorted_ids else 1

    # Append-mode streaming writer: each passed session is flushed immediately so
    # a crash / Ctrl-C never loses completed work. We reopen in append mode when
    # resuming; otherwise truncate.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resumed_ids else "w"
    write_lock = threading.Lock()
    out_f = open(output_path, write_mode, encoding="utf-8")
    try:
        with tqdm(
            total=len(sorted_ids),
            desc="Filtering",
            unit="session",
            dynamic_ncols=True,
        ) as pbar:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        process_one_session,
                        sid,
                        session_jsonls[sid],
                        ref_items.get(sid),
                        accuracy_backend=args.accuracy_backend,
                        match_mode=args.match_mode,
                        tolerance=args.tolerance,
                        max_tool_rounds=args.max_tool_rounds,
                        reject_host_paths=args.reject_host_paths,
                        reject_missing_images=args.reject_missing_images,
                        judge=judge,
                        html_judge=html_judge,
                        html_renderer=html_renderer,
                        judge_retries=args.judge_retries,
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
                    ): sid
                    for sid in sorted_ids
                }

                try:
                    for future in as_completed(futures):
                        result = future.result()
                        if result["passed"]:
                            passed += 1
                            results.append(result["result"])
                            with write_lock:
                                out_f.write(json.dumps(result["result"], ensure_ascii=False) + "\n")
                                out_f.flush()
                                os.fsync(out_f.fileno())
                        elif result["stage"] == "stage1":
                            stage1[result["reason"]] += 1
                        else:
                            stage2[result["reason"]] += 1

                        pbar.update(1)
                        pbar.set_postfix(passed=passed, workers=worker_count, refresh=False)
                except KeyboardInterrupt:
                    print(
                        f"\n[interrupted] {passed} entries already persisted to {output_path}. "
                        "Rerun the same command to resume.",
                        file=sys.stderr,
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
    finally:
        out_f.close()

    # Rewrite once at the end to sort by id (stream appends in completion order).
    results.sort(key=lambda x: x["id"])
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    total_rejected = total_sessions - passed
    print(f"\n{'=' * 60}")
    print(f"Total sessions:    {total_sessions}")
    print(f"Passed (output):   {passed}")
    print(f"Rejected:          {total_rejected}")
    print(f"\nStage 1 — Accuracy rejection:")
    for reason, count in sorted(stage1.items(), key=lambda x: -x[1]):
        print(f"  {reason:20s}: {count}")
    print(f"  {'TOTAL':20s}: {sum(stage1.values())}")
    print(f"\nStage 2 — Tool-call quality rejection:")
    for reason, count in sorted(stage2.items(), key=lambda x: -x[1]):
        print(f"  {reason:20s}: {count}")
    print(f"  {'TOTAL':20s}: {sum(stage2.values())}")
    print(f"\nIndex written to: {output_path}  ({passed} entries)")


if __name__ == "__main__":
    main()
