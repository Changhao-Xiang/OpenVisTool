"""Evaluate session predictions against ground truth with rule-based matching.

Reads session JSONL files produced by distill/run.py, collects the final
assistant-visible response for each sample, and uses deterministic rules to
decide whether it matches the dataset ground truth.

Output: one JSONL line per session with correctness judgement and match method.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Rule-based answer matching
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"(?<![a-zA-Z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:%)?")
_TRIVIAL_ANSWERS = {"0", "1", "2", "3", "4", "5", "yes", "no", "true", "false"}
_ANSWER_SIGNAL = re.compile(
    r"[^.。\n]*(?:answer|result|therefore|thus|so the|= \*\*|approximately|"
    r"is about|final|overall|in summary|conclusion|the gap|the difference)[^.。\n]*",
    re.I,
)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>")
_TOOL_CALL_RE = re.compile(r"<tool_call>[\s\S]*?</tool_call>")
_PAIR_RE = re.compile(
    r"[\[(]\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
    r"[\])]"
)
_BOX_RE = re.compile(
    r"[\[(]\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
    r"[\])]"
)
_NAMED_POINT_X_RE = re.compile(
    r'(?<!\w)(?:["\']?x["\']?|cx|center_x)\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', re.I
)
_NAMED_POINT_Y_RE = re.compile(
    r'(?<!\w)(?:["\']?y["\']?|cy|center_y)\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', re.I
)
_NAMED_BOX_RE = re.compile(
    r"(?is)"
    r"(?:x1|left)\s*[:=]\s*(?P<x1>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r".*?"
    r"(?:y1|top)\s*[:=]\s*(?P<y1>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r".*?"
    r"(?:x2|right)\s*[:=]\s*(?P<x2>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r".*?"
    r"(?:y2|bottom)\s*[:=]\s*(?P<y2>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
_GUI_ACTION_RE = re.compile(
    r"(?is)(?:pyautogui\.)?"
    r"(?P<name>click|doubleclick|double_click|rightclick|right_click|moveto|move_to|dragto|drag_to|mousedown|mouse_down|mouseup|mouse_up)"
    r"\s*\((?P<args>[^)]*)\)"
)
_MEDIA_FIELD_CANDIDATES = ("images", "image", "img", "screenshot", "screenshots")


def normalize_answer(ans: str) -> str:
    ans = ans.strip().lower()
    try:
        val = float(ans.rstrip("%"))
        if val == int(val) and "e" not in ans.lower():
            return str(int(val))
        return str(val)
    except ValueError:
        return ans


def extract_boxed(text: str) -> list[str]:
    results = []
    for match in re.finditer(r"\\\\?boxed\s*\{", text):
        start = match.end()
        depth = 1
        idx = start
        while idx < len(text) and depth > 0:
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            results.append(text[start : idx - 1].strip())
    return results


def extract_bold(text: str) -> list[str]:
    return re.findall(r"\*\*(.+?)\*\*", text)


def try_float(value: str) -> float | None:
    value = value.strip().rstrip("%")
    try:
        return float(value)
    except ValueError:
        return None


def numeric_close(val: float, ref: float, tol: float) -> bool:
    if abs(ref) > 1e-9:
        return abs(val - ref) / abs(ref) <= tol
    return False


def extract_numbers(text: str) -> list[float]:
    nums = []
    for match in _NUM_RE.finditer(text):
        value = try_float(match.group())
        if value is not None:
            nums.append(value)
    return nums


def _has_answer_context(text: str, ref_lower: str) -> bool:
    esc = re.escape(ref_lower)
    patterns = [
        rf"(?:answer|result|total|difference|gap|value|count|sum)\s*(?:is|=|:)\s*\**\s*{esc}\b",
        rf"=\s*\**\s*{esc}\b(?!\s*[%)\]])",
        rf"\*\*{esc}\*\*",
        rf"\\\\?boxed\{{\s*{esc}\s*\}}",
    ]
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def extract_answer_sentences(text: str) -> str:
    matches = _ANSWER_SIGNAL.findall(text)
    return " ".join(matches)


def _make_point(x: float, y: float) -> tuple[float, float]:
    return (float(x), float(y))


def _make_box(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    return (left, top, right, bottom)


def _coords_in_unit_range(values: tuple[float, ...]) -> bool:
    return all(0.0 <= value <= 1.0 for value in values)


def _normalize_prediction_point(point: tuple[float, float]) -> tuple[float, float]:
    if _coords_in_unit_range(point):
        return point
    return (point[0] / 1000.0, point[1] / 1000.0)


def _normalize_reference_point(
    point: tuple[float, float], image_size: tuple[int, int] | None
) -> tuple[float, float]:
    if _coords_in_unit_range(point) or image_size is None:
        return point
    width, height = image_size
    if width <= 0 or height <= 0:
        return point
    return (point[0] / width, point[1] / height)


def _normalize_reference_box(
    box: tuple[float, float, float, float], image_size: tuple[int, int] | None
) -> tuple[float, float, float, float]:
    if _coords_in_unit_range(box) or image_size is None:
        return box
    width, height = image_size
    if width <= 0 or height <= 0:
        return box
    return _make_box(box[0] / width, box[1] / height, box[2] / width, box[3] / height)


def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped = []
    seen = set()
    for x, y in points:
        key = (round(x, 8), round(y, 8))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((x, y))
    return deduped


def _literal_structure(value: str) -> object | None:
    text = value.strip()
    if not text or text[0] not in "[{(":
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def _coerce_numeric_sequence(value: object, expected_len: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != expected_len:
        return None
    numbers = []
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, (int, float)):
            numbers.append(float(item))
        else:
            item_float = try_float(str(item))
            if item_float is None:
                return None
            numbers.append(item_float)
    return numbers


def _parse_reference_point(text: str) -> tuple[float, float] | None:
    literal = _literal_structure(text)
    if literal is not None:
        if isinstance(literal, dict):
            if "point" in literal:
                values = _coerce_numeric_sequence(literal["point"], 2)
                if values is not None:
                    return _make_point(values[0], values[1])
            if {"x", "y"} <= set(literal):
                x_val = try_float(str(literal["x"]))
                y_val = try_float(str(literal["y"]))
                if x_val is not None and y_val is not None:
                    return _make_point(x_val, y_val)
        else:
            values = _coerce_numeric_sequence(literal, 2)
            if values is not None:
                return _make_point(values[0], values[1])

    match = _PAIR_RE.search(text)
    if match:
        return _make_point(float(match.group(1)), float(match.group(2)))

    x_match = _NAMED_POINT_X_RE.search(text)
    y_match = _NAMED_POINT_Y_RE.search(text)
    if x_match and y_match:
        return _make_point(float(x_match.group(1)), float(y_match.group(1)))
    return None


def _parse_reference_box(text: str) -> tuple[float, float, float, float] | None:
    literal = _literal_structure(text)
    if literal is not None:
        if isinstance(literal, dict):
            for key in ("bbox", "box"):
                if key in literal:
                    values = _coerce_numeric_sequence(literal[key], 4)
                    if values is not None:
                        return _make_box(*values)
            if {"x1", "y1", "x2", "y2"} <= set(literal):
                coords = [try_float(str(literal[key])) for key in ("x1", "y1", "x2", "y2")]
                if all(value is not None for value in coords):
                    return _make_box(*coords)
        else:
            values = _coerce_numeric_sequence(literal, 4)
            if values is not None:
                return _make_box(*values)

    match = _BOX_RE.search(text)
    if match:
        return _make_box(*[float(match.group(i)) for i in range(1, 5)])

    named_match = _NAMED_BOX_RE.search(text)
    if named_match:
        return _make_box(
            float(named_match.group("x1")),
            float(named_match.group("y1")),
            float(named_match.group("x2")),
            float(named_match.group("y2")),
        )
    return None


def _extract_action_points(text: str) -> list[tuple[float, float]]:
    points = []
    for match in _GUI_ACTION_RE.finditer(text):
        args = match.group("args")
        x_match = _NAMED_POINT_X_RE.search(args)
        y_match = _NAMED_POINT_Y_RE.search(args)
        if x_match and y_match:
            points.append(_make_point(float(x_match.group(1)), float(y_match.group(1))))
            continue

        values = extract_numbers(args)
        if len(values) >= 2:
            points.append(_make_point(values[0], values[1]))
    return _dedupe_points(points)


def _extract_response_points(text: str) -> list[tuple[float, float]]:
    points = _extract_action_points(text)
    points.extend(
        _make_point(float(match.group(1)), float(match.group(2))) for match in _PAIR_RE.finditer(text)
    )

    x_matches = list(_NAMED_POINT_X_RE.finditer(text))
    y_matches = list(_NAMED_POINT_Y_RE.finditer(text))
    if x_matches and y_matches:
        for x_match in x_matches:
            nearest_y = min(y_matches, key=lambda y_match: abs(y_match.start() - x_match.start()))
            if abs(nearest_y.start() - x_match.start()) <= 80:
                points.append(_make_point(float(x_match.group(1)), float(nearest_y.group(1))))
    return _dedupe_points(points)


def _point_close(point: tuple[float, float], ref_point: tuple[float, float], tolerance: float) -> bool:
    return abs(point[0] - ref_point[0]) <= tolerance and abs(point[1] - ref_point[1]) <= tolerance


def _point_in_box(
    point: tuple[float, float], box: tuple[float, float, float, float], tolerance: float = 0.0
) -> bool:
    x1, y1, x2, y2 = box
    return (x1 - tolerance) <= point[0] <= (x2 + tolerance) and (y1 - tolerance) <= point[1] <= (
        y2 + tolerance
    )


def _check_point_match(
    response: str,
    ref_answer: str,
    tolerance: float,
    image_size: tuple[int, int] | None,
) -> tuple[bool, str]:
    ref_point = _parse_reference_point(ref_answer)
    if ref_point is None:
        return False, "none"
    ref_point = _normalize_reference_point(ref_point, image_size)

    response_points = _extract_response_points(response)
    for point in response_points:
        point = _normalize_prediction_point(point)
        if _point_close(point, ref_point, tolerance):
            return True, "point"

    return False, "none"


def _check_bbox_match(
    response: str,
    ref_answer: str,
    image_size: tuple[int, int] | None,
) -> tuple[bool, str]:
    ref_box = _parse_reference_box(ref_answer)
    if ref_box is None:
        return False, "none"
    ref_box = _normalize_reference_box(ref_box, image_size)

    response_points = _extract_response_points(response)
    for point in response_points:
        point = _normalize_prediction_point(point)
        if _point_in_box(point, ref_box):
            return True, "point_in_bbox"

    return False, "none"


def check_match(
    response: str,
    ref_answer: str,
    *,
    tolerance: float = 0.0,
    mode: str,
    image_size: tuple[int, int] | None = None,
) -> tuple[bool, str]:
    if mode not in {"generic", "point", "bbox"}:
        raise ValueError(f"Unsupported match mode: {mode}")

    if mode == "point":
        return _check_point_match(response, ref_answer, tolerance=tolerance, image_size=image_size)
    if mode == "bbox":
        return _check_bbox_match(response, ref_answer, image_size=image_size)

    norm_ref = normalize_answer(ref_answer)
    ref_float = try_float(ref_answer)
    ref_is_near_zero = ref_float is not None and abs(ref_float) <= 1e-9

    for value in extract_boxed(response):
        if normalize_answer(value) == norm_ref:
            return True, "boxed"
    if ref_is_near_zero:
        for value in extract_boxed(response):
            candidate = try_float(value)
            if candidate == 0.0:
                return True, "boxed_zero"
    elif tolerance > 0 and ref_float is not None:
        for value in extract_boxed(response):
            candidate = try_float(value)
            if candidate is not None and numeric_close(candidate, ref_float, tolerance):
                return True, "boxed_approx"

    bold_vals = extract_bold(response)
    for value in bold_vals:
        if normalize_answer(value) == norm_ref:
            return True, "bold"
    if ref_is_near_zero:
        for value in bold_vals:
            candidate = try_float(value)
            if candidate == 0.0:
                return True, "bold_zero"
            for number in extract_numbers(value):
                if number == 0.0:
                    return True, "bold_zero"
    elif tolerance > 0 and ref_float is not None:
        for value in bold_vals:
            candidate = try_float(value)
            if candidate is not None and numeric_close(candidate, ref_float, tolerance):
                return True, "bold_approx"
            for number in extract_numbers(value):
                if numeric_close(number, ref_float, tolerance):
                    return True, "bold_approx"

    resp_lower = response.lower()
    ref_lower = ref_answer.strip().lower()

    if ref_lower in _TRIVIAL_ANSWERS:
        if _has_answer_context(response, ref_lower):
            return True, "literal_contextual"
    else:
        if re.search(r"(?<!\w)" + re.escape(ref_lower) + r"(?!\w)", resp_lower):
            return True, "literal"

    if norm_ref != ref_lower and norm_ref not in _TRIVIAL_ANSWERS:
        if re.search(r"(?<!\w)" + re.escape(norm_ref) + r"(?!\w)", resp_lower):
            return True, "literal_norm"

    ref_is_trivial = ref_lower in _TRIVIAL_ANSWERS or norm_ref in _TRIVIAL_ANSWERS
    if ref_is_near_zero and not ref_is_trivial:
        answer_text = extract_answer_sentences(response)
        if answer_text:
            numbers = extract_numbers(answer_text)
            if any(number == 0.0 for number in numbers):
                return True, "numeric_zero"
        else:
            numbers = extract_numbers(response)
            if any(number == 0.0 for number in numbers):
                return True, "numeric_zero_fallback"
    elif tolerance > 0 and ref_float is not None and not ref_is_trivial:
        answer_text = extract_answer_sentences(response)
        if answer_text:
            numbers = extract_numbers(answer_text)
            if numbers:
                closest = min(numbers, key=lambda value: abs(value - ref_float))
                if numeric_close(closest, ref_float, tolerance):
                    return True, "numeric_approx"
        else:
            numbers = extract_numbers(response)
            if numbers:
                closest = min(numbers, key=lambda value: abs(value - ref_float))
                if numeric_close(closest, ref_float, tolerance * 0.5):
                    return True, "numeric_approx_fallback"

    return False, "none"


# ---------------------------------------------------------------------------
# Session parsing
# ---------------------------------------------------------------------------


def _load_records(path: Path) -> list[dict] | None:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return None


def _extract_events(records: list[dict]) -> list[dict]:
    if not records:
        return []

    events = records[1:] if records[0].get("_type") == "metadata" else records
    last_user_idx = -1
    for idx, record in enumerate(events):
        if record.get("role") == "user":
            last_user_idx = idx
    return events[last_user_idx:] if last_user_idx >= 0 else events


def _parse_session(session_path: Path) -> tuple[str | None, bool]:
    """Return the final assistant-visible content and tool-call flag."""
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


# ---------------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------------


def _find_session_trace(sessions_dir: Path, item_id: int) -> Path | None:
    session_dir = sessions_dir / f"session_{item_id}"
    if not session_dir.is_dir():
        return None

    nested = session_dir / "sessions"
    candidates = sorted(p for p in nested.glob("*.jsonl") if p.is_file()) if nested.is_dir() else []
    if candidates:
        return candidates[0]

    direct_candidates = sorted(p for p in session_dir.glob("*.jsonl") if p.is_file())
    return direct_candidates[0] if direct_candidates else None


def _candidate_media_dirs(dataset_path: Path) -> list[Path]:
    dataset_dir = dataset_path.parent
    candidates = [dataset_dir]
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and ("image" in child.name.lower() or "screenshot" in child.name.lower()):
            candidates.append(child)
    return candidates


@lru_cache(maxsize=4096)
def _resolve_media_path(dataset_path_str: str, raw_path: str) -> str | None:
    dataset_path = Path(dataset_path_str)
    stripped = raw_path.strip()
    if not stripped:
        return None

    path = Path(stripped).expanduser()
    if path.is_absolute():
        return str(path.resolve()) if path.is_file() else None

    for media_dir in _candidate_media_dirs(dataset_path):
        candidate = (media_dir / path).resolve()
        if candidate.is_file():
            return str(candidate)
    return None


@lru_cache(maxsize=4096)
def _load_image_size(image_path_str: str) -> tuple[int, int] | None:
    image_path = Path(image_path_str)
    if not image_path.is_file():
        return None
    with Image.open(image_path) as image:
        width, height = image.size
    return (width, height)


def _extract_item_image_size(item: dict, dataset_path: Path) -> tuple[int, int] | None:
    for width_key, height_key in (
        ("width", "height"),
        ("image_width", "image_height"),
        ("img_w", "img_h"),
    ):
        width = item.get(width_key)
        height = item.get(height_key)
        if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
            return (int(width), int(height))

    for field in _MEDIA_FIELD_CANDIDATES:
        raw_value = item.get(field)
        if raw_value is None:
            continue
        raw_paths = raw_value if isinstance(raw_value, list) else [raw_value]
        for raw_path in raw_paths:
            if not isinstance(raw_path, str):
                continue
            resolved = _resolve_media_path(str(dataset_path), raw_path)
            if resolved is None:
                continue
            image_size = _load_image_size(resolved)
            if image_size is not None:
                return image_size
    return None


def evaluate_dataset(
    dataset_path: Path,
    sessions_dir: Path,
    *,
    answer_field: str,
    tolerance: float,
    match_mode: str,
    limit: int | None = None,
) -> list[dict]:
    items = [
        json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if limit is not None:
        items = items[:limit]

    results = []
    for item in items:
        item_id = item.get("id")
        if item_id is None:
            continue

        trace_path = _find_session_trace(sessions_dir, item_id)
        if trace_path is None:
            continue

        predicted, tool_called = _parse_session(trace_path)
        ground_truth = str(item.get(answer_field, "")).strip()
        image_size = _extract_item_image_size(item, dataset_path)
        correct, method = (
            check_match(
                predicted or "",
                ground_truth,
                tolerance=tolerance,
                mode=match_mode,
                image_size=image_size,
            )
            if ground_truth
            else (False, "none")
        )
        results.append(
            {
                "id": item_id,
                "question": str(item.get("question", "")).strip(),
                "ground_truth": ground_truth,
                "predicted": predicted or "",
                "tool_called": tool_called,
                "correct": correct,
                "method": method,
            }
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate final assistant responses against ground truth with rule-based matching."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sessions-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Match tolerance. In generic mode it is relative error for non-zero numeric refs; in point mode it is the normalized coordinate tolerance. Ignored in bbox mode.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("generic", "point", "bbox"),
        required=True,
        help="Ground-truth format: generic text/numeric answer, point coordinate, or bbox.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    sessions_dir = Path(args.sessions_dir).resolve()
    output_path = Path(args.output).resolve()

    print(
        f"Dataset: {dataset_path}\nSessions: {sessions_dir}\nAnswer field: {args.answer_field}\n"
        f"Tolerance: {args.tolerance}\nMatch mode: {args.match_mode}"
    )
    results = evaluate_dataset(
        dataset_path,
        sessions_dir,
        answer_field=args.answer_field,
        tolerance=args.tolerance,
        match_mode=args.match_mode,
        limit=args.limit,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for result in results:
            file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = len(results)
    correct = sum(1 for result in results if result["correct"])
    print(
        f"\nTotal: {total}, Correct: {correct}, Accuracy: {correct / total:.3f}" if total else "No results."
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
