"""GUI grounding evaluation: point-in-bbox.

Used for benches like ScreenSpot-Pro where the ground truth is a pixel
bbox `[x1, y1, x2, y2]` and the model is expected to produce a click
coordinate via a `computer_use` tool_call (Qwen3-VL schema, 0-1000
normalized space). We bypass the LLM judge here — the verdict is purely
geometric: predicted point lands inside the bbox or it doesn't.

Two entry points:
  - is_grounding_item(item): heuristic — answer is a 4-list and img_size present
  - score_grounding(turns, pred, gt_answer, img_size): returns
        {"score": 0|1, "point": [x, y]|None, "judge_raw": "..."}
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

# smart_resize + pixel budgets live in core.image (shared with the grounding
# rollout drivers). Localization-trained Qwen2.5-VL models emit ABSOLUTE pixel
# coordinates in the smart-resized input space (not 0-1000 normalized like the
# Qwen3-VL computer_use schema); scoring rescales by the resize ratio.
from ..core.image import (
    smart_resize, IMAGE_FACTOR, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS,
)


# Match `pyautogui.click(x=123, y=456)` / `.moveTo(...)` — the answer format
# Thyme & DeepEyesV2 use for ScreenSpot-Pro grounding.
_PYAUTOGUI_RE = re.compile(
    r"pyautogui\.\w+\(\s*x\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Looser fallback: a bare `x=.., y=..` pair anywhere in the text.
_XY_RE = re.compile(r"x\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*(-?\d+(?:\.\d+)?)",
                    re.IGNORECASE)
# Last-resort fallback: a bare coordinate tuple `(x, y)` or `[x, y]`.
_TUPLE_RE = re.compile(r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]")
_ANSWER_RE_G = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_pyautogui_click(text: str | None) -> tuple[float, float] | None:
    """Pull the model's final click out of the response.

    Prefers the <answer> block when present (that's the commit), then tries
    `pyautogui.click(x=, y=)`, a bare `x=, y=` pair, and finally a `(x, y)` /
    `[x, y]` tuple. Takes the LAST match of whichever form hits."""
    if not text:
        return None
    ans = _ANSWER_RE_G.findall(text)
    scope = ans[-1] if ans else text
    for rx in (_PYAUTOGUI_RE, _XY_RE, _TUPLE_RE):
        matches = rx.findall(scope)
        if matches:
            try:
                x, y = matches[-1]
                return float(x), float(y)
            except (TypeError, ValueError):
                continue
    return None


_BBOX_RE = re.compile(r"^\s*\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,"
                      r"\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]\s*$")


def is_grounding_item(item: dict) -> bool:
    """True iff the item looks like a GUI grounding sample (gt is a pixel
    bbox + img_size is present). Intentionally data-driven so any new
    grounding bench works without a config flag."""
    ans = item.get("answer")
    img_size = item.get("img_size")
    if not isinstance(ans, str) or not isinstance(img_size, (list, tuple)):
        return False
    if len(img_size) != 2:
        return False
    return bool(_BBOX_RE.match(ans))


def parse_bbox(answer: str) -> list[float] | None:
    """Parse a `[x1, y1, x2, y2]` JSON-ish string. Returns floats."""
    try:
        b = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return None
    if not (isinstance(b, list) and len(b) == 4):
        return None
    try:
        return [float(v) for v in b]
    except (TypeError, ValueError):
        return None


def _coerce_xy(coord: Any) -> tuple[float, float] | None:
    """Coerce a coordinate value into (x, y) floats.

    Accepts list/tuple of 2 numbers, OR a string like "[17, 500]" /
    "(17, 500)" / "17, 500" — qwen-style models sometimes hand the model
    a stringified array as the JSON value of `coordinate`.
    """
    if coord is None:
        return None
    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
        try:
            return float(coord[0]), float(coord[1])
        except (TypeError, ValueError):
            return None
    if isinstance(coord, str):
        # Try JSON first.
        try:
            v = json.loads(coord)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return float(v[0]), float(v[1])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        # Fallback: pull the first two numbers out of the string.
        nums = re.findall(r"-?\d+(?:\.\d+)?", coord)
        if len(nums) >= 2:
            try:
                return float(nums[0]), float(nums[1])
            except ValueError:
                return None
    return None


# Coordinate-emitting click actions in the Qwen3-VL computer_use schema.
# `mouse_move` / `left_click_drag` also carry a coordinate but we only
# accept committing actions for the final click.
_CLICK_ACTIONS = {
    "left_click", "right_click", "middle_click",
    "double_click", "triple_click",
    "left_click_drag", "mouse_move",
}


def _extract_from_args(args: Any) -> tuple[float, float] | None:
    """Pull (x, y) out of a parsed `computer_use` tool_call args dict.
    Returns None if the action is not a click or there's no usable coord.
    """
    if not isinstance(args, dict):
        return None
    action = args.get("action")
    if action not in _CLICK_ACTIONS:
        return None
    return _coerce_xy(args.get("coordinate"))


def _iter_computer_use_calls_from_turns(turns: Iterable[dict]
                                        ) -> Iterable[tuple[dict, str]]:
    """Yield (parsed_args, source_tag) for every computer_use call in turns,
    in chronological order. Looks at both `tool_results[*].args` (already
    parsed by the agent loop) and `message.tool_calls[*].function.arguments`
    (raw JSON string — handles malformed/unregistered cases too).
    """
    for t in turns or []:
        # Parsed tool results — fast path.
        for tr in t.get("tool_results") or []:
            if tr.get("name") == "computer_use":
                args = tr.get("args")
                if args is None:
                    raw = tr.get("args_raw")
                    if isinstance(raw, str):
                        try:
                            args = json.loads(raw)
                        except json.JSONDecodeError:
                            args = None
                if args is not None:
                    yield args, "tool_results"

        # Assistant tool_calls (covers the case where the call wasn't
        # executed by the registry — e.g. trace from a step where the
        # model emitted computer_use but the loop bailed before exec).
        msg = t.get("message") or {}
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") != "computer_use":
                continue
            raw = fn.get("arguments")
            if isinstance(raw, dict):
                yield raw, "tool_calls"
            elif isinstance(raw, str):
                try:
                    yield json.loads(raw), "tool_calls"
                except json.JSONDecodeError:
                    continue


# Fallback for when no structured tool_call survived (e.g. the model
# wrote the call as plain text). We look for `"coordinate": [x, y]` in
# pred text, taking the LAST one — the model's final commit.
# Also tolerates the stringified-array variant `"coordinate": "[x, y]"`
# that some models emit when they treat the value type loosely.
_TEXT_COORD_RE = re.compile(
    r'"coordinate"\s*:\s*"?\s*\[?\s*"?(-?\d+(?:\.\d+)?)"?\s*,\s*'
    r'"?(-?\d+(?:\.\d+)?)"?\s*\]?\s*"?'
)


# Hermes-style text tool_call block used in GUI grounding --no-tools prompts:
#   <tool_call>
#   <function=computer_use>
#   <parameter=action>left_click</parameter>
#   <parameter=coordinate>[542, 74]</parameter>
#   </function>
#   </tool_call>
# We accept whitespace and newlines around tags; `function` may also appear
# as a bare `<function=name>` or nested inside the outer block.
_HERMES_TOOL_CALL_RE = re.compile(
    r"<tool_call\b[^>]*>(.*?)</tool_call\s*>", re.DOTALL | re.IGNORECASE
)
_HERMES_FUNCTION_RE = re.compile(
    r"<function\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</function\s*>",
    re.DOTALL | re.IGNORECASE,
)
_HERMES_PARAM_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _parse_hermes_tool_calls(text: str) -> list[dict]:
    """Parse Hermes-style <tool_call>...</tool_call> blocks into arg dicts.

    Returns a list of {'name': ..., 'args': {param: value}} entries, in the
    order they appear. Only well-formed blocks are returned; malformed ones
    are silently skipped so a bad block doesn't wipe out a good later one.
    Parameter values are stripped and, when they look like JSON, parsed —
    so `coordinate` comes back as a real list when the model emits `[x, y]`.
    """
    out: list[dict] = []
    if not text:
        return out
    for tc_match in _HERMES_TOOL_CALL_RE.finditer(text):
        body = tc_match.group(1)
        for fn_match in _HERMES_FUNCTION_RE.finditer(body):
            fname = fn_match.group(1)
            fbody = fn_match.group(2)
            args: dict = {}
            for p_match in _HERMES_PARAM_RE.finditer(fbody):
                pname = p_match.group(1)
                pval = p_match.group(2).strip()
                # Try JSON first so `[542, 74]` and numeric strings come back
                # as typed values; fall back to the raw string otherwise.
                try:
                    args[pname] = json.loads(pval)
                except (json.JSONDecodeError, ValueError):
                    args[pname] = pval
            out.append({"name": fname, "args": args})
    return out


def extract_click_point(turns: list[dict] | None, pred: str | None
                        ) -> tuple[tuple[float, float] | None, str]:
    """Return ((x, y) in 0-1000 normalized space, source_tag).

    `source_tag` is one of: "tool_results", "tool_calls", "hermes_text",
    "pred_text", or "" when no usable point was found.

    Strategy: take the LAST click-action computer_use call we can find
    in turns. If turns yield nothing usable, parse Hermes-style
    <tool_call> blocks from `pred` (the format used in --no-tools GUI
    grounding prompts). As a last resort, scan `pred` for a literal
    `"coordinate": [x, y]` JSON snippet.
    """
    last: tuple[tuple[float, float], str] | None = None
    for args, src in _iter_computer_use_calls_from_turns(turns or []):
        xy = _extract_from_args(args)
        if xy is not None:
            last = (xy, src)
    if last is not None:
        return last[0], last[1]

    if pred:
        # Hermes-style <tool_call> text blocks (--no-tools GUI grounding).
        hermes = _parse_hermes_tool_calls(pred)
        last_hermes_xy: tuple[float, float] | None = None
        for call in hermes:
            if call.get("name") != "computer_use":
                continue
            xy = _extract_from_args(call.get("args"))
            if xy is not None:
                last_hermes_xy = xy
        if last_hermes_xy is not None:
            return last_hermes_xy, "hermes_text"

        # Plain JSON snippet fallback: `"coordinate": [x, y]`.
        matches = _TEXT_COORD_RE.findall(pred)
        if matches:
            x_str, y_str = matches[-1]
            try:
                return (float(x_str), float(y_str)), "pred_text"
            except ValueError:
                pass

    return None, ""


def is_point_in_bbox(point: tuple[float, float],
                     bbox_pixels: list[float],
                     img_size: list[int] | tuple[int, int],
                     normalized: bool = True) -> bool:
    """Decide whether `point` (default in 0-1000 normalized space) falls
    inside `bbox_pixels` (= [x1, y1, x2, y2] in original image pixels).

    Bbox edges are inclusive (a click on the boundary counts as a hit).
    """
    if len(bbox_pixels) != 4 or len(img_size) != 2:
        return False
    x, y = point
    w, h = float(img_size[0]), float(img_size[1])
    if normalized:
        # Convert 0-1000 → pixel.
        if w <= 0 or h <= 0:
            return False
        px = x / 1000.0 * w
        py = y / 1000.0 * h
    else:
        px, py = float(x), float(y)
    x1, y1, x2, y2 = bbox_pixels
    # Some sources give bbox with x2 < x1 etc — normalize.
    lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
    lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
    return lo_x <= px <= hi_x and lo_y <= py <= hi_y


def score_grounding(turns: list[dict] | None, pred: str | None,
                    gt_answer: str, img_size: list[int] | tuple[int, int],
                    ) -> dict:
    """Run the full grounding verdict: extract click → in-bbox check.

    Returns:
        {
          "score": 0 | 1,
          "extracted": "(x, y)"  | "NONE",
          "point": [x, y] | None,    # 0-1000 normalized
          "point_pixels": [px, py] | None,
          "bbox": [x1,y1,x2,y2] | None,
          "img_size": [w, h] | None,
          "source": "tool_results"|"tool_calls"|"pred_text"|"" ,
          "judge_raw": "<one-line audit>"
        }
    """
    bbox = parse_bbox(gt_answer)
    point, src = extract_click_point(turns, pred)

    if bbox is None:
        return {
            "score": 0,
            "extracted": "NONE",
            "point": None,
            "point_pixels": None,
            "bbox": None,
            "img_size": list(img_size) if img_size else None,
            "source": src,
            "judge_raw": f"point=NONE bbox=INVALID({gt_answer!r}) in_box=False",
        }

    if point is None:
        return {
            "score": 0,
            "extracted": "NONE",
            "point": None,
            "point_pixels": None,
            "bbox": bbox,
            "img_size": list(img_size),
            "source": src,
            "judge_raw": f"point=NONE bbox={bbox} in_box=False",
        }

    in_box = is_point_in_bbox(point, bbox, img_size, normalized=True)
    w, h = float(img_size[0]), float(img_size[1])
    px = round(point[0] / 1000.0 * w, 2)
    py = round(point[1] / 1000.0 * h, 2)
    return {
        "score": int(in_box),
        "extracted": f"({point[0]}, {point[1]})",
        "point": [point[0], point[1]],
        "point_pixels": [px, py],
        "bbox": bbox,
        "img_size": list(img_size),
        "source": src or "unknown",
        "judge_raw": (f"point={point} (px={px},{py}) bbox={bbox} "
                      f"img_size={list(img_size)} in_box={in_box} src={src or 'unknown'}"),
    }


def score_grounding_pyautogui(pred: str | None, gt_answer: str,
                              img_size: list[int] | tuple[int, int],
                              max_pixels: int = DEFAULT_MAX_PIXELS,
                              min_pixels: int = DEFAULT_MIN_PIXELS) -> dict:
    """Grounding verdict for the Qwen2.5-VL family (native + Thyme / DeepEyesV2).

    These models commit a bare ``(x, y)`` point (older prompts used
    ``pyautogui.click(x=, y=)``; :func:`parse_pyautogui_click` accepts both) in
    ABSOLUTE pixels within the smart-resized input space. We scale that point
    back to original-image pixels (by the resize ratio) and test point-in-bbox.
    Coordinates already in [0, 1] are treated as normalized (matches the
    upstream ScreenSpot-Pro eval, which divides by image size only when a
    coordinate exceeds 1).

    Returns the same dict shape as :func:`score_grounding`.
    """
    bbox = parse_bbox(gt_answer)
    click = parse_pyautogui_click(pred)
    img_sz = list(img_size) if img_size else None

    if bbox is None or click is None or not img_sz or len(img_sz) != 2:
        reason = ("bbox=INVALID" if bbox is None else
                  "point=NONE" if click is None else "img_size=INVALID")
        return {
            "score": 0, "extracted": "NONE", "point": None,
            "point_pixels": None, "bbox": bbox, "img_size": img_sz,
            "source": "coord", "judge_raw": f"{reason} in_box=False",
        }

    w, h = float(img_sz[0]), float(img_sz[1])
    x, y = click
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        # Normalized [0, 1] coordinates.
        px, py = x * w, y * h
        space = "norm01"
    else:
        # Absolute pixels in the smart-resized space → scale to original.
        rh, rw = smart_resize(int(h), int(w), IMAGE_FACTOR, min_pixels, max_pixels)
        px = x * (w / rw) if rw else x
        py = y * (h / rh) if rh else y
        space = f"resized={rw}x{rh}"

    in_box = is_point_in_bbox((px, py), bbox, img_sz, normalized=False)
    px, py = round(px, 2), round(py, 2)
    return {
        "score": int(in_box),
        "extracted": f"({click[0]}, {click[1]})",
        "point": [click[0], click[1]],
        "point_pixels": [px, py],
        "bbox": bbox,
        "img_size": img_sz,
        "source": "coord",
        "judge_raw": (f"click={click} ({space}) px=({px},{py}) bbox={bbox} "
                      f"img_size={img_sz} in_box={in_box}"),
    }
