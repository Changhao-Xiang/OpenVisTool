"""Select samples with positive tool-call gain.

Compares the base model's no-tool avg@k (from `filter_difficulty.py`'s progress
file or any JSONL that exposes `id` + `avg_k`) against the avg@k obtained when
the same model is given the teacher's tool_response prefix (output of
`filter_tool_gain_with_prefix.py`). Keeps samples whose gain is above a
threshold, optionally with floor/ceiling constraints.

Output schema mirrors `workspaces/.../ChartVerse-SFT_filtered_index_17k.jsonl`:
one record per kept sample with fields ``id``, ``avg_at_k_with_tool``,
``avg_at_k_no_tool``, and ``tool_calls`` (the number of tool calls the teacher
made for that sample, counted from its session trace).

Example:
    python distill/filter/select_tool_gain.py \\
        --notool-jsonl  .../foo.progress.jsonl \\
        --with-prefix-jsonl .../foo_tool_gain_prefix.jsonl \\
        --gain-threshold 0.25 \\
        --output .../tool_gain_selected.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    # Progress jsonls are written by concurrent rollout workers and can have
    # partial / interleaved lines from races. Skip those with a warning rather
    # than aborting — one missing stats row is far less harmful than losing the
    # whole dataset's selection.
    records: list[dict[str, Any]] = []
    skipped = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"warn: skipping malformed line in {path}:{line_no}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if not isinstance(record, dict):
            print(f"warn: skipping non-object line in {path}:{line_no}", file=sys.stderr)
            skipped += 1
            continue
        records.append(record)
    if skipped:
        print(f"warn: skipped {skipped} malformed line(s) in {path}", file=sys.stderr)
    return records


def _index_notool_scores(records: list[dict[str, Any]], score_field: str) -> dict[Any, float]:
    out: dict[Any, float] = {}
    for record in records:
        if "id" not in record or score_field not in record:
            continue
        score = record[score_field]
        if score is None:
            continue
        try:
            out[record["id"]] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def _count_tool_calls(trace_path: Path) -> int:
    """Count assistant tool_call invocations in a teacher session trace."""
    if not trace_path.is_file():
        return 0
    total = 0
    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if record.get("role") == "assistant":
            calls = record.get("tool_calls") or []
            if isinstance(calls, list):
                total += len(calls)
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select samples whose avg@k_with_prefix - avg_k_notool >= threshold."
    )
    p.add_argument(
        "--notool-jsonl",
        required=True,
        help=(
            "JSONL with no-tool avg@k. Typically the .progress.jsonl produced by "
            "filter_difficulty.py — it has every sample, not just the kept ones."
        ),
    )
    p.add_argument(
        "--with-prefix-jsonl",
        required=True,
        help="Output of filter_tool_gain_with_prefix.py (provides avg_k_with_prefix and trace_path).",
    )
    p.add_argument("--output", required=True, help="Output JSONL of selected samples.")
    p.add_argument("--report", default="", help="Optional summary JSON report path.")
    p.add_argument(
        "--notool-score-field",
        default="avg_k_notool",
        help="Field name for the no-tool avg@k score (default: avg_k_notool).",
    )
    p.add_argument(
        "--with-prefix-score-field",
        default="avg_k_with_prefix",
        help="Field name for the with-prefix avg@k score (default: avg_k_with_prefix).",
    )
    p.add_argument(
        "--gain-threshold",
        type=float,
        default=0.25,
        help="Keep samples with (with_prefix - notool) >= threshold (default: 0.25).",
    )
    p.add_argument(
        "--max-notool",
        type=float,
        default=None,
        help="Optional upper bound on notool avg@k (drop already-easy samples).",
    )
    p.add_argument(
        "--min-with-prefix",
        type=float,
        default=None,
        help="Optional lower bound on with-prefix avg@k (drop samples the prefix still can't solve).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    notool_path = Path(args.notool_jsonl).resolve()
    with_prefix_path = Path(args.with_prefix_jsonl).resolve()
    output_path = Path(args.output).resolve()

    for label, path in [("notool-jsonl", notool_path), ("with-prefix-jsonl", with_prefix_path)]:
        if not path.is_file():
            raise FileNotFoundError(f"--{label} not found: {path}")

    with_prefix_records = _load_jsonl(with_prefix_path)
    notool_scores = _index_notool_scores(_load_jsonl(notool_path), args.notool_score_field)

    kept = 0
    missing_notool = 0
    missing_score = 0
    missing_trace = 0
    invalid_with_prefix = 0
    below_gain = 0
    above_max_notool = 0
    below_min_prefix = 0
    gains: list[float] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in with_prefix_records:
            sample_id = record.get("id")
            if sample_id is None:
                continue

            with_prefix = record.get(args.with_prefix_score_field)
            # Upstream marks samples whose teacher trace had no tool_calls by
            # setting avg_k_with_prefix=None. Older runs of
            # filter_tool_gain_with_prefix.py predate that change and instead
            # leave a numeric score with prefix_tool_responses==0 — treat both
            # as invalid.
            if with_prefix is None or record.get("prefix_tool_responses") == 0:
                invalid_with_prefix += 1
                continue
            try:
                with_prefix = float(with_prefix)
            except (TypeError, ValueError):
                missing_score += 1
                continue

            notool = notool_scores.get(sample_id)
            if notool is None:
                missing_notool += 1
                continue

            if args.max_notool is not None and notool > args.max_notool:
                above_max_notool += 1
                continue
            if args.min_with_prefix is not None and with_prefix < args.min_with_prefix:
                below_min_prefix += 1
                continue

            gain = with_prefix - notool
            if gain < args.gain_threshold:
                below_gain += 1
                continue

            trace_path_raw = record.get("trace_path")
            if not trace_path_raw:
                missing_trace += 1
                continue
            tool_call_count = _count_tool_calls(Path(trace_path_raw))

            payload = {
                "id": sample_id,
                "avg_at_k_with_tool": round(with_prefix, 4),
                "avg_at_k_no_tool": round(notool, 4),
                "tool_calls": tool_call_count,
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1
            gains.append(gain)

    mean_gain = sum(gains) / len(gains) if gains else 0.0
    summary = {
        "with_prefix_total": len(with_prefix_records),
        "notool_indexed": len(notool_scores),
        "kept": kept,
        "dropped_invalid_with_prefix": invalid_with_prefix,
        "dropped_missing_notool": missing_notool,
        "dropped_missing_score": missing_score,
        "dropped_missing_trace": missing_trace,
        "dropped_below_gain": below_gain,
        "dropped_above_max_notool": above_max_notool,
        "dropped_below_min_with_prefix": below_min_prefix,
        "kept_mean_gain": round(mean_gain, 4),
        "params": {
            "gain_threshold": args.gain_threshold,
            "max_notool": args.max_notool,
            "min_with_prefix": args.min_with_prefix,
        },
        "output": str(output_path),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
