"""Intersect a correctness-filtered index with a tool-gain-selected index.

Each upstream filter keeps a sample for a different reason:

* correctness index (e.g. ``ChartVerse-SFT-filtered_index.jsonl``) — the teacher
  trajectory reached the right answer.
* tool-gain index (``*_tool_gain_selected.jsonl`` from ``select_tool_gain.py``) —
  giving the base model the teacher's tool prefix raised avg@k above threshold.

A sample is worth distilling only if it satisfies *both*. This script keeps the
ids present in both files and emits one merged record per kept id.

Example:
    python distill/select/select_correct_and_gain.py \\
        --correctness-index workspaces/qwen35plus/ChartVerse-SFT-filtered_index.jsonl \\
        --tool-gain-index   dataset/OpenVisTool_toolgain_only/Chart__..._tool_gain_selected.jsonl \\
        --output            dataset/OpenVisTool_toolgain_only/Chart__..._correct_and_gain.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"{path} line {line_no} must be a JSON object with an 'id' field")
            records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Keep ids present in both the correctness index and the tool-gain index."
    )
    p.add_argument("--correctness-index", required=True, help="Correctness-filtered index JSONL.")
    p.add_argument("--tool-gain-index", required=True, help="*_tool_gain_selected.jsonl from select_tool_gain.py.")
    p.add_argument("--output", required=True, help="Output JSONL of the intersection.")
    p.add_argument("--report", default="", help="Optional summary JSON report path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    correctness_path = Path(args.correctness_index).resolve()
    tool_gain_path = Path(args.tool_gain_index).resolve()
    output_path = Path(args.output).resolve()

    for label, path in [("correctness-index", correctness_path), ("tool-gain-index", tool_gain_path)]:
        if not path.is_file():
            raise FileNotFoundError(f"--{label} not found: {path}")

    # Correctness index keyed by id; later rows for a duplicate id overwrite.
    correctness_by_id: dict[Any, dict[str, Any]] = {}
    for record in _load_jsonl(correctness_path):
        correctness_by_id[record["id"]] = record

    tool_gain_records = _load_jsonl(tool_gain_path)

    kept = 0
    dropped_not_correct = 0
    seen_ids: set[Any] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in tool_gain_records:
            sample_id = record["id"]
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)

            correctness = correctness_by_id.get(sample_id)
            if correctness is None:
                dropped_not_correct += 1
                continue

            # Base on the tool-gain record (carries avg@k fields), add the
            # correctness method as provenance.
            merged = dict(record)
            method = correctness.get("method")
            if method is not None:
                merged["correctness_method"] = method
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")
            kept += 1

    summary = {
        "correctness_index_ids": len(correctness_by_id),
        "tool_gain_index_rows": len(tool_gain_records),
        "tool_gain_unique_ids": len(seen_ids),
        "kept": kept,
        "dropped_not_in_correctness": dropped_not_correct,
        "correctness_index": str(correctness_path),
        "tool_gain_index": str(tool_gain_path),
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
