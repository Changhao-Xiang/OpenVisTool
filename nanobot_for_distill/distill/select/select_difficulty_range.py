"""Select samples whose avg@k falls within [min_avg, max_avg].

Given a raw dataset JSONL and a difficulty progress JSONL produced by
filter_difficulty.py, emit samples whose avg@k is in [min_avg, max_avg].
Output format matches the difficulty-filtered JSONL (original fields +
avg_k_notool).

Example:
    python distill/filter/select_difficulty_range.py \
        --dataset dataset/Table/CoSyn-400K/table.jsonl \
        --progress dataset/Table/CoSyn-400K/table_difficulty_filtered_qwen35_9b.progress.jsonl \
        --min-avg 0.25 \
        --max-avg 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {lineno} of {path} must be a JSON object")
            items.append(item)
    return items


def _iter_jsonl_allow_partial_tail(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for lineno, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            if lineno == len(lines):
                break
            raise ValueError(f"Invalid JSON on line {lineno} of {path}")
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object on line {lineno} of {path}")
        records.append(record)
    return records


def build_progress_index(progress_path: Path) -> dict[str, float]:
    """Return mapping from key (e.g. 'id:62' or 'index:0') -> avg_k."""
    records = _iter_jsonl_allow_partial_tail(progress_path)
    index: dict[str, float] = {}
    for rec in records:
        key = str(rec.get("key", "")).strip()
        avg_k = rec.get("avg_k")
        if key and avg_k is not None:
            index[key] = float(avg_k)
    return index


def get_item_key(item: dict[str, Any], index: int) -> str:
    item_id = item.get("id")
    if item_id is not None:
        return f"id:{item_id}"
    return f"index:{index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select samples whose avg@k is in [min_avg, max_avg]."
    )
    parser.add_argument(
        "--dataset", "--input", dest="dataset", required=True,
        help="Raw dataset JSONL.",
    )
    parser.add_argument(
        "--progress", required=True,
        help="Difficulty progress JSONL produced by filter_difficulty.py.",
    )
    parser.add_argument(
        "--min-avg", type=float, default=0.0,
        help="Lower bound of avg@k range (inclusive). Default: 0.0.",
    )
    parser.add_argument(
        "--max-avg", type=float, default=1.0,
        help="Upper bound of avg@k range (inclusive). Default: 1.0.",
    )
    parser.add_argument(
        "--output", default="",
        help=(
            "Output JSONL path. Defaults to "
            "<dataset_stem>_difficulty_range_<min_avg>_<max_avg>.jsonl."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_avg > args.max_avg:
        print(
            f"[ERROR] --min-avg ({args.min_avg}) must be <= --max-avg ({args.max_avg})",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_path = Path(args.dataset).resolve()
    progress_path = Path(args.progress).resolve()

    if not dataset_path.is_file():
        print(f"[ERROR] Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    if not progress_path.is_file():
        print(f"[ERROR] Progress file not found: {progress_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        min_tag = f"{args.min_avg:.2f}".rstrip("0").rstrip(".")
        max_tag = f"{args.max_avg:.2f}".rstrip("0").rstrip(".")
        output_path = dataset_path.with_name(
            f"{dataset_path.stem}_difficulty_range_{min_tag}_{max_tag}.jsonl"
        )

    print(f"Dataset:   {dataset_path}")
    print(f"Progress:  {progress_path}")
    print(f"Range:     [{args.min_avg}, {args.max_avg}]")
    print(f"Output:    {output_path}")

    progress_index = build_progress_index(progress_path)
    print(f"Progress entries loaded: {len(progress_index)}")

    items = load_jsonl(dataset_path)
    print(f"Dataset items: {len(items)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped_no_progress = 0
    with output_path.open("w", encoding="utf-8") as out:
        for idx, item in enumerate(items):
            key = get_item_key(item, idx)
            avg_k = progress_index.get(key)
            if avg_k is None:
                skipped_no_progress += 1
                continue
            if args.min_avg <= avg_k <= args.max_avg:
                payload = dict(item)
                payload["avg_k_notool"] = avg_k
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                kept += 1

    print(f"No progress entry (skipped): {skipped_no_progress}")
    print(f"Kept: {kept}")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
