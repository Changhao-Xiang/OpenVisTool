"""Intersect two JSONL index files by a configurable identifier field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_ids(path: Path, id_field: str) -> set[object]:
    ids: set[object] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if id_field not in record:
                raise KeyError(f"{path}:{line_number} has no {id_field!r}")
            ids.add(record[id_field])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_a", type=Path, help="JSONL records to retain and write")
    parser.add_argument("input_b", type=Path, help="JSONL whose IDs define the selection")
    parser.add_argument("output", type=Path)
    parser.add_argument("--id-field", default="id")
    args = parser.parse_args()

    selected_ids = load_ids(args.input_b, args.id_field)
    input_count = 0
    output_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input_a.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            input_count += 1
            record = json.loads(line)
            if args.id_field not in record:
                raise KeyError(f"{args.input_a}:{line_number} has no {args.id_field!r}")
            if record[args.id_field] in selected_ids:
                destination.write(line if line.endswith("\n") else line + "\n")
                output_count += 1

    print(f"input_a={input_count}")
    print(f"ids_b={len(selected_ids)}")
    print(f"output={output_count}")
    print(args.output)


if __name__ == "__main__":
    main()
