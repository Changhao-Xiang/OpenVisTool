from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreparedItem:
    index: int
    item_id: Any
    query: str
    session_key: str
    media_paths: list[str]
    sample_suffix: str = ""


def load_jsonl_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            records.append(item)
    return records


def get_query_value(item: dict[str, Any], query_field: str) -> str:
    value = item.get(query_field)
    if value is None:
        raise ValueError(f"Missing query field '{query_field}'")
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        warnings.warn(f"Query field '{query_field}' is empty")
    return value


def get_session_key(item: dict[str, Any], dataset_name: str, sample_suffix: str = "") -> str:
    item_id = item.get("id")
    if item_id is None:
        raise ValueError("Missing required 'id' field for session key")
    return f"distill:{dataset_name}_{item_id}{sample_suffix}"


def resolve_media_paths(item: dict[str, Any], media_field: str, media_dir: Path) -> list[str]:
    value = item.get(media_field)
    if value is None:
        return []

    raw_paths = value if isinstance(value, list) else [value]
    resolved: list[str] = []
    for raw in raw_paths:
        if not isinstance(raw, str):
            raise ValueError(f"Field '{media_field}' must be a string or list of strings")
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (media_dir / path).resolve()
        if not path.is_file():
            raise ValueError(f"Media file not found: {path}")
        resolved.append(str(path))
    return resolved


def prepare_items(
    items: list[dict[str, Any]],
    *,
    dataset_name: str,
    query_field: str,
    media_field: str,
    media_dir: Path,
    sample_suffix: str = "",
) -> list[PreparedItem]:
    prepared: list[PreparedItem] = []
    seen_session_keys: dict[str, Any] = {}

    for index, item in enumerate(items, 1):
        query = get_query_value(item, query_field)
        session_key = get_session_key(item, dataset_name, sample_suffix)
        item_id = item.get("id")
        prev_item_id = seen_session_keys.get(session_key)
        if prev_item_id is not None:
            raise ValueError(
                "Duplicate session key detected for id fields "
                f"{prev_item_id!r} and {item_id!r}: {session_key}"
            )
        seen_session_keys[session_key] = item_id
        prepared.append(
            PreparedItem(
                index=index,
                item_id=item_id,
                query=query,
                session_key=session_key,
                media_paths=resolve_media_paths(item, media_field, media_dir),
                sample_suffix=sample_suffix,
            )
        )

    return prepared
