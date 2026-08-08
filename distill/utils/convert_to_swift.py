"""Convert filtered distill sessions to ms-swift agent-format JSONL.

This script:
1. Reads an index JSONL that lists selected session ids.
2. Resolves the corresponding session trace JSONL files.
3. Converts each trace into ms-swift agent-format with top-level:
   - tools: JSON string
   - messages: role/content list
   - images: optional multimodal file paths
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.utils.tool_loader import load_tools_and_pools  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
SAVED_TO_RE = re.compile(r"saved to:\s*(\S+)", re.IGNORECASE)
IMAGE_VIEWED_RE = re.compile(r"^\[image viewed \* \d+\]", re.IGNORECASE)
TASK_SPECIFIC_INSTRUCTIONS_DIR = (
    PROJECT_ROOT / "distill" / "run" / "task_specific_toolcall_instructions"
)

_TOOLS_BY_NAME: dict[str, dict[str, Any]] = {}
_BASE_TOOL_NAMES: list[str] = []
_TASK_SPECIFIC_INSTRUCTIONS: list[str] = sorted(
    (
        path.read_text(encoding="utf-8").strip()
        for path in TASK_SPECIFIC_INSTRUCTIONS_DIR.glob("*.md")
        if path.is_file()
    ),
    key=len,
    reverse=True,
)


def resolve_tool_names(
    tools_by_name: dict[str, Any],
    pools: dict[str, Any],
    tool_pool: str,
    tools_override: str | None,
) -> list[str]:
    """Choose tool names from --tools override, else from named pool (or none/all)."""

    def _warn_unknown(unknown: list[str]) -> None:
        if unknown:
            print(
                f"[WARN] Unknown tool names ignored: {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(tools_by_name))}",
                file=sys.stderr,
            )

    if tools_override is not None:
        raw = tools_override.strip()
        if raw.lower() == "none":
            return []
        if raw.lower() == "all":
            return sorted(tools_by_name.keys())
        names = [t.strip() for t in raw.split(",") if t.strip()]
        unknown = [n for n in names if n not in tools_by_name]
        _warn_unknown(unknown)
        return [n for n in names if n in tools_by_name]

    key = (tool_pool or "").strip()
    if not key or key.lower() == "none":
        return []
    if key.lower() == "all":
        return sorted(tools_by_name.keys())

    pool_list = pools.get(key)
    if pool_list is None:
        available = ", ".join(sorted(pools.keys())) if pools else "(no pools defined)"
        print(
            f"[WARN] Unknown tool pool {key!r}; known pools: {available}. No tools in output.",
            file=sys.stderr,
        )
        return []
    if not isinstance(pool_list, list):
        print(f"[WARN] Pool {key!r} must be a JSON array of tool names.", file=sys.stderr)
        return []

    unknown = [n for n in pool_list if n not in tools_by_name]
    _warn_unknown(unknown)
    return [n for n in pool_list if n in tools_by_name]


def _extract_system_prompt_text(raw: Any) -> str:
    """Unwrap the session metadata.system_prompt field to plain text.

    The field can be either a plain string or an OpenAI-style list of content
    blocks (e.g. ``[{"type": "text", "text": "...", "cache_control": {...}}]``)
    when prompt caching is enabled at rollout time. In the latter case we
    concatenate every ``type == "text"`` block's ``text``.
    """
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(raw, str):
        return raw
    return ""


_TASK_INSTRUCTIONS_SEPARATOR = "\n\n---\n\n"


def clean_system_prompt(prompt: str) -> str:
    """Strip task-specific instructions appended to the system prompt.

    The context builder joins the base prompt and the task-specific
    instructions with ``\\n\\n---\\n\\n`` (see
    ``nanobot.agent.context.ContextBuilder._join_prompt_parts``). Splitting on
    that separator and keeping the first segment reliably drops the trailing
    instructions block even when its contents have since been edited on disk.

    Falls back to the previous exact-suffix match against
    ``_TASK_SPECIFIC_INSTRUCTIONS`` as a safety net.
    """
    stripped_prompt = prompt.rstrip()
    if _TASK_INSTRUCTIONS_SEPARATOR in stripped_prompt:
        return stripped_prompt.split(_TASK_INSTRUCTIONS_SEPARATOR, 1)[0].rstrip()
    for instructions in _TASK_SPECIFIC_INSTRUCTIONS:
        if stripped_prompt.endswith(instructions):
            return stripped_prompt[: -len(instructions)].rstrip()
    return stripped_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert filtered sessions to ms-swift agent format.")
    parser.add_argument("--sessions-dir", required=True, help="Root directory containing session_* dirs.")
    parser.add_argument(
        "--index-file", required=True, help="Filtered index JSONL with at least an 'id' field."
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--tool-pool",
        default="default",
        help="Named tool pool from tool_loader. Ignored when --tools is provided.",
    )
    parser.add_argument(
        "--tools",
        default=None,
        help="Override tool pool with comma-separated tool names, or 'all' / 'none'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, cpu_count()),
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N ids from the index file.",
    )
    return parser.parse_args()


def _load_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} line {line_no} must be a JSON object")
            records.append(record)
    return records


def _jsonl_under_session_workspace(session_root: Path) -> Path | None:
    nested = session_root / "sessions"
    if nested.is_dir():
        jsonl_files = sorted(p for p in nested.glob("*.jsonl") if p.is_file())
        if jsonl_files:
            return jsonl_files[0]
    jsonl_files = sorted(p for p in session_root.glob("*.jsonl") if p.is_file())
    return jsonl_files[0] if jsonl_files else None


def _build_session_index(sessions_dir: Path) -> dict[int, list[Path]]:
    """Scan sessions_dir once and build a mapping from session id to jsonl paths.

    This avoids O(N) glob calls when resolving a large index file.  Each id maps
    to a list of candidate jsonl paths (one per k-sample variant), sorted so that
    the plain ``session_<id>`` entry comes first.
    """
    index: dict[int, list[Path]] = {}
    for entry in sessions_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("session_"):
            continue
        # Extract numeric id from "session_<id>" or "session_<id>_s<N>"
        rest = name[len("session_") :]
        numeric = rest.split("_")[0]
        if not numeric.isdigit():
            continue
        sid = int(numeric)
        jsonl_path = _jsonl_under_session_workspace(entry)
        if jsonl_path is not None:
            index.setdefault(sid, []).append(jsonl_path)
    # Sort each list so plain session_<id> (no _s suffix) comes first
    for sid in index:
        index[sid].sort(key=lambda p: (len(p.parts), str(p)))
    return index


def _load_index_ids(index_path: Path, limit: int | None) -> list[int]:
    ids: list[int] = []
    with index_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"{index_path} line {line_no} must contain an 'id' field")
            ids.append(int(record["id"]))
            if limit is not None and len(ids) >= limit:
                break
    return ids


def _session_root(session_path: Path) -> Path:
    if session_path.parent.name == "sessions" and session_path.parent.parent.name.startswith("session_"):
        return session_path.parent.parent
    return session_path.parent


def _is_image_path(path: Path | None) -> bool:
    return path is not None and path.suffix.lower() in IMG_EXTS


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


def _tools_json_for_events(events: list[dict[str, Any]]) -> str:
    names: list[str] = []
    seen: set[str] = set()

    for name in _BASE_TOOL_NAMES:
        if name in _TOOLS_BY_NAME and name not in seen:
            names.append(name)
            seen.add(name)

    for record in events:
        if record.get("role") != "assistant":
            continue
        for tc in record.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(function.get("name", "") or "").strip()
            if name and name in _TOOLS_BY_NAME and name not in seen:
                names.append(name)
                seen.add(name)

    return json.dumps([_TOOLS_BY_NAME[name] for name in names], ensure_ascii=False)


def _slice_events_from_last_user(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the final user turn and the events that follow it.

    Distilled traces are expected to be single-turn user/assistant interactions.
    If a trace contains multiple user messages, the earlier ones are usually
    retries from upstream API calls and should not be kept in the final sample.
    """
    last_user_idx: int | None = None
    for idx, record in enumerate(events):
        if record.get("role") == "user":
            last_user_idx = idx
    if last_user_idx is None:
        return events
    return events[last_user_idx:]


def _append_image(
    images: list[str],
    resolved_path: Path | None,
) -> bool:
    if resolved_path is None or not resolved_path.is_file():
        return False
    images.append(str(resolved_path))
    return True


def _normalize_user_content(
    content: Any,
    *,
    session_path: Path,
    media_dir: Path | None,
    images: list[str],
) -> str:
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(json.dumps(block, ensure_ascii=False))
            continue

        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
            continue
        if block_type == "image":
            raw_path = str(block.get("image", ""))
            resolved = _resolve_existing_path(raw_path, session_path, media_dir)
            if _append_image(images, resolved):
                parts.append("<image>")
            continue
        parts.append(json.dumps(block, ensure_ascii=False))

    return "\n".join(part for part in parts if part)


def _build_assistant_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    reasoning = str(record.get("reasoning_content", "") or "").strip()
    content = str(record.get("content", "") or "").strip()

    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")
    if content:
        parts.append(content)

    return "\n\n".join(parts).strip()


def _collect_read_file_images(
    events: list[dict[str, Any]],
    *,
    session_path: Path,
    media_dir: Path | None,
) -> dict[str, Path]:
    tool_images: dict[str, Path] = {}
    for record in events:
        if record.get("role") != "assistant":
            continue
        for tc in record.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tool_call_id = str(tc.get("id", ""))
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
            resolved = _resolve_existing_path(raw_path, session_path, media_dir)
            if _is_image_path(resolved):
                tool_images[tool_call_id] = resolved
    return tool_images


def _collect_tool_output_images(
    record: dict[str, Any],
    *,
    session_path: Path,
    media_dir: Path | None,
    read_file_images: dict[str, Path],
    images: list[str],
) -> list[str]:
    text = str(record.get("content", "") or "")
    tokens: list[str] = []

    tool_call_id = str(record.get("tool_call_id", ""))
    if IMAGE_VIEWED_RE.search(text):
        resolved = read_file_images.get(tool_call_id)
        if _append_image(images, resolved):
            tokens.append("<image>")

    for match in SAVED_TO_RE.finditer(text):
        resolved = _resolve_existing_path(match.group(1), session_path, media_dir)
        if not _is_image_path(resolved):
            continue
        if _append_image(images, resolved):
            tokens.append("<image>")

    return tokens


def _build_tool_response_content(
    record: dict[str, Any],
    *,
    session_path: Path,
    media_dir: Path | None,
    read_file_images: dict[str, Path],
    images: list[str],
) -> str:
    """Return the raw tool output string (with optional ``<image>`` tokens).

    The teacher model receives the unwrapped string at distillation time
    (see ``nanobot.agent.context.add_tool_result``), so we keep the training
    payload identical to avoid train/inference skew.
    """
    raw_text = str(record.get("content", "") or "").strip()
    image_tokens = _collect_tool_output_images(
        record,
        session_path=session_path,
        media_dir=media_dir,
        read_file_images=read_file_images,
        images=images,
    )

    if not image_tokens:
        return raw_text
    if IMAGE_VIEWED_RE.search(raw_text):
        return "\n".join(image_tokens + ([raw_text] if raw_text else []))
    return "\n".join(([raw_text] if raw_text else []) + image_tokens)


def convert_session(jsonl_path: str) -> dict[str, Any]:
    path = Path(jsonl_path)
    try:
        records = _load_jsonl_lines(path)
        metadata = records[0] if records else {}
        has_metadata = isinstance(metadata, dict) and metadata.get("_type") == "metadata"
        meta_block = (
            metadata.get("metadata") if has_metadata and isinstance(metadata.get("metadata"), dict) else {}
        )
        system_prompt = _extract_system_prompt_text(meta_block.get("system_prompt"))
        media_dir_str = str(meta_block.get("media_dir", "") or "").strip()
        media_dir = Path(media_dir_str).resolve() if media_dir_str else None

        if system_prompt:
            system_prompt = clean_system_prompt(system_prompt)

        events = records[1:] if has_metadata else records
        events = _slice_events_from_last_user(events)
        tools_json = _tools_json_for_events(events)

        read_file_images = _collect_read_file_images(events, session_path=path, media_dir=media_dir)
        images: list[str] = []
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for record in events:
            role = record.get("role")
            if role == "user":
                user_text = _normalize_user_content(
                    record.get("content"),
                    session_path=path,
                    media_dir=media_dir,
                    images=images,
                )
                messages.append({"role": "user", "content": user_text})
                continue

            if role == "assistant":
                assistant_text = _build_assistant_text(record)
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})

                for tc in record.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    name = str(function.get("name", "") or "").strip()
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = arguments
                    payload = {"name": name, "arguments": arguments}
                    dumped = json.dumps(payload, ensure_ascii=False)
                    messages.append({"role": "tool_call", "content": dumped})
                continue

            if role == "tool":
                tool_content = _build_tool_response_content(
                    record,
                    session_path=path,
                    media_dir=media_dir,
                    read_file_images=read_file_images,
                    images=images,
                )
                messages.append({"role": "tool_response", "content": tool_content})

        sample: dict[str, Any] = {
            "path": str(path),
            "sample": {
                "tools": tools_json,
                "messages": messages,
            },
        }
        if images:
            sample["sample"]["images"] = images
        return sample
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": str(exc)}


def main() -> None:
    global _TOOLS_BY_NAME, _BASE_TOOL_NAMES

    args = parse_args()
    sessions_dir = Path(args.sessions_dir).resolve()
    index_path = Path(args.index_file).resolve()
    output_path = Path(args.output).resolve()

    registry = load_tools_and_pools()
    tools_by_name: dict[str, Any] = registry["tools"]
    pools: dict[str, Any] = registry["pools"]
    tool_names = resolve_tool_names(tools_by_name, pools, args.tool_pool, args.tools)
    tool_defs = [tools_by_name[name] for name in tool_names if name in tools_by_name]

    _TOOLS_BY_NAME = tools_by_name
    _BASE_TOOL_NAMES = [tool["function"]["name"] for tool in tool_defs]

    index_ids = _load_index_ids(index_path, args.limit)
    print("Scanning sessions directory …")
    session_index = _build_session_index(sessions_dir)
    session_jsonls: list[str] = []
    missing_ids = 0
    for sid in index_ids:
        paths = session_index.get(sid)
        if not paths:
            missing_ids += 1
            continue
        session_jsonls.append(str(paths[0]))

    print(f"Index rows loaded: {len(index_ids)}")
    print(f"Session traces found: {len(session_jsonls)}")
    print(f"Missing session ids: {missing_ids}")
    print(f"Tools included: {tool_names or '(none)'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    failed = 0

    with Pool(args.workers) as pool, output_path.open("w", encoding="utf-8") as out:
        for result in tqdm(
            pool.imap_unordered(convert_session, session_jsonls, chunksize=128), total=len(session_jsonls)
        ):
            if "sample" in result:
                out.write(json.dumps(result["sample"], ensure_ascii=False) + "\n")
                converted += 1
                continue

            failed += 1
            print(f"[WARN] Failed to convert {result['path']}: {result.get('error', 'unknown error')}")

    print(f"\n{'=' * 60}")
    print(f"Index rows loaded:     {len(index_ids)}")
    print(f"Missing session ids:   {missing_ids}")
    print(f"Converted samples:     {converted}")
    print(f"Failed samples:        {failed}")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
