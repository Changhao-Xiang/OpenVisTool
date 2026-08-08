from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

_SAVED_TO_RE = re.compile(r"[Ss]aved to:\s*(\S+)")
_SANDBOX_ROOT = Path("/mnt/data")


@dataclass(frozen=True)
class SessionEvent:
    role: str
    timestamp: str
    content: Any
    tool_calls: list[dict[str, Any]]
    tool_name: str
    tool_call_id: str


@dataclass(frozen=True)
class SessionTrace:
    name: str
    source_path: str
    session_key: str
    created_at: str
    updated_at: str
    media_dir: str
    roles: dict[str, int]
    tool_names: dict[str, int]
    total_events: int
    total_tool_calls: int
    events: list[SessionEvent]
    image_paths: dict[str, str]  # tool_call_id -> resolved absolute image path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for distill session JSONL files.",
        add_help=False,
    )
    parser.add_argument(
        "--sessions-dir",
        default="",
        help="Root directory containing session JSONL files.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Load the first N session_<id> directories by numeric id (ascending), matching "
        "distill layouts like session_<id>/sessions/*.jsonl or session_<id>/*.jsonl. "
        "Omit or use 0 to list all *.jsonl under the tree (slow for huge dirs).",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


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


def _infer_default_sessions_dir() -> Path:
    candidates = [
        Path.cwd() / "workspaces",
        Path.cwd() / "sessions",
    ]
    for root in candidates:
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("sessions*")):
            if candidate.is_dir():
                return candidate
    return Path.cwd()


def _looks_like_image_path(path: Path) -> bool:
    mime_type, _ = mimetypes.guess_type(str(path))
    return bool(mime_type and mime_type.startswith("image/"))


def _sandbox_relative_path(path: Path) -> Path | None:
    try:
        return path.relative_to(_SANDBOX_ROOT)
    except ValueError:
        return None


def _resolve_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].expanduser().resolve()


def _resolve_image_path(raw_path: str, session_path: Path, media_dir: Path | None = None) -> Path:
    image_path = Path(raw_path.strip())
    if not str(image_path):
        return image_path

    candidates: list[Path] = []
    if image_path.is_absolute():
        sandbox_relative = _sandbox_relative_path(image_path)
        if sandbox_relative is not None:
            # Files written inside the agent sandbox are materialized beside the session JSONL.
            candidates.append(session_path.parent / sandbox_relative)
            if media_dir is not None:
                candidates.append(media_dir / sandbox_relative.name)
        candidates.append(image_path)
    else:
        candidates.append(session_path.parent / image_path)
        if media_dir is not None:
            candidates.append(media_dir / image_path)
        candidates.append(image_path)

    return _resolve_existing_path(candidates)


def _resolve_tool_output_image(raw_path: str, session_path: Path, media_dir: Path | None = None) -> Path:
    """Resolve image path from 'Saved to:' in visual tool output."""
    return _resolve_image_path(raw_path, session_path, media_dir)


def _normalize_content(content: Any, session_path: Path, media_dir: Path | None = None) -> dict[str, Any]:
    if isinstance(content, str):
        return {"kind": "text", "text": content}

    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                blocks.append(
                    {
                        "type": "raw",
                        "text": json.dumps(item, ensure_ascii=False, indent=2),
                    }
                )
                continue

            item_type = item.get("type", "raw")
            if item_type == "text":
                blocks.append({"type": "text", "text": str(item.get("text", ""))})
            elif item_type == "image":
                raw_path = str(item.get("image", "")).strip()
                image_path = _resolve_image_path(raw_path, session_path, media_dir)
                blocks.append(
                    {
                        "type": "image",
                        "name": image_path.name,
                        "path": str(image_path),
                    }
                )
            else:
                blocks.append(
                    {
                        "type": item_type,
                        "text": json.dumps(item, ensure_ascii=False, indent=2),
                    }
                )
        return {"kind": "blocks", "blocks": blocks}

    return {"kind": "raw", "text": json.dumps(content, ensure_ascii=False, indent=2)}


def _jsonl_under_session_workspace(session_root: Path) -> Path | None:
    """Resolve distill per-item layout: session_<id>/sessions/<key>.jsonl."""
    nested = session_root / "sessions"
    if nested.is_dir():
        jsonl_files = sorted(p for p in nested.glob("*.jsonl") if p.is_file())
        if jsonl_files:
            return jsonl_files[0]
    for path in sorted(session_root.glob("*.jsonl")):
        if path.is_file():
            return path
    return None


def _session_numeric_id(path: Path) -> int | None:
    match = re.fullmatch(r"session_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _discover_jsonl_paths_by_numeric_id(base: Path, count: int) -> list[Path]:
    """Load the first N session_<id> directories by numeric id."""
    out: list[Path] = []
    session_roots = []
    for session_root in base.glob("session_*"):
        if not session_root.is_dir():
            continue
        session_id = _session_numeric_id(session_root)
        if session_id is None:
            continue
        session_roots.append((session_id, session_root))

    for _, session_root in sorted(session_roots, key=lambda item: item[0])[:count]:
        path = _jsonl_under_session_workspace(session_root)
        if path is not None:
            out.append(path)
    return out


@st.cache_data(show_spinner=False)
def discover_session_files(sessions_dir: str, max_samples: int | None) -> list[str]:
    base = Path(sessions_dir).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"Sessions directory does not exist: {base}")

    if max_samples is not None and max_samples > 0:
        files = _discover_jsonl_paths_by_numeric_id(base, max_samples)
    else:
        files = [path for path in base.glob("**/*.jsonl") if path.is_file()]
        files.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)

    return [str(path) for path in files]


@st.cache_data(show_spinner=False)
def load_session_trace(path_str: str) -> SessionTrace:
    path = Path(path_str)
    records = _load_jsonl_lines(path)
    if not records:
        raise ValueError(f"Empty session file: {path}")

    metadata = records[0]
    if metadata.get("_type") != "metadata":
        raise ValueError(f"First line in {path} must be metadata")

    metadata_block = metadata.get("metadata") or {}
    media_dir_str = str(metadata_block.get("media_dir", "")).strip()
    media_dir = Path(media_dir_str).resolve() if media_dir_str else None

    role_counter: Counter[str] = Counter()
    tool_counter: Counter[str] = Counter()
    events: list[SessionEvent] = []
    total_tool_calls = 0

    for record in records[1:]:
        role = str(record.get("role", "unknown"))
        role_counter[role] += 1

        tool_calls_raw = record.get("tool_calls") if isinstance(record.get("tool_calls"), list) else []
        tool_calls: list[dict[str, Any]] = []
        for call in tool_calls_raw:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            tool_name = str(function.get("name", "unknown"))
            tool_counter[tool_name] += 1
            tool_calls.append(
                {
                    "id": str(call.get("id", "")),
                    "type": str(call.get("type", "")),
                    "name": tool_name,
                    "arguments": function.get("arguments", ""),
                }
            )
        total_tool_calls += len(tool_calls)

        tool_name = str(record.get("name", ""))
        if role == "tool" and tool_name:
            tool_counter[tool_name] += 1

        events.append(
            SessionEvent(
                role=role,
                timestamp=str(record.get("timestamp", "")),
                content=_normalize_content(
                    record.get("content"),
                    path,
                    media_dir if role == "user" else None,
                ),
                tool_calls=tool_calls,
                tool_name=tool_name,
                tool_call_id=str(record.get("tool_call_id", "")),
            )
        )

    image_paths: dict[str, str] = {}
    for event in events:
        if event.role != "assistant":
            continue
        for tc in event.tool_calls:
            if tc.get("name") != "read_file":
                continue
            try:
                args = tc.get("arguments", "")
                parsed = json.loads(args) if isinstance(args, str) else (args or {})
                raw_path = str(parsed.get("path", "")).strip()
                if raw_path and _looks_like_image_path(Path(raw_path)):
                    resolved = _resolve_image_path(raw_path, path, media_dir)
                    image_paths[tc.get("id", "")] = str(resolved)
            except Exception:
                pass

    # Capture images from visual tool output (crop, etc.) via "Saved to:"
    for event in events:
        if event.role != "tool" or not event.tool_call_id:
            continue
        content = event.content
        if not isinstance(content, dict):
            continue
        kind = content.get("kind")
        if kind in ("text", "raw"):
            content_text = content.get("text", "")
        elif kind == "blocks":
            content_text = "\n".join(
                b.get("text", "") for b in content.get("blocks", []) if b.get("type") == "text"
            )
        else:
            continue
        for match in _SAVED_TO_RE.finditer(content_text):
            resolved = _resolve_tool_output_image(match.group(1), path, media_dir)
            image_paths[event.tool_call_id] = str(resolved)

    return SessionTrace(
        name=path.stem,
        source_path=str(path),
        session_key=str(metadata.get("key", path.stem)),
        created_at=str(metadata.get("created_at", "")),
        updated_at=str(metadata.get("updated_at", "")),
        media_dir=media_dir_str,
        roles=dict(role_counter),
        tool_names=dict(tool_counter),
        total_events=len(events),
        total_tool_calls=total_tool_calls,
        events=events,
        image_paths=image_paths,
    )


def _format_counter(counter_map: dict[str, int]) -> str:
    if not counter_map:
        return "-"
    return " | ".join(
        f"{key}: {value}" for key, value in sorted(counter_map.items(), key=lambda item: (-item[1], item[0]))
    )


def _render_content(content: dict[str, Any]) -> None:
    kind = content.get("kind")
    if kind == "text":
        st.markdown(content.get("text", ""))
        return

    if kind == "raw":
        st.code(content.get("text", ""), language="json")
        return

    if kind != "blocks":
        st.code(json.dumps(content, ensure_ascii=False, indent=2), language="json")
        return

    for block in content.get("blocks", []):
        block_type = block.get("type")
        if block_type == "text":
            st.markdown(block.get("text", ""))
            continue

        if block_type == "image":
            image_path = Path(str(block.get("path", "")))
            st.caption(str(image_path))
            if image_path.is_file():
                mime_type, _ = mimetypes.guess_type(str(image_path))
                if mime_type and mime_type.startswith("image/"):
                    st.image(str(image_path), caption=block.get("name", image_path.name))
                else:
                    st.warning(f"Unsupported image type: {image_path.name}")
            else:
                st.warning(f"Image not found: {image_path}")
            continue

        st.code(block.get("text", ""), language="json")


def _render_image_file(path_str: str) -> None:
    image_path = Path(path_str)
    if image_path.is_file():
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type and mime_type.startswith("image/"):
            st.image(str(image_path), caption=image_path.name)
        else:
            st.warning(f"Unsupported image type: {image_path.name}")
    else:
        st.warning(f"Image not found: {image_path}")


def _render_assistant_tool_calls(
    tool_calls: list[dict[str, Any]],
    image_paths: dict[str, str],
) -> None:
    if not tool_calls:
        return
    st.markdown("**Tool Calls**")
    for index, tool_call in enumerate(tool_calls, 1):
        label = f"{index}. {tool_call.get('name', 'unknown')}"
        with st.expander(label, expanded=False):
            st.caption(tool_call.get("id", ""))
            arguments = tool_call.get("arguments", "")
            if isinstance(arguments, str):
                try:
                    st.json(json.loads(arguments))
                except Exception:
                    st.code(arguments, language="json")
            else:
                st.json(arguments)
            call_id = tool_call.get("id", "")
            if tool_call.get("name") == "read_file" and call_id in image_paths:
                _render_image_file(image_paths[call_id])


def _render_event(
    event: SessionEvent,
    show_tool_calls: bool,
    image_paths: dict[str, str],
) -> None:
    role = event.role.lower()
    title = f"{role} | {event.timestamp or '-'}"

    if role == "user":
        with st.chat_message("user"):
            st.caption(title)
            _render_content(event.content)
        return

    if role == "assistant":
        with st.chat_message("assistant"):
            st.caption(title)
            _render_content(event.content)
            if show_tool_calls:
                _render_assistant_tool_calls(event.tool_calls, image_paths)
        return

    with st.container(border=True):
        st.markdown(f"**TOOL** `{event.tool_name or 'unknown'}`")
        if event.tool_call_id:
            st.caption(f"{title} | {event.tool_call_id}")
        else:
            st.caption(title)
        _render_content(event.content)
        if event.tool_call_id in image_paths:
            _render_image_file(image_paths[event.tool_call_id])


def main() -> None:
    args = parse_args()

    st.set_page_config(
        page_title="Distill Session Viewer",
        page_icon="🧭",
        layout="wide",
    )

    st.title("Distill Session Viewer")
    st.caption("浏览 sessions 目录中的 JSONL 文件，查看 user / assistant / tool 的完整交互过程。")

    default_dir = args.sessions_dir or str(_infer_default_sessions_dir())
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None
    with st.sidebar:
        st.header("会话选择")
        sessions_dir = st.text_input("Sessions 目录", value=default_dir)
        if max_samples is not None:
            st.caption(f"按数字 id 升序仅加载前 {max_samples} 个 session_<id> 目录（--max-samples）。")
        file_keyword = st.text_input("搜索文件名", value="")

        try:
            file_paths = discover_session_files(sessions_dir, max_samples)
        except Exception as exc:
            st.error(str(exc))
            return

        if file_keyword:
            lowered = file_keyword.lower()
            file_paths = [path for path in file_paths if lowered in Path(path).name.lower()]

        if not file_paths:
            st.warning("没有找到匹配的 JSONL 文件。")
            return

        display_options = {f"{Path(path).name}  |  {Path(path).parent.name}": path for path in file_paths}
        selected_label = st.selectbox(
            "选择 session 文件",
            options=list(display_options.keys()),
            index=0,
        )
        selected_path = display_options[selected_label]
        st.caption(f"共 {len(file_paths)} 个文件")

        event_keyword = st.text_input("搜索事件内容", value="")
        selected_roles = st.multiselect(
            "显示角色",
            options=["user", "assistant", "tool"],
            default=["user", "assistant", "tool"],
        )
        show_tool_calls = st.checkbox("展开 assistant 的 tool_calls", value=True)

    trace = load_session_trace(selected_path)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总事件数", trace.total_events)
    col2.metric("Tool Calls", trace.total_tool_calls)
    col3.metric("角色数", len(trace.roles))
    col4.metric("工具种类", len(trace.tool_names))

    st.subheader(trace.name)
    st.caption(trace.source_path)

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(f"**Session Key**: `{trace.session_key}`")
        st.markdown(f"**Created At**: `{trace.created_at or '-'}`")
        st.markdown(f"**Updated At**: `{trace.updated_at or '-'}`")
    with info_col2:
        st.markdown(f"**Media Dir**: `{trace.media_dir or '-'}`")
        st.markdown(f"**角色分布**: { _format_counter(trace.roles) }")
        st.markdown(f"**工具分布**: { _format_counter(trace.tool_names) }")

    st.divider()
    st.subheader("交互时间线")

    keyword = event_keyword.strip().lower()
    visible_events = []
    for event in trace.events:
        if event.role not in selected_roles:
            continue
        if keyword:
            haystack = "\n".join(
                [
                    event.role,
                    event.timestamp,
                    event.tool_name,
                    event.tool_call_id,
                    json.dumps(event.content, ensure_ascii=False),
                    json.dumps(event.tool_calls, ensure_ascii=False),
                ]
            ).lower()
            if keyword not in haystack:
                continue
        visible_events.append(event)

    st.caption(f"当前展示 {len(visible_events)} / {trace.total_events} 条事件")
    for event in visible_events:
        _render_event(event, show_tool_calls=show_tool_calls, image_paths=trace.image_paths)


if __name__ == "__main__":
    main()
