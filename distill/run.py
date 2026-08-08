from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.config import build_agent_from_config, load_runtime_config
from distill.dataset import PreparedItem, load_jsonl_records, prepare_items
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.config.loader import get_config_path
from nanobot.config.paths import get_workspace_path
from nanobot.utils.helpers import safe_filename

DEFAULT_API_RETRIES = 5
DEFAULT_API_RETRY_DELAY_SECONDS = 30
RETRYABLE_API_ERROR_PATTERNS = (
    "error calling llm:",
    "error code: 400",
    "error code: 429",
    "error code: 500",
    "rate limit",
    "too many requests",
    "timeout",
    "timed out",
    "service unavailable",
    "temporarily unavailable",
    "connection error",
    "server error",
    "overloaded",
    "稍后再试",
    "业务错误",
    "超时",
)


def _is_retryable_api_error(message: str | None) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return any(pattern in lowered for pattern in RETRYABLE_API_ERROR_PATTERNS)


def _sub_workspace_for_item(resolved_workspace: Path, item: PreparedItem) -> Path:
    """Derive a per-item directory under the main workspace."""
    name = safe_filename(f"{item.item_id}{item.sample_suffix}")
    return resolved_workspace / f"session_{name}"


def _resolve_custom_instructions_override(value: str | None) -> str | None:
    """Resolve the override as a required custom instructions file path."""
    if value is None:
        return None

    raw_value = value.strip()
    if not raw_value:
        return None

    normalized = raw_value[1:] if raw_value.startswith("@") else raw_value
    raw_path = Path(normalized).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                raw_path.resolve(),
                (PROJECT_ROOT / raw_path).resolve(),
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    searched_paths = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find custom instructions file from " f"{raw_value!r}. Searched: {searched_paths}"
    )


async def run_one(
    agent: AgentLoop,
    query: str,
    session_key: str,
    media_paths: list[str],
    media_dir: Path,
    sub_workspace: Path | None = None,
) -> str:
    if sub_workspace is not None:
        agent.set_tool_workspace(sub_workspace, media_dir=media_dir)

    msg = InboundMessage(
        channel="cli",
        sender_id="user",
        chat_id="direct",
        content=query,
        media=media_paths,
        metadata={"media_dir": str(media_dir)},
    )
    last_exception: Exception | None = None

    for attempt in range(DEFAULT_API_RETRIES + 1):
        try:
            response = await agent._process_message(msg, session_key=session_key)
        except Exception as exc:
            last_exception = exc
            if attempt >= DEFAULT_API_RETRIES or not _is_retryable_api_error(str(exc)):
                raise
            print(
                f"session_key={session_key} API call failed with retryable error: {exc}. "
                f"Retrying in {DEFAULT_API_RETRY_DELAY_SECONDS}s "
                f"({attempt + 1}/{DEFAULT_API_RETRIES})",
                flush=True,
            )
            await asyncio.sleep(DEFAULT_API_RETRY_DELAY_SECONDS)
            continue

        final_response = response.content if response else ""
        if attempt >= DEFAULT_API_RETRIES or not _is_retryable_api_error(final_response):
            # Release session from in-memory cache to prevent unbounded memory growth.
            agent.sessions.invalidate(session_key)
            return final_response

        print(
            f"session_key={session_key} API returned retryable error response. "
            f"Retrying in {DEFAULT_API_RETRY_DELAY_SECONDS}s "
            f"({attempt + 1}/{DEFAULT_API_RETRIES})",
            flush=True,
        )
        await asyncio.sleep(DEFAULT_API_RETRY_DELAY_SECONDS)

    agent.sessions.invalidate(session_key)
    if last_exception is not None:
        raise last_exception
    return ""


def _print_progress(
    item: PreparedItem, total: int, final_response: str, worker_id: int | None = None
) -> None:
    worker_prefix = f"worker={worker_id} " if worker_id is not None else ""
    tqdm.write(
        f"[{item.index}/{total}] {worker_prefix}id={item.item_id} session_key={item.session_key} "
        f"images={len(item.media_paths)} final_len={len(final_response)}"
    )


async def _run_batch_sequential(
    prepared_items: list[PreparedItem],
    *,
    config: Any,
    media_dir: Path,
) -> Path:
    agent, resolved_workspace = build_agent_from_config(config)
    total = len(prepared_items)

    with tqdm(total=total, desc="Distilling", unit="item") as pbar:
        for item in prepared_items:
            sub_ws = _sub_workspace_for_item(resolved_workspace, item)
            final_response = await run_one(
                agent=agent,
                query=item.query,
                session_key=item.session_key,
                media_paths=item.media_paths,
                media_dir=media_dir,
                sub_workspace=sub_ws,
            )
            _print_progress(item, total, final_response)
            pbar.update(1)

    return resolved_workspace


async def _run_worker_loop(
    worker_id: int,
    prepared_queue: Queue[PreparedItem | None],
    *,
    config: Any,
    media_dir: Path,
    resolved_workspace: Path,
    total: int,
    print_lock: threading.Lock,
    errors: list[str],
    error_lock: threading.Lock,
    pbar: tqdm,
) -> None:
    agent, _ = build_agent_from_config(config)

    while True:
        item = prepared_queue.get()
        if item is None:
            return

        try:
            sub_ws = _sub_workspace_for_item(resolved_workspace, item)
            final_response = await run_one(
                agent=agent,
                query=item.query,
                session_key=item.session_key,
                media_paths=item.media_paths,
                media_dir=media_dir,
                sub_workspace=sub_ws,
            )
        except Exception as e:
            error_message = (
                f"[{item.index}/{total}] worker={worker_id} "
                f"id={item.item_id} session_key={item.session_key} failed: {e}"
            )
            with error_lock:
                errors.append(error_message)
            tqdm.write(error_message)
            pbar.update(1)
        else:
            with print_lock:
                _print_progress(item, total, final_response, worker_id=worker_id)
            pbar.update(1)


def _run_worker_thread(
    worker_id: int,
    prepared_queue: Queue[PreparedItem | None],
    *,
    config: Any,
    media_dir: Path,
    resolved_workspace: Path,
    total: int,
    print_lock: threading.Lock,
    errors: list[str],
    error_lock: threading.Lock,
    pbar: tqdm,
) -> None:
    try:
        asyncio.run(
            _run_worker_loop(
                worker_id,
                prepared_queue,
                config=config,
                media_dir=media_dir,
                resolved_workspace=resolved_workspace,
                total=total,
                print_lock=print_lock,
                errors=errors,
                error_lock=error_lock,
                pbar=pbar,
            )
        )
    except Exception as e:
        with error_lock:
            errors.append(f"worker={worker_id} thread/async loop crashed: {e}")


def _format_parallel_errors(errors: list[str], *, max_lines: int = 30) -> str:
    n = len(errors)
    head = errors[:max_lines]
    lines = "\n".join(head)
    if n > max_lines:
        lines += f"\n... and {n - max_lines} more"
    return f"{n} item(s) failed:\n{lines}"


def _run_batch_parallel(
    prepared_items: list[PreparedItem],
    *,
    config: Any,
    media_dir: Path,
    num_workers: int,
    continue_on_error: bool,
) -> tuple[Path, list[str]]:
    resolved_workspace = get_workspace_path(str(config.workspace_path))
    worker_count = min(num_workers, len(prepared_items))
    if worker_count == 0:
        return resolved_workspace, []

    prepared_queue: Queue[PreparedItem | None] = Queue()
    for item in prepared_items:
        prepared_queue.put(item)
    for _ in range(worker_count):
        prepared_queue.put(None)

    print_lock = threading.Lock()
    error_lock = threading.Lock()
    errors: list[str] = []

    with tqdm(total=len(prepared_items), desc="Distilling", unit="item") as pbar:
        threads = [
            threading.Thread(
                target=_run_worker_thread,
                kwargs={
                    "worker_id": worker_id,
                    "prepared_queue": prepared_queue,
                    "config": config,
                    "media_dir": media_dir,
                    "resolved_workspace": resolved_workspace,
                    "total": len(prepared_items),
                    "print_lock": print_lock,
                    "errors": errors,
                    "error_lock": error_lock,
                    "pbar": pbar,
                },
                name=f"distill-worker-{worker_id}",
            )
            for worker_id in range(1, worker_count + 1)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    if errors:
        summary = _format_parallel_errors(errors)
        if continue_on_error:
            print(summary, file=sys.stderr, flush=True)
        else:
            raise RuntimeError(summary)

    return resolved_workspace, errors


def run_batch(
    input_path: Path,
    dataset_name: str,
    query_field: str,
    media_field: str,
    media_dir: Path,
    config_path: Path | None,
    num_workers: int,
    *,
    continue_on_error: bool = False,
    sample_index: int = -1,
    workspace_override: str | None = None,
    custom_instructions_override: str | None = None,
    sample_range: str | None = None,
) -> int:
    if num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if not dataset_name:
        dataset_name = input_path.parent.name

    sample_suffix = f"_s{sample_index}" if sample_index >= 0 else ""

    items = load_jsonl_records(input_path)
    prepared_items = prepare_items(
        items,
        dataset_name=dataset_name,
        query_field=query_field,
        media_field=media_field,
        media_dir=media_dir,
        sample_suffix=sample_suffix,
    )

    if sample_range:
        parts = sample_range.split(":")
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if len(parts) > 1 and parts[1] else None
        prepared_items = prepared_items[start:end]

    config = load_runtime_config(config_path)

    if workspace_override:
        config.agents.defaults.workspace = workspace_override
    if custom_instructions_override:
        config.agents.defaults.custom_instructions = _resolve_custom_instructions_override(
            custom_instructions_override
        )

    if num_workers == 1:
        resolved_workspace = asyncio.run(
            _run_batch_sequential(
                prepared_items,
                config=config,
                media_dir=media_dir,
            )
        )
        parallel_errors: list[str] = []
    else:
        resolved_workspace, parallel_errors = _run_batch_parallel(
            prepared_items,
            config=config,
            media_dir=media_dir,
            num_workers=num_workers,
            continue_on_error=continue_on_error,
        )

    print(f"Config: {get_config_path()}", flush=True)
    print(f"Workspace: {resolved_workspace}", flush=True)
    print(f"Per-item sessions saved under: {resolved_workspace}/session_*", flush=True)

    if num_workers > 1 and parallel_errors:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nanobot on a JSONL dataset and persist traces in sessions.",
    )
    parser.add_argument("input", help="Input JSONL path")
    parser.add_argument("--dataset-name", default="", help="Dataset name")
    parser.add_argument(
        "--query-field",
        default="query",
        help="Field name to use as the nanobot query (default: query; falls back to problem)",
    )
    parser.add_argument(
        "--media-field",
        default="images",
        help="Field name to use for multimodal media inputs (default: images)",
    )
    parser.add_argument(
        "--media-dir",
        default="",
        help="Base directory for resolving relative media paths (default: input file directory)",
    )
    parser.add_argument(
        "--config",
        help="Path to config.json (default: ~/.nanobot/config.json)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Parallel mode only: if some items fail, still finish the rest, then print a summary "
            "to stderr and exit with code 1 instead of raising RuntimeError (no traceback)."
        ),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=-1,
        help="Sample index for k-sampling. Appends '_s{N}' to session keys and workspace dirs.",
    )
    parser.add_argument(
        "--workspace-override",
        default=None,
        help="Override workspace path from config.",
    )
    parser.add_argument(
        "--custom-instructions-override",
        default=None,
        help="Override customInstructions with a file path.",
    )
    parser.add_argument(
        "--sample-range",
        default=None,
        help="Slice range for dataset items, e.g. '180000:185000', ':1000', '5000:'. Uses Python slice semantics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    input_path = Path(args.input).expanduser().resolve()
    media_dir = Path(args.media_dir).expanduser().resolve() if args.media_dir else input_path.parent
    try:
        exit_code = run_batch(
            input_path=input_path,
            dataset_name=args.dataset_name,
            query_field=args.query_field,
            media_field=args.media_field,
            media_dir=media_dir,
            config_path=config_path,
            num_workers=args.num_workers,
            continue_on_error=args.continue_on_error,
            sample_index=args.sample_index,
            workspace_override=args.workspace_override,
            custom_instructions_override=args.custom_instructions_override,
            sample_range=args.sample_range,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr, flush=True)
        raise SystemExit(1) from e
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
