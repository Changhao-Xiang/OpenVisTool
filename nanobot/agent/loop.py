"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.path_mapper import PathMapper
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.vision import (
    AdjustBrightnessTool,
    ColorClustersTool,
    ColorSegmentsTool,
    ComputerUseTool,
    ConnectedComponentsTool,
    CropTool,
    DrawBBoxTool,
    DrawCircleTool,
    DrawLineTool,
    FindContoursTool,
    FlipTool,
    HoughCirclesTool,
    HoughLinesTool,
    InRangeColorTool,
    RotateTool,
    SampleColorTool,
    TemplateMatchTool,
)
from nanobot.agent.tools.vision.enhance import DetectEdgesTool, EnhanceContrastTool, GrayscaleTool
from nanobot.agent.tools.vision.render_html import RenderHtmlTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ExecToolConfig


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 10000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        enabled_tools: dict[str, bool] | None = None,
        enabled_skills: dict[str, bool] | None = None,
        custom_instructions: str | None = None,
        path_mapper: PathMapper | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig

        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.enabled_tools = enabled_tools or {}
        self.enabled_skills = enabled_skills or {}
        self.custom_instructions = custom_instructions
        self.path_mapper = path_mapper
        self.memory_enabled = self.enabled_skills.get("memory", True)
        self.history_enabled = self.enabled_skills.get("history", True)

        self.context = ContextBuilder(
            workspace,
            enabled_tools=self.enabled_tools,
            custom_instructions=custom_instructions,
            path_mapper=self.path_mapper,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()

        self._running = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register tools based on enabled_tools config."""
        et = self.enabled_tools
        allowed_dir = self.workspace if self.restrict_to_workspace else None

        # Tools that are ON by default (original behavior)
        default_on = {"read_file", "write_file", "edit_file", "list_dir", "exec"}

        tool_map = {
            "read_file": lambda: ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "write_file": lambda: WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "edit_file": lambda: EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "list_dir": lambda: ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "exec": lambda: ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            ),
            "web_search": lambda: WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy),
            "web_fetch": lambda: WebFetchTool(proxy=self.web_proxy),
            "crop": lambda: CropTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "draw_bbox": lambda: DrawBBoxTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "draw_line": lambda: DrawLineTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "draw_circle": lambda: DrawCircleTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "color_segments": lambda: ColorSegmentsTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "sample_color": lambda: SampleColorTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "color_clusters": lambda: ColorClustersTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "in_range_color": lambda: InRangeColorTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "connected_components": lambda: ConnectedComponentsTool(
                workspace=self.workspace, allowed_dir=allowed_dir
            ),
            "find_contours": lambda: FindContoursTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "hough_lines": lambda: HoughLinesTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "hough_circles": lambda: HoughCirclesTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "template_match": lambda: TemplateMatchTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "rotate": lambda: RotateTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "flip": lambda: FlipTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "enhance_contrast": lambda: EnhanceContrastTool(
                workspace=self.workspace, allowed_dir=allowed_dir
            ),
            "adjust_brightness": lambda: AdjustBrightnessTool(
                workspace=self.workspace, allowed_dir=allowed_dir
            ),
            "detect_edges": lambda: DetectEdgesTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "grayscale": lambda: GrayscaleTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "render_html": lambda: RenderHtmlTool(workspace=self.workspace, allowed_dir=allowed_dir),
            "computer_use": lambda: ComputerUseTool(),
        }

        for name, factory in tool_map.items():
            default = name in default_on
            if et.get(name, default):
                tool = factory()
                if tool is not None:
                    self.tools.register(tool)

    def set_tool_workspace(self, workspace: Path, media_dir: Path | None = None) -> None:
        """Switch the workspace used by tools **and** session storage.

        Both file/exec tools and the ``SessionManager`` are pointed at
        *workspace*, so sessions and any tool-generated files end up in
        the same directory.

        ``media_dir`` is accepted for caller compatibility (e.g. distill
        run.py) but no longer participates in path virtualisation: media
        files are exposed to the model via per-message symlinks created
        by :meth:`PathMapper.link_into_workspace` (see ``ContextBuilder``).
        """
        del media_dir  # intentionally unused — see docstring
        workspace.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace
        if self.path_mapper is not None:
            self.path_mapper = PathMapper(workspace)
        self.context = ContextBuilder(
            workspace,
            enabled_tools=self.enabled_tools,
            custom_instructions=self.custom_instructions,
            path_mapper=self.path_mapper,
        )
        self.sessions = SessionManager(workspace, flat=True)
        self._register_default_tools()

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""

        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
            # import pdb; pdb.set_trace()

            if response.has_tool_calls:
                # computer_use is a format-only terminator for GUI grounding tasks:
                # when the model emits it, treat this tool_call itself as the final
                # output. Drop any sibling tool_calls from the same turn, do not
                # execute any tool, do not append any tool_result, and stop the
                # loop immediately.
                terminator = next((tc for tc in response.tool_calls if tc.name == "computer_use"), None)

                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    hint_calls = [terminator] if terminator is not None else response.tool_calls
                    await on_progress(self._tool_hint(hint_calls), tool_hint=True)

                effective_tool_calls = [terminator] if terminator is not None else response.tool_calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in effective_tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                if terminator is not None:
                    tools_used.append(terminator.name)
                    final_content = json.dumps(terminator.arguments, ensure_ascii=False)
                    break

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    # Rewrite virtual paths in tool arguments to real paths
                    args = tool_call.arguments
                    if self.path_mapper is not None and isinstance(args, dict):
                        args = self.path_mapper.rewrite_tool_args(tool_call.name, args)
                    args_str = json.dumps(args, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    if self.path_mapper is not None and tool_call.name == "exec":
                        with self.path_mapper.materialized_for_exec():
                            result = await self.tools.execute(tool_call.name, args)
                    else:
                        result = await self.tools.execute(tool_call.name, args)
                    # Rewrite real paths in tool results back to virtual paths
                    if self.path_mapper is not None:
                        result = self.path_mapper.rewrite_tool_result(result)
                    messages = self.context.add_tool_result(messages, tool_call.id, tool_call.name, result)
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages,
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(
                    lambda t, k=msg.session_key: (
                        self._active_tasks.get(k, []) and self._active_tasks[k].remove(t)
                        if t in self._active_tasks.get(k, [])
                        else None
                    )
                )

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        content = f"⏹ Stopped {cancelled} task(s)." if cancelled else "No active task to stop."
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
            )
        )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="",
                            metadata=msg.metadata or {},
                        )
                    )
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    )
                )

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            history = session.get_history(max_messages=self.memory_window) if self.history_enabled else []
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content,
                media_dir=session.metadata.get("media_dir"),
            )
            if messages and messages[0].get("role") == "system":
                session.metadata["system_prompt"] = messages[0].get("content", "")
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(
                channel=channel, chat_id=chat_id, content=final_content or "Background task completed."
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        requested_media_dir = msg.metadata.get("media_dir") if msg.metadata else None
        if isinstance(requested_media_dir, str) and requested_media_dir:
            session.metadata["media_dir"] = requested_media_dir
        media_dir = session.metadata.get("media_dir")

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            if not self.memory_enabled or not self.history_enabled:
                session.clear()
                self.sessions.save(session)
                self.sessions.invalidate(session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id, content="New session started."
                )
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated :]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="New session started.")
        if cmd == "/help":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="🐈 nanobot commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands",
            )

        unconsolidated = len(session.messages) - session.last_consolidated
        if (
            self.memory_enabled
            and self.history_enabled
            and unconsolidated >= self.memory_window
            and session.key not in self._consolidating
        ):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        history = session.get_history(max_messages=self.memory_window) if self.history_enabled else []
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            media_dir=media_dir if isinstance(media_dir, str) and media_dir else None,
        )

        # 为system prompt和user prompt添加显式缓存
        initial_messages[0]["content"] = [
            {"type": "text", "text": initial_messages[0]["content"], "cache_control": {"type": "ephemeral"}}
        ]
        initial_messages[1]["content"] = [
            content | {"cache_control": {"type": "ephemeral"}} for content in initial_messages[1]["content"]
        ]

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        if initial_messages and initial_messages[0].get("role") == "system":
            session.metadata["system_prompt"] = initial_messages[0].get("content", "")
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, list):
                # Multimodal tool result (e.g. read_file on images): strip base64 data
                # for persistence; keep only a text placeholder.
                num_image_parts = sum(
                    1 for c in content if isinstance(c, dict) and c.get("type") == "image_url"
                )
                text_parts = [
                    c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                ]
                entry["content"] = f"[image viewed * {num_image_parts}] " + " ".join(text_parts)
            elif role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, list):
                    filtered = []
                    media_paths = entry.pop("media_paths", [])
                    media_index = 0
                    for c in content:
                        if c.get("type") == "image_url" and c.get("image_url", {}).get("url", "").startswith(
                            "data:image/"
                        ):
                            media_path = media_paths[media_index] if media_index < len(media_paths) else None
                            media_index += 1
                            if isinstance(media_path, str) and media_path:
                                filtered.append(
                                    {
                                        "type": "image",
                                        "image": self._to_relative_media_path(media_path, session),
                                    }
                                )
                            else:
                                filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    def _to_relative_media_path(self, media_path: str, session: Session) -> str:
        """Store media paths relative to the session's configured media directory."""
        media_dir = session.metadata.get("media_dir") or str(self.workspace)
        return os.path.relpath(media_path, media_dir)

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(
            session,
            self.provider,
            self.model,
            archive_all=archive_all,
            memory_window=self.memory_window,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly."""
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
