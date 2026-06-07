"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from nanobot.utils.helpers import detect_image_mime


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    DEFAULT_ON_TOOLS = {"read_file", "write_file", "edit_file", "list_dir", "exec"}

    def __init__(
        self,
        workspace: Path,
        *,
        enabled_tools: dict[str, bool] | None = None,
        custom_instructions: str | None = None,
        path_mapper: Any | None = None,
    ):
        self.workspace = workspace
        self.enabled_tools = enabled_tools or {}
        self.custom_instructions = custom_instructions
        self._has_tools = self._has_enabled_tools(self.enabled_tools)
        self._path_mapper = path_mapper

    def build_system_prompt(self) -> str:
        """Build the system prompt for the current tool mode."""
        if not self._has_tools:
            return self._build_notool_system_prompt()

        return self._build_tool_system_prompt()

    def _build_tool_system_prompt(self) -> str:
        """Build the base system prompt for agents with tools."""
        workspace_path = str(self.workspace.expanduser().resolve())
        if self._path_mapper is not None:
            workspace_path = self._path_mapper.to_virtual(workspace_path)

        return self._join_prompt_parts(
            f"""You are a helpful assistant specialized in solving real-world visual tasks with tools. Combine reasoning with available tools to answer questions accurately.

## Workspace
Your workspace is at: {workspace_path}

## Guidelines
- Use tools to inspect, transform, or measure images whenever they can reveal information that helps solve the task.
- Vision coordinates: all vision tools use `0-1000 relative` coordinates. Follow that scale exactly.
- After saving an image (e.g. plt.savefig, cv2.imwrite, PIL.Image.save), ALWAYS call read_file tool to view the saved image and verify the result.
- When writing Python/scripts that read user-supplied images, ALWAYS use the exact paths listed under `[Image file paths — use these in scripts]` in the user message (e.g. `Image.open('/mnt/data/0000000.jpg')`). NEVER invent or hard-code placeholders like `image_clue`, `image_clue[0]`, `input_file`, or other guessed filenames.
""",
            self._resolve_custom_instructions(),
        )

    def _build_notool_system_prompt(self) -> str:
        """Build a compact system prompt for agents without tools.

        When custom instructions are supplied (e.g. a self-contained GUI
        grounding prompt that declares its own computer_use tool block in text),
        they fully define the system prompt — mirroring the direct-vLLM
        --system-prompt-md path in distill/filter so no-tool rollouts stay
        consistent across domains.
        """
        custom = self._resolve_custom_instructions()
        if custom:
            return custom
        return """You are a helpful assistant.

## Guidelines
- Think step by step and show your reasoning clearly.
- Be concise and direct in your responses."""

    @staticmethod
    def _join_prompt_parts(*parts: str | None) -> str:
        """Join non-empty prompt sections with the standard separator."""
        return "\n\n---\n\n".join(part.strip() for part in parts if part and part.strip())

    @classmethod
    def _has_enabled_tools(cls, enabled_tools: dict[str, bool]) -> bool:
        """Return whether any tool is enabled, respecting default-on core tools."""
        if any(enabled_tools.get(name, True) for name in cls.DEFAULT_ON_TOOLS):
            return True
        return any(enabled for name, enabled in enabled_tools.items() if name not in cls.DEFAULT_ON_TOOLS)

    def _resolve_custom_instructions(self) -> str | None:
        """Load custom instructions from a referenced file path when possible."""
        if not self.custom_instructions:
            return None

        raw_value = self.custom_instructions.strip()
        if not raw_value:
            return None

        instruction_file = self._resolve_custom_instruction_path(raw_value)
        if instruction_file is None:
            return raw_value

        return instruction_file.read_text(encoding="utf-8").strip()

    def _resolve_custom_instruction_path(self, value: str) -> Path | None:
        """Resolve a custom instructions file from the provided path."""
        normalized = value[1:] if value.startswith("@") else value
        raw_path = Path(normalized).expanduser()
        candidates: list[Path] = []

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend(
                [
                    raw_path.resolve(),
                    (self._project_root() / raw_path).resolve(),
                ]
            )

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _project_root() -> Path:
        """Return the repository root for resolving shared instruction files."""
        return Path(__file__).resolve().parents[2]

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        media: list[str] | None = None,
        media_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        user_content = self._build_user_content(current_message, media)
        restored_history = [self._restore_history_message(message, media_dir) for message in history]

        user_message: dict[str, Any] = {"role": "user", "content": user_content}
        if media:
            user_message["media_paths"] = list(media)

        return [
            {"role": "system", "content": self.build_system_prompt()},
            *restored_history,
            user_message,
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images and file path references."""
        if not media:
            return text

        image_blocks: list[dict[str, Any]] = []
        image_paths: list[str] = []
        doc_paths: list[str] = []

        for path in media:
            image_block = self._build_image_block(path)
            if image_block is not None:
                image_blocks.append(image_block)
                # When a path_mapper is active, materialise the file as a
                # symlink inside the workspace so the model can reach it via
                # /mnt/data/<name> *and* via subprocess relative paths
                # (cd /mnt/data && open('foo.jpg')).
                if self._path_mapper is not None:
                    resolved = self._path_mapper.link_into_workspace(path)
                else:
                    resolved = str(Path(path).resolve())
                image_paths.append(resolved)
            elif Path(path).is_file():
                if self._path_mapper is not None:
                    resolved = self._path_mapper.link_into_workspace(path)
                else:
                    resolved = str(Path(path).resolve())
                doc_paths.append(resolved)

        text_parts: list[str] = []
        # File-path hints only help when tools (exec/read_file) can act on them.
        # In no-tool mode they are useless and actively mislead the model into
        # writing scripts it cannot run, so omit them — the image is already
        # provided inline, matching the direct-vLLM no-tool flow in distill/filter.
        if self._has_tools:
            if image_paths:
                paths_info = "\n".join(f"- {p}" for p in image_paths)
                text_parts.append(f"[Image file paths — use these in scripts]\n{paths_info}")
            if doc_paths:
                paths_info = "\n".join(f"- {p}" for p in doc_paths)
                text_parts.append(f"[File paths — read these using tools such as read_file]\n{paths_info}")
        text_parts.append(text)
        combined_text = "\n\n".join(text_parts)

        if not image_blocks:
            return combined_text

        image_blocks.append({"type": "text", "text": combined_text})
        return image_blocks

    def _build_image_block(self, path: str) -> dict[str, Any] | None:
        """Convert an image path into an OpenAI-compatible image_url content block."""
        p = Path(path)
        if not p.is_file():
            return None
        raw = p.read_bytes()
        # Detect real MIME type from magic bytes; fallback to filename guess
        mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
        if not mime or not mime.startswith("image/"):
            return None
        b64 = base64.b64encode(raw).decode()
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

    def _restore_history_message(
        self,
        message: dict[str, Any],
        media_dir: str | None = None,
    ) -> dict[str, Any]:
        """Restore persisted multimodal history into provider-ready content blocks."""
        content = message.get("content")
        if not isinstance(content, list):
            return message

        restored: list[dict[str, Any]] = []
        changed = False
        for item in content:
            if not isinstance(item, dict):
                restored.append(item)
                continue
            if item.get("type") not in {"image", "image_path"}:
                restored.append(item)
                continue

            changed = True
            path = self._resolve_persisted_image_path(item, media_dir)
            if path is not None:
                image_block = self._build_image_block(path)
                if image_block is not None:
                    restored.append(image_block)
                    continue
                restored.append({"type": "text", "text": f"[image: {path}]"})
                continue

            raw_path = item.get("image") if item.get("type") == "image" else item.get("path")
            if isinstance(raw_path, str) and raw_path:
                restored.append({"type": "text", "text": f"[image: {raw_path}]"})
            else:
                restored.append({"type": "text", "text": "[image]"})

        if not changed:
            return message

        restored_message = dict(message)
        restored_message["content"] = restored
        return restored_message

    def _resolve_persisted_image_path(
        self,
        item: dict[str, Any],
        media_dir: str | None = None,
    ) -> str | None:
        """Resolve a persisted image entry to an absolute filesystem path."""
        raw_path = item.get("image") if item.get("type") == "image" else item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return None
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return str(path)
        base_dir = Path(media_dir).expanduser() if media_dir else self.workspace
        return os.path.abspath(base_dir / path)

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list.

        ``result`` may be a plain string or a list of content blocks (e.g.
        image_url + text) for multimodal tool results like ``read_file`` on images.
        """
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks
        messages.append(msg)
        return messages
