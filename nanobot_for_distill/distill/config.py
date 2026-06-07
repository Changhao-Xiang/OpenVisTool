from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.loop import AgentLoop
from nanobot.agent.path_mapper import PathMapper
from nanobot.bus.queue import MessageBus
from nanobot.cli.commands import _make_provider
from nanobot.config.loader import load_config, set_config_path
from nanobot.config.paths import get_workspace_path


def load_runtime_config(config_path: Path | None):
    if config_path is not None:
        set_config_path(config_path)
    config = load_config(config_path)

    return config


def build_agent_from_config(config: Any) -> tuple[AgentLoop, Path]:
    """Create a fully configured AgentLoop from a loaded config."""
    resolved_workspace = get_workspace_path(str(config.workspace_path))
    provider = _make_provider(config)

    bus = MessageBus()

    # Build enabled_tools/enabled_skills from config
    enabled_tools = config.tools.enabled_tools.model_dump()
    enabled_skills = config.tools.enabled_skills.model_dump()

    # Create a PathMapper for distill mode (media_dir will be set per-session
    # via set_tool_workspace; start with workspace-only mapping).
    path_mapper = PathMapper(resolved_workspace)

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=resolved_workspace,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        brave_api_key=config.tools.web.search.api_key or None,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        channels_config=config.channels,
        enabled_tools=enabled_tools,
        enabled_skills=enabled_skills,
        custom_instructions=config.agents.defaults.custom_instructions,
        path_mapper=path_mapper,
    )

    return agent, resolved_workspace
