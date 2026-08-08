from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.loop import AgentLoop
from nanobot.agent.path_mapper import PathMapper
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config, set_config_path
from nanobot.config.paths import get_workspace_path


def _make_provider(config: Any):
    """Build an LLM provider without importing the personal-assistant CLI."""
    from nanobot.providers.azure_openai_provider import AzureOpenAIProvider
    from nanobot.providers.custom_provider import CustomProvider
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.registry import find_by_name

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    provider_config = config.get_provider(model)

    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=model)
    if provider_name == "custom":
        return CustomProvider(
            api_key=provider_config.api_key if provider_config else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )
    if provider_name == "azure_openai":
        if not provider_config or not provider_config.api_key or not provider_config.api_base:
            raise ValueError(
                "Azure OpenAI requires apiKey and apiBase under providers.azureOpenai"
            )
        return AzureOpenAIProvider(
            api_key=provider_config.api_key,
            api_base=provider_config.api_base,
            default_model=model,
        )

    specification = find_by_name(provider_name)
    if (
        not model.startswith("bedrock/")
        and not (provider_config and provider_config.api_key)
        and not (specification and specification.is_oauth)
    ):
        raise ValueError(f"No API key configured for provider {provider_name!r}")
    return LiteLLMProvider(
        api_key=provider_config.api_key if provider_config else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=provider_config.extra_headers if provider_config else None,
        provider_name=provider_name,
    )


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
        enabled_tools=enabled_tools,
        enabled_skills=enabled_skills,
        custom_instructions=config.agents.defaults.custom_instructions,
        path_mapper=path_mapper,
    )

    return agent, resolved_workspace
