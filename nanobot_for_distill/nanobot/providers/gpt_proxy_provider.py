"""SenseTime internal GptProxy provider (openai_proxy.GptProxy)."""

from __future__ import annotations

import asyncio
import getpass
import os
import sys
from typing import Any

import json_repair

# Resolve project root and ensure gpt_proxy_client is importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gpt_proxy_client.openai_proxy import GptProxy  # noqa: E402
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


def _detect_channel_code(model: str) -> str:
    """Map model name to the GptProxy channel_code routing field."""
    m = model.lower()
    if "doubao" in m or "deepseek" in m or "seed" in m:
        return "doubao"
    if "qwen" in m:
        return "ali"
    if "gpt" in m:
        return "azure"
    return "doubao"


class GptProxyProvider(LLMProvider):
    """
    SenseTime-internal LLM provider backed by openai_proxy.GptProxy.

    The proxy uses a custom HTTP protocol (not OpenAI-compatible), so this
    provider bypasses LiteLLM entirely and drives the synchronous GptProxy
    client via asyncio.to_thread().
    """

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "Doubao-1.5-pro-32k",
        base_url: str | None = None,
    ):
        super().__init__(api_key=api_key, api_base=base_url)
        self.default_model = default_model
        self._user = getpass.getuser()

        proxy_kwargs: dict[str, Any] = {}
        if api_key:
            proxy_kwargs["api_key"] = api_key
        if base_url:
            proxy_kwargs["base_url"] = base_url
        self._proxy = GptProxy(**proxy_kwargs)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        resolved_model = model or self.default_model
        channel_code = _detect_channel_code(resolved_model)
        transaction_id = f"{self._user}-{resolved_model}"

        call_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max(1, max_tokens),
            "channel_code": channel_code,
        }
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"
        if reasoning_effort:
            # Map nanobot's reasoning_effort levels to GptProxy thinking types.
            thinking_type = "enabled" if reasoning_effort in ("medium", "high") else "disabled"
            call_kwargs["thinking"] = {"type": thinking_type}

        clean_messages = self._sanitize_empty_content(messages)

        def _call() -> tuple[bool, Any]:
            rsp = self._proxy.generate(
                messages=clean_messages,
                model=resolved_model,
                transaction_id=transaction_id,
                **call_kwargs,
            )
            if rsp.ok:
                return True, rsp.json()
            return False, rsp.text

        try:
            ok, data = await asyncio.to_thread(_call)
        except Exception as e:
            return LLMResponse(content=f"GptProxy call error: {e}", finish_reason="error")

        if not ok:
            return LLMResponse(content=f"GptProxy error: {data}", finish_reason="error")

        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> LLMResponse:
        """Parse the GptProxy response envelope into a standard LLMResponse.

        GptProxy response structure:
            {
                "data": {
                    "response_content": {
                        "choices": [{"finish_reason": "stop", "message": {...}}],
                        "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
                    }
                }
            }
        """
        try:
            response_content = data["data"]["response_content"]
            choice = response_content["choices"][0]
            message = choice["message"]

            tool_calls: list[ToolCallRequest] = []
            for tc in message.get("tool_calls") or []:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json_repair.loads(args)
                tool_calls.append(
                    ToolCallRequest(
                        id=tc.get("id", ""),
                        name=tc["function"]["name"],
                        arguments=args,
                    )
                )

            usage: dict[str, int] = {}
            if response_content.get("usage"):
                u = response_content["usage"]
                usage = {
                    "prompt_tokens": u.get("prompt_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0),
                }

            return LLMResponse(
                content=message.get("content"),
                tool_calls=tool_calls,
                finish_reason=choice.get("finish_reason", "stop"),
                usage=usage,
                reasoning_content=message.get("reasoning_content") or None,
            )

        except (KeyError, IndexError, TypeError) as e:
            return LLMResponse(
                content=f"Error parsing GptProxy response: {e}\nraw={data}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        return self.default_model
