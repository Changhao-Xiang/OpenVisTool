"""Async OpenAI client wrapper with retry, per-call timeout, and tunable connection pool."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from .config import Endpoint, RetryConfig


_RETRIABLE = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class LLMClient:
    """Thin wrapper around AsyncOpenAI that retries transient errors.

    `.chat(messages, tools=..., **override)` merges the endpoint's default
    sampling kwargs with any per-call overrides. Each call enforces
    `request_timeout_s` server-side hang protection (default 90s).
    """

    def __init__(self, endpoint: Endpoint, retry: RetryConfig | None = None,
                 label: str = "llm",
                 request_timeout_s: float = 90.0,
                 max_connections: int = 32):
        self.endpoint = endpoint
        self.retry = retry or RetryConfig()
        self.label = label
        self.request_timeout_s = request_timeout_s

        # Explicit httpx client so we can control the connection pool.
        # Without this, httpx defaults to max_connections=100 (per-host=20),
        # which is fine for moderate concurrency, but we want it tied to
        # the eval's worker count for predictable behavior.
        self._http = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            timeout=httpx.Timeout(request_timeout_s),
        )
        self._client = AsyncOpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            http_client=self._http,
            timeout=request_timeout_s,
        )

    async def chat(self, messages: list, *, tools: list | None = None,
                   **override: Any):
        kwargs: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            **self.endpoint.sampling_kwargs(),
            **override,
        }
        if tools:
            kwargs["tools"] = tools
        # Per-call timeout: caps both connect+read time. If the upstream
        # silently hangs, AsyncOpenAI raises APITimeoutError (which is
        # retriable below).
        kwargs["timeout"] = self.request_timeout_s

        last_exc: Exception | None = None
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except _RETRIABLE as e:
                last_exc = e
                if attempt == self.retry.max_attempts:
                    raise
                delay = min(self.retry.max_delay,
                            self.retry.base_delay * (2 ** (attempt - 1)))
                delay *= 0.5 + random.random()  # jitter [0.5x, 1.5x]
                print(f"[{self.label}] {type(e).__name__} on attempt {attempt}/"
                      f"{self.retry.max_attempts}, retrying in {delay:.1f}s: {e}",
                      flush=True)
                await asyncio.sleep(delay)
            except APIError:
                # Non-transient API error — don't retry.
                raise

        # unreachable
        raise RuntimeError(f"retry exhausted without raising: last={last_exc}")

    async def close(self) -> None:
        await self._client.close()
        await self._http.aclose()
