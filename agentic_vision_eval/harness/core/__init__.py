"""Foundational, dependency-free building blocks.

Config loading, the async LLM client, the benchmark loader, multimodal I/O
encoding, the sandbox<->virtual path mapper, and the Qwen2.5-VL image geometry.
Everything else in ``harness`` builds on top of ``core``; ``core`` imports
nothing from the other subpackages, which keeps the dependency graph acyclic.
"""

from .config import (
    AppConfig,
    Endpoint,
    RetryConfig,
    DEFAULT_GENERAL_CONFIG,
    load_config,
    summary,
)
from .llm import LLMClient
from .benchmark import load_benchmark, task_key
from .io import build_content, encode_image, strip_base64
from .path_mapper import PathMapper, VIRTUAL_ROOT
from .image import (
    smart_resize, IMAGE_FACTOR, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS,
)

__all__ = [
    "AppConfig", "Endpoint", "RetryConfig", "DEFAULT_GENERAL_CONFIG",
    "load_config", "summary",
    "LLMClient",
    "load_benchmark", "task_key",
    "build_content", "encode_image", "strip_base64",
    "PathMapper", "VIRTUAL_ROOT",
    "smart_resize", "IMAGE_FACTOR", "DEFAULT_MIN_PIXELS", "DEFAULT_MAX_PIXELS",
]
