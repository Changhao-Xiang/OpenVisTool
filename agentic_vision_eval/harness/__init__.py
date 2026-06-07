"""agentic-vision-eval harness library.

The eval pipeline split into four cohesive layers (import top→down only):

    core     config / llm client / benchmark loader / io / path map / image
    agents   rollout drivers (function-calling, code-agent, grounding) + prompts
    scoring  extractor / judge / gui-grounding / vision2web render+score
    runtime  sandboxes / result aggregation / live dashboard

The entry scripts (eval.py, rejudge.py) import from the subpackages directly,
e.g. ``from harness.agents import run``. This module additionally re-exports the
public API as a flat convenience facade (``from harness import run``).
"""

from .core import (
    AppConfig, Endpoint, RetryConfig, DEFAULT_GENERAL_CONFIG, load_config, summary,
    LLMClient, load_benchmark, task_key, build_content, encode_image, strip_base64,
    PathMapper, VIRTUAL_ROOT,
    smart_resize, IMAGE_FACTOR, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS,
)
from .agents import (
    run, run_code_agent, run_qwen25_grounding, resize_grounding_images,
    SYSTEM_PROMPT_WITH_TOOLS, SYSTEM_PROMPT_NO_TOOLS,
    SYSTEM_PROMPT_GUI_GROUNDING_WITH_TOOLS,
    SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS,
    SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN25,
    SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN3VL,
    SYSTEM_PROMPT_GUI_GROUNDING_COORD_QWEN25,
    SYSTEM_PROMPT_VISION2WEB_WEBPAGE,
    build_gui_grounding_prompt, build_gui_grounding_coord_prompt,
)
from .scoring import (
    Extractor, EXTRACT_PROMPT, Judge, JUDGE_PROMPT, VISION2WEB_STATIC_JUDGE_PROMPT,
    is_grounding_item, score_grounding, score_grounding_pyautogui,
    DEVICES, DEFAULT_VIEWPORTS, is_vision2web_item, is_vision2web_source,
    vision2web_viewports, vision2web_ref_images, judge_vision2web_viewports,
    extract_html, resolve_submitted_html, render_html_to_viewport_images,
)
from .runtime import (
    build_registry, setup_sandbox, setup_sample_sandbox, WORKSPACE_TOOL_CLASSES,
    aggregate_results, failed_ids, print_summary, EvalDashboard,
)

__all__ = [
    # core
    "AppConfig", "Endpoint", "RetryConfig", "DEFAULT_GENERAL_CONFIG",
    "load_config", "summary", "LLMClient", "load_benchmark", "task_key",
    "build_content", "encode_image", "strip_base64", "PathMapper", "VIRTUAL_ROOT",
    "smart_resize", "IMAGE_FACTOR", "DEFAULT_MIN_PIXELS", "DEFAULT_MAX_PIXELS",
    # agents
    "run", "run_code_agent", "run_qwen25_grounding", "resize_grounding_images",
    "SYSTEM_PROMPT_WITH_TOOLS", "SYSTEM_PROMPT_NO_TOOLS",
    "SYSTEM_PROMPT_GUI_GROUNDING_WITH_TOOLS",
    "SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS",
    "SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN25",
    "SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN3VL",
    "SYSTEM_PROMPT_GUI_GROUNDING_COORD_QWEN25",
    "SYSTEM_PROMPT_VISION2WEB_WEBPAGE",
    "build_gui_grounding_prompt", "build_gui_grounding_coord_prompt",
    # scoring
    "Extractor", "EXTRACT_PROMPT", "Judge", "JUDGE_PROMPT",
    "VISION2WEB_STATIC_JUDGE_PROMPT",
    "is_grounding_item", "score_grounding", "score_grounding_pyautogui",
    "DEVICES", "DEFAULT_VIEWPORTS", "is_vision2web_item", "is_vision2web_source",
    "vision2web_viewports", "vision2web_ref_images", "judge_vision2web_viewports",
    "extract_html", "resolve_submitted_html", "render_html_to_viewport_images",
    # runtime
    "build_registry", "setup_sandbox", "setup_sample_sandbox",
    "WORKSPACE_TOOL_CLASSES", "aggregate_results", "failed_ids", "print_summary",
    "EvalDashboard",
]
