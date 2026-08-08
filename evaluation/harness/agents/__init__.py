"""Agent rollout drivers — the loops that turn a model + tools into a `pred`.

- ``function_call.run``           native OpenAI tool-calling loop (default)
- ``code_agent.run_code_agent``   code-as-tool baselines (Thyme / DeepEyesV2)
- ``grounding.run_qwen25_grounding`` base Qwen2.5-VL single-shot ScreenSpot-Pro
- ``prompts``                     all system/user prompt text + builders
"""

from .code_agent import run_code_agent
from .function_call import run
from .grounding import (
    computer_use_tools_block,
    grounding_click_instruction,
    resize_grounding_images,
    run_qwen25_grounding,
)
from .prompts import (
    SYSTEM_PROMPT_GUI_GROUNDING_COORD_QWEN25,
    SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS,
    SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN3VL,
    SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN25,
    SYSTEM_PROMPT_GUI_GROUNDING_WITH_TOOLS,
    SYSTEM_PROMPT_NO_TOOLS,
    SYSTEM_PROMPT_VISION2WEB_WEBPAGE,
    SYSTEM_PROMPT_WITH_TOOLS,
    build_gui_grounding_coord_prompt,
    build_gui_grounding_prompt,
)

__all__ = [
    "run",
    "run_code_agent",
    "run_qwen25_grounding",
    "resize_grounding_images",
    "computer_use_tools_block",
    "grounding_click_instruction",
    "SYSTEM_PROMPT_WITH_TOOLS",
    "SYSTEM_PROMPT_NO_TOOLS",
    "SYSTEM_PROMPT_GUI_GROUNDING_WITH_TOOLS",
    "SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS",
    "SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN25",
    "SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN3VL",
    "SYSTEM_PROMPT_GUI_GROUNDING_COORD_QWEN25",
    "SYSTEM_PROMPT_VISION2WEB_WEBPAGE",
    "build_gui_grounding_prompt",
    "build_gui_grounding_coord_prompt",
]
