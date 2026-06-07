"""System prompts and prompt builders for the agent rollout drivers.

Pure string/templating module — no I/O, no model calls. The native
function-calling loop (``function_call.py``), the code-as-tool baselines
(``code_agent.py``), and the grounding drivers (``grounding.py``) all draw
their system/user prompts from here so prompt text lives in exactly one place.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT_WITH_TOOLS = """You are a helpful assistant specialized in solving real-world visual tasks with tools. Combine reasoning with available tools to answer questions accurately.

## Workspace
Your workspace is at: {workspace_path}

## Guidelines
- Use tools to inspect, transform, or measure images whenever they can reveal information that helps solve the task.
- Vision coordinates: all vision tools use `0-1000 relative` coordinates. Follow that scale exactly.
- After saving an image (e.g. plt.savefig, cv2.imwrite, PIL.Image.save), ALWAYS call read_file tool to view the saved image and verify the result.
- When writing Python/scripts that read user-supplied images, ALWAYS use the exact paths listed under `[Image file paths — use these in scripts]` in the user message (e.g. `Image.open('/mnt/data/0000000.jpg')`). NEVER invent or hard-code placeholders like `image_clue`, `image_clue[0]`, `input_file`, or other guessed filenames.
"""


SYSTEM_PROMPT_GUI_GROUNDING_WITH_TOOLS = SYSTEM_PROMPT_WITH_TOOLS + """
## GUI Grounding Completion Rule
- You are solving a GUI grounding task. The task is only complete after calling the `computer_use` tool.
- Tools such as `crop`, `draw_bbox`, `read_file`, and image-processing tools are inspection tools only. They help locate the target but NEVER complete the task.
- Do not end with a natural-language final answer after using inspection tools.
- Your final action MUST be exactly one `computer_use` tool call that clicks the requested target UI element.
"""


# Per-tool inspection hints used to build a tool-set-aware GUI grounding
# prompt for the crop / crop+draw_bbox / crop+draw_circle ablation.
_GUI_TOOL_HINTS = {
    "crop": "- Use `crop` to zoom into the region that likely contains the target so you can see it clearly.",
    "draw_bbox": "- Use `draw_bbox` to draw a tight box around the target, look at the result, and adjust the box if it is off before clicking its center.",
    "draw_circle": "- Use `draw_circle` with a small `radius` to mark your exact candidate click point, look at the result, and move the marker if it is not precisely on the target before clicking that point.",
}


def build_gui_grounding_prompt(allowed_tools: list[str] | None) -> str:
    """GUI grounding system prompt whose inspection-tool guidance matches the
    tools actually registered. Used by the tool-set ablation so each variant
    is only prompted about the tools it has.
    """
    if not allowed_tools:
        return SYSTEM_PROMPT_GUI_GROUNDING_WITH_TOOLS
    insp = [t for t in allowed_tools if t in _GUI_TOOL_HINTS]
    names = ", ".join(f"`{t}`" for t in insp) or "the available inspection tools"
    hints = "\n".join(_GUI_TOOL_HINTS[t] for t in insp)
    return SYSTEM_PROMPT_WITH_TOOLS + f"""
## GUI Grounding Completion Rule
- You are solving a GUI grounding task. The task is only complete after calling the `computer_use` tool.
- Tools such as {names} are inspection tools only. They help locate the target but NEVER complete the task.
{hints}
- Do not end with a natural-language final answer after using inspection tools.
- Your final action MUST be exactly one `computer_use` tool call that clicks the requested target UI element.
"""


SYSTEM_PROMPT_NO_TOOLS = """You are a helpful assistant.

## Guidelines
- Think step by step and show your reasoning clearly.
- Be concise and direct in your responses."""


# Qwen2.5-VL GUI grounding: the model is trained to emit ABSOLUTE pixel
# coordinates in its (smart-resized) input space, NOT the 0-1000 normalized
# computer_use convention. So we drop the computer_use tool channel entirely
# and ask it to commit a bare `(x, y)` point; the scorer maps that back to the
# original image via smart_resize. Used for native Qwen2.5-VL and shares the
# same answer convention as the Thyme / DeepEyesV2 code-agent grounding path.
_GUI_COORD_OUTPUT_RULE = (
    "You are solving a GUI grounding task: the user describes a single target "
    "UI element shown in the screenshot, and you must locate it.\n"
    "- Report where to click using ABSOLUTE pixel coordinates, with the origin "
    "at the top-left corner of the image (x increases to the right, y downward).\n"
    "- Point at the CENTER of the target element.\n"
    "- Do NOT normalize the coordinates (they are raw pixels, not 0-1000) and "
    "do NOT wrap them in any tool call or function call.\n"
    "- After any optional reasoning, the LAST line of your reply must be the "
    "click point and nothing else.\n\n"
    "Output the coordinate pair exactly: (x,y)"
)


SYSTEM_PROMPT_GUI_GROUNDING_COORD_QWEN25 = (
    "You are a helpful assistant.\n\n" + _GUI_COORD_OUTPUT_RULE
)


def build_gui_grounding_coord_prompt(allowed_tools: list[str] | None) -> str:
    """Qwen2.5-VL GUI grounding prompt that commits a bare absolute-pixel
    ``(x, y)`` point instead of a 0-1000 ``computer_use`` call.

    With inspection tools registered (`allowed_tools`), the model may crop/zoom
    the screenshot first and then give the ``(x, y)`` as its final text answer;
    the inspection tools never submit the answer themselves. With no tools
    (`allowed_tools` falsy) it returns the plain direct-answer prompt.
    """
    if not allowed_tools:
        return SYSTEM_PROMPT_GUI_GROUNDING_COORD_QWEN25
    insp = [t for t in allowed_tools if t in _GUI_TOOL_HINTS]
    hints = "\n".join(_GUI_TOOL_HINTS[t] for t in insp)
    tool_block = (f"\n\n## Inspection tools\n{hints}\n"
                  "- These tools only help you look more closely; they NEVER "
                  "submit the answer." if hints else "")
    return (
        "You are a helpful assistant specialized in GUI grounding. You may use "
        "the available image inspection tools to zoom into the screenshot before "
        "deciding where to click." + tool_block + "\n\n" + _GUI_COORD_OUTPUT_RULE
    )




SYSTEM_PROMPT_VISION2WEB_WEBPAGE = """You are a senior front-end engineer with expertise in static responsive web development.

Build one production-quality static webpage that visually matches the provided desktop, tablet, and mobile prototype screenshots. Use the prototypes as the primary reference for layout, typography, spacing, colors, images, navigation, content sections, and responsive behavior.

## Assets
- If tools are available, inspect `resources/` and `resources/images/` before choosing image, logo, icon, or media paths.
- Use matching resource files directly with relative paths such as `resources/images/example.jpg`.
- Do not use `desktop.jpg`, `tablet.jpg`, or `mobile.jpg` as page content images; they are reference screenshots only.
- If no matching asset exists, recreate that area with CSS shapes, gradients, colors, or neutral placeholders that preserve the prototype layout.

## Implementation
- Produce a single complete HTML document with inline CSS and any minimal inline JavaScript needed for static UI behavior.
- The page must run as a local static file. Do not require a backend, database, build step, server, package install, CDN, remote URL, or external API.
- Use semantic HTML and CSS media queries to match all provided viewport prototypes.
- Use only real available resource filenames. Do not invent missing files and do not use `rick.jpg`.

## Final Answer Protocol
- Tools are only for inspection, drafting, and optional verification.
- A file written with `write_file` is not the final answer.
- The evaluator reads only your final assistant message, not files you created.
- Submit the final HTML in the assistant message. Quote the full HTML once, then stop.
- Your final assistant message must contain the complete HTML code itself.
- The final assistant message must start with `<!DOCTYPE html>` and end with `</html>`.
- Do not include markdown fences, explanations, summaries, paths, README text, deployment notes, or statements such as "created", "saved", "written", or "available at index.html".
"""


# Used when eval_mode == "gui" and tools_enabled == False: the test model
# has no native function-calling tools available, so we inline the
# `computer_use` schema in the system prompt and ask it to emit a single
# Hermes-style <tool_call> text block. The downstream gui_grounding
# extractor parses both this text format and the native tool_calls path.
SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS = """# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n* The screen's resolution is 1000x1000.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\\n* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\\n* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\\n* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\\n* `answer`: Answer a question.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "scroll", "hscroll", "wait", "terminate", "answer"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll` and `action=hscroll`.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are a helpful assistant. You are solving a GUI grounding task. Only use the `computer_use` function to click the target UI element requested by the user.
"""


_COMPUTER_USE_TOOL_SCHEMA_QWEN25 = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n"
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n"
            "* The screen's resolution is 1000x1000.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n"
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "triple_click",
                        "scroll",
                        "hscroll",
                        "wait",
                        "terminate",
                        "answer",
                    ],
                    "description": (
                        "The action to perform. The available actions are:\n"
                        "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n"
                        "* `type`: Type a string of text on the keyboard.\n"
                        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).\n"
                        "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
                        "* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).\n"
                        "* `wait`: Wait specified seconds for the change to happen.\n"
                        "* `terminate`: Terminate the current task and report its completion status.\n"
                        "* `answer`: Answer a question."
                    ),
                },
                "keys": {
                    "type": "array",
                    "description": "Required only by `action=key`.",
                },
                "text": {
                    "type": "string",
                    "description": "Required only by `action=type` and `action=answer`.",
                },
                "coordinate": {
                    "type": "array",
                    "description": (
                        "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
                        "coordinates to move the mouse to. Values are in the normalized 0-1000 space "
                        "(screen resolution is 1000x1000)."
                    ),
                },
                "pixels": {
                    "type": "number",
                    "description": (
                        "The amount of scrolling to perform. Positive values scroll up, negative "
                        "values scroll down. Required only by `action=scroll` and `action=hscroll`."
                    ),
                },
                "time": {
                    "type": "number",
                    "description": "The seconds to wait. Required only by `action=wait`.",
                },
                "status": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "The status of the task. Required only by `action=terminate`.",
                },
            },
            "required": ["action"],
        },
    },
}


# Qwen2.5-VL's chat template uses a different text tool-call protocol from the
# Hermes-style prompt above: the <tool_call> body is a single JSON object with
# top-level `name` and `arguments` keys.
SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN25 = f"""You are a helpful assistant. You are solving a GUI grounding task. Only use the `computer_use` function to click the target UI element requested by the user.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{json.dumps(_COMPUTER_USE_TOOL_SCHEMA_QWEN25, ensure_ascii=False)}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [542, 74]}}}}
</tool_call>

<IMPORTANT>
Reminder:
- Use normalized 0-1000 coordinates for `coordinate`; the screen resolution is 1000x1000.
- For GUI grounding, the final answer MUST be exactly one `computer_use` call that clicks the requested target UI element.
- The tool call MUST follow the Qwen2.5-VL format above: a single JSON object inside <tool_call></tool_call>, not nested <function> or <parameter> XML tags.
- Do not output any text after the tool call.
</IMPORTANT>
"""


# Qwen3-VL shares Qwen2.5-VL's JSON-in-<tool_call> protocol (a single JSON
# object with top-level `name` / `arguments`), but its canonical GUI grounding
# prompt frames the task as a strict single-step left_click. Mirrors
# distill/filter/rollout_prompts/gui_grounding_qwen3vl.md.
SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN3VL = f"""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{json.dumps(_COMPUTER_USE_TOOL_SCHEMA_QWEN25, ensure_ascii=False)}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Guidelines

You are solving a GUI grounding task.

- Your goal is only to identify the target UI element requested by the user and click it.
- This is a single-step grounding task, not a full desktop-assistant task.
- Only use the `computer_use` function with `action="left_click"`.
- Return exactly one tool call that contains the click coordinate.
- Use normalized 0-1000 coordinates for `coordinate`; the screen resolution is 1000x1000.
- Do not use `mouse_move`, `type`, `key`, `scroll`, `wait`, `answer`, `terminate`, or any other action.
- Do not try to complete follow-up steps after the click. Only predict the click location.
- Base the coordinate on the screenshot and the user request only.
"""
