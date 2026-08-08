"""Shared ScreenSpot-Pro grounding protocol for the Qwen2.5-VL family.

Every Qwen2.5-VL-based model (base + the Thyme / DeepEyesV2 code agents) is
scored under ONE protocol so the comparison is fair:

  1. **Pre-resize** the screenshot via ``smart_resize`` and send THAT image, so
     the model sees exactly the resolution we declare — independent of whatever
     ``max_pixels`` the serving stack uses (our server is pinned at the HF
     default 12.8M; pre-resizing to ``<= max_pixels`` makes the server's own
     resize a no-op, ``smart_resize`` being idempotent on multiples of 28).
  2. The standard Qwen2.5-VL ``computer_use`` tool, defined in the **system
     prompt** ``<tools>`` block, declaring the **resized** resolution as the
     screen size (:func:`computer_use_tools_block`, verbatim from the official
     Qwen2.5-VL schema). The code agents get this block appended to their native
     ``<code>`` protocol prompt so they keep crop/zoom AND commit the same way.
  3. The commit is a ``computer_use`` ``<tool_call>`` with ``action=left_click``
     and an ABSOLUTE-pixel ``coordinate`` in the resized space.
  4. **Greedy** decoding (``temperature=0``).

The scorer (:func:`utils.gui_grounding.score_grounding_pyautogui`, same
``max_pixels``/``min_pixels``) extracts that ``[x, y]`` and maps it back to
original-image pixels — identical for base and code agents.

This module runs the BASE model (no inspection tools → a single turn that
directly emits the ``computer_use`` tool call, no prefill). The code agents run
their own loop in :mod:`utils.code_agent`, reusing :func:`computer_use_tools_block`
and :func:`grounding_click_instruction` so the protocol matches.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

from ..core.image import smart_resize
from ..core.llm import LLMClient

# Standard Qwen2.5-VL ``computer_use`` tool block (the ``# Tools`` ... section
# the chat template wraps inside the system turn). ``__SCREEN_W__`` /
# ``__SCREEN_H__`` are filled with the RESIZED dimensions, mirroring the official
# ``.replace("{{screen_width}}", str(resized_width))``. Kept as a raw literal
# (not json.dumps of a dict) so the schema text matches upstream byte-for-byte.
_TOOLS_BLOCK = (
    "# Tools\n"
    "\n"
    "You may call one or more functions to assist with the user query.\n"
    "\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn\'t open, try wait and taking another screenshot.\\n* The screen\'s resolution is __SCREEN_W__x__SCREEN_H__.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `right_click`: Click the right mouse button.\\n* `middle_click`: Click the middle mouse button.\\n* `double_click`: Double-click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": {"description": "Required only by `action=type`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}\n'
    "</tools>\n"
    "\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
    "</tool_call>"
)


def computer_use_tools_block(resized_width: int, resized_height: int) -> str:
    """The standard ``computer_use`` ``# Tools`` block with the RESIZED screen
    resolution filled in. Shared by the base model and the code agents so every
    Qwen2.5-VL model is given the exact same tool definition."""
    return (_TOOLS_BLOCK
            .replace("__SCREEN_W__", str(resized_width))
            .replace("__SCREEN_H__", str(resized_height)))


def build_system_prompt(resized_width: int, resized_height: int) -> str:
    """Base-model system prompt: assistant preamble + the computer_use tools."""
    return ("You are a helpful assistant.\n\n\n"
            + computer_use_tools_block(resized_width, resized_height))


def grounding_click_instruction(instruction: str) -> str:
    """Shared user-turn directive: click the described target via computer_use.
    Used identically by the base model and the code agents so the task framing
    is the same across the family."""
    return (
        "Locate the single UI element that best matches the instruction below and "
        "click it using the `computer_use` function with `action=left_click`. The "
        "coordinate is in absolute pixels of the screenshot shown.\n\n"
        f"Instruction: {instruction}"
    )


def _smart_resize_rgb(im, max_pixels: int, min_pixels: int):
    """Return ``(resized RGB image, resized_width, resized_height)`` for a PIL
    image, using the same ``smart_resize`` the scorer assumes."""
    from PIL import Image

    im = im.convert("RGB")
    rh, rw = smart_resize(im.height, im.width,
                          min_pixels=min_pixels, max_pixels=max_pixels)
    return im.resize((rw, rh), Image.BICUBIC), rw, rh


def _resize_and_encode(image_path: str, max_pixels: int, min_pixels: int
                       ) -> tuple[str, int, int]:
    """Resize ``image_path`` to ``smart_resize`` dims and return
    ``(data_uri, resized_width, resized_height)``. The data URI carries the
    resized image so the model sees exactly the declared resolution."""
    from PIL import Image

    with Image.open(image_path) as im:
        resized, rw, rh = _smart_resize_rgb(im, max_pixels, min_pixels)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}", rw, rh


def resize_grounding_images(image_paths: list[str], out_dir,
                            max_pixels: int, min_pixels: int) -> list[str]:
    """Save ``smart_resize``d copies of each image under ``out_dir/_resized/``
    and return their absolute paths.

    Used to feed the code-agent baselines (Thyme / DeepEyesV2) an input at
    exactly the resolution the model sees. They are told the image *path* and
    *size* and open the file in code, so a pre-resized input makes their
    ABSOLUTE-pixel click land in the same (resized) space the scorer maps back
    from — keeping scoring correct even for the ~5% of screenshots that the
    server's 12.8M cap would otherwise downscale."""
    from PIL import Image

    dst_dir = Path(out_dir) / "_resized"
    dst_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for p in image_paths:
        dst = dst_dir / (Path(p).stem + ".png")
        with Image.open(p) as im:
            resized, _rw, _rh = _smart_resize_rgb(im, max_pixels, min_pixels)
            resized.save(dst, format="PNG")
        out.append(str(dst.resolve()))
    return out


async def run_qwen25_grounding(question: str, image_path: str, llm: LLMClient,
                               max_pixels: int, min_pixels: int,
                               max_tokens: int = 512,
                               on_step=None) -> tuple[str, dict[str, Any]]:
    """Single-turn grounding for base Qwen2.5-VL (no inspection tools).

    The computer_use tool is defined in the system prompt; the model freely
    emits one ``<tool_call>`` (no prefill — strict protocol parity with the code
    agents). Returns ``(pred, meta)`` matching :func:`utils.agent.run`; ``pred``
    is the assistant message containing the ``coordinate": [x, y]`` the scorer
    parses.
    """
    t0 = time.time()
    data_uri, rw, rh = _resize_and_encode(image_path, max_pixels, min_pixels)
    system_prompt = build_system_prompt(rw, rh)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": grounding_click_instruction(question)},
        ]},
    ]

    resp = await llm.chat(
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</tool_call>"],  # stop right after the committing tool call
    )
    dump = resp.model_dump(exclude_none=True)
    usage = dump.get("usage") or {}
    choice = resp.choices[0]
    pred = choice.message.content or ""

    if on_step is not None:
        on_step(1, 0)

    turn = {
        "step": 0,
        "model": dump.get("model"),
        "finish_reason": choice.finish_reason,
        "message": {"role": "assistant", "content": pred},
        "usage": usage,
        "tool_results": [],
    }
    meta = {
        "input_messages": messages,
        "turns": [turn],
        "usage_total": {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        },
        "n_steps": 1,
        "n_tool_calls": 0,
        "stopped_reason": "final",
        "resized_size": [rw, rh],
        "elapsed_s": round(time.time() - t0, 2),
    }
    return pred, meta
