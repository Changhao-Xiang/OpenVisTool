"""Computer-use tool (Qwen3-VL GUI grounding format).

This tool is a **format-only terminator**: it does not drive any real GUI. Its
sole purpose is to let a teacher model emit a final `computer_use` tool_call
(click coordinate / answer text / terminate status) in the Qwen3-VL schema as
the final output for a GUI grounding query. Coordinates stay in the
normalized 0-1000 space defined by the original Qwen3-VL prompt.

The agent loop (`loop.AgentLoop._run_agent_loop`) special-cases
this tool: when the model emits a `computer_use` tool_call, the loop stops
immediately without executing tools or appending any tool_result. The
assistant message carrying the tool_call itself is what gets persisted as
the final turn output.

Because the loop short-circuits, `execute()` here should never actually run
during normal rollouts. It is implemented defensively so that a stray call
(e.g. from a provider that does not go through the loop short-circuit path)
returns a harmless no-op string rather than raising.
"""

from __future__ import annotations

from typing import Any

from harness.tools.base import Tool


class ComputerUseTool(Tool):
    """Format-only terminator tool mirroring the Qwen3-VL computer_use schema."""

    @property
    def name(self) -> str:
        return "computer_use"

    @property
    def description(self) -> str:
        return (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n"
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n"
            "* The screen's resolution is 1000x1000.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n"
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
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
        }

    async def execute(self, **kwargs: Any) -> str:
        """No-op fallback. The agent loop short-circuits before reaching here."""
        return "[computer_use acknowledged as final output]"
