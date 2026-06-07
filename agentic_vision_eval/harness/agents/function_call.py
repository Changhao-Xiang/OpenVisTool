"""Native function-calling agent loop (AsyncOpenAI + retry).

Trace shape: each agent turn carries the key fields from the LLM response
(model, finish_reason, message, usage) plus the tool results that came after.
The full conversation is reconstructible by walking turns.
"""

from __future__ import annotations

import json
from typing import Any

from harness.tools.registry import ToolRegistry

from ..core.llm import LLMClient
from ..core.path_mapper import VIRTUAL_ROOT, PathMapper
from .prompts import SYSTEM_PROMPT_WITH_TOOLS, SYSTEM_PROMPT_NO_TOOLS


def _safe_message_dump(msg: Any) -> dict[str, Any]:
    """model_dump an assistant message for replay."""
    return msg.model_dump(exclude_none=True)


async def run(prompt, llm: LLMClient, registry: ToolRegistry | None = None,
              max_steps: int = 50,
              sandbox_dir: str | None = None,
              path_mapper: PathMapper | None = None,
              on_step=None,
              system_prompt: str | None = None):
    """`on_step(rounds, n_tool_calls)` is called after each LLM turn finishes
    (after tool calls of that turn have executed). Used by the dashboard for
    live progress.

    `path_mapper`, when provided, virtualizes the sandbox to ``/mnt/data``:
    incoming tool args are translated virtual->real before execution, and
    tool results are translated real->virtual before reaching the model.

    `system_prompt`, when provided, overrides the default system prompt
    (used e.g. for GUI grounding evals where we want to inject a
    computer_use-format prompt into a --no-tools run).
    """
    """Returns (final_text, meta).

    meta = {
        "input_messages": [<system?>, <user>],
        "turns": [
            {"step": int, "model": ..., "finish_reason": ..., "message": ...,
             "usage": ...,
             "tool_results": [{"tool_call_id", "name", "args", "result"}]},
            ...
        ],
        "usage_total": {prompt_tokens, completion_tokens, total_tokens},
        "n_steps": int, "n_tool_calls": int,
        "stopped_reason": "final" | "empty_reply" | "budget",
    }

    No forced-final fallback: if the model returns empty content or runs out
    of step budget, pred is returned empty and the task is judged wrong.
    """
    tools_schema = registry.get_definitions() if registry and len(registry) > 0 else None

    # Build input messages (these are the seed; we also keep them in `messages`
    # which we mutate as the conversation grows for replay-on-LLM purposes).
    messages: list = []
    if system_prompt is not None:
        sys_msg = system_prompt
    elif registry:
        workspace_label = VIRTUAL_ROOT if path_mapper is not None else (sandbox_dir or "")
        sys_msg = SYSTEM_PROMPT_WITH_TOOLS.format(workspace_path=workspace_label)
    else:
        sys_msg = SYSTEM_PROMPT_NO_TOOLS
    messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": prompt})

    input_messages = [dict(m) for m in messages]   # snapshot for trace

    turns: list[dict[str, Any]] = []
    total_prompt = total_completion = total_total = 0

    def _accum(usage: dict | None) -> None:
        nonlocal total_prompt, total_completion, total_total
        usage = usage or {}
        total_prompt += usage.get("prompt_tokens", 0) or 0
        total_completion += usage.get("completion_tokens", 0) or 0
        total_total += usage.get("total_tokens", 0) or 0

    def _total_tools() -> int:
        return sum(len(t.get("tool_results", [])) for t in turns)

    for step in range(max_steps):
        resp = await llm.chat(messages=messages, tools=tools_schema)
        response_dump = resp.model_dump(exclude_none=True)
        _accum(response_dump.get("usage"))

        # Keep only the key fields from the LLM response. choices[0] is
        # always the relevant choice (we don't request n>1), so we flatten
        # finish_reason and message up to the turn level.
        choice = (response_dump.get("choices") or [{}])[0]
        cur_turn: dict[str, Any] = {
            "step": step,
            "model": response_dump.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "message": choice.get("message"),
            "usage": response_dump.get("usage"),
            "tool_results": [],
        }
        turns.append(cur_turn)

        msg = resp.choices[0].message
        messages.append(_safe_message_dump(msg))

        if not msg.tool_calls:
            content = msg.content or ""
            stopped_reason = "final" if content.strip() else "empty_reply"
            if on_step is not None:
                on_step(step + 1, _total_tools())
            return content, _meta(input_messages, turns, total_prompt,
                                  total_completion, total_total, stopped_reason)

        for call in msg.tool_calls:
            name = call.function.name
            raw_args = call.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                err = f"Error: tool arguments are not valid JSON: {e}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": err})
                cur_turn["tool_results"].append({
                    "tool_call_id": call.id, "name": name,
                    "args_raw": raw_args, "error": str(e), "result": err,
                })
                continue

            if registry is None or not registry.has(name):
                result = (f"Error: tool '{name}' not found. Available: "
                          f"{', '.join(registry.tool_names if registry else [])}")
            else:
                # Translate virtual paths in args -> real sandbox paths.
                exec_args = (path_mapper.rewrite_tool_args(name, args)
                             if path_mapper is not None else args)
                result = await registry.execute(name, exec_args)
                # Translate real sandbox paths in the result -> virtual paths.
                if path_mapper is not None:
                    result = path_mapper.rewrite_tool_result(result)

            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            cur_turn["tool_results"].append({
                "tool_call_id": call.id, "name": name,
                "args": args, "result": result,
            })

        # GUI grounding terminator: `computer_use` is a single-step click
        # call, not a step in a multi-turn agent loop. As soon as any tool
        # call this round was `computer_use`, finalize — there is no
        # follow-up to do, and looping would burn an extra LLM round to
        # produce an empty completion.
        #
        # `pred` is set to the FIRST computer_use call's arguments JSON,
        # so the grounding scorer (and the text fallback in
        # gui_grounding._TEXT_COORD_RE) can read the click point even
        # when downstream consumers only have the flat pred string.
        cu_call = next((c for c in msg.tool_calls
                        if c.function.name == "computer_use"), None)
        if cu_call is not None:
            content = cu_call.function.arguments or ""
            if on_step is not None:
                on_step(step + 1, _total_tools())
            return content, _meta(input_messages, turns, total_prompt,
                                  total_completion, total_total,
                                  "computer_use")

        # End of this round: notify dashboard with up-to-date round + tool count.
        if on_step is not None:
            on_step(step + 1, _total_tools())

    # Step budget exhausted.
    return "", _meta(input_messages, turns, total_prompt, total_completion,
                     total_total, "budget")


def _meta(input_messages, turns, p, c, t, reason):
    n_tool_calls = sum(len(t.get("tool_results", [])) for t in turns)
    return {
        "input_messages": input_messages,
        "turns": turns,
        "usage_total": {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": t,
        },
        "n_steps": len(turns),
        "n_tool_calls": n_tool_calls,
        "stopped_reason": reason,
    }
