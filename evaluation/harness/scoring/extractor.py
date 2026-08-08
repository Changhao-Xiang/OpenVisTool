"""Extract the final answer from a (possibly verbose) agent response.

Runs BEFORE the judge. The judge then scores the clean extracted answer
against ground truth, which is much more reliable than judging raw
chain-of-thought responses.
"""

from __future__ import annotations

import re

from ..core.config import Endpoint, RetryConfig
from ..core.llm import LLMClient

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from a model response.

    Closed blocks are dropped entirely. A trailing unclosed <think> (the model
    was still reasoning when it ran out of tokens) is also dropped — there is
    no committed answer inside it.
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    return cleaned.strip()


EXTRACT_PROMPT = (
    "You receive a QUESTION and a MODEL RESPONSE that may contain long "
    "reasoning, tool output, multiple candidates, or formatting noise.\n"
    "Your job: output ONLY the final answer to the question, as a short "
    "standalone phrase or value.\n\n"
    "Guidance:\n"
    "- If the response has an explicit final answer (e.g. after 'Answer:', "
    "  'Final answer:', '**Answer: X**', '答案:', or as the last sentence), "
    "  extract exactly that.\n"
    "- If the response lists candidates but does NOT commit to one, output NONE.\n"
    "- If the response is empty, refuses, or contains no answer to the "
    "  question, output NONE.\n"
    "- Do NOT add explanations, units the source did not use, or punctuation "
    "  beyond what the source committed.\n"
    "- Keep the form the model committed to (a number, a name, a list, etc.). "
    "  For lists, preserve the model's separator.\n\n"
    "Respond with the extracted answer on a single line, nothing else."
)


class Extractor:
    def __init__(self, endpoint: Endpoint, retry: RetryConfig | None = None,
                 request_timeout_s: float = 90.0, max_connections: int = 32):
        self.endpoint = endpoint
        self._llm = LLMClient(endpoint, retry=retry, label="extract",
                              request_timeout_s=request_timeout_s,
                              max_connections=max_connections)

    async def extract(self, question: str, predicted: str) -> str:
        """Return the extracted final answer (may be the literal string 'NONE')."""
        if not (predicted or "").strip():
            return "NONE"
        predicted = strip_think_blocks(predicted)
        if not predicted:
            return "NONE"
        user = f"QUESTION: {question}\n\nMODEL RESPONSE:\n{predicted}"
        resp = await self._llm.chat(
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        # Collapse accidental multi-line outputs to the first non-empty line.
        for line in out.splitlines():
            line = line.strip()
            if line:
                return line
        return "NONE"
