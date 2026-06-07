"""YES/NO answer-equivalence judge with majority-vote across N samples,
plus a Vision2Web component-level visual-fidelity judge (0-100 score)."""

from __future__ import annotations

import asyncio
import json
import re

from ..core.config import Endpoint, RetryConfig
from ..core.io import encode_image
from ..core.llm import LLMClient


JUDGE_PROMPT = (
    "You are a strict answer equivalence judge. Decide whether PREDICTED correctly "
    "answers QUESTION when compared to GROUND TRUTH.\n\n"
    "HOW TO READ PREDICTED:\n"
    "- PREDICTED may contain long reasoning. Find the FINAL answer — usually the last "
    "  sentence, or text after markers like 'Answer:', 'Final answer:', '答案:', "
    "  '**Answer: X**', or a concluding number/label at the end.\n"
    "- IGNORE intermediate candidates discussed during reasoning. Only the final "
    "  committed answer matters.\n\n"
    "RULES:\n"
    "1. Ignore formatting, punctuation, capitalization, and minor wording differences "
    "   (e.g. '250,000' equals '250000', 'season 9' equals 'Season 9').\n"
    "2. Numeric answers must match EXACTLY after stripping commas/units. There is NO "
    "   tolerance — '250000' vs '125000' is NO; '250000' vs '250,000' is YES.\n"
    "3. The SET of items in the FINAL answer must EQUAL the set in GROUND TRUTH — no "
    "   extras, no missing.\n"
    "   - GT is a single value: PREDICTED must commit to exactly that value. Listing "
    "     it among other candidates is NOT a commitment and counts as NO.\n"
    "   - GT is a multi-label list (e.g. 'A, B'): PREDICTED must include ALL listed "
    "     items AND nothing else. Adding extras like 'A, B, C' is NO.\n"
    "4. If the FINAL answer hedges between multiple options without committing, return NO.\n"
    "5. If PREDICTED is empty, refuses, or has no extractable final conclusion, return NO.\n\n"
    "Examples:\n"
    "  GT='125000', FINAL='250,000 total households' → NO  (125000 ≠ 250000)\n"
    "  GT='2', FINAL='**Answer: 2**' → YES\n"
    "  GT='Season 9', FINAL='season 9' → YES\n"
    "  GT='Argentina', FINAL='Argentina, Brazil, Guyana, Suriname' → NO  (extras; "
    "did not commit to Argentina alone)\n"
    "  GT='A, B', FINAL='A and C' → NO  (B missing, C extra)\n"
    "  GT='A, B', FINAL='A, B, C' → NO  (C is extra)\n"
    "  GT='A, B', FINAL='A, B' → YES\n\n"
    "Respond with exactly ONE word: YES or NO."
)


VISION2WEB_STATIC_JUDGE_PROMPT = """You are a senior QA automation engineer for visual website development.

Compare two images:
1. Prototype image: the target webpage design for a specific device viewport.
2. Actual page image: a screenshot rendered from the submitted implementation.

Evaluate visual fidelity using logical UI components/blocks. Segment the page into meaningful blocks such as header/navigation, hero, content cards, product/list sections, forms, media areas, footer, and any other visually distinct functional sections. Do not make components too tiny.

For each component, assign one score from this set only: 0, 0.25, 0.5, 0.75, 1.

Scoring rubric:
- 1.0: Perfect match. Position, layout, spacing, alignment, size, text, fonts, colors, icons, images, and media match the prototype with no visible differences.
- 0.75: Minor imperfections. Mostly accurate with small alignment, spacing, typography, color, or media differences.
- 0.5: Partial match. Component is recognizable but has noticeable layout, spacing, content, typography, color, or media mismatches.
- 0.25: Poor match. Component is present but strongly misaligned, incomplete, or visually inconsistent.
- 0.0: No match. Component is missing, unrelated, or completely misplaced.

Focus on the current device viewport: {device}. Penalize responsive layout mistakes that are visible in this viewport. If the actual page is blank, broken, or mostly unrelated, all component scores should be 0.

Output only a JSON array. Each object must have:
[
  {{
    "name": "<component name>",
    "score": <0 | 0.25 | 0.5 | 0.75 | 1>,
    "reason": "<brief reason>"
  }}
]
"""


def _extract_json_array(raw: str) -> list | None:
    """Parse a JSON array from raw model output, tolerating code fences."""
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _parse_component_scores(raw: str) -> tuple[float | None, list[dict]]:
    """Return (mean_score_0_to_1, parsed_components)."""
    parsed = _extract_json_array(raw)
    if not parsed:
        return None, []

    components: list[dict] = []
    scores: list[float] = []
    valid_scores = {0.0, 0.25, 0.5, 0.75, 1.0}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if score not in valid_scores:
            score = min(valid_scores, key=lambda x: abs(x - score))
        component = {
            "name": str(item.get("name", "component")),
            "score": score,
            "reason": str(item.get("reason", "")),
        }
        components.append(component)
        scores.append(score)

    if not scores:
        return None, components
    return sum(scores) / len(scores), components


def _parse_yes_no(raw: str) -> bool | None:
    """Take the LAST standalone YES/NO token; return None if unparseable."""
    tokens = re.findall(r"\b(YES|NO)\b", raw.upper())
    if not tokens:
        return None
    return tokens[-1] == "YES"


class Judge:
    """Majority-vote judge.

    Calls the LLM `n_votes` times at `vote_temperature`, parses YES/NO,
    returns the majority verdict. Tied/unparseable votes default to NO
    (strict-judge bias).
    """

    def __init__(self, endpoint: Endpoint, retry: RetryConfig | None = None,
                 request_timeout_s: float = 90.0, max_connections: int = 32,
                 n_votes: int = 3, vote_temperature: float = 0.6):
        self.endpoint = endpoint
        self.n_votes = n_votes
        self.vote_temperature = vote_temperature
        self._llm = LLMClient(endpoint, retry=retry, label="judge",
                              request_timeout_s=request_timeout_s,
                              max_connections=max_connections)

    async def _one_vote(self, messages: list) -> tuple[bool | None, str]:
        resp = await self._llm.chat(messages=messages,
                                    temperature=self.vote_temperature)
        raw = (resp.choices[0].message.content or "").strip()
        return _parse_yes_no(raw), raw

    async def judge(self, question: str, ground_truth: str,
                    predicted: str) -> tuple[bool, str]:
        """Return (is_correct, raw_audit_string).

        raw_audit_string is "YES|NO|YES" style summary so it's easy to spot
        unstable judgments downstream.
        """
        user = (f"QUESTION: {question}\n"
                f"GROUND TRUTH: {ground_truth}\n"
                f"PREDICTED: {predicted}")
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user},
        ]

        votes = await asyncio.gather(
            *[self._one_vote(messages) for _ in range(self.n_votes)]
        )

        verdicts = [v for v, _ in votes]
        raws = [r for _, r in votes]
        yes_count = sum(1 for v in verdicts if v is True)
        no_count = sum(1 for v in verdicts if v is False)

        if yes_count > no_count:
            verdict = True
        else:
            # Strict bias: ties and majority-NO both → NO. Unparseable votes
            # are not counted as YES.
            verdict = False

        # Audit string: "YES|NO|YES" preserving order of votes.
        summary = "|".join(("YES" if v is True else "NO" if v is False else "?")
                           for v in verdicts)
        # Append first raw output for debugging (stripped to one line).
        head = (raws[0] or "").replace("\n", " ⏎ ")[:200]
        return verdict, f"{summary}  [first_raw={head!r}]"

    # ------------------------------------------------------------------
    # Vision2Web static webpage: component-level visual fidelity (0-100)
    # ------------------------------------------------------------------

    async def _one_vision2web_static_vote(
        self, messages: list,
    ) -> tuple[float | None, list[dict], str]:
        resp = await self._llm.chat(messages=messages,
                                    temperature=self.vote_temperature)
        raw = (resp.choices[0].message.content or "").strip()
        score, components = _parse_component_scores(raw)
        return score, components, raw

    async def judge_vision2web_static(
        self, ref_image_path: str, gen_image_path: str, device: str,
    ) -> tuple[float, str, list[dict]]:
        """Score one Vision2Web viewport.

        Returns (score_0_to_100, audit_string, components). Uses median across
        votes for the scalar score and keeps the components from the median vote.
        """
        ref_uri = encode_image(ref_image_path)
        gen_uri = encode_image(gen_image_path)

        messages = [
            {
                "role": "system",
                "content": VISION2WEB_STATIC_JUDGE_PROMPT.format(device=device),
            },
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": ref_uri}},
                {"type": "image_url", "image_url": {"url": gen_uri}},
                {"type": "text", "text": (
                    f"Compare the prototype and actual page for the {device} "
                    "viewport. Return only the JSON array."
                )},
            ]},
        ]

        votes = await asyncio.gather(
            *[self._one_vision2web_static_vote(messages) for _ in range(self.n_votes)]
        )

        valid = [(s, c, r) for s, c, r in votes if s is not None]
        if not valid:
            raws = [r for _, _, r in votes]
            head = (raws[0] or "").replace("\n", " ⏎ ")[:200]
            return 0.0, f"{device}: ?  [first_raw={head!r}]", []

        valid_sorted = sorted(valid, key=lambda x: x[0])
        median_score, components, median_raw = valid_sorted[len(valid_sorted) // 2]
        scores_str = "|".join(
            f"{s * 100:.0f}" if s is not None else "?" for s, _, _ in votes
        )
        head = (median_raw or "").replace("\n", " ⏎ ")[:200]
        return median_score * 100, f"{device}: {scores_str}  [median_raw={head!r}]", components
