"""Scoring — turn an agent `pred` into a score.

- ``extractor.Extractor``         LLM answer extraction (chart / QA)
- ``judge.Judge``                 majority-vote YES/NO judge + Vision2Web judge
- ``gui_grounding``               geometric point-in-bbox grounding scorers
- ``vision2web``                  HTML render + component-fidelity scoring (0-100)
"""

from .extractor import Extractor, EXTRACT_PROMPT
from .judge import Judge, JUDGE_PROMPT, VISION2WEB_STATIC_JUDGE_PROMPT
from .gui_grounding import (
    is_grounding_item, score_grounding, score_grounding_pyautogui,
)
from .vision2web import (
    DEVICES,
    DEFAULT_VIEWPORTS,
    is_vision2web_item,
    is_vision2web_source,
    vision2web_viewports,
    vision2web_ref_images,
    judge_vision2web_viewports,
    extract_html,
    resolve_submitted_html,
    render_html_to_viewport_images,
)

__all__ = [
    "Extractor", "EXTRACT_PROMPT",
    "Judge", "JUDGE_PROMPT", "VISION2WEB_STATIC_JUDGE_PROMPT",
    "is_grounding_item", "score_grounding", "score_grounding_pyautogui",
    "DEVICES", "DEFAULT_VIEWPORTS",
    "is_vision2web_item", "is_vision2web_source",
    "vision2web_viewports", "vision2web_ref_images", "judge_vision2web_viewports",
    "extract_html", "resolve_submitted_html", "render_html_to_viewport_images",
]
