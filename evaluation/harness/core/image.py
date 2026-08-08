"""Qwen2.5-VL processor image geometry shared across the framework.

``smart_resize`` is the single source of truth for the resolution a Qwen2.5-VL
input is resized to. Both the grounding rollout drivers (which pre-resize the
screenshot and declare that resolution to the model) and the grounding scorer
(which maps an absolute-pixel click back to original-image pixels) depend on it,
so it lives in ``core`` to keep that dependency one-directional.

Defaults match vLLM's Qwen2.5-VL serving defaults.
"""

from __future__ import annotations

import math

IMAGE_FACTOR = 28
DEFAULT_MIN_PIXELS = 4 * 28 * 28
DEFAULT_MAX_PIXELS = 16384 * 28 * 28


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR,
                 min_pixels: int = DEFAULT_MIN_PIXELS,
                 max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[int, int]:
    """Qwen2.5-VL processor resize: returns (resized_height, resized_width),
    both multiples of `factor`, with total pixels within [min_pixels,
    max_pixels]. Mirrors the reference impl used in DeepEyesV2 inference."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
