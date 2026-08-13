"""Window-local prompt planning for Maestro's LTX long-form sequences.

LTX rolling generation has always accepted one prompt per sliding window, but
the old Studio UI exposed that behavior implicitly: extending Duration could
silently create more windows and a multiline prompt was split without an
explicit contract.  This module gives every LTX generation the same durable
contract as H3: compute the exact window geometry, validate manual prompts,
or expand one story idea into one chronological prompt per native pass.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence


def compute_ltx_window_count(
    total_frames: int,
    window_frames: int,
    *,
    overlap_frames: int = 0,
    discard_frames: int = 0,
) -> int:
    """Match WanGP's rolling-window count without importing its GPU runtime."""

    total = max(1, int(total_frames or 0))
    window = max(1, int(window_frames or 0))
    overlap = max(0, int(overlap_frames or 0))
    discard = max(0, int(discard_frames or 0))
    if total <= window:
        return 1
    stride = window - discard - overlap
    if stride <= 0:
        raise ValueError(
            "LTX Window Length must be greater than its overlap and discarded tail."
        )
    return 1 + math.ceil((total - window + discard) / stride)


def _collapse_prompt(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def parse_ltx_window_prompts(
    value: str | Sequence[str] | None,
    *,
    expected_count: int,
) -> list[str]:
    """Return exactly one non-empty, single-line prompt per LTX window.

    The model guide asks the LLM for blank-line-separated paragraphs while
    Manual mode asks the user for one physical line per window.  Accept both
    forms, plus a list returned by a previous server-side plan.
    """

    count = max(1, int(expected_count or 1))
    if isinstance(value, (list, tuple)):
        prompts = [_collapse_prompt(item) for item in value if _collapse_prompt(item)]
    else:
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        blocks = [
            _collapse_prompt(block)
            for block in re.split(r"\n\s*\n+", text)
            if _collapse_prompt(block)
        ]
        if len(blocks) == count:
            prompts = blocks
        else:
            prompts = [
                _collapse_prompt(line)
                for line in text.split("\n")
                if _collapse_prompt(line)
            ]

    if len(prompts) != count:
        raise ValueError(
            f"LTX multi-window sequence needs exactly {count} non-empty prompt "
            f"{'line' if count == 1 else 'lines'} (window 1 through window {count}); "
            f"found {len(prompts)}."
        )
    return prompts


def _split_story_sentences(prompt: str) -> list[str]:
    normalized = _collapse_prompt(prompt)
    if not normalized:
        return []
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", normalized)
        if part.strip()
    ]


def deterministic_ltx_window_prompts(prompt: str, count: int) -> list[str]:
    """Usable no-LLM fallback that advances rather than restarting the story."""

    count = max(1, int(count or 1))
    source = _collapse_prompt(prompt)
    sentences = _split_story_sentences(source)
    prompts: list[str] = []
    for index in range(count):
        start = math.floor(index * len(sentences) / count) if sentences else 0
        end = math.floor((index + 1) * len(sentences) / count) if sentences else 0
        local = " ".join(sentences[start:end]).strip()
        if not local:
            local = source
        if index == 0:
            phase = "Establish the requested characters, setting, and action, then begin this beat."
        elif index == count - 1:
            phase = "Continue directly from the preceding window without restarting, then complete the requested final beat."
        else:
            phase = "Continue directly from the preceding window without restarting and advance only this middle beat."
        prompts.append(
            _collapse_prompt(
                f"{phase} {local} Preserve the same identities, wardrobe, location, "
                "lighting, screen direction, and visible continuity."
            )
        )
    return prompts


def plan_ltx_sliding_windows(
    prompt: str,
    *,
    model_type: str,
    duration_seconds: float,
    window_count: int,
    window_size_seconds: float,
    image_paths: Iterable[str] | None = None,
    nsfw: bool = False,
) -> dict:
    """Use Maestro's configured enhancer to create exact LTX window prompts.

    Planning deliberately uses the ordinary model-specific LTX guide so LTX
    0.9, 2.x, and 2.5 retain their established prompt vocabulary.  Geometry is
    supplied explicitly and malformed output falls back to a deterministic
    chronological plan instead of failing an expensive generation request.
    """

    count = max(1, int(window_count or 1))
    source = str(prompt or "").strip()
    planned_by = "llm"
    error = None
    try:
        from services import llm_service

        enhanced = llm_service.enhance_prompt(
            source,
            mode="video",
            max_new_tokens=max(512, count * 320),
            temperature=0.45,
            nsfw=bool(nsfw),
            model_type=str(model_type or ""),
            image_paths=list(image_paths or []) or None,
            duration_seconds=max(1, round(float(duration_seconds), 1)),
            window_count=count,
            window_size_seconds=max(1, round(float(window_size_seconds), 1)),
        )
        prompts = parse_ltx_window_prompts(enhanced, expected_count=count)
    except Exception as exc:  # LLM is optional; generation itself is not.
        error = str(exc)
        planned_by = "deterministic_fallback"
        prompts = deterministic_ltx_window_prompts(source, count)

    return {
        "source_prompt": source,
        "window_count": count,
        "window_prompts": prompts,
        "planned_by": planned_by,
        "planning_error": error,
    }
