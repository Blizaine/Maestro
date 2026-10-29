"""One-shot natural-language planning for animated title cards."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Callable


MAX_BRIEF_CHARS = 4000
MAX_STYLE_HINT_CHARS = 1600
MAX_DRAFT_TOKENS = 1800
MAX_RAW_DRAFT_CHARS = 40000
MAX_TITLE_CHARS = 160
MAX_SEQUENCE_PHRASES = 6
MAX_LOOK_CHARS = 1200
MAX_DIRECTION_CHARS = 1200
MAX_ANIMATION_CHARS = 800
NATIVE_FRAME_LENGTHS = tuple(range(124, 346, 17))
MIN_CLIP_DURATION_SECONDS = 5
MAX_CLIP_DURATION_SECONDS = 15
MAX_CLIP_FRAMES = 120 * MAX_CLIP_DURATION_SECONDS
_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}\Z")

TITLE_PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "phrases": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_SEQUENCE_PHRASES,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "animation": {"type": "string"},
                    "weight": {"type": "number"},
                },
                "required": ["text", "animation", "weight"],
                "additionalProperties": False,
            },
        },
        "look": {"type": "string"},
        "direction": {"type": "string"},
        "sequence": {"type": "boolean"},
    },
    "required": ["phrases", "look", "direction", "sequence"],
    "additionalProperties": False,
}

_QUOTED_COPY = re.compile(
    r"“([^”]+)”|‘([^’]+)’|\"([^\"\n]+)\"|(?<!\w)'([^'\n]+)'"
)


def validate_title_request(
    brief: object,
    style_hint: object,
    frames: object,
    background: object,
    *,
    fps: object = 24,
    native_h3: object = True,
) -> None:
    """Reject invalid request values before loading or calling the writer."""
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("brief must be a non-empty string.")
    if len(brief) > MAX_BRIEF_CHARS:
        raise ValueError(f"brief must be at most {MAX_BRIEF_CHARS} characters.")
    if not isinstance(style_hint, str):
        raise ValueError("style_hint must be a string.")
    if len(style_hint) > MAX_STYLE_HINT_CHARS:
        raise ValueError(f"style_hint must be at most {MAX_STYLE_HINT_CHARS} characters.")
    try:
        finite_fps = type(fps) in (int, float) and math.isfinite(fps)
    except OverflowError:
        finite_fps = False
    if not finite_fps or not 1 <= fps <= 120:
        raise ValueError("fps must be a finite number from 1 to 120.")
    if type(native_h3) is not bool:
        raise ValueError("native_h3 must be a boolean.")
    if type(frames) is not int or not 1 <= frames <= MAX_CLIP_FRAMES:
        raise ValueError(f"frames must be a positive integer no greater than {MAX_CLIP_FRAMES}.")
    duration_seconds = frames / fps
    if not MIN_CLIP_DURATION_SECONDS <= duration_seconds <= MAX_CLIP_DURATION_SECONDS:
        raise ValueError("clip duration must be between 5 and 15 seconds.")
    if native_h3 and frames not in NATIVE_FRAME_LENGTHS:
        raise ValueError("frames must be a native title duration from 124 to 345 in steps of 17.")
    if not (
        background in ("chroma", "black", "blue", "white")
        or isinstance(background, str) and _HEX_COLOR.fullmatch(background)
    ):
        raise ValueError("background must be 'chroma', 'black', 'blue', 'white', or a #RRGGBB hex color.")


def _guide_text() -> str:
    guide_path = (
        Path(__file__).resolve().parent
        / "llm_guides"
        / "creative"
        / "animated_titles.md"
    )
    return guide_path.read_text(encoding="utf-8").strip()


def _writer_prompt(
    brief: str,
    style_hint: str,
    frames: int,
    background: str,
    *,
    fps: int | float = 24,
    native_h3: bool = True,
) -> str:
    style_text = style_hint if style_hint else "No additional style hint."
    background_text = {
        "chroma": "solid chroma green",
        "black": "solid black",
        "blue": "solid blue",
        "white": "solid white",
    }.get(background)
    if background_text is None:
        background_text = f"solid custom color {background}"
    duration_seconds = frames / fps
    duration_context = "Native H3 duration" if native_h3 else "Selected-model duration"
    return (
        "Plan an animated title from the user's request. The application owns the "
        "duration and background; treat these values as fixed context and do not "
        "return or change them.\n"
        f"Frame rate: {fps:g} fps. {duration_context}: {frames} frames ({duration_seconds:.3f} seconds).\n"
        f"Background: {background_text}.\n"
        f"Style hint for the current phrase(s):\n{style_text}\n\n"
        "User brief (preserve its language, exact quoted copy, and requested order):\n"
        f"<brief>\n{brief}\n</brief>"
    )


def _strip_json_fence(raw_draft: str) -> str:
    candidate = raw_draft.strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline >= 0:
            candidate = candidate[first_newline + 1 :]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
    return candidate.strip()


def _invalid_result(raw_draft: str, warning: str) -> dict:
    return {
        "plan": None,
        "warnings": [warning],
        "raw_draft": raw_draft,
        "review_required": True,
    }


def _first_text(source: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str):
            return value
    return None


def _phrase_source(source: dict):
    for key in ("phrases", "items", "beats"):
        value = source.get(key)
        if isinstance(value, list):
            return value
    beat = source.get("beat")
    if isinstance(beat, list):
        return beat
    text = _first_text(source, ("text", "title", "beat", "copy", "words"))
    if text is not None:
        return [source]
    return None


def _scoped_motion_fallback(style_hint: str, sequence: bool) -> str:
    hint = style_hint.strip()
    if not hint:
        return "Reveal clearly and keep the words readable."
    if sequence:
        hint = re.sub(
            r"\bthe\s+(?:(?:complete|entire|full)\s+)?(?:title|lettering|wording)\b",
            "this phrase",
            hint,
            flags=re.IGNORECASE,
        )
        hint = re.sub(
            r"\b(?:complete|entire|full)\s+(?:title|lettering|wording)\b",
            "this phrase",
            hint,
            flags=re.IGNORECASE,
        )
        hint = re.sub(r"\btitle\b", "phrase", hint, flags=re.IGNORECASE)
        fallback = f"Apply only to the current phrase; keep other sequence phrases out of view: {hint}"
    else:
        fallback = f"Apply this motion to the title phrase: {hint}"
    return fallback if len(fallback) <= MAX_ANIMATION_CHARS else "Reveal clearly and keep the words readable."


def _normalize_phrase(
    value: object,
    index: int,
    *,
    style_hint: str,
    sequence: bool,
) -> tuple[dict | None, list[str]]:
    warnings = []
    if isinstance(value, str):
        source = {"text": value}
    elif isinstance(value, dict):
        source = value
    else:
        return None, [f"Phrase {index} was missing usable title text."]

    text = _first_text(source, ("text", "title", "beat", "copy", "words"))
    if text is None or not text.strip():
        return None, [f"Phrase {index} was missing usable title text."]

    animation = _first_text(source, ("animation", "motion", "reveal"))
    if not animation or not animation.strip():
        animation = _scoped_motion_fallback(style_hint, sequence)
    if len(animation) > MAX_ANIMATION_CHARS:
        return None, [f"Phrase {index} animation exceeded {MAX_ANIMATION_CHARS} characters; no text was truncated."]
    raw_weight = source.get("weight", 1.0)
    try:
        finite_weight = math.isfinite(raw_weight)
    except (OverflowError, TypeError):
        finite_weight = False
    if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)) or not finite_weight or raw_weight <= 0:
        warnings.append(f"Phrase {index} had no usable positive finite weight; the default weight was used.")
        weight = 1.0
    else:
        weight = raw_weight
    return {"text": text, "animation": animation, "weight": weight}, warnings


def _quoted_copy(brief: str) -> list[str]:
    return [next(group for group in match.groups() if group is not None) for match in _QUOTED_COPY.finditer(brief)]


def _copy_warning(brief: str, phrases: list[dict]) -> bool:
    quoted = _quoted_copy(brief)
    if not quoted:
        return False
    def normalize_whitespace(value):
        return " ".join(value.split())
    combined = normalize_whitespace(" ".join(phrase["text"] for phrase in phrases))
    expected = normalize_whitespace(" ".join(quoted))
    return combined != expected


def normalize_title_draft(
    raw_draft: object,
    *,
    brief: str,
    style_hint: str = "",
) -> dict:
    """Turn one writer response into a bounded, reviewable title plan."""
    if isinstance(raw_draft, str):
        raw = raw_draft
    else:
        try:
            raw = json.dumps(raw_draft, ensure_ascii=False)
        except (TypeError, ValueError):
            raw = str(raw_draft)

    if len(raw) > MAX_RAW_DRAFT_CHARS:
        return _invalid_result(raw, "The title writer draft was oversized; the complete raw draft was preserved for review.")

    try:
        parsed = json.loads(_strip_json_fence(raw))
    except (json.JSONDecodeError, TypeError):
        return _invalid_result(raw, "The title writer did not return complete JSON. Review the raw draft before using it.")
    if not isinstance(parsed, dict):
        return _invalid_result(raw, "The title writer response was not a JSON object. Review the raw draft before using it.")

    source = parsed.get("plan") if isinstance(parsed.get("plan"), dict) else parsed
    phrase_values = _phrase_source(source)
    if not isinstance(phrase_values, list) or not phrase_values:
        return _invalid_result(raw, "The title writer draft did not include any usable title phrases.")
    if len(phrase_values) > MAX_SEQUENCE_PHRASES:
        return _invalid_result(raw, f"The title writer draft had more than {MAX_SEQUENCE_PHRASES} phrases; no text was truncated.")

    phrases = []
    warnings = []
    sequence = len(phrase_values) > 1
    for index, value in enumerate(phrase_values, start=1):
        phrase, phrase_warnings = _normalize_phrase(
            value,
            index,
            style_hint=style_hint,
            sequence=sequence,
        )
        if phrase is None:
            return _invalid_result(raw, phrase_warnings[0])
        phrases.append(phrase)
        warnings.extend(phrase_warnings)

    explicit_sequence = source.get("sequence", source.get("is_sequence"))
    sequence = len(phrases) > 1
    if isinstance(explicit_sequence, bool) and explicit_sequence != sequence:
        warnings.append(
            "The draft's sequence flag did not match its phrase count; phrases were kept in order and sequencing was inferred. Review the plan."
        )

    if sequence:
        if len(phrases) < 2:
            return _invalid_result(raw, "A title sequence needs at least two phrases.")
        if any(len(phrase["text"]) > 80 for phrase in phrases):
            return _invalid_result(raw, "A sequence phrase exceeded 80 characters; no title text was truncated.")
        displayed_text = "\n".join(phrase["text"].strip() for phrase in phrases)
        if len(displayed_text) > MAX_TITLE_CHARS:
            return _invalid_result(raw, "The sequence exceeded 160 total title characters; no title text was truncated.")
    elif len(phrases[0]["text"]) > MAX_TITLE_CHARS:
        return _invalid_result(raw, "The title phrase exceeded 160 characters; no title text was truncated.")

    look = _first_text(source, ("look", "visual_style"))
    if not look or not look.strip():
        look = "Polished, legible animated typography."
    direction = _first_text(source, ("direction", "composition"))
    if not direction or not direction.strip():
        direction = "Keep the title legible and inside the frame."
    if len(look) > MAX_LOOK_CHARS:
        return _invalid_result(raw, f"The title look exceeded {MAX_LOOK_CHARS} characters; no text was truncated.")
    if len(direction) > MAX_DIRECTION_CHARS:
        return _invalid_result(raw, f"The title direction exceeded {MAX_DIRECTION_CHARS} characters; no text was truncated.")

    model_warnings = parsed.get("warnings")
    if not isinstance(model_warnings, list):
        model_warnings = source.get("warnings")
    if isinstance(model_warnings, list):
        warnings.extend(item for item in model_warnings if isinstance(item, str) and item.strip())

    review_required = parsed.get("review_required") is True or source.get("review_required") is True
    if _copy_warning(brief, phrases):
        warnings.append("The draft did not preserve one or more quoted phrases exactly and in order. Review its wording before using it.")
        review_required = True
    warnings = list(dict.fromkeys(warnings))
    if warnings:
        review_required = review_required or any(
            "quoted phrases" in warning or "review" in warning.lower()
            for warning in warnings
        )

    if review_required and not warnings:
        warnings.append("The title writer marked this draft for wording review.")

    return {
        "plan": {
            "phrases": phrases,
            "look": look,
            "direction": direction,
            "sequence": sequence,
        },
        "warnings": warnings,
        "raw_draft": raw,
        "review_required": review_required,
    }


def plan_animated_title(
    brief: str,
    style_hint: str,
    frames: int,
    background: str,
    writer: Callable,
    *,
    json_schema: dict | None = TITLE_PLAN_JSON_SCHEMA,
    fps: int | float = 24,
    native_h3: bool = True,
) -> dict:
    """Validate the request, make exactly one bounded writer call, and parse it."""
    validate_title_request(
        brief,
        style_hint,
        frames,
        background,
        fps=fps,
        native_h3=native_h3,
    )
    kwargs = {
        "prompt": _writer_prompt(
            brief,
            style_hint,
            frames,
            background,
            fps=fps,
            native_h3=native_h3,
        ),
        "system_prompt": _guide_text(),
        "max_new_tokens": MAX_DRAFT_TOKENS,
        "temperature": 0.45,
        "top_p": 0.9,
        "enable_thinking": False,
        "thinking_budget": 0,
    }
    if json_schema is not None:
        kwargs["json_schema"] = json_schema
    raw_draft = writer(**kwargs)
    return normalize_title_draft(raw_draft, brief=brief, style_hint=style_hint)
