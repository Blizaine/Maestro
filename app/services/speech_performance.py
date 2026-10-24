"""Bounded Fish performance direction through insertion-only native markup.

The writer can suggest exact spans and delivery, but cannot replace the spoken
script. A failed suggestion leaves the available draft intact for review.
"""
from __future__ import annotations

import json
import re

from services.enhancement_diagnostics import record_advisory
from services.speech_writer import record_speech_writer_warning


PAIRED_TAGS = frozenset({"whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful"})
MAX_EDITS = 8
MAX_CANDIDATE_EDITS = 64
MAX_SOURCE_CHARS = 12_000
MAX_PERFORMANCE_TOKENS = 1_024
_CONTROL = re.compile(r"<[^>\r\n]*>|\[[^\]\r\n]*\]")
_SPEAKER = re.compile(r"(?im)^[ \t]*Speaker[ \t]*\d+[ \t]*:[ \t]*")
_PAIRED = re.compile(r"</?(?:whisper|emphasis|soft|fast|slow|stress|forceful)>", re.I)
_NO_DIRECTIONS = re.compile(
    r"\b(?:verbatim|untagged|unmarked)\b|"
    r"\b(?:plain|unmarked)\s+(?:speech|output|text|script|delivery)\b|"
    r"\b(?:no|without|do\s+not\s+add|don't\s+add)\s+"
    r"(?:any\s+)?(?:emotion\s+|performance\s+|delivery\s+)?(?:tags?|cues?|directions?|markup)\b",
    re.I,
)

_DIRECTOR_SYSTEM = """You direct a Fish Audio Drama 3 voice performance. The supplied script is already written.
Return only JSON: {"edits":[{"anchor":"exact unique words from the supplied script","tag":"emphasis","cue":"quietly uneasy"}]}.
An edit may contain tag, cue, or both. Allowed tags: whisper, emphasis, soft, fast, slow, stress, forceful.
The application adds paired tags around anchor and inserts a short [cue] immediately before it.
Choose a few meaningful emotional turns and important words/phrases inside the passage. Word/phrase tags
should target the actual words that need localized acting. Do not rely only on paragraph-opening descriptions.
Cues describe audible emotion, pace, intensity or tone, never gestures, camera direction, scenery, physical
actions or narrative events. Keep speaker identity, habitual pitch/register and accent stable; vary acting.
Every anchor must match an exact unique, contiguous span of spoken words, with original spelling and punctuation.
Do not anchor speaker labels, existing cues/tags, whole paragraphs, or overlapping spans. No replacement text,
new speech, omitted speech, commentary or structural tokens. Keep cues under 80 characters and anchors under
240 characters on one line. At most 8 edits; there is no quota. Use {"edits":[]} if no addition would help.
Honor requests for plain, verbatim or untagged speech in any language by returning an empty edits array.
Preserve all existing directions and every spoken word. The original brief supplies intent, not new script text."""


def needs_fish_direction(text: str, brief: str = "", instructions: str = "") -> bool:
    """Review substantial passages that missed native word-level control."""
    if _NO_DIRECTIONS.search(brief + "\n" + instructions):
        return False
    spoken = _SPEAKER.sub("", _CONTROL.sub(" ", text))
    return len(re.findall(r"\b\w+\b", spoken)) >= 60 and not _PAIRED.search(text)


def apply_performance_edits(text: str, response: str) -> tuple[str, list[str], int]:
    """Compile only exact, non-overlapping insertions; never rewrite speech."""
    value = response.strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4]
    try:
        data = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError("performance response was not complete JSON") from error
    if not isinstance(data, dict) or set(data) != {"edits"} or not isinstance(data["edits"], list):
        raise ValueError("performance response needs only an edits array")
    protected = [(m.start(), m.end()) for pattern in (_CONTROL, _SPEAKER) for m in pattern.finditer(text)]
    accepted = []
    reasons = []
    for index, edit in enumerate(data["edits"][:MAX_CANDIDATE_EDITS], 1):
        if len(accepted) >= MAX_EDITS:
            break
        reason = ""
        if not isinstance(edit, dict) or set(edit) - {"anchor", "tag", "cue"}:
            reason = "unsupported edit fields"
        else:
            anchor, tag, cue = edit.get("anchor"), edit.get("tag", ""), edit.get("cue", "")
            if (not isinstance(anchor, str) or not anchor.strip() or len(anchor) > 240
                    or any(c in anchor for c in "<>[]\r\n")):
                reason = "invalid speech anchor"
            elif not isinstance(tag, str) or (tag and tag not in PAIRED_TAGS):
                reason = "unsupported paired tag"
            elif (not isinstance(cue, str) or len(cue) > 80
                  or any(c in cue for c in "<>[]\r\n") or (not tag and not cue.strip())):
                reason = "invalid delivery cue"
            else:
                start = text.find(anchor)
                end = start + len(anchor)
                if start < 0 or text.find(anchor, start + 1) >= 0:
                    reason = "anchor was missing or ambiguous"
                elif ((anchor[0].isalnum() and start > 0 and text[start - 1].isalnum())
                      or (anchor[-1].isalnum() and end < len(text) and text[end].isalnum())):
                    reason = "anchor split a spoken word"
                elif any(start < b and end > a for a, b in protected):
                    reason = "anchor overlapped native syntax or another edit"
                else:
                    accepted.append((start, end, tag, cue.strip()))
                    protected.append((start, end))
        if reason:
            reasons.append(f"edit {index}: {reason}")

    result = text
    for start, end, tag, cue in sorted(accepted, reverse=True):
        span = text[start:end]
        if tag:
            span = f"<{tag}>{span}</{tag}>"
        if cue:
            span = f"[{cue}] " + span
        result = result[:start] + span + result[end:]
    record_advisory("fish_performance", {
        "suggested": len(data["edits"]), "applied": len(accepted),
        "application_limit": MAX_EDITS, "inspection_limit": MAX_CANDIDATE_EDITS,
        "skipped_reasons": reasons,
        "remaining_suggestions_omitted": max(0, len(data["edits"]) - MAX_CANDIDATE_EDITS)
            + max(0, min(len(data["edits"]), MAX_CANDIDATE_EDITS) - len(accepted) - len(reasons)),
    })
    return result, reasons, len(accepted)


def direct_fish_section(
    *, text: str, generate, remaining_calls, max_new_tokens: int,
    brief: str, instructions: str = "", section_incomplete: bool = False,
    **_context,
) -> str:
    """Spend at most one available shared-budget call on safe performance edits."""
    if section_incomplete or not needs_fish_direction(text, brief, instructions):
        return text
    if remaining_calls() <= 0 or len(text) > MAX_SOURCE_CHARS:
        record_speech_writer_warning("Fish performance review was skipped at its call/context bound; existing script directions were retained.")
        return text
    from services.studio_enhancement import check_cancelled
    check_cancelled()
    payload = json.dumps({"brief": brief[:8_000], "performance_preferences": instructions[:8_000], "script": text}, ensure_ascii=False)
    try:
        response, truncated = generate(
            prompt=payload, system_prompt=_DIRECTOR_SYSTEM,
            max_new_tokens=min(MAX_PERFORMANCE_TOKENS, max_new_tokens),
            temperature=min(0.35, float(_context.get("temperature", 0.35))),
            enable_thinking=False, thinking_budget=0,
        )
        if truncated:
            raise ValueError("performance response reached its token limit")
        result, reasons, count = apply_performance_edits(text, response)
        # Invalid/extra suggestions cannot change the script. A verified subset
        # is usable; keep its detailed receipt without creating a repair gate.
        if not count and reasons:
            record_speech_writer_warning("Fish performance review added no verified directions; the original script was retained.")
        return result
    except InterruptedError:
        raise
    except Exception as error:
        record_speech_writer_warning(f"Fish performance review could not apply safe directions ({error}); the original script was retained.")
        return text
