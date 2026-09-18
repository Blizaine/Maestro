"""Small deterministic estimates for filmable H3 action clocks.

This module measures visible physical beats. It does not judge prose quality or
rewrite a camera plan; callers use the estimate only as a relative clock weight.
"""
from __future__ import annotations

import re


_CAMERA_CLAUSE_RE = re.compile(
    r"(?:^|(?<=[.;]))\s*(?:the\s+)?camera\b[^.;]*[.;]?",
    flags=re.IGNORECASE,
)
_FULL_BODY_RE = re.compile(
    r"\b(?:approach(?:es|ed|ing)?|accelerat(?:e|es|ed|ing)|arriv(?:e|es|ed|ing)|"
    r"backflip(?:s|ped|ping)?|board(?:s|ed|ing)?|charg(?:e|es|ed|ing)|"
    r"climb(?:s|ed|ing)?|close(?:s|d|ing)?\s+(?:the\s+)?distance|cross(?:es|ed|ing)?|"
    r"descend(?:s|ed|ing)?|div(?:e|es|ed|ing)|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|"
    r"fall(?:s|ing)?|fl(?:y|ies|ew|ying)|follow(?:s|ed|ing)?|jump(?:s|ed|ing)?|"
    r"land(?:s|ed|ing)?|launch(?:es|ed|ing)?|leap(?:s|ed|ing)?|lunge(?:s|d|ing)?|"
    r"mount(?:s|ed|ing)?|move(?:s|d|ing)?|plummet(?:s|ed|ing)?|race(?:s|d|ing)?|"
    r"rid(?:e|es|ing)|rise|rises|rose|rising|run(?:s|ning)?|sprint(?:s|ed|ing)?|"
    r"step(?:s|ped|ping)?|tumbl(?:e|es|ed|ing)|walk(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
_BLOCKING_RE = re.compile(
    r"\b(?:block(?:s|ed|ing)?|brace(?:s|d|ing)?|catch(?:es|ing)?|caught|"
    r"counter(?:s|ed|ing)?|crouch(?:es|ed|ing)?|duck(?:s|ed|ing)?|grab(?:s|bed|bing)?|"
    r"hurl(?:s|ed|ing)?|kneel(?:s|ed|ing)?|(?:lie(?:s)?|lying)|lower(?:s|ed|ing)?|"
    r"opens|opened|opening\s+(?:the|a|an|her|his|their)\b|"
    r"parr(?:y|ies|ied|ying)|pick(?:s|ed|ing)?\s+up|"
    r"pivot(?:s|ed|ing)?|raise(?:s|d|ing)?|reach(?:es|ed|ing)?|redirect(?:s|ed|ing)?|"
    r"sit(?:s|ting)?(?:\s+down)?|spin(?:s|ning)?|stand(?:s|ing)?(?:\s+up)?|"
    r"sweep(?:s|ing)?|swept|throw(?:s|ing)?|threw|turn(?:s|ed|ing)?|"
    r"twist(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
_REACTION_RE = re.compile(
    r"\b(?:breath(?:e|es|ed|ing)|gasp(?:s|ed|ing)?|glance(?:s|d|ing)?|"
    r"laugh(?:s|ed|ing)?|look(?:s|ed|ing)?|nod(?:s|ded|ding)?|"
    r"recover(?:s|ed|ing)?|recoil(?:s|ed|ing)?|regain(?:s|ed|ing)?|"
    r"shrug(?:s|ged|ging)?|sigh(?:s|ed|ing)?|smil(?:e|es|ed|ing)|"
    r"stagger(?:s|ed|ing)?|stare(?:s|d|ing)?|struggl(?:e|es|ed|ing))\b",
    flags=re.IGNORECASE,
)
_IMPACT_RE = re.compile(
    r"\b(?:attack(?:s|ed|ing)?|break(?:s|ing)?|broke|collid(?:e|es|ed|ing)|"
    r"crash(?:es|ed|ing)?|crack(?:s|ed|ing)?|fight(?:s|ing)?|"
    r"flurr(?:y|ies)\s+of\s+(?:\w+\s+){0,2}strikes?|hit(?:s|ting)?|"
    r"impact(?:s|ed|ing)?|kick(?:s|ed|ing)?|palm\s+strikes?|punch(?:es|ed|ing)?|"
    r"save(?:s|d|ing)?|shatter(?:s|ed|ing)?|slam(?:s|med|ming)?|"
    r"smash(?:es|ed|ing)?|strik(?:e|es|ing)|struck|tackle(?:s|d|ing)?)\b",
    flags=re.IGNORECASE,
)


def _actor_action_in_camera_clause(match: re.Match) -> str:
    # Camera-led prose can contain a real performance: "camera follows Mara
    # as she crosses the room". Only discard the camera-owned prefix.
    parts = re.split(r"\b(?:as|while|when|who)\s+", match.group(0), maxsplit=1, flags=re.I)
    return " " + parts[1] if len(parts) > 1 else " "


def estimate_h3_action_seconds(text: str, *, event_floor: int = 1) -> float:
    """Estimate relative screen time from concrete visible action primitives."""

    action = _CAMERA_CLAUSE_RE.sub(_actor_action_in_camera_clause, str(text or ""))
    full_body = len(_FULL_BODY_RE.findall(action))
    blocking = len(_BLOCKING_RE.findall(action))
    reactions = len(_REACTION_RE.findall(action))
    impacts = len(_IMPACT_RE.findall(action))
    count = full_body + blocking + reactions + impacts
    if not count:
        return float(max(1, event_floor))
    estimate = (
        0.75
        + full_body * 2.0
        + blocking * 1.25
        + reactions * 0.9
        + impacts * 1.2
    )
    return max(float(max(1, event_floor)), min(6.5, estimate))


def is_h3_stationary_reaction(text: str) -> bool:
    """A facial reaction can be brief; travel, contact and timed holds cannot."""
    action = _CAMERA_CLAUSE_RE.sub(_actor_action_in_camera_clause, str(text or ""))
    return bool(
        (_REACTION_RE.search(action) or re.search(r'\b(?:expression|gaze|eye contact)\b', action, re.I))
        and not any(pattern.search(action) for pattern in (_FULL_BODY_RE, _BLOCKING_RE, _IMPACT_RE))
        and not re.search(
            r'\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|'
            r'several|a few|a couple of)[\s-]*(?:seconds?|secs?|s)\b', action, re.I,
        )
    )
