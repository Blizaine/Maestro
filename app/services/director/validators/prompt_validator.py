"""
Prompt Validator — validates rendered prompts against mode-specific rules.

Runs after renderers produce final prompts. Checks for anti-patterns,
mode violations, and completeness.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from ..schema import ShotPlan, ProductionPlan
from ..policies import (
    detect_anti_patterns, detect_character_names_in_prompt,
    MONTAGE_LANGUAGE, IMAGE_META_LANGUAGE, IMAGE_ACTION_VERBS,
)


@dataclass
class PromptValidationResult:
    valid: bool
    mode: str
    warnings: list[str]
    completeness_score: float  # 0.0 to 1.0
    auto_fixes_applied: list[str]


def validate_prompt_for_mode(
    prompt: str,
    mode: str,
    shot: ShotPlan,
    plan: Optional[ProductionPlan] = None,
) -> PromptValidationResult:
    """Validate a rendered prompt against mode-specific rules.

    Modes: t2v, i2v, a2v, retake, extend, image_gen
    """
    warnings = []
    auto_fixes = []

    # ── Universal checks ─────────────────────────────────────────

    # Anti-patterns
    warnings.extend(detect_anti_patterns(prompt, mode))

    # Character name leaks
    if plan and plan.characters:
        warnings.extend(detect_character_names_in_prompt(prompt, plan.characters))

    # Present tense check (video modes)
    if mode in ("t2v", "i2v", "a2v", "retake", "extend"):
        past_tense_markers = re.findall(r'\b(walked|ran|said|looked|turned|moved|jumped|danced)\b', prompt, re.IGNORECASE)
        if past_tense_markers:
            warnings.append(f"Past tense detected: {', '.join(past_tense_markers[:3])} — use present tense")

    # Single paragraph check (video modes)
    if mode in ("t2v", "i2v", "a2v", "retake", "extend"):
        paragraphs = [p.strip() for p in prompt.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            warnings.append(f"Multiple paragraphs detected ({len(paragraphs)}) — should be single paragraph")

    # ── Mode-specific checks ─────────────────────────────────────

    if mode == "t2v":
        warnings.extend(_check_t2v(prompt, shot))

    elif mode == "i2v":
        warnings.extend(_check_i2v(prompt, shot))

    elif mode == "a2v":
        warnings.extend(_check_a2v(prompt, shot))

    elif mode == "image_gen":
        warnings.extend(_check_image_gen(prompt, shot))

    elif mode == "extend":
        warnings.extend(_check_extend(prompt, shot))

    # Score completeness
    score = score_prompt_completeness(prompt, mode, shot)

    is_valid = not any("ERROR" in w for w in warnings)
    return PromptValidationResult(
        valid=is_valid,
        mode=mode,
        warnings=warnings,
        completeness_score=score,
        auto_fixes_applied=auto_fixes,
    )


def _check_t2v(prompt: str, shot: ShotPlan) -> list[str]:
    """T2V-specific checks — most descriptive mode."""
    warnings = []
    word_count = len(prompt.split())

    if word_count < 20:
        warnings.append(f"T2V prompt is very short ({word_count} words) — this mode should be the most descriptive")

    if shot.duration_sec >= 10 and word_count < 30:
        warnings.append(f"T2V prompt may be too brief for {shot.duration_sec}s shot — aim for more detail")

    # Check for camera language
    camera_terms = ["camera", "shot", "dolly", "tracking", "pan", "orbit", "push", "pull", "zoom", "handheld", "steady"]
    if not any(term in prompt.lower() for term in camera_terms):
        warnings.append("No explicit camera language in T2V prompt")

    return warnings


def _check_i2v(prompt: str, shot: ShotPlan) -> list[str]:
    """I2V-specific checks — should be motion-focused, not re-descriptive."""
    warnings = []

    # I2V shouldn't be as long as T2V
    word_count = len(prompt.split())
    if word_count > 100:
        warnings.append(f"I2V prompt may be too verbose ({word_count} words) — focus on what CHANGES, not static description")

    # Should contain motion/change language
    motion_terms = ["moves", "begins", "shifts", "turns", "starts", "changes", "walks", "runs",
                    "gestures", "reaches", "leans", "steps", "dances"]
    if not any(term in prompt.lower() for term in motion_terms):
        warnings.append("I2V prompt lacks motion language — should describe what changes from the start image")

    return warnings


def _check_a2v(prompt: str, shot: ShotPlan) -> list[str]:
    """A2V-specific checks — should be lean, audio-aware."""
    warnings = []
    word_count = len(prompt.split())

    if word_count > 80:
        warnings.append(f"A2V prompt may be too dense ({word_count} words) — keep lean to avoid fighting audio timing")

    # If lip sync critical, dialogue should be present
    if shot.audio_plan.lip_sync_critical and shot.dialogue_beats:
        has_quotes = '"' in prompt
        if not has_quotes:
            warnings.append("Lip-sync-critical shot but no quoted dialogue found in prompt")

    return warnings


def _check_image_gen(prompt: str, shot: ShotPlan) -> list[str]:
    """Image prompt checks — no motion, no meta-language."""
    warnings = []

    # Check for motion verbs
    prompt_lower = prompt.lower()
    for verb in IMAGE_ACTION_VERBS:
        if verb in prompt_lower:
            warnings.append(f"Motion verb in image prompt: '{verb}' — image prompts describe static frames")

    # Check for meta-language (if reference-based)
    if shot.image_strategy in ("reference_edit", "reference_inspired"):
        for phrase in IMAGE_META_LANGUAGE:
            if phrase in prompt_lower:
                warnings.append(f"Meta-language in image prompt: '{phrase}' — use action verbs instead")

    return warnings


def _check_extend(prompt: str, shot: ShotPlan) -> list[str]:
    """Extend-specific checks — should continue, not reset."""
    warnings = []
    prompt_lower = prompt.lower()

    # Should not feel like a complete scene reset
    reset_indicators = ["establishing shot", "we open on", "the scene begins", "opening shot"]
    for indicator in reset_indicators:
        if indicator in prompt_lower:
            warnings.append(f"Extend prompt uses reset language: '{indicator}' — should continue from prior clip")

    return warnings


def detect_prompt_anti_patterns(prompt: str, mode: str) -> list[str]:
    """Convenience wrapper for anti-pattern detection."""
    return detect_anti_patterns(prompt, mode)


def score_prompt_completeness(prompt: str, mode: str, shot: ShotPlan) -> float:
    """Score how complete a prompt is for its mode (0.0 to 1.0).

    Checks presence of key elements expected for each mode.
    """
    if not prompt:
        return 0.0

    checks = []
    prompt_lower = prompt.lower()

    # Universal checks
    checks.append(len(prompt.split()) >= 10)  # minimum length
    checks.append(bool(prompt.strip()))  # non-empty

    if mode in ("t2v", "i2v", "a2v", "retake", "extend"):
        # Video checks
        camera_terms = ["camera", "shot", "dolly", "tracking", "pan", "orbit",
                        "push", "pull", "zoom", "handheld", "steady", "close-up",
                        "medium", "wide", "framing"]
        checks.append(any(t in prompt_lower for t in camera_terms))  # camera language
        checks.append("." in prompt)  # has sentence structure
        checks.append(not any(m in prompt_lower for m in MONTAGE_LANGUAGE))  # no montage

        if shot.dialogue_beats:
            checks.append('"' in prompt)  # dialogue in quotes

        if mode == "t2v":
            # T2V should be most complete
            checks.append(len(prompt.split()) >= 25)  # decent length
            setting_terms = ["light", "room", "street", "studio", "outdoor", "indoor",
                             "dark", "bright", "neon", "warm", "cold", "fog", "rain"]
            checks.append(any(t in prompt_lower for t in setting_terms))  # has setting

        elif mode == "i2v":
            motion_terms = ["moves", "begins", "shifts", "turns", "starts", "changes",
                            "walks", "runs", "gestures", "reaches", "leans"]
            checks.append(any(t in prompt_lower for t in motion_terms))  # has motion

    elif mode == "image_gen":
        checks.append(not any(v in prompt_lower for v in IMAGE_ACTION_VERBS[:5]))  # no motion
        checks.append(len(prompt.split()) >= 8)  # minimum descriptiveness

    total = len(checks)
    passed = sum(1 for c in checks if c)
    return passed / total if total > 0 else 0.0
