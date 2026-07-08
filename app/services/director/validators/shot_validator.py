"""
Shot Plan Validator — validates ShotPlan objects before rendering.

Checks required fields, valid values, and structural integrity.
Returns list of warnings (non-fatal) and errors (fatal).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from ..schema import (
    ShotPlan, ProductionPlan,
    VALID_SKILL_TYPES, VALID_SOURCE_MODES, VALID_IMAGE_STRATEGIES, VALID_CONTINUITY_STRATEGIES,
)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    auto_fixes: list[str]  # fixes applied automatically

    def __bool__(self):
        return self.valid


def validate_shot_plan(shot: ShotPlan, plan: Optional[ProductionPlan] = None) -> ValidationResult:
    """Validate a ShotPlan for completeness and correctness.

    Returns ValidationResult with errors (blocking) and warnings (advisory).
    Auto-fixes minor issues when possible.
    """
    errors = []
    warnings = []
    auto_fixes = []

    # ── Required fields ──────────────────────────────────────────
    if not shot.shot_id:
        errors.append("Missing shot_id")
    if not shot.scene_goal:
        warnings.append("Missing scene_goal — shot has no stated purpose")
    if shot.duration_sec <= 0:
        errors.append(f"Invalid duration: {shot.duration_sec}s")
    if shot.duration_sec > 120:
        warnings.append(f"Very long shot: {shot.duration_sec}s — consider splitting")

    # ── Enum validation ──────────────────────────────────────────
    if shot.skill_type not in VALID_SKILL_TYPES:
        warnings.append(f"Unknown skill_type: {shot.skill_type}")

    if shot.source_mode_preference and shot.source_mode_preference not in VALID_SOURCE_MODES:
        warnings.append(f"Unknown source_mode_preference: {shot.source_mode_preference}")

    if shot.image_strategy and shot.image_strategy not in VALID_IMAGE_STRATEGIES:
        warnings.append(f"Unknown image_strategy: {shot.image_strategy}")

    if shot.continuity_strategy not in VALID_CONTINUITY_STRATEGIES:
        warnings.append(f"Unknown continuity_strategy: {shot.continuity_strategy}")

    # ── Content checks ───────────────────────────────────────────
    if not shot.subjects_on_screen:
        if shot.scene_type not in ("atmospheric", "abstract", "b_roll"):
            warnings.append("No subjects_on_screen — shot may feel empty unless intentionally abstract")

    if not shot.action_beats and not shot.dialogue_beats:
        if shot.scene_type not in ("atmospheric", "abstract", "b_roll"):
            warnings.append("No action_beats or dialogue_beats — shot has nothing happening")

    if not shot.ending_beat:
        warnings.append("Missing ending_beat — shots should end on a clear visual moment")

    # ── Camera validation ────────────────────────────────────────
    if not shot.camera_plan.framing:
        shot.camera_plan.framing = "medium shot"
        auto_fixes.append("Set default camera framing to 'medium shot'")

    valid_intensities = {"static", "subtle", "moderate", "dynamic"}
    if shot.camera_plan.movement_intensity not in valid_intensities:
        shot.camera_plan.movement_intensity = "subtle"
        auto_fixes.append(f"Normalized camera movement_intensity to 'subtle'")

    # ── Audio validation ─────────────────────────────────────────
    valid_modes = {"generated_audio", "audio_driven", "dialogue_driven", "music_driven", "ambient_only"}
    if shot.audio_plan.mode not in valid_modes:
        shot.audio_plan.mode = "ambient_only"
        auto_fixes.append(f"Normalized audio mode to 'ambient_only'")

    # ── Dialogue budget check ────────────────────────────────────
    if shot.dialogue_beats:
        total_words = sum(len(db.spoken_text.split()) for db in shot.dialogue_beats)
        budget = int(shot.duration_sec * 2.5)
        if total_words > budget * 1.5:
            warnings.append(
                f"Dialogue over-budget: {total_words} words for {shot.duration_sec}s shot "
                f"(budget ~{budget} words at ~2 words/sec)"
            )

    # ── Action beat count check ──────────────────────────────────
    beat_count = len(shot.action_beats) + len(shot.performance_beats or [])
    if shot.duration_sec >= 10 and beat_count < 2:
        warnings.append(f"Only {beat_count} action beats for {shot.duration_sec}s shot — consider adding more")
    if shot.duration_sec >= 15 and beat_count < 3:
        warnings.append(f"Only {beat_count} action beats for {shot.duration_sec}s shot — aim for 3-5 meaningful beats")

    # ── Character reference check ────────────────────────────────
    if plan and plan.characters and shot.subjects_on_screen:
        for subj in shot.subjects_on_screen:
            if subj.character_id:
                char = plan.get_character(subj.character_id)
                if not char:
                    warnings.append(f"Subject references unknown character_id: {subj.character_id}")

    is_valid = len(errors) == 0
    return ValidationResult(valid=is_valid, errors=errors, warnings=warnings, auto_fixes=auto_fixes)
