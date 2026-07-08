"""
LTX Image-to-Video Renderer — motion/change-focused.

Used when a start image already exists. Does NOT re-describe what's visible.
Focuses on what CHANGES after the starting frame.

Pass 1: assembles motion/change fields into a tagged draft
Pass 2: LLM rewrites into a concise motion-focused prompt
"""

from __future__ import annotations
from typing import Optional

from ..schema import ShotPlan, ProductionPlan
from .base import BaseRenderer


class LtxI2VRenderer(BaseRenderer):
    mode = "i2v"

    def _refinement_system_prompt(self, shot: ShotPlan, **context) -> str:
        return "Rewrite into 2-4 sentences, present tense. Describe ONLY what changes from the start image — motion, expressions, camera shifts. Keep short. Output ONLY the prompt."

    def render(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        **context,
    ) -> str:
        """Pass 1: assemble tagged draft for I2V."""
        parts = []

        # Brief style anchor
        if shot.visual_style:
            parts.append(f"STYLE: {shot.visual_style}")

        # What changes / who moves first
        action = self._action_text(shot)
        if action:
            parts.append(f"ACTION: {action}")

        # Performance direction
        if shot.performance_beats:
            parts.append(f"PERFORMANCE: {'. '.join(shot.performance_beats)}")

        # Dialogue (critical for lip sync)
        dialogue = self._dialogue_text(shot, plan)
        if dialogue:
            parts.append(f"DIALOGUE: {dialogue}")

        # Camera shift
        if shot.camera_plan.movement:
            parts.append(f"CAMERA: {shot.camera_plan.movement}")

        # Sound beginning
        if shot.audio_plan.ambience:
            parts.append(f"AUDIO: {shot.audio_plan.ambience}")

        # Ending beat
        if shot.ending_beat:
            parts.append(f"ENDING: {shot.ending_beat}")

        return " | ".join(parts)
