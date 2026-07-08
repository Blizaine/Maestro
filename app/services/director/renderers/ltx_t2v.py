"""
LTX Text-to-Video Renderer — the most descriptive mode.

Used when the model must invent the whole shot from scratch.
Pass 1: assembles all ShotPlan fields into a tagged draft
Pass 2: LLM rewrites into a flowing cinematic paragraph
"""

from __future__ import annotations
from typing import Optional

from ..schema import ShotPlan, ProductionPlan
from .base import BaseRenderer


class LtxT2VRenderer(BaseRenderer):
    mode = "t2v"

    def _refinement_system_prompt(self, shot: ShotPlan, **context) -> str:
        return "Rewrite into one flowing paragraph, present tense, 4-8 sentences. Include shot type, setting, characters by appearance, action, camera movement, ending beat. No names outside dialogue. Output ONLY the prompt."

    def render(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        **context,
    ) -> str:
        """Pass 1: assemble tagged draft for T2V."""
        parts = []

        # Style and shot type
        style_shot = []
        if shot.visual_style:
            style_shot.append(shot.visual_style)
        style_shot.append(shot.camera_plan.framing)
        if shot.camera_plan.angle:
            style_shot.append(shot.camera_plan.angle)
        parts.append(f"SHOT: {', '.join(style_shot)}")

        # Setting
        setting = self._setting_text(shot)
        if setting:
            parts.append(f"SETTING: {setting}")

        # Characters
        subjects = self._subjects_text(shot, plan)
        if subjects:
            spatial = f", {shot.spatial_setup}" if shot.spatial_setup else ""
            parts.append(f"SUBJECTS: {subjects}{spatial}")

        # Action beats
        action = self._action_text(shot)
        if action:
            parts.append(f"ACTION: {action}")

        # Dialogue
        dialogue = self._dialogue_text(shot, plan)
        if dialogue:
            parts.append(f"DIALOGUE: {dialogue}")

        # Camera movement
        if shot.camera_plan.movement:
            parts.append(f"CAMERA: {shot.camera_plan.movement} ({shot.camera_plan.movement_intensity})")

        # Audio/ambient
        if shot.audio_plan.ambience:
            parts.append(f"AUDIO: {shot.audio_plan.ambience}")

        # Ending beat
        if shot.ending_beat:
            parts.append(f"ENDING: {shot.ending_beat}")

        return " | ".join(parts)
