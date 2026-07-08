"""
LTX Retake Renderer — for repairing a clip or correcting a scene region.

Preserves the intended scene goal and continuity while focusing the prompt
on the corrected section/behavior. Keeps consistent camera, subject, lighting,
and performance logic unless intentionally changed.
"""

from __future__ import annotations
from typing import Optional

from ..schema import ShotPlan, ProductionPlan
from .base import BaseRenderer


class LtxRetakeRenderer(BaseRenderer):
    mode = "retake"

    def render(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        **context,
    ) -> str:
        """Render a retake prompt.

        Context kwargs:
            prior_prompt: str — the original prompt that produced the flawed clip
            correction_notes: str — what should be different this time
            preserve_elements: list[str] — elements to keep from original

        Retake prompts preserve the scene's core visual identity while
        focusing on what needs to change.
        """
        prior_prompt = context.get("prior_prompt", "")
        correction_notes = context.get("correction_notes", "")
        preserve_elements = context.get("preserve_elements", [])

        parts = []

        # 1. Establish the core scene (same as original)
        # Use shot's structured data rather than blindly copying prior prompt
        style_shot = []
        if shot.visual_style:
            style_shot.append(shot.visual_style)
        style_shot.append(shot.camera_plan.framing)
        parts.append(", ".join(style_shot))

        # Setting (preserved from original)
        setting = self._setting_text(shot)
        if setting:
            parts.append(setting)

        # Characters (re-described for consistency)
        subjects = self._subjects_text(shot, plan)
        if subjects:
            spatial = f", {shot.spatial_setup}" if shot.spatial_setup else ""
            parts.append(f"{subjects}{spatial}")

        # 3. Corrected action sequence
        action = self._action_text(shot)
        if action:
            parts.append(action)

        # 4. Dialogue (if present)
        dialogue = self._dialogue_text(shot, plan)
        if dialogue:
            parts.append(dialogue)

        # 5. Camera (consistent unless intentionally changed)
        if shot.camera_plan.movement:
            parts.append(f"The camera {shot.camera_plan.movement}")

        # 6. Ending beat
        if shot.ending_beat:
            parts.append(shot.ending_beat)

        return self._join_prose(*parts)
