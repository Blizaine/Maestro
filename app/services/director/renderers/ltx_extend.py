"""
LTX Extend Renderer — for continuing from a previous clip.

Preserves continuity from the previous shot. Focuses on what happens NEXT.
Avoids resetting visual state. Treats the prompt as "next beat continuation."
"""

from __future__ import annotations
from typing import Optional

from ..schema import ShotPlan, ProductionPlan
from .base import BaseRenderer


class LtxExtendRenderer(BaseRenderer):
    mode = "extend"

    def render(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        **context,
    ) -> str:
        """Render an extend/continuation prompt.

        Context kwargs:
            prior_shot: ShotPlan — the shot being extended from
            prior_prompt: str — the prompt of the previous clip (for context)
            prior_ending_beat: str — how the previous clip ended

        Extension prompts:
        - Don't reset the visual state
        - Continue from where the prior clip left off
        - Focus on what happens NEXT
        """
        prior_ending = context.get("prior_ending_beat", "")
        prior_shot = context.get("prior_shot")

        parts = []

        # 1. Continuity anchor — reference where we're picking up from
        if prior_ending:
            parts.append(f"Continuing from: {prior_ending}")

        # 2. Current visual context (brief — don't reset everything)
        # Only re-state what's needed for model context
        subjects = self._subjects_text(shot, plan)
        if subjects:
            parts.append(subjects)

        # Minimal setting reference (only if environment changed)
        if prior_shot and shot.environment != getattr(prior_shot, 'environment', ''):
            setting = self._setting_text(shot)
            if setting:
                parts.append(setting)

        # 3. What happens next (the new action)
        action = self._action_text(shot)
        if action:
            parts.append(action)

        # 4. Dialogue continuation
        dialogue = self._dialogue_text(shot, plan)
        if dialogue:
            parts.append(dialogue)

        # 5. Camera continuation
        if shot.camera_plan.movement:
            parts.append(f"The camera {shot.camera_plan.movement}")

        # 6. New ending beat
        if shot.ending_beat:
            parts.append(shot.ending_beat)

        return self._join_prose(*parts)
