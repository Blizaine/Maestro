"""
Image Generation Renderer — generates start frame / scene image prompts.

Produces a single static image description. No motion language.
If based on a reference image, focuses on what should CHANGE from the reference.

Pass 1: assembles structured fields into a tagged draft
Pass 2: LLM rewrites into a concise, natural image prompt
"""

from __future__ import annotations
from typing import Optional

from ..schema import ShotPlan, ProductionPlan
from .base import BaseRenderer


class ImageGenRenderer(BaseRenderer):
    mode = "image_gen"

    def render(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        **context,
    ) -> str:
        """Pass 1: assemble a structured draft from ShotPlan fields.

        The draft is tagged for the LLM to understand each element's role.
        If no LLM is available, this is used directly.
        """
        has_reference = context.get("has_reference", False) or (
            shot.image_strategy in ("reference_edit", "reference_inspired")
        )

        if has_reference:
            return self._render_reference_edit(shot, plan)
        else:
            return self._render_fresh(shot, plan)

    def _refinement_system_prompt(self, shot: ShotPlan, **context) -> str:
        """Image-specific LLM refinement instructions — kept minimal to avoid overthinking."""
        has_reference = context.get("has_reference", False) or (
            shot.image_strategy in ("reference_edit", "reference_inspired")
        )

        if has_reference:
            return "Rewrite into a concise image edit prompt under 30 words. Describe what to CHANGE from the reference — framing, setting, lighting. No names, no motion verbs. Output ONLY the prompt."
        else:
            return "Rewrite into a concise image prompt under 35 words. One static frame — composition, subjects by appearance, setting, lighting. No names, no motion. Output ONLY the prompt."

    def _render_reference_edit(self, shot: ShotPlan, plan: Optional[ProductionPlan] = None) -> str:
        """Structured draft for reference edit mode."""
        parts = []

        # Camera reframing
        if shot.camera_plan.framing:
            parts.append(f"REFRAME: {shot.camera_plan.framing}")
            if shot.camera_plan.angle:
                parts.append(f"({shot.camera_plan.angle})")

        # Environment/setting change
        if shot.environment:
            parts.append(f"SETTING: {shot.environment}")

        # Lighting
        if shot.lighting:
            parts.append(f"LIGHTING: {shot.lighting}")

        # Subject description
        subjects = self._subjects_text(shot, plan)
        if subjects:
            spatial = f", {shot.spatial_setup}" if shot.spatial_setup else ""
            parts.append(f"SUBJECT: {subjects}{spatial}")

        # Mood
        if shot.mood:
            parts.append(f"MOOD: {shot.mood}")

        # Style
        if shot.visual_style:
            parts.append(f"STYLE: {shot.visual_style}")

        return " | ".join(p for p in parts if p)

    def _render_fresh(self, shot: ShotPlan, plan: Optional[ProductionPlan] = None) -> str:
        """Structured draft for fresh image generation."""
        parts = []

        # Style and framing
        style_parts = []
        if shot.visual_style:
            style_parts.append(shot.visual_style)
        style_parts.append(shot.camera_plan.framing)
        if shot.camera_plan.angle:
            style_parts.append(shot.camera_plan.angle)
        parts.append(f"SHOT: {', '.join(style_parts)}")

        # Environment
        if shot.environment:
            parts.append(f"SETTING: {shot.environment}")

        # Lighting
        if shot.lighting:
            parts.append(f"LIGHTING: {shot.lighting}")

        # Characters
        subjects = self._subjects_text(shot, plan)
        if subjects:
            spatial = f", {shot.spatial_setup}" if shot.spatial_setup else ""
            parts.append(f"SUBJECT: {subjects}{spatial}")

        # Mood
        if shot.mood:
            parts.append(f"MOOD: {shot.mood}")

        return " | ".join(p for p in parts if p)
