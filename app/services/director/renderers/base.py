"""
Base Renderer — interface for all mode-specific prompt renderers.

Renderers convert ShotPlan objects into final prompts for specific generation modes.

Two-pass pipeline:
  Pass 1 (deterministic): assemble structured ShotPlan fields into a draft prompt
  Pass 2 (LLM):           refine the draft into polished, concise prose

Each renderer provides mode-specific refinement instructions via _refinement_system_prompt().
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Any

from ..schema import ShotPlan, ProductionPlan, RenderedPrompts
from ..policies import (
    resolve_subjects_text, format_camera_text, format_action_sequence,
    format_scene_setting, format_dialogue_for_video,
)


class BaseRenderer(ABC):
    """Abstract base class for prompt renderers."""

    mode: str = ""  # "t2v", "i2v", "a2v", "retake", "extend", "image_gen"

    def __init__(self, llm_refine: Any = None):
        """
        Args:
            llm_refine: Optional callable for LLM-based prompt refinement (pass 2).
                        Signature: (prompt, system_prompt, max_new_tokens, temperature, ...) -> str
                        Should match llm_service.generate() signature.
        """
        self._llm_refine = llm_refine

    @abstractmethod
    def render(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        **context,
    ) -> str:
        """Pass 1: deterministic render of ShotPlan into a draft prompt string."""
        ...

    def render_and_refine(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        **context,
    ) -> str:
        """Full two-pass render: deterministic draft → LLM refinement.

        If no LLM is available, returns the deterministic draft as-is.
        """
        draft = self.render(shot, plan, **context)

        if self._llm_refine:
            return self._refine_with_llm(draft, shot, plan, **context)

        return draft

    # ── LLM Pass 2 ──────────────────────────────────────────────

    def _refinement_system_prompt(self, shot: ShotPlan, **context) -> str:
        """Mode-specific system prompt for LLM refinement.

        Subclasses override this. Keep prompts SHORT to minimize model overthinking.
        """
        return f"Rewrite this {self.mode} prompt into concise prose. Preserve all content. Output ONLY the rewritten prompt."

    def _refine_with_llm(
        self,
        draft: str,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        **context,
    ) -> str:
        """Call LLM to refine the deterministic draft into polished prose.

        Uses generate_streaming which properly separates reasoning from content
        in Qwen3.5's thinking mode, unlike the non-streaming API where
        reasoning_content eats all tokens.
        """
        if not self._llm_refine:
            return draft

        system = self._refinement_system_prompt(shot, **context)
        user_prompt = f"Rewrite this prompt concisely:\n\n{draft}"

        try:
            result = self._llm_refine(
                prompt=user_prompt,
                system_prompt=system,
                max_new_tokens=300,
                temperature=0.5,
                # Qwen3.5 always thinks with --jinja. Simplified system prompts
                # reduce thinking, but still need budget for it. 4096 thinking +
                # 300 content = 4396 total — enough for even verbose thinkers.
                thinking_budget=4096,
            )
            refined = (result or "").strip()
            if refined and len(refined) > 10:
                return refined
            print(f"[Renderer] LLM refinement returned empty/short result, using draft")
            return draft
        except Exception as e:
            print(f"[Renderer] LLM refinement failed: {e}")
            return draft

    # ── Shared Assembly Helpers ──────────────────────────────────

    @staticmethod
    def _join_prose(*parts: str, separator: str = ". ") -> str:
        """Join non-empty text parts into a flowing paragraph."""
        clean = [p.strip().rstrip(".") for p in parts if p and p.strip()]
        if not clean:
            return ""
        result = (separator.join(clean) + ".").replace(".. ", ". ").replace("..", ".")
        while "  " in result:
            result = result.replace("  ", " ")
        return result

    @staticmethod
    def _subjects_text(shot: ShotPlan, plan: Optional[ProductionPlan] = None) -> str:
        return resolve_subjects_text(shot, plan)

    @staticmethod
    def _camera_text(shot: ShotPlan) -> str:
        return format_camera_text(shot.camera_plan)

    @staticmethod
    def _action_text(shot: ShotPlan) -> str:
        return format_action_sequence(shot)

    @staticmethod
    def _setting_text(shot: ShotPlan) -> str:
        return format_scene_setting(shot)

    @staticmethod
    def _dialogue_text(shot: ShotPlan, plan: Optional[ProductionPlan] = None) -> str:
        return format_dialogue_for_video(shot, plan)
