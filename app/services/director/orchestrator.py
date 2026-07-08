"""
Director Orchestrator — top-level coordinator for the Director Mode pipeline.

Responsibilities:
  1. Choose the right skill planner
  2. Pass normalized context
  3. Receive ProductionPlan
  4. Choose render modes per shot
  5. Run validators
  6. Dispatch rendered prompts

Supports feature flags for gradual migration from old system.
"""

from __future__ import annotations
import os
from typing import Optional, Any

from .schema import ProductionPlan, ShotPlan, RenderedPrompts
from .planners import MusicVideoPlanner, ShortFilmPlanner, PodcastPlanner, ViralVideoPlanner
from .renderers import (
    LtxT2VRenderer, LtxI2VRenderer, LtxA2VRenderer,
    LtxRetakeRenderer, LtxExtendRenderer, ImageGenRenderer,
)
from .validators import validate_shot_plan, validate_prompt_for_mode, compress_prompt


# ── Feature Flags ────────────────────────────────────────────────────

class DirectorFlags:
    """Feature flags for gradual rollout of new architecture."""

    def __init__(self):
        self.use_shared_shot_schema = True
        self.use_mode_specific_renderers = True
        self.use_prompt_validation = True
        self.use_prompt_compression = True
        self.use_llm_refinement = False  # Disabled — planner generates prompts directly in ShotPlan JSON
        self.aggressive_compression = False
        self.log_validation_details = True
        self.log_compression_deltas = True

    @staticmethod
    def from_dict(d: dict) -> "DirectorFlags":
        flags = DirectorFlags()
        for key, val in d.items():
            if hasattr(flags, key):
                setattr(flags, key, val)
        return flags

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Default flags
DEFAULT_FLAGS = DirectorFlags()


# ── Planner Registry ────────────────────────────────────────────────

_PLANNER_MAP = {
    "music_video": MusicVideoPlanner,
    "short_film": ShortFilmPlanner,
    "podcast": PodcastPlanner,
    "viral_video": ViralVideoPlanner,
}

# ── Renderer Registry ───────────────────────────────────────────────

_RENDERER_MAP = {
    "t2v": LtxT2VRenderer,
    "i2v": LtxI2VRenderer,
    "a2v": LtxA2VRenderer,
    "retake": LtxRetakeRenderer,
    "extend": LtxExtendRenderer,
    "image_gen": ImageGenRenderer,
}


# ── Main Orchestrator ────────────────────────────────────────────────

class DirectorOrchestrator:
    """Top-level Director Mode coordinator.

    Usage:
        director = DirectorOrchestrator(
            llm_generate=llm_service.generate,
            llm_generate_streaming=llm_service.generate_streaming,
        )

        # Plan
        plan = director.plan("music_video", clips=clips, scene_description="...", ...)

        # Render all shots
        rendered = director.render_plan(plan)

        # Or render a single shot
        prompts = director.render_shot(plan.shots[0], plan, mode="t2v")
    """

    def __init__(
        self,
        llm_generate: Any = None,
        llm_generate_streaming: Any = None,
        llm_refine: Any = None,
        flags: Optional[DirectorFlags] = None,
    ):
        self._generate = llm_generate
        self._generate_streaming = llm_generate_streaming
        self.flags = flags or DEFAULT_FLAGS

        # Build the refine function for pass 2
        # Use streaming for refinement — Qwen3.5 with --jinja always uses
        # thinking mode. The non-streaming API puts everything in reasoning_content
        # and returns empty content. Streaming properly separates thinking from content.
        if llm_refine:
            self._llm_refine = llm_refine
        elif llm_generate_streaming and self.flags.use_llm_refinement:
            self._llm_refine = llm_generate_streaming
        elif llm_generate and self.flags.use_llm_refinement:
            self._llm_refine = llm_generate
        else:
            self._llm_refine = None

        # Initialize renderers with refine function
        refine_fn = self._llm_refine if self.flags.use_llm_refinement else None
        self._renderers = {
            mode: cls(llm_refine=refine_fn)
            for mode, cls in _RENDERER_MAP.items()
        }

    # ── Planning ─────────────────────────────────────────────────

    def plan(self, skill_type: str, **kwargs) -> ProductionPlan:
        """Create a ProductionPlan using the appropriate skill planner.

        Args:
            skill_type: "music_video" | "short_film" | "podcast" | "viral_video"
            **kwargs: Skill-specific arguments (clips, scene_description, etc.)

        Returns:
            ProductionPlan with normalized ShotPlan objects.
        """
        planner_cls = _PLANNER_MAP.get(skill_type)
        if not planner_cls:
            raise ValueError(f"Unknown skill type: {skill_type}. Available: {list(_PLANNER_MAP.keys())}")

        planner = planner_cls(
            llm_generate=self._generate,
            llm_generate_streaming=self._generate_streaming,
        )

        print(f"[Director] Planning with {planner_cls.__name__}...")
        production_plan = planner.plan(**kwargs)

        # Validate shot plans
        if self.flags.use_prompt_validation:
            for shot in production_plan.shots:
                result = validate_shot_plan(shot, production_plan)
                if self.flags.log_validation_details:
                    if result.errors:
                        print(f"[Director] Shot {shot.shot_id} ERRORS: {result.errors}")
                    if result.warnings:
                        print(f"[Director] Shot {shot.shot_id} warnings: {result.warnings}")
                    if result.auto_fixes:
                        print(f"[Director] Shot {shot.shot_id} auto-fixes: {result.auto_fixes}")

        print(f"[Director] Plan complete: {len(production_plan.shots)} shots, "
              f"{production_plan.total_duration_sec:.1f}s total")

        return production_plan

    # ── Rendering ────────────────────────────────────────────────

    def render_plan(
        self,
        plan: ProductionPlan,
        prompt_type: str = "both",
        **context,
    ) -> list[dict]:
        """Render all shots in a ProductionPlan into final prompts.

        Args:
            plan: The ProductionPlan to render.
            prompt_type: "image" | "video" | "both"
            **context: Additional context (has_reference, etc.)

        Returns:
            List of dicts with video_prompt and/or image_prompt per shot.
        """
        results = []
        has_reference = context.pop("has_reference", False)
        if not has_reference and plan.reference_assets and plan.reference_assets.start_image:
            has_reference = True

        for i, shot in enumerate(plan.shots):
            result = {}

            # Choose render mode for video
            video_mode = self._choose_video_mode(shot, plan, has_reference)

            # Render video prompt
            if prompt_type in ("video", "both"):
                # Sliding window: if window_prompts exist, join with newlines for multi-window rendering
                if shot.window_prompts and len(shot.window_prompts) > 1:
                    validated_windows = []
                    for wp in shot.window_prompts:
                        # LLM may return dicts instead of strings
                        if isinstance(wp, dict):
                            wp = wp.get("prompt", wp.get("text", str(wp)))
                        validated_windows.append(str(wp).strip())
                    result["video_prompt"] = "\n".join(validated_windows)
                    result["window_count"] = len(validated_windows)
                else:
                    video_prompt = self._render_and_validate(
                        shot=shot,
                        plan=plan,
                        mode=video_mode,
                        prior_shot=plan.shots[i - 1] if i > 0 else None,
                        has_reference=has_reference,
                        **context,
                    )
                    result["video_prompt"] = video_prompt

            # Render image prompt
            if prompt_type in ("image", "both"):
                image_prompt = self._render_and_validate(
                    shot=shot,
                    plan=plan,
                    mode="image_gen",
                    has_reference=has_reference,
                    **context,
                )
                result["image_prompt"] = image_prompt

            # Pass through keyframe prompts if present
            if shot.keyframe_prompts:
                result["keyframe_prompts"] = shot.keyframe_prompts

            results.append(result)

        return results

    def render_shot(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        mode: Optional[str] = None,
        **context,
    ) -> str:
        """Render a single shot into a prompt for the given mode.

        If mode is not specified, auto-selects based on shot preferences.
        """
        if mode is None:
            mode = shot.source_mode_preference or "t2v"

        return self._render_and_validate(shot=shot, plan=plan, mode=mode, **context)

    def _render_and_validate(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan],
        mode: str,
        **context,
    ) -> str:
        """Use LLM-generated prompt if available, else deterministic render → validate → compress."""
        renderer = self._renderers.get(mode)
        if not renderer:
            print(f"[Director] Unknown render mode: {mode}, falling back to t2v")
            renderer = self._renderers["t2v"]

        # Prefer LLM-generated prompts from the planner (single-pass, full context)
        if mode == "image_gen" and shot.image_prompt:
            prompt = shot.image_prompt
        elif mode != "image_gen" and shot.video_prompt:
            prompt = shot.video_prompt
            print(f"[Director] Shot {shot.shot_id} video_prompt ({len(prompt)} chars): {prompt[:80]}...")
        else:
            # Fallback: deterministic render from structured fields
            prompt = renderer.render(shot, plan, **context)

        # Validate
        if self.flags.use_prompt_validation:
            validation = validate_prompt_for_mode(prompt, mode, shot, plan)
            if self.flags.log_validation_details and validation.warnings:
                print(f"[Director] Prompt validation ({shot.shot_id}, {mode}): "
                      f"score={validation.completeness_score:.2f}, "
                      f"warnings={validation.warnings}")

        # Compress
        if self.flags.use_prompt_compression:
            compression = compress_prompt(
                prompt, mode, shot,
                aggressive=self.flags.aggressive_compression,
            )
            if self.flags.log_compression_deltas and compression.chars_removed > 0:
                print(f"[Director] Compression ({shot.shot_id}, {mode}): "
                      f"{compression.original_words}->{compression.compressed_words} words, "
                      f"{compression.chars_removed} chars removed "
                      f"({compression.compression_ratio:.1%})")
            prompt = compression.compressed

        return prompt

    def _choose_video_mode(
        self,
        shot: ShotPlan,
        plan: ProductionPlan,
        has_reference: bool,
    ) -> str:
        """Auto-select the best render mode for a shot's video prompt."""
        # Explicit preference takes priority
        if shot.source_mode_preference and shot.source_mode_preference != "image_gen":
            return shot.source_mode_preference

        # Continuity-based selection
        if shot.continuity_strategy == "extend_previous":
            return "extend"

        # Audio-driven scenes
        if shot.audio_plan.mode in ("audio_driven", "dialogue_driven") and shot.audio_plan.lip_sync_critical:
            return "a2v"

        # Reference image available
        if has_reference and shot.image_strategy in ("reference_edit", "reference_inspired"):
            return "i2v"

        # Default: text-to-video
        return "t2v"

    # ── Retake / Extend Helpers ──────────────────────────────────

    def render_retake(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        prior_prompt: str = "",
        correction_notes: str = "",
        **context,
    ) -> str:
        """Render a retake prompt for a specific shot."""
        return self._render_and_validate(
            shot=shot, plan=plan, mode="retake",
            prior_prompt=prior_prompt,
            correction_notes=correction_notes,
            **context,
        )

    def render_extend(
        self,
        shot: ShotPlan,
        prior_shot: Optional[ShotPlan] = None,
        plan: Optional[ProductionPlan] = None,
        prior_prompt: str = "",
        **context,
    ) -> str:
        """Render an extend/continuation prompt."""
        prior_ending = prior_shot.ending_beat if prior_shot else ""
        return self._render_and_validate(
            shot=shot, plan=plan, mode="extend",
            prior_shot=prior_shot,
            prior_prompt=prior_prompt,
            prior_ending_beat=prior_ending,
            **context,
        )

    # ── Plan Serialization ───────────────────────────────────────

    @staticmethod
    def plan_to_clip_plans(rendered: list[dict]) -> list[dict]:
        """Convert rendered output to the clip_plans format expected by director_pipeline.

        This bridges the new architecture with the existing pipeline code.
        Returns: [{"video_prompt": str, "image_prompt": str}, ...]
        """
        result = []
        for r in rendered:
            clip = {
                "video_prompt": r.get("video_prompt", ""),
                "image_prompt": r.get("image_prompt", ""),
            }
            if r.get("window_count", 0) > 1:
                clip["window_count"] = r["window_count"]
                # Preserve individual window prompts for dashboard display
                clip["window_prompts"] = r["video_prompt"].split("\n") if r.get("video_prompt") else []
            if r.get("keyframe_prompts"):
                clip["keyframe_prompts"] = r["keyframe_prompts"]
            result.append(clip)
        return result

    @staticmethod
    def plan_to_legacy_format(plan: ProductionPlan, rendered: list[dict]) -> dict:
        """Convert to the legacy format returned by plan_clip_prompts_and_images.

        For backward compatibility during migration.
        """
        clip_plans = DirectorOrchestrator.plan_to_clip_plans(rendered)

        # Also build clips array for story-driven mode
        clips = []
        cumulative = 0.0
        for shot in plan.shots:
            clip = {
                "start": cumulative,
                "end": cumulative + shot.duration_sec,
                "duration_sec": shot.duration_sec,
                "duration_frames": shot.metadata.get("duration_frames") if shot.metadata else None,
                "label": shot.narrative_role or shot.scene_type or "scene",
                "beat_count": shot.metadata.get("beat_count", 0) if shot.metadata else 0,
            }
            clips.append(clip)
            cumulative += shot.duration_sec

        return {
            "clips": clips,
            "clip_plans": clip_plans,
            "production_plan": plan.to_dict(),
        }
