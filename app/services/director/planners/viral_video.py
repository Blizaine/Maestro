"""
Viral Video Planner — creates a ProductionPlan for short-form viral content.

Optimizes for early hooks, rapid pacing, and platform-specific patterns
(TikTok, Reels, Shorts). Emphasizes the first 1-2 seconds.

Outputs: ProductionPlan with ShotPlan objects.
"""

from __future__ import annotations
from typing import Optional, Any

from ..schema import (
    ProductionPlan, ShotPlan, CharacterProfile, ReferenceAssets,
    AssetRef, SubjectRef, DialogueBeat, CameraPlan, AudioPlan,
)
from ..policies import build_character_rules_block, build_camera_style_block
from .base import BasePlanner


# Platform defaults
_PLATFORM_DURATIONS = {
    "tiktok": 30,
    "reels": 30,
    "shorts": 60,
    "general": 30,
}

_VIRAL_STYLES = {
    "meme": "bold text overlays, exaggerated reactions, jump cuts between setups",
    "ugc": "casual handheld, natural lighting, authentic feel, direct address",
    "cinematic": "polished visuals, dramatic lighting, fast-paced storytelling",
    "direct_address": "tight framing on speaker, eye contact with camera, punchy delivery",
    "reaction": "split screen energy, exaggerated expressions, before/after reveals",
    "tutorial": "clear demonstrations, close-ups on process, minimal distractions",
}


class ViralVideoPlanner(BasePlanner):
    skill_type = "viral_video"

    def plan(
        self,
        concept: str,
        audio_path: Optional[str] = None,
        reference_image_path: Optional[str] = None,
        characters: Optional[list[dict]] = None,
        target_duration: int = 30,
        platform: str = "general",
        style: str = "cinematic",
        **kwargs,
    ) -> ProductionPlan:
        """Create a ProductionPlan for a viral video.

        Args:
            concept: Video concept/idea.
            audio_path: Optional audio track.
            reference_image_path: Optional reference photo.
            characters: Character descriptions.
            target_duration: Target duration in seconds.
            platform: Target platform (tiktok, reels, shorts, general).
            style: Visual style (meme, ugc, cinematic, direct_address, reaction, tutorial).
        """
        has_reference = bool(reference_image_path)

        # Clamp duration to platform norms
        max_dur = _PLATFORM_DURATIONS.get(platform, 30)
        target_duration = min(target_duration, max_dur * 2)  # allow some flex

        char_profiles = []
        if characters:
            char_profiles = [
                CharacterProfile(
                    id=f"char_{i}",
                    display_name=c.get("name", ""),
                    physical_description=c.get("description", "person"),
                )
                for i, c in enumerate(characters)
            ]

        ref_assets = ReferenceAssets(
            start_image=AssetRef(id="ref_image", type="image", uri=reference_image_path) if has_reference else None,
            audio=AssetRef(id="audio", type="audio", uri=audio_path) if audio_path else None,
        )

        # Plan number of shots — viral videos use more cuts
        target_scenes = max(3, min(12, target_duration // 5))

        style_desc = _VIRAL_STYLES.get(style, _VIRAL_STYLES["cinematic"])

        char_rules = build_character_rules_block(has_reference, char_profiles if char_profiles else None)
        camera_block = build_camera_style_block()

        system_prompt = f"""You are a viral video director creating content optimized for {platform}.

{f"You are given a REFERENCE PHOTO." if has_reference else ""}

Style: {style} — {style_desc}

VIRAL VIDEO RULES:
- The FIRST SHOT is critical — it must hook viewers in 1-2 seconds.
- Front-load the most interesting visual, action, or statement.
- Keep shots SHORT (3-8 seconds). Rapid pacing holds attention.
- Each shot should feel like it earns its screen time.
- End with a payoff, reveal, or callback to the hook.
- For direct-address: tight framing, eye contact, punchy delivery.
- For meme: exaggerated reactions, setup/punchline structure.
- For UGC: authentic feel, handheld energy, relatable moments.

{char_rules}

{camera_block}

OUTPUT FORMAT — respond with ONLY a JSON array:
[
  {{
    "scene_goal": "Hook — grab attention immediately",
    "scene_type": "hook|setup|escalation|payoff|callback",
    "duration_sec": 3,
    "subjects_on_screen": [{{"visual_description": "person in bright outfit"}}],
    "spatial_setup": "tight center frame",
    "environment": "urban street",
    "visual_style": "{style}",
    "lighting": "natural daylight",
    "mood": "energetic",
    "action_beats": ["person turns to camera with surprised expression"],
    "dialogue_beats": [{{"spoken_text": "Wait, what?", "delivery": "shocked"}}],
    "camera_plan": {{"framing": "close-up", "movement": "snap zoom", "movement_intensity": "dynamic"}},
    "audio_plan": {{"mode": "generated_audio"}},
    "ending_beat": "freeze frame on reaction"
  }}
]

Output exactly {target_scenes} shot plans totaling ~{target_duration}s. Go:"""

        user_prompt = f"""Concept: {concept}
Platform: {platform}
Target duration: {target_duration}s
Style: {style}

Create {target_scenes} short, punchy shots. Hook first! Go:"""

        image_paths = [reference_image_path] if has_reference and reference_image_path else None
        max_tokens = max(1024, target_scenes * 300 + 512)

        shot_dicts = self._call_llm_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            image_paths=image_paths,
        )

        shots = []
        for i, raw in enumerate(shot_dicts):
            duration = raw.get("duration_sec", raw.get("duration", 5))

            subjects = [SubjectRef.from_dict(s) if isinstance(s, dict) else SubjectRef(visual_description=str(s))
                        for s in raw.get("subjects_on_screen", [])]

            cam_raw = raw.get("camera_plan", {})
            camera = CameraPlan(
                framing=cam_raw.get("framing", "medium shot"),
                angle=cam_raw.get("angle"),
                movement=cam_raw.get("movement"),
                movement_intensity=cam_raw.get("movement_intensity", "dynamic"),
            )

            audio_raw = raw.get("audio_plan", {})
            audio = AudioPlan(
                mode=audio_raw.get("mode", "generated_audio"),
                ambience=audio_raw.get("ambience"),
                timing_anchor="video",
            )

            dialogue_beats = [DialogueBeat.from_dict(db) for db in raw.get("dialogue_beats", [])] if raw.get("dialogue_beats") else None

            shot = ShotPlan(
                shot_id=self._make_shot_id(i, "vv"),
                index=i,
                duration_sec=duration,
                skill_type="viral_video",
                scene_goal=raw.get("scene_goal", f"Shot {i + 1}"),
                scene_type=raw.get("scene_type", "escalation"),
                source_mode_preference="i2v" if has_reference else "t2v",
                image_strategy="reference_edit" if has_reference else "fresh_generation",
                continuity_strategy="independent",
                subjects_on_screen=subjects,
                spatial_setup=raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                visual_style=raw.get("visual_style", style),
                lighting=raw.get("lighting", ""),
                mood=raw.get("mood", "energetic"),
                action_beats=raw.get("action_beats", []),
                dialogue_beats=dialogue_beats,
                camera_plan=camera,
                audio_plan=audio,
                ending_beat=raw.get("ending_beat", ""),
                metadata={
                    "platform": platform,
                    "style": style,
                    "is_hook": i == 0,
                },
            )
            shots.append(shot)

        total_dur = sum(s.duration_sec for s in shots)

        return ProductionPlan(
            skill_type="viral_video",
            title=None,
            global_style=f"{style} — {concept}",
            total_duration_sec=total_dur,
            reference_assets=ref_assets,
            characters=char_profiles if char_profiles else None,
            shots=shots,
            continuity_notes=[
                f"Viral video for {platform} — hook in first 1-2 seconds",
                "Short shots, rapid pacing, clear payoff",
                f"Style: {style}",
            ],
        )
