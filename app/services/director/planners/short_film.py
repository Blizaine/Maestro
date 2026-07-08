"""
Short Film Planner — creates a ProductionPlan from story/audio inputs.

Supports two paths:
  1. Audio-driven: dialogue audio + transcript → scene plans
  2. Story-driven: story description + characters → scene plans (no audio)

Outputs: ProductionPlan with ShotPlan objects (NOT final prompts).
"""

from __future__ import annotations
import os
import re
from typing import Optional, Any

from ..schema import (
    ProductionPlan, ShotPlan, CharacterProfile, ReferenceAssets,
    AssetRef, SubjectRef, DialogueBeat, CameraPlan, AudioPlan,
)
from ..policies import build_character_rules_block, build_camera_style_block
from ..guide_loader import load_guide as _load_guide_helper
from .base import BasePlanner


# Video-model architecture → Pass 2 shot-breakdown guide file.
# Currently only LTX-2/LTX-V have a dedicated Pass-2 guide. Other
# video families share the LTX-2 rules as a best-effort fallback
# until per-model Pass-2 guides land in Phase 3.
_VIDEO_PASS2_GUIDE_MAP = {
    "ltx2": "ltx2_shot_breakdown.md",
    "ltxv": "ltx2_shot_breakdown.md",
}


# ── Pass 2 JSON output schemas (llama-server grammar constraint) ──────
# These mirror the JSON examples embedded in the Pass 2 / fallback system
# prompts. llama-server compiles the schema to a GBNF grammar that masks
# every token which would break it, so a constrained pass physically
# cannot emit prose, markdown fences, or repeat-loop garbage (the Gemma 4
# 12B failure: 96K chars of looping pseudo-JSON on a 5-min film).
#
# additionalProperties=False is the actual loop-killer: a grammar-compiled
# closed object emits each key AT MOST ONCE, in this defined order, so the
# "repeat the same field/object until max_tokens" failure class becomes
# unrepresentable. The flip side: any field a prompt's output spec asks
# for MUST be listed here, in spec order, or the grammar will forbid the
# model from writing it. If you add a field to a Pass 2 output spec,
# add it to _SHOT_PROPERTIES too.
#
# Strings stay unbounded (creative prose can't be length-capped at the
# grammar level) — intra-string repetition remains covered by the
# registry-level repeat penalties in llm_service.

_SUBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "visual_description": {"type": "string"},
        "character_id": {"type": "string"},
        "speaker_name": {"type": "string"},
        "position_or_relation": {"type": "string"},
    },
    "required": ["visual_description"],
    "additionalProperties": False,
}

_DIALOGUE_BEAT_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker_id": {"type": "string"},
        "spoken_text": {"type": "string"},
        "delivery": {"type": "string"},
        "physical_cue": {"type": "string"},
        "priority": {"type": "string"},
    },
    "required": ["spoken_text"],
    "additionalProperties": False,
}

_CAMERA_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "framing": {"type": "string"},
        "angle": {"type": "string"},
        "movement": {"type": "string"},
        "movement_intensity": {"type": "string"},
        "lens_feel": {"type": "string"},
        "reframing_notes": {"type": "string"},
    },
    "required": ["framing"],
    "additionalProperties": False,
}

_AUDIO_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string"},
        "ambience": {"type": "string"},
        "effects": {"type": "array", "items": {"type": "string"}},
        "vocal_style": {"type": "string"},
        "timing_anchor": {"type": "string"},
        "lip_sync_critical": {"type": "boolean"},
    },
    "required": ["mode"],
    "additionalProperties": False,
}

# Union of every field the story-mode spec, the audio-mode spec, and the
# single-pass fallback spec request, in spec order (story-mode order;
# audio-mode-only fields slot where its spec shows them). Per-call-site
# `required` lists pick which subset the grammar forces.
_SHOT_PROPERTIES = {
    "title": {"type": "string"},
    "duration_sec": {"type": "number"},
    "scene_goal": {"type": "string"},
    "narrative_role": {"type": "string"},
    "scene_type": {"type": "string"},
    "subjects_on_screen": {"type": "array", "items": _SUBJECT_SCHEMA},
    "spatial_setup": {"type": "string"},
    "environment": {"type": "string"},
    "visual_style": {"type": "string"},
    "lighting": {"type": "string"},
    "mood": {"type": "string"},
    "action_beats": {"type": "array", "items": {"type": "string"}},
    "dialogue_beats": {"type": "array", "items": _DIALOGUE_BEAT_SCHEMA},
    "camera_plan": _CAMERA_PLAN_SCHEMA,
    "audio_plan": _AUDIO_PLAN_SCHEMA,
    "ending_beat": {"type": "string"},
    "image_source": {"type": "string"},
    "image_prompt": {"type": "string"},
    "visual_changes": {"type": "array", "items": {"type": "string"}},
    "video_prompt": {"type": "string"},
    "multishot": {"type": "boolean"},
    "keyframe_prompts": {"type": "array", "items": {"type": "string"}},
    "window_prompts": {"type": "array", "items": {"type": "string"}},
}


def _shot_list_schema(min_items: int, max_items: int, required: list[str]) -> dict:
    """JSON schema for a Pass 2 shot list: a bounded array of closed shot objects."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": _SHOT_PROPERTIES,
            "required": required,
            "additionalProperties": False,
        },
        "minItems": max(1, min_items),
        "maxItems": max(1, max_items),
    }


def _model_specific_pass2_notes(video_model: str) -> str:
    """Per-checkpoint prompting notes for the active model, or "" if none.

    CivitAI / HF checkpoint imports carry a generated prompting DELTA (trigger
    words + preferred style, see Phase 2) stored inline on the model_def as
    `enhance_guide_text`. Surfacing it in Director's Pass-2 system prompt lets a
    custom (often NSFW) checkpoint prompt as well in Director as it does in
    Studio — closing the gap where Director ignored per-checkpoint guides.

    Only the inline delta is used. Built-in fine-tunes that ship a file-based
    `enhance_guide` (Sulphur, 10Eros) are intentionally NOT pulled in here: those
    are full Studio-format "rewrite the prompt" guides that would conflict with
    Director's shot-breakdown instructions and JSON output contract.
    """
    try:
        from wgp import get_model_def
        md = get_model_def(video_model)
    except Exception:
        return ""
    notes = (md or {}).get("enhance_guide_text")
    if not (isinstance(notes, str) and notes.strip()):
        return ""
    return (
        "MODEL-SPECIFIC PROMPTING NOTES — the active checkpoint is a community "
        "fine-tune with its own conventions. Apply these to every shot prompt; "
        "they augment trigger words and style, they do not override the shot "
        "structure or output format:\n" + notes.strip()
    )


def _route_video_pass2_guide(video_model: str) -> str:
    """Pick the Pass 2 video guide for `video_model`, plus any per-checkpoint notes."""
    if not video_model:
        return _load_guide_helper("ltx2_shot_breakdown.md") or ""
    model_lower = video_model.lower()
    best_match: str | None = None
    best_len = 0
    for prefix, guide_file in _VIDEO_PASS2_GUIDE_MAP.items():
        if model_lower.startswith(prefix) and len(prefix) > best_len:
            best_match = guide_file
            best_len = len(prefix)
    chosen = best_match or "ltx2_shot_breakdown.md"
    if not best_match:
        print(f"[ShortFilmPlanner] No Pass-2 video guide for model={video_model!r}; falling back to {chosen}")
    guide = _load_guide_helper(chosen) or ""

    # Layer the active checkpoint's per-model prompting delta (Phase 2) on top.
    delta = _model_specific_pass2_notes(video_model)
    if delta:
        guide = f"{guide}\n\n{delta}" if guide else delta
    return guide


class ShortFilmPlanner(BasePlanner):
    skill_type = "short_film"

    def plan(
        self,
        story_description: str = "",
        clips: Optional[list[dict]] = None,
        audio_path: Optional[str] = None,
        reference_image_path: Optional[str] = None,
        characters: Optional[list[dict]] = None,
        lyrics: Optional[list[dict]] = None,
        speaker_mappings: Optional[dict] = None,
        target_duration: int = 60,
        target_scenes: Optional[int] = None,
        narrative_mode: bool = True,
        fps: int = 24,
        frames_steps: int = 8,
        frames_minimum: int = 41,
        **kwargs,
    ) -> ProductionPlan:
        """Create a ProductionPlan for a short film.

        If `clips` are provided → audio-driven mode (scenes follow audio structure).
        If no clips → story-driven mode (LLM plans scene structure from scratch).
        """
        has_reference = bool(reference_image_path)
        is_audio_mode = bool(clips)
        # Store extra ref info for use in private methods
        self._num_character_refs = len(kwargs.get("character_ref_paths", []) or [])
        self._num_location_refs = len(kwargs.get("location_ref_paths", []) or [])
        self._character_ref_labels = kwargs.get("character_ref_labels")
        self._location_ref_labels = kwargs.get("location_ref_labels")
        self._character_ref_paths_raw = kwargs.get("character_ref_paths", [])
        self._location_ref_paths_raw = kwargs.get("location_ref_paths", [])
        self._seamless = kwargs.get("seamless", True)
        # Capture model identifiers for Pass-2 dialect-aware guide routing.
        # These flow from director_pipeline.py's planner_kwargs and let
        # _run_story_mode + _plan_audio_driven pick the correct video and
        # image guide files (ltx2_shot_breakdown.md for LTX-2,
        # flux_image_edit_pass2.md for Flux.2 Klein, etc.).
        self._video_model = kwargs.get("video_model", "") or ""
        self._image_model = kwargs.get("image_model", "") or ""

        # Normalize speaker_mappings: frontend sends list, we need dict
        if isinstance(speaker_mappings, list):
            sm_dict: dict = {}
            for entry in speaker_mappings:
                if isinstance(entry, dict):
                    sid = entry.get("speakerId") or entry.get("speaker_id", "")
                    if sid:
                        sm_dict[sid] = {"name": entry.get("name", ""), "role": entry.get("role", "")}
            speaker_mappings = sm_dict

        # Build character profiles
        char_profiles = self._build_characters(characters)

        # Build reference assets
        ref_assets = ReferenceAssets(
            start_image=AssetRef(id="ref_image", type="image", uri=reference_image_path) if has_reference else None,
            audio=AssetRef(id="audio", type="audio", uri=audio_path) if audio_path else None,
            transcript="\n".join(l.get("text", "") for l in (lyrics or []) if l.get("text", "").strip()),
        )

        nsfw = kwargs.get("nsfw", False)
        polish_block = kwargs.get("polish_block", "")
        # Multi-shot LoRA mode — when on, Pass 2 emits storyboard-format
        # video_prompts for medium-length shots. See the toggle's
        # comment in launch.py for behavior details. Threaded through
        # to _plan_story_driven below.
        multishot_lora_mode = kwargs.get("multishot_lora_mode", False)

        if is_audio_mode:
            shots = self._plan_audio_driven(
                clips=clips,
                story_description=story_description,
                lyrics=lyrics,
                speaker_mappings=speaker_mappings,
                reference_image_path=reference_image_path,
                char_profiles=char_profiles,
                has_reference=has_reference,
                nsfw=nsfw,
                polish_block=polish_block,
            )
        else:
            shots, title = self._plan_story_driven(
                story_description=story_description,
                reference_image_path=reference_image_path,
                char_profiles=char_profiles,
                has_reference=has_reference,
                target_duration=target_duration,
                target_scenes=target_scenes,
                narrative_mode=narrative_mode,
                fps=fps,
                frames_steps=frames_steps,
                frames_minimum=frames_minimum,
                nsfw=nsfw,
                polish_block=polish_block,
                multishot_lora_mode=multishot_lora_mode,
            )

        total_duration = sum(s.duration_sec for s in shots) if shots else target_duration

        return ProductionPlan(
            skill_type="short_film",
            title=getattr(self, '_last_title', None),
            global_style=story_description,
            total_duration_sec=total_duration,
            reference_assets=ref_assets,
            characters=char_profiles if char_profiles else None,
            shots=shots,
            continuity_notes=[
                "Short film — maintain visual and narrative continuity across shots",
                "Match camera complexity to emotional content",
                "Dialogue must appear in video prompts with speaker cues",
            ],
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _build_all_image_paths(self, reference_image_path: Optional[str], has_reference: bool) -> Optional[list[str]]:
        """Build image_paths list with ALL reference images (main + character + location)."""
        paths = []
        if has_reference and reference_image_path:
            paths.append(reference_image_path)
        for cp in (getattr(self, '_character_ref_paths_raw', None) or []):
            if cp and os.path.isfile(cp):
                paths.append(cp)
        for lp in (getattr(self, '_location_ref_paths_raw', None) or []):
            if lp and os.path.isfile(lp):
                paths.append(lp)
        return paths if paths else None

    # ── Character Building ───────────────────────────────────────────

    def _build_characters(self, characters: Optional[list[dict]]) -> list[CharacterProfile]:
        if not characters:
            return []
        return [
            CharacterProfile(
                id=f"char_{i}",
                display_name=c.get("name", ""),
                physical_description=c.get("description", "person"),
            )
            for i, c in enumerate(characters)
        ]

    # ── Audio-Driven Planning ────────────────────────────────────────

    def _plan_audio_driven(
        self,
        clips: list[dict],
        story_description: str,
        lyrics: Optional[list[dict]],
        speaker_mappings: Optional[dict],
        reference_image_path: Optional[str],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        nsfw: bool = False,
        polish_block: str = "",
    ) -> list[ShotPlan]:
        """Plan shots from existing audio-segmented clips."""
        from ..nsfw_guidance import inject_nsfw_if_enabled

        speaker_names = {}
        if speaker_mappings:
            for sid, info in speaker_mappings.items():
                speaker_names[sid] = info.get("name", sid)

        # Build clip contexts
        clip_contexts = []
        for i, clip in enumerate(clips):
            start_sec = clip.get("start", 0)
            end_sec = clip.get("end", start_sec + 5)
            duration = end_sec - start_sec
            label = clip.get("label", "scene")

            # Gather dialogue
            dialogue_lines = []
            speakers_in_clip = set()
            if lyrics:
                for l in lyrics:
                    if l.get("start", 0) < end_sec and l.get("end", 0) > start_sec:
                        spk = l.get("speaker", "")
                        text = l.get("text", "")
                        if text.strip():
                            spk_name = speaker_names.get(spk, spk) if spk else ""
                            dialogue_lines.append(f'{spk_name}: "{text}"' if spk_name else f'"{text}"')
                            if spk:
                                speakers_in_clip.add(spk)

            # Characters on screen
            char_info = ""
            if speakers_in_clip and char_profiles:
                on_screen = [speaker_names.get(s, s) for s in speakers_in_clip]
                char_info = f" On screen: {', '.join(on_screen)}."

            dialogue_text = ""
            if dialogue_lines:
                dialogue_text = f" Dialogue: {' / '.join(dialogue_lines[:4])}"

            ctx = f"Shot {i + 1}: {label}, {duration:.1f}s.{char_info}{dialogue_text}"
            clip_contexts.append(ctx)

        # Build full transcript for context
        full_transcript = ""
        if lyrics:
            lines = []
            for l in lyrics:
                spk = l.get("speaker", "")
                text = l.get("text", "")
                if text.strip():
                    spk_name = speaker_names.get(spk, spk) if spk else ""
                    t_start = l.get("start", 0)
                    t_end = l.get("end", 0)
                    prefix = f"[{t_start:.1f}-{t_end:.1f}s] {spk_name}: " if spk_name else f"[{t_start:.1f}-{t_end:.1f}s] "
                    lines.append(f"{prefix}{text}")
            full_transcript = "\n".join(lines)

        # Call LLM
        char_rules = build_character_rules_block(has_reference, char_profiles if char_profiles else None)
        camera_block = build_camera_style_block()
        # Audio-driven mode also uses dialect-aware Pass 2 guides — see
        # _route_video_pass2_guide / get_image_prompt_rules for routing.
        video_model = getattr(self, '_video_model', '') or ''
        image_model = getattr(self, '_image_model', '') or ''
        video_guide = _route_video_pass2_guide(video_model)

        from ..image_prompt_rules import get_image_prompt_rules
        image_prompt_rules = get_image_prompt_rules(
            has_reference,
            num_character_refs=getattr(self, '_num_character_refs', 0),
            num_location_refs=getattr(self, '_num_location_refs', 0),
            character_ref_labels=getattr(self, '_character_ref_labels', None),
            location_ref_labels=getattr(self, '_location_ref_labels', None),
            seamless=getattr(self, '_seamless', True),
            image_model=image_model,
        )

        system_prompt = f"""You are a cinematic scene planner for a short film with dialogue audio. Output ONLY the JSON array.

{f"You are given a REFERENCE PHOTO of the characters. Use their visible appearance in all prompts." if has_reference else ""}

You are planning visuals for a scene where the AUDIO ALREADY EXISTS. The dialogue is pre-recorded.
Your job is to create compelling VISUALS that match the dialogue — environments, staging, camera work,
character actions, and facial expressions that bring the audio to life.

FULL DIALOGUE TRANSCRIPT:
{full_transcript if full_transcript else "(no transcript available)"}

STORY CONCEPT: {story_description}

Plan each shot as a structured scene — deciding visuals, camera, action, mood,
and how dialogue is staged. Write DETAILED video_prompt and image_prompt for each shot.

{char_rules}

{camera_block}

SHORT FILM PLANNING RULES:
- The audio is PRE-RECORDED — you are planning VISUALS to match existing dialogue.
- Focus on acting, body language, and emotional expression that matches what's being said.
- Stage dialogue naturally — characters should have physical business while speaking.
- Match camera complexity to emotional tone: steady for intimate, dynamic for action.
- Each shot should advance the story or reveal character.
- Describe the ENVIRONMENT in detail for each shot (room, furniture, lighting, time of day).
- video_prompt MUST be a full detailed paragraph (80-150 words) — NOT a brief label.
- image_prompt is the VERY FIRST FRAME — BEFORE any action in the video_prompt begins.
  It must be a FROZEN STILL PHOTOGRAPH — no motion, no action, no verbs of movement.
  Show the INITIAL STATE: if the scene involves removing clothing, the clothing is still ON.
  If a character enters the room, the room is EMPTY (or show whoever is already there).
  If something will be revealed, it is still hidden. The video_prompt describes the change.
  Include "create new scene, [environment]." at the start.

VIDEO PROMPT (video_prompt) — follow the LTX-2 style guide below closely:
- One single flowing paragraph, present tense, 4-8 sentences.
- Start with shot type and visual style early.
- Characters: describe by visible traits (clothing, hair, posture, expression).
- Emotion through PHYSICAL CUES only (jaw tightens, fists clench, shoulders drop) — never abstract labels like "serious expression" or "looks determined".
- Action: chronological order — setup, movement, reaction, final beat.
- Camera: explicit movement tied to the subject (slow dolly in, tracking left, orbit around, handheld follow) — never vague ("digital drift", "cinematic camera").
- Audio: include ambient sound when relevant, and any other sounds or sound effects that are relevant to the scene.
- Dialogue: in quotes with delivery cue if present.
- NEVER say montage, quick cuts, cut to.
- NEVER use character names in prompts — use "[age/role descriptor] from the reference image" (e.g. "the teen boy from the reference image", "the elderly woman from the reference image"). Preserve the age/role from the screenplay — do NOT normalize "teen boy" to "man". Names are ONLY allowed inside quoted dialogue.

{image_prompt_rules}

REFERENCE — LTX-2 video style guide:
{video_guide if video_guide else "(no guide loaded)"}

OUTPUT FORMAT — respond with ONLY a JSON array:
[
  {{
    "scene_goal": "What this shot achieves in the story",
    "scene_type": "dialogue|action|opening|closing|reaction",
    "subjects_on_screen": [
      {{"visual_description": "the woman in the white coat", "position_or_relation": "foreground left"}}
    ],
    "spatial_setup": "How subjects are arranged",
    "environment": "Setting description",
    "visual_style": "Visual look",
    "lighting": "Lighting description",
    "mood": "Emotional tone",
    "action_beats": ["Physical actions in chronological order"],
    "dialogue_beats": [
      {{"speaker_id": "char_0", "spoken_text": "Actual dialogue", "delivery": "softly", "physical_cue": "leans forward"}}
    ],
    "camera_plan": {{
      "framing": "medium shot",
      "movement": "slow push in",
      "movement_intensity": "subtle"
    }},
    "audio_plan": {{
      "mode": "dialogue_driven",
      "lip_sync_critical": true
    }},
    "ending_beat": "Final visual moment",
    "image_source": "original or previous — original=edit from user's reference photo, previous=edit from last scene's output (use for same-location continuity)",
    "image_prompt": "FIRST FRAME BEFORE action — initial state, static pose, environment. No motion verbs.",
    "visual_changes": ["what visually transforms during this scene — e.g. 'shirt is removed', 'man enters from doorway'"],
    "video_prompt": "Full flowing paragraph for video generation — describes the action AFTER the start frame...",
    "keyframe_prompts": ["(OPTIONAL) only if the video model needs visual info it can't generate from start image"],
    "window_prompts": ["(OPTIONAL) Window 1 — first ~20s of action...", "Window 2 — next ~20s, continues from where window 1 ends..."]
  }}
]

- image_source: "original" = edit from user's reference photo (default). "previous" = edit from last scene's
  output (for same-location continuity). First scene must always be "original".
- FIELD ORDER: Write image_prompt FIRST (starting state), then visual_changes, then video_prompt.
  image_prompt shows the BEFORE state. visual_changes lists what transforms. video_prompt describes the action.
- visual_changes: If it says "shirt removed", image_prompt must show shirt still ON.
- keyframe_prompts: Only when the video model needs visual info it can't generate from the start image.

WINDOW PROMPTS vs VIDEO PROMPT — use ONE or the OTHER, never both:
- Scenes 20s or under: write video_prompt, leave window_prompts as [].
- Scenes over 20s: write window_prompts, leave video_prompt as "".
  Each window covers ~20s. Windows play SEQUENTIALLY — window 2 continues exactly
  where window 1 left off, picking up the action mid-flow.
  CRITICAL: The video model only sees the last few frames — it has NO memory of
  earlier action or sound. Each window must briefly re-establish ongoing state
  (e.g. "the audience continues cheering" or "rain still falling") before
  describing new action. Without this, ongoing activity abruptly stops.
  Example: Window 1 delivers the joke → Window 2: "The audience continues laughing
  and clapping. She takes a bow, wipes her brow, and walks to stage left..."
Output exactly {len(clips)} shot plans. Go:"""

        # Inject model-specific prompt polish guide if provided
        if polish_block:
            system_prompt = f"{system_prompt}\n\n{polish_block}"

        # Mature-mode guidance is now SELF-GATING: the version-controlled
        # clinical guides apply only when the scene is actually sexual and tell
        # the model to write normally otherwise, so the block can be injected
        # whenever mature mode is on without harming clean scenes. This replaced
        # the old keyword pre-scan, which depended on an explicit wordlist that
        # cannot live in the version-controlled repo and missed scenes phrased
        # without its keywords.
        effective_nsfw = nsfw
        system_prompt = inject_nsfw_if_enabled(system_prompt, effective_nsfw, "both")
        # Note: audio mode doesn't load keyframe_rules.md as a separate
        # block (the keyframe guidance is inlined in the output spec
        # below).

        # `/no_think` prefix suppresses Qwen3 internal reasoning for this turn
        # — see story-mode pass 2 for full rationale. Pass 2 is structured-JSON
        # planning where thinking adds no creative value and on Qwen3.6-27B
        # has been observed to spiral. The marker is enforced by Qwen's Jinja
        # template directly, bypassing the broken `enable_thinking` kwarg path.
        user_prompt = f"""/no_think

TASK: Plan visuals for each of these {len(clips)} dialogue segments. Output exactly {len(clips)} shot plans — no more, no less.

CRITICAL OUTPUT REQUIREMENTS:
- Output EXACTLY {len(clips)} shots, one per audio clip below
- The audio is already recorded — write video_prompt and image_prompt that bring each segment to life visually
- Use keyframes ONLY when the video model needs visual info not in the start image; the model handles dialogue, gestures, and expressions on its own

Shots to plan:
{chr(10).join(clip_contexts)}"""

        image_paths = self._build_all_image_paths(reference_image_path, has_reference)
        # Each shot needs ~500 tokens for structured JSON + video_prompt + image_prompt.
        # `/no_think` above suppresses Qwen thinking. `thinking_budget=None`
        # delegates to _call_llm_json's model-aware default: Qwen→0 (off),
        # Gemma→4096 (on, to help small Gemma models follow structured-output
        # rules like the strict 20s window threshold).
        max_tokens = max(8192, len(clips) * 1600 + 4096)

        # Grammar constraint (applies on thinking-off models' first attempt
        # + everyone's retry — see _call_llm_json). minItems == maxItems ==
        # len(clips) makes the "output EXACTLY {len(clips)} shots" rule
        # grammar-enforced, not just prompted: the model cannot close the
        # array early or run past the clip count. keyframe_prompts /
        # window_prompts stay optional (spec tags them OPTIONAL).
        audio_schema = _shot_list_schema(
            min_items=len(clips),
            max_items=len(clips),
            required=[
                "scene_goal", "scene_type", "subjects_on_screen",
                "spatial_setup", "environment", "visual_style", "lighting",
                "mood", "action_beats", "dialogue_beats", "camera_plan",
                "audio_plan", "ending_beat", "image_source", "image_prompt",
                "visual_changes", "video_prompt",
            ],
        )

        shot_dicts = self._call_llm_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            thinking_budget=None,
            image_paths=image_paths,
            json_schema=audio_schema,
        )

        return self._convert_audio_shots(shot_dicts, clips, char_profiles, has_reference)

    def _convert_audio_shots(
        self,
        shot_dicts: list[dict],
        clips: list[dict],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
    ) -> list[ShotPlan]:
        """Convert LLM output to ShotPlan objects for audio-driven mode."""
        shots = []
        for i, clip in enumerate(clips):
            raw = shot_dicts[i] if i < len(shot_dicts) else {}
            duration = clip.get("end", 0) - clip.get("start", 0)

            subjects = [SubjectRef.from_dict(s) if isinstance(s, dict) else SubjectRef(visual_description=str(s))
                        for s in raw.get("subjects_on_screen", [])]

            cam_raw = raw.get("camera_plan", {})
            camera = CameraPlan(
                framing=cam_raw.get("framing", "medium shot"),
                angle=cam_raw.get("angle"),
                movement=cam_raw.get("movement"),
                movement_intensity=cam_raw.get("movement_intensity", "subtle"),
            )

            audio_raw = raw.get("audio_plan", {})
            audio = AudioPlan(
                mode=audio_raw.get("mode", "dialogue_driven"),
                ambience=audio_raw.get("ambience"),
                timing_anchor="audio",
                lip_sync_critical=audio_raw.get("lip_sync_critical", True),
            )

            dialogue_beats = None
            if raw.get("dialogue_beats"):
                dialogue_beats = [DialogueBeat.from_dict(db) for db in raw["dialogue_beats"]]

            shot = ShotPlan(
                shot_id=self._make_shot_id(i, "sf"),
                index=i,
                duration_sec=duration,
                skill_type="short_film",
                scene_goal=raw.get("scene_goal", f"Shot {i + 1}"),
                scene_type=raw.get("scene_type", "dialogue"),
                source_mode_preference="a2v" if audio_raw.get("lip_sync_critical") else ("i2v" if has_reference else "t2v"),
                image_strategy="reference_edit" if has_reference else "fresh_generation",
                continuity_strategy="continuous" if i > 0 else "independent",
                subjects_on_screen=subjects,
                spatial_setup=raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                visual_style=raw.get("visual_style", ""),
                lighting=raw.get("lighting", ""),
                mood=raw.get("mood", ""),
                action_beats=raw.get("action_beats", []),
                dialogue_beats=dialogue_beats,
                camera_plan=camera,
                audio_plan=audio,
                ending_beat=raw.get("ending_beat", ""),
                metadata={
                    "clip_start": clip.get("start", 0),
                    "clip_end": clip.get("end", 0),
                },
                # LLM-generated prompts (used directly, skipping renderer pass 2)
                video_prompt=raw.get("video_prompt"),
                image_prompt=raw.get("image_prompt"),
                window_prompts=raw.get("window_prompts"),
                visual_changes=raw.get("visual_changes"),
                image_source=raw.get("image_source"),
                keyframe_prompts=raw.get("keyframe_prompts"),
            )
            shots.append(shot)

        return shots

    # ── Story-Driven Planning ────────────────────────────────────────

    def _plan_story_driven(
        self,
        story_description: str,
        reference_image_path: Optional[str],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        target_duration: int,
        target_scenes: Optional[int],
        narrative_mode: bool,
        fps: int,
        frames_steps: int,
        frames_minimum: int,
        nsfw: bool = False,
        polish_block: str = "",
        multishot_lora_mode: bool = False,
    ) -> tuple[list[ShotPlan], Optional[str]]:
        """Two-pass story-driven planning.

        Pass 1 — Screenplay: LLM writes the full story as a flowing script.
        Pass 2 — Shot breakdown: LLM converts the screenplay into minimum shots with prompts.

        Args:
            multishot_lora_mode: When True, Pass 2 emits storyboard-format
                video_prompts for medium-length shots (20-30s) suitable
                for IC-LoRA-trained multi-shot models (Maque AI LTX-2.3
                IC-LoRA and similar). Short reaction shots (≤15s) and
                long sustained shots (40s+) keep the regular flowing
                video_prompt format.
        """
        from ..nsfw_guidance import inject_nsfw_if_enabled
        from ..safety_scan import (
            assert_no_minor_content,
            collect_pass2_text,
        )

        if target_scenes is None:
            target_scenes = max(2, min(20, target_duration // 20))

        image_paths = self._build_all_image_paths(reference_image_path, has_reference)

        # ── PRE-PASS-1 SAFETY SCAN: user concept ────────────────────────
        # Scan the user's input concept BEFORE running Pass 1. Catches
        # obviously-prohibited concepts ~30s earlier and avoids burning
        # an LLM call on something we'll abort anyway. Same scanner /
        # same hybrid co-occurrence policy as the post-Pass-1 check.
        assert_no_minor_content(story_description, source="user concept")

        # ── PASS 1: Screenplay ───────────────────────────────────────────
        print("[ShortFilmPlanner] Pass 1: Writing screenplay...")

        story_guide = ""
        if narrative_mode:
            story_guide = self._load_guide("Expert short-form storyteller.md")

        narrative_block = ""
        if narrative_mode and story_guide:
            narrative_block = f"""\nNARRATIVE GUIDE:\n{story_guide}\n
Structure the story with: setup, rising conflict, climax, resolution."""

        char_block = ""
        if char_profiles:
            char_lines = []
            for c in char_profiles:
                char_lines.append(f"- {c.id}: {c.physical_description}")
            char_block = (
                f"\nCHARACTERS (use appearance descriptions in the screenplay — "
                f"names are allowed ONLY in dialogue):\n"
                + "\n".join(char_lines)
                + "\n\nNOTE on character descriptions: the descriptions above are "
                "REFERENCE-PHOTO descriptions — they describe how each person LOOKS "
                "in the photo the user uploaded. They are an IDENTITY hint (face, "
                "build, gender) for the image generator. The actual STORY may "
                "transform these characters into other roles (a 'man in black' "
                "from the reference photo can become a knight in armor, a wizard, "
                "a CEO, a vampire — whatever the story needs). When you write the "
                "screenplay, describe characters as they appear IN THE STORY, not "
                "as they appear in the reference photo. The image generator will "
                "blend the reference face with the story's costume/role to render "
                "the transformed character correctly."
            )

        from ..guide_loader import load_guide
        screenplay_rules = load_guide("screenplay_writing_rules.md")

        # ── Hard length budget (CRITICAL) ────────────────────────────
        # The screenplay LLM consistently overshoots target duration —
        # observed in production: a 180s target produced a 358s
        # screenplay (~5.5 minutes of content). Without a concrete word
        # budget, "let scenes breathe" and "substantial dialogue"
        # guidance from screenplay_writing_rules.md compounds with the
        # LLM's natural tendency to elaborate, and Pass 2 inherits a
        # too-dense screenplay that no amount of consolidation can
        # actually fit.
        #
        # Math: at ~2 spoken words/sec, target_duration sets the
        # dialogue ceiling. Action lines add ~50% on top (they're
        # silent but they consume screen time).
        max_spoken_words = target_duration * 2  # 2 wps
        max_total_words = int(target_duration * 4.5)  # action + dialogue
        # Suggest a reasonable scene count window. Cinematic average is
        # ~10-25s/scene; we anchor at the wider end to prevent shot
        # explosion at Pass 2.
        scene_count_low = max(2, target_duration // 30)
        scene_count_high = max(scene_count_low + 1, target_duration // 15)

        length_budget_block = f"""
HARD LENGTH BUDGET — NON-NEGOTIABLE FOR THIS SCREENPLAY:
- Target duration: {target_duration} seconds.
- Maximum SPOKEN dialogue across the entire screenplay: {max_spoken_words} words.
  (At ~2 words/second, dialogue alone fills the runtime if you write more.)
- Maximum TOTAL screenplay length (dialogue + action lines + scene headings):
  approximately {max_total_words} words.
- Aim for {scene_count_low}-{scene_count_high} distinct scenes total.
  Fewer, fuller scenes always beat many short ones.

WHEN YOU NOTICE THE SCREENPLAY GETTING LONG — CUT, DON'T SPLIT:
- If you have written more than {max_spoken_words} words of dialogue, you are
  OVER BUDGET. Do NOT split into more scenes. Do NOT add a Pass 2
  consolidation step — there isn't one. Instead:
    * DROP a beat entirely (does the story actually need this exchange?).
    * SHORTEN a beat (one back-and-forth instead of three).
    * CONDENSE multi-line speeches into a single direct line.
- A {target_duration}-second film is SHORT. Pick the {scene_count_low}-{scene_count_high} most
  essential beats and write THOSE well. Save the rest for a longer cut.

WHY THIS MATTERS:
- Downstream Pass 2 splits the screenplay into shots; the video model
  generates each shot. If your screenplay implies 300 seconds of action,
  Pass 2 has TWO bad choices: (a) inflate total runtime to {int(target_duration * 1.7)}+
  seconds (overshoots user's target), or (b) cram 300s of content into
  {target_duration}s of shots (rushed, characters speak too fast, motion blurs).
  Both produce a worse film than a {target_duration}s screenplay paced for {target_duration}s.
"""

        pass1_system = f"""You are an acclaimed screenwriter celebrated for dialogue that sounds like real people actually talking — never stiff, formal, stagey, or "AI-like." You give every character a distinct, believable voice, and you fully commit to whatever tone, era, or style the concept calls for. Write a complete short film screenplay.

{f"You are given a REFERENCE PHOTO of the characters. Use their visible appearance in the script." if has_reference else ""}
{char_block}
{narrative_block}

{screenplay_rules}
{length_budget_block}"""

        if polish_block:
            pass1_system = f"{pass1_system}\n\n{polish_block}"
        pass1_system = inject_nsfw_if_enabled(pass1_system, nsfw, "screenplay")

        pass1_user = f"Write a short film screenplay based on this concept:\n\n{story_description}"

        # Repetition penalties are critical at Pass 1's scale (~18k token
        # output budget for a 180s film). Without them, models — especially
        # Qwen3.5/3.6 — can lock into a repetition cascade and generate
        # the same paragraph endlessly until the token budget runs out.
        # Phase 0.1 added stronger penalties (0.3 / 0.1) to Pass 2's
        # JSON output via _call_llm_json. Pass 1 is creative writing where
        # too much penalty hurts natural dialogue flow, so we use softer
        # values here — just enough to break repetition cascades without
        # discouraging legitimate word reuse in dialogue ("yes", "no",
        # character names, etc.).
        # Output token cap aligned to the word budget. Without this cap,
        # max_new_tokens defaulted to target_duration * 100 (180s →
        # 18000 tokens) which gave the LLM no signal to stop. User
        # reported a 1317-word screenplay against an 810-word budget
        # for a 180s target. Capping at ~3 tokens/word (generous for
        # English screenplay formatting) lets the LLM go ~50% over
        # budget before hitting the wall — a soft enforcement that
        # leaves room for the prompt-level guidance to do its job
        # without truncating mid-screenplay when the LLM lands close
        # to budget.
        #
        # The thinking_budget is independent — chain-of-thought
        # reasoning gets its own pool and doesn't count against this
        # cap.
        _output_token_cap = max(2000, max_total_words * 3)
        screenplay = self._generate_streaming(
            prompt=pass1_user,
            system_prompt=pass1_system,
            max_new_tokens=_output_token_cap,
            temperature=0.8,
            thinking_budget=16384,
            image_paths=image_paths or [],
            frequency_penalty=0.15,
            presence_penalty=0.05,
        )

        print(f"[ShortFilmPlanner] Screenplay: {len(screenplay)} chars")

        # ── Post-Pass-1 length warning ───────────────────────────────
        # Cheap word count to compare against the budget set in the
        # length_budget_block above. If we're over, we don't fail or
        # truncate — Pass 2 has its own duration constraints — but we
        # log so the user can see when Pass 1 ignored its budget.
        # Persistent over-budget output across runs is the signal that
        # the screenplay-LLM model is too aggressive for the budget
        # wording (consider switching models or temperature).
        _word_count = len(screenplay.split())
        if _word_count > max_total_words * 1.15:
            print(
                f"[ShortFilmPlanner] ⚠ Pass 1 over budget: {_word_count} words "
                f"(budget was {max_total_words}, +{_word_count - max_total_words}). "
                f"Pass 2 will compress; expect possible runtime overshoot."
            )
        else:
            print(
                f"[ShortFilmPlanner] Pass 1 word count: {_word_count} "
                f"(budget {max_total_words})"
            )

        # ── POST-PASS-1 SAFETY SCAN ─────────────────────────────────────
        # Catches anything the prompt-level prohibition rule failed to
        # prevent. Raises SafetyViolationError; pipeline error handler
        # in director_pipeline.py converts to a clean user-visible
        # message in chat.
        assert_no_minor_content(screenplay, source="screenplay (Pass 1)")

        if not screenplay or len(screenplay) < 50:
            print("[ShortFilmPlanner] Screenplay too short, falling back to single-pass")
            return self._plan_story_single_pass(
                story_description, reference_image_path, char_profiles,
                has_reference, target_duration, target_scenes, narrative_mode,
                fps, frames_steps, frames_minimum, nsfw, polish_block,
            )

        # ── PASS 2: Shot Breakdown ───────────────────────────────────────
        print("[ShortFilmPlanner] Pass 2: Breaking screenplay into shots...")

        char_rules = build_character_rules_block(has_reference, char_profiles if char_profiles else None)
        # video_guide now merged into ltx2_shot_breakdown.md — no separate load needed

        # Pull video/image model identifiers from the planner kwargs.
        # These flow from director_pipeline.py's planner_kwargs and let
        # us route Pass-2 guides correctly (LTX-2 video gets LTX-2
        # shot breakdown, Flux.2 Klein image gets Flux Pass-2 rules,
        # etc.) rather than always loading the legacy hardcoded files.
        video_model = getattr(self, '_video_model', '') or ''
        image_model = getattr(self, '_image_model', '') or ''

        from ..image_prompt_rules import get_image_prompt_rules
        image_prompt_rules = get_image_prompt_rules(
            has_reference,
            num_character_refs=getattr(self, '_num_character_refs', 0),
            num_location_refs=getattr(self, '_num_location_refs', 0),
            character_ref_labels=getattr(self, '_character_ref_labels', None),
            location_ref_labels=getattr(self, '_location_ref_labels', None),
            seamless=getattr(self, '_seamless', True),
            image_model=image_model,
        )

        # Load all guide content from .md files. Video shot-breakdown
        # currently routes only to LTX-2 vs. a generic fallback —
        # other model families share the LTX-2 rules until per-model
        # Pass-2 video guides land in Phase 3.
        shot_structure = load_guide("shot_structure_rules.md")
        video_rules = _route_video_pass2_guide(video_model)

        # Mature-mode guidance is self-gating (see audio-mode pass 2): the
        # version-controlled clinical guides apply only to scenes that are
        # actually sexual, so the block is injected whenever mature mode is on.
        # (Replaces the old explicit-keyword pre-scan, which can't be version-
        # controlled and missed scenes phrased without its keywords.)
        effective_nsfw = nsfw

        # Load keyframe guide — keyframes handle state changes (clothing, entering room)
        # not just long scenes, so always load the guide
        keyframe_note = load_guide("keyframe_rules.md") or "keyframe_prompts: use when the scene involves a visible state change (character enters, clothing removed, moves to new position)."

        pass2_system = f"""You are a film director breaking a screenplay into shots. Output ONLY the JSON array.

{char_rules}

{shot_structure}

{keyframe_note}

{video_rules}

{image_prompt_rules}


OUTPUT — respond with ONLY a JSON array:
[
  {{
    "title": "Shot title",
    "duration_sec": 20,
    "scene_goal": "What this shot achieves",
    "narrative_role": "setup|rising_action|climax|resolution",
    "scene_type": "dialogue|action|opening|closing",
    "subjects_on_screen": [{{"visual_description": "woman in red", "character_id": "char_0", "speaker_name": "Nancy"}}],
    "environment": "Setting details",
    "visual_style": "Style",
    "lighting": "Lighting",
    "mood": "Tone",
    "action_beats": ["Action 1", "Action 2"],
    "camera_plan": {{"framing": "medium shot", "movement": "slow push in", "movement_intensity": "subtle"}},
    "ending_beat": "Final moment",
    "image_source": "original or previous",
    "image_prompt": "FIRST FRAME BEFORE action begins — the starting visual state. Static pose, environment, lighting. No motion verbs.",
    "visual_changes": ["list what visually transforms — e.g. 'character removes jacket', 'new person enters room', 'camera reveals second character'"],
    "video_prompt": "Full detailed paragraph describing action AFTER the start frame — MUST include ALL dialogue in quotes with delivery cues. Physical actions, camera movement, atmosphere.",
    "multishot": false,
    "keyframe_prompts": ["(OPTIONAL) only when model needs visual info it can't generate — new character appearance, clothing underneath, etc."],
    "window_prompts": []
  }}
]

- multishot: false by default. Set true when the MULTI-SHOT LORA
  MODE block is in this system prompt (above) AND at least one of
  this shot's generations (video_prompt for a 20s shot, or any entry
  in window_prompts for 40s+ shots) uses the storyboard Format B
  instead of the flowing Format A paragraph. The storyboard format
  is the "Shot 1 (Camera, Xs): ..." structured form that the IC-LoRA
  renders as internal camera cuts. The decision is per generation,
  not per shot — a 40s shot can have one storyboard window and one
  flowing window, in which case multishot still equals true.

- subjects_on_screen[i].visual_description: describes how the character LOOKS
  IN THIS SHOT per the SCREENPLAY's depiction — NOT the user's reference photo
  description from the character profile above.

  The character profile (e.g. "char_1: man in black") is an IDENTITY hint —
  it tells the image model whose face to use as reference. But the screenplay
  may TRANSFORM the character: "the man in black" might become "a knight in
  gleaming silver armor", "a wizard in tattered grey robes", "a vampire in a
  velvet cloak". When that happens, visual_description MUST reflect the
  in-story appearance, not the original reference description.

  The image model uses the reference photo for face/identity AND the
  visual_description for costume/state — together they produce "a man with
  the user's face wearing silver armor." If you write "man in black" instead
  of "knight in silver armor", the image model will keep the character in
  black and the audience never sees the transformation the screenplay wrote.

  Same rule applies WITHIN a story: if Shot 5 has the character in armor and
  Shot 8 has them in a tunic later that night, Shot 8's visual_description
  should say "tunic", not the reference's "black shirt".

  Examples:
    User profile: "char_1: man in black"
    Screenplay says character is a knight →
      Shot 1 visual_description: "tall man in gleaming silver plate armor"
      Shot 5 (tavern, armor off) visual_description: "tall man in linen tunic"
    User profile: "char_0: woman with brown hair"
    Screenplay says character is a queen →
      Shot 1 visual_description: "regal woman in flowing emerald gown with circlet"

- subjects_on_screen[i].speaker_name: REQUIRED when the screenplay calls a character
  by a personal name. Record the EXACT name the screenplay uses for this character in
  this shot (e.g. "Nancy", "Blaine"). The downstream prompt-polish layer uses this to
  substitute the screenplay-invented name with the visual descriptor everywhere it
  appears in narrative prose. Without it, names like "Blaine" leak into video and
  image prompts where the generation model has no idea who that is. If the character
  has no spoken name in the screenplay (background extra, unnamed character), set to
  null or omit the field.
- image_source: "original" (default) = edit from user's uploaded reference photo. Use for most scenes.
  "previous" = edit from the previous scene's generated output. Use when scenes share the same location
  and you need visual continuity (same room state, character positions, cumulative changes like wet
  clothing or a messy room). The system sends BOTH the original reference AND the previous scene's
  output to the image model, so character identity is preserved while scene state carries forward.
  First scene must always be "original".
- FIELD ORDER MATTERS: Write image_prompt FIRST (the starting state), then visual_changes
  (what transforms), then video_prompt (the action). This ensures the start frame shows the
  BEFORE state, not the end result.
- visual_changes: List every visible transformation in the scene. This helps you write image_prompt
  as the BEFORE state. If visual_changes says "shirt is removed", then image_prompt must show
  the shirt still ON. If it says "man enters room", image_prompt shows the room WITHOUT the man.
  If visual_changes is empty, the scene is purely dialogue/expression — no keyframes needed.
- keyframe_prompts: DEFAULT IS EMPTY. Most shots — including 60s+ multi-window scenes — need
  ZERO keyframes. The video model animates motion, expressions, dialogue, camera movement,
  body language, and pacing on its own from the start image + prompt. Keyframes are only for
  visual information the model literally cannot invent.
  Ask yourself: "is there something visible here that wasn't visible in the start image AND
  cannot be reasonably inferred from existing references?" If no → no keyframe.
  KEYFRAME REQUIRED (rare):
    * A new named character appears mid-shot who is NOT in the start image and NOT in the
      character refs (so the model has nothing to copy their face from).
    * Clothing is removed and we need to render skin/anatomy that wasn't visible.
    * A specific end-state the model can't infer from the prompt alone (e.g. a written letter
      that needs to be legible at the end).
  KEYFRAME FORBIDDEN (do not add for any of these):
    * Walking, sitting down, standing up, turning, leaning, reaching, pointing, gesturing.
    * Talking, mouthing words, dialogue delivery, lip-sync.
    * Facial expressions changing (smile to frown, surprise, anger, tears).
    * Camera movement (push-in, pan, orbit, tilt) — the prompt drives this.
    * Lighting shifts that the prompt describes ("clouds pass overhead", "lamp turns on").
    * Mood / tone / atmosphere transitions.
    * "Animating" or "advancing" the action — the video model already does this.
  Each keyframe edits from the start image — describe only the specific visual change.
- window_prompts vs. video_prompt is determined by duration_sec ALONE.
  Use this STRICT decision (no soft zone, no "around 20"):
    duration_sec ≤ 20  → video_prompt populated, window_prompts MUST be []
    duration_sec ≥ 21  → window_prompts populated, video_prompt MUST be ""
  Every shot uses EXACTLY one of the two — never both, never neither.
  21s, 22s, 25s ALL count as "≥ 21" → these MUST use window_prompts.
  Window count for ≥ 21s shots:
    21-40s → 2 windows
    41-60s → 3 windows
    61-80s → 4 windows
  Each window covers ~20s of video. Windows play SEQUENTIALLY — window 2
  continues exactly where window 1 left off, picking up the action mid-flow.
  The video model only sees the last few frames between windows — re-establish
  ongoing state (crowd cheering, rain falling, music playing) at the start
  of each window.
- Each window prompt MUST be a full detailed paragraph (80-150 words).
  Do NOT reuse the same prompt for multiple windows — each window describes
  a different portion of the scene's action chronologically.
NAME CONVERSION — the screenplay may use character names, but prompts MUST NOT:
- Replace every character name with their descriptor + "from the reference image".
  PRESERVE the age/role descriptor from the screenplay — do NOT normalize to "man"/"woman".
  "teen boy Tommy" → "the teen boy from the reference image"
  "elderly Mrs. Chen" → "the elderly woman from the reference image"
  "Dr. Ava" → "the female doctor from the reference image"
  "little girl Sarah" → "the young girl from the reference image"
- Names are ONLY allowed inside quoted dialogue in video_prompt.
- NOT "Ava looks annoyed" → YES "the woman from the reference image looks annoyed".

PACING — match shot length to story beat, not to a "preferred" average:
- Total duration must sum to ~{target_duration}s.
- KEEP CONVERSATIONS TOGETHER. If two characters are mid-exchange, that is ONE shot — do NOT cut
  mid-conversation into separate shots. A 40s dialogue is one 40s scene with window_prompts,
  not three 13s scenes. Cutting mid-dialogue forces new start images, breaks character
  consistency, and wastes generation time.
- Cut to a new shot when ANY of these is true: location changes, a new character enters,
  a significant time jump, a clear story beat ends and a new one begins, OR a brief reaction
  is the entire dramatic point of the moment.
- Shot-length menu (use the whole range — variety is good filmmaking, not bias toward long):
    * 3-8s   — single reaction, glance, visual punctuation, establishing detail
    * 6-15s  — brief action, transition, short establishing shot
    * 15-40s — dialogue exchange, focused continuous action (one or two windows)
    * 40-80s — sustained scene (multiple windows, conversation that earns its length)
- STRICT 20s threshold: shots ≤ 20s use a single video_prompt with window_prompts=[];
  shots ≥ 21s use window_prompts (one per ~20s slice) with video_prompt="".
  21s, 22s, 25s ALL require window_prompts — there is no "soft zone".
Go:"""

        # ── Multi-shot LoRA mode injection ───────────────────────────
        # When the user has enabled multi-shot LoRA mode (a toggle in
        # services config; defaults off), Pass 2 gets supplementary
        # guidance for mixed-format output.
        #
        # Architecture (revised after first user test):
        # The unit of decision is the GENERATION, not the shot. A
        # generation is one LTX-2 call producing ≤20s of video. Mapping:
        #   - 20s shot = 1 generation (the video_prompt itself)
        #   - 40s shot = 2 generations (each window_prompt is one)
        #   - 60s shot = 3 generations (each window_prompt is one)
        #   - 80s shot = 4 generations (each window_prompt is one)
        #
        # For EACH generation independently, the LLM picks one of two
        # formats:
        #   1. SINGLE-CAMERA FLOWING (default): a flowing paragraph
        #      describing one continuous take.
        #   2. STORYBOARD MULTI-SHOT: a series of "Shot N (Camera, Xs):
        #      description" blocks describing internal camera cuts that
        #      the IC-LoRA will render within the single generation.
        #
        # When to use storyboard: dialogue exchanges, multi-beat
        # interaction, scenes where camera variety helps. When to keep
        # flowing: sustained single beats (a kiss, a sex act, a held
        # reaction), punchy moments, ambient establishing shots.
        #
        # Each window in a 40s+ shot can use a different format —
        # window 1 might be storyboard (dialogue) while window 2 is
        # flowing (the kiss that follows). The decision is per
        # generation.
        if multishot_lora_mode:
            _multishot_block = (
                "\n\n"
                "═══════════════════════════════════════════════════════\n"
                "MULTI-SHOT LORA MODE — USE FORMAT B FOR DIALOGUE\n"
                "═══════════════════════════════════════════════════════\n\n"

                "AN IC-LORA IS LOADED. It renders internal camera cuts "
                "inside one ~20s generation IF you write the prompt in "
                "Format B (storyboard structure). If you write Format A "
                "(flowing prose), the LoRA produces one camera angle and "
                "is doing nothing useful. For a dialogue-heavy film, "
                "Format B should be the default — Format A is the "
                "EXCEPTION for sustained beats.\n\n"

                "FORMAT B — STORYBOARD (default for dialogue/interaction):\n"
                "  Shot 1 (Wide Shot, 5s): description of action this angle.\n"
                "  Shot 2 (Medium Shot, 7s): continuation in new angle.\n"
                "  Shot 3 (Close-up, 4s): continuation in another angle.\n"
                "  Shot 4 (Two-Shot, 4s): final angle of the ~20s slice.\n\n"

                "FORMAT A — FLOWING (only for sustained single beats):\n"
                "A normal flowing paragraph describing ONE continuous "
                "camera take. Use ONLY for: a kiss, a sex act, a held "
                "reaction, a slow push-in — beats that would be RUINED "
                "by camera cuts.\n\n"

                "RULES THAT DO NOT CHANGE:\n"
                "1. The duration→field rule is unchanged. 20s shots use "
                "video_prompt; 40s/60s/80s shots use window_prompts "
                "(one entry per 20s). Format A/B is the CONTENT inside "
                "each field, never which field is populated. Putting "
                "Format B inside video_prompt of a 40s shot triggers "
                "snap-down to 20s and loses content.\n"
                "2. Camera type parens contain ONLY the shot type, "
                "never a character name. 'Close-up', not 'Close-up on "
                "Henry'. Names go inside dialogue quotes in the "
                "description text.\n"
                "3. Two-Shot and Over-the-Shoulder REQUIRE two "
                "characters on screen. For solo moments use "
                "Wide/Medium/Close-up.\n"
                "4. Internal shot durations sum to ~20s; each one is "
                "3-8 seconds; 2-5 internal shots per 20s generation.\n\n"

                "CAMERA TYPES: Wide Shot, Medium Shot, Medium Close-up, "
                "Close-up, Extreme Close-up, Two-Shot, Over-the-Shoulder, "
                "Side Shot, Overhead, Low Angle.\n\n"

                "EXAMPLE — 20s dialogue, Format B:\n"
                "  video_prompt: \"Shot 1 (Wide Shot, 5s): The woman in "
                "russet dress steps onto the porch. Shot 2 (Medium Shot, "
                "7s): The man in cowboy hat turns toward her. He says, "
                "'You're back early.' Shot 3 (Close-up, 4s): Her hand "
                "rests on the railing. Shot 4 (Two-Shot, 4s): She nods.\"\n"
                "  window_prompts: []\n"
                "  multishot: true\n\n"

                "EXAMPLE — 40s dialogue, BOTH windows in Format B:\n"
                "  video_prompt: \"\"\n"
                "  window_prompts: [\n"
                "    \"Shot 1 (Wide Shot, 5s): The woman stands at the "
                "porch railing. Shot 2 (Over-the-Shoulder, 8s): The man "
                "approaches from behind. Shot 3 (Close-up, 7s): He says, "
                "'Sun's setting.' She replies, 'I noticed.'\",\n"
                "    \"Shot 1 (Medium Shot, 6s): They stand close. Shot 2 "
                "(Side Shot, 7s): The man turns his head. He says, 'Stay "
                "a while.' Shot 3 (Close-up, 7s): She tilts her chin up. "
                "She replies, 'I'm not going anywhere.'\"\n"
                "  ]\n"
                "  multishot: true\n\n"

                "EXAMPLE — 40s mixed (dialogue then sustained kiss):\n"
                "  video_prompt: \"\"\n"
                "  window_prompts: [\n"
                "    \"Shot 1 (Medium Shot, 6s): He leans in. He whispers, "
                "'The flame needs kindling.' Shot 2 (Close-up, 7s): Her "
                "breathing hitches. Shot 3 (Two-Shot, 7s): Her hands rest "
                "flat on his chest.\",\n"
                "    \"A slow push-in on the two embracing. He wraps his "
                "arms around her shoulders. The kiss deepens. The camera "
                "holds steady as the light fades to amber.\"\n"
                "  ]\n"
                "  multishot: true   # true because window 1 uses Format B\n\n"

                "EXAMPLE — 20s sustained shot, Format A:\n"
                "  video_prompt: \"A slow push-in on the embracing couple. "
                "Their lips press together. He cups her jaw. The camera "
                "holds steady as the kiss deepens.\"\n"
                "  window_prompts: []\n"
                "  multishot: false\n\n"

                "═══════════════════════════════════════════════════════\n"
                "EXPECTATION: 60-80% of generations should be FORMAT B. If "
                "your final output has ZERO Format B generations on a "
                "script with dialogue, you have UNDERUSED the LoRA and the "
                "user paid for it to do nothing. Re-plan: every window "
                "containing dialogue or character interaction MUST be "
                "Format B. Only sustained beats stay Format A.\n"
                "═══════════════════════════════════════════════════════\n"
            )
            pass2_system = f"{pass2_system}{_multishot_block}"

        if polish_block:
            pass2_system = f"{pass2_system}\n\n{polish_block}"

        # `effective_nsfw` was computed above; reuse it for the
        # inject_nsfw_if_enabled call. The injected guides are
        # self-gating (apply only when a scene is actually sexual).
        pass2_system = inject_nsfw_if_enabled(pass2_system, effective_nsfw, "both")

        # Compute a permissive shot count range so the LLM has creative
        # freedom to match shot length to story beat. Earlier versions
        # used target//35..target//20 which forced a 60s film into 2-3
        # long shots — fine for sustained dialogue, terrible for
        # reaction beats and montages. New range: at least 2 shots
        # (no single-shot films), up to roughly target/8 (allowing a
        # mix of 4-8s reaction beats with longer scenes). The LLM
        # decides where on that spectrum each story sits.
        # Shot count guidance. The high cap is the single biggest lever
        # for forcing the LLM to use long buckets. Math:
        #
        # If shot_count_high = target / 25, the LLM CANNOT hit target
        # using only 20s shots — that would require more shots than the
        # cap allows (180s / 20 = 9, but cap is 7). To hit target, the
        # LLM is forced to mix in 40s/60s/80s buckets. This is the
        # only reliable way to get long shots; prompt-level "use long
        # buckets" guidance alone has been observed to be ignored
        # (latest user test: 10 × 20s = 200s for 180s target).
        #
        # - shot_count_low = target / 40: lower bound, allows long-shot-
        #   dominated films (e.g. 180s = 3 × 60s, 4 shots — though the
        #   floor of max(2, ...) usually wins for short targets).
        # - shot_count_high = target / 25: upper bound, forces long
        #   buckets when target requires it. 180s → 7. 300s → 12.
        #   60s → 2 max from formula but floor brings it to 4 via
        #   max(low+2, ...).
        #
        # The previous target/15 cap let 180s have 12 × 20s = 240s
        # (within accept-zone of 207s ceiling), so the LLM picked the
        # safe all-20s option. target/25 forces the math.
        shot_count_low = max(2, target_duration // 40)
        shot_count_high = max(shot_count_low + 2, target_duration // 25)

        # Pass 2 user prompt construction:
        # 1. /no_think at the top suppresses Qwen3 internal reasoning for
        #    this turn (enforced in Qwen's Jinja chat template directly).
        #    On Qwen3.6-27B, thinking has been observed to spiral into
        #    multi-thousand-token loops that exhaust the budget before
        #    producing actual output. /no_think bypasses the broken
        #    `enable_thinking` chat_template_kwarg path on some llama.cpp
        #    builds. Other models simply ignore the marker.
        # 2. Hard duration + shot-count constraint at the very top — this
        #    used to be buried at line ~643 of the system prompt, but the
        #    LLM ignored it under cognitive load. Hoisting to the user
        #    prompt's first paragraph anchors output structure decisively.
        # 3. The screenplay itself goes last so it remains in the model's
        #    most-recent attention window.
        # Multi-shot LoRA anchor injected into the POPULATION RULE
        # in pass2_user below. Empty string when multi-shot mode is
        # off; a short pointer when on. LLM weighs user-prompt rules
        # more heavily than system-prompt, so the storyboard-format
        # decision needs a visible mention here to avoid the LLM
        # cramming storyboard content into the wrong field (observed
        # production bug: 40s shots ended up with populated
        # video_prompt + empty window_prompts, then snap-down lost
        # half the runtime).
        multishot_user_anchor = (
            "   MULTI-SHOT LORA MODE: storyboard format goes INSIDE "
            "the field this rule says to populate. For a 40s shot, "
            "that means TWO entries in window_prompts, each "
            "independently formatted as storyboard OR flowing. NEVER "
            "put a storyboard inside video_prompt of a 40s+ shot — "
            "the system will snap the duration down to 20s. See the "
            "MULTI-SHOT LORA MODE block in the system prompt above "
            "for examples."
        ) if multishot_lora_mode else ""

        pass2_user = f"""/no_think

TASK: Break this {target_duration}-second screenplay into {shot_count_low}-{shot_count_high} distinct shots.

CRITICAL OUTPUT REQUIREMENTS (these override any conflicting system-prompt guidance):

1. EXACTLY {shot_count_low} TO {shot_count_high} SHOTS. No more. Going over this count
   means you're fragmenting — every shot under 20s is a sign you cut where
   the video model could have rendered continuous action. Re-merge.

2. SHOT DURATION MUST BE ONE OF: 20, 40, 60, 80 seconds.
   - 20s = single beat (a transition, a brief reaction, an
     establishing moment, a short dialogue exchange). One prompt,
     no windows.
   - 40s = TWO connected beats that flow together as one continuous
     scene (an extended dialogue, a foreplay-to-act transition, a
     slow reveal, a full kiss + embrace). USE FREELY — don't default
     to two 20s shots when the screenplay has a continuous 40s beat.
     Two windows.
   - 60s = THREE connected beats in one sustained scene (a long
     romantic encounter, a sex sequence, a confrontation that builds
     and breaks). Three windows. Common in NSFW films and any film
     with sustained dramatic scenes — don't avoid 60s shots.
   - 80s = FOUR connected beats in a single uninterrupted sequence
     (a sustained sex act, a long climactic confrontation, an extended
     seduction). Four windows. Use when the screenplay has a beat
     that genuinely needs the breathing room.
   - HEURISTIC: aim for variety. A {target_duration}s film with NINE
     20s shots feels choppy; a film with three 20s shots + two 40s +
     one 60s feels cinematic. Mix the bucket sizes.
   - NEVER 5, 8, 10, 15, 22, 25, 30, 35, 45, 50, 55, 65, 70, 75. Those
     all create stranded short tail windows that render as sluggish stubs.

3. TOTAL duration_sec MUST sum to {target_duration} seconds (±5%).
   With 20s shots that's exactly {target_duration // 20} shots. With one
   40s shot mixed in, the rest fit into {(target_duration - 40) // 20} 20s shots.

4. POPULATION RULE — single hard threshold (THIS RULE OVERRIDES THE
   MULTI-SHOT LORA MODE BLOCK BELOW IF YOU TRY TO BREAK IT):
   - duration_sec == 20 → populate video_prompt, window_prompts=[]
   - duration_sec ∈ {{40, 60, 80}} → populate window_prompts (one per 20s),
     video_prompt=""
   Each window is a full paragraph (80-150 words) describing 20s of action.
   {multishot_user_anchor}

5. THE VIDEO MODEL HANDLES INTRA-SHOT PROGRESSION. ONE 20s shot can show
   the woman walking closer, raising her hand to his chest, kneeling, and
   beginning a new action — the model renders all of that from a single
   prompt + start frame. You do NOT need separate shots for "she steps
   closer", "her hand moves", "she kneels", "she begins to..." — those
   are micro-beats, NOT shot boundaries.

   ONLY cut to a new shot when ONE of these changes:
     - LOCATION (different room, indoor↔outdoor)
     - TIME (skip ahead — "later that evening")
     - CAST (a new character enters / someone exits)
     - DRAMATIC PIVOT (clear emotional inflection)
   DO NOT cut for: position, gesture, expression, camera movement, or
   action progression within an ongoing scene.

WHEN THE SCREENPLAY IS TOO DENSE FOR {target_duration}s — DROP CONTENT, DON'T ADD SHOTS:
The user asked for {target_duration} seconds. If the screenplay implies more, do NOT
solve it by adding more shots or stretching duration_sec. Instead:
  * DROP whole beats from the screenplay (a transition, a redundant line).
  * MERGE adjacent beats into one shot — most multi-beat content fits in
    a single 20s shot's prompt.
  * SHORTEN dialogue (cut the second back-and-forth, condense speeches).
A {shot_count_high}-shot film at 20s each is {shot_count_high * 20}s. If your plan
exceeds that count or that total, you are fragmenting or over-budget — re-plan.

SHOT BOUNDARIES (do not overlap):
Each shot covers a distinct, NON-overlapping span of the screenplay's
timeline. If Shot 2 covers minute 0:00-0:30 of action, Shot 3 starts at 0:30
and never re-uses lines from Shot 2. Do NOT include the same dialogue
exchange across multiple shots.

The user's original request:
{story_description}

Shot-construction rules:
- KEEP CONTINUOUS ACTION TOGETHER — physical progression that flows from one beat to the next is ONE shot. See the WRONG/RIGHT examples above. The video model handles intra-shot action progression; do not fragment.
- KEEP CONVERSATIONS TOGETHER — one conversation = one shot, using window_prompts if over 20s.
- MIX BUCKET SIZES. Use 40s for connected dialogue/action pairs. Use 60s for long romantic / dramatic / sex scenes. Use 80s for genuinely sustained sequences. With only {shot_count_high} shots allowed total, you CANNOT hit {target_duration}s using only 20s — the math forces you to use longer buckets. That is intentional: longer buckets produce more cinematic, less choppy films.
- Only cut to a new shot when location changes, a new character enters, or there's a clear dramatic beat transition (see strict criteria above).
- Preserve ALL dialogue from the screenplay verbatim — but each line goes in EXACTLY ONE shot/window, never repeated.
- Use keyframes ONLY when the video model needs visual info it can't generate from the start image (new character entry, clothing reveal, dramatic state change). Do NOT use keyframes as a substitute for animating dialogue — the video model handles all talking, gestures, and expressions on its own.

SCREENPLAY:
{screenplay}"""

        # Budget based on duration — LLM decides scene count, so we estimate ~100 tokens/second
        # for structured JSON + video_prompt + image_prompt + keyframes + windows.
        # `/no_think` above suppresses Qwen thinking. `thinking_budget=None`
        # delegates to _call_llm_json's model-aware default (Qwen→0, Gemma→4096).
        # Gemma 4B specifically benefits from thinking when planning the strict
        # 20s window threshold and total-duration arithmetic.
        max_tokens = max(8192, target_duration * 100)

        # Grammar constraint (thinking-off models' first attempt + every
        # retry — see _call_llm_json). The shot-count bounds make the
        # prompt's "{shot_count_low}-{shot_count_high} shots" rule grammar-
        # enforced, and the closed shot object makes the observed failure
        # (Gemma 4 12B looping 96K chars of repeating shot pseudo-JSON)
        # unrepresentable. keyframe_prompts stays optional (spec tags it
        # OPTIONAL); window_prompts is required because the ≤20s/≥21s
        # pairing rule expects an explicit [] on short shots.
        pass2_schema = _shot_list_schema(
            min_items=shot_count_low,
            max_items=shot_count_high,
            required=[
                "title", "duration_sec", "scene_goal", "narrative_role",
                "scene_type", "subjects_on_screen", "environment",
                "visual_style", "lighting", "mood", "action_beats",
                "camera_plan", "ending_beat", "image_source", "image_prompt",
                "visual_changes", "video_prompt", "multishot",
                "window_prompts",
            ],
        )

        shot_dicts = self._call_llm_json(
            user_prompt=pass2_user,
            system_prompt=pass2_system,
            max_tokens=max_tokens,
            thinking_budget=None,
            image_paths=image_paths,
            json_schema=pass2_schema,
        )

        # ── POST-PASS-2 SAFETY SCAN ─────────────────────────────────────
        # Defense in depth — Pass 2's structured output (image/video
        # prompts, action beats, dialogue, subjects) gets concatenated
        # and scanned the same way the screenplay was. Catches the case
        # where Pass 1 produced clean text but Pass 2's expansion
        # introduced minor + sexual co-occurrence.
        assert_no_minor_content(
            collect_pass2_text(shot_dicts), source="shot list (Pass 2)"
        )

        # ── CHARACTER DESCRIPTOR CANONICALIZATION ────────────────────
        # User-reported bug: uploaded selfie tagged "man in black",
        # screenplay turned the character into a knight in silver armor,
        # but Pass 2 inconsistently described them — some shots said
        # "man in black" (the user's reference descriptor), others said
        # "knight in silver armor" (the in-story appearance). Result: the
        # image generator put the character in armor in some scenes and
        # back into a black shirt in others.
        #
        # Prompt-level guidance to use the in-story descriptor was added
        # in commit 9263c8a but the LLM still doesn't follow it
        # consistently. This is the deterministic safety net.
        #
        # Algorithm:
        # 1. For each character_id, collect every visual_description used
        #    across shots.
        # 2. Filter out descriptors that match the user's char_profile
        #    descriptor (case-insensitive) — those are the ones we want
        #    to REPLACE.
        # 3. Pick the most-common non-user descriptor as the "canonical
        #    in-story descriptor" for that character.
        # 4. Replace the user's descriptor with the canonical one in:
        #    - subjects_on_screen[i].visual_description
        #    - video_prompt
        #    - image_prompt
        #    - window_prompts entries
        #    - keyframe_prompts entries
        #
        # Only fires when the canonical descriptor appears in ≥2 shots —
        # if there's only a one-off transformation, the LLM may have
        # intended a one-shot variation (flashback, costume change) and
        # we should not force consistency.
        try:
            from collections import Counter as _Counter, defaultdict as _DefaultDict

            user_descriptors_by_cid: dict[str, str] = {}
            for c in (char_profiles or []):
                cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
                desc = (
                    getattr(c, "physical_description", None)
                    or (c.get("physical_description") if isinstance(c, dict) else None)
                    or ""
                )
                if cid and desc:
                    user_descriptors_by_cid[cid] = desc.strip().lower()

            descs_by_cid: dict[str, list[str]] = _DefaultDict(list)
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                for subj in (sd.get("subjects_on_screen") or []):
                    if not isinstance(subj, dict):
                        continue
                    cid = subj.get("character_id")
                    vd = (subj.get("visual_description") or "").strip()
                    if cid and vd:
                        descs_by_cid[cid].append(vd)

            canonical_by_cid: dict[str, str] = {}
            for cid, descs in descs_by_cid.items():
                user_desc = user_descriptors_by_cid.get(cid, "")
                if not user_desc:
                    continue
                non_user = [d for d in descs if d.strip().lower() != user_desc]
                if not non_user:
                    continue  # all match user descriptor — no transformation
                counter = _Counter(non_user)
                most_common, count = counter.most_common(1)[0]
                # Require ≥2 occurrences to consider it canonical.
                # Single-shot variations are likely intentional (flashback,
                # costume change) and should not be forced across the
                # whole production.
                if count >= 2:
                    canonical_by_cid[cid] = most_common

            if canonical_by_cid:
                import re as _re_can
                for cid, canonical in canonical_by_cid.items():
                    user_desc_raw = next(
                        (
                            (getattr(c, "physical_description", None)
                             or (c.get("physical_description") if isinstance(c, dict) else None))
                            for c in (char_profiles or [])
                            if (getattr(c, "id", None) == cid
                                or (isinstance(c, dict) and c.get("id") == cid))
                        ),
                        None,
                    )
                    if not user_desc_raw:
                        continue
                    user_desc_raw = user_desc_raw.strip()
                    pat = _re_can.compile(
                        r"\b" + _re_can.escape(user_desc_raw) + r"\b",
                        _re_can.IGNORECASE,
                    )
                    replacements = 0
                    for sd in shot_dicts:
                        if not isinstance(sd, dict):
                            continue
                        # subjects_on_screen
                        for subj in (sd.get("subjects_on_screen") or []):
                            if not isinstance(subj, dict):
                                continue
                            if subj.get("character_id") != cid:
                                continue
                            vd = (subj.get("visual_description") or "").strip()
                            if vd.lower() == user_desc_raw.lower():
                                subj["visual_description"] = canonical
                                replacements += 1
                        # text fields
                        for field in ("video_prompt", "image_prompt"):
                            text = sd.get(field) or ""
                            if text:
                                new_text, n = pat.subn(canonical, text)
                                if n:
                                    sd[field] = new_text
                                    replacements += n
                        # array text fields
                        for arr_field in ("window_prompts", "keyframe_prompts"):
                            arr = sd.get(arr_field) or []
                            if not isinstance(arr, list):
                                continue
                            new_arr = []
                            for item in arr:
                                if isinstance(item, str):
                                    new_item, n = pat.subn(canonical, item)
                                    if n:
                                        replacements += n
                                    new_arr.append(new_item)
                                else:
                                    new_arr.append(item)
                            sd[arr_field] = new_arr
                    if replacements:
                        print(
                            f"[ShortFilmPlanner] Canonicalized {cid} "
                            f"descriptor across {replacements} location(s): "
                            f"replaced reference description '{user_desc_raw}' "
                            f"with in-story description '{canonical}'. "
                            f"(LLM was inconsistent — some shots used the "
                            f"reference photo's description, others used "
                            f"the screenplay's transformed description; "
                            f"forcing the transformed one for consistency.)"
                        )
        except Exception as _canon_err:
            print(f"[ShortFilmPlanner] Descriptor canonicalization skipped: {_canon_err}")

        # ── POST-PASS-2 OVER-FRAGMENTATION MERGE ──────────────────────
        # When the LLM emits way more shots than the target shot-count
        # range (e.g. 36 shots for a 180s target where the range is
        # 6-12), merge adjacent short shots into single 20s shots.
        # Without this step, every short shot gets snap-up'd to 20s by
        # the per-shot post-process, ballooning the total runtime,
        # which then triggers the duration scale-down — and the result
        # is N tiny shots crammed into target seconds, the worst of
        # both worlds.
        #
        # Merge strategy: walk the shot list in order, accumulating
        # adjacent short shots (≤15s) into one merged shot until the
        # accumulated duration would exceed 20s. Concatenate their
        # video_prompts (with " " separator), drop their keyframes
        # (stale after merge), keep the FIRST shot's image_prompt and
        # subjects_on_screen (since the merged shot opens on that
        # frame). Boundary detection: stop accumulating when location
        # or scene_type changes — those are real shot boundaries even
        # in a fragmented run.
        try:
            _max_shots = max(2, target_duration // 15)  # generous ceiling
            if len(shot_dicts) > _max_shots * 1.3 and shot_dicts:
                pre_merge_count = len(shot_dicts)
                merged_shots: list[dict] = []
                bucket: list[dict] = []
                bucket_dur = 0

                def _flush_bucket():
                    nonlocal bucket, bucket_dur
                    if not bucket:
                        return
                    if len(bucket) == 1:
                        merged_shots.append(bucket[0])
                    else:
                        head = dict(bucket[0])
                        # Concatenate video_prompts in order, preserving
                        # each shot's intended action sequence.
                        prompts = []
                        for s in bucket:
                            vp = (s.get("video_prompt") or "").strip()
                            if vp:
                                prompts.append(vp)
                        if prompts:
                            head["video_prompt"] = " ".join(prompts)
                        head["window_prompts"] = []
                        head["duration_sec"] = 20
                        # Drop keyframes — they were placed for the
                        # original tiny shots and don't fit a single
                        # merged 20s shot.
                        head["keyframe_prompts"] = []
                        # Concatenate action_beats for downstream tools
                        # that read them.
                        all_beats: list = []
                        for s in bucket:
                            ab = s.get("action_beats") or []
                            if isinstance(ab, list):
                                all_beats.extend(ab)
                        if all_beats:
                            head["action_beats"] = all_beats
                        merged_shots.append(head)
                    bucket = []
                    bucket_dur = 0

                for sd in shot_dicts:
                    if not isinstance(sd, dict):
                        merged_shots.append(sd)
                        continue
                    dur = int(sd.get("duration_sec", 0) or 0)
                    has_windows = bool(sd.get("window_prompts"))
                    # Don't merge: long shots, multi-window shots, or
                    # shots that change location/scene-type from the
                    # bucket head.
                    is_short = (0 < dur <= 15) and not has_windows
                    boundary = False
                    if bucket and is_short:
                        head = bucket[0]
                        if (sd.get("environment") and head.get("environment")
                                and sd.get("environment") != head.get("environment")):
                            boundary = True
                        if (sd.get("scene_type") and head.get("scene_type")
                                and sd.get("scene_type") != head.get("scene_type")):
                            boundary = True
                    if not is_short or boundary:
                        _flush_bucket()
                        merged_shots.append(sd)
                        continue
                    # Would adding this shot push the bucket past 20s?
                    if bucket_dur + dur > 20 and bucket:
                        _flush_bucket()
                    bucket.append(sd)
                    bucket_dur += dur
                _flush_bucket()

                if len(merged_shots) < pre_merge_count:
                    print(
                        f"[ShortFilmPlanner] ⚠ Pass 2 over-fragmented: "
                        f"{pre_merge_count} shots > {_max_shots} expected. "
                        f"Merged adjacent short shots → {len(merged_shots)} shots. "
                        f"Each merged shot's video_prompts concatenated; "
                        f"keyframes dropped (stale after merge)."
                    )
                    shot_dicts[:] = merged_shots
        except Exception as _merge_err:
            print(f"[ShortFilmPlanner] Adjacent-shot merge skipped: {_merge_err}")

        # ── POST-PASS-2 DURATION ENFORCEMENT ─────────────────────────
        # User-reported lesson from production: scaling 20s shots down
        # to 17-18s "to hit the exact target" is pointless. The user
        # would rather have clean 20-second buckets and slightly miss
        # the runtime target than hit the runtime exactly with awkward
        # mid-bucket durations that violate the model's window-
        # threshold rules.
        #
        # Three-tier policy:
        #
        # Tier 1 — accept (≤15% over):
        #   The LLM's overshoot is small enough to live with. Log it
        #   and move on. This handles the common case where Pass 1
        #   was a bit dense and Pass 2 ended at, say, 200s for a 180s
        #   target. User gets a 200s film with clean buckets — better
        #   than an exact 180s film with 17s shots.
        #
        # Tier 2 — bucket-aware reduction (15% to 50% over):
        #   Find shots in larger buckets (40/60/80s) and snap each
        #   down to the next-smaller bucket until total fits. Each
        #   snap removes exactly 20s of runtime AND one window of
        #   content (the last window of that shot). Preserves the
        #   bucket grid; the only "compression" is dropping content,
        #   not stretching it.
        #
        # Tier 3 — proportional fallback (>50% over):
        #   Runaway LLM. Apply proportional scale, then run a final
        #   snap-to-bucket cleanup that rounds each shot back to a
        #   valid bucket value (20/40/60/80). The result may exceed
        #   target after rounding — accepted as a known fail mode for
        #   pathological inputs.
        _raw_total = sum(
            int(sd.get("duration_sec", 0) or 0)
            for sd in shot_dicts
            if isinstance(sd, dict)
        )
        _ceiling = int(target_duration * 1.15)
        _scale_threshold = int(target_duration * 1.50)

        def _snap_bucket(sd: dict) -> None:
            """Snap a single shot's duration_sec to nearest valid bucket
            and align window_prompts/video_prompt accordingly. Idempotent.
            """
            d = int(sd.get("duration_sec", 0) or 0)
            if d <= 0 or d in (20, 40, 60, 80):
                return
            if d < 20:
                new_d = 20
            else:
                tail = d % 20
                if tail == 0:
                    return
                new_d = (d - tail) if tail <= 10 else (d + (20 - tail))
                new_d = max(20, new_d)
            sd["duration_sec"] = new_d
            # Adjust windows to match new bucket count
            n_target = max(1, new_d // 20)
            wps = sd.get("window_prompts") or []
            if new_d == 20 and wps:
                # Convert windows to a single video_prompt
                sd["video_prompt"] = " ".join(str(w) for w in wps)
                sd["window_prompts"] = []
            elif new_d > 20 and len(wps) > n_target:
                # Trim excess windows (merge into last)
                kept = list(wps[:n_target - 1])
                merged = " ".join(str(w) for w in wps[n_target - 1:])
                kept.append(merged)
                sd["window_prompts"] = kept

        if _raw_total <= _ceiling:
            # Tier 1
            _delta = _raw_total - target_duration
            _sign = "+" if _delta >= 0 else ""
            print(
                f"[ShortFilmPlanner] Pass 2 duration: {_raw_total}s "
                f"({len(shot_dicts)} shots) vs {target_duration}s target "
                f"({_sign}{_delta}s, within {_ceiling}s ceiling — no compression)."
            )
        elif _raw_total <= _scale_threshold:
            # Tier 2 — bucket-aware reduction
            _bucket_down = {80: 60, 60: 40, 40: 20}
            excess = _raw_total - target_duration
            print(
                f"[ShortFilmPlanner] ⚠ Pass 2 over budget: "
                f"{_raw_total}s total vs {target_duration}s target "
                f"(ceiling {_ceiling}s, +{_raw_total - target_duration}s overrun). "
                f"Bucket-down: snapping large shots to smaller buckets."
            )
            # Sort largest-bucket-first so we prefer reducing 60s→40s
            # over 40s→20s when the choice exists (preserves more
            # sustained scenes).
            candidates = sorted(
                [sd for sd in shot_dicts
                 if isinstance(sd, dict)
                 and sd.get("duration_sec") in _bucket_down],
                key=lambda s: -int(s.get("duration_sec", 0) or 0),
            )
            snapped: list[str] = []
            for sd in candidates:
                if excess <= 0:
                    break
                cur = int(sd.get("duration_sec", 0) or 0)
                nxt = _bucket_down[cur]
                # Drop the last window's content (it's the one being
                # cut). For 40s→20s that means drop one window AND
                # convert the surviving window to video_prompt.
                wps = list(sd.get("window_prompts") or [])
                if wps:
                    wps = wps[:-1]
                    if nxt == 20:
                        sd["video_prompt"] = (
                            " ".join(str(w) for w in wps) if wps else
                            sd.get("video_prompt", "") or ""
                        )
                        sd["window_prompts"] = []
                    else:
                        sd["window_prompts"] = wps
                sd["duration_sec"] = nxt
                excess -= (cur - nxt)
                snapped.append(
                    f"'{sd.get('title', 'untitled')}' {cur}s→{nxt}s"
                )
            _new_total = sum(
                int(sd.get("duration_sec", 0) or 0)
                for sd in shot_dicts
                if isinstance(sd, dict)
            )
            if snapped:
                print(
                    f"[ShortFilmPlanner] Bucket-down: "
                    f"{', '.join(snapped)}. "
                    f"New total: {_new_total}s "
                    f"({_raw_total - _new_total}s removed)."
                )
            else:
                # No bucket-down candidates (all shots already 20s).
                # Accept the overshoot rather than chop content.
                print(
                    f"[ShortFilmPlanner] No bucket-down candidates "
                    f"(all shots are 20s). Accepting {_new_total}s "
                    f"overshoot vs {target_duration}s target."
                )
            # Always run snap-cleanup so any leftover non-bucket dur
            # (e.g. from earlier snap-up steps) gets normalized.
            for sd in shot_dicts:
                if isinstance(sd, dict):
                    _snap_bucket(sd)
        else:
            # Tier 3 — runaway. Proportional scale + bucket cleanup.
            scale = target_duration / _raw_total if _raw_total else 1.0
            print(
                f"[ShortFilmPlanner] ⚠ Pass 2 SEVERELY over budget: "
                f"{_raw_total}s total vs {target_duration}s target "
                f"(ceiling {_ceiling}s, +{_raw_total - target_duration}s, "
                f">{int((_scale_threshold/target_duration - 1) * 100)}% over). "
                f"Proportional scale {scale:.2%} + bucket cleanup."
            )
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                old_dur = int(sd.get("duration_sec", 0) or 0)
                if old_dur <= 0:
                    continue
                sd["duration_sec"] = max(3, int(old_dur * scale))
            for sd in shot_dicts:
                if isinstance(sd, dict):
                    _snap_bucket(sd)
            _new_total = sum(
                int(sd.get("duration_sec", 0) or 0)
                for sd in shot_dicts
                if isinstance(sd, dict)
            )
            print(
                f"[ShortFilmPlanner] After scale + bucket cleanup: "
                f"{_new_total}s ({len(shot_dicts)} shots)."
            )

        # Deterministic post-process: fix structural rule violations the LLM
        # makes despite all prompt-level guidance. Two passes:
        #
        # 1. WINDOW COUNT OVERSHOOT — Gemma 4B sometimes emits 3 windows
        #    for a 35s shot when the formula calls for 2. Trim excess
        #    windows and merge their content into the last surviving one.
        #
        # 2. RUSHED TAIL WINDOW — when duration_sec is not a multiple of 20
        #    (e.g. 25s, 35s, 45s), the backend allocates 20s to each full
        #    window and gives the tail window only the remainder. A 25s
        #    shot with 2 windows gets W1=20s, W2=5s — the 5s window is
        #    far too short to fit the dialogue/action the LLM wrote for
        #    it. Empirically, anything <10s of tail is "rushed". Fix by
        #    merging the rushed tail into the previous window AND snapping
        #    duration_sec down to the resulting clean multiple of 20.
        import math as _math
        for sd in shot_dicts:
            try:
                dur = int(sd.get("duration_sec", 0) or 0)
                wps = sd.get("window_prompts", []) or []
                if dur <= 20:
                    continue
                # ── Pass 0: shot violates the "≥21s = use window_prompts"
                # rule by populating video_prompt instead. Common LLM
                # violation, especially Gemma 4B on NSFW screenplays
                # where attention to structural rules drops. Snap down
                # to 20s so the shot fits a single video_prompt cleanly,
                # since the LLM clearly intended one continuous block of
                # action (not multiple windows).
                if not wps:
                    vp = sd.get("video_prompt", "") or ""
                    if vp.strip():
                        sd["duration_sec"] = 20
                        # Drop keyframes — they were placed for the LLM's
                        # original (longer, multi-stage) intent. After
                        # snapping to a single 20s video_prompt, those
                        # keyframes are stale visual references that
                        # over-constrain a now-simpler shot.
                        had_kfs = bool(sd.get("keyframe_prompts"))
                        if had_kfs:
                            sd["keyframe_prompts"] = []
                        print(
                            f"[ShortFilmPlanner] Snap-down (video_prompt only) in '{sd.get('title', 'untitled')}': "
                            f"{dur}s → 20s — LLM populated video_prompt for a >20s shot instead of window_prompts; "
                            "treating as single 20s shot to match the LLM's structural intent"
                            + (" (also cleared stale keyframes)" if had_kfs else "")
                        )
                    # If both video_prompt and window_prompts are empty,
                    # nothing to do — the shot is malformed.
                    continue
                # ── Pass 0c: reconcile MIXED-STATE shots ──────────────
                # The strict rule is "≤20s → video_prompt only; ≥21s →
                # window_prompts only." The LLM sometimes violates it
                # by populating BOTH. The polish layer
                # (prompt_polish.py:1046) and the gen layer both pick
                # window_prompts when it has 2+ entries and silently
                # drop video_prompt — so any dialogue the LLM put in
                # video_prompt gets discarded before it reaches the
                # video model.
                #
                # Reconcile here based on where the actual dialogue
                # lives (detected by quoted text containing 3+ words).
                # The user-reported failure looked exactly like this:
                # 25s shot with full scene + dialogue in video_prompt
                # and short "same scene, medium shot..." stub strings
                # in window_prompts (the LLM treated them as keyframes).
                # Detect quoted-dialogue spans of ≥3 words, accepting
                # straight + smart quotes. `re.finditer` caches the
                # compiled pattern internally so repeated calls are cheap.
                import re as _re_dlg
                _DIALOGUE_PAT = r'[\"\'“”‘’]([^\"\'“”‘’]{12,})[\"\'“”‘’]'
                def _has_dialogue(text: str) -> bool:
                    if not isinstance(text, str) or not text.strip():
                        return False
                    for m in _re_dlg.finditer(_DIALOGUE_PAT, text):
                        if len(m.group(1).split()) >= 3:
                            return True
                    return False

                vp_text = (sd.get("video_prompt") or "").strip()
                if vp_text and wps:
                    vp_has_dialogue = _has_dialogue(vp_text)
                    wps_have_dialogue = any(_has_dialogue(w) for w in wps if isinstance(w, str))
                    vp_words = len(vp_text.split())
                    wp_words_max = max((len(w.split()) for w in wps if isinstance(w, str)), default=0)

                    # CASE A: video_prompt has dialogue, window_prompts
                    # don't. The LLM put the real scene content in
                    # video_prompt and treated window_prompts as
                    # keyframe-shaped stubs (e.g. "same scene, close-up
                    # of her face..."). Collapse to a 20s single shot
                    # using video_prompt — the dialogue must be
                    # preserved or the scene loses its core content.
                    if vp_has_dialogue and not wps_have_dialogue:
                        wp_count_before = len(wps)
                        sd["window_prompts"] = []
                        sd["duration_sec"] = 20
                        had_kfs = bool(sd.get("keyframe_prompts"))
                        if had_kfs:
                            sd["keyframe_prompts"] = []
                        wps = []
                        print(
                            f"[ShortFilmPlanner] Mixed-state reconciled (Case A) in '{sd.get('title', 'untitled')}': "
                            f"video_prompt has dialogue ({vp_words}w), window_prompts don't "
                            f"({wp_count_before} stubs, max {wp_words_max}w) → collapsed to 20s single "
                            f"video_prompt. Without this, the polish layer would skip video_prompt entirely "
                            f"(because window_prompts has 2+ entries) and the dialogue would be silently "
                            f"dropped before video gen."
                            + (" (also cleared stale keyframes)" if had_kfs else "")
                        )
                        continue
                    # CASE B: window_prompts have dialogue (LLM
                    # followed the rule for windows but ALSO left a
                    # stale video_prompt). Clear video_prompt so the
                    # unused field doesn't confuse anyone downstream.
                    if wps_have_dialogue:
                        sd["video_prompt"] = ""
                        print(
                            f"[ShortFilmPlanner] Mixed-state reconciled (Case B) in '{sd.get('title', 'untitled')}': "
                            f"both fields populated; window_prompts have the dialogue, video_prompt cleared "
                            f"(was {vp_words}w of redundant content the polish layer would have ignored)"
                        )
                    # CASE C: neither has dialogue (action-only scene
                    # where the LLM violated the either/or rule). Keep
                    # window_prompts since the duration calls for them,
                    # clear video_prompt.
                    else:
                        sd["video_prompt"] = ""
                        print(
                            f"[ShortFilmPlanner] Mixed-state reconciled (Case C) in '{sd.get('title', 'untitled')}': "
                            f"both fields populated, no dialogue in either; window_prompts kept "
                            f"(matches {dur}s duration), video_prompt cleared (was {vp_words}w)"
                        )

                # ── Pass 0b: window-count UNDERSHOOT. LLM produced fewer
                # windows than the duration calls for (e.g. 30s shot with
                # only 1 window_prompt). Without this fix, the wgp pipeline
                # generates the full duration anyway and uses the single
                # window prompt for both windows — producing the action-
                # looping behavior the original rule was designed to
                # prevent. Snap duration down to 20 × len(wps) so the
                # shot fits the actual window count cleanly. We lose the
                # missing window's worth of intended runtime but avoid
                # repeating the same prompt across two windows.
                expected_pre = max(1, _math.ceil(dur / 20.0))
                actual_pre = len(wps)
                if actual_pre < expected_pre:
                    new_dur = 20 * actual_pre
                    if new_dur < dur:
                        sd["duration_sec"] = new_dur
                        had_kfs = bool(sd.get("keyframe_prompts"))
                        # If snapped down to a single window, switch to
                        # video_prompt to satisfy the strict ≤20s rule
                        # (window_prompts is for >20s shots only).
                        # Also drop keyframes — they were placed for the
                        # LLM's original (longer) intent and are stale
                        # references on a now-simpler single-prompt shot.
                        if actual_pre == 1:
                            sd["video_prompt"] = str(wps[0])
                            sd["window_prompts"] = []
                            wps = []
                            if had_kfs:
                                sd["keyframe_prompts"] = []
                        print(
                            f"[ShortFilmPlanner] Snap-down (window undershoot) in '{sd.get('title', 'untitled')}': "
                            f"{dur}s → {new_dur}s — LLM emitted {actual_pre} "
                            f"window(s) for a {dur}s shot (needed {expected_pre}); "
                            "duration trimmed to match actual window count"
                            + (" (also cleared stale keyframes)" if had_kfs and actual_pre == 1 else "")
                        )
                        # Update dur for subsequent passes; if windows
                        # got cleared (snap to single video_prompt),
                        # skip the rest of the per-window passes.
                        dur = new_dur
                        if not wps:
                            continue
                # ── Pass 1: window-count overshoot ─────────────────────
                expected = max(1, _math.ceil(dur / 20.0))
                actual = len(wps)
                if actual > expected:
                    keep = list(wps[: expected - 1])
                    merged_tail = " ".join(str(w) for w in wps[expected - 1:])
                    keep.append(merged_tail)
                    sd["window_prompts"] = keep
                    wps = keep
                    print(
                        f"[ShortFilmPlanner] Fixed window overshoot in '{sd.get('title', 'untitled')}': "
                        f"{actual} → {expected} windows for {dur}s shot (excess merged into last window)"
                    )
                # ── Pass 2: snap to multiple-of-20 duration buckets ────
                # User-facing rule: shots are EITHER ≤20s (single
                # video_prompt, no windows) OR exactly a multiple of 20s
                # (40, 60, 80) for sustained continuous action that
                # genuinely warrants the longer runtime. NEVER 22s, 25s,
                # 30s, 35s, 45s — these create stranded tail windows
                # (e.g. a 25s shot is W1=20s + W2=5s, where W2 renders
                # as a sluggish stub and the cut into the next shot
                # feels jagged).
                #
                # The Pass 2 user prompt already tells the LLM "duration
                # MUST be one of 20/40/60/80". This post-process is the
                # safety net for when the LLM picks an invalid value
                # anyway. Snap direction picks the NEAREST valid bucket:
                #
                #   tail = duration_sec % 20
                #   tail == 0       → already valid, no change
                #   1 ≤ tail ≤ 10   → snap DOWN (subtract tail, merge
                #                     last window's content into previous)
                #   11 ≤ tail ≤ 19  → snap UP (add 20-tail seconds, last
                #                     window covers a longer effective
                #                     time but receives no extra content)
                #
                # Why split the snap direction at the midpoint: if the
                # LLM wrote 25s of content (tail=5), it sized only ~15-30
                # words for the tail window. Snapping down merges those
                # words into the previous 20s window — minor compression,
                # acceptable. If the LLM wrote 35s of content (tail=15),
                # the tail window has a near-full 60-100 words. Cramming
                # those into the previous 20s window would rush dialogue
                # significantly. Snapping up to 40s preserves pacing
                # (the 5s expansion just gives the last window a few
                # extra seconds of breathing room). 40s shots are
                # explicitly allowed by the new rule.
                #
                # Special case: 1-window shots whose duration_sec exceeds
                # 20 by 1-10s (e.g. 25s with no windows) snap down to
                # 20s and stay single-video_prompt. Anything ≥ 21s
                # should already be in window form per the threshold
                # rules, but we handle the malformed case defensively.
                n = len(wps)
                tail_seconds = dur % 20
                if dur > 0 and tail_seconds != 0:
                    had_kfs = bool(sd.get("keyframe_prompts"))
                    cleared_kfs = False
                    if tail_seconds <= 10:
                        # Snap DOWN: drop the tail.
                        new_dur = dur - tail_seconds
                        if new_dur < 20:
                            new_dur = 20  # never go below the minimum
                        if n == 0:
                            # 1-window shot (≤20s case shouldn't reach
                            # here, but defensive). Just clamp duration.
                            sd["duration_sec"] = new_dur
                            print(
                                f"[ShortFilmPlanner] Snap-down (tail {tail_seconds}s) "
                                f"in '{sd.get('title', 'untitled')}': "
                                f"{dur}s → {new_dur}s (no windows)"
                            )
                        elif n == 1:
                            # Was 21-30s with one window. Snap to 20s,
                            # convert window to video_prompt.
                            sd["duration_sec"] = new_dur
                            if new_dur == 20:
                                sd["video_prompt"] = str(wps[0])
                                sd["window_prompts"] = []
                                if had_kfs:
                                    sd["keyframe_prompts"] = []
                                    cleared_kfs = True
                            print(
                                f"[ShortFilmPlanner] Snap-down (tail {tail_seconds}s) "
                                f"in '{sd.get('title', 'untitled')}': "
                                f"{dur}s → {new_dur}s "
                                + ("(now single video_prompt)" if new_dur == 20 else "")
                                + (" (also cleared stale keyframes)" if cleared_kfs else "")
                            )
                        else:
                            # Multi-window: merge last window into previous.
                            merged = str(wps[-2]) + " " + str(wps[-1])
                            new_windows = list(wps[:-2]) + [merged]
                            sd["window_prompts"] = new_windows
                            sd["duration_sec"] = new_dur
                            if len(new_windows) == 1:
                                sd["video_prompt"] = merged
                                sd["window_prompts"] = []
                                if had_kfs:
                                    sd["keyframe_prompts"] = []
                                    cleared_kfs = True
                            print(
                                f"[ShortFilmPlanner] Snap-down (tail {tail_seconds}s) "
                                f"in '{sd.get('title', 'untitled')}': "
                                f"{dur}s ({n} windows) → {new_dur}s "
                                f"({len(new_windows)} window(s)) — small tail merged into previous"
                                + (" (also cleared stale keyframes)" if cleared_kfs else "")
                            )
                    else:
                        # tail 11-19s → snap UP (preserve content, accept
                        # a few extra seconds of runtime). The new_dur is
                        # the next multiple of 20.
                        new_dur = dur + (20 - tail_seconds)
                        sd["duration_sec"] = new_dur
                        # If we started with no windows but now need them
                        # (≤20s → >20s wouldn't happen here since dur was
                        # already > 20 to have a non-zero tail; but
                        # defensive against edge cases like dur=11):
                        if new_dur > 20 and n == 0:
                            # Originating shot was malformed (single
                            # video_prompt with dur > 20). Convert to
                            # window form.
                            sd["window_prompts"] = [
                                str(sd.get("video_prompt", "") or ""),
                                "",  # second window blank — Pass 2 LLM
                                     # didn't intend a multi-window shot
                            ][:max(1, _math.ceil(new_dur / 20.0))]
                            sd["video_prompt"] = ""
                        print(
                            f"[ShortFilmPlanner] Snap-up (tail {tail_seconds}s) "
                            f"in '{sd.get('title', 'untitled')}': "
                            f"{dur}s → {new_dur}s — last window covers "
                            f"a slightly longer effective time, content unchanged"
                        )

                    # Diagnostic: warn when a window's content looks
                    # over-stuffed for its allocated time. Doesn't fix
                    # anything but flags pacing problems for future
                    # iteration. ~150 words/20s ≈ 7.5 words/s, so
                    # window with > 10 words/s of content is suspect.
                    try:
                        for wi, wp in enumerate(wps):
                            if not isinstance(wp, str):
                                continue
                            window_seconds = (
                                20 if wi < n - 1
                                else max(1, dur - 20 * (n - 1))
                            )
                            word_count = len(wp.split())
                            words_per_sec = word_count / window_seconds
                            if words_per_sec > 10:
                                print(
                                    f"[ShortFilmPlanner] Pacing warning in "
                                    f"'{sd.get('title', 'untitled')}' "
                                    f"window {wi+1}: {word_count} words for "
                                    f"{window_seconds}s ({words_per_sec:.1f} w/s) "
                                    f"— may render rushed"
                                )
                    except Exception:
                        pass
            except Exception as e:
                print(f"[ShortFilmPlanner] Duration post-process skipped a shot: {e}")

        # ── Image-prompt sanitization (Layer 1) ──────────────────────
        # Strip GARMENT BAN violations and narrative-filler phrases the
        # image model can't render. Runs on every shot's image_prompt
        # AND each keyframe_prompt regardless of whether Pass 3 polish
        # is enabled — Pass 2 LLM (especially Gemma 4B on NSFW) routinely
        # writes "white sweater" / "grey shirt" and emotion fillers like
        # "showing the heat of the moment" despite the rules. Pass 3
        # runs the same sanitizer again with the descriptor-dedupe pass
        # added (since it has the name_to_descriptor map). No-op when
        # the LLM already followed the rules.
        try:
            from ..prompt_polish import sanitize_image_prompt as _sanitize_ip
            for sd in shot_dicts:
                ip = sd.get("image_prompt") or ""
                if ip.strip():
                    sd["image_prompt"] = _sanitize_ip(
                        ip, log_prefix=f"[ShortFilmPlanner Pass2 image sanitize '{sd.get('title', 'untitled')}']"
                    )
                kfs = sd.get("keyframe_prompts") or []
                if isinstance(kfs, list) and kfs:
                    cleaned_kfs = []
                    for ki, kf in enumerate(kfs):
                        if isinstance(kf, str) and kf.strip():
                            cleaned_kfs.append(_sanitize_ip(
                                kf, log_prefix=f"[ShortFilmPlanner Pass2 keyframe[{ki}] sanitize '{sd.get('title', 'untitled')}']"
                            ))
                        else:
                            cleaned_kfs.append(kf)
                    sd["keyframe_prompts"] = cleaned_kfs
        except Exception as e:
            print(f"[ShortFilmPlanner] Image-prompt sanitization skipped: {e}")

        # ── Sex-act leet trigger strip (always-on safety net) ────────
        # User-reported leak: a SFW music video had "bl0wj0b" in a
        # keyframe_prompt. Same risk applies to short films when a
        # user has NSFW LoRAs in their video_loras selection from
        # prior testing and runs a SFW concept. Strip from image and
        # keyframe fields ALWAYS (still images don't use video LoRA
        # triggers). Strip from video/window fields when nsfw=False.
        try:
            from ..prompt_polish import strip_sex_act_leet_tokens as _strip_leet
            leet_count = 0
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                ip = sd.get("image_prompt") or ""
                if ip:
                    new_ip, n = _strip_leet(ip)
                    if n:
                        sd["image_prompt"] = new_ip
                        leet_count += n
                kfs = sd.get("keyframe_prompts") or []
                if isinstance(kfs, list):
                    new_kfs = []
                    for kf in kfs:
                        if isinstance(kf, str):
                            new_kf, n = _strip_leet(kf)
                            new_kfs.append(new_kf)
                            leet_count += n
                        else:
                            new_kfs.append(kf)
                    sd["keyframe_prompts"] = new_kfs
                if not nsfw:
                    vp = sd.get("video_prompt") or ""
                    if vp:
                        new_vp, n = _strip_leet(vp)
                        if n:
                            sd["video_prompt"] = new_vp
                            leet_count += n
                    wps_local = sd.get("window_prompts") or []
                    if isinstance(wps_local, list):
                        new_wps = []
                        for w in wps_local:
                            if isinstance(w, str):
                                new_w, n = _strip_leet(w)
                                new_wps.append(new_w)
                                leet_count += n
                            else:
                                new_wps.append(w)
                        sd["window_prompts"] = new_wps
            if leet_count:
                print(
                    f"[ShortFilmPlanner] Stripped {leet_count} sex-act leet "
                    f"trigger token(s) — LLM placed them in fields where they "
                    f"don't belong (still images or SFW video context)."
                )
        except Exception as e:
            print(f"[ShortFilmPlanner] Leet trigger strip skipped: {e}")

        # ── Storyboard camera-name leak strip (Multi-Shot LoRA mode) ─
        # When Pass 2 produced Format B storyboard prompts, the LLM
        # sometimes embeds character names inside the camera-type
        # parens ("Shot 2 (Close-up on Henry, 7s):"). The IC-LoRA was
        # trained on clean camera-type tokens; names in the parens
        # break the trained pattern. Strip the "on Henry" / "of Mary"
        # / "from Mary" / "with Mary" / "over Mary's shoulder" leak
        # everywhere it appears (video_prompt and each window_prompts
        # entry).
        try:
            from ..prompt_polish import strip_storyboard_camera_name_leaks
            total_stripped = 0
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                vp = sd.get("video_prompt") or ""
                if vp:
                    new_vp, n = strip_storyboard_camera_name_leaks(vp)
                    if n:
                        sd["video_prompt"] = new_vp
                        total_stripped += n
                wps_local = sd.get("window_prompts") or []
                if isinstance(wps_local, list):
                    new_wps = []
                    for w in wps_local:
                        if isinstance(w, str):
                            new_w, n = strip_storyboard_camera_name_leaks(w)
                            new_wps.append(new_w)
                            total_stripped += n
                        else:
                            new_wps.append(w)
                    sd["window_prompts"] = new_wps
            if total_stripped:
                print(
                    f"[ShortFilmPlanner] Stripped {total_stripped} character-"
                    f"name leak(s) from storyboard camera-type parens "
                    f"(e.g. 'Close-up on Henry' → 'Close-up')."
                )
        except Exception as e:
            print(f"[ShortFilmPlanner] Storyboard camera-name strip skipped: {e}")

        # Deduplicate scenes
        seen_goals = set()
        unique_dicts = []
        for sd in shot_dicts:
            goal = sd.get("scene_goal", "")
            if goal not in seen_goals:
                seen_goals.add(goal)
                unique_dicts.append(sd)

        shots = self._convert_story_shots(unique_dicts, char_profiles, has_reference, fps, frames_steps, frames_minimum)

        # Extract title from first shot if available
        title = shot_dicts[0].get("title") if shot_dicts else None
        self._last_title = title

        return shots, title

    def _convert_story_shots(
        self,
        shot_dicts: list[dict],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        fps: int,
        frames_steps: int,
        frames_minimum: int,
    ) -> list[ShotPlan]:
        """Convert LLM output to ShotPlan objects for story-driven mode."""
        shots = []
        for i, raw in enumerate(shot_dicts):
            duration = raw.get("duration_sec", raw.get("duration", 15))

            # Snap duration to valid frame count
            raw_frames = int(duration * fps)
            snapped = max(frames_minimum, ((raw_frames - 1) // frames_steps) * frames_steps + 1)
            duration = snapped / fps

            subjects = [SubjectRef.from_dict(s) if isinstance(s, dict) else SubjectRef(visual_description=str(s))
                        for s in raw.get("subjects_on_screen", [])]

            cam_raw = raw.get("camera_plan", {})
            camera = CameraPlan(
                framing=cam_raw.get("framing", "medium shot"),
                angle=cam_raw.get("angle"),
                movement=cam_raw.get("movement"),
                movement_intensity=cam_raw.get("movement_intensity", "subtle"),
            )

            audio_raw = raw.get("audio_plan", {})
            has_dialogue = bool(raw.get("dialogue_beats"))
            audio = AudioPlan(
                mode=audio_raw.get("mode", "dialogue_driven" if has_dialogue else "ambient_only"),
                ambience=audio_raw.get("ambience"),
                timing_anchor="audio" if has_dialogue else "video",
                lip_sync_critical=audio_raw.get("lip_sync_critical", has_dialogue),
            )

            dialogue_beats = None
            if raw.get("dialogue_beats"):
                dialogue_beats = [DialogueBeat.from_dict(db) for db in raw["dialogue_beats"]]
                # Enforce word budget (~2 words/sec)
                word_budget = int(duration * 2.5)
                total_words = sum(len(db.spoken_text.split()) for db in dialogue_beats)
                if total_words > word_budget * 1.5:
                    # Trim dialogue proportionally
                    for db in dialogue_beats:
                        words = db.spoken_text.split()
                        max_words = max(3, int(len(words) * word_budget / total_words))
                        db.spoken_text = " ".join(words[:max_words])

            shot = ShotPlan(
                shot_id=self._make_shot_id(i, "sf"),
                index=i,
                duration_sec=duration,
                skill_type="short_film",
                scene_goal=raw.get("scene_goal", f"Scene {i + 1}"),
                narrative_role=raw.get("narrative_role"),
                scene_type=raw.get("scene_type", "dialogue" if has_dialogue else "action"),
                source_mode_preference="i2v" if has_reference else "t2v",
                image_strategy="reference_edit" if has_reference else "fresh_generation",
                continuity_strategy="continuous" if i > 0 else "independent",
                subjects_on_screen=subjects,
                spatial_setup=raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                visual_style=raw.get("visual_style", ""),
                lighting=raw.get("lighting", ""),
                mood=raw.get("mood", ""),
                action_beats=raw.get("action_beats", []),
                dialogue_beats=dialogue_beats,
                camera_plan=camera,
                audio_plan=audio,
                ending_beat=raw.get("ending_beat", ""),
                metadata={
                    "title": raw.get("title", ""),
                    "duration_frames": snapped,
                },
                # LLM-generated prompts (used directly, skipping renderer pass 2)
                video_prompt=raw.get("video_prompt"),
                image_prompt=raw.get("image_prompt"),
                window_prompts=raw.get("window_prompts"),
                visual_changes=raw.get("visual_changes"),
                image_source=raw.get("image_source"),
                keyframe_prompts=raw.get("keyframe_prompts"),
            )
            shots.append(shot)

        return shots

    # ── Single-Pass Fallback ─────────────────────────────────────────

    def _plan_story_single_pass(
        self,
        story_description: str,
        reference_image_path: Optional[str],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        target_duration: int,
        target_scenes: Optional[int],
        narrative_mode: bool,
        fps: int,
        frames_steps: int,
        frames_minimum: int,
        nsfw: bool = False,
        polish_block: str = "",
    ) -> tuple[list[ShotPlan], Optional[str]]:
        """Fallback single-pass planning if the screenplay pass fails."""
        from ..nsfw_guidance import inject_nsfw_if_enabled

        if target_scenes is None:
            target_scenes = max(2, min(20, target_duration // 20))

        char_rules = build_character_rules_block(has_reference, char_profiles if char_profiles else None)
        # video_guide now merged into ltx2_shot_breakdown.md — no separate load needed

        system_prompt = f"""You are a short film director. Create a scene plan. Output ONLY the JSON array.

{f"You are given a REFERENCE PHOTO." if has_reference else ""}

{char_rules}

- Total duration must sum to ~{target_duration}s. YOU decide how many scenes based on the story.
- KEEP CONVERSATIONS TOGETHER — do not split dialogue across multiple shots. One conversation = one shot.
- Only cut when the location changes or a clear story beat transition happens.
- Prefer 20-40s shots. Shots over 20s need window_prompts.
- Output ONLY a JSON array with title, duration_sec, scene_goal, video_prompt, image_prompt per scene.
- image_prompt is the FIRST FRAME BEFORE action begins — initial state, static poses, zero motion verbs.
  If something changes in the scene, the image shows the BEFORE state (clothing on, room empty, etc.).


Go:"""

        if polish_block:
            system_prompt = f"{system_prompt}\n\n{polish_block}"
        system_prompt = inject_nsfw_if_enabled(system_prompt, nsfw, "both")

        # Single-pass fallback also gets the safety scan — it bypasses
        # Pass 1 entirely, so the post-Pass-1 scan above doesn't run for
        # this code path. Mirror the same hybrid co-occurrence check on
        # the user's concept (pre-call) and on the structured shot list
        # (post-call).
        from ..safety_scan import (
            assert_no_minor_content,
            collect_pass2_text,
        )
        assert_no_minor_content(story_description, source="user concept")

        image_paths = self._build_all_image_paths(reference_image_path, has_reference)
        # Grammar constraint — this path runs with thinking_budget=4096, so
        # the schema only fires on the parse-failure retry (see
        # _call_llm_json). The fallback spec asks for just five fields; the
        # rest of _SHOT_PROPERTIES stays available but optional. +2 slack
        # on maxItems since the prompt lets the LLM choose the scene count.
        fallback_schema = _shot_list_schema(
            min_items=2,
            max_items=max(4, target_scenes + 2),
            required=["title", "duration_sec", "scene_goal", "image_prompt", "video_prompt"],
        )
        shot_dicts = self._call_llm_json(
            user_prompt=f"Story: {story_description}",
            system_prompt=system_prompt,
            max_tokens=max(4096, target_duration * 60),
            thinking_budget=4096,
            image_paths=image_paths,
            json_schema=fallback_schema,
        )

        assert_no_minor_content(
            collect_pass2_text(shot_dicts), source="shot list (single-pass fallback)"
        )

        seen_goals = set()
        unique_dicts = []
        for sd in shot_dicts:
            goal = sd.get("scene_goal", sd.get("title", ""))
            if goal not in seen_goals:
                seen_goals.add(goal)
                unique_dicts.append(sd)

        shots = self._convert_story_shots(unique_dicts, char_profiles, has_reference, fps, frames_steps, frames_minimum)
        title = shot_dicts[0].get("title") if shot_dicts else None
        self._last_title = title
        return shots, title
