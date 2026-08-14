"""
Music Video Planner — creates a ProductionPlan from song analysis data.

Inputs: song audio, transcript/lyrics, beat map, section labels,
performer map, optional reference image, user scene concept.

Outputs: ProductionPlan with ShotPlan objects (NOT final prompts).
"""

from __future__ import annotations
import os
import re
from typing import Optional, Any

from ..schema import (
    ProductionPlan, ShotPlan, CharacterProfile, ReferenceAssets,
    AssetRef, SubjectRef, DialogueBeat, CameraPlan, AudioPlan,
    SpeakerMapEntry,
)
from ..policies import build_character_rules_block, build_camera_style_block
from .base import BasePlanner


def _compact_lyrics_for_context(
    value: Any,
    *,
    max_words: int = 48,
    max_chars: int = 320,
) -> str:
    """Bound noisy transcription text before it reaches the planning LLM."""

    words = re.findall(r"\S+", str(value or ""))
    compact: list[str] = []
    index = 0
    truncated = False
    while index < len(words) and len(compact) < max_words:
        word = words[index]
        key = re.sub(r"(^\W+|\W+$)", "", word).casefold()
        run_end = index + 1
        while run_end < len(words):
            candidate = re.sub(
                r"(^\W+|\W+$)", "", words[run_end],
            ).casefold()
            if not key or candidate != key:
                break
            run_end += 1
        run_length = run_end - index
        if run_length >= 4:
            compact.extend(words[index:index + 3])
            compact.append("[repeated refrain]")
        else:
            compact.extend(words[index:run_end])
        index = run_end

    if index < len(words) or len(compact) > max_words:
        truncated = True
    text = " ".join(compact[:max_words])
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
        truncated = True
    if truncated:
        text = f"{text} [transcript excerpt truncated]".strip()
    return text


# ── Section-based visual strategy ────────────────────────────────────

_SECTION_VISUAL_STRATEGY = {
    "intro": {
        "camera_default": "wide establishing shot",
        "movement_intensity": "subtle",
        "energy": "building",
        "hints": "atmospheric, slow reveal, moody lighting, set the tone",
    },
    "verse": {
        "camera_default": "medium shot",
        "movement_intensity": "subtle",
        "energy": "steady",
        "hints": "storytelling, character focus, steady camera, intimate",
    },
    "chorus": {
        "camera_default": "dynamic angle",
        "movement_intensity": "dynamic",
        "energy": "peak",
        "hints": "bold energy, wide and close-up mix, confident movement",
    },
    "bridge": {
        "camera_default": "unique angle",
        "movement_intensity": "moderate",
        "energy": "contrasting",
        "hints": "change of scenery, dreamy or surreal, unexpected perspective",
    },
    "outro": {
        "camera_default": "wide shot",
        "movement_intensity": "subtle",
        "energy": "fading",
        "hints": "pulling back, reflective, fading light, resolution",
    },
    "instrumental": {
        "camera_default": "sweeping shot",
        "movement_intensity": "moderate",
        "energy": "atmospheric",
        "hints": "environment focus, dramatic sweep, textures, abstract visuals",
    },
}


_MUSIC_IMAGE_FIELDS = frozenset({
    "image_source",
    "image_prompt",
    "visual_changes",
    "keyframe_prompts",
})

_MUSIC_SHOT_PROPERTIES = {
    "scene_goal": {"type": "string"},
    "scene_type": {"type": "string"},
    "subjects_on_screen": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "character_id": {"type": "string"},
                "speaker_name": {"type": "string"},
                "visual_description": {"type": "string"},
                "position_or_relation": {"type": "string"},
            },
            "required": ["visual_description"],
            "additionalProperties": False,
        },
    },
    "spatial_setup": {"type": "string"},
    "environment": {"type": "string"},
    "visual_style": {"type": "string"},
    "lighting": {"type": "string"},
    "mood": {"type": "string"},
    "action_beats": {"type": "array", "items": {"type": "string"}},
    "camera_plan": {
        "type": "object",
        "properties": {
            "framing": {"type": "string"},
            "angle": {"type": "string"},
            "movement": {"type": "string"},
            "movement_intensity": {"type": "string"},
            "lens_feel": {"type": "string"},
        },
        "required": ["framing"],
        "additionalProperties": False,
    },
    "ending_beat": {"type": "string"},
    "image_source": {"type": "string"},
    "image_prompt": {"type": "string"},
    "visual_changes": {"type": "array", "items": {"type": "string"}},
    "video_prompt": {"type": "string"},
    "keyframe_prompts": {"type": "array", "items": {"type": "string"}},
    "window_prompts": {"type": "array", "items": {"type": "string"}},
}


def _music_shot_schema(count: int, *, include_image_fields: bool) -> dict:
    properties = {
        key: value
        for key, value in _MUSIC_SHOT_PROPERTIES.items()
        if include_image_fields or key not in _MUSIC_IMAGE_FIELDS
    }
    required = [
        "scene_goal",
        "scene_type",
        "subjects_on_screen",
        "environment",
        "visual_style",
        "lighting",
        "mood",
        "action_beats",
        "camera_plan",
        "ending_beat",
        "image_source",
        "image_prompt",
        "visual_changes",
        "video_prompt",
        "window_prompts",
    ]
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": [field for field in required if field in properties],
            "additionalProperties": False,
        },
        "minItems": max(1, count),
        "maxItems": max(1, count),
    }


def _discard_unused_image_fields(shot_dicts: list[dict]) -> list[dict]:
    for shot in shot_dicts:
        if isinstance(shot, dict):
            for field in _MUSIC_IMAGE_FIELDS:
                shot.pop(field, None)
    return shot_dicts


# ── Performer Map Parsing ────────────────────────────────────────────

_PRONOUN_MAP = {
    "he": "man", "she": "woman", "him": "man", "her": "woman",
    "guy": "man", "girl": "woman", "boy": "man",
    "male": "man", "female": "woman",
    "rapper": "man", "singer": "woman",
}

_SECTION_ALIASES = {
    "verse": ["verse", "verses", "rap", "raps"],
    "chorus": ["chorus", "choruses", "hook", "hooks", "sing", "sings"],
    "bridge": ["bridge", "bridges"],
    "intro": ["intro", "introduction"],
    "outro": ["outro", "ending"],
    "instrumental": ["instrumental", "break"],
}


def _parse_performer_map(scene_description: str) -> dict[str, str]:
    """Extract section→performer mapping from natural language scene description.

    Returns: {"verse": "the man in the dark jacket", "chorus": "the woman", ...}
    """
    result = {}
    if not scene_description:
        return result

    text = scene_description.lower()
    # Pattern: "the man raps the verses", "chorus by the woman", etc.
    for section, aliases in _SECTION_ALIASES.items():
        for alias in aliases:
            patterns = [
                rf'(\bthe\s+\w+(?:\s+\w+)?)\s+(?:raps?|sings?|performs?|does?|handles?)\s+(?:the\s+)?{alias}',
                rf'{alias}\s+(?:by|from|performed by|sung by|rapped by)\s+(\bthe\s+\w+(?:\s+\w+)?)',
                rf'(\bthe\s+\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:on|in|doing)\s+(?:the\s+)?{alias}',
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    performer = m.group(1).strip()
                    # Normalize pronouns
                    for pronoun, replacement in _PRONOUN_MAP.items():
                        performer = re.sub(rf'\b{pronoun}\b', replacement, performer)
                    result[section] = performer
                    break
    return result


class MusicVideoPlanner(BasePlanner):
    skill_type = "music_video"

    def plan(
        self,
        clips: list[dict],
        scene_description: str,
        lyrics: Optional[list[dict]] = None,
        bpm: float = 120.0,
        reference_image_path: Optional[str] = None,
        speaker_mappings: Optional[dict] = None,
        characters: Optional[list[dict]] = None,
        **kwargs,
    ) -> ProductionPlan:
        """Create a ProductionPlan for a music video.

        Args:
            clips: List of PlannedClip dicts with beat_count, duration_frames, label, start, end, etc.
            scene_description: User's scene concept/vibe/setting.
            lyrics: Transcribed lyrics with optional speaker tags.
            bpm: Song tempo.
            reference_image_path: Path to reference photo (optional).
            speaker_mappings: {speaker_id: {name, role}} from UI.
            characters: List of character dicts [{name, description}].
        """
        has_reference = bool(reference_image_path)
        video_model = str(kwargs.get("video_model") or "")
        shot_image_policy = str(kwargs.get("shot_image_policy") or "")
        self._uses_generated_shot_images = shot_image_policy not in {
            "prompt_only",
            "direct_references",
        }
        self._preserve_video_character_names = (
            video_model.lower().startswith("minimax_h3")
            and shot_image_policy in {"prompt_only", "direct_references"}
        )
        performer_map = _parse_performer_map(scene_description)

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
        char_profiles = self._build_characters(characters, speaker_mappings, lyrics, performer_map)

        # Build speaker lookup
        speaker_names = self._build_speaker_names(speaker_mappings, lyrics)

        # Build reference assets
        ref_assets = ReferenceAssets(
            start_image=AssetRef(id="ref_image", type="image", uri=reference_image_path) if has_reference else None,
            lyrics=self._format_lyrics_text(lyrics) if lyrics else None,
            speaker_map=[
                SpeakerMapEntry(speaker_id=sid, name=info.get("name", ""), voice_description=info.get("role", ""))
                for sid, info in (speaker_mappings or {}).items()
            ] if speaker_mappings else None,
        )

        # Build clip context for LLM. H3 receives the actual source-audio slice
        # for every shot, so its transcript must never become prompt dialogue.
        clip_contexts = self._build_clip_contexts(
            clips,
            lyrics,
            performer_map,
            speaker_names,
            speaker_mappings,
            source_audio_drives_vocals=video_model.lower().startswith(
                "minimax_h3"
            ),
        )

        # Call LLM for creative planning
        nsfw = kwargs.get("nsfw", False)
        shot_dicts = self._plan_with_llm(
            clips=clips,
            clip_contexts=clip_contexts,
            scene_description=scene_description,
            bpm=bpm,
            has_reference=has_reference,
            reference_image_path=reference_image_path,
            char_profiles=char_profiles,
            performer_map=performer_map,
            nsfw=nsfw,
            **{k: v for k, v in kwargs.items() if k not in ("nsfw",)},
        )

        # ── Image-prompt sanitization (Layer 1) ──────────────────────
        # Strip GARMENT BAN violations and narrative-filler phrases the
        # image model can't render. Mirrors the same hook in short_film.py
        # so music-video shots also benefit from the deterministic cleanup
        # regardless of whether Pass 3 polish is enabled. See
        # prompt_polish.sanitize_image_prompt for the full ruleset.
        try:
            from ..prompt_polish import sanitize_image_prompt as _sanitize_ip
            for sd in shot_dicts:
                ip = sd.get("image_prompt") or ""
                if ip.strip():
                    sd["image_prompt"] = _sanitize_ip(
                        ip, log_prefix=f"[MusicVideoPlanner Pass2 image sanitize '{sd.get('scene_goal', 'untitled')[:40]}']"
                    )
                kfs = sd.get("keyframe_prompts") or []
                if isinstance(kfs, list) and kfs:
                    cleaned_kfs = []
                    for ki, kf in enumerate(kfs):
                        if isinstance(kf, str) and kf.strip():
                            cleaned_kfs.append(_sanitize_ip(
                                kf, log_prefix=f"[MusicVideoPlanner Pass2 keyframe[{ki}] sanitize]"
                            ))
                        else:
                            cleaned_kfs.append(kf)
                    sd["keyframe_prompts"] = cleaned_kfs
        except Exception as e:
            print(f"[MusicVideoPlanner] Image-prompt sanitization skipped: {e}")

        # ── Sex-act leet trigger strip (always-on safety net) ────────
        # User-reported leak: a SFW music video about a football coach
        # had "bl0wj0b" in a keyframe_prompt because the LLM treated
        # leet-coded LoRA triggers as generic energy boosters. The
        # leet tokens (bl0wj0b, m15510n4ry, c0wg1rl, etc.) are SEX-ACT
        # video-LoRA triggers — they should NEVER appear in:
        #   - image_prompt or keyframe_prompts (still images, video
        #     LoRA triggers don't apply)
        #   - SFW content of any kind
        # Strip runs on EVERY prompt field unconditionally for image
        # and keyframe fields. For video_prompt and window_prompts,
        # strip only when the surrounding screenplay/scene context
        # is SFW (handled below via nsfw flag).
        try:
            from ..prompt_polish import strip_sex_act_leet_tokens
            total_stripped = 0
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                # Image + keyframe: always strip (still images don't
                # use video LoRA triggers regardless of NSFW mode).
                ip = sd.get("image_prompt") or ""
                if ip:
                    new_ip, n = strip_sex_act_leet_tokens(ip)
                    if n:
                        sd["image_prompt"] = new_ip
                        total_stripped += n
                kfs = sd.get("keyframe_prompts") or []
                if isinstance(kfs, list):
                    new_kfs = []
                    for kf in kfs:
                        if isinstance(kf, str):
                            new_kf, n = strip_sex_act_leet_tokens(kf)
                            new_kfs.append(new_kf)
                            total_stripped += n
                        else:
                            new_kfs.append(kf)
                    sd["keyframe_prompts"] = new_kfs
                # Video / windows: strip only if NSFW mode is OFF
                # (effective_nsfw could differ from request nsfw; we
                # use the planner's own nsfw flag which is wired
                # from services config).
                if not nsfw:
                    vp = sd.get("video_prompt") or ""
                    if vp:
                        new_vp, n = strip_sex_act_leet_tokens(vp)
                        if n:
                            sd["video_prompt"] = new_vp
                            total_stripped += n
                    wps = sd.get("window_prompts") or []
                    if isinstance(wps, list):
                        new_wps = []
                        for w in wps:
                            if isinstance(w, str):
                                new_w, n = strip_sex_act_leet_tokens(w)
                                new_wps.append(new_w)
                                total_stripped += n
                            else:
                                new_wps.append(w)
                        sd["window_prompts"] = new_wps
            if total_stripped:
                print(
                    f"[MusicVideoPlanner] Stripped {total_stripped} sex-act "
                    f"leet trigger token(s) (bl0wj0b/m15510n4ry/c0wg1rl/etc.) "
                    f"from prompts that should never contain them. The LLM "
                    f"misplaced trigger words from the active LoRA selection."
                )
        except Exception as e:
            print(f"[MusicVideoPlanner] Leet trigger strip skipped: {e}")

        # Convert LLM output to ShotPlan objects
        shots = self._convert_to_shots(
            shot_dicts=shot_dicts,
            clips=clips,
            char_profiles=char_profiles,
            has_reference=has_reference,
            performer_map=performer_map,
            lyrics=lyrics,
            speaker_names=speaker_names,
        )

        total_duration = sum(c.get("end", 0) - c.get("start", 0) for c in clips) if clips else None

        return ProductionPlan(
            skill_type="music_video",
            title=None,
            global_style=scene_description,
            total_duration_sec=total_duration,
            reference_assets=ref_assets,
            characters=char_profiles if char_profiles else None,
            shots=shots,
            continuity_notes=[
                "Music video — visual variety across sections is important",
                "Chorus clips should feel higher energy than verses",
                "Performer must be visible when assigned to a clip",
            ],
        )

    # ── Character Building ───────────────────────────────────────────

    def _build_characters(
        self,
        characters: Optional[list[dict]],
        speaker_mappings: Optional[dict],
        lyrics: Optional[list[dict]],
        performer_map: dict[str, str],
    ) -> list[CharacterProfile]:
        """Build character profiles from available sources."""
        profiles = []

        # From explicit characters
        if characters:
            for i, c in enumerate(characters):
                profiles.append(CharacterProfile(
                    id=f"char_{i}",
                    display_name=c.get("name", ""),
                    physical_description=c.get("description", "person"),
                ))

        # From speaker mappings (if no explicit characters)
        if not profiles and speaker_mappings:
            for sid, info in speaker_mappings.items():
                name = info.get("name", sid)
                role = info.get("role", "")
                profiles.append(CharacterProfile(
                    id=sid,
                    display_name=name,
                    physical_description=f"the {name}" if name else "a performer",
                    voice_description=role,
                ))

        # From performer map
        if not profiles and performer_map:
            seen = set()
            for section, performer in performer_map.items():
                if performer not in seen:
                    seen.add(performer)
                    profiles.append(CharacterProfile(
                        id=f"perf_{len(profiles)}",
                        display_name=None,
                        physical_description=performer,
                    ))

        return profiles

    def _build_speaker_names(
        self,
        speaker_mappings: Optional[dict],
        lyrics: Optional[list[dict]],
    ) -> dict[str, str]:
        """Map speaker_id → display name."""
        names: dict[str, str] = {}
        if speaker_mappings:
            for sid, info in speaker_mappings.items():
                names[sid] = info.get("name", sid)
        return names

    def _format_lyrics_text(self, lyrics: Optional[list[dict]]) -> str:
        """Format lyrics list into plain text."""
        if not lyrics:
            return ""
        return "\n".join(line.get("text", "") for line in lyrics if line.get("text", "").strip())

    # ── Clip Context Building ────────────────────────────────────────

    def _build_clip_contexts(
        self,
        clips: list[dict],
        lyrics: Optional[list[dict]],
        performer_map: dict[str, str],
        speaker_names: dict[str, str],
        speaker_mappings: Optional[dict],
        *,
        source_audio_drives_vocals: bool = False,
    ) -> list[str]:
        """Build text descriptions for each clip (context for LLM)."""
        contexts = []
        for i, clip in enumerate(clips):
            section = (clip.get("label") or "verse").lower()
            beat_count = clip.get("beat_count", 8)
            start_sec = clip.get("start", 0)
            end_sec = clip.get("end", start_sec + 5)

            # Gather overlapping lyrics
            lyrics_snippet = ""
            if lyrics:
                overlapping = [
                    l.get("text", "")
                    for l in lyrics
                    if l.get("start", 0) < end_sec and l.get("end", 0) > start_sec
                ]
                if overlapping:
                    lyrics_snippet = _compact_lyrics_for_context(
                        " ".join(overlapping)
                    )

            # Identify dominant speaker
            performer_hint = ""
            if section in performer_map:
                performer_hint = f" Performer: {performer_map[section]}."
            elif clip.get("dominant_speaker") and speaker_names.get(clip["dominant_speaker"]):
                name = speaker_names[clip["dominant_speaker"]]
                role = ""
                if speaker_mappings and clip["dominant_speaker"] in speaker_mappings:
                    role = speaker_mappings[clip["dominant_speaker"]].get("role", "")
                performer_hint = f" Performer: the {name}"
                if role:
                    performer_hint += f" ({role})"
                performer_hint += "."

            # Vocal info
            if source_audio_drives_vocals:
                vocal_info = (
                    "mapped source audio drives this interval; synchronize "
                    "visible performance, movement, and any singing to that "
                    "exact audio without quoting, transcribing, or inventing "
                    "words"
                )
            else:
                vocal_info = (
                    f'lyrics excerpt: "{lyrics_snippet}"'
                    if lyrics_snippet else "instrumental"
                )

            ctx = f"Clip {i + 1}: {section}, {beat_count} beats, {vocal_info}.{performer_hint}"
            contexts.append(ctx)

        return contexts

    # ── LLM Planning Call ────────────────────────────────────────────

    def _plan_with_llm(
        self,
        clips: list[dict],
        clip_contexts: list[str],
        scene_description: str,
        bpm: float,
        has_reference: bool,
        reference_image_path: Optional[str],
        char_profiles: list[CharacterProfile],
        performer_map: dict[str, str],
        nsfw: bool = False,
        **kwargs,
    ) -> list[dict]:
        """Call LLM to generate structured shot plans."""
        from ..nsfw_guidance import inject_nsfw_if_enabled

        num_character_refs = len(kwargs.get("character_ref_paths", []) or [])
        num_location_refs = len(kwargs.get("location_ref_paths", []) or [])
        has_asset_references = bool(
            has_reference or num_character_refs or num_location_refs
        )
        preserve_names = bool(
            getattr(self, "_preserve_video_character_names", False)
        )
        uses_generated_images = bool(
            getattr(self, "_uses_generated_shot_images", True)
        )
        char_rules = build_character_rules_block(
            has_reference or bool(num_character_refs),
            char_profiles if char_profiles else None,
            preserve_names=preserve_names,
        )
        camera_block = build_camera_style_block()
        # video_guide now merged into ltx2_music_video_rules.md — no separate load needed

        image_prompt_rules = ""
        if uses_generated_images:
            from ..image_prompt_rules import get_image_prompt_rules
            image_prompt_rules = get_image_prompt_rules(
                has_reference,
                num_character_refs=num_character_refs,
                num_location_refs=num_location_refs,
                character_ref_labels=kwargs.get("character_ref_labels"),
                location_ref_labels=kwargs.get("location_ref_labels"),
                seamless=kwargs.get("seamless", True),
                image_model=kwargs.get("image_model", ""),
            )

        from ..guide_loader import load_guide
        video_model = str(kwargs.get("video_model") or "")
        music_video_rules = load_guide(
            "minimax_h3_shot_breakdown.md"
            if video_model.lower().startswith("minimax_h3")
            else "ltx2_music_video_rules.md"
        )
        h3_direct_rules = (
            "H3 DIRECT-REFERENCE MUSIC VIDEO:\n"
            "- No generated start frame will be supplied. Make each video_prompt "
            "self-contained with setting, composition, named identities plus "
            "visible traits, wardrobe, performance, camera, lighting, ambience, "
            "effects, and music.\n"
            "- The per-shot source-audio slice is mapped as driving audio. "
            "Describe visible singing, lip movement, dance, action, and camera "
            "that synchronize to it; do not invent or transcribe lyrics.\n"
            "- Character/location references are soft guidance, not fixed first "
            "frames. Describe the finished target shot.\n"
            "- Do not create image_prompt, image_source, visual_changes, or "
            "keyframe_prompts. Those fields are intentionally absent from the "
            "video-only output schema."
            if not uses_generated_images else ""
        )
        reference_aesthetic_rules = (
            """VISUAL AESTHETIC — the reference photo defines the visual style for the entire music video.
Match its aesthetic (color grading, film texture, era, tone) in every image_prompt unless the
scene concept explicitly calls for a style change. End each image_prompt with
"Use lighting and color temp from reference image." to preserve the look."""
            if uses_generated_images and has_reference else ""
        )
        image_output_fields = (
            '''    "image_source": "original or previous",
    "image_prompt": "FIRST FRAME BEFORE action — initial state, static pose, environment. No motion verbs.",
    "visual_changes": ["what transforms during the clip — e.g. 'performer jumps off stage', 'lights shift to red'"],
'''
            if uses_generated_images else ""
        )
        keyframe_output_field = (
            '    "keyframe_prompts": [],\n'
            if uses_generated_images else ""
        )
        image_workflow_notes = (
            """- image_source: "original" = user's reference photo (default). "previous" = previous scene's output for same-location continuity.
- FIELD ORDER: Write image_prompt FIRST (starting state), then visual_changes, then video_prompt.
- visual_changes: If the performer jumps off stage, image_prompt shows them still ON stage.
- keyframe_prompts: DEFAULT IS EMPTY. Add one only for a specific visual state the video model cannot infer from the start image and prompt; never for ordinary movement, camera, expression, lighting, or energy changes.
"""
            if uses_generated_images else ""
        )
        if uses_generated_images and has_reference:
            scene_anchoring_rules = """SCENE-ANCHORING (avoid off-topic content):
The user's main reference is visual ground truth. Every image_prompt and video_prompt must match its identity, setting, and aesthetic plus the Scene Concept. Do not invent unrelated worlds."""
        elif has_asset_references:
            scene_anchoring_rules = """SCENE-ANCHORING (avoid off-topic content):
Character references define identity and location references define the setting. Follow their labels and the Scene Concept in every self-contained video prompt; do not invent conflicting identities or settings."""
        else:
            scene_anchoring_rules = """SCENE-ANCHORING (avoid off-topic content):
No visual reference was provided. Invent one consistent performer and setting that fit the Scene Concept, then reuse the same artist and world across every clip. Show the performer delivering vocals on lyric clips and do not drift off-concept."""

        system_prompt = f"""You are a music video director. Plan each clip AND write its prompts. Output ONLY the JSON array.

{f"You are given a REFERENCE PHOTO. Use it to identify appearance, clothing, and setting." if has_reference else ""}

{char_rules}

{camera_block}

{reference_aesthetic_rules}

MUSIC VIDEO RULES:
- Chorus = high energy, bold framing. Verse = intimate, character focus.
- Instrumental = environment, textures. Bridge = contrasting, unexpected.
- Vary visuals across clips. Performer must be visible when assigned.

{music_video_rules}

{h3_direct_rules}

{image_prompt_rules}


OUTPUT — respond with ONLY a JSON array:
[
  {{
    "scene_goal": "What this clip achieves",
    "scene_type": "performance|narrative|atmospheric",
    "subjects_on_screen": [{{"visual_description": "the woman in red", "position_or_relation": "center frame"}}],
    "environment": "Setting details",
    "visual_style": "Style",
    "lighting": "Lighting",
    "mood": "Tone",
    "action_beats": ["Action 1", "Action 2"],
    "camera_plan": {{"framing": "medium shot", "movement": "slow dolly in", "movement_intensity": "subtle"}},
    "ending_beat": "Final image",
{image_output_fields}    "video_prompt": "Energetic prompt describing the visible performance, action, sound, and camera.",
{keyframe_output_field}    "window_prompts": []
  }}
]

Notes:
{image_workflow_notes}
- window_prompts: empty ([]) unless the scene needs >26s continuous video.

{scene_anchoring_rules}

KEEP MUSIC-VIDEO PROMPTS SIMPLE:
For each scene, the music drives the pacing and energy. You only need to identify:
  - WHO is in frame ({"preserve user-supplied proper names and pair them with useful visible traits" if preserve_names else "the performer, by descriptor — never by name"})
  - CAMERA MOVEMENT (push-in, pull-back, orbit, handheld, low angle, etc.)
  - ATMOSPHERIC ELEMENTS (smoke, pyro, crowd cheering, lighting flashes, etc.)
  - The performer's BODY MOVEMENT in broad strokes (head bob, arms raised,
    walking forward, etc.) — but don't over-specify; the model interpolates.
{"Use the H3 Context-IR fields above. Be concise but complete; do not enforce the legacy 15-40 word LTX limit." if preserve_names else "Keep video_prompt 15-40 words. Anything longer is over-described for music video."}

{"Most scenes should use a single video_prompt with empty keyframe_prompts." if uses_generated_images else "Every scene should use video_prompt/window_prompts only; omit all still-image fields."}
Output exactly {len(clips)} objects. Go:"""

        # Inject model-specific prompt polish guide if provided
        polish_block = kwargs.get("polish_block", "")
        if polish_block:
            system_prompt = f"{system_prompt}\n\n{polish_block}"

        # Inject content guidance (NSFW or safety guardrails)
        system_prompt = inject_nsfw_if_enabled(
            system_prompt,
            nsfw,
            "both" if uses_generated_images else "video",
        )

        user_prompt = f"""Scene Concept: {scene_description}
Song tempo: {bpm:.0f} BPM

Clips:
{chr(10).join(clip_contexts)}

Write {len(clips)} structured shot plans. Go:"""

        # Send ALL reference images to the LLM (main + character + location refs)
        image_paths = []
        if has_reference and reference_image_path:
            image_paths.append(reference_image_path)
        for cp in (kwargs.get("character_ref_paths") or []):
            if cp and os.path.isfile(cp):
                image_paths.append(cp)
        for lp in (kwargs.get("location_ref_paths") or []):
            if lp and os.path.isfile(lp):
                image_paths.append(lp)
        if not image_paths:
            image_paths = None
        per_clip_tokens = 700 if uses_generated_images else 520
        max_tokens = max(4096, len(clips) * per_clip_tokens + 1024)
        shot_dicts = self._call_llm_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            thinking_budget=4096,
            image_paths=image_paths,
            json_schema=_music_shot_schema(
                len(clips),
                include_image_fields=uses_generated_images,
            ),
        )
        if not uses_generated_images:
            _discard_unused_image_fields(shot_dicts)
        return shot_dicts

    # ── Convert LLM Output to ShotPlans ──────────────────────────────

    def _convert_to_shots(
        self,
        shot_dicts: list[dict],
        clips: list[dict],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        performer_map: dict[str, str],
        lyrics: Optional[list[dict]],
        speaker_names: dict[str, str],
    ) -> list[ShotPlan]:
        """Convert raw LLM JSON output into validated ShotPlan objects."""
        shots = []
        for i, clip in enumerate(clips):
            raw = shot_dicts[i] if i < len(shot_dicts) else {}
            section = (clip.get("label") or "verse").lower()
            strategy = _SECTION_VISUAL_STRATEGY.get(section, _SECTION_VISUAL_STRATEGY["verse"])
            duration = clip.get("end", 0) - clip.get("start", 0)

            # Parse subjects
            subjects = []
            for s in raw.get("subjects_on_screen", []):
                if isinstance(s, dict):
                    subjects.append(SubjectRef.from_dict(s))
                elif isinstance(s, str):
                    subjects.append(SubjectRef(visual_description=s))

            # Fallback subjects from performer map
            if not subjects and section in performer_map:
                subjects.append(SubjectRef(visual_description=performer_map[section], position_or_relation="center frame"))

            # Parse camera plan
            cam_raw = raw.get("camera_plan", {})
            camera = CameraPlan(
                framing=cam_raw.get("framing", strategy["camera_default"]),
                angle=cam_raw.get("angle"),
                movement=cam_raw.get("movement"),
                movement_intensity=cam_raw.get("movement_intensity", strategy["movement_intensity"]),
                lens_feel=cam_raw.get("lens_feel"),
            )

            # Parse audio plan
            audio_raw = raw.get("audio_plan", {})
            audio = AudioPlan(
                mode=audio_raw.get("mode", "music_driven"),
                ambience=audio_raw.get("ambience"),
                timing_anchor="audio",
            )

            # Parse dialogue beats if present
            dialogue_beats = None
            if raw.get("dialogue_beats"):
                dialogue_beats = [DialogueBeat.from_dict(db) for db in raw["dialogue_beats"]]

            # Determine image strategy
            image_strategy = "reference_edit" if has_reference else "fresh_generation"
            if section == "instrumental" and not has_reference:
                image_strategy = "fresh_generation"

            shot = ShotPlan(
                shot_id=self._make_shot_id(i, "mv"),
                index=i,
                duration_sec=duration,
                skill_type="music_video",
                scene_goal=raw.get("scene_goal", f"{section} clip — {strategy['energy']} energy"),
                narrative_role=section,
                scene_type=raw.get("scene_type", "performance" if section != "instrumental" else "atmospheric"),
                source_mode_preference="i2v" if has_reference else "t2v",
                image_strategy=image_strategy,
                continuity_strategy="independent",
                subjects_on_screen=subjects,
                spatial_setup=raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                visual_style=raw.get("visual_style", ""),
                lighting=raw.get("lighting", ""),
                mood=raw.get("mood", strategy["energy"]),
                action_beats=raw.get("action_beats", []),
                performance_beats=raw.get("performance_beats"),
                dialogue_beats=dialogue_beats,
                camera_plan=camera,
                audio_plan=audio,
                ending_beat=raw.get("ending_beat", ""),
                metadata={
                    "section": section,
                    "beat_count": clip.get("beat_count", 0),
                    "bpm": clip.get("bpm", 120),
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
