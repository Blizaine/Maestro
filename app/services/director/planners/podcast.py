"""
Podcast Planner — creates a ProductionPlan from podcast audio/transcript.

Identifies speaker changes, reaction shots, B-roll opportunities, and
determines when visuals should stay simple during long dialogue stretches.

Outputs: ProductionPlan with ShotPlan objects.
"""

from __future__ import annotations
from typing import Optional, Any

from ..schema import (
    ProductionPlan, ShotPlan, CharacterProfile, ReferenceAssets,
    AssetRef, SubjectRef, DialogueBeat, CameraPlan, AudioPlan,
    SpeakerMapEntry,
)
from ..policies import build_character_rules_block, build_camera_style_block
from .base import BasePlanner


class PodcastPlanner(BasePlanner):
    skill_type = "podcast"

    def plan(
        self,
        transcript: Optional[list[dict]] = None,
        audio_path: Optional[str] = None,
        reference_image_path: Optional[str] = None,
        speaker_mappings: Optional[dict] = None,
        characters: Optional[list[dict]] = None,
        visual_style: str = "clean, professional studio lighting",
        target_duration: Optional[float] = None,
        clips: Optional[list[dict]] = None,
        **kwargs,
    ) -> ProductionPlan:
        """Create a ProductionPlan for a podcast visualization.

        Args:
            transcript: Timestamped transcript with speaker tags.
            audio_path: Path to podcast audio.
            reference_image_path: Path to host reference photo.
            speaker_mappings: {speaker_id: {name, role}}.
            characters: Character descriptions for hosts/guests.
            visual_style: Visual style preference.
            target_duration: Total duration in seconds.
            clips: Pre-segmented clips (if available from audio analysis).
        """
        has_reference = bool(reference_image_path)

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
        char_profiles = []
        if characters:
            char_profiles = [
                CharacterProfile(
                    id=f"host_{i}",
                    display_name=c.get("name", ""),
                    physical_description=c.get("description", "person"),
                    voice_description=c.get("role", "host"),
                )
                for i, c in enumerate(characters)
            ]
        elif speaker_mappings:
            char_profiles = [
                CharacterProfile(
                    id=sid,
                    display_name=info.get("name", sid),
                    physical_description=f"the {info.get('name', 'host')}",
                    voice_description=info.get("role", "speaking"),
                )
                for sid, info in speaker_mappings.items()
            ]

        ref_assets = ReferenceAssets(
            start_image=AssetRef(id="ref_image", type="image", uri=reference_image_path) if has_reference else None,
            audio=AssetRef(id="audio", type="audio", uri=audio_path) if audio_path else None,
            transcript="\n".join(l.get("text", "") for l in (transcript or []) if l.get("text", "").strip()),
            speaker_map=[
                SpeakerMapEntry(speaker_id=sid, name=info.get("name", ""), voice_description=info.get("role", ""))
                for sid, info in (speaker_mappings or {}).items()
            ] if speaker_mappings else None,
        )

        # Use pre-segmented clips or segment from transcript
        if clips:
            shots = self._plan_from_clips(
                clips=clips,
                transcript=transcript,
                speaker_mappings=speaker_mappings,
                char_profiles=char_profiles,
                has_reference=has_reference,
                reference_image_path=reference_image_path,
                visual_style=visual_style,
            )
        else:
            shots = self._plan_from_transcript(
                transcript=transcript or [],
                speaker_mappings=speaker_mappings,
                char_profiles=char_profiles,
                has_reference=has_reference,
                reference_image_path=reference_image_path,
                visual_style=visual_style,
                target_duration=target_duration,
            )

        total_dur = sum(s.duration_sec for s in shots) if shots else target_duration

        return ProductionPlan(
            skill_type="podcast",
            title=None,
            global_style=visual_style,
            total_duration_sec=total_dur,
            reference_assets=ref_assets,
            characters=char_profiles if char_profiles else None,
            shots=shots,
            continuity_notes=[
                "Podcast — visuals should serve the conversation, not distract",
                "Long dialogue stretches should use simple, steady framing",
                "Use reaction shots and B-roll for variety, not constant switching",
                "Speaker changes are natural cut points",
            ],
        )

    def _plan_from_clips(
        self,
        clips: list[dict],
        transcript: Optional[list[dict]],
        speaker_mappings: Optional[dict],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        reference_image_path: Optional[str],
        visual_style: str,
    ) -> list[ShotPlan]:
        """Plan shots from pre-segmented clips (similar to audio-driven short film)."""
        speaker_names = {sid: info.get("name", sid) for sid, info in (speaker_mappings or {}).items()}

        clip_contexts = []
        for i, clip in enumerate(clips):
            start_sec = clip.get("start", 0)
            end_sec = clip.get("end", start_sec + 10)
            duration = end_sec - start_sec

            # Gather dialogue in this clip
            dialogue_lines = []
            speakers_here = set()
            if transcript:
                for l in transcript:
                    if l.get("start", 0) < end_sec and l.get("end", 0) > start_sec:
                        spk = l.get("speaker", "")
                        text = l.get("text", "")
                        if text.strip():
                            spk_name = speaker_names.get(spk, spk)
                            dialogue_lines.append(f'{spk_name}: "{text}"' if spk_name else f'"{text}"')
                            if spk:
                                speakers_here.add(spk)

            char_info = ""
            if speakers_here:
                on_screen = [speaker_names.get(s, s) for s in speakers_here]
                char_info = f" Speakers: {', '.join(on_screen)}."

            dial_text = f" Dialogue: {' / '.join(dialogue_lines[:3])}" if dialogue_lines else " (no dialogue)"
            ctx = f"Segment {i + 1}: {duration:.1f}s.{char_info}{dial_text}"
            clip_contexts.append(ctx)

        char_rules = build_character_rules_block(has_reference, char_profiles if char_profiles else None)
        camera_block = build_camera_style_block()

        system_prompt = f"""You are a podcast visual director creating a structured shot plan.

{f"You are given a REFERENCE PHOTO of the host(s)." if has_reference else ""}

Plan each segment as a visual scene. Podcasts need:
- Direct-to-camera shots for the active speaker
- Reaction shots of the listener
- B-roll or environmental shots for variety during long stretches
- Simple, steady framing during important dialogue — don't over-complicate

{char_rules}

{camera_block}

PODCAST PLANNING RULES:
- Favor steady framing for dialogue: medium shots, clean backgrounds.
- Use reaction shots when non-speaking host has a strong response.
- Insert B-roll or cutaway for topic transitions or long monologues.
- Keep visual complexity LOW during crucial discussion points.
- Speaker changes are natural visual transitions.

OUTPUT FORMAT — respond with ONLY a JSON array:
[
  {{
    "scene_goal": "Active speaker segment",
    "scene_type": "speaker|reaction|b_roll|transition",
    "subjects_on_screen": [{{"visual_description": "the host in the blue shirt"}}],
    "spatial_setup": "seated at desk, facing camera",
    "environment": "podcast studio",
    "visual_style": "{visual_style}",
    "lighting": "soft studio lighting",
    "mood": "conversational",
    "action_beats": ["host gestures while speaking"],
    "dialogue_beats": [{{"speaker_id": "host_0", "spoken_text": "key line", "delivery": "animated"}}],
    "camera_plan": {{"framing": "medium shot", "movement": "static", "movement_intensity": "static"}},
    "audio_plan": {{"mode": "dialogue_driven", "lip_sync_critical": true}},
    "ending_beat": "host nods thoughtfully"
  }}
]

Output exactly {len(clips)} shot plans. Go:"""

        user_prompt = f"""Visual Style: {visual_style}

Segments:
{chr(10).join(clip_contexts)}

Write {len(clips)} structured shot plans. Go:"""

        image_paths = [reference_image_path] if has_reference and reference_image_path else None
        max_tokens = max(1024, len(clips) * 300 + 512)

        shot_dicts = self._call_llm_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            image_paths=image_paths,
        )

        shots = []
        for i, clip in enumerate(clips):
            raw = shot_dicts[i] if i < len(shot_dicts) else {}
            duration = clip.get("end", 0) - clip.get("start", 0)

            subjects = [SubjectRef.from_dict(s) if isinstance(s, dict) else SubjectRef(visual_description=str(s))
                        for s in raw.get("subjects_on_screen", [])]

            cam_raw = raw.get("camera_plan", {})
            camera = CameraPlan(
                framing=cam_raw.get("framing", "medium shot"),
                movement=cam_raw.get("movement", "static"),
                movement_intensity=cam_raw.get("movement_intensity", "static"),
            )

            audio_raw = raw.get("audio_plan", {})
            audio = AudioPlan(
                mode=audio_raw.get("mode", "dialogue_driven"),
                timing_anchor="audio",
                lip_sync_critical=audio_raw.get("lip_sync_critical", True),
            )

            dialogue_beats = [DialogueBeat.from_dict(db) for db in raw.get("dialogue_beats", [])] if raw.get("dialogue_beats") else None

            shot = ShotPlan(
                shot_id=self._make_shot_id(i, "pod"),
                index=i,
                duration_sec=duration,
                skill_type="podcast",
                scene_goal=raw.get("scene_goal", f"Segment {i + 1}"),
                scene_type=raw.get("scene_type", "speaker"),
                source_mode_preference="a2v",
                image_strategy="reference_edit" if has_reference else "fresh_generation",
                continuity_strategy="continuous" if i > 0 else "independent",
                subjects_on_screen=subjects,
                spatial_setup=raw.get("spatial_setup", ""),
                environment=raw.get("environment", "podcast studio"),
                visual_style=raw.get("visual_style", visual_style),
                lighting=raw.get("lighting", "studio lighting"),
                mood=raw.get("mood", "conversational"),
                action_beats=raw.get("action_beats", []),
                dialogue_beats=dialogue_beats,
                camera_plan=camera,
                audio_plan=audio,
                ending_beat=raw.get("ending_beat", ""),
            )
            shots.append(shot)

        return shots

    def _plan_from_transcript(
        self,
        transcript: list[dict],
        speaker_mappings: Optional[dict],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        reference_image_path: Optional[str],
        visual_style: str,
        target_duration: Optional[float],
    ) -> list[ShotPlan]:
        """Segment transcript into clips and plan — stub for auto-segmentation."""
        # For now, create one shot per ~15s chunk of transcript
        if not transcript:
            return []

        total_dur = target_duration or (transcript[-1].get("end", 60) if transcript else 60)
        chunk_size = 15.0
        num_chunks = max(1, int(total_dur / chunk_size))

        clips = []
        for i in range(num_chunks):
            clips.append({
                "start": i * chunk_size,
                "end": min((i + 1) * chunk_size, total_dur),
            })

        return self._plan_from_clips(
            clips=clips,
            transcript=transcript,
            speaker_mappings=speaker_mappings,
            char_profiles=char_profiles,
            has_reference=has_reference,
            reference_image_path=reference_image_path,
            visual_style=visual_style,
        )
