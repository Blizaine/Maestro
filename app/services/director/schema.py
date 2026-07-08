"""
Shared Director Schema — canonical types for all skill planners and renderers.

All planners output ProductionPlan containing ShotPlan objects.
Renderers consume ShotPlan to produce mode-specific prompts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any


# ── Asset & Reference Types ──────────────────────────────────────────

@dataclass
class AssetRef:
    id: str
    type: str  # "image" | "audio" | "video" | "text"
    uri: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type}
        if self.uri is not None:
            d["uri"] = self.uri
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class BeatEvent:
    time_sec: float
    strength: Optional[float] = None
    section: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"time_sec": self.time_sec}
        if self.strength is not None:
            d["strength"] = self.strength
        if self.section:
            d["section"] = self.section
        return d


@dataclass
class SpeakerMapEntry:
    speaker_id: str
    name: Optional[str] = None
    voice_description: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"speaker_id": self.speaker_id}
        if self.name:
            d["name"] = self.name
        if self.voice_description:
            d["voice_description"] = self.voice_description
        return d


@dataclass
class ReferenceAssets:
    start_image: Optional[AssetRef] = None
    end_image: Optional[AssetRef] = None
    audio: Optional[AssetRef] = None
    transcript: Optional[str] = None
    lyrics: Optional[str] = None
    beat_map: Optional[list[BeatEvent]] = None
    speaker_map: Optional[list[SpeakerMapEntry]] = None

    def to_dict(self) -> dict:
        d: dict = {}
        if self.start_image:
            d["start_image"] = self.start_image.to_dict()
        if self.end_image:
            d["end_image"] = self.end_image.to_dict()
        if self.audio:
            d["audio"] = self.audio.to_dict()
        if self.transcript:
            d["transcript"] = self.transcript
        if self.lyrics:
            d["lyrics"] = self.lyrics
        if self.beat_map:
            d["beat_map"] = [b.to_dict() for b in self.beat_map]
        if self.speaker_map:
            d["speaker_map"] = [s.to_dict() for s in self.speaker_map]
        return d


# ── Character Profile ────────────────────────────────────────────────

@dataclass
class CharacterProfile:
    id: str
    physical_description: str
    display_name: Optional[str] = None
    wardrobe: Optional[str] = None
    voice_description: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"id": self.id, "physical_description": self.physical_description}
        if self.display_name:
            d["display_name"] = self.display_name
        if self.wardrobe:
            d["wardrobe"] = self.wardrobe
        if self.voice_description:
            d["voice_description"] = self.voice_description
        return d

    @staticmethod
    def from_dict(d: dict) -> "CharacterProfile":
        return CharacterProfile(
            id=d["id"],
            physical_description=d["physical_description"],
            display_name=d.get("display_name"),
            wardrobe=d.get("wardrobe"),
            voice_description=d.get("voice_description"),
        )


# ── Shot-level Types ─────────────────────────────────────────────────

@dataclass
class SubjectRef:
    visual_description: str
    character_id: Optional[str] = None
    position_or_relation: Optional[str] = None
    # The personal NAME the screenplay uses for this character in this
    # shot (e.g. "Nancy", "Blaine"). Often invented by the screenplay
    # LLM when the user-supplied character profile is generic. Carried
    # through to the polish layer so it can substitute screenplay-
    # invented names with descriptors in narrative prose.
    speaker_name: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"visual_description": self.visual_description}
        if self.character_id:
            d["character_id"] = self.character_id
        if self.position_or_relation:
            d["position_or_relation"] = self.position_or_relation
        if self.speaker_name:
            d["speaker_name"] = self.speaker_name
        return d

    @staticmethod
    def from_dict(d: dict) -> "SubjectRef":
        return SubjectRef(
            visual_description=d["visual_description"],
            character_id=d.get("character_id"),
            position_or_relation=d.get("position_or_relation"),
            speaker_name=d.get("speaker_name"),
        )


@dataclass
class DialogueBeat:
    spoken_text: str
    speaker_id: Optional[str] = None
    delivery: Optional[str] = None
    physical_cue: Optional[str] = None
    priority: str = "medium"  # "low" | "medium" | "high"

    def to_dict(self) -> dict:
        d = {"spoken_text": self.spoken_text, "priority": self.priority}
        if self.speaker_id:
            d["speaker_id"] = self.speaker_id
        if self.delivery:
            d["delivery"] = self.delivery
        if self.physical_cue:
            d["physical_cue"] = self.physical_cue
        return d

    @staticmethod
    def from_dict(d: dict) -> "DialogueBeat":
        return DialogueBeat(
            spoken_text=d["spoken_text"],
            speaker_id=d.get("speaker_id"),
            delivery=d.get("delivery"),
            physical_cue=d.get("physical_cue"),
            priority=d.get("priority", "medium"),
        )


@dataclass
class CameraPlan:
    framing: str  # "close-up", "medium shot", "wide shot", etc.
    angle: Optional[str] = None
    movement: Optional[str] = None
    movement_intensity: str = "subtle"  # "static" | "subtle" | "moderate" | "dynamic"
    lens_feel: Optional[str] = None
    reframing_notes: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"framing": self.framing, "movement_intensity": self.movement_intensity}
        if self.angle:
            d["angle"] = self.angle
        if self.movement:
            d["movement"] = self.movement
        if self.lens_feel:
            d["lens_feel"] = self.lens_feel
        if self.reframing_notes:
            d["reframing_notes"] = self.reframing_notes
        return d

    @staticmethod
    def from_dict(d: dict) -> "CameraPlan":
        return CameraPlan(
            framing=d["framing"],
            angle=d.get("angle"),
            movement=d.get("movement"),
            movement_intensity=d.get("movement_intensity", "subtle"),
            lens_feel=d.get("lens_feel"),
            reframing_notes=d.get("reframing_notes"),
        )


@dataclass
class AudioPlan:
    mode: str  # "generated_audio" | "audio_driven" | "dialogue_driven" | "music_driven" | "ambient_only"
    ambience: Optional[str] = None
    effects: Optional[list[str]] = None
    vocal_style: Optional[str] = None
    timing_anchor: str = "balanced"  # "audio" | "video" | "balanced"
    lip_sync_critical: bool = False

    def to_dict(self) -> dict:
        d = {"mode": self.mode, "timing_anchor": self.timing_anchor, "lip_sync_critical": self.lip_sync_critical}
        if self.ambience:
            d["ambience"] = self.ambience
        if self.effects:
            d["effects"] = self.effects
        if self.vocal_style:
            d["vocal_style"] = self.vocal_style
        return d

    @staticmethod
    def from_dict(d: dict) -> "AudioPlan":
        return AudioPlan(
            mode=d["mode"],
            ambience=d.get("ambience"),
            effects=d.get("effects"),
            vocal_style=d.get("vocal_style"),
            timing_anchor=d.get("timing_anchor", "balanced"),
            lip_sync_critical=d.get("lip_sync_critical", False),
        )


# ── ShotPlan — the core unit ─────────────────────────────────────────

VALID_SKILL_TYPES = {"music_video", "short_film", "podcast", "viral_video"}
VALID_SOURCE_MODES = {"t2v", "i2v", "a2v", "retake", "extend"}
VALID_IMAGE_STRATEGIES = {"reference_edit", "reference_inspired", "fresh_generation", "none"}
VALID_CONTINUITY_STRATEGIES = {"independent", "continuous", "extend_previous"}

@dataclass
class ShotPlan:
    shot_id: str
    index: int
    duration_sec: float
    skill_type: str  # "music_video" | "short_film" | "podcast" | "viral_video"

    scene_goal: str
    subjects_on_screen: list[SubjectRef]
    spatial_setup: str
    environment: str
    visual_style: str
    lighting: str
    mood: str
    action_beats: list[str]
    camera_plan: CameraPlan
    audio_plan: AudioPlan
    ending_beat: str

    # Optional fields
    narrative_role: Optional[str] = None  # "intro", "rising_action", "climax", etc.
    scene_type: Optional[str] = None  # "dialogue", "action", "opening", "closing", etc.
    source_mode_preference: Optional[str] = None  # t2v, i2v, a2v, retake, extend
    image_strategy: Optional[str] = None
    continuity_strategy: str = "independent"
    performance_beats: Optional[list[str]] = None
    dialogue_beats: Optional[list[DialogueBeat]] = None
    constraints: Optional[list[str]] = None
    continuity_refs: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None

    # LLM-generated prompts (filled by planner, used directly if available)
    video_prompt: Optional[str] = None
    image_prompt: Optional[str] = None
    # For sliding window: per-window prompts for long continuous scenes (>22s)
    window_prompts: Optional[list[str]] = None
    # Visual changes: what transforms during the scene (used to ensure image_prompt = BEFORE state)
    visual_changes: Optional[list[str]] = None
    # Image source: "original" = edit from user's reference photo (default),
    # "previous" = edit from previous scene's output (for continuity chains)
    image_source: Optional[str] = None
    # Keyframe injection prompts: image edit prompts for mid-scene visual milestones.
    # Each keyframe is generated by Qwen chained from the previous output.
    # Injected into the video as guiding frames at window boundaries.
    keyframe_prompts: Optional[list[str]] = None

    def to_dict(self) -> dict:
        d = {
            "shot_id": self.shot_id,
            "index": self.index,
            "duration_sec": self.duration_sec,
            "skill_type": self.skill_type,
            "scene_goal": self.scene_goal,
            "subjects_on_screen": [s.to_dict() for s in self.subjects_on_screen],
            "spatial_setup": self.spatial_setup,
            "environment": self.environment,
            "visual_style": self.visual_style,
            "lighting": self.lighting,
            "mood": self.mood,
            "action_beats": self.action_beats,
            "camera_plan": self.camera_plan.to_dict(),
            "audio_plan": self.audio_plan.to_dict(),
            "ending_beat": self.ending_beat,
            "continuity_strategy": self.continuity_strategy,
        }
        if self.narrative_role:
            d["narrative_role"] = self.narrative_role
        if self.scene_type:
            d["scene_type"] = self.scene_type
        if self.source_mode_preference:
            d["source_mode_preference"] = self.source_mode_preference
        if self.image_strategy:
            d["image_strategy"] = self.image_strategy
        if self.performance_beats:
            d["performance_beats"] = self.performance_beats
        if self.dialogue_beats:
            d["dialogue_beats"] = [db.to_dict() for db in self.dialogue_beats]
        if self.constraints:
            d["constraints"] = self.constraints
        if self.continuity_refs:
            d["continuity_refs"] = self.continuity_refs
        if self.metadata:
            d["metadata"] = self.metadata
        if self.video_prompt:
            d["video_prompt"] = self.video_prompt
        if self.image_prompt:
            d["image_prompt"] = self.image_prompt
        if self.window_prompts:
            d["window_prompts"] = self.window_prompts
        if self.visual_changes:
            d["visual_changes"] = self.visual_changes
        if self.image_source and self.image_source != "original":
            d["image_source"] = self.image_source
        if self.keyframe_prompts:
            d["keyframe_prompts"] = self.keyframe_prompts
        return d

    @staticmethod
    def from_dict(d: dict) -> "ShotPlan":
        return ShotPlan(
            shot_id=d.get("shot_id", f"shot_{d.get('index', 0)}"),
            index=d.get("index", 0),
            duration_sec=d.get("duration_sec", 10.0),
            skill_type=d.get("skill_type", "short_film"),
            scene_goal=d.get("scene_goal", ""),
            subjects_on_screen=[SubjectRef.from_dict(s) for s in d.get("subjects_on_screen", [])],
            spatial_setup=d.get("spatial_setup", ""),
            environment=d.get("environment", ""),
            visual_style=d.get("visual_style", ""),
            lighting=d.get("lighting", ""),
            mood=d.get("mood", ""),
            action_beats=d.get("action_beats", []),
            camera_plan=CameraPlan.from_dict(d["camera_plan"]) if "camera_plan" in d else CameraPlan(framing="medium shot"),
            audio_plan=AudioPlan.from_dict(d["audio_plan"]) if "audio_plan" in d else AudioPlan(mode="ambient_only"),
            ending_beat=d.get("ending_beat", ""),
            narrative_role=d.get("narrative_role"),
            scene_type=d.get("scene_type"),
            source_mode_preference=d.get("source_mode_preference"),
            image_strategy=d.get("image_strategy"),
            continuity_strategy=d.get("continuity_strategy", "independent"),
            performance_beats=d.get("performance_beats"),
            dialogue_beats=[DialogueBeat.from_dict(db) for db in d.get("dialogue_beats", [])] if d.get("dialogue_beats") else None,
            constraints=d.get("constraints"),
            continuity_refs=d.get("continuity_refs"),
            metadata=d.get("metadata"),
            video_prompt=d.get("video_prompt"),
            image_prompt=d.get("image_prompt"),
            window_prompts=d.get("window_prompts"),
            visual_changes=d.get("visual_changes"),
            image_source=d.get("image_source"),
            keyframe_prompts=d.get("keyframe_prompts"),
        )


# ── ProductionPlan — top-level output from planners ──────────────────

@dataclass
class ProductionPlan:
    skill_type: str  # "music_video" | "short_film" | "podcast" | "viral_video"
    shots: list[ShotPlan]
    title: Optional[str] = None
    global_style: Optional[str] = None
    total_duration_sec: Optional[float] = None
    reference_assets: Optional[ReferenceAssets] = None
    characters: Optional[list[CharacterProfile]] = None
    continuity_notes: Optional[list[str]] = None

    def to_dict(self) -> dict:
        d = {
            "skill_type": self.skill_type,
            "shots": [s.to_dict() for s in self.shots],
        }
        if self.title:
            d["title"] = self.title
        if self.global_style:
            d["global_style"] = self.global_style
        if self.total_duration_sec is not None:
            d["total_duration_sec"] = self.total_duration_sec
        if self.reference_assets:
            d["reference_assets"] = self.reference_assets.to_dict()
        if self.characters:
            d["characters"] = [c.to_dict() for c in self.characters]
        if self.continuity_notes:
            d["continuity_notes"] = self.continuity_notes
        return d

    @staticmethod
    def from_dict(d: dict) -> "ProductionPlan":
        return ProductionPlan(
            skill_type=d["skill_type"],
            shots=[ShotPlan.from_dict(s) for s in d.get("shots", [])],
            title=d.get("title"),
            global_style=d.get("global_style"),
            total_duration_sec=d.get("total_duration_sec"),
            reference_assets=None,  # reconstructed externally if needed
            characters=[CharacterProfile.from_dict(c) for c in d.get("characters", [])] if d.get("characters") else None,
            continuity_notes=d.get("continuity_notes"),
        )

    def get_character(self, character_id: str) -> Optional[CharacterProfile]:
        if not self.characters:
            return None
        for c in self.characters:
            if c.id == character_id:
                return c
        return None


# ── Rendered Prompt Output ───────────────────────────────────────────

@dataclass
class RenderedPrompts:
    """Output from render adapters — the final prompts ready for model dispatch."""
    shot_id: str
    video_prompt: Optional[str] = None
    image_prompt: Optional[str] = None
    render_mode: Optional[str] = None  # which renderer produced this
    warnings: Optional[list[str]] = None
    compression_delta: Optional[int] = None  # tokens saved by compression

    def to_dict(self) -> dict:
        d = {"shot_id": self.shot_id}
        if self.video_prompt is not None:
            d["video_prompt"] = self.video_prompt
        if self.image_prompt is not None:
            d["image_prompt"] = self.image_prompt
        if self.render_mode:
            d["render_mode"] = self.render_mode
        if self.warnings:
            d["warnings"] = self.warnings
        if self.compression_delta is not None:
            d["compression_delta"] = self.compression_delta
        return d
