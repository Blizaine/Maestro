"""Capability checks for models used by Director's automated workflows.

Studio exposes individual controls and can support specialized models that
require a trajectory, control video, mask, or other manual input.  Director
does not provide those inputs.  Its selectors therefore need a stricter
contract than the broad image/video modality filters used by Studio.

Keep these checks data-driven.  Custom checkpoints inherit the capabilities
of their base model and should become eligible automatically without adding
model identifiers to a blacklist.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from services.director_video_strategy import (
    BOUNDED_START_END,
    OMNI_REFERENCE,
    ROLLING_WINDOW,
    SHOT_IMAGES_REQUIRED,
    shot_image_support,
    video_strategy,
)


DIRECTOR_PIPELINE_TYPES = (
    "music_video",
    "short_film_audio",
    "short_film_story",
)


def _result(compatible: bool, reason: str = "") -> dict[str, Any]:
    return {"compatible": bool(compatible), "reason": reason}


def _reference_modes(model_def: Mapping[str, Any]) -> set[str]:
    config = model_def.get("image_ref_choices")
    if not isinstance(config, Mapping):
        return set()
    modes: set[str] = set()
    choices = config.get("choices") or []
    if not isinstance(choices, (list, tuple)):
        return modes
    for choice in choices:
        if isinstance(choice, (list, tuple)) and len(choice) >= 2:
            modes.add(str(choice[1] or ""))
        elif isinstance(choice, Mapping):
            value = choice.get("value")
            if value is not None:
                modes.add(str(value or ""))
    return modes


def _choice_values(config: Any, key: str) -> set[str]:
    if not isinstance(config, Mapping):
        return set()
    raw_values = config.get(key) or []
    if not isinstance(raw_values, (list, tuple)):
        return set()
    values: set[str] = set()
    for item in raw_values:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            values.add(str(item[1] or ""))
        elif isinstance(item, Mapping) and item.get("value") is not None:
            values.add(str(item.get("value") or ""))
        elif isinstance(item, str):
            values.add(item)
    return values


def _max_image_refs(model_def: Mapping[str, Any]) -> int | None:
    value = model_def.get("max_image_refs")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _image_capability(model_def: Mapping[str, Any]) -> dict[str, Any]:
    if not model_def.get("image_outputs"):
        return _result(False, "Director needs a still-image generator.")
    if (
        model_def.get("one_image_ref_needed")
        or model_def.get("at_least_one_image_ref_needed")
    ):
        return _result(
            False,
            "Director must be able to create an establishing image without a reference.",
        )

    modes = _reference_modes(model_def)
    if "KI" not in modes:
        return _result(
            False,
            "Director needs main-image plus reference editing for consistent start frames.",
        )
    if "" not in modes:
        return _result(
            False,
            "Director needs plain generation when no reference image is supplied.",
        )
    return _result(True)


def _base_video_capability(model_def: Mapping[str, Any]) -> dict[str, Any]:
    if model_def.get("image_outputs"):
        return _result(False, "Director video rendering needs a video model.")

    strategy = video_strategy(model_def)
    image_prompt_types = str(model_def.get("image_prompt_types_allowed") or "")
    if strategy == BOUNDED_START_END and "S" not in image_prompt_types:
        return _result(
            False,
            "Director needs start-frame support for this bounded-shot model.",
        )
    if strategy == ROLLING_WINDOW and "S" not in image_prompt_types:
        return _result(
            False,
            "Director needs start-frame image-to-video support.",
        )
    if strategy == ROLLING_WINDOW and not model_def.get("sliding_window"):
        return _result(
            False,
            "Director needs sliding-window support for planned shot durations.",
        )

    custom_guide = model_def.get("custom_guide")
    if isinstance(custom_guide, Mapping) and custom_guide.get("required"):
        return _result(
            False,
            "This model requires a manual guide that Director does not create.",
        )
    custom_guide_modes = _choice_values(
        model_def.get("guide_custom_choices"),
        "choices",
    )
    if custom_guide_modes and "" not in custom_guide_modes:
        return _result(
            False,
            "This model requires a control-video process that Director does not provide.",
        )
    preprocessing_modes = _choice_values(
        model_def.get("guide_preprocessing"),
        "selection",
    )
    if preprocessing_modes and "" not in preprocessing_modes:
        return _result(
            False,
            "This model requires guide preprocessing that Director does not provide.",
        )
    if model_def.get("preprocess_all") or model_def.get("forced_guide_mask_inputs"):
        return _result(
            False,
            "This model requires control-video or mask preprocessing outside Director's workflow.",
        )
    if (
        model_def.get("one_image_ref_needed")
        or model_def.get("at_least_one_image_ref_needed")
    ) and strategy != OMNI_REFERENCE:
        return _result(
            False,
            "This model requires additional references that Director does not guarantee.",
        )
    if (
        model_def.get("audio_guidance")
        and model_def.get("any_audio_prompt")
        and not model_def.get("auto_null_audio")
    ):
        return _result(
            False,
            "This model requires external speech or audio conditioning that story-only Director does not provide.",
        )
    return _result(True)


def _audio_video_capability(
    model_def: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    if not base.get("compatible"):
        return dict(base)
    if model_def.get("director_audio_input_mode") == "reference_manifest":
        return _result(True)
    if not model_def.get("any_audio_prompt"):
        return _result(
            False,
            "This model generates audio but cannot use the uploaded soundtrack as a driver."
            if model_def.get("returns_audio")
            else "This model cannot use the uploaded soundtrack as a driver.",
        )
    if not model_def.get("audio_guide_window_slicing"):
        return _result(
            False,
            "Director needs window-aware soundtrack slicing to keep long renders synchronized.",
        )
    return _result(True)


def _story_video_capability(
    model_def: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    if not base.get("compatible"):
        return dict(base)
    if not model_def.get("returns_audio"):
        return _result(
            False,
            "Story-driven Director needs a model that generates synchronized dialogue and sound.",
        )
    return _result(True)


def _seamless_capability(
    model_def: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    if not base.get("compatible"):
        return dict(base)
    if video_strategy(model_def) != ROLLING_WINDOW:
        return _result(
            False,
            "This model renders independent native-duration shots; use standard Director mode.",
        )
    if not model_def.get("custom_frames_injection"):
        return _result(
            False,
            "Seamless Director needs planned frame/keyframe injection between scenes.",
        )
    return _result(True)


def assess_director_model(
    model_type: str,
    model_def: Mapping[str, Any] | None,
    *,
    family: str = "",
    architecture: str = "",
) -> dict[str, Any]:
    """Return Director compatibility metadata for one resolved model def."""
    if not isinstance(model_def, Mapping):
        unavailable = _result(False, "Model definition is unavailable.")
        return {
            "image": dict(unavailable),
            "video": {
                pipeline_type: dict(unavailable)
                for pipeline_type in DIRECTOR_PIPELINE_TYPES
            } | {"seamless": dict(unavailable)},
            "supports_audio_input": False,
            "generates_audio": False,
            "supports_voice_reference": False,
            "voice_reference_mode": "none",
            "video_strategy": ROLLING_WINDOW,
            "audio_input_mode": "none",
            "reference_mode": "none",
            "shot_image_support": SHOT_IMAGES_REQUIRED,
            "supports_endpoint_continuity": False,
            "clip_min_frames": None,
            "clip_max_frames": None,
            "clip_frame_step": None,
            "max_image_refs": None,
        }

    image = _image_capability(model_def)
    base_video = _base_video_capability(model_def)
    audio_video = _audio_video_capability(model_def, base_video)
    story_video = _story_video_capability(model_def, base_video)
    seamless = _seamless_capability(model_def, base_video)

    architecture_key = str(
        architecture
        or model_def.get("architecture")
        or model_def.get("base_model_type")
        or model_type
    ).lower()
    strategy = video_strategy(model_def)
    audio_input_mode = str(
        model_def.get("director_audio_input_mode") or
        ("generic_audio_guide" if model_def.get("any_audio_prompt") else "none")
    )
    native_voice_reference = strategy == OMNI_REFERENCE
    configured_voice_mode = model_def.get("director_voice_reference_mode")
    if configured_voice_mode is None:
        voice_reference_mode = (
            "native_reference" if native_voice_reference
            else "id_lora" if architecture_key.startswith("ltx2")
            else "none"
        )
    else:
        voice_reference_mode = str(configured_voice_mode or "none")
    return {
        "image": image,
        "video": {
            "music_video": dict(audio_video),
            "short_film_audio": dict(audio_video),
            "short_film_story": story_video,
            "seamless": seamless,
        },
        # Keep input and output audio separate.  The old supports_audio field
        # merged these concepts and is the root cause of incompatible Music
        # Video choices appearing in Director.
        "supports_audio_input": bool(
            model_def.get("any_audio_prompt")
            or audio_input_mode == "reference_manifest"
        ),
        "generates_audio": bool(model_def.get("returns_audio")),
        "supports_voice_reference": voice_reference_mode != "none",
        "voice_reference_mode": voice_reference_mode,
        "video_strategy": strategy,
        "audio_input_mode": audio_input_mode,
        "reference_mode": str(model_def.get("director_reference_mode") or "start_frame"),
        "shot_image_support": shot_image_support(model_def),
        "supports_endpoint_continuity": bool(
            model_def.get("director_endpoint_continuity")
        ),
        "clip_min_frames": model_def.get("frames_minimum"),
        "clip_max_frames": model_def.get("frames_maximum"),
        "clip_frame_step": model_def.get("frames_steps"),
        "max_image_refs": _max_image_refs(model_def),
        "family": family,
    }
