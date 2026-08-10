"""Dependency-light validation for MiniMax H3 Omni reference manifests.

Prompt planning and API preflight need to validate reference metadata before
the generation runtime (and therefore PyTorch) is loaded.  Keep this module
limited to the Python standard library so those paths remain usable in
lightweight tools and CI.
"""

from __future__ import annotations

import os


MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
MINIMAX_H3_MAX_REFERENCES = 12

_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_AUDIO_INTENTS = {"voice", "drive", "style"}
_IMAGE_INTENTS = {"identity", "scene", "style", "composition"}


def validate_reference_manifest(
    references,
    *,
    require_files: bool = True,
    require_visual: bool = True,
    allow_empty: bool = False,
) -> list[dict]:
    """Validate and canonicalize Maestro's JSON Ref2VA manifest.

    Generation keeps the strict defaults: it needs an uploaded image or
    video whose files still exist. Prompt planning is intentionally looser;
    it can describe an Omni sequence before media is added, while the user is
    still building an audio-first manifest, or after a saved file moved.
    """

    if not isinstance(references, list) or not references:
        if allow_empty and (references is None or references == []):
            return []
        raise ValueError("MiniMax H3 Omni Reference needs at least one image or video reference.")
    if len(references) > MINIMAX_H3_MAX_REFERENCES:
        raise ValueError(
            f"MiniMax H3 accepts at most {MINIMAX_H3_MAX_REFERENCES} references, got {len(references)}."
        )

    normalized: list[dict] = []
    counts = {"image": 0, "video": 0, "audio": 0}
    allowed = {"image": _IMAGE_EXTENSIONS, "video": _VIDEO_EXTENSIONS, "audio": _AUDIO_EXTENSIONS}
    for index, raw in enumerate(references):
        if not isinstance(raw, dict):
            raise ValueError(f"Reference {index + 1} must be an object.")
        kind = str(raw.get("type") or raw.get("kind") or "").strip().lower()
        if kind not in allowed:
            raise ValueError(f"Reference {index + 1} must be an image, video, or audio reference.")
        path = str(raw.get("path") or "").strip()
        if not path:
            raise ValueError(f"Reference {index + 1} has no uploaded file.")
        if require_files and not os.path.isfile(path):
            raise ValueError(f"Reference {index + 1} file was not found: {path}")
        extension = os.path.splitext(path)[1].lower()
        if extension and extension not in allowed[kind]:
            raise ValueError(
                f"Reference {index + 1} is marked as {kind}, but {extension or 'its file'} is not a supported {kind} format."
            )

        counts[kind] += 1
        item = dict(raw)
        item["type"] = kind
        item["path"] = path
        item["role"] = str(raw.get("role") or "").strip()[:500]
        if kind == "image":
            image_intent = str(raw.get("image_intent") or "identity").strip().lower()
            if image_intent not in _IMAGE_INTENTS:
                choices = ", ".join(sorted(_IMAGE_INTENTS))
                raise ValueError(
                    f"Reference {index + 1} has invalid image intent "
                    f"{image_intent!r}; expected one of: {choices}."
                )
            item["image_intent"] = image_intent
        if kind == "audio":
            audio_intent = str(raw.get("audio_intent") or "voice").strip().lower()
            if audio_intent not in _AUDIO_INTENTS:
                choices = ", ".join(sorted(_AUDIO_INTENTS))
                raise ValueError(
                    f"Reference {index + 1} has invalid audio intent {audio_intent!r}; "
                    f"expected one of: {choices}."
                )
            item["audio_intent"] = audio_intent
        if kind == "video":
            item["include_audio"] = bool(raw.get("include_audio", True))
            audio_path = str(raw.get("audio_path") or "").strip()
            if audio_path:
                if require_files and not os.path.isfile(audio_path):
                    raise ValueError(f"Reference {index + 1} soundtrack was not found: {audio_path}")
                audio_extension = os.path.splitext(audio_path)[1].lower()
                if audio_extension and audio_extension not in _AUDIO_EXTENSIONS:
                    raise ValueError(f"Reference {index + 1} soundtrack is not a supported audio file.")
                item["audio_path"] = audio_path
        normalized.append(item)

    for kind, limit in (
        ("image", MINIMAX_H3_MAX_REFERENCE_IMAGES),
        ("video", MINIMAX_H3_MAX_REFERENCE_VIDEOS),
        ("audio", MINIMAX_H3_MAX_REFERENCE_AUDIOS),
    ):
        if counts[kind] > limit:
            raise ValueError(f"MiniMax H3 accepts at most {limit} {kind} references, got {counts[kind]}.")
    if require_visual and counts["image"] + counts["video"] == 0:
        raise ValueError("Audio references cannot be used alone; add at least one image or video reference.")
    return normalized
