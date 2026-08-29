"""Persistent local character references shared by Studio and Director.

The library deliberately stores copies beneath ``uploads/characters`` rather
than remembering temporary upload paths.  A saved character can therefore be
recalled from another browser (including a Tailscale-connected phone) without
depending on browser-local storage.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path


_LOCK = threading.RLock()
_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def _root() -> Path:
    root = Path.cwd() / "uploads" / "characters"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _index_path() -> Path:
    return _root() / "index.json"


def _load_index() -> dict:
    path = _index_path()
    if not path.is_file():
        return {"version": 1, "characters": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"version": 1, "characters": []}
    if not isinstance(data, dict) or not isinstance(data.get("characters"), list):
        return {"version": 1, "characters": []}
    return data


def _save_index(data: dict) -> None:
    path = _index_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _source_path(raw_path: str, extensions: set[str]) -> Path:
    source = Path(str(raw_path or "")).expanduser().resolve()
    uploads = (Path.cwd() / "uploads").resolve()
    try:
        source.relative_to(uploads)
    except ValueError as error:
        raise ValueError("Character media must be uploaded to Maestro first.") from error
    if not source.is_file():
        raise ValueError(f"Character media was not found: {source}")
    if source.suffix.lower() not in extensions:
        raise ValueError(f"Unsupported character media format: {source.suffix or 'unknown'}")
    return source


def _probe_duration(path: Path) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        duration = float(completed.stdout.strip())
        return round(duration, 3) if duration > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _video_has_audio(path: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return bool(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _extract_voice(video_path: Path, destination: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
    except FileNotFoundError as error:
        raise ValueError("FFmpeg is required to extract a character voice from video.") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or str(error)).strip()[-500:]
        raise ValueError(f"The selected video has no usable voice track: {detail}") from error


def _public_record(record: dict) -> dict:
    result = dict(record)
    character_id = result["id"]
    visual = dict(result["visual"])
    visual_storage_name = str(
        visual.get("storage_name") or Path(str(visual.get("path") or "visual")).name
    )
    visual["path"] = str((_root() / character_id / visual_storage_name).resolve())
    visual["url"] = f"/api/v1/characters/{character_id}/media/visual"
    result["visual"] = visual
    if isinstance(result.get("voice"), dict):
        voice = dict(result["voice"])
        voice_storage_name = str(
            voice.get("storage_name") or Path(str(voice.get("path") or "voice")).name
        )
        voice["path"] = str((_root() / character_id / voice_storage_name).resolve())
        voice["url"] = f"/api/v1/characters/{character_id}/media/voice"
        result["voice"] = voice
    return result


def list_characters() -> list[dict]:
    with _LOCK:
        records = _load_index().get("characters", [])
        return [_public_record(record) for record in records if isinstance(record, dict)]


def create_character(
    *,
    name: str,
    visual_path: str,
    visual_type: str,
    voice_path: str | None = None,
    use_video_voice: bool = False,
) -> dict:
    clean_name = " ".join(str(name or "").strip().split())[:120]
    if not clean_name:
        raise ValueError("Give this character a name.")
    kind = str(visual_type or "").strip().lower()
    if kind not in {"image", "video"}:
        raise ValueError("A saved character needs an image or video reference.")
    visual_source = _source_path(
        visual_path,
        _IMAGE_EXTENSIONS if kind == "image" else _VIDEO_EXTENSIONS,
    )
    voice_source = _source_path(voice_path, _AUDIO_EXTENSIONS) if voice_path else None
    if use_video_voice and kind != "video":
        raise ValueError("Only a video character reference can supply its own voice.")

    character_id = uuid.uuid4().hex[:16]
    character_dir = (_root() / character_id).resolve()
    character_dir.mkdir(parents=False, exist_ok=False)
    try:
        visual_name = f"visual{visual_source.suffix.lower()}"
        visual_destination = character_dir / visual_name
        shutil.copy2(visual_source, visual_destination)

        voice_destination: Path | None = None
        if voice_source is not None:
            voice_destination = character_dir / f"voice{voice_source.suffix.lower()}"
            shutil.copy2(voice_source, voice_destination)
        elif use_video_voice:
            voice_destination = character_dir / "voice.wav"
            _extract_voice(visual_destination, voice_destination)

        now = time.time()
        visual_duration = _probe_duration(visual_destination) if kind == "video" else None
        if kind == "video" and visual_duration is not None and visual_duration < 2.0:
            raise ValueError(
                f"The character video is {visual_duration:.2f}s; MiniMax H3 Omni requires at least 2 seconds."
            )
        voice_duration = _probe_duration(voice_destination) if voice_destination is not None else None
        if voice_destination is not None and voice_duration is not None and voice_duration < 2.0:
            raise ValueError(
                f"The voice reference is {voice_duration:.2f}s; MiniMax H3 Omni requires at least 2 seconds."
            )
        record = {
            "id": character_id,
            "name": clean_name,
            "created_at": now,
            "updated_at": now,
            "visual": {
                "type": kind,
                "path": str(visual_destination),
                "storage_name": visual_name,
                "filename": visual_source.name,
                "duration_seconds": visual_duration,
                "has_audio": _video_has_audio(visual_destination) if kind == "video" else False,
            },
            "voice": (
                {
                    "path": str(voice_destination),
                    "storage_name": voice_destination.name,
                    "filename": voice_source.name if voice_source is not None else f"{clean_name} voice.wav",
                    "duration_seconds": voice_duration,
                }
                if voice_destination is not None else None
            ),
        }
        with _LOCK:
            data = _load_index()
            data.setdefault("characters", []).append(record)
            _save_index(data)
        return _public_record(record)
    except Exception:
        if character_dir.is_dir():
            shutil.rmtree(character_dir, ignore_errors=True)
        raise


def get_character_media(character_id: str, slot: str) -> Path | None:
    if slot not in {"visual", "voice"}:
        return None
    with _LOCK:
        record = next(
            (
                item for item in _load_index().get("characters", [])
                if isinstance(item, dict) and item.get("id") == character_id
            ),
            None,
        )
    media = record.get(slot) if isinstance(record, dict) else None
    if not isinstance(media, dict):
        return None
    storage_name = str(
        media.get("storage_name") or Path(str(media.get("path") or slot)).name
    )
    candidate = (_root() / character_id / storage_name).resolve()
    try:
        candidate.relative_to(_root())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def delete_character(character_id: str) -> bool:
    clean_id = str(character_id or "").strip()
    with _LOCK:
        data = _load_index()
        records = data.get("characters", [])
        remaining = [item for item in records if not isinstance(item, dict) or item.get("id") != clean_id]
        if len(remaining) == len(records):
            return False
        data["characters"] = remaining
        _save_index(data)

    target = (_root() / clean_id).resolve()
    try:
        target.relative_to(_root())
    except ValueError:
        return True
    if target.is_dir():
        shutil.rmtree(target)
    return True
