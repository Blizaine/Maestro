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
from functools import wraps


_LOCK = threading.RLock()
_FILE_LOCKS = [threading.RLock() for _ in range(64)]
_THUMBNAIL_WORKERS = threading.BoundedSemaphore(2)
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
    visual["thumbnail_url"] = (f"/api/v1/characters/{character_id}/media/thumbnail"
                               if visual.get("type") == "video" else visual["url"])
    if isinstance(result.get("image_views"), dict):
        views = result["image_views"]
        version = str(result.get("updated_at", 0))
        result["image_views"] = {
            key: value for key, value in views.items() if key not in {"items", "directory"}
        }
        result["image_views"]["items"] = [
            {**item, "url": f"/api/v1/characters/{character_id}/images/{item['id']}?v={version}"}
            for item in views["items"]
        ]
        visual["thumbnail_url"] = f"/api/v1/characters/{character_id}/media/thumbnail?v={version}"
    result["visual"] = visual
    if isinstance(result.get("refmod"), dict):
        refmod = dict(result["refmod"])
        refmod["path"] = str((_root() / character_id / "reference.safetensors").resolve())
        result["refmod"] = refmod
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


def _get_character_record(character_id: str) -> dict:
    with _LOCK:
        record = next((r for r in _load_index()["characters"]
                       if isinstance(r, dict) and r.get("id") == character_id), None)
    if record is None:
        raise ValueError("Saved character not found.")
    return dict(record)


def get_character(character_id: str) -> dict:
    return _public_record(_get_character_record(character_id))


def _update_character(character_id: str, **changes) -> dict:
    with _LOCK:
        index = _load_index()
        record = next((r for r in index["characters"] if r.get("id") == character_id), None)
        if record is None:
            raise ValueError("Saved character not found.")
        record.update(changes, updated_at=time.time())
        _save_index(index)
        return _public_record(record)


def _character_directory(character_id: str) -> Path:
    if not isinstance(character_id, str) or len(character_id) != 16 or any(c not in "0123456789abcdef" for c in character_id):
        raise ValueError("Invalid character ID.")
    directory = (_root() / character_id).resolve()
    if directory.parent != _root():
        raise ValueError("Invalid character directory.")
    return directory


def _file_operation(function):
    @wraps(function)
    def guarded(character_id: str, *args, **kwargs):
        lock = _FILE_LOCKS[hash(character_id) % len(_FILE_LOCKS)]
        if not lock.acquire(blocking=False):
            raise ValueError("This character is being exported or updated. Try again when it finishes.")
        try:
            return function(character_id, *args, **kwargs)
        finally:
            lock.release()
    return guarded


def get_character_image(character_id: str, view_id: str) -> Path:
    from .character_views import view_path
    record = _get_character_record(character_id)
    return view_path(_character_directory(character_id), record.get("image_views") or {}, view_id)


@_file_operation
def recover_character_images(character_id: str, *, decode_views, update=lambda _: None, force=False) -> dict:
    from .character_views import VERSION, build_cache, original_images, view_path, selection
    from .refmod import load_visual_latent

    record = _get_character_record(character_id)
    directory = _character_directory(character_id)
    previous = record.get("image_views")
    if previous and previous.get("version") == VERSION and not force:
        try:
            for item in previous["items"]:
                view_path(directory, previous, item["id"])
            return _public_record(record)
        except ValueError:
            pass
    refmod = record.get("refmod") or {}
    visual = get_character_media(character_id, "visual")
    use_original = visual is not None and not refmod.get("preview_generated")
    # Older Maestro versions lost this flag on export. External stacked refs
    # have source=stack, whereas Maestro stores independently sampled media.
    reference = directory / "reference.safetensors"
    if reference.is_file():
        from .refmod import inspect_refmod
        info = inspect_refmod(reference)
        if "preview_generated" not in refmod and info["refmod"].get("source") == "stack":
            use_original = False
    if use_original:
        update("Reading original character media…")
        producer = lambda cache: original_images(visual, record["visual"]["type"], cache, update)
    else:
        if not reference.is_file():
            raise ValueError("This character has no original media or RefMod to recover.")
        info, latent = load_visual_latent(reference)
        producer = lambda cache: decode_views(latent, cache, info["refmod"])
    views = build_cache(directory, producer)
    if previous and previous.get("source") == views["source"]:
        valid = {item["id"] for item in views["items"]}
        chosen = [value for value in previous["selected_ids"] if value in valid]
        if chosen:
            cover = previous["cover_id"] if previous["cover_id"] in chosen else chosen[0]
            views = selection(views, chosen, cover)
    return _update_character(character_id, image_views=views)


@_file_operation
def select_character_images(character_id: str, selected_ids, cover_id) -> dict:
    from .character_views import selection
    record = _get_character_record(character_id)
    views = record.get("image_views")
    if not views:
        raise ValueError("Recover this character's images before selecting a cover.")
    selected = selection(views, selected_ids, cover_id)
    for view_id in selected["selected_ids"]:
        get_character_image(character_id, view_id)
    return _update_character(character_id, image_views=selected)


@_file_operation
def export_character_images(character_id: str) -> Path:
    import zipfile
    from .refmod import character_filename
    record = _get_character_record(character_id)
    views = record.get("image_views")
    if not views:
        raise ValueError("Recover this character's images first.")
    directory = _character_directory(character_id)
    output = directory / (character_filename(record["name"]).removesuffix(".maestro.safetensors") + ".images.zip")
    temporary = directory / f"images-{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for view_id in views["selected_ids"]:
                archive.write(get_character_image(character_id, view_id), arcname=f"{view_id}.png")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


@_file_operation
def export_character(character_id: str, *, encode_visual=None) -> Path:
    """Export a normal RefMod plus embedded source media and saved voice."""
    from .refmod import (CHARACTER_META, REFMOD_META, character_filename,
                         embedded_media, inspect_refmod, load_refmod,
                         new_refmod_metadata, write_refmod)

    record = get_character(character_id)
    directory = _character_directory(character_id)
    reference_path = directory / "reference.safetensors"
    visual_path = get_character_media(character_id, "visual")
    voice_path = get_character_media(character_id, "voice")
    if visual_path is None:
        raise ValueError("This character's visual reference is missing.")
    if record.get("voice") and voice_path is None:
        raise ValueError("This character's saved voice is missing. Restore it before exporting.")
    original_voice = (record.get("voice") or {}).get("original_storage_name")
    if original_voice:
        # Keep a compressed recording byte-for-byte in portable exports while
        # local TTS consumers use the decoded WAV made during import.
        original_path = (directory / Path(original_voice).name).resolve()
        if original_path.parent != directory or not original_path.is_file():
            raise ValueError("This character's original voice recording is missing.")
        voice_path = original_path
    if reference_path.is_file():
        info, tensors = load_refmod(reference_path)
        metadata = dict(info["metadata"])
        # Older files may carry their base metadata in a sidecar.
        metadata[REFMOD_META] = json.dumps(info["refmod"], ensure_ascii=False)
        extension = dict(info["character"] or {})
    else:
        if encode_visual is None:
            raise ValueError("This character needs an H3 visual encode before export.")
        latent = encode_visual(visual_path, record["visual"]["type"])
        tensors = {"latent": latent}
        metadata = new_refmod_metadata(record["name"], latent, source=record["visual"]["type"])
        native_meta = json.loads(metadata[REFMOD_META])
        native_meta["view_layout"] = "independent_images"
        metadata[REFMOD_META] = json.dumps(native_meta)
        extension = {}
    extension.update(schema_version=1, name=record["name"], description=record.get("description", ""))
    extension["visual"] = embedded_media(tensors, visual_path, "maestro.visual.bytes", record["visual"]["type"])
    extension["audio"] = ([embedded_media(tensors, voice_path, "maestro.audio.0.bytes", "audio", role="voice")]
                          if voice_path else [])
    extension["visual_is_preview"] = bool((record.get("refmod") or {}).get("preview_generated"))
    from .character_views import embed_views
    embed_views(directory, _get_character_record(character_id).get("image_views"), tensors, extension)
    metadata[CHARACTER_META] = json.dumps(extension, ensure_ascii=False)
    output = directory / character_filename(record["name"])
    write_refmod(output, tensors, metadata)
    # Persist the reusable visual encode alongside the library card.
    if not reference_path.is_file():
        write_refmod(reference_path, {"latent": tensors["latent"]}, {REFMOD_META: metadata[REFMOD_META]})
    info = inspect_refmod(reference_path)
    _update_character(character_id, refmod={**(record.get("refmod") or {}), "storage_name": reference_path.name,
        "kind": info["refmod"]["kind"], "tokens": info["tokens"], "mode": info["refmod"].get("mode", "encode")})
    return output


def import_character(path: str | Path, *, name: str = "", decode_preview=None) -> dict:
    """Import standard RefMods or complete .maestro.safetensors characters."""
    from .refmod import REFMOD_META, load_refmod, write_refmod

    info, tensors = load_refmod(path)
    extension = info["character"] or {}
    clean_name = " ".join(str(name or extension.get("name") or info["refmod"].get("name") or Path(path).stem).split())[:120]
    if not clean_name:
        raise ValueError("Give this character a name.")
    character_id = uuid.uuid4().hex[:16]
    directory = _character_directory(character_id)
    directory.mkdir(parents=False, exist_ok=False)
    try:
        kind = info["refmod"]["kind"]
        visual = extension.get("visual")
        image_views = None
        if visual:
            visual_path = directory / ("visual" + visual["extension"])
            visual_path.write_bytes(tensors[visual["tensor"]].numpy().tobytes())
            if kind == "image":
                from PIL import Image
                with Image.open(visual_path) as image:
                    if image.width * image.height > 32_000_000:
                        raise ValueError("The embedded character image exceeds 32 megapixels.")
                    image.verify()
            elif _probe_duration(visual_path) is None:
                raise ValueError("The embedded character video could not be read.")
        else:
            if decode_preview is None:
                raise ValueError("This RefMod needs an H3 VAE preview before it can be added to characters.")
            decoded = decode_preview(tensors["latent"], directory, kind)
            if isinstance(decoded, dict):
                visual_path, image_views = decoded["path"], decoded["image_views"]
            else:
                visual_path = decoded
        if extension.get("image_views") is not None:
            from .character_views import restore_views
            image_views = restore_views(directory, extension["image_views"], tensors)
        voice_path = None
        original_voice_name = None
        audio = extension.get("audio", [])
        if audio:
            voice_path = directory / ("voice" + audio[0]["extension"])
            voice_path.write_bytes(tensors[audio[0]["tensor"]].numpy().tobytes())
            duration = _probe_duration(voice_path)
            if duration is None or duration < 2:
                raise ValueError("Embedded voice must contain at least two seconds of readable audio.")
            if voice_path.suffix.lower() in {".mp3", ".m4a", ".aac"}:
                original_voice_name = voice_path.name
                decoded_voice = directory / "voice.wav"
                _extract_voice(voice_path, decoded_voice)
                voice_path = decoded_voice
        metadata = dict(info["metadata"])
        metadata[REFMOD_META] = json.dumps(info["refmod"], ensure_ascii=False)
        # Preserve all extension tensors/metadata for lossless re-export.
        write_refmod(directory / "reference.safetensors", tensors, metadata)
        now = time.time()
        record = {"id": character_id, "name": clean_name, "created_at": now, "updated_at": now,
                  "description": str(extension.get("description") or info["refmod"].get("description") or "")[:4000],
                  "visual": {"type": kind, "storage_name": visual_path.name, "filename": visual_path.name,
                             "duration_seconds": _probe_duration(visual_path) if kind == "video" else None,
                             "has_audio": False},
                  "voice": ({"storage_name": voice_path.name, "filename": voice_path.name,
                             **({"original_storage_name": original_voice_name} if original_voice_name else {}),
                             "duration_seconds": _probe_duration(voice_path)} if voice_path else None),
                  "refmod": {"storage_name": "reference.safetensors", "kind": kind,
                             "tokens": info["tokens"], "mode": info["refmod"].get("mode", "encode"),
                             "preview_generated": not bool(visual) or extension.get("visual_is_preview") is True}}
        if image_views:
            record["image_views"] = image_views
        with _LOCK:
            index = _load_index()
            index["characters"].append(record)
            _save_index(index)
        return _public_record(record)
    except Exception:
        # The UUID directory was resolved and checked under the library above.
        if directory.parent == _root():
            shutil.rmtree(directory, ignore_errors=True)
        raise


@_file_operation
def attach_character_voice(character_id: str, voice_path: str) -> dict:
    source = _source_path(voice_path, _AUDIO_EXTENSIONS)
    get_character(character_id)
    duration = _probe_duration(source)
    if duration is None or duration < 2:
        raise ValueError("Choose at least two seconds of readable voice audio.")
    destination = _character_directory(character_id) / ("voice" + source.suffix.lower())
    if source != destination:
        shutil.copy2(source, destination)
    return _update_character(character_id, voice={"storage_name": destination.name,
        "filename": source.name, "duration_seconds": duration})


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
    if slot == "thumbnail":
        return _character_thumbnail(character_id)
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


def _character_thumbnail(character_id: str) -> Path | None:
    """Cache a small still for mobile browsers that do not preload video frames."""
    try:
        directory = _character_directory(character_id)
    except ValueError:
        return None
    # Share the character's lock with export/delete so a thumbnail cannot
    # recreate a removed directory or race a media update.
    with _FILE_LOCKS[hash(character_id) % len(_FILE_LOCKS)]:
        record = _get_character_record(character_id)
        if record.get("image_views"):
            from .character_views import view_path
            try:
                return view_path(directory, record["image_views"], record["image_views"]["cover_id"])
            except ValueError:
                pass  # Missing caches can be recovered; keep the old preview usable.
        source = get_character_media(character_id, "visual")
        if source is None:
            return None
        if source.suffix.lower() in _IMAGE_EXTENSIONS:
            return source
        thumbnail = directory / "thumbnail.jpg"
        if thumbnail.is_file() and thumbnail.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return thumbnail
        temporary = directory / f"thumbnail-{uuid.uuid4().hex}.jpg"
        try:
            with _THUMBNAIL_WORKERS:
                # Prefer a frame past a possible black lead-in; fall back to
                # the opening frame for unusually short imported previews.
                for seek in ("0.1", "0"):
                    subprocess.run([
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", seek, "-i", str(source), "-an", "-frames:v", "1",
                        "-vf", "scale=480:480:force_original_aspect_ratio=decrease",
                        "-q:v", "3", "-threads", "1", str(temporary),
                    ], capture_output=True, check=True, timeout=30)
                    if temporary.is_file() and temporary.stat().st_size > 0:
                        os.replace(temporary, thumbnail)
                        return thumbnail
        except (OSError, subprocess.SubprocessError):
            # A failed preview should not stop a character from being usable.
            return None
        finally:
            temporary.unlink(missing_ok=True)
    return None


@_file_operation
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
