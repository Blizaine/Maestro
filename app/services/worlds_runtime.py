"""Small runtime boundary for native Worlds jobs and their saved state.

This module imports no model, Torch, or memory backend during UI/API startup.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

MAX_WORLD_BYTES = 3 << 30


def world_runtime_context(model_def, custom_settings):
    """Validate saved request options before any world model asset is loaded."""
    if not model_def.get("world_model"):
        return {}
    settings = custom_settings if isinstance(custom_settings, dict) else {}
    if settings.get("acknowledge_pixelumm_noncommercial") is not True:
        raise ValueError("Worlds requires acknowledgement of PixelUMM's non-commercial research/evaluation terms.")
    if settings.get("world_output") != "world_video":
        raise ValueError("Worlds jobs must save both their video and world state.")
    return {"acknowledge_pixelumm_noncommercial": True}


def world_generation_context(model_def, custom_settings):
    if not model_def.get("world_model"):
        return {}
    world_runtime_context(model_def, custom_settings)
    return {"world_output": "world_video"}


def persist_world_sidecar(media_path, side_files, model_def):
    """Atomically save the sole supported companion after primary media exists.

    Names cannot come from a model or client. The destination always shares
    the completed video's basename; storage validates its contents before
    publishing it in the Worlds library.
    """
    if not side_files:
        if model_def.get("world_model"):
            raise ValueError("World generation did not return its saved state.")
        return None
    if not model_def.get("world_model") or set(side_files) != {".wgpworld"}:
        raise ValueError("Unexpected model sidecar files.")
    data = side_files[".wgpworld"]
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_WORLD_BYTES:
        raise ValueError("World sidecar is empty, invalid, or exceeds its size bound.")
    media = Path(media_path)
    if media.suffix.lower() != ".mp4" or not media.is_file():
        raise ValueError("World sidecar requires its completed MP4 video.")
    destination = media.with_suffix(".wgpworld")
    # A reused filename must never silently replace an earlier saved world.
    if destination.exists():
        if destination.read_bytes() == data:
            return str(destination)
        raise FileExistsError("A different saved world already uses this output name.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=media.parent, prefix=".world-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Generation output names are exclusive to this worker. os.replace
        # keeps an interrupted write from exposing a partial ZIP.
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(destination)
