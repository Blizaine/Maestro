"""H3 RefMod files, with optional portable Maestro character media.

The upstream ``latent`` and ``refmod_meta`` contract stays at version 2.
Maestro owns a separate, optional namespace; see docs/Maestro-Characters.md.
Header inspection deliberately does not import Torch or load tensor payloads.
"""
from __future__ import annotations

import json
import os
import re
import struct
import unicodedata
import uuid
from functools import lru_cache
from pathlib import Path

REFMOD_META = "refmod_meta"
CHARACTER_META = "maestro_character_meta"
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_HEADER_BYTES = 2 * 1024 * 1024
MAX_REFERENCE_TOKENS = 32768
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def read_header(path: str | Path) -> dict:
    with open(path, "rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("This is not a complete safetensors file.")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= MAX_HEADER_BYTES:
            raise ValueError("Invalid or oversized safetensors header.")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("Incomplete safetensors header.")
    try:
        header = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid safetensors header.") from error
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header.")
    return header


def _json_object(value, label: str) -> dict:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label} metadata.") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid {label} metadata.")
    return parsed


def inspect_refmod(path: str | Path) -> dict:
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Character files must be smaller than 256 MB.")
    header = read_header(path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid safetensors metadata.")
    if REFMOD_META in metadata:
        ref = _json_object(metadata[REFMOD_META], "RefMod")
    else:
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file() or sidecar.stat().st_size > MAX_HEADER_BYTES:
            raise ValueError("This file is not an H3 RefMod. Use the LoRA or checkpoint browser for model weights.")
        ref = _json_object(sidecar.read_text(encoding="utf-8"), "RefMod")
    tensor = header.get("latent", {})
    shape = tensor.get("shape") if isinstance(tensor, dict) else None
    if (not isinstance(shape, list) or len(shape) != 5
            or any(type(n) is not int or n <= 0 for n in shape)
            or shape[:2] != [1, 24] or shape[2] > 128
            or shape[3] % 2 or shape[4] % 2
            or shape[3] > 256 or shape[4] > 256
            or tensor.get("dtype") not in {"F16", "BF16", "F32"}):
        raise ValueError("Expected an H3 visual latent [1, 24, frames, height, width] with even spatial dimensions.")
    tokens = shape[2] * (shape[3] // 2) * (shape[4] // 2)
    if tokens > MAX_REFERENCE_TOKENS:
        raise ValueError(f"RefMod exceeds the {MAX_REFERENCE_TOKENS:,} reference-token limit.")
    kind = ref.get("kind")
    if kind not in {"image", "video"} or (kind == "image" and shape[2] != 1):
        raise ValueError("RefMod kind and visual tensor do not match.")
    for key, dim in (("latent_t", 2), ("latent_h", 3), ("latent_w", 4)):
        if key in ref and ref[key] != shape[dim]:
            raise ValueError(f"RefMod {key} does not match its tensor.")
    extension = _json_object(metadata[CHARACTER_META], "Maestro character") if CHARACTER_META in metadata else None
    if extension is not None:
        if extension.get("schema_version") != 1:
            raise ValueError("This Maestro character version is not supported. Update Maestro before importing it.")
        visual = extension.get("visual")
        if visual is not None:
            _validate_media_entry(header, visual, "visual")
            if visual.get("type") != kind:
                raise ValueError("Embedded visual media does not match the RefMod kind.")
        audio = extension.get("audio", [])
        if not isinstance(audio, list) or len(audio) > 1:
            raise ValueError("A Maestro character can contain one saved voice recording.")
        for entry in audio:
            _validate_media_entry(header, entry, "audio")
            if entry.get("role") != "voice":
                raise ValueError("Character audio must be a voice reference.")
        if extension.get("image_views") is not None:
            from .character_views import validate_embedded_views
            validate_embedded_views(extension["image_views"])
            for entry in extension["image_views"]["items"]:
                _validate_media_entry(header, entry, "visual")
    return {"refmod": ref, "character": extension, "shape": shape,
            "tokens": tokens, "metadata": metadata, "header": header}


def _validate_media_entry(header: dict, entry, slot: str) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid embedded {slot}.")
    key = entry.get("tensor", "")
    if not isinstance(key, str) or not key.startswith("maestro."):
        raise ValueError(f"Invalid embedded {slot} tensor name.")
    tensor = header.get(key, {})
    shape = tensor.get("shape", []) if isinstance(tensor, dict) else []
    if (not isinstance(tensor, dict) or not isinstance(shape, list)
            or tensor.get("dtype") != "U8" or len(shape) != 1
            or type(shape[0]) is not int or not 0 < shape[0] <= MAX_FILE_BYTES):
        raise ValueError(f"Invalid embedded {slot} data.")
    kind = entry.get("type")
    allowed = AUDIO_EXTENSIONS if slot == "audio" else (
        IMAGE_EXTENSIONS if kind == "image" else VIDEO_EXTENSIONS if kind == "video" else set())
    if (slot == "audio" and kind != "audio") or entry.get("encoding") != "file" or entry.get("extension") not in allowed:
        raise ValueError(f"Unsupported embedded {slot} format.")


@lru_cache(maxsize=2048)
def _is_refmod(path: str, size: int, modified: int, sidecar_modified: int) -> bool:
    try:
        header = read_header(path)
        meta = header.get("__metadata__", {})
        # Malformed character files must also stay out of the LoRA picker.
        return (isinstance(meta, dict) and (REFMOD_META in meta or CHARACTER_META in meta)) or (
            "latent" in header and Path(path).with_suffix(".json").is_file())
    except (OSError, ValueError):
        return False


def is_refmod_file(path: str | Path) -> bool:
    try:
        stat = Path(path).stat()
        sidecar = Path(path).with_suffix(".json")
        return _is_refmod(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns,
                          sidecar.stat().st_mtime_ns if sidecar.is_file() else 0)
    except OSError:
        return False


def load_refmod(path: str | Path) -> tuple[dict, dict]:
    import torch
    from safetensors.torch import load_file

    info = inspect_refmod(path)
    try:
        tensors = load_file(str(path), device="cpu")
    except Exception as error:
        raise ValueError("The character file has invalid or incomplete tensor data.") from error
    if not torch.isfinite(tensors["latent"]).all():
        raise ValueError("RefMod visual latents contain invalid values.")
    # Detach mappings so character files can be replaced/deleted on Windows.
    return info, {key: value.clone() for key, value in tensors.items()}


def load_visual_latent(path: str | Path):
    import torch
    from safetensors import safe_open

    info = inspect_refmod(path)
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        latent = reader.get_tensor("latent").clone()
    if not torch.isfinite(latent).all():
        raise ValueError("RefMod visual latents contain invalid values.")
    return info, latent


def character_filename(name: str) -> str:
    stem = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:80] or "character"
    if stem in {"con", "prn", "aux", "nul", *(f"com{n}" for n in range(1, 10)), *(f"lpt{n}" for n in range(1, 10))}:
        stem = "character_" + stem
    return f"{stem}.maestro.safetensors"


def embedded_media(tensors: dict, path: Path, key: str, kind: str, **extra) -> dict:
    import torch

    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Character reference media is missing or larger than 256 MB.")
    tensors[key] = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
    return {"tensor": key, "encoding": "file", "extension": path.suffix.lower(),
            "type": kind, **extra}


def write_refmod(path: str | Path, tensors: dict, metadata: dict) -> Path:
    from safetensors.torch import save_file

    path = Path(path)
    if sum(t.numel() * t.element_size() for t in tensors.values()) > MAX_FILE_BYTES - MAX_HEADER_BYTES:
        raise ValueError("The complete character is larger than 256 MB. Use a shorter reference clip.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.part")
    try:
        save_file({key: tensor.contiguous().cpu() for key, tensor in tensors.items()},
                  str(temporary), metadata=metadata)
        inspect_refmod(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def new_refmod_metadata(name: str, latent, *, source: str, description: str = "") -> dict:
    t, h, w = (int(n) for n in latent.shape[2:])
    ref = {"_format_version": 2, "name": name, "kind": "image" if t == 1 else "video",
           "latent_t": t, "latent_h": h, "latent_w": w, "mode": "encode", "source": source,
           "source_shape": f"{t}x{h}x{w}", "pool": f"full-res {w*16}x{h*16}px",
           "optimize_steps": 0, "tags": [], "description": description, "concept_type": "identity"}
    return {REFMOD_META: json.dumps(ref, ensure_ascii=False)}
