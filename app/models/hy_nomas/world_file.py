# Adapted from Hy-Nomas (https://github.com/Tencent-Hunyuan/Hy-Nomas, models/hy_nomas/world_file.py), Copyright (C) 2026 Tencent, Apache-2.0.
# Modified for Maestro. Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt).
"""Bounded, validated reader/writer for versioned Hy-Nomas ``.wgpworld`` files.

The v1 container is an uncompressed ZIP with exactly four entries: a JSON
manifest, a safetensors payload, a small JPEG thumbnail, and a full-size JPEG
poster.  History KV is rebuilt from the saved latent/text/camera data; it is
not serialized.  All limits are checked before reading payloads into memory.
"""

from __future__ import annotations

import io
import json
import math
import os
import stat
import uuid
import zipfile
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from safetensors.torch import load, save

FORMAT, VERSION, EXTENSION = "wgpworld", 1, ".wgpworld"

# Worlds are uncompressed by contract. These bounds allow typical 60 second
# worlds with room for saved exploration, while rejecting hostile/accidental
# multi-gigabyte inputs before allocating tensor storage.
MAX_WORLD_BYTES = 3 << 30
MAX_TENSORS_BYTES = 2 << 30
MAX_MANIFEST_BYTES = 8 << 20
MAX_TENSOR_HEADER_BYTES = 16 << 20
MAX_THUMBNAIL_BYTES = 8 << 20
MAX_POSTER_BYTES = 32 << 20
MAX_IMAGE_DIMENSION = 8192
MAX_LATENT_FRAMES = 10000
MAX_TEXT_TOKENS = 1280
LATENT_CHANNELS = 32
TEXT_DIMENSION = 3584
_ENTRY_LIMITS = {
    "manifest.json": MAX_MANIFEST_BYTES,
    "tensors.safetensors": MAX_TENSORS_BYTES,
    "thumbnail.jpg": MAX_THUMBNAIL_BYTES,
    "poster.jpg": MAX_POSTER_BYTES,
}


def _positive_int(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"World {name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise ValueError(f"World {name} exceeds the supported limit of {maximum}.")
    return value


def _tensor_is_finite(tensor, chunk_elements=1 << 20):
    flat = tensor.reshape(-1)
    return all(torch.isfinite(flat[start:start + chunk_elements]).all().item() for start in range(0, flat.numel(), chunk_elements))


def _safe_json(data):
    def reject_constant(value):
        raise ValueError(f"World manifest contains invalid numeric constant {value}.")

    try:
        manifest = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("World manifest is not valid UTF-8 JSON.") from error
    if not isinstance(manifest, dict):
        raise ValueError("World manifest must be a JSON object.")
    return manifest


def _validate_manifest(manifest, tensors=None):
    """Validate v1 metadata and, when supplied, its tensor schema."""
    if not isinstance(manifest, dict):
        raise ValueError("World manifest must be a dictionary.")
    version = manifest.get("version")
    if manifest.get("format") != FORMAT or isinstance(version, bool) or not isinstance(version, int) or version != VERSION:
        raise ValueError(f"World file format must be {FORMAT!r} version {VERSION}.")

    required = ("model_type", "width", "height", "fps", "steps", "world", "generations", "segments")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"World manifest is missing required fields: {', '.join(missing)}.")
    if manifest["model_type"] != "hy_nomas":
        raise ValueError("World file model_type must be 'hy_nomas'.")
    width = _positive_int(manifest["width"], "width", MAX_IMAGE_DIMENSION)
    height = _positive_int(manifest["height"], "height", MAX_IMAGE_DIMENSION)
    if width % 32 or height % 32:
        raise ValueError("World dimensions must be multiples of 32 pixels.")
    if isinstance(manifest["fps"], bool) or not isinstance(manifest["fps"], (int, float)) or not math.isfinite(float(manifest["fps"])) or manifest["fps"] != 24:
        raise ValueError("Hy-Nomas world frame rate must be 24 fps.")
    if manifest["steps"] not in (4, 6, 8):
        raise ValueError("World denoising steps must be 4, 6, or 8.")
    if "history_frames" in manifest:
        _positive_int(manifest["history_frames"], "history_frames", 10000)
    if "world_id" in manifest and (not isinstance(manifest["world_id"], str) or not manifest["world_id"] or len(manifest["world_id"]) > 128):
        raise ValueError("World world_id must be a non-empty string of at most 128 characters.")

    world = manifest["world"]
    generations = manifest["generations"]
    segments = manifest["segments"]
    if not isinstance(world, dict) or not isinstance(world.get("segments"), list):
        raise ValueError("World description must contain a segments list.")
    if not isinstance(generations, list) or not generations or len(generations) > 4096:
        raise ValueError("World generations must be a non-empty list of at most 4096 entries.")
    if not isinstance(segments, list) or not segments or len(segments) > 10000:
        raise ValueError("World segments must be a non-empty list of at most 10000 entries.")
    if len(world["segments"]) != len(segments):
        raise ValueError("World description segments do not match the stored segment timeline.")

    generation_frames = [0] * len(generations)
    for generation in generations:
        if not isinstance(generation, dict):
            raise ValueError("Each world generation must be an object.")
        _positive_int(generation.get("latent_frames"), "generation latent_frames", MAX_LATENT_FRAMES)
        seed = generation.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 63):
            raise ValueError("World generation seeds must be integers in [0, 2^63).")

    expected_start = 0
    total_frames = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"World segment {index} must be an object.")
        start = segment.get("latent_start")
        frames = _positive_int(segment.get("latent_frames"), f"segment {index} latent_frames", MAX_LATENT_FRAMES)
        _positive_int(segment.get("video_frames"), f"segment {index} video_frames", MAX_LATENT_FRAMES * 4 + 1)
        generation = segment.get("generation")
        if isinstance(start, bool) or not isinstance(start, int) or start != expected_start:
            raise ValueError("World segments must form one contiguous latent timeline starting at frame zero.")
        if isinstance(generation, bool) or not isinstance(generation, int) or not 0 <= generation < len(generations):
            raise ValueError(f"World segment {index} references an invalid generation.")
        if not isinstance(segment.get("camera"), bool):
            raise ValueError(f"World segment {index} camera flag must be boolean.")

        offsets = segment.get("text_offsets")
        if offsets is not None:
            if not isinstance(offsets, list) or not offsets or len(offsets) > MAX_TEXT_TOKENS + 1:
                raise ValueError(f"World segment {index} text_offsets must be null or a bounded list.")
            previous = 0
            for offset in offsets:
                if isinstance(offset, bool) or not isinstance(offset, int) or offset < previous or offset > MAX_TEXT_TOKENS:
                    raise ValueError(f"World segment {index} text_offsets must be monotonic token indexes.")
                previous = offset

        windows = segment.get("action_windows")
        if windows is not None:
            if offsets is None:
                raise ValueError(f"World segment {index} cannot have action windows without text offsets.")
            if not isinstance(windows, list) or len(windows) > 4096:
                raise ValueError(f"World segment {index} action_windows must be null or a bounded list.")
            for window in windows:
                if not isinstance(window, (list, tuple)) or len(window) != 3:
                    raise ValueError(f"World segment {index} has a malformed action window.")
                first, last, action = window
                if (isinstance(first, bool) or not isinstance(first, int) or isinstance(last, bool) or not isinstance(last, int)
                        or not 0 <= first < last <= frames or isinstance(action, bool) or not isinstance(action, int)
                        or action < -1 or action >= len(offsets) - 1):
                    raise ValueError(f"World segment {index} has an out-of-range action window.")

        expected_start += frames
        total_frames += frames
        generation_frames[generation] += frames
    if total_frames > MAX_LATENT_FRAMES:
        raise ValueError(f"World latent timeline exceeds {MAX_LATENT_FRAMES} frames.")
    if generation_frames != [item["latent_frames"] for item in generations]:
        raise ValueError("World generation lengths do not match the segment timeline.")
    if len(world["segments"]) != len(segments):
        raise ValueError("World description segment count is inconsistent.")

    if tensors is None:
        return total_frames
    if not isinstance(tensors, dict):
        raise ValueError("World safetensors payload must be a tensor dictionary.")
    required_names = {"latents", *(f"text_{index}" for index in range(len(segments)))}
    if "camera" in tensors:
        required_names.add("camera")
    if ("camera" in tensors) != any(segment["camera"] for segment in segments):
        raise ValueError("World camera tensor presence must match its segment camera flags.")
    if set(tensors) != required_names:
        raise ValueError("World tensor keys do not match the manifest segments and optional camera tensor.")

    latent = tensors["latents"]
    if not torch.is_tensor(latent) or latent.dtype != torch.bfloat16 or tuple(latent.shape) != (1, LATENT_CHANNELS, total_frames, height // 32, width // 32):
        raise ValueError("World latents must be BF16 with shape [1, 32, total_frames, height/32, width/32].")
    if not _tensor_is_finite(latent):
        raise ValueError("World latents must contain only finite values.")
    for index in range(len(segments)):
        state = tensors[f"text_{index}"]
        if (not torch.is_tensor(state) or state.dtype != torch.bfloat16 or state.ndim != 2 or state.shape[1] != TEXT_DIMENSION
                or not 1 <= state.shape[0] <= MAX_TEXT_TOKENS):
            raise ValueError(f"World text_{index} must be BF16 [tokens, {TEXT_DIMENSION}] with 1 to {MAX_TEXT_TOKENS} tokens.")
        if not _tensor_is_finite(state):
            raise ValueError(f"World text_{index} must contain only finite values.")
        offsets = segments[index].get("text_offsets")
        if offsets is not None and offsets[-1] != state.shape[0]:
            raise ValueError(f"World segment {index} text_offsets do not end at the stored text token count.")
    if "camera" in tensors:
        camera = tensors["camera"]
        if (not torch.is_tensor(camera) or camera.dtype != torch.float32 or tuple(camera.shape) != (4 * total_frames, 4, 4)
                or not _tensor_is_finite(camera)):
            raise ValueError("World camera must be finite FP32 poses with shape [4 * total_frames, 4, 4].")
    if sum(t.numel() * t.element_size() for t in tensors.values()) > MAX_TENSORS_BYTES:
        raise ValueError("World tensors exceed the uncompressed payload limit.")
    return total_frames


def _validate_jpeg(data, name, maximum_bytes, max_dimensions):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"World {name} must be JPEG bytes.")
    data = bytes(data)
    if not data or len(data) > maximum_bytes:
        raise ValueError(f"World {name} is empty or exceeds its size limit.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or image.width <= 0 or image.height <= 0:
                raise ValueError(f"World {name} must be a valid JPEG image.")
            if image.width > max_dimensions[0] or image.height > max_dimensions[1] or image.width * image.height > 64_000_000:
                raise ValueError(f"World {name} dimensions exceed the supported limit.")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"World {name} must be a valid JPEG image.") from error
    return data


def _tensor_header(tensor_bytes):
    if len(tensor_bytes) < 8:
        raise ValueError("World safetensors payload is truncated.")
    header_size = int.from_bytes(tensor_bytes[:8], "little", signed=False)
    if header_size <= 0 or header_size > MAX_TENSOR_HEADER_BYTES or 8 + header_size > len(tensor_bytes):
        raise ValueError("World safetensors header is invalid or exceeds its size limit.")
    header = _safe_json(tensor_bytes[8:8 + header_size])
    if "__metadata__" in header and not isinstance(header["__metadata__"], dict):
        raise ValueError("World safetensors metadata must be an object.")
    return header, 8 + header_size, len(tensor_bytes) - 8 - header_size


def _validate_tensor_header(tensor_bytes, manifest):
    header, data_start, data_size = _tensor_header(tensor_bytes)
    expected = {"latents", *(f"text_{index}" for index in range(len(manifest["segments"]))) }
    if "camera" in header:
        expected.add("camera")
    names = set(header) - {"__metadata__"}
    if names != expected:
        raise ValueError("World safetensors header keys do not match the manifest.")
    if ("camera" in names) != any(segment["camera"] for segment in manifest["segments"]):
        raise ValueError("World camera tensor presence must match its segment camera flags.")
    intervals = []
    for name in names:
        item = header[name]
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"World tensor header for {name!r} is malformed.")
        shape, offsets = item["shape"], item["data_offsets"]
        if (not isinstance(shape, list) or not shape or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2 or any(isinstance(n, bool) or not isinstance(n, int) for n in offsets)):
            raise ValueError(f"World tensor header for {name!r} has invalid shape or data offsets.")
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            raise ValueError(f"World tensor header for {name!r} points outside the data payload.")
        dtype = item["dtype"]
        itemsize = {"BF16": 2, "F32": 4}.get(dtype)
        if itemsize is None or math.prod(shape) * itemsize != end - start:
            raise ValueError(f"World tensor header for {name!r} has an unsupported dtype or byte count.")
        if name == "latents":
            expected_shape, expected_dtype = [1, 32, sum(s["latent_frames"] for s in manifest["segments"]), manifest["height"] // 32, manifest["width"] // 32], "BF16"
        elif name.startswith("text_"):
            index = int(name[5:])
            expected_shape, expected_dtype = None, "BF16"
            if len(shape) != 2 or shape[1] != TEXT_DIMENSION or not 1 <= shape[0] <= MAX_TEXT_TOKENS:
                raise ValueError(f"World tensor header for {name!r} has an invalid token shape.")
            offsets_in_manifest = manifest["segments"][index].get("text_offsets")
            if offsets_in_manifest is not None and offsets_in_manifest[-1] != shape[0]:
                raise ValueError(f"World tensor header for {name!r} disagrees with the text offsets.")
        else:
            expected_shape, expected_dtype = [4 * sum(s["latent_frames"] for s in manifest["segments"]), 4, 4], "F32"
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(f"World tensor header for {name!r} has an invalid shape.")
        if dtype != expected_dtype:
            raise ValueError(f"World tensor header for {name!r} has an invalid dtype.")
        intervals.append((start, end))
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if start != cursor:
            raise ValueError("World safetensors data ranges must be contiguous and non-overlapping.")
        cursor = end
    if cursor != data_size:
        raise ValueError("World safetensors payload contains unreferenced bytes.")


def _open_source(source):
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        if not data or len(data) > MAX_WORLD_BYTES:
            raise ValueError("World file is empty or exceeds the total size limit.")
        return io.BytesIO(data), None
    try:
        path = Path(os.fspath(source))
    except TypeError as error:
        raise ValueError("World source must be a .wgpworld path or bytes.") from error
    if path.suffix.lower() != EXTENSION:
        raise ValueError(f"World path must end in {EXTENSION}.")
    if path.is_symlink():
        raise ValueError("World path must not be a symbolic link.")
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError as error:
        raise ValueError("World file path does not identify a readable file.") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("World path must identify a regular file.")
    if info.st_size <= 0 or info.st_size > MAX_WORLD_BYTES:
        raise ValueError("World file is empty or exceeds the total size limit.")
    return resolved, info.st_size


def _zip_infos(archive):
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(infos) != len(_ENTRY_LIMITS) or len(set(names)) != len(names) or set(names) != set(_ENTRY_LIMITS):
        raise ValueError("World ZIP must contain each v1 entry exactly once and no other entries.")
    total = 0
    by_name = {}
    for info in infos:
        name = info.filename
        if ("\\" in name or "\x00" in name or name.startswith("/") or ".." in Path(name).parts or info.is_dir()
                or info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED):
            raise ValueError("World ZIP contains an unsafe, encrypted, or compressed entry.")
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise ValueError("World ZIP must not contain symbolic links.")
        if info.file_size < 0 or info.file_size > _ENTRY_LIMITS[name]:
            raise ValueError(f"World ZIP entry {name!r} exceeds its size limit.")
        total += info.file_size
        by_name[name] = info
    if total > MAX_WORLD_BYTES:
        raise ValueError("World ZIP expands beyond the total size limit.")
    return by_name


def _read_manifest(archive, infos):
    try:
        data = archive.read(infos["manifest.json"])
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError("World manifest could not be read from the ZIP.") from error
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("World manifest exceeds its size limit.")
    manifest = _safe_json(data)
    _validate_manifest(manifest)
    return manifest


def build(manifest, latents, text_states, camera, thumbnail_jpeg, poster_jpeg):
    """Build a validated v1 world archive and return its bytes."""
    if not isinstance(manifest, dict):
        raise ValueError("World manifest must be a dictionary.")
    # Serialize fixed format fields after caller data so they cannot be spoofed.
    payload_manifest = {**manifest, "format": FORMAT, "version": VERSION}
    payload_manifest.setdefault("history_frames", 18)
    payload_manifest.setdefault("world_id", uuid.uuid4().hex)
    if not torch.is_tensor(latents) or latents.ndim != 5:
        raise ValueError("World latents must be a five-dimensional tensor.")
    tensors = {"latents": latents.to(device="cpu", dtype=torch.bfloat16).contiguous()}
    if not isinstance(text_states, (list, tuple)):
        raise ValueError("World text states must be a list of tensors.")
    for index, states in enumerate(text_states):
        if not torch.is_tensor(states) or states.ndim != 2:
            raise ValueError(f"World text state {index} must be a two-dimensional tensor.")
        tensors[f"text_{index}"] = states.to(device="cpu", dtype=torch.bfloat16).contiguous()
    if camera is not None:
        if not torch.is_tensor(camera):
            raise ValueError("World camera poses must be a tensor.")
        tensors["camera"] = camera.to(device="cpu", dtype=torch.float32).contiguous()
    _validate_manifest(payload_manifest, tensors)
    thumbnail_jpeg = _validate_jpeg(thumbnail_jpeg, "thumbnail", MAX_THUMBNAIL_BYTES, (320, 320))
    poster_jpeg = _validate_jpeg(poster_jpeg, "poster", MAX_POSTER_BYTES, (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

    try:
        manifest_bytes = json.dumps(payload_manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("World manifest contains unsupported values.") from error
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("World manifest exceeds its size limit.")
    tensor_bytes = save(tensors)
    if len(tensor_bytes) > MAX_TENSORS_BYTES:
        raise ValueError("World tensors exceed the uncompressed payload limit.")
    _validate_tensor_header(tensor_bytes, payload_manifest)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("tensors.safetensors", tensor_bytes)
        archive.writestr("thumbnail.jpg", thumbnail_jpeg)
        archive.writestr("poster.jpg", poster_jpeg)
    result = buffer.getvalue()
    if len(result) > MAX_WORLD_BYTES:
        raise ValueError("World archive exceeds the total size limit.")
    return result


def _read(source, *, load_tensors):
    opened, file_size = _open_source(source)
    try:
        with zipfile.ZipFile(opened, "r") as archive:
            infos = _zip_infos(archive)
            manifest = _read_manifest(archive, infos)
            # Early v1 worlds predate these optional session metadata fields.
            # Keep them readable while supplying the values the current
            # runtime expects after the file has passed structural checks.
            manifest.setdefault("history_frames", 18)
            manifest.setdefault("world_id", uuid.uuid4().hex)
            thumbnail = archive.read(infos["thumbnail.jpg"])
            _validate_jpeg(thumbnail, "thumbnail", MAX_THUMBNAIL_BYTES, (320, 320))
            poster = archive.read(infos["poster.jpg"])
            _validate_jpeg(poster, "poster", MAX_POSTER_BYTES, (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
            if not load_tensors:
                return manifest, None, thumbnail
            tensor_bytes = archive.read(infos["tensors.safetensors"])
            _validate_tensor_header(tensor_bytes, manifest)
            try:
                tensors = load(tensor_bytes)
            except (ValueError, RuntimeError, KeyError, TypeError) as error:
                raise ValueError("World safetensors payload is invalid.") from error
            _validate_manifest(manifest, tensors)
            return manifest, tensors, thumbnail
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError, RuntimeError) as error:
        raise ValueError("World file is not a valid v1 ZIP archive.") from error
    finally:
        if file_size is None:
            opened.close()


def read(path):
    """Return ``(manifest, tensors)`` after validating a complete world file."""
    manifest, tensors, _thumbnail = _read(path, load_tensors=True)
    return manifest, tensors


def read_thumbnail(path):
    """Read a validated thumbnail without loading its potentially large tensor payload."""
    _manifest, _tensors, thumbnail = _read(path, load_tensors=False)
    return thumbnail
