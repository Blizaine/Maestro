"""On-demand, pinned learned predictors for experimental H3 Veda attention."""

from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache


VEDA_REPO = "DeepBeepMeep/MiniMax-H3"
VEDA_REVISION = "49fe56ae78d4f8a410274d1e5f8c4d2b6f6e1b33"
VEDA_PREDICTORS = {
    (False, False): ("minimax_h3_t2va_veda_predictor_bf16.safetensors", 551327960,
                     "88bab248be01b706e09923075578b14feb817baccf50cf0114f016f97800715d"),
    (False, True): ("minimax_h3_t2va_veda_predictor_int8_convrot.safetensors", 280726251,
                    "21600e9c9007409d55ca6e9f65e373d95f5113cbb3939560feaf1bef3c9b389c"),
    (True, False): ("minimax_h3_r2va_veda_predictor_bf16.safetensors", 551328056,
                    "f7d10bb88a0c3a0957c5c775ca9b442cce378652791f6240fde3d1fd571dc581"),
    (True, True): ("minimax_h3_r2va_veda_predictor_int8_convrot.safetensors", 280726347,
                   "222be84ea4717959f88e7222cdba7e3ce4530e881a9bed0a44cdf7299cc1f924"),
}
_download_lock = threading.Lock()


def predictor_spec(reference_mode: bool, convrot: bool) -> dict:
    filename, size, digest = VEDA_PREDICTORS[bool(reference_mode), bool(convrot)]
    return {
        "filename": filename, "size": size, "sha256": digest,
        "repo_id": VEDA_REPO, "revision": VEDA_REVISION,
        "label": "H3 Veda References predictor" if reference_mode else "H3 Veda Frames predictor",
        "url": f"https://huggingface.co/{VEDA_REPO}/resolve/{VEDA_REVISION}/minimax_h3/{filename}",
        "source_url": f"https://huggingface.co/{VEDA_REPO}/tree/{VEDA_REVISION}/minimax_h3",
        "license": "MiniMax-H3 model license; see upstream model repository",
    }


@lru_cache(maxsize=16)
def _verified_digest(path: str, size: int, modified: int, changed: int) -> str:
    # The stat fields invalidate the cache when an installed file changes.
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_predictor(path: str, spec: dict) -> None:
    stat = os.stat(path)
    if stat.st_size != spec["size"] or _verified_digest(
        os.path.abspath(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    ) != spec["sha256"]:
        raise ValueError(f"H3 Veda predictor failed size/SHA-256 verification: {path}. Re-download the pinned predictor.")


def ensure_predictor(reference_mode: bool, convrot: bool, progress=None) -> str:
    """Reuse verified linked assets; download only into the primary model root."""
    from shared.utils import files_locator as fl
    from services.managed_preprocessors import _download_with_resume

    spec = predictor_spec(reference_mode, convrot)
    relative = os.path.join("minimax_h3", spec["filename"])
    with _download_lock:
        for root in fl.get_checkpoints_paths():
            for subdir in ("minimax_h3", os.path.join("minimax_h3", "minimax_h3")):
                candidate = os.path.abspath(os.path.join(root, subdir, spec["filename"]))
                if not os.path.isfile(candidate):
                    continue
                try:
                    verify_predictor(candidate, spec)
                except (OSError, ValueError) as error:
                    print(f"[MiniMax H3 Veda] Ignoring invalid cached predictor: {error}")
                    continue
                return candidate
        save_path = os.path.abspath(fl.get_download_location(relative))
        result = _download_with_resume(spec, save_path, progress)
        verify_predictor(result, spec)
        return result


def geometry_payload(layout, latent_shape, patch_size, anchors=(), references=()) -> tuple[tuple, dict]:
    """Translate Maestro's packed keyframe/reference geometry into Veda's contract."""
    try:
        patch_size = tuple(int(value) for value in patch_size)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("H3 Veda expects a three-dimensional (t, h, w) patch size.") from error
    if len(patch_size) != 3:
        raise ValueError("H3 Veda expects a three-dimensional (t, h, w) patch size.")
    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError("H3 Veda does not support temporal patching; transformer patch_t must be 1.")
    if (patch_h, patch_w) != (2, 2):
        raise ValueError("H3 Veda currently supports only the native 2x2 spatial patch size.")

    try:
        latent_shape = tuple(int(value) for value in latent_shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("H3 Veda latent geometry must be a three-dimensional (t, h, w) shape.") from error
    if len(latent_shape) != 3:
        raise ValueError("H3 Veda latent geometry must be a three-dimensional (t, h, w) shape.")
    frames, height, width = latent_shape
    if min(frames, height, width) <= 0:
        raise ValueError(f"H3 Veda latent dimensions must be positive, got {latent_shape}.")
    if height % patch_h or width % patch_w:
        raise ValueError(f"H3 Veda latent geometry {latent_shape} is not divisible by patch size {patch_size}.")

    grid = (frames // patch_t, height // patch_h, width // patch_w)
    keyframes = []
    for anchor in anchors:
        try:
            frame_count = int(anchor[1])
        except (IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("H3 Veda keyframe anchors must include a latent frame count.") from error
        if frame_count <= 0:
            raise ValueError(f"H3 Veda keyframe frame counts must be positive, got {frame_count}.")
        keyframes.append({"latent_frame_count": frame_count})

    refs = []
    for ref in references:
        if ref.kind == "audio":
            continue
        ref_shape = (int(ref.num_latent_frames), int(ref.latent_height), int(ref.latent_width))
        ref_t, ref_h, ref_w = ref_shape
        if min(ref_shape) <= 0:
            raise ValueError(f"H3 Veda reference dimensions must be positive, got {ref_shape}.")
        if ref_h % patch_h or ref_w % patch_w:
            raise ValueError(f"H3 Veda reference geometry {ref_shape} is not divisible by patch size {patch_size}.")
        refs.append({"kind": ref.kind, "latent_t": ref_t, "latent_h": ref_h, "latent_w": ref_w})

    condition_rows = sum(item["latent_frame_count"] * grid[1] * grid[2] for item in keyframes)
    condition_rows += sum(item["latent_t"] * (item["latent_h"] // patch_h) * (item["latent_w"] // patch_w) for item in refs)
    if condition_rows != layout.num_condition_video_rows:
        raise ValueError("H3 Veda reference geometry does not match the packed conditioning rows.")
    return grid, {"layout_signature": (id(layout), grid, condition_rows),
                  "keyframes": keyframes, "refs": refs}
