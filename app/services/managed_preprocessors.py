"""On-demand assets for optional control-video preprocessors.

Large preprocessor checkpoints should not make every Maestro install several
gigabytes larger.  This module downloads them only when a workflow actually
selects the matching control type, while keeping linked checkpoint roots
read-only and publishing only fully verified files.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable
from typing import Any

import requests

from shared.utils import files_locator as fl


ProgressCallback = Callable[[str], Any]


# Video Depth Anything source:
# https://github.com/DepthAnything/Video-Depth-Anything
#
# The immutable revisions, byte sizes, and SHA-256 values below come from the
# official depth-anything Hugging Face repositories.  Base and Large weights
# are CC-BY-NC-4.0; Maestro does not redistribute them.
VIDEO_DEPTH_CHECKPOINTS = {
    "vitb": {
        "label": "Video Depth Anything Base",
        "repo_id": "depth-anything/Video-Depth-Anything-Base",
        "revision": "7231d0c6e260f54f103eba5005d01f6efa4f43db",
        "filename": "video_depth_anything_vitb.pth",
        "size": 458_247_082,
        "sha256": "775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b",
        "license": "CC-BY-NC-4.0",
    },
    "vitl": {
        "label": "Video Depth Anything Large",
        "repo_id": "depth-anything/Video-Depth-Anything-Large",
        "revision": "7aafbcb5c6af0bac741aad2b6471894fb4761afa",
        "filename": "video_depth_anything_vitl.pth",
        "size": 1_538_392_012,
        "sha256": "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa",
        "license": "CC-BY-NC-4.0",
    },
}


# WanGP's affine fit packages let one MiniMax H3 LoRA target both the Full
# 33B timestep MLP and the compressed AdaLN curve used by Pruned 20B.  They
# are tiny support tensors rather than generation weights, but pin and verify
# them exactly like a model so a changed upstream file cannot silently alter
# an adapter at load time.
_H3_AFFINE_MAP_REVISION = "1830091bf4b27df2f901920d55b1fb748f33e7eb"
_H3_AFFINE_MAP_BASE_URL = (
    "https://raw.githubusercontent.com/deepbeepmeep/Wan2GP/"
    f"{_H3_AFFINE_MAP_REVISION}/models/minimax_h3/lora_affine_maps"
)
MINIMAX_H3_LORA_AFFINE_MAPS = {
    "fl2va": {
        8: {
            "filename": "fl2va_rank8.sft",
            "size": 130_072,
            "sha256": "a42778e02ab2708dc70e23837ec4d3061b44f938c940decbc7a5b91f2c27c59e",
        },
        64: {
            "filename": "fl2va_rank64.sft",
            "size": 955_640,
            "sha256": "df40361cba88c9d6cf300a90d506ed349b349bd23babb6b94f15ab2df1b00f6e",
        },
    },
    "ref2va": {
        8: {
            "filename": "ref2va_rank8.sft",
            "size": 130_072,
            "sha256": "7179899e59fce9c36038cd6c0c57edaced0032c769c436cef234b07bf809381f",
        },
        64: {
            "filename": "ref2va_rank64.sft",
            "size": 955_640,
            "sha256": "4b661b03438d5d5fcc86be3dad2d9dbbd129720f089f8e94914b369eee198cee",
        },
    },
}
for _h3_architecture, _h3_width_specs in MINIMAX_H3_LORA_AFFINE_MAPS.items():
    for _h3_width, _h3_spec in _h3_width_specs.items():
        _h3_spec.update(
            {
                "label": (
                    f"MiniMax H3 {_h3_architecture.upper()} "
                    f"rank-{_h3_width} LoRA compatibility"
                ),
                "url": f"{_H3_AFFINE_MAP_BASE_URL}/{_h3_spec['filename']}",
                "source_url": (
                    "https://github.com/deepbeepmeep/Wan2GP/tree/"
                    f"{_H3_AFFINE_MAP_REVISION}/models/minimax_h3/"
                    "lora_affine_maps"
                ),
                "license": "WanGP Community License 2.0",
            }
        )


_video_depth_download_lock = threading.Lock()
_h3_affine_download_lock = threading.Lock()


def uses_temporal_depth(params: dict | None) -> bool:
    """Return whether generation parameters request temporal-depth control."""

    if not isinstance(params, dict):
        return False
    value = params.get("video_prompt_type")
    if isinstance(value, (list, tuple, set)):
        value = "".join(str(item or "") for item in value)
    value = str(value or "")
    return "T" in value or "depth_temporal" in value.casefold()


def _checkpoint_spec(variant: str | None) -> tuple[str, dict]:
    normalized = str(variant or "vitl").strip().casefold()
    if normalized not in VIDEO_DEPTH_CHECKPOINTS:
        supported = ", ".join(sorted(VIDEO_DEPTH_CHECKPOINTS))
        raise RuntimeError(
            f"Unsupported Video Depth Anything variant '{variant}'. "
            f"Supported variants: {supported}."
        )
    return normalized, VIDEO_DEPTH_CHECKPOINTS[normalized]


def _checkpoint_relative_path(spec: dict) -> str:
    return os.path.join("depth", str(spec["filename"]))


def _is_complete_checkpoint(path: str | None, spec: dict) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        return os.path.getsize(path) == int(spec["size"])
    except OSError:
        return False


def _sha256_file(path: str, progress: ProgressCallback | None = None, label: str = "") -> str:
    digest = hashlib.sha256()
    total = max(1, os.path.getsize(path))
    done = 0
    last_pct = -1
    with open(path, "rb") as source:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            if progress:
                pct = int(done * 100 / total)
                if pct >= last_pct + 10:
                    last_pct = pct
                    progress(f"Verifying {label} model… {pct}%")
    return digest.hexdigest()


def _download_with_resume(
    spec: dict,
    save_path: str,
    progress: ProgressCallback | None = None,
) -> str:
    """Download, verify, and atomically publish one official checkpoint."""

    label = str(spec["label"])
    expected_size = int(spec["size"])
    expected_sha256 = str(spec["sha256"]).casefold()
    filename = str(spec["filename"])
    url = spec.get("url") or (
        f"https://huggingface.co/{spec['repo_id']}/resolve/"
        f"{spec['revision']}/{filename}"
    )
    partial_path = save_path + ".part"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    current_size = 0
    try:
        current_size = os.path.getsize(partial_path)
    except OSError:
        pass
    if current_size > expected_size:
        os.remove(partial_path)
        current_size = 0

    headers = {"Range": f"bytes={current_size}-"} if current_size else {}
    if progress:
        initial_pct = int(current_size * 100 / expected_size)
        progress(
            f"Downloading {label} model (one-time setup)… {initial_pct}%"
        )
    print(
        f"[ManagedPreprocessor] {label} not found — downloading {url} "
        f"-> {save_path}"
    )

    try:
        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(15, 60),
            allow_redirects=True,
        )
        try:
            # A stale complete .part may receive 416. It still has to pass the
            # exact size and hash checks below before it is published.
            if not (
                response.status_code == 416
                and current_size == expected_size
            ):
                response.raise_for_status()
                can_resume = current_size > 0 and response.status_code == 206
                mode = "ab" if can_resume else "wb"
                if not can_resume:
                    current_size = 0

                done = current_size
                last_pct = int(done * 100 / expected_size) - 1
                with open(partial_path, mode) as output:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if not chunk:
                            continue
                        output.write(chunk)
                        done += len(chunk)
                        if progress:
                            pct = min(
                                100, int(done * 100 / expected_size)
                            )
                            if pct >= last_pct + 2:
                                last_pct = pct
                                progress(
                                    f"Downloading {label} model "
                                    f"(one-time setup)… {pct}%"
                                )
        finally:
            response.close()

        actual_size = os.path.getsize(partial_path)
        if actual_size != expected_size:
            raise RuntimeError(
                f"incomplete download (expected {expected_size:,} bytes, "
                f"got {actual_size:,})"
            )

        if progress:
            progress(f"Verifying {label} model…")
        actual_sha256 = _sha256_file(partial_path, progress, label)
        if actual_sha256.casefold() != expected_sha256:
            try:
                os.remove(partial_path)
            except OSError:
                pass
            raise RuntimeError(
                "SHA-256 mismatch; the partial file was removed and was not "
                "published"
            )

        os.replace(partial_path, save_path)
        print(f"[ManagedPreprocessor] {label} downloaded -> {save_path}")
        return save_path
    except Exception as exc:
        # Keep a correctly bounded partial so the next attempt can resume.
        # A too-large or hash-invalid partial is removed above.
        try:
            if (
                os.path.isfile(partial_path)
                and os.path.getsize(partial_path) > expected_size
            ):
                os.remove(partial_path)
        except OSError:
            pass
        source_url = spec.get("source_url") or (
            f"https://huggingface.co/{spec['repo_id']}"
        )
        raise RuntimeError(
            f"Could not download the {label} model automatically: {exc}. "
            f"Check your internet connection and retry. You can also place "
            f"the official checkpoint at '{save_path}'. Source: "
            f"{source_url} "
            f"({spec['license']})."
        ) from exc


def ensure_video_depth_checkpoint(
    variant: str | None,
    progress: ProgressCallback | None = None,
) -> str:
    """Return a usable Video Depth Anything checkpoint, downloading if needed.

    All configured checkpoint roots are searched for reads, including linked
    installations. New files always go to Maestro's primary checkpoint root.
    """

    _variant, spec = _checkpoint_spec(variant)
    relative_path = _checkpoint_relative_path(spec)

    located = fl.locate_file(relative_path, error_if_none=False)
    if _is_complete_checkpoint(located, spec):
        return str(located)
    if located:
        print(
            f"[ManagedPreprocessor] Ignoring incomplete {spec['label']} "
            f"checkpoint at {located}"
        )

    save_path = os.path.abspath(fl.get_download_location(relative_path))
    with _video_depth_download_lock:
        # Another generation may have completed the download while this one
        # was waiting on the in-process lock.
        located = fl.locate_file(relative_path, error_if_none=False)
        if _is_complete_checkpoint(located, spec):
            return str(located)
        if _is_complete_checkpoint(save_path, spec):
            return save_path
        return _download_with_resume(spec, save_path, progress)


def ensure_minimax_h3_lora_affine_maps(
    architecture: str,
    widths: tuple[int, ...] = (8, 64),
    progress: ProgressCallback | None = None,
) -> list[str]:
    """Provision the tiny Full/Pruned H3 LoRA compatibility packages."""

    architecture = str(architecture or "").strip().casefold()
    architecture_specs = MINIMAX_H3_LORA_AFFINE_MAPS.get(architecture)
    if architecture_specs is None:
        supported = ", ".join(sorted(MINIMAX_H3_LORA_AFFINE_MAPS))
        raise RuntimeError(
            f"Unsupported MiniMax H3 LoRA architecture '{architecture}'. "
            f"Supported architectures: {supported}."
        )

    requested = []
    for width in widths:
        try:
            width = int(width)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid MiniMax H3 LoRA compatibility width '{width}'."
            ) from error
        spec = architecture_specs.get(width)
        if spec is None:
            supported = ", ".join(str(item) for item in sorted(architecture_specs))
            raise RuntimeError(
                f"Unsupported MiniMax H3 {architecture} LoRA compatibility "
                f"width {width}. Supported widths: {supported}."
            )
        relative_path = os.path.join(
            "minimax_h3",
            "lora_affine_maps",
            str(spec["filename"]),
        )
        requested.append((relative_path, spec))

    resolved = []
    with _h3_affine_download_lock:
        for relative_path, spec in requested:
            located = fl.locate_file(relative_path, error_if_none=False)
            if _is_complete_checkpoint(located, spec):
                resolved.append(str(located))
                continue
            if located:
                print(
                    f"[ManagedPreprocessor] Ignoring incomplete {spec['label']} "
                    f"checkpoint at {located}"
                )
            save_path = os.path.abspath(fl.get_download_location(relative_path))
            if _is_complete_checkpoint(save_path, spec):
                resolved.append(save_path)
                continue
            resolved.append(_download_with_resume(spec, save_path, progress))
    return resolved
