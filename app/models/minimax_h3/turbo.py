"""Compatibility helpers for MiniMax H3 low-step Turbo adapters.

The public Turbo LoRA carries a small safetensors metadata marker and needs a
different *step-count convention* from Maestro's original H3 defaults: its
advertised 4/6/8 steps are model evaluations, not sigma-grid points.  Keeping
the detection here avoids importing torch just to validate a selection.
"""

from __future__ import annotations

import json
import os
import struct
from functools import lru_cache


MINIMAX_H3_TURBO_MIN_STEPS = 4

# Maestro intentionally pins the checkpoint that was validated in its native
# H3 runtime instead of following the publisher repository's mutable ``main``
# branch.  The managed preset is optional and downloads this file on first use.
MINIMAX_H3_TURBO_LORA_FILENAME = (
    "minimax_h3_turbo_4step_ckpt500.safetensors"
)
MINIMAX_H3_TURBO_LORA_REPO_ID = "larryvrh/MiniMax-H3-Turbo-Lora"
MINIMAX_H3_TURBO_LORA_REVISION = (
    "7a44622816e16032cb0b6d044d8820da39a1dfdc"
)
MINIMAX_H3_TURBO_LORA_SHA256 = (
    "82d0acff583b04ad9a4238a7440b584b56094bfb7c4fdb2981f67c7a4784b62d"
)
MINIMAX_H3_TURBO_LORA_SIZE = 779_849_872
MINIMAX_H3_TURBO_PRESET_STEPS = 6
MINIMAX_H3_TURBO_PRESET_WEIGHT = 0.50
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024


@lru_cache(maxsize=128)
def _read_safetensors_metadata(
    path: str,
    size: int,
    modified_ns: int,
) -> dict[str, str]:
    """Read only the JSON header of a local safetensors file."""

    del size, modified_ns  # Included in the cache key so replaced files refresh.
    try:
        with open(path, "rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return {}
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 2 <= header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
                return {}
            header = json.loads(handle.read(header_length).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error):
        return {}
    metadata = header.get("__metadata__", {}) if isinstance(header, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def safetensors_metadata(path: str) -> dict[str, str]:
    try:
        stat = os.stat(path)
    except OSError:
        return {}
    return _read_safetensors_metadata(
        os.path.abspath(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def is_minimax_h3_turbo_lora(path: str) -> bool:
    """Recognize LarryVRH's H3 Turbo files, including renamed downloads."""

    basename = os.path.basename(str(path or "")).lower().replace("-", "_")
    if "minimax_h3_turbo" in basename:
        return True
    metadata = safetensors_metadata(path)
    base_model = str(metadata.get("base_model") or "").lower().replace("_", "-")
    application = str(metadata.get("application") or "").lower()
    try:
        sampler_steps = int(metadata.get("sampler_steps") or 0)
    except (TypeError, ValueError):
        sampler_steps = 0
    return (
        base_model == "minimax-h3"
        and sampler_steps >= MINIMAX_H3_TURBO_MIN_STEPS
        and "lora_b" in application
        and "lora_a" in application
    )


def find_minimax_h3_turbo_loras(paths) -> list[str]:
    return [str(path) for path in (paths or []) if is_minimax_h3_turbo_lora(str(path))]


def normalize_minimax_h3_turbo_request(
    body: dict,
    *,
    full_checkpoint: bool,
) -> bool:
    """Apply Maestro's one-click Turbo preset to a generation request.

    The checkbox is deliberately separate from the generic LoRA selector: it
    provides a reproducible low-step recipe while Advanced remains available
    for users who want to select or tune Turbo adapters manually.  Any manually
    selected H3 Turbo variant is replaced by the pinned managed checkpoint so a
    checked preset can never stack two accelerator adapters accidentally.

    Returns ``True`` when the preset was applied and ``False`` when it was not
    requested. Full and Pruned H3 checkpoints share the same adapter after
    Maestro converts its AdaLN input width at load time.
    """

    if not isinstance(body, dict) or body.get("minimax_h3_turbo_mode") is not True:
        return False
    del full_checkpoint  # Retained for request/API compatibility with v1.6.1.

    raw_loras = body.get("activated_loras")
    source_loras = (
        [str(item).strip() for item in raw_loras if str(item).strip()]
        if isinstance(raw_loras, (list, tuple))
        else []
    )
    raw_multipliers = body.get("loras_multipliers")
    if isinstance(raw_multipliers, (list, tuple)):
        source_multipliers = [str(item).strip() for item in raw_multipliers]
    else:
        source_multipliers = str(raw_multipliers or "").split()

    normalized_loras: list[str] = []
    normalized_multipliers: list[str] = []
    selected_turbo_multiplier: str | None = None
    for index, lora in enumerate(source_loras):
        # Turbo mode owns the one accelerator slot. Preserve every unrelated
        # user LoRA and its aligned multiplier, but discard other H3 Turbo
        # variants to avoid double-applying two distillation adapters. The
        # managed adapter remains visible in Advanced, so retain a valid
        # strength the user adjusted there instead of resetting it on submit.
        if is_minimax_h3_turbo_lora(lora):
            selected_name = os.path.basename(lora.replace("\\", "/"))
            if selected_name.lower() == MINIMAX_H3_TURBO_LORA_FILENAME.lower():
                token = (
                    source_multipliers[index].split(";", 1)[0]
                    if index < len(source_multipliers)
                    else ""
                )
                try:
                    value = float(token)
                except (TypeError, ValueError):
                    value = -1.0
                if 0.0 <= value <= 2.0:
                    selected_turbo_multiplier = f"{value:.2f}"
            continue
        normalized_loras.append(lora)
        normalized_multipliers.append(
            source_multipliers[index]
            if index < len(source_multipliers) and source_multipliers[index]
            else "1.00"
        )

    normalized_loras.append(MINIMAX_H3_TURBO_LORA_FILENAME)
    normalized_multipliers.append(
        selected_turbo_multiplier
        or f"{MINIMAX_H3_TURBO_PRESET_WEIGHT:.2f}"
    )
    body["activated_loras"] = normalized_loras
    body["loras_multipliers"] = " ".join(normalized_multipliers)
    body["num_inference_steps"] = MINIMAX_H3_TURBO_PRESET_STEPS
    return True


def h3_scheduler_grid_points(requested_steps: int, *, turbo_active: bool) -> int:
    """Convert the UI's requested evaluations to H3 sigma-grid points."""

    requested_steps = int(requested_steps)
    return requested_steps + 1 if turbo_active else requested_steps


__all__ = [
    "MINIMAX_H3_TURBO_LORA_FILENAME",
    "MINIMAX_H3_TURBO_LORA_REPO_ID",
    "MINIMAX_H3_TURBO_LORA_REVISION",
    "MINIMAX_H3_TURBO_LORA_SHA256",
    "MINIMAX_H3_TURBO_LORA_SIZE",
    "MINIMAX_H3_TURBO_MIN_STEPS",
    "MINIMAX_H3_TURBO_PRESET_STEPS",
    "MINIMAX_H3_TURBO_PRESET_WEIGHT",
    "find_minimax_h3_turbo_loras",
    "h3_scheduler_grid_points",
    "is_minimax_h3_turbo_lora",
    "normalize_minimax_h3_turbo_request",
    "safetensors_metadata",
]
