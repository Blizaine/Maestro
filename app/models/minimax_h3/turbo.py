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


def h3_scheduler_grid_points(requested_steps: int, *, turbo_active: bool) -> int:
    """Convert the UI's requested evaluations to H3 sigma-grid points."""

    requested_steps = int(requested_steps)
    return requested_steps + 1 if turbo_active else requested_steps


__all__ = [
    "MINIMAX_H3_TURBO_MIN_STEPS",
    "find_minimax_h3_turbo_loras",
    "h3_scheduler_grid_points",
    "is_minimax_h3_turbo_lora",
    "safetensors_metadata",
]
