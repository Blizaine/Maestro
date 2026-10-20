"""Shared resource-gated H3 duration recommendations and their evidence basis."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


_RECOMMENDATIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "shared"
    / "h3_duration_recommendations.json"
)


@lru_cache(maxsize=1)
def load_h3_duration_recommendations() -> dict[str, Any]:
    """Load the versioned backend/UI policy document once per process."""

    try:
        value = json.loads(_RECOMMENDATIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": "", "evidence": [], "tiers": []}
    if not isinstance(value, dict):
        return {"version": "", "evidence": [], "tiers": []}
    return value


def h3_duration_recommendations_enabled(model_def: Mapping[str, Any] | None) -> bool:
    """Keep empirical tiers away from alternate, audio, and fixed workflows."""

    if not isinstance(model_def, Mapping):
        return True
    excluded_flags = (
        "minimax_h3_full_checkpoint",
        "audio_only",
        "minimax_h3_audio_only",
        "minimax_h3_viggle",
        "minimax_h3_singularity",
        "minimax_h3_dasiwa",
        "minimax_h3_baked_turbo",
        "minimax_h3_fasth3",
        "vdn",
    )
    if any(bool(model_def.get(flag)) for flag in excluded_flags):
        return False
    if model_def.get("minimax_h3_import_profile"):
        return False
    architecture = str(
        model_def.get("architecture") or model_def.get("base_model_type") or ""
    ).strip().lower()
    if architecture.startswith("viggle") or architecture.endswith("_full"):
        return False
    return True


def attach_h3_duration_recommendations(
    policy: Mapping[str, Any],
    model_def: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a policy copy, attaching tiers only to covered video workflows."""

    result = dict(policy)
    if h3_duration_recommendations_enabled(model_def):
        result["resource_recommendations"] = copy.deepcopy(
            load_h3_duration_recommendations()
        )
    else:
        result.pop("resource_recommendations", None)
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def match_h3_duration_recommendation(
    resource_recommendations: Mapping[str, Any] | None,
    *,
    checkpoint: str,
    total_vram_gb: Any,
    total_ram_gb: Any,
    pixels: Any,
) -> dict[str, Any] | None:
    """Return the first fully matching tested or estimated tier, or ``None``.

    RAM is mandatory for these tiers. Callers should retain their existing
    VRAM-only profile when RAM is unknown or a row does not match.
    """

    document = resource_recommendations
    if not isinstance(document, Mapping):
        document = load_h3_duration_recommendations()
    vram = _number(total_vram_gb)
    ram = _number(total_ram_gb)
    area = _number(pixels)
    if vram is None or ram is None or area is None or min(vram, ram, area) <= 0:
        return None

    checkpoint = str(checkpoint or "").strip().lower()
    tiers = document.get("tiers") or []
    if not isinstance(tiers, list):
        return None
    for tier in tiers:
        if not isinstance(tier, Mapping):
            continue
        if str(tier.get("checkpoint") or "").strip().lower() != checkpoint:
            continue
        minimum_vram = _number(tier.get("min_vram_gb"))
        maximum_vram = _number(tier.get("max_vram_gb"))
        minimum_ram = _number(tier.get("min_ram_gb"))
        maximum_ram = _number(tier.get("max_ram_gb"))
        minimum_pixels = _number(tier.get("min_pixels", 0))
        maximum_pixels = _number(tier.get("max_pixels"))
        if None in (minimum_vram, maximum_vram, minimum_ram, minimum_pixels, maximum_pixels):
            continue
        if "max_ram_gb" in tier and maximum_ram is None:
            continue
        if not minimum_vram <= vram <= maximum_vram:
            continue
        if ram < minimum_ram or not minimum_pixels <= area <= maximum_pixels:
            continue
        if maximum_ram is not None and ram > maximum_ram:
            continue
        if "frames" not in tier:
            continue
        if tier["frames"] is not None:
            frames = _number(tier["frames"])
            if frames is None or not frames.is_integer() or frames <= 0:
                continue
        return dict(tier)
    return None


def h3_recommendation_metadata(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expose provenance without turning an estimate into a memory guarantee."""

    profile = profile or {}
    tier = profile.get("resource_recommendation") or {}
    basis = tier.get("basis")
    if basis not in {"tested", "reported", "estimated", "conservative"}:
        basis = "conservative"
    return {
        "recommendation_basis": basis,
        "recommendation_note": str(tier.get("note") or ""),
        "recommendation_version": profile.get("resource_recommendation_version"),
    }
