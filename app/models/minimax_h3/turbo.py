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
from pathlib import Path


MINIMAX_H3_TURBO_MIN_STEPS = 4

_TURBO_PRESETS_PATH = Path(__file__).with_name("turbo_presets.json")


def _load_turbo_manifest() -> dict:
    try:
        manifest = json.loads(_TURBO_PRESETS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not load MiniMax H3 Turbo presets: {error}"
        ) from error
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("presets"), list
    ):
        raise RuntimeError("MiniMax H3 Turbo preset manifest is malformed")
    default_id = str(manifest.get("default_preset_id") or "")
    ids = {
        str(item.get("id") or "")
        for item in manifest["presets"]
        if isinstance(item, dict)
    }
    if not default_id or default_id not in ids:
        raise RuntimeError("MiniMax H3 Turbo default preset is missing")
    workflow_defaults = manifest.get("workflow_default_preset_ids") or {}
    if not isinstance(workflow_defaults, dict):
        raise RuntimeError(
            "MiniMax H3 Turbo workflow defaults are malformed"
        )
    for workflow, preset_id in workflow_defaults.items():
        normalized_workflow = str(workflow or "").strip().lower()
        if normalized_workflow not in {"fl2va", "ref2va"}:
            raise RuntimeError(
                f"Unknown MiniMax H3 Turbo default workflow '{workflow}'"
            )
        if str(preset_id or "") not in ids:
            raise RuntimeError(
                f"MiniMax H3 Turbo default for {normalized_workflow} is missing"
            )
    return manifest


MINIMAX_H3_TURBO_MANIFEST = _load_turbo_manifest()
MINIMAX_H3_TURBO_DEFAULT_PRESET_ID = str(
    MINIMAX_H3_TURBO_MANIFEST["default_preset_id"]
)
MINIMAX_H3_TURBO_PRESETS = tuple(
    dict(item)
    for item in MINIMAX_H3_TURBO_MANIFEST["presets"]
    if isinstance(item, dict)
)
_MINIMAX_H3_TURBO_PRESETS_BY_ID = {
    str(item["id"]): item for item in MINIMAX_H3_TURBO_PRESETS
}
_MINIMAX_H3_TURBO_PRESETS_BY_FILENAME = {
    str(item["filename"]).lower(): item for item in MINIMAX_H3_TURBO_PRESETS
}


def _normalize_workflow(workflow: str | None) -> str | None:
    value = str(workflow or "").strip().lower()
    if not value:
        return None
    if value in {"base", "first_last", "first/last", "fl2va"}:
        return "fl2va"
    if value in {"omni", "reference", "ref2va"}:
        return "ref2va"
    raise ValueError(f"Unknown MiniMax H3 workflow '{workflow}'.")


def minimax_h3_turbo_presets_for_workflow(
    workflow: str | None,
    *,
    full_checkpoint: bool | None = None,
) -> tuple[dict, ...]:
    """Return Turbo presets compatible with one H3 workflow/checkpoint."""

    resolved = _normalize_workflow(workflow)
    return tuple(
        dict(item)
        for item in MINIMAX_H3_TURBO_PRESETS
        if (
            resolved is None
            or str(item.get("workflow") or "all").lower() in {"all", resolved}
        )
        and not (
            full_checkpoint is False
            and bool(item.get("full_checkpoint_only"))
        )
    )


def minimax_h3_turbo_preset(
    preset_id: str | None = None,
    *,
    workflow: str | None = None,
    full_checkpoint: bool | None = None,
) -> dict:
    """Return a copy of a pinned Turbo preset, defaulting to Maestro's current one."""

    resolved_workflow = _normalize_workflow(workflow)
    workflow_defaults = (
        MINIMAX_H3_TURBO_MANIFEST.get("workflow_default_preset_ids") or {}
    )
    resolved_id = str(
        preset_id
        or workflow_defaults.get(resolved_workflow)
        or MINIMAX_H3_TURBO_DEFAULT_PRESET_ID
    )
    preset = _MINIMAX_H3_TURBO_PRESETS_BY_ID.get(resolved_id)
    if preset is None:
        choices = ", ".join(_MINIMAX_H3_TURBO_PRESETS_BY_ID)
        raise ValueError(
            f"Unknown MiniMax H3 Turbo preset '{resolved_id}'. "
            f"Choose one of: {choices}."
        )
    preset_workflow = str(preset.get("workflow") or "all").lower()
    if (
        resolved_workflow is not None
        and preset_workflow not in {"all", resolved_workflow}
    ):
        raise ValueError(
            f"MiniMax H3 Turbo preset '{resolved_id}' is for "
            f"{preset_workflow.upper()}, not {resolved_workflow.upper()}."
        )
    if full_checkpoint is False and bool(preset.get("full_checkpoint_only")):
        required_model = (
            "H3 Omni — Full"
            if preset_workflow == "ref2va"
            else "H3 First / Last — Full"
        )
        raise ValueError(
            f"{preset.get('label') or resolved_id} requires {required_model}. "
            f"Choose {required_model} or another Turbo preset."
        )
    return dict(preset)


def minimax_h3_turbo_preset_for_path(path: str) -> dict | None:
    basename = os.path.basename(str(path or "").replace("\\", "/")).lower()
    preset = _MINIMAX_H3_TURBO_PRESETS_BY_FILENAME.get(basename)
    return dict(preset) if preset is not None else None


# Backward-compatible constants always describe Maestro's current default.
_DEFAULT_TURBO_PRESET = minimax_h3_turbo_preset()
MINIMAX_H3_TURBO_LORA_FILENAME = str(_DEFAULT_TURBO_PRESET["filename"])
MINIMAX_H3_TURBO_LORA_REPO_ID = str(
    _DEFAULT_TURBO_PRESET.get("repo_id")
    or MINIMAX_H3_TURBO_MANIFEST["repo_id"]
)
MINIMAX_H3_TURBO_LORA_REVISION = str(_DEFAULT_TURBO_PRESET["revision"])
MINIMAX_H3_TURBO_LORA_SHA256 = str(_DEFAULT_TURBO_PRESET["sha256"])
MINIMAX_H3_TURBO_LORA_SIZE = int(_DEFAULT_TURBO_PRESET["size"])
MINIMAX_H3_TURBO_PRESET_STEPS = int(_DEFAULT_TURBO_PRESET["steps"])
MINIMAX_H3_TURBO_PRESET_WEIGHT = float(_DEFAULT_TURBO_PRESET["weight"])
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
    """Recognize standard and PDD H3 acceleration adapters."""

    basename = os.path.basename(str(path or "")).lower().replace("-", "_")
    if "minimax_h3_turbo" in basename:
        return True
    if "minimax_h3" in basename and "acc_8step" in basename:
        return True
    metadata = safetensors_metadata(path)
    base_model = str(metadata.get("base_model") or "").lower().replace("_", "-")
    application = str(metadata.get("application") or "").lower()
    try:
        sampler_steps = int(metadata.get("sampler_steps") or 0)
    except (TypeError, ValueError):
        sampler_steps = 0
    standard = (
        base_model == "minimax-h3"
        and sampler_steps >= MINIMAX_H3_TURBO_MIN_STEPS
        and "lora_b" in application
        and "lora_a" in application
    )
    try:
        pdd_steps = int(metadata.get("pdd_num_steps") or 0)
        pdd_block = int(metadata.get("pdd_block_size") or 0)
    except (TypeError, ValueError):
        pdd_steps = pdd_block = 0
    return standard or (
        pdd_steps > 0
        and pdd_block > 0
        and pdd_steps % pdd_block == 0
    )


def is_minimax_h3_pdd_lora(path: str) -> bool:
    """Recognize Alibaba PAI's interval-head PDD acceleration adapters."""

    preset = minimax_h3_turbo_preset_for_path(path)
    if preset is not None and str(preset.get("runtime")) == "pdd":
        return True
    basename = os.path.basename(str(path or "")).lower().replace("-", "_")
    if "minimax_h3" in basename and "acc_8step" in basename:
        return True
    metadata = safetensors_metadata(path)
    try:
        steps = int(metadata.get("pdd_num_steps") or 0)
        block = int(metadata.get("pdd_block_size") or 0)
    except (TypeError, ValueError):
        return False
    return steps > 0 and block > 0 and steps % block == 0


def find_minimax_h3_turbo_loras(paths) -> list[str]:
    return [str(path) for path in (paths or []) if is_minimax_h3_turbo_lora(str(path))]


def find_minimax_h3_pdd_loras(paths) -> list[str]:
    return [str(path) for path in (paths or []) if is_minimax_h3_pdd_lora(str(path))]


def normalize_minimax_h3_turbo_request(
    body: dict,
    *,
    full_checkpoint: bool,
    workflow: str | None = None,
) -> bool:
    """Apply Maestro's one-click Turbo preset to a generation request.

    The checkbox is deliberately separate from the generic LoRA selector: it
    provides a reproducible low-step recipe while Advanced remains available
    for users who want to select or tune Turbo adapters manually.  Any manually
    selected H3 Turbo variant is replaced by the chosen pinned manifest entry
    so a checked preset can never stack two accelerator adapters accidentally.

    Returns ``True`` when the preset was applied and ``False`` when it was not
    requested. Presets can also carry workflow-specific conditioning defaults.
    An explicit request value always wins: selecting ``Match output`` must not
    be silently replaced by the Ref2VA PDD preset's official high-detail
    fallback.
    """

    if not isinstance(body, dict) or body.get("minimax_h3_turbo_mode") is not True:
        return False

    preset = minimax_h3_turbo_preset(
        body.get("minimax_h3_turbo_preset"),
        workflow=workflow,
        full_checkpoint=full_checkpoint,
    )
    preset_filename = str(preset["filename"])
    body["minimax_h3_turbo_preset"] = str(preset["id"])

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
            if selected_name.lower() == preset_filename.lower():
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

    normalized_loras.append(preset_filename)
    normalized_multipliers.append(
        selected_turbo_multiplier
        or f"{float(preset['weight']):.2f}"
    )
    body["activated_loras"] = normalized_loras
    body["loras_multipliers"] = " ".join(normalized_multipliers)
    body["num_inference_steps"] = int(preset["steps"])
    preset_reference_detail = str(
        preset.get("reference_detail") or ""
    ).strip().lower()
    if preset_reference_detail:
        if preset_reference_detail not in {"match", "max"}:
            raise ValueError(
                f"MiniMax H3 Turbo preset '{preset['id']}' has an invalid "
                f"reference detail '{preset_reference_detail}'."
            )
        requested_reference_detail = str(
            body.get("minimax_h3_reference_detail") or ""
        ).strip().lower()
        if requested_reference_detail:
            if requested_reference_detail not in {"match", "max"}:
                raise ValueError(
                    "MiniMax H3 reference detail must be 'match' or 'max'."
                )
        else:
            # Keep the official recipe as a fallback for API callers that do
            # not send this setting, while preserving the mode the UI/user
            # explicitly selected.
            requested_reference_detail = preset_reference_detail
        body["minimax_h3_reference_detail"] = requested_reference_detail
    return True


def h3_scheduler_grid_points(requested_steps: int, *, turbo_active: bool) -> int:
    """Convert the UI's requested evaluations to H3 sigma-grid points.

    H3's scheduler includes the terminal clean sigma in its grid and therefore
    performs one fewer model evaluation than there are grid points.  The UI
    value is an evaluation count for every H3 mode, so every schedule needs the
    additional terminal point.  ``turbo_active`` remains in the signature for
    compatibility with existing callers and tests; Turbo previously applied
    this correction on its own while standard H3 did not.
    """

    requested_steps = int(requested_steps)
    return requested_steps + 1


__all__ = [
    "MINIMAX_H3_TURBO_LORA_FILENAME",
    "MINIMAX_H3_TURBO_LORA_REPO_ID",
    "MINIMAX_H3_TURBO_LORA_REVISION",
    "MINIMAX_H3_TURBO_LORA_SHA256",
    "MINIMAX_H3_TURBO_LORA_SIZE",
    "MINIMAX_H3_TURBO_DEFAULT_PRESET_ID",
    "MINIMAX_H3_TURBO_MANIFEST",
    "MINIMAX_H3_TURBO_MIN_STEPS",
    "MINIMAX_H3_TURBO_PRESETS",
    "MINIMAX_H3_TURBO_PRESET_STEPS",
    "MINIMAX_H3_TURBO_PRESET_WEIGHT",
    "find_minimax_h3_turbo_loras",
    "find_minimax_h3_pdd_loras",
    "h3_scheduler_grid_points",
    "is_minimax_h3_pdd_lora",
    "is_minimax_h3_turbo_lora",
    "minimax_h3_turbo_preset",
    "minimax_h3_turbo_preset_for_path",
    "minimax_h3_turbo_presets_for_workflow",
    "normalize_minimax_h3_turbo_request",
    "safetensors_metadata",
]
