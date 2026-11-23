# Adapted from WanGP commit 1830091bf4b27df2f901920d55b1fb748f33e7eb.
# WanGP Community License 2.0; the pinned source and license are documented
# in UPSTREAM.md.

"""Convert MiniMax H3 LoRAs between Full and Pruned AdaLN widths.

The affine conversion was introduced by WanGP for its Full/Pruned H3 LoRA
compatibility.  Maestro keeps the small, revision-pinned fit packages in the
checkpoint area instead of redistributing them with the application.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache

import torch
from safetensors.torch import load, load_file

from shared.utils import files_locator as fl


FULL_TIME_DIM = 2688
_PRUNED_WIDTHS = (4, 8, 64)
_AFFINE_FIT_RELATIVE_TOLERANCE = 1e-5
_BF16_ROUNDING_BUDGET_FACTOR = 1.25
_MAP_RELATIVE_DIR = os.path.join("minimax_h3", "lora_affine_maps")
_ARCHITECTURES = {
    "viggle_animate": "viggle",
    "viggle_remake_swap": "fl2va",
    "viggle_remake_swap_aligned": "ref2va",
    "viggle_remake_edit": "fl2va",
    "minimax_h3": "fl2va",
    "minimax_h3_full": "fl2va",
    "minimax_h3_ref2va": "ref2va",
    "minimax_h3_ref2va_full": "ref2va",
    "minimax_h3_regenerate_ic": "ref2va",
    # H3 Image shares the FL2VA transformer/AdaLN basis. Its six native
    # output heads change sampling, not the backbone LoRA architecture.
    "h3_image": "fl2va",
}
_LORA_SUFFIXES = (
    ("lora_A.weight", "lora_B.weight"),
    ("lora_A.default.weight", "lora_B.default.weight"),
    ("lora_down.weight", "lora_up.weight"),
    ("lora_down.default.weight", "lora_up.default.weight"),
    ("lora.A.weight", "lora.B.weight"),
    ("lora.A.default.weight", "lora.B.default.weight"),
    ("lora.down.weight", "lora.up.weight"),
    ("lora.down.default.weight", "lora.up.default.weight"),
)
_SOURCE_TENSOR = "__wangp_h3_lora_source__"


@dataclass
class AffinePackage:
    """A source LoRA's exact pruned timestep table and full affine map."""

    name: str
    table: torch.Tensor
    affine: torch.Tensor
    full_fingerprint: str = ""

    @property
    def width(self) -> int:
        return int(self.table.shape[1])


def _read_affine_package(data: bytes, name: str = "embedded") -> AffinePackage:
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("MiniMax H3 embedded affine package exceeds 4 MiB")
    tensors = load(data)
    try:
        table = tensors["adaln_t_table"]
        affine = tensors["adaln_affine_map"]
    except KeyError as error:
        raise ValueError("MiniMax H3 embedded affine package is incomplete") from error
    header_size = int.from_bytes(data[:8], "little")
    try:
        metadata = json.loads(data[8 : 8 + header_size]).get("__metadata__", {})
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MiniMax H3 embedded affine package has invalid metadata") from error
    if (
        table.dtype != torch.float32
        or affine.dtype != torch.float32
        or table.ndim != 2
        or not 2 <= table.shape[0] <= 4097
        or table.shape[1] not in (4, 8, 64)
        or affine.shape != (table.shape[1] + 1, FULL_TIME_DIM)
    ):
        raise ValueError("Invalid MiniMax H3 affine package dimensions or dtype")
    if not torch.isfinite(table).all() or not torch.isfinite(affine).all():
        raise ValueError("MiniMax H3 affine package contains non-finite values")
    return AffinePackage(
        str(metadata.get("h3_map_id") or name),
        table,
        affine,
        str(metadata.get("full_time_fingerprint") or ""),
    )


def _decode_affine_package(value: str) -> AffinePackage:
    if not isinstance(value, str) or len(value) > 6 * 1024 * 1024:
        raise ValueError("Invalid MiniMax H3 affine metadata")
    try:
        payload = json.loads(value)
        if payload.get("version") != 1:
            raise ValueError("Unsupported MiniMax H3 affine metadata version")
        data = base64.b64decode(payload["package"], validate=True)
        expected = str(payload["sha256"])
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Invalid MiniMax H3 affine metadata") from error
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("MiniMax H3 affine metadata checksum mismatch")
    return _read_affine_package(data)


def pop_source_package(state_dict: dict) -> AffinePackage | None:
    """Remove and validate WanGP's embedded source-table marker if present."""
    value = state_dict.pop(_SOURCE_TENSOR, None)
    if value is None:
        return None
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() > 6 * 1024 * 1024:
        raise ValueError("Invalid MiniMax H3 LoRA source-table tag")
    try:
        encoded = value.detach().cpu().numpy().tobytes().decode("utf-8")
    except (UnicodeDecodeError, RuntimeError) as error:
        raise ValueError("Invalid MiniMax H3 LoRA source-table tag") from error
    return _decode_affine_package(encoded)


def _package_at_width(package: AffinePackage, width: int) -> AffinePackage:
    if package.width == width:
        return package
    if width == 4 and package.width == 8:
        return AffinePackage(
            package.name,
            package.table[:, :4],
            torch.cat((package.affine[:4], package.affine[-1:])),
            package.full_fingerprint,
        )
    raise ValueError(
        f"MiniMax H3 embedded affine width {package.width} cannot map to rank {width}"
    )


def _architecture(model_type: str) -> str:
    try:
        return _ARCHITECTURES[str(model_type)]
    except KeyError as error:
        raise ValueError(
            "Unsupported MiniMax H3 architecture for AdaLN LoRA conversion: "
            f"{model_type}"
        ) from error


def _map_path(architecture: str, width: int) -> str:
    filename = f"{architecture}_rank{width}.sft"
    relative_path = os.path.join(_MAP_RELATIVE_DIR, filename)
    path = fl.locate_file(relative_path, error_if_none=False)
    if path:
        return str(path)
    raise ValueError(
        "MiniMax H3 Full/Pruned LoRA compatibility data is missing. "
        f"Expected '{relative_path}'. Restart the generation to retry its "
        "small one-time support download."
    )


@lru_cache(maxsize=6)
def _load_affine_package(architecture: str, width: int = 8):
    package_width = 8 if width == 4 else width
    tensors = load_file(
        _map_path(architecture, package_width),
        device="cpu",
    )
    table = tensors["adaln_t_table"].float()
    affine = tensors["adaln_affine_map"].float()
    if (
        table.ndim != 2
        or table.shape[1] != package_width
        or affine.shape != (package_width + 1, FULL_TIME_DIM)
    ):
        raise ValueError(
            f"Invalid MiniMax H3 {architecture} rank-{package_width} "
            "AdaLN affine package"
        )
    if width == 4:
        table = table[:, :width]
        affine = torch.cat((affine[:width], affine[-1:]))
    return table, affine


def _bf16_rounding_fit_error(
    canonical_table: torch.Tensor,
    target_table: torch.Tensor,
) -> float | None:
    """Estimate only the affine-fit residual expected from a BF16 table.

    Imported H3 checkpoints can store an FP32 curve buffer whose values were
    rounded to BF16 before serialization. A least-squares affine map cannot
    remove that per-entry rounding noise, so compare its residual with a
    simulated BF16 rounding of the best affine prediction. This keeps the
    compatibility guard strict for ordinary FP32 tables and for residuals
    larger than the checkpoint's measured rounding floor.
    """

    target_float = target_table.float()
    if not torch.equal(target_float, target_float.to(torch.bfloat16).float()):
        return None

    canonical_h = torch.cat(
        (
            canonical_table.double(),
            canonical_table.new_ones(canonical_table.shape[0], 1, dtype=torch.float64),
        ),
        dim=1,
    )
    forward_fit = torch.linalg.lstsq(
        canonical_h,
        target_float.double(),
        rcond=1e-14,
    )
    if int(forward_fit.rank) != canonical_h.shape[1]:
        return None

    rounded_target = (canonical_h @ forward_fit.solution).to(
        torch.bfloat16
    ).float().double()
    rounded_h = torch.cat(
        (
            rounded_target,
            rounded_target.new_ones(rounded_target.shape[0], 1, dtype=torch.float64),
        ),
        dim=1,
    )
    reverse_fit = torch.linalg.lstsq(
        rounded_h,
        canonical_h,
        rcond=1e-14,
    )
    if int(reverse_fit.rank) != rounded_h.shape[1]:
        return None
    denominator = torch.linalg.vector_norm(canonical_h)
    if denominator == 0:
        return None
    return float(
        torch.linalg.vector_norm(rounded_h @ reverse_fit.solution - canonical_h)
        / denominator
    )


def _aligned_affine_map(architecture: str, target_table: torch.Tensor):
    target_table = target_table.detach().to(device="cpu", dtype=torch.float64)
    if target_table.ndim != 2 or target_table.shape[1] not in _PRUNED_WIDTHS:
        raise ValueError(
            f"Unsupported MiniMax H3 {architecture} AdaLN target table shape "
            f"{tuple(target_table.shape)}"
        )
    canonical_table, canonical_affine = _load_affine_package(
        architecture,
        target_table.shape[1],
    )
    if target_table.shape[0] != canonical_table.shape[0]:
        position = torch.linspace(
            0,
            canonical_table.shape[0] - 1,
            target_table.shape[0],
            dtype=torch.float64,
            device=canonical_table.device,
        )
        lower = position.floor().long().clamp(max=canonical_table.shape[0] - 2)
        canonical_table = torch.lerp(
            canonical_table[lower].double(),
            canonical_table[lower + 1].double(),
            (position - lower).unsqueeze(1),
        )
    elif torch.equal(target_table.float(), canonical_table):
        return canonical_affine

    ones = target_table.new_ones(target_table.shape[0], 1)
    target_h = torch.cat((target_table, ones), dim=1)
    canonical_h = torch.cat((canonical_table.double(), ones), dim=1)
    fit = torch.linalg.lstsq(target_h, canonical_h, rcond=1e-14)
    if int(fit.rank) != target_h.shape[1]:
        raise ValueError(
            f"MiniMax H3 {architecture} checkpoint has a rank-deficient "
            "AdaLN table"
        )
    relative_error = (
        torch.linalg.vector_norm(target_h @ fit.solution - canonical_h)
        / torch.linalg.vector_norm(canonical_h)
    )
    allowed_error = _AFFINE_FIT_RELATIVE_TOLERANCE
    if relative_error > allowed_error:
        rounding_floor = _bf16_rounding_fit_error(
            canonical_table,
            target_table,
        )
        if rounding_floor is not None:
            allowed_error = max(
                allowed_error,
                rounding_floor * _BF16_ROUNDING_BUDGET_FACTOR + 1e-7,
            )
    if relative_error > allowed_error:
        raise ValueError(
            f"MiniMax H3 {architecture} checkpoint AdaLN table is "
            "incompatible with the canonical LoRA map "
            f"(relative error {relative_error:.3g})"
        )
    return (fit.solution @ canonical_affine.double()).float()


@lru_cache(maxsize=6)
def _canonical_encoder(architecture: str, width: int):
    _, affine = _load_affine_package(architecture, width)
    return torch.linalg.pinv(affine[:width].double(), rtol=1e-14).T.float()


def _add_bias_delta(state_dict: dict, key: str, delta: torch.Tensor) -> None:
    existing = state_dict.get(key)
    if existing is not None:
        if existing.shape != delta.shape:
            raise ValueError(
                f"MiniMax H3 LoRA bias delta shape mismatch for {key}: "
                f"{tuple(existing.shape)} != {tuple(delta.shape)}"
            )
        delta.add_(existing.float())
    state_dict[key] = delta


def convert_adaln_loras(
    model_type: str,
    state_dict: dict,
    target_table: torch.Tensor | None = None,
    *,
    source_package: AffinePackage | None = None,
):
    """Convert AdaLN basis/width without changing the LoRA adapter rank."""

    architecture = _architecture(model_type)
    target_width = (
        FULL_TIME_DIM if target_table is None else int(target_table.shape[1])
    )
    candidates = []

    for down_suffix, up_suffix in _LORA_SUFFIXES:
        marker = "." + down_suffix
        for down_key in [
            key
            for key in state_dict
            if key.endswith(marker) and ".adaln_proj.linear." in key
        ]:
            down = state_dict[down_key]
            if down.ndim != 2:
                continue
            module_name = down_key[: -len(marker)]
            up_key = module_name + "." + up_suffix
            if up_key not in state_dict:
                raise ValueError(f"MiniMax H3 LoRA is missing {up_key}")
            up = state_dict[up_key]
            if up.ndim != 2 or up.shape[1] != down.shape[0]:
                raise ValueError(
                    "MiniMax H3 LoRA factors are incompatible for "
                    f"{module_name}: A={tuple(down.shape)}, B={tuple(up.shape)}"
                )
            candidates.append(
                (module_name, down_key, up_key, int(down.shape[1]))
            )

    if not candidates:
        return 0, architecture, target_width, target_width

    source_widths = {candidate[3] for candidate in candidates}
    if len(source_widths) != 1:
        raise ValueError(
            f"MiniMax H3 LoRA mixes AdaLN input widths: {sorted(source_widths)}"
        )
    source_width = source_widths.pop()
    supported_widths = (*_PRUNED_WIDTHS, FULL_TIME_DIM)
    if source_width not in supported_widths or target_width not in supported_widths:
        raise ValueError(
            f"Unsupported MiniMax H3 AdaLN LoRA conversion {source_width} -> "
            f"{target_width}; supported widths are {supported_widths}"
        )
    if source_package is not None:
        source_package = _package_at_width(source_package, source_width)

    target_affine = (
        None
        if target_table is None
        else _aligned_affine_map(architecture, target_table)
    )
    same_basis = source_package is None or (
        target_table is not None
        and source_package.width == target_width
        and torch.equal(
            source_package.table,
            target_table.detach().to(device="cpu", dtype=torch.float32),
        )
        and torch.equal(source_package.affine, target_affine)
    )
    if source_width == target_width and same_basis:
        return 0, architecture, source_width, target_width
    source_affine = (
        None
        if source_width == FULL_TIME_DIM
        else (
            source_package.affine
            if source_package is not None
            else _load_affine_package(architecture, source_width)[1]
        )
    )
    source_encoder = (
        None
        if source_width == FULL_TIME_DIM
        else (
            torch.linalg.pinv(
                source_affine[:source_width].double(), rtol=1e-14
            ).T.float()
            if source_package is not None
            else _canonical_encoder(architecture, source_width)
        )
    )

    for module_name, down_key, up_key, _ in candidates:
        down = state_dict[down_key]
        up = state_dict[up_key]
        mapped = (
            down.float()
            if source_encoder is None
            else down.float() @ source_encoder
        )
        inner_bias = (
            mapped.new_zeros(mapped.shape[0])
            if source_affine is None
            else -(mapped @ source_affine[-1])
        )
        if target_affine is not None:
            mapped = mapped @ target_affine.T
            inner_bias.add_(mapped[:, target_width])
            mapped = mapped[:, :target_width]
        state_dict[down_key] = mapped
        _add_bias_delta(
            state_dict,
            module_name + ".diff_b",
            up.float() @ inner_bias,
        )

    return len(candidates), architecture, source_width, target_width


__all__ = ["FULL_TIME_DIM", "convert_adaln_loras"]
