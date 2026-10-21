# SPDX-License-Identifier: Apache-2.0
"""Pinned FastH3 OmniRef Comfy checkpoints and native PDD-8 recipe.

PDD arithmetic follows FastVideo at d4f144581d6882d5bceb288ca13d767c054ca041,
fastvideo/layers/pdd.py. This is distinct from the T2AV DMD schedule and the
Alibaba acceleration adapter: its fine grid ends at .999, not 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from . import fasth3

REPO = "FastVideo/FastH3-Omni-Comfy"
REVISION = "c6e7d1f212c44b21227bbd6e76bfff7da70e285d"
FILE = "fasth3_omniref_8step_pruned_int8_convrot.safetensors"
SIZE = 22213723064
SHA256 = "851b2dc9670ed864b7f1e951ac9da047d83b1c04ded9bfcc1a9e5bba0ee33f80"
BF16_FILE = "fasth3_omniref_8step_pruned_bf16.safetensors"
BF16_SIZE = 44240780704
BF16_SHA256 = "c0fac98c0317eb85215bd186b250be9b113675baab1323add40f26262f552ed9"
FP8_REPO = "FastVideo/FastH3-Omni-FP8-Comfy"
FP8_REVISION = "03127300d4803d5b66df66a22208b1d9238c8588"
FP8_FILE = "fasth3_omniref_fp8_comfy.safetensors"
FP8_SIZE = 23046497628
FP8_SHA256 = "d3aa751e151df957b6248db7dbe4a159549dfefe78ba884056b12b0425b019f3"
NVFP4_REPO = "FastVideo/FastH3-Omni-NVFP4-QAD-Comfy"
NVFP4_REVISION = "e1f39b9187e3e4059b7507e0a62fa22621ff77a5"
NVFP4_FILE = "fasth3_omniref_qad_nvfp4_comfy.safetensors"
NVFP4_SIZE = 13773998700
NVFP4_SHA256 = "bf25ee58df65b4dfee349d612f7ac62a2fea0dddd1e68eb37e75c8ce8b166f5a"
_SOURCE_REPO = "FastVideo/FastVideo-FastH3-Omni-8-Step-V1"
_SOURCE_REVISION = "88d512c5e7cc38dda499fbb65688ce58611f9c80"
_BASE_MODEL_REVISION = "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da"
_COMFYUI_COMMIT = "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c"
_COMFY_MODEL_TOOLS_COMMIT = "d6797787e6bdb1a1fb0094d588a26f8e71a1c757"
_PDD_HEAD_STORAGE = "head0-full;heads1-31-offset-from-head0;fp32"
_QUANTIZATION_RECIPE = "omniref-comfy-ptq-v1"
_NVFP4_QUANTIZATION_SOURCE_SHA256 = "4e244969987e3c6e16bffa449bcf62c1f7c83b5cce15e5806246d3ae8055154f"
GRID_MAX = 0.999
HEADS = 32
EVALUATIONS = 8
HELP = (
    "FASTH3 OMNIREF — EXPERIMENTAL\n"
    "Official eight-step Ref2VA checkpoint with ordered image, video and audio references. "
    "Uses 32 PDD heads in eight blocks, Euler, shifts 12/3 and trained VSA-H3 "
    "attention (90% sparsity, 4×4×8 tiles). At least one image or video reference is required. "
    "Available in Studio References. Director, start/end frames, continuation, edit masks, additional LoRAs and audio refinement are unavailable. "
    "Begin at 480p / 5.2 seconds. GPU performance depends on the selected precision and hardware."
)


PINNED_CHECKPOINTS = {
    "bf16": {
        "repo": REPO,
        "revision": REVISION,
        "file": f"diffusion_models/{BF16_FILE}",
        "size": BF16_SIZE,
        "sha256": BF16_SHA256,
        "label": "official BF16",
    },
    "int8": {
        "repo": REPO,
        "revision": REVISION,
        "file": f"diffusion_models/{FILE}",
        "size": SIZE,
        "sha256": SHA256,
        "label": "official INT8 ConvRot",
    },
    "fp8": {
        "repo": FP8_REPO,
        "revision": FP8_REVISION,
        "file": FP8_FILE,
        "size": FP8_SIZE,
        "sha256": FP8_SHA256,
        "label": "official Comfy FP8",
    },
    "nvfp4": {
        "repo": NVFP4_REPO,
        "revision": NVFP4_REVISION,
        "file": NVFP4_FILE,
        "size": NVFP4_SIZE,
        "sha256": NVFP4_SHA256,
        "label": "official Comfy NVFP4 QAD",
    },
}

_OFFICIAL_INFERENCE_CONTRACT = {
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "audio_scheduler_shift": 3.0,
    "base_model_revision": _BASE_MODEL_REVISION,
    "conditioning": "fixed_ordered_references_target_only_flow",
    "grid_max_t": GRID_MAX,
    "guidance_scale": 1.0,
    "model_type": "ref2va",
    "num_inference_steps": EVALUATIONS,
    "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
    "pdd_steps": HEADS,
    "schema": "fasth3-inference-contract-v1",
    "schema_version": "fasth3-inference-contract-v1",
    "transformer_component": "transformer_ref",
    "transformer_forwards": EVALUATIONS,
    "video_scheduler_shift": 12.0,
    "vsa_ref_keep_rate": 0.1,
    "vsa_ref_policy": "p2_multi_region",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
}

_GATE_SHAPES = {
    "bf16": ("BF16", [7168, 5376]),
    "int8": ("I8", [7168, 5376]),
    "fp8": ("F8_E4M3", [7168, 5376]),
    "nvfp4": ("U8", [7168, 2688]),
}

_QUANTIZED_LAYERS = (
    "all 50 DiT blocks: fused qkv_proj, out_proj, to_gate_compress, mlp.fc1, mlp.fc2"
)
_DENSE_LAYERS = (
    "token_refiner, AdaLN, norms, embeddings, patch/condition projections, FP32 PDD heads"
)
_OFFICIAL_QUANTIZATION = {
    "fp8": {
        "format": "float8_e4m3fn",
        "layout": "TensorCoreFP8Layout",
        "weights": "per-tensor global scale",
        "activations": "dynamic-per-tensor",
        "quantized_layers": _QUANTIZED_LAYERS,
        "dense_layers": _DENSE_LAYERS,
        "qkv_treatment": "fused Q/K/V quantized as one tensor under one weight scale",
        "comfyui_commit": _COMFYUI_COMMIT,
    },
    "nvfp4": {
        "format": "nvfp4",
        "layout": "TensorCoreNVFP4Layout",
        "weights": "per-tensor global scale plus FP8 block scales, group 16",
        "activations": "dynamic-per-tensor",
        "quantized_layers": _QUANTIZED_LAYERS,
        "dense_layers": _DENSE_LAYERS,
        "qkv_treatment": "fused Q/K/V quantized as one tensor under one weight scale",
        "comfyui_commit": _COMFYUI_COMMIT,
    },
}


def checkpoint_variant_from_size(filename: str) -> str:
    """Resolve a pinned variant candidate without trusting a checkpoint filename.

    The result is only a preflight hint. ``verify_checkpoint`` still checks the
    full pinned digest before the checkpoint can be loaded.
    """
    size = Path(filename).stat().st_size
    matches = [name for name, pin in PINNED_CHECKPOINTS.items() if pin["size"] == size]
    if len(matches) != 1:
        raise ValueError("FastH3 OmniRef requires one of its four pinned checkpoints; file size is not recognized.")
    return matches[0]


def _nvfp4_kernel_supported(device: torch.device) -> bool:
    try:
        from shared.qtypes.nvfp4 import _supports_nvfp4_kernel
        return bool(_supports_nvfp4_kernel(device))
    except Exception:
        return False


def require_nvfp4_kernel_support(device: torch.device | str = "cuda") -> None:
    """Fail before model loading unless the QAD checkpoint has a native kernel."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "FastH3 OmniRef QAD NVFP4 requires a CUDA GPU with a supported NVFP4 kernel."
        )
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (10, 0):
        raise RuntimeError(
            "FastH3 OmniRef QAD NVFP4 requires Blackwell-class hardware (CUDA SM 10.0 or newer)."
        )
    if not _nvfp4_kernel_supported(device):
        raise RuntimeError(
            "FastH3 OmniRef QAD NVFP4 requires the Comfy-kitchen or lightx2v NVFP4 CUDA kernels. "
            "Install a supported kernel build before selecting this checkpoint."
        )


def _metadata_json(metadata: dict, name: str) -> dict:
    value = metadata.get(name)
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError) as error:
        raise ValueError(f"FastH3 OmniRef has invalid {name} metadata.") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"FastH3 OmniRef is missing its {name} metadata.")
    return parsed


def _verify_metadata(variant: str, header: dict) -> None:
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("FastH3 OmniRef has invalid checkpoint metadata.")
    config = _metadata_json(metadata, "config")
    transformer = config.get("transformer")
    if not isinstance(transformer, dict):
        raise ValueError("FastH3 OmniRef requires a Ref2VA transformer config.")
    if variant == "int8":
        # This official file is byte-identical to the previously supported
        # community ConvRot pin, which carries the legacy metadata schema.
        if metadata.get("source") != "transformer_ref" or transformer.get("pdd_steps") != HEADS:
            raise ValueError("FastH3 OmniRef INT8 requires the pinned Ref2VA 32-head PDD conversion.")
        return

    if metadata.get("source_repo") != _SOURCE_REPO or metadata.get("source_revision") != _SOURCE_REVISION:
        raise ValueError("FastH3 OmniRef must use the pinned official Ref2VA source revision.")
    if metadata.get("comfyui_commit") != _COMFYUI_COMMIT or metadata.get("comfy_model_tools_commit") != _COMFY_MODEL_TOOLS_COMMIT:
        raise ValueError("FastH3 OmniRef was not converted with the pinned Comfy checkpoint tools.")
    if metadata.get("pdd_head_storage") != _PDD_HEAD_STORAGE:
        raise ValueError("FastH3 OmniRef requires the verified FP32 residual PDD head layout.")
    if _metadata_json(metadata, "source_inference_contract") != _OFFICIAL_INFERENCE_CONTRACT:
        raise ValueError("FastH3 OmniRef inference contract differs from Ref2VA PDD-8; T2AV schedules are incompatible.")
    expected_config = {
        "pdd_steps": HEADS,
        "sigma_shift_video": 12.0,
        "sigma_shift_audio": 3.0,
        "time_embed_dim": 8,
    }
    if any(transformer.get(key) != value for key, value in expected_config.items()):
        raise ValueError("FastH3 OmniRef requires Ref2VA PDD-8 with video/audio shifts 12/3 and rank-8 AdaLN.")

    if variant in _OFFICIAL_QUANTIZATION:
        if metadata.get("quantization_recipe") != _QUANTIZATION_RECIPE:
            raise ValueError("FastH3 OmniRef quantized checkpoint has an unrecognized Comfy conversion recipe.")
        expected_source = BF16_SHA256 if variant == "fp8" else _NVFP4_QUANTIZATION_SOURCE_SHA256
        if metadata.get("quantization_source_sha256") != expected_source:
            raise ValueError("FastH3 OmniRef quantized checkpoint has an unrecognized BF16 source pin.")
        if _metadata_json(metadata, "quantization") != _OFFICIAL_QUANTIZATION[variant]:
            raise ValueError(f"FastH3 OmniRef {variant.upper()} quantization contract differs from its pinned Comfy export.")
    elif any(key in metadata for key in ("quantization", "quantization_recipe", "quantization_source_sha256")):
        raise ValueError("FastH3 OmniRef BF16 checkpoint must use dense BF16 gate storage.")


def _expect_tensor(header: dict, name: str, dtype: str, shape: list[int] | tuple[int, ...]) -> None:
    spec = header.get(name)
    expected_shape = list(shape)
    if not isinstance(spec, dict) or spec.get("dtype") != dtype or spec.get("shape") != expected_shape:
        raise ValueError(f"FastH3 OmniRef tensor {name} must be {dtype} with shape {expected_shape}.")


def _verify_checkpoint_geometry(variant: str, header: dict) -> None:
    _expect_tensor(header, "adaln_t_table", "F32", [1025, 8])
    for name, width in (("video", 96), ("audio", 32)):
        _expect_tensor(header, f"final_layer.{name}_out.weight", "F32", [HEADS * width, 5376])
        _expect_tensor(header, f"final_layer.{name}_out.bias", "F32", [HEADS * width])

    gate_dtype, gate_shape = _GATE_SHAPES[variant]
    qkv_dtype, qkv_shape = {
        "bf16": ("BF16", [21504, 5376]),
        "int8": ("I8", [21504, 5376]),
        "fp8": ("F8_E4M3", [21504, 5376]),
        "nvfp4": ("U8", [21504, 2688]),
    }[variant]
    gate_names = [key for key in header if key.endswith(".attn.to_gate_compress.weight")]
    if len(gate_names) != 50:
        raise ValueError("FastH3 OmniRef requires exactly 50 trained compression-gate projections.")

    for index in range(50):
        gate = f"blocks.{index}.attn.to_gate_compress"
        qkv = f"blocks.{index}.attn.qkv_proj"
        _expect_tensor(header, gate + ".weight", gate_dtype, gate_shape)
        _expect_tensor(header, qkv + ".weight", qkv_dtype, qkv_shape)
        if variant == "int8":
            _expect_tensor(header, gate + ".weight_scale", "F32", [7168, 1])
            _expect_tensor(header, gate + ".comfy_quant", "U8", [72])
            _expect_tensor(header, qkv + ".weight_scale", "F32", [21504, 1])
            _expect_tensor(header, qkv + ".comfy_quant", "U8", [72])
        elif variant == "fp8":
            _expect_tensor(header, gate + ".weight_scale", "F32", [])
            _expect_tensor(header, gate + ".comfy_quant", "U8", [26])
            _expect_tensor(header, qkv + ".weight_scale", "F32", [])
            _expect_tensor(header, qkv + ".comfy_quant", "U8", [26])
        elif variant == "nvfp4":
            for base, rows in ((gate, 7168), (qkv, 21504)):
                _expect_tensor(header, base + ".weight_scale", "F8_E4M3", [rows, 336])
                _expect_tensor(header, base + ".weight_scale_2", "F32", [])
                _expect_tensor(header, base + ".comfy_quant", "U8", [18])
        else:
            for base in (gate, qkv):
                for suffix in ("weight_scale", "weight_scale_2", "comfy_quant"):
                    if base + "." + suffix in header:
                        raise ValueError(f"FastH3 OmniRef BF16 tensor {base} has unexpected quantization metadata.")


def verify_checkpoint(filename: str, header: dict) -> str:
    variant = checkpoint_variant_from_size(filename)
    pin = PINNED_CHECKPOINTS[variant]
    fasth3.verify_checkpoint_pin(filename, pin["size"], pin["sha256"], f"FastH3 OmniRef {pin['label']}")
    _verify_metadata(variant, header)
    _verify_checkpoint_geometry(variant, header)
    return variant


def fine_sigmas(shift: float) -> torch.Tensor:
    if shift not in (12.0, 3.0):
        raise ValueError("FastH3 OmniRef requires video/audio shifts 12/3.")
    base = torch.linspace(GRID_MAX, 0, HEADS + 1, dtype=torch.float64, device="cpu")
    return shift * base * GRID_MAX / (base * (shift - 1) + GRID_MAX)


def sigmas(shift: float) -> tuple[float, ...]:
    return tuple(fine_sigmas(shift)[::4].tolist())


def head_weights(shift: float) -> torch.Tensor:
    # Algebraic sigma differences avoid cancellation near the noisy endpoint.
    if shift not in (12.0, 3.0):
        raise ValueError("FastH3 OmniRef requires video/audio shifts 12/3.")
    base = torch.linspace(GRID_MAX, 0, HEADS + 1, dtype=torch.float64, device="cpu")
    delta = shift * GRID_MAX**2 * (base[1:] - base[:-1]) / (
        (GRID_MAX + (shift - 1) * base[:-1]) * (GRID_MAX + (shift - 1) * base[1:]))
    blocks = delta.view(EVALUATIONS, 4)
    return (blocks / blocks.sum(-1, keepdim=True)).float()


class PDDOutput(nn.Linear):
    """Mmap/offload-compatible widened weights; emit one fused FP32 head.

Keep source names and weight shapes intact for MMGP. ``out_features`` is the
runtime width so the selected-row projector allocates only one head's rows.
Fusion occurs inside forward, after MMGP has made the parameter resident.
"""
    def __init__(self, input_size: int, output_size: int, shift: float):
        super().__init__(input_size, output_size * HEADS, bias=True, dtype=torch.float32)
        self.out_features = output_size
        self._lock_dtype = torch.float32
        self.step = 0
        self._plans = head_weights(shift).tolist()

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if not 0 <= self.step < EVALUATIONS:
            raise ValueError("Invalid FastH3 OmniRef PDD step")
        start = self.step * 4
        weights = self.weight.view(HEADS, self.out_features, self.in_features)
        biases = self.bias.view(HEADS, self.out_features)
        plan = torch.tensor(self._plans[self.step], device=weights.device, dtype=torch.float32)
        # This Comfy conversion stores H0, then (Hj-H0) for j>0, unlike
        # FastVideo's absolute head bank. The weighted mean is always H0
        # plus weighted residuals, with no residual for interval zero.
        if start == 0:
            plan[0] = 0
        fused_weight = weights[0].float() + torch.einsum("n,noi->oi", plan, weights[start:start + 4].float())
        fused_bias = biases[0].float() + torch.einsum("n,no->o", plan, biases[start:start + 4].float())
        return F.linear(rows.float(), fused_weight, fused_bias)


def validate_request(inputs: dict) -> None:
    # The shared validator owns fixed steps, cache, LoRA and window policy.
    # Remove only ordered Ref2VA conditioning while retaining all other checks.
    check = dict(inputs)
    refs = check.pop("minimax_h3_references", None) or []
    from .reference_manifest import validate_reference_manifest
    refs = validate_reference_manifest(refs, require_files=False)
    if any(ref.get("audio_intent") == "drive" for ref in refs):
        raise ValueError("FastH3 OmniRef supports generative reference audio, not an exact target soundtrack.")
    fasth3.validate_fasth3_request(check)
    inputs.update({key: value for key, value in check.items() if key not in inputs or key in (
        "num_inference_steps", "guidance_scale", "flow_shift", "audio_flow_shift", "override_attention",
        "skip_steps_cache_type", "minimax_h3_turbo_mode", "minimax_h3_turbo_preset",
        "minimax_h3_extended_duration", "multi_prompts_gen_type", "sliding_window_size",
        "sliding_window_overlap", "sliding_window_discard_last_frames")})
    inputs["flow_shift"] = 12.0
    inputs["minimax_h3_references"] = refs


def capabilities() -> dict:
    result = fasth3.fasth3_capabilities()
    result.update({"minimax_h3_fasth3_omniref": True, "omni_reference": True,
                   "supports_reference_audio": True, "minimax_h3_video_shift": 12.0,
                   "selector_help": HELP,
                   "inference_steps_help": "FastH3 OmniRef uses eight fixed PDD evaluations and trained VSA-H3 attention."})
    return result
