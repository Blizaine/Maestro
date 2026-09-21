"""Convert AI-Toolkit's combined, fused YuE2 LoRA without merging base weights."""
from pathlib import Path
import math

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .artist_adapter import validate_tensors


def is_combined_adapter(path):
    if Path(path).suffix.lower() != ".safetensors":
        return False
    with safe_open(str(path), framework="pt", device="cpu") as source:
        return any(key.startswith(("diffusion_model.", "text_encoders.")) for key in source.keys())


def convert_combined(tensors):
    """Split B's output rows, retaining A for each projection; alpha/rank once.

    Missing alpha means the file's stored delta is B @ A, as in AI-Toolkit's
    diffusers/Comfy export. Unknown targets are rejected rather than discarded.
    """
    remaining = set(tensors)
    branches = {}
    layout = (
        ("self_attn.qkv_proj", (("q_proj", 2048), ("k_proj", 1024), ("v_proj", 1024)), 2048),
        ("self_attn.o_proj", (("o_proj", 2048),), 2048),
        ("mlp.gate_up_proj", (("gate_proj", 6144), ("up_proj", 6144)), 2048),
        ("mlp.down_proj", (("down_proj", 2048),), 6144),
    )
    for branch, prefix in (("ar", "text_encoders"), ("nar", "diffusion_model")):
        result = {}
        for layer in range(28):
            for source_name, outputs, input_size in layout:
                name = f"{prefix}.model.layers.{layer}.{source_name}"
                a_key, b_key = name + ".lora_A.weight", name + ".lora_B.weight"
                a, b = tensors.get(a_key), tensors.get(b_key)
                if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                        or a.ndim != 2 or b.ndim != 2 or a.shape[1] != input_size
                        or b.shape != (sum(size for _, size in outputs), a.shape[0])):
                    raise ValueError(f"Incomplete or unsupported combined YuE2 adapter at {name}")
                if not a.is_floating_point() or not b.is_floating_point() or not 1 <= a.shape[0] <= 128:
                    raise ValueError("Combined YuE2 adapters need floating-point LoRA weights of rank 1–128")
                remaining.difference_update((a_key, b_key))
                alpha = tensors.get(name + ".alpha")
                factor = 1.0
                if alpha is not None:
                    if not isinstance(alpha, torch.Tensor) or alpha.numel() != 1:
                        raise ValueError("Invalid YuE2 adapter alpha")
                    factor = float(alpha) / a.shape[0]
                    if not math.isfinite(factor) or factor <= 0:
                        raise ValueError("Invalid YuE2 adapter alpha")
                    remaining.remove(name + ".alpha")
                group = source_name.split(".")[0]
                if branch == "nar":
                    group = "nar_" + group
                offset = 0
                for projection, size in outputs:
                    target = f"model.layers.{layer}.{group}.{projection}"
                    # Copies also remove safetensors shared-storage ambiguity.
                    result[target + ".A"] = a.clone()
                    result[target + ".B"] = (b[offset:offset + size].float() * factor).contiguous()
                    offset += size
        validate_tensors(result, branch, require_io=False)
        branches[branch] = result
    if remaining:
        raise ValueError("Unsupported combined YuE2 targets: " + ", ".join(sorted(remaining)[:3]))
    return branches


def read_combined_adapter(path):
    return convert_combined(load_file(str(path), device="cpu"))
