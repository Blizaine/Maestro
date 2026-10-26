"""Small, bounded helpers for native MOSS Image editing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from shared.reference_cache import CACHE_MODES, ReferenceKVLayerCache, normalize_cache_mode, quantized_storage_bytes


GIB = 1024 ** 3
MAX_REFERENCE_IMAGES = 4
DEFAULT_REFERENCE_SIZE = 384
DEFAULT_ENCODER_TILE_SIZE = 384
_CACHE_GPU_RESERVE = 2 * GIB
_CACHE_RAM_RESERVE = 2 * GIB


def is_hq_edit_variant(variant):
    """Recognize versioned HQ edit contracts without pinning one model revision."""
    value = str(variant or "").strip().lower()
    return value.startswith("hq_") or value in {"hq", "ai_toolkit"}


def edit_reference_size(model_def=None):
    """Read the model's max reference side, defaulting to a bounded 384 px."""
    model_def = model_def or {}
    value = next((model_def[name] for name in (
        "moss_edit_reference_size", "moss_edit_ref_size", "reference_size"
    ) if model_def.get(name) is not None), DEFAULT_REFERENCE_SIZE)
    if isinstance(value, bool):
        raise ValueError("MOSS Image reference size must be a positive multiple of 16 pixels.")
    try:
        size = int(value)
        if float(value) != size:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("MOSS Image reference size must be a positive multiple of 16 pixels.") from error
    if size < 16 or size % 16:
        raise ValueError("MOSS Image reference size must be a positive multiple of 16 pixels.")
    return size


def validate_reference_images(images, *, editing):
    """Validate PIL references before any VAE or cache allocation."""
    if images is None:
        images = []
    if not isinstance(images, (list, tuple)):
        images = [images]
    if not editing:
        return "MOSS Image text-to-image does not accept reference images." if images else None
    if not 1 <= len(images) <= MAX_REFERENCE_IMAGES:
        return "MOSS Image Edit requires one to four reference images."
    from PIL import Image

    if any(not isinstance(image, Image.Image) for image in images):
        return "MOSS Image Edit references must be PIL images."
    return None


def fit_reference_image(image, max_side=DEFAULT_REFERENCE_SIZE):
    """Resize an RGB PIL image proportionally to a multiple-of-16 canvas.

    The image is never padded or cropped. The only shape adjustment is rounding
    each proportional dimension down to the nearest 16-pixel VAE unit.
    """
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("MOSS Image Edit references must have nonzero dimensions.")
    scale = min(1.0, float(max_side) / max(width, height))
    target_w = max(16, int(math.floor(width * scale / 16)) * 16)
    target_h = max(16, int(math.floor(height * scale / 16)) * 16)
    if (target_w, target_h) == image.size:
        return image
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


def has_step_varying_loras(loras_slists):
    """True when a LoRA multiplier schedule can change between denoise steps."""
    if not isinstance(loras_slists, dict):
        return False
    return any(
        isinstance(value, list)
        for phase in ("phase1", "phase2", "phase3")
        for value in (loras_slists.get(phase) or [])
    )


def _cache_storage_bytes(mode, tokens, heads, head_dim, layers, branches, itemsize):
    # Per-layer key and value, one reference sequence per independent CFG arm.
    if mode in ("INT8", "INT8_RAM"):
        one = quantized_storage_bytes((1, tokens, heads, head_dim))
        return 2 * one * layers * branches
    return 2 * tokens * heads * head_dim * layers * branches * itemsize


@dataclass(frozen=True)
class ReferenceCachePlan:
    requested_mode: str
    actual_mode: str
    reason: str
    gpu_required_bytes: int
    gpu_budget_bytes: int
    ram_required_bytes: int
    ram_budget_bytes: int


def _model_non_cuda_storage_bytes(model):
    total = 0
    for tensor in (*model.parameters(), *model.buffers()):
        getter = getattr(tensor, "get_quantized_subtensors", None)
        parts = getter() if getter is not None else None
        if isinstance(parts, dict):
            parts = parts.items()
        tensors = [part for _, part in parts if torch.is_tensor(part)] if parts else [tensor]
        total += sum(
            part.numel() * part.element_size()
            for part in tensors
            if part.device.type != "cuda"
        )
    return total


def _memory_budgets(device, transformer=None):
    device = torch.device(device)
    try:
        import psutil

        ram_available = int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError):
        ram_available = 0
    ram_budget = max(0, ram_available - max(_CACHE_RAM_RESERVE, int(ram_available * 0.10)))
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info(device)
            reusable = max(0, int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device)))
            cpu_model_bytes = _model_non_cuda_storage_bytes(transformer) if transformer is not None else 0
            return max(0, int(free) + reusable - cpu_model_bytes), ram_budget
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return 0, ram_budget


def plan_reference_cache(
    requested_mode,
    *,
    device,
    layers,
    dim,
    heads,
    reference_tokens,
    body_tokens,
    branches,
    batch_size=1,
    itemsize=2,
    dynamic_loras=False,
    denoising_steps=2,
    transformer=None,
):
    """Choose cache storage only when a conservative GPU/RAM estimate fits."""
    mode = normalize_cache_mode(requested_mode)
    if mode == "Disabled":
        return ReferenceCachePlan(mode, "Disabled", "disabled by setting", 0, 0, 0, 0)
    if dynamic_loras:
        return ReferenceCachePlan(mode, "Disabled", "step-varying LoRA multipliers require fresh reference projections", 0, 0, 0, 0)
    if int(denoising_steps) <= 1:
        return ReferenceCachePlan(mode, "Disabled", "reference caching needs at least two denoising steps", 0, 0, 0, 0)
    gpu_capacity, ram_budget = _memory_budgets(device, transformer)

    storage = _cache_storage_bytes(mode, reference_tokens, heads, dim // heads, layers, branches, itemsize)
    # Account for non-CUDA DiT storage in _memory_budgets and leave head-group
    # Q/K/V, MLP, RoPE and allocator slack for the active transformer pass.
    # Batch scales body scratch; cached prefixes remain one reference per arm.
    workspace = max(
        GIB,
        (reference_tokens + body_tokens * batch_size) * dim * itemsize * 24,
    ) + GIB
    gpu_budget = max(0, gpu_capacity - workspace)
    prefix_join = 2 * (reference_tokens + body_tokens * batch_size) * dim * itemsize
    dequantized_prefix = 2 * reference_tokens * dim * itemsize if mode in ("INT8", "INT8_RAM") else 0
    gpu_required = prefix_join + dequantized_prefix + (storage if mode != "INT8_RAM" else 0)
    ram_required = storage if mode == "INT8_RAM" else 0
    if torch.device(device).type == "cpu":
        ram_required = storage + prefix_join + dequantized_prefix + workspace
        gpu_required = 0
        gpu_budget = 0
    if gpu_required > gpu_budget:
        reason = (f"estimated GPU need {gpu_required / GIB:.2f} GiB exceeds safe budget "
                  f"{gpu_budget / GIB:.2f} GiB")
        return ReferenceCachePlan(mode, "Disabled", reason, gpu_required, gpu_budget, ram_required, ram_budget)
    if ram_required > ram_budget:
        reason = (f"estimated RAM need {ram_required / GIB:.2f} GiB exceeds safe budget "
                  f"{ram_budget / GIB:.2f} GiB")
        return ReferenceCachePlan(mode, "Disabled", reason, gpu_required, gpu_budget, ram_required, ram_budget)
    return ReferenceCachePlan(mode, mode, "memory estimate fits", gpu_required, gpu_budget, ram_required, ram_budget)


def create_reference_kv_cache(layer_count, branch_count, mode, *, separate_branches):
    """Create request-local caches; HQ owns one independent list per CFG arm."""
    if mode == "Disabled":
        return None
    if mode not in CACHE_MODES:
        raise ValueError(f"Unknown reference KV cache mode: {mode!r}")
    make_branch = lambda: [ReferenceKVLayerCache(mode) for _ in range(layer_count)]
    if separate_branches:
        return [make_branch() for _ in range(branch_count)]
    return make_branch()


def cache_mode_log(plan):
    requested = plan.requested_mode
    actual = plan.actual_mode
    if actual == "Disabled" and requested != "Disabled":
        return f"[MOSS Image Edit] Reference KV cache: Disabled (requested {requested}; {plan.reason})."
    return (f"[MOSS Image Edit] Reference KV cache: {actual} ({plan.reason}; "
            f"GPU estimate {plan.gpu_required_bytes / GIB:.2f} GiB; "
            f"safe budget {plan.gpu_budget_bytes / GIB:.2f} GiB; "
            f"RAM estimate {plan.ram_required_bytes / GIB:.2f} GiB; "
            f"safe budget {plan.ram_budget_bytes / GIB:.2f} GiB).")
