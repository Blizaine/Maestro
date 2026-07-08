"""
LoRA architecture probing — header inspection helpers.

DESIGN NOTE — DO NOT USE FOR FILTERING:
A previous version of this module exposed `filter_compatible_loras()`
which the Director image-gen pipeline used to drop LoRAs that "looked
incompatible" with the target model based on safetensors attention-tensor
dims. The implementation was based on a wrong assumption — that Flux 2
Klein 9B used a 3072 hidden dim — and the resulting heuristic
false-positived on legitimate Klein 9B–trained LoRAs because Klein 9B
actually uses the SAME 4096 hidden / 12288 QKV / 24576 MLP-inner dims as
Flux 2 Pro/Dev. The "9B" in Klein 9B refers to fewer transformer
**layers**, not a narrower hidden dim. LoRAs are dim-compatible across
all Flux 2 variants; the only thing that varies between checkpoints is
depth, which LoRAs don't care about.

The faulty filter was removed from director_pipeline.py and from the
CivitAI download verification flow. Real corruption (the original
failure mode the filter was groping at, where a brotli-encoded error
page got saved as a `.safetensors` file) is caught by the file-integrity
check in `_run_civitai_download`: size > 100 KB plus a parseable
safetensors header — that gate doesn't need per-model dim knowledge.

The header-peek helpers below are kept because they're occasionally
useful for diagnostics and could underpin future architecture-aware
features that aren't filter-blocking. They DO NOT decide whether a
LoRA loads — that's mmgp's job, and mmgp will surface a clear error
if a LoRA is genuinely incompatible (which is rare across the Flux 2
variants we support).
"""
from __future__ import annotations

import os
from typing import Optional


def _peek_lora_attn_dims(lora_path: str) -> set[int]:
    """Return the set of dims found in attention-related tensors.

    Reads the safetensors header only — fast and memory-cheap. Returns
    an empty set if the file is unreadable or has no tensors with
    recognizable attention/QKV-style key fragments.

    Diagnostic-only — see module docstring.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return set()

    if not os.path.isfile(lora_path):
        return set()

    _ATTN_FRAGMENTS = (
        "qkv", "to_qkv", "add_qkv",
        "to_q", "to_k", "to_v",
        "add_q_proj", "add_k_proj", "add_v_proj",
        "attn",
    )

    dims: set[int] = set()
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lower = key.lower()
                if not any(frag in lower for frag in _ATTN_FRAGMENTS):
                    continue
                slice_handle = f.get_slice(key)
                shape = list(slice_handle.get_shape())
                for s in shape:
                    dims.add(s)
    except Exception:
        return set()

    return dims


def expected_qkv_dim_for_model(model_type: str) -> Optional[int]:
    """Best-effort estimate of a model's combined QKV output dim.

    DO NOT use the return value for go/no-go LoRA filtering. Most Flux 2
    variants share a 12288 QKV dim, so this rarely discriminates. Kept
    only because some diagnostic logs reference it.
    """
    if not model_type:
        return None
    mt = model_type.lower()
    # All Flux 2 variants observed in the wild — Pro, Dev, Klein 9B,
    # Klein 4B — use 4096 hidden / 12288 combined QKV at the LoRA-visible
    # layers. The "9B" / "4B" labels reflect total parameter counts
    # driven by depth, not hidden width.
    if "flux2" in mt or "klein" in mt or "flux_2" in mt or "flux-2" in mt:
        return 12288
    # Flux 1 family used 3072 hidden / 9216 QKV
    if mt.startswith("flux"):
        return 9216
    return None


def infer_architecture_label(lora_path: str) -> Optional[str]:
    """Best-effort label for a LoRA's training target, by peek dim.

    DO NOT use to gate loading. Returns a human-readable string for
    diagnostic surfaces only.
    """
    dims = _peek_lora_attn_dims(lora_path)
    if not dims:
        return None
    if 12288 in dims or 4096 in dims:
        return "Flux 2 family (4096 hidden / 12288 QKV)"
    if 9216 in dims or 3072 in dims:
        return "Flux 1 family (3072 hidden / 9216 QKV)"
    return None
