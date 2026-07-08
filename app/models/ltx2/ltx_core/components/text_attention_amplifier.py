"""LTX Text Attention Amplifier — port of TenStrip 10S-Comfy-nodes
LTXTextAttentionAmplifier (https://github.com/tenstrip/10S-Comfy-nodes/blob/main/latent_text_amplifier.py).

Hooks each transformer block's `attn2` (text cross-attention) module and
multiplies its output by an amplification factor. Compensates for the
conditioning dilution the 10Eros author reports at upscaled token counts —
prompt adherence and color stability improve when the text cross-attention
output is boosted by ~30% in a specific late-attention layer range
(blocks 36-48 in the 10Eros reference workflow).

Driven by a model_def `text_attention_amplifier` block:

    "text_attention_amplifier": {
        "strength": 1.3,
        "block_filter": "36-48",
        "spatial_focus": 0.15
    }

When `strength` is 1.0 (or the block is absent), no hooks are installed
and the model runs untouched. The hooks are installed for the lifetime
of one generation call and uninstalled in a `finally` so a failed
generation doesn't leave the model permanently patched.

Spatial focus is optional and matches the upstream node's behavior:
0.0 → uniform amplification of all text-tokens, >0 → Gaussian-weighted
amplification centered on the latent's spatial midpoint (tighter for
higher values). Defaults to 0.0 (uniform) for safety; the 10Eros
workflow uses 0.15.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import torch


def parse_block_filter(spec: Optional[str], n_blocks: int) -> Optional[frozenset[int]]:
    """Parse a block-index filter string like '36-48' or '0,3,5-10'.

    Returns a frozenset of indices, or None if the spec is empty/blank
    (meaning: hook ALL blocks).
    """
    if not spec or not spec.strip():
        return None
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a_s, b_s = part.split("-", 1)
                a_i, b_i = int(a_s.strip()), int(b_s.strip())
                lo, hi = min(a_i, b_i), max(a_i, b_i)
                indices.update(range(max(0, lo), min(n_blocks - 1, hi) + 1))
            except Exception:
                continue
        else:
            try:
                idx = int(part)
                if 0 <= idx < n_blocks:
                    indices.add(idx)
            except Exception:
                continue
    return frozenset(indices) if indices else None


def _build_spatial_weight(
    h_tok: int, w_tok: int, spatial_focus: float, dtype: torch.dtype, device: torch.device
) -> Optional[torch.Tensor]:
    """Gaussian per-spatial-position weight in [0, 1].

    spatial_focus=0 → returns None (caller uses uniform amplification).
    spatial_focus=0.5 → sigma = 0.65 * min(h,w) (broad).
    spatial_focus=1.0 → sigma = 0.30 * min(h,w) (tight).
    """
    if spatial_focus <= 0.0:
        return None
    smaller_dim = min(h_tok, w_tok)
    sigma = max(0.3, 1.0 - 0.7 * spatial_focus) * smaller_dim
    cy = (h_tok - 1) / 2.0
    cx = (w_tok - 1) / 2.0
    y_idx = torch.arange(h_tok, dtype=torch.float32, device=device)
    x_idx = torch.arange(w_tok, dtype=torch.float32, device=device)
    dy = y_idx - cy
    dx = x_idx - cx
    dist_sq = dy.unsqueeze(1).pow(2) + dx.unsqueeze(0).pow(2)
    gaussian = torch.exp(-dist_sq / (2.0 * sigma * sigma))
    g_min = gaussian.min()
    g_max = gaussian.max()
    gaussian = (gaussian - g_min) / (g_max - g_min + 1e-6)
    return gaussian.to(dtype=dtype)


def install_text_amplifier(
    transformer: torch.nn.Module,
    strength: float = 1.3,
    block_filter: Optional[str] = None,
    spatial_focus: float = 0.0,
    latent_shape: Optional[tuple[int, int, int]] = None,
    debug: bool = False,
) -> list:
    """Install forward hooks on transformer cross-attention modules.

    Returns a list of hook handles to be removed via remove_text_amplifier().
    Returns empty list if strength == 1.0 (no-op) or the transformer
    has no `transformer_blocks` attribute.

    Args:
        transformer: The X0Model / transformer module whose blocks should
            be patched. Must expose `.transformer_blocks` (an iterable of
            modules each having a `.attn2` submodule).
        strength: Multiplier for the cross-attention output.
            1.0 = no-op. Workflow default for 10Eros: 1.3.
        block_filter: Block-index filter string ("36-48" / "0,3,5-10").
            None / "" → hook all blocks. Workflow default: "36-48".
        spatial_focus: 0.0 = uniform; >0.0 = Gaussian-weighted center boost.
        latent_shape: Optional (F_tok, H_tok, W_tok) for spatial focus.
            If None, spatial focus is disabled (uniform mode used as a
            fallback even when spatial_focus > 0).
        debug: When True, print per-call diagnostics on first fire.
    """
    if strength == 1.0:
        return []
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        if debug:
            print("[10S TextAmp] transformer has no `transformer_blocks`; skipping")
        return []

    n_blocks = len(blocks)
    idx_filter = parse_block_filter(block_filter, n_blocks)

    # Per-hook closure state — kept in a list so all hooks share it.
    state = {
        "first_fire_logged": False,
        "spatial_weight_cache": None,
        "call_count": 0,
    }

    def make_hook(block_idx: int):
        def hook(_module, _inputs, output):
            try:
                # Maestro's Attention.forward returns a single tensor.
                # ComfyUI's version had to handle (tensor, ...) tuples,
                # dict outputs, etc. Our path is simpler.
                if not torch.is_tensor(output):
                    return None
                tensor = output
                if tensor.dim() != 3:
                    return None
                b, seq, d = tensor.shape

                # Uniform path: applies whenever spatial_focus is off OR
                # we don't have a latent shape to compute the spatial grid.
                if spatial_focus <= 0.0 or latent_shape is None:
                    if debug and not state["first_fire_logged"]:
                        print(
                            f"[10S TextAmp] HOOK ACTIVE | first fire on blk{block_idx} | "
                            f"seq={seq} D={d} amp={strength} mode=uniform"
                        )
                        state["first_fire_logged"] = True
                    state["call_count"] += 1
                    return tensor * strength

                # Spatial path: weight amplification by distance from
                # the latent's spatial center. Requires latent shape.
                f_tok, h_tok, w_tok = latent_shape
                if f_tok * h_tok * w_tok != seq:
                    # Shape mismatch (e.g. patch size != 1). Fall back to uniform.
                    state["call_count"] += 1
                    return tensor * strength

                if state["spatial_weight_cache"] is None or state["spatial_weight_cache"].shape != (h_tok, w_tok):
                    state["spatial_weight_cache"] = _build_spatial_weight(
                        h_tok, w_tok, spatial_focus, dtype=tensor.dtype, device=tensor.device
                    )
                spatial_weight = state["spatial_weight_cache"]
                if spatial_weight is None:
                    return tensor * strength

                # Per-token amp: center = full amp, edges = no amp.
                amp_grid = 1.0 + (strength - 1.0) * spatial_weight  # (H, W)
                amp_full = amp_grid.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(b, f_tok, h_tok, w_tok, 1)
                grid = tensor.reshape(b, f_tok, h_tok, w_tok, d)
                modified = (grid * amp_full).reshape(b, seq, d)

                if debug and not state["first_fire_logged"]:
                    print(
                        f"[10S TextAmp] HOOK ACTIVE | first fire on blk{block_idx} | "
                        f"grid=(F={f_tok},H={h_tok},W={w_tok}) seq={seq} D={d} "
                        f"amp={strength} mode=spatial focus={spatial_focus}"
                    )
                    state["first_fire_logged"] = True
                state["call_count"] += 1
                return modified
            except Exception as e:
                if debug:
                    print(f"[10S TextAmp] blk{block_idx} hook error: {type(e).__name__}: {e}")
                return None
        return hook

    handles: list = []
    hooked = 0
    skipped = 0
    missing = 0
    for i, block in enumerate(blocks):
        if idx_filter is not None and i not in idx_filter:
            skipped += 1
            continue
        attn2 = getattr(block, "attn2", None)
        if attn2 is None:
            missing += 1
            continue
        try:
            handle = attn2.register_forward_hook(make_hook(i))
            handles.append(handle)
            hooked += 1
        except Exception as e:
            missing += 1
            if debug:
                print(f"[10S TextAmp] blk{i}.attn2 hook failed: {type(e).__name__}: {e}")

    mode_str = "uniform" if spatial_focus <= 0 or latent_shape is None else f"spatial(focus={spatial_focus})"
    print(
        f"[10S TextAmp] {hooked}/{n_blocks} blocks hooked "
        f"(skipped={skipped}, missing={missing}) | strength={strength} mode={mode_str}"
    )
    return handles


def remove_text_amplifier(handles: Iterable) -> int:
    """Remove all hook handles. Returns the count actually removed."""
    removed = 0
    for h in handles:
        try:
            h.remove()
            removed += 1
        except Exception:
            pass
    return removed
