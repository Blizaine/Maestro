"""Adapt Wan2GP's Diffusers-format VDN adapters to Maestro's H3 modules.

Namespace, SwiGLU ordering and fused-QKV conversion adapted from Wan2GP
1e1dd2757f24923f008593d9d4ec09062234be20; WanGP Community License 2.0.
"""
import torch


_MMGP_LORA_ROOT_PREFIXES = ("diffusion_model.", "transformer.")


def _strip_mmgp_lora_root_prefixes(state_dict):
    """Canonicalize MMGP's optional per-key root wrappers without data loss."""

    normalized = {}
    source_keys = {}
    for source_key, value in state_dict.items():
        key = source_key
        for prefix in _MMGP_LORA_ROOT_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key in normalized:
            raise ValueError(
                "MiniMax H3 LoRA key collision after removing MMGP root prefixes: "
                f"{source_keys[key]!r} and {source_key!r} both map to {key!r}."
            )
        normalized[key] = value
        source_keys[key] = source_key
    return normalized, source_keys


def interleave_qkv_lora_b_rows(
    weight: torch.Tensor,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reorder a known grouped fused LoRA B matrix to H3 head order.

    Independent Q/K/V adapters are first fused with grouped output rows. Their
    B matrix's row axis can then be permuted to ``[head, QKV, channel]`` while
    retaining the rank-block columns and therefore the exact adapter delta.
    """

    expected_rows = int(heads) * 3 * int(head_dim)
    if weight.ndim != 2 or int(weight.shape[0]) != expected_rows:
        raise ValueError(
            "Fused H3 LoRA B rows do not match the target attention shape: "
            f"got {tuple(weight.shape)}, expected {expected_rows} rows."
        )
    return (
        weight.reshape(3, int(heads), int(head_dim), weight.shape[1])
        .permute(1, 0, 2, 3)
        .reshape_as(weight)
        .contiguous()
    )


def normalize_diffusers_lora(state_dict, transformer):
    prefixes = (("token_refiner.refiner_blocks.", "token_refiner.blocks."),
        ("transformer_blocks.", "blocks."),
        ("time_embedder.linear_1.", "time_embedder.proj_in."),
        ("time_embedder.linear_2.", "time_embedder.proj_out."),
        ("audio_proj_in.", "audio_patch_proj."), ("proj_in.", "video_patch_proj."),
        ("context_embedder.", "condition_proj."), ("norm_out.norm.", "final_layer.norm."),
        ("norm_out.linear.", "final_layer.adaln_proj.linear."),
        ("audio_proj_out.", "final_layer.audio_out."), ("proj_out.", "final_layer.video_out."))
    state_dict, source_keys = _strip_mmgp_lora_root_prefixes(state_dict)
    has_diffusers_prefix = any(
        key.startswith(tuple(source for source, _ in prefixes))
        for key in state_dict
    )
    # Already-local split Q/K/V keys still need fusing when the base checkpoint
    # keeps its QKV projection fused (for example, scaled-FP8 legacy H3).
    converted = {}
    converted_sources = {}
    for key, value in state_dict.items():
        source_key = source_keys[key]
        if has_diffusers_prefix:
            key = key.replace(".lora_A.turbo.weight", ".lora_A.weight").replace(".lora_B.turbo.weight", ".lora_B.weight")
            fc1 = ".ff.net.0.proj." in key
            for source, target in prefixes:
                if key.startswith(source):
                    key = target + key[len(source):]
                    break
            for source, target in ((".attn.norm_q.", ".attn.q_norm."), (".attn.norm_k.", ".attn.k_norm."),
                (".attn.orig.to_out.0.", ".attn.out_proj."), (".attn.orig.to_q.", ".attn.q_proj."),
                (".attn.orig.to_k.", ".attn.k_proj."), (".attn.orig.to_v.", ".attn.v_proj."),
                (".attn.to_out.0.", ".attn.out_proj."), (".attn.to_q.", ".attn.q_proj."),
                (".attn.to_k.", ".attn.k_proj."), (".attn.to_v.", ".attn.v_proj."),
                (".ff.net.0.proj.", ".mlp.fc1."), (".ff.net.2.", ".mlp.fc2.")):
                key = key.replace(source, target)
            for branch in ("linear_attention", "softmax_gate", "to_out_linear"):
                key = key.replace(f".attn.{branch}.", f".attn.vdn.{branch}.")
            if fc1 and any(suffix in key for suffix in (".lora_B.", ".lora_up.", ".lora.B.", ".lora.up.")):
                value = torch.cat(value.chunk(2, dim=0)[::-1], dim=0).contiguous()
        if has_diffusers_prefix and transformer.use_adaln_curves and key.startswith("time_embedder."):
            continue
        if key in converted:
            raise ValueError(
                "MiniMax H3 LoRA key collision after name normalization: "
                f"{converted_sources[key]!r} and {source_key!r} both map to {key!r}."
            )
        converted[key] = value
        converted_sources[key] = source_key
    for down_suffix, up_suffix in (("lora_A.weight", "lora_B.weight"),
        ("lora_A.default.weight", "lora_B.default.weight"), ("lora_down.weight", "lora_up.weight"),
        ("lora_down.default.weight", "lora_up.default.weight"), ("lora.A.weight", "lora.B.weight"),
        ("lora.A.default.weight", "lora.B.default.weight"), ("lora.down.weight", "lora.up.weight"),
        ("lora.down.default.weight", "lora.up.default.weight")):
        marker = "q_proj." + down_suffix
        for key in [key for key in converted if key.endswith(marker)]:
            prefix = key[:-len(marker)]
            required = [
                prefix + projection + suffix
                for projection in ("q_proj.", "k_proj.", "v_proj.")
                for suffix in (down_suffix, up_suffix)
            ]
            if any(required_key not in converted for required_key in required):
                # A partial independent-Q/K/V adapter cannot be represented
                # by this exact rank-3R fused conversion. Leave it intact for
                # MMGP's normal compatibility error instead of guessing.
                continue
            attention_path = prefix.rstrip(".")
            try:
                attention = transformer.get_submodule(attention_path)
            except (AttributeError, KeyError) as error:
                raise ValueError(
                    "Cannot resolve MiniMax H3 QKV LoRA attention target "
                    f"{attention_path!r} after removing MMGP root prefixes."
                ) from error
            if hasattr(attention, "q_proj"):
                continue
            if not hasattr(attention, "qkv_proj"):
                raise ValueError(
                    "MiniMax H3 QKV LoRA target has neither split projections nor "
                    f"a fused qkv_proj: {attention_path!r}."
                )
            down, up = [], []
            for projection in ("q_proj", "k_proj", "v_proj"):
                down.append(converted.pop(prefix + projection + "." + down_suffix))
                weight = converted.pop(prefix + projection + "." + up_suffix)
                alpha = converted.pop(prefix + projection + ".alpha", None)
                up.append(weight if alpha is None else weight * (float(alpha) / down[-1].shape[0]))
            fused_down_key = prefix + "qkv_proj." + down_suffix
            fused_up_key = prefix + "qkv_proj." + up_suffix
            if fused_down_key in converted or fused_up_key in converted:
                raise ValueError(
                    "MiniMax H3 LoRA key collision while fusing independent Q/K/V "
                    f"adapters at {attention_path!r}."
                )
            converted[fused_down_key] = torch.cat(down)
            fused_up = torch.block_diag(*up)
            layout = str(getattr(attention, "qkv_layout", "grouped"))
            layout = layout.strip().lower().replace("-", "_")
            if layout in {"interleaved", "head_interleaved"}:
                fused_up = interleave_qkv_lora_b_rows(
                    fused_up,
                    attention.heads,
                    attention.head_dim,
                )
            elif layout not in {"grouped", "contiguous"}:
                raise ValueError(
                    f"Unsupported MiniMax H3 QKV LoRA target layout {layout!r}"
                )
            converted[fused_up_key] = fused_up
    return converted


__all__ = ["interleave_qkv_lora_b_rows", "normalize_diffusers_lora"]
