"""Adapt Wan2GP's Diffusers-format VDN adapters to Maestro's H3 modules.

Namespace, SwiGLU ordering and fused-QKV conversion adapted from Wan2GP
1e1dd2757f24923f008593d9d4ec09062234be20; WanGP Community License 2.0.
"""
import torch


def normalize_diffusers_lora(state_dict, transformer):
    prefixes = (("token_refiner.refiner_blocks.", "token_refiner.blocks."),
        ("transformer_blocks.", "blocks."),
        ("time_embedder.linear_1.", "time_embedder.proj_in."),
        ("time_embedder.linear_2.", "time_embedder.proj_out."),
        ("audio_proj_in.", "audio_patch_proj."), ("proj_in.", "video_patch_proj."),
        ("context_embedder.", "condition_proj."), ("norm_out.norm.", "final_layer.norm."),
        ("norm_out.linear.", "final_layer.adaln_proj.linear."),
        ("audio_proj_out.", "final_layer.audio_out."), ("proj_out.", "final_layer.video_out."))
    if not any(key.startswith(tuple(source for source, _ in prefixes)) for key in state_dict):
        return dict(state_dict)
    converted = {}
    for key, value in state_dict.items():
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
        if transformer.use_adaln_curves and key.startswith("time_embedder."):
            continue
        converted[key] = value
    for down_suffix, up_suffix in (("lora_A.weight", "lora_B.weight"),
        ("lora_A.default.weight", "lora_B.default.weight"), ("lora_down.weight", "lora_up.weight"),
        ("lora_down.default.weight", "lora_up.default.weight"), ("lora.A.weight", "lora.B.weight"),
        ("lora.A.default.weight", "lora.B.default.weight"), ("lora.down.weight", "lora.up.weight"),
        ("lora.down.default.weight", "lora.up.default.weight")):
        marker = "q_proj." + down_suffix
        for key in [key for key in converted if key.endswith(marker)]:
            prefix = key[:-len(marker)]
            attention = transformer.get_submodule(prefix.rstrip("."))
            if hasattr(attention, "q_proj"):
                continue
            down, up = [], []
            for projection in ("q_proj", "k_proj", "v_proj"):
                down.append(converted.pop(prefix + projection + "." + down_suffix))
                weight = converted.pop(prefix + projection + "." + up_suffix)
                alpha = converted.pop(prefix + projection + ".alpha", None)
                up.append(weight if alpha is None else weight * (float(alpha) / down[-1].shape[0]))
            converted[prefix + "qkv_proj." + down_suffix] = torch.cat(down)
            converted[prefix + "qkv_proj." + up_suffix] = torch.block_diag(*up)
    return converted
