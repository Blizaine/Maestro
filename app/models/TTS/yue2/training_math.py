"""Differentiable equivalents of YuE2's consuming/in-place playback paths.

Keep playback's tensor layouts, causal AR prefix, NAR prefix attention and
normalization. Checkpointing trades recomputation for bounded training memory.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def ar_norm(x, module):
    # EngineRMSNorm computes its statistics and weight multiplication in FP32.
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + module.eps)
            * module.weight).to(x.dtype)


def nar_norm(x, module):
    return F.rms_norm(x, (x.shape[-1],), module.weight, module.eps)


def attention(q, k, v, causal=False):
    groups = q.shape[-2] // k.shape[-2]
    k, v = (item.repeat_interleave(groups, dim=-2) for item in (k, v))
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2), is_causal=causal).transpose(1, 2)


def ar_layer(layer, x):
    attn = layer.self_attn
    h = ar_norm(x, layer.input_layernorm)
    q = ar_norm(attn.q_proj(h).view(-1, attn.num_heads, attn.head_dim), attn.q_norm)
    k = ar_norm(attn.k_proj(h).view(-1, attn.num_kv_heads, attn.head_dim), attn.k_norm)
    v = attn.v_proj(h).view(-1, attn.num_kv_heads, attn.head_dim)
    q, k = attn.rotary_emb(torch.arange(len(x), device=x.device), q, k)
    x = x + attn.o_proj(attention(q[None], k[None], v[None], True).flatten(2)[0])
    h = ar_norm(x, layer.post_attention_layernorm)
    x = x + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(h)) * layer.mlp.up_proj(h))
    return x, k, v


@torch.no_grad()
def ar_condition(model, ids, cancelled=lambda: False):
    x = model.model.embed_tokens(torch.as_tensor(ids, device=model.model.embed_tokens.weight.device))
    cache = []
    for layer in model.model.layers:
        if cancelled():
            raise InterruptedError("Audio adaptation cancelled")
        x, k, v = ar_layer(layer, x)
        cache.append((k, v))
    return cache


def rotate(x, cos, sin):
    first, second = x.chunk(2, -1)
    cos, sin = cos.to(x.dtype).unsqueeze(2), sin.to(x.dtype).unsqueeze(2)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), -1)


def nar_layer(layer, x, k_prefix, v_prefix, cos, sin):
    attn = layer.nar_self_attn
    h = nar_norm(x, layer.nar_input_layernorm)
    shape = (*h.shape[:2], -1, attn.head_dim)
    q = rotate(nar_norm(attn.q_proj(h).view(shape), attn.q_norm), cos, sin)
    k = rotate(nar_norm(attn.k_proj(h).view(shape), attn.k_norm), cos, sin)
    v = attn.v_proj(h).view(shape)
    k, v = torch.cat((k_prefix[None], k), 1), torch.cat((v_prefix[None], v), 1)
    x = x + attn.o_proj(attention(q, k, v).flatten(2))
    h = nar_norm(x, layer.nar_pre_mlp_layernorm)
    mlp = layer.nar_mlp
    return x + mlp.down_proj(F.silu(mlp.gate_proj(h)) * mlp.up_proj(h))


def acoustic_forward(model, state, raw_t, cache, ar_length, *, gradients=False, cancelled=lambda: False):
    length = len(state) + 2
    dtype = model.time_embedder.mlp[0].weight.dtype
    t = torch.sigmoid(raw_t.to(device=state.device, dtype=dtype))
    shift = model.config.timestep_shift
    t = shift * t / (1 + (shift - 1) * t)
    cos, sin = model.rotary(torch.arange(ar_length, ar_length + length, device=state.device)[None])
    pos = model.latent_pos_embed(torch.arange(length, device=state.device))[None]
    x = model.vae2llm(F.pad(state, (0, 0, 1, 1))[None].to(model.vae2llm.weight.dtype)).to(dtype)
    x = x + model.time_embedder(t.reshape(1))[None] + pos
    if len(cache) != len(model.model.layers):
        raise ValueError("Acoustic conditioning must contain every AR layer")
    for layer, (k, v) in zip(model.model.layers, cache):
        if cancelled():
            raise InterruptedError("Audio adaptation cancelled")
        args = layer, x, k, v, cos, sin
        x = checkpoint(nar_layer, *args, use_reentrant=False) if gradients else nar_layer(*args)
    return model.llm2vae(nar_norm(x, model.model.norm).to(model.llm2vae.weight.dtype))[0, 1:-1]
