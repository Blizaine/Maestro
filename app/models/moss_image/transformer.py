# Adapted from OpenMOSS MOSS-Image, Apache-2.0; see LICENSE.
import math
from types import SimpleNamespace

import torch
from torch import nn
from shared import attention_kit
from shared.attention import pay_attention
from . import ops
from .lora_mapping import convert_lora_keys
from .progress import GenerationAborted


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        return ops.rms_norm_rounded([x], self.eps, manual=True) * self.weight


def situ(hidden, gate):
    # MOSS's bounded SiLU gate and value saturation are checkpoint-native.
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _capture_head_slice(capture, index, tensor, group, heads):
    if group.kv.start == 0 and group.kv.stop == heads:
        capture[index] = tensor
        return
    if capture[index] is None:
        capture[index] = tensor.new_empty((*tensor.shape[:2], heads, tensor.shape[-1]))
    capture[index][:, :, group.kv.start:group.kv.stop].copy_(tensor)


def _cache_empty(cache):
    method = getattr(cache, "empty", None)
    if callable(method):
        return bool(method())
    return getattr(cache, "k", None) is None or getattr(cache, "v", None) is None


def attend(x_list, projections, q_norm, k_norm, rope, heads, reference_kv=None, capture=None):
    def normalize(q, k, group):
        for tensor, norm in ((q, q_norm), (k, k_norm)):
            if tensor is not None:
                ops.rows_(tensor, lambda part, first, last: rotate(norm(part), rope[:, first:last]))

    def attention_with_reference(qkv, group):
        if capture is not None:
            _capture_head_slice(capture, 0, qkv[1], group, heads)
            _capture_head_slice(capture, 1, qkv[2], group, heads)
        if reference_kv is not None:
            keys, values = reference_kv
            batch = qkv[0].shape[0]
            if keys.shape[0] == 1 and batch != 1:
                keys = keys.expand(batch, -1, -1, -1)
                values = values.expand(batch, -1, -1, -1)
            keys = keys[:, :, group.kv.start:group.kv.stop]
            values = values[:, :, group.kv.start:group.kv.stop]
            qkv[1] = torch.cat((keys, qkv[1]), dim=1)
            qkv[2] = torch.cat((values, qkv[2]), dim=1)
        return pay_attention(qkv, recycle_q=True)

    custom_attention = attention_with_reference if reference_kv is not None or capture is not None else None
    return attention_kit.qkv_attention(
        x_list,
        *projections,
        heads,
        q_norm.weight.numel(),
        normalize,
        attention_group_fn=custom_attention,
    ).flatten(2)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        x = x + self.proj(attend([self.norm1(x)], (self.q, self.k, self.v), self.q_norm, self.k_norm, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.expansion = 2 * ffn_dim / dim
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')})
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

    def update(self, x, rope, modulation, reference_kv=None, capture=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = ops.norm_scale_shift([x], self.norm1, scale_a[:, None], shift_a[:, None])
        attended = attend(
            [y],
            tuple(self.proj[name] for name in ('q', 'k', 'v')),
            self.q_norm,
            self.k_norm,
            rope,
            self.heads,
            reference_kv=reference_kv,
            capture=capture,
        )
        ops.gated_residual_(x, [self.proj['o'](attended)], gate_a[:, None])
        del attended
        chunk = math.ceil(x.shape[1] / self.expansion) if x.shape[1] > 1024 else x.shape[1]
        for start in range(0, x.shape[1], chunk):
            part = x[:, start:start + chunk]
            y = ops.norm_scale_shift([part], self.norm2, scale_f[:, None], shift_f[:, None])
            hidden, gate = self.proj['ffn_up_gate'](y).chunk(2, -1)
            del y
            activated = situ(hidden, gate)
            del hidden, gate
            ops.gated_residual_(part, [self.proj['ffn_down'](activated)], gate_f[:, None])
            del activated
        return x

    def update_reference(self, reference, rope, modulation, cache_layer=None):
        if cache_layer is not None and not _cache_empty(cache_layer):
            return reference, cache_layer.get()
        if reference is None:
            return None, None
        captured = [None, None]
        reference = self.update(reference, rope, modulation, capture=captured)
        if cache_layer is not None:
            cache_layer.store(*captured)
            captured = list(cache_layer.get())
        return reference, tuple(captured)

    def forward(self, streams, ropes, modulation, skip_branch=-1,
                reference=None, reference_rope=None, reference_modulation=None, cache_layer=None):
        if reference_rope is None:
            for branch in range(len(streams)):
                if branch == skip_branch:
                    continue
                streams[branch] = self.update(streams[branch], ropes[branch], modulation)
            return streams, reference

        if isinstance(reference_rope, list):
            if reference is None:
                reference = [None] * len(streams)
            for branch in range(len(streams)):
                if branch == skip_branch:
                    continue
                branch_cache = None if cache_layer is None else cache_layer[branch]
                reference[branch], prefix = self.update_reference(
                    reference[branch], reference_rope[branch], reference_modulation, branch_cache
                )
                streams[branch] = self.update(streams[branch], ropes[branch], modulation, reference_kv=prefix)
        else:
            reference, prefix = self.update_reference(reference, reference_rope, reference_modulation, cache_layer)
            for branch in range(len(streams)):
                if branch == skip_branch:
                    continue
                streams[branch] = self.update(streams[branch], ropes[branch], modulation, reference_kv=prefix)
        return streams, reference


def rope_embed(height, width, text_length, device, head_dim=128, base=10000.0,
               reference_index=0, image_text_position=None):
    bands = 1 / (base ** (torch.arange(head_dim // 8, device='cpu', dtype=torch.float32) / (head_dim // 8)))
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    image = torch.stack((
        torch.full_like(h, reference_index),
        torch.full_like(h, text_length if image_text_position is None else image_text_position),
        h,
        w,
    ), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands.to(device)
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0):
        super().__init__()
        self.config = SimpleNamespace(dim=dim, freq_dim=freq_dim, num_heads=num_heads, latent_channels=latent_channels, rope_base=rope_base)
        self._interrupt = False
        self.split_linear_modules_map = {'qkv': {'mapped_modules': ['q', 'k', 'v'], 'split_sizes': [dim] * 3}}
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def preprocess_loras(self, model_type, state_dict):
        return convert_lora_keys(state_dict, dict(self.named_modules()), split_linear_modules_map=self.split_linear_modules_map)

    def _time_conditioning(self, timestep):
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=timestep.device, dtype=torch.float64) / half)
        phase = timestep.double().reshape(1, 1) * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        modulation = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(1, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(1, 2, -1)
        return modulation, head

    def _branch_needs_reference(self, cache, branch, skip_layer):
        if cache is None:
            return True
        branch_cache = cache[branch]
        return any(_cache_empty(layer_cache) for index, layer_cache in enumerate(branch_cache)
                   if not (branch == 0 and index == skip_layer))

    def forward(self, latents, timestep, contexts, ropes, skip_layer=22,
                reference_latents=(), reference_rope=None, kv_cache=None):
        modulation, head = self._time_conditioning(timestep)
        video = self.video_embedder(latents.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous())
        tokens = video.shape[1]
        separate_references = isinstance(reference_rope, list)
        reference = None
        reference_modulation = None
        if reference_latents and reference_rope is not None:
            embedded = torch.cat([
                self.video_embedder(
                    z.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous()
                )
                for z in reference_latents
            ], 1)
            if separate_references:
                reference = [
                    embedded.clone() if self._branch_needs_reference(kv_cache, branch, skip_layer) else None
                    for branch in range(len(contexts))
                ]
            else:
                needs_reference = kv_cache is None or any(_cache_empty(item) for item in kv_cache)
                reference = embedded if needs_reference else None
            del embedded
            zero_timestep = torch.zeros_like(timestep)
            reference_modulation, _ = self._time_conditioning(zero_timestep)
        streams = []
        for context, rope in zip(contexts, ropes):
            text = self.text_embedder.proj(context.to(self.text_embedder.proj.weight.dtype))
            for block in self.text_embedder.blocks:
                text = block(text, rope[:, tokens:])
                if self._interrupt:
                    raise GenerationAborted
            streams.append(torch.cat((video, text.expand(video.shape[0], -1, -1)), 1))
        del video, text
        for index, layer in enumerate(self.layers):
            skip_branch = 0 if len(contexts) == 2 and index == skip_layer else -1
            if reference_rope is None:
                streams, reference = layer(streams, ropes, modulation, skip_branch)
            else:
                cache_layer = None
                if kv_cache is not None:
                    cache_layer = [branch[index] for branch in kv_cache] if separate_references else kv_cache[index]
                streams, reference = layer(
                    streams,
                    ropes,
                    modulation,
                    skip_branch,
                    reference,
                    reference_rope,
                    reference_modulation,
                    cache_layer,
                )
            if self._interrupt:
                raise GenerationAborted
        results = []
        for x in streams:
            x = self.final_norm_video(x[:, :tokens].to(self.final_norm_video.weight.dtype))
            x = ops.scale_shift([x], head[:, 1, None], head[:, 0, None])
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            results.append(x.transpose(1, 2).reshape_as(latents).float())
        return results
