# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fused Hy-Nomas passes, FP32 arithmetic in registers:
- the q / k norms and MRoPE of a PixelUMM decoder head in one pass (BF16 in memory). It follows the operation order of the PyTorch path
  (PixelAttention._norm_rope_: the T and HW halves of each head normalized by their own RMSNorm, weighted, rotated by the T / H / W
  sections, rounded once); the sums of squares are added in another order: rounding-level differences;
- the dequantization of the history's q8 keys and values (int8 times the FP16 scale of 32 values, in FP32, rounded to BF16): the values
  of the PyTorch path, bit for bit, in one pass.

One binary per kernel and GPU: the token and head counts are not specialized. Launch configurations start from a static choice and are
tuned once per kernel and GPU, outside CUDA graph recording and compilation, on a scratch copy of the first rows of a real call; the
winners are kept in a cache file shared by the sessions."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_hy_nomas_kernels", 1, "Hy-Nomas")
_TUNE_TOKENS = 2048  # tokens of a call timed per candidate


@triton.jit
def _rotate(first, second, COS, SIN, offset, width: tl.constexpr):
    """A split-half section: (first * cos - second * sin, second * cos + first * sin), the products rounded before the sums."""
    cols = offset + tl.arange(0, width)[None, :]
    cos, sin = tl.load(COS + cols), tl.load(SIN + cols)  # the two halves of a section share their angles
    return first * cos - second * sin, second * cos + first * sin


@triton.jit(do_not_specialize=["HEADS"])
def _norm_mrope_kernel(X, WT, WHW, COS, SIN, HEADS, EPS, BLOCK_H: tl.constexpr):
    """BLOCK_H heads of one token per program. Channels: T [0, 64) (norm T, rotated by halves of 32), H [64, 96) and W [96, 128) (norm
    HW, each rotated by halves of 16)."""
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    valid = heads < HEADS
    row = X + (token * HEADS + heads) * 128
    c32, c16 = tl.arange(0, 32)[None, :], tl.arange(0, 16)[None, :]
    t1, t2 = tl.load(row + c32, valid, 0.0).to(tl.float32), tl.load(row + 32 + c32, valid, 0.0).to(tl.float32)
    h1, h2 = tl.load(row + 64 + c16, valid, 0.0).to(tl.float32), tl.load(row + 80 + c16, valid, 0.0).to(tl.float32)
    w1, w2 = tl.load(row + 96 + c16, valid, 0.0).to(tl.float32), tl.load(row + 112 + c16, valid, 0.0).to(tl.float32)
    rt = libdevice.rsqrt((tl.sum(t1 * t1, axis=1) + tl.sum(t2 * t2, axis=1))[:, None] / 64 + EPS)
    rhw = libdevice.rsqrt((tl.sum(h1 * h1, axis=1) + tl.sum(h2 * h2, axis=1) + tl.sum(w1 * w1, axis=1) + tl.sum(w2 * w2, axis=1))[:, None] / 64 + EPS)
    t1, t2 = tl.load(WT + c32).to(tl.float32) * (t1 * rt), tl.load(WT + 32 + c32).to(tl.float32) * (t2 * rt)
    h1, h2 = tl.load(WHW + c16).to(tl.float32) * (h1 * rhw), tl.load(WHW + 16 + c16).to(tl.float32) * (h2 * rhw)
    w1, w2 = tl.load(WHW + 32 + c16).to(tl.float32) * (w1 * rhw), tl.load(WHW + 48 + c16).to(tl.float32) * (w2 * rhw)
    tables = token * 128
    t1, t2 = _rotate(t1, t2, COS + tables, SIN + tables, 0, 32)
    h1, h2 = _rotate(h1, h2, COS + tables, SIN + tables, 64, 16)
    w1, w2 = _rotate(w1, w2, COS + tables, SIN + tables, 96, 16)
    tl.store(row + c32, t1, valid)
    tl.store(row + 32 + c32, t2, valid)
    tl.store(row + 64 + c16, h1, valid)
    tl.store(row + 80 + c16, h2, valid)
    tl.store(row + 96 + c16, w1, valid)
    tl.store(row + 112 + c16, w2, valid)


@triton.jit(do_not_specialize=["N"])
def _dequantize_q8_kernel(DATA, SCALES, OUT, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    values = tl.load(DATA + offsets, valid, 0).to(tl.float32) * tl.load(SCALES + offsets // 32, valid, 0.0).to(tl.float32)
    tl.store(OUT + offsets, values.to(tl.bfloat16), valid)


def dequantize_q8(data, scales):
    """BF16 values of q8 data (..., head_dim) int8 contiguous with scales (..., head_dim // 32) FP16 contiguous."""
    out = torch.empty(data.shape, dtype=torch.bfloat16, device=data.device)

    def launch(source, source_scales, target, config):
        block, warps = config
        _dequantize_q8_kernel[(triton.cdiv(source.numel(), block),)](source, source_scales, target, source.numel(), block, num_warps=warps)

    def scratch():
        rows = _TUNE_TOKENS * 16
        source, source_scales = data.reshape(-1, data.shape[-1])[:rows], scales.reshape(-1, scales.shape[-1])[:rows]
        target = torch.empty(source.shape, dtype=torch.bfloat16, device=data.device)
        return lambda config: launch(source, source_scales, target, config)

    config = _TUNER.config("dequantize_q8", data.shape[-1], data.device, ((2048, 4), (1024, 4), (4096, 4), (4096, 8), (8192, 8)), scratch,
                           data.numel() >= _TUNE_TOKENS * 16 * data.shape[-1])
    launch(data, scales, out, config)
    return out


def norm_mrope_(x, t_weight, hw_weight, rope, eps):
    """In place: x (1, tokens, heads, 128) BF16 contiguous, the T and HW RMSNorm weights (64 each), rope (cos, sin) of shape
    (1, tokens, 1, 128) FP32 contiguous (pixelumm.mrope_tables)."""
    _, tokens, heads, _ = x.shape
    cos, sin = rope

    def launch(target, config):
        block_heads, warps = config
        _norm_mrope_kernel[(target.shape[1], triton.cdiv(heads, block_heads))](target, t_weight, hw_weight, cos, sin, heads, eps, block_heads,
                                                                              num_warps=warps, enable_fp_fusion=False)

    def scratch():
        target = x[:, :_TUNE_TOKENS].clone()
        return lambda config: launch(target, config)

    config = _TUNER.config("norm_mrope", heads, x.device, ((8, 4), (4, 2), (8, 2), (16, 4), (32, 8), (2, 1)), scratch, tokens >= _TUNE_TOKENS)
    launch(x, config)
    torch.autograd.graph.increment_version(x)  # an in-place update, as PyTorch records it (MMGP's CUDA graphs copy back mutated arguments)
    return x


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
