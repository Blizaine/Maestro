# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Selected resident experts of routed SwiGLU banks, addressed individually through a pointer table (one row per expert: gate/up
weights, down weights, gate/up scales, down scales), for the FP8 block and INT4 group formats.

This vLLM backend uses float32 activations/accumulation, matching the CPU backend. It never transfers weights. Nonresident ids
contribute zero to the GPU result."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_gate_up(X, IDs, PTR, Mid, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    # One (token, slot) pair and ROWS rows of its gate and up projections, a scale block of columns at a time.
    pair = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    expert = tl.load(IDs + pair).to(tl.int64)
    address = tl.load(PTR + expert * 4)
    gate = tl.zeros((ROWS,), tl.float32)
    up = tl.zeros((ROWS,), tl.float32)
    if address != 0:
        weight = address.to(tl.pointer_type(tl.float8e4nv))
        scales = tl.load(PTR + expert * 4 + 2).to(tl.pointer_type(tl.float32))
        cols = tl.arange(0, BLOCK)
        for block in range(H // BLOCK):
            x = tl.load(X + (pair // K) * H + block * BLOCK + cols).to(tl.float32)
            g = tl.load(weight + rows[:, None] * H + block * BLOCK + cols[None, :]).to(tl.float32)
            u = tl.load(weight + (rows[:, None] + I) * H + block * BLOCK + cols[None, :]).to(tl.float32)
            gate += tl.sum(g * x[None, :], 1) * tl.load(scales + (rows // BLOCK) * (H // BLOCK) + block)
            up += tl.sum(u * x[None, :], 1) * tl.load(scales + ((rows + I) // BLOCK) * (H // BLOCK) + block)
    tl.store(Mid + pair * I + rows, gate * tl.sigmoid(gate) * up)


@triton.jit
def _fp8_down(Mid, IDs, Probs, PTR, Out, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    # ROWS output rows of one token, summed over its resident slots.
    token = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    total = tl.zeros((ROWS,), tl.float32)
    for slot in tl.static_range(K):
        pair = token * K + slot
        expert = tl.load(IDs + pair).to(tl.int64)
        address = tl.load(PTR + expert * 4 + 1)
        if address != 0:
            weight = address.to(tl.pointer_type(tl.float8e4nv))
            scales = tl.load(PTR + expert * 4 + 3).to(tl.pointer_type(tl.float32))
            value = tl.zeros((ROWS,), tl.float32)
            for block in range(I // BLOCK):
                x = tl.load(Mid + pair * I + block * BLOCK + cols)
                w = tl.load(weight + rows[:, None] * I + block * BLOCK + cols[None, :]).to(tl.float32)
                value += tl.sum(w * x[None, :], 1) * tl.load(scales + (rows // BLOCK) * (I // BLOCK) + block)
            total += value * tl.load(Probs + pair).to(tl.float32)
    tl.store(Out + token * H + rows, total.to(Out.dtype.element_ty))


@triton.jit
def _fp8_down_partial(Mid, IDs, Probs, PTR, Partial, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    # One (token, slot) pair and ROWS output rows. This exposes routed slots as
    # independent programs for one-frame decode, then an ordered reduction
    # restores the serial FP32 accumulation order used by _fp8_down.
    pair = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    expert = tl.load(IDs + pair).to(tl.int64)
    address = tl.load(PTR + expert * 4 + 1)
    value = tl.zeros((ROWS,), tl.float32)
    if address != 0:
        weight = address.to(tl.pointer_type(tl.float8e4nv))
        scales = tl.load(PTR + expert * 4 + 3).to(tl.pointer_type(tl.float32))
        for block in range(I // BLOCK):
            x = tl.load(Mid + pair * I + block * BLOCK + cols)
            w = tl.load(weight + rows[:, None] * I + block * BLOCK + cols[None, :]).to(tl.float32)
            value += tl.sum(w * x[None, :], 1) * tl.load(scales + (rows // BLOCK) * (I // BLOCK) + block)
        value *= tl.load(Probs + pair).to(tl.float32)
    tl.store(Partial + pair * H + rows, value)


@triton.jit
def _int4_gate_up(X, IDs, PTR, Mid, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr, ROWS: tl.constexpr):
    # As _fp8_gate_up, a scale group at a time: each byte holds an even column (low nibble) and the next one, offset by 8.
    pair = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    expert = tl.load(IDs + pair).to(tl.int64)
    address = tl.load(PTR + expert * 4)
    gate = tl.zeros((ROWS,), tl.float32)
    up = tl.zeros((ROWS,), tl.float32)
    if address != 0:
        weight = address.to(tl.pointer_type(tl.uint8))
        scales = tl.load(PTR + expert * 4 + 2).to(tl.pointer_type(tl.float16))
        cols = tl.arange(0, GROUP // 2)
        for group in range(H // GROUP):
            x = X + (pair // K) * H + group * GROUP + 2 * cols
            even, odd = tl.load(x).to(tl.float32), tl.load(x + 1).to(tl.float32)
            g = tl.load(weight + rows[:, None] * (H // 2) + group * (GROUP // 2) + cols[None, :])
            u = tl.load(weight + (rows[:, None] + I) * (H // 2) + group * (GROUP // 2) + cols[None, :])
            g_sum = tl.sum(((g & 15).to(tl.float32) - 8.0) * even[None, :] + ((g >> 4).to(tl.float32) - 8.0) * odd[None, :], 1)
            u_sum = tl.sum(((u & 15).to(tl.float32) - 8.0) * even[None, :] + ((u >> 4).to(tl.float32) - 8.0) * odd[None, :], 1)
            gate += g_sum * tl.load(scales + rows * (H // GROUP) + group).to(tl.float32)
            up += u_sum * tl.load(scales + (rows + I) * (H // GROUP) + group).to(tl.float32)
    tl.store(Mid + pair * I + rows, gate * tl.sigmoid(gate) * up)


@triton.jit
def _int4_down(Mid, IDs, Probs, PTR, Out, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr, ROWS: tl.constexpr):
    token = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, GROUP // 2)
    total = tl.zeros((ROWS,), tl.float32)
    for slot in tl.static_range(K):
        pair = token * K + slot
        expert = tl.load(IDs + pair).to(tl.int64)
        address = tl.load(PTR + expert * 4 + 1)
        if address != 0:
            weight = address.to(tl.pointer_type(tl.uint8))
            scales = tl.load(PTR + expert * 4 + 3).to(tl.pointer_type(tl.float16))
            value = tl.zeros((ROWS,), tl.float32)
            for group in range(I // GROUP):
                x = Mid + pair * I + group * GROUP + 2 * cols
                even, odd = tl.load(x).to(tl.float32), tl.load(x + 1).to(tl.float32)
                w = tl.load(weight + rows[:, None] * (I // 2) + group * (GROUP // 2) + cols[None, :])
                w_sum = tl.sum(((w & 15).to(tl.float32) - 8.0) * even[None, :] + ((w >> 4).to(tl.float32) - 8.0) * odd[None, :], 1)
                value += w_sum * tl.load(scales + rows * (I // GROUP) + group).to(tl.float32)
            total += value * tl.load(Probs + pair).to(tl.float32)
    tl.store(Out + token * H + rows, total.to(Out.dtype.element_ty))


@triton.jit
def _int4_down_partial(Mid, IDs, Probs, PTR, Partial, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr, ROWS: tl.constexpr):
    # One (token, slot) pair and ROWS output rows; see _fp8_down_partial.
    pair = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, GROUP // 2)
    expert = tl.load(IDs + pair).to(tl.int64)
    address = tl.load(PTR + expert * 4 + 1)
    value = tl.zeros((ROWS,), tl.float32)
    if address != 0:
        weight = address.to(tl.pointer_type(tl.uint8))
        scales = tl.load(PTR + expert * 4 + 3).to(tl.pointer_type(tl.float16))
        for group in range(I // GROUP):
            x = Mid + pair * I + group * GROUP + 2 * cols
            even, odd = tl.load(x).to(tl.float32), tl.load(x + 1).to(tl.float32)
            w = tl.load(weight + rows[:, None] * (I // 2) + group * (GROUP // 2) + cols[None, :])
            w_sum = tl.sum(((w & 15).to(tl.float32) - 8.0) * even[None, :] + ((w >> 4).to(tl.float32) - 8.0) * odd[None, :], 1)
            value += w_sum * tl.load(scales + rows * (I // GROUP) + group).to(tl.float32)
        value *= tl.load(Probs + pair).to(tl.float32)
    tl.store(Partial + pair * H + rows, value)


@triton.jit
def _ordered_topk_reduce(Partial, Out, H: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr):
    token = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    total = tl.zeros((ROWS,), tl.float32)
    for slot in tl.static_range(K):
        total += tl.load(Partial + (token * K + slot) * H + rows)
    tl.store(Out + token * H + rows, total.to(Out.dtype.element_ty))


_KERNELS = {"fp8_block": (_fp8_gate_up, _fp8_down), "int4_group": (_int4_gate_up, _int4_down)}
_PARALLEL_DOWN_KERNELS = {
    "fp8_block": _fp8_down_partial,
    "int4_group": _int4_down_partial,
}


def forward(
    kind,
    x,
    ids,
    probabilities,
    pointers,
    intermediate,
    block,
    out=None,
    mid=None,
    rotation=0,
    rows=8,
    *,
    single_token_rows=None,
    parallel_single_token_down=False,
    down_scratch=None,
):
    """Sum resident selected experts with FP32 accumulation.

    ``out`` and ``mid`` may be preallocated; a supplied ``out`` controls the
    output dtype, while the default output is float32. For one-token decode,
    ``single_token_rows`` overrides the row tile, and
    ``parallel_single_token_down`` computes each routed down projection in an
    independent program before an ordered FP32 top-k reduction. Its optional
    float32 scratch has shape [tokens * topk, hidden]. These options are
    opt-in until benchmarked on the target GPU. Multi-token calls keep the
    existing serial down kernel and ``rows`` setting. ``rotation`` is the
    ConvRot group of the activations of both projections, or 0.
    """
    tokens, hidden = x.shape
    topk = ids.shape[-1]
    gate_up, down = _KERNELS[kind]
    kernel_rows = single_token_rows if tokens == 1 and single_token_rows is not None else rows
    mid = torch.empty(tokens * topk, intermediate, dtype=torch.float32, device=x.device) if mid is None else mid
    out = torch.empty(tokens, hidden, dtype=torch.float32, device=x.device) if out is None else out
    if rotation:
        from shared.qtypes.int8_convrot import _rotate_activation
        x = _rotate_activation(x.float(), rotation)
    gate_up[(tokens * topk, intermediate // kernel_rows)](x, ids, pointers, mid, hidden, intermediate, topk, block, kernel_rows)
    if rotation:
        mid = _rotate_activation(mid, rotation)
    if tokens == 1 and parallel_single_token_down:
        partial = torch.empty(tokens * topk, hidden, dtype=torch.float32, device=x.device) if down_scratch is None else down_scratch
        _PARALLEL_DOWN_KERNELS[kind][(tokens * topk, hidden // kernel_rows)](
            mid, ids, probabilities, pointers, partial, hidden, intermediate, topk, block, kernel_rows
        )
        _ordered_topk_reduce[(tokens, hidden // kernel_rows)](partial, out, hidden, topk, kernel_rows)
    else:
        down[(tokens, hidden // kernel_rows)](mid, ids, probabilities, pointers, out, hidden, intermediate, topk, block, kernel_rows)
    return out
