"""Memory-saving projection and head-split helpers shared by model attentions."""

from functools import lru_cache
from collections import namedtuple

import torch
from mmgp import offload

from shared.attention import pay_attention, sage2_staged_settings


HEAD_SPLIT_CHOICES = [
    ("Off", 0),
    ("Low: saves some VRAM", 1),
    ("Medium (good balance): saves more VRAM", 2),
    ("High: saves the most VRAM", 3),
]
_TARGET_GROUPS = (1, 4, 8, 16)
MIN_SPLIT_TOKENS = 8192
head_split = 0
_unsplit_notice = False
_split_notice = False

# Ranges of q heads and kv heads for a group. mean_squares contains the
# per-token full-head statistics needed by norms that span all heads.
HeadGroup = namedtuple("HeadGroup", "q kv mean_squares")


def configure(level):
    """Set the Attention Head Split level (0 disables it)."""

    global head_split, _split_notice, _unsplit_notice
    head_split = min(max(int(level), 0), len(_TARGET_GROUPS) - 1)
    _split_notice = _unsplit_notice = False


@lru_cache(maxsize=128)
def _cached_head_groups(kv_heads, level, long_sequence):
    if level == 0 or not long_sequence or kv_heads <= 1:
        return 1
    return min(
        (groups for groups in range(1, kv_heads + 1) if kv_heads % groups == 0),
        key=lambda groups: (abs(groups - _TARGET_GROUPS[level]), groups),
    )


def head_groups(kv_heads, tokens, level=None):
    """Choose a cached head divisor; split targets apply from 8192 tokens.

    Projection splitting also requires MMGP row support. Other quantization or
    LoRA modules keep their normal forward path and run attention unsplit.
    """

    level = head_split if level is None else level
    level = min(max(int(level), 0), len(_TARGET_GROUPS) - 1)
    return _cached_head_groups(
        int(kv_heads), level, int(tokens) >= MIN_SPLIT_TOKENS
    )


def _linear_rows_supported(projections, x):
    """Keep custom quantized and LoRA modules on their ordinary forward path."""

    supports = getattr(offload, "linear_rows_supported", None)
    if supports is None:
        return False
    try:
        return bool(supports(projections, x))
    except (TypeError, ValueError, NotImplementedError, RuntimeError):
        return False


@torch.compiler.disable()
def kv_first_attention(qkv_list, force_attention=None):
    """Quantize K and V before Q when staged Sage2 can consume their storage."""

    query = qkv_list[0]
    settings = (
        sage2_staged_settings(query.device, force_attention)
        if query.shape[-1] in (64, 128)
        else None
    )
    if settings is None:
        return pay_attention(
            qkv_list, force_attention=force_attention, recycle_q=True
        )

    from shared import sage2_core

    key_list, value_list = qkv_list[1:2], qkv_list[2:3]
    qkv_list.clear()
    quantized = [
        None,
        sage2_core.staged_quantize_k(key_list, settings),
        sage2_core.staged_quantize_v(value_list, settings),
    ]
    quantized[0] = sage2_core.staged_quantize_q(query, settings)
    return sage2_core.staged_attention(query, quantized, settings)


@torch.compiler.disable()
def qkv_attention(
    x_list,
    q_proj,
    k_proj,
    v_proj,
    heads,
    head_dim,
    norm_rope,
    kv_heads=None,
    norm_spans_heads=False,
    split_heads=True,
    attention_fn=None,
):
    """Project and attend q/k/v while handing off the input tensor.

    norm_rope(query, key, group) normalizes and rotates each supplied tensor
    in place. Input tensors have shape (batch, tokens, heads, head_dim). A
    norm that spans all heads receives its per-token full-head statistics in
    group.mean_squares when the heads are split. If attention_fn is supplied,
    it handles each mutable Q/K/V group instead of the shared dense/Sage2
    dispatcher; the helper clears the group list after the callback returns.
    """

    global _unsplit_notice, _split_notice
    x = x_list[0]
    x_list.clear()
    shape = tuple(x.shape[:-1]) if x.dim() > 2 else (1, x.shape[0])
    kv_heads = int(kv_heads or heads)
    groups = head_groups(kv_heads, shape[-1]) if split_heads else 1
    # A caller-supplied attention implementation owns each Q/K/V group. In
    # particular, sparse attention cannot consume Sage2's staged quantized
    # tensors, so do not initialize or enter that path for custom dispatch.
    settings = (
        sage2_staged_settings(x.device)
        if attention_fn is None and head_dim in (64, 128)
        else None
    )
    projections = (q_proj, k_proj, v_proj)
    linear_input = None
    if _linear_rows_supported(projections, x):
        prepare = getattr(offload, "prepare_linear_input", None)
        linear_rows = getattr(offload, "linear_rows", None)
        if prepare is None or linear_rows is None:
            groups = 1
        else:
            linear_input = prepare([x], projections)
            x = None
    else:
        if groups > 1 and not _unsplit_notice:
            print(
                "[Attention] Head Split is not applied: these q/k/v projections "
                "cannot be computed by groups of heads (weight format or LoRA type)."
            )
            _unsplit_notice = True
        groups = 1

    if groups > 1 and not _split_notice:
        print(
            f"[Attention] Head Split active: {heads} query heads, "
            f"{kv_heads} key/value heads in {groups} groups ({shape[-1]:,} tokens)."
        )
        _split_notice = True

    def project(module, head_range):
        if linear_input is None:
            return module(x).view(*shape, -1, head_dim)
        return offload.linear_rows(
            module,
            linear_input,
            head_range.start * head_dim,
            head_range.stop * head_dim,
        ).view(*shape, -1, head_dim)

    mean_squares = None
    if norm_spans_heads and groups > 1:
        mean_squares = []
        for module, total in ((q_proj, heads), (k_proj, kv_heads)):
            step, squares = total // groups, None
            for group_index in range(groups):
                part = project(
                    module,
                    range(group_index * step, (group_index + 1) * step),
                ).float().square_().sum(dim=(-2, -1))
                squares = part if squares is None else squares.add_(part)
                del part
            mean_squares.append(squares.div_(total * head_dim).unsqueeze(-1))
        mean_squares = tuple(mean_squares)

    if settings is not None:
        from shared import sage2_core

    q_group_heads = heads // groups
    kv_group_heads = kv_heads // groups
    output = None
    for index in range(groups):
        group = HeadGroup(
            range(index * q_group_heads, (index + 1) * q_group_heads),
            range(index * kv_group_heads, (index + 1) * kv_group_heads),
            mean_squares,
        )
        last = index == groups - 1
        if settings is None:
            qkv_group = [
                project(q_proj, group.q),
                project(k_proj, group.kv),
                project(v_proj, group.kv),
            ]
            if last:
                linear_input = x = None
            norm_rope(qkv_group[0], qkv_group[1], group)
            if attention_fn is None:
                attention = pay_attention(qkv_group, recycle_q=True)
            else:
                try:
                    attention = attention_fn(qkv_group)
                finally:
                    qkv_group.clear()
            if groups == 1:
                return attention
            if output is None:
                output = attention.new_empty((*shape, heads, head_dim))
            output[..., group.q.start:group.q.stop, :].copy_(attention)
            del attention
            continue

        quantized = [
            None,
            None,
            sage2_core.staged_quantize_v(
                [project(v_proj, group.kv)], settings
            ),
        ]
        if groups == 1:
            key = project(k_proj, group.kv)
            norm_rope(None, key, group)
            quantized[1] = sage2_core.staged_quantize_k([key], settings)
            del key
            query = project(q_proj, group.q)
            linear_input = x = None
            norm_rope(query, None, group)
            quantized[0] = sage2_core.staged_quantize_q(query, settings)
            return sage2_core.staged_attention(query, quantized, settings)

        query, key = project(q_proj, group.q), project(k_proj, group.kv)
        if last:
            linear_input = x = None
        norm_rope(query, key, group)
        quantized[1] = sage2_core.staged_quantize_k([key], settings)
        del key
        quantized[0] = sage2_core.staged_quantize_q(query, settings)
        if output is None:
            output = query.new_empty((*shape, heads, head_dim))
        del query
        sage2_core.staged_attention(
            output[..., group.q.start:group.q.stop, :], quantized, settings
        )
    return output


__all__ = [
    "HEAD_SPLIT_CHOICES",
    "HeadGroup",
    "configure",
    "head_groups",
    "kv_first_attention",
    "qkv_attention",
]
