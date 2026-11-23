"""Memory-saving projection and head-split helpers shared by model attentions."""

from functools import lru_cache
from collections import namedtuple

import torch
import torch.nn.functional as F
from mmgp import offload

from shared.attention import pay_attention, sage2_staged_settings
from shared import attention_chunking as _chunking


CHUNKER_CONFIG_KEY = _chunking.CHUNKER_CONFIG_KEY
CHUNKER_DEFAULT = _chunking.CHUNKER_DEFAULT
HEAD_SPLIT_CHOICES = [
    ("Auto (when VRAM is tight)", -1),
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
_auto_split_decisions = {}
EXTREME_QUERY_BYTES = 64 << 20
EXTREME_INPUT_BYTES = 64 << 20
EXTREME_OUTPUT_SLICE_BYTES = 16 << 20
EXTREME_MIN_TOKENS = 8192
EXTREME_GROUP_HEADS = 1

# Ranges of q heads and kv heads for a group. mean_squares contains the
# per-token full-head statistics needed by norms that span all heads.
HeadGroup = namedtuple("HeadGroup", "q kv mean_squares rows", defaults=(None,))


def configure(level):
    """Set the head split level (-1 uses the current job's VRAM budget)."""

    global head_split, _split_notice, _unsplit_notice
    head_split = min(max(int(level), -1), len(_TARGET_GROUPS) - 1)
    _split_notice = _unsplit_notice = False
    _auto_split_decisions.clear()


def configure_chunker(mode):
    """Set the saved Smart Inputs Chunker mode (off, auto, on, or extreme)."""

    return _chunking.configure_chunker(mode)


def current_chunker():
    """Return the effective mode, including a per-generation scope override."""

    return _chunking.current_chunker()


def chunker_scope(mode=None):
    """Scope a saved or per-job mode to this generation/pass, with automatic reset."""

    return _chunking.chunker_scope(mode)


def chunking_needed(device, extra_bytes, kind):
    """Whether the effective mode requires splitting a call with this shape."""

    return _chunking.chunking_needed(device, extra_bytes, kind)


def _dtype_itemsize(dtype):
    try:
        return torch.empty((), dtype=dtype).element_size()
    except (RuntimeError, TypeError):
        return 2


def _active_lora(module):
    data = getattr(module, "_mm_lora_data", None)
    model = getattr(module, "_mm_lora_model", None)
    active = getattr(model, "_loras_active_adapters", ()) if model is not None else ()
    return bool(data and any(name in data or f"{name}_GPU" in data for name in active))


def _extreme_output_projection_supported(module, input_width):
    """The partial-column output projection is exact for plain dense linears."""

    return (
        type(module) is torch.nn.Linear
        and module.in_features == input_width
        and module.weight.layout == torch.strided
        and not _active_lora(module)
    )


def _extreme_plan_estimates(rows, hidden, heads, kv_heads, head_dim, dtype, out_proj):
    itemsize = _dtype_itemsize(dtype)
    level = max(head_split, len(_TARGET_GROUPS) - 1)
    groups = head_groups(
        kv_heads, rows, level=level, heads=heads, head_dim=head_dim, element_size=itemsize
    )
    q_per_group, kv_per_group = heads // groups, kv_heads // groups
    # Standard keeps the complete attention output while one group's Q/K/V is
    # live; its caller then allocates the output projection result.
    standard = rows * hidden * itemsize * 2
    standard += rows * head_dim * itemsize * (q_per_group + 2 * kv_per_group)
    # Extreme keeps the required projected result and one group's K/V; query,
    # attention, and partial output-projection temporaries are token-bounded.
    extreme_group_q = max(1, heads // kv_heads)
    output_width = int(out_proj.out_features)
    query_rows = _extreme_query_rows(1, extreme_group_q, head_dim, itemsize, rows, output_width)
    extreme = rows * output_width * itemsize
    extreme += rows * head_dim * itemsize * 2
    extreme += query_rows * itemsize * (3 * extreme_group_q * head_dim + output_width)
    extreme += min(
        EXTREME_INPUT_BYTES,
        rows * itemsize * (hidden + 2 * head_dim),
    )
    return standard, extreme


def _auto_head_split_level(device, tokens, heads, head_dim, element_size):
    if device is None or device.type != "cuda" or tokens < MIN_SPLIT_TOKENS:
        return 0
    key = (device, tokens, heads, head_dim, element_size)
    if key not in _auto_split_decisions:
        try:
            if torch.cuda.is_current_stream_capturing():
                return 0
            free, total = torch.cuda.mem_get_info(device)
            reusable = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
        except (RuntimeError, TypeError, ValueError):
            return 0
        budget = free + reusable - max(512 << 20, total // 10)
        # Q, K, V and the attended rows coexist on the unsplit path.
        scratch = 4 * tokens * heads * head_dim * element_size
        _auto_split_decisions[key] = next(
            (level for level, groups in enumerate(_TARGET_GROUPS) if scratch <= budget * groups),
            len(_TARGET_GROUPS) - 1,
        )
    return _auto_split_decisions[key]


@lru_cache(maxsize=128)
def _cached_head_groups(kv_heads, level, long_sequence):
    if level == 0 or not long_sequence or kv_heads <= 1:
        return 1
    return min(
        (groups for groups in range(1, kv_heads + 1) if kv_heads % groups == 0),
        key=lambda groups: (abs(groups - _TARGET_GROUPS[level]), groups),
    )


def head_groups(kv_heads, tokens, level=None, *, device=None, heads=None, head_dim=128, element_size=2):
    """Choose a cached head divisor; split targets apply from 8192 tokens.

    Projection splitting also requires MMGP row support. Other quantization or
    LoRA modules keep their normal forward path and run attention unsplit.
    """

    level = head_split if level is None else level
    if level == -1:
        level = _auto_head_split_level(device, tokens, heads or kv_heads, head_dim, element_size)
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


def _extreme_row_projection_supported(modules, x):
    if _linear_rows_supported(modules, x):
        return True
    return all(
        type(module) is torch.nn.Linear
        and module.weight.layout == torch.strided
        and not _active_lora(module)
        for module in modules
    )


def _project_columns(module, x, start, stop):
    """Compute an output-column slice while honoring MMGP row adapters."""

    prepare = getattr(offload, "prepare_linear_input", None)
    linear_rows = getattr(offload, "linear_rows", None)
    supports = getattr(offload, "linear_rows_supported", None)
    if prepare is not None and linear_rows is not None and supports is not None:
        try:
            if supports((module,), x):
                prepared = prepare([x], (module,))
                return linear_rows(module, prepared, start, stop)
        except torch.OutOfMemoryError:
            raise
        except (TypeError, ValueError, NotImplementedError, RuntimeError):
            pass
    if type(module) is torch.nn.Linear and module.weight.layout == torch.strided and not _active_lora(module):
        bias = None if module.bias is None else module.bias[start:stop]
        return F.linear(x, module.weight[start:stop], bias)
    raise NotImplementedError("This projection has no safe row-sliced implementation")


def _extreme_attention_supported(x, projections, out_proj, heads, kv_heads, head_dim, norm_rope_rows):
    if (
        not norm_rope_rows
        or torch.is_grad_enabled()
        or x.dim() not in (2, 3)
        or heads < 1
        or kv_heads < 1
        or heads % kv_heads
        or out_proj is None
        or not _extreme_output_projection_supported(out_proj, heads * head_dim)
        or out_proj.weight.device != x.device
    ):
        return False
    if len(projections) != 3 or not _extreme_row_projection_supported(projections, x):
        return False
    widths = (heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
    return all(getattr(module, "out_features", None) == width for module, width in zip(projections, widths))


def _extreme_query_rows(batch, query_heads, head_dim, element_size, tokens, output_width=0):
    bytes_per_row = max(1, batch * query_heads * head_dim * element_size * 3)
    rows = max(1, EXTREME_QUERY_BYTES // bytes_per_row)
    if output_width:
        # A single head can still project to a wide output. Bound that
        # partial result even when the small Q tensor fits in one chunk.
        rows = min(rows, max(1, EXTREME_OUTPUT_SLICE_BYTES // (batch * element_size * output_width)))
    if tokens >= 128:
        rows = max(128, rows // 128 * 128)
    return min(tokens, rows)


def _extreme_input_rows(batch, input_width, output_width, element_size, tokens):
    bytes_per_row = max(1, batch * element_size * (input_width + 2 * output_width))
    rows = max(1, EXTREME_INPUT_BYTES // bytes_per_row)
    if tokens >= 128:
        rows = max(128, rows // 128 * 128)
    return min(tokens, rows)


def _full_head_squares(module, x, heads, head_dim, tokens, element_size):
    batch, _, _ = x.shape
    rows = _extreme_query_rows(batch, heads, head_dim, element_size, tokens)
    result = torch.empty((batch, tokens, 1), dtype=torch.float32, device=x.device)
    for first in range(0, tokens, rows):
        last = min(tokens, first + rows)
        values = _project_columns(module, x[:, first:last], 0, heads * head_dim)
        values = values.view(batch, last - first, heads, head_dim)
        result[:, first:last, 0] = values.float().square().mean(dim=(-2, -1))
        del values
    return result


def _extreme_qkv_attention(
    x, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, kv_heads,
    norm_spans_heads, out_proj, force_attention=None,
):
    """Attend query chunks by KV-head groups and project each result in place."""

    if x.dim() == 2:
        x = x.unsqueeze(0)
    batch, tokens, _ = x.shape
    element_size = x.element_size()
    group_heads = max(1, min(int(EXTREME_GROUP_HEADS), kv_heads))
    group_heads = max(
        [size for size in range(1, group_heads + 1) if kv_heads % size == 0] or [1]
    )
    q_group_heads = heads * group_heads // kv_heads
    output_features = out_proj.out_features
    query_rows = _extreme_query_rows(batch, q_group_heads, head_dim, element_size, tokens, output_features)
    projections = (q_proj, k_proj, v_proj)
    squares = None
    if norm_spans_heads:
        squares = (
            _full_head_squares(q_proj, x, heads, head_dim, tokens, element_size),
            _full_head_squares(k_proj, x, kv_heads, head_dim, tokens, element_size),
        )

    output = None
    for kv_first in range(0, kv_heads, group_heads):
        kv_last = min(kv_heads, kv_first + group_heads)
        q_first = kv_first * heads // kv_heads
        q_last = kv_last * heads // kv_heads
        q_range, kv_range = range(q_first, q_last), range(kv_first, kv_last)
        group = HeadGroup(q_range, kv_range, squares, slice(0, tokens))
        key_start, key_stop = kv_first * head_dim, kv_last * head_dim
        group_width = len(kv_range) * head_dim
        input_rows = _extreme_input_rows(
            batch, x.shape[-1], group_width, element_size, tokens
        )
        key = value = None
        for source_first in range(0, tokens, input_rows):
            source_last = min(tokens, source_first + input_rows)
            source = x[:, source_first:source_last]
            key_part = _project_columns(k_proj, source, key_start, key_stop)
            value_part = _project_columns(v_proj, source, key_start, key_stop)
            if key is None:
                key = key_part.new_empty((batch, tokens, group_width))
                value = value_part.new_empty((batch, tokens, group_width))
            key[:, source_first:source_last].copy_(key_part)
            value[:, source_first:source_last].copy_(value_part)
            del source, key_part, value_part
        key = key.view(batch, tokens, len(kv_range), head_dim)
        value = value.view(batch, tokens, len(kv_range), head_dim)
        norm_rope(None, key, group)

        for first in range(0, tokens, query_rows):
            last = min(tokens, first + query_rows)
            q_input = x[:, first:last]
            q_start, q_stop = q_first * head_dim, q_last * head_dim
            query = _project_columns(q_proj, q_input, q_start, q_stop)
            query = query.view(batch, last - first, len(q_range), head_dim)
            query_squares = None
            if squares is not None:
                query_squares = (
                    squares[0][:, first:last],
                    squares[1][:, first:last],
                )
            norm_rope(
                query,
                None,
                HeadGroup(q_range, kv_range, query_squares, slice(first, last)),
            )
            qkv_group = [query, key, value]
            attended = pay_attention(qkv_group, force_attention=force_attention, recycle_q=True)
            qkv_group.clear()
            query = None
            del q_input, qkv_group

            columns_start, columns_stop = q_start, q_stop
            partial = F.linear(
                attended.flatten(2),
                out_proj.weight[:, columns_start:columns_stop],
            )
            target = None if output is None else output[:, first:last]
            if output is None:
                output = torch.empty(
                    (batch, tokens, output_features), dtype=partial.dtype, device=partial.device
                )
                target = output[:, first:last]
            if kv_first == 0:
                target.copy_(partial)
            else:
                target.add_(partial)
            if kv_last == kv_heads and out_proj.bias is not None:
                target.add_(out_proj.bias)
            del attended, partial, target
        del key, value

    return output


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
    attention_group_fn=None,
    out_proj=None,
    norm_rope_rows=False,
    force_attention=None,
):
    """Project and attend q/k/v while handing off the input tensor.

    norm_rope(query, key, group) normalizes and rotates each supplied tensor
    in place. Input tensors have shape (batch, tokens, heads, head_dim). A
    norm that spans all heads receives its per-token full-head statistics in
    group.mean_squares when the heads are split. If attention_fn is supplied,
    it handles each mutable Q/K/V group instead of the shared dense/Sage2
    dispatcher; the helper clears the group list after the callback returns.
    attention_group_fn additionally receives the absolute HeadGroup so
    learned per-head predictors retain their identity when heads are split.
    With out_proj, return its projection of the attention result. Extreme also
    requires norm_rope_rows=True: the callback uses group.rows to select the
    matching positional rows for each bounded query chunk.
    """

    global _unsplit_notice, _split_notice
    x = x_list[0]
    x_list.clear()
    shape = tuple(x.shape[:-1]) if x.dim() > 2 else (1, x.shape[0])
    kv_heads = int(kv_heads or heads)
    tokens = x.numel() // x.shape[-1]
    mode = current_chunker()
    split_level = head_split
    auto_needed = False
    if mode == "on" or mode == "extreme":
        split_level = max(split_level, len(_TARGET_GROUPS) - 1)
    elif mode == "auto":
        estimate = 4 * tokens * heads * head_dim * x.element_size()
        auto_needed = chunking_needed(
            x.device, estimate, ("attention", heads, kv_heads, head_dim, tuple(shape), x.dtype)
        )
        if auto_needed:
            split_level = max(split_level, len(_TARGET_GROUPS) - 1)
    groups = head_groups(
        kv_heads, tokens, level=split_level, device=x.device,
        heads=heads, head_dim=head_dim, element_size=x.element_size(),
    ) if split_heads else 1

    # Auto upgrades only when the normal head-group pass still exceeds the
    # estimated room and the row-bounded path is both supported and smaller.
    # An explicit Extreme request tries that same guarded path even without an
    # Auto estimate; unsupported projections retain the regular implementation.
    try_extreme = mode == "extreme" or (mode == "auto" and auto_needed)
    if (
        try_extreme
        and split_heads
        and tokens >= EXTREME_MIN_TOKENS
        and out_proj is not None
        and attention_fn is None
        and attention_group_fn is None
        and _extreme_attention_supported(
            x, (q_proj, k_proj, v_proj), out_proj, heads, kv_heads, head_dim, norm_rope_rows
        )
    ):
        use_extreme = mode == "extreme"
        if mode == "auto":
            standard, extreme = _extreme_plan_estimates(
                tokens, x.shape[-1], heads, kv_heads, head_dim, x.dtype, out_proj
            )
            room = _chunking.auto_room(x.device)
            use_extreme = (
                standard > room
                and extreme <= room
                and standard - extreme >= _chunking.EXTREME_MIN_SAVING
            )
        if use_extreme:
            return _extreme_qkv_attention(
                x, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, kv_heads,
                norm_spans_heads, out_proj, force_attention=force_attention,
            )
    # A caller-supplied attention implementation owns each Q/K/V group. In
    # particular, sparse attention cannot consume Sage2's staged quantized
    # tensors, so do not initialize or enter that path for custom dispatch.
    settings = (
        sage2_staged_settings(x.device, force_attention)
        if attention_fn is None and attention_group_fn is None and head_dim in (64, 128)
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
            if attention_fn is None and attention_group_fn is None:
                attention = pay_attention(qkv_group, force_attention=force_attention, recycle_q=True)
            else:
                try:
                    attention = (attention_group_fn(qkv_group, group)
                                 if attention_group_fn is not None else attention_fn(qkv_group))
                finally:
                    qkv_group.clear()
            if groups == 1:
                return attention if out_proj is None else out_proj(attention.flatten(2))
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
            attention = sage2_core.staged_attention(query, quantized, settings)
            return attention if out_proj is None else out_proj(attention.flatten(2))

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
    return output if out_proj is None else out_proj(output.flatten(2))


__all__ = [
    "CHUNKER_CONFIG_KEY",
    "CHUNKER_DEFAULT",
    "HEAD_SPLIT_CHOICES",
    "HeadGroup",
    "configure",
    "configure_chunker",
    "chunker_scope",
    "chunking_needed",
    "current_chunker",
    "head_groups",
    "kv_first_attention",
    "qkv_attention",
]
