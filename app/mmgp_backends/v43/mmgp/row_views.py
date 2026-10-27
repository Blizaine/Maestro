# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Row views: the projections of a fused linear whose rows interleave them head by head, (heads, slots, head_dim, in_features), as
the qkv projection of some attention checkpoints. Splitting such a weight into separate projections at load copies all its rows into
private RAM; a row view computes its projection from the fused weight in place instead, with a kernel registered by the application
(register_kernel), so the fused weight stays in its checkpoint file and MMGP manages it as one parameter (sizes, pinning, transfers).

A split map entry declares this layout with "interleaved": True and "attention_head_dim"; load_model_data keeps such weights fused and
installs row views when the registered kernel supports them, and splits them into copies of their rows otherwise."""
import torch
from torch.utils._pytree import tree_map

_kernel = None  # rows(x, weight, bias, slot, slots, head_dim, start, stop) -> (tokens, stop - start)
_available = None  # available(dtype, head_dim) -> bool


def register_kernel(rows, available):
    """rows computes columns [start, stop) (multiples of head_dim) of projection slot of an interleaved fused weight for x of shape
    (tokens, in_features); available(dtype, head_dim) tells at load whether it computes them as fast as separate weights would be."""
    global _kernel, _available
    _kernel, _available = rows, available
    from .offload import register_row_projection
    register_row_projection(_ROW_VIEW_PROJECTION)


def available(dtype, head_dim):
    return _available is not None and _available(dtype, head_dim)


def gather(tensor, slot, slots, head_dim):
    """The rows of projection slot, copied: a split without views."""
    return tensor.reshape(-1, slots, head_dim, *tensor.shape[1:])[:, slot].reshape(-1, *tensor.shape[1:]).contiguous()


class RowViewWeight(torch.Tensor):
    """The weight of a row view as code inspecting linear modules sees it (shape, dtype, device); any operation on it works on a copy
    of the projection's rows (DoRA merges, for instance)."""
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, view):
        fused = view.fused.weight
        return torch.Tensor._make_wrapper_subclass(cls, (view.out_features, view.in_features), dtype=fused.dtype, device=fused.device)

    def __init__(self, view):
        self.view = view

    def __repr__(self):
        return f"RowViewWeight(projection {self.view.slot} of {self.view.slots}, {tuple(self.shape)}, {self.dtype})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        materialize = lambda value: value.view.materialize() if isinstance(value, RowViewWeight) else value
        return func(*tree_map(materialize, args), **tree_map(materialize, kwargs or {}))


class RowViewLinear(torch.nn.Linear):
    """Projection `slot` of a fused linear, computed in place from the fused weight: a linear module without parameters of its own (the
    fused module holds them), seen as a linear by LoRAs, hooks and row projections (offload.linear_rows)."""
    _mm_requires_native_linear_forward = True  # LoRA forwards call forward(), not F.linear on the weight

    def __init__(self, fused, slot, slots, head_dim):
        torch.nn.Module.__init__(self)
        self.__dict__["fused"] = fused  # not a submodule: its parent holds it under its own name
        self.slot, self.slots, self.head_dim = slot, slots, head_dim
        self.in_features, self.out_features = fused.in_features, fused.out_features // slots

    @property
    def weight(self):
        return RowViewWeight(self)

    @property
    def bias(self):
        bias = self.fused.bias
        return None if bias is None else gather(bias, self.slot, self.slots, self.head_dim)

    def materialize(self):
        return gather(self.fused.weight, self.slot, self.slots, self.head_dim)

    def rows(self, x, start, stop):
        return _kernel(x, self.fused.weight, self.fused.bias, self.slot, self.slots, self.head_dim, start, stop)

    def forward(self, x):
        return self.rows(x.reshape(-1, x.shape[-1]), 0, self.out_features).view(*x.shape[:-1], self.out_features)

    def extra_repr(self):
        return f"projection {self.slot} of {self.slots}, in_features={self.in_features}, out_features={self.out_features}, head_dim={self.head_dim}"


class _RowViewProjection:
    """offload.linear_rows of row views: a range of a projection's output columns, from x itself."""
    @staticmethod
    def supports(module, x):
        return isinstance(module, RowViewLinear) and module.fused.weight.dtype == x.dtype

    @staticmethod
    def key(module):
        return None

    @staticmethod
    def prepare(module, x):
        return None

    @staticmethod
    def rows(module, x, start, stop):
        return module.rows(x, start, stop)


_ROW_VIEW_PROJECTION = _RowViewProjection()


def install(model, base, info, state_dict):
    """Keeps the fused linear `base` (its weight in state_dict) and puts row views in place of its split projections."""
    prefix, _, suffix = base.rpartition(".")
    parent = model.get_submodule(prefix) if prefix else model
    weight, bias = state_dict[base + ".weight"], state_dict.get(base + ".bias")
    with torch.device("meta"):
        fused = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, dtype=weight.dtype)
    setattr(parent, suffix, fused)
    mapped = info["mapped_modules"]
    for slot, name in enumerate(mapped):
        owner, _, leaf = name.rpartition(".")
        setattr(parent.get_submodule(owner) if owner else parent, leaf, RowViewLinear(fused, slot, len(mapped), info["attention_head_dim"]))
