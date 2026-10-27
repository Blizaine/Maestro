# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see app/LICENSE.txt)
# Compatibility adapter copied from WanGP revision 9905f0df941e5ec7b0c80e32555d3a61fe415726.
# Modified for Maestro's MMGP 4.0 kernel API.
"""Maestro-compatible block-scaled INT8/FP8 checkpoint adapter.

The DiT format stores ``weight_q`` INT8 values with FP32 scales per output
row and each 128 input columns. The PixelUMM format stores ``fp8_weight``
E4M3 values with FP32 scales per 128-by-128 block. MMGP 4.0 in this branch
does not have WanGP's block-scale GPU kernels, so linears use bounded
dequantization to the module's compute dtype followed by PyTorch linear.
"""
import ast
import json

import torch
from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes
from torch.utils import _pytree as pytree
from tqdm import tqdm

HANDLER_NAME = "block_scaled"
HANDLER_PRIORITY = 9


def _register_qtype(name, dtype, floating):
    if name not in _quanto_qtypes:
        info = torch.finfo(dtype) if floating else torch.iinfo(dtype)
        _quanto_qtypes[name] = _quanto_qtype(name, is_floating_point=floating, bits=8, dtype=dtype, qmin=float(info.min), qmax=float(info.max))
    return _quanto_qtypes[name]


_INT8_BLOCKS = _register_qtype("int8_blocks", torch.int8, False)
_FP8_BLOCKS = _register_qtype("float8_e4m3fn_blocks", torch.float8_e4m3fn, True)
_QTYPES = {torch.int8: _INT8_BLOCKS, torch.float8_e4m3fn: _FP8_BLOCKS}
_FORMATS = ((".weight_q", ".weight_scale"), (".fp8_weight", ".fp8_scale"))  # (data suffix, scale suffix) of the stored formats
_DEQUANT_CHUNK_BYTES = 32 << 20  # FP32 rows dequantized at a time


def _block_shape(data, scale):
    """Rows and columns of data covered by one scale: 128 along an axis the scales tile (DeepGEMM's blocks, the last one partial when
    the dimension is not a multiple of 128), 1 along an axis with one scale per element."""
    return (1 if scale.shape[0] == data.shape[0] else 128), (1 if scale.shape[1] == data.shape[1] else 128)


def dequantize_blocks(data, scale, dtype, out=None):
    """data * scale (expanded over its blocks) in FP32, rounded once to dtype, by chunks of whole row blocks."""
    rows, cols = data.shape
    block_rows, block_cols = _block_shape(data, scale)
    out = torch.empty((rows, cols), dtype=dtype, device=data.device) if out is None else out
    scale = scale.to(device=data.device, dtype=torch.float32)
    step = max(block_rows, _DEQUANT_CHUNK_BYTES // (4 * cols) // block_rows * block_rows)
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        part = scale[start // block_rows:-(-stop // block_rows)]
        if block_rows > 1:
            part = part.repeat_interleave(block_rows, 0)[:stop - start]
        out[start:stop] = data[start:stop].float().mul_(part.repeat_interleave(block_cols, 1)[:, :cols]).to(dtype)
    return out


class BlockScaledWeight(QTensor):
    @staticmethod
    def create(data, scale, size, stride, dtype, requires_grad=False):
        return BlockScaledWeight(_QTYPES[data.dtype], None, size, stride, data, scale, dtype, requires_grad)

    @staticmethod
    def __new__(cls, qtype, axis, size, stride, data, scale, dtype, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(cls, size, strides=stride, dtype=dtype, device=data.device, requires_grad=requires_grad)

    def __init__(self, qtype, axis, size, stride, data, scale, dtype, requires_grad=False):
        super().__init__(qtype, axis)
        self._data, self._scale = data, scale

    def __repr__(self):
        return f"BlockScaledWeight(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, qtype={self._qtype.name})"

    __str__ = __repr__

    def dequantize(self, dtype=None, device=None):
        data = self._data if device is None or self._data.device == torch.device(device) else self._data.to(device)
        return dequantize_blocks(data, self._scale, self.dtype if dtype is None else dtype)

    def linear(self, input, bias=None):
        return block_scaled_linear(self, input, bias)

    def get_quantized_subtensors(self):
        return [("data", self._data), ("scale", self._scale)]

    def set_quantized_subtensors(self, sub_tensors):
        sub_map = sub_tensors if isinstance(sub_tensors, dict) else dict(sub_tensors)
        self._data = sub_map.get("data", self._data)
        self._scale = sub_map.get("scale", self._scale)

    def __tensor_flatten__(self):
        meta = {"qtype": self._qtype.name, "size": str(list(self.size())), "stride": str(list(self.stride())), "dtype": str(self.dtype)}
        return ["_data", "_scale"], meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
        return BlockScaledWeight(_quanto_qtypes[meta["qtype"]], None, ast.literal_eval(meta["size"]), ast.literal_eval(meta["stride"]),
                                 inner_tensors["_data"], inner_tensors["_scale"], dtype)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            input = args[0] if len(args) > 0 else kwargs.get("input")
            weight = args[1] if len(args) > 1 else kwargs.get("weight")
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            if isinstance(weight, BlockScaledWeight):
                return block_scaled_linear(weight, input, bias)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        op, kwargs = op.overloadpacket, kwargs or {}
        if op is torch.ops.aten.detach:
            t = args[0]
            return BlockScaledWeight.create(op(t._data), op(t._scale), t.size(), t.stride(), t.dtype, t.requires_grad)
        if op in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype, device = kwargs.pop("dtype", t.dtype), kwargs.pop("device", t.device)
            if dtype != t.dtype:
                return t.dequantize(dtype=dtype, device=device)
            return BlockScaledWeight.create(op(t._data, device=device, **kwargs), op(t._scale, device=device, **kwargs), t.size(), t.stride(), t.dtype,
                                            t.requires_grad)
        args, kwargs = pytree.tree_map_only(BlockScaledWeight, lambda x: x.dequantize(), (args, kwargs))
        return op(*args, **kwargs)


def block_scaled_linear(weight, input, bias=None):
    """Run the checkpoint's block scale arithmetic with the local PyTorch fallback.

    This Maestro revision has no ``shared.kernels.fp8_backend`` or
    ``int8_backend.wgp_group_linear`` entry points.  Dequantize only the
    requested matrix, in bounded row chunks, and then use the normal linear
    implementation.  This is a compatibility path; it makes no throughput
    claim and can be much slower than WanGP's dedicated kernels.
    """
    dtype, data = weight.dtype, weight._data
    input = input.to(dtype)
    _note(data.dtype, "PyTorch (weights dequantized per call)")
    return torch.nn.functional.linear(input, dequantize_blocks(data, weight._scale, dtype), None if bias is None else bias.to(dtype))


_noted = set()


def _note(dtype, kernels):
    if (dtype, kernels) not in _noted:
        _noted.add((dtype, kernels))
        tqdm.write(f"[Kernels] Block-scaled {'INT8' if dtype == torch.int8 else 'FP8'} weights: {kernels}.")


class QLinearBlockScaled(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(cls, module, weights, activations=None, optimizer=None, device=None):
        dtype = module.weight.dtype if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point else torch.bfloat16
        return cls(module.in_features, module.out_features, module.bias is not None, device=device, dtype=dtype, weights=weights,
                   activations=activations, optimizer=optimizer, quantize_input=True)

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, weights=None, activations=None, optimizer=None,
                 quantize_input=True):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype, weights=weights, activations=activations,
                         optimizer=optimizer, quantize_input=quantize_input)
        self._block_scaled_dtype = dtype if dtype is not None and dtype.is_floating_point else torch.bfloat16

    def set_default_dtype(self, dtype):
        if dtype is not None and dtype.is_floating_point:
            self._block_scaled_dtype = dtype

    @property
    def qweight(self):
        return self.weight if isinstance(self.weight, BlockScaledWeight) else super().qweight

    def forward(self, input):
        return block_scaled_linear(self.qweight, input, self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.weight_qtype not in (_INT8_BLOCKS, _FP8_BLOCKS):
            return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        data, scale, bias = (state_dict.pop(prefix + name, None) for name in ("weight._data", "weight._scale", "bias"))
        for name in ("input_scale", "output_scale"):
            state_dict.pop(prefix + name, None)
        if data is None or scale is None:
            missing_keys.append(prefix + "weight")
            return
        self.weight = torch.nn.Parameter(BlockScaledWeight.create(data, scale, self.weight.size(), self.weight.stride(), self._block_scaled_dtype),
                                         requires_grad=False)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias.to(self._block_scaled_dtype), requires_grad=False)
        for name in ("input_scale", "output_scale"):
            if getattr(self, name).is_meta:
                setattr(self, name, torch.ones((), dtype=torch.float32, device=data.device))


def _collect_specs(state_dict):
    specs = []
    for key, tensor in state_dict.items():
        for data_suffix, scale_suffix in _FORMATS:
            if key.endswith(data_suffix) and getattr(tensor, "dtype", None) in _QTYPES and tensor.ndim == 2:
                base = key[:-len(data_suffix)]
                scale = state_dict.get(base + scale_suffix)
                if scale is not None and scale.ndim == 2 and scale.dtype == torch.float32:
                    block_rows = 1 if scale.shape[0] == tensor.shape[0] else 128
                    block_cols = 1 if scale.shape[1] == tensor.shape[1] else 128
                    if (tensor.shape[0] + block_rows - 1) // block_rows != scale.shape[0]:
                        continue
                    if (tensor.shape[1] + block_cols - 1) // block_cols != scale.shape[1]:
                        continue
                    specs.append((base, data_suffix, scale_suffix))
    return specs


def detect(state_dict, verboseLevel=1, metadata=None):
    specs = _collect_specs(state_dict)
    if not specs or not isinstance(metadata, dict):
        return {"matched": False, "kind": "none", "details": {}}
    formats = {
        ".weight_q": {
            "format": "nomas-int8-bf16-single-v1",
            "metadata": {"group_size": "128", "quantization": "symmetric_int8_absmax",
                         "compute_dtype": "bfloat16",
                         "source_sha256": "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"},
            "dtype": torch.int8,
        },
        ".fp8_weight": {
            "format": "pixelumm_fp8_bf16_single_v1",
            "metadata": {"block_size": "128", "weight_dtype": "float8_e4m3fn",
                         "scale_dtype": "float32",
                         "source_sha256": "0bbfe8fd44bd2ebf1059fe65079caba50fd6f52ea8e1b5b35393e3050b9b8378"},
            "dtype": torch.float8_e4m3fn,
        },
    }
    formats_seen = set()
    for base, data_suffix, scale_suffix in specs:
        data = state_dict[base + data_suffix]
        scale = state_dict[base + scale_suffix]
        expected = formats[data_suffix]
        if data.dtype != expected["dtype"]:
            return {"matched": False, "kind": "none", "details": {}}
        if any(metadata.get(key) != value for key, value in expected["metadata"].items()):
            return {"matched": False, "kind": "none", "details": {}}
        expected_block_rows = 1 if scale.shape[0] == data.shape[0] else 128
        expected_block_cols = 1 if scale.shape[1] == data.shape[1] else 128
        if scale.shape != ((data.shape[0] + expected_block_rows - 1) // expected_block_rows,
                           (data.shape[1] + expected_block_cols - 1) // expected_block_cols):
            return {"matched": False, "kind": "none", "details": {}}
        formats_seen.add(expected["format"])
    format_value = metadata.get("format")
    if format_value not in formats_seen or len(formats_seen) != 1:
        return {"matched": False, "kind": "none", "details": {}}
    if format_value == "nomas-int8-bf16-single-v1":
        try:
            layers = json.loads(metadata.get("layers", ""))
        except (TypeError, ValueError):
            return {"matched": False, "kind": "none", "details": {}}
        if not isinstance(layers, dict) or set(layers) != {base for base, _, _ in specs}:
            return {"matched": False, "kind": "none", "details": {}}
        if any(layers.get(base) != list(state_dict[base + suffix].shape) for base, suffix, _ in specs):
            return {"matched": False, "kind": "none", "details": {}}
    return {"matched": True, "kind": HANDLER_NAME,
            "details": {"count": len(specs), "formats": sorted(formats_seen), "names": [base for base, _, _ in specs[:8]]}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None, metadata=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    quant_map = {}
    for base, data_suffix, scale_suffix in _collect_specs(state_dict):
        data = state_dict.pop(base + data_suffix)
        state_dict[base + ".weight._data"] = data
        state_dict[base + ".weight._scale"] = state_dict.pop(base + scale_suffix)  # FP32, as stored
        quant_map[base] = {"weights": _QTYPES[data.dtype].name, "activations": "none"}
    return {"state_dict": state_dict, "quant_map": quant_map}
