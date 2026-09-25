"""Optional LTX projection sharing with normal module-call fallbacks."""

from __future__ import annotations

import torch

from shared.kernels.comfy_kitchen import project_int8_convrot_many


def project_many(modules, x: torch.Tensor):
    """Share bounded INT8 input quantization when every ConvRot layer is safe."""

    layers = tuple(modules)
    outputs = project_int8_convrot_many(layers, x)
    if outputs is not None:
        return outputs
    return tuple(module(x) for module in layers)


__all__ = ["project_many"]
