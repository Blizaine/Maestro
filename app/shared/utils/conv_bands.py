"""Convolutions of large images by bands of rows.

For stride-one 3x3 convolutions with padding one, each output band needs one
input row of halo above and below. Optional nearest-neighbor upsampling uses
the same halo after expanding each band, so this avoids allocating a full-size
upsampled input and limits convolution workspace to a band.
"""

import functools

import torch.nn as nn
import torch.nn.functional as F

BAND_BYTES = 128 << 20


def conv_bands(x, conv, upsample=False, mode="nearest", band_bytes=BAND_BYTES):
    """Apply ``conv`` to an image, optionally after nearest x2 upsampling."""
    n, c, h, w = x.shape
    scale = 2 if upsample else 1
    rows = max(1, band_bytes // (n * c * scale * scale * w * x.element_size()))
    if rows >= h:
        return conv(F.interpolate(x, scale_factor=2.0, mode=mode) if upsample else x)

    out = None
    for start in range(0, h, rows):
        stop = min(h, start + rows)
        lo, hi = max(0, start - 1), min(h, stop + 1)
        band = F.interpolate(x[:, :, lo:hi], scale_factor=2.0, mode=mode) if upsample else x[:, :, lo:hi]
        first = 1 if start > 0 else 0
        band = band[:, :, scale * (start - lo) - first : scale * (stop - lo) + (stop < h)]
        band = conv(band)[:, :, first : first + scale * (stop - start)]
        if out is None:
            out = band.new_empty((n, band.shape[1], scale * h, band.shape[-1]))
        out[:, :, scale * start : scale * stop].copy_(band)
        del band
    return out


def band_convs(module):
    """Run eligible 3x3 convolutions in ``module`` by row bands."""
    for conv in module.modules():
        if (
            type(conv) is nn.Conv2d
            and conv.kernel_size == (3, 3)
            and conv.stride == (1, 1)
            and conv.padding == (1, 1)
            and conv.dilation == (1, 1)
            and conv.padding_mode == "zeros"
        ):
            conv.forward = functools.partial(conv_bands, conv=functools.partial(nn.Conv2d.forward, conv))
