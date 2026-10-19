import pytest
import torch
import torch.nn.functional as F
from torch import nn

from app.shared.utils.conv_bands import conv_bands


@pytest.mark.parametrize("upsample", [False, True])
@pytest.mark.parametrize("mode", ["nearest", "nearest-exact"])
def test_conv_bands_matches_full_convolution_and_gradients(upsample, mode):
    torch.manual_seed(17)
    x = torch.randn(2, 3, 7, 9, dtype=torch.float64, requires_grad=True)
    conv = nn.Conv2d(3, 4, 3, padding=1, dtype=torch.float64)
    reference = conv(F.interpolate(x, scale_factor=2, mode=mode) if upsample else x)
    expected_grad = torch.autograd.grad(reference.square().mean(), (x, conv.weight, conv.bias))

    banded_x = x.detach().clone().requires_grad_()
    banded = conv_bands(banded_x, conv, upsample=upsample, mode=mode, band_bytes=1)
    actual_grad = torch.autograd.grad(banded.square().mean(), (banded_x, conv.weight, conv.bias))

    torch.testing.assert_close(banded, reference, rtol=1e-10, atol=1e-10)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
