import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.fibo2.vae import FiboVAE  # noqa: E402 - test target is under app/


class _PointwiseDecoder(nn.Module):
    """Spatially varying decoder with no tile-edge effects."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.project = nn.Conv2d(128, 3, kernel_size=1)
        with torch.no_grad():
            self.project.weight.zero_()
            self.project.weight[0, 0, 0, 0] = 0.7
            self.project.weight[0, 1, 0, 0] = -0.2
            self.project.weight[1, 2, 0, 0] = 0.5
            self.project.weight[1, 3, 0, 0] = 0.3
            self.project.weight[2, 4, 0, 0] = -0.4
            self.project.weight[2, 5, 0, 0] = 0.25
            self.project.bias.copy_(torch.tensor([0.1, -0.15, 0.05]))

    def forward(self, latents):
        decoded = torch.tanh(self.project(latents))
        return decoded.repeat_interleave(16, dim=-2).repeat_interleave(16, dim=-1)


class _SyntheticFiboVAE(FiboVAE):
    def __init__(self):
        nn.Module.__init__(self)
        self.decoder = _PointwiseDecoder()
        self._interrupt = False

    def pre_decode(self, latents):
        return latents


def _latents(height, width):
    channels = torch.arange(128, dtype=torch.float32).reshape(1, 128, 1, 1)
    rows = torch.arange(height, dtype=torch.float32).reshape(1, 1, height, 1)
    cols = torch.arange(width, dtype=torch.float32).reshape(1, 1, 1, width)
    return ((channels * 7 + rows * 3 + cols * 5) % 31 - 15) / 15


@pytest.mark.parametrize("height,width,tile_size", [
    (120, 68, 1024),  # Captured 1088x1920 portrait request: first row flush.
    (68, 120, 1024),  # Landscape counterpart, including the final row flush.
    (47, 64, 1024),   # Single partial-height final tile.
    (41, 37, 256),    # Multiple row flushes and edge-adjusted final tiles.
    (32, 32, 0),      # Square untiled control.
    (80, 80, 1024),   # Multi-tile square catches a cropped first row too.
    (120, 68, 0),     # Untiled portrait control.
    (68, 120, 0),     # Untiled landscape control.
])
def test_rectangular_decode_matches_untiled_reference(height, width, tile_size):
    vae = _SyntheticFiboVAE().eval()
    latents = _latents(height, width)

    with torch.inference_mode():
        expected = vae.decoder(vae.pre_decode(latents)).float().cpu()
        expected = expected.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
        actual = vae(latents, tile_size=tile_size)

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.uint8
    assert actual.shape == (1, 3, height * 16, width * 16)
    assert int(actual.min()) >= 0 and int(actual.max()) <= 255
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=1)
