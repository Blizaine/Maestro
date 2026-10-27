"""Optional MiniMax H3 3D latent upscaler.

The network layout and temporal chunking are adapted from LBH-123-AI's
``Comfyui_Minimax_h3_latent_Upscaler`` inference implementation. The optional
BF16 weights are distributed separately by the author under Apache-2.0; see
``H3-LATENT-UPSCALER-NOTICE.md`` for pinned provenance and both notices.
"""

from __future__ import annotations

import gc
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


LATENT_UPSCALER_REPO = "LBH-123-AI/Minimax_h3_latent_Upscaler"
LATENT_UPSCALER_REVISION = "09592c6221ec95cc8e0fae67842e34926c4e668b"
LATENT_UPSCALER_FILE = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
LATENT_UPSCALER_SHA256 = (
    "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6"
)
LATENT_UPSCALER_SOURCE_REVISION = "98934c0b5f4a213323c4d1d0e47eb3d2f27cb743"
LATENT_UPSCALER_TEMPORAL_CHUNK = 32


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, embedding_channels: int):
        super().__init__()
        self.out_channels = channels
        self.in_layers = nn.Sequential(
            _normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_channels, 2 * channels),
        )
        self.out_norm = _normalization(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)
        self.skip = nn.Identity()

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.in_layers(value)
        embedded = self.emb_layers(embedding).to(hidden.dtype)
        while embedded.ndim < hidden.ndim:
            embedded = embedded[..., None]
        scale, shift = torch.chunk(embedded, 2, dim=1)
        hidden = self.out_norm(hidden) * (1.0 + scale) + shift
        return self.skip(value) + self.out_layers(hidden)


class _TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm(value))
        hidden = self.dwconv(hidden)
        hidden = self.pwconv(hidden)
        return value + hidden


class MiniMaxH3LatentUpscaler(nn.Module):
    """24-channel latent resize network, preserving its trained state-dict keys."""

    def __init__(
        self,
        *,
        in_channels: int = 24,
        blocks: int = 12,
        channels: int = 512,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embedding_channels = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.in_blocks = self._build_blocks(
            blocks, channels, embedding_channels, temporal_every, temporal_kernel
        )
        self.out_blocks = self._build_blocks(
            blocks, channels, embedding_channels, temporal_every, temporal_kernel
        )
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)
        self._interrupt = False

    @staticmethod
    def _build_blocks(
        count: int,
        channels: int,
        embedding_channels: int,
        temporal_every: int,
        temporal_kernel: int,
    ) -> nn.ModuleList:
        blocks: list[nn.Module] = []
        for index in range(count):
            blocks.append(_ResBlockEmb3D(channels, embedding_channels))
            if temporal_every > 0 and index % temporal_every == 0:
                blocks.append(_TemporalConv(channels, temporal_kernel))
        return nn.ModuleList(blocks)

    def forward(
        self,
        value: torch.Tensor,
        scale: float | None = None,
        *,
        target_size: tuple[int, int, int] | None = None,
        chunk_temporal: bool = True,
    ) -> torch.Tensor:
        if value.ndim != 5 or int(value.shape[1]) != 24:
            raise ValueError(
                "MiniMax H3 latent upscaler expects [batch, 24, time, height, width]."
            )
        if target_size is not None:
            size = tuple(int(item) for item in target_size)
        elif scale is not None and float(scale) > 0.0:
            size = tuple(int(round(dimension * float(scale))) for dimension in value.shape[-3:])
        else:
            raise ValueError("MiniMax H3 latent upscaler needs a positive scale or target size.")
        if any(dimension <= 0 for dimension in size):
            raise ValueError("MiniMax H3 latent upscaler target dimensions must be positive.")
        if size == tuple(int(item) for item in value.shape[-3:]):
            return value

        time = int(value.shape[2])
        temporal_kernel = next(
            (
                int(block.dwconv.kernel_size[0])
                for block in self.in_blocks
                if isinstance(block, _TemporalConv)
            ),
            0,
        )
        if not chunk_temporal or time <= LATENT_UPSCALER_TEMPORAL_CHUNK or not temporal_kernel:
            return self._forward_segment(value, scale, size)

        overlap = temporal_kernel
        chunk_size = LATENT_UPSCALER_TEMPORAL_CHUNK
        padded = F.pad(value, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        output = value.new_zeros(
            (value.shape[0], value.shape[1], time, size[-2], size[-1])
        )
        weight_sum = value.new_zeros((1, 1, time, 1, 1))

        start = 0
        while start < time:
            if self._interrupt:
                raise InterruptedError("MiniMax H3 latent upscaling was interrupted.")
            segment_start = start
            segment_end = min(time, start + chunk_size)
            output_start = max(0, segment_start - overlap)
            output_end = min(time, segment_end + overlap)
            left = max(0, output_start - overlap)
            right = min(time + 2 * overlap, output_end + overlap)
            segment = padded[:, :, left:right].contiguous()
            segment_size = (right - left, size[-2], size[-1])
            segment_output = self._forward_segment(segment, scale, segment_size)

            slice_start = (output_start + overlap) - left
            slice_end = slice_start + (output_end - output_start)
            valid = segment_output[:, :, slice_start:slice_end]
            valid_count = output_end - output_start
            weights = value.new_ones(valid_count)
            if segment_start > output_start:
                blend_count = segment_start - output_start
                weights[:blend_count] = torch.arange(
                    1, blend_count + 1, device=value.device, dtype=value.dtype
                ) / (blend_count + 1)
            if output_end > segment_end:
                blend_count = output_end - segment_end
                weights[-blend_count:] = torch.arange(
                    blend_count, 0, -1, device=value.device, dtype=value.dtype
                ) / (blend_count + 1)
            weight_view = weights.view(1, 1, valid_count, 1, 1)
            output[:, :, output_start:output_end] += valid * weight_view
            weight_sum[:, :, output_start:output_end] += weight_view
            start += chunk_size
            del segment, segment_output, valid, weights, weight_view
            if start % (chunk_size * 4) == 0:
                gc.collect()
        return output / weight_sum.clamp(min=1e-8)

    def _forward_segment(
        self,
        value: torch.Tensor,
        scale: float | None,
        size: tuple[int, int, int],
    ) -> torch.Tensor:
        scale_value = 0.0 if scale is None else float(scale) - 1.0
        scale_embedding = value.new_tensor([[scale_value]])
        embedding = self.embed(scale_embedding)
        hidden = self.conv_in(value)
        for block in self.in_blocks:
            if isinstance(block, _ResBlockEmb3D):
                hidden = block(hidden, embedding.expand(hidden.shape[0], -1))
            else:
                hidden = block(hidden)
        hidden = F.interpolate(hidden, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            if isinstance(block, _ResBlockEmb3D):
                hidden = block(hidden, embedding.expand(hidden.shape[0], -1))
            else:
                hidden = block(hidden)
        return self.conv_out(F.silu(self.norm_out(hidden)))


def verify_latent_upscaler_checkpoint(
    filename: str,
    *,
    expected_sha256: str = LATENT_UPSCALER_SHA256,
) -> None:
    """Check the optional pinned asset before mapping its large tensor file."""

    digest = hashlib.sha256()
    with open(filename, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != str(expected_sha256).lower():
        raise ValueError(
            "MiniMax H3 latent upscaler checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual}."
        )


def validate_latent_upscaler_header(
    filename: str,
    model: MiniMaxH3LatentUpscaler,
) -> None:
    """Compare safetensors header shapes with the meta-device network first.

    ``safe_open.get_slice().get_shape()`` reads only tensor header metadata;
    this catches architecture drift before MMGP maps weights onto a device.
    """

    from safetensors import safe_open

    expected = {
        key: tuple(int(size) for size in tensor.shape)
        for key, tensor in model.state_dict().items()
    }
    with safe_open(str(filename), framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
        prefix = "upscaler."
        use_prefix = any(key.startswith(prefix) for key in keys)
        actual = {
            key[len(prefix):] if use_prefix and key.startswith(prefix) else key:
            tuple(int(size) for size in handle.get_slice(key).get_shape())
            for key in keys
        }

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        (key, expected[key], actual[key])
        for key in set(expected) & set(actual)
        if expected[key] != actual[key]
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing tensors: {', '.join(missing[:4])}")
        if unexpected:
            details.append(f"unexpected tensors: {', '.join(unexpected[:4])}")
        if mismatched:
            key, want, got = mismatched[0]
            details.append(f"shape mismatch for {key}: expected {want}, got {got}")
        raise ValueError(
            "MiniMax H3 latent upscaler checkpoint does not match the pinned "
            "3D network (" + "; ".join(details) + ")."
        )


def load_h3_latent_upscaler(filename: str) -> MiniMaxH3LatentUpscaler:
    """Load the pinned BF16 upscaler through MMGP's mapped-weight loader."""

    from accelerate import init_empty_weights
    from mmgp import offload

    verify_latent_upscaler_checkpoint(filename)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=True):
            model = MiniMaxH3LatentUpscaler()
    finally:
        torch.set_default_dtype(previous_dtype)
    validate_latent_upscaler_header(filename, model)

    def select_upscaler_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prefix = "upscaler."
        if any(key.startswith(prefix) for key in state_dict):
            return {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        return state_dict

    offload.load_model_data(
        model,
        filename,
        writable_tensors=False,
        default_dtype=torch.bfloat16,
        preprocess_sd=select_upscaler_weights,
    )
    model._model_dtype = torch.bfloat16
    return model.eval().requires_grad_(False)


__all__ = [
    "LATENT_UPSCALER_FILE",
    "LATENT_UPSCALER_REPO",
    "LATENT_UPSCALER_REVISION",
    "LATENT_UPSCALER_SHA256",
    "LATENT_UPSCALER_SOURCE_REVISION",
    "MiniMaxH3LatentUpscaler",
    "load_h3_latent_upscaler",
    "validate_latent_upscaler_header",
    "verify_latent_upscaler_checkpoint",
]
