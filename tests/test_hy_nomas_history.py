import sys
from pathlib import Path

import torch
from torch import nn

_APP = str(Path(__file__).resolve().parents[1] / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from models.hy_nomas.attention import PrefixCache, q8_prefix, quantize_heads
from models.hy_nomas.history import WorldHistory, temporal_context_frames


def test_temporal_context_keeps_first_and_recent_frames_with_spaced_samples():
    assert temporal_context_frames(10, 4) == [0, 4, 7, 9]
    assert temporal_context_frames(5, 8) == [0, 1, 2, 3, 4]
    assert temporal_context_frames(10, 1) == [9]


def test_q8_history_roundtrip_has_finite_scales_and_bounded_error():
    torch.manual_seed(7)
    values = torch.randn((1, 4, 2, 64), dtype=torch.float32)
    quantized, scales = quantize_heads(values)
    assert quantized.dtype == torch.int8
    assert scales.dtype == torch.float16
    assert quantized.shape == (4, 2, 64)
    assert scales.shape == (4, 2, 2)
    decoded = q8_prefix(quantized, scales, quantized, scales).to(device="cpu", dtype=torch.float32)[0]
    assert torch.isfinite(scales).all()
    assert (decoded - values).abs().max() < 0.03


def test_world_history_archives_and_reloads_only_the_selected_block_on_cpu():
    torch.manual_seed(11)
    history = WorldHistory([nn.Identity()], heads=2, head_dim=64, device=torch.device("cpu"), past_frames=2)
    keys = torch.randn((1, 6, 2, 64), dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    history.write_from(frame_start=0, frames=2, image_tokens=4, tokens=6)
    history.store(0, keys, range(0, 2))
    history.store(0, values, range(0, 2), values=True)
    history.flush(0)
    history.commit()

    assert history.total == 2
    assert history.stored_frames == 2
    assert history.seen_frames == 2
    prefix_tensors = history.block_kv(0)
    assert prefix_tensors is not None
    assert all(tensor.device.type == "cpu" for tensor in prefix_tensors)
    prefix = PrefixCache(*prefix_tensors)
    decoded_keys, decoded_values = prefix.to(device="cpu", dtype=torch.float32)
    assert decoded_keys.shape == (1, 6, 2, 64)
    assert decoded_values.shape == (1, 6, 2, 64)
    assert (decoded_keys - keys.float()).abs().max() < 0.12
    assert (decoded_values - values.float()).abs().max() < 0.12

    history.release_working_set()
    assert history.block_kv(0) is None
    assert history.stored_frames == 2
