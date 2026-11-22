"""Hardware-independent checks for MiniMax H3 aligned guide conditioning."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3.minimax_h3_main import (
    MiniMaxH3Model,
    _aligned_guide_visual_condition,
)
from models.minimax_h3.ref2va import build_ref2va_packed_sequence


def test_aligned_guide_slices_matching_window_frames_after_history():
    source = torch.linspace(-1, 1, 30).view(1, 30, 1, 1).expand(3, -1, 4, 4).contiguous()
    condition = _aligned_guide_visual_condition(
        source,
        history_count=5,
        target_frame_num=22,
        height=4,
        width=4,
    )
    assert condition["anchor"] == "first"
    assert condition["keep_all_latents"] is True
    assert torch.equal(condition["source"], source[:, 5:27])
    with pytest.raises(ValueError, match="shorter than the requested target"):
        _aligned_guide_visual_condition(
            source[:, :26], 5, 22, height=4, width=4,
        )


def test_aligned_guide_retains_all_target_latents_and_matches_target_rope_times():
    class Posterior:
        def sample(self, *, generator):
            return torch.zeros((1, 24, 7, 4, 4), dtype=torch.float32)

    class VAE:
        def encode_condition(self, pixels, *, keep_all_latents=False):
            assert keep_all_latents is True
            assert pixels.shape[2] == 22
            return Posterior()

    class Scheduler:
        @staticmethod
        def scale_noise(clean, _noise_aug, _noise):
            return clean

    model = object.__new__(MiniMaxH3Model)
    model.device = torch.device("cpu")
    model._interrupt = False
    model.transformer = SimpleNamespace(
        config=SimpleNamespace(patch_size=(1, 2, 2)),
    )
    model.vae = VAE()
    model.scheduler = Scheduler()

    guide = _aligned_guide_visual_condition(
        torch.zeros((3, 22, 128, 128), dtype=torch.float32),
        history_count=0,
        target_frame_num=22,
        height=128,
        width=128,
    )
    rows, anchors = model._encode_visual_conditions(
        [guide],
        latent_height=4,
        latent_width=4,
        generator=torch.Generator().manual_seed(3),
        height=128,
        width=128,
    )
    assert anchors == (("first", 7),)
    assert rows.shape == (28, 96)

    layout = build_ref2va_packed_sequence(
        torch.tensor([1, 1]),
        [],
        num_latent_frames=7,
        latent_height=4,
        latent_width=4,
        num_audio_latents=0,
        patch_size=model.patch_size,
        keyframe_anchors=anchors,
    )
    condition_video = layout.video_indices[: layout.num_condition_video_rows]
    target_video = layout.video_indices[layout.num_condition_video_rows :]
    guide_times = layout.position_ids[condition_video, 0].view(7, 4)
    target_times = layout.position_ids[target_video, 0].view(7, 4)
    assert torch.equal(guide_times, target_times)


def test_hyperflow_flash_rejects_other_h3_accelerator_loras():
    from models.minimax_h3.turbo import MINIMAX_H3_TURBO_PRESETS

    model = object.__new__(MiniMaxH3Model)
    model.pika_flex = False
    model.hyperflow_flash = True
    accelerator = MINIMAX_H3_TURBO_PRESETS[0]["filename"]
    with pytest.raises(ValueError, match="does not support H3 Turbo"):
        model.configure_special_loras([accelerator], [1.0])
