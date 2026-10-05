"""CPU regressions for H3's opt-in clean-estimate scheduler output."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from models.minimax_h3.scheduler import MiniMaxH3Scheduler


def test_h3_clean_estimate_uses_the_data_ward_plus_sign():
    scheduler = MiniMaxH3Scheduler(shift=1.0)
    scheduler.set_timesteps(3, device="cpu")
    sample = torch.tensor([2.0, -1.0], dtype=torch.float32)
    velocity = torch.tensor([3.0, 4.0], dtype=torch.float32)
    timestep = scheduler.timesteps[0]

    result = scheduler.step(
        velocity,
        timestep,
        sample,
        return_denoised=True,
    )

    expected = sample + (1.0 - timestep) * velocity
    assert torch.equal(result.denoised, expected)
    assert torch.equal(result.prev_sample, torch.tensor([3.5, 1.0]))
    assert not torch.equal(result.prev_sample, result.denoised)
    assert torch.equal(sample, torch.tensor([2.0, -1.0]))


@pytest.mark.parametrize("solver", ["euler", "res_multistep"])
def test_opt_in_capture_is_exactly_sample_neutral_and_default_shapes_stay_unchanged(solver):
    ordinary = MiniMaxH3Scheduler(shift=12.0, solver=solver)
    captured = MiniMaxH3Scheduler(shift=12.0, solver=solver)
    ordinary.set_timesteps(6, device="cpu")
    captured.set_timesteps(6, device="cpu")
    sample = torch.linspace(-1.5, 1.5, 18, dtype=torch.float32).reshape(3, 6)
    ordinary_sample = sample.clone()
    captured_sample = sample.clone()

    for index, timestep in enumerate(ordinary.timesteps):
        velocity = torch.sin(sample + index * 0.31) + 0.2
        default_tuple = ordinary.step(
            velocity,
            timestep,
            ordinary_sample,
            return_dict=False,
        )
        preview_tuple = captured.step(
            velocity,
            timestep,
            captured_sample,
            return_dict=False,
            return_denoised=True,
        )
        expected_denoised = captured_sample + (1.0 - timestep) * velocity

        assert len(default_tuple) == 1
        assert len(preview_tuple) == 2
        assert torch.equal(preview_tuple[1], expected_denoised)
        assert torch.equal(default_tuple[0], preview_tuple[0])
        ordinary_sample = default_tuple[0]
        captured_sample = preview_tuple[0]

    assert torch.equal(ordinary_sample, captured_sample)

    default_scheduler = MiniMaxH3Scheduler()
    default_scheduler.set_timesteps(3, device="cpu")
    default_output = default_scheduler.step(
        torch.ones(1),
        default_scheduler.timesteps[0],
        torch.ones(1),
    )
    assert default_output.denoised is None
    assert tuple(default_output.keys()) == ("prev_sample",)
