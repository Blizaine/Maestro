"""Source timing and fixed-schedule regressions for the November H3 ports."""

from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3 import regenerate_ic, viggle_remake
from models.minimax_h3.hyperflow_recipe import hyperflow_sigmas, step_conditioning
from models.minimax_h3.packing import build_row_timesteps
from models.minimax_h3.scheduler import MiniMaxH3Scheduler
from services.viggle_media import probe_control_video
from shared.utils.audio_video import extract_audio_tracks


@pytest.fixture
def source_clip(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required")
    source = tmp_path / "draft.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=96x64:r=24:d=6",
        "-f", "lavfi", "-i", "sine=frequency=400:duration=6",
        "-c:v", "libx264", "-c:a", "pcm_s16le", str(source),
    ], check=True)
    return source


def test_remake_canonicalizes_trim_and_keeps_soundtrack(source_clip):
    original = source_clip.read_bytes()
    request = {
        "prompt": "Replace the driver.",
        "video_guide": str(source_clip),
        "image_refs": ["frame.png"],
        "viggle_remake_frame_times": [2.5],
        "viggle_trim_start": 1,
        "viggle_trim_end": 5,
    }
    with viggle_remake.prepare_remake_media(
        request, "swap", resolve_media=Path, aborted=lambda: False, update=lambda _: None,
    ) as runtime:
        trimmed = Path(runtime["video_guide"])
        assert trimmed != source_clip
        assert probe_control_video(trimmed)["duration_seconds"] == pytest.approx(4, abs=0.04)
        assert extract_audio_tracks(str(trimmed), query_only=True) == 1
        assert runtime["video_length"] == 96
        assert runtime["audio_prompt_type"] == "K"
        assert "<Picture 1> is an edited target frame at 1.500 seconds" in runtime["prompt"]
    assert not trimmed.exists()
    assert source_clip.read_bytes() == original
    assert request["viggle_remake_frame_times"] == [2.5]


def test_remake_rejects_ambiguous_frame_order():
    request = {
        "video_guide": "draft.mp4", "image_refs": ["a.png", "b.png"],
        "viggle_remake_frame_times": [2, 1],
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        viggle_remake.normalize_request(request)


def test_regenerate_uses_source_aspect_and_timing(source_clip):
    with regenerate_ic.prepare_media(
        {"video_guide": str(source_clip)},
        resolve_media=Path, aborted=lambda: False, update=lambda _: None,
    ) as runtime:
        assert runtime["resolution"] == "2176x1440"
        assert runtime["video_length"] == 144
        assert runtime["audio_prompt_type"] == "K"
        assert runtime["num_inference_steps"] == 3
        assert runtime["video_guide"] == str(source_clip)
    assert regenerate_ic.target_canvas(720, 1280) == (1440, 2560)


def test_hyperflow_uses_distinct_video_audio_endpoints_and_fixed_reference():
    layout = SimpleNamespace(
        sequence_length=4, video_indices=torch.tensor([0, 1]), audio_indices=torch.tensor([2]),
        num_condition_video_rows=1, num_condition_audio_rows=0,
        num_target_condition_video_rows=0, num_target_condition_audio_latents=0,
    )
    timesteps, row_indices = build_row_timesteps(layout, 0, 0, 0.999, 1)
    video_endpoint = 1 - hyperflow_sigmas(12)[1]
    audio_endpoint = 1 - hyperflow_sigmas(3)[1]
    offsets = torch.tensor([[0.999, 0.999], [0, video_endpoint], [0, audio_endpoint]])
    rows = torch.tensor([10, 20, 30])
    _, remap, blocks, final = step_conditioning(
        (offsets, rows.view(1, 3, 1), rows.view(3, 1)),
        layout, timesteps, row_indices, next_video=video_endpoint,
        next_audio=audio_endpoint, condition_video=0.999,
    )
    assert blocks[0, remap, 0].tolist() == [10, 20, 30, 20]
    assert final[remap, 0].tolist() == [10, 20, 30, 20]
    assert hyperflow_sigmas(12, tail=3) == hyperflow_sigmas(12)[-4:]


@pytest.mark.parametrize(("sigmas", "evaluations"), [
    (hyperflow_sigmas(12), 8), (hyperflow_sigmas(3, tail=3), 3),
    (viggle_remake.sigmas("edit"), 6), (viggle_remake.sigmas("swap", 3), 6),
])
def test_fixed_schedules_are_accepted_by_the_native_scheduler(sigmas, evaluations):
    scheduler = MiniMaxH3Scheduler()
    scheduler.set_timesteps(sigmas=sigmas, device="cpu")
    assert len(scheduler.timesteps) == evaluations
    assert scheduler.sigmas[-1] == 0
