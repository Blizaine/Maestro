import sys
from pathlib import Path

import pytest

_APP = str(Path(__file__).resolve().parents[1] / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from models.hy_nomas.camera import world_camera
from models.hy_nomas.exploration import WorldExplorationAdapter, move_frames
from models.hy_nomas.hy_nomas_main import (
    PixelUMMLicenseAcknowledgementRequired,
    require_pixelumm_acknowledgement,
    with_world_video_length_override,
)
from models.hy_nomas.world_instructions import split_instructions


def test_instruction_parser_preserves_description_camera_commands_and_events():
    description, instructions = split_instructions(
        "A fox follows the trail through a snowy forest.\n"
        "[Forward+Look_Left 2s]\n"
        "[Wait] [Event] the fox sits beside a tree"
    )
    assert description == "A fox follows the trail through a snowy forest."
    assert instructions == [
        {"text": "[Forward+Look_Left 2s]", "keys": ["w", "left"], "event": "", "seconds": 2.0},
        {"text": "[Wait] [Event] the fox sits beside a tree", "keys": [], "event": "the fox sits beside a tree", "seconds": None},
    ]
    assert move_frames(instructions[1]["seconds"] or 1.0, event=True) >= 5


def test_instruction_parser_rejects_unknown_commands_and_untagged_followups():
    with pytest.raises(ValueError, match="Unknown Hy-Nomas move command"):
        split_instructions("World\n[Teleport]")
    with pytest.raises(ValueError, match="must contain a bracketed command"):
        split_instructions("World\n[Forward]\nThen it turns left")


def test_world_camera_keeps_canonical_pose_timeline_and_segment_flag():
    poses, camera_on = world_camera(
        [{"camera": "forward"}],
        latent_boundaries=[0, 5],
        segment_frames=[20],
    )
    assert tuple(poses.shape) == (20, 4, 4)
    assert camera_on == [True]
    assert poses.isfinite().all()


def test_exploration_queue_adapter_carries_model_output_events():
    adapter = WorldExplorationAdapter("world.wgpworld")
    adapter.submit({"keys": ["w"], "text": "", "seconds": 1.0, "steps": 4, "memory": 18})
    request = adapter.next_request(timeout=0.01)
    assert request["keys"] == ["w"]
    adapter.publish("ready", world_id="abc")
    assert adapter.events.get_nowait() == {"event": "ready", "world_id": "abc"}


def test_pixelumm_noncommercial_license_gate_is_explicit():
    with pytest.raises(PixelUMMLicenseAcknowledgementRequired, match="non-commercial research or evaluation"):
        require_pixelumm_acknowledgement(False)
    require_pixelumm_acknowledgement(True)


def test_world_video_records_actual_decoded_length_for_normal_job_metadata():
    result = {"x": object(), "side_files": {".wgpworld": b"world"}}
    assert with_world_video_length_override(result, "world_video", 20) == {
        "x": result["x"],
        "side_files": {".wgpworld": b"world"},
        "overridden_inputs": {"video_length": 20},
    }
    other = {"x": object()}
    assert with_world_video_length_override(other, "world", 20) == {"x": other["x"]}
