"""CPU diagnostics for distinct voice inputs, row offsets and joined outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import torch

from models.minimax_h3.handoff_metadata import (
    accumulate_reference_handoffs, capture_reference_handoff,
)


def reference(kind, role, audio=None, video_rows=0, audio_rows=0):
    return SimpleNamespace(kind=kind, role=role, has_audio=audio is not None,
                           waveform=audio, num_video_rows=video_rows,
                           num_audio_rows=audio_rows)


def test_distinct_waveforms_match_reference_and_packed_row_boundaries():
    first = torch.full((2, 17), 0.25, dtype=torch.float32)
    second = torch.full((2, 23), -0.5, dtype=torch.float32)
    refs = [reference("video", "A", video_rows=8),
            reference("video", "B", video_rows=12),
            reference("audio", "A", first, audio_rows=6),
            reference("audio", "B", second, audio_rows=10)]
    source = {"schema_version": 1, "window_index": 2, "compiled_prompt": "authored"}
    result = capture_reference_handoff(source, refs, window_start_frame=327,
                                       rendered_frames=345, fps=24,
                                       continuation_audio_rows=4)
    a, b = result["encoded_references"][2:]
    assert a["reference_audio_rows"] == [0, 6]
    assert b["reference_audio_rows"] == [6, 16]
    assert a["packed_condition_audio_rows"] == [4, 10]
    assert b["packed_condition_audio_rows"] == [10, 20]
    assert a["waveform_sha256"] == hashlib.sha256(first.numpy().tobytes()).hexdigest()
    assert b["waveform_sha256"] == hashlib.sha256(second.numpy().tobytes()).hexdigest()
    assert a["waveform_sha256"] != b["waveform_sha256"]
    assert result["encoded_references"][0]["reference_video_rows"] == [0, 8]
    assert result["encoded_references"][1]["reference_video_rows"] == [8, 20]
    assert "stage" not in source
    assert result["stage"] == "runtime"


def test_embedded_soundtrack_counts_before_standalone_voice():
    refs = [reference("video", "scene", torch.zeros((2, 8)), 4, 12),
            reference("audio", "speaker", torch.ones((2, 8)), audio_rows=8)]
    result = capture_reference_handoff({"schema_version": 1}, refs,
                                       window_start_frame=0, rendered_frames=124, fps=24)
    assert result["encoded_references"][1]["reference_audio_rows"] == [12, 20]
    assert result["reference_audio_row_count"] == 20


def test_accumulation_preserves_earlier_artifact_snapshot_and_final_windows():
    first = {"schema_version": 1, "window_index": 1, "window_start_frame": 0,
             "compiled_prompt": "first"}
    second = {"schema_version": 1, "window_index": 2, "window_start_frame": 327,
              "compiled_prompt": "second"}
    previous, first_meta = accumulate_reference_handoffs([], {"h3_reference_handoffs": [first]})
    combined, final_meta = accumulate_reference_handoffs(previous,
        {"h3_reference_handoffs": [second], "other_metadata": "preserved"})
    assert len(first_meta["h3_reference_handoffs"]) == 1
    assert [x["compiled_prompt"] for x in combined] == ["first", "second"]
    assert final_meta["other_metadata"] == "preserved"
    combined[0]["compiled_prompt"] = "changed"
    assert first_meta["h3_reference_handoffs"][0]["compiled_prompt"] == "first"
    assert final_meta["h3_reference_handoffs"][0]["compiled_prompt"] == "first"


def test_repeated_window_replaces_only_its_own_final_handoff():
    first = {"schema_version": 1, "window_index": 1, "window_start_frame": 0,
             "compiled_prompt": "old"}
    updated = {**first, "compiled_prompt": "final"}
    combined, _ = accumulate_reference_handoffs([first], {"h3_reference_handoffs": [updated]})
    assert len(combined) == 1
    assert combined[0]["compiled_prompt"] == "final"


def test_unrelated_artifact_metadata_remains_untouched():
    other = {"speech_generation": {"chunks": 2}}
    previous = []
    assert accumulate_reference_handoffs(previous, other) == (previous, other)
    assert capture_reference_handoff(None, [], window_start_frame=0,
                                     rendered_frames=124, fps=24) is None
