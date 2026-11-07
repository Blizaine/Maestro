import json
import hashlib
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.director.planners.base import BasePlanner  # noqa: E402
from services.director.schema import ProductionPlan  # noqa: E402
from services.director_pipeline import (  # noqa: E402
    _director_title_card_schema,
    _extend_director_title_card_schema,
    _interleave_director_title_cards,
    _resolve_director_title_card_anchors,
    normalize_director_title_cards,
)
from services import director_pipeline as director_pipeline_module  # noqa: E402
from services.title_cards import _resolve_ffmpeg, _resolve_ffprobe  # noqa: E402
from services.production_review import director_take_signature  # noqa: E402


class _PlannerHarness(BasePlanner):
    def plan(self, **kwargs):
        return ProductionPlan(skill_type="short_film", shots=[])


def test_base_parser_detaches_wrapper_cards_and_preserves_exact_copy():
    planner = _PlannerHarness()
    wrapper_cards = [{"text": "  The Last\nLight  ", "role": "opening"}]
    row_cards = [{"text": "Between acts", "role": "intertitle", "after_clip": 2}]
    response = {
        "shots": [{
            "scene_goal": "Keep the story row",
            "metadata": {"continuity_group": "A", "title_cards": row_cards},
            "title_cards": [{"text": "Duplicate row card", "role": "end"}],
        }],
        "title_cards": wrapper_cards,
    }

    rows = planner._parse_json_response(json.dumps(response))

    assert rows == [{
        "scene_goal": "Keep the story row",
        "metadata": {"continuity_group": "A"},
    }]
    assert planner.get_plan_title_cards() == [
        *wrapper_cards,
        {"text": "Duplicate row card", "role": "end"},
        *row_cards,
    ]


def test_planner_card_metadata_is_bounded_deduplicated_and_reset_per_plan():
    planner = _PlannerHarness()
    response = json.dumps({
        "shots": [{
            "scene_goal": "Shot",
            "title_cards": [{"text": "One", "role": "opening", "style": "classic"}],
        }],
    })
    repeated_defaults = json.dumps({
        "title_cards": [{"text": "One", "role": "opening", "style": "bold"}],
        "shots": [],
    })

    planner._parse_json_response(response)
    planner._parse_json_response(repeated_defaults)
    assert planner.get_plan_title_cards() == [
        {"text": "One", "role": "opening", "style": "classic"},
    ]

    planner.reset_plan_metadata()
    assert planner.get_plan_title_cards() == []


def test_malformed_planner_title_cards_are_ignored_with_warning(capsys):
    planner = _PlannerHarness()

    rows = planner._parse_json_response(json.dumps({
        "shots": [],
        "title_cards": [None, "not a card"],
    }))

    assert rows == []
    assert planner.get_plan_title_cards() == []
    assert "Ignoring malformed title-card suggestion" in capsys.readouterr().out


def test_production_plan_round_trips_detached_title_cards():
    cards = [
        {"text": "Film title", "role": "opening"},
        {"text": "A new act", "role": "intertitle", "after_clip": 1},
    ]
    plan = ProductionPlan(skill_type="short_film", shots=[], title_cards=cards)

    restored = ProductionPlan.from_dict(plan.to_dict())

    assert restored.title_cards == cards


def test_explicit_normalizer_preserves_copy_and_validates_anchor_contract():
    card = {
        "text": "  A\nNew Chapter  ",
        "subtitle": "At dawn",
        "role": "intertitle",
        "after_clip": 2,
    }

    result = normalize_director_title_cards(
        [card],
        clip_count=3,
        pipeline_type="short_film_story",
        has_source_audio=False,
        seamless=False,
    )

    assert result[0]["text"] == card["text"]
    assert result[0]["subtitle"] == card["subtitle"]
    assert result[0]["after_clip"] == 2
    with pytest.raises(ValueError, match="cannot have an after_clip"):
        normalize_director_title_cards([
            {"text": "Opening", "role": "opening", "after_clip": 1},
        ])
    with pytest.raises(ValueError, match="require a positive"):
        normalize_director_title_cards([
            {"text": "Chapter", "role": "intertitle"},
        ])
    with pytest.raises(ValueError, match="between 1 and 3"):
        normalize_director_title_cards(
            [{"text": "Chapter", "role": "intertitle", "after_clip": 4}],
            clip_count=3,
        )


@pytest.mark.parametrize(
    "flags",
    [
        {"pipeline_type": "music_video", "has_source_audio": False, "seamless": False},
        {"pipeline_type": "short_film_story", "has_source_audio": True, "seamless": False},
        {"pipeline_type": "short_film_story", "has_source_audio": False, "seamless": True},
    ],
)
def test_intertitles_reject_audio_locked_and_seamless_timelines(flags):
    with pytest.raises(ValueError, match="reserved-timeline"):
        normalize_director_title_cards(
            [{"text": "Act Two", "role": "intertitle", "after_clip": 1}],
            clip_count=2,
            **flags,
        )


def test_advisory_cards_skip_malformed_rows_and_locked_intertitles():
    cards = normalize_director_title_cards(
        [
            {"text": "Opening", "role": "opening"},
            {"text": "Missing anchor", "role": "intertitle"},
            {"text": "Act Two", "role": "intertitle", "after_clip": 1},
        ],
        clip_count=2,
        advisory=True,
        pipeline_type="music_video",
        has_source_audio=True,
        seamless=False,
    )

    assert [card["role"] for card in cards] == ["opening"]


def test_native_timeline_anchor_maps_to_final_clip_boundary():
    cards = [{"text": "Intermission", "role": "intertitle", "after_clip": 3}]
    plans = [
        {"_director_source_clip_indices": [0]},
        {"_director_source_clip_indices": [1, 2]},
        {"_director_source_clip_indices": [3]},
    ]

    resolved = _resolve_director_title_card_anchors(
        cards,
        source_clip_count=4,
        clip_plans=plans,
        planned_clips=[{}, {}, {}],
    )

    assert resolved[0]["after_clip"] == 2
    with pytest.raises(ValueError, match="merged into a later clip"):
        _resolve_director_title_card_anchors(
            [{"text": "Too early", "role": "intertitle", "after_clip": 2}],
            source_clip_count=4,
            clip_plans=plans,
            planned_clips=[{}, {}, {}],
        )


def test_intertitles_interleave_without_becoming_video_prompts():
    cards = [
        {"text": "Open", "role": "opening"},
        {"text": "Act Two", "role": "intertitle", "after_clip": 1},
        {"text": "Finale", "role": "intertitle", "after_clip": 3},
        {"text": "End", "role": "end"},
    ]
    rendered = [
        {"asset_path": "opening.mp4"},
        {"asset_path": "act-two.mp4"},
        {"asset_path": "finale.mp4"},
        {"asset_path": "end.mp4"},
    ]

    assert _interleave_director_title_cards(
        ["clip-1.mp4", "clip-2.mp4", "clip-3.mp4"], cards, rendered,
    ) == [
        "clip-1.mp4", "act-two.mp4", "clip-2.mp4", "clip-3.mp4", "finale.mp4",
    ]


def test_llm_card_schema_is_optional_and_does_not_mutate_planner_schema():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"scene_goal": {"type": "string"}},
            "additionalProperties": False,
        },
    }

    extended = _extend_director_title_card_schema(schema)

    assert "title_cards" not in schema["items"]["properties"]
    assert "title_cards" in extended["items"]["properties"]
    assert extended["items"]["properties"]["title_cards"] == _director_title_card_schema()


def _ffmpeg_run(ffmpeg, args):
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _make_sync_fixture_master(ffmpeg, path):
    _ffmpeg_run(ffmpeg, [
        "-f", "lavfi", "-i", "color=c=red:s=128x64:r=12:d=0.5",
        "-f", "lavfi", "-i", "color=c=blue:s=128x64:r=12:d=0.75",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000:duration=0.5",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.75",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[masterv];"
        "[3:a]aformat=channel_layouts=stereo[sine];"
        "[2:a][sine]concat=n=2:v=0:a=1[mastera]",
        "-map", "[masterv]", "-map", "[mastera]",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-r", "12", "-c:a", "aac", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(path),
    ])


def _make_audio_clip(ffmpeg, source, target, *, start, duration):
    _ffmpeg_run(ffmpeg, [
        "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
        "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-r", "12", "-c:a", "aac", "-ar", "48000",
        "-ac", "2", "-shortest", "-movflags", "+faststart", str(target),
    ])


class _FFmpegConcatShim:
    """Use real FFmpeg concat filters without importing the GPU runtime."""

    def __init__(self, ffmpeg):
        self.ffmpeg = ffmpeg
        self.server_config = {
            "video_output_codec": "libx264_8",
            "video_container": "mp4",
        }

    def concatenate_multi_clip_videos(
        self,
        video_files,
        output_path,
        audio_path=None,
        *,
        abort_callback=None,
        **kwargs,
    ):
        if audio_path:
            raise AssertionError("This fixture expects embedded audio to be preserved.")
        if abort_callback and abort_callback():
            return False
        command = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        ]
        for path in video_files:
            command.extend(["-i", str(path)])
        video_inputs = "".join(f"[{index}:v:0]" for index in range(len(video_files)))
        audio_inputs = "".join(f"[{index}:a:0]" for index in range(len(video_files)))
        filter_graph = (
            f"{video_inputs}concat=n={len(video_files)}:v=1:a=0[v];"
            f"{audio_inputs}concat=n={len(video_files)}:v=0:a=1[a]"
        )
        command.extend([
            "-filter_complex", filter_graph,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", "12", "-vsync", "cfr",
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", str(output_path),
        ])
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return (
            result.returncode == 0
            and Path(output_path).is_file()
            and Path(output_path).stat().st_size > 0
        )


def _media_duration(ffprobe, path):
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return float(result.stdout.strip())


def _audio_rms(ffmpeg, path, *, start, duration):
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
            "-map", "0:a:0", "-af",
            f"atrim=start={start:.6f}:duration={duration:.6f},asetpts=PTS-STARTPTS",
            "-ac", "1", "-ar", "48000", "-f", "f32le", "-",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    sample_count = len(result.stdout) // 4
    samples = struct.unpack(f"<{sample_count}f", result.stdout[: sample_count * 4])
    assert samples
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _mean_rgb_at(ffmpeg, path, *, frame, width=128, height=64):
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
            "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert len(result.stdout) == width * height * 3
    return tuple(
        sum(result.stdout[channel::3]) / (width * height)
        for channel in range(3)
    )


def _decode_entire_video(ffmpeg, path):
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
            "-f", "null", "-",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _title_card_params():
    cards = [
        {
            "text": "OPENING",
            "role": "opening",
            "duration_seconds": 0.5,
            "background_color": "#000000",
            "text_color": "#FFFFFF",
        },
        {
            "text": "THE END",
            "role": "end",
            "duration_seconds": 0.5,
            "background_color": "#000000",
            "text_color": "#FFFFFF",
        },
    ]
    return {
        "pipeline_type": "short_film_story",
        "seamless": True,
        "fps": 12,
        "video_params": {"resolution": "128x64"},
        "_director_video_execution_profile": {
            "normalized_resolution": "128x64",
            "fps": 12,
        },
        "title_cards": cards,
        "_director_title_cards_resolved": cards,
    }


def test_native_master_wrapper_keeps_source_audio_aligned_with_video(tmp_path, monkeypatch):
    try:
        ffmpeg = _resolve_ffmpeg(None)
        ffprobe = _resolve_ffprobe(ffmpeg)
    except Exception as error:
        pytest.skip(f"FFmpeg fixture tools are unavailable: {error}")
    if not ffprobe:
        pytest.skip("FFprobe is unavailable for media verification.")

    master_path = tmp_path / "source_multiclip.mp4"
    _make_sync_fixture_master(ffmpeg, master_path)
    original_duration = _media_duration(ffprobe, master_path)
    assert original_duration == pytest.approx(1.25, abs=0.03)

    concat = _FFmpegConcatShim(ffmpeg)
    monkeypatch.setattr(director_pipeline_module, "_wgp", concat)
    params = _title_card_params()
    result = director_pipeline_module._apply_director_title_cards_to_outputs(
        "wrapper-test",
        params,
        [master_path.name],
        [{}],
        out_dir=str(tmp_path),
    )

    finished = tmp_path / result[0]
    assert finished != master_path
    assert master_path.is_file()
    assert _media_duration(ffprobe, finished) == pytest.approx(2.25, abs=0.06)
    _decode_entire_video(ffmpeg, finished)

    # The image changes from red to blue at source time 0.5s and the sine
    # begins at the same source time. The opening card moves both to 1.0s.
    before_cut_rgb = _mean_rgb_at(ffmpeg, finished, frame=9)
    after_cut_rgb = _mean_rgb_at(ffmpeg, finished, frame=15)
    assert before_cut_rgb[0] > before_cut_rgb[2] * 2
    assert after_cut_rgb[2] > after_cut_rgb[0] * 2
    assert _audio_rms(ffmpeg, finished, start=0.08, duration=0.3) < 0.003
    assert _audio_rms(ffmpeg, finished, start=0.68, duration=0.25) < 0.01
    assert _audio_rms(ffmpeg, finished, start=1.08, duration=0.35) > 0.025
    assert _audio_rms(ffmpeg, finished, start=1.82, duration=0.3) < 0.003


def test_rejoin_wraps_one_card_set_and_never_overwrites_existing_master(
    tmp_path,
    monkeypatch,
):
    try:
        ffmpeg = _resolve_ffmpeg(None)
        ffprobe = _resolve_ffprobe(ffmpeg)
    except Exception as error:
        pytest.skip(f"FFmpeg fixture tools are unavailable: {error}")
    if not ffprobe:
        pytest.skip("FFprobe is unavailable for media verification.")

    source_master = tmp_path / "source_multiclip.mp4"
    _make_sync_fixture_master(ffmpeg, source_master)
    red_clip = tmp_path / "clip_red.mp4"
    blue_clip = tmp_path / "clip_blue.mp4"
    _make_audio_clip(ffmpeg, source_master, red_clip, start=0.0, duration=0.5)
    _make_audio_clip(ffmpeg, source_master, blue_clip, start=0.5, duration=0.75)
    existing_master = tmp_path / "existing_master.mp4"
    shutil.copy2(source_master, existing_master)
    original_master_bytes = existing_master.read_bytes()
    pipeline_file = tmp_path / "_director_pipeline_rejoin-test.json"
    pipeline_file.write_text("{}", encoding="utf-8")

    params = _title_card_params()
    params.update({
        "seamless": False,
        "_director_title_card_source_clip_count": 2,
        "_director_title_cards_resolved": [
            {"text": "OPENING", "role": "opening", "duration_seconds": 0.5},
            {"text": "THE END", "role": "end", "duration_seconds": 0.5},
        ],
    })
    state = {
        "pipeline_id": "rejoin-test",
        "pipeline_type": "short_film_story",
        "seamless": False,
        "shot_image_policy": "prompt_only",
        "output_files": [existing_master.name],
        "_params_snapshot": params,
        "clips": [
            {
                "video_filename": red_clip.name,
                "planned_clip": {"start": 0.0, "end": 0.5, "duration_sec": 0.5},
                "video_stale": False,
            },
            {
                "video_filename": blue_clip.name,
                "planned_clip": {"start": 0.5, "end": 1.25, "duration_sec": 0.75},
                "video_stale": False,
            },
        ],
    }
    saved_state = {}

    def capture_update(out_dir, pid, updater):
        updated = dict(state)
        updater(updated)
        saved_state["value"] = updated
        return updated

    monkeypatch.setattr(director_pipeline_module, "_wgp", _FFmpegConcatShim(ffmpeg))
    monkeypatch.setattr(director_pipeline_module, "load_pipeline_state", lambda *_: state)
    monkeypatch.setattr(
        director_pipeline_module,
        "_find_pipeline_file",
        lambda *_: str(pipeline_file),
    )
    monkeypatch.setattr(director_pipeline_module, "_update_saved_pipeline", capture_update)

    result = director_pipeline_module._rejoin_clips_impl(
        str(tmp_path), "rejoin-test",
    )

    finished = tmp_path / result["filename"]
    assert finished.is_file()
    assert result["filename"] != existing_master.name
    assert existing_master.read_bytes() == original_master_bytes
    assert _media_duration(ffprobe, finished) == pytest.approx(2.25, abs=0.06)
    _decode_entire_video(ffmpeg, finished)
    before_cut_rgb = _mean_rgb_at(ffmpeg, finished, frame=9)
    after_cut_rgb = _mean_rgb_at(ffmpeg, finished, frame=15)
    assert before_cut_rgb[0] > before_cut_rgb[2] * 2
    assert after_cut_rgb[2] > after_cut_rgb[0] * 2
    assert _audio_rms(ffmpeg, finished, start=0.08, duration=0.3) < 0.003
    assert _audio_rms(ffmpeg, finished, start=0.68, duration=0.25) < 0.01
    assert _audio_rms(ffmpeg, finished, start=1.08, duration=0.35) > 0.025
    assert _audio_rms(ffmpeg, finished, start=1.82, duration=0.3) < 0.003
    assert len(list((tmp_path / "_director_title_cards" / "rejoin-test").glob("card_*.mp4"))) == 2
    assert len(saved_state["value"]["output_files"]) == 2

    manifest = json.loads(
        finished.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    assert [card["role"] for card in manifest["director_title_cards"]] == [
        "opening", "end",
    ]


def _prepare_expected_take_guard(tmp_path, monkeypatch):
    pid = "title-card-take-guard"
    clip_path = tmp_path / "clip_1.mp4"
    clip_bytes = b"tiny reviewed take bytes"
    clip_path.write_bytes(clip_bytes)
    state_path = tmp_path / f"_director_pipeline_{pid}.json"
    state = {
        "pipeline_type": "short_film_story",
        "video_model": "test_video_model",
        "_params_snapshot": {"fps": 24, "seamless": False},
        "clips": [{
            "video_filename": clip_path.name,
            "video_prompt": "A red kite crosses the garden.",
            "video_stale": False,
        }],
    }
    monkeypatch.setattr(
        director_pipeline_module, "load_pipeline_state", lambda *_args: state,
    )
    monkeypatch.setattr(
        director_pipeline_module, "_find_pipeline_file", lambda *_args: str(state_path),
    )
    dispatched = []

    def fake_impl(*args):
        dispatched.append(args)
        return {"job_id": "rerun-job"}

    monkeypatch.setattr(
        director_pipeline_module, "_rerun_clip_video_impl", fake_impl,
    )
    expected_take = {
        "signature": director_take_signature(state, 0),
        "path": str(clip_path.resolve()),
        "fingerprint": hashlib.sha256(clip_bytes).hexdigest(),
    }
    return pid, clip_path, state, expected_take, dispatched


def test_expected_take_guard_dispatches_matching_reviewed_take_once(
    tmp_path, monkeypatch,
):
    pid, _clip_path, _state, expected_take, dispatched = (
        _prepare_expected_take_guard(tmp_path, monkeypatch)
    )

    result = director_pipeline_module.rerun_clip_video(
        str(tmp_path), pid, 0, "A red kite crosses the garden at dusk.",
        expected_take=expected_take,
    )

    assert result == {"job_id": "rerun-job"}
    assert dispatched == [(
        str(tmp_path), pid, 0, "A red kite crosses the garden at dusk.",
    )]
    assert pid not in director_pipeline_module._pipeline_operations


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("params", "take or settings changed after review"),
        ("prompt", "take or settings changed after review"),
        ("stale", "take or settings changed after review"),
        ("bytes", "media changed after review"),
    ],
)
def test_expected_take_guard_rejects_changed_review_state_before_dispatch(
    tmp_path, monkeypatch, change, message,
):
    pid, clip_path, state, expected_take, dispatched = (
        _prepare_expected_take_guard(tmp_path, monkeypatch)
    )
    if change == "params":
        state["_params_snapshot"]["fps"] = 30
    elif change == "prompt":
        state["clips"][0]["video_prompt"] = "A blue kite crosses the garden."
    elif change == "stale":
        state["clips"][0]["video_stale"] = True
    else:
        clip_path.write_bytes(b"the media changed after review")

    with pytest.raises(ValueError, match=message):
        director_pipeline_module.rerun_clip_video(
            str(tmp_path), pid, 0, "A polished retake.",
            expected_take=expected_take,
        )

    assert dispatched == []
    assert pid not in director_pipeline_module._pipeline_operations
