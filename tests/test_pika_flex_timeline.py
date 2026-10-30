"""Exercise the real native-prefix normalization and CPU audio/video mux path."""
# ruff: noqa: E402 -- app imports follow the repository path bootstrap.
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.minimax_h3.pika_flex import normalize_pika_flex_start_prefix
from models.minimax_h3.packing import normalize_h3_native_start_prefix


class StartImageCarrier:
    shape = (3, 1, 32, 32)

    def numel(self):
        return 3 * 32 * 32


class MultiFrameCarrier:
    shape = (3, 2, 32, 32)

    def numel(self):
        return 2 * 3 * 32 * 32


def test_only_the_first_image_carrier_loses_generic_prefix_bookkeeping():
    carrier = StartImageCarrier()
    assert normalize_pika_flex_start_prefix(carrier, 1, 1, image_start="first.png", window_no=1) == (None, 0, 0)
    assert normalize_pika_flex_start_prefix(carrier, 1, "1", image_start="first.png", window_no=1) == (None, 0, 0)
    for source_count, overlap, image, window in ((2, 1, "first.png", 1), (1, 1, "first.png", 2), (1, 1, None, 1), (1, 2, "first.png", 1)):
        assert normalize_pika_flex_start_prefix(carrier, source_count, overlap, image_start=image, window_no=window) == (carrier, source_count, overlap)


def test_native_h3_start_prefix_normalization_is_narrow():
    carrier = StartImageCarrier()
    assert normalize_h3_native_start_prefix(
        carrier,
        1,
        1,
        image_start="first.png",
        window_no=1,
        architecture="minimax_h3_fused_turbo",
    ) == (None, 0, 0)

    unchanged = (
        ("wan", None, 1, 1, carrier, "first.png", 1),
        ("minimax_h3", "source.mp4", 1, 1, carrier, "first.png", 1),
        ("minimax_h3", None, 2, 1, carrier, "first.png", 1),
        ("minimax_h3", None, 1, 2, carrier, "first.png", 1),
        ("minimax_h3", None, 1, 1, MultiFrameCarrier(), "first.png", 1),
        ("minimax_h3", None, 1, 1, carrier, "first.png", 2),
        ("minimax_h3", None, 1, 1, carrier, None, 1),
    )
    for architecture, video_source, source_count, overlap, prefix, image, window in unchanged:
        assert normalize_h3_native_start_prefix(
            prefix,
            source_count,
            overlap,
            image_start=image,
            window_no=window,
            architecture=architecture,
            video_source=video_source,
        ) == (prefix, source_count, overlap)


def test_native_first_last_timeline_keeps_all_124_frames_after_audio_mux(tmp_path):
    ffmpeg_exe, ffprobe_exe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg_exe or not ffprobe_exe:
        pytest.skip("CPU ffmpeg/ffprobe are not installed")
    ffmpeg = pytest.importorskip("ffmpeg")
    video = tmp_path / "native.mp4"
    subprocess.run([ffmpeg_exe, "-v", "error", "-f", "lavfi", "-i", "color=black:s=64x64:r=24", "-frames:v", "124", "-c:v", "libx264", "-y", str(video)], check=True)
    # Load the actual mux functions without importing a model or CUDA runtime.
    tree = ast.parse((ROOT / "app/shared/utils/audio_video.py").read_text())
    names = {"get_mp4_audio_codec_settings", "get_audio_file_channels", "combine_and_concatenate_video_with_audio_tracks"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"os": os, "subprocess": subprocess, "ffmpeg": ffmpeg}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual_audio_video.py", "exec"), namespace)
    prefix, source_count, overlap = normalize_pika_flex_start_prefix(StartImageCarrier(), 1, 1, image_start="first.png", window_no=1)
    assert prefix is None and overlap == 0
    audio = tmp_path / "native.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(32000)
        stream.writeframes(b"\0" * (round(124 / 24 * 32000) * 4))
    output = tmp_path / "complete.mp4"
    namespace["combine_and_concatenate_video_with_audio_tracks"](str(output), str(video), [], [str(audio)], source_count / 24, 32000, new_audio_from_start=True)
    probe = json.loads(subprocess.run([ffprobe_exe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(output)], capture_output=True, text=True, check=True).stdout)
    video_stream = next(row for row in probe["streams"] if row["codec_type"] == "video")
    assert int(video_stream["nb_read_frames"]) == 124
    audio_stream = next(row for row in probe["streams"] if row["codec_type"] == "audio")
    assert abs(float(audio_stream["duration"]) - 124 / 24) < 1 / 24


def test_native_h3_start_keeps_full_124_frame_video_and_audio_after_mux(tmp_path):
    ffmpeg_exe, ffprobe_exe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg_exe or not ffprobe_exe:
        pytest.skip("CPU ffmpeg/ffprobe are not installed")
    ffmpeg = pytest.importorskip("ffmpeg")
    video = tmp_path / "h3-native.mp4"
    subprocess.run([ffmpeg_exe, "-v", "error", "-f", "lavfi", "-i", "color=black:s=64x64:r=24", "-frames:v", "124", "-c:v", "libx264", "-y", str(video)], check=True)

    tree = ast.parse((ROOT / "app/shared/utils/audio_video.py").read_text())
    names = {"get_mp4_audio_codec_settings", "get_audio_file_channels", "combine_and_concatenate_video_with_audio_tracks"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"os": os, "subprocess": subprocess, "ffmpeg": ffmpeg}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual_audio_video.py", "exec"), namespace)

    frame_samples = round(48000 / 24)
    audio_samples = round(124 / 24 * 48000)
    wgp_tree = ast.parse((ROOT / "app/wgp.py").read_text())
    truncate_node = next(
        node
        for node in wgp_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "truncate_audio"
    )
    truncate_namespace = {}
    exec(
        compile(
            ast.Module(body=[truncate_node], type_ignores=[]),
            "actual_wgp_truncate_audio.py",
            "exec",
        ),
        truncate_namespace,
    )
    old_audio = truncate_namespace["truncate_audio"](
        range(audio_samples), 1, 0, 24, 48000
    )
    assert len(old_audio) == 123 * frame_samples
    prefix, source_count, overlap = normalize_h3_native_start_prefix(
        StartImageCarrier(),
        1,
        1,
        image_start="first.png",
        window_no=1,
        architecture="minimax_h3_fused_turbo",
    )
    assert prefix is None and source_count == overlap == 0
    native_audio = (
        range(audio_samples)
        if prefix is None
        else truncate_namespace["truncate_audio"](
            range(audio_samples), overlap, 0, 24, 48000
        )
    )
    assert len(native_audio) == 124 * frame_samples

    audio = tmp_path / "h3-native.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(b"\0" * (audio_samples * 4))
    output = tmp_path / "h3-complete.mp4"
    namespace["combine_and_concatenate_video_with_audio_tracks"](
        str(output),
        str(video),
        [],
        [str(audio)],
        source_count / 24,
        48000,
        new_audio_from_start=True,
    )
    probe = json.loads(subprocess.run([ffprobe_exe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(output)], capture_output=True, text=True, check=True).stdout)
    video_stream = next(row for row in probe["streams"] if row["codec_type"] == "video")
    audio_stream = next(row for row in probe["streams"] if row["codec_type"] == "audio")
    assert int(video_stream["nb_read_frames"]) == 124
    assert abs(float(audio_stream["duration"]) - 124 / 24) < 1 / 24
