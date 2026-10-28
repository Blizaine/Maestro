"""Explicit, CPU-only Editor audio analysis. Never downloads model files."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping

from .editor_projects import EditorProjectError, resolve_editor_asset

_transcription_slot = threading.Semaphore(1)
MAX_ANALYSIS_SECONDS = 1800.0


def _cached_whisper_model(app_root: str) -> Path | None:
    root = Path(app_root) / "ckpts" / "whisper"
    candidates = [root, *sorted(root.glob("models--*faster-whisper-small/snapshots/*"))]
    for candidate in reversed(candidates):
        if all((candidate / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")):
            return candidate
    return None


def editor_audio_capabilities(app_root: str) -> dict[str, Any]:
    installed = importlib.util.find_spec("faster_whisper") is not None
    cached = _cached_whisper_model(app_root)
    return {
        "transcription_available": installed and cached is not None,
        "reason": "" if installed and cached else (
            "The local Whisper transcription model is not cached. Install it through Maestro audio analysis first."
            if installed else "The local transcription component is not installed."
        ),
        "device": "cpu",
        "downloads": False,
        "maximum_clip_seconds": MAX_ANALYSIS_SECONDS,
    }


def _source_range(
    project: Mapping[str, Any], item_id: str, *, save_root: str, uploads_root: str,
) -> tuple[str, Mapping[str, Any], float]:
    for track in project.get("tracks") or []:
        for item in track.get("items") or []:
            if item.get("id") != item_id:
                continue
            asset = (project.get("assets") or {}).get(item.get("asset_id"))
            if not asset or asset.get("type") not in {"video", "audio"} or not asset.get("has_audio", True):
                raise EditorProjectError("Select a video or audio clip with sound.")
            duration = float(item.get("duration") or 0)
            speed = float(item.get("speed") or 1)
            seconds = duration * speed
            if not all(math.isfinite(value) and value > 0 for value in (duration, speed, seconds)):
                raise EditorProjectError("This clip has invalid audio timing.")
            if seconds > MAX_ANALYSIS_SECONDS:
                raise EditorProjectError("Select or trim a source range of 30 minutes or less for audio analysis.")
            source = resolve_editor_asset(asset, save_root=save_root,
                                          workspace=str(project.get("workspace") or "default"),
                                          uploads_root=uploads_root)
            return source, item, seconds
    raise EditorProjectError("The selected clip is no longer in this project.")


def analyze_editor_audio(
    project: Mapping[str, Any], item_id: str, *, save_root: str, uploads_root: str,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    source, item, seconds = _source_range(project, item_id, save_root=save_root, uploads_root=uploads_root)
    command = [ffmpeg, "-hide_banner", "-nostdin", "-ss", str(float(item.get("source_in") or 0)),
               "-t", str(seconds), "-i", source, "-vn", "-af",
               "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=180, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as error:
        raise EditorProjectError("Audio loudness analysis could not finish.") from error
    stderr = result.stderr.decode("utf-8", errors="replace")
    start, end = stderr.rfind("{"), stderr.rfind("}")
    try:
        report = json.loads(stderr[start:end + 1])
        loudness, peak = float(report["input_i"]), float(report["input_tp"])
        if result.returncode or not math.isfinite(loudness) or not math.isfinite(peak):
            raise ValueError("No measurable signal")
    except (ValueError, TypeError, KeyError) as error:
        raise EditorProjectError("This source has no measurable audio signal, or analysis failed.") from error
    gain_db = min(-16.0 - loudness, -1.5 - peak)
    gain = min(4.0, max(0.000001, 10 ** (gain_db / 20)))
    return {"integrated_lufs": round(loudness, 2), "peak_db": round(peak, 2),
            "gain_db": round(20 * math.log10(gain), 2), "gain": gain,
            "scope": "selected source range", "target_lufs": -16.0, "true_peak_limit_db": -1.5}


def transcribe_editor_audio(
    project: Mapping[str, Any], item_id: str, *, save_root: str, uploads_root: str,
    app_root: str, ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    source, item, seconds = _source_range(project, item_id, save_root=save_root, uploads_root=uploads_root)
    model_path = _cached_whisper_model(app_root)
    if model_path is None or importlib.util.find_spec("faster_whisper") is None:
        raise EditorProjectError(editor_audio_capabilities(app_root)["reason"])
    if not _transcription_slot.acquire(blocking=False):
        raise EditorProjectError("An Editor transcription is already running. Try again after it finishes.")
    try:
        from faster_whisper import WhisperModel
        with tempfile.TemporaryDirectory(prefix="maestro-editor-audio-") as folder:
            audio = str(Path(folder) / "selected.wav")
            try:
                prepared = subprocess.run(
                    [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss",
                     str(float(item.get("source_in") or 0)), "-t", str(seconds), "-i", source,
                     "-vn", "-ac", "1", "-ar", "16000", audio], capture_output=True,
                    timeout=180, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise EditorProjectError("The selected audio could not be prepared for transcription.") from error
            if prepared.returncode:
                raise EditorProjectError("The selected source has no decodable audio.")
            model = WhisperModel(str(model_path), device="cpu", compute_type="int8",
                                 cpu_threads=4, num_workers=1, local_files_only=True)
            segments, info = model.transcribe(audio, beam_size=3, word_timestamps=True, vad_filter=True)
            speed = float(item.get("speed") or 1)
            duration = float(item["duration"])
            deadline = time.monotonic() + 900
            cues = []
            for segment in segments:
                if time.monotonic() > deadline:
                    raise EditorProjectError("Transcription reached its 15-minute processing bound. Use a shorter source range.")
                text = str(segment.text or "").strip()
                start, end = max(0.0, float(segment.start) / speed), min(duration, float(segment.end) / speed)
                if not text or end <= start:
                    continue
                words = [{"start": max(0.0, float(word.start) / speed),
                          "end": min(duration, float(word.end) / speed), "text": str(word.word).strip()}
                         for word in (segment.words or []) if word.start is not None and word.end is not None]
                cues.append({"start": round(start, 3), "end": round(end, 3), "text": text, "words": words})
            if not cues:
                raise EditorProjectError("No spoken words were found in the selected audio range.")
            return {"cues": cues, "offset": float(item.get("start") or 0),
                    "language": info.language, "device": "cpu", "downloads": False}
    except EditorProjectError:
        raise
    except (RuntimeError, OSError, ValueError) as error:
        raise EditorProjectError("Local transcription failed; the project was not changed.") from error
    finally:
        _transcription_slot.release()
