"""CPU-only native music analysis, bound to the exact soundtrack bytes."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
import hashlib
import math
from pathlib import Path
import threading

from .creative_context import CreativeError


def _number(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise CreativeError(f"Music analysis returned invalid {label}; analyze the soundtrack again.", 422)
    return float(value)


def _rows(value, label, *, maximum=100_000):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > maximum or any(not isinstance(row, Mapping) for row in value):
        raise CreativeError(f"Music analysis returned invalid {label}; analyze the soundtrack again.", 422)
    return value


def _text(value, label, *, maximum=4000):
    if not isinstance(value, str) or len(value) > maximum:
        raise CreativeError(f"Music analysis returned invalid {label}.", 422)
    return value


def curate_music_analysis(value, expected_duration):
    """Keep timing evidence, excluding waveform arrays and private model paths."""
    if not isinstance(value, Mapping):
        raise CreativeError("Native music analysis returned no timing evidence.", 503)
    duration = _number(value.get("duration"), "duration", minimum=0.001)
    expected_duration = _number(expected_duration, "source duration", minimum=0.001)
    if abs(duration - expected_duration) > max(0.25, expected_duration * 0.001):
        raise CreativeError("Music analysis does not match the selected soundtrack duration; refresh the context.", 409)
    result = {"duration": duration, "bpm": _number(value.get("bpm"), "BPM"), "analysis_method": "native_cpu", "transcription_performed": bool(value.get("lyrics"))}

    def time_at(time, label):
        time = _number(time, label)
        if time > duration + 0.05:
            raise CreativeError(f"Music analysis {label} lies outside the soundtrack.", 422)
        return min(time, duration)

    result["beats"] = sorted([
        {"time": time_at(row.get("time"), "beat time"), "strength": min(1.0, _number(row.get("strength", 0), "beat strength"))}
        for row in _rows(value.get("beats"), "beats")
    ], key=lambda row: row["time"])
    downbeats = value.get("downbeats") or []
    if not isinstance(downbeats, list) or len(downbeats) > 100_000:
        raise CreativeError("Music analysis returned invalid downbeats.", 422)
    result["downbeats"] = sorted(time_at(time, "downbeat") for time in downbeats)
    result["sections"] = []
    for row in _rows(value.get("sections"), "sections", maximum=10_000):
        start, end = time_at(row.get("start"), "section start"), time_at(row.get("end"), "section end")
        if end <= start:
            raise CreativeError("Music analysis returned an empty or reversed section.", 422)
        result["sections"].append({"start": start, "end": end, "label": _text(row.get("label", "section"), "section label", maximum=160), "energy": min(1.0, _number(row.get("energy", 0.5), "section energy"))})
    result["sections"].sort(key=lambda row: row["start"])
    result["lyrics"] = []
    for row in _rows(value.get("lyrics"), "timed lyrics", maximum=10_000):
        start, end = time_at(row.get("start"), "lyric start"), time_at(row.get("end"), "lyric end")
        if end <= start:
            raise CreativeError("Music analysis returned invalid lyric timing.", 422)
        lyric = {"start": start, "end": end, "text": _text(row.get("text", ""), "lyric text")}
        if row.get("speaker") is not None:
            lyric["speaker"] = _text(row["speaker"], "speaker label", maximum=160)
        result["lyrics"].append(lyric)
    result["music_cues"] = []
    for row in _rows(value.get("music_cues"), "music cues", maximum=10_000):
        cue = {"time": time_at(row.get("time"), "cue time"), "type": _text(row.get("type", "accent"), "cue type", maximum=160)}
        for key in ("label", "confidence", "evidence", "instrument"):
            if isinstance(row.get(key), str):
                cue[key] = _text(row[key], f"cue {key}", maximum=500)
            elif isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool):
                cue[key] = _number(row[key], f"cue {key}")
        result["music_cues"].append(cue)
    activity = value.get("percussion_activity")
    result["percussion_activity"] = None if activity is None else []
    for row in _rows(activity, "percussion intervals", maximum=10_000):
        start, end = time_at(row.get("start"), "percussion start"), time_at(row.get("end"), "percussion end")
        if end <= start:
            raise CreativeError("Music analysis returned invalid percussion timing.", 422)
        result["percussion_activity"].append({"start": start, "end": end})
    result["beat_detection"] = "detected" if result["beats"] else "no_reliable_beats"
    result["vocal_timing"] = "available" if result["lyrics"] else "not_analyzed"
    return result


def native_cpu_analysis(path):
    # This branch uses librosa/percussion timing only. No Whisper, separator,
    # diarization model, local writer, or generation model is requested.
    from services.audio_analysis import analyze
    return analyze(str(path), transcribe=False, extract_vocals_for_transcription=False)


class MusicAnalysisCache:
    """Coalesce expensive CPU work and reject soundtrack changes after review."""

    def __init__(self, analyzer=None, maximum_entries=8):
        self.analyzer = analyzer or native_cpu_analysis
        self.maximum_entries = maximum_entries
        self._lock = threading.RLock()
        self._cache = OrderedDict()

    @staticmethod
    def fingerprint(path):
        path = Path(path)
        try:
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as error:
            raise CreativeError("The selected soundtrack is unavailable; select or upload it again.", 404) from error
        return {"sha256": digest.hexdigest(), "size_bytes": size}

    def read(self, path, expected_duration, previous_source=None):
        with self._lock:
            source = self.fingerprint(path)
            if previous_source is not None and source != previous_source:
                raise CreativeError("The soundtrack changed after planning. Request a fresh context and review its timing before rendering.", 409)
            key = (str(Path(path).resolve()), source["sha256"])
            if key not in self._cache:
                try:
                    raw = self.analyzer(path)
                except CreativeError:
                    raise
                except Exception as error:
                    raise CreativeError("Native CPU music analysis failed. Inspect audio-analysis status and retry the context; no render was submitted.", 503) from error
                analysis = curate_music_analysis(raw, expected_duration)
                if self.fingerprint(path) != source:
                    raise CreativeError("The soundtrack changed while it was being analyzed. Request a fresh context.", 409)
                self._cache[key] = analysis
                while len(self._cache) > self.maximum_entries:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            analysis = copy.deepcopy(self._cache[key])
            if abs(analysis["duration"] - expected_duration) > max(0.25, expected_duration * 0.001):
                raise CreativeError("Cached analysis does not match the current soundtrack duration.", 409)
            return analysis, source
