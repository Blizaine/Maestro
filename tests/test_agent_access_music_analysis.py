"""CPU-only music evidence, caching and immutable-source regressions."""
from concurrent.futures import ThreadPoolExecutor
import copy
import sys
import time
from types import SimpleNamespace

import pytest

from services.agent_access.creative_context import CreativeError
from services.agent_access.music_analysis import MusicAnalysisCache, curate_music_analysis, native_cpu_analysis


def analysis():
    return {
        "duration": 12.125, "bpm": 120,
        "beats": [{"time": 1.5, "strength": 0.8}, {"time": 0.5, "strength": 1}],
        "downbeats": [0.5],
        "sections": [{"start": 0, "end": 6, "label": "intro", "energy": 0.3}, {"start": 6, "end": 12.125, "label": "chorus", "energy": 0.9}],
        "lyrics": None, "percussion_activity": [{"start": 3, "end": 10}],
        "music_cues": [{"time": 3, "type": "percussion_entry", "confidence": "high", "evidence": "detected"}],
        "onset_envelope": [0.1] * 300, "vocals_path": "private/path/to/vocals.wav",
    }


def test_music_readback_keeps_evidence_and_distinguishes_unknown_vocals():
    result = curate_music_analysis(analysis(), 12.125)
    assert result["beats"][0]["time"] == 0.5
    assert result["sections"][1]["label"] == "chorus"
    assert result["music_cues"][0]["type"] == "percussion_entry"
    assert result["vocal_timing"] == "not_analyzed"
    assert result["transcription_performed"] is False
    assert "vocals_path" not in result and "onset_envelope" not in result
    raw = analysis()
    raw["lyrics"] = [{"start": 4, "end": 5, "text": "Hello", "speaker": "singer-1"}]
    assert curate_music_analysis(raw, 12.125)["vocal_timing"] == "available"


def test_no_detected_beats_is_reported_honestly():
    raw = analysis()
    raw["beats"] = []
    assert curate_music_analysis(raw, 12.125)["beat_detection"] == "no_reliable_beats"


@pytest.mark.parametrize("field,value", [("bpm", float("nan")), ("duration", -1), ("beats", [{"time": 30, "strength": 1}]), ("sections", [{"start": 9, "end": 8}]), ("downbeats", [float("inf")])])
def test_invalid_native_evidence_is_not_silently_used(field, value):
    raw = analysis()
    raw[field] = value
    with pytest.raises(CreativeError):
        curate_music_analysis(raw, 12.125)


def test_mismatched_soundtrack_duration_rejected():
    with pytest.raises(CreativeError, match="does not match"):
        curate_music_analysis(analysis(), 20)


def test_cpu_adapter_never_requests_transcription_or_model_downloads(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "services.audio_analysis", SimpleNamespace(analyze=lambda *args, **kwargs: calls.append((args, kwargs)) or analysis()))
    native_cpu_analysis("song.wav")
    assert calls == [(("song.wav",), {"transcribe": False, "extract_vocals_for_transcription": False})]


def test_cache_coalesces_requests_and_returns_independent_evidence(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"audio")
    calls = []
    def analyze(path):
        calls.append(path)
        time.sleep(0.02)
        return analysis()
    cache = MusicAnalysisCache(analyze)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cache.read(song, 12.125), range(4)))
    assert len(calls) == 1
    first, source = results[0]
    first["sections"][0]["label"] = "modified"
    assert cache.read(song, 12.125, source)[0]["sections"][0]["label"] == "intro"
    assert len(calls) == 1


def test_reviewed_soundtrack_change_requires_new_context(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"original")
    cache = MusicAnalysisCache(lambda _: analysis())
    _, source = cache.read(song, 12.125)
    song.write_bytes(b"different")
    with pytest.raises(CreativeError, match="changed after planning"):
        cache.read(song, 12.125, source)


def test_change_during_analysis_is_not_cached(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"original")
    def analyze(path):
        path.write_bytes(b"replacement")
        return copy.deepcopy(analysis())
    cache = MusicAnalysisCache(analyze)
    with pytest.raises(CreativeError, match="changed while"):
        cache.read(song, 12.125)
    assert not cache._cache


def test_failed_analysis_does_not_poison_cache(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"audio")
    calls = []
    def analyze(_):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("decoder failed")
        return analysis()
    cache = MusicAnalysisCache(analyze)
    with pytest.raises(CreativeError, match="analysis failed"):
        cache.read(song, 12.125)
    assert cache.read(song, 12.125)[0]["bpm"] == 120
