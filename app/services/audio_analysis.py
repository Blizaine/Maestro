"""Audio analysis service for the Maestro Music Video Director.

Analyzes audio files to extract:
- BPM and beat positions (librosa)
- Onset detection / energy envelope
- Song section segmentation (energy-based heuristic)
- Optional: lyrics transcription (faster-whisper)
"""

import os
import gc
import math
import re
import logging
import threading
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live progress reporting
# ---------------------------------------------------------------------------
# The analyze() call is synchronous from the API client's POV but
# internally goes through several phases (load → beat detect → vocals
# → transcribe → diarize) that each take seconds-to-minutes. The first
# run also downloads ~500MB of model weights (vocal-extraction,
# Whisper, pyannote diarization) silently inside those phases.
#
# Without progress signaling, the UI just shows "Analyzing audio..."
# for the entire 1-5 minute wait — looks broken on first runs that
# include a download. This module exposes a thread-safe progress dict
# updated at each phase boundary, polled by the frontend via
# /api/v1/audio/analyze/status.
#
# Reset at the start of analyze(), updated at each phase, cleared
# when analyze() returns. If two analyze() calls overlap (shouldn't
# happen via the UI but could via direct API), they overwrite each
# other's status — acceptable since the polling endpoint is meant
# for the active analyze call.
_PROGRESS_LOCK = threading.Lock()
_PROGRESS = {"step": "", "detail": ""}


def _set_progress(step: str, detail: str = "") -> None:
    """Update the shared progress state read by the status polling endpoint."""
    with _PROGRESS_LOCK:
        _PROGRESS["step"] = step
        _PROGRESS["detail"] = detail
    if step:
        print(f"[AudioAnalysis][progress] {step}{(': ' + detail) if detail else ''}")


def get_progress() -> dict:
    """Read the current analyze progress (thread-safe)."""
    with _PROGRESS_LOCK:
        return {"step": _PROGRESS["step"], "detail": _PROGRESS["detail"]}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Beat:
    time: float
    strength: float

@dataclass
class Section:
    start: float
    end: float
    label: str
    energy: float

@dataclass
class LyricSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

@dataclass
class AudioAnalysis:
    duration: float
    sample_rate: int
    bpm: float
    beats: List[Beat]
    downbeats: List[float]
    sections: List[Section]
    onset_envelope: List[float]
    lyrics: Optional[List[LyricSegment]] = None
    vocals_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    import librosa
    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    return y, sr_actual


def _detect_beats(y: np.ndarray, sr: int) -> Tuple[float, List[Beat]]:
    import librosa
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    max_onset = onset_env.max() if onset_env.max() > 0 else 1.0

    strengths = []
    for bf in beat_frames:
        if bf < len(onset_env):
            strengths.append(float(onset_env[bf] / max_onset))
        else:
            strengths.append(0.5)

    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beats = [Beat(time=float(t), strength=s) for t, s in zip(beat_times, strengths)]
    return bpm, beats


def _detect_downbeats(beats: List[Beat], bpm: float) -> List[float]:
    if not beats:
        return []
    first_n = beats[:min(8, len(beats))]
    anchor_idx = max(range(len(first_n)), key=lambda i: first_n[i].strength)
    downbeats = []
    for i in range(anchor_idx, len(beats), 4):
        downbeats.append(beats[i].time)
    return downbeats


def _compute_onset_envelope(y: np.ndarray, sr: int, target_fps: float = 10.0) -> List[float]:
    import librosa
    hop_length = int(sr / target_fps)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    max_val = onset_env.max() if onset_env.max() > 0 else 1.0
    return (onset_env / max_val).tolist()


def _segment_sections(
    y: np.ndarray, sr: int, beats: List[Beat], duration: float
) -> List[Section]:
    import librosa

    hop = sr // 2
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    rms_norm = rms / (rms.max() + 1e-8)
    sc_norm = spectral_centroid / (spectral_centroid.max() + 1e-8)
    combined = 0.7 * rms_norm + 0.3 * sc_norm

    min_section_frames = max(4, int(8.0 * 2))  # ~8 seconds minimum

    boundaries = [0]
    for i in range(min_section_frames, len(combined) - min_section_frames):
        left_mean = combined[max(0, i - min_section_frames):i].mean()
        right_mean = combined[i:min(len(combined), i + min_section_frames)].mean()
        diff = abs(right_mean - left_mean)
        if diff > 0.15 and (i - boundaries[-1]) >= min_section_frames:
            boundaries.append(i)
    boundaries.append(len(combined))

    frame_times = librosa.frames_to_time(np.arange(len(combined)), sr=sr, hop_length=hop)
    overall_rms_mean = rms_norm.mean()

    sections = []
    for idx in range(len(boundaries) - 1):
        sf = boundaries[idx]
        ef = boundaries[idx + 1]
        start_time = float(frame_times[sf]) if sf < len(frame_times) else 0.0
        end_time = float(frame_times[min(ef, len(frame_times) - 1)])

        section_energy = float(rms_norm[sf:ef].mean())
        section_brightness = float(sc_norm[sf:ef].mean())

        position_ratio = start_time / duration if duration > 0 else 0
        label = _classify_section(
            section_energy, section_brightness, overall_rms_mean,
            position_ratio, idx, len(boundaries) - 1
        )

        sections.append(Section(
            start=round(start_time, 3),
            end=round(end_time, 3),
            label=label,
            energy=round(section_energy, 3),
        ))

    return sections


def _classify_section(
    energy: float, brightness: float, mean_energy: float,
    position: float, section_idx: int, total_sections: int
) -> str:
    is_first = section_idx == 0
    is_last = section_idx == total_sections - 1
    is_high_energy = energy > mean_energy * 1.2
    is_low_energy = energy < mean_energy * 0.6

    if is_first and is_low_energy:
        return "intro"
    if is_last and is_low_energy:
        return "outro"
    if is_high_energy and brightness > 0.5:
        return "chorus"
    if is_low_energy:
        return "bridge"
    return "verse"


# ---------------------------------------------------------------------------
# Whisper transcription (optional, lazy-loaded)
# ---------------------------------------------------------------------------

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper is required for transcription. "
            "Install with: uv pip install faster-whisper"
        )

    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "ckpts", "whisper"
    )
    os.makedirs(cache_dir, exist_ok=True)

    print("[AudioAnalysis] Loading faster-whisper small model (CUDA)...")
    _whisper_model = WhisperModel(
        "small",
        device="cuda",
        compute_type="float16",
        download_root=cache_dir,
    )
    print("[AudioAnalysis] Whisper model loaded")
    return _whisper_model


def _clean_lyrics_hint(lyrics: Optional[str]) -> Optional[str]:
    """Prepare known song lyrics for use as a Whisper initial_prompt.

    Used when transcribing a GENERATED track (Director Music Video
    generate flow) — we already know exactly what ACE-Step sang, so
    seeding the decoder with that vocabulary makes the transcription
    snap to the real words instead of mishearing sung vocals (reverb,
    layering, melisma). The prompt is a soft vocabulary/spelling prior;
    timing still comes entirely from the audio.

    Cleaning:
      - Strip ACE-Step structure tags ([Verse], [Chorus], [Bridge],
        [Instrumental], ...) — they're control tokens, not sung text.
      - Collapse whitespace/newlines to plain prose.
      - Keep only the FIRST ~150 words. Whisper's prompt window is 224
        tokens, and on overflow faster-whisper keeps the TAIL — but the
        initial_prompt conditions the first ~30s window, so the HEAD of
        the lyrics is the useful end. Lyric vocabulary also saturates
        fast (choruses repeat), so 150 words covers nearly everything.

    Returns None when nothing usable remains (e.g. pure-instrumental
    placeholder), so callers can pass the result straight through.
    """
    if not lyrics:
        return None
    text = re.sub(r"\[[^\]]*\]", " ", lyrics)
    words = text.split()
    if len(words) < 3:
        return None
    return " ".join(words[:150])


def _transcribe(audio_path: str, lyrics_hint: Optional[str] = None) -> List[LyricSegment]:
    model = _get_whisper_model()
    initial_prompt = _clean_lyrics_hint(lyrics_hint)
    if initial_prompt:
        print(f"[AudioAnalysis] Seeding transcription with known lyrics ({len(initial_prompt)} chars)")
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=False,
        language=None,
        vad_filter=True,
        initial_prompt=initial_prompt,
    )

    lyrics = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            lyrics.append(LyricSegment(
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=text,
            ))
    return lyrics


def unload_whisper():
    global _whisper_model
    _whisper_model = None
    gc.collect()


# ---------------------------------------------------------------------------
# Speaker diarization (optional, runs on CUDA via pyannote)
# ---------------------------------------------------------------------------

_diarizer_pipe = None
_diarizer_profile: Optional[str] = None  # profile the cached pipeline is instantiated with

# Clustering hyperparameters per content type. The embedding model was
# trained on SPEECH — singing voice drifts far more (pitch, vibrato,
# effects, backing vocals), so the speech-tuned profile shatters one
# singer into many "speakers" (observed: 6 on a solo track). The music
# profile requires sustained singing per cluster (pyannote's default 12
# instead of the AI-dialogue-tuned 6) and merges more aggressively
# (threshold 0.82 vs 0.7046 — validated on real ACE-Step outputs: solo
# tracks collapse to 1 speaker while a genuinely distinct second voice
# still separates well above this distance).
_DIARIZER_PROFILES = {
    "speech": {
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 6,
            "threshold": 0.7045654963945799,
        },
        "segmentation": {"min_duration_off": 0.0},
    },
    "music": {
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 12,
            "threshold": 0.82,
        },
        "segmentation": {"min_duration_off": 0.0},
    },
}


def get_diarizer_pipeline(profile: str = "speech"):
    """Load (or return cached) pyannote 3.1 diarization pipeline.

    Public so postprocessing modules (voice_clone.py) can reuse the
    same loader instead of re-implementing the model-loading logic.

    Returns the loaded Pipeline instance moved to CUDA if available,
    or None if pyannote isn't installed AND no checkpoints are
    available locally AND no HF_TOKEN is set for download.

    Three load paths, in order:

      1. MANUAL ASSEMBLY (preferred when present) — reads two .bin
         files from app/ckpts/pyannote/ and builds a SpeakerDiarization
         pipeline manually via pyannote.audio.pipelines. Bypasses
         HuggingFace's gated download entirely. This is the same
         approach app/preprocessing/speakers_separator.py uses, and
         it works against the .bin files that ship with Maestro's
         install (the user's actual on-disk state):
             ckpts/pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin
             ckpts/pyannote/pytorch_model_segmentation-3.0.bin

      2. HF_HOME CACHE — if path #1's files are missing but
         <project_root>/cache/HF_HOME/hub/speaker-diarization-3.1/
         exists (Music Video mode's expected layout, populated by
         the audio_analysis module on first use), use the HF_HOME
         env var trick + canonical name.

      3. HF DOWNLOAD — if HF_TOKEN is set and the user has accepted
         the gated-model terms, download from HuggingFace.

    Falls through cleanly with a None return and an informative
    print if none of the paths work.

    Module-level cache via _diarizer_pipe means second+ calls in the
    same process are free, shared between audio_analysis._diarize
    and voice_clone. `profile` selects the clustering hyperparameters
    (see _DIARIZER_PROFILES) — a cached pipeline is re-instantiated in
    place when a different profile is requested (cheap: instantiate()
    only sets hyperparameters, no model reload).
    """
    global _diarizer_pipe, _diarizer_profile
    if _diarizer_pipe is not None:
        if profile != _diarizer_profile:
            try:
                _diarizer_pipe.instantiate(_DIARIZER_PROFILES[profile])
                _diarizer_profile = profile
            except Exception as e:
                print(f"[Diarization] Profile switch to '{profile}' failed (keeping '{_diarizer_profile}'): {e}")
        return _diarizer_pipe

    try:
        import torch
    except ImportError as e:
        print(f"[Diarization] Skipped (missing dependency): {e}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _base = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.normpath(os.path.join(_base, "..", ".."))
    _app_root = os.path.normpath(os.path.join(_base, ".."))

    # PyTorch 2.6+ weights_only=True breaks pyannote's pickle
    # checkpoints. All three load paths use torch.load internally —
    # patch globally for the duration of the load.
    _orig_torch_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_load

    try:
        # ── Path 0: fetch the ungated .bin files when missing ───────
        # These are the same two checkpoints wgp's shared-model download
        # provides (DeepBeepMeep/Wan2.1, pyannote/ subfolder — an ungated
        # mirror of the pyannote 3.1 models). That shared download only
        # runs when a generation model loads, so a fresh install that
        # reaches audio analysis first (e.g. Director on an uploaded
        # song) has no local files, no HF cache, and — without an
        # HF_TOKEN for the gated upstream repo — diarization silently
        # skipped. Fetch the two files directly so first use just works.
        embedding_path = os.path.join(_app_root, "ckpts", "pyannote",
                                       "pyannote_model_wespeaker-voxceleb-resnet34-LM.bin")
        segmentation_path = os.path.join(_app_root, "ckpts", "pyannote",
                                          "pytorch_model_segmentation-3.0.bin")
        if not (os.path.isfile(embedding_path) and os.path.isfile(segmentation_path)):
            try:
                import shutil
                import tempfile
                from huggingface_hub import hf_hub_download
                target_dir = os.path.join(_app_root, "ckpts", "pyannote")
                os.makedirs(target_dir, exist_ok=True)
                for fname in ("pyannote_model_wespeaker-voxceleb-resnet34-LM.bin",
                              "pytorch_model_segmentation-3.0.bin"):
                    dest = os.path.join(target_dir, fname)
                    if os.path.isfile(dest):
                        continue
                    print(f"[Diarization] Downloading {fname} (ungated mirror, first use)...")
                    tmp_dir = tempfile.mkdtemp(prefix="pyannote_dl_")
                    try:
                        got = hf_hub_download(repo_id="DeepBeepMeep/Wan2.1", filename=fname,
                                              subfolder="pyannote", local_dir=tmp_dir)
                        shutil.move(got, dest)
                    finally:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                print("[Diarization] Checkpoints downloaded")
            except Exception as e:
                print(f"[Diarization] Auto-download failed (trying other load paths): {e}")

        # ── Path 1: manual assembly from local .bin files ──────────
        # wgp's shared-model download (or Path 0 above) provides these
        # .bin files; Music Video mode references them too via
        # speakers_separator.py. Resolve via absolute path from this
        # module's location, since CWD isn't guaranteed to be the app/
        # folder at every entry point.
        if os.path.isfile(embedding_path) and os.path.isfile(segmentation_path):
            try:
                from pyannote.audio import Model
                from pyannote.audio.pipelines import SpeakerDiarization
                print(f"[Diarization] Loading from local .bin files...")
                segmentation_model = Model.from_pretrained(segmentation_path)
                embedding_model = Model.from_pretrained(embedding_path)
                pipeline = SpeakerDiarization(
                    segmentation=segmentation_model,
                    embedding=embedding_model,
                    clustering="AgglomerativeClustering",
                )
                # Hyperparameters come from the requested profile — see
                # _DIARIZER_PROFILES for the speech vs music rationale.
                pipeline.instantiate(_DIARIZER_PROFILES[profile])
                if device == "cuda":
                    pipeline.to(torch.device(device))
                _diarizer_pipe = pipeline
                _diarizer_profile = profile
                print(f"[Diarization] Pipeline loaded from local .bin files on {device} (profile: {profile})")
                return _diarizer_pipe
            except Exception as e:
                print(f"[Diarization] Manual assembly failed, falling back: {e}")

        # ── Path 2: HF_HOME-style cache directory ──────────────────
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
        except ImportError as e:
            print(f"[Diarization] Skipped (pyannote not installed): {e}")
            return None
        local_model_dir = os.path.join(
            _project_root, "cache", "HF_HOME", "hub", "speaker-diarization-3.1"
        )
        local_config = os.path.join(local_model_dir, "config.yaml")
        hf_token = os.environ.get("HF_TOKEN", "")

        # Apply the requested profile to a from_pretrained pipeline (it
        # arrives instantiated with the HF config defaults). Guarded so an
        # instantiate hiccup degrades to defaults instead of losing the
        # loaded pipeline; _diarizer_profile stays None so the next call
        # retries the switch.
        def _apply_profile(pipe):
            global _diarizer_profile
            try:
                pipe.instantiate(_DIARIZER_PROFILES[profile])
                _diarizer_profile = profile
            except Exception as e:
                _diarizer_profile = None
                print(f"[Diarization] Profile instantiate failed (using model defaults): {e}")
            return pipe

        if os.path.isfile(local_config):
            hf_home = os.path.join(_project_root, "cache", "HF_HOME")
            os.environ.setdefault("HF_HOME", os.path.normpath(hf_home))
            try:
                print(f"[Diarization] Loading from HF_HOME cache on {device}...")
                _diarizer_pipe = _apply_profile(PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                ).to(torch.device(device)))
                print("[Diarization] Pipeline loaded from HF_HOME cache")
                return _diarizer_pipe
            except Exception as e:
                print(f"[Diarization] HF_HOME load failed, falling back: {e}")

        # ── Path 3: HuggingFace download (requires gated-model accept + HF_TOKEN) ──
        if hf_token:
            try:
                print(f"[Diarization] Downloading from HuggingFace on {device}...")
                _diarizer_pipe = _apply_profile(PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                ).to(torch.device(device)))
                print("[Diarization] Pipeline downloaded + loaded")
                return _diarizer_pipe
            except Exception as e:
                print(f"[Diarization] HF download failed: {e}")

        print(
            "[Diarization] Skipped — no local .bin files, no HF_HOME cache, "
            "and no HF_TOKEN.\n"
            f"  Looked for:\n"
            f"    .bin path 1: {embedding_path}\n"
            f"    .bin path 2: {segmentation_path}\n"
            f"    cache: {local_model_dir}"
        )
        return None
    finally:
        torch.load = _orig_torch_load


def _diarize(audio_path: str, lyrics: List[LyricSegment]) -> List[LyricSegment]:
    """Run pyannote speaker diarization and tag each lyric segment with a speaker.

    Uses temporal overlap to assign the dominant speaker to each segment.
    Runs on CUDA if available, offloads immediately after to free VRAM.

    Model loading goes through get_diarizer_pipeline() — this function
    used to carry its own legacy loader that only knew the HF-clone and
    gated-token paths, so fresh installs (no local clone, no HF_TOKEN)
    silently skipped diarization even though the shared loader can
    assemble the pipeline from ungated .bin files (auto-downloaded on
    first use).
    """
    try:
        import torch
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"[Diarization] Skipped (missing dependency): {e}")
        return lyrics

    # Music profile: this function's only production caller is the song
    # analysis flow. voice_clone loads the speech profile via the same
    # shared loader (a cached pipeline switches profiles in place).
    pipe = get_diarizer_pipeline(profile="music")
    if pipe is None:
        return lyrics  # loader already printed the reason

    try:
        # Load audio at 16kHz mono via ffmpeg
        import subprocess
        cmd = [
            "ffmpeg", "-nostdin", "-threads", "0", "-i", audio_path,
            "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-"
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        audio_np = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        audio_data = {
            "waveform": torch.from_numpy(audio_np[None, :]),
            "sample_rate": 16000,
        }

        print("[Diarization] Running speaker diarization...")
        # Hard cap as a backstop on top of the music profile's clustering:
        # songs have 1-3 vocalists; anything beyond that is the embedding
        # model mistaking a register/effect change for a new person.
        segments = pipe(audio_data, min_speakers=1, max_speakers=3)

        # Build DataFrame of speaker segments
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

        speakers_found = diarize_df["speaker"].nunique()
        print(f"[Diarization] Found {speakers_found} speaker(s)")

        # Assign speaker to each lyric segment by temporal overlap
        for lyr in lyrics:
            diarize_df["intersection"] = (
                np.minimum(diarize_df["end"], lyr.end) -
                np.maximum(diarize_df["start"], lyr.start)
            )
            intersected = diarize_df[diarize_df["intersection"] > 0]
            if len(intersected) > 0:
                best_speaker = (
                    intersected.groupby("speaker")["intersection"]
                    .sum().sort_values(ascending=False).index[0]
                )
                lyr.speaker = best_speaker

        # Log speaker distribution
        speaker_counts: dict = {}
        for lyr in lyrics:
            s = lyr.speaker or "unknown"
            speaker_counts[s] = speaker_counts.get(s, 0) + 1
        print(f"[Diarization] Line counts: {speaker_counts}")

    except Exception as e:
        print(f"[Diarization] Failed, continuing without speaker tags: {e}")

    return lyrics


def unload_diarizer():
    """Free the diarization pipeline and reclaim VRAM."""
    global _diarizer_pipe, _diarizer_profile
    if _diarizer_pipe is not None:
        del _diarizer_pipe
        _diarizer_pipe = None
    _diarizer_profile = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(
    audio_path: str,
    transcribe: bool = False,
    extract_vocals_for_transcription: bool = True,
    lyrics_hint: Optional[str] = None,
) -> dict:
    """Analyze an audio file and return structured results.

    lyrics_hint: the song's KNOWN written lyrics, when available (the
    Director Music Video generate flow knows exactly what ACE-Step was
    given to sing). Seeds Whisper's initial_prompt so the transcription
    snaps to the real words → tighter lyric→clip timing. Ignored when
    transcribe=False; None (uploads, unknown tracks) keeps the previous
    behavior exactly.

    Updates the module-level _PROGRESS dict at each phase boundary
    so the frontend can poll /api/v1/audio/analyze/status during the
    long-running call and show meaningful status (e.g. "Downloading
    transcription model..." vs "Transcribing audio..."). The first
    run downloads ~500MB total of models silently inside the load
    helpers; the "loading X model" status is the user's signal that
    a download MAY be happening (we can't easily detect cache state
    across HF cache layouts so we don't try to differentiate
    "downloading" vs "loading from cache" — the parenthetical hint
    "first use downloads it" tells the user what to expect).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"[AudioAnalysis] Analyzing: {audio_path}")
    _set_progress("loading_audio", "Loading audio")

    import librosa
    y, sr = _load_audio(audio_path)
    duration = float(librosa.get_duration(y=y, sr=sr))
    original_sr = librosa.get_samplerate(audio_path)

    _set_progress("detecting_beats", "Detecting beats")
    bpm, beats = _detect_beats(y, sr)
    downbeats = _detect_downbeats(beats, bpm)
    onset_envelope = _compute_onset_envelope(y, sr)

    _set_progress("identifying_sections", "Identifying sections")
    sections = _segment_sections(y, sr, beats, duration)

    result = AudioAnalysis(
        duration=round(duration, 3),
        sample_rate=original_sr,
        bpm=round(bpm, 1),
        beats=beats,
        downbeats=[round(d, 3) for d in downbeats],
        sections=sections,
        onset_envelope=onset_envelope,
    )

    if transcribe:
        try:
            transcription_path = audio_path
            vocals_path = None

            if extract_vocals_for_transcription:
                try:
                    from preprocessing.extract_vocals import get_vocals
                    vocals_dir = os.path.join(os.path.dirname(audio_path), "vocals")
                    os.makedirs(vocals_dir, exist_ok=True)
                    vocals_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_vocals.wav"
                    vocals_path = os.path.join(vocals_dir, vocals_filename)

                    if not os.path.isfile(vocals_path):
                        # The first call to get_vocals() downloads the
                        # vocal-extraction model (~50MB) before running.
                        # Status string says "loading" with the "first use
                        # downloads" hint so users understand a long pause
                        # here means download, not a hang.
                        _set_progress("loading_vocal_model", "Loading vocal-extraction model (first use downloads ~50MB)")
                        print("[AudioAnalysis] Extracting vocals for transcription...")
                        _set_progress("extracting_vocals", "Extracting vocals")
                        vocals_path = get_vocals(audio_path, vocals_path)

                    transcription_path = vocals_path
                    result.vocals_path = vocals_path
                except Exception as e:
                    print(f"[AudioAnalysis] Vocal extraction failed, transcribing full mix: {e}")

            # _transcribe loads Whisper on first call (~300MB download
            # the very first time, cached after).
            _set_progress("loading_transcription_model", "Loading transcription model (first use downloads ~300MB)")
            print("[AudioAnalysis] Running transcription...")
            _set_progress("transcribing", "Transcribing audio")
            result.lyrics = _transcribe(transcription_path, lyrics_hint=lyrics_hint)

            # Run speaker diarization on the original mix (needs both voices)
            if result.lyrics:
                # _diarize loads pyannote on first call (~100MB cached).
                _set_progress("loading_diarization_model", "Loading speaker-diarization model (first use downloads ~30MB)")
                _set_progress("identifying_speakers", "Identifying speakers")
                result.lyrics = _diarize(audio_path, result.lyrics)
                unload_diarizer()  # Free VRAM immediately
            unload_whisper()  # Free Whisper VRAM before LLM loads
        except ImportError as e:
            print(f"[AudioAnalysis] Transcription skipped (faster-whisper not installed): {e}")
        except Exception as e:
            print(f"[AudioAnalysis] Transcription failed, continuing without lyrics: {e}")

    _set_progress("finalizing", "Finalizing")
    print(f"[AudioAnalysis] Done: {bpm:.1f} BPM, {len(beats)} beats, {len(sections)} sections")
    # Clear progress so subsequent /status polls don't show stale state.
    _set_progress("", "")
    return asdict(result)


def suggest_clip_boundaries(
    analysis: dict,
    clip_duration: float,
    total_duration: Optional[float] = None,
) -> List[dict]:
    """Suggest optimal clip boundaries aligned to musical structure."""
    song_duration = total_duration or analysis["duration"]
    sections = analysis.get("sections", [])
    downbeats = analysis.get("downbeats", [])

    clips = []
    current_time = 0.0

    while current_time < song_duration - 0.5:
        target_end = current_time + clip_duration
        actual_end = target_end

        # Snap to nearest downbeat within ±1 second
        for db in downbeats:
            if abs(db - target_end) < 1.0 and db > current_time + clip_duration * 0.5:
                actual_end = db
                break

        actual_end = min(actual_end, song_duration)

        # Find overlapping section
        clip_sections = [s for s in sections
                        if s["start"] < actual_end and s["end"] > current_time]
        primary_section = clip_sections[0]["label"] if clip_sections else "verse"
        primary_energy = clip_sections[0]["energy"] if clip_sections else 0.5

        energy_desc = "high energy" if primary_energy > 0.6 else ("low energy" if primary_energy < 0.3 else "moderate energy")

        clips.append({
            "start": round(current_time, 3),
            "end": round(actual_end, 3),
            "section_label": primary_section,
            "energy": round(primary_energy, 3),
            "suggested_prompt_hint": f"{primary_section}, {energy_desc}",
        })

        current_time = actual_end

    return clips


# ---------------------------------------------------------------------------
# Beat-aligned variable-duration clip structure
# ---------------------------------------------------------------------------


def _snap_to_valid_frames(
    duration_seconds: float,
    fps: int = 16,
    frames_steps: int = 4,
    frames_minimum: int = 5,
) -> int:
    """Convert a duration to the nearest valid frame count for WanGP.

    WanGP normalises frame counts via: (n-1) // latent_size * latent_size + 1
    so valid counts are 4n+1 = 5, 9, 13, 17, 21, ...
    The result is clamped to at least *frames_minimum* (model-specific).
    """
    raw_frames = round(duration_seconds * fps)
    snapped = ((raw_frames - 1) // frames_steps) * frames_steps + 1
    # Ensure we meet both the step-aligned minimum AND the model minimum
    floor = max(frames_steps + 1, frames_minimum)
    # If floor itself isn't step-aligned, round it up
    if (floor - 1) % frames_steps != 0:
        floor = ((floor - 1) // frames_steps + 1) * frames_steps + 1
    return max(floor, snapped)


def _find_speaker_change_beats(
    lyrics: list,
    beat_times: list,
    beat_duration: float,
) -> List[float]:
    """Find beat-snapped times where the speaker changes.

    Walks through lyrics in chronological order.  When the speaker changes
    from one segment to the next, records the nearest beat to the new
    segment's start time.
    """
    if not lyrics:
        return []

    change_points: List[float] = []
    prev_speaker = None
    for lyr in sorted(lyrics, key=lambda l: l.get("start", l["start"] if isinstance(l, dict) else l.start)):
        start = lyr["start"] if isinstance(lyr, dict) else lyr.start
        speaker = lyr.get("speaker") if isinstance(lyr, dict) else getattr(lyr, "speaker", None)
        if not speaker:
            continue
        if prev_speaker is not None and speaker != prev_speaker:
            # Speaker changed — snap to nearest beat
            best_beat = start
            best_dist = float("inf")
            for bt in beat_times:
                dist = abs(bt - start)
                if dist < best_dist:
                    best_dist = dist
                    best_beat = bt
                if bt > start + beat_duration:
                    break
            change_points.append(best_beat)
        prev_speaker = speaker

    return change_points


MAX_CLIP_SECONDS = 26.0  # user-validated single-window length on LTX-2.3 (was 22)
MIN_CLIP_SECONDS = 8.0   # don't create clips shorter than this


def plan_clip_structure(
    analysis: dict,
    energy_bias: int = 0,
    fps: int = 16,
    frames_steps: int = 4,
    frames_minimum: int = 5,
    total_duration: Optional[float] = None,
) -> List[dict]:
    """Plan variable-duration clips aligned to beat positions.

    Strategy: maximise clip duration (up to MAX_CLIP_SECONDS) to minimise
    clip count.  For each section, compute the fewest clips needed, then
    divide evenly and snap boundaries to the nearest beat.  Speaker changes
    can split a clip only when both halves remain >= MIN_CLIP_SECONDS.

    *energy_bias* shifts the preference: negative = longer clips,
    positive = shorter clips (adjusts MAX by ±2s per unit).

    Returns a list of clip dicts with ``beat_count`` and ``duration_frames``.
    """
    bpm = analysis.get("bpm", 120.0)
    beat_duration = 60.0 / bpm
    beats = analysis.get("beats", [])
    beat_times = sorted(b["time"] if isinstance(b, dict) else b.time for b in beats)
    sections = analysis.get("sections", [])
    song_duration = total_duration or analysis.get("duration", 180.0)

    if not beat_times:
        beat_times = [i * beat_duration for i in range(int(song_duration / beat_duration) + 1)]

    # energy_bias shifts the max clip length: -2 → 26s, 0 → 22s, +2 → 18s
    effective_max = max(MIN_CLIP_SECONDS + 2, MAX_CLIP_SECONDS - (energy_bias + 2) * 2)

    min_duration_from_frames = frames_minimum / fps
    min_clip_duration = max(MIN_CLIP_SECONDS, min_duration_from_frames)

    def _find_nearest_beat(target_time: float) -> float:
        if not beat_times:
            return target_time
        best = beat_times[0]
        for bt in beat_times:
            if abs(bt - target_time) < abs(best - target_time):
                best = bt
            if bt > target_time + beat_duration:
                break
        return best

    def _section_at(t: float) -> tuple:
        """Return (label, energy, section_end) for the section overlapping time *t*."""
        for s in sections:
            s_start = s["start"] if isinstance(s, dict) else s.start
            s_end = s["end"] if isinstance(s, dict) else s.end
            if s_start <= t < s_end:
                label = s["label"] if isinstance(s, dict) else s.label
                energy = s["energy"] if isinstance(s, dict) else s.energy
                return label, energy, s_end
        return "verse", 0.5, song_duration

    # Extract speaker change points from diarized lyrics
    lyrics = analysis.get("lyrics") or []
    speaker_changes = _find_speaker_change_beats(lyrics, beat_times, beat_duration)

    # ── Build section spans ──────────────────────────────────────────
    # Merge analysis sections into a flat list of (start, end, label, energy)
    section_spans = []
    if sections:
        for s in sections:
            s_start = s["start"] if isinstance(s, dict) else s.start
            s_end = s["end"] if isinstance(s, dict) else s.end
            label = s["label"] if isinstance(s, dict) else s.label
            energy = s["energy"] if isinstance(s, dict) else s.energy
            section_spans.append((s_start, min(s_end, song_duration), label, energy))
    if not section_spans:
        section_spans = [(0.0, song_duration, "verse", 0.5)]

    # ── Plan clips per section ───────────────────────────────────────
    clips: list = []

    for sec_start, sec_end, sec_label, sec_energy in section_spans:
        sec_duration = sec_end - sec_start
        if sec_duration < min_duration_from_frames:
            continue  # section too short for even one clip — skip

        # How many clips do we need for this section?
        # Allow up to 5% over effective_max as a single clip rather than
        # splitting into two clips that are each ~50% of max
        overshoot_tolerance = effective_max * 1.05
        if sec_duration <= overshoot_tolerance:
            num_clips = 1
        else:
            num_clips = max(1, int(math.ceil(sec_duration / effective_max)))
            # If splitting makes clips less than 75% of max, use fewer clips
            while num_clips > 1 and (sec_duration / num_clips) < effective_max * 0.75:
                num_clips -= 1
        target_clip_len = sec_duration / num_clips

        # Build evenly-spaced cut points within the section, snap to beats
        cut_points = [sec_start]
        for ci in range(1, num_clips):
            raw_cut = sec_start + ci * target_clip_len
            snapped = _find_nearest_beat(raw_cut)
            # Don't snap outside the section
            snapped = max(sec_start + min_clip_duration, min(snapped, sec_end - min_clip_duration))
            cut_points.append(round(snapped, 3))
        cut_points.append(sec_end)

        # Optionally split at speaker changes if both halves are long enough
        refined_cuts = [cut_points[0]]
        for ci in range(1, len(cut_points)):
            seg_start = refined_cuts[-1]
            seg_end = cut_points[ci]

            # Check for a speaker change that could split this segment
            best_split = None
            for sc in speaker_changes:
                if sc <= seg_start + min_clip_duration:
                    continue
                if sc >= seg_end - min_clip_duration:
                    continue
                # Valid split — both halves are long enough
                # Only split if both halves would be >= min_clip_duration
                if (sc - seg_start) >= min_clip_duration and (seg_end - sc) >= min_clip_duration:
                    best_split = sc
                    break  # take the first valid one

            if best_split is not None:
                refined_cuts.append(round(best_split, 3))
            refined_cuts.append(round(seg_end, 3))

        # Build clip dicts from the refined cut points
        for ci in range(len(refined_cuts) - 1):
            clip_start = refined_cuts[ci]
            clip_end = refined_cuts[ci + 1]
            clip_duration_s = clip_end - clip_start

            if clip_duration_s < min_duration_from_frames:
                # Merge with previous clip if too short
                if clips:
                    prev = clips[-1]
                    prev["end"] = clip_end
                    prev_dur = prev["end"] - prev["start"]
                    prev["beat_count"] = max(1, round(prev_dur / beat_duration))
                    prev["duration_frames"] = _snap_to_valid_frames(prev_dur, fps, frames_steps, frames_minimum)
                continue

            actual_beats = max(1, round(clip_duration_s / beat_duration))
            energy_desc = "high energy" if sec_energy > 0.6 else ("low energy" if sec_energy < 0.3 else "moderate energy")

            # Find dominant speaker in this clip
            clip_speaker = None
            if lyrics:
                speaker_counts: dict = {}
                for lyr in lyrics:
                    l_start = lyr["start"] if isinstance(lyr, dict) else lyr.start
                    l_end = lyr["end"] if isinstance(lyr, dict) else lyr.end
                    l_speaker = lyr.get("speaker") if isinstance(lyr, dict) else getattr(lyr, "speaker", None)
                    if l_start < clip_end and l_end > clip_start and l_speaker:
                        speaker_counts[l_speaker] = speaker_counts.get(l_speaker, 0) + 1
                if speaker_counts:
                    clip_speaker = max(speaker_counts, key=speaker_counts.get)

            clips.append({
                "start": round(clip_start, 3),
                "end": round(clip_end, 3),
                "beat_count": actual_beats,
                "section_label": sec_label,
                "energy": round(sec_energy, 3),
                "suggested_prompt_hint": f"{sec_label}, {energy_desc}",
                "duration_frames": _snap_to_valid_frames(clip_duration_s, fps, frames_steps, frames_minimum),
                "dominant_speaker": clip_speaker,
            })

    # Final merge pass: absorb any trailing runt clips
    if len(clips) >= 2:
        last = clips[-1]
        last_dur = last["end"] - last["start"]
        if last_dur < min_clip_duration:
            prev = clips[-2]
            prev["end"] = last["end"]
            prev_dur = prev["end"] - prev["start"]
            prev["beat_count"] = max(1, round(prev_dur / beat_duration))
            prev["duration_frames"] = _snap_to_valid_frames(prev_dur, fps, frames_steps, frames_minimum)
            clips.pop()

    return clips


# ---------------------------------------------------------------------------
# LLM-assisted section relabeling
# ---------------------------------------------------------------------------

_VALID_LABELS = {"intro", "verse", "chorus", "bridge", "outro", "instrumental"}


def classify_sections_with_lyrics(
    analysis: dict,
    section_labels: list,
) -> dict:
    """Replace section labels in an analysis dict with LLM-classified labels.

    Section boundaries and energy values remain unchanged —
    only the 'label' field of each section is updated.

    Args:
        analysis: Full analysis dict from analyze()
        section_labels: List of label strings from LLM, one per section

    Returns:
        New analysis dict with updated section labels
    """
    updated = dict(analysis)
    sections = [dict(s) for s in updated.get("sections", [])]

    for i, section in enumerate(sections):
        if i < len(section_labels):
            label = section_labels[i].lower().strip()
            if label in _VALID_LABELS:
                section["label"] = label

    updated["sections"] = sections
    return updated


def replace_sections_with_structure(
    analysis: dict,
    song_structure: list,
) -> dict:
    """Replace audio sections entirely with LLM-identified song structure.

    Uses the LLM's section boundaries and labels, and interpolates energy
    values from the original audio sections (weighted by overlap duration).

    Args:
        analysis: Full analysis dict from analyze()
        song_structure: List of dicts with keys: label, start
                        (from LLM classify_song_sections)

    Returns:
        New analysis dict with sections replaced by LLM structure
    """
    if not song_structure:
        return analysis

    original_sections = analysis.get("sections", [])
    duration = analysis.get("duration", 0)

    new_sections = []
    for i, ss in enumerate(song_structure):
        start = ss["start"]
        end = song_structure[i + 1]["start"] if i + 1 < len(song_structure) else duration
        label = ss.get("label", "verse")

        # Merge short sections (< 5s) into the previous section instead of
        # creating gaps — real song sections are always longer than this.
        if end - start < 5.0:
            if new_sections:
                new_sections[-1]["end"] = round(end, 3)
            continue

        # Compute energy: weighted average of overlapping original sections
        total_weight = 0.0
        weighted_energy = 0.0
        for orig in original_sections:
            overlap_start = max(start, orig["start"])
            overlap_end = min(end, orig["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > 0:
                total_weight += overlap
                weighted_energy += overlap * orig["energy"]

        energy = weighted_energy / total_weight if total_weight > 0 else 0.5

        new_sections.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "label": label,
            "energy": round(energy, 3),
        })

    updated = dict(analysis)
    updated["sections"] = new_sections
    return updated


# ---------------------------------------------------------------------------
# Dialogue-paced scene planning (Short Film mode)
# ---------------------------------------------------------------------------


def plan_dialogue_scenes(
    analysis: dict,
    pacing_bias: int = 0,
    fps: int = 16,
    frames_steps: int = 4,
    frames_minimum: int = 5,
) -> List[dict]:
    """Plan variable-duration clips based on dialogue pauses and speaker changes.

    Unlike ``plan_clip_structure`` which aligns to musical beats, this splits
    at natural speech pauses (silence gaps) and speaker transitions.

    *pacing_bias* works like energy_bias: negative = longer clips (slower pace),
    positive = shorter clips (faster pace).

    Returns a list of clip dicts compatible with the director pipeline.
    """

    duration = analysis.get("duration", 180.0)
    lyrics = analysis.get("lyrics") or []

    # Sort lyrics chronologically
    sorted_lyrics = sorted(
        [l for l in lyrics if l.get("text", "").strip()],
        key=lambda l: float(l.get("start", 0)),
    )

    # Pacing bias shifts max clip length: -2 → 22s, 0 → 18s, +2 → 14s
    effective_max = max(MIN_CLIP_SECONDS + 2, MAX_CLIP_SECONDS - (pacing_bias + 2) * 2)

    # ── Find natural cut points ─────────────────────────────────────
    # Cut points are: silence gaps > 1s between dialogue lines, and
    # sustained speaker changes (speaker changes with a gap or after 2+ lines).
    cut_candidates: List[float] = []

    SILENCE_THRESHOLD = 1.0  # seconds of silence to consider a scene break

    for i in range(1, len(sorted_lyrics)):
        prev_end = float(sorted_lyrics[i - 1].get("end", 0))
        curr_start = float(sorted_lyrics[i].get("start", 0))
        gap = curr_start - prev_end

        prev_speaker = sorted_lyrics[i - 1].get("speaker")
        curr_speaker = sorted_lyrics[i].get("speaker")

        # Natural pause — always a cut candidate
        if gap >= SILENCE_THRESHOLD:
            cut_candidates.append(curr_start)
        # Speaker change — cut candidate even with small gap
        elif prev_speaker and curr_speaker and prev_speaker != curr_speaker and gap >= 0.3:
            cut_candidates.append(curr_start)

    # Remove duplicates and sort
    cut_candidates = sorted(set(round(c, 3) for c in cut_candidates))

    # ── Build scenes from cut candidates ─────────────────────────────
    # Start with all cut candidates, then merge scenes that are too short
    # and split scenes that are too long.
    min_duration_from_frames = frames_minimum / fps
    min_clip = max(MIN_CLIP_SECONDS, min_duration_from_frames)

    # Build initial segments
    boundaries = [0.0] + cut_candidates + [duration]
    segments: List[tuple] = []
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        if seg_end - seg_start > 0.1:  # skip near-zero segments
            segments.append((seg_start, seg_end))

    # Merge segments that are too short
    merged: List[tuple] = []
    for seg_start, seg_end in segments:
        if merged and (seg_end - seg_start) < min_clip:
            # Merge with previous
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, seg_end)
        else:
            merged.append((seg_start, seg_end))

    # Handle last segment being too short
    if len(merged) >= 2:
        last_start, last_end = merged[-1]
        if (last_end - last_start) < min_clip:
            prev_start, _ = merged[-2]
            merged[-2] = (prev_start, last_end)
            merged.pop()

    # Split segments that exceed effective_max
    # Allow up to 5% over to avoid splitting into clips much shorter than max
    overshoot_tolerance = effective_max * 1.05
    final_segments: List[tuple] = []
    for seg_start, seg_end in merged:
        seg_dur = seg_end - seg_start
        if seg_dur <= overshoot_tolerance:
            final_segments.append((seg_start, seg_end))
        else:
            # Split evenly into sub-segments
            num_parts = max(2, int(math.ceil(seg_dur / effective_max)))
            # If splitting makes clips less than 75% of max, use fewer parts
            while num_parts > 1 and (seg_dur / num_parts) < effective_max * 0.75:
                num_parts -= 1
            part_dur = seg_dur / num_parts
            for p in range(num_parts):
                p_start = seg_start + p * part_dur
                p_end = seg_start + (p + 1) * part_dur if p < num_parts - 1 else seg_end

                # Try to snap split point to a dialogue gap
                if p > 0:
                    best_gap_time = p_start
                    best_gap_dist = float("inf")
                    for cc in cut_candidates:
                        dist = abs(cc - p_start)
                        if dist < best_gap_dist and dist < part_dur * 0.3:
                            best_gap_dist = dist
                            best_gap_time = cc
                    p_start = best_gap_time
                    if final_segments:
                        # Update previous segment's end
                        prev = final_segments[-1]
                        final_segments[-1] = (prev[0], p_start)

                final_segments.append((round(p_start, 3), round(p_end, 3)))

    # ── Build clip dicts ────────────────────────────────────────────
    clips: List[dict] = []

    for i, (clip_start, clip_end) in enumerate(final_segments):
        clip_dur = clip_end - clip_start
        if clip_dur < min_duration_from_frames:
            # Merge with previous
            if clips:
                prev = clips[-1]
                prev["end"] = clip_end
                prev_dur = prev["end"] - prev["start"]
                prev["duration_frames"] = _snap_to_valid_frames(prev_dur, fps, frames_steps, frames_minimum)
            continue

        # Find dominant speaker in this clip
        clip_speaker = None
        speaker_counts: dict = {}
        clip_dialogue: List[str] = []
        for lyr in sorted_lyrics:
            l_start = float(lyr.get("start", 0))
            l_end = float(lyr.get("end", 0))
            l_speaker = lyr.get("speaker")
            if l_start < clip_end and l_end > clip_start:
                if l_speaker:
                    speaker_counts[l_speaker] = speaker_counts.get(l_speaker, 0) + 1
                clip_dialogue.append(lyr.get("text", "").strip())
        if speaker_counts:
            clip_speaker = max(speaker_counts, key=speaker_counts.get)

        # Determine scene label based on position and content
        if clip_start < 2.0 and clip_dur < 15:
            scene_label = "opening"
        elif clip_end >= duration - 2.0 and clip_dur < 15:
            scene_label = "closing"
        elif not clip_dialogue:
            scene_label = "action"
        else:
            scene_label = "dialogue"

        clips.append({
            "start": round(clip_start, 3),
            "end": round(clip_end, 3),
            "beat_count": 0,  # Not applicable for dialogue scenes
            "section_label": scene_label,
            "energy": 0.5,
            "suggested_prompt_hint": f"Scene {i + 1}: {scene_label}",
            "duration_frames": _snap_to_valid_frames(clip_dur, fps, frames_steps, frames_minimum),
            "dominant_speaker": clip_speaker,
            "dialogue_lines": clip_dialogue,
        })

    # Final merge pass: absorb trailing runt clips
    if len(clips) >= 2:
        last = clips[-1]
        last_dur = last["end"] - last["start"]
        if last_dur < min_clip:
            prev = clips[-2]
            prev["end"] = last["end"]
            prev_dur = prev["end"] - prev["start"]
            prev["duration_frames"] = _snap_to_valid_frames(prev_dur, fps, frames_steps, frames_minimum)
            prev["dialogue_lines"] = prev.get("dialogue_lines", []) + last.get("dialogue_lines", [])
            clips.pop()

    # If no lyrics at all, create evenly-spaced clips
    if not clips:
        if duration <= overshoot_tolerance:
            num_clips = 1
        else:
            num_clips = max(1, int(math.ceil(duration / effective_max)))
            while num_clips > 1 and (duration / num_clips) < effective_max * 0.75:
                num_clips -= 1
        clip_dur = duration / num_clips
        for i in range(num_clips):
            c_start = i * clip_dur
            c_end = min((i + 1) * clip_dur, duration)
            clips.append({
                "start": round(c_start, 3),
                "end": round(c_end, 3),
                "beat_count": 0,
                "section_label": "scene",
                "energy": 0.5,
                "suggested_prompt_hint": f"Scene {i + 1}",
                "duration_frames": _snap_to_valid_frames(c_end - c_start, fps, frames_steps, frames_minimum),
                "dominant_speaker": None,
                "dialogue_lines": [],
            })

    print(f"[AudioAnalysis] Dialogue scene plan: {len(clips)} clips from {duration:.1f}s audio")
    return clips
