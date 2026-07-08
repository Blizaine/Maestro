"""Voice-clone postprocessing — apply SeedVC to a generated video's audio
track to replace 1 or 2 voices with user-supplied reference voices.

Mirrors WanGP v11.75/v11.77's "SeedVC across any video" feature.

Usage:
    from postprocessing.voice_clone import apply_voice_clone_to_file

    # Single-voice replacement (whole audio track → reference voice)
    apply_voice_clone_to_file(
        video_path="/path/to/video.mp4",
        voice_ref_paths=["/path/to/voice_ref.wav"],
        mode="single",
    )

    # Two-voice replacement (auto-diarize, first speaker → ref[0], second → ref[1])
    apply_voice_clone_to_file(
        video_path="/path/to/video.mp4",
        voice_ref_paths=["/path/to/voice_a.wav", "/path/to/voice_b.wav"],
        mode="two",
    )

Backend approach:
  1. Demux audio from video (ffmpeg → 22050Hz mono wav, matches SeedVC's
     native rate so we skip a resample pass).
  2. Single-voice path:
       Apply SeedVC.convert_tensor() to the entire audio with the
       single reference voice.
  3. Two-voice path:
       a. Diarize the audio with pyannote 3.1 (the same pipeline
          Music Video mode uses, reached directly without going
          through the LyricSegment-overlay machinery).
       b. Map each detected speaker_id to one of the two refs by
          appearance order: first speaker_id encountered → ref[0],
          second → ref[1]. (Order-based is simpler than embedding-
          based for v1. A future improvement would CAMPPlus-embed
          each detected speaker + each ref and match by cosine
          similarity — see project_cross_shot_voice_consistency.md
          for the design.)
       c. For each diarized speech segment, slice the audio, convert
          with the mapped reference, splice back into position.
       d. Non-speech gaps (silence, music, SFX) are kept ORIGINAL —
          this is how WanGP preserves background music.
  4. Remux: ffmpeg replaces the video's audio with the converted track
     in-place. Video stream is copied (no re-encode), so no quality
     loss on the visual side.

Background preservation (added 2026-05-31):
  * BOTH modes isolate the vocals from the background (music / SFX) via the
    RoFormer separator (preprocessing.extract_vocals.get_stems) FIRST,
    convert ONLY the vocals, then remix them over the ORIGINAL background.
    This also improves conversion quality (SeedVC sees clean vocals, not a
    voice+music mix). If the separator is unavailable, single mode falls
    back to whole-track conversion (background lost) and logs a warning.

Known limitations (documented for future improvement):
  * Two-voice speaker→ref mapping is order-based, not voice-similarity-
    based. If diarization assigns IDs in an unexpected order, voices
    end up swapped — user can re-run with the refs swapped.
  * No SeedVC v2 support yet (WanGP v11.77 added it as an Extension).
    Only v1 (which supports singing) is wired up.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

import torch
import torchaudio


def _ffmpeg_demux_audio(video_path: str, out_wav: str, sample_rate: int = 22050) -> bool:
    """Extract the video's audio to a mono WAV at the given sample rate.
    Returns True on success, False if the video had no audio or ffmpeg failed.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ac", "1", "-ar", str(sample_rate),
                "-acodec", "pcm_s16le",
                out_wav,
            ],
            capture_output=True, text=True, timeout=300,
        )
    except Exception as e:
        print(f"[VoiceClone] ffmpeg demux failed: {e}")
        return False
    if result.returncode != 0:
        # Common case: video has no audio track. Returncode!=0, stderr explains.
        stderr_tail = (result.stderr or "")[-300:]
        if "does not contain any stream" in stderr_tail or "Stream map" in stderr_tail:
            print(f"[VoiceClone] Video has no audio track — skipping voice clone")
        else:
            print(f"[VoiceClone] ffmpeg demux non-zero exit: {stderr_tail}")
        return False
    return os.path.isfile(out_wav) and os.path.getsize(out_wav) > 100


def _ffmpeg_remux_audio(video_path: str, new_audio_wav: str) -> bool:
    """Replace `video_path`'s audio track with the contents of
    `new_audio_wav`. Video stream is copied (no re-encode); audio is
    re-encoded to AAC for container compatibility.

    Preferred outcome: atomic in-place replace of video_path.

    Windows fallback: when the browser is actively streaming the
    video file via HTTP range requests (common when the user has the
    just-generated clip open in the gallery), os.replace() fails with
    WinError 5 even after retries. In that case we DON'T lose the
    cloned content — we rename the temp file to a user-visible
    sibling `<stem>_voicecloned<ext>` in the same directory. The
    gallery picks it up on next refresh; the user gets both the
    original AND the cloned version.

    Returns True on success (either in-place or sibling-rename).
    Returns False only when ffmpeg itself fails to produce the
    cloned file.
    """
    tmp_out = video_path + ".voicecloned_tmp.mp4"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", new_audio_wav,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                tmp_out,
            ],
            capture_output=True, text=True, timeout=300,
        )
    except Exception as e:
        print(f"[VoiceClone] ffmpeg remux failed: {e}")
        if os.path.isfile(tmp_out):
            try: os.remove(tmp_out)
            except OSError: pass
        return False
    if result.returncode != 0 or not os.path.isfile(tmp_out):
        stderr_tail = (result.stderr or "")[-300:]
        print(f"[VoiceClone] ffmpeg remux non-zero exit: {stderr_tail}")
        if os.path.isfile(tmp_out):
            try: os.remove(tmp_out)
            except OSError: pass
        return False

    # Try atomic in-place replace. Wider window than film-grain's
    # because voice clone runs after the user may already be playing
    # the just-generated clip in the gallery — file handles persist
    # longer than for typical generation completion.
    import time, gc
    for attempt in range(30):  # 30 × 0.5s = 15s — long enough for a streaming burst to finish
        try:
            os.replace(tmp_out, video_path)
            return True
        except (PermissionError, OSError):
            if attempt < 29:
                gc.collect()
                time.sleep(0.5)
                continue
            # All retries exhausted. Don't delete the cloned content —
            # rename to a user-visible sibling so the work is preserved
            # and the gallery can show both versions.
            stem, ext = os.path.splitext(video_path)
            sibling_path = f"{stem}_voicecloned{ext}"
            # Avoid overwriting if a previous run left one
            n = 1
            while os.path.isfile(sibling_path):
                sibling_path = f"{stem}_voicecloned_{n}{ext}"
                n += 1
            try:
                os.replace(tmp_out, sibling_path)
                print(f"[VoiceClone] Could not replace in-place (file locked by another reader); "
                      f"saved cloned version as sibling: {os.path.basename(sibling_path)}")
                return True
            except Exception as rename_err:
                print(f"[VoiceClone] Sibling-rename also failed: {rename_err}")
                # Last resort — leave the .voicecloned_tmp.mp4 file so the
                # user can rename it manually rather than losing the work.
                print(f"[VoiceClone] Cloned content preserved at: {tmp_out}")
                return False
    return False


def _load_reference_voice(ref_path: str, max_seconds: float = 25.0) -> tuple[torch.Tensor, int]:
    """Load a reference voice file (any format ffmpeg-readable), return
    (mono float32 tensor [1, samples], sample_rate). Trims to max_seconds.
    SeedVC's convert_tensor handles further resampling internally.
    """
    waveform, sr = torchaudio.load(ref_path)
    # Mixdown to mono
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    # Trim to max_seconds
    max_samples = int(max_seconds * sr)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]
    return waveform.float(), int(sr)


def _diarize_audio_for_segments(audio_path: str, num_speakers: Optional[int] = None) -> list[tuple[float, float, str]]:
    """Run pyannote 3.1 diarization on `audio_path`, return list of
    (start_sec, end_sec, speaker_id) segments sorted by start time.
    Returns empty list if pyannote isn't available or diarization fails.

    Args:
        audio_path: Path to the audio file to diarize.
        num_speakers: When set, forces pyannote to cluster speech into
            exactly this many speakers (no more, no less). Use this when
            the caller knows the expected speaker count — e.g. two-voice
            mode passes num_speakers=2 because the user explicitly
            supplied 2 voice references. Without this, pyannote may
            over-cluster the same voice into multiple speaker IDs when
            audio context shifts across the clip (e.g. across sliding
            windows in a long generation), causing identical voices to
            get DIFFERENT speaker labels in different segments.
            Default None = auto-detect speaker count.

    Reuses the shared loader from services.audio_analysis (which
    handles the HF_HOME + canonical-name trick + sub-model resolution
    + torch.load weights_only patch). Sharing the loader means we also
    share the cached pipeline instance — second+ calls in the same
    process are free.

    Bug history: an earlier version of this function tried to call
    pyannote's Pipeline.from_pretrained() with the local cache PATH
    directly. That fails because pyannote.from_pretrained expects
    either a HuggingFace identifier OR a path with a specific layout
    of sub-files. The right approach is to set HF_HOME and pass the
    canonical name — which is what get_diarizer_pipeline() does.

    User-reported issue that prompted num_speakers: on a 40s 2-sliding-
    window clip with a known-2-voice TTS source, pyannote found 3
    speakers ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02'] — the SAME
    male voice got assigned SPEAKER_00 in window 1 and SPEAKER_02 in
    window 2. The unmapped SPEAKER_02 segments skipped conversion
    entirely. Passing num_speakers=2 collapses the clustering to
    exactly 2 groups, fixing this.
    """
    try:
        from services.audio_analysis import get_diarizer_pipeline
    except Exception as e:
        print(f"[VoiceClone] could not import diarizer loader: {e}")
        return []

    pipeline = get_diarizer_pipeline()
    if pipeline is None:
        print(f"[VoiceClone] pyannote pipeline unavailable — skipping two-voice diarization")
        return []

    # Pyannote's pipeline can accept either a file path or a
    # {"waveform": tensor, "sample_rate": int} dict. Pass an audio
    # data dict so we don't re-decode via ffmpeg (which the pipeline
    # would otherwise do internally).
    try:
        import torch as _torch
        import numpy as np
        import subprocess
        # 16kHz mono is pyannote's preferred internal rate.
        cmd = [
            "ffmpeg", "-nostdin", "-threads", "0", "-i", audio_path,
            "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        audio_np = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        # Diagnostic: when diarization returns 0 segments, this tells us
        # whether the audio is silent / too short / clipped — actionable
        # info before assuming pyannote is broken.
        duration_sec = len(audio_np) / 16000.0
        peak_amp = float(np.abs(audio_np).max(initial=0.0))
        rms_amp = float(np.sqrt(np.mean(audio_np ** 2)))
        print(f"[VoiceClone] Audio: {duration_sec:.2f}s @ 16kHz mono, "
              f"peak={peak_amp:.3f}, RMS={rms_amp:.3f}")
        if duration_sec < 2.0:
            print(f"[VoiceClone] WARNING: audio is shorter than 2s — pyannote VAD often misses speech in clips this short")
        if peak_amp < 0.05:
            print(f"[VoiceClone] WARNING: audio peak is very low ({peak_amp:.3f}) — may be effectively silent")
        audio_data = {
            "waveform": _torch.from_numpy(audio_np[None, :]),
            "sample_rate": 16000,
        }
        if num_speakers is not None and num_speakers > 0:
            print(f"[VoiceClone] Running diarization on {os.path.basename(audio_path)} "
                  f"(forcing num_speakers={num_speakers})...")
            diarization = pipeline(audio_data, num_speakers=num_speakers)
        else:
            print(f"[VoiceClone] Running diarization on {os.path.basename(audio_path)}...")
            diarization = pipeline(audio_data)
    except Exception as e:
        print(f"[VoiceClone] diarization run failed: {e}")
        return []

    segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), str(speaker)))
    segments.sort(key=lambda s: s[0])
    speakers_found = sorted(set(s[2] for s in segments))
    print(f"[VoiceClone] Diarization found {len(segments)} segments across {len(speakers_found)} speaker(s): {speakers_found}")
    return segments


def _remix_vocals_with_background(vocals_wav: str, background_wav: str, out_wav: str) -> str:
    """Sum converted vocals with the original separated background (music /
    SFX), aligning sample rate + channels (mono) + length. Writes and returns
    out_wav."""
    voc, voc_sr = torchaudio.load(vocals_wav)
    bg, bg_sr = torchaudio.load(background_wav)
    if bg_sr != voc_sr:
        bg = torchaudio.functional.resample(bg, bg_sr, voc_sr)
    if voc.ndim == 2 and voc.shape[0] > 1:
        voc = voc.mean(dim=0, keepdim=True)
    if bg.ndim == 2 and bg.shape[0] > 1:
        bg = bg.mean(dim=0, keepdim=True)
    n = max(voc.shape[-1], bg.shape[-1])
    if voc.shape[-1] < n:
        voc = torch.nn.functional.pad(voc, (0, n - voc.shape[-1]))
    if bg.shape[-1] < n:
        bg = torch.nn.functional.pad(bg, (0, n - bg.shape[-1]))
    mixed = voc + bg
    peak = float(mixed.abs().max())
    if peak > 1.0:
        mixed = mixed / peak  # prevent clipping after the sum
    torchaudio.save(out_wav, mixed.float(), voc_sr)
    return out_wav


def apply_voice_clone_to_file(
    video_path: str,
    voice_ref_paths: list[str],
    mode: str = "single",
    diffusion_steps: int = 25,
    cfg_rate: float = 0.5,
) -> bool:
    """Replace the voice(s) in `video_path` using SeedVC voice conversion.

    Args:
        video_path: Path to the generated video (modified in-place on success).
        voice_ref_paths: List of reference voice file paths. mode='single'
            uses voice_ref_paths[0]; mode='two' uses voice_ref_paths[:2].
        mode: 'single' = convert whole audio with one reference voice;
              'two' = diarize and replace per-speaker (preserves non-
              speech audio like background music).
        diffusion_steps: SeedVC diffusion steps (default 25; lower=faster).
        cfg_rate: SeedVC CFG rate (default 0.5).

    Returns:
        True if the video's audio was replaced. False if the video had
        no audio, SeedVC failed to load, or any step in the pipeline
        broke — in which case the video is left UNCHANGED.

    The video stream is never re-encoded — only the audio is replaced.
    """
    if not os.path.isfile(video_path):
        print(f"[VoiceClone] video not found: {video_path}")
        return False
    if not voice_ref_paths:
        print(f"[VoiceClone] no voice references supplied — skipping")
        return False
    for p in voice_ref_paths:
        if not os.path.isfile(p):
            print(f"[VoiceClone] voice reference not found: {p}")
            return False
    if mode not in ("single", "two"):
        print(f"[VoiceClone] invalid mode {mode!r} (expected 'single' or 'two')")
        return False
    if mode == "two" and len(voice_ref_paths) < 2:
        print(f"[VoiceClone] mode='two' requires 2 voice references; falling back to single mode")
        mode = "single"

    # Lazy-import SeedVC — heavy module load, only pay it when actually used.
    try:
        from postprocessing import seedvc as _seedvc
    except Exception as e:
        print(f"[VoiceClone] SeedVC not available: {e} — run Update in Pinokio "
              "to fetch the seed-vc component (github.com/Blizaine/maestro-seedvc)")
        return False

    with tempfile.TemporaryDirectory(prefix="voice_clone_") as tmpdir:
        # Step 1: demux audio
        source_wav = os.path.join(tmpdir, "source.wav")
        # Demux at 44.1kHz (the RoFormer separator's native rate) instead of
        # SeedVC's lower native rate, so the separated background keeps full-band
        # music quality. SeedVC resamples its own input internally, so the voice
        # conversion is unaffected.
        if not _ffmpeg_demux_audio(video_path, source_wav, sample_rate=44100):
            return False

        # Isolate vocals from the background (music / SFX) up front, so we
        # convert ONLY the vocals and remix them back over the untouched
        # background. Without this, single-voice mode re-synthesizes the whole
        # mixed track and the background is destroyed. Runs before SeedVC loads
        # so the two models don't share VRAM. Falls back to full-track
        # conversion if the separator is unavailable.
        audio_for_conversion = source_wav
        instrumental_path = None
        try:
            from preprocessing.extract_vocals import get_stems
            _vocals_path, instrumental_path = get_stems(source_wav, os.path.join(tmpdir, "stems"))
            audio_for_conversion = _vocals_path
            print("[VoiceClone] Isolated vocals from background — converting vocals only; will remix over original background.")
        except Exception as e:
            audio_for_conversion = source_wav
            instrumental_path = None
            print(f"[VoiceClone] Vocal separation unavailable ({e}) — converting the FULL track (background will NOT be preserved).")

        # Step 2: ensure SeedVC assets are downloaded + load converter
        try:
            _seedvc.download_assets()
        except Exception as e:
            print(f"[VoiceClone] SeedVC asset download failed: {e}")
            return False
        try:
            converter = _seedvc.get_model(
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
        except Exception as e:
            print(f"[VoiceClone] SeedVC model load failed: {e}")
            return False

        # SeedVC's class hardcodes `app_vc.device = torch.device("cpu")` at
        # load time (seedvc/__init__.py:175). It's designed to be loaded on
        # CPU and moved to GPU by Maestro's mmgp offload system at runtime.
        # Scenema integrates with mmgp and gets GPU placement automatically;
        # we use SeedVC standalone for postprocessing, so we have to move
        # the modules ourselves.
        #
        # SeedVC's `forward()` calls `_runtime_device(self.pipe_modules())`
        # which infers the actual device from where the modules ARE at
        # call time. So manually moving each pipe_module to CUDA here
        # makes inference run on GPU. Without this, voice cloning takes
        # 5-10x longer because SeedVC runs on CPU even though CUDA is
        # available.
        if torch.cuda.is_available():
            try:
                gpu_device = torch.device("cuda")
                moved = 0
                for name, module in converter.pipe_modules().items():
                    try:
                        module.to(gpu_device)
                        moved += 1
                    except Exception as e:
                        print(f"[VoiceClone] could not move {name} to GPU: {e}")
                # Also flip the app_vc.device flag so audio tensors get
                # placed on the right device inside forward(). Belt-and-
                # suspenders — _runtime_device should set this too, but
                # being explicit here saves one call-site computation.
                if converter._app_vc is not None:
                    converter._app_vc.device = gpu_device
                print(f"[VoiceClone] SeedVC moved {moved} modules to CUDA for inference")
            except Exception as e:
                print(f"[VoiceClone] SeedVC GPU move failed (falling back to CPU): {e}")

        # Step 3: load source audio (the isolated vocals when separation
        # succeeded, else the full track).
        source_audio, source_sr = torchaudio.load(audio_for_conversion)
        if source_audio.ndim == 2 and source_audio.shape[0] > 1:
            source_audio = source_audio.mean(dim=0, keepdim=True)

        # Step 4: load reference voices
        refs: list[tuple[torch.Tensor, int]] = []
        for ref_path in voice_ref_paths[: (2 if mode == "two" else 1)]:
            refs.append(_load_reference_voice(ref_path))

        # Step 5: apply conversion
        out_wav = os.path.join(tmpdir, "converted.wav")
        if mode == "single":
            print(f"[VoiceClone] Single-voice mode: converting entire audio with ref={os.path.basename(voice_ref_paths[0])}")
            converted = converter.convert_tensor(
                source_audio=source_audio,
                source_rate=source_sr,
                reference_audio=refs[0][0],
                reference_rate=refs[0][1],
                output_rate=source_sr,
                diffusion_steps=diffusion_steps,
                cfg_rate=cfg_rate,
            )
            # Save converted audio (stereo from SeedVC; downmix to mono is fine
            # for our remux). torchaudio.save accepts (channels, samples).
            torchaudio.save(out_wav, converted.cpu().float(), source_sr)
        else:
            # Two-voice mode with diarization. The user has supplied
            # exactly 2 voice references, so we assert num_speakers=2 to
            # pyannote — this prevents the over-clustering bug where the
            # same voice gets different SPEAKER_IDs across sliding
            # windows. See _diarize_audio_for_segments() docstring.
            segments = _diarize_audio_for_segments(audio_for_conversion, num_speakers=2)
            if not segments:
                print(f"[VoiceClone] No diarized segments — falling back to single-voice with ref[0]")
                converted = converter.convert_tensor(
                    source_audio=source_audio, source_rate=source_sr,
                    reference_audio=refs[0][0], reference_rate=refs[0][1],
                    output_rate=source_sr, diffusion_steps=diffusion_steps, cfg_rate=cfg_rate,
                )
                torchaudio.save(out_wav, converted.cpu().float(), source_sr)
            else:
                # Build speaker→ref mapping by order of first appearance.
                speaker_to_ref: dict[str, int] = {}
                for _start, _end, spk in segments:
                    if spk not in speaker_to_ref:
                        speaker_to_ref[spk] = len(speaker_to_ref)
                        if len(speaker_to_ref) >= 2:
                            break
                print(f"[VoiceClone] Speaker→ref mapping: {speaker_to_ref}")

                # Build output audio: start with original (preserves silence,
                # music, SFX between speech segments), overlay converted speech.
                output_audio = source_audio.clone()
                total_samples = output_audio.shape[-1]

                converted_count = 0
                skipped_count = 0
                for seg_idx, (start_sec, end_sec, spk) in enumerate(segments):
                    ref_idx = speaker_to_ref.get(spk)
                    if ref_idx is None or ref_idx >= len(refs):
                        skipped_count += 1
                        continue  # speaker beyond our 2 refs — leave original
                    start_sample = max(0, int(start_sec * source_sr))
                    end_sample = min(total_samples, int(end_sec * source_sr))
                    if end_sample <= start_sample:
                        continue
                    seg_audio = source_audio[..., start_sample:end_sample]
                    if seg_audio.shape[-1] < int(0.3 * source_sr):
                        # Skip very-short segments (< 300ms) — SeedVC needs
                        # enough content to produce stable output.
                        continue
                    print(f"[VoiceClone] Segment {seg_idx+1}/{len(segments)}: "
                          f"{start_sec:.2f}-{end_sec:.2f}s speaker={spk} → ref[{ref_idx}]")
                    try:
                        seg_converted = converter.convert_tensor(
                            source_audio=seg_audio, source_rate=source_sr,
                            reference_audio=refs[ref_idx][0], reference_rate=refs[ref_idx][1],
                            output_rate=source_sr,
                            diffusion_steps=diffusion_steps, cfg_rate=cfg_rate,
                        )
                    except Exception as e:
                        print(f"[VoiceClone]   conversion failed (keeping original): {e}")
                        skipped_count += 1
                        continue
                    # seg_converted is [2, N] stereo — downmix to [1, N] mono
                    # to match output_audio's shape.
                    if seg_converted.ndim == 2 and seg_converted.shape[0] > 1:
                        seg_converted = seg_converted.mean(dim=0, keepdim=True)
                    # Truncate or pad to match original segment length exactly
                    # so the splice doesn't shift the timeline.
                    target_len = end_sample - start_sample
                    cur_len = seg_converted.shape[-1]
                    if cur_len > target_len:
                        seg_converted = seg_converted[..., :target_len]
                    elif cur_len < target_len:
                        pad = torch.zeros((seg_converted.shape[0], target_len - cur_len),
                                          dtype=seg_converted.dtype, device=seg_converted.device)
                        seg_converted = torch.cat([seg_converted, pad], dim=-1)
                    output_audio[..., start_sample:end_sample] = seg_converted.cpu().float()
                    converted_count += 1
                print(f"[VoiceClone] Two-voice mode complete: converted={converted_count}, "
                      f"skipped={skipped_count} (non-mapped or too-short segments)")
                torchaudio.save(out_wav, output_audio.cpu().float(), source_sr)

        # Step 5b: remix converted vocals back over the original background
        # (music / SFX) so the background is preserved. No-op when separation
        # was unavailable (instrumental_path is None → full-track conversion).
        if instrumental_path:
            try:
                remixed_wav = os.path.join(tmpdir, "remixed.wav")
                _remix_vocals_with_background(out_wav, instrumental_path, remixed_wav)
                out_wav = remixed_wav
                print("[VoiceClone] Remixed converted vocals with original background.")
            except Exception as e:
                print(f"[VoiceClone] Remix failed ({e}) — using converted vocals without background.")

        # Step 6: remux
        if not _ffmpeg_remux_audio(video_path, out_wav):
            print(f"[VoiceClone] remux failed — video left unchanged")
            return False

        print(f"[VoiceClone] Voice clone applied successfully to {os.path.basename(video_path)}")
        return True
