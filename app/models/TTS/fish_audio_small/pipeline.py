"""Maestro adapter for the streamed audio.cpp Fish Drama 3 Small runtime."""

from __future__ import annotations

import gc
import json
import math
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from ..fish_audio.duration import duration_frame_budget
from ..fish_audio.prompt import (
    check_paired_tags,
    drama3_text_chunks,
    reference_transcripts,
    speaker_lines_to_tags,
    validate_drama3_text,
)
from .small_runtime import (
    MAX_STREAM_SECONDS,
    SAMPLE_RATE,
    AudioCppServer,
    MemoryPlan,
    NativeStreamError,
    NativeStreamInterrupted,
    StreamResult,
    choose_memory_plan,
)


FRAME_LENGTH = 2048
MAX_PART_FRAMES = math.floor(MAX_STREAM_SECONDS * SAMPLE_RATE / FRAME_LENGTH)
MAX_PART_SAMPLES = MAX_PART_FRAMES * FRAME_LENGTH
MAX_JOB_SECONDS = 3600
MIN_REFERENCE_SECONDS = 1.0
MAX_REFERENCE_SECONDS = 30.0
MAX_REFERENCE_DECODE_SAMPLES = 16_000_000  # At most 64 MB of float32 input.
SPOOL_MEMORY_BYTES = 4 * 1024 * 1024
MAX_TEXT_UNITS_PER_PART = 88


class FishSmallPipeline:
    """Generate speech through a model-owned loopback audio.cpp child.

    Only the final Maestro waveform tensor is materialized in memory. Native
    audio deltas are decoded one bounded WAV event at a time and spooled to
    disk once the small in-memory threshold is crossed.
    """

    sample_rate = SAMPLE_RATE
    frame_length = FRAME_LENGTH

    def __init__(
        self,
        model_path: str | Path,
        runtime_info: Any,
        memory_plan: MemoryPlan,
        *,
        server: AudioCppServer | None = None,
        server_factory: Callable[..., AudioCppServer] = AudioCppServer,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.runtime_info = runtime_info
        self.memory_plan = memory_plan
        self.server = server or server_factory(runtime_info, self.model_path, memory_plan)
        self.last_longform_report: dict[str, Any] | None = None
        self.last_finish_reason: str | None = None
        self.last_duration_budget_frames: int | None = None
        self.last_duration_is_auto = False
        # WGP still asks external pipelines for a transformer-shaped cache
        # target. This parameter-free CPU module satisfies that lifecycle hook
        # without creating a second model or moving work into PyTorch.
        self.model = torch.nn.Identity()
        self.model.cache = None
        self.cache = None
        self._interrupt_lock = threading.RLock()
        self._child_release_sent = False
        self._reference_temp_owner: tempfile.TemporaryDirectory[str] | None = None
        self._interrupt = False
        self._early_stop = False
        self._active_spool: Any = None
        self._active_pcm_samples = 0
        self._streamed_pcm_samples = 0
        self._released = False

    @property
    def _interrupt(self) -> bool:
        return self.__interrupt

    @_interrupt.setter
    def _interrupt(self, value: bool) -> None:
        value = bool(value)
        with getattr(self, "_interrupt_lock", threading.RLock()):
            self.__interrupt = value
        if value:
            self._release_server_once()

    def request_early_stop(self) -> None:
        """Stop after preserving audio already streamed from the current part."""
        with self._interrupt_lock:
            if self._early_stop:
                return
            self._early_stop = True
        self._release_server_once()

    def _release_server_once(self) -> None:
        with self._interrupt_lock:
            if self._child_release_sent:
                return
            self._child_release_sent = True
            server = self.server
        server.release()

    def _check_cancelled(self) -> bool:
        with self._interrupt_lock:
            return self.__interrupt or self._early_stop

    @staticmethod
    def _status(set_progress_status: Callable[[str], None] | None, text: str) -> None:
        if callable(set_progress_status):
            set_progress_status(text)

    def generate(
        self,
        input_prompt: str,
        *,
        alt_prompt: str = "",
        audio_guide: str | None = None,
        audio_guide2: str | None = None,
        audio_guide3: str | None = None,
        audio_prompt_type: str = "",
        duration_seconds: float = 0,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.7,
        seed: int = -1,
        custom_settings: Mapping[str, Any] | None = None,
        callback: Callable[..., Any] | None = None,
        set_progress_status: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        del kwargs
        self._cleanup_reference_temp()
        with self._interrupt_lock:
            if self._released:
                raise RuntimeError("Fish Small pipeline was released")
            self.__interrupt = False
            self._early_stop = False
            # A cancellation owns one child-process release per generation.
            # WGP may reassert its sticky abort flag while this generation exits.
            self._child_release_sent = False
        self.last_finish_reason = None
        self.last_longform_report = None

        script = str(input_prompt or "")
        tagged_script = speaker_lines_to_tags(script)
        validate_drama3_text(tagged_script, "Speech Script")
        check_paired_tags(script, "Speech Script")
        settings = dict(custom_settings or {})
        supported_settings = {"dialogue_cut_lines", "dialogue_history_seconds"}
        unknown_settings = set(settings) - supported_settings
        if unknown_settings:
            raise ValueError(
                "Fish Small does not support custom settings: "
                + ", ".join(sorted(str(key) for key in unknown_settings))
            )
        lines_per_part = _bounded_int_setting(
            settings.get("dialogue_cut_lines", 0), "dialogue_cut_lines", 0, 50
        )
        history_seconds = _bounded_float_setting(
            settings.get("dialogue_history_seconds", 60),
            "dialogue_history_seconds",
            0,
            120,
        )
        try:
            temperature = float(temperature)
            top_p = float(top_p)
            top_k_value = int(top_k)
            seed_value = int(seed)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("temperature, top-p, top-k, and seed must be valid numbers") from error
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if (
            isinstance(top_k, bool)
            or (isinstance(top_k, float) and not top_k.is_integer())
            or (isinstance(top_k, str) and str(top_k_value) != top_k.strip())
            or not 0 <= top_k_value <= 4097
        ):
            raise ValueError("top_k must be 0 (uncapped) or between 1 and 4097")
        if (
            isinstance(seed, bool)
            or (isinstance(seed, float) and not seed.is_integer())
            or (isinstance(seed, str) and str(seed_value) != seed.strip())
            or not -1 <= seed_value <= 4_294_967_295
        ):
            raise ValueError("seed must be -1 or an unsigned 32-bit integer")
        effective_top_k = 4097 if top_k_value == 0 else top_k_value

        chunks = drama3_text_chunks(
            script,
            lines_per_part=lines_per_part,
            max_speech_units=MAX_TEXT_UNITS_PER_PART,
        )
        if not chunks:
            raise ValueError("Speech Script: text must not be empty")
        whole_budget = duration_frame_budget(
            duration_seconds, script, self.sample_rate, self.frame_length
        )
        self.last_duration_budget_frames = whole_budget.frames
        self.last_duration_is_auto = whole_budget.automatic
        if whole_budget.seconds > MAX_JOB_SECONDS:
            raise ValueError(f"duration_seconds must be at most {MAX_JOB_SECONDS} seconds")

        try:
            guides, references = self._prepare_references(
                audio_prompt_type, (audio_guide, audio_guide2, audio_guide3), alt_prompt,
                set_progress_status,
            )
            # `audio_path` is the pinned native protocol field. Resolve every
            # voice guide to an owned WAV before starting the server.
            references_json = json.dumps(references, ensure_ascii=False, separators=(",", ":"))
            del guides
        except Exception:
            self._cleanup_reference_temp()
            raise

        try:
            report = self._new_report(script, chunks, whole_budget.frames, whole_budget.automatic)
            report["memory_plan"] = self.memory_plan.report()
            report["runtime"] = {
                "backend": str(getattr(self.runtime_info, "backend", "cuda")),
                "release": str(getattr(self.runtime_info, "release", "unknown")),
                "max_stream_seconds_per_request": MAX_STREAM_SECONDS,
            }
            report["completion_basis"] = (
                "Each completed source part reached native Drama 3 EOS (reached_limit=false). "
                "Word-by-word transcript coverage is not verified."
            )
            report["history_policy"] = "latest_complete_native_take_fitted_by_whole_turn_duration"
            report["history_max_seconds"] = history_seconds
            report["rng"] = {
                "strategy": "same_native_seed_per_part" if seed_value >= 0 else "native_random_seed_per_part",
                "seed": seed_value if seed_value >= 0 else None,
                "same_seed_guarantees_voice_match": False,
            }
            self.last_longform_report = report
            produced_frames = 0
            completed_end = 0
            active_part: dict[str, Any] | None = None
            previous_artifact: dict[str, Any] | None = None
            previous_artifact_frames = 0
            spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_MEMORY_BYTES, mode="w+b")
            with self._interrupt_lock:
                self._active_spool = spool
                self._active_pcm_samples = 0
                self._streamed_pcm_samples = 0
        except Exception:
            self._cleanup_reference_temp()
            raise

        try:
            self._status(set_progress_status, "Loading Fish Drama 3 Small native model")
            for part_index, (chunk, part_report) in enumerate(zip(chunks, report["chunks"]), 1):
                if self._check_cancelled():
                    raise NativeStreamInterrupted("Fish Small generation was stopped")
                remaining_frames = whole_budget.frames - produced_frames
                if remaining_frames <= 0:
                    part_report["status"] = "not_started"
                    part_report["cap_reasons"].append("whole_job_limit")
                    report["warnings"].append(
                        "The selected whole-output maximum was reached before all script parts were generated."
                    )
                    break

                if whole_budget.automatic:
                    part_budget = duration_frame_budget(
                        0, chunk.source_text, self.sample_rate, self.frame_length
                    ).frames
                else:
                    part_budget = MAX_PART_FRAMES
                part_budget = min(part_budget, MAX_PART_FRAMES, remaining_frames)
                part_report["budget_frames"] = part_budget
                if remaining_frames < MAX_PART_FRAMES:
                    part_report["cap_reasons"].append("whole_job_remaining_budget")
                if whole_budget.automatic and part_budget < MAX_PART_FRAMES:
                    part_report["cap_reasons"].append("auto_text_budget")

                part_report["status"] = "running"
                active_part = part_report
                with self._interrupt_lock:
                    self._active_pcm_samples = 0
                self._status(set_progress_status, f"Generating Fish Small speech part {part_index}/{len(chunks)}")

                def audio_delta(pcm: bytes, _rate: int, _channels: int, pcm_samples: int) -> None:
                    with self._interrupt_lock:
                        if self._check_cancelled():
                            raise NativeStreamInterrupted("Fish Small generation was stopped")
                        current_samples = self._active_pcm_samples
                        next_samples = current_samples + pcm_samples
                        sample_budget = part_budget * self.frame_length
                        if next_samples > sample_budget:
                            raise NativeStreamError(
                                "Fish Small streamed beyond the declared part limit "
                                f"({sample_budget} PCM samples)"
                            )
                        if next_samples > MAX_PART_SAMPLES:
                            raise NativeStreamError("Fish Small streamed more than 30 seconds for one part")
                        spool.write(pcm)
                        self._active_pcm_samples = next_samples
                        self._streamed_pcm_samples += pcm_samples
                        part_report["streamed_pcm_samples"] = next_samples
                        report["streamed_pcm_samples"] = self._streamed_pcm_samples
                        progress_frames = produced_frames + next_samples // self.frame_length
                    if callback is not None:
                        callback(
                            step_idx=max(0, progress_frames - 1),
                            override_num_inference_steps=whole_budget.frames,
                            denoising_extra=(
                                f"Generating Speech - Part {part_index}/{len(chunks)} - "
                                f"{progress_frames * self.frame_length / self.sample_rate:.1f}s"
                            ),
                            progress_unit="frames",
                        )

                request_options: dict[str, Any] = {
                    "max_tokens": int(part_budget),
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": effective_top_k,
                    # Maestro already split this source part. A second native
                    # split would apply max_tokens again to each sub-part.
                    "text_chunk_size": max(128, len(chunk.prompt_text.encode("utf-8")) + 1),
                    "text_chunk_mode": "tag_aware",
                    "multi_reference_cond": references_json,
                }
                if seed_value >= 0:
                    request_options["seed"] = seed_value
                request = {
                    "text": chunk.prompt_text,
                    "options": request_options,
                }
                history_frames_budget = int(history_seconds * self.sample_rate / self.frame_length)
                include_history = bool(
                    previous_artifact is not None
                    and previous_artifact_frames > 0
                    and previous_artifact_frames <= history_frames_budget
                )
                if include_history:
                    request["artifacts"] = [previous_artifact]
                part_report["history_context"] = {
                    "previous_take_included": include_history,
                    "previous_take_frames": previous_artifact_frames if include_history else 0,
                    "previous_take_seconds": round(
                        previous_artifact_frames * self.frame_length / self.sample_rate, 3
                    ) if include_history else 0.0,
                }
                try:
                    result: StreamResult = self.server.stream(
                        request,
                        audio_delta,
                        check_cancelled=self._check_cancelled,
                    )
                except NativeStreamInterrupted:
                    raise
                except NativeStreamError:
                    raise
                except Exception as error:
                    # The POST has an ambiguous outcome now. Surface the failure
                    # and never retry the generation request automatically.
                    raise NativeStreamError(f"Fish Small native request failed: {error}") from error

                part_samples = self._active_pcm_samples
                if part_samples != result.frames * self.frame_length:
                    raise NativeStreamError(
                        "Fish Small streamed PCM sample count does not match the native codec frame count "
                        f"({part_samples} samples, {result.frames} frames x {self.frame_length} samples)"
                    )
                part_frames = result.frames
                if result.take_text != chunk.prompt_text:
                    raise NativeStreamError(
                        "Fish Small completed a take with text that differs from the submitted source part"
                    )
                part_report["generated_frames"] = part_frames
                produced_frames += part_frames
                report["generated_frames"] = produced_frames
                part_report["finish_reason"] = "length" if result.reached_limit else "eos"
                part_report["eos_completed"] = not result.reached_limit
                self.last_finish_reason = part_report["finish_reason"]
                active_part = None
                if result.reached_limit:
                    part_report["status"] = "incomplete"
                    part_report["cap_reasons"].append("native_generation_limit")
                    report["warnings"].append(
                        f"Part {part_index} reached the native frame cap before EOS. "
                        "Later parts were not generated; the audio is partial."
                    )
                    break

                part_report["status"] = "complete"
                report["completed_chunks"] += 1
                completed_end = chunk.source_end
                report["covered_source_span"]["end"] = completed_end
                previous_artifact = {
                    key: result.artifact[key]
                    for key in ("id", "kind", "payload", "meta")
                    if key in result.artifact
                }
                previous_artifact_frames = result.frames

            for part in report["chunks"]:
                if part["status"] == "pending":
                    part["status"] = "not_started"
                    part["cap_reasons"].append("previous_part_incomplete")
            report["generated_frames"] = produced_frames
            report["covered_source_span"]["end"] = completed_end
            report["incomplete"] = report["completed_chunks"] != len(chunks)
            report["status"] = "incomplete" if report["incomplete"] else "complete"
            report["text_coverage_verified"] = False
            report["eos_verified_for_completed_parts"] = True
            self.last_longform_report = report
            if produced_frames <= 0:
                if self._interrupt or self._early_stop:
                    return None
                raise NativeStreamError("Fish Small completed without streamed speech audio")
            return self._waveform_result(spool, report)
        except NativeStreamInterrupted:
            self._release_server_once()
            report["status"] = "cancelled"
            report["incomplete"] = True
            report["cancelled"] = True
            report["warnings"].append(
                "Generation stopped during streaming. The result includes only PCM received before cancellation; "
                "the active part is incomplete and no later part was generated."
            )
            if active_part is not None:
                active_part["status"] = "cancelled"
                active_part["generated_frames"] = None
                active_part["streamed_pcm_samples"] = self._active_pcm_samples
                active_part["finish_reason"] = "cancelled"
            for part in report["chunks"]:
                if part["status"] == "pending":
                    part["status"] = "not_started"
                    part["cap_reasons"].append("cancelled")
            report["generated_frames"] = produced_frames
            report["streamed_pcm_samples"] = self._streamed_pcm_samples
            report["partial_streamed_pcm_samples"] = self._active_pcm_samples
            report["covered_source_span"]["end"] = completed_end
            report["text_coverage_verified"] = False
            report["eos_verified_for_completed_parts"] = True
            self.last_longform_report = report
            if self._interrupt or self._streamed_pcm_samples <= 0:
                return None
            return self._waveform_result(spool, report)
        except Exception as error:
            # A generation POST may have an ambiguous outcome, or the native
            # stream may have violated its protocol. Do not leave a child that
            # could continue consuming GPU memory after we report failure.
            self._release_server_once()
            report["status"] = "error"
            report["incomplete"] = True
            report["text_coverage_verified"] = False
            report["error"] = str(error)
            if active_part is not None:
                active_part["status"] = "error"
                active_part["generated_frames"] = None
                active_part["streamed_pcm_samples"] = self._active_pcm_samples
                active_part["finish_reason"] = "error"
            for part in report["chunks"]:
                if part["status"] == "pending":
                    part["status"] = "not_started"
                    part["cap_reasons"].append("previous_part_error")
            report["generated_frames"] = produced_frames
            report["covered_source_span"]["end"] = completed_end
            self.last_longform_report = report
            raise
        finally:
            with self._interrupt_lock:
                self._active_spool = None
                self._active_pcm_samples = 0
            spool.close()
            self._cleanup_reference_temp()
            gc.collect()

    def _prepare_references(
        self,
        audio_prompt_type: str,
        guides: tuple[str | None, str | None, str | None],
        alt_prompt: str,
        set_progress_status: Callable[[str], None] | None,
    ) -> tuple[list[str], list[dict[str, str]]]:
        selected = str(audio_prompt_type or "")
        if selected not in {"", "A", "AB", "ABD"}:
            raise ValueError("Fish Small supports voice reference selections '', A, AB, or ABD")
        guide_letters = [letter for letter in "ABD" if letter in selected]
        selected_guides = [guides[index] for index, letter in enumerate("ABD") if letter in selected]
        if len(guide_letters) != len(selected_guides):
            raise ValueError("Fish Small voice reference selection is inconsistent")
        if not selected_guides:
            if str(alt_prompt or "").strip():
                raise ValueError("Reference Transcripts were provided without a selected voice reference")
            return [], []

        transcript_values = reference_transcripts(str(alt_prompt or ""), len(selected_guides))
        references: list[dict[str, str]] = []
        owner = tempfile.TemporaryDirectory(prefix="maestro-fish-small-references-")
        self._reference_temp_owner = owner
        reference_dir = Path(owner.name)
        try:
            for index, (letter, guide, transcript) in enumerate(
                zip(guide_letters, selected_guides, transcript_values), 1
            ):
                if not guide:
                    raise ValueError(f"Voice reference {letter} is selected but its audio file is missing")
                audio_path = Path(str(guide)).expanduser().resolve()
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Voice reference {letter} was not found: {audio_path}")
                normalized_path = _normalize_voice_reference(
                    audio_path,
                    reference_dir / f"reference_{index}.wav",
                    letter=letter,
                )
                if not transcript.strip():
                    self._status(set_progress_status, f"Transcribing Fish Small voice reference {index}")
                    transcript = self._transcribe_reference(audio_path, index)
                tagged_transcript = f"<|speaker:{index - 1}|> {transcript.strip()}"
                validate_drama3_text(tagged_transcript, f"Reference Transcript {index}")
                references.append({"text": tagged_transcript, "audio_path": str(normalized_path)})
        except Exception:
            self._cleanup_reference_temp()
            raise
        finally:
            # Maestro's optional local Whisper service is shared by other
            # workflows. Release it after blank reference transcripts are made.
            if any(not text.strip() for text in transcript_values):
                try:
                    from services.audio_analysis import unload_whisper

                    unload_whisper()
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        return [str(path) for path in selected_guides], references

    @staticmethod
    def _transcribe_reference(path: Path, index: int) -> str:
        from services.audio_analysis import _transcribe

        segments = _transcribe(str(path))
        text = " ".join(str(getattr(segment, "text", "")) for segment in segments).strip()
        validate_drama3_text(text, f"Whisper transcript of voice reference {index} (type its transcript instead)")
        print(f"[Fish Small] Whisper transcript of voice reference {index}: {text}")
        return text

    @staticmethod
    def _new_report(script: str, chunks: list[Any], max_frames: int, automatic: bool) -> dict[str, Any]:
        return {
            "version": 1,
            "status": "running",
            "mode": "auto" if automatic else "maximum",
            "duration_limit_seconds": max_frames * FRAME_LENGTH / SAMPLE_RATE,
            "duration_budget_frames": max_frames,
            "generated_frames": 0,
            "total_chunks": len(chunks),
            "completed_chunks": 0,
            "incomplete": False,
            "covered_source_span": {"start": 0, "end": 0},
            "source_characters": len(script),
            "text_coverage_verified": False,
            "eos_verified_for_completed_parts": False,
            "completion_basis": "Pending native Drama 3 task completion metadata.",
            "warnings": [],
            "chunks": [
                {
                    "index": chunk.index,
                    "source_span": {"start": chunk.source_start, "end": chunk.source_end},
                    "source_units": chunk.speech_units,
                    "budget_frames": None,
                    "streamed_pcm_samples": 0,
                    "generated_frames": 0,
                    "finish_reason": None,
                    "cap_reasons": [],
                    "status": "pending",
                    "eos_completed": False,
                }
                for chunk in chunks
            ],
        }

    @staticmethod
    def _waveform_result(spool: Any, report: dict[str, Any]) -> dict[str, Any]:
        spool.flush()
        expected_samples = int(report.get("streamed_pcm_samples", 0))
        spool.seek(0, 2)
        payload_bytes = spool.tell()
        if payload_bytes != expected_samples * 2:
            raise NativeStreamError(
                "Fish Small PCM spool size does not match the streamed sample count "
                f"({payload_bytes} bytes, {expected_samples} samples)"
            )
        spool.seek(0)
        if expected_samples <= 0:
            raise NativeStreamError("Fish Small returned no finite waveform samples")
        samples = np.empty(expected_samples, dtype=np.float32)
        offset = 0
        while offset < expected_samples:
            block = spool.read(min(131_072, expected_samples - offset) * 2)
            if not block or len(block) % 2:
                raise NativeStreamError("Fish Small stream ended on a partial PCM16 sample")
            count = len(block) // 2
            end = offset + count
            if end > expected_samples:
                raise NativeStreamError("Fish Small PCM spool contains more samples than reported")
            samples[offset:end] = np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768.0
            offset = end
        if not np.isfinite(samples).all():
            raise NativeStreamError("Fish Small returned non-finite waveform samples")
        audio = torch.from_numpy(samples).reshape(1, -1)
        return {
            "x": audio,
            "audio_sampling_rate": SAMPLE_RATE,
            "speech_generation": report,
            "generation_warnings": list(report["warnings"]),
        }

    def unload_all(self) -> None:
        self.server.unload_all_models()

    def release(self) -> None:
        with self._interrupt_lock:
            self._released = True
            server = self.server
        try:
            server.release()
        finally:
            self._cleanup_reference_temp()

    def _cleanup_reference_temp(self) -> None:
        with self._interrupt_lock:
            owner, self._reference_temp_owner = self._reference_temp_owner, None
        if owner is not None:
            owner.cleanup()


def _normalize_voice_reference(source: Path, destination: Path, *, letter: str) -> Path:
    """Decode a short reference on CPU and write native-compatible mono WAV."""
    try:
        info = sf.info(str(source))
    except Exception as error:
        raise ValueError(
            f"Voice reference {letter} could not be read by soundfile; use a supported WAV, FLAC, or MP3 file: {error}"
        ) from error
    if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
        raise ValueError(f"Voice reference {letter} has invalid audio dimensions")
    duration = info.frames / info.samplerate
    if not math.isfinite(duration) or duration < MIN_REFERENCE_SECONDS:
        raise ValueError(
            f"Voice reference {letter} must have a finite duration of at least "
            f"{MIN_REFERENCE_SECONDS:g} second"
        )
    if duration > MAX_REFERENCE_SECONDS:
        raise ValueError(
            f"Voice reference {letter} is {duration:.1f} seconds; trim it to "
            f"{MAX_REFERENCE_SECONDS:g} seconds or less before using it. Fish Small does not trim references silently."
        )
    if info.frames * info.channels > MAX_REFERENCE_DECODE_SAMPLES:
        raise ValueError(
            f"Voice reference {letter} has too many decoded samples; convert it "
            "to mono or stereo at a lower sample rate before using it"
        )
    try:
        samples, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    except Exception as error:
        raise ValueError(f"Voice reference {letter} could not be decoded: {error}") from error
    if sample_rate != info.samplerate or samples.shape != (info.frames, info.channels):
        raise ValueError(f"Voice reference {letter} changed while it was being decoded")
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Voice reference {letter} contains no finite audio samples")
    mono = samples.mean(axis=1, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        mono = resample_poly(
            mono,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32, copy=False)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"Voice reference {letter} could not be normalized safely")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(
            str(destination),
            mono,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )
    except Exception as error:
        raise ValueError(f"Voice reference {letter} could not be normalized to PCM16 WAV: {error}") from error
    return destination


def _bounded_int_setting(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}") from error
    if (
        isinstance(value, bool)
        or (isinstance(value, float) and not value.is_integer())
        or (isinstance(value, str) and str(parsed) != value.strip())
        or not minimum <= parsed <= maximum
    ):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return parsed


def _bounded_float_setting(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and between {minimum:g} and {maximum:g}") from error
    if isinstance(value, bool) or not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum:g} and {maximum:g}")
    return parsed
