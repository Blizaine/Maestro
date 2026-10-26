"""Family registration and isolated runtime adapter for Fish Small."""

from __future__ import annotations

import os
import hashlib
import threading
import time
from pathlib import Path
from typing import Any

import torch
from shared.utils import files_locator as fl


ARCHITECTURE = "fish_drama3_small"
MODEL_REPOSITORY = "pentacoxian-dev/drama3-small-GGUF"
MODEL_REVISION = "4a3078e885344bb9b24d511cec389d2726854d15"
MODEL_FILENAME = "drama3-small-IQ3_XXS.gguf"
MODEL_SIZE = 10_460_748_576
MODEL_SHA256 = "97d773973d756141a65432b08de76fd4bedbb44e8b98805755130e02d7fe2561"
_VERIFIED_CHECKPOINTS: set[tuple[str, int, int, int]] = set()
_CHECKPOINT_LOCK = threading.RLock()


class FishSmallOffloadAdapter:
    """Keep the external-runtime lifecycle visible to Maestro's model manager."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def unload_all(self) -> None:
        self.pipeline.unload_all()

    def release(self) -> None:
        self.pipeline.release()


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        return {"tts": (2200, "TTS")}

    @staticmethod
    def get_lora_dir(base_model_type, args=None, lora_root="loras"):
        return os.path.join(lora_root, ARCHITECTURE)

    @staticmethod
    def query_model_def(base_model_type, model_def):
        from ..fish_audio.fish_audio_handler import (
            DEEPY_INFOS,
            DEEPY_PROMPT_INFOS,
            DIALOGUE_ENHANCER,
            ENHANCER,
            INFOS,
            PROMPT_INFOS,
            DIALOGUE_CUT_LINES,
            DIALOGUE_HISTORY_SECONDS,
        )

        return {
            "group": "tts",
            "audio_only": True,
            "loras_disabled": True,
            "image_outputs": False,
            "sliding_window": False,
            "device_explicit": True,
            "dtype": "bf16",
            "auto_quantize": False,
            "no_negative_prompt": True,
            "inference_steps": False,
            "guidance_max_phases": 0,
            "temperature": True,
            "top_k_slider": True,
            "top_p_slider": True,
            "compile": False,
            "cuda_graphs": False,
            "supports_early_stop": True,
            "speech_auto_duration": True,
            "duration_is_maximum": True,
            "speech_longform": True,
            "audio_segment_max_seconds": 30,
            "profiles_dir": [ARCHITECTURE],
            "max_voice_count": 3,
            "audio_mode_from_voice_count": True,
            "image_prompt_types_allowed": "",
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "audio_prompt_type_sources": {
                "selection": ["", "A", "AB", "ABD"],
                "labels": {
                    "": "New Voice",
                    "A": "Voice Reference",
                    "AB": "Voice References for Speakers 1 and 2",
                    "ABD": "Voice References for Speakers 1, 2 and 3",
                },
                "letters_filter": "ABD",
                "default": "",
            },
            "audio_guide_label": "Speaker 1 Voice Reference (1–30s)",
            "audio_guide2_label": "Speaker 2 Voice Reference (1–30s)",
            "audio_guide3_label": "Speaker 3 Voice Reference (1–30s)",
            "alt_prompt": {
                "label": "Reference Transcripts",
                "name": "Reference Transcripts",
                "lines": 3,
                "audio_prompt_type": "A",
                "placeholder": (
                    "Exact words of each voice reference, one 'Speaker N: ...' "
                    "line per reference. Leave empty to transcribe them with Whisper."
                ),
            },
            "duration_slider": {
                "label": "Speech Duration (0 = Auto; otherwise maximum seconds)",
                "name": "Speech Duration",
                "min": 0,
                "max": 3600,
                "increment": 1,
                "default": 0,
            },
            "custom_settings": [
                dict(DIALOGUE_CUT_LINES),
                dict(DIALOGUE_HISTORY_SECONDS),
            ],
            "prompt_class": "Speech",
            "preserve_empty_prompt_lines": True,
            "prompt_enhancer_def": {
                "selection": ["T", "T1"],
                "labels": {
                    "T": "A Speech based on current Prompt",
                    "T1": "A Dialogue based on current Prompt",
                },
                "default": "",
            },
            "prompt_enhancer_button_label": "Write",
            "text_prompt_enhancer_instructions": ENHANCER,
            "text_prompt_enhancer_max_tokens": 2048,
            "text_prompt_enhancer_instructions1": DIALOGUE_ENHANCER,
            "text_prompt_enhancer_max_tokens1": 2048,
            "infos": INFOS,
            "prompt_infos": PROMPT_INFOS,
            "deepy_infos": DEEPY_INFOS,
            "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
            "external_runtime": "fish_small",
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        # The main GGUF is managed through the model's immutable URL in
        # defaults. Place the companion license files in a family-local folder
        # so a root license from another downloaded model cannot shadow them.
        return [
            {
                "repoId": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "sourceFolderList": [""],
                "targetFolderList": ["fish_audio_small"],
                "fileList": [["LICENSE", "NOTICE"]],
            }
        ]

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        from ..fish_audio.prompt import (
            check_paired_tags,
            reference_transcripts,
            speaker_lines_to_tags,
            validate_drama3_text,
        )

        try:
            if inputs.get("activated_loras"):
                raise ValueError("Fish Small's native runtime does not support LoRAs. Clear the selected adapters first.")
            validate_drama3_text(speaker_lines_to_tags(one_prompt), "Speech Script")
            check_paired_tags(one_prompt, "Speech Script")
            references = [letter for letter in "ABD" if letter in inputs.get("audio_prompt_type", "")]
            for index, transcript in enumerate(
                reference_transcripts(inputs.get("alt_prompt", ""), len(references)), 1
            ):
                if transcript:
                    validate_drama3_text(transcript, f"Reference Transcript {index}")
        except ValueError as error:
            return str(error)
        return None

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            prompt="Speaker 1: Hello there.",
            alt_prompt="",
            audio_prompt_type="",
            duration_seconds=0,
            temperature=0.7,
            top_k=0,
            top_p=0.7,
            prompt_enhancer="",
            guidance_scale=1.0,
            num_inference_steps=1,
            video_length=0,
            repeat_generation=1,
            multi_prompts_gen_type="FG",
        )

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("audio_prompt_type", "")
        ui_defaults.setdefault("alt_prompt", "")

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
        profile=0,
        **kwargs,
    ):
        del model_type, base_model_type, profile
        model_path = _resolve_model_path(model_filename)
        if not model_path.is_file():
            raise FileNotFoundError(
                "Fish Drama 3 Small checkpoint is missing. Run Install Models and retry: "
                f"{model_path}"
            )
        def report_load_progress(stage, downloaded_bytes, total_download_bytes):
            callback = kwargs.get("set_progress_status")
            if not callable(callback):
                return
            now = time.monotonic()
            if total_download_bytes:
                percent = int(downloaded_bytes * 100 / total_download_bytes)
                key = (str(stage), percent)
                message = f"{stage} ({percent}%)"
            else:
                key = (str(stage), None)
                message = str(stage)
            previous = getattr(report_load_progress, "_last", None)
            if previous is not None and key == previous[0] and now - previous[1] < 1.0:
                return
            report_load_progress._last = (key, now)
            callback(message)

        _verify_checkpoint(
            model_path,
            progress=report_load_progress,
            check_cancelled=kwargs.get("check_cancelled"),
        )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Fish Drama 3 Small requires its CUDA audio.cpp runtime. "
                "No CPU backend is selected implicitly."
            )

        device_index = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(device_index))
        overrides = model_def.get("fish_small_session_options")
        from .small_runtime import choose_memory_plan

        memory_plan = choose_memory_plan(total_bytes, free_bytes, overrides=overrides)
        print(
            "[Fish Small] Admission plan: "
            f"{memory_plan.name}; {memory_plan.reason}"
        )

        def report_runtime_progress(stage, downloaded_bytes, total_download_bytes):
            report_load_progress(
                f"Preparing Fish Small native runtime: {stage}",
                downloaded_bytes,
                total_download_bytes,
            )

        from services.fish_small_runtime import ensure_runtime

        runtime_info = ensure_runtime(
            backend="cuda",
            device_capability=capability,
            check_cancelled=kwargs.get("check_cancelled"),
            progress=report_runtime_progress,
        )
        from .pipeline import FishSmallPipeline

        pipeline = FishSmallPipeline(model_path, runtime_info, memory_plan)
        return pipeline, {
            "external_runtime": True,
            "offload_adapter": FishSmallOffloadAdapter,
        }


def _resolve_model_path(model_filename: Any) -> Path:
    values = list(model_filename) if isinstance(model_filename, (list, tuple)) else [model_filename]
    for value in values:
        if not value:
            continue
        candidate = Path(str(value)).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        located = fl.locate_file(str(Path("fish_audio_small") / candidate.name), error_if_none=False)
        if located:
            return Path(located).resolve()
        located = fl.locate_file(candidate.name, error_if_none=False)
        if located:
            return Path(located).resolve()
    return Path(fl.get_download_location()) / "fish_audio_small" / MODEL_FILENAME


def _verify_checkpoint(
    path: Path,
    *,
    progress: Any = None,
    check_cancelled: Any = None,
) -> None:
    stat_before = path.stat()
    cache_key = (
        str(path.resolve()),
        stat_before.st_size,
        stat_before.st_mtime_ns,
        getattr(stat_before, "st_ino", 0),
    )
    if stat_before.st_size != MODEL_SIZE:
        raise ValueError(
            f"Fish Small checkpoint size mismatch: expected {MODEL_SIZE:,} bytes, "
            f"found {stat_before.st_size:,} at {path}"
        )
    with _CHECKPOINT_LOCK:
        if cache_key in _VERIFIED_CHECKPOINTS:
            return
        digest = hashlib.sha256()
        read_bytes = 0
        if callable(progress):
            progress("Verifying Fish Small checkpoint", 0, stat_before.st_size)
        with path.open("rb") as handle:
            while True:
                if callable(check_cancelled) and check_cancelled():
                    raise RuntimeError("Fish Small checkpoint verification was cancelled")
                block = handle.read(4 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                read_bytes += len(block)
                if callable(progress):
                    progress("Verifying Fish Small checkpoint", read_bytes, stat_before.st_size)
        stat_after = path.stat()
        if (
            stat_after.st_size != stat_before.st_size
            or stat_after.st_mtime_ns != stat_before.st_mtime_ns
            or getattr(stat_after, "st_ino", 0) != getattr(stat_before, "st_ino", 0)
        ):
            raise RuntimeError("Fish Small checkpoint changed during integrity verification")
        actual = digest.hexdigest()
        if actual != MODEL_SHA256:
            raise ValueError(
                "Fish Small checkpoint SHA-256 mismatch; the local file does not match "
                f"the pinned model revision ({actual})"
            )
        _VERIFIED_CHECKPOINTS.add(cache_key)
