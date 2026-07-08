"""Minimal Whisper-medium loader for Scenema audio.

Extracted from upstream Wan2GP's `shared/deepy/transcription.py`. Only the
constants + `_load_whisper_medium` are needed by Scenema's pipeline; the rest
of the module (transcribe_media, segment serialization, ffmpeg integration)
is intentionally omitted. Strip the upstream dev-fallback path
(`e:/ml/wan2gp/ckpts`) — Maestro's `files_locator` is the canonical resolver.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import whisper
from safetensors.torch import load_file as load_safetensors_file

from shared.utils import files_locator as fl


WHISPER_MEDIUM_FOLDER = "whisper_medium"
WHISPER_MEDIUM_REPO = "DeepBeepMeep/Wan2.1"
WHISPER_MEDIUM_CONFIG_FILENAME = "config.json"
WHISPER_MEDIUM_WEIGHTS_FILENAME = "model.safetensors"
_WHISPER_MEDIUM_REQUIRED_FILES = (WHISPER_MEDIUM_CONFIG_FILENAME, WHISPER_MEDIUM_WEIGHTS_FILENAME)


def _get_main_callable(name: str) -> Any:
    main_module = sys.modules.get("__main__")
    return None if main_module is None else getattr(main_module, str(name or "").strip(), None)


def _whisper_medium_files_present(model_dir: Path | None) -> bool:
    if model_dir is None or not model_dir.is_dir():
        return False
    return all((model_dir / filename).is_file() for filename in _WHISPER_MEDIUM_REQUIRED_FILES)


def _ensure_whisper_medium_assets(model_dir: Path | None = None) -> None:
    if _whisper_medium_files_present(model_dir):
        return
    process_files_def = _get_main_callable("process_files_def")
    if callable(process_files_def):
        process_files_def(
            repoId=WHISPER_MEDIUM_REPO,
            sourceFolderList=[WHISPER_MEDIUM_FOLDER],
            fileList=[list(_WHISPER_MEDIUM_REQUIRED_FILES)],
        )


def _whisper_medium_dir() -> Path:
    located = fl.locate_folder(WHISPER_MEDIUM_FOLDER, error_if_none=False)
    located_path = None if located is None else Path(located).resolve()
    _ensure_whisper_medium_assets(located_path)
    located = fl.locate_folder(WHISPER_MEDIUM_FOLDER, error_if_none=False)
    if located is not None:
        resolved = Path(located).resolve()
        if _whisper_medium_files_present(resolved):
            return resolved
    raise FileNotFoundError(
        f"Unable to locate the Whisper medium folder '{WHISPER_MEDIUM_FOLDER}' in the configured checkpoints paths."
    )


def _load_whisper_medium(device: torch.device) -> whisper.Whisper:
    model_dir = _whisper_medium_dir()
    config_path = model_dir / WHISPER_MEDIUM_CONFIG_FILENAME
    weights_path = model_dir / WHISPER_MEDIUM_WEIGHTS_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Whisper config file not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Whisper weights file not found: {weights_path}")
    with config_path.open("r", encoding="utf-8") as reader:
        config = json.load(reader)
    dims = whisper.model.ModelDimensions(**dict(config.get("dims", {}) or {}))
    model = whisper.model.Whisper(dims)
    model.load_state_dict(load_safetensors_file(str(weights_path), device="cpu"))
    alignment_heads = str(config.get("alignment_heads", "") or "").strip()
    if len(alignment_heads) > 0:
        model.set_alignment_heads(alignment_heads.encode("ascii"))
    model.eval()
    if device.type == "cuda":
        return model.to(device=device)
    return model.to(device=device, dtype=torch.float32)
