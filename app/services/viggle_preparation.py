"""Prepare one character replacement image before Viggle's fixed video recipe.

This module performs CPU media work. The application supplies a serialized
Klein renderer, so preparation and animation share the normal GPU/job lifecycle.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

SWAP_PROMPT = (
    "Replace the main character in source frame with character in second image. "
    "Preserve the exact pose, body orientation, hands, props, background, camera "
    "framing, lighting and image dimensions."
)
MODELS = {"flux2_klein_9b", "flux2_klein_4b"}
VERSION = 1


def normalize_options(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Choose a character image for Viggle preparation.")
    model = str(value.get("image_model") or "flux2_klein_9b")
    if model not in MODELS:
        raise ValueError("Viggle character preparation requires Flux 2 Klein 9B or 4B.")
    reference = str(value.get("reference_path") or "").strip()
    if not reference:
        raise ValueError("Choose a saved character or upload a character image.")
    seconds = float(value.get("frame_seconds") or 0)
    if not math.isfinite(seconds) or not 0 <= seconds <= 3600:
        raise ValueError("Choose a source frame between 0 seconds and one hour.")
    prompt = str(value.get("swap_prompt") or SWAP_PROMPT).strip()
    appearance = str(value.get("appearance_prompt") or "").strip()
    if not prompt or len(prompt) > 8000 or len(appearance) > 8000:
        raise ValueError("Replacement and appearance prompts must be at most 8,000 characters each.")
    return {"reference_path": reference, "image_model": model, "frame_seconds": seconds,
        "swap_prompt": prompt, "appearance_prompt": appearance,
        **{key: str(value[key])[:120] for key in ("character_id", "character_name", "view_id") if value.get(key)}}


def replacement_prompt(options: dict) -> str:
    text = options["swap_prompt"]
    if options["appearance_prompt"]:
        text += "\n\nAppearance of the replacement character: " + options["appearance_prompt"]
    return text


def signature(source: Path, reference: Path, options: dict, seed) -> str:
    def identity(path):
        stat = path.stat()
        return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    data = {"version": VERSION, "source": identity(source), "reference": identity(reference),
        "options": options, "seed": seed}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def extract_frame(source: Path, seconds: float, destination: Path) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    import cv2
    import numpy as np
    from PIL import Image

    capture = cv2.VideoCapture(str(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count < 1 or seconds >= count / fps:
            raise ValueError("The chosen frame is outside the control video. Choose an earlier frame.")
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Could not read the chosen control-video frame.")
    finally:
        capture.release()
    height, width = frame.shape[:2]
    if min(width, height) < 32 or width * height > 8_500_000:
        raise ValueError("Use a control video between 32 pixels per edge and 8.5 megapixels for character preparation.")
    pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(pixels).save(destination / "source.png")
    # Padding satisfies Klein's grid without stretching the source pose.
    padded_width, padded_height = math.ceil(width / 32) * 32, math.ceil(height / 32) * 32
    pixels = np.pad(pixels, ((0, padded_height-height), (0, padded_width-width), (0, 0)), mode="edge")
    path = destination / "source-padded.png"
    Image.fromarray(pixels).save(path)
    return path, (width, height), (padded_width, padded_height)


def image_request(options: dict, source: Path, reference: Path, canvas: tuple[int, int], seed) -> dict:
    return {"model_type": options["image_model"], "generation_mode": "image", "image_mode": 1,
        "prompt": replacement_prompt(options), "image_refs": [str(source), str(reference)],
        "video_prompt_type": "KI", "image_prompt_type": "", "resolution": f"{canvas[0]}x{canvas[1]}",
        "video_length": 1, "num_inference_steps": 4, "guidance_scale": 1, "seed": seed,
        "multi_prompts_gen_type": 2, "repeat_generation": 1, "batch_size": 1,
        "remove_background_images_ref": 0, "image_refs_relative_size": 100,
        "activated_loras": [], "loras_multipliers": "", "film_grain_intensity": 0,
        "spatial_upsampling": "", "temporal_upsampling": "", "custom_settings": {},
        "image_output_codec": "png", "_studio_image_workflow": "generate", "_defer_output_publication": True}


def prepare(body: dict, *, resolve_media, render, update, aborted) -> dict:
    """Render or reuse a matching frame; publish only a completed PNG."""
    from PIL import Image

    options = normalize_options(body.get("viggle_character"))
    source = Path(resolve_media(body.get("video_guide"))).resolve()
    reference = Path(resolve_media(options["reference_path"])).resolve()
    with Image.open(reference) as picture:
        if picture.width * picture.height > 32_000_000:
            raise ValueError("The character reference exceeds 32 megapixels.")
        picture.verify()
    fingerprint = signature(source, reference, options, body.get("seed", -1))
    cached = body.get("_viggle_prepared") or {}
    uploads = (Path.cwd() / "uploads").resolve()
    uploads.mkdir(exist_ok=True)
    if isinstance(cached, dict) and cached.get("signature") == fingerprint:
        path = Path(str(cached.get("image_path") or "")).resolve()
        if path.parent == uploads and path.name.startswith("viggle_prepared_") and path.is_file():
            try:
                with Image.open(path) as picture:
                    picture.verify()
                update("Using the prepared character frame…")
                return dict(cached)
            except (OSError, ValueError):
                pass  # An incomplete/deleted preview must not prevent a fresh render.
    if aborted():
        raise InterruptedError("Character preparation cancelled.")
    root = uploads / "viggle_preparation"
    root.mkdir(exist_ok=True)
    directory = root / uuid.uuid4().hex
    directory.mkdir()
    try:
        update("Extracting the source frame…")
        frame, source_size, canvas = extract_frame(source, options["frame_seconds"], directory)
        if aborted():
            raise InterruptedError("Character preparation cancelled.")
        update("Replacing the character with Flux 2 Klein…")
        generated = Path(render(image_request(options, frame, reference, canvas, body.get("seed", -1)), directory))
        if aborted():
            raise InterruptedError("Character preparation cancelled.")
        update("Saving the prepared frame…")
        output = uploads / f"viggle_prepared_{uuid.uuid4().hex}.png"
        with Image.open(generated) as picture:
            if picture.size != canvas:
                raise ValueError(f"Klein returned {picture.width}×{picture.height}; expected {canvas[0]}×{canvas[1]}. The frame was not passed to Viggle.")
            picture.crop((0, 0, *source_size)).save(output)
        return {"signature": fingerprint, "image_path": str(output),
            "image_url": f"/api/v1/uploads/{output.name}", "width": source_size[0], "height": source_size[1],
            "frame_seconds": options["frame_seconds"], "image_model": options["image_model"],
            "prompt": replacement_prompt(options)}
    finally:
        # Only this operation's newly-created UUID directory is disposable.
        if directory.resolve().parent == root.resolve():
            shutil.rmtree(directory, ignore_errors=True)
