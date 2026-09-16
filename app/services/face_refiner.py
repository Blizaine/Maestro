"""Queued face refinement for new generations and existing Maestro videos."""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import uuid

from .media_flow import _check_abort, _encoding_options

DEFAULTS = {"enabled": False, "face_count": 0, "strength": 0.75,
            "steps": 4, "window_frames": 243, "model": "auto", "character_ids": []}


def normalize_options(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError("Face Refiner settings must be an object")
    result = {**DEFAULTS, **(value or {})}
    for field, low, high in (("face_count", 0, 5), ("steps", 4, 8), ("window_frames", 22, 345)):
        number = result[field]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or int(number) != number or not low <= number <= high:
            raise ValueError(f"Face Refiner {field.replace('_', ' ')} must be a whole number from {low} to {high}")
        result[field] = int(number)
    if (result["window_frames"] - 5) % 17:
        raise ValueError("Face Refiner window size must use 5 + 17n frames")
    strength = result["strength"]
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("Face refinement strength must be greater than 0 and at most 1")
    if result["model"] not in ("auto", "pruned", "fused"):
        raise ValueError("Choose Auto, H3 Pruned or H3 Fused for face refinement")
    if not isinstance(result["enabled"], bool):
        raise ValueError("Face Refiner enabled must be true or false")
    ids = result["character_ids"]
    if not isinstance(ids, list) or len(ids) > 5 or any(not isinstance(i, str) or not re.fullmatch(r"[a-f0-9]{16}", i) for i in ids):
        raise ValueError("Choose up to five saved characters for face matching")
    result["character_ids"] = list(dict.fromkeys(ids))
    return {key: result[key] for key in DEFAULTS}


def normalize_assignments(value=None):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 5:
        raise ValueError("Map at most five detected faces")
    result, seen = [], set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each face mapping must be an object")
        track = item.get("track_id")
        character = item.get("character_id") or None
        skip = item.get("skip", False)
        if isinstance(track, bool) or not isinstance(track, int) or not 1 <= track <= 5 or track in seen:
            raise ValueError("Each detected face can be mapped only once")
        if character is not None and (not isinstance(character, str) or not re.fullmatch(r"[a-f0-9]{16}", character)):
            raise ValueError("Invalid saved character in face mapping")
        if not isinstance(skip, bool):
            raise ValueError("Face mapping skip must be true or false")
        seen.add(track)
        result.append({"track_id": track, "character_id": character, "skip": skip})
    return result


def _root():
    path = Path("uploads/face_refiner").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def analysis_directory(analysis_id):
    if not isinstance(analysis_id, str) or not re.fullmatch(r"[a-f0-9]{32}", analysis_id):
        raise ValueError("Invalid face analysis ID")
    return _root() / analysis_id


def fingerprint(source):
    path = Path(source).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_analysis(analysis_id, source=None, face_count=None):
    path = analysis_directory(analysis_id) / "analysis.json"
    if not path.is_file():
        raise ValueError("Face analysis is not available yet. Detect faces again.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if source is not None and data["source"] != fingerprint(source):
        raise ValueError("The source video changed after face detection. Detect faces again.")
    if face_count is not None and data["face_count"] != face_count:
        raise ValueError("The face count changed. Detect faces again before applying mappings.")
    return data


def public_analysis(data):
    return {key: data[key] for key in ("id", "faces", "fps", "frames", "width", "height", "warnings")}


def source_characters(source):
    try:
        meta = json.loads(Path(source).with_suffix(".meta.json").read_text(encoding="utf-8"))
        refs = meta.get("params", {}).get("minimax_h3_references", [])
        return list(dict.fromkeys(ref["library_character_id"] for ref in refs if isinstance(ref, dict)
                                 and re.fullmatch(r"[a-f0-9]{16}", str(ref.get("library_character_id", "")))))[:5]
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def character_image(character_id):
    import cv2
    from PIL import Image
    from .character_library import get_character
    character = get_character(character_id)
    visual = character["visual"]
    path = visual["path"]
    if visual["type"] == "image":
        with Image.open(path) as image:
            return image.convert("RGB")
    capture = cv2.VideoCapture(path)
    try:
        # A RefMod's decoded visual preview works here just like a saved
        # character video. The original recorded soundtrack stays untouched.
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) / 3)))
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot read the visual reference for {character['name']}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


@contextlib.contextmanager
def decoded_video(source, abort=None, progress=None):
    """Disk-backed frames avoid retaining a full-resolution clip in RAM."""
    import cv2
    import numpy as np
    import torch
    with tempfile.TemporaryDirectory(prefix="frames-", dir=_root()) as temporary:
        path = Path(temporary) / "source.rgb"
        capture = cv2.VideoCapture(str(source))
        array = tensor = None
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError("Cannot read the source video's frame rate")
            count = 0
            with path.open("wb") as handle:
                while True:
                    _check_abort(abort)
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if count == 0:
                        height, width = frame.shape[:2]
                    if frame.shape[:2] != (height, width):
                        raise ValueError("The source video changes resolution between frames")
                    handle.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
                    count += 1
                    if progress and count % 24 == 0:
                        progress("Reading source video", count, expected)
            if not count:
                raise ValueError("The source video contains no readable frames")
            array = np.memmap(path, dtype=np.uint8, mode="r+", shape=(count, height, width, 3))
            tensor = torch.from_numpy(array)
            yield tensor, {"fps": fps, "frames": count, "width": width, "height": height}, Path(temporary)
        finally:
            capture.release()
            tensor = None
            if array is not None:
                array._mmap.close()


def analyze(source, options=None, *, analysis_id=None, abort=None, progress=None):
    from PIL import Image
    from postprocessing.h3_face_refiner.face import _detect, _select_tracks, _crop_track, crop_face_track, select_reference_frame
    from postprocessing.h3_face_refiner.runtime import ensure_tracking_assets
    options = normalize_options(options)
    analysis_id = analysis_id or uuid.uuid4().hex
    directory = analysis_directory(analysis_id)
    directory.mkdir(exist_ok=True)
    detector, identity_dir = ensure_tracking_assets(progress)
    references = [character_image(cid) for cid in options["character_ids"]]
    ref_ids = {id(image): cid for image, cid in zip(references, options["character_ids"])}
    original_fingerprint = fingerprint(source)
    with decoded_video(source, abort, progress) as (frames, geometry, _):
        detections = _detect(frames, detector, 0.25, abort, progress)
        _check_abort(abort)
        if detections is None:
            raise RuntimeError("Face detection did not complete")
        tracks = _select_tracks(frames, detections, face_count=options["face_count"],
            reference_images=references, identity_threshold=0.28, auto_min_face_height=32,
            reference_threshold=0.5, reference_margin=0.08,
            auto_min_presence=0.2, insightface_model_dir=identity_dir,
            abort_callback=abort, progress_callback=progress)
        _check_abort(abort)
        if tracks is None:
            raise RuntimeError("Face tracking did not complete")
        faces, transforms, warnings = [], [], []
        for index, track in enumerate(tracks, 1):
            _check_abort(abort)
            prepared = _crop_track(frames, track["boxes"], crop_factor=1.6,
                canvas_width=512, canvas_height=512, canvas_mode="auto_capped_768",
                strength_small_face=options["strength"], strength_large_face=options["strength"],
                abort_callback=abort, progress_callback=progress,
                track_label=f"Face {index}", include_crops=False)
            _check_abort(abort)
            if prepared is None:
                raise RuntimeError("Face crop preparation did not complete")
            transform, _strengths = prepared
            reference_frame = select_reference_frame(frames, transform["raw_face_boxes"],
                max_candidates=24, insightface_model_dir=identity_dir)
            crop = crop_face_track(frames, transform, reference_frame, reference_frame + 1, uint8_storage=True)
            image = Image.fromarray(crop[:, 0].permute(1, 2, 0).numpy())
            image.save(directory / f"face-{index}.png")
            matched = ref_ids.get(id(track.get("reference_image")))
            faces.append({"track_id": index, "thumbnail_url": f"/api/v1/face-refiner/analyses/{analysis_id}/faces/{index}",
                "character_id": matched, "similarity": round(float(track.get("reference_similarity", 0)), 3) if matched else None,
                "first_seen_seconds": round(next(i for i, box in enumerate(track["boxes"]) if box is not None) / geometry["fps"], 2),
                "presence": round(float(track["presence"]), 3)})
            transforms.append(transform)
            if track.get("anchor") is None:
                warnings.append(f"Face {index} was tracked by position; a reliable identity embedding was unavailable.")
    _check_abort(abort)
    if fingerprint(source) != original_fingerprint:
        raise ValueError("The source changed while detecting faces. Try again.")
    data = {"id": analysis_id, "source": original_fingerprint, "face_count": options["face_count"],
            "faces": faces, "transforms": transforms, "warnings": warnings, **geometry}
    (directory / "analysis.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def resolve_mappings(data, assignments=None):
    assignments = normalize_assignments(assignments)
    valid = {face["track_id"] for face in data["faces"]}
    if any(item["track_id"] not in valid for item in assignments):
        raise ValueError("A mapped face is not present in this analysis. Detect faces again.")
    mapped = {item["track_id"]: item for item in assignments}
    return [mapped.get(face["track_id"], {"track_id": face["track_id"],
            "character_id": face.get("character_id"), "skip": False}) for face in data["faces"]]


def audio_window(source, start, count, fps):
    import numpy as np
    from shared.utils.video_decode import _resolve_media_binary
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    probe = subprocess.run([_resolve_media_binary("ffprobe"), "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", "-of", "json", str(source)], capture_output=True, timeout=30, creationflags=flags)
    if probe.returncode:
        raise ValueError("Cannot inspect source audio for face refinement")
    samples = round(count / fps * 32000)
    if not json.loads(probe.stdout).get("streams"):
        return np.zeros((samples, 2), dtype=np.float32), 32000
    decoded = subprocess.run([_resolve_media_binary("ffmpeg"), "-v", "error", "-nostdin",
        "-ss", str(start / fps), "-i", str(source), "-t", str(count / fps), "-map", "0:a:0",
        "-vn", "-ac", "2", "-ar", "32000", "-f", "f32le", "pipe:1"],
        capture_output=True, timeout=60, creationflags=flags)
    if decoded.returncode:
        raise ValueError("Cannot decode source audio for face refinement")
    audio = np.frombuffer(decoded.stdout, dtype=np.float32).reshape(-1, 2).copy()
    if len(audio) < samples:
        audio = np.pad(audio, ((0, samples - len(audio)), (0, 0)))
    return audio[:samples], 32000


def _encode(source, destination, frames, fps, abort=None, progress=None):
    import numpy as np
    from shared.utils.video_decode import _resolve_media_binary
    ffmpeg = _resolve_media_binary("ffmpeg")
    ffprobe = _resolve_media_binary("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe are required for face refinement")
    encoding = _encoding_options(destination)
    probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=codec_name", "-of", "json", str(source)], capture_output=True, timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if probe.returncode:
        raise ValueError("Cannot inspect the original soundtrack")
    codecs = [item.get("codec_name") for item in json.loads(probe.stdout).get("streams", [])]
    if codecs and all(codec in {"aac", "mp3", "alac"} for codec in codecs):
        encoding[encoding.index("-c:a") + 1] = "copy"
        index = encoding.index("-b:a")
        del encoding[index:index + 2]
    count, height, width, _ = frames.shape
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", "pipe:0", "-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
        "-pix_fmt", "yuv444p" if width % 2 or height % 2 else "yuv420p", *encoding, str(destination)]
    done = threading.Event()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        def cancel_encoder():
            while not done.wait(0.2):
                if abort and abort():
                    with contextlib.suppress(OSError):
                        process.kill()
                    return
        watcher = threading.Thread(target=cancel_encoder, daemon=True)
        watcher.start()
        try:
            for index, frame in enumerate(frames):
                _check_abort(abort)
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                if progress:
                    progress("Saving refined video", index + 1, count)
            process.stdin.close()
            if process.wait(timeout=60):
                errors.seek(0)
                raise RuntimeError("FFmpeg: " + errors.read().decode("utf-8", "replace")[-2000:])
            _check_abort(abort)
        finally:
            done.set()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            with contextlib.suppress(OSError):
                process.stdin.close()
            watcher.join(timeout=1)


def blend_overlap(previous, current, overlap):
    """Blend decoded windows on their storage device, regardless of MMGP's default."""
    import torch
    if overlap:
        weight = torch.linspace(0, torch.pi, overlap, device=current.device, dtype=torch.float32)
        weight = weight.cos().mul(-0.5).add(0.5).view(1, -1, 1, 1)
        current[:, :overlap] = previous[:, -overlap:].to(current.device).float().lerp(
            current[:, :overlap].float(), weight).round().to(current.dtype)
    return current


def process_video(source, destination, *, options=None, analysis_id=None, assignments=None, abort=None, progress=None, seed=0):
    import numpy as np
    import torch
    from postprocessing.h3_face_refiner.face import crop_face_track, stitch
    from postprocessing.h3_face_refiner.runtime import refinement_model, refine_window, window_starts
    options = normalize_options(options)
    assignments = normalize_assignments(assignments)
    if assignments and not analysis_id:
        raise ValueError("Detect faces before assigning characters")
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("Face refinement must save a new video")
    data = load_analysis(analysis_id, source, options["face_count"]) if analysis_id else analyze(source, options, abort=abort, progress=progress)
    mappings = resolve_mappings(data, assignments)
    selected = [item for item in mappings if not item["skip"]]
    report = {"analysis_id": data["id"], "faces_detected": len(data["faces"]),
              "faces_refined": len(selected), "assignments": mappings, "warnings": data["warnings"]}
    if not selected:
        # No synthetic generation if there are no relevant faces (or all are
        # skipped). A byte-for-byte copy is a valid, explicitly reported no-op.
        try:
            _check_abort(abort)
            if Path(source).suffix.lower() == Path(destination).suffix.lower():
                with open(source, "rb") as original, open(destination, "wb") as copy:
                    while block := original.read(1024 * 1024):
                        _check_abort(abort)
                        copy.write(block)
                _check_abort(abort)
            else:
                # A WebM/MKV upload must not be copied into a misleading
                # .mp4 filename. Preserve its frames and audio in MP4.
                with decoded_video(source, abort, progress) as (frames, geometry, _):
                    _encode(source, destination, frames, geometry["fps"], abort, progress)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(destination).unlink()
            raise
        return {**report, "unchanged": True}
    refs = {}
    directory = analysis_directory(data["id"])
    try:
        with decoded_video(source, abort, progress) as (frames, geometry, temporary):
            if any(geometry[key] != data[key] for key in ("frames", "width", "height", "fps")):
                raise ValueError("Source geometry changed after detection. Detect faces again.")
            output_array = np.memmap(temporary / "output.rgb", mode="w+", dtype=np.uint8, shape=tuple(frames.shape))
            output = torch.from_numpy(output_array)
            base_array = np.memmap(temporary / "base.rgb", mode="w+", dtype=np.uint8, shape=tuple(frames.shape))
            base_frames = torch.from_numpy(base_array)
            for start in range(0, geometry["frames"], 24):
                _check_abort(abort)
                output[start:start + 24].copy_(frames[start:start + 24])
            try:
                for mapping in selected:
                    track = mapping["track_id"]
                    if mapping["character_id"]:
                        image = character_image(mapping["character_id"])
                        path = temporary / f"reference-{track}.png"
                        image.save(path)
                        refs[track] = path
                    else:
                        refs[track] = directory / f"face-{track}.png"
                # Decode only source audio windows. For a silent clip the
                # zero waveform locks the hidden H3 audio rows to silence.
                with refinement_model(options["model"], progress) as (model, model_type):
                    report["model_type"] = model_type
                    for face_number, mapping in enumerate(selected, 1):
                        for base_start in range(0, geometry["frames"], 24):
                            _check_abort(abort)
                            base_frames[base_start:base_start + 24].copy_(output[base_start:base_start + 24])
                        transform = data["transforms"][mapping["track_id"] - 1]
                        for segment_start, segment_stop in transform["segments"]:
                            starts = window_starts(segment_stop - segment_start, options["window_frames"])
                            previous_stop = segment_start
                            previous_tail = None
                            for window_index, offset in enumerate(starts, 1):
                                _check_abort(abort)
                                start = segment_start + offset
                                stop = min(segment_stop, start + options["window_frames"])
                                def face_progress(phase, step=None, total=None):
                                    if progress:
                                        progress(f"Face {face_number}/{len(selected)} · Window {window_index}/{len(starts)} · {phase}", step, total)
                                crop = crop_face_track(frames, transform, start, stop)
                                audio, rate = audio_window(source, start, stop - start, geometry["fps"])
                                refined = refine_window(model, crop, refs[mapping["track_id"]],
                                    fps=geometry["fps"], audio=audio, audio_rate=rate, options=options,
                                    seed=int(seed) + mapping["track_id"] + start, abort=abort, progress=face_progress)
                                overlap = max(0, previous_stop - start)
                                if overlap and previous_tail is not None:
                                    blend_overlap(previous_tail, refined, overlap)
                                next_start = segment_start + starts[window_index] if window_index < len(starts) else stop
                                previous_tail = refined[:, next_start - start:].clone() if next_start < stop else None
                                previous_stop = stop
                                segment = {**transform, "frames": stop - start}
                                for key in ("boxes", "face_rect", "weights", "detected", "active"):
                                    segment[key] = transform[key][start:stop]
                                # Each overlapping crop is composited once
                                # over the same base, avoiding doubled masks.
                                output[start:stop].copy_(base_frames[start:stop])
                                stitched = stitch(output[start:stop], refined, segment, paste_region="face_only",
                                    mask_dilation=24, feather=24, colour_match=1.0, blend=1.0,
                                    undetected_frames="fade_out", feather_scales_with_crop=False,
                                    abort_callback=abort, progress_callback=face_progress)
                                _check_abort(abort)
                                if stitched is None:
                                    raise RuntimeError("Face compositing did not complete")
                _encode(source, destination, output_array, geometry["fps"], abort, progress)
            finally:
                output = base_frames = None
                output_array._mmap.close()
                base_array._mmap.close()
        return report
    except BaseException:
        with contextlib.suppress(OSError):
            Path(destination).unlink()
        raise
