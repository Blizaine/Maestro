"""Bounded-memory finishing of existing media through Maestro's job queue."""
from __future__ import annotations

import contextlib
import itertools
import json
import os
import subprocess
import tempfile
import threading
from fractions import Fraction

from .media_processing import TEMPORAL_METHODS, dlss_scale, prepare_dlss, validate_methods


class ProcessingCancelled(RuntimeError):
    pass


def _encoding_options(destination):
    extension = os.path.splitext(os.fspath(destination))[1].lower()
    if extension == ".webm":
        return ["-c:v", "libvpx-vp9", "-deadline", "good", "-cpu-used", "4", "-crf", "24", "-b:v", "0",
                "-c:a", "libopus", "-b:a", "192k"]
    if extension not in {".mp4", ".m4v", ".mov", ".mkv", ".avi"}:
        raise ValueError("Media Flow output must be MP4, MOV, MKV, AVI or WebM")
    options = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", "-b:a", "192k"]
    if extension in {".mp4", ".m4v", ".mov"}:
        options += ["-movflags", "+faststart"]
    return options


def _check_abort(abort):
    if abort and abort():
        raise ProcessingCancelled("Media processing cancelled")


def _frames(path, abort):
    import cv2
    capture = cv2.VideoCapture(os.fspath(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open source video")
        while True:
            _check_abort(abort)
            ok, frame = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    finally:
        capture.release()


def _temporal_frames(frames, method, fps, width, height, count, options, abort, progress):
    import numpy as np
    import torch
    factor = TEMPORAL_METHODS[method]
    if factor == 1:
        yield from frames
        return
    if method.startswith("rife"):
        import wgp
        from postprocessing.rife.RIFE_V4 import Model
        from postprocessing.rife.inference import process_frames
        from shared.utils import files_locator as fl
        path = fl.locate_file(wgp.RIFE_V4_FILENAME, error_if_none=False)
        if path is None:
            wgp.process_files_def(repoId="DeepBeepMeep/Wan2.1", sourceFolderList=[""],
                                  fileList=[[wgp.RIFE_V4_FILENAME]])
            path = fl.locate_file(wgp.RIFE_V4_FILENAME)
        model = Model()
        model.load_model(path, -1, device=wgp.processing_device)
        model.eval()
        model.to(device=wgp.processing_device)
        previous = None
        try:
            with torch.inference_mode():
                for index, frame in enumerate(frames):
                    _check_abort(abort)
                    if previous is not None:
                        pair = torch.from_numpy(np.stack([previous[..., :3], frame[..., :3]])).permute(3, 0, 1, 2)
                        result = process_frames(model, wgp.processing_device, pair,
                            multiplier=factor, abort_callback=abort)
                        _check_abort(abort)
                        result = result.add_(1).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
                        rgb = result.permute(1, 2, 3, 0).numpy()
                        for intermediate in rgb[:-1]:
                            yield np.dstack([intermediate, np.full((height, width), 255, np.uint8)])
                    previous = frame
                    if progress:
                        progress("RIFE interpolation", index + 1, count)
                if previous is not None:
                    # Preserve N / source_fps duration, including the last
                    # frame's display interval. Intermediates never shift audio.
                    for _ in range(factor):
                        yield previous
        finally:
            del model
        return

    from postprocessing.dlss5 import runtime
    prepare_dlss(options)
    session = runtime.FrameGenerationSession(width, height, count, factor - 1, abort)
    completed = False
    try:
        guides = runtime.FlowGuides(width, height, options["dlss_motion"])
        generated = np.empty((factor - 1, height, width, 4), dtype=np.uint8)
        previous = None
        for index, frame in enumerate(frames):
            _check_abort(abort)
            motion, reset = guides.process(frame)
            produced = session.process_frame(index, frame, motion,
                Fraction(index, 1) / Fraction(str(fps)), reset,
                generated if previous is not None else None)
            if previous is not None:
                if reset or not produced:
                    generated[:] = previous
                for intermediate in generated:
                    yield intermediate
            yield frame
            previous = frame
            if progress:
                progress("DLSS Frame Generation", index + 1, count)
        if previous is not None:
            for _ in range(factor - 1):
                yield previous
        completed = True
    finally:
        session.close(abort=not completed)


def _neural_frames(frames, scale, width, height, count, options, abort, progress):
    import cv2
    from postprocessing.dlss5 import runtime
    prepare_dlss(options, neural=True)
    session = runtime.NeuralRenderingSession(width, height, count, scale, options["dlss_intensity"], abort)
    completed = False
    try:
        guides = runtime.FlowGuides(session.render_width, session.render_height, options["dlss_motion"])
        depths = runtime.DepthGuides(session.render_width, session.render_height, options["dlss_depth"])
        for index, frame in enumerate(frames):
            _check_abort(abort)
            frame = cv2.resize(frame, (session.render_width, session.render_height), interpolation=cv2.INTER_LANCZOS4)
            motion, reset = guides.process(frame)
            depth = depths.process(frame, reset)
            _check_abort(abort)
            yield session.process_frame(index, frame, motion, reset, depth=depth)
            if progress:
                progress("DLSS Neural Rendering", index + 1, count)
        completed = True
    finally:
        session.close(abort=not completed)


def process_video(source, destination, *, spatial="", temporal="", options=None,
                  abort=None, progress=None):
    """Stream frames through persistent processors, preserving source audio.

    Only a pair of RIFE frames / a native worker's frame buffers are held in
    memory. Neural history stays live for the entire clip, with cut resets.
    """
    import cv2
    import numpy as np
    from shared.utils.video_decode import _resolve_media_binary
    normalized = validate_methods(spatial, temporal, options=options)
    encoding = _encoding_options(destination)
    if spatial and dlss_scale(spatial) is None and not spatial.startswith("lanczos"):
        raise ValueError("Media Flow supports DLSS Neural Rendering and Lanczos spatial processing")
    capture = cv2.VideoCapture(os.fspath(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or count < 1 or min(width, height) < 1:
        raise ValueError("Source video has invalid frame geometry or timing")
    ffmpeg = _resolve_media_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for Media Flow")
    # Keep compressed audio packets when the destination container accepts
    # them. This avoids another lossy pass and preserves encoder padding.
    ffprobe = _resolve_media_binary("ffprobe")
    if ffprobe:
        probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_name", "-of", "json", os.fspath(source)],
            capture_output=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if probe.returncode == 0:
            codecs = [stream.get("codec_name") for stream in json.loads(probe.stdout).get("streams", [])]
            extension = os.path.splitext(os.fspath(destination))[1].lower()
            compatible = {".mp4": {"aac", "mp3", "alac"}, ".mov": {"aac", "mp3", "alac"},
                          ".m4v": {"aac", "mp3"}, ".webm": {"opus", "vorbis"}}
            if codecs and (extension == ".mkv" or all(codec in compatible.get(extension, set()) for codec in codecs)):
                encoding[encoding.index("-c:a") + 1] = "copy"
                if "-b:a" in encoding:
                    index = encoding.index("-b:a")
                    del encoding[index:index+2]
    scale = dlss_scale(spatial)
    process = None
    written = 0
    generators = []
    encoder_done = threading.Event()
    watchdog = None
    try:
        frames = _frames(source, abort)
        generators.append(frames)
        frames = _temporal_frames(frames, temporal, fps, width, height, count, normalized, abort, progress)
        generators.append(frames)
        factor = TEMPORAL_METHODS[temporal]
        output_fps = fps * factor
        if scale is not None:
            frames = _neural_frames(frames, scale, width, height, count * factor, normalized, abort, progress)
            generators.append(frames)
        first = next(frames, None)
        if first is None:
            raise ValueError("Source video did not decode any frames")
        if spatial.startswith("lanczos"):
            scale = float(spatial.removeprefix("lanczos"))
            output_width = max(2, round(width * scale / 2) * 2)
            output_height = max(2, round(height * scale / 2) * 2)
        else:
            output_height, output_width = first.shape[:2]
            output_height += output_height % 2
            output_width += output_width % 2
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{output_width}x{output_height}",
            "-framerate", str(output_fps), "-i", "pipe:0", "-i", os.fspath(source),
            "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-pix_fmt", "yuv420p",
            *encoding, os.fspath(destination)]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=errors, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            def watch_cancel():
                while not encoder_done.wait(0.2):
                    if abort and abort():
                        with contextlib.suppress(OSError):
                            process.kill()
                        return
            watchdog = threading.Thread(target=watch_cancel, daemon=True)
            watchdog.start()
            for frame in itertools.chain([first], frames):
                _check_abort(abort)
                if spatial.startswith("lanczos"):
                    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                elif frame.shape[:2] != (output_height, output_width):
                    frame = cv2.copyMakeBorder(frame, 0, output_height - frame.shape[0],
                        0, output_width - frame.shape[1], cv2.BORDER_REPLICATE)
                try:
                    process.stdin.write(np.ascontiguousarray(frame[..., :3]).tobytes())
                except OSError as error:
                    _check_abort(abort)
                    errors.seek(0)
                    raise RuntimeError("FFmpeg: " + errors.read().decode("utf-8", "replace")[-2000:]) from error
                written += 1
                if progress:
                    progress("Encoding media", written, count * factor)
            process.stdin.close()
            try:
                code = process.wait(timeout=60)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("FFmpeg did not finish encoding") from error
            if code:
                errors.seek(0)
                raise RuntimeError("FFmpeg: " + errors.read().decode("utf-8", "replace")[-2000:])
        _check_abort(abort)
        return {"frames": written, "fps": output_fps, "width": output_width, "height": output_height}
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        with contextlib.suppress(OSError):
            os.remove(destination)
        raise
    finally:
        encoder_done.set()
        if watchdog:
            watchdog.join(timeout=1)
        for generator in reversed(generators):
            with contextlib.suppress(Exception):
                generator.close()
        if process and process.stdin and not process.stdin.closed:
            with contextlib.suppress(OSError):
                process.stdin.close()
        if (spatial or "").startswith("dlss") or temporal.startswith("dlss"):
            from postprocessing.dlss5.runtime import release_flow_model
            release_flow_model()
