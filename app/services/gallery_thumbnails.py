"""On-demand, bounded first-frame posters for gallery videos."""

from concurrent.futures import Future
import hashlib
import os
import shutil
import subprocess
import threading
import uuid


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".gif"}
MAX_CONCURRENT_GENERATIONS = 2
GENERATION_TIMEOUT_SECONDS = 10
SLOT_WAIT_SECONDS = 8
MAX_WAIT_SECONDS = GENERATION_TIMEOUT_SECONDS + SLOT_WAIT_SECONDS + 2
POSTER_SIZE = 480

_generation_slots = threading.BoundedSemaphore(MAX_CONCURRENT_GENERATIONS)
_inflight_lock = threading.Lock()
_inflight: dict[tuple[str, int, int], Future[str | None]] = {}


def _source_signature(path: str) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    if not os.path.isfile(path):
        return None
    return stat.st_mtime_ns, stat.st_size


def _is_jpeg(path: str) -> bool:
    try:
        with open(path, "rb") as stream:
            return stream.read(3) == b"\xff\xd8\xff"
    except OSError:
        return False


def _cache_path(source_path: str, signature: tuple[int, int], cache_dir: str) -> str:
    source_key = hashlib.sha256(os.fsencode(source_path)).hexdigest()
    source_cache = os.path.join(cache_dir, source_key)
    return os.path.join(source_cache, f"{signature[0]}-{signature[1]}.jpg")


def _render_poster(source_path: str, destination: str, ffmpeg: str) -> bool:
    """Render only the first video frame, with CPU and wall-clock limits."""
    temp_path = os.path.join(
        os.path.dirname(destination), f".{uuid.uuid4().hex}.part.jpg"
    )
    command = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-threads", "1", "-i", source_path,
        "-map", "0:v:0", "-frames:v", "1", "-an", "-sn", "-dn",
        "-vf", f"scale={POSTER_SIZE}:{POSTER_SIZE}:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-filter_threads", "1", "-threads", "1", "-q:v", "4", temp_path,
    ]
    try:
        run_options = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "timeout": GENERATION_TIMEOUT_SECONDS,
            "check": False,
        }
        if os.name == "nt":
            run_options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(command, **run_options)
        if result.returncode != 0 or not _is_jpeg(temp_path):
            return False
        os.replace(temp_path, destination)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def get_video_poster(
    source_path: str,
    *,
    cache_dir: str | None = None,
    ffmpeg: str | None = None,
) -> str | None:
    """Return a cached JPEG poster, generating it only when requested.

    Cache identity includes the canonical source path, its nanosecond mtime,
    and size. Identical concurrent requests share one render. Cache files live
    outside gallery output directories and are published with ``os.replace``.
    Invalid, missing, unsupported, or un-decodable media returns ``None``.
    """
    if not source_path or os.path.splitext(source_path)[1].lower() not in VIDEO_EXTENSIONS:
        return None

    source_path = os.path.realpath(source_path)
    signature = _source_signature(source_path)
    if signature is None:
        return None

    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "cache", "gallery-thumbnails")
    cache_dir = os.path.abspath(os.fspath(cache_dir))
    destination = _cache_path(source_path, signature, cache_dir)
    if _is_jpeg(destination):
        return destination

    ffmpeg_bin = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None

    key = (source_path, signature[0], signature[1])
    with _inflight_lock:
        future = _inflight.get(key)
        owner = future is None
        if owner:
            future = Future()
            _inflight[key] = future

    if not owner:
        try:
            return future.result(timeout=MAX_WAIT_SECONDS)
        except Exception:
            return None

    poster = None
    acquired = False
    try:
        if _source_signature(source_path) != signature:
            return None
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if _is_jpeg(destination):
            poster = destination
        elif _generation_slots.acquire(timeout=SLOT_WAIT_SECONDS):
            acquired = True
            if _render_poster(source_path, destination, ffmpeg_bin):
                # Do not publish a frame as current if the source changed while
                # ffmpeg was decoding it.
                if _source_signature(source_path) == signature and _is_jpeg(destination):
                    poster = destination
                else:
                    try:
                        os.unlink(destination)
                    except OSError:
                        pass
    except OSError:
        poster = None
    finally:
        if acquired:
            _generation_slots.release()
        if not future.done():
            future.set_result(poster)
        with _inflight_lock:
            if _inflight.get(key) is future:
                _inflight.pop(key, None)
    return poster
