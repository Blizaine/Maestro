"""Filesystem storage callbacks for the CPU-only Hy-Nomas Worlds workflow.

The service owns ``<output_root>/Worlds`` and never changes the active
generation workspace. Completed-job filenames are treated only as basenames
under the job's frozen output directory. Media is copied into a private
staging directory, validated there, and atomically published under a stable
request-derived world ID.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from typing import Any, Callable, Mapping


MAX_WORLD_BYTES = 3 << 30
MAX_VIDEO_BYTES = 3 << 30
MAX_RECORDING_BYTES = 3 << 30
MAX_POSTER_BYTES = 32 << 20
MAX_TENSOR_HEADER_BYTES = 16 << 20
MAX_CHAIN_LENGTH = 256
_WORLD_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov", ".webm"})
_STORAGE_NAMESPACE = uuid.UUID("5c939a0a-b5be-4fb0-a0f9-7bb8b61696c1")


class WorldsStorageError(ValueError):
    """A completed world or storage record did not meet its contract."""


def _safe_request_id(value: Any) -> str:
    if not isinstance(value, str) or not _UUID_RE.fullmatch(value):
        raise WorldsStorageError("World request ID must be a canonical lowercase UUID.")
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError("non-canonical UUID")
    except (ValueError, AttributeError) as error:
        raise WorldsStorageError("World request ID must be a canonical lowercase UUID.") from error
    return value


def _safe_world_id(value: Any) -> str:
    if not isinstance(value, str) or not _WORLD_ID_RE.fullmatch(value):
        raise WorldsStorageError("World ID is invalid.")
    return value


def _safe_basename(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "\x00" in value
        or any(ord(character) < 32 or character in ':<>|"*?' for character in value)
        or value.endswith((" ", "."))
        or "/" in value
        or "\\" in value
        or Path(value).name != value
        or PureWindowsPath(value).name != value
        or PureWindowsPath(value).drive
    ):
        raise WorldsStorageError("Job output filenames must be basenames.")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing a partial receipt."""
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _validate_jpeg(data: bytes) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise WorldsStorageError("World archive poster is not image data.")
    data = bytes(data)
    if not data or len(data) > MAX_POSTER_BYTES:
        raise WorldsStorageError("World archive poster is empty or too large.")
    try:
        from PIL import Image

        import io

        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or image.width <= 0 or image.height <= 0:
                raise WorldsStorageError("World archive poster must be a JPEG image.")
            image.verify()
    except WorldsStorageError:
        raise
    except Exception as error:
        raise WorldsStorageError("World archive poster is not a valid JPEG image.") from error
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise WorldsStorageError("World archive poster has invalid JPEG boundaries.")
    return data


class _TensorHeaderOnly:
    """Byte view exposing only a safetensors header and the true entry size."""

    def __init__(self, prefix: bytes, header: bytes, total_size: int):
        self._prefix = prefix
        self._header = header
        self._total_size = total_size

    def __len__(self) -> int:
        return self._total_size

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise IndexError("Only header slices are available.")
        start = 0 if key.start is None else key.start
        stop = self._total_size if key.stop is None else key.stop
        if key.step not in (None, 1) or start < 0 or stop < start:
            raise IndexError("Invalid safetensors header slice.")
        available = self._prefix + self._header
        if stop > len(available):
            raise IndexError("Safetensors payload bytes are intentionally unavailable.")
        return available[start:stop]


def _read_world_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read validated world metadata and poster without loading tensor payloads.

    ``world_file._read(load_tensors=False)`` validates the ZIP contract,
    manifest, thumbnail, and poster. We additionally validate the safetensors
    header using a bounded read so tensor dimensions and byte ranges agree
    with the manifest without allocating the potentially multi-gigabyte data.
    Importing the Hy-Nomas reader is lazy to keep storage import CPU-light.
    """
    from models.hy_nomas import world_file

    manifest, _tensors, _thumbnail = world_file._read(path, load_tensors=False)
    with zipfile.ZipFile(path, "r") as archive:
        tensor_info = archive.getinfo("tensors.safetensors")
        if tensor_info.file_size < 8 + 2 or tensor_info.file_size > (2 << 30) + MAX_TENSOR_HEADER_BYTES + 8:
            raise WorldsStorageError("World safetensors entry has an invalid size.")
        with archive.open(tensor_info, "r") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise WorldsStorageError("World safetensors header is truncated.")
            header_size = int.from_bytes(prefix, "little", signed=False)
            if not 0 < header_size <= MAX_TENSOR_HEADER_BYTES:
                raise WorldsStorageError("World safetensors header exceeds its size limit.")
            header = stream.read(header_size)
            if len(header) != header_size:
                raise WorldsStorageError("World safetensors header is truncated.")
            entry_bytes_read = 8 + header_size
            # Stream the opaque tensor payload to validate the ZIP member CRC
            # while keeping memory bounded; tensor values are intentionally
            # not materialized or loaded into Torch.
            while True:
                payload_chunk = stream.read(8 * 1024 * 1024)
                if not payload_chunk:
                    break
                entry_bytes_read += len(payload_chunk)
            if entry_bytes_read != tensor_info.file_size:
                raise WorldsStorageError("World safetensors payload is truncated.")
        world_file._validate_tensor_header(
            _TensorHeaderOnly(prefix, header, tensor_info.file_size), manifest
        )
        poster = archive.read("poster.jpg")
    raw_manifest = json.loads(archive_manifest_bytes(path).decode("utf-8"))
    if not raw_manifest.get("world_id"):
        # world_file historically fills this value with uuid4 while parsing.
        # Keep legacy identity deterministic across reads and restarts.
        archive_digest, _size = _digest_file(path)
        manifest["world_id"] = f"legacy-{archive_digest[:32]}"
    return manifest, _validate_jpeg(poster)


def archive_manifest_bytes(path: Path) -> bytes:
    """Read just the bounded manifest bytes from a validated-size archive."""
    with zipfile.ZipFile(path, "r") as archive:
        info = archive.getinfo("manifest.json")
        if info.file_size <= 0 or info.file_size > (8 << 20):
            raise WorldsStorageError("World manifest exceeds its size limit.")
        return archive.read(info)


def _validate_manifest_for_clip(manifest: Any) -> tuple[int, int, int, int, str | None]:
    if not isinstance(manifest, dict) or manifest.get("model_type") != "hy_nomas":
        raise WorldsStorageError("World archive manifest is not a Hy-Nomas world.")
    width, height = manifest.get("width"), manifest.get("height")
    fps = manifest.get("fps")
    if (
        not _is_int(width)
        or not _is_int(height)
        or width <= 0
        or height <= 0
        or width % 32
        or height % 32
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) != 24.0
    ):
        raise WorldsStorageError("World archive geometry or frame rate is invalid.")
    generations, segments = manifest.get("generations"), manifest.get("segments")
    if not isinstance(generations, list) or not generations or not isinstance(segments, list) or not segments:
        raise WorldsStorageError("World archive must contain generations and segments.")
    current_generation = len(generations) - 1
    expected_frames = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise WorldsStorageError("World archive contains a malformed segment.")
        generation = segment.get("generation")
        frames = segment.get("video_frames")
        if (
            not _is_int(generation)
            or not 0 <= generation < len(generations)
            or not _is_int(frames)
            or frames <= 0
        ):
            raise WorldsStorageError("World archive contains invalid segment ownership or frame counts.")
        if generation == current_generation:
            expected_frames += frames
    if expected_frames <= 0:
        raise WorldsStorageError("World archive has no video frames for its latest generation.")
    archive_world_id = manifest.get("world_id")
    if archive_world_id is not None and (
        not isinstance(archive_world_id, str) or not archive_world_id or len(archive_world_id) > 128
    ):
        raise WorldsStorageError("World archive identity is invalid.")
    return width, height, 24, expected_frames, archive_world_id


def _normalise_probe_result(value: Any) -> dict[str, int | float]:
    """Accept a compact injected result or ffprobe's JSON result."""
    if not isinstance(value, Mapping):
        raise WorldsStorageError("ffprobe did not return video metadata.")
    if "streams" in value:
        streams = value.get("streams")
        stream = next(
            (item for item in streams if isinstance(item, Mapping)), None
        ) if isinstance(streams, list) else None
        if stream is None:
            raise WorldsStorageError("Video has no readable video stream.")
        value = {
            "width": stream.get("width"),
            "height": stream.get("height"),
            "frame_count": stream.get("nb_read_frames", stream.get("nb_frames")),
            "r_frame_rate": stream.get("r_frame_rate"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
        }
    def as_positive_int(item):
        if isinstance(item, str) and item.isdecimal():
            item = int(item)
        return item if _is_int(item) and item > 0 else None

    width = as_positive_int(value.get("width"))
    height = as_positive_int(value.get("height"))
    frames = as_positive_int(
        value.get("frame_count", value.get("nb_read_frames", value.get("nb_frames")))
    )
    if any(item is None for item in (width, height, frames)):
        raise WorldsStorageError("ffprobe must report positive geometry and an actual decoded frame count.")

    def parse_rate(rate):
        if isinstance(rate, bool):
            return None
        if isinstance(rate, (int, float)):
            parsed = float(rate)
        elif isinstance(rate, str):
            value = rate.strip()
            if not value or value in {"N/A", "0/0"}:
                return None
            if "/" in value:
                numerator, denominator = value.split("/", 1)
                if not numerator.isdecimal() or not denominator.isdecimal() or int(denominator) == 0:
                    return None
                parsed = int(numerator) / int(denominator)
            else:
                try:
                    parsed = float(value)
                except ValueError:
                    return None
        else:
            return None
        return parsed if math.isfinite(parsed) else None

    raw_rates = [
        value.get(name)
        for name in ("r_frame_rate", "avg_frame_rate", "frame_rate", "fps")
        if value.get(name) is not None
        and str(value.get(name)).strip() not in {"", "N/A", "0/0"}
    ]
    rates = [parse_rate(rate) for rate in raw_rates]
    if not rates or any(rate is None or not math.isclose(rate, 24.0, rel_tol=0.0, abs_tol=0.001) for rate in rates):
        raise WorldsStorageError("ffprobe must confirm a 24 fps world video.")
    return {"width": width, "height": height, "frame_count": frames, "fps": 24}


class WorldsStorage:
    """Own durable Worlds files and implement ``worlds_workflow`` callbacks.

    ``output_root`` is the application's fixed outputs directory; the service
    always appends ``Worlds``. Job callbacks must supply ``out_dir`` (or
    ``workspace_dir``) plus basename-only ``output_files``. The source
    workspace is read-only from this service's perspective.
    """

    def __init__(
        self,
        output_root: str | os.PathLike[str],
        *,
        ffprobe: str = "ffprobe",
        ffmpeg: str = "ffmpeg",
        probe_video: Callable[[Path], Any] | None = None,
        decode_video: Callable[[Path], Any] | None = None,
        read_world_archive: Callable[[Path], tuple[dict[str, Any], bytes]] | None = None,
        ffmpeg_runner: Callable[[list[str]], Any] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.output_root = Path(output_root).expanduser().resolve()
        self.root = self.output_root / "Worlds"
        self.worlds_dir = self.root / "worlds"
        self.receipts_dir = self.root / "receipts"
        self.recordings_dir = self.root / "recordings"
        if self.root.is_symlink():
            raise WorldsStorageError("Worlds storage root must not be a symbolic link.")
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (self.worlds_dir, self.receipts_dir, self.recordings_dir):
            if directory.is_symlink():
                raise WorldsStorageError("Worlds storage directories must not be symbolic links.")
        self.worlds_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.ffprobe = ffprobe
        self.ffmpeg = ffmpeg
        self._probe_video_cb = probe_video
        self._decode_video_cb = decode_video
        self._world_reader_cb = read_world_archive
        self._ffmpeg_runner = ffmpeg_runner
        self._clock = clock
        self._lock = threading.RLock()

    # Durable request receipt callbacks ---------------------------------

    def read_receipt(self, request_id: str) -> dict[str, Any] | None:
        request_id = _safe_request_id(request_id)
        path = self.receipts_dir / f"{request_id}.json"
        with self._lock:
            if path.is_symlink():
                raise WorldsStorageError("Stored world receipt must not be a symbolic link.")
            try:
                resolved = path.resolve(strict=True)
                if resolved.parent != self.receipts_dir.resolve(strict=True):
                    raise WorldsStorageError("Stored world receipt escapes its receipt directory.")
                value = json.loads(resolved.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise WorldsStorageError("Stored world receipt is unreadable.") from error
        if not isinstance(value, dict) or value.get("client_request_id") != request_id:
            raise WorldsStorageError("Stored world receipt has a mismatched request owner.")
        return value

    def write_receipt(self, request_id: str, receipt: dict[str, Any]) -> None:
        request_id = _safe_request_id(request_id)
        if not isinstance(receipt, dict) or receipt.get("client_request_id") != request_id:
            raise WorldsStorageError("Receipt request ID does not match its storage key.")
        with self._lock:
            _atomic_json(self.receipts_dir / f"{request_id}.json", receipt)

    # Public world lookup and files -------------------------------------

    def _world_dir(self, world_id: str) -> Path:
        safe_id = _safe_world_id(world_id)
        return self.worlds_dir / safe_id

    def _read_record(self, world_id: str) -> dict[str, Any]:
        directory = self._world_dir(world_id)
        if directory.is_symlink():
            raise WorldsStorageError("Stored world directory must not be a symbolic link.")
        resolved_directory = directory.resolve(strict=True)
        if resolved_directory.parent != self.worlds_dir.resolve(strict=True):
            raise WorldsStorageError("Stored world directory escapes Worlds storage.")
        path = directory / "record.json"
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("World was not found.")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WorldsStorageError("Stored world record is unreadable.") from error
        if not isinstance(record, dict) or record.get("world_id") != world_id:
            raise WorldsStorageError("Stored world record has a mismatched identity.")
        if not _UUID_RE.fullmatch(str(record.get("client_request_id", ""))):
            raise WorldsStorageError("Stored world record has an invalid request owner.")
        return record

    @staticmethod
    def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
        fields = (
            "world_id", "title", "description", "created_at", "updated_at",
            "video_url", "poster_url", "thumbnail_url", "recording_available",
            "segment_count", "last_action",
        )
        result = {key: record[key] for key in fields if record.get(key) is not None}
        result["recording_available"] = record.get("recording_available") is True
        return result

    def list_worlds(self) -> list[dict[str, Any]]:
        records = []
        for directory in self.worlds_dir.iterdir():
            if not directory.is_dir() or directory.is_symlink() or not _WORLD_ID_RE.fullmatch(directory.name):
                continue
            try:
                records.append(self._public_record(self._read_record(directory.name)))
            except (FileNotFoundError, WorldsStorageError):
                continue
        records.sort(key=lambda item: (float(item.get("updated_at", 0)), item["world_id"]), reverse=True)
        return records

    def get_world(self, world_id: str) -> dict[str, Any] | None:
        try:
            return self._public_record(self._read_record(_safe_world_id(world_id)))
        except FileNotFoundError:
            return None

    def _world_asset(self, world_id: str, basename: str, *, max_bytes: int) -> Path:
        unresolved_directory = self._world_dir(world_id)
        if unresolved_directory.is_symlink():
            raise WorldsStorageError("Stored world directory must not be a symbolic link.")
        directory = unresolved_directory.resolve(strict=True)
        if directory.parent != self.worlds_dir.resolve(strict=True):
            raise WorldsStorageError("Stored world directory escapes Worlds storage.")
        name = _safe_basename(basename)
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Saved world asset is unavailable.")
        resolved = path.resolve(strict=True)
        if resolved.parent != directory:
            raise WorldsStorageError("Saved world asset escapes its world directory.")
        size = resolved.stat().st_size
        if size <= 0 or size > max_bytes:
            raise WorldsStorageError("Saved world asset has an invalid size.")
        return resolved

    def resolve_world_file(self, world_id: str) -> Path:
        record = self._read_record(_safe_world_id(world_id))
        path = self._world_asset(world_id, record["world_file"], max_bytes=MAX_WORLD_BYTES)
        if _digest_file(path)[0] != record.get("world_file_sha256"):
            raise WorldsStorageError("Stored world archive failed its integrity check.")
        return path

    def save_world(self, world_id: str) -> Path:
        # Save is a download of the already validated archive, never a model call.
        return self.resolve_world_file(world_id)

    def capture_world(self, world_id: str) -> bytes:
        record = self._read_record(_safe_world_id(world_id))
        path = self._world_asset(world_id, record["poster_file"], max_bytes=MAX_POSTER_BYTES)
        data = _validate_jpeg(path.read_bytes())
        if hashlib.sha256(data).hexdigest() != record.get("poster_sha256"):
            raise WorldsStorageError("Stored world poster failed its integrity check.")
        return data

    # Completed-job publication ----------------------------------------

    def _validate_job_owner(self, job: Any, receipt: Any) -> tuple[str, str, str, dict[str, Any]]:
        if not isinstance(job, dict) or not isinstance(receipt, dict):
            raise WorldsStorageError("Completed job and receipt must be objects.")
        request_id = _safe_request_id(receipt.get("client_request_id"))
        submission_id = receipt.get("client_submission_id")
        if submission_id != request_id:
            raise WorldsStorageError("Receipt submission ID does not match its request ID.")
        job_id = receipt.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise WorldsStorageError("Receipt has no reconciled job ID.")
        job_ids = [value for value in (job.get("job_id"), job.get("id")) if value is not None]
        if not job_ids or any(not isinstance(value, str) or not value for value in job_ids):
            raise WorldsStorageError("Completed job has no valid ID.")
        if len(set(job_ids)) != 1 or job_ids[0] != job_id:
            raise WorldsStorageError("Completed job ID does not match the receipt owner.")
        if job.get("client_submission_id") != submission_id:
            raise WorldsStorageError("Completed job submission ID does not match the receipt owner.")
        if job.get("status") != "completed":
            raise WorldsStorageError("Only completed generation jobs can publish worlds.")
        request = receipt.get("request")
        if not isinstance(request, dict) or request.get("operation") not in {"create", "continue", "move"}:
            raise WorldsStorageError("Receipt has no supported world operation.")
        return request_id, job_id, submission_id, request

    def _source_paths(self, job: dict[str, Any]) -> tuple[Path, Path]:
        workspace_value = job.get("out_dir", job.get("workspace_dir"))
        if not isinstance(workspace_value, (str, os.PathLike)):
            raise WorldsStorageError("Completed job does not identify its frozen output directory.")
        workspace = Path(workspace_value).expanduser().resolve(strict=True)
        if not workspace.is_dir():
            raise WorldsStorageError("Completed job output directory is unavailable.")
        outputs = job.get("output_files")
        if not isinstance(outputs, list):
            raise WorldsStorageError("Completed job does not list output filenames.")
        names = [_safe_basename(item) for item in outputs]
        video_names = [name for name in names if Path(name).suffix.lower() in _VIDEO_SUFFIXES]
        sidecar_names = [name for name in names if Path(name).suffix.lower() == ".wgpworld"]
        if len(video_names) != 1:
            raise WorldsStorageError("Completed world job must own exactly one video output.")
        video_name = video_names[0]
        expected_sidecar_name = f"{Path(video_name).stem}.wgpworld"
        if sidecar_names:
            if len(sidecar_names) != 1:
                raise WorldsStorageError("Completed world job has ambiguous world sidecars.")
            sidecar_name = sidecar_names[0]
            if sidecar_name != expected_sidecar_name:
                raise WorldsStorageError("World sidecar basename does not match its video output.")
        else:
            sidecar_name = expected_sidecar_name
        video = self._resolve_job_basename(workspace, video_name)
        sidecar = self._resolve_job_basename(workspace, sidecar_name)
        if video.stat().st_size <= 0 or video.stat().st_size > MAX_VIDEO_BYTES:
            raise WorldsStorageError("Completed world video is empty or too large.")
        if sidecar.stat().st_size <= 0 or sidecar.stat().st_size > MAX_WORLD_BYTES:
            raise WorldsStorageError("Completed world sidecar is empty or too large.")
        return video, sidecar

    @staticmethod
    def _resolve_job_basename(workspace: Path, basename: str) -> Path:
        candidate = workspace / _safe_basename(basename)
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(f"Completed job output {basename!r} is unavailable.")
        resolved = candidate.resolve(strict=True)
        if resolved.parent != workspace:
            raise WorldsStorageError("Completed job output escapes its frozen workspace.")
        return resolved

    def _reader(self, path: Path) -> tuple[dict[str, Any], bytes]:
        callback = self._world_reader_cb or _read_world_archive
        result = callback(path)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], dict)
        ):
            raise WorldsStorageError("World archive reader returned an invalid result.")
        manifest, poster = result
        return manifest, _validate_jpeg(poster)

    def _probe(self, path: Path) -> dict[str, int]:
        if self._probe_video_cb is not None:
            return _normalise_probe_result(self._probe_video_cb(path))
        result = subprocess.run(
            [
                self.ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=width,height,nb_read_frames,nb_frames,r_frame_rate,avg_frame_rate",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            raise WorldsStorageError(f"ffprobe could not inspect world video: {result.stderr[-1000:]}")
        try:
            return _normalise_probe_result(json.loads(result.stdout))
        except json.JSONDecodeError as error:
            raise WorldsStorageError("ffprobe returned malformed video metadata.") from error

    def _decode(self, path: Path) -> None:
        if self._decode_video_cb is not None:
            result = self._decode_video_cb(path)
            if result is False:
                raise WorldsStorageError("CPU full-decode validation failed.")
            return
        result = subprocess.run(
            [
                self.ffmpeg,
                "-nostdin", "-v", "error", "-xerror", "-hwaccel", "none",
                "-i", str(path), "-map", "0:v:0", "-an", "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if result.returncode != 0:
            raise WorldsStorageError(f"CPU full-decode validation failed: {result.stderr[-1000:]}")

    @staticmethod
    def _validate_probe(
        probe: Mapping[str, int | float], width: int, height: int, frames: int
    ) -> None:
        if probe.get("fps") != 24:
            raise WorldsStorageError("World video frame rate must be 24 fps.")
        if (probe["width"], probe["height"]) != (width, height):
            raise WorldsStorageError(
                "World video geometry does not match the validated .wgpworld manifest."
            )
        if probe["frame_count"] != frames:
            raise WorldsStorageError(
                "World video frame count does not match its latest manifest generation."
            )

    @staticmethod
    def _request_world_id(request_id: str) -> str:
        return uuid.uuid5(_STORAGE_NAMESPACE, request_id).hex

    def _parent_record(
        self, request: Mapping[str, Any], world_id: str
    ) -> tuple[str | None, list[str], dict[str, Any] | None, dict[str, Any] | None]:
        operation = request["operation"]
        if operation == "create":
            return None, [], None, None
        parent_id = _safe_world_id(request.get("world_id"))
        parent = self._read_record(parent_id)
        parent_path = self._world_asset(
            parent_id, parent["world_file"], max_bytes=MAX_WORLD_BYTES
        )
        if _digest_file(parent_path)[0] != parent.get("world_file_sha256"):
            raise WorldsStorageError("Saved parent world archive failed its integrity check.")
        parent_manifest, _parent_poster = self._reader(parent_path)
        parent_geometry = _validate_manifest_for_clip(parent_manifest)
        if (
            parent_geometry[0] != parent.get("width")
            or parent_geometry[1] != parent.get("height")
            or len(parent_manifest.get("generations", [])) != parent.get("generation_count")
        ):
            raise WorldsStorageError("Saved parent world metadata disagrees with its archive.")
        chain = parent.get("parent_chain")
        if not isinstance(chain, list) or any(not isinstance(item, str) for item in chain):
            raise WorldsStorageError("Stored parent world has an invalid ancestry chain.")
        normalized_chain = [_safe_world_id(item) for item in chain]
        if len(normalized_chain) >= MAX_CHAIN_LENGTH:
            raise WorldsStorageError("World parent chain exceeds the supported limit.")
        if world_id == parent_id or world_id in normalized_chain or len(set(normalized_chain)) != len(normalized_chain):
            raise WorldsStorageError("World parent chain contains a cycle or duplicate owner.")
        return parent_id, [*normalized_chain, parent_id], parent, parent_manifest

    def publish_completed_generation(
        self, job: dict[str, Any], receipt: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate then idempotently store one completed generation result."""
        request_id, job_id, submission_id, request = self._validate_job_owner(job, receipt)
        world_id = self._request_world_id(request_id)
        final_dir = self.worlds_dir / world_id
        with self._lock:
            if final_dir.exists():
                return self._published_result(final_dir, request_id, job_id, submission_id)

            parent_id, parent_chain, parent, parent_manifest = self._parent_record(request, world_id)
            source_video, source_sidecar = self._source_paths(job)
            stage = self.worlds_dir / f".publish-{request_id}-{uuid.uuid4().hex}"
            stage.mkdir(mode=0o700)
            try:
                video_hash, video_size = self._copy_into(source_video, stage / "clip.mp4", MAX_VIDEO_BYTES)
                world_hash, world_size = self._copy_into(source_sidecar, stage / "state.wgpworld", MAX_WORLD_BYTES)
                manifest, poster = self._reader(stage / "state.wgpworld")
                width, height, fps, expected_frames, archive_world_id = _validate_manifest_for_clip(manifest)
                if parent is not None and (parent.get("width"), parent.get("height")) != (width, height):
                    raise WorldsStorageError("Continued world geometry must match its saved parent.")
                if parent_manifest is not None:
                    parent_generations = parent_manifest["generations"]
                    parent_segments = parent_manifest["segments"]
                    if (
                        manifest["generations"][:-1] != parent_generations
                        or manifest["segments"][:len(parent_segments)] != parent_segments
                        or manifest["world"]["segments"][:len(parent_manifest["world"]["segments"])]
                        != parent_manifest["world"]["segments"]
                    ):
                        raise WorldsStorageError("Continuation archive does not preserve its verified parent history.")
                _atomic_bytes(stage / "poster.jpg", poster)
                poster_hash = hashlib.sha256(poster).hexdigest()
                probe = self._probe(stage / "clip.mp4")
                self._validate_probe(probe, width, height, expected_frames)
                self._decode(stage / "clip.mp4")
                parent_generation_count = int(parent.get("generation_count", 0)) if parent else 0
                generation_count = len(manifest["generations"])
                if generation_count != parent_generation_count + 1:
                    raise WorldsStorageError("World archive generation history does not match its parent chain.")
                now = float(self._clock())
                if not math.isfinite(now):
                    raise WorldsStorageError("Storage clock returned an invalid timestamp.")
                if request["operation"] == "create":
                    description = str(request.get("description", "")).strip()
                    if not description:
                        raise WorldsStorageError("Create request has no world description.")
                    title = description.splitlines()[0][:120]
                    created_at = now
                    last_action = None
                else:
                    description = parent.get("description", "") if parent else ""
                    title = parent.get("title", "Untitled world") if parent else "Untitled world"
                    created_at = parent.get("created_at", now) if parent else now
                    if request["operation"] == "move":
                        last_action = f"{request.get('action', '')} {request.get('duration_ms', '')}ms".strip()
                    else:
                        last_action = str(request.get("text", "")).strip()
                record = {
                    "version": 1,
                    "world_id": world_id,
                    "client_request_id": request_id,
                    "job_id": job_id,
                    "client_submission_id": submission_id,
                    "operation": request["operation"],
                    "parent_world_id": parent_id,
                    "parent_chain": parent_chain,
                    "created_at": created_at,
                    "updated_at": now,
                    "title": title,
                    "description": description,
                    "last_action": last_action,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": expected_frames,
                    "segment_count": len(manifest["segments"]),
                    "generation_count": generation_count,
                    "archive_world_id": archive_world_id or f"legacy-{world_hash[:32]}",
                    "recording_available": True,
                    "video_url": f"/api/v1/file/worlds/{world_id}/clip.mp4?workspace=Worlds",
                    "poster_url": f"/api/v1/worlds/{world_id}/capture",
                    "thumbnail_url": f"/api/v1/worlds/{world_id}/capture",
                    "world_file": "state.wgpworld",
                    "poster_file": "poster.jpg",
                    "clip_file": "clip.mp4",
                    "world_file_sha256": world_hash,
                    "world_file_size": world_size,
                    "clip_sha256": video_hash,
                    "clip_size": video_size,
                    "poster_sha256": poster_hash,
                }
                _atomic_json(stage / "record.json", record)
                try:
                    os.rename(stage, final_dir)
                except FileExistsError:
                    # A competing process may have published the same UUID.
                    return self._published_result(final_dir, request_id, job_id, submission_id)
                return self._published_result(final_dir, request_id, job_id, submission_id)
            finally:
                if stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def _copy_into(source: Path, destination: Path, maximum: int) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            while True:
                chunk = input_stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum:
                    raise WorldsStorageError("Completed generation output exceeds its size limit.")
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if size <= 0:
            raise WorldsStorageError("Completed generation output is empty.")
        return digest.hexdigest(), size

    def _published_result(
        self, directory: Path, request_id: str, job_id: str, submission_id: str
    ) -> dict[str, Any]:
        record_path = directory / "record.json"
        if not directory.is_dir() or directory.is_symlink() or record_path.is_symlink() or not record_path.is_file():
            raise WorldsStorageError("A world ID is already occupied by incomplete storage data.")
        record = self._read_record(directory.name)
        if (
            record.get("client_request_id") != request_id
            or record.get("job_id") != job_id
            or record.get("client_submission_id") != submission_id
        ):
            raise WorldsStorageError("A request ID is already bound to a different completed job.")
        world_file = self._world_asset(directory.name, record["world_file"], max_bytes=MAX_WORLD_BYTES)
        clip = self._world_asset(directory.name, record["clip_file"], max_bytes=MAX_VIDEO_BYTES)
        poster = self._world_asset(directory.name, record["poster_file"], max_bytes=MAX_POSTER_BYTES)
        if _digest_file(world_file)[0] != record.get("world_file_sha256"):
            raise WorldsStorageError("Stored world archive failed its integrity check.")
        if _digest_file(clip)[0] != record.get("clip_sha256"):
            raise WorldsStorageError("Stored world video failed its integrity check.")
        _validate_jpeg(poster.read_bytes())
        if _digest_file(poster)[0] != record.get("poster_sha256"):
            raise WorldsStorageError("Stored world poster failed its integrity check.")
        return {
            "verified": True,
            "video_verified": True,
            "world_file_verified": True,
            "world": self._public_record(record),
        }

    # CPU-only recording of the already verified clip chain -------------

    def _chain_records(self, world_id: str) -> list[dict[str, Any]]:
        current_id = _safe_world_id(world_id)
        reverse_chain = []
        seen = set()
        while current_id:
            if current_id in seen or len(reverse_chain) >= MAX_CHAIN_LENGTH:
                raise WorldsStorageError("World clip chain is cyclic or too long.")
            seen.add(current_id)
            record = self._read_record(current_id)
            reverse_chain.append(record)
            parent_id = record.get("parent_world_id")
            current_id = _safe_world_id(parent_id) if parent_id is not None else ""
        chain = list(reversed(reverse_chain))
        expected_ids = [record["world_id"] for record in chain[:-1]]
        if chain and chain[-1].get("parent_chain") != expected_ids:
            raise WorldsStorageError("Stored world ancestry does not match its parent records.")
        return chain

    def record_world(self, world_id: str) -> Path:
        safe_id = _safe_world_id(world_id)
        final_dir = self.recordings_dir / safe_id
        with self._lock:
            if final_dir.exists():
                return self._validated_recording(final_dir, safe_id)
            chain = self._chain_records(safe_id)
            if not chain:
                raise FileNotFoundError("World clip chain is unavailable.")
            clips = []
            width, height = chain[0]["width"], chain[0]["height"]
            for record in chain:
                if (record.get("width"), record.get("height")) != (width, height):
                    raise WorldsStorageError("World clip chain changes geometry and cannot be recorded.")
                clip = self._world_asset(record["world_id"], record["clip_file"], max_bytes=MAX_VIDEO_BYTES)
                digest, _size = _digest_file(clip)
                if digest != record.get("clip_sha256"):
                    raise WorldsStorageError("A saved world clip failed its integrity check.")
                clips.append(clip)
            stage = self.recordings_dir / f".record-{safe_id}-{uuid.uuid4().hex}"
            stage.mkdir(mode=0o700)
            temporary = stage / "recording.mp4"
            try:
                self._run_concat(clips, temporary, width, height)
                if not temporary.is_file() or temporary.stat().st_size <= 0 or temporary.stat().st_size > MAX_RECORDING_BYTES:
                    raise WorldsStorageError("ffmpeg did not produce a bounded world recording.")
                probe = self._probe(temporary)
                expected_frames = sum(int(record["frame_count"]) for record in chain)
                self._validate_probe(probe, width, height, expected_frames)
                self._decode(temporary)
                video_hash, video_size = _digest_file(temporary)
                _atomic_json(
                    stage / "recording.json",
                    {
                        "version": 1,
                        "world_id": safe_id,
                        "chain": [record["world_id"] for record in chain],
                        "width": width,
                        "height": height,
                        "frame_count": expected_frames,
                        "sha256": video_hash,
                        "size": video_size,
                    },
                )
                try:
                    os.rename(stage, final_dir)
                except FileExistsError:
                    pass
                return self._validated_recording(final_dir, safe_id)
            finally:
                if stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)

    def _validated_recording(self, directory: Path, world_id: str) -> Path:
        if directory.is_symlink() or not directory.is_dir():
            raise WorldsStorageError("Existing recording directory is not safe.")
        if directory.resolve(strict=True).parent != self.recordings_dir.resolve(strict=True):
            raise WorldsStorageError("Existing recording directory escapes Worlds storage.")
        metadata_path = directory / "recording.json"
        video_path = directory / "recording.mp4"
        if (
            metadata_path.is_symlink()
            or video_path.is_symlink()
            or not metadata_path.is_file()
            or not video_path.is_file()
        ):
            raise WorldsStorageError("Existing recording is incomplete.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WorldsStorageError("Existing recording metadata is unreadable.") from error
        if not isinstance(metadata, dict) or metadata.get("world_id") != world_id:
            raise WorldsStorageError("Existing recording has a mismatched world owner.")
        if video_path.stat().st_size <= 0 or video_path.stat().st_size > MAX_RECORDING_BYTES:
            raise WorldsStorageError("Existing recording exceeds its size limit.")
        chain = self._chain_records(world_id)
        expected_chain = [record["world_id"] for record in chain]
        if metadata.get("chain") != expected_chain:
            raise WorldsStorageError("Existing recording has a mismatched source clip chain.")
        expected_frames = sum(int(record["frame_count"]) for record in chain)
        if metadata.get("frame_count") != expected_frames:
            raise WorldsStorageError("Existing recording has a mismatched frame count.")
        expected_width, expected_height = chain[0]["width"], chain[0]["height"]
        if (metadata.get("width"), metadata.get("height")) != (expected_width, expected_height):
            raise WorldsStorageError("Existing recording has mismatched source geometry.")
        if metadata.get("size") != video_path.stat().st_size:
            raise WorldsStorageError("Existing recording metadata has an incorrect size.")
        if _digest_file(video_path)[0] != metadata.get("sha256"):
            raise WorldsStorageError("Existing recording failed its integrity check.")
        probe = self._probe(video_path)
        self._validate_probe(
            probe,
            expected_width,
            expected_height,
            expected_frames,
        )
        self._decode(video_path)
        return video_path.resolve(strict=True)

    def _run_concat(self, clips: list[Path], output: Path, width: int, height: int) -> None:
        args: list[str] = [self.ffmpeg, "-nostdin", "-y", "-v", "error"]
        for clip in clips:
            args.extend(["-i", str(clip)])
        graph_parts = []
        input_labels = []
        for index in range(len(clips)):
            label = f"v{index}"
            graph_parts.append(
                f"[{index}:v:0]setpts=PTS-STARTPTS,fps=24,scale={width}:{height},setsar=1[{label}]"
            )
            input_labels.append(f"[{label}]")
        graph = ";".join(graph_parts)
        graph += ";" + "".join(input_labels) + f"concat=n={len(clips)}:v=1:a=0[outv]"
        args.extend(
            [
                "-filter_complex", graph,
                "-map", "[outv]",
                "-an", "-c:v", "libx264", "-threads", "1",
                "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-r", "24", "-movflags", "+faststart", str(output),
            ]
        )
        if self._ffmpeg_runner is not None:
            result = self._ffmpeg_runner(args)
            if result is False:
                raise WorldsStorageError("CPU ffmpeg could not concatenate the saved clip chain.")
            return
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if result.returncode != 0:
            raise WorldsStorageError(f"CPU ffmpeg could not concatenate the saved clip chain: {result.stderr[-1000:]}")


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Atomically publish a small binary asset inside a staging directory."""
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
