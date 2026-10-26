"""Install the pinned audio.cpp server runtime used by Fish Small.

The large upstream archives are downloaded only when Fish Small is first used.
This module has no import-time installation side effects.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator


RELEASE = "v0.9.2-drama3-base-small"
REPOSITORY = "0xShug0/audio.cpp"
_RELEASE_URL = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE}"
_DOWNLOAD_TIMEOUT_SECONDS = 30
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_HASH_CHUNK_BYTES = 2 * 1024 * 1024
_OWNER_FILE = ".fish-small-runtime-owner"
_MARKER_FILE = ".fish-small-runtime.json"
_STAGING_OWNER_FILE = ".fish-small-staging-owner"
_OWNER_CONTENT = b"maestro-fish-small-runtime-v1\n"
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_VERIFIED_CACHE: dict[str, tuple[tuple[tuple[object, ...], ...], bool]] = {}


ProgressCallback = Callable[[str, int, int | None], None]
CancellationCallback = Callable[[], object]


@dataclass(frozen=True)
class RuntimeInfo:
    executable: Path
    root: Path
    environment: dict[str, str]
    backend: str
    release: str


class RuntimeInstallError(RuntimeError):
    """A pinned Fish Small native runtime could not be prepared safely."""


class RuntimeInstallCancelled(RuntimeInstallError):
    """The caller cancelled a runtime download or installation."""


class RuntimeIntegrityError(RuntimeInstallError):
    """A downloaded or installed runtime failed its integrity checks."""


class UnsafeArchiveError(RuntimeInstallError):
    """An upstream archive contains an unsafe path or file type."""


class RuntimeDependencyError(RuntimeInstallError):
    """The pinned package is missing a required runtime component."""


@dataclass(frozen=True)
class _Asset:
    name: str
    sha256: str
    size: int
    archive_format: str

    @property
    def url(self) -> str:
        return f"{_RELEASE_URL}/{self.name}"


@dataclass(frozen=True)
class _RuntimeSpec:
    variant: str
    platform_name: str
    assets: tuple[_Asset, ...]
    required_files: tuple[str, ...]
    server_name: str


_WIN_CUDA_12_4 = _RuntimeSpec(
    variant="windows-x64-cuda12.4",
    platform_name="windows",
    assets=(
        _Asset(
            "audio-v0.9.2-drama3-base-small-bin-windows-x64-cuda12.4.zip",
            "48852da40790c40f3285bca6c53ff7829b3963080f0e95b6742f868fc1d7c757",
            464_268_606,
            "zip",
        ),
        _Asset(
            "audio-v0.9.2-drama3-base-small-cudart-windows-x64-cuda12.4.zip",
            "4b1186a525c9ac70a4fbd031b47e0dfca2ebaa36cfcb760d5be1a76eb2b12da5",
            607_273_675,
            "zip",
        ),
    ),
    required_files=(
        "ggml-cuda.dll",
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
    ),
    server_name="audiocpp_server.exe",
)

_WIN_CUDA_13_3 = _RuntimeSpec(
    variant="windows-x64-cuda13.3",
    platform_name="windows",
    assets=(
        _Asset(
            "audio-v0.9.2-drama3-base-small-bin-windows-x64-cuda13.3.zip",
            "b570efa4b17c2aa32c431c653489c009b58f57f60d68782b3b94b6c06304a9e8",
            275_837_338,
            "zip",
        ),
        _Asset(
            "audio-v0.9.2-drama3-base-small-cudart-windows-x64-cuda13.3.zip",
            "cf58e70c53db680af5e4b8e5ef538ffd1d81bd489ef49843fe486fcd079c9182",
            575_457_454,
            "zip",
        ),
    ),
    required_files=(
        "ggml-cuda.dll",
        "cudart64_13.dll",
        "cublas64_13.dll",
        "cublasLt64_13.dll",
        "cufft64_12.dll",
    ),
    server_name="audiocpp_server.exe",
)

_LINUX_CUDA_12_8 = _RuntimeSpec(
    variant="linux-x64-cuda12.8",
    platform_name="linux",
    assets=(
        _Asset(
            "audio-v0.9.2-drama3-base-small-bin-ubuntu-x64-cuda12.8-colab.tar.gz",
            "fc59545092eab869013c74d11273c17c96a050befe257523b9019d1a5073962c",
            226_937_120,
            "tar.gz",
        ),
        _Asset(
            "audio-v0.9.2-drama3-base-small-cudart-ubuntu-x64-cuda12.8.tar.gz",
            "d6fdfd11a3d09a4df537a9f5b2aadbdcafb2d6761fc64adbb6f863ae3b2cfa8d",
            787_019_365,
            "tar.gz",
        ),
    ),
    required_files=(
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcufft.so.11",
    ),
    server_name="audiocpp_server",
)


def _validate_capability(capability: tuple[int, int] | None) -> tuple[int, int] | None:
    if capability is None:
        return None
    try:
        major, minor = capability
    except (TypeError, ValueError) as error:
        raise ValueError("device_capability must be a (major, minor) pair") from error
    if (
        isinstance(major, bool)
        or isinstance(minor, bool)
        or not isinstance(major, int)
        or not isinstance(minor, int)
        or major < 0
        or minor < 0
    ):
        raise ValueError("device_capability must contain non-negative integer values")
    return major, minor


def _select_runtime_spec(
    backend: str,
    device_capability: tuple[int, int] | None,
) -> _RuntimeSpec:
    if not isinstance(backend, str) or backend.lower() != "cuda":
        raise RuntimeInstallError(
            "Fish Small's pinned native runtime supports backend='cuda' only. "
            "Select CUDA explicitly; the runtime will not fall back to CPU."
        )
    capability = _validate_capability(device_capability)
    machine = platform.machine().lower().replace("-", "_")
    if sys.platform == "win32":
        if machine not in {"amd64", "x86_64", "x64"}:
            raise RuntimeInstallError(
                f"Fish Small's pinned CUDA runtime does not support Windows {machine or 'unknown architecture'}; "
                "Windows x64 is required."
            )
        # Blackwell reports SM 10.x or newer. CUDA 13.3 is the pinned
        # Blackwell package; earlier supported NVIDIA GPUs use CUDA 12.4.
        return _WIN_CUDA_13_3 if capability and capability[0] >= 10 else _WIN_CUDA_12_4
    if sys.platform.startswith("linux"):
        if machine not in {"x86_64", "amd64", "x64"}:
            raise RuntimeInstallError(
                f"Fish Small's pinned CUDA runtime does not support Linux {machine or 'unknown architecture'}; "
                "Linux x86_64 is required."
            )
        return _LINUX_CUDA_12_8
    raise RuntimeInstallError(
        f"Fish Small's pinned CUDA runtime does not support {sys.platform!r}; "
        "use Windows x64 or Linux x86_64 with an NVIDIA GPU."
    )


def _runtime_base_dir() -> Path:
    app_root = Path(__file__).resolve().parents[1]
    cache_root = app_root / ".cache"
    audiocpp_root = cache_root / "audiocpp"
    for path in (cache_root, audiocpp_root):
        if _is_link_or_reparse(path):
            raise RuntimeInstallError(f"Refusing to use a symlinked Fish Small runtime cache: {path}")
        path.mkdir(parents=True, exist_ok=True)
        if _is_link_or_reparse(path):
            raise RuntimeInstallError(f"Refusing to use a symlinked Fish Small runtime cache: {path}")
    return audiocpp_root / RELEASE


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & reparse_flag)


def _check_cancelled(check_cancelled: CancellationCallback | None) -> None:
    if check_cancelled is not None and check_cancelled():
        raise RuntimeInstallCancelled("Fish Small runtime installation was cancelled")


def _report_progress(
    progress: ProgressCallback | None,
    stage: str,
    completed: int,
    total: int | None,
) -> None:
    if progress is not None:
        progress(stage, completed, total)


def _thread_lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


def _lock_is_busy(error: OSError) -> bool:
    if os.name == "nt":
        return error.errno in {errno.EACCES, errno.EDEADLK} or getattr(error, "winerror", None) in {33, 36}
    return error.errno in {errno.EACCES, errno.EAGAIN}


@contextmanager
def _runtime_install_lock(
    lock_path: Path,
    check_cancelled: CancellationCallback | None,
) -> Iterator[None]:
    thread_lock = _thread_lock_for(lock_path)
    while not thread_lock.acquire(timeout=0.1):
        _check_cancelled(check_cancelled)
    handle = None
    process_locked = False
    try:
        if _is_link_or_reparse(lock_path):
            raise RuntimeInstallError(f"Refusing to use a symlinked Fish Small install lock: {lock_path}")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        handle = os.fdopen(descriptor, "r+b")
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
        while True:
            _check_cancelled(check_cancelled)
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                process_locked = True
                break
            except OSError as error:
                if not _lock_is_busy(error):
                    raise
                time.sleep(0.1)
        yield
    finally:
        try:
            if handle is not None and process_locked:
                try:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    # Closing the descriptor also releases a process lock.
                    pass
        finally:
            if handle is not None:
                handle.close()
            thread_lock.release()


def _staging_token(variant: str) -> bytes:
    return f"maestro-fish-small-staging-v1\n{RELEASE}\n{variant}\n".encode("utf-8")


def _owned_file_matches(path: Path, expected: bytes) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(info.st_mode) and not _is_link_or_reparse(path) and path.read_bytes() == expected


def _remove_owned_tree(path: Path, owner_content: bytes | None = None) -> bool:
    """Remove a helper-owned directory without following links."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return True
    if _is_link_or_reparse(path) or not stat.S_ISDIR(info.st_mode):
        return False
    if owner_content is not None and not _owned_file_matches(path / _STAGING_OWNER_FILE, owner_content):
        return False
    shutil.rmtree(path)
    return True


def _clean_stale_staging(parent: Path, variant: str) -> None:
    prefix = f".fish-small-runtime-{variant}-staging-"
    token = _staging_token(variant)
    for child in parent.iterdir():
        if not child.name.startswith(prefix) or _is_link_or_reparse(child):
            continue
        if child.is_dir():
            _remove_owned_tree(child, token)


def _download_asset(
    asset: _Asset,
    destination: Path,
    *,
    check_cancelled: CancellationCallback | None,
    progress: ProgressCallback | None,
) -> None:
    if Path(asset.name).name != asset.name or asset.name in {"", ".", ".."}:
        raise RuntimeInstallError(f"Invalid pinned audio.cpp asset filename: {asset.name!r}")
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "Maestro-FishSmall-runtime"},
    )
    digest = hashlib.sha256()
    downloaded = 0
    try:
        _check_cancelled(check_cancelled)
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with destination.open("xb") as output:
                _report_progress(progress, f"download:{asset.name}", 0, asset.size)
                while True:
                    _check_cancelled(check_cancelled)
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if downloaded > asset.size:
                        raise RuntimeIntegrityError(
                            f"Pinned audio.cpp asset {asset.name} exceeded its published size"
                        )
                    _report_progress(progress, f"download:{asset.name}", downloaded, asset.size)
                output.flush()
                os.fsync(output.fileno())
    except RuntimeInstallCancelled:
        destination.unlink(missing_ok=True)
        raise
    except Exception as error:
        destination.unlink(missing_ok=True)
        if isinstance(error, RuntimeInstallError):
            raise
        raise RuntimeInstallError(f"Could not download pinned audio.cpp asset {asset.name}: {error}") from error

    actual = digest.hexdigest()
    if downloaded != asset.size or actual.lower() != asset.sha256.lower():
        destination.unlink(missing_ok=True)
        raise RuntimeIntegrityError(
            f"Pinned audio.cpp asset {asset.name} failed verification "
            f"(size {downloaded}/{asset.size}, SHA-256 {actual})"
        )


def _safe_member_path(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise UnsafeArchiveError("Archive contains an empty or invalid member name")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//") or re.match(r"^[A-Za-z]:", normalized):
        raise UnsafeArchiveError(f"Archive contains an absolute path: {name!r}")
    parts = normalized.split("/")
    clean: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == ".." or ":" in part:
            raise UnsafeArchiveError(f"Archive path escapes its destination: {name!r}")
        if part.endswith((".", " ")):
            raise UnsafeArchiveError(f"Archive path is not portable: {name!r}")
        clean.append(part)
    return PurePosixPath(*clean).as_posix() if clean else ""


def _claim_archive_path(seen: dict[str, str], relative: str, kind: str) -> bool:
    if not relative:
        if kind == "directory":
            return False
        raise UnsafeArchiveError("Archive contains a file with no name")
    key = relative.casefold() if os.name == "nt" else relative
    components = relative.split("/")
    for index in range(1, len(components)):
        ancestor = "/".join(components[:index])
        if seen.get(ancestor.casefold() if os.name == "nt" else ancestor) == "file":
            raise UnsafeArchiveError(f"Archive path conflicts with a file: {relative!r}")
    existing = seen.get(key)
    if existing is not None:
        if existing == kind == "directory":
            return False
        raise UnsafeArchiveError(f"Archive contains a duplicate path: {relative!r}")
    if kind == "file" and any(
        candidate.startswith(key + ("/" if os.name != "nt" else "/"))
        for candidate in seen
    ):
        raise UnsafeArchiveError(f"Archive file conflicts with an existing directory: {relative!r}")
    seen[key] = kind
    return True


def _destination_for_member(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    destination = root.joinpath(*relative.split("/"))
    resolved = destination.resolve(strict=False)
    try:
        if os.path.commonpath((os.fspath(root_resolved), os.fspath(resolved))) != os.fspath(root_resolved):
            raise UnsafeArchiveError(f"Archive member escaped its extraction root: {relative!r}")
    except ValueError as error:
        raise UnsafeArchiveError(f"Archive member escaped its extraction root: {relative!r}") from error
    return destination


def _copy_member(source, destination: Path, expected_size: int, progress, stage: str, done: int, total: int,
                 check_cancelled: CancellationCallback | None) -> int:
    written = 0
    with destination.open("xb") as output:
        while True:
            _check_cancelled(check_cancelled)
            chunk = source.read(_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            written += len(chunk)
            _report_progress(progress, stage, done + written, total)
        if written != expected_size:
            raise RuntimeIntegrityError(
                f"Archive member {destination.name} was truncated ({written}/{expected_size} bytes)"
            )
    return written


def _extract_zip(
    archive_path: Path,
    destination_root: Path,
    *,
    seen: dict[str, str],
    stage: str,
    check_cancelled: CancellationCallback | None,
    progress: ProgressCallback | None,
) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            total = sum(member.file_size for member in members if not member.is_dir())
            completed = 0
            _report_progress(progress, stage, 0, total)
            for member in members:
                _check_cancelled(check_cancelled)
                mode = member.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                is_directory = member.is_dir() or file_type == stat.S_IFDIR
                relative = _safe_member_path(member.filename)
                if file_type == stat.S_IFLNK or file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise UnsafeArchiveError(f"Archive contains an unsupported file type: {member.filename!r}")
                kind = "directory" if is_directory else "file"
                if not _claim_archive_path(seen, relative, kind):
                    continue
                if not relative:
                    continue
                target = _destination_for_member(destination_root, relative)
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if _is_link_or_reparse(target.parent):
                    raise UnsafeArchiveError(f"Archive path traverses a link: {relative!r}")
                with archive.open(member, "r") as source:
                    completed += _copy_member(
                        source, target, member.file_size, progress, stage, completed, total, check_cancelled
                    )
            _report_progress(progress, stage, total, total)
    except (UnsafeArchiveError, RuntimeInstallCancelled, RuntimeIntegrityError):
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise RuntimeInstallError(f"Could not safely extract audio.cpp ZIP {archive_path.name}: {error}") from error


def _extract_tar_gz(
    archive_path: Path,
    destination_root: Path,
    *,
    seen: dict[str, str],
    stage: str,
    check_cancelled: CancellationCallback | None,
    progress: ProgressCallback | None,
) -> None:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            regular_files = [member for member in members if member.isfile()]
            total = sum(member.size for member in regular_files)
            completed = 0
            _report_progress(progress, stage, 0, total)
            for member in members:
                _check_cancelled(check_cancelled)
                relative = _safe_member_path(member.name)
                if member.issym() or member.islnk():
                    raise UnsafeArchiveError(f"Archive contains a link: {member.name!r}")
                if member.isdir():
                    kind = "directory"
                elif member.isfile():
                    kind = "file"
                else:
                    raise UnsafeArchiveError(f"Archive contains a special file: {member.name!r}")
                if not _claim_archive_path(seen, relative, kind):
                    continue
                if not relative:
                    continue
                target = _destination_for_member(destination_root, relative)
                if kind == "directory":
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if _is_link_or_reparse(target.parent):
                    raise UnsafeArchiveError(f"Archive path traverses a link: {relative!r}")
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeIntegrityError(f"Archive member could not be read: {member.name!r}")
                with source:
                    completed += _copy_member(
                        source, target, member.size, progress, stage, completed, total, check_cancelled
                    )
                os.chmod(target, member.mode & 0o777)
            _report_progress(progress, stage, total, total)
    except (UnsafeArchiveError, RuntimeInstallCancelled, RuntimeIntegrityError):
        raise
    except (OSError, tarfile.TarError, EOFError) as error:
        raise RuntimeInstallError(f"Could not safely extract audio.cpp tar archive {archive_path.name}: {error}") from error


def _extract_archive(
    archive_path: Path,
    destination_root: Path,
    archive_format: str,
    *,
    seen: dict[str, str],
    stage: str,
    check_cancelled: CancellationCallback | None,
    progress: ProgressCallback | None,
) -> None:
    if archive_format == "zip":
        _extract_zip(
            archive_path,
            destination_root,
            seen=seen,
            stage=stage,
            check_cancelled=check_cancelled,
            progress=progress,
        )
    elif archive_format == "tar.gz":
        _extract_tar_gz(
            archive_path,
            destination_root,
            seen=seen,
            stage=stage,
            check_cancelled=check_cancelled,
            progress=progress,
        )
    else:
        raise RuntimeInstallError(f"Unsupported pinned archive format: {archive_format!r}")


def _tree_files(root: Path) -> dict[str, os.stat_result]:
    files: dict[str, os.stat_result] = {}
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for directory in tuple(directories):
            path = current_path / directory
            info = os.lstat(path)
            if _is_link_or_reparse(path) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeIntegrityError(f"Installed Fish Small runtime contains a link or non-directory: {path}")
        for filename in filenames:
            path = current_path / filename
            info = os.lstat(path)
            if _is_link_or_reparse(path) or not stat.S_ISREG(info.st_mode):
                raise RuntimeIntegrityError(f"Installed Fish Small runtime contains a link or special file: {path}")
            relative = path.relative_to(root).as_posix()
            files[relative] = info
    return files


def _hash_file(
    path: Path,
    check_cancelled: CancellationCallback | None = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            _check_cancelled(check_cancelled)
            chunk = source.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(
    root: Path,
    *,
    check_cancelled: CancellationCallback | None,
    progress: ProgressCallback | None,
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    files = [
        (relative, info)
        for relative, info in sorted(_tree_files(root).items())
        if relative != _MARKER_FILE
    ]
    _report_progress(progress, "verify", 0, len(files))
    for index, (relative, info) in enumerate(files, start=1):
        _check_cancelled(check_cancelled)
        path = root.joinpath(*relative.split("/"))
        inventory.append({
            "path": relative,
            "size": info.st_size,
            "sha256": _hash_file(path, check_cancelled),
        })
        _report_progress(progress, "verify", index, len(files))
    return inventory


def _find_server(root: Path, spec: _RuntimeSpec, files: dict[str, os.stat_result]) -> Path:
    candidates = [
        root.joinpath(*relative.split("/"))
        for relative in files
        if PurePosixPath(relative).name.casefold() == spec.server_name.casefold()
    ]
    if len(candidates) != 1:
        raise RuntimeDependencyError(
            f"The pinned audio.cpp package must contain exactly one {spec.server_name}; "
            f"found {len(candidates)}."
        )
    executable = candidates[0]
    if spec.platform_name == "linux" and os.name != "nt" and not (os.stat(executable).st_mode & 0o111):
        raise RuntimeDependencyError(f"The pinned audio.cpp server is not executable: {executable}")
    return executable


def _validate_payload(root: Path, spec: _RuntimeSpec) -> Path:
    files = _tree_files(root)
    folded_names = {PurePosixPath(relative).name.casefold() for relative in files}
    missing = [name for name in spec.required_files if name.casefold() not in folded_names]
    if missing:
        raise RuntimeDependencyError(
            "The pinned audio.cpp CUDA package is incomplete; missing required files: "
            + ", ".join(missing)
        )
    if not any(PurePosixPath(relative).name.casefold() in {"license", "license.txt", "copying"} for relative in files):
        raise RuntimeDependencyError("The pinned audio.cpp package did not include its upstream license file")
    return _find_server(root, spec, files)


def _environment_for(root: Path, platform_name: str) -> dict[str, str]:
    directories: set[str] = set()
    for current, _, filenames in os.walk(root, topdown=True, followlinks=False):
        if platform_name == "windows":
            has_runtime = any(filename.lower().endswith(".dll") for filename in filenames)
        else:
            has_runtime = any(".so" in filename.lower() for filename in filenames)
        if has_runtime:
            directories.add(os.path.abspath(current))
    directories.add(os.path.abspath(root))
    ordered = [os.fspath(root)] + sorted(path for path in directories if os.path.normcase(path) != os.path.normcase(os.fspath(root)))
    variable = "PATH" if platform_name == "windows" else "LD_LIBRARY_PATH"
    existing = os.environ.get(variable, "")
    if existing:
        ordered.append(existing)
    return {variable: os.pathsep.join(ordered)}


def _runtime_info(root: Path, executable: Path, spec: _RuntimeSpec) -> RuntimeInfo:
    return RuntimeInfo(
        executable=executable,
        root=root,
        environment=_environment_for(root, spec.platform_name),
        backend="cuda",
        release=RELEASE,
    )


def _expected_assets(spec: _RuntimeSpec) -> list[dict[str, object]]:
    return [
        {"name": asset.name, "size": asset.size, "sha256": asset.sha256}
        for asset in spec.assets
    ]


def _stat_fingerprint(root: Path, files: dict[str, os.stat_result]) -> tuple[tuple[object, ...], ...]:
    result: list[tuple[object, ...]] = []
    for relative, info in sorted(files.items()):
        result.append((relative, info.st_size, info.st_mtime_ns, getattr(info, "st_ino", 0)))
    marker = root / _MARKER_FILE
    marker_info = os.stat(marker)
    result.append((_MARKER_FILE, marker_info.st_size, marker_info.st_mtime_ns, getattr(marker_info, "st_ino", 0)))
    return tuple(result)


def _verify_existing_install(
    root: Path,
    spec: _RuntimeSpec,
    *,
    check_cancelled: CancellationCallback | None,
    progress: ProgressCallback | None,
) -> RuntimeInfo | None:
    try:
        root_info = os.lstat(root)
    except FileNotFoundError:
        return None
    if _is_link_or_reparse(root) or not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeIntegrityError(f"Refusing to use a non-directory or symlinked runtime root: {root}")
    if not _owned_file_matches(root / _OWNER_FILE, _OWNER_CONTENT):
        raise RuntimeInstallError(
            f"Fish Small runtime path already exists but is not owned by this installer: {root}"
        )
    marker_path = root / _MARKER_FILE
    if _is_link_or_reparse(marker_path) or not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict) or any(
        marker.get(key) != expected
        for key, expected in (
            ("schema", 1),
            ("release", RELEASE),
            ("variant", spec.variant),
            ("backend", "cuda"),
            ("assets", _expected_assets(spec)),
        )
    ):
        return None
    recorded_files = marker.get("files")
    if not isinstance(recorded_files, list):
        return None
    expected_file_data: dict[str, tuple[int, str]] = {}
    try:
        for item in recorded_files:
            if not isinstance(item, dict):
                return None
            relative = item.get("path")
            size = item.get("size")
            digest = item.get("sha256")
            safe_relative = _safe_member_path(relative)
            if (
                not safe_relative
                or safe_relative != relative
                or safe_relative == _MARKER_FILE
                or not isinstance(size, int)
                or size < 0
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                return None
            if relative in expected_file_data:
                return None
            expected_file_data[relative] = (size, digest)
    except (UnsafeArchiveError, TypeError):
        return None
    executable_relative = marker.get("executable")
    if not isinstance(executable_relative, str):
        return None

    try:
        files = _tree_files(root)
    except RuntimeIntegrityError:
        return None
    actual_files = {relative: info for relative, info in files.items() if relative != _MARKER_FILE}
    if set(actual_files) != set(expected_file_data):
        return None
    for relative, info in actual_files.items():
        if info.st_size != expected_file_data[relative][0]:
            return None
    fingerprint = _stat_fingerprint(root, files)
    cache_key = os.path.normcase(os.path.abspath(root))
    cached = _VERIFIED_CACHE.get(cache_key)
    if cached is None or cached[0] != fingerprint:
        _report_progress(progress, "verify", 0, len(actual_files))
        for index, (relative, info) in enumerate(sorted(actual_files.items()), start=1):
            _check_cancelled(check_cancelled)
            path = root.joinpath(*relative.split("/"))
            if _hash_file(path, check_cancelled) != expected_file_data[relative][1]:
                _VERIFIED_CACHE.pop(cache_key, None)
                return None
            _report_progress(progress, "verify", index, len(actual_files))
        _VERIFIED_CACHE[cache_key] = (fingerprint, True)

    try:
        server = _find_server(root, spec, actual_files)
    except RuntimeDependencyError:
        return None
    if Path(executable_relative).as_posix() != server.relative_to(root).as_posix():
        return None
    try:
        _validate_payload(root, spec)
    except RuntimeDependencyError:
        return None
    return _runtime_info(root, server, spec)


def _write_marker(
    root: Path,
    spec: _RuntimeSpec,
    executable: Path,
    *,
    check_cancelled: CancellationCallback | None,
    progress: ProgressCallback | None,
) -> None:
    owner_path = root / _OWNER_FILE
    marker_path = root / _MARKER_FILE
    temporary = root / f"{_MARKER_FILE}.tmp"
    if any(path.exists() or path.is_symlink() for path in (owner_path, marker_path, temporary)):
        raise RuntimeIntegrityError("Upstream archive conflicts with an installer-owned marker path")
    owner_path.write_bytes(_OWNER_CONTENT)
    inventory = _file_inventory(
        root,
        check_cancelled=check_cancelled,
        progress=progress,
    )
    marker = {
        "schema": 1,
        "release": RELEASE,
        "variant": spec.variant,
        "backend": "cuda",
        "assets": _expected_assets(spec),
        "executable": executable.relative_to(root).as_posix(),
        "files": inventory,
    }
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker_path)


def _assert_replaceable(root: Path) -> bool:
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        return False
    if _is_link_or_reparse(root) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeIntegrityError(f"Refusing to replace a non-directory or symlinked runtime path: {root}")
    if not _owned_file_matches(root / _OWNER_FILE, _OWNER_CONTENT):
        raise RuntimeInstallError(f"Refusing to replace an unowned runtime directory: {root}")
    return True


def _promote_staged_install(staging: Path, staged_root: Path, root: Path) -> None:
    backup = root.parent / f".fish-small-runtime-{root.name}-backup-{os.getpid()}-{time.time_ns()}"
    staged_cache_key = os.path.normcase(os.path.abspath(staged_root))
    root_cache_key = os.path.normcase(os.path.abspath(root))
    staged_verification = _VERIFIED_CACHE.pop(staged_cache_key, None)
    had_old_install = _assert_replaceable(root)
    if had_old_install:
        os.replace(root, backup)
    try:
        os.replace(staged_root, root)
    except Exception:
        if had_old_install and backup.exists() and not root.exists():
            os.replace(backup, root)
        raise
    if staged_verification is None:
        _VERIFIED_CACHE.pop(root_cache_key, None)
    else:
        _VERIFIED_CACHE[root_cache_key] = staged_verification
    if had_old_install:
        try:
            _remove_owned_tree(backup)
        except OSError:
            # The new runtime is already in place; a locked old backup can be
            # safely left for manual cleanup rather than invalidating it.
            pass


def ensure_runtime(
    *,
    backend: str = "cuda",
    device_capability: tuple[int, int] | None = None,
    check_cancelled: CancellationCallback | None = None,
    progress: ProgressCallback | None = None,
) -> RuntimeInfo:
    """Return the verified audio.cpp server, downloading its pinned CUDA runtime if needed.

    Windows x64 selects CUDA 13.3 for Blackwell (SM 10.x+) and CUDA 12.4 for
    earlier GPUs. Linux x86_64 uses the official CUDA 12.8 bundle. Other
    backends and platforms fail explicitly without a CPU fallback.
    """
    _check_cancelled(check_cancelled)
    spec = _select_runtime_spec(backend, device_capability)
    parent = _runtime_base_dir()
    parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(parent):
        raise RuntimeIntegrityError(f"Refusing to use a symlinked Fish Small release directory: {parent}")
    root = parent / spec.variant
    lock_path = parent / f".{spec.variant}.install.lock"
    with _runtime_install_lock(lock_path, check_cancelled):
        _check_cancelled(check_cancelled)
        cached_info = _verify_existing_install(
            root,
            spec,
            check_cancelled=check_cancelled,
            progress=progress,
        )
        if cached_info is not None:
            return cached_info
        _clean_stale_staging(parent, spec.variant)
        staging = Path(tempfile.mkdtemp(prefix=f".fish-small-runtime-{spec.variant}-staging-", dir=parent))
        staging_owner = staging / _STAGING_OWNER_FILE
        try:
            staging_owner.write_bytes(_staging_token(spec.variant))
            downloads = staging / "downloads"
            staged_root = staging / "payload"
            downloads.mkdir()
            staged_root.mkdir()
            seen: dict[str, str] = {}
            for index, asset in enumerate(spec.assets):
                _check_cancelled(check_cancelled)
                archive_path = downloads / f"{index}-{asset.name}"
                _download_asset(
                    asset,
                    archive_path,
                    check_cancelled=check_cancelled,
                    progress=progress,
                )
                _extract_archive(
                    archive_path,
                    staged_root,
                    asset.archive_format,
                    seen=seen,
                    stage=f"extract:{asset.name}",
                    check_cancelled=check_cancelled,
                    progress=progress,
                )
                archive_path.unlink()
            _check_cancelled(check_cancelled)
            executable = _validate_payload(staged_root, spec)
            _write_marker(
                staged_root,
                spec,
                executable,
                check_cancelled=check_cancelled,
                progress=progress,
            )
            # Ensure the newly written marker and every extracted file agree
            # before replacing a previously installed runtime.
            staged_info = _verify_existing_install(
                staged_root,
                spec,
                check_cancelled=check_cancelled,
                progress=progress,
            )
            if staged_info is None:
                raise RuntimeIntegrityError("The staged Fish Small runtime failed its own file verification")
            _check_cancelled(check_cancelled)
            try:
                _promote_staged_install(staging, staged_root, root)
            except OSError as error:
                raise RuntimeInstallError(
                    f"Could not atomically install the pinned Fish Small runtime at {root}: {error}"
                ) from error
        finally:
            _remove_owned_tree(staging, _staging_token(spec.variant))
        installed = _verify_existing_install(
            root,
            spec,
            check_cancelled=check_cancelled,
            progress=progress,
        )
        if installed is None:
            raise RuntimeIntegrityError("The installed Fish Small runtime failed post-install verification")
        return installed
