"""Verified downloads for named model checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_CIVITAI_HOSTS = {"civitai.com", "civitai.red"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024


class CheckpointDownloadError(ValueError):
    """A named checkpoint could not be downloaded and verified."""


def _safe_filename(filename: object) -> str:
    if not isinstance(filename, str) or not filename:
        raise CheckpointDownloadError("Checkpoint filename must be a plain filename.")
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise CheckpointDownloadError("Checkpoint filename must not contain a path.")
    if any(ord(char) < 32 or char in '<>:"|?*' for char in filename):
        raise CheckpointDownloadError("Checkpoint filename contains unsupported characters.")
    if filename.endswith((".", " ")):
        raise CheckpointDownloadError("Checkpoint filename must not end with a dot or space.")
    device_name = filename.split(".", 1)[0].rstrip(" .").upper()
    if device_name in _WINDOWS_RESERVED_NAMES:
        raise CheckpointDownloadError("Checkpoint filename is reserved by Windows.")
    return filename


def find_named_checkpoint_source(model_def: object, model_filename: object) -> dict | None:
    """Return only the source registered for this exact clean model filename."""
    if not isinstance(model_def, dict):
        return None
    try:
        filename = _safe_filename(model_filename)
    except CheckpointDownloadError:
        return None
    sources = model_def.get("download_sources")
    if not isinstance(sources, dict):
        return None
    source = sources.get(filename)
    return source if isinstance(source, dict) else None


def _validate_source(source: object) -> tuple[str, str, int, str]:
    if not isinstance(source, dict):
        raise CheckpointDownloadError("Checkpoint download source is missing or invalid.")
    url = source.get("url")
    expected_sha256 = source.get("sha256")
    expected_size = source.get("size_bytes")
    if not isinstance(url, str) or not url or any(char in url for char in "\r\n\x00"):
        raise CheckpointDownloadError("Checkpoint download source has an invalid URL.")
    try:
        parsed_url = urlsplit(url)
        hostname = (parsed_url.hostname or "").lower().rstrip(".")
        # Accessing .port also validates malformed port syntax.
        _ = parsed_url.port
    except ValueError:
        raise CheckpointDownloadError("Checkpoint download source has an invalid URL.") from None
    if parsed_url.scheme.lower() != "https" or not hostname or parsed_url.username or parsed_url.password:
        raise CheckpointDownloadError("Checkpoint source must use a valid HTTPS URL.")
    if not isinstance(expected_sha256, str) or not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise CheckpointDownloadError("Checkpoint source must include a lowercase SHA-256 digest.")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise CheckpointDownloadError("Checkpoint source must include a positive byte size.")
    return url, expected_sha256, expected_size, hostname


def _authenticated_request(url: str, hostname: str, api_key: str | None) -> tuple[str, dict[str, str]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
    }
    if not isinstance(api_key, str) or not api_key or hostname not in _CIVITAI_HOSTS:
        return url, headers

    parts = urlsplit(url)
    query_items = parse_qsl(parts.query, keep_blank_values=True)
    # A source URL may already carry a Civitai per-download token. Preserve it
    # and avoid adding a second token or sending a different key in a header.
    if any(key.casefold() == "token" for key, _ in query_items):
        return url, headers

    query_items.append(("token", api_key))
    headers["Authorization"] = f"Bearer {api_key}"
    return urlunsplit(parts._replace(query=urlencode(query_items))), headers


def _validate_safetensors_header(path: Path, expected_size: int) -> None:
    if expected_size < 8:
        raise CheckpointDownloadError("Downloaded file is too small to be a safetensors checkpoint.")
    with path.open("rb") as handle:
        encoded_length = handle.read(8)
        if len(encoded_length) != 8:
            raise CheckpointDownloadError("Downloaded file has no safetensors header.")
        header_size = int.from_bytes(encoded_length, "little", signed=False)
        if header_size <= 0 or header_size > _MAX_SAFETENSORS_HEADER_BYTES:
            raise CheckpointDownloadError("Downloaded file has an invalid safetensors header size.")
        if 8 + header_size > expected_size:
            raise CheckpointDownloadError("Downloaded file has a truncated safetensors header.")
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise CheckpointDownloadError("Downloaded file does not have a valid safetensors header.") from None

    if not isinstance(header, dict):
        raise CheckpointDownloadError("Downloaded file does not have a valid safetensors header.")
    metadata = header.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items())
    ):
        raise CheckpointDownloadError("Downloaded file has invalid safetensors metadata.")

    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    payload_size = expected_size - 8 - header_size
    if not tensors:
        raise CheckpointDownloadError("Downloaded safetensors checkpoint contains no tensors.")
    data_ranges = []
    for name, descriptor in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise CheckpointDownloadError("Downloaded file has an invalid safetensors tensor entry.")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise CheckpointDownloadError("Downloaded file has an invalid safetensors tensor dtype.")
        if not isinstance(shape, list) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        ):
            raise CheckpointDownloadError("Downloaded file has an invalid safetensors tensor shape.")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload_size
        ):
            raise CheckpointDownloadError("Downloaded file has invalid safetensors data offsets.")
        data_ranges.append((offsets[0], offsets[1]))

    data_cursor = 0
    for start, end in sorted(data_ranges):
        if start != data_cursor:
            raise CheckpointDownloadError("Downloaded file has non-contiguous safetensors data offsets.")
        data_cursor = end
    if data_cursor != payload_size:
        raise CheckpointDownloadError("Downloaded file has non-contiguous safetensors data offsets.")


def download_named_checkpoint(
    source: object,
    target: str | os.PathLike[str],
    *,
    civitai_api_key: str | None = None,
    progress_hook=None,
    validate_checkpoint=None,
    file_format="safetensors",
    before_publish=None,
) -> str:
    """Download a named checkpoint to ``target`` after validating its bytes.

    The target filename and directory come only from the caller. A temporary
    file is created alongside the target and the final rename is atomic.
    """
    url, expected_sha256, expected_size, hostname = _validate_source(source)
    if file_format not in {"safetensors", "gguf"} or (file_format == "gguf" and not callable(validate_checkpoint)):
        raise CheckpointDownloadError("GGUF downloads require a verified architecture-specific importer.")
    try:
        target_path = Path(os.fspath(target))
    except (TypeError, ValueError):
        raise CheckpointDownloadError("Checkpoint target path is invalid.") from None
    if ".." in target_path.parts:
        raise CheckpointDownloadError("Checkpoint target path must not contain parent traversal.")
    _safe_filename(target_path.name)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # requests is already part of the runtime, but keep this module safe to
    # import in CPU-only tooling and model metadata tests.
    import requests
    from tqdm import tqdm
    try:
        from services.download_control import DownloadCancelled, check_download_cancelled, download_publication
    except ModuleNotFoundError:
        from app.services.download_control import DownloadCancelled, check_download_cancelled, download_publication

    request_url, headers = _authenticated_request(url, hostname, civitai_api_key)
    temp_path: Path | None = None
    response = None
    progress_bar = None
    try:
        check_download_cancelled()
        try:
            response = requests.get(
                request_url,
                headers=headers,
                stream=True,
                timeout=30,
                allow_redirects=True,
            )
        except Exception as exc:
            check_download_cancelled()
            raise CheckpointDownloadError(
                f"Checkpoint download failed because of a network error ({type(exc).__name__})."
            ) from None

        if response.status_code in (401, 403) and hostname in _CIVITAI_HOSTS:
            raise CheckpointDownloadError(
                "Civitai rejected this checkpoint download. Add or verify the key in "
                "Settings > Services > Civitai API key, then confirm your Civitai account "
                "has authorized access to the official creator file, including any required "
                "terms acceptance or download gate."
            )
        if response.status_code >= 400:
            raise CheckpointDownloadError(
                f"Checkpoint download failed with HTTP {response.status_code} from {hostname}."
            )

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target_path.name}.",
            suffix=".download",
            dir=str(target_path.parent),
        )
        temp_path = Path(temp_name)
        digest = hashlib.sha256()
        bytes_written = 0
        with os.fdopen(fd, "wb") as handle:
            # Maestro observes byte-based tqdm bars for its download banner.
            # Use the clean local name, never a credential-bearing URL.
            progress_bar = tqdm(total=expected_size, desc=target_path.name,
                                unit="B", unit_scale=True, leave=False)
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                check_download_cancelled()
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > expected_size:
                    raise CheckpointDownloadError("Checkpoint download exceeded its expected byte size.")
                handle.write(chunk)
                digest.update(chunk)
                progress_bar.update(len(chunk))
                if progress_hook is not None:
                    # create_progress_hook(filename) computes block_num *
                    # block_size, so pass one block whose size is cumulative.
                    progress_hook(1, bytes_written, expected_size)

        check_download_cancelled()
        if bytes_written != expected_size:
            raise CheckpointDownloadError(
                f"Checkpoint download size mismatch: received {bytes_written} of {expected_size} bytes."
            )
        if digest.hexdigest() != expected_sha256:
            raise CheckpointDownloadError("Checkpoint download SHA-256 verification failed.")
        if file_format == "gguf":
            from services.h3_gguf_import import read_gguf_index
            with temp_path.open("rb") as handle:
                read_gguf_index(handle, expected_size)
        else:
            _validate_safetensors_header(temp_path, expected_size)
        if validate_checkpoint is not None:
            validate_checkpoint(str(temp_path))
        with download_publication():
            if before_publish is not None:
                before_publish()
            check_download_cancelled()
            os.replace(temp_path, target_path)
        temp_path = None
        return str(target_path)
    except (CheckpointDownloadError, DownloadCancelled):
        raise
    except Exception as exc:
        check_download_cancelled()
        # requests exceptions can include the original URL, whose query may
        # contain a Civitai token. Keep details local to the exception type.
        raise CheckpointDownloadError(
            f"Checkpoint download failed ({type(exc).__name__})."
        ) from None
    finally:
        if progress_bar is not None:
            try:
                progress_bar.close()
            except Exception:
                pass
        if response is not None:
            close = getattr(response, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
