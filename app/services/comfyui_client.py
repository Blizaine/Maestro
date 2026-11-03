"""Small, bounded HTTP client for a separately running ComfyUI server.

This module intentionally knows only ComfyUI's HTTP API. It does not import
ComfyUI, Torch, or any of the server's Python internals.
"""

from __future__ import annotations

import ipaddress
import json
import math
import mimetypes
import re
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx


MAX_OUTPUT_BYTES = 256 * 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024
_SAFE_CATEGORY = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class ComfyError(RuntimeError):
    """Base error for a sanitized ComfyUI API failure."""


class ComfyHTTPError(ComfyError):
    """An HTTP response was received, so the server rejected or failed it."""

    def __init__(self, status_code: int, *, ambiguous: bool = False):
        self.status_code = int(status_code)
        self.ambiguous = bool(ambiguous)
        suffix = " The server may have accepted the request; reconcile before retrying." if self.ambiguous else ""
        super().__init__(f"ComfyUI returned HTTP {self.status_code}.{suffix}")


class ComfyValidationError(ComfyHTTPError):
    """ComfyUI returned a client error for a request."""


class ComfyServerError(ComfyHTTPError):
    """ComfyUI returned a server error for a request."""


class ComfyTransportError(ComfyError):
    """A request failed before a usable HTTP response arrived.

    ``ambiguous`` is true for mutations whose server-side outcome may be
    unknown. Callers must reconcile those requests through queue/history and
    must not automatically submit them again.
    """

    def __init__(self, operation: str, *, ambiguous: bool):
        self.operation = operation
        self.ambiguous = bool(ambiguous)
        if ambiguous and operation == "submit":
            suffix = " The server may have accepted it; reconcile by submission ID before retrying."
        elif ambiguous:
            suffix = " The server may have accepted it; check server state before retrying."
        else:
            suffix = ""
        super().__init__(f"ComfyUI {operation} request failed.{suffix}")


class ComfyResponseError(ComfyError):
    """The server response could not be safely consumed."""

    def __init__(self, message: str, *, ambiguous: bool = False):
        self.ambiguous = bool(ambiguous)
        self.message = message
        if self.ambiguous:
            message = f"{message} The server may have accepted the request; reconcile before retrying."
        super().__init__(message)


def _validated_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("ComfyUI endpoint must be a non-empty URL.")
    try:
        parsed = urlsplit(endpoint.strip())
        port = parsed.port
    except ValueError:
        raise ValueError("ComfyUI endpoint has an invalid host or port.") from None
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("ComfyUI endpoint must use HTTP or HTTPS.")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("ComfyUI endpoint must not contain credentials.")
    if parsed.query or parsed.fragment or "?" in endpoint.strip() or "#" in endpoint.strip():
        raise ValueError("ComfyUI endpoint must not contain a query or fragment.")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("ComfyUI endpoint has an invalid port.")
    if parsed.scheme == "http":
        host = parsed.hostname.rstrip(".").lower()
        is_local = host == "localhost" or host.endswith(".localhost") or host.endswith(".local")
        if not is_local:
            try:
                address = ipaddress.ip_address(host.split("%", 1)[0])
                is_local = address.is_loopback or address.is_private or address.is_link_local
            except ValueError:
                is_local = False
        if not is_local:
            raise ValueError("Plain HTTP is allowed only for loopback or private LAN ComfyUI endpoints.")
    # Drop a trailing slash for stable route joining, while retaining a
    # reverse-proxy path prefix if the user configured one.
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class ComfyClient:
    """Synchronous HTTP transport for a remote ComfyUI process."""

    def __init__(self, endpoint: str, *, token: str | None = None, timeout: float = 15):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a positive finite number of seconds.")
        if not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError("timeout must be a positive finite number of seconds.")
        if token is not None and not isinstance(token, str):
            raise ValueError("token must be a string when provided.")
        self._base_url = _validated_endpoint(endpoint)
        self._token = token
        self._timeout = float(timeout)
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(
            timeout=self._timeout,
            trust_env=False,
            follow_redirects=False,
            verify=True,
            headers=headers,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ComfyClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _request(self, method: str, path: str, *, operation: str, ambiguous: bool = False, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, self._url(path), **kwargs)
        except httpx.ConnectError:
            raise ComfyTransportError(operation, ambiguous=False) from None
        except httpx.ConnectTimeout:
            raise ComfyTransportError(operation, ambiguous=False) from None
        except httpx.HTTPError:
            raise ComfyTransportError(operation, ambiguous=ambiguous) from None
        if response.status_code >= 400:
            if 400 <= response.status_code < 500:
                raise ComfyValidationError(response.status_code)
            raise ComfyServerError(response.status_code, ambiguous=ambiguous)
        if 300 <= response.status_code < 400:
            raise ComfyHTTPError(response.status_code, ambiguous=ambiguous)
        return response

    @staticmethod
    def _json(response: httpx.Response, *, operation: str) -> Any:
        if len(response.content) > _MAX_JSON_BYTES:
            raise ComfyResponseError(f"ComfyUI {operation} response exceeded the size limit.")
        try:
            return response.json()
        except (ValueError, UnicodeDecodeError):
            raise ComfyResponseError(f"ComfyUI returned invalid JSON for {operation}.") from None

    def _get_json(self, path: str, *, operation: str, params: dict[str, str] | None = None) -> Any:
        response = self._request("GET", path, operation=operation, params=params)
        return self._json(response, operation=operation)

    def _mutation_json(self, response: httpx.Response, *, operation: str, allow_empty: bool = False) -> dict[str, Any]:
        # Core /queue deletion and /free acknowledge with an empty 200 body.
        # Submission and upload still require their identifying JSON receipts.
        if allow_empty and not response.content:
            return {}
        try:
            payload = self._json(response, operation=operation)
        except ComfyResponseError as exc:
            raise ComfyResponseError(exc.message, ambiguous=True) from None
        if not isinstance(payload, dict):
            raise ComfyResponseError(
                f"ComfyUI {operation} response was not an object.", ambiguous=True
            )
        return payload

    def object_info(self) -> dict[str, Any]:
        """Return ComfyUI's node schema map."""
        payload = self._get_json("/object_info", operation="object_info")
        if not isinstance(payload, dict):
            raise ComfyResponseError("ComfyUI object_info response was not an object.")
        return payload

    def models(self, category: str) -> list[Any]:
        """Return filenames advertised by ComfyUI for one model folder."""
        if not isinstance(category, str) or not _SAFE_CATEGORY.fullmatch(category):
            raise ValueError("model category must be a simple folder name.")
        payload = self._get_json(f"/models/{quote(category, safe='')}", operation="models")
        if not isinstance(payload, list):
            raise ComfyResponseError("ComfyUI models response was not a list.")
        return payload

    def system_stats(self) -> dict[str, Any]:
        """Return ComfyUI system statistics."""
        payload = self._get_json("/system_stats", operation="system_stats")
        if not isinstance(payload, dict):
            raise ComfyResponseError("ComfyUI system_stats response was not an object.")
        return payload

    def queue(self) -> dict[str, Any]:
        """Return the current queue snapshot."""
        payload = self._get_json("/queue", operation="queue")
        if not isinstance(payload, dict):
            raise ComfyResponseError("ComfyUI queue response was not an object.")
        return payload

    def history(self, prompt_id: str | None = None) -> dict[str, Any]:
        """Return all history, or history for one prompt ID."""
        if prompt_id is None:
            path = "/history"
        else:
            if not isinstance(prompt_id, str) or not prompt_id or len(prompt_id) > 256:
                raise ValueError("prompt_id must be a non-empty string up to 256 characters.")
            path = f"/history/{quote(prompt_id, safe='')}"
        payload = self._get_json(path, operation="history")
        if not isinstance(payload, dict):
            raise ComfyResponseError("ComfyUI history response was not an object.")
        return payload

    def submit(self, graph: dict[str, Any], *, client_id: str, submission_id: str) -> dict[str, Any]:
        """Queue one API-format prompt without retrying ambiguous failures."""
        if not isinstance(graph, dict):
            raise ValueError("graph must be an API-format workflow object.")
        if not isinstance(client_id, str) or not client_id or len(client_id) > 256:
            raise ValueError("client_id must be a non-empty string up to 256 characters.")
        if not isinstance(submission_id, str) or not submission_id or len(submission_id) > 256:
            raise ValueError("submission_id must be a non-empty string up to 256 characters.")
        body = {
            "prompt": graph,
            "client_id": client_id,
            "extra_data": {
                "maestro_submission_id": submission_id,
                "client_id": client_id,
            },
        }
        try:
            # Fail before the request if a caller bypassed workflow validation
            # and supplied a non-JSON value or a non-finite float.
            json.dumps(body, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("graph must contain only finite JSON values.") from None
        response = self._request(
            "POST", "/prompt", operation="submit", ambiguous=True, json=body
        )
        return self._mutation_json(response, operation="submit")

    def upload_image(self, data: bytes, filename: str) -> dict[str, Any]:
        """Upload image bytes to ComfyUI's input directory."""
        if not isinstance(data, bytes) or not data:
            raise ValueError("image data must be non-empty bytes.")
        if not isinstance(filename, str) or not filename or len(filename) > 255:
            raise ValueError("filename must be a non-empty name up to 255 characters.")
        if filename in {".", ".."} or "/" in filename or "\\" in filename or "\x00" in filename:
            raise ValueError("filename must be a simple file name without path separators.")
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        response = self._request(
            "POST",
            "/upload/image",
            operation="upload_image",
            ambiguous=True,
            files={"image": (filename, data, content_type)},
        )
        return self._mutation_json(response, operation="upload_image")

    def output(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
        """Download one ComfyUI file, enforcing a fixed maximum size."""
        if not isinstance(filename, str) or not filename or len(filename) > 1024:
            raise ValueError("filename must be a non-empty string up to 1024 characters.")
        if "\x00" in filename:
            raise ValueError("filename contains an invalid character.")
        if not isinstance(subfolder, str) or len(subfolder) > 1024 or "\x00" in subfolder:
            raise ValueError("subfolder must be a string up to 1024 characters.")
        if not isinstance(type, str) or type not in {"output", "input", "temp"}:
            raise ValueError("type must be output, input, or temp.")
        params = {"filename": filename, "subfolder": subfolder, "type": type}
        try:
            with self._http.stream("GET", self._url("/view"), params=params) as response:
                if response.status_code >= 400:
                    error_type = ComfyValidationError if response.status_code < 500 else ComfyServerError
                    raise error_type(response.status_code)
                if 300 <= response.status_code < 400:
                    raise ComfyHTTPError(response.status_code)
                size_header = response.headers.get("content-length")
                if size_header:
                    try:
                        if int(size_header) > MAX_OUTPUT_BYTES:
                            raise ComfyResponseError("ComfyUI output exceeded the size limit.")
                    except ValueError:
                        raise ComfyResponseError("ComfyUI output had an invalid size header.") from None
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > MAX_OUTPUT_BYTES:
                        raise ComfyResponseError("ComfyUI output exceeded the size limit.")
                    chunks.append(chunk)
                return b"".join(chunks)
        except ComfyError:
            raise
        except httpx.ConnectError:
            raise ComfyTransportError("output", ambiguous=False) from None
        except httpx.ConnectTimeout:
            raise ComfyTransportError("output", ambiguous=False) from None
        except httpx.HTTPError:
            raise ComfyTransportError("output", ambiguous=False) from None

    def delete_queued(self, prompt_id: str) -> dict[str, Any]:
        """Delete only the specified queued prompt; never interrupt current work."""
        if not isinstance(prompt_id, str) or not prompt_id or len(prompt_id) > 256:
            raise ValueError("prompt_id must be a non-empty string up to 256 characters.")
        response = self._request(
            "POST",
            "/queue",
            operation="delete_queued",
            ambiguous=True,
            json={"delete": [prompt_id]},
        )
        return self._mutation_json(response, operation="delete_queued", allow_empty=True)

    def release_memory(self) -> dict[str, Any]:
        """Explicitly ask ComfyUI to unload models and release cached memory.

        Callers must gate this operation to dedicated-server maintenance after
        confirming its queue is empty. This transport method never does so on
        its own and never interrupts a workflow.
        """
        response = self._request(
            "POST",
            "/free",
            operation="release_memory",
            ambiguous=True,
            json={"unload_models": True, "free_memory": True},
        )
        return self._mutation_json(response, operation="release_memory", allow_empty=True)


__all__ = [
    "MAX_OUTPUT_BYTES",
    "ComfyClient",
    "ComfyError",
    "ComfyHTTPError",
    "ComfyValidationError",
    "ComfyServerError",
    "ComfyTransportError",
    "ComfyResponseError",
]
