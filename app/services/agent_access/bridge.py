"""Authenticated, same-installation bridge; never a general-purpose URL proxy."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json as jsonlib
import mimetypes
import os
from pathlib import Path
import re
import math
from urllib.parse import quote, unquote, urlsplit
from email.message import Message
from collections.abc import AsyncIterable, Mapping, Sequence

import httpx

from .assets import AssetStore
from .auth import AuthError
from .boundary import redact
from .catalog import required_scope
from .context import credential_context, principal_context
from .http import scope_required
from .jobs import SubmissionJournal
from .state import process_alive
from .transfers import TransferStore


class BridgeError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class BackendClient:
    def __init__(self, runtime, *, client=None, assets=None, journal=None, transfers=None):
        self.runtime = runtime
        self.assets = assets or AssetStore(runtime.directory / "assets", runtime.app_root)
        self.journal = journal or SubmissionJournal(runtime.directory)
        self.transfers = transfers or TransferStore(
            runtime.directory / "transfers",
            origin=getattr(runtime, "origin", ""),
        )
        self.client = client or httpx.AsyncClient(
            trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(660, connect=5, pool=5, write=60),
        )
        self.discovery = None
        self.lifecycle_manager = None

    async def close(self):
        await self.client.aclose()

    def online(self) -> bool:
        backend = self.runtime.backend()
        return bool(backend.get("ready") and process_alive(backend.get("pid")))

    def target(self) -> str:
        record = self.runtime.backend()
        if not record.get("ready") or not process_alive(record.get("pid")):
            raise BridgeError(503, "Maestro is stopped or not ready. Inspect workstation status before starting it.")
        return record["url"]

    def catalog_revision(self):
        return self.runtime.backend().get("session_id")

    @staticmethod
    def headers() -> dict:
        token = credential_context.get()
        if not token or principal_context.get() is None:
            raise BridgeError(401, "Pair and authenticate this agent first")
        return {"Authorization": "Bearer " + token, "Accept-Encoding": "identity"}

    def _durable_job(self, principal, job_id):
        result = self.journal.get_job(job_id, principal.id)
        if result is None:
            raise BridgeError(404, "No owned durable job receipt was found")
        return result

    @staticmethod
    def _asset_reference(value: str) -> str:
        match = re.fullmatch(
            r"(?:asset|maestro-asset):([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
            value,
        )
        if match is None:
            raise BridgeError(400, "Multipart file fields must use an uploaded asset:<UUID> reference")
        return match.group(1).lower()

    def _asset_descriptor(self, principal_id: str, reference: str) -> tuple[dict, Path]:
        asset_id = self._asset_reference(reference)
        record = self.assets.status(principal_id, asset_id)
        if record.get("state") != "imported":
            raise BridgeError(409, "Uploaded file asset is not registered yet")
        path = self.assets.resolve(principal_id, reference)
        try:
            stat = path.stat()
        except OSError as error:
            raise BridgeError(404, "Uploaded file asset is unavailable") from error
        if not path.is_file() or path.is_symlink() or stat.st_size != record.get("total"):
            raise BridgeError(409, "Uploaded file asset changed after registration")
        return record, path

    def _fingerprint_tree(self, principal_id: str, value):
        if isinstance(value, str) and value.startswith(("asset:", "maestro-asset:")):
            record, _path = self._asset_descriptor(principal_id, value)
            return {
                "asset_sha256": record.get("sha256"),
                "filename": record.get("filename"),
                "size": record.get("total"),
            }
        if isinstance(value, Mapping):
            return {str(key): self._fingerprint_tree(principal_id, item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._fingerprint_tree(principal_id, item) for item in value]
        return value

    def _request_fingerprint(self, principal_id, method, path, query, payload, file_fields):
        normalized = self._fingerprint_tree(principal_id, payload)
        file_names = sorted(str(name) for name in file_fields)
        canonical = jsonlib.dumps(
            {
                "method": method,
                "path": path,
                "query": self._fingerprint_tree(principal_id, query),
                "body": normalized,
                "file_fields": file_names,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _multipart_parts(self, principal_id: str, payload, file_fields):
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise BridgeError(400, "Multipart request body must be an object of documented fields")
        file_names = set(file_fields or ())
        if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name) for name in file_names):
            raise BridgeError(400, "Multipart field names are invalid")
        unknown_asset = [
            key for key, value in payload.items()
            if key not in file_names and (
                isinstance(value, str) and value.startswith(("asset:", "maestro-asset:"))
                or isinstance(value, list) and any(isinstance(item, str) and item.startswith(("asset:", "maestro-asset:")) for item in value)
            )
        ]
        if unknown_asset:
            raise BridgeError(400, "Asset references are accepted only in documented multipart file fields")

        parts: list[dict] = []
        for name in sorted(payload):
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
                raise BridgeError(400, "Multipart field names are invalid")
            value = payload[name]
            values = value if isinstance(value, list) else [value]
            if name in file_names:
                for reference in values:
                    if reference is None:
                        continue
                    if not isinstance(reference, str):
                        raise BridgeError(400, f"Multipart file field {name} must use an uploaded asset reference")
                    record, path = self._asset_descriptor(principal_id, reference)
                    guessed, _encoding = mimetypes.guess_type(record["filename"])
                    media_type = guessed or "application/octet-stream"
                    if not re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", media_type):
                        media_type = "application/octet-stream"
                    parts.append({
                        "name": name,
                        "filename": record["filename"],
                        "path": path,
                        "size": int(record["total"]),
                        "sha256": str(record.get("sha256", "")),
                        "content_type": media_type,
                    })
                continue

            for item in values:
                if item is None:
                    continue
                if isinstance(item, bool):
                    text = "true" if item else "false"
                elif isinstance(item, (str, int)):
                    text = str(item)
                elif isinstance(item, float) and math.isfinite(item):
                    text = str(item)
                else:
                    raise BridgeError(400, f"Multipart field {name} must be a string, number, or boolean")
                parts.append({"name": name, "value": text.encode("utf-8")})
        return parts

    @staticmethod
    def _multipart_header(name: str, *, filename: str | None = None, content_type: str | None = None) -> bytes:
        escaped_name = name.replace('"', "")
        disposition = f'Content-Disposition: form-data; name="{escaped_name}"'
        if filename is not None:
            escaped_file = filename.replace('"', "")
            disposition += f'; filename="{escaped_file}"'
        lines = [disposition]
        if content_type:
            lines.append(f"Content-Type: {content_type}")
        return ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")

    async def _multipart_content(self, parts, boundary: str) -> AsyncIterable[bytes]:
        marker = boundary.encode("ascii")
        collision = b"--" + marker
        for part in parts:
            yield b"--" + marker + b"\r\n"
            if "path" in part:
                yield self._multipart_header(
                    part["name"],
                    filename=part["filename"],
                    content_type=part["content_type"],
                )
                try:
                    with part["path"].open("rb") as stream:
                        if os.fstat(stream.fileno()).st_size != part["size"]:
                            raise BridgeError(409, "Uploaded file asset changed after registration")
                        digest = hashlib.sha256()
                        sent = 0
                        overlap = b""
                        while True:
                            block = stream.read(64 * 1024)
                            if not block:
                                break
                            window = overlap + block
                            if collision in window:
                                raise BridgeError(400, "Uploaded file contains the deterministic multipart boundary")
                            overlap = window[-(len(collision) - 1):]
                            sent += len(block)
                            if sent > part["size"]:
                                raise BridgeError(409, "Uploaded file asset changed after registration")
                            digest.update(block)
                            yield block
                        if sent != part["size"] or digest.hexdigest() != part["sha256"]:
                            raise BridgeError(409, "Uploaded file asset changed after registration")
                except BridgeError:
                    raise
                except OSError as error:
                    raise BridgeError(409, "Uploaded file asset became unavailable during transfer") from error
            else:
                if collision in part["value"]:
                    raise BridgeError(400, "Multipart field contains the deterministic boundary")
                yield self._multipart_header(part["name"], content_type="text/plain; charset=utf-8")
                yield part["value"]
            yield b"\r\n"
        yield b"--" + marker + b"--\r\n"

    @staticmethod
    def _multipart_length(parts, boundary: str) -> int:
        marker = boundary.encode("ascii")
        total = 0
        for part in parts:
            total += len(b"--" + marker + b"\r\n")
            if "path" in part:
                header = BackendClient._multipart_header(
                    part["name"], filename=part["filename"], content_type=part["content_type"],
                )
                total += len(header) + part["size"]
            else:
                header = BackendClient._multipart_header(part["name"], content_type="text/plain; charset=utf-8")
                total += len(header) + len(part["value"])
            total += 2
        return total + len(b"--" + marker + b"--\r\n")

    @staticmethod
    def _response_filename(response, path: str) -> str:
        disposition = response.headers.get("content-disposition", "")
        if disposition:
            message = Message()
            message["content-disposition"] = disposition
            filename = message.get_filename()
            if filename:
                return filename
        fallback = unquote(urlsplit(path).path).rstrip("/").rsplit("/", 1)[-1]
        if not fallback or fallback in {"export", "capture", "preview", "record"}:
            suffixes = {
                "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
                "video/mp4": ".mp4", "audio/mpeg": ".mp3", "audio/flac": ".flac",
                "application/zip": ".zip",
            }
            media = response.headers.get("content-type", "").partition(";")[0].casefold()
            fallback = "download" + suffixes.get(media, ".bin")
        return fallback

    @staticmethod
    async def _response_body(response, *, limit: int = 8 * 1024 * 1024) -> bytes:
        chunks = []
        size = 0
        async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
            size += len(chunk)
            if size > limit:
                raise BridgeError(413, "JSON API response is too large")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _parse_json(raw: bytes):
        if not raw:
            return {}
        try:
            return jsonlib.loads(raw)
        except ValueError as error:
            raise BridgeError(502, "The backend did not return a JSON API response") from error

    async def request(
        self,
        method,
        path,
        *,
        query=None,
        json=None,
        idempotency_key=None,
        request_kind="json",
        response_kind="json",
        file_fields=(),
    ):
        method = str(method).upper()
        request_kind = str(request_kind or "none").casefold()
        response_kind = str(response_kind or "json").casefold()
        if request_kind not in {"none", "json", "multipart"}:
            raise BridgeError(400, "Use a documented JSON or multipart operation")
        if response_kind not in {"json", "binary", "none"}:
            raise BridgeError(400, "Use a documented JSON or binary response operation")
        if request_kind == "none" and json is not None:
            raise BridgeError(400, "This operation does not accept a request body")
        parsed = urlsplit(path)
        decoded = unquote(path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or not path.startswith("/api/v1/") or "\\" in decoded or any(p in {".", ".."} for p in decoded.split("/")):
            raise BridgeError(400, "Use a registered Maestro API operation")
        principal = scope_required(required_scope(method, path))
        if path.startswith("/api/v1/agent-access/jobs/"):
            return self._durable_job(principal, path.rsplit("/", 1)[-1])
        if not self.online():
            if method == "GET" and path == "/api/v1/system-stats":
                return {**self.runtime.public_status(), "backend_online": False, "generation_available": False}
            if method == "GET" and path.startswith("/api/v1/status/"):
                return self._durable_job(principal, path.rsplit("/", 1)[-1])
        payload = json
        multipart_parts = None
        try:
            if request_kind == "json":
                payload = self.assets.resolve_tree(principal.id, json)
            elif request_kind == "multipart":
                multipart_parts = self._multipart_parts(principal.id, json, file_fields)
        except Exception as error:
            status = getattr(error, "status", None)
            if isinstance(status, int):
                raise BridgeError(status, str(error)) from error
            raise
        headers = self.headers()
        headers["Accept"] = "*/*" if response_kind == "binary" else "application/json"
        target = self.target()
        fingerprint = None
        reserved_binary = False
        if method not in {"GET", "HEAD", "OPTIONS"}:
            if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                raise BridgeError(400, "Use the same request_id for retries of this mutation")
            try:
                key_bytes = idempotency_key.encode("utf-8")
            except UnicodeEncodeError as error:
                raise BridgeError(400, "request_id must be valid UTF-8") from error
            if len(key_bytes) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in idempotency_key):
                raise BridgeError(400, "request_id is too long or contains a control character")
            headers["Idempotency-Key"] = idempotency_key
        if response_kind == "binary" and method not in {"GET", "HEAD", "OPTIONS"}:
            try:
                fingerprint = self._request_fingerprint(principal.id, method, path, query, json, file_fields)
                prior = self.transfers.reserve_request(principal.id, idempotency_key, fingerprint)
            except Exception as error:
                status = getattr(error, "status", None)
                if isinstance(status, int):
                    raise BridgeError(status, str(error)) from error
                raise
            if prior is not None:
                return prior
            reserved_binary = True

        request_options = {"params": query, "headers": headers}
        if request_kind == "json":
            request_options["json"] = payload
        elif request_kind == "multipart":
            boundary_source = f"{method}\n{path}\n{idempotency_key or ''}".encode("utf-8")
            boundary = "maestro-agent-" + hashlib.sha256(boundary_source).hexdigest()[:32]
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            headers["Content-Length"] = str(self._multipart_length(multipart_parts or [], boundary))
            request_options["content"] = self._multipart_content(multipart_parts or [], boundary)
        try:
            async with self.client.stream(method, target + path, **request_options) as response:
                if 300 <= response.status_code < 400:
                    raise BridgeError(502, "Unexpected backend redirect; no credentials were forwarded")
                if response.status_code >= 400:
                    raw = await self._response_body(response)
                    result = self._parse_json(raw)
                    message = jsonlib.dumps(redact(result), ensure_ascii=False)[:2000]
                    if reserved_binary:
                        self.transfers.mark_uncertain(principal.id, idempotency_key, fingerprint)
                        reserved_binary = False
                    raise BridgeError(response.status_code, message or f"Maestro returned HTTP {response.status_code}")

                if response_kind == "binary":
                    content_type = response.headers.get("content-type", "application/octet-stream")
                    if "json" in content_type.casefold():
                        result = self._parse_json(await self._response_body(response))
                        if isinstance(result, Mapping) and result.get("kind") == "binary_receipt":
                            if reserved_binary:
                                self.transfers.mark_uncertain(principal.id, idempotency_key, fingerprint)
                                reserved_binary = False
                            raise BridgeError(
                                409,
                                "The native operation already ran, but its binary response is no longer available. Inspect its status; do not re-submit it.",
                            )
                        raise BridgeError(502, "The backend returned JSON where this operation documents a binary response")
                    try:
                        return await self.transfers.capture(
                            principal.id,
                            self._response_filename(response, path),
                            content_type,
                            response.aiter_bytes(chunk_size=64 * 1024),
                            request_id=idempotency_key if reserved_binary else None,
                            fingerprint=fingerprint if reserved_binary else None,
                            access_scope=required_scope(method, path),
                        )
                    finally:
                        reserved_binary = False

                raw = await self._response_body(response)
                result = self._parse_json(raw)
                return result if principal.owner else redact(result)
        except httpx.HTTPError as error:
            if reserved_binary:
                self.transfers.mark_uncertain(principal.id, idempotency_key, fingerprint)
            raise BridgeError(503, "Connection ended. Inspect the same submission receipt; do not submit under a new request_id.") from error
        except Exception as error:
            if reserved_binary:
                self.transfers.mark_uncertain(principal.id, idempotency_key, fingerprint)
            if isinstance(error, BridgeError):
                raise
            status = getattr(error, "status", None)
            if isinstance(status, int):
                raise BridgeError(status, str(error)) from error
            raise

    async def openapi(self):
        scope_required("observe")
        headers = self.headers()
        try:
            response = await self.client.get(self.target() + "/openapi.json", headers=headers)
        except httpx.HTTPError as error:
            raise BridgeError(503, "Start Maestro to discover its installed capabilities") from error
        if response.status_code != 200 or len(response.content) > 8 * 1024 * 1024:
            raise BridgeError(503, "Maestro's operation catalogue is not ready")
        return response.json()

    async def get_submission(self, request_id):
        """Read a principal-owned durable request receipt, including offline."""
        principal = scope_required("observe")
        if not isinstance(request_id, str) or not request_id.strip():
            raise BridgeError(400, "request_id must be a non-empty string")
        try:
            encoded_request_id = request_id.encode("utf-8")
        except UnicodeEncodeError as error:
            raise BridgeError(400, "request_id must be valid UTF-8") from error
        if len(encoded_request_id) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in request_id):
            raise BridgeError(400, "request_id is too long or contains a control character")
        try:
            result = self.journal.find_submission(principal.id, request_id)
        except Exception as error:
            if isinstance(getattr(error, "status", None), int):
                raise BridgeError(error.status, str(error)) from error
            raise
        if result is None:
            raise BridgeError(404, "No owned submission receipt was found for this request_id")
        return result if principal.owner else redact(result)

    async def fetch_media(self, filename, workspace=None):
        """Fetch one visible Maestro file by basename into a private transfer artifact."""
        if not isinstance(filename, str) or not filename.strip():
            raise BridgeError(400, "filename must be a non-empty basename")
        name = filename.strip()
        if name in {".", ".."} or any(char in name for char in ("/", "\\", ":", "\x00")):
            raise BridgeError(400, "filename must be a basename without path separators")
        if any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise BridgeError(400, "filename contains a control character")
        query = None
        if workspace is not None:
            if not isinstance(workspace, str) or not workspace.strip():
                raise BridgeError(400, "workspace must be a non-empty workspace name or ID")
            from ..workspace_paths import WorkspaceNameError, normalize_workspace_name
            try:
                # Nested workspaces are folder paths such as "ClientA/Shoot-1".
                workspace = normalize_workspace_name(workspace)
            except WorkspaceNameError:
                raise BridgeError(400, "workspace must be a safe name or ID") from None
            query = {"workspace": workspace}
        return await self.request(
            "GET",
            "/api/v1/file/" + quote(name, safe=""),
            query=query,
            response_kind="binary",
        )

    async def read_transfer_chunk(self, transfer_id, offset=0, max_bytes=65_536):
        principal = principal_context.get()
        if principal is None:
            raise BridgeError(401, "Pair and authenticate this agent first")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise BridgeError(400, "offset must be a non-negative byte count")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1 or max_bytes > 1_048_576:
            raise BridgeError(400, "max_bytes must be between 1 byte and 1 MiB")
        is_admin = "admin" in principal.scopes
        try:
            access_scope = self.transfers.access_scope(
                transfer_id, principal.id, owner=principal.owner, admin=is_admin,
            )
            if not principal.owner and not is_admin and access_scope not in principal.scopes:
                raise BridgeError(403, f"This transfer requires the {access_scope} scope")
            return self.transfers.read_chunk(
                transfer_id, principal.id, owner=principal.owner, admin=is_admin,
                offset=offset, max_bytes=max_bytes,
            )
        except BridgeError:
            raise
        except Exception as error:
            if isinstance(getattr(error, "status", None), int):
                raise BridgeError(error.status, str(error)) from error
            raise

    async def peers(self):
        scope_required("observe")
        return {"this_installation": self.runtime.installation_id, "this_workstation": self.runtime.public_status().get("workstation_id"), "workstations": self.discovery.peers() if self.discovery else [], "note": "Discoveries require pairing and certificate verification; discovery is not authorization."}

    async def identity(self):
        scope_required("observe")
        status = self.runtime.public_status()
        return {"installation_id": self.runtime.installation_id, "workstation_id": status.get("workstation_id")}

    async def lifecycle(self, action, request_id=None):
        principal = scope_required("observe" if action == "status" else "admin")
        if action == "status":
            result = {**self.runtime.public_status(), "backend_online": self.online()}
            if self.lifecycle_manager:
                try:
                    result["pinokio"] = await self.lifecycle_manager.execute("status", principal.id, None)
                except Exception as error:
                    if isinstance(getattr(error, "status", None), int):
                        raise BridgeError(error.status, str(error)) from error
                    raise
            return result
        if not self.lifecycle_manager:
            raise BridgeError(503, "The Pinokio lifecycle controller is unavailable")
        try:
            return await self.lifecycle_manager.execute(action, principal.id, request_id)
        except Exception as error:
            if isinstance(getattr(error, "status", None), int):
                raise BridgeError(error.status, str(error)) from error
            raise

    async def upload_begin(self, filename, size, sha256):
        principal = scope_required("manage")
        return self.assets.begin(principal.id, filename, size, sha256)

    async def upload_status(self, upload_id):
        principal = scope_required("observe")
        return self.assets.status(principal.id, upload_id)

    async def upload_chunk(self, upload_id, offset, data_base64, request_id):
        principal = scope_required("manage")
        if not isinstance(data_base64, str) or len(data_base64) > 1_398_104:
            raise BridgeError(413, "Use chunks no larger than one MiB")
        try:
            content = base64.b64decode(data_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise BridgeError(400, "Use valid base64 upload bytes") from error
        if len(content) > 1024 * 1024:
            raise BridgeError(413, "Use chunks no larger than one MiB")
        if not isinstance(request_id, str) or not request_id.strip():
            raise BridgeError(400, "A stable request_id is required")
        return self.assets.write_chunk(principal.id, upload_id, offset, content)

    async def upload_complete(self, upload_id, request_id):
        principal = scope_required("manage")
        status = self.assets.status(principal.id, upload_id)
        if status["state"] == "imported":
            return status
        self.assets.finish(principal.id, upload_id)
        return await self.request("POST", f"/api/v1/agent-access/assets/uploads/{upload_id}/import", json={}, idempotency_key=request_id)

    async def idle_probe(self):
        """Conservative: missing or failed evidence never authorizes stopping."""
        if not self.online():
            return {"idle": not process_alive(self.runtime.backend().get("pid")), "certain": not process_alive(self.runtime.backend().get("pid")), "reason": "Backend is stopped"}
        try:
            evidence = await self.request("GET", "/api/v1/agent-access/idle")
        except Exception:
            return {"idle": False, "certain": False, "reason": "Could not verify every queue and engine"}
        if not isinstance(evidence, dict) or type(evidence.get("idle")) is not bool or type(evidence.get("certain")) is not bool:
            return {"idle": False, "certain": False, "reason": "Invalid idle evidence"}
        return evidence
