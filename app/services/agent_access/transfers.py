"""Private, bounded storage for binary responses returned through Agent Access.

Transfer records expose only an opaque ID, media metadata and a controller
URL. Files stay in the installation's private state directory; no local path
is returned to a client.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping
import base64
import hashlib
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Callable
from urllib.parse import urlsplit
from .context import public_origin_context
import uuid


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE_RE = re.compile(r"^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$")
_DEFAULT_MAX_FILE_BYTES = 500 * 1024 * 1024
_DEFAULT_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
_DEFAULT_MAX_PRINCIPAL_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_MAX_TRANSFERS = 128
_DEFAULT_MAX_PRINCIPAL_TRANSFERS = 32
_DEFAULT_COMPLETE_TTL = 7 * 24 * 60 * 60
_DEFAULT_UNFINISHED_TTL = 60 * 60
_IO_CHUNK_BYTES = 64 * 1024


class TransferStoreError(Exception):
    """An expected transfer-store error with an HTTP-friendly status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)


def _required_text(value: Any, name: str, *, max_bytes: int = 512) -> str:
    if not isinstance(value, str) or not value:
        raise TransferStoreError(400, f"{name} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise TransferStoreError(400, f"{name} must be valid UTF-8") from error
    if len(encoded) > max_bytes:
        raise TransferStoreError(400, f"{name} is too long")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise TransferStoreError(400, f"{name} contains a control character")
    return value


def _transfer_id(value: Any) -> str:
    if not isinstance(value, str) or not _UUID_RE.fullmatch(value):
        raise TransferStoreError(404, "Transfer not found")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise TransferStoreError(404, "Transfer not found") from error
    if str(parsed) != value.lower():
        raise TransferStoreError(404, "Transfer not found")
    return str(parsed)


def _safe_filename(value: Any) -> str:
    if not isinstance(value, str):
        value = "download.bin"
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        value = "download.bin"
        encoded = value.encode("utf-8")
    if (
        not value
        or len(encoded) > 255
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(char in '<>:"|?*' for char in value)
        or value.endswith((".", " "))
    ):
        return "download.bin"
    return value


def _safe_media_type(value: Any) -> str:
    if not isinstance(value, str):
        return "application/octet-stream"
    media_type = value.partition(";")[0].strip().casefold()
    if len(media_type) > 127 or not _MEDIA_TYPE_RE.fullmatch(media_type):
        return "application/octet-stream"
    return media_type


class TransferStore:
    """Persistent, principal-owned storage for bounded streamed responses."""

    MAX_FILE_BYTES = _DEFAULT_MAX_FILE_BYTES

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        origin: str = "",
        clock: Callable[[], float] | None = None,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
        max_principal_bytes: int = _DEFAULT_MAX_PRINCIPAL_BYTES,
        max_transfers: int = _DEFAULT_MAX_TRANSFERS,
        max_principal_transfers: int = _DEFAULT_MAX_PRINCIPAL_TRANSFERS,
        complete_ttl_seconds: int = _DEFAULT_COMPLETE_TTL,
        unfinished_ttl_seconds: int = _DEFAULT_UNFINISHED_TTL,
    ):
        self.directory = Path(directory)
        self._clock = clock or time.time
        for name, value, minimum in (
            ("max_file_bytes", max_file_bytes, 0),
            ("max_total_bytes", max_total_bytes, 0),
            ("max_principal_bytes", max_principal_bytes, 0),
            ("max_transfers", max_transfers, 1),
            ("max_principal_transfers", max_principal_transfers, 1),
            ("complete_ttl_seconds", complete_ttl_seconds, 1),
            ("unfinished_ttl_seconds", unfinished_ttl_seconds, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.max_file_bytes = min(max_file_bytes, self.MAX_FILE_BYTES)
        self.max_total_bytes = max_total_bytes
        self.max_principal_bytes = max_principal_bytes
        self.max_transfers = max_transfers
        self.max_principal_transfers = max_principal_transfers
        self.complete_ttl_seconds = complete_ttl_seconds
        self.unfinished_ttl_seconds = unfinished_ttl_seconds
        self.origin = self._validate_origin(origin)

        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise TransferStoreError(500, "Transfer-store directory must be a real directory")
        self.staging_directory = self.directory / "staging"
        self.blob_directory = self.directory / "blobs"
        for path in (self.staging_directory, self.blob_directory):
            path.mkdir(mode=0o700, exist_ok=True)
            if path.is_symlink() or not path.is_dir():
                raise TransferStoreError(500, "Transfer storage directories must be real directories")
            self._secure_permissions(path, directory=True)
        self._secure_permissions(self.directory, directory=True)
        self.database_path = self.directory / "transfer-store.sqlite3"
        if self.database_path.is_symlink() or (
            self.database_path.exists() and not self.database_path.is_file()
        ):
            raise TransferStoreError(500, "Transfer-store database path must be a regular file")
        self._initialize()
        self.cleanup()

    @staticmethod
    def _validate_origin(value: Any) -> str:
        if not isinstance(value, str) or not value:
            return ""
        try:
            parsed = urlsplit(value)
        except ValueError as error:
            raise ValueError("origin must be a valid HTTPS origin") from error
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("origin must be a valid HTTPS origin")
        try:
            _ = parsed.port
        except ValueError as error:
            raise ValueError("origin must be a valid HTTPS origin") from error
        return f"https://{parsed.netloc}"

    @staticmethod
    def _secure_permissions(path: Path, *, directory: bool) -> None:
        try:
            os.chmod(path, 0o700 if directory else 0o600)
        except OSError:
            pass

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError) as error:
            raise TransferStoreError(500, "Transfer-store clock did not return a timestamp") from error
        if not math.isfinite(value):
            raise TransferStoreError(500, "Transfer-store clock did not return a finite timestamp")
        return value

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path), timeout=30.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS transfers (
                    transfer_id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    request_id TEXT,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    access_scope TEXT NOT NULL DEFAULT 'observe',
                    size INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT,
                    state TEXT NOT NULL CHECK (state IN ('writing','complete','failed','expired')),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS transfers_owner_state
                    ON transfers (principal_id, state, created_at);
                CREATE INDEX IF NOT EXISTS transfers_expiry
                    ON transfers (state, expires_at);
                CREATE TABLE IF NOT EXISTS requests (
                    principal_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending','complete','uncertain')),
                    transfer_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (principal_id, request_id)
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(transfers)")}
            if "access_scope" not in columns:
                connection.execute(
                    "ALTER TABLE transfers ADD COLUMN access_scope TEXT NOT NULL DEFAULT 'observe'"
                )
        finally:
            connection.close()
        self._secure_permissions(self.database_path, directory=False)

    def _stage_path(self, transfer_id: str) -> Path:
        identifier = _transfer_id(transfer_id)
        if self.staging_directory.is_symlink() or not self.staging_directory.is_dir():
            raise TransferStoreError(500, "Transfer staging directory is unavailable")
        path = self.staging_directory / f"{identifier}.part"
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise TransferStoreError(500, "Transfer staging path is unsafe")
        return path

    def _blob_path(self, transfer_id: str) -> Path:
        identifier = _transfer_id(transfer_id)
        if self.blob_directory.is_symlink() or not self.blob_directory.is_dir():
            raise TransferStoreError(500, "Transfer storage directory is unavailable")
        path = self.blob_directory / f"{identifier}.bin"
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise TransferStoreError(500, "Transfer blob path is unsafe")
        return path

    @staticmethod
    def _fsync_file(stream: Any) -> None:
        stream.flush()
        os.fsync(stream.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def _public_record(self, row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        transfer_id = str(row["transfer_id"])
        origin = public_origin_context.get() or self.origin
        url = f"{origin}/api/v1/agent-access/transfers/{transfer_id}"
        return {
            "transfer_id": transfer_id,
            "filename": str(row["filename"]),
            "content_type": str(row["content_type"]),
            "size": int(row["size"]),
            "sha256": str(row["sha256"] or ""),
            "url": url,
        }

    def _lookup_transfer_row(self, connection: sqlite3.Connection, transfer_id: str) -> sqlite3.Row:
        identifier = _transfer_id(transfer_id)
        row = connection.execute(
            "SELECT * FROM transfers WHERE transfer_id = ? AND state = 'complete'",
            (identifier,),
        ).fetchone()
        if row is None:
            raise TransferStoreError(404, "Transfer not found")
        if float(row["expires_at"]) <= self._now():
            raise TransferStoreError(410, "Transfer has expired")
        return row

    @staticmethod
    def _authorized(row: sqlite3.Row, principal_id: str, *, owner: bool, admin: bool) -> None:
        if not (owner or admin) and row["principal_id"] != principal_id:
            raise TransferStoreError(404, "Transfer not found")

    def _remove_file(self, path: Path) -> None:
        try:
            parent = path.parent.resolve(strict=True)
            allowed = {self.staging_directory.resolve(strict=True), self.blob_directory.resolve(strict=True)}
            if parent not in allowed or path.is_symlink():
                raise TransferStoreError(500, "Refusing to remove an unsafe transfer path")
            if path.exists():
                if not path.is_file():
                    raise TransferStoreError(500, "Refusing to remove a non-file transfer path")
                path.unlink()
        except FileNotFoundError:
            return
        except OSError as error:
            raise TransferStoreError(500, "Unable to remove an expired transfer") from error

    def cleanup(self) -> int:
        """Remove expired artifacts and interrupted stream files safely."""
        now = self._now()
        connection = self._connect()
        removed_rows: list[sqlite3.Row] = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM transfers WHERE state='failed' OR (state='writing' AND updated_at <= ?) OR (state='complete' AND expires_at <= ?)",
                (now - self.unfinished_ttl_seconds, now),
            ).fetchall()
            for row in rows:
                is_expired = row["state"] == "complete"
                # Preserve a pending/uncertain write receipt indefinitely so a
                # retry cannot accidentally repeat a native mutation.
                if row["request_id"]:
                    connection.execute(
                        "UPDATE requests SET state='uncertain', updated_at=? WHERE transfer_id=? AND state!='complete'",
                        (now, row["transfer_id"]),
                    )
                if is_expired:
                    connection.execute(
                        "UPDATE transfers SET state='expired', size=0, updated_at=? WHERE transfer_id=?",
                        (now, row["transfer_id"]),
                    )
                    removed_rows.append(row)
                else:
                    connection.execute(
                        "UPDATE transfers SET state='failed', size=0, updated_at=? WHERE transfer_id=?",
                        (now, row["transfer_id"]),
                    )
                    removed_rows.append(row)
            connection.execute(
                "UPDATE requests SET state='uncertain',updated_at=? WHERE state='pending' AND updated_at <= ?",
                (now, now - self.unfinished_ttl_seconds),
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        for row in removed_rows:
            self._remove_file(self._stage_path(row["transfer_id"]))
            self._remove_file(self._blob_path(row["transfer_id"]))
        if removed_rows:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(
                    "DELETE FROM transfers WHERE transfer_id=? AND state IN ('failed','expired')",
                    [(str(row["transfer_id"]),) for row in removed_rows],
                )
                connection.commit()
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()
        if removed_rows:
            self._fsync_directory(self.staging_directory)
            self._fsync_directory(self.blob_directory)
        return len(removed_rows)

    def reserve_request(self, principal_id: str, request_id: str, fingerprint: str) -> dict[str, Any] | None:
        """Reserve a binary mutation receipt, returning a completed replay if present.

        A pending or uncertain receipt is never dispatched again. ``fingerprint``
        must be a SHA-256 digest of the exact logical operation and inputs.
        """
        principal_id = _required_text(principal_id, "principal_id")
        request_id = _required_text(request_id, "request_id", max_bytes=256)
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise TransferStoreError(400, "Request fingerprint must be a SHA-256 digest")
        now = self._now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM requests WHERE principal_id=? AND request_id=?",
                (principal_id, request_id),
            ).fetchone()
            if row is not None:
                if row["fingerprint"] != fingerprint:
                    raise TransferStoreError(409, "This request_id was already used with different operation inputs")
                if row["state"] == "complete" and row["transfer_id"]:
                    try:
                        transfer = self._lookup_transfer_row(connection, row["transfer_id"])
                    except TransferStoreError as error:
                        raise TransferStoreError(410, "The binary response has expired; do not re-submit the native operation") from error
                    connection.commit()
                    return self._public_record(transfer)
                if row["state"] == "pending":
                    raise TransferStoreError(409, "This binary operation is still being captured; check its status before retrying")
                raise TransferStoreError(409, "The binary response is uncertain; inspect the native receipt and do not re-submit")
            total_requests = connection.execute(
                "SELECT COUNT(*) FROM requests WHERE principal_id=?",
                (principal_id,),
            ).fetchone()[0]
            global_requests = connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            if total_requests >= 256 or global_requests >= 16384:
                raise TransferStoreError(429, "Binary request receipt quota reached")
            connection.execute(
                "INSERT INTO requests (principal_id,request_id,fingerprint,state,created_at,updated_at) VALUES (?,?,?,'pending',?,?)",
                (principal_id, request_id, fingerprint, now, now),
            )
            connection.commit()
            return None
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def mark_uncertain(self, principal_id: str, request_id: str, fingerprint: str | None = None) -> None:
        principal_id = _required_text(principal_id, "principal_id")
        request_id = _required_text(request_id, "request_id", max_bytes=256)
        now = self._now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT fingerprint, state FROM requests WHERE principal_id=? AND request_id=?",
                (principal_id, request_id),
            ).fetchone()
            if row is None:
                return
            if fingerprint is not None and row["fingerprint"] != fingerprint:
                raise TransferStoreError(409, "This request_id was already used with different operation inputs")
            if row["state"] != "complete":
                connection.execute(
                    "UPDATE requests SET state='uncertain', updated_at=? WHERE principal_id=? AND request_id=?",
                    (now, principal_id, request_id),
                )
                connection.execute(
                    "UPDATE transfers SET state='failed', size=0, updated_at=? WHERE principal_id=? AND request_id=? AND state='writing'",
                    (now, principal_id, request_id),
                )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _begin_capture(
        self,
        principal_id: str,
        filename: str,
        content_type: str,
        request_id: str | None,
        fingerprint: str | None,
        access_scope: str,
    ) -> tuple[str, Path]:
        now = self._now()
        transfer_id = str(uuid.uuid4())
        path = self._stage_path(transfer_id)
        connection = self._connect()
        created_file = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                request = connection.execute(
                    "SELECT fingerprint,state FROM requests WHERE principal_id=? AND request_id=?",
                    (principal_id, request_id),
                ).fetchone()
                if request is None or request["state"] != "pending":
                    raise TransferStoreError(409, "Binary request was not safely reserved")
                if request["fingerprint"] != fingerprint:
                    raise TransferStoreError(409, "Binary request fingerprint changed before capture")
            count_global, count_owner = connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN principal_id=? THEN 1 ELSE 0 END) FROM transfers WHERE state IN ('writing','complete')",
                (principal_id,),
            ).fetchone()
            if count_global >= self.max_transfers or (count_owner or 0) >= self.max_principal_transfers:
                raise TransferStoreError(429, "Transfer record quota reached")
            while connection.execute("SELECT 1 FROM transfers WHERE transfer_id=?", (transfer_id,)).fetchone():
                transfer_id = str(uuid.uuid4())
                path = self._stage_path(transfer_id)
            with path.open("xb") as stream:
                created_file = True
                self._secure_permissions(path, directory=False)
                self._fsync_file(stream)
            connection.execute(
                "INSERT INTO transfers (transfer_id,principal_id,request_id,filename,content_type,access_scope,size,state,created_at,updated_at,expires_at) VALUES (?,?,?,?,?,?,0,'writing',?,?,?)",
                (transfer_id, principal_id, request_id, filename, content_type, access_scope, now, now, now + self.complete_ttl_seconds),
            )
            if request_id is not None:
                connection.execute(
                    "UPDATE requests SET transfer_id=?,updated_at=? WHERE principal_id=? AND request_id=?",
                    (transfer_id, now, principal_id, request_id),
                )
            connection.commit()
            self._fsync_directory(self.staging_directory)
            return transfer_id, path
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            if created_file:
                self._remove_file(path)
            raise
        finally:
            connection.close()

    def _reserve_bytes(self, transfer_id: str, principal_id: str, amount: int) -> None:
        if amount <= 0:
            return
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT principal_id,size,state FROM transfers WHERE transfer_id=?",
                (transfer_id,),
            ).fetchone()
            if row is None or row["principal_id"] != principal_id or row["state"] != "writing":
                raise TransferStoreError(409, "Transfer capture is no longer active")
            next_size = int(row["size"]) + amount
            if next_size > self.max_file_bytes:
                raise TransferStoreError(413, "Binary response exceeds the per-file size limit")
            total, owner_total = connection.execute(
                "SELECT COALESCE(SUM(size),0), COALESCE(SUM(CASE WHEN principal_id=? THEN size ELSE 0 END),0) FROM transfers WHERE state IN ('writing','complete')",
                (principal_id,),
            ).fetchone()
            if total + amount > self.max_total_bytes:
                raise TransferStoreError(429, "Transfer storage quota reached")
            if owner_total + amount > self.max_principal_bytes:
                raise TransferStoreError(429, "Principal transfer storage quota reached")
            connection.execute(
                "UPDATE transfers SET size=?,updated_at=? WHERE transfer_id=?",
                (next_size, self._now(), transfer_id),
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    async def capture(
        self,
        principal_id: str,
        filename: str,
        content_type: str,
        chunks: AsyncIterable[bytes],
        *,
        request_id: str | None = None,
        fingerprint: str | None = None,
        access_scope: str = "observe",
    ) -> dict[str, Any]:
        """Stream and verify a binary response, then return opaque metadata.

        When ``request_id`` is supplied, the caller must first call
        :meth:`reserve_request`. Any interrupted capture marks that receipt
        uncertain so the native mutation cannot be repeated.
        """
        principal_id = _required_text(principal_id, "principal_id")
        filename = _safe_filename(filename)
        content_type = _safe_media_type(content_type)
        if request_id is not None:
            request_id = _required_text(request_id, "request_id", max_bytes=256)
            if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
                raise TransferStoreError(400, "Request fingerprint must be a SHA-256 digest")
        if access_scope not in {"observe", "generate", "manage", "admin"}:
            raise TransferStoreError(400, "access_scope is invalid")
        self.cleanup()
        try:
            transfer_id, stage_path = self._begin_capture(
                principal_id, filename, content_type, request_id, fingerprint, access_scope,
            )
        except Exception:
            if request_id is not None:
                self.mark_uncertain(principal_id, request_id, fingerprint)
            raise
        digest = hashlib.sha256()
        size = 0
        blob_path = self._blob_path(transfer_id)
        try:
            with stage_path.open("r+b") as output:
                async for incoming in chunks:
                    if not isinstance(incoming, (bytes, bytearray, memoryview)):
                        raise TransferStoreError(502, "Backend returned an invalid binary stream")
                    view = memoryview(incoming)
                    for start in range(0, len(view), _IO_CHUNK_BYTES):
                        piece = view[start:start + _IO_CHUNK_BYTES]
                        if not piece:
                            continue
                        self._reserve_bytes(transfer_id, principal_id, len(piece))
                        try:
                            written = output.write(piece)
                        except OSError as error:
                            raise TransferStoreError(500, "Unable to store the binary response") from error
                        if written != len(piece):
                            raise TransferStoreError(500, "Binary response write was incomplete")
                        digest.update(piece)
                        size += len(piece)
                self._fsync_file(output)
            if stage_path.is_symlink() or not stage_path.is_file() or stage_path.stat().st_size != size:
                raise TransferStoreError(500, "Stored binary response size could not be verified")
            if blob_path.exists() or blob_path.is_symlink():
                raise TransferStoreError(500, "Transfer blob path already exists")
            os.replace(stage_path, blob_path)
            self._secure_permissions(blob_path, directory=False)
            self._fsync_directory(self.blob_directory)
            now = self._now()
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT principal_id,size,state FROM transfers WHERE transfer_id=?",
                    (transfer_id,),
                ).fetchone()
                if row is None or row["principal_id"] != principal_id or row["state"] != "writing" or row["size"] != size:
                    raise TransferStoreError(500, "Transfer capture record failed verification")
                connection.execute(
                    "UPDATE transfers SET state='complete',sha256=?,updated_at=?,expires_at=? WHERE transfer_id=?",
                    (digest.hexdigest(), now, now + self.complete_ttl_seconds, transfer_id),
                )
                if request_id is not None:
                    connection.execute(
                        "UPDATE requests SET state='complete',transfer_id=?,updated_at=? WHERE principal_id=? AND request_id=? AND fingerprint=? AND state='pending'",
                        (transfer_id, now, principal_id, request_id, fingerprint),
                    )
                    receipt = connection.execute(
                        "SELECT state FROM requests WHERE principal_id=? AND request_id=?",
                        (principal_id, request_id),
                    ).fetchone()
                    if receipt is None or receipt["state"] != "complete":
                        raise TransferStoreError(409, "Binary operation receipt changed during capture")
                result = self._public_record(connection.execute(
                    "SELECT * FROM transfers WHERE transfer_id=?", (transfer_id,),
                ).fetchone())
                connection.commit()
                return result
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()
        except Exception as error:
            self._fail_capture(transfer_id, request_id, principal_id)
            if isinstance(error, TransferStoreError):
                raise
            raise TransferStoreError(503, "Binary response capture was interrupted; do not re-submit the same operation under a new request_id") from error

    def _fail_capture(self, transfer_id: str, request_id: str | None, principal_id: str) -> None:
        now = self._now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE transfers SET state='failed',size=0,updated_at=? WHERE transfer_id=? AND state='writing'",
                (now, transfer_id),
            )
            if request_id is not None:
                connection.execute(
                    "UPDATE requests SET state='uncertain',updated_at=? WHERE principal_id=? AND request_id=? AND state!='complete'",
                    (now, principal_id, request_id),
                )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
        finally:
            connection.close()
        cleaned = True
        for path in (self._stage_path(transfer_id), self._blob_path(transfer_id)):
            try:
                self._remove_file(path)
            except TransferStoreError:
                cleaned = False
        if cleaned:
            connection = self._connect()
            try:
                connection.execute(
                    "DELETE FROM transfers WHERE transfer_id=? AND state='failed'",
                    (transfer_id,),
                )
            finally:
                connection.close()

    def metadata(
        self,
        transfer_id: str,
        principal_id: str,
        *,
        owner: bool = False,
        admin: bool = False,
    ) -> dict[str, Any]:
        principal_id = _required_text(principal_id, "principal_id")
        connection = self._connect()
        try:
            row = self._lookup_transfer_row(connection, transfer_id)
            self._authorized(row, principal_id, owner=owner, admin=admin)
            return self._public_record(row)
        finally:
            connection.close()

    def access_scope(
        self,
        transfer_id: str,
        principal_id: str,
        *,
        owner: bool = False,
        admin: bool = False,
    ) -> str:
        """Return the scope that created this transfer after checking ownership."""
        principal_id = _required_text(principal_id, "principal_id")
        connection = self._connect()
        try:
            row = self._lookup_transfer_row(connection, transfer_id)
            self._authorized(row, principal_id, owner=owner, admin=admin)
            return str(row["access_scope"])
        finally:
            connection.close()

    async def iter_bytes(
        self,
        transfer_id: str,
        principal_id: str,
        *,
        owner: bool = False,
        admin: bool = False,
        offset: int = 0,
        length: int | None = None,
        chunk_size: int = _IO_CHUNK_BYTES,
    ):
        """Yield an authorized transfer in bounded blocks for a streaming HTTP route."""
        record = self.metadata(transfer_id, principal_id, owner=owner, admin=admin)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset > record["size"]:
            raise TransferStoreError(416, "Transfer byte offset is outside the stored file")
        if length is None:
            length = record["size"] - offset
        if isinstance(length, bool) or not isinstance(length, int) or length < 0 or offset + length > record["size"]:
            raise TransferStoreError(416, "Transfer byte range is outside the stored file")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1 or chunk_size > 1024 * 1024:
            raise TransferStoreError(400, "chunk_size must be between 1 byte and 1 MiB")
        path = self._blob_path(transfer_id)
        if not path.exists() or path.is_symlink() or not path.is_file():
            raise TransferStoreError(410, "Stored transfer bytes are unavailable")
        remaining = length
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                while remaining:
                    block = stream.read(min(chunk_size, remaining))
                    if not block:
                        raise TransferStoreError(410, "Stored transfer ended before its recorded size")
                    remaining -= len(block)
                    yield block
        except OSError as error:
            raise TransferStoreError(410, "Stored transfer bytes are unavailable") from error

    def read_chunk(
        self,
        transfer_id: str,
        principal_id: str,
        *,
        owner: bool = False,
        admin: bool = False,
        offset: int = 0,
        max_bytes: int = 64 * 1024,
    ) -> dict[str, Any]:
        """Return one bounded base64 block and the verified artifact digest."""
        record = self.metadata(transfer_id, principal_id, owner=owner, admin=admin)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset > record["size"]:
            raise TransferStoreError(416, "Transfer byte offset is outside the stored file")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= 1024 * 1024:
            raise TransferStoreError(400, "max_bytes must be between 1 byte and 1 MiB")
        length = min(max_bytes, record["size"] - offset)
        path = self._blob_path(transfer_id)
        if not path.exists() or path.is_symlink() or not path.is_file():
            raise TransferStoreError(410, "Stored transfer bytes are unavailable")
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(length)
        except OSError as error:
            raise TransferStoreError(410, "Stored transfer bytes are unavailable") from error
        if len(data) != length:
            raise TransferStoreError(410, "Stored transfer ended before its recorded size")
        next_offset = offset + len(data)
        return {
            "transfer_id": record["transfer_id"],
            "offset": offset,
            "data_base64": base64.b64encode(data).decode("ascii"),
            "bytes_read": len(data),
            "eof": next_offset >= record["size"],
            "sha256": record["sha256"],
        }


__all__ = ["TransferStore", "TransferStoreError"]
