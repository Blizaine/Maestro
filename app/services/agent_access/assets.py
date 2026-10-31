"""Private, resumable asset uploads and trusted backend-path references.

This module stores upload bytes in a private staging directory and records
committed offsets in SQLite. It does not perform HTTP transfers, contact model
backends, or trust paths supplied in request payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import os
from pathlib import Path, PurePath
import re
import sqlite3
import time
from typing import Any, Callable
import unicodedata
import uuid


class AssetStoreError(Exception):
    """An expected asset-store error with an HTTP-friendly status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_REFERENCE_RE = re.compile(r"^(?:asset|maestro-asset):([0-9a-fA-F-]+)$")
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul", "conin$", "conout$",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
    *(f"com{number}" for number in "¹²³"),
    *(f"lpt{number}" for number in "¹²³"),
}
_ALLOWED_BACKEND_ROOTS = ("uploads", "outputs", "storage")


def _required_text(value: Any, name: str, max_bytes: int = 512) -> str:
    if not isinstance(value, str) or not value:
        raise AssetStoreError(400, f"{name} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AssetStoreError(400, f"{name} must be valid UTF-8") from exc
    if len(encoded) > max_bytes:
        raise AssetStoreError(400, f"{name} is too long")
    return value


def _canonical_uuid(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _UUID_RE.fullmatch(value):
        raise AssetStoreError(400, f"{name} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise AssetStoreError(400, f"{name} must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise AssetStoreError(400, f"{name} must be a canonical UUID")
    return str(parsed)


def _safe_basename(filename: Any) -> str:
    if not isinstance(filename, str):
        raise AssetStoreError(400, "filename must be a string")
    try:
        filename_bytes = filename.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AssetStoreError(400, "filename must be valid UTF-8") from exc
    if not filename or len(filename_bytes) > 255:
        raise AssetStoreError(400, "filename is empty or too long")
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise AssetStoreError(400, "filename must be a safe basename")
    if any(ord(char) < 32 or ord(char) == 127 for char in filename):
        raise AssetStoreError(400, "filename contains control characters")
    if any(char in '<>:"|?*' for char in filename):
        raise AssetStoreError(400, "filename contains characters invalid on Windows")
    if filename.endswith((".", " ")):
        raise AssetStoreError(400, "filename must not end in a dot or space")
    normalized = unicodedata.normalize("NFKC", filename).casefold()
    device_base = normalized.split(".", 1)[0].rstrip(" .")
    if device_base in _WINDOWS_RESERVED_NAMES:
        raise AssetStoreError(400, "filename uses a reserved Windows device name")
    return filename


def _reference(asset_id: str) -> str:
    return f"asset:{asset_id}"


class AssetStore:
    """Persistent resumable uploads and trusted imported-asset references."""

    MAX_CHUNK_BYTES = 8 * 1024 * 1024
    MAX_FILE_BYTES = 500 * 1024 * 1024

    def __init__(
        self,
        directory: str | os.PathLike[str],
        app_root: str | os.PathLike[str],
        clock: Callable[[], float] | None = None,
        *,
        max_chunk_bytes: int = MAX_CHUNK_BYTES,
        max_file_bytes: int = MAX_FILE_BYTES,
        max_pending_per_principal: int = 8,
        max_pending_global: int = 128,
        max_pending_bytes_per_principal: int = 1024 * 1024 * 1024,
        max_pending_bytes_global: int = 4 * 1024 * 1024 * 1024,
    ):
        self.directory = Path(directory)
        self.app_root = Path(app_root).absolute()
        self._clock = clock or time.time
        for value, name, minimum in (
            (max_chunk_bytes, "max_chunk_bytes", 1),
            (max_file_bytes, "max_file_bytes", 0),
            (max_pending_per_principal, "max_pending_per_principal", 1),
            (max_pending_global, "max_pending_global", 1),
            (max_pending_bytes_per_principal, "max_pending_bytes_per_principal", 0),
            (max_pending_bytes_global, "max_pending_bytes_global", 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.max_chunk_bytes = min(max_chunk_bytes, self.MAX_CHUNK_BYTES)
        self.max_file_bytes = min(max_file_bytes, self.MAX_FILE_BYTES)
        self.max_pending_per_principal = max_pending_per_principal
        self.max_pending_global = max_pending_global
        self.max_pending_bytes_per_principal = max_pending_bytes_per_principal
        self.max_pending_bytes_global = max_pending_bytes_global

        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise AssetStoreError(500, "Asset-store directory must be a real directory")
        self.staging_directory = self.directory / "staging"
        self.staging_directory.mkdir(mode=0o700, exist_ok=True)
        if self.staging_directory.is_symlink() or not self.staging_directory.is_dir():
            raise AssetStoreError(500, "Asset staging directory must be a real directory")
        self._secure_permissions(self.directory, directory=True)
        self._secure_permissions(self.staging_directory, directory=True)
        self.database_path = self.directory / "asset-store.sqlite3"
        if self.database_path.is_symlink() or (
            self.database_path.exists() and not self.database_path.is_file()
        ):
            raise AssetStoreError(500, "Asset-store database path must be a regular file")
        self._initialize()
        self._recover_uncommitted_tails()

    @staticmethod
    def _secure_permissions(path: Path, *, directory: bool) -> None:
        try:
            os.chmod(path, 0o700 if directory else 0o600)
        except OSError:
            pass

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
                CREATE TABLE IF NOT EXISTS uploads (
                    upload_id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL UNIQUE,
                    principal_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('receiving','ready','imported','failed','expired')),
                    received INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    failure_reason TEXT,
                    backend_path TEXT
                );
                CREATE INDEX IF NOT EXISTS uploads_owner_state
                    ON uploads (principal_id, state, created_at);
                CREATE INDEX IF NOT EXISTS uploads_state_updated
                    ON uploads (state, updated_at);
                CREATE TABLE IF NOT EXISTS chunks (
                    upload_id TEXT NOT NULL,
                    offset INTEGER NOT NULL,
                    length INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY (upload_id, offset),
                    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS chunks_upload
                    ON chunks (upload_id, offset);
                """
            )
        finally:
            connection.close()
        self._secure_permissions(self.database_path, directory=False)

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError) as exc:
            raise AssetStoreError(500, "Asset-store clock did not return a timestamp") from exc
        if not math.isfinite(value):
            raise AssetStoreError(500, "Asset-store clock did not return a finite timestamp")
        return value

    def _stage_path(self, upload_id: str) -> Path:
        upload_id = _canonical_uuid(upload_id, "upload_id")
        if self.staging_directory.is_symlink() or not self.staging_directory.is_dir():
            raise AssetStoreError(500, "Asset staging directory is unavailable")
        path = self.staging_directory / f"{upload_id}.part"
        if path.is_symlink():
            raise AssetStoreError(500, "Staging file must not be a symbolic link")
        if path.exists() and not path.is_file():
            raise AssetStoreError(500, "Staging path is not a regular file")
        return path

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

    @staticmethod
    def _fsync_file(stream: Any) -> None:
        stream.flush()
        os.fsync(stream.fileno())

    def _create_empty_stage_file(self, upload_id: str) -> Path:
        path = self._stage_path(upload_id)
        created = False
        try:
            with path.open("xb") as stream:
                created = True
                self._secure_permissions(path, directory=False)
                self._fsync_file(stream)
            self._fsync_directory(self.staging_directory)
        except FileExistsError as exc:
            raise AssetStoreError(409, "Upload staging file already exists") from exc
        except OSError as exc:
            if created and path.exists() and not path.is_symlink():
                try:
                    path.unlink()
                except OSError:
                    pass
            raise AssetStoreError(500, "Unable to create upload staging file") from exc
        return path

    def _truncate_stage(self, path: Path, size: int) -> None:
        if path.is_symlink() or not path.exists() or not path.is_file():
            raise AssetStoreError(500, "Committed staging file is missing or unsafe")
        try:
            with path.open("r+b") as stream:
                stream.truncate(size)
                self._fsync_file(stream)
        except OSError as exc:
            raise AssetStoreError(500, "Unable to recover the staging file") from exc

    def _recover_uncommitted_tails(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT upload_id, received FROM uploads WHERE state = 'receiving'"
            ).fetchall()
            for row in rows:
                path = self._stage_path(row["upload_id"])
                committed = row["received"]
                if not path.exists():
                    if committed != 0:
                        raise AssetStoreError(500, "Committed staging file is missing")
                    self._create_empty_stage_file(row["upload_id"])
                    continue
                file_size = path.stat().st_size
                if file_size < committed:
                    raise AssetStoreError(500, "Staging file is shorter than its committed offset")
                if file_size > committed:
                    self._truncate_stage(path, committed)
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _public_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "upload_id": row["upload_id"],
            "asset_id": row["asset_id"],
            "state": row["state"],
            "received": row["received"],
            "total": row["total"],
            "filename": row["filename"],
            "sha256": row["sha256"],
            "reference": _reference(row["asset_id"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            **({"failure_reason": row["failure_reason"]} if row["failure_reason"] else {}),
        }

    def begin(
        self,
        principal_id: str,
        filename: str,
        size: int,
        sha256: str,
    ) -> dict[str, Any]:
        """Reserve a bounded upload and create its private staging file."""
        principal_id = _required_text(principal_id, "principal_id")
        filename = _safe_basename(filename)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise AssetStoreError(400, "size must be a non-negative integer")
        if size > self.max_file_bytes:
            raise AssetStoreError(413, "Asset exceeds the configured file-size limit")
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            raise AssetStoreError(400, "sha256 must be a 64-character hexadecimal digest")
        sha256 = sha256.lower()
        upload_id, asset_id = str(uuid.uuid4()), str(uuid.uuid4())
        now = self._now()
        connection = self._connect()
        stage_path: Path | None = None
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM uploads WHERE principal_id=? AND filename=? AND total=? AND sha256=? AND state IN ('receiving','ready','imported') ORDER BY created_at DESC LIMIT 1",
                (principal_id, filename, size, sha256),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return self._public_record(existing)
            while connection.execute(
                """SELECT 1 FROM uploads
                   WHERE upload_id IN (?, ?) OR asset_id IN (?, ?) LIMIT 1""",
                (upload_id, asset_id, upload_id, asset_id),
            ).fetchone() is not None:
                upload_id, asset_id = str(uuid.uuid4()), str(uuid.uuid4())
            total_global, bytes_global = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(total), 0) FROM uploads WHERE state IN ('receiving','ready')"
            ).fetchone()
            total_owner, bytes_owner = connection.execute(
                """SELECT COUNT(*), COALESCE(SUM(total), 0) FROM uploads
                   WHERE principal_id = ? AND state IN ('receiving','ready')""",
                (principal_id,),
            ).fetchone()
            if total_owner >= self.max_pending_per_principal:
                connection.rollback()
                raise AssetStoreError(429, "Principal pending-upload quota reached")
            if total_global >= self.max_pending_global:
                connection.rollback()
                raise AssetStoreError(429, "Global pending-upload quota reached")
            if bytes_owner + size > self.max_pending_bytes_per_principal:
                connection.rollback()
                raise AssetStoreError(429, "Principal pending-upload byte quota reached")
            if bytes_global + size > self.max_pending_bytes_global:
                connection.rollback()
                raise AssetStoreError(429, "Global pending-upload byte quota reached")
            stage_path = self._create_empty_stage_file(upload_id)
            connection.execute(
                """INSERT INTO uploads
                   (upload_id, asset_id, principal_id, filename, total, sha256,
                    state, received, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'receiving', 0, ?, ?)""",
                (upload_id, asset_id, principal_id, filename, size, sha256, now, now),
            )
            row = connection.execute(
                "SELECT * FROM uploads WHERE upload_id = ?", (upload_id,)
            ).fetchone()
            connection.commit()
            return self._public_record(row)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            if stage_path is not None:
                # If a commit outcome is ambiguous, retain the stage when the
                # row did persist. Never delete a file from an untrusted path.
                keep_stage = False
                try:
                    check = self._connect()
                    try:
                        keep_stage = check.execute(
                            "SELECT 1 FROM uploads WHERE upload_id = ?", (upload_id,)
                        ).fetchone() is not None
                    finally:
                        check.close()
                except sqlite3.Error:
                    keep_stage = True
                if not keep_stage:
                    self._unlink_stage(stage_path)
            raise
        finally:
            connection.close()

    def status(self, principal_id: str, upload_id: str) -> dict[str, Any]:
        principal_id = _required_text(principal_id, "principal_id")
        upload_id = _canonical_uuid(upload_id, "upload_id")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM uploads WHERE principal_id = ? AND (upload_id = ? OR asset_id = ?)",
                (principal_id, upload_id, upload_id),
            ).fetchone()
            if row is None:
                raise AssetStoreError(404, "Upload not found")
            return self._public_record(row)
        finally:
            connection.close()

    def _owned_upload(
        self, connection: sqlite3.Connection, principal_id: str, upload_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM uploads WHERE principal_id = ? AND upload_id = ?",
            (principal_id, upload_id),
        ).fetchone()
        if row is None:
            raise AssetStoreError(404, "Upload not found")
        return row

    def _read_stage_range(self, path: Path, offset: int, length: int) -> bytes:
        if path.is_symlink() or not path.exists() or not path.is_file():
            raise AssetStoreError(500, "Staging file is missing or unsafe")
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(length)
        except OSError as exc:
            raise AssetStoreError(500, "Unable to read the staging file") from exc
        if len(data) != length:
            raise AssetStoreError(500, "Staging file is shorter than its committed chunk")
        return data

    def write_chunk(
        self,
        principal_id: str,
        upload_id: str,
        offset: int,
        data: bytes | bytearray | memoryview,
        total: int | None = None,
    ) -> dict[str, Any]:
        """Commit one sequential chunk, or accept an identical committed retry."""
        principal_id = _required_text(principal_id, "principal_id")
        upload_id = _canonical_uuid(upload_id, "upload_id")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AssetStoreError(400, "offset must be a non-negative integer")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise AssetStoreError(400, "chunk data must be bytes")
        chunk = bytes(data)
        if not chunk:
            raise AssetStoreError(400, "chunk data must not be empty")
        if len(chunk) > self.max_chunk_bytes:
            raise AssetStoreError(413, "Chunk exceeds the configured size limit")
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
            raise AssetStoreError(400, "total must be a non-negative integer")
        chunk_hash = hashlib.sha256(chunk).hexdigest()
        connection = self._connect()
        attempted_write = False
        old_received: int | None = None
        stage_path: Path | None = None
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._owned_upload(connection, principal_id, upload_id)
            old_received = row["received"]
            if total is not None and total != row["total"]:
                raise AssetStoreError(409, "Chunk total does not match the reserved upload size")
            if row["state"] not in {"receiving", "ready", "imported"}:
                raise AssetStoreError(409, "Upload no longer accepts chunks")
            if offset < old_received:
                previous = connection.execute(
                    "SELECT length, sha256 FROM chunks WHERE upload_id = ? AND offset = ?",
                    (upload_id, offset),
                ).fetchone()
                if previous is None or previous["length"] != len(chunk) or previous["sha256"] != chunk_hash:
                    raise AssetStoreError(409, "Retry does not match a committed chunk")
                stage_path = self._stage_path(upload_id)
                if stage_path.exists():
                    stored = self._read_stage_range(stage_path, offset, len(chunk))
                    if stored != chunk:
                        raise AssetStoreError(409, "Retry bytes differ from the committed chunk")
                elif row["state"] != "imported":
                    raise AssetStoreError(500, "Committed staging file is missing")
                result = self._public_record(row)
                connection.commit()
                return result
            if offset != old_received:
                raise AssetStoreError(409, "Chunk offset does not match the committed position")
            if row["state"] != "receiving":
                raise AssetStoreError(409, "Upload no longer accepts new chunks")
            next_received = old_received + len(chunk)
            if next_received > row["total"]:
                raise AssetStoreError(413, "Chunk exceeds the reserved upload size")

            stage_path = self._stage_path(upload_id)
            if not stage_path.exists():
                if old_received != 0:
                    raise AssetStoreError(500, "Committed staging file is missing")
                self._create_empty_stage_file(upload_id)
            file_size = stage_path.stat().st_size
            if file_size < old_received:
                raise AssetStoreError(500, "Staging file is shorter than its committed offset")
            if file_size > old_received:
                self._truncate_stage(stage_path, old_received)
            attempted_write = True
            try:
                with stage_path.open("r+b") as stream:
                    stream.seek(old_received)
                    written = stream.write(chunk)
                    if written != len(chunk):
                        raise OSError("short write")
                    self._fsync_file(stream)
                self._secure_permissions(stage_path, directory=False)
            except OSError as exc:
                raise AssetStoreError(500, "Unable to persist upload chunk") from exc
            connection.execute(
                "INSERT INTO chunks (upload_id, offset, length, sha256) VALUES (?, ?, ?, ?)",
                (upload_id, old_received, len(chunk), chunk_hash),
            )
            now = self._now()
            connection.execute(
                "UPDATE uploads SET received = ?, updated_at = ? WHERE upload_id = ?",
                (next_received, now, upload_id),
            )
            result_row = connection.execute(
                "SELECT * FROM uploads WHERE upload_id = ?", (upload_id,)
            ).fetchone()
            connection.commit()
            return self._public_record(result_row)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            if attempted_write and stage_path is not None and old_received is not None:
                committed = old_received
                try:
                    check = self._connect()
                    try:
                        row = check.execute(
                            "SELECT received FROM uploads WHERE upload_id = ?", (upload_id,)
                        ).fetchone()
                        if row is not None:
                            committed = row["received"]
                    finally:
                        check.close()
                except sqlite3.Error:
                    committed = old_received
                try:
                    if stage_path.exists() and not stage_path.is_symlink():
                        self._truncate_stage(stage_path, committed)
                except AssetStoreError:
                    pass
            raise
        finally:
            connection.close()

    def finish(self, principal_id: str, upload_id: str) -> tuple[dict[str, Any], Path]:
        """Verify the complete staged file hash and mark the upload ready."""
        principal_id = _required_text(principal_id, "principal_id")
        upload_id = _canonical_uuid(upload_id, "upload_id")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._owned_upload(connection, principal_id, upload_id)
            if row["state"] not in {"receiving", "ready"}:
                raise AssetStoreError(409, "Upload cannot be finished in its current state")
            if row["received"] != row["total"]:
                raise AssetStoreError(409, "Upload is incomplete")
            snapshot = dict(row)
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

        path = self._stage_path(upload_id)
        if path.is_symlink() or not path.exists() or not path.is_file():
            raise AssetStoreError(500, "Completed staging file is missing or unsafe")
        if path.stat().st_size != snapshot["total"]:
            raise AssetStoreError(500, "Staging file length differs from committed upload size")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise AssetStoreError(500, "Unable to verify staged asset") from exc
        if digest.hexdigest() != snapshot["sha256"]:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                latest = self._owned_upload(connection, principal_id, upload_id)
                if latest["state"] in {"receiving", "ready"}:
                    connection.execute(
                        "UPDATE uploads SET state = 'failed', failure_reason = ?, updated_at = ? WHERE upload_id = ?",
                        ("uploaded bytes did not match the reserved SHA-256", self._now(), upload_id),
                    )
                connection.commit()
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()
            raise AssetStoreError(422, "Uploaded bytes do not match the reserved SHA-256")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            latest = self._owned_upload(connection, principal_id, upload_id)
            if latest["state"] == "receiving":
                if latest["received"] != latest["total"]:
                    raise AssetStoreError(409, "Upload changed while it was being verified")
                connection.execute(
                    "UPDATE uploads SET state = 'ready', updated_at = ? WHERE upload_id = ?",
                    (self._now(), upload_id),
                )
            elif latest["state"] != "ready":
                raise AssetStoreError(409, "Upload changed while it was being verified")
            result = self._public_record(connection.execute(
                "SELECT * FROM uploads WHERE upload_id = ?", (upload_id,)
            ).fetchone())
            connection.commit()
            return result, path
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _allowed_backend_path(self, backend_path: str | os.PathLike[str]) -> tuple[Path, Path]:
        try:
            raw_path = os.fspath(backend_path)
        except TypeError as exc:
            raise AssetStoreError(400, "backend_path must be a filesystem path") from exc
        if not isinstance(raw_path, str) or not raw_path or len(raw_path) > 32768:
            raise AssetStoreError(400, "backend_path is empty or too long")
        if "\x00" in raw_path or any(ord(char) < 32 or ord(char) == 127 for char in raw_path):
            raise AssetStoreError(400, "backend_path contains control characters")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            raise AssetStoreError(400, "backend_path must be absolute")
        if ".." in PurePath(raw_path).parts:
            raise AssetStoreError(400, "backend_path must not contain parent traversal")
        lexical = Path(os.path.abspath(candidate))
        app_root = self.app_root
        for root_name in _ALLOWED_BACKEND_ROOTS:
            lexical_root = app_root / root_name
            try:
                lexical.relative_to(lexical_root)
            except ValueError:
                continue
            if not lexical_root.exists():
                raise AssetStoreError(403, "Backend path is outside an approved storage root")
            try:
                resolved_root = lexical_root.resolve(strict=True)
                resolved = lexical.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise AssetStoreError(400, "backend_path does not resolve to an existing file") from exc
            if not resolved_root.is_dir() or not resolved.is_file():
                raise AssetStoreError(400, "backend_path must identify a file in an approved storage root")
            try:
                resolved.relative_to(resolved_root)
            except ValueError as exc:
                raise AssetStoreError(403, "Backend path escapes its approved storage root") from exc
            return lexical, resolved
        raise AssetStoreError(403, "Backend path is outside uploads, outputs, and storage")

    def register(
        self,
        principal_id: str,
        upload_id: str,
        backend_path: str | os.PathLike[str],
    ) -> str:
        """Register a trusted backend-import path and return its asset reference."""
        principal_id = _required_text(principal_id, "principal_id")
        upload_id = _canonical_uuid(upload_id, "upload_id")
        check = self._connect()
        try:
            self._owned_upload(check, principal_id, upload_id)
        finally:
            check.close()
        lexical_path, _ = self._allowed_backend_path(backend_path)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._owned_upload(connection, principal_id, upload_id)
            if row["state"] == "imported":
                if row["backend_path"] != str(lexical_path):
                    raise AssetStoreError(409, "Asset is already registered to another backend path")
                asset_id = row["asset_id"]
                connection.commit()
            else:
                if row["state"] != "ready":
                    raise AssetStoreError(409, "Asset must be verified before backend registration")
                asset_id = row["asset_id"]
                connection.execute(
                    "UPDATE uploads SET state = 'imported', backend_path = ?, updated_at = ? WHERE upload_id = ?",
                    (str(lexical_path), self._now(), upload_id),
                )
                connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        stage_path = self._stage_path(upload_id)
        self._unlink_stage(stage_path)
        self._fsync_directory(self.staging_directory)
        return _reference(asset_id)

    def resolve(self, principal_id: str, reference: str) -> Path:
        """Resolve an exact owned asset reference to a currently safe backend path."""
        principal_id = _required_text(principal_id, "principal_id")
        asset_id = self._parse_reference(reference)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM uploads WHERE principal_id = ? AND asset_id = ?",
                (principal_id, asset_id),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise AssetStoreError(404, "Asset not found")
        if row["state"] != "imported" or not row["backend_path"]:
            raise AssetStoreError(409, "Asset has not been imported into the backend")
        _, resolved = self._allowed_backend_path(row["backend_path"])
        return resolved

    def reference_for_path(
        self,
        principal_id: str,
        backend_path: str | os.PathLike[str],
    ) -> str | None:
        """Return this principal's registered reference for an exact backend file.

        Preparation responses can echo an uploaded file's trusted local path
        after the request boundary resolves its asset ID. This owner-scoped
        reverse lookup lets those responses return the usable asset reference
        instead of exposing the machine-local path.
        """
        principal_id = _required_text(principal_id, "principal_id")
        try:
            lexical, resolved = self._allowed_backend_path(backend_path)
        except AssetStoreError:
            return None
        path_keys = {
            os.path.normcase(str(lexical)),
            os.path.normcase(str(resolved)),
        }
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT asset_id, backend_path FROM uploads
                   WHERE principal_id = ? AND state = 'imported'
                   ORDER BY created_at, asset_id""",
                (principal_id,),
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            try:
                registered_lexical, registered_resolved = self._allowed_backend_path(row["backend_path"])
            except AssetStoreError:
                continue
            if path_keys.intersection({
                os.path.normcase(str(registered_lexical)),
                os.path.normcase(str(registered_resolved)),
            }):
                return _reference(row["asset_id"])
        return None

    @staticmethod
    def _parse_reference(reference: Any) -> str:
        if not isinstance(reference, str):
            raise AssetStoreError(400, "Asset reference must be a string")
        match = _REFERENCE_RE.fullmatch(reference)
        if match is None:
            raise AssetStoreError(400, "Invalid asset reference")
        return _canonical_uuid(match.group(1), "asset_id")

    def resolve_tree(self, principal_id: str, payload: Any) -> Any:
        """Copy a JSON-like payload, resolving exact owned asset-reference values."""
        principal_id = _required_text(principal_id, "principal_id")
        budget = [10_000]

        def visit(value: Any, depth: int) -> Any:
            budget[0] -= 1
            if budget[0] < 0 or depth > 32:
                raise AssetStoreError(413, "Asset reference tree exceeds the configured complexity limit")
            if isinstance(value, str):
                if value.startswith("asset:") or value.startswith("maestro-asset:"):
                    return str(self.resolve(principal_id, value))
                return value
            if isinstance(value, Mapping):
                return {key: visit(item, depth + 1) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [visit(item, depth + 1) for item in value]
            return value

        return visit(payload, 0)

    def cleanup_expired(self, max_age_seconds: float) -> int:
        """Explicitly expire old unfinished uploads and unlink only stage files."""
        if isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, (int, float)):
            raise AssetStoreError(400, "max_age_seconds must be a positive number")
        if not math.isfinite(float(max_age_seconds)) or max_age_seconds < 0:
            raise AssetStoreError(400, "max_age_seconds must be a finite non-negative number")
        cutoff = self._now() - float(max_age_seconds)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT upload_id FROM uploads
                   WHERE state IN ('receiving','ready','failed','expired') AND updated_at <= ?""",
                (cutoff,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE uploads SET state = 'expired' WHERE upload_id = ?",
                    (row["upload_id"],),
                )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        removed = 0
        for row in rows:
            path = self._stage_path(row["upload_id"])
            if self._unlink_stage(path):
                removed += 1
        if rows:
            self._fsync_directory(self.staging_directory)
        return removed

    def _unlink_stage(self, path: Path) -> bool:
        try:
            parent = path.parent.resolve(strict=True)
            expected = self.staging_directory.resolve(strict=True)
            if parent != expected or path.is_symlink():
                raise AssetStoreError(500, "Refusing to remove an unsafe staging path")
            if not path.exists():
                return False
            if not path.is_file():
                raise AssetStoreError(500, "Refusing to remove a non-file staging path")
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise AssetStoreError(500, "Unable to remove staging file") from exc


__all__ = ["AssetStore", "AssetStoreError"]
