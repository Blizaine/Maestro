"""Durable idempotency receipts and privacy-bounded agent job snapshots.

The journal records a reservation before a caller dispatches work. A repeated
reservation returns its existing state, so a caller must only dispatch when
``reserve()`` returns ``state == "new"``. A pending reservation is deliberately
not retried after a restart: a crash may have happened after backend dispatch
and before a receipt was committed.

This module never starts, stops, or otherwise controls backend processes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from collections.abc import Mapping
from typing import Any, Callable
import uuid

from .context import submission_context


class JournalError(Exception):
    """An expected journal error with an HTTP-friendly status code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)


_SAFE_RESPONSE_HEADERS = frozenset({
    "cache-control",
    "content-disposition",
    "content-type",
    "etag",
    "last-modified",
    "location",
    "retry-after",
    "vary",
})
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_TERMINAL_STATES = frozenset({
    "cancelled", "canceled", "complete", "completed", "done", "error",
    "failed", "finished", "interrupted", "stopped", "success", "succeeded",
})
_JOB_FIELDS = frozenset({
    "id",
    "job_id",
    "type",
    "kind",
    "task",
    "title",
    "status",
    "state",
    "progress",
    "percentage",
    "message",
    "phase",
    "error",
    "created_at",
    "updated_at",
    "started_at",
    "completed_at",
    "finished_at",
    "duration",
    "eta_seconds",
    "queue_position",
    "current_step",
    "steps_completed",
    "total_steps",
    "output_file",
    "output_files",
    "output_paths",
    "output_dir",
    "output_metadata",
    "outputs",
    "result_files",
    "result_metadata",
    "filename",
})
_PRIVATE_FIELD_PARTS = (
    "api_key", "authorization", "checkpoint", "cookie", "credential",
    "password", "secret", "state_dict", "tensor", "token", "weight",
    "model_state",
)
_OMIT = object()


def _canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return encoded.encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as exc:
        raise JournalError(400, "Request payload must be JSON-safe") from exc


def _required_text(value: Any, name: str, max_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise JournalError(400, f"{name} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise JournalError(400, f"{name} must be valid UTF-8") from exc
    if len(encoded) > max_bytes:
        raise JournalError(400, f"{name} is too long")
    return value


def _optional_text(value: Any, name: str, max_bytes: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise JournalError(400, f"{name} must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise JournalError(400, f"{name} must be valid UTF-8") from exc
    if len(encoded) > max_bytes:
        raise JournalError(400, f"{name} is too long")
    return value


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class SubmissionJournal:
    """SQLite-backed, per-principal idempotency journal and job snapshot store.

    ``directory`` must be a private settings directory controlled by the
    caller. The journal stores principal identifiers and SHA-256 hashes for
    submissions, never the submitted body or credentials. It does not evict
    records; when ``max_entries`` total submission and job records is reached,
    new records are rejected.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        clock: Callable[[], float] | None = None,
        *,
        max_entries: int = 10_000,
        max_key_bytes: int = 256,
        max_response_bytes: int = 1_048_576,
        max_job_snapshot_bytes: int = 262_144,
    ):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if isinstance(max_key_bytes, bool) or not isinstance(max_key_bytes, int) or max_key_bytes < 1:
            raise ValueError("max_key_bytes must be a positive integer")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes < 0:
            raise ValueError("max_response_bytes must be a non-negative integer")
        if isinstance(max_job_snapshot_bytes, bool) or not isinstance(max_job_snapshot_bytes, int) or max_job_snapshot_bytes < 1:
            raise ValueError("max_job_snapshot_bytes must be a positive integer")

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database_path = self.directory / "agent-access-journal.sqlite3"
        self._clock = clock or time.time
        self.max_entries = max_entries
        self.max_key_bytes = max_key_bytes
        self.max_response_bytes = max_response_bytes
        self.max_job_snapshot_bytes = max_job_snapshot_bytes
        self._initialize()

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
                CREATE TABLE IF NOT EXISTS submissions (
                    reservation_id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'complete', 'uncertain')),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    response_status INTEGER,
                    response_body BLOB,
                    response_headers TEXT,
                    uncertain_reason TEXT,
                    UNIQUE (principal_id, key_hash)
                );
                CREATE INDEX IF NOT EXISTS submissions_principal_created
                    ON submissions (principal_id, created_at DESC, reservation_id);

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    session_id TEXT,
                    snapshot TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_principal_updated
                    ON jobs (principal_id, updated_at DESC, job_id);
                CREATE INDEX IF NOT EXISTS jobs_session_updated
                    ON jobs (session_id, updated_at DESC, job_id);
                CREATE TABLE IF NOT EXISTS resource_owners (
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    session_id TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (resource_type, resource_id)
                );
                CREATE TABLE IF NOT EXISTS submission_resources (
                    reservation_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    resource_kind TEXT NOT NULL CHECK (resource_kind IN ('job', 'pipeline', 'queue')),
                    resource_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (resource_kind, resource_id),
                    FOREIGN KEY (reservation_id) REFERENCES submissions(reservation_id)
                );
                CREATE INDEX IF NOT EXISTS submission_resources_reservation
                    ON submission_resources (reservation_id, resource_kind, resource_id);
                """
            )
        finally:
            connection.close()
        # This is a best-effort extra guard. The caller is responsible for
        # providing a private directory, and Windows may ignore POSIX modes.
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError) as exc:
            raise JournalError(500, "Journal clock did not return a timestamp") from exc
        if not math.isfinite(value):
            raise JournalError(500, "Journal clock did not return a finite timestamp")
        return value

    def _link_submission_resource(
        self,
        connection: sqlite3.Connection,
        principal_id: str,
        resource_kind: str,
        resource_id: str,
    ) -> None:
        """Attach a newly observed resource to the active matching submission.

        The caller owns an open write transaction. The unique resource key
        preserves the first correlation if later snapshots repeat or update
        the same resource under a different submission context.
        """
        reservation_id = submission_context.get()
        if reservation_id is None:
            return
        reservation_id = _required_text(reservation_id, "reservation_id", 64)
        reservation = connection.execute(
            "SELECT principal_id FROM submissions WHERE reservation_id = ?",
            (reservation_id,),
        ).fetchone()
        if reservation is None or reservation["principal_id"] != principal_id:
            return
        existing = connection.execute(
            "SELECT reservation_id FROM submission_resources WHERE resource_kind = ? AND resource_id = ?",
            (resource_kind, resource_id),
        ).fetchone()
        if existing is not None:
            return
        count = connection.execute(
            "SELECT COUNT(*) FROM submission_resources"
        ).fetchone()[0]
        if count >= self.max_entries:
            raise JournalError(429, "Submission resource linkage capacity reached")
        connection.execute(
            """INSERT INTO submission_resources
               (reservation_id, principal_id, resource_kind, resource_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (reservation_id, principal_id, resource_kind, resource_id, self._now()),
        )

    @staticmethod
    def _reservation(row: sqlite3.Row, *, include_body: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "reservation_id": row["reservation_id"],
            "state": row["state"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if row["state"] == "complete":
            result["status_code"] = row["response_status"]
            result["body_bytes"] = len(row["response_body"] or b"")
            result["headers"] = json.loads(row["response_headers"] or "[]")
            if include_body:
                result["body"] = bytes(row["response_body"] or b"")
        elif row["state"] == "uncertain" and row["uncertain_reason"]:
            result["reason"] = row["uncertain_reason"]
        return result

    def reserve(
        self,
        principal_id: str,
        key: str,
        method: str,
        path: str,
        payload: Any,
        query: str = "",
    ) -> dict[str, Any]:
        """Atomically reserve an idempotency key for one canonical request.

        Only the first caller gets ``state == "new"``. Callers must not
        dispatch work for pending, uncertain, or complete reservations.
        """
        principal_id = _required_text(principal_id, "principal_id", 512)
        key = _required_text(key, "key", self.max_key_bytes)
        method = _required_text(method, "method", 32).upper()
        if not _HEADER_NAME_RE.fullmatch(method):
            raise JournalError(400, "method must be a valid HTTP token")
        path = _required_text(path, "path", 8192)
        query = _optional_text(query, "query", 16_384)

        key_hash = _digest(key.encode("utf-8"))
        if isinstance(payload, (bytes, bytearray, memoryview)):
            raw = bytes(payload)
            canonical_payload: Any = {
                "raw_body_sha256": _digest(raw),
                "raw_body_bytes": len(raw),
            }
        else:
            # Serialize once to validate input before hashing it. The content
            # itself remains in memory only and is never inserted in SQLite.
            canonical_payload = json.loads(_canonical_json(payload))
        request_hash = _digest(_canonical_json({
            "principal_id": principal_id,
            "key_sha256": key_hash,
            "method": method,
            "path": path,
            "query": query,
            "payload": canonical_payload,
        }))

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM submissions WHERE principal_id = ? AND key_hash = ?",
                (principal_id, key_hash),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    connection.rollback()
                    raise JournalError(409, "Idempotency key was already used for a different request")
                result = self._reservation(existing)
                connection.commit()
                if result["state"] == "pending":
                    result["state"] = "pending"
                return result

            count = (
                connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
                + connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                + connection.execute("SELECT COUNT(*) FROM resource_owners").fetchone()[0]
            )
            if count >= self.max_entries:
                connection.rollback()
                raise JournalError(429, "Idempotency journal capacity reached")
            now = self._now()
            reservation_id = str(uuid.uuid4())
            connection.execute(
                """INSERT INTO submissions
                   (reservation_id, principal_id, key_hash, request_hash, state, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                (reservation_id, principal_id, key_hash, request_hash, now, now),
            )
            row = connection.execute(
                "SELECT * FROM submissions WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            connection.commit()
            result = self._reservation(row, include_body=False)
            result["state"] = "new"
            return result
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def complete(
        self,
        reservation_id: str,
        status_code: int,
        body: bytes | bytearray | memoryview | Any,
        headers: Mapping[str, Any] | list[tuple[str, Any]] | tuple[tuple[str, Any], ...] = (),
    ) -> dict[str, Any]:
        """Persist a bounded response receipt for a pending reservation."""
        reservation_id = _required_text(reservation_id, "reservation_id", 64)
        if isinstance(status_code, bool) or not isinstance(status_code, int) or not 100 <= status_code <= 599:
            raise JournalError(400, "status_code must be an HTTP status code")
        if isinstance(body, (bytes, bytearray, memoryview)):
            body_bytes = bytes(body)
        else:
            body_bytes = _canonical_json(body)
        if len(body_bytes) > self.max_response_bytes:
            raise JournalError(413, "Response receipt exceeds the configured size limit")
        safe_headers = self._safe_headers(headers)
        headers_json = _canonical_json(safe_headers).decode("utf-8")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM submissions WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise JournalError(404, "Reservation not found")
            if row["state"] == "complete":
                same = (
                    row["response_status"] == status_code
                    and bytes(row["response_body"] or b"") == body_bytes
                    and row["response_headers"] == headers_json
                )
                if not same:
                    connection.rollback()
                    raise JournalError(409, "Reservation already has a different response receipt")
                result = self._reservation(row)
                connection.commit()
                return result

            now = self._now()
            connection.execute(
                """UPDATE submissions SET state = 'complete', updated_at = ?,
                       response_status = ?, response_body = ?, response_headers = ?,
                       uncertain_reason = NULL
                   WHERE reservation_id = ?""",
                (now, status_code, sqlite3.Binary(body_bytes), headers_json, reservation_id),
            )
            updated = connection.execute(
                "SELECT * FROM submissions WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            connection.commit()
            return self._reservation(updated)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _safe_headers(
        headers: Mapping[str, Any] | list[tuple[str, Any]] | tuple[tuple[str, Any], ...],
    ) -> list[list[str]]:
        if isinstance(headers, Mapping):
            items = list(headers.items())
        elif isinstance(headers, (list, tuple)):
            items = list(headers)
        else:
            raise JournalError(400, "headers must be a mapping or a list of pairs")
        if len(items) > 64:
            raise JournalError(400, "Too many response headers")
        safe: list[list[str]] = []
        total_bytes = 0
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise JournalError(400, "Each response header must be a name/value pair")
            name, value = item
            if not isinstance(name, str) or not _HEADER_NAME_RE.fullmatch(name):
                raise JournalError(400, "Invalid response header name")
            if not isinstance(value, str) or "\r" in value or "\n" in value:
                raise JournalError(400, "Invalid response header value")
            normalized_name = name.lower()
            if normalized_name not in _SAFE_RESPONSE_HEADERS:
                continue
            try:
                value_bytes = value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise JournalError(400, "Invalid response header value") from exc
            if len(value_bytes) > 2048:
                raise JournalError(400, "Response header value is too long")
            total_bytes += len(normalized_name.encode("ascii")) + len(value_bytes)
            safe.append([normalized_name, value])
        if total_bytes > 8192:
            raise JournalError(400, "Response headers exceed the configured size limit")
        safe.sort(key=lambda pair: (pair[0], pair[1]))
        return safe

    def fail_uncertain(self, reservation_id: str, reason: str) -> dict[str, Any]:
        """Mark a reservation uncertain without making its key reusable."""
        reservation_id = _required_text(reservation_id, "reservation_id", 64)
        if not isinstance(reason, str) or not reason.strip():
            raise JournalError(400, "reason must be a non-empty string")
        reason = " ".join(reason.split())[:512]
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM submissions WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise JournalError(404, "Reservation not found")
            if row["state"] == "pending":
                now = self._now()
                connection.execute(
                    """UPDATE submissions SET state = 'uncertain', updated_at = ?,
                           uncertain_reason = ? WHERE reservation_id = ? AND state = 'pending'""",
                    (now, reason, reservation_id),
                )
                row = connection.execute(
                    "SELECT * FROM submissions WHERE reservation_id = ?", (reservation_id,)
                ).fetchone()
            result = self._reservation(row)
            connection.commit()
            return result
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def lookup(self, principal_id: str, reservation_id: str) -> dict[str, Any] | None:
        """Return one reservation only to its owning authenticated principal."""
        principal_id = _required_text(principal_id, "principal_id", 512)
        reservation_id = _required_text(reservation_id, "reservation_id", 64)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM submissions WHERE principal_id = ? AND reservation_id = ?",
                (principal_id, reservation_id),
            ).fetchone()
            return None if row is None else self._reservation(row)
        finally:
            connection.close()

    def find_submission(self, principal_id: str, request_id: str) -> dict[str, Any] | None:
        """Find an idempotency receipt by caller key without exposing request data.

        The raw key is never stored. This applies the same exact UTF-8 SHA-256
        digest used by :meth:`reserve`, scoped to the authenticated principal.
        A completed JSON response is decoded for receipt recovery; non-JSON
        response bytes are never returned.
        """
        principal_id = _required_text(principal_id, "principal_id", 512)
        request_id = _required_text(request_id, "request_id", self.max_key_bytes)
        key_hash = _digest(request_id.encode("utf-8"))
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM submissions WHERE principal_id = ? AND key_hash = ?",
                (principal_id, key_hash),
            ).fetchone()
            if row is None:
                return None
            result = {
                "reservation_id": row["reservation_id"],
                "state": row["state"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            resources = connection.execute(
                """SELECT sr.resource_kind, sr.resource_id
                   FROM submission_resources AS sr
                   JOIN submissions AS s ON s.reservation_id = sr.reservation_id
                   WHERE s.reservation_id = ? AND s.principal_id = ?
                     AND sr.principal_id = s.principal_id
                   ORDER BY sr.resource_kind, sr.resource_id""",
                (row["reservation_id"], principal_id),
            ).fetchall()
            result["resources"] = [
                {"kind": resource["resource_kind"], "id": resource["resource_id"]}
                for resource in resources
            ]
            if row["state"] == "complete":
                result["status_code"] = row["response_status"]
                result["headers"] = json.loads(row["response_headers"] or "[]")
                body = bytes(row["response_body"] or b"")
                try:
                    result["result"] = json.loads(body)
                    result["result_available"] = True
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError):
                    result["result_available"] = False
            elif row["state"] == "uncertain" and row["uncertain_reason"]:
                result["reason"] = row["uncertain_reason"]
            return result
        finally:
            connection.close()

    def list_reservations(
        self, principal_id: str, *, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List reservation summaries for one principal, newest first."""
        principal_id = _required_text(principal_id, "principal_id", 512)
        limit, offset = self._page(limit, offset)
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT * FROM submissions WHERE principal_id = ?
                   ORDER BY created_at DESC, reservation_id LIMIT ? OFFSET ?""",
                (principal_id, limit, offset),
            ).fetchall()
            return [self._reservation(row, include_body=False) for row in rows]
        finally:
            connection.close()

    def record_job(
        self,
        job: Mapping[str, Any],
        principal_id: str,
        session_id: str | None,
    ) -> dict[str, Any]:
        """Persist a JSON-safe metadata snapshot with immutable ownership."""
        if not isinstance(job, Mapping):
            raise JournalError(400, "job must be a mapping")
        principal_id = _required_text(principal_id, "principal_id", 512)
        session_id = None if session_id is None else _required_text(session_id, "session_id", 512)
        raw_job_id = job.get("id", job.get("job_id"))
        job_id = _required_text(raw_job_id, "job_id", 512)
        snapshot = self._job_snapshot(job, job_id)
        snapshot_json = _canonical_json(snapshot).decode("utf-8")
        if len(snapshot_json.encode("utf-8")) > self.max_job_snapshot_bytes:
            raise JournalError(413, "Job metadata exceeds the configured size limit")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            now = self._now()
            if old is None:
                count = (
                    connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
                    + connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                    + connection.execute("SELECT COUNT(*) FROM resource_owners").fetchone()[0]
                )
                if count >= self.max_entries:
                    connection.rollback()
                    raise JournalError(429, "Journal capacity reached")
                connection.execute(
                    """INSERT INTO jobs (job_id, principal_id, session_id, snapshot, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (job_id, principal_id, session_id, snapshot_json, now, now),
                )
                self._link_submission_resource(connection, principal_id, "job", job_id)
            else:
                if old["principal_id"] != principal_id or old["session_id"] != session_id:
                    connection.rollback()
                    raise JournalError(409, "Job owner and session cannot be reassigned")
                connection.execute(
                    "UPDATE jobs SET snapshot = ?, updated_at = ? WHERE job_id = ?",
                    (snapshot_json, now, job_id),
                )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            connection.commit()
            return self._job_record(row)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @classmethod
    def _job_snapshot(cls, job: Mapping[str, Any], job_id: str) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"job_id": job_id}
        for key in _JOB_FIELDS:
            if key in {"id", "job_id"} or key not in job:
                continue
            clean = cls._json_safe(job[key], key=key, depth=0)
            if clean is not _OMIT:
                snapshot[key] = clean
        return snapshot

    @classmethod
    def _json_safe(cls, value: Any, *, key: str | None, depth: int) -> Any:
        if key is not None:
            normalized = key.casefold().replace("-", "_")
            if any(part in normalized for part in _PRIVATE_FIELD_PARTS):
                return _OMIT
        if depth > 8:
            return _OMIT
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else _OMIT
        if isinstance(value, str):
            return value[:4096]
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for index, (nested_key, nested_value) in enumerate(value.items()):
                if index >= 256:
                    break
                if not isinstance(nested_key, str) or len(nested_key) > 128:
                    continue
                clean = cls._json_safe(nested_value, key=nested_key, depth=depth + 1)
                if clean is not _OMIT:
                    result[nested_key] = clean
            return result
        if isinstance(value, (list, tuple)):
            result = []
            for nested_value in value[:512]:
                clean = cls._json_safe(nested_value, key=None, depth=depth + 1)
                if clean is not _OMIT:
                    result.append(clean)
            return result
        # Framework tensors, model instances, bytes, and arbitrary objects are
        # intentionally not stringified or serialized into durable metadata.
        return _OMIT

    @staticmethod
    def _job_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "principal_id": row["principal_id"],
            "session_id": row["session_id"],
            "snapshot": json.loads(row["snapshot"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_job(self, job_id: str, principal_id: str) -> dict[str, Any] | None:
        """Get a persisted job snapshot only when the principal owns it."""
        job_id = _required_text(job_id, "job_id", 512)
        principal_id = _required_text(principal_id, "principal_id", 512)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ? AND principal_id = ?",
                (job_id, principal_id),
            ).fetchone()
            return None if row is None else self._job_record(row)
        finally:
            connection.close()

    def get_job_any(self, job_id: str) -> dict[str, Any] | None:
        """Internal unscoped lookup for a caller that already authorized an admin."""
        job_id = _required_text(job_id, "job_id", 512)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return None if row is None else self._job_record(row)
        finally:
            connection.close()

    def register_resource_owner(
        self,
        resource_type: str,
        resource_id: str,
        principal_id: str,
        session_id: str | None,
    ) -> dict[str, str | None]:
        """Persist immutable ownership for a trusted server-created resource."""
        if resource_type not in {"pipeline", "queue"}:
            raise JournalError(400, "Unsupported resource type")
        resource_id = _required_text(resource_id, "resource_id", 512)
        principal_id = _required_text(principal_id, "principal_id", 512)
        session_id = None if session_id is None else _required_text(session_id, "session_id", 512)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT principal_id, session_id FROM resource_owners WHERE resource_type = ? AND resource_id = ?",
                (resource_type, resource_id),
            ).fetchone()
            if row is None:
                count = (
                    connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
                    + connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                    + connection.execute("SELECT COUNT(*) FROM resource_owners").fetchone()[0]
                )
                if count >= self.max_entries:
                    connection.rollback()
                    raise JournalError(429, "Journal capacity reached")
                connection.execute(
                    "INSERT INTO resource_owners (resource_type, resource_id, principal_id, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (resource_type, resource_id, principal_id, session_id, self._now()),
                )
                self._link_submission_resource(connection, principal_id, resource_type, resource_id)
            elif row["principal_id"] != principal_id or row["session_id"] != session_id:
                connection.rollback()
                raise JournalError(409, "Resource owner and session cannot be reassigned")
            connection.commit()
            return {"principal_id": principal_id, "session_id": session_id}
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def resource_owner(self, resource_type: str, resource_id: str) -> dict[str, str | None] | None:
        """Read ownership for a resource ID supplied by trusted backend state."""
        if resource_type not in {"pipeline", "queue"}:
            return None
        try:
            resource_id = _required_text(resource_id, "resource_id", 512)
        except JournalError:
            return None
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT principal_id, session_id FROM resource_owners WHERE resource_type = ? AND resource_id = ?",
                (resource_type, resource_id),
            ).fetchone()
            return None if row is None else {
                "principal_id": row["principal_id"],
                "session_id": row["session_id"],
            }
        finally:
            connection.close()

    def list_jobs(
        self,
        principal_id: str,
        *,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List snapshots for one principal, optionally limited to a session."""
        principal_id = _required_text(principal_id, "principal_id", 512)
        session_id = None if session_id is None else _required_text(session_id, "session_id", 512)
        limit, offset = self._page(limit, offset)
        connection = self._connect()
        try:
            if session_id is None:
                rows = connection.execute(
                    """SELECT * FROM jobs WHERE principal_id = ?
                       ORDER BY updated_at DESC, job_id LIMIT ? OFFSET ?""",
                    (principal_id, limit, offset),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM jobs WHERE principal_id = ? AND session_id = ?
                       ORDER BY updated_at DESC, job_id LIMIT ? OFFSET ?""",
                    (principal_id, session_id, limit, offset),
                ).fetchall()
            return [self._job_record(row) for row in rows]
        finally:
            connection.close()

    def mark_session_interrupted(
        self, session_id: str, verified_stopped: bool = False
    ) -> int:
        """Mark non-terminal records interrupted after verified backend shutdown.

        ``verified_stopped`` must be explicitly true. This only updates the
        journal; it performs no process or backend actions.
        """
        session_id = _required_text(session_id, "session_id", 512)
        if verified_stopped is not True:
            raise JournalError(409, "Session shutdown must be verified before recovery updates")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM jobs WHERE session_id = ?", (session_id,)
            ).fetchall()
            now = self._now()
            changed = 0
            for row in rows:
                snapshot = json.loads(row["snapshot"])
                current_state = snapshot.get("status") or snapshot.get("state", "")
                if isinstance(current_state, str) and current_state.casefold() in _TERMINAL_STATES:
                    continue
                snapshot["status"] = "interrupted"
                snapshot["message"] = "Backend stopped before job completion"
                snapshot.setdefault("completed_at", now)
                encoded = _canonical_json(snapshot).decode("utf-8")
                if len(encoded.encode("utf-8")) > self.max_job_snapshot_bytes:
                    raise JournalError(413, "Job metadata exceeds the configured size limit")
                connection.execute(
                    "UPDATE jobs SET snapshot = ?, updated_at = ? WHERE job_id = ?",
                    (encoded, now, row["job_id"]),
                )
                changed += 1
            connection.commit()
            return changed
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _page(limit: int, offset: int) -> tuple[int, int]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise JournalError(400, "limit must be a positive integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise JournalError(400, "offset must be a non-negative integer")
        return min(limit, 500), offset


class JobRegistry(dict):
    """A dict that captures immutable principal/session ownership on insert.

    Snapshot persistence remains explicit through ``record_snapshot`` or
    ``record_all`` so the parent can choose its terminal listener and periodic
    snapshot policy.
    """

    def __init__(
        self,
        *args: Any,
        current_principal: Callable[[], str | None] | None = None,
        current_session: Callable[[], str | None] | None = None,
        on_insert: Callable[[Any, Any, dict[str, str | None]], None] | None = None,
        owner_resolver: Callable[[Any, Any], dict[str, str | None] | tuple[str | None, str | None] | None] | None = None,
        **kwargs: Any,
    ):
        self._current_principal = current_principal
        self._current_session = current_session
        self._on_insert = on_insert
        self._owner_resolver = owner_resolver
        self._owners: dict[Any, tuple[str | None, str | None]] = {}
        dict.__init__(self)
        if len(args) > 1:
            raise TypeError(f"expected at most 1 argument, got {len(args)}")
        if args:
            self.update(args[0])
        if kwargs:
            self.update(kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        is_new = key not in self
        if key not in self._owners:
            principal_id = self._current_principal() if self._current_principal else None
            session_id = self._current_session() if self._current_session else None
            if principal_id is not None and not isinstance(principal_id, str):
                principal_id = None
            if session_id is not None and not isinstance(session_id, str):
                session_id = None
            if principal_id is None and self._owner_resolver is not None:
                resolved = self._owner_resolver(key, value)
                if isinstance(resolved, Mapping):
                    principal_id = resolved.get("principal_id")
                    session_id = resolved.get("session_id")
                elif isinstance(resolved, tuple) and len(resolved) == 2:
                    principal_id, session_id = resolved
                if principal_id is not None and not isinstance(principal_id, str):
                    principal_id = None
                if session_id is not None and not isinstance(session_id, str):
                    session_id = None
            self._owners[key] = (principal_id, session_id)
        dict.__setitem__(self, key, value)
        if is_new and self._on_insert is not None:
            owner = self.ownership_for(key) or {"principal_id": None, "session_id": None}
            try:
                self._on_insert(key, value, owner)
            except Exception:
                dict.__delitem__(self, key)
                self._owners.pop(key, None)
                raise

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        if isinstance(other, Mapping):
            items = other.items()
        else:
            items = other
        for key, value in items:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other: Any) -> "JobRegistry":
        self.update(other)
        return self

    def ownership_for(self, job_id: Any) -> dict[str, str | None] | None:
        owner = self._owners.get(job_id)
        if owner is None:
            return None
        return {"principal_id": owner[0], "session_id": owner[1]}

    def restore_owner(self, job_id: Any, principal_id: str, session_id: str | None) -> None:
        """Restore the immutable owner from a durable journal during startup."""
        if job_id not in self:
            raise KeyError(job_id)
        principal_id = _required_text(principal_id, "principal_id", 512)
        session_id = None if session_id is None else _required_text(session_id, "session_id", 512)
        old = self._owners.get(job_id)
        if old is not None and old[0] is not None and old != (principal_id, session_id):
            raise JournalError(409, "Job owner and session cannot be reassigned")
        self._owners[job_id] = (principal_id, session_id)

    def bind_owner(self, job_id: Any, principal_id: str, session_id: str | None) -> None:
        """Attach a trusted background owner once, then persist before dispatch."""
        if job_id not in self:
            raise KeyError(job_id)
        principal_id = _required_text(principal_id, "principal_id", 512)
        session_id = None if session_id is None else _required_text(session_id, "session_id", 512)
        old = self._owners.get(job_id)
        if old is not None and old[0] is not None:
            if old != (principal_id, session_id):
                raise JournalError(409, "Job owner and session cannot be reassigned")
            return
        if old is not None and old[1] not in (None, session_id):
            raise JournalError(409, "Job owner and session cannot be reassigned")
        self._owners[job_id] = (principal_id, session_id)
        if self._on_insert is not None:
            try:
                self._on_insert(job_id, self[job_id], self.ownership_for(job_id) or {})
            except Exception:
                self._owners[job_id] = old or (None, None)
                raise

    def record_snapshot(self, journal: SubmissionJournal, job_id: Any) -> dict[str, Any] | None:
        if job_id not in self:
            return None
        owner = self.ownership_for(job_id)
        if owner is None or owner["principal_id"] is None:
            return None
        return journal.record_job(self[job_id], **owner)

    def record_all(self, journal: SubmissionJournal) -> list[dict[str, Any]]:
        snapshots = []
        for job_id in list(self.keys()):
            record = self.record_snapshot(journal, job_id)
            if record is not None:
                snapshots.append(record)
        return snapshots


__all__ = ["JournalError", "SubmissionJournal", "JobRegistry"]
