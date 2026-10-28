"""Durable ownership records for Editor-to-Studio AI round trips.

The service stores intent and job identity only. It never creates, starts, or
cancels jobs; callers bind an operation to the exact job they create.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,160}$")
_TOOLS = {
    "retake",
    "edit_anything",
    "recast",
    "repaint",
    "outpaint",
    "upscale",
    "film_grain",
    "revoice",
}
_STATUSES = {
    "armed",
    "queued",
    "running",
    "result_ready",
    "completed",
    "failed",
    "needs_review",
    "cancelled",
}
_TERMINAL = {"completed", "failed", "needs_review", "cancelled"}
_SOURCE_KEYS = (
    "item_id",
    "track_id",
    "track_type",
    "asset_id",
    "asset_name",
    "asset_path",
    "start",
    "duration",
    "source_in",
    "speed",
)
_SOURCE_COMPARISON_KEYS = (
    "item_id",
    "asset_id",
    "asset_name",
    "asset_path",
    "duration",
    "source_in",
    "speed",
)
_lock = threading.RLock()


class EditorOperationError(ValueError):
    """An Editor operation is invalid, unsafe, or no longer owned."""


def _require_id(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not _ID_RE.fullmatch(result):
        raise EditorOperationError(f"Invalid {label}.")
    return result


def _root_path(root: str | os.PathLike[str]) -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve(strict=True)


def _operation_path(root: str | os.PathLike[str], operation_id: Any) -> Path:
    safe_id = _require_id(operation_id, "operation id")
    resolved_root = _root_path(root)
    candidate = resolved_root / f"{safe_id}.json"
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate.parent != resolved_root:
        raise EditorOperationError("Operation path is outside the configured storage root.")
    return candidate


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    # Keep temporary files in the validated directory so os.replace is atomic
    # and cannot cross a filesystem boundary.
    root = path.parent.resolve(strict=True)
    if path.resolve(strict=False).parent != root:
        raise EditorOperationError("Operation path is outside the configured storage root.")
    descriptor, temp_name = tempfile.mkstemp(prefix=".editor-operation-", suffix=".tmp", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _save(root: str | os.PathLike[str], operation: dict[str, Any]) -> dict[str, Any]:
    path = _operation_path(root, operation.get("operation_id"))
    _atomic_write(path, operation)
    return dict(operation)


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EditorOperationError(f"{label} must be a finite number.") from error
    if not math.isfinite(number):
        raise EditorOperationError(f"{label} must be a finite number.")
    return number


def _normalize_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EditorOperationError("Source snapshot must be an object.")
    result: dict[str, Any] = {}
    for key in _SOURCE_KEYS:
        if key not in value:
            continue
        item = value[key]
        if key in {"start", "duration", "source_in", "speed"}:
            item = _finite_number(item, f"Source snapshot {key}")
        elif item is not None:
            item = str(item)
            if len(item) > 4096:
                raise EditorOperationError("Source snapshot fields are too long.")
        result[key] = item
    if not result.get("item_id") or not result.get("asset_id"):
        raise EditorOperationError("Source snapshot needs item and asset ids.")
    return result


def create_operation(root: str | os.PathLike[str], payload: Any) -> dict[str, Any]:
    """Persist a new armed operation. Repeating an identical create is safe."""
    if not isinstance(payload, dict):
        raise EditorOperationError("Operation request must be an object.")
    operation_id = _require_id(payload.get("operation_id"), "operation id")
    client_submission_id = _require_id(payload.get("client_submission_id"), "client submission id")
    tool = str(payload.get("tool") or "")
    if tool not in _TOOLS:
        raise EditorOperationError("Unsupported Editor AI tool.")
    return_mode = str(payload.get("return_mode") or "")
    if return_mode not in {"replace", "alternate"}:
        raise EditorOperationError("Return mode must be replace or alternate.")
    workspace = str(payload.get("workspace") or "").strip()
    project_id = str(payload.get("project_id") or "").strip()
    item_id = str(payload.get("item_id") or "").strip()
    item_name = str(payload.get("item_name") or "").strip()
    if not workspace or len(workspace) > 512 or not project_id or not item_id:
        raise EditorOperationError("Workspace, project id, and item id are required.")
    source_snapshot = _normalize_snapshot(payload.get("source_snapshot"))
    if source_snapshot["item_id"] != item_id:
        raise EditorOperationError("Source snapshot item id does not match the operation.")
    source_range = payload.get("source_range")
    if not isinstance(source_range, dict):
        raise EditorOperationError("Source range must be an object.")
    range_start = _finite_number(source_range.get("start"), "Source range start")
    range_end = _finite_number(source_range.get("end"), "Source range end")
    if range_start < 0 or range_end <= range_start:
        raise EditorOperationError("Source range must be a positive interval.")

    now = time.time()
    operation = {
        "schema_version": 1,
        "operation_id": operation_id,
        "client_submission_id": client_submission_id,
        "workspace": workspace,
        "project_id": project_id,
        "project_revision": payload.get("project_revision"),
        "item_id": item_id,
        "item_name": item_name[:1024],
        "tool": tool,
        "return_mode": return_mode,
        "source_snapshot": source_snapshot,
        "source_range": {"start": range_start, "end": range_end},
        "status": "armed",
        "job_id": None,
        "job_kind": None,
        "job_status": None,
        "output_file": None,
        "error": None,
        "review_reason": None,
        "created_at": now,
        "updated_at": now,
    }
    with _lock:
        path = _operation_path(root, operation_id)
        if path.exists():
            existing = _read_path(path)
            if (
                existing.get("client_submission_id") == client_submission_id
                and existing.get("workspace") == workspace
                and existing.get("project_id") == project_id
                and existing.get("item_id") == item_id
                and existing.get("tool") == tool
            ):
                return existing
            raise EditorOperationError("An operation with this id already exists.")
        _atomic_write(path, operation)
    return dict(operation)


def _read_path(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as error:
        raise EditorOperationError("Stored Editor operation could not be read.") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise EditorOperationError("Stored Editor operation has an unsupported format.")
    return value


def get_operation(root: str | os.PathLike[str], operation_id: Any) -> dict[str, Any]:
    path = _operation_path(root, operation_id)
    with _lock:
        return _read_path(path)


def validate_submission(
    root: str | os.PathLike[str],
    operation_id: Any,
    client_submission_id: Any,
    *,
    allow_linked: bool = False,
    expected_tool: str | None = None,
) -> dict[str, Any]:
    operation = get_operation(root, operation_id)
    client_id = _require_id(client_submission_id, "client submission id")
    if operation.get("client_submission_id") != client_id:
        raise EditorOperationError("Client submission does not own this Editor operation.")
    if expected_tool is not None and operation.get("tool") != expected_tool:
        raise EditorOperationError("The submitted action does not match this Editor operation's requested tool.")
    if operation.get("status") in _TERMINAL:
        raise EditorOperationError("This Editor operation is already finished.")
    if operation.get("job_id") and not allow_linked:
        raise EditorOperationError("This Editor operation already owns a job.")
    return operation


def attach_job(
    root: str | os.PathLike[str],
    operation_id: Any,
    client_submission_id: Any,
    job_id: Any,
    job_kind: str,
) -> dict[str, Any]:
    safe_job_id = _require_id(job_id, "job id")
    with _lock:
        operation = validate_submission(root, operation_id, client_submission_id)
        normalized_job_kind = str(job_kind or "").strip()
        if normalized_job_kind != operation.get("tool"):
            raise EditorOperationError("The created job kind does not match this Editor operation's requested tool.")
        operation.update(
            status="queued",
            job_id=safe_job_id,
            job_kind=normalized_job_kind[:100],
            job_status="queued",
            updated_at=time.time(),
        )
        return _save(root, operation)


def _job_outputs(job: dict[str, Any]) -> list[str]:
    values = job.get("output_files")
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value) for value in values if isinstance(value, str) and value]


def reconcile_operation(
    root: str | os.PathLike[str],
    operation_id: Any,
    jobs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Refresh from only the operation's linked job, validating both IDs."""
    with _lock:
        operation = get_operation(root, operation_id)
        if operation.get("status") in _TERMINAL or not operation.get("job_id"):
            return operation
        job = jobs.get(str(operation["job_id"]))
        if job is None:
            operation.update(
                status="needs_review",
                review_reason="The linked job is no longer available in the queue.",
                error="The linked job could not be recovered. Its output may still be in the workspace library.",
                updated_at=time.time(),
            )
            return _save(root, operation)
        if (
            operation.get("job_kind") != operation.get("tool")
            or str(job.get("id") or "") != operation["job_id"]
            or job.get("editor_operation_id") != operation["operation_id"]
            or job.get("client_submission_id") != operation["client_submission_id"]
        ):
            operation.update(
                status="needs_review",
                review_reason="The linked job ownership metadata did not match this operation.",
                error="The Editor result could not be verified as belonging to this request.",
                updated_at=time.time(),
            )
            return _save(root, operation)
        raw_status = str(job.get("status") or "")
        status_map = {
            "held": "queued",
            "queued": "queued",
            "running": "running",
            "completed": "result_ready",
            "failed": "failed",
            "cancelled": "cancelled",
        }
        next_status = status_map.get(raw_status)
        if next_status:
            operation["status"] = next_status
        operation["job_status"] = raw_status or None
        if raw_status == "completed":
            outputs = _job_outputs(job)
            operation["output_file"] = outputs[-1] if outputs else None
            operation["error"] = None if outputs else "The linked job completed without an output file."
        elif raw_status in {"failed", "cancelled"}:
            operation["error"] = str(job.get("error") or job.get("message") or "The linked job did not complete.")[:2000]
        operation["updated_at"] = time.time()
        return _save(root, operation)


def transition_operation(
    root: str | os.PathLike[str],
    operation_id: Any,
    client_submission_id: Any,
    status: Any,
    *,
    job_id: Any = None,
    output_file: Any = None,
    error: Any = None,
    review_reason: Any = None,
) -> dict[str, Any]:
    next_status = str(status or "")
    if next_status not in _STATUSES or next_status not in _TERMINAL:
        raise EditorOperationError("Only a terminal Editor operation status can be set explicitly.")
    with _lock:
        operation = validate_submission(
            root,
            operation_id,
            client_submission_id,
            allow_linked=True,
        )
        linked_job_id = operation.get("job_id")
        if linked_job_id:
            if str(job_id or "") != linked_job_id:
                raise EditorOperationError("Terminal update does not own the linked job.")
        elif job_id:
            raise EditorOperationError("An unlinked operation cannot claim a job during completion.")
        if next_status == "completed" and (
            not linked_job_id
            or operation.get("job_status") != "completed"
            or not operation.get("output_file")
            or str(output_file or "") != str(operation.get("output_file"))
        ):
            raise EditorOperationError("Only the verified completed job output can finish an operation.")
        if next_status == "needs_review" and not (linked_job_id or error):
            raise EditorOperationError("A review state needs a linked result or an error.")
        operation["status"] = next_status
        if output_file:
            operation["output_file"] = str(output_file)[:4096]
        if error is not None:
            operation["error"] = str(error)[:2000]
        if review_reason is not None:
            operation["review_reason"] = str(review_reason)[:2000]
        operation["updated_at"] = time.time()
        return _save(root, operation)


def find_owned_job(
    operation: dict[str, Any],
    jobs: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a job only when its ID and both ownership tokens match."""
    linked_job_id = operation.get("job_id")
    if not linked_job_id:
        return None
    return next((
        job for job in jobs
        if str(job.get("id") or "") == linked_job_id
        and job.get("editor_operation_id") == operation.get("operation_id")
        and job.get("client_submission_id") == operation.get("client_submission_id")
    ), None)


def source_snapshot_matches(expected: Any, current: Any) -> bool:
    """Compare only the originating clip identity and timing, not the timeline."""
    if not isinstance(expected, dict) or not isinstance(current, dict):
        return False
    for key in _SOURCE_COMPARISON_KEYS:
        if key not in expected:
            continue
        left, right = expected.get(key), current.get(key)
        if key in {"duration", "source_in", "speed"}:
            try:
                if abs(float(left) - float(right)) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif left != right:
            return False
    return True
