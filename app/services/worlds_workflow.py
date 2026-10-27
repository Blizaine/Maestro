"""Durable HTTP workflow boundary for experimental Hy-Nomas Worlds.

This module never owns a model or GPU. Generation is submitted through an
injected durable-job callback; completed outputs are made visible only by an
idempotent app-storage callback after it verifies both the video and the
``.wgpworld`` sidecar. The router accepts world identifiers and prompts, never
filesystem paths or model settings.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response


WORLD_MODEL_TYPE = "hy_nomas"
WORLD_RESOLUTION = "832x480"
WORLD_VIDEO_LENGTH = 97
WORLD_STEPS = 8
WORLD_GUIDANCE = 1.0
MAX_WORLD_FILE_BYTES = 3 * 1024 * 1024 * 1024
MAX_CAPTURE_BYTES = 32 * 1024 * 1024
MAX_RECORDING_BYTES = 3 * 1024 * 1024 * 1024
PIXELUMM_LICENSE_URL = (
    "https://developer.download.nvidia.com/licenses/"
    "NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf"
)

_ACTIONS = {
    "Forward", "Backward", "Left", "Right", "Look_Left", "Look_Right",
    "Look_Up", "Look_Down", "Wait",
}
_WORLD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL = {"completed", "failed", "cancelled"}
_ACTIVE = {"held", "queued", "running"}
_RECEIPT_TERMINAL = {
    "published", "failed", "cancelled", "verification_failed", "ownership_conflict",
}


def _bounded_detail(value: Any, fallback: str) -> str:
    """Return a compact, single-line reason safe to persist in a receipt."""
    if not isinstance(value, str):
        return fallback
    detail = " ".join(value.split())[:400]
    return detail or fallback


def canonical_request_digest(request_fields: dict[str, Any]) -> str:
    """Hash the normalized public fields shared with ``ui/src/api/worlds.ts``."""
    encoded = json.dumps(
        request_fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_world_request(payload: Any) -> tuple[str, dict[str, Any], str]:
    """Validate the intentionally small client contract and recompute digest.

    Returns ``(request_id, normalized_fields, digest)``. Unknown keys are
    rejected so client input cannot smuggle a model choice, path, or setting
    into the regular generation submission.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="World request must be a JSON object.")
    operation = payload.get("operation")
    common = {
        "client_request_id", "request_digest", "operation",
        "acknowledge_pixelumm_noncommercial",
    }
    specific = {
        "create": {"description"},
        "continue": {"world_id", "text"},
        "move": {"world_id", "action", "duration_ms"},
    }.get(operation)
    if specific is None:
        raise HTTPException(status_code=422, detail="operation must be create, continue, or move.")
    if set(payload) != common | specific:
        raise HTTPException(status_code=422, detail="World request has missing or unsupported fields.")

    request_id = payload.get("client_request_id")
    if not isinstance(request_id, str):
        raise HTTPException(status_code=422, detail="client_request_id must be a UUID.")
    try:
        if str(uuid.UUID(request_id)) != request_id:
            raise ValueError("non-canonical UUID")
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail="client_request_id must be a canonical UUID.")

    digest = payload.get("request_digest")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise HTTPException(status_code=422, detail="request_digest must be a lowercase SHA-256 digest.")
    if payload.get("acknowledge_pixelumm_noncommercial") is not True:
        raise HTTPException(
            status_code=403,
            detail=f"Acknowledge PixelUMM's non-commercial terms before using Worlds: {PIXELUMM_LICENSE_URL}",
        )

    fields: dict[str, Any] = {
        "operation": operation,
        "acknowledge_pixelumm_noncommercial": True,
    }
    if operation == "create":
        description = payload.get("description")
        if not isinstance(description, str) or not description.strip() or len(description.strip()) > 5000:
            raise HTTPException(status_code=422, detail="description must contain 1 to 5000 characters.")
        fields["description"] = description.strip()
    else:
        world_id = payload.get("world_id")
        if not isinstance(world_id, str) or not _WORLD_ID_RE.fullmatch(world_id):
            raise HTTPException(status_code=422, detail="world_id is invalid.")
        fields["world_id"] = world_id
        if operation == "continue":
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip() or len(text.strip()) > 2000:
                raise HTTPException(status_code=422, detail="text must contain 1 to 2000 characters.")
            fields["text"] = text.strip()
        else:
            action = payload.get("action")
            if action not in _ACTIONS:
                raise HTTPException(status_code=422, detail="action is not supported.")
            duration = payload.get("duration_ms")
            if isinstance(duration, bool) or not isinstance(duration, int) or not 100 <= duration <= 10_000:
                raise HTTPException(status_code=422, detail="duration_ms must be between 100 and 10000.")
            fields["action"] = action
            fields["duration_ms"] = duration

    actual_digest = canonical_request_digest(fields)
    if digest != actual_digest:
        raise HTTPException(status_code=409, detail="World request digest does not match its fields.")
    return request_id, fields, actual_digest


def build_native_prompt(fields: dict[str, Any]) -> str:
    """Map the public action form to Hy-Nomas' documented bracket syntax."""
    operation = fields["operation"]
    if operation == "create":
        return fields["description"]
    if operation == "continue":
        return f"[Event] {fields['text']}"
    seconds = fields["duration_ms"] / 1000
    duration = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return f"[{fields['action']} {duration}s]"


def build_generation_body(
    fields: dict[str, Any],
    *,
    client_submission_id: str,
    custom_guide: str | None = None,
) -> dict[str, Any]:
    """Create the fixed, trusted generation body used by the app callback."""
    body: dict[str, Any] = {
        "model_type": WORLD_MODEL_TYPE,
        "prompt": build_native_prompt(fields),
        "num_inference_steps": WORLD_STEPS,
        "guidance_scale": WORLD_GUIDANCE,
        "video_length": WORLD_VIDEO_LENGTH,
        "resolution": WORLD_RESOLUTION,
        "custom_settings": {
            "world_output": "world_video",
            "acknowledge_pixelumm_noncommercial": True,
        },
        "_client_submission_id": client_submission_id,
    }
    if custom_guide is not None:
        body["custom_guide"] = custom_guide
    return body


async def _call(callback: Callable, *args):
    result = callback(*args)
    if inspect.isawaitable(result):
        return await result
    return result


def _safe_world_id(value: Any) -> str:
    if not isinstance(value, str) or not _WORLD_ID_RE.fullmatch(value):
        raise HTTPException(status_code=404, detail="World was not found.")
    return value


def _safe_child_file(path_value: Any, workspace_root: Path, suffix: str, max_bytes: int) -> Path:
    if not isinstance(path_value, (str, Path)):
        raise HTTPException(status_code=404, detail="World file is unavailable.")
    try:
        root = workspace_root.expanduser().resolve()
        path = Path(path_value).expanduser().resolve()
    except OSError:
        raise HTTPException(status_code=404, detail="World file is unavailable.")
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="World file is unavailable.")
    if path.suffix.lower() != suffix or not path.is_file():
        raise HTTPException(status_code=404, detail="World file is unavailable.")
    try:
        size = path.stat().st_size
    except OSError:
        raise HTTPException(status_code=404, detail="World file is unavailable.")
    if size < 1 or size > max_bytes:
        raise HTTPException(status_code=413, detail="World file exceeds the allowed size.")
    return path


def _public_world(world: Any) -> dict[str, Any]:
    if not isinstance(world, dict):
        raise HTTPException(status_code=404, detail="World was not found.")
    # Only app-owned presentation fields cross this boundary; in particular,
    # storage paths and internal manifest data are never exposed.
    result = {}
    for key in (
        "world_id", "title", "description", "created_at", "updated_at",
        "video_url", "poster_url", "thumbnail_url", "recording_available",
        "segment_count", "last_action",
    ):
        value = world.get(key)
        if value is not None:
            result[key] = value
    world_id = result.get("world_id")
    if not isinstance(world_id, str) or not _WORLD_ID_RE.fullmatch(world_id):
        raise HTTPException(status_code=500, detail="Stored world record has an invalid identifier.")
    if "recording_available" in result:
        result["recording_available"] = result["recording_available"] is True
    return result


def _world_for_id(world: Any, world_id: str) -> dict[str, Any]:
    public = _public_world(world)
    if public["world_id"] != world_id:
        raise HTTPException(status_code=404, detail="World was not found.")
    return public


def _job_id(job: Any) -> str | None:
    if not isinstance(job, dict):
        return None
    value = job.get("job_id", job.get("id"))
    return value if isinstance(value, str) and value else None


def _jobs_from(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("jobs", list(value.values()))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def create_worlds_router(
    *,
    workspace_root: str | Path,
    submit_generation: Callable[[dict[str, Any]], Any],
    lookup_jobs: Callable[[], Any],
    get_job_status: Callable[[str], Any],
    cancel_generation: Callable[[str], Any],
    list_worlds: Callable[[], Any],
    get_world: Callable[[str], Any],
    resolve_world_file: Callable[[str], Any],
    publish_completed_generation: Callable[[dict[str, Any], dict[str, Any]], Any],
    save_world: Callable[[str], Any],
    record_world: Callable[[str], Any],
    capture_world: Callable[[str], Any],
    read_receipt: Callable[[str], Any],
    write_receipt: Callable[[str, dict[str, Any]], Any],
) -> APIRouter:
    """Build the Worlds API from app-owned durable jobs and storage callbacks.

    ``write_receipt`` must be durable. The router serializes requests sharing
    an ID inside this process and never submits an ID with an existing receipt.
    ``publish_completed_generation`` must be idempotent by client request ID;
    it must verify the real video and ``.wgpworld`` sidecar and return all of
    ``verified``, ``video_verified`` and ``world_file_verified`` as true before
    the router makes a new world visible.
    """
    root = Path(workspace_root)
    router = APIRouter()
    request_locks: dict[str, asyncio.Lock] = {}

    def lock_for(request_id: str) -> asyncio.Lock:
        return request_locks.setdefault(request_id, asyncio.Lock())

    async def read_one(request_id: str) -> dict[str, Any] | None:
        value = await _call(read_receipt, request_id)
        return value if isinstance(value, dict) else None

    async def write_one(request_id: str, receipt: dict[str, Any]) -> None:
        await _call(write_receipt, request_id, receipt)

    async def resolve_source_path(world_id: str) -> Path:
        world_id = _safe_world_id(world_id)
        world = await _call(get_world, world_id)
        if not isinstance(world, dict):
            raise HTTPException(status_code=404, detail="World was not found.")
        _world_for_id(world, world_id)
        try:
            value = await _call(resolve_world_file, world_id)
        except (FileNotFoundError, KeyError):
            raise HTTPException(status_code=404, detail="Saved world state is unavailable.")
        return _safe_child_file(value, root, ".wgpworld", MAX_WORLD_FILE_BYTES)

    async def reconcile(receipt: dict[str, Any]) -> dict[str, Any]:
        # Durable terminal outcomes must remain terminal across duplicate POST
        # and status GET requests. In particular, invalid verified outputs
        # must not invoke the idempotent publisher on every poll.
        if receipt.get("state") in _RECEIPT_TERMINAL:
            return receipt

        request_id = receipt["client_request_id"]
        submission_id = receipt["client_submission_id"]
        job_id = receipt.get("job_id")
        job = None
        if not job_id:
            try:
                matches = [
                    candidate for candidate in _jobs_from(await _call(lookup_jobs))
                    if candidate.get("client_submission_id") == submission_id
                ]
            except Exception:
                matches = []
            if len(matches) == 1:
                job_id = _job_id(matches[0])
                if job_id:
                    receipt["job_id"] = job_id
                    receipt["state"] = "queued"
                    await write_one(request_id, receipt)
            elif len(matches) > 1:
                receipt["state"] = "ownership_conflict"
                receipt["detail"] = "More than one job has this client submission ID."
                await write_one(request_id, receipt)
                return receipt
        if not job_id:
            if receipt.get("state") == "submitting":
                receipt["state"] = "reconciling"
                await write_one(request_id, receipt)
            return receipt
        try:
            job = await _call(get_job_status, job_id)
        except Exception:
            # A transient status failure does not authorize another submit or
            # cancel. The next GET can retry the read.
            return receipt
        if not isinstance(job, dict):
            return receipt
        if _job_id(job) != job_id or job.get("client_submission_id") != submission_id:
            receipt["state"] = "ownership_conflict"
            receipt["detail"] = "The job status does not match the stored request owner."
            await write_one(request_id, receipt)
            return receipt

        status = job.get("status")
        receipt["job_status"] = status
        if status in _ACTIVE:
            if receipt.get("state") not in {"cancel_requested", "cancel_sent"}:
                receipt["state"] = status
            await write_one(request_id, receipt)
            return receipt
        if status in {"failed", "cancelled"}:
            receipt["state"] = status
            await write_one(request_id, receipt)
            return receipt
        if status != "completed":
            receipt["state"] = "reconciling"
            await write_one(request_id, receipt)
            return receipt
        if receipt.get("state") == "published" and isinstance(receipt.get("world"), dict):
            return receipt

        # Persist the intent first. The app callback is required to be
        # idempotent, so a process interruption during publication can safely
        # be reconciled by a later GET.
        receipt["state"] = "publication_pending"
        receipt["job_status"] = "completed"
        await write_one(request_id, receipt)
        try:
            publication = await _call(publish_completed_generation, job, dict(receipt))
        except FileNotFoundError:
            receipt["state"] = "verification_failed"
            receipt["detail"] = "A required completed world output is missing; it was not published."
            await write_one(request_id, receipt)
            return receipt
        except ValueError:
            # WorldsStorageError is a ValueError subclass. These failures mean
            # the completed files do not meet the storage contract, rather
            # than a transient publication interruption.
            receipt["state"] = "verification_failed"
            receipt["detail"] = "The completed world output failed storage validation; it was not published."
            await write_one(request_id, receipt)
            return receipt
        except Exception:
            # Unknown exceptions may be transient. Keep the durable intent so
            # a later GET can retry the idempotent publication, never a submit.
            return receipt
        if not isinstance(publication, dict) or not all(
            publication.get(flag) is True
            for flag in ("verified", "video_verified", "world_file_verified")
        ):
            receipt["state"] = "verification_failed"
            receipt["detail"] = "The completed video and world sidecar could not be verified."
            await write_one(request_id, receipt)
            return receipt
        try:
            receipt["world"] = _public_world(publication.get("world"))
        except HTTPException:
            receipt["state"] = "verification_failed"
            receipt["detail"] = "The verified world record is invalid."
            await write_one(request_id, receipt)
            return receipt
        receipt["state"] = "published"
        receipt.pop("detail", None)
        await write_one(request_id, receipt)
        return receipt

    def receipt_view(receipt: dict[str, Any]) -> dict[str, Any]:
        view = {
            "client_request_id": receipt.get("client_request_id"),
            "request_digest": receipt.get("request_digest"),
            "state": receipt.get("state", "reconciling"),
            "job_id": receipt.get("job_id"),
            "job_status": receipt.get("job_status"),
            "cancel_state": receipt.get("cancel_state", "none"),
            "detail": receipt.get("detail"),
        }
        if isinstance(receipt.get("world"), dict):
            view["world"] = receipt["world"]
        return view

    async def receive(request_id: str) -> dict[str, Any]:
        async with lock_for(request_id):
            receipt = await read_one(request_id)
            if receipt is None:
                raise HTTPException(status_code=404, detail="World request was not found.")
            return receipt_view(await reconcile(receipt))

    @router.get("/api/v1/worlds")
    async def get_worlds():
        values = await _call(list_worlds)
        if not isinstance(values, list):
            raise HTTPException(status_code=503, detail="World library is unavailable.")
        return {"worlds": [_public_world(value) for value in values if isinstance(value, dict)]}

    @router.get("/api/v1/worlds/{world_id}")
    async def get_world_by_id(world_id: str):
        safe_id = _safe_world_id(world_id)
        value = await _call(get_world, safe_id)
        return {"world": _world_for_id(value, safe_id)}

    @router.post("/api/v1/worlds/requests")
    async def submit_world_request(request: Request):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=422, detail="World request must contain valid JSON.")
        request_id, fields, digest = normalize_world_request(payload)
        async with lock_for(request_id):
            existing = await read_one(request_id)
            if existing is not None:
                if existing.get("request_digest") != digest:
                    raise HTTPException(status_code=409, detail="client request ID is already bound to another digest.")
                # A duplicate or ambiguous receipt is always GET/reconcile;
                # this branch never calls submit_generation again.
                return receipt_view(await reconcile(existing))

            # Resolve source state before reserving a receipt: bad IDs or
            # missing sidecars must not consume an otherwise valid request ID.
            # For an existing receipt this path is skipped, so a duplicate can
            # still reconcile even if the user's original file was moved.
            source_path = None
            if fields["operation"] != "create":
                source_path = await resolve_source_path(fields["world_id"])

            receipt: dict[str, Any] = {
                "version": 1,
                "client_request_id": request_id,
                "client_submission_id": request_id,
                "request_digest": digest,
                "request": fields,
                "state": "submitting",
                "job_id": None,
                "job_status": None,
                "cancel_state": "none",
            }
            await write_one(request_id, receipt)
            body = build_generation_body(
                fields,
                client_submission_id=request_id,
                custom_guide=str(source_path) if source_path is not None else None,
            )
            try:
                accepted = await _call(submit_generation, body)
            except Exception:
                # The callback may have accepted before a transport error.
                # Keep the durable receipt and reconcile by submission ID.
                receipt["state"] = "reconciling"
                await write_one(request_id, receipt)
                return receipt_view(await reconcile(receipt))
            if not isinstance(accepted, dict):
                receipt["state"] = "reconciling"
                await write_one(request_id, receipt)
                return receipt_view(await reconcile(receipt))
            if (
                accepted.get("rejected") is True
                and accepted.get("client_submission_id") == request_id
            ):
                # The generation boundary may explicitly report that a
                # request was rejected before queueing. Only trust this
                # terminal outcome when its ID matches the durable receipt.
                receipt["state"] = "failed"
                receipt["detail"] = _bounded_detail(
                    accepted.get("detail"),
                    "World generation was rejected before it entered the queue.",
                )
                await write_one(request_id, receipt)
                return receipt_view(receipt)
            accepted_job_id = _job_id(accepted)
            if accepted.get("client_submission_id") != request_id or not accepted_job_id:
                receipt["state"] = "reconciling"
                await write_one(request_id, receipt)
                return receipt_view(await reconcile(receipt))
            receipt["job_id"] = accepted_job_id
            receipt["job_status"] = accepted.get("status")
            receipt["state"] = accepted.get("status") if accepted.get("status") in _ACTIVE else "reconciling"
            await write_one(request_id, receipt)
            return receipt_view(await reconcile(receipt))

    @router.get("/api/v1/worlds/requests/{client_request_id}")
    async def get_world_request(client_request_id: str):
        try:
            if str(uuid.UUID(client_request_id)) != client_request_id:
                raise ValueError("non-canonical")
        except (ValueError, AttributeError):
            raise HTTPException(status_code=404, detail="World request was not found.")
        return await receive(client_request_id)

    @router.post("/api/v1/worlds/requests/{client_request_id}/cancel")
    async def cancel_world_request(client_request_id: str):
        try:
            if str(uuid.UUID(client_request_id)) != client_request_id:
                raise ValueError("non-canonical")
        except (ValueError, AttributeError):
            raise HTTPException(status_code=404, detail="World request was not found.")
        async with lock_for(client_request_id):
            receipt = await read_one(client_request_id)
            if receipt is None:
                raise HTTPException(status_code=404, detail="World request was not found.")
            expected_job_id = receipt.get("job_id")
            if not isinstance(expected_job_id, str) or not expected_job_id:
                raise HTTPException(status_code=409, detail="World request has no reconciled job to cancel.")
            if receipt.get("client_submission_id") != client_request_id:
                raise HTTPException(status_code=409, detail="World request ownership could not be verified.")
            if receipt.get("cancel_state") in {"requested", "sent"}:
                return receipt_view(receipt)
            try:
                latest = await _call(get_job_status, expected_job_id)
            except Exception:
                raise HTTPException(status_code=503, detail="Could not verify the latest job status; no cancel was sent.")
            if (
                not isinstance(latest, dict)
                or _job_id(latest) != expected_job_id
                or latest.get("client_submission_id") != client_request_id
            ):
                raise HTTPException(status_code=409, detail="Latest job status does not match the stored request owner.")
            status = latest.get("status")
            if status not in _ACTIVE:
                raise HTTPException(status_code=409, detail="This owned job is no longer cancellable.")
            # Mark before invoking the callback. A lost response must never
            # lead the client to issue a second cancellation.
            receipt["cancel_state"] = "requested"
            await write_one(client_request_id, receipt)
            try:
                await _call(cancel_generation, expected_job_id)
            except Exception:
                # The attempted cancel may have reached the queue. Preserve a
                # one-shot boundary and let the UI reconcile via GET.
                return receipt_view(receipt)
            receipt["cancel_state"] = "sent"
            await write_one(client_request_id, receipt)
            return receipt_view(receipt)

    @router.post("/api/v1/worlds/{world_id}/save")
    async def download_world(world_id: str, request: Request):
        safe_id = _safe_world_id(world_id)
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if not isinstance(payload, dict) or payload != {"acknowledge_pixelumm_noncommercial": True}:
            raise HTTPException(
                status_code=403,
                detail=f"Acknowledge PixelUMM's non-commercial terms before downloading: {PIXELUMM_LICENSE_URL}",
            )
        world = await _call(get_world, safe_id)
        if not isinstance(world, dict):
            raise HTTPException(status_code=404, detail="World was not found.")
        _world_for_id(world, safe_id)
        try:
            value = await _call(save_world, safe_id)
        except (FileNotFoundError, KeyError):
            raise HTTPException(status_code=404, detail="Saved world state is unavailable.")
        path = _safe_child_file(value, root, ".wgpworld", MAX_WORLD_FILE_BYTES)
        return FileResponse(path, media_type="application/zip", filename=f"{safe_id}.wgpworld")

    @router.post("/api/v1/worlds/{world_id}/record")
    async def record_world_video(world_id: str, request: Request):
        safe_id = _safe_world_id(world_id)
        world = await _call(get_world, safe_id)
        if not isinstance(world, dict):
            raise HTTPException(status_code=404, detail="World was not found.")
        _world_for_id(world, safe_id)
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if not isinstance(payload, dict) or payload != {"acknowledge_pixelumm_noncommercial": True}:
            raise HTTPException(
                status_code=403,
                detail=f"Acknowledge PixelUMM's non-commercial terms before recording: {PIXELUMM_LICENSE_URL}",
            )
        if world.get("recording_available") is not True:
            raise HTTPException(status_code=409, detail="Recording is unavailable because this world has no saved clip chain.")
        try:
            path_value = await asyncio.to_thread(record_world, safe_id)
        except (FileNotFoundError, KeyError, NotImplementedError):
            raise HTTPException(status_code=409, detail="Recording is unavailable for this world.")
        path = _safe_child_file(path_value, root, ".mp4", MAX_RECORDING_BYTES)
        return FileResponse(path, media_type="video/mp4", filename=f"{safe_id}-recording.mp4")

    @router.get("/api/v1/worlds/{world_id}/capture")
    async def capture_world_frame(world_id: str):
        safe_id = _safe_world_id(world_id)
        world = await _call(get_world, safe_id)
        if not isinstance(world, dict):
            raise HTTPException(status_code=404, detail="World was not found.")
        _world_for_id(world, safe_id)
        try:
            image = await _call(capture_world, safe_id)
        except (FileNotFoundError, KeyError, NotImplementedError):
            raise HTTPException(status_code=409, detail="A saved capture is unavailable for this world.")
        if isinstance(image, (bytes, bytearray)):
            data = bytes(image)
            if not 4 <= len(data) <= MAX_CAPTURE_BYTES or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                raise HTTPException(status_code=422, detail="Stored capture is not a valid bounded JPEG.")
            return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        path = _safe_child_file(image, root, ".jpg", MAX_CAPTURE_BYTES)
        with path.open("rb") as stream:
            head = stream.read(2)
            stream.seek(-2, 2)
            tail = stream.read(2)
        if head != b"\xff\xd8" or tail != b"\xff\xd9":
            raise HTTPException(status_code=422, detail="Stored capture is not a valid JPEG.")
        return FileResponse(path, media_type="image/jpeg", filename=f"{safe_id}.jpg")

    return router
