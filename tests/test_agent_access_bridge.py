"""CPU-only bridge integration coverage using HTTPX's in-memory transport."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import unittest

if any(importlib.util.find_spec(name) is None for name in ("mcp", "fastapi", "httpx")):
    raise unittest.SkipTest("Agent Access bridge integration dependencies are not installed")

import httpx
import pytest
from fastapi import HTTPException

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.auth import Principal  # noqa: E402
from services.agent_access.bridge import BackendClient, BridgeError  # noqa: E402
from services.agent_access.context import credential_context, principal_context  # noqa: E402
from services.agent_access.state import RuntimeState, atomic_json  # noqa: E402


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _runtime(tmp_path: Path) -> RuntimeState:
    app_root = tmp_path / "maestro-app"
    app_root.mkdir(parents=True)
    for name in ("uploads", "outputs", "outputs2", "storage", "temp", "ckpts", "loras", "loras_i2v"):
        (app_root / name).mkdir()
    return RuntimeState(tmp_path / "settings" / "agent-access", app_root=app_root, port=_port())


def _run_as(bridge: BackendClient, principal: Principal, token: str, operation):
    async def invoke():
        principal_token = principal_context.set(principal)
        credential_token = credential_context.set(token)
        try:
            return await operation()
        finally:
            credential_context.reset(credential_token)
            principal_context.reset(principal_token)
    return asyncio.run(invoke())


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        return None


def _registered_asset(bridge: BackendClient, principal_id: str, filename: str, data: bytes) -> str:
    upload = bridge.assets.begin(principal_id, filename, len(data), hashlib.sha256(data).hexdigest())
    if data:
        bridge.assets.write_chunk(principal_id, upload["upload_id"], 0, data)
    bridge.assets.finish(principal_id, upload["upload_id"])
    destination = bridge.runtime.app_root / "uploads" / filename
    destination.write_bytes(data)
    return bridge.assets.register(principal_id, upload["upload_id"], destination)


def test_bridge_forwards_only_to_registered_backend_and_redacts_credentials(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="httpx-mock")
    principal = Principal("gr_bridge", "client-bridge", ("observe",), False)
    calls: list[httpx.Request] = []

    def dispatch(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={
            "status": "ready",
            "apiKey": "secret-camel",
            "refreshToken": "secret-refresh",
            "nested": {"privateKey": "secret-private"},
            "secretKey": "secret-key-value",
        })

    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(dispatch)))
    try:
        result = _run_as(
            bridge, principal, "opaque-test-token",
            lambda: bridge.request("GET", "/api/v1/system-stats"),
        )
        assert result["status"] == "ready"
        assert result["apiKey"] == "[redacted]"
        assert result["refreshToken"] == "[redacted]"
        assert result["nested"]["privateKey"] == "[redacted]"
        assert result["secretKey"] == "[redacted]"
        assert calls[0].url == "http://127.0.0.1:65534/api/v1/system-stats"
        assert calls[0].headers["authorization"] == "Bearer opaque-test-token"
        serialized = json.dumps(result)
        for secret in ("secret-camel", "secret-refresh", "secret-private", "secret-key-value", "opaque-test-token"):
            assert secret not in serialized

        with pytest.raises(HTTPException) as denied:
            _run_as(
                bridge, principal, "opaque-test-token",
                lambda: bridge.request("POST", "/api/v1/system-config", json={"debug": True}, idempotency_key="stable-1"),
            )
        assert denied.value.status_code == 403
        assert len(calls) == 1
    finally:
        asyncio.run(bridge.close())


def test_bridge_returns_conservative_errors_for_closed_backend_and_idle_probe(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="closed-backend")
    principal = Principal("gr_admin", "client-admin", ("generate", "admin"), False)
    calls = 0

    def closed(request: httpx.Request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("private host detail", request=request)

    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(closed)))
    try:
        with pytest.raises(BridgeError) as uncertain:
            _run_as(
                bridge, principal, "opaque-token",
                lambda: bridge.request("POST", "/api/v1/generate", json={"prompt": "private prompt"}, idempotency_key="same-key"),
            )
        assert uncertain.value.status == 503
        assert "private host detail" not in str(uncertain.value)
        assert "submission receipt" in str(uncertain.value)
        assert calls == 1

        evidence = _run_as(bridge, principal, "opaque-token", bridge.idle_probe)
        assert evidence["idle"] is False
        assert evidence["certain"] is False
        assert calls == 2
    finally:
        asyncio.run(bridge.close())


def test_offline_bridge_returns_only_the_callers_durable_job_receipt(tmp_path):
    runtime = _runtime(tmp_path)
    principal = Principal("gr_owner_job", "client-job", ("observe",), False)
    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(500, json={"detail": "backend must not be contacted"})
    )))
    bridge.journal.record_job(
        {"id": "job-owned", "status": "running", "prompt": "private prompt", "api_key": "secret", "progress": 30},
        principal.id,
        "backend-session",
    )
    bridge.journal.record_job({"id": "job-foreign", "status": "running"}, "another-principal", "backend-session")
    try:
        receipt = _run_as(
            bridge, principal, "opaque-token",
            lambda: bridge.request("GET", "/api/v1/status/job-owned"),
        )
        assert receipt["snapshot"]["job_id"] == "job-owned"
        assert receipt["snapshot"]["status"] == "running"
        assert "prompt" not in receipt["snapshot"]
        assert "api_key" not in receipt["snapshot"]
        with pytest.raises(BridgeError) as foreign:
            _run_as(
                bridge, principal, "opaque-token",
                lambda: bridge.request("GET", "/api/v1/status/job-foreign"),
            )
        assert foreign.value.status == 404
    finally:
        asyncio.run(bridge.close())


def test_submission_lookup_is_principal_scoped_and_available_offline(tmp_path):
    runtime = _runtime(tmp_path)
    principal = Principal("gr_receipt", "client-receipt", ("observe",), False)
    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(500, json={"detail": "offline path must not contact backend"})
    )))
    reservation = bridge.journal.reserve(
        principal.id, "prepare-timeout", "POST", "/api/v1/agent-access/prepare", {"prompt": "private"},
    )
    bridge.journal.complete(
        reservation["reservation_id"],
        200,
        {"job_id": "job-recovered", "api_key": "secret-value", "status": "queued"},
        {"content-type": "application/json"},
    )
    bridge.journal.reserve(
        "gr_foreign_receipt", "prepare-timeout", "POST", "/api/v1/agent-access/prepare", {"prompt": "private"},
    )
    try:
        receipt = _run_as(bridge, principal, "opaque-token", lambda: bridge.get_submission("prepare-timeout"))
        assert receipt["state"] == "complete"
        assert receipt["result_available"] is True
        assert receipt["result"]["job_id"] == "job-recovered"
        assert receipt["result"]["api_key"] == "[redacted]"
        with pytest.raises(BridgeError) as foreign:
            _run_as(bridge, principal, "opaque-token", lambda: bridge.get_submission("foreign-only"))
        assert foreign.value.status == 404
    finally:
        asyncio.run(bridge.close())


def test_public_status_and_bridge_both_report_dead_backend_offline(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="dead-backend")
    backend_path = runtime.directory / "backend.json"
    record = json.loads(backend_path.read_text(encoding="utf-8"))
    record["pid"] = 0
    atomic_json(backend_path, record)
    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={})
    )))
    try:
        assert bridge.online() is False
        assert runtime.public_status()["backend_online"] is False
    finally:
        asyncio.run(bridge.close())


def test_binary_get_streams_to_owned_transfer_and_respects_transfer_scope(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="binary-get")
    principal = Principal("gr_binary", "client-binary", ("observe",), False)
    body = b"0123456789" * 800_000
    calls: list[httpx.Request] = []

    async def dispatch(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "video/mp4", "content-disposition": 'attachment; filename="clip.mp4"'},
            stream=_Chunks([body[:3_000_000], body[3_000_000:]]),
        )

    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(dispatch)))
    try:
        record = _run_as(
            bridge, principal, "opaque-token",
            lambda: bridge.request("GET", "/api/v1/jobs/job-1/preview/2", response_kind="binary"),
        )
        assert record["filename"] == "clip.mp4"
        assert record["size"] == len(body)
        assert record["sha256"] == hashlib.sha256(body).hexdigest()
        assert record["url"].startswith(runtime.origin + "/api/v1/agent-access/transfers/")
        assert calls[0].headers["accept"] == "*/*"

        block = _run_as(
            bridge, principal, "opaque-token",
            lambda: bridge.read_transfer_chunk(record["transfer_id"], offset=123, max_bytes=111),
        )
        assert base64.b64decode(block["data_base64"]) == body[123:234]
        assert block["sha256"] == record["sha256"]
        with pytest.raises(BridgeError) as foreign:
            _run_as(
                bridge, Principal("gr_other", "client-other", ("observe",), False), "opaque-token",
                lambda: bridge.read_transfer_chunk(record["transfer_id"]),
            )
        assert foreign.value.status == 404
    finally:
        asyncio.run(bridge.close())


def test_binary_mutation_reuses_saved_artifact_without_native_resubmission(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="binary-post")
    principal = Principal("gr_generate", "client-generate", ("generate", "manage"), False)
    calls = 0
    body = b"export-bytes"

    async def dispatch(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await request.aread()
        return httpx.Response(200, headers={"content-type": "application/zip"}, stream=_Chunks([body]))

    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(dispatch)))
    try:
        operation = lambda: bridge.request(
            "POST", "/api/v1/worlds/world-1/save", json={"format": "zip"},
            idempotency_key="save-1", response_kind="binary",
        )
        first = _run_as(bridge, principal, "opaque-token", operation)
        second = _run_as(bridge, principal, "opaque-token", operation)
        assert first == second
        assert first["sha256"] == hashlib.sha256(body).hexdigest()
        assert calls == 1
    finally:
        asyncio.run(bridge.close())


def test_binary_mutation_timeout_or_native_receipt_never_retriggers(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="binary-lost")
    principal = Principal("gr_generate_lost", "client-generate-lost", ("generate", "manage"), False)
    calls = 0

    async def disconnected(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadError("connection ended after dispatch", request=request)

    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(disconnected)))
    try:
        operation = lambda: bridge.request(
            "POST", "/api/v1/worlds/world-2/save", json={"format": "zip"},
            idempotency_key="save-lost", response_kind="binary",
        )
        with pytest.raises(BridgeError) as lost:
            _run_as(bridge, principal, "opaque-token", operation)
        assert lost.value.status == 503
        with pytest.raises(BridgeError) as uncertain:
            _run_as(bridge, principal, "opaque-token", operation)
        assert uncertain.value.status == 409
        assert "do not re-submit" in str(uncertain.value)
        assert calls == 1
    finally:
        asyncio.run(bridge.close())

    runtime2 = _runtime(tmp_path / "receipt")
    runtime2.register_backend("http://127.0.0.1:65534", ready=True, session_id="binary-receipt")
    receipt_calls = 0

    async def receipt(request: httpx.Request) -> httpx.Response:
        nonlocal receipt_calls
        receipt_calls += 1
        return httpx.Response(200, json={"kind": "binary_receipt", "bytes": 12, "sha256": "a" * 64, "delivery_verified": False})

    bridge2 = BackendClient(runtime2, client=httpx.AsyncClient(transport=httpx.MockTransport(receipt)))
    try:
        operation = lambda: bridge2.request(
            "POST", "/api/v1/worlds/world-3/save", json={"format": "zip"},
            idempotency_key="save-receipt", response_kind="binary",
        )
        with pytest.raises(BridgeError) as receipt_missing:
            _run_as(bridge2, principal, "opaque-token", operation)
        assert receipt_missing.value.status == 409
        with pytest.raises(BridgeError):
            _run_as(bridge2, principal, "opaque-token", operation)
        assert receipt_calls == 1
    finally:
        asyncio.run(bridge2.close())


def test_multipart_requests_use_owned_assets_and_a_stable_wire_encoding(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="multipart")
    principal = Principal("gr_manage", "client-manage", ("manage",), False)
    captured: list[tuple[bytes, str, str]] = []
    upload_bytes = b"audio-data\x00\x01"

    async def dispatch(request: httpx.Request) -> httpx.Response:
        captured.append((await request.aread(), request.headers["content-type"], request.headers["content-length"]))
        return httpx.Response(200, json={"imported": True})

    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(dispatch)))
    try:
        asset_ref = _registered_asset(bridge, principal.id, "voice.wav", upload_bytes)
        arguments = {
            "file": asset_ref,
            "name": "Night voice",
            "labels": ["spoken", "quiet"],
        }
        operation = lambda: bridge.request(
            "POST", "/api/v1/music-styles/import", json=arguments,
            idempotency_key="style-import-1", request_kind="multipart",
            file_fields=("file",),
        )
        first = _run_as(bridge, principal, "opaque-token", operation)
        second = _run_as(bridge, principal, "opaque-token", operation)
        assert first == second == {"imported": True}
        assert len(captured) == 2
        assert captured[0] == captured[1]
        wire, content_type, content_length = captured[0]
        assert content_type.startswith("multipart/form-data; boundary=maestro-agent-")
        assert int(content_length) == len(wire)
        assert b"name=\"file\"; filename=\"voice.wav\"" in wire
        assert b"Content-Type: audio/" in wire
        assert upload_bytes in wire
        assert b'name="labels"' in wire
        assert wire.index(b'name="file"') < wire.index(b'name="labels"') < wire.index(b'name="name"')

        with pytest.raises(BridgeError) as foreign:
            _run_as(
                bridge, Principal("gr_foreign_asset", "client-foreign", ("manage",), False), "opaque-token",
                lambda: bridge.request(
                    "POST", "/api/v1/music-styles/import", json={"file": asset_ref},
                    idempotency_key="foreign-import", request_kind="multipart", file_fields=("file",),
                ),
            )
        assert foreign.value.status == 404
        assert len(captured) == 2
    finally:
        asyncio.run(bridge.close())


def test_multipart_boundary_collision_is_rejected_before_native_dispatch(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="multipart-collision")
    principal = Principal("gr_manage_collision", "client-manage-collision", ("manage",), False)
    request_id = "collision-1"
    boundary = "maestro-agent-" + hashlib.sha256(
        f"POST\n/api/v1/music-styles/import\n{request_id}".encode("utf-8")
    ).hexdigest()[:32]
    file_bytes = b"audio\r\n--" + boundary.encode("ascii") + b"\r\nforged-part"
    native_side_effects = 0

    async def dispatch(request: httpx.Request) -> httpx.Response:
        nonlocal native_side_effects
        await request.aread()  # the authenticated boundary must finish spooling before dispatch
        native_side_effects += 1
        return httpx.Response(200, json={"ok": True})

    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(dispatch)))
    try:
        asset_ref = _registered_asset(bridge, principal.id, "collision.wav", file_bytes)
        with pytest.raises(BridgeError) as rejected:
            _run_as(
                bridge, principal, "opaque-token",
                lambda: bridge.request(
                    "POST", "/api/v1/music-styles/import", json={"file": asset_ref},
                    idempotency_key=request_id, request_kind="multipart", file_fields=("file",),
                ),
            )
        assert rejected.value.status == 400
        assert "boundary" in str(rejected.value)
        assert native_side_effects == 0
    finally:
        asyncio.run(bridge.close())


def test_fetch_media_accepts_only_a_safe_basename_and_uses_transfer_store(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="fetch-media")
    principal = Principal("gr_media", "client-media", ("observe",), False)
    calls: list[httpx.Request] = []

    async def dispatch(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "image/png"}, stream=_Chunks([b"png-data"]))

    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(dispatch)))
    try:
        result = _run_as(
            bridge, principal, "opaque-token",
            lambda: bridge.fetch_media("cover art.png", "default"),
        )
        assert result["filename"] == "cover art.png"
        assert calls[0].url.raw_path.split(b"?", 1)[0] == b"/api/v1/file/cover%20art.png"
        assert calls[0].url.params["workspace"] == "default"
        with pytest.raises(BridgeError):
            _run_as(bridge, principal, "opaque-token", lambda: bridge.fetch_media("../secrets.json"))
        assert len(calls) == 1
    finally:
        asyncio.run(bridge.close())
