"""Streaming multipart and binary-receipt regressions for the API boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

pytest.importorskip("starlette")
from services.agent_access.auth import Principal
from services.agent_access.boundary import AccessBoundary, redact
from services.agent_access.jobs import SubmissionJournal


class FakeAuth:
    enabled = True

    def __init__(self, principal):
        self.principal = principal

    def authenticate(self, token):
        assert token == "agent-token"
        return self.principal


class FakeRuntime:
    security_required = True
    installation_id = "binary-boundary-test"
    local_url = "https://maestro.test/mcp"
    origin = "https://maestro.test"

    def __init__(self, directory: Path, principal):
        self.directory = directory
        self.auth = FakeAuth(principal)
        self.audit_events = []

    def audit(self, actor, action, status):
        self.audit_events.append((actor, action, status))

    def public_status(self):
        return {"controller_url": self.origin, "local_setup_url": self.local_url}


@pytest.mark.parametrize("path,mime,protected", [
    ("/uploads/title.svg", b"image/svg+xml", True),
    ("/api/v1/uploads/file/title.html", b"text/html", True),
    ("/uploads/clip.mp4", b"video/mp4", False),
    ("/api/v1/file/report.html", b"text/html", False),
])
def test_owner_cannot_execute_uploaded_active_content_but_reports_remain_available(tmp_path, path, mime, protected):
    owner = Principal("owner", "owner", ("observe", "generate", "manage", "admin"), True)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", mime)]})
        await send({"type": "http.response.body", "body": b"asset"})

    boundary = AccessBoundary(app, runtime=FakeRuntime(tmp_path, owner))
    scope = request_scope(body=b"", key="read-asset", path=path)
    scope["method"] = "GET"
    messages = run_request(boundary, scope, b"")
    assert response_status(messages) == 200
    headers = dict(next(message["headers"] for message in messages if message["type"] == "http.response.start"))
    assert (b"content-security-policy" in headers) is protected
    if protected:
        assert headers[b"content-disposition"] == b"attachment"
        assert headers[b"x-content-type-options"] == b"nosniff"


def multipart(boundary: str, *, prompt: str = "draw", filename: str = "frame.png", image: bytes = b"PNG-bytes") -> bytes:
    return (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{prompt}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + image + f"\r\n--{boundary}--\r\n".encode()


def request_scope(*, body: bytes, key: str, content_type: str = "application/json", path: str = "/api/v1/generate"):
    return {
        "type": "http",
        "method": "POST",
        "scheme": "https",
        "path": path,
        "query_string": b"",
        "headers": [
            (b"host", b"maestro.test"),
            (b"authorization", b"Bearer agent-token"),
            (b"content-type", content_type.encode("latin1")),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"idempotency-key", key.encode("ascii")),
        ],
        "client": ("198.51.100.22", 49000),
        "route": SimpleNamespace(path=path),
    }


def chunked_receive(body: bytes, chunk_size: int = 8192):
    offset = 0
    if not body:
        yield {"type": "http.request", "body": b"", "more_body": False}
    while offset < len(body):
        end = min(offset + chunk_size, len(body))
        yield {
            "type": "http.request",
            "body": body[offset:end],
            "more_body": end < len(body),
        }
        offset = end


def run_request(boundary, scope, body: bytes, *, send_callback=None):
    messages = []
    stream = iter(chunked_receive(body))

    async def receive():
        try:
            return next(stream)
        except StopIteration:
            return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if send_callback is not None:
            result = send_callback(message)
            if asyncio.iscoroutine(result):
                await result

    asyncio.run(boundary(scope, receive, send))
    return messages


def response_body(messages) -> bytes:
    return b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")


def response_status(messages) -> int:
    return next(message["status"] for message in messages if message["type"] == "http.response.start")


def setup(
    tmp_path,
    app,
    *,
    principal=None,
    additional_media_roots=None,
    dynamic_scope_resolver=None,
    app_root=None,
):
    principal = principal or Principal("grant-binary", "client-binary", ("observe", "generate", "manage", "admin"), False)
    runtime = FakeRuntime(tmp_path / "settings", principal)
    runtime.directory.mkdir(parents=True, exist_ok=True)
    journal = SubmissionJournal(runtime.directory)
    boundary = AccessBoundary(
        app,
        runtime=runtime,
        journal=journal,
        app_root=tmp_path if app_root is None else app_root,
        additional_media_roots=additional_media_roots,
        dynamic_scope_resolver=dynamic_scope_resolver,
    )
    return boundary, runtime, journal


def test_multipart_is_spooled_hashed_replayed_once_and_cleaned(tmp_path):
    body = multipart("maestro-boundary-1", image=b"image-content" * 4096)
    calls = []
    native_request = {}

    async def app(scope, receive, send):
        calls.append(1)
        headers = dict(scope["headers"])
        chunks = []
        while True:
            message = await receive()
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        native_request["body"] = b"".join(chunks)
        native_request["headers"] = headers
        await send({"type": "http.response.start", "status": 201, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"job_id":"job-1"}'})

    boundary, runtime, journal = setup(tmp_path, app)
    scope = request_scope(body=body, key="multipart-key", content_type="multipart/form-data; boundary=maestro-boundary-1")
    first = run_request(boundary, scope, body)
    assert response_status(first) == 201
    assert native_request["body"] == body
    assert int(native_request["headers"][b"content-length"]) == len(body)
    assert b"transfer-encoding" not in native_request["headers"]
    assert calls == [1]
    assert list(runtime.directory.glob(".agent-access-upload-*")) == []

    retry = run_request(boundary, scope, body)
    assert response_status(retry) == 201
    assert response_body(retry) == b'{"job_id":"job-1"}'
    assert calls == [1]
    assert list(runtime.directory.glob(".agent-access-upload-*")) == []

    changed = multipart("maestro-boundary-1", prompt="different", image=b"image-content" * 4096)
    changed_scope = request_scope(body=changed, key="multipart-key", content_type="multipart/form-data; boundary=maestro-boundary-1")
    conflict = run_request(boundary, changed_scope, changed)
    assert response_status(conflict) == 409
    assert calls == [1]

    changed_type_scope = request_scope(
        body=body,
        key="multipart-key",
        content_type="multipart/form-data; boundary=maestro-boundary-1; charset=utf-8",
    )
    changed_type = run_request(boundary, changed_type_scope, body)
    assert response_status(changed_type) == 409
    assert calls == [1]
    records = journal.list_reservations("grant-binary")
    assert len(records) == 1
    assert hashlib.sha256(body).hexdigest() not in (runtime.directory / "agent-access-journal.sqlite3").read_bytes().decode("latin1", errors="ignore")


def test_multipart_rejects_path_traversal_and_redacted_credential_and_cleans_spool(tmp_path):
    calls = []

    async def app(scope, receive, send):
        calls.append(1)
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    boundary, runtime, _journal = setup(tmp_path, app)
    traversal = (
        b"--maestro-boundary-2\r\n"
        b'Content-Disposition: form-data; name="image_path"\r\n\r\n'
        b"../../outside.png\r\n"
        b"--maestro-boundary-2--\r\n"
    )
    scope = request_scope(body=traversal, key="traversal-key", content_type="multipart/form-data; boundary=maestro-boundary-2")
    rejected = run_request(boundary, scope, traversal)
    assert response_status(rejected) == 403
    assert calls == []
    assert list(runtime.directory.glob(".agent-access-upload-*")) == []

    camel_case_traversal = (
        b"--maestro-boundary-2b\r\n"
        b'Content-Disposition: form-data; name="imagePath"\r\n\r\n'
        b"../../outside.png\r\n"
        b"--maestro-boundary-2b--\r\n"
    )
    camel_case_scope = request_scope(
        body=camel_case_traversal,
        key="camel-path-key",
        content_type="multipart/form-data; boundary=maestro-boundary-2b",
    )
    camel_rejected = run_request(boundary, camel_case_scope, camel_case_traversal)
    assert response_status(camel_rejected) == 403
    assert calls == []
    assert list(runtime.directory.glob(".agent-access-upload-*")) == []

    filename_traversal = multipart("maestro-boundary-2c", filename="../outside.png")
    filename_scope = request_scope(
        body=filename_traversal,
        key="filename-path-key",
        content_type="multipart/form-data; boundary=maestro-boundary-2c",
    )
    filename_rejected = run_request(boundary, filename_scope, filename_traversal)
    assert response_status(filename_rejected) == 400
    assert calls == []
    assert list(runtime.directory.glob(".agent-access-upload-*")) == []

    placeholder = (
        b"--maestro-boundary-3\r\n"
        b'Content-Disposition: form-data; name="apiKey"\r\n\r\n'
        b"[redacted]\r\n"
        b"--maestro-boundary-3--\r\n"
    )
    placeholder_scope = request_scope(body=placeholder, key="placeholder-key", content_type="multipart/form-data; boundary=maestro-boundary-3")
    placeholder_response = run_request(boundary, placeholder_scope, placeholder)
    assert response_status(placeholder_response) == 400
    assert calls == []
    assert list(runtime.directory.glob(".agent-access-upload-*")) == []


def test_oversized_chunked_multipart_is_rejected_and_partial_spool_removed(tmp_path, monkeypatch):
    import services.agent_access.boundary as boundary_module

    monkeypatch.setattr(boundary_module, "MAX_MULTIPART_BODY", 64)
    body = b"x" * 100
    calls = []

    async def app(scope, receive, send):
        calls.append(1)

    boundary, runtime, _journal = setup(tmp_path, app)
    scope = request_scope(body=body, key="oversized-key", content_type="multipart/form-data; boundary=unused")
    scope["headers"] = [(k, v) for k, v in scope["headers"] if k != b"content-length"]
    rejected = run_request(boundary, scope, body)
    assert response_status(rejected) == 413
    assert calls == []
    assert list(runtime.directory.glob(".agent-access-upload-*")) == []


def test_multipart_dispatch_failure_marks_key_uncertain_and_cleans_spool(tmp_path):
    body = multipart("maestro-boundary-fail")

    async def app(scope, receive, send):
        while (await receive()).get("more_body"):
            pass
        raise RuntimeError("native operation failed after dispatch")

    boundary, runtime, journal = setup(tmp_path, app)
    scope = request_scope(body=body, key="dispatch-failed", content_type="multipart/form-data; boundary=maestro-boundary-fail")
    with pytest.raises(RuntimeError, match="after dispatch"):
        run_request(boundary, scope, body)

    record = journal.list_reservations("grant-binary")[0]
    assert journal.lookup("grant-binary", record["reservation_id"])["state"] == "uncertain"
    assert list(runtime.directory.glob(".agent-access-upload-*")) == []


def test_multipart_upload_slots_bound_concurrent_native_dispatch(tmp_path):
    body = multipart("maestro-boundary-concurrent")
    condition = threading.Condition()
    release = threading.Event()
    entered = []

    async def app(scope, receive, send):
        while (await receive()).get("more_body"):
            pass
        with condition:
            entered.append(True)
            condition.notify_all()
        await asyncio.to_thread(release.wait, 5)
        await send({"type": "http.response.start", "status": 201, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    boundary, runtime, _journal = setup(tmp_path, app)
    results = {}

    def submit(name):
        scope = request_scope(
            body=body,
            key=f"concurrent-{name}",
            content_type="multipart/form-data; boundary=maestro-boundary-concurrent",
        )
        results[name] = run_request(boundary, scope, body)

    first = threading.Thread(target=submit, args=("a",))
    second = threading.Thread(target=submit, args=("b",))
    first.start()
    second.start()
    deadline = time.monotonic() + 5
    with condition:
        while len(entered) < 2 and time.monotonic() < deadline:
            condition.wait(timeout=0.05)
    assert len(entered) == 2

    third_scope = request_scope(
        body=body,
        key="concurrent-c",
        content_type="multipart/form-data; boundary=maestro-boundary-concurrent",
    )
    rejected = run_request(boundary, third_scope, body)
    assert response_status(rejected) == 429
    assert len(entered) == 2

    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert response_status(results["a"]) == 201
    assert response_status(results["b"]) == 201
    assert list(runtime.directory.glob(".agent-access-upload-*")) == []


def test_binary_response_streams_over_eight_mib_and_replays_only_a_receipt(tmp_path):
    request_body = b'{"operation":"export"}'
    total_bytes = 9 * 1024 * 1024
    chunk = b"z" * (128 * 1024)
    chunk_count = total_bytes // len(chunk)
    calls = []
    streamed = {"start": False, "count": 0, "size": 0}

    async def app(scope, receive, send):
        calls.append(1)
        request_message = await receive()
        assert json.loads(request_message["body"]) == {"operation": "export"}
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/octet-stream"), (b"content-length", str(total_bytes).encode())],
        })
        assert streamed["start"] is True, "the boundary must forward headers before consuming the native file"
        for index in range(chunk_count):
            await send({
                "type": "http.response.body",
                "body": chunk,
                "more_body": index + 1 < chunk_count,
            })

    def sink(message):
        if message["type"] == "http.response.start":
            streamed["start"] = True
        elif message["type"] == "http.response.body":
            streamed["count"] += 1
            streamed["size"] += len(message.get("body", b""))

    boundary, _runtime, journal = setup(tmp_path, app)
    scope = request_scope(body=request_body, key="binary-export-key")
    first = run_request(boundary, scope, request_body, send_callback=sink)
    assert response_status(first) == 200
    first_headers = dict(next(message for message in first if message["type"] == "http.response.start")["headers"])
    assert first_headers[b"content-type"] == b"application/octet-stream"
    assert streamed["size"] == total_bytes
    assert streamed["count"] == chunk_count
    assert calls == [1]

    saved = journal.list_reservations("grant-binary")[0]
    receipt = journal.lookup("grant-binary", saved["reservation_id"])
    assert receipt["state"] == "complete"
    receipt_json = json.loads(receipt["body"])
    assert receipt_json == {
        "kind": "binary_receipt",
        "delivery_verified": False,
        "bytes": total_bytes,
        "sha256": hashlib.sha256(chunk * chunk_count).hexdigest(),
        "content_type": "application/octet-stream",
        "message": "The native binary response was delivered once; it cannot be regenerated with this idempotency key.",
    }

    retry = run_request(boundary, scope, request_body)
    assert response_status(retry) == 200
    retry_headers = dict(next(message for message in retry if message["type"] == "http.response.start")["headers"])
    assert retry_headers[b"content-type"] == b"application/json"
    assert json.loads(response_body(retry)) == receipt_json
    assert calls == [1]


def test_binary_disconnect_marks_reservation_uncertain(tmp_path):
    request_body = b'{"operation":"export"}'

    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/zip")]})
        await send({"type": "http.response.body", "body": b"partial archive", "more_body": False})

    boundary, _runtime, journal = setup(tmp_path, app)
    scope = request_scope(body=request_body, key="disconnect-key")
    messages = []
    stream = iter(chunked_receive(request_body))

    async def receive():
        return next(stream)

    async def disconnecting_send(message):
        messages.append(message)
        if message["type"] == "http.response.body":
            raise RuntimeError("client disconnected")

    with pytest.raises(RuntimeError, match="disconnected"):
        asyncio.run(boundary(scope, receive, disconnecting_send))

    assert response_status(messages) == 200
    record = journal.list_reservations("grant-binary")[0]
    assert journal.lookup("grant-binary", record["reservation_id"])["state"] == "uncertain"


def test_redacted_json_credentials_are_rejected_without_changing_owner_payloads(tmp_path):
    calls = []

    async def app(scope, receive, send):
        calls.append(1)
        body = await receive()
        assert json.loads(body["body"]) == {"api_key": "[redacted]"}
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    boundary, _runtime, _journal = setup(tmp_path, app)
    body = b'{"api_key":"[redacted]"}'
    scope = request_scope(body=body, key="redacted-json-key")
    rejected = run_request(boundary, scope, body)
    assert response_status(rejected) == 400
    assert calls == []

    owner = Principal("owner", "owner", ("observe", "generate", "manage", "admin"), True)
    owner_boundary, _runtime, _journal = setup(tmp_path / "owner", app, principal=owner)
    owner_scope = request_scope(body=body, key="owner-redacted-key")
    owner_response = run_request(owner_boundary, owner_scope, body)
    assert response_status(owner_response) == 200
    assert calls == [1]


def test_trusted_custom_media_root_is_allowed_but_siblings_and_traversal_are_denied(tmp_path):
    custom_root = tmp_path / "MaestroOutputs"
    custom_root.mkdir()
    sibling_root = tmp_path / "MaestroOutputs-backup"
    sibling_root.mkdir()
    allowed_path = custom_root / "session" / "result.png"
    allowed_path.parent.mkdir()
    allowed_path.write_bytes(b"image")
    sibling_path = sibling_root / "result.png"
    sibling_path.write_bytes(b"image")
    calls = []

    async def app(scope, receive, send):
        calls.append(1)
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    boundary, _runtime, _journal = setup(
        tmp_path,
        app,
        additional_media_roots=lambda: custom_root,
    )

    allowed_body = json.dumps({"image_path": str(allowed_path)}).encode()
    allowed = run_request(boundary, request_scope(body=allowed_body, key="custom-root-ok"), allowed_body)
    assert response_status(allowed) == 200
    assert calls == [1]

    for key, unsafe_path in (
        ("custom-sibling", sibling_path),
        ("custom-traversal", custom_root / ".." / sibling_root.name / "result.png"),
    ):
        unsafe_body = json.dumps({"image_path": str(unsafe_path)}).encode()
        denied = run_request(boundary, request_scope(body=unsafe_body, key=key), unsafe_body)
        assert response_status(denied) == 403
    assert calls == [1]


def test_boundary_without_native_root_defers_json_media_path_to_backend(tmp_path):
    principal = Principal("grant-controller", "client-controller", ("generate",), False)
    runtime = FakeRuntime(tmp_path / "settings", principal)
    runtime.directory.mkdir(parents=True, exist_ok=True)
    journal = SubmissionJournal(runtime.directory)
    calls = []

    async def app(scope, receive, send):
        calls.append(json.loads((await receive())["body"]))
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    boundary = AccessBoundary(app, runtime=runtime, journal=journal, app_root=None)
    body = json.dumps({"image_path": str(tmp_path.parent / "user-output" / "image.png")}).encode()
    response = run_request(boundary, request_scope(body=body, key="defer-media-path"), body)
    assert response_status(response) == 200
    assert calls == [json.loads(body)]


def test_custom_media_root_symlink_escape_and_callback_failure_fail_closed(tmp_path):
    custom_root = tmp_path / "TrustedOutputs"
    custom_root.mkdir()
    outside_root = tmp_path / "OutsideOutputs"
    outside_root.mkdir()
    outside_path = outside_root / "image.png"
    outside_path.write_bytes(b"image")
    escape_link = custom_root / "escape"
    try:
        escape_link.symlink_to(outside_root, target_is_directory=True)
    except (OSError, NotImplementedError):
        symlink_available = False
    else:
        symlink_available = True

    calls = []

    async def app(scope, receive, send):
        calls.append(1)
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    boundary, _runtime, _journal = setup(tmp_path, app, additional_media_roots=lambda: custom_root)
    if symlink_available:
        escaped_body = json.dumps({"image_path": str(escape_link / outside_path.name)}).encode()
        escaped = run_request(boundary, request_scope(body=escaped_body, key="symlink-escape"), escaped_body)
        assert response_status(escaped) == 403
        assert calls == []

    def failed_roots():
        raise RuntimeError("private config source unavailable")

    failing_boundary, _runtime, _journal = setup(tmp_path / "failed", app, additional_media_roots=failed_roots)
    failed_body = json.dumps({"image_path": str(outside_path)}).encode()
    failed = run_request(failing_boundary, request_scope(body=failed_body, key="bad-root-callback"), failed_body)
    assert response_status(failed) == 403
    assert calls == []


def test_multipart_custom_media_root_is_passed_to_form_path_validation(tmp_path):
    custom_root = tmp_path / "CustomOutputs"
    custom_root.mkdir()
    allowed = custom_root / "sample.png"
    allowed.write_bytes(b"image")
    sibling_root = tmp_path / "CustomOutputs-copy"
    sibling_root.mkdir()
    sibling = sibling_root / "sample.png"
    sibling.write_bytes(b"image")
    calls = []

    async def app(scope, receive, send):
        calls.append(1)
        while (await receive()).get("more_body"):
            pass
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    boundary, _runtime, _journal = setup(
        tmp_path,
        app,
        additional_media_roots=lambda: custom_root,
    )

    def form_body(path):
        return (
            b"--maestro-media-root\r\n"
            b'Content-Disposition: form-data; name="imagePath"\r\n\r\n'
            + str(path).encode()
            + b"\r\n--maestro-media-root--\r\n"
        )

    for key, value, expected_status in (
        ("custom-form-allowed", allowed, 200),
        ("custom-form-sibling", sibling, 403),
    ):
        body = form_body(value)
        scope = request_scope(
            body=body,
            key=key,
            content_type="multipart/form-data; boundary=maestro-media-root",
        )
        response = run_request(boundary, scope, body)
        assert response_status(response) == expected_status
    assert calls == [1]


def test_dynamic_transfer_scope_resolver_precedes_static_scope_and_fails_closed(tmp_path):
    principal = Principal("grant-observe", "client-observe", ("observe",), False)
    resolver_calls = []
    app_calls = []

    async def app(scope, receive, send):
        app_calls.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    def resolve_scope(caller, method, path):
        resolver_calls.append((caller.id, method, path))
        return "observe"

    boundary, _runtime, _journal = setup(
        tmp_path,
        app,
        principal=principal,
        dynamic_scope_resolver=resolve_scope,
    )

    def get_request(path):
        return {
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "path": path,
            "query_string": b"",
            "headers": [(b"host", b"maestro.test"), (b"authorization", b"Bearer agent-token")],
            "client": ("198.51.100.22", 49000),
            "route": SimpleNamespace(path=path),
        }

    for path in (
        "/api/v1/agent-access/transfers/transfer-id",
        "/api/v1/agent-access/transfers/transfer-id/chunk",
    ):
        messages = run_request(boundary, get_request(path), b"")
        assert response_status(messages) == 200
    assert resolver_calls == [
        (principal.id, "GET", "/api/v1/agent-access/transfers/transfer-id"),
        (principal.id, "GET", "/api/v1/agent-access/transfers/transfer-id/chunk"),
    ]

    # Static observe routes bypass the transfer resolver and keep their prior policy.
    normal = run_request(boundary, get_request("/api/v1/system-stats"), b"")
    assert response_status(normal) == 200
    assert len(resolver_calls) == 2

    def broken_resolver(_caller, _method, _path):
        raise RuntimeError("transfer lookup unavailable")

    failing, _runtime, _journal = setup(
        tmp_path / "failing-resolver",
        app,
        principal=principal,
        dynamic_scope_resolver=broken_resolver,
    )
    denied = run_request(failing, get_request("/api/v1/agent-access/transfers/missing"), b"")
    assert response_status(denied) == 403
    assert app_calls == [
        "/api/v1/agent-access/transfers/transfer-id",
        "/api/v1/agent-access/transfers/transfer-id/chunk",
        "/api/v1/system-stats",
    ]


def test_redact_scrubs_credential_url_queries_and_preserves_other_components():
    url = (
        "https://example.test/resource?view=thumbnail&token=token-value"
        "&apiKey=api-value&apikey=legacy-api&access_token=access-value"
        "&refresh_token=refresh-value&client_secret=client-value#frame-4"
    )
    result = redact({"url": url, "message": f"Open {url}."})
    expected = (
        "https://example.test/resource?view=thumbnail&token=%5Bredacted%5D"
        "&apiKey=%5Bredacted%5D&apikey=%5Bredacted%5D&access_token=%5Bredacted%5D"
        "&refresh_token=%5Bredacted%5D&client_secret=%5Bredacted%5D#frame-4"
    )
    assert result["url"] == expected
    assert result["message"] == f"Open {expected}."
    for secret in ("token-value", "api-value", "legacy-api", "access-value", "refresh-value", "client-value"):
        assert secret not in result["url"]


def test_submission_context_links_owned_resources_before_disconnect_without_redispatch(tmp_path):
    from services.agent_access.context import submission_context

    principal = Principal("grant-recovery", "client-recovery", ("observe", "generate"), False)
    runtime = FakeRuntime(tmp_path / "settings", principal)
    runtime.directory.mkdir(parents=True, exist_ok=True)
    journal = SubmissionJournal(runtime.directory)
    dispatches = []
    observed_context = []

    async def app(scope, receive, send):
        dispatches.append(f"{scope['method']} {scope['path']}")
        observed_context.append(submission_context.get())
        if scope["method"] == "POST":
            await receive()
            journal.record_job(
                {"id": "job-before-disconnect", "status": "running", "prompt": "private"},
                principal.id,
                "session-recovery",
            )
            journal.register_resource_owner("pipeline", "pipeline-before-disconnect", principal.id, "session-recovery")
            await send({"type": "http.response.start", "status": 202, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"job_id":', "more_body": True})
            raise RuntimeError("native connection disconnected after creating the job")
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    boundary = AccessBoundary(app, runtime=runtime, journal=journal, app_root=tmp_path)
    body = b'{"prompt":"private request"}'
    scope = request_scope(body=body, key="request-for-recovery")

    def receive_for(body_bytes):
        chunks = iter(chunked_receive(body_bytes))

        async def receive():
            try:
                return next(chunks)
            except StopIteration:
                return {"type": "http.disconnect"}

        return receive

    async def invoke_and_check_reset():
        async def downstream_send(_message):
            return None

        try:
            await boundary(scope, receive_for(body), downstream_send)
        except RuntimeError as error:
            assert "disconnected" in str(error)
        else:
            raise AssertionError("the simulated disconnect should escape the boundary")
        after_error = submission_context.get()

        get_scope = {
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "path": "/api/v1/system-stats",
            "query_string": b"",
            "headers": [(b"host", b"maestro.test"), (b"authorization", b"Bearer agent-token")],
            "client": ("198.51.100.22", 49000),
        }

        async def send_ok(_message):
            return None

        await boundary(get_scope, receive_for(b""), send_ok)
        return after_error, submission_context.get()

    after_error, after_unjournaled = asyncio.run(invoke_and_check_reset())

    assert dispatches == ["POST /api/v1/generate", "GET /api/v1/system-stats"]
    assert observed_context[0] is not None
    assert observed_context[1] is None
    assert after_error is None
    assert after_unjournaled is None
    found = journal.find_submission(principal.id, "request-for-recovery")
    assert found["state"] == "uncertain"
    assert {(resource["kind"], resource["id"]) for resource in found["resources"]} == {
        ("job", "job-before-disconnect"),
        ("pipeline", "pipeline-before-disconnect"),
    }
    assert journal.get_job("job-before-disconnect", principal.id)["snapshot"]["status"] == "running"

    retry = run_request(boundary, scope, body)
    assert response_status(retry) == 409
    assert dispatches == ["POST /api/v1/generate", "GET /api/v1/system-stats"]
