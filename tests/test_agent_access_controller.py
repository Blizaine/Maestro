"""CPU-only integration coverage for the built-in Agent Access controller."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import socket
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

if any(importlib.util.find_spec(name) is None for name in ("mcp", "fastapi", "httpx")):
    raise unittest.SkipTest("Agent Access controller integration dependencies are not installed")

import httpx
import pytest
from fastapi.testclient import TestClient

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.auth import Principal
from services.agent_access.boundary import cookie_name
from services.agent_access.bridge import BackendClient
from services.agent_access.controller import create_app
from services.agent_access.state import RuntimeState
from services.agent_access.tls import ensure_tls


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _runtime(tmp_path: Path) -> RuntimeState:
    app_root = tmp_path / "maestro-app"
    app_root.mkdir()
    for name in ("uploads", "outputs", "outputs2", "storage", "temp", "ckpts", "loras", "loras_i2v"):
        (app_root / name).mkdir()
    runtime = RuntimeState(tmp_path / "settings" / "agent-access", app_root=app_root, port=_port())
    runtime.register_backend(
        f"http://127.0.0.1:{_port()}",
        ready=False,
        session_id="controller-integration-test",
    )
    return runtime


class _AsyncBody(httpx.AsyncByteStream):
    def __init__(self, body: bytes):
        self.body = body

    async def __aiter__(self):
        yield self.body

    async def aclose(self):
        return None


def _app(runtime: RuntimeState, handler=None, console=None, tls_material=None):
    backend_calls: list[httpx.Request] = []

    def dispatch(request: httpx.Request) -> httpx.Response:
        backend_calls.append(request)
        response = handler(request) if handler is not None else httpx.Response(200, json={"ok": True})
        if isinstance(response.stream, _AsyncBody):
            return response
        body = response.content
        headers = dict(response.headers)
        status_code = response.status_code
        response.close()
        return httpx.Response(status_code, headers=headers, stream=_AsyncBody(body))

    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(dispatch)))
    app = create_app(
        runtime,
        bridge=bridge,
        console_printer=(console if console is not None else []).append,
        tls_material=tls_material,
        frame_ancestor_origins=(),
    )
    return app, backend_calls


def test_public_certificate_download_contains_only_the_fixed_public_certificate(tmp_path):
    from services.agent_access.tls import ensure_tls

    runtime = _runtime(tmp_path)
    material = ensure_tls(runtime.directory / "tls", runtime.installation_id)
    app, _calls = _app(runtime, tls_material=material)
    with TestClient(app, base_url=runtime.origin) as client:
        result = client.get("/api/v1/agent-access/certificate?filename=server-key.pem")
        assert result.status_code == 200
        assert result.content == material.cert_path.read_bytes()
        assert b"PRIVATE KEY" not in result.content
        assert result.headers["content-disposition"].startswith("attachment;")
        assert runtime.auth.status()["owner_configured"] is False


def _login_owner(client: TestClient, app, runtime: RuntimeState, console: list[str], code: str | None = None) -> str:
    if code is None:
        app.state.show_code()
        line = next(line for line in reversed(console) if "Browser code:" in line)
        code = line.split("Browser code:", 1)[1].strip()
    response = client.post(
        "/api/v1/agent-access/owner-login",
        json={"code": code},
        headers={"Origin": runtime.origin},
    )
    assert response.status_code == 200, response.text
    assert code not in response.text
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=lax" in cookie
    assert client.cookies.get(cookie_name(runtime, "https"))
    return code


def _pkce() -> tuple[str, str]:
    verifier = "maestro-controller-test-code-verifier-012345678901234567890123"
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode("ascii").rstrip("=")
    return verifier, challenge


def _agent_token(runtime: RuntimeState, owner: Principal, scopes=("observe",)) -> dict:
    verifier, challenge = _pkce()
    client = runtime.auth.register_client({
        "client_name": "MCP integration test",
        "redirect_uris": ["http://127.0.0.1:49173/callback"],
    })
    pending = runtime.auth.create_authorization(
        client["client_id"],
        client["redirect_uris"][0],
        scopes,
        challenge,
        runtime.mcp_url,
        "controller-test-state",
    )
    approved = runtime.auth.approve_authorization(pending["request_id"], owner, scopes)
    return runtime.auth.exchange_code(
        client["client_id"], approved["code"], client["redirect_uris"][0], verifier, runtime.mcp_url,
    )


def test_secure_owner_onboarding_and_off_on_preserve_owner_and_grants(tmp_path):
    runtime = _runtime(tmp_path)
    console: list[str] = []
    app, _ = _app(runtime, console=console)

    with TestClient(app, base_url=runtime.origin, client=("127.0.0.1", 12345)) as secure:
        status = secure.get("/api/v1/agent-access/status").json()
        assert status["enabled"] is False
        assert status["owner_configured"] is False

        console_response = secure.post(
            "/api/v1/agent-access/request-console-code", headers={"Origin": runtime.origin},
        )
        assert console_response.status_code == 200
        assert "code" not in console_response.json()

        code = next(line.split("Browser code:", 1)[1].strip() for line in console if "Browser code:" in line)
        remote_origin = f"http://{runtime.hostname}:{runtime.port}"
        page = secure.get(f"{remote_origin}/agent-access/login")
        assert page.status_code == 200
        assert 'name="code"' not in page.text
        assert runtime.origin in page.text
        insecure_login = secure.post(
            f"{remote_origin}/api/v1/agent-access/owner-login",
            json={"code": code},
            headers={"Origin": remote_origin},
        )
        assert insecure_login.status_code == 426
        assert code not in insecure_login.text

        wrong_origin = secure.post(
            "/api/v1/agent-access/owner-login",
            json={"code": code},
            headers={"Origin": "https://attacker.invalid"},
        )
        assert wrong_origin.status_code == 403
        assert code not in wrong_origin.text
        assert _login_owner(secure, app, runtime, console, code=code) == code

        owner = runtime.auth.authenticate(secure.cookies.get(cookie_name(runtime, "https")))
        settings = secure.post(
            "/api/v1/agent-access/settings", json={"enabled": True}, headers={"Origin": runtime.origin},
        )
        assert settings.status_code == 200
        token = _agent_token(runtime, owner)
        assert secure.get("/api/v1/agent-access/grants").json()["grants"]
        disabled = secure.post(
            "/api/v1/agent-access/settings", json={"enabled": False}, headers={"Origin": runtime.origin},
        )
        assert disabled.status_code == 200
        assert secure.get("/mcp", headers={"Authorization": f"Bearer {token['access_token']}"}).status_code == 404
        # Off pauses MCP. It does not revoke owner sessions or agent grants.
        assert secure.get("/api/v1/agent-access/status").json()["owner"] is True
        assert secure.get("/api/v1/agent-access/grants").json()["grants"]
        anonymous_owner_route = secure.get(
            "/api/v1/agent-access/grants", headers={"Authorization": "Bearer invalid-after-off"},
        )
        assert anonymous_owner_route.status_code == 401
        enabled = secure.post(
            "/api/v1/agent-access/settings", json={"enabled": True}, headers={"Origin": runtime.origin},
        )
        assert enabled.status_code == 200
        assert runtime.auth.authenticate(token["access_token"]).owner is False


@pytest.mark.parametrize("backend_proof", ["missing", "legacy"])
def test_owner_login_requires_updated_backend_proof_without_consuming_code(tmp_path, backend_proof):
    runtime = _runtime(tmp_path)
    console: list[str] = []
    app, _ = _app(runtime, console=console)
    if backend_proof == "missing":
        (runtime.directory / "backend.json").unlink()
    else:
        (runtime.directory / "backend.json").write_text(json.dumps({
            "schema": 1,
            "installation_id": runtime.installation_id,
            "url": f"http://127.0.0.1:{_port()}",
            "ready": False,
            "session_id": "legacy-backend",
        }), encoding="utf-8")

    with TestClient(app, base_url=runtime.origin, client=("127.0.0.1", 12345)) as client:
        requested = client.post(
            "/api/v1/agent-access/request-console-code",
            headers={"Origin": runtime.origin},
        )
        assert requested.status_code == 200
        code = next(line.split("Browser code:", 1)[1].strip() for line in console if "Browser code:" in line)

        denied = client.post(
            "/api/v1/agent-access/owner-login",
            json={"code": code},
            headers={"Origin": runtime.origin},
        )
        assert denied.status_code == 409
        assert code not in denied.text

        runtime.register_backend(
            f"http://127.0.0.1:{_port()}",
            ready=False,
            session_id="updated-backend-proof",
        )
        accepted = client.post(
            "/api/v1/agent-access/owner-login",
            json={"code": code},
            headers={"Origin": runtime.origin},
        )
        assert accepted.status_code == 200, accepted.text
        assert code not in accepted.text


def test_oauth_routes_bind_pkce_redirect_resource_and_refresh_family(tmp_path):
    runtime = _runtime(tmp_path)
    console: list[str] = []
    app, _ = _app(runtime, console=console)
    with TestClient(app, base_url=runtime.origin, client=("127.0.0.1", 12345), follow_redirects=False) as client:
        _login_owner(client, app, runtime, console)
        assert client.post("/api/v1/agent-access/settings", json={"enabled": True}, headers={"Origin": runtime.origin}).status_code == 200

        resource = client.get("/.well-known/oauth-protected-resource/mcp").json()
        assert resource["resource"] == runtime.mcp_url
        unauthenticated = client.get(
            "/mcp",
            headers={"Authorization": "Bearer invalid-token", "Accept": "application/json, text/event-stream"},
        )
        assert unauthenticated.status_code == 401
        challenge = unauthenticated.headers["www-authenticate"]
        assert f'resource_metadata="{runtime.origin}/.well-known/oauth-protected-resource/mcp"' in challenge
        assert 'scope="observe"' in challenge
        assert "invalid-token" not in unauthenticated.text

        registered = client.post(
            "/oauth/register",
            json={"client_name": "Codex Desktop", "redirect_uris": ["http://127.0.0.1:49173/callback"]},
            headers={"Origin": runtime.origin},
        )
        assert registered.status_code == 201, registered.text
        metadata = registered.json()
        verifier, challenge = _pkce()
        authorize = client.get("/oauth/authorize", params={
            "response_type": "code",
            "client_id": metadata["client_id"],
            "redirect_uri": metadata["redirect_uris"][0],
            "scope": "observe generate",
            "resource": runtime.mcp_url,
            "state": "anti-csrf-state",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }, headers={"Origin": runtime.origin})
        assert authorize.status_code == 303
        consent_url = authorize.headers["location"]
        consent = client.get(consent_url)
        assert consent.status_code == 200
        approved = client.post(
            consent_url,
            data={"decision": "approve", "scope": ["observe"]},
            headers={"Origin": runtime.origin},
        )
        assert approved.status_code == 303
        callback = urlsplit(approved.headers["location"])
        callback_query = parse_qs(callback.query)
        assert callback.netloc == "127.0.0.1:49173"
        assert callback_query["state"] == ["anti-csrf-state"]
        code = callback_query["code"][0]

        body = {
            "grant_type": "authorization_code",
            "client_id": metadata["client_id"],
            "code": code,
            "redirect_uri": metadata["redirect_uris"][0],
            "code_verifier": verifier,
            "resource": runtime.mcp_url,
        }
        wrong_resource = client.post("/oauth/token", json={**body, "resource": runtime.mcp_url + "/wrong"})
        assert wrong_resource.status_code == 400
        assert code not in wrong_resource.text
        wrong_verifier = client.post("/oauth/token", json={**body, "code_verifier": "x" * len(verifier)})
        assert wrong_verifier.status_code == 400
        assert code not in wrong_verifier.text
        wrong_redirect = client.post("/oauth/token", json={**body, "redirect_uri": "http://127.0.0.1:49173/other"})
        assert wrong_redirect.status_code == 400
        assert code not in wrong_redirect.text
        issued = client.post("/oauth/token", json=body)
        assert issued.status_code == 200, issued.text
        first = issued.json()
        assert first["scope"] == "observe"
        assert code not in issued.text
        assert "anti-csrf-state" not in issued.text

        rotation = client.post("/oauth/token", json={
            "grant_type": "refresh_token",
            "client_id": metadata["client_id"],
            "refresh_token": first["refresh_token"],
            "resource": runtime.mcp_url,
        })
        assert rotation.status_code == 200
        second = rotation.json()
        assert second["access_token"] != first["access_token"]
        assert second["refresh_token"] != first["refresh_token"]
        revoked = client.post("/oauth/revoke", json={"token": second["access_token"]})
        assert revoked.status_code == 200
        revoked_use = client.get(
            "/api/v1/system-stats", headers={"Authorization": f"Bearer {second['access_token']}"},
        )
        assert revoked_use.status_code == 401

        replay = client.post("/oauth/token", json={
            "grant_type": "refresh_token",
            "client_id": metadata["client_id"],
            "refresh_token": first["refresh_token"],
            "resource": runtime.mcp_url,
        })
        assert replay.status_code == 400
        assert first["refresh_token"] not in replay.text
        assert client.get(
            "/api/v1/system-stats", headers={"Authorization": f"Bearer {first['access_token']}"},
        ).status_code == 401


@pytest.mark.parametrize("protocol_version", ["2024-11-05", "2025-03-26"])
def test_authenticated_mcp_handshake_calls_scoped_backend_tool(tmp_path, protocol_version):
    runtime = _runtime(tmp_path)
    owner_code = runtime.auth.issue_owner_code()
    owner_session = runtime.auth.redeem_owner_code(owner_code)
    runtime.mark_security_configured()
    owner = runtime.auth.authenticate(owner_session["access_token"])
    runtime.auth.set_enabled(True, owner)
    grant = _agent_token(runtime, owner, ("observe",))
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="mock-backend")
    seen: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["authorization"] == f"Bearer {grant['access_token']}"
        if request.url.path == "/api/v1/system-stats":
            return httpx.Response(200, json={
                "status": "ready",
                "apiKey": "never-return-this",
                "api_key": "also-secret",
                "secretKey": "secret-key-never-return-this",
            })
        return httpx.Response(404, json={"detail": "not found"})

    app, _ = _app(runtime, handler=backend)
    with TestClient(app, base_url=runtime.origin, client=("127.0.0.1", 12345)) as client:
        headers = {
            "Authorization": f"Bearer {grant['access_token']}",
            "Origin": runtime.origin,
            "Accept": "application/json, text/event-stream",
        }
        initialize = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "controller-test", "version": "1"},
            },
        })
        assert initialize.status_code == 200, initialize.text
        assert initialize.json()["result"]["protocolVersion"] == protocol_version
        initialized = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        assert initialized.status_code in {200, 202, 204}
        tools = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        assert tools.status_code == 200, tools.text
        names = {item["name"] for item in tools.json()["result"]["tools"]}
        assert {"get_status", "prepare_generation", "execute_operation"}.issubset(names)

        result = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_status", "arguments": {}},
        })
        assert result.status_code == 200, result.text
        assert not result.json().get("error")
        assert len(seen) == 1
        tool_result = result.json()["result"]
        serialized = json.dumps(tool_result)
        assert "never-return-this" not in serialized
        assert "also-secret" not in serialized
        assert "secret-key-never-return-this" not in serialized

        direct_status = client.get("/api/v1/system-stats", headers=headers)
        assert direct_status.status_code == 200
        assert direct_status.json()["apiKey"] == "[redacted]"
        assert direct_status.json()["api_key"] == "[redacted]"
        assert direct_status.json()["secretKey"] == "[redacted]"
        assert len(seen) == 2

        direct_admin = client.post(
            "/api/v1/system-config",
            headers={**headers, "Idempotency-Key": "scope-denial-1"},
            json={"debug": True},
        )
        assert direct_admin.status_code == 403
        assert len(seen) == 2

        # A scoped grant can list tools, but a generation tool is stopped by
        # the bridge before an authenticated backend request is made.
        denied = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "prepare_generation", "arguments": {"body": {}, "request_id": "stable-1"}},
        })
        assert denied.status_code == 200
        assert denied.json()["result"]["isError"] is True
        assert "requires generate permission" in denied.json()["result"]["content"][0]["text"]
        assert len(seen) == 2


def test_controller_proxy_does_not_expose_closed_backend_details(tmp_path):
    runtime = _runtime(tmp_path)
    owner_code = runtime.auth.issue_owner_code()
    session = runtime.auth.redeem_owner_code(owner_code)
    runtime.mark_security_configured()
    owner = runtime.auth.authenticate(session["access_token"])
    runtime.auth.set_enabled(True, owner)
    grant = _agent_token(runtime, owner, ("observe",))
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="closed-mock-backend")

    def closed(request: httpx.Request):
        raise httpx.ConnectError("private closed-backend detail", request=request)

    app, _ = _app(runtime, handler=closed)
    with TestClient(app, base_url=runtime.origin, client=("127.0.0.1", 12345)) as client:
        response = client.get(
            "/api/v1/system-stats",
            headers={
                "Authorization": f"Bearer {grant['access_token']}",
                "Origin": runtime.origin,
            },
        )
        assert response.status_code == 503
        assert "private closed-backend detail" not in response.text
        assert "Inspect the same submission receipt" in response.json()["detail"]


def test_boundary_resolves_owned_asset_reference_and_rejects_foreign_id(tmp_path):
    runtime = _runtime(tmp_path)
    owner_code = runtime.auth.issue_owner_code()
    session = runtime.auth.redeem_owner_code(owner_code)
    runtime.mark_security_configured()
    owner = runtime.auth.authenticate(session["access_token"])
    runtime.auth.set_enabled(True, owner)
    grants_before = {item["grant_id"] for item in runtime.auth.list_grants(owner)}
    grant = _agent_token(runtime, owner, ("generate",))
    agent_grant_id = next(
        item["grant_id"] for item in runtime.auth.list_grants(owner) if item["grant_id"] not in grants_before
    )
    foreign_before = {item["grant_id"] for item in runtime.auth.list_grants(owner)}
    foreign_grant = _agent_token(runtime, owner, ("generate",))
    foreign_grant_id = next(item["grant_id"] for item in runtime.auth.list_grants(owner) if item["grant_id"] not in foreign_before)
    assert foreign_grant_id != agent_grant_id
    payload_bytes = b"owned-reference-data"

    observed: list[dict] = []

    def backend(request: httpx.Request) -> httpx.Response:
        observed.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            stream=_AsyncBody(b'{"accepted":true}'),
        )

    app, _ = _app(runtime, handler=backend)
    bridge: BackendClient = app.state.bridge
    upload = bridge.assets.begin(agent_grant_id, "reference.bin", len(payload_bytes), hashlib.sha256(payload_bytes).hexdigest())
    bridge.assets.write_chunk(agent_grant_id, upload["upload_id"], 0, payload_bytes, len(payload_bytes))
    _, staging = bridge.assets.finish(agent_grant_id, upload["upload_id"])
    trusted_file = runtime.app_root / "uploads" / "reference.bin"
    trusted_file.write_bytes(payload_bytes)
    reference = bridge.assets.register(agent_grant_id, upload["upload_id"], trusted_file)
    assert not staging.exists()
    runtime.register_backend("http://127.0.0.1:65534", ready=True, session_id="mock-backend")

    with TestClient(app, base_url=runtime.origin, client=("127.0.0.1", 12345)) as client:
        headers = {"Authorization": f"Bearer {grant['access_token']}", "Origin": runtime.origin}
        accepted = client.post(
            "/api/v1/generate",
            json={"image_path": reference, "prompt": "safe", "request_id": "asset-owned-1"},
            headers={**headers, "Idempotency-Key": "asset-owned-1"},
        )
        assert accepted.status_code == 200, accepted.text
        assert observed[-1]["image_path"] == str(trusted_file.resolve())
        assert reference not in json.dumps(observed[-1])

        foreign = client.post(
            "/api/v1/generate",
            json={"image_path": reference, "prompt": "foreign", "request_id": "asset-foreign-1"},
            headers={
                "Authorization": f"Bearer {foreign_grant['access_token']}",
                "Origin": runtime.origin,
                "Idempotency-Key": "asset-foreign-1",
            },
        )
        # The reference belongs to the first agent, not this grant.
        assert foreign.status_code == 404
        assert len(observed) == 1


def test_authenticated_transfer_reads_ranges_chunks_scope_and_revocation_while_backend_offline(tmp_path):
    runtime = _runtime(tmp_path)
    owner_code = runtime.auth.issue_owner_code()
    session = runtime.auth.redeem_owner_code(owner_code)
    runtime.mark_security_configured()
    owner = runtime.auth.authenticate(session["access_token"])
    runtime.auth.set_enabled(True, owner)
    grants_before = {item["grant_id"] for item in runtime.auth.list_grants(owner)}
    grant = _agent_token(runtime, owner, ("observe", "manage"))
    grant_id = next(
        item["grant_id"] for item in runtime.auth.list_grants(owner)
        if item["grant_id"] not in grants_before
    )
    principal = runtime.auth.authenticate(grant["access_token"])
    payload = b"<!doctype html><html><body>private artifact</body></html>"

    async def artifact_chunks():
        yield payload[:19]
        yield payload[19:]

    app, _ = _app(runtime)
    bridge: BackendClient = app.state.bridge
    artifact = asyncio.run(bridge.transfers.capture(
        principal.id,
        "report.html",
        "text/html; charset=utf-8",
        artifact_chunks(),
        access_scope="manage",
    ))
    assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
    assert bridge.online() is False

    with TestClient(app, base_url=runtime.origin, client=("127.0.0.1", 12345)) as client:
        headers = {
            "Authorization": f"Bearer {grant['access_token']}",
            "Origin": runtime.origin,
        }
        url = f"/api/v1/agent-access/transfers/{artifact['transfer_id']}"
        full = client.get(url, headers=headers)
        assert full.status_code == 200
        assert full.content == payload
        assert full.headers["content-type"].startswith("text/html")
        assert full.headers["content-length"] == str(len(payload))
        assert full.headers["etag"] == f'"{artifact["sha256"]}"'
        assert full.headers["content-disposition"].startswith("attachment;")
        assert "inline" not in full.headers["content-disposition"].casefold()
        assert "sandbox" in full.headers["content-security-policy"].casefold()
        assert full.headers["x-content-type-options"] == "nosniff"
        assert full.headers["cache-control"] == "private, no-store"

        head = client.head(url, headers=headers)
        assert head.status_code == 200
        assert head.content == b""
        assert head.headers["content-length"] == str(len(payload))
        assert head.headers["etag"] == f'"{artifact["sha256"]}"'

        range_response = client.get(url, headers={**headers, "Range": "bytes=2-8"})
        assert range_response.status_code == 206
        assert range_response.content == payload[2:9]
        assert range_response.headers["content-range"] == f"bytes 2-8/{len(payload)}"
        assert range_response.headers["content-length"] == "7"

        suffix_response = client.get(url, headers={**headers, "Range": "bytes=-5"})
        assert suffix_response.status_code == 206
        assert suffix_response.content == payload[-5:]
        assert suffix_response.headers["content-range"] == f"bytes {len(payload) - 5}-{len(payload) - 1}/{len(payload)}"

        invalid_range = client.get(url, headers={**headers, "Range": "bytes=999-"})
        assert invalid_range.status_code == 416
        assert invalid_range.headers["content-range"] == f"bytes */{len(payload)}"
        oversized_range = client.get(url, headers={**headers, "Range": "bytes=" + "9" * 5000 + "-"})
        assert oversized_range.status_code == 416

        chunk = client.get(
            url + "/chunk?offset=3&max_bytes=8",
            headers=headers,
        )
        assert chunk.status_code == 200, chunk.text
        chunk_json = chunk.json()
        assert base64.b64decode(chunk_json["data_base64"]) == payload[3:11]
        assert chunk_json["offset"] == 3
        assert chunk_json["bytes_read"] == 8
        assert chunk_json["sha256"] == artifact["sha256"]

        oversized_chunk = client.get(
            url + "/chunk?offset=0&max_bytes=1048577",
            headers=headers,
        )
        assert oversized_chunk.status_code == 400

        # The artifact belongs to this agent and requires the scope recorded at
        # capture time, even though the URL itself has a static admin default.
        observe_only = _agent_token(runtime, owner, ("observe",))
        observe_denied = client.get(url, headers={
            "Authorization": f"Bearer {observe_only['access_token']}",
            "Origin": runtime.origin,
        })
        assert observe_denied.status_code == 403
        chunk_scope_denied = client.get(url + "/chunk", headers={
            "Authorization": f"Bearer {observe_only['access_token']}",
            "Origin": runtime.origin,
        })
        assert chunk_scope_denied.status_code == 403

        foreign = _agent_token(runtime, owner, ("observe", "manage"))
        foreign_denied = client.get(url, headers={
            "Authorization": f"Bearer {foreign['access_token']}",
            "Origin": runtime.origin,
        })
        assert foreign_denied.status_code in {403, 404}

        assert runtime.auth.revoke_grant(grant_id, owner)
        revoked = client.get(url, headers=headers)
        assert revoked.status_code == 401


def test_tls_ip_alias_metadata_oauth_and_real_mcp_work_without_mdns(tmp_path):
    runtime = _runtime(tmp_path)
    lan_ip = "192.168.50.20"
    alias_origin = f"https://{lan_ip}:{runtime.port}"
    alias_resource = alias_origin + "/mcp"
    material = ensure_tls(
        runtime.directory / "tls",
        runtime.installation_id,
        ip_addresses=[lan_ip],
    )
    runtime.configure_certificate_ips(material.ip_addresses)
    runtime.update(discovery_enabled=False)
    app, _ = _app(runtime, tls_material=material)
    network_metadata = runtime.lan_endpoint_metadata([lan_ip, "192.168.50.21"])
    owner_code = runtime.auth.issue_owner_code()

    with TestClient(
        app,
        base_url=alias_origin,
        client=("192.168.50.44", 43210),
        follow_redirects=False,
    ) as client:
        metadata = client.get("/.well-known/oauth-authorization-server")
        assert metadata.status_code == 200
        assert metadata.json()["issuer"] == alias_origin
        assert metadata.json()["authorization_endpoint"] == alias_origin + "/oauth/authorize"

        resource_metadata = client.get("/.well-known/oauth-protected-resource/mcp")
        assert resource_metadata.status_code == 200
        assert resource_metadata.json()["resource"] == alias_resource
        assert resource_metadata.json()["authorization_servers"] == [alias_origin]

        login = client.post(
            "/api/v1/agent-access/owner-login",
            json={"code": owner_code},
            headers={"Origin": alias_origin},
        )
        assert login.status_code == 200, login.text
        assert client.post(
            "/api/v1/agent-access/settings",
            json={"enabled": True},
            headers={"Origin": alias_origin},
        ).status_code == 200
        (runtime.directory / "controller.json").write_text(json.dumps({
            "status": "ready",
            "pid": os.getpid(),
            "certificate_sha256": material.fingerprint_sha256,
            "discovery_status": "disabled",
            **network_metadata,
        }), encoding="utf-8")
        status = client.get("/api/v1/agent-access/status")
        assert status.status_code == 200
        assert status.json()["enabled"] is True
        assert status.json()["discovery_status"] == "disabled"
        assert status.json()["lan_mcp_urls"] == [alias_resource]

        invalid_challenge = client.get(
            "/mcp",
            headers={"Authorization": "Bearer invalid-alias-token", "Accept": "application/json, text/event-stream"},
        )
        assert invalid_challenge.status_code == 401
        assert f'resource_metadata="{alias_origin}/.well-known/oauth-protected-resource/mcp"' in invalid_challenge.headers["www-authenticate"]
        assert invalid_challenge.json()["detail"]["login_url"] == alias_origin
        assert "invalid-alias-token" not in invalid_challenge.text

        registered = client.post(
            "/oauth/register",
            json={"client_name": "IP alias MCP test", "redirect_uris": ["http://127.0.0.1:49173/callback"]},
            headers={"Origin": alias_origin},
        )
        assert registered.status_code == 201, registered.text
        client_meta = registered.json()
        verifier, challenge = _pkce()
        authorize = client.get("/oauth/authorize", params={
            "response_type": "code",
            "client_id": client_meta["client_id"],
            "redirect_uri": client_meta["redirect_uris"][0],
            "scope": "observe",
            "resource": alias_resource,
            "state": "alias-state",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }, headers={"Origin": alias_origin})
        assert authorize.status_code == 303, authorize.text
        consent_url = authorize.headers["location"]
        approved = client.post(
            consent_url,
            data={"decision": "approve", "scope": "observe"},
            headers={"Origin": alias_origin},
        )
        assert approved.status_code == 303
        callback = urlsplit(approved.headers["location"])
        code = parse_qs(callback.query)["code"][0]
        token_body = {
            "grant_type": "authorization_code",
            "client_id": client_meta["client_id"],
            "code": code,
            "redirect_uri": client_meta["redirect_uris"][0],
            "code_verifier": verifier,
            "resource": alias_resource,
        }
        untrusted_resource = client.post("/oauth/token", json={
            **token_body,
            "resource": f"https://192.168.50.21:{runtime.port}/mcp",
        }, headers={"Origin": alias_origin})
        assert untrusted_resource.status_code == 400
        assert code not in untrusted_resource.text

        token_response = client.post("/oauth/token", json=token_body, headers={"Origin": alias_origin})
        assert token_response.status_code == 200, token_response.text
        token = token_response.json()
        assert code not in token_response.text
        assert not runtime.auth.authenticate(token["access_token"]).owner

        initialized = client.post("/mcp", headers={
            "Authorization": f"Bearer {token['access_token']}",
            "Origin": alias_origin,
            "Accept": "application/json, text/event-stream",
        }, json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "verified-IP-alias", "version": "1"},
            },
        })
        assert initialized.status_code == 200, initialized.text
        assert initialized.json()["result"]["protocolVersion"] == "2025-03-26"

        refresh = client.post("/oauth/token", json={
            "grant_type": "refresh_token",
            "client_id": client_meta["client_id"],
            "refresh_token": token["refresh_token"],
            "resource": alias_resource,
        }, headers={"Origin": alias_origin})
        assert refresh.status_code == 200, refresh.text
        assert refresh.json()["access_token"] != token["access_token"]

        untrusted_ip = "192.168.50.21"
        denied = client.get(
            "/.well-known/oauth-authorization-server",
            headers={"Host": f"{untrusted_ip}:{runtime.port}"},
        )
        assert denied.status_code == 421


def test_existing_tls_without_lan_san_is_not_silently_rotated_or_used_as_ip_alias(tmp_path):
    runtime = _runtime(tmp_path)
    lan_ip = "192.168.50.30"
    material = ensure_tls(runtime.directory / "tls", runtime.installation_id)
    original_certificate = material.cert_path.read_bytes()
    restarted = ensure_tls(runtime.directory / "tls", runtime.installation_id, ip_addresses=[lan_ip])
    assert restarted.cert_path.read_bytes() == original_certificate
    assert lan_ip not in restarted.ip_addresses
    app, _ = _app(runtime, tls_material=restarted)
    with TestClient(
        app,
        base_url=f"https://{lan_ip}:{runtime.port}",
        client=("192.168.50.44", 43210),
    ) as client:
        response = client.get("/.well-known/oauth-authorization-server")
        assert response.status_code == 421
