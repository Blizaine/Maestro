"""CPU integration of Pinokio's HTTPS web route with the existing MCP boundary."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import os
import socket
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

if any(importlib.util.find_spec(name) is None for name in ("mcp", "fastapi", "httpx")):
    raise unittest.SkipTest("CPU controller integration dependencies are not installed")

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.agent_access.boundary import AccessBoundary, cookie_name
from services.agent_access.bridge import BackendClient
from services.agent_access.controller import create_app
from services.agent_access.pinokio_proxy import PinokioNativeMcpProxy, _signature
from services.agent_access.state import RuntimeState, atomic_json
from services.agent_access.tls import ensure_tls

ORIGIN = "https://pinokio.example-tail.ts.net:43004"
EVIDENCE = {"available": True, "phase": "ready", "browser_url": ORIGIN + "/?maestro_ui_policy=lan-v2",
            "dns_name": "pinokio.example-tail.ts.net", "node_hostname": "TEST-PC", "app_id": "Maestro-dev"}


class Discovery:
    def status(self):
        return dict(EVIDENCE)


def setup(tmp_path):
    root = tmp_path / "Maestro-dev" / "app"
    root.mkdir(parents=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    runtime = RuntimeState(root / "settings" / "agent-access", app_root=root, port=port)
    runtime.register_backend("http://127.0.0.1:42015", ready=True, session_id="test", auth_boundary_version=3)
    session = runtime.auth.redeem_owner_code(runtime.auth.issue_owner_code())
    owner = runtime.auth.authenticate(session["access_token"])
    runtime.auth.set_enabled(True, owner)
    runtime.configure_pinokio_remote(True)
    runtime._pinokio_remote_evidence = dict(EVIDENCE)
    runtime._pinokio_remote_discovery = Discovery()
    material = ensure_tls(runtime.directory / "tls", runtime.installation_id)
    atomic_json(runtime.directory / "controller.json", {"status": "ready", "pid": os.getpid(), "pinokio_remote": EVIDENCE})
    bridge = BackendClient(runtime, client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"ok": True}))))
    controller = create_app(runtime, bridge=bridge, tls_material=material, frame_ancestor_origins=())
    native = FastAPI()

    @native.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
    def native_route(path: str):
        return {"native": True, "path": path}

    factory_calls = []
    def factory(**kwargs):
        factory_calls.append(kwargs)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=controller, client=("127.0.0.1", 9123)))

    gateway = PinokioNativeMcpProxy(native, runtime, discovery=Discovery(), client_factory=factory)
    return runtime, owner, session, controller, gateway, factory_calls


def run(coro):
    return asyncio.run(coro)


def test_metadata_unauthenticated_challenge_and_full_oauth_over_published_route(tmp_path):
    runtime, owner, session, controller, gateway, calls = setup(tmp_path)
    async def scenario():
        async with controller.router.lifespan_context(controller):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 1234)), base_url=ORIGIN) as client:
                metadata = await client.get("/.well-known/oauth-authorization-server")
                assert metadata.status_code == 200, metadata.text
                assert metadata.json()["issuer"] == ORIGIN
                resource = await client.get("/.well-known/oauth-protected-resource/mcp")
                assert resource.json()["resource"] == ORIGIN + "/mcp"
                challenge = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
                assert challenge.status_code == 401
                assert ORIGIN + "/.well-known/oauth-protected-resource/mcp" in challenge.headers["www-authenticate"]
                registration = await client.post("/oauth/register", json={"client_name": "Private remote Mac", "redirect_uris": ["http://127.0.0.1:49173/callback"]})
                assert registration.status_code == 201, registration.text
                registered = registration.json()
                verifier = "pinokio-test-0123456789012345678901234567890123456789"
                pkce = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
                query = {"response_type": "code", "client_id": registered["client_id"],
                         "redirect_uri": registered["redirect_uris"][0], "code_challenge": pkce,
                         "code_challenge_method": "S256", "resource": ORIGIN + "/mcp", "scope": "observe", "state": "test-state"}
                pending = await client.get("/oauth/authorize?" + urlencode(query))
                assert pending.status_code == 303
                assert pending.headers["location"].startswith("/agent-access/login?")
                # Browser login uses the same remote origin and retains CSRF checks.
                login = await client.post("/api/v1/agent-access/owner-login", json={"code": runtime.auth.issue_owner_code()}, headers={"Origin": ORIGIN})
                assert login.status_code == 200, login.text
                assert "Secure" in login.headers["set-cookie"]
                assert client.cookies.get(cookie_name(runtime, "https"))
                pending = await client.get("/oauth/authorize?" + urlencode(query))
                assert pending.status_code == 303, pending.text
                consent_url = pending.headers["location"]
                assert (await client.get(consent_url)).status_code == 200
                approval = await client.post(consent_url, data={"decision": "approve", "scope": "observe"}, headers={"Origin": ORIGIN})
                assert approval.status_code == 303, approval.text
                code = parse_qs(urlsplit(approval.headers["location"]).query)["code"][0]
                tokens = await client.post("/oauth/token", data={"grant_type": "authorization_code", "client_id": registered["client_id"], "code": code, "redirect_uri": query["redirect_uri"], "code_verifier": verifier, "resource": ORIGIN + "/mcp"})
                assert tokens.status_code == 200, tokens.text
                token = tokens.json()["access_token"]
                client.cookies.clear()
                tools = await client.post("/mcp", headers={"Authorization": "Bearer " + token, "Accept": "application/json, text/event-stream"}, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
                assert tools.status_code == 200, tools.text
                assert tools.json()["result"]["tools"]
                denied = await client.post("/api/v1/agent-access/pinokio", headers={"Authorization": "Bearer " + token}, json={"enabled": False})
                assert denied.status_code == 403
    run(scenario())
    assert calls and all(call["trust_env"] is False and call["follow_redirects"] is False for call in calls)
    assert all(call["verify"].verify_mode == 2 for call in calls)
    assert "pinokio_proxy_key" not in str(runtime.public_status())


@pytest.mark.parametrize("mutation", ["remote_peer", "http", "bad_signature", "expired", "wrong_origin", "wrong_path", "duplicate"])
def test_controller_rejects_forged_origin_envelopes(tmp_path, mutation):
    runtime, _owner, _session, controller, _gateway, _calls = setup(tmp_path)
    import time
    timestamp = str(int(time.time()) - (120 if mutation == "expired" else 0))
    path = "/outside" if mutation == "wrong_path" else "/.well-known/oauth-authorization-server"
    origin = "https://evil.example" if mutation == "wrong_origin" else ORIGIN
    scope = {"method": "GET", "raw_path": path.encode(), "query_string": b""}
    headers = [("x-maestro-pinokio-origin", origin), ("x-maestro-pinokio-time", timestamp),
               ("x-maestro-pinokio-signature", "bad" if mutation == "bad_signature" else _signature(runtime.auth.pinokio_proxy_key(), scope, origin, timestamp).decode())]
    if mutation == "duplicate":
        headers.append(("x-maestro-pinokio-origin", ORIGIN))
    peer = "192.168.1.4" if mutation == "remote_peer" else "127.0.0.1"
    base = runtime.local_url.replace("https:", "http:") if mutation == "http" else runtime.local_url
    with TestClient(controller, base_url=base, client=(peer, 1234)) as client:
        assert client.get(path, headers=headers).status_code == 421


def test_remote_proxy_never_grants_local_owner_or_accepts_cross_origin(tmp_path):
    runtime, _owner, _session, controller, gateway, _calls = setup(tmp_path)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 1234)), base_url=ORIGIN) as client:
            status = await client.get("/api/v1/agent-access/status")
            assert status.status_code == 200
            assert status.json()["local_approval_available"] is False
            assert status.json()["authenticated"] is False
            csrf = await client.post("/oauth/register", headers={"Origin": "https://evil.example"}, json={"redirect_uris": ["http://127.0.0.1:41000/callback"]})
            assert csrf.status_code == 403
            local = await client.post("/api/v1/agent-access/local-challenge", headers={"Origin": ORIGIN})
            assert local.status_code == 403
    run(scenario())


def test_disabled_stale_and_other_routes_retain_native_access_and_grants(tmp_path):
    runtime, owner, _session, _controller, gateway, calls = setup(tmp_path)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 1234)), base_url=ORIGIN) as client:
            assert (await client.get("/")).json()["native"] is True
            assert (await client.get("http://192.168.1.7:42015/api/v1/models")).json()["native"] is True
            assert (await client.get("https://old.tail-other.ts.net/api/v1/agent-access/status")).json()["native"] is True
            runtime.configure_pinokio_remote(False)
            assert (await client.get("/mcp")).json()["native"] is True
            assert runtime.auth.enabled and runtime.auth.authenticate(_session["access_token"]).owner
            runtime.configure_pinokio_remote(True)
            gateway.discovery = type("Offline", (), {"status": lambda _: {"phase": "offline"}})()
            assert (await client.get("/mcp")).json()["native"] is True
            runtime._pinokio_remote_discovery = gateway.discovery
            runtime.refresh_pinokio_remote()
            assert runtime.pinokio_remote_status()["ready"] is False
            assert runtime.public_status()["pinokio_remote"]["mcp_url"] is None
            runtime.auth.set_enabled(False, owner)
            assert runtime.pinokio_remote_status()["ready"] is False
    run(scenario())
    assert calls == []


def test_verified_portal_can_embed_ui_without_wildcards_or_cached_old_policy(tmp_path):
    runtime, _owner, _session, _controller, gateway, _calls = setup(tmp_path)
    async def html_app(scope, receive, send):
        from starlette.responses import HTMLResponse
        await HTMLResponse("<html>Maestro</html>")(scope, receive, send)
    gateway.app = AccessBoundary(html_app, runtime=runtime, allow_legacy_browser=True, frame_ancestor_origins=())
    client = TestClient(gateway, base_url=ORIGIN, client=("127.0.0.1", 1234))
    try:
        result = client.get("/", headers={"If-None-Match": '"old"'})
        assert result.status_code == 200
        assert "https://pinokio.example-tail.ts.net" in result.headers["content-security-policy"]
        assert "https://*.ts.net" not in result.headers["content-security-policy"]
        assert result.headers["cache-control"] == "no-store"
    finally:
        client.close()


def test_uvicorn_forwarded_client_does_not_hide_actual_pinokio_proxy_peer(tmp_path):
    _runtime, _owner, _session, controller, gateway, _calls = setup(tmp_path)
    from services.server_startup import PreserveTransportPeer
    app = PreserveTransportPeer(gateway)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1234)), base_url=ORIGIN) as client:
            metadata = await client.get("/.well-known/oauth-authorization-server", headers={
                "X-Forwarded-For": "100.64.0.20", "X-Forwarded-Proto": "https"})
            assert metadata.status_code == 200
            assert metadata.json()["issuer"] == ORIGIN
            status = await client.get("/api/v1/agent-access/status", headers={
                "X-Forwarded-For": "100.64.0.20", "X-Forwarded-Proto": "https"})
            assert status.json()["local_approval_available"] is False
    run(scenario())


def test_mcp_settings_require_owner_and_current_native_bridge(tmp_path):
    runtime, _owner, session, controller, _gateway, _calls = setup(tmp_path)
    runtime.configure_pinokio_remote(False)
    with TestClient(controller, base_url=runtime.local_url, client=("127.0.0.1", 1234)) as client:
        assert client.post("/api/v1/agent-access/pinokio", json={"enabled": True}, headers={"Origin": runtime.local_url}).status_code == 401
        headers = {"Authorization": "Bearer " + session["access_token"], "Origin": runtime.local_url}
        assert client.post("/api/v1/agent-access/pinokio", json={"enabled": "yes"}, headers=headers).status_code == 400
        runtime.register_backend("http://127.0.0.1:42015", ready=True, session_id="old", auth_boundary_version=2)
        assert client.post("/api/v1/agent-access/pinokio", json={"enabled": True}, headers=headers).status_code == 409
        runtime.register_backend("http://127.0.0.1:42015", ready=True, session_id="new", auth_boundary_version=3)
        assert client.post("/api/v1/agent-access/pinokio", json={"enabled": True}, headers=headers).status_code == 200
        assert runtime.config["pinokio_remote_enabled"] is True
        assert client.post("/api/v1/agent-access/pinokio", json={"enabled": False}, headers=headers).status_code == 200
        assert runtime.auth.enabled


def test_transfer_ranges_and_stream_cleanup_cross_proxy_without_redirect_or_retry(tmp_path):
    _runtime, _owner, _session, _controller, gateway, _calls = setup(tmp_path)
    seen = []
    closed = []
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"
            yield b"second"
        async def aclose(self):
            closed.append(True)
    async def dispatch(request):
        seen.append(request)
        assert await request.aread() == b""
        assert request.headers["range"] == "bytes=0-10"
        assert "x-forwarded-for" not in request.headers
        return httpx.Response(206, headers={"Content-Type": "application/octet-stream", "Content-Range": "bytes 0-10/99"}, stream=Stream())
    gateway.client_factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 1234)), base_url=ORIGIN) as client:
            result = await client.get("/api/v1/agent-access/transfers/test/chunk", headers={"Range": "bytes=0-10", "Authorization": "Bearer test", "X-Forwarded-For": "spoof"})
            assert result.status_code == 206
            assert result.content == b"firstsecond"
            assert result.headers["content-range"] == "bytes 0-10/99"
    run(scenario())
    assert len(seen) == 1
    assert closed


def test_untrusted_native_tls_ca_is_never_disabled(tmp_path):
    runtime, _owner, _session, _controller, gateway, calls = setup(tmp_path)
    (runtime.directory / "tls" / "server-cert.pem").write_bytes(b"not a certificate")
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 1234)), base_url=ORIGIN) as client:
            result = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            assert result.status_code == 503
    run(scenario())
    assert calls == []


def test_status_shape_is_complete_before_pinokio_opt_in(tmp_path):
    runtime, _owner, _session, _controller, _gateway, _calls = setup(tmp_path)
    runtime.configure_pinokio_remote(False)
    runtime._pinokio_remote_evidence = {}
    atomic_json(runtime.directory / "controller.json", {"status": "ready", "pid": os.getpid()})
    status = runtime.public_status()["pinokio_remote"]
    assert status == {"available": False, "phase": "unsupported", "browser_url": None,
                      "node_hostname": None, "dns_name": None, "app_id": None, "reason": None,
                      "enabled": False, "ready": False, "mcp_url": None}


def test_sanitized_discovery_works_without_cookie_on_direct_controller_ui(tmp_path):
    runtime, _owner, _session, controller, _gateway, _calls = setup(tmp_path)
    with TestClient(controller, base_url=runtime.local_url, client=("127.0.0.1", 1234)) as client:
        status = client.get("/api/v1/remote-access/pinokio/status")
        assert status.status_code == 200
        assert status.json() == EVIDENCE
        assert "access_token" not in status.text
