"""Connection refreshes preserve OAuth state; only safe reads may reconnect."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("mcp", "fastapi", "httpx")):
    raise unittest.SkipTest("CPU controller integration dependencies are not installed")

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.agent_access.controller import _listener_configuration, _refresh_pinokio_transport
from test_agent_access_pinokio_remote import Discovery, EVIDENCE, ORIGIN, setup


def test_pinokio_refresh_keeps_same_sdk_and_authorization_state(tmp_path):
    runtime, owner, session, app, gateway, _calls = setup(tmp_path)
    metadata = {"pinokio_remote": EVIDENCE}
    runtime.configure_pinokio_remote(False)
    asyncio.run(_refresh_pinokio_transport(runtime, app, metadata))
    settings = app.state.transport_security
    manager = app.state.mcp.session_manager
    grant_count = len(runtime.auth.list_grants(owner))
    client_info = runtime.auth.register_client({"client_name": "Pending reconnect", "redirect_uris": ["http://127.0.0.1:49173/callback"]})
    pending = runtime.auth.create_authorization(client_info["client_id"], client_info["redirect_uris"][0], ["observe"], "A" * 43, runtime.mcp_url, "kept-state")
    async def scenario():
        async with app.router.lifespan_context(app):
            runtime.configure_pinokio_remote(True)
            await _refresh_pinokio_transport(runtime, app, metadata)
            assert settings is app.state.transport_security
            assert manager is app.state.mcp.session_manager
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 9000)), base_url=ORIGIN) as client:
                assert (await client.get("/.well-known/oauth-authorization-server")).json()["issuer"] == ORIGIN
                headers = {"Authorization": "Bearer " + session["access_token"], "Accept": "application/json, text/event-stream"}
                payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
                tools = await client.post("/mcp", json=payload, headers=headers)
                assert tools.status_code == 200, tools.text
                # A temporary discovery failure removes the alias without
                # restarting or discarding the MCP session manager.
                runtime._pinokio_remote_discovery = type("Offline", (), {"status": lambda _: {"available": False, "phase": "offline"}})()
                await _refresh_pinokio_transport(runtime, app, metadata)
                assert ORIGIN not in settings.allowed_origins
                assert runtime.pinokio_remote_origin is None
                assert manager is app.state.mcp.session_manager
                unavailable = await client.get("/oauth/authorize", headers={"Accept": "text/html"})
                assert unavailable.status_code == 503 and "Reconnecting to Agent Access" in unavailable.text
                runtime._pinokio_remote_discovery = Discovery()
                await _refresh_pinokio_transport(runtime, app, metadata)
                assert ORIGIN in settings.allowed_origins
                assert (await client.get("/.well-known/oauth-authorization-server")).status_code == 200
                tools = await client.post("/mcp", json=payload, headers=headers)
                assert tools.status_code == 200, tools.text
                assert runtime.auth.get_authorization(pending["request_id"])["state"] == "kept-state"
    asyncio.run(scenario())
    assert len(runtime.auth.list_grants(owner)) == grant_count
    assert _listener_configuration({"label": "A", "pinokio_remote_enabled": False}) == _listener_configuration({"label": "A", "pinokio_remote_enabled": True})
    assert _listener_configuration({"label": "A"}) != _listener_configuration({"label": "B"})


def test_get_recovers_short_connection_gap_without_browser_error(tmp_path):
    _runtime, _owner, _session, _app, gateway, _calls = setup(tmp_path)
    attempts = []
    async def handler(request):
        attempts.append(request.method)
        if len(attempts) < 3:
            raise httpx.ConnectError("private details must not be logged", request=request)
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"connected":true}'
        return httpx.Response(200, headers={"Content-Type": "application/json"}, stream=Body())
    gateway.client_factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 9000)), base_url=ORIGIN) as client:
            result = await client.get("/oauth/authorize?state=keep-this", headers={"Accept": "text/html"})
            assert result.status_code == 200 and result.json()["connected"]
    asyncio.run(scenario())
    assert attempts == ["GET", "GET", "GET"]


def test_browser_get_recovery_is_friendly_and_preserves_url(tmp_path, caplog):
    _runtime, _owner, _session, _app, gateway, _calls = setup(tmp_path)
    attempts = []
    async def handler(request):
        attempts.append(request.method)
        raise httpx.ConnectError("token-DO-NOT-LOG", request=request)
    gateway.client_factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 9000)), base_url=ORIGIN) as client:
            result = await client.get("/api/v1/agent-access/authorizations/request-test", headers={"Accept": "text/html"})
            assert result.status_code == 503
            assert result.headers["content-type"].startswith("text/html")
            assert "Reconnecting to Agent Access" in result.text
            assert 'http-equiv="refresh"' in result.text
            assert "/api/v1/agent-access/authorizations/request-test" in result.text
            assert result.headers["cache-control"] == "no-store"
    asyncio.run(scenario())
    assert len(attempts) == 3
    assert "token-DO-NOT-LOG" not in caplog.text


def test_form_post_is_never_repeated_and_does_not_redisplay_code(tmp_path, caplog):
    _runtime, _owner, _session, _app, gateway, _calls = setup(tmp_path)
    bodies = []
    async def handler(request):
        bodies.append(await request.aread())
        raise httpx.RemoteProtocolError("code-secret-DO-NOT-LOG", request=request)
    gateway.client_factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 9000)), base_url=ORIGIN) as client:
            result = await client.post("/api/v1/agent-access/owner-login", data={"code": "code-secret-DO-NOT-LOG"}, headers={"Accept": "text/html", "Referer": ORIGIN + "/agent-access/login?next=%2Foauth%2Fauthorize%3Fstate%3Dkeep"})
            assert result.status_code == 503
            assert "This form was not sent again" in result.text
            assert 'http-equiv="refresh"' not in result.text
            assert 'href="/agent-access/login?next=' in result.text
            assert "code-secret-DO-NOT-LOG" not in result.text
            assert "code-secret-DO-NOT-LOG" not in caplog.text
    asyncio.run(scenario())
    assert len(bodies) == 1


def test_mcp_json_post_remains_single_attempt_machine_readable(tmp_path):
    _runtime, _owner, _session, _app, gateway, _calls = setup(tmp_path)
    attempts = []
    async def handler(request):
        attempts.append(request.method)
        raise httpx.ConnectError("unavailable", request=request)
    gateway.client_factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 9000)), base_url=ORIGIN) as client:
            result = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            assert result.status_code == 503
            assert result.headers["content-type"].startswith("application/json")
            assert result.headers["retry-after"] == "2"
    asyncio.run(scenario())
    assert attempts == ["POST"]


def test_discovery_gap_displays_only_recovery_never_forwards_stale_route(tmp_path):
    _runtime, _owner, _session, _app, gateway, calls = setup(tmp_path)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway, client=("127.0.0.1", 9000)), base_url=ORIGIN) as client:
            assert (await client.get("/")).status_code == 200
            gateway.discovery = type("Offline", (), {"status": lambda _: {"available": False, "phase": "offline"}})()
            response = await client.get("/oauth/authorize?state=keep", headers={"Accept": "text/html"})
            assert response.status_code == 503 and "Continue" in response.text
            assert 'http-equiv="refresh"' in response.text
    asyncio.run(scenario())
    assert calls == []
