"""Native loopback OAuth must work without borrowing local browser authority."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from test_agent_access_controller import _app, _pkce, _runtime
from services.agent_access.auth import Principal


def configured(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.auth.mark_owner_configured()
    runtime.mark_security_configured()
    owner = Principal("fixture-owner", "fixture-owner", ("admin",), True)
    runtime.auth.set_enabled(True, owner)
    app, _ = _app(runtime)
    return runtime, owner, app


def test_native_loopback_register_exchange_refresh_revoke_without_browser_origin(tmp_path):
    runtime, owner, app = configured(tmp_path)
    origin = f"https://127.0.0.1:{runtime.port}"
    resource = origin + "/mcp"
    with TestClient(app, base_url=origin, client=("127.0.0.1", 49171)) as client:
        registered = client.post("/oauth/register", json={
            "client_name": "Native OAuth fixture",
            "redirect_uris": ["http://127.0.0.1:49173/callback"],
        })
        assert registered.status_code == 201
        info = registered.json()
        assert client.get("/mcp").status_code == 401
        verifier, challenge = _pkce()
        pending = runtime.auth.create_authorization(
            info["client_id"], info["redirect_uris"][0], ("observe",),
            challenge, runtime.mcp_url, "fixture-state",
        )
        approved = runtime.auth.approve_authorization(pending["request_id"], owner, ("observe",))
        body = {
            "grant_type": "authorization_code", "client_id": info["client_id"],
            "redirect_uri": info["redirect_uris"][0], "code": approved["code"],
            "code_verifier": verifier, "resource": resource,
        }
        wrong = client.post("/oauth/token", data={**body, "code_verifier": "x" * len(verifier)})
        assert wrong.status_code == 400
        issued = client.post("/oauth/token", data=body)
        assert issued.status_code == 200
        first = issued.json()
        assert first["scope"] == "observe"
        refreshed = client.post("/oauth/token", data={
            "grant_type": "refresh_token", "client_id": info["client_id"],
            "refresh_token": first["refresh_token"], "resource": resource,
        })
        assert refreshed.status_code == 200
        second = refreshed.json()
        assert second["access_token"] != first["access_token"]
        assert second["refresh_token"] != first["refresh_token"]
        revoked = client.post("/oauth/revoke", data={"token": second["access_token"]})
        assert revoked.status_code == 200
        assert client.get("/mcp", headers={"Authorization": "Bearer " + second["access_token"]}).status_code == 401
        assert not client.cookies
    with sqlite3.connect(runtime.auth.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM access_tokens WHERE kind='owner'").fetchone()[0] == 0


@pytest.mark.parametrize("endpoint", ["/oauth/register", "/oauth/token", "/oauth/revoke"])
@pytest.mark.parametrize("origin", ["null", "https://untrusted.example"])
def test_native_oauth_still_rejects_explicit_untrusted_browser_origin(tmp_path, endpoint, origin):
    runtime, _, app = configured(tmp_path)
    with TestClient(app, base_url=f"https://127.0.0.1:{runtime.port}", client=("127.0.0.1", 49171)) as client:
        response = client.post(endpoint, json={}, headers={"Origin": origin})
        assert response.status_code == 403


def test_native_oauth_exception_does_not_grant_owner_api_or_mcp_access(tmp_path):
    runtime, _, app = configured(tmp_path)
    with TestClient(app, base_url=f"https://127.0.0.1:{runtime.port}", client=("127.0.0.1", 49171)) as client:
        assert client.post("/api/v1/agent-access/settings", json={"enabled": False}).status_code == 403
        assert client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}).status_code == 401
        assert runtime.auth.enabled is True
