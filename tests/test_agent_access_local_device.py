"""Cookie-free local-device enrollment and UI boundary coverage."""

from __future__ import annotations

import json
import secrets
import socket
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.testclient import TestClient

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.auth import Principal  # noqa: E402
from services.agent_access.boundary import (
    AccessBoundary,
    _apply_frame_ancestors,
    cookie_name,
    frame_ancestor_sources,
    login_document,
    pinokio_access_origin,
)  # noqa: E402
from services.agent_access.context import principal_context  # noqa: E402
from services.agent_access.http import management_router  # noqa: E402
from services.agent_access.state import RuntimeState  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _runtime(tmp_path: Path, *, boundary_version: int = 2) -> RuntimeState:
    runtime = RuntimeState(tmp_path / "settings" / "agent-access", port=_port())
    runtime.register_backend(
        f"http://127.0.0.1:{_port()}",
        ready=False,
        session_id="local-device-test",
        auth_boundary_version=boundary_version,
    )
    runtime.update(label="Local device fixture", discovery_enabled=False)
    return runtime


def _app(runtime: RuntimeState, *, frame_ancestor_origins=(), clock=None, always_protect=True):
    inner = FastAPI()

    @inner.get("/")
    async def index():
        return HTMLResponse("<!doctype html><title>Maestro</title><main>Local UI</main>")

    @inner.get("/mcp")
    @inner.get("/mcp/private")
    async def mcp():
        principal = principal_context.get()
        return JSONResponse({"called": True, "owner": bool(principal and principal.owner)})

    @inner.websocket("/ws")
    async def websocket(websocket: WebSocket):
        principal = principal_context.get()
        await websocket.accept()
        await websocket.send_json({"owner": bool(principal and principal.owner)})
        await websocket.close()

    inner.include_router(management_router(
        runtime,
        request_console_code=lambda: None,
        local_approval_clock=clock or (lambda: 1000.0),
        local_device_access=True,
    ))
    return AccessBoundary(
        inner,
        runtime=runtime,
        always_protect=always_protect,
        allow_local_ui=True,
        frame_ancestor_origins=frame_ancestor_origins,
    )


def _origin() -> str:
    return f"http://127.0.0.1:{_port()}"


def _same_origin_headers(origin: str) -> dict[str, str]:
    return {"Origin": origin, "Sec-Fetch-Site": "same-origin"}


def _challenge(client: TestClient, origin: str, proof: str | None = None) -> tuple[str, str]:
    proof = proof or secrets.token_hex(32)
    response = client.post(
        "/api/v1/agent-access/local-challenge",
        json={"proof": proof},
        headers=_same_origin_headers(origin),
    )
    assert response.status_code == 200, response.text
    assert response.headers.get("set-cookie") is None
    return response.json()["nonce"], proof


def _approve(client: TestClient, origin: str, nonce: str, proof: str):
    return client.post(
        "/api/v1/agent-access/local-approve",
        json={"nonce": nonce, "proof": proof, "approve": True},
        headers=_same_origin_headers(origin),
    )


def _access_token_count(runtime: RuntimeState) -> int:
    with runtime.auth._connection() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM access_tokens").fetchone()[0])


def test_local_device_approval_is_cookie_free_explicit_and_persists_across_restart(tmp_path):
    runtime = _runtime(tmp_path)
    app = _app(runtime, frame_ancestor_origins=("http://172.31.48.1:42000",))
    origin = _origin()
    before = runtime.auth.status()
    settings_before = runtime.config

    with TestClient(app, base_url=origin, client=("127.0.0.1", 41234)) as browser:
        page = browser.get("/agent-access/login")
        assert page.status_code == 200
        assert "crypto.getRandomValues" in page.text
        assert "auth_method!=='local_device'" in page.text
        assert runtime.auth.owner_configured is False

        nonce, proof = _challenge(browser, origin)
        approval = _approve(browser, origin, nonce, proof)
        assert approval.status_code == 200, approval.text
        assert approval.json() == {"ok": True}
        assert approval.headers.get("set-cookie") is None
        assert len(browser.cookies) == 0
        assert runtime.auth.owner_configured is True
        assert runtime.auth.enabled is False
        assert runtime.config == settings_before
        assert runtime.auth.status()["grants"] == before["grants"]
        assert _access_token_count(runtime) == 0

        status = browser.get("/api/v1/agent-access/status").json()
        assert status["authenticated"] is True
        assert status["owner"] is True
        assert status["auth_method"] == "local_device"
        assert status["enabled"] is False

        # Pinokio's iframe navigation has no Origin and reports cross-site at
        # the top level; direct loopback remains usable without a cookie.
        ui = browser.get(
            "/",
            headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "iframe"},
        )
        assert ui.status_code == 200
        assert "Local UI" in ui.text
        frame_policy = ui.headers["content-security-policy"]
        assert "frame-ancestors" in frame_policy
        assert "http://172.31.48.1:42000" in frame_policy
        assert "http://127.0.0.1:*" in frame_policy

        replay = _approve(browser, origin, nonce, proof)
        assert replay.status_code == 401

    restarted = RuntimeState(tmp_path / "settings" / "agent-access")
    restarted_app = _app(restarted, frame_ancestor_origins=())
    with TestClient(restarted_app, base_url=origin, client=("127.0.0.1", 41235)) as browser:
        status = browser.get("/api/v1/agent-access/status").json()
        assert status["authenticated"] is True
        assert status["auth_method"] == "local_device"
        assert browser.get("/").status_code == 200


def test_local_device_approval_requires_fresh_matching_proof_and_v2_backend(tmp_path):
    runtime = _runtime(tmp_path)
    clock = Clock()
    app = _app(runtime, clock=clock)
    origin = _origin()
    with TestClient(app, base_url=origin, client=("127.0.0.1", 41236)) as browser:
        nonce, proof = _challenge(browser, origin)
        mismatch = _approve(browser, origin, nonce, secrets.token_hex(32))
        assert mismatch.status_code == 401
        assert runtime.auth.owner_configured is False
        approved = _approve(browser, origin, nonce, proof)
        assert approved.status_code == 200
        assert _approve(browser, origin, nonce, proof).status_code == 401

        expired_nonce, expired_proof = _challenge(browser, origin)
        clock.advance(121)
        expired = _approve(browser, origin, expired_nonce, expired_proof)
        assert expired.status_code == 401

    legacy = _runtime(tmp_path / "legacy", boundary_version=1)
    legacy_app = _app(legacy)
    legacy_origin = _origin()
    with TestClient(legacy_app, base_url=legacy_origin, client=("127.0.0.1", 41237)) as browser:
        nonce, proof = _challenge(browser, legacy_origin)
        rejected = _approve(browser, legacy_origin, nonce, proof)
        assert rejected.status_code == 409
        assert legacy.auth.owner_configured is False
        assert legacy.security_required is False


def test_local_device_authority_is_loopback_only_same_origin_and_never_mcp(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.auth.mark_owner_configured()
    runtime.mark_security_configured()
    app = _app(runtime)
    local_origin = f"http://127.0.0.1:{runtime.port}"

    with TestClient(app, base_url=local_origin, client=("127.0.0.1", 41238)) as browser:
        assert browser.get("/mcp").status_code == 404
        runtime.auth.set_enabled(True, Principal("fixture-owner", "fixture-owner", ("admin",), True))
        assert browser.get("/mcp").status_code == 401
        assert browser.get("/mcp/private").status_code == 401

        invalid_bearer = browser.get(
            "/api/v1/agent-access/grants",
            headers={"Authorization": "Bearer invalid-local-fixture"},
        )
        assert invalid_bearer.status_code == 401

        cross_origin = browser.post(
            "/api/v1/agent-access/request-console-code",
            headers={"Origin": "http://attacker.example", "Sec-Fetch-Site": "cross-site"},
        )
        assert cross_origin.status_code == 403

        no_origin_mutation = browser.post("/api/v1/agent-access/request-console-code")
        assert no_origin_mutation.status_code == 403

        with browser.websocket_connect(
            "/ws",
            headers={
                "Host": f"127.0.0.1:{runtime.port}",
                "Origin": local_origin,
                "Sec-Fetch-Site": "same-origin",
            },
        ) as socket:
            assert socket.receive_json() == {"owner": True}

    remote_origin = f"https://{runtime.hostname}:{runtime.port}"
    with TestClient(app, base_url=remote_origin, client=("192.0.2.55", 41239)) as remote:
        status = remote.get("/api/v1/agent-access/status").json()
        assert status["authenticated"] is False
        assert status["owner"] is False
        assert status.get("auth_method") is None
        assert remote.get("/").status_code == 401

    forwarded_origin = f"http://127.0.0.1:{runtime.port}"
    with TestClient(app, base_url=forwarded_origin, client=("127.0.0.1", 41240)) as forwarded:
        status = forwarded.get(
            "/api/v1/agent-access/status",
            headers={"X-Forwarded-For": "127.0.0.1"},
        ).json()
        assert status["authenticated"] is False
        assert forwarded.get("/", headers={"X-Real-IP": "127.0.0.1"}).status_code == 401


def test_local_device_precedes_browser_cookies_but_bearer_uses_normal_auth(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.auth.issue_owner_session()
    runtime.mark_security_configured()
    app = _app(runtime)
    origin = f"http://127.0.0.1:{runtime.port}"

    with TestClient(app, base_url=origin, client=("127.0.0.1", 41241)) as browser:
        # A stale/valid browser cookie cannot turn the local-device identity
        # into a cookie session that the UI would try to sign out of.
        browser.cookies.set(cookie_name(runtime, "http"), session["access_token"])
        status = browser.get("/api/v1/agent-access/status").json()
        assert status["auth_method"] == "local_device"
        assert status["owner"] is True

        bearer_status = browser.get(
            "/api/v1/agent-access/status",
            headers={"Authorization": "Bearer " + session["access_token"]},
        ).json()
        assert bearer_status["owner"] is True
        assert bearer_status.get("auth_method") is None


def test_pinokio_frame_allowlist_uses_only_valid_private_config_origin(tmp_path):
    config = tmp_path / "pinokio-config.json"
    config.write_text(json.dumps({
        "access": {"protocol": "http", "host": "172.31.48.1", "port": 42000},
    }), encoding="utf-8")
    assert pinokio_access_origin(config) == "http://172.31.48.1:42000"

    config.write_text(json.dumps({
        "access": {"protocol": "http", "host": "8.8.8.8", "port": 42000},
    }), encoding="utf-8")
    assert pinokio_access_origin(config) is None

    sources = frame_ancestor_sources(("http://172.31.48.1:42000", "http://evil.example"))
    assert "http://172.31.48.1:42000" in sources
    assert "http://evil.example" not in sources
    assert "'self'" in sources
    assert "http://127.0.0.1:*" in sources


def test_login_copy_names_pc_management_without_changing_legacy_wording():
    status = {"local_approval_available": True}
    local_device_page = login_document(status, local_device_access=True)
    legacy_page = login_document(status)
    assert "Approve local management on this PC" in local_device_page
    assert "Approve this browser to sign in on this computer" in legacy_page


def test_unconfigured_native_ui_keeps_legacy_local_post_behavior_and_csp_intersects(tmp_path):
    headers = [
        (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'; sandbox"),
        (b"x-content-type-options", b"nosniff"),
    ]
    updated = _apply_frame_ancestors(headers, ("'self'", "http://127.0.0.1:*"))
    policies = [value for name, value in updated if name.lower() == b"content-security-policy"]
    assert policies[0] == b"default-src 'none'; frame-ancestors 'none'; sandbox"
    assert b"frame-ancestors 'self' http://127.0.0.1:*" in policies[1]
    assert (b"x-content-type-options", b"nosniff") in updated

    runtime = _runtime(tmp_path)
    app = _app(runtime, always_protect=False)
    origin = f"http://127.0.0.1:{runtime.port}"
    with TestClient(app, base_url=origin, client=("127.0.0.1", 41242)) as browser:
        assert runtime.auth.owner_configured is False
        assert browser.get("/").status_code == 200
        # Agent Access request-console-code is already public and is only
        # same-origin checked when a browser sends an Origin header.
        assert browser.post("/api/v1/agent-access/request-console-code").status_code == 200
