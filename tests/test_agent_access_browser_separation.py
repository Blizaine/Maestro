"""Native browser access stays separate from Agent Access credentials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.testclient import TestClient

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.auth import Principal  # noqa: E402
from services.agent_access.boundary import AccessBoundary, cookie_name  # noqa: E402
from services.agent_access.context import principal_context  # noqa: E402
from services.agent_access.http import management_router  # noqa: E402
from test_agent_access_controller import _agent_token, _runtime  # noqa: E402


def _configure_owner(runtime, *, enabled: bool) -> Principal:
    runtime.auth.mark_owner_configured()
    runtime.mark_security_configured()
    owner = Principal("fixture-owner", "fixture-owner", ("admin",), True)
    runtime.auth.set_enabled(enabled, owner)
    return owner


def _request_context() -> dict[str, bool]:
    principal = principal_context.get()
    return {"principal_present": principal is not None, "owner": bool(principal and principal.owner)}


def _native_app(runtime, *, always_protect: bool = False):
    inner = FastAPI()

    @inner.get("/")
    async def index():
        return HTMLResponse("<!doctype html><title>Maestro</title><main>Native UI</main>")

    @inner.get("/main.js")
    async def javascript():
        return Response("window.maestroFixture = true;", media_type="application/javascript")

    @inner.get("/api/v1/models")
    async def models():
        return JSONResponse(_request_context())

    @inner.post("/api/v1/generate")
    async def generate():
        return JSONResponse(_request_context())

    @inner.get("/outputs/sample.mp4")
    async def media():
        return Response(b"fixture-video", media_type="video/mp4")

    @inner.get("/uploads/probe.html")
    async def uploaded_html():
        return Response(
            "<script>fixture</script>",
            media_type="text/html",
            headers={
                "Content-Security-Policy": "default-src *",
                "Content-Disposition": "inline",
            },
        )

    @inner.websocket("/ws")
    async def websocket(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json(_request_context())
        await websocket.close()

    @inner.get("/mcp")
    async def mcp():
        return JSONResponse(_request_context())

    inner.include_router(management_router(
        runtime,
        request_console_code=lambda: None,
        local_device_access=True,
    ))
    return AccessBoundary(
        inner,
        runtime=runtime,
        always_protect=always_protect,
        allow_local_ui=True,
        allow_legacy_browser=True,
    )


@pytest.mark.parametrize("enabled", [False, True], ids=["agent-access-off", "agent-access-on"])
def test_native_lan_ui_api_media_and_websocket_ignore_valid_or_stale_owner_cookies(tmp_path, enabled):
    runtime = _runtime(tmp_path)
    _configure_owner(runtime, enabled=enabled)
    owner_session = runtime.auth.issue_owner_session()
    app = _native_app(runtime)
    origin = f"http://192.0.2.10:{runtime.port}"
    cookies = (
        (None, None),
        (cookie_name(runtime, "http"), owner_session["access_token"]),
        (cookie_name(runtime), "expired-fixture-owner-session"),
    )

    with TestClient(app, base_url=origin, client=("192.0.2.55", 49171)) as browser:
        for name, value in cookies:
            headers = {} if name is None else {"Cookie": f"{name}={value}"}

            page = browser.get("/", headers=headers)
            assert page.status_code == 200
            assert "Native UI" in page.text

            script = browser.get("/main.js", headers=headers)
            assert script.status_code == 200
            assert "maestroFixture" in script.text

            api = browser.get("/api/v1/models", headers=headers)
            assert api.status_code == 200
            assert api.json() == {"principal_present": False, "owner": False}

            fake_generation = browser.post(
                "/api/v1/generate",
                json={"fixture": True},
                headers=headers,
            )
            assert fake_generation.status_code == 200
            assert fake_generation.json() == {"principal_present": False, "owner": False}

            media = browser.get("/outputs/sample.mp4", headers=headers)
            assert media.status_code == 200
            assert media.content == b"fixture-video"

            uploaded_html = browser.get("/uploads/probe.html", headers=headers)
            assert uploaded_html.status_code == 200
            assert "sandbox" in uploaded_html.headers["content-security-policy"]
            assert uploaded_html.headers["content-disposition"].startswith("attachment")
            assert uploaded_html.headers["x-content-type-options"] == "nosniff"

            with browser.websocket_connect("/ws", headers=headers) as socket:
                assert socket.receive_json() == {"principal_present": False, "owner": False}


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("enabled", [False, True], ids=["agent-access-off", "agent-access-on"])
def test_remote_anonymous_management_stays_protected_while_public_status_is_redacted(tmp_path, scheme, enabled):
    runtime = _runtime(tmp_path)
    _configure_owner(runtime, enabled=enabled)
    app = _native_app(runtime)
    origin = f"{scheme}://192.0.2.10:{runtime.port}"
    old_label = runtime.config["label"]

    with TestClient(app, base_url=origin, client=("192.0.2.55", 49172)) as remote:
        status = remote.get("/api/v1/agent-access/status")
        assert status.status_code == 200
        public = status.json()
        assert public["enabled"] is enabled
        assert public["authenticated"] is False
        assert public["owner"] is False
        serialized = json.dumps(public).casefold()
        assert not any(secret in serialized for secret in (
            "access_token", "refresh_token", "private_key", "client_secret", "authorization",
        ))

        assert remote.get("/api/v1/agent-access/grants").status_code == 401
        assert remote.post("/api/v1/agent-access/settings", json={"label": "anonymous-change"}).status_code == 401
        assert remote.delete("/api/v1/agent-access/grants/fixture-grant").status_code == 401
        assert runtime.config["label"] == old_label
        assert runtime.auth.enabled is enabled


def test_native_explicit_bearer_still_enforces_validation_scopes_and_transport(tmp_path):
    runtime = _runtime(tmp_path)
    owner = _configure_owner(runtime, enabled=True)
    owner_token = runtime.auth.issue_owner_session()["access_token"]
    observe_token = _agent_token(runtime, owner, scopes=("observe",))["access_token"]
    app = _native_app(runtime)
    cookie = f"{cookie_name(runtime, 'http')}={owner_token}"

    with TestClient(
        app,
        base_url=f"http://192.0.2.10:{runtime.port}",
        client=("192.0.2.55", 49173),
    ) as insecure:
        invalid = insecure.get(
            "/api/v1/models",
            headers={"Authorization": "Bearer invalid-fixture-token", "Cookie": cookie},
        )
        assert invalid.status_code == 401

        secure_transport_required = insecure.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {observe_token}", "Cookie": cookie},
        )
        assert secure_transport_required.status_code == 426

    with TestClient(
        app,
        base_url=f"https://192.0.2.10:{runtime.port}",
        client=("192.0.2.55", 49174),
    ) as secure:
        observed = secure.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {observe_token}", "Cookie": cookie},
        )
        assert observed.status_code == 200
        assert observed.json() == {"principal_present": True, "owner": False}

        insufficient_scope = secure.post(
            "/api/v1/generate",
            json={"fixture": True},
            headers={"Authorization": f"Bearer {observe_token}", "Cookie": cookie},
        )
        assert insufficient_scope.status_code == 403

        not_owner = secure.get(
            "/api/v1/agent-access/grants",
            headers={"Authorization": f"Bearer {observe_token}", "Cookie": cookie},
        )
        assert not_owner.status_code == 403

        owner_mutation = secure.post(
            "/api/v1/agent-access/settings",
            json={"label": "owner-bearer-change"},
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        assert owner_mutation.status_code == 200
        assert runtime.config["label"] == "owner-bearer-change"


def test_same_origin_local_device_management_still_works_with_legacy_browser_opt_in(tmp_path):
    runtime = _runtime(tmp_path)
    _configure_owner(runtime, enabled=False)
    app = _native_app(runtime)
    origin = f"http://127.0.0.1:{runtime.port}"

    with TestClient(app, base_url=origin, client=("127.0.0.1", 49175)) as local:
        updated = local.post(
            "/api/v1/agent-access/settings",
            json={"label": "local-owner-change"},
            headers={"Origin": origin, "Sec-Fetch-Site": "same-origin"},
        )
        assert updated.status_code == 200, updated.text
        assert runtime.config["label"] == "local-owner-change"


def test_always_protect_and_mcp_override_legacy_browser_opt_in(tmp_path):
    runtime = _runtime(tmp_path)
    _configure_owner(runtime, enabled=True)
    app = _native_app(runtime, always_protect=True)

    with TestClient(
        app,
        base_url=f"https://192.0.2.10:{runtime.port}",
        client=("192.0.2.55", 49176),
    ) as remote:
        assert remote.get("/").status_code == 401
        assert remote.get("/mcp").status_code == 401


@pytest.mark.parametrize(("enabled", "expected_status"), [(False, 404), (True, 401)])
def test_native_mcp_remains_gated_with_legacy_browser_opt_in(tmp_path, enabled, expected_status):
    runtime = _runtime(tmp_path)
    _configure_owner(runtime, enabled=enabled)
    app = _native_app(runtime)

    with TestClient(
        app,
        base_url=f"https://192.0.2.10:{runtime.port}",
        client=("192.0.2.55", 49177),
    ) as remote:
        assert remote.get("/mcp").status_code == expected_status
