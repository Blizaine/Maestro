"""Browser cookie regressions for the shared Agent Access login routes."""

from __future__ import annotations

import importlib.util
import socket
import sys
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("fastapi", "httpx")):
    raise unittest.SkipTest("Agent Access login integration dependencies are not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.boundary import AccessBoundary, cookie_name
from services.agent_access.http import management_router
from services.agent_access.state import RuntimeState


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _app(tmp_path: Path):
    port = _free_port()
    runtime = RuntimeState(tmp_path / "settings" / "agent-access", port=port)
    runtime.register_backend(
        f"http://127.0.0.1:{_free_port()}", ready=False, session_id="proxy-login-test",
    )
    issued_codes: list[str] = []

    def show_code() -> None:
        # The real callback writes only to the private Pinokio terminal.
        issued_codes.append(runtime.auth.issue_owner_code())

    inner = FastAPI()
    inner.include_router(management_router(runtime, request_console_code=show_code))
    return AccessBoundary(inner, runtime=runtime), runtime, issued_codes


def _request_code(client: TestClient, origin: str, codes: list[str]) -> str:
    response = client.post(
        f"{origin}/api/v1/agent-access/request-console-code",
        headers={"Origin": origin},
    )
    assert response.status_code == 200, response.text
    assert "code" not in response.json()
    assert codes
    return codes.pop()


def test_http_and_https_same_host_use_independent_owner_cookies_and_form_recovers(tmp_path):
    app, runtime, codes = _app(tmp_path)
    https_origin = runtime.local_url
    http_origin = "http://127.0.0.1:42015"

    with TestClient(app, base_url=https_origin, client=("127.0.0.1", 12345)) as browser:
        https_code = _request_code(browser, https_origin, codes)
        https_login = browser.post(
            f"{https_origin}/api/v1/agent-access/owner-login",
            data={"code": https_code},
            headers={"Origin": https_origin},
            follow_redirects=False,
        )
        assert https_login.status_code == 303
        assert cookie_name(runtime, "https") in https_login.headers["set-cookie"]
        assert "Secure" in https_login.headers["set-cookie"]
        assert browser.get(f"{https_origin}/api/v1/agent-access/status").json()["owner"] is True

        # Pinokio's native UI is HTTP on loopback. It shares the hostname with
        # the local HTTPS controller but must not collide with its Secure
        # browser cookie, even though browser cookies are not port-scoped.
        legacy_session = runtime.auth.redeem_owner_code(runtime.auth.issue_owner_code())
        runtime.mark_security_configured()
        browser.cookies.set(cookie_name(runtime), legacy_session["access_token"], domain="127.0.0.1", path="/")
        http_code = _request_code(browser, http_origin, codes)
        http_login = browser.post(
            f"{http_origin}/api/v1/agent-access/owner-login",
            data={"code": http_code},
            headers={"Origin": http_origin},
            follow_redirects=False,
        )
        assert http_login.status_code == 303
        assert cookie_name(runtime, "http") in http_login.headers["set-cookie"]
        assert "Secure" not in http_login.headers["set-cookie"]
        status = browser.get(f"{http_origin}/api/v1/agent-access/status").json()
        assert status["authenticated"] is True
        assert status["owner"] is True
        assert status["enabled"] is False

        # A consumed one-time code returns a usable HTML recovery page rather
        # than a JSON exception page; it never echoes the submitted code.
        replay = browser.post(
            f"{http_origin}/api/v1/agent-access/owner-login",
            data={"code": http_code},
            headers={"Origin": http_origin},
            follow_redirects=False,
        )
        assert replay.status_code == 401
        assert replay.headers["content-type"].startswith("text/html")
        assert "invalid, expired, or already used" in replay.text
        assert "Show a code in Pinokio" in replay.text
        assert http_code not in replay.text
        assert 'name="code"' in replay.text

        logout = browser.post(
            f"{http_origin}/api/v1/agent-access/logout",
            headers={"Origin": http_origin},
        )
        assert logout.status_code == 200
        deleted = logout.headers.get_list("set-cookie")
        assert any(cookie_name(runtime, "http") in value and "Max-Age=0" in value for value in deleted)
        assert any(cookie_name(runtime) in value and "Max-Age=0" in value for value in deleted)
        assert browser.get(f"{http_origin}/api/v1/agent-access/status").json()["authenticated"] is False


def test_tailscale_style_https_origin_sets_secure_cookie_and_authenticates_same_origin(tmp_path):
    app, runtime, codes = _app(tmp_path)
    # TestClient's HTTPS origin represents the externally secure side of a
    # trusted loopback proxy such as Tailscale Serve.
    https_origin = f"https://maestro.tailnet.example:{runtime.port}"
    with TestClient(app, base_url=https_origin, client=("127.0.0.1", 12345)) as browser:
        code = _request_code(browser, https_origin, codes)
        login = browser.post(
            "/api/v1/agent-access/owner-login",
            data={"code": code},
            headers={"Origin": https_origin},
            follow_redirects=False,
        )
        assert login.status_code == 303
        assert cookie_name(runtime, "https") in login.headers["set-cookie"]
        assert "Secure" in login.headers["set-cookie"]
        status = browser.get("/api/v1/agent-access/status").json()
        assert status["authenticated"] is True
        assert status["owner"] is True


def test_wrapped_or_full_terminal_line_is_accepted_without_echoing_code(tmp_path):
    app, runtime, codes = _app(tmp_path)
    origin = runtime.local_url
    with TestClient(app, base_url=origin, client=("127.0.0.1", 12345)) as browser:
        code = _request_code(browser, origin, codes)
        pasted_line = "[Maestro Agent Access] Browser code: " + code[:17] + "\n" + code[17:]
        login = browser.post(
            "/api/v1/agent-access/owner-login",
            json={"code": pasted_line},
            headers={"Origin": origin},
        )
        assert login.status_code == 200
        assert login.json() == {"ok": True}
        assert code not in login.text
        assert browser.get("/api/v1/agent-access/status").json()["owner"] is True
