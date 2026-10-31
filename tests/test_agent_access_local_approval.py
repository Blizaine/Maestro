"""Local-only explicit browser approval for Agent Access owner setup."""

from __future__ import annotations

import base64
import hashlib
import socket
import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.boundary import (
    AccessBoundary,
    cookie_name,
    local_approval_cookie_name,
    local_browser_approval_eligible,
)
from services.agent_access.http import management_router
from services.agent_access.state import RuntimeState


class Clock:
    def __init__(self, now: float = 1_800_000_000.0):
        self._now = now
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _app(tmp_path: Path, *, clock: Clock | None = None):
    runtime = RuntimeState(tmp_path / "settings" / "agent-access", port=_free_port())
    runtime.register_backend(
        f"http://127.0.0.1:{_free_port()}", ready=False, session_id="local-approval-test",
    )
    runtime.update(label="Local approval fixture", discovery_enabled=False)
    inner = FastAPI()
    inner.include_router(management_router(
        runtime,
        request_console_code=lambda: None,
        local_approval_clock=clock or Clock(),
    ))
    return AccessBoundary(inner, runtime=runtime, always_protect=True), runtime


def _local_headers(origin: str) -> dict[str, str]:
    return {"Origin": origin}


def _challenge(client: TestClient, origin: str):
    response = client.post("/api/v1/agent-access/local-challenge", headers=_local_headers(origin))
    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) == {"nonce", "expires_in"}
    assert payload["expires_in"] == 120
    assert "access_token" not in payload and "code" not in payload
    return payload["nonce"], response.headers["set-cookie"]


def _approve(client: TestClient, origin: str, nonce: str):
    return client.post(
        "/api/v1/agent-access/local-approve",
        json={"nonce": nonce, "approve": True},
        headers=_local_headers(origin),
    )


def test_initial_local_http_approval_enrolls_owner_without_enabling_mcp_or_changing_settings(tmp_path):
    app, runtime = _app(tmp_path)
    origin = f"http://127.0.0.1:{_free_port()}"
    before = runtime.auth.status()
    assert before["owner_configured"] is False
    assert before["enabled"] is False
    with TestClient(app, base_url=origin, client=("127.0.0.1", 12345)) as browser:
        status = browser.get("/api/v1/agent-access/status").json()
        assert status["local_approval_available"] is True
        page = browser.get("/agent-access/login?next=%2Foauth%2Fauthorize%3Fclient_id%3Ddemo")
        assert page.status_code == 200
        assert "Approve on this PC" in page.text
        assert "No code or terminal needed" in page.text
        assert "<details><summary>Use a browser sign-in code</summary>" in page.text
        assert 'id="code" name="code" autocomplete="off" required autofocus' not in page.text
        assert 'name="next" value="/oauth/authorize?client_id=demo"' in page.text
        assert "location.assign(n.startsWith('/oauth/authorize?')?n:'/')" in page.text
        assert runtime.auth.status()["owner_configured"] is False
        assert runtime.auth.enabled is False

        nonce, challenge_cookie = _challenge(browser, origin)
        assert "HttpOnly" in challenge_cookie
        assert "Path=/api/v1/agent-access/local-approve" in challenge_cookie
        assert "SameSite=strict" in challenge_cookie
        assert "Secure" not in challenge_cookie
        assert "nonce" not in challenge_cookie
        approval = _approve(browser, origin, nonce)
        assert approval.status_code == 200, approval.text
        assert approval.json() == {"ok": True}
        assert "access_token" not in approval.text and "code" not in approval.text
        owner_cookie = approval.headers["set-cookie"]
        assert cookie_name(runtime, "http") in owner_cookie
        assert "HttpOnly" in owner_cookie and "SameSite=lax" in owner_cookie
        assert "access_token" not in owner_cookie

        after = browser.get("/api/v1/agent-access/status").json()
        assert after["authenticated"] is True
        assert after["owner"] is True
        assert after["enabled"] is False
        assert after["local_approval_available"] is True
        assert runtime.auth.status()["owner_configured"] is True
        assert runtime.auth.status()["grants"] == before["grants"]
        assert runtime.config["label"] == "Local approval fixture"
        assert runtime.config["discovery_enabled"] is False


@pytest.mark.parametrize("origin", [
    "https://localhost:43127",
])
def test_https_loopback_local_approval_sets_secure_port_namespaced_cookies(tmp_path, origin):
    app, runtime = _app(tmp_path)
    with TestClient(app, base_url=origin, client=("::1", 12345)) as browser:
        status = browser.get("/api/v1/agent-access/status").json()
        assert status["local_approval_available"] is True
        nonce, challenge_cookie = _challenge(browser, origin)
        assert "HttpOnly" in challenge_cookie
        assert "Secure" in challenge_cookie
        assert "SameSite=strict" in challenge_cookie
        challenge_name = local_approval_cookie_name(runtime, {
            "type": "http",
            "scheme": "https",
            "client": ("::1", 12345),
            "headers": [(b"host", origin.removeprefix("https://").encode()), (b"origin", origin.encode())],
        })
        assert challenge_name in challenge_cookie
        approval = _approve(browser, origin, nonce)
        assert approval.status_code == 200, approval.text
        assert cookie_name(runtime, "https") in approval.headers["set-cookie"]
        assert "Secure" in approval.headers["set-cookie"]
        assert browser.get("/api/v1/agent-access/status").json()["owner"] is True


def test_ipv6_loopback_scope_is_eligible_and_port_namespaced():
    scope = {
        "type": "http",
        "scheme": "https",
        "client": ("::1", 12345),
        "headers": [
            (b"host", b"[::1]:43127"),
            (b"origin", b"https://[::1]:43127"),
        ],
    }
    assert local_browser_approval_eligible(scope)
    assert local_browser_approval_eligible(scope, require_same_origin=True)
    runtime = type("Runtime", (), {"installation_id": "install-id"})()
    first = local_approval_cookie_name(runtime, scope)
    second_scope = dict(scope)
    second_scope["headers"] = [(b"host", b"[::1]:43128"), (b"origin", b"https://[::1]:43128")]
    second = local_approval_cookie_name(runtime, second_scope)
    assert first != second


def test_duplicate_headers_and_malformed_or_spoofed_hosts_fail_closed():
    base = {
        "type": "http",
        "scheme": "http",
        "client": ("127.0.0.1", 12345),
    }
    unsafe_headers = [
        [(b"host", b"127.0.0.1:8000"), (b"host", b"localhost:8000")],
        [(b"host", b"localhost:8000"), (b"origin", b"http://localhost:8000"), (b"origin", b"http://localhost:8000")],
        [(b"host", b"localhost/path")],
        [(b"host", b"localhost:bad-port")],
        [(b"host", b"user@localhost:8000")],
        [(b"host", b"localhost:99999")],
        [(b"host", b"127.0.0.2:8000")],
        [(b"host", b"127.0.0.1:8000"), (b"authorization", b"")],
    ]
    for headers in unsafe_headers:
        assert not local_browser_approval_eligible({**base, "headers": headers})


def test_local_challenge_is_origin_cookie_bound_expiring_and_single_use(tmp_path):
    clock = Clock()
    app, runtime = _app(tmp_path, clock=clock)
    origin = f"http://localhost:{_free_port()}"
    with TestClient(app, base_url=origin, client=("127.0.0.1", 12345)) as browser:
        nonce, challenge_cookie = _challenge(browser, origin)
        cookie_pair = challenge_cookie.split(";", 1)[0]
        cookie_name, cookie_value = cookie_pair.split("=", 1)

        # The nonce alone is insufficient; a different browser has no bound cookie.
        with TestClient(app, base_url=origin, client=("127.0.0.1", 12346)) as other_browser:
            mismatch = _approve(other_browser, origin, nonce)
            assert mismatch.status_code == 401
            wrong_cookie = other_browser.post(
                "/api/v1/agent-access/local-approve",
                json={"nonce": nonce, "approve": True},
                headers={**_local_headers(origin), "Cookie": f"{cookie_name}=wrong"},
            )
            assert wrong_cookie.status_code == 401
            assert runtime.auth.status()["owner_configured"] is False

        approval = _approve(browser, origin, nonce)
        assert approval.status_code == 200
        replay = _approve(browser, origin, nonce)
        assert replay.status_code == 401

        expired_nonce, _cookie = _challenge(browser, origin)
        clock.advance(121)
        expired = _approve(browser, origin, expired_nonce)
        assert expired.status_code == 401


def test_challenge_rejects_cookie_replayed_from_a_different_loopback_hostname(tmp_path):
    app, runtime = _app(tmp_path)
    port = _free_port()
    origin = f"http://localhost:{port}"
    other_origin = f"http://127.0.0.1:{port}"
    with TestClient(app, base_url=origin, client=("127.0.0.1", 12345)) as browser:
        nonce, challenge_cookie = _challenge(browser, origin)
        cookie_pair = challenge_cookie.split(";", 1)[0]
    with TestClient(app, base_url=other_origin, client=("127.0.0.1", 12346)) as other_host:
        mismatched_origin = other_host.post(
            "/api/v1/agent-access/local-approve",
            json={"nonce": nonce, "approve": True},
            headers={**_local_headers(other_origin), "Cookie": cookie_pair},
        )
        assert mismatched_origin.status_code == 401
    assert runtime.auth.status()["owner_configured"] is False


def test_local_approval_is_port_namespaced_and_origin_bound(tmp_path):
    app, runtime = _app(tmp_path)
    first_origin = f"http://127.0.0.1:{_free_port()}"
    second_origin = f"http://127.0.0.1:{_free_port()}"
    assert first_origin != second_origin
    with TestClient(app, base_url=first_origin, client=("127.0.0.1", 12345)) as first:
        nonce, first_cookie = _challenge(first, first_origin)
        first_name = local_approval_cookie_name(runtime, {
            "type": "http", "scheme": "http", "client": ("127.0.0.1", 12345),
            "headers": [(b"host", first_origin.removeprefix("http://").encode())],
        })
        with TestClient(app, base_url=second_origin, client=("127.0.0.1", 12346)) as second:
            second_name = local_approval_cookie_name(runtime, {
                "type": "http", "scheme": "http", "client": ("127.0.0.1", 12346),
                "headers": [(b"host", second_origin.removeprefix("http://").encode())],
            })
            assert first_name != second_name
            assert first_name in first_cookie
            assert second_name not in first_cookie
            crossed = _approve(second, second_origin, nonce)
            assert crossed.status_code == 401
        assert runtime.auth.status()["owner_configured"] is False


def test_remote_proxy_forwarded_host_origin_and_bearer_requests_are_not_eligible(tmp_path):
    app, runtime = _app(tmp_path)
    remote_cases = [
        ("http://192.168.1.20:54321", ("192.168.1.20", 2345), {}),
        ("https://maestro.tailnet.example:54321", ("127.0.0.1", 2345), {}),
        ("http://127.0.0.1:54321", ("127.0.0.1", 2345), {"Forwarded": "for=127.0.0.1;proto=http"}),
        ("http://127.0.0.1:54321", ("127.0.0.1", 2345), {"X-Forwarded-For": "127.0.0.1"}),
        ("http://127.0.0.1:54321", ("127.0.0.1", 2345), {"X-Real-IP": "127.0.0.1"}),
    ]
    for origin, peer, extra_headers in remote_cases:
        with TestClient(app, base_url=origin, client=peer) as client:
            status = client.get("/api/v1/agent-access/status", headers=extra_headers).json()
            assert status["local_approval_available"] is False
            response = client.post(
                "/api/v1/agent-access/local-challenge",
                headers={"Origin": origin, **extra_headers},
            )
            assert response.status_code == 403

    local_origin = f"http://127.0.0.1:{_free_port()}"
    with TestClient(app, base_url=local_origin, client=("127.0.0.1", 12345)) as browser:
        for headers in (
            {"Origin": "http://attacker.example"},
            {"Origin": "null"},
            {"Origin": local_origin, "Sec-Fetch-Site": "cross-site"},
            {"Origin": local_origin, "Authorization": "Bearer an-agent-token"},
            {"Origin": local_origin, "Host": "example.com"},
        ):
            response = browser.post("/api/v1/agent-access/local-challenge", headers=headers)
            assert response.status_code == 403
        assert browser.get("/api/v1/agent-access/status", headers={"Authorization": "Bearer an-agent-token"}).json()["local_approval_available"] is False
        assert runtime.auth.status()["owner_configured"] is False


def test_originless_same_origin_fetch_metadata_is_supported_but_get_never_enrolls(tmp_path):
    app, runtime = _app(tmp_path)
    origin = f"http://localhost:{_free_port()}"
    with TestClient(app, base_url=origin, client=("127.0.0.1", 12345)) as browser:
        get_result = browser.get("/api/v1/agent-access/local-challenge")
        assert get_result.status_code == 405
        assert runtime.auth.status()["owner_configured"] is False
        response = browser.post(
            "/api/v1/agent-access/local-challenge",
            headers={"Sec-Fetch-Site": "same-origin"},
        )
        assert response.status_code == 200, response.text
        assert runtime.auth.status()["owner_configured"] is False


def test_local_approval_challenges_are_rate_limited_and_remote_login_hides_button(tmp_path):
    app, runtime = _app(tmp_path)
    origin = f"http://127.0.0.1:{_free_port()}"
    with TestClient(app, base_url=origin, client=("127.0.0.1", 12345)) as browser:
        for _ in range(10):
            assert browser.post(
                "/api/v1/agent-access/local-challenge",
                headers=_local_headers(origin),
            ).status_code == 200
        limited = browser.post("/api/v1/agent-access/local-challenge", headers=_local_headers(origin))
        assert limited.status_code == 429
        assert runtime.auth.status()["owner_configured"] is False

    remote_origin = f"https://192.168.1.22:{_free_port()}"
    with TestClient(app, base_url=remote_origin, client=("192.168.1.22", 2345)) as remote:
        page = remote.get("/agent-access/login")
        assert page.status_code == 200
        assert "Approve on this PC" not in page.text
        assert 'name="code"' in page.text
        assert "autofocus" in page.text


def _agent_token(runtime: RuntimeState) -> str:
    owner = runtime.auth.issue_owner_session()["principal"]
    runtime.auth.set_enabled(True, owner)
    client = runtime.auth.register_client({
        "client_name": "Local route restriction test",
        "redirect_uris": ["http://127.0.0.1:49173/callback"],
    })
    verifier = "a" * 43
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    redirect = client["redirect_uris"][0]
    pending = runtime.auth.create_authorization(
        client["client_id"], redirect, "observe", challenge, runtime.mcp_url, "local-route-test",
    )
    authorization = runtime.auth.approve_authorization(pending["request_id"], owner, "observe")
    return runtime.auth.exchange_code(
        client["client_id"], authorization["code"], redirect, verifier, runtime.mcp_url,
    )["access_token"]


def test_authenticated_bearer_agent_cannot_use_local_approval_routes(tmp_path):
    app, runtime = _app(tmp_path)
    token = _agent_token(runtime)
    origin = f"http://127.0.0.1:{_free_port()}"
    with TestClient(app, base_url=origin, client=("127.0.0.1", 12345)) as agent:
        headers = {"Authorization": f"Bearer {token}", "Origin": origin}
        status = agent.get("/api/v1/agent-access/status", headers=headers).json()
        assert status["authenticated"] is True
        assert status["owner"] is False
        assert status["local_approval_available"] is False
        assert agent.post("/api/v1/agent-access/local-challenge", headers=headers).status_code == 403
        assert agent.post(
            "/api/v1/agent-access/local-approve",
            json={"nonce": "not-a-challenge", "approve": True},
            headers=headers,
        ).status_code == 403


def test_local_approval_preserves_existing_owner_grants_enabled_and_runtime_settings(tmp_path):
    app, runtime = _app(tmp_path)
    owner = runtime.auth.issue_owner_session()["principal"]
    runtime.auth.set_enabled(True, owner)
    client = runtime.auth.register_client({
        "client_name": "Preserved grant",
        "redirect_uris": ["http://127.0.0.1:49174/callback"],
    })
    verifier = "b" * 43
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    redirect = client["redirect_uris"][0]
    pending = runtime.auth.create_authorization(
        client["client_id"], redirect, "observe", challenge, runtime.mcp_url, None,
    )
    approval = runtime.auth.approve_authorization(pending["request_id"], owner, "observe")
    runtime.auth.exchange_code(client["client_id"], approval["code"], redirect, verifier, runtime.mcp_url)
    before = runtime.auth.status()
    config_before = runtime.config

    origin = f"http://127.0.0.1:{_free_port()}"
    with TestClient(app, base_url=origin, client=("127.0.0.1", 12345)) as browser:
        nonce, _cookie = _challenge(browser, origin)
        response = _approve(browser, origin, nonce)
        assert response.status_code == 200

    after = runtime.auth.status()
    assert after["owner_configured"] is True
    assert after["enabled"] is before["enabled"] is True
    assert after["grants"] == before["grants"]
    assert runtime.config == config_before
