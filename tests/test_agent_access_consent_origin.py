"""Regression coverage for browser-origin checks on OAuth consent forms."""

from __future__ import annotations

import importlib.util
import unittest
from urllib.parse import parse_qs, urlsplit

if any(importlib.util.find_spec(name) is None for name in ("mcp", "fastapi", "httpx")):
    raise unittest.SkipTest("Agent Access controller integration dependencies are not installed")

import pytest
from fastapi.testclient import TestClient
from services.agent_access.auth import AuthError
from test_agent_access_controller import _app, _login_owner, _pkce, _runtime

_REDIRECT_URI = "http://127.0.0.1:49173/callback"


def _begin_authorization(client: TestClient, runtime) -> tuple[str, str]:
    registration = client.post(
        "/oauth/register",
        json={"client_name": "Consent origin regression", "redirect_uris": [_REDIRECT_URI]},
        headers={"Origin": runtime.origin},
    )
    assert registration.status_code == 201, registration.text
    metadata = registration.json()
    _verifier, challenge = _pkce()
    response = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": metadata["client_id"],
            "redirect_uri": _REDIRECT_URI,
            "scope": "observe",
            "resource": runtime.mcp_url,
            "state": "consent-origin-state",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        headers={"Origin": runtime.origin},
    )
    assert response.status_code == 303, response.text
    consent_url = response.headers["location"]
    assert consent_url.startswith("/api/v1/agent-access/authorizations/")
    request_id = consent_url.rsplit("/", 1)[-1]
    return consent_url, request_id


def _owner_client(tmp_path):
    runtime = _runtime(tmp_path)
    console: list[str] = []
    app, _calls = _app(runtime, console=console)
    client = TestClient(
        app,
        base_url=runtime.origin,
        client=("127.0.0.1", 12345),
        follow_redirects=False,
    )
    return runtime, app, client, console


def _login_and_enable(client: TestClient, app, runtime, console: list[str]) -> None:
    _login_owner(client, app, runtime, console)
    enabled = client.post(
        "/api/v1/agent-access/settings",
        json={"enabled": True},
        headers={"Origin": runtime.origin},
    )
    assert enabled.status_code == 200, enabled.text
    return runtime, client


def test_native_consent_form_preserves_same_origin_and_callback_stays_no_referrer(tmp_path):
    runtime, app, client, console = _owner_client(tmp_path)
    with client:
        _login_and_enable(client, app, runtime, console)
        consent_url, _request_id = _begin_authorization(client, runtime)

        page = client.get(consent_url)
        assert page.status_code == 200, page.text
        assert page.headers["referrer-policy"] == "same-origin"
        assert '<form method="post">' in page.text

        approved = client.post(
            consent_url,
            data={"decision": "approve", "scope": ["observe"]},
            headers={"Origin": runtime.origin},
        )
        assert approved.status_code == 303, approved.text
        assert approved.headers["referrer-policy"] == "no-referrer"
        callback = urlsplit(approved.headers["location"])
        query = parse_qs(callback.query)
        assert callback.geturl().startswith(_REDIRECT_URI)
        assert query["state"] == ["consent-origin-state"]
        assert len(query["code"]) == 1
        assert "error" not in query


def test_same_origin_consent_can_be_denied_without_issuing_a_code(tmp_path):
    runtime, app, client, console = _owner_client(tmp_path)
    with client:
        _login_and_enable(client, app, runtime, console)
        consent_url, request_id = _begin_authorization(client, runtime)
        denied = client.post(
            consent_url,
            data={"decision": "deny"},
            headers={"Origin": runtime.origin},
        )

        assert denied.status_code == 303, denied.text
        assert denied.headers["referrer-policy"] == "no-referrer"
        query = parse_qs(urlsplit(denied.headers["location"]).query)
        assert query["state"] == ["consent-origin-state"]
        assert query["error"] == ["access_denied"]
        assert "code" not in query
        with pytest.raises(AuthError, match="Authorization request not found"):
            runtime.auth.get_authorization(request_id)


@pytest.mark.parametrize(
    "origin_header",
    ["null", "https://attacker.invalid", None],
    ids=["opaque-origin", "untrusted-origin", "missing-origin"],
)
def test_consent_post_rejects_null_missing_and_untrusted_origins_without_approval(
    tmp_path, origin_header
):
    runtime, app, client, console = _owner_client(tmp_path)
    with client:
        _login_and_enable(client, app, runtime, console)
        consent_url, request_id = _begin_authorization(client, runtime)
        headers = {} if origin_header is None else {"Origin": origin_header}
        rejected = client.post(
            consent_url,
            data={"decision": "approve", "scope": ["observe"]},
            headers=headers,
        )

        assert rejected.status_code == 403, rejected.text
        assert "location" not in rejected.headers
        pending = runtime.auth.get_authorization(request_id)
        assert pending["request_id"] == request_id
        assert pending["state"] == "consent-origin-state"
