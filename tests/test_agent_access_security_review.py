"""Focused regressions for Agent Access boundary and client retention policy."""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from types import SimpleNamespace

import pytest

from services.agent_access import auth as auth_module
from services.agent_access.auth import AuthError, AuthStore, Principal
from services.agent_access.boundary import AccessBoundary, login_document, redact, same_origin, validate_media_paths


RESOURCE = "https://maestro.test/mcp"
REDIRECT = "https://agent.example/callback"
VERIFIER = "a" * 43


class Clock:
    def __init__(self, now: float = 1_800_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def challenge() -> str:
    digest = hashlib.sha256(VERIFIER.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def enabled_store(tmp_path, clock: Clock | None = None):
    store = AuthStore(tmp_path / "settings", clock=clock or Clock(), resource=RESOURCE)
    owner_session = store.redeem_owner_code(store.issue_owner_code())
    store.set_enabled(True, owner_session["principal"])
    return store, owner_session


def issue_agent(store: AuthStore, owner: Principal, *, scope: str = "observe"):
    registered = store.register_client({"client_name": "Test client", "redirect_uris": [REDIRECT]})
    pending = store.create_authorization(
        registered["client_id"], REDIRECT, scope, challenge(), RESOURCE, "test-state"
    )
    approval = store.approve_authorization(pending["request_id"], owner, scope)
    tokens = store.exchange_code(
        registered["client_id"], approval["code"], REDIRECT, VERIFIER, RESOURCE
    )
    return registered, tokens


def test_same_origin_compares_scheme_and_effective_port():
    mismatch = {
        "type": "http",
        "scheme": "https",
        "headers": [
            (b"host", b"maestro.test"),
            (b"origin", b"http://maestro.test"),
            (b"sec-fetch-site", b"cross-site"),
        ],
    }
    assert same_origin(mismatch) is False

    default_port = {
        "type": "http",
        "scheme": "https",
        "headers": [(b"host", b"maestro.test:443"), (b"origin", b"https://maestro.test")],
    }
    assert same_origin(default_port) is True

    for malformed_origin in (b"https://@maestro.test", b"https://maestro.test?x=1"):
        malformed = {
            "scheme": "https",
            "headers": [(b"host", b"maestro.test"), (b"origin", malformed_origin)],
        }
        assert same_origin(malformed) is False


def test_redact_handles_camel_case_credentials_without_hiding_set_flags():
    payload = {
        "accessToken": "access-value",
        "refreshToken": "refresh-value",
        "bearerToken": "bearer-value",
        "privateKey": "private-value",
        "apiKey": "api-value",
        "google_api_key_set": True,
        "nested": [{"clientSecret": "client-value"}],
    }

    assert redact(payload) == {
        "accessToken": "[redacted]",
        "refreshToken": "[redacted]",
        "bearerToken": "[redacted]",
        "privateKey": "[redacted]",
        "apiKey": "[redacted]",
        "google_api_key_set": True,
        "nested": [{"clientSecret": "[redacted]"}],
    }


def test_remote_http_login_page_never_accepts_a_browser_code():
    document = login_document(
        {"controller_url": "https://maestro.test:53000"}, secure=False
    )
    assert "https://maestro.test:53000" in document
    assert "<form" not in document
    assert "name=\"code\"" not in document

    secure_document = login_document(
        {"local_setup_url": "https://127.0.0.1:53000"}, secure=True
    )
    assert 'href="https://127.0.0.1:53000"' in secure_document
    assert "+ setup_url +" not in secure_document


def test_controller_can_require_authentication_before_owner_bootstrap():
    class FakeAuth:
        enabled = False

        def authenticate(self, token):
            raise AssertionError("anonymous request should not authenticate")

    class FakeRuntime:
        security_required = False
        auth = FakeAuth()
        installation_id = "test-installation"
        local_url = RESOURCE
        origin = "https://maestro.test"

        def public_status(self):
            return {"local_setup_url": "https://127.0.0.1:53000"}

    called = []

    async def app(scope, receive, send):
        called.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    boundary = AccessBoundary(app, runtime=FakeRuntime(), always_protect=True)
    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "https",
        "path": "/api/v1/models",
        "query_string": b"",
        "headers": [(b"host", b"maestro.test"), (b"accept", b"application/json")],
        "client": ("198.51.100.4", 12345),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(boundary(scope, receive, send))
    assert called == []
    assert messages[0]["status"] == 401

    legacy_boundary = AccessBoundary(app, runtime=FakeRuntime())
    messages.clear()
    asyncio.run(legacy_boundary(scope, receive, send))
    assert called == [True]
    assert messages[0]["status"] == 200
    called.clear()

    scope.update(path="/mcp", method="POST")
    messages.clear()
    asyncio.run(boundary(scope, receive, send))
    assert messages[0]["status"] == 404
    assert b"WWW-Authenticate" not in dict(messages[0]["headers"])

    FakeRuntime.auth.enabled = True
    messages.clear()
    asyncio.run(boundary(scope, receive, send))
    headers = dict(messages[0]["headers"])
    assert messages[0]["status"] == 401
    assert headers[b"www-authenticate"] == (
        b'Bearer resource_metadata="https://maestro.test/.well-known/oauth-protected-resource/mcp"'
    )
    assert b"agent-token" not in b"".join(message.get("body", b"") for message in messages)


def test_media_path_boundary_rejects_external_filesystem_paths(tmp_path):
    allowed = tmp_path / "outputs" / "result.mp4"
    allowed.parent.mkdir()
    allowed.write_bytes(b"x")
    validate_media_paths({"video_path": str(allowed)}, tmp_path)

    outside = tmp_path.parent / "private-output.mp4"
    with pytest.raises(AuthError) as caught:
        validate_media_paths({"video_path": str(outside)}, tmp_path)
    assert caught.value.status == 403


def test_audit_uses_route_template_instead_of_user_filename():
    principal = Principal("grant-1", "client-1", ("observe", "manage"), False)

    class FakeAuth:
        enabled = True

        def authenticate(self, token):
            assert token == "agent-token"
            return principal

    class FakeRuntime:
        security_required = True
        auth = FakeAuth()
        local_url = RESOURCE
        origin = "https://maestro.test"

        def __init__(self):
            self.audit_events = []

        def audit(self, actor, action, status):
            self.audit_events.append((actor, action, status))

    runtime = FakeRuntime()

    async def app(scope, receive, send):
        scope["route"] = SimpleNamespace(path="/api/v1/uploads/{filename}")
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": b'{"ok":true}'})

    boundary = AccessBoundary(app, runtime=runtime)
    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "https",
        "path": "/api/v1/uploads/private-project-title.mp4",
        "query_string": b"",
        "headers": [(b"host", b"maestro.test"), (b"authorization", b"Bearer agent-token")],
        "client": ("198.51.100.4", 12345),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(boundary(scope, receive, send))
    assert len(messages) == 2
    assert runtime.audit_events == [
        ("grant-1", "GET /api/v1/uploads/{filename}", "200")
    ]
    assert "private-project-title.mp4" not in runtime.audit_events[0][1]


def test_asset_resolver_uses_a_copy_and_replays_resolved_json_with_correct_length(tmp_path):
    media = tmp_path / "outputs" / "registered-asset.png"
    media.parent.mkdir()
    media.write_bytes(b"image")
    principal = Principal("grant-asset", "client-asset", ("generate",), False)

    class FakeAuth:
        enabled = True

        def authenticate(self, token):
            assert token == "agent-token"
            return principal

    class FakeRuntime:
        security_required = True
        auth = FakeAuth()
        installation_id = "test-installation"
        local_url = RESOURCE
        origin = "https://maestro.test"

        def audit(self, actor, action, status):
            pass

    class FakeJournal:
        def __init__(self):
            self.original_payload = None

        def reserve(self, principal_id, key, method, path, payload, *, query=""):
            assert principal_id == principal.id
            self.original_payload = payload
            return {"state": "new", "reservation_id": "reservation-1"}

        def complete(self, reservation_id, status, body, headers):
            assert reservation_id == "reservation-1"

    journal = FakeJournal()
    original_payload = {"image": "asset:registered-image-id", "prompt": "keep the original digest"}
    original_body = json.dumps(original_payload).encode("utf-8")
    received = {}

    def resolve(principal_id, payload):
        assert principal_id == principal.id
        assert payload == original_payload
        payload["image"] = str(media)
        return payload

    async def app(scope, receive, send):
        headers = dict(scope["headers"])
        message = await receive()
        received["body"] = message["body"]
        received["content_length"] = headers[b"content-length"]
        received["transfer_encoding"] = headers.get(b"transfer-encoding")
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": b'{"job_id":"job-1"}'})

    boundary = AccessBoundary(
        app,
        runtime=FakeRuntime(),
        journal=journal,
        app_root=tmp_path,
        asset_resolver=resolve,
    )
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "https",
        "path": "/api/v1/generate",
        "query_string": b"",
        "headers": [
            (b"host", b"maestro.test"),
            (b"authorization", b"Bearer agent-token"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(original_body)).encode("ascii")),
            (b"transfer-encoding", b"chunked"),
            (b"idempotency-key", b"submission-1"),
        ],
        "client": ("198.51.100.4", 12345),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": original_body, "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(boundary(scope, receive, send))
    app_payload = json.loads(received["body"])
    assert app_payload["image"] == str(media)
    assert int(received["content_length"]) == len(received["body"])
    assert received["transfer_encoding"] is None
    assert journal.original_payload == original_payload
    assert journal.original_payload["image"] == "asset:registered-image-id"


def test_off_pauses_agent_access_and_reenabling_preserves_unexpired_tokens(tmp_path):
    store, owner_session = enabled_store(tmp_path)
    owner = owner_session["principal"]
    _registered, tokens = issue_agent(store, owner)

    store.set_enabled(False, owner)
    assert store.authenticate(owner_session["access_token"]).owner is True
    with pytest.raises(AuthError) as disabled_access:
        store.authenticate(tokens["access_token"])
    assert disabled_access.value.status == 401
    with pytest.raises(AuthError) as disabled_refresh:
        store.refresh_token(_registered["client_id"], tokens["refresh_token"], RESOURCE)
    assert disabled_refresh.value.status == 403

    store.set_enabled(True, owner)
    assert store.authenticate(tokens["access_token"]).owner is False
    rotated = store.refresh_token(_registered["client_id"], tokens["refresh_token"], RESOURCE)
    assert store.authenticate(rotated["access_token"]).scopes == ("observe",)


def test_inactive_dcr_clients_are_reclaimed_but_pending_clients_are_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_module, "_MAX_CLIENTS", 1)
    store, owner_session = enabled_store(tmp_path)
    unused = store.register_client({"redirect_uris": [REDIRECT]})
    replacement = store.register_client({"redirect_uris": [REDIRECT]})
    assert store.get_client(replacement["client_id"])["client_id"] == replacement["client_id"]
    with pytest.raises(AuthError) as reclaimed:
        store.get_client(unused["client_id"])
    assert reclaimed.value.status == 404

    pending_client = replacement
    store.create_authorization(
        pending_client["client_id"], REDIRECT, "observe", challenge(), RESOURCE, "pending"
    )
    with pytest.raises(AuthError) as full:
        store.register_client({"redirect_uris": [REDIRECT]})
    assert full.value.status == 429

    active_store, active_owner_session = enabled_store(tmp_path / "active")
    active_client, active_tokens = issue_agent(
        active_store, active_owner_session["principal"]
    )
    with pytest.raises(AuthError) as active_full:
        active_store.register_client({"redirect_uris": [REDIRECT]})
    assert active_full.value.status == 429
    assert active_store.get_client(active_client["client_id"])["client_id"] == active_client["client_id"]
    assert active_store.authenticate(active_tokens["access_token"]).owner is False


def test_dcr_registration_rate_limit_expires_with_its_window(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(auth_module, "_MAX_CLIENT_REGISTRATIONS_PER_HOUR", 2)
    store, _owner_session = enabled_store(tmp_path, clock)
    store.register_client({"redirect_uris": [REDIRECT]})
    store.register_client({"redirect_uris": [REDIRECT]})
    with pytest.raises(AuthError) as limited:
        store.register_client({"redirect_uris": [REDIRECT]})
    assert limited.value.status == 429

    clock.advance(60 * 60 + 1)
    allowed = store.register_client({"redirect_uris": [REDIRECT]})
    assert allowed["client_id"].startswith("ma_")


def test_authorization_code_exchange_is_single_use_under_concurrency(tmp_path):
    store, owner_session = enabled_store(tmp_path)
    registered = store.register_client({"redirect_uris": [REDIRECT]})
    pending = store.create_authorization(
        registered["client_id"], REDIRECT, "observe", challenge(), RESOURCE, "state"
    )
    approval = store.approve_authorization(pending["request_id"], owner_session["principal"], "observe")

    def exchange(_):
        try:
            return store.exchange_code(
                registered["client_id"], approval["code"], REDIRECT, VERIFIER, RESOURCE
            )
        except AuthError:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(exchange, range(24)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert store.status()["grants"]["active"] == 1


def test_concurrent_refresh_replay_revokes_the_rotated_family(tmp_path):
    store, owner_session = enabled_store(tmp_path)
    registered, tokens = issue_agent(store, owner_session["principal"])

    def rotate(_):
        try:
            return store.refresh_token(registered["client_id"], tokens["refresh_token"], RESOURCE)
        except AuthError:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(rotate, range(24)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    with pytest.raises(AuthError):
        store.authenticate(winners[0]["access_token"])
    with pytest.raises(AuthError):
        store.refresh_token(registered["client_id"], winners[0]["refresh_token"], RESOURCE)
