from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.auth import AuthError, AuthStore, Principal


RESOURCE = "http://127.0.0.1:9000/mcp"
REDIRECT = "https://agent.example/callback"
VERIFIER = "a" * 43


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


def challenge(verifier: str = VERIFIER) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def enabled_store(tmp_path, clock: Clock | None = None) -> tuple[AuthStore, Principal]:
    store = AuthStore(tmp_path / "settings", clock=clock or Clock(), resource=RESOURCE)
    owner_session = store.redeem_owner_code(store.issue_owner_code())
    owner = owner_session["principal"]
    store.set_enabled(True, owner)
    return store, owner


def client(store: AuthStore, redirect_uri: str = REDIRECT) -> dict:
    return store.register_client({"client_name": "Example Agent", "redirect_uris": [redirect_uri]})


def begin_auth(
    store: AuthStore,
    client_id: str,
    *,
    redirect_uri: str = REDIRECT,
    scope: str = "observe generate",
    resource: str = RESOURCE,
    state: str | None = "state-123",
) -> dict:
    return store.create_authorization(
        client_id,
        redirect_uri,
        scope,
        challenge(),
        resource,
        state,
    )


def approve_and_exchange(
    store: AuthStore,
    owner: Principal,
    client_id: str,
    pending: dict,
    *,
    approved_scopes: str = "observe generate",
    verifier: str = VERIFIER,
) -> dict:
    result = store.approve_authorization(pending["request_id"], owner, approved_scopes)
    return store.exchange_code(client_id, result["code"], result["redirect_uri"], verifier, RESOURCE)


def expect_auth_error(status: int, operation) -> AuthError:
    with pytest.raises(AuthError) as caught:
        operation()
    assert caught.value.status == status
    return caught.value


def test_dcr_redirect_policy_and_loopback_variable_ports(tmp_path):
    store, _owner = enabled_store(tmp_path)

    for redirect_uri in (
        "https://user:pass@agent.example/callback",
        "https://agent.example/callback#fragment",
        "https://*.example/callback",
        "http://192.0.2.10/callback",
        "http://127.0.0.1:0/callback",
        "custom-agent://callback",
    ):
        expect_auth_error(
            400,
            lambda redirect_uri=redirect_uri: store.register_client({"redirect_uris": [redirect_uri]}),
        )

    loopback = store.register_client({"redirect_uris": ["http://127.0.0.1/callback"]})
    pending = begin_auth(
        store,
        loopback["client_id"],
        redirect_uri="http://127.0.0.1:43127/callback",
        scope="observe",
    )
    assert pending["redirect_uri"] == "http://127.0.0.1:43127/callback"

    expect_auth_error(
        400,
        lambda: begin_auth(store, loopback["client_id"], redirect_uri="http://127.0.0.1:43127/other", scope="observe"),
    )


def test_pkce_client_redirect_and_resource_are_bound_to_authorization_code(tmp_path):
    store, owner = enabled_store(tmp_path)
    registered = client(store)
    pending = begin_auth(store, registered["client_id"])
    approval = store.approve_authorization(pending["request_id"], owner, "observe generate")

    expect_auth_error(
        401,
        lambda: store.exchange_code("other-client", approval["code"], REDIRECT, VERIFIER, RESOURCE),
    )
    expect_auth_error(
        400,
        lambda: store.exchange_code(registered["client_id"], approval["code"], "https://other.example/callback", VERIFIER, RESOURCE),
    )
    expect_auth_error(
        400,
        lambda: store.exchange_code(registered["client_id"], approval["code"], REDIRECT, "b" * 43, RESOURCE),
    )
    expect_auth_error(
        400,
        lambda: store.exchange_code(registered["client_id"], approval["code"], REDIRECT, VERIFIER, "http://127.0.0.1:9001/mcp"),
    )

    tokens = store.exchange_code(registered["client_id"], approval["code"], REDIRECT, VERIFIER, RESOURCE)
    principal = store.authenticate(tokens["access_token"])
    assert principal.client_id == registered["client_id"]
    assert principal.owner is False
    assert principal.scopes == ("observe", "generate")
    expect_auth_error(
        400,
        lambda: store.exchange_code(registered["client_id"], approval["code"], REDIRECT, VERIFIER, RESOURCE),
    )


def test_owner_only_operations_scope_escalation_and_full_expansion(tmp_path):
    store, owner = enabled_store(tmp_path)
    registered = client(store)

    limited = begin_auth(store, registered["client_id"], scope="observe")
    expect_auth_error(
        403,
        lambda: store.approve_authorization(limited["request_id"], owner, "manage"),
    )
    token_set = approve_and_exchange(store, owner, registered["client_id"], limited, approved_scopes="observe")
    agent = store.authenticate(token_set["access_token"])

    expect_auth_error(403, lambda: store.set_enabled(False, agent))
    expect_auth_error(403, lambda: store.list_grants(agent))
    expect_auth_error(403, lambda: store.revoke_grant("missing", agent))
    assert store.enabled is True

    full = begin_auth(store, registered["client_id"], scope="full")
    full_tokens = approve_and_exchange(store, owner, registered["client_id"], full, approved_scopes="full")
    assert full_tokens["scope"] == "observe generate manage admin"
    full_principal = store.authenticate(full_tokens["access_token"])
    assert full_principal.owner is False
    assert full_principal.scopes == ("observe", "generate", "manage", "admin")
    assert len(store.list_grants(owner)) == 2


def test_owner_code_single_use_and_concurrent_redemption(tmp_path):
    store = AuthStore(tmp_path / "settings", clock=Clock(), resource=RESOURCE)
    code = store.issue_owner_code()

    def redeem():
        try:
            return store.redeem_owner_code(code)
        except AuthError:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _index: redeem(), range(12)))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0]["principal"].owner is True
    expect_auth_error(401, lambda: store.redeem_owner_code(code))


def test_owner_code_and_access_token_expiry(tmp_path):
    clock = Clock()
    store = AuthStore(tmp_path / "settings", clock=clock, resource=RESOURCE)
    code = store.issue_owner_code()
    clock.advance(10 * 60)
    expect_auth_error(401, lambda: store.redeem_owner_code(code))

    store, owner = enabled_store(tmp_path / "other", clock)
    client_record = client(store)
    tokens = approve_and_exchange(
        store,
        owner,
        client_record["client_id"],
        begin_auth(store, client_record["client_id"], scope="observe"),
        approved_scopes="observe",
    )
    clock.advance(15 * 60)
    expect_auth_error(401, lambda: store.authenticate(tokens["access_token"]))


def test_authorization_request_expiry_and_deny_are_one_time(tmp_path):
    clock = Clock()
    store, owner = enabled_store(tmp_path, clock)
    registered = client(store)
    pending = begin_auth(store, registered["client_id"], scope="observe")
    clock.advance(10 * 60)
    expect_auth_error(404, lambda: store.get_authorization(pending["request_id"]))
    expect_auth_error(404, lambda: store.approve_authorization(pending["request_id"], owner, "observe"))

    pending = begin_auth(store, registered["client_id"], scope="observe")
    denied = store.deny_authorization(pending["request_id"], owner)
    assert denied == {"redirect_uri": REDIRECT, "state": "state-123", "error": "access_denied"}
    expect_auth_error(404, lambda: store.deny_authorization(pending["request_id"], owner))


def test_refresh_rotation_replay_revokes_the_whole_family(tmp_path):
    clock = Clock()
    store, owner = enabled_store(tmp_path, clock)
    registered = client(store)
    first = approve_and_exchange(
        store,
        owner,
        registered["client_id"],
        begin_auth(store, registered["client_id"], scope="observe"),
        approved_scopes="observe",
    )
    clock.advance(29 * 24 * 60 * 60)
    rotated = store.refresh_token(registered["client_id"], first["refresh_token"], RESOURCE)
    assert rotated["refresh_token"] != first["refresh_token"]
    assert store.authenticate(rotated["access_token"]).scopes == ("observe",)
    # The old digest remains available for replay detection while the rotated
    # token is still valid, even after the old token's original expiry date.
    clock.advance(2 * 24 * 60 * 60)

    expect_auth_error(
        401,
        lambda: store.refresh_token(registered["client_id"], first["refresh_token"], RESOURCE),
    )
    expect_auth_error(401, lambda: store.authenticate(rotated["access_token"]))
    expect_auth_error(
        401,
        lambda: store.refresh_token(registered["client_id"], rotated["refresh_token"], RESOURCE),
    )


def test_restart_persists_installation_clients_grants_and_sessions(tmp_path):
    clock = Clock()
    store, owner = enabled_store(tmp_path, clock)
    owner_session = store.redeem_owner_code(store.issue_owner_code())
    registered = client(store)
    tokens = approve_and_exchange(
        store,
        owner,
        registered["client_id"],
        begin_auth(store, registered["client_id"], scope="observe"),
        approved_scopes="observe",
    )
    before = store.status()

    restarted = AuthStore(tmp_path / "settings", clock=clock, resource=RESOURCE)
    assert restarted.status()["installation_id"] == before["installation_id"]
    assert restarted.status()["grants"] == {"active": 1, "revoked": 0}
    assert restarted.get_client(registered["client_id"])["client_name"] == "Example Agent"
    assert restarted.authenticate(owner_session["access_token"]).owner is True
    assert restarted.authenticate(tokens["access_token"]).client_id == registered["client_id"]


def test_revocation_is_immediate_and_owner_can_disable_access(tmp_path):
    store, owner = enabled_store(tmp_path)
    registered = client(store)
    tokens = approve_and_exchange(
        store,
        owner,
        registered["client_id"],
        begin_auth(store, registered["client_id"], scope="observe"),
        approved_scopes="observe",
    )
    grant = store.list_grants(owner)[0]
    assert store.revoke_grant(grant["grant_id"], owner) is True
    expect_auth_error(401, lambda: store.authenticate(tokens["access_token"]))
    assert store.list_grants(owner)[0]["revoked"] is True

    store.set_enabled(False, owner)
    assert store.enabled is False
    expect_auth_error(403, lambda: client(store))


def test_database_never_contains_raw_bearer_or_one_time_codes(tmp_path):
    store, owner = enabled_store(tmp_path)
    owner_login_code = store.issue_owner_code()
    owner_login = store.redeem_owner_code(owner_login_code)
    registered = client(store)
    pending = begin_auth(store, registered["client_id"], scope="observe")
    approval = store.approve_authorization(pending["request_id"], owner, "observe")
    tokens = store.exchange_code(registered["client_id"], approval["code"], REDIRECT, VERIFIER, RESOURCE)

    database = store.db_path.read_bytes()
    secrets_to_check = [
        owner_login_code,
        owner_login["access_token"],
        approval["code"],
        tokens["access_token"],
        tokens["refresh_token"],
    ]
    for raw_secret in secrets_to_check:
        assert raw_secret.encode("ascii") not in database


def test_resource_configuration_persists_and_audience_binding_is_required(tmp_path):
    store = AuthStore(tmp_path / "settings", clock=Clock())
    owner = store.redeem_owner_code(store.issue_owner_code())["principal"]
    store.set_enabled(True, owner)
    registered = store.register_client({"redirect_uris": [REDIRECT]})
    with pytest.raises(AuthError) as missing_resource:
        begin_auth(store, registered["client_id"])
    assert missing_resource.value.status == 503

    store.set_resource(RESOURCE)
    with pytest.raises(AuthError) as conflict:
        store.set_resource("http://127.0.0.1:9001/mcp")
    assert conflict.value.status == 409
    assert store.status()["owner_configured"] is True
