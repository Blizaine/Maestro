"""Scoped local stdio client coverage without starting Maestro or models."""
from __future__ import annotations

import hashlib
import importlib.util
import asyncio
import json
import os
import sqlite3
import sys
import unittest
from pathlib import Path

if importlib.util.find_spec("mcp") is None:
    raise unittest.SkipTest("Local MCP tests run in the separate Agent Access environment")

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from services.agent_access.auth import AuthError, AuthStore, Principal
from services.agent_access.boundary import AccessBoundary, cookie_name
from services.agent_access.bridge import BackendClient
from services.agent_access.context import principal_context
from services.agent_access.controller import create_app
from services.agent_access.local_connector import (
    LocalCredentialError,
    create_local_mcp_server,
    local_launch_configuration,
    read_local_credential,
    run_local_stdio,
    write_local_credential,
)
from services.agent_access.private_permissions import PrivatePermissionError, protect_private_path
from services.agent_access.state import RuntimeState
import services.agent_access.local_connector as local_connector


RESOURCE = "https://maestro.example/mcp"
OWNER = Principal("owner", "owner-console", ("observe", "generate", "manage", "admin"), True)


def _owner(store: AuthStore) -> tuple[dict, Principal]:
    session = store.issue_owner_session()
    return session, session["principal"]


def _challenge(verifier: str) -> str:
    import base64

    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode("ascii").rstrip("=")


def _remote_token(store: AuthStore, owner: Principal, scopes=("observe",), resource=RESOURCE) -> str:
    verifier = "local-transport-integration-verifier-123456789012345"
    client = store.register_client({
        "client_name": "Remote integration client",
        "redirect_uris": ["http://127.0.0.1:49173/callback"],
    })
    pending = store.create_authorization(
        client["client_id"], client["redirect_uris"][0], scopes,
        _challenge(verifier), resource, "test-state",
    )
    approved = store.approve_authorization(pending["request_id"], owner, scopes)
    exchanged = store.exchange_code(
        client["client_id"], approved["code"], client["redirect_uris"][0], verifier, resource,
    )
    return exchanged["access_token"]


def _settings_runtime(tmp_path: Path) -> RuntimeState:
    app_root = tmp_path / "app"
    app_root.mkdir()
    return RuntimeState(
        tmp_path / "settings" / "agent-access",
        app_root=app_root,
    )


def test_enablement_matrix_migration_and_transport_specific_auth(tmp_path):
    store = AuthStore(tmp_path / "agent-access", resource=RESOURCE)
    owner_session, owner = _owner(store)
    local, local_token = store.create_local_client("Local test", ["observe"], owner)
    store.set_enabled(True, owner)
    remote_token = _remote_token(store, owner)

    for remote_enabled in (False, True):
        for local_enabled in (False, True):
            store.set_enabled(remote_enabled, owner)
            store.set_local_enabled(local_enabled, owner)
            status = store.status()
            assert status["enabled"] is remote_enabled
            assert status["remote_enabled"] is remote_enabled
            assert status["local_enabled"] is local_enabled
            if remote_enabled:
                assert store.authenticate(remote_token).owner is False
            else:
                with pytest.raises(AuthError, match="disabled"):
                    store.authenticate(remote_token)
            if local_enabled:
                assert store.authenticate_local(local_token, local["id"]).id == local["grant_id"]
            else:
                with pytest.raises(AuthError, match="disabled"):
                    store.authenticate_local(local_token, local["id"])
            # Browser owner sessions keep their separate authentication path.
            assert store.authenticate(owner_session["access_token"]).owner is True

    store.set_enabled(False, owner)
    store.set_local_enabled(True, owner)
    assert store.authenticate_local(local_token, local["id"]).scopes == ("observe",)
    with pytest.raises(AuthError):
        store.authenticate(local_token)
    with pytest.raises(AuthError):
        store.authenticate_local(remote_token)


def test_legacy_grants_migrate_as_remote_without_losing_access(tmp_path):
    directory = tmp_path / "agent-access"
    directory.mkdir()
    db_path = directory / "agent-access.sqlite3"
    token = "legacy-remote-access-token"
    grant_id = "legacy-grant"
    client_id = "legacy-client"
    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE clients(client_id TEXT PRIMARY KEY, client_name TEXT NOT NULL, redirect_uris TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE grants(grant_id TEXT PRIMARY KEY, client_id TEXT NOT NULL, scopes TEXT NOT NULL, resource TEXT NOT NULL, created_at REAL NOT NULL, revoked_at REAL);
            CREATE TABLE access_tokens(token_hash TEXT PRIMARY KEY, kind TEXT NOT NULL, principal_id TEXT NOT NULL, client_id TEXT NOT NULL, scopes TEXT NOT NULL, expires_at REAL NOT NULL, revoked_at REAL, family_id TEXT);
            INSERT INTO meta(key, value) VALUES('enabled', '1');
            INSERT INTO clients VALUES('legacy-client', 'Legacy remote client', '[]', 1);
            INSERT INTO grants VALUES('legacy-grant', 'legacy-client', '["observe"]', 'https://maestro.example/mcp', 1, NULL);
            """
        )
        db.execute(
            "INSERT INTO access_tokens VALUES(?, 'agent', ?, ?, '[\"observe\"]', 9999999999, NULL, NULL)",
            (hashlib.sha256(token.encode("utf-8")).hexdigest(), grant_id, client_id),
        )

    store = AuthStore(directory, resource=RESOURCE)
    assert store.status()["remote_enabled"] is True
    assert store.status()["local_enabled"] is False
    assert store.authenticate(token).id == grant_id
    assert store.list_grants(OWNER)[0]["transport"] == "remote"


def test_local_credentials_are_private_scoped_and_never_in_launch_config(tmp_path):
    directory = tmp_path / "agent-access"
    store = AuthStore(directory, resource=RESOURCE)
    _session, owner = _owner(store)
    store.set_local_enabled(True, owner)
    record, token = store.create_local_client("Studio helper", ["observe", "generate"], owner)
    path = write_local_credential(directory, record["id"], token)

    assert read_local_credential(directory, record["id"]) == token
    assert token not in json.dumps(record)
    assert token not in json.dumps(store.list_local_clients(owner))
    assert token not in json.dumps(store.list_grants(owner))
    with sqlite3.connect(store.db_path) as db:
        stored = db.execute("SELECT token_hash FROM local_credentials").fetchone()[0]
    assert stored == hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert token.encode("ascii") not in store.db_path.read_bytes()
    assert "local_enabled" in store.status()
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700

    config = local_launch_configuration(tmp_path / "app", record["id"])
    assert token not in json.dumps(config)
    assert config["args"][-2:] == ["--client", record["id"]]
    interpreter_parts = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    assert Path(config["command"]) == tmp_path / "app" / "env-agent" / Path(*interpreter_parts)
    with pytest.raises(LocalCredentialError):
        read_local_credential(directory, "../outside")
    with pytest.raises(AuthError, match="Owner"):
        store.create_local_client("Not owner", ["observe"], Principal("agent", "client", ("observe",), False))


def test_mocked_windows_private_acl_success_and_failure(tmp_path):
    class RecordingSecurityAPI:
        def __init__(self, failure=None):
            self.calls = []
            self.failure = failure

        def protect(self, path, *, directory):
            self.calls.append((path, directory))
            if self.failure:
                raise self.failure

    path = tmp_path / "local-clients"
    successful = RecordingSecurityAPI()
    protect_private_path(path, directory=True, windows=True, security_api=successful)
    assert successful.calls == [(path, True)]

    failing = RecordingSecurityAPI(PrivatePermissionError("simulated ACL failure"))
    with pytest.raises(PrivatePermissionError, match="simulated ACL failure"):
        protect_private_path(path, directory=True, windows=True, security_api=failing)
    assert failing.calls == [(path, True)]


def test_credential_creation_fails_closed_when_windows_acl_cannot_be_set(tmp_path, monkeypatch):
    directory = tmp_path / "agent-access"
    directory.mkdir()
    client_id = "b4f6e560-1234-4a56-8abc-0123456789ab"
    calls = []

    def deny_acl(path, *, directory):
        calls.append((path, directory))
        raise PrivatePermissionError("simulated Windows ACL failure")

    monkeypatch.setattr(local_connector, "protect_private_path", deny_acl)
    with pytest.raises(LocalCredentialError, match="Could not protect"):
        write_local_credential(directory, client_id, "A" * 64)
    assert calls == [(directory / "local-clients", True)]
    assert not (directory / "local-clients").exists()


def test_credential_creation_protects_only_client_directory_and_credential(tmp_path, monkeypatch):
    directory = tmp_path / "agent-access"
    directory.mkdir()
    client_id = "b4f6e560-1234-4a56-8abc-0123456789ab"
    calls = []
    monkeypatch.setattr(
        local_connector,
        "protect_private_path",
        lambda path, *, directory: calls.append((path, directory)),
    )
    path = write_local_credential(directory, client_id, "B" * 64)
    assert calls == [(path.parent, True), (path, False)]
    assert read_local_credential(directory, client_id) == "B" * 64


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL integration requires Windows")
def test_windows_acl_integration_uses_only_temporary_credential_paths(tmp_path):
    settings_directory = tmp_path / "agent-access"
    settings_directory.mkdir()
    credential = "C" * 64
    client_id = "b4f6e560-1234-4a56-8abc-0123456789ab"
    path = write_local_credential(settings_directory, client_id, credential)
    assert read_local_credential(settings_directory, client_id) == credential
    # Reapplication also performs a read-back check for owner, protected DACL,
    # and the exact current-user + SYSTEM ACE set.
    protect_private_path(path.parent, directory=True)
    protect_private_path(path, directory=False)


def test_local_mcp_rechecks_toggle_revocation_and_scope_for_each_call(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    store = AuthStore(tmp_path / "agent-access", resource=RESOURCE)
    _session, owner = _owner(store)
    store.set_local_enabled(True, owner)
    record, credential = store.create_local_client("Read-only local", ["observe"], owner)

    class FakeBackend:
        async def request(self, method, path, **kwargs):
            principal = principal_context.get()
            assert principal is not None and not principal.owner
            return {"principal_id": principal.id}

    server = create_local_mcp_server(FakeBackend(), tmp_path, "test", store, record["id"], credential)

    async def call_status():
        return await server.call_tool("get_status", {})

    assert asyncio.run(call_status()).is_error is False
    store.set_local_enabled(False, owner)
    with pytest.raises(ToolError, match="disabled"):
        asyncio.run(call_status())
    store.set_local_enabled(True, owner)
    assert store.revoke_grant(record["grant_id"], owner)
    with pytest.raises(ToolError, match="revoked"):
        asyncio.run(call_status())


def test_local_mcp_authoring_tools_require_their_declared_scopes():
    from services.agent_access.local_connector import _tool_scope

    for name in ("get_creation_defaults", "get_creation_context", "get_authored_draft"):
        assert _tool_scope(name, {}) == "observe"
    for name in ("prepare_authored_generation", "submit_authored_generation"):
        assert _tool_scope(name, {}) == "generate"


def test_access_boundary_accepts_local_token_only_for_native_loopback_api(tmp_path):
    runtime = _settings_runtime(tmp_path)
    owner_session, owner = _owner(runtime.auth)
    runtime.mark_security_configured()
    runtime.auth.set_enabled(True, owner)
    runtime.auth.set_local_enabled(True, owner)
    local, local_token = runtime.auth.create_local_client("Native stdio", ["observe"], owner)
    remote_token = _remote_token(runtime.auth, owner, resource=runtime.mcp_url)

    async def status(_request: Request):
        principal = principal_context.get()
        return JSONResponse({"principal_id": principal.id if principal else None, "owner": bool(principal and principal.owner)})

    async def prepare(_request: Request):
        return JSONResponse({"ok": True})

    inner = Starlette(routes=[
        Route("/api/v1/system-stats", status, methods=["GET"]),
        Route("/api/v1/agent-access/prepare", prepare, methods=["POST"]),
        Route("/mcp", status, methods=["GET", "POST", "DELETE"]),
    ])
    native = AccessBoundary(inner, runtime=runtime, always_protect=True, allow_local_agent_tokens=True)
    native_client = TestClient(
        native,
        base_url=f"http://127.0.0.1:{runtime.port}",
        client=("127.0.0.1", 41111),
    )
    local_headers = {"Authorization": f"Bearer {local_token}"}
    assert native_client.get("/api/v1/system-stats", headers=local_headers).json()["principal_id"] == local["grant_id"]
    assert native_client.post("/api/v1/agent-access/prepare", headers=local_headers).status_code == 403

    controller = AccessBoundary(inner, runtime=runtime, always_protect=True)
    controller_client = TestClient(
        controller,
        base_url=f"https://127.0.0.1:{runtime.port}",
        client=("127.0.0.1", 41112),
    )
    assert controller_client.get("/api/v1/system-stats", headers=local_headers).status_code == 401
    assert controller_client.get("/mcp", headers=local_headers).status_code == 401
    assert controller_client.get("/api/v1/system-stats", headers={"Authorization": f"Bearer {remote_token}"}).status_code == 200

    runtime.auth.set_enabled(False, owner)
    assert native_client.get("/api/v1/system-stats", headers=local_headers).status_code == 200
    assert native_client.get("/api/v1/system-stats", headers={"Authorization": f"Bearer {remote_token}"}).status_code == 401
    assert controller_client.get("/mcp", headers={"Authorization": f"Bearer {remote_token}"}).status_code == 404
    controller_client.cookies.set(cookie_name(runtime, "https"), owner_session["access_token"])
    owner_status = controller_client.get("/api/v1/system-stats")
    assert owner_status.status_code == 200
    assert owner_status.json()["owner"] is True


def test_owner_http_settings_local_client_lifecycle_and_safe_response(tmp_path):
    runtime = _settings_runtime(tmp_path)
    owner_session = runtime.auth.issue_owner_session()
    owner_token = owner_session["access_token"]
    runtime.register_backend(
        f"http://127.0.0.1:{runtime.port + 1}",
        ready=False,
        session_id="local-client-http-test",
    )
    bridge = BackendClient(
        runtime,
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))),
    )
    app = create_app(runtime, bridge=bridge, frame_ancestor_origins=())
    with TestClient(app, base_url=runtime.origin, client=("127.0.0.1", 41113)) as client:
        client.cookies.set(cookie_name(runtime, "https"), owner_token)
        origin = {"Origin": runtime.origin}
        enabled = client.put("/api/v1/agent-access/settings", json={"local_enabled": True}, headers=origin)
        assert enabled.status_code == 200, enabled.text
        assert enabled.json()["local_enabled"] is True
        assert enabled.json()["remote_enabled"] is False
        assert enabled.json()["enabled"] is False

        created = client.post(
            "/api/v1/agent-access/local-clients",
            json={"name": "Editor", "scopes": ["observe", "manage"]},
            headers=origin,
        )
        assert created.status_code == 200, created.text
        record = created.json()
        assert set(record["scopes"]) == {"observe", "manage"}
        assert "credential" not in record and "token" not in record and "secret" not in record
        saved_token = read_local_credential(runtime.directory, record["id"])
        assert saved_token not in created.text
        assert saved_token not in client.get("/api/v1/agent-access/local-clients").text
        assert record["args"][-2:] == ["--client", record["id"]]

        extension = client.get(
            f"/api/v1/agent-access/local-clients/{record['id']}/extension",
        )
        assert extension.status_code == 200, extension.text
        assert extension.headers["content-type"] == "application/octet-stream"
        assert extension.headers["content-disposition"] == f'attachment; filename="maestro-local-{record["id"]}.mcpb"'
        import io
        from zipfile import ZipFile
        with ZipFile(io.BytesIO(extension.content)) as bundle:
            launch = json.loads(bundle.read("server/launch-config.json"))
            assert set(launch) == {"command", "args"}
            assert launch["args"][-2:] == ["--client", record["id"]]
            assert saved_token.encode("ascii") not in extension.content

        grants = client.get("/api/v1/agent-access/grants").json()["grants"]
        grant = next(item for item in grants if item["grant_id"] == record["grant_id"])
        assert grant["transport"] == "local"
        revoked = client.delete(f"/api/v1/agent-access/grants/{record['grant_id']}", headers=origin)
        assert revoked.status_code == 200 and revoked.json()["revoked"] is True
        with pytest.raises(LocalCredentialError):
            read_local_credential(runtime.directory, record["id"])
        assert client.get(
            f"/api/v1/agent-access/local-clients/{record['id']}/extension",
        ).status_code == 404
        disabled = client.put("/api/v1/agent-access/settings", json={"local_enabled": False}, headers=origin)
        assert disabled.status_code == 200
        assert client.get(
            f"/api/v1/agent-access/local-clients/{record['id']}/extension",
        ).status_code == 409
        assert runtime.auth.authenticate(owner_token).owner is True


def test_stdio_subprocess_uses_native_mcp_sdk_with_fake_backend_and_no_stdout_noise(tmp_path):
    settings = tmp_path / "settings" / "agent-access"
    store = AuthStore(settings, resource=RESOURCE)
    _owner_session, owner = _owner(store)
    store.set_local_enabled(True, owner)
    record, token = store.create_local_client("stdio smoke", ["observe"], owner)
    write_local_credential(settings, record["id"], token)

    source = r'''
import sys
from pathlib import Path
from services.agent_access.auth import AuthStore
from services.agent_access.local_connector import create_local_mcp_server, read_local_credential
from services.agent_access.context import principal_context

class FakeBackend:
    async def request(self, method, path, **kwargs):
        principal = principal_context.get()
        assert principal is not None and not principal.owner
        return {"generation_available": False, "principal_id": principal.id}

settings, client_id = Path(sys.argv[1]), sys.argv[2]
auth = AuthStore(settings, resource="https://maestro.example/mcp")
credential = read_local_credential(settings, client_id)
server = create_local_mcp_server(FakeBackend(), Path(sys.argv[3]), "test", auth, client_id, credential)
server.run("stdio")
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_APP) + os.pathsep + env.get("PYTHONPATH", "")
    error_path = tmp_path / "stdio-child.stderr"

    async def exercise_stdio():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-c", source, str(settings), record["id"], str(_APP)],
            env=env,
            cwd=str(tmp_path),
        )
        with error_path.open("w+", encoding="utf-8") as error_log:
            async with stdio_client(parameters, errlog=error_log) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    initialize = await session.initialize()
                    tools = await session.list_tools()
                    call = await session.call_tool("get_status", {})
                    error_log.flush()
                    error_log.seek(0)
                    child_stderr = error_log.read()
                    return initialize, tools, call, child_stderr

    initialize, tools, call, child_stderr = asyncio.run(asyncio.wait_for(exercise_stdio(), timeout=30))
    assert initialize.server_info.name == "Maestro Agent Access"
    assert any(tool.name == "get_status" for tool in tools.tools)
    assert call.is_error is not True
    assert "generation_available" in call.content[0].text
    # The MCP SDK parser would reject any non-protocol bytes on child stdout.
    assert token not in child_stderr


def test_run_local_stdio_wires_lifecycle_for_owned_backend_and_close_never_stops_engine(tmp_path, monkeypatch):
    runtime = _settings_runtime(tmp_path)
    _session, owner = _owner(runtime.auth)
    runtime.auth.set_local_enabled(True, owner)
    record, token = runtime.auth.create_local_client("lifecycle helper", ["observe", "admin"], owner)
    write_local_credential(runtime.directory, record["id"], token)

    class FakeBackend:
        def __init__(self):
            self.lifecycle_manager = None
            self.closed = False

        async def idle_probe(self):
            return {"idle": True, "certain": True}

        async def close(self):
            self.closed = True

    class FakeLifecycleManager:
        def __init__(self, runtime_arg, *, idle_probe):
            self.runtime = runtime_arg
            self.idle_probe = idle_probe
            self.actions = []

        async def execute(self, action, principal_id, request_id):
            self.actions.append(action)

    class FakeServer:
        def __init__(self, selected_backend):
            self.backend = selected_backend

        def run(self, transport):
            assert transport == "stdio"
            assert self.backend.lifecycle_manager is not None

    backend = FakeBackend()
    managers = []

    def build_manager(*args, **kwargs):
        manager = FakeLifecycleManager(*args, **kwargs)
        managers.append(manager)
        return manager

    import services.agent_access.bridge as bridge_module
    import services.agent_access.lifecycle as lifecycle_module

    monkeypatch.setattr(bridge_module, "BackendClient", lambda _runtime: backend)
    monkeypatch.setattr(lifecycle_module, "LifecycleManager", build_manager)
    monkeypatch.setattr(
        local_connector,
        "create_local_mcp_server",
        lambda selected_backend, *_args, **_kwargs: FakeServer(selected_backend),
    )

    run_local_stdio(runtime, record["id"], version="test")
    assert len(managers) == 1
    assert managers[0].runtime is runtime
    assert managers[0].idle_probe == backend.idle_probe
    assert managers[0].actions == []
    assert backend.closed is True

    # Explicitly supplied fakes are kept as supplied and do not gain a real
    # Pinokio lifecycle manager as a side effect.
    injected = FakeBackend()
    sentinel = object()
    injected.lifecycle_manager = sentinel
    run_local_stdio(runtime, record["id"], backend=injected, version="test")
    assert injected.lifecycle_manager is sentinel
    assert injected.closed is True
    assert managers[0].actions == []
