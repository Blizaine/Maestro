"""Tailscale origin/OAuth gates and real TLS/MCP transport, without live state."""
import asyncio
import os
import socket
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from services.agent_access.auth import Principal
from services.agent_access.controller import _configure_tailscale_sni
from services.agent_access.tls import ensure_tls
from test_agent_access_controller import _agent_token, _app, _pkce, _runtime

HOST = "workstation.example-tail.ts.net"


@pytest.mark.parametrize("hostname", ["evil.ts.net", "*.tail.ts.net", "host.tail.ts.net.evil.com", "https://host.tail.ts.net", "host.tail.ts.net:443", "host..tail.ts.net", "HOST.tail.ts.net", "host.tail.ts.net/other"])
def test_only_exact_owner_provisioned_tailnet_identity_is_accepted(tmp_path, hostname):
    runtime = _runtime(tmp_path)
    with pytest.raises(ValueError):
        runtime.configure_tailscale(enabled=True, hostname=hostname)
    assert runtime.tailscale_hostname is None


def test_approved_alias_is_exact_port_and_https_and_can_be_disabled(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.configure_tailscale(enabled=True, hostname=HOST)
    origin = f"https://{HOST}:{runtime.port}"
    assert runtime.canonical_resource(origin + "/mcp") == runtime.mcp_url
    assert origin in runtime.allowed_origins
    assert f"{HOST}:{runtime.port}" in runtime.allowed_authorities
    for resource in (f"http://{HOST}:{runtime.port}/mcp", f"https://{HOST}/mcp", origin + "/mcp/", origin + "/mcp?token=x", f"https://other.example-tail.ts.net:{runtime.port}/mcp"):
        with pytest.raises(ValueError):
            runtime.canonical_resource(resource)
    scope = {"scheme": "https", "client": ("100.64.0.2", 1000), "headers": [(b"host", f"{HOST}:{runtime.port}".encode())]}
    assert runtime.origin_for_scope(scope) == origin
    runtime.configure_tailscale(enabled=False)
    assert runtime.origin_for_scope(scope) is None
    assert origin not in runtime.allowed_origins
    with pytest.raises(ValueError):
        runtime.canonical_resource(origin + "/mcp")


def test_readiness_requires_current_live_controller_matching_exact_certificate(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    runtime.configure_tailscale(enabled=True, hostname=HOST)
    runtime.auth.set_enabled(True, Principal("owner", "Test owner", (), owner=True))
    evidence = {"status": "ready", "pid": os.getpid(), "tailscale": {"installed": True, "connected": True, "hostname": HOST, "certificate_ready": True, "mcp_url": f"https://{HOST}:{runtime.port}/mcp", "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}}
    assert runtime.tailscale_status(evidence)["ready"] is True
    for key, value in (("hostname", "other.example-tail.ts.net"), ("connected", False), ("certificate_ready", False), ("mcp_url", f"https://{HOST}:443/mcp"), ("expires_at", "2000-01-01T00:00:00+00:00"), ("expires_at", None)):
        bad = {**evidence, "tailscale": {**evidence["tailscale"], key: value}}
        assert runtime.tailscale_status(bad)["ready"] is False
    assert runtime.tailscale_status({**evidence, "status": "starting"})["ready"] is False
    assert runtime.tailscale_status({**evidence, "pid": -1})["ready"] is False
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: pytest.fail("status must not execute commands"))
    assert runtime.tailscale_status(evidence)["ready"] is True
    runtime.auth.set_enabled(False, Principal("owner", "Test owner", (), owner=True))
    assert runtime.tailscale_status(evidence)["ready"] is False


def _oauth_token(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:49172", ready=False, session_id="test", auth_boundary_version=2)
    runtime.configure_tailscale(enabled=True, hostname=HOST)
    app, _ = _app(runtime)
    origin = f"https://{HOST}:{runtime.port}"
    with TestClient(app, base_url=origin, follow_redirects=False) as client:
        metadata = client.get("/.well-known/oauth-authorization-server")
        assert metadata.status_code == 200
        assert metadata.json()["issuer"] == origin
        assert all(metadata.json()[field].startswith(origin + "/oauth/") for field in ("authorization_endpoint", "token_endpoint", "registration_endpoint"))
        assert client.get("/.well-known/oauth-protected-resource/mcp").json()["resource"] == origin + "/mcp"
        response = client.post("/api/v1/agent-access/owner-login", json={"code": runtime.auth.issue_owner_code()}, headers={"Origin": origin})
        assert response.status_code == 200
        assert client.post("/api/v1/agent-access/settings", json={"enabled": True}, headers={"Origin": origin}).status_code == 200
        registered = client.post("/oauth/register", json={"client_name": "Disposable Tailscale test", "redirect_uris": ["http://127.0.0.1:49173/callback"]})
        assert registered.status_code == 201
        info = registered.json()
        verifier, challenge = _pkce()
        response = client.get("/oauth/authorize", params={"response_type": "code", "client_id": info["client_id"], "redirect_uri": info["redirect_uris"][0], "scope": "observe", "resource": origin + "/mcp", "state": "test-state", "code_challenge": challenge, "code_challenge_method": "S256"})
        assert response.status_code == 303
        consent_url = response.headers["location"]
        assert consent_url.startswith("/api/v1/agent-access/authorizations/")
        response = client.post(consent_url, data={"decision": "approve", "scope": "observe"}, headers={"Origin": origin})
        assert response.status_code == 303
        code = parse_qs(urlsplit(response.headers["location"]).query)["code"][0]
        response = client.post("/oauth/token", data={"grant_type": "authorization_code", "client_id": info["client_id"], "code": code, "redirect_uri": info["redirect_uris"][0], "code_verifier": verifier, "resource": origin + "/mcp"})
        assert response.status_code == 200
        token = response.json()["access_token"]
        client.cookies.clear()
        assert client.get("/mcp", headers={"Accept": "application/json, text/event-stream"}).status_code == 401
        bad = client.get("/.well-known/oauth-authorization-server", headers={"Host": f"other.example-tail.ts.net:{runtime.port}"})
        assert bad.status_code == 421
    return runtime, token


def test_tailscale_oauth_uses_approved_domain_and_pkce(tmp_path):
    _oauth_token(tmp_path)


def test_provisioning_requires_owner_and_boolean_opt_in_and_waits_for_listener(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    runtime.register_backend("http://127.0.0.1:49172", ready=False, session_id="test", auth_boundary_version=2)
    calls = []
    class Certificates:
        def __init__(self, _directory):
            pass
        def detect(self):
            return self.status()
        def status(self):
            return SimpleNamespace(installed=True, connected=True, hostname=HOST, certificate_ready=True, error=None)
        def ensure_certificate(self):
            calls.append("provision")
            return SimpleNamespace(hostname=HOST)
    monkeypatch.setattr("services.agent_access.tailscale.TailscaleCertificateManager", Certificates)
    app, _ = _app(runtime)
    with TestClient(app, base_url=runtime.origin) as client:
        assert client.post("/api/v1/agent-access/tailscale", json={"enabled": True}).status_code == 401
        assert calls == []
        assert client.post("/api/v1/agent-access/owner-login", json={"code": runtime.auth.issue_owner_code()}, headers={"Origin": runtime.origin}).status_code == 200
        # Provisioning is not a back door to enable the whole agent listener.
        assert client.post("/api/v1/agent-access/tailscale", json={"enabled": True}, headers={"Origin": runtime.origin}).status_code == 409
        assert calls == []
        assert client.post("/api/v1/agent-access/settings", json={"enabled": True}, headers={"Origin": runtime.origin}).status_code == 200
        for body in ({"enabled": "true"}, {"enabled": True, "hostname": "attacker.tail.ts.net"}, {}):
            assert client.post("/api/v1/agent-access/tailscale", json=body, headers={"Origin": runtime.origin}).status_code == 400
        response = client.post("/api/v1/agent-access/tailscale", json={"enabled": True}, headers={"Origin": runtime.origin})
        assert response.status_code == 200
        assert response.json()["enabled"] is True
        assert response.json()["ready"] is False
        assert response.json()["mcp_url"] is None
        assert runtime.tailscale_hostname == HOST
        assert calls == ["provision"]
        assert client.post("/api/v1/agent-access/tailscale", json={"enabled": False}, headers={"Origin": runtime.origin}).status_code == 200
        assert runtime.tailscale_hostname is None
        token = _agent_token(runtime, Principal("owner", "owner", (), owner=True))["access_token"]
        client.cookies.clear()
        assert client.get("/api/v1/agent-access/tailscale", headers={"Authorization": "Bearer " + token}).status_code == 403
        assert client.post("/api/v1/agent-access/tailscale", json={"enabled": True}, headers={"Authorization": "Bearer " + token}).status_code == 403
        assert calls == ["provision"]


def _fixture_certificate(directory):
    key = ec.generate_private_key(ec.SECP256R1())
    root_key = ec.generate_private_key(ec.SECP256R1())
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Disposable test root")])
    now = datetime.now(timezone.utc)
    root = x509.CertificateBuilder().subject_name(root_name).issuer_name(root_name).public_key(root_key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1)).add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True).sign(root_key, hashes.SHA256())
    cert = x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, HOST)])).issuer_name(root_name).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1)).add_extension(x509.SubjectAlternativeName([x509.DNSName(HOST)]), critical=False).add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False).sign(root_key, hashes.SHA256())
    cert_path, key_path, root_path = (directory / name for name in ("tail-cert.pem", "tail-key.pem", "test-root.pem"))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    root_path.write_bytes(root.public_bytes(serialization.Encoding.PEM))
    return SimpleNamespace(hostname=HOST, cert_path=cert_path, key_path=key_path), root_path


def test_real_sdk_mcp_over_sni_keeps_lan_certificate_intact(tmp_path, monkeypatch):
    runtime, token = _oauth_token(tmp_path)
    app, _ = _app(runtime)
    lan = ensure_tls(runtime.directory / "tls", runtime.installation_id)
    material, root_path = _fixture_certificate(tmp_path)
    config = uvicorn.Config(app, host="127.0.0.1", port=runtime.port, ssl_certfile=str(lan.cert_path), ssl_keyfile=str(lan.key_path), access_log=False, log_level="critical")
    config.load()
    _configure_tailscale_sni(config.ssl, runtime, material)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    real_lookup = socket.getaddrinfo
    def lookup(host, *args, **kwargs):
        if host in (HOST, HOST.encode()):
            host = "127.0.0.1"
        return real_lookup(host, *args, **kwargs)
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    thread.start()
    try:
        deadline = time.monotonic() + 8
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        lan_context = ssl.create_default_context(cafile=str(lan.cert_path))
        with (
            socket.create_connection(("127.0.0.1", runtime.port), timeout=3) as raw,
            lan_context.wrap_socket(raw, server_hostname="127.0.0.1") as secure,
        ):
            assert secure.getpeercert(binary_form=True) == ssl.PEM_cert_to_DER_cert(lan.cert_path.read_text())
        async def verify():
            context = ssl.create_default_context(cafile=str(root_path))
            async with (
                httpx2.AsyncClient(verify=context, headers={"Authorization": "Bearer " + token}, trust_env=False) as client,
                streamable_http_client(f"https://{HOST}:{runtime.port}/mcp", http_client=client) as streams,
                ClientSession(*streams, read_timeout_seconds=10.0) as session,
            ):
                initialized = await session.initialize()
                assert initialized.server_info.name
                listing = await session.list_tools()
                assert "get_status" in {tool.name for tool in listing.tools}
        asyncio.run(verify())
        runtime.configure_tailscale(enabled=False)
        with socket.create_connection(("127.0.0.1", runtime.port), timeout=3) as raw, pytest.raises(ssl.SSLCertVerificationError):
            ssl.create_default_context(cafile=str(root_path)).wrap_socket(raw, server_hostname=HOST)
    finally:
        server.should_exit = True
        thread.join(timeout=8)
        assert not thread.is_alive()
