"""Control-plane identity and read-only process checks without model imports."""
import concurrent.futures
import json
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.agent_access.state import (
    ControllerLock,
    RuntimeState,
    private_lan_ipv4_addresses,
    process_alive,
)


def test_process_probe_is_read_only():
    assert process_alive(os.getpid()) is True
    assert process_alive(-1) is False
    assert process_alive(None) is False


def test_initial_port_uses_os_fallback_for_windows_reserved_blocks(tmp_path, monkeypatch):
    real_socket = socket.socket

    class ReservedBlockProbe:
        def __init__(self):
            self.socket = real_socket()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.socket.close()

        def bind(self, address):
            if address[1] != 0:
                raise OSError("Reserved by the operating system")
            self.socket.bind(address)

        def getsockname(self):
            return self.socket.getsockname()

    monkeypatch.setattr(socket, "socket", ReservedBlockProbe)
    runtime = RuntimeState(tmp_path / "agent-access")
    selected = runtime.port
    assert 1024 <= selected <= 65535
    assert RuntimeState(tmp_path / "agent-access").port == selected


def test_controller_lock_is_exclusive_and_released(tmp_path):
    with ControllerLock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with ControllerLock(tmp_path):
                pytest.fail("Second controller acquired the installation lock")
    with ControllerLock(tmp_path):
        pass


def test_concurrent_identity_and_endpoint_are_stable(tmp_path):
    app_root = tmp_path / "app"
    directory = app_root / "settings" / "agent-access"
    def create(_):
        runtime = RuntimeState(directory, app_root=app_root)
        return runtime.installation_id, runtime.mcp_url
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        identities = list(pool.map(create, range(10)))
    assert len(set(identities)) == 1
    assert RuntimeState(directory, app_root=app_root).auth.enabled is False
    with pytest.raises(ValueError, match="another installation"):
        RuntimeState(directory, app_root=tmp_path / "copy" / "app")


def test_private_lan_ipv4_lookup_is_local_filtered_and_deduplicated():
    records = [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.8", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.8", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.8", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.1.2", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0)),
        (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("fd12::8", 0, 0, 0)),
    ]
    with patch("services.agent_access.state.socket.getaddrinfo", return_value=records) as resolver:
        assert private_lan_ipv4_addresses() == ("10.0.0.8", "192.168.1.8")
    resolver.assert_called_once_with(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)


def test_controller_origin_and_resource_aliases_are_exact_certificate_and_port_bound(tmp_path):
    runtime = RuntimeState(tmp_path / "agent-access")
    port = runtime.port
    runtime.configure_certificate_ips(("192.168.4.25", "2001:db8::25", "127.0.0.1"))

    def scope(host, scheme="https", peer="192.168.4.30"):
        return {
            "scheme": scheme,
            "client": (peer, 44444),
            "headers": [(b"host", host.encode("ascii"))],
        }

    assert runtime.origin_for_scope(scope(f"{runtime.hostname}:{port}")) == runtime.origin
    assert runtime.origin_for_scope(scope(f"192.168.4.25:{port}")) == f"https://192.168.4.25:{port}"
    assert runtime.origin_for_scope(scope(f"[2001:db8::25]:{port}")) == f"https://[2001:db8::25]:{port}"
    assert runtime.origin_for_scope(scope(f"192.168.4.25:{port + 1}")) is None
    assert runtime.origin_for_scope(scope(f"192.168.4.26:{port}")) is None
    assert runtime.origin_for_scope(scope(f"192.168.4.25:{port}", scheme="http")) is None
    assert runtime.origin_for_scope(scope(f"127.0.0.1:{port}", scheme="http", peer="127.0.0.1")) == f"http://127.0.0.1:{port}"

    assert runtime.canonical_resource(f"https://192.168.4.25:{port}/mcp") == runtime.mcp_url
    assert runtime.canonical_resource(f"https://[2001:db8::25]:{port}/mcp") == runtime.mcp_url
    with pytest.raises(ValueError, match="approved Maestro MCP URL"):
        runtime.canonical_resource(f"https://192.168.4.26:{port}/mcp")
    with pytest.raises(ValueError, match="approved Maestro MCP URL"):
        runtime.canonical_resource(f"https://192.168.4.25:{port + 1}/mcp")


def test_public_status_exposes_only_current_lan_ips_covered_by_certificate_sans(tmp_path):
    runtime = RuntimeState(tmp_path / "agent-access")
    port = runtime.port
    runtime.configure_certificate_ips(("192.168.4.25", "192.168.4.26"))
    network = runtime.lan_endpoint_metadata(("192.168.4.25", "192.168.4.27", "127.0.0.1"))
    expected_url = f"https://192.168.4.25:{port}/mcp"
    assert network["lan_mcp_urls"] == [expected_url]

    (runtime.directory / "controller.json").write_text(
        json.dumps({"status": "stopped", "discovery_status": "disabled", "lan_mcp_urls": []}),
        encoding="utf-8",
    )
    status = runtime.public_status()
    assert status["lan_mcp_urls"] == []
    assert status["discovery_status"] == "disabled"

    (runtime.directory / "controller.json").write_text(json.dumps({
        "status": "ready", "discovery_status": "disabled", "lan_mcp_urls": [expected_url],
    }), encoding="utf-8")
    assert runtime.public_status()["lan_mcp_urls"] == []
    session = runtime.auth.redeem_owner_code(runtime.auth.issue_owner_code())
    owner = runtime.auth.authenticate(session["access_token"])
    runtime.auth.set_enabled(True, owner)
    assert runtime.public_status()["lan_mcp_urls"] == [expected_url]
    runtime.auth.set_enabled(False, owner)
    assert runtime.public_status()["lan_mcp_urls"] == []
