from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.error import URLError
from urllib.request import ProxyHandler

import pytest


_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.pinokio_remote import PinokioRemoteDiscovery  # noqa: E402


_DNS_NAME = "pinokio.tail1234.ts.net"
_REMOTE_URL = f"https://{_DNS_NAME}:8443/"


class _Response:
    def __init__(self, payload: bytes | dict, *, status: int = 200, headers=None):
        self.status = status
        self.headers = dict(headers or {})
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode("utf-8")
        self.payload = payload
        self.read_sizes: list[int] = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.payload[:size]


class _Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, *, timeout: float):
        self.requests.append((request, timeout))
        next_response = self.responses.pop(0)
        if isinstance(next_response, BaseException):
            raise next_response
        return next_response if hasattr(next_response, "__enter__") else _Response(next_response)


def _payload(*apps: dict, **overrides) -> dict:
    value = {
        "phase": "ready",
        "dnsName": _DNS_NAME,
        "machineName": "MAESTRO-DESKTOP",
        "apps": list(apps),
    }
    value.update(overrides)
    return value


def _app(app_id: str = "Maestro-dev", **overrides) -> dict:
    value = {
        "id": app_id,
        "state": "ready",
        "exposed": True,
        "remote": _REMOTE_URL,
    }
    value.update(overrides)
    return value


def _installed(tmp_path: Path, app_id: str = "Maestro-dev") -> tuple[Path, Path, Path]:
    home = tmp_path / "pinokio-home"
    project = home / "api" / app_id
    (project / "app").mkdir(parents=True)
    (project / "pinokio.js").write_text("// Maestro launcher marker\n", encoding="utf-8")
    (project / "start.js").write_text("// Maestro launcher marker\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"home": str(home)}), encoding="utf-8")
    return home, project, config_path


def _without_pinokio_env(monkeypatch):
    for name in ("PINOKIO_HOME", "PINOKIO_PORT", "PINOKIO_REMOTE_STATUS_URL"):
        monkeypatch.delenv(name, raising=False)


def _discovery(project: Path, home: Path, config_path: Path, opener, **kwargs):
    return PinokioRemoteDiscovery(
        project,
        pinokio_home=home,
        endpoint="http://127.0.0.1:42000/anywhere/status",
        opener=opener,
        config_path=config_path,
        cache_ttl=0,
        **kwargs,
    )


def test_returns_only_the_exact_installed_app_route_and_sanitizes_url(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    opener = _Opener([_payload(
        {"id": {}, "state": "ready", "remote": "https://elsewhere.ts.net/"},
        _app("Maestro", remote="https://maestro.ts.net/"),
        _app(
            remote=(
                f"https://{_DNS_NAME}:8443/studio?token=secret"
                "&maestro_ui_policy=lan-v2&other=discard#fragment-secret"
            ),
        ),
    )])

    result = _discovery(project, home, config_path, opener).status()

    assert result == {
        "available": True,
        "phase": "ready",
        "browser_url": f"https://{_DNS_NAME}:8443/studio?maestro_ui_policy=lan-v2",
        "dns_name": _DNS_NAME,
        "node_hostname": "MAESTRO-DESKTOP",
        "app_id": "Maestro-dev",
        "reason": None,
    }
    request, timeout = opener.requests[0]
    assert request.full_url == "http://127.0.0.1:42000/anywhere/status"
    assert request.get_method() == "GET"
    assert "Authorization" not in request.headers
    assert 0 < timeout <= 1


def test_exact_app_id_distinguishes_maestro_from_maestro_dev(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path, "Maestro")
    opener = _Opener([_payload(
        _app("Maestro-dev", remote="https://dev.ts.net/"),
        _app("Maestro", remote=f"https://{_DNS_NAME}/"),
    )])

    result = _discovery(project, home, config_path, opener).status()

    assert result["available"] is True
    assert result["app_id"] == "Maestro"
    assert result["browser_url"] == f"https://{_DNS_NAME}/"


def test_renamed_pinokio_install_uses_its_exact_folder_id(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path, "Maestro.git-dev")
    opener = _Opener([_payload(
        _app("Maestro-dev", remote="https://other.tail1234.ts.net/"),
        _app("Maestro.git-dev", remote=f"https://{_DNS_NAME}/"),
    )])

    result = _discovery(project, home, config_path, opener).status()

    assert result["available"] is True
    assert result["app_id"] == "Maestro.git-dev"
    assert result["browser_url"] == f"https://{_DNS_NAME}/"


def test_sibling_app_in_same_api_directory_is_not_selected(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path, "Maestro.git-dev")
    sibling = home / "api" / "Maestro"
    (sibling / "app").mkdir(parents=True)
    (sibling / "pinokio.js").write_text("// sibling launcher\n", encoding="utf-8")
    (sibling / "start.js").write_text("// sibling launcher\n", encoding="utf-8")
    opener = _Opener([_payload(_app("Maestro", remote=f"https://{_DNS_NAME}/"))])

    result = _discovery(project, home, config_path, opener).status()

    assert result["available"] is False
    assert result["app_id"] == "Maestro.git-dev"
    assert result["reason"] == "app_not_published"


def test_duplicate_exact_app_ids_are_ambiguous(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    opener = _Opener([_payload(_app(), _app(remote=f"https://other.{_DNS_NAME}/"))])

    result = _discovery(project, home, config_path, opener).status()

    assert result["available"] is False
    assert result["reason"] == "ambiguous_app"


@pytest.mark.parametrize("remote", [
    f"https://user:password@{_DNS_NAME}/",
    "https://pinokio.other-tailnet.ts.net/",
    f"https://mcp.{_DNS_NAME}/",
    "http://pinokio.tail1234.ts.net/",
    f"https://{_DNS_NAME}:not-a-port/",
    f"https://{_DNS_NAME}/\\\\evil.example/",
])
def test_untrusted_remote_routes_are_not_returned(tmp_path, monkeypatch, remote):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    opener = _Opener([_payload(_app(remote=remote))])

    result = _discovery(project, home, config_path, opener).status()

    assert result["available"] is False
    assert result["browser_url"] is None
    assert result["reason"] == "invalid_remote_route"


def test_published_route_requires_ready_and_exposed_app_state(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    opener = _Opener([_payload(_app(state="publishing", exposed=False))])

    result = _discovery(project, home, config_path, opener).status()

    assert result["available"] is False
    assert result["reason"] == "app_not_published"


def test_dns_name_must_be_a_tailscale_ts_net_hostname(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    opener = _Opener([_payload(_app(), dnsName="pinokio.example.com")])

    result = _discovery(project, home, config_path, opener).status()

    assert result["available"] is False
    assert result["dns_name"] is None
    assert result["reason"] == "invalid_dns_name"


def test_incomplete_install_and_outside_path_never_query_pinokio(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, _, config_path = _installed(tmp_path)
    wrong_name = home / "api" / "Maestro-preview"
    (wrong_name / "app").mkdir(parents=True)
    wrong_opener = _Opener([_payload(_app("Maestro-preview"))])
    wrong_result = PinokioRemoteDiscovery(
        wrong_name,
        pinokio_home=home,
        endpoint="http://127.0.0.1:42000/anywhere/status",
        opener=wrong_opener,
        config_path=config_path,
    ).status()
    assert wrong_result["reason"] == "app_not_installed"
    assert wrong_opener.requests == []

    outside = tmp_path / "outside" / "Maestro-dev"
    (outside / "app").mkdir(parents=True)
    outside_opener = _Opener([_payload(_app())])
    outside_result = PinokioRemoteDiscovery(
        outside,
        pinokio_home=home,
        endpoint="http://127.0.0.1:42000/anywhere/status",
        opener=outside_opener,
        config_path=config_path,
    ).status()
    assert outside_result["reason"] == "app_not_installed"
    assert outside_opener.requests == []


@pytest.mark.parametrize("endpoint", [
    "http://example.com:42000/anywhere/status",
    "http://192.168.1.4:42000/anywhere/status",
    "https://127.0.0.1:42000/anywhere/status",
    "http://user:secret@127.0.0.1:42000/anywhere/status",
    "http://127.0.0.1:42000/anywhere/status?token=secret",
    "http://localhost:42000/anywhere/status",
])
def test_control_endpoint_must_be_explicit_loopback_without_credentials(endpoint, tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    opener = _Opener([_payload(_app())])

    result = PinokioRemoteDiscovery(
        project,
        pinokio_home=home,
        endpoint=endpoint,
        opener=opener,
        config_path=config_path,
    ).status()

    assert result["available"] is False
    assert result["reason"] == "invalid_control_endpoint"
    assert opener.requests == []


def test_configured_access_port_is_used_on_loopback_only(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    monkeypatch.setenv("PINOKIO_PORT", "80")  # Home Server/Caddy, not daemon API.
    home, project, config_path = _installed(tmp_path)
    config_path.write_text(json.dumps({
        "home": str(home),
        "access": {"host": "192.168.50.10", "port": 42123},
    }), encoding="utf-8")
    opener = _Opener([_payload(_app())])

    result = PinokioRemoteDiscovery(
        project,
        opener=opener,
        config_path=config_path,
        cache_ttl=0,
    ).status()

    assert result["available"] is True
    assert opener.requests[0][0].full_url == "http://127.0.0.1:42123/anywhere/status"


def test_known_default_port_is_used_without_scanning(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    monkeypatch.setenv("PINOKIO_PORT", "80")
    home, project, config_path = _installed(tmp_path)
    opener = _Opener([_payload(_app())])

    PinokioRemoteDiscovery(
        project,
        pinokio_home=home,
        opener=opener,
        config_path=config_path,
        cache_ttl=0,
    ).status()

    assert opener.requests[0][0].full_url == "http://127.0.0.1:42000/anywhere/status"


def test_proxy_environment_is_not_used_by_default_opener(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    home, project, config_path = _installed(tmp_path)

    discovery = PinokioRemoteDiscovery(project, pinokio_home=home, config_path=config_path)
    handlers = [handler for handler in discovery._opener.handlers if isinstance(handler, ProxyHandler)]

    # build_opener removes the environment-aware default handler; the explicit
    # empty ProxyHandler suppresses proxy support and has no request methods.
    assert handlers == []


def test_old_endpoint_and_malformed_payload_fail_gracefully(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    old_endpoint = _discovery(
        project, home, config_path, _Opener([_Response(b"not found", status=404)])
    ).status()
    assert old_endpoint["available"] is False
    assert old_endpoint["phase"] == "unsupported"

    malformed = _discovery(project, home, config_path, _Opener([_Response(b"not json")])).status()
    assert malformed["available"] is False
    assert malformed["phase"] == "error"
    assert malformed["reason"] == "malformed_status"


def test_body_limit_and_timeout_are_bounded(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    response = _Response(b"x" * 2048)
    opener = _Opener([response])

    result = PinokioRemoteDiscovery(
        project,
        pinokio_home=home,
        endpoint="http://127.0.0.1:42000/anywhere/status",
        opener=opener,
        config_path=config_path,
        max_response_bytes=1024,
        timeout=30,
    ).status()

    assert result["available"] is False
    assert result["reason"] == "response_too_large"
    assert response.read_sizes == [1025]
    assert opener.requests[0][1] == 5.0


def test_negative_result_is_cached_and_force_refresh_bypasses_cache(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    now = [0.0]
    opener = _Opener([URLError("offline"), _payload(_app())])
    discovery = PinokioRemoteDiscovery(
        project,
        pinokio_home=home,
        endpoint="http://127.0.0.1:42000/anywhere/status",
        opener=opener,
        config_path=config_path,
        clock=lambda: now[0],
        cache_ttl=2.0,
    )

    first = discovery.status()
    second = discovery.status()
    assert first["phase"] == second["phase"] == "offline"
    assert len(opener.requests) == 1

    refreshed = discovery.status(force_refresh=True)
    assert refreshed["available"] is True
    assert len(opener.requests) == 2


def test_redirects_are_never_followed(tmp_path, monkeypatch):
    _without_pinokio_env(monkeypatch)
    home, project, config_path = _installed(tmp_path)
    requested_paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested_paths.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/followed")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        result = PinokioRemoteDiscovery(
            project,
            pinokio_home=home,
            endpoint=f"http://127.0.0.1:{port}/anywhere/status",
            config_path=config_path,
            cache_ttl=0,
        ).status()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["available"] is False
    assert result["phase"] == "unsupported"
    assert requested_paths == ["/anywhere/status"]

