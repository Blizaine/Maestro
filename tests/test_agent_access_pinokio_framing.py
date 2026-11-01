"""CSP permits only the configured Pinokio shell and this host's own LAN IPs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.testclient import TestClient

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access import boundary as boundary_module  # noqa: E402
from services.agent_access import state as state_module  # noqa: E402
from services.agent_access.boundary import (  # noqa: E402
    AccessBoundary,
    frame_ancestor_sources,
    pinokio_access_origin,
    pinokio_frame_origins,
)
from test_agent_access_local_device import _runtime  # noqa: E402


def _write_access_config(path: Path, *, protocol="http", host="172.31.48.1", port=42000) -> Path:
    path.write_text(
        json.dumps({"access": {"protocol": protocol, "host": host, "port": port}}),
        encoding="utf-8",
    )
    return path


def _patch_private_candidates(monkeypatch, candidates: tuple[str, ...]) -> None:
    def candidates_for_test():
        return tuple(candidates)

    monkeypatch.setattr(state_module, "private_lan_ipv4_addresses", candidates_for_test)


def test_pinokio_frame_origins_expands_only_this_hosts_valid_private_ipv4s(tmp_path, monkeypatch):
    config = _write_access_config(tmp_path / "pinokio.json")
    _patch_private_candidates(monkeypatch, (
        "192.168.68.65",
        "10.12.0.9",
        "8.8.8.8",
        "169.254.4.5",
        "127.0.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "fd00::1",
        "192.168.68.65/24",
        "not-an-ip",
        "",
    ))

    assert pinokio_access_origin(config) == "http://172.31.48.1:42000"
    origins = set(pinokio_frame_origins(config))
    assert origins == {
        "http://172.31.48.1:42000",
        "http://192.168.68.65:42000",
        "http://10.12.0.9:42000",
    }
    assert "http://192.168.68.66:42000" not in origins
    assert "http://192.168.68.65:42015" not in origins
    assert "https://192.168.68.65:42000" not in origins


def test_pinokio_frame_origins_preserves_configured_scheme_for_host_aliases(tmp_path, monkeypatch):
    config = _write_access_config(tmp_path / "pinokio.json", protocol="https")
    _patch_private_candidates(monkeypatch, ("192.168.68.65",))

    assert set(pinokio_frame_origins(config)) == {
        "https://172.31.48.1:42000",
        "https://192.168.68.65:42000",
    }


@pytest.mark.parametrize(
    "host",
    ["8.8.8.8", "169.254.4.5", "127.0.0.1", "0.0.0.0", "not-an-ip", "192.168.68.65/24"],
)
def test_invalid_configured_origin_never_expands_to_this_hosts_lan_ips(tmp_path, monkeypatch, host):
    config = _write_access_config(tmp_path / "pinokio.json", host=host)
    _patch_private_candidates(monkeypatch, ("192.168.68.65",))

    assert pinokio_access_origin(config) is None
    assert pinokio_frame_origins(config) == ()


def test_missing_pinokio_origin_does_not_expand_host_addresses(tmp_path, monkeypatch):
    missing_config = tmp_path / "missing-pinokio.json"
    _patch_private_candidates(monkeypatch, ("192.168.68.65",))
    monkeypatch.setattr(
        boundary_module,
        "pinokio_frame_origins",
        lambda config_path=None: pinokio_frame_origins(missing_config),
    )

    assert pinokio_frame_origins(missing_config) == ()
    assert set(frame_ancestor_sources()) == {
        "'self'",
        "http://localhost:*",
        "https://localhost:*",
        "http://127.0.0.1:*",
        "https://127.0.0.1:*",
        "http://[::1]:*",
        "https://[::1]:*",
    }


def test_explicit_frame_origins_do_not_expand_from_pinokio_config_or_host_ips(monkeypatch):
    def unexpected_config_lookup(config_path=None):
        raise AssertionError("unexpected Pinokio config lookup")

    monkeypatch.setattr(
        boundary_module,
        "pinokio_frame_origins",
        unexpected_config_lookup,
        raising=False,
    )

    sources = frame_ancestor_sources(("http://172.31.48.1:42000",))
    assert sources == (
        "'self'",
        "http://localhost:*",
        "https://localhost:*",
        "http://127.0.0.1:*",
        "https://127.0.0.1:*",
        "http://[::1]:*",
        "https://[::1]:*",
        "http://172.31.48.1:42000",
    )
    assert frame_ancestor_sources(()) == sources[:7]


def test_pinokio_origins_are_added_to_html_csp_without_removing_existing_sandbox(tmp_path, monkeypatch):
    config = _write_access_config(tmp_path / "pinokio.json")
    _patch_private_candidates(monkeypatch, ("192.168.68.65",))
    monkeypatch.setattr(
        boundary_module,
        "pinokio_frame_origins",
        lambda config_path=None: pinokio_frame_origins(config),
    )
    runtime = _runtime(tmp_path / "runtime")
    inner = FastAPI()

    @inner.get("/")
    async def index():
        return HTMLResponse(
            "<!doctype html><title>Maestro</title>",
            headers={"Content-Security-Policy": "default-src 'none'; sandbox"},
        )

    app = AccessBoundary(inner, runtime=runtime, allow_local_ui=True)
    with TestClient(app, base_url=f"http://127.0.0.1:{runtime.port}") as client:
        response = client.get("/")

    assert response.status_code == 200
    policies = response.headers.get_list("content-security-policy")
    assert policies[0] == "default-src 'none'; sandbox"
    assert len(policies) == 2
    assert "frame-ancestors" in policies[1]
    assert "http://172.31.48.1:42000" in policies[1]
    assert "http://192.168.68.65:42000" in policies[1]


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/"), ("HEAD", "/"), ("GET", "/index.html"), ("HEAD", "/index.html")],
)
def test_shell_requests_bypass_stale_conditionals_and_return_fresh_no_store_html(tmp_path, method, path):
    runtime = _runtime(tmp_path)
    inner = FastAPI()
    observed_headers = {}

    async def shell(request: Request):
        observed_headers[request.url.path] = {
            "if-none-match": request.headers.get("if-none-match"),
            "if-modified-since": request.headers.get("if-modified-since"),
        }
        if any(observed_headers[request.url.path].values()):
            return Response(
                status_code=304,
                headers={"ETag": '"shell-v1"', "Cache-Control": "public, max-age=3600"},
            )
        return HTMLResponse(
            "<!doctype html><title>Fresh Maestro shell</title>",
            headers={
                "ETag": '"shell-v1"',
                "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
                "Cache-Control": "public, max-age=3600",
                "Content-Security-Policy": "default-src 'self'; sandbox",
            },
        )

    inner.add_api_route("/", shell, methods=["GET", "HEAD"])
    inner.add_api_route("/index.html", shell, methods=["GET", "HEAD"])
    app = AccessBoundary(
        inner,
        runtime=runtime,
        allow_local_ui=True,
        frame_ancestor_origins=("http://172.31.48.1:42000",),
    )

    with TestClient(app, base_url=f"http://127.0.0.1:{runtime.port}") as client:
        response = client.request(
            method,
            path,
            headers={
                "If-None-Match": '"stale-shell"',
                "If-Modified-Since": "Mon, 01 Jan 2000 00:00:00 GMT",
            },
        )

    assert response.status_code == 200
    assert observed_headers[path] == {"if-none-match": None, "if-modified-since": None}
    assert response.headers["cache-control"] == "no-store"
    assert "etag" not in response.headers
    assert "last-modified" not in response.headers
    if method == "GET":
        assert "Fresh Maestro shell" in response.text
    else:
        assert response.content == b""
    policies = response.headers.get_list("content-security-policy")
    assert policies[0] == "default-src 'self'; sandbox"
    assert any("frame-ancestors" in policy and "http://172.31.48.1:42000" in policy for policy in policies)


@pytest.mark.parametrize("path", ["/main.js", "/uploads/probe.html"])
def test_assets_and_uploads_keep_conditional_responses_and_their_own_policy(tmp_path, path):
    runtime = _runtime(tmp_path)
    runtime.auth.mark_owner_configured()
    runtime.mark_security_configured()
    inner = FastAPI()
    observed_headers = {}

    @inner.api_route("/main.js", methods=["GET", "HEAD"])
    async def javascript(request: Request):
        observed_headers[request.url.path] = (
            request.headers.get("if-none-match"),
            request.headers.get("if-modified-since"),
        )
        return Response(
            status_code=304,
            headers={
                "ETag": '"asset-v1"',
                "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
                "Cache-Control": "public, max-age=3600",
            },
        )

    @inner.api_route("/uploads/probe.html", methods=["GET", "HEAD"])
    async def uploaded_html(request: Request):
        observed_headers[request.url.path] = (
            request.headers.get("if-none-match"),
            request.headers.get("if-modified-since"),
        )
        return Response(
            status_code=304,
            headers={
                "ETag": '"upload-v1"',
                "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
                "Cache-Control": "private, max-age=120",
                "Content-Type": "text/html",
                "Content-Security-Policy": "default-src *",
                "Content-Disposition": "inline",
            },
        )

    app = AccessBoundary(
        inner,
        runtime=runtime,
        allow_local_ui=True,
        allow_legacy_browser=True,
        frame_ancestor_origins=("http://172.31.48.1:42000",),
    )
    headers = {
        "If-None-Match": '"cached-resource"',
        "If-Modified-Since": "Mon, 01 Jan 2000 00:00:00 GMT",
    }

    with TestClient(app, base_url=f"http://127.0.0.1:{runtime.port}") as client:
        response = client.get(path, headers=headers)

    assert response.status_code == 304
    assert observed_headers[path] == (
        '"cached-resource"',
        "Mon, 01 Jan 2000 00:00:00 GMT",
    )
    assert "no-store" not in response.headers.get("cache-control", "")
    if path == "/main.js":
        assert response.headers["cache-control"] == "public, max-age=3600"
        assert response.headers["etag"] == '"asset-v1"'
        assert response.headers["last-modified"] == "Mon, 01 Jan 2024 00:00:00 GMT"
        assert not response.headers.get_list("content-security-policy")
    else:
        assert response.headers["cache-control"] == "private, max-age=120"
        assert response.headers["etag"] == '"upload-v1"'
        assert response.headers["last-modified"] == "Mon, 01 Jan 2024 00:00:00 GMT"
        assert response.headers["content-disposition"] == "attachment"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "sandbox" in response.headers["content-security-policy"]

