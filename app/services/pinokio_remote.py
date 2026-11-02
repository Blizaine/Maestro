"""Read-only discovery of Pinokio Remote's published Maestro web route.

This module talks only to Pinokio's local HTTP control plane. It deliberately
does not inspect Tailscale state, invoke a Tailscale CLI, or infer a published
port from another service.
"""

from __future__ import annotations

from copy import deepcopy
import ipaddress
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


_ALLOWED_PHASES = frozenset({
    "setup", "error", "off", "starting", "login", "machine_auth", "https", "ready",
})
_STATUS_PATH = "/anywhere/status"
_REMOTE_STATUS_ENV = "PINOKIO_REMOTE_STATUS_URL"
_DEFAULT_PINOKIO_PORT = 42000
_REQUEST_TIMEOUT_SECONDS = 0.75
_MAX_RESPONSE_BYTES = 256 * 1024
_DEFAULT_CACHE_TTL_SECONDS = 1.5
_DNS_NAME_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+ts\.net$",
    re.IGNORECASE,
)
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}$")


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep even local control-plane requests from following redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _DiscoveryFailure(Exception):
    def __init__(self, reason: str, phase: str):
        super().__init__(reason)
        self.reason = reason
        self.phase = phase


def _default_opener():
    # An empty ProxyHandler disables urllib's HTTP_PROXY / HTTPS_PROXY support.
    return build_opener(ProxyHandler({}), _NoRedirectHandler())


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right))


def _read_pinokio_config(config_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _valid_port(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        port = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return port if 1 <= port <= 65535 else None


def _status_url(value: str) -> str | None:
    """Validate a local Pinokio endpoint and bind it to the known status path.

    Callers may supply either the Pinokio server origin or the exact status
    URL. Only literal loopback IP addresses are accepted; hostnames such as
    ``localhost`` are intentionally not resolved.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
        address = ipaddress.ip_address(parsed.hostname or "")
    except (ValueError, TypeError):
        return None
    if (
        parsed.scheme.lower() != "http"
        or not address.is_loopback
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is None
        or parsed.path not in {"", "/", _STATUS_PATH}
    ):
        return None
    host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"http://{host}:{port}{_STATUS_PATH}"


def _verified_home(
    app_root: Path,
    home_override: Path | str | None,
    config: dict[str, Any],
) -> tuple[Path | None, str | None]:
    raw_home: Path | str | None = home_override
    if raw_home is None:
        configured_home = config.get("home")
        if isinstance(configured_home, str) and configured_home.strip():
            raw_home = configured_home.strip()
    if raw_home is None:
        environment_home = os.environ.get("PINOKIO_HOME")
        if isinstance(environment_home, str) and environment_home.strip():
            raw_home = environment_home.strip()
    if raw_home is None:
        return None, "pinokio_home_unavailable"
    try:
        home_candidate = Path(raw_home).expanduser()
        if not home_candidate.is_absolute():
            return None, "app_not_installed"
        home = home_candidate.resolve(strict=True)
        api_root = (home / "api").resolve(strict=True)
        supplied_path = Path(app_root).expanduser()
        supplied = supplied_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None, "app_not_installed"
    if supplied_path.name == "app":
        if supplied.name != "app":
            return None, "app_not_installed"
        project_candidate = supplied_path.parent
        project_root = supplied.parent
    else:
        project_candidate = supplied_path
        project_root = supplied
    try:
        resolved_project = project_root.resolve(strict=True)
        resolved_api_root = api_root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None, "app_not_installed"
    if (
        project_candidate.name != resolved_project.name
        or not _same_path(resolved_project.parent, resolved_api_root)
    ):
        return None, "app_not_installed"
    app_id = resolved_project.name
    if not app_id or app_id in {".", ".."}:
        return None, "app_not_installed"
    try:
        app_directory = (resolved_project / "app").resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None, "app_not_installed"
    if (
        app_directory.name != "app"
        or not app_directory.is_dir()
        or not _same_path(app_directory.parent, resolved_project)
    ):
        return None, "app_not_installed"
    for marker_name in ("pinokio.js", "start.js"):
        marker_path = resolved_project / marker_name
        try:
            marker = marker_path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return None, "app_not_installed"
        if not marker.is_file() or not _same_path(marker.parent, resolved_project):
            return None, "app_not_installed"
    return resolved_project, None


def _select_endpoint(endpoint: str | None, config: dict[str, Any]) -> tuple[str | None, str | None]:
    explicit = endpoint
    if explicit is None:
        explicit = os.environ.get(_REMOTE_STATUS_ENV)
    if explicit is not None:
        validated = _status_url(explicit)
        return (validated, None) if validated else (None, "invalid_control_endpoint")

    # PINOKIO_PORT can be Home Server's Caddy port (commonly 80), not the
    # daemon/control API port. Only the saved control-plane access port is
    # relevant here; an explicit dedicated status URL remains available.
    access = config.get("access")
    raw_port: Any = access.get("port") if isinstance(access, dict) else None
    port = _valid_port(raw_port) if raw_port is not None else _DEFAULT_PINOKIO_PORT
    if port is None:
        return None, "invalid_control_endpoint"
    validated = _status_url(f"http://127.0.0.1:{port}")
    return (validated, None) if validated else (None, "invalid_control_endpoint")


def _safe_dns_name(value: Any) -> str | None:
    if not isinstance(value, str) or value != value.strip() or len(value) > 253:
        return None
    if not _DNS_NAME_RE.fullmatch(value):
        return None
    return value


def _safe_machine_name(value: Any) -> str | None:
    if not isinstance(value, str) or value != value.strip():
        return None
    return value if _HOSTNAME_RE.fullmatch(value) else None


def _safe_remote_url(value: Any, dns_name: str) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if parsed.username is not None or parsed.password is not None:
            return None
        if parsed.scheme.lower() != "https" or not host or not _DNS_NAME_RE.fullmatch(host):
            return None
        if host.lower() != dns_name.lower():
            return None
        if port is not None and not 1 <= port <= 65535:
            return None
        if any(ord(char) < 32 or ord(char) == 127 for char in parsed.path):
            return None
        query_items = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=False, max_num_fields=64)
    except (ValueError, TypeError):
        return None

    # This is the only query field the Maestro launcher deliberately puts on
    # its published UI URL. Do not forward other query values or fragments.
    policy_values = [value for key, value in query_items if key == "maestro_ui_policy"]
    safe_query = urlencode([("maestro_ui_policy", "lan-v2")]) if policy_values == ["lan-v2"] else ""
    netloc = host.lower()
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit(("https", netloc, parsed.path or "/", safe_query, ""))


class PinokioRemoteDiscovery:
    """Discover the published HTTPS route for this exact Maestro install.

    ``app_root`` is the installed launcher root directly under
    ``PINOKIO_HOME/api``; its ``app`` directory is also accepted. The root
    must contain its own ``app/``, ``pinokio.js`` and ``start.js``. Its exact
    folder name is the app id used to select the status entry. The control
    endpoint may be injected for tests or deployments, but must be an HTTP
    loopback URL. Runtime discovery otherwise uses
    ``~/.pinokio/config.json``'s ``access.port``, or Pinokio's known 42000
    default, always connecting to 127.0.0.1 and never scanning ports.
    """

    def __init__(
        self,
        app_root: Path,
        *,
        pinokio_home: Path | str | None = None,
        endpoint: str | None = None,
        opener: Any = None,
        clock: Callable[[], float] | None = None,
        cache_ttl: float = _DEFAULT_CACHE_TTL_SECONDS,
        config_path: Path | str | None = None,
        timeout: float = _REQUEST_TIMEOUT_SECONDS,
        max_response_bytes: int = _MAX_RESPONSE_BYTES,
    ):
        self._app_root = Path(app_root)
        self._pinokio_home = pinokio_home
        self._endpoint = endpoint
        self._opener = opener if opener is not None else _default_opener()
        self._clock = clock if clock is not None else time.monotonic
        self._cache_ttl = max(0.0, float(cache_ttl))
        self._config_path = (
            Path(config_path)
            if config_path is not None
            else Path.home() / ".pinokio" / "config.json"
        )
        self._timeout = max(0.05, min(float(timeout), 5.0))
        self._max_response_bytes = max(1024, min(int(max_response_bytes), 2 * 1024 * 1024))
        self._lock = threading.RLock()
        self._cache: dict[str, Any] | None = None
        self._cache_until = 0.0

    def status(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """Return a sanitized status snapshot, using a short positive/negative cache."""
        with self._lock:
            now = self._clock()
            if not force_refresh and self._cache is not None and now < self._cache_until:
                return deepcopy(self._cache)
            result = self._discover()
            self._cache = deepcopy(result)
            self._cache_until = self._clock() + self._cache_ttl
            return deepcopy(result)

    def _base_status(
        self,
        *,
        phase: str = "unsupported",
        reason: str,
        app_id: str | None = None,
        dns_name: str | None = None,
        node_hostname: str | None = None,
    ) -> dict[str, Any]:
        return {
            "available": False,
            "phase": phase,
            "browser_url": None,
            "dns_name": dns_name,
            "node_hostname": node_hostname,
            "app_id": app_id,
            "reason": reason,
        }

    def _discover(self) -> dict[str, Any]:
        config = _read_pinokio_config(self._config_path)
        project_root, path_error = _verified_home(self._app_root, self._pinokio_home, config)
        if path_error:
            return self._base_status(reason=path_error)
        assert project_root is not None
        app_id = project_root.name

        endpoint, endpoint_error = _select_endpoint(self._endpoint, config)
        if endpoint_error:
            return self._base_status(reason=endpoint_error, app_id=app_id)
        assert endpoint is not None
        try:
            payload = self._request(endpoint)
        except _DiscoveryFailure as failure:
            return self._base_status(phase=failure.phase, reason=failure.reason, app_id=app_id)

        if not isinstance(payload, dict):
            return self._base_status(reason="malformed_status", app_id=app_id)
        raw_phase = payload.get("phase")
        if not isinstance(raw_phase, str) or raw_phase not in _ALLOWED_PHASES:
            return self._base_status(reason="unsupported_status", app_id=app_id)
        dns_name = _safe_dns_name(payload.get("dnsName"))
        node_hostname = _safe_machine_name(payload.get("machineName"))
        base = self._base_status(
            phase=raw_phase,
            reason="remote_not_ready",
            app_id=app_id,
            dns_name=dns_name,
            node_hostname=node_hostname,
        )
        if raw_phase != "ready":
            return base
        if dns_name is None:
            base["reason"] = "invalid_dns_name"
            return base

        apps = payload.get("apps")
        if not isinstance(apps, list):
            base["reason"] = "malformed_status"
            return base
        matching_apps = [
            app for app in apps
            if isinstance(app, dict) and app.get("id") == app_id
        ]
        if not matching_apps:
            base["reason"] = "app_not_published"
            return base
        if len(matching_apps) != 1:
            base["reason"] = "ambiguous_app"
            return base
        app = matching_apps[0]
        if app.get("state") != "ready" or app.get("exposed") is not True:
            base["reason"] = "app_not_published"
            return base
        remote_url = _safe_remote_url(app.get("remote"), dns_name)
        if remote_url is None:
            base["reason"] = "invalid_remote_route"
            return base
        return {
            "available": True,
            "phase": "ready",
            "browser_url": remote_url,
            "dns_name": dns_name,
            "node_hostname": node_hostname,
            "app_id": app_id,
            "reason": None,
        }

    def _request(self, endpoint: str) -> Any:
        request = Request(
            endpoint,
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-cache",
                "User-Agent": "Maestro-Pinokio-Remote-Discovery/1",
            },
            method="GET",
        )
        try:
            response = self._opener.open(request, timeout=self._timeout)
        except HTTPError as error:
            if error.code == 404 or 300 <= error.code < 400:
                raise _DiscoveryFailure("unsupported_endpoint", "unsupported") from None
            raise _DiscoveryFailure("control_plane_unavailable", "offline") from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise _DiscoveryFailure("control_plane_unavailable", "offline") from None
        except Exception:
            # An injected opener or platform transport must never leak details
            # such as proxy credentials into the public status response.
            raise _DiscoveryFailure("control_plane_unavailable", "offline") from None

        try:
            with response:
                status_code = getattr(response, "status", None)
                if status_code is None and hasattr(response, "getcode"):
                    status_code = response.getcode()
                if status_code != 200:
                    unsupported = isinstance(status_code, int) and (
                        status_code == 404 or 300 <= status_code < 400
                    )
                    phase = "unsupported" if unsupported else "offline"
                    reason = "unsupported_endpoint" if phase == "unsupported" else "control_plane_unavailable"
                    raise _DiscoveryFailure(reason, phase)
                content_length = response.headers.get("Content-Length") if hasattr(response, "headers") else None
                if content_length is not None:
                    try:
                        if int(content_length) > self._max_response_bytes:
                            raise _DiscoveryFailure("response_too_large", "error")
                    except ValueError:
                        raise _DiscoveryFailure("malformed_status", "error") from None
                raw = response.read(self._max_response_bytes + 1)
        except _DiscoveryFailure:
            raise
        except (OSError, TimeoutError, ValueError):
            raise _DiscoveryFailure("control_plane_unavailable", "offline") from None
        except Exception:
            raise _DiscoveryFailure("control_plane_unavailable", "offline") from None
        if len(raw) > self._max_response_bytes:
            raise _DiscoveryFailure("response_too_large", "error")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError, TypeError):
            raise _DiscoveryFailure("malformed_status", "error") from None


__all__ = ["PinokioRemoteDiscovery"]
