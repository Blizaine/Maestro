"""Reuse Pinokio's verified HTTPS web route without configuring its network.

Only MCP/OAuth/Agent Access requests cross this bridge. The native browser UI
keeps its existing access policy. The CPU controller verifies a private signed
origin envelope, then applies its ordinary OAuth, CSRF and scope checks.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import ssl
import time
from urllib.parse import urlsplit

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

from .state import _pinokio_web_origin

_ORIGIN = b"x-maestro-pinokio-origin"
_TIME = b"x-maestro-pinokio-time"
_SIGNATURE = b"x-maestro-pinokio-signature"
_ENVELOPE = {_ORIGIN, _TIME, _SIGNATURE}
_HOP = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
        b"te", b"trailer", b"transfer-encoding", b"upgrade"}
_METADATA = {"/.well-known/maestro", "/.well-known/oauth-authorization-server",
             "/.well-known/openid-configuration", "/.well-known/oauth-protected-resource",
             "/.well-known/oauth-protected-resource/mcp"}


def _protocol_path(path):
    return path in _METADATA or path in {"/mcp", "/mcp/"} or path.startswith(
        ("/oauth/", "/agent-access/", "/api/v1/agent-access/")
    )


def _loopback(scope):
    try:
        peer = scope.get("maestro.transport_client", scope.get("client"))
        return ipaddress.ip_address((peer or ("", 0))[0]).is_loopback
    except ValueError:
        return False


def _clean_headers(headers):
    connection_names = set()
    for key, value in headers:
        if key.lower() == b"connection":
            connection_names.update(item.strip().lower() for item in value.split(b","))
    return [(key, value) for key, value in headers
            if key.lower() not in _HOP | _ENVELOPE | connection_names
            and key.lower() not in {b"forwarded", b"x-real-ip"}
            and not key.lower().startswith(b"x-forwarded-")]


def _signature(key, scope, origin, timestamp):
    target = scope.get("raw_path") or scope.get("path", "/").encode("utf-8")
    query = scope.get("query_string", b"")
    payload = (timestamp.encode("ascii") + b"\n" + origin.encode("ascii") + b"\n"
               + scope.get("method", "GET").encode("ascii") + b"\n" + target + b"?" + query)
    return hmac.new(key, payload, hashlib.sha256).hexdigest().encode("ascii")


class PinokioControllerEnvelope:
    """Accept the native bridge only on TLS loopback, never forwarded identity."""

    def __init__(self, app, runtime, *, clock=time.time):
        self.app, self.runtime, self.clock = app, runtime, clock

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = scope.get("headers", [])
        if not any(key.lower() in _ENVELOPE for key, _ in headers):
            return await self.app(scope, receive, send)
        try:
            envelope = {}
            for name in _ENVELOPE:
                values = [v for k, v in headers if k.lower() == name]
                if len(values) != 1:
                    raise ValueError()
                envelope[name] = values[0].decode("ascii")
            origin, timestamp = envelope[_ORIGIN], envelope[_TIME]
            if (not _loopback(scope) or scope.get("scheme") != "https"
                    or not _protocol_path(scope.get("path", ""))
                    or origin != self.runtime.pinokio_remote_origin
                    or not timestamp.isdecimal() or abs(self.clock() - int(timestamp)) > 30
                    or not hmac.compare_digest(envelope[_SIGNATURE].encode("ascii"),
                        _signature(self.runtime.auth.pinokio_proxy_key(), scope, origin, timestamp))):
                raise ValueError()
        except (ValueError, UnicodeError, TypeError):
            return await JSONResponse({"detail": "Invalid Pinokio Remote connection"}, 421)(scope, receive, send)
        parsed = urlsplit(origin)
        # This synthetic peer can never receive direct-local owner authority.
        scope = {**scope, "scheme": "https", "server": (parsed.hostname, parsed.port or 443),
                 "client": ("pinokio-remote", 0), "maestro.pinokio_origin": origin,
                 "headers": [(k, v) for k, v in _clean_headers(headers) if k.lower() != b"host"]
                            + [(b"host", parsed.netloc.encode("ascii")),
                               (b"x-forwarded-proto", b"https")]}
        return await self.app(scope, receive, send)


class PinokioNativeMcpProxy:
    """Forward only this installation's approved Pinokio origin to its controller."""

    def __init__(self, app, runtime, *, discovery=None, client_factory=None, clock=time.time):
        self.app, self.runtime, self.discovery = app, runtime, discovery
        self.client_factory, self.clock = client_factory, clock

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # HTTP headers are never used to choose a host to contact. Discover
        # from the trusted local control plane only for plausible proxy traffic.
        hosts = [v for k, v in scope.get("headers", []) if k.lower() == b"host"]
        if not _loopback(scope) or len(hosts) != 1 or b".ts.net" not in hosts[0].lower():
            return await self.app(scope, receive, send)
        if self.discovery is None:
            from ..pinokio_remote import PinokioRemoteDiscovery
            self.discovery = PinokioRemoteDiscovery(self.runtime.app_root)
        evidence = await asyncio.to_thread(self.discovery.status)
        origin = _pinokio_web_origin(evidence)
        if not origin or hosts[0] != urlsplit(origin).netloc.encode("ascii"):
            # Other Tailscale routes and ordinary LAN access retain their own
            # existing policy. A stale/unknown route never becomes this proxy.
            return await self.app(scope, receive, send)
        if not _protocol_path(scope.get("path", "")):
            # Pinokio's portal uses :443 and opens apps on their own stable
            # HTTPS ports. Allow only that verified node to embed this UI.
            node_origin = "https://" + urlsplit(origin).hostname
            scope = {**scope, "maestro.pinokio_frame_origin": node_origin}
            return await self.app(scope, receive, send)
        if not self.runtime.auth.enabled or self.runtime.config.get("pinokio_remote_enabled") is not True:
            return await self.app(scope, receive, send)
        timestamp = str(int(self.clock()))
        headers = [(k, v) for k, v in _clean_headers(scope.get("headers", []))
                   if k.lower() not in {b"host", b"accept-encoding"}]
        headers += [(_ORIGIN, origin.encode("ascii")), (_TIME, timestamp.encode("ascii")),
                    (_SIGNATURE, _signature(self.runtime.auth.pinokio_proxy_key(), scope, origin, timestamp)),
                    (b"accept-encoding", b"identity")]
        request = Request(scope, receive)
        async def body():
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > 500 * 1024 * 1024:
                    raise ValueError("Request body is too large")
                yield chunk
        cert = self.runtime.directory / "tls" / "server-cert.pem"
        response = None
        try:
            context = ssl.create_default_context(cafile=str(cert))
            factory = self.client_factory or httpx.AsyncClient
            async with factory(verify=context, trust_env=False, follow_redirects=False,
                               timeout=httpx.Timeout(660, connect=5, write=60, pool=5)) as client:
                target = self.runtime.local_url + scope.get("raw_path", scope["path"].encode()).decode("ascii")
                if scope.get("query_string"):
                    target += "?" + scope["query_string"].decode("ascii")
                upstream = client.build_request(scope.get("method", "GET"), target, headers=headers, content=body())
                response = await client.send(upstream, stream=True)
                await send({"type": "http.response.start", "status": response.status_code,
                            "headers": _clean_headers(response.headers.raw)})
                async for chunk in response.aiter_raw():
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        except (httpx.HTTPError, OSError, ValueError):
            if response is None:
                await JSONResponse({"detail": "Start or refresh Agent Access in Pinokio to use Remote MCP"}, 503,
                                   headers={"Cache-Control": "no-store"})(scope, receive, send)
            else:
                raise
        finally:
            if response is not None:
                await response.aclose()
