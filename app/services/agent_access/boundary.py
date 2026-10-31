"""One authorization boundary for UI, REST, media, WebSockets and MCP.

This module imports no MCP SDK or model code. The model backend and the bundled
CPU controller use the same identity store and policy. Proxy headers never grant
authority; only an owner session or an authenticated, scoped grant can do so.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from urllib.parse import unquote, unquote_plus, urlsplit

from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from .auth import AuthError
from .catalog import required_scope
from .context import credential_context, principal_context, public_origin_context, submission_context
from .jobs import JournalError

COOKIE_NAME = "maestro_owner"
MUTATIONS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
MAX_JSON = 8 * 1024 * 1024
MAX_MULTIPART_BODY = 500 * 1024 * 1024
MAX_MULTIPART_HEADERS = 64 * 1024
MAX_MULTIPART_HEADER_COUNT = 128
MAX_MULTIPART_FIELDS = 128
MAX_MULTIPART_FILES = 32
MAX_MULTIPART_FIELD_BYTES = 1024 * 1024
MULTIPART_CHUNK_SIZE = 64 * 1024
MAX_CONCURRENT_MULTIPART = 2
PUBLIC = frozenset({
    "/api/v1/agent-access/status", "/api/v1/agent-access/request-console-code",
    "/api/v1/agent-access/certificate",
    "/api/v1/agent-access/owner-login", "/agent-access/login",
    "/agent-access/certificate.pem", "/.well-known/maestro",
    "/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp",
    "/.well-known/oauth-authorization-server", "/.well-known/openid-configuration",
    "/oauth/register", "/oauth/authorize", "/oauth/token", "/oauth/revoke",
})
OWNER_PREFIXES = (
    "/api/v1/agent-access/settings", "/api/v1/agent-access/grants",
    "/api/v1/agent-access/browser-code", "/api/v1/agent-access/audit",
    "/api/v1/agent-access/authorizations", "/api/v1/testbench", "/api/v1/promptbench",
)
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:api_?key|token|access_token|refresh_token|password|secret(?:_?key)?|private_key|authorization|authorization_header|bearer_token)$",
    re.IGNORECASE,
)
_CONSOLE_CODE = re.compile(r"(\[Maestro Agent Access\][^\r\n]*?(?:owner|browser) code\s*:)\s*[^\r\n]+", re.IGNORECASE)
_BEARER_VALUE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_HF_VALUE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
_URL_IN_TEXT = re.compile(r"\b(?:https?|wss?)://[^\s\"'<>]+", re.IGNORECASE)
_URL_SECRET_QUERY_KEYS = frozenset({
    "token", "api_key", "apikey", "access_token", "refresh_token", "client_secret",
})
_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_MEDIA_KEYS = frozenset({
    "path", "image", "image_path", "image_start", "image_end", "video_path", "audio_path",
    "input_path", "input_file", "source_path", "file_path", "output_path", "output_dir",
    "image_refs", "ref_images", "reference_images", "video_refs", "audio_refs",
    "control_video", "video_guide", "audio_guide", "audio_prompt", "voice_ref",
    "audio_reference", "reference_video", "reference_audio", "init_image",
})


def cookie_name(runtime) -> str:
    # Cookies are not port-scoped. Main and Dev on 127.0.0.1 must not replace
    # each other's owner sessions.
    return COOKIE_NAME + "_" + runtime.installation_id.replace("-", "")[:12]


def redact(value):
    if isinstance(value, dict):
        return {
            key: ("[redacted]" if _is_secret_key(key) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        text = _HF_VALUE.sub("[redacted]", _BEARER_VALUE.sub("Bearer [redacted]", _CONSOLE_CODE.sub(r"\1 [redacted]", value)))
        return _URL_IN_TEXT.sub(_redact_url_match, text)
    return value


def _redact_url_match(match) -> str:
    candidate = match.group(0)
    suffix = ""
    while candidate and candidate[-1] in ".,;)]}":
        suffix = candidate[-1] + suffix
        candidate = candidate[:-1]
    try:
        parts = urlsplit(candidate)
        if not parts.scheme or not parts.netloc or not parts.query:
            return match.group(0)
    except ValueError:
        return match.group(0)
    base, marker, query_and_fragment = candidate.partition("?")
    if not marker:
        return match.group(0)
    query, fragment_marker, fragment = query_and_fragment.partition("#")
    redacted_segments = []
    changed = False
    for segment in query.split("&"):
        raw_key, equals, raw_value = segment.partition("=")
        normalized_key = _normalize_key(unquote_plus(raw_key))
        if normalized_key in _URL_SECRET_QUERY_KEYS and equals and raw_value:
            redacted_segments.append(raw_key + "=%5Bredacted%5D")
            changed = True
        else:
            redacted_segments.append(segment)
    if not changed:
        return match.group(0)
    sanitized = base + "?" + "&".join(redacted_segments)
    if fragment_marker:
        sanitized += "#" + fragment
    return sanitized + suffix


def _is_secret_key(key: object) -> bool:
    # API responses use both snake_case and camelCase credential names.
    normalized = _normalize_key(key)
    return _SECRET_KEY.search(normalized) is not None


def _normalize_key(key: object) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).replace("-", "_").casefold()


def _reject_redacted_credentials(value, *, depth: int = 0) -> None:
    """Reject placeholders in write payloads so they cannot erase real secrets."""
    if depth > 40:
        raise AuthError(400, "Request nesting is too deep")
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_secret_key(key) and isinstance(item, str) and item.strip().casefold() == "[redacted]":
                raise AuthError(400, "Replace redacted credential placeholders before submitting this change")
            _reject_redacted_credentials(item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_redacted_credentials(item, depth=depth + 1)


def _file_receive(path: Path, upstream=None, *, chunk_size: int = MULTIPART_CHUNK_SIZE):
    """Replay a private spool file as bounded ASGI request chunks."""
    handle = path.open("rb")
    sent_final = False

    async def receive():
        nonlocal sent_final
        if sent_final:
            if upstream is not None:
                return await upstream()
            return {"type": "http.request", "body": b"", "more_body": False}
        body = handle.read(chunk_size)
        if body:
            return {"type": "http.request", "body": body, "more_body": True}
        sent_final = True
        handle.close()
        return {"type": "http.request", "body": b"", "more_body": False}

    receive.close_spool = handle.close
    return receive


def secure_transport(scope: dict) -> bool:
    if scope.get("scheme") in {"https", "wss"}:
        return True
    headers = {k.lower(): v for k, v in scope.get("headers", [])}
    try:
        host = urlsplit("http://" + headers.get(b"host", b"").decode("latin1")).hostname
    except ValueError:
        return False
    peer = str((scope.get("client") or ("", 0))[0])
    try:
        loopback = ipaddress.ip_address(peer).is_loopback
    except ValueError:
        loopback = False
    # A loopback proxy is not evidence that the caller is local. Do not trust a
    # forwarded header here; a configured ASGI proxy layer may set scheme=https.
    forwarded = any(k.startswith(b"x-forwarded-") or k == b"forwarded" for k in headers)
    return loopback and host in {"127.0.0.1", "localhost", "::1"} and not forwarded


def same_origin(scope: dict) -> bool:
    headers = {k.lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
    origin = headers.get(b"origin")
    if not origin:
        return headers.get(b"sec-fetch-site") == "same-origin"
    try:
        parsed = urlsplit(origin)
        scheme = str(scope.get("scheme", "")).lower()
        if scheme == "ws":
            scheme = "http"
        elif scheme == "wss":
            scheme = "https"
        host = urlsplit(scheme + "://" + headers.get(b"host", ""))
        parsed_port = parsed.port
        host_port = host.port
    except ValueError:
        return False
    if (
        scheme not in {"http", "https"}
        or parsed.scheme.lower() != scheme
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.hostname
        or host.username is not None
        or host.password is not None
        or not host.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or host.path
        or host.query
        or host.fragment
    ):
        return False
    default_port = 443 if scheme == "https" else 80
    return (
        parsed.hostname.casefold() == host.hostname.casefold()
        and (parsed_port or default_port) == (host_port or default_port)
    )


def _canonical_additional_roots(additional_roots) -> tuple[Path, ...]:
    if additional_roots is None:
        return ()
    if isinstance(additional_roots, (str, os.PathLike)):
        items = (additional_roots,)
    else:
        try:
            items = tuple(additional_roots)
        except Exception as error:
            raise AuthError(403, "Trusted media roots could not be verified") from error
    roots = []
    try:
        for item in items:
            if not isinstance(item, (str, os.PathLike)):
                raise TypeError("media root must be a filesystem path")
            text = os.fspath(item)
            if not isinstance(text, str) or not text or "\x00" in text:
                raise ValueError("media root is invalid")
            roots.append(Path(text).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise AuthError(403, "Trusted media roots could not be verified") from error
    return tuple(roots)


def validate_media_paths(
    value,
    app_root: Path | None,
    *,
    key: str = "",
    depth: int = 0,
    additional_roots=(),
) -> None:
    trusted_extra_roots = _canonical_additional_roots(additional_roots)
    trusted_roots: list[Path] = []
    if app_root is not None:
        try:
            native_root = Path(app_root).resolve(strict=False)
            trusted_roots.extend(
                (native_root / name).resolve(strict=False)
                for name in ("uploads", "outputs", "outputs2", "storage", "ckpts", "loras", "loras_i2v", "temp")
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise AuthError(403, "Maestro media roots could not be verified") from error
    trusted_roots.extend(trusted_extra_roots)

    def visit(item, field_key: str, current_depth: int) -> None:
        if current_depth > 40:
            raise AuthError(400, "Request nesting is too deep")
        if isinstance(item, dict):
            for name, child in item.items():
                visit(child, _normalize_key(name), current_depth + 1)
        elif isinstance(item, list):
            for child in item:
                visit(child, field_key, current_depth + 1)
        elif isinstance(item, str) and field_key in _MEDIA_KEYS and item:
            if item.startswith(("asset:", "maestro-asset:")):
                raise AuthError(400, "Resolve the uploaded asset ID before using this endpoint")
            if item.startswith(("https://", "http://", "data:")):
                return  # The named operation's own URL/data validator applies.
            if "\x00" in item:
                raise AuthError(400, "Invalid media path")
            # Plain named assets and model IDs are validated by the existing route.
            if not (_WINDOWS_ABSOLUTE.match(item) or "/" in item or "\\" in item):
                return
            path = Path(item)
            candidates = []
            try:
                if path.is_absolute() or _WINDOWS_ABSOLUTE.match(item):
                    candidates.append(path.resolve(strict=False))
                else:
                    if app_root is not None:
                        candidates.append((Path(app_root) / path).resolve(strict=False))
                    candidates.extend((root / path).resolve(strict=False) for root in trusted_extra_roots)
            except (OSError, RuntimeError, ValueError) as error:
                raise AuthError(403, "Media path could not be verified") from error
            if not any(
                candidate == root or candidate.is_relative_to(root)
                for candidate in candidates
                for root in trusted_roots
            ):
                raise AuthError(403, "Agents may use registered media and Maestro storage paths; upload external files first")

    visit(value, key, depth)


def browser_login_origin(status: dict, scope: dict) -> str:
    """Use a certificate-backed address for this browser's actual host."""
    try:
        headers = dict(scope.get("headers", []))
        host = urlsplit("//" + headers.get(b"host", b"").decode("latin1")).hostname
        if host in {"127.0.0.1", "localhost", "::1"}:
            return status.get("local_setup_url") or status.get("controller_url") or ""
        for candidate in status.get("lan_mcp_urls") or []:
            parsed = urlsplit(candidate)
            if parsed.scheme == "https" and parsed.hostname == host and parsed.path == "/mcp":
                return candidate[:-4]
    except (ValueError, TypeError):
        pass
    return status.get("controller_url") or ""


def login_document(status: dict, *, secure: bool = True) -> str:
    # The form sends only a short-lived owner code. Long-lived credentials are
    # delivered as HttpOnly cookies, never exposed to page scripts or URLs.
    import html
    setup_url = html.escape(str(status.get("local_setup_url") or ""), quote=True)
    if not secure:
        address = html.escape(str(status.get("controller_url") or ""), quote=True)
        return f'<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Maestro secure connection</title><main><h1>Open Maestro securely</h1><p>Browser codes are entered over HTTPS. Open the secure address and verify the certificate fingerprint shown in Pinokio.</p><p><a href="{address}">Open secure Maestro</a></p></main>'
    parts = [
        """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,interactive-widget=resizes-content"><title>Connect to Maestro</title>
<style>body{margin:0;background:#171b1e;color:#edf0f2;font:16px system-ui;display:grid;min-height:100svh;place-items:center}main{box-sizing:border-box;width:min(92vw,440px);padding:28px;background:#20262b;border:1px solid #3b444c;border-radius:20px}input,button{box-sizing:border-box;width:100%;font:inherit;padding:13px;border-radius:10px;margin-top:12px}input{background:#171b1e;color:#fff;border:1px solid #56616b}button{background:#ff9d61;color:#1a1a1a;border:0;cursor:pointer}p{color:#b6bfc8;line-height:1.5}a{color:#ffad7b}#message{white-space:pre-wrap}</style></head><body><main><h1>Connect to Maestro</h1><p>Enter the one-time browser code from Pinokio's Agent Access terminal, or from another signed-in Maestro browser.</p><form method="post" action="/api/v1/agent-access/owner-login"><label for="code">Browser code</label><input id="code" name="code" autocomplete="off" required autofocus><button>Sign in</button></form><button id="request">Show a code in Pinokio</button><p id="message" role="status"></p>""",
        f'<p><a href="{setup_url}">Open this computer\'s secure setup</a></p>',
        """</main><script>
document.getElementById('request').onclick=async()=>{let e=document.getElementById('message');try{let r=await fetch('/api/v1/agent-access/request-console-code',{method:'POST'});let j=await r.json();e.textContent=j.message||j.detail?.message||j.detail||'Check the Pinokio terminal.'}catch{e.textContent='Could not request a code. Open Agent Access in Pinokio.'}};
</script></body></html>""",
    ]
    return "".join(parts)

class AccessBoundary:
    def __init__(
        self,
        app,
        *,
        runtime,
        journal=None,
        app_root: Path | None = None,
        always_protect: bool = False,
        asset_resolver=None,
        additional_media_roots=None,
        dynamic_scope_resolver=None,
        validate_controller_origin: bool = False,
    ):
        self.app, self.runtime, self.journal = app, runtime, journal
        self.app_root = Path(app_root).resolve() if app_root else None
        self.always_protect = bool(always_protect)
        self.asset_resolver = asset_resolver
        self.additional_media_roots = additional_media_roots
        self.dynamic_scope_resolver = dynamic_scope_resolver
        self.validate_controller_origin = bool(validate_controller_origin)
        self._multipart_slots = threading.BoundedSemaphore(MAX_CONCURRENT_MULTIPART)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        path = scope.get("path", "/")
        method = scope.get("method", "GET").upper()
        headers = {k.lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
        allow_insecure_login_transport = (
            scope.get("scheme") == "http"
            and (
                (method == "GET" and path == "/agent-access/login")
                # The route rejects remote HTTP before reading the code body;
                # it returns the secure controller URL to the browser.
                or (method == "POST" and path == "/api/v1/agent-access/owner-login")
            )
        )
        if self.validate_controller_origin and self.runtime.origin_for_scope(
            scope, allow_insecure_remote=allow_insecure_login_transport,
        ) is None:
            response = JSONResponse(
                {"detail": "Use the canonical Maestro endpoint, loopback, or a verified certificate IP alias on the configured port"},
                status_code=421,
                headers={"Cache-Control": "no-store"},
            )
            return await response(scope, receive, send)
        if path == "/mcp" and not self.runtime.auth.enabled:
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 4404})
            response = JSONResponse({"detail": "Not Found"}, status_code=404, headers={"Cache-Control": "no-store"})
            return await response(scope, receive, send)
        configured = self.always_protect or self.runtime.security_required
        if configured and path.startswith(("/uploads/", "/api/v1/uploads/")):
            original_send = send

            async def protect_uploaded_content(message):
                if message["type"] == "http.response.start":
                    response_headers = list(message.get("headers", []))
                    content_type = {key.lower(): value for key, value in response_headers}.get(b"content-type", b"").split(b";", 1)[0].lower()
                    if content_type in {b"text/html", b"application/xhtml+xml", b"image/svg+xml", b"application/xml", b"text/xml", b"text/javascript", b"application/javascript"}:
                        protected = {b"content-security-policy", b"content-disposition", b"x-content-type-options"}
                        response_headers = [(key, value) for key, value in response_headers if key.lower() not in protected]
                        response_headers.extend([(b"content-security-policy", b"sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:"), (b"content-disposition", b"attachment"), (b"x-content-type-options", b"nosniff")])
                        message = {**message, "headers": response_headers}
                await original_send(message)

            send = protect_uploaded_content
        principal = token = None
        try:
            bearer = headers.get(b"authorization", "")
            if bearer:
                if not bearer.startswith("Bearer "):
                    raise AuthError(401, "Use a Bearer access token")
                token = bearer[7:]
            else:
                request = Request({**scope, "type": "http"})
                token = request.cookies.get(cookie_name(self.runtime))
            if token:
                try:
                    principal = self.runtime.auth.authenticate(token)
                except AuthError:
                    if path not in PUBLIC:
                        raise
                    token = None
            if path not in PUBLIC and configured:
                if principal is None:
                    raise AuthError(401, "Sign in to Maestro or pair this agent")
                if not secure_transport(scope):
                    raise AuthError(426, "Use the secure Maestro connection for authenticated access")
                if headers.get(b"origin") and not same_origin(scope):
                    raise AuthError(403, "This request origin is not allowed")
                if principal.owner and method in MUTATIONS and not bearer and not same_origin(scope):
                    raise AuthError(403, "A same-origin browser request is required")
                if not principal.owner:
                    if path.startswith(OWNER_PREFIXES) or path.startswith("/classic"):
                        raise AuthError(403, "Only the owner can manage access or developer diagnostics")
                    needed = "observe" if path == "/mcp" else self._required_scope(principal, method, path)
                    if needed not in principal.scopes:
                        raise AuthError(403, f"This operation requires {needed} permission")
                    if path not in {"/mcp", "/openapi.json"} and not path.startswith(("/api/v1/", "/uploads/")):
                        raise AuthError(403, "Use the documented Maestro API for this operation")
            # An off server never becomes an unpaired MCP endpoint. Once an
            # owner exists, disabling MCP retains the browser/API auth boundary.
            if path == "/mcp" and (not self.runtime.auth.enabled or principal is None):
                raise AuthError(401 if self.runtime.auth.enabled else 404, "Agent Access is disabled or this client is not paired")
            if not principal or path in PUBLIC or not path.startswith("/api/v1/"):
                ptoken = principal_context.set(principal)
                ctoken = credential_context.set(token)
                otoken = public_origin_context.set(self.runtime.origin_for_scope(scope) if self.validate_controller_origin else None)
                try:
                    return await self.app(scope, receive, send)
                finally:
                    principal_context.reset(ptoken)
                    credential_context.reset(ctoken)
                    public_origin_context.reset(otoken)
            return await self._agent_http(scope, receive, send, principal, token, headers)
        except (AuthError, JournalError, ValueError) as error:
            status = getattr(error, "status", 400)
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 4401 if status == 401 else 4403})
            request_origin = self.runtime.origin_for_scope(scope) if self.validate_controller_origin else None
            login_status = self.runtime.public_status()
            origin = request_origin or browser_login_origin(login_status, scope) or self.runtime.origin
            detail = {"code": "maestro_auth_required" if status == 401 else "maestro_access_denied", "message": str(error), "login_url": origin}
            if status == 401 and method == "GET" and "text/html" in headers.get(b"accept", "") and not path.startswith("/api/"):
                response = HTMLResponse(login_document({**login_status, "controller_url": origin}, secure=secure_transport(scope)), status_code=401, headers={"Cache-Control": "no-store"})
            else:
                metadata_path = "/.well-known/oauth-protected-resource/mcp" if path == "/mcp" else "/.well-known/oauth-protected-resource"
                challenge = f'Bearer resource_metadata="{origin}{metadata_path}"'
                if path == "/mcp":
                    challenge += ' scope="observe"'
                response = JSONResponse({"detail": detail}, status_code=status, headers={
                    "Cache-Control": "no-store",
                    **({"WWW-Authenticate": challenge} if status == 401 else {}),
                })
            return await response(scope, receive, send)

    def _required_scope(self, principal, method: str, path: str) -> str:
        dynamic_transfer_path = (
            method in {"GET", "HEAD"}
            and re.fullmatch(r"/api/v1/agent-access/transfers/[^/]+(?:/chunk)?", path) is not None
        )
        if self.dynamic_scope_resolver is not None and dynamic_transfer_path:
            try:
                dynamic_scope = self.dynamic_scope_resolver(principal, method, path)
            except Exception as error:
                raise AuthError(403, "The transfer permission could not be verified") from error
            if dynamic_scope is not None:
                if not isinstance(dynamic_scope, str) or dynamic_scope not in {"observe", "generate", "manage", "admin"}:
                    raise AuthError(403, "The transfer permission could not be verified")
                return dynamic_scope
        return required_scope(method, path)

    async def _spool_multipart(self, scope, receive):
        raw_headers = scope.get("headers", [])
        if len(raw_headers) > MAX_MULTIPART_HEADER_COUNT or sum(len(k) + len(v) for k, v in raw_headers) > MAX_MULTIPART_HEADERS:
            raise AuthError(431, "Multipart request headers exceed the configured limit")
        content_types = [v for k, v in raw_headers if k.lower() == b"content-type"]
        content_lengths = [v for k, v in raw_headers if k.lower() == b"content-length"]
        transfer_encodings = [v for k, v in raw_headers if k.lower() == b"transfer-encoding"]
        content_encodings = [v for k, v in raw_headers if k.lower() == b"content-encoding"]
        if (
            len(content_types) != 1
            or len(content_lengths) > 1
            or len(transfer_encodings) > 1
            or len(content_encodings) > 1
            or (content_lengths and transfer_encodings)
        ):
            raise AuthError(400, "Multipart request headers are ambiguous")
        try:
            content_type = content_types[0].decode("latin1")
            if content_lengths and not re.fullmatch(rb"[0-9]+", content_lengths[0]):
                raise ValueError("invalid Content-Length")
            declared_size = int(content_lengths[0].decode("ascii")) if content_lengths else None
        except (UnicodeDecodeError, ValueError):
            raise AuthError(400, "Multipart request headers are invalid") from None
        if declared_size is not None and (declared_size < 0 or declared_size > MAX_MULTIPART_BODY):
            raise AuthError(413, "Multipart upload exceeds the 500 MiB limit")
        normalized_encodings = [value.decode("latin1").strip().casefold() for value in content_encodings]
        if any(value not in {"", "identity"} for value in normalized_encodings):
            raise AuthError(415, "Compressed multipart request bodies are not supported")

        directory = getattr(self.runtime, "directory", None)
        if directory is None:
            raise AuthError(500, "Multipart upload storage is unavailable")
        spool_directory = Path(directory)
        try:
            spool_directory.mkdir(parents=True, exist_ok=True)
            try:
                spool_directory.chmod(0o700)
            except OSError:
                pass
            descriptor, name = tempfile.mkstemp(prefix=".agent-access-upload-", dir=spool_directory)
        except OSError as error:
            raise AuthError(507, "Could not create private upload storage") from error

        spool_path = Path(name)
        hasher = hashlib.sha256()
        total = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                while True:
                    message = await receive()
                    if message.get("type") == "http.disconnect":
                        raise AuthError(400, "Upload disconnected before completion")
                    if message.get("type") != "http.request":
                        raise AuthError(400, "Invalid multipart request stream")
                    chunk = message.get("body", b"")
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise AuthError(400, "Invalid multipart request body")
                    total += len(chunk)
                    if total > MAX_MULTIPART_BODY:
                        raise AuthError(413, "Multipart upload exceeds the 500 MiB limit")
                    if chunk:
                        handle.write(chunk)
                        hasher.update(chunk)
                    if not message.get("more_body", False):
                        break
                handle.flush()
            if declared_size is not None and total != declared_size:
                raise AuthError(400, "Multipart Content-Length does not match the received body")
        except BaseException:
            spool_path.unlink(missing_ok=True)
            raise
        return spool_path, total, hasher.hexdigest(), content_type

    def _trusted_additional_media_roots(self):
        if self.additional_media_roots is None:
            return ()
        try:
            configured_roots = self.additional_media_roots()
        except Exception as error:
            raise AuthError(403, "Trusted media roots could not be verified") from error
        return _canonical_additional_roots(configured_roots)

    async def _validate_multipart(
        self,
        scope,
        spool_path: Path,
        app_root: Path | None,
        additional_roots=(),
    ) -> None:
        parser_receive = _file_receive(spool_path)
        form = None
        try:
            request = Request(scope, parser_receive)
            form = await request.form(
                max_files=MAX_MULTIPART_FILES,
                max_fields=MAX_MULTIPART_FIELDS,
                max_part_size=MAX_MULTIPART_FIELD_BYTES,
            )
            for name, value in form.multi_items():
                field_name = _normalize_key(name)
                if isinstance(value, UploadFile):
                    filename = value.filename or ""
                    decoded_filename = filename
                    for _ in range(2):
                        decoded_filename = unquote(decoded_filename)
                    if (
                        not filename
                        or decoded_filename in {".", ".."}
                        or "\x00" in decoded_filename
                        or "/" in decoded_filename
                        or "\\" in decoded_filename
                        or _WINDOWS_ABSOLUTE.match(decoded_filename)
                        or len(filename.encode("utf-8", "replace")) > 1024
                    ):
                        raise AuthError(400, "Multipart file names must be simple file names")
                    continue
                if not isinstance(value, str):
                    raise AuthError(400, "Multipart form fields must be text or files")
                _reject_redacted_credentials({field_name: value})
                if app_root is None and not additional_roots:
                    continue
                validate_media_paths(value, app_root, key=field_name, additional_roots=additional_roots)
                stripped = value.lstrip()
                if stripped.startswith(("{", "[")):
                    try:
                        structured = json.loads(value)
                    except (ValueError, RecursionError):
                        continue
                    _reject_redacted_credentials(structured)
                    validate_media_paths(structured, app_root, additional_roots=additional_roots)
        except AuthError:
            raise
        except Exception as error:
            raise AuthError(400, "Multipart form data is invalid or exceeds its field limits") from error
        finally:
            try:
                if form is not None:
                    await form.close()
            finally:
                parser_receive.close_spool()

    async def _agent_http(self, scope, receive, send, principal, token, headers):
        path, method = scope["path"], scope["method"].upper()
        for _ in range(2):
            if any(part == ".." for part in path.replace("\\", "/").split("/")):
                raise AuthError(400, "Invalid operation path")
            path = unquote(path)
        payload = None
        reservation = None
        request_body = None
        if (self.app_root is not None or self.additional_media_roots is not None) and not principal.owner and scope.get("query_string"):
            from urllib.parse import parse_qs
            validate_media_paths(
                parse_qs(scope["query_string"].decode("utf-8")),
                self.app_root,
                additional_roots=self._trusted_additional_media_roots(),
            )
        request_content_type = headers.get(b"content-type", "")
        media_type = request_content_type.split(";", 1)[0].strip().casefold()
        is_json = "json" in media_type
        is_multipart = media_type == "multipart/form-data"
        multipart_spool = None
        multipart_slot = False
        multipart_receiver = None
        context_tokens = None
        try:
            if is_json and method in MUTATIONS:
                body = bytearray()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        raise AuthError(400, "Request disconnected before submission")
                    chunk = message.get("body", b"")
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise AuthError(400, "Invalid JSON request body")
                    body.extend(chunk)
                    if len(body) > MAX_JSON:
                        raise AuthError(413, "The JSON request is too large; transfer media through the asset API")
                    if not message.get("more_body", False):
                        break
                try:
                    payload = json.loads(body or b"{}")
                except (ValueError, RecursionError) as error:
                    raise AuthError(400, "Request body must contain valid JSON") from error
                request_body = bytes(body)
                resolved_payload = payload
                if not principal.owner:
                    _reject_redacted_credentials(payload)
                if self.asset_resolver is not None:
                    try:
                        resolved_payload = self.asset_resolver(principal.id, deepcopy(payload))
                    except Exception as error:
                        status = getattr(error, "status", None)
                        message = getattr(error, "message", None)
                        if isinstance(status, int) and 400 <= status <= 599 and isinstance(message, str):
                            raise AuthError(status, message) from error
                        raise AuthError(400, "A registered media asset could not be resolved") from error
                    if not principal.owner:
                        _reject_redacted_credentials(resolved_payload)
                    try:
                        request_body = json.dumps(
                            resolved_payload,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
                        raise AuthError(400, "Resolved request payload must be JSON-safe") from error
                    if len(request_body) > MAX_JSON:
                        raise AuthError(413, "Resolved request payload exceeds the configured size limit")
                    request_headers = [
                        (key, value)
                        for key, value in scope.get("headers", [])
                        if key.lower() not in {b"content-length", b"transfer-encoding"}
                    ]
                    request_headers.append((b"content-length", str(len(request_body)).encode("ascii")))
                    scope = {**scope, "headers": request_headers}
                if (self.app_root is not None or self.additional_media_roots is not None) and not principal.owner:
                    validate_media_paths(
                        resolved_payload,
                        self.app_root,
                        additional_roots=self._trusted_additional_media_roots(),
                    )
                original_receive = receive
                sent_body = False

                async def replay_json_receive():
                    nonlocal sent_body
                    if not sent_body:
                        sent_body = True
                        return {"type": "http.request", "body": request_body or b"", "more_body": False}
                    return await original_receive()

                receive = replay_json_receive

            if is_multipart and method in MUTATIONS and not principal.owner and self.journal is not None:
                key = headers.get(b"idempotency-key")
                if not key:
                    raise AuthError(400, "Mutating agent multipart requests require a stable Idempotency-Key")
                if not self._multipart_slots.acquire(blocking=False):
                    raise AuthError(429, "Too many multipart uploads are already being processed")
                multipart_slot = True
                multipart_spool, body_size, body_sha256, content_type = await self._spool_multipart(scope, receive)
                await self._validate_multipart(
                    scope,
                    multipart_spool,
                    self.app_root,
                    additional_roots=self._trusted_additional_media_roots(),
                )
                normalized_headers = [
                    (key_name, value)
                    for key_name, value in scope.get("headers", [])
                    if key_name.lower() not in {b"content-length", b"transfer-encoding"}
                ]
                normalized_headers.append((b"content-length", str(body_size).encode("ascii")))
                scope = {**scope, "headers": normalized_headers}
                multipart_receiver = _file_receive(multipart_spool, upstream=receive)
                receive = multipart_receiver
                reservation = self.journal.reserve(
                    principal.id,
                    key,
                    method,
                    path,
                    {
                        "wire_body_sha256": body_sha256,
                        "wire_body_bytes": body_size,
                        "content_type": content_type,
                    },
                    query=scope.get("query_string", b"").decode("utf-8"),
                )
                if reservation["state"] == "complete":
                    from starlette.responses import Response
                    return await Response(
                        reservation["body"],
                        status_code=reservation["status_code"],
                        headers=dict(reservation["headers"]),
                    )(scope, receive, send)
                if reservation["state"] != "new":
                    return await JSONResponse({
                        "detail": {
                            "code": "submission_" + reservation["state"],
                            "reservation_id": reservation["reservation_id"],
                            "message": "Inspect this existing submission; do not submit it with a new key",
                        }
                    }, status_code=409)(scope, receive, send)

            if method in MUTATIONS and self.journal is not None and reservation is None:
                key = headers.get(b"idempotency-key")
                if isinstance(payload, dict) and not principal.owner:
                    key = key or payload.get("client_submission_id") or payload.get("_client_submission_id")
                if not key and not principal.owner:
                    raise AuthError(400, "Mutating agent requests require a stable Idempotency-Key")
                if key:
                    if not is_json and method not in {"DELETE"}:
                        if not principal.owner and not path.startswith("/api/v1/agent-access/assets/"):
                            raise AuthError(400, "Use the multipart API or asset upload API for binary transfers")
                    else:
                        reservation = self.journal.reserve(
                            principal.id,
                            key,
                            method,
                            path,
                            payload,
                            query=scope.get("query_string", b"").decode("utf-8"),
                        )
                        if reservation["state"] == "complete":
                            from starlette.responses import Response
                            return await Response(
                                reservation["body"],
                                status_code=reservation["status_code"],
                                headers=dict(reservation["headers"]),
                            )(scope, receive, send)
                        if reservation["state"] != "new":
                            return await JSONResponse({
                                "detail": {
                                    "code": "submission_" + reservation["state"],
                                    "reservation_id": reservation["reservation_id"],
                                    "message": "Inspect this existing submission; do not submit it with a new key",
                                }
                            }, status_code=409)(scope, receive, send)

            # Keep JSON visible to the boundary even if the client requested gzip.
            if not principal.owner or reservation:
                scope = {**scope, "headers": [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"accept-encoding"]}
            response_start = None
            chunks = []
            size = 0
            buffered = False
            binary_receipt = False
            response_complete = False
            binary_receipt_body = None
            binary_receipt_status = None
            digest = hashlib.sha256()

            async def capture(message):
                nonlocal response_start, buffered, size, binary_receipt, response_complete
                nonlocal binary_receipt_body, binary_receipt_status
                if message["type"] == "http.response.start":
                    response_start = message
                    response_headers = dict(message.get("headers", []))
                    content = response_headers.get(b"content-type", b"").lower()
                    binary_receipt = bool(reservation) and b"json" not in content and (
                        b"content-disposition" in response_headers or bool(content)
                    )
                    buffered = (bool(reservation) and not binary_receipt) or (not principal.owner and b"json" in content)
                    if not buffered:
                        await send(message)
                elif message["type"] == "http.response.body" and binary_receipt:
                    body_part = message.get("body", b"")
                    size += len(body_part)
                    digest.update(body_part)
                    await send(message)
                    if not message.get("more_body", False):
                        binary_receipt_body = json.dumps({
                            "kind": "binary_receipt",
                            "delivery_verified": False,
                            "bytes": size,
                            "sha256": digest.hexdigest(),
                            "content_type": dict(response_start.get("headers", [])).get(b"content-type", b"").decode("latin1"),
                            "message": "The native binary response was delivered once; it cannot be regenerated with this idempotency key.",
                        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                        binary_receipt_status = response_start["status"]
                        response_complete = True
                elif message["type"] == "http.response.body" and buffered:
                    body_part = message.get("body", b"")
                    size += len(body_part)
                    if size > MAX_JSON:
                        raise AuthError(413, "Response is too large; use a paginated operation or asset download")
                    chunks.append(body_part)
                    if not message.get("more_body", False):
                        raw = b"".join(chunks)
                        content = dict(response_start.get("headers", [])).get(b"content-type", b"")
                        if not principal.owner and b"json" in content.lower() and raw:
                            raw = json.dumps(redact(json.loads(raw)), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                        safe_headers = [(k, v) for k, v in response_start.get("headers", []) if k.lower() not in {b"content-length", b"content-encoding"}]
                        safe_headers.append((b"content-length", str(len(raw)).encode("ascii")))
                        if reservation:
                            self.journal.complete(
                                reservation["reservation_id"],
                                response_start["status"],
                                raw,
                                [(k.decode("latin1"), v.decode("latin1")) for k, v in safe_headers],
                            )
                        await send({**response_start, "headers": safe_headers})
                        await send({"type": "http.response.body", "body": raw})
                        response_complete = True
                else:
                    await send(message)
                    if message["type"] == "http.response.body" and not message.get("more_body", False):
                        response_complete = True
            context_tokens = (
                principal_context.set(principal),
                credential_context.set(token),
                submission_context.set(reservation["reservation_id"] if reservation else None),
            )
            origin_token = public_origin_context.set(self.runtime.origin_for_scope(scope) if self.validate_controller_origin else None)
            try:
                await self.app(scope, receive, capture)
                if reservation:
                    if binary_receipt_body is not None and response_complete:
                        self.journal.complete(
                            reservation["reservation_id"],
                            binary_receipt_status,
                            binary_receipt_body,
                            {"content-type": "application/json", "cache-control": "no-store"},
                        )
                    elif not response_complete:
                        self.journal.fail_uncertain(
                            reservation["reservation_id"],
                            "The native operation ended before a complete response was recorded",
                        )
                route = scope.get("route")
                route_template = getattr(route, "path", None)
                if not isinstance(route_template, str) or not route_template.startswith("/"):
                    route_template = "API request" if scope["path"].startswith("/api/v1/") else "media request"
                self.runtime.audit(principal.id, f"{method} {route_template}", str((response_start or {}).get("status", "unknown")))
            except BaseException:
                if reservation:
                    self.journal.fail_uncertain(reservation["reservation_id"], "The connection ended before a durable response was recorded")
                raise
        finally:
            if context_tokens is not None:
                principal_context.reset(context_tokens[0])
                credential_context.reset(context_tokens[1])
                submission_context.reset(context_tokens[2])
                public_origin_context.reset(origin_token)
            if multipart_receiver is not None:
                multipart_receiver.close_spool()
            if multipart_spool is not None:
                multipart_spool.unlink(missing_ok=True)
            if multipart_slot:
                self._multipart_slots.release()
