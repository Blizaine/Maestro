"""Owner onboarding and management routes shared by Maestro and its controller."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import inspect
import json
from pathlib import Path
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .auth import LOCAL_DEVICE_PRINCIPAL, AuthError
from .boundary import (
    browser_login_origin,
    cookie_name,
    cookie_names,
    local_approval_cookie_name,
    local_approval_origin,
    local_browser_approval_eligible,
    login_document,
    same_origin,
    secure_transport,
)
from .context import credential_context, principal_context


def owner_required():
    principal = principal_context.get()
    if principal is None or not principal.owner:
        raise HTTPException(403, "Only a signed-in Maestro owner can manage Agent Access")
    return principal


def scope_required(scope: str):
    principal = principal_context.get()
    if principal is None:
        raise HTTPException(401, "Pair this agent before using Maestro")
    if not principal.owner and scope not in principal.scopes:
        raise HTTPException(403, f"This operation requires {scope} permission")
    return principal


async def json_body(request: Request) -> dict:
    body = await request.body()
    if len(body) > 1024 * 1024:
        raise HTTPException(413, "Request is too large")
    import json
    try:
        value = json.loads(body or b"{}")
    except (ValueError, UnicodeError) as error:
        raise HTTPException(400, "Use a JSON object") from error
    if not isinstance(value, dict):
        raise HTTPException(400, "Use a JSON object")
    return value


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON member")
        value[key] = item
    return value


async def local_approval_json_body(request: Request) -> dict:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Use a JSON approval request")
    raw = await request.body()
    if len(raw) > 4096:
        raise HTTPException(413, "Approval request is too large")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (ValueError, UnicodeError):
        raise HTTPException(400, "Use a JSON approval object") from None
    if not isinstance(value, dict):
        raise HTTPException(400, "Use a JSON approval object")
    return value


class RateLimiter:
    """Bound unauthenticated work without persisting caller-supplied addresses."""
    def __init__(self, maximum: int = 30, seconds: float = 60):
        self.maximum, self.seconds = maximum, seconds
        self._events = deque()

    def check(self):
        now = time.monotonic()
        while self._events and self._events[0] <= now - self.seconds:
            self._events.popleft()
        if len(self._events) >= self.maximum:
            raise HTTPException(429, "Too many attempts; wait a moment and try again")
        self._events.append(now)


def login_return(value):
    if isinstance(value, str) and value.startswith("/oauth/authorize?") and len(value) <= 8192 and not any(c in value for c in ("\\", "\r", "\n", "\x00")):
        return value
    return "/"


def _login_failure(
    runtime,
    request: Request,
    message: str,
    status: int,
    *,
    next_path=None,
    local_device_access: bool = False,
):
    public_status = runtime.public_status()
    origin = browser_login_origin(public_status, request.scope)
    document = login_document(
        {
            **public_status,
            "controller_url": origin,
            "local_approval_available": local_browser_approval_eligible(
                request.scope, require_same_origin=True,
            ),
        },
        secure=secure_transport(request.scope),
        notice=message,
        local_device_access=local_device_access,
    )
    if "</form>" in document:
        target = html.escape(login_return(next_path), quote=True)
        document = document.replace(
            "</form>", f'<input type="hidden" name="next" value="{target}"></form>', 1,
        )
    return HTMLResponse(document, status_code=status, headers={"Cache-Control": "no-store"})


def _http_error_message(error: HTTPException) -> str:
    detail = error.detail
    if isinstance(detail, dict):
        return str(detail.get("message") or "Sign-in is temporarily unavailable")
    return str(detail)


def _normalize_owner_code(value) -> str:
    """Accept a raw console code or its exact printed label/line wrapping."""
    if not isinstance(value, str) or len(value) > 512:
        return ""
    candidate = value.strip()
    prefix = "[Maestro Agent Access] Browser code:"
    if candidate[:len(prefix)].casefold() == prefix.casefold():
        candidate = candidate[len(prefix):]
    # Owner codes use URL-safe Base64 and contain no whitespace. Removing only
    # whitespace tolerates terminal line wrapping without weakening matching.
    return "".join(candidate.split())


def require_updated_backend(runtime, *, min_auth_boundary_version: int = 1):
    # An older model process cannot acquire agent credential/ownership checks
    # retroactively. Require that middleware even though ordinary native
    # browser/API access intentionally retains the existing LAN workflow.
    backend = runtime.backend()
    version = backend.get("auth_boundary_version")
    if type(version) is not int or version < min_auth_boundary_version:
        raise HTTPException(409, "Start the updated Maestro once before setting up Agent Access. An older running instance must be restarted when idle.")
    return backend


LOCAL_APPROVAL_TTL = 120
_MAX_LOCAL_APPROVAL_CHALLENGES = 128
_LOCAL_DEVICE_PROOF = re.compile(r"^[0-9a-fA-F]{64}$")


def _challenge_digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("ascii")).digest()


class LocalApprovalChallenges:
    """Small, in-memory registry for origin- and proof-bound local approvals."""

    def __init__(self, *, clock=time.monotonic, ttl: int = LOCAL_APPROVAL_TTL):
        self.clock, self.ttl = clock, int(ttl)
        self._entries = {}
        self._lock = threading.Lock()

    def issue(self, origin: str, proof: str) -> str:
        now = self.clock()
        nonce = secrets.token_urlsafe(32)
        key = _challenge_digest(nonce)
        with self._lock:
            self._purge(now)
            if len(self._entries) >= _MAX_LOCAL_APPROVAL_CHALLENGES:
                raise HTTPException(429, "Too many local approval attempts; wait a moment and try again")
            self._entries[key] = (origin, _challenge_digest(proof), now + self.ttl)
        return nonce

    def consume(self, nonce: str, proof: str, origin: str) -> bool:
        if not isinstance(nonce, str) or not nonce or len(nonce) > 128:
            return False
        try:
            key = _challenge_digest(nonce)
            proof_hash = _challenge_digest(proof)
        except (UnicodeEncodeError, AttributeError):
            return False
        now = self.clock()
        with self._lock:
            self._purge(now)
            saved = self._entries.get(key)
            if saved is None:
                return False
            saved_origin, saved_proof_hash, _expires = saved
            if not hmac.compare_digest(saved_origin, origin) or not hmac.compare_digest(saved_proof_hash, proof_hash):
                return False
            del self._entries[key]
            return True

    def _purge(self, now: float) -> None:
        expired = [key for key, (_origin, _cookie_hash, expires_at) in self._entries.items() if expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)


def _local_approval_cookie(request: Request, name: str) -> str | None:
    values = [value for key, value in request.scope.get("headers", []) if key.lower() == b"cookie"]
    if len(values) != 1:
        return None
    try:
        header = values[0].decode("ascii")
    except (UnicodeDecodeError, AttributeError):
        return None
    found = []
    for part in header.split(";"):
        key, separator, value = part.strip().partition("=")
        if separator and key == name:
            found.append(value)
    return found[0] if len(found) == 1 and found[0] else None


def management_router(
    runtime,
    *,
    request_console_code,
    status_extra=None,
    local_approval_clock=time.monotonic,
    local_device_access: bool = False,
) -> APIRouter:
    router = APIRouter()
    login_limit, code_limit = RateLimiter(30), RateLimiter(3)
    local_challenge_limit, local_approve_limit = RateLimiter(10), RateLimiter(10)
    local_challenges = LocalApprovalChallenges(clock=local_approval_clock)
    tailscale_manager = None

    @router.get("/api/v1/remote-access/pinokio/status")
    def pinokio_browser_status():
        return runtime.refresh_pinokio_remote()

    @router.get("/api/v1/agent-access/pinokio")
    def pinokio_mcp_status():
        owner_required()
        runtime.refresh_pinokio_remote()
        return runtime.pinokio_remote_status()

    @router.post("/api/v1/agent-access/pinokio")
    async def pinokio_mcp_settings(request: Request):
        owner = owner_required()
        body = await json_body(request)
        if set(body) != {"enabled"} or type(body["enabled"]) is not bool:
            raise HTTPException(400, "Provide only enabled as a boolean")
        if body["enabled"]:
            if not runtime.auth.enabled:
                raise HTTPException(409, "Enable Remote MCP first")
            backend = runtime.backend()
            if backend.get("auth_boundary_version", 0) < 3:
                raise HTTPException(409, "Restart Maestro with the updated Pinokio Remote support first")
            evidence = await asyncio.to_thread(runtime.refresh_pinokio_remote)
            from .state import _pinokio_web_origin
            if _pinokio_web_origin(evidence) is None:
                raise HTTPException(409, "Enable Pinokio Remote and wait for Maestro's published address")
        runtime.configure_pinokio_remote(body["enabled"])
        runtime.audit(owner.id, "agent_access_pinokio_remote", "enabled" if body["enabled"] else "disabled")
        return runtime.pinokio_remote_status()

    def certificate_manager():
        nonlocal tailscale_manager
        if tailscale_manager is None:
            from .tailscale import TailscaleCertificateManager
            tailscale_manager = TailscaleCertificateManager(runtime.directory)
        return tailscale_manager

    def tailscale_result(manager, current=None):
        current = current if current is not None else manager.status()
        result = runtime.tailscale_status()
        result.update(
            installed=result["installed"] if current.installed is None else bool(current.installed),
            connected=result["connected"] if current.connected is None else bool(current.connected),
            hostname=runtime.tailscale_hostname or current.hostname,
            error=current.error or result.get("error"),
        )
        matching = current.hostname == runtime.tailscale_hostname
        result["certificate_ready"] = bool(result["certificate_ready"] and matching and current.certificate_ready)
        result["ready"] = bool(result["ready"] and result["connected"] and result["certificate_ready"])
        if result["enabled"] and current.connected and not matching:
            result["error"] = "Tailscale identity changed; disable and enable this connection to verify the new hostname"
        if not result["ready"]:
            result["mcp_url"] = None
        return result

    @router.get("/api/v1/agent-access/tailscale")
    async def tailscale_status():
        owner_required()
        manager = certificate_manager()
        current = await asyncio.to_thread(manager.detect)
        return JSONResponse(tailscale_result(manager, current), headers={"Cache-Control": "no-store"})

    @router.post("/api/v1/agent-access/tailscale")
    async def tailscale_settings(request: Request):
        owner = owner_required()
        body = await json_body(request)
        if set(body) != {"enabled"} or type(body["enabled"]) is not bool:
            raise HTTPException(400, "Use one boolean enabled setting")
        if body["enabled"]:
            require_updated_backend(runtime, min_auth_boundary_version=2)
            if not runtime.auth.enabled:
                raise HTTPException(409, "Enable Agent Access before connecting through Tailscale")
            manager = certificate_manager()
            try:
                material = await asyncio.to_thread(manager.ensure_certificate)
                runtime.configure_tailscale(enabled=True, hostname=material.hostname)
            except (OSError, RuntimeError, ValueError) as error:
                # The certificate manager exposes bounded, sanitized messages.
                raise HTTPException(409, str(error)) from None
        else:
            runtime.configure_tailscale(enabled=False)
            manager = certificate_manager()
        runtime.audit(owner.id, "agent_access_tailscale", "enabled" if body["enabled"] else "disabled")
        current = await asyncio.to_thread(manager.detect)
        return JSONResponse(tailscale_result(manager, current), headers={"Cache-Control": "no-store"})

    @router.get("/api/v1/agent-access/certificate")
    async def public_certificate():
        path = runtime.directory / "tls" / "server-cert.pem"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
            raise HTTPException(503, "Start Agent Access in Pinokio to create its public certificate")
        content = path.read_bytes()
        if not content.startswith(b"-----BEGIN CERTIFICATE-----") or b"PRIVATE KEY" in content:
            raise HTTPException(503, "The public certificate is unavailable")
        return Response(content, media_type="application/x-pem-file", headers={"Content-Disposition": 'attachment; filename="maestro-agent-access.pem"', "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @router.get("/api/v1/agent-access/status")
    async def status(request: Request):
        principal = principal_context.get()
        result = runtime.public_status()
        if status_extra:
            extra = status_extra()
            if inspect.isawaitable(extra):
                extra = await extra
            result.update(extra)
        result.update(
            authenticated=principal is not None,
            owner=bool(principal and principal.owner),
            local_approval_available=local_browser_approval_eligible(request.scope),
        )
        if principal is not None and principal.id == LOCAL_DEVICE_PRINCIPAL.id:
            result["auth_method"] = "local_device"
        if principal:
            result["principal"] = asdict(principal)
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @router.get("/agent-access/login")
    async def login_page(request: Request):
        status = runtime.public_status()
        document = login_document({
            **status,
            "controller_url": browser_login_origin(status, request.scope),
            "local_approval_available": local_browser_approval_eligible(request.scope),
        }, secure=secure_transport(request.scope), local_device_access=local_device_access)
        target = html.escape(login_return(request.query_params.get("next")), quote=True)
        document = document.replace("</form>", f'<input type="hidden" name="next" value="{target}"></form>', 1)
        return HTMLResponse(document, headers={"Cache-Control": "no-store"})

    @router.post("/api/v1/agent-access/local-challenge")
    async def local_challenge(request: Request):
        if not local_browser_approval_eligible(request.scope, require_same_origin=True):
            raise HTTPException(403, "Local approval requires this browser on this computer")
        local_challenge_limit.check()
        origin = local_approval_origin(request.scope)
        if local_device_access:
            body = await local_approval_json_body(request)
            proof = body.get("proof")
            if set(body) != {"proof"} or not isinstance(proof, str) or not _LOCAL_DEVICE_PROOF.fullmatch(proof):
                raise HTTPException(400, "Use a fresh local approval proof")
            nonce = local_challenges.issue(origin, proof)
            return JSONResponse({"nonce": nonce, "expires_in": LOCAL_APPROVAL_TTL}, headers={"Cache-Control": "no-store"})
        cookie_name_for_challenge = local_approval_cookie_name(runtime, request.scope)
        cookie = secrets.token_urlsafe(32)
        nonce = local_challenges.issue(origin, cookie)
        response = JSONResponse({"nonce": nonce, "expires_in": LOCAL_APPROVAL_TTL}, headers={"Cache-Control": "no-store"})
        response.set_cookie(
            cookie_name_for_challenge,
            cookie,
            max_age=LOCAL_APPROVAL_TTL,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/api/v1/agent-access/local-approve",
        )
        return response

    @router.post("/api/v1/agent-access/local-approve")
    async def local_approve(request: Request):
        if not local_browser_approval_eligible(request.scope, require_same_origin=True):
            raise HTTPException(403, "Local approval requires this browser on this computer")
        local_approve_limit.check()
        body = await local_approval_json_body(request)
        expected_keys = {"nonce", "proof", "approve"} if local_device_access else {"nonce", "approve"}
        if set(body) != expected_keys or body.get("approve") is not True:
            raise HTTPException(400, "Explicit approval is required")
        if local_device_access and (
            not isinstance(body.get("proof"), str)
            or not _LOCAL_DEVICE_PROOF.fullmatch(body["proof"])
        ):
            raise HTTPException(400, "Use a fresh local approval proof")
        require_updated_backend(runtime, min_auth_boundary_version=2 if local_device_access else 1)
        origin = local_approval_origin(request.scope)
        if local_device_access:
            proof = body["proof"]
            if not local_challenges.consume(body.get("nonce"), proof, origin):
                raise HTTPException(401, "Local approval challenge is invalid, expired, or already used")
            runtime.mark_security_configured()
            runtime.auth.mark_owner_configured()
        else:
            challenge_cookie_name = local_approval_cookie_name(runtime, request.scope)
            challenge_cookie = _local_approval_cookie(request, challenge_cookie_name)
            if challenge_cookie is None or not local_challenges.consume(body.get("nonce"), challenge_cookie, origin):
                raise HTTPException(401, "Local approval challenge is invalid, expired, or already used")
            session = runtime.auth.issue_owner_session()
            runtime.mark_security_configured()
        response = JSONResponse({"ok": True}, headers={"Cache-Control": "no-store"})
        if not local_device_access:
            response.delete_cookie(
                challenge_cookie_name,
                path="/api/v1/agent-access/local-approve",
                secure=request.url.scheme == "https",
                httponly=True,
                samesite="strict",
            )
            response.set_cookie(
                cookie_name(runtime, request.url.scheme),
                session["access_token"],
                max_age=session["expires_in"],
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="lax",
                path="/",
            )
        runtime.audit("owner", "browser_login_local_approval", "success")
        return response

    @router.post("/api/v1/agent-access/request-console-code")
    async def console_code(request: Request):
        # This endpoint discloses no credential. The controller prints to its
        # own Pinokio terminal, outside any model worker's captured stdout.
        if request.headers.get("origin") and not same_origin(request.scope):
            raise HTTPException(403, "This request origin is not allowed")
        code_limit.check()
        try:
            result = request_console_code()
            if inspect.isawaitable(result):
                result = await result
        except (OSError, RuntimeError) as error:
            raise HTTPException(503, "Open Agent Access in Pinokio, then request a browser code") from error
        return {"message": "A one-time browser code is shown in Pinokio's Agent Access terminal.", "expires_in": 600}

    @router.post("/api/v1/agent-access/owner-login")
    async def owner_login(request: Request):
        is_form = "application/x-www-form-urlencoded" in request.headers.get("content-type", "")
        if not secure_transport(request.scope):
            if is_form:
                return _login_failure(
                    runtime,
                    request,
                    "Open the secure Maestro address before entering a browser code.",
                    426,
                    local_device_access=local_device_access,
                )
            raise HTTPException(426, {"message": "Open the secure Maestro address before entering a browser code", "login_url": browser_login_origin(runtime.public_status(), request.scope)})
        if not same_origin(request.scope):
            if is_form:
                return _login_failure(runtime, request, "Sign-in must come from this same Maestro address.", 403, local_device_access=local_device_access)
            raise HTTPException(403, "A same-origin sign-in is required")
        try:
            require_updated_backend(runtime)
            login_limit.check()
        except HTTPException as error:
            if is_form:
                return _login_failure(runtime, request, _http_error_message(error), error.status_code, local_device_access=local_device_access)
            raise
        if is_form:
            raw = await request.body()
            if len(raw) > 16384:
                return _login_failure(runtime, request, "Sign-in request is too large.", 413, local_device_access=local_device_access)
            try:
                body = {k: v[-1] for k, v in parse_qs(raw.decode("utf-8"), max_num_fields=8).items()}
            except (ValueError, UnicodeError):
                return _login_failure(runtime, request, "Invalid sign-in form. Request a fresh code and try again.", 400, local_device_access=local_device_access)
        else:
            body = await json_body(request)
        next_path = body.get("next")
        try:
            session = runtime.auth.redeem_owner_code(_normalize_owner_code(body.get("code", "")))
            runtime.mark_security_configured()
        except AuthError as error:
            if is_form:
                return _login_failure(
                    runtime,
                    request,
                    "This browser code is invalid, expired, or already used. Request a fresh code in Pinokio and try again.",
                    error.status,
                    next_path=next_path,
                    local_device_access=local_device_access,
                )
            raise HTTPException(error.status, str(error)) from error
        response = RedirectResponse(login_return(next_path), status_code=303) if is_form else JSONResponse({"ok": True})
        response.set_cookie(cookie_name(runtime, request.url.scheme), session["access_token"], max_age=session["expires_in"], httponly=True, secure=request.url.scheme == "https", samesite="lax", path="/")
        response.headers["Cache-Control"] = "no-store"
        runtime.audit("owner", "browser_login", "success")
        return response

    @router.post("/api/v1/agent-access/logout")
    async def logout(request: Request):
        owner_required()
        runtime.auth.logout(credential_context.get() or "")
        response = JSONResponse({"ok": True})
        for name in cookie_names(runtime, request.url.scheme):
            response.delete_cookie(name, path="/", secure=request.url.scheme == "https", httponly=True, samesite="lax")
        response.headers["Cache-Control"] = "no-store"
        return response

    @router.post("/api/v1/agent-access/browser-code")
    async def browser_code():
        owner_required()
        return JSONResponse({"code": runtime.auth.issue_owner_code(), "expires_in": 600}, headers={"Cache-Control": "no-store"})

    @router.post("/api/v1/agent-access/settings")
    @router.put("/api/v1/agent-access/settings")
    async def settings(request: Request):
        owner = owner_required()
        body = await json_body(request)
        if set(body) - {"enabled", "remote_enabled", "local_enabled", "label", "discovery_enabled"}:
            raise HTTPException(400, "Unsupported Agent Access setting")
        try:
            runtime.validate_settings(**body)
            remote_enabled = body.get("remote_enabled", body.get("enabled"))
            if remote_enabled is True:
                require_updated_backend(runtime)
            if remote_enabled is not None:
                runtime.auth.set_enabled(remote_enabled, owner)
            if "local_enabled" in body:
                runtime.auth.set_local_enabled(body["local_enabled"], owner)
            runtime.update(label=body.get("label"), discovery_enabled=body.get("discovery_enabled"))
        except (ValueError, AuthError) as error:
            raise HTTPException(getattr(error, "status", 400), str(error)) from error
        runtime.audit(owner.id, "agent_access_settings", "updated")
        return runtime.public_status()

    @router.get("/api/v1/agent-access/grants")
    async def grants():
        return {"grants": runtime.auth.list_grants(owner_required())}

    @router.get("/api/v1/agent-access/local-clients")
    async def local_clients():
        from .local_connector import local_launch_configuration

        owner = owner_required()
        app_root = runtime.app_root or Path(__file__).resolve().parents[2]
        clients = runtime.auth.list_local_clients(owner)
        for client in clients:
            if not client["revoked"]:
                client.update(local_launch_configuration(app_root, client["id"]))
        return {"clients": clients}

    @router.post("/api/v1/agent-access/local-clients")
    async def create_local_client(request: Request):
        from .local_connector import (
            LocalCredentialError,
            local_launch_configuration,
            write_local_credential,
        )

        owner = owner_required()
        body = await json_body(request)
        if set(body) != {"name", "scopes"}:
            raise HTTPException(400, "Use only name and scopes for a local client")
        item = None
        try:
            item, credential = runtime.auth.create_local_client(body["name"], body["scopes"], owner)
            write_local_credential(runtime.directory, item["id"], credential)
        except (AuthError, LocalCredentialError) as error:
            if item is not None:
                runtime.auth.revoke_grant(item["grant_id"], owner)
            raise HTTPException(getattr(error, "status", 400), str(error)) from error
        except OSError as error:
            if item is not None:
                runtime.auth.revoke_grant(item["grant_id"], owner)
            raise HTTPException(500, "Could not create a private local client credential") from error
        app_root = runtime.app_root or Path(__file__).resolve().parents[2]
        item.update(local_launch_configuration(app_root, item["id"]))
        runtime.audit(owner.id, "create_local_agent", "created")
        return JSONResponse(item, headers={"Cache-Control": "no-store"})

    @router.get("/api/v1/agent-access/local-clients/{client_id}/extension")
    async def local_client_extension(client_id: str):
        from .desktop_bundle import build_desktop_bundle
        from .local_connector import LocalCredentialError, local_launch_configuration, validate_client_id

        owner = owner_required()
        if not runtime.auth.local_enabled:
            raise HTTPException(409, "Enable Local MCP before downloading a client extension")
        try:
            identifier = validate_client_id(client_id)
        except LocalCredentialError:
            raise HTTPException(404, "Local client not found") from None
        record = next(
            (item for item in runtime.auth.list_local_clients(owner) if item["id"] == identifier),
            None,
        )
        if record is None or record["revoked"]:
            raise HTTPException(404, "Active local client not found")
        app_root = runtime.app_root or Path(__file__).resolve().parents[2]
        launch = local_launch_configuration(app_root, identifier)
        try:
            bundle = build_desktop_bundle(
                {
                    "id": identifier,
                    "name": record["name"],
                    "command": launch["command"],
                    "args": launch["args"],
                },
            )
        except Exception as error:
            raise HTTPException(500, "Could not build the local client extension") from error
        if not isinstance(bundle, bytes) or not bundle or len(bundle) > 2 * 1024 * 1024:
            raise HTTPException(500, "The local client extension is invalid")
        runtime.audit(owner.id, "download_local_agent_extension", "created")
        return Response(
            bundle,
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": f'attachment; filename="maestro-local-{identifier}.mcpb"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.delete("/api/v1/agent-access/grants/{grant_id}")
    async def revoke(grant_id: str):
        owner = owner_required()
        local_client = next(
            (item for item in runtime.auth.list_local_clients(owner) if item["grant_id"] == grant_id),
            None,
        )
        result = runtime.auth.revoke_grant(grant_id, owner)
        if result and local_client is not None:
            from .local_connector import LocalCredentialError, delete_local_credential
            try:
                delete_local_credential(runtime.directory, local_client["id"])
            except LocalCredentialError:
                # Grant revocation is already durable and immediately blocks
                # every loaded credential. A substituted link is never followed.
                runtime.audit(owner.id, "revoke_local_credential_cleanup", "deferred")
        runtime.audit(owner.id, "revoke_agent", "revoked")
        return {"revoked": result}

    @router.get("/api/v1/agent-access/audit")
    async def audit(limit: int = 50):
        owner_required()
        return {"events": runtime.events(limit)}

    return router
