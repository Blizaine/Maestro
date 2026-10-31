"""Bundled CPU-only HTTPS/MCP controller, launched by ordinary Pinokio scripts."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import html
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket
from starlette.background import BackgroundTask
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

from .assets import AssetStoreError
from .auth import AuthError
from .boundary import AccessBoundary, same_origin, secure_transport
from .bridge import BackendClient, BridgeError
from .catalog import build_catalog, domain_counts
from .context import principal_context
from .guides import GUIDES, SERVER_INSTRUCTIONS
from .http import RateLimiter, json_body, owner_required, scope_required
from .mcp_server import create_mcp_server
from .state import ControllerLock, RuntimeState, atomic_json, private_lan_ipv4_addresses
from .transfers import TransferStoreError

SCOPES = {
    "observe": "Read models, capabilities, workspace media and owned job receipts",
    "generate": "Plan, enhance and render; control this agent's jobs",
    "manage": "Upload media, edit projects, manage models and downloads",
    "admin": "Change app configuration and control the idle workstation",
}


def _page(title, body):
    return HTMLResponse('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title><style>body{font:16px system-ui;background:#171b1e;color:#eef0f2;margin:0;padding:5vh 6vw}main{max-width:640px;margin:auto;padding:28px;background:#242b30;border-radius:18px;border:1px solid #414a52}p,label{line-height:1.6}label{display:block;margin:16px 0}button,a{color:inherit}button{font:inherit;border:0;border-radius:10px;padding:12px 22px;margin:8px;background:#ff9d61;color:#151a1c}.secondary{background:#47515a;color:white}code{overflow-wrap:anywhere}</style><main><h1>' + html.escape(title) + '</h1>' + body + '</main>', headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY"})


async def _oauth_body(request):
    raw = await request.body()
    if len(raw) > 65536:
        raise HTTPException(413, "OAuth request is too large")
    try:
        if "application/json" in request.headers.get("content-type", ""):
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        return {k: v[-1] for k, v in parse_qs(raw.decode("utf-8"), max_num_fields=32).items()}
    except (ValueError, UnicodeError) as error:
        raise HTTPException(400, "Invalid OAuth request") from error


def _canonical_resource(runtime, value):
    try:
        return runtime.canonical_resource(value)
    except ValueError as error:
        raise AuthError(400, "resource must identify this installation's approved MCP endpoint") from error


def _redirect(result):
    parts = urlsplit(result["redirect_uri"])
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in {"code", "state", "error"}]
    query += [(k, result[k]) for k in ("code", "state", "error") if result.get(k) is not None]
    return RedirectResponse(urlunsplit(parts._replace(query=urlencode(query))), status_code=303, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


class _MCPRoute:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


def create_app(
    runtime,
    *,
    bridge=None,
    version="4.0.0",
    console_printer=print,
    tls_material=None,
    frame_ancestor_origins: tuple[str, ...] | list[str] | None = None,
):
    from mcp.server.transport_security import TransportSecuritySettings

    from .http import management_router

    bridge = bridge or BackendClient(runtime)
    mcp = create_mcp_server(bridge, runtime.app_root, version)
    if tls_material:
        runtime.configure_certificate_ips(tls_material.ip_addresses)
    transport = mcp.streamable_http_app(
        json_response=True, stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(runtime.allowed_authorities),
            allowed_origins=list(runtime.allowed_origins),
        ),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with mcp.session_manager.run():
            yield
        await bridge.close()

    app = FastAPI(title="Maestro Agent Access", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.bridge, app.state.runtime, app.state.mcp = bridge, runtime, mcp
    # Native mutations reserve their receipt in the backend, never twice in
    # this same shared journal. Controller-owned mutations have domain-specific
    # receipts (lifecycle, content-hashed uploads and OAuth codes).
    def transfer_scope(principal, method, path):
        match = re.fullmatch(r"/api/v1/agent-access/transfers/([^/]+)(?:/chunk)?", path)
        if method not in {"GET", "HEAD"} or not match:
            return None
        admin = "admin" in principal.scopes
        needed = bridge.transfers.access_scope(match[1], principal.id, owner=principal.owner, admin=admin)
        return "admin" if admin else needed

    # Native routes validate paths using their current configured storage roots.
    # The controller's own files are always resolved by opaque owned IDs.
    app.add_middleware(
        AccessBoundary,
        runtime=runtime,
        always_protect=True,
        asset_resolver=bridge.assets.resolve_tree,
        dynamic_scope_resolver=transfer_scope,
        validate_controller_origin=True,
        allow_local_ui=True,
        frame_ancestor_origins=frame_ancestor_origins,
    )
    # /mcp is mounted without a trailing-slash redirect or scope rewriting.
    app.router.routes.append(Route("/mcp", _MCPRoute(transport), methods=["GET", "POST", "DELETE"]))

    @app.exception_handler(AuthError)
    @app.exception_handler(AssetStoreError)
    @app.exception_handler(BridgeError)
    @app.exception_handler(TransferStoreError)
    async def expected_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status, headers={"Cache-Control": "no-store"})

    def show_code():
        code = runtime.auth.issue_owner_code()
        console_printer(f"[Maestro Agent Access] Browser code: {code}")
        console_printer("This code expires in 10 minutes. Enter it only in the verified secure Maestro browser.")

    app.state.show_code = show_code
    app.include_router(management_router(
        runtime,
        request_console_code=show_code,
        status_extra=lambda: {"backend_online": bridge.online()},
        local_device_access=True,
    ))

    def request_origin(request):
        origin = runtime.origin_for_scope(request.scope)
        if origin is None:
            raise HTTPException(421, "Use a verified Maestro endpoint on its configured port")
        return origin

    def metadata(origin):
        return {"issuer": origin, "authorization_endpoint": origin + "/oauth/authorize", "token_endpoint": origin + "/oauth/token", "registration_endpoint": origin + "/oauth/register", "revocation_endpoint": origin + "/oauth/revoke", "response_types_supported": ["code"], "grant_types_supported": ["authorization_code", "refresh_token"], "token_endpoint_auth_methods_supported": ["none"], "code_challenge_methods_supported": ["S256"], "scopes_supported": list(SCOPES), "client_id_metadata_document_supported": False}

    @app.get("/.well-known/oauth-authorization-server")
    async def authorization_metadata(request: Request):
        return JSONResponse(metadata(request_origin(request)), headers={"Cache-Control": "no-store"})

    @app.get("/.well-known/oauth-protected-resource")
    @app.get("/.well-known/oauth-protected-resource/mcp")
    async def resource_metadata(request: Request):
        origin = request_origin(request)
        return {"resource": origin + "/mcp", "authorization_servers": [origin], "scopes_supported": list(SCOPES), "bearer_methods_supported": ["header"], "resource_name": runtime.config["label"]}

    @app.get("/.well-known/maestro")
    async def maestro_metadata(request: Request):
        origin = request_origin(request)
        return {"schema": 1, "installation_id": runtime.installation_id, "workstation_id": runtime.config["workstation_id"], "label": runtime.config["label"], "version": version, "mcp_url": origin + "/mcp", "pairing_required": True, "enabled": runtime.auth.enabled, "instructions": SERVER_INSTRUCTIONS, "guide_url": origin + "/api/v1/agent-access/guide/overview", "certificate_sha256": runtime.public_status()["certificate_sha256"]}

    @app.get("/agent-access/certificate.pem")
    async def certificate():
        cert = runtime.directory / "tls" / "server-cert.pem"
        if not cert.is_file():
            raise HTTPException(404, "Certificate is not ready")
        return FileResponse(cert, media_type="application/x-pem-file", filename=runtime.hostname + ".pem")

    registration_limit, token_limit = RateLimiter(12), RateLimiter(60)

    def public_secure(request):
        if not secure_transport(request.scope):
            raise HTTPException(426, "OAuth requires the secure Maestro address")
        request_origin(request)
        if request.headers.get("origin") and not same_origin(request.scope):
            raise HTTPException(403, "This origin is not allowed")

    @app.post("/oauth/register")
    async def register(request: Request):
        public_secure(request)
        registration_limit.check()
        return JSONResponse(runtime.auth.register_client(await json_body(request)), status_code=201, headers={"Cache-Control": "no-store"})

    @app.get("/oauth/authorize")
    async def authorize(request: Request):
        public_secure(request)
        query = request.query_params
        if query.get("response_type") != "code" or query.get("code_challenge_method") != "S256":
            raise HTTPException(400, "Authorization code with S256 PKCE is required")
        principal = principal_context.get()
        if principal is None or not principal.owner:
            target = "/oauth/authorize?" + request.url.query
            return RedirectResponse("/agent-access/login?" + urlencode({"next": target}), status_code=303)
        pending = runtime.auth.create_authorization(
            query.get("client_id"),
            query.get("redirect_uri"),
            query.get("scope", "observe generate manage admin"),
            query.get("code_challenge"),
            _canonical_resource(runtime, query.get("resource")),
            query.get("state"),
        )
        return RedirectResponse("/api/v1/agent-access/authorizations/" + pending["request_id"], status_code=303)

    @app.get("/api/v1/agent-access/authorizations/{request_id}")
    async def consent(request_id: str):
        owner_required()
        pending = runtime.auth.get_authorization(request_id)
        controls = "".join(f'<label><input type="checkbox" name="scope" value="{scope}" checked> <strong>{scope}</strong> — {html.escape(SCOPES[scope])}</label>' for scope in pending["scopes"])
        return _page("Connect an agent", f'<p><strong>{html.escape(pending["client_name"])}</strong> requests access to <strong>{html.escape(runtime.config["label"])}</strong>.</p><p>This client can read shared workspace media. Generation and management can use GPU time, disk space and network downloads. Administration can change configuration and stop an idle app. It cannot grant itself owner access.</p><form method="post">{controls}<button name="decision" value="approve">Allow selected access</button><button class="secondary" name="decision" value="deny">Deny</button></form><p>Callback: <code>{html.escape(pending["redirect_uri"])}</code></p>')

    @app.post("/api/v1/agent-access/authorizations/{request_id}")
    async def consent_post(request_id: str, request: Request):
        owner = owner_required()
        if not same_origin(request.scope):
            raise HTTPException(403, "A same-origin approval is required")
        raw = await request.body()
        if len(raw) > 8192:
            raise HTTPException(413, "Approval is too large")
        form = parse_qs(raw.decode("utf-8"), max_num_fields=12)
        selected = form.get("scope", [])
        result = runtime.auth.approve_authorization(request_id, owner, selected) if form.get("decision") == ["approve"] else runtime.auth.deny_authorization(request_id, owner)
        runtime.audit(owner.id, "oauth_consent", "approved" if result.get("code") else "denied")
        return _redirect(result)

    @app.post("/oauth/token")
    async def token(request: Request):
        public_secure(request)
        token_limit.check()
        body = await _oauth_body(request)
        try:
            if body.get("grant_type") == "authorization_code":
                resource = _canonical_resource(runtime, body.get("resource"))
                result = runtime.auth.exchange_code(body.get("client_id"), body.get("code"), body.get("redirect_uri"), body.get("code_verifier"), resource)
            elif body.get("grant_type") == "refresh_token":
                resource = _canonical_resource(runtime, body.get("resource"))
                result = runtime.auth.refresh_token(body.get("client_id"), body.get("refresh_token"), resource)
            else:
                return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
        except AuthError as error:
            if error.status == 503 or (error.status == 403 and not runtime.auth.enabled):
                return JSONResponse({"error": "temporarily_unavailable", "error_description": str(error)}, status_code=503, headers={"Cache-Control": "no-store", "Retry-After": "30"})
            return JSONResponse({"error": "invalid_grant", "error_description": str(error)}, status_code=429 if error.status == 429 else 400, headers={"Cache-Control": "no-store"})
        return JSONResponse(result, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    @app.post("/oauth/revoke")
    async def revoke_token(request: Request):
        public_secure(request)
        token_limit.check()
        body = await _oauth_body(request)
        runtime.auth.logout(str(body.get("token", "")))
        return JSONResponse({}, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/agent-access/peers")
    async def peers():
        return await bridge.peers()

    @app.get("/api/v1/agent-access/guide/{topic}")
    async def guide(topic: str):
        scope_required("observe")
        if topic not in GUIDES:
            raise HTTPException(404, "Guide not found")
        return {"topic": topic, "markdown": GUIDES[topic]}

    @app.get("/api/v1/agent-access/coverage")
    async def coverage():
        scope_required("observe")
        catalog = build_catalog(await bridge.openapi())
        return {"operation_count": len(catalog), "domains": domain_counts(catalog), "binary_transfers": "Use the resumable asset tools and authenticated native download URLs."}

    @app.post("/api/v1/agent-access/lifecycle")
    async def lifecycle(request: Request):
        body = await json_body(request)
        return await bridge.lifecycle(body.get("action"), request.headers.get("idempotency-key"))

    @app.post("/api/v1/agent-access/assets/uploads")
    async def begin_upload(request: Request):
        body = await json_body(request)
        return await bridge.upload_begin(body.get("filename"), body.get("size"), body.get("sha256"))

    @app.get("/api/v1/agent-access/assets/uploads/{upload_id}")
    async def upload_status(upload_id: str):
        return await bridge.upload_status(upload_id)

    @app.put("/api/v1/agent-access/assets/uploads/{upload_id}")
    async def upload_chunk(upload_id: str, request: Request):
        scope_required("manage")
        offset = request.query_params.get("offset", "")
        if not offset.isdecimal() or len(offset) > 12:
            raise HTTPException(400, "Use a numeric byte offset")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > bridge.assets.max_chunk_bytes:
                raise HTTPException(413, "Upload chunk is too large")
        return bridge.assets.write_chunk(principal_context.get().id, upload_id, int(offset), bytes(data))

    @app.post("/api/v1/agent-access/assets/uploads/{upload_id}/complete")
    async def complete_upload(upload_id: str, request: Request):
        return await bridge.upload_complete(upload_id, request.headers.get("idempotency-key"))

    @app.get("/api/v1/agent-access/transfers/{transfer_id}/chunk")
    async def transfer_chunk(transfer_id: str, offset: int = 0, max_bytes: int = 65536):
        return await bridge.read_transfer_chunk(transfer_id, offset, max_bytes)

    @app.api_route("/api/v1/agent-access/transfers/{transfer_id}", methods=["GET", "HEAD"])
    async def transfer_file(transfer_id: str, request: Request):
        principal = principal_context.get()
        if principal is None:
            raise HTTPException(401, "Pair this agent first")
        admin = "admin" in principal.scopes
        options = {"owner": principal.owner, "admin": admin}
        metadata = bridge.transfers.metadata(transfer_id, principal.id, **options)
        needed = bridge.transfers.access_scope(transfer_id, principal.id, **options)
        if not principal.owner and not admin:
            scope_required(needed)
        size = metadata["size"]
        start, length, code = 0, size, 200
        headers = {
            "Cache-Control": "private, no-store", "Accept-Ranges": "bytes",
            "ETag": '"' + metadata["sha256"] + '"',
            "Content-Disposition": "attachment; filename*=UTF-8''" + quote(metadata["filename"], safe=""),
            "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox; default-src 'none'",
        }
        if request.headers.get("range"):
            if len(request.headers["range"]) > 64:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", request.headers["range"])
            if not match or not any(match.groups()) or size == 0:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                suffix = int(last)
                start, end = max(0, size - suffix), size - 1
            if start >= size or end < start or (not first and int(last) == 0):
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            length, code = end - start + 1, 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(length)
        if request.method == "HEAD":
            return Response(status_code=code, headers=headers, media_type=metadata["content_type"])
        return StreamingResponse(bridge.transfers.iter_bytes(transfer_id, principal.id, **options, offset=start, length=length), status_code=code, headers=headers, media_type=metadata["content_type"])

    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request):
        if not bridge.online():
            if path == "" and principal_context.get() and principal_context.get().owner:
                return _page("Maestro Agent Access", f'<p>Maestro is stopped. The paired controller is ready.</p><p>MCP endpoint: <code>{html.escape(runtime.mcp_url)}</code></p><p>Use Pinokio to start Maestro, or the authenticated workstation lifecycle tool.</p>')
            raise HTTPException(503, "Maestro is stopped or not ready")
        # Tokens are forwarded only to this installation's registered loopback
        # backend. Never follow redirects or trust proxy/environment settings.
        headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "origin", "cookie", "authorization", "connection", "transfer-encoding", "accept-encoding", "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto"}}
        headers.update(bridge.headers())
        target = bridge.target() + request.url.path + ("?" + request.url.query if request.url.query else "")
        upstream = bridge.client.build_request(request.method, target, headers=headers, content=request.stream())
        try:
            response = await bridge.client.send(upstream, stream=True)
        except httpx.HTTPError as error:
            raise BridgeError(503, "The backend connection ended. Inspect the same submission receipt before retrying with its original request ID.") from error
        outgoing = {k: v for k, v in response.headers.items() if k.lower() not in {"connection", "transfer-encoding", "content-encoding"}}
        return StreamingResponse(response.aiter_raw(), status_code=response.status_code, headers=outgoing, background=BackgroundTask(response.aclose))

    @app.websocket("/{path:path}")
    async def websocket_proxy(websocket: WebSocket, path: str):
        from websockets.asyncio.client import connect
        if not bridge.online():
            await websocket.close(code=1013)
            return
        target = bridge.target().replace("http://", "ws://", 1) + websocket.url.path + ("?" + websocket.url.query if websocket.url.query else "")
        protocols = websocket.headers.get("sec-websocket-protocol", "").split(",")
        protocols = [p.strip() for p in protocols if p.strip()]
        async with connect(target, additional_headers=bridge.headers(), subprotocols=protocols or None, proxy=None, max_size=8 * 1024 * 1024) as upstream:
            await websocket.accept(subprotocol=upstream.subprotocol)
            async def incoming():
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    await upstream.send(message.get("bytes") if message.get("bytes") is not None else message.get("text", ""))
            async def outgoing():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)
            tasks = [asyncio.create_task(incoming()), asyncio.create_task(outgoing())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return app


async def serve(args):
    app_root = Path(args.app_root).resolve()
    runtime = RuntimeState(app_root / "settings" / "agent-access", app_root=app_root)
    with ControllerLock(runtime.directory):
        await _serve_runtime(args, runtime)


async def _serve_runtime(args, runtime):
    import uvicorn

    from .discovery import DiscoveryService
    from .lifecycle import LifecycleManager
    from .tls import ensure_tls

    lan_addresses = private_lan_ipv4_addresses()
    material = ensure_tls(runtime.directory / "tls", runtime.installation_id, ip_addresses=list(lan_addresses))
    runtime.configure_certificate_ips(material.ip_addresses)
    network_metadata = runtime.lan_endpoint_metadata(lan_addresses)
    print(f"[Maestro Agent Access] Secure setup: {runtime.local_url}", flush=True)
    print(f"[Maestro Agent Access] Certificate SHA-256: {material.fingerprint_sha256}", flush=True)
    print("Agent Access is opt-in. Trust this exact certificate on connecting devices; never disable TLS verification.", flush=True)
    setup_pending = args.setup
    while True:
        enabled = runtime.auth.enabled
        active_network_metadata = {
            **network_metadata,
            "lan_mcp_urls": network_metadata["lan_mcp_urls"] if enabled else [],
        }
        bridge = BackendClient(runtime)
        bridge.lifecycle_manager = LifecycleManager(runtime, idle_probe=bridge.idle_probe)
        config = runtime.config
        label = config["label"].encode("utf-8")[:180].decode("utf-8", errors="ignore")
        discovery = DiscoveryService(runtime.installation_id, runtime.hostname, runtime.port, label, args.version, material.fingerprint_sha256, addresses=lan_addresses, enabled=enabled and config["discovery_enabled"], workstation_id=config.get("workstation_id"))
        bridge.discovery = discovery
        app = create_app(runtime, bridge=bridge, version=args.version, tls_material=material)
        if setup_pending:
            app.state.show_code()
            setup_pending = False
        discovery_status = "disabled"
        try:
            discovery.start()
            discovery_status = "advertising" if discovery.enabled else "disabled"
        except Exception:
            discovery_status = "unavailable"
            print("[Maestro Agent Access] LAN discovery unavailable; use the verified endpoint directly.", flush=True)
        atomic_json(runtime.directory / "controller.json", {"status": "starting", "pid": os.getpid(), "certificate_sha256": material.fingerprint_sha256, "discovery_status": discovery_status, **active_network_metadata, "updated_at": time.time()})
        server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0" if enabled else "127.0.0.1", port=runtime.port, ssl_certfile=str(material.cert_path), ssl_keyfile=str(material.key_path), access_log=False, log_level="warning", timeout_graceful_shutdown=10))
        async def watch():
            announced = False
            while not server.should_exit:
                if server.started and not announced:
                    atomic_json(runtime.directory / "controller.json", {"status": "ready", "pid": os.getpid(), "certificate_sha256": material.fingerprint_sha256, "discovery_status": discovery_status, **active_network_metadata, "updated_at": time.time()})
                    print(f"MAESTRO_AGENT_ACCESS_READY {runtime.local_url}", flush=True)
                    announced = True
                await asyncio.sleep(1)
                if runtime.auth.enabled != enabled or runtime.config != config:
                    server.should_exit = True
                    return True
            return False
        watcher = asyncio.create_task(watch())
        try:
            await server.serve()
            restart = watcher.done() and not watcher.cancelled() and watcher.result()
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            discovery.stop()
            atomic_json(runtime.directory / "controller.json", {"status": "stopped", "pid": os.getpid(), "certificate_sha256": material.fingerprint_sha256, "discovery_status": "disabled", **active_network_metadata, "updated_at": time.time()})
        if not restart:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--version")
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args()
    if not args.version:
        args.version = (Path(args.app_root).resolve().parent / "VERSION").read_text(encoding="utf-8").strip()
    if not args.version or len(args.version) > 32:
        parser.error("Invalid Maestro release version")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
