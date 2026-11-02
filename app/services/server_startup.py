"""Run the API on its reserved socket and announce only verified startup."""
from __future__ import annotations

from collections.abc import Callable
import os
import socket
from typing import Any

import uvicorn
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware


class PreserveTransportPeer:
    """Keep the socket peer separate from Uvicorn's forwarded client address."""

    def __init__(self, app):
        self.app = ProxyHeadersMiddleware(app, trusted_hosts=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"))

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            scope = {**scope, "maestro.transport_client": scope.get("client")}
        await self.app(scope, receive, send)


def run_reserved_server(
    app: Any,
    *,
    host: str,
    server_socket: socket.socket,
    on_ready: Callable[[uvicorn.Server], None],
) -> None:
    """Take socket ownership, keeping reservation through startup and cleanup.

    Uvicorn's startup flag is set after lifespan startup and listener creation.
    A launcher URL emitted before that point can incorrectly mark a failed
    bind as ready. Passing the bound socket also removes the probe/bind race.
    """
    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                on_ready(self)

    try:
        port = int(server_socket.getsockname()[1])
        config = uvicorn.Config(PreserveTransportPeer(app), host=host, port=port, proxy_headers=False)
        ReadyServer(config).run(sockets=[server_socket])
    finally:
        server_socket.close()
