"""Run the API on its reserved socket and announce only verified startup."""
from __future__ import annotations

from collections.abc import Callable
import socket
from typing import Any

import uvicorn


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
        config = uvicorn.Config(app, host=host, port=port)
        ReadyServer(config).run(sockets=[server_socket])
    finally:
        server_socket.close()
