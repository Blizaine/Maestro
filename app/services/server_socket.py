"""Reserve the actual HTTP bind socket before starting the server."""

from __future__ import annotations

import socket
import sys


_MAX_PORT = 65535


def reserve_server_socket(
    bind_host: str,
    preferred: int,
    span: int = 20,
) -> socket.socket | None:
    """Bind and retain a port candidate for the caller.

    ``span`` is the number of fallback ports after ``preferred``. A preferred
    port of zero asks the OS for one ephemeral port and ignores the span. The
    returned socket remains bound; pass it to the server (for example,
    ``uvicorn.Server.run(sockets=[sock])``) and close it if startup is aborted.
    """
    if not isinstance(bind_host, str) or not bind_host or bind_host.strip() != bind_host:
        raise ValueError("bind_host must be a non-empty host or interface")
    if type(preferred) is not int or not 0 <= preferred <= _MAX_PORT:
        raise ValueError("preferred must be an integer port between 0 and 65535")
    if type(span) is not int or not 0 <= span <= _MAX_PORT:
        raise ValueError("span must be an integer between 0 and 65535")

    if preferred == 0:
        candidates = (0,)
    else:
        last = preferred + span
        if last > _MAX_PORT:
            raise ValueError("preferred + span must not exceed port 65535")
        candidates = range(preferred, last + 1)

    for port in candidates:
        try:
            addresses = socket.getaddrinfo(
                bind_host,
                port,
                type=socket.SOCK_STREAM,
            )
        except OSError:
            return None

        for family, socktype, proto, _canonname, address in addresses:
            candidate_socket = None
            try:
                candidate_socket = socket.socket(family, socktype, proto)
                if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    candidate_socket.setsockopt(
                        socket.SOL_SOCKET,
                        socket.SO_EXCLUSIVEADDRUSE,
                        1,
                    )
                # Deliberately do not set SO_REUSEADDR: ownership of the bound
                # socket must prevent another process from taking this port.
                candidate_socket.bind(address)
                return candidate_socket
            except OSError:
                if candidate_socket is not None:
                    try:
                        candidate_socket.close()
                    except OSError:
                        pass

    return None
