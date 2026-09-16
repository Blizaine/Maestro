"""Keep a dropped Windows client from closing the entire HTTP listener.

CPython issue https://github.com/python/cpython/issues/93821 affects the
Proactor accept loop: ERROR_NETNAME_DELETED from one AcceptEx completion
closes the listening socket. Install a retry on this server loop only, before
Uvicorn starts accepting. Other loops, I/O operations and platforms keep their
normal behavior; no global event-loop policy or installed dependency is edited.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import sys


_logger = logging.getLogger(__name__)
_ERROR_NETNAME_DELETED = 64
_ACCEPT_RETRY_DELAY = 0.05


async def _accept_once(proactor, listener):
    # AcceptEx setup follows CPython's asyncio/windows_events.py (PSF-2.0).
    # Own the pending connection here so errors as well as cancellation close
    # it, without the stock accept_coro's unobserved OSError task exception.
    import _overlapped

    proactor._register_with_iocp(listener)
    conn = proactor._get_accept_socket(listener.family)
    try:
        ov = _overlapped.Overlapped(0)
        ov.AcceptEx(listener.fileno(), conn.fileno())

        def finish_accept(transferred, key, completed):
            completed.getresult()
            conn.setsockopt(
                socket.SOL_SOCKET, _overlapped.SO_UPDATE_ACCEPT_CONTEXT,
                struct.pack("@P", listener.fileno()),
            )
            conn.settimeout(listener.gettimeout())
            return conn, conn.getpeername()

        return await proactor._register(ov, listener, finish_accept)
    except BaseException:
        conn.close()
        raise


async def _accept_with_retry(proactor, listener):
    reported = False
    while True:
        try:
            return await _accept_once(proactor, listener)
        except OSError as exc:
            if (
                getattr(exc, "winerror", None) != _ERROR_NETNAME_DELETED
                or listener.fileno() == -1
                or proactor._loop.is_closed()
            ):
                raise
            if not reported:
                _logger.warning(
                    "[Maestro] Incoming connection disconnected during accept "
                    "(WinError 64); keeping the HTTP listener available."
                )
                reported = True
            # Yield during repeated aborted connections. Cancellation during
            # this wait propagates, so shutdown never reopens a listener.
            await asyncio.sleep(_ACCEPT_RETRY_DELAY)


def install_windows_accept_retry(loop=None) -> bool:
    """Protect the current Proactor instance; leave selectors/Unix untouched."""
    if sys.platform != "win32":
        return False
    from asyncio.windows_events import IocpProactor

    loop = loop if loop is not None else asyncio.get_running_loop()
    proactor = getattr(loop, "_proactor", None)
    if not isinstance(proactor, IocpProactor):
        return False
    if getattr(proactor, "_maestro_accept_retry", False):
        return True

    def accept(listener):
        return loop.create_task(_accept_with_retry(proactor, listener))

    proactor.accept = accept
    proactor._maestro_accept_retry = True
    return True


async def configure_http_runtime() -> None:
    if install_windows_accept_retry():
        print("[Maestro] Windows HTTP connection recovery enabled.", flush=True)
