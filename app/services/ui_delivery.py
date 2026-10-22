"""Targeted response compression for the initial UI payloads."""

from __future__ import annotations

import re

from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


_MODEL_OPTIONS_PATH = re.compile(r"/api/v1/model-options/[^/]+\Z")
_HASHED_ASSET_PATH = re.compile(
    r"/assets/[^/]+-[A-Za-z0-9_-]{8,}\.(?:js|css)\Z"
)
_COMPRESSIBLE_METHODS = frozenset({"GET", "HEAD"})


def _is_startup_payload(scope: Scope) -> bool:
    """Return whether this request is one of the bounded UI payload routes."""
    if scope.get("type") != "http" or scope.get("method", "").upper() not in _COMPRESSIBLE_METHODS:
        return False

    path = scope.get("path", "")
    return (
        path == "/api/v1/models"
        or _MODEL_OPTIONS_PATH.fullmatch(path) is not None
        or _HASHED_ASSET_PATH.fullmatch(path) is not None
    )


class UIStartupGZipMiddleware:
    """Apply Starlette gzip negotiation only to large startup UI payloads.

    Starlette remains responsible for compression, content-encoding handling,
    and ``Vary: Accept-Encoding``. The allowlist avoids buffering or changing
    delivery for generated media, downloads, thumbnails, and event streams.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=1024, compresslevel=4)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if _is_startup_payload(scope):
            await self.gzip(scope, receive, send)
            return

        await self.app(scope, receive, send)
