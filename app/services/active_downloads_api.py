"""HTTP boundary for the model-download progress banner."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Callable, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .safe_download import downloads_change_key, get_active_downloads as _read_active_downloads


def _version(downloads: list) -> str:
    # Derived from the state itself, so no server-side bookkeeping is needed
    # and an unchanged state keeps its version across restarts.
    key = repr(downloads_change_key(downloads)).encode("utf-8")
    return hashlib.blake2b(key, digest_size=8).hexdigest()


def create_router(
    *,
    read_downloads: Callable[[], list] = _read_active_downloads,
    check_interval: float = 1.0,
    wait_timeout: float = 25.0,
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/v1/downloads/active")
    async def get_active_downloads(since: Optional[str] = None):
        """Return a snapshot of in-progress model file downloads.

        UI polls this during long generation prep phases to surface
        download progress and stall warnings. Each entry includes
        `seconds_since_progress` so the UI can render "stalled — waiting
        for retry" badges without doing time math itself.

        Without `since` the snapshot is returned at once. With the
        `version` of the snapshot the caller already has, the request is
        held until the downloads change or `wait_timeout` passes, so an
        idle UI makes one request per 25 s instead of one every 2 s.
        The downloads are read directly on the event loop: a dict walk
        under a briefly held lock.

        Empty list when nothing is downloading. Best-effort tracking —
        if a download path bypasses `huggingface_hub`'s tqdm progress
        bar (some upstream Wan2GP code does this for misc files), it
        won't appear here even though the safe_download timeout
        protections still apply.
        """
        downloads = read_downloads()
        version = _version(downloads)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_timeout
        while since == version and loop.time() < deadline:
            await asyncio.sleep(min(check_interval, deadline - loop.time()))
            downloads = read_downloads()
            version = _version(downloads)
        return JSONResponse(
            {"downloads": downloads, "version": version},
            headers={"Cache-Control": "no-store"},
        )

    return router
