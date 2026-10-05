"""Bounded, ephemeral media for in-flight generation previews.

Preview bytes never enter job archives or the output gallery. Each window has
an immutable context, so a late encoder cannot publish into the next window.
"""
from __future__ import annotations

import io
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote


PREVIEW_MODES = ("off", "rgb", "tiny_vae_frames", "tiny_vae_video")


def preview_mode(value: Any) -> str:
    return value if isinstance(value, str) and value in PREVIEW_MODES else "off"


@dataclass(frozen=True)
class PreviewMedia:
    data: bytes
    content_type: str
    revision: int


@dataclass
class _Entry:
    context: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    notice: str | None = None
    media: OrderedDict[int, PreviewMedia] = field(default_factory=OrderedDict)


class GenerationPreviewCache:
    """Keep at most two small revisions per active job under a global byte cap."""

    def __init__(self, max_bytes: int = 16 * 1024 * 1024,
                 max_media_bytes: int = 4 * 1024 * 1024, max_jobs: int = 8):
        self.max_bytes = max_bytes
        self.max_media_bytes = max_media_bytes
        self.max_jobs = max_jobs
        self._jobs: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._revision = 0
        self._bytes = 0

    def begin(self, job_id: str) -> None:
        with self._lock:
            self.clear(job_id)
            self._jobs[job_id] = _Entry()
            while len(self._jobs) > self.max_jobs:
                self.clear(next(iter(self._jobs)))

    def clear(self, job_id: str) -> None:
        with self._lock:
            entry = self._jobs.pop(job_id, None)
            if entry is not None:
                self._bytes -= sum(len(media.data) for media in entry.media.values())

    def set_context(self, job_id: str, context: dict[str, Any]) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is not None and entry.context != context:
                entry.context = dict(context)
                entry.latest = None
                entry.notice = None

    def set_notice(self, job_id: str, notice: str | None) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is not None:
                entry.notice = str(notice)[:500] if notice else None

    def publish(self, job_id: str, media: Any, context: dict[str, Any]) -> dict[str, Any] | None:
        # Serialize outside the cache lock so HTTP polling stays responsive.
        from PIL import Image

        if isinstance(media, Image.Image):
            buffer = io.BytesIO()
            media.convert("RGB").save(buffer, format="JPEG", quality=80)
            data, content_type, kind = buffer.getvalue(), "image/jpeg", "image"
        elif isinstance(getattr(media, "video", None), bytes):
            data, content_type, kind = media.video, "video/mp4", "video"
        else:
            return None
        if not data or len(data) > min(self.max_media_bytes, self.max_bytes):
            return None

        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry.context is None or (
                {key: value for key, value in entry.context.items() if key != "mode"}
                != {key: value for key, value in context.items() if key != "mode"}
            ):
                return None
            self._revision += 1
            revision = self._revision
            item = PreviewMedia(data, content_type, revision)
            entry.media[revision] = item
            self._bytes += len(data)
            while len(entry.media) > 2:
                _, removed = entry.media.popitem(last=False)
                self._bytes -= len(removed.data)
            self._jobs.move_to_end(job_id)
            while self._bytes > self.max_bytes:
                oldest_job = next(key for key, value in self._jobs.items() if value.media)
                oldest = self._jobs[oldest_job]
                removed_revision, removed = oldest.media.popitem(last=False)
                self._bytes -= len(removed.data)
                if oldest.latest and oldest.latest["revision"] == removed_revision:
                    oldest.latest = None
            entry.latest = {
                "url": f"/api/v1/jobs/{quote(job_id, safe='')}/preview/{revision}",
                "kind": kind,
                "revision": revision,
                "mode": context.get("mode", "rgb"),
                **{key: max(1, int(context.get(key, 1) or 1)) for key in (
                    "window", "total_windows", "clip", "total_clips",
                )},
            }
            return dict(entry.latest)

    def fields(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            entry = self._jobs.get(str(job.get("id") or ""))
            if job.get("status") != "running" or job.get("cancel_requested") or entry is None:
                return {"preview": None, "preview_notice": None}
            return {"preview": dict(entry.latest) if entry.latest else None,
                    "preview_notice": entry.notice}

    def get(self, job_id: str, revision: int) -> PreviewMedia | None:
        with self._lock:
            entry = self._jobs.get(job_id)
            return entry.media.get(revision) if entry else None


def preview_response(media: PreviewMedia, range_header: str | None = None,
                     if_none_match: str | None = None):
    """Serve immutable in-memory media, including browser video range requests."""
    from starlette.responses import Response

    etag = f'"preview-{media.revision}"'
    headers = {"Cache-Control": "private, max-age=60, immutable", "ETag": etag,
               "Accept-Ranges": "bytes", "X-Content-Type-Options": "nosniff"}
    if if_none_match == etag and not range_header:
        return Response(status_code=304, headers=headers)
    data, status = media.data, 200
    if range_header:
        try:
            if not range_header.startswith("bytes=") or "," in range_header:
                raise ValueError("Invalid byte range")
            start_text, end_text = range_header[6:].split("-", 1)
            if start_text:
                start = int(start_text)
                end = min(len(data) - 1, int(end_text)) if end_text else len(data) - 1
            else:
                suffix = int(end_text)
                if suffix <= 0:
                    raise ValueError("Invalid suffix range")
                start, end = max(0, len(data) - suffix), len(data) - 1
            if start < 0 or start >= len(data) or end < start:
                raise ValueError("Unsatisfiable byte range")
        except (ValueError, TypeError):
            headers["Content-Range"] = f"bytes */{len(data)}"
            return Response(status_code=416, headers=headers)
        headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
        data, status = data[start:end + 1], 206
    return Response(data, status_code=status, media_type=media.content_type, headers=headers)
