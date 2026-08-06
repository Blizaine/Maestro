"""Keep high-frequency UI polling out of Maestro's console output."""

from __future__ import annotations

import logging


# These endpoints are read-only UI heartbeats/status feeds. Successful polls
# add no diagnostic value and can fire multiple times per second. Their errors
# remain visible because QuietPollingAccessFilter only drops status < 400.
QUIET_POLL_PATHS = frozenset(
    {
        "/health",
        "/api/v1/audio/analyze/status",
        "/api/v1/civitai/downloads",
        "/api/v1/downloads/active",
        "/api/v1/jobs",
        "/api/v1/llm/status",
        "/api/v1/llm/stream-status",
        "/api/v1/models/downloads/status",
        "/api/v1/outputs",
        "/api/v1/sam/status",
        "/api/v1/system-stats",
    }
)

QUIET_POLL_PREFIXES = (
    "/api/v1/director/pipeline/",
    "/api/v1/director/pipelines/",
    "/api/v1/loras/scan-status/",
    "/api/v1/status/",
)


def _is_quiet_poll_path(path: str) -> bool:
    normalized = str(path or "").split("?", 1)[0]
    return normalized in QUIET_POLL_PATHS or any(
        normalized.startswith(prefix) for prefix in QUIET_POLL_PREFIXES
    )


class QuietPollingAccessFilter(logging.Filter):
    """Drop only successful read-only polls from Uvicorn's access logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn access records use:
        # (client_addr, method, full_path, http_version, status_code).
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True

        method = str(args[1]).upper()
        path = str(args[2])
        try:
            status_code = int(args[4])
        except (TypeError, ValueError):
            return True

        if method not in {"GET", "HEAD"} or status_code >= 400:
            return True
        return not _is_quiet_poll_path(path)


def install_quiet_access_filter(
    logger_name: str = "uvicorn.access",
) -> QuietPollingAccessFilter:
    """Install the polling filter once and return the active instance."""

    logger = logging.getLogger(logger_name)
    for current in logger.filters:
        if isinstance(current, QuietPollingAccessFilter):
            return current
    quiet_filter = QuietPollingAccessFilter()
    logger.addFilter(quiet_filter)
    return quiet_filter


__all__ = [
    "QUIET_POLL_PATHS",
    "QUIET_POLL_PREFIXES",
    "QuietPollingAccessFilter",
    "install_quiet_access_filter",
]
