"""Keep routine HTTP access logs out of Maestro's generation progress."""

from __future__ import annotations

import logging


class QuietPollingAccessFilter(logging.Filter):
    """Hide routine successful requests; retain the historical class name."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn access records use:
        # (client_addr, method, full_path, http_version, status_code).
        if record.levelno >= logging.WARNING or record.exc_info:
            return True
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True

        try:
            status_code = int(args[4])
        except (TypeError, ValueError):
            return True

        # Includes asset/preview downloads, cached responses and preference
        # writes. HTTP failures stay visible even though Uvicorn logs them at
        # INFO; startup messages use the separate uvicorn.error logger.
        return not 200 <= status_code < 400


def install_quiet_access_filter(
    logger_name: str = "uvicorn.access",
) -> QuietPollingAccessFilter:
    """Install the access filter once and return the active instance."""

    logger = logging.getLogger(logger_name)
    for current in logger.filters:
        if isinstance(current, QuietPollingAccessFilter):
            return current
    quiet_filter = QuietPollingAccessFilter()
    logger.addFilter(quiet_filter)
    return quiet_filter


__all__ = [
    "QuietPollingAccessFilter",
    "install_quiet_access_filter",
]
