"""Regression tests for Maestro's quiet Uvicorn access log."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys
import unittest


_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.access_log_filter import (  # noqa: E402
    QuietPollingAccessFilter,
    install_quiet_access_filter,
)


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:12345", method, path, "1.1", status),
        None,
    )


class TestQuietPollingAccessFilter(unittest.TestCase):
    def setUp(self):
        self.quiet_filter = QuietPollingAccessFilter()

    def test_successful_requests_are_quiet_across_ui_and_generation_routes(self):
        requests = (
            ("GET", "/"),
            ("HEAD", "/assets/index.js"),
            ("GET", "/api/v1/jobs/job-id/preview/38"),
            ("GET", "/api/v1/system-stats?detail=1"),
            ("GET", "/api/v1/file/image.png?workspace=Test"),
            ("GET", "/api/v1/models"),
            ("PUT", "/api/v1/studio-preferences"),
            ("POST", "/api/v1/loras/check-updates"),
            ("DELETE", "/api/v1/outputs/image.png"),
        )
        for method, path in requests:
            for status in (200, 201, 204, 206, 302, 304, 307):
                with self.subTest(method=method, path=path, status=status):
                    self.assertFalse(
                        self.quiet_filter.filter(_access_record(method, path, status))
                    )

    def test_http_failures_remain_visible_for_all_methods(self):
        for method in ("GET", "HEAD", "POST", "PUT", "DELETE"):
            for status in (400, 401, 403, 404, 409, 422, 500, 503):
                with self.subTest(method=method, status=status):
                    self.assertTrue(
                        self.quiet_filter.filter(
                            _access_record(method, "/api/v1/jobs", status)
                        )
                    )

    def test_warnings_and_exceptions_are_never_suppressed(self):
        for level in (logging.WARNING, logging.ERROR, logging.CRITICAL):
            record = _access_record("GET", "/", 200)
            record.levelno = level
            self.assertTrue(self.quiet_filter.filter(record))
        record = _access_record("GET", "/", 200)
        record.exc_info = (RuntimeError, RuntimeError("fixture"), None)
        self.assertTrue(self.quiet_filter.filter(record))

    def test_unexpected_log_record_shapes_are_preserved(self):
        record = logging.LogRecord(
            "uvicorn.access",
            logging.INFO,
            __file__,
            1,
            "ordinary message",
            (),
            None,
        )
        self.assertTrue(self.quiet_filter.filter(record))
        for status in (None, "unknown", 101):
            with self.subTest(status=status):
                self.assertTrue(
                    self.quiet_filter.filter(_access_record("GET", "/", status))
                )

    def test_installer_is_idempotent(self):
        logger_name = "maestro.tests.quiet-access"
        logger = logging.getLogger(logger_name)
        original_filters = list(logger.filters)
        try:
            logger.filters.clear()
            first = install_quiet_access_filter(logger_name)
            second = install_quiet_access_filter(logger_name)
            self.assertIs(first, second)
            self.assertEqual(
                sum(isinstance(item, QuietPollingAccessFilter) for item in logger.filters),
                1,
            )
        finally:
            logger.filters[:] = original_filters

    def test_filter_survives_real_uvicorn_logging_configuration(self):
        # Uvicorn replaces logging handlers during startup. Exercise that
        # configuration in isolation, without starting a server or using CUDA.
        script = r'''
import io
import logging
import uvicorn
from services.access_log_filter import install_quiet_access_filter

quiet_filter = install_quiet_access_filter()
uvicorn.Config("fixture:app", log_level="info")
access = logging.getLogger("uvicorn.access")
assert quiet_filter in access.filters
assert install_quiet_access_filter() is quiet_filter
output = io.StringIO()
handler = logging.StreamHandler(output)
access.handlers[:] = [handler]
message = '%s - "%s %s HTTP/%s" %d'
access.info(message, "client", "GET", "/assets/index.js", "1.1", 200)
access.info(message, "client", "PUT", "/api/v1/studio-preferences", "1.1", 200)
assert output.getvalue() == ""
access.info(message, "client", "GET", "/missing", "1.1", 404)
access.info(message, "client", "POST", "/api/v1/generate", "1.1", 500)
logging.getLogger("uvicorn.error").handlers[:] = [handler]
logging.getLogger("uvicorn.error").propagate = False
logging.getLogger("uvicorn.error").info("Application startup complete.")
text = output.getvalue()
assert "404" in text and "500" in text
assert "Application startup complete." in text
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(_APP)
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_APP,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
