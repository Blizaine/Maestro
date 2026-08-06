"""Regression tests for Maestro's quiet Uvicorn polling access log."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import unittest


_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.access_log_filter import (  # noqa: E402
    QUIET_POLL_PATHS,
    QUIET_POLL_PREFIXES,
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

    def test_successful_polling_gets_and_heads_are_suppressed(self):
        for path in QUIET_POLL_PATHS:
            with self.subTest(path=path):
                self.assertFalse(self.quiet_filter.filter(_access_record("GET", path, 200)))
                self.assertFalse(self.quiet_filter.filter(_access_record("HEAD", path, 204)))

        for prefix in QUIET_POLL_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertFalse(
                    self.quiet_filter.filter(
                        _access_record("GET", prefix + "fixture-id", 200)
                    )
                )

    def test_query_strings_do_not_restore_poll_noise(self):
        self.assertFalse(
            self.quiet_filter.filter(
                _access_record("GET", "/api/v1/system-stats?detail=1", 200)
            )
        )

    def test_errors_mutations_and_meaningful_requests_remain_visible(self):
        self.assertTrue(
            self.quiet_filter.filter(
                _access_record("GET", "/api/v1/system-stats", 500)
            )
        )
        self.assertTrue(
            self.quiet_filter.filter(
                _access_record("POST", "/api/v1/status/job-id", 200)
            )
        )
        self.assertTrue(
            self.quiet_filter.filter(_access_record("GET", "/api/v1/models", 200))
        )

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


if __name__ == "__main__":
    unittest.main()
