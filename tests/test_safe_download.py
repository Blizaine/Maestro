"""Change detection for the active-downloads long-poll."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.safe_download import STALLED_AFTER_SECONDS, downloads_change_key  # noqa: E402


def _download(file_id: str, **overrides) -> dict:
    entry = {
        "file_id": file_id,
        "filename": f"{file_id}.safetensors",
        "status": "downloading",
        "downloaded_bytes": 1000,
        "total_bytes": 5000,
        "seconds_since_progress": 0.0,
    }
    entry.update(overrides)
    return entry


class DownloadsChangeKeyTests(unittest.TestCase):
    def test_elapsed_seconds_alone_do_not_change_the_key(self):
        self.assertEqual(
            downloads_change_key([_download("a", seconds_since_progress=5.0)]),
            downloads_change_key([_download("a", seconds_since_progress=12.0)]),
        )

    def test_crossing_the_stall_threshold_changes_the_key(self):
        self.assertEqual(STALLED_AFTER_SECONDS, 30)
        self.assertNotEqual(
            downloads_change_key([_download("a", seconds_since_progress=30.0)]),
            downloads_change_key([_download("a", seconds_since_progress=30.1)]),
        )

    def test_progress_status_and_membership_change_the_key(self):
        base = downloads_change_key([_download("a")])
        self.assertNotEqual(base, downloads_change_key([_download("a", downloaded_bytes=2000)]))
        self.assertNotEqual(base, downloads_change_key([_download("a", status="incomplete")]))
        self.assertNotEqual(base, downloads_change_key([_download("a"), _download("b")]))
        self.assertNotEqual(base, downloads_change_key([]))

    def test_order_does_not_matter_and_unknown_totals_are_accepted(self):
        first = _download("a", total_bytes=None)
        second = _download("b")
        self.assertEqual(
            downloads_change_key([first, second]),
            downloads_change_key([second, first]),
        )


if __name__ == "__main__":
    unittest.main()
