"""Finished media can replace a generated original that is still being read."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import win_safe_files  # noqa: E402
from services.win_safe_files import safe_replace  # noqa: E402

WINDOWS = sys.platform == "win32"


def visible_files(folder: Path) -> set[str]:
    return {entry.name for entry in folder.iterdir() if not entry.name.startswith(".")}


class SafeReplaceTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)
        self.target = self.folder / "generated.mp4"
        self.source = self.folder / "generated.media-flow.mp4"
        self.target.write_bytes(b"original")
        self.source.write_bytes(b"finished")

    def tearDown(self):
        self._folder.cleanup()

    def test_replaces_when_nothing_holds_the_file(self):
        self.assertEqual(safe_replace(self.source, self.target),
                         {"replaced": True, "deferred_cleanup": False})
        self.assertEqual(self.target.read_bytes(), b"finished")
        self.assertEqual(visible_files(self.folder), {"generated.mp4"})

    @unittest.skipUnless(WINDOWS, "Windows sharing rules")
    def test_replaces_while_the_gallery_streams_the_original(self):
        # /api/v1/file opens videos this way; plain os.replace fails here.
        reader = win_safe_files._open_share_delete(str(self.target))
        try:
            with self.assertRaises(PermissionError):
                os.replace(self.source, self.target)
            result = safe_replace(self.source, self.target, timeout=2)
            self.assertTrue(result["replaced"])
            self.assertEqual(self.target.read_bytes(), b"finished")
            self.assertEqual(reader.read(), b"original")
        finally:
            reader.close()
        self.assertFalse(self.source.exists())
        self.assertEqual(visible_files(self.folder), {"generated.mp4"})

    @unittest.skipUnless(WINDOWS, "Windows sharing rules")
    def test_waits_for_a_brief_reader_without_share_delete(self):
        reader = open(self.target, "rb")
        threading.Timer(0.5, reader.close).start()
        started = time.monotonic()
        result = safe_replace(self.source, self.target, timeout=10, retry_delay=0.05)
        self.assertTrue(result["replaced"])
        self.assertGreaterEqual(time.monotonic() - started, 0.4)
        self.assertEqual(self.target.read_bytes(), b"finished")
        self.assertEqual(visible_files(self.folder), {"generated.mp4"})

    @unittest.skipUnless(WINDOWS, "Windows sharing rules")
    def test_gives_up_after_timeout_and_keeps_both_files(self):
        with open(self.target, "rb"):
            with self.assertRaises(PermissionError):
                safe_replace(self.source, self.target, timeout=0.3, retry_delay=0.05)
        self.assertEqual(self.target.read_bytes(), b"original")
        self.assertEqual(self.source.read_bytes(), b"finished")
        self.assertEqual(sorted(os.listdir(self.folder)),
                         ["generated.media-flow.mp4", "generated.mp4"])

    @unittest.skipUnless(WINDOWS, "fallback path is Windows-only")
    def test_restores_the_original_when_the_new_file_cannot_move_in(self):
        denied = PermissionError(13, "Access is denied")
        with patch.object(win_safe_files.os, "replace", side_effect=denied):
            with self.assertRaises(PermissionError):
                safe_replace(self.source, self.target, timeout=0.2, retry_delay=0.05)
        self.assertEqual(self.target.read_bytes(), b"original")
        self.assertEqual(self.source.read_bytes(), b"finished")
        self.assertEqual(sorted(os.listdir(self.folder)),
                         ["generated.media-flow.mp4", "generated.mp4"])

    @unittest.skipIf(WINDOWS, "POSIX behavior")
    def test_posix_permission_errors_are_not_retried(self):
        with patch.object(win_safe_files.os, "replace", side_effect=PermissionError(13, "denied")):
            started = time.monotonic()
            with self.assertRaises(PermissionError):
                safe_replace(self.source, self.target, timeout=5)
            self.assertLess(time.monotonic() - started, 1)


class GenerationFinishingTests(unittest.TestCase):
    @unittest.skipUnless(WINDOWS, "Windows sharing rules")
    def test_media_flow_result_replaces_a_video_the_browser_is_playing(self):
        from services.media_flow import process_generation_video

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "generated.mp4"
            destination = Path(folder) / "generated.media-flow.mp4"
            source.write_bytes(b"generated-original")

            def render(_source, output, **_kwargs):
                Path(output).write_bytes(b"finished-48fps")
                return {"frames": 2}

            reader = win_safe_files._open_share_delete(str(source))
            try:
                with patch("services.media_flow.process_video", side_effect=render):
                    result = process_generation_video(source, destination, temporal="rife2")
            finally:
                reader.close()
            self.assertEqual(result, {"frames": 2})
            self.assertEqual(source.read_bytes(), b"finished-48fps")
            self.assertFalse(destination.exists())
            self.assertEqual(visible_files(Path(folder)), {"generated.mp4"})


if __name__ == "__main__":
    unittest.main()
