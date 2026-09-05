from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))


class H3CharacterLibraryTests(unittest.TestCase):
    def test_character_library_copies_media_and_returns_runtime_urls(self):
        from services import character_library

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                uploads = Path(directory) / "uploads"
                uploads.mkdir()
                visual = uploads / "portrait.png"
                voice = uploads / "voice.wav"
                visual.write_bytes(b"png")
                voice.write_bytes(b"wav")
                with mock.patch.object(character_library, "_probe_duration", return_value=4.0):
                    saved = character_library.create_character(
                        name="  Blaine  ",
                        visual_path=str(visual),
                        visual_type="image",
                        voice_path=str(voice),
                    )

                self.assertEqual(saved["name"], "Blaine")
                self.assertTrue(Path(saved["visual"]["path"]).is_file())
                self.assertTrue(Path(saved["voice"]["path"]).is_file())
                self.assertIn("/media/visual", saved["visual"]["url"])
                self.assertEqual(character_library.list_characters()[0]["id"], saved["id"])
                self.assertTrue(character_library.delete_character(saved["id"]))
                self.assertEqual(character_library.list_characters(), [])
            finally:
                os.chdir(original_cwd)

    def test_character_sources_must_come_from_uploads(self):
        from services import character_library

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                outside = Path(directory) / "portrait.png"
                outside.write_bytes(b"png")
                with self.assertRaisesRegex(ValueError, "uploaded to Maestro"):
                    character_library.create_character(
                        name="Person",
                        visual_path=str(outside),
                        visual_type="image",
                    )
            finally:
                os.chdir(original_cwd)


class H3ReferenceDurationTests(unittest.TestCase):
    def test_water_filling_matches_h3_total_budget(self):
        from models.minimax_h3.reference_media import allocate_reference_durations

        self.assertEqual(allocate_reference_durations([10, 10, 10]), [5.0, 5.0, 5.0])
        self.assertEqual(allocate_reference_durations([2, 10, 10]), [2.0, 6.5, 6.5])
        self.assertEqual(allocate_reference_durations([4, 5]), [4.0, 5.0])
        with self.assertRaisesRegex(ValueError, "at least 2 seconds"):
            allocate_reference_durations([1.5, 10])

    def test_manifest_uses_cached_derivatives_without_trimming_drive_audio(self):
        from models.minimax_h3 import reference_media

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                uploads = Path(directory) / "uploads"
                uploads.mkdir()
                videos = []
                for index in range(3):
                    path = uploads / f"character-{index}.mp4"
                    path.write_bytes(b"video")
                    videos.append(path)
                drive = uploads / "song.wav"
                drive.write_bytes(b"audio")

                durations = {str(path): 10.0 for path in videos}
                durations[str(drive)] = 120.0
                with mock.patch.object(
                    reference_media,
                    "_probe_duration",
                    side_effect=lambda path: durations[str(path)],
                ), mock.patch.object(
                    reference_media,
                    "_trim_media",
                    side_effect=lambda source, duration, kind, **kwargs: f"{source}.trim-{duration}",
                ) as trim:
                    result = reference_media.normalize_reference_manifest([
                        *[
                            {
                                "type": "video",
                                "path": str(path),
                                "include_audio": False,
                                "video_intent": "character",
                            }
                            for path in videos
                        ],
                        {
                            "type": "audio",
                            "path": str(drive),
                            "audio_intent": "drive",
                        },
                    ])

                self.assertEqual(
                    [item.get("effective_duration_seconds") for item in result[:3]],
                    [5.0, 5.0, 5.0],
                )
                self.assertEqual(result[3]["path"], str(drive))
                self.assertEqual(trim.call_count, 3)
            finally:
                os.chdir(original_cwd)
