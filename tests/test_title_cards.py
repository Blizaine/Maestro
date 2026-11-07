"""Model-free validation and CPU rendering checks for title cards."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.services.title_cards import (
    TitleCardRenderError,
    _font_for_text,
    _resolve_ffmpeg,
    _resolve_ffprobe,
    _wrap_paragraph,
    normalize_title_card,
    render_title_card,
)


class TitleCardValidationTests(unittest.TestCase):
    def test_normalize_preserves_exact_copy_and_fills_defaults(self):
        text = "  Café — first line\nsecond line  "
        card = normalize_title_card({"text": text, "role": "opening", "background_color": "#0a1b2c"})

        self.assertEqual(card["text"], text)
        self.assertEqual(card["subtitle"], "")
        self.assertEqual(card["role"], "opening")
        self.assertEqual(card["style"], "classic")
        self.assertEqual(card["background_color"], "#0A1B2C")
        self.assertEqual(card["text_color"], "#FFFFFF")
        self.assertEqual(card["duration_seconds"], 3.0)

    def test_rejects_unknown_fields_and_invalid_enums_or_colors(self):
        cases = (
            ({"text": "Title", "duration": 4}, "Unsupported title-card field"),
            ({"text": "Title", "role": "middle"}, "role must"),
            ({"text": "Title", "style": "neon"}, "style must"),
            ({"text": "Title", "background_color": "red"}, "background_color must"),
            ({"text": "Title", "text_color": None}, "text_color must"),
        )
        for value, message in cases:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                normalize_title_card(value)

    def test_rejects_blank_or_overlong_copy_and_bad_duration(self):
        cases = (
            ({"text": " \n "}, "visible text"),
            ({"text": "x" * 501}, "at most 500"),
            ({"text": "Title", "subtitle": "x" * 1001}, "at most 1000"),
            ({"text": "Title", "duration_seconds": True}, "finite"),
            ({"text": "Title", "duration_seconds": float("nan")}, "finite"),
            ({"text": "Title", "duration_seconds": 0.49}, "between 0.5 and 15"),
            ({"text": "Title", "duration_seconds": 15.01}, "between 0.5 and 15"),
        )
        for value, message in cases:
            with self.subTest(value=str(value)[:50]), self.assertRaisesRegex(ValueError, message):
                normalize_title_card(value)


class TitleCardLayoutTests(unittest.TestCase):
    def test_wraps_long_words_newlines_and_unicode_without_dropping_characters(self):
        text = "Überlongwordwithoutspaces—repeat\nCafe\u0301 déjà vu"
        font = _font_for_text("minimal", text, 24)
        lines = []
        for paragraph in text.split("\n"):
            wrapped = _wrap_paragraph(paragraph, font, 100)
            self.assertEqual("".join(wrapped), paragraph)
            lines.extend(wrapped)
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        self.assertGreater(len(lines), 2)
        self.assertTrue(all(draw.textlength(line, font=font) <= 100 for line in lines))

    def test_reports_when_no_available_font_can_render_copy(self):
        with self.assertRaises(TitleCardRenderError):
            _font_for_text("minimal", "Title\ufffe", 24)


class TitleCardRenderTests(unittest.TestCase):
    def test_render_is_atomic_and_decodes_with_exact_frame_count_when_ffmpeg_exists(self):
        try:
            ffmpeg = _resolve_ffmpeg(None)
        except TitleCardRenderError:
            self.skipTest("No configured or bundled FFmpeg is available to this test.")
        encoder = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
            check=False,
        )
        if encoder.returncode != 0 or "libx264" not in encoder.stdout:
            self.skipTest("FFmpeg is present but does not include the software libx264 encoder.")

        ffprobe = _resolve_ffprobe(ffmpeg)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artwork = root / "artwork.png"
            output = root / "card.mp4"
            original = b"existing output stays until success"
            output.write_bytes(original)
            Image.new("RGB", (320, 200), "navy").save(artwork)

            result = render_title_card(
                {"text": "Café\nA very longwordthatmustwrap", "subtitle": "A subtitle", "duration_seconds": 0.5},
                output,
                width=320,
                height=240,
                fps=12,
                image_path=artwork,
                ffmpeg_path=ffmpeg,
            )

            self.assertEqual(output.read_bytes()[4:8], b"ftyp")
            self.assertNotEqual(output.read_bytes(), original)
            self.assertEqual(result["frame_count"], 6)
            self.assertEqual((result["width"], result["height"], result["fps"]), (320, 240, 12.0))
            self.assertAlmostEqual(result["duration"], 0.5)
            self.assertEqual(result["card"]["text"], "Café\nA very longwordthatmustwrap")
            self.assertTrue(any("center-cropped" in warning for warning in result["warnings"]))

            decoded = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(output), "-f", "null", "-"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            self.assertEqual(decoded.returncode, 0, decoded.stderr.decode("utf-8", "replace"))

            if ffprobe:
                probe = subprocess.run(
                    [ffprobe, "-v", "error", "-count_frames", "-show_entries", "stream=codec_type,codec_name,nb_read_frames,width,height", "-of", "json", str(output)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(probe.returncode, 0, probe.stderr)
                streams = json.loads(probe.stdout)["streams"]
                video = next(stream for stream in streams if stream["codec_type"] == "video")
                audio = next(stream for stream in streams if stream["codec_type"] == "audio")
                self.assertEqual(video["codec_name"], "h264")
                self.assertEqual(int(video["nb_read_frames"]), 6)
                self.assertEqual((int(video["width"]), int(video["height"])), (320, 240))
                self.assertEqual(audio["codec_name"], "aac")

    def test_invalid_dimensions_are_rejected_before_render(self):
        for width, height in ((319, 240), (64, 4098)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                render_title_card(
                    {"text": "Title"},
                    "unused.mp4",
                    width=width,
                    height=height,
                    ffmpeg_path="ffmpeg-not-needed-for-invalid-dimensions",
                )

    def test_failed_ffmpeg_render_keeps_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing.mp4"
            original = b"keep this output on failure"
            output.write_bytes(original)
            with patch(
                "app.services.title_cards.subprocess.run",
                return_value=SimpleNamespace(returncode=1, stderr="encoder failed"),
            ):
                with self.assertRaisesRegex(TitleCardRenderError, "Could not render title card"):
                    render_title_card(
                        {"text": "A\nsmall title"},
                        output,
                        width=320,
                        height=240,
                        fps=12,
                        ffmpeg_path=sys.executable,
                    )
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(Path(temporary).glob("*.part.*")), [])


if __name__ == "__main__":
    unittest.main()
