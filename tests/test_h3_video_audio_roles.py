"""Video source-audio roles normalize into existing H3 audio behavior."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3.reference_manifest import (
    apply_exact_drive_audio_prompt_contract,
    split_exact_drive_audio_reference,
    validate_reference_manifest,
)
from services.h3_performance_audio import (
    has_h3_performance_audio,
    performance_audio_duration_seconds,
)
from services.h3_reference_audio import (
    canonicalize_video_audio_roles,
    measure_video_audio_metadata,
    normalize_video_audio_role,
    video_audio_contributes_reference_ordinal,
    video_drive_prompt_context,
)


class H3VideoAudioRoleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "source.mp4"
        self.attached = self.root / "attached.wav"
        self.cached = self.root / "cached.wav"
        self.video.touch()
        self.attached.touch()
        self.cached.touch()

    def tearDown(self):
        self.temp.cleanup()

    def _video_ref(self, **kwargs):
        return {
            "id": "video-1",
            "type": "video",
            "path": str(self.video),
            "filename": "source.mp4",
            "has_audio": True,
            "duration_seconds": 31.0,
            **kwargs,
        }

    def test_legacy_audio_role_preserves_saved_checkbox_behavior(self):
        enabled = self._video_ref()
        disabled = self._video_ref(include_audio=False)
        self.assertEqual(normalize_video_audio_role(enabled), "legacy")
        self.assertEqual(normalize_video_audio_role(disabled), "ignore")
        canonical = canonicalize_video_audio_roles([enabled])
        self.assertEqual(len(canonical), 1)
        self.assertTrue(canonical[0]["include_audio"])
        normalized = validate_reference_manifest([enabled, disabled], require_files=False)
        self.assertEqual(normalized[0]["include_audio"], True)
        self.assertEqual(normalized[1]["include_audio"], False)

    def test_explicit_audio_roles_make_one_standalone_row_and_keep_input_unchanged(self):
        source = self._video_ref(
            video_audio_role="VOICE",
            audio_path=str(self.attached),
            audio_filename="external.wav",
        )
        original = dict(source)

        def probe(path):
            if path == str(self.attached):
                return True, 7.25
            self.fail(f"unexpected audio probe: {path}")

        with patch("services.h3_reference_audio._probe_audio", side_effect=probe):
            first = canonicalize_video_audio_roles([source])
            second = canonicalize_video_audio_roles(first)

        self.assertEqual(source, original)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0]["video_audio_role"], "voice")
        self.assertFalse(first[0]["include_audio"])
        self.assertEqual(first[0]["audio_duration_seconds"], 7.25)
        self.assertEqual(first[1]["audio_intent"], "voice")
        self.assertEqual(first[1]["path"], str(self.attached))
        self.assertEqual(first[1]["duration_seconds"], 7.25)
        self.assertEqual(first, second)

    def test_canonical_video_roles_do_not_duplicate_audio_ordinals(self):
        voice = self._video_ref(
            id="voice-video",
            video_audio_role="voice",
            audio_path=str(self.attached),
        )
        style = self._video_ref(
            id="style-video",
            video_audio_role="style",
            audio_path=str(self.attached),
        )
        standalone = {"id": "standalone", "type": "audio", "path": str(self.attached)}

        # Before conversion, each explicit typed video owns its one manifest
        # audio slot. After conversion, each derived audio row owns that slot.
        self.assertTrue(video_audio_contributes_reference_ordinal(voice))
        self.assertTrue(video_audio_contributes_reference_ordinal(style))
        with patch(
            "services.h3_reference_audio._probe_audio",
            return_value=(True, 5.0),
        ):
            canonical = canonicalize_video_audio_roles([voice, style, standalone])

        self.assertFalse(video_audio_contributes_reference_ordinal(canonical[0]))
        self.assertFalse(video_audio_contributes_reference_ordinal(canonical[2]))

        # Import only the pure ordinal function from ref2va.py so this CPU
        # test does not initialize the model's heavyweight runtime imports.
        ref2va_path = APP / "models/minimax_h3/ref2va.py"
        tree = ast.parse(ref2va_path.read_text(encoding="utf-8"), filename=str(ref2va_path))
        ordinal_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_ref2va_audio_ordinals_by_item"
        )
        namespace = {
            "video_audio_contributes_reference_ordinal": video_audio_contributes_reference_ordinal,
        }
        exec(
            compile(ast.Module(body=[ordinal_function], type_ignores=[]), str(ref2va_path), "exec"),
            namespace,
        )
        labels = namespace["_ref2va_audio_ordinals_by_item"](canonical)

        self.assertEqual([item["type"] for item in canonical], ["video", "audio", "video", "audio", "audio"])
        self.assertEqual(labels, {1: 1, 3: 2, 4: 3})

    def test_embedded_audio_with_unknown_container_duration_uses_decoded_audio_duration(self):
        source = self._video_ref(video_audio_role="drive")

        def probe(path):
            if path == str(self.video):
                return True, None
            if path == str(self.cached):
                return True, 9.5
            self.fail(f"unexpected audio probe: {path}")

        with (
            patch("services.h3_reference_audio._probe_audio", side_effect=probe),
            patch("services.h3_reference_audio._extract_full_audio", return_value=str(self.cached)),
        ):
            canonical = canonicalize_video_audio_roles([source])

        self.assertEqual(canonical[0]["audio_duration_seconds"], 9.5)
        self.assertEqual(canonical[1]["duration_seconds"], 9.5)
        self.assertEqual(source["duration_seconds"], 31.0)
        self.assertEqual(canonical[1]["audio_intent"], "drive")

    def test_measure_helper_copies_only_audio_duration_and_never_derived_rows(self):
        source = self._video_ref(
            video_audio_role="style",
            include_audio=False,
            audio_path=str(self.attached),
            _ui_custom_value="keep",
        )
        original = dict(source)
        with patch(
            "services.h3_reference_audio._probe_audio",
            return_value=(True, 5.125),
        ):
            measured = measure_video_audio_metadata([source])

        self.assertEqual(source, original)
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0]["audio_duration_seconds"], 5.125)
        self.assertEqual(measured[0]["duration_seconds"], 31.0)
        self.assertEqual(measured[0]["video_audio_role"], "style")
        self.assertEqual(measured[0]["include_audio"], False)
        self.assertEqual(measured[0]["audio_path"], str(self.attached))
        self.assertEqual(measured[0]["_ui_custom_value"], "keep")
        self.assertNotIn("_h3_video_audio_source_id", measured[0])

    def test_unavailable_audio_and_encoded_refmod_errors_are_actionable(self):
        no_audio = self._video_ref(has_audio=False, video_audio_role="style")
        with self.assertRaisesRegex(ValueError, "has no audio track"):
            canonicalize_video_audio_roles([no_audio])

        encoded = self._video_ref(refmod_path=str(self.root / "encoded.refmod"), video_audio_role="sound")
        encoded.pop("has_audio")
        with self.assertRaisesRegex(ValueError, "encoded RefMod video"):
            canonicalize_video_audio_roles([encoded])

    def test_exact_video_drive_routes_once_without_owning_audio_ordinal(self):
        legacy = self._video_ref(id="legacy", include_audio=True)
        drive = self._video_ref(
            id="drive",
            video_audio_role="drive",
            audio_path=str(self.attached),
        )
        with patch(
            "services.h3_reference_audio._probe_audio",
            return_value=(True, 6.0),
        ):
            runtime, drive_path, ordinal = split_exact_drive_audio_reference([legacy, drive])
            normalized_again = canonicalize_video_audio_roles(runtime)

        self.assertEqual(drive_path, str(self.attached))
        self.assertIsNone(ordinal)
        self.assertEqual([item["type"] for item in runtime], ["video", "video"])
        self.assertTrue(runtime[1]["_h3_video_audio_role_routed"])
        self.assertFalse(video_audio_contributes_reference_ordinal(runtime[1]))
        self.assertEqual(normalized_again, runtime)

    def test_video_drive_does_not_rewrite_voice_audio_one_but_standalone_drive_still_does(self):
        drive_video = self._video_ref(
            id="drive-video",
            video_audio_role="drive",
            audio_path=str(self.attached),
        )
        voice = {"type": "audio", "path": str(self.attached), "audio_intent": "voice"}
        with patch(
            "services.h3_reference_audio._probe_audio",
            return_value=(True, 6.0),
        ):
            runtime, drive_path, ordinal = split_exact_drive_audio_reference([drive_video, voice])

        prompt = "<Audio 1> supplies the voice."
        rewritten = apply_exact_drive_audio_prompt_contract(prompt, ordinal)
        self.assertEqual(drive_path, str(self.attached))
        self.assertIsNone(ordinal)
        self.assertEqual([item["type"] for item in runtime], ["video", "audio"])
        self.assertEqual(runtime[1], voice)
        self.assertIn("<Audio 1> supplies the voice.", rewritten)

        standalone_drive = {"type": "audio", "path": str(self.attached), "audio_intent": "drive"}
        runtime, drive_path, ordinal = split_exact_drive_audio_reference([standalone_drive, voice])
        rewritten = apply_exact_drive_audio_prompt_contract(
            "<Audio 1> drives the song; <Audio 2> supplies the voice.",
            ordinal,
        )
        self.assertEqual(drive_path, str(self.attached))
        self.assertEqual(ordinal, 1)
        self.assertEqual(runtime, [voice])
        self.assertIn("the exact target soundtrack drives the song", rewritten)
        self.assertIn("<Audio 1> supplies the voice", rewritten)

    def test_one_exact_driver_limit_includes_video_and_standalone_drives(self):
        drive = self._video_ref(video_audio_role="drive", audio_path=str(self.attached))
        standalone = {
            "type": "audio",
            "path": str(self.attached),
            "audio_intent": "drive",
        }
        with patch(
            "services.h3_reference_audio._probe_audio",
            return_value=(True, 6.0),
        ):
            with self.assertRaisesRegex(ValueError, "one Music / performance timeline"):
                split_exact_drive_audio_reference([drive, standalone])

    def test_video_drive_context_is_recognized_as_an_exact_performance_clock(self):
        relationships = "\n".join(video_drive_prompt_context(
            self._video_ref(video_audio_role="drive", audio_duration_seconds=4.75),
            video_index=1,
            role="the source performance",
        ))
        self.assertIn("The exact target soundtrack supplies", relationships)
        self.assertIn("H3_PERFORMANCE_AUDIO_CLOCK duration_seconds=4.75", relationships)
        self.assertTrue(has_h3_performance_audio(relationships))
        self.assertEqual(performance_audio_duration_seconds(relationships), 4.75)


if __name__ == "__main__":
    unittest.main()
