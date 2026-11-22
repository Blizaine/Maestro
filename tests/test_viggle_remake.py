"""Remake's bounded source edit keeps selected timing and source audio."""
from contextlib import contextmanager
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.minimax_h3 import viggle_remake


class ViggleRemakeTests(unittest.TestCase):
    def test_model_contract_keeps_source_audio_and_reviewable_frame_refs(self):
        definition = viggle_remake.model_definition(
            {"fps": 24, "sliding_window_defaults": {}}, "swap",
        )

        self.assertEqual(definition["set_video_prompt_type"], "IVU")
        self.assertEqual(definition["image_ref_choices"]["default"], "I")
        self.assertTrue(definition["output_audio_is_input_audio"])
        self.assertEqual(definition["fps"], viggle_remake.FPS)
        self.assertNotIn("VU2", str(definition.get("guide_custom_choices", {})))

    def test_preparation_uses_absolute_keyframe_times_and_same_trim_for_audio(self):
        body = {
            "video_guide": "source.mp4",
            "image_refs": ["edited.png"],
            "viggle_remake_frame_times": [2.25],
            "viggle_trim_start": 1.0,
            "viggle_trim_end": 4.75,
            "prompt": "Replace the performer with the edited character.",
        }
        observed = {}

        @contextmanager
        def prepare(settings, **_kwargs):
            observed.update(settings)
            yield {"video_guide": "trimmed-source.mp4"}

        with patch(
            "services.viggle_media.probe_control_video",
            return_value={"duration_seconds": 6.0},
        ), patch(
            "services.viggle_media.prepare_control_media",
            side_effect=prepare,
        ), patch(
            "shared.utils.audio_video.extract_audio_tracks",
            return_value=["source-audio.wav"],
        ):
            with viggle_remake.prepare_remake_media(
                body,
                "swap",
                resolve_media=lambda path: path,
                aborted=lambda: False,
                update=lambda _message: None,
            ) as prepared:
                self.assertEqual(prepared["video_guide"], "trimmed-source.mp4")

        self.assertEqual(observed["_viggle_trim_start"], 1.0)
        self.assertEqual(observed["_viggle_trim_end"], 4.75)
        self.assertEqual(observed["video_length"], 90)
        self.assertEqual(observed["audio_prompt_type"], "K")
        self.assertEqual(observed["video_prompt_type"], "IVU")
        self.assertIn("<Picture 1> is an edited target frame at 1.250 seconds.", observed["prompt"])
        self.assertIn("The source soundtrack, unchanged.", observed["prompt"])

    def test_frame_timestamps_must_be_strictly_ordered(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            viggle_remake.frame_times({"viggle_remake_frame_times": [2.0, 2.0]}, 2)

    def test_aligned_swap_uses_ref2va_and_native_recipe(self):
        definition = viggle_remake.model_definition(
            {
                "fps": 24,
                "sliding_window_defaults": {},
                "omni_reference": True,
                "minimax_h3_lora_workflow": "ref2va",
                "num_inference_steps": 20,
                "inference_steps_min": 1,
                "inference_steps_max": 50,
            },
            "swap_aligned",
        )
        self.assertTrue(definition["minimax_h3_aligned_guide"])
        self.assertTrue(definition["omni_reference"])
        self.assertEqual(definition["minimax_h3_viggle_remake"], "swap")
        self.assertEqual(definition["minimax_h3_lora_workflow"], "ref2va")
        self.assertEqual(definition["set_video_prompt_type"], "IVU2")
        self.assertFalse(definition["lock_inference_steps"])
        self.assertEqual(definition["inference_steps_min"], 2)
        self.assertEqual(definition["inference_steps_max"], 50)
        self.assertFalse(definition["minimax_h3_two_pass_compatible"])
        self.assertTrue(definition["loras_disabled"])

    def test_aligned_request_needs_one_identity_and_pins_adapter(self):
        request = {
            "video_guide": "source.mp4",
            "image_refs": ["identity.png"],
            "viggle_trim_start": 1.0,
            "viggle_trim_end": 5.0,
            "prompt": "",
            "num_inference_steps": 20,
        }
        viggle_remake.normalize_request(request, aligned_guide=True)
        self.assertEqual(request["video_prompt_type"], "IVU2")
        self.assertEqual(request["activated_loras"], [viggle_remake.ALIGNED_GUIDE_LORA_URL])
        self.assertEqual(request["loras_multipliers"], "1.0")
        self.assertEqual(request["num_inference_steps"], 20)
        self.assertTrue(request["prompt"])
        self.assertNotIn("viggle_remake_frame_times", request)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            viggle_remake.validate_remake_inputs(
                {**request, "image_refs": ["one.png", "two.png"]},
                aligned_guide=True,
            )

    def test_aligned_prompt_and_references_hide_source_video(self):
        import torch
        from PIL import Image

        source = torch.zeros((3, 22, 64, 64), dtype=torch.float32)
        refs = viggle_remake.prepare_references(
            source,
            [Image.new("RGB", (64, 64), "white")],
            0,
            22,
            64,
            64,
            aligned_guide=True,
        )
        presentation = viggle_remake.text_presentation(
            "Replace the performer.", 22, 1, aligned_guide=True,
        )
        self.assertEqual([reference.kind for reference in refs], ["image"])
        self.assertIn("<Picture 1>", presentation)
        self.assertNotIn("<Video 1>", presentation)

    def test_aligned_adapter_pin_verifies_exact_bytes(self):
        import hashlib
        from tempfile import TemporaryDirectory
        from unittest.mock import patch as mock_patch

        payload = b"test aligned adapter bytes"
        with TemporaryDirectory() as directory:
            path = Path(directory) / viggle_remake.ALIGNED_GUIDE_LORA_FILE
            path.write_bytes(payload)
            with mock_patch.object(viggle_remake, "ALIGNED_GUIDE_LORA_SIZE", len(payload)), \
                    mock_patch.object(viggle_remake, "ALIGNED_GUIDE_LORA_SHA256", hashlib.sha256(payload).hexdigest()):
                self.assertEqual(
                    Path(viggle_remake.validate_aligned_guide_loras([path])),
                    path.resolve(),
                )
                path.write_bytes(payload + b"tamper")
                with self.assertRaisesRegex(ValueError, "size differs"):
                    viggle_remake.validate_aligned_guide_loras([path])


if __name__ == "__main__":
    unittest.main()
