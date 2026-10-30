"""Pika H3 Flex defaults and adapter pins stay separate from other H3 recipes."""
import inspect
# ruff: noqa: E402 -- app imports follow the repository path bootstrap.
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.minimax_h3 import fasth3
from models.minimax_h3 import pika_flex_assets as assets


class PikaFlexCatalogueTests(unittest.TestCase):
    def _definition(self, filename):
        return json.loads((ROOT / "app" / "defaults" / filename).read_text())

    def test_definitions_pin_distinct_managed_adapters_on_approved_base_urls(self):
        base_frames = self._definition("minimax_h3.json")["model"]["URLs"]
        base_references = self._definition("minimax_h3_ref2va.json")["model"]["URLs"]
        cases = (
            (
                "minimax_h3_pika_flex.json",
                False,
                "minimax_h3",
                assets.PIKA_FLEX_FRAMES_FILE,
                assets.PIKA_FLEX_FRAMES_SHA256,
                base_frames,
            ),
            (
                "minimax_h3_ref2va_pika_flex.json",
                True,
                "minimax_h3_ref2va",
                assets.PIKA_FLEX_REFERENCES_FILE,
                assets.PIKA_FLEX_REFERENCES_SHA256,
                base_references,
            ),
        )
        for filename, reference, architecture, adapter, digest, base_urls in cases:
            with self.subTest(filename=filename):
                model = self._definition(filename)["model"]
                frames_caps = assets.pika_flex_capabilities(False)
                references_caps = assets.pika_flex_capabilities(True)
                pin = assets.PIKA_FLEX_LORA_SPECS[reference]
                self.assertEqual(model["architecture"], architecture)
                self.assertTrue(model["minimax_h3_pika_flex"])
                self.assertTrue(model["experimental"])
                self.assertEqual(model["minimax_h3_pika_flex_reference"], reference)
                self.assertEqual(model["URLs"], [base_urls[-1]])
                self.assertIn("Comfy-Org/MiniMax-H3", model["URLs"][0])
                self.assertIn("minimax_h3_fl2va_pruned_fp8_scaled.safetensors" if not reference else "minimax_h3_ref2va_pruned_fp8_scaled.safetensors", model["URLs"][0])
                self.assertNotIn("minimax_h3_qkv_layout", model)
                self.assertEqual(model["loras"], [pin["url"]])
                self.assertEqual(Path(model["loras"][0]).name, adapter)
                self.assertIn(assets.PIKA_FLEX_REVISION, model["loras"][0])
                self.assertEqual(model["pika_flex_adapter_file"], adapter)
                self.assertEqual(model["pika_flex_adapter_size_bytes"], assets.PIKA_FLEX_LORA_SIZE)
                self.assertEqual(model["pika_flex_adapter_sha256"], digest)
                self.assertEqual(model["pika_flex_adapter_repo"], assets.PIKA_FLEX_REPO)
                self.assertEqual(model["pika_flex_adapter_revision"], assets.PIKA_FLEX_REVISION)
                self.assertEqual(model["pika_flex_base_model_repo"], assets.PIKA_FLEX_MODEL_REPO)
                self.assertEqual(model["pika_flex_base_model_revision"], assets.PIKA_FLEX_MODEL_REVISION)
                base_pin = assets.PIKA_FLEX_CHECKPOINT_SPECS[reference]
                self.assertEqual(model["pika_flex_checkpoint_repo"], assets.PIKA_FLEX_BASE_REPO)
                self.assertEqual(model["pika_flex_checkpoint_revision"], assets.PIKA_FLEX_BASE_REVISION)
                self.assertEqual(model["pika_flex_checkpoint_file"], base_pin["file"])
                self.assertEqual(model["pika_flex_checkpoint_size_bytes"], assets.PIKA_FLEX_BASE_SIZE)
                self.assertEqual(model["pika_flex_checkpoint_sha256"], base_pin["sha256"])
                self.assertEqual(model["loras_multipliers"], ["1"])
                self.assertTrue(model["loras_disabled"])
                self.assertTrue(model["lock_inference_steps"])
                self.assertTrue(model["lock_guidance_scale"])
                self.assertEqual(model["inference_steps_min"], assets.PIKA_FLEX_STEPS)
                self.assertEqual(model["inference_steps_max"], assets.PIKA_FLEX_STEPS)
                self.assertEqual(model["minimax_h3_sampler"], "euler")
                self.assertEqual(model["minimax_h3_video_shift"], 12.0)
                self.assertEqual(model["minimax_h3_audio_shift"], 3.0)
                self.assertEqual(model["minimax_h3_video_vae_filename"], "minimax_h3_video_vae_int8_convrot.safetensors")
                self.assertEqual(model["minimax_h3_text_encoder_default"], "gguf_q2_k")
                self.assertEqual(model["fps"], 24)
                self.assertEqual(model["frames_minimum"], 124)
                self.assertEqual(model["frames_steps"], 17)
                self.assertEqual(model["frames_maximum"], 362)
                self.assertEqual(model["resolution_preset_order"], ["480p", "768p"])
                for tier in assets.PIKA_FLEX_RESOLUTION_BUCKETS:
                    self.assertEqual(tuple(model["resolution_presets"][tier]["values"].values()), assets.PIKA_FLEX_RESOLUTION_BUCKETS[tier])
                    self.assertEqual(tuple(model["resolution_presets"][tier]["values"].keys()), tuple(frames_caps["resolution_presets"][tier]["values"].keys()) if not reference else tuple(references_caps["resolution_presets"][tier]["values"].keys()))
                self.assertEqual(model["license_name"], "MiniMax H3 Community License")
                for key in (
                    "minimax_h3_turbo_mode",
                    "minimax_h3_fasth3",
                    "minimax_h3_fasth3_omniref",
                    "minimax_h3_fused_turbo",
                    "minimax_h3_ptd_refasymopd",
                    "minimax_h3_vdn",
                    "minimax_h3_singularity",
                    "minimax_h3_dasiwa",
                ):
                    self.assertFalse(model.get(key, False), key)

    def test_catalog_overrides_preserve_continuation_capabilities(self):
        for filename, reference in (
            ("minimax_h3_pika_flex.json", False),
            ("minimax_h3_ref2va_pika_flex.json", True),
        ):
            with self.subTest(filename=filename):
                definition = self._definition(filename)
                # Model metadata overrides family capabilities in the runtime
                # catalogue, so these two layers must agree.
                effective = {**assets.pika_flex_capabilities(reference), **definition["model"]}
                self.assertTrue(effective["sliding_window"])
                self.assertTrue(effective["video_continuation"])
                self.assertTrue(effective["sliding_window_audio_history"])
                self.assertEqual(definition["sliding_window_overlap"], 18)
                self.assertFalse(definition["minimax_h3_multi_window"])
                self.assertFalse(definition["minimax_h3_reference_sequence"])

    def test_capabilities_and_defaults_keep_frames_and_references_narrow(self):
        frames = assets.pika_flex_capabilities(False)
        references = assets.pika_flex_capabilities(True)
        for capabilities in (frames, references):
            self.assertTrue(capabilities["minimax_h3_pika_flex"])
            self.assertTrue(capabilities["lock_inference_steps"])
            self.assertEqual(capabilities["inference_steps_min"], 6)
            self.assertEqual(capabilities["inference_steps_max"], 6)
            self.assertTrue(capabilities["lock_guidance_scale"])
            self.assertEqual(capabilities["minimax_h3_sampler"], "euler")
            self.assertEqual(capabilities["minimax_h3_video_shift"], 12.0)
            self.assertEqual(capabilities["minimax_h3_audio_shift"], 3.0)
            self.assertTrue(capabilities["loras_disabled"])
            self.assertEqual(capabilities["compatible_model_paths"], {})
            self.assertEqual(capabilities["compatible_model_qkv_layouts"], {})
            self.assertNotIn("minimax_h3_qkv_layout", capabilities)
            self.assertTrue(capabilities["sliding_window"])
            self.assertTrue(capabilities["video_continuation"])
            self.assertTrue(capabilities["sliding_window_exact_total_frames"])
            self.assertTrue(capabilities["sliding_window_trim_to_requested"])
            self.assertTrue(capabilities["sliding_window_audio_history"])

            self.assertFalse(capabilities["first_block_cache"])
            self.assertTrue(capabilities["sla_attention"])
            self.assertTrue(capabilities["sla_attention_experimental"])
            self.assertTrue(capabilities["sol_attention"])
            self.assertFalse(capabilities["veda_attention"])
            self.assertEqual(tuple(capabilities["resolution_presets"]["480p"]["values"].values()), assets.PIKA_FLEX_RESOLUTION_BUCKETS["480p"])
            self.assertEqual(tuple(capabilities["resolution_presets"]["768p"]["values"].values()), assets.PIKA_FLEX_RESOLUTION_BUCKETS["768p"])
            self.assertEqual(capabilities["minimax_h3_pika_flex_knots"], (1000, 750, 500, 250, 100, 40, 0))
            self.assertEqual(capabilities["minimax_h3_pika_flex_lora_strength"], 1.0)

        self.assertEqual(frames["image_prompt_types_allowed"], "TSE")
        self.assertTrue(frames["i2v_class"])
        self.assertTrue(frames["minimax_h3_multi_window"])
        self.assertFalse(frames["minimax_h3_reference_sequence"])
        self.assertFalse(frames["omni_reference"])
        self.assertFalse(frames["custom_frames_injection"])
        self.assertEqual(references["image_prompt_types_allowed"], "")
        self.assertFalse(references["i2v_class"])
        self.assertFalse(references["minimax_h3_multi_window"])
        self.assertTrue(references["minimax_h3_reference_sequence"])
        self.assertTrue(references["omni_reference"])
        self.assertTrue(references["supports_reference_audio"])
        self.assertEqual(references["omni_reference_limits"], {"image": 9, "video": 3, "audio": 3, "total": 12})

        defaults = assets.pika_flex_defaults(False)
        reference_defaults = assets.pika_flex_defaults(True)
        self.assertNotIn("video_length", defaults)
        self.assertNotIn("resolution", defaults)
        self.assertEqual(assets.PIKA_FLEX_DEFAULT_RESOLUTION, "832x480")
        frames_json = self._definition("minimax_h3_pika_flex.json")
        self.assertEqual(frames_json["video_length"], 124)
        self.assertEqual(frames_json["resolution"], "832x480")
        self.assertEqual(frames_json["image_prompt_type"], "")
        self.assertEqual(frames_json["video_prompt_type"], "")
        self.assertEqual(frames_json["audio_prompt_type"], "")
        references_json = self._definition("minimax_h3_ref2va_pika_flex.json")
        self.assertEqual(references_json["minimax_h3_references"], [])
        self.assertEqual(references_json["minimax_h3_reference_detail"], "match")
        self.assertEqual(references_json["image_prompt_type"], "")
        self.assertEqual(defaults["num_inference_steps"], 6)
        self.assertEqual(defaults["guidance_scale"], 1.0)
        self.assertEqual(defaults["flow_shift"], 12.0)
        self.assertEqual(defaults["audio_flow_shift"], 3.0)
        self.assertNotIn("minimax_h3_references", defaults)
        self.assertNotIn("minimax_h3_reference_detail", reference_defaults)
        self.assertNotIn("image_prompt_type", defaults)
        self.assertNotIn("video_prompt_type", defaults)
        self.assertNotIn("audio_prompt_type", defaults)
        self.assertFalse(defaults["minimax_h3_multi_window"])
        self.assertFalse(defaults["minimax_h3_reference_sequence"])
        self.assertEqual(list(inspect.signature(assets.pika_flex_capabilities).parameters), ["reference"])
        self.assertEqual(list(inspect.signature(assets.pika_flex_defaults).parameters), ["reference", "attention"])
        self.assertEqual(list(inspect.signature(assets.verify_pika_flex_loras).parameters), ["paths", "reference"])
        self.assertEqual(list(inspect.signature(assets.verify_pika_flex_checkpoint).parameters), ["filename", "reference"])
        self.assertEqual(inspect.signature(assets.verify_pika_flex_checkpoint).parameters["reference"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(inspect.signature(assets.verify_pika_flex_loras).parameters["reference"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(assets.PIKA_FLEX_KNOTS, (1000, 750, 500, 250, 100, 40, 0))
        with self.assertRaises(TypeError):
            assets.pika_flex_capabilities("references")

    def test_checkpoint_verification_selects_fp8_pin_and_rejects_aliases_before_hashing(self):
        self.assertEqual(assets.PIKA_FLEX_BASE_SIZE, 20_958_205_608)
        for reference, digest in (
            (False, "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993"),
            (True, "f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9"),
        ):
            pin = assets.PIKA_FLEX_CHECKPOINT_SPECS[reference]
            self.assertEqual(pin["sha256"], digest)
            path = Path("cache") / pin["file"]
            with patch.object(fasth3, "verify_checkpoint_pin") as verify:
                assets.verify_pika_flex_checkpoint(path, reference=reference)
            verify.assert_called_once_with(str(path), assets.PIKA_FLEX_BASE_SIZE, digest, pin["label"])

        with patch.object(fasth3, "verify_checkpoint_pin") as verify:
            with self.assertRaisesRegex(ValueError, "qualified checkpoint alias"):
                assets.verify_pika_flex_checkpoint("alternate-base.safetensors", reference=False)
            verify.assert_not_called()

    def test_lora_verification_rejects_mismatch_or_extras_before_hashing(self):
        with patch.object(fasth3, "verify_checkpoint_pin") as verify:
            with self.assertRaisesRegex(ValueError, "pinned"):
                assets.verify_pika_flex_loras(["wrong-adapter.safetensors"], reference=False)
            with self.assertRaisesRegex(ValueError, "exactly"):
                assets.verify_pika_flex_loras(
                    [assets.PIKA_FLEX_FRAMES_FILE, "extra.safetensors"],
                    reference=False,
                )
            verify.assert_not_called()

    def test_lora_verification_uses_expected_pin_after_basename_match(self):
        path = Path("cache") / assets.PIKA_FLEX_REFERENCES_FILE
        with patch.object(fasth3, "verify_checkpoint_pin") as verify:
            assets.verify_pika_flex_loras([path], reference=True)
        verify.assert_called_once_with(
            str(path),
            assets.PIKA_FLEX_LORA_SIZE,
            assets.PIKA_FLEX_REFERENCES_SHA256,
            "Pika H3 Flex References LoRA",
        )
        with self.assertRaisesRegex(ValueError, "pinned"):
            assets.verify_pika_flex_loras([path], reference=False)


if __name__ == "__main__":
    unittest.main()
