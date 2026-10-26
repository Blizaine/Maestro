"""Keep a verified hybrid's conditioning modes separate from its native basis."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tests"))

from services.civitai_checkpoints import build_h3_definitions
from services.h3_checkpoint_import import H3CheckpointError, inspect_h3_header
from test_h3_checkpoint_import import make_beta6_int8_header


def beta6_source(**overrides):
    return {
        "modelId": 2851079, "versionId": 3393047, "fileId": 3283276,
        "sha256": "e2c5493e5997726ab12f17caecd6fe88c7e64b135bd3435c0f158c8f1b632a5f",
        "name": "H3 Eros Max", "versionName": "beta6",
        "filename": "h3ErosMax_beta6_3283276.safetensors",
        "versionDescription": (
            "Rebuilt from the full FL2VA base with reference deltas added. "
            "Merged Turbo targets 8-12 steps and video/audio shifts 10/3."
        ),
        **overrides,
    }


class H3HybridCapabilitiesTests(unittest.TestCase):
    def inspect(self, **kwargs):
        header, payloads = make_beta6_int8_header()
        return inspect_h3_header(header, tensor_reader=payloads.__getitem__, **kwargs)

    def test_verified_beta6_enables_both_modes_with_its_fl2va_basis(self):
        source = beta6_source()
        profile = self.inspect(source=source)
        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["needs_selection"], [])
        self.assertEqual(profile["native_workflow"], "fl2va")
        self.assertTrue(profile["supports_references"])
        self.assertEqual(profile["architectures"], ["minimax_h3", "minimax_h3_ref2va"])
        self.assertEqual(profile["quantized_layers"], 195)
        self.assertEqual(profile["sampling_profile"], "turbo")
        self.assertEqual((profile["sampler"], profile["video_shift"], profile["audio_shift"]),
                         ("euler", 10, 3))
        self.assertEqual((profile["default_steps"], profile["min_steps"], profile["max_steps"]),
                         (8, 8, 12))
        entries = build_h3_definitions(source, profile, source["filename"], ROOT / "app/defaults")
        self.assertEqual(len(entries), 2)
        for entry in entries.values():
            self.assertEqual(entry["model"]["URLs"], [source["filename"]])
            self.assertEqual(entry["model"]["minimax_h3_lora_workflow"], "fl2va")
            self.assertEqual(set(entry["model"]["h3_companion_models"].values()), set(entries))
            self.assertTrue(entry["model"]["minimax_h3_baked_turbo"])

    def test_unknown_fl2va_export_does_not_inherit_reference_capability(self):
        for source in (beta6_source(fileId=123), beta6_source(versionId=123)):
            with self.subTest(source=source):
                source["supports_references"] = True  # Client metadata is not proof.
                source["versionDescription"] = "Built from FL2VA, with reference deltas added."
                profile = self.inspect(source=source, native_workflow="fl2va", sampling_profile="standard")
                self.assertEqual(profile["architectures"], ["minimax_h3"])
                self.assertFalse(profile["supports_references"])

    def test_known_file_identity_requires_its_published_hash(self):
        with self.assertRaises(H3CheckpointError) as caught:
            self.inspect(source=beta6_source(sha256="a" * 64))
        self.assertEqual(caught.exception.code, "source_hash_mismatch")

    def test_reference_capability_does_not_relabel_the_native_basis(self):
        with self.assertRaises(H3CheckpointError) as caught:
            self.inspect(source=beta6_source(), native_workflow="ref2va")
        self.assertEqual(caught.exception.code, "native_workflow_conflict")


if __name__ == "__main__":
    unittest.main()
