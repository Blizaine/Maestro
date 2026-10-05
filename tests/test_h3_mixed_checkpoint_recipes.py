"""A shared Turbo/non-Turbo page must not choose a recipe for every file."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_h3_checkpoint_import import make_header
from services.h3_checkpoint_import import H3CheckpointError, inspect_h3_header


MIXED_SOURCE = {
    "name": "Example H3 Hybrid",
    "versionName": "beta5",
    "filename": "exampleH3_beta5_123456.safetensors",
    "versionDescription": "Normalized hybrid Turbo and standard releases.",
    "description": (
        "Ref2VA hybrid. The non-Turbo version and baked Turbo files are both available. "
        "Turbo has a delta fusion baked in. Use res_multistep, 6-9 steps. "
        "An older version recommends 6 steps, video shift 5, audio shift 7."
    ),
}


class H3MixedCheckpointRecipeTests(unittest.TestCase):
    def inspect(self, sampling_profile="auto", header_title=None, source=None):
        header, payloads = make_header(quantization="int8")
        if header_title:
            header["__metadata__"]["modelspec.title"] = header_title
        return inspect_h3_header(
            header, tensor_reader=payloads.__getitem__,
            source=MIXED_SOURCE if source is None else source,
            sampling_profile=sampling_profile,
        )

    def test_mixed_page_requests_the_selected_files_recipe(self):
        profile = self.inspect()
        self.assertEqual(profile["status"], "needs_selection")
        self.assertEqual(profile["needs_selection"], ["sampling_profile"])
        self.assertIsNone(profile["sampling_profile"])
        self.assertEqual(profile["selection_options"]["sampling_profile"], ["standard", "turbo", "fused"])

    def test_explicit_standard_keeps_standard_defaults(self):
        profile = self.inspect("standard")
        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["default_steps"], 20)
        self.assertEqual((profile["min_steps"], profile["max_steps"]), (2, 50))
        self.assertEqual((profile["sampler"], profile["video_shift"], profile["audio_shift"]), ("euler", 12, 3))
        self.assertFalse(profile["baked_turbo"])
        self.assertFalse(profile["fused_turbo"])

    def test_explicit_turbo_uses_confirmed_defaults_without_sibling_schedule(self):
        profile = self.inspect("turbo")
        self.assertEqual(profile["status"], "verified")
        self.assertEqual((profile["default_steps"], profile["min_steps"], profile["max_steps"]), (8, 4, 8))
        self.assertEqual((profile["sampler"], profile["video_shift"], profile["audio_shift"]), ("euler", 12, 3))
        self.assertTrue(profile["baked_turbo"])
        self.assertFalse(profile["fused_turbo"])

    def test_explicit_header_recipe_still_takes_precedence(self):
        standard = self.inspect(header_title="minimax_h3_ref2va_pruned_standard")
        self.assertEqual(standard["status"], "verified")
        self.assertEqual(standard["sampling_profile"], "standard")
        with self.assertRaises(H3CheckpointError):
            self.inspect("standard", header_title="minimax_h3_ref2va_pruned_turbo")

    def test_non_turbo_header_identifies_the_standard_edition(self):
        profile = self.inspect(header_title="minimax_h3_ref2va_pruned_non-turbo")
        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["sampling_profile"], "standard")
        self.assertEqual(profile["default_steps"], 20)

    def test_concrete_version_recipe_cannot_be_overridden(self):
        source = dict(MIXED_SOURCE, versionName="Ref2VA Turbo")
        pending = self.inspect(source=source)
        self.assertEqual(pending["sampling_profile"], "turbo")
        self.assertIn("sampling_recipe", pending["needs_selection"])
        with self.assertRaises(H3CheckpointError):
            self.inspect("standard", source=source)


if __name__ == "__main__":
    unittest.main()
