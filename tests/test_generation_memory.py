"""Failure teardown must run after traceback-owned activations are released."""
import gc
import inspect
import os
import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from services.generation_memory import cleanup_failed_generation, release_auxiliary_models
from shared.utils import offload_registry


class Activation:
    pass


class TestGenerationMemory(unittest.TestCase):
    def test_traceback_releases_activation_before_cleanup_and_keeps_error(self):
        refs = []
        observed = []

        def cleanup():
            gc.collect()
            observed.append(refs[0]() is None)

        @cleanup_failed_generation(cleanup)
        def generate(prompt, *, frames=124):
            activation = Activation()
            refs.append(weakref.ref(activation))
            raise RuntimeError("CUDA out of memory")

        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            generate("test")
        self.assertEqual(observed, [True])
        self.assertIn("frames", inspect.signature(generate).parameters)

    def test_handled_failure_cleans_but_success_keeps_loaded_model(self):
        cleanup = mock.Mock()

        @cleanup_failed_generation(cleanup)
        def generate(success):
            return success

        self.assertTrue(generate(True))
        cleanup.assert_not_called()
        self.assertFalse(generate(False))
        cleanup.assert_called_once()

    def test_cleanup_error_cannot_mask_generation_failure(self):
        @cleanup_failed_generation(mock.Mock(side_effect=ValueError("cleanup")))
        def generate():
            raise RuntimeError("original failure")

        with self.assertRaisesRegex(RuntimeError, "original failure"):
            generate()

    def test_release_includes_flashvsr_and_registered_postprocessors(self):
        flash_release = mock.Mock()
        flash = SimpleNamespace(_RUNTIME=SimpleNamespace(dit=object()), release_models=flash_release)
        with mock.patch.dict(sys.modules, {"postprocessing.flashvsr.runtime": flash}), mock.patch.object(offload_registry, "registered_names", return_value=["DLSS Motion Vectors"]), mock.patch.object(offload_registry, "release_all", return_value=["DLSS Motion Vectors"]) as registry_release:
            released = release_auxiliary_models()
        self.assertEqual(released, ["DLSS Motion Vectors", "FlashVSR"])
        registry_release.assert_called_once_with(["DLSS Motion Vectors"])
        flash_release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
