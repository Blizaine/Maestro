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
from services.generation_memory import classify_memory_error, cleanup_failed_generation, log_generation_memory, release_auxiliary_models
from shared.utils import offload_registry


class Activation:
    pass


class TestGenerationMemory(unittest.TestCase):
    def test_allocator_diagnostics_do_not_infer_host_ram_from_generic_cuda_oom(self):
        for error, expected in (
            (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "VRAM"),
            (RuntimeError("CUDA error: out of memory"), "CUDA"),
            (RuntimeError("CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported"), "CUDA"),
            (MemoryError(), "RAM"),
            (RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 2048 bytes"), "RAM"),
            (RuntimeError("CUDA error: too many resources requested for launch"), ""),
            (RuntimeError("Tried to allocate a buffer with invalid dimensions"), ""),
        ):
            with self.subTest(error=str(error)):
                self.assertEqual(classify_memory_error(error), expected)

    def test_failure_readings_include_host_commit_and_device_memory(self):
        gib = 1024 ** 3
        cuda = mock.Mock()
        cuda.is_initialized.return_value = True
        cuda.memory_allocated.return_value = 8 * gib
        cuda.memory_reserved.return_value = 9 * gib
        cuda.mem_get_info.return_value = (10 * gib, 19 * gib)
        process = SimpleNamespace(rss=20 * gib, private=24 * gib)
        with mock.patch("psutil.virtual_memory", return_value=SimpleNamespace(available=2 * gib, total=28 * gib)), mock.patch("psutil.Process") as get_process, mock.patch("builtins.print") as output:
            get_process.return_value.memory_info.return_value = process
            log_generation_memory(cuda)
        line = output.call_args.args[0]
        self.assertIn("system RAM available 2.00/28.00 GiB", line)
        self.assertIn("process RSS 20.00 GiB", line)
        self.assertIn("private commit 24.00 GiB", line)
        self.assertIn("VRAM allocated 8.00 GiB", line)
        self.assertIn("CUDA free 10.00/19.00 GiB", line)

    def test_failure_readings_do_not_initialize_cuda_or_mask_diagnostic_failure(self):
        cuda = mock.Mock()
        cuda.is_initialized.return_value = False
        with mock.patch("psutil.virtual_memory", side_effect=OSError("unavailable")), mock.patch("builtins.print") as output:
            log_generation_memory(cuda)
            cuda.memory_allocated.assert_not_called()
            cuda.mem_get_info.assert_not_called()
            self.assertIn("RAM readings unavailable", output.call_args.args[0])
            cuda.is_initialized.return_value = True
            cuda.memory_allocated.side_effect = RuntimeError("CUDA error: out of memory")
            log_generation_memory(cuda)
            self.assertIn("CUDA readings incomplete", output.call_args.args[0])

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
