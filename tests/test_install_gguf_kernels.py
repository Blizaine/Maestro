"""CPU-only contracts for the optional GGUF kernel updater."""

import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
INSTALLER_PATH = ROOT / "app/scripts/install_gguf_kernels.py"
_spec = importlib.util.spec_from_file_location("install_gguf_kernels_under_test", INSTALLER_PATH)
installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(installer)


class InstallGGUFKernelsTests(unittest.TestCase):
    def _spec_for(self, py_minor, torch, cuda, os_name, machine, impl="cpython", libc=None):
        return installer._wheel_spec(py_minor, torch, cuda, os_name, machine, impl, libc)

    def test_wheel_spec_selects_the_four_published_cuda_assets(self):
        cases = (
            (10, "2.7.1", "12.8", "windows", "AMD64", None, "torch271cu128py310", "win_amd64"),
            (10, "2.7.1", "12.8", "linux", "x86_64", ("glibc", "2.28"), "torch271cu128py310", "linux_x86_64"),
            (11, "2.10.0", "13.0", "windows", "AMD64", None, "torch210cu130py311", "win_amd64"),
            (11, "2.10.0", "13.0", "linux", "x86_64", ("glibc", "2.36"), "torch210cu130py311", "linux_x86_64"),
        )
        for py_minor, torch, cuda, os_name, machine, libc, build, wheel_platform in cases:
            with self.subTest(py_minor=py_minor, os_name=os_name):
                digest = installer._WHEEL_SHA256[(build, "windows" if os_name == "windows" else "linux")]
                cp = f"cp3{py_minor}"
                expected_url = (
                    f"{installer.RELEASE_URL}llamacpp_gguf_cuda-{installer.VERSION}%2B{build}-"
                    f"{cp}-{cp}-{wheel_platform}.whl#sha256={digest}"
                )
                self.assertEqual(
                    self._spec_for(py_minor, torch, cuda, os_name, machine, libc=libc),
                    (expected_url, f"{installer.VERSION}+{build}"),
                )

    def test_wheel_spec_rejects_unsupported_runtime_platform_and_libc(self):
        args = (11, "2.10.0", "13.0")
        self.assertIsNone(self._spec_for(*args, "darwin", "x86_64"))
        self.assertIsNone(self._spec_for(*args, "linux", "aarch64", libc=("glibc", "2.36")))
        self.assertIsNone(self._spec_for(*args, "linux", "x86_64", libc=("glibc", "2.27")))
        self.assertIsNone(self._spec_for(*args, "linux", "x86_64", libc=("musl", "1.2.5")))
        self.assertIsNone(self._spec_for(*args, "windows", "AMD64", impl="pypy"))
        self.assertIsNone(self._spec_for(10, "2.7.0", "12.8", "windows", "AMD64"))
        self.assertIsNone(self._spec_for(10, "2.7.1", "12.6", "windows", "AMD64"))
        self.assertIsNone(self._spec_for(12, "2.10.0", "13.0", "windows", "AMD64"))

    def test_importable_stale_package_version_is_upgraded(self):
        stale_module = ModuleType("llamacpp_gguf_cuda")
        output = io.StringIO()
        runtime = (10, "2.7.1", "12.8", "windows", "AMD64", "cpython", ("", ""))
        with (
            patch.object(installer, "_detect_env", return_value=runtime),
            patch.dict(sys.modules, {"llamacpp_gguf_cuda": stale_module}),
            patch.object(installer.importlib.metadata, "version", return_value="1.0.2+torch271cu128py310"),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", output),
        ):
            self.assertEqual(installer.main(), 0)
        command = install.call_args.args[0]
        self.assertEqual(command[-1], self._spec_for(*runtime[:4], runtime[4], runtime[5], runtime[6])[0])
        self.assertIn("--no-deps", command)
        self.assertTrue(install.call_args.kwargs["check"])
        self.assertEqual(install.call_args.kwargs["timeout"], 300)
        self.assertIn("Installing:", output.getvalue())

    def test_current_exact_stack_version_skips_install(self):
        runtime = (11, "2.10.0", "13.0", "windows", "AMD64", "cpython", ("", ""))
        output = io.StringIO()
        with (
            patch.object(installer, "_detect_env", return_value=runtime),
            patch.object(installer.importlib.metadata, "version", return_value="1.0.29+torch210cu130py311"),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", output),
        ):
            self.assertEqual(installer.main(), 0)
        install.assert_not_called()
        self.assertIn("already installed", output.getvalue())

    def test_older_patch_1_0_9_is_upgraded_to_1_0_29(self):
        runtime = (11, "2.10.0", "13.0", "windows", "AMD64", "cpython", ("", ""))
        with (
            patch.object(installer, "_detect_env", return_value=runtime),
            patch.object(installer.importlib.metadata, "version", return_value="1.0.9+torch210cu130py311"),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(installer.main(), 0)
        self.assertEqual(install.call_count, 1)
        self.assertIn("1.0.29", install.call_args.args[0][-1])

    def test_newer_public_version_preserves_a_manual_pin(self):
        runtime = (11, "2.10.0", "13.0", "windows", "AMD64", "cpython", ("", ""))
        output = io.StringIO()
        with (
            patch.object(installer, "_detect_env", return_value=runtime),
            patch.object(installer.importlib.metadata, "version", return_value="1.0.30+torch210cu130py311"),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", output),
        ):
            self.assertEqual(installer.main(), 0)
        install.assert_not_called()
        self.assertIn("newer than the optional", output.getvalue())

    def test_newer_release_with_wrong_stack_build_is_not_preserved(self):
        self.assertFalse(
            installer._newer_public_release_is_installed(
                "1.0.30+torch271cu128py310",
                "1.0.29+torch210cu130py311",
            )
        )

    def test_missing_distribution_metadata_does_not_skip_importable_module(self):
        stale_module = ModuleType("llamacpp_gguf_cuda")
        runtime = (10, "2.7.1", "12.8", "windows", "AMD64", "cpython", ("", ""))
        with (
            patch.object(installer, "_detect_env", return_value=runtime),
            patch.dict(sys.modules, {"llamacpp_gguf_cuda": stale_module}),
            patch.object(
                installer.importlib.metadata,
                "version",
                side_effect=installer.importlib.metadata.PackageNotFoundError,
            ),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(installer.main(), 0)
        install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
