"""CPU-only contracts for optional WGPKernels wheel selection and pinning."""

import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
INSTALLER_PATH = ROOT / "app/scripts/install_wgp_kernels.py"
_spec = importlib.util.spec_from_file_location("install_wgp_kernels_under_test", INSTALLER_PATH)
installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(installer)


class InstallWGPKernelsTests(unittest.TestCase):
    def test_wheel_url_selects_release_assets_for_supported_runtime_stacks(self):
        cases = (
            (10, "2.7.1", "12.8", "cu128"),
            (11, "2.10.0", "13.0", "cu130"),
        )
        for py_minor, torch_version, cuda, tag in cases:
            for system, machine, wheel_platform, libc in (
                ("Windows", "AMD64", "win_amd64", None),
                ("Linux", "x86_64", "linux_x86_64", ("glibc", "2.34")),
            ):
                with self.subTest(py_minor=py_minor, system=system, cuda=cuda):
                    digest = installer._WHEEL_SHA256[(tag, wheel_platform)]
                    self.assertEqual(
                        installer.wheel_url(cuda, system, machine, (8, 0), py_minor, torch_version, libc),
                        f"{installer.BASE}wgpkernels-1.4.2%2B{tag}-py3-none-{wheel_platform}.whl#sha256={digest}",
                    )

    def test_wheel_url_rejects_unsupported_stack_platform_libc_and_sm(self):
        self.assertIsNone(installer.wheel_url("12.6", "Windows", "AMD64", (8, 9), 10, "2.7.1"))
        self.assertIsNone(installer.wheel_url("12.8", "Windows", "AMD64", (8, 9), 11, "2.7.1"))
        self.assertIsNone(installer.wheel_url("13.0", "Windows", "AMD64", (8, 9), 11, "2.10.1"))
        self.assertIsNone(installer.wheel_url("12.8", "Windows", "AMD64", (7, 5), 10, "2.7.1"))
        self.assertIsNone(installer.wheel_url("13.0", "Darwin", "x86_64", (9, 0), 11, "2.10.0"))
        self.assertIsNone(installer.wheel_url("13.0", "Linux", "aarch64", (9, 0), 11, "2.10.0", ("glibc", "2.36")))
        self.assertIsNone(installer.wheel_url("13.0", "Linux", "x86_64", (9, 0), 11, "2.10.0", ("glibc", "2.33")))
        self.assertIsNone(installer.wheel_url("13.0", "Linux", "x86_64", (9, 0), 11, "2.10.0", ("musl", "1.2.5")))

    def test_newer_release_with_wrong_cuda_variant_is_not_preserved(self):
        self.assertFalse(installer._newer_public_release_is_installed("1.4.3+cu128", "1.4.2+cu130"))

    def test_main_skips_when_mocked_torch_reports_no_cuda_gpu(self):
        torch_stub = ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(is_available=Mock(return_value=False))
        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"torch": torch_stub}),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", output),
        ):
            self.assertEqual(installer.main(), 0)
        install.assert_not_called()
        self.assertIn("No CUDA GPU", output.getvalue())

    def test_main_uses_hashed_wheel_with_mocked_torch_and_installer(self):
        torch_stub = ModuleType("torch")
        torch_stub.__version__ = "2.7.1+cu128"
        torch_stub.cuda = SimpleNamespace(
            is_available=Mock(return_value=True),
            get_device_capability=Mock(return_value=(8, 9)),
        )
        torch_stub.version = SimpleNamespace(cuda="12.8")
        with (
            patch.dict(sys.modules, {"torch": torch_stub}),
            patch.object(installer.platform, "system", return_value="Windows"),
            patch.object(installer.platform, "machine", return_value="AMD64"),
            patch.object(installer.sys, "version_info", SimpleNamespace(minor=10)),
            patch.object(
                installer.importlib.metadata,
                "version",
                side_effect=installer.importlib.metadata.PackageNotFoundError,
            ),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(installer.main(), 0)
        command = install.call_args.args[0]
        self.assertEqual(
            command[-1],
            installer.wheel_url("12.8", "Windows", "AMD64", (8, 9), 10, "2.7.1"),
        )
        self.assertIn("--no-deps", command)
        self.assertTrue(install.call_args.kwargs["check"])
        self.assertEqual(install.call_args.kwargs["timeout"], 300)

    def test_main_skips_only_the_exact_installed_stack_version(self):
        torch_stub = ModuleType("torch")
        torch_stub.__version__ = "2.10.0+cu130"
        torch_stub.cuda = SimpleNamespace(
            is_available=Mock(return_value=True),
            get_device_capability=Mock(return_value=(8, 9)),
        )
        torch_stub.version = SimpleNamespace(cuda="13.0")
        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"torch": torch_stub}),
            patch.object(installer.platform, "system", return_value="Windows"),
            patch.object(installer.platform, "machine", return_value="AMD64"),
            patch.object(installer.sys, "version_info", SimpleNamespace(minor=11)),
            patch.object(installer.importlib.metadata, "version", return_value="1.4.2+cu130"),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", output),
        ):
            self.assertEqual(installer.main(), 0)
        install.assert_not_called()
        self.assertIn("1.4.2+cu130 already installed", output.getvalue())

    def test_main_preserves_a_newer_manually_installed_release(self):
        torch_stub = ModuleType("torch")
        torch_stub.__version__ = "2.10.0+cu130"
        torch_stub.cuda = SimpleNamespace(
            is_available=Mock(return_value=True),
            get_device_capability=Mock(return_value=(8, 9)),
        )
        torch_stub.version = SimpleNamespace(cuda="13.0")
        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"torch": torch_stub}),
            patch.object(installer.platform, "system", return_value="Windows"),
            patch.object(installer.platform, "machine", return_value="AMD64"),
            patch.object(installer.sys, "version_info", SimpleNamespace(minor=11)),
            patch.object(installer.importlib.metadata, "version", return_value="1.4.3+cu130"),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", output),
        ):
            self.assertEqual(installer.main(), 0)
        install.assert_not_called()
        self.assertIn("newer than the optional", output.getvalue())

    def test_main_repairs_a_wrong_stack_label_for_the_same_release(self):
        torch_stub = ModuleType("torch")
        torch_stub.__version__ = "2.10.0+cu130"
        torch_stub.cuda = SimpleNamespace(
            is_available=Mock(return_value=True),
            get_device_capability=Mock(return_value=(8, 9)),
        )
        torch_stub.version = SimpleNamespace(cuda="13.0")
        with (
            patch.dict(sys.modules, {"torch": torch_stub}),
            patch.object(installer.platform, "system", return_value="Windows"),
            patch.object(installer.platform, "machine", return_value="AMD64"),
            patch.object(installer.sys, "version_info", SimpleNamespace(minor=11)),
            patch.object(installer.importlib.metadata, "version", return_value="1.4.2+cu128"),
            patch.object(installer.subprocess, "run") as install,
            patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(installer.main(), 0)
        install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
