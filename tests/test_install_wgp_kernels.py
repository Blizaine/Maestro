"""CPU-only contracts for optional WGPKernels wheel selection."""

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
    def test_wheel_url_selects_cuda_platform_and_supported_sm(self):
        base = installer.BASE + "wgpkernels-1.1.0%2B"
        for cuda, cuda_tag in (("12.8", "cu128"), ("13.0", "cu130")):
            for system, machine, wheel_tag in (
                ("Windows", "AMD64", "win_amd64"),
                ("Linux", "x86_64", "linux_x86_64"),
            ):
                with self.subTest(cuda=cuda, system=system):
                    self.assertEqual(
                        installer.wheel_url(cuda, system, machine, (8, 0)),
                        base + f"{cuda_tag}-py3-none-{wheel_tag}.whl",
                    )
        self.assertEqual(
            installer.wheel_url("13.0", "Linux", "x86_64", (9, 0)),
            base + "cu130-py3-none-linux_x86_64.whl",
        )

    def test_wheel_url_rejects_unsupported_runtime_platform_and_sm(self):
        self.assertIsNone(installer.wheel_url("12.6", "Windows", "AMD64", (8, 9)))
        self.assertIsNone(installer.wheel_url("12.8", "Windows", "AMD64", (7, 5)))
        self.assertIsNone(installer.wheel_url("13.0", "Darwin", "x86_64", (9, 0)))
        self.assertIsNone(installer.wheel_url("13.0", "Linux", "aarch64", (9, 0)))

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

    def test_main_uses_selected_wheel_with_mocked_torch_and_installer(self):
        torch_stub = ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(
            is_available=Mock(return_value=True),
            get_device_capability=Mock(return_value=(8, 9)),
        )
        torch_stub.version = SimpleNamespace(cuda="12.8")
        with (
            patch.dict(sys.modules, {"torch": torch_stub}),
            patch.object(installer.platform, "system", return_value="Windows"),
            patch.object(installer.platform, "machine", return_value="AMD64"),
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
        self.assertEqual(command[-1], installer.wheel_url("12.8", "Windows", "AMD64", (8, 9)))
        self.assertIn("--no-deps", command)
        self.assertTrue(install.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
