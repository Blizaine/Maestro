from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from scripts.install_optional_cuda_acceleration import (  # noqa: E402
    FLASHATTENTION_WHEEL,
    SAGEATTENTION_WHEEL,
    install_optional_wheel,
)
from scripts.verify_sol_runtime import validate_required_runtime  # noqa: E402


class TestOptionalLinuxAccelerationInstall(unittest.TestCase):
    def test_uses_prebuilt_cuda13_wheels(self):
        self.assertIn("sageattention-2.2.0-cp311-cp311-linux_x86_64.whl", SAGEATTENTION_WHEEL)
        self.assertIn("sha256=2ce936012a361e80", SAGEATTENTION_WHEEL)
        self.assertIn("cu130torch2.10-cp311-cp311-linux_x86_64.whl", FLASHATTENTION_WHEEL)

    def test_optional_wheel_failure_never_raises_or_blocks_runtime(self):
        calls = []

        def failed_runner(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=1, stdout="compiler output")

        with patch("scripts.install_optional_cuda_acceleration.shutil.which", return_value="uv"):
            self.assertFalse(
                install_optional_wheel(
                    "Optional test wheel",
                    "https://example.invalid/test.whl",
                    runner=failed_runner,
                )
            )

        self.assertEqual(calls[0][0][:3], ["uv", "pip", "install"])
        self.assertFalse(calls[0][1]["check"])


class TestRequiredSolRuntimeVerification(unittest.TestCase):
    def test_supported_cuda13_runtime_passes(self):
        self.assertEqual(
            validate_required_runtime(
                python_version="3.11.14",
                torch_version="2.10.0+cu130",
                cuda_version="13.0",
                triton_version="3.6.0",
                cuda_available=True,
                capability=(8, 9),
            ),
            [],
        )

    def test_cuda12_or_unsupported_gpu_does_not_publish_runtime_marker(self):
        problems = validate_required_runtime(
            python_version="3.11.14",
            torch_version="2.10.0+cu130",
            cuda_version="12.8",
            triton_version="3.6.0",
            cuda_available=True,
            capability=(8, 6),
        )
        self.assertTrue(any("CUDA 13" in problem for problem in problems))
        self.assertTrue(any("Sol-compatible" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
