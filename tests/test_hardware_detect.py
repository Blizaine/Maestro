"""CPU-only tests for hardware probes and bundled allocator capability."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
try:
    import psutil  # noqa: F401
except ImportError:
    # The focused test interpreter may not include app dependencies. The
    # detector's RAM interface is mocked in these tests, so a small stub is
    # sufficient to import the module without installing anything.
    psutil_stub = ModuleType("psutil")
    psutil_stub.virtual_memory = lambda: SimpleNamespace(total=0, available=0)
    psutil_stub.cpu_count = lambda logical=True: 1
    sys.modules["psutil"] = psutil_stub
from services import hardware_detect as hardware  # noqa: E402


class TestHardwareDetection(unittest.TestCase):
    @staticmethod
    def _fake_torch(*, cuda_available=True, total_memory_gb=10):
        class FakeCuda:
            @staticmethod
            def is_available():
                return cuda_available

            @staticmethod
            def get_device_properties(_index):
                return SimpleNamespace(total_memory=total_memory_gb * (1024 ** 3))

            @staticmethod
            def get_device_capability(_index):
                return 8, 6

            @staticmethod
            def get_device_name(_index):
                return "NVIDIA GeForce RTX 3080"

        return SimpleNamespace(
            __version__="2.10.0+cu130",
            version=SimpleNamespace(cuda="13.0", hip=None),
            cuda=FakeCuda(),
        )

    def test_gpu_probe_preserves_exact_10_and_12_gb_capacities(self):
        for capacity in (10, 12):
            with self.subTest(capacity=capacity):
                fake_torch = self._fake_torch(total_memory_gb=capacity)
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    detected = hardware._detect_gpu()
                self.assertTrue(detected["cuda_available"])
                self.assertEqual(detected["gpu_vram_gb"], float(capacity))
                self.assertEqual(detected["gpu_name"], "NVIDIA GeForce RTX 3080")
                self.assertTrue(detected["nvidia_cuda"])
                self.assertEqual(detected["torch_version"], "2.10.0+cu130")
                self.assertEqual(detected["runtime_version"], "13.0")

    def test_allocator_support_requires_cuda_supported_platform_and_bundled_binary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "mmgp" / "allocator" / "vmm_alloc_win_amd64.dll"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"bundled library placeholder")

            args = {"app_root": root, "cuda_available": True, "nvidia_cuda": True,
                    "system": "win32", "machine": "AMD64"}
            self.assertTrue(hardware._supports_mmgp_allocator(**args))
            self.assertFalse(hardware._supports_mmgp_allocator(**{**args, "cuda_available": False}))
            self.assertFalse(hardware._supports_mmgp_allocator(**{**args, "nvidia_cuda": False}))
            self.assertFalse(hardware._supports_mmgp_allocator(**{**args, "system": "darwin"}))
            self.assertFalse(hardware._supports_mmgp_allocator(**{**args, "machine": "arm64"}))

            library.unlink()
            self.assertFalse(hardware._supports_mmgp_allocator(**args))

            linux_library = root / "mmgp" / "allocator" / "vmm_alloc_linux_x86_64.so"
            linux_library.write_bytes(b"bundled library placeholder")
            self.assertTrue(hardware._supports_mmgp_allocator(
                app_root=root, cuda_available=True, nvidia_cuda=True,
                system="linux", machine="x86_64",
            ))

    def test_driver_version_probe_returns_string_and_tolerates_unknown(self):
        fake_nvml = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlSystemGetDriverVersion=lambda: b"580.88",
        )
        with patch.dict(sys.modules, {"pynvml": fake_nvml}):
            self.assertEqual(hardware._detect_driver_version(), "580.88")
        with patch.dict(sys.modules, {"pynvml": None}):
            self.assertEqual(hardware._detect_driver_version(), "unknown")

    def test_non_cuda_and_unknown_probes_return_stable_string_fields(self):
        fake_torch = self._fake_torch(cuda_available=False)
        memory = SimpleNamespace(total=32 * (1024 ** 3), available=9 * (1024 ** 3))
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(hardware.psutil, "virtual_memory", return_value=memory),
            patch.object(hardware.psutil, "cpu_count", return_value=16),
            patch.object(hardware, "_supports_mmgp_allocator", return_value=False),
        ):
            detected = hardware.detect_hardware()
        self.assertFalse(detected["cuda_available"])
        self.assertFalse(detected["supports_mmgp_allocator"])
        self.assertEqual(detected["gpu_vram_gb"], 0.0)
        self.assertEqual(detected["ram_gb"], 32.0)
        self.assertEqual(detected["ram_available_gb"], 9.0)
        self.assertEqual(detected["driver_version"], "unknown")
        self.assertEqual(detected["torch_version"], "2.10.0+cu130")
        self.assertEqual(detected["runtime_version"], "13.0")

        with (
            patch.dict(sys.modules, {"torch": None}),
            patch.object(hardware.psutil, "virtual_memory", return_value=memory),
            patch.object(hardware.psutil, "cpu_count", return_value=16),
            patch.object(hardware, "_supports_mmgp_allocator", return_value=False),
        ):
            unknown = hardware.detect_hardware()
        self.assertFalse(unknown["cuda_available"])
        self.assertFalse(unknown["supports_mmgp_allocator"])
        self.assertEqual(unknown["driver_version"], "unknown")
        self.assertEqual(unknown["torch_version"], "unknown")
        self.assertEqual(unknown["runtime_version"], "unknown")


if __name__ == "__main__":
    unittest.main()
