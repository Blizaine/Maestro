"""Regression tests for live telemetry on unified-memory NVIDIA systems."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services import live_stats  # noqa: E402


class _UnifiedMemoryNvml:
    @staticmethod
    def nvmlDeviceGetHandleByIndex(index):
        return f"gpu-{index}"

    @staticmethod
    def nvmlDeviceGetUtilizationRates(handle):
        del handle
        return SimpleNamespace(gpu=37)

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle):
        del handle
        raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")


class TestLiveStats(unittest.TestCase):
    def test_gpu_remains_available_when_nvml_memory_is_unified(self):
        vm = SimpleNamespace(
            percent=25.0,
            used=32 * 1024**3,
            total=128 * 1024**3,
        )
        with (
            patch.object(live_stats, "_nvml_ok", True),
            patch.object(live_stats, "pynvml", _UnifiedMemoryNvml),
            patch.object(live_stats.psutil, "virtual_memory", return_value=vm),
            patch.object(live_stats.psutil, "cpu_percent", return_value=1.0),
        ):
            stats = live_stats.get_live_stats()

        self.assertTrue(stats["gpu"]["available"])
        self.assertEqual(stats["gpu"]["percent"], 37.0)
        self.assertEqual(stats["gpu"]["vram_used_gb"], 32.0)
        self.assertEqual(stats["gpu"]["vram_total_gb"], 128.0)
        self.assertEqual(stats["gpu"]["memory_source"], "unified_system")


if __name__ == "__main__":
    unittest.main()
