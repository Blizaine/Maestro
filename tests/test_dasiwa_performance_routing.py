"""CPU checks for live CUDA telemetry and unavailable H3 attention backends."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class LiveGpuTelemetryTests(unittest.TestCase):
    def snapshot(self, compute, *, unavailable=False):
        nvml = ModuleType("pynvml")
        nvml.nvmlInit = Mock()
        nvml.nvmlDeviceGetHandleByIndex = Mock(return_value="gpu0")
        nvml.nvmlDeviceGetUtilizationRates = Mock(return_value=SimpleNamespace(gpu=compute))
        nvml.nvmlDeviceGetMemoryInfo = Mock(return_value=SimpleNamespace(used=21 * 1024**3, total=24 * 1024**3))
        if unavailable:
            nvml.nvmlDeviceGetUtilizationRates.side_effect = RuntimeError("driver unavailable")
        psutil = ModuleType("psutil")
        psutil.cpu_percent = Mock(return_value=12)
        psutil.virtual_memory = Mock(return_value=SimpleNamespace(percent=60, used=60 * 1024**3, total=100 * 1024**3))
        graphics = ModuleType("services.gpu_engine_win")
        graphics.get_gpu_3d_utilization = Mock(return_value=0)
        with patch.dict(sys.modules, {"pynvml": nvml, "psutil": psutil, "services.gpu_engine_win": graphics}):
            spec = importlib.util.spec_from_file_location("telemetry_fixture", ROOT / "app/services/live_stats.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.get_live_stats()
        graphics.get_gpu_3d_utilization.assert_not_called()
        return result, nvml

    def test_cuda_load_is_not_replaced_by_an_idle_graphics_engine(self):
        result, nvml = self.snapshot(100)
        self.assertEqual(result["gpu"]["percent"], 100)
        self.assertEqual(result["gpu"]["compute_percent"], 100)
        self.assertEqual(result["gpu"]["vram_used_gb"], 21)
        self.assertEqual(result["gpu"]["vram_percent"], 87.5)
        nvml.nvmlDeviceGetMemoryInfo.assert_called_once_with("gpu0")

    def test_real_compute_idle_time_remains_visible(self):
        result, _ = self.snapshot(0)
        self.assertEqual(result["gpu"]["percent"], 0)
        self.assertTrue(result["gpu"]["available"])

    def test_driver_failure_keeps_cpu_and_ram_stats_available(self):
        result, _ = self.snapshot(100, unavailable=True)
        self.assertFalse(result["gpu"]["available"])
        self.assertEqual(result["cpu"]["percent"], 12)
        self.assertEqual(result["ram"]["total_gb"], 100)


class H3AttentionFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "app/wgp.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        selection = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                         and isinstance(node.test, ast.Compare)
                         and isinstance(node.test.left, ast.Name) and node.test.left.id == "attn"
                         and len(node.test.comparators) == 1
                         and isinstance(node.test.comparators[0], ast.Constant)
                         and node.test.comparators[0].value == "auto"
                         and 'model_def.get("sol_attention"' in ast.get_source_segment(source, node))
        function = ast.FunctionDef(name="select", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name)
            for name in ("attn", "model_def", "override_attention_modes_supported")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[selection, ast.Return(value=ast.Name(id="attn", ctx=ast.Load()))], decorator_list=[])
        cls.code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(ROOT / "app/wgp.py"), "exec")

    def select(self, mode, supported, *, compatible=True):
        attention = ModuleType("shared.attention")
        attention.get_default_attention_mode = lambda: "sage2"
        attention.get_sol_attention_status = lambda: {"reason": "Triton unavailable"}
        attention.get_sla_attention_status = lambda: {"reason": "Triton unavailable"}
        messages = []
        namespace = {"get_auto_attention": lambda: "sage2", "send_cmd": lambda *args: messages.append(args), "print": lambda *_: None}
        exec(self.code, namespace)
        with patch.dict(sys.modules, {"shared.attention": attention}):
            result = namespace["select"](mode, {"sol_attention": compatible, "sla_attention": compatible}, supported)
        return result, messages

    def test_saved_sol_request_on_unsupported_runtime_generates_with_dense_attention(self):
        result, messages = self.select("sol", ["sdpa", "sage2"])
        self.assertEqual(result, "sage2")
        self.assertTrue(any("safe dense" in message[1] for message in messages))
        self.assertNotIn(("exit",), messages)

    def test_supported_opt_in_stays_sol(self):
        result, messages = self.select("sol", ["sdpa", "sage2", "sol"])
        self.assertEqual(result, "sol")
        self.assertFalse(messages)

    def test_dense_auto_and_explicit_sdpa_are_retained(self):
        self.assertEqual(self.select("auto", ["sdpa", "sage2"])[0], "sage2")
        self.assertEqual(self.select("sdpa", ["sdpa", "sage2"])[0], "sdpa")

    def test_unrelated_models_do_not_gain_sol_support(self):
        result, messages = self.select("sol", ["sdpa", "sage2", "sol"], compatible=False)
        self.assertIs(result, True)
        self.assertIn(("exit",), messages)


if __name__ == "__main__":
    unittest.main()
