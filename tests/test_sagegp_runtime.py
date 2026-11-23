"""Qualification of SageGP selection and unsupported-call fallbacks."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import torch
from torch.nn.functional import scaled_dot_product_attention

from scripts.install_sagegp import compatible_wheel
from shared.attention import pay_attention
from shared import sagegp_runtime as gp


class SageGPTests(unittest.TestCase):
    def test_first_step_precision_is_scoped_to_a_pass_and_job(self):
        self.assertEqual(gp.effective_mode(gp.FAST), gp.FAST)
        with gp.sampling_scope():
            self.assertEqual(gp.effective_mode(gp.FAST), gp.PRECISE)
            gp.set_step(1)
            self.assertEqual(gp.effective_mode(gp.FAST), gp.FAST)
            gp.set_step(0)
            self.assertEqual(gp.effective_mode(gp.ADAPTIVE), gp.PRECISE)
            gp.configure_sampling(False)
            self.assertEqual(gp.effective_mode(gp.ADAPTIVE), gp.ADAPTIVE)
        self.assertEqual(gp.effective_mode(gp.FAST), gp.FAST)

    def test_non_blackwell_does_not_list_blackwell_presets(self):
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
                self.assertEqual(gp.available_modes(), ())

    def test_cpu_fallback_preserves_causal_attention_and_explicit_scale(self):
        generator = torch.Generator().manual_seed(42)
        q, k, v = [torch.randn((1, 12, 2, 16), generator=generator) for _ in range(3)]
        expected = scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=0.125
        ).transpose(1, 2)
        inputs = [q, k, v]
        actual = pay_attention(inputs, force_attention=gp.FAST, causal=True, softmax_scale=0.125)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(inputs, [])

    def test_wheels_are_pinned_and_legacy_stacks_are_excluded(self):
        args = ("Windows", "AMD64", (3, 11), (2, 10), "13.0", (12, 0))
        filename, digest = compatible_wheel(*args)
        self.assertTrue(filename.endswith("cp310-abi3-win_amd64.whl"))
        self.assertEqual(digest, "d85e0630b4dcb6ca5cb0a1dbd72c44287a760373570c15cb76fae64071cf09e9")
        for replacement in (("Windows", "AMD64", (3, 10), (2, 7), "12.8", (12, 0)),
                            ("Windows", "AMD64", (3, 11), (2, 10), "13.0", (8, 9)),
                            ("Darwin", "arm64", (3, 11), (2, 10), "", (0, 0))):
            self.assertIsNone(compatible_wheel(*replacement))

    def test_runtime_failure_falls_back_without_consuming_inputs(self):
        q = torch.zeros((1, 2, 1, 2))
        kernel = SimpleNamespace(
            __name__="test_sagegp_kernel",
            attention_owned=Mock(side_effect=RuntimeError("optional compiler unavailable")),
        )
        errors = SimpleNamespace(CompilationError=ValueError)
        with patch.dict(sys.modules, {"triton.compiler.errors": errors}), patch.object(
            gp, "_failed_kernels", set()
        ):
            self.assertIsNone(gp._run_kernel(kernel, q, q, q))
            self.assertIsNone(gp._run_kernel(kernel, q, q, q))
        kernel.attention_owned.assert_called_once_with([q, q, q], recycle_q=False)


if __name__ == "__main__":
    unittest.main()
