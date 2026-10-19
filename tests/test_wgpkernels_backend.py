"""CPU contract tests for optional WGPKernels INT8 dispatch."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

_APP = Path(__file__).resolve().parents[1] / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))


def _import_project_module(name):
    return __import__(name, fromlist=["*"])


class TestWgpBackendSelection(unittest.TestCase):
    def setUp(self):
        self.backend = _import_project_module("shared.kernels.int8_backend")
        self.triton = _import_project_module("shared.kernels.quanto_int8_inject")

    def test_probe_skips_package_loading_without_gpu(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(self.backend._probe_wgpkernels(), (None, "requires a supported GPU"))

    def test_auto_prefers_wgpkernels(self):
        expected = object()
        with mock.patch.object(self.backend, "_probe_wgpkernels", return_value=(expected, "available")) as wgp:
            with mock.patch.object(self.backend, "_probe_kitchen") as kitchen:
                with mock.patch.object(self.triton, "_probe_triton_backend") as triton:
                    self.assertEqual(self.backend.resolve_backend("auto"), ("wgpkernels", expected))
        wgp.assert_called_once_with()
        kitchen.assert_not_called()
        triton.assert_not_called()

    def test_unavailable_explicit_backend_falls_back_through_auto_order(self):
        kitchen_module = object()
        with mock.patch.object(self.backend, "_probe_wgpkernels", return_value=(None, "missing wheel")):
            with mock.patch.object(self.backend, "_probe_kitchen", return_value=(kitchen_module, "available")):
                with mock.patch.object(self.triton, "_probe_triton_backend") as triton:
                    self.assertEqual(self.backend.resolve_backend("wgpkernels"), ("kitchen", kitchen_module))
        triton.assert_not_called()

    def test_unknown_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown INT8 kernel backend"):
            self.backend.resolve_backend("unknown")


class TestWgpLinearDispatch(unittest.TestCase):
    def setUp(self):
        self.backend = _import_project_module("shared.kernels.int8_backend")
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(mock.patch.object(self.backend, "_backend", "wgpkernels"))
        self.patches.enter_context(mock.patch.object(self.backend, "_weight_scale", return_value=torch.ones(3)))
        self.patches.enter_context(mock.patch.object(self.backend, "_wgp_aligned", side_effect=lambda x: x))
        self.patches.enter_context(mock.patch.object(torch.compiler, "is_compiling", return_value=False))
        self.calls = []

        def linear(x, weight, scale, bias, group, precise):
            self.calls.append((x, weight, scale, bias, group, precise))
            result = x.float() @ weight.float().t()
            if bias is not None:
                result.add_(bias)
            return result.to(x.dtype)

        self.wgp = SimpleNamespace(linear=linear)
        self.patches.enter_context(mock.patch.object(self.backend, "_wgp", self.wgp))

    def test_linear_uses_requested_rotation_group_and_precision(self):
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16)
        data = torch.arange(24, dtype=torch.int8).reshape(3, 8)
        bias = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
        weight = SimpleNamespace(_data=data)

        result = self.backend.wgp_linear(x, weight, bias, group=64, precise=True)

        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][4:], (64, True))
        torch.testing.assert_close(result, (x.float() @ data.float().t() + bias.float()).to(x.dtype))

    def test_marked_projection_uses_full_precision_activation_path(self):
        module = SimpleNamespace(
            qweight=object(),
            bias=None,
            _int8_residual_forward=lambda _x: "fallback",
        )
        expected = object()
        x = torch.ones(1)
        with mock.patch.object(self.backend, "_wgp_supports", return_value=True):
            with mock.patch.object(self.backend, "wgp_linear", return_value=expected) as linear:
                self.assertIs(self.backend._residual_module_forward(module, x), expected)
        linear.assert_called_once()
        self.assertIs(linear.call_args.args[0], x)
        self.assertEqual(linear.call_args.kwargs, {"precise": True})


class TestWgpRowProjection(unittest.TestCase):
    def setUp(self):
        self.backend = _import_project_module("shared.kernels.int8_backend")
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(mock.patch.object(self.backend, "_backend", "wgpkernels"))
        self.patches.enter_context(mock.patch.object(torch.compiler, "is_compiling", return_value=False))
        self.patches.enter_context(mock.patch.object(self.backend.triton, "_is_fake_tensor", return_value=False))

    def _module(self):
        data = torch.arange(64, dtype=torch.int8).reshape(4, 16)
        weight = SimpleNamespace(_data=data)
        return SimpleNamespace(
            weight_qtype=SimpleNamespace(name="qint8_convrot"),
            _convrot_group_size=16,
            qweight=weight,
            bias=torch.arange(4, dtype=torch.bfloat16),
        )

    def test_row_provider_supports_only_compatible_cuda_dispatch_metadata(self):
        module = self._module()
        x = torch.empty((8, 16), dtype=torch.bfloat16)
        with mock.patch.object(self.backend, "_wgp_supports", return_value=True):
            self.assertTrue(self.backend._WGP_ROW_PROJECTION.supports(module, x))

        module._convrot_group_size = 0
        with mock.patch.object(self.backend, "_wgp_supports", return_value=True):
            self.assertFalse(self.backend._WGP_ROW_PROJECTION.supports(module, x))

    def test_prepare_and_output_rows_preserve_precise_flag_and_weight_slice(self):
        module = self._module()
        module._int8_residual_forward = lambda x: x
        x = torch.empty((8, 16), dtype=torch.bfloat16)
        prepared = SimpleNamespace(dtype=torch.bfloat16)
        path = self.patches.enter_context(mock.patch.object(self.backend, "_wgp_path", return_value="full"))
        wgp = SimpleNamespace(prepare=mock.Mock(return_value=prepared))
        self.patches.enter_context(mock.patch.object(self.backend, "_wgp", wgp))

        actual_prepared = self.backend._WGP_ROW_PROJECTION.prepare(module, x)
        self.assertIs(actual_prepared, prepared)
        path.assert_called_once_with(module, x, 16, True)
        wgp.prepare.assert_called_once_with(x, 16, "full")

        module.qweight._scale = torch.ones(4, dtype=torch.float32)
        output = torch.empty((8, 2), dtype=torch.bfloat16)
        linear_prepared = mock.Mock(return_value=output)
        wgp.linear_prepared = linear_prepared
        with mock.patch.object(self.backend, "_weight_scale", return_value=torch.arange(4, dtype=torch.float32)):
            actual = self.backend._WGP_ROW_PROJECTION.rows(module, prepared, 1, 3)
        self.assertIs(actual, output)
        args, kwargs = linear_prepared.call_args
        self.assertIs(args[0], prepared)
        torch.testing.assert_close(args[1], module.qweight._data[1:3])
        torch.testing.assert_close(args[2], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(args[3], module.bias[1:3])

    def test_row_registration_requires_the_full_mmgp_contract(self):
        names = ("register_row_projection", "unregister_row_projection", "linear_rows_supported",
                 "prepare_linear_input", "linear_rows")
        compatible = SimpleNamespace(**{name: lambda *_args: None for name in names})
        self.assertTrue(self.backend._mmgp_row_projection_api_compatible(compatible))
        del compatible.linear_rows
        self.assertFalse(self.backend._mmgp_row_projection_api_compatible(compatible))


class TestKernelFamilyPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = _import_project_module("shared.kernels.kernel_policy")
        self.original = self.policy.families
        self.addCleanup(setattr, self.policy, "families", self.original)

    def test_selected_family_is_preferred_without_removing_fallbacks(self):
        self.policy.configure_families("kitchen")
        self.assertEqual(self.policy.families, ("kitchen", "wgpkernels", "triton", "pytorch"))
        self.assertEqual(self.policy.first({"wgpkernels": True, "kitchen": True}), "kitchen")

    def test_auto_keeps_default_family_order(self):
        self.policy.configure_families("auto")
        self.assertEqual(self.policy.families, self.policy.RANKING)


if __name__ == "__main__":
    unittest.main()
