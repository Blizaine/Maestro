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

    def _module(self, features=16, group=16, outputs=4):
        data = torch.arange(outputs * features, dtype=torch.int8).reshape(outputs, features)
        weight = SimpleNamespace(_data=data, _scale=torch.ones((outputs, 1), dtype=torch.float32))
        return SimpleNamespace(
            weight_qtype=SimpleNamespace(name="qint8_convrot"),
            _convrot_group_size=group,
            qweight=weight,
            bias=torch.arange(outputs, dtype=torch.bfloat16),
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

    def test_prepare_tiles_at_wgpkernels_grid_limit_for_long_video_rows(self):
        module = self._module()
        cases = (
            (65_535, (65_535,)),
            (65_536, (65_535, 1)),
            (80_400, (65_535, 14_865)),
            (236_012, (65_535, 65_535, 65_535, 39_407)),
        )
        for rows, expected_tiles in cases:
            with self.subTest(rows=rows):
                calls = []
                wgp = SimpleNamespace(
                    prepare=mock.Mock(side_effect=lambda tile, group, path: (
                        calls.append((tile.shape[0], group, path))
                        or SimpleNamespace(dtype=tile.dtype, device=tile.device)
                    ))
                )
                with mock.patch.object(self.backend, "_wgp", wgp):
                    with mock.patch.object(self.backend, "_wgp_path", return_value="block"):
                        prepared = self.backend._WGP_ROW_PROJECTION.prepare(
                            module, torch.empty((rows, 16), dtype=torch.bfloat16)
                        )

                self.assertEqual(tuple(call[0] for call in calls), expected_tiles)
                self.assertTrue(all(tile_rows <= 65_535 for tile_rows, _, _ in calls))
                self.assertTrue(all(group == 16 and path == "block" for _, group, path in calls))
                self.assertEqual(sum(tile_rows for tile_rows, _, _ in calls), rows)
                if len(expected_tiles) == 1:
                    self.assertIsInstance(prepared, SimpleNamespace)
                else:
                    self.assertEqual(prepared.row_count, rows)
                    self.assertEqual(
                        tuple((start, stop) for start, stop, _ in prepared.chunks),
                        tuple((sum(expected_tiles[:index]), sum(expected_tiles[:index + 1]))
                              for index in range(len(expected_tiles))),
                    )

    def test_scratch_tiled_prepared_rows_preserve_group_scales_and_math(self):
        module = self._module(features=64, group=64)
        torch.manual_seed(191)
        x = torch.randn((11, 64), dtype=torch.bfloat16)
        module.qweight._data = torch.randint(-6, 7, (4, 64), dtype=torch.int8)
        scales = torch.tensor([[0.01], [0.02], [0.03], [0.04]], dtype=torch.float32)
        module.qweight._scale = scales
        module.bias = torch.randn((4,), dtype=torch.bfloat16)
        prepare_calls = []

        def prepare(tile, group, path):
            prepare_calls.append((tile.shape[0], group, path))
            return SimpleNamespace(value=tile.clone(), group=group, dtype=tile.dtype, device=tile.device)

        def linear_prepared(prepared, weight, scale, bias):
            self.assertEqual(prepared.group, 64)
            result = prepared.value.float() @ weight.float().t()
            result.mul_(scale.reshape(1, -1))
            if bias is not None:
                result.add_(bias)
            return result.to(prepared.dtype)

        wgp = SimpleNamespace(prepare=prepare, linear_prepared=linear_prepared)
        with mock.patch.object(self.backend, "_SCRATCH_BYTES", (2 * 64 + 4) * 4):
            with mock.patch.object(self.backend, "_wgp", wgp):
                with mock.patch.object(
                    self.backend, "_wgp_path",
                    side_effect=lambda _module, _rows, group, precise: f"group-{group}-precise-{precise}",
                ) as choose_path:
                    with mock.patch.object(self.backend, "_weight_scale", return_value=scales):
                        prepared = self.backend._WGP_ROW_PROJECTION.prepare(module, x)
                        actual = self.backend._WGP_ROW_PROJECTION.rows(module, prepared, 1, 4)

        self.assertEqual([rows for rows, _, _ in prepare_calls], [4, 4, 3])
        self.assertTrue(all(group == 64 for _, group, _ in prepare_calls))
        self.assertEqual({path for _, _, path in prepare_calls}, {"group-64-precise-False"})
        self.assertEqual(choose_path.call_count, 1)
        self.assertEqual(choose_path.call_args.args[1].shape[0], x.shape[0])
        expected = x.float() @ module.qweight._data[1:4].float().t()
        expected.mul_(scales[1:4].reshape(1, -1)).add_(module.bias[1:4])
        torch.testing.assert_close(actual, expected.to(x.dtype), rtol=0, atol=0)

    def test_grouped_tiling_preserves_native_prepare_protocol_and_projection_order(self):
        qtype_module = _import_project_module("shared.qtypes.int8_convrot")
        qlinear_type = qtype_module.QLinearInt8ConvRot

        def make_module(weight_data, scales, bias):
            qtype = qtype_module._QINT8_CONVROT_QTYPE
            module = qlinear_type(
                16, weight_data.shape[0], bias=True, device="cpu",
                dtype=torch.bfloat16, weights=qtype, quantize_input=True,
            )
            weight = qtype_module.Int8ConvRotWeightTensor.create(
                qtype,
                0,
                tuple(weight_data.shape),
                tuple(weight_data.stride()),
                weight_data,
                scales,
                compute_dtype=torch.bfloat16,
            )
            delattr(module, "weight")
            module.weight = weight
            module._convrot_group_size = 16
            with torch.no_grad():
                module.bias.copy_(bias)
            return module

        torch.manual_seed(191)
        modules = [
            make_module(
                torch.randint(-5, 6, (8, 16), dtype=torch.int8),
                torch.linspace(0.01, 0.08, 8),
                torch.linspace(-0.2, 0.2, 8, dtype=torch.bfloat16),
            ),
            make_module(
                torch.randint(-5, 6, (8, 16), dtype=torch.int8),
                torch.linspace(0.03, 0.10, 8),
                torch.linspace(0.3, -0.3, 8, dtype=torch.bfloat16),
            ),
        ]
        x = torch.randn((2, 32_768, 16), dtype=torch.bfloat16)
        prepare_rows = []

        class Prepared:
            def __init__(self, value):
                self.value = value.clone()

            def rows(self, start, stop):
                return Prepared(self.value[start:stop])

        def prepare(tile, group, path):
            prepare_rows.append((tile.shape[0], group, path))
            return Prepared(tile)

        def linear_prepared(prepared, weight, scale, bias, *, out=None):
            result = prepared.value.float() @ weight.float().t()
            result.mul_(scale.reshape(1, -1))
            if bias is not None:
                result.add_(bias)
            if out is not None:
                out.copy_(result.to(out.dtype))
                return out
            return result.to(prepared.value.dtype)

        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(mock.patch.object(self.backend, "_backend", "wgpkernels"))
        patches.enter_context(mock.patch.object(self.backend, "_SCRATCH_BYTES", 16 * 1024 * 1024))
        patches.enter_context(mock.patch.object(self.backend, "_wgp", SimpleNamespace(
            prepare=prepare, linear_prepared=linear_prepared,
        )))
        patches.enter_context(mock.patch.object(self.backend, "_wgp_path", return_value="block"))
        patches.enter_context(mock.patch.object(self.backend, "_weight_scale", side_effect=lambda weight, _n, _device: weight._scale))
        patches.enter_context(mock.patch.object(self.backend.triton, "_is_fake_tensor", return_value=False))
        patches.enter_context(mock.patch.object(torch.compiler, "is_compiling", return_value=False))
        patches.enter_context(mock.patch.object(
            torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True,
        ))

        with torch.inference_mode():
            actual = self.backend.linear_grouped(modules, x)

        self.assertEqual([rows for rows, _, _ in prepare_rows], [65_535, 1])
        self.assertTrue(all(group == 16 and path == "block" for _, group, path in prepare_rows))
        self.assertEqual(actual.shape, (2, 32_768, 8))
        for index, module in enumerate(modules):
            expected = x[index].float() @ module.qweight._data.float().t()
            expected.mul_(module.qweight._scale.reshape(1, -1)).add_(module.bias)
            torch.testing.assert_close(actual[index], expected.to(x.dtype), rtol=0, atol=0)

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
