"""CPU-only contract tests for the H3 Veda policy and tiling rules."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3.veda import attention, plans, selection, tiling


def _plan(name="landscape", grid=(2, 2, 4)):
    return {
        "geometry": name,
        "grid": list(grid),
        "shapes": ["2x2x32"],
        "head_shape": [[0] * 56 for _ in range(50)],
    }


def _metadata(plan=None):
    return {
        "format": attention.FORMAT,
        "num_layers": "50",
        "num_heads": "56",
        "head_dim": "128",
        "plans": json.dumps({"landscape": plan or _plan()}),
        "target_budget_kind": "ratio",
        "target_budget_value": "0.25",
        "ref_budget_kind": "tiles",
        "ref_budget_value": "1",
    }


class _MetadataReader:
    def __init__(self, metadata):
        self._metadata = metadata

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def metadata(self):
        return self._metadata


class TestVedaMetadataAndPolicy(unittest.TestCase):
    def test_sparse_int8_real_import_uses_maestro_logger_api(self):
        module_names = (
            "models.minimax_h3.veda.sparse_int8",
            "shared.kernels.triton_tuner",
        )
        previous = {name: sys.modules.get(name) for name in module_names}
        try:
            for name in module_names:
                sys.modules.pop(name, None)
            sparse_int8 = importlib.import_module(module_names[0])
            self.assertTrue(callable(sparse_int8.attend))
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
            for name, module in previous.items():
                if module is not None:
                    sys.modules[name] = module

    def test_metadata_requires_pinned_format_and_h3_predictor_geometry(self):
        good = _metadata()
        with mock.patch.object(attention, "safe_open", return_value=_MetadataReader(good)):
            self.assertEqual(attention.read_metadata("predictor.safetensors"), good)

        for metadata, message in (
            ({**good, "format": "unknown"}, "Expected a WanGP H3 Veda"),
            ({**good, "num_heads": "55"}, "50-layer, 56-head"),
            (
                {key: value for key, value in good.items() if key != "head_dim"},
                "missing valid predictor geometry",
            ),
        ):
            with mock.patch.object(attention, "safe_open", return_value=_MetadataReader(metadata)):
                with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, message):
                    attention.read_metadata("predictor.safetensors")

    def test_malformed_plans_and_nonfinite_budgets_are_rejected(self):
        malformed = _plan()
        malformed["head_shape"][0] = [0] * 55
        with self.assertRaisesRegex(ValueError, "56 heads"):
            attention.VedaPolicy(_metadata(malformed))

        malformed = _plan()
        malformed["head_shape"][3][8] = 1
        with self.assertRaisesRegex(ValueError, "unknown tile shape"):
            attention.VedaPolicy(_metadata(malformed))

        metadata = _metadata()
        metadata["target_budget_value"] = "nan"
        with self.assertRaisesRegex(ValueError, "Invalid Veda target tile budget"):
            attention.VedaPolicy(metadata)

    def test_plan_selection_handles_exact_and_nearest_geometries(self):
        table = plans.PlanTable([plans.TilePlan.from_json(_plan())])
        exact = table.select((2, 2, 4))
        self.assertTrue(exact.exact)
        self.assertEqual(exact.plan.grid, (2, 2, 4))

        nearest = table.select((2, 2, 5))
        self.assertFalse(nearest.exact)
        self.assertIn("nearest trained size", nearest.how)

    def test_disabled_policy_skips_gpu_validation_and_geometry_building(self):
        policy = attention.VedaPolicy(_metadata())
        with mock.patch.object(attention.offload, "shared_state", {"_attention": "dense"}):
            policy.begin_forward(None, (2, 2, 4), {}, torch.device("cpu"), torch.float32)
        self.assertFalse(policy.enabled)
        self.assertIsNone(policy.plan)
        self.assertEqual(policy.layouts, {})

    def test_head_groups_keep_shape_assignment_and_local_indices(self):
        policy = attention.VedaPolicy(_metadata())
        policy.plan = policy.plans.select((2, 2, 4)).plan
        layout = tiling.build_tile_layout(
            [tiling.TiledSpan(0, (1, 1, 3), tiling.TileShape(1, 1, 128))], 3
        )
        policy.layouts[policy.plan.shapes[0]] = layout
        groups = policy.groups(3, range(8, 13), torch.device("cpu"))
        self.assertEqual(len(groups), 1)
        self.assertIs(groups[0][0], layout)
        self.assertEqual(groups[0][1], (8, 9, 10, 11, 12))
        self.assertEqual(groups[0][2].tolist(), [0, 1, 2, 3, 4])

    def test_generation_context_resets_geometry_when_generation_is_cancelled(self):
        policy = attention.VedaPolicy(_metadata())
        policy.signature = ("old",)
        policy.plan = policy.plans.select((2, 2, 4)).plan
        policy.layouts[policy.plan.shapes[0]] = object()

        class Cancelled(Exception):
            pass

        class Harness:
            def __init__(self):
                self.transformer = SimpleNamespace(veda_attention=policy)

            @attention.generation_context
            def generate(self):
                raise Cancelled("cancel")

        with self.assertRaises(Cancelled):
            Harness().generate()
        self.assertIsNone(policy.signature)
        self.assertIsNone(policy.plan)
        self.assertEqual(policy.layouts, {})
        self.assertEqual(policy.head_groups, {})


class TestVedaCpuTiling(unittest.TestCase):
    def test_tile_layout_gather_and_scatter_preserve_packed_rows(self):
        shape = tiling.TileShape(1, 1, 128)
        layout = tiling.build_tile_layout(
            [
                tiling.TiledSpan(0, (1, 1, 3), shape),
                tiling.TiledSpan(4, (1, 1, 3), shape),
            ],
            7,
        )
        self.assertEqual(layout.valid_count.tolist(), [3, 3, 1])
        self.assertEqual(layout.n_ref_tiles, 1)
        self.assertEqual(layout.n_video_tiles, 2)
        self.assertEqual(layout.n_global_tiles, 1)

        x = torch.arange(7 * 2 * 3, dtype=torch.float32).view(7, 2, 3)
        heads = torch.tensor([1])
        gathered = tiling.gather_tiles(x, layout, heads)
        self.assertTrue(torch.equal(gathered[:3, 0], x[:3, 1]))
        self.assertTrue(torch.equal(gathered[128:131, 0], x[4:7, 1]))
        self.assertEqual(torch.count_nonzero(gathered[3:128]).item(), 0)

        out = torch.zeros(8, 2, 3)
        tiling.scatter_tiles_(out, gathered, layout, heads)
        self.assertTrue(torch.equal(out[:7, 1], x[:, 1]))
        self.assertTrue(torch.equal(out[7], torch.zeros_like(out[7])))

    def test_cpu_trip_pool_excludes_padding_from_extrema(self):
        layout = tiling.build_tile_layout(
            [tiling.TiledSpan(0, (1, 1, 3), tiling.TileShape(1, 1, 128))], 3
        )
        x = torch.tensor([-3.0, -4.0, -2.0]).view(3, 1, 1)
        tiled = tiling.gather_tiles(x, layout, torch.tensor([0]))
        pooled = attention.pool_tiles(tiled, layout)
        self.assertTrue(torch.allclose(pooled[0, 0], torch.tensor([-3.0, -2.0, -4.0])))

    def test_selection_forces_own_tiles_and_keeps_global_queries_dense(self):
        shape = tiling.TileShape(1, 1, 128)
        layout = tiling.build_tile_layout(
            [
                tiling.TiledSpan(0, (1, 1, 2), shape),
                tiling.TiledSpan(3, (1, 1, 2), shape),
            ],
            5,
        )
        blocks = selection.column_blocks(
            layout, selection.Budget(tiles=1), selection.Budget(tiles=1)
        )
        scores = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        index, keep = selection.select(scores, layout, blocks)
        for row in range(layout.n_video_tiles):
            selected = index[0, row][keep[0, row]].tolist()
            self.assertIn(row, selected)
        mask = selection.block_mask(index, keep, layout)
        self.assertTrue(mask[0, layout.n_video_tiles].all())
        self.assertTrue(mask[0, :, layout.n_video_tiles].all())


if __name__ == "__main__":
    unittest.main()
