"""CPU regressions for Recast tracking-array lifetime."""

import ast
import gc
from pathlib import Path
import sys
import unittest
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.recast_tracking_memory import (  # noqa: E402
    release_recast_tracking_arrays,
)


def _call_name(node):
    function = node.func
    return function.id if isinstance(function, ast.Name) else None


def _contains_call(node, name):
    return any(
        isinstance(item, ast.Call) and _call_name(item) == name
        for item in ast.walk(node)
    )


class RecastTrackingMemoryTests(unittest.TestCase):
    def test_release_drops_arrays_and_views_but_preserves_materialized_metadata(self):
        source_frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        merged_masks = np.ones((3, 8, 8, 3), dtype=np.uint8)
        mapping_mask = np.full((3, 8, 8, 3), 2, dtype=np.uint8)
        source_view = source_frames[1]
        mask_view = merged_masks[1]
        tracking = {
            "source_frames": source_frames,
            "mask_frames": merged_masks,
            "mapping_masks": [mapping_mask],
            "video_mask": "materialized-mask.mp4",
            "fps": 25.0,
            "shot_ranges": [[0, 3]],
            "mapping_summaries": [{"anchor_frame_index": 1}],
            "timeline_aware": False,
        }
        mapping_anchors = [{
            "source_frame": source_view,
            "source_mask": mask_view,
        }]
        refs = [
            weakref.ref(source_frames),
            weakref.ref(merged_masks),
            weakref.ref(mapping_mask),
        ]
        del source_frames, merged_masks, mapping_mask, source_view, mask_view

        released = release_recast_tracking_arrays(tracking, mapping_anchors)
        gc.collect()

        self.assertEqual(
            released,
            ("source_frames", "mask_frames", "mapping_masks"),
        )
        self.assertEqual(mapping_anchors, [])
        self.assertTrue(all(reference() is None for reference in refs))
        self.assertEqual(tracking["video_mask"], "materialized-mask.mp4")
        self.assertEqual(tracking["fps"], 25.0)
        self.assertEqual(tracking["shot_ranges"], [[0, 3]])
        self.assertEqual(
            tracking["mapping_summaries"],
            [{"anchor_frame_index": 1}],
        )
        self.assertFalse(tracking["timeline_aware"])

    def test_release_is_after_optional_manifest_and_before_generation_dispatch(self):
        launch_path = ROOT / "app" / "launch.py"
        module = ast.parse(launch_path.read_text(encoding="utf-8"))
        recast_worker = next(
            node for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_recast"
        )
        worker_body = recast_worker.body
        prep_try_index = next(
            index for index, statement in enumerate(worker_body)
            if isinstance(statement, ast.Try)
            and _contains_call(statement, "_build_recast_shot_manifest")
            and _contains_call(statement, "release_recast_tracking_arrays")
        )
        prep_try = worker_body[prep_try_index]
        prep_scope = next(
            statement for statement in prep_try.body
            if isinstance(statement, ast.With)
            and _contains_call(statement, "_build_recast_shot_manifest")
        )
        manifest_if_index = next(
            index for index, statement in enumerate(prep_scope.body)
            if isinstance(statement, ast.If)
            and _contains_call(statement, "_build_recast_shot_manifest")
        )
        release_if_index = next(
            index for index, statement in enumerate(prep_scope.body)
            if isinstance(statement, ast.If)
            and _contains_call(statement, "release_recast_tracking_arrays")
        )
        self.assertLess(manifest_if_index, release_if_index)

        # It is a direct sibling of the optional shot-manifest branch, so the
        # same release runs when the single-branch condition is false.
        release_if = prep_scope.body[release_if_index]
        self.assertIsInstance(release_if, ast.If)
        release_index = next(
            index for index, statement in enumerate(release_if.body)
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "released_arrays"
                for target in statement.targets
            )
            and _contains_call(statement, "release_recast_tracking_arrays")
        )
        cleared_views = {
            target.id
            for statement in release_if.body[release_index + 1:]
            if isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is None
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        self.assertTrue({
                "probe_frame",
                "probe_mask",
                "scene_anchor_frame",
                "scene_anchor_mask",
                "mapping_anchors",
            }.issubset(cleared_views))

        requeue_index = next(
            index for index, statement in enumerate(worker_body)
            if isinstance(statement, ast.If)
            and _contains_call(statement, "try_requeue")
        )
        self.assertLess(prep_try_index, requeue_index)
        self.assertLess(release_if_index, len(prep_scope.body))
        dispatch_index = next(
            index for index, statement in enumerate(worker_body)
            if isinstance(statement, ast.If)
            and (
                _contains_call(statement, "_run_recast_shot_generation")
                or _contains_call(statement, "_run_generation")
            )
        )
        self.assertLess(requeue_index, dispatch_index)


if __name__ == "__main__":
    unittest.main()
