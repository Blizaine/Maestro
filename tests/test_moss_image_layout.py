"""CPU-only tests for the MOSS Image absolute-pixel draft compiler."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.moss_image_layout import compile_layout_draft, strict_loads  # noqa: E402


def draft(width=64, height=48, *, items=None, summary=None):
    return {
        "width": width,
        "height": height,
        "bbox_coord_space": "abs",
        "bbox_format": "xyxy",
        "summary": summary or {"caption": "A carefully composed scene."},
        "items": items if items is not None else [
            {"label": "object", "caption": "A small ceramic cup.", "bbox": [8, 8, 40, 40]}
        ],
    }


class MossImageLayoutCompilerTests(unittest.TestCase):
    def test_official_rectangular_example_uses_centered_yxyx_boxes(self):
        source = draft(
            1536,
            1024,
            summary={
                "meta": ["product campaign", "studio photography"],
                "canvas": "single",
                "caption": "A warm studio campaign for a handmade ceramic cup.",
            },
            items=[
                {
                    "label": "object",
                    "caption": "An ivory ceramic cup with a rounded handle.",
                    "bbox": [176, 224, 800, 864],
                }
            ],
        )

        compiled = compile_layout_draft(source, "1536x1024")
        result = strict_loads(compiled)

        self.assertEqual(result["items"][0]["bbox"], [-288, -592, 352, 32])
        self.assertEqual(result["bbox_coord_space"], "abs_centralized")
        self.assertEqual(result["bbox_format"], "yxyx")
        self.assertNotIn("width", result)
        self.assertNotIn("height", result)

    def test_fractional_edges_round_to_nine_places_then_enclose_pixels(self):
        source = draft(items=[{
            "label": "object",
            "caption": "A box with small floating point noise.",
            "bbox": [16.0000000004, 8.9999999996, 40.0000000004, 40.0000000004],
        }])

        result = strict_loads(compile_layout_draft(source, "64×48"))

        self.assertEqual(result["items"][0]["bbox"], [-15, -16, 16, 8])

    def test_multilingual_text_and_annotations_are_preserved(self):
        source = draft(
            summary={
                "caption": "An editorial collage with clear reading order.",
                "meta": ["editorial collage", "print design"],
                "canvas": "collage",
                "palette": ["cobalt blue", "warm white"],
            },
            items=[
                {
                    "label": "character",
                    "caption": "A smiling artist holding a brush.",
                    "bbox": [2, 3, 28, 40],
                    "tags": ["foreground", "artist"],
                    "palette": ["ochre", "navy"],
                },
                {
                    "label": "text",
                    "caption": "A crisp multilingual headline on two lines.",
                    "text": "把日常，过成诗。\nПривет, мир!\nこんにちは",
                    "bbox": [30, 5, 62, 22],
                },
            ],
        )
        original = copy.deepcopy(source)

        compiled = compile_layout_draft(source)
        result = strict_loads(compiled)

        self.assertEqual(source, original)
        self.assertEqual(result["summary"]["canvas"], "collage")
        self.assertEqual(result["summary"]["palette"], ["cobalt blue", "warm white"])
        self.assertEqual(result["items"][0]["label"], "character")
        self.assertEqual(result["items"][0]["tags"], ["foreground", "artist"])
        self.assertEqual(result["items"][0]["palette"], ["ochre", "navy"])
        self.assertEqual(result["items"][1]["text"], "把日常，过成诗。\nПривет, мир!\nこんにちは")
        self.assertIn("把日常，过成诗。", compiled)
        self.assertNotIn("\n", compiled)

    def test_canvas_dimensions_must_match_the_selected_resolution(self):
        source = draft(64, 48)

        with self.assertRaisesRegex(ValueError, "does not match selected canvas"):
            compile_layout_draft(source, "48x64")
        with self.assertRaisesRegex(ValueError, "WIDTHxHEIGHT"):
            compile_layout_draft(source, "auto")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            compile_layout_draft(source, "2064x2048")

    def test_canvas_dimensions_are_integer_multiples_of_sixteen_and_within_pixel_cap(self):
        for width, height in ((True, 48), (64.0, 48), (65, 48), (64, 47), (0, 48), (-16, 48),
                              (4096, 2048)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                compile_layout_draft(draft(width, height))

    def test_unknown_or_missing_schema_fields_are_rejected(self):
        bad_root = draft()
        bad_root["id"] = "editor-node"
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            compile_layout_draft(bad_root)

        bad_summary = draft()
        bad_summary["summary"]["notes"] = "not a prompt field"
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            compile_layout_draft(bad_summary)

        bad_item = draft()
        bad_item["items"][0]["children"] = []
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            compile_layout_draft(bad_item)

        missing_root = draft()
        del missing_root["bbox_format"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            compile_layout_draft(missing_root)

    def test_canvas_kind_summary_arrays_and_item_arrays_are_validated(self):
        for canvas in ("poster", "", None, True):
            source = draft(summary={"caption": "Scene.", "canvas": canvas})
            with self.subTest(canvas=canvas), self.assertRaises(ValueError):
                compile_layout_draft(source)

        for field, value in (("meta", "photography"), ("palette", ["  "]), ("meta", [4])):
            source = draft(summary={"caption": "Scene.", field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                compile_layout_draft(source)

        for items in ([], {}, [None], [["nested item"]]):
            with self.subTest(items=items), self.assertRaises(ValueError):
                compile_layout_draft(draft(items=items))

    def test_item_labels_text_and_tags_are_validated_without_rewriting(self):
        for label in ("animal", "", 1, True):
            source = draft(items=[{"label": label, "caption": "Thing.", "bbox": [1, 1, 2, 2]}])
            with self.subTest(label=label), self.assertRaises(ValueError):
                compile_layout_draft(source)

        missing_text = draft(items=[{"label": "text", "caption": "A headline.", "bbox": [1, 1, 2, 2]}])
        with self.assertRaisesRegex(ValueError, "require literal text"):
            compile_layout_draft(missing_text)

        for text in ("", " \n ", 123):
            source = draft(items=[{
                "label": "text", "caption": "A headline.", "text": text, "bbox": [1, 1, 2, 2]
            }])
            with self.subTest(text=text), self.assertRaises(ValueError):
                compile_layout_draft(source)

        for field, value in (("caption", "\t"), ("tags", "single tag"), ("tags", [""])):
            item = {"label": "object", "caption": "Object.", "bbox": [1, 1, 2, 2]}
            item[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                compile_layout_draft(draft(items=[item]))

    def test_boxes_reject_boolean_nonfinite_degenerate_and_out_of_bounds_values(self):
        bad_boxes = (
            [True, 0, 2, 3],
            [0, 0, False, 3],
            [0, 0, float("nan"), 3],
            [0, 0, float("inf"), 3],
            [0, 0, 0, 3],
            [3, 0, 2, 3],
            [-1, 0, 2, 3],
            [0, 0, 65, 3],
            [0, 0, 2, 49],
            [0, 1, 2],
            (0, 0, 2, 3),
        )
        for box in bad_boxes:
            source = draft(items=[{"label": "object", "caption": "Object.", "bbox": box}])
            with self.subTest(box=box), self.assertRaises(ValueError):
                compile_layout_draft(source)

        too_small_after_rounding = draft(items=[{
            "label": "object", "caption": "Tiny object.",
            "bbox": [2.0000000001, 2.0000000001, 2.0000000002, 2.0000000002],
        }])
        with self.assertRaisesRegex(ValueError, "too small"):
            compile_layout_draft(too_small_after_rounding)

    def test_strict_loader_rejects_duplicate_keys_nonfinite_constants_and_non_objects(self):
        for text in (
            '{"caption":"one","caption":"two"}',
            '{"summary":{"caption":"a","caption":"b"}}',
            '{"value":NaN}',
            '{"value":Infinity}',
            '{"value":-Infinity}',
            '{"value":1e999}',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_loads(text)

        with self.assertRaisesRegex(ValueError, "must be an object"):
            strict_loads('[1,2,3]')
        with self.assertRaises(ValueError):
            strict_loads("not JSON")

    def test_compiler_accepts_only_new_drafts_and_does_not_interpret_centered_legacy_boxes(self):
        legacy = {
            "summary": {"caption": "Already compiled."},
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
            "items": [{"label": "object", "caption": "Object.", "bbox": [-10, -10, 10, 10]}],
        }
        with self.assertRaises(ValueError):
            compile_layout_draft(legacy)


if __name__ == "__main__":
    unittest.main()
