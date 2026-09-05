import json
import tempfile
import unittest
from pathlib import Path

from app.services.search_index import SearchIndex


class GallerySearchIndexTests(unittest.TestCase):
    def _write_meta(self, root: Path, filename: str, params: dict) -> None:
        payload = {"generation_mode": "video", "params": params}
        (root / f"{filename}.meta.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_indexes_human_facing_h3_details_and_accelerations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            filename = "accelerated.mp4"
            self._write_meta(
                root,
                filename,
                {
                    "prompt": "Yoda trains beside a swamp",
                    "model_type": "minimax_h3_ref2va_pruned",
                    "resolution": "1280x704",
                    "num_inference_steps": 8,
                    "seed": 4242,
                    "activated_loras": ["MiniMax-H3-Ref2VA-Acc-PDD.safetensors"],
                    "minimax_h3_turbo_mode": True,
                    "minimax_h3_turbo_preset": "alibaba-pai-ref2va-pdd-8step",
                    "skip_steps_cache_type": "first_block",
                    "override_attention": "sol",
                },
            )

            index = SearchIndex()
            for query in (
                "omni",
                "pdd",
                "turbo",
                "first block",
                "sol engine",
                "1280x704",
                "720p",
                "ref2va acc",
                "8 steps",
                "seed 4242",
            ):
                self.assertEqual(index.search(query, str(root)), {filename}, query)

    def test_disabled_default_turbo_preset_is_not_a_false_positive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            filename = "ordinary.mp4"
            self._write_meta(
                root,
                filename,
                {
                    "model_type": "minimax_h3_ref2va",
                    "minimax_h3_turbo_mode": False,
                    "minimax_h3_turbo_preset": "v4-step600-ema",
                    "activated_loras": [],
                },
            )

            index = SearchIndex()
            self.assertEqual(index.search("omni", str(root)), {filename})
            self.assertEqual(index.search("turbo", str(root)), set())

    def test_indexes_effective_h3_window_prompts_not_only_source_concept(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            filename = "planned-omni.mp4"
            self._write_meta(
                root,
                filename,
                {
                    "prompt": "Three characters meet in a swamp",
                    "model_type": "minimax_h3_ref2va_pruned",
                    "h3_window_prompts": [
                        "Thanos questions Yoda beside the flooded tree roots",
                        "Blaine turns to dust after the gauntlet snap",
                    ],
                },
            )

            index = SearchIndex()
            self.assertEqual(index.search("flooded tree roots", str(root)), {filename})
            self.assertEqual(index.search("gauntlet snap", str(root)), {filename})


if __name__ == "__main__":
    unittest.main()
