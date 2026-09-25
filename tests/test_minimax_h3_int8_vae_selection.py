"""CPU-only coverage for MiniMax H3 runtime VAE selection."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class TestMiniMaxH3RuntimeVAESelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.minimax_h3.minimax_h3_handler import family_handler

        cls.handler = family_handler

    @staticmethod
    def _downloaded_files(downloads):
        return [
            (download["repoId"], filename)
            for download in downloads
            for group in download["fileList"]
            for filename in group
        ]

    def test_int8_transformer_selects_and_downloads_convrot_vae(self):
        definition = self.handler.query_model_def("minimax_h3", {})
        resolved = self.handler.resolve_runtime_model_def(
            definition,
            {"transformer_quantization": "int8"},
        )

        self.assertEqual(
            resolved["minimax_h3_video_vae_filename"],
            "minimax_h3_video_vae_int8_convrot.safetensors",
        )
        downloads = self.handler.query_model_files([], "minimax_h3", resolved)
        files = self._downloaded_files(downloads)
        self.assertIn(
            (
                "Kijai/MiniMax-H3-experimental",
                "minimax_h3_video_vae_int8_convrot.safetensors",
            ),
            files,
        )
        self.assertNotIn(
            ("Comfy-Org/MiniMax-H3", "minimax_h3_video_vae_fp16.safetensors"),
            files,
        )

    def test_non_int8_runtime_keeps_fp16_vae_and_download(self):
        definition = self.handler.query_model_def("minimax_h3", {})

        for quantization in ("fp8", "bf16", None):
            with self.subTest(quantization=quantization):
                resolved = self.handler.resolve_runtime_model_def(
                    definition,
                    {"transformer_quantization": quantization},
                )
                self.assertEqual(
                    resolved["minimax_h3_video_vae_filename"],
                    "minimax_h3_video_vae_fp16.safetensors",
                )
                files = self._downloaded_files(
                    self.handler.query_model_files(
                        [], "minimax_h3", resolved
                    )
                )
                self.assertIn(
                    (
                        "Comfy-Org/MiniMax-H3",
                        "minimax_h3_video_vae_fp16.safetensors",
                    ),
                    files,
                )
                self.assertNotIn(
                    (
                        "Kijai/MiniMax-H3-experimental",
                        "minimax_h3_video_vae_int8_convrot.safetensors",
                    ),
                    files,
                )

    def test_explicit_user_vae_filename_wins_over_runtime_default(self):
        definition = self.handler.query_model_def(
            "minimax_h3",
            {"minimax_h3_video_vae_filename": "user/custom_h3_vae.safetensors"},
        )

        resolved = self.handler.resolve_runtime_model_def(
            definition,
            {"transformer_quantization": "int8"},
        )

        self.assertFalse(definition["minimax_h3_video_vae_auto"])
        self.assertEqual(
            resolved["minimax_h3_video_vae_filename"],
            "user/custom_h3_vae.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
