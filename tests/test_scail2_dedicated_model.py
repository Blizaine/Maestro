"""Contract tests for Maestro's dedicated upstream SCAIL-2 transformer."""
from __future__ import annotations

import os
import sys
import types
import unittest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)


class TestDedicatedScail2Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from models.wan.modules.model_scail2 import SCAIL2Model
        except ImportError as exc:
            raise unittest.SkipTest(
                "SCAIL-2 model dependencies are unavailable",
            ) from exc
        cls.torch = torch
        cls.model_class = SCAIL2Model

    def test_compact_checkpoint_embedding_names_are_remapped(self):
        torch = self.torch
        state = {
            "pose_patch_embedding.weight": torch.ones(1),
            "mask_patch_embedding.bias": torch.ones(1),
            "blocks.0.self_attn.q.weight": torch.ones(1),
        }

        converted = self.model_class.preprocess_sd_with_dtype(
            torch.bfloat16, state,
        )

        self.assertIn("patch_embedding_pose.weight", converted)
        self.assertIn("patch_embedding_mask.bias", converted)
        self.assertIn("blocks.0.self_attn.q.weight", converted)
        self.assertNotIn("pose_patch_embedding.weight", converted)
        self.assertNotIn("mask_patch_embedding.bias", converted)

    def test_maestro_batches_are_normalized_to_upstream_sample_lists(self):
        torch = self.torch
        batch = torch.zeros((2, 4, 3, 5, 7))

        samples = self.model_class._as_sample_list(batch, name="test")

        self.assertEqual(len(samples), 2)
        self.assertEqual(tuple(samples[0].shape), (4, 3, 5, 7))

    def test_tiny_forward_keeps_reference_regions_separate(self):
        torch = self.torch
        torch.manual_seed(7)
        model = self.model_class(
            model_type="i2v",
            patch_size=(1, 2, 2),
            text_len=8,
            in_dim=8,
            mask_dim=28,
            dim=24,
            ffn_dim=48,
            freq_dim=8,
            text_dim=12,
            out_dim=4,
            num_heads=3,
            num_layers=1,
        ).eval()

        result = model(
            x=[torch.randn(1, 4, 2, 4, 4)],
            t=torch.tensor([500.0]),
            context=[torch.randn(1, 5, 12)],
            clip_fea=torch.randn(1, 257, 1280),
            scail2_pose_latents=torch.randn(1, 4, 2, 2, 2),
            scail2_driving_masks=torch.randn(1, 28, 2, 2, 2),
            scail2_ref_latents=torch.randn(1, 4, 1, 4, 4),
            scail2_ref_masks=torch.randn(1, 28, 3, 4, 4),
            scail2_history_mask=torch.ones(1, 4, 2, 4, 4),
            scail2_additional_ref_latents=torch.randn(1, 4, 1, 4, 4),
            scail2_additional_ref_masks=torch.randn(1, 28, 1, 4, 4),
            scail2_replace_flag=True,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (1, 4, 2, 4, 4))

    def test_any2video_routes_only_scail2_to_the_dedicated_class(self):
        path = os.path.join(_APP, "models", "wan", "any2video.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("MAESTRO_SCAIL2_DEDICATED_MODEL", source)
        self.assertIn("from .modules.model_scail2 import SCAIL2Model", source)
        self.assertIn(
            "transformer_model_class = SCAIL2Model",
            source,
        )
        self.assertIn(
            "transformer_model_class = WanModel",
            source,
        )

    def test_conditioning_passes_official_regions_independently(self):
        path = os.path.join(
            _APP, "models", "wan", "scail2", "__init__.py",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        for key in (
            '"scail2_ref_latents"',
            '"scail2_additional_ref_latents"',
            '"scail2_ref_masks"',
            '"scail2_additional_ref_masks"',
            '"scail2_pose_latents"',
            '"scail2_driving_masks"',
            '"scail2_history_mask"',
            '"scail2_replace_flag"',
        ):
            self.assertIn(key, source)

    def test_dedicated_continuation_prefers_reencoded_pixel_history(self):
        path = os.path.join(
            _APP, "models", "wan", "scail2", "__init__.py",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        rendered_history_branch = source.index(
            "if dedicated_model and prefix_frames_count > 0 "
            "and input_video is not None:",
        )
        legacy_latent_branch = source.index(
            "elif overlapped_latents is not None",
            rendered_history_branch,
        )

        self.assertLess(rendered_history_branch, legacy_latent_branch)
        self.assertIn(
            "pipeline.vae.encode([history_frames], VAE_tile_size)",
            source[rendered_history_branch:legacy_latent_branch],
        )

    def test_dedicated_recast_reuses_a_stable_source_scene_reference(self):
        scail2_path = os.path.join(
            _APP, "models", "wan", "scail2", "__init__.py",
        )
        handler_path = os.path.join(
            _APP, "models", "wan", "wan_handler.py",
        )
        launch_path = os.path.join(_APP, "launch.py")
        with open(scail2_path, "r", encoding="utf-8") as handle:
            scail2_source = handle.read()
        with open(handler_path, "r", encoding="utf-8") as handle:
            handler_source = handle.read()
        with open(launch_path, "r", encoding="utf-8") as handle:
            launch_source = handle.read()

        for key in (
            '"scail2_source_scene_reference_path"',
            '"scail2_source_scene_mask_path"',
        ):
            self.assertIn(key, scail2_source)
            self.assertIn(key, handler_source)
            self.assertIn(key, launch_source)
        self.assertIn(
            "def _load_static_source_scene_reference",
            scail2_source,
        )
        self.assertIn(
            '"stable_official_reference"',
            launch_source,
        )

    def test_dedicated_model_skips_legacy_identity_reinforcement(self):
        path = os.path.join(
            _APP, "models", "wan", "scail2", "__init__.py",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn(
            "None\n        if dedicated_model\n        else "
            "_load_clip_identity_reference(",
            source,
        )
        isolation_branch = source.index(
            "use_identity_latent_isolation = (",
        )
        self.assertIn(
            "not dedicated_model",
            source[isolation_branch:isolation_branch + 300],
        )
        self.assertIn(
            "legacy CLIP/VAE identity reinforcement",
            source,
        )

    def test_dedicated_attention_uses_shared_backend_contract(self):
        from models.wan.modules import scail2_attention

        torch = self.torch
        captured = {}

        def pay_attention(qkv_list, **kwargs):
            captured["qkv"] = list(qkv_list)
            captured["kwargs"] = kwargs
            return torch.ones_like(qkv_list[0])

        dispatcher = (
            pay_attention,
            lambda: "sdpa",
            types.SimpleNamespace(shared_state={}),
        )
        query = torch.ones((1, 2, 1, 4), dtype=torch.float32)
        result = scail2_attention._shared_attention(
            dispatcher,
            query,
            query,
            query,
            q_lens=None,
            k_lens=None,
            dropout_p=0.0,
            softmax_scale=None,
            q_scale=2.0,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            dtype=torch.bfloat16,
            fa_version=None,
        )

        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(captured["qkv"][0].dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(
                captured["qkv"][0],
                torch.full_like(captured["qkv"][0], 2.0),
            )
        )
        self.assertEqual(captured["kwargs"]["force_attention"], "sdpa")

    def test_scail2_flash_imports_handle_binary_load_failures(self):
        path = os.path.join(
            _APP,
            "models",
            "wan",
            "modules",
            "scail2_attention.py",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertGreaterEqual(
            source.count("except (ImportError, OSError):"),
            2,
        )
        self.assertIn("return _shared_attention(", source)


if __name__ == "__main__":
    unittest.main()
