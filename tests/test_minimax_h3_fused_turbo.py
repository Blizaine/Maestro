"""Regression coverage for the optional MATLOWAI fused H3 recipe."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

FRAMES_DEFAULT = APP / "defaults" / "minimax_h3_fused_turbo.json"
REFERENCES_DEFAULT = APP / "defaults" / "minimax_h3_ref2va_fused_turbo.json"


def _load_default(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestFusedH3Definitions(unittest.TestCase):
    def test_frames_and_references_share_one_pinned_checkpoint(self):
        frames = _load_default(FRAMES_DEFAULT)
        references = _load_default(REFERENCES_DEFAULT)
        frames_model = frames["model"]
        references_model = references["model"]

        self.assertEqual(frames_model["URLs"], references_model["URLs"])
        self.assertIn(
            "3b51096a1bf67608d98131116558202208fcf195",
            frames_model["URLs"][0],
        )
        self.assertEqual(
            frames_model["source_sha256"],
            "4262e4e9963c553fa00016bbe83961407a4fc0a888be95fd836c8d4f2304e48b",
        )
        self.assertEqual(
            frames_model["video_vae_source_sha256"],
            "9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410",
        )
        self.assertEqual(
            frames_model["video_vae_source_revision"],
            "a3e7d8da4ae7ba8df0779094cf5ab9d6ee855fe4",
        )
        self.assertEqual(
            frames_model["video_vae_source_sha256"],
            references_model["video_vae_source_sha256"],
        )
        self.assertEqual(frames["num_inference_steps"], 4)
        self.assertEqual(references["num_inference_steps"], 4)
        self.assertEqual(frames["video_length"], 243)
        self.assertEqual(references["video_length"], 243)
        self.assertEqual(frames["sliding_window_size"], 243)
        self.assertEqual(references["sliding_window_size"], 243)
        self.assertEqual(frames_model["minimax_h3_sampler"], "res_multistep")
        self.assertEqual(frames_model["minimax_h3_qkv_layout"], "grouped")
        self.assertEqual(references_model["minimax_h3_qkv_layout"], "grouped")
        self.assertFalse(frames_model["lock_inference_steps"])
        self.assertFalse(references_model["lock_inference_steps"])
        self.assertEqual(frames_model["inference_steps_min"], 4)
        self.assertEqual(frames_model["inference_steps_max"], 8)
        self.assertEqual(frames_model["inference_steps_label"], "Total Steps")
        self.assertTrue(frames_model["loras_disabled"])
        self.assertTrue(references_model["loras_disabled"])
        self.assertEqual(frames_model["architecture"], "minimax_h3")
        self.assertEqual(references_model["architecture"], "minimax_h3_ref2va")

    def test_handler_adds_experimental_vae_and_model_notices(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        partial = {"minimax_h3_fused_turbo": True}
        definition = family_handler.query_model_def("minimax_h3", partial)
        reference_definition = family_handler.query_model_def(
            "minimax_h3_ref2va",
            partial,
        )
        downloads = family_handler.query_model_files(
            [],
            "minimax_h3",
            definition,
        )

        self.assertTrue(definition["sla_attention"])
        self.assertFalse(definition["sol_attention"])
        self.assertFalse(definition["first_block_cache"])
        self.assertFalse(definition["lock_inference_steps"])
        self.assertEqual(definition["inference_steps_min"], 4)
        self.assertEqual(definition["inference_steps_max"], 8)
        self.assertEqual(definition["minimax_h3_qkv_layout"], "grouped")
        self.assertEqual(
            definition["sliding_window_defaults"]["window_default"],
            243,
        )
        self.assertEqual(
            definition["sliding_window_memory_policy"]["checkpoint"],
            "fused_4step",
        )
        self.assertTrue(reference_definition["omni_reference"])
        self.assertTrue(reference_definition["supports_reference_audio"])
        self.assertEqual(
            reference_definition["omni_sequence_memory_policy"][
                "reference_margin_steps"
            ],
            0,
        )
        flattened = [
            item
            for download in downloads
            for group in download["fileList"]
            for item in group
        ]
        self.assertIn("minimax_h3_video_vae_int8_convrot.safetensors", flattened)
        self.assertIn("LICENSE", flattened)
        self.assertIn("NOTICE", flattened)

    def test_baked_recipe_rejects_user_loras_and_removes_stale_turbo(self):
        from models.minimax_h3.fused_turbo import normalize_fused_h3_request

        body = {
            "activated_loras": ["MiniMax-H3-FL2VA-Acc-8Step.safetensors"],
            "loras_multipliers": "1.00",
            "num_inference_steps": 6,
            "guidance_scale": 4.0,
            "override_attention": "",
            "skip_steps_cache_type": "first_block",
            "minimax_h3_turbo_mode": True,
        }
        self.assertEqual(normalize_fused_h3_request(body), 1)
        self.assertEqual(body["activated_loras"], [])
        self.assertEqual(body["num_inference_steps"], 6)
        self.assertEqual(body["guidance_scale"], 1.0)
        self.assertEqual(body["flow_shift"], 12.0)
        self.assertEqual(body["audio_flow_shift"], 3.0)
        self.assertEqual(body["override_attention"], "sla")
        self.assertEqual(body["skip_steps_cache_type"], "")

        with self.assertRaisesRegex(ValueError, "additional LoRAs"):
            normalize_fused_h3_request(
                {"activated_loras": ["my_character.safetensors"]}
            )

        for invalid_steps in (3, 9, 5.5, "many"):
            with self.subTest(invalid_steps=invalid_steps):
                with self.assertRaisesRegex(ValueError, "steps"):
                    normalize_fused_h3_request(
                        {
                            "activated_loras": [],
                            "num_inference_steps": invalid_steps,
                        }
                    )

        default_body = {"activated_loras": []}
        normalize_fused_h3_request(default_body)
        self.assertEqual(default_body["num_inference_steps"], 4)

    def test_attribution_and_ui_contracts_are_present(self):
        notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        ui = (
            ROOT
            / "ui"
            / "src"
            / "components"
            / "Sidebar"
            / "MiniMaxH3Optimizations.tsx"
        ).read_text(encoding="utf-8")
        advanced = (
            ROOT
            / "ui"
            / "src"
            / "components"
            / "Sidebar"
            / "AdvancedSettings.tsx"
        ).read_text(encoding="utf-8")
        director = (
            ROOT
            / "ui"
            / "src"
            / "components"
            / "Sidebar"
            / "DirectorH3Optimizations.tsx"
        ).read_text(encoding="utf-8")
        store = (
            ROOT / "ui" / "src" / "stores" / "useStore.ts"
        ).read_text(encoding="utf-8")
        curated_defaults = store.split(
            "const DEFAULT_ENABLED_MODELS = new Set([",
            1,
        )[1].split("])" , 1)[0]

        self.assertIn("MATLOWAI MiniMax H3 fused four-step checkpoint", notice)
        self.assertIn("fd26ffb89dee294ca740a59632e5b3423b9a9d2a", notice)
        self.assertIn("Fused Turbo Recipe", ui)
        self.assertIn("Total Steps can be adjusted from 4-8", ui)
        self.assertIn("SLA Sparse Attention", ui)
        self.assertIn("inference_steps_min", advanced)
        self.assertIn("inference_steps_max", advanced)
        self.assertIn("!modelOptions?.loras_disabled", advanced)
        self.assertIn("Fused Turbo Recipe", director)
        self.assertIn("startsWith('ltx2')", advanced)
        self.assertIn(
            "9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410",
            notice,
        )
        self.assertIn("'minimax_h3_fused_turbo'", curated_defaults)
        self.assertIn("'minimax_h3_ref2va_fused_turbo'", curated_defaults)
        self.assertIn(
            "11: ['minimax_h3_fused_turbo', 'minimax_h3_ref2va_fused_turbo']",
            store,
        )
        active_defaults = store.split(
            "const modeDefaultModel: Record<GenerationMode, string> = {",
            1,
        )[1].split("}\n", 1)[0]
        self.assertNotIn("minimax_h3_fused_turbo", active_defaults)
        self.assertNotIn("minimax_h3_ref2va_fused_turbo", active_defaults)
        self.assertIn(
            "minimax_h3_fused_turbo: { firstLast: 'minimax_h3_fused_turbo', omni: 'minimax_h3_ref2va_fused_turbo' }",
            store,
        )

    def test_res_audio_scale_and_shared_sla_policy_are_wired(self):
        pipeline = (
            APP / "models" / "minimax_h3" / "minimax_h3_main.py"
        ).read_text(encoding="utf-8")
        transformer = (
            APP / "models" / "minimax_h3" / "transformer.py"
        ).read_text(encoding="utf-8")

        compact_pipeline = " ".join(pipeline.split())
        self.assertIn(
            "audio_target[generated_audio_local_indices] / audio_scale",
            compact_pipeline,
        )
        self.assertNotIn(
            "audio_target[generated_audio_local_indices].div_(audio_scale)",
            pipeline,
        )
        self.assertIn(
            "self.sla_attention = MiniMaxH3SLAAttention(sla_config)",
            transformer,
        )
        self.assertIn("sla_attention=self.sla_attention", transformer)

    def test_fused_convrot_qkv_keeps_native_grouped_rows_before_split(self):
        """The baked checkpoint must retain its contiguous Q/K/V row groups."""

        import torch

        from models.minimax_h3.minimax_h3_main import (
            _strip_transformer_wrappers,
        )

        grouped = torch.arange(24, dtype=torch.int8).reshape(12, 2)
        descriptor = torch.tensor(
            list(
                json.dumps(
                    {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": 256,
                    }
                ).encode("utf-8")
            ),
            dtype=torch.uint8,
        )
        state_dict = {
            "model.diffusion_model.blocks.0.attn.q_norm.weight": torch.ones(2),
            "model.diffusion_model.blocks.0.attn.qkv_proj.weight": grouped.clone(),
            "model.diffusion_model.blocks.0.attn.qkv_proj.comfy_quant": descriptor,
        }

        normalized, _, _ = _strip_transformer_wrappers(
            state_dict,
            interleave_qkv=False,
        )

        self.assertTrue(
            torch.equal(normalized["blocks.0.attn.qkv_proj.weight"], grouped)
        )

        pipeline = (
            APP / "models" / "minimax_h3" / "minimax_h3_main.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'if qkv_layout in {"grouped", "interleaved"}',
            pipeline,
        )

    def test_fused_frames_policy_promotes_validated_24gb_window(self):
        from models.minimax_h3.minimax_h3_handler import (
            recommended_h3_window_profile,
        )

        default_canvas = recommended_h3_window_profile(
            24,
            "1152x640",
            {"minimax_h3_fused_turbo": True},
        )
        tested_720p = recommended_h3_window_profile(
            24,
            "1280x704",
            {"minimax_h3_fused_turbo": True},
        )
        tested_native_768p = recommended_h3_window_profile(
            24,
            "1344x768",
            {"minimax_h3_fused_turbo": True},
        )

        self.assertEqual(default_canvas["checkpoint"], "fused_4step")
        self.assertEqual(default_canvas["frames"], 345)
        self.assertEqual(tested_720p["frames"], 345)
        self.assertEqual(tested_native_768p["frames"], 345)

    def test_fused_policy_keeps_unvalidated_tiers_conservative(self):
        from models.minimax_h3.minimax_h3_handler import (
            recommended_h3_window_profile,
        )

        cases = [
            (23, "1280x704", 243),
            (23, "1344x768", 243),
            (16, "1152x640", 243),
            (24, "1664x704", 243),
        ]
        for vram_gb, resolution, expected_frames in cases:
            with self.subTest(vram_gb=vram_gb, resolution=resolution):
                profile = recommended_h3_window_profile(
                    vram_gb,
                    resolution,
                    {"minimax_h3_fused_turbo": True},
                )
                self.assertEqual(profile["frames"], expected_frames)

    def test_fused_references_policy_keeps_published_window(self):
        from models.minimax_h3.minimax_h3_handler import (
            family_handler,
            recommended_h3_window_profile,
        )

        reference_model = {
            "minimax_h3_fused_turbo": True,
            "architecture": "minimax_h3_ref2va",
        }
        profile = recommended_h3_window_profile(
            24,
            "1280x704",
            reference_model,
        )
        definition = family_handler.query_model_def(
            "minimax_h3_ref2va",
            {"minimax_h3_fused_turbo": True},
        )

        self.assertEqual(profile["checkpoint"], "fused_4step_references")
        self.assertEqual(profile["frames"], 243)
        self.assertEqual(
            definition["omni_sequence_memory_policy"]["checkpoint"],
            "fused_4step_references",
        )


class TestFusedH3Scheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("Fused H3 scheduler tests require PyTorch") from error
        cls.torch = torch

    def test_four_requested_evaluations_have_four_res_intervals(self):
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler
        from models.minimax_h3.turbo import h3_scheduler_grid_points

        scheduler = MiniMaxH3Scheduler(shift=12.0, solver="res_multistep")
        scheduler.set_timesteps(
            h3_scheduler_grid_points(4, turbo_active=False),
            device="cpu",
        )

        self.assertEqual(len(scheduler.timesteps), 4)
        self.assertEqual(len(scheduler.sigmas), 5)
        self.assertEqual(len(scheduler._res_coefficients), 4)
        self.assertEqual(scheduler.coefficients_for_step(3)[2], 0.0)

    def test_six_and_eight_requested_evaluations_build_complete_res_schedules(self):
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler
        from models.minimax_h3.turbo import h3_scheduler_grid_points

        for evaluations in (6, 8):
            with self.subTest(evaluations=evaluations):
                scheduler = MiniMaxH3Scheduler(
                    shift=12.0,
                    solver="res_multistep",
                )
                scheduler.set_timesteps(
                    h3_scheduler_grid_points(
                        evaluations,
                        turbo_active=False,
                    ),
                    device="cpu",
                )
                self.assertEqual(len(scheduler.timesteps), evaluations)
                self.assertEqual(len(scheduler.sigmas), evaluations + 1)
                self.assertEqual(
                    len(scheduler._res_coefficients),
                    evaluations,
                )
                self.assertEqual(
                    scheduler.coefficients_for_step(evaluations - 1)[2],
                    0.0,
                )

    def test_res_uses_history_after_the_first_interval(self):
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler

        torch = self.torch
        scheduler = MiniMaxH3Scheduler(shift=12.0, solver="res_multistep")
        scheduler.set_timesteps(5, device="cpu")
        sample = torch.ones(1, dtype=torch.float32)
        first = scheduler.step(
            torch.full_like(sample, 0.25),
            scheduler.timesteps[0],
            sample,
        ).prev_sample
        second_coefficients = scheduler.coefficients_for_step(1)
        self.assertNotEqual(second_coefficients[2], 0.0)
        second = scheduler.step(
            torch.full_like(sample, -0.5),
            scheduler.timesteps[1],
            first,
        ).prev_sample
        self.assertTrue(torch.isfinite(second).all())


class TestFusedH3SLAFallback(unittest.TestCase):
    def test_policy_keeps_short_masked_and_trailing_steps_dense(self):
        from models.minimax_h3.sla_attention import MiniMaxH3SLAAttention

        policy = MiniMaxH3SLAAttention({
            "min_seq_len": 128,
            "dense_last_steps": 1,
        })
        policy.enabled = True
        policy.begin_step(0, 4)
        self.assertFalse(policy.use_for_layer(127))
        self.assertFalse(policy.use_for_layer(256, attention_mask=object()))
        self.assertTrue(policy.use_for_layer(256))
        policy.begin_step(3, 4)
        self.assertFalse(policy.use_for_layer(256))

    def test_sparse_runtime_failure_returns_dense_result(self):
        from models.minimax_h3.sla_attention import MiniMaxH3SLAAttention

        torch = __import__("torch")
        policy = MiniMaxH3SLAAttention()
        qkv = [torch.zeros(1, 2, 1, 2) for _ in range(3)]
        dense_result = torch.ones_like(qkv[0])
        with (
            patch(
                "models.minimax_h3.sla_block_map.get_block_map",
                side_effect=RuntimeError("synthetic SLA failure"),
            ),
            patch(
                "shared.attention.pay_attention",
                return_value=dense_result,
            ) as dense,
            patch(
                "shared.attention.get_default_attention_mode",
                return_value="sdpa",
            ),
        ):
            result = policy(qkv, True)

        self.assertIs(result, dense_result)
        self.assertTrue(policy._runtime_failed)
        dense.assert_called_once()


if __name__ == "__main__":
    unittest.main()
