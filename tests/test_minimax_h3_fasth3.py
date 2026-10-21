"""FastH3's fixed trained recipe must not inherit ordinary FL2VA controls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.minimax_h3 import fasth3
from models.minimax_h3.minimax_h3_handler import family_handler


class FastH3RecipeTests(unittest.TestCase):
    def test_fixed_rungs_shift_once_and_drive_eight_euler_forwards(self):
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler
        for shift in (10.0, 3.0):
            scheduler = MiniMaxH3Scheduler(shift=shift)
            scheduler.set_timesteps(sigmas=fasth3.fasth3_sigmas(shift))
            expected = [shift * r / (1000 + (shift - 1) * r) for r in fasth3.FASTH3_RUNGS] + [0]
            self.assertEqual(scheduler.num_inference_steps, 8)
            for actual, target in zip(scheduler.sigmas.tolist(), expected):
                self.assertAlmostEqual(actual, target, places=6)
            self.assertLess(scheduler.sigmas[0].item(), 1)
            self.assertEqual(scheduler.timesteps.numel(), 8)
            self.assertEqual(scheduler.solver, "euler")

    def test_definition_is_pinned_and_not_a_reference_model(self):
        source = json.loads((ROOT / "app/defaults/minimax_h3_fasth3.json").read_text())
        model = source["model"]
        result = family_handler.query_model_def("minimax_h3", model)
        self.assertEqual(model["source_size_bytes"], fasth3.FASTH3_SIZE)
        self.assertEqual(model["source_sha256"], fasth3.FASTH3_SHA256)
        self.assertIn(fasth3.FASTH3_REVISION, model["URLs"][0])
        for capability in ("i2v_class", "omni_reference", "sliding_window", "sol_attention",
                           "sla_attention", "veda_attention", "first_block_cache", "custom_frames_injection"):
            self.assertFalse(result[capability], capability)
        self.assertEqual(result["image_prompt_types_allowed"], "")
        self.assertTrue(result["returns_audio"])
        self.assertTrue(result["loras_disabled"])
        self.assertEqual(result["compatible_model_paths"], {})
        self.assertEqual(result["inference_steps_min"], result["inference_steps_max"])
        self.assertEqual(result["minimax_h3_qkv_layout"], "grouped")
        with self.assertRaisesRegex(ValueError, "separate"):
            family_handler.query_model_def("minimax_h3_ref2va", model)
        ordinary = family_handler.query_model_def("minimax_h3", {})
        self.assertTrue(ordinary["i2v_class"])
        self.assertTrue(ordinary["sliding_window"])
        self.assertFalse(ordinary.get("minimax_h3_fasth3"))

    def test_unverified_model_does_not_inherit_benchmarked_duration_tiers(self):
        from services.h3_duration_recommendations import h3_duration_recommendations_enabled
        from models.minimax_h3.duration import h3_duration_model_def
        model = {"architecture": "minimax_h3", "minimax_h3_fasth3": True, "frames_maximum": 345}
        self.assertFalse(h3_duration_recommendations_enabled(model))
        self.assertIs(h3_duration_model_def(model, {"minimax_h3_extended_duration": True}), model)

    def test_rejects_media_loras_windows_and_wrong_schedule(self):
        cases = [
            {"image_start": "start.png"}, {"image_refs": ["ref.png"]},
            {"minimax_h3_references": [{"path": "ref.png"}]},
            {"audio_guide": "song.wav"}, {"input_video": object()},
            {"input_waveform": object()}, {"video_length": 362},
            {"window_start_frame_no": 106}, {"activated_loras": ["turbo"]},
            {"num_inference_steps": 9}, {"num_inference_steps": True},
            {"denoising_strength": .8}, {"video_prompt_type": "GV"},
            {"custom_settings": {"audio_refinement": "enabled"}},
            {"multi_prompts_gen_type": 3},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises((ValueError, TypeError)):
                fasth3.validate_fasth3_request(case)

    def test_handler_enforces_fixed_settings_and_rejects_reference(self):
        inputs = {"video_length": 243, "sliding_window_size": 124, "num_inference_steps": 8,
                  "flow_shift": 12, "override_attention": "sla", "skip_steps_cache_type": "first_block"}
        error = family_handler.validate_generative_settings("minimax_h3", {"minimax_h3_fasth3": True}, inputs)
        self.assertIsNone(error)
        self.assertEqual(inputs["flow_shift"], 10)
        self.assertEqual(inputs["skip_steps_cache_type"], "")
        self.assertEqual(inputs["override_attention"], "sdpa")
        self.assertEqual(inputs["video_length"], 243)
        self.assertEqual(inputs["sliding_window_size"], 243)
        self.assertEqual(inputs["multi_prompts_gen_type"], 2)
        self.assertIsNotNone(family_handler.validate_generative_settings(
            "minimax_h3", {"minimax_h3_fasth3": True}, {"image_refs": ["ref.png"]}))

    def test_checkpoint_digest_is_checked_and_cache_tracks_file_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            data = b"pinned checkpoint test"
            path.write_bytes(data)
            with patch.object(fasth3, "FASTH3_SIZE", len(data)), patch.object(
                fasth3, "FASTH3_SHA256", hashlib.sha256(data).hexdigest()
            ):
                fasth3.verify_fasth3_checkpoint(str(path))
                fasth3.verify_fasth3_checkpoint(str(path))
                path.write_bytes(b"x" * len(data))
                with self.assertRaisesRegex(ValueError, "checksum"):
                    fasth3.verify_fasth3_checkpoint(str(path))

    def test_ordinary_loader_refuses_trained_gate_weights(self):
        from models.minimax_h3 import minimax_h3_main as runtime
        with patch.object(runtime, "probe_h3_checkpoint", return_value={}), patch(
            "models.minimax_h3.turbo.safetensors_header",
            return_value={"blocks.0.attn.to_gate_compress.weight": {"shape": [7168, 5376]}},
        ), patch.object(runtime.offload, "load_model_data") as load:
            with self.assertRaisesRegex(ValueError, "dedicated FastH3"):
                runtime._load_transformer("unknown.safetensors", runtime.torch.bfloat16)
            load.assert_not_called()

    def test_transformer_hands_raw_gate_to_dedicated_attention(self):
        import torch
        from models.minimax_h3.transformer import MiniMaxH3Attention, _split_fused_qkv
        module = MiniMaxH3Attention(8, 1, 128, 1e-5, torch.float32)
        module.to_gate_compress = torch.nn.Linear(8, 128, bias=False)
        with torch.no_grad():
            module.to_gate_compress.weight.fill_(2.0)
        observed = []
        def vsa(query, key, value, gate):
            observed.append((query.clone(), key.clone(), value.clone(), gate.clone()))
            return value
        module.fasth3_attention = vsa
        hidden = torch.ones(1, 3, 8)
        with torch.inference_mode():
            result = module([hidden])
            q, k, v = _split_fused_qkv(module.qkv_proj(hidden), 1, 128, "grouped")
            self.assertTrue(torch.allclose(observed[0][0], module.q_norm(q)))
            self.assertTrue(torch.allclose(observed[0][1], module.k_norm(k)))
            self.assertTrue(torch.equal(observed[0][3], torch.full((1, 3, 1, 128), 16.0)))
            self.assertTrue(torch.allclose(result, module.out_proj(v.reshape(1, 3, 128))))
            # INT8 ConvRot loading splits Q/K/V into independently streamed
            # projections. Exercise that path across more than one row chunk.
            for name, weight in zip(("q_proj", "k_proj", "v_proj"), module.qkv_proj.weight.chunk(3, dim=0)):
                linear = torch.nn.Linear(8, 128, bias=False)
                linear.weight.copy_(weight)
                setattr(module, name, linear)
            del module.qkv_proj
            observed.clear()
            with patch("models.minimax_h3.transformer.MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 2):
                split_result = module([hidden])
            self.assertTrue(torch.allclose(split_result, result))
            self.assertTrue(torch.equal(observed[0][3], torch.full((1, 3, 1, 128), 16.0)))


if __name__ == "__main__":
    unittest.main()
