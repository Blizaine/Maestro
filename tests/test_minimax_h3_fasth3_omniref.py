"""FastH3 OmniRef must use its own PDD integration and conditioning contract."""
import json
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.minimax_h3 import fasth3_omniref as omni
from models.minimax_h3.minimax_h3_handler import family_handler


class OmniRefTests(unittest.TestCase):
    def test_meta_construction_keeps_cpu_sampling_plans(self):
        from accelerate import init_empty_weights
        with init_empty_weights(include_buffers=True):
            layer = omni.PDDOutput(5, 7, 12.)
        self.assertEqual(layer.weight.device.type, 'meta')
        self.assertEqual(len(layer._plans), 8)
        self.assertEqual(layer._lock_dtype, torch.float32)

    def test_grid_is_checkpoint_specific_not_t2av_or_alibaba_grid(self):
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler
        for shift in (12., 3.):
            scheduler = MiniMaxH3Scheduler(shift=shift)
            scheduler.set_timesteps(sigmas=omni.sigmas(shift))
            u = torch.linspace(.999, 0, 33, dtype=torch.float64)
            expected = shift * u / (1 + (shift - 1) * u / .999)
            torch.testing.assert_close(scheduler.sigmas.double(), expected[::4], rtol=1e-6, atol=1e-7)
            self.assertEqual(scheduler.num_inference_steps, 8)
            self.assertAlmostEqual(float(scheduler.sigmas[0]), .999, places=6)
        self.assertFalse(torch.equal(omni.fine_sigmas(12), omni.fine_sigmas(3)))

    def test_fused_projection_equals_integrated_32_head_predictions(self):
        torch.manual_seed(13)
        x = torch.randn(2, 3, 5)
        for shift in (12., 3.):
            layer = omni.PDDOutput(5, 7, shift)
            raw_heads = F.linear(x, layer.weight, layer.bias).view(2, 3, 32, 7)
            # Independent dense oracle reconstructs each physical head
            # before integrating its prediction, rather than fusing params.
            all_heads = torch.cat((raw_heads[:, :, :1], raw_heads[:, :, 1:] + raw_heads[:, :, :1]), dim=2)
            fine = omni.fine_sigmas(shift)
            delta = fine[1:] - fine[:-1]
            for step in range(8):
                layer.step = step
                start = step * 4
                weights = delta[start:start+4].float()
                actual = layer(x) * weights.sum()
                expected = (all_heads[:, :, start:start+4] * weights[None, None, :, None]).sum(-2)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
            self.assertEqual(layer.out_features, 7)
            self.assertEqual(layer.weight.shape, (32 * 7, 5))
            layer.step = 8
            with self.assertRaises(ValueError):
                layer(x)

    def test_normalized_head_weights_match_comfy_unit_grid(self):
        # Scaling the base clock and sigma by .999 cancels when each
        # interval block is normalized; only node sigmas/noise scale differ.
        unit_grid = torch.linspace(1, 0, 33, dtype=torch.float64)
        for shift in (12., 3.):
            sigma = shift * unit_grid / (1 + (shift - 1) * unit_grid)
            intervals = (sigma[1:] - sigma[:-1]).view(8, 4)
            expected = (intervals / intervals.sum(-1, keepdim=True)).float()
            torch.testing.assert_close(omni.head_weights(shift), expected, rtol=0, atol=1e-7)

    def test_zero_residual_bank_preserves_complete_base_head_at_every_step(self):
        layer = omni.PDDOutput(5, 7, 12.)
        with torch.no_grad():
            layer.weight[7:] = 0
            layer.bias[7:] = 0
        rows = torch.randn(1, 3, 5)
        expected = F.linear(rows, layer.weight[:7], layer.bias[:7])
        for step in range(8):
            layer.step = step
            torch.testing.assert_close(layer(rows), expected)

    def test_definition_is_pinned_and_exposes_only_single_window_references(self):
        definition = json.loads((ROOT / 'app/defaults/minimax_h3_fasth3_omniref.json').read_text())
        model = definition['model']
        options = family_handler.query_model_def('minimax_h3_ref2va', model)
        self.assertTrue(options['omni_reference'])
        self.assertTrue(options['supports_reference_audio'])
        self.assertTrue(options['returns_audio'])
        for key in ('sliding_window', 'video_continuation', 'i2v_class', 'sol_attention', 'sla_attention', 'veda_attention', 'first_block_cache'):
            self.assertFalse(options[key], key)
        self.assertTrue(options['loras_disabled'])
        self.assertEqual(model['source_sha256'], omni.SHA256)
        self.assertEqual(model['source_size_bytes'], omni.SIZE)
        self.assertIn(omni.REVISION, model['URLs'][0])
        self.assertEqual(options['minimax_h3_video_shift'], 12)
        with self.assertRaises(ValueError):
            family_handler.query_model_def('minimax_h3', model)

    def test_request_keeps_ordered_refs_but_locks_recipe_and_rejects_unsupported_routes(self):
        refs = [{'type': 'image', 'path': 'character.png'}, {'type': 'video', 'path': 'motion.mp4'},
                {'type': 'audio', 'path': 'voice.wav'}]
        request = {'minimax_h3_references': refs, 'num_inference_steps': 8,
                   'video_length': 124, 'override_attention': 'sla', 'flow_shift': 10}
        omni.validate_request(request)
        self.assertEqual([r['path'] for r in request['minimax_h3_references']], [r['path'] for r in refs])
        self.assertEqual(request['flow_shift'], 12)
        self.assertEqual(request['override_attention'], 'sdpa')
        for update in ({'num_inference_steps': 4}, {'image_start': 'start.png'},
                       {'minimax_h3_multi_window': True}, {'input_masks': object()},
                       {'activated_loras': ['turbo']}, {'minimax_h3_references': []},
                       {'minimax_h3_references': [refs[0], {**refs[2], 'audio_intent': 'drive'}]}):
            with self.subTest(update=update), self.assertRaises((TypeError, ValueError)):
                omni.validate_request({**request, **update})


if __name__ == '__main__':
    unittest.main()
