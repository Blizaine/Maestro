"""Projection/scale parity and backward-compatible style bundle contracts."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from models.TTS.yue2.combined_adapter import convert_combined
from models.TTS.yue2.artist_adapter import validate_tensors, active_adapters
from services.music_styles import save_style, load_style


def fixture():
    generator = torch.Generator().manual_seed(83)
    weights = {}
    for branch in ('text_encoders', 'diffusion_model'):
        for layer in range(28):
            for name, inputs, outputs in (('self_attn.qkv_proj', 2048, 4096), ('self_attn.o_proj', 2048, 2048),
                                           ('mlp.gate_up_proj', 2048, 12288), ('mlp.down_proj', 6144, 2048)):
                target = f'{branch}.model.layers.{layer}.{name}'
                weights[target + '.lora_A.weight'] = torch.randn(1, inputs, generator=generator)
                weights[target + '.lora_B.weight'] = torch.randn(outputs, 1, generator=generator)
    return weights


class CombinedAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.weights = fixture()

    def test_split_projection_matches_fused_delta_and_alpha(self):
        weights = dict(self.weights)
        weights['text_encoders.model.layers.0.self_attn.qkv_proj.alpha'] = torch.tensor(.5)
        result = convert_combined(weights)
        for branch, source in (('ar', 'text_encoders'), ('nar', 'diffusion_model')):
            group = 'self_attn' if branch == 'ar' else 'nar_self_attn'
            x = torch.randn(2, 2048)
            name = f'{source}.model.layers.0.self_attn.qkv_proj'
            expected = F.linear(F.linear(x, weights[name + '.lora_A.weight']), weights[name + '.lora_B.weight'])
            if branch == 'ar':
                expected *= .5
            actual = torch.cat([F.linear(F.linear(x, result[branch][f'model.layers.0.{group}.{part}.A']),
                                        result[branch][f'model.layers.0.{group}.{part}.B']) for part in ('q_proj', 'k_proj', 'v_proj')], -1)
            torch.testing.assert_close(actual, expected)
            mlp = 'mlp' if branch == 'ar' else 'nar_mlp'
            name = f'{source}.model.layers.0.mlp.gate_up_proj'
            expected = F.linear(F.linear(x, weights[name + '.lora_A.weight']), weights[name + '.lora_B.weight'])
            actual = torch.cat([F.linear(F.linear(x, result[branch][f'model.layers.0.{mlp}.{part}.A']),
                                        result[branch][f'model.layers.0.{mlp}.{part}.B']) for part in ('gate_proj', 'up_proj')], -1)
            torch.testing.assert_close(actual, expected)
        with self.assertRaisesRegex(ValueError, 'vae2llm'):
            validate_tensors(result['nar'], 'nar')  # Old imports still require full decoder I/O.

    def test_rejects_missing_extra_and_nonfinite_weights(self):
        missing = dict(self.weights)
        missing.pop(next(iter(missing)))
        for weights in (missing, {**self.weights, 'unexpected': torch.zeros(1)},
                        {**self.weights, 'text_encoders.model.layers.0.self_attn.qkv_proj.alpha': torch.tensor(float('nan'))}):
            with self.subTest(case=len(weights)), self.assertRaises(ValueError):
                convert_combined(weights)

    def test_bundle_roundtrip_preserves_joint_semantics(self):
        pair = convert_combined(self.weights)
        with tempfile.TemporaryDirectory() as directory:
            saved = save_style('Example', 'trigger', pair['ar'], pair['nar'], root=directory,
                               adapter_mode='joint', unknown_tokenizer=True)
            loaded = load_style(saved['id'], root=directory, verify=True)
            self.assertEqual(loaded['adapter_mode'], 'joint')
            self.assertIsNone(loaded['tokenizer_revision'])
            validate_tensors(pair['nar'], 'nar', require_io=False)

    def test_combined_import_export_does_not_add_a_generic_decoder(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from safetensors.torch import save
        from services.music_training_api import create_router
        with tempfile.TemporaryDirectory() as directory, patch('services.music_styles.STYLE_ROOT', Path(directory)), \
                patch('models.TTS.yue2.music_assets.ensure_asset', side_effect=AssertionError('Unexpected companion download')):
            app = FastAPI(); app.include_router(create_router(lambda *args: None, {}))
            client = TestClient(app)
            result = client.post('/api/v1/music-styles/import', data={'name': 'Joint example', 'trigger': 'Example'},
                                 files={'ar': ('combined.safetensors', save(self.weights))})
            self.assertEqual(result.status_code, 200, result.text)
            first = result.json()
            self.assertEqual(first['adapter_mode'], 'joint')
            bundle = client.get('/api/v1/music-styles/' + first['id'] + '/export')
            self.assertEqual(bundle.status_code, 200)
            again = client.post('/api/v1/music-styles/import', files={'ar': ('style.zip', bundle.content)})
            self.assertEqual(again.status_code, 200, again.text)
            second = again.json()
            self.assertEqual(second['adapter_mode'], 'joint')
            self.assertIsNone(second['tokenizer_revision'])
            self.assertEqual(second['nar']['sha256'], first['nar']['sha256'])
            self.assertEqual(second['ar']['sha256'], first['ar']['sha256'])

    def test_joint_strength_scales_both_branches_and_hooks_are_removed(self):
        from types import SimpleNamespace
        ar, nar = torch.nn.Module(), torch.nn.Module()
        ar.linear = torch.nn.Linear(2, 2, bias=False).bfloat16()
        nar.linear = torch.nn.Linear(2, 2, bias=False).bfloat16()
        pipeline = SimpleNamespace(text_encoder=ar, transformer=nar,
                                   engine=SimpleNamespace(release_runtime_allocations=lambda: None))
        weights = {'linear.A': torch.ones(1, 2), 'linear.B': torch.ones(2, 1)}
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return original_to(tensor, *args, **kwargs)
        x = torch.ones(1, 2, dtype=torch.bfloat16)
        baseline = [ar.linear(x).detach(), nar.linear(x).detach()]
        with ExitStack() as stack:
            stack.enter_context(patch('models.TTS.yue2.artist_adapter.read_upstream_adapter', return_value=weights))
            stack.enter_context(patch('models.TTS.yue2.artist_adapter.target_shapes', return_value=[('linear', 2, 2)]))
            stack.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
            for strength in (0, .5, 1):
                with active_adapters(pipeline, {'ar': 'ar', 'nar': 'nar'}, strength, mode='joint'):
                    for module, base in zip((ar.linear, nar.linear), baseline):
                        torch.testing.assert_close(module(x), base + 2 * strength)
                for module, base in zip((ar.linear, nar.linear), baseline):
                    torch.testing.assert_close(module(x), base)
                    self.assertFalse(module._forward_hooks)


if __name__ == '__main__':
    unittest.main()
