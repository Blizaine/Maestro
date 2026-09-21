"""Saved joint starts preserve split adapters, decoder I/O and inference mode."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from models.TTS.yue2 import joint_training as trainer
from models.TTS.yue2.training_audition import checkpoint_adapter_mode
from services.music_contracts import V9_REVISION


class JointInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.targets = [(f'attention.{name}_proj', 4, 4) for name in ('q', 'k', 'v')]
        self.model = nn.Module()
        self.model.attention = nn.Module()
        for name, _, _ in self.targets:
            setattr(self.model.attention, name.split('.')[-1], nn.Linear(4, 4, bias=False))
        self.model.vae2llm = nn.Linear(2, 4)
        self.model.llm2vae = nn.Linear(4, 2)
        self.model.requires_grad_(False)
        self.weights = {name + suffix: torch.randn(shape) for name, _, _ in self.targets
                        for suffix, shape in (('.A', (2, 4)), ('.B', (4, 2)))}

    def test_start_preserves_each_projection_and_its_forward_math(self):
        values = torch.randn(3, 4)
        expected = {name: self.model.get_submodule(name)(values)
                    + F.linear(F.linear(values, self.weights[name + '.A']), self.weights[name + '.B'])
                    for name, _, _ in self.targets}
        with patch.object(trainer, 'target_shapes', return_value=self.targets):
            adapters = trainer.install_adapters(self.model, 'ar', 2, self.weights, shared_fused_a=False)
        self.assertEqual(len({adapter.A.data_ptr() for adapter in adapters.values()}), 3)
        for name, adapter in adapters.items():
            torch.testing.assert_close(adapter(values), expected[name], rtol=0, atol=0)
            torch.testing.assert_close(adapter.A, self.weights[name + '.A'], rtol=0, atol=0)
        sum(adapter(values).square().mean() for adapter in adapters.values()).backward()
        self.assertTrue(all(adapter.A.grad is not None and adapter.B.grad is not None for adapter in adapters.values()))
        self.assertTrue(all(adapter.base.weight.grad is None for adapter in adapters.values()))

    def test_legacy_sharing_refuses_to_overwrite_unequal_saved_matrices(self):
        with patch.object(trainer, 'target_shapes', return_value=self.targets), self.assertRaisesRegex(ValueError, 'unequal'):
            trainer.install_adapters(self.model, 'ar', 2, self.weights)

    def test_decoder_io_and_independent_adapters_survive_a_snapshot(self):
        weights = dict(self.weights)
        for name in ('vae2llm', 'llm2vae'):
            weights.update({name + '.' + key: torch.randn_like(value)
                            for key, value in getattr(self.model, name).state_dict().items()})
        with patch.object(trainer, 'target_shapes', return_value=self.targets):
            adapters = trainer.install_adapters(self.model, 'nar', 2, weights, shared_fused_a=False)
        trainer.restore_decoder_io(self.model, weights)
        restored = trainer.adapter_snapshot(self.model, adapters, include_io=True)
        self.assertEqual(set(restored), set(weights))
        for key in weights:
            torch.testing.assert_close(restored[key], weights[key], rtol=0, atol=0)
        self.assertFalse(self.model.vae2llm.weight.requires_grad)
        self.assertFalse(self.model.llm2vae.weight.requires_grad)
        self.model.vae2llm.weight.data.zero_()
        self.assertFalse(torch.equal(restored['vae2llm.weight'], self.model.vae2llm.weight))

    def test_style_identity_requires_matching_tokens_and_trigger(self):
        project = {'trigger': 'My voice', 'tokenizer_pair': 'v9'}
        manifest = {'id': 'source', 'name': 'Source', 'trigger': 'My voice', 'version': 2,
                    'adapter_mode': 'separate', 'tokenizer_revision': V9_REVISION,
                    'ar': {'file': 'ar.safetensors', 'sha256': 'ar-hash'},
                    'nar': {'file': 'nar.safetensors', 'sha256': 'nar-hash'}}
        with patch.object(trainer, 'load_style', return_value=manifest) as loader, \
                patch.object(trainer, 'style_directory', return_value=Path('source')), \
                patch.object(trainer, 'read_upstream_adapter', return_value=self.weights) as reader:
            identity, _ = trainer.initial_style_weights(project, 'source')
            loader.assert_called_once_with('source', verify=True)
            self.assertEqual(identity['ar_rank'], 2)
            self.assertEqual(identity['nar_rank'], 2)
            self.assertEqual(identity['ar_sha256'], 'ar-hash')
            self.assertTrue(all(call.kwargs['require_io'] for call in reader.call_args_list))
            for changed in ({'tokenizer_pair': 'v4'}, {'trigger': 'Another voice'}):
                with self.assertRaises(ValueError):
                    trainer.initial_style_weights({**project, **changed}, 'source')

    def test_export_and_preview_keep_the_checkpoint_decoder_mode(self):
        self.assertEqual(checkpoint_adapter_mode('joint', {}), 'joint')
        self.assertEqual(checkpoint_adapter_mode('joint', {'adapter_mode': 'separate'}), 'separate')
        self.assertEqual(checkpoint_adapter_mode('audio', {}), 'separate')
        with self.assertRaises(ValueError):
            checkpoint_adapter_mode('joint', {'adapter_mode': 'invalid'})


if __name__ == '__main__':
    unittest.main()
