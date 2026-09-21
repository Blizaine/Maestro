"""Tiny real CUDA training: joint checkpoint pauses preserve optimizer and RNG."""
from contextlib import ExitStack
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file, load_file
from transformers import Qwen3Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import music_training as projects
from services.music_auditions import audition_options, train_with_auditions
from models.TTS.yue2 import joint_training as trainer


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA joint resume check')
class JointResumeTests(unittest.TestCase):
    def test_joint_preview_resume_matches_uninterrupted_training(self):
        self._compare_resume(False)

    def test_saved_style_preview_resume_keeps_independent_adapters_and_decoder(self):
        self._compare_resume(True)

    def _compare_resume(self, warm_start):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(projects, 'PROJECT_ROOT', root / 'projects'))
            config = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                          head_dim=16, intermediate_size=64, vocab_size=184704, max_position_embeddings=256)
            ar_config, nar_config = Qwen3Config(**config), trainer.YuE2Config(**config, latent_dim=64, max_latent_frames=256)
            torch.manual_seed(432)
            for branch, model in (('ar', trainer.YuE2AR(ar_config)), ('nar', trainer.YuE2Acoustic(nar_config))):
                model = model.bfloat16().eval()
                if branch == 'ar':
                    with torch.no_grad():
                        model.model.embed_tokens.weight.normal_(std=.05)
                        model.lm_head.weight.normal_(std=.05)
                save_file(model.state_dict(), str(root / (branch + '_training.safetensors')))
                del model
            tracks = []
            for i in range(2):
                audio = root / f'song-{i}.wav'; audio.write_bytes(bytes([i]))
                tracks.append({'audio_path': str(audio), 'style': 'Pop', 'lyrics': 'New words', 'holdout': bool(i)})
            originals = [projects.create_project(name, 'Sample', tracks, pair='v4') for name in ('continuous', 'previewed')]
            groups = {name: [{'prefix': [1,2,3,4], 'codec': list(range(12)), 'latent': np.zeros((12, 64), np.float32)}]
                      for name in ('artist', 'heldout', 'minted', 'minted_val')}
            stack.enter_context(patch.object(trainer, 'ensure_asset', side_effect=lambda name, **kwargs: root / (name + '.safetensors')))
            stack.enter_context(patch.object(trainer, 'YuE2TextTokenizer'))
            stack.enter_context(patch.object(trainer, 'load_data', return_value=(groups, 'test-regularizer')))
            stack.enter_context(patch.object(trainer.Qwen3Config, 'from_json_file', return_value=ar_config))
            stack.enter_context(patch.object(trainer, 'YuE2Config', return_value=nar_config))
            def targets(branch):
                group = 'self_attn' if branch == 'ar' else 'nar_self_attn'
                return [(f'model.layers.0.{group}.{name}', 32, size) for name, size in (('q_proj', 32), ('k_proj', 16), ('v_proj', 16))]
            stack.enter_context(patch.object(trainer, 'target_shapes', side_effect=targets))
            initial = None
            initial_weights = None
            if warm_start:
                initial = {'style_id': 'saved-e', 'style_name': 'Saved E', 'adapter_mode': 'separate',
                           'ar_sha256': 'original-ar', 'nar_sha256': 'original-nar', 'ar_rank': 4, 'nar_rank': 4}
                initial_weights = {branch: {name + suffix: torch.randn(shape) * .01
                    for name, inputs, outputs in targets(branch)
                    for suffix, shape in (('.A', (4, inputs)), ('.B', (outputs, 4)))} for branch in ('ar', 'nar')}
                initial_weights['nar'].update({key: torch.randn_like(value.float()) * .01 for key, value in
                    load_file(str(root / 'nar_training.safetensors')).items() if key.startswith(('vae2llm.', 'llm2vae.'))})
                stack.enter_context(patch.object(trainer, 'initial_style_weights', return_value=(initial, initial_weights)))
            options = {**projects.joint_training_options({'steps': 2, 'rank': 4, 'seed': 10}), 'max_tokens': 128, 'checkpoint_every': 1}
            if warm_start:
                options['initial_style_id'] = 'saved-e'
            trainer.train_joint(originals[0], options, report=lambda *args: None, cancelled=lambda: False)
            calls = []
            def preview(project, branch, checkpoint, settings, **kwargs):
                calls.append(checkpoint)
                random.seed(919); torch.manual_seed(919); torch.randn(33); torch.randn(33, device='cuda')
            settings = audition_options({'enabled': True, 'style': 'Piano', 'lyrics': 'New words'})
            with patch('services.music_auditions.render_checkpoint', side_effect=preview):
                train_with_auditions(originals[1]['id'], {**options, 'audition': settings}, trainer.train_joint,
                                     'joint', report=lambda *args: None, cancelled=lambda: False)
            self.assertEqual(calls, ['ar-step-1.safetensors', 'ar-step-2.safetensors'])
            directories = [projects.project_directory(p['id']) / 'joint_checkpoints' for p in originals]
            if warm_start:
                for branch in ('ar', 'nar'):
                    baseline = {key: value.clone() for key, value in load_file(str(directories[0] / f'{branch}-step-0.safetensors')).items()}
                    for key, value in initial_weights[branch].items():
                        expected = value.bfloat16() if key.startswith(('vae2llm.', 'llm2vae.')) else value
                        torch.testing.assert_close(baseline[key], expected, rtol=0, atol=0)
                project = projects.get_project(originals[0]['id'])
                self.assertEqual(project['joint_checkpoints'][0]['adapter_mode'], 'separate')
                self.assertFalse(project['joint_training_options']['shared_fused_a'])
                with self.assertRaisesRegex(ValueError, 'starting style'):
                    trainer.train_joint(project, {**options, 'resume': True, 'initial_style_id': 'different'},
                                        report=lambda *args: None, cancelled=lambda: False)
            for branch in ('ar', 'nar'):
                left, right = [{k: v.clone() for k, v in load_file(str(d / f'{branch}-step-2.safetensors')).items()} for d in directories]
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            before, after = [torch.load(d / 'resume.pt', map_location='cpu', weights_only=True) for d in directories]
            self.assertEqual(before['python_rng'], after['python_rng'])
            torch.testing.assert_close(before['torch_rng'], after['torch_rng'], rtol=0, atol=0)
            for a, b in zip(before['cuda_rng'], after['cuda_rng']):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(before['optimizer']['param_groups'], after['optimizer']['param_groups'])
            for key, state in before['optimizer']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, after['optimizer']['state'][key][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
