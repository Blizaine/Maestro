"""Actual optimizer/RNG parity when the production AR trainer stops for previews.

Uses a tiny YuE2 model on CUDA, no downloads or user datasets/checkpoints.
"""
from contextlib import ExitStack
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file, load_file
from transformers import Qwen3Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import music_training as projects
from services.music_auditions import audition_options, train_with_auditions
from models.TTS.yue2 import artist_training as trainer


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA training resume check')
class TrainingResumeTests(unittest.TestCase):
    def test_preview_interleaving_preserves_weights_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(projects, 'PROJECT_ROOT', root / 'projects'))
            config = Qwen3Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, intermediate_size=64, vocab_size=184704, max_position_embeddings=64)
            torch.manual_seed(543)
            base = trainer.YuE2AR(config).bfloat16().eval()
            base_path = root / 'base.safetensors'
            save_file(base.state_dict(), str(base_path)); del base
            tracks = []
            for i in range(2):
                audio = root / f'song-{i}.wav'; audio.write_bytes(bytes([i]))
                tracks.append({'audio_path': str(audio), 'style': 'Pop', 'lyrics': 'New words', 'holdout': bool(i)})
            originals = [projects.create_project(name, 'Sample', tracks, pair='v4') for name in ('continuous', 'previewed')]
            for project in originals:
                projects.update_project(project['id'], prepared={'tokenizer_revision': trainer.TOKENIZER_REVISION})
            groups = {group: [{'prefix': [1, 2, 3, 4], 'codec': [1, 2, 3, 4, 5]}] for group in ('artist', 'heldout', 'minted', 'minted_val')}
            stack.enter_context(patch.object(trainer, 'ensure_asset', return_value=base_path))
            stack.enter_context(patch.object(trainer, 'YuE2TextTokenizer'))
            stack.enter_context(patch.object(trainer, '_dataset', return_value=groups))
            stack.enter_context(patch.object(trainer.Qwen3Config, 'from_json_file', return_value=config))
            stack.enter_context(patch.object(trainer, 'target_shapes', return_value=[('model.layers.0.self_attn.q_proj', 32, 32)]))
            options = {**projects.training_options({'steps': 2, 'rank': 4, 'seed': 10}), 'max_tokens': 32, 'checkpoint_every': 1}
            trainer.train_project(projects.get_project(originals[0]['id']), options, report=lambda *args: None, cancelled=lambda: False)
            calls = []
            def preview(project, branch, checkpoint, settings, **kwargs):
                calls.append(checkpoint)
                # Model initialization/sampling in an audition disturbs all the
                # generators training must restore from its durable snapshot.
                random.seed(919); random.random()
                torch.manual_seed(919); torch.randn(33); torch.randn(33, device='cuda')
            settings = audition_options({'enabled': True, 'style': 'Piano', 'lyrics': 'New sample words'})
            with patch('services.music_auditions.render_checkpoint', side_effect=preview):
                train_with_auditions(originals[1]['id'], {**options, 'audition': settings}, trainer.train_project,
                                     'style', report=lambda *args: None, cancelled=lambda: False)
            self.assertEqual(calls, ['step-1.safetensors', 'step-2.safetensors'])
            checkpoints = [projects.project_directory(project['id']) / 'checkpoints' for project in originals]
            left, right = [{key: value.clone() for key, value in load_file(str(path / 'step-2.safetensors')).items()} for path in checkpoints]
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
            before, after = [torch.load(path / 'resume.pt', map_location='cpu', weights_only=True) for path in checkpoints]
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
