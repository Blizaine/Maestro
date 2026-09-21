"""Matched-pair publication, frozen-model gradients, and waveform supervision."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import numpy as np
import torch
from services import music_training as projects
from services.music_contracts import tokenizer_pair, pair_asset
from services.music_pair_adaptation import select_pair, adaptation_options, checkpoint_pair, comparison_options
from services.music_styles import file_digest
from models.TTS.yue2.pair_training_math import straight_through_embeddings, soft_token_loss, SpectralLoss, waveform_loss


class PairContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(projects, 'PROJECT_ROOT', self.root / 'projects').start()
        tracks = []
        for n in range(2):
            path = self.root / f'{n}.wav'; path.write_bytes(f'original-{n}'.encode())
            tracks.append({'audio_path': str(path), 'lyrics': 'Original words', 'style': 'Rap', 'holdout': bool(n)})
        self.project = projects.create_project('Pair test', 'My voice', tracks, pair='v9')
        directory = projects.project_directory(self.project['id']) / 'pair_checkpoints'; directory.mkdir()
        row = {'step': 25, 'file': 'head-25.safetensors', 'nar_file': 'nar-25.safetensors'}
        for branch in ('head', 'nar'):
            path = directory / row['file' if branch == 'head' else 'nar_file']; path.write_bytes(branch.encode())
            row[branch + '_sha256'] = file_digest(path)
        self.project = projects.update_project(self.project['id'], pair_checkpoints=[row], prepared={'old': True},
            pair_prepared={'dataset_digest': self.project['dataset_digest']})
        self.row = row

    def test_selection_keeps_source_immutable_and_forces_new_tokens(self):
        before = copy.deepcopy(projects.get_project(self.project['id']))
        result = select_pair(self.project['id'], self.row['file'])
        self.assertNotEqual(result['id'], before['id'])
        self.assertEqual(result['tracks'], before['tracks'])
        self.assertEqual(result['dataset_digest'], before['dataset_digest'])
        self.assertNotIn('prepared', result)
        self.assertNotIn('pair_checkpoints', result)
        self.assertNotEqual(tokenizer_pair(before)['revision'], tokenizer_pair(result)['revision'])
        for branch in ('head', 'nar'):
            self.assertEqual(file_digest(pair_asset(result, branch)), self.row[branch + '_sha256'])
        self.assertEqual(projects.get_project(before['id']), before)
        fork = projects.fork_project(result['id'])
        self.assertEqual(tokenizer_pair(result), tokenizer_pair(fork))
        self.assertEqual(pair_asset(fork, 'head').read_bytes(), b'head')

    def test_corrupt_and_unregistered_pair_cannot_be_selected(self):
        with self.assertRaises(ValueError):
            select_pair(self.project['id'], '../head-25.safetensors')
        row, paths = checkpoint_pair(self.project, self.row['file'])
        paths['head'].write_bytes(b'changed')
        with self.assertRaises(ValueError):
            select_pair(self.project['id'], row['file'])

    def test_diagnostic_uses_check_first_and_freezes_both_hashes(self):
        options = comparison_options({'checkpoint': self.row['file']}, self.project)
        self.assertEqual(options['track_ids'][0], self.project['tracks'][1]['id'])
        self.assertEqual(options['head_sha256'], self.row['head_sha256'])
        self.assertEqual(options['nar_sha256'], self.row['nar_sha256'])
        self.assertEqual(adaptation_options({})['steps'], 100)
        self.assertEqual(adaptation_options({'steps': 2})['audio_weight'], 4)


class PairMathTests(unittest.TestCase):
    def test_hard_forward_retains_soft_gradient_into_head(self):
        logits = torch.randn(8, 12, requires_grad=True)
        table = torch.randn(12, 6)
        value, indices = straight_through_embeddings(logits, table)
        torch.testing.assert_close(value, table[indices])
        value.square().mean().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_soft_neighbor_targets_match_manual_loss(self):
        logits = torch.randn(2, 3, 5, requires_grad=True)
        targets = torch.tensor([[1, 2, -100], [3, 2, 1]])
        neighbors = torch.tensor([[i, (i + 1) % 5] for i in range(5)])
        weights = torch.tensor([[.7, .3]] * 5)
        loss = soft_token_loss(logits, targets, neighbors, weights)
        logp = logits.log_softmax(-1); expected = []
        for i in range(2):
            for j in range(3):
                y = int(targets[i, j])
                if y >= 0:
                    expected.append(-.75 * logp[i, j, y] - .25 * (.7 * logp[i, j, y] + .3 * logp[i, j, (y + 1) % 5]))
        torch.testing.assert_close(loss, torch.stack(expected).mean())
        loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())

    def test_stereo_loss_zero_for_identical_and_differentiable_for_difference(self):
        spectral = SpectralLoss()
        ref = torch.randn(2, 8000) * .1
        self.assertEqual(float(spectral(ref, ref)), 0)
        pred = (ref * torch.tensor([[.6], [.8]])).requires_grad_()
        spectral(pred, ref).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_waveform_loss_uses_correct_offset_and_frozen_decoder_input_gradient(self):
        class Decoder(torch.nn.Module):
            def forward(self, x):
                return x[:, :2].repeat_interleave(1920, -1)
        prediction = torch.randn(100, 2, requires_grad=True)
        mixed = torch.randn(100, 2)
        audio = np.zeros((130 * 1920, 2), dtype=np.float32)
        seen = {}
        def objective(wav, ref):
            seen['shape'] = ref.shape
            return (wav - ref).square().mean()
        loss = waveform_loss(Decoder(), objective, mixed, prediction, .2, audio, 10, frames=25, margin=10)
        loss.backward()
        self.assertEqual(seen['shape'], (2, 25 * 1920))
        self.assertGreater(float(prediction.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            waveform_loss(Decoder(), objective, mixed, prediction, .2, audio[:100], 10)


if __name__ == '__main__':
    unittest.main()
