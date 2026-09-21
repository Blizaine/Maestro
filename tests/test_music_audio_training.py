"""Alignment masking, immutable experiments and strict acoustic contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import music_training
from models.TTS.yue2.lyric_alignment import lyric_words, validate_words, cursor_targets, cursor_loss, identity
from models.TTS.yue2.tokenization_yue2 import YuE2TextTokenizer
from models.TTS.yue2.protocol import SongRequest, token_prefixes


class AudioTrainingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        patcher = patch.object(music_training, 'PROJECT_ROOT', self.root / 'projects')
        patcher.start(); self.addCleanup(patcher.stop)
        tracks = []
        for i in range(2):
            path = self.root / f'{i}.wav'; path.write_bytes(bytes([i]) * 200)
            tracks.append({'audio_path': str(path), 'lyrics': '[Verse]\nHello hello, café!\n[Chorus]\nDon’t wait.',
                           'style': 'Acoustic', 'holdout': bool(i)})
        self.project = music_training.create_project('Source', 'My sound', tracks, pair='v4')

    def test_new_experiment_preserves_checkpoint_and_resume(self):
        project = self.project
        directory = music_training.project_directory(project['id'])
        (directory / 'prepared').mkdir()
        (directory / 'prepared' / 'x').write_bytes(b'cache')
        music_training.update_project(project['id'], prepared={'version': 1}, resume_available=True, completed_steps=800)
        original = (directory / 'project.json').read_bytes()
        copied = music_training.fork_project(project['id'])
        self.assertNotEqual(copied['id'], project['id'])
        self.assertEqual(copied['dataset_digest'], project['dataset_digest'])
        self.assertFalse(copied.get('resume_available'))
        self.assertEqual((directory / 'project.json').read_bytes(), original)
        self.assertEqual((music_training.project_directory(copied['id']) / 'prepared' / 'x').read_bytes(), b'cache')

    def test_audio_options_are_bounded_and_legacy_alignment_stays_opt_in(self):
        self.assertFalse(music_training.training_options({})['lyric_alignment'])
        self.assertEqual(music_training.audio_training_options({})['rank'], 32)
        for raw in ({'steps': 2000}, {'steps': True}, {'learning_rate': float('nan')}, {'conditioning_checkpoint': '../x'}):
            with self.assertRaises(ValueError):
                music_training.audio_training_options(raw)

    def test_v9_fork_requires_fresh_tokens_and_keeps_original_untouched(self):
        directory = music_training.project_directory(self.project['id'])
        (directory / 'prepared').mkdir()
        (directory / 'prepared' / 'marker').write_bytes(b'v4')
        music_training.update_project(self.project['id'], prepared={'tokenizer_revision': 'v4'},
                                     resume_available=True, completed_steps=800)
        before = (directory / 'project.json').read_bytes()
        forked = music_training.fork_project(self.project['id'], pair='v9')
        self.assertEqual(forked['tokenizer_pair'], 'v9')
        self.assertFalse(forked.get('prepared'))
        self.assertFalse(forked.get('resume_available'))
        self.assertFalse((music_training.project_directory(forked['id']) / 'prepared').exists())
        self.assertEqual((directory / 'project.json').read_bytes(), before)
        with self.assertRaises(ValueError):
            music_training.fork_project(self.project['id'], pair='unknown')

    def test_joint_options_keep_separate_training_unchanged(self):
        joint = music_training.joint_training_options({})
        self.assertEqual(joint['window_frames'], 1500)
        self.assertEqual(joint['rank'], 32)
        self.assertEqual(joint['accumulation_steps'], 1)
        self.assertEqual(music_training.training_options({})['accumulation_steps'], 2)
        self.assertEqual(joint['initial_style_id'], '')
        continued = music_training.joint_training_options({'initial_style_id': 'saved-style'})
        self.assertEqual(continued['learning_rate'], 2e-5)
        self.assertEqual(music_training.joint_training_options({'initial_style_id': 'saved-style', 'learning_rate': 3e-5})['learning_rate'], 3e-5)
        for raw in (None, [], {'window_frames': True}, {'window_frames': 2000}, {'lyric_alignment': True}):
            with self.assertRaises(ValueError):
                music_training.joint_training_options(raw)
        for identity in (None, [], True, '../outside', 'bad/name'):
            with self.assertRaises(ValueError):
                music_training.joint_training_options({'initial_style_id': identity})

    def test_pair_selection_rejects_invalid_types_without_changing_a_project(self):
        from services.music_contracts import tokenizer_pair, adapter_contract
        for value in (None, [], {}, 9, True, 'unknown'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tokenizer_pair({'tokenizer_pair': value})
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter_contract({'adapter_mode': value})

    def test_words_keep_original_unicode_offsets_and_skip_tags(self):
        lyrics = self.project['tracks'][0]['lyrics']
        words = lyric_words(lyrics)
        self.assertEqual([w[2] for w in words], ['hello','hello','cafe',"don't",'wait'])
        self.assertTrue(all('[' not in lyrics[a:b] for a,b,_ in words))
        with self.assertRaises(ValueError):
            lyric_words('这是不支持的歌词 测试 你好 再见')

    def test_alignment_rejects_invalid_or_reversed_time(self):
        for rows in ([[2, 1, .9, 0, 1]], [[-1, 2, .9, 0, 1]], [[0, 3, float('nan'), 0, 1]], [[0, 50, .9, 0, 1]]):
            with self.assertRaises(ValueError):
                validate_words(rows, 'word', 10)

    def test_cursor_masks_intro_low_confidence_and_truncation(self):
        path = Path(__file__).resolve().parents[1] / 'app/ckpts/yue2_music_training/qwen.tiktoken'
        if not path.exists():
            self.skipTest('Optional text tokenizer not installed')
        tokenizer = YuE2TextTokenizer(str(path))
        track = self.project['tracks'][0]
        directory = music_training.project_directory(self.project['id']) / 'alignment'; directory.mkdir()
        words = lyric_words(track['lyrics'])
        rows = [[2+i, 2.3+i, .9 if i != 1 else .01, a, b] for i,(a,b,_) in enumerate(words)]
        data = {'identity': identity(self.project, track), 'summary': {'status': 'ready'}, 'words': rows}
        (directory / (track['id']+'.json')).write_text(json.dumps(data))
        prefix = token_prefixes(SongRequest(style='My sound, Acoustic', lyrics=track['lyrics'], cot='off'), tokenizer)
        targets = cursor_targets(self.project, track, prefix, tokenizer, 200)
        self.assertGreaterEqual(min(targets['rows']), 50)
        self.assertFalse(any(75 <= r < 83 for r in targets['rows']))
        hidden = torch.randn(len(prefix)+65, 8, requires_grad=True)
        head = torch.nn.Linear(8,8,bias=False)
        result = cursor_loss(hidden, len(prefix), targets, head)
        result.backward()
        self.assertTrue(torch.isfinite(result))
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        empty = cursor_loss(hidden[:len(prefix)+40], len(prefix), targets, head)
        self.assertEqual(float(empty), 0)
        data['identity']['lyrics_sha256'] = 'changed'
        (directory / (track['id']+'.json')).write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            cursor_targets(self.project, track, prefix, tokenizer, 200)


if __name__ == '__main__':
    unittest.main()
