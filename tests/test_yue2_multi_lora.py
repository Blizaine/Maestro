"""Queue contracts and real multi-adapter math on small CPU-only modules."""
import ast
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))
from services import music_styles
from services.music_contracts import PAIRS, tokenizer_pair
from models.TTS.yue2 import artist_adapter as adapter
from models.TTS.yue2.prompting import validate_song_inputs


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(music_styles, 'STYLE_ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        for name in ('rap', 'singer'):
            self.save(name)

    def save(self, name, pair='v9', adapted=False):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        manifest = {'id': name, 'version': 2, 'architecture': 'yue2', 'name': name,
                    'trigger': name + ' voice', 'base_revision': music_styles.BASE_REVISION,
                    'tokenizer_pair': pair, 'tokenizer_revision': PAIRS[pair]['revision']}
        if adapted:
            manifest['adapted_pair'] = {'head_sha256': ('a' if name == 'rap' else 'b') * 64, 'nar_sha256': 'c' * 64}
            manifest['tokenizer_revision'] = tokenizer_pair(manifest)['revision']
        for branch in ('ar', 'nar'):
            weight = directory / f'{branch}.safetensors'
            weight.write_bytes(f'{name}-{branch}'.encode())
            manifest[branch] = {'file': weight.name, 'sha256': music_styles.file_digest(weight)}
        (directory / 'style.json').write_text(json.dumps(manifest))
        return manifest

    def test_legacy_and_explicit_list_precedence(self):
        self.assertEqual(music_styles.selected_style_entries({'artist_id': 'rap', 'artist_strength': .75}),
                         [{'id': 'rap', 'strength': .75}])
        self.assertEqual(music_styles.selected_style_entries({'artist_id': 'rap', 'artist_loras': []}), [])
        settings = {'artist_id': 'stale', 'artist_loras': [{'id': 'rap', 'strength': .5}, {'id': 'singer', 'strength': 1.25}]}
        items = music_styles.generation_styles(settings, verify=True)
        self.assertEqual([(i['id'], i['strength']) for i in items], [('rap', .5), ('singer', 1.25)])
        # Library visibility cannot invalidate a queued selection.
        music_styles.update_style_library('rap', {'archived': True, 'in_selector': False})
        self.assertEqual(len(music_styles.generation_styles(settings, verify=True)), 2)

    def test_bad_selections_fail_before_generation(self):
        for raw in (None, {}, 'rap', [{'id': 'rap'}, {'id': 'rap'}], [{'id': '../outside'}],
                    [{'id': 12}], [{'id': 'rap', 'file': '/override'}],
                    *[[{'id': 'rap', 'strength': v}] for v in (True, None, -1, 2, float('nan'), float('inf'))]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                music_styles.generation_styles({'artist_loras': raw})
        with self.assertRaises(ValueError):
            music_styles.generation_styles({'artist_loras': [{'id': 'missing'}]})

    def test_different_adapted_voices_allowed_but_v4_v9_mix_rejected(self):
        self.save('rap', adapted=True)
        self.save('singer', adapted=True)
        settings = {'artist_loras': [{'id': 'rap'}, {'id': 'singer'}]}
        self.assertEqual(len(music_styles.generation_styles(settings, verify=True)), 2)
        self.save('singer', pair='v4')
        with self.assertRaisesRegex(ValueError, 'v4 and v9'):
            music_styles.generation_styles(settings)

    def test_song_validation_checks_all_ids_and_requires_direct_mode(self):
        inputs = {'alt_prompt': 'Rap verses and sung chorus', 'custom_settings': {
            'artist_loras': [{'id': 'rap', 'strength': .5}, {'id': 'singer', 'strength': 1.25}]}}
        self.assertIsNone(validate_song_inputs(inputs, '[Verse]\nNew lyrics'))
        self.assertIn('Direct generation', validate_song_inputs({**inputs, 'model_mode': 0}, 'Hello'))
        self.assertIn('missing', validate_song_inputs({**inputs, 'custom_settings': {
            'artist_loras': [{'id': 'rap'}, {'id': 'missing'}]}}, 'Hello'))

    def test_active_bundles_resolve_metadata_and_preserve_single_diagnostics(self):
        settings = {'artist_loras': [{'id': 'rap', 'strength': .5}, {'id': 'singer', 'strength': 1.25}]}
        captured = []
        @contextmanager
        def capture(pipeline, items):
            captured.extend(items)
            yield
        with patch.object(adapter, 'active_adapter_stack', capture):
            with adapter.active_artists(None, settings) as artists:
                self.assertEqual([a['id'] for a in artists], ['rap', 'singer'])
                self.assertEqual([a['trigger'] for a in artists], ['rap voice', 'singer voice'])
                self.assertTrue(all(len(a['ar_sha256']) == 64 for a in artists))
            self.assertEqual([i['strength'] for i in captured], [.5, 1.25])
            with adapter.active_artist(None, {'artist_id': 'rap', 'artist_strength': 0}) as artist:
                self.assertEqual((artist['id'], artist['strength']), ('rap', 0))
        (self.root / 'singer/nar.safetensors').write_bytes(b'changed')
        with patch.object(adapter, 'active_adapter_stack') as hooks:
            with self.assertRaisesRegex(ValueError, 'changed'), adapter.active_artists(None, settings):
                pass
            hooks.assert_not_called()

    def test_runtime_settings_survive_the_real_wgp_collector(self):
        source = ast.parse((APP / 'wgp.py').read_text(encoding='utf-8'))
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_custom_settings_from_inputs')
        handler = ast.parse((APP / 'models/TTS/yue2/yue2_handler.py').read_text(encoding='utf-8'))
        runtime = next(ast.literal_eval(value) for n in ast.walk(handler) if isinstance(n, ast.Dict)
                       for key, value in zip(n.keys, n.values) if isinstance(key, ast.Constant) and key.value == 'runtime_custom_settings')
        ns = {'get_model_custom_settings': lambda _: []}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'wgp.py', 'exec'), ns)
        settings = {'artist_loras': [{'id': 'rap', 'strength': .6}, {'id': 'singer', 'strength': 1.2}], 'instrumental': True}
        result, error = ns[fn.name]({'runtime_custom_settings': runtime}, {'custom_settings': settings}, strict=True)
        self.assertIsNone(error)
        self.assertEqual(result, settings)


class AdapterMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.ar, self.nar = torch.nn.Module(), torch.nn.Module()
        self.ar.linear = torch.nn.Linear(2, 2, bias=False).bfloat16()
        self.nar.linear = torch.nn.Linear(2, 2, bias=False).bfloat16()
        self.nar.vae2llm = torch.nn.Linear(2, 2).bfloat16()
        self.nar.llm2vae = torch.nn.Linear(2, 2).bfloat16()
        self.pipeline = SimpleNamespace(text_encoder=self.ar, transformer=self.nar,
                                        engine=SimpleNamespace(release_runtime_allocations=Mock()))
        self.x = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
        self.modules = [self.ar.linear, self.nar.linear, self.nar.vae2llm, self.nar.llm2vae]
        self.baselines = [module(self.x).detach() for module in self.modules]
        self.weights = {}
        for label, scale in [('first', 1.), ('second', 3.)]:
            self.weights[label] = {'linear.A': torch.ones(1, 2), 'linear.B': torch.ones(2, 1) * scale,
                **{f'{name}.{key}': torch.full((2, 2) if key == 'weight' else (2,), scale)
                   for name in ('vae2llm', 'llm2vae') for key in ('weight', 'bias')}}
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return original_to(tensor, *args, **kwargs)
        self.stack.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
        self.stack.enter_context(patch.object(adapter, 'target_shapes', return_value=[('linear', 2, 2)]))
        self.reader = self.stack.enter_context(patch.object(adapter, 'read_upstream_adapter', side_effect=lambda p, *a, **k: self.weights[p]))

    def item(self, label, strength, mode='separate', branches=('ar', 'nar')):
        return {'sources': dict.fromkeys(branches, label), 'strength': strength, 'mode': mode}

    def assert_clean(self):
        for module, base in zip(self.modules, self.baselines):
            self.assertFalse(module._forward_hooks)
            torch.testing.assert_close(module(self.x), base)

    def test_full_companions_are_weighted_once_order_independent_and_cleaned(self):
        items = [self.item('first', .5), self.item('second', 1.5)]
        for order in (items, list(reversed(items))):
            with adapter.active_adapter_stack(self.pipeline, order):
                torch.testing.assert_close(self.ar.linear(self.x), self.baselines[0] + 3 * (.5 + 3 * 1.5))
                torch.testing.assert_close(self.nar.linear(self.x), self.baselines[1] + 3 * (.25 + 3 * .75))
                expected = F.linear(self.x, torch.full((2, 2), 2.5, dtype=torch.bfloat16), torch.full((2,), 2.5, dtype=torch.bfloat16))
                for projection in self.modules[2:]:
                    torch.testing.assert_close(projection(self.x), expected)
                    self.assertEqual(len(projection._forward_hooks), 1)
            self.assert_clean()

    def test_joint_loras_add_at_independent_strengths(self):
        with adapter.active_adapter_stack(self.pipeline, [self.item('first', .5, 'joint'), self.item('second', .25, 'joint')]):
            for module, base in zip(self.modules[:2], self.baselines[:2]):
                torch.testing.assert_close(module(self.x), base + 3 * (.5 + 3 * .25))
            for projection, base in zip(self.modules[2:], self.baselines[2:]):
                torch.testing.assert_close(projection(self.x), base)
        self.assert_clean()

    def test_same_companion_is_not_doubled_and_joint_delta_can_mix(self):
        with adapter.active_adapter_stack(self.pipeline, [self.item('first', .25), self.item('first', .75)]):
            torch.testing.assert_close(self.nar.linear(self.x), self.baselines[1] + 3)
        with adapter.active_adapter_stack(self.pipeline, [self.item('first', .5), self.item('second', .25, 'joint')]):
            torch.testing.assert_close(self.nar.linear(self.x), self.baselines[1] + 3 * (1 + 3 * .25))
        self.assert_clean()

    def test_legacy_zero_strength_and_ar_only_audio_only_experiments(self):
        with adapter.active_adapters(self.pipeline, {'ar': 'first', 'nar': 'first'}, 0):
            torch.testing.assert_close(self.ar.linear(self.x), self.baselines[0])
            torch.testing.assert_close(self.nar.linear(self.x), self.baselines[1] + 3)
        with adapter.active_adapters(self.pipeline, {'ar': 'first'}, 1):
            torch.testing.assert_close(self.ar.linear(self.x), self.baselines[0] + 3)
            torch.testing.assert_close(self.nar.linear(self.x), self.baselines[1])
        with adapter.active_adapters(self.pipeline, {'nar': 'first'}, 1):
            torch.testing.assert_close(self.ar.linear(self.x), self.baselines[0])
            torch.testing.assert_close(self.nar.linear(self.x), self.baselines[1] + 3)
        with adapter.active_adapter_stack(self.pipeline, [self.item('first', 0), self.item('second', 0)]):
            torch.testing.assert_close(self.ar.linear(self.x), self.baselines[0])
            torch.testing.assert_close(self.nar.linear(self.x), self.baselines[1] + 6)
        self.assert_clean()

    def test_no_stale_hooks_on_load_generation_or_teardown_failure(self):
        with self.assertRaises(KeyError), adapter.active_adapter_stack(self.pipeline, [self.item('first', 1), self.item('missing', 1)]):
            pass
        self.assert_clean()
        with self.assertRaisesRegex(RuntimeError, 'generation'), adapter.active_adapters(self.pipeline, {'ar': 'first'}, 1):
            raise RuntimeError('generation failed')
        self.assert_clean()
        self.pipeline.engine.release_runtime_allocations.side_effect = [None, RuntimeError('release failed')]
        with self.assertRaisesRegex(RuntimeError, 'release'), adapter.active_adapters(self.pipeline, {'ar': 'first'}, 1):
            pass
        self.assert_clean()


class PromptMetadataTests(unittest.TestCase):
    def test_pipeline_uses_all_triggers_and_saves_all_artists(self):
        artists = [{'id': 'a', 'name': 'A', 'trigger': 'Rap voice', 'strength': .75},
                   {'id': 'b', 'name': 'B', 'trigger': 'Pop voice', 'strength': 1.25}]
        self.assertEqual(adapter.artist_style_prompt('RAP VOICE verses; pop chorus', artists),
                         'Pop voice, RAP VOICE verses; pop chorus')
        self.assertEqual(adapter.artist_style_prompt('duet', [artists[0], artists[0]]), 'Rap voice, duet')
        source = ast.parse((APP / 'models/TTS/yue2/pipeline.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'YuE2Pipeline')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        ns = {'torch': torch, '__package__': 'models.TTS.yue2'}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'pipeline.py', 'exec'), ns)
        @contextmanager
        def selected(*args):
            yield artists
        pipeline = SimpleNamespace(_generate=Mock(return_value={'artifact_metadata': {}}))
        with patch.object(adapter, 'active_artists', selected):
            result = ns['generate'](pipeline, input_prompt='[Verse]\nNew words', alt_prompt='Duet', custom_settings={})
        self.assertEqual(pipeline._generate.call_args.kwargs['alt_prompt'], 'Rap voice, Pop voice, Duet')
        self.assertEqual(result['artifact_metadata']['artists'], artists)
        self.assertIn('artist_mix', result['artifact_metadata'])
        self.assertNotIn('artist', result['artifact_metadata'], 'Do not mislabel a mix as its first singer')


if __name__ == '__main__':
    unittest.main()
