"""Instrumental routing, production adapter format, and score-first generation."""
import ast
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))
from models.TTS.yue2 import artist_adapter, instrumental, music_assets
from models.TTS.yue2.prompting import validate_song_inputs
from models.TTS.yue2.protocol import GenerationConfig, SongRequest


def pipeline_method(name):
    source = ast.parse((APP / 'models/TTS/yue2/pipeline.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'YuE2Pipeline')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'torch': torch, '__package__': 'models.TTS.yue2', 'replace': replace,
                 'SongRequest': SongRequest, 'hashlib': hashlib, 'json': json,
                 'token_prefixes': lambda *a: [1, 2], 'negative_prefix': lambda *a: [3, 4]}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'pipeline.py', 'exec'), namespace)
    return namespace[name]


class InstrumentalTests(unittest.TestCase):
    def test_legacy_sentinel_and_flags_select_instrumental(self):
        for inputs, lyrics in [({}, ' [Instrumental] '), ({'_music_instrumental': True}, 'old lyrics'),
                               ({'custom_settings': {'instrumental': True}}, '')]:
            with self.subTest(inputs=inputs):
                self.assertTrue(instrumental.is_instrumental(inputs, lyrics))
        for inputs, lyrics in [({}, '[Verse]\nSung words'), ({'custom_settings': {'instrumental': False}}, 'Words'),
                               ({'custom_settings': {'instrumental': 'false'}}, 'Words')]:
            self.assertFalse(instrumental.is_instrumental(inputs, lyrics))

    def test_section_plans_allow_only_the_authors_trained_forms(self):
        self.assertEqual(instrumental.section_plan('[Intro]\n\n[Verse]\n[Pre-chorus]\n[Chorus]\n[Outro]'),
                         '[intro]\n[verse]\n[pre-chorus]\n[chorus]\n[outro]')
        plan = '[intro 0:00-0:15]\n[verse 0:15-0:45]\n[outro 0:45-1:00]'
        self.assertEqual(instrumental.section_plan(plan), plan)
        for value in ('', '[Instrumental]', '[Verse]\nWords to sing', '[guitar solo, distorted]',
                      '[intro]\\n[outro]', '[intro 0:00-0:99]', '[intro 0:15-0:10]',
                      '[intro 0:00-0:20]\n[outro 0:10-0:30]'):
            with self.subTest(value=value):
                self.assertEqual(instrumental.section_plan(value), '[instrumental]')

    def test_validation_overrides_vocal_mode_without_loading_paused_artists(self):
        settings = {'instrumental': True, 'artist_loras': [{'id': 'unavailable'}], 'abc': 'old score'}
        inputs = {'custom_settings': settings, 'model_mode': 2, 'alt_prompt': 'Piano and strings',
                  'audio_prompt_type': 'A', 'audio_guide': 'old-source.wav'}
        self.assertIsNone(validate_song_inputs(inputs, ''))
        self.assertEqual(inputs['model_mode'], 0)
        self.assertEqual(inputs['audio_prompt_type'], '')
        self.assertEqual(inputs['custom_settings']['abc'], '')
        self.assertEqual(inputs['custom_settings']['artist_loras'], settings['artist_loras'])
        self.assertEqual(settings['abc'], 'old score', 'Do not change the saved vocal settings object')
        for value in ('false', 1):
            self.assertIn('on or off', validate_song_inputs({'alt_prompt': 'piano', 'custom_settings': {'instrumental': value}}, 'words'))

    def test_named_upstream_ar_weights_map_to_real_model_targets(self):
        name = 'model.layers.0.self_attn.q_proj'
        tensors = {'layers.0.self_attn.q_proj.lora_A': torch.ones(1, 2),
                   'layers.0.self_attn.q_proj.lora_B': torch.ones(2, 1)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'instrumental.safetensors'
            save_file(tensors, str(path))
            with patch.object(artist_adapter, 'target_shapes', return_value=[(name, 2, 2)]):
                loaded = artist_adapter.read_upstream_adapter(path, 'ar')
            self.assertEqual(set(loaded), {name + '.A', name + '.B'})
            loaded.clear()  # Release the safetensors mmap before Windows removes the fixture.

    def test_adapter_download_and_scope_are_ar_only_at_unit_strength(self):
        calls = []
        @contextmanager
        def adapter(pipeline, sources, strength):
            calls.append((sources, strength))
            yield
            calls.append('removed')
        pipeline = SimpleNamespace(_abort_requested=lambda: False)
        with patch.object(music_assets, 'ensure_asset', return_value=Path('instrumental.safetensors')) as download, \
             patch.object(artist_adapter, 'active_adapters', adapter):
            with instrumental.active_instrumental(pipeline) as metadata:
                self.assertEqual(metadata['cot'], 'full')
                self.assertEqual(metadata['decoder'], 'stock')
                self.assertEqual(calls, [({'ar': Path('instrumental.safetensors')}, 1.0)])
            self.assertEqual(calls[-1], 'removed')
            self.assertEqual(download.call_args.args, ('instrumental_ar',))

    def test_generation_bypasses_artists_records_recipe_and_next_song_is_vocal(self):
        generate = pipeline_method('generate')
        scopes = []
        @contextmanager
        def inst(_):
            scopes.append('instrumental')
            try:
                yield {'cot': 'full', 'strength': 1.0}
            finally:
                scopes.append('removed')
        @contextmanager
        def artists(*_):
            scopes.append('artists')
            yield [{'id': 'rap', 'trigger': 'Rap voice'}]
        pipeline = SimpleNamespace(_generate=Mock(side_effect=lambda **kw: {}))
        with patch.object(instrumental, 'active_instrumental', inst), patch.object(artist_adapter, 'active_artists', artists):
            result = generate(pipeline, input_prompt='[Instrumental]', alt_prompt='Piano', model_mode=2,
                              custom_settings={'artist_loras': [{'id': 'rap'}]})
            self.assertEqual(pipeline._generate.call_args.kwargs['input_prompt'], '[instrumental]')
            self.assertEqual(pipeline._generate.call_args.kwargs['model_mode'], 0)
            self.assertEqual(pipeline._generate.call_args.kwargs['alt_prompt'], 'Piano')
            self.assertEqual(result['overridden_inputs']['model_mode'], 0)
            self.assertTrue(result['overridden_inputs']['_music_instrumental'])
            self.assertNotIn('artists', result['artifact_metadata'])
            generate(pipeline, input_prompt='New lyrics', alt_prompt='Rap', model_mode=2, custom_settings={})
            self.assertEqual(pipeline._generate.call_args.kwargs['model_mode'], 2)
            self.assertEqual(pipeline._generate.call_args.kwargs['alt_prompt'], 'Rap voice, Rap')
            self.assertEqual(scopes, ['instrumental', 'removed', 'artists'])
            pipeline._generate.side_effect = RuntimeError('generation failed')
            with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                generate(pipeline, input_prompt='[instrumental]', alt_prompt='Piano')
            self.assertEqual(scopes[-1], 'removed')

    def test_full_mode_really_writes_score_before_semantic_tokens(self):
        phases = []
        def tokens(prefix, sampling, seed, phase, *args):
            phases.append(phase)
            return [7, 8]
        pipeline = SimpleNamespace(frame_rate=25, sample_rate=48000, generation_config=GenerationConfig(),
            engine=SimpleNamespace(release_runtime_allocations=Mock()), _tokens=tokens,
            tokenizer=SimpleNamespace(decode=lambda ids: 'X:1\nK:C\nC D E F'), lm_decoder_engine='test',
            decode_codec=Mock(return_value=torch.zeros(1, 2, 32)))
        result = pipeline_method('_generate')(pipeline, input_prompt='[instrumental]', alt_prompt='Piano',
            seed=3, duration_seconds=30, sampling_steps=32, guide_scale=1, temperature=1,
            top_k=100, top_p=.95, model_mode=0, custom_settings={'instrumental': True, 'abc': ''})
        self.assertEqual(phases, ['abc', 'semantic'])
        self.assertEqual(result['artifact_metadata']['plan']['request']['cot'], 'full')
        self.assertTrue(result['artifact_metadata']['plan']['abc'])


if __name__ == '__main__':
    unittest.main()
