"""Exercise the preview routes before any Director queue/project is opened."""

import ast
from pathlib import Path
import sys
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from services import director_pipeline


class JsonRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


def fresh_routes():
    model = {
        'architecture': 'minimax_h3_ref2va', 'bounded_video': True,
        'fps': 24, 'frames_minimum': 124, 'frames_steps': 17, 'frames_maximum': 345,
        'director_memory_policy': {
            'resolution_bands': [{'min_pixels': 0, 'vram_tiers': [{'frames': 243}]}],
        },
    }
    engine = SimpleNamespace(get_model_def=lambda _: model,
                             get_model_min_frames_and_step=lambda _: (124, 17, 1))
    namespace = {
        'Request': JsonRequest, 'HTTPException': RuntimeError, 'traceback': traceback,
        'wgp': engine, '_pipeline_initialized': False, '_jobs': {}, '_run_generation': None,
        '_gen_lock': None, '_active_gen_states': {}, '_on_director_pipeline_terminal': None,
        '_on_director_queue_terminal': None,
    }
    source = ROOT / 'app/launch.py'
    names = {'_init_pipeline', 'get_director_music_clip_limits', 'plan_audio_structure'}
    nodes = [node for node in ast.parse(source.read_text(encoding='utf-8')).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class MusicPreviewStartupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        refs = ('_wgp', '_jobs', '_run_generation', '_gen_lock', '_active_gen_states',
                '_terminal_callback', '_queue_terminal_callback')
        for guard in (patch.multiple(director_pipeline, **{name: None for name in refs}),
                      patch.object(director_pipeline, '_director_hardware_snapshot',
                                   return_value={'gpu_vram_gb': 24})):
            guard.start()
            self.addCleanup(guard.stop)

    async def test_limits_initialize_model_registry_on_the_first_request(self):
        for override, expected in ((None, 243), (345, 345)):
            with self.subTest(override=override):
                director_pipeline._wgp = None
                routes = fresh_routes()
                body = {'video_model': 'minimax_h3_ref2va_fused_turbo',
                        'video_params': {'resolution': '1280x704'},
                        'director_music_clip_seconds': None}
                if override:
                    body['director_max_shot_frames'] = override
                limits = await routes['get_director_music_clip_limits'](JsonRequest(body))
                self.assertTrue(routes['_pipeline_initialized'])
                self.assertEqual(limits['frames_minimum'], 124)
                self.assertEqual(limits['hard_max_frames'], 345)
                self.assertEqual(limits['recommended_frames'], 243)
                self.assertEqual(limits['max_frames'], expected)

    async def test_first_structure_preview_uses_native_frames_and_musical_output(self):
        routes = fresh_routes()
        result = await routes['plan_audio_structure'](JsonRequest({
            'video_model': 'minimax_h3_ref2va_fused_turbo',
            'video_params': {'resolution': '1280x704'},
            'director_music_clip_seconds': None, 'director_max_shot_frames': 345,
            'analysis': {'duration': 30, 'bpm': 120,
                         'beats': [{'time': index / 2} for index in range(60)],
                         'sections': [{'start': 0, 'end': 16, 'label': 'verse', 'energy': .4},
                                      {'start': 16, 'end': 30, 'label': 'chorus', 'energy': .9}]},
        }))
        self.assertEqual(result['clip_limits']['max_frames'], 345)
        self.assertIn(16, [clip['start'] for clip in result['clips']])
        self.assertEqual(sum(clip['output_frames'] for clip in result['clips']), 30 * 24)
        for clip in result['clips']:
            self.assertLessEqual(clip['duration_frames'], 345)
            self.assertEqual((clip['duration_frames'] - 124) % 17, 0)

    async def test_slowest_preview_uses_manual_gpu_limit_and_full_soundtrack(self):
        routes = fresh_routes()
        for limit, expected_count in ((243, 12), (345, 9)):
            with self.subTest(limit=limit):
                result = await routes['plan_audio_structure'](JsonRequest({
                    'video_model': 'minimax_h3_ref2va_fused_turbo',
                    'video_params': {'resolution': '1280x704'},
                    'director_music_clip_seconds': None, 'director_max_shot_frames': limit,
                    'energy_bias': -2,
                    'analysis': {'duration': 119.999, 'bpm': 120,
                                 'beats': [{'time': index / 2} for index in range(240)],
                                 'sections': [{'start': index*10, 'end': (index+1)*10,
                                               'label': 'verse', 'energy': .6} for index in range(12)]},
                }))
                clips = result['clips']
                self.assertEqual(result['clip_limits']['max_frames'], limit)
                self.assertEqual(len(clips), expected_count)
                self.assertEqual(sum(c['output_frames'] for c in clips), 120*24)
                self.assertTrue(all(c['duration_frames'] <= limit for c in clips))
                section_times = {clip['start'] for clip in clips} | {
                    cue['time'] for clip in clips for cue in clip['music_cues'] if cue['type'] == 'section_change'}
                self.assertTrue(set(range(0, 120, 10)).issubset(section_times))


class HardwarePreviewTests(unittest.TestCase):
    def test_pinokio_main_module_supplies_hardware_without_reimporting_server(self):
        main = SimpleNamespace(_get_cached_hardware=lambda: {'gpu_vram_gb': 24})
        with patch.dict(sys.modules, {'launch': None, '__main__': main}):
            self.assertEqual(director_pipeline._director_hardware_snapshot(), {'gpu_vram_gb': 24})


if __name__ == '__main__':
    unittest.main()
