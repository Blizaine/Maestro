"""A fixed H3 music soundtrack must not become newly generated dialogue."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP = Path(os.environ.get('MAESTRO_TEST_MUSIC_APP', ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'app'))


def module_from_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMusicAudioContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogue = module_from_source('services.director.music_contract_subject', APP / 'services/director/h3_dialogue.py')
        path = APP / 'services/director_pipeline.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_apply_h3_music_audio_contract')
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), namespace)
        cls.apply = staticmethod(namespace['_apply_h3_music_audio_contract'])

    def plan(self):
        return {
            'video_prompt': 'Mat performs at center stage. Eli plays drums behind Mat. Mat sings <d>[English] running free</d>.',
            '_director_h3_model_family': 'ref2va',
            '_director_subjects_on_screen': [
                {'character_id': 'mat', 'speaker_name': 'Mat', 'visual_description': 'Mat in a blue jacket'},
                {'character_id': 'eli', 'speaker_name': 'Eli', 'visual_description': 'Eli in a red shirt'},
            ],
            '_director_dialogue_beats': [{'speaker_id': 'mat', 'spoken_text': 'running free'}],
            '_director_duration_sec': 8,
            '_director_audio_plan': {'mode': 'dialogue_driven'},
        }

    def test_song_lyrics_cannot_request_generated_dialogue_between_two_characters(self):
        plan = self.plan()
        original = plan['video_prompt']
        self.apply('minimax_h3_ref2va', [plan], {'pipeline_type': 'music_video', 'audio_path': 'song.wav'})
        self.dialogue.compile_h3_clip_plans([plan])
        self.assertNotIn('<d>', plan['video_prompt'])
        self.assertIn('mapped driving audio', plan['video_prompt'])
        self.assertIn('Mat', plan['video_prompt'])
        self.assertIn('Eli', plan['video_prompt'])
        self.assertEqual(plan['_director_h3_source_prompt'], original)
        self.assertEqual(plan['_director_dialogue_beats'], [])
        self.assertEqual(self.dialogue.validate_h3_prompt_contract(plan['video_prompt'], [], mode='ref2va'), [])

    def test_saved_music_plan_recompilation_stays_idempotent(self):
        plan = self.plan()
        plan['_director_audio_plan']['mode'] = 'music_driven'
        self.dialogue.compile_h3_clip_plans([plan])
        first = plan['video_prompt']
        self.dialogue.compile_h3_clip_plans([plan])
        self.assertEqual(plan['video_prompt'], first)
        self.assertNotIn('<d>', first)

    def test_narrative_dialogue_is_retained_even_with_a_soundtrack(self):
        plan = self.plan()
        original = copy.deepcopy(plan)
        self.apply('minimax_h3_ref2va', [plan], {'pipeline_type': 'short_film_story', 'audio_path': 'score.wav'})
        self.assertEqual(plan, original)
        self.dialogue.compile_h3_clip_plans([plan])
        self.assertIn('<d>[English] running free</d>', plan['video_prompt'])

    def test_no_source_audio_and_other_models_are_not_changed(self):
        for model, params in [('minimax_h3', {'pipeline_type': 'music_video'}),
                              ('ltx2_25', {'pipeline_type': 'music_video', 'audio_path': 'song.wav'})]:
            plan = self.plan()
            original = copy.deepcopy(plan)
            self.apply(model, [plan], params)
            self.assertEqual(plan, original)

    def test_explicit_user_speech_in_music_brief_is_not_discarded(self):
        plan = self.plan()
        self.apply('minimax_h3_ref2va', [plan], {
            'pipeline_type': 'music_video', 'audio_path': 'song.wav',
            'scene_description': 'Mat says, "running free" while Eli plays drums.',
        })
        self.assertEqual(plan['_director_audio_plan']['mode'], 'audio_driven')
        self.assertEqual(len(plan['_director_dialogue_beats']), 1)
        self.dialogue.compile_h3_clip_plans([plan])
        self.assertIn('<d>[English] running free</d>', plan['video_prompt'])

    def test_quoted_visual_description_does_not_become_a_speech_exception(self):
        plan = self.plan()
        self.apply('minimax_h3_ref2va', [plan], {
            'pipeline_type': 'music_video', 'audio_path': 'song.wav',
            'scene_description': 'Use a "warm vintage" photographic look for the band.',
        })
        self.assertEqual(plan['_director_audio_plan']['mode'], 'music_driven')
        self.assertEqual(plan['_director_dialogue_beats'], [])

    def test_saved_reviewed_music_speech_survives_cleared_ai_cache(self):
        plan = self.plan()
        plan.update({
            '_director_prompt_user_edited': True,
            '_director_h3_source_prompt': 'Mat says, "Welcome to the show." Eli stays behind the drums.',
            '_director_h3_compiled_prompt': '',
            '_director_dialogue_beats': [],
        })
        self.apply('minimax_h3_ref2va', [plan], {
            'pipeline_type': 'music_video', 'audio_path': 'song.wav',
            'scene_description': 'A band performs on stage.',
        })
        self.dialogue.compile_h3_clip_plans([plan])
        self.assertIn('<d>[English] Welcome to the show.</d>', plan['video_prompt'])
        self.assertNotIn('running free', plan['video_prompt'])

    def test_direct_compiler_uses_music_audio_even_if_writer_supplies_beats(self):
        plan = self.plan()
        prompt, contract = self.dialogue.compile_h3_official_prompt(
            plan['video_prompt'], plan['_director_subjects_on_screen'], plan['_director_dialogue_beats'],
            mode='ref2va', duration_seconds=8, audio_plan={'mode': 'music_driven'},
        )
        self.assertNotIn('<d>', prompt)
        self.assertIn('mapped driving audio', contract)


if __name__ == '__main__':
    unittest.main()
