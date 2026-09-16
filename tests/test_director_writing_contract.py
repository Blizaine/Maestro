import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.director.writing_contract import director_writing_contract
from services.director.h3_dialogue import compile_h3_official_prompt, validate_h3_prompt_contract


class DirectorWritingContractTests(unittest.TestCase):
    def test_actual_adaptive_rules_reach_director(self):
        contract = director_writing_contract('Two martial artists fight in a courtyard. No dialogue.')
        for expected in ('UNIFIED ENHANCE', 'ACTION SCENE CRAFT', 'exact user-written', 'SOURCE SONG'):
            if expected == 'SOURCE SONG':
                self.assertNotIn(expected, contract)
            else:
                self.assertIn(expected, contract)
        self.assertIn('occupied hands', contract)
        self.assertIn('reviewed edits', contract)

    def test_song_authority_overrides_conversation_craft(self):
        contract = director_writing_contract('A conversation between singers', source_song=True)
        self.assertIn('Do not invent dialogue', contract.replace('\n', ' '))
        self.assertIn('analyzed vocal intervals', contract)

    def test_recorded_dialogue_is_not_creatively_expanded(self):
        contract = director_writing_contract('Two hosts discuss a painting', source_audio=True)
        self.assertIn('SOURCE RECORDING AUTHORITY', contract)
        self.assertIn('Do not expand, paraphrase, duplicate or invent speech', contract)

    def test_podcast_and_short_form_writers_receive_shared_guidance(self):
        from services.director.planners.podcast import PodcastPlanner
        from services.director.planners.viral_video import ViralVideoPlanner

        contract = director_writing_contract('Two hosts discuss a painting', source_audio=True)
        for planner_type, params in (
            (PodcastPlanner, {'clips': [{'start': 0, 'end': 10}]}),
            (ViralVideoPlanner, {'concept': 'Two hosts discuss a painting'}),
        ):
            calls = []

            def generate(**kwargs):
                calls.append(kwargs)
                return json.dumps([{'duration_sec': 10, 'scene_goal': 'Listen to the host',
                                    'action_beats': ['The listener nods'], 'dialogue_beats': []}])

            with self.subTest(planner=planner_type.__name__):
                planner_type(llm_generate=generate).plan(**params, polish_block=contract)
                self.assertEqual(1, len(calls))
                self.assertIn(contract, calls[0]['system_prompt'])

    def test_optional_guide_is_only_loaded_when_enabled(self):
        with patch('services.director.writing_contract.load_guide', return_value='OPTIONAL_GUIDANCE') as load:
            self.assertNotIn('OPTIONAL_GUIDANCE', director_writing_contract('A quiet landscape'))
            load.assert_not_called()
            self.assertIn('OPTIONAL_GUIDANCE', director_writing_contract('A quiet landscape', nsfw=True))
            load.assert_called_once_with('enhance', 'nsfw_shared')

    def test_final_compiler_keeps_late_actions_camera_destination_and_final_state(self):
        action = (
            'Char_0 stands beside the table, takes the blue folder with her left hand, and turns toward the open doorway. '
            'She waits for the other person to clear the threshold before stepping forward and moving around the chair. '
            'She pauses beside the door, transfers the folder to her right hand, and uses her free left hand to grasp the handle. '
            'She pulls the door closed behind both people and places the blue folder on the shelf inside the room.'
        )
        camera = ('A medium view follows the folder as she crosses the threshold, then moves sideways to show both people inside '
                  'the adjoining room before settling on the closed door and the folder resting on the shelf.')
        ending = ('Both people remain inside the adjoining room, with the closed door behind them, the chair clear of the threshold, '
                  'the table outside, and the blue folder visibly resting on the shelf beside the window.')
        prompt, _ = compile_h3_official_prompt(
            f'An office. Action: {action} Camera: {camera} Final beat: {ending}',
            [{'character_id': 'Char_0', 'speaker_name': 'Mara', 'visual_description': 'A woman in a blue jacket'}],
            [], duration_seconds=14.375,
        )
        self.assertIn('pulls the door closed behind both people', prompt)
        self.assertIn('the blue folder visibly resting on the shelf beside the window', prompt)
        self.assertIn(camera, prompt)
        self.assertNotIn('Char_0', prompt)
        self.assertIn('Mara stands beside the table', prompt)

    def test_prose_containing_visual_label_is_not_a_field_boundary(self):
        anchor = (
            'Dynamic kung fu fight scene set in the mountains with power hits and cinematic action: '
            'a narrow rocky mountain ridge above a sheer cliff face, with wind-blown mist and distant peaks'
        )
        action = 'The fighters trade blows, separate, and brace on opposite sides.'
        camera = 'A wide shot follows them with a handheld camera: both fighters remain visible.'
        for mode in ('t2va', 'i2va', 'ref2va'):
            with self.subTest(mode=mode):
                prompt, _ = compile_h3_official_prompt(
                    f'An exposed ridge. Action: {action} Camera: {camera} Final beat: Both brace.',
                    [], [], mode=mode, context_anchors=[anchor], duration_seconds=14.375,
                )
                self.assertIn(anchor, prompt)
                self.assertIn(action, prompt)
                self.assertIn(camera, prompt)
                self.assertEqual([], validate_h3_prompt_contract(prompt, mode=mode, context_anchors=[anchor]))
