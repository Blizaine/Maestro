"""Completed actions and still-only directions must not reset later windows."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_authored_brief import explicit_negative_constraints, video_direction_source
from services.h3_story_ledger import (
    _canonicalize_segment_contract, _materialize_segment, extract_h3_source_intent,
    extract_source_events, plan_h3_story_segments, segment_violations,
)


OPEN = 'The courier opens the truck doors and dumps the candy into the cola.'
LAUNCH = 'A white foam column launches the courier into the air. The jug falls onto the road.'
HANDOFF = 'The truck doors are open. The candy is in the cola. The courier stands beside the truck, holding the jug.'
TIMELINE = f'[0s-4s] {OPEN}\n[4s-8s] {LAUNCH}'
STILL = 'Produce one still. The truck doors are closed. No candy spilling.\n'
BOUNDARY = "Once it's ready, use it as reference for the video model and this prompt:\n"


def card(action):
    return {'action': action, 'camera': 'Continuous reframing from inside the car',
            'framing': 'Wide', 'transition': 'Continue', 'sound_effects': 'Road ambience'}


class ContinuationHandoffTests(unittest.TestCase):
    def test_explicit_still_restrictions_do_not_become_video_prohibitions(self):
        source = STILL + BOUNDARY + 'No cuts. No dialogue.\n' + TIMELINE
        self.assertNotIn('No candy spilling', explicit_negative_constraints(source))
        self.assertIn('No cuts', explicit_negative_constraints(source))
        intent = extract_h3_source_intent(source)
        self.assertNotIn('No candy spilling', intent['global_instructions'])
        self.assertEqual(len(extract_source_events(source)), 2)
        self.assertIn(OPEN, extract_source_events(source)[0]['text'])

    def test_separately_labelled_image_and_video_prompts(self):
        source = 'IMAGE PROMPT: A closed door. No people.\nVIDEO PROMPT: No cuts. A person opens the door.'
        self.assertEqual(video_direction_source(source), ' No cuts. A person opens the door.')

    def test_ordinary_first_frame_and_global_rules_are_preserved(self):
        for source in (
            'Use the uploaded image as the first frame. No dialogue.\n' + TIMELINE,
            'No dialogue.\nVIDEO PROMPT: ' + TIMELINE,
            'Produce one still. No dialogue.\n' + TIMELINE,
        ):
            with self.subTest(source=source):
                self.assertEqual(video_direction_source(source), source)
                self.assertIn('No dialogue', explicit_negative_constraints(source))

    def test_filmed_subjects_hands_do_not_belong_to_viewpoint(self):
        for source in (
            "First-person POV from inside a car. The courier's hands are gripping a jug outside.",
            'First-person POV. A woman holds a cup with both hands across the table.',
            'A courier is holding a jug in both hands.',
        ):
            with self.subTest(source=source):
                intent = extract_h3_source_intent(source)
                self.assertFalse(intent['hands_visible'])
                self.assertNotIn('held object', intent['perspective_contract'])

    def test_explicit_viewpoint_hands_are_preserved(self):
        for source in (
            'First-person POV. Both hands are gripping a broom in the foreground.',
            'First-person POV. Your hands hold the rope.',
            'First-person POV with hands gripping a broom handle.',
        ):
            with self.subTest(source=source):
                self.assertTrue(extract_h3_source_intent(source)['hands_visible'])

    def canonical(self, closing=HANDOFF):
        events = extract_source_events(TIMELINE)
        beats = [{'beat_id': 'B1', 'description': OPEN, 'source_event_ids': ['E1'],
                  'dialogue_ids': [], 'state_after': 'the immediate visible state is the result of this event: ' + OPEN}]
        segment = _canonicalize_segment_contract(
            {'event_cards': {'event_1': {'phases': [card(OPEN)]}}, 'closing_state': closing},
            segment_number=1, duration=4, assigned_beats=beats, dialogue_catalog=[],
            opening_state='The truck doors are closed.', source_intent={}, source_events=events,
            use_camera_handoff=True,
        )
        return events, beats, segment

    def test_concrete_handoff_survives_final_compilation(self):
        events, beats, segment = self.canonical()
        self.assertEqual(segment['closing_state'], HANDOFF)
        materialized = _materialize_segment(segment, beats=beats, dialogue_catalog=[], source_events=events)
        self.assertEqual(materialized['closing_state'], HANDOFF)

    def test_missing_handoff_keeps_source_fallback(self):
        _, beats, segment = self.canonical(closing='')
        self.assertEqual(segment['closing_state'], beats[-1]['state_after'])

    def test_handoff_does_not_substitute_for_missing_action(self):
        _, beats, segment = self.canonical()
        segment['shots'][0]['action'] = 'Clouds drift across the sky.'
        self.assertTrue(any('omits required source step' in message for message in segment_violations(
            TIMELINE, segment, segment_number=1, duration=4,
            assigned_beats=beats, dialogue_catalog=[],
        )))

    def test_next_camera_writer_receives_achieved_state_and_ending_stays_locked(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            props = kwargs['json_schema']['properties']
            if 'setting_continuity' in props:
                return json.dumps({'character_appearance': 'A courier in a cap.',
                    'setting_continuity': 'Trucks on a road.', 'visual_continuity': 'Daylight.',
                    'editing_style': 'Continuous take', 'ambient_audio': 'Wind'})
            number = props['segment']['minimum']
            if number == 2:
                self.assertIn('Required opening state: ' + HANDOFF, kwargs['prompt'])
                self.assertNotIn('result of this event: ' + OPEN, kwargs['prompt'])
                self.assertNotIn('No candy spilling', kwargs['prompt'])
            return json.dumps({'segment': number, 'coverage': 'Continuous car POV',
                'event_cards': {'event_1': {'phases': [card(OPEN if number == 1 else LAUNCH)]}},
                'closing_state': HANDOFF if number == 1 else 'The courier is magically back in the car.'})

        result = plan_h3_story_segments(STILL + BOUNDARY + 'No dialogue.\n' + TIMELINE,
            segment_durations=[4, 4], mode='sliding_window', camera_coverage='continuous',
            planning_style='adaptive', llm_generate=generate)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['planning_warnings'], [])
        self.assertEqual(result['segments'][1]['opening_state'], HANDOFF)
        self.assertIn('jug falls', result['segments'][1]['closing_state'])
        self.assertNotIn('magically', result['segments'][1]['closing_state'])


if __name__ == '__main__':
    unittest.main()
