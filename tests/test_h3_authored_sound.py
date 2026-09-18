"""Local sound directions should not force an otherwise valid camera rewrite."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_authored_brief import authored_sound_cues
from services.h3_story_ledger import (
    _canonicalize_segment_contract, _materialize_segment, extract_source_events,
    plan_h3_story_segments, segment_violations,
)


SOURCE = (
    '[0s-4s] Hard fizz, then a white foam column erupts between the trucks. '
    'The foam strikes the courier from below and launches him up the road. '
    'His feet leave the asphalt and the jug flies from his hand.\n'
    '[4s-8s] The courier disappears into the horizon haze. The road is empty.'
)
LAUNCH = ('A white foam column erupts between the trucks, hits the courier from '
          'below and launches him up the road. His feet leave the asphalt and '
          'the jug flies out of his hand.')
ENDING = 'The courier disappears into the horizon haze, leaving the road empty.'


def card(action, audio='A loud chemical hiss and a rising roar'):
    return {'action': action, 'framing': 'Wide', 'camera': 'Track the courier',
            'transition': 'Continue', 'sound_effects': audio}


class AuthoredSoundTests(unittest.TestCase):
    def canonical(self, source=SOURCE, first_phases=None, second_phases=None):
        events = extract_source_events(source)
        beats = [{'beat_id': f'B{i}', 'segment': 1, 'description': event['text'],
                  'source_event_ids': [event['event_id']], 'dialogue_ids': [],
                  'state_after': event['text'], 'authored_duration': 4}
                 for i, event in enumerate(events, 1)]
        draft = {'segment': 1, 'event_cards': {
            'event_1': {'phases': first_phases or [card(LAUNCH)]},
            'event_2': {'phases': second_phases or [card(ENDING, 'Wind')]},
        }, 'closing_state': 'The road is empty.'}
        original = deepcopy(draft)
        segment = _canonicalize_segment_contract(draft, segment_number=1, duration=8,
            assigned_beats=beats, dialogue_catalog=[], opening_state='The courier is by the trucks.',
            source_intent={}, source_events=events)
        self.assertEqual(draft, original)
        return events, beats, segment

    def errors(self, source, beats, segment):
        return segment_violations(source, segment, segment_number=1, duration=8,
            assigned_beats=beats, dialogue_catalog=[])

    def test_reported_sound_is_retained_locally_without_rewriting_action(self):
        events, beats, segment = self.canonical()
        self.assertEqual(self.errors(SOURCE, beats, segment), [])
        self.assertEqual(segment['shots'][0]['action'], LAUNCH)
        self.assertIn('Hard fizz', segment['shots'][0]['sound_effects'])
        self.assertNotIn('Hard fizz', segment['shots'][1]['sound_effects'])
        rendered = _materialize_segment(segment, beats=beats, dialogue_catalog=[], source_events=events)
        self.assertIn('Hard fizz', rendered['shots'][0]['sound_effects'])

    def test_existing_cue_keeps_its_phase_without_duplication(self):
        _, beats, segment = self.canonical(first_phases=[
            card(LAUNCH, 'A hiss'), card('The courier continues flying up the road.', 'Hard fizz'),
        ])
        self.assertEqual(self.errors(SOURCE, beats, segment), [])
        self.assertNotIn('Hard fizz', segment['shots'][0]['sound_effects'])
        self.assertEqual(' '.join(s['sound_effects'] for s in segment['shots']).count('Hard fizz'), 1)

    def test_trailing_cue_is_retained_at_the_end_of_its_event(self):
        source = SOURCE.replace('Hard fizz, then a', 'A').replace(
            'flies from his hand.', 'flies from his hand. A sharp metallic clang.')
        _, _, segment = self.canonical(source=source, first_phases=[
            card(LAUNCH), card('The jug lands on the road.', 'Wind'),
        ])
        self.assertNotIn('metallic clang', segment['shots'][0]['sound_effects'])
        self.assertIn('A sharp metallic clang', segment['shots'][1]['sound_effects'])

    def test_sfx_can_prove_sound_but_not_missing_physical_action(self):
        _, beats, segment = self.canonical()
        segment['shots'][0]['action'] = 'Clouds drift across the empty sky.'
        segment['shots'][0]['sound_effects'] = SOURCE
        self.assertTrue(any('omits required source step' in e
                            for e in self.errors(SOURCE, beats, segment)))

    def test_other_beat_and_global_audio_cannot_satisfy_local_cue(self):
        _, beats, segment = self.canonical()
        segment['shots'][0]['sound_effects'] = 'Wind'
        segment['shots'][1]['sound_effects'] = 'Hard fizz'
        segment['ambient_audio'] = 'Hard fizz'
        self.assertTrue(any('Hard fizz' in e for e in self.errors(SOURCE, beats, segment)))

    def test_continuing_sound_is_retained_without_rewriting_visible_action(self):
        source = SOURCE + ' Loop-open: empty horizon, roar still going.'
        _, beats, segment = self.canonical(source=source)
        self.assertEqual(self.errors(source, beats, segment), [])
        self.assertEqual(segment['shots'][-1]['action'], ENDING)
        self.assertIn('roar still going', segment['shots'][-1]['sound_effects'])
        self.assertNotIn('roar still going', segment['shots'][0]['sound_effects'])

    def test_continuing_sound_grammar_cannot_hide_a_physical_event(self):
        for cue in ('roar still going', 'The hiss continues', 'A distant rumble fades away',
                    'The hum is still going', 'A faint ringing keeps echoing'):
            with self.subTest(cue=cue):
                self.assertEqual(authored_sound_cues(cue + '.'), [cue])
        for action in ('The courier roars', 'A roar knocks him down',
                       'The roar fades after the door opens', 'The footsteps follow the courier',
                       'Roar still going while the wall collapses'):
            with self.subTest(action=action):
                self.assertEqual(authored_sound_cues(action + '.'), [])

    def test_compound_action_with_sound_is_not_proven_by_sfx_alone(self):
        source = ('[0s-4s] Hard fizz. The courier opens the gate. ' + 'The camera holds. ' * 40
                  + '[4s-8s] The courier disappears into the horizon haze.')
        _, beats, segment = self.canonical(source=source)
        segment['shots'][0]['action'] = 'A clock ticks.'
        segment['shots'][0]['framing'] = 'Close-up'
        segment['shots'][0]['camera'] = 'Static'
        segment['shots'][0]['sound_effects'] = source
        self.assertTrue(any('omits required source step' in e
                            for e in self.errors(source, beats, segment)))

    def test_sound_extraction_excludes_actions_dialogue_and_dependencies(self):
        for cue in ('Hard fizz', 'A sharp metallic clang', 'Soft footsteps',
                    'A distant rumble', 'SFX: A loud hiss', 'A brief electronic beep'):
            with self.subTest(cue=cue):
                self.assertEqual(authored_sound_cues(cue + '.'), [cue])
        for text in ('The courier clicks a latch.', 'Glass shatters.', 'The wall collapses.',
                     'Hard fizz before the door opens.', 'A loud hiss after the truck stops.',
                     'Nora says, "Hard fizz."', "Nora says 'A loud hiss.'", '<d>Hard fizz.</d>'):
            with self.subTest(text=text):
                self.assertEqual(authored_sound_cues(text), [])

    def test_reversed_physical_order_still_fails(self):
        source = ('[0s-4s] Hard fizz. The courier opens the gate before the truck departs. '
                  '[4s-8s] The courier disappears into the horizon haze.')
        _, beats, segment = self.canonical(source=source, first_phases=[
            card('The courier opens the gate after the truck departs.', 'Hard fizz'),
        ])
        self.assertTrue(any('chronology relation' in e for e in self.errors(source, beats, segment)))

    def test_complete_planner_does_not_retry_an_audio_cue_as_a_visual_action(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            props = kwargs['json_schema']['properties']
            self.assertNotIn('REPAIR ONLY', kwargs['prompt'])
            if 'setting_continuity' in props:
                return json.dumps({'character_appearance': 'A courier in a cap.',
                    'setting_continuity': 'Two parked trucks on a road.',
                    'visual_continuity': 'Daylight.', 'motion_mechanics': 'Foam pushes the courier.',
                    'editing_style': 'A continuous take.', 'ambient_audio': 'Wind.',
                    'source_adaptation': 'Two events over eight seconds.'})
            number = props['segment']['minimum']
            return json.dumps({'segment': number,
                'event_cards': {'event_1': {'phases': [card(LAUNCH if number == 1 else ENDING)]}},
                'closing_state': LAUNCH if number == 1 else ENDING,
                'coverage': 'Continuous take', 'pacing': 'Immediate launch'})

        result = plan_h3_story_segments(SOURCE, segment_durations=[4, 4],
            mode='frames', planning_style='adaptive', camera_coverage='continuous',
            llm_generate=generate)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['planning_warnings'], [])
        self.assertIn('Hard fizz', result['segments'][0]['shots'][0]['sound_effects'])


if __name__ == '__main__':
    unittest.main()
