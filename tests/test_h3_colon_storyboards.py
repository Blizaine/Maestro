"""Imported timeline syntax must not manufacture speech, cast or chronology."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_authored_brief import authored_timed_brief, explicit_character_profiles
from services.h3_story_ledger import (
    _h3_missing_relation_markers, _h3_ordered_relation_pairs,
    _h3_required_relation_markers, extract_h3_source_intent,
    extract_locked_dialogue, extract_source_events, segment_violations,
)


BRIEF = '''DIRECTIVE:
Create an opening still showing two delivery trucks on a road.
CAMERA PACK:
Windshield view.
[STYLE + CAMERA + ATMOSPHERE]
Use the opening image to make a continuous video.
LOCKED SET / VEHICLES:
- LEFT: A red truck with readable livery ("Red Delivery").
- BESIDE it: A blue truck, rear doors CLOSED, livery readable ("The Freshmaker").
LOCKED CAST:
- THE COURIER: one man alone in a red cap. No second person.
- HERO OBJECT: one large plastic jug, initially empty.
[TIMELINE SECOND BY SECOND]
0-3s: [HOOK — ARRIVING + ZOOM] Phone moves closer to the trucks.
3-5s: [HE ENTERS FROM RIGHT] The courier enters from the right, carrying the empty jug.
5-9s: [FILL — HOLD IT] He holds the jug under a stream of water and fills it.
9-12s: [DRINK] The courier drinks from the jug and gasps.
12-16s: [OPEN + DUMP] He opens the rear doors and tips a case into the puddle.
16-20s: [IT GOES] A foam column launches him down the road.
20-24s: [FOUNTAIN] He flies toward the vanishing point, trailing foam.
24-26s: [SHRINKS] His silhouette shrinks into the distance.
26-30s: [PAYOFF — GONE] He disappears into the horizon haze (~1s earlier than before).
Phone holds on the empty road.
[STYLE & QUALITY BOOSTERS]
Continuous take. Keep both trucks consistent. Realistic liquid physics.
'''


class ColonStoryboardTests(unittest.TestCase):
    def test_imported_colon_timeline_preserves_nine_events_and_source_offsets(self):
        brief = authored_timed_brief(BRIEF)
        self.assertEqual(
            [(e['source_start_seconds'], e['source_end_seconds']) for e in brief['events']],
            [(0, 3), (3, 5), (5, 9), (9, 12), (12, 16), (16, 20), (20, 24), (24, 26), (26, 30)],
        )
        self.assertEqual(brief['events'][0]['source_title'], 'HOOK — ARRIVING + ZOOM')
        self.assertEqual(brief['events'][0]['text'], 'Phone moves closer to the trucks.')
        for event in brief['events']:
            self.assertEqual(BRIEF[event['source_offset']:event['source_end']].strip(), event['text'])
        self.assertIn('STYLE & QUALITY BOOSTERS', brief['context'])
        self.assertIn('Create an opening still', brief['context'])
        self.assertNotIn('Realistic liquid physics', brief['events'][-1]['text'])
        events = extract_source_events(BRIEF)
        self.assertEqual(len(events), 9)
        self.assertIn('disappears into the horizon', events[-1]['text'])

    def test_common_line_time_delimiters_and_optional_titles(self):
        for delimiter in (': ', ':', '：', ' | ', ' — ', ' ', '\n'):
            with self.subTest(delimiter=delimiter):
                prompt = (f'0-3s{delimiter}Nora opens a gate.\n'
                          f'3-6s{delimiter}Nora walks through the gate.')
                self.assertEqual(len(authored_timed_brief(prompt)['events']), 2)
        for title in ('[ARRIVAL]', '【ARRIVAL】'):
            prompt = f'0-3s: {title} Nora arrives.\n3-6s: [EXIT] Nora leaves.'
            self.assertEqual(authored_timed_brief(prompt)['events'][0]['source_title'], 'ARRIVAL')
        # An ordinary description or dialogue about time is not a storyboard.
        self.assertEqual(authored_timed_brief('Nora says "0-3s: wait, 3-6s: run."')['events'], [])

    def test_cast_and_written_livery_do_not_become_speakers(self):
        self.assertEqual(extract_locked_dialogue(BRIEF), [])
        self.assertEqual([p['name'] for p in explicit_character_profiles(BRIEF)], ['THE COURIER'])
        self.assertEqual(extract_h3_source_intent(BRIEF)['cast_names'], ['THE COURIER'])

    def test_vehicle_notes_do_not_hide_an_actual_spoken_line(self):
        prompt = ('LOCKED SET / VEHICLES:\n'
                  '- LEFT: A red truck with readable livery ("Red Delivery").\n'
                  'Pat: "Move back!"\n')
        self.assertEqual([(d['speaker'], d['text']) for d in extract_locked_dialogue(prompt)],
                         [('Pat', 'Move back!')])

    def test_single_cast_cap_does_not_swallow_a_later_entering_character(self):
        prompt = ('[Cast]\nNora: a courier.\n[Story]\n'
                  'No second person enters until the courier leaves.\n'
                  'Nora walks away. Zoe enters and waves.')
        self.assertIn('Zoe', extract_h3_source_intent(prompt)['cast_names'])

    def test_comparison_is_not_a_chronology_dependency_on_the_following_event(self):
        # The event compiler joins sentence fragments without their periods.
        # A closing parenthesis formerly let "before" attach to the next event.
        source = ('He disappears into the horizon haze (~1s earlier than before) '
                  'Phone holds on the empty road')
        action = 'He disappears into the horizon haze; the phone holds on the empty road.'
        self.assertEqual(_h3_ordered_relation_pairs(source), [])
        self.assertEqual(_h3_required_relation_markers(source), [])
        self.assertEqual(_h3_missing_relation_markers(source, action), [])
        for comparison in ('earlier than before', 'more quickly than before', 'longer than before'):
            self.assertEqual(_h3_ordered_relation_pairs(f'He vanishes {comparison}, the road is empty.'), [])

    def test_comparisons_do_not_disable_real_before_after_checks(self):
        source = ('Nora closes the gate before the truck departs. '
                  'The truck vanishes earlier than before.')
        self.assertEqual(len(_h3_ordered_relation_pairs(source)), 1)
        self.assertEqual(_h3_missing_relation_markers(
            source, 'The truck departs only after Nora closes the gate.'), [])
        self.assertEqual(_h3_missing_relation_markers(
            source, 'Nora closes the gate after the truck departs.'), ['before'])
        self.assertTrue(_h3_ordered_relation_pairs(
            'Nora feels calmer than before the truck departed.'))

    def test_camera_validator_accepts_comparison_note_without_repeating_it(self):
        prompt = ('Nora disappears into the horizon haze (~1s earlier than before). '
                  'Phone holds on the empty road.')
        events = extract_source_events(prompt)
        beats = [{'beat_id': 'B1', 'segment': 1, 'description': prompt,
                  'source_event_ids': [e['event_id'] for e in events], 'dialogue_ids': []}]
        segment = {'segment': 1, 'semantic_actions': True,
                   'opening_state': 'Nora is visible down the road.',
                   'closing_state': 'The road is empty after Nora vanishes.', 'shots': [{
            'shot': 1, 'beat_ids': ['B1'], 'start_seconds': 0, 'end_seconds': 4,
            'action': 'Nora disappears into the horizon haze; the phone holds on the empty road.',
            'dialogue': [],
        }]}
        self.assertEqual(segment_violations(
            prompt, segment, segment_number=1, duration=4,
            assigned_beats=beats, dialogue_catalog=[],
        ), [])


if __name__ == '__main__':
    unittest.main()
