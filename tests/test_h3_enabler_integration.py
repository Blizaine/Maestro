"""Source-proven passage dependencies survive schedule and camera compilation."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.h3_story_ledger import (
    _camera_phase_beats, _co_locate_h3_source_enablers, _deterministic_ledger,
    _grouped_source_order_error, extract_source_events, plan_h3_story_segments,
)


class EnablerIntegrationTests(unittest.TestCase):
    source = (
        "Two adult exhibit volunteers, Sam and Priya, move one red ribbon spool "
        "from a gallery to a workshop so they can hang a banner. "
        "At the start Sam holds the spool in his right hand; it remains unused while "
        "they cross the gallery and pass through the closed workshop door. "
        "Priya opens the door, Sam enters first, and Priya closes it behind them. "
        "Only after they reach the workshop bench does Sam hand the same spool to Priya. "
        "Priya then threads it through the banner's grommets and ties the banner to the wall. "
        "End with Priya holding the spool's loose end, the banner hanging securely, "
        "and the workshop door closed. No dialogue or extra volunteers."
    )

    def test_deterministic_and_camera_paths_keep_both_actions_together(self):
        events = extract_source_events(self.source)
        before = deepcopy(events)
        ledger = _deterministic_ledger(self.source, segment_count=6,
            segment_durations=[8.] * 6, locked_dialogue=[],
            camera_coverage='multi_shot', reference_context='')
        group = next(b for b in ledger['beats'] if b.get('_source_enabler_groups'))
        self.assertEqual(group['source_event_ids'], ['E3', 'E4'])
        phases = _camera_phase_beats([group], source_events=events,
            expected_dialogue_events={}, preserve_adaptation=False)
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0]['_grouped_source_event_ids'], ['E3', 'E4'])
        error = _grouped_source_order_error(phases[0], events,
            'Priya opens the workshop door. Sam enters first. '
            'Sam and Priya cross the gallery and pass through the workshop door.',
            re.compile(r'\b(?:Sam|Priya)\b', re.I))
        self.assertIn('shot action omits required source step', error)
        self.assertNotIn('out of order', error)
        self.assertIn('opening action enables', error)
        self.assertEqual(extract_source_events(self.source), before)

    def test_cross_window_relocation_removes_old_action_staging(self):
        events = extract_source_events(self.source)
        beats = [{'beat_id': f'B{i}', 'segment': min(i, 6),
                  'source_event_ids': [event['event_id']], 'dialogue_ids': [],
                  'description': event['text'], 'state_after': 'OLD_STAGING'}
                 for i, event in enumerate(events[1:], 1)]
        old_ids = [eid for b in beats for eid in b['source_event_ids']]
        result = _co_locate_h3_source_enablers(self.source, {'beats': beats}, locked_dialogue=[])
        self.assertEqual([eid for b in result['beats'] for eid in b['source_event_ids']], old_ids)
        connector = next(b for b in result['beats'] if b.get('_source_enabler_connector'))
        self.assertEqual(connector['source_event_ids'], [])
        self.assertNotIn('OLD_STAGING', str(connector))
        self.assertNotIn('pass through', connector['description'])
        self.assertTrue(connector['description'])
        self.assertIn('OLD_STAGING', str(result['source_enabler_receipts']))

    def test_rejected_camera_keeps_one_achievable_passage_in_final_fallback(self):
        original = extract_source_events(self.source)

        def generate(**kwargs):
            properties = kwargs['json_schema']['properties']
            if 'setting_continuity' in properties:
                return json.dumps({
                    'character_appearance': 'Sam and Priya are adult volunteers.',
                    'setting_continuity': 'A gallery connected to a workshop.',
                    'visual_continuity': 'Natural daylight.',
                    'editing_style': 'Clear sequential coverage.',
                })
            raise ValueError('Deliberately rejected camera draft')

        result = plan_h3_story_segments(
            self.source, segment_durations=[8.] * 6, mode='sliding_window',
            planning_style='faithful', camera_coverage='multi_shot',
            llm_generate=generate,
        )
        actions = [shot['action'] for segment in result['segments']
                   for shot in segment['shots']]
        passage = next(action for action in actions if 'Priya opens' in action)
        entry = re.search(r'Sam enter(?:s|ing) first', passage)
        self.assertIsNotNone(entry)
        self.assertLess(passage.index('Priya opens'), entry.start())
        self.assertIn('cross the gallery', passage)
        combined = ' '.join(actions)
        self.assertIn('Priya closes it behind them', combined)
        self.assertLess(re.search(r'Sam enter(?:s|ing) first', combined).start(),
                        combined.index('Priya closes it behind them'))
        self.assertNotIn('pass through the closed', ' '.join(actions))
        self.assertNotIn('pass through the closed', ' '.join(
            segment['summary'] for segment in result['segments']))
        self.assertEqual(sum(action.count('Priya opens') for action in actions), 1)
        self.assertEqual(len(re.findall(r'Sam enter(?:s|ing) first', combined)), 1)
        self.assertTrue(result['planning_warnings'])  # The failed writer remains visible.
        self.assertEqual(extract_source_events(self.source), original)


if __name__ == '__main__':
    unittest.main()
