"""Semantic coverage review is conditional, bounded and tied to visual evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

from services.h3_camera_fidelity import (
    clear_confirmed_coverage_errors, review_missing_camera_actions,
)
from services.h3_story_ledger import _h3_contract_clauses, plan_h3_story_segments


REQUIREMENT = 'the stream hammers the mouth of the jug the whole time'
PARAPHRASE = 'The soda jet pours continuously through the opening, filling the jug for two seconds.'
ERROR = 'B1 shot action omits required source step: ' + REQUIREMENT
HARD_ERROR = 'B1 shot action drops the explicit before chronology relation'


def decision(quote=PARAPHRASE, **overrides):
    result = {'verdict': 'preserved', 'evidence': [{'card': 1, 'field': 'action', 'quote': quote}]}
    result.update(overrides)
    return json.dumps({'check_1': result})


class CameraFidelityTests(unittest.TestCase):
    def setUp(self):
        self.beats = [{'beat_id': 'B1', 'source_event_ids': ['E1']}]
        self.events = [{'event_id': 'E1', 'text': REQUIREMENT}]
        self.segment = {'camera_contract': 'event_cards', 'shots': [
            {'beat_ids': ['B1'], 'action': PARAPHRASE, 'camera': 'Hold on the filling process.',
             'sound_effects': 'A loud rush of liquid.'},
            {'beat_ids': ['B2'], 'action': 'The courier drinks from the jug.'},
        ]}

    def review(self, response=decision(), errors=None):
        generate = Mock(return_value=response)
        receipts = review_missing_camera_actions(
            [ERROR] if errors is None else errors, self.segment,
            assigned_beats=self.beats, source_events=self.events, generate=generate)
        return receipts, generate

    def test_valid_paraphrase_removes_only_its_lexical_flag(self):
        before = deepcopy(self.segment)
        receipts, generate = self.review(errors=[ERROR, HARD_ERROR])
        self.assertEqual(clear_confirmed_coverage_errors([ERROR, HARD_ERROR], self.segment, receipts), [HARD_ERROR])
        self.assertEqual(self.segment, before)
        generate.assert_called_once()
        packet = json.loads(generate.call_args.kwargs['prompt'])['check_1']
        self.assertEqual(packet['source_requirement'], REQUIREMENT)
        self.assertEqual(len(packet['visual_cards']), 1)
        self.assertNotIn('sound_effects', packet['visual_cards'][0])
        self.assertNotIn('drinks', generate.call_args.kwargs['prompt'])
        self.assertFalse(generate.call_args.kwargs['enable_thinking'])

    def test_no_call_for_clean_or_hard_only_failures(self):
        for errors in ([], [HARD_ERROR], ['assigned beat IDs are missing, foreign, or repeated']):
            receipts, generate = self.review(errors=errors)
            self.assertEqual(receipts, {})
            generate.assert_not_called()

    def test_missing_and_contradicted_action_still_need_repair(self):
        for verdict in ('missing', 'contradicted'):
            receipts, _ = self.review(decision(verdict=verdict, evidence=[]))
            self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [ERROR])

    def test_evidence_cannot_be_invented_global_audio_or_another_event(self):
        self.segment['opening_state'] = 'The jet hammers the mouth of the jug.'
        for evidence in (
            [], [{'card': 1, 'field': 'action', 'quote': 'An invented filling sentence.'}],
            [{'card': 1, 'field': 'opening_state', 'quote': self.segment['opening_state']}],
            [{'card': 1, 'field': 'sound_effects', 'quote': 'A loud rush of liquid.'}],
            [{'card': 2, 'field': 'action', 'quote': 'The courier drinks from the jug.'}],
            [{'card': True, 'field': 'action', 'quote': PARAPHRASE}],
            [{'card': 1, 'field': 'action', 'quote': ''}],
        ):
            with self.subTest(evidence=evidence):
                self.assertEqual(self.review(decision(evidence=evidence))[0], {})

    def test_repair_invalidates_a_review_for_changed_visuals(self):
        receipts, _ = self.review()
        self.segment['shots'][0]['action'] = 'The courier stares at an empty jug.'
        self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [ERROR])

    def test_repair_to_another_event_keeps_valid_evidence(self):
        receipts, _ = self.review()
        self.segment['shots'][1]['action'] = 'The courier sets down the jug.'
        self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [])

    def test_bad_review_and_unavailable_writer_do_not_approve(self):
        for response in ('broken JSON', '{}', '[]', decision(verdict='maybe')):
            self.assertEqual(self.review(response)[0], {})
        generate = Mock(side_effect=RuntimeError('offline'))
        self.assertEqual(review_missing_camera_actions([ERROR], self.segment,
            assigned_beats=self.beats, source_events=self.events, generate=generate), {})

    def test_cancellation_propagates(self):
        with self.assertRaises(InterruptedError):
            review_missing_camera_actions([ERROR], self.segment, assigned_beats=self.beats,
                source_events=self.events, generate=Mock(side_effect=InterruptedError('cancelled')))

    def test_only_then_does_not_leave_a_dangling_requirement(self):
        clauses = _h3_contract_clauses('Liquid keeps flowing into the jug. Only then does he lift it.')
        self.assertFalse(any(clause.endswith('Only') for clause in clauses))
        self.assertIn('does he lift it', clauses)

    def test_complete_planning_preserves_approved_draft_without_repair(self):
        source = '[0s-4s] The stream hammers the mouth of the jug the whole time.\n[4s-8s] The courier drinks from the jug.'
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            props = kwargs['json_schema']['properties']
            if 'setting_continuity' in props:
                return json.dumps({'character_appearance': 'A courier.', 'setting_continuity': 'A road.',
                    'visual_continuity': 'Daylight', 'editing_style': 'Continuous take'})
            if 'check_1' in props:
                return decision()
            self.assertNotIn('REPAIR ONLY', kwargs['prompt'])
            n = props['segment']['minimum']
            action = PARAPHRASE if n == 1 else 'The courier drinks from the jug.'
            return json.dumps({'segment': n, 'coverage': 'Continuous', 'closing_state': action,
                'event_cards': {'event_1': {'phases': [{'action': action, 'camera': 'Hold steady',
                    'framing': 'Wide', 'transition': 'Continue', 'sound_effects': 'Wind'}]}}})

        result = plan_h3_story_segments(source, segment_durations=[4, 4],
            mode='sliding_window', planning_style='adaptive', camera_coverage='continuous',
            llm_generate=generate)
        self.assertEqual(result['planning_warnings'], [])
        self.assertEqual(result['planned_by'], 'llm')
        self.assertEqual(len(calls), 4)  # treatment, camera 1, short review, camera 2
        self.assertEqual(sum('check_1' in call['json_schema']['properties'] for call in calls), 1)
        self.assertIn(PARAPHRASE, result['segments'][0]['shots'][0]['action'])

    def test_real_missing_action_still_gets_one_focused_repair(self):
        source = '[0s-4s] The courier opens the gate.\n[4s-8s] The courier runs down the road.'
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            props = kwargs['json_schema']['properties']
            if 'setting_continuity' in props:
                return json.dumps({'character_appearance': 'A courier.', 'setting_continuity': 'A road.',
                    'visual_continuity': 'Daylight', 'editing_style': 'Continuous take'})
            if 'check_1' in props:
                return decision(verdict='missing', evidence=[])
            n = props['segment']['minimum']
            action = ('The courier opens the gate.' if 'REPAIR ONLY' in kwargs['prompt']
                      else 'Clouds drift across the sky.' if n == 1
                      else 'The courier runs down the road.')
            return json.dumps({'segment': n, 'coverage': 'Continuous', 'closing_state': action,
                'event_cards': {'event_1': {'phases': [{'action': action, 'camera': 'Hold steady',
                    'framing': 'Wide', 'transition': 'Continue', 'sound_effects': 'Wind'}]}}})

        result = plan_h3_story_segments(source, segment_durations=[4, 4],
            mode='sliding_window', planning_style='adaptive', camera_coverage='continuous',
            llm_generate=generate)
        self.assertEqual(result['planning_warnings'], [])
        self.assertEqual(len(calls), 5)
        self.assertEqual(sum('REPAIR ONLY' in call['prompt'] for call in calls), 1)
        self.assertIn('courier opens the gate', result['segments'][0]['shots'][0]['action'])


if __name__ == '__main__':
    unittest.main()
