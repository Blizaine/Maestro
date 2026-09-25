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
COMPOUND_REQUIREMENT = "Priya threads it through the banner's grommets and ties the banner to the wall"
COMPOUND_ERROR = 'B1 shot action omits required source step: ' + COMPOUND_REQUIREMENT


def decision(span_id='check_1_card_1_action_span_1', **overrides):
    result = {'verdict': 'preserved', 'evidence_span_ids': [span_id]}
    result.update(overrides)
    return json.dumps({'check_1': result})


def action_decisions_response(action_decisions, check_id='check_1'):
    return json.dumps({check_id: {'action_decisions': action_decisions}})


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

    def set_compound_coverage(self):
        self.beats = [{'beat_id': 'B1', 'source_event_ids': ['E7']}]
        self.events = [{'event_id': 'E7', 'text': COMPOUND_REQUIREMENT}]
        self.segment = {'camera_contract': 'event_cards', 'shots': [
            {'beat_ids': ['B1'],
             'action': "Priya feeds the ribbon through the banner's metal eyelets.",
             'camera': 'Follow the ribbon at the banner.'},
            {'beat_ids': ['B1'],
             'action': "Priya loops the loose end around the spool anchor and secures a knot.",
             'camera': 'Hold on the knot and spool.'},
        ]}

    def test_valid_paraphrase_removes_only_its_lexical_flag(self):
        before = deepcopy(self.segment)
        receipts, generate = self.review(errors=[ERROR, HARD_ERROR])
        self.assertEqual(clear_confirmed_coverage_errors([ERROR, HARD_ERROR], self.segment, receipts), [HARD_ERROR])
        self.assertEqual(self.segment, before)
        generate.assert_called_once()
        packet = json.loads(generate.call_args.kwargs['prompt'])['check_1']
        self.assertEqual(packet['source_requirement'], REQUIREMENT)
        self.assertEqual({span['card'] for span in packet['visual_spans']}, {1})
        self.assertNotIn('sound_effects', json.dumps(packet))
        self.assertEqual(packet['visual_spans'][0]['text'], PARAPHRASE)
        self.assertEqual(packet['visual_spans'][0]['field'], 'action')
        self.assertEqual(packet['visual_spans'][0]['span_id'], 'check_1_card_1_action_span_1')
        self.assertNotIn('drinks', generate.call_args.kwargs['prompt'])
        self.assertFalse(generate.call_args.kwargs['enable_thinking'])

    def test_no_call_for_clean_or_hard_only_failures(self):
        for errors in ([], [HARD_ERROR], ['assigned beat IDs are missing, foreign, or repeated']):
            receipts, generate = self.review(errors=errors)
            self.assertEqual(receipts, {})
            generate.assert_not_called()

    def test_missing_and_contradicted_action_still_need_repair(self):
        for verdict in ('missing', 'contradicted'):
            receipts, _ = self.review(decision(verdict=verdict, evidence_span_ids=[]))
            self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [ERROR])

    def test_evidence_ids_cannot_be_invented_global_audio_or_another_event(self):
        self.segment['opening_state'] = 'The jet hammers the mouth of the jug.'
        for evidence_ids in (
            [], ['invented_span_id'],
            ['check_1_card_1_opening_state_span_1'],
            ['check_1_card_1_sound_effects_span_1'],
            ['check_1_card_2_action_span_1'],
            [1], [''],
        ):
            with self.subTest(evidence_span_ids=evidence_ids):
                self.assertEqual(self.review(decision(evidence_span_ids=evidence_ids))[0], {})

    def test_short_visual_text_does_not_receive_an_evidence_id(self):
        self.segment['shots'][0]['action'] = 'Fills.'
        self.segment['shots'][0]['camera'] = 'Hold.'
        self.segment['shots'][0]['framing'] = 'Wide.'
        generate = Mock(return_value=decision('check_1_card_1_action_span_1'))
        receipts = review_missing_camera_actions(
            [ERROR], self.segment, assigned_beats=self.beats,
            source_events=self.events, generate=generate)
        self.assertEqual(receipts, {})
        generate.assert_not_called()

    def test_tiny_sentence_is_excluded_when_other_visual_spans_are_available(self):
        self.segment['shots'][0]['action'] = 'Fills. ' + PARAPHRASE
        packet = json.loads(self.review()[1].call_args.kwargs['prompt'])['check_1']
        self.assertEqual([span['text'] for span in packet['visual_spans'] if span['field'] == 'action'],
                         [PARAPHRASE])

    def test_another_source_event_card_cannot_clear_this_events_omission(self):
        self.segment["shots"] = [
            {"beat_ids": ["B1"], "action": "The courier looks at an empty jug.",
             "camera": "Hold on the courier."},
            {"beat_ids": ["B2"], "action": "The courier drinks from the jug.",
             "camera": "Follow the courier."},
        ]
        generate = Mock(return_value=decision('check_1_card_2_action_span_1'))
        receipts = review_missing_camera_actions(
            [ERROR], self.segment,
            assigned_beats=[{"beat_id": "B1", "source_event_ids": ["E1"]}],
            source_events=[
                {"event_id": "E1", "text": REQUIREMENT},
                {"event_id": "E2", "text": "The courier drinks from the jug."},
            ],
            generate=generate,
        )
        self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [ERROR])
        packet = json.loads(generate.call_args.kwargs["prompt"])["check_1"]
        self.assertEqual({span["card"] for span in packet["visual_spans"]}, {1})
        self.assertNotIn('check_1_card_2_action_span_1',
                         [span['span_id'] for span in packet['visual_spans']])
        self.assertNotIn("drinks from the jug", json.dumps(packet).casefold())

    def test_evidence_ids_are_check_scoped(self):
        self.segment['shots'].append({
            'beat_ids': ['B2'], 'action': PARAPHRASE, 'framing': 'Wide framing.',
            'camera': 'Hold on the action.',
        })
        b2_error = 'B2 shot action omits required source step: ' + REQUIREMENT
        b2_beat = {'beat_id': 'B2', 'source_event_ids': ['E1']}
        response = json.dumps({
            'check_1': {'verdict': 'preserved', 'evidence_span_ids': ['check_2_card_3_action_span_1']},
            'check_2': {'verdict': 'preserved', 'evidence_span_ids': ['check_2_card_3_action_span_1']},
        })
        generate = Mock(return_value=response)
        receipts = review_missing_camera_actions(
            [ERROR, b2_error], self.segment, assigned_beats=self.beats + [b2_beat],
            source_events=self.events, generate=generate)
        self.assertNotIn(ERROR, receipts)
        self.assertIn(b2_error, receipts)
        packet = json.loads(generate.call_args.kwargs['prompt'])['check_2']
        proof = next(span for span in packet['visual_spans']
                     if span['span_id'] == 'check_2_card_3_action_span_1')
        self.assertEqual(proof['text'], PARAPHRASE)

    def test_valid_check_survives_a_missing_or_invalid_sibling_decision(self):
        self.segment['shots'].append({
            'beat_ids': ['B2'], 'action': PARAPHRASE, 'framing': 'Wide framing.',
            'camera': 'Hold on the action.',
        })
        b2_error = 'B2 shot action omits required source step: ' + REQUIREMENT
        response = json.dumps({
            'check_1': {'verdict': 'preserved',
                        'evidence_span_ids': ['check_1_card_1_action_span_1']},
            'check_2': {'verdict': 'preserved', 'evidence_span_ids': ['not-issued']},
            # Unknown checks cannot approve any source error.
            'check_3': {'verdict': 'preserved',
                        'evidence_span_ids': ['check_2_card_3_action_span_1']},
        })
        generate = Mock(return_value=response)
        receipts = review_missing_camera_actions(
            [ERROR, b2_error], self.segment,
            assigned_beats=self.beats + [{'beat_id': 'B2', 'source_event_ids': ['E1']}],
            source_events=self.events, generate=generate,
        )
        self.assertIn(ERROR, receipts)
        self.assertNotIn(b2_error, receipts)
        self.assertEqual(clear_confirmed_coverage_errors(
            [ERROR, b2_error], self.segment, receipts), [b2_error])

    def test_quote_only_legacy_response_is_rejected(self):
        legacy = json.dumps({'check_1': {
            'verdict': 'preserved',
            'evidence': [{'card': 1, 'field': 'action', 'quote': PARAPHRASE}],
        }})
        receipts, _ = self.review(legacy)
        self.assertEqual(receipts, {})

    def test_repair_invalidates_a_review_for_changed_visuals(self):
        receipts, _ = self.review()
        self.segment['shots'][0]['action'] = 'The courier stares at an empty jug.'
        self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [ERROR])

    def test_repair_to_another_event_keeps_valid_evidence(self):
        receipts, _ = self.review()
        self.segment['shots'][1]['action'] = 'The courier sets down the jug.'
        self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [])

    def test_adjacent_connective_can_complete_the_same_physical_event(self):
        self.segment['shots'][0]['action'] = 'The courier positions the jug below the tap.'
        self.segment['shots'][1]['action'] = PARAPHRASE
        self.beats.append({'beat_id': 'B2', 'source_event_ids': []})
        receipts, generate = self.review(decision('check_1_card_2_action_span_1'))
        packet = json.loads(generate.call_args.kwargs['prompt'])['check_1']
        self.assertEqual([card['card'] for card in packet['visual_cards']], [1, 2])
        self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [])
        self.segment['shots'][1]['action'] = 'The courier watches the empty jug.'
        self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [ERROR])

    def test_adjacent_scope_does_not_cross_an_independent_source_event(self):
        self.beats.extend([
            {'beat_id': 'B2', 'source_event_ids': ['E2']},
            {'beat_id': 'B3', 'source_event_ids': []},
        ])
        self.segment['shots'].append({'beat_ids': ['B3'], 'action': PARAPHRASE})
        receipts, generate = self.review(decision('check_1_card_3_action_span_1'))
        packet = json.loads(generate.call_args.kwargs['prompt'])['check_1']
        self.assertEqual([card['card'] for card in packet['visual_cards']], [1])
        self.assertEqual(receipts, {})

    def test_adjacent_scope_is_bounded_and_reordering_expires_its_receipt(self):
        self.beats.extend({'beat_id': f'B{i}', 'source_event_ids': []} for i in range(2, 5))
        self.segment['shots'][1]['action'] = PARAPHRASE
        self.segment['shots'].extend([
            {'beat_ids': ['B3'], 'action': 'The courier watches the steady flow.'},
            {'beat_ids': ['B4'], 'action': 'The courier closes the tap.'},
        ])
        receipts, generate = self.review(decision('check_1_card_2_action_span_1'))
        packet = json.loads(generate.call_args.kwargs['prompt'])['check_1']
        self.assertEqual([card['card'] for card in packet['visual_cards']], [1, 2, 3])
        self.assertEqual(clear_confirmed_coverage_errors([ERROR, HARD_ERROR], self.segment, receipts), [HARD_ERROR])
        self.segment['shots'][0], self.segment['shots'][1] = self.segment['shots'][1], self.segment['shots'][0]
        self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [ERROR])

    def test_same_source_subdivision_is_eligible_but_other_event_is_not(self):
        self.beats.append({'beat_id': 'B2', 'source_event_ids': ['E1']})
        self.segment['shots'][1]['action'] = PARAPHRASE
        receipts, generate = self.review(decision('check_1_card_2_action_span_1'))
        self.assertEqual(clear_confirmed_coverage_errors([ERROR], self.segment, receipts), [])
        self.assertIn('same actor, object, physical effect, order', generate.call_args.kwargs['system_prompt'])
        self.assertIn('Do not assume omitted actions from a resulting state', generate.call_args.kwargs['system_prompt'])
        self.beats[1]['source_event_ids'] = ['E1', 'E2']
        receipts, _ = self.review(decision('check_1_card_2_action_span_1'))
        self.assertEqual(receipts, {})

    def test_recurring_occasions_cannot_borrow_each_others_action(self):
        self.beats[0]['_spaced_recurrence'] = {'source_event_id': 'E1', 'occurrence': 1}
        self.beats.append({'beat_id': 'B2', 'source_event_ids': ['E1'],
                           '_spaced_recurrence': {'source_event_id': 'E1', 'occurrence': 2}})
        self.segment['shots'][1]['action'] = PARAPHRASE
        receipts, generate = self.review(decision('check_1_card_2_action_span_1'))
        packet = json.loads(generate.call_args.kwargs['prompt'])['check_1']
        self.assertEqual([card['card'] for card in packet['visual_cards']], [1])
        self.assertEqual(receipts, {})

    def test_bad_review_and_unavailable_writer_do_not_approve(self):
        for response in ('broken JSON', '{}', '[]', decision(verdict='maybe'),
                         decision(evidence_span_ids=['unknown_span']),
                         decision(evidence_span_ids=[
                             'check_1_card_1_action_span_1', 'check_1_card_1_action_span_1'])):
            self.assertEqual(self.review(response)[0], {})
        generate = Mock(side_effect=RuntimeError('offline'))
        self.assertEqual(review_missing_camera_actions([ERROR], self.segment,
            assigned_beats=self.beats, source_events=self.events, generate=generate), {})

    def test_compound_omission_requires_every_action_subdecision(self):
        self.set_compound_coverage()
        response = action_decisions_response({
            'action_1': {'verdict': 'preserved',
                         'evidence_span_ids': ['check_1_card_1_action_span_1']},
            'action_2': {'verdict': 'missing', 'evidence_span_ids': []},
        })
        generate = Mock(return_value=response)
        receipts = review_missing_camera_actions(
            [COMPOUND_ERROR], self.segment, assigned_beats=self.beats,
            source_events=self.events, generate=generate,
        )
        self.assertEqual(clear_confirmed_coverage_errors(
            [COMPOUND_ERROR], self.segment, receipts), [COMPOUND_ERROR])
        check = json.loads(generate.call_args.kwargs['prompt'])['check_1']
        self.assertEqual(len(check['action_obligations']), 2)
        tie = check['action_obligations'][1]
        self.assertEqual(tie['action_focus']['predicate'], 'tie')
        self.assertIn('banner', tie['action_focus']['affected_entity_terms'])
        self.assertIn('wall', tie['action_focus']['affected_entity_terms'])
        self.assertIn('same actor, affected object, destination or endpoint',
                      generate.call_args.kwargs['system_prompt'])

    def test_single_legacy_decision_cannot_clear_a_compound_omission(self):
        self.set_compound_coverage()
        receipts, generate = self.review(
            decision('check_1_card_1_action_span_1'), errors=[COMPOUND_ERROR],
        )
        self.assertEqual(receipts, {})
        self.assertEqual(
            generate.call_args.kwargs['json_schema']['properties']['check_1']['required'],
            ['action_decisions'],
        )

    def test_wrong_attachment_endpoint_does_not_complete_tie_subcheck(self):
        self.set_compound_coverage()
        response = action_decisions_response({
            'action_1': {'verdict': 'preserved',
                         'evidence_span_ids': ['check_1_card_1_action_span_1']},
            # The visible knot is tied to the spool anchor rather than the wall.
            'action_2': {'verdict': 'missing', 'evidence_span_ids': []},
        })
        receipts, generate = self.review(response, errors=[COMPOUND_ERROR])
        self.assertEqual(receipts, {})
        tie = json.loads(generate.call_args.kwargs['prompt'])['check_1']['action_obligations'][1]
        self.assertEqual(tie['source_clause'], COMPOUND_REQUIREMENT)
        self.assertEqual(tie['action_focus']['predicate'], 'tie')
        self.assertEqual(tie['action_focus']['affected_entity_terms'], ['banner', 'wall'])

    def test_compound_synonyms_can_preserve_each_action(self):
        self.set_compound_coverage()
        self.segment['shots'][0]['action'] = "Priya feeds the ribbon through both banner eyelets."
        self.segment['shots'][1]['action'] = "Priya fastens the banner to a hook on the wall with the ribbon."
        response = action_decisions_response({
            'action_1': {'verdict': 'preserved',
                         'evidence_span_ids': ['check_1_card_1_action_span_1']},
            'action_2': {'verdict': 'preserved',
                         'evidence_span_ids': ['check_1_card_2_action_span_1']},
        })
        receipts, _ = self.review(response, errors=[COMPOUND_ERROR])
        self.assertEqual(clear_confirmed_coverage_errors(
            [COMPOUND_ERROR], self.segment, receipts), [])

    def test_compound_decisions_require_real_ids_for_every_action(self):
        self.set_compound_coverage()
        for second_ids in ([], ['invented_span_id'], ['check_2_card_1_action_span_1']):
            with self.subTest(second_ids=second_ids):
                response = action_decisions_response({
                    'action_1': {'verdict': 'preserved',
                                 'evidence_span_ids': ['check_1_card_1_action_span_1']},
                    'action_2': {'verdict': 'preserved', 'evidence_span_ids': second_ids},
                })
                receipts, _ = self.review(response, errors=[COMPOUND_ERROR])
                self.assertEqual(clear_confirmed_coverage_errors(
                    [COMPOUND_ERROR], self.segment, receipts), [COMPOUND_ERROR])

    def test_compound_decisions_reject_unknown_action_ids(self):
        self.set_compound_coverage()
        response = action_decisions_response({
            'action_1': {'verdict': 'preserved',
                         'evidence_span_ids': ['check_1_card_1_action_span_1']},
            'action_2': {'verdict': 'preserved',
                         'evidence_span_ids': ['check_1_card_2_action_span_1']},
            'action_3': {'verdict': 'preserved',
                         'evidence_span_ids': ['check_1_card_2_action_span_1']},
        })
        receipts, _ = self.review(response, errors=[COMPOUND_ERROR])
        self.assertEqual(clear_confirmed_coverage_errors(
            [COMPOUND_ERROR], self.segment, receipts), [COMPOUND_ERROR])

    def test_compound_receipt_expires_when_either_visual_action_changes(self):
        self.set_compound_coverage()
        response = action_decisions_response({
            'action_1': {'verdict': 'preserved',
                         'evidence_span_ids': ['check_1_card_1_action_span_1']},
            'action_2': {'verdict': 'preserved',
                         'evidence_span_ids': ['check_1_card_2_action_span_1']},
        })
        receipts, _ = self.review(response, errors=[COMPOUND_ERROR])
        self.assertEqual(clear_confirmed_coverage_errors(
            [COMPOUND_ERROR], self.segment, receipts), [])
        self.segment['shots'][1]['action'] = 'Priya tightens a knot around the spool anchor.'
        self.assertEqual(clear_confirmed_coverage_errors(
            [COMPOUND_ERROR], self.segment, receipts), [COMPOUND_ERROR])

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
                check = json.loads(kwargs['prompt'])['check_1']
                span_id = next(span['span_id'] for span in check['visual_spans']
                               if span['text'] == PARAPHRASE)
                return decision(span_id)
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
                return decision(verdict='missing', evidence_span_ids=[])
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
