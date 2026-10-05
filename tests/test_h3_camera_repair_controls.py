"""Camera repairs get actionable diagnostics and isolated experiment controls."""
from contextvars import Context
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))

from promptbench.experiments import experiment_context
from services import llm_service
from services.h3_camera_repair import camera_repair_advice
from services.h3_story_ledger import _camera_repair_feedback, segment_violations
from services.llm_sampling import caller_sampling, caller_sampling_enabled


class SamplingControlsTests(unittest.TestCase):
    def setUp(self):
        self.saved = llm_service._model_id, llm_service._provider
        llm_service._model_id = 'JonathanColetti/Qwen3.8-27B-Uncensored-GGUF'
        llm_service._provider = 'local'

    def tearDown(self):
        llm_service._model_id, llm_service._provider = self.saved

    def sampling(self):
        payload = {}
        result = llm_service._apply_model_defaults(0.08, 0.78, payload, enable_thinking=False)
        return result, payload

    def test_normal_calls_keep_model_tuned_sampling(self):
        result, payload = self.sampling()
        self.assertEqual(result, (0.7, 0.8))
        self.assertEqual(payload['top_k'], 20)

    def test_repair_context_honors_caller_temperature_and_preserves_guard_defaults(self):
        with caller_sampling(True):
            result, payload = self.sampling()
            self.assertEqual(result, (0.08, 0.78))
            self.assertEqual(payload['top_k'], 20)
            self.assertEqual(payload['repeat_penalty'] if 'repeat_penalty' in payload else 1.0, 1.0)
        self.assertEqual(self.sampling()[0], (0.7, 0.8))

    def test_nested_context_and_exception_do_not_leak_override(self):
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with caller_sampling(True):
                with caller_sampling(False):
                    self.assertEqual(self.sampling()[0], (0.7, 0.8))
                self.assertTrue(caller_sampling_enabled())
                raise RuntimeError('failed')
        self.assertFalse(caller_sampling_enabled())
        self.assertEqual(self.sampling()[0], (0.7, 0.8))

    def test_unrelated_request_context_keeps_model_sampling(self):
        with caller_sampling(True):
            self.assertEqual(Context().run(self.sampling)[0], (0.7, 0.8))


class CameraAdviceTests(unittest.TestCase):
    def advice(self, generate):
        return camera_repair_advice(generate, source_prompt='Assigned event: Nora shuts the door.',
                                   rejected='Nora looks at the open door.',
                                   feedback=['event_cards.event_1: show the door closing'],
                                   image_paths=['opening-frame.png'])

    def test_production_baseline_makes_no_advice_request(self):
        generate = Mock()
        with experiment_context('baseline'):
            self.assertEqual(self.advice(generate), '')
        generate.assert_not_called()

    def test_reasoning_advice_is_unconstrained_and_cannot_approve_a_plan(self):
        generate = Mock(return_value='Show Nora swinging the door shut.')
        with experiment_context('camera_repair_reasoning'):
            self.assertIn('swinging', self.advice(generate))
        request = generate.call_args.kwargs
        self.assertTrue(request['enable_thinking'])
        self.assertNotIn('json_schema', request)
        self.assertEqual(request['thinking_budget'], 2048)
        self.assertEqual(request['max_new_tokens'], 768)
        self.assertEqual(request['image_paths'], ['opening-frame.png'])
        self.assertIn('cannot authorize a change', request['system_prompt'])

    def test_optional_advice_failure_does_not_prevent_normal_repair(self):
        with experiment_context('camera_repair_reasoning'):
            self.assertEqual(self.advice(Mock(side_effect=ValueError('bad response'))), '')

    def test_advice_cancellation_propagates(self):
        with experiment_context('camera_repair_reasoning'), self.assertRaises(InterruptedError):
            self.advice(Mock(side_effect=InterruptedError('cancelled')))

    def test_advice_length_is_bounded(self):
        with experiment_context('camera_repair_reasoning'):
            self.assertEqual(len(self.advice(Mock(return_value='x' * 5000))), 3000)


class CameraDiagnosticTests(unittest.TestCase):
    def test_specific_missing_action_replaces_whole_paragraph_feedback(self):
        source = 'Nora enters, walks, lifts and places several objects. ' * 15
        error = 'B2 shot action omits required source step: ' + source
        beat = {'beat_id': 'B2', 'source_event_ids': ['E2']}
        details = {error: ['action_9 (close; actor Nora; object blue door): Nora closes the blue door']}
        feedback = _camera_repair_feedback([error], [beat], coverage_feedback=details)
        self.assertTrue(feedback[0].startswith('event_cards.event_1:'))
        self.assertIn('Nora closes the blue door', feedback[0])
        self.assertNotIn(source, feedback[0])

    def test_unknown_review_keeps_source_requirement_and_hard_failures(self):
        error = 'B1 shot action omits required source step: Nora closes the blue door'
        hard_error = 'shot 1 previews later source event E9 through unassigned physical action: open'
        feedback = _camera_repair_feedback([error, hard_error], [{'beat_id': 'B1'}], coverage_feedback={})
        self.assertIn('Nora closes the blue door', feedback[0])
        self.assertEqual(feedback[1], hard_error)

    def test_long_source_reports_the_missing_clause_instead_of_the_entire_event(self):
        from services.h3_story_ledger import extract_source_events
        source = ('Mara carries the brass tray past ' +
                  'plain wooden benches arranged along the plaster workshop wall with warm afternoon light ' * 7 +
                  '. Then Mara sets the brass tray on the table. Then Mara closes the blue door.')
        self.assertGreater(len(source), 500)
        prompt = '[0s-12s] ' + source + ' [12s-16s] Mara watches the distant courtyard.'
        events = extract_source_events(prompt)
        self.assertEqual(len(events), 2)
        beats = [{'beat_id': 'B1', 'source_event_ids': ['E1'], 'description': events[0]['text'],
                  'dialogue_ids': []}]
        segment = {
            'segment': 1, 'semantic_actions': True,
            'shots': [{'shot': 1, 'beat_ids': ['B1'], 'start_seconds': 0, 'end_seconds': 12,
                       'transition': 'opening composition', 'camera': '', 'framing': '',
                       'action': 'Mara carries the brass tray across the workshop and sets the brass tray on the table.',
                       'sound_effects': ''}],
            'closing_state': 'The brass tray rests on the table.',
        }
        errors = segment_violations(prompt, segment, segment_number=1, duration=12.0,
                                    assigned_beats=beats, dialogue_catalog=[])
        omissions = [error for error in errors if 'shot action omits required source step:' in error]
        door_errors = [error for error in omissions if 'Mara closes the blue door' in error]
        self.assertTrue(door_errors, errors)
        self.assertTrue(all('brass tray' not in error for error in door_errors), door_errors)

    def test_repair_rechecks_omissions_when_a_hard_preview_failure_remains(self):
        from test_h3_fidelity_retries import H3FidelityRetryTests
        omission = 'B2 shot action omits required source step: Lee opens the door'
        preview = 'shot 1 previews later source event E9 through unassigned physical action: break'
        with patch('services.h3_camera_fidelity.review_missing_camera_actions', return_value={}) as review:
            result, _ = H3FidelityRetryTests().run_camera(1, [[omission, preview], [omission, preview]])
        self.assertGreaterEqual(review.call_count, 3)  # first window, second draft, then repaired draft
        self.assertEqual(review.call_args.args[0], [omission])
        self.assertIn('repair_feedback', review.call_args.kwargs)
        budgets = {}
        for call in review.call_args_list:
            window = call.args[1]['segment']
            budget = call.kwargs['review_budget']
            self.assertEqual(budget.max_requests, 6)
            if window in budgets:
                self.assertIs(budget, budgets[window])
            else:
                budgets[window] = budget
        self.assertEqual(len(budgets), 2)
        self.assertIsNot(budgets[1], budgets[2])
        self.assertTrue(any(preview in item for item in result['planning_diagnostics']))


if __name__ == '__main__':
    unittest.main()
