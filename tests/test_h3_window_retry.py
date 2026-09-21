"""A review retry repairs local camera writing, never the whole story."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import llm_service
from services.h3_plan_retry import H3PlanRetryError
from services.h3_window_planner import plan_h3_sliding_windows
from services.h3_sequence_planner import plan_h3_reference_sequence


ACTIONS = ['The courier opens the gate.', 'The courier walks to the bench.',
           'The courier puts a red bag on the bench.', 'The courier opens the red bag.',
           'The courier takes a book out of the red bag.', 'The courier reads the book.']
SOURCE = 'No dialogue.\n' + '\n'.join(f'[{i * 4}s-{(i + 1) * 4}s] {action}' for i, action in enumerate(ACTIONS))


class WindowRetryTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.fail = {4}
        self.kwargs = dict(model_type='minimax_h3_fused_turbo', resolution='864x480',
                           total_frames=576, window_frames=96, overlap_frames=0, fps=24,
                           camera_coverage='continuous', planning_style='faithful')
        self.writer = patch.object(llm_service, 'generate', side_effect=self.generate)
        self.writer.start()
        self.addCleanup(self.writer.stop)

    def generate(self, **kwargs):
        props = kwargs['json_schema']['properties']
        if 'setting_continuity' in props:
            self.calls.append('story')
            return json.dumps({'character_appearance': 'A courier.', 'setting_continuity': 'A garden.',
                               'visual_continuity': 'Daylight', 'editing_style': 'Continuous take'})
        if 'segment' not in props:
            raise AssertionError('Unexpected writer pass: ' + str(props.keys()))
        n = props['segment']['minimum']
        self.calls.append(n)
        if n in self.fail:
            raise ValueError('Deliberately unavailable camera draft')
        action = ACTIONS[n - 1]
        return json.dumps({'segment': n, 'coverage': 'Continuous', 'closing_state': action,
            'event_cards': {'event_1': {'phases': [{'action': action, 'camera': 'Hold steady',
                'framing': 'Wide', 'transition': 'Continue', 'sound_effects': 'Wind'}]}}})

    def plan(self, **extra):
        return plan_h3_sliding_windows(SOURCE, **{**self.kwargs, **extra})

    def test_six_windows_retry_only_four_and_keep_all_other_prompts_exact(self):
        original = self.plan()
        self.assertEqual(original['retryable_windows'], [4])
        self.assertEqual(self.calls, ['story', 1, 2, 3, 4, 5, 6])
        frozen = deepcopy(original)
        self.calls.clear()
        self.fail.clear()
        # Disk round-trip proves the checkpoint survives an app restart.
        repaired = self.plan(retry_plan=json.loads(json.dumps(original)))
        self.assertEqual(self.calls, [4])
        self.assertEqual(repaired['retryable_windows'], [])
        self.assertEqual(repaired['planning_warnings'], [])
        self.assertEqual(repaired['retried_windows'], [4])
        self.assertEqual(repaired['story_ledger'], original['story_ledger'])
        self.assertEqual(original, frozen)
        for index in range(6):
            if index != 3:
                self.assertEqual(repaired['windows'][index], original['windows'][index])
            self.assertEqual(repaired['windows'][index]['opening_state'], original['windows'][index]['opening_state'])
            self.assertEqual(repaired['windows'][index]['closing_state'], original['windows'][index]['closing_state'])

    def test_failed_retry_is_still_local_and_can_be_retried_again(self):
        original = self.plan()
        self.calls.clear()
        again = self.plan(retry_plan=original)
        self.assertEqual(self.calls, [4])
        self.assertEqual(again['retryable_windows'], [4])
        self.fail.clear()
        self.calls.clear()
        final = self.plan(retry_plan=again)
        self.assertEqual(self.calls, [4])
        self.assertEqual(final['planning_warnings'], [])

    def test_multiple_failed_windows_preserve_intervening_passed_windows(self):
        self.fail = {2, 5}
        original = self.plan()
        self.fail.clear()
        self.calls.clear()
        repaired = self.plan(retry_plan=original)
        self.assertEqual(self.calls, [2, 5])
        self.assertEqual(repaired['retryable_windows'], [])
        for index in [0, 2, 3, 5]:
            self.assertEqual(repaired['windows'][index], original['windows'][index])

    def test_changed_inputs_reject_before_writer_without_replacing_saved_draft(self):
        original = self.plan()
        self.calls.clear()
        for change in ({'resolution': '1280x704'}, {'total_frames': 480}, {'image_paths': ['changed.png']}, {'nsfw': True}):
            with self.subTest(change=change), self.assertRaises(H3PlanRetryError):
                self.plan(retry_plan=original, **change)
        self.assertEqual(self.calls, [])

    def test_reference_sequences_also_preserve_passed_prompts(self):
        kwargs = dict(model_type='minimax_h3_ref2va_fused_turbo', resolution='864x480',
                      total_frames=576, references=[], max_clip_frames=96, overlap_frames=0,
                      native_continuation=True, camera_coverage='continuous')
        original = plan_h3_reference_sequence(SOURCE, **kwargs)
        self.assertEqual(original['retryable_windows'], [4])
        self.fail.clear()
        self.calls.clear()
        repaired = plan_h3_reference_sequence(SOURCE, **kwargs, retry_plan=original)
        self.assertEqual(self.calls, [4])
        self.assertEqual(repaired['planning_warnings'], [])
        for index in [0, 1, 2, 4, 5]:
            self.assertEqual(repaired['windows'][index], original['windows'][index])

    def test_compilation_failure_cannot_replace_accepted_windows(self):
        original = self.plan()
        self.fail.clear()
        with patch('services.h3_window_planner.compile_h3_window_prompts', side_effect=ValueError('bad compiler')):
            with self.assertRaisesRegex(ValueError, 'bad compiler'):
                self.plan(retry_plan=original)

    def test_cancelled_retry_preserves_saved_draft(self):
        original = self.plan()
        frozen = deepcopy(original)
        with patch.object(llm_service, 'generate', side_effect=InterruptedError('Cancelled')):
            with self.assertRaises(InterruptedError):
                self.plan(retry_plan=original)
        self.assertEqual(original, frozen)

    def test_full_refresh_intentionally_rewrites_all_windows(self):
        self.plan()
        self.calls.clear()
        self.fail.clear()
        self.plan()
        self.assertEqual(self.calls, ['story', 1, 2, 3, 4, 5, 6])


if __name__ == '__main__':
    unittest.main()
