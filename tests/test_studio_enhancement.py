"""The queued writing contract is independent of the browser and GPU engine."""
import asyncio
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.studio_enhancement import (
    StudioJobArchive, captured_settings, current_settings, enhancement_context,
    enhancement_request, new_enhancement, prepare_enhanced_job, public_enhancement,
)


class QueuedEnhancementTests(unittest.TestCase):
    def setUp(self):
        self.model = {'architecture': 'minimax_h3', 'fps': 24, 'frames_maximum': 345}
        self.params = {'prompt': 'A mountain duel. No dialogue.', 'model_type': 'minimax_h3_fused_turbo',
            'video_length': 345, 'sliding_window_size': 345, 'workspace': 'testing',
            'activated_loras': ['action.safetensors'], 'image_start': 'first.png',
            'image_end': 'last.png', 'generation_mode': 'video'}
        self.enhance = AsyncMock(return_value={'enhanced': 'integrated_multimodal_description: A duel.'})
        async def prepare(body, prepare_only):
            self.assertTrue(prepare_only)
            return {'params': body, 'workspace': body['workspace']}
        self.prepare = AsyncMock(side_effect=prepare)

    def run_prepare(self):
        return asyncio.run(prepare_enhanced_job(self.params, self.model, self.enhance, self.prepare))

    def test_single_window_enhances_then_normalizes_without_mutating_source(self):
        original = deepcopy(self.params)
        result = self.run_prepare()
        self.assertEqual(self.params, original)
        self.assertEqual(result['params']['prompt'], self.enhance.return_value['enhanced'])
        payload = self.enhance.call_args.args[0]
        self.assertEqual(payload['duration_seconds'], 14.375)
        self.assertEqual(payload['image_paths'], ['first.png', 'last.png'])
        self.assertEqual(payload['planning_style'], 'adaptive')
        self.assertEqual(result['params']['multi_prompts_gen_type'], 2)

    def test_multi_window_uses_native_planner_once_not_single_prompt_rewrite(self):
        self.params.update(video_length=672, minimax_h3_multi_window=True)
        self.run_prepare()
        self.enhance.assert_not_called()
        request = self.prepare.call_args.args[0]
        self.assertEqual(request['prompt'], self.params['prompt'])
        self.assertEqual(request['minimax_h3_sequence_prompt_mode'], 'adaptive')
        self.assertTrue(request['minimax_h3_window_storyboard'])

    def test_multi_window_ltx_retains_existing_adapter(self):
        self.params.update(video_length=600, ltx_multi_window=True)
        self.model.update(architecture='ltx2', multi_window_sequence_controls=True)
        self.run_prepare()
        self.enhance.assert_not_called()
        self.assertEqual(self.prepare.call_args.args[0]['ltx_window_prompt_mode'], 'auto')

    def test_review_retry_passes_saved_camera_plan_to_native_preparation(self):
        self.params.update(video_length=672, minimax_h3_multi_window=True)
        saved = {'h3_window_plan': {'camera_checkpoint': {'version': 1},
            'windows': [{'prompt': 'Passed first'}, {'prompt': 'Flagged second'}],
            'planning_warnings': ["Window 2's camera plan needs review."]}}
        frozen = deepcopy(saved)
        asyncio.run(prepare_enhanced_job(self.params, self.model, self.enhance, self.prepare,
                                       previous_prepared=saved))
        self.enhance.assert_not_awaited()
        self.assertEqual(self.prepare.call_args.args[0]['_h3_retry_plan'], saved['h3_window_plan'])
        self.assertEqual(saved, frozen)
        self.assertNotIn('_h3_retry_plan', self.params)

    def test_explicit_refresh_does_not_pass_the_previous_camera_plan(self):
        self.params.update(video_length=672, minimax_h3_multi_window=True)
        self.run_prepare()
        self.assertNotIn('_h3_retry_plan', self.prepare.call_args.args[0])

    def test_enhancement_failure_never_prepares_plain_generation(self):
        self.enhance.side_effect = RuntimeError('writer offline')
        with self.assertRaisesRegex(RuntimeError, 'writer offline'):
            self.run_prepare()
        self.prepare.assert_not_called()
        self.assertEqual(self.params['prompt'], 'A mountain duel. No dialogue.')

    def test_cancellation_after_response_prevents_generation_preparation(self):
        cancelled = False
        async def cancel_during_call(payload):
            nonlocal cancelled
            cancelled = True
            return {'enhanced': 'A late response'}
        self.enhance.side_effect = cancel_during_call
        with enhancement_context({}, lambda: cancelled):
            with self.assertRaises(InterruptedError):
                self.run_prepare()
            # Reset only for this test context's exit guard.
            cancelled = False
        self.prepare.assert_not_called()

    def test_draft_warnings_pause_unattended_generation_but_notes_do_not(self):
        self.prepare.side_effect = None
        for warnings, planned_by, expected in [([], 'llm', False), (['Source fallback'], 'llm', True),
                                               ([], 'deterministic_fallback', True)]:
            with self.subTest(warnings=warnings, planned_by=planned_by):
                self.prepare.return_value = {'params': self.params, 'h3_window_plan': {
                    'planned_by': planned_by, 'planning_warnings': warnings, 'planning_notes': ['Timing adjusted']}}
                self.assertEqual(bool(self.run_prepare().get('enhancement_review_required')), expected)

    def test_frozen_settings_assets_and_source_do_not_follow_later_edits(self):
        self.params['minimax_h3_references'] = [{'path': 'identity.png', 'role': 'Actor'}]
        record = new_enhancement(self.params, {'llm_model_id': 'writer-a', 'nsfw_mode': False, 'openai_api_key': 'secret'})
        self.params['minimax_h3_references'][0]['role'] = 'Different character'
        self.assertEqual(record['original_params']['minimax_h3_references'][0]['role'], 'Actor')
        self.assertNotIn('openai_api_key', record['settings'])
        with enhancement_context(record['settings'], lambda: False):
            settings = current_settings({'llm_model_id': 'writer-b', 'nsfw_mode': True, 'openai_api_key': 'current'})
            self.assertEqual(settings['llm_model_id'], 'writer-a')
            self.assertFalse(settings['nsfw_mode'])
            self.assertEqual(settings['openai_api_key'], 'current')
        self.assertNotIn('settings', public_enhancement(record))
        self.assertFalse(captured_settings({})['nsfw_mode'])

    def test_reference_roles_remain_distinct_from_exact_first_frame(self):
        self.model.update(architecture='minimax_h3_ref2va', omni_reference=True)
        self.params['minimax_h3_references'] = [{'type': 'image', 'path': 'actor.png', 'role': 'Actor', 'image_intent': 'identity'}]
        payload, sequence = enhancement_request(self.params, self.model)
        self.assertFalse(sequence)
        self.assertEqual(payload['image_paths'], ['actor.png'])
        self.assertIn('do not define the target scene', payload['reference_context'])
        self.assertNotIn('exact target frame', payload['reference_context'])

    def test_restart_keeps_completed_draft_and_holds_interrupted_work(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = StudioJobArchive(directory)
            record = new_enhancement(self.params, {})
            record.update(state='complete', prepared={'params': {'prompt': 'Finished draft'}})
            job = {'id': 'example1', 'status': 'running', 'params': self.params, 'enhancement': record}
            archive.save(job)
            restored = archive.recover()['example1']
            self.assertEqual(restored['status'], 'held')
            self.assertEqual(restored['enhancement']['prepared']['params']['prompt'], 'Finished draft')
            for status in ('held', 'completed', 'failed', 'cancelled'):
                job['status'] = status
                archive.save(job)
                self.assertEqual(archive.recover()['example1']['status'], status)
            archive.remove('example1')
            self.assertEqual(archive.recover(), {})

    def test_interrupted_enhancement_restarts_from_original_and_invalid_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = StudioJobArchive(directory)
            record = new_enhancement(self.params, {})
            record['state'] = 'enhancing'
            archive.save({'id': 'example', 'status': 'running', 'enhancement': record, 'params': self.params})
            Path(directory, 'broken.json').write_text('{bad', encoding='utf-8')
            restored = archive.recover()['example']
            self.assertEqual(restored['enhancement']['state'], 'pending')
            self.assertEqual(restored['params']['prompt'], self.params['prompt'])
            with self.assertRaises(ValueError):
                archive.remove('../escape')


if __name__ == '__main__':
    unittest.main()
