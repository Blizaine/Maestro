"""Exercise real API/worker functions with a fake writer and diffusion engine."""
import ast
import asyncio
from copy import deepcopy
from functools import wraps
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from services import llm_service, studio_enhancement as enhancement
from services.job_lifecycle import finish_job, generation_slot, is_cancel_requested, request_cancel, snapshot_job, try_start, update_job

TREE = ast.parse((ROOT / 'app/launch.py').read_text(encoding='utf-8'))


def load(name, namespace):
    node = deepcopy(next(n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name))
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, 'app/launch.py', 'exec'), namespace)
    return namespace[name]


class Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return deepcopy(self.body)


class HttpError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class EnhancedJobWiringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive = enhancement.StudioJobArchive(self.temp.name)
        self.engine = SimpleNamespace(wan_model=None, offloadobj=None, release_model=Mock(),
            get_model_def=lambda _: {'architecture': 'minimax_h3', 'fps': 24, 'frames_maximum': 345})
        self.params = {'model_type': 'minimax_h3_fused_turbo', 'prompt': 'A mountain duel.',
                       'video_length': 345, 'sliding_window_size': 345, 'workspace': 'captured'}
        self.job = {'id': 'testjob', 'status': 'queued', 'params': deepcopy(self.params),
                    'enhancement': enhancement.new_enhancement(self.params, {'llm_model_id': 'writer-a'}),
                    'workspace': 'captured', 'out_dir': 'captured-output'}
        self.jobs = {self.job['id']: self.job}
        async def prepare(params, prepare_only):
            return {'params': deepcopy(params), 'workspace': 'captured'}
        self.prepare = AsyncMock(side_effect=prepare)
        self.writer = AsyncMock(return_value={'enhanced': 'Finished native prompt.'})
        self.ns = dict(asyncio=asyncio, deepcopy=deepcopy, wraps=wraps, Request=Request, HTTPException=HttpError,
            uuid=uuid, time=time, threading=threading, wgp=self.engine, update_job=update_job,
            is_cancel_requested=is_cancel_requested, snapshot_job=snapshot_job,
            _studio_job_archive=self.archive, _jobs=self.jobs, _studio_submission_lock=threading.RLock(),
            _gen_lock=threading.Lock(), enhancement_context=enhancement.enhancement_context,
            prepare_enhanced_job=enhancement.prepare_enhanced_job, public_enhancement=enhancement.public_enhancement,
            new_enhancement=enhancement.new_enhancement, _enhancement_settings_snapshot=lambda: {'llm_model_id': 'writer-a'},
            _llm_enhance_prompt_payload=self.writer, _prepare_generation_submission=self.prepare,
            _get_active_workspace=lambda: 'captured', _workspace_dir=lambda name: 'captured-output',
            _run_generation=Mock())
        self.run_enhancement = load('_prepare_job_enhancement', self.ns)
        for method in ('is_loaded', 'unload_model'):
            mock = patch.object(llm_service, method, return_value=(method == 'is_loaded'))
            mock.start()
            self.addCleanup(mock.stop)

    def test_completed_draft_is_checkpointed_and_generation_retry_skips_writer(self):
        self.assertTrue(try_start(self.job))
        self.run_enhancement(self.job)
        self.assertEqual(self.archive.recover()['testjob']['enhancement']['state'], 'complete')
        self.assertEqual(self.job['params']['prompt'], 'Finished native prompt.')
        self.writer.assert_awaited_once()
        self.assertEqual(self.job['enhancement']['original_prompt'], 'A mountain duel.')
        self.job['params']['prompt'] = 'Unrelated later edit'
        self.run_enhancement(self.job)
        self.writer.assert_awaited_once()
        self.assertEqual(self.job['params']['prompt'], 'Finished native prompt.')
        llm_service.unload_model.assert_called_once()

    def test_public_unload_cannot_interrupt_a_writer_that_just_acquired_the_slot(self):
        self.ns['_guard_interactive_llm_against_generation'] = Mock()
        unload = load('llm_unload', self.ns)
        self.ns['_gen_lock'].acquire()
        with self.assertRaises(HttpError) as caught:
            unload()
        self.assertEqual(caught.exception.status_code, 409)
        llm_service.unload_model.assert_not_called()
        self.ns['_gen_lock'].release()
        llm_service.unload_model.side_effect = RuntimeError('unload failed')
        with self.assertRaises(RuntimeError):
            unload()
        self.assertFalse(self.ns['_gen_lock'].locked())

    def test_ensure_writer_passes_frozen_device_and_endpoint_to_reuse_check(self):
        self.engine.server_config = {'services': {'llm_model_id': 'later-writer',
            'llm_device': 'cpu', 'llm_remote_url': 'http://later'}}
        self.ns.update(enhancement_settings=enhancement.current_settings,
                       _DEFAULT_LLM_REPO='default', _llm_default_device=lambda: 'cpu')
        ensure = load('_ensure_llm_loaded', self.ns)
        frozen = {'llm_model_id': 'captured-writer', 'llm_device': 'cuda',
                  'llm_provider': 'remote', 'llm_remote_url': 'http://captured'}
        with patch.object(llm_service, 'load_model') as loader:
            with enhancement.enhancement_context(frozen, lambda: False):
                ensure()
        self.assertEqual(loader.call_args.kwargs['device'], 'cuda')
        self.assertEqual(loader.call_args.kwargs['remote_url'], 'http://captured')
        self.assertEqual(loader.call_args.kwargs['model_id'], 'captured-writer')

    def test_warning_checkpoints_review_draft_and_prevents_diffusion(self):
        self.writer.return_value['warnings'] = ['The writer used a source fallback.']
        with self.assertRaisesRegex(ValueError, 'needs review'):
            self.run_enhancement(self.job)
        saved = self.archive.recover()['testjob']['enhancement']
        self.assertEqual(saved['state'], 'review')
        self.assertEqual(saved['prepared']['params']['prompt'], 'Finished native prompt.')
        self.assertEqual(self.job['params']['prompt'], self.params['prompt'])

    def test_writer_failure_is_retained_and_never_prepares_generation(self):
        self.writer.side_effect = RuntimeError('Writer unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Writer unavailable'):
            self.run_enhancement(self.job)
        self.prepare.assert_not_awaited()
        self.assertEqual(self.archive.recover()['testjob']['enhancement']['state'], 'failed')
        llm_service.unload_model.assert_called_once()

    def test_cancelled_writer_does_not_start_diffusion_or_release_slot_early(self):
        entered, release = threading.Event(), threading.Event()
        async def writer(payload):
            entered.set()
            release.wait(3)
            return {'enhanced': 'Late response'}
        self.writer.side_effect = writer
        def worker():
            with generation_slot(self.ns['_gen_lock'], self.job) as acquired:
                if acquired and try_start(self.job):
                    try:
                        self.run_enhancement(self.job)
                    except Exception as error:
                        finish_job(self.job, 'failed', error=str(error))
        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(3))
        request_cancel(self.job, job_id='testjob', active_states={})
        self.assertTrue(self.ns['_gen_lock'].locked())
        release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.ns['_gen_lock'].locked())
        self.assertEqual(self.job['status'], 'cancelled')
        self.prepare.assert_not_awaited()

    def test_held_submission_snapshots_inputs_without_starting_writer_or_worker(self):
        enqueue = load('_enqueue_deferred_generation_preparation', self.ns)
        body = {**self.params, '_enhance_on_generation': True, '_queue_mode': 'held'}
        with patch.object(threading, 'Thread') as thread:
            response = enqueue(body)
        thread.assert_not_called()
        self.writer.assert_not_awaited()
        body['prompt'] = 'Changed after submission'
        saved = self.archive.recover()[response['job_id']]
        self.assertEqual(saved['status'], 'held')
        self.assertEqual(saved['enhancement']['original_prompt'], 'A mountain duel.')
        self.assertNotIn('_deferred_generation_prepare', saved['enhancement']['original_params'])

    def test_submission_retry_does_not_create_second_job(self):
        load('_enqueue_deferred_generation_preparation', self.ns)
        self.ns['_generation_request_uses_serial_auto_planner'] = lambda body: True
        generate = load('generate', self.ns)
        request = Request({**self.params, '_enhance_on_generation': True, '_client_submission_id': 'same-request'})
        with patch.object(threading, 'Thread') as thread:
            first = asyncio.run(generate(request))
            second = asyncio.run(generate(request))
        self.assertEqual(first['job_id'], second['job_id'])
        self.assertEqual(thread.call_count, 1)

    def test_retry_reuses_draft_refreshes_only_on_request_and_accepts_review_explicitly(self):
        retry = load('retry_enhanced_job', self.ns)
        for action, original_state, expected in [('retry', 'complete', 'complete'),
                ('refresh', 'complete', 'pending'), ('accept_draft', 'review', 'complete')]:
            with self.subTest(action=action):
                self.job.update(status='failed', retry_job_id=None)
                self.job['enhancement'].update(state=original_state, prepared={'params': {'prompt': 'Saved draft'}})
                with patch.object(threading, 'Thread'):
                    result = asyncio.run(retry('testjob', Request({'action': action})))
                record = self.jobs[result['job_id']]['enhancement']
                self.assertEqual(record['state'], expected)
                self.assertEqual(record['original_prompt'], 'A mountain duel.')
                self.writer.assert_not_awaited()

    def test_generate_as_written_uses_manual_validation_without_enhancement(self):
        retry = load('retry_enhanced_job', self.ns)
        self.job['status'] = 'failed'
        with patch.object(threading, 'Thread'):
            result = asyncio.run(retry('testjob', Request({'action': 'as_written'})))
        record = self.jobs[result['job_id']]['enhancement']
        self.assertEqual(record['prepared']['params']['prompt'], 'A mountain duel.')
        self.assertEqual(record['prepared']['params']['minimax_h3_sequence_prompt_mode'], 'manual')
        self.writer.assert_not_awaited()

    def test_queued_review_retry_retains_checkpoint_and_prepares_full_job(self):
        retry = load('retry_enhanced_job', self.ns)
        self.job['status'] = 'failed'
        self.job['enhancement']['original_params'].update(
            video_length=1962, minimax_h3_multi_window=True)
        saved_plan = {'camera_checkpoint': {'version': 1},
            'windows': [{'prompt': f'Saved window {i}'} for i in range(1, 7)],
            'planning_warnings': ["Window 4's camera plan needs review."]}
        self.job['enhancement'].update(state='review', prepared={'params': self.params,
            'h3_window_plan': deepcopy(saved_plan)}, warnings=saved_plan['planning_warnings'])
        async def repair(body, prepare_only):
            self.assertEqual(body['_h3_retry_plan'], saved_plan)
            prompts = [w['prompt'] for w in saved_plan['windows']]
            prompts[3] = 'Repaired window 4'
            return {'params': {**body, 'h3_window_prompts': prompts},
                    'h3_window_plan': {'window_prompts': prompts, 'planned_by': 'llm', 'planning_warnings': []}}
        self.prepare.side_effect = repair
        with patch.object(threading, 'Thread'):
            result = asyncio.run(retry('testjob', Request({'action': 'retry'})))
        accepted = self.jobs[result['job_id']]
        self.assertTrue(try_start(accepted))
        self.run_enhancement(accepted)
        self.assertEqual(accepted['enhancement']['state'], 'complete')
        self.assertEqual(len(accepted['params']['h3_window_prompts']), 6)
        self.assertEqual(accepted['params']['h3_window_prompts'][3], 'Repaired window 4')
        self.assertEqual(self.job['enhancement']['prepared']['h3_window_plan'], saved_plan)
        self.writer.assert_not_awaited()

    def test_accept_reviewed_multiwindow_draft_reaches_worker_without_enhancing_again(self):
        retry = load('retry_enhanced_job', self.ns)
        prepared_params = {**self.params, 'prompt': 'Reviewed native script',
            'h3_window_prompts': ['Exact first window', 'Exact second window'],
            'h3_window_plan_signature': 'saved-signature'}
        self.job.update(status='failed')
        self.job['enhancement'].update(state='review', prepared={
            'params': prepared_params, 'enhancement_review_required': True,
            'h3_window_plan': {'planning_warnings': ['Review the camera fallback.']}},
            warnings=['Review the camera fallback.'])
        self.archive.save(self.job)
        with patch.object(threading, 'Thread') as thread:
            response = asyncio.run(retry('testjob', Request({'action': 'accept_draft'})))
        accepted = self.jobs[response['job_id']]
        self.assertEqual(response['status'], 'queued')
        self.assertEqual(accepted['enhancement']['state'], 'complete')
        thread.return_value.start.assert_called_once()
        self.assertTrue(try_start(accepted))
        self.run_enhancement(accepted)
        self.assertEqual({key: accepted['params'][key] for key in prepared_params}, prepared_params)
        self.assertEqual(accepted['enhancement']['state'], 'complete')
        self.writer.assert_not_awaited()
        self.prepare.assert_not_awaited()

    def test_dismissed_cancelled_job_cannot_be_resurrected_by_worker_teardown(self):
        self.job.update(status='cancelled', dismissed=True)
        self.archive.save(self.job)
        self.job['enhancement'].update(state='failed', error='Cancelled')
        self.archive.save(self.job)
        self.assertEqual(self.archive.recover(), {})

    def test_browser_disconnect_keeps_gpu_owned_until_interactive_writer_finishes(self):
        self.jobs.clear()
        load('_guard_interactive_llm_against_generation', self.ns)
        decorator = load('_interactive_enhancement_slot', self.ns)
        entered, release = threading.Event(), threading.Event()
        async def endpoint(request):
            entered.set()
            release.wait(3)
            return {'enhanced': 'Draft'}
        async def run():
            task = asyncio.create_task(decorator(endpoint)(Request({})))
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            task.cancel()
            await asyncio.sleep(0.05)
            self.assertTrue(self.ns['_gen_lock'].locked())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(self.ns['_gen_lock'].locked())
        asyncio.run(run())


class StreamingCancellationTests(unittest.TestCase):
    def test_cancellation_interrupts_prefill_without_waiting_for_first_token(self):
        import requests
        release, cancelled, ready = threading.Event(), threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.flush()
                ready.set()
                release.wait(4)
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        errors = []
        def read():
            try:
                with enhancement.enhancement_context({}, cancelled.is_set):
                    response = requests.get(f'http://127.0.0.1:{server.server_port}', stream=True, timeout=5)
                    try:
                        list(enhancement.cancellable_lines(response))
                    finally:
                        enhancement.close_response(response)
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=read)
        try:
            worker.start()
            self.assertTrue(ready.wait(2))
            cancelled.set()
            worker.join(2)
            self.assertFalse(worker.is_alive(), 'Cancellation waited for the first token')
            self.assertTrue(any(isinstance(error, InterruptedError) for error in errors), errors)
        finally:
            release.set()
            worker.join(5)
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
