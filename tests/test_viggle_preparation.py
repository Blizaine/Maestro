"""Character replacement: ordered refs, native geometry, cache and job phases."""
import ast
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import numpy as np
from PIL import Image
from services import viggle_preparation as vp
from services import job_lifecycle as lifecycle
from models.minimax_h3 import viggle


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        cwd = os.getcwd()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, cwd)
        os.chdir(temporary.name)
        Path('uploads').mkdir()
        self.source = Path('uploads/control.avi').resolve()
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'MJPG'), 24, (70, 50))
        self.assertTrue(writer.isOpened())
        for index in range(24):
            pixels = np.full((50, 70, 3), (index * 8, 40, 80), dtype=np.uint8)
            writer.write(pixels)
        writer.release()
        self.reference = Path('uploads/person.png').resolve()
        Image.new('RGB', (64, 96), 'red').save(self.reference)
        self.options = vp.normalize_options({'reference_path': str(self.reference), 'character_name': 'Blaine'})
        self.body = {'video_guide': str(self.source), 'viggle_character': self.options, 'seed': 123,
                     'video_length': 124, 'model_type': 'viggle_animate'}
        self.calls = []

    def render(self, request, directory):
        self.calls.append(request)
        self.assertEqual(request['model_type'], 'flux2_klein_9b')
        self.assertEqual(request['image_refs'][1], str(self.reference))
        self.assertEqual(request['resolution'], '96x64')
        self.assertEqual(request['image_output_codec'], 'png')
        self.assertEqual(request['multi_prompts_gen_type'], 2)
        self.assertEqual(request['activated_loras'], [])
        self.assertNotIn('face_refiner', request)
        path = directory / 'rendered.png'
        with Image.open(request['image_refs'][0]) as source:
            source.save(path)
        return path

    def prepare(self, **callbacks):
        return vp.prepare(self.body, resolve_media=lambda value: value,
                          render=callbacks.get('render', self.render), update=lambda text: None,
                          aborted=callbacks.get('aborted', lambda: False))

    def test_source_first_reference_second_and_preserve_exact_unaligned_geometry(self):
        result = self.prepare()
        self.assertEqual((result['width'], result['height']), (70, 50))
        with Image.open(result['image_path']) as image:
            self.assertEqual(image.size, (70, 50))
            capture = cv2.VideoCapture(str(self.source))
            _, frame = capture.read()
            capture.release()
            np.testing.assert_array_equal(np.array(image), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.assertEqual(list(Path('uploads/viggle_preparation').iterdir()), [])
        self.assertTrue(self.source.exists())
        self.assertTrue(self.reference.exists())

    def test_prompt_and_frame_selection(self):
        self.options['appearance_prompt'] = 'A blue jacket.\nKeep the glasses.'
        self.options['frame_seconds'] = 0.5
        result = self.prepare()
        self.assertEqual(result['prompt'], vp.SWAP_PROMPT + '\n\nAppearance of the replacement character: A blue jacket.\nKeep the glasses.')
        with Image.open(result['image_path']) as picture:
            self.assertGreater(picture.getpixel((10, 10))[2], 50)
        self.assertEqual(result['frame_seconds'], 0.5)

    def test_cache_reuses_only_identical_inputs_and_regenerates_corruption(self):
        first = self.prepare()
        self.body['_viggle_prepared'] = first
        self.assertEqual(self.prepare(), first)
        self.assertEqual(len(self.calls), 1)
        for key, value in [('appearance_prompt', 'Red coat'), ('frame_seconds', 0.25), ('swap_prompt', 'Replace the subject only.')]:
            self.options[key] = value
            next_result = self.prepare()
            self.assertNotEqual(next_result['signature'], first['signature'])
            self.body['_viggle_prepared'] = next_result
            first = next_result
        self.body['seed'] = 456
        next_result = self.prepare()
        self.assertNotEqual(next_result['signature'], first['signature'])
        self.body['_viggle_prepared'] = next_result
        Path(next_result['image_path']).write_bytes(b'incomplete')
        repaired = self.prepare()
        with Image.open(repaired['image_path']) as picture:
            picture.verify()
        self.assertNotEqual(repaired['image_path'], next_result['image_path'])

    def test_reference_change_invalidates_preview_and_untrusted_path_not_reused(self):
        first = self.prepare()
        self.body['_viggle_prepared'] = first
        Image.new('RGB', (64, 96), 'blue').save(self.reference)
        second = self.prepare()
        self.assertNotEqual(first['signature'], second['signature'])
        self.body['_viggle_prepared'] = {**second, 'image_path': str(self.reference)}
        self.assertNotEqual(self.prepare()['image_path'], str(self.reference))

    def test_bad_dimensions_or_cancel_never_publish(self):
        def wrong_canvas(request, directory):
            path = directory / 'wrong.png'
            Image.new('RGB', (64, 64)).save(path)
            return path
        with self.assertRaisesRegex(ValueError, 'not passed to Viggle'):
            self.prepare(render=wrong_canvas)
        self.assertEqual(list(Path('uploads').glob('viggle_prepared_*.png')), [])
        cancelled = False
        def cancel_after_render(request, directory):
            nonlocal cancelled
            result = self.render(request, directory)
            cancelled = True
            return result
        with self.assertRaises(InterruptedError):
            self.prepare(render=cancel_after_render, aborted=lambda: cancelled)
        self.assertEqual(list(Path('uploads/viggle_preparation').iterdir()), [])
        self.assertEqual(list(Path('uploads').glob('viggle_prepared_*.png')), [])

    def test_invalid_requests_and_preview_boundary(self):
        for values in ({'reference_path': ''}, {'image_model': 'unrelated'}, {'frame_seconds': -1},
                       {'frame_seconds': float('nan')}, {'appearance_prompt': 'x' * 8001}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                vp.normalize_options({**self.options, **values})
        self.options['frame_seconds'] = 1
        with self.assertRaisesRegex(ValueError, 'outside the control video'):
            self.prepare()
        self.assertFalse(self.calls)

    def test_submission_allows_character_but_runtime_still_requires_prepared_image(self):
        viggle.normalize_settings(self.body, allow_preparation=True)
        with self.assertRaisesRegex(ValueError, 'edited'):
            viggle.normalize_settings(self.body)
        self.body['image_refs'] = [str(self.reference)]
        viggle.normalize_settings(self.body)
        self.assertEqual(self.body['num_inference_steps'], 3)

    def test_trim_does_not_offset_character_frame_or_invalidate_unchanged_preview(self):
        self.options['frame_seconds'] = 0.75
        self.body.update(_viggle_trim_start=0.5, _viggle_trim_end=1)
        viggle.normalize_settings(self.body, allow_preparation=True)
        result = self.prepare()
        self.assertEqual(result['frame_seconds'], 0.75)
        with Image.open(result['image_path']) as image:
            # Source frame 18 (0.75s), not frame 6 (0.25s after the trim).
            self.assertAlmostEqual(float(np.asarray(image)[:, :, 2].mean()), 144, delta=5)
        self.body['_viggle_prepared'] = result
        self.body['_viggle_trim_start'] = 0.6
        self.assertEqual(self.prepare()['signature'], result['signature'])
        self.assertEqual(len(self.calls), 1, 'Same original frame can reuse its Klein preview')

    def lifecycle_worker(self, job, *, outcome='success'):
        # Load the actual integration function without importing the web server/models.
        tree = ast.parse((ROOT / 'app/launch.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_prepare_viggle_character_frame')
        lock = threading.Lock()
        releases = []
        model = SimpleNamespace(wan_model=object(), offloadobj=None, release_model=lambda: releases.append(True),
                                save_path='saved-video', image_save_path='saved-image', audio_save_path='saved-audio')
        async def submission(body, **kwargs):
            self.assertTrue(kwargs['prepare_only'])
            return {'params': body}
        def run(job_id, *, finalize, _slot_owned):
            self.assertTrue(lock.locked(), 'Klein and Viggle must own the same slot')
            self.assertTrue(_slot_owned)
            self.assertFalse(finalize)
            self.assertEqual(job_id, job['id'], 'One identity for cancellation')
            model.save_path = model.image_save_path = model.audio_save_path = job['out_dir']
            if outcome == 'cancel':
                lifecycle.request_cancel(job)
                return False
            if outcome == 'fail':
                lifecycle.finish_job(job, 'failed', error='Image model failed')
                return False
            output = self.render(job['params'], Path(job['out_dir']))
            return lifecycle.update_job(job, _internal_output_files=[output.name])
        namespace = dict(asyncio=asyncio, os=os, _prepare_generation_submission=submission,
                         _resolve_tool_clip_path=lambda raw, _: raw, _run_generation=run,
                         update_job=lifecycle.update_job, finish_job=lifecycle.finish_job,
                         is_cancel_requested=lifecycle.is_cancel_requested,
                         wgp=model)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'integration', 'exec'), namespace)
        with lifecycle.generation_slot(lock, job) as acquired:
            self.assertTrue(acquired)
            try:
                return namespace[function.name](job)
            finally:
                self.assertEqual(releases, [True])
                self.assertEqual(job['out_dir'], 'original-output')
                self.assertIs(job['params'], self.body)
                self.assertNotIn('_internal_output_files', job)
                self.assertEqual((model.save_path, model.image_save_path, model.audio_save_path),
                                 ('saved-video', 'saved-image', 'saved-audio'))

    def test_internal_phase_restores_video_request_and_keeps_parent_running(self):
        job = {'id': 'same-job', 'status': 'running', 'params': self.body, 'out_dir': 'original-output'}
        self.assertTrue(self.lifecycle_worker(job))
        self.assertEqual(job['status'], 'running')
        self.assertEqual(job['params']['model_type'], 'viggle_animate')
        self.assertEqual(job['params']['image_refs'], [job['viggle_preparation']['image_path']])

    def test_preview_finishes_after_image_and_cancel_failure_restore_parent(self):
        self.body['_viggle_prepare_only'] = True
        job = {'id': 'preview', 'status': 'running', 'params': self.body, 'out_dir': 'original-output'}
        self.assertTrue(self.lifecycle_worker(job))
        self.assertEqual(job['status'], 'completed')
        self.body.pop('_viggle_prepared')
        for outcome, status in [('cancel', 'cancelled'), ('fail', 'failed')]:
            job = {'id': outcome, 'status': 'running', 'params': self.body, 'out_dir': 'original-output'}
            with self.subTest(outcome=outcome), self.assertRaises(RuntimeError):
                self.lifecycle_worker(job, outcome=outcome)
            self.assertEqual(job['status'], status)


if __name__ == '__main__': unittest.main()
