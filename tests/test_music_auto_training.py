"""CPU-only Auto orchestration: real dataset/checkpoint handoffs, mocked GPU work."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from services import music_auto_training as auto, music_dataset as data, music_training as projects
from services.music_styles import file_digest
from services.music_training_api import create_router
from services.music_training_runner import MusicTrainingRunner
from services.job_lifecycle import request_cancel


class AutoTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(projects, 'PROJECT_ROOT', self.root / 'projects').start()
        patch.object(data, 'audio_duration', return_value=60.).start()
        patch.object(data, 'run_media', side_effect=lambda args, cancelled: Path(args[-1]).write_text(json.dumps(args[:-1]))).start()
        tracks = []
        for index in range(2):
            path = self.root / f'song-{index}.wav'
            path.write_bytes(f'original fixture {index}'.encode())
            tracks.append({'audio_path': str(path), 'holdout': bool(index), 'style': 'Rhythmic rap'})
        self.draft = data.create_draft('Automatic fixture', 'My test voice', tracks)
        self.jobs, self.states = {}, {}
        self.lock = threading.Lock()
        self.runner = MusicTrainingRunner(self.jobs, self.lock, self.states, lambda: None, lambda: 'test')
        patch.object(self.runner, '_cleanup_music_memory').start()
        self.calls, self.fail_stage, self.cancel_at = [], None, None
        self.actual_execute = auto.execute_stage
        patch.object(auto, 'execute_stage', side_effect=self.execute).start()
        self.job_id = None
        app = FastAPI(); app.include_router(create_router(self.queue, self.jobs))
        self.client = TestClient(app)

    def queue(self, operation, project_id, options):
        with patch('services.music_training_runner.threading.Thread'):
            return self.runner.submit(operation, project_id, options)

    def analyze(self, project):
        prep = copy.deepcopy(project['preparation'])
        for track in project['tracks']:
            prep['songs'][track['id']] = {'duration': 60., 'holdout': track['holdout'], 'warnings': ['Check automatic transcript'],
                'selected_speakers': ['lead'], 'speakers': [], 'clips': [
                    {'id': 'excerpt', 'start': 1., 'end': 26., 'lyrics': 'These are original test words',
                     'style': 'Rap', 'delivery': 'rap', 'included': True, 'reviewed': False}]}
        return projects.update_project(project['id'], preparation=prep)

    def execute(self, operation, target, options, *, report, cancelled):
        self.calls.append((operation, target, copy.deepcopy(options)))
        self.assertTrue(self.lock.locked(), 'Every stage holds the shared GPU slot')
        if operation == self.fail_stage:
            raise RuntimeError('Fixture stage failed')
        project = projects.get_project(target)
        if operation == 'analyze-songs':
            return self.analyze(project)
        if operation in {'build-dataset', 'select-pair'}:
            return self.actual_execute(operation, target, options, report=report, cancelled=cancelled)
        if operation in {'prepare-pair', 'prepare'}:
            return projects.update_project(target, **{'pair_prepared' if operation == 'prepare-pair' else 'prepared': {'version': 1}})
        if operation == 'adapt-pair':
            steps = 25 if self.cancel_at == operation else options['steps']
            directory = projects.project_directory(target) / 'pair_checkpoints'; directory.mkdir(exist_ok=True)
            row = {'step': steps, 'file': f'head-{steps}.safetensors', 'nar_file': f'nar-{steps}.safetensors'}
            for branch in ('head', 'nar'):
                path = directory / row['file' if branch == 'head' else 'nar_file']
                path.write_bytes(f'{branch} fixture {steps}'.encode())
                row[branch + '_sha256'] = file_digest(path)
            result = projects.update_project(target, pair_completed_steps=steps, pair_resume_available=True,
                pair_training_options={k: v for k, v in options.items() if k not in ('resume', 'steps')},
                pair_checkpoints=[*project.get('pair_checkpoints', []), row])
        elif operation == 'train':
            directory = projects.project_directory(target) / 'checkpoints'; directory.mkdir(exist_ok=True)
            file = f"step-{options['steps']}.safetensors"; (directory / file).write_bytes(b'AR fixture')
            result = projects.update_project(target, completed_steps=options['steps'], resume_available=True,
                checkpoints=[{'step': options['steps'], 'file': file}],
                training_options={k: v for k, v in options.items() if k not in ('resume', 'steps')})
        elif operation == 'publish':
            self.assertEqual(project['completed_steps'], options['steps'])
            self.assertEqual(project['adapted_pair']['source_project'], projects.get_project(self.draft['id'])['auto_training']['sound_project_id'])
            result = {'id': 'saved-lora'}
        else:
            self.fail(f'Unexpected stage: {operation}')
        if operation == self.cancel_at:
            request_cancel(self.jobs[self.job_id], job_id=self.job_id, active_states=self.states)
            report('Saving current step', 25)
            self.assertTrue(cancelled())
        return result

    def submit(self, options=None):
        response = self.queue('auto-train', self.draft['id'], options or {})
        self.job_id = response['job_id']
        return self.job_id

    def run_job(self):
        self.runner.run(self.job_id)
        self.assertFalse(self.runner.project_workers)
        self.assertFalse(self.states)
        self.assertFalse(self.lock.locked())
        return projects.get_project(self.draft['id'])

    def test_whole_run_uses_defaults_and_matched_pair_without_faking_review(self):
        self.submit(); result = self.run_job()
        self.assertEqual([row[0] for row in self.calls], [row[0] for row in auto.STAGES])
        self.assertEqual(self.calls[3][2]['steps'], 100)
        self.assertEqual(self.calls[6][2]['steps'], 200)
        self.assertFalse(self.calls[6][2]['lyric_alignment'])
        plan = result['auto_training']
        self.assertEqual(plan['status'], 'completed')
        self.assertEqual(plan['style_id'], 'saved-lora')
        self.assertEqual(plan['selection_summary']['unreviewed_clips'], 2)
        sound = projects.get_project(plan['sound_project_id'])
        style = projects.get_project(plan['style_project_id'])
        self.assertEqual(sound['reviews'], {})
        self.assertEqual(style['reviews'], {})
        self.assertEqual(sound['tracks'], style['tracks'])
        self.assertEqual(len({track['source_song'] for track in sound['tracks']}), 2)
        self.assertEqual([track['holdout'] for track in sound['tracks']], [False, True])
        self.assertEqual(style['adapted_pair']['step'], 100)
        self.assertEqual((projects.project_directory(style['id']) / 'adapted_pair/head.safetensors').read_bytes(), b'head fixture 100')
        self.assertEqual(self.jobs[self.job_id]['music_style_id'], 'saved-lora')
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.submit()

    def test_stop_during_voice_then_resume_skips_dataset_and_keeps_targets(self):
        self.cancel_at = 'adapt-pair'
        self.submit({'voice_steps': 80, 'style_steps': 150}); first = self.run_job()
        self.assertEqual(first['auto_training']['status'], 'cancelled')
        self.assertEqual(len(self.calls), 4)
        self.assertNotIn('adapt-pair', first['auto_training']['done'])
        sound = projects.get_project(first['auto_training']['sound_project_id'])
        self.assertEqual(sound['pair_completed_steps'], 25)
        with self.assertRaisesRegex(ValueError, 'saved Auto targets'):
            self.submit({'voice_steps': 100})
        self.cancel_at = None; self.calls.clear(); self.submit(); result = self.run_job()
        self.assertEqual([row[0] for row in self.calls], ['adapt-pair', 'select-pair', 'prepare', 'train', 'publish'])
        self.assertTrue(self.calls[0][2]['resume'])
        self.assertEqual(self.calls[0][2]['steps'], 80)
        self.assertEqual(self.calls[3][2]['steps'], 150)
        self.assertEqual(result['auto_training']['sound_project_id'], sound['id'])

    def test_existing_dataset_starts_at_voice_training(self):
        self.draft = data.build_dataset(self.analyze(self.draft), automatic=True,
                                       report=lambda *args: None, cancelled=lambda: False)
        self.submit(); result = self.run_job()
        self.assertEqual(self.calls[0][0], 'prepare-pair')
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(result['auto_training']['sound_project_id'], result['id'])
        self.assertEqual(result['training_workflow'], 'author')

    def test_failed_style_resumes_only_unfinished_stage_and_child(self):
        self.fail_stage = 'train'; self.submit(); first = self.run_job()
        self.assertEqual(first['auto_training']['status'], 'failed')
        self.assertNotIn('publish', [row[0] for row in self.calls])
        self.fail_stage = None; self.calls.clear(); self.submit(); result = self.run_job()
        self.assertEqual([row[0] for row in self.calls], ['train', 'publish'])
        self.assertEqual(first['auto_training']['style_project_id'], result['auto_training']['style_project_id'])
        self.assertEqual(len(projects.list_projects()), 3)

    def test_queued_cancel_does_not_touch_recordings_and_duplicate_start_blocked(self):
        self.submit()
        with self.assertRaisesRegex(ValueError, 'queued or running'):
            self.submit()
        request_cancel(self.jobs[self.job_id], job_id=self.job_id, active_states=self.states)
        result = self.run_job()
        self.assertEqual(result['auto_training']['status'], 'cancelled')
        self.assertEqual(self.calls, [])
        self.assertEqual(len(projects.list_projects()), 1)

    def test_auto_waits_for_gpu_then_advances_without_client_requests(self):
        self.submit()
        self.lock.acquire()
        worker = threading.Thread(target=self.runner.run, args=(self.job_id,))
        worker.start()
        try:
            worker.join(.15)
            self.assertTrue(worker.is_alive())
            self.assertEqual(self.jobs[self.job_id]['status'], 'queued')
            self.assertEqual(self.calls, [])
        finally:
            self.lock.release()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.jobs[self.job_id]['status'], 'completed')
        self.assertEqual(len(self.calls), 8)

    def test_publish_pairs_ar_with_its_adapted_decoder(self):
        self.fail_stage = 'publish'; self.submit(); result = self.run_job()
        child = projects.get_project(result['auto_training']['style_project_id'])
        def read(path, branch):
            return (branch, Path(path).read_bytes())
        with patch('services.music_styles.list_styles', return_value=[]), \
             patch('models.TTS.yue2.artist_adapter.read_upstream_adapter', side_effect=read), \
             patch('services.music_styles.save_style', return_value={'id': 'new-style'}) as save:
            self.actual_execute('publish', child['id'], {'auto_root': result['id'], 'steps': 200, 'name': result['name']},
                                report=lambda *args: None, cancelled=lambda: False)
        self.assertEqual(save.call_args.args[2:], (('ar', b'AR fixture'), ('nar', b'nar fixture 100')))
        self.assertEqual(save.call_args.kwargs['adapted_pair'], child['adapted_pair'])
        self.assertEqual(save.call_args.kwargs['training']['auto_root'], result['id'])

    def test_api_validation_and_restart_resume(self):
        url = f"/api/v1/music-training/projects/{self.draft['id']}"
        for invalid in (0, 1601, 5.5, True, '100'):
            self.assertEqual(self.client.post(url + '/auto-train', json={'voice_steps': invalid}).status_code, 400)
        self.assertEqual(self.jobs, {})
        response = self.client.post(url + '/auto-train', json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.post(url + '/save-preparation', json={}).status_code, 400)
        self.jobs.clear(); self.runner.project_workers.clear(); self.runner.job_projects.clear()
        interrupted = self.client.get(url).json()
        self.assertEqual(interrupted['auto_training']['status'], 'interrupted')
        self.assertEqual(self.client.post(url + '/auto-train', json={}).status_code, 200)

    def test_linked_project_cannot_be_changed_or_queued_while_auto_owns_it(self):
        self.fail_stage = 'train'; self.submit(); result = self.run_job()
        child_id = result['auto_training']['style_project_id']
        self.submit()
        for operation in ('fork', 'review-track', 'train', 'select-pair'):
            response = self.client.post(f'/api/v1/music-training/projects/{child_id}/{operation}', json={})
            self.assertEqual(response.status_code, 400)
            self.assertIn('Auto training is using', response.json()['detail'])
        with self.assertRaisesRegex(ValueError, 'queued or running'):
            self.runner.submit('train', child_id, {'steps': 200})

    def test_autobuild_does_not_weaken_manual_review_and_reuses_saved_excerpts(self):
        project = self.analyze(self.draft)
        with self.assertRaisesRegex(ValueError, 'review'):
            data.dataset_selection(project)
        first = self.actual_execute('build-dataset', project['id'], {}, report=lambda *args: None, cancelled=lambda: False)
        second = self.actual_execute('build-dataset', project['id'], {}, report=lambda *args: None, cancelled=lambda: False)
        self.assertEqual(first['id'], second['id'])
        self.assertEqual(first['selection_mode'], 'auto')
        self.assertFalse(first['reviews'])
        prep = project['preparation']
        prep['songs'][project['tracks'][0]['id']]['clips'][0]['end'] = 10
        projects.update_project(project['id'], preparation=prep)
        with self.assertRaisesRegex(ValueError, '20.5 seconds'):
            self.actual_execute('build-dataset', project['id'], {}, report=lambda *args: None, cancelled=lambda: False)

    def test_recover_after_child_copy_and_publish_before_plan_checkpoint(self):
        self.fail_stage = 'train'; self.submit(); result = self.run_job()
        plan = result['auto_training']; sound = projects.get_project(plan['sound_project_id'])
        child = self.actual_execute('select-pair', sound['id'], {'checkpoint': 'head-100.safetensors', 'auto_root': result['id']}, report=lambda *args: None, cancelled=lambda: False)
        self.assertEqual(child['id'], plan['style_project_id'])
        with patch('services.music_styles.list_styles', return_value=[{'id': 'already-saved', 'training': {'auto_root': result['id']}}]), \
             patch('services.music_styles.load_style', return_value={'id': 'already-saved'}) as load, \
             patch('services.music_styles.save_style') as save:
            recovered = self.actual_execute('publish', child['id'], {'auto_root': result['id']}, report=lambda *args: None, cancelled=lambda: False)
        self.assertEqual(recovered['id'], 'already-saved'); save.assert_not_called()
        load.assert_called_once_with('already-saved', verify=True)


if __name__ == '__main__':
    unittest.main()
