"""Fixed requests, queue ownership, durable samples and immutable data review."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import music_training as projects
from services.music_auditions import audition_options, render_checkpoint, train_with_auditions, request_fingerprint
from services.music_training_runner import MusicTrainingRunner
from services.music_training_api import create_router


class AuditionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patcher = patch.object(projects, 'PROJECT_ROOT', self.root / 'projects')
        self.patcher.start(); self.addCleanup(self.patcher.stop)
        self.tracks = []
        for i in range(2):
            path = self.root / f'song-{i}.wav'; path.write_bytes(bytes([i]))
            self.tracks.append({'audio_path': str(path), 'lyrics': '[Verse]\nNew morning', 'style': 'Warm acoustic pop', 'holdout': bool(i), 'reviewed': True})
        self.project = projects.create_project('Music', 'My music', self.tracks)
        self.settings = audition_options({'enabled': True, 'style': 'Solo voice and piano', 'lyrics': '[Verse]\nAnother morning brings a new beginning', 'seconds': 8, 'seed': 123})

    def checkpoint(self, step, branch='style'):
        field = 'audio_checkpoints' if branch == 'audio' else 'checkpoints'
        directory = projects.project_directory(self.project['id']) / field
        directory.mkdir(exist_ok=True)
        filename = f'step-{step}.safetensors'; (directory / filename).write_bytes(str(step).encode())
        project = projects.get_project(self.project['id'])
        history = [row for row in project.get(field, []) if row['step'] != step]
        history.append({'step': step, 'file': filename, 'scores': {}, 'conditioning_checkpoint': ''})
        completed = 'audio_completed_steps' if branch == 'audio' else 'completed_steps'
        return projects.update_project(project['id'], **{field: history, completed: step})

    def test_request_bounds_and_fingerprint(self):
        for values in ([], {'enabled': 'yes'}, {'enabled': True}, {**self.settings, 'seconds': 0},
                       {**self.settings, 'seed': 1.5}, {**self.settings, 'strength': float('nan')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                audition_options(values)
        self.assertEqual(audition_options(None), {'enabled': False})
        self.assertEqual(audition_options(self.settings), self.settings)
        original = request_fingerprint(self.project, self.settings)
        self.assertNotEqual(original, request_fingerprint(self.project, {**self.settings, 'seed': 124}))
        self.assertEqual(projects.audio_training_options({'audition': self.settings})['audition'], self.settings)

    def test_interleaves_only_after_saved_checkpoints_and_resumes_both_trainers(self):
        for branch in ('style', 'audio'):
            calls = []
            def trainer(project, options, *, report, cancelled, pause_at_checkpoint):
                field = 'audio_completed_steps' if branch == 'audio' else 'completed_steps'
                before = project.get(field, 0)
                calls.append(('train', before, options['resume'], options['steps']))
                self.assertTrue(pause_at_checkpoint)
                self.checkpoint(before + 200, branch)
            def render(project, actual_branch, checkpoint, settings, **kwargs):
                calls.append(('audition', checkpoint))
                self.assertEqual(actual_branch, branch)
                self.assertEqual(settings, self.settings)
                self.assertTrue((projects.project_directory(project['id']) / ('audio_checkpoints' if branch == 'audio' else 'checkpoints') / checkpoint).exists())
            with patch('services.music_auditions.render_checkpoint', side_effect=render):
                failures = train_with_auditions(self.project['id'], {'steps': 400, 'resume': False, 'audition': self.settings}, trainer, branch, report=lambda *args: None, cancelled=lambda: False)
            self.assertEqual(failures, 0)
            self.assertEqual(calls, [('train', 0, False, 400), ('audition', 'step-200.safetensors'), ('train', 200, True, 400), ('audition', 'step-400.safetensors')])

    def test_failed_preview_is_saved_then_training_continues(self):
        project = self.checkpoint(200)
        with patch('models.TTS.yue2.training_audition.render_audition', side_effect=RuntimeError('decoder failed')):
            with self.assertRaisesRegex(RuntimeError, 'decoder failed'):
                render_checkpoint(project, 'style', 'step-200.safetensors', self.settings, report=lambda *args: None, cancelled=lambda: False)
        saved = projects.get_project(project['id'])
        self.assertEqual(saved['auditions'][0]['status'], 'failed')
        self.assertEqual(saved['completed_steps'], 200)
        def trainer(project, options, **kwargs):
            self.checkpoint(project['completed_steps'] + 200)
        with patch('services.music_auditions.render_checkpoint', side_effect=RuntimeError('preview failed')):
            self.assertEqual(train_with_auditions(project['id'], {'steps': 600, 'resume': True, 'audition': self.settings}, trainer, 'style', report=lambda *args: None, cancelled=lambda: False), 2)
        self.assertEqual(projects.get_project(project['id'])['completed_steps'], 600)

    def test_cancellation_during_preview_keeps_checkpoint_and_does_not_resume(self):
        steps = []
        def trainer(project, options, **kwargs):
            steps.append(200); self.checkpoint(200)
        with patch('services.music_auditions.render_checkpoint', side_effect=InterruptedError('cancelled')):
            with self.assertRaises(InterruptedError):
                train_with_auditions(self.project['id'], {'steps': 400, 'resume': False, 'audition': self.settings}, trainer, 'style', report=lambda *args: None, cancelled=lambda: False)
        self.assertEqual(steps, [200])
        self.assertEqual(projects.get_project(self.project['id'])['completed_steps'], 200)

    def test_data_review_is_bound_to_exact_labels_and_song_split(self):
        track = self.project['tracks'][0]
        self.assertEqual(self.project['reviews'][track['id']], projects.review_fingerprint(track))
        edited = projects.create_project('Edited', 'My music', [{**self.tracks[0], 'lyrics': 'Changed', 'reviewed': False}, self.tracks[1]])
        self.assertNotIn(track['id'], edited['reviews'])
        self.assertEqual(projects.get_project(self.project['id'])['tracks'][0]['lyrics'], self.tracks[0]['lyrics'])
        with self.assertRaisesRegex(ValueError, 'same original song'):
            projects.create_project('Bad split', 'My music', [{**track, 'source_song': 'Same song'} for track in self.tracks])
        Path(track['audio_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'recording changed'):
            projects.review_track(self.project['id'], track['id'], True)

    def test_audio_routes_only_serve_recorded_project_files_and_recover_interrupted_preview(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI(); app.include_router(create_router(lambda *args: None, {}))
        client = TestClient(app)
        project = self.checkpoint(200)
        projects.update_project(project['id'], status='auditioning', job_id='gone', auditions=[{'id': 'pending', 'status': 'running'}])
        base = f'/api/v1/music-training/projects/{project["id"]}'
        result = client.get(base).json()
        self.assertEqual(result['status'], 'interrupted')
        self.assertEqual(result['auditions'][0]['status'], 'cancelled')
        self.assertEqual(len(result['reviewed_track_ids']), 2)
        self.assertEqual(client.get(base + '/recordings/unknown').status_code, 404)
        self.assertEqual(client.get(base + '/recordings/' + project['tracks'][0]['id']).content, b'\0')
        projects.update_project(project['id'], auditions=[{'id': 'escape', 'status': 'completed', 'file': '../project.json'}])
        self.assertEqual(client.get(base + '/auditions/escape').status_code, 404)

    def test_queue_holds_gpu_slot_during_training_and_audition(self):
        project = projects.update_project(self.project['id'], prepared={'version': 1})
        lock, jobs, states = threading.Lock(), {}, {}
        runner = MusicTrainingRunner(jobs, lock, states, lambda: None, lambda: '')
        with patch('threading.Thread.start'):
            result = runner.submit('train', project['id'], projects.training_options({'steps': 1, 'audition': self.settings}))
        def training(*args, **kwargs):
            self.assertTrue(lock.locked())
            self.assertEqual(len(states), 1)
            return 0
        with patch('services.music_auditions.train_with_auditions', side_effect=training):
            runner.run(result['job_id'])
        self.assertEqual(jobs[result['job_id']]['status'], 'completed')
        self.assertFalse(lock.locked()); self.assertFalse(states)
        with patch('threading.Thread.start'):
            runner.submit('train', project['id'], projects.training_options({'steps': 2}))
        settings = projects.get_project(project['id'])['audition_settings']
        self.assertFalse(settings['enabled'])
        self.assertEqual(settings['lyrics'], self.settings['lyrics'])

    def test_local_draft_is_independent_and_never_changes_training_labels(self):
        from services import audio_analysis
        from services.music_data_review import draft_lyrics
        segments = [SimpleNamespace(text=' Different words ', start=0, end=2, avg_logprob=-1, no_speech_prob=.1)]
        model = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter(segments), SimpleNamespace(language='en', duration=10)))
        before = projects.get_project(self.project['id'])
        with patch.object(audio_analysis, '_get_whisper_model', return_value=model), patch.object(audio_analysis, 'unload_whisper') as unload, patch('services.music_data_review.subprocess.run', return_value=SimpleNamespace(stdout='{"format":{"duration":10}}')):
            draft_lyrics(before, report=lambda *args: None, cancelled=lambda: False)
        after = projects.get_project(self.project['id'])
        self.assertEqual(before['tracks'], after['tracks'])
        self.assertEqual(before['reviews'], after['reviews'])
        self.assertTrue(after['review_drafts'][before['tracks'][0]['id']]['segments'][0]['uncertain'])
        self.assertEqual(after['review_drafts'][before['tracks'][0]['id']]['text'], 'Different words')
        unload.assert_called_once()

    def test_empty_transcription_retries_without_speech_filter_and_stays_a_draft(self):
        from services import audio_analysis
        from services.music_data_review import draft_lyrics
        calls = []
        def transcribe(path, **kwargs):
            calls.append(kwargs['vad_filter'])
            segments = [] if kwargs['vad_filter'] else [SimpleNamespace(text='Heard words', start=0, end=2, avg_logprob=-.2, no_speech_prob=.1)]
            return iter(segments), SimpleNamespace(language='en', duration=10)
        with patch.object(audio_analysis, '_get_whisper_model', return_value=SimpleNamespace(transcribe=transcribe)), patch.object(audio_analysis, 'unload_whisper'), patch('services.music_data_review.subprocess.run', return_value=SimpleNamespace(stdout='{"format":{"duration":10}}')):
            draft_lyrics(self.project, report=lambda *args: None, cancelled=lambda: False)
        self.assertEqual(calls, [True, False, True, False])
        saved = projects.get_project(self.project['id'])
        self.assertEqual(saved['tracks'], self.project['tracks'])
        self.assertTrue(all(row['review_required'] and row['speech_filter_retry'] for row in saved['review_drafts'].values()))

    def test_sequence_coverage_surfaces_truncation_without_changing_prepared_tokens(self):
        import numpy as np
        from services.music_data_review import sequence_coverage
        directory = projects.project_directory(self.project['id']) / 'prepared'; directory.mkdir()
        for track, count in zip(self.project['tracks'], (1000, 20000)):
            np.save(directory / (track['id'] + '.npy'), np.zeros(count, dtype=np.int32))
        with patch('models.TTS.yue2.music_assets.ensure_asset', return_value='unused'), patch('models.TTS.yue2.tokenization_yue2.YuE2TextTokenizer'), patch('models.TTS.yue2.protocol.token_prefixes', return_value=list(range(10))):
            result = sequence_coverage(self.project, report=lambda *args: None, cancelled=lambda: False)
        self.assertFalse(result[0]['truncated'])
        self.assertTrue(result[1]['truncated'])
        self.assertEqual(result[1]['training_seconds'], (12288 - 11) / 25)
        self.assertEqual(len(np.load(directory / (self.project['tracks'][1]['id'] + '.npy'))), 20000)


if __name__ == '__main__':
    unittest.main()
