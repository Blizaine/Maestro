"""Song preparation: speaker selection, reviewed crops, resume, and queue safety."""
import copy
import json
import math
from pathlib import Path
import sys
import shutil
import struct
import tempfile
import threading
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import music_dataset as data, music_dataset_analysis as analysis, music_training as projects
from services.music_contracts import tokenizer_pair
from services.music_training_runner import MusicTrainingRunner
from services.job_lifecycle import request_cancel


def words(count=80, speaker='a', start=0):
    return [{'text': f'word{i}', 'start': start + i * .4, 'end': start + i * .4 + .3,
             'probability': .95, 'uncertain': False, 'phrase_end': i % 10 == 9,
             'speaker': speaker, 'overlap': False} for i in range(count)]


class SegmentationTests(unittest.TestCase):
    def test_long_song_language_is_detected_per_chunk_and_boundary_words_owned_once(self):
        from services import audio_analysis
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'song.wav'
            sf.write(path, np.zeros(130 * 8000, dtype=np.float32), 8000)
            calls = []
            def transcribe(audio, **kwargs):
                n = len(calls); calls.append(kwargs)
                samples = [(1., 'first'), (60.5, 'boundary')] if n == 0 else [(2., 'boundary'), (3., 'second')] if n == 1 else [(2., 'last')]
                row = SimpleNamespace(avg_logprob=-.1, no_speech_prob=.01, words=[
                    SimpleNamespace(start=t, end=t + .2, word=word, probability=.9) for t, word in samples])
                return iter([row]), SimpleNamespace(language='nn' if n == 0 else 'en')
            with patch.object(audio_analysis, '_get_whisper_model', return_value=SimpleNamespace(transcribe=transcribe)), \
                 patch.object(audio_analysis, 'unload_whisper'):
                result = analysis.transcribe_words(path, cancelled=lambda: False)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call['language'] is None for call in calls))
            self.assertEqual(result['language'], 'en')
            self.assertEqual([w['text'] for w in result['words']], ['first', 'boundary', 'second', 'last'])
            self.assertAlmostEqual(result['words'][1]['start'], 60.5)

    def test_most_confident_words_beats_longest_duration_or_first_speaker(self):
        first = words(2, 'guest')
        first[0]['end'] = 14
        rows, selected = data.voice_summary(first + words(15, 'lead', 15) + words(7, 'unknown', 25))
        self.assertEqual(selected, ['lead'])
        self.assertEqual(rows[0]['word_count'], 15)
        self.assertLessEqual(rows[0]['end'] - rows[0]['start'], 8.3)

    def test_boundary_ambiguity_and_missing_diarization_are_not_false_identity(self):
        tagged = data.assign_speakers(words(3), [
            {'start': 0, 'end': .5, 'speaker': 'a'}, {'start': .5, 'end': .8, 'speaker': 'b'}])
        self.assertEqual(tagged[0]['speaker'], 'a')
        self.assertTrue(tagged[1]['overlap'])
        self.assertEqual(tagged[2]['speaker'], 'unknown')

    def test_clips_follow_phrase_ends_and_do_not_bridge_guest_or_overlap(self):
        source = words(160)
        source[61]['speaker'] = 'guest'
        source[85]['overlap'] = True
        clips = data.suggest_clips(source, ['a'], 65, 'Fast rap')
        self.assertGreater(len(clips), 2)
        for clip in clips:
            self.assertLessEqual(clip['end'] - clip['start'], 30)
            self.assertNotIn('word61 ', clip['lyrics'] + ' ')
            self.assertFalse(clip['reviewed'])
            for excluded in (source[61], source[85]):
                self.assertTrue(clip['end'] <= excluded['start'] + .001 or clip['start'] >= excluded['end'] - .001)
        self.assertTrue(all(a['end'] <= b['start'] for a, b in zip(clips, clips[1:])))
        self.assertTrue(any(20 <= c['end'] - c['start'] <= 30 for c in clips))

    def test_multiple_clusters_can_be_selected_and_short_fragments_are_excluded(self):
        source = words(50) + words(30, 'other-register', 20)
        clips = data.suggest_clips(source, ['a', 'other-register'], 40, '')
        self.assertTrue(any('word29' in c['lyrics'] for c in clips))
        short = data.suggest_clips(words(5), ['a'], 4, '')
        self.assertFalse(short[0]['included'])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(projects, 'PROJECT_ROOT', self.root / 'projects').start()
        patch.object(data, 'audio_duration', return_value=60.).start()
        self.tracks = []
        for index in range(2):
            path = self.root / f'original{index}.wav'
            path.write_bytes(f'original song {index}'.encode())
            self.tracks.append({'audio_path': str(path), 'holdout': bool(index), 'style': 'Rhythmic rap'})
        self.project = data.create_draft('Preparation test', 'Original test music', self.tracks)

    def summarized(self):
        prep = copy.deepcopy(self.project['preparation'])
        for track in self.project['tracks']:
            directory = data.song_directory(self.project, track['id'])
            data.write_json(directory / 'words.json', {'words': words(120)})
            speakers, selected = data.voice_summary(words(120))
            prep['songs'][track['id']] = {'duration': 60., 'language': 'en', 'speakers': speakers,
                'selected_speakers': selected, 'holdout': track['holdout'], 'warnings': [],
                'clips': data.suggest_clips(words(120), selected, 60, track['style'])}
        self.project = projects.update_project(self.project['id'], preparation=prep)
        return self.project

    def review(self):
        self.summarized()
        for track in self.project['tracks']:
            song = self.project['preparation']['songs'][track['id']]
            self.project = data.edit_song(self.project['id'], {'track_id': track['id'],
                'revision': self.project['preparation']['revision'], 'holdout': song['holdout'],
                'clips': [{**clip, 'reviewed': True, 'delivery': 'rap'} for clip in song['clips']]})
        return self.project

    def test_no_lyrics_required_new_default_v9_legacy_fallback_v4(self):
        self.assertEqual(self.project['tokenizer_pair'], 'v9')
        self.assertEqual(self.project['tracks'][0]['lyrics'], '')
        self.assertEqual(tokenizer_pair({}), tokenizer_pair({'tokenizer_pair': 'v4'}))
        self.assertEqual(Path(self.tracks[0]['audio_path']).read_bytes(), b'original song 0')

    def test_split_and_duplicate_sources_rejected(self):
        for tracks in ([{**t, 'holdout': False} for t in self.tracks],
                       [self.tracks[0], {**self.tracks[0], 'holdout': True}]):
            with self.assertRaises(ValueError):
                data.create_draft('Test', 'Trigger', tracks)

    def test_review_required_and_revision_conflicts_protected(self):
        self.summarized()
        with self.assertRaisesRegex(ValueError, 'review'):
            data.dataset_selection(self.project)
        track = self.project['tracks'][0]
        for revision in (-1, None):
            with self.assertRaisesRegex(ValueError, 'changed'):
                data.edit_song(self.project['id'], {'track_id': track['id'], 'revision': revision})
        projects.update_project(self.project['id'], status='preparing')
        with self.assertRaisesRegex(ValueError, 'stopping'):
            data.edit_song(self.project['id'], {})

    def test_regenerate_resets_reviews_for_only_selected_song(self):
        self.review()
        before = copy.deepcopy(self.project['preparation']['songs'])
        track = self.project['tracks'][0]
        updated = data.edit_song(self.project['id'], {'track_id': track['id'], 'revision': self.project['preparation']['revision'],
                                                     'selected_speakers': ['a']}, regenerate=True)
        self.assertFalse(any(c['reviewed'] for c in updated['preparation']['songs'][track['id']]['clips']))
        other = self.project['tracks'][1]['id']
        self.assertEqual(updated['preparation']['songs'][other], before[other])

    def test_manual_clip_validation(self):
        self.summarized()
        track = self.project['tracks'][0]
        body = {'track_id': track['id'], 'revision': 0, 'holdout': False,
                'clips': [{'id': 'manual-1', 'start': 1, 'end': 26, 'lyrics': 'Manually checked words',
                           'style': 'Rap', 'delivery': 'rap', 'included': True, 'reviewed': True}]}
        for bad in (float('nan'), -1, 61):
            with self.assertRaises(ValueError):
                data.edit_song(self.project['id'], {**body, 'clips': [{**body['clips'][0], 'start': bad}]})
        with self.assertRaisesRegex(ValueError, 'overlap'):
            data.edit_song(self.project['id'], {**body, 'clips': body['clips'] + [{**body['clips'][0], 'id': 'manual-2'}]})
        updated = data.edit_song(self.project['id'], body)
        self.assertEqual(updated['preparation']['songs'][track['id']]['clips'][0]['id'], 'manual-1')

    def test_build_is_idempotent_immutable_grouped_and_originals_untouched(self):
        self.review()
        before = [Path(t['audio_path']).read_bytes() for t in self.tracks]
        def convert(args, cancelled):
            Path(args[-1]).write_bytes(json.dumps(args[:-1]).encode())
        with patch.object(data, 'run_media', side_effect=convert) as media:
            created = data.build_dataset(self.project, report=lambda *args: None, cancelled=lambda: False)
            again = data.build_dataset(self.project, report=lambda *args: None, cancelled=lambda: False)
        self.assertEqual(created['id'], again['id'])
        self.assertEqual(media.call_count, len(created['tracks']))
        self.assertEqual(created['tokenizer_pair'], 'v9')
        self.assertEqual(len(created['reviews']), len(created['tracks']))
        self.assertEqual(len({t['source_song'] for t in created['tracks']}), 2)
        self.assertTrue(all('Rapped lead vocals.' in t['style'] for t in created['tracks']))
        self.assertEqual(before, [Path(t['audio_path']).read_bytes() for t in self.tracks])
        original = copy.deepcopy(created)
        self.project = projects.get_project(self.project['id'])
        track = self.project['tracks'][0]
        song = self.project['preparation']['songs'][track['id']]
        edited = data.edit_song(self.project['id'], {'track_id': track['id'], 'revision': self.project['preparation']['revision'],
            'holdout': False, 'clips': [{**c, 'lyrics': 'Changed lyrics'} for c in song['clips']]})
        self.assertNotIn('dataset_project_id', edited['preparation'])
        self.assertEqual(original, projects.get_project(created['id']))

    def test_cancelled_build_resumes_completed_crops(self):
        self.review()
        stop = False
        def convert(args, cancelled):
            Path(args[-1]).write_bytes(json.dumps(args[:-1]).encode())
        def report(*args):
            nonlocal stop
            stop = True
        with patch.object(data, 'run_media', side_effect=convert):
            with self.assertRaises(InterruptedError):
                data.build_dataset(self.project, report=report, cancelled=lambda: stop)
        self.assertEqual(len(projects.list_projects()), 1)
        with patch.object(data, 'run_media', side_effect=convert) as media:
            created = data.build_dataset(self.project, report=lambda *args: None, cancelled=lambda: False)
        self.assertEqual(media.call_count, len(created['tracks']) - 1)

    def test_changed_source_rejected(self):
        self.review()
        Path(self.tracks[0]['audio_path']).write_bytes(b'Changed source')
        with self.assertRaisesRegex(ValueError, 'changed'):
            data.build_dataset(self.project, report=lambda *args: None, cancelled=lambda: False)

    def test_analysis_resumes_stage_cache_and_keeps_user_review(self):
        def media(args, cancelled):
            Path(args[-1]).write_bytes(b'preview')
        def separate(track, dest, **kwargs):
            dest.write_bytes(b'vocals')
        with patch.object(analysis, 'run_media', side_effect=media), patch.object(analysis, 'separate_vocals', side_effect=separate) as stem, \
             patch.object(analysis, 'transcribe_words', return_value={'words': words(), 'language': 'en'}) as transcribe, \
             patch.object(analysis, 'diarize_voices', side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):
                analysis.analyze_songs(self.project, report=lambda *args: None, cancelled=lambda: False)
            self.assertEqual(stem.call_count, 1)
            self.assertEqual(transcribe.call_count, 1)
        with patch.object(analysis, 'run_media', side_effect=media), patch.object(analysis, 'separate_vocals', side_effect=separate) as stem, \
             patch.object(analysis, 'transcribe_words', return_value={'words': words(), 'language': 'en'}) as transcribe, \
             patch.object(analysis, 'diarize_voices', return_value=[{'speaker': 'a', 'start': 0, 'end': 60}]):
            analysis.analyze_songs(self.project, report=lambda *args: None, cancelled=lambda: False)
            self.assertEqual(stem.call_count, 1)  # second song only
            self.assertEqual(transcribe.call_count, 1)
        restored = projects.get_project(self.project['id'])
        with patch.object(analysis, 'separate_vocals', side_effect=AssertionError('replaced completed analysis')):
            analysis.analyze_songs(restored, report=lambda *args: None, cancelled=lambda: False)
        self.assertEqual(projects.get_project(self.project['id']), restored)

    def test_queue_cancellation_never_unloads_a_running_generation(self):
        lock = threading.Lock(); lock.acquire()
        try:
            jobs = {}
            runner = MusicTrainingRunner(jobs, lock, {}, lambda: self.fail('Unloaded running model'), lambda: '')
            with patch('threading.Thread.start'):
                queued = runner.submit('analyze-songs', self.project['id'], {})
            request_cancel(jobs[queued['job_id']])
            with self.assertRaises(ValueError):
                runner.submit('analyze-songs', self.project['id'], {})
            runner.run(queued['job_id'])
            self.assertNotIn(self.project['id'], runner.project_workers)
            self.assertEqual(projects.get_project(self.project['id'])['status'], 'cancelled')
            with self.assertRaisesRegex(ValueError, 'reviewed'):
                runner.submit('train', self.project['id'], {})
        finally:
            lock.release()

    def test_retry_failed_voices_reuses_stems_and_transcripts(self):
        def media(args, cancelled):
            Path(args[-1]).write_bytes(b'preview')
        def separate(track, dest, **kwargs):
            dest.write_bytes(b'vocals')
        with patch.object(analysis, 'run_media', side_effect=media), patch.object(analysis, 'separate_vocals', side_effect=separate), \
             patch.object(analysis, 'transcribe_words', return_value={'words': words(), 'language': 'en'}), \
             patch.object(analysis, 'diarize_voices', side_effect=ValueError('model unavailable')):
            analysis.analyze_songs(self.project, report=lambda *args: None, cancelled=lambda: False)
        updated = projects.get_project(self.project['id'])
        self.assertTrue(all(s['diarization_failed'] for s in updated['preparation']['songs'].values()))
        with patch.object(analysis, 'separate_vocals', side_effect=AssertionError('recomputed stems')), \
             patch.object(analysis, 'transcribe_words', side_effect=AssertionError('recomputed lyrics')), \
             patch.object(analysis, 'diarize_voices', return_value=[{'speaker': 'lead', 'start': 0, 'end': 60}]):
            analysis.analyze_songs(updated, report=lambda *args: None, cancelled=lambda: False, retry_voices=True)
        updated = projects.get_project(self.project['id'])
        self.assertTrue(all(not s['diarization_failed'] and s['selected_speakers'] == ['lead'] for s in updated['preparation']['songs'].values()))

    def test_rescan_preserves_reviewed_manual_clip_and_other_song(self):
        def media(args, cancelled):
            Path(args[-1]).write_bytes(b'preview')
        def separate(track, dest, **kwargs):
            dest.write_bytes(b'vocals')
        with patch.object(analysis, 'run_media', side_effect=media), patch.object(analysis, 'separate_vocals', side_effect=separate), \
             patch.object(analysis, 'transcribe_words', return_value={'words': words(), 'language': 'nn'}), \
             patch.object(analysis, 'diarize_voices', return_value=[]):
            analysis.analyze_songs(self.project, report=lambda *args: None, cancelled=lambda: False)
        updated = projects.get_project(self.project['id'])
        first, other = [t['id'] for t in updated['tracks']]
        manual = {'id': 'manual-1', 'start': 35, 'end': 41, 'lyrics': 'My checked words', 'style': 'Rap',
                  'delivery': 'rap', 'included': True, 'reviewed': True, 'warnings': []}
        updated['preparation']['songs'][first]['clips'] = [manual]
        projects.update_project(updated['id'], preparation=updated['preparation'])
        unchanged = copy.deepcopy(updated['preparation']['songs'][other])
        with patch.object(analysis, 'transcribe_words', return_value={'words': words(), 'language': 'en'}) as transcribe, \
             patch.object(analysis, 'separate_vocals', side_effect=AssertionError('Unneeded stem rebuild')):
            analysis.analyze_songs(updated, report=lambda *args: None, cancelled=lambda: False, track_id=first, language='en')
        transcribe.assert_called_once()
        self.assertEqual(transcribe.call_args.kwargs['language'], 'en')
        result = projects.get_project(updated['id'])['preparation']['songs']
        self.assertIn(manual, result[first]['clips'])
        self.assertEqual(result[other], unchanged)
        self.assertEqual(result[first]['language_override'], 'en')

    def test_http_review_and_preview_boundaries(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from services.music_training_api import create_router
        except ImportError:
            self.skipTest('HTTP test dependencies unavailable')
        app = FastAPI()
        calls = []
        def submit(*args):
            calls.append(args)
            return {'job_id': 'test-queue'}
        app.include_router(create_router(submit, {}))
        client = TestClient(app)
        self.summarized()
        prefix = f"/api/v1/music-training/projects/{self.project['id']}"
        self.assertEqual(client.post(prefix + '/train', json={}).status_code, 400)
        self.assertEqual(client.post(prefix + '/fork', json={}).status_code, 400)
        self.assertEqual(client.get(prefix + '/preparation/not-a-source/audio').status_code, 404)
        self.assertEqual(client.post(prefix + '/save-preparation', json={'revision': -1}).status_code, 400)
        response = client.post(prefix + '/analyze-songs', json={'retry_voices': True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls, [('analyze-songs', self.project['id'], {'retry_voices': True})])


class RealMediaTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Media tools unavailable')
    def test_real_audio_crops_have_requested_duration_channels_and_sample_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(projects, 'PROJECT_ROOT', root / 'projects'):
                tracks = []
                for i in range(2):
                    path = root / f'test-{i}.wav'
                    with wave.open(str(path), 'wb') as audio:
                        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(8000)
                        audio.writeframes(b''.join(struct.pack('<h', round(1000 * math.sin(t * (220+i*100) * math.tau / 8000))) for t in range(24000)))
                    tracks.append({'audio_path': str(path), 'holdout': bool(i)})
                draft = data.create_draft('Temporary media validation', 'Synthetic fixture', tracks)
                prep = draft['preparation']
                for track in draft['tracks']:
                    prep['songs'][track['id']] = {'duration': 3., 'speakers': [], 'holdout': track['holdout'],
                        'clips': [{'id': 'test-crop', 'start': .5, 'end': 2., 'lyrics': '[verse]\nTest fixture',
                                   'style': 'Synthetic test tone', 'delivery': 'unspecified', 'included': True, 'reviewed': True}]}
                draft = projects.update_project(draft['id'], preparation=prep)
                created = data.build_dataset(draft, report=lambda *args: None, cancelled=lambda: False)
                for track in created['tracks']:
                    with wave.open(track['audio_path']) as cropped:
                        self.assertEqual(cropped.getnchannels(), 2)
                        self.assertEqual(cropped.getframerate(), 48000)
                        self.assertAlmostEqual(cropped.getnframes() / cropped.getframerate(), 1.5, places=2)
                    self.assertEqual(track['lyrics'].count('[verse]'), 1)
                    self.assertIn('0.5–2.0s', track['name'])


if __name__ == '__main__':
    unittest.main()
