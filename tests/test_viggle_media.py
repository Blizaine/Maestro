"""Animate trim timing, soundtrack alignment, source preservation and cleanup."""
import ast
import copy
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import cv2
import numpy as np
from models.minimax_h3 import viggle
from services import viggle_media as media


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ViggleMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        frames = root / 'frames.avi'
        writer = cv2.VideoWriter(str(frames), cv2.VideoWriter_fourcc(*'MJPG'), 30, (96, 64))
        for index in range(90):
            writer.write(np.full((64, 96, 3), (index * 2, 40, 80), dtype=np.uint8))
        writer.release()
        cls.audio = root / 'audio.wav'
        sample_rate = 32000
        t = np.arange(sample_rate * 3) / sample_rate
        signal = np.sin(2 * math.pi * np.where(t < 1, 440, 880) * t)
        with wave.open(str(cls.audio), 'wb') as output:
            output.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
            output.writeframes((signal * 20000).astype('<i2').tobytes())
        cls.source = root / 'control.mkv'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(frames), '-i', str(cls.audio),
                        '-c:v', 'libx264', '-crf', '0', '-c:a', 'pcm_s16le', str(cls.source)], check=True)
        cls.source_hash = hashlib.sha256(cls.source.read_bytes()).digest()

    def body(self, **changes):
        return dict(video_guide=str(self.source), image_refs=['edited.png'], video_length=124,
                    _viggle_trim_start=1.25, _viggle_trim_end=2.25, _viggle_frame_seconds=1.75,
                    audio_prompt_type='K', **changes)

    def prepare(self, body, **kwargs):
        return media.prepare_control_media(body, resolve_media=lambda path: path,
            aborted=kwargs.get('aborted', lambda: False), update=lambda message: None)

    def test_video_and_control_audio_use_same_absolute_interval(self):
        body = self.body()
        saved = copy.deepcopy(body)
        with self.prepare(body) as runtime:
            target = Path(runtime['video_guide'])
            self.assertEqual(runtime['video_length'], 24)
            video = cv2.VideoCapture(str(target))
            try:
                self.assertEqual(video.get(cv2.CAP_PROP_FPS), 30)
                self.assertEqual(round(video.get(cv2.CAP_PROP_FRAME_COUNT)), 30)
                ok, frame = video.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (64, 96, 3))
                # 1.25s is around original source frame 38, not frame zero.
                self.assertAlmostEqual(float(frame[:, :, 0].mean()), 76, delta=5)
            finally:
                video.release()
            audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(target),
                '-t', '0.2', '-f', 's16le', '-ac', '1', '-ar', '32000', '-'])
            samples = np.frombuffer(audio, dtype='<i2')
            peak = np.argmax(np.abs(np.fft.rfft(samples))) * 32000 / len(samples)
            self.assertAlmostEqual(peak, 880, delta=10)
        self.assertFalse(target.parent.exists())
        self.assertEqual(body, saved, 'Saved metadata must retain original paths and frame time')
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), self.source_hash)

    def test_custom_audio_trim_and_cleanup_when_generation_raises(self):
        body = self.body()
        body.update(audio_prompt_type='A', audio_guide=str(self.audio))
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            with self.prepare(body) as runtime:
                target = Path(runtime['video_guide'])
                with wave.open(runtime['audio_guide'], 'rb') as audio:
                    self.assertEqual(audio.getnframes(), 32000)
                    samples = np.frombuffer(audio.readframes(6400), dtype='<i2')
                    self.assertAlmostEqual(np.argmax(np.abs(np.fft.rfft(samples))) * 5, 880, delta=10)
                raise RuntimeError('generation failed')
        self.assertFalse(target.parent.exists())
        self.assertEqual(body['audio_guide'], str(self.audio))

    def test_full_source_and_old_jobs_avoid_transcoding(self):
        body = self.body()
        body.update(_viggle_trim_start=0, _viggle_trim_end=3)
        with self.prepare(body) as runtime:
            self.assertEqual(runtime, {'video_length': 72})
        body.pop('_viggle_trim_start'); body.pop('_viggle_trim_end')
        with patch.object(media, '_run', side_effect=AssertionError('No media work for old jobs')):
            with self.prepare(body) as runtime:
                self.assertEqual(runtime, {})

    def test_invalid_ranges_frames_and_runtime_duration(self):
        for changes in ({'_viggle_trim_start': -1}, {'_viggle_trim_end': None},
                        {'_viggle_trim_start': float('nan')}, {'_viggle_trim_end': float('inf')},
                        {'_viggle_trim_end': 1.26}, {'_viggle_frame_seconds': 0.5},
                        {'_viggle_frame_seconds': 2.25}):
            body = self.body(); body.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                viggle.normalize_settings(body)
        body = self.body(); body['_viggle_trim_end'] = 5
        with self.assertRaisesRegex(ValueError, 'outside the control video'):
            with self.prepare(body):
                self.fail('An invalid range must not reach the model')
        body = self.body(); viggle.normalize_settings(body)
        self.assertEqual(body['video_length'], 24)
        self.assertEqual(body['sliding_window_size'], 124)
        self.assertEqual(body['_viggle_frame_seconds'], 1.75)

    def test_trim_failure_does_not_fall_back_to_full_source_and_cleans_temp(self):
        original = media._run
        targets = []
        def fail(command, aborted):
            if command[0] == 'ffmpeg':
                targets.append(Path(command[-1]))
                targets[-1].touch()
                raise ValueError('encoder failed')
            return original(command, aborted)
        with patch.object(media, '_run', side_effect=fail), self.assertRaisesRegex(ValueError, 'encoder failed'):
            with self.prepare(self.body()):
                self.fail('Never generate against untrimmed media after a trim error')
        self.assertFalse(targets[0].parent.exists())

    def test_cancellation_stops_media_subprocess(self):
        started = time.monotonic()
        with self.assertRaises(InterruptedError):
            media._run([sys.executable, '-c', 'import time; time.sleep(30)'],
                       lambda: time.monotonic() - started > 0.3)
        self.assertLess(time.monotonic() - started, 3)

    def test_runtime_keeps_exact_output_length_but_native_window_alignment(self):
        # Execute the real CPU timing helpers without importing the GPU server.
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app/wgp.py').read_text(encoding='utf-8'))
        names = {'normalize_model_total_frame_count', 'align_model_frame_count', 'compute_sliding_window_no'}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {'math': math}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'wgp-timing', 'exec'), namespace)
        definition = dict(minimax_h3_viggle=True, frames_minimum=124, frames_maximum=124,
                          frame_alignment_modulus=17, frame_alignment_remainder=5,
                          sliding_window_exact_total_frames=True)
        for frames in (1, 24, 96, 124, 125, 230, 1200):
            self.assertEqual(namespace['normalize_model_total_frame_count'](frames, definition, 124), frames)
            self.assertGreaterEqual(namespace['compute_sliding_window_no'](frames, 124, 0, 18), 1)
        self.assertEqual(namespace['compute_sliding_window_no'](1, 124, 0, 18), 1)
        self.assertEqual(namespace['compute_sliding_window_no'](125, 124, 0, 18), 2)
        self.assertEqual(namespace['align_model_frame_count'](96, definition, for_generation=True), 124)
        definition.pop('minimax_h3_viggle')
        self.assertEqual(namespace['normalize_model_total_frame_count'](96, definition, 124), 124,
                         'Other H3 variants retain their existing alignment')


if __name__ == '__main__':
    unittest.main()
