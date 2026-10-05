"""Exercise source-clock wiring and real control/audio extraction without a model."""

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))


class WindowMediaTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from PIL import Image
        cls.np, cls.torch = np, torch
        cls.tree = ast.parse((ROOT / 'app/wgp.py').read_text(encoding='utf-8'))
        cls.generate = next(n for n in cls.tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'generate_video')
        cls.namespace = dict(np=np, torch=torch, Image=Image, os=__import__('os'),
                             _AUDIO_TRANSCODE_CACHE={},
                             has_image_file_extension=lambda _: False)
        names = {'get_resampled_video', '_ensure_soundfile_readable', 'slice_audio_window'}
        for node in cls.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'wgp.py', 'exec'), cls.namespace)
        cls.decode = staticmethod(cls.namespace['get_resampled_video'])
        cls.slice_audio = staticmethod(cls.namespace['slice_audio_window'])

    def reference_start(self, guide_start, history, *, clip_origin=0, reset=False, source_frames=0, source_overlap=0):
        # Execute the actual scheduler's clock calculation, through its first
        # input-waveform assignment, rather than duplicating that calculation.
        block = next(n.body for n in ast.walk(self.generate) if isinstance(n, ast.While)
                     and any(isinstance(a, ast.Assign) and isinstance(a.targets[0], ast.Name)
                             and a.targets[0].id == 'window_start_frame' for a in n.body))
        begin = next(i for i, n in enumerate(block) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'window_start_frame')
        end = next(i for i in range(begin, len(block)) if isinstance(block[i], ast.Assign)
                   and isinstance(block[i].targets[0], ast.Tuple))
        env = dict(guide_start_frame=guide_start, window_no=2 if history else 1,
                   reuse_frames=history, source_video_overlap_frames_count=source_overlap,
                   source_video_frames_count=source_frames, reset_control_aligment=reset,
                   current_video_length=124, audio_frame_offset=clip_origin)
        exec(compile(ast.Module(body=block[begin:end], type_ignores=[]), 'wgp.py', 'exec'), env)
        return env['reference_start_frame']

    def test_native_overlap_and_independent_reference_origins(self):
        self.assertEqual(self.reference_start(0, 0), 0)
        self.assertEqual(self.reference_start(124, 18), 106)
        self.assertEqual(self.reference_start(230, 18), 212)
        self.assertEqual(self.reference_start(0, 0, clip_origin=230), 230)
        self.assertEqual(self.reference_start(124, 18, clip_origin=230), 336)
        self.assertEqual(self.reference_start(240, 0, reset=True, source_frames=240, source_overlap=18), 0)

    def test_model_receives_reference_origin_without_changing_other_windows(self):
        call = next(n for n in ast.walk(self.generate) if isinstance(n, ast.Call)
                    and any(k.arg == 'window_start_frame_no' for k in n.keywords))
        expr = next(k.value for k in call.keywords if k.arg == 'window_start_frame_no')
        code = compile(ast.Expression(expr), 'wgp.py', 'eval')
        self.assertEqual(eval(code, dict(reference_start_frame=442, window_start_frame=106,
                                        model_def={'omni_reference': True})), 442)
        self.assertEqual(eval(code, dict(reference_start_frame=442, window_start_frame=106,
                                        model_def={})), 106)

    def test_clip_control_origin_advances_by_assembled_duration(self):
        tree = ast.parse((ROOT / 'app/launch.py').read_text(encoding='utf-8'))
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                                  and t.value.id == 'clip_params' and isinstance(t.slice, ast.Constant)
                                  and t.slice.value == 'video_frame_offset' for t in n.targets))
        for rendered, expected in ((0, 48), (124, 172), (230, 278)):
            env = dict(clip_params={}, multi_clip_video_origin_frame=48,
                       cumulative_offset=240 + rendered, multi_clip_audio_origin_frame=240)
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'launch.py', 'exec'), env)
            self.assertEqual(env['clip_params']['video_frame_offset'], expected)

    def test_real_control_frames_and_audio_advance_together(self):
        import soundfile as sf
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            self.skipTest('ffmpeg is required for the media fixture')
        import decord  # Assert the actual installed decoder is exercised.
        del decord
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            video, audio = path / 'control.mp4', path / 'soundtrack.wav'
            colors = ['red', 'green', 'blue']
            command = [ffmpeg, '-v', 'error', '-y']
            for color in colors:
                command += ['-f', 'lavfi', '-i', f'color={color}:s=64x32:r=24:d=2']
            command += ['-filter_complex', '[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]',
                        '-map', '[v]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)]
            subprocess.run(command, check=True, capture_output=True, timeout=30)
            waveform = self.np.repeat(self.np.array([.1, .2, .3], dtype='float32'), 32000 * 2)
            sf.write(audio, waveform, 32000, subtype='FLOAT')

            # A second pass includes one second of overlap with the first.
            for origin, color, amplitude in ((0, 0, .1), (48, 1, .2), (96, 2, .3)):
                start = self.reference_start(0, 0, clip_origin=origin)
                frames = self.decode(str(video), start, 24, 24)
                samples, rate = self.slice_audio(str(audio), start, 24, 24, directory)
                self.assertEqual(frames.shape[0], 24)
                self.assertEqual(int(frames[0].float().mean(dim=(0, 1)).argmax()), color)
                self.assertEqual(rate, 32000)
                self.assertAlmostEqual(float(samples.mean()), amplitude, places=5)

            # The source guide itself advances by the explicit origin, and
            # overlap samples reference the same original media times.
            offset = next(n for n in ast.walk(self.generate) if isinstance(n, ast.AugAssign)
                          and isinstance(n.target, ast.Name) and n.target.id == 'guide_frames_extract_start')
            env = dict(guide_frames_extract_start=24, video_frame_offset=48)
            exec(compile(ast.Module(body=[offset], type_ignores=[]), 'wgp.py', 'exec'), env)
            actual = self.decode(str(video), env['guide_frames_extract_start'], 48, 24)
            expected = self.decode(str(video), 72, 48, 24)
            self.assertTrue(self.torch.equal(actual, expected))
            overlap = self.decode(str(video), 48, 24, 24)
            first = self.decode(str(video), 0, 72, 24)
            self.assertTrue(self.torch.equal(overlap, first[48:72]))


if __name__ == '__main__':
    unittest.main()
