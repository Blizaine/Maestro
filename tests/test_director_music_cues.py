import ast
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from services.director.music_cues import detect_percussion, format_music_cues
from services.director.planners.music_video import MusicVideoPlanner


class MusicCueTests(unittest.TestCase):
    def test_percussion_entry_timing_and_silent_tonal_controls(self):
        sr = 22050
        t = np.arange(sr * 12) / sr
        tonal = (.1 * np.sin(2*np.pi*220*t)).astype(np.float32)
        for audio in (np.zeros_like(tonal), tonal):
            intervals, cues = detect_percussion(audio, sr)
            self.assertFalse(intervals)
            self.assertEqual(cues, [])
        drums = tonal.copy()
        rng = np.random.default_rng(41)
        for time in np.arange(5.2, 10.5, .5):
            length = int(.12*sr)
            hit = rng.normal(size=length) * np.exp(-np.arange(length)/(sr*.025))
            start = round(time*sr)
            drums[start:start+length] += hit.astype(np.float32)
        intervals, cues = detect_percussion(drums, sr)
        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0]['time'], 5.2, delta=.2)
        self.assertEqual(cues[0]['confidence'], 'estimated')
        self.assertGreater(intervals[0]['end'], 10)

    def test_planner_receives_local_instrument_timing_without_inventing_certainty(self):
        clip = {'start': 20, 'end': 28, 'section_label': 'chorus', 'percussion_activity': 'active',
                'music_cues': [{'type': 'percussion_entry', 'time': 22.4, 'confidence': 'estimated'}]}
        planner = MusicVideoPlanner()
        context = planner._build_clip_contexts([clip], [], {}, {}, {})[0]
        self.assertIn('chorus', context)
        self.assertIn('song 22.40s, shot +2.40s', context)
        self.assertIn('not a verified instrument identity', context)
        self.assertIn('same drummer', context)
        self.assertIn('No sustained percussion', format_music_cues({'percussion_activity': 'quiet'}))
        self.assertIn('timing is unknown', format_music_cues({'percussion_activity': 'unknown'}))
        self.assertEqual(format_music_cues({}), '')

    def test_long_clip_keeps_internal_section_and_percussion_timing_for_the_writer(self):
        from services.director_music_timing import plan_capped_music_clips, prepare_music_timeline

        analysis = {'duration': 28, 'bpm': 120,
                    'sections': [{'start': 0, 'end': 16, 'label': 'verse', 'energy': .4},
                                 {'start': 16, 'end': 28, 'label': 'chorus', 'energy': .9}],
                    'percussion_activity': [{'start': 18, 'end': 28}],
                    'music_cues': [{'type': 'percussion_entry', 'time': 18, 'confidence': 'estimated'}]}
        clips = plan_capped_music_clips(analysis, maximum_seconds=345/24, fps=24,
                                       frames_minimum=124, frames_steps=17, energy_bias=-2)
        _, clips = prepare_music_timeline([{}, {}], clips, fps=24, minimum_frames=124,
                                         maximum_frames=345, frame_step=17)
        self.assertEqual(len(clips), 2)
        context = MusicVideoPlanner()._build_clip_contexts(clips, [], {}, {}, {})[1]
        self.assertIn('section changes to chorus at song 16.00s, shot +2.00s', context)
        self.assertIn('percussion entrance at song 18.00s, shot +4.00s', context)
        self.assertIn('preserve the full planned clip duration', context)

    def test_real_task_setup_advances_soundtrack_by_visible_frames(self):
        # Execute the production queue's clip timing block without starting
        # the server or GPU. Padding must not shift the next audio slice.
        tree = ast.parse((ROOT / 'app/launch.py').read_text(encoding='utf-8'))
        target = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and any(
            isinstance(c, ast.Assign) and isinstance(c.value, ast.IfExp)
            and any(isinstance(t, ast.Name) and t.id == 'clip_frames' for t in c.targets)
            for c in n.body))
        start = next(i for i, n in enumerate(target.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'clip_frames' for t in n.targets))
        stop = next(i for i, n in enumerate(target.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                            and t.slice.value == 'multi_clip_info' for t in n.targets))
        program = compile(ast.Module(body=target.body[start:stop+1], type_ignores=[]), '<queue clip timing>', 'exec')
        env = dict(per_clip_frames=[192, 175, 243], per_clip_output_frames=[180, 166, 231],
                   sw_size=243, _mc_bounded_director=True, _mc_model_def={'frames_maximum':345},
                   _mc_min_f=124, _mc_fs=17, has_end=False, _mc_trim_end_frames=False,
                   cumulative_offset=0, total_trimmed_frames=0, group_id='test', clip_count=3,
                   multi_clip_audio_start_sec=0, multi_clip_concat_audio='song.wav',
                   omni_sequence_continuity=False, omni_sequence_target_frames=0)
        offsets, trims = [], []
        for i in range(3):
            env.update(i=i, clip_params={})
            exec(program, env)
            offsets.append(env['clip_params']['audio_frame_offset'])
            trims.append(env['clip_params']['trim_tail_frames'])
        self.assertEqual(offsets, [0, 180, 346])
        self.assertEqual(trims, [12, 9, 12])
        self.assertEqual(env['cumulative_offset'], 577)
        compensation = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                            and 'total_trimmed_frames > 0' in ast.unparse(n.test))
        condition = compile(ast.Expression(compensation.test), '<compensation condition>', 'eval')
        self.assertFalse(eval(condition, {**env, '_mc_trim_end_frames': True}), 'Editorial padding must not produce an extra compensation clip')
        self.assertTrue(eval(condition, {**env, '_mc_trim_end_frames': True, 'per_clip_output_frames': None}))


if __name__ == '__main__':
    unittest.main()
