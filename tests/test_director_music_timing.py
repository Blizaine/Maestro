import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'app'))
from services.director_music_timing import resolve_clip_limits, plan_capped_music_clips, prepare_music_timeline
from services.director_video_strategy import adapt_bounded_timeline
from services import director_pipeline


class DirectorMusicTimingTests(unittest.TestCase):
    H3 = {'fps': 24, 'frames_minimum': 124, 'frames_steps': 17, 'frames_maximum': 345,
          'architecture': 'minimax_h3', 'bounded_video': True}
    LTX = {'fps': 25, 'frames_minimum': 17, 'frames_steps': 8, 'sliding_window': True,
           'sliding_window_defaults': {'window_default': 257, 'window_max': 649}}

    def analysis(self, duration):
        return {'duration': duration, 'bpm': 137, 'beats': [{'time': i*.438} for i in range(math.ceil(duration/.438))],
                'sections': [{'start': 0, 'end': .1, 'label': 'intro', 'energy': .2},
                             {'start': .1, 'end': duration-.1, 'label': 'verse', 'energy': .8},
                             {'start': duration-.1, 'end': duration, 'label': 'outro', 'energy': .4}],
                'lyrics': [{'start': 2, 'end': 5, 'speaker': 'singer'}]}

    def test_limits_distinguish_recommendation_manual_and_model_max(self):
        auto = resolve_clip_limits(self.H3, execution_profile={'effective_max_frames': 243})
        self.assertEqual(auto['max_frames'], 243)
        manual = resolve_clip_limits(self.H3, 6, {'effective_max_frames': 243})
        self.assertEqual(manual['max_frames'], 141)
        self.assertEqual(manual['recommended_frames'], 243)
        self.assertLess(manual['max_seconds'], 8)
        self.assertLessEqual(resolve_clip_limits(self.LTX, 2)['max_seconds'], 2)
        self.assertEqual(resolve_clip_limits(self.H3, 200)['max_frames'], 345)
        for invalid in [True, float('nan'), float('inf'), -1, 0]:
            with self.assertRaises(ValueError):
                resolve_clip_limits(self.H3, invalid)

    def test_gpu_override_can_raise_auto_without_changing_recommendation(self):
        profile = {'recommended_max_frames': 243, 'effective_max_frames': 345, 'manual_max_frames': 345}
        limits = resolve_clip_limits(self.H3, execution_profile=profile)
        self.assertEqual(limits['max_frames'], 345)
        self.assertEqual(limits['recommended_frames'], 243)
        self.assertEqual(resolve_clip_limits(self.H3, 30, {**profile, 'manual_max_frames': 158})['max_frames'], 158)
        params = {'video_model': 'h3', 'director_music_clip_seconds': None,
                  '_director_video_execution_profile': {**profile, 'music_clip_limits': limits}}
        self.assertEqual(director_pipeline.director_music_clip_limits(params, self.H3)['max_frames'], 345)

    def test_musical_cuts_keep_sections_performer_and_percussion_entrances(self):
        analysis = self.analysis(53)
        analysis['sections'] = [{'start': 0, 'end': 16, 'label': 'intro', 'energy': .2},
                                {'start': 16, 'end': 36, 'label': 'verse', 'energy': .5},
                                {'start': 36, 'end': 53, 'label': 'chorus', 'energy': .9}]
        analysis['lyrics'] = [{'start': 17, 'end': 22, 'speaker': 'a'}, {'start': 24, 'end': 30, 'speaker': 'b'}]
        analysis['percussion_activity'] = [{'start': 7.3, 'end': 53}]
        analysis['music_cues'] = [{'type': 'percussion_entry', 'time': 7.3, 'confidence': 'estimated'},
                                 {'type': 'percussion_entry', 'time': 34.8, 'confidence': 'estimated'}]
        clips = plan_capped_music_clips(analysis, maximum_seconds=243/24, fps=24, frames_minimum=124, frames_steps=17)
        cuts = [c['start'] for c in clips]
        for time in (7.3, 16, 24, 36):
            self.assertIn(time, cuts)
        self.assertGreater(len({round(c['end']-c['start'], 2) for c in clips}), 3)
        params = {'video_model': 'h3', 'pipeline_type': 'music_video', 'director_music_clip_seconds': None,
                  '_director_video_execution_profile': {'effective_max_frames': 243}}
        with patch.object(director_pipeline, '_wgp', SimpleNamespace(get_model_def=lambda _: self.H3)):
            plans, native = director_pipeline.prepare_director_timeline(params, [{} for _ in clips], clips)
            _, again = director_pipeline.prepare_director_timeline(params, plans, native)
        self.assertEqual(native, again, 'Repeated planning must not drift cuts')
        for source, clip in zip(clips, native):
            self.assertLessEqual(abs(clip['start']-source['start']), .5/24 + 1e-8)
            self.assertGreaterEqual(clip['duration_frames'], clip['output_frames'])
            self.assertLessEqual(clip['duration_frames'], 243)
        self.assertEqual(sum(c['output_frames'] for c in native), 53*24)
        self.assertEqual(next(c for c in native if c['music_cues'])['music_cues'][0]['time'], 7.3)

    def test_full_song_coverage_and_cap_survive_beats_tiny_sections_and_tail(self):
        for model in [self.H3, self.LTX]:
            for seconds in [2, 6, 10, None]:
                limits = resolve_clip_limits(model, seconds)
                for duration in [1.9, 8.2, 16.51, 29.99, 181.25]:
                    with self.subTest(model=model['fps'], seconds=seconds, duration=duration):
                        clips = plan_capped_music_clips(self.analysis(duration), maximum_seconds=limits['max_seconds'],
                                  fps=model['fps'], frames_steps=model['frames_steps'], frames_minimum=model['frames_minimum'])
                        self.assertAlmostEqual(clips[0]['start'], 0)
                        self.assertAlmostEqual(clips[-1]['end'], duration)
                        for left, right in zip(clips, clips[1:]):
                            self.assertEqual(left['end'], right['start'])
                        self.assertTrue(all(c['end']-c['start'] <= limits['max_seconds']+1e-8 for c in clips))
                        _, native = adapt_bounded_timeline([{} for _ in clips], clips, fps=model['fps'],
                                    minimum_frames=model['frames_minimum'], maximum_frames=limits['max_frames'],
                                    frame_step=model['frames_steps'], cover_source_duration=True)
                        self.assertGreaterEqual(native[-1]['end']+1e-7, duration)
                        for clip in native:
                            self.assertLessEqual(clip['duration_frames'], limits['max_frames'])
                            self.assertEqual((clip['duration_frames'] - model['frames_minimum']) % model['frames_steps'], 0)

    def test_timeline_preparation_caps_rolling_model_below_old_five_second_floor(self):
        params = {'video_model': 'ltx', 'pipeline_type': 'music_video', 'director_music_clip_seconds': 2,
                  '_director_video_execution_profile': {'fps': 25, 'effective_max_frames': 257}}
        with patch.object(director_pipeline, '_wgp', SimpleNamespace(get_model_def=lambda _: self.LTX)):
            plans, clips = director_pipeline.prepare_director_timeline(params, [{}], [{'start': 0, 'end': 16.5}])
        self.assertEqual(len(plans), len(clips))
        self.assertGreaterEqual(clips[-1]['end'], 16.5)
        self.assertTrue(all(c['duration_frames'] <= 49 for c in clips))

    def test_slower_cuts_use_the_selected_cap_instead_of_forcing_every_music_marker(self):
        analysis = self.analysis(120)
        analysis['sections'] = [
            {'start': start, 'end': end, 'label': label, 'energy': energy}
            for start, end, label, energy in ((0, 16, 'intro', .345), (16, 69, 'verse', .376),
                                             (69, 94, 'chorus', .552), (94, 120, 'verse', .602))]
        analysis['music_cues'] = [{'type': 'percussion_entry', 'time': 7.3},
                                  {'type': 'percussion_entry', 'time': 80.2}]
        analysis['lyrics'] = [{'start': 20, 'end': 24, 'speaker': 'a'},
                              {'start': 28, 'end': 32, 'speaker': 'b'}]
        counts = []
        for bias in range(-2, 3):
            clips = plan_capped_music_clips(analysis, maximum_seconds=345/24,
                                            fps=24, frames_minimum=124, frames_steps=17, energy_bias=bias)
            counts.append(len(clips))
            if bias == -2:
                self.assertEqual(len(clips), 9)
                self.assertTrue(all(c['end']-c['start'] >= 12 for c in clips))
                self.assertAlmostEqual(sum(c['end']-c['start'] for c in clips), 120)
                self.assertTrue(any(c['start'] < 16 < c['end'] for c in clips))
                cues = [cue for clip in clips for cue in clip['music_cues']]
                self.assertIn(16, [cue['time'] for cue in cues if cue['type'] == 'section_change'])
                self.assertIn(7.3, [cue['time'] for cue in cues if cue['type'] == 'percussion_entry'])
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])
        self.assertLess(counts[2], counts[4])
        self.assertEqual(counts, sorted(counts))

    def test_fast_cuts_can_subdivide_a_section_that_already_fits_one_generation(self):
        analysis = {'duration': 40, 'bpm': 120, 'sections': [
            {'start': n * 10, 'end': (n+1) * 10, 'label': 'verse', 'energy': .8} for n in range(4)]}
        normal, fast = [plan_capped_music_clips(analysis, maximum_seconds=345/24, fps=24,
                        frames_minimum=124, frames_steps=17, energy_bias=bias) for bias in (0, 2)]
        self.assertEqual(len(normal), 4)
        self.assertEqual(len(fast), 8)

    def test_longest_cuts_keep_minimum_count_after_frame_rounding_and_tail_trim(self):
        for model in (self.H3, self.LTX):
            for requested in (2, 10, 14.4, None):
                limits = resolve_clip_limits(model, requested)
                fps, cap = model['fps'], limits['max_seconds']
                for duration in (1.9, 16.51, 119.999, cap * 2, cap * 2 + .001):
                    with self.subTest(fps=fps, requested=requested, duration=duration):
                        clips = plan_capped_music_clips(self.analysis(duration), maximum_seconds=cap, fps=fps,
                                  frames_minimum=model['frames_minimum'], frames_steps=model['frames_steps'], energy_bias=-2)
                        _, native = prepare_music_timeline([{} for _ in clips], clips, fps=fps,
                                    minimum_frames=model['frames_minimum'], maximum_frames=limits['max_frames'],
                                    frame_step=model['frames_steps'])
                        total_frames = math.ceil(duration * fps - 1e-7)
                        self.assertEqual(len(native), math.ceil(total_frames / limits['max_frames']))
                        self.assertEqual(sum(c['output_frames'] for c in native), total_frames)
                        for left, right in zip(native, native[1:]):
                            self.assertEqual(left['end'], right['start'])
                        for clip in native:
                            self.assertLessEqual(clip['output_frames'], clip['duration_frames'])
                            self.assertLessEqual(clip['duration_frames'], limits['max_frames'])
                            self.assertEqual((clip['duration_frames']-model['frames_minimum']) % model['frames_steps'], 0)

    def test_older_nonvocal_rolling_project_retains_its_timeline(self):
        params = {'video_model': 'ltx', 'pipeline_type': 'music_video'}
        clips, plans = [{'start': 0, 'end': 20}], [{}]
        with patch.object(director_pipeline, '_wgp', SimpleNamespace(get_model_def=lambda _: self.LTX)):
            self.assertEqual(director_pipeline.prepare_director_timeline(params, plans, clips), (plans, clips))


if __name__ == '__main__':
    unittest.main()
