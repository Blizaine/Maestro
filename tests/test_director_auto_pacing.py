import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'app'))
from services.director_music_timing import resolve_music_pacing_bias, plan_capped_music_clips

class DirectorAutoPacingTests(unittest.TestCase):
    def test_auto_is_longer_for_h3_and_neutral_for_other_models(self):
        self.assertEqual(resolve_music_pacing_bias(None, {'architecture':'minimax_h3_ref2va'}), -2)
        self.assertEqual(resolve_music_pacing_bias(None, {'architecture':'minimax_h3'}), -2)
        self.assertEqual(resolve_music_pacing_bias(None, {'architecture':'ltx2_25'}), 0)
        self.assertEqual(resolve_music_pacing_bias(None, {'architecture':'minimax_h3_voice_audio','audio_only':True}), 0)
        self.assertEqual(resolve_music_pacing_bias(None), 0)
        for bias in [-2,-1,0,1,2]:
            self.assertEqual(resolve_music_pacing_bias(bias, {'architecture':'minimax_h3_ref2va'}), bias)

    def test_auto_covers_song_with_fewer_bounded_clips(self):
        duration = 255
        analysis = {'duration':duration,'bpm':120,
                    'beats':[{'time':n/2,'strength':.5} for n in range(duration*2)],
                    'sections':[{'start':0,'end':35,'label':'intro','energy':.2},
                                {'start':35,'end':120,'label':'verse','energy':.5},
                                {'start':120,'end':255,'label':'chorus','energy':.9}],
                    'music_cues':[{'type':'percussion_entry','time':35}]}
        args = dict(maximum_seconds=345/24,fps=24,frames_steps=17,frames_minimum=124)
        auto = plan_capped_music_clips(analysis,energy_bias=resolve_music_pacing_bias(None,{'architecture':'minimax_h3'}),**args)
        neutral = plan_capped_music_clips(analysis,energy_bias=0,**args)
        self.assertEqual(len(auto), math.ceil(duration/(345/24)))
        self.assertLess(len(auto),len(neutral))
        self.assertEqual(auto[0]['start'],0)
        self.assertEqual(auto[-1]['end'],duration)
        self.assertAlmostEqual(sum(c['end']-c['start'] for c in auto),duration)
        self.assertTrue(all(c['duration_frames']<=345 for c in auto))
        self.assertTrue(all(a['end']==b['start'] for a,b in zip(auto,auto[1:])))
        self.assertTrue(any(c['music_cues'] for c in auto),'Musical cues remain inside longer shots')

if __name__ == '__main__': unittest.main()
