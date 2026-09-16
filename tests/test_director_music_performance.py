"""Vocal ownership survives band cutaways, serialization and H3 compilation."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.director.h3_dialogue import compile_h3_official_prompt, validate_h3_prompt_contract
from services.director.music_performance import music_performance_direction
from services.director.planners.music_video import MusicVideoPlanner
from services.director.schema import SubjectRef, ProductionPlan
from services.director_pipeline import _apply_ltx25_music_video_sync_contract


def subject(description, role):
    return {'visual_description': description, 'performance_role': role}


class MusicPerformanceTests(unittest.TestCase):
    def test_planner_keeps_roles_in_saved_plans_and_instructs_cutaways(self):
        captures = []
        cast = [subject('the blond lead singer', 'vocalist'),
                subject('the dark-haired guitarist', 'instrumentalist'),
                subject('the long-haired drummer', 'instrumentalist')]

        def generate(**kwargs):
            captures.append(kwargs)
            return json.dumps([{
                'subjects_on_screen': [person],
                'video_prompt': f"Medium shot of {person['visual_description']} performing on stage.",
                'camera_plan': {'framing': 'medium shot'},
                'window_prompts': [],
            } for person in cast])

        for model in ('minimax_h3_ref2va_fused_turbo', 'ltx2_25_22B_distilled'):
            with self.subTest(model=model), patch(
                'services.director.planners.music_video.classify_vocal_intervals',
                return_value=['active'] * 3,
            ):
                planner = MusicVideoPlanner(llm_generate=generate, llm_generate_streaming=generate)
                plan = planner.plan(
                    clips=[{'start': i*7, 'end': (i+1)*7, 'label': 'verse'} for i in range(3)],
                    scene_description='The blond man sings lead. The dark-haired man plays guitar. The long-haired man plays drums.',
                    video_model=model, shot_image_policy='direct_references',
                )
                restored = ProductionPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
                self.assertEqual([s.subjects_on_screen[0].performance_role for s in restored.shots],
                                 ['vocalist', 'instrumentalist', 'instrumentalist'])
                self.assertIn('Camera focus never turns an instrumentalist into a singer', captures[-1]['system_prompt'])
                self.assertIn("visible musician's mouth closed", captures[-1]['prompt'])
                self.assertIn('performance_role', captures[-1]['system_prompt'])

    def test_h3_frames_and_references_keep_cutaway_mouth_direction_after_recompile(self):
        subjects = [subject('the long-haired drummer', 'instrumentalist'),
                    subject('the cymbals', 'non_performer')]
        for mode in ('i2va', 'fl2va', 'ref2va'):
            with self.subTest(mode=mode):
                kwargs = dict(mode=mode, duration_seconds=7, audio_plan={'mode': 'music_driven'})
                compiled, contract = compile_h3_official_prompt(
                    'The long-haired drummer plays a fast fill. Camera pushes toward the cymbals.',
                    subjects, [], **kwargs,
                )
                self.assertIn('singer continues off screen', compiled)
                self.assertIn('keeps their mouth closed and does not sing', compiled)
                self.assertNotIn('the cymbals keeps their mouth closed', compiled)
                self.assertIn('natural body movement continues', compiled)
                self.assertIn('mapped driving audio', contract)
                self.assertEqual(validate_h3_prompt_contract(compiled, mode=mode), [])
                recompiled, _ = compile_h3_official_prompt(compiled, subjects, [], **kwargs)
                self.assertEqual(recompiled.count('Vocal ownership stays'), 1)

    def test_wide_band_shot_keeps_assigned_singers_including_singing_guitarist(self):
        subjects = [subject('the lead singer in red', 'vocalist'),
                    subject('the guitarist who sings backing vocals', 'vocalist'),
                    subject('the drummer in black', 'instrumentalist')]
        direction = music_performance_direction(subjects)
        self.assertIn('Assigned visible vocalist: the lead singer in red', direction)
        self.assertIn('Assigned visible vocalist: the guitarist who sings backing vocals', direction)
        self.assertNotIn('the guitarist who sings backing vocals keeps their mouth closed', direction)
        self.assertIn('the drummer in black keeps their mouth closed', direction)

    def test_wind_players_and_older_subjects_are_not_forced_into_singing_or_frozen(self):
        direction = music_performance_direction([
            subject('the trumpet player', 'instrumentalist'),
            subject('the cheering fans', 'non_vocal'),
            SubjectRef(visual_description='the unidentified person'),
        ])
        self.assertIn("the trumpet player uses the instrument's embouchure", direction)
        self.assertNotIn('the trumpet player keeps their mouth closed', direction)
        self.assertNotIn('the unidentified person keeps their mouth closed', direction)
        self.assertNotIn('the cheering fans keeps their mouth closed', direction)
        self.assertIn('non-singing expression or cheering', direction)
        self.assertIsNone(SubjectRef.from_dict({'visual_description': 'a performer'}).performance_role)

    def test_story_audio_does_not_receive_band_direction(self):
        compiled, _ = compile_h3_official_prompt(
            'A person listens beside a window.', [], [],
            mode='i2va', audio_plan={'mode': 'audio_driven'},
        )
        self.assertNotIn('Vocal ownership stays', compiled)

    def test_ltx_final_contract_keeps_offscreen_vocal_and_is_idempotent(self):
        kwargs = dict(video_model='ltx2_25_22B_distilled', model_def={},
                      pipeline_type='music_video', audio_path='song.wav')
        prompts = _apply_ltx25_music_video_sync_contract(['Guitar close-up.', 'Drum close-up.'], **kwargs)
        self.assertEqual(_apply_ltx25_music_video_sync_contract(prompts, **kwargs), prompts)
        for prompt in prompts:
            self.assertIn('singer continues off screen', prompt)
            self.assertIn('their lips closed', prompt)


if __name__ == '__main__':
    unittest.main()
