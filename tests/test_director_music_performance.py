"""Vocal ownership survives band cutaways, serialization and H3 compilation."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.director.h3_dialogue import compile_h3_official_prompt, validate_h3_prompt_contract
from services.director.music_performance import (
    constrain_music_performance,
    music_performance_direction,
)
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
                self.assertIn('Do not infer a role from camera focus', captures[-1]['system_prompt'])
                self.assertIn('A visible assigned instrumentalist plays with relaxed closed lips', captures[-1]['system_prompt'])
                self.assertIn('performance_role', captures[-1]['system_prompt'])

    def test_h3_frames_and_references_keep_cutaway_mouth_direction_after_recompile(self):
        subjects = [{'character_id': 'drummer', 'speaker_name': 'Drummer',
                     **subject('the long-haired drummer', 'instrumentalist')},
                    subject('the cymbals', 'non_performer')]
        for mode in ('i2va', 'fl2va', 'ref2va'):
            with self.subTest(mode=mode):
                kwargs = dict(
                    mode=mode,
                    duration_seconds=7,
                    references=([{
                        'type': 'image', 'path': 'frame.png', 'role': 'Drummer',
                    }] if mode == 'ref2va' else None),
                    audio_plan={'mode': 'music_driven'},
                )
                compiled, contract = compile_h3_official_prompt(
                    'The long-haired drummer plays a fast fill. Camera pushes toward the cymbals.',
                    subjects, [], **kwargs,
                )
                self.assertIn('the long-haired drummer plays with relaxed closed lips', compiled)
                self.assertNotIn('singer', compiled.casefold())
                self.assertNotIn('the cymbals plays with relaxed closed lips', compiled)
                self.assertIn('keep hands and body active', compiled)
                self.assertIn('mapped driving audio', contract)
                self.assertEqual(validate_h3_prompt_contract(compiled, mode=mode), [])
                recompiled, _ = compile_h3_official_prompt(compiled, subjects, [], **kwargs)
                self.assertEqual(
                    recompiled.count('the long-haired drummer plays with relaxed closed lips'),
                    1,
                )
                self.assertNotIn('singer continues off screen', recompiled)
                if mode != 'ref2va':
                    self.assertEqual(recompiled, compiled)

    def test_wide_band_shot_keeps_assigned_singers_including_singing_guitarist(self):
        subjects = [subject('the lead singer in red', 'vocalist'),
                    subject('the guitarist who sings backing vocals', 'vocalist'),
                    subject('the drummer in black', 'instrumentalist')]
        direction = music_performance_direction(subjects)
        self.assertIn('the lead singer in red lip-syncs only their own audible vocal part', direction)
        self.assertIn('the guitarist who sings backing vocals lip-syncs only their own audible vocal part', direction)
        self.assertNotIn('the guitarist who sings backing vocals plays with relaxed closed lips', direction)
        self.assertIn('the drummer in black plays with relaxed closed lips', direction)

    def test_wind_players_and_older_subjects_are_not_forced_into_singing_or_frozen(self):
        direction = music_performance_direction([
            subject('the trumpet player', 'instrumentalist'),
            subject('the cheering fans', 'non_vocal'),
            SubjectRef(visual_description='the unidentified person'),
        ])
        self.assertIn("the trumpet player uses the instrument's natural embouchure", direction)
        self.assertNotIn('the trumpet player plays with relaxed closed lips', direction)
        self.assertNotIn('the unidentified person', direction)
        self.assertNotIn('the cheering fans lip-sync', direction)
        self.assertIn('preserve their described action', direction)
        self.assertIsNone(SubjectRef.from_dict({'visual_description': 'a performer'}).performance_role)

    def test_story_audio_does_not_receive_band_direction(self):
        compiled, _ = compile_h3_official_prompt(
            'A person listens beside a window.', [], [],
            mode='i2va', audio_plan={'mode': 'audio_driven'},
        )
        self.assertNotIn('Vocal ownership stays', compiled)

    def test_empty_landscape_and_legacy_unknown_roles_receive_no_performer_direction(self):
        self.assertEqual(music_performance_direction(), '')
        self.assertEqual(music_performance_direction([subject('a moonlit landscape', 'non_performer')]), '')
        self.assertEqual(music_performance_direction([{'visual_description': 'an unidentified person'}], 'active'), '')

    def test_narrative_and_dance_subjects_do_not_pull_in_an_absent_band(self):
        direction = music_performance_direction(
            [subject('the dancer in a yellow coat', 'non_vocal')],
            'active',
        )
        self.assertIn('the dancer in a yellow coat remains a non-singing presence', direction)
        self.assertNotIn('singer', direction.casefold())
        self.assertNotIn('band', direction.casefold())
        self.assertNotIn('instrument', direction.casefold())

    def test_h3_narrative_shot_without_subjects_gets_no_named_performer_direction(self):
        compiled, contract = compile_h3_official_prompt(
            'A mother walks beside a moonlit lake. Camera tracks the reflected lights.',
            [],
            [],
            mode='i2va',
            audio_plan={'mode': 'music_driven', 'vocal_activity': 'unknown'},
        )
        self.assertIn('A mother walks beside a moonlit lake', compiled)
        for role in ('singer', 'vocalist', 'band', 'instrument'):
            self.assertNotIn(role, compiled.casefold())
            self.assertNotIn(role, contract.casefold())

    def test_single_singer_only_syncs_their_part_and_closes_lips_in_silent_intro(self):
        singer = subject('Mara, the assigned singer in red', 'vocalist')
        active = music_performance_direction([singer], 'active')
        silent = music_performance_direction([singer], 'silent')
        self.assertIn('Mara, the assigned singer in red lip-syncs only their own audible vocal part', active)
        self.assertIn('relaxed closed lips during instrumental gaps', active)
        self.assertIn('Mara, the assigned singer in red keeps relaxed closed lips through this interval', silent)
        self.assertNotIn('Mara', music_performance_direction([], 'active'))

    def test_legacy_global_music_boilerplate_is_removed_before_role_scoped_recompile(self):
        legacy = (
            'A close shot of a drummer. Vocal ownership stays with the assigned singer across camera cuts. '
            'Only an explicitly assigned vocalist lip-syncs, and only to their own audible vocal part. '
            'Non-singing guitarists, bassists and drummers keep their lips closed without mouthing lyrics, '
            'except for a non-singing expression explicitly requested by the user; their hands and bodies '
            'continue the instrumental performance. During an instrument-only cutaway with audible vocals, '
            'the singer continues off screen; do not transfer the vocal to the person in view or insert a singer into the shot.'
        )
        compiled, _ = compile_h3_official_prompt(
            legacy,
            [subject('the visible drummer', 'instrumentalist')],
            [],
            mode='ref2va',
            audio_plan={'mode': 'music_driven'},
        )
        self.assertNotIn('Vocal ownership stays', compiled)
        self.assertNotIn('singer continues off screen', compiled)
        self.assertIn('the visible drummer plays with relaxed closed lips', compiled)

    def test_saved_role_clause_removal_preserves_h3_fields_and_camera_action(self):
        prompt = (
            'FRAME: Start with the kit in shadow.\n\n'
            'CAMERA: Camera pushes in as the visible drummer strikes the crash cymbal, '
            'opens his mouth wide, and shouts loudly; '
            'the visible drummer keeps their mouth closed and does not sing, mouth lyrics, '
            'or lip-sync; natural body movement continues.\n\n'
            'REFERENCE: Match the supplied frame.'
        )
        result = constrain_music_performance(
            prompt,
            [subject('the visible drummer', 'instrumentalist')],
            'active',
        )
        self.assertIn('FRAME: Start with the kit in shadow.\n\nCAMERA:', result)
        self.assertIn('Camera pushes in as the visible drummer strikes the crash cymbal', result)
        self.assertIn('his lips relaxed and closed', result)
        self.assertNotIn('opens his mouth wide', result)
        self.assertNotIn('shouts loudly', result)
        self.assertIn('\n\nREFERENCE: Match the supplied frame.', result)
        self.assertNotIn('natural body movement continues', result)

    def test_ltx_final_contract_keeps_source_audio_without_adding_performers(self):
        kwargs = dict(video_model='ltx2_25_22B_distilled', model_def={},
                      pipeline_type='music_video', audio_path='song.wav')
        prompts = _apply_ltx25_music_video_sync_contract(
            ['A landscape at night.', 'Two dancers cross a room.'], **kwargs,
        )
        self.assertEqual(_apply_ltx25_music_video_sync_contract(prompts, **kwargs), prompts)
        for prompt in prompts:
            self.assertIn('Keep the supplied track as the source of sound', prompt)
            self.assertIn('The soundtrack alone does not require anyone to appear on screen', prompt)
            self.assertNotIn('off-screen singer', prompt.casefold())

    def test_planner_preserves_a_landscape_concept_without_inventing_performers(self):
        captures = []

        def generate(**kwargs):
            captures.append(kwargs)
            return json.dumps([{
                'subjects_on_screen': [],
                'video_prompt': 'A moonlit desert landscape, wind-blown sand, wide aerial view.',
                'camera_plan': {'framing': 'wide aerial'},
                'window_prompts': [],
            }])

        planner = MusicVideoPlanner(llm_generate=generate, llm_generate_streaming=generate)
        with patch(
            'services.director.planners.music_video.classify_vocal_intervals',
            return_value=['unknown'],
        ):
            plan = planner.plan(
                clips=[{'start': 0, 'end': 8, 'label': 'instrumental'}],
                scene_description='A moonlit desert landscape with wind-blown sand and no visible people.',
                video_model='ltx2_25_22B_distilled',
                shot_image_policy='direct_references',
            )
        shot = plan.shots[0]
        self.assertEqual(shot.subjects_on_screen, [])
        self.assertNotIn('singer', shot.video_prompt.casefold())
        self.assertNotIn('band', shot.video_prompt.casefold())
        self.assertIn('Do not add people or objects solely because the soundtrack contains vocals', captures[-1]['system_prompt'])
        self.assertNotIn('Stage the lead singer', captures[-1]['prompt'])


if __name__ == '__main__':
    unittest.main()
