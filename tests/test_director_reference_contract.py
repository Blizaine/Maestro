"""Regression cases for saved Director reference identity and field ownership."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.director.h3_dialogue import (
    H3DialogueContractError, compile_h3_official_prompt, validate_h3_prompt_contract,
)
from services.director.music_performance import constrain_music_performance, music_performance_direction


class DirectorReferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.subjects = [
            {'visual_description': 'the Kitten, in silver armor, approaching the Dragon.', 'performance_role': 'vocalist'},
            {'visual_description': 'the Dragon, a large dark-scaled creature.', 'performance_role': 'non_vocal'},
        ]
        self.refs = [{'type': 'image', 'role': 'Kitten with pink bow standing upright holding a sword'}]

    def test_anonymous_visual_cast_binds_existing_kitten_without_pose_leak(self):
        prompt, _ = compile_h3_official_prompt(
            'The Kitten rides the Dragon over the valley.', self.subjects, [],
            mode='ref2va', references=self.refs,
        )
        self.assertIn('<Subject 1> is Kitten', prompt)
        self.assertIn('<Subject 2> is Dragon', prompt)
        self.assertNotIn('<Subject 3>', prompt)
        self.assertNotIn('standing upright holding a sword', prompt)
        self.assertIn('<Subject 1> (Kitten) rides the <Subject 2> (Dragon)', prompt)
        self.assertEqual(validate_h3_prompt_contract(prompt, mode='ref2va', subjects=self.subjects, references=self.refs), [])

    def test_unnamed_semantic_cast_ids_bind_the_visible_identity_in_prose(self):
        subjects = [
            {'character_id': 'lead_performer', 'speaker_name': '', 'visual_description': 'Kitten, silver armor'},
            {'character_id': 'mount', 'speaker_name': '', 'visual_description': 'Dragon, dark scales'},
        ]
        prompt, _ = compile_h3_official_prompt('Kitten rides Dragon.', subjects, [], mode='ref2va', references=self.refs)
        self.assertIn('<Subject 1> (Kitten) rides <Subject 2> (Dragon)', prompt)
        self.assertNotIn('is visible in the described blocking', prompt)
        self.assertNotIn('<Subject 3>', prompt)

    def test_ambiguous_identity_reference_does_not_choose_first_actor(self):
        with self.assertRaisesRegex(H3DialogueContractError, 'multiple visible subjects'):
            compile_h3_official_prompt('Two kittens play.', [
                {'speaker_name': 'Red kitten', 'visual_description': 'Red kitten, with a red bow'},
                {'speaker_name': 'Blue kitten', 'visual_description': 'Blue kitten, with a blue bow'},
            ], [], mode='ref2va', references=[{'type': 'image', 'role': 'kitten'}])

    def test_reference_names_use_word_boundaries(self):
        prompt, _ = compile_h3_official_prompt('Ann watches Anna.', [
            {'speaker_name': 'Ann', 'visual_description': 'Ann: red coat'},
            {'speaker_name': 'Anna', 'visual_description': 'Anna: blue coat'},
        ], [], mode='ref2va', references=[{'type': 'image', 'role': 'Anna'}])
        self.assertEqual(validate_h3_prompt_contract(prompt, mode='ref2va', subjects=[{'speaker_name': 'Ann'}, {'speaker_name': 'Anna'}], references=[{'type': 'image', 'role': 'Anna'}]), [])
        self.assertIn('<Subject 2> is Anna: Anna: blue coat; facial', prompt)

    def test_explicit_name_wins_over_shared_generic_visual_description(self):
        subjects = [
            {'speaker_name': 'Ann', 'visual_description': 'a woman, wearing red'},
            {'speaker_name': 'Anna', 'visual_description': 'a woman, wearing blue'},
        ]
        refs = [{'type': 'image', 'role': 'Ann, a woman in a portrait'}]
        prompt, _ = compile_h3_official_prompt('Ann greets Anna.', subjects, [], mode='ref2va', references=refs)
        self.assertNotIn('<Subject 3>', prompt)
        self.assertEqual(validate_h3_prompt_contract(prompt, mode='ref2va', subjects=subjects, references=refs), [])

    def test_distinct_full_reference_name_beats_shared_identity_noun(self):
        subjects = [
            {'speaker_name': 'Red Kitten', 'visual_description': 'Red Kitten: red bow'},
            {'speaker_name': 'Blue Kitten', 'visual_description': 'Blue Kitten: blue bow'},
        ]
        refs = [{'type': 'image', 'role': 'Red Kitten'}]
        prompt, _ = compile_h3_official_prompt('Red Kitten greets Blue Kitten.', subjects, [], mode='ref2va', references=refs)
        self.assertNotIn('<Subject 3>', prompt)
        self.assertEqual(validate_h3_prompt_contract(prompt, mode='ref2va', subjects=subjects, references=refs), [])

    def test_reviewed_cast_survives_missing_structured_metadata(self):
        source = ('subject_definitions: <Subject 1> (S1) is the single warrior kitten from <Picture 1>. '
                  '<Subject 2> is the same enormous dark-scaled dragon.\n\n'
                  'detailed_description: [Shot 1] The kitten rides the dragon through clouds.\n\n'
                  'overall_soundscape: Wind.\n\nnon_diegetic_music: Source song.')
        prompt, _ = compile_h3_official_prompt(source, [], [], mode='ref2va', references=self.refs)
        self.assertIn('<Subject 2> is the same enormous dark-scaled dragon.', prompt)
        self.assertNotIn('<Subject 3>', prompt)
        self.assertEqual(prompt.count('subject_definitions:'), 1)

    def test_recompiling_reviewed_prompt_rebuilds_audio_reference_once(self):
        refs = [*self.refs, {'type': 'audio', 'audio_intent': 'drive', 'role': 'the source song'}]
        first, _ = compile_h3_official_prompt('Kitten rides Dragon.', self.subjects, [], mode='ref2va', references=refs)
        second, _ = compile_h3_official_prompt(first, [], [], mode='ref2va', references=refs)
        definitions = second.split('summary:')[0]
        self.assertEqual(definitions.count('<Audio 1> is'), 1)
        self.assertIn('<Subject 2> is Dragon', definitions)

    def test_reviewed_extra_actor_is_preserved_with_existing_cast(self):
        source = ('subject_definitions: <Subject 1> is Kitten from <Picture 1>. '
                  '<Subject 2> is Dragon, a dark-scaled creature.\n\n'
                  'detailed_description: [Shot 1] Kitten rides Dragon.\n\n'
                  'overall_soundscape: Wind.\n\nnon_diegetic_music: Source song.')
        prompt, _ = compile_h3_official_prompt(source, self.subjects[:1], [], mode='ref2va', references=self.refs)
        self.assertIn('<Subject 2> is Dragon', prompt)
        self.assertNotIn('<Subject 3>', prompt)

    def test_validator_rejects_duplicate_reference_identity(self):
        prompt, _ = compile_h3_official_prompt('Kitten rides Dragon.', self.subjects, [], mode='ref2va', references=self.refs)
        broken = prompt.replace('summary:', '<Subject 3> is Kitten from <Picture 1>.\n\nsummary:', 1)
        errors = validate_h3_prompt_contract(broken, mode='ref2va', subjects=self.subjects, references=self.refs)
        self.assertTrue(any('must bind only' in error for error in errors), errors)

    def test_validator_rejects_global_shot_numbers_and_undefined_cast(self):
        prompt, _ = compile_h3_official_prompt('Kitten rides Dragon.', self.subjects, [], mode='ref2va', references=self.refs)
        # Change the visual field, not its summary copy.
        a, visual = prompt.split('detailed_description:', 1)
        broken = a + 'detailed_description:' + visual.replace('rides', 'rides <Subject 7> and [Shot 13] follows', 1)
        errors = validate_h3_prompt_contract(broken, mode='ref2va', subjects=self.subjects, references=self.refs)
        self.assertTrue(any('undefined subjects' in error for error in errors), errors)
        self.assertTrue(any('consecutive' in error for error in errors), errors)

    def test_shot_validation_distinguishes_reference_mentions_from_cuts(self):
        prompt, _ = compile_h3_official_prompt(
            'Kitten rides Dragon. [Shot 2] The camera follows the flight.',
            self.subjects, [], mode='ref2va', references=self.refs,
        )
        a, visual = prompt.split('detailed_description:', 1)
        prompt = a + 'detailed_description:' + visual.replace(
            'overall_soundscape:', 'Motion follows the source throughout [Shot 1].\n\noverall_soundscape:', 1,
        )
        self.assertEqual(validate_h3_prompt_contract(prompt, mode='ref2va'), [])
        a, visual = prompt.split('detailed_description:', 1)
        broken = a + 'detailed_description:' + visual.replace('[Shot 2]', '[Shot 3]')
        self.assertTrue(any('consecutive' in error for error in validate_h3_prompt_contract(broken, mode='ref2va')))

    def test_preposition_does_not_hide_a_skipped_shot_number(self):
        prompt, _ = compile_h3_official_prompt('Kitten rides Dragon. In [Shot 3], the camera follows.', self.subjects, [], mode='ref2va', references=self.refs)
        self.assertTrue(any('consecutive' in error for error in validate_h3_prompt_contract(prompt, mode='ref2va')))

    def test_audio_prose_stops_at_shot_marker_and_keeps_visual_action(self):
        prompt, _ = compile_h3_official_prompt(
            'Kitten waits. Non-diegetic music is a sweeping theme. [Shot 1] The Dragon banks over the valley.',
            self.subjects, [], mode='ref2va', references=self.refs,
        )
        music = prompt.split('non_diegetic_music:', 1)[1]
        self.assertNotIn('[Shot', music)
        self.assertIn('<Subject 2> (Dragon) banks over the valley', prompt.split('overall_soundscape:')[0])

    def test_world_anchor_is_emitted_once_across_style_opening_recompilation(self):
        kwargs = dict(mode='ref2va', references=self.refs, context_anchors=['Airborne above the valley'])
        first, _ = compile_h3_official_prompt('Kitten rides Dragon.', self.subjects, [], **kwargs)
        second, _ = compile_h3_official_prompt(first, self.subjects, [], **kwargs)
        for prompt in (first, second):
            visual = prompt.split('detailed_description:', 1)[1].split('overall_soundscape:', 1)[0]
            self.assertEqual(visual.count('Canonical identity and world:'), 1)
            self.assertEqual(visual.count('Airborne above the valley'), 1)

    def test_music_compilation_is_idempotent_and_preserves_non_song_expression(self):
        subject = {'speaker_name': 'Kitten', 'visual_description': 'Kitten in silver armor', 'performance_role': 'vocalist'}
        direction = music_performance_direction([subject], 'active')
        source = 'Kitten raises its sword. ' + direction + ' ' + direction
        first, _ = compile_h3_official_prompt(source, [subject], [], mode='ref2va', audio_plan={'mode': 'music_driven', 'vocal_activity': 'active'})
        self.assertEqual(first.count(direction), 1)
        second, _ = compile_h3_official_prompt(first, [subject], [], mode='ref2va', audio_plan={'mode': 'music_driven', 'vocal_activity': 'active'})
        self.assertEqual(second.count(direction), 1)
        cleaned = constrain_music_performance('Kitten cheers. ' + direction, [subject], 'silent', project_context='Kitten cheers victoriously.')
        self.assertIn('Kitten cheers.', cleaned)
        self.assertNotIn(direction, cleaned)


if __name__ == '__main__':
    unittest.main()
