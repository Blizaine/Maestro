"""Intent, native reference binding, and prose survive unified enhancement."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import json
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.adaptive_enhancement import adaptive_dialogue_expected, adaptive_writing_guide, draft_spoken_exchange
from services.dialogue_writing import dialogue_forbidden
from services.h3_story_ledger import _canonicalize_story_ledger, _deterministic_ledger, extract_source_events, extract_h3_source_intent
from services import llm_service


CONCEPT = 'dynamic kung fu fight scene set in the mountains with power hits and cinematic action'


class AdaptiveWritingTests(unittest.TestCase):
    def test_single_window_repairs_only_unambiguous_missing_speaker_ids(self):
        source = (
            'Adult sisters Lena and Priya stand beside one red suitcase. '
            'Lena places the ticket in Priya\'s left hand. Priya closes that hand '
            'before the boarding gate opens. Lena says, "You hold the ticket." '
            'Priya replies, "And you keep the suitcase." Add no other speech.'
        )
        draft = (
            'integrated_multimodal_description: [Shot 1] Lena places the ticket in '
            'Priya\'s left hand. Priya closes that hand before the boarding gate opens. '
            'Lena says: <d>[English] You hold the ticket.</d> Priya replies: '
            '<d>[English] And you keep the suitcase.</d> They walk through together.\n'
            'overall_soundscape: Quiet terminal ambience.\n'
            'non_diegetic_music: N/A'
        )
        expected = draft.replace(
            '<d>[English] You hold the ticket.</d>',
            '(S1) <d>[English] You hold the ticket.</d>',
        ).replace(
            '<d>[English] And you keep the suitcase.</d>',
            '(S2) <d>[English] And you keep the suitcase.</d>',
        )
        repaired = llm_service._repair_unambiguous_h3_context_speaker_ids(source, draft)
        self.assertEqual(repaired, expected)
        self.assertIn('closes that hand before the boarding gate opens', repaired)
        self.assertTrue(llm_service._h3_context_dialogue_binding_contract_satisfied(source, repaired))
        self.assertEqual(
            llm_service._repair_unambiguous_h3_context_speaker_ids(source, repaired),
            repaired,
        )

        with patch.object(llm_service, 'generate', return_value=draft) as generate, \
                patch('services.enhance_guides.get_enhance_guide', return_value='H3 guide'):
            enhanced = llm_service.enhance_prompt(
                prompt=source, mode='video', model_type='minimax_h3_fused_turbo',
                duration_seconds=10.125, planning_style='adaptive', max_new_tokens=1280,
            )
        self.assertEqual(generate.call_count, 1)
        self.assertIn('closes that hand before the boarding gate opens', enhanced)
        self.assertIn('(S1) <d>[English] You hold the ticket.</d>', enhanced)
        self.assertIn('(S2) <d>[English] And you keep the suitcase.</d>', enhanced)

    def test_single_window_speaker_id_repair_declines_unsafe_bindings(self):
        fields = lambda body: (
            f'integrated_multimodal_description: [Shot 1] {body}\n'
            'overall_soundscape: Quiet room tone.\nnon_diegetic_music: N/A'
        )
        source = 'Lena says, "Keep it." Priya replies, "I will."'
        clean = fields(
            'Lena says: <d>[English] Keep it.</d> '
            'Priya replies: <d>[English] I will.</d>'
        )
        conflicting = clean.replace(
            '<d>[English] Keep it.</d>', '(S2) <d>[English] Keep it.</d>'
        )
        self.assertEqual(
            llm_service._repair_unambiguous_h3_context_speaker_ids(source, conflicting),
            conflicting,
        )
        self.assertFalse(
            llm_service._h3_context_dialogue_binding_contract_satisfied(source, conflicting)
        )

        unsafe = (
            ('Lena says, "Keep it." Priya replies, "I will."',
             fields('Lena says: <d>[English] Keep it.</d>')),  # missing line
            ('Lena says, "Keep it." Priya replies, "I will."',
             fields('Priya replies: <d>[English] I will.</d> Lena says: <d>[English] Keep it.</d>')),
            ('Lena and Priya face each other. She says, "Keep it."',
             fields('She says: <d>[English] Keep it.</d>')),
            ('Lena and Priya say together, "Keep it."',
             fields('Lena and Priya say together: <d>[English] Keep it.</d>')),
            ('Lena says, "Keep it." Priya replies, "Keep it."',
             fields('Lena says: <d>[English] Keep it.</d> Priya replies: <d>[English] Keep it.</d>')),
            ('Lena says, "Keep it."',
             fields('Priya says: <d>[English] Keep it.</d>')),
        )
        for unsafe_source, unsafe_draft in unsafe:
            with self.subTest(source=unsafe_source, draft=unsafe_draft):
                self.assertEqual(
                    llm_service._repair_unambiguous_h3_context_speaker_ids(
                        unsafe_source, unsafe_draft
                    ),
                    unsafe_draft,
                )

    def test_adaptive_script_markers_repair_frame_writer_ids_without_rewrite(self):
        source = (
            'A concise four-turn repair conversation.\n\n'
            'Spoken script for this adaptation.\n'
            'Nia (S1) says, <d>[English] That pad rubs.</d>\n'
            'Omar (S2) says, <d>[English] What shifted?</d>\n'
            'Nia (S1) says, <d>[English] Caliper is off-center.</d>\n'
            'Omar (S2) says, <d>[English] Centering it fixes that.</d>'
        )
        draft = (
            'integrated_multimodal_description: [Shot 1] Nia (S1), in a navy apron, '
            'and Omar (S2), in a grey shirt, inspect one brake. '
            'Nia says: <d>[English] That pad rubs.</d> '
            'Omar asks: <d>[English] What shifted?</d> '
            'Nia explains: <d>[English] Caliper is off-center.</d> '
            'Omar replies: <d>[English] Centering it fixes that.</d> '
            'They center the caliper, spin the wheel, and hear the rubbing stop.\n'
            'overall_soundscape: Workshop room tone and tool clicks.\n'
            'non_diegetic_music: N/A'
        )
        repaired = llm_service._repair_unambiguous_h3_context_speaker_ids(source, draft)
        for speaker_id, words in (
            (1, 'That pad rubs.'),
            (2, 'What shifted?'),
            (1, 'Caliper is off-center.'),
            (2, 'Centering it fixes that.'),
        ):
            self.assertIn(f'(S{speaker_id}) <d>[English] {words}</d>', repaired)
        self.assertIn('center the caliper, spin the wheel, and hear the rubbing stop', repaired)
        self.assertTrue(llm_service._h3_dialogue_contract_satisfied(source, repaired))
        self.assertTrue(llm_service._h3_context_dialogue_binding_contract_satisfied(source, repaired))

    def test_pronoun_continuation_repairs_only_proven_previous_owner(self):
        source = 'Theo (S1) says, <d>[English] Thanks, you saved it.</d>'
        fields = lambda action: (
            f'integrated_multimodal_description: [Shot 1] {action}\n'
            'overall_soundscape: Rain and one relieved laugh.\n'
            'non_diegetic_music: N/A'
        )
        sound = fields(
            'Theo takes his umbrella back and exhales one short relieved laugh. '
            'He looks directly at Mina and says: <d>[English] Thanks, you saved it.</d> '
            'Mina silently nods.'
        )
        repaired = llm_service._repair_unambiguous_h3_context_speaker_ids(source, sound)
        self.assertIn('(S1) <d>[English] Thanks, you saved it.</d>', repaired)
        self.assertIn('exhales one short relieved laugh', repaired)
        self.assertTrue(llm_service._h3_context_dialogue_binding_contract_satisfied(source, repaired))

        unsafe = (
            fields('Theo stands beside Mina. She says: <d>[English] Thanks, you saved it.</d>'),
            fields('Theo and Dev wait. He says: <d>[English] Thanks, you saved it.</d>'),
            fields('Theo watches Dev take his coat. He looks at Mina and says: '
                   '<d>[English] Thanks, you saved it.</d>'),
            fields('Theo takes his umbrella. He smiles while Dev says: '
                   '<d>[English] Thanks, you saved it.</d>'),
            fields('Theo watches the man take his coat. He looks at Mina and says: '
                   '<d>[English] Thanks, you saved it.</d>'),
            fields('Dev takes his coat. He looks at Mina and says: '
                   '<d>[English] Thanks, you saved it.</d>'),
        )
        for draft in unsafe:
            with self.subTest(draft=draft):
                self.assertEqual(
                    llm_service._repair_unambiguous_h3_context_speaker_ids(source, draft),
                    draft,
                )

    def test_adaptive_source_fallback_does_not_fabricate_early_speech_clock(self):
        source = (
            'An umbrella rolls toward traffic. Mina catches it, Theo takes it back, '
            'gives one relieved laugh, then Theo (S1) says, '
            '<d>[English] Thanks.</d> Mina silently nods.'
        )
        fallback = llm_service._build_h3_context_fallback(
            source,
            has_start_image=False,
            duration_seconds=14.375,
            planning_style='adaptive',
        )
        self.assertNotIn('approximately 0.25 seconds', fallback)
        self.assertIn('after every action that causes or motivates it', fallback)
        self.assertIn('Preserve requested nonverbal reactions such as a laugh', fallback)
        self.assertIn('gives one relieved laugh', fallback)
        self.assertIn('(S1) <d>[English] Thanks.</d>', fallback)
        self.assertNotIn('all mouths remain closed before and after it', fallback)
        self.assertNotIn('no human voices, whispers, grunts', fallback)

        faithful = llm_service._build_h3_context_fallback(
            source,
            has_start_image=False,
            duration_seconds=14.375,
        )
        self.assertIn('approximately 0.25 seconds', faithful)

        ref2va = llm_service._build_h3_ref2va_tagged_fallback(
            source,
            '<Subject 1> is Theo from <Picture 1>, preserving identity.',
            duration_seconds=14.375,
            planning_style='adaptive',
        )
        self.assertNotIn('approximately 0.25 seconds', ref2va)
        self.assertNotIn('all mouths remain closed before and after it', ref2va)
        self.assertNotIn('no human voices, whispers, grunts', ref2va)
        self.assertIn('Preserve explicitly requested nonverbal reactions', ref2va)

    def test_adaptive_silent_fallback_preserves_requested_nonverbal_reaction(self):
        source = (
            'Mina catches the rolling umbrella. Theo takes it back, gives one relieved laugh, '
            'and Mina silently nods. No spoken words.'
        )
        for fallback in (
            llm_service._build_h3_context_fallback(
                source,
                has_start_image=False,
                duration_seconds=14.375,
                planning_style='adaptive',
            ),
            llm_service._build_h3_ref2va_tagged_fallback(
                source,
                '<Subject 1> is Theo from <Picture 1>, preserving identity.',
                duration_seconds=14.375,
                planning_style='adaptive',
            ),
        ):
            with self.subTest(fallback=fallback):
                self.assertIn('gives one relieved laugh', fallback)
                self.assertIn('silently nods', fallback)
                self.assertIn('Add no spoken words or speech-like vocalizations', fallback)
                self.assertIn('Preserve explicitly requested nonverbal reactions', fallback)
                self.assertNotIn('approximately 0.25 seconds', fallback)
                self.assertNotIn('all mouths remain closed before and after it', fallback)
                self.assertNotIn('no human voices, whispers, grunts', fallback)

    def test_screenplay_contract_is_shared_by_single_and_multi_window_paths(self):
        from services.h3_story_ledger import extract_locked_dialogue
        for heading in ('## Cast', '**Characters**', '__Roles__', '[Subjects]'):
            for first, second in (('Nora', 'Kai'), ('Astronomer', 'Assistant')):
                with self.subTest(heading=heading, first=first):
                    source = (f'{heading}\n{first}: a gray coat and a red scarf.\n'
                              f'{second}: a brown jacket and a wool cap.\n## Scene\n'
                              f'{first}: Is the lens ready?\n{second}: Yes. Look at that star.\n')
                    exact = [item['text'] for item in extract_locked_dialogue(source)]
                    self.assertEqual(llm_service._extract_h3_quoted_dialogue(source), exact)
                    self.assertTrue(llm_service._h3_requests_speech(source))
                    silent = ('integrated_multimodal_description: [Shot 1] They adjust a telescope.\n'
                              'overall_soundscape: Wind.\nnon_diegetic_music: N/A')
                    self.assertFalse(llm_service._h3_dialogue_contract_satisfied(source, silent))
                    compiled = llm_service._compile_h3_explicit_dialogue(source)
                    self.assertIn(f'{first}: (S1) <d>[English] Is the lens ready?</d>', compiled)
                    self.assertIn(f'{second}: (S2) <d>[English] Yes. Look at that star.</d>', compiled)
                    self.assertEqual(llm_service._extract_h3_dialogue_blocks(compiled), exact)
                    self.assertTrue(llm_service._h3_dialogue_contract_satisfied(source, compiled))
                    self.assertEqual(llm_service._compile_h3_explicit_dialogue(compiled), compiled)

    def test_profile_quotes_are_not_speech_in_single_window_contract(self):
        source = ('## Characters\nNora: "the careful one", wearing a gray coat.\n'
                  '## Scene\nNora adjusts a telescope.')
        self.assertEqual(llm_service._extract_h3_quoted_dialogue(source), [])
        self.assertFalse(llm_service._h3_requests_speech(source))

    def test_screenplay_repairs_preserve_returning_speaker_and_reference_identity(self):
        source = 'Nora: Is it ready?\nKai: It is.\nNora: Then go.'
        self.assertEqual([e['speaker_id'] for e in llm_service._extract_h3_source_dialogue_entries(source)], [1, 2, 1])
        draft = ('integrated_multimodal_description: [Shot 1] Nora and Kai wait.\n'
                 'overall_soundscape: Wind.\nnon_diegetic_music: N/A')
        repaired = llm_service._inject_missing_h3_dialogue(draft, source, ref2va=False)
        self.assertIn('Nora (S1) says exactly once: <d>[English] Then go.</d>', repaired)
        self.assertIn('Kai (S2) says exactly once: <d>[English] It is.</d>', repaired)
        refs = ('Saved character "Kai" is exactly <Subject 1>: <Picture 1> defines this character.\n'
                'Saved character "Nora" is exactly <Subject 2>: <Picture 2> defines this character.')
        entries = llm_service._extract_h3_source_dialogue_entries(source, refs)
        self.assertEqual([(e['subject_id'], e['speaker_id']) for e in entries], [(2, 1), (1, 2), (2, 1)])

    def test_silence_lists_have_consistent_scope_with_conjunctions(self):
        for restriction in ('No narration or dialogue.', 'No music and speech.',
                            'No subtitles, music, or dialogue.'):
            with self.subTest(restriction=restriction):
                source = 'A tortoise crosses the garden. ' + restriction
                self.assertTrue(dialogue_forbidden(source))
                self.assertFalse(adaptive_dialogue_expected(source))
                self.assertFalse(llm_service._h3_requests_speech(source))
        self.assertFalse(dialogue_forbidden('No music or dialogue before five seconds. Then Nora speaks.'))
        self.assertFalse(dialogue_forbidden('No music or dialogue begins later.'))
        self.assertFalse(dialogue_forbidden('Nora says, "No narration or dialogue."'))

    def test_native_output_retains_visible_words_and_does_not_invent_voice_assets(self):
        source = 'Mae hands Jules a note that reads "Platform 4". Mae says, "Keep this safe."'
        draft = ('subject_definitions: Mae and Jules.\nsummary: A handoff.\nretention_analysis: N/A\n'
                 'detailed_description: [Shot 1] Mae hands Jules a note. Mae (S1) says in an off-screen voiceover, '
                 '<d>[English] Keep this safe.</d>\noverall_soundscape: Paper.\nnon_diegetic_music: N/A')
        result = llm_service._preserve_adaptive_h3_literals(source, draft)
        self.assertIn('note displays the exact text "Platform 4"', result)
        self.assertEqual(result.count('Mae hands Jules'), 1)
        self.assertNotIn('off-screen voiceover', result)
        self.assertEqual(llm_service._extract_h3_dialogue_blocks(result), ['Keep this safe.'])
        self.assertEqual(llm_service._preserve_adaptive_h3_literals(source, result), result)
        self.assertIn('off-screen voiceover', llm_service._preserve_adaptive_h3_literals(source + ' Mae speaks in voiceover.', draft))
        self.assertNotIn('voice referenced from N/A', llm_service._preserve_adaptive_h3_literals(source, draft.replace('an off-screen voiceover', 'the voice referenced from N/A')))

    def test_exchange_stage_locks_spoken_words_and_speaker_order_for_visual_writer(self):
        prompt = 'Mina and Theo discuss why he ignored her letter, ending with Mina asking him to stay.'
        turns = [{'speaker': 'Mina', 'text': 'Ten years, Theo. Why did you never answer my letter?'},
                 {'speaker': 'Theo', 'text': 'I was ashamed. I thought you had already forgotten me.'},
                 {'speaker': 'Mina', 'text': 'I never did. Stay with me tonight. We have so much to say.'}]
        generator = Mock(return_value=json.dumps({'turns': turns}))
        result = draft_spoken_exchange(prompt, 14.375, generator, language='English')
        self.assertTrue(result.startswith(prompt))
        self.assertEqual(llm_service._extract_h3_dialogue_blocks(result), [turn['text'] for turn in turns])
        self.assertEqual(result.count('Mina (S1)'), 2)
        self.assertIn('Theo (S2)', result)
        self.assertNotRegex(result, r'\d+(?:\.\d+)?[–-]\d+(?:\.\d+)? seconds')
        self.assertIn('story and camera scheduler owns their timing', result)
        self.assertTrue(llm_service._h3_dialogue_contract_satisfied(result, result))
        generator.assert_called_once()

    def test_concise_four_turn_exchange_is_not_rejected_for_missing_filler(self):
        prompt = (
            "Adult mechanics Nia and Omar have a concise natural four-turn conversation: "
            "Nia identifies the rubbing pad, Omar asks what shifted, Nia explains the "
            "caliper is off-center, and Omar confirms the fix after centering it."
        )
        turns = [
            {'speaker': 'Nia', 'text': 'Front pad is rubbing.'},
            {'speaker': 'Omar', 'text': 'What shifted?'},
            {'speaker': 'Nia', 'text': 'The caliper is off-center.'},
            {'speaker': 'Omar', 'text': 'Centered. The rubbing stopped.'},
        ]
        generator = Mock(return_value=json.dumps({'turns': turns}))
        result = draft_spoken_exchange(prompt, 14.375, generator, language='English')
        self.assertEqual(llm_service._extract_h3_dialogue_blocks(result), [
            turn['text'] for turn in turns
        ])
        request = generator.call_args.kwargs['prompt']
        self.assertIn('Use exactly 4 concise turns and 14 spoken words', request)
        self.assertIn('allowed 8–43 words', request)

    def test_empty_silent_action_row_is_not_treated_as_failed_speech(self):
        prompt = (
            "Mina catches Theo's umbrella. Theo takes it back, laughs, and blurts one "
            "brief natural thank-you. Keep his reaction to one short line; this is not "
            "a conversation. Mina answers only with a silent nod."
        )
        generator = Mock(return_value=json.dumps({'turns': [
            {'speaker': 'Mina', 'text': ''},
            {'speaker': 'Theo', 'text': 'Thanks.'},
        ]}))
        result = draft_spoken_exchange(prompt, 14.375, generator, language='English')
        self.assertEqual(llm_service._extract_h3_dialogue_blocks(result), ['Thanks.'])
        self.assertNotIn('Mina (S', result)
        self.assertIn('Theo (S1)', result)
        generator.assert_called_once()

    def test_exchange_stage_skips_silent_and_exact_scripts_and_rejects_overlong_speech(self):
        generator = Mock()
        for prompt in [CONCEPT, 'Mina says, "Stay." Theo replies, "Yes." Only use these lines.']:
            self.assertEqual(draft_spoken_exchange(prompt, 14, generator, language='English'), prompt)
        generator.assert_not_called()
        generator.return_value = json.dumps({'turns': [{'speaker': 'Mina', 'text': 'word ' * 100}]})
        with self.assertRaisesRegex(ValueError, 'could not complete the requested dialogue'):
            draft_spoken_exchange('Mina and Theo discuss the letter.', 14, generator, language='English')
        self.assertEqual(generator.call_count, 2)

    def test_explicit_turn_count_does_not_relax_a_developed_conversation(self):
        prompt = 'Mina and Theo have a developed four-turn conversation about the missing letter.'
        short = {'turns': [
            {'speaker': 'Mina', 'text': 'Where is it?'},
            {'speaker': 'Theo', 'text': 'I looked.'},
            {'speaker': 'Mina', 'text': 'Look again.'},
            {'speaker': 'Theo', 'text': 'All right.'},
        ]}
        generator = Mock(return_value=json.dumps(short))
        with self.assertRaisesRegex(ValueError, 'could not complete the requested dialogue'):
            draft_spoken_exchange(prompt, 14.375, generator, language='English')
        self.assertEqual(generator.call_count, 2)
        self.assertIn('allowed 31–43 words', generator.call_args.kwargs['prompt'])

    def test_visual_interaction_and_camera_instructions_do_not_request_speech(self):
        examples = [CONCEPT,
            'White-Clothed Martial Monk attacks Deep Brown-Clothed Martial Monk. The camera explains the force.',
            'Two people wearing badges reading "Hello" and "How are you?" fight.',
            (Path(__file__).parent / 'fixtures/h3_silent_temple_prompt.txt').read_text(encoding='utf-8')]
        for prompt in examples:
            with self.subTest(prompt=prompt[:70]):
                self.assertFalse(adaptive_dialogue_expected(prompt))

    def test_actual_conversation_requests_and_exact_lines_remain_speech(self):
        for prompt in ['Mina and Theo discuss the letter.', 'Two friends have an affectionate conversation.',
                       'Mae says, "Keep this safe." Jules replies, "I promise."']:
            with self.subTest(prompt=prompt):
                self.assertTrue(adaptive_dialogue_expected(prompt))
        self.assertFalse(adaptive_dialogue_expected('Two friends argue through gestures. No dialogue.'))
        self.assertFalse(adaptive_dialogue_expected('Two fighters circle. No cuts, slow motion, dialogue, or magic.'))

    def test_descriptive_and_quantified_global_silence_skip_exchange_generation(self):
        prompts = (
            'A silent live-action martial-arts scene.',
            'Two rivals settle the dispute through gestures. No one speaks or mouths words.',
            'Two rivals settle the dispute through gestures. Nobody talks.',
            'A jogger catches a stroller. No baby, dialogue, collision, impossible speed, cuts, or extra rescuer.',
            'A jogger catches a stroller. No baby, collision, speech, cuts, or extra rescuer.',
            'A jogger catches a stroller. No baby, dialogue.',
            'A jogger catches a stroller. No baby, collision, or speech.',
            'Nobody talks during the entire scene.',
            'During the whole scene, no one speaks.',
            'During the entire film, nobody talks.',
        )
        generator = Mock()
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertTrue(dialogue_forbidden(prompt))
                self.assertFalse(adaptive_dialogue_expected(prompt))
                self.assertEqual(
                    draft_spoken_exchange(prompt, 14.375, generator, language='English'),
                    prompt,
                )
        generator.assert_not_called()

    def test_shared_no_list_does_not_erase_positive_or_quoted_dialogue(self):
        prompts = (
            'No baby cries, and dialogue begins after the rescue.',
            'Mara says, "No baby, dialogue, collision, or cuts." Dev answers her.',
            'No music plays while she delivers dialogue, then she leaves.',
            'No subtitles are shown as two friends give a speech, then bow.',
            'No music, but keep the dialogue, with one short reply.',
            'No hesitation as she starts talking, then laughs.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertFalse(dialogue_forbidden(prompt))
                self.assertTrue(adaptive_dialogue_expected(prompt))

    def test_local_silence_before_requested_speech_is_not_global(self):
        prompts = (
            'No one speaks until Mara asks Dev whether the door is locked.',
            'For the opening beat, nobody speaks. Then Mara asks Dev whether the door is locked.',
            'During the fight, no one speaks. After it ends, Mara asks Dev a question.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertFalse(dialogue_forbidden(prompt))
                self.assertTrue(adaptive_dialogue_expected(prompt))

    def test_quoted_speech_is_local_but_quoted_prompt_keeps_global_silence(self):
        spoken = (
            'Mara says, "No one speaks here." Dev replies that she is mistaken.',
            'Mara says, “Nobody talks during rehearsal.” Dev answers her.',
        )
        for prompt in spoken:
            with self.subTest(prompt=prompt):
                self.assertFalse(dialogue_forbidden(prompt))
                self.assertTrue(adaptive_dialogue_expected(prompt))
        wrapped_prompts = (
            '"A silent live-action martial-arts scene."',
            '“Two fighters circle. No one speaks or mouths words.”',
            'Prompt: "No one speaks."',
        )
        for prompt in wrapped_prompts:
            with self.subTest(prompt=prompt):
                self.assertTrue(dialogue_forbidden(prompt))
                self.assertFalse(adaptive_dialogue_expected(prompt))

    def test_conversation_does_not_need_canned_silence_phrases(self):
        source = 'Mina and Theo discuss the letter.'
        draft = 'Mina (S1) asks <d>[English] Why did you never write back?</d> Theo (S2) answers <d>[English] I was afraid.</d>'
        self.assertTrue(llm_service._h3_speech_timing_satisfied(source, draft, 14.375, 'adaptive'))
        self.assertFalse(llm_service._h3_speech_timing_satisfied(source, draft, 2, 'adaptive'))
        self.assertFalse(llm_service._h3_speech_timing_satisfied('A duel. No dialogue.', draft, 14.375, 'adaptive'))

    def test_dialogue_budget_revision_keeps_cameras_actions_and_exact_lines(self):
        source = 'Mina says, "Stay." Theo explains why he left. They discuss a new beginning.'
        draft = 'Camera follows Mina beside the train. Mina (S1): <d>[English] Stay.</d> Theo (S2): <d>[English] Sorry.</d> They hold hands.'
        revised = 'I thought leaving would make things easier for us. It did not. I should have answered your letter instead of hiding. Could we try again, starting with an honest conversation?'
        generator = Mock(return_value=json.dumps({'lines': ['Stay.', revised]}))
        result = llm_service._fit_adaptive_dialogue(source, draft, 14.375, generator)
        self.assertEqual(result, draft.replace('Sorry.', revised))
        generator.assert_called_once()
        self.assertIn('json_schema', generator.call_args.kwargs)

    def test_adaptive_context_does_not_wrap_source_in_creative_story_scaffolding(self):
        context = llm_service._build_enhance_user_prompt(CONCEPT, 'video', 14.375, 1, 14.375,
            model_type='minimax_h3_ref2va', planning_style='adaptive')
        self.assertNotIn('[CREATIVE', context)
        self.assertIn('No spoken exchange', context)
        self.assertTrue(context.endswith(CONCEPT))

    def test_scene_craft_is_selected_without_adding_a_style_choice(self):
        action = adaptive_writing_guide(CONCEPT)
        self.assertIn('preparation', action.lower())
        self.assertNotEqual(action, adaptive_writing_guide('A still product on a desk.'))

    def test_writing_instruction_is_not_an_event(self):
        events = extract_source_events('Two fighters clash beside a pillar. Then write the shot-for-shot choreography.')
        self.assertTrue(events)
        self.assertNotIn('shot-for-shot', str(events))
        source = 'Two martial artists fight. No cuts, slow motion, dialogue, or magic. The agile fighter dodges a punch.'
        self.assertNotIn('No cuts', str(extract_source_events(source)))
        self.assertNotIn('Two', extract_h3_source_intent(source)['cast_names'])
        self.assertIn('Eleven', extract_h3_source_intent('Eleven fights the monster.')['cast_names'])

    def test_causal_consequence_stays_with_its_strike_and_negated_pacing_stays_real_time(self):
        source = 'The agile fighter dodges a heavy punch; it cracks a stone pillar behind him.'
        events = extract_source_events(source)
        self.assertEqual(len(events), 1)
        self.assertIn('dodges', events[0]['text'])
        self.assertIn('cracks a stone pillar', events[0]['text'])
        intent = extract_h3_source_intent(source + ' No cuts, slow motion, dialogue, or magic.')
        self.assertNotIn('preserve slow-motion', intent['pacing_contract'])

    def test_ref2va_compilation_keeps_complete_camera_and_sound_phrases(self):
        from services.h3_sequence_planner import compile_h3_reference_sequence_prompts, compute_h3_sequence_clips
        clips, _ = compute_h3_sequence_clips(226)
        camera = ('The camera follows the two fighters along the edge of the courtyard, '
                  'keeping both bodies in the same wide frame while moving past the established pillar '
                  'to reveal the new fracture and the open escape route.')
        music = ('A low cello drone builds beneath the exchange, changes to a descending phrase '
                 'when the pillar cracks, and resolves into a single sustained note as they separate.')
        shots = [{'shot': index + 1, 'start_seconds': start, 'end_seconds': end,
                  'transition': 'opening composition' if not index else 'continuous reframe',
                  'framing': 'wide master', 'camera': camera, 'action': action,
                  'sound_effects': 'A sharp crack of stone.', 'dialogue': []}
                 for index, (start, end, action) in enumerate([
                     (0, 4, 'The missed punch cracks the pillar.'),
                     (4, clips[0]['duration_seconds'], 'Both fighters circle beside the damage.')])]
        plan = {'subject_definitions': 'Two fighters.', 'visual_style': camera,
                'music': music, 'clips': [{'shots': shots, 'opening_state': 'They stand beside the pillar.',
                                         'closing_state': 'They circle beside the fracture.'}]}
        prompt = compile_h3_reference_sequence_prompts(plan, clips, reference_relationships='',
                 default_retention='', task_types='reference generation')[0]['prompt']
        self.assertIn(camera.rstrip('.'), prompt)
        self.assertIn(music, prompt)
        self.assertIn('without a cut', prompt)
        self.assertNotIn('[Shot 2]', prompt)

    def test_source_event_ownership_does_not_erase_developed_prose(self):
        canonical = _deterministic_ledger(CONCEPT, segment_count=2, segment_durations=[14, 14],
            locked_dialogue=[], camera_coverage='multi_shot', reference_context='')
        candidate = deepcopy(canonical)
        candidate['beats'] = [
            {'segment': 1, 'source_event_ids': ['E1'], 'description': 'The gray-robed boxer plants his foot and drives a fist toward the kicker, who pivots beside the established stone wall.',
             'state_after': 'The kicker is beside the wall.', 'dialogue_ids': []},
            {'segment': 2, 'source_event_ids': ['E1'], 'description': 'The kicker rebounds off that wall; his heel meets the boxer’s block and both skid apart through dust.',
             'state_after': 'Both stand apart on the damaged terrace.', 'dialogue_ids': []}]
        result = _canonicalize_story_ledger(CONCEPT, canonical, candidate, locked_dialogue=[],
            segment_count=2, preserve_adaptation=True)
        self.assertEqual([b['description'] for b in result['beats']], [b['description'] for b in candidate['beats']])
        self.assertEqual([b['source_event_ids'] for b in result['beats']], [['E1'], []])
        # Legacy schedules still recover source wording.
        legacy = _canonicalize_story_ledger(CONCEPT, canonical, candidate, locked_dialogue=[], segment_count=2)
        self.assertEqual(legacy['beats'][0]['description'], CONCEPT)

    def test_camera_handoff_does_not_restore_source_over_adapted_aftermath(self):
        from services.h3_story_ledger import _camera_phase_beats
        beat = {'beat_id': 'B2', 'description': 'The agile fighter completes his evasive step as both circle the damaged pillar.',
                'state_after': 'Both circle beside the fracture.', 'source_event_ids': ['E2', 'E3'], 'dialogue_ids': []}
        events = [{'event_id': 'E2', 'text': 'A punch misses and cracks the pillar.'},
                  {'event_id': 'E3', 'text': 'Both circle beside it.'}]
        adaptive = _camera_phase_beats([beat], source_events=events, expected_dialogue_events={}, preserve_adaptation=True)
        self.assertEqual(adaptive, [beat])
        legacy = _camera_phase_beats([beat], source_events=events, expected_dialogue_events={})
        self.assertEqual(len(legacy), 2)

    def test_no_media_keeps_prompt_native_cast_and_rejects_phantom_audio(self):
        context = 'No reference media were supplied.'
        draft = ('subject_definitions: Gray wears gray robes; Gold wears gold robes.\n'
            'summary: A mountain duel.\nretention_analysis: N/A\n'
            'detailed_description: Gray blocks Gold’s kick.\noverall_soundscape: A sharp thud.\nnon_diegetic_music: N/A')
        compiled = llm_service._canonicalize_h3_ref2va_reference_fields(draft, context, CONCEPT)
        self.assertIn('Gray wears gray robes', compiled)
        self.assertTrue(llm_service._h3_ref2va_reference_contract_satisfied(compiled, context))
        for token in ['<Subject 1>', '<Picture 1>', '<Audio 1>']:
            self.assertFalse(llm_service._h3_ref2va_reference_contract_satisfied(compiled + token, context))

    def test_visual_reference_does_not_imply_a_voice_reference(self):
        context = '<Subject 1> is Mae from <Picture 1>, preserving identity.'
        draft = llm_service._build_h3_ref2va_tagged_fallback('Mae walks.', context, duration_seconds=5)
        self.assertTrue(llm_service._h3_ref2va_reference_contract_satisfied(draft, context))
        self.assertFalse(llm_service._h3_ref2va_reference_contract_satisfied(draft + ' Voice from <Audio 1>.', context))


if __name__ == '__main__':
    unittest.main()
