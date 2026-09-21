"""Reference conversations retain exact words, owners, and a feasible duration."""
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from models.minimax_h3.speakers import is_h3_spoken_quote
from services import llm_service
from services.adaptive_enhancement import adaptive_dialogue_expansion_requested
from services.h3_prompt_budget import H3PromptBudgetError
from services.h3_story_ledger import (
    extract_locked_dialogue, extract_h3_source_intent, _expected_dialogue_events,
    extract_source_events, _deterministic_ledger,
    _apply_h3_filmable_shot_clock,
    _canonicalize_segment_contract, _materialize_segment,
    _infer_h3_opening_state_contract,
    _strip_planner_speech_cues,
)
from services.studio_enhancement import enhancement_request


SOURCE = '''Alex and Sam sit opposite each other at an office table. A closed cardboard box rests between them. Preserve both reference identities and their seating positions. Use natural conversational pacing and alternating medium close-ups on the current speaker. The listener reacts silently. No overlapping speech or narration. Keep the box closed throughout.

Alex says, “You said the delivery would arrive before lunch. It’s almost dinner.”

Sam replies, “It arrived. I just haven’t decided how to explain what happened.”

Alex looks at the box and asks, “Start with the box. Why is it moving?”

Sam says, “Because your automatic vacuum apparently came with a very enthusiastic raccoon.”

Alex stares at Sam. “Please tell me you didn’t give it my office.”

Sam replies, “Only temporarily. It already has a better filing system than you.”

End on Alex’s silent, disbelieving expression. No animal is shown.'''
LINES = re.findall('“([^”]+)”', SOURCE)
REFERENCES = [
    {'type': 'image', 'path': 'sam.png', 'role': 'Sam'},
    {'type': 'image', 'path': 'alex.png', 'role': 'Alex'},
]


def request(references=REFERENCES):
    payload, sequence = enhancement_request({
        'prompt': SOURCE, 'model_type': 'minimax_h3_ref2va_fused_turbo',
        'video_length': 345, 'sliding_window_size': 345,
        'minimax_h3_references': references,
    }, {'architecture': 'minimax_h3_ref2va', 'omni_reference': True, 'fps': 24})
    assert not sequence
    return payload


class ReferenceDialogueTests(unittest.TestCase):
    def test_prop_movement_is_not_speech_or_transferred_to_a_character(self):
        prop = 'The closed cardboard box rests between them, vibrating with subtle, rhythmic shifts that suggest a small creature inside.'
        source = 'Alex and Sam sit at the office table. ' + prop
        cleaned = _strip_planner_speech_cues(source)
        self.assertIn(prop, cleaned)
        self.assertNotIn('Alex is vibrating', cleaned)
        for visual in ('Her expression suggests disbelief.', 'She nods, suggesting agreement.'):
            self.assertEqual(_strip_planner_speech_cues(visual), visual)
        self.assertNotIn('they leave now', _strip_planner_speech_cues('Alex suggests they leave now.'))

    def test_muffled_physical_sounds_survive_but_muffled_speech_does_not(self):
        sound = 'Low, rhythmic cardboard rustling and faint, muffled thumps from inside the box'
        cleaned = _strip_planner_speech_cues(sound, sound_field=True)
        self.assertIn('muffled thumps from inside the box', cleaned)
        self.assertEqual(_strip_planner_speech_cues('Muffled voices and chatter.', sound_field=True), '')

    def test_reported_arrivals_do_not_create_visible_entrances(self):
        for source in (
            SOURCE,
            'Sam says, "Mara arrived yesterday." Alex listens.',
            'Sam: Mara arrived yesterday.\nAlex: I know.',
            'Sam says, <d>[English] Mara enters tomorrow.</d> Alex listens.',
        ):
            with self.subTest(source=source):
                self.assertEqual(_infer_h3_opening_state_contract(source, ['Alex', 'Sam', 'Mara']), '')

    def test_real_entrance_after_reported_arrival_keeps_its_stationary_listener(self):
        source = 'Sam says, "It arrived yesterday." Then Mara enters and approaches Alex, who is seated at a desk.'
        opening = _infer_h3_opening_state_contract(source, ['Alex', 'Sam', 'Mara'])
        self.assertIn('Mara has not yet entered', opening)
        self.assertIn('Alex is seated at a desk', opening)
        self.assertNotIn('It enters', opening)

    def test_every_quoted_turn_keeps_a_chronological_event_anchor(self):
        entries = extract_locked_dialogue(SOURCE)
        anchors = _expected_dialogue_events(SOURCE, entries)
        self.assertEqual(list(anchors), [f'D{i}' for i in range(1, 7)])
        positions = [int(event_id[1:]) for event_id in anchors.values()]
        self.assertEqual(positions, sorted(set(positions)))
        self.assertEqual(extract_h3_source_intent(SOURCE)['cast_names'], ['Alex', 'Sam'])

    def test_context_ir_code_fences_are_not_model_instructions(self):
        prompt = 'subject_definitions: Alex\nsummary: [Reference Generation] A conversation.\n'
        prompt += 'detailed_description: (S1) <d>[English] Use `one` box.</d>'
        for fence in ('```', '```text'):
            self.assertEqual(llm_service._clean_enhance_output(
                fence + '\n' + prompt + '\n```', preserve_structure=True), prompt)

    def test_global_conversation_directions_do_not_consume_separate_beats(self):
        directions = [
            'Use natural conversational pacing and alternating medium close-ups on the current speaker',
            'The listener reacts silently', 'Keep the box closed throughout',
        ]
        events = extract_source_events(SOURCE)
        global_rules = extract_h3_source_intent(SOURCE)['global_instructions']
        for direction in directions:
            self.assertIn(direction, global_rules)
            self.assertFalse(any(direction in event['text'] for event in events))
        # Named reactions and actual prop state changes remain timed actions.
        for action in ('Alex reacts silently', 'Alex closes the box', 'Keep Mara beside the door'):
            self.assertTrue(any(action in event['text'] for event in extract_source_events(action)))

    def test_supplied_exchange_balances_whole_turns_across_two_windows(self):
        ledger = _deterministic_ledger(
            SOURCE, segment_count=2, segment_durations=[14.375, 13.625],
            locked_dialogue=extract_locked_dialogue(SOURCE),
            camera_coverage='multi_shot', reference_context=request()['reference_context'],
        )
        turns = [[did for beat in ledger['beats'] if beat['segment'] == number
                  for did in beat['dialogue_ids']] for number in (1, 2)]
        self.assertEqual(turns, [['D1', 'D2', 'D3'], ['D4', 'D5', 'D6']])

    def test_complete_script_gets_direction_but_open_dialogue_still_gets_writing(self):
        from services.h3_window_planner import plan_h3_sliding_windows

        for source, needs_writing in (
            (SOURCE, False),
            (SOURCE + '\nSam then explains how he plans to remove the raccoon.', True),
        ):
            with self.subTest(needs_writing=needs_writing), patch.object(
                llm_service, 'generate', return_value='{}',
            ) as writer:
                plan_h3_sliding_windows(
                    source, model_type='minimax_h3_fused_turbo', resolution='1280x704',
                    total_frames=672, window_frames=345, overlap_frames=18, fps=24,
                    planning_style='adaptive',
                )
                fields = writer.call_args_list[0].kwargs['json_schema']['properties']
                self.assertEqual('beats' in fields, needs_writing)
                self.assertEqual('character_appearance' in fields, not needs_writing)

    def test_silent_stares_can_share_spare_time_without_squeezing_exact_speech(self):
        from services.h3_action_clock import is_h3_stationary_reaction
        from services.dialogue_writing import spoken_word_count

        dialogue = extract_locked_dialogue(SOURCE)
        catalog = {line['dialogue_id']: line for line in dialogue}
        assignments = [
            [{'description': 'Sam delivers the assigned line.', 'dialogue_ids': ['D4']}],
            [{'description': 'Alex stares at Sam, eyes narrowing; she breathes steadily.'}],
            [{'description': 'Alex delivers the assigned line.', 'dialogue_ids': ['D5']}],
            [{'description': 'Sam delivers the assigned line.', 'dialogue_ids': ['D6']}],
            [{'description': 'Alex stares silently in disbelief; she looks at Sam.'}],
        ]
        shots = [{'start_seconds': i * 2.725, 'end_seconds': (i + 1) * 2.725} for i in range(5)]
        _apply_h3_filmable_shot_clock(shots, assignments, catalog, duration=13.625, only_when_squeezed=True)
        for shot, assignment in zip(shots, assignments):
            ids = assignment[0].get('dialogue_ids', [])
            required = sum(spoken_word_count(catalog[did]['text']) / 3 + 0.2 for did in ids)
            self.assertGreaterEqual(shot['end_seconds'] - shot['start_seconds'] + 0.002, required)
        self.assertEqual(shots[-1]['end_seconds'], 13.625)
        for action in ('Alex crosses the room and stares at Sam.', 'Alex stares for three seconds.',
                       'Alex opens the box and looks inside.', 'Sam hits Alex; Alex stares back.'):
            self.assertFalse(is_h3_stationary_reaction(action))

    def test_decorative_reaction_prose_does_not_override_the_source_action_clock(self):
        from services.h3_dialogue_writing import fit_camera_dialogue

        catalog = extract_locked_dialogue(SOURCE)
        descriptions = [
            ('Sam says', ['D4']), ('Alex stares at Sam', []),
            ('Alex speaks', ['D5']), ('Sam replies', ['D6']),
            ('End on Alex’s silent, disbelieving expression', []),
        ]
        beats = [{'beat_id': f'B{i}', 'description': text, 'dialogue_ids': ids,
                  'source_event_ids': [f'E{i}'], 'state_after': text}
                 for i, (text, ids) in enumerate(descriptions, 1)]
        events = [{'event_id': f'E{i}', 'text': text} for i, (text, ids) in enumerate(descriptions, 1)]
        shots = [{'event_indices': [i], 'dialogue_ids': ids, 'action': text,
                  'start_seconds': (i - 1) * 2.725, 'end_seconds': i * 2.725}
                 for i, (text, ids) in enumerate(descriptions, 1)]
        shots[-1]['action'] = 'Alex stares in disbelief. The box sits motionless. Her composure cracks.'
        candidate = _canonicalize_segment_contract(
            {'shots': shots}, segment_number=2, duration=13.625, assigned_beats=beats,
            dialogue_catalog=catalog, opening_state='Seated at the table.', source_intent={},
            source_events=events,
        )
        segment = _materialize_segment(candidate, beats=beats, dialogue_catalog=catalog, source_events=events)
        writer = Mock(side_effect=AssertionError('Do not rewrite exact dialogue'))
        self.assertEqual(fit_camera_dialogue(SOURCE, segment, catalog, [], generate=writer, system_prompt=''), [])
        self.assertIn('composure cracks', segment['shots'][-1]['action'])
        self.assertEqual([line['text'] for shot in segment['shots'] for line in shot['dialogue']], LINES[3:])
        writer.assert_not_called()

    def test_action_beat_line_is_locked_by_both_extractors(self):
        entries = llm_service._extract_h3_source_dialogue_entries(SOURCE)
        ledger = extract_locked_dialogue(SOURCE)
        self.assertEqual([item['words'] for item in entries], LINES)
        self.assertEqual([item['text'] for item in ledger], LINES)
        self.assertEqual([item['speaker'] for item in ledger], ['Alex', 'Sam'] * 3)
        self.assertEqual([item['speaker_id'] for item in entries], [1, 2] * 3)

    def test_action_beat_does_not_promote_labels_or_ambiguous_actors_to_speech(self):
        for source in (
            'Camera turns left. "slow-build tension"',
            'Alex looks at a sign reading "Welcome home."',
            'Alex and Sam stare at each other. "Keep it."',
            'Alex stares at Sam.\n\nSTYLE: "movie-grade realism"',
            'Alex thinks. "What if it moves?"',
        ):
            with self.subTest(source=source):
                match = re.search(r'"([^\"]+)"', source)
                self.assertFalse(is_h3_spoken_quote(source, match))
                self.assertEqual(extract_locked_dialogue(source), [])

    def test_complete_script_and_overlap_direction_do_not_request_more_speech(self):
        self.assertFalse(adaptive_dialogue_expansion_requested(SOURCE))
        self.assertTrue(adaptive_dialogue_expansion_requested(
            SOURCE + '\nSam then explains how he plans to remove the raccoon.'))

    def test_no_writer_calls_to_revise_all_locked_lines(self):
        draft = '\n'.join(f'(S{1 + index % 2}) <d>[English] {line}</d>'
                          for index, line in enumerate(LINES))
        writer = Mock()
        self.assertEqual(llm_service._fit_adaptive_dialogue(SOURCE, draft, 14.375, writer), draft)
        writer.assert_not_called()

    def test_short_duration_reports_the_actual_conflict_before_writing(self):
        for model in ('minimax_h3_ref2va_fused_turbo', 'minimax_h3_fused_turbo'):
            with self.subTest(model=model), patch.object(
                llm_service, 'generate', side_effect=AssertionError('Unexpected writer call')
            ) as writer, patch('services.enhance_guides.get_enhance_guide', return_value='H3 guide'):
                with self.assertRaises(H3PromptBudgetError) as raised:
                    llm_service.enhance_prompt(
                        SOURCE, mode='video', model_type=model,
                        duration_seconds=14.375, planning_style='adaptive',
                    )
                self.assertIn('spoken words', str(raised.exception))
                self.assertIn('14.4', str(raised.exception))
                self.assertIn('43', str(raised.exception))
                self.assertIn('duration', str(raised.exception))
                writer.assert_not_called()

    def test_queued_inventory_preserves_reference_order_not_speaking_order(self):
        context = request()['reference_context']
        manifest = llm_service._parse_h3_ref2va_subject_manifest(context)
        self.assertEqual([(s['index'], s['name'], s['pictures']) for s in manifest],
                         [(1, 'Sam', ['<Picture 1>']), (2, 'Alex', ['<Picture 2>'])])
        entries = llm_service._extract_h3_source_dialogue_entries(SOURCE, context)
        self.assertEqual([e['subject_id'] for e in entries], [2, 1] * 3)
        self.assertEqual([e['speaker_id'] for e in entries], [1, 2] * 3)

    def test_queued_voice_references_keep_their_character_bindings(self):
        context = request(REFERENCES + [
            {'type': 'audio', 'path': 'alex.wav', 'role': 'Alex', 'audio_intent': 'voice'},
            {'type': 'audio', 'path': 'sam.wav', 'role': 'Sam', 'audio_intent': 'voice'},
            {'type': 'audio', 'path': 'music.wav', 'role': 'Music', 'audio_intent': 'style'},
        ])['reference_context']
        manifest = llm_service._parse_h3_ref2va_subject_manifest(context)
        self.assertEqual([(s['name'], s['audios']) for s in manifest],
                         [('Sam', ['<Audio 2>']), ('Alex', ['<Audio 1>'])])

    def test_fallback_keeps_every_line_and_correct_reference_speaker(self):
        context = request()['reference_context']
        draft = llm_service._build_h3_ref2va_tagged_fallback(
            SOURCE, context, duration_seconds=28, planning_style='adaptive')
        self.assertEqual(llm_service._extract_h3_dialogue_blocks(draft), LINES)
        self.assertIn('<Subject 2> (S1)', draft)
        self.assertIn('<Subject 1> (S2)', draft)
        self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(SOURCE, draft, context))
        self.assertTrue(llm_service._h3_ref2va_reference_contract_satisfied(draft, context))

    def test_nonidentity_reference_roles_survive_canonicalization(self):
        context = request(REFERENCES + [
            {'type': 'image', 'path': 'office.png', 'role': 'Office', 'image_intent': 'scene'},
            {'type': 'image', 'path': 'style.png', 'role': 'Look', 'image_intent': 'style'},
            {'type': 'image', 'path': 'layout.png', 'role': 'Layout', 'image_intent': 'composition'},
        ])['reference_context']
        draft = llm_service._build_h3_ref2va_tagged_fallback(
            SOURCE, context, duration_seconds=28, planning_style='adaptive')
        self.assertIn('environment and location', draft)
        self.assertIn('visual treatment', draft)
        self.assertIn('composition and blocking reference', draft)
        self.assertTrue(llm_service._h3_ref2va_reference_contract_satisfied(draft, context))

    def test_authored_speech_uses_only_its_uploaded_voice_reference(self):
        source = 'Alex says, "It arrived."'
        draft = ('subject_definitions: invented\nsummary: Conversation\n'
                 'retention_analysis: invented\ndetailed_description: '
                 '<Subject 2> (S2) says in the voice referenced from <Audio 9>, '
                 '<d>[English] It arrived.</d>\noverall_soundscape: Office hum.\n'
                 'non_diegetic_music: N/A')
        for references, expected in (
            (REFERENCES, 'in their own natural voice'),
            (REFERENCES + [{'type': 'audio', 'path': 'alex.wav', 'role': 'Alex',
                            'audio_intent': 'voice'}], 'in the voice referenced from <Audio 1>'),
        ):
            with self.subTest(expected=expected):
                context = request(references)['reference_context']
                normalized = llm_service._canonicalize_h3_ref2va_reference_fields(draft, context, source)
                normalized = llm_service._canonicalize_h3_ref2va_dialogue_speakers(normalized, source, context)
                self.assertIn(expected, normalized)
                self.assertNotIn('<Audio 9>', normalized)
                self.assertEqual(llm_service._extract_h3_dialogue_blocks(normalized), ['It arrived.'])
                self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(
                    source, normalized, context))


if __name__ == '__main__':
    unittest.main()
