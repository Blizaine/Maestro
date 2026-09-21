"""Reported brief parsing and final speech checks must agree with rendering."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.adaptive_enhancement import adaptive_dialogue_expansion_requested, is_writing_instruction
from services.h3_dialogue_writing import fit_camera_dialogue
from services.h3_story_ledger import (
    _camera_event_card_schema, _camera_phase_beats, _canonicalize_story_ledger, _deterministic_ledger,
    _h3_ordered_relation_pairs, _ledger_schema, _strip_planner_speech_cues, _find_spoken_verb,
    extract_h3_source_intent, extract_locked_dialogue, extract_source_events,
    ledger_violations, segment_violations,
)


REPORTED = (
    "expand this prompt. An older man is standing outside in a neighborhood. "
    "Near him is Sydney Sweeney with reference image. She walks up to him and pats his shoulder. "
    "the man then says in surprise 'oh shoot! I know you!’. As she laughs."
)


class EnhancementReliabilityTests(unittest.TestCase):
    def test_transcript_preservation_and_no_addition_are_not_final_story_events(self):
        for note in ('Preserve these exact lines. Do not add other dialogue.',
                     'Keep the supplied words. Never invent additional speech.',
                     "Preserve the wording. Don't add any extra narration."):
            source = 'Iris says "Ready." Omar closes the door. ' + note
            events = extract_source_events(source)
            self.assertEqual(len(events), 2, events)
            self.assertIn('closes the door', events[-1]['text'])
            self.assertEqual(extract_locked_dialogue(source)[0]['text'], 'Ready.')

    def test_character_can_speak_a_writing_instruction_verbatim(self):
        source = 'Iris says "Do not add other dialogue." Omar closes the door.'
        self.assertEqual(extract_locked_dialogue(source)[0]['text'], 'Do not add other dialogue.')
        events = extract_source_events(source)
        self.assertEqual(len(events), 2)
        # Spoken words live in the immutable catalogue, while the timeline
        # retains their speaker/action anchor.
        self.assertEqual(events[0]['text'], 'Iris says')

    def test_named_subject_appositives_are_not_separate_actions(self):
        source = 'Two pilots, Iris and Omar, are checking the cockpit.'
        self.assertEqual(self.physical_errors(source, [
            'Iris checks the cockpit gauges while Omar checks the overhead controls.'
        ]), [])
        self.assertTrue(self.physical_errors(source, ['An empty hangar remains still.']))

    def test_spoken_advice_is_not_a_required_visual_prop(self):
        source = 'Iris offers Omar one helpful, practical tip about flying.'
        self.assertIsNotNone(_find_spoken_verb(source))
        self.assertNotIn('tip', _strip_planner_speech_cues(source))
        self.assertEqual(self.physical_errors(source, ['Iris gestures warmly toward Omar.']), [])
        physical = 'Iris gives Omar a toolbox.'
        self.assertIsNone(_find_spoken_verb(physical))
        self.assertIn('toolbox', _strip_planner_speech_cues(physical))

    def test_dialogue_writing_directions_do_not_consume_story_time(self):
        for note in ('Write a gentle conversation with natural reactions.',
                     'Develop lively dialogue with distinct voices.'):
            self.assertTrue(is_writing_instruction(note))
            events = extract_source_events('Iris enters the hangar. ' + note + ' Omar closes the door.')
            self.assertEqual(len(events), 2)
            self.assertFalse(any('conversation' in event['text'] or 'dialogue' in event['text'] for event in events))
        self.assertFalse(is_writing_instruction('Iris writes a conversation in her notebook.'))

    def test_ai_acting_notes_do_not_become_spoken_lines_or_leave_id_gaps(self):
        source = 'Maya talks to Jonah, then both enter the cafe. Maya says "(Smiling)".'
        locked = extract_locked_dialogue(source)
        canonical = _deterministic_ledger(source, segment_count=1, segment_durations=[14.4],
                    locked_dialogue=locked, camera_coverage='multi_shot', reference_context='')
        draft = deepcopy(canonical)
        draft['generated_dialogue'] = [
            {'speaker': 'Jonah', 'text': '(Gesturing to the entrance)', 'segment': 1},
            {'speaker': 'Maya', 'text': 'Let us go inside.', 'segment': 1},
            {'speaker': 'Jonah', 'text': '[Nods]', 'segment': 1},
            {'speaker': 'Jonah', 'text': 'After you.', 'segment': 1},
        ]
        ledger = _canonicalize_story_ledger(source, canonical, draft, locked_dialogue=locked,
                    segment_count=1, allow_generated_dialogue=True, preserve_adaptation=True)
        self.assertEqual([line['text'] for line in ledger['generated_dialogue']],
                         ['Let us go inside.', 'After you.'])
        self.assertEqual([line['dialogue_id'] for line in ledger['generated_dialogue']], ['D2', 'D3'])
        self.assertEqual(locked[0]['text'], '(Smiling)')

    def physical_errors(self, source, actions):
        events = extract_source_events(source)
        self.assertEqual(len(events), len(actions))
        beats = [{'beat_id': f'B{i}', 'segment': 1, 'description': event['text'],
                  'source_event_ids': [event['event_id']], 'dialogue_ids': []}
                 for i, event in enumerate(events, 1)]
        segment = {'segment': 1, 'semantic_actions': True, 'opening_state': 'Outside a neighborhood cafe.',
                   'closing_state': actions[-1],
                   'shots': [{'shot': i, 'beat_ids': [f'B{i}'], 'action': action, 'dialogue': [],
                              'start_seconds': (i - 1) * 4, 'end_seconds': i * 4}
                             for i, action in enumerate(actions, 1)]}
        return segment_violations(source, segment, segment_number=1, duration=len(actions)*4,
                                  assigned_beats=beats, dialogue_catalog=[])

    def test_reported_mixed_quote_is_one_immutable_line_owned_by_the_man(self):
        dialogue = extract_locked_dialogue(REPORTED)
        self.assertEqual([(d['speaker'], d['text']) for d in dialogue],
                         [('older man', 'oh shoot! I know you!')])
        self.assertEqual(extract_h3_source_intent(REPORTED)['cast_names'],
                         ['Sydney Sweeney', 'older man'])
        self.assertFalse(adaptive_dialogue_expansion_requested(REPORTED))
        events = extract_source_events(REPORTED)
        self.assertEqual(len(events), 5)
        self.assertTrue(any('man says in surprise' in e['text'] for e in events))
        self.assertFalse(any('know you' in e['text'] or e['text'] == 'the man' for e in events))

    def test_quote_styles_preserve_apostrophes_and_multisentence_transcripts(self):
        for start, end in [('"', '"'), ('“', '”'), ("'", "'"), ('‘', '’'), ("'", '’')]:
            for words in ("Don't move. I'll be back!", 'Don’t move. I’ll be back!'):
                with self.subTest(start=start, words=words):
                    lines = extract_locked_dialogue(f'Maya says {start}{words}{end}. Leo smiles.')
                    self.assertEqual([(d['speaker'], d['text']) for d in lines], [('Maya', words)])
        self.assertEqual(extract_locked_dialogue("Maya's jacket is torn. Use 'slow build' lighting. No dialogue."), [])

    def test_bare_role_does_not_take_the_nearby_named_listeners_identity(self):
        source = "An elderly woman waits with Maya. The woman then whispers 'Stay here.'"
        self.assertEqual(extract_locked_dialogue(source)[0]['speaker'], 'elderly woman')
        ambiguous = "An older man meets a younger man. The man says 'Hello.'"
        self.assertEqual(extract_locked_dialogue(ambiguous)[0]['speaker'], 'man')

    def test_namesakes_and_past_history_are_not_two_visible_actions(self):
        for text in ('Maya greets Jonah after years apart.',
                     'Jonah explains that the shop is named after his grandmother.',
                     'Jonah named it after his grandmother.'):
            self.assertEqual(_h3_ordered_relation_pairs(text), [])
        self.assertTrue(_h3_ordered_relation_pairs('Maya closes the door after Jonah enters the room.'))

    def test_arrival_and_greeting_depict_meeting_without_copying_generic_roles(self):
        source = 'Outside a neighborhood cafe, two old friends meet after years apart.'
        self.assertEqual(self.physical_errors(source, [
            'Maya spots Jonah approaching and steps forward to greet him. Jonah smiles as he arrives beside her.'
        ]), [])
        self.assertTrue(self.physical_errors(source, ['A waiter wipes an empty table.']))
        self.assertEqual(self.physical_errors(source, [
            'Maya spots Jonah. They smile and walk towards each other, closing the distance.'
        ]), [])

    def test_meeting_does_not_preview_a_later_joint_departure(self):
        source = ('Maya meets Jonah outside a cafe. They discuss their plans. '
                  'Both walk through the open cafe door together.')
        event = extract_source_events(source)[0]
        beat = {'beat_id': 'B1', 'segment': 1, 'description': event['text'],
                'source_event_ids': [event['event_id']], 'dialogue_ids': []}
        segment = {'segment': 1, 'semantic_actions': True, 'opening_state': 'Outside a cafe.',
                   'closing_state': 'They stand facing one another outside.',
                   'shots': [{'shot': 1, 'beat_ids': ['B1'], 'dialogue': [],
                              'start_seconds': 0, 'end_seconds': 10,
                              'action': 'Maya spots Jonah; they both smile and walk towards each other, closing the gap.'}]}
        def errors():
            return segment_violations(source, segment, segment_number=1, duration=10,
                                      assigned_beats=[beat], dialogue_catalog=[])
        self.assertEqual(errors(), [])
        segment['shots'][0]['action'] = 'Maya greets Jonah, then both walk through the open cafe door together.'
        self.assertTrue(any('previews later source event' in error for error in errors()))

    def test_story_schema_leaves_room_for_source_actions_and_reactions(self):
        schema = _ledger_schema(2, source_event_count=6, locked_dialogue_count=0,
                                allow_generated_dialogue=True)
        beats = schema['properties']['beats']
        self.assertGreaterEqual(beats['maxItems'], 8)
        self.assertEqual(beats['items']['properties']['source_event_ids']['items']['enum'],
                         ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'])

    def test_speaking_window_does_not_multiply_each_reaction_into_four_cuts(self):
        beats = [{'beat_id': 'B1', 'description': 'Jonah explains.', 'dialogue_ids': ['D1']},
                 {'beat_id': 'B2', 'description': 'Maya smiles.', 'dialogue_ids': []}]
        schema = _camera_event_card_schema(1, beats)
        self.assertEqual(schema['properties']['event_cards']['properties']['event_2']
                         ['properties']['phases']['maxItems'], 1)
        beats[0]['dialogue_ids'] = []
        silent = _camera_event_card_schema(1, beats)
        self.assertEqual(silent['properties']['event_cards']['properties']['event_2']
                         ['properties']['phases']['maxItems'], 4)

    def test_missing_source_action_gets_specific_repair_feedback(self):
        source = 'Maya meets Jonah. Maya opens the door. Both enter together.'
        ledger = _deterministic_ledger(source, segment_count=2, segment_durations=[14.4, 14.4],
                                      locked_dialogue=[], camera_coverage='multi_shot', reference_context='')
        for beat in ledger['beats']:
            beat['source_event_ids'] = [eid for eid in beat['source_event_ids'] if eid != 'E3']
        errors = ledger_violations(source, ledger, segment_count=2, locked_dialogue=[], expect_dialogue=False)
        self.assertTrue(any("missing=['E3']" in error for error in errors))

    def test_connective_answer_stays_before_its_anchored_reaction(self):
        events = [{'event_id': 'E1', 'text': 'Jonah explains the shop name.'},
                  {'event_id': 'E2', 'text': 'Maya is delighted.'}]
        beat = {'beat_id': 'B1', 'segment': 1, 'source_event_ids': ['E1', 'E2'],
                'dialogue_ids': ['D1', 'D2'], 'description': 'They discuss the bookshop.'}
        phases = _camera_phase_beats([beat], source_events=events,
                                    expected_dialogue_events={'D2': 'E2'}, preserve_adaptation=True)
        self.assertEqual([did for phase in phases for did in phase['dialogue_ids']], ['D1', 'D2'])

    def test_followup_question_cannot_jump_back_before_an_authored_answer(self):
        source = 'Maya asks Jonah about his shop. They discuss how he opened it. They enter the cafe.'
        canonical = _deterministic_ledger(source, segment_count=2, segment_durations=[14.4, 14.4],
                    locked_dialogue=[], camera_coverage='multi_shot', reference_context='')
        draft = deepcopy(canonical)
        draft['beats'] = [{'segment': 1 if i == 1 else 2, 'source_event_ids': [f'E{i}'],
                           'dialogue_ids': [], 'description': event['text']}
                          for i, event in enumerate(extract_source_events(source), 1)]
        draft['generated_dialogue'] = [
            {'speaker': 'Maya', 'text': 'Did the shop open?', 'segment': 1, 'source_event_id': 'E1'},
            {'speaker': 'Jonah', 'text': 'Yesterday! The shelves arrived just in time.', 'segment': 2, 'source_event_id': 'E2'},
            {'speaker': 'Maya', 'text': 'Who helped you install them?', 'segment': 1, 'source_event_id': 'E1'},
            {'speaker': 'Jonah', 'text': 'My brother did.', 'segment': 2, 'source_event_id': 'E2'},
        ]
        for _ in range(2):
            draft = _canonicalize_story_ledger(source, canonical, draft, locked_dialogue=[],
                    segment_count=2, allow_generated_dialogue=True, preserve_adaptation=True)
            self.assertEqual([did for beat in draft['beats'] for did in beat['dialogue_ids']],
                             ['D1', 'D2', 'D3', 'D4'])
            self.assertEqual([line['segment'] for line in draft['generated_dialogue']], [1, 2, 2, 2])
            self.assertEqual(draft['beats'][-1]['source_event_ids'], ['E3'])

    def test_supporting_reaction_keeps_story_order_despite_stale_window_label(self):
        source = 'Maya arrives. Jonah greets Maya. Both enter the cafe.'
        canonical = _deterministic_ledger(source, segment_count=2, segment_durations=[14.4, 14.4],
                    locked_dialogue=[], camera_coverage='multi_shot', reference_context='')
        draft = deepcopy(canonical)
        draft['beats'] = [
            {'segment': 1, 'source_event_ids': ['E1'], 'description': 'Maya arrives.'},
            {'segment': 2, 'source_event_ids': ['E2'], 'description': 'Jonah greets Maya.'},
            {'segment': 1, 'source_event_ids': [], 'description': 'Maya smiles warmly.'},
            {'segment': 2, 'source_event_ids': ['E3'], 'description': 'Both enter the cafe.'},
        ]
        result = _canonicalize_story_ledger(source, canonical, draft, locked_dialogue=[],
                    segment_count=2, allow_generated_dialogue=True, preserve_adaptation=True)
        self.assertEqual([beat['segment'] for beat in result['beats']], [1, 2, 2, 2])
        self.assertEqual([eid for beat in result['beats'] for eid in beat['source_event_ids']],
                         ['E1', 'E2', 'E3'])

    def test_only_backward_window_labels_are_repaired_without_discarding_ai_staging(self):
        source = 'Maya arrives. Jonah greets Maya. Maya smiles. Both enter the cafe.'
        canonical = _deterministic_ledger(source, segment_count=2, segment_durations=[14.4, 14.4],
                    locked_dialogue=[], camera_coverage='multi_shot', reference_context='')
        draft = deepcopy(canonical)
        draft['beats'] = [
            {'segment': segment, 'source_event_ids': [f'E{i}'],
             'description': f"{event['text']} A warm afternoon breeze moves the leaves."}
            for i, (segment, event) in enumerate(zip([1, 2, 1, 2], extract_source_events(source)), 1)
        ]
        result = _canonicalize_story_ledger(source, canonical, draft, locked_dialogue=[],
                    segment_count=2, allow_generated_dialogue=True, preserve_adaptation=True)
        self.assertTrue(result['_timing_labels_repaired'])
        self.assertEqual([beat['segment'] for beat in result['beats']], [1, 1, 2, 2])
        self.assertEqual([beat['description'] for beat in result['beats']],
                         [beat['description'] for beat in draft['beats']])
        self.assertEqual(ledger_violations(source, result, segment_count=2,
                         locked_dialogue=[], expect_dialogue=False), [])

    def test_ongoing_conversation_keeps_local_turns_without_replaying_source_event(self):
        source = ('Maya and Jonah have a warm conversation about how he finally opened the shop '
                  'and why he named it after his grandmother. Jonah invites Maya inside. Both enter the cafe.')
        canonical = _deterministic_ledger(source, segment_count=2, segment_durations=[14.4, 14.4],
                                         locked_dialogue=[], camera_coverage='multi_shot', reference_context='')
        draft = deepcopy(canonical)
        draft['beats'] = [
            {'segment': 1, 'source_event_ids': ['E1'], 'dialogue_ids': [], 'description': 'Jonah explains the shop name.'},
            {'segment': 2, 'source_event_ids': ['E1'], 'dialogue_ids': [], 'description': 'Maya reacts warmly to the tribute.'},
            {'segment': 2, 'source_event_ids': ['E2'], 'dialogue_ids': [], 'description': 'Jonah invites her in.'},
            {'segment': 2, 'source_event_ids': ['E3'], 'dialogue_ids': [], 'description': 'Both enter the cafe.'},
        ]
        draft['generated_dialogue'] = [
            {'speaker': 'Jonah', 'language': 'English', 'text': "Grandma loved books. I named it Rose's Corner.", 'segment': 1, 'source_event_id': 'E1'},
            {'speaker': 'Maya', 'language': 'English', 'text': 'She would be proud.', 'segment': 2, 'source_event_id': 'E1'},
            {'speaker': 'Jonah', 'language': 'English', 'text': 'Come inside.', 'segment': 2, 'source_event_id': 'E2'},
        ]
        def canonicalize(candidate):
            return _canonicalize_story_ledger(source, canonical, candidate, locked_dialogue=[],
                                             segment_count=2, allow_generated_dialogue=True, preserve_adaptation=True)
        ledger = canonicalize(draft)
        for candidate in (ledger, canonicalize(ledger)):
            self.assertEqual([eid for b in candidate['beats'] for eid in b['source_event_ids']], ['E1', 'E2', 'E3'])
            self.assertEqual(candidate['beats'][1]['dialogue_ids'], ['D2'])
            self.assertEqual(candidate['beats'][2]['dialogue_ids'], ['D3'])
            self.assertEqual(ledger_violations(source, candidate, segment_count=2, locked_dialogue=[],
                                             expect_dialogue=True, allow_generated_dialogue=True), [])

    def test_connective_lines_stay_between_neighbors_without_preempting_entrance(self):
        events = [{'event_id': 'E1', 'text': 'Maya enters the cafe.'},
                  {'event_id': 'E2', 'text': 'Maya asks about the bookshop.'},
                  {'event_id': 'E3', 'text': 'Jonah explains the name.'}]
        beat = {'beat_id': 'B1', 'segment': 1, 'source_event_ids': ['E1', 'E2', 'E3'],
                'dialogue_ids': ['D1', 'D2', 'D3', 'D4', 'D5'], 'description': 'They discuss the bookshop.'}
        phases = _camera_phase_beats([beat], source_events=events,
                                    expected_dialogue_events={'D2': 'E2', 'D4': 'E3'})
        self.assertEqual(phases[0]['dialogue_ids'], [])
        self.assertEqual([phase['dialogue_ids'] for phase in phases],
                         [[], ['D1', 'D2', 'D3'], ['D4', 'D5']])

    def test_conversation_subject_is_not_an_unfilmed_physical_action(self):
        source = ('They have a warm, natural conversation about how he finally opened it '
                  'and why he named it after his grandmother. Maya opens the cafe door.')
        self.assertEqual(self.physical_errors(source, [
            'Jonah gestures, looking proud; Maya listens intently with a warm smile.',
            'Maya pushes open the cafe door.',
        ]), [])
        self.assertTrue(self.physical_errors(source, ['Both nod.', 'Rain falls on the pavement.']))

    def test_single_quoted_transcript_is_not_duplicated_into_visual_action(self):
        cleaned = _strip_planner_speech_cues("Maya says 'Don't worry. I'll help!' She opens the gate.")
        self.assertNotIn("Don't worry", cleaned)
        self.assertNotIn("I'll help", cleaned)
        self.assertIn('opens the gate', cleaned)

    def test_unequal_beat_counts_are_not_a_story_fidelity_failure(self):
        source = ('Maya enters the room. She closes the door. She sits at the desk. '
                  'She opens a book. She reads the page. She smiles.')
        events = extract_source_events(source)
        ledger = _deterministic_ledger(source, segment_count=2, segment_durations=[14.4, 14.4],
                    locked_dialogue=[], camera_coverage='multi_shot', reference_context='')
        ledger['beats'] = [{'beat_id': f'B{i}', 'segment': 1 if i <= 4 else 2,
                           'description': event['text'], 'state_after': event['text'],
                           'source_event_ids': [event['event_id']], 'dialogue_ids': []}
                          for i, event in enumerate(events, 1)]
        self.assertEqual(ledger_violations(source, ledger, segment_count=2,
                         locked_dialogue=[], expect_dialogue=False), [])

    def test_ending_instruction_is_not_an_extra_character(self):
        source = 'Maya meets Jonah. Finish with both entering the cafe.'
        self.assertEqual(extract_h3_source_intent(source)['cast_names'], ['Maya', 'Jonah'])
        self.assertTrue(any('entering the cafe' in e['text'] for e in extract_source_events(source)))

    def test_already_fitting_opening_line_keeps_its_clock(self):
        catalog = [{'dialogue_id': 'D1', 'text': 'Hello there.', 'speaker': 'Maya'}]
        segment = {'segment': 1, 'shots': [
            {'shot': 1, 'action': 'Maya walks up the long winding garden path and opens the heavy gate.',
             'start_seconds': 0, 'end_seconds': 2, 'dialogue': []},
            {'shot': 2, 'action': 'Maya smiles.', 'start_seconds': 2, 'end_seconds': 6,
             'dialogue': [{'dialogue_id': 'D1'}]},
        ]}
        original = deepcopy(segment)
        writer = Mock(side_effect=AssertionError('A fitting exact line needs no rewrite'))
        self.assertEqual(fit_camera_dialogue('Maya greets Leo.', segment, catalog, [],
                         generate=writer, system_prompt=''), [])
        self.assertEqual(segment, original)
        writer.assert_not_called()

    def test_fallback_speech_borrows_available_time_before_asking_for_copyedit(self):
        catalog = [{'dialogue_id': 'D1', 'text': 'Please come over here and look at this.', 'speaker': 'Maya'}]
        segment = {'segment': 1, 'shots': [
            {'shot': 1, 'action': 'Maya smiles.', 'start_seconds': 0, 'end_seconds': 1,
             'dialogue': [{'dialogue_id': 'D1'}]},
            {'shot': 2, 'action': 'Leo nods.', 'start_seconds': 1, 'end_seconds': 8, 'dialogue': []},
        ]}
        writer = Mock(side_effect=AssertionError('This script fits'))
        self.assertEqual(fit_camera_dialogue('Maya greets Leo.', segment, catalog, [],
                         generate=writer, system_prompt=''), [])
        self.assertGreater(segment['shots'][0]['end_seconds'], 3)
        self.assertEqual(catalog[0]['text'], 'Please come over here and look at this.')
        writer.assert_not_called()

    def test_a_genuinely_overlong_exact_line_still_requires_review(self):
        catalog = [{'dialogue_id': 'D1', 'text': ' '.join(['word'] * 30), 'speaker': 'Maya'}]
        segment = {'segment': 1, 'shots': [
            {'shot': 1, 'action': 'Maya smiles.', 'start_seconds': 0, 'end_seconds': 2,
             'dialogue': [{'dialogue_id': 'D1'}]},
        ]}
        original = deepcopy(catalog)
        writer = Mock(side_effect=AssertionError('Exact words must not be shortened'))
        self.assertIn('more speaking time', fit_camera_dialogue('Maya speaks.', segment, catalog, [],
                      generate=writer, system_prompt='')[0])
        self.assertEqual(catalog, original)
        writer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
