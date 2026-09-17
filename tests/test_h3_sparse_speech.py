"""Supplied lines remain local performances, not whole-film dialogue quotas."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.adaptive_enhancement import (
    adaptive_dialogue_expected, adaptive_dialogue_expansion_requested,
)
from services import llm_service
from services.h3_story_ledger import (
    _canonicalize_story_ledger, _cap_h3_opening_dialogue_lead,
    _canonicalize_segment_contract,
    _deterministic_ledger, extract_h3_source_intent, extract_locked_dialogue,
    extract_source_events, ledger_violations, segment_violations,
    _filmable_source_event,
    _explicit_speech_action_order,
    _camera_event_card_schema, _expand_camera_event_cards,
    _lock_h3_source_owned_context, _source_owns_h3_subject_appearance,
)
from services.h3_sequence_planner import _reference_context


FINALE = (
    'A man named Blaine is in an epic fight with Thanos. At the final winning punch '
    'Blaine says to Thanos, "Snap this, bitch" right before punching Thanos, '
    'making him smash through a building. Cinematic.'
)


class SparseSpeechTests(unittest.TestCase):
    def test_global_source_clock_is_removed_from_local_filmable_action(self):
        self.assertEqual(
            _filmable_source_event(
                'At approximately 6.0 seconds, she swings upward and at 13.5 seconds lands.'),
            'She swings upward and lands',
        )
        self.assertEqual(
            _filmable_source_event(
                'At the 1.5-second mark, she fires the first line and holds for 2 seconds.'),
            'She fires the first line and holds for 2 seconds',
        )
        for clock in ('At the 1.5 second mark,', 'At the 1.5s mark,', 'By 1.5 sec,'):
            with self.subTest(clock=clock):
                self.assertEqual(
                    _filmable_source_event(clock + ' she catches the rope.'),
                    'She catches the rope',
                )

    def test_camera_cards_remove_global_marks_before_local_timing(self):
        result = _expand_camera_event_cards(
            {
                'event_cards': {
                    'event_1': {
                        'phases': [{
                            'transition': 'continuous track',
                            'framing': 'wide profile',
                            'camera': 'track the subject',
                            'action': 'At the 1.5-second mark, Mina catches the rope.',
                            'sound_effects': 'Rope snaps taut',
                        }],
                    },
                },
            },
            assigned_beats=[{'dialogue_ids': []}],
            segment_number=1,
            duration=4.0,
        )
        self.assertEqual(result['shots'][0]['action'], 'Mina catches the rope.')

    def test_structured_or_media_source_owns_visual_identity_and_mechanics(self):
        context_ir = (
            'integrated_multimodal_description: Spider-Man lands beside Peter Griffin.\n\n'
            'overall_soundscape: City ambience.\n\nnon_diegetic_music: N/A'
        )
        self.assertTrue(_source_owns_h3_subject_appearance(context_ir))
        self.assertTrue(_source_owns_h3_subject_appearance('Mina crosses a room', start_frame_supplied=True))
        identity, _, _ = _reference_context([{
            'type': 'image', 'path': 'mina.png', 'role': 'Mina', 'image_intent': 'identity',
        }])
        self.assertTrue(_source_owns_h3_subject_appearance(
            'Mina crosses a room', reference_context=identity,
        ))
        self.assertFalse(_source_owns_h3_subject_appearance(
            'Mina crosses a room',
            reference_context='<Audio 1> supplies background music style',
        ))
        self.assertFalse(_source_owns_h3_subject_appearance('Mina crosses a room'))

    def test_real_reference_context_intents_only_lock_subject_appearance_for_identity(self):
        def context(reference):
            return _reference_context([reference])[0]

        image_cases = {
            intent: context({
                'type': 'image', 'path': f'{intent}.png', 'role': intent,
                'image_intent': intent,
            })
            for intent in ('identity', 'composition', 'style', 'scene')
        }
        self.assertTrue(_source_owns_h3_subject_appearance(
            'A dancer enters.', reference_context=image_cases['identity'],
        ))
        for intent in ('composition', 'style', 'scene'):
            with self.subTest(image_intent=intent):
                self.assertFalse(_source_owns_h3_subject_appearance(
                    'A dancer enters.', reference_context=image_cases[intent],
                ))

        video_cases = {
            intent: context({
                'type': 'video', 'path': f'{intent}.mp4', 'role': intent,
                'video_intent': intent, 'include_audio': False,
            })
            for intent in ('character', 'motion', 'scene')
        }
        self.assertTrue(_source_owns_h3_subject_appearance(
            'A dancer enters.', reference_context=video_cases['character'],
        ))
        for intent in ('motion', 'scene'):
            with self.subTest(video_intent=intent):
                self.assertFalse(_source_owns_h3_subject_appearance(
                    'A dancer enters.', reference_context=video_cases[intent],
                ))

        audio = context({
            'type': 'audio', 'path': 'voice.wav', 'role': 'voice', 'audio_intent': 'voice',
        })
        self.assertFalse(_source_owns_h3_subject_appearance(
            'A dancer enters.', reference_context=audio,
        ))
        self.assertTrue(_source_owns_h3_subject_appearance(
            'A dancer enters.',
            reference_context=image_cases['style'] + '\n' + image_cases['identity'],
        ))

    def test_exact_first_frame_and_native_context_ir_still_own_appearance(self):
        self.assertTrue(_source_owns_h3_subject_appearance(
            'A dancer enters.',
            reference_context='<Picture 1> is the exact first frame and visual identity/scene anchor.',
        ))
        self.assertTrue(_source_owns_h3_subject_appearance(
            'integrated_multimodal_description: A dancer in a red coat enters.',
            reference_context='<Audio 1> supplies background music style',
        ))

        candidate = {
            'subject_continuity': 'Peter is blue-skinned; Spider-Man has exposed hair',
            'initial_state': 'Peter waits at the bottom of the tower',
            'motion_mechanics': 'Spider-Man uses innate flight',
            'required_final_outcome': 'They keep moving',
        }
        canonical = {
            'subject_continuity': 'Keep Spider-Man and Peter Griffin in their requested identities',
            'initial_state': 'Spider-Man begins alone at the skyscraper top',
            'motion_mechanics': '',
            'required_final_outcome': 'They remain perfectly still on the upper ledge',
        }
        _lock_h3_source_owned_context(candidate, canonical)
        self.assertEqual(candidate, canonical)

        image_treatment = {
            'subject_continuity': 'Map roles to the supplied image',
            'initial_state': 'The observed kick is already blocked',
            'motion_mechanics': 'The observed contact drives the recovery',
            'required_final_outcome': 'The requested ending completes',
        }
        _lock_h3_source_owned_context(
            image_treatment,
            canonical,
            lock_initial_state=False,
            lock_mechanics=False,
        )
        self.assertEqual(image_treatment['initial_state'], 'The observed kick is already blocked')
        self.assertEqual(image_treatment['motion_mechanics'], 'The observed contact drives the recovery')

    def test_headings_directions_and_chronology_words_do_not_become_cast(self):
        archive = (
            'In an archive, adult clerk Hana waits while adult inspector Luis stands outside. '
            'Next Luis says, "The seal is intact." Hana answers, "It matches the ledger." '
            'Finally Hana opens the cage and both step out. Preserve speaker and action order. Add no dialogue.'
        )
        self.assertEqual(extract_h3_source_intent(archive)['cast_names'], ['Hana', 'Luis'])
        self.assertEqual(
            [(line['speaker'], line['text']) for line in extract_locked_dialogue(archive)],
            [('Luis', 'The seal is intact.'), ('Hana', 'It matches the ledger.')],
        )
        archive_events = ' '.join(item['text'] for item in extract_source_events(archive))
        self.assertNotIn('Preserve speaker', archive_events)
        self.assertNotIn('Add no dialogue', archive_events)

        timed = (
            '[0.0-7.0s | Establish] Adult gardener Mei enters with an empty can. '
            '[7.0-14.0s | Fill] Mei fills it. '
            '[14.0-21.0s | Water] Mei waters three pots. '
            '[21.0-28.0s | Finish] Mei opens the vent. Preserve pot order and tap state. No dialogue or cuts.'
        )
        self.assertEqual(extract_h3_source_intent(timed)['cast_names'], ['Mei'])
        self.assertFalse(any(
            'Preserve pot order' in event['text'] or 'No dialogue' in event['text']
            for event in extract_source_events(timed)
        ))

        tracking = (
            'One continuous real-time tracking shot follows adult courier Bo running east. '
            'Bo catches the cart and stops it. No cuts or dialogue.'
        )
        self.assertEqual(extract_h3_source_intent(tracking)['cast_names'], ['Bo'])

    def test_introductory_phrases_carry_the_actor_not_the_preposition(self):
        source = (
            'At blue hour, adult teacher Amara unfolds a letter, notices a flower, and sits. '
            'She reads silently, smooths one corner, then closes it against her chest. '
            'After a long breath she watches the porch light. Convey relief through hands, breath, and gaze. '
            'No dialogue, visitor, or supernatural sign.'
        )
        events = [item['text'] for item in extract_source_events(source)]
        self.assertIn('Amara sits', events)
        self.assertIn('After a long breath she watches the porch light', events)
        self.assertFalse(any(item.startswith(('At sits', 'After watches')) for item in events))
        self.assertFalse(any('Convey relief' in item or item == 'gaze' for item in events))

    def test_short_residual_tail_holds_outcome_after_last_usable_event_window(self):
        source = (
            'Spider-Man leaps from a roof, swings across the city, releases the web, '
            'and flips toward a ledge. Near the end, he lands next to Peter Griffin, '
            'who exclaims, "Wow! That was incredible!"'
        )
        durations = [14.375, 13.625, 0.542]
        locked = extract_locked_dialogue(source)
        ledger = _deterministic_ledger(
            source, segment_count=3, segment_durations=durations,
            locked_dialogue=locked, camera_coverage='multi_shot', reference_context='')
        ending = next(beat for beat in ledger['beats'] if 'D1' in beat['dialogue_ids'])
        self.assertEqual(ending['segment'], 2)
        tail = [beat for beat in ledger['beats'] if beat['segment'] == 3]
        self.assertTrue(tail)
        self.assertTrue(all(not beat['source_event_ids'] for beat in tail))
        self.assertEqual(ledger_violations(
            source, ledger, segment_count=3, locked_dialogue=locked,
            expect_dialogue=True, segment_durations=durations), [])

        # If every native window is too short, do not reorder the ending into
        # an earlier slot merely because the last slot also cannot fit it.
        uniform_short = _deterministic_ledger(
            source, segment_count=3, segment_durations=[2.0, 2.0, 2.0],
            locked_dialogue=locked, camera_coverage='multi_shot', reference_context='')
        ending = next(beat for beat in uniform_short['beats'] if 'D1' in beat['dialogue_ids'])
        self.assertEqual(ending['segment'], 3)

    def test_semantic_camera_must_keep_steps_relations_and_future_events_local(self):
        def segment(action, beat, *, closing='writer changed the state'):
            return {
                'segment': 1, 'semantic_actions': True,
                'shots': [{
                    'shot': 1, 'start_seconds': 0.0, 'end_seconds': 10.0,
                    'action': action, 'beat_ids': ['B1'], 'dialogue': [],
                }],
                'closing_state': closing,
            }

        quiet = {'beat_id': 'B1', 'description': 'She reads silently, smooths one torn corner',
                 'source_event_ids': ['E1'], 'dialogue_ids': [], 'state_after': 'letter remains in her hands'}
        errors = segment_violations(
            'She reads silently, smooths one torn corner.',
            segment('She smooths the torn corner.', quiet), segment_number=1,
            duration=10.0, assigned_beats=[quiet], dialogue_catalog=[])
        self.assertTrue(any('omits required source step' in error for error in errors), errors)

        ordered = {'beat_id': 'B1',
                   'description': 'Priya closes that hand before the boarding gate opens',
                   'source_event_ids': ['E1'], 'dialogue_ids': [], 'state_after': 'gate is open'}
        errors = segment_violations(
            ordered['description'],
            segment('Priya closes her hand just as the boarding gate opens.', ordered),
            segment_number=1, duration=10.0, assigned_beats=[ordered], dialogue_catalog=[])
        self.assertTrue(any("drops the explicit 'before'" in error for error in errors), errors)
        equivalent = segment_violations(
            ordered['description'],
            segment('The boarding gate opens only after Priya closes her hand.', ordered),
            segment_number=1, duration=10.0, assigned_beats=[ordered], dialogue_catalog=[])
        self.assertFalse(any("drops the explicit 'before'" in error for error in equivalent), equivalent)
        reversed_order = segment_violations(
            ordered['description'],
            segment('Priya closes her hand only after the boarding gate opens.', ordered),
            segment_number=1, duration=10.0, assigned_beats=[ordered], dialogue_catalog=[])
        self.assertTrue(any("drops the explicit 'before'" in error for error in reversed_order), reversed_order)

        mixed = {'beat_id': 'B1',
                 'description': ('Mara unlocks the door before Dev enters. '
                                 'Dev enters only after the alarm stops'),
                 'source_event_ids': ['E1'], 'dialogue_ids': [], 'state_after': 'Dev is inside'}
        missing_second = segment_violations(
            mixed['description'],
            segment('Mara unlocks the door before Dev enters.', mixed),
            segment_number=1, duration=10.0, assigned_beats=[mixed], dialogue_catalog=[])
        self.assertTrue(any("only after" in error for error in missing_second), missing_second)
        equivalent_mixed = segment_violations(
            mixed['description'],
            segment(('Dev enters only after Mara unlocks the door. '
                     'The alarm stops before Dev enters.'), mixed),
            segment_number=1, duration=10.0, assigned_beats=[mixed], dialogue_catalog=[])
        self.assertFalse(any('chronology relation' in error for error in equivalent_mixed), equivalent_mixed)
        reversed_mixed = segment_violations(
            mixed['description'],
            segment(('Mara unlocks the door before Dev enters. '
                     'Dev enters before the alarm stops.'), mixed),
            segment_number=1, duration=10.0, assigned_beats=[mixed], dialogue_catalog=[])
        self.assertTrue(any("only after" in error for error in reversed_mixed), reversed_mixed)

        attributed = {'beat_id': 'B1',
                      'description': 'Spider-Man lands next to Peter Griffin, who exclaims',
                      'source_event_ids': ['E1'], 'dialogue_ids': [],
                      'state_after': 'Spider-Man remains crouched beside Peter Griffin'}
        attributed_errors = segment_violations(
            attributed['description'],
            segment('Spider-Man lands next to Peter Griffin.', attributed),
            segment_number=1, duration=10.0, assigned_beats=[attributed], dialogue_catalog=[])
        self.assertFalse(any('who exclaims' in error for error in attributed_errors), attributed_errors)

        crane_prompt = (
            "Inez clips a steel hook to the pallet and keeps Cole clear of the taut cable. "
            "The winch pulls vertically, the pallet rises, and Cole slides his boot free."
        )
        crane_ledger = _deterministic_ledger(
            crane_prompt, segment_count=2, segment_durations=[10.0, 10.0],
            locked_dialogue=[], camera_coverage='multi_shot', reference_context='')
        first = [beat for beat in crane_ledger['beats'] if beat['segment'] == 1]
        bad = {
            'segment': 1, 'semantic_actions': True,
            'shots': [{
                'shot': 1, 'start_seconds': 0.0, 'end_seconds': 10.0,
                'action': 'Inez clips the hook, then the pallet rises and Cole slides his boot free.',
                'beat_ids': [beat['beat_id'] for beat in first], 'dialogue': [],
            }],
            'closing_state': 'Cole is already free.',
        }
        errors = segment_violations(
            crane_prompt, bad, segment_number=1, duration=10.0,
            assigned_beats=first, dialogue_catalog=[])
        self.assertTrue(any('previews later source event' in error for error in errors), errors)

        canonical = _canonicalize_segment_contract(
            segment('She reads silently and smooths the corner.', quiet),
            segment_number=1, duration=10.0, assigned_beats=[quiet],
            dialogue_catalog=[], opening_state='letter is open',
            source_intent={},
        )
        self.assertEqual(canonical['closing_state'], 'letter remains in her hands')

    def test_recurring_cast_name_does_not_preview_a_later_winning_event(self):
        for name in ('Power Girl', 'Maya Chen'):
            with self.subTest(name=name):
                source = (
                    f'A bald monk fights {name}. '
                    'They destroy the surrounding environment from their powerful hits, '
                    'smashing each other through walls. '
                    f'{name} wins.'
                )
                events = extract_source_events(source)
                destruction = next(event for event in events if 'destroy' in event['text'])
                beat = {
                    'beat_id': 'B2', 'source_event_ids': [destruction['event_id']],
                    'description': destruction['text'], 'dialogue_ids': [],
                    'state_after': 'The fighters still trade blows amid shattered walls.',
                }
                action = (
                    f'{name} drives the monk through a wall. He catches her arm and counters, '
                    'smashing her through the opposite wall; their powerful hits destroy '
                    'the surrounding environment while they keep fighting.'
                )
                segment = {
                    'segment': 2, 'semantic_actions': True,
                    'shots': [{'shot': 1, 'start_seconds': 0, 'end_seconds': 10,
                               'beat_ids': ['B2'], 'dialogue': [], 'action': action}],
                    'closing_state': beat['state_after'],
                }
                errors = segment_violations(
                    source, segment, segment_number=2, duration=10,
                    assigned_beats=[beat], dialogue_catalog=[],
                )
                self.assertFalse(any('previews later' in error for error in errors), errors)
                segment['shots'][0]['action'] += f' {name} wins.'
                errors = segment_violations(
                    source, segment, segment_number=2, duration=10,
                    assigned_beats=[beat], dialogue_catalog=[],
                )
                self.assertTrue(any('previews later' in error for error in errors), errors)

    def test_context_ir_exclamation_retains_real_cast_and_speech(self):
        examples = (
            ("Spider-Man", "Peter Griffin", "Strong", "Wow! That was incredible!"),
            ("Storm-Rider", "Maya Chen", "Thunderous", "We made it!"),
        )
        for mover, witness, sound_adjective, words in examples:
            with self.subTest(mover=mover):
                source = (
                    "integrated_multimodal_description: [Shot 1] A low tracking shot follows "
                    f"{mover} as she leaps from a skyscraper, swings across the city, and lands "
                    f"next to {witness}, who exclaims, \"{words}\"\n\n"
                    f"overall_soundscape: {sound_adjective} rush of wind and distant traffic.\n\n"
                    "non_diegetic_music: Fast-paced orchestral theme with bright brass."
                )
                intent = extract_h3_source_intent(source)
                self.assertEqual(intent["cast_names"], [mover, witness])
                self.assertIn(sound_adjective, intent["global_instructions"])
                dialogue = extract_locked_dialogue(source)
                self.assertEqual(len(dialogue), 1)
                self.assertEqual(dialogue[0]["speaker"], witness)
                self.assertEqual(dialogue[0]["text"], words)
                events = extract_source_events(source)
                self.assertTrue(any(mover in item["text"] for item in events))
                self.assertIn(f"next to {witness}", events[-1]["text"])
                self.assertFalse(any(item["text"].startswith(f"to {witness}") for item in events))
                self.assertFalse(any("soundscape" in item["text"] for item in events))
                self.assertFalse(any("orchestral" in item["text"] for item in events))

    def test_supplied_speech_does_not_request_additional_writing(self):
        for source in (
            FINALE,
            'Mae hands Jules a note and says, "Keep this safe."',
            'Mina says, "Stay." Theo replies, "Yes."',
            'Mina says, "We should discuss this. Tell me everything."',
            'Mina: Stay.\nTheo: Yes.',
            'Mina (S1) says <d>[English] Stay.</d> before closing the door.',
        ):
            with self.subTest(source=source):
                self.assertTrue(adaptive_dialogue_expected(source))
                self.assertFalse(adaptive_dialogue_expansion_requested(source))

    def test_unscripted_exchange_is_still_written(self):
        for source in (
            'Mina and Theo discuss the letter.',
            'Mina says, "Stay." Theo explains why he left. They discuss a new beginning.',
            'Mina says, "Stay." Theo replies that he cannot.',
            'Mina says, "Stay," and Theo answers her.',
            'Mina asks Theo whether he received the letter.',
        ):
            with self.subTest(source=source):
                self.assertTrue(adaptive_dialogue_expansion_requested(source))

    def test_no_padding_or_rewrite_for_a_short_exact_line(self):
        draft = ('integrated_multimodal_description: [Shot 1] Blaine (S1) says '
                 '<d>[English] Snap this, bitch</d> just before his finishing punch. '
                 'Thanos smashes through the building.')
        writer = Mock()
        self.assertEqual(llm_service._fit_adaptive_dialogue(FINALE, draft, 28, writer), draft)
        writer.assert_not_called()
        requirement = llm_service._build_h3_dialogue_requirement(FINALE, 28, 'adaptive')
        self.assertIn('Do not add speech.', requirement)
        self.assertIn('Snap this, bitch', requirement)
        context = llm_service._build_enhance_user_prompt(
            FINALE, 'video', 28, 2, 14, model_type='minimax_h3_ref2va', planning_style='adaptive')
        self.assertNotIn('allowed', context.lower())
        self.assertTrue(context.endswith(FINALE))
        # Duration ceilings still apply; this change removes only artificial minimums.
        overlong = draft.replace('Snap this, bitch', 'word ' * 100)
        self.assertFalse(llm_service._h3_speech_timing_satisfied(FINALE, overlong, 2, 'adaptive'))

    def test_speech_and_its_causal_action_remain_one_event(self):
        examples = (
            'Mina says, "Catch!" right before throwing the ball to Theo.',
            'Mina says, "Done." after closing the valve.',
            'Mina says <d>[English] Ready.</d> while lifting the crate.',
            'Mina: "Catch!" before throwing the ball to Theo.',
        )
        for source in examples:
            with self.subTest(source=source):
                events = extract_source_events(source)
                self.assertEqual(len(events), 1, events)
                self.assertIn('Mina', events[0]['text'])
                self.assertTrue(any(word in events[0]['text'] for word in ('throwing', 'closing', 'lifting')))
                self.assertEqual(len(extract_locked_dialogue(source)), 1)

    def test_finale_is_not_moved_to_opening_or_split_across_windows(self):
        events = extract_source_events(FINALE)
        self.assertEqual(len(events), 2, events)
        self.assertIn('before punching', events[-1]['text'])
        self.assertIn('smash through a building', events[-1]['text'])
        intent = extract_h3_source_intent(FINALE)
        self.assertEqual(intent['cast_names'], ['Blaine', 'Thanos'])
        self.assertEqual(intent['opening_dialogue_id'], '')
        locked = extract_locked_dialogue(FINALE)
        ledger = _deterministic_ledger(
            FINALE, segment_count=3, segment_durations=[10, 10, 10],
            locked_dialogue=locked, camera_coverage='auto', reference_context='')
        canonical = _canonicalize_story_ledger(
            FINALE, ledger, deepcopy(ledger), locked_dialogue=locked,
            segment_count=3, allow_generated_dialogue=False)
        speaking = [beat for beat in canonical['beats'] if 'D1' in beat['dialogue_ids']]
        self.assertEqual(len(speaking), 1)
        self.assertEqual(speaking[0]['segment'], 3)
        self.assertIn('E2', speaking[0]['source_event_ids'])
        self.assertEqual(ledger_violations(
            FINALE, canonical, segment_count=3, locked_dialogue=locked,
            allow_generated_dialogue=False, expect_dialogue=True), [])

    def test_delay_and_immediate_greeting_have_different_opening_contracts(self):
        for prefix in ('At the climax,', 'Near the end,', 'Only after the landing,',
                       'Eventually,', 'After waiting several minutes,'):
            with self.subTest(prefix=prefix):
                source = f'A diver inspects the wreck. {prefix} Mina says, "Ready."'
                self.assertEqual(extract_h3_source_intent(source)['opening_dialogue_id'], '')
        immediate = 'Mina enters the shop. Mina says, "Hello, Theo."'
        self.assertEqual(extract_h3_source_intent(immediate)['opening_dialogue_id'], 'D1')

    def test_opening_clock_keeps_all_camera_phases_usable(self):
        for lengths in ((4, 4, .75, 1.25), (6, .75, .75, 2.5)):
            shots = []
            cursor = 0
            for index, length in enumerate(lengths):
                shots.append({'start_seconds': cursor, 'end_seconds': cursor + length,
                              'action': f'physical action {index}'})
                cursor += length
            assignments = [[{'dialogue_ids': []}] for _ in shots]
            assignments[-1][0]['dialogue_ids'] = ['D1']
            _cap_h3_opening_dialogue_lead(shots, assignments, duration=10, dialogue_id='D1')
            self.assertAlmostEqual(shots[-1]['start_seconds'], 3)
            self.assertEqual(shots[-1]['end_seconds'], 10)
            self.assertEqual([s['action'] for s in shots], [f'physical action {i}' for i in range(4)])
            for index, shot in enumerate(shots):
                self.assertGreaterEqual(shot['end_seconds'] - shot['start_seconds'], .749)
                if index:
                    self.assertEqual(shot['start_seconds'], shots[index - 1]['end_seconds'])

    def test_explicit_action_after_speech_cannot_disappear_into_closing_state(self):
        order = _explicit_speech_action_order(FINALE)
        self.assertIn('punching Thanos', order['D1']['after_speech'])
        beat = {'dialogue_ids': ['D1'], '_speech_action_order': order}
        schema = _camera_event_card_schema(3, [beat])
        follow = schema['properties']['event_cards']['properties']['event_1']['properties']['follow_through']
        self.assertEqual(follow['type'], 'object')
        def card(action):
            return dict(action=action, framing='Two shot', camera='Tracking',
                        transition='Continue', sound_effects='Rain')
        draft = {'event_cards': {'event_1': {'D1': {'lead_in': None,
                 'performance': card('Blaine delivers his taunt facing Thanos.')}, 'follow_through': None}},
                 'closing_state': 'Thanos has smashed through the building.'}
        with self.assertRaisesRegex(ValueError, 'complete camera/action'):
            _expand_camera_event_cards(draft, assigned_beats=[beat], segment_number=3, duration=10)
        draft['event_cards']['event_1']['follow_through'] = card(
            'Blaine punches Thanos, sending him through the building.')
        result = _expand_camera_event_cards(draft, assigned_beats=[beat], segment_number=3, duration=10)
        self.assertEqual([s['dialogue_ids'] for s in result['shots']], [['D1'], []])
        self.assertIn('punches Thanos', result['shots'][-1]['action'])

    def test_action_before_speech_requires_lead_in_but_concurrent_action_does_not(self):
        for connector, slot in (('after', 'before_speech'), ('while', 'during_speech')):
            source = f'Mina says, "Done." {connector} closing the valve.'
            order = _explicit_speech_action_order(source)
            self.assertEqual(order, {'D1': {slot: 'closing the valve'}})
            beat = {'dialogue_ids': ['D1'], '_speech_action_order': order}
            schema = _camera_event_card_schema(1, [beat])
            turn = schema['properties']['event_cards']['properties']['event_1']['properties']['D1']['properties']
            self.assertEqual(turn['lead_in']['type'], 'null' if connector == 'while' else 'object')

    def test_ongoing_activity_can_develop_before_separate_ending_without_replaying_it(self):
        locked = extract_locked_dialogue(FINALE)
        canonical = _deterministic_ledger(
            FINALE, segment_count=3, locked_dialogue=locked,
            camera_coverage='auto', reference_context='')
        candidate = deepcopy(canonical)
        candidate['beats'][1]['source_event_ids'] = ['E1']
        candidate['beats'][1]['description'] = 'Thanos counters; Blaine ducks and returns an uppercut.'
        result = _canonicalize_story_ledger(
            FINALE, canonical, candidate, locked_dialogue=locked,
            segment_count=3, allow_generated_dialogue=False, preserve_adaptation=True)
        self.assertEqual(result['beats'][1]['source_event_ids'], [])
        self.assertIn('uppercut', result['beats'][1]['description'])
        self.assertEqual(ledger_violations(FINALE, result, segment_count=3,
            locked_dialogue=locked, allow_generated_dialogue=False, expect_dialogue=True), [])
        # Repeating a concrete impact is still invalid, even with different prose.
        source = FINALE.replace('A man named Blaine is in an epic fight with Thanos',
                                'Blaine smashes Thanos through a wall')
        concrete = _canonicalize_story_ledger(
            source, canonical, candidate, locked_dialogue=extract_locked_dialogue(source),
            segment_count=3, allow_generated_dialogue=False, preserve_adaptation=True)
        self.assertTrue(ledger_violations(source, concrete, segment_count=3,
            locked_dialogue=extract_locked_dialogue(source), allow_generated_dialogue=False,
            expect_dialogue=True))


if __name__ == '__main__':
    unittest.main()
