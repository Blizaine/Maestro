"""Formatting changes must not turn visual metadata into exact spoken lines."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_authored_brief import authored_timed_brief
from services.h3_story_ledger import (
    extract_locked_dialogue, extract_source_events, extract_h3_source_intent,
    _prepare_render_dialogue_schedule, H3DialogueTimingError, _spectacle_violations,
)


class BriefStructureTests(unittest.TestCase):
    def test_role_heading_inflections_do_not_change_speech_interpretation(self):
        for role in ('Character', 'Subject', 'Actor', 'Role', 'Profile'):
            for suffix in ('', 's'):
                with self.subTest(heading=role + suffix):
                    source = f'## {role}{suffix}\nTraveler: a green coat and a leather satchel.\n## Scene\nTraveler: We made it.'
                    self.assertEqual([d['text'] for d in extract_locked_dialogue(source)], ['We made it.'])
                    self.assertIn('green coat', extract_h3_source_intent(source)['global_instructions'])

    def test_authored_energy_can_be_visualized_without_identical_wording(self):
        for source in (
            'The hero channels cursed-energy glow and opens a domain of blue swords.',
            'The sorcerer shapes arcane power into a luminous dome.',
            'A blue energy glow condenses around the statue.',
        ):
            with self.subTest(source=source):
                self.assertEqual(_spectacle_violations(source, {'action': 'A blue energy field forms.'}), [])

    def test_energy_permission_does_not_allow_unrelated_powers(self):
        source = 'Cursed energy surrounds the blade.'
        for action in ('She moves the boat with telekinesis.', 'He fires a laser beam.'):
            with self.subTest(action=action):
                self.assertTrue(_spectacle_violations(source, {'action': action}))
        self.assertTrue(_spectacle_violations('Two people trade ordinary punches.', {'action': 'A golden energy wave erupts.'}))
        self.assertEqual(_spectacle_violations('A laser beam crosses an optics bench. No magic.',
                                              {'action': 'The laser beam hits a prism.'}), [])

    def test_negative_effect_rules_are_not_permissions_or_performed_effects(self):
        source = 'A close-range duel. No energy field or laser beam.'
        self.assertTrue(_spectacle_violations(source, {'action': 'An energy field appears.'}))
        self.assertEqual(_spectacle_violations(source, {'action': 'He blocks the punch. No energy field appears.'}), [])
        for negative in ('No magic.', 'No glowing energy.', 'No colored energies.'):
            with self.subTest(negative=negative):
                self.assertTrue(_spectacle_violations('Use arcane energy. ' + negative, {'action': 'An energy field forms.'}))

    def test_profile_format_matrix_preserves_description_and_actual_dialogue(self):
        headings = ["**Characters**", "__Character profiles__", "## Cast", "【Characters】", "[Cast]", "Characters:"]
        for heading in headings:
            for name in ("Hero", "Mina", "Chef", "Archivist"):
                with self.subTest(heading=heading, name=name):
                    source = (
                        f'{heading}\n{name} (“The Swift One”): messy black hair, a blue coat, a long brass key.\n'
                        'Visitors: several adults in raincoats wait in the doorway.\n'
                        '**Scene**\n'
                        f'{name}: Please wait here.\n'
                        f'{name} unlocks the gate and the visitors enter.\n'
                        '**Lighting**\nKey: warm light from the east.\n'
                    )
                    self.assertEqual([(d['speaker'], d['text']) for d in extract_locked_dialogue(source)],
                                     [(name, 'Please wait here.')])
                    intent = extract_h3_source_intent(source)
                    self.assertIn('messy black hair', intent['global_instructions'])
                    self.assertIn('warm light from the east', intent['global_instructions'])
                    self.assertIn(name, intent['cast_names'])
                    self.assertNotIn('Key', intent['cast_names'])
                    events = ' '.join(e['text'] for e in extract_source_events(source))
                    self.assertIn('unlocks the gate', events)
                    self.assertNotIn('messy black hair', events)
                    self.assertNotIn('warm light', events)

    def test_quoted_metadata_does_not_reenter_through_quote_extractor(self):
        source = ('**Characters**\nNavigator: "the quiet one", silver hair and a patched jacket.\n'
                  '**Effects**\nHalo: "a soft ring", glowing above the compass.\n'
                  '**Story**\nThe navigator lifts the compass.')
        self.assertEqual(extract_locked_dialogue(source), [])
        self.assertIn('silver hair', extract_h3_source_intent(source)['global_instructions'])
        self.assertIn('lifts the compass', extract_source_events(source)[-1]['text'])

    def test_explicit_speech_inside_notes_is_still_exact(self):
        for declaration in (
            'Mina says, "Keep the blue light."',
            'Mina (whispers): "Keep the blue light."',
            'Mina: <d>Keep the blue light.</d>',
        ):
            with self.subTest(declaration=declaration):
                source = '**Lighting notes**\n' + declaration + '\n**Story**\nMina opens the gate.'
                self.assertEqual([d['text'] for d in extract_locked_dialogue(source)], ['Keep the blue light.'])
                events = ' '.join(e['text'] for e in extract_source_events(source))
                self.assertIn('speaks', events)
                self.assertIn('opens the gate', events)

    def test_generic_quoted_note_labels_are_not_speakers(self):
        for label in ("Example", "Examples", "Template", "Templates", "Instruction", "Instructions"):
            with self.subTest(label=label):
                source = f'## Technical Notes\n{label}: "Keep the camera steady."\n## Story\nNora enters.'
                self.assertEqual(extract_locked_dialogue(source), [])

    def test_plain_screenplay_and_quoted_directions_remain_supported(self):
        for source in (
            'Mina: The blue coat belongs to me.\nDev: I will bring it.',
            'Mina: "The blue coat belongs to me."\nDev: "I will bring it."',
            'Mina says, "The blue coat belongs to me." Dev replies, "I will bring it."',
        ):
            with self.subTest(source=source):
                self.assertEqual([d['text'] for d in extract_locked_dialogue(source)],
                                 ['The blue coat belongs to me.', 'I will bring it.'])

    def test_numbered_timed_brief_keeps_shots_and_never_reparses_camera_as_speaker(self):
        source = ('**Characters**\nPilot: silver hair and a blue jumpsuit.\n'
                  '**12-second shot plan**\n'
                  '1. 0–4s MS low angle: pilot grabs the throttle.\n'
                  '2. 4–8s CU of the hand: she banks left through the arch.\n'
                  '3. 8–12s MS: she lands beside the beacon.\n'
                  '**Negatives**\nNo on-screen text or logos.')
        self.assertEqual(extract_locked_dialogue(source), [])
        brief = authored_timed_brief(source)
        self.assertEqual(len(brief['events']), 3)
        self.assertIn('No on-screen text', brief['context'])
        events = extract_source_events(source)
        self.assertEqual([e['text'] for e in events], [
            'MS low angle: pilot grabs the throttle.',
            'CU of the hand: she banks left through the arch.',
            'MS: she lands beside the beacon.',
        ])
        for event in brief['events']:
            self.assertEqual(source[event['source_offset']:event['source_end']].strip(), event['text'])

    def test_timed_sections_use_full_source_dialogue_offsets(self):
        source = ('**Cast**\nMina: silver hair, blue coat.\n'
                  '1. 0–4s Mina raises a lantern.\nMina: "Come with me."\n'
                  '2. 4–8s CU of the gate: it swings open.\nMina: "It is safe."\n')
        self.assertEqual([d['text'] for d in extract_locked_dialogue(source)], ['Come with me.', 'It is safe.'])
        events = extract_source_events(source)
        self.assertIn('Mina speaks.', events[0]['text'])
        self.assertIn('CU of the gate: it swings open.', events[1]['text'])
        self.assertIn('Mina speaks.', events[1]['text'])
        self.assertNotIn('Come with me', events[0]['text'])

    def test_long_descriptions_cost_no_speech_but_real_long_speech_is_rejected(self):
        description = 'Blue fabric, worn seams, silver clasps, rain-dark sleeves. ' * 20
        source = '**Characters**\nTraveler: ' + description + '\n**Story**\nThe traveler crosses the bridge.'
        self.assertEqual(extract_locked_dialogue(source), [])
        spoken = 'Mina: ' + description
        catalog = extract_locked_dialogue(spoken)
        self.assertEqual(catalog[0]['text'], description.strip())
        with self.assertRaises(H3DialogueTimingError):
            _prepare_render_dialogue_schedule(
                beats=[{'segment': 1, 'dialogue_ids': ['D1']}], dialogue_catalog=catalog,
                expected_dialogue_events={'D1': 'E1'}, source_events=[], segment_durations=[4.0],
            )


if __name__ == '__main__':
    unittest.main()
