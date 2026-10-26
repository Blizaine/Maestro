"""Descriptive headings are not physical actions; real story transitions are."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.music_narrative import (
    build_music_narrative_contract, reconcile_music_narrative_checkpoint,
    music_narrative_row_issue, validate_music_narrative_allocation,
)

BRIEF = '''CHARACTER (lock this look for every shot): A wooden puppet singer holding a guitar.
His nose grows whenever he lies. He has already cut his own strings.
SETTING: A smoky dive bar with monitors and an audience.
STORY ARC:
1. Opening: Puppet on strings, operators on screens pulling the strings.
2. Middle: His nose grows absurdly long.
3. Climax: He cuts the strings with a guitar pick, smashes the screens, nose shrinks back.
4. Ending: Close-up, puppet standing free.
PERFORMANCE: Big gestures, guitar raised, looking at camera throughout the chorus.
STYLE: No deformed hands in close-up. No text watermarks.
'''


class MusicSourceSectionTests(unittest.TestCase):
    def test_real_story_actions_survive_without_metadata_milestones(self):
        contract = build_music_narrative_contract(BRIEF)
        self.assertEqual(contract['source'], BRIEF)
        self.assertEqual(contract['revision'], 5)
        self.assertEqual([event['event_id'] for event in contract['events']],
                         ['M1', 'M2', 'M3', 'M4', 'M5'])
        self.assertEqual([event['text'] for event in contract['events']],
                         ['pulling the strings', 'grows absurdly long',
                          'cuts the strings with a guitar pick', 'smashes the screens', 'shrinks back'])
        self.assertEqual(contract['events'][0]['actor_aliases'], ['operators'])
        self.assertEqual(contract['events'][1]['actor_aliases'], ['nose'])
        self.assertEqual(contract['events'][4]['actor_aliases'], ['nose'])
        for event in contract['events']:
            self.assertFalse(set(event.get('actor_aliases', [])) & {'opening', 'climax', 'style', 'big'})

    def test_flat_descriptive_words_are_not_actions(self):
        source = ('Keep this character look for every shot. A singer performs in a dive bar. '
                  'No deformed hands in close-up. Use the same shot throughout.')
        self.assertEqual(build_music_narrative_contract(source)['events'], [])

    def test_literal_attacks_diving_and_handoffs_are_not_suppressed(self):
        for source, token in (
            ('The guard is shot in the leg.', 'shot'),
            ('The diver dives into the ocean.', 'dives'),
            ('The bartender hands the microphone to the singer.', 'hands'),
            ('The performer cuts the rope in close-up.', 'cuts'),
        ):
            with self.subTest(source=source):
                self.assertTrue(any(token in event['text']
                    for event in build_music_narrative_contract(source)['events']))

    def test_camera_cuts_do_not_become_physical_cutting(self):
        self.assertEqual(build_music_narrative_contract(
            'The camera cuts to a close-up, then cuts back to a wide shot.')['events'], [])

    def test_style_is_preserved_but_not_replayed_after_climax(self):
        contract = build_music_narrative_contract(BRIEF)
        self.assertIn('guitar raised', contract['source'])
        self.assertIn('No deformed hands', contract['source'])
        self.assertNotIn('raised', ' '.join(event['text'] for event in contract['events']))

    def test_story_headings_allow_markdown_and_extra_spaces(self):
        source = BRIEF.replace('STORY ARC:', '## STORY   ARC:').replace('STYLE:', '## STYLE:')
        self.assertEqual(build_music_narrative_contract(source)['events'],
                         build_music_narrative_contract(BRIEF)['events'])

    def test_character_context_remains_available_for_collective_combat(self):
        source = ('CHARACTERS: Dr Strange and Doctor Doom use their respective magic powers.\n'
                  'STORY ARC: They fight using their respective powers.\nSTYLE: Cinematic.')
        events = build_music_narrative_contract(source)['events']
        self.assertTrue(any(event['kind'] == 'combat' for event in events))

    def test_changed_meanings_invalidate_dependent_caches_without_mutating_original(self):
        checkpoint = {'music_narrative_contract': {'revision': 4, 'events': [{'event_id': 'M1', 'text': 'shot): singer'}]},
                      'music_creative_arc': ['old arc'], 'music_narrative_allocation': ['old assignment'],
                      'music_video_batches': {'0': {'rows': ['old shot']}}, 'unrelated_stage': {'keep': True}}
        before = copy.deepcopy(checkpoint)
        updated = reconcile_music_narrative_checkpoint(checkpoint, build_music_narrative_contract(BRIEF))
        self.assertEqual(checkpoint, before)
        for key in ('music_creative_arc', 'music_narrative_allocation', 'music_video_batches'):
            self.assertNotIn(key, updated)
        self.assertEqual(updated['unrelated_stage'], {'keep': True})
        self.assertIn('music_narrative_contract_migration', updated)

    def test_unchanged_real_actions_reuse_existing_checkpoint(self):
        contract = build_music_narrative_contract(BRIEF)
        old = {**contract, 'revision': 4}
        checkpoint = {'music_narrative_contract': old, 'music_creative_arc': ['accepted']}
        self.assertIs(reconcile_music_narrative_checkpoint(checkpoint, contract), checkpoint)

    def test_real_action_coverage_remains_mandatory(self):
        contract = build_music_narrative_contract('The performer cuts the strings with a pick.')
        self.assertEqual(len(contract['events']), 1)
        allocation = [{'event_id': 'M1', 'clip_index': 0}]
        self.assertEqual(validate_music_narrative_allocation(contract, allocation, 1), [])
        issue = music_narrative_row_issue(
            {'source_event_ids': ['M1'], 'video_prompt': 'The performer stands silently on stage.'},
            contract=contract, allocation=allocation, index=0)
        self.assertIsNotNone(issue)

    def test_growth_and_crowd_reaction_are_separate_required_actions(self):
        contract = build_music_narrative_contract(
            'His nose grows absurdly long, crowd boos the screens.')
        self.assertEqual([e['text'] for e in contract['events']],
                         ['grows absurdly long', 'boos the screens'])
        allocation = [{'event_id': 'M1', 'clip_index': 0}, {'event_id': 'M2', 'clip_index': 0}]
        row = {'source_event_ids': ['M1', 'M2'], 'video_prompt':
               'His wooden nose grows absurdly long, extending toward the floor. '
               'The crowd boos the screens and raises their fists.'}
        self.assertIsNone(music_narrative_row_issue(row, contract=contract, allocation=allocation, index=0))
        for prompt in ('His nose grows absurdly long.',
                       'His nose grows absurdly long. The politicians boo the screens.'):
            with self.subTest(prompt=prompt):
                self.assertIsNotNone(music_narrative_row_issue(
                    {**row, 'video_prompt': prompt}, contract=contract, allocation=allocation, index=0))

    def test_later_clause_is_not_the_first_actions_object(self):
        contract = build_music_narrative_contract(
            'His nose grows absurdly long, crowd watches the screens.')
        row = {'source_event_ids': ['M1'], 'video_prompt':
               'His wooden nose grows absurdly long, extending toward the floor. The crowd watches the screens.'}
        self.assertIsNone(music_narrative_row_issue(row, contract=contract,
            allocation=[{'event_id': 'M1', 'clip_index': 0}], index=0))


if __name__ == '__main__':
    unittest.main()
