"""Regression for project 50f698d3: Markdown story and camera metadata."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.director.music_narrative import (
    build_music_narrative_contract, reconcile_music_narrative_checkpoint,
    music_narrative_row_issue,
)
from services.director.music_arc import normalize_music_arc
from services.director.music_arc_repair import allocation_report, apply_allocation_patch

BRIEF = '''# MUSIC VIDEO
## MAIN CHARACTERS
### THE TRAVELER
A traveler wears a black coat.
### THE SPEAKER
He gives theatrical speeches from enormous podiums while surrounded by devoted officials.
## STORY
### ACT I — OPENING
Rows of government officials applaud in perfect synchronization.
### ACT II — DEVELOPMENT
The traveler watches the ceremony.
### ACT III — FINALE
End with a dramatic cinematic image of the traveler walking away from the spectacle.
## CINEMATOGRAPHY
Use striking close-ups, sweeping camera movements, realistic crowds, sophisticated lighting.
## IMAGE-FIRST CHARACTER CONSISTENCY
Reuse these references throughout the production.
## MUSIC AND EDITING
Build intensity during choruses. Use the original song.
'''


def beats(count):
    return [{'clip_index': i, 'visual_beat': f'The courier examines parcel number {i}.',
             'purpose': 'Advance the journey', 'environment': 'Town square',
             'camera': 'Steady medium view', 'intentional_reprise': False,
             'reprise_reason': ''} for i in range(count)]


class MarkdownMusicRecoveryTests(unittest.TestCase):
    def test_markdown_story_keeps_all_acts_and_excludes_nonstory_sections(self):
        contract = build_music_narrative_contract(BRIEF)
        self.assertEqual(contract['source'], BRIEF)
        self.assertEqual([e['text'] for e in contract['events']], [
            'Rows of government officials applaud in perfect synchronization',
            'walking away from the spectacle'])
        self.assertEqual(contract['events'][1]['actor_aliases'], ['traveler'])
        self.assertIn('striking close-ups', contract['source'])

    def test_markdown_story_heading_variants(self):
        baseline = build_music_narrative_contract(BRIEF)['events']
        for title in ('## STORY:', '## Story ##', '## NARRATIVE', '## STORY ARC'):
            with self.subTest(title=title):
                self.assertEqual(build_music_narrative_contract(
                    BRIEF.replace('## STORY\n', title + '\n'))['events'], baseline)

    def test_colon_story_with_nested_markdown_acts(self):
        self.assertEqual(build_music_narrative_contract(BRIEF.replace('## STORY', 'STORY:'))['events'],
                         build_music_narrative_contract(BRIEF)['events'])

    def test_markdown_character_context_supports_collective_combat(self):
        source = ('## MAIN CHARACTERS\nDr Strange and Doctor Doom use their respective magic powers.\n'
                  '## STORY\n### ACT I\nThey fight using their respective powers.\n'
                  '## CINEMATOGRAPHY\nUse striking close-ups.')
        events = build_music_narrative_contract(source)['events']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['kind'], 'combat')

    def test_exact_failed_checkpoint_is_migrated_without_modifying_original(self):
        old = {'revision': 5, 'events': [
            {'event_id': 'M1', 'text': 'gives theatrical speeches'},
            {'event_id': 'M2', 'text': 'Rows of government officials applaud'},
            {'event_id': 'M3', 'text': 'walking away from the spectacle'},
            {'event_id': 'M4', 'text': 'striking close-ups'}]}
        checkpoint = {'music_narrative_contract': old,
                      'music_arc_allocation_repair': {'contract': old, 'response': ['rejected']},
                      'music_creative_arc': ['old'], 'music_video_batches': {'0': ['old']},
                      'audio_analysis': {'song': 'keep'}}
        before = copy.deepcopy(checkpoint)
        new = reconcile_music_narrative_checkpoint(checkpoint, build_music_narrative_contract(BRIEF))
        self.assertEqual(checkpoint, before)
        for key in ('music_arc_allocation_repair', 'music_creative_arc', 'music_video_batches'):
            self.assertNotIn(key, new)
        self.assertEqual(new['audio_analysis'], before['audio_analysis'])
        # Failed pre-arc checkpoints store the contract only inside the
        # pending repair, as in the actual saved project.
        checkpoint.pop('music_narrative_contract')
        before = copy.deepcopy(checkpoint)
        new = reconcile_music_narrative_checkpoint(checkpoint, build_music_narrative_contract(BRIEF))
        self.assertNotIn('music_arc_allocation_repair', new)
        self.assertEqual(checkpoint, before)

    def test_exact_failed_sequence_reports_all_ids_and_failed_patch_is_transactional(self):
        contract = {'events': [{'event_id': f'M{i}', 'text': f'action {i}'} for i in range(1, 5)]}
        draft = {'beats': beats(14), 'source_event_clips': {'M1': 1, 'M2': 2, 'M3': 13, 'M4': 6}}
        report = allocation_report(draft['source_event_clips'], contract, 14)
        self.assertEqual(report['expected'], ['M1', 'M2', 'M3', 'M4'])
        self.assertEqual(report['received'], ['M1', 'M2', 'M4', 'M3'])
        for field in ('missing', 'duplicate', 'unknown'):
            self.assertEqual(report[field], [])
        before = copy.deepcopy(draft)
        with self.assertRaisesRegex(ValueError, 'expected=.*received=.*out_of_order='):
            apply_allocation_patch(draft, {'source_event_clips': {
                'M1': 1, 'M2': 2, 'M3': 13, 'M4': 7}}, [{}] * 14, contract)
        self.assertEqual(draft, before)

    def test_repeated_chorus_motif_does_not_duplicate_story_ownership(self):
        contract = build_music_narrative_contract('The traveler walks away from the square.')
        rows = [{**row, 'source_event_ids': []} for row in beats(100)]
        for i in (10, 40, 70):
            rows[i].update(visual_beat='The red flag ripples above the crowd.',
                           intentional_reprise=True, reprise_reason='Recurring chorus visual motif.')
        rows[-1]['source_event_ids'] = ['M1']
        self.assertEqual(sum(len(r['source_event_ids']) for r in
                             normalize_music_arc(rows, [{}] * 100, contract)), 1)
        rows[40]['source_event_ids'] = ['M1']
        with self.assertRaisesRegex(ValueError, 'duplicate='):
            normalize_music_arc(rows, [{}] * 100, contract)

    def test_final_action_requires_real_actor_not_framing_imperative(self):
        contract = build_music_narrative_contract(BRIEF)
        allocation = [{'event_id': 'M1', 'clip_index': 0}, {'event_id': 'M2', 'clip_index': 1}]
        self.assertIsNone(music_narrative_row_issue(
            {'source_event_ids': ['M2'], 'video_prompt': 'The traveler walks away from the spectacle.'},
            contract=contract, allocation=allocation, index=1))


if __name__ == '__main__':
    unittest.main()
