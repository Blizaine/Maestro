"""AI connective staging cannot consume a later immutable source event."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import (
    _camera_phase_beats, _h3_preview_action_frames,
    _h3_preview_action_matches, extract_source_events,
    segment_violations,
)


class ConnectiveSourceSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.prompt = (
            "Each evening Riva leaves a bouquet beneath the station clock. "
            "On the last night Sol recognizes the note and places a coin beside the flowers. No dialogue."
        )
        self.events = extract_source_events(self.prompt)
        self.schedule = [
            dict(beat_id="B1", segment=1, source_event_ids=[self.events[0]["event_id"]],
                 dialogue_ids=[], description=self.events[0]["text"], state_after="The bouquet rests beneath the clock."),
            dict(beat_id="B2", segment=2, source_event_ids=[], dialogue_ids=[],
                 description="On the next evening Riva leaves another bouquet beneath the station clock.",
                 state_after="Two bouquets rest beneath the clock."),
            dict(beat_id="B3", segment=3, source_event_ids=[], dialogue_ids=[],
                 description="Sol approaches the flowers and recognizes the note.",
                 state_after="Sol has recognized the note."),
            dict(beat_id="B4", segment=4, source_event_ids=[e["event_id"] for e in self.events[1:]],
                 dialogue_ids=[], description=self.events[-1]["text"], state_after="The reply is visible."),
        ]

    def phases(self, index):
        return _camera_phase_beats([self.schedule[index]], source_events=self.events,
            expected_dialogue_events={}, preserve_adaptation=True, full_schedule_beats=self.schedule)

    def test_early_recognition_is_not_sent_as_executable_staging(self):
        before = [dict(event) for event in self.events]
        phase = self.phases(2)[0]
        self.assertIn("recognizes the note", phase["_discarded_early_staging"])
        self.assertNotIn("recognizes", phase["description"])
        self.assertNotIn("recognized", phase["state_after"])
        self.assertEqual(self.events, before)
        self.assertEqual(phase["_transition_before_source_event_ids"], [self.events[1]["event_id"]])

    def test_ordinary_approach_can_remain_as_connective_staging(self):
        self.schedule[2]["description"] = "Sol approaches the station clock, carrying a folded umbrella."
        phase = self.phases(2)[0]
        self.assertNotIn("_discarded_early_staging", phase)
        self.assertEqual(phase["description"], self.schedule[2]["description"])

    def test_scheduled_later_occasion_avoids_extra_early_recurrence(self):
        phases = self.phases(0)
        self.assertEqual(len(phases), 1)
        self.assertNotIn("_spaced_recurrence", phases[0])
        self.assertEqual(phases[0]["_recurrence_continues_in_schedule"], [self.events[0]["event_id"]])

    def test_role_pronouns_do_not_assign_recognition_to_the_gaze(self):
        source = (
            "Each evening an adult florist leaves a bouquet beneath a station clock. "
            "On the last night, an adult commuter recognizes their own handwriting "
            "on its wrapping and leaves one fresh stem in reply. No dialogue."
        )
        events = extract_source_events(source)
        beat = dict(beat_id="B2", segment=2, source_event_ids=[], dialogue_ids=[],
            description=("An adult commuter approaches the bouquet. Their gaze settles on the wrapping. "
                         "They recognize their own handwriting and pause."),
            sound_effects="A sound marks the recognition.")
        schedule = [
            dict(beat_id="B1", segment=1, source_event_ids=[events[0]["event_id"]], dialogue_ids=[],
                 description=events[0]["text"]), beat,
            dict(beat_id="B3", segment=3, source_event_ids=[e["event_id"] for e in events[1:]],
                 dialogue_ids=[], description=events[-1]["text"]),
        ]
        phase = _camera_phase_beats([beat], source_events=events,
            expected_dialogue_events={}, preserve_adaptation=True, full_schedule_beats=schedule)[0]
        self.assertIn("_discarded_early_staging", phase)
        self.assertNotIn("recognition", phase["sound_effects"])

    def test_unrelated_time_cut_does_not_count_as_source_recurrence(self):
        self.schedule[1]["description"] = "On the next evening Sol polishes a silver bell."
        phases = self.phases(0)
        self.assertEqual(len(phases), 2)
        self.assertEqual(phases[1]["_spaced_recurrence"]["occurrence"], 2)

    def test_different_action_on_same_prop_does_not_erase_required_recurrence(self):
        self.schedule[1]["description"] = "On the next evening Riva removes the bouquet beneath the station clock."
        self.assertEqual(len(self.phases(0)), 2)

    def test_action_before_the_time_jump_does_not_count_as_later_recurrence(self):
        before_jump = (
            "Riva leaves another bouquet beneath the station clock in the same evening. "
            "A time-lapse transition then jumps to the next evening."
        )
        self.schedule[1]["description"] = before_jump
        self.assertEqual(len(self.phases(0)), 2)
        self.schedule[1]["description"] = "On the next evening Riva leaves another bouquet beneath the station clock."
        phase = self.phases(1)[0]
        segment = dict(segment=2, semantic_actions=True, camera_contract="event_cards",
            closing_state="The station is quiet.", shots=[dict(shot=1,
                beat_ids=[phase["beat_id"]], start_seconds=0, end_seconds=4,
                action=before_jump, camera="Wide view", framing="Wide view", dialogue=[])])
        errors = segment_violations(self.prompt, segment, segment_number=2, duration=4,
            assigned_beats=[phase], dialogue_catalog=[])
        self.assertTrue(any("later occasion" in e for e in errors), errors)

    def test_later_recurrence_remains_a_camera_obligation(self):
        phase = self.phases(1)[0]
        self.assertEqual(phase["source_event_ids"], [])
        self.assertEqual(phase["_scheduled_recurrence_source_event_ids"], [self.events[0]["event_id"]])
        segment = dict(segment=2, semantic_actions=True, camera_contract="event_cards",
            closing_state="The station is quiet.", shots=[dict(shot=1,
                beat_ids=[phase["beat_id"]], start_seconds=0, end_seconds=4,
                action="On the next evening Riva removes the bouquet beneath the station clock.",
                camera="Wide view", framing="Wide view", dialogue=[])])
        errors = segment_violations(self.prompt, segment, segment_number=2, duration=4,
            assigned_beats=[phase], dialogue_catalog=[])
        self.assertTrue(any("required source action on the distinct later occasion" in e for e in errors), errors)

    def test_recognition_requires_the_same_actor_and_object(self):
        required = _h3_preview_action_frames("Sol recognizes the note.", None)
        good = _h3_preview_action_frames("Sol recognizes the note in their hands.", None)
        wrong = _h3_preview_action_frames("Riva recognizes a station employee.", None)
        self.assertTrue(required)
        self.assertTrue(any(_h3_preview_action_matches(a, b) for a in required for b in good))
        self.assertFalse(any(_h3_preview_action_matches(a, b) for a in required for b in wrong))

    def test_music_and_ongoing_words_do_not_exempt_completed_one_off_actions(self):
        for source_action, visible, should_replay in (
            ("Nora performs a short solo.", "Nora performs a short solo again.", True),
            ("Nora keeps the door closed and opens the drawer.",
             "Nora opens the drawer again.", True),
            ("Nora plays the piano.", "Nora plays the piano softly.", False),
            ("Nora keeps playing the piano.", "Nora plays the piano softly.", False),
            ("Nora dances and sings to the music.", "Nora dances and sings to the music.", False),
            ("Nora dances and closes the door.", "Nora closes the door again.", True),
        ):
            with self.subTest(source=source_action):
                prompt = f"[0s-4s] {source_action} [4s-8s] Lee places a score on the chair."
                events = extract_source_events(prompt)
                beat = dict(beat_id="B2", source_event_ids=[], dialogue_ids=[],
                    description="A connective visual beat.", _audio_driven=True,
                    _transition_after_source_event_ids=[events[0]["event_id"]])
                segment = dict(segment=2, semantic_actions=True,
                    closing_state="The room remains quiet.", shots=[dict(shot=1,
                        beat_ids=["B2"], start_seconds=0, end_seconds=4,
                        action=visible, camera="Wide view", framing="Wide view", dialogue=[])])
                errors = segment_violations(prompt, segment, segment_number=2, duration=4,
                    assigned_beats=[beat], dialogue_catalog=[])
                self.assertEqual(any("replays completed" in e for e in errors), should_replay, errors)


if __name__ == "__main__":
    unittest.main()
