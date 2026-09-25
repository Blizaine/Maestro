"""Conservative checks for source-authorized, visibly spaced recurrence."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_recurrence import (  # noqa: E402
    action_establishes_later_occasion,
    allows_spaced_recurrence,
    explicitly_requests_spaced_recurrence,
    expand_spaced_recurrence_phases,
    later_occasion_action_text,
    recurring_source_event_ids,
)


class SpacedRecurrenceTests(unittest.TestCase):
    def test_camera_phases_show_two_occasions_without_duplicating_source_ids(self):
        source = {"event_id": "E1", "text": "Each evening Nora leaves one fresh bouquet by the clock."}
        phase = {"beat_id": "B1", "source_event_ids": ["E1"], "dialogue_ids": [],
                 "description": source["text"], "_action_seconds": 6.0,
                 "_start_frame_continuation": True}
        phases = expand_spaced_recurrence_phases([phase], [source])
        self.assertEqual([item["beat_id"] for item in phases], ["B1.R1", "B1.R2"])
        self.assertEqual([item["source_event_ids"] for item in phases], [["E1"], ["E1"]])
        self.assertEqual(sum(item["_action_seconds"] for item in phases), 6.0)
        self.assertIn("On one evening Nora leaves one fresh bouquet", phases[0]["_canonical_action"])
        self.assertIn("On a subsequent evening Nora leaves one fresh bouquet", phases[1]["_canonical_action"])
        self.assertTrue(action_establishes_later_occasion(phases[1]["_canonical_action"]))
        self.assertNotIn("_start_frame_continuation", phases[1])
        self.assertEqual(source["text"], "Each evening Nora leaves one fresh bouquet by the clock.")
        self.assertEqual(phase["beat_id"], "B1")
        self.assertEqual(expand_spaced_recurrence_phases(phases, [source]), phases)

    def test_recurrence_does_not_duplicate_speech_timed_events_or_unique_reply(self):
        events = [
            {"event_id": "E1", "text": "Each evening Nora leaves a bouquet."},
            {"event_id": "E2", "text": "On the last evening Ivo leaves a reply."},
        ]
        phases = [
            {"beat_id": "B1", "source_event_ids": ["E1"], "dialogue_ids": ["D1"]},
            {"beat_id": "B2", "source_event_ids": ["E2"], "dialogue_ids": []},
        ]
        self.assertEqual(expand_spaced_recurrence_phases(phases, events), phases)
        phases[0]["dialogue_ids"] = []
        events[0]["start_seconds"], events[0]["end_seconds"] = 0, 4
        self.assertEqual(expand_spaced_recurrence_phases(phases, events), phases)
        del events[0]["start_seconds"], events[0]["end_seconds"]
        events[0]["source_start_seconds"], events[0]["source_end_seconds"] = 0, 4
        self.assertEqual(expand_spaced_recurrence_phases(phases, events), phases)

    def test_normal_retries_and_quoted_time_words_stay_authored(self):
        for text in ("Nora opens, closes and reopens the door.",
                     "Nora reads a note that says 'each evening'.",
                     "Every other evening Nora leaves one fresh bouquet.",
                     "On dawn one, dawn two and dawn three Nora waters the tray."):
            event = {"event_id": "E1", "text": text}
            phase = {"beat_id": "B1", "source_event_ids": ["E1"], "dialogue_ids": []}
            self.assertEqual(expand_spaced_recurrence_phases([phase], [event]), [phase])

    def test_recognizes_explicit_time_and_occasion_intervals(self):
        for phrase in (
            "Each morning the florist waters the display.",
            "Every evening she leaves a bouquet by the station clock.",
            "Each night the lighthouse keeper checks the lamp.",
            "Every day the child visits the old tree.",
            "Each week they return to the workshop.",
            "Every month the volunteers refresh the mural.",
            "Each year the town lights the lanterns.",
            "Every shift the nurse updates the board.",
            "Each visit the guide points out the fountain.",
            "Each and every morning the baker lights the oven.",
            "Every other evening the florist checks the display.",
            "Nora's note is signed; every morning she returns to the station.",
            "Night after night, the beacon sweeps the harbor.",
            "Day after day the gardener trims the roses.",
            "The character says, 'every morning', before the real routine begins each evening.",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(explicitly_requests_spaced_recurrence(phrase))

    def test_does_not_infer_recurrence_from_people_again_or_final_night(self):
        for phrase in (
            "Every person carries one bouquet.",
            "Each musician takes an instrument.",
            "The florist leaves a bouquet again.",
            "On the last night, the commuter leaves a stem.",
            "The florist visits the station each time the clock chimes.",
            "The character reads a note saying 'every morning'.",
            "Nora's note says 'every morning'.",
            "A line of dialogue says, “each evening.”",
        ):
            with self.subTest(phrase=phrase):
                self.assertFalse(explicitly_requests_spaced_recurrence(phrase))

    def test_requires_a_distinct_later_occasion_in_local_action(self):
        for action in (
            "The next evening, the florist leaves another bouquet.",
            "On the following night, she returns to the clock.",
            "A week later, she places a fresh bouquet.",
            "Several days later the keeper checks the lantern again.",
            "Night after night, the florist leaves a new bouquet.",
            "On a separate visit, she refreshes the display.",
            "EXT. TRAIN STATION - NEXT EVENING",
            "A week later, the florist leaves another bouquet.",
            "Montage across successive evenings: she leaves one bouquet each time.",
        ):
            with self.subTest(action=action):
                self.assertTrue(action_establishes_later_occasion(action))

        for action in (
            "The florist leaves a bouquet again.",
            "On the last night, the commuter leaves a stem.",
            "Later that evening, she adjusts the same bouquet.",
            "The camera shows the florist leave a bouquet.",
            "The same evening continues as she leaves a bouquet.",
            "She reads a sign saying 'next evening' and repeats the placement.",
            "She thinks about the following evening before repeating the placement.",
            "No new day passes as she repeats the placement.",
            "The next day never arrives, so she repeats the same placement.",
            "Nora's note says 'next evening'; she repeats the placement now.",
            "On the next evening, she does not return to the station.",
        ):
            with self.subTest(action=action):
                self.assertFalse(action_establishes_later_occasion(action))

    def test_shown_elapsed_time_can_anchor_a_nonleading_later_occasion(self):
        for action in (
            "The clock face is shown in close-up, the hands ticking forward to indicate the passage of time to a subsequent evening.",
            "The camera holds on the large vintage clock face. The second hand sweeps visibly across the dial, passing the minute markers to establish the passage of time into a distinct subsequent evening.",
            "A calendar page flips forward to the following day before Nora returns to the station.",
            "The platform lighting shifts from dusk into another evening as the florist approaches.",
            "A time-lapse transition jumps to the next evening, when the keeper relights the lamp.",
            "The clock face dominates the frame, its hands sweeping past the twelve in a subtle, accelerated time-lapse to mark the passage of several nights.",
            "A time-lapse or subtle change in the clock’s hands indicates the passage of several nights.",
        ):
            with self.subTest(action=action):
                self.assertTrue(action_establishes_later_occasion(action))

        for action in (
            "The clock hands tick steadily while the florist talks about the following evening.",
            "The clock hands tick forward while the florist mentions a subsequent evening.",
            "The clock hands tick forward to ‘a subsequent evening.’",
            "The clock hands do not tick forward into the next day.",
            "Mara describes the clock hands ticking forward into the next day.",
            "Mara thinks about the clock hands ticking forward into the following evening.",
            "Nora imagines a calendar page flipping to the next day.",
            "The florist plans for the lighting to shift into another evening.",
            "A sign reads ‘the next evening’ while the clock is shown and the florist repeats the placement now.",
            "The calendar page does not flip to the following day; the same evening continues.",
            "The clock hands tick forward, but no new evening arrives.",
            "The clock hands tick steadily as the florist repeats the placement.",
            "The second hand sweeps visibly across the dial, passing the minute markers, but no later evening is shown.",
            "The clock's hands will be shown in time-lapse to mark passage of several nights.",
        ):
            with self.subTest(action=action):
                self.assertFalse(action_establishes_later_occasion(action))

    def test_c4_clock_transitions_only_authorize_an_explicitly_recurring_event(self):
        source_events = [
            {"event_id": "E1", "text": "Each evening the florist leaves one fresh bouquet by the clock."},
            {"event_id": "E2", "text": "On the last evening the commuter leaves a stem."},
        ]
        recurring = recurring_source_event_ids(source_events)
        transition = (
            "The camera holds on the large vintage clock face. The second hand sweeps visibly across the dial, "
            "passing the minute markers to establish the passage of time into a distinct subsequent evening."
        )
        self.assertTrue(action_establishes_later_occasion(transition))
        self.assertEqual(later_occasion_action_text(transition), "")

        later_action = transition + " Then the florist leaves one fresh bouquet by the clock."
        self.assertIn("the florist leaves one fresh bouquet", later_occasion_action_text(later_action))
        self.assertTrue(allows_spaced_recurrence("E1", recurring, later_action))
        self.assertFalse(allows_spaced_recurrence("E2", recurring, later_action))

    def test_repeated_action_before_later_time_jump_is_outside_recurrence_scope(self):
        before_transition = (
            "Nora leaves a fresh bouquet again in the same evening. "
            "A time-lapse transition then jumps to the next evening."
        )
        self.assertTrue(action_establishes_later_occasion(before_transition))
        self.assertEqual(later_occasion_action_text(before_transition), "")

        after_transition = (
            "A time-lapse transition jumps to the next evening. "
            "Nora leaves a fresh bouquet at the station clock."
        )
        self.assertIn("Nora leaves a fresh bouquet", later_occasion_action_text(after_transition))

    def test_later_action_scope_keeps_leading_and_elapsed_time_cues(self):
        self.assertIn(
            "Nora leaves a fresh bouquet",
            later_occasion_action_text("The following evening, Nora leaves a fresh bouquet."),
        )
        self.assertIn(
            "the florist leaves one fresh bouquet",
            later_occasion_action_text(
                "A time-lapse marks the passage of several nights. "
                "Then the florist leaves one fresh bouquet."
            ),
        )
        self.assertEqual(
            later_occasion_action_text(
                "A clock is planned to show the next evening. The florist leaves a bouquet."
            ),
            "",
        )

    def test_later_action_scope_keeps_action_after_inline_scene_slugline(self):
        for separator in (":", "—", "-"):
            slugline = f"EXT. STATION - NEXT EVENING{separator} Nora leaves a fresh bouquet."
            with self.subTest(slugline=slugline):
                self.assertIn("Nora leaves a fresh bouquet", later_occasion_action_text(slugline))

        self.assertIn(
            "Nora leaves a fresh bouquet",
            later_occasion_action_text("EXT. STATION - NEXT EVENING\nNora leaves a fresh bouquet."),
        )

    def test_visual_transition_then_clock_readout_establishes_later_occasion(self):
        for action in (
            "A slow dissolve transition indicates the passage of time. "
            "The clock now reads 7:00 PM on a different evening. "
            "Then the same florist places one fresh bouquet on the platform.",
            "A slow dissolve transition indicates the passage of time. "
            "The clock now reads 7:00 PM on the third evening. "
            "Then the same florist places one fresh bouquet on the platform.",
            "A dissolve transitions to the third evening. "
            "The florist places one fresh bouquet by the clock.",
        ):
            with self.subTest(action=action):
                self.assertTrue(action_establishes_later_occasion(action))
                scoped = later_occasion_action_text(action)
                self.assertIn("florist places one fresh bouquet", scoped)
                self.assertNotIn("dissolve", scoped)
                self.assertNotIn("clock now reads", scoped)

    def test_specific_dissolve_plus_third_evening_clock_matches_saved_replay(self):
        action = (
            "Another slow dissolve transition. "
            "The clock reads 7:00 PM on a third evening. "
            "The adult florist enters from the left and places a third wrapped "
            "bouquet beside the others."
        )
        self.assertTrue(action_establishes_later_occasion(action))
        scoped = later_occasion_action_text(action)
        self.assertIn("The adult florist enters from the left", scoped)
        self.assertIn("places a third wrapped bouquet", scoped)
        self.assertNotIn("dissolve", scoped)
        self.assertNotIn("clock reads", scoped)

    def test_specific_dissolve_requires_an_immediate_visible_later_clock(self):
        for action in (
            "Another slow dissolve transition.",
            "Another transition. The clock reads 7:00 PM on the third evening.",
            "Another slow dissolve transition. Nora checks the bouquet. "
            "The clock reads 7:00 PM on the third evening.",
            "Mara describes another slow dissolve transition. "
            "The clock reads 7:00 PM on the third evening.",
            "A slow dissolve transition is planned. "
            "The clock reads 7:00 PM on the third evening.",
            "No slow dissolve transition. "
            "The clock reads 7:00 PM on the third evening.",
        ):
            with self.subTest(action=action):
                self.assertFalse(action_establishes_later_occasion(action))
                self.assertEqual(later_occasion_action_text(action), "")

    def test_adjacent_visual_time_cue_does_not_accept_narration_plans_or_negation(self):
        for action in (
            "Mara describes a slow dissolve transition indicating passage of time. "
            "The clock now reads 7:00 PM on the third evening. "
            "Then the florist places a bouquet.",
            "A slow dissolve transition is planned to indicate passage of time. "
            "The clock now reads 7:00 PM on a different evening. "
            "Then the florist places a bouquet.",
            "A slow dissolve transition does not indicate passage of time. "
            "The clock now reads 7:00 PM on a different evening. "
            "Then the florist places a bouquet.",
            "A slow dissolve transition indicates passage of time. "
            "Mara says the clock reads 7:00 PM on the third evening. "
            "Then the florist places a bouquet.",
            "A slow dissolve transition indicates passage of time. "
            "The clock is planned to read 7:00 PM on a different evening. "
            "Then the florist places a bouquet.",
            "A slow dissolve transition indicates passage of time. "
            "The clock now reads 7:00 PM on the first evening. "
            "Then the florist places a bouquet.",
            "‘A slow dissolve transition indicates passage of time.’ "
            "The clock now reads 7:00 PM on the third evening. "
            "Then the florist places a bouquet.",
        ):
            with self.subTest(action=action):
                self.assertFalse(action_establishes_later_occasion(action))
                self.assertEqual(later_occasion_action_text(action), "")

    def test_adjacent_clock_scope_excludes_actions_before_the_transition(self):
        before_only = (
            "The florist places one fresh bouquet now. "
            "A slow dissolve transition indicates the passage of time. "
            "The clock now reads 7:00 PM on a different evening."
        )
        self.assertTrue(action_establishes_later_occasion(before_only))
        self.assertEqual(later_occasion_action_text(before_only), "")

        after_transition = before_only + " Then the florist places another bouquet."
        self.assertIn(
            "Then the florist places another bouquet",
            later_occasion_action_text(after_transition),
        )

    def test_later_action_scope_rejects_planned_negated_and_narrated_actions(self):
        cue = (
            "A slow dissolve transition indicates the passage of time. "
            "The clock now reads 7:00 PM on a different evening. "
        )
        for action in (
            "Then the florist plans to place a bouquet.",
            "Then the florist does not place a bouquet.",
            "Then Mara says the florist places a bouquet.",
            "‘The florist places a bouquet.’",
        ):
            with self.subTest(action=action):
                self.assertTrue(action_establishes_later_occasion(cue + action))
                self.assertEqual(later_occasion_action_text(cue + action), "")

    def test_only_the_explicitly_recurring_source_event_can_replay(self):
        events = [
            {"event_id": "E1", "text": "Each evening the florist leaves a bouquet."},
            {"event_id": "E2", "text": "On the last night the commuter leaves a stem."},
        ]
        recurring = recurring_source_event_ids(events)
        self.assertEqual(recurring, frozenset({"E1"}))

        action = "The following evening, the florist leaves another bouquet."
        self.assertTrue(allows_spaced_recurrence("e1", recurring, action))
        self.assertFalse(allows_spaced_recurrence("E2", recurring, action))
        self.assertFalse(allows_spaced_recurrence("E1", recurring, "She leaves it again."))
        self.assertFalse(allows_spaced_recurrence("", recurring, action))

    def test_same_time_duplicate_and_other_event_do_not_get_exception(self):
        recurring = recurring_source_event_ids([
            {"event_id": "E4", "text": "Every evening the keeper trims the wick."},
        ])
        self.assertFalse(allows_spaced_recurrence(
            "E4", recurring, "That evening the keeper trims the wick again."
        ))
        self.assertFalse(allows_spaced_recurrence(
            "E3", recurring, "The next evening the commuter leaves a flower."
        ))


if __name__ == "__main__":
    unittest.main()
