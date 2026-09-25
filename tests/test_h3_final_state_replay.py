"""An already-achieved ending must not reopen and re-close its prop."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import extract_source_events, segment_violations


class FinalStateReplayTests(unittest.TestCase):
    def check(self, action, *, assigned_action=False, ending="closed"):
        source = (
            "Priya opens the workshop door. Sam carries a banner inside. "
            "Priya closes the workshop door. "
            + ("Sam opens the workshop door again." if assigned_action
               else f"End with the workshop door {ending}.")
        )
        events = extract_source_events(source)
        beat = {"beat_id": "B4", "source_event_ids": [events[-1]["event_id"]],
                "dialogue_ids": [], "description": events[-1]["text"],
                "_final_state_only": not assigned_action,
                "_transition_after_source_event_ids": [event["event_id"] for event in events[:-1]]}
        segment = {"segment": 4, "semantic_actions": True, "camera_contract": "event_cards",
                   "opening_state": "Priya and Sam are inside; the workshop door is closed.",
                   "closing_state": "The workshop door is closed.", "shots": [{
                       "shot": 1, "beat_ids": ["B4"], "start_seconds": 0., "end_seconds": 4.,
                       "action": action, "framing": "Wide view", "camera": "Locked view",
                       "sound_effects": "", "dialogue": [],
                   }]}
        return segment_violations(source, segment, segment_number=4, duration=4.,
                                  assigned_beats=[beat], dialogue_catalog=[])

    def same_window_check(self, source_actions, visible_prior_actions, endpoint, endpoint_action,
                          *, opening=None, closing=None):
        source = " ".join([*source_actions, f"End with {endpoint}."])
        events = extract_source_events(source)
        beats = []
        shots = []
        for index, event in enumerate(events[:-1]):
            beat_id = f"B{index + 1}"
            beats.append({
                "beat_id": beat_id, "source_event_ids": [event["event_id"]],
                "dialogue_ids": [], "description": event["text"],
            })
            visible = visible_prior_actions[index] if index < len(visible_prior_actions) else ""
            shots.append({
                "shot": index + 1, "beat_ids": [beat_id], "start_seconds": index * 2.,
                "end_seconds": (index + 1) * 2., "action": visible,
                "framing": "Wide view", "camera": "Locked view", "sound_effects": "",
                "dialogue": [],
            })
        final_event = events[-1]
        final_beat_id = f"B{len(beats) + 1}"
        beats.append({
            "beat_id": final_beat_id, "source_event_ids": [final_event["event_id"]],
            "dialogue_ids": [], "description": final_event["text"],
            "_final_state_only": True, "_transition_after_source_event_ids": [],
        })
        final_index = len(shots)
        shots.append({
            "shot": final_index + 1, "beat_ids": [final_beat_id],
            "start_seconds": final_index * 2., "end_seconds": (final_index + 1) * 2.,
            "action": endpoint_action, "framing": "Wide view", "camera": "Locked view",
            "sound_effects": "", "dialogue": [],
        })
        segment = {
            "segment": 1, "semantic_actions": True, "camera_contract": "event_cards",
            "opening_state": opening or f"The {endpoint} is open.",
            # Deliberately untrusted: only earlier shot actions may establish history.
            "closing_state": closing or f"The {endpoint} is closed.", "shots": shots,
        }
        return segment_violations(source, segment, segment_number=1,
            duration=max(4., len(shots) * 2.), assigned_beats=beats, dialogue_catalog=[])

    def test_finite_repeat_or_wrong_owner_reset_is_rejected(self):
        for action in (
            "Priya opens the workshop door and closes the workshop door again.",
            "Sam closes the workshop door. The workshop door is closed behind Sam.",
        ):
            with self.subTest(action=action):
                self.assertTrue(any("replays completed" in error for error in self.check(action)))

    def test_state_observation_does_not_replay_prior_actions(self):
        errors = self.check("Priya and Sam remain inside. The workshop door is still closed behind them.")
        self.assertFalse(any("replays completed" in error for error in errors), errors)

    def test_an_explicit_new_open_action_remains_allowed(self):
        errors = self.check("Sam opens the workshop door again.", assigned_action=True)
        self.assertFalse(any("replays completed" in error for error in errors), errors)

    def test_unrelated_prop_change_does_not_borrow_door_history(self):
        errors = self.check("Sam closes a small jewelry box. The workshop door remains closed behind Priya.")
        self.assertFalse(any("replays completed" in error for error in errors), errors)

    def test_different_requested_ending_can_require_another_transition(self):
        errors = self.check("Priya opens the workshop door. The workshop door is open.", ending="open")
        self.assertFalse(any("replays completed" in error for error in errors), errors)

    def test_same_window_prior_visible_close_blocks_repeated_close(self):
        errors = self.same_window_check(
            ["Priya closes the workshop door."],
            ["Priya closes the workshop door."],
            "the workshop door closed", "Sam closes the workshop door again.",
        )
        self.assertTrue(any("replays an already visible closed transition" in error for error in errors), errors)

    def test_closing_claim_without_prior_visible_close_is_not_evidence(self):
        errors = self.same_window_check(
            ["Priya closes the workshop door."],
            ["Priya reaches toward the workshop door."],
            "the workshop door closed", "Priya closes the workshop door.",
            closing="The workshop door is already closed.",
        )
        self.assertFalse(any("replays an already visible" in error for error in errors), errors)

    def test_different_named_door_does_not_share_prior_close_evidence(self):
        errors = self.same_window_check(
            ["Priya closes the workshop door."],
            ["Priya closes the workshop door."],
            "the archive door closed", "Sam closes the archive door.",
        )
        self.assertFalse(any("replays an already visible" in error for error in errors), errors)

    def test_intervening_visible_open_allows_a_needed_reclose(self):
        errors = self.same_window_check(
            ["Priya closes the workshop door.", "Sam opens the workshop door."],
            ["Priya closes the workshop door.", "Sam opens the workshop door."],
            "the workshop door closed", "Priya closes the workshop door.",
        )
        self.assertFalse(any("replays an already visible" in error for error in errors), errors)

    def test_differently_colored_door_does_not_share_prior_close_evidence(self):
        errors = self.same_window_check(
            ["Priya closes the blue door."], ["Priya closes the blue door."],
            "the red door closed", "Sam closes the red door.",
        )
        self.assertFalse(any("replays an already visible" in error for error in errors), errors)

    def test_same_colored_prop_still_matches_and_unrelated_color_does_not(self):
        repeated = self.same_window_check(
            ["Priya closes the red door."], ["Priya closes the red door."],
            "the red door closed", "Sam closes the red door.",
        )
        self.assertTrue(any("replays an already visible closed transition" in error for error in repeated), repeated)

        unrelated = self.same_window_check(
            ["Priya closes the blue door."],
            ["Priya closes the blue door while her red gloves catch the light."],
            "the red door closed", "Sam closes the red door.",
        )
        self.assertFalse(any("replays an already visible" in error for error in unrelated), unrelated)


if __name__ == "__main__":
    unittest.main()
