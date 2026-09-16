"""Filmable-clock regressions for dense H3 choreography."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_action_clock import estimate_h3_action_seconds
from services.h3_story_ledger import _h3_filmable_shot_durations


class H3ActionClockTests(unittest.TestCase):
    def test_compound_exchange_outweighs_aftermath_camera_tour(self):
        exchange = (
            "Dev pushes up from one knee, closes the distance with a flurry of palm strikes. "
            "Mara blocks each blow, catches his wrist, twists her hips to spin him, and hurls him forward."
        )
        aftermath = (
            "The camera executes a fast orbit around the ruined courtyard. Dust hangs in golden light "
            "while the fighters struggle against the rubble."
        )
        self.assertGreaterEqual(estimate_h3_action_seconds(exchange), 6.0)
        self.assertGreater(estimate_h3_action_seconds(exchange), 2 * estimate_h3_action_seconds(aftermath))

    def test_crash_follow_and_landing_receive_visible_time(self):
        action = (
            "Dev crashes through the wall and tumbles beyond it. Mara accelerates through "
            "the opening, follows him, and lands inside the broken cavity."
        )
        self.assertGreaterEqual(estimate_h3_action_seconds(action), 5.0)

    def test_dense_actions_receive_most_of_a_silent_window(self):
        texts = [
            "Dev rises, closes the distance, strikes, is blocked, gets caught, spun, and hurled.",
            "He crashes through a wall, tumbles; Mara follows and lands.",
            "They struggle in rubble.",
            "The camera orbits the wreckage and surveys the courtyard.",
        ]
        weights = [estimate_h3_action_seconds(text) for text in texts]
        durations = _h3_filmable_shot_durations(weights, 13.625)
        self.assertGreaterEqual(durations[0], 4.5)
        self.assertGreaterEqual(durations[1], 3.0)
        self.assertGreater(sum(durations[:2]), sum(durations[2:]))

    def test_event_floor_and_short_action_remain_bounded(self):
        self.assertEqual(estimate_h3_action_seconds("Dust hangs in the air.", event_floor=2), 2.0)
        self.assertLessEqual(estimate_h3_action_seconds("Punch and kick. " * 30), 6.5)

    def test_office_blocking_keeps_existing_weights(self):
        action = "Mara sits, looks at Dev, reaches for the plan, raises it, opens the door, and stands."
        self.assertEqual(estimate_h3_action_seconds(action), 6.5)
        simple = "Mara sits and looks at Dev."
        self.assertAlmostEqual(estimate_h3_action_seconds(simple), 2.9)

    def test_catch_is_counted_once_as_blocking(self):
        self.assertEqual(estimate_h3_action_seconds("Mara catches Dev's wrist."), 2.0)

    def test_camera_owned_motion_does_not_gain_actor_time(self):
        self.assertEqual(
            estimate_h3_action_seconds("The camera moves around the office and rises above the table."),
            1.0,
        )

    def test_tumble_inflections_are_recognized(self):
        for word in ("tumble", "tumbles", "tumbled", "tumbling"):
            with self.subTest(word=word):
                self.assertEqual(estimate_h3_action_seconds(f"Mara {word} through the doorway."), 2.75)

    def test_camera_led_sentence_retains_embedded_actor_action(self):
        self.assertEqual(
            estimate_h3_action_seconds("The camera follows Mara as she crosses the room."),
            estimate_h3_action_seconds("Mara crosses the room."),
        )

    def test_open_state_does_not_inflate_action_time(self):
        for state in ("the open doorway", "his open hand", "the door remains open", "the opening shot"):
            with self.subTest(state=state):
                self.assertEqual(estimate_h3_action_seconds(state), 1.0)
        self.assertGreater(estimate_h3_action_seconds("Mara opens the door."), 1.0)


if __name__ == "__main__":
    unittest.main()
