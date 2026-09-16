"""Production directions constrain a story without consuming an event slot."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import extract_h3_source_intent, extract_source_events, extract_locked_dialogue


class DirectionClassificationTests(unittest.TestCase):
    def test_brief_slow_motion_does_not_dilute_requested_fast_action(self):
        source = ("The duel uses a brief slow-motion fist wind-up, then snaps to ultra-high-speed "
                  "punches and rapid counters. Show both fighters recoiling and immediately attacking again.")
        intent = extract_h3_source_intent(source)
        self.assertTrue(intent["fast_action"])
        self.assertIn("slow-motion accents", intent["pacing_contract"])
        self.assertIn("high-speed bursts", intent["pacing_contract"])
        self.assertNotIn("no slow motion", intent["pacing_contract"])
        self.assertLessEqual(len(intent["pacing_contract"]), 180)

    def test_slow_scene_without_fast_action_does_not_gain_explosive_speed(self):
        intent = extract_h3_source_intent("A dancer raises one hand in gentle slow motion.")
        self.assertFalse(intent["fast_action"])
        self.assertIn("preserve slow-motion", intent["pacing_contract"])
        self.assertNotIn("high-speed", intent["pacing_contract"])

    def test_global_camera_direction_is_retained_without_final_story_event(self):
        source = "Mara ducks the punch. Dev crashes through the wall. Use dynamic camera movements."
        events = extract_source_events(source)
        self.assertEqual(len(events), 2)
        self.assertIn("crashes through the wall", events[-1]["text"])
        intent = extract_h3_source_intent(source)
        self.assertIn("Use dynamic camera movements", intent["perspective_contract"])
        self.assertIn("Use dynamic camera movements", intent["global_instructions"])

    def test_choreography_writing_request_is_context(self):
        direction = "write the shot for shot continuously chained fighting choreography"
        source = f"Mara fights Dev. {direction}. Dev falls into the fountain."
        self.assertFalse(any("write" in e["text"] for e in extract_source_events(source)))
        self.assertIn(direction, extract_h3_source_intent(source)["global_instructions"])

    def test_specific_camera_reveal_and_actor_camera_use_remain_events(self):
        for source in (
            "Use a camera movement to reveal Dev hidden behind the door.",
            "The camera circles the pillar to reveal Mara on the far side.",
            "Mara uses her camera to photograph Dev.",
            "Mara writes fighting choreography on the blackboard.",
        ):
            with self.subTest(source=source):
                self.assertNotEqual(extract_source_events(source)[0]["text"], "Establish and carry out the requested scene")
                self.assertFalse(extract_h3_source_intent(source)["global_instructions"])

    def test_spoken_direction_is_still_exact_dialogue(self):
        source = 'Mara says, "Use dynamic camera movements." Dev nods.'
        self.assertEqual(extract_locked_dialogue(source)[0]["text"], "Use dynamic camera movements.")
        self.assertFalse(extract_h3_source_intent(source)["perspective_contract"])

    def test_common_camera_preferences_do_not_become_endings(self):
        for direction in (
            "Use a handheld camera", "Use fast cuts", "Use close-ups and wide shots",
            "Keep the camera moving with the action",
        ):
            with self.subTest(direction=direction):
                source = f"Mara ducks. Dev falls. {direction}."
                self.assertEqual(extract_source_events(source)[-1]["text"], "Dev falls")
                self.assertIn(direction, extract_h3_source_intent(source)["global_instructions"])

    def test_static_abilities_and_general_directions_do_not_replace_the_ending(self):
        directions = (
            "Mara can fly and has heat vision.",
            "Her flight is innate levitation and her heat vision comes from her eyes.",
            "Show connected actions and a readable exchange across the 28 seconds.",
            "Keep the action and geography readable across 28 seconds.",
            "No dialogue, slow motion, or new powers.",
            "No dialogue, eye lasers, or extra powers.",
        )
        source = "Mara fights Dev. Dev crashes through the wall. " + " ".join(directions)
        events = extract_source_events(source)
        self.assertEqual([e["text"] for e in events], ["Mara fights Dev", "Dev crashes through the wall"])
        context = extract_h3_source_intent(source)["global_instructions"]
        for direction in directions:
            self.assertIn(direction, context)

    def test_ability_use_and_change_remain_timed_events(self):
        for action in (
            "Mara flies across the gap",
            "Mara can fly across the gap",
            "Mara can finally fly",
            "Now Mara can fly",
            "Mara learns to fly",
            "Mara loses her heat vision",
            "Mara can fly and punches Dev",
            "Mara can barely reach the lever",
            "Her heat vision comes from her eyes and melts the lock",
        ):
            with self.subTest(action=action):
                source = action + ". Dev ducks."
                self.assertFalse(extract_h3_source_intent(source)["global_instructions"])
                self.assertIn(action.split(" and ")[0], " ".join(e["text"] for e in extract_source_events(source)))

    def test_coordinated_traits_abilities_and_location_are_shared_context(self):
        facts = (
            "They are both super human, fast, strong, and they can fly and have heat vision.",
            "They are in a Tibetan Mountain temple.",
        )
        source = "Mara fights Dev. " + " ".join(facts) + " Dev crashes through the wall."
        self.assertEqual([e["text"] for e in extract_source_events(source)],
                         ["Mara fights Dev", "Dev crashes through the wall"])
        context = extract_h3_source_intent(source)["global_instructions"]
        for fact in facts:
            self.assertIn(fact, context)

    def test_common_static_fact_forms_do_not_become_later_plot_events(self):
        for fact in (
            "Both fighters are superhuman, fast and strong, and can fly.",
            "Mara and Dev can fly and have heat vision.",
            "they can fly and have heat vision.",
            "They are strong and they can levitate.",
            "The fight takes place in a mountain temple.",
            "The scene is set inside an old warehouse.",
        ):
            with self.subTest(fact=fact):
                source = "Mara ducks. Dev falls. " + fact
                self.assertEqual([e["text"] for e in extract_source_events(source)],
                                 ["Mara ducks", "Dev falls"])
                self.assertIn(fact, extract_h3_source_intent(source)["global_instructions"])

    def test_qualified_traits_locations_and_mixed_actions_remain_events(self):
        for action in (
            "Then they are both superhuman",
            "They are strong and they smash the wall",
            "They are in the temple after crossing the bridge",
            "They are in the temple and run into the courtyard",
            "They enter the Tibetan Mountain temple",
            "They can fly and have heat vision after taking the serum",
        ):
            with self.subTest(action=action):
                source = action + ". Dev ducks."
                self.assertFalse(extract_h3_source_intent(source)["global_instructions"])
                self.assertGreaterEqual(len(extract_source_events(source)), 2)

    def test_spoken_and_timed_scene_facts_keep_their_source_ownership(self):
        source = 'Mara says, "They are both strong, and they can fly. They are in a temple." Dev nods.'
        self.assertEqual(extract_locked_dialogue(source)[0]["text"],
                         "They are both strong, and they can fly. They are in a temple.")
        self.assertFalse(extract_h3_source_intent(source)["global_instructions"])
        timed = "[0s-4s] Mara runs. [4s-8s] They are in a temple."
        events = extract_source_events(timed)
        self.assertIn("They are in a temple", events[-1]["text"])

    def test_specific_action_directions_are_preserved(self):
        for action in (
            "Show the beam hitting the floor",
            "Keep Mara beside the pillar",
            "Mara shows Dev the cut hinge",
            "No one catches the falling glass",
        ):
            with self.subTest(action=action):
                self.assertIn(action, " ".join(e["text"] for e in extract_source_events(action + ".")))
                self.assertFalse(extract_h3_source_intent(action + ".")["global_instructions"])
        sequence = "Mina drops a glass. No one catches the glass. It shatters on the floor."
        self.assertEqual(
            [event["text"] for event in extract_source_events(sequence)],
            ["Mina drops a glass", "No one catches the glass", "It shatters on the floor"],
        )

        punctuated = (
            "Mara waits. Without anyone, Mara takes the key. "
            "No one, including Dev, catches the glass. It shatters."
        )
        self.assertEqual(
            [event["text"] for event in extract_source_events(punctuated)],
            ["Mara waits", "Without anyone, Mara takes the key",
             "No one, including Dev, catches the glass", "It shatters"],
        )
        self.assertFalse(extract_h3_source_intent(punctuated)["global_instructions"])

        constraints = "No dialogue, flashback, or visitor. Without narration or subtitles."
        self.assertEqual(
            [event["text"] for event in extract_source_events(constraints)],
            ["Establish and carry out the requested scene"],
        )
        global_rules = extract_h3_source_intent(constraints)["global_instructions"]
        self.assertIn("No dialogue, flashback, or visitor", global_rules)
        self.assertIn("Without narration or subtitles", global_rules)

    def test_spoken_capability_and_restrictions_are_not_global_rules(self):
        source = ('Mara says, "Listen. Dev can fly. No dialogue or new powers." '
                  'Dev points to the sky.')
        self.assertEqual(extract_locked_dialogue(source)[0]["text"],
                         "Listen. Dev can fly. No dialogue or new powers.")
        self.assertFalse(extract_h3_source_intent(source)["global_instructions"])
        self.assertIn("points to the sky", extract_source_events(source)[-1]["text"])

    def test_requested_powered_equipment_and_operation_survive(self):
        source = ("Mara wears jet boots and a wrist cutting tool. "
                  "She ignites the boots. She aims her wrist tool at the hinge. "
                  "She shuts it off and pulls the door open.")
        events = " ".join(e["text"] for e in extract_source_events(source))
        for fact in ("jet boots", "wrist cutting tool", "ignites the boots",
                     "aims her wrist tool", "shuts it off", "pulls the door open"):
            self.assertIn(fact, events)
        self.assertFalse(extract_h3_source_intent(source)["global_instructions"])


if __name__ == "__main__":
    unittest.main()
