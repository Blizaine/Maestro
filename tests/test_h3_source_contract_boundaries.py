"""Source context and physical-action boundaries for H3 planning contracts."""

from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.h3_story_ledger import (  # noqa: E402
    extract_h3_source_intent,
    extract_source_events,
    segment_violations,
)


def _segment(action, *, camera="", framing="Medium shot", duration=8):
    return {
        "segment": 1,
        "semantic_actions": True,
        "opening_state": "The greenhouse door is locked; the red spool is on the shelf.",
        "closing_state": "Mara is back in the hall with the red spool; the greenhouse door is locked.",
        "shots": [{
            "beat_ids": ["B1"],
            "dialogue_ids": [],
            "start_seconds": 0.0,
            "end_seconds": float(duration),
            "action": action,
            "camera": camera,
            "framing": framing,
            "transition": "continue",
            "sound_effects": "",
        }],
    }


def _assigned_beat(source_event_ids, description, *, segment=1, duration=8, state_after=""):
    return {
        "beat_id": "B1",
        "segment": segment,
        "description": description,
        "source_event_ids": list(source_event_ids),
        "dialogue_ids": [],
        "state_after": state_after,
        "authored_duration": duration,
    }


def _violations(prompt, segment, beat, *, segment_number=1, duration=8):
    return segment_violations(
        prompt,
        segment,
        segment_number=segment_number,
        duration=duration,
        assigned_beats=[beat],
        dialogue_catalog=[],
    )


class H3SourceContractBoundaryTests(unittest.TestCase):
    def test_scene_profiles_and_prohibitions_stay_context_while_actions_remain_ordered(self):
        source = (
            "A quiet greenhouse has one blue door and one wall shelf. One red spool rests on the shelf. "
            "Character A (Mara, adult gardener in an olive apron): has the only brass key. "
            "Character B (Eli, adult courier in a navy jacket): carries one sealed envelope. "
            "No cuts. No dialogue. No subtitles or extra people. "
            "Never show a close-up of the key before the door opens. Never let Eli handle the key. "
            "Mara unlocks the greenhouse door with the brass key, then Mara opens the door, "
            "then Mara enters the greenhouse, then Mara retrieves one red spool from the shelf, "
            "then Mara returns to the hall, and Mara closes and locks the same door."
        )

        events = extract_source_events(source)
        event_text = "\n".join(event["text"] for event in events).casefold()
        action_text = "\n".join(
            event["text"] for event in events
            if re.match(r"^Mara\b", event["text"], re.IGNORECASE)
        ).casefold()
        intent = extract_h3_source_intent(source)
        profile_text = str(intent["cast_profiles"]).casefold()
        negative_text = str(intent["negative_constraints"]).casefold()
        persistent_text = "\n".join((str(intent["global_instructions"]), profile_text, negative_text)).casefold()

        # Environment, cast biographies, and prohibitions are planning context,
        # not timed choreography.
        for context_fragment in (
            "quiet greenhouse", "olive apron", "navy jacket", "no cuts",
            "no dialogue", "no subtitles", "no extra people", "never show a close-up",
            "never let eli handle the key",
        ):
            with self.subTest(context_fragment=context_fragment):
                self.assertNotIn(context_fragment, event_text)
        self.assertIn("olive apron", profile_text)
        self.assertIn("navy jacket", profile_text)
        for constraint in ("no cuts", "no dialogue", "no extra people", "never show a close-up",
                           "never let eli handle the key"):
            with self.subTest(constraint=constraint):
                expected = "no subtitles or extra people" if constraint == "no extra people" else constraint
                self.assertIn(expected, persistent_text)

        # All user-authored physical stages survive in their original order.
        required = ("unlocks", "opens", "enters", "retrieves", "returns", "closes", "locks")
        positions = []
        for verb in required:
            match = re.search(rf"\b{re.escape(verb)}\b", action_text)
            self.assertIsNotNone(match, f"missing source action {verb!r}: {action_text}")
            positions.append(match.start())
        self.assertEqual(positions, sorted(positions))

    def test_key_insertion_and_final_state_do_not_cover_a_compound_action_chain(self):
        source = (
            "Mara unlocks the greenhouse door with the brass key, then Mara opens the door, "
            "then Mara enters the greenhouse, then Mara retrieves one red spool from the shelf, "
            "then Mara returns to the hall, and Mara closes and locks the same door."
        )
        events = extract_source_events(source)
        self.assertGreaterEqual(len(events), 5, events)
        beat = _assigned_beat(
            [event["event_id"] for event in events],
            source,
            state_after="Mara is back in the hall with the spool; the greenhouse door is closed and locked.",
        )
        insertion_only = _segment(
            "Mara inserts the brass key into the greenhouse door lock.",
            camera="A close-up stays on the key entering the lock.",
            framing="Close-up",
        )
        errors = _violations(source, insertion_only, beat)
        full_chain_errors = _violations(source, _segment(source), beat)
        self.assertTrue(
            any("omits required source step" in error for error in errors),
            "a final-state claim and key insertion must not stand in for the remaining physical sequence",
        )
        self.assertTrue(any("retrieves one red spool" in error for error in errors), errors)

        # A complete visual action is a positive control for the same contract.
        with self.subTest(control="complete compound action"):
            self.assertEqual(full_chain_errors, [])

    def test_hand_positions_and_close_up_are_not_handoff_or_door_close_evidence(self):
        source = (
            "[0s-4s] Mara stands outside the greenhouse with one red spool in her right hand; "
            "Eli's left hand is empty and the blue door is closed.\n"
            "[4s-8s] Mara hands the same spool to Eli.\n"
            "[8s-12s] Eli closes the same blue greenhouse door."
        )
        events = extract_source_events(source)
        self.assertEqual(len(events), 3, events)

        handoff_event = events[1]
        handoff_beat = _assigned_beat(
            [handoff_event["event_id"]], handoff_event["text"],
            segment=2, duration=4,
            state_after="The red spool is now with Eli.",
        )
        hand_position_only = _segment(
            "Mara keeps the same red spool in her right hand while Eli's left hand stays empty.",
            camera="A close-up shows the spool and both hands.",
            framing="Close-up",
            duration=4,
        )
        hand_position_only["segment"] = 2
        hand_errors = _violations(
            source, hand_position_only, handoff_beat, segment_number=2, duration=4,
        )
        handoff_control = _segment(
            handoff_event["text"],
            camera="A medium two-shot keeps both volunteers and the spool visible.",
            duration=4,
        )
        handoff_control["segment"] = 2
        handoff_control_errors = _violations(
            source, handoff_control, handoff_beat, segment_number=2, duration=4,
        )
        with self.subTest(case="nominal hand positions"):
            self.assertTrue(
                any("omits required source step" in error for error in hand_errors),
                "right/left hand positions do not depict the user-requested spool handoff",
            )
        with self.subTest(case="actual handoff control"):
            self.assertEqual(handoff_control_errors, [])

        close_event = events[2]
        close_beat = _assigned_beat(
            [close_event["event_id"]], close_event["text"],
            segment=3, duration=4,
            state_after="The blue greenhouse door is closed.",
        )
        close_up_only = _segment(
            "The camera holds on the closed blue greenhouse door.",
            camera="A close-up frames the same closed door.",
            framing="Close-up",
            duration=4,
        )
        close_up_only["segment"] = 3
        close_errors = _violations(
            source, close_up_only, close_beat, segment_number=3, duration=4,
        )
        close_control = _segment(
            close_event["text"],
            camera="Hold on the door as Eli closes it.",
            duration=4,
        )
        close_control["segment"] = 3
        close_control_errors = _violations(
            source, close_control, close_beat, segment_number=3, duration=4,
        )
        with self.subTest(case="close-up / closed state"):
            self.assertTrue(
                any("omits required source step" in error for error in close_errors),
                "a camera close-up or closed-state description does not depict Eli closing the door",
            )
        with self.subTest(case="actual door close control"):
            self.assertEqual(close_control_errors, [])


if __name__ == "__main__":
    unittest.main()
