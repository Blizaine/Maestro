"""Regressions for meaning lost between user/LLM text and native H3 prompts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import (
    _canonicalize_segment_contract,
    _infer_h3_opening_state_contract,
    _materialize_segment,
    _strip_planner_speech_cues,
    extract_h3_source_intent,
    extract_source_events,
)


ROOM = (
    "In a modest architecture office, Mara stands beside a table showing blueprints to her colleague Dev "
    "while Lin listens from the open doorway. Mara explains that the north wall cannot move; "
    "Dev studies the page before answering that the stair can shift instead. Lin enters, closes the door "
    "behind her, crosses to the table, and asks whether the change preserves the skylight. "
    "Mara turns from Dev to Lin and confirms that it does. Continue naturally into a second beat in the "
    "adjoining materials room: Mara opens the connecting door, all three walk through, Dev enters last "
    "and closes it, and Lin compares a wood sample with the blueprint while the others listen. "
    "Keep entrances, exits, listeners, room geography, and each door's open or closed state coherent "
    "across the transition."
)


class SemanticCleanupTests(unittest.TestCase):
    def test_temporal_split_keeps_the_subject_of_an_invitation(self):
        for source in (
            "Michael then invites her into his office.",
            "Michael then offers her a chair.",
            "Michael then apologizes to her.",
            "Michael waves, then invites her into his office.",
        ):
            with self.subTest(source=source):
                events = extract_source_events(source)
                self.assertTrue(events)
                self.assertTrue(all("Michael" in e["text"] for e in events), events)

    def test_temporal_split_does_not_assign_a_new_actors_action_to_previous_actor(self):
        events = extract_source_events("Michael waves. Pam then offers him a chair.")
        self.assertIn("Michael", events[0]["text"])
        self.assertIn("Pam", events[-1]["text"])
        self.assertNotIn("Michael", events[-1]["text"])

    def test_instruction_words_do_not_become_people_or_timed_events(self):
        intent = extract_h3_source_intent(ROOM)
        self.assertEqual(intent["cast_names"], ["Mara", "Dev", "Lin"])
        events = extract_source_events(ROOM)
        self.assertFalse(any(e["text"].startswith("Keep entrances") for e in events))
        self.assertFalse(any(e["text"].startswith("each door's") for e in events))
        self.assertTrue(any("Mara opens the connecting door" in e["text"] for e in events))
        self.assertIn("Lin compares", events[-1]["text"])
        self.assertIn("coherent", intent["global_instructions"])

    def test_already_visible_doorway_listener_is_not_forced_off_screen(self):
        for source in (
            ROOM,
            "Lin is already waiting in the open doorway. Mara nods. Lin enters the office.",
        ):
            self.assertEqual(_infer_h3_opening_state_contract(source, ["Mara", "Dev", "Lin"]), "")

    def test_actual_opening_entrance_still_starts_before_arrival(self):
        self.assertIn(
            "Lin has not yet entered",
            _infer_h3_opening_state_contract("Lin enters the office and greets Mara.", ["Lin", "Mara"]),
        )

    def test_physical_keep_directive_is_retained(self):
        source = "Mara stands beside Dev. Keep Mara beside the door while Dev crosses the room."
        self.assertEqual(extract_h3_source_intent(source)["cast_names"], ["Mara", "Dev"])
        self.assertTrue(any("Mara beside the door" in e["text"] for e in extract_source_events(source)))

    def test_visual_acting_survives_near_speech(self):
        samples = [
            ("Pam speaks in a quiet, unsure voice, maintaining eye contact with the approaching figure.",
             "maintaining eye contact"),
            ("Pam shifts her weight on her chair, tucking a loose strand of hair behind her ear, "
             "a subtle preparatory gesture before speaking.", "tucking a loose strand"),
            ("Pam looks up from the desk, meeting Power Girl's gaze, and speaks the line with a quiet, "
             "hesitant quality, tilting her head slightly as she finishes.", "tilting her head"),
            ("Dwight beams brightly, his eyes shining with enthusiasm, and leans slightly toward the desk "
             "as he prepares to speak, ready to deliver information.", "leans slightly toward the desk"),
            ("Pam reaches for the phone while she says, \"Hello. Welcome back.\"", "Pam reaches for the phone"),
        ]
        for source, physical in samples:
            with self.subTest(source=source):
                clean = _strip_planner_speech_cues(source)
                self.assertIn(physical, clean)
                self.assertNotIn("Welcome back", clean)

    def test_tagged_transcript_cannot_leak_into_visual_action(self):
        clean = _strip_planner_speech_cues(
            "Pam holds the phone. <d>[English] Hello. Welcome back.</d> Pam turns toward Dev."
        )
        self.assertIn("holds the phone", clean)
        self.assertIn("turns toward Dev", clean)
        self.assertNotIn("Hello", clean)
        self.assertNotIn("Welcome back", clean)

    def test_pure_speech_cue_is_not_a_silent_action(self):
        self.assertEqual(_strip_planner_speech_cues("Yoda speaks."), "")

    def test_object_state_is_preserved_and_spoken_states_still_uses_the_catalog(self):
        for text in (
            "The door's closed state remains unchanged.",
            "The two door states stay consistent across the transition.",
        ):
            self.assertEqual(_strip_planner_speech_cues(text), text)
            self.assertIn(text.rstrip("."), [e["text"] for e in extract_source_events(text)])
        clean = _strip_planner_speech_cues(
            'Mara states, "The door is locked." while pointing at the handle.',
            assigned_speakers=("Mara",),
        )
        self.assertIn("Mara performs the assigned line", clean)
        self.assertIn("pointing at the handle", clean)
        self.assertNotIn("The door is locked", clean)

    def test_mixed_action_keeps_its_actor_even_without_a_valid_dialogue_binding(self):
        for source in (
            'Mara says, "Wait. Come back." then turns to Dev.',
            'Mara explains the proposal while pointing to the door.',
        ):
            clean = _strip_planner_speech_cues(source)
            self.assertIn("Mara", clean)
            self.assertTrue("turns to Dev" in clean or "pointing to the door" in clean)
            self.assertNotIn("Come back", clean)

    def test_owned_performance_retains_acting_without_copying_the_transcript(self):
        beat = {"beat_id": "B1", "segment": 1, "source_event_ids": ["E1"],
                "dialogue_ids": ["D1"], "description": "Pam answers calmly", "state_after": "Pam is at the desk"}
        catalog = [{"dialogue_id": "D1", "speaker": "Pam", "text": "How can I help?", "language": "English"}]
        result = _canonicalize_segment_contract(
            {"shots": [{"event_indices": [1], "dialogue_ids": ["D1"], "start_seconds": 0, "end_seconds": 5,
                        "action": "Pam speaks in a quiet voice, maintaining eye contact with Dev.",
                        "framing": "Pam at her desk", "camera": "Medium shot of Pam"}],
             "closing_state": "Pam remains at the desk"},
            segment_number=1, duration=5, assigned_beats=[beat], dialogue_catalog=catalog,
            opening_state="Pam sits at the desk", source_intent={},
        )
        self.assertIn("Pam", result["shots"][0]["action"])
        self.assertIn("maintaining eye contact", result["shots"][0]["action"])
        final = _materialize_segment(result, beats=[beat], dialogue_catalog=catalog, source_events=[])
        self.assertEqual(final["shots"][0]["dialogue"][0]["speaker"], "Pam")
        self.assertEqual(final["shots"][0]["dialogue"][0]["text"], "How can I help?")

    def test_after_speaking_does_not_become_an_actor_or_a_dangling_cue(self):
        clean = _strip_planner_speech_cues(
            "Pam speaks with a slightly anxious, unsure tone. After speaking, Pam remains leaning "
            "forward with a slightly anxious expression, having just finished speaking.",
            assigned_speakers=("Pam",),
        )
        self.assertIn("Pam remains leaning forward", clean)
        self.assertIn("with mouth now closed", clean)
        self.assertNotIn("After is", clean)
        self.assertNotIn("; After;", clean)
        clean = _strip_planner_speech_cues("Michael interjects forcefully as Dwight leans into the frame.")
        self.assertIn("Dwight leans into the frame", clean)
        self.assertNotIn("As Dwight is", clean)

    def test_mixed_overlap_cues_cannot_leak_but_physical_acting_remains(self):
        clean = _strip_planner_speech_cues(
            "Michael points at a stack of paper, his face close to Power Girl, and speaks with an "
            "eager tone, talking over Dwight. Power Girl's expression shifts to mild confusion "
            "as she listens to the overlapping speech.",
            assigned_speakers=("Michael",),
        )
        self.assertIn("Michael points at a stack of paper", clean)
        self.assertIn("expression shifts to mild confusion", clean)
        self.assertNotIn("talking over", clean)
        self.assertNotIn("overlapping speech", clean)
        self.assertEqual(_strip_planner_speech_cues("Sound of overlapping vocalizations.", sound_field=True), "")


if __name__ == "__main__":
    unittest.main()
