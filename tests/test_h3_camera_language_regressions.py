"""Focused H3 camera-language regressions from candidate 3 diagnostics."""

from __future__ import annotations

import re
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_story_ledger import (  # noqa: E402
    _h3_preview_action_frames,
    _materialize_segment,
    extract_h3_source_intent,
    extract_locked_dialogue,
    extract_source_events,
    segment_violations,
)


def _segment_violations_for_event(
    source: str,
    event_index: int,
    action: str,
    *,
    state_after: str = "The visible action reaches its stated end state.",
) -> list[str]:
    """Validate a single source event in its own compact timed segment."""

    events = extract_source_events(source)
    if not 1 <= event_index <= len(events):
        raise AssertionError(f"Expected source event E{event_index}; got {events!r}")
    event = events[event_index - 1]
    segment_number = event_index
    beat = {
        "beat_id": "B1",
        "segment": segment_number,
        "description": event["text"],
        "source_event_ids": [event["event_id"]],
        "dialogue_ids": [],
        "state_after": state_after,
    }
    segment = {
        "segment": segment_number,
        "semantic_actions": True,
        "opening_state": "Continue the same established scene.",
        "closing_state": state_after,
        "shots": [{
            "shot": 1,
            "beat_ids": ["B1"],
            "dialogue_ids": [],
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "transition": "continuous composition",
            "action": action,
            "camera": "wide view",
            "framing": "wide view",
            "sound_effects": "",
        }],
    }
    return segment_violations(
        source,
        segment,
        segment_number=segment_number,
        duration=4.0,
        assigned_beats=[beat],
        dialogue_catalog=[],
    )


def _matching(errors: list[str], phrase: str) -> list[str]:
    return [error for error in errors if phrase.casefold() in error.casefold()]


class H3CameraLanguageRegressionTests(unittest.TestCase):
    def test_preparing_to_open_is_not_an_early_open_action(self):
        source = (
            "[0s-4s] Nora waits beside the closed blue workshop door. "
            "[4s-8s] Nora opens the blue workshop door."
        )
        errors = _segment_violations_for_event(
            source,
            1,
            "Nora waits beside the closed blue door and reaches toward its latch, "
            "preparing to open it; the door remains closed.",
        )
        self.assertFalse(_matching(errors, "previews later source event"), errors)

    def test_actively_opening_the_door_early_remains_a_preview_violation(self):
        source = (
            "[0s-4s] Nora waits beside the closed blue workshop door. "
            "[4s-8s] Nora opens the blue workshop door."
        )
        errors = _segment_violations_for_event(
            source,
            1,
            "Nora waits beside the closed blue door, then opens the blue workshop door.",
        )
        self.assertTrue(_matching(errors, "previews later source event"), errors)

    def test_reach_lock_and_hand_nouns_are_not_executed_actions(self):
        cast_pattern = re.compile(r"\bNora\b", re.IGNORECASE)
        frames = _h3_preview_action_frames(
            "The brass lock stays on the case; the handle remains within Nora's reach, "
            "and her right hand rests beside it.",
            cast_pattern,
        )
        verbs = {frame[0] for frame in frames}
        self.assertTrue(verbs.isdisjoint({"lock", "reach", "hand"}), frames)

    def test_active_lock_reach_and_handoff_verbs_remain_action_frames(self):
        frames = _h3_preview_action_frames(
            "Nora locks the case, reaches for the handle, and hands Lee the map.",
            re.compile(r"\b(?:Nora|Lee)\b", re.IGNORECASE),
        )
        verbs = {frame[0] for frame in frames}
        self.assertTrue({"lock", "reach", "hand"}.issubset(verbs), frames)

    def test_immediate_post_handoff_spool_hold_is_not_a_later_action_preview(self):
        source = (
            "[0s-4s] Sam and Priya arrive at the workshop bench. "
            "[4s-8s] Sam hands the red spool to Priya. "
            "[8s-12s] Priya holds the spool's loose end beside the banner."
        )
        errors = _segment_violations_for_event(
            source,
            2,
            "At the bench, Sam hands the red spool to Priya, who then holds its loose end.",
            state_after="Priya now has the spool at the bench.",
        )
        self.assertFalse(_matching(errors, "previews later source event"), errors)

    def test_an_early_real_spool_handoff_still_triggers_a_preview(self):
        source = (
            "[0s-4s] Sam waits beside the red spool in the gallery. "
            "[4s-8s] At the workshop bench, Sam hands the same red spool to Priya."
        )
        errors = _segment_violations_for_event(
            source,
            1,
            "Sam waits in the gallery but then hands the red spool to Priya "
            "before reaching the workshop bench.",
        )
        self.assertTrue(_matching(errors, "previews later source event"), errors)

    def test_holding_the_banner_does_not_match_a_later_spool_hold(self):
        source = (
            "[0s-4s] Priya ties the banner to the workshop wall. "
            "[4s-8s] Priya holds the loose end of the red spool."
        )
        errors = _segment_violations_for_event(
            source,
            1,
            "Priya ties the banner and holds it against the wall.",
        )
        self.assertFalse(_matching(errors, "previews later source event"), errors)

    def test_holding_the_same_spool_early_remains_a_preview_control(self):
        source = (
            "[0s-4s] Priya ties the banner to the workshop wall. "
            "[4s-8s] Priya holds the loose end of the red spool."
        )
        errors = _segment_violations_for_event(
            source,
            1,
            "Priya ties the banner and holds the loose end of the red spool.",
        )
        self.assertTrue(_matching(errors, "previews later source event"), errors)

    def test_terminal_closed_door_state_does_not_cover_active_closing(self):
        source = (
            "[0s-4s] Nora walks into the hall. "
            "[4s-8s] Nora closes the workshop door."
        )
        state_only = _segment_violations_for_event(
            source,
            2,
            "The workshop door remains closed.",
            state_after="The workshop door is closed.",
        )
        self.assertTrue(_matching(state_only, "omits required source step"), state_only)
        self.assertTrue(_matching(state_only, "closes the workshop door"), state_only)

    def test_visible_active_door_close_covers_the_required_action(self):
        source = (
            "[0s-4s] Nora walks into the hall. "
            "[4s-8s] Nora closes the workshop door."
        )
        errors = _segment_violations_for_event(
            source,
            2,
            "Nora pulls the workshop door shut and checks the latch.",
            state_after="The workshop door is closed.",
        )
        self.assertFalse(_matching(errors, "omits required source step"), errors)

    def test_parenthetical_semicolon_cast_notes_do_not_become_speech_or_events(self):
        source = (
            "Character notes (casting only; not spoken): Ada: adult bookbinder; "
            "wears a green apron; keeps the brass key. Ren: adult courier; "
            "carries one sealed envelope. [0s-4s] Ada unlocks the display case "
            "and retrieves one map. [4s-8s] Ren carries the envelope to the exit. "
            "No dialogue, captions, or voiceover."
        )
        events = extract_source_events(source)
        event_text = " ".join(event["text"] for event in events).casefold()
        self.assertEqual(extract_locked_dialogue(source), [])
        self.assertIn("ada unlocks the display case", event_text)
        self.assertIn("retrieves one map", event_text)
        self.assertIn("ren carries the envelope", event_text)
        for profile_only in ("character notes", "casting only", "adult bookbinder", "not spoken"):
            with self.subTest(profile_only=profile_only):
                self.assertNotIn(profile_only, event_text)

    def test_materialized_casting_note_does_not_invent_a_dialogue_performance(self):
        source = (
            "Character notes (casting only; not spoken): Ada: adult bookbinder; "
            "wears a green apron; keeps the brass key. [0s-4s] Ada unlocks "
            "the display case and retrieves one map. No dialogue or voiceover."
        )
        event = extract_source_events(source)[0]
        beat = {
            "beat_id": "B1",
            "description": event["text"],
            "source_event_ids": [event["event_id"]],
            "dialogue_ids": [],
            "state_after": "The case is open and the map is in Ada's hand.",
        }
        segment = {
            "segment": 1,
            "semantic_actions": True,
            "opening_state": "Ada stands at the display case.",
            "closing_state": beat["state_after"],
            "shots": [{
                "shot": 1,
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "transition": "continuous composition",
                "framing": "medium view",
                "camera": "steady camera follows the action",
                "action": event["text"],
                "dialogue": [],
                "beat_ids": ["B1"],
            }],
        }
        materialized = _materialize_segment(
            segment,
            beats=[beat],
            dialogue_catalog=[],
            source_events=[event],
        )
        self.assertEqual(materialized["shots"][0]["dialogue"], [])
        self.assertNotIn(
            "visibly delivers the assigned dialogue line",
            materialized["shots"][0]["action"].casefold(),
        )

    def test_picture_and_audio_setup_directives_do_not_replace_physical_actions(self):
        source = (
            "Preserve Nora's appearance from <Picture 1>; <Audio 1> is only a "
            "voice-timbre reference. [0s-4s] Nora waits beside the closed blue door. "
            "[4s-8s] Nora opens the blue door."
        )
        events = extract_source_events(source)
        event_text = " ".join(event["text"] for event in events).casefold()
        cast = {name.casefold() for name in extract_h3_source_intent(source)["cast_names"]}
        self.assertEqual(len(events), 2, events)
        self.assertIn("nora opens the blue door", event_text)
        self.assertNotIn("picture 1", event_text)
        self.assertNotIn("audio 1", event_text)
        self.assertNotIn("picture", cast)
        self.assertNotIn("audio", cast)

    def test_untimed_parenthetical_casting_notes_leave_only_the_pour_and_tie_actions(self):
        source = (
            "Character notes (casting only; not spoken): Hadi: adult greenhouse keeper; "
            "Ren: adult plant archivist. Hadi gives one brief pour to the basil plant "
            "while Ren ties a blank tag to that same pot. No dialogue, voiceover, or captions."
        )
        events = extract_source_events(source)
        event_text = " ".join(event["text"] for event in events).casefold()
        self.assertEqual(extract_locked_dialogue(source), [])
        self.assertIn("hadi gives one brief pour", event_text)
        self.assertIn("ren ties a blank tag", event_text)
        for context in (
            "character notes",
            "casting only",
            "not spoken",
            "adult greenhouse keeper",
            "adult plant archivist",
            "no dialogue",
            "voiceover",
            "captions",
        ):
            with self.subTest(context=context):
                self.assertNotIn(context, event_text)

        physical_events = [
            event for event in events
            if any(phrase in event["text"].casefold() for phrase in (
                "hadi gives one brief pour", "ren ties a blank tag",
            ))
        ]
        beat = {
            "beat_id": "B1",
            "description": " ".join(event["text"] for event in physical_events),
            "source_event_ids": [event["event_id"] for event in physical_events],
            "dialogue_ids": [],
            "state_after": "The basil is watered and its tag is tied to the pot.",
        }
        materialized = _materialize_segment(
            {
                "segment": 1,
                "semantic_actions": True,
                "opening_state": "Hadi and Ren stand by the greenhouse bench.",
                "closing_state": beat["state_after"],
                "shots": [{
                    "shot": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                    "transition": "continuous composition",
                    "framing": "medium-wide view",
                    "camera": "steady camera follows both actions",
                    "action": beat["description"],
                    "dialogue": [],
                    "beat_ids": ["B1"],
                }],
            },
            beats=[beat],
            dialogue_catalog=[],
            source_events=physical_events,
        )
        self.assertEqual(materialized["shots"][0]["dialogue"], [])
        self.assertNotIn(
            "visibly delivers the assigned dialogue line",
            materialized["shots"][0]["action"].casefold(),
        )

    def test_untimed_music_brief_keeps_audio_contract_and_visual_tail_out_of_setup_events(self):
        source = (
            "Create a short narrative music video in the warm public hall shown by <Picture 1>, "
            "using <Audio 1> as the exact performance-driving soundtrack. One adult percussionist "
            "plays at center while an adult neighbor carries a paper lantern from the rear aisle "
            "toward a table. During a later musical phrase, the performer steadies the lantern so "
            "the neighbor can hang it. Keep visible playing aligned to the full supplied track. "
            "The 8-second track ends naturally; after its final note, hold on the lantern glowing "
            "in silence for the remaining video tail. Do not loop, stretch, replace, or supplement "
            "the supplied audio; add no dialogue, replacement music, or new vocals."
        )
        events = extract_source_events(source)
        event_text = "\n".join(event["text"] for event in events).casefold()
        intent = extract_h3_source_intent(source)
        intent_text = "\n".join(
            str(value or "") for value in intent.values() if isinstance(value, (str, list, dict))
        ).casefold()
        all_extracted_text = event_text + "\n" + intent_text
        global_audio_contract = "\n".join(
            str(intent.get(key) or "")
            for key in ("global_instructions", "negative_constraints", "ambient_contract")
        ).casefold()

        self.assertNotIn("<picture 1>", event_text)
        self.assertNotIn("<audio 1>", event_text)
        self.assertNotIn("create a short narrative music video", event_text)
        for setup_or_contract in (
            "keep visible playing aligned",
            "do not loop, stretch, replace, or supplement",
            "add no dialogue",
            "new vocals",
        ):
            with self.subTest(setup_or_contract=setup_or_contract):
                self.assertNotIn(setup_or_contract, event_text)

        for physical_action in (
            "percussionist plays at center",
            "neighbor carries a paper lantern",
            "performer steadies the lantern",
            "hold on the lantern glowing in silence",
        ):
            with self.subTest(physical_action=physical_action):
                self.assertIn(physical_action, event_text)

        # Setup/audio restrictions stay global, while the track-end and silent
        # tail are retained as real constraints on this performance.
        self.assertIn("do not loop, stretch, replace, or supplement", global_audio_contract)
        self.assertIn("no dialogue", global_audio_contract)
        self.assertIn("new vocals", global_audio_contract)
        self.assertIn("the 8-second track ends naturally", all_extracted_text)
        self.assertIn("after its final note", all_extracted_text)
        self.assertIn("remaining video tail", all_extracted_text)

    def test_landing_on_feet_does_not_cover_the_safe_palm_stop(self):
        source = (
            "[0s-4s] Mara lands after the low sweep. "
            "[4s-8s] Mara stops an open palm safely short of Ivo's chest."
        )
        missing = _segment_violations_for_event(
            source,
            2,
            "Mara lands on both feet after the sweep and regains balance.",
        )
        self.assertTrue(_matching(missing, "omits required source step"), missing)
        self.assertTrue(_matching(missing, "open palm"), missing)

    def test_visible_safe_palm_stop_covers_its_source_action(self):
        source = (
            "[0s-4s] Mara lands after the low sweep. "
            "[4s-8s] Mara stops an open palm safely short of Ivo's chest."
        )
        errors = _segment_violations_for_event(
            source,
            2,
            "Mara stops her open palm a safe distance from Ivo's chest.",
        )
        self.assertFalse(_matching(errors, "omits required source step"), errors)

    def test_feet_landing_does_not_cover_the_partner_lowering_both_hands(self):
        source = (
            "[0s-4s] Mara steps over the low sweep. "
            "[4s-8s] Ivo lowers both hands to end the practice bout."
        )
        missing = _segment_violations_for_event(
            source,
            2,
            "Mara lands on both feet while Ivo watches from the mat.",
        )
        self.assertTrue(_matching(missing, "omits required source step"), missing)
        self.assertTrue(_matching(missing, "lowers both hands"), missing)

    def test_visible_partner_hand_lowering_covers_its_source_action(self):
        source = (
            "[0s-4s] Mara steps over the low sweep. "
            "[4s-8s] Ivo lowers both hands to end the practice bout."
        )
        errors = _segment_violations_for_event(
            source,
            2,
            "Ivo lowers both hands to his sides as Mara watches.",
        )
        self.assertFalse(_matching(errors, "omits required source step"), errors)


if __name__ == "__main__":
    unittest.main()
