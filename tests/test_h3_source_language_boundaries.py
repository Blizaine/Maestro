"""Regressions for source-language boundaries in H3 planning and validation."""

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
    _canonicalize_segment_contract,
    _h3_preview_action_frames,
    _h3_preview_action_matches,
    extract_h3_source_intent,
    extract_locked_dialogue,
    extract_source_events,
    _spectacle_violations,
    segment_violations,
)


_COLON_CASTING_SOURCE = (
    "NON-DIALOGUE ROLE DESCRIPTIONS (casting notes only; not utterances, captions, or words to speak): "
    "Hadi: adult greenhouse keeper who alone carries and operates the one blue watering can. "
    "Ren: adult plant archivist who alone owns the one blank paper tag and never handles the watering can. "
    "These colon-labeled descriptions are not spoken. In one continuous shot with no cuts, "
    "Hadi gives one brief pour to the single basil plant while Ren ties the blank tag to that same pot. "
    "No dialogue, voiceover, lyrics, captions, or speech-like mouth movement. Keep both adults and their "
    "assigned objects distinct; add no other people, pots, tags, or watering cans."
)

_CURLY_DIALOGUE_SOURCE = (
    "Two adults only: Celia, an adult paper conservator, solely holds one dry blue print by its lower corners; "
    "Omar, an adult framing assistant, solely holds one empty black frame by its sides. "
    "Use one uninterrupted shot with no cuts. Preserve these four exact lines with curly quotation marks, "
    "in this order, and add no other spoken words: Celia says, “The blue copy is dry.” "
    "Omar replies, “I’ll bring the frame closer.” Celia says, “Keep the top edge level.” "
    "Omar says, “Ready when you are.” Omar brings only the frame closer after his first line. "
    "Celia places the same print inside only after her second line. They set the single completed frame "
    "on the worktable after the final line. Do not paraphrase, duplicate, reorder, or reassign any line or object."
)

_SPARRING_SOURCE = (
    "Use the supplied opening frame as the exact first composition and as the appearance reference for the "
    "adult fighter in the cream training robe; begin with the controlled high kick already visible. "
    "At the exact first frame, the second adult fighter is just outside the frame at right; reveal them only "
    "as the same uninterrupted tracking move widens or pans right. They wear a plain charcoal training jacket "
    "and face the cream-robed fighter across the same covered stone courtyard. Continue from the opening pose: "
    "the cream-robed fighter completes the kick, lands, and steps back; the charcoal-jacketed partner checks "
    "the movement with a forearm guard and releases; both regain balanced footing. The charcoal-jacketed fighter "
    "then makes one low sweep, the cream-robed fighter steps over it, lands, and stops an open palm a safe "
    "distance from the partner’s chest. The partner lowers both hands to signal the practice bout is over. "
    "Keep the action in this order and both fighters in the same courtyard. One uninterrupted real-time tracking "
    "shot, no cuts, no weapons, no injury, no magic, no slow motion, and no dialogue or speech-like mouth movement."
)

_MUSIC_IMPERATIVE_SOURCE = (
    "Create a short narrative music video in the warm public hall shown by <Picture 1>, using <Audio 1> as the "
    "exact performance-driving soundtrack. One adult percussionist plays at the center while an adult neighbor "
    "carries a paper lantern from the rear aisle toward a table. During a later musical phrase the performer "
    "steadies the lantern so the neighbor can hang it above the table. Keep visible playing aligned to the full "
    "supplied track. The 37.33-second track ends naturally; after its final note, hold on the lantern glowing in "
    "the hall in silence for the remaining video tail. Do not loop, stretch, replace, or supplement the supplied "
    "audio; add no dialogue, replacement music, new vocals, or extra soundtrack. Keep the pictured hall and "
    "audience as scene context, but do not copy the pictured martial-arts pose or identity."
)


def _coverage_errors(source: str, event_index: int, visible_action: str) -> list[str]:
    events = extract_source_events(source)
    event = events[event_index - 1]
    beat = {
        "beat_id": "B1",
        "segment": 1,
        "description": event["text"],
        "source_event_ids": [event["event_id"]],
        "dialogue_ids": [],
        "state_after": "The visible source action is complete",
    }
    segment = {
        "segment": 1,
        "semantic_actions": True,
        "shots": [{
            "shot": 1,
            "beat_ids": ["B1"],
            "start_seconds": 0,
            "end_seconds": 4,
            "transition": "opening composition",
            "action": visible_action,
            "camera": "wide view",
            "framing": "wide view",
            "sound_effects": "",
        }],
        "closing_state": "The visible source action is complete",
    }
    return segment_violations(
        source,
        segment,
        segment_number=1,
        duration=4,
        assigned_beats=[beat],
        dialogue_catalog=[],
    )


def _preview_errors(source: str, visible_action: str) -> list[str]:
    events = extract_source_events(source)
    if len(events) < 2:
        raise AssertionError(f"Expected two chronological source events, got {events!r}")
    first = events[0]
    beat = {
        "beat_id": "B1",
        "segment": 1,
        "description": first["text"],
        "source_event_ids": [first["event_id"]],
        "dialogue_ids": [],
        "state_after": first["text"],
    }
    draft = {
        "segment": 1,
        "title": "Source boundary preview",
        "opening_state": first["text"],
        "coverage": "chronological action coverage",
        "pacing": "real time",
        "shots": [{
            "shot": 1,
            "event_indices": [1],
            "start_seconds": 0,
            "end_seconds": 4,
            "transition": "opening composition",
            "action": visible_action,
            "camera": "wide view",
            "framing": "wide view",
            "sound_effects": "",
        }],
        "closing_state": first["text"],
    }
    segment = _canonicalize_segment_contract(
        draft,
        segment_number=1,
        duration=4,
        assigned_beats=[beat],
        dialogue_catalog=[],
        opening_state=draft["opening_state"],
        source_intent={},
    )
    return segment_violations(
        source,
        segment,
        segment_number=1,
        duration=4,
        assigned_beats=[beat],
        dialogue_catalog=[],
    )


class H3SourceLanguageBoundaryTests(unittest.TestCase):
    def test_colon_casting_header_does_not_consume_paragraph_as_dialogue(self):
        self.assertEqual(extract_locked_dialogue(_COLON_CASTING_SOURCE), [])

    def test_colon_casting_header_preserves_the_real_actions_as_source_events(self):
        events = extract_source_events(_COLON_CASTING_SOURCE)
        event_text = " ".join(item["text"] for item in events).casefold()
        self.assertIn("hadi gives one brief pour", event_text)
        self.assertIn("ren ties the blank tag", event_text)
        self.assertNotIn("non-dialogue role descriptions speaks", event_text)

    def test_two_adults_only_header_does_not_own_four_curly_quote_lines(self):
        locked = extract_locked_dialogue(_CURLY_DIALOGUE_SOURCE)
        self.assertEqual(
            [(item["speaker"], item["text"]) for item in locked],
            [
                ("Celia", "The blue copy is dry."),
                ("Omar", "I’ll bring the frame closer."),
                ("Celia", "Keep the top edge level."),
                ("Omar", "Ready when you are."),
            ],
        )
        self.assertEqual([item["dialogue_id"] for item in locked], ["D1", "D2", "D3", "D4"])
        event_text = " ".join(item["text"] for item in extract_source_events(_CURLY_DIALOGUE_SOURCE))
        for physical_action in (
            "Omar brings only the frame closer",
            "Celia places the same print inside",
            "They set the single completed frame on the worktable",
        ):
            self.assertIn(physical_action, event_text)

    def test_three_or_numeric_adults_header_does_not_consume_quotes_or_actions(self):
        expected_lines = [
            ("Celia", "The blue copy is dry."),
            ("Omar", "I’ll bring the frame closer."),
            ("Celia", "Keep the top edge level."),
            ("Omar", "Ready when you are."),
        ]
        for header in ("Three adults only:", "2 adults only:"):
            with self.subTest(header=header):
                source = _CURLY_DIALOGUE_SOURCE.replace("Two adults only:", header, 1)
                self.assertEqual(
                    [(item["speaker"], item["text"]) for item in extract_locked_dialogue(source)],
                    expected_lines,
                )
                event_text = " ".join(item["text"] for item in extract_source_events(source))
                self.assertIn("Omar brings only the frame closer", event_text)
                self.assertIn("They set the single completed frame on the worktable", event_text)

    def test_character_note_relative_profile_does_not_swallow_main_clause_action(self):
        source = "Character notes: Nora: adult gardener enters the shed."
        self.assertEqual(extract_locked_dialogue(source), [])
        event_text = " ".join(item["text"] for item in extract_source_events(source))
        self.assertIn("enters the shed", event_text)
        intent = extract_h3_source_intent(source)
        context_text = " ".join((
            event_text,
            str(intent.get("cast_profiles") or ""),
            str(intent.get("global_instructions") or ""),
        )).casefold()
        self.assertIn("adult gardener", context_text)

    def test_real_unquoted_first_person_and_question_turns_survive_cast_context(self):
        source = (
            "Mara and Len are two adult archivists waiting by the archive door. "
            "Mara wears a gray coat and Len wears a blue coat. "
            "Mara keeps the only brass key while Len carries the folded plan.\n"
            "Mara: I have the only key.\n"
            "Len: Why is the archive door still locked?"
        )
        locked = extract_locked_dialogue(source)
        self.assertEqual(
            [(item["speaker"], item["text"]) for item in locked],
            [("Mara", "I have the only key."), ("Len", "Why is the archive door still locked?")],
        )

    def test_action_led_role_rows_remain_visual_events(self):
        source = (
            "Role A: walks around the table and lifts the red folder.\n"
            "Role B: catches the folder before it falls."
        )
        self.assertEqual(extract_locked_dialogue(source), [])
        event_text = " ".join(item["text"] for item in extract_source_events(source)).casefold()
        self.assertIn("walks around the table", event_text)
        self.assertIn("lifts the red folder", event_text)
        self.assertIn("catches the folder", event_text)

    def test_composite_camera_restrictions_are_global_not_timed_story_actions(self):
        source = (
            "[0s-4s] Hadi pours once into the single basil plant. "
            "[4s-8s] Ren ties one blank tag to the same pot. "
            "One uninterrupted real-time tracking shot with no cuts, no weapons, no injury, "
            "no magic, and no slow motion."
        )
        events = extract_source_events(source)
        event_text = " ".join(item["text"] for item in events).casefold()
        self.assertIn("hadi pours once", event_text)
        self.assertIn("ren ties one blank tag", event_text)
        self.assertNotIn("no cuts", event_text)
        self.assertNotIn("no slow motion", event_text)

        intent = extract_h3_source_intent(source)
        global_contract = "\n".join(
            str(intent.get(key) or "")
            for key in ("global_instructions", "negative_constraints", "pacing_contract", "perspective_contract")
        ).casefold()
        self.assertIn("no cuts", global_contract)
        self.assertIn("no slow motion", global_contract)
        self.assertIn("no weapons", global_contract)

    def test_negative_narrative_statement_is_not_discarded_as_a_global_prohibition(self):
        source = (
            "[0s-4s] Sam carries the red spool across the gallery and does not touch the latch "
            "until Priya opens the workshop door. [4s-8s] Priya opens the workshop door."
        )
        events = extract_source_events(source)
        self.assertIn("does not touch the latch until Priya opens", events[0]["text"])

    def test_sparring_actions_survive_beside_composite_no_slow_motion_restriction(self):
        events = extract_source_events(_SPARRING_SOURCE)
        event_text = " ".join(item["text"] for item in events).casefold()
        self.assertIn("completes the kick", event_text)
        self.assertIn("steps back", event_text)
        self.assertIn("makes one low sweep", event_text)
        self.assertIn("lowers both hands", event_text)
        self.assertNotIn("no slow motion", event_text)

    def test_continue_from_opening_pose_is_not_cast_or_a_separate_actor(self):
        source = (
            "Mara and Ivo are adult sparring partners in the same courtyard. "
            "Continue from the opening pose: Mara completes one controlled kick, lands, and steps back "
            "while Ivo raises a forearm guard. Do not loop or repeat the action."
        )
        intent = extract_h3_source_intent(source)
        self.assertIn("Mara", intent["cast_names"])
        self.assertIn("Ivo", intent["cast_names"])
        self.assertNotIn("Continue", intent["cast_names"])
        event_text = " ".join(item["text"] for item in extract_source_events(source)).casefold()
        self.assertNotIn("continue steps back", event_text)
        self.assertIn("steps back", event_text)

    def test_do_not_loop_audio_instruction_does_not_create_a_cast_member(self):
        intent = extract_h3_source_intent(_MUSIC_IMPERATIVE_SOURCE)
        self.assertNotIn("Do", intent["cast_names"])

    def test_source_authorized_laser_survives_an_unrelated_no_lightning_constraint(self):
        source = (
            "A superhero directs one red laser beam from the eyes toward an empty target. No lightning."
        )
        draft = {"action": "The superhero fires one red laser beam toward the target."}
        self.assertEqual(_spectacle_violations(source, draft), [])

    def test_storm_allows_weather_lightning_but_not_staff_or_hand_generated_bolts(self):
        source = (
            "A stormy sky with low clouds hangs over the ridge. Ivo waits below with a plain wooden staff."
        )
        natural_flash = {"action": "A lightning flash forks across the storm clouds above Ivo."}
        self.assertEqual(_spectacle_violations(source, natural_flash), [])

        for action in (
            "A lightning bolt flashes from Ivo’s staff.",
            "Ivo directs a lightning bolt from his hands at the target.",
        ):
            with self.subTest(action=action):
                errors = _spectacle_violations(source, {"action": action})
                self.assertTrue(any("unrequested power/effect" in error for error in errors), errors)

    def test_narrative_repair_clauses_are_not_classified_as_persistent_visual_style(self):
        source = (
            "A silent observatory story follows Elin, one adult clockmaker, through a stormy night. "
            "The beacon has gone dark because one brass gear slipped out of place; Elin finds it, "
            "carries it to the workshop, repairs the lamp, and returns the same gear and housing "
            "to the lantern room. Preserve this order and the same gear and housing. "
            "Cinematic imagery, cool blue moonlight, warm amber work lamps, restrained contrast."
        )
        style = extract_h3_source_intent(source)["style_contract"].casefold()
        self.assertNotIn("beacon has gone dark", style)
        self.assertNotIn("gear slipped", style)
        self.assertNotIn("repairs the lamp", style)

    def test_persistent_lighting_color_and_cinematography_remain_visual_style(self):
        source = (
            "Cinematic imagery, cool blue moonlight, warm amber practicals, restrained live-action contrast."
        )
        style = extract_h3_source_intent(source)["style_contract"].casefold()
        self.assertIn("cool blue moonlight", style)
        self.assertIn("warm amber practicals", style)
        self.assertIn("cinematic", style)

    def test_adverbial_closed_and_locked_state_has_no_physical_action_frames(self):
        frames = _h3_preview_action_frames(
            "The door is visibly closed and locked, the brass keyhole catching the light.",
            re.compile(r"\bNora\b", re.IGNORECASE),
        )
        self.assertEqual(frames, [])

    def test_coordinated_closed_and_sealed_states_have_no_action_frames(self):
        for description in (
            "The door is closed and locked.",
            "The box remains firmly closed and sealed.",
        ):
            with self.subTest(description=description):
                self.assertEqual(_h3_preview_action_frames(description, None), [])

    def test_lock_and_hand_nouns_do_not_become_physical_verb_frames(self):
        frames = _h3_preview_action_frames(
            "Nora glances at the brass lock on the door; her right hand rests on the latch.",
            re.compile(r"\bNora\b", re.IGNORECASE),
        )
        self.assertNotIn("lock", {frame[0] for frame in frames})
        self.assertNotIn("hand", {frame[0] for frame in frames})

    def test_finite_active_close_and_lock_remain_action_frames(self):
        frames = _h3_preview_action_frames(
            "Ada closes and locks the oak door.", re.compile(r"\bAda\b", re.IGNORECASE)
        )
        self.assertTrue({"close", "lock"}.issubset({frame[0] for frame in frames}), frames)

    def test_active_progressive_door_close_remains_an_action_frame(self):
        frames = _h3_preview_action_frames(
            "Nora is closing the oak door.", re.compile(r"\bNora\b", re.IGNORECASE)
        )
        self.assertTrue(any(frame[0] == "close" for frame in frames), frames)

    def test_progressive_hold_and_coordinated_finite_lock_both_remain_action_frames(self):
        frames = _h3_preview_action_frames(
            "Nora is holding the key and locks the door.", re.compile(r"\bNora\b", re.IGNORECASE)
        )
        verbs = {frame[0] for frame in frames}
        self.assertIn("hold", verbs, frames)
        self.assertIn("lock", verbs, frames)

    def test_passive_progressive_door_close_remains_an_action_frame(self):
        frames = _h3_preview_action_frames(
            "The oak door is being closed by Nora.", re.compile(r"\bNora\b", re.IGNORECASE)
        )
        close_frames = [frame for frame in frames if frame[0] == "close"]
        self.assertTrue(close_frames, frames)
        self.assertTrue(any("nora" in frame[2] for frame in close_frames), close_frames)

    def test_real_handoff_verb_is_not_suppressed_as_a_hand_noun(self):
        frames = _h3_preview_action_frames(
            "Sam hands Priya the red spool.", re.compile(r"\b(?:Sam|Priya)\b", re.IGNORECASE)
        )
        self.assertTrue(any(frame[0] == "hand" for frame in frames), frames)

    def test_other_owners_holding_same_spool_does_not_trigger_a_preview(self):
        source = (
            "[0s-4s] Sam holds the red spool at the gallery. "
            "[4s-8s] Priya holds the same red spool at the workshop."
        )
        errors = _preview_errors(source, "Sam holds the red spool at the gallery.")
        self.assertFalse(any("previews later" in error for error in errors), errors)

    def test_wrong_actor_cannot_perform_reserved_prop_change_early(self):
        errors = _preview_errors(
            "Tavi and Lio prepare a stand at the west column. "
            "Lio places the program flat on the stand.",
            "Tavi places the paper program onto the wooden stand.",
        )
        self.assertTrue(any("previews later" in error for error in errors), errors)

    def test_explicitly_assigned_other_owners_action_is_not_a_preview(self):
        errors = _preview_errors(
            "Tavi places the program on the stand. "
            "Later Lio places the program on the stand.",
            "Tavi places the program on the stand.",
        )
        self.assertFalse(any("previews later" in error for error in errors), errors)

    def test_other_actor_changing_a_different_prop_is_not_a_preview(self):
        errors = _preview_errors(
            "Tavi and Lio prepare a stand. Lio places the program on the stand.",
            "Tavi places a cup on the floor.",
        )
        self.assertFalse(any("previews later" in error for error in errors), errors)

    def test_preview_match_requires_the_same_owner_when_a_shared_prop_overlaps(self):
        cast_pattern = re.compile(r"\b(?:Sam|Priya)\b", re.IGNORECASE)
        sam_frame = _h3_preview_action_frames("Sam holds the red spool at the bench.", cast_pattern)[0]
        priya_frame = _h3_preview_action_frames("Priya holds the same red spool at the bench.", cast_pattern)[0]
        self.assertFalse(_h3_preview_action_matches(priya_frame, sam_frame))
        self.assertTrue(_h3_preview_action_matches(sam_frame, sam_frame))

    def test_same_owner_and_same_spool_remain_a_true_premature_action_control(self):
        source = (
            "[0s-4s] Sam waits beside the red spool at the gallery. "
            "[4s-8s] Sam holds the same red spool at the workshop."
        )
        errors = _preview_errors(source, "Sam holds the red spool at the gallery.")
        self.assertTrue(any("previews later" in error for error in errors), errors)

    def test_named_cast_apposition_is_satisfied_by_both_people_visibly_waiting(self):
        source = (
            "[0s-4s] Two adult library volunteers, Ada and Len, wait in the corridor "
            "outside a locked archive. [4s-8s] Ada unlocks the archive door."
        )
        errors = _coverage_errors(
            source,
            1,
            "Ada and Len visibly wait together in the corridor outside the locked archive.",
        )
        self.assertFalse(any("omits required source step" in error for error in errors), errors)

    def test_appositive_cast_setup_does_not_weaken_the_required_binder_retrieval(self):
        source = (
            "[0s-4s] Two adult library volunteers, Ada and Len, wait outside the archive. "
            "[4s-8s] Ada unlocks the archive door, opens it, enters the archive, retrieves one blue binder, "
            "returns to the corridor, then closes and locks the door."
        )
        errors = _coverage_errors(
            source,
            2,
            "Ada unlocks and opens the archive door, enters the archive, returns to the corridor, "
            "then closes and locks the door.",
        )
        self.assertTrue(any("retriev" in error.casefold() for error in errors), errors)

    def test_appositive_cast_setup_does_not_weaken_the_required_corridor_return(self):
        source = (
            "[0s-4s] Two adult library volunteers, Ada and Len, wait outside the archive. "
            "[4s-8s] Ada unlocks the archive door, opens it, enters the archive, retrieves one blue binder, "
            "returns to the corridor, then closes and locks the door."
        )
        errors = _coverage_errors(
            source,
            2,
            "Ada unlocks and opens the archive door, enters the archive, retrieves one blue binder, "
            "then closes and locks the door.",
        )
        self.assertTrue(any("return" in error.casefold() for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
