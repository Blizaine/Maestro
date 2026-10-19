"""Focused tests for the bounded single-window H3 continuity review."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.h3_single_window_continuity import (
    review_single_window_continuity, _entity_words_appear_in_order,
)


SOURCE = (
    "A hiker climbs to the ridge, plants a red flag there, then looks into the valley."
)
DRAFT = (
    "subject_definitions: The hiker wears a yellow coat.\n"
    "summary: A hiker reaches the ridge.\n"
    "retention_analysis: Preserve the hiker and flag.\n"
    "detailed_description: [Shot 1] From 0 to 5 seconds, the hiker climbs to the ridge and plants a red flag. "
    "The hiker says <d>[English] We made it.</d> [Shot 2] From 5 to 10 seconds, cut to a tight view; "
    "the hiker is back at the valley base holding the flag.\n"
    "overall_soundscape: Wind on the ridge.\n"
    "non_diegetic_music: N/A"
)
ORIGINAL_TIMELINE = (
    "[Shot 1] From 0 to 5 seconds, the hiker climbs to the ridge and plants a red flag. "
    "The hiker says <d>[English] We made it.</d> [Shot 2] From 5 to 10 seconds, cut to a tight view; "
    "the hiker is back at the valley base holding the flag."
)
REPAIRED_TIMELINE = (
    "[Shot 1] From 0 to 5 seconds, the hiker climbs to the ridge and plants a red flag. "
    "The hiker says <d>[English] We made it.</d> [Shot 2] From 5 to 10 seconds, cut to a tight view; "
    "the hiker remains at the ridge while the planted red flag stands beside them."
)


def response(
    timeline: str = REPAIRED_TIMELINE,
    *,
    status: str = "repair",
    issues: list[dict[str, str]] | None = None,
    handoffs: list[dict[str, object]] | None = None,
    groups: list[dict[str, object]] | None = None,
) -> str:
    return json.dumps({
        "state_handoffs": handoffs or [],
        "reference_groups": groups or [],
        "status": status,
        "issues": issues if issues is not None else ([{
            "evidence": "the hiker is back at the valley base holding the flag",
            "issue": "The later shot resets the hiker to a prior physical state without a source-established return.",
        }] if status == "repair" else []),
        "repaired_timeline": timeline,
    })


class H3SingleWindowContinuityTests(unittest.TestCase):
    def test_repeated_source_words_with_absent_final_word_are_rejected(self):
        # A bounded but repetitive model response must not stall validation.
        self.assertFalse(_entity_words_appear_in_order('bead ' * 18 + 'missing', 'bead ' * 100))

    def setUp(self):
        self.calls: list[dict[str, object]] = []

    def generator(self, result: str):
        def generate(**kwargs):
            self.calls.append(kwargs)
            return result
        return generate

    def review(self, result: str, **kwargs):
        return review_single_window_continuity(
            kwargs.pop("source_prompt", SOURCE),
            kwargs.pop("draft_prompt", DRAFT),
            kwargs.pop("reference_context", ""),
            kwargs.pop("duration_seconds", 10),
            self.generator(result),
            kwargs.pop("validate_candidate", lambda _candidate: True),
            kwargs.pop("check_cancelled", lambda: None),
            **kwargs,
        )

    def test_repairs_cross_shot_state_and_preserves_all_other_fields_and_dialogue(self):
        result = self.review(response(), max_new_tokens=9000)

        self.assertEqual(result["status"], "repaired")
        self.assertTrue(result["checked"])
        self.assertTrue(result["repaired"])
        self.assertIn("the hiker remains at the ridge", result["prompt"])
        self.assertEqual(result["prompt"].count("<d>[English] We made it.</d>"), 1)
        for field in (
            "subject_definitions: The hiker wears a yellow coat.",
            "summary: A hiker reaches the ridge.",
            "retention_analysis: Preserve the hiker and flag.",
            "overall_soundscape: Wind on the ridge.",
            "non_diegetic_music: N/A",
        ):
            self.assertIn(field, result["prompt"])
            self.assertEqual(result["prompt"].count(field), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["max_new_tokens"], 4096)
        self.assertEqual(self.calls[0]["temperature"], 0.15)
        self.assertIs(self.calls[0]["enable_thinking"], False)
        self.assertIn("json_schema", self.calls[0])
        self.assertEqual(result["issues"][0]["evidence"], "the hiker is back at the valley base holding the flag")

    def test_state_handoff_is_grounded_in_adjacent_target_shot_and_preserves_timing_dialogue(self):
        handoff = {
            "from_shot": 1,
            "to_shot": 2,
            "carried_state": "the previously established ridge location and flag placement persist",
            "visible_cue": "the hiker remains at the ridge while the planted red flag stands beside them.",
        }
        result = self.review(response(handoffs=[handoff]))

        self.assertEqual(result["status"], "repaired")
        self.assertEqual(result["state_handoffs"], [handoff])
        self.assertIn("<d>[English] We made it.</d>", result["prompt"])
        self.assertIn("From 0 to 5 seconds", result["prompt"])
        self.assertIn("From 5 to 10 seconds", result["prompt"])
        self.assertEqual(next(iter(self.calls[0]["json_schema"]["properties"])), "state_handoffs")

    def test_state_handoff_rejects_nonadjacent_or_unfounded_visible_cues(self):
        invalid_handoffs = (
            {
                "from_shot": 1,
                "to_shot": 3,
                "carried_state": "the hiker remains on the ridge",
                "visible_cue": "the hiker remains at the ridge while the planted red flag stands beside them.",
            },
            {
                "from_shot": 1,
                "to_shot": 2,
                "carried_state": "the hiker remains on the ridge",
                "visible_cue": "the hiker stands beneath a red moon.",
            },
        )
        for handoff in invalid_handoffs:
            with self.subTest(handoff=handoff):
                result = self.review(response(handoffs=[handoff]))

                self.assertEqual(result["status"], "invalid_response")
                self.assertEqual(result["prompt"], DRAFT)
                self.assertEqual(result["state_handoffs"], [])

    def test_explicit_same_object_views_are_encoded_in_opening_shot(self):
        source = "Picture 1 and Picture 3 are two views of the same red bicycle."
        draft = DRAFT.replace(
            "subject_definitions: The hiker wears a yellow coat.",
            "subject_definitions: N/A; <Picture 1> and <Picture 3> are object refs.",
        )
        context = (
            "<Picture 1> is an object / prop reference for red bicycle; preserve its shape.\n"
            "<Picture 3> is an object / prop reference for bicycle cabin view; preserve its details."
        )
        group = {
            "pictures": ["<Picture 1>", "<Picture 3>"],
            "entity": "red bicycle",
            "source_quote": source,
        }
        result = self.review(
            response(ORIGINAL_TIMELINE, status="ok", groups=[group]),
            source_prompt=source,
            draft_prompt=draft,
            reference_context=context,
        )

        self.assertEqual(result["status"], "repaired")
        self.assertIn(
            "<Picture 1> and <Picture 3> are views of one persistent object (red bicycle)",
            result["prompt"],
        )
        self.assertEqual(result["reference_groups"], [group])
        self.assertEqual(list(self.calls[0]["json_schema"]["properties"])[0], "state_handoffs")
        self.assertEqual(list(self.calls[0]["json_schema"]["properties"])[1], "reference_groups")
        self.assertEqual(self.calls[0]["json_schema"]["required"][0], "state_handoffs")
        self.assertNotIn("<Subject", result["prompt"])
        self.assertIn("overall_soundscape: Wind on the ridge.", result["prompt"])

    def test_implicit_but_unambiguous_same_object_views_are_allowed(self):
        source = (
            "Picture 1 shows the courier's bicycle, Picture 2 shows that bicycle with its cargo rack, "
            "and Picture 3 shows its handlebar view."
        )
        context = "\n".join(
            f"<Picture {number}> is an object / prop reference for bicycle."
            for number in (1, 2, 3)
        )
        group = {
            "pictures": ["<Picture 1>", "<Picture 2>", "<Picture 3>"],
            "entity": "bicycle",
            "source_quote": source,
        }
        draft = DRAFT.replace(
            "subject_definitions: The hiker wears a yellow coat.",
            "subject_definitions: N/A; <Picture 1>, <Picture 2>, and <Picture 3> are object refs.",
        )
        result = self.review(
            response(ORIGINAL_TIMELINE, status="ok", groups=[group]),
            source_prompt=source,
            draft_prompt=draft,
            reference_context=context,
        )

        self.assertEqual(result["status"], "repaired")
        self.assertEqual(result["reference_groups"], [group])
        self.assertIn("<Picture 1> and <Picture 2> and <Picture 3>", result["prompt"])

    def test_noncontiguous_entity_words_in_source_quote_are_allowed_in_order(self):
        source = (
            "Picture 1 shows the car in reference picture 1, and Picture 2 shows that car with four white jets."
        )
        context = (
            "<Picture 1> is an object / prop reference for the car exterior.\n"
            "<Picture 2> is an object / prop reference for the car with jets."
        )
        group = {
            "pictures": ["<Picture 1>", "<Picture 2>"],
            "entity": "car with four white jets",
            "source_quote": source,
        }
        draft = DRAFT.replace(
            "subject_definitions: The hiker wears a yellow coat.",
            "subject_definitions: N/A; <Picture 1> and <Picture 2> are object refs.",
        )
        result = self.review(
            response(ORIGINAL_TIMELINE, status="ok", groups=[group]),
            source_prompt=source,
            draft_prompt=draft,
            reference_context=context,
        )

        self.assertEqual(result["status"], "repaired")
        self.assertEqual(result["reference_groups"], [group])

    def test_noncontiguous_entity_evidence_rejects_absent_or_reordered_words(self):
        source = (
            "Picture 1 shows the car in reference picture 1, and Picture 2 shows that car with four white jets."
        )
        context = (
            "<Picture 1> is an object / prop reference for the car exterior.\n"
            "<Picture 2> is an object / prop reference for the car with jets."
        )
        for entity in ("car with six white jets", "car with jets white four"):
            with self.subTest(entity=entity):
                group = {
                    "pictures": ["<Picture 1>", "<Picture 2>"],
                    "entity": entity,
                    "source_quote": source,
                }
                result = self.review(
                    response(ORIGINAL_TIMELINE, status="ok", groups=[group]),
                    source_prompt=source,
                    reference_context=context,
                )

                self.assertEqual(result["status"], "invalid_response")
                self.assertEqual(result["prompt"], DRAFT)
                self.assertEqual(result["reference_groups"], [])

    def test_similar_but_distinct_props_cannot_be_grouped(self):
        source = "Picture 1 shows a red bicycle; Picture 2 shows a different bicycle with the same design."
        context = (
            "<Picture 1> is an object / prop reference for red bicycle.\n"
            "<Picture 2> is an object / prop reference for another bicycle."
        )
        group = {
            "pictures": ["<Picture 1>", "<Picture 2>"],
            "entity": "bicycle",
            "source_quote": source,
        }
        result = self.review(
            response(ORIGINAL_TIMELINE, status="ok", groups=[group]),
            source_prompt=source,
            reference_context=context,
        )

        self.assertEqual(result["status"], "invalid_response")
        self.assertEqual(result["prompt"], DRAFT)
        self.assertEqual(result["reference_groups"], [])

    def test_invalid_group_quote_and_non_object_reference_are_rejected(self):
        source = "Picture 1 and Picture 2 are views of the same lantern."
        group = {
            "pictures": ["<Picture 1>", "<Picture 2>"],
            "entity": "lantern",
            "source_quote": "Picture 1 and Picture 2 show the same lantern.",
        }
        for context in (
            "<Picture 1> is an object / prop reference for lantern.",
            "<Picture 1> is an object / prop reference for lantern.\n"
            "<Picture 2> is a character reference for the lantern keeper.",
        ):
            with self.subTest(context=context):
                result = self.review(
                    response(ORIGINAL_TIMELINE, status="ok", groups=[group]),
                    source_prompt=source,
                    reference_context=context,
                )
                self.assertEqual(result["status"], "invalid_response")
                self.assertEqual(result["prompt"], DRAFT)

    def test_malformed_response_falls_back_without_retrying_or_validating(self):
        validator_calls = []
        result = self.review(
            '{"status":"repair"',
            validate_candidate=lambda candidate: validator_calls.append(candidate) or True,
        )

        self.assertEqual(result["status"], "invalid_response")
        self.assertEqual(result["prompt"], DRAFT)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(validator_calls, [])

        malformed_status = json.dumps({
            "state_handoffs": [],
            "reference_groups": [],
            "status": [],
            "issues": [],
            "repaired_timeline": ORIGINAL_TIMELINE,
        })
        invalid = self.review(malformed_status)
        self.assertEqual(invalid["status"], "invalid_response")
        self.assertEqual(invalid["prompt"], DRAFT)
        self.assertEqual(len(self.calls), 2)

    def test_issue_evidence_matches_case_and_whitespace_and_returns_exact_draft_text(self):
        normalized_evidence = "The   hiker is\nback at the valley base holding the flag."
        result = self.review(response(issues=[{
            "evidence": normalized_evidence,
            "issue": "The later shot resets the hiker to a prior state.",
        }]))

        self.assertEqual(result["status"], "repaired")
        self.assertEqual(
            result["issues"][0]["evidence"],
            "the hiker is back at the valley base holding the flag.",
        )

    def test_issue_evidence_rejects_changed_entity_or_number(self):
        for evidence in (
            "the climber is back at the valley base holding the flag",
            "the hiker is back at the valley base holding flag 2",
        ):
            with self.subTest(evidence=evidence):
                validator_calls = []
                result = self.review(response(issues=[{
                    "evidence": evidence,
                    "issue": "The later shot resets the hiker to a prior state.",
                }]), validate_candidate=lambda prompt: validator_calls.append(prompt) or True)

                self.assertEqual(result["status"], "invalid_response")
                self.assertEqual(result["prompt"], DRAFT)
                self.assertEqual(validator_calls, [])

    def test_repair_that_does_not_change_the_timeline_is_rejected(self):
        no_op_repair = response(ORIGINAL_TIMELINE, status="repair")
        result = self.review(no_op_repair)

        self.assertEqual(result["status"], "invalid_response")
        self.assertEqual(result["prompt"], DRAFT)

    def test_structural_changes_are_rejected_before_external_validation(self):
        bad_candidates = (
            REPAIRED_TIMELINE.replace("cut to a tight view", "cut to a tight view, then cut back to the valley"),
            REPAIRED_TIMELINE.replace("5 to 10 seconds", "5 to 11 seconds"),
            REPAIRED_TIMELINE.replace("cut to a tight view", "cut to <Picture 9>"),
            REPAIRED_TIMELINE.replace("[Shot 2]", "[Shot 3]"),
        )
        for candidate in bad_candidates:
            with self.subTest(candidate=candidate):
                validator_calls = []
                result = self.review(
                    response(candidate),
                    validate_candidate=lambda prompt: validator_calls.append(prompt) or True,
                )
                self.assertEqual(result["status"], "invalid_candidate")
                self.assertEqual(result["prompt"], DRAFT)
                self.assertEqual(validator_calls, [])

    def test_timestamp_values_cannot_drift_within_the_clip_duration(self):
        candidate = REPAIRED_TIMELINE.replace("From 5 to 10 seconds", "From 6 to 10 seconds")
        validator_calls = []
        result = self.review(
            response(candidate),
            validate_candidate=lambda prompt: validator_calls.append(prompt) or True,
        )

        self.assertEqual(result["status"], "invalid_candidate")
        self.assertIn("timestamp values", result["diagnostic"])
        self.assertEqual(result["prompt"], DRAFT)
        self.assertEqual(validator_calls, [])

    def test_reference_group_insertion_uses_original_offset_after_prior_dialogue(self):
        source = "Picture 1 shows the courier's bicycle, and Picture 2 shows its handlebar view."
        context = (
            "<Picture 1> is an object / prop reference for bicycle.\n"
            "<Picture 2> is an object / prop reference for bicycle handlebar."
        )
        timeline = (
            "<d>[English] The slate reads [Shot 1].</d> "
            "[Shot 1] From 0 to 5 seconds, the courier reaches the ridge. "
            "[Shot 2] From 5 to 10 seconds, the courier pauses."
        )
        draft = DRAFT.replace(ORIGINAL_TIMELINE, timeline).replace(
            "subject_definitions: The hiker wears a yellow coat.",
            "subject_definitions: <Picture 1> and <Picture 2> are object refs.",
        )
        group = {
            "pictures": ["<Picture 1>", "<Picture 2>"],
            "entity": "bicycle",
            "source_quote": source,
        }
        result = self.review(
            response(timeline, status="ok", groups=[group]),
            source_prompt=source,
            draft_prompt=draft,
            reference_context=context,
        )

        self.assertEqual(result["status"], "repaired")
        repaired_timeline = result["prompt"].split("detailed_description: ", 1)[1].split(
            "\noverall_soundscape:", 1
        )[0]
        insertion = "<Picture 1> and <Picture 2> are views of one persistent object (bicycle)"
        dialogue = "<d>[English] The slate reads [Shot 1].</d>"
        self.assertEqual(repaired_timeline.count(dialogue), 1)
        shot_start = repaired_timeline.index("[Shot 1]", repaired_timeline.index("</d>") + len("</d>"))
        self.assertGreater(repaired_timeline.index(insertion), shot_start)
        self.assertLess(repaired_timeline.index(insertion), repaired_timeline.index("[Shot 2]"))

    def test_external_validator_rejection_falls_back_to_original(self):
        result = self.review(response(), validate_candidate=lambda _candidate: False)

        self.assertEqual(result["status"], "validator_rejected")
        self.assertEqual(result["prompt"], DRAFT)
        self.assertTrue(result["checked"])
        self.assertFalse(result["repaired"])

    def test_independent_flashback_may_have_no_state_handoffs(self):
        source = "A person enters the room. A separate flashback shows that person outside before arriving."
        timeline = (
            "[Shot 1] From 0 to 5 seconds, a person enters the room. "
            "[Shot 2] From 5 to 10 seconds, a separate flashback shows the person outside before arriving."
        )
        draft = DRAFT.replace(ORIGINAL_TIMELINE, timeline)
        result = self.review(
            response(timeline, status="ok"),
            source_prompt=source,
            draft_prompt=draft,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["prompt"], draft)
        self.assertEqual(result["state_handoffs"], [])

    def test_single_shot_and_unsupported_field_skip_without_generation(self):
        one_shot = DRAFT.replace(
            " [Shot 2] From 5 to 10 seconds, cut to a tight view; "
            "the hiker is back at the valley base holding the flag.",
            "",
        )
        single_result = self.review(response(), draft_prompt=one_shot)
        self.assertEqual(single_result["status"], "skipped_single_shot")
        self.assertEqual(single_result["prompt"], one_shot)

        unsupported = "summary: A hiker reaches the ridge.\noverall_soundscape: Wind."
        unsupported_result = self.review(response(), draft_prompt=unsupported)
        self.assertEqual(unsupported_result["status"], "skipped_unsupported_field")
        self.assertEqual(unsupported_result["prompt"], unsupported)
        self.assertEqual(self.calls, [])

    def test_oversized_input_skips_without_silent_truncation(self):
        oversized = self.review(DRAFT, source_prompt="x" * 30_001)
        self.assertEqual(oversized["status"], "skipped_input_too_large")
        self.assertEqual(oversized["prompt"], DRAFT)
        self.assertEqual(self.calls, [])

    def test_missing_duration_skips_explicitly(self):
        result = self.review(response(), duration_seconds=None)
        self.assertEqual(result["status"], "skipped_invalid_duration")
        self.assertEqual(result["prompt"], DRAFT)
        self.assertEqual(self.calls, [])

    def test_cancellation_propagates(self):
        def cancelled():
            raise InterruptedError("cancelled")

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            self.review(response(), check_cancelled=cancelled)
        self.assertEqual(self.calls, [])

        checks = 0

        def cancelled_after_generation():
            nonlocal checks
            checks += 1
            if checks == 3:
                raise InterruptedError("cancelled after generation")

        with self.assertRaisesRegex(InterruptedError, "cancelled after generation"):
            self.review(response(), check_cancelled=cancelled_after_generation)
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
