"""CPU regressions for source-owned dialogue timing anchors."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_dialogue_anchors import (  # noqa: E402
    resolve_h3_dialogue_timing_anchors,
    validate_h3_dialogue_timing_anchors,
)
from services.h3_story_ledger import extract_locked_dialogue, extract_source_events  # noqa: E402


CURLY_DIALOGUE_SOURCE = (
    "Two adults only: Celia, an adult paper conservator, solely holds one dry blue print by its lower corners; "
    "Omar, an adult framing assistant, solely holds one empty black frame by its sides. "
    "Use one uninterrupted shot with no cuts. Preserve these four exact lines with curly quotation marks, "
    "in this order, and add no other spoken words: Celia says, “The blue copy is dry.” "
    "Omar replies, “I’ll bring the frame closer.” Celia says, “Keep the top edge level.” "
    "Omar says, “Ready when you are.” Omar brings only the frame closer after his first line. "
    "Celia places the same print inside only after her second line. "
    "They set the single completed frame on the worktable after the final line. "
    "Do not paraphrase, duplicate, reorder, or reassign any line or object."
)


def _event_id(events, text_fragment):
    matches = [
        event["event_id"] for event in events
        if text_fragment.casefold() in event["text"].casefold()
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one event for {text_fragment!r}, got {matches!r}")
    return matches[0]


def _line(line, *, dialogue_id=None, speaker=None, text=None):
    return {
        "dialogue_id": dialogue_id or line["dialogue_id"],
        "speaker": speaker or line["speaker"],
        "text": text if text is not None else line["text"],
    }


class H3DialogueAnchorTests(unittest.TestCase):
    def setUp(self):
        self.events = extract_source_events(CURLY_DIALOGUE_SOURCE)
        self.locked = extract_locked_dialogue(CURLY_DIALOGUE_SOURCE)
        self.by_id = {item["dialogue_id"]: item for item in self.locked}
        self.e_first = _event_id(self.events, "Omar brings only the frame")
        self.e_second = _event_id(self.events, "Celia places the same print")
        self.e_final = _event_id(self.events, "They set the single completed frame")

    def _accepted_windows(self, *, include_celia_second=True, d3_speaker=None, d3_text=None,
                          d3_in_window3=False):
        d1, d2, d3, d4 = (self.by_id[f"D{i}"] for i in range(1, 5))
        window1 = {
            "index": 1,
            "shots": [{
                "shot": 1, "start_seconds": 0.0, "end_seconds": 10.0,
                "dialogue": [_line(d1)],
            }],
        }
        window2_lines = [_line(d2)]
        if include_celia_second and not d3_in_window3:
            window2_lines.append(_line(d3, speaker=d3_speaker, text=d3_text))
        window2 = {
            "index": 2,
            "shots": [
                {"shot": 1, "start_seconds": 0.0, "end_seconds": 6.8,
                 "dialogue": [window2_lines[0]]},
                {"shot": 2, "start_seconds": 6.8, "end_seconds": 13.6,
                 "dialogue": window2_lines[1:]},
            ],
        }
        window3_shots = [
            {"shot": 1, "start_seconds": 0.0, "end_seconds": 5.0,
             "dialogue": [_line(d4)]},
            {"shot": 2, "start_seconds": 5.0, "end_seconds": 7.0, "dialogue": []},
            {"shot": 3, "start_seconds": 7.0, "end_seconds": 9.0, "dialogue": []},
            {"shot": 4, "start_seconds": 9.0, "end_seconds": 13.6, "dialogue": []},
        ]
        if d3_in_window3:
            window3_shots[3]["dialogue"] = [_line(d3)]
        return [window1, window2, {"index": 3, "shots": window3_shots}]

    def _event_positions(self, *, second_at=3, final_at=4):
        return {
            self.e_first: [{"segment": 3, "shot": 2}],
            self.e_second: [{"segment": 3, "shot": second_at}],
            self.e_final: [{"segment": 3, "shot": final_at}],
        }

    def test_candidate2_curly_quote_case_resolves_and_verifies_real_lines(self):
        resolved = resolve_h3_dialogue_timing_anchors(self.events, self.locked)
        requirements = {
            (item["event_id"], item["dialogue_id"], item["speaker"])
            for item in resolved["requirements"]
        }
        self.assertEqual(requirements, {
            (self.e_first, "D2", "Omar"),
            (self.e_second, "D3", "Celia"),
            (self.e_final, "D4", "Omar"),
        })
        self.assertEqual(resolved["unresolved"], [])

        result = validate_h3_dialogue_timing_anchors(
            self.events, self.locked, self._accepted_windows(),
            source_event_positions=self._event_positions(),
        )
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(result["violations"], [])
        self.assertEqual(
            {item["dialogue_id"] for item in result["verified"]},
            {"D2", "D3", "D4"},
        )

    def test_wording_without_exact_dialogue_delivery_does_not_prove_anchor(self):
        windows = self._accepted_windows(include_celia_second=False)
        windows[2]["shots"][2]["action"] = (
            "Celia places the same print inside only after her second line."
        )
        result = validate_h3_dialogue_timing_anchors(
            self.events, self.locked, windows,
            source_event_positions=self._event_positions(),
        )
        self.assertTrue(any(
            item.get("dialogue_id") == "D3"
            and "absent from accepted shots" in item["reason"]
            for item in result["violations"]
        ), result)

    def test_wrong_speaker_on_exact_ordinal_id_does_not_satisfy_anchor(self):
        result = validate_h3_dialogue_timing_anchors(
            self.events, self.locked,
            self._accepted_windows(d3_speaker="Omar"),
            source_event_positions=self._event_positions(),
        )
        self.assertTrue(any(
            item.get("dialogue_id") == "D3"
            and "wrong immutable speaker or text" in item["reason"]
            for item in result["violations"]
        ), result)
        self.assertNotIn("D3", {item["dialogue_id"] for item in result["verified"]})

    def test_action_before_required_line_fails_even_when_line_is_delivered_later(self):
        result = validate_h3_dialogue_timing_anchors(
            self.events, self.locked,
            self._accepted_windows(d3_in_window3=True),
            source_event_positions=self._event_positions(second_at=2),
        )
        self.assertTrue(any(
            item.get("dialogue_id") == "D3"
            and "shot order does not prove" in item["reason"]
            for item in result["violations"]
        ), result)

    def test_same_shot_line_and_action_order_is_unresolved(self):
        positions = self._event_positions()
        positions[self.e_second] = [{"segment": 2, "shot": 2}]
        result = validate_h3_dialogue_timing_anchors(
            self.events, self.locked, self._accepted_windows(),
            source_event_positions=positions,
        )
        self.assertFalse(any(
            item.get("dialogue_id") == "D3" for item in result["violations"]
        ), result)
        self.assertTrue(any(
            item.get("dialogue_id") == "D3"
            and "share a shot" in item["reason"]
            for item in result["unresolved"]
        ), result)

    def test_first_global_and_last_per_speaker_ordinals_resolve_from_catalog(self):
        events = [
            {"event_id": "E1", "text": "Omar adjusts the frame after his first line."},
            {"event_id": "E2", "text": "Omar rests after his last line."},
            {"event_id": "E3", "text": "Celia waits after the second line."},
        ]
        result = resolve_h3_dialogue_timing_anchors(events, self.locked)
        resolved = {item["event_id"]: item["dialogue_id"] for item in result["requirements"]}
        self.assertEqual(resolved, {"E1": "D2", "E2": "D4", "E3": "D2"})
        self.assertEqual(result["unresolved"], [])

    def test_named_possessive_speaker_resolves_even_with_multiple_catalog_lines(self):
        events = [
            {"event_id": "E1", "text": "Celia places the print after Celia's second line."},
            {"event_id": "E2", "text": "Omar lifts the frame after Omar's final line."},
        ]
        result = resolve_h3_dialogue_timing_anchors(events, self.locked)
        self.assertEqual(
            {item["event_id"]: item["dialogue_id"] for item in result["requirements"]},
            {"E1": "D3", "E2": "D4"},
        )
        self.assertEqual(result["unresolved"], [])

    def test_ambiguous_pronoun_reference_is_unresolved(self):
        events = [{
            "event_id": "E9",
            "text": "Mara and Celia set the completed frame after her second line.",
        }]
        result = resolve_h3_dialogue_timing_anchors(events, self.locked)
        self.assertEqual(result["requirements"], [])
        self.assertEqual(len(result["unresolved"]), 1)
        self.assertIn("not have one uniquely named", result["unresolved"][0]["reason"])

    def test_action_before_line_anchor_is_valid_only_when_order_is_proven(self):
        events = [{"event_id": "E9", "text": "Celia sets the frame before her second line."}]
        accepted = self._accepted_windows()
        result = validate_h3_dialogue_timing_anchors(
            events, self.locked, accepted,
            source_event_positions={"E9": [{"segment": 2, "shot": 1}]},
        )
        self.assertEqual(result["violations"], [])
        self.assertEqual([item["dialogue_id"] for item in result["verified"]], ["D3"])

    def test_fragmented_locked_line_counts_only_when_manifest_reconstructs_exact_text(self):
        source_events = [{"event_id": "E1", "text": "Nora places the map after her first line."}]
        locked = [{
            "dialogue_id": "D1", "speaker": "Nora", "text": "Bring the map to the desk.",
            "source_offset": 10,
        }]
        fragment_catalog = [
            {"dialogue_id": "D1F1", "source_dialogue_id": "D1", "fragment_index": 1,
             "fragment_count": 2, "speaker": "Nora", "text": "Bring the map"},
            {"dialogue_id": "D1F2", "source_dialogue_id": "D1", "fragment_index": 2,
             "fragment_count": 2, "speaker": "Nora", "text": "to the desk."},
        ]
        accepted = [{
            "index": 1,
            "shots": [
                {"shot": 1, "start_seconds": 0, "end_seconds": 3,
                 "dialogue": [{"dialogue_id": "D1F1", "speaker": "Nora", "text": "Bring the map"}]},
                {"shot": 2, "start_seconds": 3, "end_seconds": 6,
                 "dialogue": [{"dialogue_id": "D1F2", "speaker": "Nora", "text": "to the desk."}]},
                {"shot": 3, "start_seconds": 6, "end_seconds": 8, "dialogue": []},
            ],
        }]
        result = validate_h3_dialogue_timing_anchors(
            source_events, locked, accepted,
            source_event_positions={"E1": [{"segment": 1, "shot": 3}]},
            render_dialogue_catalog=fragment_catalog,
        )
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(result["violations"], [])
        self.assertEqual([item["dialogue_id"] for item in result["verified"]], ["D1"])

        missing_fragment = [{
            **accepted[0],
            "shots": accepted[0]["shots"][:1] + accepted[0]["shots"][2:],
        }]
        rejected = validate_h3_dialogue_timing_anchors(
            source_events, locked, missing_fragment,
            source_event_positions={"E1": [{"segment": 1, "shot": 3}]},
            render_dialogue_catalog=fragment_catalog,
        )
        self.assertTrue(rejected["violations"])
        self.assertEqual(rejected["verified"], [])

    def test_fragment_without_source_manifest_is_unresolved_not_assumed_missing(self):
        source_events = [{"event_id": "E1", "text": "Nora places the map after her first line."}]
        locked = [{"dialogue_id": "D1", "speaker": "Nora", "text": "Bring the map to the desk."}]
        accepted = [{
            "index": 1,
            "shots": [{
                "shot": 1, "start_seconds": 0, "end_seconds": 3,
                "dialogue": [{
                    "dialogue_id": "D1F1", "speaker": "Nora", "text": "Bring the map",
                    "source_dialogue_id": "D1", "fragment_index": 1,
                }],
            }],
        }]
        result = validate_h3_dialogue_timing_anchors(
            source_events, locked, accepted,
            source_event_positions={"E1": [{"segment": 1, "shot": 1}]},
        )
        self.assertEqual(result["violations"], [])
        self.assertTrue(any(
            "lacks complete source/index/count metadata" in item["reason"]
            for item in result["unresolved"]
        ), result)

    def test_unanchored_fragment_delivery_errors_do_not_affect_anchor_result(self):
        source_events = [{"event_id": "E1", "text": "Nora sets the map after her first line."}]
        locked = [
            {"dialogue_id": "D1", "speaker": "Nora", "text": "Ready to begin."},
            {"dialogue_id": "D2", "speaker": "Eli", "text": "I have the key."},
        ]
        accepted = [{
            "index": 1,
            "shots": [
                {"shot": 1, "start_seconds": 0, "end_seconds": 3,
                 "dialogue": [{"dialogue_id": "D1", "speaker": "Nora", "text": "Ready to begin."}]},
                {"shot": 2, "start_seconds": 3, "end_seconds": 6,
                 "dialogue": []},
                {"shot": 3, "start_seconds": 6, "end_seconds": 8,
                 "dialogue": [{
                     "dialogue_id": "D2F1", "source_dialogue_id": "D2",
                     "fragment_index": 1, "speaker": "Eli", "text": "I have",
                 }]},
            ],
        }]
        result = validate_h3_dialogue_timing_anchors(
            source_events, locked, accepted,
            source_event_positions={"E1": [{"segment": 1, "shot": 2}]},
        )
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["unresolved"], [])
        self.assertEqual([item["dialogue_id"] for item in result["verified"]], ["D1"])


if __name__ == "__main__":
    unittest.main()
