"""Chronology repairs preserve accepted ownership and fixed creative beats."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.music_arc import normalize_music_arc
from services.director.music_arc_repair import (
    allocation_report, build_allocation_repair_request, apply_allocation_patch,
)
from services.director.planners.music_video import MusicVideoPlanner


def fixture():
    contract = {"source": "A courier collects and delivers parcels.", "events": [
        {"event_id": f"M{i}", "text": f"source action M{i}"} for i in range(1, 7)]}
    groups = [{"start": i * 6, "end": (i + 1) * 6, "label": "scene", "energy": None}
              for i in range(14)]
    actions = ["ties a shoelace", "opens the gate", "folds a map", "checks a clock",
               "crosses a bridge", "rings the bell", "packs a parcel", "unlocks a bicycle",
               "collects a receipt", "examines the address", "delivers the box",
               "raises a hand", "closes the door", "rests on the bench"]
    beats = [{"clip_index": i, "visual_beat": "The courier " + action + ".",
              "purpose": f"Advance delivery beat {i}", "environment": "Town square",
              "camera": "steady medium view", "intentional_reprise": False, "reprise_reason": ""}
             for i, action in enumerate(actions)]
    original = {"beats": beats, "source_event_clips": {
        "M1": 0, "M2": 0, "M3": 1, "M4": 10, "M5": 7, "M6": 12}}
    corrected = {"source_event_clips": {"M1": 0, "M2": 0, "M3": 1, "M4": 10, "M5": 11, "M6": 12}}
    return contract, groups, original, corrected


class MusicArcAllocationRepairTests(unittest.TestCase):
    def test_exact_saved_failure_sequences(self):
        contract, groups, response, _ = fixture()
        report = allocation_report(response["source_event_clips"], contract, len(groups))
        self.assertEqual(report["expected"], [f"M{i}" for i in range(1, 7)])
        self.assertEqual(report["received"], ["M1", "M2", "M3", "M5", "M4", "M6"])
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["duplicate"], [])
        self.assertEqual(report["unknown"], [])
        self.assertEqual(report["out_of_order"][0]["earlier_id"], "M4")
        self.assertEqual(report["preserved_assignments"], {"M1": 0, "M2": 0, "M3": 1, "M6": 12})
        retry = {"M1": 0, "M2": 0, "M3": 1, "M4": 7, "M5": 7, "M6": 6}
        self.assertEqual(allocation_report(retry, contract, 14)["received"],
                         ["M1", "M2", "M3", "M6", "M4", "M5"])

    def test_missing_duplicate_unknown_and_reordered_allocations_are_rejected(self):
        contract, groups, original, corrected = fixture()
        for kind in ("missing", "duplicate", "unknown", "reordered"):
            rows = [{**beat, "source_event_ids": []} for beat in original["beats"]]
            for event, index in corrected["source_event_clips"].items():
                rows[index]["source_event_ids"].append(event)
            if kind == "missing":
                rows[12]["source_event_ids"] = []
            elif kind == "duplicate":
                rows[13]["source_event_ids"] = ["M6"]
            elif kind == "unknown":
                rows[13]["source_event_ids"] = ["M99"]
            else:
                rows[7]["source_event_ids"] = rows[11]["source_event_ids"]
                rows[11]["source_event_ids"] = []
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                normalize_music_arc(rows, groups, contract)

    def test_diagnostics_include_missing_duplicate_and_unknown_ids(self):
        contract, groups, _original, _corrected = fixture()
        allocation = [{"event_id": "M1", "clip_index": 0},
                      {"event_id": "M1", "clip_index": 1},
                      {"event_id": "M99", "clip_index": 2}]
        report = allocation_report(allocation, contract, len(groups))
        self.assertEqual(report["missing"], [f"M{i}" for i in range(2, 7)])
        self.assertEqual(report["duplicate"], ["M1"])
        self.assertEqual(report["unknown"], ["M99"])

    def test_focused_request_pins_valid_assignments_and_never_requests_beats(self):
        contract, groups, original, _ = fixture()
        prompt, schema, report = build_allocation_repair_request(original, groups, contract)
        self.assertIn("preserved_assignments", prompt)
        self.assertNotIn("beats", schema["properties"])
        properties = schema["properties"]["source_event_clips"]["properties"]
        for event, index in report["preserved_assignments"].items():
            self.assertEqual((properties[event]["minimum"], properties[event]["maximum"]), (index, index))
        self.assertEqual((properties["M5"]["minimum"], properties["M5"]["maximum"]), (1, 12))

    def test_successful_candidate_preserves_original_beats_and_input(self):
        contract, groups, original, correction = fixture()
        before = copy.deepcopy(original)
        repaired = apply_allocation_patch(original, correction, groups, contract)
        self.assertEqual(original, before)
        self.assertEqual(repaired["beats"], before["beats"])
        self.assertEqual(repaired["source_event_clips"], correction["source_event_clips"])

    def test_production_json_parser_preserves_the_allocation_envelope(self):
        contract, groups, original, correction = fixture()
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        parsed = planner._parse_json_response(json.dumps(correction))
        self.assertEqual(apply_allocation_patch([original], parsed, groups, contract)
                         ["source_event_clips"], correction["source_event_clips"])

    def test_incomplete_duplicate_and_unknown_maps_receive_focused_correction(self):
        contract, groups, original, correction = fixture()
        valid_map = correction["source_event_clips"]
        cases = [
            {key: value for key, value in valid_map.items() if key != "M6"},
            {**valid_map, "M99": 13},
            [{"event_id": key, "clip_index": value} for key, value in valid_map.items()]
            + [{"event_id": "M6", "clip_index": 13}],
        ]
        for assignments in cases:
            with self.subTest(assignments=assignments):
                draft = {**original, "source_event_clips": assignments}
                before = copy.deepcopy(draft)
                repaired = apply_allocation_patch(draft, correction, groups, contract)
                self.assertEqual(repaired["source_event_clips"], valid_map)
                self.assertEqual(draft, before)

    def test_valid_original_cannot_be_corrupted_by_an_allocation_patch(self):
        contract, groups, original, correction = fixture()
        original["source_event_clips"] = copy.deepcopy(correction["source_event_clips"])
        before = copy.deepcopy(original)
        correction["source_event_clips"]["M5"] = 10
        with self.assertRaisesRegex(ValueError, "preserved"):
            apply_allocation_patch(original, correction, groups, contract)
        self.assertEqual(original, before)

    def test_rejected_patches_cannot_corrupt_original(self):
        contract, groups, original, correction = fixture()
        before = copy.deepcopy(original)
        proposals = [copy.deepcopy(original), {}, {"source_event_clips": {"M1": 0}},
                     {"source_event_clips": {**correction["source_event_clips"], "M99": 3}},
                     {"source_event_clips": {**correction["source_event_clips"], "M3": 2}},
                     {"source_event_clips": {**correction["source_event_clips"], "M4": True}},
                     {**correction, "beats": []}]
        for proposal in proposals:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                apply_allocation_patch(original, proposal, groups, contract)
            self.assertEqual(original, before)

    def test_planner_commits_only_a_valid_focused_correction(self):
        contract, groups, original, correction = fixture()
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"test": "allocation"})
        planner._music_narrative_contract = contract
        planner._use_lyrics_context = False
        with patch.object(planner, "_call_llm_json", side_effect=[original, correction]) as writer:
            result = planner._plan_music_arc(groups, contract["source"])
        self.assertEqual(writer.call_count, 2)
        self.assertNotIn("beats", writer.call_args.kwargs["json_schema"]["properties"])
        self.assertEqual([event for row in result for event in row["source_event_ids"]],
                         [f"M{i}" for i in range(1, 7)])
        self.assertNotIn("music_arc_allocation_repair", planner._planning_resume_checkpoint)

    def test_failed_repair_leaves_draft_and_accepted_checkpoint_unchanged(self):
        contract, groups, original, _correction = fixture()
        before = copy.deepcopy(original)
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"test": "failure"})
        planner._music_narrative_contract = contract
        planner._use_lyrics_context = False
        with patch.object(planner, "_call_llm_json", side_effect=[original, {"source_event_clips": {}}]) as writer:
            with self.assertRaisesRegex(RuntimeError, "expected=.*M6"):
                planner._plan_music_arc(groups, contract["source"])
        self.assertEqual(writer.call_count, 2)
        self.assertNotIn("music_creative_arc", planner._planning_resume_checkpoint)
        self.assertEqual(planner._planning_resume_checkpoint["music_arc_allocation_repair"]["response"], before)
        self.assertEqual(original, before)

    def test_resume_reuses_fixed_outline_and_requests_only_ownership(self):
        contract, groups, original, correction = fixture()
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"test": "resume"})
        planner._music_narrative_contract = contract
        planner._use_lyrics_context = False
        planner._planning_resume_checkpoint["music_arc_allocation_repair"] = {
            "contract": copy.deepcopy(contract), "groups": copy.deepcopy(groups),
            "response": copy.deepcopy(original),
        }
        with patch.object(planner, "_call_llm_json", return_value=correction) as writer:
            result = planner._plan_music_arc(groups, contract["source"])
        self.assertEqual(writer.call_count, 1)
        self.assertNotIn("beats", writer.call_args.kwargs["json_schema"]["properties"])
        self.assertEqual([row["visual_beat"] for row in result],
                         [row["visual_beat"] for row in original["beats"]])


if __name__ == "__main__":
    unittest.main()
