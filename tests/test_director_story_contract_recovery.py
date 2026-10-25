"""Regressions for fixed-shot H3 source ownership and physical-state recovery."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.physical_continuity import validate_physical_state_sequence
from services.director.planners.short_film import (
    ShortFilmPlanner,
    _h3_source_physical_patch_is_applicable,
)
from services.director.source_contract import source_contract_issues


def _source_contract():
    return {
        "revision": 1,
        "events": [{"event_id": "E1", "text": "A courier sits down"}],
        "audio_policy": {
            "generated_speech_allowed": True,
            "generated_music_allowed": True,
        },
    }


def _bad_state(index):
    state = {
        "opening": [
            {"subject_id": "char_0", "predicate": "grounded", "value": "grounded"},
        ],
        "transitions": [],
        "closing": [
            {"subject_id": "char_0", "predicate": "grounded", "value": "grounded"},
        ],
        "continuity": "independent" if index == 0 else "continuous",
    }
    if index == 0:
        state["transitions"] = [{
            "subject_id": "char_0",
            "predicate": "posture",
            "from_value": "standing",
            "to_value": "seated",
            "action": "Courier sits down",
        }]
    return state


def _shot(index, *, source_ids=True, broken_state=True, dialogue=False, title_prefix="Draft"):
    row = {
        "title": f"{title_prefix} shot {index + 1}",
        "duration_sec": 10,
        "scene_goal": "A courier settles into the room",
        "narrative_role": "setup" if index == 0 else "rising_action",
        "scene_type": "dialogue" if dialogue and index == 0 else "action",
        "continuity_strategy": "independent" if index == 0 else "continuous",
        "continuity_group": "courier_room",
        "story_scene_number": 1,
        "causal_handoff": "The courier enters the quiet room",
        "persistent_story_state": "The courier is sitting in the room",
        "physical_state": _bad_state(index) if broken_state else {
            "opening": [
                {"subject_id": "char_0", "predicate": "grounded", "value": "grounded"},
                {"subject_id": "char_0", "predicate": "posture", "value": "seated"},
            ],
            "transitions": [],
            "closing": [
                {"subject_id": "char_0", "predicate": "grounded", "value": "grounded"},
                {"subject_id": "char_0", "predicate": "posture", "value": "seated"},
            ],
            "continuity": "independent" if index == 0 else "continuous",
        },
        "subjects_on_screen": [{
            "visual_description": "An adult courier in a navy jacket",
            "character_id": "char_0",
            "speaker_name": "Courier",
            "position_or_relation": "center frame beside a chair",
            "wardrobe": "navy jacket, gray trousers, black shoes",
        }],
        "spatial_setup": "The courier stands beside a chair in a quiet room",
        "environment": "A quiet room in a short film",
        "visual_style": "Natural cinematic realism",
        "lighting": "Soft window light",
        "mood": "Quiet and focused",
        "action_beats": ["The courier sits down" if index == 0 else "The courier looks across the room"],
        "dialogue_beats": [],
        "camera_plan": {
            "framing": "medium shot",
            "movement": "static",
            "movement_intensity": "none",
        },
        "audio_plan": {
            "mode": "ambient_only",
            "ambience": "Quiet room tone",
            "effects": [],
            "timing_anchor": "video",
            "lip_sync_critical": False,
        },
        "ending_beat": "The courier settles into the chair",
        "closing_blocking": "The courier sits at center frame",
        "video_prompt": "The courier sits down beside a chair.",
        "multishot": False,
        "window_prompts": [],
    }
    if source_ids:
        row["source_event_ids"] = ["E1"]
    if dialogue and index == 0:
        row["dialogue_beats"] = [{
            "speaker_id": "char_0",
            "spoken_text": "Ready.",
            "language": "English",
            "delivery": "quietly",
            "physical_cue": "The courier settles back in the chair",
            "priority": "high",
        }]
        row["video_prompt"] += " <d>[English] Ready.</d>"
        row["audio_plan"] = {
            "mode": "dialogue_driven",
            "ambience": "Quiet room tone",
            "effects": [],
            "timing_anchor": "audio",
            "lip_sync_critical": True,
        }
    return row


def _three_shots(**kwargs):
    return [_shot(index, **kwargs) for index in range(3)]


def _compact_patch():
    return {
        "event_shots": {"E1": 1},
        "initial_facts": [
            {"subject_id": "char_0", "predicate": "grounded", "value": "grounded"},
            {"subject_id": "char_0", "predicate": "posture", "value": "standing"},
        ],
        "event_transitions": {
            "E1": [{
                "subject_id": "char_0",
                "predicate": "posture",
                "to_value": "seated",
                "action": "The courier sits down",
            }],
        },
    }


class _SequencePlanner(ShortFilmPlanner):
    def __init__(self, responses):
        super().__init__(llm_generate=lambda **kwargs: "")
        self.responses = list(responses)
        self.calls = []
        self._uses_generated_shot_images = False
        self._video_model = "minimax_h3"

    def _call_llm_json(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


def _plan(planner, rows, *, manifest=None):
    return planner._plan_story_h3_native(
        story_description="A courier sits down.",
        screenplay="COURIER\nReady." if manifest else "A courier sits down.",
        screenplay_dialogue_manifest=manifest,
        reference_image_path=None,
        char_profiles=[],
        has_reference=False,
        target_duration=30,
        fps=24,
        frames_steps=17,
        frames_minimum=124,
        frames_maximum=345,
        require_source_contract=False,
    )


class DirectorStoryContractRecoveryTests(unittest.TestCase):
    def test_fixed_shot_gate_excludes_authored_dialogue_motion_errors(self):
        self.assertFalse(_h3_source_physical_patch_is_applicable([
            "Shot 1: dialogue-bearing video_prompt omits assigned source motion (boarding)",
        ]))
        self.assertFalse(_h3_source_physical_patch_is_applicable([
            "Shot 1: dialogue-bearing video_prompt omits assigned source prop operation (draw: sword)",
        ]))
        self.assertTrue(_h3_source_physical_patch_is_applicable([
            "Source actions must be allocated exactly once in source order",
            "Shot 1 transition 1 for 'char_0 posture' has no opening/current fact",
        ]))

    def test_fixed_shot_patch_repairs_repeated_e1_and_replays_posture_across_three_shots(self):
        rows = _three_shots()
        original = copy.deepcopy(rows)
        contract = _source_contract()
        issues = source_contract_issues(rows, contract, require_ids=True)
        issues.extend(validate_physical_state_sequence(rows))
        planner = _SequencePlanner([_compact_patch()])

        repaired, request_count = planner._repair_h3_source_physical_plan(
            rows,
            contract,
            issues,
            planner_token_budget=4096,
            request_budget=2,
        )

        self.assertEqual(request_count, 1)
        self.assertEqual(
            [event_id for row in repaired for event_id in row["source_event_ids"]],
            ["E1"],
        )
        self.assertEqual(repaired[0]["physical_state"]["opening"][-1]["value"], "standing")
        self.assertEqual(repaired[0]["physical_state"]["transitions"][0]["to_value"], "seated")
        self.assertEqual(repaired[0]["physical_state"]["closing"][-1]["value"], "seated")
        for row in repaired[1:]:
            self.assertEqual(row["physical_state"]["opening"][-1]["value"], "seated")
            self.assertEqual(row["physical_state"]["closing"][-1]["value"], "seated")
        self.assertEqual(
            source_contract_issues(repaired, contract, require_ids=True)
            + validate_physical_state_sequence(repaired),
            [],
        )
        for before, after in zip(original, repaired):
            for key in set(before) | set(after):
                if key not in {"source_event_ids", "physical_state", "_director_transition_event_ids"}:
                    self.assertEqual(after.get(key), before.get(key), key)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(planner.calls[0]["max_tokens"], 4096)
        self.assertEqual(planner.calls[0]["image_paths"], [])

    def test_speech_allowed_optional_id_plan_uses_patch_and_preserves_dialogue(self):
        rows = _three_shots(dialogue=True)
        planner = _SequencePlanner([rows, _compact_patch()])
        manifest = [{"speaker_name": "Courier", "spoken_text": "Ready."}]

        shots, _ = _plan(planner, rows, manifest=manifest)

        self.assertEqual(len(planner.calls), 2)
        self.assertIn("event_shots", planner.calls[1]["json_schema"]["properties"])
        self.assertEqual(shots[0].dialogue_beats[0].spoken_text, "Ready.")
        self.assertIn("Ready.", shots[0].video_prompt)
        self.assertEqual(shots[0].metadata["source_event_ids"], ["E1"])
        self.assertEqual([shot.metadata["source_event_ids"] for shot in shots[1:]], [[], []])
        self.assertEqual(shots[0].metadata["title"], "Draft shot 1")

    def test_leftover_contract_error_after_full_plan_repair_gets_one_focused_recovery(self):
        incomplete = [_shot(0), _shot(1)]
        # The first response opted into source ownership, but the whole-plan
        # repair drops those optional IDs. The original opt-in must survive so
        # the final focused recovery still restores ownership metadata.
        rewritten = _three_shots(source_ids=False, title_prefix="Rewritten")
        planner = _SequencePlanner([incomplete, rewritten, _compact_patch()])

        shots, _ = _plan(planner, incomplete, manifest=[])

        self.assertEqual(len(planner.calls), 3)
        self.assertEqual(planner.calls[1]["json_schema"]["type"], "array")
        self.assertIn("event_shots", planner.calls[2]["json_schema"]["properties"])
        self.assertEqual([shot.metadata["title"] for shot in shots], [
            "Rewritten shot 1", "Rewritten shot 2", "Rewritten shot 3",
        ])
        self.assertEqual(
            [event_id for shot in shots for event_id in shot.metadata["source_event_ids"]],
            ["E1"],
        )
        self.assertEqual(shots[2].metadata["physical_state"]["opening"][-1]["value"], "seated")

    def test_two_invalid_compact_proposals_remain_a_failure(self):
        invalid = _compact_patch()
        invalid["event_shots"]["E1"] = 4
        planner = _SequencePlanner([invalid, invalid])
        rows = _three_shots()
        issues = ["Source actions must be allocated exactly once in source order"]

        with self.assertRaisesRegex(RuntimeError, "could not repair the source/physical contract"):
            planner._repair_h3_source_physical_plan(
                rows,
                _source_contract(),
                issues,
                planner_token_budget=4096,
                request_budget=2,
            )

        self.assertEqual(len(planner.calls), 2)

    def test_legacy_plan_without_source_ids_does_not_use_focused_patch(self):
        initial = _three_shots(source_ids=False, broken_state=True)
        repaired = _three_shots(source_ids=False, broken_state=False, title_prefix="Legacy")
        planner = _SequencePlanner([initial, repaired])

        shots, _ = _plan(planner, initial, manifest=[])

        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(planner.calls[1]["json_schema"]["type"], "array")
        self.assertTrue(all(shot.metadata["source_event_ids"] is None for shot in shots))


if __name__ == "__main__":
    unittest.main()
