"""Exact start frames and imported scripts share one resolved scene contract."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.h3_dialogue_writing import fit_camera_dialogue
from services.h3_story_ledger import (
    _apply_faithful_treatment, _faithful_treatment_schema, extract_source_events,
)
from services.h3_window_planner import plan_h3_sliding_windows


class FramesAdaptationTests(unittest.TestCase):
    def test_faithful_treatment_schema_returns_required_options(self):
        plain = _faithful_treatment_schema(["Mara"])
        self.assertIsInstance(plain, dict)
        self.assertEqual(plain["properties"]["character_appearance"]["required"], ["Mara"])
        self.assertNotIn("source_adaptation", plain["properties"])
        self.assertNotIn("initial_state", plain["properties"])

        adapted = _faithful_treatment_schema(
            ["Mara"], resolve_adaptation=True, start_frame=True)
        self.assertEqual(next(iter(adapted["properties"])), "initial_state")
        self.assertIn("source_adaptation", adapted["properties"])
        self.assertEqual(adapted["required"], list(adapted["properties"]))

    def test_rejected_adapted_style_cannot_restore_superseded_source_descriptions(self):
        canonical = {"visual_continuity": "Lock two adult men for 30 seconds."}
        result = _apply_faithful_treatment(
            canonical, {"visual_continuity": "**Sequence progression**: a different scene."},
            resolve_adaptation=True,
        )
        self.assertEqual(result["visual_continuity"], "")
        self.assertIn("visual_continuity", result["_treatment_review_fields"])

    def test_silent_dialogue_review_cannot_change_authored_action_timing(self):
        segment = {"segment": 1, "shots": [
            {"shot": 1, "start_seconds": 0, "end_seconds": 4,
             "action": "A plants her foot, winds up, punches, and launches B into the wall.", "dialogue": []},
            {"shot": 2, "start_seconds": 4, "end_seconds": 7,
             "action": "Dust settles.", "dialogue": []},
        ]}
        before = deepcopy(segment)
        generate = Mock(side_effect=AssertionError("Silent action needs no dialogue call"))
        self.assertEqual(fit_camera_dialogue("Silent fight", segment, [], [],
                                           generate=generate, system_prompt=""), [])
        self.assertEqual(segment, before)
        generate.assert_not_called()

    def test_identity_reference_treatment_cannot_override_the_opening(self):
        canonical = {"initial_state": "The courier waits outside.", "source_intent": {"cast_names": ["Mara"]}}
        candidate = {"initial_state": "Mara is inside.", "source_adaptation": "Replace the courier."}
        unchanged = _apply_faithful_treatment(canonical, candidate)
        self.assertEqual(unchanged["initial_state"], canonical["initial_state"])
        self.assertNotIn("source_adaptation", unchanged)
        self.assertNotIn("initial_state", _faithful_treatment_schema(["Mara"])["properties"])
        resolved = _apply_faithful_treatment(canonical, candidate, resolve_adaptation=True, start_frame=True)
        self.assertEqual(resolved["initial_state"], candidate["initial_state"])
        self.assertEqual(resolved["source_adaptation"], candidate["source_adaptation"])

    def test_frames_treatment_owns_appearance_opening_and_camera_handoff(self):
        source = "Adapt the copied duel to the two women in the start image, lasting 28 seconds. " + (
            ROOT / "tests/fixtures/h3_silent_wuxia_prompt.txt"
        ).read_text(encoding="utf-8")
        events = extract_source_events(source)
        initial = ("Character A in ivory robes balances on her left leg with her right kick held "
                   "against Character B's block. B wears dark clothes with a red sash. "
                   "The camera sees both full bodies in the sunlit temple courtyard.")
        adaptation = ("Both roles are the adult women in the start image. A keeps ivory robes; "
                      "B keeps dark clothes and a red sash. Continue the blocked kick with a landing "
                      "before A's fist counter. Keep the source's attack ownership and ending.")
        recovery = ("From the pictured wide view, A retracts her blocked kick, plants her right foot, "
                    "and chambers her fist.")
        opening = ("B releases the block and starts to sidestep; A's punch "
                   "catches her upper body and launches her toward the mountain wall.")
        coverage = ("Track along the courtyard's southern edge, keeping the stone platform on screen left "
                    "and the waterfall cliff on screen right. Stay on that axis for the outgoing flight "
                    "and show the wall push-off before tracking the return leftward.")
        visual = (
            "Live-action movie-grade hyper-realism with a focus on aerodynamics and physical impact. "
            "Visual effects rely on transparent compressed wind rings, Mach rings, and white air-burst "
            "cones rather than colored energy or lasers. The camera style blends low-angle worship "
            "shots with sudden wide pull-backs to emphasize the scale of destruction. Fabric and hair "
            "react realistically to the immense wind pressure generated by the strikes. The stone "
            "platform and surrounding statues serve as the primary indicators of force, cracking and "
            "shattering under the weight of the combatants."
        )
        mechanics = "Both fighters travel from physical impacts and push-offs; their boots remain unlit and non-emitting."
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            props = kwargs["json_schema"]["properties"]
            if "character_appearance" in props:
                self.assertIn("initial_state", props)
                self.assertIn("source_adaptation", props)
                self.assertEqual(next(iter(props)), "initial_state")
                self.assertEqual(kwargs["image_paths"], ["start-frame.png"])
                return json.dumps({
                    "character_appearance": {"Character A": "The woman in ivory robes.",
                                             "Character B": "The woman in dark clothes and a red sash."},
                    "setting_continuity": "Sunlit temple courtyard with mountains visible beyond.",
                    "visual_continuity": visual,
                    "motion_mechanics": mechanics,
                    "editing_style": "Readable impact coverage.", "ambient_audio": "Mountain wind.",
                    "initial_state": initial, "source_adaptation": adaptation,
                })
            number = props["segment"]["minimum"]
            self.assertIn(adaptation, kwargs["prompt"])
            self.assertIn(mechanics, kwargs["prompt"])
            shared = kwargs["prompt"].split("Shared subjects:")[1].split("Active principal")[0]
            self.assertIn("ivory robes", shared)
            self.assertNotIn("earth-yellow", shared)
            self.assertIn("Exact frame continuity", kwargs["prompt"])
            if number == 1:
                self.assertIn("Required opening state: " + initial, kwargs["prompt"])
                self.assertIn("FIRST-FRAME ACTION", kwargs["prompt"])
                self.assertIn("arm-locked", kwargs["prompt"])  # Original first event remains available.
                self.assertEqual(kwargs["image_paths"], ["start-frame.png"])
                event_schema = props["event_cards"]["properties"]["event_1"]["properties"]
                self.assertIn("opening", event_schema)
                self.assertEqual(event_schema["phases"]["minItems"], 0)
                self.assertEqual(event_schema["opening"]["properties"]["transition"]["const"],
                                 "continue supplied frame")
            else:
                self.assertIsNone(kwargs["image_paths"])  # Later windows use the advancing state.
                self.assertIn("Previous camera geography (carry its landmarks and axis forward): " + coverage,
                              kwargs["prompt"])
            self.assertNotIn("Produce a full 30-second", kwargs["prompt"])
            result = {
                "segment": number, "title": "Duel", "coverage": coverage,
                "pacing": "Authored anticipation and fast impacts", "closing_state": "Both hold their final clash pose.",
                "event_cards": {f"event_{i + 1}": {"phases": [{
                    "transition": "continuous reframe" if number == 1 and i == 0 else "cut",
                    "framing": "Full view of both fighters", "camera": "Track the contact and its result.",
                    "action": events[(number - 1) * 4 + i]["text"].replace("gray", "ivory").replace("earth-yellow", "dark"),
                    "sound_effects": "Stone impacts and wind.",
                }]} for i in range(4)},
            }
            if number == 1:
                card = result["event_cards"]["event_1"]["phases"].pop()
                card["recovery"] = recovery
                card["action"] = opening
                card["transition"] = "continue supplied frame"
                card["framing"] = "The supplied frame's exact opening composition"
                result["event_cards"]["event_1"]["opening"] = card
            return json.dumps(result)

        with patch("services.llm_service.generate", side_effect=generate):
            result = plan_h3_sliding_windows(
                source, model_type="minimax_h3_fused_turbo", resolution="1280x704",
                total_frames=672, window_frames=345, overlap_frames=18,
                has_start_image=True, image_paths=["start-frame.png"], planning_style="adaptive",
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["planning_warnings"], [])
        self.assertEqual(result["source_prompt"], source.strip())
        self.assertEqual(result["story_ledger"]["initial_state"], initial)
        self.assertEqual(result["windows"][0]["opening_state"], initial)
        self.assertEqual(result["windows"][1]["opening_state"], result["windows"][0]["closing_state"])
        self.assertEqual([eid for beat in result["story_ledger"]["beats"] for eid in beat["source_event_ids"]],
                         [f"E{i}" for i in range(1, 9)])
        self.assertAlmostEqual(result["windows"][0]["shots"][0]["end_seconds"], 14.375 / 4, places=3)
        self.assertEqual(len(result["windows"][0]["shots"]), 4)
        opening_action = result["windows"][0]["shots"][0]["action"]
        self.assertIn(recovery + " " + opening, opening_action)
        self.assertNotIn("earth-yellow", result["subject_continuity"])
        self.assertEqual(result["story_ledger"]["visual_continuity"], visual + ". " + mechanics)
        for native in result["window_prompts"]:
            self.assertIn(visual, native)
            self.assertIn(mechanics, native)
            self.assertIn(coverage, native)
            self.assertNotIn("two adult Asian male", native)
            self.assertNotIn("Produce a full 30-second", native)


if __name__ == "__main__":
    unittest.main()
