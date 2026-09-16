"""Imported production notes and numbered shots retain their authored meaning."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.adaptive_enhancement import adaptive_dialogue_expected, adaptive_dialogue_expansion_requested
from services.dialogue_writing import conversation_brief, creative_dialogue_budget
from services.h3_authored_brief import authored_timed_brief, explicit_character_profiles
from services.h3_story_ledger import extract_h3_source_intent, extract_locked_dialogue, extract_source_events
from services.h3_window_planner import plan_h3_sliding_windows


class ImportedActionBriefTests(unittest.TestCase):
    def test_prohibited_dialogue_mentions_do_not_request_speech(self):
        for negative in (
            "【Complete Negative Prompt】 Prohibited: logos, banter, excessive dialogue, stutter frames.",
            "## Negative prompt\nText, chit-chat, excessive dialogue, frozen action.",
            "Prohibited: subtitles, non-combat segments (banter, chit-chat), excessive dialogue.",
            "Negative prompt: slow movement, extended speech.",
            "Avoid excessive dialogue and banter.",
            "No excessive dialogue, banter, or extended speeches.",
        ):
            with self.subTest(negative=negative):
                source = "A swordsman fights a horde in a storm.\n" + negative
                self.assertFalse(adaptive_dialogue_expected(source))
                self.assertFalse(adaptive_dialogue_expansion_requested(source))
                self.assertFalse(conversation_brief(source))
                self.assertIsNone(creative_dialogue_budget(source, 14.4))
                self.assertEqual(extract_locked_dialogue(source), [])

    def test_restrictions_do_not_suppress_actual_requests_or_exact_lines(self):
        for source in (
            "Negative prompt: excessive dialogue. Nora asks Eli for directions.",
            "Avoid long monologues, but write a short conversation between Nora and Eli.",
            "## Negative prompt\nBanter, long dialogue.\n## Story\nNora and Eli discuss the route.",
            'Nora says, "We have arrived." 【Negative Prompt】 Excessive dialogue, banter.',
        ):
            with self.subTest(source=source):
                self.assertTrue(adaptive_dialogue_expected(source))
        exact = 'Nora says, "We have arrived." 【Negative Prompt】 Excessive dialogue, banter.'
        self.assertEqual([d["text"] for d in extract_locked_dialogue(exact)], ["We have arrived."])
        self.assertFalse(adaptive_dialogue_expansion_requested(exact))

    def test_numbered_shots_outrank_repeated_chapter_and_summary_clocks(self):
        source = (
            "【Global Characters and Effects】 "
            "Traveler (Red-cloaked courier): carries a brass key. "
            "Guards (A masked crowd): surround the gate. "
            "【Overview】 TRACK through the gate (t0-4s), then ORBIT the courtyard (t4-8s). "
            "【Complete Shot-by-Shot Breakdown】 "
            "Phase A · Escape (t0-4s) · "
            "Shot 1 (t0-2s) MS+WHIP whip-pan: Traveler opens the gate with the key. · "
            "Shot 2 (t2-4s) CU+PUSH: Traveler passes through. Rear guards turn to follow. "
            "Phase B · Arrival (t4-8s) · "
            "Shot 3 (t4-6s) MS: Traveler closes the gate. · "
            "Shot 4 (t6-8s) CU: Traveler pockets the key. "
            "【Complete Negative Prompt】 Prohibited: banter, excessive dialogue."
        )
        for prompt in (source, source.replace(" · Shot", "\nShot"), " ".join(source.split())):
            with self.subTest(prompt=prompt):
                brief = authored_timed_brief(prompt)
                self.assertEqual([(e["source_start_seconds"], e["source_end_seconds"]) for e in brief["events"]],
                                 [(0, 2), (2, 4), (4, 6), (6, 8)])
                self.assertEqual(len(extract_source_events(prompt)), 4)
                self.assertNotIn("Phase B", brief["events"][1]["text"])
                self.assertIn("Prohibited", brief["context"])
                for event in brief["events"]:
                    self.assertEqual(prompt[event["source_offset"]:event["source_end"]].strip(" ·*\\\r\n"), event["text"])
                intent = extract_h3_source_intent(prompt)
                self.assertEqual(intent["cast_names"], ["Traveler", "Guards"])
                self.assertIn("requested multiplicity for Guards", intent["cast_cardinality_contract"])
                self.assertEqual(extract_locked_dialogue(prompt), [])

    def test_flattened_profiles_do_not_include_effects_or_quantities(self):
        source = ("【Global Characters and Effects】 Hero (A cloaked swordsman): dark hair. "
                  "Monster (Shadow horde): many shadows. Opening: 80. Effects Density: rain 60%. "
                  "【Shots】 Shot 1 (t0-2s) Hero attacks. Shot 2 (t2-4s) Monsters turn away.")
        self.assertEqual([p["name"] for p in explicit_character_profiles(source)], ["Hero", "Monster"])
        intent = extract_h3_source_intent(source)
        self.assertEqual(intent["cast_names"], ["Hero", "Monster"])
        self.assertIn("requested multiplicity for Monster", intent["cast_cardinality_contract"])

    def test_overlapping_numbered_shots_are_not_silently_retimed(self):
        source = "Shot 1 (t0-4s) Nora enters. Shot 2 (t2-6s) Eli waves."
        self.assertEqual(authored_timed_brief(source)["events"], [])

    def test_negative_pacing_lists_cannot_request_slow_motion(self):
        for restriction in (
            "No frame pauses/slow motion/freeze frames/bullet time whatsoever.",
            "【Negative Prompt】 Prohibited: charges over 0.5s, slow motion, time dilation.",
            "No long dialogue or slow motion.",
        ):
            source = "Extremely fast combat with explosive acceleration. " + restriction
            pacing = extract_h3_source_intent(source)["pacing_contract"]
            self.assertIn("no slow motion", pacing)
            self.assertNotIn("requested slow-motion", pacing)
        positive = "No dialogue. Use brief slow motion before impact, then explode forward at full speed."
        self.assertIn("slow-motion", extract_h3_source_intent(positive)["pacing_contract"])

    def test_a_named_team_leader_is_not_a_group(self):
        source = "【Characters】 Captain (A team leader): blue coat. 【Story】 Captain walks through the gate."
        contract = extract_h3_source_intent(source)["cast_cardinality_contract"]
        self.assertIn("one identity instance", contract)
        self.assertNotIn("requested multiplicity", contract)

    def test_dense_timed_shots_keep_separate_camera_cards_and_middle_actions(self):
        actions = [
            "Nora unlocks the courtyard gate with a brass key.",
            "Nora crosses the threshold and closes the gate behind her.",
            "Nora ducks beneath a falling branch.",
            "Nora vaults over a stone bench.",
            "Nora catches a tumbling lantern before it strikes the ground.",
            "Nora places the lantern on the fountain ledge.",
            "Nora unwinds a rope from a wooden post.",
            "Nora throws the rope over an overhead beam.",
            "Nora climbs the rope to the balcony.",
            "Nora opens the balcony shutters.",
            "Nora enters the room and retrieves a folded map from the desk.",
            "Nora unfolds the map beneath the window light.",
        ]
        times = [(i, i + 1) for i in range(10)] + [(10, 15), (15, 20)]
        source = "Nora wears a red coat. No dialogue.\n" + "\n".join(
            f"Shot {i} (t{start}-{end}s) {action}"
            for i, ((start, end), action) in enumerate(zip(times, actions), 1)
        )
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            props = (kwargs.get("json_schema") or {}).get("properties", {})
            if "event_cards" not in props:
                return json.dumps({
                    "character_appearance": {"Nora": "A courier in a red coat."},
                    "setting_continuity": "A courtyard beneath a balcony and its map room.",
                    "visual_continuity": "Daylight and natural materials.",
                    "editing_style": "Continuous progress with motivated cuts.",
                    "ambient_audio": "Footsteps and courtyard wind.",
                })
            number = props["segment"]["minimum"]
            assigned = actions[:10] if number == 1 else actions[10:]
            self.assertEqual(props["event_cards"]["required"],
                             [f"event_{i}" for i in range(1, len(assigned) + 1)])
            return json.dumps({
                "segment": number, "title": "The courier's route",
                "coverage": "Follow Nora from the gate through the courtyard to the balcony room.",
                "pacing": "Purposeful movement", "closing_state": assigned[-1],
                "event_cards": {f"event_{i}": {"phases": [{
                    "action": action, "framing": "Medium view of Nora and her surroundings",
                    "camera": "Track the current movement and its destination.",
                    "transition": "cut", "sound_effects": "Footsteps and cloth movement.",
                }]} for i, action in enumerate(assigned, 1)},
            })

        for style in ("faithful", "adaptive"):
            with self.subTest(style=style):
                calls.clear()
                with patch("services.llm_service.generate", side_effect=generate):
                    result = plan_h3_sliding_windows(
                        source, model_type="minimax_h3_fused_turbo", resolution="1280x704",
                        total_frames=480, window_frames=240, overlap_frames=0, fps=24,
                        planning_style=style,
                    )
                self.assertEqual(result["planning_warnings"], [])
                self.assertEqual(len(calls), 3)
                self.assertEqual([len(w["shots"]) for w in result["windows"]], [10, 2])
                for index, assigned in enumerate((actions[:10], actions[10:])):
                    for action in assigned:
                        self.assertIn(action, result["window_prompts"][index])
                self.assertNotIn("<d>", " ".join(result["window_prompts"]))


if __name__ == "__main__":
    unittest.main()
