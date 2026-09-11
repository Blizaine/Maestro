"""Production directions must not consume the screenplay speech budget."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service
from services.h3_authored_brief import authored_timed_brief, explicit_character_profiles
from services.h3_story_ledger import (
    H3DialogueTimingError,
    extract_h3_source_intent,
    extract_locked_dialogue,
    extract_source_events,
    plan_h3_story_segments,
)
from models.minimax_h3.ref2va import ensure_ref2va_prompt_relationships


class ProductionHeadingTests(unittest.TestCase):
    def test_reported_temple_brief_has_five_phases_and_no_spoken_words(self):
        source = (Path(__file__).parent / "fixtures/h3_silent_temple_prompt.txt").read_text(encoding="utf-8")
        for prompt in (source, " ".join(source.split())):
            with self.subTest(flattened="\n" not in prompt):
                self.assertEqual(extract_locked_dialogue(prompt), [])
                self.assertEqual(llm_service._extract_h3_source_dialogue_entries(prompt), [])
                brief = authored_timed_brief(prompt)
                self.assertEqual(len(brief["events"]), 5)
                self.assertEqual(
                    [(item["source_start_seconds"], item["source_end_seconds"]) for item in brief["events"]],
                    [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)],
                )
                self.assertIn("Rooftop Chase", brief["events"][-1]["source_title"])
                for event in brief["events"]:
                    self.assertEqual(prompt[event["source_offset"]:event["source_end"]].strip(), event["text"])
                self.assertIn("White-Clothed Martial Monk", brief["context"])
                self.assertIn("Ordinary punch: Ultra-short air compression flicker", brief["context"])
                self.assertEqual([p["name"] for p in explicit_character_profiles(prompt)], ["Role A", "Role B"])
                self.assertEqual(len(extract_source_events(prompt)), 5)
                intent = extract_h3_source_intent(prompt)
                self.assertEqual(intent["cast_names"], ["Role A", "Role B"])
                self.assertIn("no slow motion", intent["pacing_contract"])
                self.assertIn("Strictly prohibited: Slow motion", intent["negative_constraints"])

        def offline(**_kwargs):
            raise RuntimeError("No inference in this regression test")

        # Same 86-word spoken capacity as the reported error. All authored
        # directions can proceed even when they exceed this speech-only limit.
        result = plan_h3_story_segments(
            source, segment_durations=[14.375, 14.375], mode="reference_sequence",
            camera_coverage="continuous", expect_dialogue=False,
            planning_style="faithful", llm_generate=offline,
        )
        self.assertEqual(result["locked_dialogue"], [])
        self.assertEqual(len(result["segments"]), 2)
        self.assertFalse(any(shot.get("dialogue") for segment in result["segments"] for shot in segment["shots"]))

    def test_production_sections_preserve_real_speech_and_end_at_story_headings(self):
        prompt = (
            'Role A: White-Clothed Martial Monk.\nMira waits beside him.\n'
            '【Colorless Physical Effects System】\n'
            'Ordinary punch: Ultra-short air compression flicker.\n'
            'Role A: Stay behind me.\n'
            'Mira: I can help.\n'
            'Leo: "Watch out!"\n'
            '【Dialogue】\nGuard: Close the gate.\n'
            '[Shot 1]\nCourier: It is too late.\n'
            '【0.00—4.00｜The gate】\nRole A: Hold it open.\n'
            '【4.00—8.00｜The escape】\nRole A steps outside.'
        )
        self.assertEqual([line["text"] for line in extract_locked_dialogue(prompt)], [
            "Stay behind me.", "I can help.", "Watch out!", "Close the gate.",
            "It is too late.", "Hold it open.",
        ])

    def test_partial_role_profiles_do_not_hide_a_third_requested_character(self):
        prompt = (
            "Only three fighters appear in this scene.\n"
            "Role A: White-Clothed Martial Monk.\n"
            "Role B: Dark-Robed Martial Monk.\n"
            "Mira runs into the courtyard and approaches Role A."
        )
        self.assertIn("Mira", extract_h3_source_intent(prompt)["cast_names"])

    def test_compound_production_headings_remain_visual_with_or_without_quotes(self):
        for heading in (
            "Scene description", "Visual requirements", "Main prompt",
            "Camera treatment", "Animation style", "Lighting setup",
            "Character description", "Action choreography", "Final state",
        ):
            for opening, closing in (("", ""), ('"', '"'), ("“", "”")):
                prompt = f"{heading}: {opening}A cyclist descends the mountain road.{closing}"
                with self.subTest(heading=heading, quoted=bool(opening)):
                    self.assertEqual(extract_locked_dialogue(prompt), [])
                    self.assertEqual(llm_service._extract_h3_source_dialogue_entries(prompt), [])
                    self.assertNotIn("<d>", ensure_ref2va_prompt_relationships(prompt, []))
                    events = " ".join(item["text"] for item in extract_source_events(prompt))
                    self.assertIn("A cyclist descends the mountain road", events)

    def test_real_screenplay_and_attributed_lines_survive_production_headings(self):
        prompt = (
            'Visual requirements: "Natural light and clear movement."\n'
            'Mira (quietly): "We should leave now."\n'
            'Scene description: Leo replies, "I am ready."\n'
            'Character A: Follow the road.\n'
            'Final state: Both riders reach the valley.'
        )
        self.assertEqual(
            [line["text"] for line in extract_locked_dialogue(prompt)],
            ["We should leave now.", "I am ready.", "Follow the road."],
        )

    def test_visual_prose_can_exceed_the_selected_spoken_word_budget(self):
        prompt = (
            "Scene description: A cyclist descends a narrow mountain road, leans into "
            "the curve, and passes beneath an old stone bridge. The camera tracks "
            "alongside the wheels, rises to reveal the valley, and settles behind "
            "the rider as the road opens into a sunlit meadow.\n"
            "Visual requirements: Realistic light, clear geography, detailed rocks "
            "and fabrics, natural motion, and a continuous view of the cyclist."
        )

        def offline(**_kwargs):
            raise RuntimeError("No inference in this regression test")

        settings = dict(
            segment_durations=[5.0, 5.0], mode="reference_sequence",
            camera_coverage="multi_shot", expect_dialogue=False,
            planning_style="faithful", llm_generate=offline,
        )
        result = plan_h3_story_segments(prompt, **settings)
        self.assertEqual(result["locked_dialogue"], [])
        self.assertFalse(any(
            shot.get("dialogue")
            for segment in result["segments"] for shot in segment["shots"]
        ))
        # The same number of actual user-written spoken words must still be
        # rejected when they cannot fit; this is not a duration-check bypass.
        with self.assertRaises(H3DialogueTimingError):
            plan_h3_story_segments("Mira: " + "Stay with me. " * 20, **settings)


if __name__ == "__main__":
    unittest.main()
