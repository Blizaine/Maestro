"""Whole-song music arcs preserve story timing and visual variety."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.music_arc import (
    MAX_MUSIC_ARC_ROWS,
    arc_allocation,
    creative_plan_schema,
    format_music_arc_context,
    music_arc_quality_issues,
    music_arc_action_issues,
    normalize_music_arc,
)
from services.director.music_narrative import build_music_narrative_contract


def arc_row(
    index: int,
    beat: str,
    *,
    purpose: str = "advance the visual progression",
    environment: str = "the same location",
    camera: str = "locked medium-wide frame",
    source_event_ids: list[str] | None = None,
    reprise: bool = False,
    reprise_reason: str = "",
) -> dict:
    return {
        "clip_index": index,
        "visual_beat": beat,
        "purpose": purpose,
        "environment": environment,
        "camera": camera,
        "source_event_ids": source_event_ids or [],
        "intentional_reprise": reprise,
        "reprise_reason": reprise_reason,
    }


def contract(source: str = "A performer makes a journey through changing places.", event_ids=()) -> dict:
    return {
        "source": source,
        "events": [{"event_id": event_id, "text": f"source action {event_id}"} for event_id in event_ids],
    }


class MusicArcTests(unittest.TestCase):
    def test_action_ids_cannot_assign_a_run_to_a_standing_beat(self):
        source_contract = build_music_narrative_contract(
            "The traveler runs across the field, then climbs onto a horse."
        )
        rows = [
            arc_row(0, "The traveler stands alone in the field and sings.", source_event_ids=["M1"]),
            arc_row(1, "The traveler runs across the field toward the waiting horse."),
            arc_row(2, "The traveler climbs onto the horse and settles onto its back.", source_event_ids=["M2"]),
        ]
        issues = music_arc_action_issues(rows, source_contract)
        self.assertTrue(any("M1" in issue and "Outline action mismatch" in issue for issue in issues))
        rows[0]["source_event_ids"] = []
        rows[1]["source_event_ids"] = ["M1"]
        self.assertEqual(music_arc_action_issues(rows, source_contract), [])

    def test_outline_must_show_boarding_in_its_assigned_beat(self):
        source_contract = build_music_narrative_contract("The traveler climbs onto a horse.")
        rows = [
            arc_row(0, "The traveler reaches toward the horse and prepares to jump.", source_event_ids=["M1"]),
            arc_row(1, "The traveler climbs onto the horse and settles onto its back."),
        ]
        self.assertTrue(music_arc_action_issues(rows, source_contract))
        rows[0]["source_event_ids"] = []
        rows[1]["source_event_ids"] = ["M1"]
        self.assertEqual(music_arc_action_issues(rows, source_contract), [])

    def test_outline_action_check_leaves_non_narrative_concepts_alone(self):
        rows = [arc_row(0, "Thin ink threads disperse into rolling clouds.")]
        self.assertEqual(music_arc_action_issues(rows, {"events": []}), [])

    def test_schema_is_a_strict_bounded_array_of_arc_rows(self):
        schema = creative_plan_schema()
        self.assertEqual(schema["type"], "array")
        self.assertEqual(schema["maxItems"], MAX_MUSIC_ARC_ROWS)
        row_schema = schema["items"]
        self.assertFalse(row_schema["additionalProperties"])
        self.assertEqual(
            set(row_schema["required"]),
            {
                "clip_index", "visual_beat", "purpose", "environment", "camera",
                "source_event_ids", "intentional_reprise", "reprise_reason",
            },
        )

    def test_varied_genres_concepts_and_fixed_camera_are_supported(self):
        scenarios = {
            "road trip": [
                arc_row(0, "The driver unfolds a paper map across the steering wheel."),
                arc_row(1, "The traveler watches the coast appear beyond the passenger window."),
                arc_row(2, "The navigator circles a new destination on the motel notepad."),
            ],
            "cooking": [
                arc_row(0, "The cook cracks pepper into a sizzling pan."),
                arc_row(1, "The cook folds bright herbs through the sauce."),
            ],
            "stage performance, fixed camera": [
                arc_row(0, "The vocalist leans toward the microphone as the verse begins."),
                arc_row(1, "The vocalist lifts one hand while the chorus opens."),
            ],
            "abstract visualizer": [
                arc_row(0, "Thin amber lines gather around each kick drum."),
                arc_row(1, "The lines break into scattered points during the soft bridge."),
            ],
        }
        for label, rows in scenarios.items():
            with self.subTest(label=label):
                result = normalize_music_arc(rows, [{} for _ in rows], contract())
                self.assertEqual([row["clip_index"] for row in result], list(range(len(rows))))
                self.assertEqual(music_arc_quality_issues(result, contract()), [])

    def test_intentional_refrain_reprise_requires_and_accepts_a_reason(self):
        beat = "The vocalist closes their eyes and holds a sustained note as the chorus rises."
        rows = [
            arc_row(0, beat, environment="bare rehearsal stage"),
            arc_row(
                1,
                beat,
                environment="full concert stage",
                camera="fixed front camera with the whole band visible",
                reprise=True,
                reprise_reason="The refrain returns after the bridge as the song reaches its peak.",
            ),
        ]
        result = normalize_music_arc(rows, [{}, {}], contract())
        self.assertTrue(result[1]["intentional_reprise"])
        self.assertEqual(music_arc_quality_issues(result, contract()), [])

        rows[1]["reprise_reason"] = ""
        with self.assertRaisesRegex(ValueError, "reprise needs a concise reason"):
            normalize_music_arc(rows, [{}, {}], contract())

    def test_duplicate_beats_are_detected_even_when_camera_and_environment_change(self):
        rows = [
            arc_row(0, "The vocalist sings toward the audience beneath crossing beams."),
            arc_row(
                1,
                "The vocalist sings toward the audience beneath crossing beams.",
                environment="an outdoor festival stage",
                camera="low close view from the crowd",
            ),
        ]
        issues = music_arc_quality_issues(rows, contract())
        self.assertTrue(any("repeats the visual beat" in issue for issue in issues))
        with self.assertRaisesRegex(ValueError, "repeats the visual beat"):
            normalize_music_arc(rows, [{}, {}], contract())

    def test_repeated_whole_brief_and_generic_filler_are_not_visual_beats(self):
        source = "A singer crosses the city at night, meets a friend, and watches the sunrise together."
        copied = [arc_row(0, source)]
        with self.assertRaisesRegex(ValueError, "full source brief"):
            normalize_music_arc(copied, [{}], contract(source))

        filler = [arc_row(0, "Cinematic visuals that match the mood of the music.")]
        with self.assertRaisesRegex(ValueError, "specific visible action"):
            normalize_music_arc(filler, [{}], contract())

    def test_camera_adjectives_do_not_hide_concrete_visible_beats(self):
        for beat in (
            "Wide, sweeping cinematic shot of the dragon soaring high above the battlefield, "
            "the kitten visible in sharp focus on its back. The cape is fully extended in the wind.",
            "A cinematic shot of the cook folding bright herbs through the sauce.",
            "Beautiful imagery of black ink branching through clear water.",
        ):
            with self.subTest(beat=beat):
                rows = [arc_row(0, beat)]
                self.assertEqual(music_arc_quality_issues(rows, contract()), [])

        for beat in (
            "Various cinematic shots that match the music.",
            "Beautiful imagery supporting the mood of the song.",
            "Cinematic visuals that reflect the song mood with a sweeping camera move.",
            "Various cinematic shots, dynamic changes, and atmospheric imagery.",
        ):
            with self.subTest(beat=beat):
                issues = music_arc_quality_issues([arc_row(0, beat)], contract())
                self.assertTrue(any("specific visible action" in issue for issue in issues))

    def test_source_events_keep_contract_order_and_are_allocated_once(self):
        source_contract = contract(event_ids=("M1", "M2", "M3"))
        rows = [
            arc_row(2, "The traveler reaches the overlook and raises a paper flag.", source_event_ids=["M3"]),
            arc_row(0, "The driver circles a route on the folded paper map.", source_event_ids=["M1"]),
            arc_row(1, "The passenger leaves a postcard beside the open suitcase.", source_event_ids=["M2"]),
        ]
        normalized = normalize_music_arc(rows, [{}, {}, {}], source_contract)
        self.assertEqual(arc_allocation(normalized), [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 1},
            {"event_id": "M3", "clip_index": 2},
        ])

        rows[2]["source_event_ids"] = ["M1"]
        with self.assertRaisesRegex(ValueError, "exactly once and in source order"):
            normalize_music_arc(rows, [{}, {}, {}], source_contract)

    def test_expanded_long_timeline_can_reuse_group_beats_and_stays_bounded_in_context(self):
        source_contract = contract(event_ids=("M1",))
        group = arc_row(
            0,
            "The traveler follows the shoreline toward a distant lighthouse.",
            source_event_ids=["M1"],
            purpose="develop the travel section",
        )
        expanded = []
        for index in range(60):
            row = dict(group)
            row["clip_index"] = index
            if index:
                row["source_event_ids"] = []
            expanded.append(row)

        normalized = normalize_music_arc(expanded, [{} for _ in expanded], source_contract)
        self.assertEqual(len(normalized), 60)
        self.assertEqual(arc_allocation(normalized), [{"event_id": "M1", "clip_index": 0}])
        self.assertEqual(music_arc_quality_issues(normalized, source_contract), [])
        context = format_music_arc_context(normalized, 57, 60)
        self.assertIn("0-59: The traveler follows", context)
        self.assertIn("clip 57:", context)
        self.assertIn("clip 59:", context)

    def test_four_source_events_in_groups_zero_through_three_of_twenty_three_are_flagged(self):
        ids = ("M1", "M2", "M3", "M4")
        source_contract = contract(event_ids=ids)
        verbs = [
            "studies", "opens", "reads", "marks", "photographs", "counts", "follows", "collects",
            "sketches", "tastes", "checks", "folds", "compares", "chooses", "packs", "unpacks",
            "notices", "records", "shares", "trades", "carries", "finds", "frames",
        ]
        objects = [
            "the paper map", "the glovebox", "a postcard", "the route", "the bridge", "the mile marker",
            "the river", "a seashell", "the skyline", "the local pie", "the fuel gauge", "a receipt",
            "the road sign", "a new playlist", "the camera", "the suitcase", "the harbor", "a voice memo",
            "the last coin", "a souvenir", "the raincoat", "the overlook", "the horizon",
        ]
        rows = [
            arc_row(
                index,
                f"The traveler {verbs[index]} {objects[index]} during a changing road trip moment {index}.",
                source_event_ids=[ids[index]] if index < 4 else [],
            )
            for index in range(23)
        ]
        issues = music_arc_quality_issues(rows, source_contract)
        self.assertTrue(any(issue.startswith("Repairable planning issue:") for issue in issues))
        self.assertFalse(any("exactly once" in issue for issue in issues))
        self.assertFalse(any("repeats the visual beat" in issue for issue in issues))

        self.assertEqual(music_arc_quality_issues(rows, source_contract, allow_early_events=True), [])

    def test_arc_context_includes_full_overview_and_only_current_assignments(self):
        rows = [
            arc_row(0, "A musician arrives alone at the empty station."),
            arc_row(1, "A missed train leaves a red scarf on the platform."),
            arc_row(2, "The musician follows the scarf through the morning crowd."),
        ]
        context = format_music_arc_context(rows, 1, 2)
        self.assertIn("WHOLE-SONG MUSIC VIDEO ARC", context)
        self.assertIn("0: A musician arrives alone", context)
        self.assertIn("2: The musician follows", context)
        self.assertIn("ASSIGNED GROUPS FOR THIS BATCH [zero-based 1, 2)", context)
        self.assertIn("clip 1:", context)
        self.assertNotIn("clip 0:", context)
        self.assertIn("Use source_event_ids exactly as assigned", context)


if __name__ == "__main__":
    unittest.main()
