"""Whole-song music arcs preserve story timing and visual variety."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.music_arc import (
    MAX_MUSIC_ARC_ROWS,
    arc_allocation,
    coerce_music_arc_rows,
    creative_plan_schema,
    filter_music_arc_context,
    format_music_arc_context,
    music_arc_quality_issues,
    music_arc_action_issues,
    music_arc_repetition_findings,
    music_arc_reprise_reason_findings,
    normalize_music_arc,
)
from services.director.music_narrative import build_music_narrative_contract
from services.director.planners.music_video import MusicVideoPlanner


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


_REMOTE_VERBS = (
    "studies", "opens", "reads", "marks", "photographs", "counts", "follows", "collects",
    "sketches", "tastes", "checks", "folds", "compares", "chooses", "packs", "unpacks",
    "notices", "records", "shares", "trades", "carries", "finds", "frames",
)
_REMOTE_OBJECTS = (
    "the paper map", "the glovebox", "a postcard", "the route", "the bridge", "the mile marker",
    "the river", "a seashell", "the skyline", "the local pie", "the fuel gauge", "a receipt",
    "the road sign", "a new playlist", "the camera", "the suitcase", "the harbor", "a voice memo",
    "the last coin", "a souvenir", "the raincoat", "the overlook", "the horizon",
)


def remote_arc_rows(count: int, *, one_based: bool = False, include_event_ids: bool = True) -> list[dict]:
    rows = []
    for index in range(count):
        row = arc_row(
            index,
            f"The traveler {_REMOTE_VERBS[index % len(_REMOTE_VERBS)]} "
            f"{_REMOTE_OBJECTS[index % len(_REMOTE_OBJECTS)]} during road trip moment {index + 1}.",
        )
        row.pop("clip_index")
        if not include_event_ids:
            row.pop("source_event_ids")
        if one_based:
            row["number"] = str(index + 1)
        else:
            row["index"] = index
        rows.append(row)
    return rows


class MusicArcTests(unittest.TestCase):
    def test_formatting_text_does_not_exhaust_source_event_budget(self):
        source = (
            "Create twelve beats running sequentially from 0 through 11. " * 20
            + "Show the sailor boarding the boat."
        )
        source_contract = build_music_narrative_contract(source)
        self.assertEqual(len(source_contract["events"]), 1)
        normalized = normalize_music_arc([
            arc_row(0, "The sailor boards the boat and settles beside the mast.",
                    source_event_ids=["M1"]),
        ], [{}], source_contract)
        self.assertEqual(normalized[0]["source_event_ids"], ["M1"])

    def test_real_actions_are_not_silently_truncated_to_the_outline_budget(self):
        source_contract = build_music_narrative_contract(" ".join(
            f"The courier opens numbered parcel {index}." for index in range(17)
        ))
        self.assertEqual(len(source_contract["events"]), 17)
        with self.assertRaisesRegex(ValueError, "exceeds 16 source events"):
            normalize_music_arc([
                arc_row(0, "The courier examines the numbered parcels."),
            ], [{}], source_contract)

    def test_remote_wrappers_and_numbered_rows_cover_fixed_15_and_23_group_timelines(self):
        for count in (15, 23):
            with self.subTest(count=count):
                numbered = remote_arc_rows(count, one_based=True)
                normalized = coerce_music_arc_rows([{"beats": numbered}], count)
                self.assertEqual(
                    [row["clip_index"] for row in normalized], list(range(count)),
                )
                self.assertEqual(
                    [row["visual_beat"] for row in normalized],
                    [row["visual_beat"] for row in numbered],
                )

    def test_complete_numeric_maps_and_singleton_ranges_resolve_their_base(self):
        for count in (15, 23):
            with self.subTest(count=count):
                rows = remote_arc_rows(count)
                numeric = {str(index + 1): row for index, row in enumerate(rows)}
                mapped = coerce_music_arc_rows({"groups": numeric}, count)
                self.assertEqual([row["clip_index"] for row in mapped], list(range(count)))

                ranged = {}
                for index, raw in enumerate(rows):
                    row = dict(raw)
                    row.pop("index")
                    ranged[f"{index + 1}-{index + 1}"] = row
                ranged_rows = coerce_music_arc_rows({"beats": ranged}, count)
                self.assertEqual([row["clip_index"] for row in ranged_rows], list(range(count)))

    def test_incomplete_duplicate_unrelated_and_expanded_ranges_remain_errors(self):
        rows = remote_arc_rows(15)
        with self.assertRaisesRegex(ValueError, "cover exactly 15.*received 14"):
            coerce_music_arc_rows({"beats": rows[:-1]}, 15)

        duplicate = remote_arc_rows(15, one_based=True)
        duplicate[-1]["number"] = duplicate[-2]["number"]
        with self.assertRaisesRegex(ValueError, "cover the complete timeline"):
            coerce_music_arc_rows(duplicate, 15)

        with self.assertRaisesRegex(ValueError, "unrelated fields"):
            coerce_music_arc_rows([{"beats": rows, "note": "discard?"}], 15)

        expanded = {"range": [1, 2], **rows[0]}
        expanded.pop("index")
        with self.assertRaisesRegex(ValueError, "one timeline group"):
            coerce_music_arc_rows([expanded] + rows[1:], 15)

    def test_visual_beat_aliases_normalize_only_unambiguous_row_values(self):
        valid_beat = "The traveler studies the folded map beside the roadside lantern."
        source_clip = [{}]

        for alias in ("beat", "prompt"):
            with self.subTest(alias=alias):
                row = arc_row(0, valid_beat)
                row[alias] = row.pop("visual_beat")
                normalized = normalize_music_arc([row], source_clip, contract())
                self.assertEqual(normalized[0]["visual_beat"], valid_beat)
                self.assertNotIn(alias, normalized[0])

        canonical = arc_row(0, valid_beat)
        self.assertEqual(
            normalize_music_arc([canonical], source_clip, contract())[0]["visual_beat"],
            valid_beat,
        )

        spaced = dict(canonical)
        spaced["visual_beat"] = f"  {valid_beat}  "
        spaced["beat"] = "  The traveler studies the folded map\n beside the roadside lantern.  "
        spaced["prompt"] = valid_beat
        self.assertEqual(coerce_music_arc_rows([spaced], 1)[0]["visual_beat"], f"  {valid_beat}  ")
        normalized = normalize_music_arc([spaced], source_clip, contract())
        self.assertEqual(normalized[0]["visual_beat"], valid_beat)

        alias_only = dict(canonical)
        alias_only.pop("visual_beat")
        alias_only["beat"] = "  The traveler studies the folded map\n beside the roadside lantern.  "
        self.assertEqual(
            normalize_music_arc([alias_only], source_clip, contract())[0]["visual_beat"],
            valid_beat,
        )

        alias_pair = dict(canonical)
        alias_pair.pop("visual_beat")
        alias_pair["beat"] = f"  {valid_beat}  "
        alias_pair["prompt"] = valid_beat
        self.assertEqual(coerce_music_arc_rows([alias_pair], 1)[0]["visual_beat"], f"  {valid_beat}  ")

    def test_visual_beat_alias_conflicts_malformed_values_and_unrelated_fields_fail(self):
        base = arc_row(0, "The traveler studies the folded map beside the roadside lantern.")

        conflicting_rows = []
        canonical_conflict = dict(base)
        canonical_conflict["beat"] = "The driver unfolds a paper map across the steering wheel."
        conflicting_rows.append(canonical_conflict)

        alias_conflict = dict(base)
        alias_conflict.pop("visual_beat")
        alias_conflict["beat"] = "The driver unfolds a paper map across the steering wheel."
        alias_conflict["prompt"] = "The passenger watches the coast beyond the window."
        conflicting_rows.append(alias_conflict)

        for row in conflicting_rows:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "conflicting visual_beat alias"):
                    normalize_music_arc([row], [{}], contract())

        malformed = dict(base)
        malformed["visual_beat"] = 7
        malformed["prompt"] = "The traveler studies the folded map beside the roadside lantern."
        with self.assertRaisesRegex(ValueError, "aliases must be text"):
            normalize_music_arc([malformed], [{}], contract())

        alias_only_malformed = dict(base)
        alias_only_malformed.pop("visual_beat")
        alias_only_malformed["prompt"] = None
        with self.assertRaisesRegex(ValueError, "visual_beat must be text"):
            normalize_music_arc([alias_only_malformed], [{}], contract())

        unrelated = dict(base, note="preserve me")
        coerced = coerce_music_arc_rows([unrelated], 1)
        self.assertEqual(coerced[0]["note"], "preserve me")
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            normalize_music_arc([unrelated], [{}], contract())

    def test_visual_beat_aliases_do_not_bypass_required_timeline_or_event_checks(self):
        rows = remote_arc_rows(2)
        for row in rows:
            row["beat"] = row.pop("visual_beat")
        with self.assertRaisesRegex(ValueError, "cover exactly 2.*received 1"):
            normalize_music_arc(rows[:1], [{}, {}], contract())

        missing = dict(rows[0])
        missing.pop("beat")
        with self.assertRaisesRegex(ValueError, "missing required fields.*visual_beat"):
            normalize_music_arc([missing], [{}], contract())

        event_rows = remote_arc_rows(2)
        for row in event_rows:
            row["prompt"] = row.pop("visual_beat")
        source_contract = contract(event_ids=("M1", "M2"))
        event_rows[0]["source_event_ids"] = ["M2"]
        event_rows[1]["source_event_ids"] = ["M1"]
        with self.assertRaisesRegex(ValueError, "exactly once and in source order"):
            normalize_music_arc(event_rows, [{}, {}], source_contract)

        generic = dict(arc_row(0, "Cinematic visuals that match the mood of the music."))
        generic["beat"] = generic.pop("visual_beat")
        with self.assertRaisesRegex(ValueError, "specific visible action"):
            normalize_music_arc([generic], [{}], contract())

    def test_mixed_number_bases_and_mixed_indexed_rows_cannot_shift_assignments(self):
        rows = remote_arc_rows(3, one_based=True)
        rows[0]["number"] = 1
        rows[1]["number"] = 3
        rows[2]["number"] = 2
        rows[2]["clip_index"] = 2
        with self.assertRaisesRegex(ValueError, "conflicting timeline index fields"):
            coerce_music_arc_rows(rows, 3)

        mixed = remote_arc_rows(3)
        mixed[1].pop("index")
        with self.assertRaisesRegex(ValueError, "mix indexed and unindexed"):
            coerce_music_arc_rows(mixed, 3)

    def test_remote_object_repair_is_bounded_and_keeps_fixed_timeline(self):
        for count in (15, 23):
            with self.subTest(count=count):
                calls = []
                complete = remote_arc_rows(count)

                def generate(**kwargs):
                    calls.append(kwargs)
                    response_rows = complete[:-1] if len(calls) == 1 else complete
                    return json.dumps({"beats": response_rows})

                planner = MusicVideoPlanner(llm_generate=generate)
                planner._music_narrative_contract = contract()
                planner._use_lyrics_context = False
                clips = [{"start": index, "end": index + 1, "label": "verse"} for index in range(count)]

                result = planner._plan_music_arc(clips, "A traveler crosses the country.")

                self.assertEqual(len(calls), 2)
                self.assertEqual([row["clip_index"] for row in result], list(range(count)))
                self.assertTrue(all("fixed and exhaustive" in call["prompt"] for call in calls))
                self.assertIn("previous outline could not be normalized", calls[1]["prompt"])
                self.assertEqual(calls[0]["json_schema"]["type"], "object")

    def test_truncated_and_repeated_incomplete_remote_objects_never_get_filled(self):
        count = 15
        rows = remote_arc_rows(count)
        truncated = json.dumps({"beats": rows[:-1]})[:-1]
        planner = MusicVideoPlanner()
        parsed = planner._parse_json_response(truncated)
        with self.assertRaisesRegex(ValueError, "cover exactly 15|missing required fields"):
            normalize_music_arc(parsed or [], [{} for _ in range(count)], contract())

        calls = []

        def generate(**_kwargs):
            calls.append(None)
            return json.dumps({"beats": rows[:-1]})

        planner = MusicVideoPlanner(llm_generate=generate)
        planner._music_narrative_contract = contract()
        planner._use_lyrics_context = False
        clips = [{"start": index, "end": index + 1, "label": "verse"} for index in range(count)]
        with self.assertRaisesRegex(RuntimeError, "needs revision"):
            planner._plan_music_arc(clips, "A traveler crosses the country.")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("music_creative_arc", planner._planning_resume_checkpoint)

    def test_remote_object_with_h3_assignments_keeps_zero_based_event_slot(self):
        count = 23
        rows = remote_arc_rows(count, one_based=True, include_event_ids=False)
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps({
                "beats": rows,
                "source_event_clips": {"M1": 5},
            })

        planner = MusicVideoPlanner(llm_generate=generate)
        planner._music_narrative_contract = contract(event_ids=("M1",))
        planner._use_lyrics_context = False
        clips = [{"start": index, "end": index + 1, "label": "chorus"} for index in range(count)]

        result = planner._plan_music_arc(clips, "A traveler crosses the country.")

        self.assertEqual(len(calls), 1)
        self.assertEqual([row["clip_index"] for row in result], list(range(count)))
        self.assertEqual(result[4]["source_event_ids"], [])
        self.assertEqual(result[5]["source_event_ids"], ["M1"])
        self.assertEqual(sum(row["source_event_ids"].count("M1") for row in result), 1)

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

    def test_context_filter_uses_canonical_allocation_for_unassigned_and_reversed_future_actions(self):
        source_contract = build_music_narrative_contract(
            "The kitten climbs on the dragon, then the dragon breathes fire over enemies."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 2},
            {"event_id": "M2", "clip_index": 4},
        ]
        rows = [
            arc_row(0, "The kitten climbs on the dragon."),
            arc_row(1, "The kitten watches lanterns drift above the road."),
            arc_row(
                2,
                "The dragon breathes fire over enemies, then the kitten climbs on the dragon.",
                source_event_ids=["M2"],
            ),
            arc_row(3, "The dragon banks over a valley as the clouds open."),
            arc_row(4, "The dragon breathes fire over enemies.", source_event_ids=["M1"]),
        ]

        filtered = filter_music_arc_context(rows, source_contract, allocation)

        self.assertEqual([row["clip_index"] for row in filtered], [1, 3, 4])
        self.assertIn("The dragon breathes fire", filtered[-1]["visual_beat"])
        self.assertEqual(filtered[-1]["source_event_ids"], ["M2"])
        context = format_music_arc_context(filtered, 4, 5, one_based=True)
        self.assertIn("source_event_ids=['M2']", context)
        self.assertNotIn("source_event_ids=['M1']", context)
        self.assertEqual(rows[-1]["source_event_ids"], ["M1"])

    def test_context_filter_preserves_inconclusive_and_unrelated_arc_rows(self):
        source_contract = build_music_narrative_contract(
            "The kitten climbs on the dragon, then the dragon breathes fire over enemies."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 2},
            {"event_id": "M2", "clip_index": 4},
        ]
        rows = [
            arc_row(
                2,
                "The kitten reaches toward the dragon and prepares to climb.",
                source_event_ids=["M1"],
            ),
            arc_row(3, "The dragon banks over a valley as the clouds open."),
        ]

        filtered = filter_music_arc_context(rows, source_contract, allocation)

        self.assertEqual([row["clip_index"] for row in filtered], [2, 3])

    def test_suppressed_repair_index_disappears_from_overview_and_is_deep_copied(self):
        rows = [
            arc_row(0, "A musician arrives alone at the empty station."),
            arc_row(
                1,
                "A missed train leaves a red scarf on the platform.",
                purpose="mark the missed connection",
                environment="the east platform beneath a clock",
                camera="tight insert on the red scarf",
            ),
            arc_row(2, "The musician follows the scarf through the morning crowd."),
        ]

        filtered = filter_music_arc_context(
            rows, contract(), [], suppressed_indices=1,
        )
        context = format_music_arc_context(filtered, 0, 3)
        filtered[1]["source_event_ids"].append("changed")
        filtered[1]["environment"] = "changed environment"

        self.assertEqual([row["clip_index"] for row in filtered], [0, 2])
        self.assertIn("0: A musician arrives alone", context)
        self.assertIn("2: The musician follows", context)
        self.assertNotIn("A missed train leaves a red scarf", context)
        self.assertNotIn("mark the missed connection", context)
        self.assertNotIn("the east platform beneath a clock", context)
        self.assertNotIn("tight insert on the red scarf", context)
        self.assertNotIn("clip 1:", context)
        self.assertEqual(rows[2]["source_event_ids"], [])
        self.assertEqual(rows[2]["environment"], "the same location")

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

    def test_intentional_refrain_reprise_reason_is_advisory(self):
        beat = "The vocalist closes their eyes and holds a sustained note as the chorus rises."
        literal_reprise = [
            arc_row(0, beat),
            arc_row(
                1, beat, reprise=True,
                reprise_reason="The final refrain returns to resolve the song's tension.",
            ),
        ]
        normalized_reprise = normalize_music_arc(literal_reprise, [{}, {}], contract())
        self.assertEqual(music_arc_repetition_findings(normalized_reprise), [])

        literal_reprise[1]["reprise_reason"] = ""
        normalized_missing_reason = normalize_music_arc(literal_reprise, [{}, {}], contract())
        self.assertEqual(len(music_arc_repetition_findings(normalized_missing_reason)), 1)
        self.assertEqual(len(music_arc_reprise_reason_findings(normalized_missing_reason)), 1)

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
        self.assertEqual(music_arc_repetition_findings(result), [])
        self.assertEqual(music_arc_reprise_reason_findings(result), [])

        rows[1]["reprise_reason"] = ""
        result = normalize_music_arc(rows, [{}, {}], contract())
        self.assertEqual(len(music_arc_reprise_reason_findings(result)), 1)
        self.assertEqual(music_arc_repetition_findings(result), [])

    def test_same_beat_with_distinct_camera_or_environment_is_not_a_duplicate(self):
        rows = [
            arc_row(0, "The vocalist sings toward the audience beneath crossing beams."),
            arc_row(
                1,
                "The vocalist sings toward the audience beneath crossing beams.",
                environment="an outdoor festival stage",
                camera="low close view from the crowd",
            ),
        ]
        normalized = normalize_music_arc(rows, [{}, {}], contract())
        self.assertEqual(music_arc_quality_issues(normalized, contract()), [])
        self.assertEqual(music_arc_repetition_findings(normalized), [])

    def test_literal_beat_and_matching_treatment_are_an_advisory_finding(self):
        beat = "The vocalist sings toward the audience beneath crossing beams."
        rows = [arc_row(0, beat), arc_row(1, beat)]
        normalized = normalize_music_arc(rows, [{}, {}], contract())
        findings = music_arc_repetition_findings(normalized)
        self.assertEqual(len(findings), 1)
        self.assertEqual(
            (findings[0]["clip_index"], findings[0]["prior_clip_index"]), (1, 0),
        )
        self.assertEqual(findings[0]["evidence"]["visual_beat"]["current"], beat)
        self.assertEqual(music_arc_quality_issues(normalized, contract()), [])

    def test_recurring_performance_with_different_visible_actions_is_allowed(self):
        rows = [
            arc_row(0, "The vocalist leans into the microphone and sings the opening verse."),
            arc_row(1, "The vocalist raises a hand as the chorus reaches its final note."),
        ]
        normalized = normalize_music_arc(rows, [{}, {}], contract())
        self.assertEqual(music_arc_repetition_findings(normalized), [])

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

    def test_detailed_writer_arc_uses_same_clip_numbers_as_source_schedule(self):
        rows = [arc_row(i, f"Unique visual beat for shot {i + 1}.") for i in range(6)]
        rows[4]["source_event_ids"] = ["M2"]
        original = copy.deepcopy(rows)
        context = format_music_arc_context(rows, 3, 6, one_based=True)
        assigned = context.split("ASSIGNED CLIPS FOR THIS BATCH", 1)[1]
        self.assertIn("global clips 4-6", assigned)
        self.assertIn("clip 4: visual_beat=Unique visual beat for shot 4.", assigned)
        self.assertIn("clip 5: visual_beat=Unique visual beat for shot 5.", assigned)
        self.assertIn("source_event_ids=['M2']", assigned)
        self.assertNotIn("clip 3:", assigned)
        self.assertNotIn("zero-based", context)
        self.assertEqual(rows, original)

    def test_arc_quality_note_only_appears_when_a_current_filtered_row_overlaps(self):
        rows = [arc_row(0, "A musician arrives alone at the empty station.")]
        notes = [{"clip_index": 1, "clip_end_index": 1, "message": "Develop this beat differently."}]
        context = format_music_arc_context(rows, 0, 2, quality_notes=notes)
        self.assertNotIn("WHOLE-SONG ARC QUALITY NOTES", context)

        notes[0].update(clip_index=0, clip_end_index=1)
        context = format_music_arc_context(rows, 0, 2, quality_notes=notes)
        self.assertIn("Develop this beat differently.", context)


if __name__ == "__main__":
    unittest.main()
