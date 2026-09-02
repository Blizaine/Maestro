"""Long-form Director planning regressions."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.director.planners.music_video import MusicVideoPlanner  # noqa: E402
from services.director.planners.podcast import PodcastPlanner  # noqa: E402
from services.director.planners.short_film import (  # noqa: E402
    ShortFilmPlanner,
    _build_h3_budgeted_screenplay_fallback,
    _expand_h3_dialogue_coverage_slots,
    _h3_screenplay_budget_issues,
    _h3_screenplay_recovery_reasons,
    _normalize_long_form_dialogue_ownership,
    _normalize_long_form_dialogue_targets,
    _normalize_long_form_event_ownership,
    _normalize_long_form_plan_references,
    _prepare_long_form_ltx_prompt_contract,
    _restore_h3_dialogue_after_pacing_repair,
    _restore_missing_h3_screenplay_dialogue,
)
from services.director.long_form_story import (  # noqa: E402
    apply_long_form_recurring_motifs,
    build_long_form_story_bible_fallback,
    compact_long_form_h3_prompt,
    ensure_long_form_location_coverage,
    long_form_outline_quality_issues,
    long_form_story_bible_quality_issues,
    normalize_long_form_outline,
    normalize_long_form_sequence_states,
    normalize_long_form_story_bible,
    sanitize_long_form_shot_dicts,
)
from services.director.schema import AudioPlan, CameraPlan, ShotPlan  # noqa: E402
from services.h3_story_ledger import plan_h3_story_segments  # noqa: E402


def _shot(duration: float = 10.0) -> ShotPlan:
    return ShotPlan(
        shot_id="local",
        index=0,
        duration_sec=duration,
        skill_type="short_film",
        scene_goal="Advance the chapter",
        subjects_on_screen=[],
        spatial_setup="Readable composition",
        environment="Established location",
        visual_style="Cinematic",
        lighting="Natural",
        mood="Focused",
        action_beats=["A new event occurs"],
        camera_plan=CameraPlan(framing="medium shot"),
        audio_plan=AudioPlan(mode="ambient_only"),
        ending_beat="A visible handoff",
    )


def _architect_response(**kwargs) -> str:
    """Return valid chapter/sequence JSON for the requested exact count."""

    prompt = str(kwargs.get("prompt") or "")
    match = re.search(r"exactly\s+(\d+)\s+causal", prompt, re.IGNORECASE)
    count = int(match.group(1)) if match else 1
    is_chapter = "chapters for one" in prompt
    noun = "chapter" if is_chapter else "sequence"
    rows = [
        {
            noun: index + 1,
            "title": f"{noun.title()} {index + 1}",
            "location_time": f"Location {index + 1}, continuous story time",
            "objective": f"Advance beat {index + 1}",
            "opening_state": f"Opening {index + 1}",
            "closing_state": f"Closing {index + 1}",
            "causal_handoff": "The prior consequence carries forward",
            "persistent_state": "Identity and story state remain stable",
            "dialogue_ids": [],
            "source_event_ids": [],
            "dialogue_mode": "visual",
            "dialogue_target_turns": 0,
            "dialogue_target_words": 0,
        }
        for index in range(count)
    ]
    return json.dumps(rows)


class LongFormDirectorPlanningTests(unittest.TestCase):
    def test_repeated_sitcom_premise_becomes_an_evolving_recurring_motif(self):
        story = (
            'A sitcom starring Thanos. He gets drunk and goes around to different '
            'iconic TV shows, enters the room with the characters, says '
            '"Perfectly balanced bitches!", and snaps so half the people in '
            "whatever room he is in turn to dust. He visits Friends, The Office, "
            "Parks and Rec, and many more."
        )
        dialogue = [{"dialogue_id": "D1", "text": "Perfectly balanced bitches!"}]
        events = [
            {"event_id": "E1", "text": "A sitcom starring Thanos"},
            {"event_id": "E2", "text": "He goes around to different iconic TV shows"},
            {"event_id": "E3", "text": "He says [D1: exact dialogue assigned by Maestro]"},
            {"event_id": "E4", "text": "He snaps and half the room turns to dust"},
            {"event_id": "E5", "text": "He visits named shows and many more"},
        ]
        bible = build_long_form_story_bible_fallback(
            story,
            locked_dialogue=dialogue,
            source_events=events,
            character_names=["Thanos"],
            chapter_count=12,
        )
        rows = [
            {
                "chapter": index + 1,
                "title": f"Visit {index + 1}",
                "location_time": f"Distinct sitcom set {index + 1}",
                "objective": "Stage a new escalating encounter",
                "opening_state": "Thanos arrives",
                "closing_state": "The room changes",
                "causal_handoff": "The consequence sends him onward",
                "persistent_state": "Thanos remembers prior reactions",
                "dialogue_ids": [],
                "source_event_ids": [],
                "recurring_motif_ids": [],
            }
            for index in range(12)
        ]
        planned = apply_long_form_recurring_motifs(rows, story_bible=bible)

        motif_rows = [row for row in planned if "M1" in row["recurring_motif_ids"]]
        self.assertEqual(len(motif_rows), 10)
        self.assertTrue(all("D1" in row["dialogue_ids"] for row in motif_rows))
        self.assertTrue(all("E4" in row["source_event_ids"] for row in motif_rows))
        self.assertTrue(all(row.get("motif_variation_contract") for row in motif_rows))

    def test_abstract_location_flythrough_does_not_invent_a_recurring_motif(self):
        bible = build_long_form_story_bible_fallback(
            (
                "A high speed supersonic fly through of different locations, "
                "such as a beach, castle, snowy mountains, and the moon."
            ),
            locked_dialogue=[],
            source_events=[],
            character_names=[],
            chapter_count=8,
        )
        self.assertEqual(bible["recurring_motifs"], [])
        self.assertTrue(bible["allow_location_expansion"])
        self.assertFalse(bible["allow_cast_expansion"])

    def test_expanding_story_bible_requires_enough_macro_variety(self):
        bible = build_long_form_story_bible_fallback(
            "A visitor goes around to different sitcom shows and many more.",
            locked_dialogue=[],
            source_events=[],
            character_names=["Visitor"],
            chapter_count=12,
        )
        issues = long_form_story_bible_quality_issues(
            bible,
            chapter_count=12,
        )
        self.assertTrue(any("location registry" in issue for issue in issues))
        self.assertTrue(any("cast registry" in issue for issue in issues))

    def test_collapsed_long_form_outline_is_detected_before_sequence_writing(self):
        bible = {
            "allow_location_expansion": True,
            "canonical_locations": [
                {"location_id": f"set_{index}"} for index in range(8)
            ],
        }
        rows = [
            {
                "location_id": "set_1",
                "objective": "Repeat the same encounter",
            }
            for _index in range(12)
        ]
        issues = long_form_outline_quality_issues(
            rows,
            story_bible=bible,
            chapter_count=12,
        )
        self.assertTrue(any("objectives" in issue for issue in issues))
        self.assertTrue(any("locations" in issue for issue in issues))

    def test_failed_ai_variety_repair_uses_only_approved_bible_locations(self):
        bible = {
            "allow_location_expansion": True,
            "canonical_locations": [
                {
                    "location_id": f"set_{index}",
                    "name": f"Approved Set {index}",
                }
                for index in range(1, 7)
            ],
        }
        rows, warnings = ensure_long_form_location_coverage(
            [
                {
                    "chapter": index + 1,
                    "location_id": "set_1",
                    "location_time": "Approved Set 1",
                    "opening_state": "Continue from the prior consequence",
                }
                for index in range(6)
            ],
            story_bible=bible,
        )
        represented = {row["location_id"] for row in rows}
        self.assertGreaterEqual(len(represented), 4)
        self.assertTrue(represented.issubset({f"set_{index}" for index in range(1, 7)}))
        self.assertTrue(warnings)

    def test_named_character_state_persists_until_explicit_restoration(self):
        bible = {
            "allow_cast_expansion": False,
            "allow_location_expansion": False,
            "canonical_characters": [
                {"name": "Thanos"},
                {"name": "Monica"},
            ],
            "canonical_locations": [],
        }
        outline, _updated = normalize_long_form_outline(
            [
                {
                    "objective": "Thanos snaps while Monica watches",
                    "opening_state": "Both stand in the apartment",
                    "closing_state": "Monica turns to dust",
                    "persistent_state": "Dust covers the couch",
                    "cast_present": ["Thanos", "Monica"],
                    "character_state_changes": ["Monica: turns to dust"],
                },
                {
                    "objective": "Thanos leaves the survivors behind",
                    "opening_state": "The room is dusty",
                    "closing_state": "Thanos reaches the door",
                    "persistent_state": "Monica remains absent",
                    "cast_present": ["Thanos", "Monica"],
                    "character_state_changes": [],
                },
                {
                    "objective": "Monica reappears after restoration",
                    "opening_state": "A reversal restores Monica",
                    "closing_state": "Monica confronts Thanos",
                    "persistent_state": "Monica is physically present again",
                    "cast_present": ["Thanos", "Monica"],
                    "character_state_changes": ["Monica: restored and reappears"],
                },
            ],
            story_bible=bible,
            chapter_count=3,
        )
        self.assertEqual(outline[0]["cast_present"], ["Thanos", "Monica"])
        self.assertEqual(outline[1]["cast_present"], ["Thanos"])
        self.assertIn("turns to dust", outline[1]["inherited_character_state"])
        self.assertIn("Monica", outline[2]["cast_present"])

    def test_missing_cast_does_not_put_entire_bible_in_every_sequence(self):
        rows = normalize_long_form_sequence_states(
            [{
                "objective": "Alice discovers the hidden lever",
                "opening_state": "Alice enters alone",
                "closing_state": "The lever moves",
                "persistent_state": "The mechanism remains active",
                "character_state_changes": [],
            }],
            story_bible={
                "canonical_characters": [
                    {"name": "Alice"},
                    {"name": "Bob"},
                    {"name": "Carol"},
                ],
            },
        )
        self.assertEqual(rows[0]["cast_present"], ["Alice"])

    def test_story_bible_locks_location_registry_and_causal_inheritance(self):
        candidate = {
            "working_title": "The Tour",
            "logline": "One escalating comic tour.",
            "story_engine": "A new sitcom encounter in each chapter.",
            "tone_contract": "Irreverent ensemble comedy throughout.",
            "ending_contract": "End on the user's final comic payoff.",
            "allow_cast_expansion": True,
            "allow_location_expansion": True,
            "canonical_characters": [{
                "name": "Thanos",
                "role": "Disruptive visitor",
                "initial_state": "Drunk but purposeful",
                "continuity_rules": "Remembers every prior visit",
            }, {
                "name": "Rachel Green",
                "role": "First ensemble host",
                "initial_state": "Working at the coffee shop",
                "continuity_rules": "Appears only in the assigned show world",
            }],
            "canonical_locations": [{
                "location_id": "central_perk",
                "name": "Central Perk",
                "visual_identity": "Recognizable Friends coffee shop",
                "story_function": "First collision",
            }],
            "recurring_motifs": [],
            "world_rules": [],
            "forbidden_drift": [],
        }
        bible = normalize_long_form_story_bible(
            candidate,
            story_description="Thanos visits Friends and many more shows.",
            locked_dialogue=[],
            source_events=[],
            character_names=["Thanos"],
            chapter_count=2,
        )
        outline, updated = normalize_long_form_outline(
            [
                {
                    "location_id": "central_perk",
                    "location_time": "Central Perk, evening",
                    "opening_state": "Thanos enters",
                    "closing_state": "The couch is covered in dust",
                    "persistent_state": "Thanos carries the empty glass",
                    "cast_present": ["Thanos"],
                },
                {
                    "location_id": "dinner_party",
                    "location_time": "A distinct sitcom dinner party",
                    "opening_state": "The hosts look up",
                    "closing_state": "The joke escalates",
                    "persistent_state": "Thanos still carries the empty glass",
                    "cast_present": ["Thanos"],
                },
            ],
            story_bible=bible,
            chapter_count=2,
        )
        self.assertTrue(updated["registry_complete"])
        self.assertEqual(len(updated["canonical_locations"]), 2)
        self.assertIn("couch is covered in dust", outline[1]["inherited_state"].casefold())
        self.assertIn("direct visible consequence", outline[1]["opening_state"].casefold())

    def test_whole_film_dialogue_audit_repairs_typos_and_removes_phantom_speakers(self):
        bible = {
            "registry_complete": True,
            "allow_cast_expansion": True,
            "canonical_characters": [
                {"name": "Thanos"},
                {"name": "Ron"},
            ],
        }
        shots, warnings = sanitize_long_form_shot_dicts(
            [{
                "subjects_on_screen": [
                    {"character_id": "thano_id", "speaker_name": "Thano"},
                    {"character_id": "whoosh_id", "speaker_name": "Whoosh"},
                    {"character_id": "thor_id", "speaker_name": "Thor"},
                ],
                "dialogue_beats": [
                    {"speaker_id": "thano_id", "spoken_text": "A real line."},
                    {"speaker_id": "whoosh_id", "spoken_text": "Air crosses the room."},
                    {"speaker_id": "thor_id", "spoken_text": "Camera pushes closer."},
                ],
                "video_prompt": (
                    "Thano speaks: <d>[English] A real line.</d>. "
                    "Whoosh speaks: <d>[English] Air crosses the room.</d>. "
                    "Thor speaks: <d>[English] Camera pushes closer.</d>."
                ),
                "window_prompts": [],
            }],
            story_bible=bible,
        )
        self.assertEqual(
            shots[0]["subjects_on_screen"][0]["speaker_name"],
            "Thanos",
        )
        self.assertEqual(len(shots[0]["dialogue_beats"]), 1)
        self.assertIn("A real line.", shots[0]["video_prompt"])
        self.assertNotIn("Air crosses the room.", shots[0]["video_prompt"])
        self.assertNotIn("Camera pushes closer.", shots[0]["video_prompt"])
        self.assertGreaterEqual(len(warnings), 2)

    def test_long_form_h3_prompt_compactor_preserves_refs_and_exact_dialogue(self):
        filler = " ".join(["redundant cinematic continuity prose"] * 220)
        prompt, compacted = compact_long_form_h3_prompt({
            "scene_goal": "Thanos confronts the room",
            "subjects_on_screen": [{
                "character_id": "thanos",
                "speaker_name": "Thanos",
                "visual_description": "Thanos in purple and gold armor",
                "position_or_relation": "screen center beside the couch",
            }],
            "spatial_setup": "Thanos faces the seated ensemble",
            "environment": "A recognizable sitcom living room at night",
            "visual_style": "Live-action ensemble comedy",
            "lighting": "Warm practical studio lighting",
            "mood": "Irreverent",
            "action_beats": ["Thanos raises the gauntlet and the room freezes"],
            "dialogue_beats": [{
                "speaker_id": "thanos",
                "spoken_text": "Perfectly balanced bitches!",
                "delivery": "with a drunken slur",
                "physical_cue": "holding up the gauntlet",
            }],
            "camera_plan": {
                "framing": "medium wide shot",
                "movement": "fast push-in",
            },
            "audio_plan": {"ambience": "Uneasy studio-audience laughter"},
            "ending_beat": "Half the room turns to dust",
            "video_prompt": (
                "subject_definitions: <Picture 1> defines Thanos identity and "
                "appearance; reject its source background and pose.\n\n"
                "summary: Thanos confronts a sitcom ensemble.\n\n"
                "retention_analysis: <Picture 1> fully preserves identity.\n\n"
                "integrated_multimodal_description: " + filler + "\n\n"
                "overall_soundscape: Studio laughter and a magical snap.\n\n"
                "non_diegetic_music: N/A"
            ),
        })

        self.assertTrue(compacted)
        self.assertLessEqual(len(prompt.split()), 650)
        self.assertIn("<Picture 1>", prompt)
        self.assertIn(
            "<d>[English] Perfectly balanced bitches!</d>",
            prompt,
        )
        self.assertEqual(prompt.count("Perfectly balanced bitches!"), 1)

    def test_story_over_five_minutes_is_planned_in_bounded_chapters(self):
        planner = ShortFilmPlanner(llm_generate=_architect_response)
        sequence_durations: list[int] = []

        def plan_chapter(**kwargs):
            duration = kwargs["target_duration"]
            sequence_durations.append(duration)
            return [_shot(duration)], f"Local title {len(sequence_durations)}"

        with patch.object(planner, "_plan_story_driven", side_effect=plan_chapter):
            result = planner.plan(
                story_description="One long causal adventure",
                target_duration=601,
                video_model="ltx2",
            )

        self.assertEqual(
            sequence_durations,
            [67, 67, 67, 67, 67, 66, 67, 67, 66],
        )
        self.assertEqual(len(result.shots), 9)
        self.assertEqual([shot.index for shot in result.shots], list(range(9)))
        self.assertEqual(
            [shot.metadata["long_form_chapter"] for shot in result.shots],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
        )
        self.assertEqual(result.total_duration_sec, 601.0)

    def test_one_hour_plan_never_asks_for_unbounded_screenplay(self):
        planner = ShortFilmPlanner(llm_generate=_architect_response)
        sequence_durations: list[int] = []
        sequence_contracts: list[dict] = []

        def plan_sequence(**kwargs):
            duration = kwargs["target_duration"]
            sequence_durations.append(duration)
            sequence_contracts.append(kwargs)
            return [_shot(duration)], "Hour film"

        with patch.object(planner, "_plan_story_driven", side_effect=plan_sequence):
            result = planner.plan(
                story_description="One coherent hour-long story",
                target_duration=3600,
                video_model="ltx2",
            )

        self.assertEqual(len(sequence_durations), 48)
        self.assertEqual(sum(sequence_durations), 3600)
        self.assertLessEqual(max(sequence_durations), 90)
        self.assertEqual(result.total_duration_sec, 3600.0)
        self.assertTrue(all(
            call["dialogue_density_override"] == {
                "mode": "visual",
                "minimum_turns": 0,
                "minimum_words": 0,
            }
            for call in sequence_contracts
        ))
        self.assertTrue(all(
            "dialogue manifest" not in call["dialogue_intent_text"].casefold()
            for call in sequence_contracts
        ))

    def test_one_hour_studio_story_uses_bounded_ledger_chapters(self):
        calls: list[dict] = []

        def generate(**kwargs):
            calls.append(kwargs)
            schema = kwargs["json_schema"]
            if "chapters" in schema["properties"]:
                count = schema["properties"]["chapters"]["minItems"]
                return json.dumps({
                    "chapters": [
                        {
                            "chapter": index + 1,
                            "objective": f"Advance chapter {index + 1}",
                            "opening_state": f"Opening {index + 1}",
                            "closing_state": f"Closing {index + 1}",
                            "continuity_notes": "Carry established state",
                        }
                        for index in range(count)
                    ],
                })
            count = schema["properties"]["segments"]["minItems"]
            prompt = str(kwargs.get("prompt") or "")
            match = re.search(
                r"SEGMENT OBLIGATIONS:\n(\[.*?\])\n\nGLOBAL",
                prompt,
                re.S,
            )
            obligations = json.loads(match.group(1)) if match else []
            return json.dumps({
                "segments": [
                    {
                        "window": (
                            obligations[index]["window"]
                            if index < len(obligations)
                            else index + 1
                        ),
                        "supporting_progression": (
                            "New visible progression "
                            f"{obligations[index]['window'] if index < len(obligations) else index + 1}"
                        ),
                        "resulting_state": (
                            "New state "
                            f"{obligations[index]['window'] if index < len(obligations) else index + 1}"
                        ),
                        "sound_effects": "Continuous ambience",
                        "dialogue": [],
                    }
                    for index in range(count)
                ],
            })

        result = plan_h3_story_segments(
            "A traveler crosses a strange world and finally reaches a distant city.",
            segment_durations=[10.0] * 360,
            mode="sliding_window",
            camera_coverage="multi_shot",
            llm_generate=generate,
        )

        segment_calls = [
            call for call in calls
            if "segments" in call["json_schema"]["properties"]
        ]
        self.assertEqual(len(result["segments"]), 360)
        self.assertEqual(result["planned_by"], "hierarchical_llm")
        self.assertIn("long_form_story_bible", result["ledger"])
        self.assertEqual(len(result["ledger"]["long_form_chapters"]), 15)
        self.assertIn(
            "Closing 1",
            result["ledger"]["long_form_chapters"][1]["inherited_state"],
        )
        self.assertEqual(len(calls), 16)
        self.assertEqual(len(segment_calls), 15)
        self.assertLessEqual(
            max(
                call["json_schema"]["properties"]["segments"]["maxItems"]
                for call in segment_calls
            ),
            24,
        )

    def test_locked_dialogue_is_exposed_to_exactly_one_sequence(self):
        line = "Perfectly balanced bitches!"

        def architect(**kwargs):
            prompt = str(kwargs.get("prompt") or "")
            rows = json.loads(_architect_response(**kwargs))
            if "chapters for one" in prompt:
                rows[-1]["dialogue_ids"] = ["D1"]
            elif "Divide chapter 2" in prompt:
                rows[-1]["dialogue_ids"] = ["D1"]
            return json.dumps(rows)

        planner = ShortFilmPlanner(llm_generate=architect)
        sequence_stories: list[str] = []

        def plan_sequence(**kwargs):
            sequence_stories.append(kwargs["story_description"])
            return [_shot(kwargs["target_duration"])], "Dialogue film"

        with patch.object(planner, "_plan_story_driven", side_effect=plan_sequence):
            planner.plan(
                story_description=(
                    "Thanos crosses the ruined hall, then Thanos says "
                    f'"{line}" before the final reversal.'
                ),
                target_duration=301,
                video_model="minimax_h3_ref2va",
                shot_image_policy="direct_references",
            )

        owners = [story for story in sequence_stories if line in story]
        self.assertEqual(len(owners), 1)
        self.assertTrue(all("[D1:" in story for story in sequence_stories))

    def test_locked_dialogue_stays_with_its_redacted_speech_event(self):
        rows = [
            {
                "source_event_ids": ["E1"],
                "dialogue_ids": [],
            },
            {
                "source_event_ids": ["E2"],
                "dialogue_ids": ["D1"],
            },
        ]
        normalized = _normalize_long_form_dialogue_ownership(
            rows,
            allowed_dialogue=[{
                "dialogue_id": "D1",
                "text": "Perfectly balanced bitches!",
                "source_offset": 25,
            }],
            source_events=[
                {
                    "event_id": "E1",
                    "text": "Thanos says [D1: exact dialogue assigned by Maestro]",
                },
                {"event_id": "E2", "text": "The room reacts"},
            ],
        )

        self.assertEqual(normalized[0]["dialogue_ids"], ["D1"])
        self.assertEqual(normalized[1]["dialogue_ids"], [])

    def test_pathological_event_dump_is_clamped_to_source_order(self):
        rows = [
            {
                "source_event_ids": (
                    ["E1"] if index == 0 else
                    ["E2", "E3", "E4", "E5", "E6"] if index == 11 else
                    []
                ),
            }
            for index in range(12)
        ]
        events = [
            {"event_id": f"E{index}", "text": f"Event {index}"}
            for index in range(1, 7)
        ]

        normalized = _normalize_long_form_event_ownership(
            rows,
            source_events=events,
        )
        owners = {
            event_id: row_index
            for row_index, row in enumerate(normalized)
            for event_id in row["source_event_ids"]
        }

        self.assertEqual(set(owners), {f"E{index}" for index in range(1, 7)})
        self.assertEqual(owners["E1"], 0)
        self.assertLessEqual(owners["E2"], 4)
        self.assertLess(owners["E3"], 11)
        self.assertEqual(owners["E6"], 11)

    def test_unowned_registry_ids_and_literal_dialogue_are_removed_from_prose(self):
        exact = "Perfectly balanced bitches!"
        rows = [{
            "objective": "Thanos delivers D1 before E1.",
            "source_event_ids": ["E1"],
            "dialogue_ids": ["D1"],
        }, {
            "objective": f"After delivering D1 — {exact} — continue to E2.",
            "source_event_ids": ["E2"],
            "dialogue_ids": [],
        }]
        normalized = _normalize_long_form_plan_references(
            rows,
            allowed_dialogue=[{"dialogue_id": "D1", "text": exact}],
            source_events=[
                {"event_id": "E1", "text": "Thanos speaks"},
                {"event_id": "E2", "text": "The room reacts"},
            ],
        )

        self.assertIn("D1", normalized[0]["objective"])
        self.assertIn("E1", normalized[0]["objective"])
        self.assertNotIn("D1", normalized[1]["objective"])
        self.assertNotIn(exact, normalized[1]["objective"])
        self.assertIn("E2", normalized[1]["objective"])

    def test_long_form_dialogue_targets_use_only_local_owned_intent(self):
        rows = [{
            "objective": "Cross the silent ruined hall",
            "source_event_ids": ["E1"],
            "dialogue_ids": [],
            "dialogue_mode": "visual",
            "dialogue_target_turns": 10,
            "dialogue_target_words": 52,
        }, {
            "objective": "Thanos says D1 and the room reacts",
            "source_event_ids": ["E2"],
            "dialogue_ids": ["D1"],
            "dialogue_mode": "natural",
            "dialogue_target_turns": 4,
            "dialogue_target_words": 30,
        }]
        dialogue = [{
            "dialogue_id": "D1",
            "text": "Perfectly balanced bitches!",
        }]
        events = [
            {"event_id": "E1", "text": "Thanos crosses the hall"},
            {"event_id": "E2", "text": "Thanos says [D1]"},
        ]

        normalized = _normalize_long_form_dialogue_targets(
            rows,
            durations=[75, 75],
            allowed_dialogue=dialogue,
            source_events=events,
        )

        self.assertEqual(normalized[0]["dialogue_target_turns"], 0)
        self.assertEqual(normalized[0]["dialogue_target_words"], 0)
        self.assertEqual(normalized[1]["dialogue_target_turns"], 4)
        self.assertEqual(normalized[1]["dialogue_target_words"], 30)

    def test_budgeted_screenplay_fallback_preserves_literal_user_dialogue(self):
        exact = "Keep this exact."
        story = f'The hero says "{exact}" before the villain answers.'
        screenplay = (
            "INT. HALL - NIGHT\n\n"
            f"HERO\n{exact}\n\n"
            "VILLAIN\nThis generated answer is deliberately far too long "
            "and keeps adding unnecessary words that cannot fit inside the "
            "bounded native video dialogue timing.\n\nFADE OUT."
        )

        bounded = _build_h3_budgeted_screenplay_fallback(
            story_description=story,
            story_blueprint=[{
                "opening_cause": "The hero enters.",
                "visible_beats": ["The villain turns."],
                "outgoing_handoff": "The door closes.",
            }],
            screenplay=screenplay,
            max_total_words=42,
            max_spoken_words=9,
            maximum_line_words=6,
        )

        self.assertIn(exact, bounded)
        self.assertEqual(
            _h3_screenplay_recovery_reasons(
                bounded,
                story_description=story,
            ),
            [],
        )
        self.assertEqual(
            _h3_screenplay_budget_issues(
                bounded,
                story_description=story,
                max_total_words=42,
                max_spoken_words=9,
                maximum_line_words=6,
            ),
            [],
        )

    def test_h3_overlong_sequence_is_bounded_before_native_shot_planning(self):
        oversized = (
            "INT. RUINED HALL - NIGHT\n\n"
            + " ".join(
                f"Visible action {index} continues through the hall."
                for index in range(100)
            )
            + "\n\nFADE OUT."
        )
        planner = ShortFilmPlanner(
            llm_generate=lambda **_kwargs: oversized,
            llm_generate_streaming=lambda **_kwargs: oversized,
        )
        planner._video_model = "minimax_h3_ref2va"
        planner._image_model = ""
        planner._uses_generated_shot_images = False
        planner._preserve_video_character_names = True
        planner._long_form_story_blueprint_override = [{
            "scene_number": 1,
            "location_time": "Ruined hall at night",
            "active_objective": "Cross the hall",
            "story_purpose": "Reach the far door",
            "opening_cause": "The traveler enters the hall",
            "visible_beats": ["The traveler crosses the hall"],
            "choice_or_discovery": "The far door opens",
            "outgoing_handoff": "The traveler exits through the far door",
            "persistent_state_after": "The traveler remains dusty",
        }]
        captured: dict = {}

        def native_plan(**kwargs):
            captured.update(kwargs)
            return [_shot(kwargs["target_duration"])], "Bounded sequence"

        with (
            patch.object(
                planner,
                "_build_h3_character_voice_bible",
                return_value=[],
            ),
            patch.object(
                planner,
                "_plan_story_h3_native",
                side_effect=native_plan,
            ),
        ):
            planner._plan_story_driven(
                story_description=(
                    "The traveler silently crosses a ruined hall and leaves "
                    "through the far door."
                ),
                reference_image_path=None,
                char_profiles=[],
                has_reference=False,
                target_duration=75,
                target_scenes=None,
                narrative_mode=True,
                fps=24,
                frames_steps=17,
                frames_minimum=124,
                frames_maximum=345,
            )

        self.assertLessEqual(len(captured["screenplay"].split()), 337)
        self.assertEqual(
            _h3_screenplay_budget_issues(
                captured["screenplay"],
                max_total_words=337,
                max_spoken_words=150,
                maximum_line_words=28,
            ),
            [],
        )

    def test_ltx_overlong_sequence_is_bounded_before_image_and_video_prompts(self):
        exact = "We leave together."
        oversized = (
            "INT. FLOODED STATION - NIGHT\n\n"
            + " ".join(
                f"Visible action {index} keeps elaborating the same crossing."
                for index in range(100)
            )
            + "\n\nFADE OUT."
        )
        planner = ShortFilmPlanner(
            llm_generate=lambda **_kwargs: oversized,
            llm_generate_streaming=lambda **_kwargs: oversized,
        )
        planner._video_model = "ltx2_25"
        planner._image_model = "flux"
        planner._uses_generated_shot_images = True
        planner._preserve_video_character_names = False
        planner._long_form_story_blueprint_override = [{
            "scene_number": 1,
            "location_time": "Flooded station at night",
            "active_objective": "Reach the evacuation train",
            "story_purpose": "Escape the rising water",
            "opening_cause": "The warning lights switch from amber to red",
            "visible_beats": ["Maya and Ren cross the flooded platform"],
            "choice_or_discovery": "The final train doors begin closing",
            "outgoing_handoff": "Maya and Ren board as the doors seal",
            "persistent_state_after": "Both travelers remain soaked",
        }]
        captured: dict = {}

        def pass2(**kwargs):
            captured.update(kwargs)
            rows = []
            for index in range(2):
                rows.append({
                    "title": f"Station beat {index + 1}",
                    "duration_sec": 40,
                    "scene_goal": f"Advance station beat {index + 1}",
                    "narrative_role": "rising_action",
                    "scene_type": "action",
                    "subjects_on_screen": [{
                        "visual_description": "two soaked travelers",
                        "character_id": "travelers",
                        "position_or_relation": "center frame on the platform",
                    }],
                    "spatial_setup": "The travelers face the evacuation train",
                    "environment": "A flooded underground station at night",
                    "visual_style": "Cinematic realism",
                    "lighting": "Flashing red warning lights",
                    "mood": "Urgent",
                    "action_beats": [f"They advance through beat {index + 1}"],
                    "dialogue_beats": [],
                    "camera_plan": {"framing": "medium wide shot"},
                    "audio_plan": {"mode": "ambient_only"},
                    "ending_beat": f"Beat {index + 1} reaches a visible result",
                    "image_source": "",
                    "image_prompt": "",
                    "visual_changes": [],
                    "video_prompt": "",
                    "multishot": False,
                    "window_prompts": [],
                })
            return rows

        with patch.object(planner, "_call_llm_json", side_effect=pass2):
            shots, _title = planner._plan_story_driven(
                story_description=(
                    "Maya crosses the flooded station, says "
                    f'"{exact}" and boards the evacuation train.'
                ),
                reference_image_path=None,
                char_profiles=[],
                has_reference=False,
                target_duration=80,
                target_scenes=None,
                narrative_mode=True,
                fps=24,
                frames_steps=17,
                frames_minimum=9,
            )

        screenplay = captured["user_prompt"].rsplit("SCREENPLAY:\n", 1)[-1]
        self.assertLessEqual(len(screenplay.split()), 360)
        self.assertIn(exact, screenplay)
        self.assertEqual(len(shots), 2)
        self.assertTrue(all(str(shot.image_prompt or "").strip() for shot in shots))
        self.assertIn(
            "warning lights switch from amber to red",
            shots[0].image_prompt.casefold(),
        )
        self.assertTrue(all(len(shot.window_prompts or []) == 2 for shot in shots))
        self.assertTrue(all(
            all(str(prompt or "").strip() for prompt in shot.window_prompts or [])
            for shot in shots
        ))
        self.assertIn(
            exact,
            " ".join(
                prompt
                for shot in shots
                for prompt in shot.window_prompts or []
            ),
        )
        self.assertIn(
            "warning lights switch from amber to red",
            shots[0].window_prompts[0].casefold(),
        )
        self.assertIn(
            "board as the doors seal",
            shots[-1].window_prompts[-1].casefold(),
        )

    def test_long_form_ltx_video_only_contract_does_not_create_image_prompts(self):
        rows = [{
            "title": "Desert crossing",
            "duration_sec": 20,
            "scene_goal": "Reach the ridge",
            "subjects_on_screen": [{
                "visual_description": "a dust-covered cyclist",
                "character_id": "cyclist",
                "position_or_relation": "screen center in the foreground",
            }],
            "spatial_setup": "The cyclist faces a steep ridge",
            "environment": "A sunlit desert basin",
            "visual_style": "Naturalistic adventure film",
            "lighting": "Hard late-afternoon sunlight",
            "mood": "Determined",
            "action_beats": ["The cyclist pedals toward the ridge"],
            "dialogue_beats": [],
            "camera_plan": {"framing": "wide tracking shot"},
            "ending_beat": "The front wheel reaches the ridge line",
            "video_prompt": "",
            "window_prompts": [],
        }]
        prepared = _prepare_long_form_ltx_prompt_contract(
            rows,
            [{
                "opening_cause": "A dust storm clears to reveal the ridge",
                "active_objective": "Reach the ridge before sunset",
                "outgoing_handoff": "The cyclist crests the ridge",
                "persistent_state_after": "The cyclist remains dust-covered",
            }],
            uses_generated_images=False,
        )

        self.assertNotIn("image_prompt", prepared[0])
        self.assertNotIn("image_source", prepared[0])
        self.assertIn("sunlit desert basin", prepared[0]["video_prompt"].casefold())
        self.assertIn("dust storm clears", prepared[0]["video_prompt"].casefold())
        self.assertIn("cyclist crests the ridge", prepared[0]["video_prompt"].casefold())

    def test_dialogue_coverage_expansion_adds_capacity_without_replaying_action(self):
        spoken = [
            "one two three four five six seven eight nine ten",
            "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty",
            "twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine thirty",
            "thirty-one thirty-two thirty-three thirty-four thirty-five thirty-six thirty-seven thirty-eight thirty-nine forty",
        ]
        source = [{
            "title": "One visual shot",
            "subjects_on_screen": [{
                "speaker_name": "Alex",
                "character_id": "alex",
                "visual_description": "Alex",
            }],
            "environment": "One room",
            "action_beats": ["Alex enters through the only door."],
            "dialogue_beats": [
                {"speaker_id": "alex", "spoken_text": line}
                for line in spoken
            ],
            "audio_plan": {},
            "video_prompt": "Alex enters through the only door.",
        }]
        expanded = _expand_h3_dialogue_coverage_slots(
            source,
            desired_count=2,
        )
        restored = _restore_h3_dialogue_after_pacing_repair(
            source,
            expanded,
            [14.0, 14.0],
        )

        self.assertEqual(len(restored), 2)
        self.assertEqual(
            [
                beat["spoken_text"]
                for shot in restored
                for beat in shot["dialogue_beats"]
            ],
            spoken,
        )
        self.assertNotIn(
            "enters through the only door",
            " ".join(restored[1]["action_beats"]).casefold(),
        )

    def test_old_planning_checkpoint_schema_is_not_resumed(self):
        planner = ShortFilmPlanner(llm_generate=_architect_response)
        kwargs = {}
        planner._configure_planning_runtime(
            kwargs,
            kind="short_film_story",
            fingerprint_payload={"request": "same"},
        )
        fingerprint = planner._planning_checkpoint_fingerprint

        planner._configure_planning_runtime(
            {
                "_planning_checkpoint": {
                    "version": 1,
                    "kind": "short_film_story",
                    "fingerprint": fingerprint,
                    "completed_sequences": {"1:1": {"shots": [{"stale": True}]}},
                },
            },
            kind="short_film_story",
            fingerprint_payload={"request": "same"},
        )
        self.assertEqual(planner._planning_resume_checkpoint, {})

    def test_long_story_resume_skips_completed_sequences(self):
        checkpoints: list[dict] = []
        request = dict(
            story_description="A continuous expedition",
            target_duration=301,
            video_model="ltx2",
            _planning_checkpoint_callback=lambda value: checkpoints.append(value),
        )
        first = ShortFilmPlanner(llm_generate=_architect_response)
        first_calls = 0

        def interrupt_second(**kwargs):
            nonlocal first_calls
            first_calls += 1
            if first_calls == 2:
                raise RuntimeError("simulated interruption")
            return [_shot(kwargs["target_duration"])], "Expedition"

        with patch.object(first, "_plan_story_driven", side_effect=interrupt_second):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                first.plan(**request)

        checkpoint = checkpoints[-1]
        self.assertEqual(len(checkpoint["completed_sequences"]), 1)

        resumed = ShortFilmPlanner(llm_generate=_architect_response)
        remaining: list[int] = []

        def finish_sequence(**kwargs):
            remaining.append(kwargs["target_duration"])
            return [_shot(kwargs["target_duration"])], "Expedition"

        with patch.object(resumed, "_plan_story_driven", side_effect=finish_sequence):
            result = resumed.plan(
                **request,
                _planning_checkpoint=checkpoint,
            )

        self.assertEqual(len(result.shots), 4)
        self.assertEqual(len(remaining), 3)

    def test_missing_locked_dialogue_is_restored_without_regeneration(self):
        story = 'Thanos looks up and says "Perfectly balanced bitches!"'
        screenplay = (
            "INT. RUINED HALL - NIGHT\n\n"
            "Thanos crosses the room while dust falls around him.\n\n"
            "FADE OUT."
        )
        repaired = _restore_missing_h3_screenplay_dialogue(
            screenplay,
            story_description=story,
        )
        self.assertIn("THANOS", repaired)
        self.assertIn("Perfectly balanced bitches!", repaired)
        self.assertEqual(
            _h3_screenplay_recovery_reasons(
                repaired,
                story_description=story,
            ),
            [],
        )

    def test_music_video_plans_long_timeline_in_twelve_clip_batches(self):
        calls: list[int] = []

        def generate(**kwargs):
            match = re.search(
                r"Write\s+(\d+)\s+structured shot plans",
                str(kwargs.get("prompt") or ""),
                re.IGNORECASE,
            )
            count = int(match.group(1))
            calls.append(count)
            return json.dumps([
                {
                    "scene_goal": f"Clip {index + 1}",
                    "scene_type": "performance",
                    "subjects_on_screen": [],
                    "environment": "stage",
                    "visual_style": "cinematic",
                    "lighting": "concert lighting",
                    "mood": "energetic",
                    "action_beats": ["performer continues"],
                    "camera_plan": {"framing": "medium shot"},
                    "ending_beat": f"ending {index + 1}",
                    "video_prompt": "The performance continues.",
                    "window_prompts": [],
                }
                for index in range(count)
            ])

        clips = [
            {"start": index * 10, "end": (index + 1) * 10, "label": "verse"}
            for index in range(25)
        ]
        checkpoints: list[dict] = []
        planner = MusicVideoPlanner(llm_generate=generate)
        result = planner.plan(
            clips=clips,
            scene_description="One continuous performance",
            video_model="minimax_h3_ref2va",
            shot_image_policy="direct_references",
            _planning_checkpoint_callback=lambda value: checkpoints.append(value),
        )

        self.assertEqual(calls, [12, 12, 1])
        self.assertEqual(len(result.shots), 25)
        self.assertTrue(checkpoints[-1]["complete"])

    def test_podcast_plans_long_timeline_in_twelve_clip_batches(self):
        calls: list[int] = []

        def generate(**kwargs):
            match = re.search(
                r"Write exactly\s+(\d+)\s+structured shot plans",
                str(kwargs.get("prompt") or ""),
                re.IGNORECASE,
            )
            count = int(match.group(1))
            calls.append(count)
            return json.dumps([
                {
                    "scene_goal": f"Segment {index + 1}",
                    "scene_type": "speaker",
                    "subjects_on_screen": [],
                    "spatial_setup": "at the desk",
                    "environment": "podcast studio",
                    "visual_style": "clean",
                    "lighting": "soft",
                    "mood": "conversational",
                    "action_beats": ["speaker talks"],
                    "dialogue_beats": [],
                    "camera_plan": {"framing": "medium shot"},
                    "audio_plan": {
                        "mode": "dialogue_driven",
                        "lip_sync_critical": True,
                    },
                    "ending_beat": f"ending {index + 1}",
                }
                for index in range(count)
            ])

        clips = [
            {"start": index * 10, "end": (index + 1) * 10}
            for index in range(25)
        ]
        planner = PodcastPlanner(llm_generate=generate)
        result = planner.plan(
            clips=clips,
            visual_style="clean studio",
            _planning_checkpoint_callback=lambda _value: None,
        )

        self.assertEqual(calls, [12, 12, 1])
        self.assertEqual(len(result.shots), 25)

    def test_audio_film_plans_long_timeline_in_twelve_clip_batches(self):
        calls: list[int] = []

        def generate(**kwargs):
            match = re.search(
                r"Plan visuals for each of these\s+(\d+)\s+dialogue segments",
                str(kwargs.get("prompt") or ""),
                re.IGNORECASE,
            )
            count = int(match.group(1))
            calls.append(count)
            return json.dumps([
                {
                    "scene_goal": f"Audio segment {index + 1}",
                    "scene_type": "dialogue",
                    "subjects_on_screen": [],
                    "spatial_setup": "maintain staging",
                    "environment": "room",
                    "visual_style": "cinematic",
                    "lighting": "natural",
                    "mood": "focused",
                    "action_beats": ["speaker continues"],
                    "dialogue_beats": [],
                    "camera_plan": {"framing": "medium shot"},
                    "audio_plan": {
                        "mode": "dialogue_driven",
                        "lip_sync_critical": True,
                    },
                    "ending_beat": f"ending {index + 1}",
                    "image_source": "original",
                    "image_prompt": "A room before the speaker moves.",
                    "visual_changes": [],
                    "video_prompt": "The speaker performs with the audio.",
                    "window_prompts": [],
                }
                for index in range(count)
            ])

        clips = [
            {"start": index * 10, "end": (index + 1) * 10, "label": "scene"}
            for index in range(25)
        ]
        planner = ShortFilmPlanner(llm_generate=generate)
        result = planner.plan(
            clips=clips,
            story_description="One continuous recorded conversation",
            video_model="ltx2",
            image_model="flux",
            _planning_checkpoint_callback=lambda _value: None,
        )

        self.assertEqual(calls, [12, 12, 1])
        self.assertEqual(len(result.shots), 25)

    def test_cancelled_batch_is_not_checkpointed_as_a_fallback(self):
        checkpoints: list[dict] = []
        planner = MusicVideoPlanner(llm_generate=lambda **_: "[]")
        planner._configure_planning_runtime(
            {
                "_planning_checkpoint_callback": (
                    lambda value: checkpoints.append(value)
                ),
            },
            kind="cancellation_test",
            fingerprint_payload={"request": "same"},
        )

        with self.assertRaisesRegex(InterruptedError, "stop planning"):
            planner._run_checkpointed_json_batches(
                items=[{"clip": 1}],
                batch_size=1,
                checkpoint_key="batches",
                stage="test",
                progress_label="test",
                call_batch=lambda *_args: (_ for _ in ()).throw(
                    InterruptedError("stop planning")
                ),
                fallback_factory=lambda *_args: {"fallback": True},
            )

        self.assertEqual(checkpoints, [])


if __name__ == "__main__":
    unittest.main()
