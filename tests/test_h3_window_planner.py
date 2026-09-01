"""Model-free regressions for MiniMax H3 window-local storyboarding."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_window_planner import (  # noqa: E402
    _compact,
    _creative_dialogue_expected,
    _dialogue_sentence,
    _fallback_plan,
    _narrative_dialogue_expected,
    _plan_contract_violations,
    compile_h3_window_prompts,
    compute_h3_window_boundaries,
    h3_window_plan_signature,
    normalize_h3_injected_keyframes,
    parse_h3_manual_window_prompts,
    plan_h3_sliding_windows,
    reviewed_h3_window_plan_matches,
)
from services.h3_story_ledger import (  # noqa: E402
    extract_source_events,
    plan_h3_story_segments,
    recover_h3_plain_story,
)


def _staged_ledger(
    prompt: str,
    segment_count: int,
    *,
    dialogue: bool = False,
    golden: bool = False,
) -> dict:
    source_events = extract_source_events(prompt)
    event_buckets: list[list[str]] = [[] for _ in range(segment_count)]
    for index, event in enumerate(source_events):
        target = min(
            segment_count - 1,
            round(index * (segment_count - 1) / max(1, len(source_events) - 1)),
        )
        event_buckets[target].append(event["event_id"])
    generated = ([{
        "dialogue_id": "D1",
        "speaker": "Lana Lang",
        "language": "English",
        "delivery": "asks in stunned disbelief",
        "text": "Clark, how did you do that?",
    }] if dialogue else [])
    return {
        "subject_continuity": "One unchanged subject",
        "setting_continuity": "One unchanged location",
        "visual_continuity": "One coherent live-action style",
        "editing_style": "Motivated cinematic coverage",
        "initial_state": "The subject begins at rest",
        "ambient_audio": "continuous room tone",
        "music": "N/A",
        "required_final_outcome": "The requested final action completes",
        "beats": [
            {
                "beat_id": f"B{index + 1}",
                "segment": index + 1,
                "description": (
                    "Clark emits a golden energy wave"
                    if golden and index == 1
                    else (
                        " Then ".join(
                            event["text"] for event in source_events
                            if event["event_id"] in event_buckets[index]
                        ) or f"Only action {index + 1} happens"
                    )
                ),
                "source_event_ids": event_buckets[index],
                "dialogue_ids": (["D1"] if dialogue and index + 1 == segment_count else []),
                "state_after": f"The subject reaches state {index + 1}",
                "sound_effects": "Natural synchronized action",
            }
            for index in range(segment_count)
        ],
        "generated_dialogue": generated,
    }


def _staged_segment(
    index: int,
    duration: float,
    *,
    dialogue_id: str | None = None,
    action: str | None = None,
) -> dict:
    return {
        "segment": index,
        "title": f"Beat {index}",
        "opening_state": f"Opening state {index}",
        "coverage": "cinematic coverage",
        "pacing": "natural real-time pacing",
        "shots": [{
            "shot": 1,
            "start_seconds": 0.0,
            "end_seconds": duration,
            "transition": "opening composition",
            "framing": "cinematic medium shot",
            "camera": "the camera follows the action",
            "beat_ids": [f"B{index}"],
            "action": action or f"Only action {index} happens",
            "dialogue": ([{
                "dialogue_id": dialogue_id,
                "delivery": "asks in stunned disbelief",
                "action": "staring at Clark",
            }] if dialogue_id else []),
            "sound_effects": "Natural synchronized action",
        }],
        "closing_state": f"The subject reaches state {index}",
    }


class H3WindowPlannerTests(unittest.TestCase):
    def test_official_dialogue_format_supports_groups_and_voiceover(self):
        speaker_ids: dict[str, str] = {}
        group = _dialogue_sentence({
            "speaker": "The two children",
            "speaker_id": "S1, S2",
            "language": "English",
            "delivery": "shout together",
            "action": "running toward the door",
            "text": "Wait for us!",
        }, speaker_ids)
        voiceover = _dialogue_sentence({
            "speaker": "The narrator",
            "speaker_id": "",
            "language": "English",
            "delivery": "quiet reflective voiceover",
            "action": "off-screen while his younger self walks",
            "text": "I still remember that road.",
        }, speaker_ids)

        self.assertIn("(S1,S2)", group)
        self.assertIn("<d>[English] Wait for us!</d>", group)
        self.assertIn("(S3) says in an off-screen voiceover", voiceover)
        self.assertIn("on-screen character's lips remain completely closed", voiceover)

    def test_context_ir_is_unwrapped_before_sequence_planning(self):
        recovered = recover_h3_plain_story(
            "subject_definitions: <Picture 1> defines Superman.\n\n"
            "summary: [reference generation] Superman crosses the city, then confronts Thanos.\n\n"
            "retention_analysis: <Picture 1>: reference\n\n"
            "detailed_description: Superman (S1) faces Thanos and speaks: "
            "<d>[English] Enough.</d>\n\n"
            "overall_soundscape: Wind.\n\n"
            "non_diegetic_music: N/A"
        )
        self.assertEqual(
            recovered,
            'Superman crosses the city, then confronts Thanos. Superman says "Enough.".',
        )
        self.assertNotIn("Picture 1", recovered)

    def test_source_event_parser_preserves_user_line_breaks_and_camera_cuts(self):
        events = extract_source_events(
            "Superman fights Thanos\n"
            "Superman accelerates through the ruined city\n"
            "Cut to Superman striking Thanos at street level"
        )
        self.assertEqual(len(events), 3)
        self.assertEqual(
            [item["text"] for item in events],
            [
                "Superman fights Thanos",
                "Superman accelerates through the ruined city",
                "Cut to Superman striking Thanos at street level",
            ],
        )
        repeated_then = extract_source_events(
            "Superman crosses the city Then then cut to Superman striking Thanos"
        )
        self.assertEqual(
            [item["text"] for item in repeated_then],
            ["Superman crosses the city", "cut to Superman striking Thanos"],
        )

    @patch("services.llm_service.generate", side_effect=RuntimeError("offline"))
    def test_story_fallback_uses_concrete_unique_handoff_states(self, _generate):
        result = plan_h3_story_segments(
            "Superman crosses the ruined city\nSuperman strikes Thanos",
            segment_durations=[10.0, 10.0, 10.0],
            mode="reference_sequence_continuation",
            camera_coverage="multi_shot",
        )
        descriptions = [item["description"] for item in result["ledger"]["beats"]]
        closing_states = [item["closing_state"] for item in result["segments"]]
        self.assertEqual(len(descriptions), len(set(descriptions)))
        self.assertNotIn(
            "concrete continuation state",
            " ".join(closing_states).casefold(),
        )
        self.assertIn("Superman strikes Thanos", closing_states[-1])

    @patch("services.llm_service.generate", side_effect=RuntimeError("offline"))
    def test_fallback_preserves_first_person_story_across_two_windows(self, _generate):
        prompt = (
            "POV: The viewer is Harry Potter as he stands on top of a scenic mountain. "
            "The scene starts with Hermione Granger and Ron Weasley standing with you, "
            "each holding a broom. Hermione says, \"Come on! Let's go!\" Ron says in a "
            "nervous voice, \"Ah, I don't know about this, Hermione.\" The POV off-camera "
            "voice of Harry Potter yells, \"Leeroy Jenkins!!!\" Then they all laugh as "
            "they mount their brooms and plummet over the edge above the clouds, dropping "
            "at an extremely high rate of speed while flying between canyon walls, through "
            "waterfalls, and through caves. Extremely exciting POV speed with two hands "
            "holding the end of the broom. Epic, thrilling, cinematic, realistic film."
        )
        result = plan_h3_sliding_windows(
            prompt,
            model_type="minimax_h3_fl2va_full",
            resolution="1280x704",
            total_frames=648,
            window_frames=345,
            overlap_frames=18,
            fps=24,
        )

        self.assertEqual(result["window_count"], 2)
        self.assertEqual(result["planned_by"], "deterministic_fallback")
        self.assertTrue(result.get("planning_warnings"))
        first, second = result["window_prompts"]
        joined = "\n".join(result["window_prompts"])

        for line in (
            "Come on! Let's go!",
            "Ah, I don't know about this, Hermione.",
            "Leeroy Jenkins!!!",
        ):
            self.assertEqual(joined.count(line), 1)
        self.assertIn("first-person POV", first)
        self.assertNotIn("wide or medium-wide establishing view", joined)
        self.assertIn("mount their brooms", first)
        self.assertIn("plummet over the edge", first)
        self.assertIn("between canyon walls", second)
        self.assertIn("through waterfalls", second)
        self.assertIn("through caves", second)
        self.assertIn("hands and held object remain visible", second)
        self.assertIn("extremely fast real-time movement", second)
        self.assertNotIn("<d>[English]", second)
        self.assertIn(
            "<Picture 1> (from [Shot 1]) is fully referenced",
            second,
        )
        self.assertNotIn("central outcome", joined.casefold())
        self.assertNotIn("Hermione Granger and Ron.", joined)

    def test_native_reference_segment_instruction_rejects_keyframe_restaging(self):
        calls = []

        def offline_generate(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("offline")

        plan_h3_story_segments(
            "Superman crosses the city. Superman strikes Thanos.",
            segment_durations=[10.0, 10.0],
            mode="reference_sequence_continuation",
            camera_coverage="multi_shot",
            llm_generate=offline_generate,
        )
        first_segment_prompt = calls[1]["prompt"]
        self.assertIn("native Ref2VA motion-and-audio overlap continuation", first_segment_prompt)
        self.assertIn("not opening keyframes", first_segment_prompt)
        self.assertIn("without restarting, restaging, or replaying", first_segment_prompt)

    def test_compaction_keeps_complete_sentences_or_clauses(self):
        value = (
            "Clark walks along Smallville's main street in his familiar everyday clothes. "
            "He notices an approaching truck and begins to turn toward the sound."
        )
        compacted = _compact(value, 91)
        self.assertEqual(
            compacted,
            "Clark walks along Smallville's main street in his familiar everyday clothes",
        )
        self.assertNotIn("..", compacted)

    def test_boundaries_match_h3_committed_continuation_frames(self):
        spans = compute_h3_window_boundaries(
            345,
            124,
            fps=24,
            overlap_frames=1,
            discard_frames=0,
        )
        self.assertEqual(
            [(item["start_frame"], item["end_frame"]) for item in spans],
            [(0, 124), (124, 247), (247, 345)],
        )
        self.assertAlmostEqual(spans[1]["start_seconds"], 124 / 24, places=3)
        self.assertAlmostEqual(spans[-1]["end_seconds"], 345 / 24, places=3)

    def test_manual_first_last_prompts_remain_local_to_their_windows(self):
        spans = compute_h3_window_boundaries(
            998,
            345,
            fps=24,
            overlap_frames=17,
        )
        self.assertEqual(len(spans), 3)
        self.assertEqual(
            parse_h3_manual_window_prompts(
                "First action only\nSecond action only\nThird action only",
                expected_count=len(spans),
            ),
            ["First action only", "Second action only", "Third action only"],
        )

    def test_manual_first_last_prompts_accept_the_ui_array(self):
        self.assertEqual(
            parse_h3_manual_window_prompts(
                "This fallback must not replace the explicit array",
                expected_count=3,
                window_prompts=[" Window one ", "Window two", "Window three"],
            ),
            ["Window one", "Window two", "Window three"],
        )

    def test_manual_first_last_prompt_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 3 non-empty prompt lines"):
            parse_h3_manual_window_prompts(
                "first window\nsecond window",
                expected_count=3,
            )

    def test_compiler_keeps_actions_and_dialogue_local_to_each_window(self):
        spans = compute_h3_window_boundaries(345, 124, fps=24, overlap_frames=1)
        plan = {
            "subject_continuity": "Clark Kent wears the same blue shirt and red jacket",
            "setting_continuity": "Smallville main street in warm afternoon light",
            "visual_continuity": "live-action television drama, steady tracking camera",
            "initial_state": "Clark walks screen-right on the sidewalk",
            "ambient_audio": "light traffic, footsteps, and a soft Kansas breeze",
            "music": "N/A",
            "windows": [
                {
                    "window": 1,
                    "title": "Danger appears",
                    "action": "Clark hears a runaway truck and turns toward it",
                    "dialogue": [],
                    "sound_effects": "a distant truck horn",
                    "closing_state": "Clark faces the approaching truck with one foot planted forward",
                },
                {
                    "window": 2,
                    "title": "The rescue",
                    "action": "Clark accelerates, reaches the truck, and braces both hands against its grille",
                    "dialogue": [
                        {
                            "speaker": "Driver",
                            "speaker_id": "S1",
                            "language": "English",
                            "delivery": "shouts urgently",
                            "action": "gripping the wheel",
                            "text": "Look out!",
                        }
                    ],
                    "sound_effects": "tires skid and metal groans",
                    "closing_state": "Clark holds the slowing truck while the driver grips the wheel",
                },
                {
                    "window": 3,
                    "title": "Safe landing",
                    "action": "Clark stops the truck, checks the driver, and steps back into the crowd",
                    "dialogue": [],
                    "sound_effects": "the engine settles to idle",
                    "closing_state": "the truck is safely stopped and Clark stands unnoticed among pedestrians",
                },
            ],
        }
        compiled = compile_h3_window_prompts(plan, spans)
        self.assertEqual(len(compiled), 3)
        first = compiled[0]["prompt"]
        second = compiled[1]["prompt"]
        final = compiled[2]["prompt"]
        self.assertIn("hears a runaway truck", first)
        self.assertNotIn("braces both hands", first)
        self.assertNotIn("stops the truck", first)
        self.assertIn("braces both hands", second)
        self.assertIn("<d>[English] Look out!</d>", second)
        self.assertNotIn("Look out!", first)
        self.assertNotIn("Look out!", final)
        self.assertIn(plan["windows"][0]["closing_state"], compiled[1]["opening_state"])
        for item in compiled:
            self.assertEqual(item["prompt"].count("integrated_multimodal_description:"), 1)
            self.assertEqual(item["prompt"].count("overall_soundscape:"), 1)
            self.assertEqual(item["prompt"].count("non_diegetic_music:"), 1)
            self.assertIn("Clark Kent wears the same blue shirt", item["prompt"])

    def test_compiler_assigns_endpoint_images_to_the_correct_passes(self):
        spans = compute_h3_window_boundaries(345, 124, fps=24, overlap_frames=1)
        plan = {
            "subject_continuity": "The same traveler in the same coat",
            "setting_continuity": "The same overlook at sunrise",
            "visual_continuity": "One continuous tracking shot",
            "initial_state": "The traveler faces the valley",
            "ambient_audio": "steady wind",
            "music": "N/A",
            "windows": [
                {
                    "window": index + 1,
                    "title": f"Beat {index + 1}",
                    "action": f"The traveler performs beat {index + 1}",
                    "dialogue": [],
                    "sound_effects": "N/A",
                    "closing_state": f"The traveler holds position {index + 1}",
                }
                for index in range(3)
            ],
        }
        compiled = compile_h3_window_prompts(
            plan,
            spans,
            has_start_image=True,
            has_end_image=True,
        )
        self.assertIn("<Picture 1>", compiled[0]["prompt"])
        self.assertNotIn("<Picture 2>", compiled[0]["prompt"])
        self.assertIn("<Picture 1>", compiled[1]["prompt"])
        self.assertNotIn("<Picture 2>", compiled[1]["prompt"])
        self.assertIn("<Picture 1>", compiled[-1]["prompt"])
        self.assertIn("<Picture 2>", compiled[-1]["prompt"])

    def test_injected_keyframes_resolve_on_exact_committed_window_spans(self):
        spans = compute_h3_window_boundaries(
            999,
            345,
            fps=24,
            overlap_frames=18,
        )
        resolved = normalize_h3_injected_keyframes(
            [
                {"path": "a.png", "position": "W1:50"},
                {"path": "b.png", "position": "W2:0"},
                {"path": "c.png", "position": "W2:100"},
                {"path": "d.png", "position": "W99:50"},
            ],
            spans,
            fps=24,
        )
        self.assertEqual(
            [item["absolute_frame"] for item in resolved],
            [172, 345, 671, 835],
        )
        self.assertEqual([item["window"] for item in resolved], [1, 2, 2, 3])
        self.assertEqual(resolved[1]["local_frame"], 0)

    def test_compiler_numbers_injected_pictures_after_local_boundaries(self):
        spans = compute_h3_window_boundaries(999, 345, fps=24, overlap_frames=18)
        plan = _fallback_plan("A traveler crosses three rooms", 3)
        resolved = normalize_h3_injected_keyframes(
            [
                {"path": "opening-detail.png", "position": "W1:50"},
                {"path": "middle-door.png", "position": "W2:50"},
                {"path": "final-pose.png", "position": "W3:50"},
            ],
            spans,
            fps=24,
        )
        compiled = compile_h3_window_prompts(
            plan,
            spans,
            has_start_image=True,
            has_end_image=True,
            injected_keyframes=resolved,
        )
        # First pass: start is Picture 1, injection is Picture 2.
        self.assertIn("<Picture 2> is fully referenced as an exact injected frame", compiled[0]["prompt"])
        # Middle pass: continuation boundary is Picture 1, injection is Picture 2.
        self.assertIn("<Picture 2> is fully referenced as an exact injected frame", compiled[1]["prompt"])
        # Final pass: boundary Picture 1 + end Picture 2 + injection Picture 3.
        self.assertIn("<Picture 3> is fully referenced as an exact injected frame", compiled[2]["prompt"])
        self.assertEqual(compiled[2]["injected_keyframes"][0]["picture_index"], 3)

    @patch("services.llm_service.generate")
    def test_planner_compiles_a_valid_llm_storyboard(self, generate):
        source_prompt = "A three-beat continuous action"
        spans = compute_h3_window_boundaries(345, 124, fps=24, overlap_frames=1)
        durations = [item["end_seconds"] - item["start_seconds"] for item in spans]
        ledger = _staged_ledger(source_prompt, 3)
        generate.side_effect = [
            json.dumps(ledger),
            *(
                json.dumps(_staged_segment(
                    index + 1,
                    duration,
                    action=ledger["beats"][index]["description"],
                ))
                for index, duration in enumerate(durations)
            ),
        ]
        result = plan_h3_sliding_windows(
            source_prompt,
            model_type="minimax_h3",
            resolution="1920x1088",
            total_frames=345,
            window_frames=124,
            overlap_frames=1,
            fps=24,
        )
        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(result["window_count"], 3)
        self.assertIn("A three-beat continuous action", result["window_prompts"][0])
        self.assertNotIn("Only action 2", result["window_prompts"][0])
        self.assertIn("Only action 3", result["window_prompts"][-1])
        self.assertEqual(generate.call_count, 4)
        ledger_prompt = generate.call_args_list[0].kwargs["prompt"]
        first_segment_prompt = generate.call_args_list[1].kwargs["prompt"]
        self.assertIn("Segment geometry", ledger_prompt)
        self.assertIn("local duration 0.000", first_segment_prompt)
        self.assertIn("Immutable chronological events", first_segment_prompt)

    @patch("services.llm_service.generate")
    def test_mature_mode_uses_fidelity_note_not_general_enhancer(self, generate):
        generate.side_effect = RuntimeError("offline")
        plan_h3_sliding_windows(
            "A named character completes one requested action",
            model_type="minimax_h3",
            resolution="1920x1088",
            total_frames=345,
            window_frames=124,
            overlap_frames=1,
            fps=24,
            nsfw=True,
        )
        system_prompt = generate.call_args_list[0].kwargs["system_prompt"]
        self.assertIn("SOURCE FIDELITY", system_prompt)
        self.assertIn("MATURE-MODE FIDELITY", system_prompt)
        self.assertIn("Do not censor it, add to it, or intensify it", system_prompt)

    def test_signature_changes_when_timing_or_media_contract_changes(self):
        common = dict(
            prompt="Clark saves a truck",
            model_type="minimax_h3",
            resolution="1920x1088",
            total_frames=345,
            window_frames=124,
            overlap_frames=1,
            discard_frames=0,
            fps=24,
            has_start_image=False,
            has_end_image=False,
        )
        base = h3_window_plan_signature(**common)
        self.assertEqual(base, h3_window_plan_signature(**common))
        self.assertNotEqual(base, h3_window_plan_signature(**{**common, "window_frames": 175}))
        self.assertNotEqual(base, h3_window_plan_signature(**{**common, "has_start_image": True}))
        self.assertNotEqual(
            base,
            h3_window_plan_signature(**common, planning_style="creative"),
        )
        self.assertNotEqual(
            base,
            h3_window_plan_signature(
                **common,
                injected_keyframes=[{"path": "anchor.png", "position": "W2:50"}],
            ),
        )

    def test_reviewed_plan_snapshot_survives_only_matching_geometry(self):
        boundaries = compute_h3_window_boundaries(
            540,
            192,
            fps=24,
            overlap_frames=18,
            discard_frames=0,
        )
        prompts = [f"Exact reviewed prompt {index}." for index in range(1, len(boundaries) + 1)]
        plan = {
            "source_prompt": "Two women browse an outdoor market.",
            "signature": "older-planner-signature",
            "plan_kind": "sliding_window",
            "camera_coverage": "auto",
            "model_type": "minimax_h3",
            "resolution": "864x480",
            "window_frames": 192,
            "windows": [
                {
                    **boundary,
                    "prompt": prompts[index],
                }
                for index, boundary in enumerate(boundaries)
            ],
            "window_prompts": prompts,
        }
        common = {
            "source_prompt": plan["source_prompt"],
            "model_type": plan["model_type"],
            "resolution": plan["resolution"],
            "window_frames": 192,
            "boundaries": boundaries,
            "camera_coverage": "auto",
        }
        self.assertTrue(
            reviewed_h3_window_plan_matches(plan, prompts, **common)
        )
        self.assertFalse(
            reviewed_h3_window_plan_matches(
                plan,
                prompts,
                **{**common, "source_prompt": "A different story."},
            )
        )
        changed_geometry = [dict(item) for item in boundaries]
        changed_geometry[-1]["end_frame"] -= 1
        self.assertFalse(
            reviewed_h3_window_plan_matches(
                plan,
                prompts,
                **{**common, "boundaries": changed_geometry},
            )
        )

    def test_fallback_holds_an_obligation_out_of_the_opening_window(self):
        plan = _fallback_plan(
            "Clark Kent walks down the street in Smallville and has to save a runaway truck",
            3,
        )
        self.assertIn("walks down the street", plan["windows"][0]["action"])
        self.assertNotIn("save a runaway truck", plan["windows"][0]["action"])
        self.assertIn("save a runaway truck", plan["windows"][-1]["action"])

    def test_named_character_rescue_requires_natural_dialogue(self):
        prompt = (
            "Tom Welling is Clark Kent in Smallville when a person attacks "
            "Lana Lang played by Kristin Kreuk. Clark saves her and she "
            "can't believe what she sees."
        )
        self.assertTrue(_narrative_dialogue_expected(prompt, 4))
        self.assertFalse(
            _narrative_dialogue_expected(
                prompt + " The entire sequence is silent and nonverbal.",
                4,
            )
        )

    def test_creative_telling_brief_requires_an_authored_script(self):
        prompt = (
            "George Costanza walks into the coffee shop on Friends and starts "
            "excitedly telling Joey that Maestro version two just dropped. "
            "George explains the new features. Joey has no idea what George "
            "is talking about. Include audience laughter."
        )
        self.assertTrue(_creative_dialogue_expected(prompt, 4))
        self.assertTrue(_narrative_dialogue_expected(prompt, 4))
        self.assertFalse(
            _creative_dialogue_expected(
                "A hawk silently crosses a mountain landscape without dialogue.",
                4,
            )
        )

    def test_compiler_preserves_timed_cuts_inside_each_local_window(self):
        boundaries = compute_h3_window_boundaries(
            480,
            240,
            fps=24,
            overlap_frames=0,
        )
        plan = {
            "subject_continuity": "The two fighters keep the same identities and wardrobe",
            "setting_continuity": "A dark concrete arena",
            "visual_continuity": "Fast live-action screen direction",
            "editing_style": "Dynamic action coverage",
            "initial_state": "The fighters square off in a wide composition",
            "ambient_audio": "Arena room tone",
            "music": "N/A",
            "windows": [
                {
                    "window": 1,
                    "title": "Opening exchange",
                    "coverage": "dynamic multi-shot action",
                    "pacing": "fast real-time action",
                    "shots": [
                        {
                            "shot": 1,
                            "start_seconds": 0,
                            "end_seconds": 4,
                            "transition": "opening composition",
                            "framing": "wide tracking shot",
                            "camera": "truck left with both fighters",
                            "action": "They sprint toward each other",
                            "dialogue": [],
                            "sound_effects": "Rapid footsteps",
                        },
                        {
                            "shot": 2,
                            "start_seconds": 4,
                            "end_seconds": 10,
                            "transition": "hard cut",
                            "framing": "low-angle medium shot",
                            "camera": "short handheld push in",
                            "action": "The first punch lands once",
                            "dialogue": [],
                            "sound_effects": "Single heavy impact",
                        },
                    ],
                    "closing_state": "Both fighters hold at center frame after the impact",
                },
                {
                    "window": 2,
                    "title": "Follow-through",
                    "coverage": "continuous reaction shot",
                    "pacing": "fast real-time action",
                    "shots": [{
                        "shot": 1,
                        "start_seconds": 0,
                        "end_seconds": 10,
                        "transition": "opening composition",
                        "framing": "medium two-shot",
                        "camera": "orbit clockwise and settle",
                        "action": "They recover and separate",
                        "dialogue": [],
                        "sound_effects": "Shoes scrape the floor",
                    }],
                    "closing_state": "They stop at opposite sides of the arena",
                },
            ],
        }
        compiled = compile_h3_window_prompts(plan, boundaries)
        self.assertIn("[Shot 2] At 00:04.000, hard cut", compiled[0]["prompt"])
        self.assertIn("The first punch lands once", compiled[0]["prompt"])
        self.assertNotIn("They recover and separate", compiled[0]["prompt"])
        self.assertIn(
            "Both fighters hold at center frame after the impact",
            compiled[1]["prompt"],
        )

    def test_contract_allows_local_cut_but_rejects_global_timeline_labels(self):
        prompt = (
            "Tom Welling as Clark Kent uses his powers to save Lana Lang "
            "played by Kristin Kreuk."
        )
        bad_plan = {
            "windows": [{
                "shots": [{
                    "start_seconds": 4.0,
                    "end_seconds": 8.0,
                    "transition": "hard cut",
                    "action": "Clark emits a golden energy wave.",
                    "dialogue": [],
                }],
            }],
        }
        violations = _plan_contract_violations(
            prompt,
            bad_plan,
            expect_dialogue=True,
        )
        joined = " ".join(violations)
        self.assertIn("invented unrequested power", joined)
        self.assertNotIn("global window", joined)
        self.assertIn("entirely mute", joined)

        global_plan = {
            "windows": [{
                "shots": [{
                    "action": "At global 10.125s, Clark turns toward Lana.",
                    "dialogue": [],
                }],
            }],
        }
        global_violations = _plan_contract_violations(
            prompt,
            global_plan,
            expect_dialogue=False,
        )
        self.assertIn(
            "global window label/timestamp",
            " ".join(global_violations),
        )

    @patch("services.llm_service.generate")
    def test_planner_repairs_mute_invented_spectacle(self, generate):
        source_prompt = (
            "Tom Welling is Clark Kent in Smallville when a person "
            "attacks Lana Lang played by Kristin Kreuk. Clark uses his "
            "powers to save her and she can't believe what she sees."
        )
        spans = compute_h3_window_boundaries(972, 243, fps=24, overlap_frames=0)
        durations = [item["end_seconds"] - item["start_seconds"] for item in spans]
        repaired_ledger = _staged_ledger(source_prompt, 4, dialogue=True)
        generate.side_effect = [
            json.dumps(_staged_ledger(source_prompt, 4, golden=True, dialogue=False)),
            json.dumps(repaired_ledger),
            *(
                json.dumps(
                    _staged_segment(
                        index + 1,
                        duration,
                        dialogue_id="D1" if index + 1 == 4 else None,
                        action=repaired_ledger["beats"][index]["description"],
                    )
                )
                for index, duration in enumerate(durations)
            ),
        ]
        result = plan_h3_sliding_windows(
            source_prompt,
            model_type="minimax_h3",
            resolution="960x544",
            total_frames=972,
            window_frames=243,
            overlap_frames=0,
            fps=24,
            planning_style="creative",
        )
        self.assertEqual(generate.call_count, 6)
        self.assertEqual(result["planned_by"], "llm")
        repair_prompt = generate.call_args_list[1].kwargs["prompt"]
        self.assertIn("PREVIOUS REJECTED STORY-SCHEDULE JSON", repair_prompt)
        self.assertIn("golden energy", repair_prompt)
        joined = " ".join(result["window_prompts"])
        self.assertNotIn("golden energy", joined)
        self.assertIn(
            "<d>[English] Clark, how did you do that?</d>",
            joined,
        )
        for window_prompt in result["window_prompts"]:
            self.assertEqual(
                window_prompt.count("integrated_multimodal_description:"),
                1,
            )
            self.assertEqual(window_prompt.count("overall_soundscape:"), 1)
            self.assertEqual(window_prompt.count("non_diegetic_music:"), 1)

    def test_ui_and_runtime_use_explicit_prompt_arrays(self):
        handler = (APP / "wgp.py").read_text(encoding="utf-8")
        launch = (APP / "launch.py").read_text(encoding="utf-8")
        store = (ROOT / "ui" / "src" / "stores" / "useStore.ts").read_text(encoding="utf-8")
        advanced = (ROOT / "ui" / "src" / "components" / "Sidebar" / "AdvancedSettings.tsx").read_text(encoding="utf-8")
        multi_window = (ROOT / "ui" / "src" / "components" / "Sidebar" / "H3MultiWindowControls.tsx").read_text(encoding="utf-8")
        prompt_input = (ROOT / "ui" / "src" / "components" / "Sidebar" / "PromptInput.tsx").read_text(encoding="utf-8")
        main_content = (ROOT / "ui" / "src" / "components" / "MainContent" / "MainContent.tsx").read_text(encoding="utf-8")
        media_item = (ROOT / "ui" / "src" / "components" / "MainContent" / "MediaFeedItem.tsx").read_text(encoding="utf-8")
        guide = APP / "services" / "llm_guides" / "enhance" / "minimax_h3_sliding_windows.md"
        self.assertIn("h3_window_prompts=None", handler)
        self.assertIn("Using {len(prompts)} explicit", handler)
        self.assertIn('/api/v1/llm/plan-h3-windows', launch)
        self.assertIn("h3_window_plan_signature", launch)
        self.assertIn("api.planH3Windows", store)
        self.assertIn("params._h3_window_plan_reviewed = true", store)
        self.assertIn("preserve_reviewed_h3_plan", launch)
        self.assertIn("planner LLM bypassed", launch)
        self.assertNotIn("Plan Prompt Across Windows", advanced)
        self.assertIn("Window prompts", multi_window)
        self.assertIn("Exact H3 prompts", prompt_input)
        self.assertIn("H3WindowPromptTextarea", prompt_input)
        self.assertIn("Generated window prompts", media_item)
        self.assertIn("AI window 1", media_item)
        self.assertIn("textarea.scrollHeight", prompt_input)
        self.assertNotIn("max-h-[360px] overflow-y-auto", prompt_input)
        self.assertNotIn("min-h-[118px] resize-y", prompt_input)
        self.assertIn("Exact H3 prompt", main_content)
        self.assertIn('"h3_window_plan"', launch)
        self.assertTrue(guide.is_file())
        guide_text = guide.read_text(encoding="utf-8")
        self.assertIn("SOURCE FIDELITY", guide_text)
        self.assertIn("prompt-local clock", guide_text)
        self.assertIn("energy wave/pulse/blast", guide_text)
        self.assertIn("own complete Context-IR prompt", guide_text)
        self.assertIn("minimax_h3_window_storyboard: true", store)


if __name__ == "__main__":
    unittest.main()
