"""Regressions for the shared staged MiniMax H3 story contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_story_ledger import (  # noqa: E402
    _canonicalize_segment_contract,
    _canonicalize_story_ledger,
    _coalesce_camera_phases,
    _creative_conversation_brief,
    _deterministic_ledger,
    _dialogue_catalog,
    _dialogue_word_count,
    _expected_dialogue_events,
    _fallback_segment,
    _ledger_schema,
    _materialize_segment,
    _materialized_segment_violations,
    _only_supplied_dialogue_requested,
    _apply_faithful_treatment,
    _prepare_render_dialogue_schedule,
    _repair_materialized_segment_staging,
    _split_h3_shots_at_speaker_changes,
    extract_h3_source_intent,
    extract_locked_dialogue,
    extract_source_events,
    ledger_violations,
    plan_h3_story_segments,
    sanitize_h3_nonverbal_audio,
    sanitize_h3_prompt_text,
    segment_violations,
)
from services.h3_window_planner import (  # noqa: E402
    compile_h3_window_prompts,
    compute_h3_window_boundaries,
)


def _ledger() -> dict:
    return {
        "subject_continuity": 'Superman and Thanos remain unchanged {"S1": "Superman"}',
        "setting_continuity": "A damaged dark concrete structure",
        "visual_continuity": "Dark live-action cinematic realism",
        "editing_style": "Fast motivated action coverage",
        "initial_state": "Superman and Thanos face each other",
        "ambient_audio": "Wind and settling concrete dust",
        "music": "N/A",
        "required_final_outcome": "Thanos refuses Superman's demand",
        "beats": [
            {
                "beat_id": "B1",
                "segment": 1,
                "description": "Superman stands firm and demands the gauntlet",
                "source_event_ids": ["E1"],
                "dialogue_ids": ["D1"],
                "state_after": "Superman holds his ground while Thanos watches",
                "sound_effects": "Wind and settling grit",
            },
            {
                "beat_id": "B2",
                "segment": 2,
                "description": "Thanos raises the gauntlet and refuses",
                "source_event_ids": ["E2"],
                "dialogue_ids": ["D2"],
                "state_after": "Thanos holds the raised gauntlet toward Superman",
                "sound_effects": "The stones emit a low nonverbal hum",
            },
        ],
        "generated_dialogue": [],
    }


def _segment(number: int, *, duration: float = 10.0) -> dict:
    return {
        "segment": number,
        "title": f"Beat {number}",
        "opening_state": "The required opening composition",
        "coverage": "dynamic cinematic coverage",
        "pacing": "fast real-time action",
        "shots": [{
            "shot": 1,
            "start_seconds": 0.0,
            "end_seconds": duration,
            "transition": "opening composition",
            "framing": "medium two-shot",
            "camera": "a short motivated push in",
            "beat_ids": [f"B{number}"],
            "action": (
                "Superman stands firm and demands the gauntlet"
                if number == 1 else "Thanos raises the gauntlet and refuses"
            ),
            "dialogue": [{
                "dialogue_id": f"D{number}",
                "delivery": "calm and clear" if number == 1 else "breathy and resolute",
                "action": "holding eye contact",
            }],
            "sound_effects": "Natural synchronized effects",
        }],
        "closing_state": f"Proposed closing state {number}",
    }


class H3StoryLedgerTests(unittest.TestCase):
    def setUp(self):
        self.prompt = (
            'Superman says calmly, "Enough. Give me the glove, Thanos." '
            'Thanos raises his left hand and says in a breathy voice, "I am inevitable."'
        )
        self.locked = extract_locked_dialogue(self.prompt)

    def test_creative_dialogue_schema_cannot_return_an_empty_script(self):
        creative = _ledger_schema(
            4,
            source_event_count=3,
            locked_dialogue_count=0,
            allow_generated_dialogue=True,
        )
        faithful = _ledger_schema(
            4,
            source_event_count=3,
            locked_dialogue_count=0,
            allow_generated_dialogue=False,
        )
        self.assertEqual(
            creative["properties"]["generated_dialogue"]["minItems"],
            1,
        )
        self.assertEqual(
            faithful["properties"]["generated_dialogue"]["maxItems"],
            0,
        )

    def test_reference_story_filters_creation_directive_and_keeps_single_names(self):
        prompt = (
            "Make a scene from Friends. Blaine walks into Central Perk and sits "
            "between Ross and Rachel. Both Ross and Rachel look at Blaine."
        )

        events = extract_source_events(prompt)
        intent = extract_h3_source_intent(prompt)

        self.assertNotIn("Make a scene", json.dumps(events))
        self.assertEqual(intent["proper_names"], ["Blaine", "Central Perk", "Ross", "Rachel"])
        self.assertEqual(intent["cast_names"], ["Blaine", "Ross", "Rachel"])
        self.assertIn("exactly one identity instance", intent["cast_cardinality_contract"])
        self.assertIn("Ross - Blaine - Rachel", intent["blocking_contract"])

    def test_reference_and_prompt_native_cast_are_kept_in_one_exact_contract(self):
        prompt = (
            "Blaine walks into the coffee shop and sits on the couch between "
            "Ross and Rachel. Rachel looks at Blaine."
        )
        ledger = _deterministic_ledger(
            prompt,
            segment_count=2,
            segment_durations=[10.0, 10.0],
            locked_dialogue=[],
            camera_coverage="multi_shot",
            reference_context=(
                "<Subject 1> is Blaine from <Picture 1>, preserving identity. "
                "<Audio 1> is the voice-timbre reference for <Subject 1>."
            ),
        )

        continuity = ledger["subject_continuity"]
        self.assertIn("<Subject 1> is Blaine", continuity)
        self.assertIn("Ross, Rachel are named prompt-native", continuity)
        self.assertIn("Blaine, Ross, Rachel", continuity)
        self.assertIn("Ross - Blaine - Rachel", continuity)
        self.assertNotIn("Central Perk", continuity)
        self.assertIn("Before Blaine sits", ledger["initial_state"])

    def test_actor_and_role_are_one_principal_but_other_cast_remain_distinct(self):
        intent = extract_h3_source_intent(
            "Henry Cavill as Superman fights Thanos. Thanos punches Superman."
        )
        self.assertEqual(
            intent["cast_names"],
            ["Henry Cavill as Superman", "Thanos"],
        )

    def test_fallback_gives_compound_entrance_and_blocking_time_to_read(self):
        beats = [
            {
                "beat_id": "B1",
                "description": (
                    "Blaine walks into the coffee shop and sits down between "
                    "Ross and Rachel"
                ),
                "source_event_ids": ["E1"],
                "dialogue_ids": [],
                "state_after": "Blaine is seated between Ross and Rachel",
            },
            {
                "beat_id": "B2",
                "description": "Blaine breathes a sigh of relief",
                "source_event_ids": ["E2"],
                "dialogue_ids": [],
                "state_after": "Blaine relaxes into the couch",
            },
            {
                "beat_id": "B3",
                "description": "Ross and Rachel look at Blaine in confusion",
                "source_event_ids": ["E3"],
                "dialogue_ids": [],
                "state_after": "Ross and Rachel watch Blaine",
            },
            {
                "beat_id": "B4",
                "description": "Rachel asks Blaine whether they can help him",
                "source_event_ids": ["E4"],
                "dialogue_ids": ["D1"],
                "state_after": "Rachel waits for Blaine's answer",
            },
        ]
        segment = _fallback_segment(
            1,
            duration=13.667,
            beats=beats,
            opening_state="Ross and Rachel sit with an empty place between them",
            camera_coverage="multi_shot",
            dialogue_catalog=[{
                "dialogue_id": "D1",
                "speaker": "Rachel",
                "text": "Uh, can we help you?",
                "delivery": "confused but polite",
            }],
            source_intent={},
        )

        shots = segment["shots"]
        durations = [
            float(shot["end_seconds"]) - float(shot["start_seconds"])
            for shot in shots
        ]
        self.assertGreaterEqual(durations[0], 4.5)
        self.assertGreater(durations[0], durations[1])
        self.assertGreater(durations[0], durations[2])
        self.assertEqual(shots[0]["start_seconds"], 0.0)
        self.assertEqual(shots[-1]["end_seconds"], 13.667)
        for previous, following in zip(shots, shots[1:]):
            self.assertEqual(previous["end_seconds"], following["start_seconds"])

        # The same protection applies to a valid LLM camera plan that tried to
        # squeeze the compound entrance into its first 2.5 seconds.
        proposed = json.loads(json.dumps(segment))
        proposed_edges = [(0.0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 13.667)]
        for shot, (start, end) in zip(proposed["shots"], proposed_edges):
            shot["start_seconds"] = start
            shot["end_seconds"] = end
        canonical = _canonicalize_segment_contract(
            proposed,
            segment_number=1,
            duration=13.667,
            assigned_beats=beats,
            dialogue_catalog=[{
                "dialogue_id": "D1",
                "speaker": "Rachel",
                "text": "Uh, can we help you?",
                "delivery": "confused but polite",
            }],
            opening_state="Ross and Rachel sit with an empty place between them",
            source_intent={},
        )
        self.assertIsNotNone(canonical)
        canonical_first = canonical["shots"][0]
        self.assertGreaterEqual(
            canonical_first["end_seconds"] - canonical_first["start_seconds"],
            4.5,
        )

    def test_long_exact_turn_is_fragmented_across_adjacent_windows(self):
        prompt = (
            "Make a scene from Friends. Blaine walks into the coffee shop, sits "
            "between Ross and Rachel, and breathes a sigh of relief. Rachel asks, "
            '"Uh, can we help you?" Blaine responds, "Oh, hey, yeah, glad you '
            "asked! Maestro version two just dropped and has so many cool new "
            "features. Like this one. You can save characters and cast them into "
            'scenes with anyone. Oh! And there is a new Editor." Ross responds, '
            '"Uh, who are you?" The audience laughs.'
        )
        durations = [13.667, 12.917]
        locked = extract_locked_dialogue(prompt)
        events = extract_source_events(prompt)
        ledger = _deterministic_ledger(
            prompt,
            segment_count=2,
            segment_durations=durations,
            locked_dialogue=locked,
            camera_coverage="multi_shot",
            reference_context="Blaine remains the saved reference character",
        )
        catalog = _dialogue_catalog(ledger, locked)

        render_beats, render_catalog, _render_events, fragments = (
            _prepare_render_dialogue_schedule(
                ledger["beats"],
                catalog,
                segment_durations=durations,
                source_events=events,
                expected_dialogue_events=_expected_dialogue_events(prompt, locked),
            )
        )

        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0]["source_dialogue_id"], "D2")
        self.assertEqual(fragments[0]["segments"], [1, 2])
        pieces = [
            item["text"] for item in render_catalog
            if item.get("source_dialogue_id") == "D2"
        ]
        self.assertEqual(" ".join(pieces), locked[1]["text"])
        segment_by_dialogue = {
            str(dialogue_id): int(beat.get("segment") or 0)
            for beat in render_beats
            for dialogue_id in (beat.get("dialogue_ids") or [])
        }
        dialogue_by_segment = {
            segment: sum(
                _dialogue_word_count(item.get("text"))
                for item in render_catalog
                if int(
                    item.get("segment")
                    or segment_by_dialogue.get(str(item.get("dialogue_id") or ""), 0)
                ) == segment
            )
            for segment in (1, 2)
        }
        self.assertLessEqual(dialogue_by_segment[1], int(durations[0] * 2.1))
        self.assertLessEqual(dialogue_by_segment[2], int(durations[1] * 2.1))
        self.assertEqual(
            [
                event_id
                for beat in render_beats
                for event_id in beat.get("source_event_ids") or []
            ],
            [item["event_id"] for item in events],
        )

    def test_long_exact_turn_borrows_headroom_for_a_clean_sentence_boundary(self):
        prompt = (
            "George Costanza walks into the coffee shop and walks up to Joey. "
            'George says "Maestro two is out!" Joey says "What, who?" '
            'George says "It is crazy! You can now generate videos up to an hour '
            "with a single prompt! You can save and cast Characters just like "
            "Sora 2's Cameos, and it has push notifications. It even has Qwen "
            '3.8! And get this. It even has an editor!" Joey replies "Cool, '
            'so, who are you again?"'
        )
        durations = [14.375, 13.625, 13.625]
        locked = extract_locked_dialogue(prompt)
        events = extract_source_events(prompt)
        ledger = _deterministic_ledger(
            prompt,
            segment_count=3,
            segment_durations=durations,
            locked_dialogue=locked,
            camera_coverage="multi_shot",
            reference_context="",
        )
        catalog = _dialogue_catalog(ledger, locked)

        _render_beats, render_catalog, _render_events, fragments = (
            _prepare_render_dialogue_schedule(
                ledger["beats"],
                catalog,
                segment_durations=durations,
                source_events=events,
                expected_dialogue_events=_expected_dialogue_events(prompt, locked),
            )
        )

        self.assertEqual(fragments[0]["source_dialogue_id"], "D3")
        pieces = [
            item["text"] for item in render_catalog
            if item.get("source_dialogue_id") == "D3"
        ]
        self.assertEqual(len(pieces), 2)
        self.assertTrue(pieces[0].endswith("single prompt!"))
        self.assertTrue(pieces[1].startswith("You can save and cast Characters"))
        self.assertIn("Sora 2's Cameos", pieces[1])
        self.assertIn("Qwen 3.8", pieces[1])
        self.assertEqual(" ".join(pieces), locked[2]["text"])

    def test_conversation_schema_can_require_one_authored_line_per_segment(self):
        schema = _ledger_schema(
            4,
            source_event_count=4,
            locked_dialogue_count=0,
            allow_generated_dialogue=True,
            minimum_generated_dialogue=4,
        )
        self.assertEqual(
            schema["properties"]["generated_dialogue"]["minItems"],
            4,
        )

    def test_background_chatter_is_removed_without_losing_nonverbal_ambience(self):
        cleaned = sanitize_h3_nonverbal_audio(
            "Soft jazz piano, gentle clinking of coffee cups, and low-level "
            "murmur of indistinct background chatter."
        )
        self.assertIn("Soft jazz piano", cleaned)
        self.assertIn("clinking of coffee cups", cleaned)
        self.assertNotIn("murmur", cleaned.casefold())
        self.assertNotIn("chatter", cleaned.casefold())
        acoustic = sanitize_h3_nonverbal_audio(
            "Swamp insects; Character voices sound natural in the environment."
        )
        self.assertIn("Swamp insects", acoustic)
        self.assertIn("Character voices sound natural", acoustic)

    def test_creative_conversation_spreads_authored_lines_from_segment_one(self):
        prompt = (
            "George Costanza walks into Central Perk and starts excitedly "
            "telling Joey that Maestro version two just dropped. George "
            "explains its new features. Joey has no idea what George is "
            "talking about. Include audience laughter."
        )
        self.assertTrue(_creative_conversation_brief(prompt))
        candidate = _deterministic_ledger(
            prompt,
            segment_count=4,
            segment_durations=[10.0] * 4,
            locked_dialogue=[],
            camera_coverage="multi_shot",
            reference_context="",
        )
        candidate["ambient_audio"] = (
            "Soft jazz piano, clinking cups, and indistinct background chatter"
        )
        candidate["generated_dialogue"] = [
            {
                "speaker": "George",
                "language": "English",
                "delivery": "frantic and excited",
                "text": "Joey, Maestro version two just dropped!",
                "segment": 2,
            },
            {
                "speaker": "George",
                "language": "English",
                "delivery": "rapid-fire enthusiasm",
                "text": "It can make much longer videos now.",
                "segment": 3,
            },
            {
                "speaker": "George",
                "language": "English",
                "delivery": "proudly",
                "text": "There is even a full editor.",
                "segment": 4,
            },
            {
                "speaker": "Joey",
                "language": "English",
                "delivery": "completely confused",
                "text": "Is Maestro the guy who makes the coffee?",
                "segment": 4,
            },
        ]
        responses = iter([
            json.dumps(candidate),
            *(json.dumps(_segment(index, duration=10.0)) for index in range(1, 5)),
        ])

        result = plan_h3_story_segments(
            prompt,
            segment_durations=[10.0] * 4,
            mode="sliding_window",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            planning_style="creative",
            llm_generate=lambda **_kwargs: next(responses),
        )

        self.assertEqual(
            [item["segment"] for item in result["ledger"]["generated_dialogue"]],
            [1, 2, 3, 4],
        )
        for segment in result["segments"]:
            dialogue = [
                line
                for shot in segment["shots"]
                for line in shot.get("dialogue") or []
            ]
            self.assertEqual(len(dialogue), 1)
        self.assertNotIn(
            "chatter",
            result["ledger"]["ambient_audio"].casefold(),
        )
        compiled = compile_h3_window_prompts(
            {
                **{
                    key: result["ledger"].get(key, "")
                    for key in (
                        "subject_continuity",
                        "setting_continuity",
                        "visual_continuity",
                        "editing_style",
                        "initial_state",
                        "ambient_audio",
                        "music",
                    )
                },
                "windows": result["segments"],
            },
            compute_h3_window_boundaries(
                960,
                240,
                fps=24,
                overlap_frames=0,
            ),
        )
        for window in compiled:
            self.assertIn("<d>[English]", window["prompt"])
            self.assertNotIn("background chatter", window["prompt"].casefold())

    def test_user_quotes_are_locked_with_speakers_and_order(self):
        self.assertEqual(
            [(item["dialogue_id"], item["speaker"], item["text"]) for item in self.locked],
            [
                ("D1", "Superman", "Enough. Give me the glove, Thanos."),
                ("D2", "Thanos", "I am inevitable."),
            ],
        )

    def test_screenplay_rows_are_locked_dialogue_not_visual_events(self):
        prompt = (
            "George Costanza walks into the coffee shop on the TV show Friends. "
            "Starts passionately talking to Joey.\n\n"
            "Joey sits on the couch eating a muffin wearing a grey sweatshirt. "
            "George Costanza bursts through the door, frantic, wearing a dark brown sport coat.\n\n"
            "GEORGE: Joey! Maestro 2.0! It's here!\n\n"
            "JOEY: Do I know you?\n\n"
            "GEORGE (excitedly): Forget who I am! There's an Editor now!\n\n"
            "Everyone looks over.\n"
            "JOEY: Who are you?"
        )

        locked = extract_locked_dialogue(prompt)
        self.assertEqual(
            [(item["speaker"], item["text"]) for item in locked],
            [
                ("GEORGE", "Joey! Maestro 2.0! It's here!"),
                ("JOEY", "Do I know you?"),
                ("GEORGE", "Forget who I am! There's an Editor now!"),
                ("JOEY", "Who are you?"),
            ],
        )
        self.assertEqual(locked[2]["delivery"], "speaks excitedly")
        self.assertTrue(all(item["source_form"] == "screenplay" for item in locked))

        events = extract_source_events(prompt)
        event_text = " | ".join(item["text"] for item in events)
        self.assertNotIn("Starts passionately talking", event_text)
        self.assertNotIn("walks into the coffee shop", event_text)
        self.assertEqual(event_text.count("bursts through the door"), 1)
        for spoken in ("Maestro 2.0", "Do I know you", "Forget who I am", "Who are you"):
            self.assertNotIn(spoken, event_text)

        intent = extract_h3_source_intent(prompt)
        self.assertEqual(intent["cast_names"], ["George Costanza", "Joey"])
        self.assertEqual(intent["opening_dialogue_id"], "D1")
        self.assertFalse(intent["fast_action"])
        self.assertTrue(intent["energetic_performance"])
        self.assertNotIn("bursts through", intent["style_contract"])

    def test_screenplay_dialogue_is_mandatory_and_entrance_is_not_persistent(self):
        prompt = (
            "George Costanza walks into the coffee shop on the TV show Friends. "
            "Starts passionately talking to Joey.\n\n"
            "Joey sits on the couch eating a muffin. George Costanza bursts "
            "through the door, frantic, wearing a dark brown sport coat.\n\n"
            "GEORGE: Joey! Maestro 2.0 is here!\n"
            "JOEY: Are you selling me cable?\n"
            "GEORGE: No! It has an Editor now!\n"
            "JOEY: Okay, seriously, who are you?"
        )
        locked = extract_locked_dialogue(prompt)
        canonical = _deterministic_ledger(
            prompt,
            segment_count=3,
            segment_durations=[14.375, 14.375, 13.25],
            locked_dialogue=locked,
            camera_coverage="multi_shot",
            reference_context="",
        )
        candidate = json.loads(json.dumps(canonical))
        candidate["setting_continuity"] = (
            "George Costanza and Joey are already seated opposite each other"
        )
        candidate["visual_continuity"] = (
            "George Costanza bursts through the door in every segment"
        )
        calls = 0

        def planned_then_offline(**_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return json.dumps(candidate)
            raise RuntimeError("offline")

        # Deliberately pass the legacy false value: the shared planner must
        # discover screenplay-form dialogue on its own.
        result = plan_h3_story_segments(
            prompt,
            segment_durations=[14.375, 14.375, 13.25],
            mode="sliding_window",
            camera_coverage="multi_shot",
            expect_dialogue=False,
            planning_style="faithful",
            llm_generate=planned_then_offline,
        )

        rendered_dialogue = [
            (line["speaker"], line["text"])
            for segment in result["segments"]
            for shot in segment["shots"]
            for line in (shot.get("dialogue") or [])
        ]
        self.assertEqual(
            rendered_dialogue,
            [
                ("George Costanza", "Joey! Maestro 2.0 is here!"),
                ("Joey", "Are you selling me cable?"),
                ("George Costanza", "No! It has an Editor now!"),
                ("Joey", "Okay, seriously, who are you?"),
            ],
        )
        self.assertEqual(
            result["ledger"]["setting_continuity"],
            canonical["setting_continuity"],
        )
        self.assertNotIn(
            "bursts through the door in every segment",
            result["ledger"]["visual_continuity"],
        )
        first_lines = [
            line
            for shot in result["segments"][0]["shots"]
            for line in (shot.get("dialogue") or [])
        ]
        self.assertTrue(first_lines)
        self.assertEqual(first_lines[0]["dialogue_id"], "D1")

    def test_spaced_h3_tags_lock_speakers_and_leave_only_visual_events(self):
        prompt = (
            "Yoda waits in the Dagobah swamp. Thanos says to Yoda, "
            "< d>First line. Second sentence.</d>"
            "Yoda waves his hand while saying, <d>Powerful, it is.</d> "
            "Thanos responds <d>As all things should be.</d>. "
            "Atmospheric ambiance. Character voices sound natural in the environment. "
            "Camera pans to Blaine, who waves and says "
            "<d>Hey guys, check out Maestro.</d>. "
            "Thanos snaps his fingers. Blaine turns to dust."
        )
        locked = extract_locked_dialogue(prompt)
        self.assertEqual(
            [(item["speaker"], item["text"]) for item in locked],
            [
                ("Thanos", "First line. Second sentence."),
                ("Yoda", "Powerful, it is."),
                ("Thanos", "As all things should be."),
                ("Blaine", "Hey guys, check out Maestro."),
            ],
        )
        events = " | ".join(item["text"] for item in extract_source_events(prompt))
        for spoken in ("First line", "Powerful", "all things", "Hey guys"):
            self.assertNotIn(spoken, events)
        self.assertNotIn("Atmospheric ambiance", events)
        self.assertNotIn("Character voices sound natural", events)
        self.assertIn("Thanos snaps his fingers", events)
        self.assertIn("Blaine turns to dust", events)

    def test_quoted_title_is_not_misclassified_as_spoken_dialogue(self):
        prompt = 'A sitcom episode titled "The One With the Broken Robot" follows Alex repairing it.'
        self.assertEqual(extract_locked_dialogue(prompt), [])
        rendered = " ".join(item["text"] for item in extract_source_events(prompt))
        self.assertIn("The One With the Broken Robot", rendered)

    def test_pov_identity_and_opening_pose_are_one_source_event(self):
        events = extract_source_events(
            "POV: The viewer is Harry Potter as he stands on top of a scenic mountain. "
            "Hermione waits beside him."
        )
        self.assertEqual(
            events[0]["text"],
            "POV: The viewer is Harry Potter as he stands on top of a scenic mountain",
        )
        self.assertFalse(any(item["text"] == "he stands on top of a scenic mountain" for item in events))

    def test_ledger_rejects_repeated_events_and_dialogue(self):
        ledger = _ledger()
        ledger["beats"][1]["description"] = ledger["beats"][0]["description"]
        ledger["beats"][1]["dialogue_ids"] = ["D1", "D2"]
        violations = ledger_violations(
            self.prompt,
            ledger,
            segment_count=2,
            locked_dialogue=self.locked,
            expect_dialogue=True,
        )
        joined = " ".join(violations)
        self.assertIn("story event is duplicated", joined)
        self.assertIn("dialogue IDs are missing, duplicated", joined)

    def test_source_events_keep_the_requested_ending_and_drop_style_fragments(self):
        prompt = (
            "Superman and Thanos are trading punches. High-speed action movie dynamic superhero fight scenes. "
            "Superman punches Thanos through a wall. Superman looks at the gauntlet, and heat vision cuts off "
            "Thanos's arm, and Thanos yells as his knees buckle in defeat. Dark Cinematic. Sniderverse style. "
            "rated-r graphic, realistic film scene."
        )
        events = extract_source_events(prompt)
        rendered = " | ".join(item["text"] for item in events)
        self.assertIn("heat vision cuts off Thanos's arm", rendered)
        self.assertIn("Thanos yells as his knees buckle in defeat", events[-1]["text"])
        self.assertNotIn("Dark Cinematic", rendered)
        self.assertNotIn("Sniderverse style", rendered)
        self.assertNotIn("rated-r", rendered)

    def test_source_events_drop_orphaned_character_name_from_compound_action(self):
        prompt = (
            "Blaine waves. Thanos, while standing in the swamp near Yoda, "
            "snaps his fingers. Blaine turns to dust."
        )
        events = extract_source_events(prompt)
        rendered = [item["text"] for item in events]
        self.assertNotIn("Thanos", rendered)
        self.assertTrue(any("Thanos keeps standing" in item for item in rendered))

    def test_locked_dialogue_counts_toward_each_segment_timing_budget(self):
        ledger = _ledger()
        ledger["beats"][0]["dialogue_ids"] = ["D1", "D2"]
        ledger["beats"][1]["dialogue_ids"] = []
        violations = ledger_violations(
            self.prompt,
            ledger,
            segment_count=2,
            locked_dialogue=self.locked,
            expect_dialogue=True,
            segment_durations=[4.0, 16.0],
        )
        self.assertTrue(any("segment 1 dialogue uses" in item for item in violations))

    def test_segment_rejects_tiny_tail_and_repeated_or_foreign_beats(self):
        segment = _segment(1)
        segment["shots"] = [
            {
                **segment["shots"][0],
                "end_seconds": 9.9,
            },
            {
                **segment["shots"][0],
                "shot": 2,
                "start_seconds": 9.9,
                "end_seconds": 10.0,
                "transition": "hard cut",
                "beat_ids": ["B2"],
                "dialogue": [],
            },
        ]
        violations = segment_violations(
            self.prompt,
            segment,
            segment_number=1,
            duration=10.0,
            assigned_beats=[_ledger()["beats"][0]],
            dialogue_catalog=self.locked,
        )
        joined = " ".join(violations)
        self.assertIn("assigned beat IDs are missing, foreign, or repeated", joined)
        self.assertIn("unusably short tail shot", joined)

    def test_segment_rejects_unrequested_window_word_but_preserves_user_literal(self):
        segment = _segment(1)
        segment["shots"][0]["action"] = (
            "Superman stands firm at the next window and demands the gauntlet"
        )
        segment["closing_state"] = "Superman waits beside the window"
        kwargs = {
            "segment_number": 1,
            "duration": 10.0,
            "assigned_beats": [_ledger()["beats"][0]],
            "dialogue_catalog": self.locked,
        }

        violations = segment_violations(self.prompt, segment, **kwargs)
        self.assertIn(
            "introduced the internal term 'window' as visible scene content",
            violations,
        )

        literal_prompt = self.prompt + " Superman stands beside a stained-glass window."
        literal_violations = segment_violations(literal_prompt, segment, **kwargs)
        self.assertNotIn(
            "introduced the internal term 'window' as visible scene content",
            literal_violations,
        )

    def test_final_prompt_camera_phases_split_when_visible_speaker_changes(self):
        shots = [{
            "shot": 4,
            "start_seconds": 5.8,
            "end_seconds": 14.375,
            "transition": "hard cut",
            "framing": "a motivated medium reaction angle",
            "camera": "a coherent motivated camera follows the visible action",
            "action": (
                "Visual direction only, never spoken narration: Joey visibly "
                "delivers the assigned dialogue line. Then George Costanza "
                "visibly begins the assigned response"
            ),
            "dialogue": [
                {
                    "speaker": "Joey",
                    "speaker_id": "S2",
                    "text": "What, who?",
                    "action": "only Joey's mouth moves",
                },
                {
                    "speaker": "George Costanza",
                    "speaker_id": "S1",
                    "text": "It is crazy! Maestro now has an editor.",
                    "action": "only George Costanza's mouth moves",
                },
            ],
            "sound_effects": "Natural synchronized effects",
        }]

        split = _split_h3_shots_at_speaker_changes(
            shots,
            known_speakers=["George Costanza", "Joey"],
        )

        self.assertEqual(len(split), 2)
        self.assertEqual(
            [[line["speaker"] for line in shot["dialogue"]] for shot in split],
            [["Joey"], ["George Costanza"]],
        )
        self.assertAlmostEqual(split[0]["end_seconds"], split[1]["start_seconds"], places=3)
        self.assertEqual(split[1]["end_seconds"], 14.375)
        self.assertIn("Joey visibly", split[0]["action"])
        self.assertNotIn("George Costanza visibly", split[0]["action"])
        self.assertIn("George Costanza visibly", split[1]["action"])
        self.assertNotIn("Joey visibly", split[1]["action"])

    def test_camera_planner_merges_continuous_silent_setup_before_speaker_turns(self):
        phases = [
            {
                "beat_id": "B1",
                "source_event_ids": ["E1"],
                "dialogue_ids": [],
                "description": "George enters the coffee shop",
                "state_after": "George is inside near the entrance",
                "sound_effects": "Door bell",
            },
            {
                "beat_id": "B2",
                "source_event_ids": ["E2"],
                "dialogue_ids": [],
                "description": "George walks to Joey on the couch",
                "state_after": "George stands beside Joey",
                "sound_effects": "Footsteps",
            },
            *[
                {
                    "beat_id": f"B{index + 3}",
                    "source_event_ids": [f"E{index + 3}"],
                    "dialogue_ids": [f"D{index + 1}"],
                    "description": f"Speaker turn {index + 1}",
                    "state_after": f"Speaker turn {index + 1} is complete",
                    "sound_effects": "Room tone",
                }
                for index in range(3)
            ],
        ]

        fitted = _coalesce_camera_phases(phases, target_count=4)

        self.assertEqual(len(fitted), 4)
        self.assertEqual(fitted[0]["source_event_ids"], ["E1", "E2"])
        self.assertEqual(fitted[0]["dialogue_ids"], [])
        self.assertIn("George enters", fitted[0]["description"])
        self.assertIn("George walks", fitted[0]["description"])
        self.assertEqual(
            [item["dialogue_ids"] for item in fitted[1:]],
            [["D1"], ["D2"], ["D3"]],
        )

    def test_reported_george_joey_dwight_faithful_plan_has_no_id_repair(self):
        prompt = (
            "George Costanza walks into the coffee shop on the TV show Friends, "
            "from the outside, and walks up to Joey, who is sitting on the couch. "
            'George passionately says "Maestro two is out!" '
            'Joey says "Wha, who?" George says "It is crazy! You can now generate '
            "videos up to an hour with a single prompt! You can save and cast "
            "Characters just like Sora, and it has push notifications! It has "
            "Qwen 3.8! It has the latest turbo LoRAs. And get this. It even has "
            'an editor!" Joey replies "Wow, cool. Um, who are you again?" '
            "Camera pans to Dwight from The Office, who is also in the Friends "
            'coffee shop, and Dwight says with frustration "Ugh, Joey, this is '
            'George. George—Joey" as he introduces them. Dwight then muffles '
            'softly "I hate A.I."'
        )
        calls: list[dict] = []

        def generate(**kwargs):
            calls.append(kwargs)
            schema = kwargs["json_schema"]
            if "setting_continuity" in schema.get("properties", {}):
                return json.dumps({
                    "setting_continuity": "The same busy Friends coffee shop",
                    "visual_continuity": "Warm multi-camera sitcom realism",
                    "editing_style": "Motivated speaker coverage and reaction cuts",
                    "ambient_audio": "Coffee cups, footsteps, and room tone",
                })
            segment_number = schema["properties"]["segment"]["minimum"]
            maximum_shots = schema["properties"]["shots"]["maxItems"]
            match = kwargs["prompt"].split(
                "Immutable chronological events (depict each once, in order):\n",
                1,
            )[1].split(
                "\n\nImmutable dialogue performances",
                1,
            )[0]
            beat_count = len(json.loads(match))
            shot_count = min(maximum_shots, max(1, beat_count))
            duration = [14.375, 13.625, 13.625][segment_number - 1]
            shots = []
            for index in range(shot_count):
                shots.append({
                    "shot": index + 1,
                    "start_seconds": duration * index / shot_count,
                    "end_seconds": duration * (index + 1) / shot_count,
                    "transition": "opening composition" if index == 0 else "hard cut",
                    "framing": "cinematic medium scene composition",
                    "camera": "a motivated camera follows the active performance",
                    "action": "The assigned visible event advances",
                    "sound_effects": "Natural synchronized effects",
                })
            return json.dumps({
                "segment": segment_number,
                "title": f"Segment {segment_number}",
                "opening_state": "The supplied opening state",
                "coverage": "motivated multi-shot coverage",
                "pacing": "brisk natural pacing",
                "shots": shots,
                "closing_state": "The assigned visible result holds",
            })

        result = plan_h3_story_segments(
            prompt,
            segment_durations=[14.375, 13.625, 13.625],
            mode="sliding_window",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            planning_style="faithful",
            llm_generate=generate,
        )

        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(result["planning_warnings"], [])
        self.assertEqual(result["planning_diagnostics"], [])
        self.assertNotIn("beats", calls[0]["json_schema"]["properties"])
        self.assertNotIn("MANDATORY OUTPUT CHECKSUM", calls[0]["prompt"])
        self.assertLessEqual(len(result["segments"][0]["shots"]), 4)
        rendered_dialogue = [
            (line["dialogue_id"], line["speaker"])
            for segment in result["segments"]
            for shot in segment["shots"]
            for line in shot.get("dialogue") or []
        ]
        self.assertEqual(
            rendered_dialogue,
            [
                ("D1", "George Costanza"),
                ("D2", "Joey"),
                ("D3F1", "George Costanza"),
                ("D3F2", "George Costanza"),
                ("D4", "Joey"),
                ("D5", "Dwight"),
                ("D6", "Dwight"),
            ],
        )

    def test_staged_planner_keeps_dialogue_exact_and_canonicalizes_bad_timing(self):
        invalid_second = _segment(2)
        invalid_second["shots"][0]["start_seconds"] = 9.9
        invalid_second["shots"][0]["end_seconds"] = 10.0
        responses = iter([
            json.dumps(_ledger()),
            json.dumps(_segment(1)),
            json.dumps(invalid_second),
            json.dumps(_segment(2)),
        ])
        calls: list[dict] = []

        def generate(**kwargs):
            calls.append(kwargs)
            return next(responses)

        result = plan_h3_story_segments(
            self.prompt,
            segment_durations=[10.0, 10.0],
            mode="reference_sequence",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            llm_generate=generate,
        )
        self.assertEqual(result["planned_by"], "llm")
        # Maestro owns the local shot clock now, so malformed model-authored
        # timing is snapped locally instead of spending another LLM pass.
        self.assertEqual(len(calls), 3)
        self.assertNotIn("REPAIR ONLY THIS SEGMENT", calls[1]["prompt"])
        self.assertNotIn("REPAIR ONLY THIS SEGMENT", calls[2]["prompt"])
        rendered = json.dumps(result["segments"], ensure_ascii=False)
        self.assertEqual(rendered.count("Enough. Give me the glove, Thanos."), 1)
        self.assertEqual(rendered.count("I am inevitable."), 1)
        self.assertEqual(result["segments"][1]["opening_state"], result["segments"][0]["closing_state"])
        # The ledger owns the close; the camera response cannot silently alter it.
        self.assertIn(
            "Thanos raises his left hand",
            result["segments"][1]["closing_state"],
        )
        self.assertNotIn("MANDATORY OUTPUT CHECKSUM", calls[0]["prompt"])
        self.assertNotIn("beats", calls[0]["json_schema"]["properties"])
        self.assertIn("Maestro has already parsed", calls[0]["prompt"])
        self.assertIn("dialogue_performances", calls[1]["prompt"])
        self.assertIn("SPEAKER-CAMERA LOCK", calls[1]["prompt"])

    def test_faithful_treatment_never_asks_llm_to_copy_internal_story_ids(self):
        candidate = _ledger()
        candidate["beats"] = [{
            "beat_id": "B99",
            "segment": 1,
            "description": "A reordered replacement event",
            "source_event_ids": ["E999", "E1", "E1"],
            "dialogue_ids": ["D2", "D1"],
            "state_after": "The wrong ending",
            "sound_effects": "N/A",
        }]
        responses = iter([
            json.dumps(candidate),
            json.dumps(_segment(1)),
            json.dumps(_segment(2)),
        ])
        calls: list[dict] = []

        def generate(**kwargs):
            calls.append(kwargs)
            return next(responses)

        result = plan_h3_story_segments(
            self.prompt,
            segment_durations=[10.0, 10.0],
            mode="reference_sequence",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            llm_generate=generate,
        )

        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(result["planning_warnings"], [])
        self.assertEqual(len(calls), 3)
        self.assertNotIn("REPAIR THE COMPLETE STORY SCHEDULE", calls[0]["prompt"])
        self.assertNotIn("source_event_ids", calls[0]["json_schema"]["properties"])
        referenced = [
            event_id
            for beat in result["ledger"]["beats"]
            for event_id in beat["source_event_ids"]
        ]
        self.assertEqual(referenced, [item["event_id"] for item in extract_source_events(self.prompt)])
        self.assertNotIn("beats", calls[0]["json_schema"]["properties"])

    def test_canonicalizer_anchors_immediate_first_line_without_llm_repair(self):
        prompt = (
            "George Costanza walks into the coffee shop and walks up to Joey, "
            "who is sitting on the couch. "
            'George passionately says "Maestro two is out!" '
            'Joey says "What, who?" '
            'George says "It is crazy! Maestro now has an editor." '
            'Joey replies "Who are you?"'
        )
        durations = [14.375, 14.375, 13.25]
        locked = extract_locked_dialogue(prompt)
        events = extract_source_events(prompt)
        canonical = _deterministic_ledger(
            prompt,
            segment_count=3,
            segment_durations=durations,
            locked_dialogue=locked,
            camera_coverage="multi_shot",
            reference_context="",
        )
        candidate = {
            **{
                key: canonical[key]
                for key in (
                    "subject_continuity",
                    "setting_continuity",
                    "visual_continuity",
                    "editing_style",
                    "initial_state",
                    "ambient_audio",
                    "music",
                    "required_final_outcome",
                )
            },
            "beats": [
                {
                    "segment": 1,
                    "description": "George enters",
                    "source_event_ids": [events[0]["event_id"]],
                    "dialogue_ids": [],
                    "state_after": "George is inside",
                    "sound_effects": "Footsteps",
                },
                {
                    "segment": 2,
                    "description": "George approaches and speaks",
                    "source_event_ids": [
                        events[1]["event_id"],
                        events[2]["event_id"],
                    ],
                    "dialogue_ids": ["D1"],
                    "state_after": "George finishes his first line",
                    "sound_effects": "Room tone",
                },
                {
                    "segment": 3,
                    "description": "Joey reacts, George explains, and Joey replies",
                    "source_event_ids": [
                        events[3]["event_id"],
                        events[4]["event_id"],
                        events[5]["event_id"],
                    ],
                    "dialogue_ids": ["D2", "D3", "D4"],
                    "state_after": "Joey finishes his reply",
                    "sound_effects": "Room tone",
                },
            ],
            "generated_dialogue": [],
        }

        compiled = _canonicalize_story_ledger(
            prompt,
            canonical,
            candidate,
            locked_dialogue=locked,
            segment_count=3,
            allow_generated_dialogue=False,
        )

        owner = next(
            beat for beat in compiled["beats"]
            if "D1" in beat.get("dialogue_ids", [])
        )
        self.assertEqual(owner["segment"], 1)
        self.assertEqual(
            ledger_violations(
                prompt,
                compiled,
                segment_count=3,
                locked_dialogue=locked,
                expect_dialogue=True,
                allow_generated_dialogue=False,
                segment_durations=durations,
            ),
            [],
        )

    def test_creative_fallback_salvages_valid_dialogue_from_rejected_structure(self):
        prompt = (
            "George Costanza tells Joey that Maestro version two just dropped. "
            "Joey asks what Maestro is."
        )
        invalid = _deterministic_ledger(
            prompt,
            segment_count=2,
            segment_durations=[10.0, 10.0],
            locked_dialogue=[],
            camera_coverage="multi_shot",
            reference_context="",
        )
        invalid["ambient_audio"] = (
            "Coffee cups, soft piano, and indistinct background chatter"
        )
        invalid["beats"][0]["source_event_ids"] = ["E999"]
        invalid["generated_dialogue"] = [
            {
                "speaker": "George",
                "language": "English",
                "delivery": "rapid-fire excitement",
                "text": "Joey, Maestro version two just dropped!",
                "segment": 2,
            },
            {
                "speaker": "Joey",
                "language": "English",
                "delivery": "warmly confused",
                "text": "Is Maestro another kind of sandwich?",
                "segment": 2,
            },
        ]
        segment_one = _segment(1)
        segment_one["shots"][0]["action"] = "George excitedly approaches Joey"
        segment_two = _segment(2)
        segment_two["shots"][0]["action"] = "Joey reacts with complete confusion"
        responses = iter([
            json.dumps(invalid),
            json.dumps(invalid),
            json.dumps(segment_one),
            json.dumps(segment_two),
        ])

        result = plan_h3_story_segments(
            prompt,
            segment_durations=[10.0, 10.0],
            mode="sliding_window",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            planning_style="creative",
            llm_generate=lambda **_kwargs: next(responses),
        )

        self.assertEqual(result["planned_by"], "hybrid_repair")
        self.assertTrue(result["planning_warnings"])
        self.assertTrue(result["planning_diagnostics"])
        self.assertIn(
            "source event IDs are missing, foreign, or repeated",
            result["planning_diagnostics"],
        )
        self.assertEqual(
            [item["segment"] for item in result["ledger"]["generated_dialogue"]],
            [1, 2],
        )
        self.assertNotIn(
            "chatter",
            result["ledger"]["ambient_audio"].casefold(),
        )
        self.assertEqual(
            [
                line["dialogue_id"]
                for segment in result["segments"]
                for shot in segment["shots"]
                for line in (shot.get("dialogue") or [])
            ],
            ["D1", "D2"],
        )
        self.assertEqual(
            [item["text"] for item in result["ledger"]["generated_dialogue"]],
            [
                "Joey, Maestro version two just dropped!",
                "Is Maestro another kind of sandwich?",
            ],
        )

    def test_long_form_planning_uses_bounded_chapters_not_one_call_per_window(self):
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
                            "continuity_notes": "Carry the established state",
                        }
                        for index in range(count)
                    ],
                })
            count = schema["properties"]["segments"]["minItems"]
            prompt = kwargs["prompt"]
            match = __import__("re").search(r"WINDOW OBLIGATIONS:\n(\[.*?\])\n\nGLOBAL", prompt, __import__("re").S)
            obligations = json.loads(match.group(1)) if match else []
            return json.dumps({
                "segments": [
                    {
                        "window": obligations[index]["window"] if index < len(obligations) else index + 1,
                        "supporting_progression": f"New visible progression {index + 1}",
                        "resulting_state": f"New ending state {index + 1}",
                        "sound_effects": "Synchronized ambience",
                        "dialogue": [],
                    }
                    for index in range(count)
                ],
            })

        result = plan_h3_story_segments(
            "A traveler crosses a strange world and finally reaches a distant city.",
            segment_durations=[10.0] * 25,
            mode="sliding_window",
            camera_coverage="multi_shot",
            llm_generate=generate,
        )

        self.assertEqual(result["planned_by"], "hierarchical_llm")
        self.assertEqual(len(result["segments"]), 25)
        self.assertEqual(len(calls), 3)
        self.assertIn("chapters", calls[0]["json_schema"]["properties"])
        self.assertEqual(
            [
                call["json_schema"]["properties"]["segments"]["minItems"]
                for call in calls[1:]
            ],
            [24, 1],
        )

    def test_generated_dialogue_selects_a_segment_without_owning_story_ids(self):
        prompt = "Clark saves Lana from danger, and Lana reacts in disbelief."
        context = {
            "subject_continuity": "Clark and Lana remain visually unchanged",
            "setting_continuity": "The same Smallville street",
            "visual_continuity": "Live-action television drama",
            "editing_style": "Motivated cinematic coverage",
            "initial_state": "Clark sees Lana in danger",
            "ambient_audio": "Quiet nonverbal small-town ambience",
            "music": "N/A",
            "required_final_outcome": "Lana reacts after Clark saves her",
            "beats": [
                {
                    "segment": 1,
                    "source_event_ids": ["E1"],
                    "dialogue_ids": [],
                    "state_after": "Clark has moved Lana out of danger",
                    "sound_effects": "A fast rush of air and footsteps",
                },
                {
                    "segment": 2,
                    "source_event_ids": ["E2"],
                    "dialogue_ids": [],
                    "state_after": "Lana stares at Clark in disbelief",
                    "sound_effects": "Quiet small-town ambience",
                },
            ],
            "generated_dialogue": [{
                "speaker": "Lana",
                "language": "English",
                "delivery": "stunned and breathless",
                "text": "Clark, how did you do that?",
                "segment": 2,
            }],
        }
        responses = iter([
            json.dumps(context),
            json.dumps(_segment(1)),
            json.dumps(_segment(2)),
        ])

        result = plan_h3_story_segments(
            prompt,
            segment_durations=[10.0, 10.0],
            mode="sliding_window",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            planning_style="creative",
            llm_generate=lambda **_kwargs: next(responses),
        )

        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(result["ledger"]["generated_dialogue"][0]["dialogue_id"], "D1")
        rendered = json.dumps(result["segments"], ensure_ascii=False)
        self.assertEqual(rendered.count("Clark, how did you do that?"), 1)
        self.assertNotIn("Clark, how did you do that?", json.dumps(result["segments"][0]))

    def test_creative_mode_can_add_lines_around_locked_quotes_without_reordering_them(self):
        canonical = _deterministic_ledger(
            self.prompt,
            segment_count=2,
            segment_durations=[10.0, 10.0],
            locked_dialogue=self.locked,
            camera_coverage="multi_shot",
            reference_context="",
        )
        candidate = _ledger()
        candidate["generated_dialogue"] = [{
            "speaker": "Thanos",
            "language": "English",
            "delivery": "dryly amused",
            "text": "You still think this is a negotiation?",
            "segment": 2,
        }]
        compiled = _canonicalize_story_ledger(
            self.prompt,
            canonical,
            candidate,
            locked_dialogue=self.locked,
            segment_count=2,
            allow_generated_dialogue=True,
        )
        self.assertEqual(compiled["generated_dialogue"][0]["dialogue_id"], "D3")
        self.assertEqual(compiled["beats"][1]["dialogue_ids"], ["D2", "D3"])
        self.assertEqual(
            ledger_violations(
                self.prompt,
                compiled,
                segment_count=2,
                locked_dialogue=self.locked,
                expect_dialogue=True,
                allow_generated_dialogue=True,
                segment_durations=[10.0, 10.0],
            ),
            [],
        )

    def test_canonicalizer_rebuilds_locked_dialogue_ids_from_source_events(self):
        canonical = _deterministic_ledger(
            self.prompt,
            segment_count=2,
            segment_durations=[10.0, 10.0],
            locked_dialogue=self.locked,
            camera_coverage="multi_shot",
            reference_context="",
        )
        candidate = _ledger()
        candidate["beats"][0]["dialogue_ids"] = ["D2", "D2"]
        candidate["beats"][1]["dialogue_ids"] = ["D1"]

        compiled = _canonicalize_story_ledger(
            self.prompt,
            canonical,
            candidate,
            locked_dialogue=self.locked,
            segment_count=2,
            allow_generated_dialogue=False,
        )

        self.assertEqual(compiled["beats"][0]["dialogue_ids"], ["D1"])
        self.assertEqual(compiled["beats"][1]["dialogue_ids"], ["D2"])
        self.assertEqual(
            ledger_violations(
                self.prompt,
                compiled,
                segment_count=2,
                locked_dialogue=self.locked,
                expect_dialogue=True,
                segment_durations=[10.0, 10.0],
            ),
            [],
        )

    def test_faithful_mode_rejects_extra_dialogue_around_locked_quotes(self):
        ledger = _ledger()
        ledger["generated_dialogue"] = [{
            "dialogue_id": "D3",
            "speaker": "Thanos",
            "language": "English",
            "delivery": "calmly",
            "text": "No.",
            "segment": 2,
        }]
        ledger["beats"][1]["dialogue_ids"].append("D3")
        violations = ledger_violations(
            self.prompt,
            ledger,
            segment_count=2,
            locked_dialogue=self.locked,
            expect_dialogue=True,
            allow_generated_dialogue=False,
            segment_durations=[10.0, 10.0],
        )
        self.assertIn("invented extra dialogue despite locked user dialogue", violations)

    def test_creative_mode_honors_only_these_lines_override(self):
        self.assertTrue(_only_supplied_dialogue_requested(
            'Alex says, "Stay here." Use only these lines of dialogue.',
        ))
        self.assertTrue(_only_supplied_dialogue_requested(
            'Alex says, "Stay here." Do not add any additional dialogue.',
        ))
        self.assertFalse(_only_supplied_dialogue_requested(
            'Alex says, "Stay here," and the others argue about the plan.',
        ))

    def test_one_camera_shot_is_canonicalized_to_cover_every_assigned_beat(self):
        prompt = (
            "Alex opens the wooden door. Alex crosses the dark room. "
            "Alex picks up the red book."
        )
        context = {
            "subject_continuity": "Alex remains visually unchanged",
            "setting_continuity": "The same dark room",
            "visual_continuity": "Natural cinematic realism",
            "editing_style": "One continuous motivated shot",
            "initial_state": "Alex stands outside the closed wooden door",
            "ambient_audio": "Quiet nonverbal room tone",
            "music": "N/A",
            "required_final_outcome": "Alex holds the red book",
            "beats": [{
                "segment": 1,
                "source_event_ids": ["E1", "E2", "E3"],
                "dialogue_ids": [],
                "state_after": "Alex holds the red book inside the room",
                "sound_effects": "Door creak, footsteps, and the book lifting",
            }],
            "generated_dialogue": [],
        }
        camera_plan = _segment(1, duration=12.0)
        camera_plan["shots"][0]["action"] = "Alex opens the wooden door and steps inside"
        calls: list[dict] = []
        responses = iter([json.dumps(context), json.dumps(camera_plan)])

        def generate(**kwargs):
            calls.append(kwargs)
            return next(responses)

        result = plan_h3_story_segments(
            prompt,
            segment_durations=[12.0],
            mode="sliding_window",
            camera_coverage="continuous",
            expect_dialogue=False,
            llm_generate=generate,
        )

        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(result["planning_warnings"], [])
        self.assertEqual(len(calls), 2)
        action = result["segments"][0]["shots"][0]["action"]
        self.assertIn("opens the wooden door", action)
        self.assertIn("crosses the dark room", action)
        self.assertIn("picks up the red book", action)

    def test_coarse_dialogue_beat_binds_each_line_to_its_matching_camera_phase(self):
        prompt = (
            "Yoda is in Dagobah. "
            "Thanos stands in the swamp and says <d>Tell me what you know.</d> "
            "Yoda waves slowly while saying <d>Powerful, it has become.</d> "
            "Thanos responds <d>As all things should be.</d>"
        )
        ledger = {
            "subject_continuity": "Thanos and Yoda retain their requested identities",
            "setting_continuity": "The same misty Dagobah swamp",
            "visual_continuity": "Grounded cinematic live-action realism",
            "editing_style": "Motivated speaker coverage",
            "initial_state": "Yoda and Thanos face each other in the swamp",
            "ambient_audio": "Wetland insects, water, and foliage",
            "music": "N/A",
            "required_final_outcome": "Thanos finishes his response",
            # This deliberately reproduces the coarse semantic beat from the
            # reported run: three speakers/turns are grouped under one beat.
            "beats": [{
                "segment": 1,
                "source_event_ids": ["E1", "E2", "E3", "E4"],
                "dialogue_ids": ["D1", "D2", "D3"],
                "state_after": "Thanos has finished responding to Yoda",
                "sound_effects": "Natural swamp movement",
            }],
            "generated_dialogue": [],
        }
        camera_plan = {
            "segment": 1,
            "title": "Dagobah exchange",
            "opening_state": "The supplied opening state",
            "coverage": "multi_shot",
            "pacing": "natural real-time pacing",
            "shots": [
                {
                    "shot": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 3.0,
                    "transition": "opening composition",
                    "framing": "wide establishing shot",
                    "camera": "locked camera",
                    "action": "Yoda and Thanos stand in the swamp",
                    "sound_effects": "Swamp ambience",
                },
                {
                    "shot": 2,
                    "start_seconds": 3.0,
                    "end_seconds": 6.5,
                    "transition": "hard cut",
                    "framing": "Thanos close-up",
                    "camera": "slow push in",
                    "action": "Thanos raises his chin and speaks with a deep voice",
                    "sound_effects": "Thanos's deep voice and swamp ambience",
                },
                {
                    "shot": 3,
                    "start_seconds": 6.5,
                    "end_seconds": 10.0,
                    "transition": "hard cut",
                    "framing": "Yoda close-up",
                    "camera": "locked camera",
                    "action": "Yoda waves slowly as he speaks",
                    "sound_effects": "Yoda's raspy voice",
                },
                {
                    "shot": 4,
                    "start_seconds": 10.0,
                    "end_seconds": 13.667,
                    "transition": "hard cut",
                    # Deliberately give the camera planner the exact visual
                    # contradiction seen in the reported run: the transcript
                    # and Audio reference belong to Thanos, but the proposed
                    # close-up/action still favor Yoda.
                    "framing": "Yoda close-up",
                    "camera": "subtle push in on Yoda",
                    "action": "Yoda gestures while Thanos speaks a short phrase",
                    "sound_effects": "Yoda's robe and Thanos's voice",
                },
            ],
            "closing_state": "The supplied closing state",
        }
        responses = iter([json.dumps(ledger), json.dumps(camera_plan)])

        result = plan_h3_story_segments(
            prompt,
            segment_durations=[13.667],
            mode="reference_sequence_continuation",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            llm_generate=lambda **_kwargs: next(responses),
        )

        self.assertEqual(result["planned_by"], "llm")
        shots = result["segments"][0]["shots"]
        self.assertEqual(
            [[line["speaker"] for line in shot["dialogue"]] for shot in shots],
            [[], ["Thanos"], ["Yoda"], ["Thanos"]],
        )
        self.assertEqual(
            [[line["text"] for line in shot["dialogue"]] for shot in shots],
            [
                [],
                ["Tell me what you know."],
                ["Powerful, it has become."],
                ["As all things should be."],
            ],
        )
        self.assertNotIn(
            "Stable speaking identities",
            result["ledger"]["subject_continuity"],
        )
        for shot in shots:
            self.assertNotRegex(shot["action"], r"(?i)\b(?:speaks?|says?|responds?)\b")
            self.assertNotRegex(shot["sound_effects"], r"(?i)\bvoice\b")
        final_shot = shots[-1]
        self.assertIn("established target setting", final_shot["framing"])
        self.assertNotIn("Yoda", final_shot["framing"])
        self.assertIn("target-scene camera coverage", final_shot["camera"])
        self.assertNotIn("Yoda", final_shot["camera"])
        self.assertIn("only Thanos's mouth moves", final_shot["dialogue"][0]["action"])
        self.assertIn("every other visible mouth stays closed", final_shot["dialogue"][0]["action"])
        self.assertLessEqual(len(final_shot["dialogue"][0]["action"]), 120)
        self.assertNotIn(";", final_shot["dialogue"][0]["action"])
        self.assertNotIn("speaker-focused", final_shot["framing"])
        self.assertNotIn("hold Thanos's visible face", final_shot["camera"])
        self.assertNotIn("Yoda", final_shot["action"])

    def test_late_dwight_entrance_keeps_cast_and_adjacent_dialogue_local(self):
        prompt = (
            "George Costanza walks into the coffee shop on the TV show Friends, "
            "from the outside, and walks up to Joey, who is sitting on the couch. "
            'George passionately says "Maestro two is out!" '
            'Joey says "Wha, who?" George says "It is crazy! It even has an editor!" '
            'Joey replies "Wow, cool. Um, who are you again?" '
            "Camera pans to Dwight from The Office, who is also in the Friends "
            'Coffee shop, and Dwight says with frustration "Ugh, Joey, this is '
            'George. George—Joey" as he introduces them. Dwight then muffles '
            'softly "I hate A.I."'
        )

        intent = extract_h3_source_intent(prompt)
        dialogue = extract_locked_dialogue(prompt)

        self.assertEqual(intent["cast_names"], ["George Costanza", "Joey", "Dwight"])
        self.assertNotIn("Camera", intent["cast_cardinality_contract"])
        self.assertNotIn("Office", intent["cast_cardinality_contract"])
        self.assertNotIn("Friends Coffee", intent["cast_cardinality_contract"])
        self.assertEqual(dialogue[-1]["speaker"], "Dwight")
        self.assertEqual(dialogue[-1]["delivery"], "muffles softly")
        self.assertNotIn("George—Joey", dialogue[-1]["delivery"])
        events = extract_source_events(prompt)
        event_text = [item["text"] for item in events]
        self.assertFalse(any(
            value.casefold() == "as he introduces them"
            for value in event_text
        ))
        self.assertTrue(any(
            "while he introduces them" in value.casefold()
            for value in event_text
        ))
        self.assertTrue(any(
            "dwight muffles softly" in value.casefold()
            for value in event_text
        ))
        dialogue_events = _expected_dialogue_events(prompt, dialogue)
        self.assertEqual(
            next(
                item["text"] for item in events
                if item["event_id"] == dialogue_events["D6"]
            ).casefold(),
            "dwight muffles softly",
        )

        result = plan_h3_story_segments(
            prompt,
            segment_durations=[14.375, 13.625, 13.583],
            mode="sliding_window",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            planning_style="faithful",
            llm_generate=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("offline")
            ),
        )

        self.assertEqual(result["source_intent"]["cast_names"], [
            "George Costanza",
            "Joey",
            "Dwight",
        ])
        self.assertEqual(
            result["segments"][0]["continuity_handoff_cast"],
            ["George Costanza", "Joey"],
        )
        self.assertEqual(
            result["segments"][1]["continuity_handoff_cast"],
            ["George Costanza", "Joey"],
        )
        self.assertEqual(result["segments"][2]["continuity_handoff_cast"], [])
        self.assertNotIn("Dwight", json.dumps(result["segments"][1]))
        rendered = [
            (line["dialogue_id"], line["speaker"], line["text"])
            for segment in result["segments"]
            for shot in segment["shots"]
            for line in shot.get("dialogue") or []
        ]
        self.assertEqual(
            [(dialogue_id, speaker) for dialogue_id, speaker, _text in rendered],
            [
                ("D1", "George Costanza"),
                ("D2", "Joey"),
                ("D3", "George Costanza"),
                ("D4", "Joey"),
                ("D5", "Dwight"),
                ("D6", "Dwight"),
            ],
        )
        self.assertTrue(result["planning_warnings"])
        self.assertNotIn(
            "AI-authored dialogue",
            " ".join(result["planning_warnings"]),
        )

    def test_materialization_removes_a_future_character_reaction_angle(self):
        segment = {
            "segment": 2,
            "title": "George continues",
            "opening_state": "George stands beside Joey",
            "coverage": "multi_shot",
            "pacing": "natural real-time pacing",
            "shots": [{
                "shot": 1,
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "transition": "opening composition",
                    "framing": "Medium shot focused on Dwight while George remains behind him",
                    "camera": "A slow push in on Dwight",
                    "sound_effects": "Dwight sighs softly near the counter",
                "beat_ids": ["B1"],
                "action": "George continues explaining the editor to Joey",
                "dialogue": [],
                "sound_effects": "Coffee shop ambience",
            }],
            "closing_state": "George finishes the explanation",
        }
        beats = [{
            "beat_id": "B1",
            "description": "George continues explaining the editor to Joey",
            "source_event_ids": ["E1"],
            "dialogue_ids": [],
            "state_after": "George finishes the explanation",
        }]
        materialized = _materialize_segment(
            segment,
            beats=beats,
            dialogue_catalog=[{
                "dialogue_id": "D1",
                "speaker": "Dwight",
                "text": "I hate A.I.",
            }],
            source_events=[{
                "event_id": "E1",
                "text": "George continues explaining the editor to Joey",
            }],
            future_cast=["Dwight"],
        )

        shot = materialized["shots"][0]
        self.assertNotIn("Dwight", shot["framing"])
        self.assertNotIn("Dwight", shot["camera"])
        self.assertNotIn("Dwight", shot["sound_effects"])

    def test_faithful_treatment_rejects_a_shot_plan_inside_ambient_audio(self):
        canonical = _ledger()
        canonical["source_intent"] = {
            "cast_names": ["George Costanza", "Joey", "Dwight"],
        }
        candidate = {
            "setting_continuity": "The same warm coffee shop interior",
            "visual_continuity": "Warm live-action sitcom photography",
            "editing_style": "Motivated conversational coverage",
            "ambient_audio": (
                "--- Sequence Progression: Camera starts on Joey, then George "
                "enters, then pan to Dwight for his dialogue."
            ),
        }

        result = _apply_faithful_treatment(canonical, candidate)

        self.assertEqual(result["ambient_audio"], canonical["ambient_audio"])
        self.assertEqual(
            result["editing_style"],
            "Motivated conversational coverage",
        )

    def test_final_staging_repairs_listener_focus_before_visible_dialogue(self):
        segment = {
            "shots": [
                {
                    "shot": 1,
                    "framing": "Medium Shot of Joey seated on the couch",
                    "camera": (
                        "Maintain target-scene coverage with George Costanza "
                        "as the active visible speaker"
                    ),
                    "dialogue": [{"speaker": "George Costanza", "text": "Hello"}],
                },
                {
                    "shot": 2,
                    "framing": "Medium scene composition",
                    "camera": "Locked camera subtly elevates George above Joey",
                    "dialogue": [{"speaker": "Joey", "text": "Who are you?"}],
                },
                {
                    "shot": 3,
                    "framing": "Medium Shot of Dwight",
                    "camera": "A subtle rack focus from Dwight to Joey",
                    "dialogue": [{"speaker": "Dwight", "text": "I hate A.I."}],
                },
            ],
        }
        speakers = ["George Costanza", "Joey", "Dwight"]

        violations = _materialized_segment_violations(
            segment,
            known_speakers=speakers,
        )
        self.assertTrue(any("shot 1 framing" in item for item in violations))
        self.assertTrue(any("shot 2 camera" in item for item in violations))
        self.assertTrue(any("shot 3 camera" in item for item in violations))

        repaired = _repair_materialized_segment_staging(
            segment,
            known_speakers=speakers,
        )
        self.assertEqual(
            _materialized_segment_violations(
                repaired,
                known_speakers=speakers,
            ),
            [],
        )
        self.assertIn(
            "George Costanza carries the visible speaking performance",
            repaired["shots"][0]["framing"],
        )
        self.assertIn("settle on Joey", repaired["shots"][1]["camera"])
        self.assertIn("settle on Dwight", repaired["shots"][2]["camera"])

    def test_action_only_shots_cannot_become_narration_or_repeat_planner_prose(self):
        prompt = (
            "Blaine waves and says <d>Hello from Maestro.</d> "
            "Thanos snaps his fingers. "
            "Blaine turns to dust and blows away."
        )
        ledger = {
            "subject_continuity": "Blaine and Thanos retain their identities",
            "setting_continuity": "The same misty swamp",
            "visual_continuity": "Grounded cinematic live action",
            "editing_style": "Motivated three-shot coverage",
            "initial_state": "Blaine and Thanos face each other",
            "ambient_audio": "Quiet swamp ambience",
            "music": "N/A",
            "required_final_outcome": "Blaine turns to dust and blows away",
            "beats": [{
                "segment": 1,
                "source_event_ids": ["E1", "E2", "E3"],
                "dialogue_ids": ["D1"],
                "state_after": "Only Thanos remains after the dust disperses",
                "sound_effects": "A finger snap and wind through dust",
            }],
            "generated_dialogue": [],
        }
        camera_plan = {
            "segment": 1,
            "title": "The snap",
            "opening_state": "Blaine and Thanos face each other",
            "coverage": "multi_shot",
            "pacing": "natural real-time pacing",
            "shots": [
                {
                    "shot": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 4.0,
                    "transition": "opening composition",
                    "framing": "medium shot",
                    "camera": "pan toward Blaine",
                    "action": "Blaine repeats his introduction twice",
                    "sound_effects": "Blaine's voice",
                },
                {
                    "shot": 2,
                    "start_seconds": 4.0,
                    "end_seconds": 7.0,
                    "transition": "hard cut",
                    "framing": "Thanos close-up",
                    "camera": "hold on the gauntlet",
                    "action": "Blaine repeats his line while Thanos waits",
                    "sound_effects": "Finger snap",
                },
                {
                    "shot": 3,
                    "start_seconds": 7.0,
                    "end_seconds": 10.0,
                    "transition": "hard cut",
                    "framing": "medium wide",
                    "camera": "track the drifting dust",
                    "action": "Blaine says turns to blows while disappearing",
                    "sound_effects": "Wind",
                },
            ],
            "closing_state": "Only Thanos remains",
        }
        responses = iter([json.dumps(ledger), json.dumps(camera_plan)])

        result = plan_h3_story_segments(
            prompt,
            segment_durations=[10.0],
            mode="reference_sequence_continuation",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            llm_generate=lambda **_kwargs: next(responses),
        )

        shots = result["segments"][0]["shots"]
        combined = " ".join(shot["action"] for shot in shots)
        self.assertNotIn("repeats", combined)
        self.assertNotIn("turns to blows", combined)
        self.assertEqual([len(shot["dialogue"]) for shot in shots], [1, 0, 0])
        self.assertTrue(shots[1]["action"].startswith("Silent visual action"))
        self.assertTrue(shots[2]["action"].startswith("Silent visual action"))
        self.assertIn("No words are spoken or mouthed", shots[2]["action"])
        self.assertIn("Blaine turns to dust and blows away", shots[2]["action"])

    def test_compiler_neutralizes_braces_and_nested_context_labels(self):
        spans = compute_h3_window_boundaries(480, 240, fps=24, overlap_frames=0)
        plan = {
            "subject_continuity": 'Heroes {"S1": "Superman"}; summary: never nest this',
            "setting_continuity": "The same arena",
            "visual_continuity": "Cinematic realism",
            "editing_style": "Motivated cuts",
            "initial_state": "The fighters face each other",
            "ambient_audio": "Wind",
            "music": "N/A",
            "windows": [
                {
                    "window": index + 1,
                    "title": f"Beat {index + 1}",
                    "coverage": "cinematic coverage",
                    "pacing": "real-time",
                    "shots": [{
                        "shot": 1,
                        "start_seconds": 0.0,
                        "end_seconds": 10.0,
                        "transition": "opening composition",
                        "framing": "medium shot",
                        "camera": "locked camera",
                        "action": f"Action {index + 1}",
                        "dialogue": [],
                        "sound_effects": "N/A",
                    }],
                    "closing_state": f"State {index + 1}",
                }
                for index in range(2)
            ],
        }
        compiled = compile_h3_window_prompts(plan, spans)
        for item in compiled:
            prompt = item["prompt"]
            self.assertNotIn("{", prompt)
            self.assertNotIn("}", prompt)
            self.assertNotIn("summary:", prompt)
            self.assertEqual(prompt.count("integrated_multimodal_description:"), 1)
            self.assertEqual(prompt.count("overall_soundscape:"), 1)
            self.assertEqual(prompt.count("non_diegetic_music:"), 1)

    def test_sanitizer_neutralizes_template_and_context_ir_syntax(self):
        value = sanitize_h3_prompt_text('{"S1": "Neo"} detailed_description: action')
        self.assertEqual(value, '("S1": "Neo") detailed_description - action')


if __name__ == "__main__":
    unittest.main()
