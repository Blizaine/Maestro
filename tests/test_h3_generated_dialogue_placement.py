"""AI-authored speech uses the same cast and event placement through each pass."""

from copy import deepcopy
import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import (
    H3DialogueTimingError,
    _camera_phase_beats, _canonicalize_story_ledger, _complete_creative_dialogue,
    _deterministic_ledger, _dialogue_catalog, _prepare_render_dialogue_schedule,
    _spread_generated_dialogue_across_segments, extract_h3_source_intent,
    extract_locked_dialogue, extract_source_events, plan_h3_story_segments, segment_violations,
)
from services.h3_dialogue_writing import _shorten_generated_dialogue
from services.studio_enhancement import prepare_enhanced_job


BRIEF = (
    "A workplace sitcom. Nora walks into the office. The receptionist Eva asks if she needs help. "
    "Dean and Miles immediately see Nora and they both rush over to assist her. "
    "Both of them argue about paper. Miles then invites her into his office and has an awkward dialogue with her."
)


def line(text, *, speaker="Eva", segment=1, event=""):
    return {"speaker": speaker, "language": "English", "delivery": "naturally",
            "text": text, "segment": segment, "source_event_id": event}


def ledger_for(prompt, durations):
    return _deterministic_ledger(
        prompt, segment_count=len(durations), segment_durations=durations,
        locked_dialogue=extract_locked_dialogue(prompt), camera_coverage="multi_shot", reference_context="",
    )


class GeneratedDialoguePlacementTests(unittest.TestCase):
    def test_named_arrivals_and_helpers_remain_available_to_dialogue_writer(self):
        self.assertEqual(extract_h3_source_intent(BRIEF)["cast_names"], ["Nora", "Eva", "Dean", "Miles"])
        canonical = ledger_for(BRIEF, [14.4, 13.6])
        generator = Mock(return_value="{}")
        _complete_creative_dialogue(
            BRIEF, canonical, canonical_ledger=canonical, locked_dialogue=[],
            durations=[14.4, 13.6], generate=generator, system_prompt="Write a sitcom.",
        )
        self.assertTrue(generator.called)
        for call in generator.call_args_list:
            schema = call.kwargs["json_schema"]["properties"]["generated_dialogue"]["items"]["properties"]
            self.assertEqual(schema["speaker"]["enum"], ["Nora", "Eva", "Dean", "Miles"])
            self.assertIn(f"E{len(extract_source_events(BRIEF))}", schema["source_event_id"]["enum"])

    def test_greeting_stays_at_its_event_through_balancing_and_camera_phases(self):
        canonical = ledger_for(BRIEF, [14.4, 13.6])
        events = extract_source_events(BRIEF)
        greeting = next(e for e in events if "asks if" in e["text"])
        argument = next(e for e in events if "argue about" in e["text"])
        candidate = deepcopy(canonical)
        # Several distinct actions share a window; the greeting belongs to
        # its own event, not the last event of that window.
        candidate["beats"] = [
            {"beat_id": f"B{i}", "segment": 1 if i <= 3 else 2,
             "source_event_ids": [e["event_id"]], "dialogue_ids": [],
             "description": e["text"], "state_after": "The conversation advances.", "sound_effects": "Footsteps"}
            for i, e in enumerate(events, 1)
        ]
        owners = {e: b["segment"] for b in candidate["beats"] for e in b["source_event_ids"]}
        candidate["generated_dialogue"] = [
            line("Can I help you?", event=greeting["event_id"], segment=owners[greeting["event_id"]]),
            line("Our paper has excellent texture.", speaker="Dean", event=argument["event_id"], segment=owners[argument["event_id"]]),
            line("It is still just paper.", speaker="Miles", event=argument["event_id"], segment=owners[argument["event_id"]]),
        ]
        compiled = _canonicalize_story_ledger(
            BRIEF, canonical, candidate, locked_dialogue=[], segment_count=2,
            allow_generated_dialogue=True, preserve_adaptation=True,
        )
        original = deepcopy(compiled)
        _spread_generated_dialogue_across_segments(compiled, segment_count=2)
        self.assertEqual(compiled, original)
        for item in compiled["generated_dialogue"]:
            owner = next(b for b in compiled["beats"] if item["dialogue_id"] in b["dialogue_ids"])
            self.assertIn(item["source_event_id"], owner["source_event_ids"])
        phases = _camera_phase_beats(
            compiled["beats"], source_events=events,
            expected_dialogue_events={d["dialogue_id"]: d["source_event_id"] for d in compiled["generated_dialogue"]},
            preserve_adaptation=True,
        )
        self.assertEqual(next(b for b in phases if "D1" in b["dialogue_ids"])["source_event_ids"], [greeting["event_id"]])

    def test_only_an_exact_opening_quote_imposes_the_early_speech_deadline(self):
        prompt = "Nora enters the office. Eva asks if she needs help."
        beats = [
            {"beat_id": "B1", "description": "Nora enters the office", "dialogue_ids": []},
            {"beat_id": "B2", "description": "Eva asks if she needs help", "dialogue_ids": ["D1"]},
        ]
        catalog = [{"dialogue_id": "D1", "speaker": "Eva", "text": "Can I help you?"}]
        segment = {"segment": 1, "closing_state": "Nora stands at Eva's desk.", "shots": [
            {"action": "Nora enters the office", "beat_ids": ["B1"], "dialogue": [], "start_seconds": 0, "end_seconds": 5},
            {"action": "Eva asks if she needs help", "beat_ids": ["B2"], "dialogue": catalog, "start_seconds": 5, "end_seconds": 14.4},
        ]}
        kwargs = dict(segment_number=1, duration=14.4, assigned_beats=beats, dialogue_catalog=catalog)
        self.assertEqual(segment_violations(prompt, segment, **kwargs), [])
        quoted = 'Nora enters the office. Eva asks "Can I help you?"'
        self.assertIn("the first requested line begins too late in the opening segment", segment_violations(quoted, segment, **kwargs))

    def test_shortening_ai_speech_preserves_adapted_staging_and_exact_user_line(self):
        prompt = 'Eva and Nora discuss a film project. Eva says "Welcome!"'
        durations = [10.1]
        locked = extract_locked_dialogue(prompt)
        canonical = ledger_for(prompt, durations)
        candidate = deepcopy(canonical)
        prose = "Eva lays out the storyboard; Nora pulls a chair closer and studies the page."
        candidate["beats"][0]["description"] = prose
        candidate["generated_dialogue"] = [line("word " * 35)]
        initial = _canonicalize_story_ledger(prompt, canonical, candidate, locked_dialogue=locked,
            segment_count=1, allow_generated_dialogue=True, preserve_adaptation=True)
        written = "Let's group the shots by scene first, then review the storyboard together so we can agree on what to film tomorrow morning."
        result, warnings = _complete_creative_dialogue(
            prompt, initial, canonical_ledger=canonical, locked_dialogue=locked,
            durations=durations, generate=Mock(return_value=json.dumps({"generated_dialogue": [line(written)]})),
            system_prompt="Write dialogue.",
        )
        self.assertEqual(warnings, [])
        self.assertEqual(result["beats"][0]["description"], prose)
        self.assertEqual(result["generated_dialogue"][0]["text"], written)
        self.assertEqual(locked[0]["text"], "Welcome!")
        self.assertEqual([b["source_event_ids"] for b in initial["beats"]], [b["source_event_ids"] for b in result["beats"]])

    def test_safe_short_draft_is_usable_instead_of_retaining_overlong_ai_words(self):
        prompt = "Eva and Nora discuss their film project."
        canonical = ledger_for(prompt, [10.1])
        candidate = deepcopy(canonical)
        candidate["generated_dialogue"] = [line("word " * 35)]
        initial = _canonicalize_story_ledger(prompt, canonical, candidate, locked_dialogue=[],
            segment_count=1, allow_generated_dialogue=True)
        result, warnings = _complete_creative_dialogue(
            prompt, initial, canonical_ledger=canonical, locked_dialogue=[], durations=[10.1],
            generate=Mock(return_value=json.dumps({"generated_dialogue": [line("Let's start with the storyboard.")]})),
            system_prompt="Write dialogue.",
        )
        self.assertEqual(warnings, [])
        self.assertEqual(result["generated_dialogue"][0]["text"], "Let's start with the storyboard.")
        # The downstream exact-speech scheduler can preserve this draft and
        # return it for review; it needn't throw away the whole enhancement.
        _prepare_render_dialogue_schedule(
            result["beats"], _dialogue_catalog(result, []), segment_durations=[10.1],
            source_events=extract_source_events(prompt), expected_dialogue_events={},
        )

    def test_copyedit_changes_only_spoken_words_and_keeps_every_turn(self):
        lines = [
            line("We use twenty pound bond paper. It has a unique texture.", speaker="Dean", segment=2, event="E5"),
            line("No, it is twenty four pound. Much better opacity. You cannot see through it.", speaker="Miles", segment=2, event="E5"),
            line("Nora, you are not from around here, are you? Very distinctive presence.", speaker="Miles", segment=2, event="E6"),
            line("Um, no. I am from another city. It is a nice city.", speaker="Nora", segment=2, event="E6"),
        ]
        before = deepcopy(lines)
        texts = {"L1": "Our bond paper has texture.", "L2": "Ours is thicker. Excellent opacity.",
                 "L3": "Nora, are you local? You really stand out.", "L4": "No. I'm visiting from another city."}
        writer = Mock(return_value=json.dumps(texts))
        result = _shorten_generated_dialogue(BRIEF, lines, target_words=32, generate=writer, system_prompt="Write dialogue.")
        self.assertEqual(lines, before)
        self.assertEqual([item["text"] for item in result], list(texts.values()))
        self.assertEqual([{k: v for k, v in item.items() if k != "text"} for item in result],
                         [{k: v for k, v in item.items() if k != "text"} for item in before])
        self.assertIsNone(writer.call_args.kwargs['json_schema'])
        self.assertTrue(writer.call_args.kwargs['enable_thinking'])
        for response in ({"L1": "Dropped the rest."}, {**texts, "L4": ""}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                _shorten_generated_dialogue(BRIEF, lines, target_words=32,
                    generate=Mock(return_value=json.dumps(response)), system_prompt="Write dialogue.")

    def test_overlength_copyedit_keeps_latest_draft_and_locked_quote_intact(self):
        prompt = 'Eva and Nora discuss their film project. Eva says "Welcome!"'
        locked = extract_locked_dialogue(prompt)
        canonical = ledger_for(prompt, [10.1])
        candidate = deepcopy(canonical)
        candidate["generated_dialogue"] = [line("word " * 40, event="E1")]
        initial = _canonicalize_story_ledger(prompt, canonical, candidate, locked_dialogue=locked,
            segment_count=1, allow_generated_dialogue=True, preserve_adaptation=True)
        # Edit the existing spoken draft directly, keeping exact quotes out
        # of the editable text and retaining event ownership locally.
        shortened = "Let's group the shots by scene first, then review the storyboard together so we can agree on what to film tomorrow morning."
        writer = Mock(side_effect=[
            json.dumps({"L1": shortened}),
        ])
        result, warnings = _complete_creative_dialogue(prompt, initial, canonical_ledger=canonical,
            locked_dialogue=locked, durations=[10.1], generate=writer, system_prompt="Write dialogue.")
        self.assertEqual(warnings, [])
        writer.assert_called_once()
        self.assertTrue(writer.call_args.kwargs["prompt"].startswith("SHORTEN AI-WRITTEN LINES"))
        self.assertIn("word word", writer.call_args.kwargs["prompt"])
        self.assertEqual(result["generated_dialogue"][0]["text"], shortened)
        self.assertEqual(result["generated_dialogue"][0]["source_event_id"], "E1")
        self.assertEqual(_dialogue_catalog(result, locked)[0]["text"], "Welcome!")

    def test_camera_copyedit_retries_only_overlong_text_with_explicit_limits(self):
        lines = [line("Would you please take a seat over here?"),
                 line("Thank you very much.", speaker="Nora")]
        before = deepcopy(lines)
        writer = Mock(side_effect=[
            json.dumps({"L1": "Would you take a seat here?", "L2": "Thank you."}),
            json.dumps({"L1": "Please sit here.", "L2": "Thanks."}),
        ])
        revised = _shorten_generated_dialogue(BRIEF, lines, target_words=6, per_line_targets=[4, 2],
            generate=writer, system_prompt="Write dialogue.")
        self.assertEqual([item["text"] for item in revised], ["Please sit here.", "Thank you."])
        self.assertEqual(lines, before)
        self.assertIn("HARD per-line limit", writer.call_args_list[0].kwargs["prompt"])
        retry_turns = json.loads(writer.call_args_list[1].kwargs['prompt'].split('Editable turns:\n', 1)[1])
        self.assertEqual(set(retry_turns), {'L1'})
        self.assertEqual(retry_turns['L1']['maximum_words'], 4)
        self.assertEqual(retry_turns['L1']['text'], 'Would you take a seat here?')
        self.assertEqual([{k: v for k, v in item.items() if k != "text"} for item in revised],
                         [{k: v for k, v in item.items() if k != "text"} for item in lines])

        # A provider failure on the focused retry cannot discard a useful
        # initial edit. Its final timing still goes through the camera audit.
        writer = Mock(side_effect=[json.dumps({"L1": "Would you take a seat here?", "L2": "Thank you."}),
                                   RuntimeError("provider unavailable")])
        revised = _shorten_generated_dialogue(BRIEF, lines, target_words=6, per_line_targets=[4, 2],
            generate=writer, system_prompt="Write dialogue.")
        self.assertEqual(revised[0]["text"], "Would you take a seat here?")

    def test_copyedit_reasoning_has_separate_budget_and_preserves_ownership(self):
        lines = [line('Would you please sit over here?', event='E2')]
        writer = Mock(return_value=json.dumps({'L1': 'Please sit here.'}))
        result = _shorten_generated_dialogue(BRIEF, lines, target_words=4,
            per_line_targets=[4], generate=writer, system_prompt='Copyedit speech.')
        self.assertEqual(result[0]['text'], 'Please sit here.')
        self.assertEqual(result[0]['source_event_id'], 'E2')
        self.assertTrue(writer.call_args.kwargs['enable_thinking'])
        self.assertEqual(writer.call_args.kwargs['thinking_budget'], 512)
        self.assertEqual(writer.call_args.kwargs['max_new_tokens'], 768)
        self.assertEqual(lines[0]['text'], 'Would you please sit over here?')

    def test_an_unsuccessful_editor_retains_its_best_shorter_draft(self):
        prompt = "Eva and Nora discuss their film project."
        canonical = ledger_for(prompt, [10.1])
        candidate = deepcopy(canonical)
        candidate["generated_dialogue"] = [line("word " * 50)]
        initial = _canonicalize_story_ledger(prompt, canonical, candidate, locked_dialogue=[],
            segment_count=1, allow_generated_dialogue=True)
        writer = Mock(side_effect=[
            json.dumps({"L1": "revised " * 32}),
            json.dumps({"generated_dialogue": [line("draft " * 33)]}),
        ])
        result, warnings = _complete_creative_dialogue(prompt, initial, canonical_ledger=canonical,
            locked_dialogue=[], durations=[10.1], generate=writer, system_prompt="Write dialogue.")
        self.assertTrue(warnings)
        self.assertEqual(result["generated_dialogue"][0]["text"], ("revised " * 32).strip())
        self.assertEqual(writer.call_count, 2)  # Only one text-only shortening pass.

    def test_an_empty_window_does_not_beat_a_useful_overlong_reviewable_draft(self):
        prompt = 'Eva and Nora discuss their film project.'
        canonical = ledger_for(prompt, [10.1])
        writer = Mock(side_effect=[
            json.dumps({'generated_dialogue': [line('draft ' * 35)]}),
            json.dumps({'L1': 'revised ' * 32}),
        ])
        result, warnings = _complete_creative_dialogue(prompt, deepcopy(canonical),
            canonical_ledger=canonical, locked_dialogue=[], durations=[10.1],
            generate=writer, system_prompt='Write dialogue.')
        self.assertTrue(warnings)  # Genuine overflow still needs review.
        self.assertEqual(result['generated_dialogue'][0]['text'], ('revised ' * 32).strip())

    def test_density_development_that_overshoots_still_gets_copyedited(self):
        prompt = "Eva and Nora discuss their film project."
        canonical = ledger_for(prompt, [10.1])
        candidate = deepcopy(canonical)
        candidate["generated_dialogue"] = [line("Ready?", event="E1")]
        initial = _canonicalize_story_ledger(prompt, canonical, candidate, locked_dialogue=[],
            segment_count=1, allow_generated_dialogue=True)
        written = "Let's group the shots by scene first, then review the storyboard together so we can agree on what to film tomorrow morning."
        for final, expected in ((json.dumps({"L1": written}), written), ("{}", "Ready?")):
            with self.subTest(final=final):
                writer = Mock(side_effect=[
                    json.dumps({"generated_dialogue": [line("expanded " * 34, event="E1")]}), final,
                ])
                result, warnings = _complete_creative_dialogue(prompt, initial, canonical_ledger=canonical,
                    locked_dialogue=[], durations=[10.1], generate=writer, system_prompt="Write dialogue.")
                self.assertEqual(writer.call_count, 2)
                self.assertTrue(writer.call_args_list[1].kwargs["prompt"].startswith("SHORTEN AI-WRITTEN LINES"))
                self.assertEqual(result["generated_dialogue"][0]["text"], expected)
                self.assertEqual(warnings, [])  # The best shorter draft fits.

    def test_overlong_ai_draft_is_saved_for_review_and_pauses_queued_generation(self):
        prompt = "Eva and Nora discuss their film project."
        story = ledger_for(prompt, [10.1])
        story["generated_dialogue"] = [line("word " * 40, event="E1")]
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return json.dumps(story)
            if kwargs["prompt"].startswith("FIT THE SPOKEN SCRIPT"):
                return json.dumps({"generated_dialogue": [line("draft " * 33, event="E1")]})
            if kwargs["prompt"].startswith("SHORTEN AI-WRITTEN LINES"):
                return json.dumps({"L1": "revised " * 32})
            return "{}"  # A camera fallback must also preserve the reviewable draft.

        result = plan_h3_story_segments(prompt, segment_durations=[10.1], mode="sliding_window",
            camera_coverage="multi_shot", planning_style="adaptive", llm_generate=generate)
        self.assertTrue(any("more speaking time" in warning or "AI dialogue needs review" in warning
                            for warning in result["planning_warnings"]))
        self.assertNotIn("Increase total duration", " ".join(result["planning_warnings"]))
        self.assertEqual(result["ledger"]["generated_dialogue"][0]["text"], ("revised " * 32).strip())
        self.assertTrue(result["segments"])
        params = {"prompt": prompt, "model_type": "minimax_h3_ref2va_fused_turbo", "video_length": 672,
                  "sliding_window_size": 345, "minimax_h3_multi_window": True}
        prepare = AsyncMock(return_value={"params": params, "h3_window_plan": result})
        prepared = asyncio.run(prepare_enhanced_job(params,
            {"architecture": "minimax_h3", "fps": 24, "frames_maximum": 345}, AsyncMock(), prepare))
        self.assertTrue(prepared["enhancement_review_required"])

    def test_overlong_user_quote_still_reports_the_actual_duration_problem(self):
        prompt = 'Eva tells Nora "' + "word " * 50 + '"'
        with self.assertRaises(H3DialogueTimingError):
            plan_h3_story_segments(prompt, segment_durations=[10.1], mode="sliding_window",
                camera_coverage="multi_shot", planning_style="adaptive", llm_generate=Mock(return_value="{}"))

    def test_overlong_ai_dialogue_uses_dialogue_editor_without_rewriting_story(self):
        prompt = "Eva and Nora discuss their film project."
        story = ledger_for(prompt, [10.1])
        prose = "Eva spreads the storyboard on the desk while Nora draws a chair beside her."
        story["beats"][0]["description"] = prose
        story["generated_dialogue"] = [line("word " * 35, event="E1")]
        written = "Let's group the shots by scene first, then review the storyboard together so we can agree on what to film tomorrow morning."
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return json.dumps(story)
            if kwargs["prompt"].startswith("SHORTEN AI-WRITTEN LINES"):
                return json.dumps({"L1": written})
            self.assertNotIn("REPAIR THE COMPLETE STORY SCHEDULE", kwargs["prompt"])
            return json.dumps({
                "segment": 1, "title": "Storyboard discussion", "opening_state": "Eva and Nora are at the desk",
                "coverage": "single shot", "pacing": "natural", "closing_state": "Both study the storyboard",
                "shots": [{"shot": 1, "event_indices": [1], "start_seconds": 0, "end_seconds": 10.1,
                           "framing": "medium two-shot", "camera": "static", "action": prose,
                           "sound_effects": "Paper rustles", "transition": "opening"}],
            })

        result = plan_h3_story_segments(
            prompt, segment_durations=[10.1], mode="sliding_window", camera_coverage="multi_shot",
            planning_style="adaptive", llm_generate=generate,
        )
        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(result["planning_warnings"], [])
        self.assertEqual(result["ledger"]["beats"][0]["description"], prose)
        self.assertEqual(result["ledger"]["generated_dialogue"][0]["text"], written)
        self.assertEqual(len(calls), 3)
        self.assertGreaterEqual(calls[0]["max_new_tokens"], 3200)


if __name__ == "__main__":
    unittest.main()
