"""Bounded structured-dialogue repair and local H3 turn compilation."""

from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service
from services.dialogue_writing import (
    creative_dialogue_violations,
    dialogue_minimum_satisfied,
    dialogue_minimum_threshold,
    requested_dialogue_speakers,
    spoken_word_count,
)


GROUP_BRIEF = (
    "At a holiday party, Jerry Seinfeld, Michael Scott, and Ron Swanson are all "
    "standing and talking to each other. Write a brief conversation."
)
GROUP_BUDGET_BRIEF = (
    "At a holiday party, Jerry Seinfeld, Michael Scott, and Ron Swanson all "
    "stand and talk to each other."
)
LINES = [
    {"speaker": "Jerry Seinfeld", "text": "The holiday party starts with pizza and cake."},
    {"speaker": "Michael Scott", "text": "I'll bring paper plates tomorrow."},
    {"speaker": "Ron Swanson", "text": "Beer. Cake. Go home."},
]
EXACT_ANA_BEA_LINES = [
    "You kept the blue ticket.",
    "I kept both of them.",
    "Then why are you still here?",
    "Because you are.",
]
EXACT_ANA_BEA_PROMPT = (
    "Selected duration: 14.375 seconds.\n"
    "Preserve the supplied spoken lines beside their requested actions. Do not add or pad speech to fill the selected duration; develop the visual scene.\n\n"
    "User brief:\n"
    "At a quiet railway platform before dawn, adult sisters Ana and Bea wait beside one suitcase. "
    "Preserve these words exactly and assign them to the named speaker. "
    'Ana says, "You kept the blue ticket." Bea answers, "I kept both of them." '
    'Ana asks, "Then why are you still here?" Bea replies, "Because you are." '
    "Allow enough natural time for every line, reaction, and pause. Do not add, remove, paraphrase, split, or reorder dialogue."
)


def structured_script():
    rows = [
        f"{turn['speaker']} (S{index}) says, <d>[English] {turn['text']}</d>"
        for index, turn in enumerate(LINES, start=1)
    ]
    return (
        GROUP_BRIEF
        + "\n\nSpoken script for this adaptation. Only use these lines, in this order.\n"
        + "\n".join(rows)
    )


def frame(body):
    return (
        "integrated_multimodal_description: " + body + "\n"
        "overall_soundscape: Quiet party room tone and cloth movement.\n"
        "non_diegetic_music: N/A"
    )


class CreativeDialogueCompilationTests(unittest.TestCase):
    def test_group_cue_enforces_participation_but_cast_mentions_and_listeners_do_not(self):
        self.assertEqual(
            requested_dialogue_speakers(GROUP_BRIEF),
            ["Jerry Seinfeld", "Michael Scott", "Ron Swanson"],
        )
        self.assertEqual(
            requested_dialogue_speakers(
                "Ana and Bea talk to each other about the plan while a dog listens quietly."
            ),
            ["Ana", "Bea"],
        )
        self.assertEqual(
            requested_dialogue_speakers(
                "At opening, Ana, Bea, and Cara talk to each other about the plan. "
                "Later, Cara listens to a voice message."
            ),
            ["Ana", "Bea", "Cara"],
        )
        self.assertEqual(
            requested_dialogue_speakers(
                "At a holiday party, Jerry Seinfeld, Michael Scott, and Ron Swanson "
                "stand beside the buffet while Jerry and Michael talk to each other and Ron listens."
            ),
            ["Jerry Seinfeld", "Michael Scott"],
        )
        self.assertEqual(
            requested_dialogue_speakers(
                "At a holiday party, Jerry Seinfeld, Michael Scott, and Ron Swanson "
                "stand near the buffet. The room feels awkward."
            ),
            [],
        )
        silent = (
            "Jerry Seinfeld and Michael Scott talk to each other about the plan. "
            "Ron Swanson remains silent and listens while the other two talk."
        )
        self.assertEqual(
            requested_dialogue_speakers(silent),
            ["Jerry Seinfeld", "Michael Scott"],
        )

    def test_contract_reports_budget_missing_and_unrequested_speakers_and_exact_quote(self):
        violations = creative_dialogue_violations(
            GROUP_BRIEF,
            [{"speaker": "Ron Swanson", "text": "Beer. Cake. Go home."}],
            minimum_words=8,
            maximum_words=30,
        )
        self.assertTrue(any("at least 8" in issue for issue in violations))
        self.assertTrue(any("Jerry Seinfeld" in issue and "Michael Scott" in issue for issue in violations))

        topic_brief = GROUP_BRIEF + "\nTalking points:\n- holiday pizza\n- paper plates\n"
        topic_violations = creative_dialogue_violations(
            topic_brief,
            [{"speaker": "Ron Swanson", "text": "Beer. Cake. Go home."}],
            minimum_words=1,
            maximum_words=30,
        )
        self.assertTrue(any("missing spoken talking points" in issue for issue in topic_violations))
        self.assertEqual(
            creative_dialogue_violations(topic_brief, LINES, minimum_words=1, maximum_words=30),
            [],
        )

        extra_speaker = creative_dialogue_violations(
            GROUP_BRIEF,
            LINES[:2] + [{"speaker": "Dwight Schrute", "text": "I have the beets."}],
            minimum_words=1,
            maximum_words=30,
        )
        self.assertTrue(any("unrequested dialogue speakers: Dwight Schrute" in issue for issue in extra_speaker))

        quoted = 'Jerry and Michael talk to each other. Jerry says, "The pie stays here."'
        changed = creative_dialogue_violations(
            quoted,
            [{"speaker": "Jerry", "text": "The pie goes over there."}],
            minimum_words=1,
            maximum_words=30,
        )
        self.assertTrue(any("must remain verbatim" in issue for issue in changed))

    def test_automatic_minimum_is_a_complete_exchange_target_and_hard_max_stays_strict(self):
        self.assertEqual(dialogue_minimum_threshold(22, automatic=True, exchange_complete=True), 0)
        self.assertEqual(dialogue_minimum_threshold(22, automatic=True, exchange_complete=False), 22)
        self.assertTrue(dialogue_minimum_satisfied(16, 22, automatic=True))
        self.assertFalse(dialogue_minimum_satisfied(16, 22, automatic=True, exchange_complete=False))
        self.assertFalse(dialogue_minimum_satisfied(20, 22))

        complete_twenty_word_exchange = [
            {"speaker": "Jerry Seinfeld", "text": "That turkey disappeared before our holiday party started."},
            {"speaker": "Michael Scott", "text": "I brought it home because the office oven broke."},
            {"speaker": "Ron Swanson", "text": "It was delicious."},
        ]
        self.assertEqual(
            sum(spoken_word_count(turn["text"]) for turn in complete_twenty_word_exchange),
            20,
        )
        self.assertEqual(
            creative_dialogue_violations(
                GROUP_BUDGET_BRIEF,
                complete_twenty_word_exchange,
                minimum_words=22,
                maximum_words=28,
                automatic_minimum=True,
            ),
            [],
        )
        strict = creative_dialogue_violations(
            GROUP_BUDGET_BRIEF,
            complete_twenty_word_exchange,
            minimum_words=22,
            maximum_words=28,
        )
        self.assertTrue(any("at least 22 required" in issue for issue in strict))

        incomplete = creative_dialogue_violations(
            GROUP_BUDGET_BRIEF,
            [complete_twenty_word_exchange[0], complete_twenty_word_exchange[2]],
            minimum_words=22,
            maximum_words=28,
            automatic_minimum=True,
        )
        self.assertTrue(any("missing requested speaking participants" in issue for issue in incomplete))
        self.assertTrue(any("at least 22 required" in issue for issue in incomplete))

        over_maximum = complete_twenty_word_exchange + [
            {"speaker": "Jerry Seinfeld", "text": "One two three four five six seven eight nine ten eleven."}
        ]
        self.assertTrue(any(
            "at most 28 allowed" in issue
            for issue in creative_dialogue_violations(
                GROUP_BUDGET_BRIEF,
                over_maximum,
                minimum_words=22,
                maximum_words=28,
                automatic_minimum=True,
            )
        ))

    def test_qwen_shape_gets_one_full_exchange_repair_and_keeps_the_valid_turn(self):
        first = {"turns": [{"speaker": "Ron Swanson", "text": "Beer. Cake. Go home."}]}
        second = {"turns": [{"speaker": "Ron Swanson", "text": "Beer. Cake. Go home. Simple."}]}
        repaired_lines = LINES[:2] + [
            {"speaker": "Ron Swanson", "text": "Beer. Cake. Go home. Simple."}
        ]
        third = {"turns": repaired_lines}
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps((first, second, third)[len(calls) - 1])

        result = llm_service._draft_adaptive_dialogue_with_contract(
            GROUP_BRIEF, 10, generate, language="English",
        )

        self.assertEqual(len(calls), 3)
        self.assertIn("FULL EXCHANGE REPAIR", calls[1]["system_prompt"])
        self.assertIn("missing requested speaking participants", calls[1]["prompt"])
        self.assertIn("at least 8", calls[1]["prompt"])
        self.assertIn("Keep current valid speaker/text turns exactly", calls[2]["prompt"])
        parsed = llm_service._extract_h3_source_dialogue_entries(result)
        self.assertEqual([entry["speaker"] for entry in parsed], [turn["speaker"] for turn in LINES])
        self.assertEqual([entry["words"] for entry in parsed], [turn["text"] for turn in repaired_lines])
        self.assertIn("Ron Swanson (S3)", result)

    def test_complete_exchange_below_automatic_minimum_is_not_padded(self):
        short_turns = [
            {"speaker": "Jerry Seinfeld", "text": "That turkey disappeared before our holiday party started."},
            {"speaker": "Michael Scott", "text": "I brought it home because the office oven broke."},
            {"speaker": "Ron Swanson", "text": "Delicious."},
        ]
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps({"turns": short_turns})

        result = llm_service._draft_adaptive_dialogue_with_contract(
            GROUP_BUDGET_BRIEF, 10.125, generate, language="English",
        )

        self.assertEqual(len(calls), 1)
        entries = llm_service._extract_h3_source_dialogue_entries(result)
        self.assertEqual([entry["speaker"] for entry in entries], [
            turn["speaker"] for turn in short_turns
        ])
        self.assertEqual([entry["words"] for entry in entries], [
            turn["text"] for turn in short_turns
        ])

    def test_qwen_holiday_party_15_word_exchange_is_accepted_in_one_call(self):
        turns = [
            {"speaker": "Jerry Seinfeld", "text": "Why is this fruit salad so sad?"},
            {"speaker": "Michael Scott", "text": "It's got potential, Jerry."},
            {"speaker": "Ron Swanson", "text": "Give me a steak."},
        ]
        self.assertEqual(sum(spoken_word_count(turn["text"]) for turn in turns), 15)
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps({"turns": turns})

        result = llm_service._draft_adaptive_dialogue_with_contract(
            GROUP_BRIEF, 10.125, generate, language="English",
        )
        entries = llm_service._extract_h3_source_dialogue_entries(result)
        self.assertEqual(len(calls), 1)
        self.assertEqual([entry["speaker"] for entry in entries], [turn["speaker"] for turn in turns])
        self.assertEqual([entry["words"] for entry in entries], [turn["text"] for turn in turns])

    def test_explicit_numeric_spoken_minimum_remains_strict(self):
        prompt = GROUP_BUDGET_BRIEF + " Include at least 24 spoken words."
        contract = llm_service._adaptive_dialogue_contract(prompt, 10.125)
        self.assertIsNotNone(contract)
        self.assertEqual((contract[2], contract[5]), (24, False))
        short_turns = [
            {"speaker": "Jerry Seinfeld", "text": "Why is this fruit salad so sad?"},
            {"speaker": "Michael Scott", "text": "It's got potential, Jerry."},
            {"speaker": "Ron Swanson", "text": "Give me a steak."},
        ]
        long_turns = [
            {"speaker": "Jerry Seinfeld", "text": "Why is this fruit salad so sad at a holiday party?"},
            {"speaker": "Michael Scott", "text": "It's got potential, Jerry, if we add more cherries."},
            {"speaker": "Ron Swanson", "text": "Give me a steak and a plate."},
        ]
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps({"turns": short_turns if len(calls) == 1 else long_turns})

        llm_service._draft_adaptive_dialogue_with_contract(
            prompt, 10.125, generate, language="English",
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("at least 24", calls[1]["prompt"])

    def test_generated_wording_can_be_revised_to_cover_a_missing_topic(self):
        prompt = GROUP_BUDGET_BRIEF + "\nTalking points:\n- paper plates\n"
        first_turns = [
            {"speaker": "Jerry Seinfeld", "text": "That turkey disappeared before our holiday party started."},
            {"speaker": "Michael Scott", "text": "I brought it home because the office oven broke."},
            {"speaker": "Ron Swanson", "text": "It was delicious."},
        ]
        revised_turns = [
            first_turns[0],
            {"speaker": "Michael Scott", "text": "I brought paper plates because the office oven broke."},
            first_turns[2],
        ]
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps({"turns": first_turns if len(calls) == 1 else revised_turns})

        result = llm_service._draft_adaptive_dialogue_with_contract(
            prompt, 10.125, generate, language="English",
        )

        self.assertEqual(len(calls), 2)
        self.assertIn("missing spoken talking points", calls[1]["prompt"])
        self.assertIn("may revise their wording", calls[1]["prompt"])
        self.assertEqual(
            [entry["words"] for entry in llm_service._extract_h3_source_dialogue_entries(result)],
            [turn["text"] for turn in revised_turns],
        )

    def test_structured_turn_parser_rejects_schema_junk_and_markup(self):
        valid = json.dumps({"turns": [{"speaker": "Jerry", "text": "Hello there."}]})
        self.assertEqual(llm_service._parse_adaptive_dialogue_turns(valid)[0]["speaker"], "Jerry")
        for payload in (
            {"turns": [{"speaker": "Jerry", "text": "Hello."}], "debug": True},
            {"turns": [{"speaker": "Jerry", "text": "Hello.", "owner_id": "S1"}]},
            {"turns": [{"speaker": "Jerry", "text": "<d>Hello.</d>"}]},
        ):
            with self.subTest(payload=payload):
                self.assertIsNone(llm_service._parse_adaptive_dialogue_turns(json.dumps(payload)))
        self.assertIsNone(llm_service._parse_adaptive_dialogue_turns("Here is the JSON: " + valid))

    def test_compiler_preserves_visual_action_and_places_speech_at_named_cues(self):
        source = structured_script()
        output = frame(
            "[Shot 1] Jerry takes a small canapé from the buffet. Jerry delivers his line. "
            "Michael looks from Jerry to the empty table, then Michael responds. "
            "Ron folds his napkin, then Ron delivers his line. A waiter crosses behind them."
        )
        compiled = llm_service._compile_adaptive_h3_dialogue(source, output)
        entries = llm_service._extract_h3_source_dialogue_entries(compiled)
        self.assertEqual([entry["words"] for entry in entries], [turn["text"] for turn in LINES])
        self.assertEqual([entry["speaker_id"] for entry in entries], [1, 2, 3])
        self.assertIn("takes a small canapé", compiled)
        self.assertIn("looks from Jerry to the empty table", compiled)
        self.assertIn("folds his napkin", compiled)
        self.assertIn("A waiter crosses behind them", compiled)
        self.assertLess(compiled.index("takes a small canapé"), compiled.index(LINES[0]["text"]))
        self.assertLess(compiled.index(LINES[0]["text"]), compiled.index("looks from Jerry"))
        self.assertLess(compiled.index("looks from Jerry"), compiled.index(LINES[1]["text"]))
        self.assertLess(compiled.index(LINES[1]["text"]), compiled.index("folds his napkin"))

    def test_compiler_binds_exact_tags_and_refuses_owner_or_word_conflicts(self):
        source = structured_script()
        tagged = frame(
            "[Shot 1] Jerry smiles before speaking. Michael glances at the buffet. "
            "Ron sets down his napkin. "
            + " ".join(f"<d>[English] {turn['text']}</d>" for turn in LINES)
        )
        compiled = llm_service._compile_adaptive_h3_dialogue(source, tagged)
        self.assertEqual(compiled.count(" (S1) says, <d>"), 1)
        self.assertEqual(compiled.count(" (S2) says, <d>"), 1)
        self.assertEqual(compiled.count(" (S3) says, <d>"), 1)
        self.assertIn("Jerry smiles before speaking", compiled)
        self.assertIn("Michael glances at the buffet", compiled)

        wrong_owner = frame(
            "[Shot 1] Michael delivers the first line. Jerry responds. Ron delivers his line. "
            + " ".join(f"<d>[English] {turn['text']}</d>" for turn in LINES)
        )
        self.assertEqual(llm_service._compile_adaptive_h3_dialogue(source, wrong_owner), wrong_owner)

        swapped_ids = frame(
            "[Shot 1] Jerry (S2) says, <d>[English] " + LINES[0]["text"] + "</d>. "
            "Michael (S1) says, <d>[English] " + LINES[1]["text"] + "</d>. "
            "Ron (S3) says, <d>[English] " + LINES[2]["text"] + "</d>."
        )
        self.assertEqual(llm_service._compile_adaptive_h3_dialogue(source, swapped_ids), swapped_ids)

        changed_words = frame(
            "[Shot 1] Jerry says, <d>[English] The holiday party begins with pizza and cake.</d>. "
            "Michael says, <d>[English] " + LINES[1]["text"] + "</d>. "
            "Ron says, <d>[English] " + LINES[2]["text"] + "</d>."
        )
        self.assertEqual(llm_service._compile_adaptive_h3_dialogue(source, changed_words), changed_words)

    def test_compiler_binds_unique_named_coordinated_speech_cues(self):
        qwen_lines = [
            {"speaker": "Michael Scott", "text": "I brought a turkey!"},
            {"speaker": "Ron Swanson", "text": "It's dead."},
            {"speaker": "Jerry Seinfeld", "text": "They're all dead."},
        ]
        source = (
            GROUP_BRIEF
            + "\n\nSpoken script for this adaptation. Only use these lines, in this order.\n"
            + "\n".join(
                f"{turn['speaker']} (S{index}) says, <d>[English] {turn['text']}</d>"
                for index, turn in enumerate(qwen_lines, start=1)
            )
        )
        output = frame(
            "[Shot 1] Michael lifts the turkey with a proud grin. Michael says: "
            f"<d>[English] {qwen_lines[0]['text']}</d> "
            "Ron stares at the bird with a deadpan expression, then speaks at 3.20 seconds: "
            f"<d>[English] {qwen_lines[1]['text']}</d> "
            "Jerry looks between them, shrugging with mild confusion, and speaks at 5.20 seconds: "
            f"<d>[English] {qwen_lines[2]['text']}</d>"
        )
        compiled = llm_service._compile_adaptive_h3_dialogue(
            source, output, require_existing_tags=True,
        )
        entries = llm_service._extract_h3_source_dialogue_entries(compiled)
        self.assertEqual(
            [entry["words"] for entry in entries],
            [turn["text"] for turn in qwen_lines],
        )
        self.assertEqual([entry["speaker_id"] for entry in entries], [1, 2, 3])
        self.assertTrue(
            llm_service._h3_context_dialogue_binding_contract_satisfied(source, compiled)
        )
        self.assertLess(
            compiled.index("Ron stares at the bird"),
            compiled.index(qwen_lines[1]["text"]),
        )
        self.assertLess(
            compiled.index("Jerry looks between them"),
            compiled.index(qwen_lines[2]["text"]),
        )

        wrong_id = output.replace(
            "Ron stares at the bird", "Ron (S1) stares at the bird"
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                source, wrong_id, require_existing_tags=True,
            ),
            wrong_id,
        )

    def test_compiler_binds_v3_same_clause_named_subject_forms(self):
        output = frame(
            "[Shot 1] Ana subtly shifts her weight, tilting her head toward Bea, "
            "her expression soft and accusatory as she speaks: "
            f"<d>[English] {EXACT_ANA_BEA_LINES[0]}</d> "
            "Bea slowly turns to meet Ana's gaze, her demeanor measured, before answering: "
            f"<d>[English] {EXACT_ANA_BEA_LINES[1]}</d> "
            "Ana raises an eyebrow, her lips parting as she asks: "
            f"<d>[English] {EXACT_ANA_BEA_LINES[2]}</d> "
            "Bea remains still, holding Ana's gaze as she delivers the final, resolute line: "
            f"<d>[English] {EXACT_ANA_BEA_LINES[3]}</d>"
        )
        compiled = llm_service._compile_adaptive_h3_dialogue(
            EXACT_ANA_BEA_PROMPT, output, require_existing_tags=True,
        )
        entries = llm_service._extract_h3_source_dialogue_entries(compiled)
        self.assertEqual([entry["words"] for entry in entries], EXACT_ANA_BEA_LINES)
        self.assertEqual([entry["speaker_id"] for entry in entries], [1, 2, 1, 2])
        self.assertTrue(
            llm_service._h3_context_dialogue_binding_contract_satisfied(
                EXACT_ANA_BEA_PROMPT, compiled,
            )
        )
        for action in (
            "tilting her head toward Bea",
            "her demeanor measured, before answering",
            "her lips parting as she asks",
            "holding Ana's gaze as she delivers the final, resolute line",
        ):
            self.assertIn(action, compiled)

        creative_turns = [
            {"speaker": "Jerry Seinfeld", "text": "You just eat the little crackers, don't you?"},
            {"speaker": "Michael Scott", "text": "It's a party! We need fuel!"},
            {"speaker": "Ron Swanson", "text": "The quality is questionable."},
        ]
        creative_source = (
            GROUP_BRIEF
            + "\n\nSpoken script for this adaptation. Only use these lines, in this order.\n"
            + "\n".join(
                f"{turn['speaker']} (S{index}) says, <d>[English] {turn['text']}</d>"
                for index, turn in enumerate(creative_turns, start=1)
            )
        )
        creative_output = frame(
            "[Shot 1] Jerry leans toward Michael, his skeptical expression fixed as he speaks: "
            f"<d>[English] {creative_turns[0]['text']}</d> "
            "Michael mirrors Jerry's lean with a burst of energy, gesturing across the buffet as he exclaims: "
            f"<d>[English] {creative_turns[1]['text']}</d> "
            "Ron slowly shifts his weight, delivering his line with minimal movement: "
            f"<d>[English] {creative_turns[2]['text']}</d>"
        )
        creative_compiled = llm_service._compile_adaptive_h3_dialogue(
            creative_source, creative_output,
        )
        creative_entries = llm_service._extract_h3_source_dialogue_entries(creative_compiled)
        self.assertEqual(
            [entry["words"] for entry in creative_entries],
            [turn["text"] for turn in creative_turns],
        )
        self.assertEqual([entry["speaker_id"] for entry in creative_entries], [1, 2, 3])
        self.assertTrue(
            llm_service._h3_context_dialogue_binding_contract_satisfied(
                creative_source, creative_compiled,
            )
        )

    def test_compiler_binds_saved_v5_exact_named_subject_cues_and_refuses_conflicts(self):
        output = frame(
            "[Shot 1] At 0.00 seconds, both sisters remain still. "
            "At 1.50 seconds, Ana turns her head slightly toward Bea and says: "
            f"<d>[English] {EXACT_ANA_BEA_LINES[0]}</d> "
            "Bea remains still, her expression unreadable, and replies at 3.50 seconds: "
            f"<d>[English] {EXACT_ANA_BEA_LINES[1]}</d> "
            "Ana’s jaw tightens as she steps closer, asking at 6.50 seconds: "
            f"<d>[English] {EXACT_ANA_BEA_LINES[2]}</d> "
            "Bea holds Ana’s gaze, her lips parting to answer at 9.50 seconds: "
            f"<d>[English] {EXACT_ANA_BEA_LINES[3]}</d>"
        )
        compiled = llm_service._compile_adaptive_h3_dialogue(
            EXACT_ANA_BEA_PROMPT, output, require_existing_tags=True,
        )
        entries = llm_service._extract_h3_source_dialogue_entries(compiled)
        self.assertEqual([entry["words"] for entry in entries], EXACT_ANA_BEA_LINES)
        self.assertEqual([entry["speaker_id"] for entry in entries], [1, 2, 1, 2])
        for cue_action in (
            "turns her head slightly toward Bea and says",
            "jaw tightens as she steps closer, asking",
            "holds Ana’s gaze, her lips parting to answer",
        ):
            self.assertIn(cue_action, compiled)

        wrong_id = output.replace(
            "Ana turns her head", "Ana (S2) turns her head",
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                EXACT_ANA_BEA_PROMPT, wrong_id, require_existing_tags=True,
            ),
            wrong_id,
        )
        competing_subject = output.replace(
            "Ana’s jaw tightens as she steps closer, asking",
            "Ana listens while Bea speaks, then asks",
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                EXACT_ANA_BEA_PROMPT, competing_subject, require_existing_tags=True,
            ),
            competing_subject,
        )

    def test_compiler_replays_saved_v3_exact_and_creative_writer_outputs(self):
        attempts = (
            Path(__file__).resolve().parents[1]
            / ".codex-tmp"
            / "director-remediation-20261011"
            / "writer-results-v3"
            / "attempts"
        )
        if not attempts.exists():
            self.skipTest("saved v3 writer artifacts are not present")

        cases = (
            ("exact-dialogue", True),
            ("creative-word-budget", False),
        )
        replay_count = 0
        for case_id, require_existing_tags in cases:
            paths = sorted(attempts.glob(f"{case_id}--*.json"))
            if not paths:
                self.skipTest(f"saved v3 {case_id} attempts are not present")
            for path in paths:
                attempt = json.loads(path.read_text(encoding="utf-8"))
                calls = attempt["result"]["calls"]
                if require_existing_tags:
                    source = EXACT_ANA_BEA_PROMPT
                else:
                    structured = json.loads(calls[0]["output"])
                    turns = structured["turns"]
                    source = (
                        GROUP_BRIEF
                        + "\n\nSpoken script for this adaptation. Only use these lines, in this order.\n"
                        + "\n".join(
                            f"{turn['speaker']} (S{index}) says, <d>[English] {turn['text']}</d>"
                            for index, turn in enumerate(turns, start=1)
                        )
                    )
                source_entries = llm_service._unambiguous_h3_context_dialogue_entries(source)
                self.assertIsNotNone(source_entries, path.name)
                expected_words = [entry["words"] for entry in source_entries]
                expected_ids = [entry["speaker_id"] for entry in source_entries]
                for call in calls:
                    output = str(call.get("output") or "")
                    if "<d>" not in output:
                        continue
                    with self.subTest(artifact=path.name, call=call.get("index")):
                        compiled = llm_service._compile_adaptive_h3_dialogue(
                            source,
                            output,
                            require_existing_tags=require_existing_tags,
                        )
                        entries = llm_service._extract_h3_source_dialogue_entries(compiled)
                        self.assertEqual(
                            [entry["words"] for entry in entries], expected_words,
                        )
                        self.assertEqual(
                            [entry["speaker_id"] for entry in entries], expected_ids,
                        )
                        self.assertTrue(
                            llm_service._h3_context_dialogue_binding_contract_satisfied(
                                source, compiled,
                            )
                        )
                        replay_count += 1
        self.assertGreater(replay_count, 0)

    def test_compiler_refuses_object_pronoun_and_competing_named_subject_ambiguity(self):
        correct_suffix = " ".join(
            f"{speaker} {verb}, <d>[English] {text}</d>"
            for speaker, verb, text in (
                ("Bea", "replies", EXACT_ANA_BEA_LINES[1]),
                ("Ana", "asks", EXACT_ANA_BEA_LINES[2]),
                ("Bea", "answers", EXACT_ANA_BEA_LINES[3]),
            )
        )
        ambiguous_pronoun = frame(
            "Ana turns toward Bea, then she asks, "
            f"<d>[English] {EXACT_ANA_BEA_LINES[0]}</d> {correct_suffix}"
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                EXACT_ANA_BEA_PROMPT,
                ambiguous_pronoun,
                require_existing_tags=True,
            ),
            ambiguous_pronoun,
        )

        competing_subject = frame(
            "Ana turns toward Bea while Bea answers, "
            f"<d>[English] {EXACT_ANA_BEA_LINES[0]}</d> {correct_suffix}"
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                EXACT_ANA_BEA_PROMPT,
                competing_subject,
                require_existing_tags=True,
            ),
            competing_subject,
        )

        # A recognized first subject must not swallow another explicit speech
        # subject inside the broad clause, whether that person is in the cast
        # or is an unrecognized named speaker.
        for first_cue in (
            "Ana watches while Bea says something as she speaks",
            "Ana watches while Pat says something as she speaks",
            "Ana watches while Pat shouts something, then speaks",
            "Ana watches Pat delivering his line",
            "Ana listens while the waiter replies as she speaks",
            "Ana watches the waiter delivering his line",
            "Ana listens while Bea (S2) says something as she speaks",
        ):
            with self.subTest(first_cue=first_cue):
                competing_clause = frame(
                    first_cue
                    + ", <d>[English] "
                    + EXACT_ANA_BEA_LINES[0]
                    + "</d> "
                    + correct_suffix
                )
                self.assertEqual(
                    llm_service._compile_adaptive_h3_dialogue(
                        EXACT_ANA_BEA_PROMPT,
                        competing_clause,
                        require_existing_tags=True,
                    ),
                    competing_clause,
                )

        wrong_named_id = frame(
            "Ana (S2) tilts her head toward Bea, her expression soft as she speaks, "
            f"<d>[English] {EXACT_ANA_BEA_LINES[0]}</d> {correct_suffix}"
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                EXACT_ANA_BEA_PROMPT,
                wrong_named_id,
                require_existing_tags=True,
            ),
            wrong_named_id,
        )

    def test_exact_ordered_tags_ignore_visual_smile_as_a_false_speech_cue(self):
        output = frame(
            "[Shot 1] Ana gestures subtly toward the suitcase while speaking. "
            f"<d>[English] {EXACT_ANA_BEA_LINES[0]}</d> "
            "Bea slowly raises her head to meet Ana's gaze. "
            f"<d>[English] {EXACT_ANA_BEA_LINES[1]}</d> "
            "Ana's brow furrows as she watches Bea, her posture tightening. "
            f"<d>[English] {EXACT_ANA_BEA_LINES[2]}</d> "
            "Bea offers a small, gentle smile, her eyes lingering on Ana. "
            f"<d>[English] {EXACT_ANA_BEA_LINES[3]}</d>"
        )
        compiled = llm_service._compile_adaptive_h3_dialogue(
            EXACT_ANA_BEA_PROMPT, output, require_existing_tags=True,
        )
        self.assertEqual(compiled.count("Ana (S1) says, <d>"), 2)
        self.assertEqual(compiled.count("Bea (S2) says, <d>"), 2)
        self.assertTrue(
            llm_service._h3_context_dialogue_binding_contract_satisfied(
                EXACT_ANA_BEA_PROMPT, compiled,
            )
        )
        self.assertLess(compiled.index("Bea offers a small, gentle smile"),
                        compiled.index(EXACT_ANA_BEA_LINES[3]))

    def test_compiler_refuses_ambiguous_shared_first_names_and_pronoun_cues(self):
        collision_source = (
            'Michael Scott (S1) says, <d>[English] The lights are on.</d> '
            'Michael Corleone (S2) replies, <d>[English] I know.</d>'
        )
        collision_output = frame(
            "Michael says: <d>[English] The lights are on.</d> "
            "Michael says: <d>[English] I know.</d>"
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                collision_source, collision_output, require_existing_tags=True,
            ),
            collision_output,
        )

        pronoun_output = frame(
            f"She says: <d>[English] {EXACT_ANA_BEA_LINES[0]}</d> "
            f"He replies: <d>[English] {EXACT_ANA_BEA_LINES[1]}</d> "
            f"She asks: <d>[English] {EXACT_ANA_BEA_LINES[2]}</d> "
            f"He replies: <d>[English] {EXACT_ANA_BEA_LINES[3]}</d>"
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                EXACT_ANA_BEA_PROMPT, pronoun_output, require_existing_tags=True,
            ),
            pronoun_output,
        )

        unknown_speaker = frame(
            f"The waiter says: <d>[English] {EXACT_ANA_BEA_LINES[0]}</d>"
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                EXACT_ANA_BEA_PROMPT, unknown_speaker, require_existing_tags=True,
            ),
            unknown_speaker,
        )

        qwen_source = (
            "Ron Swanson (S1) says, <d>[English] Beer. Cake. Go home.</d>"
        )
        concatenated_alias = frame(
            "Ronsays: <d>[English] Beer. Cake. Go home.</d>"
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                qwen_source, concatenated_alias, require_existing_tags=True,
            ),
            concatenated_alias,
        )

    def test_legacy_object_name_mismatch_uses_unique_named_speaker_cue_only(self):
        result = frame(
            "Ana (S1) says, <d>[English] " + EXACT_ANA_BEA_LINES[0] + "</d> "
            "Bea (S2) replies, <d>[English] " + EXACT_ANA_BEA_LINES[1] + "</d> "
            "Ana (S1) raises an eyebrow slightly, her gaze fixed on Bea, and asks: "
            "<d>[English] " + EXACT_ANA_BEA_LINES[2] + "</d> "
            "Bea (S2) replies, <d>[English] " + EXACT_ANA_BEA_LINES[3] + "</d>"
        )
        legacy_entries = llm_service._unambiguous_h3_context_dialogue_entries(result)
        self.assertIsNotNone(legacy_entries)
        self.assertEqual(legacy_entries[2]["speaker"], "Bea")

        plan = llm_service._h3_context_speaker_binding_plan(EXACT_ANA_BEA_PROMPT, result)
        self.assertIsNotNone(plan)
        compiled = llm_service._compile_adaptive_h3_dialogue(
            EXACT_ANA_BEA_PROMPT,
            result,
            require_existing_tags=True,
        )
        self.assertEqual(llm_service._extract_h3_dialogue_blocks(compiled), EXACT_ANA_BEA_LINES)
        self.assertTrue(llm_service._h3_context_dialogue_binding_contract_satisfied(
            EXACT_ANA_BEA_PROMPT, compiled,
        ))
        self.assertIn("Ana (S1) raises an eyebrow slightly, her gaze fixed on Bea, and asks", compiled)

        competing_owner = frame(
            "Ana (S1) says, <d>[English] " + EXACT_ANA_BEA_LINES[0] + "</d> "
            "Bea (S2) replies, <d>[English] " + EXACT_ANA_BEA_LINES[1] + "</d> "
            "Ana turns toward Bea while Bea answers, then asks: "
            "<d>[English] " + EXACT_ANA_BEA_LINES[2] + "</d> "
            "Bea (S2) replies, <d>[English] " + EXACT_ANA_BEA_LINES[3] + "</d>"
        )
        self.assertIsNone(llm_service._h3_context_speaker_binding_plan(
            EXACT_ANA_BEA_PROMPT, competing_owner,
        ))

        wrong_id = frame(
            "Ana (S1) says, <d>[English] " + EXACT_ANA_BEA_LINES[0] + "</d> "
            "Bea (S2) replies, <d>[English] " + EXACT_ANA_BEA_LINES[1] + "</d> "
            "Ana (S2) raises an eyebrow slightly, her gaze fixed on Bea, and asks: "
            "<d>[English] " + EXACT_ANA_BEA_LINES[2] + "</d> "
            "Bea (S2) replies, <d>[English] " + EXACT_ANA_BEA_LINES[3] + "</d>"
        )
        self.assertIsNone(llm_service._h3_context_speaker_binding_plan(
            EXACT_ANA_BEA_PROMPT, wrong_id,
        ))

    def test_explicit_source_ids_bind_local_pronouns_and_descriptive_speakers(self):
        holiday_lines = [
            "This punch tastes like regret and artificial cherry.",
            "It’s actually grape, Jerry. You’re overthinking it.",
            "Give me a whiskey and I’ll be done.",
        ]
        holiday_source = (
            "Selected duration: 10.125 seconds.\nJerry Seinfeld, "
            "Michael Scott, and Ron Swanson are standing together at a company "
            "holiday party.\nSpoken script for this adaptation. Only use these "
            "lines in this order.\n"
            + "\n".join(
                f"{name} (S{index}) says, <d>[English] {line}</d>"
                for index, (name, line) in enumerate(zip(
                    ["Jerry Seinfeld", "Michael Scott", "Ron Swanson"],
                    holiday_lines,
                ), start=1)
            )
        )
        holiday_result = frame(
            "[Shot 1] A medium shot captures Jerry Seinfeld, Michael Scott, and "
            "Ron Swanson standing close together at a corporate holiday party. "
            "The background features blurred office decorations and warm string "
            "lights. Jerry (S1), wearing a brown blazer, holds a plastic cup and "
            "frowns at the liquid inside. He says: <d>[English] "
            + holiday_lines[0]
            + "</d> Michael (S2), in his signature gray suit, leans in with an "
            "exaggerated, earnest expression and replies: <d>[English] "
            + holiday_lines[1]
            + "</d> Ron (S3), wearing a brown corduroy jacket, stands rigidly to "
            "the side, eyes closed as if enduring a personal trial. He opens his "
            "eyes and states flatly: <d>[English] "
            + holiday_lines[2]
            + "</d>"
        )
        holiday_plan = llm_service._h3_context_speaker_binding_plan(
            holiday_source, holiday_result,
        )
        self.assertIsNotNone(holiday_plan)
        self.assertEqual([speaker_id for _, speaker_id in holiday_plan], [None, None, None])
        holiday_repaired = llm_service._repair_unambiguous_h3_context_speaker_ids(
            holiday_source, holiday_result,
        )
        repaired_holiday_plan = llm_service._h3_context_speaker_binding_plan(
            holiday_source, holiday_repaired,
        )
        self.assertEqual(
            [speaker_id for _, speaker_id in repaired_holiday_plan or []], [1, 2, 3],
        )
        self.assertEqual(
            [entry["words"] for entry in llm_service._extract_h3_source_dialogue_entries(
                holiday_repaired,
            )],
            holiday_lines,
        )
        self.assertTrue(llm_service._h3_context_dialogue_binding_contract_satisfied(
            holiday_source, holiday_repaired,
        ))

        anonymous_source = (
            "Selected duration: 10.125 seconds.\nUser brief:\nWrite a short "
            "conversation between two strangers waiting for a delayed train.\n"
            "Spoken script for this adaptation. Only use these lines in this order.\n"
            "Anxious Passenger (S1) says, <d>[English] We'll share the seat?</d>\n"
            "Calm Passenger (S2) says, <d>[English] Deal.</d>"
        )
        anonymous_result = frame(
            "[Shot 1] The Anxious Passenger (S1), a woman in a beige coat, "
            "fidgets with her hands. The Calm Passenger (S2), a man in a navy "
            "sweater, sits upright. At 0.00 seconds, S1 looks up from her phone. "
            "At 2.50 seconds, S1 turns to S2 and asks: <d>[English] "
            "We'll share the seat?</d> S2 listens and nods. At 5.00 seconds, "
            "S2 replies calmly: <d>[English] Deal.</d>"
        )
        anonymous_plan = llm_service._h3_context_speaker_binding_plan(
            anonymous_source, anonymous_result,
        )
        self.assertIsNotNone(anonymous_plan)
        self.assertEqual([speaker_id for _, speaker_id in anonymous_plan], [None, None])
        anonymous_repaired = llm_service._repair_unambiguous_h3_context_speaker_ids(
            anonymous_source, anonymous_result,
        )
        repaired_anonymous_plan = llm_service._h3_context_speaker_binding_plan(
            anonymous_source, anonymous_repaired,
        )
        self.assertEqual(
            [speaker_id for _, speaker_id in repaired_anonymous_plan or []], [1, 2],
        )
        self.assertEqual(
            llm_service._repair_unambiguous_h3_context_speaker_ids(
                anonymous_source, anonymous_repaired,
            ),
            anonymous_repaired,
        )
        self.assertEqual(
            [entry["words"] for entry in llm_service._extract_h3_source_dialogue_entries(
                anonymous_repaired,
            )],
            ["We'll share the seat?", "Deal."],
        )
        self.assertTrue(llm_service._h3_context_dialogue_binding_contract_satisfied(
            anonymous_source, anonymous_repaired,
        ))

        borrowed_id_source = (
            "Selected duration: 10.125 seconds.\nUser brief:\nAna and Bea are "
            "walking through a doorway.\nSpoken script for this adaptation. "
            "Only use these lines in this order.\n"
            "Ana (S1) says, <d>[English] Hold the door.</d>\n"
            "Bea (S2) says, <d>[English] I have it.</d>"
        )
        borrowed_id_result = frame(
            "Morgan (S1) says: <d>[English] Hold the door.</d> "
            "Bea (S2) replies: <d>[English] I have it.</d>"
        )
        self.assertIsNone(llm_service._h3_context_speaker_binding_plan(
            borrowed_id_source, borrowed_id_result,
        ))
        self.assertFalse(llm_service._h3_context_dialogue_binding_contract_satisfied(
            borrowed_id_source, borrowed_id_result,
        ))

        wrong_named_owner = frame(
            "Bea (S1) says: <d>[English] " + EXACT_ANA_BEA_LINES[0] + "</d>"
        )
        wrong_id = frame(
            "S2 says: <d>[English] " + EXACT_ANA_BEA_LINES[0] + "</d>"
        )
        ambiguous_pronoun = frame(
            "A woman looks up. She says: <d>[English] "
            + EXACT_ANA_BEA_LINES[0]
            + "</d>"
        )
        changed_quote = frame(
            "Ana (S1) says: <d>[English] You kept the red ticket.</d>"
        )
        for invalid in [wrong_named_owner, wrong_id, ambiguous_pronoun, changed_quote]:
            with self.subTest(invalid=invalid):
                self.assertFalse(llm_service._h3_context_dialogue_binding_contract_satisfied(
                    EXACT_ANA_BEA_PROMPT, invalid,
                ))

    def test_local_named_and_pronoun_cues_bind_only_to_their_subject_clause(self):
        source = (
            "Selected duration: 8 seconds.\nUser brief:\nAna and Bea wait at a "
            "doorway.\nSpoken script for this adaptation. Only use these lines "
            "in this order.\n"
            "Ana (S1) says, <d>[English] Hold the door.</d>\n"
            "Bea (S2) says, <d>[English] I have it.</d>"
        )
        exact_lines = ["Hold the door.", "I have it."]
        named_then_pronoun = frame(
            "Ana says: <d>[English] Hold the door.</d> "
            "Bea reaches for the handle. She replies: "
            "<d>[English] I have it.</d>"
        )
        plan = llm_service._h3_context_speaker_binding_plan(source, named_then_pronoun)
        self.assertIsNotNone(plan)
        self.assertEqual([speaker_id for _, speaker_id in plan], [None, None])
        repaired = llm_service._repair_unambiguous_h3_context_speaker_ids(
            source, named_then_pronoun,
        )
        repaired_plan = llm_service._h3_context_speaker_binding_plan(source, repaired)
        self.assertEqual([speaker_id for _, speaker_id in repaired_plan or []], [1, 2])
        self.assertEqual(
            [entry["words"] for entry in llm_service._extract_h3_source_dialogue_entries(
                repaired,
            )],
            exact_lines,
        )
        self.assertTrue(
            llm_service._h3_context_dialogue_binding_contract_satisfied(source, repaired)
        )

        rejected = [
            frame(
                "Morgan says: <d>[English] Hold the door.</d> "
                "Bea replies: <d>[English] I have it.</d>"
            ),
            frame(
                "A passerby gestures toward Ana. She says: "
                "<d>[English] Hold the door.</d> Bea replies: "
                "<d>[English] I have it.</d>"
            ),
            frame(
                "Ana and Bea glance toward the door. She says: "
                "<d>[English] Hold the door.</d> Bea replies: "
                "<d>[English] I have it.</d>"
            ),
        ]
        for candidate in rejected:
            with self.subTest(candidate=candidate):
                self.assertIsNone(
                    llm_service._h3_context_speaker_binding_plan(source, candidate)
                )
                self.assertFalse(
                    llm_service._h3_context_dialogue_binding_contract_satisfied(
                        source, candidate,
                    )
                )

    def test_duration_header_is_not_a_turn_and_exact_tags_get_proven_speaker_ids(self):
        source_entries = llm_service._unambiguous_h3_context_dialogue_entries(
            EXACT_ANA_BEA_PROMPT
        )
        self.assertIsNotNone(source_entries)
        self.assertEqual(
            [entry["words"] for entry in source_entries],
            EXACT_ANA_BEA_LINES,
        )
        self.assertEqual(
            [entry["speaker"] for entry in source_entries],
            ["Ana", "Bea", "Ana", "Bea"],
        )
        self.assertEqual(
            [entry["speaker_id"] for entry in source_entries],
            [1, 2, 1, 2],
        )
        self.assertEqual(
            llm_service._extract_h3_quoted_dialogue(EXACT_ANA_BEA_PROMPT),
            EXACT_ANA_BEA_LINES,
        )

        tagged = frame(
            "[Shot 1] "
            + " ".join(f"<d>[English] {line}</d>" for line in EXACT_ANA_BEA_LINES)
        )
        compiled = llm_service._compile_adaptive_h3_dialogue(
            EXACT_ANA_BEA_PROMPT,
            tagged,
            require_existing_tags=True,
        )
        self.assertEqual(compiled.count("Ana (S1) says, <d>"), 2)
        self.assertEqual(compiled.count("Bea (S2) says, <d>"), 2)
        self.assertTrue(
            llm_service._h3_dialogue_contract_satisfied(EXACT_ANA_BEA_PROMPT, compiled)
        )
        self.assertTrue(
            llm_service._h3_context_dialogue_binding_contract_satisfied(
                EXACT_ANA_BEA_PROMPT, compiled
            )
        )

        wrong_owner = frame(
            "[Shot 1] "
            "Bea answers, <d>[English] " + EXACT_ANA_BEA_LINES[0] + "</d> "
            "Ana replies, <d>[English] " + EXACT_ANA_BEA_LINES[1] + "</d> "
            "Ana asks, <d>[English] " + EXACT_ANA_BEA_LINES[2] + "</d> "
            "Bea replies, <d>[English] " + EXACT_ANA_BEA_LINES[3] + "</d>"
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                EXACT_ANA_BEA_PROMPT,
                wrong_owner,
                require_existing_tags=True,
            ),
            wrong_owner,
        )

        cue_only = frame(
            "[Shot 1] Ana says the first line. Bea responds with the second line."
        )
        self.assertEqual(
            llm_service._compile_adaptive_h3_dialogue(
                EXACT_ANA_BEA_PROMPT,
                cue_only,
                require_existing_tags=True,
            ),
            cue_only,
        )

    def test_untimed_exact_dialogue_needs_native_blocks_and_respects_word_budget(self):
        tagged = frame(
            "[Shot 1] "
            + " ".join(f"<d>[English] {line}</d>" for line in EXACT_ANA_BEA_LINES)
        )
        self.assertIsNone(
            llm_service._h3_explicit_source_dialogue_timing(EXACT_ANA_BEA_PROMPT)
        )
        self.assertTrue(
            llm_service._h3_speech_timing_satisfied(
                EXACT_ANA_BEA_PROMPT, tagged, 14.375, "faithful"
            )
        )
        self.assertFalse(
            llm_service._h3_speech_timing_satisfied(
                EXACT_ANA_BEA_PROMPT,
                frame("[Shot 1] Ana speaks naturally while Bea listens."),
                14.375,
                "faithful",
            )
        )
        # The four exact lines contain 19 spoken words, so six seconds at the
        # existing three-words-per-second ceiling is not enough.
        self.assertFalse(
            llm_service._h3_speech_timing_satisfied(
                EXACT_ANA_BEA_PROMPT, tagged, 6, "faithful"
            )
        )
        self.assertTrue(
            llm_service._h3_speech_timing_satisfied(
                EXACT_ANA_BEA_PROMPT, tagged, 7, "faithful"
            )
        )

    def test_explicit_source_intervals_must_be_preserved_with_their_quiet_state(self):
        source = (
            "From 0 to 2 seconds, Ana and Bea remain quiet; both mouths stay closed "
            "and no voices are heard. From 2 to 7 seconds, Ana and Bea speak these "
            "exact lines in order: "
            'Ana says, "You kept the blue ticket." Bea answers, "I kept both of them." '
            'Ana asks, "Then why are you still here?" Bea replies, "Because you are." '
            "From 7 to 10 seconds, both remain quiet with their mouths closed and no voices."
        )
        correct = frame(
            "[Shot 1] From 0.00 to 2.00 seconds, both mouths remain closed and no human voices are heard. "
            "From 2.00 to 7.00 seconds, Ana and Bea exchange the exact lines in order: "
            + " ".join(f"<d>[English] {line}</d>" for line in EXACT_ANA_BEA_LINES)
            + " From 7.00 to 10.00 seconds, both mouths remain closed and no voices are heard."
        )
        self.assertTrue(
            llm_service._h3_speech_timing_satisfied(source, correct, 10, "faithful")
        )

        missing_speech_window = correct.replace(
            "From 2.00 to 7.00 seconds, ", "During the conversation, "
        )
        self.assertFalse(
            llm_service._h3_speech_timing_satisfied(
                source, missing_speech_window, 10, "faithful"
            )
        )

        wrong_speech_window = correct.replace(
            "From 2.00 to 7.00 seconds, ", "From 3.00 to 7.00 seconds, "
        )
        self.assertFalse(
            llm_service._h3_speech_timing_satisfied(
                source, wrong_speech_window, 10, "faithful"
            )
        )

        missing_quiet_state = correct.replace(
            "both mouths remain closed and no human voices are heard",
            "the platform hums softly",
        )
        self.assertFalse(
            llm_service._h3_speech_timing_satisfied(
                source, missing_quiet_state, 10, "faithful"
            )
        )

        adjacent_quiet_borrow = correct.replace(
            "From 0.00 to 2.00 seconds, both mouths remain closed and no human voices are heard.",
            "From 0.00 to 2.00 seconds, the platform lights reflect across the rails.",
        )
        self.assertFalse(
            llm_service._h3_speech_timing_satisfied(
                source, adjacent_quiet_borrow, 10, "faithful"
            )
        )

        adjacent_mouth_borrow = correct.replace(
            "both mouths remain closed", "both sisters stay silent", 1
        )
        self.assertFalse(
            llm_service._h3_speech_timing_satisfied(
                source, adjacent_mouth_borrow, 10, "faithful"
            )
        )

    def test_explicit_speech_point_requires_that_point_or_an_interval_boundary(self):
        source = 'At 2 seconds Ana says, "You kept the blue ticket."'
        exact_point = frame(
            "[Shot 1] At 2.00 seconds Ana says, "
            "<d>[English] You kept the blue ticket.</d>"
        )
        broad_interval = frame(
            "[Shot 1] From 0 to 10 seconds Ana says, "
            "<d>[English] You kept the blue ticket.</d>"
        )
        self.assertTrue(
            llm_service._h3_speech_timing_satisfied(source, exact_point, 10, "faithful")
        )
        self.assertFalse(
            llm_service._h3_speech_timing_satisfied(
                source, broad_interval, 10, "faithful"
            )
        )

    def test_soundscape_silence_removes_unrequested_chatter_but_keeps_practical_sounds(self):
        prompt = (
            "At a company holiday party, Ana and Bea talk to each other. "
            "Outside the tagged dialogue, no human voices, whispers, grunts, audible breathing, "
            "or speech-like vocalizations occur."
        )
        output = (
            "integrated_multimodal_description: [Shot 1] Ana and Bea exchange a few lines.\n"
            "overall_soundscape: Low office party chatter, and the faint clink of plastic cups against each other.\n"
            "non_diegetic_music: N/A"
        )
        cleaned = llm_service._enforce_h3_soundscape_silence(output, prompt)
        self.assertNotIn("office party chatter", cleaned.casefold())
        self.assertIn("faint clink of plastic cups", cleaned)
        self.assertNotIn("and the faint clink", cleaned)

    def test_soundscape_silence_preserves_an_explicit_nonverbal_reaction(self):
        prompt = "Ana and Bea talk. Ron gives a brief laugh after the last line."
        output = (
            "integrated_multimodal_description: [Shot 1] Ron smiles.\n"
            "overall_soundscape: Ron's brief laugh, and the faint clink of a cup.\n"
            "non_diegetic_music: N/A"
        )
        self.assertEqual(
            llm_service._enforce_h3_soundscape_silence(output, prompt),
            output,
        )

    def test_soundscape_silence_neutralizes_only_unrequested_visual_laughter(self):
        prompt = (
            structured_script()
            + "\nDo not add laughter or other vocalizations outside the dialogue."
        )
        output = (
            'integrated_multimodal_description: [Shot 1] Jerry chuckles softly at Ron\'s assessment. '
            'Bea says, <d>[English] "I laugh when nervous."</d> '
            'A nearby sign reads "Laugh now."\n'
            "overall_soundscape: A cup taps against the table.\n"
            "non_diegetic_music: N/A"
        )
        cleaned = llm_service._enforce_h3_soundscape_silence(output, prompt)
        self.assertIn("Jerry smiles softly at Ron's assessment", cleaned)
        self.assertNotIn("chuckles", cleaned.casefold())
        self.assertIn('<d>[English] "I laugh when nervous."</d>', cleaned)
        self.assertIn('"Laugh now."', cleaned)


if __name__ == "__main__":
    unittest.main()
