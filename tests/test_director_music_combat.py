"""Combat source obligations remain visible without making every clip a fight."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.music_narrative import (
    build_music_narrative_contract,
    combat_source_action,
    format_music_combat_context,
    music_narrative_row_issue,
)
from services.director.music_narrative_repair import compile_music_narrative_source_row


class MusicCombatTests(unittest.TestCase):
    def _assigned_row(self, contract: dict, prompt: str) -> tuple[dict, list[dict]]:
        event = contract["events"][0]
        allocation = [{"event_id": event["event_id"], "clip_index": 0}]
        row = {
            "source_event_ids": [event["event_id"]],
            "video_prompt": prompt,
        }
        return row, allocation

    def test_collective_fight_resolves_named_opponents_and_preserves_cast_credits(self):
        source = (
            "Captain Vega (played by Alex North) sings about hating Dr. Voss. "
            "They fight and have epic battles with high-budget effects, "
            "each using their respective powers."
        )
        contract = build_music_narrative_contract(source)

        self.assertEqual(len(contract["events"]), 1)
        event = contract["events"][0]
        self.assertEqual(event["kind"], "combat")
        self.assertEqual(event["participant_names"], ["Captain Vega", "Dr. Voss"])
        self.assertIn("vega", event["participant_alias_groups"][0])
        self.assertIn("voss", event["participant_alias_groups"][1])
        self.assertNotIn("alex north", str(event["participant_alias_groups"]).casefold())
        self.assertTrue(event["ongoing"])
        self.assertTrue(event["requires_distinct_powers"])
        self.assertEqual(
            combat_source_action(event),
            "Captain Vega and Dr. Voss fight and have epic battles with high-budget effects, each using their respective powers",
        )

    def test_collective_pronoun_can_resolve_article_led_creatures_without_hints(self):
        contract = build_music_narrative_contract(
            "The kitten sings about the dragon. They fight across the arena."
        )

        self.assertEqual(len(contract["events"]), 1)
        self.assertEqual(contract["events"][0]["participant_names"], ["kitten", "dragon"])
        self.assertEqual(
            contract["events"][0]["participant_alias_groups"],
            [["kitten"], ["dragon"]],
        )

    def test_nonphysical_or_lyric_mentions_do_not_create_combat_obligations(self):
        for source in (
            "Astra sings lyrics about fighting the Masked Regent.",
            "Astra fights her inner demons.",
            "Astra refuses to fight the Masked Regent.",
            "Two artists perform a rap battle.",
            "Astra fights in the battle of the bands.",
            "Astra sings a fight song beside the Regent.",
        ):
            with self.subTest(source=source):
                self.assertEqual(build_music_narrative_contract(source)["events"], [])

        dance_battle = build_music_narrative_contract("The dancers enter a dance battle.")
        self.assertEqual([event["kind"] for event in dance_battle["events"]], ["action"])
        dragon_attack = build_music_narrative_contract(
            "The kitten breathes fire over a battle field of enemies."
        )
        self.assertEqual(
            [event["text"] for event in dragon_attack["events"]],
            ["breathes fire over a battle field of enemies"],
        )

    def test_combat_coverage_needs_both_opponents_and_physical_exchange(self):
        contract = build_music_narrative_contract(
            "Captain Vega and Dr. Voss fight and have epic battles, each using their respective powers."
        )
        event = contract["events"][0]
        accepted = (
            "Captain Vega strikes at Dr. Voss with blue lightning. "
            "Dr. Voss deflects the bolt with an energy shield."
        )
        row, allocation = self._assigned_row(contract, accepted)
        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        ))
        row, allocation = self._assigned_row(
            contract,
            "Captain Vega and Dr. Voss fight in darkness, each using their respective powers.",
        )
        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        ))

        rejected_prompts = (
            "Captain Vega sings angrily about Dr. Voss and their fight.",
            "Captain Vega fights inner demons alone.",
            "Captain Vega and Dr. Voss are in a fighting stance.",
            "Captain Vega does not fight Dr. Voss.",
            "Captain Vega and Dr. Voss are about to fight.",
            'Captain Vega says, "I fight Dr. Voss," while standing still.',
            "Captain Vega strikes with blue lightning while Dr. Voss watches.",
        )
        for prompt in rejected_prompts:
            with self.subTest(prompt=prompt):
                row, allocation = self._assigned_row(contract, prompt)
                self.assertIn("both named opponents", music_narrative_row_issue(
                    row, contract=contract, allocation=allocation, index=0,
                ) or "")

    def test_combat_context_allows_solo_clips_and_does_not_promote_one_time_fight(self):
        ongoing = build_music_narrative_contract(
            "Captain Vega and Dr. Voss fight and have epic battles."
        )
        context = format_music_combat_context(ongoing)
        self.assertIn("need not appear in every clip or every frame", context)
        self.assertIn("solo performance", context)

        one_time = build_music_narrative_contract("Astra and the Masked Regent fight once.")
        self.assertFalse(one_time["events"][0]["ongoing"])
        self.assertEqual(format_music_combat_context(one_time), "")

    def test_source_recovery_declines_if_it_cannot_retain_both_opponents(self):
        contract = build_music_narrative_contract("Astra and the Masked Regent fight once.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        row = {
            "scene_goal": "Astra waits for the next musical phrase.",
            "scene_type": "narrative",
            "environment": "A dark stone arena.",
            "visual_style": "Cinematic fantasy realism.",
            "lighting": "Cool overhead light.",
            "mood": "Tense.",
            "ending_beat": "Astra turns toward the arena entrance.",
            "video_prompt": "[Shot 1] Astra waits by the arena wall.",
            "action_beats": ["Astra watches the entrance."],
            "source_event_ids": ["M1"],
            "subjects_on_screen": [{
                "character_id": "astra",
                "speaker_name": "Astra",
                "visual_description": "Astra in silver armor.",
                "position_or_relation": "beside the arena wall",
                "performance_role": "non_vocal",
            }],
            "camera_plan": {"framing": "medium-wide"},
            "window_prompts": [],
        }
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        )
        self.assertIn("both named opponents", issue or "")
        self.assertIsNone(compile_music_narrative_source_row(
            index=0,
            row=row,
            issue=issue or "missing combat participants",
            contract=contract,
            allocation=allocation,
        ))
        row["subjects_on_screen"].append({
            "character_id": "masked_regent",
            "speaker_name": "Masked Regent",
            "visual_description": "The Masked Regent in a black cloak.",
            "position_or_relation": "opposite Astra",
            "performance_role": "non_vocal",
        })
        recovered = compile_music_narrative_source_row(
            index=0, row=row, issue=issue, contract=contract, allocation=allocation,
        )
        self.assertIsNotNone(recovered)
        self.assertEqual(len(recovered[0]["subjects_on_screen"]), 2)
        self.assertIsNone(music_narrative_row_issue(
            recovered[0], contract=contract, allocation=allocation, index=0,
        ))


if __name__ == "__main__":
    unittest.main()
