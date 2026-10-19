"""Mixed performance/action concepts retain their visible conflict in the arc."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.music_arc import ground_music_combat_arc, normalize_music_arc
from services.director.music_narrative import build_music_narrative_contract, music_narrative_row_issue
from services.director.planners.music_video import MusicVideoPlanner


SOURCE = (
    "Astra sings a metal song about hating the Masked Regent. "
    "They fight and have epic battles, each using their respective powers. Cinematic."
)


def solo_rows():
    gestures = [
        "Astra sings quietly beside the empty microphone stand.",
        "Astra grips the microphone during the opening verse.",
        "Astra belts a note beneath a crossing light beam.",
        "Astra sings directly to the camera in the empty arena.",
        "Astra kneels as the bass changes to a sustained chord.",
        "Astra sings the chorus with one arm extended overhead.",
        "Astra turns toward the empty seats during an instrumental phrase.",
        "Astra lowers the microphone as the final chord decays.",
        "Astra bows beneath a single fading spotlight.",
    ]
    return [dict(clip_index=i, visual_beat=beat, purpose=f"Musical section {i}",
                 environment="a stone arena", camera=f"composition {i}",
                 source_event_ids=[], intentional_reprise=False, reprise_reason="")
            for i, beat in enumerate(gestures)]


class MusicCombatPlanningTests(unittest.TestCase):
    def setUp(self):
        self.contract = build_music_narrative_contract(SOURCE)
        self.event = next(e for e in self.contract["events"] if e["kind"] == "combat")
        self.clips = [{"start": i * 10, "end": (i + 1) * 10} for i in range(9)]

    def test_solo_outline_anchor_recovers_both_fighters_without_replacing_other_beats(self):
        rows = solo_rows()
        rows[3]["source_event_ids"] = [self.event["event_id"]]
        original = copy.deepcopy(rows)

        grounded, notes = ground_music_combat_arc(rows, self.contract)

        self.assertEqual(rows, original)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["clip_index"], 3)
        self.assertIn("Astra".casefold(), grounded[3]["visual_beat"].casefold())
        self.assertIn("regent", grounded[3]["visual_beat"].casefold())
        self.assertIn("powers", grounded[3]["visual_beat"])
        for i in range(len(rows)):
            if i != 3:
                self.assertEqual(grounded[i], rows[i])
        self.assertEqual(grounded[3]["camera"], rows[3]["camera"])
        self.assertEqual(grounded[3]["environment"], rows[3]["environment"])
        normalize_music_arc(grounded, self.clips, self.contract)

    def test_concrete_combat_outline_keeps_writer_direction(self):
        rows = solo_rows()
        rows[3]["source_event_ids"] = [self.event["event_id"]]
        rows[3]["visual_beat"] = (
            "Astra and the Masked Regent fight, exchanging lightning and energy blasts "
            "with their respective powers across the stone arena."
        )
        grounded, notes = ground_music_combat_arc(rows, self.contract)
        self.assertEqual(grounded, rows)
        self.assertEqual(notes, [])

    def test_battle_anchor_does_not_forbid_later_exchanges_or_force_every_clip_to_fight(self):
        allocation = [{"event_id": self.event["event_id"], "clip_index": 3}]
        later_battle = {"source_event_ids": [], "video_prompt": (
            "Astra and the Masked Regent fight across the shattered arena, "
            "using their respective powers as lightning strikes an energy shield."
        )}
        solo_insert = {"source_event_ids": [], "video_prompt": "Astra sings the chorus alone."}
        for row in (later_battle, solo_insert):
            self.assertIsNone(music_narrative_row_issue(
                row, contract=self.contract, allocation=allocation, index=5,
            ))

    def test_whole_song_planner_recovers_omitted_anchor_in_one_existing_writer_call(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"combat": 1})
        planner._music_narrative_contract = self.contract
        planner._use_lyrics_context = False
        rows = solo_rows()
        envelope = [{"beats": [{k: v for k, v in row.items() if k != "source_event_ids"}
                                for row in rows],
                     "source_event_clips": {self.event["event_id"]: 3}}]
        with patch.object(planner, "_call_llm_json", return_value=envelope) as writer:
            result = planner._plan_music_arc(self.clips, SOURCE)
        self.assertEqual(writer.call_count, 1)
        self.assertIn("regent", result[3]["visual_beat"].casefold())
        self.assertIn("powers", result[3]["visual_beat"])
        self.assertEqual(result[0], rows[0])
        self.assertTrue(any(n["kind"] == "source_combat_recovery"
                            for n in planner._music_arc_quality_notes))
        prompt = writer.call_args.kwargs["user_prompt"]
        self.assertIn("vocal performance does not replace requested story action", prompt)
        self.assertIn("powers", prompt)

    def test_performance_only_arc_does_not_invent_a_fight_or_opponent(self):
        contract = build_music_narrative_contract(
            "Astra sings a metal song about hating the Masked Regent. Cinematic."
        )
        rows = solo_rows()
        grounded, notes = ground_music_combat_arc(rows, contract)
        self.assertEqual(grounded, rows)
        self.assertEqual(notes, [])

    def test_brief_containing_only_a_fight_can_recover_its_one_assigned_anchor(self):
        contract = build_music_narrative_contract(
            "Astra and the Masked Regent fight using their respective powers."
        )
        event = next(e for e in contract["events"] if e["kind"] == "combat")
        rows = solo_rows()
        rows[3]["source_event_ids"] = [event["event_id"]]
        grounded, notes = ground_music_combat_arc(rows, contract)
        self.assertEqual(len(notes), 1)
        normalize_music_arc(grounded, self.clips, contract)
        # The same action in another unassigned slot is still a brief copy.
        grounded[4]["visual_beat"] = grounded[3]["visual_beat"]
        with self.assertRaisesRegex(ValueError, "full source brief"):
            normalize_music_arc(grounded, self.clips, contract)


if __name__ == "__main__":
    unittest.main()
