"""End-to-end prompt integrity for imported, detailed shooting briefs."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.h3_authored_brief import authored_timed_brief, explicit_character_profiles
from services.h3_story_ledger import (
    _canonicalize_segment_contract, _expected_dialogue_events, _deterministic_ledger,
    extract_h3_source_intent, extract_locked_dialogue, extract_source_events,
)
from services.h3_window_planner import _parse_json_object, plan_h3_sliding_windows
from services.h3_prompt_budget import fit_h3_base_prompt


class AuthoredBriefTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "tests/fixtures/h3_silent_wuxia_prompt.txt").read_text(encoding="utf-8")

    def test_inline_and_multiline_briefs_keep_authored_phases_not_production_notes(self):
        for source in (self.source, " ".join(self.source.split())):
            with self.subTest(flattened="\n" not in source):
                brief = authored_timed_brief(source)
                self.assertEqual(len(brief["events"]), 8)
                self.assertEqual([(item["source_start_seconds"], item["source_end_seconds"]) for item in brief["events"]],
                                 [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 27), (27, 30)])
                events = extract_source_events(source)
                self.assertEqual(len(events), 8)
                self.assertIn("hundreds of meters", events[0]["text"])
                self.assertIn("frozen in clash pose", events[-1]["text"])
                self.assertNotIn("VFX Requirements", " ".join(event["text"] for event in events))
                self.assertIn("VFX Requirements", brief["context"])
                self.assertIn("Biased toward explosive fist techniques", brief["context"])
                intent = extract_h3_source_intent(source)
                self.assertEqual(intent["cast_names"], ["Character A", "Character B"])
                self.assertIn("no third parties", intent["negative_constraints"])
                self.assertNotIn("VFX Requirements", intent["negative_constraints"])
                self.assertNotIn("VFX centers", intent["negative_constraints"])

    def test_other_time_notations_and_profiles_do_not_require_line_breaks(self):
        source = "Subject 1 (red coat): waits. Subject 2 (blue hat): runs. [00:00–00:03] A courier opens a gate. [00:03–00:07] She cycles away. [Camera notes] Natural light."
        self.assertEqual([item["name"] for item in explicit_character_profiles(source)], ["Subject 1", "Subject 2"])
        brief = authored_timed_brief(source)
        self.assertEqual([item["text"] for item in brief["events"]], ["A courier opens a gate.", "She cycles away."])
        self.assertIn("Natural light", brief["context"])
        for unstructured in ("Hold for 0-4s, then run for 4-8s.", "[0s-4s] Open. [2s-6s] Run."):
            self.assertEqual(authored_timed_brief(unstructured)["events"], [])

    def test_camera_and_sound_fields_inside_a_timed_shot_remain_local(self):
        source = (
            "[0s-3s]\nCamera: Push toward the gate.\nAction: A courier arrives.\nVFX notes: A gust lifts the dust.\n"
            "[3s-7s]\nCamera: Track the courier.\nAction: She cycles away.\nSound: Footsteps fade.\n"
            "[Cinematography Requirements] Natural daylight throughout."
        )
        brief = authored_timed_brief(source)
        self.assertEqual(len(brief["events"]), 2)
        self.assertIn("A gust lifts the dust", brief["events"][0]["text"])
        self.assertIn("Track the courier", brief["events"][1]["text"])
        self.assertIn("Footsteps fade", brief["events"][1]["text"])
        self.assertNotIn("Track the courier", brief["context"])
        self.assertIn("Natural daylight throughout", brief["context"])

    def test_final_prompts_keep_complete_ai_actions_cameras_constraints_and_ending(self):
        source = " ".join(self.source.split())
        calls = []
        actions = [
            "Character A parries, winds up his fist, and punches Character B hundreds of meters into the cliff.",
            "Character B smashes the mountain face, pauses in the rubble, then stomps and launches back toward A.",
            "Character B's spinning whip kick blasts Character A across the platform and embeds him in the opposite cliff.",
            "Character A tears free, runs along the cliff, and dives fist-first toward B, who charges an upward kick.",
            "A's downward fist meets B's aerial kick; the platform cracks and both spin back into combat stances.",
            "B skims into a sweep; A vaults and elbows; B counters with a roundhouse; A fires two cannon punches.",
            "They separate to opposite ends; A charges compressed fist rings while B draws rubble into rotating leg winds.",
            "A's ultimate fist and B's storm whip kick collide; the platform collapses and they remain frozen in clash pose.",
        ]
        # Deliberately put a meaningful consequence beyond the former 330-char
        # cut and make the complete prompt exceed the cosmetic token target.
        for index in range(8):
            actions[index] += (
                " The grey and earth-yellow robes react to the wind pressure; the force has a clear source, "
                "direction and physical contact. Transparent compression rings expand across the fractured stone, "
                "driving dust, waterfall spray and fragments outward without changing the two fighters' identities. "
                f"The final physical consequence of phase {index + 1} remains visible."
            )

        def generate(**kwargs):
            calls.append(kwargs)
            if "setting_continuity" in kwargs["json_schema"]["properties"]:
                self.assertEqual(kwargs["json_schema"]["properties"]["character_appearance"]["required"], ["Character A", "Character B"])
                return json.dumps({"character_appearance": {"Character A": "An adult Asian male grandmaster in gray robes.",
                                                          "Character B": "An adult Asian male grandmaster in earth-yellow monk robes."},
                                   "setting_continuity": "Ruined mountain platform, cliffs, statues and waterfalls.",
                                   "visual_continuity": "Realistic live-action wuxia with transparent air impacts.",
                                   "editing_style": "Slow tension followed by explosive action.", "ambient_audio": "Mountain wind."})
            number = kwargs["json_schema"]["properties"]["segment"]["minimum"]
            self.assertGreaterEqual(kwargs["max_new_tokens"], 4096)
            self.assertEqual(kwargs["json_schema"]["properties"]["shots"]["minItems"], 4)
            self.assertIn("event_indices", kwargs["json_schema"]["properties"]["shots"]["items"]["required"])
            duration = [14.375, 13.625][number - 1]
            return json.dumps({
                "segment": number, "title": "The duel", "opening_state": "Match the supplied scene.",
                "coverage": "Readable impact coverage", "pacing": "Authored slow motion and explosive speed",
                "closing_state": "A plummets toward B's charged kick." if number == 1 else "Both fighters remain frozen in clash pose amid the ruined platform.",
                "shots": [{"shot": index + 1, "event_indices": [index + 1],
                           "start_seconds": duration * index / 4, "end_seconds": duration * (index + 1) / 4,
                           "transition": "cut", "framing": "Wide view of Character A and Character B",
                           "camera": f"Follow the contact and travel described in phase {4 * (number - 1) + index + 1}.",
                           "action": actions[4 * (number - 1) + index], "sound_effects": "Stone impacts and wind."}
                          for index in range(4)],
            })

        with patch("services.llm_service.generate", side_effect=generate):
            result = plan_h3_sliding_windows(source, model_type="minimax_h3_fused_turbo", resolution="1280x704",
                                            total_frames=672, window_frames=345, overlap_frames=18,
                                            fps=24, planning_style="faithful", has_start_image=True)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["planning_warnings"], [])
        self.assertEqual(result["source_intent"]["cast_names"], ["Character A", "Character B"])
        self.assertLess(result["windows"][0]["shots"][0]["end_seconds"], 4)
        self.assertEqual(result["windows"][1]["opening_state"], result["windows"][0]["closing_state"])
        for number, prompt in enumerate(result["window_prompts"]):
            for action in actions[number * 4:(number + 1) * 4]:
                self.assertIn(action, prompt)
            self.assertIn("no third parties", prompt)
            self.assertIn("no weapons", prompt)
            self.assertIn("no dialogue", prompt)
            self.assertIn("Character A: An adult Asian male grandmaster in gray robes.", prompt)
            self.assertIn("Character B: An adult Asian male grandmaster in earth-yellow monk robes.", prompt)
            self.assertNotIn("Lens,", prompt)
            self.assertNotIn("<d>", prompt)
        self.assertEqual(result["windows"][-1]["closing_state"], "Both fighters remain frozen in clash pose amid the ruined platform.")

        # A provider can ignore the requested schema and return only its
        # first phase. Keep that draft for one focused repair, not a silent
        # action replacement or an immediate deterministic fallback.
        calls.clear()
        incomplete_returned = False

        def partial_once(**kwargs):
            nonlocal incomplete_returned
            response = generate(**kwargs)
            if not incomplete_returned and "segment" in kwargs["json_schema"]["properties"]:
                incomplete_returned = True
                partial = json.loads(response)
                partial["shots"] = partial["shots"][:1]
                return json.dumps(partial)
            return response

        with patch("services.llm_service.generate", side_effect=partial_once):
            repaired = plan_h3_sliding_windows(source, model_type="minimax_h3_fused_turbo", resolution="1280x704",
                                              total_frames=672, window_frames=345, overlap_frames=18,
                                              fps=24, planning_style="faithful", has_start_image=True)
        self.assertEqual(len(calls), 4)
        self.assertIn("REPAIR ONLY THIS SEGMENT", calls[2]["prompt"])
        self.assertEqual(repaired["planning_warnings"], [])
        for action in actions[:4]:
            self.assertIn(action, repaired["window_prompts"][0])

    def test_incomplete_camera_json_is_not_repaired_into_a_successful_draft(self):
        self.assertIsNone(_parse_json_object('{"segment": 2, "shots": [{"action": "unfinished', allow_repair=False))
        self.assertEqual(_parse_json_object('```json\n{"segment": 2}\n```', allow_repair=False), {"segment": 2})

    def test_multiple_dialogue_lines_stay_in_their_authored_phase(self):
        source = (
            '【0s - 6s】 Alice opens the door and says, "Come in." '
            'Bob steps inside and replies, "Thank you." '
            '【6s - 12s】 Alice closes the door and says, "We are safe."'
        )
        locked = extract_locked_dialogue(source)
        self.assertEqual([item["text"] for item in locked], ["Come in.", "Thank you.", "We are safe."])
        self.assertEqual(_expected_dialogue_events(source, locked), {"D1": "E1", "D2": "E1", "D3": "E2"})
        ledger = _deterministic_ledger(source, segment_count=2, segment_durations=[6, 6],
                                       locked_dialogue=locked, camera_coverage="multi_shot", reference_context="")
        self.assertEqual([item for beat in ledger["beats"] if beat["segment"] == 1 for item in beat["dialogue_ids"]], ["D1", "D2"])
        self.assertIn("Bob steps inside", ledger["beats"][0]["description"])
        self.assertEqual([item for beat in ledger["beats"] if beat["segment"] == 2 for item in beat["dialogue_ids"]], ["D3"])

    def test_written_sign_text_is_not_speech_but_a_person_beside_it_can_speak(self):
        source = 'The door says "EXIT". Alice approaches the door and says, "Follow me."'
        self.assertEqual([item["text"] for item in extract_locked_dialogue(source)], ["Follow me."])

    def test_uneven_event_groups_stay_with_their_own_camera_and_action(self):
        beats = [{"beat_id": f"B{i}", "description": f"Source phase {i}", "dialogue_ids": [],
                  "authored_duration": 1} for i in range(1, 5)]
        candidate = {"shots": [
            {"event_indices": [1], "end_seconds": 5, "camera": "Close on the wind-up", "action": "The fist draws back."},
            {"event_indices": [2, 3, 4], "end_seconds": 10, "camera": "Follow the full flight", "action": "The punch lands, the opponent flies, and the cliff shatters."},
        ], "closing_state": "The opponent lies in the shattered cliff."}
        kwargs = dict(segment_number=1, duration=10, assigned_beats=beats,
                      dialogue_catalog=[], opening_state="The fighters face each other.", source_intent={})
        result = _canonicalize_segment_contract(candidate, **kwargs)
        self.assertEqual([item["beat_ids"] for item in result["shots"]], [["B1"], ["B2", "B3", "B4"]])
        for before, after in zip(candidate["shots"], result["shots"]):
            self.assertEqual(after["camera"], before["camera"])
            self.assertEqual(after["action"], before["action"])
        self.assertEqual(result["shots"][0]["end_seconds"], 2.5)
        candidate["shots"][1]["event_indices"] = [3, 2, 4]
        rejected = _canonicalize_segment_contract(candidate, **kwargs)
        self.assertIn("once, in order", rejected["event_assignment_error"])
        self.assertEqual(rejected["shots"], candidate["shots"])

    def test_timed_exchange_compiles_each_exact_line_once_with_its_action(self):
        source = ('[0s-6s] Alice opens the door and says, "Come in." Bob steps inside and replies, "Thank you." '
                  '[6s-12s] Alice closes the door and says, "We are safe."')

        def generate(**kwargs):
            schema = kwargs["json_schema"]["properties"]
            if "setting_continuity" in schema:
                return json.dumps({"character_appearance": {"Alice": "As supplied.", "Bob": "As supplied."},
                                   "setting_continuity": "A quiet doorway.", "visual_continuity": "Natural light.",
                                   "editing_style": "Simple coverage.", "ambient_audio": "Quiet room tone."})
            number = schema["segment"]["minimum"]
            return json.dumps({"segment": number, "title": "The doorway", "opening_state": "At the doorway.",
                               "coverage": "Medium shots", "pacing": "Natural",
                               "shots": [{"event_indices": [1], "shot": 1, "start_seconds": 0, "end_seconds": 6,
                                          "transition": "cut", "framing": "Medium two-shot", "camera": "Hold on the doorway.",
                                          "action": "Alice opens the door. Bob steps inside." if number == 1 else "Alice closes the door.",
                                          "sound_effects": "Footsteps and the door latch."}],
                               "closing_state": "They stand inside the doorway." if number == 1 else "The door is closed."})

        with patch("services.llm_service.generate", side_effect=generate):
            result = plan_h3_sliding_windows(source, model_type="minimax_h3_fused_turbo", resolution="1280x704",
                                            total_frames=288, window_frames=144, overlap_frames=0, fps=24,
                                            planning_style="faithful")
        self.assertEqual(result["planning_warnings"], [])
        self.assertIn("Bob steps inside", result["window_prompts"][0])
        self.assertIn("Alice closes the door", result["window_prompts"][1])
        for index, lines in enumerate([["Come in.", "Thank you."], ["We are safe."]]):
            for line in lines:
                self.assertEqual(result["window_prompts"][index].count(line), 1)

    def test_soft_prompt_target_never_cuts_unique_actions_or_sounds(self):
        actions = " ".join(f"At marker {i} a courier opens gate {i} and releases bird {i}." for i in range(80))
        sound = "The final gate slams, followed by a bell and footsteps fading into the distance."
        prompt = f"integrated_multimodal_description: [Shot 1] {actions}\n\noverall_soundscape: {sound}\n\nnon_diegetic_music: N/A"
        result = fit_h3_base_prompt(prompt, target_tokens=128)
        self.assertIn(actions, result.prompt)
        self.assertIn(sound, result.prompt)
        self.assertGreater(result.token_count, 128)


if __name__ == "__main__":
    unittest.main()
