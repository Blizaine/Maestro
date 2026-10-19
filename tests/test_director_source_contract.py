"""CPU checks for source ownership independent of screenplay compression."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.physical_continuity import (
    physical_transition_actions_missing_from_prompt,
)
from services.director.planners.short_film import _h3_rebuilt_visual_prompt
from services.director.source_contract import (
    apply_source_contract,
    build_source_contract,
    director_audio_policy,
    format_source_contract,
    source_contract_issues,
)


class DirectorSourceContractTests(unittest.TestCase):
    def test_explicit_audio_permissions_survive_writer_placeholders(self):
        policy = director_audio_policy("A silent fantasy short film. No speech or music.")
        self.assertEqual(policy, {"generated_speech_allowed": False, "generated_music_allowed": False})
        self.assertEqual(director_audio_policy("A noisy musical party."),
                         {"generated_speech_allowed": True, "generated_music_allowed": True})
        self.assertEqual(director_audio_policy('A musician says "No music." A score plays.'),
                         {"generated_speech_allowed": True, "generated_music_allowed": True})
        self.assertFalse(director_audio_policy(
            "No repeated boarding, floating, unrelated battle, speech or music."
        )["generated_music_allowed"])
        self.assertFalse(director_audio_policy(
            "Do not add dialogue or any music."
        )["generated_music_allowed"])
        self.assertTrue(director_audio_policy(
            "No hats, the music plays in the next room."
        )["generated_music_allowed"])

    def test_reference_roles_are_not_performed_events(self):
        contract = build_source_contract(
            "A courier climbs the ladder. The courier opens the hatch. "
            "Preserve this ordered progression. Picture 1 supplies only identity."
        )
        self.assertEqual(len(contract["events"]), 2)
        self.assertIn("E2", format_source_contract(contract))

    def test_silent_source_keeps_visual_action_without_dangling_speech(self):
        contract = build_source_contract('A silent short film. A courier opens the hatch and says, "We made it." No speech or background music.')
        self.assertFalse(contract["audio_policy"]["generated_music_allowed"])
        self.assertEqual(len(contract["events"]), 1)
        self.assertIn("opens the hatch", contract["events"][0]["text"])
        self.assertNotIn("says", contract["events"][0]["text"])
        self.assertIn("eats food", build_source_contract("A silent film. A courier eats food.")["events"][0]["text"])

    def test_omitted_repeated_and_reordered_actions_rejected(self):
        contract = build_source_contract("A courier opens the box. The courier closes the box.")
        self.assertEqual(source_contract_issues([
            {"source_event_ids": ["E1"]}, {"source_event_ids": ["E2"]}
        ], contract), [])
        for ids in (["E1"], ["E1", "E1", "E2"], ["E2", "E1"]):
            self.assertTrue(source_contract_issues([{"source_event_ids": ids}], contract))
        self.assertTrue(source_contract_issues([{}], contract))
        self.assertEqual(source_contract_issues([{}], contract, require_ids=False), [])

    def test_source_prop_reaches_later_clip_and_rewrite_does_not_repeat_action(self):
        contract = build_source_contract(
            "A courier climbs the ladder with a knife sheathed at the left hip. "
            "The courier opens the hatch."
        )
        rows = [{"source_event_ids": ["E1"], "action_beats": ["A wrong paraphrase."]},
                {"source_event_ids": ["E2"], "action_beats": ["The courier climbs again."]}]
        original = copy.deepcopy(contract)
        apply_source_contract(rows, contract)
        self.assertEqual(contract, original)
        self.assertEqual(rows[0]["action_beats"], [contract["events"][0]["text"]])
        prompt = _h3_rebuilt_visual_prompt(rows[1])
        self.assertIn("knife sheathed at the left hip", prompt)
        self.assertIn("opens the hatch", prompt)
        self.assertNotIn("climbs again", prompt)

    def test_canonical_opening_excludes_second_subject_pose_and_forbidden_music(self):
        prompt = _h3_rebuilt_visual_prompt({
            "spatial_setup": "The courier stands on the floor.",
            "subjects_on_screen": [{"speaker_name": "Courier", "visual_description": "Courier in blue", "position_or_relation": "already on the roof"}],
            "action_beats": ["The courier climbs the ladder."],
            "video_prompt": "A contradictory draft. non_diegetic_music: An orchestral score",
            "_director_audio_policy": {"generated_music_allowed": False},
        })
        self.assertIn("stands on the floor", prompt)
        self.assertNotIn("already on the roof", prompt)
        self.assertNotIn("orchestral", prompt)
        self.assertNotIn("contradictory draft", prompt)

    def test_empty_state_retains_opening_and_declared_transition_reaches_prompt(self):
        raw = {"spatial_setup": "The courier stands on the floor.",
               "physical_state": {"opening": [], "transitions": [], "closing": []}}
        self.assertIn("stands on the floor", _h3_rebuilt_visual_prompt(raw))
        raw["physical_state"] = {
            "opening": [{"subject_id": "Courier", "predicate": "support", "value": "standing on floor"}],
            "transitions": [{"subject_id": "Courier", "predicate": "support", "from_value": "standing on floor", "to_value": "seated on ledge", "action": "The courier climbs the ladder and sits on the ledge"}],
            "closing": [{"subject_id": "Courier", "predicate": "support", "value": "seated on ledge"}],
        }
        prompt = _h3_rebuilt_visual_prompt(raw)
        self.assertIn("climbs the ladder", prompt)
        self.assertIn("sits on the ledge", prompt)
        self.assertLess(prompt.index("climbs the ladder"), prompt.index("sits on the ledge"))
        self.assertLess(prompt.index("sits on the ledge"), prompt.index("At the final frame"))
        self.assertIn("Opening physical facts: Courier support: standing on floor", prompt)
        self.assertNotIn("The courier stands on the floor", prompt)

    def test_spoken_clip_cannot_keep_a_static_body_after_source_boarding(self):
        contract = build_source_contract("The traveler climbs aboard the carrier and settles astride its back.")
        row = {
            "source_event_ids": ["E1"],
            "dialogue_beats": [{"speaker_id": "traveler", "spoken_text": "Ready."}],
            "video_prompt": "The traveler remains standing beside the carrier. <d>[English] Ready.</d>",
            "physical_state": {
                "opening": [{"subject_id": "traveler", "predicate": "support", "value": "ground"}],
                "transitions": [{"subject_id": "traveler", "predicate": "support", "from_value": "ground", "to_value": "carrier", "action": "The traveler climbs aboard the carrier."}],
                "closing": [{"subject_id": "traveler", "predicate": "support", "value": "carrier"}],
            },
        }
        self.assertTrue(any("source motion" in issue for issue in source_contract_issues([row], contract)))
        row["video_prompt"] = "The traveler mounts the carrier, settles on its back, then says <d>[English] Ready.</d>"
        self.assertEqual(source_contract_issues([row], contract), [])

    def test_dialogue_body_terminal_grounding_cannot_override_typed_mounted_closing(self):
        contract = build_source_contract(
            "The rider mounts the dragon, then the dragon takes flight."
        )
        row = {
            "source_event_ids": [event["event_id"] for event in contract["events"]],
            "subjects_on_screen": [
                {"character_id": "rider", "speaker_name": "Rider"},
                {"character_id": "dragon", "speaker_name": "Dragon"},
            ],
            "dialogue_beats": [{"speaker_id": "rider", "spoken_text": "Stay close."}],
            "video_prompt": (
                "Rider mounts Dragon, then Dragon takes flight, "
                "<d>Stay close.</d>, then remains standing on the ground beside it."
            ),
            "physical_state": {
                "opening": [
                    {"subject_id": "rider", "predicate": "support", "value": "on ground"},
                    {"subject_id": "rider", "predicate": "grounded", "value": "grounded"},
                ],
                "transitions": [
                    {
                        "subject_id": "rider", "predicate": "support",
                        "from_value": "on ground", "to_value": "astride dragon",
                        "action": "Rider mounts Dragon.",
                    },
                    {
                        "subject_id": "rider", "predicate": "grounded",
                        "from_value": "grounded", "to_value": "airborne",
                        "action": "Dragon takes flight with Rider aboard.",
                    },
                ],
                "closing": [
                    {"subject_id": "rider", "predicate": "support", "value": "astride dragon"},
                    {"subject_id": "rider", "predicate": "grounded", "value": "airborne"},
                ],
            },
        }
        original_prompt = row["video_prompt"]

        issues = source_contract_issues([row], contract)

        self.assertTrue(any("explicit final ground support" in issue for issue in issues), issues)
        self.assertTrue(any("closing grounded='airborne'" in issue for issue in issues), issues)
        self.assertEqual(row["video_prompt"], original_prompt)

    def test_dialogue_terminal_guard_allows_initial_ground_before_mount(self):
        contract = build_source_contract("The rider mounts the dragon.")
        row = {
            "source_event_ids": ["E1"],
            "subjects_on_screen": [
                {"character_id": "rider", "speaker_name": "Rider"},
                {"character_id": "dragon", "speaker_name": "Dragon"},
            ],
            "dialogue_beats": [{"speaker_id": "rider", "spoken_text": "Stay close."}],
            "video_prompt": "Rider stands on the ground, mounts Dragon, <d>Stay close.</d>",
            "physical_state": {
                "opening": [{"subject_id": "rider", "predicate": "support", "value": "on ground"}],
                "transitions": [{
                    "subject_id": "rider", "predicate": "support",
                    "from_value": "on ground", "to_value": "astride dragon",
                    "action": "Rider mounts Dragon.",
                }],
                "closing": [{"subject_id": "rider", "predicate": "support", "value": "astride dragon"}],
            },
        }

        self.assertEqual(source_contract_issues([row], contract), [])

    def test_dialogue_terminal_guard_accepts_authored_final_dismount(self):
        contract = build_source_contract(
            "The rider mounts the dragon, then dismounts onto the ground."
        )
        row = {
            "source_event_ids": [event["event_id"] for event in contract["events"]],
            "subjects_on_screen": [
                {"character_id": "rider", "speaker_name": "Rider"},
                {"character_id": "dragon", "speaker_name": "Dragon"},
            ],
            "dialogue_beats": [{"speaker_id": "rider", "spoken_text": "Stay close."}],
            "video_prompt": (
                "Rider mounts Dragon, <d>Stay close.</d>, "
                "then dismounts and stands on the ground."
            ),
            "physical_state": {
                "opening": [{"subject_id": "rider", "predicate": "support", "value": "on ground"}],
                "transitions": [
                    {
                        "subject_id": "rider", "predicate": "support",
                        "from_value": "on ground", "to_value": "astride dragon",
                        "action": "Rider mounts Dragon.",
                    },
                    {
                        "subject_id": "rider", "predicate": "support",
                        "from_value": "astride dragon", "to_value": "on ground",
                        "action": "Rider dismounts onto the ground.",
                    },
                ],
                "closing": [{"subject_id": "rider", "predicate": "support", "value": "on ground"}],
            },
        }

        self.assertEqual(source_contract_issues([row], contract), [])

    def test_dialogue_terminal_state_is_attributed_to_named_actor(self):
        contract = build_source_contract("The rider mounts the dragon.")
        row = {
            "source_event_ids": ["E1"],
            "subjects_on_screen": [
                {"character_id": "rider", "speaker_name": "Rider"},
                {"character_id": "dragon", "speaker_name": "Dragon"},
            ],
            "dialogue_beats": [{"speaker_id": "dragon", "spoken_text": "Stay close."}],
            "video_prompt": (
                "Rider mounts Dragon, <d>Stay close.</d>, "
                "then Dragon remains standing on the ground beside Rider."
            ),
            "physical_state": {
                "opening": [{"subject_id": "rider", "predicate": "support", "value": "on ground"}],
                "transitions": [{
                    "subject_id": "rider", "predicate": "support",
                    "from_value": "on ground", "to_value": "astride dragon",
                    "action": "Rider mounts Dragon.",
                }],
                "closing": [
                    {"subject_id": "rider", "predicate": "support", "value": "astride dragon"},
                    {"subject_id": "dragon", "predicate": "grounded", "value": "grounded"},
                ],
            },
        }

        self.assertEqual(source_contract_issues([row], contract), [])

    def test_named_landing_and_dismount_bind_terminal_facts_to_each_actor(self):
        source = (
            "The Dragon lands, becomes stationary and folds its wings before the "
            "Kitten slides down its shoulder and puts both feet on the ground."
        )
        contract = {"events": [{"event_id": "E1", "text": source}]}
        row = {
            "source_event_ids": ["E1"],
            "subjects_on_screen": [
                {"character_id": "char_0", "speaker_name": "Kitten"},
                {"character_id": "char_1", "speaker_name": "Dragon"},
            ],
            "physical_state": {
                "opening": [
                    {"subject_id": "char_0", "predicate": "support", "value": "astride char_1"},
                    {"subject_id": "char_0", "predicate": "grounded", "value": "airborne"},
                    {"subject_id": "char_0", "predicate": "posture", "value": "seated"},
                    {"subject_id": "char_1", "predicate": "grounded", "value": "airborne"},
                    {"subject_id": "char_1", "predicate": "location", "value": "misty valley"},
                ],
                "transitions": [],
                "closing": [
                    {"subject_id": "char_0", "predicate": "support", "value": "astride char_1"},
                    {"subject_id": "char_0", "predicate": "grounded", "value": "grounded"},
                    {"subject_id": "char_0", "predicate": "posture", "value": "seated"},
                    {"subject_id": "char_1", "predicate": "grounded", "value": "grounded"},
                    {"subject_id": "char_1", "predicate": "location", "value": "aerial misty valley"},
                ],
            },
        }

        issues = source_contract_issues([row], contract)

        self.assertTrue(any("feet-on-ground" in issue and "closing support" in issue for issue in issues), issues)
        self.assertTrue(any(
            "feet-on-ground cue conflicts with closing support/contact or preceding dismount state" in issue
            for issue in issues
        ), issues)
        self.assertTrue(any("landing cue conflicts with closing aerial location" in issue for issue in issues), issues)

        row["physical_state"]["closing"] = [
            {"subject_id": "char_0", "predicate": "support", "value": "stone ledge"},
            {"subject_id": "char_0", "predicate": "grounded", "value": "grounded"},
            {"subject_id": "char_0", "predicate": "posture", "value": "standing"},
            {"subject_id": "char_1", "predicate": "grounded", "value": "grounded"},
            {"subject_id": "char_1", "predicate": "location", "value": "stone ledge"},
        ]
        self.assertFalse(any(
            issue.startswith("Shot 1 explicit named-actor") for issue in source_contract_issues([row], contract)
        ))

    def test_named_actor_cues_do_not_transfer_from_objects_or_competing_subjects(self):
        row = {
            "source_event_ids": ["E1"],
            "subjects_on_screen": [
                {"character_id": "ana", "speaker_name": "Ana"},
                {"character_id": "bea", "speaker_name": "Bea"},
            ],
            "physical_state": {
                "opening": [
                    {"subject_id": "ana", "predicate": "grounded", "value": "airborne"},
                    {"subject_id": "bea", "predicate": "grounded", "value": "grounded"},
                ],
                "transitions": [],
                "closing": [
                    {"subject_id": "ana", "predicate": "grounded", "value": "airborne"},
                    {"subject_id": "bea", "predicate": "grounded", "value": "grounded"},
                ],
            },
        }
        contract = {"events": [{
            "event_id": "E1", "text": "Ana watches while Bea lands on the ledge."
        }]}
        self.assertEqual(source_contract_issues([row], contract), [])

        contract["events"][0]["text"] = "Ana turns to Bea as she lands on the ledge."
        self.assertEqual(source_contract_issues([row], contract), [])

    def test_named_takeoff_without_physical_state_reports_missing_state(self):
        row = {
            "source_event_ids": ["E1"],
            "subjects_on_screen": [{"character_id": "char_1", "speaker_name": "Bird"}],
        }
        contract = {"events": [{"event_id": "E1", "text": "Bird takes flight."}]}

        issues = source_contract_issues([row], contract)

        self.assertTrue(any("explicit physical relation but no physical_state" in issue for issue in issues), issues)

    def test_terminal_actor_state_uses_last_explicit_cue_and_limits_dismount(self):
        row = {
            "source_event_ids": ["E1"],
            "subjects_on_screen": [{"character_id": "ana", "speaker_name": "Ana"}],
            "physical_state": {
                "opening": [{"subject_id": "ana", "predicate": "grounded", "value": "grounded"}],
                "transitions": [],
                "closing": [
                    {"subject_id": "ana", "predicate": "grounded", "value": "grounded"},
                    {"subject_id": "ana", "predicate": "support", "value": "stone ledge"},
                    {"subject_id": "ana", "predicate": "location", "value": "no longer airborne, stone ledge"},
                ],
            },
        }
        contract = {"events": [{
            "event_id": "E1", "text": "Ana takes flight, then lands on a stone ledge."
        }]}
        self.assertEqual(source_contract_issues([row], contract), [])

        contract["events"][0]["text"] = "Ana slides down the wall and sits on a stool."
        row["physical_state"]["closing"][1]["value"] = "stool"
        row["physical_state"]["closing"].append({
            "subject_id": "ana", "predicate": "posture", "value": "seated",
        })
        self.assertEqual(source_contract_issues([row], contract), [])

    def test_feet_on_ground_can_coexist_with_chair_support_and_seated_posture(self):
        facts = [
            {"subject_id": "maya", "predicate": "grounded", "value": "grounded"},
            {"subject_id": "maya", "predicate": "contact", "value": "feet on ground"},
            {"subject_id": "maya", "predicate": "support", "value": "chair"},
            {"subject_id": "maya", "predicate": "posture", "value": "seated"},
        ]
        row = {
            "source_event_ids": ["E1"],
            "subjects_on_screen": [{"character_id": "maya", "speaker_name": "Maya"}],
            "physical_state": {"opening": facts, "transitions": [], "closing": facts},
        }
        contract = {"events": [{
            "event_id": "E1",
            "text": "Maya sits in a chair and puts both feet on the ground.",
        }]}

        self.assertEqual(source_contract_issues([row], contract), [])

    def test_full_ledger_transition_lowering_uses_single_assigned_event(self):
        contract = {"events": [{
            "event_id": "E1", "text": "The Dragon lowers its shoulder."
        }]}
        row = {
            "source_event_ids": ["E1"],
            "subjects_on_screen": [
                {"character_id": "char_0", "speaker_name": "Kitten"},
                {"character_id": "char_1", "speaker_name": "Dragon"},
            ],
            "physical_state": {
                "opening": [{
                    "subject_id": "char_1", "predicate": "posture", "value": "upright shoulders",
                }],
                "transitions": [{
                    "subject_id": "char_1", "predicate": "posture",
                    "from_value": "upright shoulders", "to_value": "lowered shoulders",
                    "action": "Dragon lowers its shoulder.",
                }],
                "closing": [{
                    "subject_id": "char_1", "predicate": "posture", "value": "lowered shoulders",
                }],
            },
        }
        self.assertEqual(source_contract_issues([row], contract), [])

        row["physical_state"]["transitions"][0]["subject_id"] = "char_0"
        row["physical_state"]["opening"][0]["subject_id"] = "char_0"
        row["physical_state"]["closing"][0]["subject_id"] = "char_0"
        issues = source_contract_issues([row], contract)
        self.assertTrue(any(
            "transition 1 lowering posture/action is not supported by exact assigned source event 'E1'" in issue
            for issue in issues
        ), issues)

        row["source_event_ids"] = ["E1", "E2"]
        contract["events"].append({"event_id": "E2", "text": "The Dragon stays still."})
        row["physical_state"]["transitions"][0]["subject_id"] = "char_1"
        row["physical_state"]["opening"][0]["subject_id"] = "char_1"
        row["physical_state"]["closing"][0]["subject_id"] = "char_1"
        self.assertEqual(source_contract_issues([row], contract), [])

    def test_spoken_clip_preserves_explicit_prop_operation_and_object(self):
        contract = build_source_contract(
            'The courier tucks the brass key into her coat pocket, then says, "It is safe."'
        )
        row = {
            "source_event_ids": [event["event_id"] for event in contract["events"]],
            "dialogue_beats": [{"speaker_id": "courier", "spoken_text": "It is safe."}],
            "video_prompt": "The courier looks at the door, then says <d>[English] It is safe.</d>",
        }
        issues = source_contract_issues([row], contract)
        self.assertTrue(any("source prop operation (stow: key)" in issue for issue in issues))

        row["video_prompt"] = (
            "The courier tucks the brass key into her coat pocket before saying "
            "<d>[English] It is safe.</d>"
        )
        self.assertEqual(source_contract_issues([row], contract), [])

    def test_spoken_prop_guard_ignores_unrelated_actions_and_dialogue_quotes(self):
        unrelated_contract = build_source_contract(
            'The courier steadies her breath, then says, "Ready."'
        )
        unrelated_row = {
            "source_event_ids": [event["event_id"] for event in unrelated_contract["events"]],
            "dialogue_beats": [{"speaker_id": "courier", "spoken_text": "Ready."}],
            "video_prompt": "The courier glances toward the gate, then says <d>[English] Ready.</d>",
        }
        self.assertEqual(source_contract_issues([unrelated_row], unrelated_contract), [])

        quote_contract = build_source_contract(
            'A courier says, "Tuck the brass key into your coat pocket."'
        )
        quote_row = {
            "source_event_ids": ["E1"],
            "dialogue_beats": [{
                "speaker_id": "courier",
                "spoken_text": "Tuck the brass key into your coat pocket.",
            }],
            "video_prompt": (
                "The courier speaks without handling a key. "
                "<d>[English] Tuck the brass key into your coat pocket.</d>"
            ),
        }
        self.assertFalse(any(
            "source prop operation" in issue
            for issue in source_contract_issues([quote_row], quote_contract)
        ))

    def test_prop_operation_families_accept_clear_synonyms(self):
        cases = (
            ("The courier pockets the brass key.", "The courier tucks the brass key into her coat pocket."),
            ("The courier sheaths the knife.", "The courier holsters the knife."),
            ("The courier draws the knife from its sheath.", "The courier unsheathes the knife."),
            ("The courier gives the key to Mara.", "The courier hands Mara the key."),
            ("The courier passes the key to Mara.", "The courier hands Mara the key."),
            ("The courier opens the hatch.", "The courier unlocks the hatch."),
            ("The courier closes the hatch.", "The courier shuts the hatch."),
        )
        for source_action, retained_action in cases:
            with self.subTest(source_action=source_action):
                contract = {
                    "events": [{"event_id": "E1", "text": source_action}],
                    "audio_policy": {
                        "generated_speech_allowed": True,
                        "generated_music_allowed": True,
                    },
                }
                row = {
                    "source_event_ids": ["E1"],
                    "dialogue_beats": [{"spoken_text": "Ready."}],
                    "video_prompt": f"{retained_action} Then the courier says <d>Ready.</d>",
                }
                self.assertEqual(source_contract_issues([row], contract), [])

    def test_passive_sheathed_prop_description_satisfies_carried_state(self):
        contract = {
            "events": [{
                "event_id": "E1",
                "text": "The courier enters with a sword sheathed at the left hip.",
            }],
            "audio_policy": {
                "generated_speech_allowed": True,
                "generated_music_allowed": True,
            },
        }
        row = {
            "source_event_ids": ["E1"],
            "dialogue_beats": [{"spoken_text": "Ready."}],
            "video_prompt": (
                "The courier has a sword sheathed at the left hip, then says "
                "<d>Ready.</d>"
            ),
        }
        self.assertEqual(source_contract_issues([row], contract), [])

    def test_common_speech_and_body_actions_are_not_prop_requirements(self):
        actions = (
            "The courier takes a moment to breathe.",
            "The courier offers advice.",
            "The courier passes the time.",
            "The courier passes the gate and keeps walking.",
            "The courier offers an apology.",
            "The courier draws a deep breath.",
            "The courier opens her mouth.",
            "The courier closes the distance.",
            "The courier passes the guard.",
            "Her hands clasp around the handle.",
            "The courtyard gate stands open.",
            "The courier faces a closed gate.",
            "A scene begins with the opening shot.",
        )
        for action in actions:
            with self.subTest(action=action):
                contract = {
                    "events": [{"event_id": "E1", "text": action}],
                    "audio_policy": {
                        "generated_speech_allowed": True,
                        "generated_music_allowed": True,
                    },
                }
                row = {
                    "source_event_ids": ["E1"],
                    "dialogue_beats": [{"spoken_text": "Ready."}],
                    "video_prompt": "The courier says <d>Ready.</d>",
                }
                self.assertFalse(any(
                    "source prop operation" in issue
                    for issue in source_contract_issues([row], contract)
                ))

    def test_source_owned_silent_ending_uses_assigned_action_and_typed_closing(self):
        contract = build_source_contract(
            "A courier steps to the iron gate. The iron gate swings open behind her."
        )
        row = {
            "source_event_ids": ["E2"],
            "action_beats": ["A future scene begins at the tower."],
            "ending_beat": "The courier reaches the tower in a later shot.",
            "closing_blocking": "The tower fills the frame.",
            "physical_state": {
                "opening": [], "transitions": [],
                "closing": [{
                    "subject_id": "courier", "predicate": "support",
                    "value": "standing beside the open iron gate",
                }],
            },
        }
        apply_source_contract([row], contract)
        self.assertEqual(row["action_beats"], [contract["events"][1]["text"]])
        self.assertIn("gate swings open", row["action_beats"][0])
        self.assertEqual(row["ending_beat"], "")
        self.assertIn("standing beside the open iron gate", row["closing_blocking"])
        self.assertNotIn("tower", row["closing_blocking"])

        row_without_state = {
            "source_event_ids": ["E2"],
            "ending_beat": "An invented distant ending.",
            "closing_blocking": "An invented distant pose.",
        }
        apply_source_contract([row_without_state], contract)
        self.assertEqual(row_without_state["ending_beat"], "")
        self.assertEqual(row_without_state["closing_blocking"], "")

    def test_spoken_source_row_keeps_its_authored_ending_timing(self):
        contract = build_source_contract("The courier tucks the key away and says, \"Ready.\"")
        row = {
            "source_event_ids": ["E1"],
            "dialogue_beats": [{"spoken_text": "Ready."}],
            "ending_beat": "The courier lowers her eyes after speaking.",
            "closing_blocking": "The courier faces the door.",
        }
        original_ending = row["ending_beat"]
        original_closing = row["closing_blocking"]
        apply_source_contract([row], contract)
        self.assertEqual(row["ending_beat"], original_ending)
        self.assertEqual(row["closing_blocking"], original_closing)

    def test_typed_final_pose_does_not_erase_authored_ending(self):
        prompt = _h3_rebuilt_visual_prompt({
            "action_beats": ["The courier puts the parcel down."],
            "ending_beat": "The castle gate opens behind the courier.",
            "physical_state": {
                "opening": [{"subject_id": "courier", "predicate": "grounded", "value": "grounded"}],
                "transitions": [],
                "closing": [{"subject_id": "courier", "predicate": "grounded", "value": "grounded"}],
            },
        })
        self.assertIn("castle gate opens", prompt)
        self.assertLess(prompt.index("puts the parcel down"), prompt.index("castle gate opens"))

    def test_source_owned_actions_do_not_append_invented_state_motion(self):
        contract = build_source_contract("A courier enters the room. No dialogue or music.")
        state = {
            "opening": [{"subject_id": "courier", "predicate": "prop", "value": "not carrying brass key"}],
            "transitions": [{
                "subject_id": "courier", "predicate": "prop",
                "from_value": "not carrying brass key", "to_value": "holding brass key",
                "action": "The courier pockets a brass key.",
            }],
            "closing": [{"subject_id": "courier", "predicate": "prop", "value": "holding brass key"}],
        }
        row = {"source_event_ids": ["E1"], "physical_state": state}
        apply_source_contract([row], contract)
        self.assertEqual(row["action_beats"], [contract["events"][0]["text"]])
        self.assertNotIn("pockets", _h3_rebuilt_visual_prompt(row))
        # Connecting and legacy shots still use their authored transition.
        for ids in ([], None):
            connecting = {"physical_state": copy.deepcopy(state), "action_beats": []}
            if ids is not None:
                connecting["source_event_ids"] = ids
                apply_source_contract([connecting], contract)
            self.assertIn("pockets a brass key", _h3_rebuilt_visual_prompt(connecting))

    def test_single_word_transition_already_in_actions_is_not_repeated(self):
        state = {
            "opening": [{"subject_id": "courier", "predicate": "posture", "value": "still"}],
            "transitions": [{"subject_id": "courier", "predicate": "posture", "from_value": "still", "to_value": "jumping", "action": "Jumps"}],
            "closing": [{"subject_id": "courier", "predicate": "posture", "value": "jumping"}],
        }
        self.assertEqual(physical_transition_actions_missing_from_prompt(state, "Jumps"), [])


if __name__ == "__main__":
    unittest.main()
