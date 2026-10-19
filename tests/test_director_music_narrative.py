"""Music Director source-action contracts keep requested story beats visible."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.music_narrative import (
    build_music_narrative_contract,
    format_music_narrative_context,
    music_narrative_row_issue,
    validate_music_narrative_allocation,
)


def contract_for(source: str) -> dict:
    return build_music_narrative_contract(source)


class MusicNarrativeTests(unittest.TestCase):
    def test_extracts_ordered_performed_actions_and_recovers_creation_prefix(self):
        source = (
            "A cute warrior kitten is holding a sword and has a cape. "
            "Music video has the kitten singing as it teams up with a scary dragon. "
            "The kitten climbs on the dragon and sings as they fly through the air. "
            "Also include high speed POV shots riding the dragon as it swoops down "
            "and breaths fire over a battlefield of enemies. Cinematic, epic realism."
        )
        contract = contract_for(source)
        events = contract["events"]
        texts = [event["text"] for event in events]
        kinds = [event["kind"] for event in events]

        self.assertEqual(contract["source"], source)
        self.assertEqual([event["event_id"] for event in events], [f"M{i}" for i in range(1, len(events) + 1)])
        self.assertNotIn("teams up with a scary dragon", texts)
        self.assertIn("climbs on the dragon", texts)
        self.assertIn("fly through the air", texts)
        self.assertIn("swoops down", texts)
        self.assertIn("breaths fire over a battlefield of enemies", texts)
        self.assertIn("boarding", kinds)
        self.assertIn("flight", kinds)
        self.assertIn("attack", kinds)
        self.assertTrue(any("kitten" in alias for event in events for alias in event.get("actor_aliases", [])))
        self.assertFalse(any("sing" in text.casefold() or "holding" in text.casefold() for text in texts))

    def test_empty_for_static_music_performance_and_negated_action(self):
        for source in (
            "A kitten is riding a dragon while singing and dancing in the wind.",
            "The passenger is already aboard a boat and sings to camera.",
            "The kitten never climbs on the dragon and does not attack.",
            'A singer says, "The kitten climbs on the dragon and breathes fire."',
            "The singer performs in an open grassy field with closed lips and raised arms beneath a burning sunset. Landing lights glow behind the camera.",
        ):
            with self.subTest(source=source):
                self.assertEqual(contract_for(source)["events"], [])

    def test_stage_performance_description_has_no_static_action_milestones(self):
        source = (
            "A cinematic soul music video. Ari, an adult singer in a blue jacket, "
            "performs on a warehouse stage, holding a microphone in the right hand. "
            "Begin with intimate facial close-ups, develop into side-tracking "
            "performance shots and wide silhouettes against warm stage lights, then "
            "end on a quiet close-up. Include natural head tilts, glances, small arm "
            "gestures and changes of stance. Preserve the singer, wardrobe and "
            "microphone throughout while varying camera distance, framing, energy "
            "and lighting with the song."
        )
        self.assertEqual(contract_for(source)["events"], [])

    def test_sailor_description_extracts_only_climb_and_sailing_actions(self):
        source = (
            "A cinematic coastal folk music video. Ari, an adult sailor in a navy "
            "coat, sings on a harbor pier at sunrise. Establish the singer with "
            "solo performance shots. Ari then climbs onto a small wooden sailboat, "
            "settles securely aboard, and the boat sails across the bay. Show the "
            "boarding action before sailing. Develop varied close-ups, shore-wide "
            "views, water-level tracking and onboard POV, finishing at golden-hour "
            "open water. Preserve identity, wardrobe, clear physical support and "
            "natural story progression without replaying the boarding."
        )
        events = contract_for(source)["events"]
        self.assertEqual([event["text"] for event in events], [
            "climbs onto a small wooden sailboat",
            "sails across the bay",
        ])
        self.assertEqual([event["kind"] for event in events], ["boarding", "action"])
        self.assertIn("ari", events[0]["actor_aliases"])
        self.assertIn("sailboat", events[0]["target_aliases"])
        self.assertIn("boat", events[1]["actor_aliases"])

    def test_static_or_directorial_wording_does_not_hide_real_actions(self):
        cases = (
            ("Ari hands the microphone to Jo.", "hands the microphone to Jo", "action", "ari"),
            ("The sailor opens a door.", "opens a door", "action", "sailor"),
            ("Show the sailor boarding the boat.", "boarding the boat", "boarding", "sailor"),
            ("The ship sails across the bay.", "sails across the bay", "action", "ship"),
            ("A soldier fires a gun at the target.", "fires a gun at the target", "attack", "soldier"),
            ("The sailor sings before boarding the boat.", "boarding the boat", "boarding", "sailor"),
            (
                "Ari performs after climbing onto the dragon.",
                "climbing onto the dragon",
                "boarding",
                "ari",
            ),
        )
        for source, expected_text, expected_kind, expected_actor in cases:
            with self.subTest(source=source):
                events = contract_for(source)["events"]
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["text"], expected_text)
                self.assertEqual(events[0]["kind"], expected_kind)
                self.assertIn(expected_actor, events[0]["actor_aliases"])
        boarding = contract_for("Show the sailor boarding the boat.")["events"][0]
        self.assertIn("boat", boarding["target_aliases"])

    def test_explicit_action_references_do_not_create_milestones(self):
        for source in (
            "Show the boarding action before sailing.",
            "Show boarding before sailing.",
            "Include the boarding before sailing.",
            "Depict flying after boarding.",
            "Preserve natural story progression without replaying the boarding.",
        ):
            with self.subTest(source=source):
                self.assertEqual(contract_for(source)["events"], [])

    def test_actor_free_order_reminder_does_not_add_an_event_to_the_story(self):
        source = (
            "Ari climbs onto a small wooden sailboat, settles securely aboard, "
            "and the boat sails across the bay. Show boarding before sailing."
        )
        events = contract_for(source)["events"]
        self.assertEqual([event["text"] for event in events], [
            "climbs onto a small wooden sailboat", "sails across the bay",
        ])
        self.assertEqual(events[0]["actor_aliases"], ["ari"])
        self.assertEqual(events[1]["actor_aliases"], ["boat"])

    def test_order_reminder_filter_keeps_directed_actions_with_an_actor(self):
        for source in (
            "Show the sailor boarding the boat before sailing across the bay.",
            "Show Ari boarding the boat before sailing across the bay.",
        ):
            with self.subTest(source=source):
                events = contract_for(source)["events"]
                self.assertEqual(len(events), 2)
                self.assertEqual(events[0]["kind"], "boarding")
                self.assertEqual(events[1]["text"], "sailing across the bay")

    def test_camera_directions_do_not_become_physical_attack_events(self):
        sources = (
            "One vocalist sings with a drummer and guitarist in a single rehearsal room. "
            "Keep the camera locked in a wide shot throughout. Develop the performance "
            "with gestures, interplay and the changing musical energy. No costume or "
            "location changes.",
            "The music video opens with a close-up shot, then holds a fixed camera "
            "throughout the performance.",
            "A filmmaker shoots a close-up of the vocalist while the band performs.",
            "The camera holds a wide shot while a soldier shoots a gun at the target.",
        )
        for source in sources:
            with self.subTest(source=source):
                events = contract_for(source)["events"]
                if "soldier" in source:
                    self.assertEqual([event["text"] for event in events], [
                        "shoots a gun at the target",
                    ])
                    self.assertEqual(events[0]["kind"], "attack")
                else:
                    self.assertEqual(events, [])

    def test_physical_shooting_and_throwing_remain_narrative_events(self):
        contract = contract_for(
            "A soldier shoots a gun at the target, then throws the rifle aside."
        )
        self.assertEqual([event["kind"] for event in contract["events"]], ["attack", "action"])
        self.assertEqual([event["text"] for event in contract["events"]], [
            "shoots a gun at the target", "throws the rifle aside",
        ])

    def test_walk_and_stop_are_separate_from_observational_music_video_context(self):
        contract = contract_for(
            "A singer walks along the shoreline, stops beside a lighthouse, and "
            "watches the sun disappear. Develop a quiet reflective music video "
            "with ocean views and performance. One singer, no additional characters."
        )
        events = contract["events"]
        self.assertEqual([event["text"] for event in events], [
            "walks along the shoreline",
            "stops beside a lighthouse",
        ])
        self.assertEqual([event["kind"] for event in events], ["action", "action"])
        self.assertTrue(all("singer" in event.get("actor_aliases", []) for event in events))
        self.assertFalse(any("watch" in event["text"].casefold() for event in events))

    def test_same_verb_paraphrases_keep_actor_place_and_negation_grounded(self):
        contract = contract_for(
            "A singer walks along the shoreline, stops beside a lighthouse, and "
            "watches the sun disappear."
        )
        walk_allocation = [{"event_id": "M1", "clip_index": 0}]
        accepted_walks = (
            "The singer walks steadily along the shoreline.",
            "The singer walked at an unhurried pace along the wet shoreline.",
            "The singer begins a slow, measured, contemplative walk along the wet shoreline.",
            "The singer enters from the left side, walking slowly and deliberately along the wet shoreline.",
        )
        for prompt in accepted_walks:
            with self.subTest(prompt=prompt):
                self.assertIsNone(music_narrative_row_issue(
                    {"source_event_ids": ["M1"], "video_prompt": prompt},
                    contract=contract, allocation=walk_allocation, index=0,
                ))

        rejected_walks = (
            "The drummer walks steadily along the shoreline.",
            "The singer walks away from the shoreline.",
            "The singer walks steadily away from the lighthouse; the shoreline remains visible.",
            "The singer stands steadily along the shoreline.",
            "The singer does not walk along the shoreline.",
        )
        for prompt in rejected_walks:
            with self.subTest(prompt=prompt):
                issue = music_narrative_row_issue(
                    {"source_event_ids": ["M1"], "video_prompt": prompt},
                    contract=contract, allocation=walk_allocation, index=0,
                )
                self.assertIsNotNone(issue)
                self.assertIn("source verb family", issue)

        stop_allocation = [{"event_id": "M2", "clip_index": 1}]
        self.assertIsNone(music_narrative_row_issue(
            {
                "source_event_ids": ["M2"],
                "video_prompt": "The singer stopped briefly beside the lighthouse.",
            },
            contract=contract, allocation=stop_allocation, index=1,
        ))
        wrong_stop = music_narrative_row_issue(
            {
                "source_event_ids": ["M2"],
                "video_prompt": "The drummer stopped beside the lighthouse.",
            },
            contract=contract, allocation=stop_allocation, index=1,
        )
        self.assertIsNotNone(wrong_stop)

    def test_broad_traversal_can_use_established_setting_without_repeating_it(self):
        contract = contract_for(
            "A cute warrior kitten sings alone in a war-torn fantasy landscape. The kitten "
            "runs across the battlefield while singing, then climbs onto a scary dragon. "
            "The dragon flies through the air carrying the kitten; include high speed POV "
            "views from behind the kitten. The dragon swoops down and breathes fire over "
            "enemies. Wind whips the kitten's cape. Cinematic realism."
        )
        allocation = [{"event_id": "M1", "clip_index": 0}]
        captured_row = {
            "source_event_ids": ["M1"],
            "subjects_on_screen": [{
                "character_id": "warrior_kitten",
                "speaker_name": "",
                "visual_description": "a small feline warrior wearing a crimson cape",
                "performance_role": "vocalist",
            }],
            "video_prompt": (
                "[Shot 1] The dusty, war-torn battlefield stretches out. The warrior kitten "
                "starts sprinting from a poised, forward-leaning stance, running directly "
                "towards the camera's path while singing with intense focus. Its crimson "
                "cape streams dynamically behind it, whipping dramatically in the strong "
                "wind. The camera executes a low-angle tracking shot, moving at high speed "
                "parallel to the kitten, emphasizing the raw energy of the run. [Shot 2] At "
                "00:03.000, the kitten pumps its legs faster, leaning deeper into the sprint, "
                "its mouth wide open, perfectly lip-syncing to the driving orchestral melody. "
                "The dust kicked up by its paws creates a visible, low cloud effect in the "
                "immediate foreground. [Shot 3] The camera slightly pulls back (zooming out) "
                "as the kitten continues running, allowing the environment\uFFFDscorched earth, "
                "fractured stone\uFFFDto blur past in the background. The sunlight hits the "
                "edges of the kitten's fur, creating a bright, kinetic rim-lighting effect. "
                "[Shot 4] At 00:08.000, the kitten maintains peak velocity, running directly "
                "toward the lens with its gaze locked onto the viewer, cape fully extended, "
                "mid-song delivery. The soundscape is dominated by the rush of air against "
                "the ground and the high, energetic swell of the non-diegetic music. The "
                "kitten remains silent with mouths closed only for the final beat transition "
                "before the next action begins.\n\n"
                "overall_soundscape: Intense rushing wind noise (whoosh), heavy thud of tiny "
                "paws impacting dust, and the bright, driving melody of the main instrumentation.\n"
                "non_diegetic_music: Fast-paced, driving epic orchestral track, strong percussion "
                "hitting every two beats."
            ),
        }
        self.assertIsNone(music_narrative_row_issue(
            captured_row, contract=contract, allocation=allocation, index=0,
        ))

        rejected_prompts = (
            "The war-torn battlefield stretches out behind the singer. "
            "The kitten runs across the beach.",
            "The battlefield is not visible. The kitten runs toward the camera.",
            "The battlefield stretches out. The kitten stays still in the battlefield.",
            "The battlefield stretches out. The drummer runs toward the camera.",
            "The battlefield stretches out. The kitten runs away from the battlefield "
            "toward the lens.",
        )
        for prompt in rejected_prompts:
            with self.subTest(prompt=prompt):
                issue = music_narrative_row_issue(
                    {"source_event_ids": ["M1"], "video_prompt": prompt},
                    contract=contract, allocation=allocation, index=0,
                )
                self.assertIsNotNone(issue)

        # Once the requested traversal is explicitly present, continuing
        # along another route is not itself a missing source action.
        self.assertIsNone(music_narrative_row_issue(
            {"source_event_ids": ["M1"], "video_prompt":
             "The kitten runs across the battlefield, then follows the nearby beach."},
            contract=contract, allocation=allocation, index=0,
        ))

    def test_unique_declared_vocalist_id_bridges_source_singer_role(self):
        contract = contract_for(
            "A singer walks along the shoreline, stops beside a lighthouse, and "
            "watches the sun disappear. One singer, no additional characters."
        )
        allocation = [{"event_id": "M2", "clip_index": 1}]
        row = {
            "source_event_ids": ["M2"],
            "subjects_on_screen": [{
                "character_id": "lead_performer",
                "speaker_name": "",
                "visual_description": "A man in a sand-colored coat.",
                "performance_role": "vocalist",
            }],
            "video_prompt": (
                "The lead_performer stops abruptly beside the base of the tall, "
                "sturdy lighthouse, then turns toward the sunset."
            ),
        }
        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=1,
        ))

        for prompt, subjects in (
            (
                "The drummer stops beside the lighthouse.",
                row["subjects_on_screen"],
            ),
            (
                "The lead_performer stops away from the lighthouse.",
                row["subjects_on_screen"],
            ),
            (
                "The lead_performer does not stop beside the lighthouse.",
                row["subjects_on_screen"],
            ),
            (
                "The lead_performer stops beside the lighthouse.",
                [
                    *row["subjects_on_screen"],
                    {
                        "character_id": "backing_vocalist",
                        "performance_role": "vocalist",
                    },
                ],
            ),
            (
                "The lead_performer stops beside the lighthouse.",
                [{
                    "character_id": "lead_performer",
                    "performance_role": "non_vocal",
                }],
            ),
        ):
            with self.subTest(prompt=prompt, subjects=subjects):
                issue = music_narrative_row_issue(
                    {**row, "video_prompt": prompt, "subjects_on_screen": subjects},
                    contract=contract, allocation=allocation, index=1,
                )
                self.assertIsNotNone(issue)
                self.assertIn("source verb family", issue)

    def test_same_verb_object_paraphrase_requires_the_source_object(self):
        contract = contract_for("A singer opens the wooden gate.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        self.assertIsNone(music_narrative_row_issue(
            {
                "source_event_ids": ["M1"],
                "video_prompt": "The singer carefully opened the gate.",
            },
            contract=contract, allocation=allocation, index=0,
        ))
        issue = music_narrative_row_issue(
            {
                "source_event_ids": ["M1"],
                "video_prompt": "The singer carefully opened the window.",
            },
            contract=contract, allocation=allocation, index=0,
        )
        self.assertIsNotNone(issue)

    def test_nonfantasy_boarding_and_sailing_are_narrative_actions(self):
        contract = contract_for("A passenger boards a boat, then sails across the bay.")
        self.assertEqual([event["kind"] for event in contract["events"]], ["boarding", "action"])
        self.assertEqual([event["text"] for event in contract["events"]], [
            "boards a boat", "sails across the bay",
        ])
        self.assertIn("passenger", contract["events"][0]["actor_aliases"])
        self.assertIn("boat", contract["events"][0]["target_aliases"])

    def test_abstract_social_beats_do_not_become_literal_action_requirements(self):
        contract = contract_for(
            "The kitten teams up with the dragon, helps the villagers, and rescues the town."
        )
        self.assertEqual(contract["events"], [])

    def test_allocation_requires_every_event_once_in_source_order_and_valid_indices(self):
        contract = contract_for("A passenger boards a boat, then sails across the bay.")
        valid = [
            {"event_id": "M1", "clip_index": 2},
            {"event_id": "M2", "clip_index": 2},
        ]
        self.assertEqual(validate_music_narrative_allocation(contract, valid, 23), [])
        self.assertTrue(validate_music_narrative_allocation(contract, [valid[1], valid[0]], 23))
        self.assertTrue(validate_music_narrative_allocation(contract, [valid[0]], 23))
        self.assertTrue(validate_music_narrative_allocation(contract, [
            {"event_id": "M1", "clip_index": True}, valid[1],
        ], 23))
        self.assertTrue(validate_music_narrative_allocation(contract, [
            {"event_id": "M1", "clip_index": 3}, {"event_id": "M2", "clip_index": 2},
        ], 23))

    def test_pending_boarding_cannot_be_skipped_by_an_independent_cut(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 18}]
        row = {
            "source_event_ids": [],
            "video_prompt": "Independent cut: the kitten is perched on the dragon, singing to camera.",
            "subjects_on_screen": [],
        }
        issue = music_narrative_row_issue(row, contract=contract, allocation=allocation, index=17)
        self.assertIsNotNone(issue)
        self.assertIn("pending boarding", issue)

    def test_boarding_clip_cannot_open_mounted_then_append_the_action(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 4}]
        row = {
            "source_event_ids": ["M1"],
            "video_prompt": (
                "<Subject 1> (the Kitten) is already perched on the dragon; "
                "then the kitten climbs on the dragon."
            ),
            "subjects_on_screen": [],
        }
        issue = music_narrative_row_issue(row, contract=contract, allocation=allocation, index=4)
        self.assertIsNotNone(issue)
        self.assertIn("opens with", issue)

    def test_valid_climb_then_riding_and_later_hold(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 4}]
        climb = {
            "source_event_ids": ["M1"],
            "video_prompt": "The kitten climbs on the dragon, then rides the dragon through the sky.",
            "subjects_on_screen": [],
        }
        self.assertIsNone(music_narrative_row_issue(
            climb, contract=contract, allocation=allocation, index=4,
        ))
        later = {
            "source_event_ids": [],
            "video_prompt": "A later independent shot holds on the kitten riding the dragon.",
            "subjects_on_screen": [],
        }
        self.assertIsNone(music_narrative_row_issue(
            later, contract=contract, allocation=allocation, index=5,
        ))

        grounded_opening = {
            "source_event_ids": ["M1"],
            "video_prompt": (
                "The kitten stands on the ground beside the dragon. "
                "The kitten climbs on the dragon. The kitten is then seated securely "
                "on the dragon, paws gripping its neck scales."
            ),
        }
        self.assertIsNone(music_narrative_row_issue(
            grounded_opening, contract=contract, allocation=allocation, index=4,
        ))

    def test_mounted_guard_requires_the_requested_target_relation(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 3}]
        row = {
            "source_event_ids": [],
            "video_prompt": (
                "The kitten stands on the ground beside the dragon, then is seated "
                "securely on the dragon before the boarding clip."
            ),
        }
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=1,
        )
        self.assertIsNotNone(issue)
        self.assertIn("pending boarding", issue)

    def test_wrong_actor_target_and_unrelated_or_negated_relations_are_scoped(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        wrong_actor = {
            "source_event_ids": ["M1"],
            "video_prompt": "A puppy climbs on the dragon.",
            "subjects_on_screen": [],
        }
        self.assertIn("requested actor", music_narrative_row_issue(
            wrong_actor, contract=contract, allocation=allocation, index=0,
        ))
        wrong_actor_after_watch = {
            "source_event_ids": ["M1"],
            "video_prompt": "The kitten watches as the puppy climbs on the dragon.",
        }
        self.assertIn("requested actor", music_narrative_row_issue(
            wrong_actor_after_watch, contract=contract, allocation=allocation, index=0,
        ))
        wrong_target = {
            "source_event_ids": ["M1"],
            "video_prompt": "The kitten climbs onto a horse.",
            "subjects_on_screen": [],
        }
        self.assertIn("source verb family", music_narrative_row_issue(
            wrong_target, contract=contract, allocation=allocation, index=0,
        ))

        pending = [{"event_id": "M1", "clip_index": 3}]
        unrelated = {
            "source_event_ids": [],
            "video_prompt": "A puppy is perched on the dragon while the kitten sings.",
            "subjects_on_screen": [],
        }
        self.assertIsNone(music_narrative_row_issue(
            unrelated, contract=contract, allocation=pending, index=1,
        ))
        negated = {
            "source_event_ids": [],
            "video_prompt": "The kitten is not mounted on the dragon; she sings on the ground.",
            "subjects_on_screen": [],
        }
        self.assertIsNone(music_narrative_row_issue(
            negated, contract=contract, allocation=pending, index=1,
        ))

    def test_future_event_cannot_appear_before_assignment_or_out_of_order_in_clip(self):
        contract = contract_for(
            "The kitten climbs on the dragon, then the dragon breathes fire over enemies."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 1},
            {"event_id": "M2", "clip_index": 2},
        ]
        premature = {
            "source_event_ids": [],
            "video_prompt": "The kitten climbs on the dragon as an early independent cut.",
        }
        issue = music_narrative_row_issue(
            premature, contract=contract, allocation=allocation, index=0,
        )
        self.assertIsNotNone(issue)
        self.assertIn("future music narrative event M1", issue)

        together = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 0},
        ]
        reversed_actions = {
            "source_event_ids": ["M1", "M2"],
            "video_prompt": (
                "The dragon breathes fire over enemies, then the kitten climbs on the dragon."
            ),
        }
        issue = music_narrative_row_issue(
            reversed_actions, contract=contract, allocation=together, index=0,
        )
        self.assertIsNotNone(issue)
        self.assertIn("out of source order", issue)

    def test_future_event_in_ending_beat_cannot_appear_before_assignment(self):
        contract = contract_for(
            "The kitten climbs on the dragon, then the dragon soars into the sky."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 2},
        ]
        row = {
            "source_event_ids": [],
            "video_prompt": "The kitten waits beside the dragon at the ridge.",
            "ending_beat": "The dragon is soaring above the clouds.",
        }

        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=1,
        )

        self.assertIsNotNone(issue)
        self.assertIn("future music narrative event M2", issue)

    def test_ending_beat_cannot_fulfill_an_assigned_event(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        row = {
            "source_event_ids": ["M1"],
            "video_prompt": "The kitten sings to camera in a close shot.",
            "ending_beat": "The kitten climbs on the dragon.",
        }

        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        )

        self.assertIsNotNone(issue)
        self.assertIn("source verb family", issue)

    def test_unrelated_static_ending_beat_does_not_trigger_future_event(self):
        contract = contract_for(
            "The kitten climbs on the dragon, then the dragon soars into the sky."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 2},
        ]
        row = {
            "source_event_ids": [],
            "video_prompt": "The kitten watches from the ridge beside the dragon.",
            "ending_beat": "The kitten holds a steady pose in the sunrise.",
        }

        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=1,
        ))

    def test_identity_definition_cannot_bake_in_unperformed_boarding(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 2}]
        row = {
            "source_event_ids": [],
            "video_prompt": "The kitten sings in a close shot.",
            "subjects_on_screen": [{
                "character_id": "kitten",
                "visual_description": "The kitten is already mounted on the dragon.",
            }],
        }
        issue = music_narrative_row_issue(row, contract=contract, allocation=allocation, index=1)
        self.assertIsNotNone(issue)
        self.assertIn("identity definitions", issue)

    def test_context_marks_completed_assigned_and_pending_milestones(self):
        contract = contract_for("The passenger boards a boat, then sails across the bay.")
        allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 2},
        ]
        context = format_music_narrative_context(contract, allocation, 2, 3)
        self.assertIn("Completed before this batch", context)
        self.assertIn("M1", context)
        self.assertIn("Assigned within this batch", context)
        self.assertIn("Pending after this batch", context)
        self.assertIn("source_event_ids must be exactly ['M2']", context)
        self.assertIn("literal source clause", context)
        self.assertIn("Visibly perform every assigned action", context)
        self.assertIn("ongoing vocals", context)
        self.assertEqual(format_music_narrative_context({"events": []}, [], 0, 1), "")

    def test_compiled_context_ir_coverage_uses_detailed_description_only(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        compiled = (
            "subject_definitions: <Subject 1> (the kitten) is a small kitten; "
            "the dragon is a large dragon.\n\n"
            "summary: The kitten climbs on the dragon and celebrates the result.\n\n"
            "retention_analysis: Keep the music and cast consistent.\n\n"
            "detailed_description: [Shot 1] <Subject 1> (the Kitten) sings to camera "
            "in a close shot.\n\n"
            "overall_soundscape: Instrumental beat.\n\n"
            "non_diegetic_music: Ongoing upbeat vocals."
        )
        row = {"source_event_ids": ["M1"], "video_prompt": compiled}
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        )
        self.assertIsNotNone(issue)
        self.assertIn("source verb family", issue)

        compiled_with_action = compiled.replace(
            "sings to camera in a close shot.",
            "climbs on the dragon, then sings to camera in a close shot.",
        )
        row["video_prompt"] = compiled_with_action
        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        ))

    def test_legacy_visual_prefix_survives_audio_only_context_ir_suffix(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        row = {
            "source_event_ids": ["M1"],
            "video_prompt": (
                "The kitten climbs on the dragon, then settles into a secure riding pose.\n\n"
                "overall_soundscape: Upbeat instrumental.\n\n"
                "non_diegetic_music: Ongoing vocals."
            ),
        }
        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        ))

    def test_full_native_ir_without_detailed_description_still_cannot_claim_summary_action(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        row = {
            "source_event_ids": ["M1"],
            "video_prompt": (
                "subject_definitions: The kitten and dragon.\n\n"
                "summary: The kitten climbs on the dragon.\n\n"
                "overall_soundscape: Music."
            ),
        }
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        )
        self.assertIsNotNone(issue)
        self.assertIn("source verb family", issue)

    def test_compiled_subject_definitions_cannot_bake_in_pending_mounted_state(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 2}]
        compiled = (
            "subject_definitions: <Subject 1> (the kitten) is already mounted on the dragon.\n\n"
            "summary: A music performance.\n\n"
            "retention_analysis: Keep the cast consistent.\n\n"
            "detailed_description: <Subject 1> (the kitten) sings in a close shot.\n\n"
            "overall_soundscape: Instrumental.\n\n"
            "non_diegetic_music: Vocals continue."
        )
        row = {"source_event_ids": [], "video_prompt": compiled}
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=1,
        )
        self.assertIsNotNone(issue)
        self.assertIn("identity definitions", issue)

    def test_actual_music_brief_does_not_turn_singing_into_an_actor_alias(self):
        source = (
            "The kitten teams up with a scary dragon. The kitten climbs on the dragon "
            "and sings as they fly through the air. Also include high speed POV shots "
            "riding the dragon as it swoops down and breaths fire over a battle field of enemies."
        )
        events = contract_for(source)["events"]
        self.assertEqual(
            [event["text"] for event in events],
            [
                "climbs on the dragon", "fly through the air", "swoops down",
                "breaths fire over a battle field of enemies",
            ],
        )
        self.assertIn("kitten", events[0]["actor_aliases"])

    def test_compiled_details_accept_bounded_climb_paraphrase_with_actor_and_target(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        compiled = (
            "subject_definitions: <Subject 1> (the kitten) and the dragon.\n\n"
            "summary: The kitten climbs on the dragon and celebrates.\n\n"
            "retention_analysis: Keep the same cast.\n\n"
            "detailed_description: [Shot 1] The Kitten takes a determined stance "
            "near the Dragon’s massive foreleg. As it climbs, its small paws grip "
            "the plates, and its body twists upward. The Kitten reaches the Dragon’s "
            "shoulder joint. It pulls itself onto the Dragon’s shoulder plating, "
            "settling securely, its lower paws anchoring onto the armor.\n\n"
            "overall_soundscape: Music.\n\nnon_diegetic_music: Vocals continue."
        )
        row = {"source_event_ids": ["M1"], "video_prompt": compiled}
        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        ))

    def test_captured_boarding_clause_binds_unambiguous_pronoun_to_kitten(self):
        source = (
            "The kitten climbs onto a scary dragon."
        )
        contract = contract_for(source)
        allocation = [{"event_id": "M1", "clip_index": 0}]
        captured_prompt = (
            "The kitten scrambles up the dragon's scaled leg with surprising speed "
            "and confidence. As it reaches the top of the leg, it leaps and climbs "
            "onto a scary dragon, moving across the broad, wet scales of its back. "
            "The kitten continues forward, settling securely on the dragon's neck."
        )
        self.assertIsNone(music_narrative_row_issue(
            {"source_event_ids": ["M1"], "video_prompt": captured_prompt},
            contract=contract, allocation=allocation, index=0,
        ))

    def test_pronoun_boarding_rejects_ambiguous_wrong_actor_and_negation(self):
        contract = contract_for("The kitten climbs onto a scary dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        rejected_prompts = (
            (
                "The kitten watches the puppy. As it reaches the top of a hill, "
                "it leaps and climbs onto a scary dragon.",
                "ambiguous antecedent",
            ),
            (
                "The puppy scrambles up the dragon's scaled leg with surprising speed "
                "and confidence. As it reaches the top of the leg, it leaps and climbs "
                "onto a scary dragon.",
                "wrong antecedent",
            ),
            (
                "The kitten scrambles up the dragon's scaled leg with surprising speed "
                "and confidence. As it reaches the top of the leg, it does not leap and "
                "climb onto a scary dragon.",
                "negated action",
            ),
        )
        for prompt, reason in rejected_prompts:
            with self.subTest(reason=reason):
                self.assertIsNotNone(music_narrative_row_issue(
                    {"source_event_ids": ["M1"], "video_prompt": prompt},
                    contract=contract, allocation=allocation, index=0,
                ))

    def test_simple_pronoun_continuation_keeps_existing_behavior(self):
        for actor, prompt in (
            ("singer", "The singer raises a lantern. He opens the door."),
            ("robot", "The robot grips a handle. It opens the door."),
        ):
            with self.subTest(actor=actor):
                self.assertIsNone(music_narrative_row_issue(
                    {"source_event_ids": ["M1"], "video_prompt": prompt},
                    contract=contract_for(f"The {actor} opens the door."),
                    allocation=[{"event_id": "M1", "clip_index": 0}], index=0,
                ))

    def test_paraphrased_climb_still_needs_requested_actor_target_and_climb(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        for prompt in (
            "The puppy takes a determined stance. As it climbs, its paws grip the dragon.",
            "The kitten takes a stance. As it climbs onto a horse beside the dragon, its paws grip the saddle.",
            "The kitten takes a stance beside the dragon, then settles securely on its shoulder.",
        ):
            with self.subTest(prompt=prompt):
                issue = music_narrative_row_issue(
                    {"source_event_ids": ["M1"], "video_prompt": prompt},
                    contract=contract, allocation=allocation, index=0,
                )
                self.assertIsNotNone(issue)

    def test_climb_equivalent_requires_actor_target_contact_and_upward_self_movement(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        equivalent = {
            "source_event_ids": ["M1"],
            "video_prompt": (
                "The Kitten plants hind paws on the Dragon foreleg, uses front paws "
                "to grip the Dragon snout scales, pulling its body upward."
            ),
        }
        self.assertIsNone(music_narrative_row_issue(
            equivalent, contract=contract, allocation=allocation, index=0,
        ))

        for prompt in (
            "The kitten is already seated on the dragon's neck, singing to camera.",
            "The kitten floats upward beside the dragon without touching it.",
            "The kitten grips the dragon's foreleg while the dragon pulls the kitten upward.",
            "The puppy plants paws on the dragon, grips its scales, and pulls itself upward.",
            "The kitten plants paws on a horse, grips the saddle, and pulls itself upward.",
            "The kitten watches as the puppy plants paws on the dragon, grips its scales, and pulls itself upward.",
        ):
            with self.subTest(prompt=prompt):
                issue = music_narrative_row_issue(
                    {"source_event_ids": ["M1"], "video_prompt": prompt},
                    contract=contract, allocation=allocation, index=0,
                )
                self.assertIsNotNone(issue)


    def test_climb_paraphrase_allows_adjectives_in_target_noun_phrase(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        row = {
            "source_event_ids": ["M1"],
            "video_prompt": "The kitten climbs onto the massive dragon’s shoulder plating.",
        }
        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        ))

    def test_flight_swoop_and_breath_fire_accept_their_own_inflections(self):
        contract = contract_for(
            "The dragon flies through the air. The dragon swoops down and breaths fire over enemies."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 0},
            {"event_id": "M3", "clip_index": 0},
        ]
        row = {
            "source_event_ids": ["M1", "M2", "M3"],
            "video_prompt": (
                "The Dragon flew through the air. The Dragon is swooping down. "
                "The Dragon breathes fire over the enemies."
            ),
        }
        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        ))

        soar_contract = contract_for("The dragon flies through the air.")
        self.assertIsNone(music_narrative_row_issue(
            {
                "source_event_ids": ["M1"],
                "video_prompt": "The dragon soars through the sky.",
            },
            contract=soar_contract,
            allocation=[{"event_id": "M1", "clip_index": 0}],
            index=0,
        ))

    def test_negation_does_not_leak_from_a_previous_sentence(self):
        contract = contract_for("The kitten climbs on the dragon.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        row = {
            "source_event_ids": ["M1"],
            "video_prompt": (
                "The kitten never climbs on the dragon. The kitten climbs on the dragon."
            ),
        }
        self.assertIsNone(music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        ))


if __name__ == "__main__":
    unittest.main()
