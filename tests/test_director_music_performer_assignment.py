"""Explicit visual performers survive narrative cuts and native compilation."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.h3_dialogue import compile_h3_official_prompt
from services.director.music_performance import music_performance_direction
from services.director.planners.music_video import MusicVideoPlanner
from services.director.schema import ShotPlan


class MusicPerformerAssignmentTests(unittest.TestCase):
    def test_explicit_singer_role_survives_conversion_and_serialization(self):
        singer = {
            "speaker_name": "Mara", "character_id": "lead_performer",
            "visual_description": "Mara in a blue coat", "performance_role": "non_vocal",
        }
        observer = {
            "speaker_name": "Cal", "character_id": "observer",
            "visual_description": "Cal in a green coat", "performance_role": "non_vocal",
        }
        context = "Mara sings as she walks along the harbor. Cal watches."
        old_cue = music_performance_direction([singer], "active")
        rows = [{
            "scene_type": "narrative",
            "video_prompt": f"Mara walks along the harbor. Cal watches. {old_cue}",
            "subjects_on_screen": [{**singer, "character_id": f"singer_{i}"}, observer],
            "camera_plan": {"framing": "medium-wide", "movement": "tracking"},
        } for i in range(3)]
        clips = [{"start": i * 7, "end": (i + 1) * 7} for i in range(3)]
        shots = MusicVideoPlanner()._convert_to_shots(
            rows, clips, [], False, {}, None, {},
            vocal_activity=["active", "silent", "unknown"], project_context=context,
            ignore_physical_state_ledger=True,
        )

        for i, shot in enumerate(shots):
            restored = ShotPlan.from_dict(json.loads(json.dumps(shot.to_dict())))
            self.assertEqual(
                [s.performance_role for s in restored.subjects_on_screen],
                ["vocalist", "non_vocal"],
            )
            self.assertEqual(restored.scene_type, "narrative")
            self.assertEqual(restored.camera_plan.movement, "tracking")
            self.assertEqual(rows[i]["subjects_on_screen"][0]["performance_role"], "non_vocal")
            for mode in ("fl2va", "ref2va"):
                with self.subTest(activity=restored.audio_plan.vocal_activity, mode=mode):
                    kwargs = dict(
                        mode=mode, project_context=context,
                        audio_plan=restored.audio_plan.to_dict(),
                    )
                    native, _ = compile_h3_official_prompt(
                        restored.video_prompt, restored.subjects_on_screen, [], **kwargs,
                    )
                    native, _ = compile_h3_official_prompt(
                        native, restored.subjects_on_screen, [], **kwargs,
                    )
                    self.assertIn("walks along the harbor", native)
                    self.assertNotIn("Mara remains a non-singing presence", native)
                    if i == 0:
                        self.assertEqual(native.count("lip-syncs only their own audible vocal part"), 1)
                    else:
                        self.assertNotIn("lip-syncs only their own audible vocal part", native)
                        self.assertIn("keeps relaxed closed lips through this interval", native)
                    self.assertEqual(native.count("remains a non-singing presence"), 1)


if __name__ == "__main__":
    unittest.main()
