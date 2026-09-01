"""Long-form Director planning regressions."""

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

from services.director.planners.short_film import ShortFilmPlanner  # noqa: E402
from services.director.schema import AudioPlan, CameraPlan, ShotPlan  # noqa: E402


def _shot() -> ShotPlan:
    return ShotPlan(
        shot_id="local",
        index=0,
        duration_sec=10.0,
        skill_type="short_film",
        scene_goal="Advance the chapter",
        subjects_on_screen=[],
        spatial_setup="Readable composition",
        environment="Established location",
        visual_style="Cinematic",
        lighting="Natural",
        mood="Focused",
        action_beats=["A new event occurs"],
        camera_plan=CameraPlan(framing="medium shot"),
        audio_plan=AudioPlan(mode="ambient_only"),
        ending_beat="A visible handoff",
    )


class LongFormDirectorPlanningTests(unittest.TestCase):
    def test_story_over_five_minutes_is_planned_in_bounded_chapters(self):
        outline = [
            {
                "chapter": index + 1,
                "title": f"Chapter {index + 1}",
                "objective": f"Advance beat {index + 1}",
                "opening_state": f"Opening {index + 1}",
                "closing_state": f"Closing {index + 1}",
                "causal_handoff": "The prior consequence carries forward",
                "persistent_state": "Identity and story state remain stable",
            }
            for index in range(3)
        ]
        planner = ShortFilmPlanner(llm_generate=lambda **_: json.dumps(outline))
        chapter_durations: list[int] = []

        def plan_chapter(**kwargs):
            chapter_durations.append(kwargs["target_duration"])
            return [_shot()], f"Local title {len(chapter_durations)}"

        with patch.object(planner, "_plan_story_driven", side_effect=plan_chapter):
            result = planner.plan(
                story_description="One long causal adventure",
                target_duration=601,
                video_model="ltx2",
            )

        self.assertEqual(chapter_durations, [201, 200, 200])
        self.assertEqual(len(result.shots), 3)
        self.assertEqual([shot.index for shot in result.shots], [0, 1, 2])
        self.assertEqual(
            [shot.metadata["long_form_chapter"] for shot in result.shots],
            [1, 2, 3],
        )
        self.assertEqual(result.total_duration_sec, 30.0)


if __name__ == "__main__":
    unittest.main()
