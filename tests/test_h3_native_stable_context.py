import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services import h3_sequence_planner, h3_window_planner  # noqa: E402


_STALE_ACTION = (
    "LEGACY-CONTEXT-77: Nora opens the archive door and carries a red spool "
    "into the studio."
)
_SOURCE_PROMPT = "A person waits in a plain studio and takes one step."


def _ledger(*, with_stable_context: bool) -> dict:
    ledger = {
        "subject_continuity": _STALE_ACTION,
        "setting_continuity": _STALE_ACTION,
        "visual_continuity": _STALE_ACTION,
        "editing_style": _STALE_ACTION,
        "initial_state": "A person stands in the middle of a plain studio.",
        "ambient_audio": "Quiet studio room tone",
        "music": "N/A",
        "sequence_shape": "resolved",
        "motion_mechanics": "The requested flight remains unaided with no exhaust.",
        "source_relationships": [{"source_event_id": "E1", "related_event_id": "E2"}],
    }
    if with_stable_context:
        ledger["stable_camera_context"] = {
            "subject_continuity": "",
            "setting_continuity": "",
            "visual_continuity": ledger["motion_mechanics"],
            "editing_style": "",
        }
    return ledger


def _segments(durations: list[float]) -> list[dict]:
    return [
        {
            "title": f"Moment {index + 1}",
            "summary": "A person takes one step and pauses.",
            "opening_state": "A person stands in the studio.",
            "closing_state": "A person pauses after one step.",
            "coverage": "one continuous camera move",
            "pacing": "natural real-time pacing",
            "shots": [{
                "shot": 1,
                "start_seconds": 0.0,
                "end_seconds": duration,
                "transition": "opening composition",
                "framing": "medium-wide studio view",
                "camera": "the camera follows the person",
                "action": "A person takes one step and pauses",
                "dialogue": [],
                "sound_effects": "N/A",
            }],
        }
        for index, duration in enumerate(durations)
    ]


def _staged(ledger: dict, durations: list[float]) -> dict:
    return {
        "planned_by": "test_fixture",
        "ledger": ledger,
        "segments": _segments(durations),
    }


def _run_window_plan(ledger: dict) -> dict:
    def stage(_prompt: str, *, segment_durations: list[float], **_kwargs) -> dict:
        return _staged(ledger, segment_durations)

    with patch.object(h3_window_planner, "plan_h3_story_segments", side_effect=stage):
        return h3_window_planner.plan_h3_sliding_windows(
            _SOURCE_PROMPT,
            model_type="minimax_h3",
            resolution="1280x720",
            total_frames=47,
            window_frames=24,
            overlap_frames=1,
            camera_coverage="continuous",
        )


def _run_sequence_plan(ledger: dict) -> dict:
    def stage(_prompt: str, *, segment_durations: list[float], **_kwargs) -> dict:
        return _staged(ledger, segment_durations)

    with patch.object(h3_sequence_planner, "plan_h3_story_segments", side_effect=stage):
        return h3_sequence_planner.plan_h3_reference_sequence(
            _SOURCE_PROMPT,
            model_type="minimax_h3_ref2va",
            resolution="1280x720",
            total_frames=100,
            references=[],
            max_clip_frames=60,
            overlap_frames=10,
            native_continuation=True,
            camera_coverage="continuous",
        )


class H3NativeStableCameraContextTests(unittest.TestCase):
    def test_stable_map_removes_stale_global_action_from_native_prompts(self):
        for name, run_plan in (
            ("sliding-window", _run_window_plan),
            ("reference-sequence", _run_sequence_plan),
        ):
            with self.subTest(planner=name):
                ledger = _ledger(with_stable_context=True)
                result = run_plan(ledger)
                prompt = "\n".join(result["window_prompts"])

                self.assertNotIn("LEGACY-CONTEXT-77", prompt)
                self.assertNotIn("opens the archive door", prompt)
                self.assertNotIn("carries a red spool", prompt)
                self.assertIn(
                    ledger["motion_mechanics"].rstrip(".").casefold(),
                    prompt.casefold(),
                )
                self.assertIs(result["story_ledger"], ledger)
                for field in (
                    "subject_continuity",
                    "setting_continuity",
                    "visual_continuity",
                    "editing_style",
                    "motion_mechanics",
                    "source_relationships",
                ):
                    self.assertEqual(result["story_ledger"][field], ledger[field])
                self.assertEqual(
                    result["story_ledger"]["stable_camera_context"],
                    {
                        **{key: "" for key in (
                            "subject_continuity",
                            "setting_continuity",
                            "editing_style",
                        )},
                        "visual_continuity": ledger["motion_mechanics"],
                    },
                )
                if name == "sliding-window":
                    self.assertEqual(result["editing_style"], "")

    def test_legacy_fields_are_used_only_when_stable_map_is_absent(self):
        for name, run_plan in (
            ("sliding-window", _run_window_plan),
            ("reference-sequence", _run_sequence_plan),
        ):
            with self.subTest(planner=name):
                result = run_plan(_ledger(with_stable_context=False))
                prompt = "\n".join(result["window_prompts"])

                self.assertIn("LEGACY-CONTEXT-77", prompt)
                self.assertIn("opens the archive door", prompt)
                self.assertIn("carries a red spool", prompt)
                if name == "sliding-window":
                    self.assertEqual(result["editing_style"], _STALE_ACTION)


if __name__ == "__main__":
    unittest.main()
