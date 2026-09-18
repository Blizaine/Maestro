"""Issue #115: an inline product timeline is visual direction, not speech."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3.speakers import is_h3_production_label
from services.dialogue_writing import dialogue_forbidden
from services.h3_story_ledger import (
    extract_locked_dialogue, extract_source_events, plan_h3_story_segments,
)


SOURCE = """A silent 10.1-second vertical 9:16 photorealistic studio product video.

A black water-purifier tower with four floating 3D elements, identical to Picture 1, in the Picture-1 grey studio. Frame 0 replicates Picture 1 exactly — background, framing, pose and lighting are reproduced, not replaced. Picture 2 defines only the floating glass logotype above the product: world-locked, camera-facing.

Motion from Video 1, 1:1: purifier and all floating elements rotate as one rigid group around the vertical axis — single direction, constant ~24°/s, strictly monotonic, never pausing or reversing — reaching ~240° at the final frame; each element keeps its gentle vertical bob.

Logotype timeline: [0–1s] absent; [1–3s] soft blue water particles fade in and condense left-to-right into crisp glass letters; [3–4.5s] complete, one gentle shimmer; [4.5–7.5s] dissolves bottom-up into rising blue mist; [7.5–8s] particles fade; [8–10.1s] absent, never returns. Exactly two particle events.

Camera fully static — no pan, tilt, zoom, push-in or shake. Never adopt Video 1's flat shading or viewport gizmo."""


class ProductTimelineTests(unittest.TestCase):
    def test_original_report_has_no_dialogue_and_retains_visual_timeline(self):
        self.assertTrue(dialogue_forbidden(SOURCE))
        self.assertEqual(extract_locked_dialogue(SOURCE), [])
        events = " ".join(event["text"] for event in extract_source_events(SOURCE))
        for detail in ("condense left-to-right", "one gentle shimmer",
                       "dissolves bottom-up", "never returns"):
            self.assertIn(detail, events)

    def test_timeline_heading_is_not_a_speaker_even_without_silent_instruction(self):
        for label in ("Logotype timeline", "Logo timing", "Product animation timeline", "Timeline"):
            with self.subTest(label=label):
                self.assertTrue(is_h3_production_label(label))
                lines = extract_locked_dialogue(
                    f'{label}: [0-1s] absent; [1-3s] letters form.\nMira: "It is ready."'
                )
                self.assertEqual([(line["speaker"], line["text"]) for line in lines],
                                 [("Mira", "It is ready.")])

    def test_silent_duration_qualifiers_preserve_local_silence_and_explicit_speech(self):
        self.assertTrue(dialogue_forbidden("A silent 30s commercial video."))
        self.assertFalse(dialogue_forbidden("A silent 10-second product video plays before Mira speaks."))
        self.assertTrue(dialogue_forbidden("A silent 10-second product video plays throughout the entire scene."))
        self.assertFalse(dialogue_forbidden('Mira says, "A silent 10-second product video is ready."'))
        lines = extract_locked_dialogue(SOURCE + '\nMira: "This line is explicitly supplied."')
        self.assertEqual([line["text"] for line in lines], ["This line is explicitly supplied."])

    def test_three_windows_do_not_charge_visual_events_to_the_speech_budget(self):
        # Exercise the real scheduler with the writer unavailable. Failure to
        # write a camera draft can require review, but must not invent speech
        # or raise H3DialogueTimingError before the draft can be reviewed.
        def offline(**_kwargs):
            raise RuntimeError("test writer unavailable")

        result = plan_h3_story_segments(
            SOURCE, segment_durations=[5.0, 5.0, 5.0], mode="reference_sequence",
            camera_coverage="continuous", expect_dialogue=True,
            planning_style="faithful", llm_generate=offline,
        )
        self.assertEqual(len(result["segments"]), 3)
        self.assertEqual([
            line for segment in result["segments"] for shot in segment["shots"]
            for line in shot.get("dialogue", [])
        ], [])


if __name__ == "__main__":
    unittest.main()
