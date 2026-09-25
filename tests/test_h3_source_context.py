import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_source_context import (
    is_audio_binding_direction, is_media_no_additions, is_media_subject_only, is_story_scope_direction,
    media_subject_prefix, strip_modified_context_headings,
)


class SourceContextTests(unittest.TestCase):
    def test_modified_heading_preserves_action_tail(self):
        actual = strip_modified_context_headings(
            "A quiet, hopeful story: each evening Nora leaves a bouquet.",
            is_static=lambda text: text == "quiet, hopeful",
        )
        self.assertEqual(actual, " each evening Nora leaves a bouquet.")

    def test_action_or_unrecognized_prefix_is_not_removed(self):
        for text in ("Nora rewrites the story: keep the ending.",
                     "A child changes the story: Nora leaves."):
            self.assertEqual(strip_modified_context_headings(text, is_static=lambda _: False), text)

    def test_duration_subject_stays_whole(self):
        for subject in ("The 55.25-second video", "This 20-second clip", "The camera"):
            self.assertTrue(is_media_subject_only(subject))
            self.assertEqual(media_subject_prefix(subject + " holds the final frame"), subject)
            self.assertFalse(is_media_subject_only(subject + " holds the frame"))

    def test_negative_output_lists(self):
        for text in (
            "Add no dialogue, new vocals, instruments, effects, replacement or supplemental music, loop, or stretch.",
            "show no musician", "Include no extra audience", "Add no speech or music",
        ):
            self.assertTrue(is_media_no_additions(text), text)
        for text in ("No musician enters the room", "Show no hesitation as Nora crosses the hall",
                     "Add no music, then Nora opens the door", "Nora says add no dialogue"):
            self.assertFalse(is_media_no_additions(text), text)

    def test_scope_direction_cannot_hide_new_ending(self):
        self.assertTrue(is_story_scope_direction(
            "Keep the story at this station and end on the exchanged gesture."
        ))
        for text in ("Keep Nora at this station", "Keep the story at this station and end with Nora opening the box",
                     "Keep the story at this station and leave the suitcase"):
            self.assertFalse(is_story_scope_direction(text), text)

    def test_audio_binding_does_not_consume_visual_timing(self):
        for text in ("Use <Audio 1> as the exact performance-driving soundtrack, once at its supplied speed and without edits.",
                     "The source track is 37.327 seconds."):
            self.assertTrue(is_audio_binding_direction(text), text)
        for text in ("Finish the movement at its final note.",
                     "The source track is 37.327 seconds and Ada waves.",
                     "Use <Audio 1> as the soundtrack while Ada waves."):
            self.assertFalse(is_audio_binding_direction(text), text)

    def test_source_catalog_excludes_metadata_but_intent_keeps_it(self):
        from services.h3_story_ledger import extract_h3_source_intent, extract_source_events
        source = (
            "A quiet, hopeful story: each evening a florist leaves a bouquet. "
            "On the last night a commuter leaves one fresh stem. No dialogue. "
            "Keep the story at this station and end on the exchanged gesture."
        )
        events = extract_source_events(source)
        self.assertTrue(any("each evening" in event["text"] for event in events))
        self.assertIn("fresh stem", events[-1]["text"])
        self.assertFalse(any(event["text"] == "A quiet, hopeful" for event in events))
        intent = extract_h3_source_intent(source)
        self.assertIn("Keep the story at this station", str(intent))

    def test_audio_source_keeps_movement_and_silent_hold_on_timeline(self):
        from services.h3_story_ledger import extract_h3_source_intent, extract_source_events
        source = (
            "A performer raises one cloth, then lowers it beside a lantern. "
            "Use <Audio 1> as the exact performance-driving soundtrack, once at its supplied speed and without edits. "
            "The source track is 37.327 seconds; finish the movement at its final note. "
            "The 55.25-second video then holds the final lantern-and-cloth composition in silence for the remaining 17.923 seconds. "
            "Add no dialogue, new vocals, instruments, effects, replacement or supplemental music, loop, or stretch; show no musician."
        )
        events = extract_source_events(source)
        texts = [item["text"] for item in events]
        self.assertTrue(any("finish the movement" in text for text in texts))
        self.assertTrue(any("video holds the final" in text for text in texts))
        self.assertFalse(any(text == "The 55.25-second video" or text.startswith("The holds") for text in texts))
        self.assertFalse(any(text.lower().startswith(("add no", "show no", "use <audio", "the source track")) for text in texts))
        intent = str(extract_h3_source_intent(source))
        self.assertIn("37.327", intent)
        self.assertIn("show no musician", intent)


if __name__ == "__main__":
    unittest.main()
