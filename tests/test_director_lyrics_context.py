"""Lyric meaning is opt-in; timing and source vocal ownership remain intact."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.lyrics_context import lyrics_context_enabled
from services.director.planners.music_video import MusicVideoPlanner
from services import llm_service


class LyricsContextTests(unittest.TestCase):
    def setUp(self):
        self.lyrics = [{"start": 0, "end": 8, "text": "Purple submarines carry cinnamon clocks", "speaker": "singer"}]
        self.clips = [{"start": 0, "end": 10, "label": "verse", "beat_count": 8}]

    def test_only_explicit_boolean_enables_lyric_meaning(self):
        for value in (None, False, 0, 1, "false", "true", [], {}):
            self.assertFalse(lyrics_context_enabled(value))
        self.assertTrue(lyrics_context_enabled(True))

    def test_context_keeps_vocal_and_clip_timing_without_lyric_words(self):
        planner = MusicVideoPlanner()
        for source_audio in (True, False):
            with self.subTest(source_audio=source_audio):
                off = planner._build_clip_contexts(
                    self.clips, self.lyrics, {}, {"singer": "Lead"}, {},
                    source_audio_drives_vocals=source_audio,
                    use_lyrics_context=False,
                )[0]
                on = planner._build_clip_contexts(
                    self.clips, self.lyrics, {}, {"singer": "Lead"}, {},
                    source_audio_drives_vocals=source_audio,
                    use_lyrics_context=True,
                )[0]
                self.assertNotIn(self.lyrics[0]["text"], off)
                self.assertIn(self.lyrics[0]["text"], on)
                self.assertIn("00:10.000", off)
                self.assertIn("vocal", off)

    def test_legacy_visual_planner_respects_opt_out_and_retains_performer_timing(self):
        off = llm_service._build_clip_description_v2(self.clips[0], 0, self.lyrics, {}, {}, use_lyrics_context=False)
        on = llm_service._build_clip_description_v2(self.clips[0], 0, self.lyrics, {}, {}, use_lyrics_context=True)
        self.assertNotIn(self.lyrics[0]["text"], off)
        self.assertIn(self.lyrics[0]["text"], on)
        self.assertIn("vocals", off)

    def test_whole_song_outline_uses_lyrics_only_when_enabled(self):
        clips = [{"start": i * 10, "end": (i + 1) * 10} for i in range(4)]
        beats = ["The singer opens the curtain and watches the empty hall.",
                 "A hand traces the scratched edge of an old piano.",
                 "The drummer raises both sticks above the snare.",
                 "The group bows together beneath fading stage lights."]
        outline = [dict(clip_index=i, visual_beat=beat, purpose="Develop the performance",
                        environment="the hall", camera="fixed view", source_event_ids=[],
                        intentional_reprise=False, reprise_reason="") for i, beat in enumerate(beats)]
        for enabled in (False, True):
            planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
            planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"enabled": enabled})
            planner._music_narrative_contract = {"events": [], "source": "A stage performance."}
            planner._use_lyrics_context = enabled
            with patch.object(planner, "_call_llm_json", return_value=outline) as call:
                planner._plan_music_arc(clips, "A stage performance.", lyrics=self.lyrics)
            self.assertEqual(self.lyrics[0]["text"] in call.call_args.kwargs["user_prompt"], enabled)


if __name__ == "__main__":
    unittest.main()
