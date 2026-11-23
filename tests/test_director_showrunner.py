"""Model-free regressions for Director Showrunner boundary and lyric helpers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director_showrunner import (  # noqa: E402
    _current_joined_cuts,
    _enqueue_clip_upscale,
    _enqueue_joined_cut_upscale,
    _lyrics_subtitle_payload,
    _set_song_subtitles,
    lyrics_to_srt,
)


class TestDirectorShowrunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.originals = {
            "pipelines": pipeline._pipelines,
            "wgp": pipeline._wgp,
        }
        pipeline._pipelines = {}
        pipeline._wgp = SimpleNamespace(save_path=self.temp_dir.name)

    def tearDown(self):
        pipeline._pipelines = self.originals["pipelines"]
        pipeline._wgp = self.originals["wgp"]
        self.temp_dir.cleanup()

    def _saved_pipeline(self, clips) -> str:
        pid = "showrunner-test"
        path = os.path.join(
            self.temp_dir.name, f"_director_pipeline_{pid}.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "id": pid,
                "pipeline_id": pid,
                "pipeline_type": "music_video",
                "clips": clips,
                "_params_snapshot": {"lyrics": []},
            }, handle)
        return pid

    def test_lyrics_srt_sorts_offsets_and_clips_cues(self):
        content = lyrics_to_srt(
            [
                {"start": 15.0, "end": 19.1, "text": "late line"},
                {"start": 12.2, "end": 14.4, "text": "early\r\nline"},
                {"start": 11.0, "end": 12.0, "text": "before edit"},
                {"start": 20.0, "end": 21.0, "text": "after edit"},
                {"start": "bad", "end": 22.0, "text": "invalid"},
            ],
            timeline_start=12.0,
            duration=6.0,
        )

        self.assertEqual(
            content,
            "1\n00:00:00,200 --> 00:00:02,400\nearly\nline\n\n"
            "2\n00:00:03,000 --> 00:00:06,000\nlate line\n",
        )

    def test_lyrics_srt_rejects_untimed_or_empty_lyrics(self):
        with self.assertRaisesRegex(ValueError, "timed lyrics"):
            lyrics_to_srt("A song without timestamps")
        with self.assertRaisesRegex(ValueError, "usable timed lyrics"):
            lyrics_to_srt([{"start": 4, "end": 3, "text": "invalid"}])

    def test_boundary_choice_persists_and_marks_render_stale(self):
        pid = self._saved_pipeline([
            {"video_filename": "opening.mp4", "video_stale": False},
            {
                "video_filename": "next.mp4",
                "video_stale": False,
                "tag": "good",
                "approved_take_filename": "next.mp4",
                "approved_take_fingerprint": "approved",
            },
        ])

        updated = pipeline.update_clip_boundary_anchors(
            self.temp_dir.name,
            pid,
            1,
            start_from_previous=True,
            end_on_next=False,
        )

        state = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertTrue(updated["video_stale"])
        self.assertTrue(state["clips"][1]["start_from_previous"])
        self.assertFalse(state["clips"][1]["end_on_next"])
        self.assertTrue(state["clips"][1]["video_stale"])
        self.assertIsNone(state["clips"][1]["tag"])
        self.assertNotIn("approved_take_fingerprint", state["clips"][1])

    def test_source_take_change_marks_anchor_dependent_stale(self):
        state = {
            "clips": [
                {"video_filename": "source-new.mp4", "selected_take_id": "final"},
                {
                    "video_filename": "dependent.mp4",
                    "video_stale": False,
                    "tag": "good",
                    "approved_take_filename": "dependent.mp4",
                    "approved_take_fingerprint": "approved",
                    "_director_boundary_anchor_sources": [{
                        "clip_index": 0,
                        "filename": "source-old.mp4",
                        "selected_take_id": "draft",
                    }],
                },
            ],
        }

        pipeline._mark_director_anchor_dependents_stale(
            state,
            0,
            {"clip_index": 0, "filename": "source-old.mp4", "selected_take_id": "draft"},
            "source-new.mp4",
        )

        dependent = state["clips"][1]
        self.assertTrue(dependent["video_stale"])
        self.assertIsNone(dependent["tag"])
        self.assertNotIn("approved_take_fingerprint", dependent)

    def test_source_change_marks_planned_motion_continuation_stale(self):
        state = {"clips": [
            {
                "video_filename": "source-new.mp4",
                "planned_clip": {"_director_continuity_group": "scene-1"},
            },
            {
                "video_filename": "continued.mp4",
                "video_stale": False,
                "planned_clip": {
                    "_director_continuity_group": "scene-1",
                    "_director_continuity_strategy": "extend_previous",
                    "_director_segment_index": 1,
                },
            },
        ]}

        pipeline._mark_director_anchor_dependents_stale(
            state,
            0,
            {"clip_index": 0, "filename": "source-old.mp4"},
            "source-new.mp4",
        )

        self.assertTrue(state["clips"][1]["video_stale"])

    def test_song_subtitle_opt_in_returns_a_timed_soft_track(self):
        pid = self._saved_pipeline([{
            "planned_clip": {"start": 10.0, "end": 14.0},
        }])
        path = os.path.join(
            self.temp_dir.name, f"_director_pipeline_{pid}.json",
        )
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        state["_params_snapshot"]["lyrics"] = [
            {"start": 10.5, "end": 12.0, "text": "first line"},
        ]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)

        # Song subtitles are opt-in and default off.
        with self.assertRaisesRegex(ValueError, "Turn on"):
            _lyrics_subtitle_payload(
                pipeline.load_pipeline_state(self.temp_dir.name, pid),
            )
        _set_song_subtitles(self.temp_dir.name, pid, True)
        payload = _lyrics_subtitle_payload(
            pipeline.load_pipeline_state(self.temp_dir.name, pid),
        )
        self.assertEqual(payload["cue_count"], 1)
        self.assertEqual(
            payload["content"],
            "1\n00:00:00,500 --> 00:00:02,000\nfirst line\n",
        )

    def test_delivery_upscale_delegates_selected_take_to_media_flow(self):
        source = os.path.join(self.temp_dir.name, "shot.mp4")
        with open(source, "wb") as handle:
            handle.write(b"video")
        pid = self._saved_pipeline([{"video_filename": "shot.mp4"}])
        with open(
            os.path.join(self.temp_dir.name, f"_director_pipeline_{pid}.json"),
            "r",
            encoding="utf-8",
        ) as handle:
            state = json.load(handle)
        state["workspace"] = "editorial"
        with open(
            os.path.join(self.temp_dir.name, f"_director_pipeline_{pid}.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(state, handle)

        captured = []
        result = _enqueue_clip_upscale(
            self.temp_dir.name,
            pid,
            0,
            {"spatial_upsampling": "relayvsr2", "dlss_motion": "raft"},
            lambda body: captured.append(body) or {"job_ids": ["job-1"]},
        )

        self.assertEqual(result, {"job_ids": ["job-1"]})
        self.assertEqual(captured[0]["files"][0]["path"], os.path.realpath(source))
        self.assertEqual(captured[0]["spatial_upsampling"], "relayvsr2")
        self.assertEqual(captured[0]["temporal_upsampling"], "")
        self.assertEqual(captured[0]["workspace"], "editorial")
        self.assertEqual(captured[0]["dlss_motion"], "raft")

    def test_current_joined_cut_requires_matching_current_source_manifest(self):
        source_names = ["shot-1.mp4", "shot-2.mp4"]
        for filename in source_names:
            with open(os.path.join(self.temp_dir.name, filename), "wb") as handle:
                handle.write(b"source")
        cut_name = "2026_rejoin_multiclip.mp4"
        with open(os.path.join(self.temp_dir.name, cut_name), "wb") as handle:
            handle.write(b"joined")
        pid = self._saved_pipeline([
            {"video_filename": source_names[0]},
            {"video_filename": source_names[1]},
        ])
        state_path = os.path.join(
            self.temp_dir.name, f"_director_pipeline_{pid}.json",
        )
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        state["output_files"] = [cut_name]
        state["workspace"] = "editorial"
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        with open(
            os.path.join(self.temp_dir.name, "2026_rejoin_multiclip.meta.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump({
                "director_pipeline_id": pid,
                "director_artifact_kind": "joined",
                "output_filename": cut_name,
                "director_source_clips": source_names,
            }, handle)

        self.assertEqual(
            _current_joined_cuts(
                pipeline.load_pipeline_state(self.temp_dir.name, pid),
                self.temp_dir.name,
            ),
            [{"filename": cut_name}],
        )
        captured = []
        result = _enqueue_joined_cut_upscale(
            self.temp_dir.name,
            pid,
            {"filename": cut_name, "spatial_upsampling": "relayvsr2"},
            lambda body: captured.append(body) or {"job_ids": ["job-cut"]},
        )
        self.assertEqual(result, {"job_ids": ["job-cut"]})
        self.assertEqual(
            captured[0]["files"][0]["path"],
            os.path.realpath(os.path.join(self.temp_dir.name, cut_name)),
        )
        self.assertEqual(captured[0]["workspace"], "editorial")

        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        state["clips"][1]["video_filename"] = "new-shot-2.mp4"
        with open(os.path.join(self.temp_dir.name, "new-shot-2.mp4"), "wb") as handle:
            handle.write(b"new source")
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        self.assertEqual(
            _current_joined_cuts(
                pipeline.load_pipeline_state(self.temp_dir.name, pid),
                self.temp_dir.name,
            ),
            [],
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            _enqueue_joined_cut_upscale(
                self.temp_dir.name,
                pid,
                {"filename": cut_name, "spatial_upsampling": "relayvsr2"},
                lambda body: body,
            )


if __name__ == "__main__":
    unittest.main()
