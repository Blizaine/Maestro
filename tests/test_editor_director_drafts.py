"""Saved Editor projects preserve Director prompt drafts separately from renders."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.editor_projects import (  # noqa: E402
    create_editor_project,
    load_editor_project,
    save_editor_project,
)


class TestEditorDirectorDrafts(unittest.TestCase):
    def test_project_round_trip_keeps_draft_separate_from_last_rendered_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = "Director-Draft-Test"
            project = create_editor_project(name="Director draft", workspace=workspace)
            project["assets"]["source-shot"] = {
                "id": "source-shot",
                "name": "shot.mp4",
                "type": "video",
                "origin": "output",
                "workspace": workspace,
                "duration": 8,
                "width": 1280,
                "height": 720,
                "fps": 24,
                "has_audio": True,
            }
            project["tracks"][0]["items"] = [{
                "id": "director-shot",
                "asset_id": "source-shot",
                "name": "Shot 1",
                "start": 0,
                "duration": 5,
                "source_in": 0,
                "speed": 1,
                "volume": 1,
                "opacity": 1,
                "fit": "cover",
                "transform": {"x": 0, "y": 0, "scale": 1, "rotation": 0},
                "director": {
                    "pipeline_id": "pipeline-draft-test",
                    "clip_index": 0,
                    "pipeline_type": "music_video",
                    "workspace": workspace,
                    "video_prompt": "Original rendered prompt",
                    "window_prompts": ["Original rendered prompt"],
                    "draft_video_prompt": "  Revised draft, with exact spacing.  ",
                },
            }]

            saved = save_editor_project(temporary, workspace, project)
            loaded = load_editor_project(temporary, workspace, saved["id"])
            source = loaded["tracks"][0]["items"][0]["director"]

            self.assertEqual(source["video_prompt"], "Original rendered prompt")
            self.assertEqual(source["draft_video_prompt"], "  Revised draft, with exact spacing.  ")


if __name__ == "__main__":
    unittest.main()
