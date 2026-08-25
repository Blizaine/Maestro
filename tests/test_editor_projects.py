"""Model-free regressions for Maestro's non-destructive Editor projects."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.editor_projects import (  # noqa: E402
    EditorProjectError,
    compile_editor_render,
    create_editor_project,
    delete_editor_project,
    editor_project_duration,
    list_editor_projects,
    load_editor_project,
    normalize_editor_project,
    probe_media,
    resolve_editor_asset,
    save_editor_project,
)


class TestEditorProjects(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.save_root = os.path.join(self.temp.name, "outputs")
        self.uploads_root = os.path.join(self.temp.name, "uploads")
        os.makedirs(self.save_root)
        os.makedirs(self.uploads_root)

    def tearDown(self):
        self.temp.cleanup()

    def _media(self, name: str, *, upload: bool = False) -> str:
        root = self.uploads_root if upload else self.save_root
        path = os.path.join(root, name)
        with open(path, "wb") as handle:
            handle.write(b"editor-test-media")
        return path

    def test_atomic_round_trip_list_and_delete_preserve_timestamps(self):
        project = create_editor_project(name="My cut", workspace="default")
        saved = save_editor_project(self.save_root, "default", project)
        loaded = load_editor_project(self.save_root, "default", saved["id"])
        self.assertEqual(loaded["id"], saved["id"])
        self.assertEqual(loaded["updated_at"], saved["updated_at"])

        summaries = list_editor_projects(self.save_root, "default")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["name"], "My cut")
        self.assertEqual(summaries[0]["updated_at"], saved["updated_at"])
        self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(
            os.path.join(self.save_root, ".maestro_editor")
        )))

        self.assertTrue(delete_editor_project(self.save_root, "default", saved["id"]))
        self.assertEqual(list_editor_projects(self.save_root, "default"), [])

    def test_normalization_keeps_valid_text_and_rejects_orphan_media(self):
        project = create_editor_project()
        project["tracks"][0]["items"] = [{
            "id": "orphan",
            "asset_id": "missing",
            "start": -12,
            "duration": 2,
        }]
        project["tracks"][2]["items"] = [{
            "id": "title-one",
            "text": "Hello",
            "start": 1,
            "duration": 3,
        }]
        normalized = normalize_editor_project(project)
        self.assertEqual(normalized["tracks"][0]["items"], [])
        self.assertEqual(normalized["tracks"][2]["items"][0]["text"], "Hello")
        self.assertEqual(editor_project_duration(normalized), 4)

    def test_asset_resolution_never_escapes_owned_roots(self):
        outside = os.path.join(self.temp.name, "private.mp4")
        with open(outside, "wb") as handle:
            handle.write(b"private")
        with self.assertRaises(EditorProjectError):
            resolve_editor_asset(
                {"name": "private.mp4", "path": outside, "origin": "output"},
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
            )

        upload = self._media("voice.wav", upload=True)
        resolved = resolve_editor_asset(
            {"name": "voice.wav", "origin": "upload"},
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
        )
        self.assertEqual(os.path.realpath(resolved), os.path.realpath(upload))

    def test_render_compiler_builds_visual_text_and_audio_graph(self):
        video_path = self._media("clip.mp4")
        audio_path = self._media("score.wav", upload=True)
        project = create_editor_project(name="Timeline")
        project["assets"] = {
            "video-asset": {
                "id": "video-asset",
                "name": os.path.basename(video_path),
                "type": "video",
                "origin": "output",
                "duration": 8,
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "has_audio": True,
            },
            "audio-asset": {
                "id": "audio-asset",
                "name": os.path.basename(audio_path),
                "type": "audio",
                "origin": "upload",
                "duration": 8,
                "width": 0,
                "height": 0,
                "fps": 0,
                "has_audio": True,
            },
        }
        common = {
            "start": 0,
            "duration": 5,
            "source_in": 1,
            "speed": 1,
            "volume": 0.8,
            "opacity": 1,
            "fit": "cover",
            "transform": {"x": 0, "y": 0, "scale": 1, "rotation": 0},
        }
        project["tracks"][0]["items"] = [{
            **common,
            "id": "video-item",
            "asset_id": "video-asset",
            "name": "Clip",
        }]
        project["tracks"][1]["items"] = [{
            **common,
            "id": "audio-item",
            "asset_id": "audio-asset",
            "name": "Score",
        }]
        project["tracks"][2]["items"] = [{
            **common,
            "id": "title-item",
            "name": "Title",
            "text": "Opening title",
            "style": {"x": 0, "y": 0, "font_size": 64, "color": "#ffffff"},
        }]
        filter_path = os.path.join(self.temp.name, "filter.txt")
        output_path = os.path.join(self.temp.name, "export.mp4")
        command, duration = compile_editor_render(
            project,
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
            filter_script_path=filter_path,
            output_path=output_path,
        )
        self.assertEqual(duration, 5)
        self.assertIn("-filter_complex_script", command)
        self.assertIn(output_path, command)
        with open(filter_path, "r", encoding="utf-8") as handle:
            graph = handle.read()
        self.assertIn("overlay=", graph)
        self.assertIn("drawtext=", graph)
        self.assertIn("amix=inputs=2", graph)
        self.assertIn("[editor_video]", graph)
        self.assertIn("[editor_audio]", graph)

    def test_muted_video_track_is_excluded_from_export(self):
        video_path = self._media("muted.mp4")
        project = create_editor_project(name="Muted track")
        project["assets"] = {
            "video-asset": {
                "id": "video-asset",
                "name": os.path.basename(video_path),
                "type": "video",
                "origin": "output",
                "duration": 3,
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "has_audio": True,
            },
        }
        project["tracks"][0]["muted"] = True
        project["tracks"][0]["items"] = [{
            "id": "muted-video",
            "asset_id": "video-asset",
            "name": "Muted video",
            "start": 0,
            "duration": 3,
            "source_in": 0,
            "speed": 1,
            "volume": 1,
            "opacity": 1,
            "fit": "cover",
            "transform": {"x": 0, "y": 0, "scale": 1, "rotation": 0},
        }]

        filter_path = os.path.join(self.temp.name, "muted-filter.txt")
        output_path = os.path.join(self.temp.name, "muted-export.mp4")
        command, duration = compile_editor_render(
            project,
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
            filter_script_path=filter_path,
            output_path=output_path,
        )

        self.assertEqual(duration, 3)
        self.assertNotIn(video_path, command)
        with open(filter_path, "r", encoding="utf-8") as handle:
            graph = handle.read()
        self.assertNotIn("overlay=", graph)
        self.assertNotIn("[0:a]", graph)
        self.assertIn("anullsrc", graph)

    def test_probe_accepts_non_numeric_frame_count(self):
        image_path = self._media("still.png")
        payload = {
            "streams": [{
                "codec_type": "video",
                "nb_frames": "N/A",
                "width": 640,
                "height": 480,
                "avg_frame_rate": "0/0",
            }],
            "format": {"duration": "0"},
        }
        completed = Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with patch("services.editor_projects.subprocess.run", return_value=completed):
            result = probe_media(image_path)
        self.assertEqual(result["type"], "image")
        self.assertEqual(result["width"], 640)


if __name__ == "__main__":
    unittest.main()
