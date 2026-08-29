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
    build_editor_media_preview,
    EditorProjectError,
    compile_editor_render,
    create_editor_project,
    delete_editor_project,
    editor_export_dimensions,
    editor_export_capabilities,
    editor_project_duration,
    inspect_editor_assets,
    list_editor_projects,
    load_editor_project,
    normalize_editor_project,
    probe_media,
    resolve_editor_asset,
    resolve_editor_media_cache_file,
    resolve_editor_export_encoder,
    save_editor_project,
)
from services.director_pipeline import build_pipeline_first_frame_thumbnail  # noqa: E402


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

    def test_normalization_preserves_safe_director_rerun_provenance(self):
        project = create_editor_project(workspace="Director-Testing")
        project["assets"]["director-asset"] = {
            "id": "director-asset",
            "name": "shot-1.mp4",
            "type": "video",
            "origin": "output",
            "workspace": "Director-Testing",
            "duration": 8,
            "width": 1280,
            "height": 720,
            "fps": 24,
            "has_audio": True,
        }
        project["tracks"][0]["items"] = [{
            "id": "director-shot-1",
            "asset_id": "director-asset",
            "name": "Shot 1",
            "start": 0,
            "duration": 8,
            "source_in": 0,
            "speed": 1,
            "volume": 1,
            "opacity": 1,
            "fit": "contain",
            "transform": {"x": 0, "y": 0, "scale": 1, "rotation": 0},
            "director": {
                "pipeline_id": "7e0f5020",
                "clip_index": 0,
                "pipeline_type": "short_film_story",
                "workspace": "Director-Testing",
                "video_prompt": "A carefully directed first shot.",
                "window_prompts": ["Opening beat", "Closing beat"],
            },
            "ai_history": [{
                "id": "director-rerun-1",
                "tool": "director_rerun",
                "asset_id": "director-asset",
                "created_at": 42,
            }],
        }]

        normalized = normalize_editor_project(project)
        item = normalized["tracks"][0]["items"][0]

        self.assertEqual(item["director"]["pipeline_id"], "7e0f5020")
        self.assertEqual(item["director"]["clip_index"], 0)
        self.assertEqual(item["director"]["window_prompts"], [
            "Opening beat", "Closing beat",
        ])
        self.assertEqual(item["ai_history"][0]["tool"], "director_rerun")

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

    def test_asset_resolution_can_use_media_from_another_workspace(self):
        source_workspace = os.path.join(self.save_root, "archive")
        os.makedirs(source_workspace)
        source = os.path.join(source_workspace, "source.mp4")
        with open(source, "wb") as handle:
            handle.write(b"cross-workspace-editor-media")

        resolved = resolve_editor_asset(
            {
                "name": "source.mp4",
                "origin": "output",
                "workspace": "archive",
            },
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
        )

        self.assertEqual(os.path.realpath(resolved), os.path.realpath(source))

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

    def test_schema_five_normalizes_fades_title_style_and_track_volume(self):
        project = create_editor_project()
        project["schema_version"] = 1
        project["tracks"][0]["volume"] = 9
        project["tracks"][2]["items"] = [{
            "id": "styled-title",
            "name": "Styled title",
            "text": "Hello",
            "start": 0,
            "duration": 2,
            "fade_in": 8,
            "fade_out": -3,
            "opacity": 0.7,
            "style": {
                "font_family": "Georgia",
                "font_size": 72,
                "color": "#ff8800",
                "background_color": "#101010",
                "background_opacity": 0.6,
                "text_align": "right",
            },
        }]

        normalized = normalize_editor_project(project)

        self.assertEqual(normalized["schema_version"], 5)
        self.assertEqual(normalized["tracks"][0]["volume"], 4)
        title = normalized["tracks"][2]["items"][0]
        self.assertEqual(title["fade_in"], 2)
        self.assertEqual(title["fade_out"], 0)
        self.assertEqual(title["style"]["text_align"], "right")
        self.assertEqual(title["style"]["background_color"], "#101010")
        self.assertEqual(title["style"]["font_family"], "Georgia")

        project["tracks"][2]["items"][0]["style"]["font_family"] = "Untrusted font path"
        normalized = normalize_editor_project(project)
        self.assertEqual(normalized["tracks"][2]["items"][0]["style"]["font_family"], "Arial")

    def test_schema_five_normalizes_markers_links_takes_and_ai_history(self):
        project = create_editor_project()
        project["assets"] = {
            "source": {
                "id": "source", "name": "source.mp4", "type": "video",
                "origin": "output", "duration": 4, "width": 640, "height": 360,
                "fps": 24, "has_audio": True,
            },
            "take-two": {
                "id": "take-two", "name": "take-two.mp4", "type": "video",
                "origin": "output", "duration": 4, "width": 640, "height": 360,
                "fps": 24, "has_audio": True,
            },
        }
        project["markers"] = [
            {"id": "marker-b", "time": 3, "label": "  Reveal  ", "color": "invalid"},
            {"id": "marker-a", "time": -4, "label": "", "color": "#123456"},
        ]
        project["tracks"][0]["items"] = [{
            "id": "clip-one",
            "asset_id": "source",
            "name": "Source",
            "start": 0,
            "duration": 4,
            "link_group_id": "linked-pair",
            "transition_in": "dissolve",
            "transition_out": "unsupported",
            "take_asset_ids": ["source", "missing", "take-two", "source"],
            "take_states": {
                "source": {"source_in": 4.25, "speed": 1.5},
                "take-two": {"source_in": 999, "speed": 99},
                "missing": {"source_in": 2, "speed": 1},
            },
            "ai_history": [
                {"id": "history-one", "tool": "recast", "asset_id": "take-two", "created_at": 10},
                {"id": "history-bad", "tool": "unknown", "asset_id": "source"},
            ],
        }]

        normalized = normalize_editor_project(project)
        item = normalized["tracks"][0]["items"][0]

        self.assertEqual(normalized["schema_version"], 5)
        self.assertEqual([marker["time"] for marker in normalized["markers"]], [0, 3])
        self.assertEqual(normalized["markers"][0]["label"], "Marker")
        self.assertEqual(normalized["markers"][1]["label"], "Reveal")
        self.assertEqual(normalized["markers"][1]["color"], "#f59e0b")
        self.assertEqual(item["link_group_id"], "linked-pair")
        self.assertEqual(item["transition_in"], "dissolve")
        self.assertEqual(item["transition_out"], "none")
        self.assertEqual(item["take_asset_ids"], ["source", "take-two"])
        self.assertEqual(item["take_states"]["source"], {"source_in": 0, "speed": 1})
        self.assertAlmostEqual(item["take_states"]["take-two"]["source_in"], 3.9666666667)
        self.assertEqual(item["take_states"]["take-two"]["speed"], 8.0)
        self.assertEqual(len(item["ai_history"]), 1)
        self.assertEqual(item["ai_history"][0]["tool"], "recast")

    def test_normalization_clamps_media_items_to_source_duration(self):
        project = create_editor_project()
        project["assets"] = {
            "short-video": {
                "id": "short-video",
                "name": "short.mp4",
                "type": "video",
                "origin": "output",
                "duration": 10,
                "width": 1280,
                "height": 720,
                "fps": 30,
                "has_audio": True,
            },
        }
        project["tracks"][0]["items"] = [{
            "id": "trimmed-video",
            "asset_id": "short-video",
            "name": "Trimmed video",
            "start": 0,
            "duration": 5,
            "source_in": 8,
            "speed": 1.5,
            "fade_in": 4,
            "fade_out": 4,
        }]

        normalized = normalize_editor_project(project)
        item = normalized["tracks"][0]["items"][0]

        self.assertAlmostEqual(item["duration"], 4 / 3)
        self.assertAlmostEqual(item["source_in"] + item["duration"] * item["speed"], 10)
        self.assertAlmostEqual(item["fade_in"], item["duration"])
        self.assertAlmostEqual(item["fade_out"], item["duration"])

    def test_normalization_prevents_items_overlapping_on_one_track(self):
        project = create_editor_project()
        project["tracks"][2]["items"] = [
            {
                "id": "title-a",
                "name": "A",
                "text": "A",
                "start": 0,
                "duration": 3,
            },
            {
                "id": "title-b",
                "name": "B",
                "text": "B",
                "start": 1,
                "duration": 2,
            },
            {
                "id": "title-c",
                "name": "C",
                "text": "C",
                "start": 5,
                "duration": 1,
            },
        ]

        normalized = normalize_editor_project(project)
        items = normalized["tracks"][2]["items"]

        self.assertEqual([item["id"] for item in items], ["title-a", "title-b", "title-c"])
        self.assertEqual([item["start"] for item in items], [0, 3, 5])
        for left, right in zip(items, items[1:]):
            self.assertLessEqual(left["start"] + left["duration"], right["start"])

    def test_render_compiler_emits_visual_audio_and_title_fades(self):
        video_path = self._media("fade.mp4")
        project = create_editor_project(name="Fades")
        project["assets"] = {
            "fade-asset": {
                "id": "fade-asset",
                "name": os.path.basename(video_path),
                "type": "video",
                "origin": "output",
                "duration": 5,
                "width": 1280,
                "height": 720,
                "fps": 30,
                "has_audio": True,
            },
        }
        project["tracks"][0]["items"] = [{
            "id": "fade-video",
            "asset_id": "fade-asset",
            "name": "Fade video",
            "start": 0,
            "duration": 5,
            "source_in": 0,
            "speed": 1,
            "volume": 1,
            "opacity": 1,
            "fit": "cover",
            "transform": {"x": 0, "y": 0, "scale": 1, "rotation": 0},
            "fade_in": 0.5,
            "fade_out": 0.75,
            "transition_in": "fade_black",
            "transition_out": "dissolve",
        }]
        project["tracks"][2]["items"] = [{
            "id": "fade-title",
            "name": "Fade title",
            "text": "Hello",
            "start": 1,
            "duration": 2,
            "source_in": 0,
            "speed": 1,
            "volume": 1,
            "opacity": 0.8,
            "fit": "contain",
            "transform": {"x": 0, "y": 0, "scale": 1, "rotation": 0},
            "fade_in": 0.25,
            "fade_out": 0.25,
            "style": {
                "x": 0,
                "y": 0,
                "font_family": "Impact",
                "font_size": 64,
                "color": "#ffffff",
                "background_color": "#112233",
                "background_opacity": 0.5,
                "text_align": "left",
            },
        }]

        filter_path = os.path.join(self.temp.name, "fade-filter.txt")
        with patch(
            "services.editor_projects._drawtext_font_option",
            return_value=":fontfile='selected-font.ttf'",
        ) as font_option:
            compile_editor_render(
                project,
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
                filter_script_path=filter_path,
                output_path=os.path.join(self.temp.name, "fade-output.mp4"),
            )
        font_option.assert_called_once_with("Impact")
        with open(filter_path, "r", encoding="utf-8") as handle:
            graph = handle.read()
        self.assertIn("fade=t=in:st=0.00000000:d=0.50000000:color=black", graph)
        self.assertIn("fade=t=out:st=4.25000000:d=0.75000000:alpha=1", graph)
        self.assertIn("afade=t=out:st=4.25000000:d=0.75000000", graph)
        self.assertIn("boxcolor=#112233@0.50000000", graph)
        self.assertIn("alpha='0.80000000*", graph)
        self.assertIn(":fontfile='selected-font.ttf'", graph)

    def test_delivery_settings_control_resolution_codec_audio_and_history(self):
        video_path = self._media("delivery.mp4")
        project = create_editor_project(
            name="Vertical delivery", width=1080, height=1920, fps=30
        )
        project["assets"] = {
            "delivery-asset": {
                "id": "delivery-asset",
                "name": os.path.basename(video_path),
                "type": "video",
                "origin": "output",
                "duration": 3,
                "width": 1080,
                "height": 1920,
                "fps": 30,
                "has_audio": True,
            },
        }
        project["tracks"][0]["items"] = [{
            "id": "delivery-clip",
            "asset_id": "delivery-asset",
            "name": "Delivery",
            "start": 0,
            "duration": 3,
            "source_in": 0,
            "speed": 1,
            "volume": 1,
            "opacity": 1,
            "fit": "cover",
            "transform": {"x": 0, "y": 0, "scale": 1, "rotation": 0},
        }]
        project["export"] = {
            "quality": "balanced",
            "codec": "h265",
            "include_audio": False,
            "resolution": "720p",
            "frame_rate": 24,
            "filename": "My Final / unsafe.mp4",
        }
        project["exports"] = [{
            "id": "past-export",
            "filename": "../past.mp4",
            "workspace": "default",
            "created_at": 12,
            "duration": 3,
            "width": 720,
            "height": 1280,
            "fps": 24,
            "codec": "h265",
            "quality": "balanced",
        }]

        normalized = normalize_editor_project(project)
        self.assertEqual(normalized["schema_version"], 5)
        self.assertEqual(normalized["export"]["encoder"], "auto")
        self.assertEqual(normalized["exports"][0]["filename"], "past.mp4")
        self.assertEqual(editor_export_dimensions(normalized), (720, 1280, 24.0))

        filter_path = os.path.join(self.temp.name, "delivery-filter.txt")
        command, _ = compile_editor_render(
            normalized,
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
            filter_script_path=filter_path,
            output_path=os.path.join(self.temp.name, "delivery-output.mp4"),
        )
        self.assertIn("libx265", command)
        self.assertIn("hvc1", command)
        self.assertIn("-an", command)
        self.assertNotIn("[editor_audio]", command)
        with open(filter_path, "r", encoding="utf-8") as handle:
            graph = handle.read()
        self.assertIn("scale=720:1280:flags=lanczos", graph)
        self.assertNotIn("anullsrc", graph)

    def test_media_audit_reports_available_and_offline_assets(self):
        available_path = self._media("available.mp4")
        project = create_editor_project()
        project["assets"] = {
            "available": {
                "id": "available", "name": os.path.basename(available_path),
                "type": "video", "origin": "output", "duration": 1,
                "width": 640, "height": 360, "fps": 24, "has_audio": False,
            },
            "offline": {
                "id": "offline", "name": "moved.mp4", "type": "video",
                "origin": "output", "duration": 1, "width": 640,
                "height": 360, "fps": 24, "has_audio": False,
            },
        }

        status = inspect_editor_assets(
            project,
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
        )

        by_id = {entry["asset_id"]: entry for entry in status}
        self.assertTrue(by_id["available"]["available"])
        self.assertEqual(os.path.realpath(by_id["available"]["path"]), os.path.realpath(available_path))
        self.assertFalse(by_id["offline"]["available"])

    def test_mobile_preview_always_builds_lightweight_ios_safe_proxy(self):
        source = self._media("mobile-preview.mp4")
        commands = []

        def fake_preview_command(command, *, timeout=180):
            del timeout
            commands.append(command)
            output = command[-1]
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, "wb") as handle:
                handle.write(b"cached-editor-preview")
            return True

        with patch(
            "services.editor_projects._run_preview_command",
            side_effect=fake_preview_command,
        ):
            preview = build_editor_media_preview(
                {
                    "id": "mobile-preview",
                    "name": os.path.basename(source),
                    "path": source,
                    "type": "video",
                    "origin": "output",
                    "duration": 4,
                    "width": 640,
                    "height": 360,
                    "fps": 24,
                    "has_audio": False,
                    "size": os.path.getsize(source),
                },
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
                include_proxy=True,
                proxy_profile="mobile",
            )

        self.assertTrue(preview["proxy_mobile"])
        proxy_command = next(
            command for command in commands
            if str(command[-1]).endswith(".part.mp4")
        )
        self.assertIn("fps=30", proxy_command[proxy_command.index("-vf") + 1])
        self.assertIn("scale=960:540", proxy_command[proxy_command.index("-vf") + 1])
        self.assertIn("main", proxy_command)
        proxy_path = resolve_editor_media_cache_file(
            self.save_root,
            "default",
            preview["preview_id"],
            "proxy-mobile",
        )
        self.assertTrue(os.path.isfile(proxy_path))

    def test_director_gallery_thumbnail_uses_first_clip_and_cache(self):
        pipeline_id = "thumbnail-test"
        source = self._media("director-first-shot.mp4")
        os.utime(source, (1, 1))
        pipeline_path = os.path.join(
            self.save_root, f"_director_pipeline_{pipeline_id}.json"
        )
        with open(pipeline_path, "w", encoding="utf-8") as handle:
            json.dump({
                "pipeline_id": pipeline_id,
                "clips": [{"video_filename": os.path.basename(source)}],
                "output_files": [],
            }, handle)

        def fake_ffmpeg(command, **_kwargs):
            output = command[-1]
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, "wb") as handle:
                handle.write(b"director-first-frame")
            return Mock(returncode=0)

        with patch(
            "services.director_pipeline.subprocess.run",
            side_effect=fake_ffmpeg,
        ) as run:
            first = build_pipeline_first_frame_thumbnail(
                self.save_root, pipeline_id, ffmpeg="mock-ffmpeg"
            )
            second = build_pipeline_first_frame_thumbnail(
                self.save_root, pipeline_id, ffmpeg="mock-ffmpeg"
            )

        self.assertEqual(first, second)
        self.assertTrue(os.path.isfile(first))
        self.assertEqual(run.call_count, 1)
        self.assertIn("-frames:v", run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("-i") + 1], source)

    def test_hardware_encoder_selection_is_capability_aware(self):
        export = {"codec": "h264", "quality": "high", "encoder": "auto"}
        selected, codec, args = resolve_editor_export_encoder(
            export,
            {"encoders": {"software": True, "nvidia": True, "intel": False, "apple": False}},
        )
        self.assertEqual((selected, codec), ("nvidia", "h264_nvenc"))
        self.assertIn("-cq", args)

        selected, codec, args = resolve_editor_export_encoder(export, None)
        self.assertEqual((selected, codec), ("software", "libx264"))
        self.assertIn("-crf", args)

    def test_export_capability_probe_detects_encoder_families(self):
        completed = Mock(
            returncode=0,
            stdout="h264_nvenc hevc_nvenc h264_qsv hevc_qsv",
            stderr="",
        )
        editor_export_capabilities.cache_clear()
        with patch("services.editor_projects.subprocess.run", return_value=completed):
            result = editor_export_capabilities("mock-ffmpeg")
        editor_export_capabilities.cache_clear()
        self.assertTrue(result["encoders"]["nvidia"])
        self.assertTrue(result["encoders"]["intel"])
        self.assertFalse(result["encoders"]["apple"])
        self.assertEqual(result["recommended"], "nvidia")


if __name__ == "__main__":
    unittest.main()
