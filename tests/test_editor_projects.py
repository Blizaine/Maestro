"""Model-free regressions for Maestro's non-destructive Editor projects."""

from __future__ import annotations

import json
import io
import math
import os
import shutil
import subprocess
import sys
import struct
import tempfile
import threading
import time
import unittest
import wave
import zlib
from array import array
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.editor_projects import (  # noqa: E402
    build_editor_media_preview,
    EditorProjectError,
    EditorProjectRevisionConflict,
    EditorRenderCancelled,
    compile_editor_render,
    create_editor_project,
    delete_editor_project,
    editor_export_dimensions,
    editor_export_capabilities,
    editor_project_duration,
    inspect_editor_assets,
    list_editor_projects,
    load_editor_project,
    list_editor_project_revisions,
    normalize_editor_project,
    probe_media,
    resolve_editor_asset,
    resolve_editor_media_cache_file,
    resolve_editor_export_encoder,
    render_editor_project,
    restore_editor_project_revision,
    save_editor_project,
)
from services.editor_animation import (  # noqa: E402
    editor_color_filters,
    editor_keyframe_expression,
    evaluate_editor_keyframes,
)
import services.editor_projects as editor_projects  # noqa: E402
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

    def _write_ppm(
        self,
        name: str,
        rgb: tuple[int, int, int],
        *,
        width: int = 64,
        height: int = 64,
    ) -> str:
        path = os.path.join(self.save_root, name)
        with open(path, "wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(bytes(rgb) * width * height)
        return path

    def _write_rgba_png(
        self,
        name: str,
        rgba: tuple[int, int, int, int],
        *,
        width: int = 64,
        height: int = 64,
    ) -> str:
        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        row = b"\x00" + bytes(rgba) * width
        image = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * height))
            + chunk(b"IEND", b"")
        )
        path = os.path.join(self.save_root, name)
        with open(path, "wb") as handle:
            handle.write(image)
        return path

    def _render_cpu_project(self, project: dict, output_path: str) -> str:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("FFmpeg is not installed")
        filter_script_path = os.path.join(self.temp.name, "cpu-render-filter.txt")
        command, _duration = compile_editor_render(
            project,
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
            filter_script_path=filter_script_path,
            output_path=output_path,
            ffmpeg=ffmpeg,
            encoder_capabilities={
                "encoders": {
                    "software": True,
                    "nvidia": False,
                    "intel": False,
                    "apple": False,
                },
                "recommended": "software",
            },
        )
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode("utf-8", errors="replace")[-2000:],
        )
        return ffmpeg

    def _sample_render_rgb(
        self,
        ffmpeg: str,
        path: str,
        time_seconds: float,
        *,
        width: int = 64,
        height: int = 64,
        fps: float = 2,
    ) -> tuple[int, int, int]:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode("utf-8", errors="replace")[-2000:],
        )
        frame_size = width * height * 3
        frame_index = int(round(time_seconds * fps))
        self.assertGreater(
            len(result.stdout),
            frame_index * frame_size,
            f"decoded {len(result.stdout) // frame_size} video frames",
        )
        offset = (
            frame_index * frame_size
            + ((height // 2) * width + (width // 2)) * 3
        )
        return tuple(result.stdout[offset:offset + 3])

    def test_atomic_round_trip_list_and_delete_preserve_timestamps(self):
        project = create_editor_project(name="My cut", workspace="default")
        saved = save_editor_project(self.save_root, "default", project)
        loaded = load_editor_project(self.save_root, "default", saved["id"])
        self.assertEqual(loaded["id"], saved["id"])
        self.assertEqual(loaded["updated_at"], saved["updated_at"])
        self.assertEqual(saved["revision"], 1)
        self.assertEqual(loaded["revision"], 1)

        summaries = list_editor_projects(self.save_root, "default")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["name"], "My cut")
        self.assertEqual(summaries[0]["updated_at"], saved["updated_at"])
        self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(
            os.path.join(self.save_root, ".maestro_editor")
        )))

        self.assertTrue(delete_editor_project(self.save_root, "default", saved["id"]))
        self.assertEqual(list_editor_projects(self.save_root, "default"), [])

    def test_project_revision_increments_and_rejects_stale_explicit_writes(self):
        project = create_editor_project(name="Revisioned edit")
        first = save_editor_project(self.save_root, "default", project)
        second = save_editor_project(self.save_root, "default", first)
        self.assertEqual((first["revision"], second["revision"]), (1, 2))

        stale = dict(first)
        stale["name"] = "Stale overwrite"
        with self.assertRaises(EditorProjectRevisionConflict) as raised:
            save_editor_project(self.save_root, "default", stale)
        self.assertEqual(raised.exception.expected_revision, 1)
        self.assertEqual(raised.exception.current_revision, 2)
        self.assertIn("current revision 2", str(raised.exception))
        self.assertEqual(
            load_editor_project(self.save_root, "default", project["id"])["name"],
            "Revisioned edit",
        )

    def test_revisionless_legacy_save_still_succeeds_and_advances_revision(self):
        project = create_editor_project(name="Legacy client edit")
        first = save_editor_project(self.save_root, "default", project)
        legacy_payload = dict(first)
        legacy_payload.pop("revision")
        legacy_payload["name"] = "Saved by a revisionless client"

        saved = save_editor_project(self.save_root, "default", legacy_payload)

        self.assertEqual(saved["revision"], 2)
        loaded = load_editor_project(self.save_root, "default", project["id"])
        self.assertEqual(loaded["revision"], 2)
        self.assertEqual(loaded["name"], "Saved by a revisionless client")

    def test_recovery_restores_prior_snapshot_monotonically_and_checks_revision(self):
        project = create_editor_project(name="First draft")
        first = save_editor_project(self.save_root, "default", project)
        second_draft = {**first, "name": "Second draft"}
        second = save_editor_project(
            self.save_root,
            "default",
            second_draft,
            expected_revision=first["revision"],
        )

        history = list_editor_project_revisions(
            self.save_root, "default", first["id"]
        )
        self.assertEqual([entry["revision"] for entry in history], [1])
        restored = restore_editor_project_revision(
            self.save_root,
            "default",
            first["id"],
            1,
            expected_revision=second["revision"],
        )
        self.assertEqual(restored["name"], "First draft")
        self.assertEqual(restored["revision"], 3)
        with self.assertRaises(EditorProjectRevisionConflict):
            restore_editor_project_revision(
                self.save_root,
                "default",
                first["id"],
                1,
                expected_revision=2,
            )
        self.assertEqual(
            [entry["revision"] for entry in list_editor_project_revisions(
                self.save_root, "default", first["id"]
            )],
            [2, 1],
        )

    def test_recovery_history_is_bounded_and_old_snapshots_are_immutable(self):
        project = create_editor_project(name="Revision 0")
        saved = save_editor_project(self.save_root, "default", project)
        snapshot_path = os.path.join(
            self.save_root,
            ".maestro_editor",
            "history",
            saved["id"],
            "r3.json",
        )
        for index in range(2, 24):
            saved = save_editor_project(
                self.save_root,
                "default",
                {**saved, "name": f"Revision {index}"},
                expected_revision=saved["revision"],
            )
        with open(snapshot_path, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        self.assertEqual(snapshot["name"], "Revision 3")
        revisions = list_editor_project_revisions(
            self.save_root, "default", saved["id"]
        )
        self.assertEqual(len(revisions), 20)
        self.assertEqual(revisions[0]["revision"], 22)
        self.assertEqual(revisions[-1]["revision"], 3)

    def test_recovery_skips_mismatched_snapshot_and_refuses_to_restore_it(self):
        first = save_editor_project(
            self.save_root,
            "default",
            create_editor_project(name="Original snapshot"),
        )
        second = save_editor_project(
            self.save_root,
            "default",
            {**first, "name": "Second revision"},
            expected_revision=first["revision"],
        )
        current = save_editor_project(
            self.save_root,
            "default",
            {**second, "name": "Current live project"},
            expected_revision=second["revision"],
        )
        corrupt_snapshot_path = os.path.join(
            self.save_root,
            ".maestro_editor",
            "history",
            first["id"],
            "r1.json",
        )
        with open(corrupt_snapshot_path, "r", encoding="utf-8") as handle:
            corrupt_snapshot = json.load(handle)
        corrupt_snapshot["revision"] = 2
        corrupt_snapshot["name"] = "Mismatched payload"
        with open(corrupt_snapshot_path, "w", encoding="utf-8") as handle:
            json.dump(corrupt_snapshot, handle)

        listed = list_editor_project_revisions(
            self.save_root, "default", first["id"]
        )
        self.assertEqual([entry["revision"] for entry in listed], [2])
        with self.assertRaisesRegex(
            EditorProjectError,
            "snapshot revision does not match its filename",
        ):
            restore_editor_project_revision(
                self.save_root,
                "default",
                first["id"],
                1,
                expected_revision=current["revision"],
            )

        live = load_editor_project(self.save_root, "default", first["id"])
        self.assertEqual(live["revision"], current["revision"])
        self.assertEqual(live["name"], "Current live project")

    def test_save_accepts_an_explicit_expected_revision_argument(self):
        project = create_editor_project(name="Explicit revision")
        first = save_editor_project(self.save_root, "default", project)
        revisionless_payload = dict(first)
        revisionless_payload.pop("revision")

        second = save_editor_project(
            self.save_root,
            "default",
            revisionless_payload,
            expected_revision=1,
        )
        self.assertEqual(second["revision"], 2)
        with self.assertRaises(EditorProjectRevisionConflict):
            save_editor_project(
                self.save_root,
                "default",
                revisionless_payload,
                expected_revision=1,
            )

    def test_new_project_id_starts_at_revision_one_even_when_duplicated(self):
        original = save_editor_project(
            self.save_root,
            "default",
            create_editor_project(name="Original"),
        )
        duplicate = dict(original)
        duplicate["id"] = "duplicate-project"
        duplicate["name"] = "Duplicate"
        duplicate["revision"] = 7

        saved_duplicate = save_editor_project(self.save_root, "default", duplicate)

        self.assertEqual(saved_duplicate["revision"], 1)
        self.assertEqual(
            load_editor_project(self.save_root, "default", "duplicate-project")["name"],
            "Duplicate",
        )

    def test_old_projects_without_revision_load_as_revision_zero(self):
        project = create_editor_project(name="Old project")
        project.pop("revision")
        project_path = os.path.join(
            self.save_root, ".maestro_editor", f"{project['id']}.json"
        )
        os.makedirs(os.path.dirname(project_path), exist_ok=True)
        with open(project_path, "w", encoding="utf-8") as handle:
            json.dump(project, handle)

        loaded = load_editor_project(self.save_root, "default", project["id"])
        self.assertEqual(loaded["revision"], 0)
        saved = save_editor_project(self.save_root, "default", loaded)
        self.assertEqual(saved["revision"], 1)

    def test_normalization_rejects_oversized_project_collections_without_truncating(self):
        cases = [
            ("assets", {f"asset-{index}": {} for index in range(2001)}, "2000 assets"),
            ("tracks", [{} for _ in range(101)], "100 tracks"),
            ("markers", [{} for _ in range(1001)], "1000 markers"),
            ("exports", [{} for _ in range(51)], "50 export records"),
        ]
        for field, value, message in cases:
            with self.subTest(field=field):
                project = create_editor_project()
                project[field] = value
                with self.assertRaisesRegex(EditorProjectError, message):
                    normalize_editor_project(project)

        project = create_editor_project()
        project["tracks"][0]["items"] = [{} for _ in range(5001)]
        with self.assertRaisesRegex(EditorProjectError, "5000 items"):
            normalize_editor_project(project)

        for field, message in (
            ("take_asset_ids", "100 takes"),
            ("ai_history", "100 AI history entries"),
        ):
            with self.subTest(field=field):
                project = create_editor_project()
                item = {"start": 0, "duration": 1, field: ["entry"] * 101}
                project["tracks"][0]["items"] = [item]
                with self.assertRaisesRegex(EditorProjectError, message):
                    normalize_editor_project(project)

        project = create_editor_project()
        project["tracks"][0]["items"] = [{
            "start": 0,
            "duration": 1,
            "take_states": {f"take-{index}": {} for index in range(101)},
        }]
        with self.assertRaisesRegex(EditorProjectError, "100 take states"):
            normalize_editor_project(project)

        project = create_editor_project()
        project["tracks"][0]["items"] = [{
            "start": 0,
            "duration": 1,
            "director": {"window_prompts": ["prompt"] * 101},
        }]
        with self.assertRaisesRegex(EditorProjectError, "100 Director window prompts"):
            normalize_editor_project(project)

    def test_normalization_rejects_nonfinite_and_over_24_hour_timeline_times(self):
        invalid_items = [
            ({"start": float("nan"), "duration": 1}, "start must be a finite number"),
            ({"start": 0, "duration": float("inf")}, "duration must be a finite number"),
            ({"start": 86401, "duration": 1}, "starts beyond the 24-hour project limit"),
            ({"start": 86399, "duration": 2}, "ends beyond the 24-hour project limit"),
        ]
        for item, message in invalid_items:
            with self.subTest(item=item):
                project = create_editor_project()
                project["tracks"][2]["items"] = [item]
                with self.assertRaisesRegex(EditorProjectError, message):
                    normalize_editor_project(project)

        project = create_editor_project()
        project["tracks"][2]["items"] = [
            {"id": "a", "start": 86399, "duration": 1},
            {"id": "b", "start": 86399, "duration": 1},
        ]
        with self.assertRaisesRegex(EditorProjectError, "24-hour project limit"):
            normalize_editor_project(project)

        project = create_editor_project()
        project["markers"] = [{"id": "late-marker", "time": 86401}]
        with self.assertRaisesRegex(EditorProjectError, "marker 1 is beyond"):
            normalize_editor_project(project)

        project = create_editor_project()
        project["tracks"][2]["items"] = [{
            "id": "boundary-title",
            "start": 86399,
            "duration": 1,
            "text": "Boundary",
        }]
        self.assertEqual(
            editor_project_duration(normalize_editor_project(project)),
            86400,
        )

        # Keep the old source-duration clamp behavior when it brings the actual
        # rendered end inside the limit, even if the raw item duration is longer.
        project = create_editor_project()
        project["assets"] = {
            "short-source": {
                "id": "short-source",
                "name": "short.mp4",
                "type": "video",
                "origin": "output",
                "duration": 1,
            },
        }
        project["tracks"][0]["items"] = [{
            "id": "late-short-clip",
            "asset_id": "short-source",
            "start": 86399,
            "duration": 2,
        }]
        normalized = normalize_editor_project(project)
        self.assertEqual(normalized["tracks"][0]["items"][0]["duration"], 1)

    def test_project_validation_runs_before_save_and_preserves_existing_file(self):
        project = create_editor_project(name="Safe project")
        saved = save_editor_project(self.save_root, "default", project)
        invalid = dict(saved)
        invalid["tracks"] = [{} for _ in range(101)]
        with self.assertRaisesRegex(EditorProjectError, "100 tracks"):
            save_editor_project(self.save_root, "default", invalid)

        loaded = load_editor_project(self.save_root, "default", saved["id"])
        self.assertEqual(loaded["name"], "Safe project")
        self.assertEqual(loaded["revision"], 1)

    def test_project_summaries_reuse_stat_keyed_cache_but_load_still_normalizes(self):
        saved = save_editor_project(
            self.save_root,
            "default",
            create_editor_project(name="Cached summary"),
        )
        project_path = os.path.join(
            self.save_root, ".maestro_editor", f"{saved['id']}.json"
        )
        with patch(
            "services.editor_projects.normalize_editor_project",
            wraps=normalize_editor_project,
        ) as normalize:
            first = list_editor_projects(self.save_root, "default")
            second = list_editor_projects(self.save_root, "default")
            self.assertEqual(first, second)
            normalize.assert_not_called()

            loaded = load_editor_project(self.save_root, "default", saved["id"])
            self.assertEqual(loaded["name"], "Cached summary")
            normalize.assert_called_once()

            with open(project_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["name"] = "Externally modified project name"
            with open(project_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            normalize.reset_mock()

            changed = list_editor_projects(self.save_root, "default")
            self.assertEqual(changed[0]["name"], "Externally modified project name")
            normalize.assert_called_once()
            normalize.reset_mock()
            list_editor_projects(self.save_root, "default")
            normalize.assert_not_called()

    def test_project_summary_cache_has_a_fixed_entry_bound(self):
        for index in range(editor_projects.EDITOR_PROJECT_SUMMARY_CACHE_MAX_ENTRIES + 8):
            project = create_editor_project(name=f"Cached {index}")
            project["id"] = f"summary-cache-{index}"
            save_editor_project(self.save_root, "default", project)

        self.assertLessEqual(
            len(editor_projects._project_summary_cache),
            editor_projects.EDITOR_PROJECT_SUMMARY_CACHE_MAX_ENTRIES,
        )

    def test_same_preview_source_is_built_once_for_concurrent_callers(self):
        source = self._media("shared-preview.mp4")
        asset = {
            "id": "shared-preview",
            "name": os.path.basename(source),
            "path": source,
            "type": "video",
            "origin": "output",
            "duration": 4,
            "width": 640,
            "height": 360,
            "has_audio": False,
            "size": os.path.getsize(source),
        }
        barrier = threading.Barrier(6)
        command_count = 0
        command_lock = threading.Lock()

        def fake_preview_command(command, *, timeout=180):
            nonlocal command_count
            del timeout
            with command_lock:
                command_count += 1
            time.sleep(0.04)
            output = command[-1]
            with open(output, "wb") as handle:
                handle.write(b"thumbnail")
            return True

        def build():
            barrier.wait()
            return build_editor_media_preview(
                asset,
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
            )

        with patch(
            "services.editor_projects._run_preview_command",
            side_effect=fake_preview_command,
        ):
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(lambda _index: build(), range(6)))

        self.assertEqual(command_count, 1)
        self.assertEqual(len({result["preview_id"] for result in results}), 1)
        self.assertTrue(all(result["thumbnail"] for result in results))

    def test_distinct_preview_builds_use_at_most_two_workers(self):
        assets = []
        for index in range(5):
            source = self._media(f"parallel-preview-{index}.mp4")
            assets.append({
                "id": f"parallel-{index}",
                "name": os.path.basename(source),
                "path": source,
                "type": "video",
                "origin": "output",
                "duration": 4,
                "width": 640,
                "height": 360,
                "has_audio": False,
                "size": os.path.getsize(source),
            })
        barrier = threading.Barrier(len(assets))
        command_lock = threading.Lock()
        current_builds = 0
        maximum_builds = 0

        def fake_preview_command(command, *, timeout=180):
            nonlocal current_builds, maximum_builds
            del timeout
            with command_lock:
                current_builds += 1
                maximum_builds = max(maximum_builds, current_builds)
            time.sleep(0.08)
            output = command[-1]
            with open(output, "wb") as handle:
                handle.write(b"thumbnail")
            with command_lock:
                current_builds -= 1
            return True

        def build(asset):
            barrier.wait()
            return build_editor_media_preview(
                asset,
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
            )

        with patch(
            "services.editor_projects._run_preview_command",
            side_effect=fake_preview_command,
        ):
            with ThreadPoolExecutor(max_workers=len(assets)) as pool:
                results = list(pool.map(build, assets))

        self.assertEqual(len(results), len(assets))
        self.assertEqual(maximum_builds, editor_projects.EDITOR_MEDIA_PREVIEW_WORKERS)
        self.assertLessEqual(maximum_builds, 2)

    def test_preview_cache_pruning_removes_expired_entries_only_inside_root(self):
        cache_root = editor_projects.editor_media_cache_directory(
            self.save_root, "default"
        )
        os.makedirs(cache_root, exist_ok=True)
        old_id = "a" * 24
        protected_id = "b" * 24
        old_entry = os.path.join(cache_root, old_id)
        protected_entry = os.path.join(cache_root, protected_id)
        os.makedirs(old_entry)
        os.makedirs(protected_entry)
        old_artifact = os.path.join(old_entry, "thumbnail.jpg")
        protected_artifact = os.path.join(protected_entry, "thumbnail.jpg")
        for artifact in (old_artifact, protected_artifact):
            with open(artifact, "wb") as handle:
                handle.write(b"cached")

        now = time.time()
        expired = now - editor_projects.EDITOR_MEDIA_PREVIEW_CACHE_TTL_SECONDS - 10
        os.utime(old_entry, (expired, expired))
        resolve_editor_media_cache_file(
            self.save_root, "default", protected_id, "thumbnail"
        )

        editor_projects._prune_editor_media_cache(
            cache_root,
            now=now,
            max_bytes=1,
            ttl_seconds=editor_projects.EDITOR_MEDIA_PREVIEW_CACHE_TTL_SECONDS,
            active_seconds=editor_projects.EDITOR_MEDIA_PREVIEW_ACTIVE_SECONDS,
        )

        self.assertFalse(os.path.exists(old_entry))
        self.assertTrue(os.path.isfile(protected_artifact))

    def test_preview_cache_quota_ignores_external_symlink_targets(self):
        cache_root = editor_projects.editor_media_cache_directory(
            self.save_root, "default"
        )
        os.makedirs(cache_root, exist_ok=True)
        external_root = os.path.join(self.temp.name, "external-cache-target")
        os.makedirs(external_root)
        external_file = os.path.join(external_root, "keep.txt")
        with open(external_file, "w", encoding="utf-8") as handle:
            handle.write("outside the cache")
        link_path = os.path.join(cache_root, "c" * 24)
        try:
            os.symlink(external_root, link_path, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"directory symlinks are unavailable: {error}")

        editor_projects._prune_editor_media_cache(
            cache_root,
            now=time.time() + 10_000,
            max_bytes=0,
            ttl_seconds=0,
            active_seconds=0,
        )

        self.assertTrue(os.path.isfile(external_file))

    def test_keyframe_and_color_values_are_normalized_and_validated(self):
        project = create_editor_project()
        project["assets"] = {
            "clip": {
                "id": "clip",
                "name": "clip.mp4",
                "type": "video",
                "origin": "output",
                "duration": 10,
                "width": 1280,
                "height": 720,
                "has_audio": True,
            }
        }
        project["tracks"][0]["items"] = [{
            "id": "animated",
            "asset_id": "clip",
            "start": 0,
            "duration": 2,
            "transform": {"x": 4},
            "keyframes": {
                "x": [
                    {"time": 2, "value": 20, "easing": "ease-out"},
                    {"time": 0, "value": -5, "easing": "linear"},
                ],
                "opacity": [
                    {"time": 0, "value": 0.2, "easing": "linear"},
                ],
            },
            "color": {
                "exposure": 9,
                "contrast": -1,
                "temperature": 0.5,
            },
        }]

        normalized = normalize_editor_project(project)
        item = normalized["tracks"][0]["items"][0]
        self.assertEqual(
            [point["time"] for point in item["keyframes"]["x"]],
            [0, 2],
        )
        self.assertEqual(item["keyframes"]["x"][0]["value"], -5)
        self.assertEqual(item["color"], {
            "exposure": 2,
            "contrast": 0,
            "saturation": 1,
            "temperature": 0.5,
        })

        invalid_keyframes = [
            {"x": [{"time": float("nan"), "value": 0}]},
            {"scale": [{"time": 0, "value": 5}]},
            {"unknown": [{"time": 0, "value": 1}]},
            {"x": [
                {"time": 0, "value": 0},
                {"time": 0, "value": 1},
            ]},
            {"x": [{"time": 1, "value": 0, "easing": "cubic"}]},
            {"x": [{"time": 0, "value": 0}] * 1001},
        ]
        for keyframes in invalid_keyframes:
            with self.subTest(keyframes=str(keyframes)[:100]):
                bad_project = copy_project = json.loads(json.dumps(project))
                copy_project["tracks"][0]["items"][0]["keyframes"] = keyframes
                with self.assertRaises(EditorProjectError):
                    normalize_editor_project(bad_project)

        clamped_project = json.loads(json.dumps(project))
        clamped_project["assets"]["clip"]["duration"] = 1
        clamped_project["tracks"][0]["items"][0]["keyframes"]["x"] = [
            {"time": 1.5, "value": 0, "easing": "linear"}
        ]
        with self.assertRaisesRegex(EditorProjectError, "clip duration"):
            normalize_editor_project(clamped_project)

        invalid_color = json.loads(json.dumps(project))
        invalid_color["tracks"][0]["items"][0]["color"] = {
            "exposure": float("inf")
        }
        with self.assertRaisesRegex(EditorProjectError, "color.exposure"):
            normalize_editor_project(invalid_color)

    def test_keyframe_expressions_match_easing_and_hold_semantics(self):
        for easing, expected in (
            ("linear", 5),
            ("hold", 0),
            ("ease-in", 2.5),
            ("ease-out", 7.5),
            ("ease-in-out", 5),
        ):
            with self.subTest(easing=easing):
                points = [
                    {"time": 0, "value": 0, "easing": easing},
                    {"time": 2, "value": 10, "easing": "linear"},
                ]
                expression = editor_keyframe_expression(
                    {"x": points},
                    "x",
                    fallback=99,
                    time_expression="(t-3.00000000)",
                )
                self.assertEqual(
                    evaluate_editor_keyframes(points, 1, fallback=99),
                    expected,
                )
                self.assertLess(len(expression), 1000)
                self.assertNotIn("nan", expression.lower())
                self.assertNotIn(";", expression)
                self.assertEqual(
                    evaluate_editor_keyframes(points, -1, fallback=99),
                    0,
                )
                self.assertEqual(
                    evaluate_editor_keyframes(points, 5, fallback=99),
                    10,
                )

    def test_explicit_dissolve_overlap_is_preserved_but_legacy_overlap_is_shifted(self):
        project = create_editor_project()
        project["assets"] = {
            asset_id: {
                "id": asset_id,
                "name": f"{asset_id}.png",
                "type": "image",
                "origin": "output",
                "duration": 4,
                "width": 64,
                "height": 64,
                "has_audio": False,
            }
            for asset_id in ("first", "second")
        }
        project["tracks"][0]["items"] = [
            {"id": "first-item", "asset_id": "first", "start": 0, "duration": 2},
            {
                "id": "second-item",
                "asset_id": "second",
                "start": 1,
                "duration": 2,
                "fade_in": 1,
                "transition_in": "dissolve",
                "transition_overlap": 1,
            },
        ]

        normalized = normalize_editor_project(project)
        items = normalized["tracks"][0]["items"]
        self.assertEqual(items[1]["start"], 1)
        self.assertEqual(items[1]["transition_overlap"], 1)

        legacy_project = json.loads(json.dumps(project))
        legacy_project["tracks"][0]["items"][1].pop("transition_overlap")
        legacy_project["tracks"][0]["items"][1]["transition_in"] = "none"
        legacy = normalize_editor_project(legacy_project)
        self.assertEqual(legacy["tracks"][0]["items"][1]["start"], 2)

        invalid_project = json.loads(json.dumps(project))
        invalid_project["tracks"][0]["items"][1]["transition_overlap"] = 1.5
        with self.assertRaisesRegex(EditorProjectError, "does not match"):
            normalize_editor_project(invalid_project)

        invalid_project = json.loads(json.dumps(project))
        invalid_project["tracks"][0]["items"][1]["transition_in"] = "fade_black"
        with self.assertRaisesRegex(EditorProjectError, "requires transition_in"):
            normalize_editor_project(invalid_project)

        invalid_project = json.loads(json.dumps(project))
        invalid_project["tracks"][0]["items"][1]["fade_in"] = 0.5
        with self.assertRaisesRegex(EditorProjectError, "fade_in to cover"):
            normalize_editor_project(invalid_project)

        invalid_project = json.loads(json.dumps(project))
        invalid_project["tracks"][0]["items"][0]["fade_out"] = 0.25
        with self.assertRaisesRegex(EditorProjectError, "opaque preceding clip"):
            normalize_editor_project(invalid_project)

    def test_color_filter_order_and_temperature_gains_are_bounded(self):
        filters = editor_color_filters({
            "exposure": 1,
            "contrast": 1.25,
            "saturation": 0.75,
            "temperature": 0.5,
        })
        self.assertEqual(
            [entry.split("=")[0] for entry in filters],
            ["exposure", "eq", "eq", "colorchannelmixer"],
        )
        self.assertIn("exposure=exposure=1.00000000", filters[0])
        self.assertIn("contrast=1.25000000", filters[1])
        self.assertIn("saturation=0.75000000", filters[2])
        self.assertIn("rr=1.10000000", filters[3])
        self.assertIn("bb=0.90000000", filters[3])

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_cpu_export_opacity_multiplies_source_alpha(self):
        source = self._write_rgba_png("half-alpha-red.png", (255, 0, 0, 128))
        project = create_editor_project(name="Opacity preserves source alpha")
        project["canvas"].update({"width": 64, "height": 64, "fps": 2})
        project["export"].update({"encoder": "software", "include_audio": False})
        project["assets"] = {
            "half-alpha-red": {
                "id": "half-alpha-red",
                "name": os.path.basename(source),
                "path": source,
                "type": "image",
                "origin": "output",
                "duration": 1,
                "width": 64,
                "height": 64,
                "has_audio": False,
            }
        }
        project["tracks"][0]["items"] = [{
            "id": "animated-opacity",
            "asset_id": "half-alpha-red",
            "start": 0,
            "duration": 1,
            "keyframes": {
                "opacity": [
                    {"time": 0, "value": 0.5, "easing": "linear"},
                    {"time": 1, "value": 0.5, "easing": "linear"},
                ]
            },
        }]
        output_path = os.path.join(self.temp.name, "source-alpha-opacity.mp4")

        ffmpeg = self._render_cpu_project(project, output_path)
        pixel = self._sample_render_rgb(ffmpeg, output_path, 0.5)
        # Source alpha (~.5) multiplied by animated opacity (.5) yields ~.25
        # red over black. Replacing source alpha would yield roughly .5 red.
        self.assertTrue(35 <= pixel[0] <= 85, pixel)
        self.assertLess(pixel[1], 25, pixel)
        self.assertLess(pixel[2], 25, pixel)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_cpu_export_renders_animated_text_scale(self):
        project = create_editor_project(name="Animated text scale check")
        project["canvas"].update({"width": 128, "height": 128, "fps": 2})
        project["export"].update({"encoder": "software", "include_audio": False})
        project["tracks"][2]["items"] = [{
            "id": "title",
            "text": "Scale",
            "start": 0,
            "duration": 1,
            "style": {"font_size": 12, "color": "#ffffff"},
            "keyframes": {
                "scale": [
                    {"time": 0, "value": 1, "easing": "linear"},
                    {"time": 1, "value": 3, "easing": "linear"},
                ]
            },
        }]
        output_path = os.path.join(self.temp.name, "animated-text-scale.mp4")

        ffmpeg = self._render_cpu_project(project, output_path)
        decoded = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                output_path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(
            decoded.returncode,
            0,
            decoded.stderr.decode("utf-8", errors="replace")[-2000:],
        )
        frame_size = 128 * 128 * 3
        self.assertGreaterEqual(len(decoded.stdout), frame_size * 2)

        def bright_pixels(frame: bytes) -> int:
            return sum(
                1
                for offset in range(0, len(frame), 3)
                if frame[offset] > 160
                and frame[offset + 1] > 160
                and frame[offset + 2] > 160
            )

        first = bright_pixels(decoded.stdout[:frame_size])
        second = bright_pixels(decoded.stdout[frame_size:frame_size * 2])
        self.assertGreater(first, 0)
        self.assertGreater(second, first * 1.5, (first, second))

    def test_static_text_scale_is_compiled_into_font_size(self):
        project = create_editor_project(name="Static text scale")
        project["canvas"].update({"width": 128, "height": 128, "fps": 2})
        project["tracks"][2]["items"] = [{
            "id": "title",
            "text": "Scale",
            "start": 0,
            "duration": 1,
            "transform": {"scale": 2},
            "style": {"font_size": 12, "color": "#ffffff"},
        }]
        filter_path = os.path.join(self.temp.name, "static-text-scale-filter.txt")

        compile_editor_render(
            project,
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
            filter_script_path=filter_path,
            output_path=os.path.join(self.temp.name, "static-text-scale.mp4"),
        )

        with open(filter_path, "r", encoding="utf-8") as handle:
            graph = handle.read()
        self.assertIn("fontsize='12*(2.00000000)'", graph)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_cpu_export_renders_animated_transform_opacity_and_color(self):
        source = self._write_ppm("animated-gray.ppm", (50, 50, 50))
        project = create_editor_project(name="Animated pixel check")
        project["canvas"].update({"width": 64, "height": 64, "fps": 2})
        project["export"].update({"encoder": "software", "include_audio": False})
        project["assets"] = {
            "gray": {
                "id": "gray",
                "name": os.path.basename(source),
                "path": source,
                "type": "image",
                "origin": "output",
                "duration": 1,
                "width": 64,
                "height": 64,
                "has_audio": False,
                "size": os.path.getsize(source),
            }
        }
        project["tracks"][0]["items"] = [{
            "id": "animated-gray",
            "asset_id": "gray",
            "start": 0,
            "duration": 1,
            "transform": {"x": 0, "y": 0, "scale": 1, "rotation": 0},
            "keyframes": {
                "x": [
                    {"time": 0, "value": 0, "easing": "linear"},
                    {"time": 1, "value": 4, "easing": "linear"},
                ],
                "y": [
                    {"time": 0, "value": 0, "easing": "linear"},
                    {"time": 1, "value": 4, "easing": "linear"},
                ],
                "scale": [
                    {"time": 0, "value": 1, "easing": "linear"},
                    {"time": 1, "value": 1.25, "easing": "linear"},
                ],
                "rotation": [
                    {"time": 0, "value": 0, "easing": "linear"},
                    {"time": 1, "value": 10, "easing": "linear"},
                ],
                "opacity": [
                    {"time": 0, "value": 0, "easing": "linear"},
                    {"time": 1, "value": 1, "easing": "linear"},
                ],
            },
            "color": {
                "exposure": 1,
                "contrast": 1,
                "saturation": 1,
                "temperature": 0.5,
            },
        }]
        output_path = os.path.join(self.temp.name, "animated-pixel.mp4")

        ffmpeg = self._render_cpu_project(project, output_path)
        pixel = self._sample_render_rgb(ffmpeg, output_path, 0.5)

        # At t=.5 the input is ~100 RGB after +1 EV, the animated opacity is
        # .5, and temperature gains make the red channel slightly brighter
        # than green and blue slightly darker. Lossy YUV/H.264 permits a range.
        self.assertTrue(40 <= pixel[0] <= 75, pixel)
        self.assertTrue(35 <= pixel[1] <= 70, pixel)
        self.assertTrue(28 <= pixel[2] <= 65, pixel)
        self.assertGreater(pixel[0], pixel[2], pixel)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_cpu_export_dissolves_incoming_clip_over_opaque_predecessor(self):
        red_path = self._write_ppm("crossfade-red.ppm", (200, 0, 0))
        blue_path = self._write_ppm("crossfade-blue.ppm", (0, 0, 200))
        project = create_editor_project(name="Dissolve pixel check")
        project["canvas"].update({"width": 64, "height": 64, "fps": 2})
        project["export"].update({"encoder": "software", "include_audio": False})
        project["assets"] = {
            "red": {
                "id": "red",
                "name": os.path.basename(red_path),
                "path": red_path,
                "type": "image",
                "origin": "output",
                "duration": 2,
                "width": 64,
                "height": 64,
                "has_audio": False,
            },
            "blue": {
                "id": "blue",
                "name": os.path.basename(blue_path),
                "path": blue_path,
                "type": "image",
                "origin": "output",
                "duration": 2,
                "width": 64,
                "height": 64,
                "has_audio": False,
            },
        }
        project["tracks"][0]["items"] = [
            {"id": "red-clip", "asset_id": "red", "start": 0, "duration": 2},
            {
                "id": "blue-clip",
                "asset_id": "blue",
                "start": 1,
                "duration": 2,
                "fade_in": 1,
                "transition_in": "dissolve",
                "transition_overlap": 1,
            },
        ]
        output_path = os.path.join(self.temp.name, "dissolve-pixel.mp4")

        ffmpeg = self._render_cpu_project(project, output_path)
        before = self._sample_render_rgb(ffmpeg, output_path, 0.5)
        middle = self._sample_render_rgb(ffmpeg, output_path, 1.5)
        after = self._sample_render_rgb(ffmpeg, output_path, 2.0)

        self.assertGreater(before[0], 150, before)
        self.assertLess(before[2], 45, before)
        self.assertTrue(70 <= middle[0] <= 165, middle)
        self.assertTrue(70 <= middle[2] <= 165, middle)
        self.assertLess(middle[1], 50, middle)
        self.assertLess(after[0], 45, after)
        self.assertGreater(after[2], 150, after)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_cpu_export_uses_animated_audio_volume_over_clip_time(self):
        audio_path = os.path.join(self.save_root, "volume-keyframes.wav")
        sample_rate = 16_000
        samples = array(
            "h",
            (
                int(5000 * math.sin(2 * math.pi * 440 * index / sample_rate))
                for index in range(sample_rate)
            ),
        )
        with wave.open(audio_path, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(samples.tobytes())

        project = create_editor_project(name="Animated volume check")
        project["canvas"].update({"width": 64, "height": 64, "fps": 2})
        project["export"].update({"encoder": "software", "include_audio": True})
        project["assets"] = {
            "audio": {
                "id": "audio",
                "name": os.path.basename(audio_path),
                "path": audio_path,
                "type": "audio",
                "origin": "output",
                "duration": 1,
                "width": 0,
                "height": 0,
                "has_audio": True,
            }
        }
        project["tracks"][1]["items"] = [{
            "id": "volume-clip",
            "asset_id": "audio",
            "start": 0,
            "duration": 0.5,
            "speed": 2,
            "keyframes": {
                "volume": [
                    {"time": 0, "value": 0, "easing": "linear"},
                    {"time": 0.5, "value": 2, "easing": "linear"},
                ]
            },
        }]
        output_path = os.path.join(self.temp.name, "animated-volume.mp4")

        ffmpeg = self._render_cpu_project(project, output_path)
        decoded = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                output_path,
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(
            decoded.returncode,
            0,
            decoded.stderr.decode("utf-8", errors="replace")[-2000:],
        )
        audio_samples = array("f")
        audio_samples.frombytes(decoded.stdout)
        self.assertGreater(len(audio_samples), sample_rate // 4)
        earlier = audio_samples[sample_rate // 20:sample_rate * 3 // 20]
        later = audio_samples[sample_rate * 7 // 20:sample_rate * 9 // 20]
        early_rms = math.sqrt(sum(value * value for value in earlier) / len(earlier))
        late_rms = math.sqrt(sum(value * value for value in later) / len(later))
        self.assertGreater(late_rms, early_rms * 2, (early_rms, late_rms))

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

    def test_legacy_text_rotation_is_preserved_as_metadata_and_new_rotation_is_rejected(self):
        cases = [
            {"transform": {"rotation": 15}},
            {"style": {"rotation": -15}},
            {
                "keyframes": {
                    "rotation": [
                        {"time": 0, "value": 0, "easing": "linear"},
                        {"time": 1, "value": 45, "easing": "linear"},
                    ]
                }
            },
        ]
        for extra in cases[:2]:
            with self.subTest(legacy=extra):
                project = create_editor_project(name="Legacy text rotation")
                project["schema_version"] = 5
                project["tracks"][2]["items"] = [{
                    "id": "title",
                    "text": "Title",
                    "start": 0,
                    "duration": 1,
                    **extra,
                }]

                normalized = normalize_editor_project(project)
                item = normalized["tracks"][2]["items"][0]

                self.assertEqual(normalized["schema_version"], 6)
                self.assertEqual(item["transform"]["rotation"], 0)
                self.assertTrue(item["legacy_text_rotation"])
                self.assertIn("Legacy text rotation was reset", normalized["compatibility_warnings"][0])
                renormalized = normalize_editor_project(normalized)
                self.assertEqual(
                    renormalized["tracks"][2]["items"][0]["legacy_text_rotation"],
                    item["legacy_text_rotation"],
                )
                self.assertEqual(
                    renormalized["compatibility_warnings"],
                    normalized["compatibility_warnings"],
                )

        for extra in cases:
            with self.subTest(current=extra):
                project = create_editor_project(name="Unsupported current text rotation")
                project["tracks"][2]["items"] = [{
                    "id": "title",
                    "text": "Title",
                    "start": 0,
                    "duration": 1,
                    **extra,
                }]
                with self.assertRaisesRegex(
                    EditorProjectError,
                    "text rotation|keyframes.rotation is not supported",
                ):
                    normalize_editor_project(project)

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
            "keyframes": {
                "scale": [
                    {"time": 0, "value": 1, "easing": "linear"},
                    {"time": 5, "value": 2, "easing": "linear"},
                ],
            },
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
        self.assertIn("fontsize='64*(if(", graph)
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

    def test_legacy_schema_normalizes_fades_title_style_and_track_volume(self):
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

        self.assertEqual(normalized["schema_version"], 6)
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

    def test_legacy_schema_normalizes_markers_links_takes_and_ai_history(self):
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

        self.assertEqual(normalized["schema_version"], 6)
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
            "spatial_upsampling": "flashvsr2",
            "film_grain_intensity": 1.5,
            "film_grain_saturation": -0.25,
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
        self.assertEqual(normalized["schema_version"], 6)
        self.assertEqual(normalized["export"]["encoder"], "auto")
        self.assertEqual(normalized["export"]["spatial_upsampling"], "flashvsr2")
        self.assertEqual(normalized["export"]["film_grain_intensity"], 1.0)
        self.assertEqual(normalized["export"]["film_grain_saturation"], 0.0)
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
        mobile_filter = proxy_command[proxy_command.index("-vf") + 1]
        self.assertNotIn("fps=", mobile_filter)
        self.assertIn("scale=854:480", mobile_filter)
        self.assertIn("main", proxy_command)
        proxy_path = resolve_editor_media_cache_file(
            self.save_root,
            "default",
            preview["preview_id"],
            "proxy-mobile",
        )
        self.assertTrue(os.path.isfile(proxy_path))

    def test_mobile_editor_keeps_gallery_video_decoders_out_of_playback(self):
        root = os.path.abspath(os.path.join(_HERE, ".."))
        with open(
            os.path.join(root, "ui", "src", "editor", "EditorMediaBin.tsx"),
            encoding="utf-8",
        ) as handle:
            media_bin = handle.read()
        with open(
            os.path.join(root, "ui", "src", "editor", "EditorPreview.tsx"),
            encoding="utf-8",
        ) as handle:
            preview = handle.read()
        with open(
            os.path.join(root, "ui", "src", "editor", "editorUtils.ts"),
            encoding="utf-8",
        ) as handle:
            editor_utils = handle.read()
        with open(os.path.join(root, "app", "launch.py"), encoding="utf-8") as handle:
            launch_source = handle.read()

        self.assertIn("LazyVideoThumbnail", media_bin)
        self.assertIn("useEditorMediaPreview", media_bin)
        self.assertNotIn('<video src={asset.url}', media_bin)
        self.assertIn("isMobile ? 10 : 30", preview)
        self.assertIn("playbackDriftTolerance", preview)
        self.assertIn("preparedEditorMediaItems", preview)
        self.assertIn("data-editor-media-active", preview)
        self.assertIn("data-editor-master-clock", preview)
        self.assertIn("mediaTimelineTime", preview)
        self.assertIn("mediaTime ?? latestPlayhead", preview)
        self.assertIn("mobilePlayback && playing", preview)
        self.assertIn("waitingForMobileProxy", preview)
        self.assertIn("src={sourceUrl || undefined}", preview)
        self.assertIn("Boolean(sourceState.url)", preview)
        self.assertIn("video-deck-${trackItemIndex % 2}", preview)
        self.assertIn("audio-deck-${trackItemIndex % 2}", preview)
        self.assertIn("video[data-editor-media-active]", preview)
        self.assertIn("Prime both alternating decks", preview)
        self.assertIn("playbackBoundaries", preview)
        self.assertIn("crossedBoundary || now - lastPublishedAt", preview)
        self.assertIn("playheadRef.current = next", preview)
        self.assertIn("Promote the already-buffered deck without an unconditional seek", preview)
        self.assertIn("promotionSyncPendingRef", preview)
        self.assertIn("onPlaying={() =>", preview)
        self.assertIn("syncActiveVideo(true)", preview)
        self.assertIn("native media clocks run continuously", preview)
        self.assertIn("onCanPlay={() => syncActiveVideo(false)}", preview)
        self.assertIn("Keep the active media plus the next clip", editor_utils)
        self.assertIn(
            'headers={"Cache-Control": "public, max-age=86400, immutable"}',
            launch_source,
        )

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
        with patch("services.editor_projects.subprocess.run", return_value=completed) as run:
            result = editor_export_capabilities("mock-ffmpeg")
        editor_export_capabilities.cache_clear()
        self.assertTrue(result["encoders"]["nvidia"])
        self.assertTrue(result["encoders"]["intel"])
        self.assertFalse(result["encoders"]["apple"])
        self.assertEqual(result["recommended"], "nvidia")
        self.assertEqual(run.call_count, 5)
        first_probe = run.call_args_list[1].args[0]
        self.assertEqual(first_probe[first_probe.index("-c:v") + 1], "h264_nvenc")

    def test_export_capabilities_require_each_codec_runtime_probe(self):
        listing = Mock(
            returncode=0,
            stdout="h264_nvenc hevc_nvenc h264_qsv hevc_qsv",
            stderr="",
        )
        success = Mock(returncode=0, stdout="", stderr="")
        unavailable = Mock(returncode=1, stdout="", stderr="device unavailable")
        editor_export_capabilities.cache_clear()
        with patch(
            "services.editor_projects.subprocess.run",
            side_effect=[listing, success, unavailable, success, success],
        ):
            result = editor_export_capabilities("probe-each-codec")
        editor_export_capabilities.cache_clear()

        self.assertFalse(result["encoders"]["nvidia"])
        self.assertTrue(result["encoders"]["intel"])
        self.assertEqual(result["recommended"], "intel")

    def test_render_retries_only_for_hardware_initialization_errors(self):
        class CompletedProcess:
            def __init__(self, command, stderr_text, returncode):
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO(stderr_text)
                self.returncode = returncode
                if returncode == 0:
                    with open(command[-1], "wb") as output:
                        output.write(b"rendered")

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        project = create_editor_project(name="Narrow fallback")
        project["tracks"][2]["items"] = [{
            "id": "title",
            "text": "Title",
            "start": 0,
            "duration": 1,
        }]
        capabilities = {
            "encoders": {
                "software": True,
                "nvidia": True,
                "intel": False,
                "apple": False,
            },
            "recommended": "nvidia",
        }

        def fake_compile(_project, **kwargs):
            backend = (
                "hardware"
                if kwargs["encoder_capabilities"]["encoders"]["nvidia"]
                else "software"
            )
            return ["mock-ffmpeg", backend, kwargs["output_path"]], 1.0

        hardware_failure_processes = []

        def fallback_popen(command, **_kwargs):
            process = CompletedProcess(
                command,
                "No NVENC capable devices found" if command[1] == "hardware" else "",
                1 if command[1] == "hardware" else 0,
            )
            hardware_failure_processes.append(command[1])
            return process

        with (
            patch("services.editor_projects.editor_export_capabilities", return_value=capabilities),
            patch("services.editor_projects.compile_editor_render", side_effect=fake_compile),
            patch("services.editor_projects.subprocess.Popen", side_effect=fallback_popen),
        ):
            output = render_editor_project(
                project,
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
                ffmpeg="mock-ffmpeg",
            )
        self.assertTrue(os.path.isfile(output))
        self.assertEqual(hardware_failure_processes, ["hardware", "software"])

        unrelated_project = create_editor_project(name="No broad retry")
        unrelated_project["tracks"][2]["items"] = [{
            "id": "title",
            "text": "Title",
            "start": 0,
            "duration": 1,
        }]
        attempted: list[str] = []

        def unrelated_popen(command, **_kwargs):
            attempted.append(command[1])
            return CompletedProcess(command, "Invalid filter graph", 1)

        with (
            patch("services.editor_projects.editor_export_capabilities", return_value=capabilities),
            patch("services.editor_projects.compile_editor_render", side_effect=fake_compile),
            patch("services.editor_projects.subprocess.Popen", side_effect=unrelated_popen),
            self.assertRaisesRegex(RuntimeError, "Invalid filter graph"),
        ):
            render_editor_project(
                unrelated_project,
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
                ffmpeg="mock-ffmpeg",
            )
        self.assertEqual(attempted, ["hardware"])

    def test_render_cancellation_is_responsive_while_stdout_is_stalled(self):
        import threading as threading_module

        release_stdout = threading_module.Event()
        created = []

        class StalledStream:
            def __iter__(self):
                release_stdout.wait(timeout=5)
                return iter(())

        class StalledProcess:
            def __init__(self):
                self.stdout = StalledStream()
                self.stderr = io.StringIO("")
                self.returncode = None
                self.terminated = False
                created.append(self)

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15
                release_stdout.set()

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9
                release_stdout.set()

        project = create_editor_project(name="Cancel stalled export")
        project["tracks"][2]["items"] = [{
            "id": "title",
            "text": "Title",
            "start": 0,
            "duration": 1,
        }]
        capabilities = {
            "encoders": {"software": True, "nvidia": False, "intel": False, "apple": False},
            "recommended": "software",
        }
        with (
            patch("services.editor_projects.editor_export_capabilities", return_value=capabilities),
            patch(
                "services.editor_projects.compile_editor_render",
                return_value=(["mock-ffmpeg", "software", "unused.mp4"], 1.0),
            ),
            patch("services.editor_projects.subprocess.Popen", side_effect=lambda *_a, **_k: StalledProcess()),
            self.assertRaises(EditorRenderCancelled),
        ):
            render_editor_project(
                project,
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
                ffmpeg="mock-ffmpeg",
                cancel_check=lambda: True,
            )
        self.assertTrue(created[0].terminated)


if __name__ == "__main__":
    unittest.main()
