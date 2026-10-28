"""Offline safety and round-trip tests for Editor project archives."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
import warnings
import zipfile
from unittest.mock import patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.editor_archives import (  # noqa: E402
    EditorProjectArchiveCancelled,
    EditorProjectArchiveError,
    export_editor_project_archive,
    import_editor_project_archive,
)
from services.editor_projects import create_editor_project, load_editor_project  # noqa: E402
import services.editor_archives as editor_archives  # noqa: E402


class TestEditorProjectArchives(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.save_root = os.path.join(self.temp.name, "outputs")
        self.uploads_root = os.path.join(self.temp.name, "uploads")
        self.import_root = os.path.join(self.temp.name, "imported")
        os.makedirs(self.save_root)
        os.makedirs(self.uploads_root)

    def tearDown(self):
        self.temp.cleanup()

    def _write_source(self, name: str, data: bytes) -> str:
        path = os.path.join(self.save_root, name)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    @staticmethod
    def _sha256(path: str) -> str:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def _project(self, *, missing: bool = False) -> dict:
        active_path = self._write_source("active-source.mp4", b"active-source-bytes")
        alternate_path = self._write_source("alternate-source.mp4", b"alternate-source-bytes")
        unused_path = self._write_source("unreferenced-source.mp4", b"unused-bytes")
        project = create_editor_project(name="Portable edit", workspace="default")
        project["assets"] = {
            "active": {
                "id": "active",
                "name": "active-source.mp4",
                "path": active_path if not missing else os.path.join(self.save_root, "gone.mp4"),
                "type": "video",
                "origin": "output",
                "duration": 4,
                "width": 1280,
                "height": 720,
                "fps": 30,
                "has_audio": True,
            },
            "alternate": {
                "id": "alternate",
                "name": "alternate-source.mp4",
                "path": alternate_path,
                "type": "video",
                "origin": "output",
                "duration": 4,
                "width": 1280,
                "height": 720,
                "fps": 30,
                "has_audio": True,
            },
            "unused": {
                "id": "unused",
                "name": "unreferenced-source.mp4",
                "path": unused_path,
                "type": "video",
                "origin": "output",
                "duration": 4,
                "width": 1280,
                "height": 720,
                "fps": 30,
                "has_audio": True,
            },
        }
        project["tracks"][0]["items"] = [{
            "id": "clip",
            "asset_id": "active",
            "take_asset_ids": ["active", "alternate"],
            "start": 0,
            "duration": 2,
        }]
        return project

    def _export(
        self,
        path: str,
        project: dict | None = None,
        *,
        cancel_check=None,
    ) -> str:
        return export_editor_project_archive(
            project or self._project(),
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
            output_path=path,
            cancel_check=cancel_check,
        )

    def test_round_trip_only_active_and_alternate_media_without_host_paths(self):
        project = self._project()
        active_original = project["assets"]["active"]["path"]
        alternate_original = project["assets"]["alternate"]["path"]
        unused_original = project["assets"]["unused"]["path"]
        original_hashes = {
            path: self._sha256(path)
            for path in (active_original, alternate_original, unused_original)
        }
        package = self._export(os.path.join(self.temp.name, "edit.maestro.zip"), project)

        with zipfile.ZipFile(package) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            portable = json.loads(archive.read("project.json"))
            self.assertEqual(names, {
                "project.json",
                "manifest.json",
                manifest["assets"]["active"]["path"],
                manifest["assets"]["alternate"]["path"],
            })
            self.assertEqual(set(portable["assets"]), {"active", "alternate"})
            self.assertNotIn("path", portable["assets"]["active"])
            self.assertNotIn(self.save_root, archive.read("project.json").decode("utf-8"))
            for record in manifest["assets"].values():
                self.assertEqual(archive.getinfo(record["path"]).compress_type, zipfile.ZIP_STORED)
                self.assertEqual(
                    hashlib.sha256(archive.read(record["path"])).hexdigest(),
                    record["sha256"],
                )

        imported = import_editor_project_archive(
            package,
            save_root=self.import_root,
            workspace="ImportedWorkspace",
        )
        self.assertNotEqual(imported["id"], project["id"])
        self.assertEqual(imported["revision"], 1)
        self.assertEqual(imported["workspace"], "ImportedWorkspace")
        self.assertEqual(set(imported["assets"]), {"active", "alternate"})
        imported_dir = os.path.join(
            self.import_root,
            "ImportedWorkspace",
            ".maestro_editor",
            "imports",
            imported["id"],
        )
        for asset_id, asset in imported["assets"].items():
            self.assertTrue(os.path.isfile(asset["path"]))
            self.assertEqual(os.path.commonpath([asset["path"], imported_dir]), imported_dir)
            self.assertEqual(
                self._sha256(asset["path"]),
                original_hashes[active_original if asset_id == "active" else alternate_original],
            )
        self.assertTrue(os.path.isfile(unused_original))
        self.assertEqual(
            {path: self._sha256(path) for path in original_hashes},
            original_hashes,
        )
        loaded = load_editor_project(self.import_root, "ImportedWorkspace", imported["id"])
        self.assertEqual(loaded["assets"]["active"]["path"], imported["assets"]["active"]["path"])

    def test_export_fails_when_referenced_media_is_missing(self):
        project = self._project(missing=True)
        os.remove(os.path.join(self.save_root, "active-source.mp4"))
        with self.assertRaisesRegex(EditorProjectArchiveError, "Cannot package referenced media"):
            self._export(
                os.path.join(self.temp.name, "missing.maestro.zip"),
                project,
            )
        self.assertFalse(os.path.exists(os.path.join(self.temp.name, "missing.maestro.zip")))

    def test_import_rejects_media_hash_mismatch_without_publishing_project(self):
        package = self._export(os.path.join(self.temp.name, "edit.maestro.zip"))
        tampered = os.path.join(self.temp.name, "tampered.maestro.zip")
        with zipfile.ZipFile(package) as source, zipfile.ZipFile(tampered, "w") as target:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename.startswith("media/"):
                    data = bytes([data[0] ^ 1]) + data[1:]
                target.writestr(info, data)
        with self.assertRaisesRegex(EditorProjectArchiveError, "SHA-256"):
            import_editor_project_archive(
                tampered, save_root=self.import_root, workspace="default"
            )
        self.assertFalse(os.path.exists(os.path.join(self.import_root, ".maestro_editor")))

    def test_import_rejects_traversal_symlinks_and_duplicate_names(self):
        attacks: list[tuple[str, list[tuple[zipfile.ZipInfo, bytes]]]] = []
        traversal = zipfile.ZipInfo("../escape")
        attacks.append(("traversal", [(traversal, b"x")]))
        symlink = zipfile.ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        attacks.append(("symlink", [(symlink, b"target")]))
        attacks.append(("duplicate", [
            (zipfile.ZipInfo("project.json"), b"{}"),
            (zipfile.ZipInfo("project.json"), b"{}"),
        ]))
        for label, malicious_entries in attacks:
            with self.subTest(label=label):
                path = os.path.join(self.temp.name, f"{label}.zip")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(path, "w") as archive:
                        for info, contents in malicious_entries:
                            archive.writestr(info, contents)
                with self.assertRaises(EditorProjectArchiveError):
                    import_editor_project_archive(
                        path, save_root=self.import_root, workspace="default"
                    )

    def test_import_rejects_unpacked_size_over_limit_before_extracting(self):
        path = os.path.join(self.temp.name, "over-limit.zip")
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.json", b"x" * 101)
            archive.writestr("manifest.json", b"{}")
        with patch.object(editor_archives, "EDITOR_ARCHIVE_UNPACKED_MAX_BYTES", 100):
            with self.assertRaisesRegex(EditorProjectArchiveError, "unpacked size limit"):
                import_editor_project_archive(
                    path, save_root=self.import_root, workspace="default"
                )
        self.assertFalse(os.path.exists(os.path.join(self.import_root, ".maestro_editor")))

    def test_export_and_import_honor_cancellation_without_publishing(self):
        output_path = os.path.join(self.temp.name, "cancelled.maestro.zip")
        with self.assertRaises(EditorProjectArchiveCancelled):
            self._export(output_path, self._project(), cancel_check=lambda: True)
        # Cancellation is checked while collecting referenced media; package
        # output itself is not created unless the entire atomic write succeeds.
        self.assertFalse(os.path.exists(output_path))

        package = self._export(os.path.join(self.temp.name, "valid.maestro.zip"))
        with self.assertRaises(EditorProjectArchiveCancelled):
            import_editor_project_archive(
                package,
                save_root=self.import_root,
                workspace="default",
                cancel_check=lambda: True,
            )
        self.assertFalse(os.path.exists(os.path.join(self.import_root, ".maestro_editor")))


if __name__ == "__main__":
    unittest.main()
