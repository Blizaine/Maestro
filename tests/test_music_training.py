"""Durable dataset boundaries and training/queue ownership, without loading models."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import music_training, music_styles
from services.job_lifecycle import request_cancel
from services.music_training_runner import MusicTrainingRunner


class MusicProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tracks = []
        for number in range(2):
            path = self.root / f"song-{number}.wav"
            path.write_bytes(b"original recording " + bytes([number]))
            self.tracks.append({"audio_path": str(path), "lyrics": "[Verse]\nThe morning comes",
                                "style": "Warm acoustic pop", "holdout": number == 1})

    def create(self, tracks=None):
        return music_training.create_project("Original music", "My original style", tracks or self.tracks, root=self.root / "projects")

    def test_project_persists_source_identity_without_modifying_recordings(self):
        before = [Path(track["audio_path"]).read_bytes() for track in self.tracks]
        project = self.create()
        restored = music_training.get_project(project["id"], root=self.root / "projects")
        self.assertEqual(project, restored)
        self.assertNotEqual(project["tracks"][0]["audio_sha256"], project["tracks"][1]["audio_sha256"])
        self.assertEqual(before, [Path(track["audio_path"]).read_bytes() for track in self.tracks])
        updated = music_training.update_project(project["id"], root=self.root / "projects", tracks=[], status="prepared")
        self.assertEqual(updated["tracks"], project["tracks"])
        self.assertEqual(updated["status"], "prepared")

    def test_heldout_is_required_and_cannot_duplicate_training_recording(self):
        for tracks in ([{**track, "holdout": False} for track in self.tracks],
                       [self.tracks[0], {**self.tracks[0], "holdout": True}]):
            with self.subTest(tracks=tracks), self.assertRaises(ValueError):
                self.create(tracks)

    def test_project_and_bundle_paths_reject_escape(self):
        for name in ("../other", "C:/outside", "a/b", "..", "", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                music_styles.style_directory(name, self.root)

    def test_training_bounds_reject_fractional_counts_and_nonfinite_values(self):
        for options in ({"steps": 1.5}, {"steps": True}, {"steps": 1601}, {"rank": 3},
                        {"seed": -1}, {"learning_rate": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                music_training.training_options(options)
        options = music_training.training_options({"steps": 400, "rank": 8, "resume": True})
        self.assertEqual(options["checkpoint_every"], 200)
        self.assertTrue(options["resume"])

    def test_bundle_revision_and_checksum_are_enforced(self):
        directory = self.root / "style"
        directory.mkdir()
        manifest = {"id": "style", "version": 1, "architecture": "yue2", "base_revision": music_styles.BASE_REVISION,
                    "tokenizer_revision": music_styles.TOKENIZER_REVISION, "name": "Test music", "trigger": "Original music"}
        for branch in ("ar", "nar"):
            path = directory / f"{branch}.safetensors"
            path.write_bytes(branch.encode())
            manifest[branch] = {"file": path.name, "sha256": music_styles.file_digest(path)}
        (directory / "style.json").write_text(json.dumps(manifest))
        music_styles.load_style("style", root=self.root, verify=True)
        (directory / "ar.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            music_styles.load_style("style", root=self.root, verify=True)
        manifest["tokenizer_revision"] = "wrong dialect"
        (directory / "style.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "revision"):
            music_styles.load_style("style", root=self.root)

    def test_cancelled_queued_preparation_never_unloads_generation_model(self):
        with patch.object(music_training, "PROJECT_ROOT", self.root / "projects"):
            project = self.create()
            jobs, states = {}, {}
            generation_lock = threading.Lock()
            generation_lock.acquire()
            try:
                with patch("threading.Thread.start"):
                    runner = MusicTrainingRunner(jobs, generation_lock, states, lambda: self.fail("Unloaded a running model"), lambda: "")
                    submitted = runner.submit("prepare", project["id"], {})
                job = jobs[submitted["job_id"]]
                request_cancel(job)
                with self.assertRaisesRegex(ValueError, "already"):
                    runner.submit("prepare", project["id"], {})
                runner.run(job["id"])
                self.assertEqual(job["status"], "cancelled")
                self.assertEqual(music_training.get_project(project["id"])["status"], "cancelled")
                self.assertFalse(states)
                self.assertNotIn(project['id'], runner.project_workers)
            finally:
                generation_lock.release()

    def test_malformed_style_manifest_has_useful_error(self):
        directory = self.root / 'style'
        directory.mkdir()
        for manifest in ([], None, {'id': 'style', 'ar': []}):
            (directory / 'style.json').write_text(json.dumps(manifest))
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                music_styles.load_style('style', root=self.root)


if __name__ == "__main__":
    unittest.main()
