"""Resumable asset storage and backend-path ownership tests."""

from pathlib import Path
import hashlib
import os
import tempfile
import unittest

import sys

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.assets import AssetStore, AssetStoreError  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AssetStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = self.root / "settings"
        self.app_root = self.root / "app"
        self.uploads = self.app_root / "uploads"
        self.outputs = self.app_root / "outputs"
        self.storage = self.app_root / "storage"
        for directory in (self.uploads, self.outputs, self.storage):
            directory.mkdir(parents=True)
        self.now = 1000.0
        self.store = AssetStore(
            self.settings, self.app_root, clock=lambda: self.now
        )

    def begin(self, data: bytes, *, principal="actor-a", filename="clip.wav", store=None):
        return (store or self.store).begin(principal, filename, len(data), _sha(data))

    def test_chunked_upload_finishes_with_verified_bytes_and_public_metadata(self):
        data = b"first-second-third"
        upload = self.begin(data)
        self.assertEqual(upload["state"], "receiving")
        self.assertEqual(upload["received"], 0)
        self.assertEqual(upload["total"], len(data))
        self.assertRegex(upload["reference"], r"^asset:[0-9a-f-]{36}$")

        first = self.store.write_chunk("actor-a", upload["upload_id"], 0, b"first-", len(data))
        self.assertEqual(first["received"], 6)
        retried = self.store.write_chunk("actor-a", upload["upload_id"], 0, b"first-", len(data))
        self.assertEqual(retried["received"], 6)
        self.store.write_chunk("actor-a", upload["upload_id"], 6, b"second-", len(data))
        self.store.write_chunk("actor-a", upload["upload_id"], 13, b"third", len(data))

        record, staging_path = self.store.finish("actor-a", upload["upload_id"])
        self.assertEqual(record["state"], "ready")
        self.assertEqual(record["received"], len(data))
        self.assertEqual(staging_path.read_bytes(), data)
        self.assertEqual(_sha(staging_path.read_bytes()), record["sha256"])
        status = self.store.status("actor-a", upload["upload_id"])
        self.assertEqual(
            self.store.status("actor-a", upload["asset_id"])["upload_id"],
            upload["upload_id"],
        )
        self.assertNotIn("backend_path", status)
        self.assertNotIn(str(staging_path), str(status))

    def test_offsets_totals_chunk_bounds_and_conflicting_retries_are_rejected(self):
        store = AssetStore(
            self.settings, self.app_root, clock=lambda: self.now,
            max_chunk_bytes=4, max_file_bytes=16,
        )
        data = b"abcdefgh"
        upload = self.begin(data, store=store)
        upload_id = upload["upload_id"]
        with self.assertRaises(AssetStoreError) as skip:
            store.write_chunk("actor-a", upload_id, 1, b"abc", len(data))
        self.assertEqual(skip.exception.status, 409)
        with self.assertRaises(AssetStoreError) as total:
            store.write_chunk("actor-a", upload_id, 0, b"abcd", len(data) + 1)
        self.assertEqual(total.exception.status, 409)
        with self.assertRaises(AssetStoreError) as too_large:
            store.write_chunk("actor-a", upload_id, 0, b"abcde", len(data))
        self.assertEqual(too_large.exception.status, 413)

        store.write_chunk("actor-a", upload_id, 0, b"abcd", len(data))
        with self.assertRaises(AssetStoreError) as different:
            store.write_chunk("actor-a", upload_id, 0, b"abce", len(data))
        self.assertEqual(different.exception.status, 409)
        store.write_chunk("actor-a", upload_id, 0, b"abcd", len(data))
        store.write_chunk("actor-a", upload_id, 4, b"efgh", len(data))
        _, path = store.finish("actor-a", upload_id)
        self.assertEqual(path.read_bytes(), data)

    def test_restart_truncates_uncommitted_tail_and_preserves_committed_offset(self):
        data = b"abcdef"
        upload = self.begin(data)
        upload_id = upload["upload_id"]
        self.store.write_chunk("actor-a", upload_id, 0, b"abc", len(data))
        stage_path = self.store.staging_directory / f"{upload_id}.part"
        with stage_path.open("ab") as stream:
            stream.write(b"uncommitted-tail")
            stream.flush()
            os.fsync(stream.fileno())

        restarted = AssetStore(self.settings, self.app_root, clock=lambda: self.now)
        self.assertEqual(stage_path.stat().st_size, 3)
        self.assertEqual(restarted.status("actor-a", upload_id)["received"], 3)
        restarted.write_chunk("actor-a", upload_id, 3, b"def", len(data))
        _, completed_path = restarted.finish("actor-a", upload_id)
        self.assertEqual(completed_path.read_bytes(), data)

    def test_hash_mismatch_marks_upload_failed(self):
        upload = self.store.begin("actor-a", "wrong.bin", 4, _sha(b"good"))
        self.store.write_chunk("actor-a", upload["upload_id"], 0, b"baad", 4)
        with self.assertRaises(AssetStoreError) as mismatch:
            self.store.finish("actor-a", upload["upload_id"])
        self.assertEqual(mismatch.exception.status, 422)
        self.assertEqual(self.store.status("actor-a", upload["upload_id"])["state"], "failed")

    def test_filename_validation_blocks_traversal_windows_names_and_invalid_forms(self):
        data = b"x"
        for filename in (
            "../secret.txt", "..\\secret.txt", "folder/name.wav", "CON", "con.txt",
            "NUL.tar.gz", "COM1.log", "LPT9", "foo.", "foo ", "bad:name", "bad|name",
            "bad\nname",
        ):
            with self.subTest(filename=filename):
                with self.assertRaises(AssetStoreError):
                    self.begin(data, filename=filename)
        good = self.begin(data, filename="take 01.wav")
        self.assertEqual(good["filename"], "take 01.wav")

    def test_size_digest_and_quota_limits_are_enforced(self):
        store = AssetStore(
            self.settings, self.app_root, clock=lambda: self.now,
            max_file_bytes=8, max_pending_per_principal=1, max_pending_global=2,
        )
        with self.assertRaises(AssetStoreError) as too_big:
            store.begin("actor-a", "large.bin", 9, _sha(b"x"))
        self.assertEqual(too_big.exception.status, 413)
        with self.assertRaises(AssetStoreError):
            store.begin("actor-a", "bad-hash.bin", 1, "not-a-hash")

        first = self.begin(b"a", store=store)
        with self.assertRaises(AssetStoreError) as owner_quota:
            self.begin(b"b", store=store)
        self.assertEqual(owner_quota.exception.status, 429)
        second = self.begin(b"b", principal="actor-b", store=store)
        with self.assertRaises(AssetStoreError) as global_quota:
            self.begin(b"c", principal="actor-c", store=store)
        self.assertEqual(global_quota.exception.status, 429)
        self.assertNotEqual(first["upload_id"], second["upload_id"])

        byte_store = AssetStore(
            self.root / "byte-settings", self.app_root, clock=lambda: self.now,
            max_file_bytes=8,
            max_pending_bytes_per_principal=2,
            max_pending_bytes_global=3,
        )
        byte_store.begin("actor-a", "two.bin", 2, _sha(b"ab"))
        with self.assertRaises(AssetStoreError) as principal_bytes:
            byte_store.begin("actor-a", "one.bin", 1, _sha(b"c"))
        self.assertEqual(principal_bytes.exception.status, 429)
        with self.assertRaises(AssetStoreError) as global_bytes:
            byte_store.begin("actor-b", "two.bin", 2, _sha(b"de"))
        self.assertEqual(global_bytes.exception.status, 429)

    def test_zero_byte_asset_can_be_verified_without_chunks(self):
        upload = self.store.begin("actor-a", "empty.dat", 0, _sha(b""))
        record, path = self.store.finish("actor-a", upload["upload_id"])
        self.assertEqual(record["state"], "ready")
        self.assertEqual(record["received"], 0)
        self.assertEqual(path.read_bytes(), b"")

    def test_foreign_upload_and_asset_ids_are_not_visible(self):
        data = b"sample"
        upload = self.begin(data)
        with self.assertRaises(AssetStoreError) as status:
            self.store.status("actor-b", upload["upload_id"])
        self.assertEqual(status.exception.status, 404)
        with self.assertRaises(AssetStoreError) as chunk:
            self.store.write_chunk("actor-b", upload["upload_id"], 0, data)
        self.assertEqual(chunk.exception.status, 404)
        with self.assertRaises(AssetStoreError) as unknown:
            self.store.resolve("actor-a", "asset:00000000-0000-0000-0000-000000000000")
        self.assertEqual(unknown.exception.status, 404)
        with self.assertRaises(AssetStoreError):
            self.store.resolve("actor-a", str(self.uploads / "sample.wav"))

    def _ready(self, data=b"asset-bytes", principal="actor-a", store=None):
        store = store or self.store
        upload = self.begin(data, principal=principal, store=store)
        store.write_chunk(principal, upload["upload_id"], 0, data, len(data))
        record, staging = store.finish(principal, upload["upload_id"])
        return upload, record, staging

    def test_register_resolve_and_nested_references_use_only_owned_backend_files(self):
        data = b"trusted backend import"
        upload, _, staging = self._ready(data)
        backend_path = self.uploads / "registered.bin"
        backend_path.write_bytes(data)
        reference = self.store.register("actor-a", upload["upload_id"], backend_path)
        self.assertEqual(reference, upload["reference"])
        self.assertEqual(self.store.resolve("actor-a", reference), backend_path.resolve())
        self.assertEqual(self.store.reference_for_path("actor-a", backend_path.resolve()), reference)
        self.assertIsNone(self.store.reference_for_path("actor-b", backend_path.resolve()))
        self.assertFalse(staging.exists())
        self.assertNotIn("backend_path", self.store.status("actor-a", upload["upload_id"]))

        payload = {
            "audio": reference,
            "nested": [{"image": reference.replace("asset:", "maestro-asset:")}],
            "text": "asset IDs are exact values",
        }
        resolved = self.store.resolve_tree("actor-a", payload)
        self.assertEqual(resolved["audio"], str(backend_path.resolve()))
        self.assertEqual(resolved["nested"][0]["image"], str(backend_path.resolve()))
        self.assertEqual(resolved["text"], payload["text"])
        self.assertEqual(payload["audio"], reference)

        with self.assertRaises(AssetStoreError) as foreign:
            self.store.resolve("actor-b", reference)
        self.assertEqual(foreign.exception.status, 404)
        with self.assertRaises(AssetStoreError) as missing_tree:
            self.store.resolve_tree("actor-a", {"audio": "asset:00000000-0000-0000-0000-000000000000"})
        self.assertEqual(missing_tree.exception.status, 404)

        # A committed-chunk retry remains safe after staging cleanup because
        # the exact chunk length and SHA-256 are durable in SQLite.
        retried = self.store.write_chunk("actor-a", upload["upload_id"], 0, data, len(data))
        self.assertEqual(retried["state"], "imported")

    def test_register_rejects_outside_paths_and_allows_only_safe_symlink_roots(self):
        data = b"registered"
        upload, _, _ = self._ready(data)
        outside = self.root / "outside.bin"
        outside.write_bytes(data)
        with self.assertRaises(AssetStoreError) as external:
            self.store.register("actor-a", upload["upload_id"], outside)
        self.assertEqual(external.exception.status, 403)
        with self.assertRaises(AssetStoreError) as traversal:
            self.store.register("actor-a", upload["upload_id"], self.uploads / ".." / "outside.bin")
        self.assertEqual(traversal.exception.status, 400)

        # A storage root itself is an approved symlink location. A nested
        # symlink that escapes an ordinary root must still be rejected.
        external_store = self.root / "external-store"
        external_store.mkdir()
        self.storage.rmdir()
        try:
            os.symlink(external_store, self.storage, target_is_directory=True)
        except OSError:
            self.storage.mkdir()
        else:
            linked_file = self.storage / "backend.bin"
            linked_file.write_bytes(data)
            self.store.register("actor-a", upload["upload_id"], linked_file)

        escape = self.uploads / "escape"
        try:
            os.symlink(self.root, escape, target_is_directory=True)
        except OSError:
            return
        (self.root / "escaped.bin").write_bytes(data)
        another, _, _ = self._ready(b"other bytes")
        with self.assertRaises(AssetStoreError) as escaped:
            self.store.register("actor-a", another["upload_id"], escape / "escaped.bin")
        self.assertEqual(escaped.exception.status, 403)

    def test_quota_releases_after_register_and_only_explicit_cleanup_expires_uploads(self):
        store = AssetStore(
            self.settings, self.app_root, clock=lambda: self.now,
            max_pending_per_principal=1, max_pending_global=2,
        )
        upload, _, stage = self._ready(b"one", store=store)
        with self.assertRaises(AssetStoreError):
            self.begin(b"two", store=store)
        backend_path = self.uploads / "one.bin"
        backend_path.write_bytes(b"one")
        store.register("actor-a", upload["upload_id"], backend_path)
        second = self.begin(b"two", store=store)
        self.assertEqual(second["state"], "receiving")

        self.now += 1000
        restarted = AssetStore(self.settings, self.app_root, clock=lambda: self.now)
        second_stage = restarted.staging_directory / f"{second['upload_id']}.part"
        self.assertTrue(second_stage.exists(), "restart alone does not expire unfinished uploads")
        self.assertEqual(restarted.cleanup_expired(max_age_seconds=100), 1)
        self.assertFalse(second_stage.exists())
        self.assertEqual(restarted.status("actor-a", second["upload_id"])["state"], "expired")
        self.assertFalse(stage.exists())
        self.assertEqual(backend_path.read_bytes(), b"one")


if __name__ == "__main__":
    unittest.main()
