"""CPU-only ownership, validation, and persistence tests for Worlds storage."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
import zipfile

from PIL import Image

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from app.services.worlds_storage import (
    WorldsStorage,
    WorldsStorageError,
    _read_world_archive,
    _normalise_probe_result,
)


def _jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), (30, 80, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _manifest(*, frames=4, generations=1, width=832, height=480, world_id=None):
    value = {
        "format": "wgpworld",
        "version": 1,
        "model_type": "hy_nomas",
        "width": width,
        "height": height,
        "fps": 24,
        "steps": 8,
        "world": {"segments": [{} for _ in range(generations)]},
        "generations": [
            {"seed": index, "latent_frames": 1} for index in range(generations)
        ],
        "segments": [
            {
                "latent_start": index,
                "latent_frames": 1,
                "video_frames": frames if index == generations - 1 else 4,
                "generation": index,
                "camera": False,
                "text_offsets": None,
                "action_windows": None,
            }
            for index in range(generations)
        ],
    }
    if world_id is not None:
        value["world_id"] = world_id
    return value


class WorldsStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.outputs = self.base / "app" / "outputs"
        self.job_workspace = self.base / "user-workspace" / "generation"
        self.job_workspace.mkdir(parents=True)
        self.poster = _jpeg()
        self.manifests = {}
        self.probes = {}
        self.decoded = []
        self.ffmpeg_calls = []
        self.clock_value = [1000.0]

        def reader(path):
            manifest, poster = self.manifests[path.read_bytes()]
            return manifest, poster

        def probe(path):
            return self.probes[path.read_bytes()]

        def decode(path):
            self.decoded.append(path.resolve())
            return True

        def ffmpeg_runner(args):
            self.ffmpeg_calls.append(list(args))
            recording = f"recording-{args.count('-i')}".encode()
            Path(args[-1]).write_bytes(recording)
            return True

        def clock():
            self.clock_value[0] += 1
            return self.clock_value[0]

        self.storage = WorldsStorage(
            self.outputs,
            probe_video=probe,
            decode_video=decode,
            read_world_archive=reader,
            ffmpeg_runner=ffmpeg_runner,
            clock=clock,
        )
        self.probes[b"recording-1"] = {"width": 832, "height": 480, "frame_count": 4, "fps": 24}
        self.probes[b"recording-2"] = {"width": 832, "height": 480, "frame_count": 10, "fps": 24}

    def tearDown(self):
        self.temporary.cleanup()

    def _source_files(self, name, video, archive, manifest, probe):
        video_name = f"{name}.mp4"
        archive_name = f"{name}.wgpworld"
        (self.job_workspace / video_name).write_bytes(video)
        (self.job_workspace / archive_name).write_bytes(archive)
        self.manifests[archive] = (manifest, self.poster)
        self.probes[video] = probe
        return video_name

    @staticmethod
    def _ids(request_id, job_id, *, status="completed", client_id=None, outputs=None):
        job = {
            "id": job_id,
            "job_id": job_id,
            "client_submission_id": request_id if client_id is None else client_id,
            "status": status,
            "out_dir": None,
            "output_files": outputs or [],
        }
        receipt = {
            "client_request_id": request_id,
            "client_submission_id": request_id,
            "job_id": job_id,
        }
        return job, receipt

    def _publish(self, *, request_id, job_id, name, operation="create", parent=None,
                 manifest=None, frames=4, video=None, archive=None, job_updates=None):
        video = video or f"video-{name}".encode()
        archive = archive or f"archive-{name}".encode()
        manifest = manifest or _manifest(frames=frames, generations=1 if parent is None else 2)
        output_name = self._source_files(
            name,
            video,
            archive,
            manifest,
            {"width": 832, "height": 480, "frame_count": frames, "fps": 24},
        )
        job, receipt = self._ids(request_id, job_id, outputs=[output_name])
        job["out_dir"] = str(self.job_workspace)
        if job_updates:
            job.update(job_updates)
        request = {"operation": operation}
        if operation == "create":
            request["description"] = f"World {name} opening scene"
        elif operation == "continue":
            request.update({"world_id": parent, "text": f"Continue {name}"})
        else:
            request.update({"world_id": parent, "action": "Forward", "duration_ms": 1000})
        receipt["request"] = request
        return self.storage.publish_completed_generation(job, receipt), job, receipt

    def test_storage_is_fixed_under_outputs_and_never_changes_cwd_or_user_workspace(self):
        cwd = Path.cwd()
        workspace_before = sorted(self.job_workspace.iterdir())
        self.assertEqual(self.storage.root, self.outputs.resolve() / "Worlds")
        self.assertEqual(Path.cwd(), cwd)
        self.assertEqual(sorted(self.job_workspace.iterdir()), workspace_before)

    def test_receipts_are_atomic_and_keyed_by_canonical_uuid(self):
        request_id = "12345678-1234-4234-9234-123456789abc"
        receipt = {"client_request_id": request_id, "state": "queued", "job_id": "job-1"}
        self.storage.write_receipt(request_id, receipt)
        self.assertEqual(self.storage.read_receipt(request_id), receipt)
        files = list(self.storage.receipts_dir.iterdir())
        self.assertEqual([item.name for item in files], [f"{request_id}.json"])
        with self.assertRaises(WorldsStorageError):
            self.storage.read_receipt("../outside")
        with self.assertRaises(WorldsStorageError):
            self.storage.write_receipt(request_id, {"client_request_id": "other"})
        self.assertFalse((self.storage.root / "outside.json").exists())

    def test_ffprobe_json_rate_fields_must_confirm_24_fps(self):
        result = _normalise_probe_result(
            {
                "streams": [
                    {
                        "width": 832,
                        "height": 480,
                        "nb_read_frames": "97",
                        "r_frame_rate": "24/1",
                        "avg_frame_rate": "24/1",
                    }
                ]
            }
        )
        self.assertEqual(result, {"width": 832, "height": 480, "frame_count": 97, "fps": 24})
        with self.assertRaises(WorldsStorageError):
            _normalise_probe_result(
                {
                    "streams": [
                        {
                            "width": 832,
                            "height": 480,
                            "nb_read_frames": "97",
                            "r_frame_rate": "24/1",
                            "avg_frame_rate": "23/1",
                        }
                    ]
                }
            )

    def test_publish_verifies_job_owner_media_manifest_and_is_idempotent(self):
        request_id = "aaaaaaaa-1234-4234-9234-123456789abc"
        result, job, receipt = self._publish(
            request_id=request_id,
            job_id="job-a",
            name="create",
            manifest=_manifest(frames=4, world_id="model-random-id"),
        )
        world = result["world"]
        self.assertTrue(result["verified"] and result["video_verified"] and result["world_file_verified"])
        self.assertEqual(world["world_id"], self.storage._request_world_id(request_id))
        self.assertEqual(world["title"], "World create opening scene")
        self.assertTrue(world["recording_available"])
        self.assertEqual(
            world["video_url"],
            f"/api/v1/file/worlds/{world['world_id']}/clip.mp4?workspace=Worlds",
        )
        self.assertEqual(self.storage.capture_world(world["world_id"]), self.poster)
        sidecar = self.storage.save_world(world["world_id"])
        self.assertEqual(sidecar.read_bytes(), b"archive-create")
        self.assertEqual(self.storage.resolve_world_file(world["world_id"]), sidecar)
        self.assertEqual(len(self.decoded), 1)

        again = self.storage.publish_completed_generation(job, receipt)
        self.assertEqual(again, result)
        self.assertEqual(len(self.decoded), 1)
        mismatched_job = dict(job)
        mismatched_job["id"] = "someone-else"
        mismatched_job["job_id"] = "someone-else"
        with self.assertRaises(WorldsStorageError):
            self.storage.publish_completed_generation(mismatched_job, receipt)
        self.assertEqual(self.storage.get_world(world["world_id"]), world)
        self.assertEqual(self.storage.list_worlds(), [world])
        record = self.storage._read_record(world["world_id"])
        self.assertEqual(record["archive_world_id"], "model-random-id")

    def test_legacy_archive_without_world_id_gets_stable_content_identity(self):
        request_id = "bbbbbbbb-1234-4234-9234-123456789abc"
        result, _job, _receipt = self._publish(
            request_id=request_id,
            job_id="job-legacy",
            name="legacy",
            manifest=_manifest(frames=4),
        )
        record = self.storage._read_record(result["world"]["world_id"])
        self.assertEqual(record["archive_world_id"], "legacy-" + record["world_file_sha256"][:32])

    def test_publish_rejects_mismatched_job_ids_submission_and_noncompleted_status(self):
        request_id = "cccccccc-1234-4234-9234-123456789abc"
        invalid_jobs = [
            {"job_id": "wrong", "id": "wrong", "client_submission_id": request_id, "status": "completed"},
            {"job_id": "job-c", "id": "job-other", "client_submission_id": request_id, "status": "completed"},
            {"job_id": "job-c", "id": "job-c", "client_submission_id": "other", "status": "completed"},
            {"job_id": "job-c", "id": "job-c", "client_submission_id": request_id, "status": "running"},
        ]
        for index, job in enumerate(invalid_jobs):
            with self.subTest(index=index), self.assertRaises(WorldsStorageError):
                self.storage.publish_completed_generation(
                    job,
                    {
                        "client_request_id": request_id,
                        "client_submission_id": request_id,
                        "job_id": "job-c",
                        "request": {"operation": "create", "description": "test"},
                    },
                )
        self.assertEqual(list(self.storage.worlds_dir.iterdir()), [])

    def test_output_paths_must_be_basenames_under_the_frozen_job_directory(self):
        for filename in ("../outside.mp4", "subdir\\outside.mp4", "C:\\outside.mp4"):
            with self.subTest(filename=filename), self.assertRaises(WorldsStorageError):
                self.storage._source_paths(
                    {"out_dir": str(self.job_workspace), "output_files": [filename]}
                )
        video = self.job_workspace / "clip.mp4"
        sidecar = self.job_workspace / "clip.wgpworld"
        video.write_bytes(b"video")
        sidecar.write_bytes(b"world")
        with self.assertRaises(WorldsStorageError):
            self.storage._source_paths(
                {
                    "out_dir": str(self.job_workspace),
                    "output_files": ["clip.mp4", "other.wgpworld"],
                }
            )
        with self.assertRaises(WorldsStorageError):
            self.storage._source_paths(
                {
                    "out_dir": str(self.job_workspace),
                    "output_files": ["clip.mp4", "second.mp4"],
                }
            )

    def test_missing_sidecar_wrong_geometry_frame_count_and_decode_failure_do_not_publish(self):
        request_id = "dddddddd-1234-4234-9234-123456789abc"
        video_name = "bad.mp4"
        (self.job_workspace / video_name).write_bytes(b"bad-video")
        job, receipt = self._ids(request_id, "job-bad", outputs=[video_name])
        job["out_dir"] = str(self.job_workspace)
        receipt["request"] = {"operation": "create", "description": "bad"}
        with self.assertRaises(FileNotFoundError):
            self.storage.publish_completed_generation(job, receipt)

        bad_request = "ffffffff-1234-4234-9234-123456789abc"
        video = b"wrong-count-video"
        archive = b"wrong-count-archive"
        output_name = self._source_files(
            "wrong-count", video, archive, _manifest(frames=5),
            {"width": 832, "height": 480, "frame_count": 4, "fps": 24},
        )
        bad_job, bad_receipt = self._ids(bad_request, "job-frame", outputs=[output_name])
        bad_job["out_dir"] = str(self.job_workspace)
        bad_receipt["request"] = {"operation": "create", "description": "bad frame"}
        with self.assertRaises(WorldsStorageError):
            self.storage.publish_completed_generation(bad_job, bad_receipt)
        self.assertFalse((self.storage.worlds_dir / self.storage._request_world_id(bad_request)).exists())

        geometry_request = "eeeeeeee-1234-4234-9234-123456789abc"
        video = b"wrong-geometry-video"
        archive = b"wrong-geometry-archive"
        output_name = self._source_files(
            "wrong-geometry", video, archive, _manifest(frames=4),
            {"width": 640, "height": 480, "frame_count": 4, "fps": 24},
        )
        geometry_job, geometry_receipt = self._ids(
            geometry_request, "job-geometry", outputs=[output_name]
        )
        geometry_job["out_dir"] = str(self.job_workspace)
        geometry_receipt["request"] = {"operation": "create", "description": "bad geometry"}
        with self.assertRaises(WorldsStorageError):
            self.storage.publish_completed_generation(geometry_job, geometry_receipt)
        self.assertFalse((self.storage.worlds_dir / self.storage._request_world_id(geometry_request)).exists())

        decode_request = "eeeeeeee-1234-4234-9234-123456789abd"
        self.storage._decode_video_cb = lambda _path: False
        video = b"decode-failure-video"
        archive = b"decode-failure-archive"
        output_name = self._source_files(
            "decode-failure", video, archive, _manifest(frames=4),
            {"width": 832, "height": 480, "frame_count": 4, "fps": 24},
        )
        decode_job, decode_receipt = self._ids(
            decode_request, "job-decode", outputs=[output_name]
        )
        decode_job["out_dir"] = str(self.job_workspace)
        decode_receipt["request"] = {"operation": "create", "description": "bad decode"}
        with self.assertRaises(WorldsStorageError):
            self.storage.publish_completed_generation(decode_job, decode_receipt)
        self.assertFalse((self.storage.worlds_dir / self.storage._request_world_id(decode_request)).exists())

    def test_continuation_persists_parent_chain_and_record_uses_cpu_ffmpeg(self):
        first_id = "11111111-1234-4234-9234-123456789abc"
        first, _first_job, _first_receipt = self._publish(
            request_id=first_id,
            job_id="job-root",
            name="root",
            frames=4,
            manifest=_manifest(frames=4, generations=1),
            video=b"root-video",
            archive=b"root-archive",
        )
        parent_id = first["world"]["world_id"]
        second_id = "22222222-1234-4234-9234-123456789abc"
        second, _second_job, _second_receipt = self._publish(
            request_id=second_id,
            job_id="job-child",
            name="child",
            operation="continue",
            parent=parent_id,
            frames=6,
            manifest=_manifest(frames=6, generations=2),
            video=b"child-video",
            archive=b"child-archive",
        )
        child_id = second["world"]["world_id"]
        child_record = self.storage._read_record(child_id)
        self.assertEqual(child_record["parent_world_id"], parent_id)
        self.assertEqual(child_record["parent_chain"], [parent_id])
        self.assertEqual(child_record["title"], first["world"]["title"])

        recording = self.storage.record_world(child_id)
        self.assertTrue(recording.is_file())
        self.assertEqual(recording.parent.name, child_id)
        self.assertEqual(len(self.ffmpeg_calls), 1)
        args = self.ffmpeg_calls[0]
        self.assertEqual(args.count("-i"), 2)
        self.assertIn("libx264", args)
        self.assertIn("-threads", args)
        self.assertIn("1", args)
        self.assertIn("concat=n=2:v=1:a=0[outv]", args[args.index("-filter_complex") + 1])
        self.assertNotIn("-hwaccel", args)
        self.assertEqual(self.storage.record_world(child_id), recording)
        self.assertEqual(len(self.ffmpeg_calls), 1)

    def test_parent_chain_requires_existing_parent_and_record_detects_tampered_clips(self):
        missing_parent_request = "33333333-1234-4234-9234-123456789abc"
        video_name = self._source_files(
            "orphan", b"orphan-video", b"orphan-archive", _manifest(frames=4, generations=2),
            {"width": 832, "height": 480, "frame_count": 4, "fps": 24},
        )
        job, receipt = self._ids(missing_parent_request, "job-orphan", outputs=[video_name])
        job["out_dir"] = str(self.job_workspace)
        receipt["request"] = {
            "operation": "continue",
            "world_id": "99999999999999999999999999999999",
            "text": "orphan",
        }
        with self.assertRaises(FileNotFoundError):
            self.storage.publish_completed_generation(job, receipt)

        first, _job, _receipt = self._publish(
            request_id="44444444-1234-4234-9234-123456789abc",
            job_id="job-tamper",
            name="tamper",
            video=b"original-clip",
            archive=b"tamper-archive",
        )
        world_id = first["world"]["world_id"]
        record = self.storage._read_record(world_id)
        clip_path = self.storage._world_asset(world_id, record["clip_file"], max_bytes=3 << 30)
        clip_path.write_bytes(b"changed-clip")
        with self.assertRaises(WorldsStorageError):
            self.storage.record_world(world_id)
        self.assertEqual(self.ffmpeg_calls, [])

    def test_default_world_reader_validates_manifest_header_and_returns_archive_poster(self):
        manifest = {
            "format": "wgpworld",
            "version": 1,
            "model_type": "hy_nomas",
            "width": 32,
            "height": 32,
            "fps": 24,
            "steps": 8,
            "world": {"segments": [{}]},
            "generations": [{"seed": 1, "latent_frames": 1}],
            "segments": [
                {
                    "latent_start": 0,
                    "latent_frames": 1,
                    "video_frames": 2,
                    "generation": 0,
                    "camera": False,
                    "text_offsets": None,
                    "action_windows": None,
                }
            ],
        }
        header = {
            "latents": {"dtype": "BF16", "shape": [1, 32, 1, 1, 1], "data_offsets": [0, 64]},
            "text_0": {"dtype": "BF16", "shape": [1, 3584], "data_offsets": [64, 7232]},
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        tensor_data = len(header_bytes).to_bytes(8, "little") + header_bytes + bytes(7232)
        archive_path = self.base / "valid.wgpworld"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
            archive.writestr("tensors.safetensors", tensor_data)
            archive.writestr("thumbnail.jpg", self.poster)
            archive.writestr("poster.jpg", self.poster)
        parsed, poster = _read_world_archive(archive_path)
        parsed_again, _poster_again = _read_world_archive(archive_path)
        self.assertEqual(parsed["model_type"], "hy_nomas")
        self.assertEqual(poster, self.poster)
        self.assertTrue(str(parsed["world_id"]).startswith("legacy-"))
        self.assertEqual(parsed["world_id"], parsed_again["world_id"])


if __name__ == "__main__":
    unittest.main()
