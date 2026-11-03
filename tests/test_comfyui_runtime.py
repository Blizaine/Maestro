"""CPU-only ComfyUI service and durable execution regressions."""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest import mock

from PIL import Image

from app.services.comfyui_client import ComfyTransportError
from app.services.comfyui_runtime import ComfyService, ServiceError
from app.services.comfyui_workflows import builtin_templates


def _png_bytes(color=(30, 80, 160)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (5, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _object_info(checkpoints=("installed.safetensors",)):
    return {
        "EmptyImage": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 512}],
                    "height": ["INT", {"default": 512}],
                    "batch_size": ["INT", {"default": 1}],
                    "color": ["INT", {"default": 0}],
                }
            }
        },
        "SaveImage": {
            "input": {
                "required": {
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                    "images": ["IMAGE", {}],
                }
            }
        },
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [list(checkpoints), {}]}}
        },
    }


def _history_row(*, completed=True, outputs=None, prompt=None):
    return {
        "prompt": prompt or {},
        "outputs": outputs if outputs is not None else {
            "2": {"images": [{"filename": "result.png", "subfolder": "", "type": "output"}]}
        },
        "status": {"status_str": "success" if completed else "running", "completed": completed},
    }


class FakeComfyClient:
    def __init__(self, endpoint="http://127.0.0.1:8188"):
        self.endpoint = endpoint
        self.info = _object_info()
        self.stats = {"system": {"device": "cpu"}, "token": "secret-value"}
        self.submit_response = {"prompt_id": "prompt-1", "number": 1}
        self.submit_error = None
        self.submit_calls = []
        self.object_info_calls = 0
        self.history_calls = []
        self.history_all = {}
        self.history_by_id = {}
        self.queue_calls = 0
        self.queue_default = {"queue_running": [], "queue_pending": []}
        self.deleted_ids = []
        self.interrupt_calls = 0
        self.free_calls = 0
        self.release_memory_calls = 0
        self.output_calls = []
        self.output_bytes = _png_bytes()
        self.upload_calls = []

    def system_stats(self):
        return deepcopy(self.stats)

    def object_info(self):
        self.object_info_calls += 1
        return deepcopy(self.info)

    def queue(self):
        self.queue_calls += 1
        if self.deleted_ids:
            deleted = set(self.deleted_ids)
            return {
                "queue_running": [
                    entry for entry in self.queue_default.get("queue_running", [])
                    if len(entry) < 2 or str(entry[1]) not in deleted
                ],
                "queue_pending": [
                    entry for entry in self.queue_default.get("queue_pending", [])
                    if len(entry) < 2 or str(entry[1]) not in deleted
                ],
            }
        return deepcopy(self.queue_default)

    def history(self, prompt_id=None):
        self.history_calls.append(prompt_id)
        if prompt_id is None:
            return deepcopy(self.history_all)
        if prompt_id in self.history_by_id:
            return {prompt_id: deepcopy(self.history_by_id[prompt_id])}
        return {}

    def submit(self, graph, *, client_id, submission_id):
        self.submit_calls.append({
            "graph": deepcopy(graph),
            "client_id": client_id,
            "submission_id": submission_id,
        })
        if self.submit_error is not None:
            raise self.submit_error
        return deepcopy(self.submit_response)

    def upload_image(self, data, filename):
        self.upload_calls.append((data, filename))
        return {"name": filename, "subfolder": "", "type": "input"}

    def output(self, filename, subfolder="", type="output"):
        self.output_calls.append({"filename": filename, "subfolder": subfolder, "type": type})
        return self.output_bytes

    def delete_queued(self, prompt_id):
        self.deleted_ids.append(prompt_id)
        return {"deleted": [prompt_id]}

    def interrupt(self):
        self.interrupt_calls += 1

    def free(self, **kwargs):
        self.free_calls += 1

    def release_memory(self):
        self.release_memory_calls += 1


class ComfyRuntimeTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.app_root = Path(self.temporary.name) / "app"
        self.client = FakeComfyClient()
        self.access_checks = []
        self.service = self._new_service()
        self.service.configure({"enabled": True})

    def _new_service(self, *, require_job_access=None, principal_id=None, resume=None, job_exists=None, is_owner=None, submit=None):
        return ComfyService(
            self.app_root,
            submit=submit,
            client_factory=lambda endpoint: self.client,
            require_job_access=require_job_access or (lambda job_id: self.access_checks.append(job_id)),
            principal_id=principal_id or (lambda: "owner-1"),
            resume=resume,
            job_exists=job_exists,
            is_owner=is_owner,
        )

    def _prepare(self, *, inputs=None, workflow_id="cpu_empty_image_smoke", service=None):
        service = service or self.service
        submission = str(uuid.uuid4())
        prepared = service.prepare(
            workflow_id,
            {"client_submission_id": submission, "inputs": inputs or {}, "workspace": "default"},
        )
        receipt, created = service.reserve(prepared, f"job-{submission[:8]}")
        self.assertTrue(created)
        return prepared, receipt

    def test_connection_discovery_is_explicit_bounded_and_does_not_change_settings(self):
        contacted = []
        good = self.client
        def factory(endpoint, **options):
            contacted.append((endpoint, options))
            if endpoint.endswith(":8189"):
                raise ComfyTransportError("connect", ambiguous=False)
            return good
        service = ComfyService(self.app_root, client_factory=factory)
        before = service._config()
        result = service.discover_connections()
        self.assertEqual([row[0] for row in contacted], ["http://127.0.0.1:8188", "http://127.0.0.1:8189"])
        self.assertTrue(all(row[1]["timeout"] == 3 for row in contacted))
        self.assertTrue(result["candidates"][0]["connected"])
        self.assertFalse(result["candidates"][1]["connected"])
        self.assertEqual(service._config(), before)
        self.assertEqual(good.submit_calls, [])

    def test_workflow_options_use_current_schema_without_scanning_or_submission(self):
        result = self.service.workflow_options("sdxl_checkpoint_txt2img")
        self.assertEqual(result["choices"], {"checkpoint": ["installed.safetensors"]})
        self.assertEqual(result["workflow_id"], "sdxl_checkpoint_txt2img")
        self.assertEqual(self.client.object_info_calls, 1)
        self.assertEqual(self.client.submit_calls, [])
        self.service.configure({"enabled": False})
        with self.assertRaisesRegex(ServiceError, "Enable ComfyUI"):
            self.service.workflow_options("sdxl_checkpoint_txt2img")

    def _run(self, prepared, *, cancelled=lambda: False, resolve_media=None, out_dir=None):
        resolve_media = resolve_media or (lambda _reference, _workspace: self.fail("unexpected media resolution"))
        out_dir = Path(out_dir or (Path(self.temporary.name) / "outputs"))
        return self.service.execute(
            prepared,
            out_dir,
            cancelled=cancelled,
            progress=lambda _message: None,
            resolve_media=resolve_media,
        )

    def _complete(self, prompt_id="prompt-1", *, outputs=None, prompt=None):
        self.client.history_by_id[prompt_id] = _history_row(outputs=outputs, prompt=prompt)


class TestComfyRuntimeSettingsAndWorkflows(ComfyRuntimeTestCase):
    def test_settings_connection_and_workflow_crud(self):
        status = self.service.status()
        self.assertTrue(status["enabled"])
        self.assertFalse(status["connected"])
        with self.assertRaises(ServiceError):
            self.service.configure({"enabled": "yes"})

        connection = self.service.test_connection()
        self.assertTrue(connection["connected"])
        self.assertEqual(connection["system"]["token"], "[redacted]")
        self.assertIsNone(connection["connection_error"])

        template = next(row for row in builtin_templates() if row["id"] == "cpu_empty_image_smoke")
        saved = self.service.save_workflow({
            "name": "Smoke copy",
            "kind": "image",
            "description": "Saved test workflow",
            "workflow": template["workflow"],
            "bindings": template["bindings"],
        })
        listed = self.service.workflows()
        self.assertTrue(any(row["id"] == saved["id"] for row in listed["workflows"]))
        self.assertTrue(self.service.validate_workflow(saved["id"])["valid"])
        with self.assertRaises(ServiceError):
            self.service.delete_workflow("cpu_empty_image_smoke")
        self.assertEqual(self.service.delete_workflow(saved["id"]), {"deleted": True})
        with self.assertRaises(ServiceError):
            self.service.validate_workflow(saved["id"])

    def test_reservation_is_durable_bound_to_request_and_job_owner(self):
        submission = str(uuid.uuid4())
        prepared = self.service.prepare(
            "cpu_empty_image_smoke",
            {"client_submission_id": submission, "inputs": {"width": 96}},
        )
        original, created = self.service.reserve(prepared, "job-owned")
        self.assertTrue(created)
        self.assertEqual(original["owner_id"], "owner-1")

        restarted = self._new_service()
        same_request = restarted.prepare(
            "cpu_empty_image_smoke",
            {"client_submission_id": submission, "inputs": {"width": 96}},
        )
        replay, created = restarted.reserve(same_request, "job-should-not-replace")
        self.assertFalse(created)
        self.assertEqual(replay["job_id"], "job-owned")
        self.assertEqual(self.access_checks, ["job-owned"])

        changed_request = restarted.prepare(
            "cpu_empty_image_smoke",
            {"client_submission_id": submission, "inputs": {"width": 128}},
        )
        with self.assertRaises(ServiceError) as raised:
            restarted.reserve(changed_request, "job-3")
        self.assertEqual(raised.exception.status, 409)

        def deny(_job_id):
            raise ServiceError(404, "owned job not found")

        foreign = self._new_service(require_job_access=deny)
        foreign_replay = foreign.prepare(
            "cpu_empty_image_smoke",
            {"client_submission_id": submission, "inputs": {"width": 96}},
        )
        with self.assertRaises(ServiceError) as raised:
            foreign.reserve(foreign_replay, "job-4")
        self.assertEqual(raised.exception.status, 404)

    def test_resume_reconnects_owned_intent_without_replaying_prompt_post(self):
        prepared, original = self._prepare()
        submission = prepared["client_submission_id"]
        self.service._update_receipt(submission, state="running", prompt_id="prompt-1")
        resumed_receipts = []
        submit_requests = []

        def missing_job(_job_id):
            raise ServiceError(404, "The in-memory job was lost during restart.")

        restarted = self._new_service(
            require_job_access=missing_job,
            principal_id=lambda: "owner-1",
            job_exists=lambda _job_id: False,
            resume=lambda receipt: resumed_receipts.append(receipt) or {"job_id": receipt["job_id"], "status": "queued"},
            submit=lambda request: submit_requests.append(request),
        )
        result = restarted.resume_submission(submission)

        self.assertEqual(result, {"job_id": original["job_id"], "status": "queued"})
        self.assertEqual(len(resumed_receipts), 1)
        self.assertEqual(resumed_receipts[0]["state"], "running")
        self.assertEqual(resumed_receipts[0]["prompt_id"], "prompt-1")
        self.assertEqual(self.client.submit_calls, [])
        self.assertEqual(submit_requests, [])

    def test_resume_rejects_foreign_owner_and_never_replays_prepared_or_completed_intents(self):
        resume_calls = []
        submit_calls = []
        prepared, _ = self._prepare()
        submission = prepared["client_submission_id"]

        def missing_job(_job_id):
            raise ServiceError(404, "The in-memory job was lost during restart.")

        def resume(receipt):
            resume_calls.append(receipt)
            return {"job_id": receipt["job_id"], "status": "queued"}

        restarted = self._new_service(
            require_job_access=missing_job,
            principal_id=lambda: "owner-1",
            job_exists=lambda _job_id: False,
            resume=resume,
            submit=lambda request: submit_calls.append(request),
        )

        foreign = self._new_service(
            principal_id=lambda: "owner-2",
            job_exists=lambda _job_id: False,
            resume=resume,
        )
        with self.assertRaises(ServiceError) as raised:
            foreign.resume_submission(submission)
        self.assertEqual(raised.exception.status, 404)
        self.assertEqual(resume_calls, [])

        with self.assertRaises(ServiceError) as raised:
            restarted.resume_submission(submission)
        self.assertEqual(raised.exception.status, 409, "A prepared-but-never-dispatched intent cannot be resumed.")
        self.assertEqual(resume_calls, [])

        self.service._update_receipt(submission, state="completed", output_files=["owned.png"])
        result = restarted.resume_submission(submission)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(resume_calls, [], "A completed receipt is returned without starting another monitor or prompt POST.")
        self.assertEqual(self.client.submit_calls, [])
        self.assertEqual(submit_calls, [])

    def test_abandon_only_dismisses_orphaned_prepared_intents(self):
        with self.assertRaises(ServiceError) as raised:
            self.service.abandon_submission(str(uuid.uuid4()))
        self.assertEqual(raised.exception.status, 404)

        prepared, _ = self._prepare()
        submission = prepared["client_submission_id"]
        with self.assertRaises(ServiceError) as raised:
            self.service.abandon_submission(submission)
        self.assertEqual(raised.exception.status, 409, "A queued job must be cancelled through Maestro's queue.")

        self.service._update_receipt(submission, state="running", prompt_id="prompt-1")
        with self.assertRaises(ServiceError) as raised:
            self.service.abandon_submission(submission)
        self.assertEqual(raised.exception.status, 409, "Dispatched remote work cannot be dismissed as a draft.")

        orphaned, _ = self._prepare()
        orphaned_service = self._new_service(job_exists=lambda _job_id: False)
        result = orphaned_service.abandon_submission(orphaned["client_submission_id"])
        self.assertEqual(result["state"], "cancelled")
        with self.assertRaises(ServiceError) as raised:
            orphaned_service.abandon_submission(orphaned["client_submission_id"])
        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(self.client.submit_calls, [])


class TestComfyRuntimeExecution(ComfyRuntimeTestCase):
    def test_builtin_template_defaults_are_applied_and_png_published_with_metadata(self):
        prepared, _ = self._prepare()
        self._complete()
        output_dir = Path(self.temporary.name) / "published"

        names = self._run(prepared, out_dir=output_dir)

        self.assertEqual(len(names), 1)
        image_path = output_dir / names[0]
        self.assertTrue(image_path.is_file())
        with Image.open(image_path) as image:
            self.assertEqual(image.size, (5, 3))
        submitted = self.client.submit_calls[0]["graph"]
        self.assertEqual(submitted["1"]["inputs"]["width"], 64)
        self.assertEqual(submitted["1"]["inputs"]["height"], 64)
        self.assertEqual(submitted["1"]["inputs"]["color"], 0)
        self.assertEqual(submitted["1"]["inputs"]["batch_size"], 1)
        self.assertEqual(self.client.submit_calls[0]["client_id"], prepared["client_submission_id"])
        receipt = self.service.submission_receipt(prepared["client_submission_id"])
        self.assertEqual(receipt["state"], "completed")
        self.assertEqual(receipt["prompt_id"], "prompt-1")

        metadata = json.loads(image_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["generation_engine"], "comfyui")
        self.assertEqual(metadata["width"], 5)
        self.assertEqual(metadata["height"], 3)
        self.assertEqual(metadata["comfyui_workflow_id"], "cpu_empty_image_smoke")

    def _save_bound_checkpoint_workflow(self):
        graph = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "template-placeholder.safetensors"}},
            "2": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}},
            "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "bound", "images": ["2", 0]}},
        }
        saved = self.service.save_workflow({
            "name": "Bound checkpoint",
            "kind": "image",
            "workflow": graph,
            "bindings": [{
                "name": "checkpoint",
                "label": "Checkpoint",
                "node_id": "1",
                "input": "ckpt_name",
                "type": "string",
                "required": True,
            }],
        })
        return saved

    def test_bound_model_is_applied_before_preflight_and_submission(self):
        saved = self._save_bound_checkpoint_workflow()
        self.client.info = _object_info(checkpoints=("installed.safetensors",))
        selected = {"checkpoint": "installed.safetensors"}

        validation = self.service.validate_workflow(saved["id"], inputs=selected)
        self.assertTrue(validation["valid"], validation)

        prepared, _ = self._prepare(
            inputs=selected,
            workflow_id=saved["id"],
        )
        self._complete()
        self._run(prepared)
        self.assertEqual(
            self.client.submit_calls[0]["graph"]["1"]["inputs"]["ckpt_name"],
            "installed.safetensors",
        )

    def _native_video_route(self, *, map_audio_guide=False):
        mappings = {
            "prompt": "prompt",
            "seed": "seed",
            "width": "width",
            "height": "height",
            "num_inference_steps": "steps",
            "guidance_scale": "cfg",
            "video_length": "frames",
            "fps": "fps",
        }
        binding_types = {
            "prompt": "string",
            "seed": "integer",
            "width": "integer",
            "height": "integer",
            "steps": "integer",
            "cfg": "number",
            "frames": "integer",
            "fps": "number",
        }
        bindings = [{"name": name, "type": binding_types[name]} for name in mappings.values()]
        if map_audio_guide:
            mappings["audio_guide"] = "driving_audio"
            bindings.append({"name": "driving_audio", "type": "audio"})
        route = {
            "workflow_id": "video-fixture",
            "enabled": True,
            "kind": "video",
            "field_bindings": mappings,
            "input_defaults": {},
        }
        config = {"enabled": True, "routes": {"native-video-model": route}}
        workflow = {"id": "video-fixture", "kind": "video", "bindings": bindings}
        return config, workflow

    def test_native_route_rejects_unmapped_video_and_audio_guides(self):
        base = {
            "model_type": "native-video-model",
            "generation_mode": "video",
            "prompt": "A moving scene",
            "seed": 8,
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
            "video_length": 24,
            "fps": 24,
        }
        config, workflow = self._native_video_route()
        with mock.patch.object(self.service, "_config", return_value=config), mock.patch.object(self.service, "_workflow", return_value=workflow):
            with self.assertRaises(ServiceError) as raised:
                self.service.native_request({**base, "video_guide": {"path": "uploads/guide.mp4"}})
        self.assertEqual(raised.exception.status, 422)
        self.assertIn("no binding for video_guide", str(raised.exception))

        config, workflow = self._native_video_route(map_audio_guide=True)
        with mock.patch.object(self.service, "_config", return_value=config), mock.patch.object(self.service, "_workflow", return_value=workflow):
            with self.assertRaises(ServiceError) as raised:
                self.service.native_request({
                    **base,
                    "audio_guide": {"path": "uploads/guide.wav"},
                    "audio_frame_offset": 48,
                })
        self.assertEqual(raised.exception.status, 422)
        self.assertIn("source offset mapped", str(raised.exception))

    def test_ambiguous_post_reconciles_exact_queue_metadata_without_resubmit(self):
        prepared, _ = self._prepare()
        submission = prepared["client_submission_id"]
        self.client.submit_error = ComfyTransportError("submit", ambiguous=True)
        self.client.queue_default = {
            "queue_running": [],
            "queue_pending": [
                [1, "wrong-prompt", {}, {"maestro_submission_id": submission, "client_id": "other"}],
                [2, "prompt-1", {}, {"maestro_submission_id": submission, "client_id": submission}],
            ],
        }
        self._complete()

        names = self._run(prepared)

        self.assertEqual(len(names), 1)
        self.assertEqual(len(self.client.submit_calls), 1)
        self.assertEqual(self.client.queue_calls, 1)
        self.assertEqual(self.service.submission_receipt(submission)["state"], "completed")

    def test_ambiguous_post_reconciles_actual_history_prompt_tuple(self):
        prepared, _ = self._prepare()
        submission = prepared["client_submission_id"]
        self.client.submit_error = ComfyTransportError("submit", ambiguous=True)
        self.client.history_all = {
            "prompt-from-history": _history_row(
                prompt=[
                    3,
                    "prompt-from-history",
                    {},
                    {"maestro_submission_id": submission, "client_id": submission},
                    ["2"],
                ]
            )
        }
        self.client.history_by_id["prompt-from-history"] = _history_row()

        names = self._run(prepared)

        self.assertEqual(len(names), 1)
        self.assertEqual(len(self.client.submit_calls), 1)
        self.assertEqual(self.service.submission_receipt(submission)["prompt_id"], "prompt-from-history")

    def test_unreconciled_ambiguous_post_stays_uncertain_and_never_resubmits(self):
        prepared, _ = self._prepare()
        self.client.submit_error = ComfyTransportError("submit", ambiguous=True)
        with self.assertRaises(ServiceError) as raised:
            self._run(prepared)
        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(self.service.submission_receipt(prepared["client_submission_id"])["state"], "uncertain")
        with self.assertRaises(ServiceError):
            self._run(prepared)
        self.assertEqual(len(self.client.submit_calls), 1)

    def test_timeout_stays_uncertain_without_interrupt_or_resubmission(self):
        prepared, _ = self._prepare()
        prepared["execution_timeout"] = 0.001
        with mock.patch("app.services.comfyui_runtime.time.sleep", return_value=None):
            with self.assertRaises(ServiceError) as raised:
                self._run(prepared)
        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(self.service.submission_receipt(prepared["client_submission_id"])["state"], "uncertain")
        self.assertEqual(len(self.client.submit_calls), 1)
        self.assertEqual(self.client.interrupt_calls, 0)
        self.assertEqual(self.client.free_calls, 0)

    def test_cancel_deletes_only_owned_pending_prompt_and_never_interrupts(self):
        prepared, _ = self._prepare()
        self.client.queue_default = {
            "queue_running": [],
            "queue_pending": [
                [1, "prompt-1", {}, {"maestro_submission_id": prepared["client_submission_id"], "client_id": prepared["client_submission_id"]}],
                [2, "foreign-prompt", {}, {}],
            ],
        }
        cancellation_checks = 0

        def cancelled():
            nonlocal cancellation_checks
            cancellation_checks += 1
            return cancellation_checks >= 2

        with mock.patch("app.services.comfyui_runtime.time.sleep", return_value=None):
            result = self._run(prepared, cancelled=cancelled)

        self.assertEqual(result, [])
        self.assertEqual(self.client.deleted_ids, ["prompt-1"])
        self.assertEqual(self.service.submission_receipt(prepared["client_submission_id"])["state"], "cancelled")
        self.assertEqual(self.client.interrupt_calls, 0)
        self.assertEqual(self.client.free_calls, 0)

    def test_cancel_refuses_queue_entries_without_matching_ownership_metadata(self):
        prepared, _ = self._prepare()
        self.client.queue_default = {
            "queue_running": [],
            "queue_pending": [[1, "prompt-1", {}, {}]],
        }
        cancellation_checks = 0

        def cancelled():
            nonlocal cancellation_checks
            cancellation_checks += 1
            return cancellation_checks >= 2

        with mock.patch("app.services.comfyui_runtime.time.sleep", return_value=None):
            with self.assertRaises(ServiceError) as raised:
                self._run(prepared, cancelled=cancelled)

        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(self.client.deleted_ids, [])
        self.assertEqual(self.service.submission_receipt(prepared["client_submission_id"])["state"], "uncertain")
        self.assertEqual(self.client.interrupt_calls, 0)

    def test_cancel_refuses_truncated_queue_entry_with_owned_prompt_id(self):
        prepared, _ = self._prepare()
        self.client.queue_default = {
            "queue_running": [],
            "queue_pending": [[1, "prompt-1", {}]],
        }
        cancellation_checks = 0

        def cancelled():
            nonlocal cancellation_checks
            cancellation_checks += 1
            return cancellation_checks >= 2

        with mock.patch("app.services.comfyui_runtime.time.sleep", return_value=None):
            with self.assertRaises(ServiceError) as raised:
                self._run(prepared, cancelled=cancelled)

        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(self.client.deleted_ids, [])
        self.assertEqual(self.service.submission_receipt(prepared["client_submission_id"])["state"], "uncertain")
        self.assertEqual(self.client.interrupt_calls, 0)

    def test_missing_or_truncated_png_never_becomes_a_successful_receipt(self):
        prepared, _ = self._prepare()
        self._complete(outputs={})
        with self.assertRaises(ServiceError):
            self._run(prepared)
        self.assertEqual(self.service.submission_receipt(prepared["client_submission_id"])["state"], "failed")

        prepared, _ = self._prepare()
        self._complete()
        self.client.output_bytes = b"truncated png"
        output_dir = Path(self.temporary.name) / "truncated"
        with self.assertRaises(Exception):
            self._run(prepared, out_dir=output_dir)
        self.assertNotEqual(self.service.submission_receipt(prepared["client_submission_id"])["state"], "completed")
        self.assertEqual(list(output_dir.iterdir()), [])

    def test_history_path_traversal_records_are_rejected(self):
        prepared, _ = self._prepare()
        self._complete(outputs={
            "2": {"images": [{"filename": "../escape.png", "subfolder": "", "type": "output"}]},
            "3": {"images": [{"filename": "safe.png", "subfolder": "../outside", "type": "output"}]},
            "4": {"images": [{"filename": "drive.png", "subfolder": "C:/outside", "type": "output"}]},
        })
        with self.assertRaises(ServiceError):
            self._run(prepared)
        self.assertEqual(self.client.output_calls, [])
        self.assertFalse((Path(self.temporary.name) / "escape.png").exists())
        self.assertFalse((Path(self.temporary.name) / "outside").exists())


class TestComfyRuntimeDedicatedServer(ComfyRuntimeTestCase):
    def test_dedicated_mode_releases_memory_only_after_empty_queue(self):
        self.service.configure({"dedicated": True})
        self.client.queue_default = {"queue_running": [], "queue_pending": []}

        self.service.assert_native_idle()

        self.assertEqual(self.client.queue_calls, 1)
        self.assertEqual(self.client.release_memory_calls, 1)
        self.assertEqual(self.client.free_calls, 0)

    def test_dedicated_mode_does_not_release_memory_while_queue_is_busy(self):
        self.service.configure({"dedicated": True})
        self.client.queue_default = {
            "queue_running": [],
            "queue_pending": [[1, "foreign-prompt", {}, {}]],
        }

        with self.assertRaises(ServiceError) as raised:
            self.service.assert_native_idle()

        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(self.client.release_memory_calls, 0)
        self.assertEqual(self.client.free_calls, 0)


if __name__ == "__main__":
    unittest.main()
