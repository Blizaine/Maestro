"""CPU-only tests for the ComfyUI adapter into Maestro's owned job lifecycle."""

from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
import ast
import io
import json
from pathlib import Path
import threading
import uuid

from PIL import Image
import pytest

from app.services.comfyui_integration import ComfyJobAdapter, resolve_workflow_media
from app.services.comfyui_runtime import ServiceError
from app.services.comfyui_workflows import apply_bindings


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (4, 3), (30, 80, 160)).save(stream, format="PNG")
    return stream.getvalue()


def _object_info() -> dict:
    return {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["installed.safetensors"], {}]}}},
        "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}], "clip": ["CLIP", {}]}}},
        "EmptyLatentImage": {"input": {"required": {"width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}]}}},
        "KSampler": {"input": {"required": {
            "seed": ["INT", {}], "steps": ["INT", {}], "cfg": ["FLOAT", {}],
            "sampler_name": [["euler"], {}], "scheduler": [["normal"], {}],
            "denoise": ["FLOAT", {}], "model": ["MODEL", {}], "positive": ["CONDITIONING", {}],
            "negative": ["CONDITIONING", {}], "latent_image": ["LATENT", {}],
        }}},
        "VAEDecode": {"input": {"required": {"samples": ["LATENT", {}], "vae": ["VAE", {}]}}},
        "SaveImage": {"input": {"required": {"filename_prefix": ["STRING", {}], "images": ["IMAGE", {}]}}},
        "EmptyImage": {"input": {"required": {
            "width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}], "color": ["INT", {}],
        }}},
    }


class FakeComfyClient:
    def __init__(self, endpoint: str, events: list[tuple]):
        self.endpoint = endpoint
        self.events = events
        self.closed = False
        self.submit_calls: list[dict] = []

    def close(self):
        self.closed = True

    def object_info(self):
        self.events.append(("object_info",))
        return deepcopy(_object_info())

    def submit(self, graph, *, client_id, submission_id):
        self.events.append(("submit",))
        self.submit_calls.append({"graph": deepcopy(graph), "client_id": client_id, "submission_id": submission_id})
        return {"prompt_id": "remote-prompt-1"}

    def history(self, prompt_id=None):
        self.events.append(("history", prompt_id))
        return {"remote-prompt-1": {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"7": {"images": [{"filename": "result.png", "subfolder": "", "type": "output"}]}},
        }}

    def output(self, filename, subfolder="", type="output"):
        self.events.append(("output", filename))
        return _png_bytes()

    def upload_image(self, data, filename):
        raise AssertionError("These adapter tests do not require media upload.")


def _adapter(tmp_path: Path, *, cancel_state: dict | None = None):
    jobs: dict[str, dict] = {}
    events: list[tuple] = []
    clients: list[FakeComfyClient] = []
    starts: list[tuple] = []
    finishes: list[tuple] = []
    updates: list[tuple] = []
    context = ContextVar(f"comfy_test_request_{uuid.uuid4().hex}", default="outside")

    def factory(endpoint):
        client = FakeComfyClient(endpoint, events)
        clients.append(client)
        return client

    def start_worker(job_id):
        starts.append((job_id, jobs.get(job_id), context.get()))

    def finish_job(job, status, **kwargs):
        finishes.append((job, status, kwargs))
        job["status"] = status
        job.update(kwargs)
        return True

    def is_cancelled(_job):
        return bool(cancel_state and cancel_state.get("cancelled"))

    adapter = ComfyJobAdapter(
        tmp_path,
        jobs=jobs,
        workspace_dir=lambda workspace: tmp_path / "outputs" / workspace,
        start_worker=start_worker,
        require_job_access=lambda _job_id: None,
        principal_id=lambda: "owner-1",
        link_folders=lambda paths: paths,
        known_assets=lambda: [],
        resolve_media=lambda raw, _workspace: Path(raw),
        finish_job=finish_job,
        update_job=lambda job, **kwargs: updates.append((job, kwargs)),
        is_cancelled=is_cancelled,
        unload_native=lambda: events.append(("unload_native",)),
    )
    adapter.service.client_factory = factory
    return {
        "adapter": adapter,
        "jobs": jobs,
        "events": events,
        "clients": clients,
        "starts": starts,
        "finishes": finishes,
        "updates": updates,
        "context": context,
    }


def _enable(adapter: ComfyJobAdapter):
    adapter.service.configure({
        "enabled": True,
        "endpoint": "http://127.0.0.1:8188",
        "execution_timeout": 30,
    })


def _image_route(adapter: ComfyJobAdapter, *, prompt_target="positive_prompt") -> dict:
    negative_target = "positive_prompt" if prompt_target == "negative_prompt" else "negative_prompt"
    return adapter.service.save_route({
        "model_type": "native-photo-model",
        "workflow_id": "sdxl_checkpoint_txt2img",
        "enabled": True,
        "kind": "image",
        "field_bindings": {
            "prompt": prompt_target,
            "negative_prompt": negative_target,
            "seed": "seed",
            "width": "width",
            "height": "height",
            "num_inference_steps": "steps",
            "guidance_scale": "cfg",
        },
        "input_defaults": {"checkpoint": "installed.safetensors"},
    })


def test_submit_registers_before_worker_and_replays_submission_once(tmp_path):
    state = _adapter(tmp_path)
    adapter, jobs = state["adapter"], state["jobs"]
    _enable(adapter)
    submission = str(uuid.uuid4())
    payload = {"client_submission_id": submission, "inputs": {}, "workspace": "default"}

    token = state["context"].set("authenticated-request")
    try:
        first = adapter.service.run_workflow("cpu_empty_image_smoke", payload)
        second = adapter.service.run_workflow("cpu_empty_image_smoke", payload)
    finally:
        state["context"].reset(token)

    assert first["job_id"] == second["job_id"]
    assert first["status"] == "queued"
    assert second["replayed"] is True
    assert len(state["starts"]) == 1
    registered_id, registered_job, request_context = state["starts"][0]
    assert registered_id == first["job_id"]
    assert registered_job is jobs[registered_id]
    assert request_context == "authenticated-request"
    assert jobs[registered_id]["params"]["_comfyui_submission"] == submission
    receipt = adapter.service.submission_receipt(submission)
    assert receipt["job_id"] == first["job_id"]


def test_native_default_and_explicit_missing_route_do_not_make_http_calls(tmp_path):
    state = _adapter(tmp_path)
    adapter = state["adapter"]
    native_body = {"model_type": "native-photo-model", "generation_mode": "image", "prompt": "hello"}

    assert adapter.capture_engine(native_body) == {"engine": "native"}
    assert adapter.attach_native(native_body, "default", "job-1", str(uuid.uuid4())) is None
    assert "generation_engine" not in native_body
    assert state["clients"] == []
    assert state["starts"] == []

    with pytest.raises(ServiceError) as error:
        adapter.capture_engine({**native_body, "generation_engine": "comfyui"})
    assert error.value.status == 409
    assert state["clients"] == []


def test_disabled_integration_keeps_saved_native_model_route_on_native(tmp_path):
    state = _adapter(tmp_path)
    adapter = state["adapter"]
    _enable(adapter)
    _image_route(adapter)
    adapter.service.configure({"enabled": False})
    state["clients"].clear()
    state["events"].clear()

    body = {"model_type": "native-photo-model", "generation_mode": "image", "prompt": "hello"}
    assert adapter.capture_engine(body) == {"engine": "native"}
    assert adapter.attach_native(body, "default", "job-disabled", str(uuid.uuid4())) is None
    assert "generation_engine" not in body
    assert state["clients"] == []
    assert state["events"] == []

    with pytest.raises(ServiceError) as error:
        adapter.capture_engine({**body, "generation_engine": "comfyui"})
    assert error.value.status == 409
    assert state["clients"] == []


def test_native_route_is_frozen_before_dispatch_and_completes_only_with_verified_png(tmp_path):
    state = _adapter(tmp_path)
    adapter, jobs = state["adapter"], state["jobs"]
    _enable(adapter)
    original_route = _image_route(adapter)
    body = {
        "model_type": "native-photo-model",
        "generation_mode": "image",
        "prompt": "Frozen prompt",
        "negative_prompt": "Avoid blur",
        "seed": 17,
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
    }
    snapshot = adapter.capture_engine(body)
    assert snapshot == {"engine": "comfyui", "route": original_route}

    # The model route can change after acceptance; this job continues to use
    # the accepted mapping stored in its snapshot and durable prepared receipt.
    _image_route(adapter, prompt_target="negative_prompt")
    job_id = "job-frozen"
    submission = str(uuid.uuid4())
    assert adapter.attach_native(body, "default", job_id, submission, engine_snapshot=snapshot) is None
    assert body["_comfyui_submission"] == submission
    assert body["generation_engine"] == "comfyui"
    assert state["starts"] == [], "Route preparation must be durable before launch dispatches the worker."
    frozen = adapter.service.prepared_submission(submission)
    assert frozen["native_route"] == original_route
    assert adapter.service.submission_receipt(submission)["state"] == "prepared"

    # Change the route again after the receipt exists. Execution still uses
    # the frozen route and the fake ComfyUI process returns a real PNG payload.
    _image_route(adapter, prompt_target="negative_prompt")
    job = {
        "id": job_id,
        "status": "running",
        "params": deepcopy(body),
        "out_dir": str(tmp_path / "outputs" / "default"),
    }
    jobs[job_id] = job
    assert adapter.execute(job) is True

    submit_events = [client.submit_calls for client in state["clients"] if client.submit_calls]
    assert len(submit_events) == 1
    graph = submit_events[0][0]["graph"]
    assert graph["2"]["inputs"]["text"] == "Frozen prompt"
    assert graph["3"]["inputs"]["text"] == "Avoid blur"
    assert state["events"].index(("unload_native",)) < state["events"].index(("submit",))
    assert len(state["finishes"]) == 1
    finished_job, status, metadata = state["finishes"][0]
    assert finished_job is job
    assert status == "completed"
    assert metadata["progress"] == 100
    output_name = metadata["output_files"][0]
    published = Path(job["out_dir"]) / output_name
    assert published.is_file()
    with Image.open(published) as image:
        assert image.size == (4, 3)
    assert adapter.service.submission_receipt(submission)["state"] == "completed"


def test_cancelled_before_remote_execute_does_not_unload_native_models(tmp_path):
    cancel_state = {"cancelled": False}
    state = _adapter(tmp_path, cancel_state=cancel_state)
    adapter = state["adapter"]
    _enable(adapter)
    submission = str(uuid.uuid4())
    queued = adapter.service.run_workflow("cpu_empty_image_smoke", {
        "client_submission_id": submission, "inputs": {}, "workspace": "default",
    })
    job = state["jobs"][queued["job_id"]]
    cancel_state["cancelled"] = True

    assert adapter.execute(job) is False
    assert not any(event[0] == "unload_native" for event in state["events"])
    assert not any(event[0] in {"object_info", "submit", "history", "output"} for event in state["events"])
    assert adapter.service.submission_receipt(submission)["state"] == "cancelled"
    assert state["finishes"] == []


def test_persisted_cancel_is_restored_after_restart_and_never_submits_again(tmp_path):
    original = _adapter(tmp_path)
    first_adapter = original["adapter"]
    _enable(first_adapter)
    submission = str(uuid.uuid4())
    queued = first_adapter.service.run_workflow("cpu_empty_image_smoke", {
        "client_submission_id": submission, "inputs": {}, "workspace": "default",
    })
    first_adapter.service._update_receipt(
        submission, state="running", prompt_id="remote-prompt-1", cancel_requested=True,
    )

    restarted = _adapter(tmp_path)
    adapter = restarted["adapter"]
    resumed = adapter.service.resume_submission(submission)
    job = restarted["jobs"][queued["job_id"]]
    assert resumed["job_id"] == queued["job_id"]
    assert job["comfyui_cancel_requested"] is True
    assert adapter.execute(job) is False
    assert not any(event[0] == "submit" for event in restarted["events"])
    assert any(event == ("history", "remote-prompt-1") for event in restarted["events"])
    assert adapter.service.submission_receipt(submission)["state"] == "cancelled"
    assert restarted["finishes"] == []


def test_duplicate_active_monitor_for_same_job_is_rejected(tmp_path, monkeypatch):
    state = _adapter(tmp_path)
    adapter = state["adapter"]
    entered = threading.Event()
    release = threading.Event()
    results = []

    def blocking_execute(_job, *, finalize=True):
        entered.set()
        assert release.wait(3), "The active-monitor test must release its worker."
        return True

    monkeypatch.setattr(adapter, "_execute", blocking_execute)
    job = {"id": "active-monitor-job"}
    worker = threading.Thread(target=lambda: results.append(adapter.execute(job)))
    worker.start()
    try:
        assert entered.wait(3), "The first monitor should enter its owned execution section."
        with pytest.raises(ServiceError) as error:
            adapter.execute(job)
        assert error.value.status == 409
    finally:
        release.set()
        worker.join(3)

    assert not worker.is_alive()
    assert results == [True]
    assert "active-monitor-job" not in adapter.active_monitors


def test_empty_execute_result_never_reports_success(tmp_path, monkeypatch):
    state = _adapter(tmp_path)
    adapter = state["adapter"]
    _enable(adapter)
    queued = adapter.service.run_workflow("cpu_empty_image_smoke", {
        "client_submission_id": str(uuid.uuid4()), "inputs": {}, "workspace": "default",
    })
    job = state["jobs"][queued["job_id"]]

    def no_verified_outputs(*_args, **_kwargs):
        return []

    monkeypatch.setattr(adapter.service, "execute", no_verified_outputs)
    with pytest.raises(ServiceError) as error:
        adapter.execute(job)
    assert error.value.status == 422
    assert state["finishes"] == []


def test_late_cancellation_removes_owned_published_outputs_and_cancels_receipt(tmp_path, monkeypatch):
    cancel_state = {"cancelled": False}
    state = _adapter(tmp_path, cancel_state=cancel_state)
    adapter = state["adapter"]
    _enable(adapter)
    submission = str(uuid.uuid4())
    queued = adapter.service.run_workflow("cpu_empty_image_smoke", {
        "client_submission_id": submission, "inputs": {}, "workspace": "default",
    })
    job = state["jobs"][queued["job_id"]]
    job["status"] = "running"
    output_dir = Path(job["out_dir"])
    filename = f"comfyui-{submission}-1.png"
    output = output_dir / filename
    sidecar = output.with_suffix(".meta.json")

    def publish_then_cancel(prepared, target_dir, **_kwargs):
        target_dir.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_png_bytes())
        sidecar.write_text(json.dumps({"model_type": "comfyui_workflow"}), encoding="utf-8")
        adapter.service._update_receipt(submission, state="completed", output_files=[filename])
        cancel_state["cancelled"] = True
        return [filename]

    monkeypatch.setattr(adapter.service, "execute", publish_then_cancel)
    assert adapter.execute(job) is False

    assert not output.exists(), "A late cancel removes only this submission's published file."
    assert not sidecar.exists(), "The cancelled file's metadata sidecar is removed too."
    assert adapter.service.submission_receipt(submission)["state"] == "cancelled"
    assert state["finishes"] == [], "Cancellation is finalized by Maestro's existing cancel path, not as a workflow completion callback."


def test_adapter_resume_monitors_owned_prompt_without_resubmitting_after_restart(tmp_path):
    original = _adapter(tmp_path)
    first_adapter = original["adapter"]
    _enable(first_adapter)
    submission = str(uuid.uuid4())
    queued = first_adapter.service.run_workflow("cpu_empty_image_smoke", {
        "client_submission_id": submission, "inputs": {}, "workspace": "default",
    })
    first_adapter.service._update_receipt(submission, state="running", prompt_id="remote-prompt-1")

    # A fresh adapter represents a backend restart: the receipt and remote
    # prompt survive, but the original in-memory job registry does not.
    restarted = _adapter(tmp_path)
    adapter = restarted["adapter"]
    restored = adapter.service.resume_submission(submission)
    assert restored == {"job_id": queued["job_id"], "status": "queued", "client_submission_id": submission}
    job = restarted["jobs"][queued["job_id"]]
    assert job["comfyui_monitor_only"] is True
    assert restarted["starts"][0][1] is job
    assert restarted["starts"][0][2] == "outside"

    assert adapter.execute(job) is True
    assert not any(event[0] == "submit" for event in restarted["events"]), "Recovery only polls the known prompt; it never POSTs /prompt again."
    assert any(event == ("history", "remote-prompt-1") for event in restarted["events"])
    assert len(restarted["finishes"]) == 1
    assert restarted["finishes"][0][1] == "completed"
    assert adapter.service.submission_receipt(submission)["state"] == "completed"


def test_native_route_resume_preserves_accepted_inputs_graph_and_model_identity(tmp_path):
    original = _adapter(tmp_path)
    first_adapter = original["adapter"]
    _enable(first_adapter)
    route = _image_route(first_adapter)
    body = {
        "model_type": "native-photo-model",
        "generation_mode": "image",
        "prompt": "Accepted prompt",
        "negative_prompt": "Accepted negative prompt",
        "seed": 17,
        "width": 640,
        "height": 384,
        "num_inference_steps": 29,
        "guidance_scale": 6.5,
    }
    snapshot = first_adapter.capture_engine(body)
    submission = str(uuid.uuid4())
    assert first_adapter.attach_native(body, "default", "native-job", submission, engine_snapshot=snapshot) is None
    frozen = first_adapter.service.prepared_submission(submission)
    accepted_inputs = deepcopy(frozen["inputs"])
    accepted_graph = apply_bindings(
        frozen["workflow"]["workflow"], frozen["workflow"]["bindings"], accepted_inputs,
    )
    first_adapter.service._update_receipt(
        submission,
        state="running",
        prompt_id="remote-prompt-1",
        effective_inputs=accepted_inputs,
        applied_workflow_digest="accepted-graph-digest",
        applied_workflow=accepted_graph,
    )

    # Simulate the route being edited after dispatch. The monitor must publish
    # provenance from the durable accepted workflow, not remap reduced job data.
    _image_route(first_adapter, prompt_target="negative_prompt")
    restarted = _adapter(tmp_path)
    adapter = restarted["adapter"]
    resumed = adapter.service.resume_submission(submission)
    job = restarted["jobs"][resumed["job_id"]]
    assert job["comfyui_monitor_only"] is True
    assert adapter.execute(job) is True

    assert not any(event[0] == "submit" for event in restarted["events"])
    output_name = job["output_files"][0]
    metadata_path = Path(job["out_dir"]) / output_name
    metadata = json.loads(metadata_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["model_type"] == "native-photo-model"
    assert metadata["seed"] == 17
    assert metadata["comfyui_inputs"] == accepted_inputs
    assert metadata["comfyui_applied_workflow"] == accepted_graph
    assert metadata["comfyui_workflow_name"] == route["name"]


def test_resolve_workflow_media_accepts_managed_roots_and_rejects_escape(tmp_path):
    app_root = tmp_path / "app"
    output_root = app_root / "outputs"
    gallery = output_root / "default" / "gallery.png"
    upload = app_root / "uploads" / "ref.wav"
    character = app_root / "settings" / "characters" / "portrait.webp"
    outside = tmp_path / "outside.png"
    for path in (gallery, upload, character, outside):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"managed media placeholder")

    assert resolve_workflow_media(app_root, "outputs", "gallery.png", "default") == gallery.resolve()
    assert resolve_workflow_media(app_root, "outputs", str(upload), "default") == upload.resolve()
    assert resolve_workflow_media(app_root, "outputs", str(character), "default") == character.resolve()

    for raw in ("", "../../outside.png", str(outside), "notes.txt", "bad\x00name.png"):
        with pytest.raises(ServiceError) as error:
            resolve_workflow_media(app_root, "outputs", raw, "default")
        assert error.value.status == 422

    escaped_link = app_root / "uploads" / "outside-link.png"
    try:
        escaped_link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("This Windows environment does not permit creating symlinks for the containment regression.")
    with pytest.raises(ServiceError) as error:
        resolve_workflow_media(app_root, "outputs", str(escaped_link), "default")
    assert error.value.status == 422

    # The managed directory itself may not be redirected outside the app
    # root, even when a caller supplies the resolved external filename.
    redirected_app = tmp_path / "redirected-app"
    (redirected_app / "outputs").mkdir(parents=True)
    external_uploads = tmp_path / "external-uploads"
    external_uploads.mkdir()
    redirected_file = external_uploads / "redirected.png"
    redirected_file.write_bytes(b"outside managed upload root")
    (redirected_app / "uploads").symlink_to(external_uploads, target_is_directory=True)
    with pytest.raises(ServiceError) as error:
        resolve_workflow_media(redirected_app, "outputs", str(redirected_file), "default")
    assert error.value.status == 422


def _launch_ast() -> ast.Module:
    launch = Path(__file__).resolve().parents[1] / "app" / "launch.py"
    return ast.parse(launch.read_text(encoding="utf-8"), filename=str(launch))


def _functions(tree: ast.AST, name: str):
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]


def _attr_calls(node: ast.AST, attribute: str):
    return [call for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == attribute]


def test_launch_attaches_native_routes_during_generation_submission_and_executes_under_owned_slot():
    tree = _launch_ast()
    prepare_functions = _functions(tree, "_prepare_generation_submission")
    run_functions = _functions(tree, "_run_generation")
    assert len(prepare_functions) == 1
    assert len(run_functions) == 1
    prepare, run = prepare_functions[0], run_functions[0]

    attach_calls = _attr_calls(tree, "attach_native")
    assert len(attach_calls) == 2, "Only request submission and queued-job recovery should attach Comfy routes."
    assert all(prepare.lineno <= call.lineno <= prepare.end_lineno or run.lineno <= call.lineno <= run.end_lineno for call in attach_calls)
    prepare_attach = [call for call in _attr_calls(prepare, "attach_native") if prepare.lineno <= call.lineno <= prepare.end_lineno]
    assert len(prepare_attach) == 1
    assert any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "capture_engine"
        for call in ast.walk(prepare)
    )

    prepare_text = Path(__file__).resolve().parents[1] / "app" / "launch.py"
    source_lines = prepare_text.read_text(encoding="utf-8").splitlines()
    attach_line = prepare_attach[0].lineno
    job_assignment_lines = [
        node.lineno for node in prepare.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "job" for target in node.targets)
    ]
    job_assignments = [
        node for node in prepare.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "job" for target in node.targets)
    ]
    assert len(job_assignments) == 1 and isinstance(job_assignments[0].value, ast.Dict)
    job_fields = {
        key.value: value for key, value in zip(job_assignments[0].value.keys, job_assignments[0].value.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert isinstance(job_fields.get("generation_engine_snapshot"), ast.Name)
    assert job_fields["generation_engine_snapshot"].id == "engine_snapshot"
    registry_lines = [
        node.lineno for node in prepare.body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "_jobs"
            for target in node.targets
        )
    ]
    thread_start_lines = [call.lineno for call in _attr_calls(prepare, "start")]
    assert job_assignment_lines and attach_line < min(job_assignment_lines)
    assert registry_lines and thread_start_lines
    assert attach_line < min(job_assignment_lines) < min(registry_lines) < min(thread_start_lines)
    assert "engine_snapshot" in source_lines[attach_line - 1]

    try_start = [call for call in ast.walk(run) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "try_start"]
    enhancement = [call for call in ast.walk(run) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_apply_deferred_generation_preparation"]
    assert try_start and enhancement
    run_attach = [call for call in _attr_calls(run, "attach_native") if run.lineno <= call.lineno <= run.end_lineno]
    assert len(run_attach) == 1
    execute_calls = _attr_calls(run, "execute")
    assert len(execute_calls) == 2
    native_dispatch = [max(execute_calls, key=lambda call: call.lineno)]
    assert max(call.lineno for call in try_start) < native_dispatch[0].lineno
    assert max(call.lineno for call in enhancement) < native_dispatch[0].lineno
    first_wgp_reference = min(node.lineno for node in ast.walk(run) if isinstance(node, ast.Name) and node.id == "wgp")
    assert max(call.lineno for call in try_start) < run_attach[0].lineno
    assert max(call.lineno for call in enhancement) < run_attach[0].lineno < first_wgp_reference
    assert native_dispatch[0].lineno < first_wgp_reference
    assert any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "generation_slot"
        for node in ast.walk(run) if isinstance(node, ast.With)
        for item in node.items for call in ast.walk(item.context_expr)
    )
