"""Opt-in ComfyUI library, durable submission receipts and execution adapter.

ComfyUI remains a separate process. This module owns no model or GPU; launch.py
supplies Maestro's existing queue, generation slot and approved media resolver.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import Callable
import uuid

from .comfyui_client import ComfyClient, ComfyError
from .comfyui_models import discover_comfyui_installations, scan_comfyui_models
from .comfyui_workflows import (
    apply_bindings, binding_choices, builtin_templates, normalize_workflow, preflight_workflow,
    validate_bindings,
)

_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_TERMINAL = {"completed", "failed", "cancelled"}
_PRIVATE_KEY = re.compile(r"(?i)(?:token|password|secret|api[\s_-]?key|authorization|cookie)")
_NATIVE_FIELDS = {
    "prompt", "negative_prompt", "seed", "width", "height", "num_inference_steps",
    "guidance_scale", "video_length", "fps", "image_start", "image_end",
    "image_refs", "image_prompt", "video_guide", "audio_guide", "video_source",
    "audio_frame_offset", "audio_start_seconds", "duration_seconds",
}


class ServiceError(ValueError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".comfyui-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path: Path, default):
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ServiceError(409, "The saved ComfyUI record is too large.")
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return deepcopy(default)
    except (OSError, ValueError) as error:
        if isinstance(error, ServiceError):
            raise
        raise ServiceError(409, "A saved ComfyUI record could not be read; preserve it and review the installation.") from None


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ServiceError(422, "Invalid ComfyUI record identifier.")
    return value


def _submission_id(value: str) -> str:
    try:
        parsed = str(uuid.UUID(value))
        if parsed != value:
            raise ValueError
        return parsed
    except (ValueError, TypeError, AttributeError):
        raise ServiceError(422, "client_submission_id must be a canonical UUID.") from None


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _redact(value):
    if isinstance(value, dict):
        return {key: ("[redacted]" if _PRIVATE_KEY.search(str(key)) else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _close_client(client) -> None:
    close = getattr(client, "close", None)
    if close is not None:
        close()


def _safe_output_records(history: dict) -> list[dict]:
    result, seen = [], set()
    for output in (history.get("outputs") or {}).values():
        if not isinstance(output, dict):
            continue
        for key in ("images", "videos", "gifs", "audio", "files"):
            records = output.get(key, [])
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                name, folder, kind = record.get("filename"), record.get("subfolder", ""), record.get("type", "output")
                if not isinstance(name, str) or Path(name).name != name or any(c in name for c in "/\\:") or any(ord(c) < 32 for c in name):
                    continue
                if Path(name).suffix.lower() not in _MEDIA or kind not in {"output", "temp"}:
                    continue
                if not isinstance(folder, str) or any(c in folder for c in "\\:") or any(ord(c) < 32 for c in folder) or any(part in {".", ".."} for part in folder.split("/")) or folder.startswith("/"):
                    continue
                identity = (name, folder, kind)
                if identity not in seen:
                    seen.add(identity)
                    result.append({"filename": name, "subfolder": folder, "type": kind})
                if len(result) >= 64:
                    return result
    return result


def _queue_entries(queue: dict) -> list:
    return [entry for key in ("queue_running", "queue_pending") for entry in (queue.get(key) or []) if isinstance(entry, (list, tuple)) and len(entry) >= 4]


def _matches_submission(entry, submission: str) -> bool:
    extra = entry[3] if isinstance(entry, (list, tuple)) and len(entry) > 3 else None
    return isinstance(extra, dict) and extra.get("maestro_submission_id") == submission and extra.get("client_id") == submission


class ComfyService:
    def __init__(
        self, app_root, *, submit: Callable[[dict], dict] | None = None,
        link_folders: Callable[[list[str]], list[str]] | None = None,
        known_assets: Callable[[], list[dict]] | None = None,
        require_job_access: Callable[[str], None] | None = None,
        principal_id: Callable[[], str | None] | None = None,
        authorize_media: Callable[[str], str] | None = None,
        is_owner: Callable[[], bool] | None = None,
        resume: Callable[[dict], dict] | None = None,
        job_exists: Callable[[str], bool] | None = None,
        client_factory=ComfyClient,
    ):
        self.app_root = Path(app_root).resolve()
        self.directory = self.app_root / "settings" / "comfyui"
        self.submit_job = submit
        self.link_folders = link_folders
        self.known_assets = known_assets or (lambda: [])
        self.require_job_access = require_job_access or (lambda _job_id: None)
        self.principal_id = principal_id or (lambda: None)
        self.authorize_media = authorize_media or (lambda value: value)
        self.is_owner = is_owner or (lambda: self.principal_id() is None)
        self.resume_job = resume
        self.job_exists = job_exists or (lambda _job_id: True)
        self.client_factory = client_factory
        self.lock = threading.RLock()
        self.connection = {"connected": False, "connection_error": None}
        self._pending = None

    def _config(self) -> dict:
        default = {"enabled": False, "endpoint": "http://127.0.0.1:8188", "execution_timeout": 7200, "dedicated": False, "routes": {}}
        return {**default, **_read_json(self.directory / "settings.json", {})}

    def _client(self, config=None):
        config = config or self._config()
        if not config["enabled"]:
            raise ServiceError(409, "Enable ComfyUI integration in Settings first.")
        return self.client_factory(config["endpoint"])

    def status(self) -> dict:
        with self.lock:
            config = self._config()
            pending = []
            if config["enabled"]:
                for row in self.pending_receipts():
                    if self.is_owner() or row.get("owner_id") == self.principal_id():
                        pending.append({key: row.get(key) for key in ("submission_id", "job_id", "state", "error")})
            return _redact({**config, **self.connection, "pending_submissions": pending})

    def configure(self, payload: dict) -> dict:
        with self.lock:
            if set(payload) - {"enabled", "endpoint", "execution_timeout", "dedicated", "routes"}:
                raise ServiceError(422, "Unknown ComfyUI setting.")
            config = {**self._config(), **payload}
            if type(config["enabled"]) is not bool or type(config["dedicated"]) is not bool:
                raise ServiceError(422, "ComfyUI enabled and dedicated settings must be booleans.")
            if type(config["execution_timeout"]) is not int or not 30 <= config["execution_timeout"] <= 14400:
                raise ServiceError(422, "ComfyUI execution timeout must be between 30 and 14400 seconds.")
            probe = self.client_factory(config["endpoint"])
            _close_client(probe)
            if config.get("routes") != self._config().get("routes"):
                raise ServiceError(422, "Use the route-mapping endpoint to configure verified workflows.")
            if any(row.get("state") not in _TERMINAL for row in self._receipts()):
                raise ServiceError(409, "Wait for the owned ComfyUI work to finish before changing its connection.")
            _atomic_json(self.directory / "settings.json", config)
            self.connection = {"connected": False, "connection_error": None}
            return self.status()

    def test_connection(self) -> dict:
        client = None
        try:
            client = self.client_factory(self._config()["endpoint"])
            stats = client.system_stats()
            info = client.object_info()
            if not isinstance(stats, dict) or not isinstance(info, dict) or not info:
                raise ServiceError(409, "The endpoint did not return ComfyUI node and system information.")
            result = {"connected": True, "connection_error": None, "system": _redact(stats), "node_count": len(info)}
        except (ComfyError, ValueError) as error:
            result = {"connected": False, "connection_error": str(error), "error": str(error)}
        finally:
            if client is not None:
                _close_client(client)
        with self.lock:
            self.connection = {key: result.get(key) for key in ("connected", "connection_error")}
        return result

    def discover(self, paths=None) -> dict:
        return {"installations": discover_comfyui_installations(self.app_root, extra_roots=paths)}

    def discover_connections(self) -> dict:
        """Explicitly probe the saved endpoint and two loopback defaults.

        No LAN scanning, settings mutation, server launch or automatic enable.
        A refused candidate does not prevent checking the remaining addresses.
        """
        endpoints = list(dict.fromkeys((self._config()["endpoint"], "http://127.0.0.1:8188", "http://127.0.0.1:8189")))
        candidates = []
        for endpoint in endpoints:
            client = None
            row = {"endpoint": endpoint, "connected": False}
            try:
                client = self.client_factory(endpoint, timeout=3)
                stats, info = client.system_stats(), client.object_info()
                if not isinstance(stats, dict) or not isinstance(info, dict) or not info:
                    raise ServiceError(502, "The endpoint did not advertise ComfyUI node schemas.")
                row.update(connected=True, node_count=len(info))
            except (ComfyError, ValueError):
                row["error"] = "ComfyUI was not reachable at this address."
            finally:
                if client is not None:
                    _close_client(client)
            candidates.append(row)
        return {"candidates": candidates}

    def scan(self, paths: list[str]) -> dict:
        return scan_comfyui_models(paths, known_assets=self.known_assets())

    def link(self, paths: list[str]) -> dict:
        if not self.link_folders:
            raise ServiceError(409, "Model folder linking is unavailable in this installation.")
        return {"linked_paths": self.link_folders(paths)}

    def workflows(self) -> dict:
        saved = []
        for path in sorted((self.directory / "workflows").glob("*.json")):
            saved.append(_redact(_read_json(path, {})))
        return {"workflows": saved, "templates": builtin_templates()}

    def workflow_options(self, workflow_id: str) -> dict:
        value = self._workflow(workflow_id)
        client = self._client()
        try:
            choices = binding_choices(value["workflow"], value["bindings"], client.object_info())
        finally:
            _close_client(client)
        return {"workflow_id": workflow_id, "choices": choices}

    def _workflow(self, workflow_id: str) -> dict:
        _identifier(workflow_id)
        for template in builtin_templates():
            if template["id"] == workflow_id:
                return {**deepcopy(template), "template": True}
        value = _read_json(self.directory / "workflows" / f"{workflow_id}.json", None)
        if value is None:
            raise ServiceError(404, "ComfyUI workflow not found.")
        return value

    def save_workflow(self, payload: dict) -> dict:
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ServiceError(422, "Name the workflow using 1–120 characters.")
        if payload.get("kind") not in {"image", "video", "audio"}:
            raise ServiceError(422, "Workflow kind must be image, video or audio.")
        graph = normalize_workflow(payload.get("workflow"))
        bindings = deepcopy(payload.get("bindings", []))
        binding_errors = validate_bindings(graph, bindings)
        if binding_errors:
            raise ServiceError(422, "; ".join(binding_errors)[:1500])
        value = {"id": uuid.uuid4().hex, "name": name.strip(), "kind": payload["kind"], "description": str(payload.get("description") or "")[:1000], "workflow": graph, "bindings": bindings, "created_at": time.time()}
        with self.lock:
            _atomic_json(self.directory / "workflows" / f"{value['id']}.json", value)
        return _redact(value)

    def delete_workflow(self, workflow_id: str) -> dict:
        value = self._workflow(workflow_id)
        if value.get("template"):
            raise ServiceError(409, "Built-in templates cannot be deleted.")
        with self.lock:
            if any(row.get("workflow_id") == workflow_id and row.get("state") not in _TERMINAL for row in self._receipts()):
                raise ServiceError(409, "This workflow still has owned active work.")
            (self.directory / "workflows" / f"{workflow_id}.json").unlink()
        return {"deleted": True}

    def validate_workflow(self, workflow_id: str, inputs: dict | None = None) -> dict:
        value = self._workflow(workflow_id)
        graph = value["workflow"]
        if inputs is not None:
            graph = apply_bindings(graph, value["bindings"], inputs)
        client = self._client()
        try:
            info = deepcopy(client.object_info())
            # Media already uploaded to Maestro is not yet in ComfyUI's file
            # dropdown. Permit that pending filename for this read-only check;
            # execution uploads it and then validates the real returned name.
            if inputs is not None:
                for binding in value["bindings"]:
                    if binding["type"] not in {"image", "video", "audio"} or binding["name"] not in inputs:
                        continue
                    node = graph[binding["node_id"]]
                    definitions = (info.get(node["class_type"], {}).get("input") or {})
                    for group in ("required", "optional"):
                        definition = definitions.get(group, {}).get(binding["input"])
                        if isinstance(definition, (list, tuple)) and definition and isinstance(definition[0], list):
                            definition[0].append(inputs[binding["name"]])
            check = preflight_workflow(graph, info)
        finally:
            _close_client(client)
        # Newly uploaded Maestro media will be transferred to ComfyUI at run
        # time. Its LoadImage combo will list only ComfyUI's existing files.
        media_inputs = {(item["node_id"], item["input"]) for item in value["bindings"] if item["type"] in {"image", "video", "audio"}}
        if media_inputs and inputs is not None:
            check["pending_media_upload"] = True
        return check

    def save_route(self, payload: dict) -> dict:
        model_type = payload.get("model_type")
        if not isinstance(model_type, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", model_type):
            raise ServiceError(422, "Select a valid Maestro model identifier.")
        if payload.get("enabled") is False:
            with self.lock:
                config = self._config()
                previous = config["routes"].get(model_type)
                if previous:
                    previous["enabled"] = False
                    _atomic_json(self.directory / "settings.json", config)
                    return deepcopy(previous)
        value = self._workflow(payload.get("workflow_id", ""))
        if value["kind"] != payload.get("kind"):
            raise ServiceError(422, "The route kind must match the workflow output.")
        mappings = payload.get("field_bindings", {})
        if not isinstance(mappings, dict):
            raise ServiceError(422, "Provide native-field to workflow-input mappings.")
        names = {binding["name"] for binding in value["bindings"]}
        if not all(isinstance(target, str) for target in mappings.values()):
            raise ServiceError(422, "Workflow input names must be strings.")
        if len(set(mappings.values())) != len(mappings):
            raise ServiceError(422, "Map each native control to a distinct workflow input; controls cannot overwrite one another.")
        for source, target in mappings.items():
            base = source.split(".")[0] if isinstance(source, str) else ""
            if base not in _NATIVE_FIELDS or not re.fullmatch(r"[a-z_]+(?:\.\d+)?", source) or target not in names:
                raise ServiceError(422, "The route contains an unsupported field mapping.")
            binding = next(item for item in value["bindings"] if item["name"] == target)
            media_type = "audio" if base == "audio_guide" else "video" if base in {"video_guide", "video_source"} else "image" if base in {"image_start", "image_end", "image_refs", "image_prompt"} else None
            if media_type and binding["type"] != media_type:
                raise ServiceError(422, f"Map {source} to a {media_type} upload binding.")
            if not media_type and base not in {"prompt", "negative_prompt"} and binding["type"] not in {"integer", "number", "choice"}:
                raise ServiceError(422, f"Map {source} to a numeric workflow input.")
        if "prompt" not in mappings:
            raise ServiceError(422, "A native route must bind Maestro's prompt field.")
        required_fields = {"prompt", "seed"}
        if value["kind"] in {"image", "video"}:
            required_fields.update({"width", "height", "num_inference_steps", "guidance_scale"})
        if value["kind"] == "video":
            required_fields.update({"video_length", "fps"})
        if payload.get("enabled") and not required_fields.issubset(mappings):
            missing = ", ".join(sorted(required_fields - set(mappings)))
            raise ServiceError(422, "Map the visible Maestro generation controls before enabling this route: " + missing)
        defaults = payload.get("input_defaults", {})
        if not isinstance(defaults, dict):
            raise ServiceError(422, "Workflow input defaults must be an object.")
        representative = deepcopy(defaults)
        for source, name in mappings.items():
            binding = next(item for item in value["bindings"] if item["name"] == name)
            representative[name] = binding.get("default", binding.get("min", "A test prompt" if binding["type"] in {"string", "image", "video", "audio"} else False if binding["type"] == "boolean" else 1))
        check = self.validate_workflow(value["id"], representative)
        if not check.get("valid"):
            raise ServiceError(409, "Resolve this workflow's node/model requirements before enabling a native route: " + "; ".join(check.get("errors", []))[:700])
        route = {"workflow_id": value["id"], "enabled": payload.get("enabled") is True, "field_bindings": deepcopy(mappings), "input_defaults": deepcopy(defaults), "kind": value["kind"], "name": value["name"]}
        with self.lock:
            config = self._config()
            config["routes"][model_type] = route
            _atomic_json(self.directory / "settings.json", config)
        return route

    def native_request(self, body: dict, *, frozen_route: dict | None = None) -> dict | None:
        explicit = body.get("generation_engine")
        if explicit not in {None, "native", "comfyui"}:
            raise ServiceError(422, "Generation engine must be native or comfyui.")
        route = frozen_route or self._config()["routes"].get(str(body.get("model_type") or ""))
        if explicit == "native" or not route or not route.get("enabled"):
            if explicit == "comfyui":
                raise ServiceError(409, "Configure a verified ComfyUI workflow mapping for this model first.")
            return None
        if not self._config()["enabled"]:
            raise ServiceError(409, "The configured ComfyUI engine is disabled; choose Native or enable it.")
        if body.get("comfyui_workflow_id") not in {None, route["workflow_id"]}:
            raise ServiceError(409, "This clip used a different workflow. Select its saved workflow or review the current mapping before generating.")
        kind = body.get("generation_mode") or ("image" if body.get("image_mode") == 1 else "video")
        if kind != route["kind"]:
            raise ServiceError(422, "This workflow mapping does not support the requested media mode.")
        face_refiner = body.get("face_refiner")
        if body.get("activated_loras") or (isinstance(face_refiner, dict) and face_refiner.get("enabled")):
            raise ServiceError(422, "This ComfyUI mapping does not implement native LoRAs or face refinement. Configure those operations inside its workflow.")
        if any(body.get(key) for key in ("spatial_upsampling", "temporal_upsampling")) or any(len(body.get(key) or []) > 1 for key in ("h3_window_prompts", "ltx_window_prompts")):
            raise ServiceError(422, "This ComfyUI mapping supports a single workflow execution. Configure finishing or window continuation inside that workflow.")
        if 0 < int(body.get("sliding_window_size") or 0) < int(body.get("video_length") or 0):
            raise ServiceError(422, "This request needs native sliding-window continuation. Use Native, or run a ComfyUI workflow that implements its own continuation.")
        values = deepcopy(route.get("input_defaults", {}))
        workflow_bindings = {item["name"]: item for item in self._workflow(route["workflow_id"])["bindings"]}
        for source, name in route["field_bindings"].items():
            field, _, index = source.partition(".")
            value = body.get(field)
            if field == "fps" and value is None:
                value = float(body.get("force_fps") or 24)
            if field == "audio_start_seconds":
                fps = float(body.get("fps") or body.get("force_fps") or 24)
                value = float(body.get("audio_frame_offset") or 0) / fps
            if field == "duration_seconds" and value is None:
                fps = float(body.get("fps") or body.get("force_fps") or 24)
                value = float(body.get("video_length") or 0) / fps
            if field in {"width", "height"} and value is None:
                match = re.fullmatch(r"(\d+)[xX](\d+)", str(body.get("resolution") or ""))
                if match:
                    value = int(match[1 if field == "width" else 2])
            if index and value is not None:
                if not isinstance(value, list) or int(index) >= len(value):
                    raise ServiceError(422, f"The workflow needs reference {int(index) + 1} from {field}.")
                value = value[int(index)]
            if isinstance(value, dict) and "path" in value:
                value = value["path"]
            binding = workflow_bindings[name]
            if isinstance(value, str) and binding["type"] == "integer" and re.fullmatch(r"-?\d+", value):
                value = int(value)
            if isinstance(value, str) and binding["type"] == "number":
                try:
                    value = float(value)
                except ValueError:
                    raise ServiceError(422, f"Provide a numeric value for {field}.") from None
            if value is not None:
                values[name] = value
        mapped = {source.split(".")[0] for source in route["field_bindings"]}
        if kind == "audio" and body.get("duration_seconds") and "duration_seconds" not in mapped:
            raise ServiceError(422, "Bind the selected audio duration to this workflow; it will not be silently ignored.")
        if body.get("audio_guide") and body.get("audio_frame_offset") and not mapped.intersection({"audio_frame_offset", "audio_start_seconds"}):
            raise ServiceError(422, "This audio-driven clip needs its source offset mapped to the workflow; the beginning of the song will not be substituted.")
        for field in ("image_start", "image_end", "image_refs", "image_prompt", "video_guide", "audio_guide", "video_source"):
            if body.get(field) and field not in mapped:
                raise ServiceError(422, f"The selected workflow has no binding for {field}; the reference will not be silently ignored.")
            if isinstance(body.get(field), list) and body[field]:
                indices = {int(source.split(".")[1]) for source in route["field_bindings"] if source.startswith(field + ".")}
                if indices != set(range(len(body[field]))):
                    raise ServiceError(422, f"Bind every ordered {field} reference; partial reference mappings are not supported.")
        return {"workflow_id": route["workflow_id"], "inputs": values, "native_model_type": body.get("model_type"), "native_route": deepcopy(route)}

    def _receipt_path(self, submission: str) -> Path:
        return self.directory / "receipts" / f"{_submission_id(submission)}.json"

    def _receipts(self) -> list[dict]:
        return [_read_json(path, {}) for path in (self.directory / "receipts").glob("*.json")]

    def pending_receipts(self) -> list[dict]:
        with self.lock:
            if self._pending is None:
                self._pending = {row["submission_id"] for row in self._receipts() if row.get("state") not in _TERMINAL}
            return [_read_json(self._receipt_path(submission), {}) for submission in self._pending]

    def prepare(self, workflow_id: str, payload: dict, *, native_model_type=None) -> dict:
        if not self._config()["enabled"]:
            raise ServiceError(409, "Enable ComfyUI integration in Settings first.")
        submission = _submission_id(payload.get("client_submission_id"))
        value = self._workflow(workflow_id)
        inputs = deepcopy(payload.get("inputs", {}))
        if not isinstance(inputs, dict):
            raise ServiceError(422, "Workflow inputs must be an object.")
        for binding in value["bindings"]:
            if binding["type"] in {"image", "video", "audio"} and binding["name"] in inputs:
                inputs[binding["name"]] = self.authorize_media(inputs[binding["name"]])
        apply_bindings(value["workflow"], value["bindings"], inputs)
        workspace = payload.get("workspace") or "default"
        if not isinstance(workspace, str) or len(workspace) > 120 or workspace in {".", ".."} or any(c in workspace for c in "/\\:\x00"):
            raise ServiceError(422, "Use an existing Maestro workspace name.")
        return {"workflow_id": workflow_id, "workflow": deepcopy(value), "inputs": deepcopy(inputs), "client_submission_id": submission, "workspace": workspace, "native_model_type": native_model_type, "endpoint": self._config()["endpoint"], "execution_timeout": self._config()["execution_timeout"]}

    def reserve(self, prepared: dict, job_id: str, *, owner_id=None) -> tuple[dict, bool]:
        submission = prepared["client_submission_id"]
        identity = _digest({key: prepared[key] for key in ("workflow_id", "workflow", "inputs", "workspace", "native_model_type", "endpoint")})
        with self.lock:
            path = self._receipt_path(submission)
            prior = _read_json(path, None)
            if prior is not None:
                if prior["request_digest"] != identity:
                    raise ServiceError(409, "This submission ID already belongs to different workflow inputs.")
                self._require_receipt_access(prior)
                return prior, False
            receipt = {"submission_id": submission, "job_id": job_id, "workflow_id": prepared["workflow_id"], "request_digest": identity, "state": "prepared", "prompt_id": None, "owner_id": owner_id or self.principal_id(), "endpoint": prepared["endpoint"], "prepared": deepcopy(prepared), "created_at": time.time(), "updated_at": time.time()}
            _atomic_json(path, receipt)
            if self._pending is not None:
                self._pending.add(submission)
            return receipt, True

    def _update_receipt(self, submission: str, **changes) -> dict:
        with self.lock:
            path = self._receipt_path(submission)
            value = _read_json(path, None)
            if value is None:
                raise ServiceError(404, "ComfyUI submission receipt not found.")
            value.update(changes, updated_at=time.time())
            _atomic_json(path, value)
            if self._pending is not None:
                if value["state"] in _TERMINAL:
                    self._pending.discard(submission)
                else:
                    self._pending.add(submission)
            return value

    def run_workflow(self, workflow_id: str, payload: dict) -> dict:
        if self.submit_job is None:
            raise ServiceError(409, "Maestro's workflow queue is unavailable.")
        prepared = self.prepare(workflow_id, payload)
        return self.submit_job(prepared)

    def submission_receipt(self, submission: str) -> dict:
        value = _read_json(self._receipt_path(submission), None)
        if value is None:
            raise ServiceError(404, "ComfyUI submission receipt not found.")
        self._require_receipt_access(value)
        return {**{key: value.get(key) for key in ("submission_id", "job_id", "state", "prompt_id", "error")}, "status": value["state"], "client_submission_id": value["submission_id"]}

    def _require_receipt_access(self, value: dict) -> None:
        if not self.is_owner() and (not value.get("owner_id") or value["owner_id"] != self.principal_id()):
            raise ServiceError(404, "ComfyUI submission receipt not found.")
        try:
            self.require_job_access(value["job_id"])
        except Exception as error:
            if self.job_exists(value["job_id"]) or getattr(error, "status_code", getattr(error, "status", None)) != 404:
                raise
        # The durable owner remains authoritative if the in-memory job was
        # lost during a backend restart. Do not turn recovery into a new grant.

    def resume_submission(self, submission: str) -> dict:
        value = _read_json(self._receipt_path(submission), None)
        if value is None:
            raise ServiceError(404, "ComfyUI submission receipt not found.")
        self._require_receipt_access(value)
        if value["state"] in _TERMINAL:
            return self.submission_receipt(submission)
        if value["state"] == "prepared":
            raise ServiceError(409, "This saved intent was never dispatched. It cannot be replayed during recovery; cancel it and explicitly create a new request.")
        if self.resume_job is None:
            raise ServiceError(409, "The workflow monitor is unavailable.")
        return self.resume_job(deepcopy(value))

    def abandon_submission(self, submission: str) -> dict:
        with self.lock:
            value = _read_json(self._receipt_path(submission), None)
            if value is None:
                raise ServiceError(404, "ComfyUI submission receipt not found.")
            self._require_receipt_access(value)
            if value["state"] != "prepared":
                raise ServiceError(409, "Only an undispatched intent can be dismissed. Reconnect to existing remote work instead.")
            if self.job_exists(value["job_id"]):
                raise ServiceError(409, "Cancel this queued job using Maestro's queue.")
            self._update_receipt(submission, state="cancelled", error=None)
            return self.submission_receipt(submission)

    def prepared_submission(self, submission: str) -> dict:
        value = _read_json(self._receipt_path(submission), None)
        if value is None or not isinstance(value.get("prepared"), dict):
            raise ServiceError(409, "The saved workflow intent is missing its frozen request.")
        return deepcopy(value["prepared"])

    def cancel_prepared(self, submission: str) -> None:
        with self.lock:
            value = _read_json(self._receipt_path(submission), None)
            if value and value.get("state") not in _TERMINAL:
                changes = {"cancel_requested": True}
                if value.get("state") == "prepared":
                    changes["state"] = "cancelled"
                self._update_receipt(submission, **changes)

    def fail_prepared(self, submission: str, error: str) -> None:
        with self.lock:
            value = _read_json(self._receipt_path(submission), None)
            if value and value.get("state") == "prepared":
                self._update_receipt(submission, state="failed", error=error[:800])

    def discard_cancelled_outputs(self, submission: str, out_dir: Path) -> None:
        """Cancellation won after publication; remove only this receipt's files."""
        receipt = _read_json(self._receipt_path(submission), {})
        for name in receipt.get("output_files", []):
            if Path(name).name != name or not name.startswith("comfyui-" + submission + "-"):
                raise ServiceError(409, "The saved output identity needs review.")
            path = Path(out_dir) / name
            path.unlink(missing_ok=True)
            path.with_suffix(".meta.json").unlink(missing_ok=True)
        self._update_receipt(submission, state="cancelled", output_files=[])

    def assert_native_idle(self) -> None:
        """Do not load a native GPU model beside uncertain remote execution."""
        for value in self.pending_receipts():
            if value.get("state") in {"dispatching", "running", "uncertain"}:
                client = self.client_factory(value["endpoint"])
                try:
                    entries = _queue_entries(client.queue())
                finally:
                    _close_client(client)
                if any(str(entry[1]) == value.get("prompt_id") or _matches_submission(entry, value["submission_id"]) for entry in entries):
                    raise ServiceError(409, "Owned ComfyUI work is still active. Wait for it before starting the native engine.")
                if value.get("state") in {"uncertain", "dispatching"}:
                    raise ServiceError(409, "An owned ComfyUI submission needs reconciliation before the native engine can start.")
        config = self._config()
        if config["enabled"] and config["dedicated"]:
            client = self.client_factory(config["endpoint"])
            try:
                if _queue_entries(client.queue()):
                    raise ServiceError(409, "The dedicated ComfyUI server is busy. Wait before switching to Native.")
                client.release_memory()
            finally:
                _close_client(client)

    def _reconcile(self, client, submission: str) -> str | None:
        candidates = {str(entry[1]) for entry in _queue_entries(client.queue()) if _matches_submission(entry, submission)}
        history = client.history()
        if isinstance(history, dict):
            for prompt_id, row in history.items():
                if isinstance(row, dict) and _matches_submission(row.get("prompt"), submission):
                    candidates.add(str(prompt_id))
        if len(candidates) > 1:
            raise ServiceError(409, "Multiple ComfyUI prompts share this owned submission ID; no work was resubmitted.")
        return next(iter(candidates), None)

    def execute(self, prepared: dict, out_dir: Path, *, cancelled: Callable[[], bool], progress: Callable[[str], None], resolve_media: Callable[[str, str], Path]) -> list[str]:
        submission = prepared["client_submission_id"]
        receipt = _read_json(self._receipt_path(submission), None)
        if receipt is None:
            raise ServiceError(409, "Missing durable workflow intent; no generation was submitted.")
        if receipt["state"] == "completed":
            paths = receipt.get("output_files") or []
            if paths and all((out_dir / name).is_file() for name in paths):
                return paths
            raise ServiceError(409, "This workflow completed previously but its published media is missing; no generation was repeated.")
        if receipt["state"] == "cancelled":
            return []
        if receipt["state"] == "failed":
            raise ServiceError(409, receipt.get("error") or "This workflow previously failed; no generation was repeated.")
        client = self.client_factory(prepared["endpoint"])
        workflow = prepared["workflow"]
        values = deepcopy(prepared["inputs"])
        started = time.monotonic()
        dispatched = receipt["state"] != "prepared"
        prompt_id = receipt.get("prompt_id")
        remote_terminal = False
        try:
            if not dispatched:
                for binding in workflow["bindings"]:
                    name = binding["name"]
                    if binding["type"] in {"image", "video", "audio"} and name in values:
                        source = resolve_media(values[name], prepared["workspace"])
                        if source.stat().st_size > 256 * 1024 * 1024:
                            raise ServiceError(422, "The workflow reference exceeds the 256 MiB transfer limit.")
                        upload = client.upload_image(source.read_bytes(), source.name)
                        uploaded_name = upload.get("name")
                        subfolder = upload.get("subfolder", "")
                        if not isinstance(uploaded_name, str) or not uploaded_name or "/" in uploaded_name or "\\" in uploaded_name:
                            raise ServiceError(409, "ComfyUI returned an invalid uploaded reference name.")
                        if not isinstance(subfolder, str) or any(c in subfolder for c in "\\:") or subfolder.startswith("/") or any(part in {".", ".."} for part in subfolder.split("/")):
                            raise ServiceError(409, "ComfyUI returned an invalid reference subfolder.")
                        values[name] = f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name
                graph = apply_bindings(workflow["workflow"], workflow["bindings"], values)
                check = preflight_workflow(graph, client.object_info())
                if not check["valid"]:
                    raise ServiceError(422, "ComfyUI workflow preflight failed: " + json.dumps(check.get("errors", []))[:700])
                if cancelled():
                    self._update_receipt(submission, state="cancelled")
                    return []
                # Persist intent before the only prompt POST. No automatic replay.
                self._update_receipt(submission, state="dispatching", prepared=deepcopy(prepared), out_dir=str(Path(out_dir).resolve()), dispatched_at=time.time(), applied_workflow_digest=_digest(graph), applied_workflow=_redact(graph), effective_inputs=_redact(values))
                try:
                    response = client.submit(graph, client_id=submission, submission_id=submission)
                    prompt_id = response.get("prompt_id")
                    if not isinstance(prompt_id, str) or not prompt_id:
                        raise ServiceError(409, "ComfyUI did not return a usable prompt receipt.")
                    self._update_receipt(submission, state="running", prompt_id=prompt_id)
                except Exception as error:
                    self._update_receipt(submission, state="uncertain", error="Prompt submission needs read-only reconciliation.")
                    if isinstance(error, ComfyError) and not getattr(error, "ambiguous", True):
                        self._update_receipt(submission, state="failed", error=str(error))
                        raise
                    prompt_id = self._reconcile(client, submission)
                    if not prompt_id:
                        raise ServiceError(409, "The submission outcome is uncertain. Maestro saved its intent and did not resubmit. Reconnect to ComfyUI and review this receipt.") from None
                    self._update_receipt(submission, state="running", prompt_id=prompt_id, error=None)
            elif not prompt_id:
                prompt_id = self._reconcile(client, submission)
                if not prompt_id:
                    raise ServiceError(409, "The previous ComfyUI submission is not reconciled; no duplicate was submitted.")
                self._update_receipt(submission, state="running", prompt_id=prompt_id, error=None)

            cancel_deleted = False
            while time.monotonic() - started < prepared["execution_timeout"]:
                records = client.history(prompt_id)
                row = records.get(prompt_id) if isinstance(records, dict) else None
                if isinstance(row, dict):
                    if isinstance(row.get("prompt"), (list, tuple)) and not _matches_submission(row["prompt"], submission):
                        raise ServiceError(409, "The remote history does not match this workflow's ownership identifiers.")
                    status = row.get("status") or {}
                    if status.get("status_str") in {"error", "failed"}:
                        remote_terminal = True
                        details = ""
                        for message in status.get("messages", []):
                            if isinstance(message, (list, tuple)) and len(message) > 1 and message[0] == "execution_error" and isinstance(message[1], dict):
                                node_error = message[1]
                                details = " Node " + str(node_error.get("node_id", "?"))[:40] + " (" + str(node_error.get("node_type", "unknown"))[:80] + "): " + str(node_error.get("exception_type") or "execution error")[:120]
                                break
                        raise ServiceError(422, "ComfyUI reported a workflow execution error." + details + " Inspect its node error in ComfyUI.")
                    if status.get("completed") is True or status.get("status_str") == "success":
                        remote_terminal = True
                        if cancelled():
                            self._update_receipt(submission, state="cancelled")
                            return []
                        output_records = _safe_output_records(row)
                        if not output_records:
                            raise ServiceError(422, "The workflow finished without a supported saved image, video or audio output.")
                        evidence = _read_json(self._receipt_path(submission), {})
                        elapsed = max(0.0, time.time() - evidence["dispatched_at"]) if evidence.get("dispatched_at") else time.monotonic() - started
                        paths = self._publish_outputs(client, output_records, prepared, out_dir, elapsed, evidence=evidence)
                        self._update_receipt(submission, state="completed", output_files=paths, error=None)
                        return paths
                if cancelled():
                    queue = client.queue()
                    matching_ids = [entry for key in ("queue_running", "queue_pending") for entry in queue.get(key, []) if isinstance(entry, (list, tuple)) and len(entry) > 1 and str(entry[1]) == prompt_id]
                    if any(not _matches_submission(entry, submission) for entry in matching_ids):
                        raise ServiceError(409, "Cancellation ownership could not be verified; no remote work was interrupted.")
                    pending = [entry for entry in queue.get("queue_pending", []) if isinstance(entry, (list, tuple)) and len(entry) > 1 and str(entry[1]) == prompt_id and _matches_submission(entry, submission)]
                    running = [entry for entry in queue.get("queue_running", []) if isinstance(entry, (list, tuple)) and len(entry) > 1 and str(entry[1]) == prompt_id]
                    if pending and not cancel_deleted:
                        client.delete_queued(prompt_id)
                        cancel_deleted = True
                    if not pending and not running:
                        self._update_receipt(submission, state="cancelled")
                        return []
                    progress("Cancellation requested. Waiting for this owned workflow; shared ComfyUI work will not be interrupted.")
                else:
                    progress("ComfyUI is executing the workflow.")
                time.sleep(0.75)
            self._update_receipt(submission, state="uncertain", error="The monitoring bound expired; remote work was not interrupted or resubmitted.")
            raise ServiceError(409, "The ComfyUI monitoring bound expired. Check the owned workflow before starting further GPU work.")
        except Exception as error:
            current = _read_json(self._receipt_path(submission), {})
            if current.get("state") not in _TERMINAL | {"uncertain", "dispatching"}:
                uncertain = not remote_terminal and current.get("state") == "running"
                self._update_receipt(submission, state="uncertain" if uncertain else "failed", error=str(error)[:800])
            raise
        finally:
            _close_client(client)

    def _publish_outputs(self, client, records, prepared, out_dir: Path, elapsed: float, *, evidence=None) -> list[str]:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        names, created = [], []
        try:
            for index, record in enumerate(records):
                data = client.output(**record)
                if not data:
                    raise ServiceError(422, "ComfyUI returned an empty output file.")
                suffix = Path(record["filename"]).suffix.lower()
                name = f"comfyui-{prepared['client_submission_id']}-{index + 1}{suffix}"
                destination = out_dir / name
                if destination.exists():
                    raise ServiceError(409, "A published output already occupies this workflow identity.")
                destination.write_bytes(data)
                created.append(destination)
                actual = _verify_media(destination)
                metadata = {
                    **actual, "model_type": prepared.get("native_model_type") or "comfyui_workflow",
                    "model_name": "ComfyUI · " + prepared["workflow"]["name"],
                    "generation_mode": prepared["workflow"]["kind"],
                    "generation_engine": "comfyui", "comfyui_workflow_id": prepared["workflow_id"],
                    "comfyui_workflow_digest": _digest(prepared["workflow"]["workflow"]),
                    "comfyui_workflow_name": prepared["workflow"]["name"],
                    "comfyui_applied_workflow_digest": (evidence or {}).get("applied_workflow_digest"),
                    "comfyui_applied_workflow": (evidence or {}).get("applied_workflow"),
                    "comfyui_inputs": _redact(prepared["inputs"]),
                    "comfyui_workflow": _redact(prepared["workflow"]["workflow"]),
                    "generation_time": round(elapsed, 3), "generation_time_basis": "comfyui_wall",
                    "comfyui_wall_origin": "prompt_dispatch" if (evidence or {}).get("dispatched_at") else "monitor_start",
                    "prompt": str(prepared["inputs"].get("prompt") or prepared["inputs"].get("positive_prompt") or ""),
                    "seed": prepared["inputs"].get("seed"), "created_at": time.time(),
                }
                sidecar = destination.with_suffix(".meta.json")
                _atomic_json(sidecar, metadata)
                created.append(sidecar)
                names.append(name)
            return names
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise


def _verify_media(path: Path) -> dict:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        if width < 1 or height < 1:
            raise ServiceError(422, "The generated image has invalid dimensions.")
        return {"width": width, "height": height, "resolution": f"{width}x{height}"}
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=45, check=False,
    )
    if result.returncode:
        raise ServiceError(422, "The generated media could not be verified with ffprobe.")
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise ServiceError(422, "The generated media has no usable streams.")
    metadata = {}
    for stream in streams:
        if stream.get("codec_type") == "video":
            metadata.update(width=stream.get("width"), height=stream.get("height"))
            if str(stream.get("nb_frames", "")).isdigit():
                metadata["num_frames"] = int(stream["nb_frames"])
        if stream.get("codec_type") == "audio":
            metadata["has_audio"] = True
    duration = float((info.get("format") or {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ServiceError(422, "The generated media has an invalid duration.")
    metadata["duration"] = duration
    return metadata
