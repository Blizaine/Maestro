"""CPU-only creative context/defaults shared by native API and MCP writers.

Callbacks supply the already-running registry. This module never imports a
model runtime, invokes a writer, downloads weights, or submits a render.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping

from .state import atomic_json, read_json

WORKFLOWS = {
    "video.frames": "Video · Frames",
    "video.references": "Video · References",
    "image.generate": "Image · Generate",
    "image.edit": "Image · Reference editing",
    "audio.music": "Music",
    "audio.speech": "Speech",
    "audio.sfx": "Sound effects",
}
DIRECTOR_WORKFLOWS = {"director.story", "director.music_video"}
TITLE_WORKFLOWS = {"title.card"}
_FALLBACKS = {
    "video.frames": ("ltx2_25", "minimax_h3_fused_turbo", "ltx2_22B_distilled_1_1"),
    "video.references": ("minimax_h3_ref2va_fused_turbo",),
    "image.generate": ("flux2_klein_9b", "moss_image_sft", "fibo2_dev"),
    "image.edit": ("flux2_klein_9b", "moss_image_edit", "krea2_raw_edit"),
    "audio.music": ("yue2", "ace_step_v1_5_turbo"),
    "audio.speech": ("kugelaudio_0_open", "fish_drama3_int4", "fish_drama3_small"),
    "audio.sfx": ("mmaudio_v2",),
}
_MUSIC = ("yue2", "ace_step", "heartmula", "minimax_music3")
_MODEL_ID = re.compile(r"[A-Za-z0-9_.-]{1,200}\Z")
_RESOLUTION = re.compile(r"[1-9][0-9]{1,4}x[1-9][0-9]{1,4}\Z")
_PRIVATE = re.compile(r"(?:^|_)(?:api_?key|password|secret|token|authorization|credential)(?:_|$)", re.I)


class CreativeError(ValueError):
    def __init__(self, message: str, status: int = 422):
        self.status = status
        super().__init__(message)


def normalize_defaults(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) - set(WORKFLOWS):
        raise CreativeError("Use defaults keyed by a supported Studio workflow.")
    result = {}
    for workflow, settings in value.items():
        if not isinstance(settings, dict) or set(settings) - {"model_type", "duration_seconds", "resolution"}:
            raise CreativeError(f"{workflow}: use model_type, duration_seconds and resolution only.")
        row = {}
        model = settings.get("model_type")
        if model:
            if not isinstance(model, str) or not _MODEL_ID.fullmatch(model):
                raise CreativeError(f"{workflow}: invalid model identifier.")
            row["model_type"] = model
        if "duration_seconds" in settings and settings["duration_seconds"] is not None:
            duration = settings["duration_seconds"]
            if workflow.startswith("image."):
                raise CreativeError("Image defaults do not have a duration.")
            if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 <= duration <= 3600:
                raise CreativeError(f"{workflow}: duration must be finite seconds between 0 and 3600.")
            if duration == 0 and workflow != "audio.speech":
                raise CreativeError("Only speech supports an Auto duration of 0.")
            row["duration_seconds"] = duration
        resolution = settings.get("resolution")
        if resolution:
            if not isinstance(resolution, str) or not _RESOLUTION.fullmatch(resolution):
                raise CreativeError(f"{workflow}: resolution must be WIDTHxHEIGHT.")
            row["resolution"] = resolution
        if row:
            result[workflow] = row
    return result


def compatible_workflows(model: Mapping, definition: Mapping) -> list[str]:
    mt = str(model.get("model_type") or "")
    arch = str(model.get("architecture") or definition.get("architecture") or mt)
    if model.get("world_model") or definition.get("world_model"):
        return []
    if mt.startswith("mmaudio"):
        return ["audio.sfx"]
    if mt.startswith(_MUSIC) or definition.get("group") == "music":
        return ["audio.music"]
    if definition.get("audio_only") or model.get("family") == "tts":
        return ["audio.speech"]
    image = bool(definition.get("image_outputs") or model.get("family") in {
        "flux", "flux2", "flux3", "qwen_image", "krea2", "moss_image", "fibo2", "z_image",
    })
    has_refs = bool(model.get("supports_ref_images") or definition.get("image_ref_choices") or definition.get("omni_reference"))
    if image:
        # instruction_image_edit advertises an ability, not an edit-only
        # restriction (Fibo Dev supports both). Honor native capability facts.
        overrides = definition.get("capability_overrides") or {}
        edit_only = any(definition.get(key) for key in ("moss_edit", "requires_image_reference", "at_least_one_image_ref_needed", "inpaint_only", "reference_only")) or overrides.get("text_to_image") is False
        result = [] if edit_only else ["image.generate"]
        if has_refs:
            result.append("image.edit")
        return result
    # The registry is authoritative for special edit-only models. Do not
    # select a Recast/Retake/World checkpoint for an ordinary video request.
    if not (model.get("is_t2v") or model.get("is_i2v") or arch.startswith(("minimax_h3", "ltx2", "ltxv"))):
        return []
    result = ["video.references"] if has_refs else []
    if not arch.startswith("minimax_h3_ref2va") and not definition.get("reference_only"):
        result.insert(0, "video.frames")
    return result


def _safe_params(value: Mapping) -> dict:
    # Stored example/personal prompts are never the user's new brief, and
    # old private plan/signature or enhancement flags must not be inherited.
    from .backend import _PATH_KEYS
    excluded = {"prompt", "prompts", "alt_prompt", "seed", "activated_loras", "loras_multipliers", "state", "minimax_h3_references", "voice_reference", "batch_size", "repeat_generation", "workspace", "workspace_id", "out_dir", *_PATH_KEYS}
    return {
        key: copy.deepcopy(item) for key, item in value.items()
        if isinstance(key, str) and key not in excluded and not key.startswith("_")
        and not re.fullmatch(r"audio_guide[2-6]", key)
        and not key.endswith(("_path", "_file", "_dir", "_directory"))
        and not _PRIVATE.search(key) and not key.startswith(("h3_window_plan", "h3_window_prompts", "ltx_window_plan", "ltx_window_prompts"))
    }


class CreativeService:
    def __init__(
        self, directory: Path, *, models: Callable, model_def: Callable,
        defaults: Callable, options: Callable, preferences: Callable,
        guides: Callable, session: Callable = lambda: None, resolve: Callable | None = None, clock=time.monotonic,
    ):
        self.path = Path(directory) / "creative-defaults.json"
        self.models, self.model_def, self.defaults = models, model_def, defaults
        self.options, self.preferences, self.guides = options, preferences, guides
        self.session, self.resolve, self.clock = session, resolve, clock
        self._contexts: OrderedDict[str, tuple] = OrderedDict()
        self._prepared: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.RLock()

    def model_rows(self) -> list[dict]:
        value = self.models()
        values = value.get("models", []) if isinstance(value, dict) else value
        result = []
        for model in values:
            md = self.model_def(model["model_type"]) or {}
            workflows = compatible_workflows(model, md)
            if workflows:
                result.append({**model, "workflows": workflows})
        return result

    def default_settings(self) -> dict:
        data = read_json(self.path)
        defaults = normalize_defaults(data.get("defaults", {}))
        return {
            "version": 1, "defaults": defaults,
            "workflows": [{"id": key, "label": label} for key, label in WORKFLOWS.items()],
            "director_workflows": ["director.story", "director.music_video"],
            "title_workflows": [{"id": "title.card", "label": "Title card", "requires_model": False}],
            "models": [{key: row.get(key) for key in ("model_type", "name", "workflows", "is_downloaded")} for row in self.model_rows()],
        }

    def save_defaults(self, value: Any) -> dict:
        normalized = normalize_defaults(value)
        registry = {row["model_type"]: row for row in self.model_rows()}
        for workflow, row in normalized.items():
            if row.get("model_type") and workflow not in registry.get(row["model_type"], {}).get("workflows", []):
                raise CreativeError(f"{workflow}: choose a compatible model registered on this workstation.")
        atomic_json(self.path, {"version": 1, "defaults": normalized})
        return self.default_settings()

    def _select(self, workflow: str, requested: str | None, configured: dict, rows: list[dict]) -> tuple[dict, str]:
        available = {row["model_type"]: row for row in rows if workflow in row["workflows"]}
        explicit = requested or configured.get("model_type")
        if explicit:
            if explicit not in available:
                raise CreativeError(f"{explicit}: model does not support {workflow}.")
            if available[explicit].get("is_downloaded") is not True:
                raise CreativeError(f"{explicit} is not downloaded. Choose an installed model or authorize its download first.", 409)
            return available[explicit], "requested" if requested else "agent_default"
        prefs = self.preferences() or {}
        mode = workflow.split(".", 1)[0]
        selected = (prefs.get("selected_model_per_audio_sub_mode", {}) or {}).get(workflow.split(".")[1]) if mode == "audio" else None
        selected = selected or (prefs.get("selected_model_per_mode", {}) or {}).get(mode)
        companions = []
        if isinstance(selected, str) and selected.startswith("minimax_h3"):
            companions.append(selected.replace("minimax_h3_ref2va", "minimax_h3", 1) if workflow == "video.frames" else selected.replace("minimax_h3", "minimax_h3_ref2va", 1) if "ref2va" not in selected else selected)
        for candidate in (selected, *companions, *_FALLBACKS[workflow], *available):
            if candidate in available and available[candidate].get("is_downloaded") is True:
                return available[candidate], "maestro_selection" if candidate == selected else "compatible_installed_fallback"
        raise CreativeError(f"No installed compatible model is available for {workflow}. Authorize a model download before creating media.", 409)

    def create_context(self, body: dict, principal_id: str) -> dict:
        from .authoring_plans import compute_studio_timeline

        if not isinstance(body, dict):
            raise CreativeError("Use a creative context object.")
        workflow = body.get("workflow")
        if workflow == "title.card":
            return self._title_context(body, principal_id)
        if not isinstance(workflow, str) or workflow not in {*WORKFLOWS, *DIRECTOR_WORKFLOWS}:
            raise CreativeError("Choose a supported workflow; get_creation_defaults lists Studio choices.")
        references = body.get("references") or []
        if not isinstance(references, list) or len(references) > 64 or any(not isinstance(ref, dict) or not isinstance(ref.get("id"), str) or not ref["id"] or len(ref["id"]) > 300 for ref in references):
            raise CreativeError("references must contain stable, nonempty IDs and explicit roles.")
        if len({ref["id"] for ref in references}) != len(references):
            raise CreativeError("Reference IDs must be unique.")
        visual_refs = any(str(ref.get("type") or ref.get("media_type") or "image").split("/")[0] in {"image", "video"} for ref in references)
        target = "video.references" if workflow in DIRECTOR_WORKFLOWS and visual_refs else "video.frames" if workflow in DIRECTOR_WORKFLOWS else workflow
        if body.get("model_type") is not None and (not isinstance(body["model_type"], str) or not _MODEL_ID.fullmatch(body["model_type"])):
            raise CreativeError("model_type must identify a model registered on this workstation.")
        configured = self.default_settings()["defaults"].get(target, {})
        model, selection = self._select(target, body.get("model_type"), configured, self.model_rows())
        mt = model["model_type"]
        md = self.model_def(mt) or {}
        options = self.options(mt) or {}
        params = _safe_params(self.defaults(mt) or {})
        params.update({key: value for key, value in configured.items() if key != "model_type"})
        supplied = body.get("params") or {}
        if not isinstance(supplied, dict) or len(supplied) > 300:
            raise CreativeError("params must be a bounded settings object.")
        params.update(copy.deepcopy(supplied))
        params["model_type"] = mt
        params["generation_mode"] = target.split(".")[0]
        params.setdefault("batch_size", 1)
        params.setdefault("repeat_generation", 1)
        params["prompt_enhancer"] = ""
        params["enhance_on_generation"] = False
        brief = body.get("brief", "")
        if not isinstance(brief, str) or len(brief) > 100_000:
            raise CreativeError("brief must be text up to 100,000 characters.")
        duration = body.get("duration_seconds", supplied.get("duration_seconds", configured.get("duration_seconds")))
        if duration is not None:
            normalize_defaults({target: {"duration_seconds": duration}})
            params["duration_seconds"] = duration
        elif target == "audio.speech" and options.get("speech_auto_duration"):
            params["duration_seconds"] = 0
        fps = float(md.get("fps") or model.get("fps") or 24)
        if workflow in DIRECTOR_WORKFLOWS:
            params["duration_seconds"] = duration if duration is not None else 30
        elif target.startswith("video.") and duration is not None:
            minimum = int(md.get("frames_minimum") or options.get("frames_minimum") or 1)
            step = int(md.get("frames_steps") or options.get("frames_steps") or 1)
            # Match the native floor-to-lattice convention; report actual time.
            desired = max(minimum, round(float(duration) * fps))
            params["video_length"] = minimum + ((desired - minimum) // step) * step
        if target.startswith("image."):
            params.update(image_mode=1, video_length=1, multi_prompts_gen_type=2)
        elif target.startswith("audio."):
            params["multi_prompts_gen_type"] = 2
        if body.get("resolution") is not None:
            normalize_defaults({target: {"resolution": body["resolution"]}})
            params["resolution"] = body["resolution"]
        context = {
            "version": 1, "context_id": str(uuid.uuid4()), "workflow": workflow,
            "model_type": mt, "model": model, "model_def": md, "options": options,
            "params": params, "brief": brief, "references": copy.deepcopy(references),
            "fps": fps, "selection_source": selection, "session_revision": self.session(),
            "author": "connected_agent", "local_writer": False,
        }
        if self.resolve is not None:
            self.resolve(context, body)
        context["timeline"] = compute_studio_timeline(context)
        if workflow.startswith("video."):
            context["params"]["duration_seconds"] = context["timeline"]["total_frames"] / context["timeline"]["fps"]
        context["guides"] = self.guides(mt, target, {**body, "references": context["references"], "params": context["params"]}, md)
        context["guides"]["reference_context"] = context.get("reference_context", "")
        context["context_revision"] = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()
        self._store_context(context, principal_id)
        # Registry definitions can contain internal disk paths or URL caches.
        # Agents receive curated capabilities/guides; the server retains full IR.
        public = copy.deepcopy(context)
        public.pop("model_def", None)
        public.pop("_input_references", None)
        if isinstance(public.get("music_analysis"), dict):
            analysis = public["music_analysis"]
            analysis["beat_count"] = len(analysis.pop("beats", []))
            analysis["downbeat_count"] = len(analysis.pop("downbeats", []))
            analysis.pop("percussion_activity", None)
        return public

    def _store_context(self, context: dict, principal_id: str) -> None:
        with self._lock:
            now = self.clock()
            for key, (created, owner, _) in list(self._contexts.items()):
                if now - created > 1800 or owner == principal_id and sum(row[1] == owner for row in self._contexts.values()) >= 16:
                    self._contexts.pop(key, None)
            while len(self._contexts) >= 128:
                self._contexts.popitem(last=False)
            self._contexts[context["context_id"]] = (now, principal_id, copy.deepcopy(context))

    def _title_context(self, body: dict, principal_id: str) -> dict:
        brief = body.get("brief", "")
        if not isinstance(brief, str) or len(brief) > 100_000:
            raise CreativeError("brief must be text up to 100,000 characters.")
        if body.get("references") or body.get("model_type"):
            raise CreativeError("Title cards render exact text without a model or character references. Generate artwork separately with image.generate, then pass image_asset in plan.")
        duration = body.get("duration_seconds", 3)
        if type(duration) not in (int, float) or not math.isfinite(duration) or not .5 <= duration <= 15:
            raise CreativeError("Title-card duration must be between 0.5 and 15 seconds.")
        resolution = body.get("resolution", "1280x720")
        if not isinstance(resolution, str) or not _RESOLUTION.fullmatch(resolution):
            raise CreativeError("Title-card resolution must be WIDTHxHEIGHT.")
        width, height = map(int, resolution.split("x"))
        if not all(64 <= value <= 4096 and value % 2 == 0 for value in (width, height)):
            raise CreativeError("Title-card dimensions must be even and between 64 and 4096.")
        params = body.get("params") or {}
        if not isinstance(params, dict) or set(params) - {"workspace", "fps"}:
            raise CreativeError("Title-card params accept workspace and fps only.")
        fps = params.get("fps", 24)
        if type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 120:
            raise CreativeError("Title-card fps must be between 1 and 120.")
        frames = max(1, round(duration * fps))
        context = {
            "version": 1, "context_id": str(uuid.uuid4()), "workflow": "title.card",
            "model_type": "maestro_title_card", "model": {"name": "Exact-text title card", "requires_model": False},
            "model_def": {}, "options": {}, "brief": brief, "references": [],
            "params": {"workspace": params.get("workspace", "default"), "width": width, "height": height, "fps": fps, "duration_seconds": frames / fps},
            "fps": fps, "selection_source": "cpu_title_renderer", "session_revision": self.session(),
            "author": "connected_agent", "local_writer": False,
            "timeline": {"kind": "title_card", "total_frames": frames, "fps": fps, "duration_seconds": frames / fps},
            "guides": {"title_cards": "Author plan.card with exact text and optional subtitle; role opening/intertitle/end; style minimal/classic/bold; #RRGGBB background_color/text_color. Keep the context duration. No character references or local writer. For artistic cards generate an image with MOSS/FLUX/etc using image.generate, review its spelling and supply its uploaded image_asset alongside card. For animated typography use video.frames and the title guide available through the filmmaking guide. Review before submitting; no render happens during preparation."},
        }
        context["context_revision"] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        self._store_context(context, principal_id)
        return copy.deepcopy(context)

    def lookup(self, context_id: str, principal_id: str) -> dict:
        if not isinstance(context_id, str):
            raise CreativeError("context_id must be the identifier returned by get_creation_context.")
        with self._lock:
            entry = self._contexts.get(context_id)
            if entry is None or entry[1] != principal_id or self.clock() - entry[0] > 1800:
                raise CreativeError("Creative context is unavailable or expired; request a fresh context before preparing this plan.", 409)
            value = copy.deepcopy(entry[2])
        if value["session_revision"] != self.session():
            raise CreativeError("Maestro restarted; refresh the creative context before preparing this plan.", 409)
        return value

    def remember_prepared(self, context_id: str, principal_id: str, kind: str, payload: dict) -> str:
        self.lookup(context_id, principal_id)
        identifier = str(uuid.uuid4())
        with self._lock:
            while len(self._prepared) >= 128:
                self._prepared.popitem(last=False)
            self._prepared[identifier] = (self.clock(), principal_id, context_id, kind, copy.deepcopy(payload))
        return identifier

    def prepared(self, identifier: str, principal_id: str) -> tuple[str, dict, dict]:
        if not isinstance(identifier, str):
            raise CreativeError("prepared_id must be the identifier returned by prepare_authored_generation.")
        with self._lock:
            value = self._prepared.get(identifier)
            if value is None or value[1] != principal_id or self.clock() - value[0] > 1800:
                raise CreativeError("Prepared plan is unavailable or expired; prepare a new revision before submitting.", 409)
            _, _, context_id, kind, payload = copy.deepcopy(value)
        return kind, self.lookup(context_id, principal_id), payload
