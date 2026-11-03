"""Validation, bindings, preflight checks, and built-in ComfyUI API graphs."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


MAX_WORKFLOW_NODES = 500
MAX_WORKFLOW_BYTES = 4 * 1024 * 1024
API_EXPORT_GUIDANCE = (
    "Export the workflow as ComfyUI API format (use Save (API) in ComfyUI), "
    "then retry."
)
_BINDING_TYPES = {"string", "integer", "number", "boolean", "choice", "image", "video", "audio"}
_MODEL_NAME_MARKERS = (
    "ckpt", "checkpoint", "lora", "vae", "unet", "diffusion_model", "clip",
    "controlnet", "ipadapter", "ip_adapter", "embedding", "hypernetwork",
    "upscale", "text_encoder", "vision_model", "audio_model", "vocoder",
    "model_name", "model_path", "style_model",
)


class WorkflowError(ValueError):
    """A ComfyUI workflow or binding definition is invalid."""


def _fail_api_format(detail: str) -> WorkflowError:
    return WorkflowError(f"{detail} {API_EXPORT_GUIDANCE}")


def _json_size(graph: Any) -> int:
    try:
        encoded = json.dumps(graph, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, UnicodeError):
        raise WorkflowError("Workflow must contain only finite JSON values.") from None
    return len(encoded.encode("utf-8"))


def _validate_reference(value: Any, node_ids: set[str], *, node_id: str, input_name: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _validate_reference(nested, node_ids, node_id=node_id, input_name=input_name)
        return
    if not isinstance(value, list):
        return
    if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int) and not isinstance(value[1], bool):
        ref_id, output_index = value
        if ref_id not in node_ids:
            raise WorkflowError(
                f"Node {node_id} input {input_name} references missing node {ref_id}."
            )
        if output_index < 0:
            raise WorkflowError(
                f"Node {node_id} input {input_name} has a negative output index."
            )
        return
    for nested in value:
        _validate_reference(nested, node_ids, node_id=node_id, input_name=input_name)


def normalize_workflow(graph: Any) -> dict[str, dict[str, Any]]:
    """Return a detached API-format graph, optionally unwrapping ``prompt``.

    UI-format node/link documents are rejected with direct export guidance.
    The API graph is bounded by both node count and serialized UTF-8 size.
    """
    if not isinstance(graph, Mapping):
        raise _fail_api_format("Workflow must be an object.")
    source: Any = graph
    if "prompt" in graph and isinstance(graph.get("prompt"), Mapping):
        source = graph["prompt"]
    if "nodes" in source or "links" in source:
        raise _fail_api_format("This looks like a ComfyUI UI workflow with nodes/links.")
    if not source:
        raise _fail_api_format("Workflow has no API prompt nodes.")
    if not all(isinstance(key, str) for key in source):
        raise _fail_api_format("API workflow node IDs must be strings.")
    if len(source) > MAX_WORKFLOW_NODES:
        raise WorkflowError(f"Workflow has too many nodes (limit {MAX_WORKFLOW_NODES}).")
    try:
        result = copy.deepcopy(dict(source))
    except Exception:
        raise WorkflowError("Workflow could not be copied safely.") from None
    if _json_size(result) > MAX_WORKFLOW_BYTES:
        raise WorkflowError(f"Workflow exceeds the {MAX_WORKFLOW_BYTES // (1024 * 1024)} MiB size limit.")

    for node_id, node in result.items():
        if not node_id or len(node_id) > 256:
            raise WorkflowError("Workflow node IDs must be non-empty strings up to 256 characters.")
        if not isinstance(node, dict):
            raise _fail_api_format(f"Node {node_id} is not an API node object.")
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        if not isinstance(class_type, str) or not class_type.strip() or len(class_type) > 256:
            if "type" in node or "pos" in node or "widgets_values" in node:
                raise _fail_api_format(f"Node {node_id} looks like a UI node.")
            raise _fail_api_format(f"Node {node_id} is missing class_type.")
        if not isinstance(inputs, dict):
            raise _fail_api_format(f"Node {node_id} is missing an inputs object.")
        for input_name in inputs:
            if not isinstance(input_name, str) or not input_name:
                raise WorkflowError(f"Node {node_id} has an invalid input name.")
    node_ids = set(result)
    for node_id, node in result.items():
        for input_name, value in node["inputs"].items():
            _validate_reference(value, node_ids, node_id=node_id, input_name=input_name)
    return result


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and not (
        isinstance(value, float) and not math.isfinite(value)
    )


def _binding_value_error(binding: Mapping[str, Any], value: Any, *, label: str = "value") -> str | None:
    binding_type = binding.get("type")
    name = binding.get("name", "<unnamed>")
    valid = True
    if binding_type in {"string", "image", "video", "audio"}:
        valid = isinstance(value, str)
        if binding_type in {"image", "video", "audio"}:
            valid = valid and bool(value.strip())
    elif binding_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif binding_type == "number":
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    elif binding_type == "boolean":
        valid = isinstance(value, bool)
    elif binding_type == "choice":
        valid = _is_scalar(value)
    else:
        return f"Binding {name} has unsupported type {binding_type!r}."
    if not valid:
        return f"Binding {name} {label} does not match type {binding_type}."
    if binding_type in {"integer", "number"}:
        minimum = binding.get("min")
        maximum = binding.get("max")
        if minimum is not None and value < minimum:
            return f"Binding {name} {label} is below its minimum {minimum}."
        if maximum is not None and value > maximum:
            return f"Binding {name} {label} is above its maximum {maximum}."
    options = binding.get("options")
    if options is not None and value not in options:
        return f"Binding {name} {label} is not one of its configured options."
    return None


def validate_bindings(graph: Any, bindings: Any) -> list[str]:
    """Return configuration errors for user-editable workflow bindings."""
    errors: list[str] = []
    try:
        normalized = normalize_workflow(graph)
    except WorkflowError as exc:
        return [str(exc)]
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes, bytearray)):
        return ["Bindings must be a list of binding objects."]
    names: set[str] = set()
    targets: set[tuple[str, str]] = set()
    known_fields = {"name", "label", "node_id", "input", "type", "required", "default", "options", "min", "max"}
    for index, binding in enumerate(bindings):
        prefix = f"Binding {index + 1}"
        if not isinstance(binding, Mapping):
            errors.append(f"{prefix} must be an object.")
            continue
        name = binding.get("name")
        node_id = binding.get("node_id")
        input_name = binding.get("input")
        binding_type = binding.get("type")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix} needs a non-empty name.")
        elif name in names:
            errors.append(f"Binding name {name!r} is duplicated.")
        else:
            names.add(name)
        if not isinstance(node_id, str) or node_id not in normalized:
            errors.append(f"{prefix} references an unknown node_id.")
        if not isinstance(input_name, str) or not input_name:
            errors.append(f"{prefix} needs a non-empty input name.")
        elif isinstance(node_id, str) and node_id in normalized:
            inputs = normalized[node_id]["inputs"]
            if input_name not in inputs:
                errors.append(f"Binding {name or index + 1} targets missing input {node_id}.{input_name}.")
            elif isinstance(inputs[input_name], list) and len(inputs[input_name]) == 2 and isinstance(inputs[input_name][0], str) and isinstance(inputs[input_name][1], int):
                errors.append(f"Binding {name or index + 1} cannot replace linked input {node_id}.{input_name}.")
            else:
                target = (node_id, input_name)
                if target in targets:
                    errors.append(
                        f"Binding {name or index + 1} shares graph input {node_id}.{input_name} with another binding; each workflow input can have only one binding."
                    )
                else:
                    targets.add(target)
        if not isinstance(binding_type, str) or binding_type not in _BINDING_TYPES:
            errors.append(f"Binding {name or index + 1} has unsupported type {binding_type!r}.")
        required = binding.get("required", False)
        if not isinstance(required, bool):
            errors.append(f"Binding {name or index + 1} required must be a boolean.")
        label = binding.get("label")
        if label is not None and (not isinstance(label, str) or len(label) > 256):
            errors.append(f"Binding {name or index + 1} label must be a string up to 256 characters.")
        unknown = set(binding) - known_fields
        if unknown:
            errors.append(f"Binding {name or index + 1} has unsupported fields: {', '.join(sorted(map(str, unknown)))}.")
        options = binding.get("options")
        if options is not None:
            if not isinstance(options, list) or any(not _is_scalar(option) for option in options):
                errors.append(f"Binding {name or index + 1} options must be a list of scalar values.")
            elif binding_type != "choice":
                errors.append(f"Binding {name or index + 1} options are only valid for choice bindings.")
        minimum = binding.get("min")
        maximum = binding.get("max")
        if minimum is not None or maximum is not None:
            if binding_type not in {"integer", "number"}:
                errors.append(f"Binding {name or index + 1} min/max are only valid for numeric bindings.")
            for bound_name, bound in (("min", minimum), ("max", maximum)):
                if bound is not None and (
                    not isinstance(bound, (int, float))
                    or isinstance(bound, bool)
                    or (isinstance(bound, float) and not math.isfinite(bound))
                ):
                    errors.append(f"Binding {name or index + 1} {bound_name} must be a finite number.")
            if (
                isinstance(minimum, (int, float))
                and not isinstance(minimum, bool)
                and (not isinstance(minimum, float) or math.isfinite(minimum))
                and isinstance(maximum, (int, float))
                and not isinstance(maximum, bool)
                and (not isinstance(maximum, float) or math.isfinite(maximum))
                and minimum > maximum
            ):
                errors.append(f"Binding {name or index + 1} min cannot exceed max.")
        if "default" in binding:
            if not _is_scalar(binding["default"]):
                errors.append(f"Binding {name or index + 1} default must be a finite scalar value.")
            elif isinstance(binding_type, str) and binding_type in _BINDING_TYPES:
                value_error = _binding_value_error(binding, binding["default"], label="default")
                if value_error:
                    errors.append(value_error)
    return errors


def _value_error(binding: Mapping[str, Any], value: Any) -> str | None:
    return _binding_value_error(binding, value)


def apply_bindings(graph: Any, bindings: Any, values: Any) -> dict[str, dict[str, Any]]:
    """Apply typed values to a detached workflow graph without changing links."""
    errors = validate_bindings(graph, bindings)
    if errors:
        raise WorkflowError(" ".join(errors))
    if not isinstance(values, Mapping):
        raise WorkflowError("Binding values must be an object keyed by binding name.")
    by_name = {binding["name"]: binding for binding in bindings}
    unknown = set(values) - set(by_name)
    if unknown:
        raise WorkflowError(f"Unknown binding values: {', '.join(sorted(map(str, unknown)))}.")
    result = normalize_workflow(graph)
    for name, binding in by_name.items():
        if name in values:
            value = values[name]
            error = _value_error(binding, value)
            if error:
                raise WorkflowError(error)
        elif "default" in binding:
            value = binding["default"]
        elif binding.get("required", False):
            raise WorkflowError(f"Required binding {name} has no value.")
        else:
            continue
        # Asset bindings accept only the filename returned by the upload step;
        # this module does not read paths or perform uploads itself.
        result[binding["node_id"]]["inputs"][binding["input"]] = copy.deepcopy(value)
    return result


def _input_definitions(node_info: Mapping[str, Any]) -> dict[str, Any]:
    input_root = node_info.get("input", {})
    if not isinstance(input_root, Mapping):
        return {}
    definitions: dict[str, Any] = {}
    for group in ("required", "optional"):
        group_values = input_root.get(group, {})
        if isinstance(group_values, Mapping):
            definitions.update(group_values)
    return definitions


def _combo_choices(definition: Any) -> list[Any] | None:
    if isinstance(definition, (list, tuple)) and definition:
        first = definition[0]
        if isinstance(first, (list, tuple)):
            return list(first)
    return None


def binding_choices(graph: Any, bindings: Any, object_info: Any) -> dict[str, list[Any]]:
    """Return bounded choices for registered inputs advertised by this server.

    This inspects node schemas only. It never scans disk, picks a checkpoint,
    or infers compatibility from a model's filename.
    """
    normalized = normalize_workflow(graph)
    errors = validate_bindings(normalized, bindings)
    if errors:
        raise WorkflowError("; ".join(errors))
    if not isinstance(object_info, Mapping):
        raise WorkflowError("ComfyUI object_info must be an object.")
    result: dict[str, list[Any]] = {}
    remaining = 10000
    for binding in bindings:
        if binding["type"] in {"image", "video", "audio"}:
            # Media must come from an authorized Maestro asset/upload, not an
            # arbitrary filename enumerated from a shared server's input folder.
            continue
        if remaining <= 0:
            break
        node = normalized[binding["node_id"]]
        node_info = object_info.get(node["class_type"], {})
        definitions = _input_definitions(node_info) if isinstance(node_info, Mapping) else {}
        choices = _combo_choices(definitions.get(binding["input"]))
        if choices is None:
            choices = binding.get("options")
        if choices is None:
            continue
        allowed = binding.get("options")
        values: list[Any] = []
        for value in choices:
            if not _is_scalar(value) or isinstance(value, str) and len(value) > 4096:
                continue
            if allowed is not None and value not in allowed:
                continue
            if value not in values:
                values.append(value)
            if len(values) >= min(2000, remaining):
                break
        result[binding["name"]] = values
        remaining -= len(values)
    return result


def _is_model_input(node_class: str, input_name: str) -> bool:
    name = input_name.lower().replace("-", "_")
    return any(marker in name for marker in _MODEL_NAME_MARKERS)


def _catalog_values(model_catalog: Any, input_name: str) -> set[str]:
    if model_catalog is None:
        return set()
    category_hints = {
        "ckpt": ("checkpoints", "checkpoint"),
        "checkpoint": ("checkpoints", "checkpoint"),
        "lora": ("loras", "lora"),
        "vae": ("vae", "vaes"),
        "unet": ("diffusion_models", "unet"),
        "diffusion_model": ("diffusion_models",),
        "controlnet": ("controlnet", "controlnets"),
        "ipadapter": ("ipadapter", "ipadapter_models"),
        "ip_adapter": ("ipadapter", "ipadapter_models"),
        "embedding": ("embeddings",),
        "upscale": ("upscale_models", "upscalers"),
        "clip": ("clip", "text_encoders"),
        "text_encoder": ("text_encoders", "clip"),
        "vision_model": ("clip_vision", "vision_models"),
        "audio_model": ("checkpoints", "audio_models"),
        "vocoder": ("checkpoints", "vocoders"),
        "model_name": (),
        "model_path": (),
    }
    name = input_name.lower().replace("-", "_")
    keys = {key for marker, categories in category_hints.items() if marker in name for key in categories}
    candidates: list[Any] = []
    if isinstance(model_catalog, Mapping):
        for category, values in model_catalog.items():
            if not keys or str(category).lower() in keys:
                candidates.append(values)
    else:
        candidates.append(model_catalog)
    found: set[str] = set()
    stack = list(candidates)
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
    return found


def preflight_workflow(
    graph: Any,
    object_info: Any,
    model_catalog: Any = None,
) -> dict[str, Any]:
    """Check workflow nodes, input schemas, choices, and advertised models."""
    errors: list[str] = []
    missing_nodes: list[dict[str, str]] = []
    missing_models: list[dict[str, Any]] = []
    try:
        normalized = normalize_workflow(graph)
    except WorkflowError as exc:
        return {"valid": False, "errors": [str(exc)], "missing_nodes": [], "missing_models": []}
    if not isinstance(object_info, Mapping):
        return {
            "valid": False,
            "errors": ["ComfyUI object_info must be an object."],
            "missing_nodes": [],
            "missing_models": [],
        }
    for node_id, node in normalized.items():
        class_type = node["class_type"]
        node_info = object_info.get(class_type)
        if not isinstance(node_info, Mapping):
            missing_nodes.append({"node_id": node_id, "class_type": class_type})
            continue
        definitions = _input_definitions(node_info)
        required_group = node_info.get("input", {}).get("required", {}) if isinstance(node_info.get("input"), Mapping) else {}
        if isinstance(required_group, Mapping):
            for required_input in required_group:
                if required_input not in node["inputs"]:
                    errors.append(f"Node {node_id} ({class_type}) is missing required input {required_input}.")
        for input_name, value in node["inputs"].items():
            if input_name not in definitions:
                errors.append(f"Node {node_id} ({class_type}) has unknown input {input_name}.")
                continue
            choices = _combo_choices(definitions[input_name])
            if not isinstance(value, str):
                continue
            is_model = _is_model_input(class_type, input_name)
            if choices is None:
                if is_model and model_catalog is not None and value not in _catalog_values(model_catalog, input_name):
                    missing_models.append({"node_id": node_id, "input": input_name, "value": value})
                continue
            if value in choices:
                continue
            if is_model:
                available = {str(candidate) for candidate in choices}
                available.update(_catalog_values(model_catalog, input_name))
                if value not in available:
                    missing_models.append({"node_id": node_id, "input": input_name, "value": value})
            else:
                errors.append(
                    f"Node {node_id} ({class_type}) input {input_name} value {value!r} is not an advertised choice."
                )
    valid = not errors and not missing_nodes and not missing_models
    return {
        "valid": valid,
        "errors": errors,
        "missing_nodes": missing_nodes,
        "missing_models": missing_models,
    }


def builtin_templates() -> list[dict[str, Any]]:
    """Return model-free CPU smoke and standard checkpoint image templates."""
    checkpoint_workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ""}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1,
                "steps": 30,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "maestro", "images": ["6", 0]}},
    }
    cpu_workflow = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}},
        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "maestro_cpu_smoke", "images": ["1", 0]}},
    }
    flux3_klein_workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "", "type": "flux3", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": ""}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "6": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1,
                "steps": 16,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "maestro_flux3_klein_distilled", "images": ["8", 0]},
        },
    }
    flux3_klein_bindings = [
        {"name": "diffusion_model", "label": "Diffusion model", "node_id": "1", "input": "unet_name", "type": "string", "required": True},
        {"name": "text_encoder", "label": "Text encoder", "node_id": "2", "input": "clip_name", "type": "string", "required": True},
        {"name": "vae", "label": "VAE", "node_id": "3", "input": "vae_name", "type": "string", "required": True},
        {"name": "positive_prompt", "label": "Prompt", "node_id": "4", "input": "text", "type": "string", "required": True},
        {"name": "width", "label": "Width", "node_id": "6", "input": "width", "type": "integer", "default": 512, "min": 64, "max": 2048},
        {"name": "height", "label": "Height", "node_id": "6", "input": "height", "type": "integer", "default": 512, "min": 64, "max": 2048},
        {"name": "steps", "label": "Steps", "node_id": "7", "input": "steps", "type": "integer", "default": 16, "min": 1, "max": 64},
        {"name": "cfg", "label": "CFG", "node_id": "7", "input": "cfg", "type": "number", "default": 1.0, "min": 0.0, "max": 5.0},
        {"name": "seed", "label": "Seed", "node_id": "7", "input": "seed", "type": "integer", "default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
    ]
    flux3_klein_edit_workflow = copy.deepcopy(flux3_klein_workflow)
    flux3_klein_edit_workflow["7"]["inputs"]["positive"] = ["12", 0]
    flux3_klein_edit_workflow["7"]["inputs"]["negative"] = ["13", 0]
    flux3_klein_edit_workflow.update(
        {
            "10": {"class_type": "LoadImage", "inputs": {"image": ""}},
            "11": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["10", 0], "vae": ["3", 0]},
            },
            "12": {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["4", 0], "latent": ["11", 0]},
            },
            "13": {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["5", 0], "latent": ["11", 0]},
            },
        }
    )
    flux3_klein_edit_bindings = copy.deepcopy(flux3_klein_bindings)
    flux3_klein_edit_bindings.append(
        {"name": "reference_image", "label": "Reference image", "node_id": "10", "input": "image", "type": "image", "required": True}
    )
    return [
        {
            "id": "sdxl_checkpoint_txt2img",
            "name": "SDXL checkpoint text to image",
            "kind": "image",
            "description": (
                "Uses the ComfyUI core CheckpointLoaderSimple, CLIPTextEncode, EmptyLatentImage, "
                "KSampler, VAEDecode, and SaveImage nodes at 1024×1024. Requires a locally "
                "installed checkpoint compatible with the intended output (SDXL for SDXL); "
                "compatibility is not inferred."
            ),
            "workflow": checkpoint_workflow,
            "bindings": [
                {"name": "checkpoint", "label": "Checkpoint", "node_id": "1", "input": "ckpt_name", "type": "string", "required": True},
                {"name": "positive_prompt", "label": "Positive prompt", "node_id": "2", "input": "text", "type": "string", "required": True, "default": ""},
                {"name": "negative_prompt", "label": "Negative prompt", "node_id": "3", "input": "text", "type": "string", "default": ""},
                {"name": "width", "label": "Width", "node_id": "4", "input": "width", "type": "integer", "default": 1024, "min": 64, "max": 8192},
                {"name": "height", "label": "Height", "node_id": "4", "input": "height", "type": "integer", "default": 1024, "min": 64, "max": 8192},
                {"name": "steps", "label": "Steps", "node_id": "5", "input": "steps", "type": "integer", "default": 30, "min": 1, "max": 200},
                {"name": "cfg", "label": "CFG", "node_id": "5", "input": "cfg", "type": "number", "default": 7.0, "min": 0, "max": 100},
                {"name": "seed", "label": "Seed", "node_id": "5", "input": "seed", "type": "integer", "default": 1, "min": 0},
            ],
        },
        {
            "id": "cpu_empty_image_smoke",
            "name": "CPU image connectivity smoke test",
            "kind": "image",
            "description": "Creates a small solid-color image with the ComfyUI core EmptyImage and SaveImage nodes; no checkpoint or GPU model is needed.",
            "workflow": cpu_workflow,
            "bindings": [
                {"name": "width", "label": "Width", "node_id": "1", "input": "width", "type": "integer", "default": 64, "min": 1, "max": 4096},
                {"name": "height", "label": "Height", "node_id": "1", "input": "height", "type": "integer", "default": 64, "min": 1, "max": 4096},
                {"name": "color", "label": "Color", "node_id": "1", "input": "color", "type": "integer", "default": 0, "min": 0, "max": 16777215},
            ],
        },
        {
            "id": "flux3_klein_distilled_txt2img",
            "name": "FLUX3 Klein distilled text to image",
            "kind": "image",
            "description": (
                "Uses ComfyUI core UNETLoader, CLIPLoader (flux3), VAELoader, CLIPTextEncode, "
                "EmptyFlux2LatentImage, KSampler, VAEDecode, and SaveImage nodes. Select a locally "
                "installed FLUX3 Klein distilled diffusion model, its matching Qwen3-VL 4B text "
                "encoder, and VAE. Defaults match the validated 512×512, 16-step Euler/simple, "
                "CFG 1.0 path. The negative prompt is intentionally empty for this distilled "
                "setup. Width and height should be multiples of 16. This is not a generic Flux 2 "
                "or FLUX3 base workflow."
            ),
            "workflow": flux3_klein_workflow,
            "bindings": flux3_klein_bindings,
        },
        {
            "id": "flux3_klein_distilled_reference_edit",
            "name": "FLUX3 Klein distilled single-reference edit (Experimental)",
            "kind": "image",
            "description": (
                "Uses ComfyUI core nodes for FLUX3 Klein distilled reference editing. Select a "
                "locally installed compatible diffusion model, Qwen3-VL 4B text encoder, and VAE, "
                "then provide one reference image. VAEEncode and ReferenceLatent append the image "
                "to positive and empty-negative conditioning. Sampling starts from an independent "
                "empty Flux2 latent, so the width and height bindings set output size regardless "
                "of the reference image dimensions. Defaults use 16-step Euler/simple sampling "
                "with distilled CFG 1.0. Width and height should be multiples of 16. This is not a "
                "generic Flux 2 or FLUX3 base workflow. Inspect the rendered edit: successful "
                "reference conditioning does not guarantee the requested visual change."
            ),
            "workflow": flux3_klein_edit_workflow,
            "bindings": flux3_klein_edit_bindings,
        },
    ]


__all__ = [
    "API_EXPORT_GUIDANCE",
    "MAX_WORKFLOW_NODES",
    "MAX_WORKFLOW_BYTES",
    "WorkflowError",
    "normalize_workflow",
    "validate_bindings",
    "apply_bindings",
    "preflight_workflow",
    "builtin_templates",
    "binding_choices",
]
