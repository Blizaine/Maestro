"""Bounded structural verification for MiniMax H3 transformer checkpoints.

The importer consumes only a SafeTensor header plus explicitly small metadata
payloads (the rank-8 AdaLN table and Comfy quantization descriptors).  It never
opens or materializes transformer weight tensors.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Mapping
from typing import Any, Callable


class H3CheckpointError(ValueError):
    """An H3 import candidate failed a structural or safety check."""

    def __init__(self, message: str, *, code: str = "invalid_checkpoint", **details):
        super().__init__(message)
        self.code = code
        self.details = details


_HIDDEN = 5376
_ATTENTION_INNER = 56 * 128
_LAYERS = 50
_CURVE_GRID = 1025
_CURVE_DIM = 8
_FULL_TIME_DIM = 2688
_FFN = 14336
_QKV_ROWS = 3 * _ATTENTION_INNER
_TOKEN_REFINER_LAYERS = 2
_QUANT_MODULE_SUFFIXES = (
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)
_FLOAT_PARAMETER_DTYPES = {"BF16", "F16"}
_FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN"}
_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
}
_PINNED_CIVITAI_PROFILES = {
    # DaSiWa v3 releases.  A source name/version alone is not a pin: all three
    # numeric identities and the advertised SHA-256 must match.
    (2877206, 3374445, 3263052): {
        "require_dasiwa_header": True,
        "sha256": "0ce4dfecce862b010823a5a74903543bd7f2ad21872272232f70f9a836d30e53",
        "sampling_profile": "standard",
        "native_workflow": "ref2va",
        "sampler": "euler",
        "video_shift": 11,
        "audio_shift": 4,
        "default_steps": 25,
        "min_steps": 20,
        "max_steps": 30,
        "baked_turbo": False,
        "fused_turbo": False,
    },
    (2877206, 3374439, 3263048): {
        "require_dasiwa_header": True,
        "sha256": "5da4bbf303f91b468010f504951e3959cd77420ec5f93ae007c9660b485f7df6",
        "sampling_profile": "turbo",
        "native_workflow": "ref2va",
        "sampler": "euler",
        "video_shift": 9,
        "audio_shift": 4,
        "default_steps": 8,
        "min_steps": 4,
        "max_steps": 8,
        "baked_turbo": True,
        "fused_turbo": False,
    },
}

# These CivitAI copies match the official Comfy-Org files byte-for-byte.
# The published digests prove their grouped QKV layout when the old exports
# have no model metadata. Other FP8 imports still need exporter layout evidence.
_VERIFIED_COMFY_FP8_RELEASES = {
    "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993": "fl2va",
    "f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9": "ref2va",
}

for _identity, _int8_identity, _digest in (
    ((2877206, 3374445, 3262967), (2877206, 3374445, 3263052), "b0ba2d0baad7b2ebca1182e9fb13e209626decf8bb13cbc4b0d996889fc25ab8"),
    ((2877206, 3374439, 3262925), (2877206, 3374439, 3263048), "bbae6ed20254793bc3f8f3f896566225306c30a66f29b3d625114458c0bb291e"),
):
    _PINNED_CIVITAI_PROFILES[_identity] = {
        **_PINNED_CIVITAI_PROFILES[_int8_identity], "sha256": _digest,
        "quantization_format": "asym_w4a8_int8",
    }

_SUPPORTED_CONVROT_GROUPS = {64, 256}
_GGUF_PACKED_DTYPES = {"GGUF_" + name for name in (
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
)}
_SUPPORTED_SAMPLING_PROFILES = {"standard", "turbo", "fused"}
_SUPPORTED_NATIVE_WORKFLOWS = {"fl2va", "ref2va"}
_MAX_SMALL_TENSOR_BYTES = 64 * 1024
_MAX_QUANT_DESCRIPTOR_BYTES = 16 * 1024


def _fail(message: str, code: str, **details):
    raise H3CheckpointError(message, code=code, **details)


def _shape_text(shape) -> str:
    return "x".join(str(value) for value in shape)


def _normalize_dtype(value) -> str:
    name = str(value or "").strip().upper().replace("TORCH.", "")
    aliases = {
        "BFLOAT16": "BF16",
        "FLOAT16": "F16",
        "HALF": "F16",
        "FLOAT32": "F32",
        "FLOAT": "F32",
        "INT8": "I8",
        "UINT8": "U8",
        "F8_E4M3FN": "F8_E4M3FN",
        "F8_E4M3": "F8_E4M3",
        "FLOAT8_E4M3FN": "F8_E4M3FN",
        "FLOAT8_E4M3": "F8_E4M3",
        "F8_E5M2": "F8_E5M2",
        "FLOAT8_E5M2": "F8_E5M2",
    }
    return aliases.get(name, name)


def _normalize_tensor_name(name: str) -> str:
    value = str(name)
    prefixes = (
        "model.diffusion_model.",
        "diffusion_model.",
        "module.model.diffusion_model.",
        "module.diffusion_model.",
        "module.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix) :]
                changed = True
                break
    return value


def _header_parts(header):
    if not isinstance(header, Mapping):
        _fail("SafeTensor header must be a mapping of tensor descriptors.", "invalid_header")

    root = header.get("header") if isinstance(header.get("header"), Mapping) else header
    metadata = root.get("__metadata__", root.get("metadata", header.get("metadata", {})))
    if not isinstance(metadata, Mapping):
        _fail("SafeTensor metadata must be a mapping.", "invalid_header_metadata")
    tensors = root.get("tensors", header.get("tensors"))
    if not isinstance(tensors, Mapping):
        tensors = {
            key: value
            for key, value in root.items()
            if key not in {"__metadata__", "metadata", "header", "tensors"}
        }

    normalized = {}
    for raw_name, descriptor in tensors.items():
        if not isinstance(raw_name, str) or not isinstance(descriptor, Mapping):
            _fail("SafeTensor tensor entries must have string names and descriptor objects.", "invalid_tensor_descriptor")
        if "shape" not in descriptor or "dtype" not in descriptor:
            _fail(f"Tensor {raw_name!r} is missing its dtype or shape descriptor.", "invalid_tensor_descriptor", tensor=raw_name)
        raw_shape = descriptor.get("shape")
        if not isinstance(raw_shape, (list, tuple)):
            _fail(f"Tensor {raw_name!r} has an invalid shape descriptor.", "invalid_tensor_shape", tensor=raw_name)
        shape = []
        for dim in raw_shape:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                _fail(f"Tensor {raw_name!r} has a non-positive or non-integer dimension.", "invalid_tensor_shape", tensor=raw_name)
            shape.append(dim)
        dtype = _normalize_dtype(descriptor.get("dtype"))
        if not dtype:
            _fail(f"Tensor {raw_name!r} has no dtype name.", "invalid_tensor_descriptor", tensor=raw_name)
        name = _normalize_tensor_name(raw_name)
        if name in normalized:
            _fail(
                f"Multiple wrapped tensor names normalize to {name!r}; the checkpoint is ambiguous.",
                "duplicate_tensor",
                tensor=name,
            )
        normalized[name] = {
            "raw_name": raw_name,
            "dtype": dtype,
            "shape": tuple(shape),
            "descriptor": descriptor,
        }
    if not normalized:
        _fail("SafeTensor header contains no tensor descriptors.", "empty_header")
    return normalized, dict(metadata)


def _source_pin(source):
    if not isinstance(source, Mapping):
        return None
    try:
        identity = (
            int(source.get("modelId")),
            int(source.get("versionId")),
            int(source.get("fileId")),
        )
    except (TypeError, ValueError):
        return None
    profile = _PINNED_CIVITAI_PROFILES.get(identity)
    if profile is None:
        workflow = _VERIFIED_COMFY_FP8_RELEASES.get(str(source.get("sha256") or "").lower())
        if workflow is None:
            return None
        return {
            "sampling_profile": "standard", "native_workflow": workflow,
            "sampler": "euler", "video_shift": 12, "audio_shift": 3,
            "default_steps": 20, "min_steps": 2, "max_steps": 50,
            "baked_turbo": False, "fused_turbo": False,
            "qkv_layout": "grouped", "quantization_format": "scaled_fp8",
        }
    sha256 = str(source.get("sha256") or "").strip().lower()
    if sha256 != profile["sha256"]:
        _fail(
            "The CivitAI file ID matches a pinned H3 release, but its SHA-256 does not. Refresh source metadata and verify the file before importing.",
            "source_hash_mismatch",
            expected_sha256=profile["sha256"],
            actual_sha256=sha256 or None,
        )
    return profile


def _metadata_text(metadata, keys=None) -> str:
    selected = []
    for key, value in metadata.items():
        if keys is not None and key not in keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            selected.append(str(value))
    return "\n".join(selected)


def _source_text(source, field_names) -> str:
    if not isinstance(source, Mapping):
        return ""
    return "\n".join(str(source.get(key) or "") for key in field_names)


def _workflow_from_text(value: str):
    text = str(value or "").lower().replace("_", "-")
    has_ref = bool(
        re.search(r"\bref\s*-?\s*2\s*-?\s*va\b", text)
        or re.search(r"\breference\s*(?:to\s*)?video\b", text)
        or re.search(r"\breference\s*video\b", text)
    )
    has_fl = bool(
        re.search(r"\bfl\s*-?\s*2\s*-?\s*va\b", text)
        or re.search(r"\bfirst\s*(?:and|/)\s*last\s*frame", text)
        or re.search(r"\bfirst\s*last\s*frame", text)
    )
    if has_ref == has_fl:
        return None
    return "ref2va" if has_ref else "fl2va"


def _recipe_from_text(value: str):
    text = str(value or "").lower().replace("_", " ")
    text = re.sub(r"\bnon[- ]?turbo\b", "standard", text)
    fused = bool(
        re.search(r"\bfused\s+(?:h3\s+)?turbo\b", text)
        or re.search(r"\bturbo\b.{0,60}\b(?:fused|fusion)\b", text)
        or re.search(r"\b(?:fused|fusion)\b.{0,60}\bturbo\b", text)
    )
    if fused:
        return "fused"
    turbo = bool(re.search(r"\bturbo\b", text))
    if turbo:
        return "turbo"
    standard = bool(
        re.search(r"\bnon[- ]?distilled\b", text)
        or re.search(r"\bstandard\b", text)
        or re.search(r"\bbase\s+(?:minimax\s*)?h3\b", text)
    )
    return "standard" if standard else None


def _source_has_mixed_recipes(source):
    # One CivitAI version can contain Turbo and non-Turbo files with generic
    # numeric filenames. Page-wide acceleration instructions cannot identify
    # the selected file or supply its schedule in that case.
    text = _source_text(source, ("description",)).lower().replace("_", " ")
    non_turbo = re.search(r"\bnon[- ]?turbo\b", text)
    other_text = re.sub(r"\bnon[- ]?turbo\b", "", text)
    return bool(non_turbo and re.search(r"\bturbo\b", other_text))


def _explicit_header_workflow(metadata):
    keys = (
        "modelspec.title",
        "modelspec.description",
        "modelspec.architecture",
        "native_workflow",
        "workflow",
        "model_type",
    )
    return _workflow_from_text(_metadata_text(metadata, keys))


def _detect_workflow(metadata, source, pinned):
    header_workflow = _explicit_header_workflow(metadata)
    version_name_workflow = _workflow_from_text(_source_text(source, ("versionName",)))
    version_description_workflow = _workflow_from_text(
        _source_text(source, ("versionDescription",))
    )
    if (
        version_name_workflow
        and version_description_workflow
        and version_name_workflow != version_description_workflow
    ):
        _fail(
            "The selected CivitAI version name and version description identify different native H3 workflows.",
            "native_workflow_conflict",
            version_name=version_name_workflow,
            version_description=version_description_workflow,
        )
    version_workflow = version_name_workflow or version_description_workflow
    # A model page description often lists sibling FL2VA and Ref2VA files.
    # Use it only when the selected version and header carry no concrete
    # workflow identity; never let a sibling description erase a version pick.
    page_workflow = (
        _workflow_from_text(_source_text(source, ("name", "description")))
        if not header_workflow and not version_workflow
        else None
    )
    source_workflow = version_workflow or page_workflow
    if pinned:
        pinned_workflow = pinned["native_workflow"]
        for evidence in (header_workflow, version_workflow, page_workflow):
            if evidence and evidence != pinned_workflow:
                _fail(
                    "CivitAI source identity conflicts with the checkpoint's native H3 workflow metadata.",
                    "native_workflow_conflict",
                    pinned=pinned_workflow,
                    metadata=evidence,
                )
        return pinned_workflow
    if header_workflow and version_workflow and header_workflow != version_workflow:
        _fail(
            "The SafeTensor model metadata and CivitAI version metadata name different native H3 workflows.",
            "native_workflow_conflict",
            header=header_workflow,
            source=version_workflow,
        )
    return header_workflow or version_workflow or page_workflow


def _sampling_profile_from_source(metadata, source, pinned):
    if pinned:
        detected = pinned["sampling_profile"]
        explicit_header_profile = _recipe_from_text(
            _metadata_text(metadata, ("modelspec.title", "sampling_profile", "recipe"))
        )
        explicit_source_profile = _recipe_from_text(
            _source_text(source, ("versionName",))
        )
        for evidence in (explicit_header_profile, explicit_source_profile):
            if evidence and evidence != detected:
                _fail(
                    "Pinned CivitAI source identity conflicts with explicit header or version recipe metadata.",
                    "sampling_profile_conflict",
                    pinned=detected,
                    metadata=evidence,
                )
    else:
        header_fields = (
            "modelspec.title",
            "modelspec.description",
            "sampling_profile",
            "recipe",
        )
        header_profile = _recipe_from_text(_metadata_text(metadata, header_fields))
        source_name_profile = _recipe_from_text(
            _source_text(source, ("versionName", "name"))
        )
        source_description = _source_text(
            source, ("versionDescription", "description")
        ).lower()
        # A model description often discusses optional Turbo add-ons. Only
        # treat it as a baked recipe when it says Turbo is baked/merged or
        # gives an explicit Turbo step schedule.
        source_recipe_profile = None
        if (
            not _source_has_mixed_recipes(source)
            and re.search(r"\bturbo\b", source_description)
            and (
                re.search(r"\b(?:baked|fused|merge|merged|distillation|step(?:s)?\s*[:=]?\s*\d)", source_description)
                or re.search(r"\d\s*(?:-|–|to|till)\s*\d\s*(?:denoising\s*)?steps?", source_description)
            )
        ):
            source_recipe_profile = _recipe_from_text(source_description)
        evidence = [value for value in (header_profile, source_name_profile, source_recipe_profile) if value]
        if len(set(evidence)) > 1:
            _fail(
                "The header and CivitAI metadata describe conflicting H3 sampling recipes.",
                "sampling_profile_conflict",
                evidence=evidence,
            )
        detected = evidence[0] if evidence else None
    if pinned and detected != pinned["sampling_profile"]:
        _fail(
            "The pinned CivitAI release conflicts with checkpoint recipe metadata.",
            "sampling_profile_conflict",
            pinned=pinned["sampling_profile"],
            metadata=detected,
        )
    return detected


def _selection(value, allowed, label):
    normalized = str(value or "auto").strip().lower()
    if normalized == "auto":
        return None
    if normalized not in allowed:
        _fail(
            f"Unsupported {label} {value!r}. Choose one of: {', '.join(sorted(allowed))}.",
            f"unsupported_{label}",
            value=value,
            choices=sorted(allowed),
        )
    return normalized


def _quant_text_class(value: str):
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    if text in {"none", "unquantized", "bf16", "bfloat16", "fp16", "float16"}:
        return "plain"
    if text in {"asym_w4a8_int8", "w4a8_int8", "w4a8"}:
        return "asym_w4a8_int8"
    if any(token in text for token in ("int4", "int6", "nf4", "nvfp4", "gguf", "custom_backend", "custom backend")):
        return "unsupported"
    if "int8" in text and "convrot" in text:
        return "int8_convrot"
    if "int8_tensorwise" in text:
        return "int8_convrot"
    if "fp8" in text and "scaled" in text:
        return "scaled_fp8"
    if text in {"scaled_fp8", "fp8_e4m3fn_scaled", "fp8_e4m3_scaled",
                "float8_e4m3fn", "float8_e4m3", "fp8_e4m3fn", "fp8_e4m3"}:
        return "scaled_fp8"
    if "fp8" in text:
        return "fp8_unsupported"
    if "int8" in text or "int6" in text or "int4" in text:
        return "unsupported"
    return "unknown"


def _top_quantization(metadata):
    labels = []
    for key in ("quantization_format", "quantization.format", "quantization.bits"):
        if key not in metadata:
            continue
        classification = _quant_text_class(metadata[key])
        if classification in {"unsupported", "fp8_unsupported", "unknown"}:
            _fail(
                f"Header quantization metadata {key!r} declares an unsupported or unknown format.",
                "unsupported_quantization",
                field=key,
                value=str(metadata[key]),
            )
        if classification:
            labels.append(classification)
    if len(set(labels)) > 1:
        _fail(
            "Header quantization metadata contains conflicting format declarations.",
            "quantization_conflict",
            formats=sorted(set(labels)),
        )
    return labels[0] if labels else None


def _descriptor_convrot_fields(descriptor, format_class, *, layer, error_code):
    convrot = descriptor.get("convrot", False)
    if type(convrot) is not bool:
        _fail(f"ConvRot descriptor for {layer!r} must use a JSON boolean.", error_code, layer=layer)
    group_keys = ("convrot_groupsize", "convrot_group_size", "group_size")
    if format_class == "int8_convrot":
        groups = [descriptor[key] for key in group_keys if key in descriptor]
        if not groups or any(type(group) is not int for group in groups):
            _fail(f"INT8 ConvRot descriptor for {layer!r} needs an explicit integer group size.", error_code, layer=layer)
        if len(set(groups)) != 1:
            _fail(f"INT8 ConvRot descriptor for {layer!r} has conflicting group sizes.", "quantization_conflict", layer=layer)
        return convrot, groups[0]
    # W4A8's group_size describes weight packing, separate from its ConvRot
    # group. FP8 descriptors need no ConvRot group when rotation is disabled.
    group = next((descriptor[key] for key in group_keys if key in descriptor), None)
    try:
        group = int(group) if group is not None else None
    except (TypeError, ValueError):
        group = None
    return convrot, group


def _parse_quantization_metadata(metadata):
    raw = metadata.get("_quantization_metadata")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            _fail(
                "SafeTensor _quantization_metadata is not valid JSON.",
                "invalid_quantization_metadata",
                reason=str(exc),
            )
    else:
        value = raw
    if not isinstance(value, Mapping):
        _fail("SafeTensor _quantization_metadata must be a JSON object.", "invalid_quantization_metadata")
    version = str(value.get("format_version", "1.0"))
    if version not in {"1.0", "1", ""}:
        _fail(
            f"Unsupported _quantization_metadata version {version!r}.",
            "unsupported_quantization_metadata_version",
            version=version,
        )
    layers = value.get("layers")
    if not isinstance(layers, Mapping):
        _fail("_quantization_metadata must contain a per-layer object.", "invalid_quantization_metadata")
    normalized = {}
    for raw_name, descriptor in layers.items():
        if not isinstance(raw_name, str) or not isinstance(descriptor, Mapping):
            _fail("Every quantization metadata layer must have a name and object descriptor.", "invalid_quantization_metadata")
        name = _normalize_tensor_name(raw_name).removesuffix(".weight")
        if name in normalized:
            _fail(f"Duplicate quantization metadata for {name!r}.", "duplicate_quantization_metadata", layer=name)
        fmt = _quant_text_class(descriptor.get("format"))
        if fmt in {"unsupported", "fp8_unsupported", "unknown", None}:
            _fail(
                f"Quantization layer {name!r} declares an unsupported format.",
                "unsupported_quantization",
                layer=name,
                value=descriptor.get("format"),
            )
        if fmt == "int8_convrot" and str(descriptor.get("format")).strip().lower().replace("-", "_") not in {
            "int8_tensorwise", "int8_convrot"
        }:
            _fail(f"INT8 header descriptor for {name!r} names a format unsupported by this loader.", "unsupported_quantization", layer=name)
        convrot, group_size = _descriptor_convrot_fields(
            descriptor, fmt, layer=name, error_code="invalid_quantization_metadata"
        )
        orig_dtype = _normalize_dtype(descriptor.get("orig_dtype"))
        if orig_dtype and orig_dtype not in _FLOAT_PARAMETER_DTYPES:
            _fail(
                f"Quantization source dtype for {name!r} must be BF16 or FP16.",
                "unsupported_quantization",
                layer=name,
                orig_dtype=orig_dtype,
            )
        normalized[name] = {
            "format": fmt,
            "convrot": convrot,
            "group_size": group_size,
            "weight_group_size": descriptor.get("group_size") if fmt == "asym_w4a8_int8" and type(descriptor.get("group_size")) is int else None,
            "orig_dtype": orig_dtype,
        }
    return normalized


def _expected_quant_layers():
    return {
        f"blocks.{index}.{suffix}"
        for index in range(_LAYERS)
        for suffix in _QUANT_MODULE_SUFFIXES
    }


def _read_small_payload(reader, info, *, limit, label):
    if reader is None:
        _fail(
            f"A small-payload reader is required to verify {label}; no model weights are needed.",
            "small_tensor_reader_required",
            tensor=info["raw_name"],
        )
    dtype = info["dtype"]
    itemsize = _DTYPE_BYTES.get(dtype)
    if itemsize is None:
        _fail(f"Cannot bound payload for {label} dtype {dtype!r}.", "invalid_tensor_dtype", tensor=info["raw_name"])
    expected = math.prod(info["shape"]) * itemsize
    if expected > limit:
        _fail(f"{label} payload is too large for bounded inspection.", "small_tensor_too_large", tensor=info["raw_name"], expected_bytes=expected, max_bytes=limit)
    try:
        payload = reader(info["raw_name"])
    except Exception as exc:
        _fail(f"Could not read small {label} tensor {info['raw_name']!r}.", "small_tensor_read_failed", tensor=info["raw_name"], reason=str(exc))
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        _fail(f"The reader did not return bytes for {label} tensor {info['raw_name']!r}.", "invalid_small_tensor_payload", tensor=info["raw_name"])
    payload = bytes(payload)
    if len(payload) != expected:
        _fail(
            f"Small {label} tensor {info['raw_name']!r} returned {len(payload)} bytes; expected {expected}.",
            "invalid_small_tensor_payload",
            tensor=info["raw_name"],
            expected_bytes=expected,
            actual_bytes=len(payload),
        )
    return payload


def _quant_descriptor_from_tensor(reader, info):
    payload = _read_small_payload(
        reader,
        info,
        limit=_MAX_QUANT_DESCRIPTOR_BYTES,
        label="Comfy quantization descriptor",
    )
    try:
        value = json.loads(payload.decode("utf-8").rstrip("\0 \t\r\n"))
    except (UnicodeDecodeError, ValueError) as exc:
        _fail(
            f"Comfy quantization descriptor {info['raw_name']!r} is not valid JSON.",
            "invalid_quantization_descriptor",
            tensor=info["raw_name"],
            reason=str(exc),
        )
    if not isinstance(value, Mapping):
        _fail(f"Comfy quantization descriptor {info['raw_name']!r} must be a JSON object.", "invalid_quantization_descriptor", tensor=info["raw_name"])
    fmt = _quant_text_class(value.get("format"))
    if fmt in {"unsupported", "fp8_unsupported", "unknown", None}:
        _fail(
            f"Comfy quantization descriptor {info['raw_name']!r} names an unsupported or unknown format.",
            "unsupported_quantization",
            tensor=info["raw_name"],
            value=value.get("format"),
        )
    convrot, group = _descriptor_convrot_fields(
        value, fmt, layer=info["raw_name"], error_code="invalid_quantization_descriptor"
    )
    return {
        "format": fmt,
        "convrot": convrot,
        "group_size": group,
        "orig_dtype": _normalize_dtype(value.get("orig_dtype")),
    }


def _quantization_profile(tensors, metadata, tensor_reader, *, file_format="safetensors"):
    top_format = _top_quantization(metadata)
    quant_metadata = _parse_quantization_metadata(metadata)
    marker_infos = {
        name[: -len(".comfy_quant")]: info
        for name, info in tensors.items()
        if name.endswith(".comfy_quant")
    }
    marker_metadata = None
    if marker_infos:
        marker_metadata = {
            name: _quant_descriptor_from_tensor(tensor_reader, info)
            for name, info in marker_infos.items()
        }
    if marker_metadata and quant_metadata:
        if set(marker_metadata) != set(quant_metadata):
            _fail(
                "Comfy quantization tensors and _quantization_metadata describe different layer sets.",
                "quantization_conflict",
                marker_layers=len(marker_metadata),
                metadata_layers=len(quant_metadata),
            )
        for name, marker in marker_metadata.items():
            reference = quant_metadata[name]
            if any(
                marker[field] != reference[field]
                for field in ("format", "convrot", "group_size")
            ) or (marker["orig_dtype"] and reference["orig_dtype"] and marker["orig_dtype"] != reference["orig_dtype"]):
                _fail(
                    f"Comfy quantization tensor and _quantization_metadata disagree for {name!r}.",
                    "quantization_conflict",
                    layer=name,
                    marker=marker,
                    metadata=quant_metadata[name],
                )
            if not marker["orig_dtype"]:
                marker["orig_dtype"] = reference["orig_dtype"]

    descriptors = marker_metadata or quant_metadata
    int8_weights = any(
        tensors.get(f"blocks.{index}.{suffix}.weight", {}).get("dtype") == "I8"
        for index in range(_LAYERS)
        for suffix in _QUANT_MODULE_SUFFIXES
    )
    fp8_weights = any(
        tensors.get(f"blocks.{index}.{suffix}.weight", {}).get("dtype") in _FP8_DTYPES
        for index in range(_LAYERS)
        for suffix in _QUANT_MODULE_SUFFIXES
    )
    scales_present = any(name.endswith(".weight_scale") for name in tensors)

    if file_format == "gguf":
        if descriptors or top_format or marker_infos or scales_present:
            _fail("GGUF storage cannot also declare a SafeTensor quantization layout.", "quantization_conflict")
        unsupported = {info["dtype"] for info in tensors.values()} - (_GGUF_PACKED_DTYPES | {"F32", "BF16", "F16"})
        if unsupported:
            _fail("Unsupported H3 GGUF tensor storage type.", "unsupported_quantization", dtypes=sorted(unsupported))
        packed = [info for info in tensors.values() if info["dtype"] in _GGUF_PACKED_DTYPES]
        if not packed:
            _fail("Choose a quantized H3 GGUF diffusion checkpoint.", "unsupported_quantization")
        for name, info in tensors.items():
            if info["dtype"] in _GGUF_PACKED_DTYPES and (not name.endswith(".weight") or len(info["shape"]) != 2):
                _fail("H3 GGUF quantization is supported only for two-dimensional linear weights; curves, norms and biases must stay floating point.", "unsupported_quantization", tensor=name)
        return {
            "quantization_format": "gguf", "convrot": False, "convrot_group_size": None,
            "quantized_layers": len(packed), "tensor_dtypes": sorted({info["dtype"] for info in tensors.values()}),
            "gguf_quant_types": sorted({info["dtype"].removeprefix("GGUF_") for info in packed}),
            "qkv_layout": None, "qkv_layout_evidence": None,
        }

    if descriptors and {v["format"] for v in descriptors.values()} == {"asym_w4a8_int8"}:
        expected = _expected_quant_layers()
        if set(descriptors) != expected:
            _fail("W4A8 H3 requires descriptors for all 200 main-block linear layers.", "incomplete_quantization_metadata")
        groups = {v["group_size"] for v in descriptors.values()}
        weight_groups = {v["weight_group_size"] for v in descriptors.values()}
        if any(not v["convrot"] for v in descriptors.values()) or len(groups) != 1 or not groups <= _SUPPORTED_CONVROT_GROUPS:
            _fail("H3 W4A8 requires a consistent supported ConvRot group size (64 or 256).", "unsupported_convrot_group_size")
        if weight_groups != {16}:
            _fail("H3 W4A8 currently requires the verified weight group size 16.", "unsupported_quantization")
        if top_format and top_format != "asym_w4a8_int8":
            _fail("Header format conflicts with W4A8 layer descriptors.", "quantization_conflict")
        for name in sorted(expected):
            info = tensors.get(name + ".weight")
            if info is None or info["dtype"] != "I8" or len(info["shape"]) != 2:
                _fail("W4A8 weights must use packed two-dimensional INT8 storage.", "unsupported_quantization", tensor=name + ".weight")
            rows, packed_columns = info["shape"]
            columns = 2 * packed_columns
            if columns % 16 or columns % next(iter(groups)):
                _fail("W4A8 input width must align with its weight and ConvRot groups.", "invalid_tensor_shape", tensor=name + ".weight")
            _require_tensor(tensors, name + ".weight_s_rel", (rows, columns // 16), _FP8_DTYPES)
            _require_tensor(tensors, name + ".weight_s_channel", (rows,), {"F32"})
            if name + ".weight_codebook" in tensors:
                _require_tensor(tensors, name + ".weight_codebook", (16,), {"F32"})
            if name + ".weight_correction" in tensors:
                _require_tensor(tensors, name + ".weight_correction", (columns // 16, rows), {"F32"})
            if name + ".weight_scale" in tensors or name + ".comfy_quant" in tensors:
                _fail("W4A8 cannot also contain a different quantized layout for the same layer.", "quantization_conflict", layer=name)
        return {
            "quantization_format": "asym_w4a8_int8", "convrot": True,
            "convrot_group_size": next(iter(groups)), "weight_group_size": 16,
            "quantized_layers": len(descriptors), "tensor_dtypes": sorted({info["dtype"] for info in tensors.values()}),
            "qkv_layout": "grouped", "qkv_layout_evidence": "WanGP/DaSiWa W4A8 ConvRot export uses contiguous grouped Q/K/V rows",
        }

    if int8_weights:
        if not descriptors:
            _fail(
                "INT8 H3 weights need actual Comfy descriptors or _quantization_metadata proving ConvRot format and group size.",
                "quantization_evidence_required",
            )
        expected = _expected_quant_layers()
        if set(descriptors) != expected:
            _fail(
                "INT8 H3 requires one recognized quantization descriptor for each of its 200 main-block linear layers.",
                "incomplete_quantization_metadata",
                missing=sorted(expected - set(descriptors))[:8],
                extra=sorted(set(descriptors) - expected)[:8],
            )
        formats = {value["format"] for value in descriptors.values()}
        groups = {value["group_size"] for value in descriptors.values()}
        if formats != {"int8_convrot"} or any(not value["convrot"] for value in descriptors.values()):
            _fail("H3 INT8 is supported only when every main-block linear descriptor proves ConvRot INT8.", "unsupported_quantization")
        if len(groups) != 1 or None in groups:
            _fail("H3 ConvRot descriptors must agree on one explicit group size.", "quantization_conflict", group_sizes=sorted(str(v) for v in groups))
        group_size = next(iter(groups))
        if group_size not in _SUPPORTED_CONVROT_GROUPS:
            _fail(
                f"H3 ConvRot group size {group_size} is unsupported; this runtime supports groups 64 and 256.",
                "unsupported_convrot_group_size",
                group_size=group_size,
                supported=sorted(_SUPPORTED_CONVROT_GROUPS),
            )
        if top_format and top_format != "int8_convrot":
            _fail("Header-level quantization format conflicts with INT8 ConvRot descriptors.", "quantization_conflict")
        if fp8_weights:
            _fail("A checkpoint cannot mix INT8 ConvRot and FP8 transformer weights.", "mixed_quantization")
        for name, value in descriptors.items():
            if value["orig_dtype"] and value["orig_dtype"] not in {"BF16", "F16"}:
                _fail(f"ConvRot source dtype for {name!r} must be BF16 or FP16.", "unsupported_quantization", layer=name, orig_dtype=value["orig_dtype"])
        return {
            "quantization_format": "int8_tensorwise",
            "convrot": True,
            "convrot_group_size": group_size,
            "quantized_layers": len(descriptors),
            "tensor_dtypes": sorted({info["dtype"] for info in tensors.values()}),
            "qkv_layout": "grouped",
            "qkv_layout_evidence": "Comfy INT8 ConvRot stores contiguous grouped Q/K/V rows",
        }

    if fp8_weights:
        if top_format != "scaled_fp8" and not descriptors:
            _fail(
                "FP8 H3 weights need explicit scaled-FP8 metadata; an FP8 dtype or filename alone is not enough.",
                "quantization_evidence_required",
            )
        if descriptors:
            expected = _expected_quant_layers()
            if set(descriptors) != expected or {v["format"] for v in descriptors.values()} != {"scaled_fp8"}:
                _fail("Scaled-FP8 metadata must describe every main-block linear layer with one consistent format.", "incomplete_quantization_metadata")
            if any(value["convrot"] for value in descriptors.values()):
                _fail("Scaled FP8 descriptors cannot also declare ConvRot INT8.", "quantization_conflict")
        if top_format and top_format != "scaled_fp8":
            _fail("Header-level quantization format conflicts with scaled-FP8 tensors.", "quantization_conflict")
        if int8_weights:
            _fail("A checkpoint cannot mix INT8 ConvRot and FP8 transformer weights.", "mixed_quantization")
        return {
            "quantization_format": "scaled_fp8",
            "convrot": False,
            "convrot_group_size": None,
            "quantized_layers": len(descriptors or {}),
            "tensor_dtypes": sorted({info["dtype"] for info in tensors.values()}),
            "qkv_layout": None,
            "qkv_layout_evidence": None,
        }

    if descriptors or (top_format and top_format != "plain"):
        _fail(
            "Quantization metadata is present, but the transformer weights are not a supported consistent quantized format.",
            "quantization_conflict",
        )
    if scales_present:
        _fail("Scale tensors are present without a supported explicit quantization format.", "quantization_evidence_required")
    return {
        "quantization_format": "none",
        "convrot": False,
        "convrot_group_size": None,
        "quantized_layers": 0,
        "tensor_dtypes": sorted({info["dtype"] for info in tensors.values()}),
        "qkv_layout": None,
        "qkv_layout_evidence": None,
    }


def _require_tensor(tensors, name, shape, dtypes=None):
    info = tensors.get(name)
    expected_shape = tuple(shape)
    if info is None:
        _fail(f"Required H3 tensor {name!r} is missing.", "missing_tensor", tensor=name)
    if info["shape"] != expected_shape:
        _fail(
            f"H3 tensor {name!r} has shape {_shape_text(info['shape'])}; expected {_shape_text(expected_shape)}.",
            "tensor_shape_mismatch",
            tensor=name,
            expected=list(expected_shape),
            actual=list(info["shape"]),
        )
    if dtypes is not None and info["dtype"] not in set(dtypes):
        _fail(
            f"H3 tensor {name!r} has unsupported dtype {info['dtype']}; expected one of {', '.join(sorted(dtypes))}.",
            "unsupported_tensor_dtype",
            tensor=name,
            expected=sorted(dtypes),
            actual=info["dtype"],
        )
    return info


def _validate_shape_groups(tensors, *, compressed, quantization_format):
    modulation_dim = _CURVE_DIM if compressed else _FULL_TIME_DIM
    main_weight_dtypes = {"I8"} if quantization_format == "int8_tensorwise" else (
        _FP8_DTYPES if quantization_format == "scaled_fp8" else _FLOAT_PARAMETER_DTYPES
    )
    dense_dtypes = _FLOAT_PARAMETER_DTYPES
    if quantization_format == "gguf":
        # GGUF may quantize the condition/time/refiner linears too. Keep
        # norms, biases, and the AdaLN curve in floating point.
        main_weight_dtypes = _GGUF_PACKED_DTYPES | {"F32", "BF16", "F16"}
        dense_dtypes = {"F32", "BF16", "F16"}
    elif quantization_format == "asym_w4a8_int8":
        main_weight_dtypes = {"I8"}

    def require(name, shape, dtypes):
        if quantization_format == "gguf" and name.endswith(".weight") and len(shape) == 2:
            dtypes = dtypes | _GGUF_PACKED_DTYPES
        if quantization_format == "asym_w4a8_int8" and name.removesuffix(".weight") in _expected_quant_layers():
            shape = (shape[0], shape[1] // 2)
        return _require_tensor(tensors, name, shape, dtypes)

    # Distinct H3 audio/video anchors prevent a superficially similar video-only
    # transformer from being accepted as the joint omni-modal model.
    retained_projection_dtypes = dense_dtypes | {"F32"}
    for name, shape in (
        ("video_patch_proj.weight", (_HIDDEN, 96)),
        ("audio_patch_proj.weight", (_HIDDEN, 32)),
        ("condition_proj.weight", (_HIDDEN, 5120)),
        ("final_layer.video_out.weight", (96, _HIDDEN)),
        ("final_layer.audio_out.weight", (32, _HIDDEN)),
        ("final_layer.norm.weight", (_HIDDEN,)),
        ("token_refiner.final_norm.weight", (_HIDDEN,)),
        ("rope.inv_freq", (16,)),
    ):
        if name == "rope.inv_freq":
            # Community exports may cast this buffer along with the model.
            # RoPE multiplies it by FP32 positions, promoting BF16/FP16
            # storage back to FP32; quantized frequencies remain unsupported.
            allowed = {"F32", "BF16", "F16"}
        elif quantization_format in {"int8_tensorwise", "scaled_fp8", "asym_w4a8_int8"} and name in {
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "final_layer.video_out.weight",
            "final_layer.audio_out.weight",
        }:
            # Comfy's supported INT8 and FP8 exports keep these small I/O
            # projections in FP32 while quantizing only the 200 block
            # linears.  Keep FP32 disallowed for the large condition/time,
            # modulation, norm, and plain transformer parameters.
            allowed = retained_projection_dtypes
        else:
            allowed = dense_dtypes
        require(name, shape, allowed)
    for name in ("video_patch_proj.bias", "audio_patch_proj.bias"):
        _require_tensor(tensors, name, (_HIDDEN,), {"F32", "BF16", "F16"})
    _require_tensor(tensors, "condition_proj.bias", (_HIDDEN,), dense_dtypes)
    _require_tensor(tensors, "final_layer.video_out.bias", (96,), {"F32", "BF16", "F16"})
    _require_tensor(tensors, "final_layer.audio_out.bias", (32,), {"F32", "BF16", "F16"})

    table_info = tensors.get("adaln_t_table")
    if compressed:
        table_info = _require_tensor(tensors, "adaln_t_table", (_CURVE_GRID, _CURVE_DIM),
                                     {"F32", "F16", "BF16"} if quantization_format == "gguf" else {"F32"})
    elif table_info is not None:
        _fail("A full H3 checkpoint cannot also contain a compressed AdaLN table.", "adaln_layout_conflict")

    for name, shape in (
        ("final_layer.adaln_proj.linear.weight", (2 * _HIDDEN, modulation_dim)),
        ("final_layer.adaln_proj.linear.bias", (2 * _HIDDEN,)),
    ):
        require(name, shape, dense_dtypes)

    if compressed:
        time_embedder = [name for name in tensors if name.startswith("time_embedder.")]
        if time_embedder:
            _fail("Compressed H3 must use its finite rank-8 AdaLN table and cannot include the full timestep MLP.", "adaln_layout_conflict")
    else:
        time_shapes = {
            "time_embedder.proj_in.weight": (5376, 256),
            "time_embedder.proj_in.bias": (5376,),
            "time_embedder.proj_out.weight": (_FULL_TIME_DIM, 5376),
            "time_embedder.proj_out.bias": (_FULL_TIME_DIM,),
        }
        for name, shape in time_shapes.items():
            require(name, shape, dense_dtypes)

    for index in range(_LAYERS):
        prefix = f"blocks.{index}."
        for relative, shape in (
            ("norm1.weight", (_HIDDEN,)),
            ("norm2.weight", (_HIDDEN,)),
            ("attn.qkv_proj.weight", (_QKV_ROWS, _HIDDEN)),
            ("attn.q_norm.weight", (128,)),
            ("attn.k_norm.weight", (128,)),
            ("attn.out_proj.weight", (_HIDDEN, _ATTENTION_INNER)),
            ("mlp.fc1.weight", (2 * _FFN, _HIDDEN)),
            ("mlp.fc2.weight", (_HIDDEN, _FFN)),
            ("adaln_proj.linear.weight", (18 * _HIDDEN, modulation_dim)),
            ("adaln_proj.linear.bias", (18 * _HIDDEN,)),
        ):
            quantized_linear = relative.startswith(("attn.qkv_proj.", "attn.out_proj.", "mlp."))
            weight_dtypes = main_weight_dtypes if relative.endswith(".weight") and quantized_linear else dense_dtypes
            require(prefix + relative, shape, weight_dtypes)
        if quantization_format == "int8_tensorwise":
            for relative, rows in (
                ("attn.qkv_proj", _QKV_ROWS),
                ("attn.out_proj", _HIDDEN),
                ("mlp.fc1", 2 * _FFN),
                ("mlp.fc2", _HIDDEN),
            ):
                base = prefix + relative
                _require_tensor(tensors, base + ".weight_scale", (rows, 1), {"F32"})
                marker = tensors.get(base + ".comfy_quant")
                # The quantization profile already requires all 200 ConvRot
                # descriptors, either in _quantization_metadata or in marker
                # tensors. Header-based exports need no redundant marker, but
                # any marker that is present must retain its bounded U8 shape.
                if marker is not None and (
                    marker["dtype"] != "U8"
                    or len(marker["shape"]) != 1
                    or math.prod(marker["shape"]) <= 0
                    or math.prod(marker["shape"]) > _MAX_QUANT_DESCRIPTOR_BYTES
                ):
                    _fail(
                        f"INT8 H3 layer {base!r} needs a small one-dimensional U8 Comfy descriptor tensor.",
                        "invalid_quantization_descriptor",
                        tensor=base + ".comfy_quant",
                    )
        elif quantization_format == "scaled_fp8":
            for relative, rows in (
                ("attn.qkv_proj", _QKV_ROWS),
                ("attn.out_proj", _HIDDEN),
                ("mlp.fc1", 2 * _FFN),
                ("mlp.fc2", _HIDDEN),
            ):
                scale = tensors.get(prefix + relative + ".weight_scale")
                if scale is None:
                    scale = tensors.get(prefix + relative + ".scale_weight")
                if scale is None or scale["shape"] not in {(), (1,), (1, 1), (rows, 1)} or scale["dtype"] != "F32":
                    _fail(
                        f"Scaled-FP8 layer {prefix + relative!r} needs an F32 scalar or column of row scales supported by this loader.",
                        "missing_fp8_scale",
                        layer=prefix + relative,
                    )

    for index in range(_TOKEN_REFINER_LAYERS):
        prefix = f"token_refiner.blocks.{index}."
        for relative, shape in (
            ("norm1.weight", (_HIDDEN,)),
            ("norm2.weight", (_HIDDEN,)),
            ("attn.qkv_proj.weight", (_QKV_ROWS, _HIDDEN)),
            ("attn.q_norm.weight", (128,)),
            ("attn.k_norm.weight", (128,)),
            ("attn.out_proj.weight", (_HIDDEN, _ATTENTION_INNER)),
            ("mlp.fc1.weight", (2 * _FFN, _HIDDEN)),
            ("mlp.fc2.weight", (_HIDDEN, _FFN)),
        ):
            require(prefix + relative, shape, dense_dtypes)

    block_numbers = set()
    for name in tensors:
        match = re.match(r"^blocks\.(\d+)\.", name)
        if match:
            block_numbers.add(int(match.group(1)))
    if block_numbers != set(range(_LAYERS)):
        missing = sorted(set(range(_LAYERS)) - block_numbers)
        extra = sorted(block_numbers - set(range(_LAYERS)))
        _fail(
            "MiniMax H3 requires exactly 50 complete transformer blocks (0 through 49).",
            "invalid_block_count",
            missing=missing,
            extra=extra,
        )
    refiner_numbers = set()
    for name in tensors:
        match = re.match(r"^token_refiner\.blocks\.(\d+)\.", name)
        if match:
            refiner_numbers.add(int(match.group(1)))
    if refiner_numbers != set(range(_TOKEN_REFINER_LAYERS)):
        _fail(
            "MiniMax H3 requires both token-refiner blocks.",
            "invalid_token_refiner",
            missing=sorted(set(range(_TOKEN_REFINER_LAYERS)) - refiner_numbers),
            extra=sorted(refiner_numbers - set(range(_TOKEN_REFINER_LAYERS))),
        )

    if quantization_format == "none":
        allowed_all = _FLOAT_PARAMETER_DTYPES | {"F32"}
        unsupported = sorted({info["dtype"] for info in tensors.values()} - allowed_all)
        if unsupported:
            _fail(
                f"Plain H3 checkpoint contains unsupported storage dtype(s): {', '.join(unsupported)}.",
                "unsupported_tensor_dtype",
                dtypes=unsupported,
            )
    elif quantization_format == "int8_tensorwise":
        allowed_all = _FLOAT_PARAMETER_DTYPES | {"F32", "I8", "U8"}
        unsupported = sorted({info["dtype"] for info in tensors.values()} - allowed_all)
        if unsupported:
            _fail(
                f"INT8 ConvRot H3 checkpoint contains unsupported storage dtype(s): {', '.join(unsupported)}.",
                "unsupported_tensor_dtype",
                dtypes=unsupported,
            )
    elif quantization_format == "scaled_fp8":
        allowed_all = _FLOAT_PARAMETER_DTYPES | {"F32", "U8"} | _FP8_DTYPES
        unsupported = sorted({info["dtype"] for info in tensors.values()} - allowed_all)
        if unsupported:
            _fail(
                f"Scaled-FP8 H3 checkpoint contains unsupported storage dtype(s): {', '.join(unsupported)}.",
                "unsupported_tensor_dtype",
                dtypes=unsupported,
            )
    elif quantization_format == "asym_w4a8_int8":
        allowed_all = _FLOAT_PARAMETER_DTYPES | {"F32", "I8"} | _FP8_DTYPES
        unsupported = sorted({info["dtype"] for info in tensors.values()} - allowed_all)
        if unsupported:
            _fail("W4A8 H3 contains unsupported tensor storage types.", "unsupported_tensor_dtype", dtypes=unsupported)

    return table_info


def _validate_modulation_payload(reader, table_info):
    payload = _read_small_payload(
        reader,
        table_info,
        limit=_MAX_SMALL_TENSOR_BYTES,
        label="rank-8 AdaLN table",
    )
    count = _CURVE_GRID * _CURVE_DIM
    if table_info["dtype"] == "F16":
        values = struct.unpack("<" + "e" * count, payload)
    elif table_info["dtype"] == "BF16":
        # Reconstruct BF16 values without importing torch into preflight.
        values = (struct.unpack("<f", struct.pack("<I", value << 16))[0]
                  for value in struct.unpack("<" + "H" * count, payload))
    else:
        values = struct.unpack("<" + "f" * count, payload)
    if not all(math.isfinite(value) for value in values):
        _fail(
            "The rank-8 H3 AdaLN table contains NaN or infinite values.",
            "nonfinite_adaln_table",
            tensor=table_info["raw_name"],
        )


def _qkv_layout(tensors, metadata, quantization, source_pin=None):
    candidates = []
    if quantization["qkv_layout"]:
        candidates.append((quantization["qkv_layout"], quantization["qkv_layout_evidence"]))
    if source_pin and source_pin.get("qkv_layout"):
        candidates.append((source_pin["qkv_layout"], "published SHA-256 of the official Comfy-Org H3 export"))
    for key in ("modelspec.qkv_layout", "qkv_layout", "modelspec.qkv_order", "qkv_order"):
        value = str(metadata.get(key) or "").strip().lower().replace("-", "_")
        if not value:
            continue
        if value in {"grouped", "contiguous", "contiguous_qkv", "q_k_v"}:
            candidates.append(("grouped", f"header metadata {key}"))
        elif value in {"interleaved", "head_interleaved", "head_qkv_interleaved"}:
            candidates.append(("interleaved", f"header metadata {key}"))
        else:
            _fail(f"Header QKV metadata {key!r} has unknown value {value!r}.", "unsupported_qkv_layout", field=key, value=value)
    raw_qmeta = metadata.get("_quantization_metadata")
    if isinstance(raw_qmeta, str):
        # Per-layer quant metadata may carry an exporter declaration alongside
        # its format table. It is used only when the declaration is explicit.
        try:
            qmeta = json.loads(raw_qmeta)
        except ValueError:
            qmeta = {}
        value = str(qmeta.get("qkv_layout") or "").strip().lower().replace("-", "_") if isinstance(qmeta, Mapping) else ""
        if value:
            if value in {"grouped", "contiguous", "contiguous_qkv", "q_k_v"}:
                candidates.append(("grouped", "_quantization_metadata.qkv_layout"))
            elif value in {"interleaved", "head_interleaved", "head_qkv_interleaved"}:
                candidates.append(("interleaved", "_quantization_metadata.qkv_layout"))
            else:
                _fail(f"_quantization_metadata has unknown QKV layout {value!r}.", "unsupported_qkv_layout", value=value)
    source_layout = str(metadata.get("source_qkv_layout") or "").strip().lower().replace("-", "_")
    if source_layout:
        if source_layout in {"grouped", "contiguous", "contiguous_qkv", "q_k_v"}:
            candidates.append(("grouped", "header metadata source_qkv_layout"))
        elif source_layout in {"interleaved", "head_interleaved", "head_qkv_interleaved"}:
            candidates.append(("interleaved", "header metadata source_qkv_layout"))
        else:
            _fail(f"Header source_qkv_layout has unknown value {source_layout!r}.", "unsupported_qkv_layout", value=source_layout)
    if candidates:
        layouts = {value for value, _ in candidates}
        if len(layouts) != 1:
            _fail("Header/exporter metadata declares conflicting QKV layouts.", "qkv_layout_conflict", evidence=candidates)
        return candidates[0][0], "; ".join(source for _, source in candidates)

    implementation = str(metadata.get("modelspec.implementation") or "").lower()
    compressed = "adaln_t_table" in tensors
    if "github.com/minimax-ai/minimax-h3" in implementation and not compressed:
        return "interleaved", "official MiniMax H3 implementation metadata on the full checkpoint"
    _fail(
        "QKV row order cannot be proven from this header. Add explicit qkv_layout/qkv_order exporter metadata; do not import by guessing.",
        "qkv_layout_ambiguous",
        choices=["grouped", "interleaved"],
    )


def _settings_for_profile(profile, source, metadata, pinned, *, explicit_selection=False):
    if pinned:
        settings = {
            key: pinned[key]
            for key in (
                "sampler",
                "video_shift",
                "audio_shift",
                "default_steps",
                "min_steps",
                "max_steps",
                "baked_turbo",
                "fused_turbo",
            )
        }
        return settings, {"source": "pinned published identity and SHA-256"}

    if profile is None:
        return {
            "sampler": None,
            "video_shift": None,
            "audio_shift": None,
            "default_steps": None,
            "min_steps": None,
            "max_steps": None,
            "baked_turbo": False,
            "fused_turbo": False,
        }, {}

    source_fields = ("name", "versionName") if _source_has_mixed_recipes(source) else (
        "name", "versionName", "versionDescription", "description"
    )
    texts = [
        _source_text(source, source_fields),
        _metadata_text(metadata, ("modelspec.title", "modelspec.description", "sampler", "scheduler")),
    ]
    text = "\n".join(texts).lower().replace("：", ":").replace("–", "-").replace("—", "-")
    sampler = None
    sampler_match = re.search(r"\b(?:sampler|sampling method)\s*[:=]\s*([a-z][a-z0-9_ -]*)", text)
    if sampler_match:
        candidate = sampler_match.group(1).strip().split()[0].strip(".,;:")
        if candidate in {"euler", "res_multistep"}:
            sampler = candidate
    if sampler is None and "res_multistep" in text:
        sampler = "res_multistep"
    if sampler is None and re.search(r"\beuler\b", text) and re.search(r"\b(?:use|using|sampler|sampling)\s+euler\b", text):
        sampler = "euler"

    def shift(label):
        match = re.search(
            rf"\b(?:shift\s+{label}|{label}\s+shift|{label}_shift)\s*[:=]?\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?",
            text,
        )
        if not match:
            return None, None
        lower = float(match.group(1))
        upper = float(match.group(2)) if match.group(2) else None
        if upper is None:
            return lower, None
        # A creator-published range is explicit evidence. Use its midpoint as
        # the default while retaining the range for the caller's UI.
        midpoint = (lower + upper) / 2
        return (int(midpoint) if midpoint.is_integer() else midpoint), [lower, upper]

    video_shift, video_shift_range = shift("video")
    audio_shift, audio_shift_range = shift("audio")
    step_ranges = [
        (int(match.group(1)), int(match.group(2)))
        for match in re.finditer(
            r"\bsteps?\s*[:=]\s*(\d+)\s*(?:-|to|till)\s*(\d+)\b",
            text,
        )
    ]
    step_ranges.extend(
        (int(match.group(1)), int(match.group(2)))
        for match in re.finditer(
            r"\b(\d+)\s*(?:-|to|till)\s*(\d+)\s*(?:denoising\s*)?steps?\b",
            text,
        )
    )
    step_ranges.extend(
        (int(match.group(1)), int(match.group(2)))
        for match in re.finditer(r"\b(\d+)\s*(?:-|to|till)\s*(\d+)\s*[- ]step\b", text)
    )
    min_steps = min((pair[0] for pair in step_ranges), default=None)
    max_steps = max((pair[1] for pair in step_ranges), default=None)
    single_step = re.search(
        r"\b(?:default|start at|recommended)\s*(?:of\s*)?(\d+)\s*(?:denoising\s*)?steps?\b",
        text,
    )
    if not single_step:
        single_step = re.search(r"\b(\d+)\s*[- ]steps?\s+(?:model|checkpoint)s?\b", text)
    if not single_step:
        single_step = re.search(r"\b(\d+)\s+steps?\b", text)
    default_steps = int(single_step.group(1)) if single_step else None
    if default_steps is None and max_steps is not None:
        # When the creator documents only an accepted range, start from its
        # highest quality evaluation count and keep the full range available.
        default_steps = max_steps
    if default_steps is None:
        exact = re.search(r"\bsteps?\s*[:=]\s*(\d+)\b", text)
        if exact:
            default_steps = int(exact.group(1))
            min_steps = max_steps = default_steps

    settings = {
        "sampler": sampler,
        "video_shift": video_shift,
        "audio_shift": audio_shift,
        "default_steps": default_steps,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "baked_turbo": profile in {"turbo", "fused"},
        "fused_turbo": profile == "fused",
    }
    evidence = {"source": "explicit CivitAI version/header recipe text"}
    if video_shift_range:
        settings["video_shift_range"] = video_shift_range
        evidence["video_shift"] = "midpoint of the creator-published range"
    if audio_shift_range:
        settings["audio_shift_range"] = audio_shift_range
        evidence["audio_shift"] = "midpoint of the creator-published range"
    if step_ranges:
        evidence["default_steps"] = "maximum of the creator-published step range" if default_steps == max_steps else "creator-published default"

    if profile == "standard":
        defaults = {
            "sampler": "euler",
            "video_shift": 12,
            "audio_shift": 3,
            "default_steps": 20,
            "min_steps": 2,
            "max_steps": 50,
        }
        for key, fallback in defaults.items():
            if settings[key] is None:
                settings[key] = fallback
        evidence["fallback"] = "Maestro stock H3 standard defaults"
    elif explicit_selection and profile == "turbo":
        defaults = {
            "sampler": "euler",
            "video_shift": 12,
            "audio_shift": 3,
            "default_steps": 8,
            "min_steps": 4,
            "max_steps": 8,
        }
        for key, fallback in defaults.items():
            if settings[key] is None:
                settings[key] = fallback
        evidence["fallback"] = "explicit user selection of the Maestro H3 Turbo default recipe"
    elif explicit_selection and profile == "fused":
        defaults = {
            "sampler": "res_multistep",
            "video_shift": 12,
            "audio_shift": 3,
            "default_steps": 4,
            "min_steps": 4,
            "max_steps": 12,
        }
        for key, fallback in defaults.items():
            if settings[key] is None:
                settings[key] = fallback
        evidence["fallback"] = "explicit user selection of the Maestro H3 fused-Turbo default recipe"
    return settings, evidence


def inspect_h3_header(
    header,
    *,
    tensor_reader: Callable[[str], bytes] | None = None,
    source: Mapping[str, Any] | None = None,
    sampling_profile: str = "auto",
    native_workflow: str = "auto",
    qkv_layout: str = "auto",
    file_format: str = "safetensors",
) -> dict:
    """Verify a joint MiniMax H3 SafeTensor header without loading model weights.

    ``tensor_reader`` is called only for the 32.8 KiB rank-8 AdaLN table and
    small ``.comfy_quant`` JSON marker tensors.  All transformer tensor data
    remains untouched.

    Ambiguous workflow/recipe selections return ``status='needs_selection'``
    with actionable option fields. Structural incompatibilities raise
    :class:`H3CheckpointError`.
    """

    tensors, metadata = _header_parts(header)
    source = source if isinstance(source, Mapping) else {}
    normalized_sampling = _selection(sampling_profile, _SUPPORTED_SAMPLING_PROFILES, "sampling_profile")
    normalized_workflow = _selection(native_workflow, _SUPPORTED_NATIVE_WORKFLOWS, "native_workflow")
    normalized_qkv = _selection(qkv_layout, {"grouped", "interleaved"}, "qkv_layout")
    if file_format not in {"safetensors", "gguf"}:
        _fail("Unsupported H3 checkpoint container.", "unsupported_checkpoint_format")

    lowered_names = [name.lower() for name in tensors]
    for name in lowered_names:
        if any(token in name for token in ("lora_", ".lora", "lora_down", "lora_up", "adapter.")):
            _fail("LoRA/adapter tensors cannot be imported as a base H3 checkpoint.", "unsupported_checkpoint_component", component="lora")
        if re.search(r"(^|\.)(?:video_vae|audio_vae|vae)(?:\.|$)", name):
            _fail("VAE tensors cannot be imported as an H3 transformer checkpoint.", "unsupported_checkpoint_component", component="vae")
        if re.search(r"(^|\.)vdn(?:\.|$)", name) or ".attn.vdn." in name:
            _fail("H3 VDN/custom attention tensors are not supported by the generic checkpoint importer.", "unsupported_checkpoint_component", component="vdn")

    top_format = _top_quantization(metadata)
    source_pin = _source_pin(source)
    quantization = _quantization_profile(tensors, metadata, tensor_reader, file_format=file_format)
    if source_pin and source_pin.get("require_dasiwa_header"):
        title = str(metadata.get("modelspec.title") or "").lower()
        if not title or "minimax_h3" not in title or "ref2va" not in title:
            _fail(
                "Pinned DaSiWa source metadata does not match the expected Ref2VA H3 header identity.",
                "pinned_header_mismatch",
                modelspec_title=metadata.get("modelspec.title"),
            )
        if not any(token in title for token in ("pruned", "hybrid")):
            _fail("Pinned DaSiWa source is expected to use the rank-8 pruned hybrid H3 layout.", "pinned_header_mismatch", modelspec_title=metadata.get("modelspec.title"))
    if top_format and top_format != quantization["quantization_format"]:
        expected_top = "int8_convrot" if quantization["quantization_format"] == "int8_tensorwise" else quantization["quantization_format"]
        if top_format == "plain" and quantization["quantization_format"] == "none":
            expected_top = "plain"
        if top_format != expected_top:
            _fail("Header quantization label conflicts with tensor descriptors.", "quantization_conflict", header=top_format, tensors=quantization["quantization_format"])

    compressed = "adaln_t_table" in tensors
    if source_pin and source_pin.get("quantization_format") and (
        not compressed or source_pin["quantization_format"] != quantization["quantization_format"]
    ):
        _fail("This published H3 file identity does not match its expected pruned quantization layout.", "pinned_header_mismatch")
    table_info = _validate_shape_groups(
        tensors,
        compressed=compressed,
        quantization_format=quantization["quantization_format"],
    )
    if compressed:
        _validate_modulation_payload(tensor_reader, table_info)

    try:
        qkv_layout, qkv_evidence = _qkv_layout(tensors, metadata, quantization, source_pin)
    except H3CheckpointError as exc:
        if file_format != "gguf" or exc.code != "qkv_layout_ambiguous":
            raise
        qkv_layout, qkv_evidence = normalized_qkv, ("creator's QKV layout explicitly selected at import" if normalized_qkv else None)
    if normalized_qkv and qkv_layout != normalized_qkv:
        _fail("The selected QKV row order conflicts with the checkpoint's exporter metadata.", "qkv_layout_conflict")
    if (
        quantization["quantization_format"] == "scaled_fp8"
        and qkv_layout == "interleaved"
    ):
        _fail(
            "This runtime can load scaled-FP8 H3 only with grouped QKV rows; interleaved FP8 must be converted or verified by a compatible loader before import.",
            "unsupported_qkv_layout",
            quantization_format="scaled_fp8",
            qkv_layout=qkv_layout,
        )
    detected_workflow = _detect_workflow(metadata, source, source_pin)
    detected_sampling = _sampling_profile_from_source(metadata, source, source_pin)
    if normalized_workflow and detected_workflow and normalized_workflow != detected_workflow:
        _fail(
            "The requested native workflow conflicts with explicit checkpoint/source metadata.",
            "native_workflow_conflict",
            requested=normalized_workflow,
            detected=detected_workflow,
        )
    if normalized_sampling and detected_sampling and normalized_sampling != detected_sampling:
        _fail(
            "The requested sampling profile conflicts with explicit checkpoint/source metadata.",
            "sampling_profile_conflict",
            requested=normalized_sampling,
            detected=detected_sampling,
        )
    chosen_workflow = normalized_workflow or detected_workflow
    chosen_sampling = normalized_sampling or detected_sampling

    needs_selection = []
    selection_options = {}
    if qkv_layout is None:
        needs_selection.append("qkv_layout")
        selection_options["qkv_layout"] = ["grouped", "interleaved"]
    if chosen_workflow is None:
        needs_selection.append("native_workflow")
        selection_options["native_workflow"] = ["fl2va", "ref2va"]
    if chosen_sampling is None:
        needs_selection.append("sampling_profile")
        selection_options["sampling_profile"] = ["standard", "turbo", "fused"]

    settings, settings_evidence = _settings_for_profile(
        chosen_sampling,
        source,
        metadata,
        source_pin,
        explicit_selection=normalized_sampling is not None,
    )
    required_recipe_fields = (
        "sampler",
        "video_shift",
        "audio_shift",
        "default_steps",
        "min_steps",
        "max_steps",
    )
    incomplete_recipe = [
        key for key in required_recipe_fields if settings.get(key) is None
    ]
    if chosen_sampling in {"turbo", "fused"} and incomplete_recipe:
        needs_selection.append("sampling_recipe")
        selection_options["sampling_recipe"] = {
            "profile": chosen_sampling,
            "missing_fields": incomplete_recipe,
            "action": "Confirm this profile to use its explicit Maestro default recipe, or provide a complete creator-published schedule.",
        }
    if source_pin and source_pin.get("require_dasiwa_header"):
        # A pinned known release must agree with the observed H3 body too.
        expected_format = source_pin.get("quantization_format", "int8_tensorwise")
        if quantization["quantization_format"] != expected_format or not quantization["convrot"] or quantization["convrot_group_size"] != 256 or not compressed:
            _fail("Pinned DaSiWa v3 source does not match its rank-8 ConvRot group-256 layout.", "pinned_header_mismatch")

    if compressed:
        fl2va_model_type = "minimax_h3"
        ref2va_model_type = "minimax_h3_ref2va"
    else:
        fl2va_model_type = "minimax_h3_full"
        ref2va_model_type = "minimax_h3_ref2va_full"
    supported_architectures = (
        [fl2va_model_type, ref2va_model_type]
        if chosen_workflow == "ref2va"
        else ([fl2va_model_type] if chosen_workflow == "fl2va" else [])
    )
    template_model_type = (
        ref2va_model_type if chosen_workflow == "ref2va"
        else (fl2va_model_type if chosen_workflow == "fl2va" else None)
    )
    profile = {
        "status": "needs_selection" if needs_selection else "verified",
        "needs_selection": needs_selection,
        "selection_options": selection_options,
        "compressed_modulation": compressed,
        "adaln_curve_grid": _CURVE_GRID if compressed else None,
        "time_embed_dim": _CURVE_DIM if compressed else _FULL_TIME_DIM,
        "quantization_format": quantization["quantization_format"],
        "file_format": file_format,
        "weight_group_size": quantization.get("weight_group_size"),
        "gguf_quant_types": quantization.get("gguf_quant_types", []),
        "convrot": quantization["convrot"],
        "convrot_group_size": quantization["convrot_group_size"],
        "qkv_layout": qkv_layout,
        "qkv_layout_evidence": qkv_evidence,
        "qkv_layout_selection": normalized_qkv or "auto",
        "native_workflow": chosen_workflow,
        "sampling_profile": chosen_sampling,
        "template_model_type": template_model_type,
        "architectures": supported_architectures,
        "sampler": settings["sampler"],
        "video_shift": settings["video_shift"],
        "audio_shift": settings["audio_shift"],
        "default_steps": settings["default_steps"],
        "min_steps": settings["min_steps"],
        "max_steps": settings["max_steps"],
        "baked_turbo": settings["baked_turbo"],
        "fused_turbo": settings["fused_turbo"],
        "default_settings": settings,
        "settings_evidence": settings_evidence,
        "tensor_dtypes": quantization["tensor_dtypes"],
        "quantized_layers": quantization["quantized_layers"],
        "verified_anchors": [
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "condition_proj.weight",
            "final_layer.video_out.weight",
            "final_layer.audio_out.weight",
            "blocks.0.attn.qkv_proj.weight",
            "blocks.49.attn.qkv_proj.weight",
        ],
        "source_identity": {
            key: source.get(key)
            for key in ("modelId", "versionId", "fileId", "sha256")
            if source.get(key) is not None
        },
    }
    profile["model_overrides"] = {
        "architecture": template_model_type,
        "minimax_h3_qkv_layout": qkv_layout,
        "minimax_h3_checkpoint_requirements": {
            "compressed_modulation": compressed,
            "adaln_curve_grid": profile["adaln_curve_grid"],
            "time_embed_dim": profile["time_embed_dim"],
            "quantization_format": profile["quantization_format"],
            "convrot": profile["convrot"],
            "convrot_group_size": profile["convrot_group_size"],
            "qkv_layout": qkv_layout,
        },
    }
    if chosen_sampling:
        profile["model_overrides"].update(
            {
                "minimax_h3_sampler": settings["sampler"],
                "minimax_h3_video_shift": settings["video_shift"],
                "minimax_h3_audio_shift": settings["audio_shift"],
                "minimax_h3_unaccelerated_default_steps": settings["default_steps"],
                "minimax_h3_baked_turbo": settings["baked_turbo"],
                "minimax_h3_fused_turbo": settings["fused_turbo"],
            }
        )
    return profile


__all__ = ["H3CheckpointError", "inspect_h3_header"]
