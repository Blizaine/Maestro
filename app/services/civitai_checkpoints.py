"""CivitAI H3 import transport, kept independent of the generation runtime."""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct

from shared.checkpoint_downloads import _authenticated_request


class CivitaiCheckpointError(ValueError):
    """An actionable checkpoint preflight/download failure."""


_MAX_HEADER = 8 * 1024 * 1024
_MAX_DESCRIPTOR = 16 * 1024
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _is_curve_key(key):
    prefixes = ("model.diffusion_model.", "diffusion_model.", "module.")
    while any(key.startswith(prefix) for prefix in prefixes):
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
    return key == "adaln_t_table"


def is_h3_base(base_model: object) -> bool:
    return " ".join(str(base_model or "").casefold().split()) == "minimax h3"


def _positive_id(value, label):
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        result = 0
    if isinstance(value, bool) or result <= 0:
        raise CivitaiCheckpointError(f"Select a CivitAI {label} before importing this checkpoint.")
    return result


def resolve_h3_source(body: dict, *, api_key="", get=None) -> dict:
    """Resolve the selected file from CivitAI, never trust client file URLs/hashes."""
    if get is None:
        import requests
        get = requests.get
    model_id = _positive_id(body.get("model_id"), "model")
    version_id = _positive_id(body.get("version_id"), "version")
    file_id = _positive_id(body.get("file_id"), "file")
    headers = {"User-Agent": "Maestro/CheckpointImport", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = None
    try:
        response = get(f"https://civitai.com/api/v1/models/{model_id}", headers=headers, timeout=20)
        if response.status_code in (401, 403):
            raise CivitaiCheckpointError("CivitAI denied access. Check Settings > Services > CivitAI API Key and your account's access to this model.")
        if response.status_code != 200:
            raise CivitaiCheckpointError(f"Could not verify this CivitAI model (HTTP {response.status_code}). Try again.")
        model = response.json()
    except CivitaiCheckpointError:
        raise
    except Exception:
        raise CivitaiCheckpointError("Could not reach CivitAI to verify the selected file. Try again.") from None
    finally:
        if response is not None:
            response.close()
    if not isinstance(model, dict) or model.get("id") != model_id or model.get("type") != "Checkpoint":
        raise CivitaiCheckpointError("The selected CivitAI entry is not a checkpoint.")
    version = next((item for item in model.get("modelVersions", [])
                    if isinstance(item, dict) and item.get("id") == version_id), None)
    if not version or not is_h3_base(version.get("baseModel")):
        raise CivitaiCheckpointError("The selected version is not a MiniMax H3 checkpoint.")
    file = next((item for item in version.get("files", [])
                 if isinstance(item, dict) and item.get("id") == file_id), None)
    if not file:
        raise CivitaiCheckpointError("This file is no longer listed in the selected CivitAI version. Refresh the model browser.")
    filename = str(file.get("name") or "")
    from shared.checkpoint_downloads import _safe_filename
    try:
        _safe_filename(filename)
    except ValueError:
        raise CivitaiCheckpointError("CivitAI supplied an invalid checkpoint filename.") from None
    if not filename.casefold().endswith((".safetensors", ".sft", ".gguf")):
        raise CivitaiCheckpointError("H3 import supports individual SafeTensor or GGUF diffusion checkpoints. Workflows and archives are not checkpoints.")
    fp = str((file.get("metadata") or {}).get("fp") or "").casefold()
    if not filename.casefold().endswith(".gguf") and fp in {"int6", "nvfp4", "mxfp4", "mxfp8", "w6a8"}:
        raise CivitaiCheckpointError(f"This H3 file uses unsupported {fp.upper()} storage. Choose W4A8 INT4, INT8 ConvRot, scaled FP8, BF16, FP16 or a supported GGUF file.")
    sha256 = str((file.get("hashes") or {}).get("SHA256") or "").lower()
    if not _SHA256.fullmatch(sha256):
        raise CivitaiCheckpointError("CivitAI has not supplied a SHA-256 checksum for this file. Maestro cannot verify the import yet.")
    try:
        size_bytes = round(float(file["sizeKB"]) * 1024)
    except (ValueError, KeyError, TypeError, OverflowError):
        size_bytes = 0
    if size_bytes <= 8:
        raise CivitaiCheckpointError("CivitAI has not supplied a valid size for this checkpoint.")
    # fileId is essential: a version may have BF16, INT8 and INT4 files.
    return {
        "modelId": model_id, "versionId": version_id, "fileId": file_id,
        "name": str(model.get("name") or "MiniMax H3"),
        "versionName": str(version.get("name") or ""),
        "filename": filename, "sha256": sha256, "size_bytes": size_bytes,
        "url": f"https://civitai.com/api/download/models/{version_id}?fileId={file_id}",
        "description": str(model.get("description") or ""),
        "versionDescription": str(version.get("description") or ""),
        "nsfw": bool(model.get("nsfw")),
    }


def _read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        try:
            chunk = stream.read(remaining)
        except Exception:
            raise CivitaiCheckpointError("The checkpoint read was interrupted. Try again.") from None
        if not chunk:
            raise CivitaiCheckpointError("CivitAI returned a truncated checkpoint header. Try again.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class RemoteH3Header:
    """Read only the header and small quantization markers, with bounded ranges."""
    def __init__(self, source, *, api_key="", get=None):
        if get is None:
            import requests
            get = requests.get
        self.source, self.get = source, get
        self.url, self.headers = _authenticated_request(source["url"], "civitai.com", api_key)
        self.markers = {}
        self._markers_prepared = False
        response = self._open_range(0, _MAX_HEADER + 7)
        try:
            if source["filename"].casefold().endswith(".gguf"):
                from services.h3_gguf_import import read_gguf_index
                index = read_gguf_index(response.raw, source["size_bytes"])
                self.header, self.payload_start = index.header, index.payload_start
                self.file_format = "gguf"
                # Many GGUF exporters put the tiny curve first. Cache it from
                # the same bounded stream, including servers that ignore Range
                # on redirected downloads. Large transformer payloads stay
                # unread; distant curves still require verified byte ranges.
                for name, descriptor in self.header.items():
                    if not _is_curve_key(name) or not isinstance(descriptor, dict):
                        continue
                    start, end = descriptor["data_offsets"]
                    if 0 <= start < end <= 128 * 1024 and end - start <= 64 * 1024:
                        cursor = response.raw.tell()
                        absolute_start = self.payload_start + start
                        if 0 <= absolute_start - cursor <= 128 * 1024:
                            _read_exact(response.raw, absolute_start - cursor)
                            self.markers[name] = _read_exact(response.raw, end - start)
                return
            prefix = _read_exact(response.raw, 8)
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= _MAX_HEADER or header_size + 8 >= source["size_bytes"]:
                raise CivitaiCheckpointError("CivitAI did not return a valid H3 SafeTensor header. Check your API key and file access.")
            self.header = json.loads(_read_exact(response.raw, header_size))
            self.payload_start = 8 + header_size
            self.file_format = "safetensors"
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise CivitaiCheckpointError("CivitAI returned an invalid SafeTensor header.") from None
        finally:
            response.close()
        if not isinstance(self.header, dict):
            raise CivitaiCheckpointError("CivitAI returned an invalid SafeTensor tensor index.")

    def _open_range(self, start, end):
        headers = {**self.headers, "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        try:
            response = self.get(self.url, headers=headers, stream=True, timeout=30, allow_redirects=True)
        except Exception:
            raise CivitaiCheckpointError("Could not inspect the checkpoint because of a network error. Try again.") from None
        error = None
        if response.status_code in (401, 403):
            error = "CivitAI denied access to this file. Check Settings > Services > CivitAI API Key and your account's model access."
        elif response.status_code not in (200, 206):
            error = f"CivitAI could not serve this checkpoint header (HTTP {response.status_code}). Try again."
        elif response.status_code == 200 and start != 0:
            error = "This download server does not support the bounded reads needed to verify H3 quantization. Try another compatible file."
        elif response.status_code == 206:
            content_range = str(response.headers.get("Content-Range") or "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match[1]) != start or int(match[3]) != self.source["size_bytes"]:
                error = "CivitAI returned an inconsistent checkpoint byte range. Refresh and try again."
        if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
            error = "CivitAI returned a compressed byte range that cannot be verified safely. Try again."
        if error:
            response.close()
            raise CivitaiCheckpointError(error)
        return response

    def _prepare_markers(self):
        # FP8 exports interleave 200 tiny markers with their large weights.
        # Fetch disjoint bounded spans concurrently instead of imposing 200
        # sequential HTTP round-trips on every selection in the model browser.
        from concurrent.futures import ThreadPoolExecutor
        entries = []
        for name, descriptor in self.header.items():
            if not name.endswith(".comfy_quant"):
                continue
            offsets = descriptor.get("data_offsets") if isinstance(descriptor, dict) else None
            if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(x) is int for x in offsets):
                raise CivitaiCheckpointError("Invalid H3 quantization descriptor offsets.")
            start, end = offsets
            if not 0 <= start < end or end - start > _MAX_DESCRIPTOR or end + self.payload_start > self.source["size_bytes"]:
                raise CivitaiCheckpointError("H3 quantization descriptor exceeds the bounded preflight size.")
            entries.append((start, end, name))
        if len(entries) > 200:
            raise CivitaiCheckpointError("This H3 export has additional quantized components beyond the supported 200 transformer linears.")
        spans = []
        for start, end, name in sorted(entries):
            if spans and end - spans[-1][0] <= 128 * 1024:
                spans[-1][1] = max(spans[-1][1], end)
                spans[-1][2].append((start, end, name))
            else:
                spans.append([start, end, [(start, end, name)]])
        def read_span(span):
            start, end, names = span
            response = self._open_range(self.payload_start + start, self.payload_start + end - 1)
            try:
                raw = _read_exact(response.raw, end - start)
            finally:
                response.close()
            return {name: raw[left - start:right - start] for left, right, name in names}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for markers in pool.map(read_span, spans):
                self.markers.update(markers)
        self._markers_prepared = True

    def tensor_reader(self, key):
        if key.endswith(".comfy_quant") and not self._markers_prepared:
            self._prepare_markers()
        if key in self.markers:
            return self.markers[key]
        descriptor = self.header.get(key)
        is_curve = _is_curve_key(key)
        if not (key.endswith(".comfy_quant") or is_curve) or not isinstance(descriptor, dict):
            raise CivitaiCheckpointError("Only small H3 quantization descriptors may be read during preflight.")
        offsets = descriptor.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(x) is int for x in offsets):
            raise CivitaiCheckpointError("Invalid H3 quantization descriptor offsets.")
        start, end = offsets
        limit = 64 * 1024 if is_curve else _MAX_DESCRIPTOR
        if not 0 <= start < end or end - start > limit or end + self.payload_start > self.source["size_bytes"]:
            raise CivitaiCheckpointError("H3 quantization descriptor exceeds the bounded preflight size.")
        # Nearby marker tensors share a single request, so a 50-block model
        # does not need a separate HTTP round-trip for every tiny JSON marker.
        nearby = {}
        range_end = end
        if is_curve:
            nearby[key] = offsets
        for name, value in self.header.items():
            if name.endswith(".comfy_quant") and isinstance(value, dict):
                bounds = value.get("data_offsets", [])
                if len(bounds) == 2 and all(type(x) is int for x in bounds) and start <= bounds[0] < bounds[1] <= start + 128 * 1024 and bounds[1] - bounds[0] <= _MAX_DESCRIPTOR and bounds[1] + self.payload_start <= self.source["size_bytes"]:
                    nearby[name] = bounds
                    range_end = max(range_end, bounds[1])
        response = self._open_range(self.payload_start + start, self.payload_start + range_end - 1)
        try:
            raw = _read_exact(response.raw, range_end - start)
        finally:
            response.close()
        for name, (left, right) in nearby.items():
            self.markers[name] = raw[left - start:right - start]
        return self.markers[key]


def inspect_remote_h3(body, *, api_key="", get=None):
    from services.h3_checkpoint_import import inspect_h3_header
    source = resolve_h3_source(body, api_key=api_key, get=get)
    reader = RemoteH3Header(source, api_key=api_key, get=get)
    profile = inspect_h3_header(reader.header, tensor_reader=reader.tensor_reader, source=source,
                                sampling_profile=body.get("h3_sampling_profile", "auto"),
                                native_workflow=body.get("h3_native_workflow", "auto"),
                                qkv_layout=body.get("h3_qkv_layout", "auto"), file_format=reader.file_format)
    return source, profile


def inspect_local_h3(path, source, *, sampling_profile="auto", native_workflow="auto", qkv_layout="auto"):
    from services.checkpoint_compatibility import read_safetensors_header
    from services.h3_checkpoint_import import inspect_h3_header
    with open(path, "rb") as handle:
        is_gguf = handle.read(4) == b"GGUF"
        handle.seek(0)
        expected_name = str(source.get("filename") or path).casefold()
        if expected_name.endswith(".gguf") != is_gguf and expected_name.endswith((".gguf", ".safetensors", ".sft")):
            raise CivitaiCheckpointError("The checkpoint container does not match its filename. Re-import the creator's original file.")
        if is_gguf:
            from services.h3_gguf_import import read_gguf_index
            index = read_gguf_index(handle, os.path.getsize(path))
            header, payload_start = index.header, index.payload_start
        else:
            header = read_safetensors_header(str(path))
            payload_start = 8 + struct.unpack("<Q", handle.read(8))[0]
        def read_marker(key):
            descriptor = header[key]
            start, end = descriptor["data_offsets"]
            is_curve = _is_curve_key(key)
            if not (key.endswith(".comfy_quant") or is_curve) or end - start > (64 * 1024 if is_curve else _MAX_DESCRIPTOR):
                raise CivitaiCheckpointError("Invalid H3 quantization descriptor size.")
            handle.seek(payload_start + start)
            return _read_exact(handle, end - start)
        return inspect_h3_header(header, tensor_reader=read_marker, source=source,
                                 sampling_profile=sampling_profile, native_workflow=native_workflow,
                                 qkv_layout=qkv_layout, file_format="gguf" if is_gguf else "safetensors")


def verify_h3_digest(path, source, *, progress=None):
    """Check an existing file before reusing it; filenames alone are insufficient."""
    if os.path.getsize(path) != source["size_bytes"]:
        return False
    digest = hashlib.sha256()
    count = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            count += len(chunk)
            digest.update(chunk)
            if progress:
                progress(count, source["size_bytes"])
    return digest.hexdigest() == source["sha256"]


def build_h3_definitions(source, profile, filename, defaults_dir, *, auto_quantize=False):
    """Create native H3 workflows sharing one verified local transformer."""
    if profile.get("status") != "verified":
        raise CivitaiCheckpointError("The H3 checkpoint has not passed compatibility verification.")
    identity = f"civitai_h3_{source['modelId']}_{source['versionId']}_{source['fileId']}"
    entries = {}
    architectures = list(profile["architectures"])
    pairs = {"references" if "ref2va" in architecture else "frames":
             identity + ("_references" if "ref2va" in architecture else "_frames")
             for architecture in architectures}
    turbo = bool(profile.get("baked_turbo") or profile.get("fused_turbo"))
    quant_label = {
        "asym_w4a8_int8": "W4A8 INT4", "int8_tensorwise": "INT8 ConvRot",
        "scaled_fp8": "FP8", "none": "BF16 / FP16",
        "gguf": "GGUF " + " / ".join(profile.get("gguf_quant_types", [])),
    }.get(profile.get("quantization_format"), str(profile.get("quantization_format", "")))
    # The serialized profile contains public provenance only; auth stays in
    # the service settings and is never written into model definitions.
    receipt_profile = {**profile, "source": {key: source[key] for key in
        ("modelId", "versionId", "fileId", "name", "versionName", "filename", "sha256", "size_bytes",
         "description", "versionDescription") if key in source}}
    for architecture in architectures:
        workflow = "references" if "ref2va" in architecture else "frames"
        slug = pairs[workflow]
        with open(os.path.join(defaults_dir, architecture + ".json"), encoding="utf-8") as handle:
            definition = json.load(handle)
        model = definition["model"]
        label = source["name"]
        if source.get("versionName"):
            label += f" · {source['versionName']}"
        model.update({
            "name": f"{label} · {quant_label} — {workflow.title()} (CivitAI)",
            "architecture": architecture, "URLs": [filename],
            "visible": True, "nsfw_only": bool(source.get("nsfw")),
            "minimax_h3_model_id": slug, "minimax_h3_import_profile": receipt_profile,
            "minimax_h3_lora_workflow": profile["native_workflow"],
            "minimax_h3_qkv_layout": profile["qkv_layout"],
            "minimax_h3_full_checkpoint": not profile["compressed_modulation"],
            "minimax_h3_baked_turbo": turbo,
            "sol_attention": True,
            "minimax_h3_sampler": profile.get("sampler", "euler"),
            "minimax_h3_video_shift": profile.get("video_shift", 12.0),
            "minimax_h3_audio_shift": profile.get("audio_shift", 3.0),
            "minimax_h3_unaccelerated_default_steps": profile.get("default_steps", 8 if turbo else 20),
            "minimax_h3_turbo_mode_default": False,
            "compatible_model_paths": {}, "compatible_model_qkv_layouts": {},
            "h3_companion_models": pairs,
            "description": f"Verified CivitAI MiniMax H3 checkpoint. {workflow.title()} workflow, "
                           f"{profile['sampling_profile']} sampling; synchronized video and audio.",
            "selector_help": f"Imported from the creator on CivitAI. {profile['sampling_profile'].title()} "
                             f"recipe, {profile.get('default_steps', 8 if turbo else 20)} steps. "
                             + ("Frames and References share this checkpoint. " if len(pairs) > 1 else "")
                             + ("Acceleration is baked in; do not add a Turbo or PDD adapter." if turbo else ""),
            "civitai": {**receipt_profile["source"], "filename": filename,
                        "modelType": "Checkpoint", "baseModel": "MiniMax H3",
                        "compatibility": {"status": "verified", "architecture": architecture,
                            "base_model": "MiniMax H3", "signature_version": 2,
                            "matched_layouts": architectures, "h3_profile": receipt_profile}},
        })
        model.pop("source", None)
        model.pop("source2", None)
        # Only plain floating-point imports should be quantized again.
        model["auto_quantize"] = bool(auto_quantize and profile.get("quantization_format") == "none")
        definition.update({
            "num_inference_steps": profile.get("default_steps", 8 if turbo else 20),
            "guidance_scale": 1.0, "flow_shift": model["minimax_h3_video_shift"],
            "audio_flow_shift": model["minimax_h3_audio_shift"],
            "minimax_h3_turbo_mode": False, "minimax_h3_turbo_preset": "",
        })
        if turbo:
            definition.update({"override_attention": "", "skip_steps_cache_type": ""})
        entries[slug] = definition
    return entries
