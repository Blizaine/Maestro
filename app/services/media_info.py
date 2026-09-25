"""Small, measured gallery facts and durable finishing provenance.

Probes read headers only, never decode frames or load a generation model.
File stat signatures invalidate the bounded cache after in-place processing.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import json
import math
import os
import re
import time


def _positive(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


@lru_cache(maxsize=256)
def _probe(path, size, mtime_ns):
    facts = {"size_bytes": size}
    try:
        if os.path.splitext(path)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            from PIL import Image
            with Image.open(path) as image:
                facts.update(width=image.width, height=image.height)
        else:
            import av
            with av.open(path) as container:
                if container.streams.video:
                    stream = container.streams.video[0]
                    facts.update(width=stream.width, height=stream.height)
                    rate = _positive(stream.average_rate)
                    if rate:
                        facts["fps"] = round(rate, 6)
                    if stream.frames:
                        facts["frames"] = int(stream.frames)
                    duration = _positive(stream.duration * stream.time_base) if stream.duration is not None and stream.time_base else None
                else:
                    duration = None
                duration = duration or _positive(container.duration / av.time_base if container.duration is not None else None)
                if duration:
                    facts["duration_seconds"] = round(duration, 3)
    except Exception:
        # An incomplete/unsupported file can still show its date and byte size.
        pass
    return facts


def probe_media(path):
    try:
        stat = os.stat(path)
        return dict(_probe(os.path.realpath(path), stat.st_size, stat.st_mtime_ns))
    except (OSError, TypeError, ValueError):
        return {}


def record_upload(path, original_name):
    """Retain the first upload date when identical-file reuse is requested."""
    try:
        with open(os.path.splitext(path)[0] + ".meta.json", "x", encoding="utf-8") as stream:
            json.dump({"uploaded_at": time.time(), "original_filename": original_name}, stream)
    except FileExistsError:
        pass
    except OSError:
        pass  # The file's modification date is still available.


def spatial_details(method):
    method = str(method or "")
    match = re.fullmatch(r"(dlss5\*|lanczos|flashvsr2pass|flashvsr)([0-9.]+)", method)
    if match and (scale := _positive(match[2])):
        label = {"dlss5*": "DLSS 5 Neural Rendering", "lanczos": "Lanczos",
                 "flashvsr": "FlashVSR", "flashvsr2pass": "FlashVSR (two passes)"}[match[1]]
        return {"method": method, "method_label": label, "multiplier": scale}
    if method in {"vae1", "vae2"}:
        return {"method": method, "method_label": "VAE upscaling"}
    return {"method": method, "method_label": method} if method else {}


def processing_record(*, spatial="", temporal="", source_name=None, before=None,
                      after=None, elapsed=None, completed_at=None, options=None):
    record = spatial_details(spatial)
    if temporal:
        record.update(temporal_method=temporal,
                      temporal_label="DLSS Frame Generation" if str(temporal).startswith("dlssg*") else "RIFE" if str(temporal).startswith("rife") else str(temporal))
        match = re.fullmatch(r"(?:rife|dlssg\*)([2-6])", str(temporal))
        if match:
            record["frame_multiplier"] = int(match[1])
    if source_name:
        record["source_name"] = os.path.basename(source_name)
    if before:
        record["input"] = dict(before)
    if after:
        record["output"] = dict(after)
    if elapsed is not None and (seconds := _positive(elapsed)):
        record["elapsed_seconds"] = round(seconds, 3)
    if completed_at is not None and (stamp := _positive(completed_at)):
        record["completed_at"] = stamp
    if str(spatial).startswith("dlss5*"):
        settings = options or {}
        record["options"] = {k: settings[k] for k in ("dlss_intensity", "dlss_depth", "dlss_motion") if k in settings}
    return record


def _legacy_source(path, metadata, roots):
    params = metadata.get("params") or {}
    # Prefer a recorded full source path. Never probe a path outside media roots.
    raw = params.get("media_path") or params.get("video_path")
    name = metadata.get("tool_source")
    candidates = [raw] if isinstance(raw, str) and os.path.isabs(raw) else []
    if not candidates and isinstance(name, str) and name == os.path.basename(name):
        candidates = [os.path.join(root, name) for root in roots]
    permitted = []
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real == os.path.realpath(path) or not os.path.isfile(real):
            continue
        for root in roots:
            try:
                if os.path.normcase(os.path.commonpath([real, os.path.realpath(root)])) == os.path.normcase(os.path.realpath(root)):
                    permitted.append(real)
                    break
            except ValueError:
                pass
    unique = set(permitted)
    return unique.pop() if len(unique) == 1 else None


def enrich_metadata(path, metadata, *, source_roots=()):
    """Add facts without rewriting old sidecars or guessing absent provenance."""
    result = copy.deepcopy(metadata)
    result.setdefault("params", None)
    params = result.get("params") if isinstance(result.get("params"), dict) else {}
    result["media_info"] = probe_media(path)
    processing = result.get("processing") or params.get("processing")
    if not isinstance(processing, dict):
        processing = None
    # Old Tools sidecars were only published after success. Generation settings
    # alone describe requested finishing, which may have failed; don't call that
    # completed processing without an explicit record.
    if not processing and result.get("tool") in {"upscale", "media_flow"}:
        source = _legacy_source(path, result, source_roots)
        processing = processing_record(
            spatial=params.get("spatial_upsampling") or params.get("method", ""),
            temporal=params.get("temporal_upsampling", ""), source_name=result.get("tool_source"),
            before=probe_media(source) if source else None, after=result["media_info"],
            elapsed=result.get("generation_time"), completed_at=result.get("created_at"), options=params)
    if processing:
        if not processing.get("output"):
            processing["output"] = result["media_info"]
        result["processing"] = processing
    stamp = _positive(result.get("uploaded_at"))
    kind = "uploaded"
    if not stamp:
        stamp = (_positive(processing.get("completed_at")) if processing else None) or _positive(result.get("created_at")) or _positive(params.get("creation_timestamp"))
        kind = "processed" if result.get("tool") or processing else "generated"
    if not stamp:
        try:
            stamp = os.stat(path).st_mtime
        except OSError:
            stamp = None
        kind = "file_modified"
    if stamp:
        result["timestamp"] = {"value": stamp, "kind": kind}
    return result
