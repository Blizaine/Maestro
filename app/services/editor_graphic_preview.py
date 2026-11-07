"""CPU-only previews of editable graphics using the export renderer.

All image paths pass through the existing Editor media boundary. Cache keys
contain only normalized design data and resolved source fingerprints; no files
are created and a preview never changes a saved project.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections import OrderedDict
from typing import Any

_CACHE: OrderedDict[str, bytes] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_BYTES = 32 * 1024 * 1024


def build_editor_graphic_preview(
    body: dict[str, Any], *, save_root: str, uploads_root: str, workspace: str,
) -> bytes:
    from services.editor_graphics import normalize_editor_graphic, render_editor_graphic
    from services.editor_projects import resolve_editor_asset

    if not isinstance(body, dict) or set(body) - {"graphic", "width", "height", "assets", "workspace"}:
        raise ValueError("Use graphic, width, height, referenced image assets and workspace.")
    dimensions = []
    for key in ("width", "height"):
        value = body.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or int(value) != value or not 64 <= value <= 8192:
            raise ValueError("Graphic preview dimensions must be integers from 64 to 8192.")
        dimensions.append(int(value))
    assets = body.get("assets", {})
    if not isinstance(assets, dict) or len(assets) > 40:
        raise ValueError("Graphic previews accept at most 40 referenced image assets.")
    graphic = normalize_editor_graphic(body.get("graphic"), assets=assets)
    paths: dict[str, str] = {}
    fingerprints = []
    for element in graphic["elements"]:
        if element["type"] != "image" or element["asset_id"] in paths:
            continue
        identifier = element["asset_id"]
        asset = assets[identifier]
        path = resolve_editor_asset(asset, save_root=save_root, workspace=workspace, uploads_root=uploads_root)
        stat = os.stat(path)
        if stat.st_size > 64 * 1024 * 1024:
            raise ValueError("A graphic image source exceeds the 64 MiB preview limit.")
        paths[identifier] = path
        fingerprints.append((identifier, os.path.realpath(path), stat.st_size, stat.st_mtime_ns))
    ratio = min(1.0, 1280 / max(dimensions))
    width, height = (max(1, round(v * ratio)) for v in dimensions)
    key = hashlib.sha256(json.dumps([graphic, width, height, fingerprints], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
    content = render_editor_graphic(graphic, width=width, height=height, resolve_image=lambda identifier: paths[identifier])
    with _CACHE_LOCK:
        _CACHE[key] = content
        _CACHE.move_to_end(key)
        while len(_CACHE) > 24 or sum(map(len, _CACHE.values())) > _CACHE_BYTES:
            _CACHE.popitem(last=False)
    return content
