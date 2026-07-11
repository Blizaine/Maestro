"""Remote 3D model provider adapters.

The first integration layer treats 3D generators as external services. This
keeps Maestro lightweight and lets users connect Pinokio launchers or other
servers that already host Hunyuan3D, InstantMesh, TripoSR, Trellis, etc.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import requests


MODEL3D_EXTENSIONS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".usdz", ".zip"}


@dataclass(frozen=True)
class Model3DProvider:
    id: str
    label: str
    default_endpoint: str
    input_modes: tuple[str, ...]
    notes: str


PROVIDERS: dict[str, Model3DProvider] = {
    "hunyuan3d": Model3DProvider(
        id="hunyuan3d",
        label="Hunyuan3D",
        default_endpoint="/generate",
        input_modes=("text", "image"),
        notes="Best for high quality text/image to GLB services.",
    ),
    "instantmesh": Model3DProvider(
        id="instantmesh",
        label="InstantMesh",
        default_endpoint="/generate",
        input_modes=("image",),
        notes="Image to mesh provider. Requires a compatible remote API.",
    ),
    "triposr": Model3DProvider(
        id="triposr",
        label="TripoSR",
        default_endpoint="/generate",
        input_modes=("image",),
        notes="Fast image to mesh provider. Requires a compatible remote API.",
    ),
    "trellis": Model3DProvider(
        id="trellis",
        label="Trellis",
        default_endpoint="/generate",
        input_modes=("text", "image"),
        notes="Text/image to 3D provider. Requires a compatible remote API.",
    ),
}


def provider_list() -> list[dict[str, Any]]:
    return [
        {
            "id": p.id,
            "label": p.label,
            "default_endpoint": p.default_endpoint,
            "input_modes": list(p.input_modes),
            "notes": p.notes,
        }
        for p in PROVIDERS.values()
    ]


def _normalize_url(base_url: str, endpoint: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    endpoint = (endpoint or "").strip() or "/generate"
    if not base_url:
        raise ValueError("3D provider server URL is required")
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return urljoin(base_url + "/", endpoint.lstrip("/"))


def _image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_payload(provider: str, body: dict[str, Any], image_path: str | None) -> dict[str, Any]:
    prompt = (body.get("prompt") or body.get("text") or "").strip()
    output_format = (body.get("output_format") or "glb").lower().lstrip(".")
    payload: dict[str, Any] = {
        "type": output_format,
        "format": output_format,
    }
    if prompt:
        payload["text"] = prompt
        payload["prompt"] = prompt
    if image_path:
        image_data = _image_to_data_url(image_path)
        payload["image"] = image_data
        payload["image_url"] = image_data
    for key in (
        "texture",
        "seed",
        "num_inference_steps",
        "guidance_scale",
        "octree_resolution",
        "num_chunks",
        "remove_background",
        "foreground_ratio",
    ):
        if key in body and body[key] is not None:
            payload[key] = body[key]

    # Hunyuan3D launchers commonly expect `text`/`image` plus llama.cpp-like
    # generation knobs. The generic providers receive the same minimal shape;
    # provider-specific refinements can be added here without changing callers.
    if provider == "hunyuan3d":
        payload.pop("prompt", None)
    return payload


def _guess_extension(content_type: str, output_format: str) -> str:
    output_format = (output_format or "glb").lower().lstrip(".")
    if output_format in {"glb", "gltf", "obj", "ply", "stl", "usdz", "zip"}:
        return f".{output_format}"
    if "model/gltf-binary" in content_type or "octet-stream" in content_type:
        return ".glb"
    if "zip" in content_type:
        return ".zip"
    return ".glb"


def _write_bytes(output_dir: str, provider: str, content: bytes, ext: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    filename = f"{stamp}_{provider}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _extract_json_asset(data: dict[str, Any], base_url: str) -> tuple[bytes, str] | None:
    for key in ("file", "path"):
        value = data.get(key)
        if isinstance(value, str) and os.path.isfile(value):
            with open(value, "rb") as f:
                return f.read(), os.path.splitext(value)[1] or ".glb"
    for key in ("url", "file_url", "output_url", "download_url"):
        value = data.get(key)
        if isinstance(value, str) and value:
            url = value if value.startswith(("http://", "https://")) else urljoin(base_url + "/", value.lstrip("/"))
            r = requests.get(url, timeout=(10, 600))
            r.raise_for_status()
            ext = os.path.splitext(url.split("?", 1)[0])[1] or _guess_extension(r.headers.get("content-type", ""), "glb")
            return r.content, ext
    for key in ("glb", "gltf", "obj", "ply", "stl", "usdz", "model", "output"):
        value = data.get(key)
        if isinstance(value, str) and value:
            if value.startswith("data:"):
                _, encoded = value.split(",", 1)
                return base64.b64decode(encoded), f".{key if key != 'model' else 'glb'}"
            try:
                return base64.b64decode(value), f".{key if key != 'model' else 'glb'}"
            except Exception:
                pass
    return None


def generate_model3d(
    *,
    provider: str,
    remote_url: str,
    endpoint: str,
    output_dir: str,
    body: dict[str, Any],
    image_path: str | None = None,
) -> dict[str, Any]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported 3D provider: {provider}")
    url = _normalize_url(remote_url, endpoint or PROVIDERS[provider].default_endpoint)
    payload = _build_payload(provider, body, image_path)
    timeout = int(body.get("timeout_seconds") or 1800)
    response = requests.post(url, json=payload, timeout=(15, timeout))
    response.raise_for_status()

    output_format = (body.get("output_format") or "glb").lower().lstrip(".")
    content_type = response.headers.get("content-type", "")
    asset: tuple[bytes, str] | None = None
    if "application/json" in content_type:
        asset = _extract_json_asset(response.json(), remote_url.rstrip("/"))
    else:
        asset = (response.content, _guess_extension(content_type, output_format))
    if asset is None:
        raise RuntimeError("3D provider response did not contain a downloadable model asset")

    content, ext = asset
    if not content:
        raise RuntimeError("3D provider returned an empty model asset")
    if ext.lower() not in MODEL3D_EXTENSIONS:
        ext = f".{output_format}" if f".{output_format}" in MODEL3D_EXTENSIONS else ".glb"
    path = _write_bytes(output_dir, provider, content, ext)
    filename = os.path.basename(path)
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generation_mode": "model3d",
                "provider": provider,
                "remote_url": remote_url,
                "endpoint": endpoint or PROVIDERS[provider].default_endpoint,
                "params": {
                    "prompt": body.get("prompt") or body.get("text") or "",
                    "image_path": image_path,
                    "output_format": output_format,
                    "provider": provider,
                },
            },
            f,
            indent=2,
        )
    return {
        "filename": filename,
        "path": path,
        "url": f"/api/v1/file/{filename}",
        "provider": provider,
        "size": os.path.getsize(path),
    }
