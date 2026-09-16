"""Local, versioned YuE2 artist bundles; never executable model extensions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import uuid

STYLE_ROOT = Path("settings/music_styles")
TOKENIZER_REVISION = "f2278a2e005dc4ecc421c53a0929f62b3aeb2280"
BASE_REVISION = "864a479cbf3e810e1b2c1993b438510750e383b2"
SCHEMA_VERSION = 1


def style_directory(style_id: str, root=None) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", str(style_id)):
        raise ValueError("Invalid music style ID")
    base = Path(root or STYLE_ROOT).resolve()
    target = (base / style_id).resolve()
    if target.parent != base:
        raise ValueError("Music style must stay inside its library")
    return target


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_style(style_id, *, root=None, verify=False):
    directory = style_directory(style_id, root)
    try:
        manifest = json.loads((directory / "style.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Music style is missing or its manifest is unreadable") from error
    if (not isinstance(manifest, dict) or manifest.get("version") != SCHEMA_VERSION or manifest.get("architecture") != "yue2"
            or manifest.get("base_revision") != BASE_REVISION
            or manifest.get("tokenizer_revision") != TOKENIZER_REVISION):
        raise ValueError("This music style uses an unsupported YuE2 model or tokenizer revision")
    if (not isinstance(manifest.get("name"), str) or not 1 <= len(manifest["name"]) <= 100
            or not isinstance(manifest.get("trigger"), str) or len(manifest["trigger"]) > 200):
        raise ValueError("The music style manifest needs a valid name and trigger")
    for branch in ("ar", "nar"):
        asset = manifest.get(branch) or {}
        filename = asset.get("file", "") if isinstance(asset, dict) else ""
        if filename != f"{branch}.safetensors":
            raise ValueError("Music style weights must use the supported AR/NAR bundle format")
        path = directory / filename
        if not path.is_file() or path.resolve().parent != directory:
            raise ValueError(f"The {branch.upper()} weights are missing from this music style")
        if verify and file_digest(path) != asset.get("sha256"):
            raise ValueError(f"The {branch.upper()} weights have changed since this music style was imported")
    return manifest


def list_styles(*, root=None):
    result = []
    for path in sorted(Path(root or STYLE_ROOT).glob("*/style.json")):
        try:
            manifest = load_style(path.parent.name, root=root)
            result.append({key: manifest.get(key) for key in ("id", "name", "trigger", "license", "training")})
        except ValueError:
            continue
    return result


def validate_strength(value):
    try:
        strength = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Music style strength must be a number") from error
    if not math.isfinite(strength) or not 0 <= strength <= 1.5:
        raise ValueError("Music style strength must be between 0 and 1.5")
    return strength


def save_style(name, trigger, ar_tensors, nar_tensors, *, root=None, training=None):
    """Called only after the tensor shapes have been checked by the YuE2 adapter."""
    from safetensors.torch import save_file
    name = str(name or "").strip()
    trigger = str(trigger or "").strip()
    if not name or len(name) > 100 or len(trigger) > 200:
        raise ValueError("Give this style a name (up to 100 characters) and a short trigger")
    style_id = uuid.uuid4().hex[:16]
    directory = style_directory(style_id, root)
    directory.mkdir(parents=True)
    manifest = {"version": SCHEMA_VERSION, "id": style_id, "name": name, "trigger": trigger,
                "architecture": "yue2", "base_revision": BASE_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION, "license": "CC BY-NC 4.0",
                "training": training}
    for branch, tensors in (("ar", ar_tensors), ("nar", nar_tensors)):
        destination = directory / f"{branch}.safetensors"
        save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(destination))
        manifest[branch] = {"file": destination.name, "sha256": file_digest(destination)}
    # The manifest is the commit marker: incomplete imports never appear in the picker.
    (directory / "style.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
