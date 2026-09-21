"""Local, versioned YuE2 artist bundles; never executable model extensions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid
from .music_contracts import TOKENIZER_REVISION, tokenizer_pair, adapter_contract

STYLE_ROOT = Path("settings/music_styles")
BASE_REVISION = "864a479cbf3e810e1b2c1993b438510750e383b2"
SCHEMA_VERSION = 2
_library_lock = threading.RLock()


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
    if (not isinstance(manifest, dict) or manifest.get("version") not in (1, SCHEMA_VERSION) or manifest.get("architecture") != "yue2"
            or manifest.get("base_revision") != BASE_REVISION):
        raise ValueError("This music style uses an unsupported YuE2 model or tokenizer revision")
    mode = adapter_contract(manifest)
    expected_revision = tokenizer_pair(manifest)['revision']
    if manifest.get('tokenizer_revision') != expected_revision and not (mode == 'joint' and manifest.get('tokenizer_revision') is None):
        raise ValueError("This music style uses an unsupported tokenizer revision")
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


def list_styles(*, root=None, include_archived=False):
    result = []
    for path in sorted(Path(root or STYLE_ROOT).glob("*/style.json")):
        try:
            manifest = load_style(path.parent.name, root=root)
            if manifest.get('archived') and not include_archived:
                continue
            result.append({key: manifest.get(key) for key in ("id", "name", "trigger", "license", "training", "tokenizer_pair", "tokenizer_revision", "adapted_pair")} | {
                'archived': manifest.get('archived') is True,
                'in_selector': manifest.get('in_selector') is True,
                'created_at': manifest.get('created_at', path.stat().st_mtime),
            })
        except ValueError:
            continue
    return sorted(result, key=lambda item: (-float(item['created_at']), item['name'].casefold()))


def update_style_library(style_id, changes, *, root=None):
    """Update library visibility without changing adapter weights or job lookup."""
    if (not isinstance(changes, dict) or not changes
            or not set(changes) <= {'archived', 'in_selector'}
            or any(not isinstance(value, bool) for value in changes.values())):
        raise ValueError('Supply archived and/or in_selector as true or false')
    with _library_lock:
        manifest = load_style(style_id, root=root)
        destination = style_directory(style_id, root) / 'style.json'
        manifest.setdefault('created_at', destination.stat().st_mtime)
        manifest.update(changes)
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        temporary.replace(destination)
        return manifest


def set_style_archived(style_id, archived, *, root=None):
    """Hide a library entry while preserving weights for queued jobs and restore."""
    return update_style_library(style_id, {'archived': archived}, root=root)


def validate_strength(value):
    if isinstance(value, bool):
        raise ValueError("Music style strength must be a number")
    try:
        strength = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Music style strength must be a number") from error
    if not math.isfinite(strength) or not 0 <= strength <= 1.5:
        raise ValueError("Music style strength must be between 0 and 1.5")
    return strength


def selected_style_entries(settings):
    """Normalize the queue contract; an explicit empty list clears legacy IDs."""
    settings = settings or {}
    if not isinstance(settings, dict):
        raise ValueError("Music LoRA settings must be an object")
    if "artist_loras" in settings:
        entries = settings["artist_loras"]
        if not isinstance(entries, list):
            raise ValueError("Music LoRAs must be a list of IDs and strengths")
    else:
        entries = [{"id": settings["artist_id"], "strength": settings.get("artist_strength", 1.0)}] if settings.get("artist_id") else []
    result, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or not set(entry) <= {"id", "strength"}:
            raise ValueError("Each music LoRA needs an ID and a strength")
        style_id = entry.get("id")
        if not isinstance(style_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", style_id):
            raise ValueError("Invalid music style ID")
        if style_id in seen:
            raise ValueError("Select each music LoRA only once")
        seen.add(style_id)
        result.append({"id": style_id, "strength": validate_strength(entry.get("strength", 1.0))})
    return result


def generation_styles(settings, *, root=None, verify=False):
    """Resolve the whole selection before installing any inference hooks."""
    result = [{**entry, "manifest": load_style(entry["id"], root=root, verify=verify)}
              for entry in selected_style_entries(settings)]
    known_pairs = {item["manifest"].get("tokenizer_pair", "v4") for item in result
                   if item["manifest"].get("tokenizer_revision") is not None}
    if len(known_pairs) > 1:
        raise ValueError("Choose music LoRAs trained with the same tokenizer version: v4 and v9 cannot be combined")
    if len(result) > 1 and (settings or {}).get("base_acoustic") is True:
        raise ValueError("The base-acoustic comparison requires one music LoRA")
    return result


def save_style(name, trigger, ar_tensors, nar_tensors, *, root=None, training=None,
               adapter_mode="separate", pair="v4", unknown_tokenizer=False, adapted_pair=None):
    """Called only after the tensor shapes have been checked by the YuE2 adapter."""
    from safetensors.torch import save_file
    name = str(name or "").strip()
    trigger = str(trigger or "").strip()
    if not name or len(name) > 100 or len(trigger) > 200:
        raise ValueError("Give this style a name (up to 100 characters) and a short trigger")
    adapter_contract({'version': SCHEMA_VERSION, 'adapter_mode': adapter_mode})
    revision = tokenizer_pair({'tokenizer_pair': pair, 'adapted_pair': adapted_pair})['revision']
    if unknown_tokenizer and adapter_mode != 'joint':
        raise ValueError('A separate AR adapter needs its matched tokenizer/decoder pair')
    style_id = uuid.uuid4().hex[:16]
    directory = style_directory(style_id, root)
    directory.mkdir(parents=True)
    manifest = {"version": SCHEMA_VERSION, "id": style_id, "name": name, "trigger": trigger,
                "architecture": "yue2", "base_revision": BASE_REVISION,
                "tokenizer_revision": None if unknown_tokenizer else revision,
                "tokenizer_pair": pair, "adapter_mode": adapter_mode, "license": "CC BY-NC 4.0",
                "training": training, "created_at": time.time()}
    if adapted_pair:
        manifest['adapted_pair'] = dict(adapted_pair)
    for branch, tensors in (("ar", ar_tensors), ("nar", nar_tensors)):
        destination = directory / f"{branch}.safetensors"
        save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(destination))
        manifest[branch] = {"file": destination.name, "sha256": file_digest(destination)}
    # The manifest is the commit marker: incomplete imports never appear in the picker.
    (directory / "style.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
