"""
Recipes — one-click generation presets (Studio).

A recipe bundles a model, its LoRAs (by pointer, never the weights), the
tuned generation settings, and an editable example prompt. Applying one
switches the model, fills the settings into the active sub-mode's working
set, and prepopulates the prompt so the user just edits the subject.

Two sources, both scanned at list time:
  - app/recipes/       — bundled starter pack, tracked in the repo.
  - app/recipes_user/  — user-saved + imported recipes, gitignored.

Each recipe is a JSON file; its thumbnail (same basename, .jpg) lives
beside it. LoRAs are referenced by {filename, multiplier, source_url,
size_mb} so a recipe carries no weights and no licensing exposure — the
weights are fetched on demand through the existing CivitAI/managed-LoRA
path when a recipe that needs them is applied.

Recipe id == the JSON file's basename (without extension).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_HERE)
BUNDLED_DIR = os.path.join(_APP_DIR, "recipes")
USER_DIR = os.path.join(_APP_DIR, "recipes_user")

RECIPE_SCHEMA_VERSION = 1

# Generation-driving settings a recipe captures. Deliberately an allowlist:
# a recipe reproduces a LOOK (model + tuning), not a specific frame — so
# seed, repeat count, and the per-generation inputs (start/end frames,
# refs, control/audio guides, image_mode) are intentionally excluded.
RECIPE_PARAM_KEYS = [
    "resolution", "video_length", "num_inference_steps", "guidance_scale",
    "negative_prompt", "flow_shift", "guidance_phases",
    # alt_prompt = ACE-Step "Music Caption" (style/genre/instruments) — the
    # actual "look" of a music recipe, so it's part of the preset.
    "alt_prompt",
    "sliding_window_size", "sliding_window_overlap",
    "self_refiner_setting", "stage2_steps", "stg_scale", "cfg_rescale",
    "modality_scale", "use_gradient_estimation", "ge_gamma", "ge_alpha",
    "progressive_pipeline", "single_stage_pipeline",
    "progressive_stage1_image_weight", "progressive_stage2_steps",
    "progressive_stage2_sigma", "progressive_stage3_steps",
    "progressive_stage3_sigma", "progressive_stage3_image_weight",
    "keyframe_conditioning_mode", "keyframe_inject_mode",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or f"recipe-{int(time.time())}"


def _ensure_user_dir() -> None:
    os.makedirs(USER_DIR, exist_ok=True)


def _recipe_dirs() -> list[tuple[str, str]]:
    """(source_label, dir) pairs, bundled first so a user recipe with the
    same id shadows the bundled one."""
    return [("bundled", BUNDLED_DIR), ("user", USER_DIR)]


def _thumb_path_for(json_path: str) -> str:
    return os.path.splitext(json_path)[0] + ".jpg"


def _load_recipe_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data.get("name"):
            return None
        return data
    except Exception:
        return None


def _card(recipe: dict, rid: str, source: str, has_thumb: bool) -> dict:
    return {
        "id": rid,
        "name": recipe.get("name", rid),
        "description": recipe.get("description", ""),
        "mode": recipe.get("mode", "video"),
        "model_type": recipe.get("model_type", ""),
        "lora_count": len(recipe.get("loras", []) or []),
        "prompt_example": recipe.get("prompt_example", ""),
        "nsfw": bool(recipe.get("nsfw", False)),
        "source": source,
        "thumbnail_url": f"/api/v1/recipes/{rid}/thumbnail" if has_thumb else None,
    }


def list_recipes(nsfw_allowed: bool = False) -> list[dict]:
    """Return recipe cards from both dirs. NSFW recipes are hidden unless
    mature mode is on. User recipes shadow bundled ones with the same id."""
    seen: dict[str, dict] = {}
    for source, d in _recipe_dirs():
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(d, fname)
            recipe = _load_recipe_file(path)
            if recipe is None:
                continue
            if recipe.get("nsfw") and not nsfw_allowed:
                continue
            rid = os.path.splitext(fname)[0]
            seen[rid] = _card(recipe, rid, source, os.path.isfile(_thumb_path_for(path)))
    return sorted(seen.values(), key=lambda c: c["name"].lower())


def _resolve_recipe_path(rid: str) -> Optional[str]:
    """Locate a recipe JSON by id. User dir wins (shadows bundled)."""
    safe = _slugify(rid)
    for _source, d in reversed(_recipe_dirs()):  # user first
        path = os.path.join(d, f"{safe}.json")
        if os.path.isfile(path):
            return path
    return None


def get_recipe(rid: str) -> Optional[dict]:
    path = _resolve_recipe_path(rid)
    if not path:
        return None
    recipe = _load_recipe_file(path)
    if recipe is not None:
        recipe["id"] = os.path.splitext(os.path.basename(path))[0]
    return recipe


def get_recipe_thumbnail_path(rid: str) -> Optional[str]:
    path = _resolve_recipe_path(rid)
    if not path:
        return None
    thumb = _thumb_path_for(path)
    return thumb if os.path.isfile(thumb) else None


def _make_thumbnail(source_media: str, dest_jpg: str) -> bool:
    """Render a ~480px-wide JPEG thumbnail from a video or image via ffmpeg.
    Returns True on success. Non-fatal — a recipe without a thumbnail still
    works, it just shows a placeholder card."""
    if not source_media or not os.path.isfile(source_media):
        return False
    ext = os.path.splitext(source_media)[1].lower()
    # For video, grab a frame ~1s in (avoids black lead-ins); for images,
    # just rescale. scale keeps aspect, caps width at 480.
    cmd = ["ffmpeg", "-y"]
    if ext in (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"):
        cmd += ["-ss", "1.0"]
    cmd += ["-i", source_media, "-frames:v", "1",
            "-vf", "scale='min(480,iw)':-2", "-q:v", "3", dest_jpg]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return proc.returncode == 0 and os.path.isfile(dest_jpg)
    except Exception:
        return False


def save_recipe_from_params(
    name: str,
    description: str,
    params: dict,
    mode: str,
    loras: list[dict],
    prompt_example: str,
    source_media: Optional[str] = None,
    nsfw: bool = False,
) -> dict:
    """Write a user recipe from a generation's params + LoRA pointers.

    `loras` is a list of {filename, multiplier, source_url?, size_mb?}.
    `source_media` (a path to the output video/image) is used only to
    render the thumbnail. Returns the saved recipe card.
    """
    _ensure_user_dir()
    rid = _slugify(name)
    # Avoid clobbering an existing user recipe with the same slug.
    json_path = os.path.join(USER_DIR, f"{rid}.json")
    if os.path.isfile(json_path):
        rid = f"{rid}-{int(time.time()) % 100000}"
        json_path = os.path.join(USER_DIR, f"{rid}.json")

    clean_params = {k: params[k] for k in RECIPE_PARAM_KEYS if k in params and params[k] is not None}

    recipe = {
        "recipe_version": RECIPE_SCHEMA_VERSION,
        "name": name.strip(),
        "description": (description or "").strip(),
        "mode": mode,
        "model_type": params.get("model_type", ""),
        "loras": loras or [],
        "params": clean_params,
        "prompt_example": (prompt_example or "").strip(),
        "nsfw": bool(nsfw),
        "created_at": time.time(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)

    has_thumb = False
    if source_media:
        has_thumb = _make_thumbnail(source_media, _thumb_path_for(json_path))

    return _card(recipe, rid, "user", has_thumb)


def import_recipe(data: dict) -> dict:
    """Validate and store an imported recipe JSON into the user dir."""
    if not isinstance(data, dict) or not data.get("name"):
        raise ValueError("Invalid recipe: missing name")
    _ensure_user_dir()
    rid = _slugify(data.get("name", ""))
    json_path = os.path.join(USER_DIR, f"{rid}.json")
    if os.path.isfile(json_path):
        rid = f"{rid}-{int(time.time()) % 100000}"
        json_path = os.path.join(USER_DIR, f"{rid}.json")
    # Keep only known top-level fields; coerce shape.
    recipe = {
        "recipe_version": int(data.get("recipe_version", RECIPE_SCHEMA_VERSION)),
        "name": str(data.get("name", "")).strip(),
        "description": str(data.get("description", "")).strip(),
        "mode": str(data.get("mode", "video")),
        "model_type": str(data.get("model_type", "")),
        "loras": data.get("loras", []) or [],
        "params": {k: v for k, v in (data.get("params", {}) or {}).items() if k in RECIPE_PARAM_KEYS},
        "prompt_example": str(data.get("prompt_example", "")).strip(),
        "nsfw": bool(data.get("nsfw", False)),
        "created_at": time.time(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)
    return _card(recipe, rid, "user", False)


def delete_recipe(rid: str) -> bool:
    """Delete a USER recipe (bundled recipes can't be deleted). Returns
    True if something was removed."""
    safe = _slugify(rid)
    json_path = os.path.join(USER_DIR, f"{safe}.json")
    if not os.path.isfile(json_path):
        return False
    thumb = _thumb_path_for(json_path)
    try:
        os.remove(json_path)
    except OSError:
        return False
    if os.path.isfile(thumb):
        try:
            os.remove(thumb)
        except OSError:
            pass
    return True
