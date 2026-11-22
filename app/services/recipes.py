"""
Recipes — reusable generation presets and bounded workflow guides.

A recipe bundles a model, its LoRAs (by pointer, never the weights), the
tuned generation settings, and an editable example prompt. Applying one
switches the model, fills the settings into the active sub-mode's working
set, and prepopulates the prompt so the user just edits the subject.

Two sources, both scanned at list time:
  - app/recipes/       — bundled starter pack, tracked in the repo.
  - app/recipes_user/  — user-saved + imported recipes, gitignored.

Each recipe is a JSON file; its thumbnail (same basename, .jpg) and optional
short video preview (.preview.mp4) live beside it. LoRAs are referenced by {filename, multiplier, source_url,
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

RECIPE_SCHEMA_VERSION = 2

# Imported recipes may include a small, declarative workflow guide. These
# limits keep recipe details useful without turning the format into scripts.
MAX_REQUIREMENTS = 12
MAX_WORKFLOW_STEPS = 12
MAX_RECIPE_LORAS = 32
MAX_REQUIREMENT_LENGTH = 240
MAX_STEP_TITLE_LENGTH = 100
MAX_STEP_INSTRUCTION_LENGTH = 600
MAX_GUIDANCE_LENGTH = 2400
WORKFLOW_STEP_TYPES = frozenset({
    "prepare_prompt", "configure", "verify_requirements", "generate", "review", "iterate",
})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"})

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


def _preview_path_for(json_path: str) -> str:
    return os.path.splitext(json_path)[0] + ".preview.mp4"


def _normalise_text(value: object, field: str, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        if value is None and not required:
            return ""
        raise ValueError(f"Invalid recipe: {field} must be text")
    result = value.strip()
    if required and not result:
        raise ValueError(f"Invalid recipe: {field} is required")
    if len(result) > limit:
        raise ValueError(f"Invalid recipe: {field} exceeds {limit} characters")
    return result


def _normalise_workflow_fields(data: dict, *, strict: bool) -> dict:
    """Return bounded guide fields, accepting old v1 recipes by default."""
    defaults = {
        "requirements": [],
        "model_guidance": "",
        "prompt_guidance": "",
        "workflow_guidance": "",
        "workflow_steps": [],
    }
    try:
        requirements = data.get("requirements", [])
        if requirements is None and not strict:
            requirements = []
        if not isinstance(requirements, list) or len(requirements) > MAX_REQUIREMENTS:
            raise ValueError(f"requirements must be a list of at most {MAX_REQUIREMENTS} items")
        clean_requirements = [
            _normalise_text(item, "requirement", MAX_REQUIREMENT_LENGTH, required=True)
            for item in requirements
        ]

        clean_steps = []
        steps = data.get("workflow_steps", [])
        if steps is None and not strict:
            steps = []
        if not isinstance(steps, list) or len(steps) > MAX_WORKFLOW_STEPS:
            raise ValueError(f"workflow_steps must be a list of at most {MAX_WORKFLOW_STEPS} items")
        for item in steps:
            if not isinstance(item, dict):
                raise ValueError("each workflow step must be an object")
            step_type = item.get("type")
            if not isinstance(step_type, str) or step_type not in WORKFLOW_STEP_TYPES:
                raise ValueError("workflow step type is not supported")
            step = {
                "type": step_type,
                "title": _normalise_text(item.get("title"), "step title", MAX_STEP_TITLE_LENGTH, required=True),
                "instruction": _normalise_text(
                    item.get("instruction"), "step instruction", MAX_STEP_INSTRUCTION_LENGTH, required=True
                ),
            }
            setting = item.get("setting")
            if setting is not None:
                if step_type != "configure" or not isinstance(setting, str) or setting not in RECIPE_PARAM_KEYS:
                    raise ValueError("workflow step setting must name an allowlisted recipe parameter")
                step["setting"] = setting
            if "value" in item:
                value = item["value"]
                if step_type != "configure" or not isinstance(value, (str, int, float, bool, type(None))):
                    raise ValueError("workflow step values must be scalar configure values")
                if isinstance(value, float) and (value != value or abs(value) == float("inf")):
                    raise ValueError("workflow step values must be finite")
                if isinstance(value, str) and len(value) > 500:
                    raise ValueError("workflow step value exceeds 500 characters")
                step["value"] = value
            clean_steps.append(step)

        return {
            "requirements": clean_requirements,
            "model_guidance": _normalise_text(data.get("model_guidance"), "model_guidance", MAX_GUIDANCE_LENGTH),
            "prompt_guidance": _normalise_text(data.get("prompt_guidance"), "prompt_guidance", MAX_GUIDANCE_LENGTH),
            "workflow_guidance": _normalise_text(data.get("workflow_guidance"), "workflow_guidance", MAX_GUIDANCE_LENGTH),
            "workflow_steps": clean_steps,
        }
    except ValueError:
        if strict:
            raise
        return defaults


def _default_workflow_fields(model_type: str, loras: list[dict], mode: str) -> dict:
    requirements = [f"Model: {model_type}"] if model_type else []
    requirements.extend(f"LoRA available: {lora.get('filename')}" for lora in loras if lora.get("filename"))
    sample = "a short clip" if mode == "video" else "a single image" if mode == "image" else "a short audio sample"
    return {
        "requirements": requirements[:MAX_REQUIREMENTS],
        "model_guidance": f"Confirm that {model_type or 'the selected model'} is installed and that the listed LoRAs are available.",
        "prompt_guidance": "Adapt the example prompt to your subject. Preserve any style trigger words associated with the listed LoRAs.",
        "workflow_guidance": "Review the model and settings before generating. Start with a small sample and adjust after reviewing it.",
        "workflow_steps": [
            {"type": "prepare_prompt", "title": "Adapt the prompt", "instruction": "Replace subject-specific details while keeping the style cues you want."},
            {"type": "verify_requirements", "title": "Check the setup", "instruction": "Confirm the model and any listed LoRAs are available."},
            {"type": "generate", "title": "Generate a sample", "instruction": f"Render {sample} with the recipe settings."},
            {"type": "review", "title": "Review the result", "instruction": "Check prompt adherence and motion or composition, then decide what to change."},
        ],
    }


def _recipe_workflow_fields(recipe: dict, *, strict: bool = False) -> dict:
    fields = _normalise_workflow_fields(recipe, strict=strict)
    defaults = _default_workflow_fields(
        str(recipe.get("model_type", "") or ""),
        [item for item in recipe.get("loras", []) if isinstance(item, dict)]
        if isinstance(recipe.get("loras", []), list) else [],
        str(recipe.get("mode", "video") or "video"),
    )
    for key, value in defaults.items():
        if key not in recipe:
            fields[key] = value
    return fields


def _load_recipe_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data.get("name"):
            return None
        return data
    except Exception:
        return None


def _card(recipe: dict, rid: str, source: str, has_thumb: bool, has_preview: bool = False) -> dict:
    card = {
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
        "preview_url": f"/api/v1/recipes/{rid}/thumbnail?preview=true" if has_preview else None,
    }
    return card


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
            seen[rid] = _card(
                recipe,
                rid,
                source,
                os.path.isfile(_thumb_path_for(path)),
                os.path.isfile(_preview_path_for(path)),
            )
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
        rid = os.path.splitext(os.path.basename(path))[0]
        source = "user" if os.path.dirname(path) == USER_DIR else "bundled"
        recipe.update(_card(
            recipe,
            rid,
            source,
            os.path.isfile(_thumb_path_for(path)),
            os.path.isfile(_preview_path_for(path)),
        ))
        recipe.update(_recipe_workflow_fields(recipe))
    return recipe


def get_recipe_thumbnail_path(rid: str) -> Optional[str]:
    path = _resolve_recipe_path(rid)
    if not path:
        return None
    thumb = _thumb_path_for(path)
    return thumb if os.path.isfile(thumb) else None


def get_recipe_preview_path(rid: str) -> Optional[str]:
    path = _resolve_recipe_path(rid)
    if not path:
        return None
    preview = _preview_path_for(path)
    return preview if os.path.isfile(preview) else None


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
    if ext in VIDEO_EXTENSIONS:
        cmd += ["-ss", "1.0"]
    cmd += ["-i", source_media, "-frames:v", "1",
            "-vf", "scale='min(480,iw)':-2", "-q:v", "3", dest_jpg]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return proc.returncode == 0 and os.path.isfile(dest_jpg)
    except Exception:
        return False


def _make_preview(source_media: str, dest_mp4: str) -> bool:
    """Store a short, muted, web-friendly looping preview for video outputs."""
    if not source_media or not os.path.isfile(source_media):
        return False
    if os.path.splitext(source_media)[1].lower() not in VIDEO_EXTENSIONS:
        return False
    cmd = [
        "ffmpeg", "-y", "-i", source_media, "-t", "4", "-an",
        "-vf", "fps=12,scale='min(480,iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", dest_mp4,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        return proc.returncode == 0 and os.path.isfile(dest_mp4)
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
    requirements: Optional[list[str]] = None,
    model_guidance: str = "",
    prompt_guidance: str = "",
    workflow_guidance: str = "",
    workflow_steps: Optional[list[dict]] = None,
) -> dict:
    """Write a user recipe from a generation's params + LoRA pointers.

    `loras` is a list of {filename, multiplier, source_url?, size_mb?}.
    `source_media` is used for a thumbnail and, for video outputs, an optional
    short looping preview. Workflow fields are bounded guidance, never code.
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
    workflow_fields = _default_workflow_fields(str(recipe["model_type"] or ""), recipe["loras"], mode)
    workflow_fields.update({
        "requirements": requirements if requirements is not None else workflow_fields["requirements"],
        "model_guidance": model_guidance or workflow_fields["model_guidance"],
        "prompt_guidance": prompt_guidance or workflow_fields["prompt_guidance"],
        "workflow_guidance": workflow_guidance or workflow_fields["workflow_guidance"],
        "workflow_steps": workflow_steps if workflow_steps is not None else workflow_fields["workflow_steps"],
    })
    recipe.update(_normalise_workflow_fields(workflow_fields, strict=True))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)

    has_thumb = False
    has_preview = False
    if source_media:
        has_thumb = _make_thumbnail(source_media, _thumb_path_for(json_path))
        has_preview = _make_preview(source_media, _preview_path_for(json_path))

    return _card(recipe, rid, "user", has_thumb, has_preview)


def import_recipe(data: dict) -> dict:
    """Validate and store an imported recipe JSON into the user dir."""
    if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ValueError("Invalid recipe: missing name")
    name = _normalise_text(data["name"], "name", 120, required=True)
    raw_params = data.get("params", {})
    raw_loras = data.get("loras", [])
    if raw_params is None:
        raw_params = {}
    if raw_loras is None:
        raw_loras = []
    if not isinstance(raw_params, dict):
        raise ValueError("Invalid recipe: params must be an object")
    if not isinstance(raw_loras, list) or len(raw_loras) > MAX_RECIPE_LORAS:
        raise ValueError(f"Invalid recipe: loras must be a list of at most {MAX_RECIPE_LORAS} items")
    if any(
        not isinstance(lora, dict)
        or not isinstance(lora.get("filename"), str)
        or not lora["filename"].strip()
        for lora in raw_loras
    ):
        raise ValueError("Invalid recipe: each LoRA needs a filename")
    _ensure_user_dir()
    rid = _slugify(name)
    json_path = os.path.join(USER_DIR, f"{rid}.json")
    if os.path.isfile(json_path):
        rid = f"{rid}-{int(time.time()) % 100000}"
        json_path = os.path.join(USER_DIR, f"{rid}.json")
    # Keep only known top-level fields; coerce shape.
    recipe = {
        "recipe_version": max(1, min(RECIPE_SCHEMA_VERSION, int(data.get("recipe_version", RECIPE_SCHEMA_VERSION)))),
        "name": name,
        "description": str(data.get("description", "")).strip(),
        "mode": str(data.get("mode", "video")),
        "model_type": str(data.get("model_type", "")),
        "loras": raw_loras,
        "params": {k: v for k, v in raw_params.items() if k in RECIPE_PARAM_KEYS},
        "prompt_example": str(data.get("prompt_example", "")).strip(),
        "nsfw": bool(data.get("nsfw", False)),
        "created_at": time.time(),
    }
    recipe.update(_recipe_workflow_fields({**recipe, **data}, strict=True))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)
    return _card(recipe, rid, "user", False, False)


def delete_recipe(rid: str) -> bool:
    """Delete a USER recipe (bundled recipes can't be deleted). Returns
    True if something was removed."""
    safe = _slugify(rid)
    json_path = os.path.join(USER_DIR, f"{safe}.json")
    if not os.path.isfile(json_path):
        return False
    thumb = _thumb_path_for(json_path)
    preview = _preview_path_for(json_path)
    try:
        os.remove(json_path)
    except OSError:
        return False
    if os.path.isfile(thumb):
        try:
            os.remove(thumb)
        except OSError:
            pass
    if os.path.isfile(preview):
        try:
            os.remove(preview)
        except OSError:
            pass
    return True
