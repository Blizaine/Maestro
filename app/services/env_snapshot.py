"""Freeze the machine-level settings that shaped one generation.

A sidecar records what the user picked in Studio, but the settings that live
in wgp_config.json -- allocators, quantization, codecs, preload policy -- are
global and mutable. A sidecar written today therefore says nothing about how
the same file was produced last month, which makes two runs impossible to
compare after the global config moves on. This copies the subset that affects
output or speed into the sidecar at write time.

Credentials are never copied. The snapshot is built from an explicit
allowlist, and any secret-shaped key is reduced to a "<name>_set" boolean,
matching the convention the settings API already uses.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

# A key whose name looks like a credential never has its value copied.
_SECRET_NAME = re.compile(
    r"(api[_-]?key|apikey|secret|token|password|passwd|credential)",
    re.IGNORECASE,
)

# Root-level config keys worth freezing, grouped the way Maestro's settings
# tabs present them so a diff between two runs reads in a familiar order.
_ROOT_KEYS = (
    # Memory and allocators
    "vram_allocator",
    "ram_allocator",
    "smart_memory_pinning",
    "read_ahead",
    "perc_reserved_mem_max",
    "vram_safety_coefficient",
    "attention_head_split",
    # Preload policy
    "preload_model_policy",
    "video_preload_mode",
    "video_preload_in_VRAM",
    "image_preload_mode",
    "image_preload_in_VRAM",
    "audio_preload_mode",
    "audio_preload_in_VRAM",
    # Compute and quantization
    "attention_mode",
    "transformer_quantization",
    "text_encoder_quantization",
    "int8_kernels",
    "enable_int8_kernels",
    "compile",
    "boost",
    "vae_config",
    "profile",
    "video_profile",
    "image_profile",
    "audio_profile",
    # Output encoding
    "video_output_codec",
    "video_container",
    "image_output_codec",
    "audio_output_codec",
    "audio_stand_alone_output_codec",
    "embed_source_images",
    # Upscaling and interpolation
    "flashvsr_mode",
    "flashvsr_backend",
    "flashvsr_topk_ratio",
    "flashvsr_persistence",
    "rife_version",
    "matanyone_version",
    "mmaudio_mode",
    "mmaudio_persistence",
    # Prompt enhancer
    "prompt_enhancer_quantization",
    "prompt_enhancer_temperature",
    "prompt_enhancer_top_p",
    "prompt_enhancer_randomize_seed",
)

# Keys under server_config["services"]. This block also holds credentials,
# so it is allowlisted rather than copied wholesale.
_SERVICE_KEYS = (
    "auto_performance",
    "auto_performance_revision",
    "llm_model_id",
    "llm_device",
    "flashvsr_mode",
    "flashvsr_topk_ratio",
)


# Config branches that track UI and session state rather than anything that
# shapes a render: which mode the user last opened, which model is selected,
# which models are visible in the picker. They move constantly -- switching
# from image to video rewrites several -- so including them would make the
# digest flip between almost any two runs and stop meaning "something that
# matters moved". h3_optimizations lives under studio preferences but is a
# real acceleration setting, so it is captured in the snapshot above.
_VOLATILE_ROOT = frozenset({
    "maestro_studio_preferences",
    "maestro_model_visibility",
    "UI_theme",
    "queue_color_scheme",
    "generation_preview",
    "clear_file_list",
    "save_queue_if_crash",
    "model_hierarchy_type",
    "notification_sound_enabled",
    "maestro_host_notification_sound_enabled",
    "maestro_host_notification_sound_volume",
})

_VOLATILE_SERVICE = frozenset({
    "active_workspace",
    "nsfw_accepted_at",
    "auto_performance_applied",
})


def _drop_volatile(config: dict) -> dict:
    """Remove UI and session state so the digest tracks machine config only."""
    stable = {
        key: value
        for key, value in config.items()
        if key not in _VOLATILE_ROOT
    }
    services = stable.get("services")
    if isinstance(services, dict):
        stable["services"] = {
            key: value
            for key, value in services.items()
            if key not in _VOLATILE_SERVICE
        }
    return stable


def _strip_secrets(value):
    """Recursively drop secret-shaped values, keeping a set/unset boolean."""
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if _SECRET_NAME.search(str(key)):
                clean[f"{key}_set"] = bool(item)
            else:
                clean[key] = _strip_secrets(item)
        return clean
    if isinstance(value, list):
        return [_strip_secrets(item) for item in value]
    return value


def config_digest(server_config: dict) -> str:
    """Fingerprint the whole non-secret config.

    The allowlist above cannot anticipate every setting. When two runs agree
    on every frozen value but disagree here, some unlisted setting moved and
    the allowlist is worth widening. UI and session state is excluded so the
    signal stays rare enough to be worth acting on.
    """
    try:
        clean = _strip_secrets(_drop_volatile(server_config or {}))
        # A repr would embed a memory address and make the digest differ
        # between runs of an identical config, so an unserializable value
        # degrades to its type name instead.
        payload = json.dumps(
            clean,
            sort_keys=True,
            default=lambda item: f"<{type(item).__name__}>",
        )
    except (TypeError, ValueError):
        return ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_env_snapshot(server_config: dict, app_version: str = "") -> dict:
    """Return the frozen environment for one generation.

    Never raises: a sidecar is worth writing even when the snapshot fails.
    """
    try:
        config = server_config or {}
        snapshot = {
            "maestro_version": app_version or "",
            "config_digest": config_digest(config),
        }

        for key in _ROOT_KEYS:
            if key in config:
                snapshot[key] = _strip_secrets(config[key])

        services = config.get("services") or {}
        if isinstance(services, dict):
            service_snapshot = {
                key: _strip_secrets(services[key])
                for key in _SERVICE_KEYS
                if key in services
            }
            if service_snapshot:
                snapshot["services"] = service_snapshot

        # Studio keeps the H3 acceleration toggles outside the root block.
        prefs = config.get("maestro_studio_preferences") or {}
        if isinstance(prefs, dict):
            h3 = prefs.get("h3_optimizations")
            if isinstance(h3, dict) and h3:
                snapshot["h3_optimizations"] = _strip_secrets(h3)

        return snapshot
    except Exception:
        return {}


def current_env_snapshot() -> dict:
    """Snapshot of the running Maestro, for a sidecar written in launch.py.

    Reads the already-imported engine's config and the repo VERSION file
    instead of launch.py globals, so a sidecar writer can import this
    locally and still run when a test executes that one function alone.
    Never raises.
    """
    try:
        wgp = sys.modules.get("wgp")
        config = getattr(wgp, "server_config", None) or {}
        try:
            version = (Path(__file__).resolve().parents[2] / "VERSION").read_text(encoding="utf-8").strip()
        except OSError:
            version = ""
        return build_env_snapshot(config, version)
    except Exception:
        return {}
