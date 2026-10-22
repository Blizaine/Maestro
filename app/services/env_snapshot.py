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
    the allowlist is worth widening.
    """
    try:
        clean = _strip_secrets(server_config or {})
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
