"""Maestro's persisted MMGP v4 controls and migration from one preload value."""
from __future__ import annotations

OUTPUT_TYPES = ("video", "image", "audio")
PRELOAD_MODES = ("default", "dynamic", "manual")
VRAM_ALLOCATORS = ("default", "vmm", "vmm_spill")
INT8_BACKENDS = ("disabled", "auto", "triton", "kitchen")
MEMORY_DEFAULTS = {
    # Keep the installed allocator policy until the user chooses the new one.
    "vram_allocator": "default",
    "smart_memory_pinning": True,
    "read_ahead": False,
    "perc_reserved_mem_max": 0,
    "attention_head_split": 0,
}
MEMORY_KEYS = frozenset(MEMORY_DEFAULTS) | {"int8_kernels"} | frozenset(
    f"{kind}_{suffix}" for kind in OUTPUT_TYPES
    for suffix in ("preload_mode", "preload_in_VRAM")
)
MODEL_LOAD_KEYS = MEMORY_KEYS - {"vram_allocator", "attention_head_split"}


def _nonnegative_int(value, default=0):
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return default


def apply_memory_defaults(config):
    """Idempotently preserve old manual preload and each explicit new choice."""
    legacy = _nonnegative_int(config.get("preload_in_VRAM", 0))
    config.setdefault("int8_kernels", "triton" if config.get("enable_int8_kernels", 1) == 1 else "disabled")
    for key, value in MEMORY_DEFAULTS.items():
        config.setdefault(key, value)
    for kind in OUTPUT_TYPES:
        config.setdefault(f"{kind}_preload_in_VRAM", legacy)
        config.setdefault(f"{kind}_preload_mode", "manual" if legacy else "default")
    return config


def preload_for_output(config, args, output_type="video"):
    """CLI MB overrides every output; legacy saved MB remains a manual choice."""
    kind = output_type if output_type in OUTPUT_TYPES else "video"
    cli = _nonnegative_int(getattr(args, "preload", 0))
    if cli:
        return "manual", cli
    legacy = _nonnegative_int(config.get("preload_in_VRAM", 0))
    mode = config.get(f"{kind}_preload_mode", "manual" if legacy else "default")
    if mode not in PRELOAD_MODES:
        mode = "default"
    mb = _nonnegative_int(config.get(f"{kind}_preload_in_VRAM", legacy)) if mode == "manual" else 0
    return mode, mb


def dynamic_preload_supported(profile):
    """Maestro's 3/3.5 have budgeted towers; 4.5 disables async transfers."""
    return profile in (2, 3, 3.5, 4, 5)


def reserved_ram_fraction(config, args):
    cli = float(getattr(args, "perc_reserved_mem_max", 0) or 0)
    return cli or float(config.get("perc_reserved_mem_max", 0) or 0) / 100


def per_job_memory_options(args, output_type="video"):
    """A generation worker's temporary options, scoped to its output family."""
    plan = getattr(args, "_maestro_per_job_memory_plan", None) or {}
    if plan.get("applied") and plan.get("output_type") == output_type:
        options = {}
        if plan.get("override_read_ahead"):
            options["read_ahead"] = plan["read_ahead"]
        if plan.get("override_reserved_ram"):
            options["reserved_ram_fraction"] = plan["reserved_ram_fraction"]
        return options
    return {}


def validate_memory_updates(body):
    """Validate all memory fields before an API request changes any state."""
    updated = {}
    for key in MEMORY_KEYS & body.keys():
        value = body[key]
        if key == "vram_allocator":
            valid = isinstance(value, str) and value in VRAM_ALLOCATORS
        elif key == "int8_kernels":
            valid = isinstance(value, str) and value in INT8_BACKENDS
        elif key.endswith("_preload_mode"):
            valid = isinstance(value, str) and value in PRELOAD_MODES
        elif key in ("smart_memory_pinning", "read_ahead"):
            valid = isinstance(value, bool)
        else:
            limit = 3 if key == "attention_head_split" else 80 if key == "perc_reserved_mem_max" else 40000
            valid = isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= limit
        if not valid:
            raise ValueError(f"Invalid {key}: {value!r}")
        updated[key] = value
    return updated


def memory_settings_snapshot(config):
    normalized = apply_memory_defaults(dict(config))
    return {key: normalized[key] for key in sorted(MEMORY_KEYS)}
