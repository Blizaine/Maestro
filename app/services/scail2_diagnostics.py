"""Privacy-safe SCAIL-2 workload and host-memory diagnostics.

This module reports the effective settings and the selected checkpoint
filenames for a job. It never changes a profile, precision, or sampling value.
"""

from __future__ import annotations

import math
import ntpath
import re


_SCAIL2_ARCHITECTURES = {"scail2_14b", "scail2_1.3b"}
_QUANTIZED_FILENAME_MARKERS = (
    "int8", "int4", "fp8", "fp4", "nvfp4", "nf4", "quanto", "gguf",
    "convrot", "q2_k", "q4_k", "q5_k", "q6_k", "q8_0",
)
LOW_AVAILABLE_HOST_RAM_WARNING = (
    "Low available host RAM can cause paging; close other memory-heavy "
    "applications before generating."
)


def _basename(value):
    """Return a checkpoint basename without leaking its parent directories."""
    value = str(value or "").replace("/", "\\")
    return ntpath.basename(value)


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value):
    number = _number(value)
    return int(number) if number is not None else None


def _display_number(value, digits=2):
    number = _number(value)
    return None if number is None else round(number, digits)


def _safe_model_type(value):
    candidate = str(value or "")
    return (
        candidate
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", candidate)
        else "custom"
    )


def _safe_resolution(value):
    candidate = str(value or "").strip()
    if re.fullmatch(r"\d{2,5}\s*[xX]\s*\d{2,5}", candidate):
        return re.sub(r"\s+", "", candidate).lower()
    return "unknown"


def _checkpoint_quantization_markers(filename):
    name = _basename(filename).casefold()
    return [marker for marker in _QUANTIZED_FILENAME_MARKERS if marker in name]


def _unquantized_bf16_checkpoint(filename):
    name = _basename(filename).casefold()
    return "bf16" in name and not _checkpoint_quantization_markers(filename)


def _estimated_host_spare_gb(total_ram_gb, available_ram_gb, own_pinned_gb):
    total = _number(total_ram_gb)
    available = _number(available_ram_gb)
    pinned = _number(own_pinned_gb) or 0.0
    if total is None or total <= 0 or available is None:
        return None
    return max(0.0, available + pinned - max(4.0, total * 0.10))


def _bf16_ram_capacity_label(total_ram_gb):
    total = _number(total_ram_gb)
    if total is None or total <= 0 or total >= 40.0:
        return ""
    if total >= 31.5:
        return "32GB-class RAM"
    return f"limited-RAM system ({total:.0f} GiB)"


def build_scail2_performance_diagnostic(
    *,
    architecture,
    model_type,
    params,
    transformer_checkpoint,
    text_encoder_checkpoint,
    transformer_quantization,
    text_encoder_quantization,
    total_ram_gb,
    available_ram_gb,
    own_pinned_gb,
    profile,
    transformer_budget_mb,
    attention_head_split,
    reserved_ram_fraction,
    attention_head_split_auto_override=False,
    plan_applied=False,
    preload_mode="default",
    manual_preload_mb=0,
    gpu_name="",
    gpu_vram_gb=None,
    smart_memory_pinning=None,
    ram_allocator_mode="",
    ram_allocator_active="",
):
    """Describe one SCAIL-2 job and report measured, actionable RAM risks."""
    architecture = str(architecture or "").strip().lower()
    if architecture not in _SCAIL2_ARCHITECTURES:
        return None

    params = params if isinstance(params, dict) else {}
    transformer_name = _basename(transformer_checkpoint)
    text_encoder_name = _basename(text_encoder_checkpoint)
    total_ram = _number(total_ram_gb)
    available_ram = _number(available_ram_gb)
    own_pinned = _number(own_pinned_gb)
    spare_ram = _estimated_host_spare_gb(
        total_ram, available_ram, own_pinned,
    )
    transformer_bf16 = _unquantized_bf16_checkpoint(transformer_name)
    text_encoder_bf16 = _unquantized_bf16_checkpoint(text_encoder_name)
    transformer_budget = _integer(transformer_budget_mb)
    manual_preload = _integer(manual_preload_mb) or 0
    if manual_preload > 0:
        # Manual preload remains an independent setting even when Auto-tune
        # owns the profile. Do not label it as an automatically chosen budget.
        transformer_budget_kind = "manual preload"
    elif plan_applied:
        transformer_budget_kind = (
            "auto-tuned transformer budget"
            if transformer_budget and transformer_budget > 0
            else "auto placement with MMGP profile default"
        )
    elif transformer_budget and transformer_budget > 0:
        transformer_budget_kind = "configured transformer budget"
    elif str(preload_mode or "default") == "dynamic":
        transformer_budget_kind = "dynamic preload; MMGP profile default"
    else:
        transformer_budget_kind = "MMGP profile default"

    summary = {
        "architecture": architecture,
        "model_type": _safe_model_type(model_type),
        "resolution": _safe_resolution(params.get("resolution")),
        "video_length_frames": _integer(params.get("video_length")),
        "sliding_window_frames": _integer(params.get("sliding_window_size")),
        "steps": _integer(params.get("num_inference_steps")),
        "guidance_scale": _display_number(params.get("guidance_scale")),
        "transformer_quantization": str(transformer_quantization or ""),
        "selected_transformer_checkpoint": transformer_name,
        "transformer_filename_quantization_markers": (
            _checkpoint_quantization_markers(transformer_name)
        ),
        "transformer_unquantized_bf16": transformer_bf16,
        "text_encoder_quantization": str(text_encoder_quantization or ""),
        "selected_text_encoder_checkpoint": text_encoder_name,
        "text_encoder_filename_quantization_markers": (
            _checkpoint_quantization_markers(text_encoder_name)
        ),
        "text_encoder_unquantized_bf16": text_encoder_bf16,
        "profile": _display_number(profile),
        "configured_transformer_budget_mb": transformer_budget,
        "transformer_budget_setting_kind": transformer_budget_kind,
        "transformer_budget_uses_profile_default": (
            transformer_budget is None or transformer_budget <= 0
        ),
        "preload_mode": str(preload_mode or "default"),
        "configured_manual_preload_mb": manual_preload,
        "attention_head_split": _integer(attention_head_split),
        "attention_head_split_source": (
            "auto/per-job"
            if attention_head_split_auto_override else "saved"
        ),
        "reserved_ram_fraction": _display_number(reserved_ram_fraction, 4),
        "smart_memory_pinning": (
            bool(smart_memory_pinning)
            if smart_memory_pinning is not None else None
        ),
        "ram_allocator_mode": str(ram_allocator_mode or ""),
        "ram_allocator_active": str(ram_allocator_active or ""),
        "total_ram_gb": _display_number(total_ram),
        "available_ram_gb": _display_number(available_ram),
        "mmgp_pinned_before_load_gb": _display_number(own_pinned),
        "estimated_host_spare_gb": _display_number(spare_ram),
        "gpu_name": str(gpu_name or ""),
        "gpu_vram_gb": _display_number(gpu_vram_gb),
    }

    warnings = []
    # Catch BF16 placement pressure on 32GB-class systems and smaller hosts,
    # while measured free-memory pressure below covers larger systems too.
    capacity_label = _bf16_ram_capacity_label(total_ram)
    if capacity_label and (transformer_bf16 or text_encoder_bf16):
        components = []
        if transformer_bf16:
            components.append("transformer")
        if text_encoder_bf16:
            components.append("text encoder")
        warnings.append(
            "The selected unquantized BF16 SCAIL-2 "
            f"{' and '.join(components)} checkpoint(s) are running on a "
            f"{capacity_label}; host offload may cause paging. Select "
            "the supported INT8 checkpoint variant(s) to reduce host-memory "
            "use. Maestro did not change the selected settings."
        )

    if spare_ram is not None and spare_ram < 8.0:
        warnings.append(LOW_AVAILABLE_HOST_RAM_WARNING)

    return {"summary": summary, "warnings": warnings}


def format_scail2_performance_summary(summary):
    """Format a compact one-line log without paths or user media details."""
    if not isinstance(summary, dict):
        return ""

    def value(key, fallback="?"):
        found = summary.get(key)
        return fallback if found is None or found == "" else found

    return (
        "[SCAIL-2 Performance] "
        f"model={value('model_type')}; "
        f"selected_transformer={value('selected_transformer_checkpoint')} "
        f"({value('transformer_quantization')}); "
        f"selected_text_encoder={value('selected_text_encoder_checkpoint')} "
        f"({value('text_encoder_quantization')}); "
        f"steps={value('steps')}, CFG={value('guidance_scale')}, "
        f"resolution={value('resolution')}, "
        f"frames={value('video_length_frames')}, "
        f"window={value('sliding_window_frames')}; "
        f"planned_profile={value('profile')}, "
        f"transformer_budget_setting={value('configured_transformer_budget_mb')} MB "
        f"({value('transformer_budget_setting_kind')}), "
        f"preload_mode={value('preload_mode')}, "
        f"head_split={value('attention_head_split')} "
        f"({value('attention_head_split_source')}); "
        f"smart_pinning={value('smart_memory_pinning')}, "
        f"RAM_allocator={value('ram_allocator_mode')}"
        f"/{value('ram_allocator_active')}; "
        f"RAM={value('total_ram_gb')} GiB, "
        f"available={value('available_ram_gb')} GiB, "
        f"MMGP_pinned_before_load={value('mmgp_pinned_before_load_gb')} GiB, "
        f"reserved_fraction={value('reserved_ram_fraction')}; "
        f"GPU={value('gpu_name')} ({value('gpu_vram_gb')} GiB)"
    )


def collect_scail2_performance_diagnostic(
    *, wgp, model_definition, params, plan, hardware, live_stats, own_pinned_gb,
    settings, preload_mode="default", manual_preload_mb=0,
):
    """Resolve selected SCAIL-2 filenames and build a privacy-safe summary.

    This collector only consults the model registry and precision settings.
    It intentionally does not locate, download, or load checkpoint files.
    """
    model_definition = (
        model_definition if isinstance(model_definition, dict) else {}
    )
    architecture = str(model_definition.get("architecture") or "").lower()
    if architecture not in _SCAIL2_ARCHITECTURES:
        return None

    params = params if isinstance(params, dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    hardware = hardware if isinstance(hardware, dict) else {}
    live_stats = live_stats if isinstance(live_stats, dict) else {}
    settings = settings if isinstance(settings, dict) else {}
    model_type = params.get("model_type")
    transformer_quantization = getattr(
        wgp,
        "transformer_quantization",
        "int8",
    )
    text_encoder_quantization = (
        model_definition.get("text_encoder_quantization")
        or getattr(wgp, "text_encoder_quantization", "int8")
    )
    transformer_filename = wgp.get_model_filename(
        model_type,
        transformer_quantization,
        getattr(wgp, "transformer_dtype_policy", ""),
    )
    text_encoder_urls = wgp.get_model_recursive_prop(
        model_type,
        "text_encoder_URLs",
        return_list=True,
    )
    text_encoder_filename = ""
    if text_encoder_urls is not None:
        text_encoder_filename = wgp.get_model_filename(
            model_type=model_type,
            quantization=text_encoder_quantization,
            dtype_policy=getattr(wgp, "transformer_dtype_policy", ""),
            URLs=text_encoder_urls,
        )

    ram_stats = live_stats.get("ram") or {}
    total_ram_gb = ram_stats.get("total_gb") or hardware.get("ram_gb")
    available_ram_gb = ram_stats.get("available_gb")
    if available_ram_gb is None:
        available_ram_gb = hardware.get("ram_available_gb")
    return build_scail2_performance_diagnostic(
        architecture=architecture,
        model_type=model_type,
        params=params,
        transformer_checkpoint=transformer_filename,
        text_encoder_checkpoint=text_encoder_filename,
        transformer_quantization=transformer_quantization,
        text_encoder_quantization=text_encoder_quantization,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        own_pinned_gb=own_pinned_gb,
        profile=plan.get("profile"),
        transformer_budget_mb=plan.get("transformer_budget_mb"),
        attention_head_split=plan.get(
            "attention_head_split",
            settings.get("attention_head_split", 0),
        ),
        attention_head_split_auto_override=bool(
            plan.get("override_attention_head_split"),
        ),
        plan_applied=bool(plan.get("applied")),
        preload_mode=preload_mode,
        manual_preload_mb=manual_preload_mb,
        reserved_ram_fraction=plan.get("reserved_ram_fraction"),
        gpu_name=hardware.get("gpu_name", ""),
        gpu_vram_gb=hardware.get("gpu_vram_gb"),
        smart_memory_pinning=settings.get("smart_memory_pinning"),
        ram_allocator_mode=settings.get("ram_allocator", "default"),
        ram_allocator_active=hardware.get("ram_allocator_active", ""),
    )
