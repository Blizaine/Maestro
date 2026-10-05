"""Runtime safeguards for verified, user-imported H3 sampling recipes."""
from __future__ import annotations


def normalize_imported_h3_steps(value, profile):
    if value in (None, ""):
        value = profile.get("default_steps", 8)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float("nan")
    minimum, maximum = profile.get("min_steps", 4), profile.get("max_steps", 8)
    if isinstance(value, bool) or not numeric.is_integer() or not minimum <= numeric <= maximum:
        raise ValueError(f"This imported H3 recipe supports {minimum}–{maximum} whole denoising steps.")
    return int(numeric)


def validate_imported_h3_loras(selected):
    from .fused_turbo import fused_h3_lora_incompatibility
    for path in selected or ():
        reason = fused_h3_lora_incompatibility(str(path))
        if reason:
            raise ValueError(f"This imported H3 checkpoint already includes acceleration and cannot use {str(path).replace(chr(92), '/').rsplit('/', 1)[-1]}: {reason}.")


def normalize_baked_h3_attention(value):
    """Preserve the selected attention backend except unsupported baked SLA."""

    attention = str(value or "").strip().lower()
    return "" if attention == "sla" else attention


def normalize_imported_h3_request(body, profile, *, resolve_lora=None):
    if not (profile.get("baked_turbo") or profile.get("fused_turbo")):
        return
    selected = list(body.get("activated_loras") or ())
    if resolve_lora:
        selected += [resolve_lora(path) for path in selected]
    validate_imported_h3_loras(selected)
    body.update({"num_inference_steps": normalize_imported_h3_steps(body.get("num_inference_steps"), profile),
                 "minimax_h3_turbo_mode": False, "minimax_h3_turbo_preset": "", "guidance_scale": 1.0,
                 "flow_shift": profile["video_shift"], "audio_flow_shift": profile["audio_shift"],
                 "skip_steps_cache_type": ""})
    body["override_attention"] = normalize_baked_h3_attention(body.get("override_attention"))
    if (body.get("custom_settings") or {}).get("audio_refinement") == "enabled":
        raise ValueError("Audio refinement is unavailable with this imported H3 acceleration recipe.")


def normalize_baked_h3_request(body, model_def, *, resolve_lora=None):
    profile = (model_def or {}).get("minimax_h3_import_profile")
    if profile:
        normalize_imported_h3_request(body, profile, resolve_lora=resolve_lora)
    else:
        from .dasiwa import normalize_dasiwa_turbo_request
        normalize_dasiwa_turbo_request(body, resolve_lora=resolve_lora)
