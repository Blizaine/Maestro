"""
Performance recommendation engine for the Performance Auto-Tune feature.

Single entry point: `recommend_settings(hw)` takes the dict from
`hardware_detect.detect_hardware()` and returns a dict of settings
to apply to wgp_config.json. Pure function — no side effects, no
imports beyond stdlib, easy to unit test.

The recommendation table mirrors the design in
`memory/project_performance_auto.md`. Rationale lives there; this
module is just the lookup.

Coefficient policy (decided 2026-05-02):
- 0.80 across all tiers with VRAM ≥ 12 GB
- 0.70 for VRAM < 12 GB
- We do NOT bump above 0.80 even on 24 GB cards. Real-world data
  (user's RTX 4090) shows 0.80 sometimes OOMs on heavy workloads;
  going higher would be actively harmful.
"""
from __future__ import annotations

from typing import Optional


# Wan2GP `mmgp.profile_type` integer values, hardcoded here to avoid
# importing torch/mmgp from a pure-logic module:
#   1   = HighRAM_HighVRAM
#   2   = HighRAM_LowVRAM
#   3   = LowRAM_HighVRAM
#   3.5 = VeryLowRAM_HighVRAM
#   4   = LowRAM_LowVRAM ("recommended" in Wan2GP)
#   4.5 = LowRAM_LowVRAM+ (saves ~1 GB extra)
#   5   = VeryLowRAM_LowVRAM


# Plain-English profile descriptions for the auto card. Power users
# still see "Profile N — HighRAM_LowVRAM" in the advanced dropdown
# (so they can grep Wan2GP forums); novices see this.
PROFILE_DESCRIPTIONS = {
    1:   "Optimized for fastest generation",
    2:   "Balanced for versatility (large batches, long videos)",
    3:   "Optimized for short videos with limited RAM",
    3.5: "Optimized for short videos with very limited RAM",
    4:   "Optimized for longer videos with limited VRAM",
    4.5: "Optimized for very limited VRAM",
    5:   "Maximum offload — slower but works on small machines",
}


def _pick_profile(ram_tier: str, vram_tier: str) -> float:
    """Pick the mmgp profile number for a (ram, vram) tier pair.

    Maps directly to the table in project_performance_auto.md.
    Returns a float because Wan2GP's profile values include 3.5 and 4.5.
    """
    if vram_tier == "high":  # ≥24 GB VRAM
        if ram_tier == "high":
            return 1     # HighRAM_HighVRAM
        elif ram_tier == "low":
            return 3     # LowRAM_HighVRAM
        else:  # very_low
            return 3.5   # VeryLowRAM_HighVRAM

    if vram_tier == "low":   # 12-23 GB VRAM
        if ram_tier == "high":
            return 2     # HighRAM_LowVRAM
        elif ram_tier == "low":
            return 4     # LowRAM_LowVRAM (Wan2GP's "recommended")
        else:  # very_low
            return 5     # VeryLowRAM_LowVRAM

    # vram_tier == "tight" (<12 GB) — squeeze every last byte
    if ram_tier == "high":
        return 4         # plenty of RAM to absorb offload
    elif ram_tier == "low":
        return 4.5       # 4+ saves an extra ~1 GB VRAM
    else:
        return 5         # the tightest profile


def _pick_quantization(vram_tier: str, supports_fp8: bool, supports_nvfp4: bool) -> str:
    """Pick the transformer quantization mode.

    Returns INT8 universally as of 2026-05-03. User-side quality testing
    (Flux 2 Klein 9B and LTX-2.3 22B, both BF16 vs INT8 side-by-side on
    an RTX 4090) found virtually no quality improvement from BF16 — the
    extra VRAM and slower load are wasted. Picking INT8 across all tiers
    means:
      - Faster cold loads (smaller files, less RAM↔VRAM streaming)
      - More headroom for the activation spikes that actually OOM
        (long videos, high resolution, VAE decode)
      - Power users who want to A/B BF16 themselves can flip the
        Transformer Quantization dropdown in advanced settings — that
        still works exactly as before, this only changes the auto-tune
        recommendation.

    Power users who genuinely need different quants (e.g. for FP8 on
    RTX 40xx+ where FP8 kernels are faster than INT8 ops) should pick
    a model variant with the quant baked in (e.g. "LTX-2.3 Distilled
    FP8 22B") rather than relying on this global setting — that's the
    cleaner mental model and avoids the FP8-falls-back-to-INT8 footgun.

    Args kept for API stability — supports_fp8 and supports_nvfp4 may
    matter again if a future model materially benefits from FP8/NVFP4
    over INT8. Currently unused.
    """
    return "int8"


def _pick_vae_config(vram_tier: str) -> int:
    """Pick the VAE tiling mode.

    Wan2GP values:
      0 = Auto (let the runtime pick per-model)
      1 = Full (fast, high VRAM)
      2 = Medium tiling
      3 = Aggressive tiling (low VRAM)
    """
    if vram_tier == "high":
        return 1   # Full — VRAM is plentiful
    if vram_tier == "low":
        return 0   # Auto — let runtime decide per-model
    return 3       # Aggressive — squeeze it


def _pick_coefficient(vram_tier: str) -> float:
    """Pick the VRAM safety coefficient.

    See module docstring — flat 0.80 for ≥12 GB, 0.70 for <12 GB.
    Real-world data shows even 24 GB cards OOM at 0.80 on heavy
    workloads, so bumping above 0.80 would be harmful.
    """
    if vram_tier == "tight":
        return 0.70
    return 0.80


def _profile_label(profile: float) -> str:
    """Human-readable label combining profile number + description."""
    desc = PROFILE_DESCRIPTIONS.get(profile, "Auto-tuned profile")
    # Format profile cleanly: "1" not "1.0", "3.5" stays as "3.5"
    profile_str = f"{profile:g}"
    return f"Profile {profile_str} — {desc}"


def recommend_settings(hw: dict) -> dict:
    """Translate hardware-detection dict → recommended config values.

    Returns a dict ready to merge into wgp_config.json. Includes a
    `_recommendation_label` field for the UI card readout (the
    underscore prefix marks it as not a config value to apply, just
    metadata for display).

    AMD/CPU fallback: when `cuda_available=False`, returns the most
    conservative profile (4.5 + INT8 + aggressive tiling) and a
    label noting auto-tune isn't really applicable. The user will
    likely want to manually configure, but at least nothing crashes
    on first launch.

    Schema:
      video_profile: float   — 1, 2, 3, 3.5, 4, 4.5, or 5
      image_profile: float   — same scale
      audio_profile: float   — kept at 3.5 (legacy default — see note)
      transformer_quantization: str — "int8" | "fp8" | "bf16"
      vae_config: int        — 0 (auto) | 1 (full) | 2 (medium) | 3 (aggressive)
      vram_safety_coefficient: float — 0.70 or 0.80
      attention_mode: str    — always "auto" (let wgp pick from installed kernels)
      compile: str           — always "" (off; user opts in via advanced)
      _recommendation_label: str — for the UI card readout
      _recommendation_reason: str — for the tooltip / debug log
    """
    if not hw.get("cuda_available", False):
        # No CUDA — return conservative fallback. User probably needs
        # manual setup, but defaults shouldn't crash.
        return {
            "video_profile": 4.5,
            "image_profile": 4.5,
            "audio_profile": 4.5,  # match the others — same hardware, same constraint
            "transformer_quantization": "int8",
            "vae_config": 3,
            "vram_safety_coefficient": 0.70,
            "attention_mode": "auto",
            "compile": "",
            "_recommendation_label": "Auto-tune unavailable on this hardware",
            "_recommendation_reason": (
                "No CUDA-capable GPU detected. Maestro requires NVIDIA GPU "
                "with 6+ GB VRAM. Default conservative profile applied — "
                "performance may be limited."
            ),
        }

    ram_tier = hw.get("ram_tier", "low")
    vram_tier = hw.get("vram_tier", "low")
    supports_fp8 = bool(hw.get("supports_fp8", False))
    supports_nvfp4 = bool(hw.get("supports_nvfp4", False))

    profile = _pick_profile(ram_tier, vram_tier)
    quant = _pick_quantization(vram_tier, supports_fp8, supports_nvfp4)
    vae = _pick_vae_config(vram_tier)
    coef = _pick_coefficient(vram_tier)

    # Build a one-line "why" string for the tooltip / debug log.
    # Mentions the actual numbers so support tickets have context.
    gpu_name = hw.get("gpu_name", "GPU")
    vram_gb = hw.get("gpu_vram_gb", 0)
    ram_gb = hw.get("ram_gb", 0)
    reason = (
        f"{gpu_name} ({vram_gb} GB VRAM, {ram_gb} GB RAM): "
        f"VRAM tier={vram_tier}, RAM tier={ram_tier} → profile {profile:g}, "
        f"{quant.upper()}, VAE config {vae}, coefficient {coef}"
    )

    return {
        "video_profile": profile,
        "image_profile": profile,
        # Audio uses the same profile as video/image — the hardware is
        # the same hardware regardless of mode. Wan2GP's historical
        # default of 3.5 (VeryLowRAM_HighVRAM) was a one-size-fits-all
        # fallback that's actively wrong on machines with plenty of
        # RAM: a user with 128 GB RAM + 24 GB VRAM should get Profile 1
        # for audio too, not be artificially constrained to a "very low
        # RAM" profile.
        "audio_profile": profile,
        "transformer_quantization": quant,
        "vae_config": vae,
        "vram_safety_coefficient": coef,
        "attention_mode": "auto",
        "compile": "",
        "_recommendation_label": _profile_label(profile),
        "_recommendation_reason": reason,
    }


def applied_keys() -> list:
    """Return the list of config keys that recommend_settings() actually
    sets (i.e. excludes the underscore-prefixed metadata fields).

    Used by the apply endpoint to know which keys to write into the
    config file vs which are display-only.
    """
    return [
        "video_profile",
        "image_profile",
        "audio_profile",
        "transformer_quantization",
        "vae_config",
        "vram_safety_coefficient",
        "attention_mode",
        "compile",
    ]


# ── Per-job coefficient adjustment ─────────────────────────────────
# The base vram_safety_coefficient (set once by auto-tune) doesn't
# account for memory categories that vary per generation:
#   - LoRA weights merged into the transformer (varies by file size)
#   - Multi-stage pipeline overhead (intermediate latents stay
#     resident across passes)
#
# Real-world data (RTX 4090, 24 GB):
#   LTX-2.3 dist 1.1, 720p × 20s, no LoRAs:
#     - 1 stage:  ~19 GB peak (matches 0.80 cap)
#     - 2 stage:  ~21 GB peak (over the cap by ~2 GB)
#     - + 3 LoRAs: 23.7+ GB peak (OOM territory)
#
# This helper computes a per-job effective coefficient that accounts
# for both factors, so the user's auto-tuned base can stay sensible
# (0.80) while real jobs get a tighter cap when they actually need it.

# Per additional pipeline stage at the BASELINE 720p resolution:
# ~2 GB of intermediate latent memory stays resident on top of stage
# 1's working set. Empirical from RTX 4090 stage-2 measurements at
# 720p; the actual per-stage cost scales with output resolution since
# stage 2 (refinement / upscale) operates at full output resolution
# and its activation peak grows linearly with pixel count.
#
# Calibrated 2026-05-15: a 20s × 1080p × 2-stage × no-LoRA job on
# RTX 4090 OOM'd on stage 2 with the previous flat-2.0 GB penalty
# (effective coef = 0.513 = 12.3 GB cap). Scaling pass overhead by
# the resolution factor (1080p = 2.25× of 720p baseline) raises the
# 1080p penalty to 4.5 GB, dropping the effective coef to ~0.41 (cap
# ~9.8 GB). That triggers more aggressive offload at stage 2 and
# stays under the 24 GB ceiling. 720p jobs are unaffected (scale = 1.0).
_PASS_OVERHEAD_GB_PER_EXTRA_STAGE = 2.0

# LoRA VRAM cost — empirically much lower than the file size because
# mmgp can offload LoRA-merged weights along with the rest of the
# transformer. The 1.15× factor we used initially was based on a
# "fully merged + resident" assumption that doesn't hold in practice.
# Calibrated against RTX 4090 LTX-2.3 distilled data: at the previous
# 1.15× factor, 5 LoRAs (4 GB total) over-offloaded by ~5 GB compared
# to what would have been safe. 0.6 lands closer to optimal for that
# case while still preventing OOM on heavy stacks.
_LORA_OVERHEAD_FACTOR = 0.6

# Compute-size scaling — resolution × frame count drives activation
# memory in attention and VAE decode. We compare each job to a
# baseline (720p × ~10 s @ 24 fps) and apply a penalty proportional
# to the excess.
#
# Calibrated against RTX 4090 LTX-2.3 distilled data:
#   - 1080p × 20s + 3 LoRAs (compute ratio 4.5×):
#       · 1.4×/unit + LoRA 1.15 → effective 0.411, peak 20.1 GB ✓
#       · 1.0×/unit + LoRA 0.6  → effective 0.518, slight OOM ✗
#       · 1.4×/unit + LoRA 0.6  → effective 0.459, predicted 21-22 GB peak ✓
#
# 1.4 GB/unit was correct for the compute penalty; the over-aggressive
# part was the LoRA penalty (now 0.6×). Keep compute at 1.4 so heavy
# resolution × clip-length jobs still get appropriate offloading.
_BASELINE_PIXELS = 1280 * 720          # 720p
_BASELINE_FRAMES = 240                 # ~10 s at 24 fps
_BASELINE_COMPUTE = float(_BASELINE_PIXELS * _BASELINE_FRAMES)
_GB_PER_COMPUTE_RATIO_UNIT = 1.4

# Light-job bonus: when compute_ratio < 1, allow the coefficient to
# rise modestly above the base so we don't leave performance on the
# table for short low-res clips (480p × 5s, etc.). Capped by
# _COEFFICIENT_CEILING below.
_BONUS_GB_PER_RATIO_UNIT_BELOW_1 = 0.7  # half-strength of the penalty

# Coefficient floor — prevents the helper from recommending a setting
# so tight that the model can't even load. If we'd go below this,
# the user's job is genuinely too heavy for the hardware and an OOM
# is likely; the OOM recovery banner will catch it and prompt the
# user to lower their LoRA stack, resolution, or clip length.
_COEFFICIENT_FLOOR = 0.40

# Coefficient ceiling — prevents the helper from raising the coefficient
# so high that even a "light" job blows up due to unaccounted overhead
# (e.g. text encoder, VAE, peak attention). Light jobs benefit from
# faster generation but we cap the upside conservatively.
_COEFFICIENT_CEILING = 0.92


def _parse_resolution(resolution) -> Optional[tuple]:
    """Parse a resolution string like "1280x720" or "1920x1080" → (w, h).

    Returns None if input is missing or malformed. Accepts either a
    string with 'x' separator or a (w, h) tuple/list (defensive against
    callers that pre-parse).
    """
    if resolution is None:
        return None
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        try:
            return int(resolution[0]), int(resolution[1])
        except (TypeError, ValueError):
            return None
    if not isinstance(resolution, str):
        return None
    parts = resolution.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def compute_per_job_coefficient(
    base_coef: float,
    total_vram_gb: float,
    active_loras: Optional[list] = None,
    lora_dir: Optional[str] = None,
    stage_count: int = 1,
    resolution: Optional[str] = None,
    video_length_frames: Optional[int] = None,
) -> dict:
    """Compute a per-job VRAM safety coefficient.

    Adjusts the auto-tuned base coefficient based on how much the
    *current job* deviates from a baseline single-stage no-LoRA run
    at 720p × ~10 s. Heavier jobs get a tighter cap (more aggressive
    offloading); lighter jobs get a modest bonus (less offload, faster
    generation). Pure function — no side effects, no imports beyond
    os/stdlib at call time.

    Adjustment factors:
      1. LoRA overhead: file size of merged LoRAs adds to peak VRAM
      2. Multi-stage: each extra stage holds intermediate latents
      3. Compute size: resolution × frame count drives activation
         memory in attention and VAE decode

    Args:
        base_coef: Auto-tuned coefficient from server config (e.g. 0.80).
        total_vram_gb: Total GPU VRAM in GB. Used to convert byte-level
            overhead into a coefficient delta. Must be > 0.
        active_loras: List of LoRA filenames active for this job. If
            None or empty, no LoRA penalty is applied.
        lora_dir: Directory containing the LoRA files. Used with
            active_loras to compute total LoRA bytes. If None, LoRA
            penalty is skipped (we can't size what we can't find).
        stage_count: Number of pipeline stages (1 = single,
            2 = standard, 3 = progressive). Each additional stage
            past 1 adds the per-stage overhead.
        resolution: Output resolution string ("WIDTHxHEIGHT") or None.
            None skips the compute-size penalty/bonus entirely.
        video_length_frames: Total output frames. None or 0 skips the
            compute-size penalty/bonus. (For images, pass 1.)

    Returns:
        Dict with keys:
            effective_coef: float — adjusted coefficient, clamped to
                [_COEFFICIENT_FLOOR, _COEFFICIENT_CEILING]
            base_coef: float — pass-through of the input
            lora_total_gb: float — total bytes of active LoRAs in GB
            lora_penalty: float — coefficient delta for LoRAs (≥0)
            pass_penalty: float — coefficient delta for multi-pass (≥0)
            compute_penalty: float — coefficient delta for resolution ×
                length. Positive = penalty (heavier than baseline);
                negative = bonus (lighter than baseline).
            compute_ratio: float — job_compute / baseline_compute
                (1.0 = baseline, 4.5 = 1080p × 20s, etc.)
            stage_count: int — pass-through of the input
            floored: bool — True if result was clamped to floor
            ceilinged: bool — True if result was clamped to ceiling
            reasons: list[str] — human-readable breakdown lines for
                log lines and UI tooltips. Empty when no adjustment
                was applied.
    """
    import os

    if total_vram_gb <= 0:
        # Can't compute coefficient deltas without knowing VRAM.
        # Pass through the base unchanged.
        return {
            "effective_coef": base_coef,
            "base_coef": base_coef,
            "lora_total_gb": 0.0,
            "lora_penalty": 0.0,
            "pass_penalty": 0.0,
            "compute_penalty": 0.0,
            "compute_ratio": 1.0,
            "stage_count": stage_count,
            "floored": False,
            "ceilinged": False,
            "reasons": [],
        }

    reasons = []

    # 1. LoRA overhead — sum file sizes for any LoRAs we can find.
    lora_total_bytes = 0
    lora_count = 0
    if active_loras and lora_dir:
        for fname in active_loras:
            if not isinstance(fname, str) or not fname:
                continue
            path = os.path.join(lora_dir, fname)
            if os.path.isfile(path):
                try:
                    lora_total_bytes += os.path.getsize(path)
                    lora_count += 1
                except OSError:
                    pass  # File disappeared between listing and sizing
    lora_total_gb = lora_total_bytes / (1024 ** 3)
    lora_penalty = (lora_total_gb * _LORA_OVERHEAD_FACTOR) / total_vram_gb
    if lora_penalty > 0:
        reasons.append(
            f"-{lora_penalty:.3f} for {lora_count} LoRA(s) totaling "
            f"{lora_total_gb:.2f} GB"
        )

    # 2. Multi-pass overhead — extra stages add intermediate latent
    # memory. Scaled by the OUTPUT resolution but ONLY UPWARD: 720p
    # baseline = 1.0× (unchanged behavior); 1080p = 2.25×; 4K = 9×.
    # We don't scale DOWN for low-res because the existing 2 GB
    # constant was empirically validated for 480/540/720p and those
    # resolutions already work fine. Loosening their caps risks
    # surfacing regressions we haven't tested for. The fix is purely
    # additive for high-res jobs: it forces tighter caps where the
    # existing flat constant is empirically too lax (1080p stage 2
    # OOM observed on RTX 4090 with 0 LoRAs at compute_ratio 4.5×,
    # confirmed by user 2026-05-15).
    extra_stages = max(0, int(stage_count) - 1)
    res_tuple = _parse_resolution(resolution)
    res_scale = 1.0
    if res_tuple:
        rw, rh = res_tuple
        # max(1.0, ...) — never DECREASE pass overhead for low-res jobs.
        res_scale = max(1.0, (rw * rh) / float(_BASELINE_PIXELS))
    pass_overhead_gb = extra_stages * _PASS_OVERHEAD_GB_PER_EXTRA_STAGE * res_scale
    pass_penalty = pass_overhead_gb / total_vram_gb
    if pass_penalty > 0:
        scale_note = f", {res_scale:.2f}× resolution scale" if res_scale > 1.0 else ""
        reasons.append(
            f"-{pass_penalty:.3f} for {extra_stages} extra pipeline "
            f"stage(s) ({pass_overhead_gb:.1f} GB{scale_note})"
        )

    # 3. Compute-size penalty/bonus — resolution × frames vs baseline.
    compute_ratio = 1.0
    compute_penalty = 0.0
    # res_tuple computed above for pass overhead; reuse here.
    frames = video_length_frames if isinstance(video_length_frames, (int, float)) else None
    if res_tuple and frames and frames > 0:
        width, height = res_tuple
        job_compute = float(width * height * frames)
        compute_ratio = job_compute / _BASELINE_COMPUTE

        if compute_ratio > 1.0:
            # Heavier than baseline → penalize
            extra_ratio = compute_ratio - 1.0
            compute_overhead_gb = extra_ratio * _GB_PER_COMPUTE_RATIO_UNIT
            compute_penalty = compute_overhead_gb / total_vram_gb
            reasons.append(
                f"-{compute_penalty:.3f} for {compute_ratio:.2f}× compute "
                f"vs baseline ({width}×{height} × {frames} frames, "
                f"{compute_overhead_gb:.1f} GB)"
            )
        elif compute_ratio < 1.0:
            # Lighter than baseline → bonus (negative penalty)
            ratio_below = 1.0 - compute_ratio
            bonus_gb = ratio_below * _BONUS_GB_PER_RATIO_UNIT_BELOW_1
            compute_penalty = -(bonus_gb / total_vram_gb)
            reasons.append(
                f"+{(-compute_penalty):.3f} bonus for {compute_ratio:.2f}× compute "
                f"vs baseline ({width}×{height} × {frames} frames, lighter than baseline)"
            )

    # 4. Apply all adjustments, then clamp to floor/ceiling.
    raw_effective = base_coef - lora_penalty - pass_penalty - compute_penalty
    effective = min(max(raw_effective, _COEFFICIENT_FLOOR), _COEFFICIENT_CEILING)
    floored = raw_effective < _COEFFICIENT_FLOOR
    ceilinged = raw_effective > _COEFFICIENT_CEILING
    if floored:
        reasons.append(
            f"floored at {_COEFFICIENT_FLOOR} "
            f"(would have been {raw_effective:.3f}) — workload may still OOM"
        )
    elif ceilinged:
        reasons.append(
            f"ceilinged at {_COEFFICIENT_CEILING} "
            f"(would have been {raw_effective:.3f})"
        )

    return {
        "effective_coef": effective,
        "base_coef": base_coef,
        "lora_total_gb": lora_total_gb,
        "lora_penalty": lora_penalty,
        "pass_penalty": pass_penalty,
        "compute_penalty": compute_penalty,
        "compute_ratio": compute_ratio,
        "stage_count": stage_count,
        "floored": floored,
        "ceilinged": ceilinged,
        "reasons": reasons,
    }
