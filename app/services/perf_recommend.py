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


def recommend_h3_reserved_ram_fraction(
    requested_fraction: float,
    total_ram_gb: float,
    *,
    full_checkpoint: bool,
) -> float:
    """Raise MMGP's reserved-RAM ceiling only when H3 Full can fit safely.

    H3 Full's complete pinned working set is roughly 53-55 GB. MMGP's
    implicit 40% ceiling leaves a 128 GB workstation just short, forcing
    partial pinning and avoidable CPU/RAM transfers. Reserve a 60 GB ceiling
    on systems with at least 96 GB RAM, while retaining at least 36 GB for the
    OS, decoded media, and transient allocations. This is a ceiling, not an
    eager allocation. Explicit user values always win.
    """
    requested = max(0.0, float(requested_fraction or 0.0))
    ram_gb = max(0.0, float(total_ram_gb or 0.0))
    if requested > 0.0 or not full_checkpoint or ram_gb < 96.0:
        return requested

    target_fraction = 60.0 / ram_gb
    safe_fraction = min(0.65, max(0.50, target_fraction))
    return round(safe_fraction, 3)


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
    # Audio gets its own tiering. The shared profile table is calibrated
    # for 20+ GB VIDEO models, but the whole ACE-Step 1.5 audio stack
    # (XL transformer int8 ~5.3 GB + LM 4B int8 ~4.4 GB + codec) fits in
    # VRAM on much smaller cards — and the fast LM decoders (vllm/cg)
    # only engage when int(profile) is 1 or 3 (wgp.py gate: models
    # resident in VRAM). Inheriting the video profile silently condemned
    # every card under 24 GB to the legacy LM decoder at <1 token/sec,
    # making LM songs take 10-15 minutes and look hung — exactly the
    # hand-tuning Auto-Tune exists to prevent. On 12 GB+ cards pick
    # profile 3 (LowRAM_HighVRAM) for audio; below 12 GB keep the
    # conservative shared profile (the LM stack wouldn't fit anyway).
    if (vram_gb or 0) >= 12 and int(profile) not in (1, 3):
        audio_profile = 3
    else:
        audio_profile = profile

    reason = (
        f"{gpu_name} ({vram_gb} GB VRAM, {ram_gb} GB RAM): "
        f"VRAM tier={vram_tier}, RAM tier={ram_tier} → profile {profile:g}, "
        f"audio profile {audio_profile:g}, "
        f"{quant.upper()}, VAE config {vae}, coefficient {coef}"
    )

    return {
        "video_profile": profile,
        "image_profile": profile,
        "audio_profile": audio_profile,
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

# MiniMax H3 attention uses one packed text + audio + video sequence.  Its
# peak therefore needs substantially more free activation VRAM than the
# generic pixels x frames curve reserves.  The 960x544 / 345-frame baseline
# is the released 540p near-15-second workload.  A clean 24 GB run needs
# roughly 7 GB left outside the streamed transformer; every Ref2VA video adds
# another full packed media sequence and needs about 10 GB at that baseline.
_H3_BASELINE_PIXELS = 960 * 544
_H3_BASELINE_FRAMES = 345
_H3_MIN_ACTIVATION_RESERVE_GB = 5.0
_H3_MAX_WEIGHT_BUDGET_GB = 18.0
_H3_MIN_WEIGHT_BUDGET_GB = 3.5
_H3_VIDEO_REFERENCE_RESERVE_GB = 10.0
# MMGP receives the transformer's fixed ``workingVRAM`` reserve independently,
# but its preload plan is also constrained by the per-job residency
# coefficient. Letting the coefficient retain 17 GB at baseline 540p left
# only 7 GB of actual device headroom and triggered Windows shared-memory
# paging. Blend the measured workspace into that coefficient as packed load
# approaches baseline. Native 768p x 345 frames is almost 2x the baseline
# token load and hard-OOMed a 24 GB 4090 while MMGP kept 13.35 GB of weights
# resident. H3 also routes through the allocation-efficient attention backend
# and bounds projection temporaries.
# A subsequent 1280x704 x 345-frame measurement still filled 24.0/24.6 GB at
# a 9.9 GB transformer cap, before denoising step zero. Preserve enough room
# for that measured full-window peak while shorter recommended windows retain
# more transformer residency and therefore run faster.
# Blend the measured runtime workspace into the residency cap as a request
# approaches the native full-window baseline. The previous 1.10x hard gate
# left only 7 GB outside model residency at exactly 540p / 345 frames, then
# abruptly reserved more than 12 GB immediately above the gate. On Windows,
# that made the smaller 540p job spill into shared GPU memory while 720p ran
# normally. A blended ramp keeps short jobs fast without that inversion.
_H3_RUNTIME_BLEND_START_RATIO = 0.75
_H3_RUNTIME_BLEND_FULL_RATIO = 1.0
_H3_LARGE_CANVAS_MIN_PIXELS = 1_400_000
_H3_NATIVE_RUNTIME_EXCESS_SCALE = 0.70
_H3_NATIVE_RUNTIME_SAFETY_MARGIN_GB = 1.0
_H3_LARGE_CANVAS_RUNTIME_SAFETY_MARGIN_GB = 1.0

# MiniMax-Music3 first runs a 25 Hz autoregressive Qwen stage. Its bf16
# language-model weights are about 16 GB, the RVQ depth decoder is another
# 1.2 GB, and a conditional/unconditional KV cache remains resident for the
# entire requested song. A generic video coefficient cannot see that cache,
# so a 120-second job on a 20 GB card can try to retain too much Qwen and fail
# while MMGP is loading the next block. Keep the geometry here model-free so
# the job planner can reserve the cache before the checkpoint is loaded.
_MUSIC3_QWEN_LAYERS = 36
_MUSIC3_QWEN_KV_HEADS = 8
_MUSIC3_QWEN_HEAD_DIM = 128
_MUSIC3_CACHE_BYTES_PER_VALUE = 2  # bf16
_MUSIC3_CFG_BATCH_SIZE = 2
_MUSIC3_SEMANTIC_FRAMES_PER_SECOND = 25
_MUSIC3_PROMPT_TOKEN_ALLOWANCE = 1200
_MUSIC3_NON_CACHE_RUNTIME_RESERVE_GB = 6.75
_MUSIC3_MAX_WEIGHT_BUDGET_GB = 16.0
_MUSIC3_MIN_WEIGHT_BUDGET_GB = 3.5


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


def compute_h3_weight_budget(
    total_vram_gb: float,
    resolution: Optional[str],
    video_length_frames: Optional[int],
    video_reference_count: int = 0,
    runtime_workspace_gb: float = 0.0,
    additional_reserve_gb: float = 0.0,
) -> dict:
    """Reserve packed-sequence activation memory for a MiniMax H3 job.

    This is intentionally separate from ``compute_per_job_coefficient``:
    H3 is a single-stage pipeline, but treating it like an ordinary
    single-stage video model lets the light-job bonus keep too many weights
    resident.  On a 24 GB RTX 4090, 960x544 x 336 frames then raised the
    model cap to 19.3 GB and the process exited during its first denoising
    step.  The same class of jobs had worked under the old incidental
    two-stage reserve at roughly a 17.5 GB cap.

    ``runtime_workspace_gb`` is MMGP's transformer's measured working-VRAM
    allowance. MMGP receives it independently, while this function also uses
    it to keep the residency coefficient from occupying that same device
    headroom. The allowance blends in near the baseline rather than penalizing
    short previews, then scales to cover larger Q/K/V and residual tensors.
    Native canvases use partial excess scaling to retain useful model
    residency; experimental large canvases retain the stricter measured
    scaling that prevents the observed 1080p OOM.

    Returns the maximum resident-weight budget and the activation reserve
    used to derive it.  MMGP streams weights that do not fit the budget.
    """

    try:
        total_vram_gb = float(total_vram_gb)
    except (TypeError, ValueError):
        total_vram_gb = 0.0
    if total_vram_gb <= 0:
        return {
            "weight_budget_gb": 0.0,
            "activation_reserve_gb": 0.0,
            "compute_ratio": 1.0,
            "video_reference_count": max(0, int(video_reference_count or 0)),
            "runtime_workspace_gb": 0.0,
            "scaled_runtime_workspace_gb": 0.0,
            "runtime_scaling_active": False,
            "runtime_safety_margin_gb": 0.0,
            "additional_reserve_gb": 0.0,
        }

    parsed = _parse_resolution(resolution)
    if parsed:
        width, height = parsed
        pixels = max(1, width * height)
    else:
        pixels = _H3_BASELINE_PIXELS
    try:
        frames = max(1, int(video_length_frames or _H3_BASELINE_FRAMES))
    except (TypeError, ValueError):
        frames = _H3_BASELINE_FRAMES

    compute_ratio = (
        (pixels / float(_H3_BASELINE_PIXELS))
        * (frames / float(_H3_BASELINE_FRAMES))
    )
    # A fixed floor covers Q/K/V, packed hidden states, the audio branch,
    # allocator fragmentation, and VAE handoff.  The square-root term grows
    # the reserve for H3's larger native 768p canvas without over-penalizing
    # short 480p previews.
    base_reserve_gb = _H3_MIN_ACTIVATION_RESERVE_GB + 2.0 * (compute_ratio ** 0.5)
    reference_count = max(0, int(video_reference_count or 0))
    reference_reserve_gb = (
        _H3_VIDEO_REFERENCE_RESERVE_GB * reference_count * compute_ratio
    )
    try:
        runtime_workspace_gb = max(0.0, float(runtime_workspace_gb or 0.0))
    except (TypeError, ValueError):
        runtime_workspace_gb = 0.0
    large_canvas = pixels >= _H3_LARGE_CANVAS_MIN_PIXELS
    runtime_excess_scale = (
        1.0 if large_canvas else _H3_NATIVE_RUNTIME_EXCESS_SCALE
    )
    scaled_runtime_workspace_gb = 0.0
    runtime_blend = 0.0
    if runtime_workspace_gb > 0.0 and compute_ratio > _H3_RUNTIME_BLEND_START_RATIO:
        runtime_blend = min(
            1.0,
            max(
                0.0,
                (compute_ratio - _H3_RUNTIME_BLEND_START_RATIO)
                / (
                    _H3_RUNTIME_BLEND_FULL_RATIO
                    - _H3_RUNTIME_BLEND_START_RATIO
                ),
            ),
        )
        raw_runtime_workspace_gb = runtime_workspace_gb * (
            compute_ratio
            if compute_ratio <= 1.0
            else 1.0 + (compute_ratio - 1.0) * runtime_excess_scale
        )
        if runtime_blend < 1.0:
            # Meet the analytical reserve continuously at the beginning of
            # the ramp instead of introducing another piecewise jump.
            scaled_runtime_workspace_gb = base_reserve_gb + (
                raw_runtime_workspace_gb - base_reserve_gb
            ) * runtime_blend
        else:
            scaled_runtime_workspace_gb = raw_runtime_workspace_gb
    try:
        additional_reserve_gb = max(0.0, float(additional_reserve_gb or 0.0))
    except (TypeError, ValueError):
        additional_reserve_gb = 0.0
    # MMGP budgets against total device memory. Keep an extra allocator and
    # Windows display/browser margin once a pass exceeds H3's measured
    # baseline load. Large experimental canvases retain the full-gigabyte
    # margin; native canvases use a smaller one to preserve useful residency.
    runtime_safety_margin_gb = 0.0
    if scaled_runtime_workspace_gb > 0:
        runtime_safety_margin_gb = runtime_blend * (
            _H3_LARGE_CANVAS_RUNTIME_SAFETY_MARGIN_GB
            if large_canvas
            else _H3_NATIVE_RUNTIME_SAFETY_MARGIN_GB
        )
    # The analytical reserve already includes any reference-video surcharge.
    # Treat the runtime measurement as an alternative lower bound rather than
    # adding both estimates together. At 1080p it tightens model residency in
    # proportion to the larger packed sequence.
    analytical_reserve_gb = base_reserve_gb + reference_reserve_gb
    runtime_reserve_gb = (
        scaled_runtime_workspace_gb + runtime_safety_margin_gb
    )
    runtime_scaling_active = runtime_reserve_gb > analytical_reserve_gb
    requested_reserve_gb = max(
        analytical_reserve_gb,
        runtime_reserve_gb,
    ) + additional_reserve_gb

    # Always leave enough room to stream at least a small transformer slice.
    max_reserve_gb = max(0.0, total_vram_gb - _H3_MIN_WEIGHT_BUDGET_GB)
    activation_reserve_gb = min(requested_reserve_gb, max_reserve_gb)
    weight_budget_gb = min(
        _H3_MAX_WEIGHT_BUDGET_GB,
        max(_H3_MIN_WEIGHT_BUDGET_GB, total_vram_gb - activation_reserve_gb),
    )
    return {
        "weight_budget_gb": weight_budget_gb,
        "activation_reserve_gb": activation_reserve_gb,
        "compute_ratio": compute_ratio,
        "video_reference_count": reference_count,
        "runtime_workspace_gb": runtime_workspace_gb,
        "scaled_runtime_workspace_gb": scaled_runtime_workspace_gb,
        "runtime_scaling_active": runtime_scaling_active,
        "runtime_safety_margin_gb": runtime_safety_margin_gb,
        "additional_reserve_gb": additional_reserve_gb,
    }


def compute_music3_weight_budget(
    total_vram_gb: float,
    duration_seconds: float,
    prompt_token_allowance: int = _MUSIC3_PROMPT_TOKEN_ALLOWANCE,
) -> dict:
    """Reserve Music3 semantic-cache and runtime VRAM before model loading.

    Music3 uses classifier-free guidance during its autoregressive Qwen pass,
    so every prompt and generated 25 Hz semantic token retains key/value
    tensors for a batch of two across all 36 layers. The fixed reserve also
    covers the co-resident RVQ depth decoder, Qwen workspaces, MMGP's async
    shuttle, allocator fragmentation, and display/driver headroom.

    The returned weight budget is an absolute MMGP residency cap. Weights
    beyond it are streamed from reserved RAM; later Music3 stages are smaller
    than Qwen and therefore remain unaffected by the cap.
    """

    try:
        total_vram_gb = max(0.0, float(total_vram_gb or 0.0))
    except (TypeError, ValueError):
        total_vram_gb = 0.0
    try:
        duration_seconds = min(300.0, max(5.0, float(duration_seconds or 120.0)))
    except (TypeError, ValueError):
        duration_seconds = 120.0
    try:
        prompt_tokens = max(1, int(prompt_token_allowance or 0))
    except (TypeError, ValueError):
        prompt_tokens = _MUSIC3_PROMPT_TOKEN_ALLOWANCE

    semantic_tokens = max(
        1,
        int(duration_seconds * _MUSIC3_SEMANTIC_FRAMES_PER_SECOND),
    )
    cache_bytes = (
        2  # key and value
        * _MUSIC3_QWEN_LAYERS
        * _MUSIC3_QWEN_KV_HEADS
        * _MUSIC3_QWEN_HEAD_DIM
        * _MUSIC3_CACHE_BYTES_PER_VALUE
        * _MUSIC3_CFG_BATCH_SIZE
        * (prompt_tokens + semantic_tokens)
    )
    kv_cache_gb = cache_bytes / float(1024**3)
    runtime_reserve_gb = _MUSIC3_NON_CACHE_RUNTIME_RESERVE_GB + kv_cache_gb

    if total_vram_gb <= 0.0:
        weight_budget_gb = 0.0
    else:
        max_reserve_gb = max(
            0.0,
            total_vram_gb - _MUSIC3_MIN_WEIGHT_BUDGET_GB,
        )
        runtime_reserve_gb = min(runtime_reserve_gb, max_reserve_gb)
        weight_budget_gb = min(
            _MUSIC3_MAX_WEIGHT_BUDGET_GB,
            max(
                _MUSIC3_MIN_WEIGHT_BUDGET_GB,
                total_vram_gb - runtime_reserve_gb,
            ),
        )

    return {
        "weight_budget_gb": weight_budget_gb,
        "runtime_reserve_gb": runtime_reserve_gb,
        "kv_cache_gb": kv_cache_gb,
        "duration_seconds": duration_seconds,
        "prompt_token_allowance": prompt_tokens,
    }


def compute_per_job_coefficient(
    base_coef: float,
    total_vram_gb: float,
    active_loras: Optional[list] = None,
    lora_dir: Optional[str] = None,
    stage_count: int = 1,
    resolution: Optional[str] = None,
    video_length_frames: Optional[int] = None,
    model_activation_gb: float = 0.0,
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
        model_activation_gb: Flat activation surcharge in GB for model
            classes whose conditioning inflates the attention sequence
            beyond what resolution × frames predicts (e.g. SCAIL-2
            appends the driving video as in-context tokens and prepends
            reference latents — ~35% more sequence than a classic Wan
            job at the same size). The generic compute curve rates such
            jobs "lighter than baseline" and would loosen the cap; this
            surcharge tightens it instead so activations fit. 0 = no-op.

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

    # 3b. Model-class activation surcharge — see docstring. Sized by the
    # caller from measurement; applied as a straight cap reduction.
    model_activation_penalty = 0.0
    if model_activation_gb and model_activation_gb > 0:
        model_activation_penalty = float(model_activation_gb) / total_vram_gb
        reasons.append(
            f"-{model_activation_penalty:.3f} for model-class activation "
            f"overhead ({model_activation_gb:.1f} GB)"
        )

    # 4. Apply all adjustments, then clamp to floor/ceiling.
    raw_effective = base_coef - lora_penalty - pass_penalty - compute_penalty - model_activation_penalty
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
