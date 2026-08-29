"""Maestro family handler for MiniMax H3 Base FL2VA and Ref2VA."""

from __future__ import annotations

import os


_MODEL_TYPE = "minimax_h3"
_REF2VA_MODEL_TYPE = "minimax_h3_ref2va"
_FULL_MODEL_TYPE = "minimax_h3_full"
_REF2VA_FULL_MODEL_TYPE = "minimax_h3_ref2va_full"
_COMFY_REPO = "Comfy-Org/MiniMax-H3"
_COMFY_REVISION = "0543966fbdce5ba05709a8f2031c94bdba629b4a"
_OFFICIAL_REPO = "MiniMaxAI/MiniMax-H3"
_OFFICIAL_REVISION = "5d9b308a59ab12e67147f191e184baf704185bd1"
_DEEPBEEP_REPO = "DeepBeepMeep/MiniMax-H3"
_DEEPBEEP_REVISION = "fec7846aef352e58a1cfb699455e3d104281e68b"
_ASSETS_ROOT = "minimax_h3"

_TRANSFORMER = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
_REF2VA_TRANSFORMER = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
_TEXT_ENCODER_BF16 = "Qwen3-VL-32B-Instruct-layer50_bf16.safetensors"
_TEXT_ENCODER_INT8 = "Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"
_TEXT_ENCODER_GGUF_Q2 = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
_TEXT_ENCODER_GGUF_Q4 = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
_WANGP_TEXT_ENCODER_FOLDER = "Qwen3-VL-32B-Instruct"
_WANGP_FL2VA_PRUNED_TRANSFORMER = (
    "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors"
)
_WANGP_REF2VA_PRUNED_TRANSFORMER = (
    "MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors"
)
_WANGP_FL2VA_PRUNED_BF16_TRANSFORMER = (
    "MiniMax-H3-FL2VA-pruned_rank8_bf16.safetensors"
)
_WANGP_REF2VA_PRUNED_BF16_TRANSFORMER = (
    "MiniMax-H3-Ref2VA-pruned_rank8_bf16.safetensors"
)

# H3 packs video, audio, and text into one unusually long transformer
# sequence.  At 480p / 10 seconds the token-wise activations alone need
# several gigabytes, so MMGP must not treat its model-weight safety cap as
# the entire available VRAM budget.  ``workingVRAM`` reserves this amount
# independently of the user's card size; MMGP streams more transformer
# blocks on smaller cards instead of starving the first denoising step.
_TRANSFORMER_WORKING_VRAM_MB = 10 * 1024

# H3's video VAE accepts 17*n+5 pixel frames. 345 is the final valid
# frame count at or below the official 15-second limit (14.375s at 24fps).
# First/Last may continue beyond that duration, but every individual model
# pass remains inside this native limit. Continuation uses a 17*n+1 overlap:
# complete 17-frame chunks carry motion history and the final frame is the
# ordinary FL2VA boundary anchor.
_H3_MIN_FRAMES = 124
_H3_MAX_FRAMES = 345
_H3_FRAME_STEP = 17
_H3_OVERLAP_DEFAULT = 18
_H3_OVERLAP_MAX = 103
# Ref2VA appends its ordered reference context to the target sequence. Keep
# Auto sequence clips one legal H3 frame step below the ordinary FL2VA pass
# recommendation so reference conditioning has a small activation cushion.
# Expert users can lock Sequence Window Length to reclaim the native ceiling.
_H3_OMNI_REFERENCE_MARGIN_STEPS = 1
_H3_SLIDING_WINDOW_DEFAULTS = {
    "window_min": _H3_MIN_FRAMES,
    "window_max": _H3_MAX_FRAMES,
    "window_step": _H3_FRAME_STEP,
    "window_default": _H3_MAX_FRAMES,
    "overlap_min": 1,
    "overlap_max": _H3_OVERLAP_MAX,
    "overlap_step": _H3_FRAME_STEP,
    "overlap_offset": 1,
    "overlap_default": _H3_OVERLAP_DEFAULT,
    "discard_last_frames": 0,
}

# Opt-in diagnostics for investigating very long FL2VA continuations. These
# live in ``custom_settings`` so they remain out of the ordinary H3 workflow,
# survive presets/metadata, and can be removed without expanding the shared
# WanGP generation signature. None of them changes the default pipeline.
_H3_LONG_SEQUENCE_RUNTIME_SETTINGS = (
    "h3_long_sequence_clean_tail",
    "h3_long_sequence_single_frame_after_three",
    "h3_long_sequence_vary_seed",
    "h3_long_sequence_periodic_reset",
    "h3_long_sequence_diagnostics",
)
_H3_LONG_SEQUENCE_CLEAN_TAIL_FRAMES = _H3_FRAME_STEP
_H3_LONG_SEQUENCE_SEED_STRIDE = 1_000_003


def resolve_h3_long_sequence_discard_frames(
    custom_settings,
    *,
    enabled: bool,
) -> int:
    """Return the opt-in clean-tail discard for a rolling FL2VA sequence."""

    settings = custom_settings if isinstance(custom_settings, dict) else {}
    if enabled and settings.get("h3_long_sequence_clean_tail") is True:
        return _H3_LONG_SEQUENCE_CLEAN_TAIL_FRAMES
    return 0


def resolve_h3_long_sequence_window_policy(
    window_no: int,
    overlap_frames: int,
    seed,
    custom_settings,
):
    """Resolve one experimental FL2VA continuation handoff.

    The scheduler's output stride remains based on ``overlap_frames``. A
    one-frame handoff therefore uses the remaining overlap as generated
    warm-up that the outer assembler trims, preserving the exact requested
    total duration while removing recursive multi-frame history.
    """

    settings = custom_settings if isinstance(custom_settings, dict) else {}
    window = max(1, int(window_no or 1))
    overlap = max(0, int(overlap_frames or 0))
    is_continuation = window > 1 and overlap > 0
    persistent_single_frame = (
        is_continuation
        and window >= 4
        and settings.get(
            "h3_long_sequence_single_frame_after_three"
        ) is True
    )
    periodic_reset = (
        is_continuation
        and (window - 1) % 3 == 0
        and settings.get("h3_long_sequence_periodic_reset") is True
    )
    conditioning_frames = (
        1
        if persistent_single_frame or periodic_reset
        else overlap
    )
    effective_seed = seed
    if (
        settings.get("h3_long_sequence_vary_seed") is True
        and window > 1
        and seed is not None
    ):
        effective_seed = (
            int(seed) + (window - 1) * _H3_LONG_SEQUENCE_SEED_STRIDE
        ) % 2_147_483_647
    return {
        "conditioning_frames": conditioning_frames,
        "output_trim_frames": overlap,
        "seed": effective_seed,
        "persistent_single_frame": persistent_single_frame,
        "periodic_reset": periodic_reset,
        "diagnostics": settings.get(
            "h3_long_sequence_diagnostics"
        ) is True,
    }


def align_h3_num_frames(num_frames: int) -> int:
    """Snap a request to the next ``17 * n + 5`` H3 frame count."""

    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % _H3_FRAME_STEP != 5:
        num_frames += 1
    return num_frames


def normalize_h3_clip_frame_count(
    value,
    *,
    minimum_frames: int = _H3_MIN_FRAMES,
    maximum_frames: int = _H3_MAX_FRAMES,
    frame_step: int = _H3_FRAME_STEP,
) -> int:
    """Repair one bounded H3 clip onto its native frame lattice.

    Uploaded media and legacy Director metadata commonly describe duration in
    ordinary seconds. At H3's 24 fps, an exact five-second clip becomes 120
    frames even though the model's first legal duration is 124 frames. Round
    upward so source audio/video is never shortened merely to satisfy the
    lattice, and clamp to the model's published one-pass bounds.
    """

    minimum = max(1, int(minimum_frames or _H3_MIN_FRAMES))
    step = max(1, int(frame_step or _H3_FRAME_STEP))
    maximum = max(minimum, int(maximum_frames or _H3_MAX_FRAMES))
    # A hardware/profile ceiling may be expressed as an ordinary frame count.
    # Keep the returned value legal even when that ceiling is not itself one
    # of H3's ``minimum + n * step`` values.
    maximum = minimum + ((maximum - minimum) // step) * step
    try:
        requested = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        requested = minimum
    if requested <= minimum:
        return minimum
    lattice_steps = (requested - minimum + step - 1) // step
    return min(maximum, minimum + lattice_steps * step)


def normalize_h3_clip_frame_schedule(
    values,
    *,
    minimum_frames: int = _H3_MIN_FRAMES,
    maximum_frames: int = _H3_MAX_FRAMES,
    frame_step: int = _H3_FRAME_STEP,
) -> list[int]:
    """Repair an H3 clip schedule while preserving cumulative media timing.

    A single clip rounds upward so it cannot lose source content. For a series
    of audio/video-derived clips, carry each rounding residual into the next
    choice instead; this prevents a small lattice adjustment from accumulating
    into visible A/V drift across a long sequence.
    """

    minimum = max(1, int(minimum_frames or _H3_MIN_FRAMES))
    step = max(1, int(frame_step or _H3_FRAME_STEP))
    maximum = max(minimum, int(maximum_frames or _H3_MAX_FRAMES))
    maximum = minimum + ((maximum - minimum) // step) * step
    valid = list(range(minimum, maximum + 1, step))
    residual = 0.0
    schedule: list[int] = []
    for value in (values or []):
        try:
            requested = int(round(float(value)))
        except (TypeError, ValueError, OverflowError):
            requested = minimum
        target = max(1.0, float(requested)) + residual
        chosen = min(valid, key=lambda candidate: (abs(candidate - target), candidate))
        residual = target - chosen
        schedule.append(chosen)
    return schedule


def normalize_h3_overlap_frames(value, *, window_frames=None) -> int:
    """Round and safely cap an FL2VA overlap on the 17*n+1 lattice."""

    try:
        value = int(value)
    except (TypeError, ValueError):
        value = _H3_OVERLAP_DEFAULT
    if value <= 0:
        value = 1
    value = (
        ((value - 1 + _H3_FRAME_STEP // 2) // _H3_FRAME_STEP)
        * _H3_FRAME_STEP
        + 1
    )
    maximum = _H3_OVERLAP_MAX
    if window_frames is not None:
        available = max(1, int(window_frames) - _H3_FRAME_STEP)
        maximum = min(
            maximum,
            ((available - 1) // _H3_FRAME_STEP) * _H3_FRAME_STEP + 1,
        )
    return max(1, min(maximum, value))

# First Block Cache is intentionally opt-in. It compares a compact signature
# from block one and can reuse the remaining 49-block residual when adjacent
# scheduler steps are sufficiently similar. These are the thresholds exposed
# by WanGP's current H3 runtime; larger values skip more work with a greater
# chance of changing motion or fine detail.
_FIRST_BLOCK_CACHE_THRESHOLDS = (0.06, 0.08, 0.10, 0.12, 0.14)
_LEGACY_FIRST_BLOCK_CACHE_THRESHOLDS = {
    1.5: 0.06,
    1.75: 0.08,
    2.0: 0.10,
    2.25: 0.12,
    2.5: 0.14,
}
_FIRST_BLOCK_CACHE_STRENGTHS = [
    ("Low (0.06)", 0.06),
    ("Balanced (0.08)", 0.08),
    ("High (0.10)", 0.10),
    ("Very High (0.12)", 0.12),
    ("Maximum (0.14)", 0.14),
]


def _hf_url(repo_id: str, revision: str, *parts: str) -> str:
    path = "/".join(part.strip("/\\") for part in parts if part)
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{path}"


def _text_encoder_variants() -> dict[str, dict]:
    deepbeep_folder = "Qwen3-VL-32B-Instruct"
    return {
        "nvfp4_awq": {
            "name": "NVFP4 AWQ (Recommended)",
            "size_hint": (
                "~15.7 GB download · native acceleration requires an RTX 50-series "
                "GPU; RTX 40 and older use Maestro's proven compatibility fallback"
            ),
            "URLs": [
                _hf_url(
                    _COMFY_REPO,
                    _COMFY_REVISION,
                    "text_encoders",
                    _TEXT_ENCODER,
                )
            ],
        },
        "gguf_q2_k": {
            "name": "GGUF Q2_K (Lowest RAM)",
            "size_hint": "~8.5 GB download · lowest system-memory use",
            "URLs": [
                _hf_url(
                    _DEEPBEEP_REPO,
                    _DEEPBEEP_REVISION,
                    deepbeep_folder,
                    _TEXT_ENCODER_GGUF_Q2,
                )
            ],
        },
        "gguf_q4_k_m": {
            "name": "GGUF Q4_K_M",
            "size_hint": "~14.6 GB download · balanced quality and system-memory use",
            "URLs": [
                _hf_url(
                    _DEEPBEEP_REPO,
                    _DEEPBEEP_REVISION,
                    deepbeep_folder,
                    _TEXT_ENCODER_GGUF_Q4,
                )
            ],
        },
        "int8": {
            "name": "Quanto INT8",
            "size_hint": "~26.7 GB download · optional high-fidelity encoder with high system-memory use",
            "URLs": [
                _hf_url(
                    _DEEPBEEP_REPO,
                    _DEEPBEEP_REVISION,
                    deepbeep_folder,
                    _TEXT_ENCODER_INT8,
                )
            ],
        },
        "bf16": {
            "name": "BF16 (Maximum Fidelity)",
            "size_hint": "~51.5 GB download · maximum fidelity and very high system-memory use",
            "URLs": [
                _hf_url(
                    _DEEPBEEP_REPO,
                    _DEEPBEEP_REVISION,
                    deepbeep_folder,
                    _TEXT_ENCODER_BF16,
                )
            ],
        },
    }


def _recommend_text_encoder(hardware: dict | None, available=None) -> str:
    """Choose a proven encoder whose format fits system RAM."""

    choices = set(available or _text_encoder_variants())
    hardware = hardware or {}
    try:
        ram_gb = float(hardware.get("ram_gb") or 0)
    except (TypeError, ValueError):
        ram_gb = 0
    try:
        vram_gb = float(hardware.get("gpu_vram_gb") or 0)
    except (TypeError, ValueError):
        vram_gb = 0
    # The Comfy NVFP4-AWQ conditioner is Maestro's known-good H3 path.  RTX
    # 50 cards execute it natively. Older NVIDIA cards use compatibility
    # kernels, which work but are not the lowest-memory option. On 16 GB GPUs,
    # leave more headroom for H3's packed video/audio attention by preferring
    # the Q2 encoder even when the machine has abundant system RAM.
    if "nvfp4_awq" in choices and hardware.get("supports_nvfp4"):
        return "nvfp4_awq"
    if 0 < vram_gb <= 16 and "gguf_q2_k" in choices:
        return "gguf_q2_k"
    if "nvfp4_awq" in choices and ram_gb >= 24:
        return "nvfp4_awq"
    if ram_gb >= 56 and "int8" in choices:
        return "int8"
    if ram_gb >= 24 and "gguf_q4_k_m" in choices:
        return "gguf_q4_k_m"
    if "gguf_q2_k" in choices:
        return "gguf_q2_k"
    if "nvfp4_awq" in choices:
        return "nvfp4_awq"
    return next(iter(choices), "nvfp4_awq")


_H3_RESOLUTION_PRESETS = {
    "480p": {
        "label": "480p",
        "values": {
            "auto": "auto_480p",
            "21:9": "1120x480",
            "16:9": "864x480",
            "9:16": "480x864",
            "1:1": "640x640",
            "4:3": "640x480",
            "3:4": "480x640",
        },
    },
    "540p": {
        "label": "540p",
        "values": {
            "auto": "auto_540p",
            "21:9": "1280x544",
            "16:9": "960x544",
            "9:16": "544x960",
            "1:1": "736x736",
            "4:3": "736x544",
            "3:4": "544x736",
        },
    },
    # Consumer-friendly 720p tier. H3 requires multiples of 32, so 704 is
    # the nearest lower aligned short edge to broadcast 720p. This canvas is
    # 12.7% smaller than the released 1344x768 tier and materially reduces
    # the packed attention sequence while retaining a familiar UI label.
    "720p": {
        "label": "720p",
        "hint": "Uses a model-aligned 1280x704 canvas for faster H3 generation.",
        "values": {
            "auto": "auto_720p",
            "21:9": "1632x704",
            "16:9": "1280x704",
            "9:16": "704x1280",
            "1:1": "704x704",
            "4:3": "928x704",
            "3:4": "704x928",
        },
    },
    # H3's native trained tier uses a 768px short edge. Keep it distinct from
    # the lighter aligned 720p canvas so users can choose the model's native
    # resolution without jumping all the way to experimental 1080p.
    "768p": {
        "label": "768p",
        "hint": (
            "MiniMax H3's native trained resolution (1344x768 at 16:9). "
            "It uses 14.5% more pixels than 720p and may require a shorter "
            "window on lower-VRAM GPUs."
        ),
        "values": {
            "auto": "auto_768p",
            "21:9": "1792x768",
            "16:9": "1344x768",
            "9:16": "768x1344",
            "1:1": "768x768",
            "4:3": "1024x768",
            "3:4": "768x1024",
        },
    },
    # The released local workflow is tuned for a 768px short edge, but the
    # transformer accepts larger multiple-of-32 canvases and Maestro users
    # successfully used the shared 1080p preset before H3 gained a dedicated
    # resolution menu. Keep the native tier recommended and make this opt-in
    # rather than silently rewriting an existing high-resolution request.
    "1080p": {
        "label": "1080p",
        "experimental": True,
        "hint": (
            "Experimental high-resolution H3 generation. It uses more than "
            "twice the pixels of 720p; shorter First / Last windows or 720p + "
            "upscaling are recommended on consumer GPUs."
        ),
        "values": {
            "auto": "auto_1080p",
            "21:9": "2528x1088",
            "16:9": "1920x1088",
            "9:16": "1088x1920",
            "1:1": "1088x1088",
            "4:3": "1440x1088",
            "3:4": "1088x1440",
        },
    },
}
_H3_RESOLUTION_PRESET_ORDER = ["480p", "540p", "720p", "768p", "1080p"]
_H3_AUTO_RESOLUTION_BUDGETS = {
    "auto": 1280 * 704,
    "auto_480p": 864 * 480,
    "auto_540p": 960 * 544,
    "auto_720p": 1280 * 704,
    "auto_768p": 1344 * 768,
    "auto_1080p": 1920 * 1088,
}
_H3_AUTO_RESOLUTION_FALLBACKS = {
    "auto": "1280x704",
    "auto_480p": "864x480",
    "auto_540p": "960x544",
    "auto_720p": "1280x704",
    "auto_768p": "1344x768",
    "auto_1080p": "1920x1088",
}

# Shared with the Studio UI through /api/v1/model-options. The backend
# remains authoritative, while the UI can display and select the same safe
# pass length before a job is submitted. Full 33B and Pruned 20B do not have
# the same peak-memory shape: Full streams more weights, while Pruned uses a
# fused QKV projection. Keep checkpoint-specific curves, but calibrate them
# against the current H3 residency cap rather than older pre-cap OOM results.
# On a 24 GB RTX 4090, both checkpoints now complete a 345-frame 1280x704
# pass while the cap preserves roughly 16.1 GB for packed-sequence workspace.
# Bands are evaluated top-down; VRAM tiers are evaluated left-to-right.
_H3_FULL_WINDOW_MEMORY_POLICY = {
    "checkpoint": "full",
    "manual_override": True,
    "auto_resolution_pixels": dict(_H3_AUTO_RESOLUTION_BUDGETS),
    "resolution_bands": [
        {
            "min_pixels": 1_800_000,
            "vram_tiers": [
                {
                    "max_vram_gb": 8,
                    "frames": None,
                    "fallback_resolution": "480p",
                },
                {
                    "max_vram_gb": 12,
                    "frames": None,
                    "fallback_resolution": "540p or lower",
                },
                {
                    "max_vram_gb": 16,
                    "frames": None,
                    "fallback_resolution": "720p or lower",
                },
                # A 24 GB RTX 4090 sustained a 243-frame 1088x1728 pass,
                # but peaked close enough to the VRAM ceiling that it is an
                # expert override rather than a portable default. Recommend
                # 192 frames (exactly 8.0s) for the full 1080p canvas so
                # display use, First Block Cache, and allocator variation
                # retain useful headroom.
                {"max_vram_gb": 24, "frames": 192},
                {"max_vram_gb": 32, "frames": 192},
                {"max_vram_gb": 40, "frames": 243},
                {"frames": 345},
            ],
        },
        {
            "min_pixels": 1_000_000,
            "vram_tiers": [
                {
                    "max_vram_gb": 8,
                    "frames": None,
                    "fallback_resolution": "480p",
                },
                {
                    "max_vram_gb": 12,
                    "frames": None,
                    "fallback_resolution": "540p or lower",
                },
                {"max_vram_gb": 16, "frames": 124},
                {"max_vram_gb": 24, "frames": 243},
                {"frames": 345},
            ],
        },
        {
            "min_pixels": 500_000,
            "vram_tiers": [
                {
                    "max_vram_gb": 8,
                    "frames": None,
                    "fallback_resolution": "480p",
                },
                {"max_vram_gb": 12, "frames": 124},
                {"max_vram_gb": 16, "frames": 243},
                {"frames": 345},
            ],
        },
        {
            "min_pixels": 0,
            "vram_tiers": [
                {"max_vram_gb": 8, "frames": 124},
                {"max_vram_gb": 12, "frames": 243},
                {"frames": 345},
            ],
        },
    ],
}

_H3_PRUNED_WINDOW_MEMORY_POLICY = {
    "checkpoint": "pruned",
    "manual_override": True,
    "auto_resolution_pixels": dict(_H3_AUTO_RESOLUTION_BUDGETS),
    "resolution_bands": [
        {
            "min_pixels": 1_800_000,
            "vram_tiers": [
                {
                    "max_vram_gb": 8,
                    "frames": None,
                    "fallback_resolution": "480p",
                },
                {
                    "max_vram_gb": 12,
                    "frames": None,
                    "fallback_resolution": "720p or lower",
                },
                {"max_vram_gb": 16, "frames": 124},
                # A 1920x1088, 158-frame Pruned FL2VA pass completed on a
                # 24 GB RTX 4090 with allocator headroom. The next lattice
                # point (175) reaches the transformer's streaming floor and
                # remains intentionally unadvertised until measured.
                {"max_vram_gb": 24, "frames": 158},
                {"max_vram_gb": 32, "frames": 243},
                {"frames": 345},
            ],
        },
        {
            "min_pixels": 1_000_000,
            "vram_tiers": [
                {
                    "max_vram_gb": 8,
                    "frames": None,
                    "fallback_resolution": "480p",
                },
                {"max_vram_gb": 12, "frames": 124},
                {"max_vram_gb": 16, "frames": 124},
                {"max_vram_gb": 24, "frames": 243},
                {"frames": 345},
            ],
        },
        {
            "min_pixels": 800_000,
            "vram_tiers": [
                {"max_vram_gb": 8, "frames": 124},
                {"max_vram_gb": 12, "frames": 124},
                # Keep unmeasured 16-23 GB cards on the established 243-frame
                # recommendation; the new evidence is specifically 24 GB.
                {"max_vram_gb": 23, "frames": 243},
                # Revalidated after the H3-specific transformer-residency
                # cap landed: 1280x704 x 345 completed in 14m16s on a 24 GB
                # RTX 4090, preserving the same 16.1 GB workspace budget as
                # the Full checkpoint's successful 345-frame pass.
                {"max_vram_gb": 24, "frames": 345},
                {"frames": 345},
            ],
        },
        {
            "min_pixels": 500_000,
            "vram_tiers": [
                {"max_vram_gb": 8, "frames": 124},
                {"max_vram_gb": 12, "frames": 243},
                {"frames": 345},
            ],
        },
        {
            "min_pixels": 0,
            "vram_tiers": [
                {"max_vram_gb": 8, "frames": 243},
                {"frames": 345},
            ],
        },
    ],
}

# Full H3's transformer, text/vision encoders, video/audio VAEs, and managed
# Turbo adapter total roughly 54 GB before normal OS/application headroom.
# This is a preflight recommendation, not a hard gate: MMGP can still stream
# a Full job on other configurations, and expert users remain free to try it.
_H3_FULL_ESTIMATED_PIPELINE_RAM_GB = 54.0
_H3_FULL_MINIMUM_SYSTEM_RAM_GB = 64.0

_RESOLUTIONS = [
    ("2528x1088 (21:9 experimental)", "2528x1088"),
    ("1920x1088 (16:9 experimental)", "1920x1088"),
    ("1088x1920 (9:16 experimental)", "1088x1920"),
    ("1440x1088 (4:3 experimental)", "1440x1088"),
    ("1088x1440 (3:4 experimental)", "1088x1440"),
    ("1088x1088 (1:1 experimental)", "1088x1088"),
    ("1792x768 (21:9 native)", "1792x768"),
    ("1344x768 (16:9 high)", "1344x768"),
    ("768x1344 (9:16 high)", "768x1344"),
    ("1024x768 (4:3 high)", "1024x768"),
    ("768x1024 (3:4 high)", "768x1024"),
    ("768x768 (1:1 high)", "768x768"),
    ("1632x704 (21:9 720p)", "1632x704"),
    ("1280x704 (16:9 720p)", "1280x704"),
    ("704x1280 (9:16 720p)", "704x1280"),
    ("928x704 (4:3 720p)", "928x704"),
    ("704x928 (3:4 720p)", "704x928"),
    ("704x704 (1:1 720p)", "704x704"),
    ("1152x640 (16:9)", "1152x640"),
    ("640x1152 (9:16)", "640x1152"),
    ("1280x544 (21:9)", "1280x544"),
    ("960x544 (16:9)", "960x544"),
    ("544x960 (9:16)", "544x960"),
    ("736x544 (4:3)", "736x544"),
    ("544x736 (3:4)", "544x736"),
    ("736x736 (1:1)", "736x736"),
    ("1120x480 (21:9 low VRAM)", "1120x480"),
    ("864x480 (16:9 low VRAM)", "864x480"),
    ("480x864 (9:16 low VRAM)", "480x864"),
    ("640x480 (4:3 low VRAM)", "640x480"),
    ("480x640 (3:4 low VRAM)", "480x640"),
    ("640x640 (1:1 low VRAM)", "640x640"),
    ("608x352 (16:9 minimum)", "608x352"),
    ("352x608 (9:16 minimum)", "352x608"),
]

_LEGACY_RESOLUTION_ALIASES = {
    "848x480": "864x480",
    "480x848": "480x864",
    "672x672": "640x640",
    "832x608": "736x544",
    "608x832": "544x736",
    "1280x720": "1280x704",
    "720x1280": "704x1280",
    "1024x1024": "768x768",
    "1104x832": "1024x768",
    "832x1104": "768x1024",
}


def _normalize_h3_resolution(value) -> str:
    """Preserve requested orientation while snapping old presets to H3."""

    resolution = str(value or "864x480").strip().lower()
    if resolution in _H3_AUTO_RESOLUTION_BUDGETS:
        return resolution
    if resolution in _LEGACY_RESOLUTION_ALIASES:
        return _LEGACY_RESOLUTION_ALIASES[resolution]

    supported = [item for _, item in _RESOLUTIONS]
    if resolution in supported:
        return resolution
    try:
        width_text, height_text = resolution.split("x", 1)
        width, height = int(width_text), int(height_text)
        if width <= 0 or height <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return "864x480"

    orientation = 0 if width == height else (1 if width > height else -1)
    candidates = []
    for candidate in supported:
        candidate_width, candidate_height = (int(part) for part in candidate.split("x", 1))
        candidate_orientation = (
            0
            if candidate_width == candidate_height
            else (1 if candidate_width > candidate_height else -1)
        )
        if candidate_orientation == orientation:
            candidates.append((candidate, candidate_width, candidate_height))
    if not candidates:
        return "864x480"

    target_aspect = width / height
    target_area = width * height

    def score(item):
        _, candidate_width, candidate_height = item
        aspect_error = abs((candidate_width / candidate_height) - target_aspect) / target_aspect
        area_error = abs((candidate_width * candidate_height) - target_area) / target_area
        return aspect_error * 8 + area_error

    return min(candidates, key=score)[0]


def _h3_resolution_pixels(value) -> int:
    resolution = _normalize_h3_resolution(value)
    if resolution in _H3_AUTO_RESOLUTION_FALLBACKS:
        resolution = _H3_AUTO_RESOLUTION_FALLBACKS[resolution]
    try:
        width, height = (int(part) for part in resolution.split("x", 1))
    except (TypeError, ValueError):
        return 864 * 480
    return max(1, width * height)


def recommended_h3_window_profile(
    total_vram_gb,
    resolution,
    model_def: dict | None = None,
) -> dict:
    """Return the checkpoint-aware native H3 pass profile for this GPU/canvas.

    ``frames=None`` is intentional: H3 cannot use a pass shorter than 124
    frames (about 5.2 seconds), so a low-VRAM/high-resolution combination may
    have no honest automatic window recommendation. In that case the caller
    should recommend the supplied lower resolution instead of presenting the
    model's minimum legal window as though it were memory-safe.
    """

    model_def = model_def or {}
    full_checkpoint = bool(model_def.get("minimax_h3_full_checkpoint", False))
    policy = (
        _H3_FULL_WINDOW_MEMORY_POLICY
        if full_checkpoint
        else _H3_PRUNED_WINDOW_MEMORY_POLICY
    )
    checkpoint = "full" if full_checkpoint else "pruned"

    try:
        total_vram_gb = float(total_vram_gb)
    except (TypeError, ValueError):
        total_vram_gb = 0.0
    normalized_resolution = _normalize_h3_resolution(resolution)
    effective_resolution = _H3_AUTO_RESOLUTION_FALLBACKS.get(
        normalized_resolution,
        normalized_resolution,
    )
    pixels = _h3_resolution_pixels(normalized_resolution)
    if total_vram_gb <= 0:
        return {
            "supported": True,
            "frames": _H3_MAX_FRAMES,
            "fallback_resolution": None,
            "gpu_vram_gb": total_vram_gb,
            "resolution": effective_resolution,
            "pixels": pixels,
            "checkpoint": checkpoint,
        }

    for band in policy["resolution_bands"]:
        if pixels < int(band.get("min_pixels", 0)):
            continue
        for tier in band.get("vram_tiers", []):
            maximum_vram = tier.get("max_vram_gb")
            if maximum_vram is None or total_vram_gb <= float(maximum_vram):
                frames = tier.get("frames")
                frames = int(frames) if frames is not None else None
                return {
                    "supported": frames is not None and frames > 0,
                    "frames": frames,
                    "fallback_resolution": tier.get("fallback_resolution"),
                    "gpu_vram_gb": total_vram_gb,
                    "resolution": effective_resolution,
                    "pixels": pixels,
                    "checkpoint": checkpoint,
                }
        break
    return {
        "supported": True,
        "frames": _H3_MAX_FRAMES,
        "fallback_resolution": None,
        "gpu_vram_gb": total_vram_gb,
        "resolution": effective_resolution,
        "pixels": pixels,
        "checkpoint": checkpoint,
    }


def recommended_h3_window_frames(
    total_vram_gb,
    resolution,
    model_def: dict | None = None,
) -> int:
    """Return the automatic H3 pass size, or zero when none is safe."""

    profile = recommended_h3_window_profile(
        total_vram_gb,
        resolution,
        model_def,
    )
    return int(profile.get("frames") or 0)


def recommended_h3_omni_sequence_profile(
    total_vram_gb,
    resolution,
    model_def: dict | None = None,
) -> dict:
    """Return the safe native Ref2VA pass size for an Omni sequence.

    Ref2VA packs canonical reference media beside every target pass. Auto
    therefore reserves one legal frame step as lightweight reference-context
    headroom while preserving H3's minimum; native continuation only changes
    how completed passes exchange their motion and audio tails.
    """

    model_def = model_def or {}
    profile = dict(
        recommended_h3_window_profile(
            total_vram_gb,
            resolution,
            model_def,
        )
    )
    profile["base_frames"] = profile.get("frames")
    profile["reference_margin_frames"] = 0
    if not profile.get("supported") or not profile.get("frames"):
        return profile

    minimum = max(1, int(model_def.get("frames_minimum") or _H3_MIN_FRAMES))
    maximum = max(minimum, int(model_def.get("frames_maximum") or _H3_MAX_FRAMES))
    step = max(1, int(model_def.get("frames_steps") or _H3_FRAME_STEP))
    policy = model_def.get("omni_sequence_memory_policy") or {}
    try:
        margin_steps = max(
            0,
            int(
                policy.get(
                    "reference_margin_steps",
                    _H3_OMNI_REFERENCE_MARGIN_STEPS,
                )
            ),
        )
    except (TypeError, ValueError):
        margin_steps = _H3_OMNI_REFERENCE_MARGIN_STEPS

    base_frames = min(maximum, max(minimum, int(profile["frames"])))
    safe_frames = max(minimum, base_frames - margin_steps * step)
    safe_frames = minimum + ((safe_frames - minimum) // step) * step
    safe_frames = min(maximum, max(minimum, safe_frames))
    profile["frames"] = safe_frames
    profile["reference_margin_frames"] = base_frames - safe_frames
    profile["reference_margin_steps"] = margin_steps
    return profile


def apply_h3_omni_sequence_memory_policy(
    inputs: dict,
    model_def: dict,
    hardware: dict | None,
) -> dict | None:
    """Resolve one native Omni Sequence pass without changing total duration.

    Auto uses the GPU/canvas/checkpoint recommendation plus Ref2VA headroom.
    A locked manual value remains available for experimentation and is only
    normalized to the model's legal frame lattice.
    """

    if not isinstance(inputs, dict) or not (model_def or {}).get(
        "omni_reference"
    ):
        return None
    hardware = hardware or {}
    minimum = max(1, int(model_def.get("frames_minimum") or _H3_MIN_FRAMES))
    maximum = max(minimum, int(model_def.get("frames_maximum") or _H3_MAX_FRAMES))
    step = max(1, int(model_def.get("frames_steps") or _H3_FRAME_STEP))
    try:
        total_vram_gb = float(hardware.get("gpu_vram_gb") or 0)
        total_frames = int(inputs.get("video_length") or minimum)
        requested_clip = int(
            inputs.get("minimax_h3_sequence_clip_frames") or maximum
        )
    except (TypeError, ValueError):
        return None

    requested_clip = min(maximum, max(minimum, requested_clip))
    requested_clip = minimum + ((requested_clip - minimum) // step) * step
    manual_override = inputs.get(
        "minimax_h3_sequence_memory_override", False
    ) is True
    resolution = _normalize_h3_resolution(
        inputs.get("resolution", "864x480")
    )
    profile = recommended_h3_omni_sequence_profile(
        total_vram_gb,
        resolution,
        model_def,
    )
    if not manual_override and not profile.get("supported"):
        fallback = profile.get("fallback_resolution") or "a lower resolution"
        return {
            "unsupported": True,
            "message": (
                "MiniMax H3 Omni Auto does not recommend "
                f"{profile['resolution']} on a {total_vram_gb:.0f} GB GPU: "
                "H3 cannot use a native clip shorter than its 124-frame "
                "(5.2-second) minimum. "
                f"Choose {fallback}, or manually lock Sequence Window Length "
                "in Advanced to try this combination experimentally."
            ),
            "gpu_vram_gb": total_vram_gb,
            "resolution": profile["resolution"],
            "requested_clip_frames": requested_clip,
            "effective_clip_frames": None,
            "output_frames": total_frames,
            "fallback_resolution": fallback,
            "checkpoint": profile["checkpoint"],
            "manual_override": False,
        }

    safe_clip = int(profile.get("frames") or maximum)
    effective_clip = requested_clip if manual_override else safe_clip
    inputs["minimax_h3_sequence_clip_frames"] = effective_clip
    return {
        "gpu_vram_gb": total_vram_gb,
        "resolution": resolution,
        "requested_clip_frames": requested_clip,
        "effective_clip_frames": effective_clip,
        "recommended_clip_frames": safe_clip,
        "output_frames": total_frames,
        "checkpoint": profile["checkpoint"],
        "manual_override": manual_override,
        "reference_margin_frames": int(
            profile.get("reference_margin_frames") or 0
        ),
    }


def apply_h3_native_omni_memory_policy(
    inputs: dict,
    model_def: dict,
    hardware: dict | None,
) -> dict | None:
    """Keep an automatic one-shot Ref2VA clip within its safe native pass.

    Unlike Omni Reference Sequence, a one-shot Ref2VA request cannot divide
    the requested duration into continuation windows. Auto therefore reduces
    the output duration only when the request still carries Auto's shorter
    Window Length. A matching native window is an intentional user override
    and must be honored, including requests from cached v1.7.x frontends that
    predate the explicit memory-override flag. Sequence mode is handled
    separately and keeps its full requested timeline.
    """

    if (
        not isinstance(inputs, dict)
        or not (model_def or {}).get("omni_reference")
        or inputs.get("minimax_h3_reference_sequence") is True
    ):
        return None
    if (
        inputs.get("sliding_window_memory_override") is True
        or inputs.get("minimax_h3_sequence_memory_override") is True
    ):
        return None
    hardware = hardware or {}
    try:
        total_vram_gb = float(hardware.get("gpu_vram_gb") or 0)
        requested_frames = int(inputs.get("video_length") or _H3_MIN_FRAMES)
        requested_window_frames = int(
            inputs.get("sliding_window_size") or _H3_MIN_FRAMES
        )
    except (TypeError, ValueError):
        return None
    if total_vram_gb <= 0:
        return None

    minimum_frames = max(
        1,
        int((model_def or {}).get("frames_minimum") or _H3_MIN_FRAMES),
    )
    maximum_frames = max(
        minimum_frames,
        int((model_def or {}).get("frames_maximum") or _H3_MAX_FRAMES),
    )
    requested_native_frames = normalize_h3_clip_frame_count(
        requested_frames,
        minimum_frames=minimum_frames,
        maximum_frames=maximum_frames,
        frame_step=int(
            (model_def or {}).get("frames_steps") or _H3_FRAME_STEP
        ),
    )
    requested_native_window_frames = normalize_h3_clip_frame_count(
        requested_window_frames,
        minimum_frames=minimum_frames,
        maximum_frames=maximum_frames,
        frame_step=int(
            (model_def or {}).get("frames_steps") or _H3_FRAME_STEP
        ),
    )
    if (
        requested_frames > minimum_frames
        and requested_native_window_frames >= requested_native_frames
    ):
        # v1.7.0-v1.7.2 could lose the explicit boolean during model-option
        # or restored-sidecar state changes even though Duration and Window
        # Length still visibly matched. Treat that unambiguous request shape
        # as the same manual override instead of silently returning 5.2s.
        inputs["sliding_window_memory_override"] = True
        return None

    resolution = _normalize_h3_resolution(
        inputs.get("resolution", "864x480")
    )
    profile = recommended_h3_window_profile(
        total_vram_gb,
        resolution,
        model_def,
    )
    if not profile["supported"]:
        fallback = profile.get("fallback_resolution") or "a lower resolution"
        return {
            "unsupported": True,
            "message": (
                "MiniMax H3 Omni Auto does not recommend "
                f"{profile['resolution']} on a {total_vram_gb:.0f} GB GPU: "
                "H3 cannot use a native clip shorter than its 124-frame "
                "(5.2-second) minimum. "
                f"Choose {fallback}, or enable Multi-window sequence to divide "
                "a longer timeline into VRAM-aware native windows."
            ),
            "gpu_vram_gb": total_vram_gb,
            "resolution": profile["resolution"],
            "requested_frames": requested_frames,
            "effective_frames": None,
            "fallback_resolution": fallback,
            "checkpoint": profile["checkpoint"],
        }

    safe_frames = int(profile["frames"])
    if requested_frames <= safe_frames:
        return None

    inputs["video_length"] = safe_frames
    inputs["sliding_window_size"] = safe_frames
    return {
        "gpu_vram_gb": total_vram_gb,
        "resolution": resolution,
        "requested_frames": requested_frames,
        "effective_frames": safe_frames,
        "checkpoint": profile["checkpoint"],
    }


def h3_runtime_preflight(
    model_def: dict | None,
    hardware: dict | None,
) -> dict | None:
    """Warn when Full H3 is likely to fall onto its very slow path.

    Full uses the INT8 ConvRot checkpoint. Its practical fast path depends on
    Triton-backed Quanto kernels and substantial system RAM for MMGP's pinned
    model blocks. Pruned does not carry those same requirements, so it is the
    safe one-click recommendation. This intentionally never blocks Full.
    """

    model_def = model_def or {}
    hardware = hardware or {}
    architecture = str(model_def.get("architecture") or "")
    if not architecture.startswith("minimax_h3") or not bool(
        model_def.get("minimax_h3_full_checkpoint", False)
    ):
        return None

    reasons = []
    if (
        "supports_triton" in hardware
        and hardware.get("supports_triton") is not True
    ):
        reasons.append({
            "code": "triton_unavailable",
            "message": (
                "Triton is unavailable, so Full H3 cannot use its injected "
                "Quanto INT8 fast path and may run dramatically slower."
            ),
        })

    try:
        ram_gb = float(hardware.get("ram_gb") or 0.0)
    except (TypeError, ValueError):
        ram_gb = 0.0
    if 0 < ram_gb < _H3_FULL_MINIMUM_SYSTEM_RAM_GB:
        reasons.append({
            "code": "system_ram_low",
            "message": (
                f"Full H3 needs roughly "
                f"{_H3_FULL_ESTIMATED_PIPELINE_RAM_GB:.0f} GB of model RAM "
                f"before OS and app headroom; this system has {ram_gb:.0f} GB."
            ),
        })

    if not reasons:
        return None

    omni_reference = bool(model_def.get("omni_reference", False))
    recommended_model_type = (
        _REF2VA_MODEL_TYPE if omni_reference else _MODEL_TYPE
    )
    workflow_label = "Omni" if omni_reference else "First / Last"
    return {
        "level": "warning",
        "title": "Full H3 may be very slow on this system",
        "message": (
            " ".join(reason["message"] for reason in reasons)
            + f" Use Pruned Turbo for the same {workflow_label} workflow "
            "with much lower RAM and weight-streaming cost."
        ),
        "reasons": reasons,
        "recommended_model_type": recommended_model_type,
        "recommended_turbo": True,
        "estimated_pipeline_ram_gb": _H3_FULL_ESTIMATED_PIPELINE_RAM_GB,
        "minimum_system_ram_gb": _H3_FULL_MINIMUM_SYSTEM_RAM_GB,
        "detected_ram_gb": ram_gb or None,
        "supports_triton": hardware.get("supports_triton"),
        "blocking": False,
    }


def apply_h3_window_memory_policy(
    inputs: dict,
    model_def: dict,
    hardware: dict | None,
) -> dict | None:
    """Shrink only an FL2VA pass window when the detected GPU needs it.

    The requested final duration stays unchanged. A smaller pass simply makes
    the existing continuation scheduler use more windows, which is the same
    memory lever WanGP documents for long H3 generations.
    """

    if not isinstance(inputs, dict) or (model_def or {}).get("omni_reference"):
        return None
    if inputs.get("sliding_window_memory_override", False) is True:
        return None
    hardware = hardware or {}
    try:
        total_vram_gb = float(hardware.get("gpu_vram_gb") or 0)
        total_frames = int(inputs.get("video_length") or _H3_MIN_FRAMES)
        requested_window = int(
            inputs.get("sliding_window_size") or _H3_MAX_FRAMES
        )
    except (TypeError, ValueError):
        return None
    if total_vram_gb <= 0:
        return None

    resolution = _normalize_h3_resolution(
        inputs.get("resolution", "864x480")
    )
    profile = recommended_h3_window_profile(
        total_vram_gb,
        resolution,
        model_def,
    )
    if not profile["supported"]:
        fallback = profile.get("fallback_resolution") or "a lower resolution"
        message = (
            "MiniMax H3 Auto mode does not recommend "
            f"{profile['resolution']} on a {total_vram_gb:.0f} GB GPU: "
            "H3 cannot use a window shorter than its 124-frame "
            "(5.2-second) minimum. "
            f"Choose {fallback}, or manually lock Window Length in Advanced "
            "to try this combination experimentally."
        )
        return {
            "unsupported": True,
            "message": message,
            "gpu_vram_gb": total_vram_gb,
            "resolution": profile["resolution"],
            "requested_window_frames": requested_window,
            "effective_window_frames": None,
            "output_frames": total_frames,
            "fallback_resolution": fallback,
            "checkpoint": profile["checkpoint"],
        }

    safe_window = int(profile["frames"])
    if total_frames <= safe_window or requested_window <= safe_window:
        return None

    inputs["sliding_window_size"] = safe_window
    return {
        "gpu_vram_gb": total_vram_gb,
        "resolution": resolution,
        "requested_window_frames": requested_window,
        "effective_window_frames": safe_window,
        "output_frames": total_frames,
        "checkpoint": profile["checkpoint"],
    }


def pace_h3_sliding_window_prompt(
    prompt,
    window_no,
    total_windows,
    *,
    fps=24,
    current_video_length=None,
    requested_frames_to_generate=None,
    num_frames_generated=0,
    reuse_frames=0,
):
    """Turn one full-shot H3 prompt into a continuation-window contract.

    The scheduler historically reused the same complete prompt for every
    continuation pass. H3 therefore tried to finish a 15-second action plan
    inside the first five-second 1080p pass, then repeated it.  Keep the full
    visual context available, but tell H3 which chronological slice it owns.
    Tagged dialogue outside that slice is made non-spoken so a line is not
    repeated in every continuation window.
    """

    import re

    try:
        window_no = max(1, int(window_no))
        total_windows = max(1, int(total_windows))
    except (TypeError, ValueError):
        return prompt
    if total_windows <= 1:
        return prompt

    try:
        fps = max(1.0, float(fps))
        total_frames = max(1, int(requested_frames_to_generate or 0))
        completed_frames = max(0, int(num_frames_generated or 0))
        pass_frames = max(1, int(current_video_length or 0))
        if window_no > 1:
            pass_frames = max(1, pass_frames - max(0, int(reuse_frames or 0)))
    except (TypeError, ValueError):
        total_frames = total_windows
        completed_frames = window_no - 1
        pass_frames = 1

    if total_frames <= 0:
        total_frames = total_windows
        completed_frames = window_no - 1
        pass_frames = 1
    segment_start = min(1.0, completed_frames / float(total_frames))
    segment_end = min(
        1.0,
        (completed_frames + pass_frames) / float(total_frames),
    )
    if segment_end <= segment_start:
        segment_start = (window_no - 1) / float(total_windows)
        segment_end = window_no / float(total_windows)

    dialogue_pattern = re.compile(r"<d>.*?</d>", re.IGNORECASE | re.DOTALL)
    dialogue_matches = list(dialogue_pattern.finditer(str(prompt)))
    allowed_dialogue_indices = set()
    if dialogue_matches:
        dialogue_weights = [
            max(1, len(re.findall(r"\b\w+\b", match.group(0))))
            for match in dialogue_matches
        ]
        total_weight = max(1, sum(dialogue_weights))
        cumulative = 0
        for index, weight in enumerate(dialogue_weights):
            midpoint = (cumulative + (weight / 2.0)) / total_weight
            cumulative += weight
            if (
                segment_start <= midpoint < segment_end
                or (window_no == total_windows and midpoint >= segment_start)
            ):
                allowed_dialogue_indices.add(index)

        dialogue_index = -1

        def replace_dialogue(match):
            nonlocal dialogue_index
            dialogue_index += 1
            if dialogue_index in allowed_dialogue_indices:
                return match.group(0)
            return "[DIALOGUE RESERVED FOR ANOTHER CONTINUATION WINDOW]"

        prompt = dialogue_pattern.sub(replace_dialogue, str(prompt))

    start_seconds = completed_frames / fps
    end_seconds = min(total_frames, completed_frames + pass_frames) / fps
    total_seconds = total_frames / fps
    if window_no == 1:
        phase_instruction = (
            "Perform only the opening portion of the plan. Do not rush to, "
            "show, or resolve its later actions or final beat; end with the "
            "action naturally still in progress."
        )
    elif window_no == total_windows:
        phase_instruction = (
            "Continue directly from the supplied previous frame without "
            "restarting or repeating earlier action, then complete only the "
            "remaining actions and final beat by this segment's end."
        )
    else:
        phase_instruction = (
            "Continue directly from the supplied previous frame without "
            "restarting or repeating earlier action. Advance only the middle "
            "portion assigned here and reserve the outcome for a later window."
        )

    if dialogue_matches:
        if allowed_dialogue_indices:
            dialogue_instruction = (
                "Only the <d> dialogue tags still present below may be spoken "
                "in this segment, once each and in order. Reserved dialogue "
                "markers are silent and must not produce speech or gibberish."
            )
        else:
            dialogue_instruction = (
                "This segment has no assigned scripted dialogue. Everyone "
                "remains silent with mouths closed; do not add speech, muttering, "
                "or gibberish. Reserved dialogue markers are not spoken."
            )
    else:
        dialogue_instruction = (
            "Do not invent or repeat dialogue beyond what this segment of the "
            "full-shot plan explicitly requires."
        )

    return (
        "SLIDING-WINDOW TIMING CONTRACT (highest priority): This is "
        f"continuation window {window_no} of {total_windows} for one "
        f"continuous {total_seconds:.1f}-second shot, covering approximately "
        f"{start_seconds:.1f}s to {end_seconds:.1f}s. The FULL-SHOT PLAN below "
        "describes the complete timeline, not a command to perform everything "
        f"inside this shorter segment. {phase_instruction} "
        f"{dialogue_instruction}\n\nFULL-SHOT PLAN:\n{prompt}"
    )


def enforce_h3_source_continuation_prompt(prompt):
    """Keep Studio Extend from treating the source tail as a new-shot ref.

    H3 receives native motion history plus the final boundary frame, but the
    model may still interpret a normal action prompt as permission to begin a
    fresh camera setup at the first generated frame.  Extend promises a
    temporal continuation, so make the boundary behavior explicit while
    leaving the user free to request cuts later in the generated portion.
    """

    prompt = str(prompt or "").strip()
    if not prompt:
        return prompt
    marker = "SOURCE-VIDEO CONTINUATION CONTRACT"
    if marker in prompt:
        return prompt
    return (
        f"{marker} (highest priority): At 0.00 seconds, continue directly "
        "from the supplied source video's final moment as the same "
        "uninterrupted take. The opening generated frames preserve the exact "
        "camera position, lens, framing, environment, lighting, subject "
        "identities, poses, screen positions, and current motion trajectories. "
        "Do not cut, reset the action, change angle, or begin a new establishing "
        "shot at the extension boundary. Continue the existing motion naturally "
        "before introducing the requested action. If a later camera cut is "
        "requested, perform it only after the continuation is visibly "
        "established.\n\nEXTENSION PLAN:\n"
        f"{prompt}"
    )


class family_handler:
    @staticmethod
    def query_supported_types():
        return [
            _MODEL_TYPE,
            _FULL_MODEL_TYPE,
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        ]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "minimax_h3"

    @staticmethod
    def query_family_infos():
        return {"minimax_h3": (55, "MiniMax H3")}

    @staticmethod
    def recommend_text_encoder(hardware, model_def=None):
        variants = (model_def or {}).get("minimax_h3_text_encoder_variants")
        return _recommend_text_encoder(hardware, variants)

    @staticmethod
    def set_cache_parameters(
        cache_type,
        base_model_type,
        model_def,
        inputs,
        skip_steps_cache,
    ):
        if cache_type != "first_block":
            raise ValueError(
                "MiniMax H3 supports only First Block Cache; "
                f"received {cache_type!r}."
            )
        value = float(skip_steps_cache.multiplier)
        skip_steps_cache.threshold = _LEGACY_FIRST_BLOCK_CACHE_THRESHOLDS.get(
            value,
            value,
        )

    @staticmethod
    def query_model_def(base_model_type, model_def):
        omni_reference = base_model_type in {
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        full_checkpoint = base_model_type in {
            _FULL_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        text_encoder_variants = _text_encoder_variants()
        workflow_help = (
            "OMNI REFERENCES\n"
            "Use ordered image, video, and audio references to guide identity, "
            "appearance, motion, scenes, voices, or sound. The references guide "
            "a newly generated result rather than becoming fixed first/last frames. "
            "H3 also generates synchronized stereo audio. Multi-window sequence can "
            "continue beyond one native pass while carrying recent motion and "
            "matching audio alongside the same canonical references."
            if omni_reference
            else
            "FIRST / LAST\n"
            "Generate from text alone, a first frame, a last frame, or both. H3 "
            "can also pass through multiple additional Frames at exact points "
            "on the timeline. It can follow an uploaded soundtrack or a Control "
            "Video's audio, and it can preserve a Control Video's pictures while "
            "creating a new soundtrack. Native video-to-video editing can reimagine "
            "the whole Control Video, only a white masked area, or everything outside "
            "that mask. These are source-media workflows, not voice-reference "
            "cloning. Longer videos continue through native 14.4-second "
            "windows while carrying recent motion and matching stereo audio into "
            "the next window."
        )
        checkpoint_help = (
            "FULL 33B\n"
            "The larger original checkpoint uses more disk, RAM, and weight "
            "streaming. Choose it when you specifically want the Full model; "
            "Turbo also works here, but Pruned is the safer choice on 16 GB GPUs."
            if full_checkpoint
            else
            "PRUNED 20B (RECOMMENDED)\n"
            "The lighter checkpoint has the same workflow controls with lower "
            "disk, RAM, and loading cost. H3 LoRAs, including Turbo, are "
            "converted automatically when their AdaLN layout differs."
        )
        result = {
            "dtype": "bf16",
            "fps": 24,
            # H3's video VAE accepts only 17*n+5 frames.  124 is the first
            # valid count at or above five seconds; 345 is the last at or
            # below fifteen seconds.
            "frames_minimum": _H3_MIN_FRAMES,
            "frames_steps": 17,
            "frames_maximum": _H3_MAX_FRAMES,
            "latent_size": 17,
            "block_size": 32,
            "vae_block_size": 32,
            "frame_alignment_modulus": 17,
            "frame_alignment_remainder": 5,
            "frame_alignment_mode": "ceil",
            "sliding_window": True,
            "video_continuation": True,
            # The overall joined timeline need not itself lie on H3's
            # per-pass 17*n+5 grid. Keep the requested total exact, align
            # each pass independently, then trim only the final joined tail.
            "sliding_window_exact_total_frames": True,
            "sliding_window_trim_to_requested": True,
            "sliding_window_end_image_at_final": not omni_reference,
            "sliding_window_auto_prompt_pacing": not omni_reference,
            # The rolling FL2VA contract consumes the exact generated audio
            # tail alongside its multi-frame visual history.  The generic
            # scheduler supplies that tail through ``input_waveform``.
            "audio_guide_window_slicing": True,
            "sliding_window_audio_history": True,
            # Standard Director renders independent native-duration shots;
            # its Seamless option may opt FL2VA into the same native rolling
            # continuation contract used by Studio.
            "director_video_strategy": (
                "omni_reference" if omni_reference else "bounded_start_end"
            ),
            "director_audio_input_mode": (
                "reference_manifest" if omni_reference else "none"
            ),
            "director_reference_mode": (
                "omni_manifest" if omni_reference else "start_end"
            ),
            # Ref2VA consumes the user's character/location references
            # directly. FL2VA can render T2V or use generated start/end
            # frames, selected per Director project.
            "director_shot_image_support": (
                "direct_references" if omni_reference else "optional"
            ),
            "director_endpoint_continuity": not omni_reference,
            "director_trim_end_frames": False,
            "t2v_class": True,
            "i2v_class": not omni_reference,
            # FL2VA supports native continuation from a source video's
            # audiovisual tail.  ``V`` is not merely a Classic-UI label: the
            # shared task normalizer uses this capability string as an
            # allowlist.  Omitting it caused Studio Extend to retain the path
            # in metadata while silently stripping the source before H3 ran.
            "image_prompt_types_allowed": "" if omni_reference else "TSEV",
            "end_frames_always_enabled": not omni_reference,
            # FL2VA can pin additional pictures to exact target positions.
            # Maestro exposes this through the unified Frame strip instead of
            # adding WanGP's separate frames-injection selector.
            "custom_frames_injection": not omni_reference,
            "returns_audio": True,
            "control_video_trim_disabled": True,
            # Studio Ref2VA audio is normally an ordered Omni reference.
            # Director can additionally use a hidden exact-target soundtrack
            # route for music/dialogue timing; keep the public capability
            # explicit so the UI still distinguishes Audio In from Audio Out.
            "supports_reference_audio": omni_reference,
            "no_negative_prompt": True,
            "guidance_max_phases": 0,
            "visible_phases": 0,
            "compile": False,
            # H3's packed BF16 head-dimension-128 attention can use the
            # bundled Sol Engine from a compatible SM89+ / Triton 3.6 runtime.
            "sol_attention": True,
            "first_block_cache": True,
            "first_block_cache_thresholds": _FIRST_BLOCK_CACHE_THRESHOLDS,
            "skip_steps_multiplier_choices": _FIRST_BLOCK_CACHE_STRENGTHS,
            "skip_steps_multiplier_label": "First Block Cache Threshold",
            "runtime_custom_settings": list(
                _H3_LONG_SEQUENCE_RUNTIME_SETTINGS
            ),
            "resolutions": _RESOLUTIONS,
            "resolution_presets": _H3_RESOLUTION_PRESETS,
            "resolution_preset_order": _H3_RESOLUTION_PRESET_ORDER,
            "supports_auto_aspect": True,
            "auto_resolution_budgets": _H3_AUTO_RESOLUTION_BUDGETS,
            "auto_resolution_fallbacks": _H3_AUTO_RESOLUTION_FALLBACKS,
            "profiles_dir": ["minimax_h3"],
            "minimax_h3_assets_root": _ASSETS_ROOT,
            "text_encoder_folder": _ASSETS_ROOT,
            "text_encoder_quantization": "int8",
            "text_encoder_URLs": text_encoder_variants["nvfp4_awq"]["URLs"],
            "minimax_h3_text_encoder_default": "nvfp4_awq",
            "minimax_h3_text_encoder_variants": text_encoder_variants,
            # New installs follow WanGP's BF16/INT8 pruned exports while
            # existing Maestro installs may still hold Comfy's scaled-FP8
            # checkpoint. They are not byte duplicates, but this runtime
            # supports both tensor formats. Keep the INT8 and legacy FP8
            # names as bidirectional migration aliases so the default switch
            # to INT8 does not download a second ~20B transformer.
            "compatible_model_paths": (
                {}
                if full_checkpoint
                else {
                    (
                        _REF2VA_TRANSFORMER
                        if omni_reference
                        else _TRANSFORMER
                    ): [
                        (
                            _WANGP_REF2VA_PRUNED_TRANSFORMER
                            if omni_reference
                            else _WANGP_FL2VA_PRUNED_TRANSFORMER
                        )
                    ],
                    (
                        _WANGP_REF2VA_PRUNED_TRANSFORMER
                        if omni_reference
                        else _WANGP_FL2VA_PRUNED_TRANSFORMER
                    ): [
                        (
                            _REF2VA_TRANSFORMER
                            if omni_reference
                            else _TRANSFORMER
                        )
                    ],
                }
            ),
            "compatible_model_qkv_layouts": (
                {}
                if full_checkpoint
                else {
                    (
                        _WANGP_REF2VA_PRUNED_TRANSFORMER
                        if omni_reference
                        else _WANGP_FL2VA_PRUNED_TRANSFORMER
                    ): "interleaved",
                    (
                        _WANGP_REF2VA_PRUNED_BF16_TRANSFORMER
                        if omni_reference
                        else _WANGP_FL2VA_PRUNED_BF16_TRANSFORMER
                    ): "interleaved",
                }
            ),
            # WanGP stores every Qwen variant in its upstream folder while
            # Maestro keeps the weight beside its other H3 assets. These are
            # the same published files (the Comfy NVFP4 artifact is also
            # byte-identical), so support both relative layouts.
            "compatible_text_encoder_paths": {
                filename: [os.path.join(_WANGP_TEXT_ENCODER_FOLDER, filename)]
                for filename in (
                    _TEXT_ENCODER,
                    _TEXT_ENCODER_BF16,
                    _TEXT_ENCODER_INT8,
                    _TEXT_ENCODER_GGUF_Q2,
                    _TEXT_ENCODER_GGUF_Q4,
                )
            },
            "minimax_h3_full_checkpoint": full_checkpoint,
            "minimax_h3_transformer_working_vram_gb": (
                _TRANSFORMER_WORKING_VRAM_MB / 1024
            ),
            # Director needs this policy before its LLM allocates shot
            # duration. Publish it for both First / Last and Omni; only the
            # former exposes Studio's sliding-window controls.
            "director_memory_policy": (
                _H3_FULL_WINDOW_MEMORY_POLICY
                if full_checkpoint
                else _H3_PRUNED_WINDOW_MEMORY_POLICY
            ),
            "selector_help": f"{workflow_help}\n\n{checkpoint_help}",
            "lora_compatibility_note": (
                "H3 LoRAs are supported; Maestro converts Pruned adapters when needed."
                if full_checkpoint
                else
                "H3 LoRAs are supported; Maestro converts Full adapters when needed."
            ),
        }
        if omni_reference:
            sequence_memory_policy = (
                _H3_FULL_WINDOW_MEMORY_POLICY
                if full_checkpoint
                else _H3_PRUNED_WINDOW_MEMORY_POLICY
            )
            result.update(
                {
                    "omni_reference": True,
                    "omni_reference_limits": {
                        "image": 9,
                        "video": 3,
                        "audio": 3,
                        "total": 12,
                    },
                    "omni_reference_detail_choices": [
                        ("Match output (faster)", "match"),
                        ("High detail (official PDD recipe)", "max"),
                    ],
                    "omni_reference_detail_default": "match",
                    # Native Ref2VA continuation has the same target-pass
                    # memory curve plus canonical-reference activation cost.
                    # Publish it under an Omni-specific name so Studio can
                    # auto-size each sequence window independently.
                    "omni_sequence_memory_policy": {
                        **sequence_memory_policy,
                        "reference_margin_steps": (
                            _H3_OMNI_REFERENCE_MARGIN_STEPS
                        ),
                    },
                    "sliding_window_defaults": dict(
                        _H3_SLIDING_WINDOW_DEFAULTS
                    ),
                }
            )
        else:
            result["sliding_window_defaults"] = dict(
                _H3_SLIDING_WINDOW_DEFAULTS
            )
            result["sliding_window_memory_policy"] = (
                _H3_FULL_WINDOW_MEMORY_POLICY
                if full_checkpoint
                else _H3_PRUNED_WINDOW_MEMORY_POLICY
            )
            result.update(
                {
                    "guide_custom_choices": {
                        "choices": [
                            ("No Control Video", ""),
                            ("Use Control Video", "GV"),
                            ("Inject Frames", "KFI"),
                        ],
                        "letters_filter": "GVKFI",
                        "default": "",
                        "label": "Control Video / Frames",
                    },
                    "video_guide_label": "Control Video",
                    "video_to_video_inpaint": True,
                    "mask_preprocessing": {
                        "selection": ["", "A", "NA"],
                        "labels": {
                            "": "Whole Frame",
                            "A": "Inside White Mask",
                            "NA": "Outside White Mask",
                        },
                        "default": "",
                        "label": "Area to Edit",
                    },
                    "any_audio_prompt": True,
                    "audio_prompt_choices": True,
                    "audio_guide_label": "Source Audio / Soundtrack",
                    "audio_prompt_type_sources": {
                        "selection": ["", "A", "K", "2"],
                        "labels": {
                            "": "Generate Video and Audio from Text",
                            "A": "Generate Video from Soundtrack + Text",
                            "K": "Generate Video from Control-Video Audio + Text",
                            "2": "Keep Control Video and Generate New Audio",
                        },
                        "letters_filter": "AK2",
                        "label": "Media Source",
                        "show_label": True,
                        "default": "",
                    },
                    # A visible standalone soundtrack is unambiguously the
                    # FL2VA Audio-to-Video source. Repair older sidecars and
                    # model-switch state if their hidden A selector was lost.
                    "infer_audio_prompt_from_guide": True,
                    "video_length_not_limited_by_audio": True,
                    "output_audio_is_input_audio": True,
                    "minimax_h3_media_sources": True,
                }
            )
        return result

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-minimax-h3",
            type=str,
            default=None,
            help=(
                "Path to a directory that contains MiniMax H3 LoRAs "
                f"(default: {os.path.join(lora_root, 'minimax_h3')})"
            ),
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_minimax_h3", None) or os.path.join(lora_root, "minimax_h3")

    @staticmethod
    def get_vae_block_size(base_model_type):
        return 32

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        processor_files = [
            "chat_template.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ]
        return [
            {
                "repoId": _COMFY_REPO,
                "revision": _COMFY_REVISION,
                "sourceFolderList": ["vae"],
                "targetFolderList": [_ASSETS_ROOT],
                "fileList": [[_VIDEO_VAE, _AUDIO_VAE]],
            },
            {
                "repoId": _OFFICIAL_REPO,
                "revision": _OFFICIAL_REVISION,
                "sourceFolderList": ["processor", "text_encoder"],
                "targetFolderList": [_ASSETS_ROOT, _ASSETS_ROOT],
                "fileList": [processor_files, ["config.json"]],
            },
        ]

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        dtype=None,
        text_encoder_filename=None,
        **kwargs,
    ):
        if dtype is None:
            import torch

            dtype = torch.bfloat16
        from .minimax_h3_main import MiniMaxH3Model

        model = MiniMaxH3Model(
            model_filename=model_filename,
            model_def=model_def or {},
            text_encoder_filename=text_encoder_filename,
            dtype=dtype,
            minimax_h3_text_encoder=kwargs.get(
                "minimax_h3_text_encoder",
                (model_def or {}).get("minimax_h3_text_encoder_default", "nvfp4_awq"),
            ),
        )
        pipe = {
            "transformer": model.transformer,
            # Profile the two Qwen towers independently. Text-only FL2VA
            # never needs the vision tower, while Ref2VA can release it
            # before the 50-layer language model runs. This mirrors WanGP's
            # H3 memory layout and avoids pinning both large components as a
            # single co-resident conditioner.
            "text_encoder": model.conditioner.language_model,
            "vision_encoder": model.conditioner.visual,
            "vae": model.vae,
            "audio_vae": model.audio_vae,
        }
        return model, {
            "pipe": pipe,
            "workingVRAM": {
                "transformer": _TRANSFORMER_WORKING_VRAM_MB,
            },
        }

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        omni_reference = base_model_type in {
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        ui_defaults.update(
            {
                "num_inference_steps": 20,
                "video_length": _H3_MIN_FRAMES,
                "resolution": "864x480",
                "guidance_scale": 1.0,
                "image_prompt_type": "",
                "video_prompt_type": "",
                "audio_prompt_type": "",
                "sliding_window_size": _H3_MAX_FRAMES,
                "sliding_window_overlap": _H3_OVERLAP_DEFAULT,
                "sliding_window_discard_last_frames": 0,
                "skip_steps_cache_type": "",
                "skip_steps_multiplier": 0.08,
                "skip_steps_start_step_perc": 25,
                "denoising_strength": 1.0,
                "masking_strength": 1.0,
            }
        )

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        # Saved settings created before this family existed cannot need a
        # migration, but imported presets still need valid H3 geometry.
        from .packing import align_num_frames

        try:
            requested_frames = int(ui_defaults.get("video_length", 124))
        except (TypeError, ValueError):
            requested_frames = 124
        omni_reference = base_model_type in {
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        aligned_frames = align_num_frames(max(1, requested_frames))
        omni_sequence = (
            omni_reference
            and ui_defaults.get("minimax_h3_reference_sequence") is True
        )
        if requested_frames <= _H3_MAX_FRAMES + 1:
            ui_defaults["video_length"] = min(
                _H3_MAX_FRAMES,
                max(_H3_MIN_FRAMES, aligned_frames),
            )
        elif omni_reference and not omni_sequence:
            ui_defaults["video_length"] = _H3_MAX_FRAMES
        else:
            # A long First/Last or enabled Omni Reference Sequence setting is
            # the joined output duration, not one H3 pass.
            ui_defaults["video_length"] = max(
                _H3_MIN_FRAMES,
                requested_frames,
            )

        try:
            requested_window = int(
                ui_defaults.get("sliding_window_size", _H3_MAX_FRAMES)
            )
        except (TypeError, ValueError):
            requested_window = _H3_MAX_FRAMES
        aligned_window = align_num_frames(max(1, requested_window))
        ui_defaults["sliding_window_size"] = min(
            _H3_MAX_FRAMES,
            max(_H3_MIN_FRAMES, aligned_window),
        )
        if (
            not omni_reference
            and ui_defaults.get("minimax_h3_multi_window") is not True
        ):
            ui_defaults["video_length"] = min(
                ui_defaults["video_length"],
                ui_defaults["sliding_window_size"],
            )
        raw_overlap = ui_defaults.get(
            "sliding_window_overlap",
            _H3_OVERLAP_DEFAULT,
        )
        # Existing Maestro H3 presets used a zero/one-frame anchor. Upgrade
        # that legacy default once for both FL2VA and Ref2VA.
        if settings_version < 2.58 and raw_overlap in (None, 0, 1, "1"):
            raw_overlap = _H3_OVERLAP_DEFAULT
        ui_defaults["sliding_window_overlap"] = normalize_h3_overlap_frames(
            raw_overlap,
            window_frames=ui_defaults["sliding_window_size"],
        )
        ui_defaults["sliding_window_discard_last_frames"] = 0
        ui_defaults["resolution"] = _normalize_h3_resolution(
            ui_defaults.get("resolution", "864x480")
        )
        ui_defaults["guidance_scale"] = 1.0
        ui_defaults.setdefault("denoising_strength", 1.0)
        ui_defaults.setdefault("masking_strength", 1.0)
        cache_value = float(ui_defaults.get("skip_steps_multiplier", 0.08))
        ui_defaults["skip_steps_multiplier"] = (
            _LEGACY_FIRST_BLOCK_CACHE_THRESHOLDS.get(cache_value, cache_value)
        )
        if ui_defaults.get("skip_steps_cache_type") not in {
            "",
            "first_block",
        }:
            ui_defaults["skip_steps_cache_type"] = ""

    @staticmethod
    def custom_prompt_preprocess(
        prompt,
        window_no=1,
        total_windows=1,
        prompts=None,
        model_def=None,
        **kwargs,
    ):
        """Pace one full H3 shot prompt across automatic continuations."""

        is_source_extension = (
            int(window_no or 1) == 1
            and kwargs.get("video_source") is not None
            and "V" in str(kwargs.get("image_prompt_type") or "")
        )
        if is_source_extension:
            prompt = enforce_h3_source_continuation_prompt(prompt)
            print(
                "[MiniMax H3 Extend] Enforcing uninterrupted same-shot "
                "continuity at the source boundary."
            )

        if (model_def or {}).get("omni_reference", False):
            return prompt
        # Multiple prompt lines are already an explicit user-authored mapping
        # of one prompt per window. Never rewrite that advanced workflow.
        if isinstance(prompts, (list, tuple)) and len(prompts) > 1:
            return prompt
        if int(window_no or 1) == 1 and int(total_windows or 1) > 1:
            print(
                "[MiniMax H3] Auto-pacing one full-shot prompt across "
                f"{int(total_windows)} continuation windows."
            )
        return pace_h3_sliding_window_prompt(
            prompt,
            window_no,
            total_windows,
            fps=kwargs.get("fps", 24),
            current_video_length=kwargs.get("current_video_length"),
            requested_frames_to_generate=kwargs.get(
                "requested_frames_to_generate"
            ),
            num_frames_generated=kwargs.get("num_frames_generated", 0),
            reuse_frames=kwargs.get("reuse_frames", 0),
        )

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        """Enforce H3's single-pass and continuation geometry server-side."""

        omni_reference = base_model_type in {
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        video_prompt_type = str(inputs.get("video_prompt_type") or "")
        audio_prompt_type = str(inputs.get("audio_prompt_type") or "")
        if not omni_reference:
            frozen_video_mode = "2" in audio_prompt_type
            control_video = (
                "G" in video_prompt_type
                and "V" in video_prompt_type
                and inputs.get("video_guide") is not None
            )
            try:
                denoising_strength = float(
                    inputs.get("denoising_strength", 1.0)
                )
                masking_strength = float(
                    inputs.get("masking_strength", 1.0)
                )
            except (TypeError, ValueError):
                return "MiniMax H3 video editing strengths must be numbers from 0 to 1."
            if not 0.0 <= denoising_strength <= 1.0:
                return "MiniMax H3 denoising strength must be between 0 and 1."
            if not 0.0 <= masking_strength <= 1.0:
                return "MiniMax H3 masking strength must be between 0 and 1."
            inputs["denoising_strength"] = denoising_strength
            inputs["masking_strength"] = masking_strength
            # Video-to-audio freezes the source pictures. A remembered edit
            # mask/mode is intentionally inert so switching the Audio behavior
            # dropdown cannot make this workflow demand a mask it will not use.
            if frozen_video_mode:
                inputs["video_mask"] = None
                inputs["denoising_strength"] = 1.0
                inputs["masking_strength"] = 1.0
            elif "A" in video_prompt_type:
                if not control_video:
                    return (
                        "MiniMax H3 masked video editing requires a Control "
                        "Video and Use Control Video."
                    )
                if inputs.get("video_mask") is None:
                    return (
                        "MiniMax H3 masked video editing requires a mask video "
                        "(white is the selected area)."
                    )
            elif "N" in video_prompt_type:
                return (
                    "MiniMax H3 Outside Mask editing requires a mask video."
                )
            else:
                inputs["video_mask"] = None
            if not control_video:
                inputs["denoising_strength"] = 1.0
                inputs["masking_strength"] = 1.0
            if frozen_video_mode:
                if "A" in audio_prompt_type or "K" in audio_prompt_type:
                    return (
                        "MiniMax H3 video-to-audio cannot also use a source "
                        "soundtrack."
                    )
                if (
                    "G" not in video_prompt_type
                    or "V" not in video_prompt_type
                    or inputs.get("video_guide") is None
                ):
                    return (
                        "MiniMax H3 video-to-audio requires a Control Video "
                        "and Use Control Video."
                    )
                # A previously loaded soundtrack is not part of this mode.
                # Clear it before WGP's generic audio mux path can mistake it
                # for the requested output soundtrack.
                inputs["audio_guide"] = None
                inputs["audio_guide2"] = None
            if "K" in audio_prompt_type:
                if (
                    "G" not in video_prompt_type
                    or "V" not in video_prompt_type
                    or inputs.get("video_guide") is None
                ):
                    return (
                        "MiniMax H3 Control-Video Audio mode requires a "
                        "Control Video and Use Control Video."
                    )
                from shared.utils.audio_video import extract_audio_tracks

                try:
                    if extract_audio_tracks(
                        inputs["video_guide"],
                        query_only=True,
                    ) == 0:
                        return "The selected MiniMax H3 Control Video has no audio track."
                except Exception as error:
                    return (
                        "Unable to inspect the MiniMax H3 Control Video's "
                        f"audio track: {error}"
                    )
            if "A" in audio_prompt_type and inputs.get("audio_guide") is None:
                return "MiniMax H3 Soundtrack mode requires an uploaded Soundtrack."
            if not any(flag in audio_prompt_type for flag in "AK"):
                inputs["audio_guide"] = None
                inputs["audio_guide2"] = None
        if "F" in str(inputs.get("video_prompt_type") or ""):
            if omni_reference:
                return (
                    "Timed frame injection is available with MiniMax H3 "
                    "First / Last, not H3 Omni references."
                )
            image_refs = list(inputs.get("image_refs") or ())
            frame_positions = [
                item
                for item in str(inputs.get("frames_positions") or "")
                .replace(",", " ")
                .split()
                if item
            ]
            if not image_refs:
                return "Add at least one Frame before enabling H3 frame injection."
            if len(image_refs) != len(frame_positions):
                return (
                    "MiniMax H3 needs one position per injected Frame; "
                    f"received {len(image_refs)} images and "
                    f"{len(frame_positions)} positions."
                )
        try:
            requested_frames = int(inputs.get("video_length", _H3_MIN_FRAMES))
        except (TypeError, ValueError):
            requested_frames = _H3_MIN_FRAMES

        omni_sequence = (
            omni_reference
            and inputs.get("minimax_h3_reference_sequence") is True
        )
        if omni_reference and not omni_sequence:
            inputs["video_length"] = min(
                _H3_MAX_FRAMES,
                max(_H3_MIN_FRAMES, align_h3_num_frames(max(1, requested_frames))),
            )
            inputs["sliding_window_size"] = inputs["video_length"]
        else:
            if requested_frames <= _H3_MAX_FRAMES + 1:
                requested_frames = min(
                    _H3_MAX_FRAMES,
                    max(
                        _H3_MIN_FRAMES,
                        align_h3_num_frames(max(1, requested_frames)),
                    ),
                )
            else:
                requested_frames = max(_H3_MIN_FRAMES, requested_frames)
            inputs["video_length"] = requested_frames

            try:
                requested_window = int(
                    inputs.get("sliding_window_size", _H3_MAX_FRAMES)
                )
            except (TypeError, ValueError):
                requested_window = _H3_MAX_FRAMES
            inputs["sliding_window_size"] = min(
                _H3_MAX_FRAMES,
                max(
                    _H3_MIN_FRAMES,
                    align_h3_num_frames(max(1, requested_window)),
                ),
            )
            if (
                not omni_reference
                and inputs.get("minimax_h3_multi_window") is not True
            ):
                inputs["video_length"] = min(
                    inputs["video_length"],
                    inputs["sliding_window_size"],
                )
            inputs["sliding_window_overlap"] = normalize_h3_overlap_frames(
                inputs.get(
                    "sliding_window_overlap",
                    _H3_OVERLAP_DEFAULT,
                ),
                window_frames=inputs["sliding_window_size"],
            )

        if omni_reference and not omni_sequence:
            inputs["sliding_window_overlap"] = normalize_h3_overlap_frames(
                inputs.get(
                    "sliding_window_overlap",
                    _H3_OVERLAP_DEFAULT,
                ),
                window_frames=inputs["sliding_window_size"],
            )

        inputs["sliding_window_discard_last_frames"] = (
            resolve_h3_long_sequence_discard_frames(
                inputs.get("custom_settings"),
                enabled=(
                    not omni_reference
                    and inputs.get("minimax_h3_multi_window") is True
                    and int(inputs.get("video_length") or 0)
                    > int(inputs.get("sliding_window_size") or 0)
                ),
            )
        )
        inputs["sliding_window_overlap_noise"] = 0
        inputs["sliding_window_color_correction_strength"] = 0
        try:
            import torch

            detected_vram_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if torch.cuda.is_available()
                else 0.0
            )
        except Exception:
            detected_vram_gb = 0.0
        adjustment = apply_h3_window_memory_policy(
            inputs,
            model_def or {"omni_reference": omni_reference},
            {"gpu_vram_gb": detected_vram_gb},
        )
        if adjustment:
            if adjustment.get("unsupported"):
                return adjustment["message"]
            print(
                "[MiniMax H3] VRAM-aware FL2VA window: "
                f"{adjustment['gpu_vram_gb']:.1f} GB, "
                f"{adjustment['resolution']}, "
                f"{adjustment['requested_window_frames']} -> "
                f"{adjustment['effective_window_frames']} frames. "
                "Requested output duration is unchanged."
            )
        if (
            not omni_reference
            and inputs.get("minimax_h3_multi_window") is not True
        ):
            inputs["video_length"] = min(
                inputs["video_length"],
                inputs["sliding_window_size"],
            )
        # The memory policy may shorten the pass after the first overlap
        # validation. Re-cap it so history always leaves at least one legal
        # 17*n+5 target chunk for either FL2VA or Ref2VA to generate.
        inputs["sliding_window_overlap"] = normalize_h3_overlap_frames(
            inputs.get(
                "sliding_window_overlap",
                _H3_OVERLAP_DEFAULT,
            ),
            window_frames=inputs["sliding_window_size"],
        )
        return None
