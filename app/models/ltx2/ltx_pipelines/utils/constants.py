# =============================================================================
# Diffusion Schedule
# =============================================================================

# Noise schedule for the distilled pipeline. These sigma values control noise
# levels at each denoising step and were tuned to match the distillation process.
from ...ltx_core.types import SpatioTemporalScaleFactors

DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

# Stage 2 schedule — ComfyUI's reference distilled 2-stage workflow starts
# stage 2 at sigma 0.85 (not 0.909). Lower start = tighter refinement that
# preserves stage 1's composition instead of drifting. Matches the values
# in the official LTX-2.3 2-stage ComfyUI workflow.
STAGE_2_DISTILLED_SIGMA_VALUES = [0.85, 0.725, 0.4219, 0.0]

# Progressive pipeline upscale stages (same 0.85 start as stage 2 distilled)
PROGRESSIVE_UPSCALE_SIGMA_VALUES = [0.85, 0.725, 0.421875, 0.0]


def build_stage2_sigmas(stage2_steps: int = 3) -> list[float]:
    """Build stage 2 sigma schedule starting at 0.85.

    Matches ComfyUI's reference distilled 2-stage workflow exactly for
    stage2_steps=3. For other step counts, interpolates linearly between
    0.85 and 0 (the sigma values don't have published references for
    custom step counts — linear is a reasonable default).
    """
    if stage2_steps == 3:
        return list(STAGE_2_DISTILLED_SIGMA_VALUES)
    n = max(1, int(stage2_steps))
    step = 0.85 / n
    return [round(0.85 - i * step, 6) for i in range(n)] + [0.0]


# =============================================================================
# Video Generation Defaults
# =============================================================================

DEFAULT_SEED = 10
DEFAULT_1_STAGE_HEIGHT = 512
DEFAULT_1_STAGE_WIDTH = 768
DEFAULT_2_STAGE_HEIGHT = DEFAULT_1_STAGE_HEIGHT * 2
DEFAULT_2_STAGE_WIDTH = DEFAULT_1_STAGE_WIDTH * 2
DEFAULT_NUM_FRAMES = 121
DEFAULT_FRAME_RATE = 24.0
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_CFG_GUIDANCE_SCALE = 4.0


# =============================================================================
# Audio
# =============================================================================

AUDIO_SAMPLE_RATE = 24000


# =============================================================================
# LoRA
# =============================================================================

DEFAULT_LORA_STRENGTH = 1.0


# =============================================================================
# Video VAE Architecture
# =============================================================================

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()
VIDEO_LATENT_CHANNELS = 128


# =============================================================================
# Image Preprocessing
# =============================================================================

# CRF (Constant Rate Factor) for H.264 encoding used in image conditioning.
# Lower = higher quality, 0 = lossless. This mimics compression artifacts.
DEFAULT_IMAGE_CRF = 33


# =============================================================================
# Prompts
# =============================================================================

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)
