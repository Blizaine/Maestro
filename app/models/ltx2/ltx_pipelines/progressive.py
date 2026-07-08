"""
Three-stage progressive resolution pipeline for LTX-2.3.

Stage 1: Generate at 1/4 target resolution (8 steps from noise)
Stage 2: 2x latent upscale to 1/2 resolution (3 steps, partial denoise)
Stage 3: 2x latent upscale to full resolution (3 steps, partial denoise)

Total: 14 steps. Produces better motion because the model focuses on
composition/motion at tiny resolution without splitting capacity on spatial detail.
"""

import logging
import torch

from collections.abc import Callable, Iterator

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep, EulerAncestralDiffusionStep
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.upsampler import upsample_video
from ..ltx_core.model.video_vae import TilingConfig
from ..ltx_core.model.video_vae import decode_video_to_tensor as vae_decode_video_to_tensor
from ..ltx_core.text_encoders.gemma import encode_text, postprocess_text_embeddings, resolve_text_connectors
from ..ltx_core.tools import VideoLatentTools
from ..ltx_core.types import LatentState, VideoPixelShape
from .utils import ModelLedger
from .utils.constants import (
    DISTILLED_SIGMA_VALUES,
    PROGRESSIVE_UPSCALE_SIGMA_VALUES,  # Default: [0.85, 0.725, 0.421875, 0.0]
)
from .utils.helpers import (
    assert_resolution,
    bind_interrupt_check,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    latent_conditionings_by_latent_sequence,
    prepare_mask_injection,
    simple_denoising_func,
    video_conditionings_by_keyframe,
    video_conditionings_by_reference_latent,
)
from .utils.types import PipelineComponents
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.self_refiner import create_self_refiner_handler, normalize_self_refiner_plan
from shared.utils.text_encoder_cache import TextEncoderCache

# Reuse from distilled module
from .distilled import _env_flag, _TransformerBenchWrapper

device = get_device()
_BENCH_TRANSFORMER_ENV = "WAN2GP_LTX2_BENCH_TRANSFORMER"

logger = logging.getLogger(__name__)


def _align_32(v: int) -> int:
    """Snap dimension down to nearest multiple of 32 (VAE requirement)."""
    return max(32, (v // 32) * 32)


def _build_upscale_sigmas(start_sigma: float, steps: int) -> list[float]:
    """Build a sigma schedule for upscale refinement stages.

    Interpolates from start_sigma down to 0.0 using `steps` intermediate values.
    For 3 steps at 0.85: [0.85, 0.567, 0.283, 0.0] — evenly spaced.
    The default [0.85, 0.725, 0.422, 0.0] from the constant is hand-tuned;
    this function provides a general alternative for custom step counts.
    """
    if steps <= 0:
        return [start_sigma, 0.0]
    sigmas = []
    for i in range(steps + 1):
        t = i / steps
        sigmas.append(start_sigma * (1.0 - t))
    return sigmas


def _compute_progressive_resolutions(height: int, width: int):
    """Compute stage resolutions for the 3-stage progressive pipeline.

    Expects height and width to already be 128-aligned (caller's responsibility).
    Returns quarter, half, and full resolution tuples.
    """
    h1, w1 = height // 4, width // 4   # Quarter: 32-aligned (128/4=32)
    h2, w2 = height // 2, width // 2   # Half: 64-aligned (128/2=64)
    h3, w3 = height, width              # Full: 128-aligned

    return (h1, w1), (h2, w2), (h3, w3)


class ProgressivePipeline:
    """
    Three-stage progressive resolution video generation pipeline.

    Stage 1 generates at 1/4 resolution, Stage 2 upscales to 1/2 and refines,
    Stage 3 upscales to full resolution and refines. Uses 14 total steps.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        gemma_root: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device = device,
        fp8transformer: bool = False,
        model_device: torch.device | None = None,
        models: object | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.models = models

        if self.models is None:
            if checkpoint_path is None or gemma_root is None or spatial_upsampler_path is None:
                raise ValueError("checkpoint_path, gemma_root, and spatial_upsampler_path are required.")
            self.model_ledger = ModelLedger(
                dtype=self.dtype,
                device=model_device or device,
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=spatial_upsampler_path,
                gemma_root_path=gemma_root,
                loras=loras or [],
                fp8transformer=fp8transformer,
            )
        else:
            self.model_ledger = None

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self.pipeline_components._pipeline_name = 'progressive'
        self.text_encoder_cache = TextEncoderCache()

    def _get_model(self, name: str):
        if self.models is not None:
            return getattr(self.models, name)
        if self.model_ledger is None:
            raise ValueError(f"Missing model source for '{name}'.")
        return getattr(self.model_ledger, name)()

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        end_images: list[tuple] | None = None,
        end_images_stage2: list[tuple] | None = None,
        inject_images: list[tuple] | None = None,
        inject_images_stage2: list[tuple] | None = None,
        alt_guidance_scale: float = 1.0,
        video_conditioning: list[tuple[str, float]] | None = None,
        video_conditioning_downscale_factor: int = 1,
        latent_conditioning_stage2: torch.Tensor | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        audio_conditionings: list | None = None,
        callback: Callable[..., None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
        loras_slists: dict | None = None,
        text_connectors: dict | None = None,
        masking_source: dict | None = None,
        masking_strength: float | None = None,
        return_latent_slice: slice | None = None,
        self_refiner_setting: int = 0,
        self_refiner_plan: str = "",
        self_refiner_f_uncertainty: float = 0.1,
        self_refiner_certain_percentage: float = 0.999,
        self_refiner_max_plans: int = 1,
        stage2_steps: int = 0,
        keyframe_conditioning_mode: str = "replace",
        keyframe_inject_mode: str = "additive",
        progressive_stage2_steps: int = 5,
        progressive_stage3_steps: int = 3,
        progressive_stage2_sigma: float = 0.85,
        progressive_stage3_sigma: float = 0.85,
        progressive_stage1_image_weight: float = 0.7,
        progressive_stage3_image_weight: float = 0.7,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        alt_guidance_scale = 1.0

        generator = torch.Generator(device=self.device).manual_seed(seed)
        mask_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        ancestral_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 2)
        noiser = GaussianNoiser(generator=generator)
        # Stage 1: ancestral (stochastic) euler for more natural motion
        # Stages 2-3: deterministic euler for crisp detail refinement
        stepper_ancestral = EulerAncestralDiffusionStep(generator=ancestral_generator, eta=1.0)
        stepper_deterministic = EulerDiffusionStep()

        # Self-refiner: up to 3 plans for 3 stages
        self_refiner_handlers = [None, None, None]  # [stage1, stage2, stage3]
        self_refiner_audio_handlers = [None, None, None]
        if self_refiner_setting and self_refiner_setting > 0:
            plans, _ = normalize_self_refiner_plan(self_refiner_plan or "", max_plans=3)
            for si in range(3):
                plan_i = plans[si] if si < len(plans) else []
                if plan_i:
                    self_refiner_handlers[si] = create_self_refiner_handler(
                        plan_i, self_refiner_f_uncertainty, self_refiner_setting,
                        self_refiner_certain_percentage, channel_dim=-1,
                    )
                    self_refiner_audio_handlers[si] = create_self_refiner_handler(
                        plan_i, self_refiner_f_uncertainty, self_refiner_setting,
                        self_refiner_certain_percentage, channel_dim=-1,
                    )

        dtype = torch.bfloat16

        # ── Text encoding ─────────────────────────────────────────────
        text_encoder = self._get_model("text_encoder")
        if enhance_prompt:
            prompt = generate_enhanced_prompt(text_encoder, prompt, images[0][0] if len(images) > 0 else None)
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(
            text_encoder, text_connectors
        )
        encode_fn = lambda prompts: postprocess_text_embeddings(
            encode_text(text_encoder, prompts=prompts),
            feature_extractor, video_connector, audio_connector,
        )
        contexts = self.text_encoder_cache.encode(encode_fn, [prompt], device=self.device, parallel=True)
        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()
        video_context, audio_context = contexts[0]

        # ── Shared setup ──────────────────────────────────────────────
        bench_transformer = _env_flag(_BENCH_TRANSFORMER_ENV, "0")
        video_encoder = self._get_model("video_encoder")
        transformer = _TransformerBenchWrapper(self._get_model("transformer"), enabled=bench_transformer)
        bind_interrupt_check(transformer, interrupt_check)

        _kf_se_fn = image_conditionings_by_adding_guiding_latent if keyframe_conditioning_mode == "additive" else image_conditionings_by_replacing_latent
        _kf_inject_fn = image_conditionings_by_adding_guiding_latent if keyframe_inject_mode == "additive" else image_conditionings_by_replacing_latent

        # Compute stage resolutions (caller must ensure height/width are 128-aligned)
        (h_quarter, w_quarter), (h_half, w_half), (h_full, w_full) = _compute_progressive_resolutions(height, width)
        print(f"[Progressive] Resolutions: {w_quarter}x{h_quarter} → {w_half}x{h_half} → {w_full}x{h_full}")

        # ── Helper: build conditionings at a given resolution ─────────
        def _build_conditionings(h, w, i2v_images, end_imgs=None, inject_imgs=None,
                                 latent_cond=None, i2v_strength_override=None):
            """Build image/video conditionings at the given resolution."""
            # Override I2V strength if requested
            imgs = i2v_images
            if i2v_strength_override is not None and imgs:
                imgs = [
                    (*img[:2], i2v_strength_override, *img[3:]) if len(img) > 3
                    else (*img[:2], i2v_strength_override)
                    for img in imgs
                ]

            conds = _kf_se_fn(
                images=imgs, height=h, width=w,
                video_encoder=video_encoder, dtype=dtype,
                device=self.device, tiling_config=tiling_config,
            )
            if end_imgs:
                conds += _kf_se_fn(
                    images=end_imgs, height=h, width=w,
                    video_encoder=video_encoder, dtype=dtype,
                    device=self.device, tiling_config=tiling_config,
                )
            if inject_imgs:
                conds += _kf_inject_fn(
                    images=inject_imgs, height=h, width=w,
                    video_encoder=video_encoder, dtype=dtype,
                    device=self.device, tiling_config=tiling_config,
                )
            if video_conditioning:
                if int(video_conditioning_downscale_factor or 1) > 1:
                    conds += video_conditionings_by_reference_latent(
                        video_conditioning=video_conditioning,
                        height=h, width=w, num_frames=num_frames,
                        video_encoder=video_encoder, dtype=dtype,
                        device=self.device,
                        downscale_factor=video_conditioning_downscale_factor,
                        tiling_config=tiling_config,
                    )
                else:
                    conds += video_conditionings_by_keyframe(
                        video_conditioning=video_conditioning,
                        height=h, width=w, num_frames=num_frames,
                        video_encoder=video_encoder, dtype=dtype,
                        device=self.device, tiling_config=tiling_config,
                    )
            if latent_cond is not None:
                conds += latent_conditionings_by_latent_sequence(
                    latent_cond, strength=1.0, start_index=0,
                )
            return conds

        # ── Helper: build denoising loop for a stage ──────────────────
        def _make_denoising_loop(pass_no, sr_handler, sr_audio_handler):
            def loop_fn(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper: EulerDiffusionStep,
                preview_tools: VideoLatentTools | None = None,
                mask_context=None,
            ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=simple_denoising_func(
                        video_context=video_context,
                        audio_context=audio_context,
                        transformer=transformer,
                        alt_guidance_scale=alt_guidance_scale,
                    ),
                    mask_context=mask_context,
                    interrupt_check=interrupt_check,
                    callback=callback,
                    preview_tools=preview_tools,
                    pass_no=pass_no,
                    transformer=transformer,
                    self_refiner_handler=sr_handler,
                    self_refiner_handler_audio=sr_audio_handler,
                    self_refiner_generator=generator,
                )
            return loop_fn

        upsampler = self._get_model("spatial_upsampler")

        # ══════════════════════════════════════════════════════════════
        # STAGE 1: Generate at 1/4 resolution (8 steps from noise)
        # ══════════════════════════════════════════════════════════════
        print(f"[Progressive] Stage 1: {w_quarter}x{h_quarter}, 8 steps")
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        if loras_slists is not None:
            s1_steps = len(stage_1_sigmas) - 1
            update_loras_slists(transformer, loras_slists, s1_steps,
                                phase_switch_step=s1_steps, phase_switch_step2=s1_steps)

        # Declare total steps upfront so progress bar shows correct denominator from the start
        _total_progressive_steps = (len(stage_1_sigmas) - 1) + progressive_stage2_steps + progressive_stage3_steps
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(stage_1_sigmas) - 1, pass_no=1, total_steps_hint=_total_progressive_steps)

        stage_1_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=w_quarter, height=h_quarter, fps=frame_rate,
        )
        # Stage 1 I2V weight: separate control to trade off start image fidelity vs motion freedom
        _s1_weight = progressive_stage1_image_weight if progressive_stage1_image_weight < 1.0 else None
        if _s1_weight is not None:
            print(f"[Progressive] Stage 1 image weight: {_s1_weight:.2f}")
        stage_1_conds = _build_conditionings(
            h_quarter, w_quarter, images, end_images, inject_images,
            i2v_strength_override=_s1_weight,
        )
        mask_ctx = prepare_mask_injection(
            masking_source=masking_source, masking_strength=masking_strength,
            output_shape=stage_1_shape, video_encoder=video_encoder,
            components=self.pipeline_components, dtype=dtype, device=self.device,
            tiling_config=tiling_config, generator=mask_generator,
            num_steps=len(stage_1_sigmas) - 1,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_shape,
            conditionings=stage_1_conds,
            audio_conditionings=audio_conditionings,
            noiser=noiser, sigmas=stage_1_sigmas, stepper=stepper_deterministic,  # ancestral disabled pending investigation
            denoising_loop_fn=_make_denoising_loop(1, self_refiner_handlers[0], self_refiner_audio_handlers[0]),
            components=self.pipeline_components,
            dtype=dtype, device=self.device, mask_context=mask_ctx,
        )
        if bench_transformer:
            s1_ms, s1_calls = transformer.consume()
            print(f"[Progressive][bench] Stage 1: {s1_ms / 1000.0:.3f}s ({s1_calls} calls)")
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        # ══════════════════════════════════════════════════════════════
        # STAGE 2: 2x upscale to 1/2 resolution (3 steps, partial denoise)
        # ══════════════════════════════════════════════════════════════
        print(f"[Progressive] Stage 2: {w_half}x{h_half}, 3 steps (sigma 0.85)")
        upscaled_1 = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=upsampler,
        )
        # Center-crop if Stage 1 * 2 doesn't match Stage 2 exactly
        torch.cuda.synchronize()
        cleanup_memory()

        # Use known schedules when possible, build custom schedule otherwise
        if progressive_stage2_steps == 3 and progressive_stage2_sigma == 0.85:
            stage_2_sigmas = torch.Tensor(PROGRESSIVE_UPSCALE_SIGMA_VALUES).to(self.device)
        elif progressive_stage2_steps == 8 and progressive_stage2_sigma >= 0.99:
            # Full distilled schedule — matches ComfyUI 3-stage workflow
            stage_2_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        else:
            stage_2_sigmas = torch.Tensor(_build_upscale_sigmas(progressive_stage2_sigma, progressive_stage2_steps)).to(self.device)
        print(f"[Progressive] Stage 2 sigmas: {[f'{s:.4f}' for s in stage_2_sigmas.tolist()]}")
        if loras_slists is not None:
            s2_steps = len(stage_2_sigmas) - 1
            update_loras_slists(transformer, loras_slists, s2_steps,
                                phase_switch_step=0, phase_switch_step2=s2_steps)
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(stage_2_sigmas) - 1, pass_no=2)

        stage_2_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=w_half, height=h_half, fps=frame_rate,
        )
        # I2V at full strength for upscale stages
        stage_2_conds = _build_conditionings(h_half, w_half, images, end_images, inject_images)
        mask_ctx = prepare_mask_injection(
            masking_source=masking_source, masking_strength=masking_strength,
            output_shape=stage_2_shape, video_encoder=video_encoder,
            components=self.pipeline_components, dtype=dtype, device=self.device,
            tiling_config=tiling_config, generator=mask_generator,
            num_steps=len(stage_2_sigmas) - 1,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape,
            conditionings=stage_2_conds,
            audio_conditionings=audio_conditionings,
            noiser=noiser, sigmas=stage_2_sigmas, stepper=stepper_deterministic,
            denoising_loop_fn=_make_denoising_loop(2, self_refiner_handlers[1], self_refiner_audio_handlers[1]),
            components=self.pipeline_components,
            dtype=dtype, device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_1,
            initial_audio_latent=audio_state.latent,
            mask_context=mask_ctx,
        )
        if bench_transformer:
            s2_ms, s2_calls = transformer.consume()
            print(f"[Progressive][bench] Stage 2: {s2_ms / 1000.0:.3f}s ({s2_calls} calls)")
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        # ══════════════════════════════════════════════════════════════
        # STAGE 3: 2x upscale to full resolution (3 steps, partial denoise)
        # ══════════════════════════════════════════════════════════════
        print(f"[Progressive] Stage 3: {w_full}x{h_full}, 3 steps (sigma 0.85)")
        upscaled_2 = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=upsampler,
        )
        torch.cuda.synchronize()
        cleanup_memory()

        if progressive_stage3_steps == 3 and progressive_stage3_sigma == 0.85:
            stage_3_sigmas = torch.Tensor(PROGRESSIVE_UPSCALE_SIGMA_VALUES).to(self.device)
        elif progressive_stage3_steps == 8 and progressive_stage3_sigma >= 0.99:
            stage_3_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        else:
            stage_3_sigmas = torch.Tensor(_build_upscale_sigmas(progressive_stage3_sigma, progressive_stage3_steps)).to(self.device)
        print(f"[Progressive] Stage 3 sigmas: {[f'{s:.4f}' for s in stage_3_sigmas.tolist()]}")
        if loras_slists is not None:
            s3_steps = len(stage_3_sigmas) - 1
            update_loras_slists(transformer, loras_slists, s3_steps,
                                phase_switch_step=0, phase_switch_step2=s3_steps)
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(stage_3_sigmas) - 1, pass_no=3)

        stage_3_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=w_full, height=h_full, fps=frame_rate,
        )
        # Final stage: use stage2-specific overrides + Stage 3 image weight
        _s3_weight = progressive_stage3_image_weight if progressive_stage3_image_weight < 1.0 else None
        if _s3_weight is not None:
            print(f"[Progressive] Stage 3 image weight: {_s3_weight:.2f}")
        stage_3_conds = _build_conditionings(
            h_full, w_full, images, end_images_stage2, inject_images_stage2,
            latent_cond=latent_conditioning_stage2,
            i2v_strength_override=_s3_weight,
        )
        mask_ctx = prepare_mask_injection(
            masking_source=masking_source, masking_strength=masking_strength,
            output_shape=stage_3_shape, video_encoder=video_encoder,
            components=self.pipeline_components, dtype=dtype, device=self.device,
            tiling_config=tiling_config, generator=mask_generator,
            num_steps=len(stage_3_sigmas) - 1,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_3_shape,
            conditionings=stage_3_conds,
            audio_conditionings=audio_conditionings,
            noiser=noiser, sigmas=stage_3_sigmas, stepper=stepper_deterministic,
            denoising_loop_fn=_make_denoising_loop(3, self_refiner_handlers[2], self_refiner_audio_handlers[2]),
            components=self.pipeline_components,
            dtype=dtype, device=self.device,
            noise_scale=stage_3_sigmas[0],
            initial_video_latent=upscaled_2,
            initial_audio_latent=audio_state.latent,
            mask_context=mask_ctx,
        )
        if bench_transformer:
            s3_ms, s3_calls = transformer.consume()
            print(f"[Progressive][bench] Stage 3: {s3_ms / 1000.0:.3f}s ({s3_calls} calls)")
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        # ── Decode ────────────────────────────────────────────────────
        torch.cuda.synchronize()
        del transformer
        del video_encoder
        del upsampler
        cleanup_memory()

        latent_slice = None
        if return_latent_slice is not None:
            latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
        decoded_video = vae_decode_video_to_tensor(
            video_state.latent,
            self._get_model("video_decoder"),
            tiling_config,
            expected_frames=int(stage_3_shape.frames),
            expected_height=int(stage_3_shape.height),
            expected_width=int(stage_3_shape.width),
            interrupt_check=interrupt_check,
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self._get_model("audio_decoder"), self._get_model("vocoder")
        )

        if latent_slice is not None:
            return decoded_video, decoded_audio, latent_slice
        return decoded_video, decoded_audio
