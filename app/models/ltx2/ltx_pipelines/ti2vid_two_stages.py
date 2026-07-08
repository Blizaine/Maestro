import logging
from collections.abc import Callable, Iterator

import torch

from ..ltx_core.components.diffusion_steps import (
    DPMSolverPlusPlus2MDiffusionStep,
    EulerAncestralDiffusionStep,
    EulerDiffusionStep,
)
from ..ltx_core.components.text_attention_amplifier import (
    install_text_amplifier,
    remove_text_amplifier,
)
from ..ltx_core.components.guiders import CFGGuider, CFGStarRescalingGuider, LtxAPGGuider
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.components.schedulers import LTX2Scheduler
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.upsampler import upsample_video
from ..ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ..ltx_core.model.video_vae import decode_video_to_tensor as vae_decode_video_to_tensor
from ..ltx_core.text_encoders.gemma import encode_text, postprocess_text_embeddings, resolve_text_connectors
from ..ltx_core.tools import VideoLatentTools
from ..ltx_core.types import LatentState, VideoPixelShape
from .utils import ModelLedger
from .utils.args import default_2_stage_arg_parser
from .utils.constants import (
    AUDIO_SAMPLE_RATE,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from .utils.helpers import (
    assert_resolution,
    bind_interrupt_check,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    gradient_estimating_euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    guider_denoising_func,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    latent_conditionings_by_latent_sequence,
    prepare_mask_injection,
    simple_denoising_func,
    video_conditionings_by_keyframe,
    video_conditionings_by_reference_latent,
)
from .utils.media_io import encode_video
from .utils.types import PipelineComponents
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.self_refiner import create_self_refiner_handler, normalize_self_refiner_plan
from shared.utils.text_encoder_cache import TextEncoderCache

device = get_device()


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output. Supports optional image conditioning via the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        distilled_lora: list[LoraPathStrengthAndSDOps] | None = None,
        spatial_upsampler_path: str | None = None,
        gemma_root: str | None = None,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: str = device,
        fp8transformer: bool = False,
        model_device: torch.device | None = None,
        stage_1_models: object | None = None,
        stage_2_models: object | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.stage_1_models = stage_1_models
        self.stage_2_models = stage_2_models or stage_1_models
        if self.stage_1_models is None:
            if checkpoint_path is None or gemma_root is None or spatial_upsampler_path is None:
                raise ValueError("checkpoint_path, gemma_root, and spatial_upsampler_path are required.")
            ledger_device = model_device or device
            distilled_lora = distilled_lora or []
            self.stage_1_model_ledger = ModelLedger(
                dtype=self.dtype,
                device=ledger_device,
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
                spatial_upsampler_path=spatial_upsampler_path,
                loras=loras or [],
                fp8transformer=fp8transformer,
            )

            if distilled_lora:
                self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
                    loras=distilled_lora,
                )
            else:
                self.stage_2_model_ledger = self.stage_1_model_ledger
        else:
            self.stage_1_model_ledger = None
            self.stage_2_model_ledger = None

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self.text_encoder_cache = TextEncoderCache()

    def _get_stage_model(self, stage: int, name: str):
        if stage == 1:
            models = self.stage_1_models
            ledger = self.stage_1_model_ledger
        else:
            models = self.stage_2_models
            ledger = self.stage_2_model_ledger
        if models is not None:
            return getattr(models, name)
        if ledger is None:
            raise ValueError(f"Missing model source for stage {stage} '{name}'.")
        return getattr(ledger, name)()

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
        audio_cfg_guidance_scale: float | None = None,
        cfg_star_switch: int = 0,
        apg_switch: int = 0,
        perturbation_switch: int = 0,
        perturbation_layers: list[int] | None = None,
        perturbation_start: float = 0.0,
        perturbation_end: float = 1.0,
        alt_guidance_scale: float = 1.0,
        alt_scale: float = 0.0,
        end_images: list[tuple] | None = None,
        end_images_stage2: list[tuple] | None = None,
        inject_images: list[tuple] | None = None,
        inject_images_stage2: list[tuple] | None = None,
        images_stage2: list[tuple[str, int, float]] | None = None,
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
        stg_scale: float = 1.0,
        cfg_rescale: float = 0.0,
        modality_scale: float = 1.0,
        use_gradient_estimation: bool = False,
        ge_gamma: float = 2.0,
        keyframe_conditioning_mode: str = "replace",
        keyframe_inject_mode: str = "additive",
        sample_solver: str = "euler",
        stg_schedule: list[float] | None = None,
        text_attention_amplifier: dict | None = None,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        mask_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        noiser = GaussianNoiser(generator=generator)
        # Sampler selection — was hardcoded to plain Euler, which produces
        # noticeably more motion noise than Euler Ancestral on community
        # fine-tunes (notably TenStrip/LTX2.3-10Eros, whose reference
        # ComfyUI workflow specifies `euler_ancestral`). Driven by the
        # `sample_solver` field on the model_def (settable in the model
        # JSON as a top-level setting default; wgp.py already routes
        # configs["sample_solver"] into the generate() kwargs).
        # Backward compat: unknown / empty / "euler" → plain Euler.
        _solver = (sample_solver or "euler").lower()
        if _solver in ("euler_ancestral", "euler_a", "ancestral"):
            ancestral_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 2)
            stepper = EulerAncestralDiffusionStep(generator=ancestral_generator, eta=1.0)
        elif _solver in ("euler_ancestral_cfg_pp", "ancestral_cfg_pp"):
            ancestral_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 2)
            # eta=0.25 matches the ComfyUI ClownSampler convention referenced
            # by the LTX-2 distilled pipeline (see distilled.py:250).
            stepper = EulerAncestralDiffusionStep(generator=ancestral_generator, eta=0.25)
        elif _solver in ("dpm++_2m", "dpmpp_2m", "dpm++2m", "dpm_pp_2m"):
            stepper = DPMSolverPlusPlus2MDiffusionStep()
        else:
            stepper = EulerDiffusionStep()
        self_refiner_handler = None
        self_refiner_handler_audio = None
        self_refiner_handler_stage2 = None
        self_refiner_handler_audio_stage2 = None
        if self_refiner_setting and self_refiner_setting > 0:
            plans, _ = normalize_self_refiner_plan(self_refiner_plan or "", max_plans=self_refiner_max_plans)
            plan_stage1 = plans[0] if plans else []
            plan_stage2 = plans[1] if len(plans) > 1 else []
            self_refiner_handler = create_self_refiner_handler(
                plan_stage1,
                self_refiner_f_uncertainty,
                self_refiner_setting,
                self_refiner_certain_percentage,
                channel_dim=-1,
            )
            self_refiner_handler_audio = create_self_refiner_handler(
                plan_stage1,
                self_refiner_f_uncertainty,
                self_refiner_setting,
                self_refiner_certain_percentage,
                channel_dim=-1,
            )
            if plan_stage2:
                self_refiner_handler_stage2 = create_self_refiner_handler(
                    plan_stage2,
                    self_refiner_f_uncertainty,
                    self_refiner_setting,
                    self_refiner_certain_percentage,
                    channel_dim=-1,
                )
                self_refiner_handler_audio_stage2 = create_self_refiner_handler(
                    plan_stage2,
                    self_refiner_f_uncertainty,
                    self_refiner_setting,
                    self_refiner_certain_percentage,
                    channel_dim=-1,
                )
        audio_cfg_guidance_scale = cfg_guidance_scale if audio_cfg_guidance_scale is None else audio_cfg_guidance_scale
        guider_cls = CFGGuider
        if apg_switch:
            guider_cls = LtxAPGGuider
        elif cfg_star_switch:
            guider_cls = CFGStarRescalingGuider
        video_cfg_guider = guider_cls(cfg_guidance_scale)
        audio_cfg_guider = guider_cls(audio_cfg_guidance_scale)
        dtype = torch.bfloat16

        text_encoder = self._get_stage_model(1, "text_encoder")
        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
        # Codex: needs to return only the text embeddings from the text encoder for all the prompts
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(
            text_encoder, text_connectors
        )
        encode_fn = lambda prompts: postprocess_text_embeddings(
            encode_text(text_encoder, prompts=prompts),
            feature_extractor,
            video_connector,
            audio_connector,
        )
        contexts = self.text_encoder_cache.encode(
            encode_fn,
            [prompt, negative_prompt],
            device=self.device,
            parallel=True,
        )

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()
        # Codex: now that the text encoder has been released, compute the text_embedding_projection,
        # audio_embeddings_connector, video_embeddings_connector in order to get v_context_p, a_context_p
        # and v_context_n, a_context_n
        context_p, context_n = contexts
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        # Stage 1: Initial low resolution video generation.
        video_encoder = self._get_stage_model(1, "video_encoder")
        transformer = self._get_stage_model(1, "transformer")
        bind_interrupt_check(transformer, interrupt_check)

        # Install text-cross-attention amplifier hooks for the lifetime
        # of this generate call. Driven by model_def's
        # `text_attention_amplifier` block. Returns [] when no config is
        # present (strength=1.0 path) — completely no-op for non-10Eros
        # models. Hooks are torn down in the explicit cleanup just before
        # `del transformer` further down, ensuring the transformer is
        # left clean for any subsequent calls (it's a long-lived module
        # cached by the model ledger).
        _amp_handles: list = []
        if text_attention_amplifier:
            try:
                _amp_handles = install_text_amplifier(
                    transformer,
                    strength=float(text_attention_amplifier.get("strength", 1.3)),
                    block_filter=text_attention_amplifier.get("block_filter")
                        or text_attention_amplifier.get("block_index_filter"),
                    spatial_focus=float(text_attention_amplifier.get("spatial_focus", 0.0)),
                    latent_shape=None,  # uniform-only for now; spatial focus needs latent
                                        # shape capture which is a follow-up.
                    debug=bool(text_attention_amplifier.get("debug", False)),
                )
            except Exception as _amp_err:
                print(f"[10S TextAmp] install failed (continuing without amplifier): {_amp_err}")
                _amp_handles = []
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
        if loras_slists is not None:
            stage_1_steps = len(sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_1_steps,
                phase_switch_step=stage_1_steps,
                phase_switch_step2=stage_1_steps,
            )

        if callback is not None:
            # Pre-declare total across both stages so the progress bar doesn't
            # fill to 100% at end of stage 1 and then jump back. Stage 2 uses
            # the fixed STAGE_2_DISTILLED_SIGMA_VALUES schedule.
            _pipeline_total_steps = (len(sigmas) - 1) + (len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1)
            callback(-1, None, True, override_num_inference_steps=len(sigmas) - 1,
                     pass_no=1, total_steps_hint=_pipeline_total_steps)

        def first_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            preview_tools: VideoLatentTools | None = None,
            mask_context=None,
        ) -> tuple[LatentState, LatentState]:
            denoise_fn = guider_denoising_func(
                video_cfg_guider,
                audio_cfg_guider,
                v_context_p,
                v_context_n,
                a_context_p,
                a_context_n,
                transformer=transformer,  # noqa: F821
                alt_guidance_scale=alt_guidance_scale,
                alt_scale=alt_scale,
                perturbation_switch=perturbation_switch,
                perturbation_layers=perturbation_layers,
                perturbation_start=perturbation_start,
                perturbation_end=perturbation_end,
                stg_scale=stg_scale,
                cfg_rescale=cfg_rescale,
                modality_scale=modality_scale,
                stg_schedule=stg_schedule,
                # ID-LoRA: read identity guidance scale from pipeline
                # components (set by ltx2.py when voice_reference is
                # provided). Default 0.0 = inactive. Dev (two-stage) loads
                # the ID-LoRA with "1;0" multiplier (phase 1 only), so
                # this is the only place we activate it.
                audio_identity_guidance_scale=float(getattr(self.pipeline_components, 'identity_guidance_scale', 0.0) or 0.0),
            )
            if use_gradient_estimation:
                # Gradient estimation loop: allows fewer steps (20-25 instead of 40)
                # Note: incompatible with self-refiner (different function signature)
                return gradient_estimating_euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=denoise_fn,
                    ge_gamma=ge_gamma,
                    mask_context=mask_context,
                    interrupt_check=interrupt_check,
                    callback=callback,
                    preview_tools=preview_tools,
                    pass_no=1,
                )
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=denoise_fn,
                mask_context=mask_context,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=1,
                transformer=transformer,
                self_refiner_handler=self_refiner_handler,
                self_refiner_handler_audio=self_refiner_handler_audio,
                self_refiner_generator=generator,
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        _kf_se_fn = image_conditionings_by_adding_guiding_latent if keyframe_conditioning_mode == "additive" else image_conditionings_by_replacing_latent
        _kf_inject_fn = image_conditionings_by_adding_guiding_latent if keyframe_inject_mode == "additive" else image_conditionings_by_replacing_latent
        stage_1_conditionings = _kf_se_fn(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
        )
        if end_images:
            stage_1_conditionings += _kf_se_fn(
                images=end_images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        if inject_images:
            stage_1_conditionings += _kf_inject_fn(
                images=inject_images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        if video_conditioning:
            if int(video_conditioning_downscale_factor or 1) > 1:
                stage_1_conditionings += video_conditionings_by_reference_latent(
                    video_conditioning=video_conditioning,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    num_frames=num_frames,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=self.device,
                    downscale_factor=video_conditioning_downscale_factor,
                    tiling_config=tiling_config,
                )
            else:
                stage_1_conditionings += video_conditionings_by_keyframe(
                    video_conditioning=video_conditioning,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    num_frames=num_frames,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=self.device,
                    tiling_config=tiling_config,
                )
        mask_context = prepare_mask_injection(
            masking_source=masking_source,
            masking_strength=masking_strength,
            output_shape=stage_1_output_shape,
            video_encoder=video_encoder,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
            generator=mask_generator,
            num_steps=len(sigmas) - 1,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            audio_conditionings=audio_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            mask_context=mask_context,
        )
        if video_state is None or audio_state is None:
            # Early return between stage 1 and stage 2 — tear down amplifier
            # hooks installed on the stage 1 transformer so it stays clean.
            if _amp_handles:
                remove_text_amplifier(_amp_handles)
                _amp_handles = []
            return None, None
        if interrupt_check is not None and interrupt_check():
            if _amp_handles:
                remove_text_amplifier(_amp_handles)
                _amp_handles = []
            return None, None

        torch.cuda.synchronize()
        # Amplifier hooks were installed on the stage 1 transformer;
        # remove BEFORE we drop our reference so the cached module is
        # clean. If stage 2 uses the same transformer (the common case
        # where stage_1_models == stage_2_models), we re-install below.
        if _amp_handles:
            remove_text_amplifier(_amp_handles)
            _amp_handles = []
        del transformer
        cleanup_memory()

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self._get_stage_model(2, "spatial_upsampler"),
        )

        torch.cuda.synchronize()
        cleanup_memory()

        transformer = self._get_stage_model(2, "transformer")
        bind_interrupt_check(transformer, interrupt_check)

        # Re-install amplifier hooks for stage 2 (the upscale-refinement
        # pass that the upstream node was specifically designed for —
        # "compensates for conditioning dilution at upscaled token counts").
        # If stage_1_models == stage_2_models, this is the same module
        # we just cleaned; install fresh.
        if text_attention_amplifier:
            try:
                _amp_handles = install_text_amplifier(
                    transformer,
                    strength=float(text_attention_amplifier.get("strength", 1.3)),
                    block_filter=text_attention_amplifier.get("block_filter")
                        or text_attention_amplifier.get("block_index_filter"),
                    spatial_focus=float(text_attention_amplifier.get("spatial_focus", 0.0)),
                    latent_shape=None,  # uniform-only path; spatial focus is a follow-up
                    debug=bool(text_attention_amplifier.get("debug", False)),
                )
            except Exception as _amp_err:
                print(f"[10S TextAmp] stage-2 install failed (continuing without): {_amp_err}")
                _amp_handles = []

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        if loras_slists is not None:
            stage_2_steps = len(distilled_sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_2_steps,
                phase_switch_step=0,
                phase_switch_step2=stage_2_steps,
            )

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(distilled_sigmas) - 1, pass_no=2)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            preview_tools: VideoLatentTools | None = None,
            mask_context=None,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                    alt_guidance_scale=1.0,
                ),
                mask_context=mask_context,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=2,
                transformer=transformer,
                self_refiner_handler=self_refiner_handler_stage2,
                self_refiner_handler_audio=self_refiner_handler_audio_stage2,
                self_refiner_generator=generator,
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_images = images if images_stage2 is None else images_stage2
        stage_2_conditionings = _kf_se_fn(
            images=stage_2_images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
        )
        if end_images_stage2:
            stage_2_conditionings += _kf_se_fn(
                images=end_images_stage2,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        if inject_images_stage2:
            stage_2_conditionings += _kf_inject_fn(
                images=inject_images_stage2,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        if latent_conditioning_stage2 is not None:
            stage_2_conditionings += latent_conditionings_by_latent_sequence(
                latent_conditioning_stage2,
                strength=1.0,
                start_index=0,
            )
        mask_context = prepare_mask_injection(
            masking_source=masking_source,
            masking_strength=masking_strength,
            output_shape=stage_2_output_shape,
            video_encoder=video_encoder,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
            generator=mask_generator,
            num_steps=len(distilled_sigmas) - 1,
        )
        # ID-LoRA: clear ref-audio conditioning before stage 2. The
        # ID-LoRA has multiplier 0 on stage 2 ("1;0" convention), so the
        # model lacks the LoRA-trained weights to interpret prepended
        # ref tokens here — leaving them in confuses the audio
        # prediction. Mirrors WanGP's `audio_conditionings_stage2 = []`
        # gate when ID-LoRA is active. ltx2.py resets these on the next
        # generation's stage 1 so this is a safe transient clear.
        _saved_ref_latent = getattr(self.pipeline_components, 'ref_audio_latent', None)
        _saved_ref_positions = getattr(self.pipeline_components, 'ref_audio_positions', None)
        _saved_id_scale = getattr(self.pipeline_components, 'identity_guidance_scale', 0.0)
        if _saved_ref_latent is not None:
            self.pipeline_components.ref_audio_latent = None
            self.pipeline_components.ref_audio_positions = None
            self.pipeline_components.identity_guidance_scale = 0.0
            print(f"[ID-LoRA] Cleared ref-audio conditioning before stage 2 "
                  f"(LoRA multiplier is 0 on stage 2)")
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            audio_conditionings=audio_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            mask_context=mask_context,
        )
        # Restore so the caller's view of components stays consistent
        # if anyone introspects them post-generation.
        if _saved_ref_latent is not None:
            self.pipeline_components.ref_audio_latent = _saved_ref_latent
            self.pipeline_components.ref_audio_positions = _saved_ref_positions
            self.pipeline_components.identity_guidance_scale = _saved_id_scale
        if video_state is None or audio_state is None:
            # Early return — clean up any installed amplifier hooks so the
            # transformer (long-lived in the ledger cache) doesn't carry
            # stale hooks into the next generate() call.
            if _amp_handles:
                remove_text_amplifier(_amp_handles)
                _amp_handles = []
            return None, None
        if interrupt_check is not None and interrupt_check():
            if _amp_handles:
                remove_text_amplifier(_amp_handles)
                _amp_handles = []
            return None, None

        torch.cuda.synchronize()
        # Tear down amplifier hooks BEFORE releasing the transformer ref
        # so we leave the underlying module clean for the next call.
        if _amp_handles:
            remove_text_amplifier(_amp_handles)
            _amp_handles = []
        del transformer
        del video_encoder
        cleanup_memory()

        latent_slice = None
        if return_latent_slice is not None:
            latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
        decoded_video = vae_decode_video_to_tensor(
            video_state.latent,
            self._get_stage_model(2, "video_decoder"),
            tiling_config,
            expected_frames=int(stage_2_output_shape.frames),
            expected_height=int(stage_2_output_shape.height),
            expected_width=int(stage_2_output_shape.width),
            interrupt_check=interrupt_check,
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self._get_stage_model(2, "audio_decoder"), self._get_stage_model(2, "vocoder")
        )
        if latent_slice is not None:
            return decoded_video, decoded_audio, latent_slice
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_guidance_scale=args.cfg_guidance_scale,
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
