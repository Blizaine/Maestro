"""
Lightricks-native RetakePipeline for LTX-2.

Single-stage pipeline that regenerates a temporal region of an existing video
while preserving the rest. Uses denoise_mask conditioning (not MaskInjection)
for native integration with the diffusion process.

Key mechanism:
- TemporalRegionMask sets denoise_mask=0 for preserved tokens, =1 for retake region
- GaussianNoiser only adds noise where denoise_mask=1
- timesteps_from_mask() gives timestep=0 for frozen tokens
- post_process_latent() blends denoised output with clean source at every step
"""

import logging
import os
import torch
from collections.abc import Callable, Iterator

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.guiders import CFGGuider
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.conditioning.types.temporal_mask import TemporalRegionMask
from ..ltx_core.conditioning.types.spatial_mask import SpatialRegionMask
from ..ltx_core.conditioning.types.latent_cond import AudioConditionByLatent
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.video_vae import TilingConfig
from ..ltx_core.model.video_vae import decode_video_to_tensor as vae_decode_video_to_tensor
from ..ltx_core.text_encoders.gemma import encode_text, postprocess_text_embeddings, resolve_text_connectors
from ..ltx_core.tools import VideoLatentTools
from ..ltx_core.types import LatentState, VideoPixelShape
from .utils.constants import DISTILLED_SIGMA_VALUES
from .utils.helpers import (
    denoise_audio_video,
    euler_denoising_loop,
    guider_denoising_func,
    image_conditionings_by_adding_guiding_latent,
    load_video_conditioning,
    simple_denoising_func,
    vae_encode_video,
    bind_interrupt_check,
    cleanup_memory,
)
from shared.utils.loras_mutipliers import update_loras_slists
from .utils.types import PipelineComponents
from shared.utils.text_encoder_cache import TextEncoderCache

log = logging.getLogger("retake_pipeline")


class RetakePipeline:
    """
    Native retake pipeline using denoise_mask conditioning.

    Processes a source video clip by encoding it, marking a temporal region
    for regeneration via TemporalRegionMask, then running a single-stage
    diffusion pass. Preserved regions pass through with minimal quality loss
    thanks to the native denoise_mask/clean_latent blending.
    """

    def __init__(self, models: object, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.models = models
        self.device = device
        self.dtype = dtype
        self.pipeline_components = PipelineComponents(dtype=dtype, device=device)
        self.pipeline_components._pipeline_name = 'retake_native'
        self.text_encoder_cache = TextEncoderCache()

    def _get_model(self, name: str):
        return getattr(self.models, name)

    def __call__(
        self,
        source_video_path: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        start_frame: int = 0,
        end_frame: int = -1,
        *,
        tiling_config: TilingConfig | None = None,
        callback: Callable[..., None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
        text_connectors: dict | None = None,
        images: list[tuple] | None = None,
        loras_slists: dict | None = None,
        regenerate_audio: bool = True,
        source_audio_path: str | None = None,
        spatial_mask_path: str | None = None,
        # CFG for prompt-guided retake. At 1.0 the pipeline runs a single
        # unconditional pass (legacy behavior) — which is fine for content-
        # preserving regenerations but makes prompt-driven inpainting (e.g.
        # "woman is wearing armor") produce output nearly identical to the
        # source, because the masked latent tokens get regenerated without
        # any push toward the prompt. At >1.0 the pipeline runs both a
        # positive (prompt) and a negative (negative_prompt) pass and uses
        # classifier-free guidance: out = uncond + scale * (cond - uncond).
        negative_prompt: str = "",
        cfg_guidance_scale: float = 1.0,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        """
        Run retake on a source video clip.

        Args:
            source_video_path: Path to the source video clip
            prompt: Text prompt for the retake region
            seed: Random seed
            height, width, num_frames, frame_rate: Output dimensions (should match source)
            start_frame: Pixel-space start frame of retake region (0 = full retake)
            end_frame: Pixel-space end frame of retake region (-1 = to end)
            images: Optional image conditionings (start/end frame keyframes)
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        if end_frame < 0:
            end_frame = num_frames

        # ── Text Encoding ───────────────────────────────────────────────
        text_encoder = self._get_model("text_encoder")
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(
            text_encoder, text_connectors
        )
        encode_fn = lambda prompts: postprocess_text_embeddings(
            encode_text(text_encoder, prompts=prompts),
            feature_extractor,
            video_connector,
            audio_connector,
        )
        # Encode positive prompt + (if CFG enabled) negative prompt so the
        # guider_denoising_func can subtract the unconditional pass.
        _use_cfg = bool(cfg_guidance_scale is not None and float(cfg_guidance_scale) > 1.0 + 1e-6)
        _prompts_to_encode = [prompt] + ([negative_prompt or ""] if _use_cfg else [])
        contexts = self.text_encoder_cache.encode(encode_fn, _prompts_to_encode, device=self.device, parallel=True)
        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()
        video_context, audio_context = contexts[0]
        video_context_neg = audio_context_neg = None
        if _use_cfg and len(contexts) > 1:
            video_context_neg, audio_context_neg = contexts[1]

        # ── Source Video Encoding ────────────────────────────────────────
        video_encoder = self._get_model("video_encoder")

        # Load and encode the source video clip
        source_video_tensor = load_video_conditioning(
            video_path=source_video_path,
            height=height,
            width=width,
            frame_cap=num_frames,
            dtype=self.dtype,
            device=self.device,
        )
        source_video_latent = vae_encode_video(source_video_tensor, video_encoder, tiling_config)
        source_video_latent = source_video_latent.to(device=self.device, dtype=self.dtype)

        # ── Source Audio Encoding ────────────────────────────────────────
        source_audio_latent = None
        audio_conds = []
        if source_audio_path and os.path.isfile(source_audio_path):
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(source_audio_path)
                # waveform: [channels, samples]
                log.info(f"Loaded audio: {waveform.shape}, sr={sample_rate}")

                # Add batch dim: [1, channels, samples]
                waveform = waveform.unsqueeze(0)

                audio_encoder = self._get_model("audio_encoder")
                target_channels = int(getattr(audio_encoder, "in_channels", waveform.shape[1]))
                if target_channels <= 0:
                    target_channels = waveform.shape[1]
                if waveform.shape[1] != target_channels:
                    if target_channels == 1:
                        waveform = waveform.mean(dim=1, keepdim=True)
                    elif waveform.shape[1] == 1 and target_channels > 1:
                        waveform = waveform.repeat(1, target_channels, 1)
                    else:
                        waveform = waveform[:, :target_channels, :]

                from ..ltx_core.model.audio_vae.ops import AudioProcessor
                audio_processor = AudioProcessor(
                    sample_rate=audio_encoder.sample_rate,
                    mel_bins=audio_encoder.mel_bins,
                    mel_hop_length=audio_encoder.mel_hop_length,
                    n_fft=audio_encoder.n_fft,
                )
                waveform_cpu = waveform.to(device="cpu", dtype=torch.float32)
                audio_processor = audio_processor.to("cpu")
                mel = audio_processor.waveform_to_mel(waveform_cpu, sample_rate)
                mel = mel.to(device=self.device, dtype=self.dtype)
                source_audio_latent = audio_encoder(mel)
                source_audio_latent = source_audio_latent.to(device=self.device, dtype=self.dtype)

                # Pad/trim audio latent to match expected duration
                output_shape_for_audio = VideoPixelShape(
                    batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
                )
                from ..ltx_core.types import AudioLatentShape
                target_audio_shape = AudioLatentShape.from_video_pixel_shape(output_shape_for_audio)
                target_frames = target_audio_shape.frames
                if source_audio_latent.shape[2] < target_frames:
                    pad = torch.zeros(
                        source_audio_latent.shape[0], source_audio_latent.shape[1],
                        target_frames - source_audio_latent.shape[2],
                        source_audio_latent.shape[3],
                        device=source_audio_latent.device, dtype=source_audio_latent.dtype,
                    )
                    source_audio_latent = torch.cat([source_audio_latent, pad], dim=2)
                elif source_audio_latent.shape[2] > target_frames:
                    source_audio_latent = source_audio_latent[:, :, :target_frames, :]

                log.info(f"Source audio encoded: {source_audio_latent.shape} (target: {target_frames} frames)")

                if not regenerate_audio:
                    audio_conds = [AudioConditionByLatent(source_audio_latent, strength=1.0)]
                    log.info("Audio mode: PRESERVE source (denoise_mask=0, frozen)")
                else:
                    log.info("Audio mode: REGENERATE (audio as initial latent, fully denoised)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                log.warning(f"Could not encode source audio: {e}")

        log.info(f"Source video encoded: {source_video_latent.shape}, "
                 f"retake region: frames {start_frame}-{end_frame}/{num_frames}")

        # ── Compute Latent-Space Frame Indices ───────────────────────────
        scale_factors = self.pipeline_components.video_scale_factors
        latent_stride = int(getattr(scale_factors, "time", scale_factors[0]))

        def _pixel_to_latent(frame_idx: int) -> int:
            if frame_idx <= 0:
                return 0
            return (frame_idx - 1) // latent_stride + 1

        latent_start = _pixel_to_latent(start_frame)
        latent_end = _pixel_to_latent(end_frame)

        # ── Build Conditionings ──────────────────────────────────────────
        # Spatial mask (from SAM inpaint) takes priority over temporal mask
        if spatial_mask_path and os.path.isfile(spatial_mask_path):
            import numpy as np
            sam_mask = np.load(spatial_mask_path)  # [T, H_px, W_px] bool
            pixel_mask = torch.from_numpy(sam_mask.astype(np.float32))
            print(f"[Retake] Using spatial mask: {pixel_mask.shape} "
                  f"({sam_mask.sum() / sam_mask.size * 100:.1f}% masked)")
            conditionings = [
                SpatialRegionMask(
                    source_latent=source_video_latent,
                    pixel_mask=pixel_mask,
                    scale_factors=scale_factors,
                )
            ]
        else:
            conditionings = [
                TemporalRegionMask(
                    source_latent=source_video_latent,
                    start_frame=latent_start,
                    end_frame=latent_end,
                )
            ]

        # Add image conditionings (start/end frame keyframes for visual anchoring)
        output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
        )
        if images:
            img_conds = image_conditionings_by_adding_guiding_latent(
                images=images,
                height=height,
                width=width,
                video_encoder=video_encoder,
                dtype=self.dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
            conditionings.extend(img_conds)

        # ── Transformer + Denoising ──────────────────────────────────────
        transformer = self._get_model("transformer")
        bind_interrupt_check(transformer, interrupt_check)

        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        # Apply LoRAs if provided
        if loras_slists is not None:
            num_steps = len(sigmas) - 1
            update_loras_slists(
                transformer, loras_slists, num_steps,
                phase_switch_step=num_steps, phase_switch_step2=num_steps,
            )

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(sigmas) - 1, pass_no=1)

        # Build the denoising function once outside the loop so CFG guiders
        # get constructed with the right scale.
        if _use_cfg and video_context_neg is not None:
            # Video path uses CFG at the user-specified scale; audio stays
            # at CFG=1.0 (audio guidance is usually set separately).
            _video_cfg_guider = CFGGuider(float(cfg_guidance_scale))
            _audio_cfg_guider = CFGGuider(1.0)
            _denoise_fn = guider_denoising_func(
                video_guider=_video_cfg_guider,
                audio_guider=_audio_cfg_guider,
                v_context_p=video_context,
                v_context_n=video_context_neg,
                a_context_p=audio_context,
                a_context_n=audio_context_neg if audio_context_neg is not None else audio_context,
                transformer=transformer,
                alt_guidance_scale=1.0,
            )
            # Print (not log.info) so users see it without raising log level.
            # Distilled LTX-2.3's positive and negative predictions tend to be
            # near-identical (model trained for 1-step unconditional inference),
            # so CFG alone typically doesn't drive visible prompt-directed
            # changes — a strong negative_prompt is what makes CFG actually
            # do work here.
            print(f"[Retake] CFG guided denoising: scale={cfg_guidance_scale:.2f}, "
                  f"neg_prompt={'(set)' if negative_prompt else '(empty)'}")
        else:
            _denoise_fn = simple_denoising_func(
                video_context=video_context,
                audio_context=audio_context,
                transformer=transformer,
                alt_guidance_scale=1.0,
            )

        def denoising_loop(
            sigmas_arg, video_state, audio_state, stepper_arg,
            preview_tools=None, mask_context=None,
        ):
            return euler_denoising_loop(
                sigmas=sigmas_arg,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper_arg,
                denoise_fn=_denoise_fn,
                mask_context=None,  # Native retake: no MaskInjection
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=1,
                transformer=transformer,
            )

        # Run denoising — denoise_audio_video handles state creation, noising, and cleanup
        # Always pass source audio as initial latent when available:
        # - Preserve mode: conditioning freezes it (denoise_mask=0), output = source audio
        # - Regenerate mode: no conditioning (denoise_mask=1), full denoising from source
        #   structure — retains voice/tone characteristics while generating new content
        audio_init = source_audio_latent
        log.info(f"Denoising setup: audio_init={'yes' if audio_init is not None else 'no'}, "
                 f"audio_conds={len(audio_conds)}, regenerate_audio={regenerate_audio}")

        video_state, audio_state = denoise_audio_video(
            output_shape=output_shape,
            conditionings=conditionings,
            audio_conditionings=audio_conds or None,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
            initial_video_latent=source_video_latent,
            initial_audio_latent=audio_init,
        )

        del transformer
        cleanup_memory()

        # ── Decode Video ─────────────────────────────────────────────────
        if video_state is None:
            return iter([]), torch.zeros(0)

        video_decoder = self._get_model("video_decoder")
        video_chunks = vae_decode_video_to_tensor(
            video_state.latent,
            video_decoder,
            tiling_config,
            expected_frames=int(output_shape.frames),
            expected_height=int(output_shape.height),
            expected_width=int(output_shape.width),
            interrupt_check=interrupt_check,
        )

        # ── Decode Audio ─────────────────────────────────────────────────
        if audio_state is not None and audio_state.latent is not None:
            try:
                audio_decoder = self._get_model("audio_decoder")
                vocoder = self._get_model("vocoder")
                print(f"[Retake Audio] Decoding audio: latent shape={audio_state.latent.shape}")
                audio_tensor = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)
                print(f"[Retake Audio] Decoded: shape={audio_tensor.shape}, dtype={audio_tensor.dtype}")
            except Exception as e:
                print(f"[Retake Audio] Decode failed: {e}")
                import traceback as _tb
                _tb.print_exc()
                audio_tensor = None
        else:
            print(f"[Retake Audio] No audio state (audio_state is None: {audio_state is None})")
            audio_tensor = None

        return video_chunks, audio_tensor
