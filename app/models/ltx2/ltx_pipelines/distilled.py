import logging
import os
import time
from collections.abc import Callable, Iterator

import torch
import torch.nn.functional as F

from ..inpainting import (
    _apply_ltx2_mask_blend,
    _edge_extend_ltx2_masked_control_video,
)
from ..ltx_core.components.diffusion_steps import (
    DPMSolverPlusPlus2MDiffusionStep,
    EulerAncestralDiffusionStep,
    EulerDiffusionStep,
    LTX25EulerAncestralDiffusionStep,
)
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.upsampler import upsample_video
from ..ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ..ltx_core.model.video_vae import decode_video_to_tensor as vae_decode_video_to_tensor
from ..ltx_core.model.video_vae import encode_video as vae_encode_video
from ..ltx_core.text_encoders.gemma import encode_text, postprocess_text_embeddings, resolve_text_connectors
from ..ltx_core.tools import VideoLatentTools
from ..ltx_core.types import LatentState, VideoPixelShape
from .utils import ModelLedger
from .utils.args import default_2_stage_distilled_arg_parser
from .utils.constants import (
    AUDIO_SAMPLE_RATE,
    DISTILLED_SIGMA_VALUES,
    OUTPAINT_ATTENTION_STAGE_2_SIGMA_VALUES,
    OUTPAINT_FULL_RES_REFINE_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
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
from .utils.media_io import encode_video, load_video_conditioning
from .utils.types import PipelineComponents
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.self_refiner import create_self_refiner_handler, normalize_self_refiner_plan
from shared.utils.text_encoder_cache import TextEncoderCache

device = get_device()
_BENCH_TRANSFORMER_ENV = "WAN2GP_LTX2_BENCH_TRANSFORMER"


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _select_reference_attention_generation_mask(
    full_resolution_refine: bool,
    generation_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Select official full attention or compatibility source attention."""
    if full_resolution_refine:
        return None
    return generation_mask


def _align_seq_len(tensor: torch.Tensor | None, target_len: int) -> torch.Tensor | None:
    if tensor is None:
        return tensor
    seq_dim = 0 if tensor.dim() == 2 else 1
    cur_len = tensor.shape[seq_dim]
    if cur_len == target_len:
        return tensor
    if cur_len < target_len:
        pad_len = target_len - cur_len
        if seq_dim == 0:
            pad = tensor[-1:].repeat(pad_len, 1)
            return torch.cat([tensor, pad], dim=0)
        pad = tensor[:, -1:, :].repeat(1, pad_len, 1)
        return torch.cat([tensor, pad], dim=1)
    return tensor.narrow(seq_dim, 0, target_len)


def _coerce_refinement_source_cthw(
    video_conditioning: list[tuple] | None,
    *,
    height: int,
    width: int,
    num_frames: int,
    generation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover the first IC-LoRA reference as normalized CTHW pixels."""
    if not video_conditioning:
        raise ValueError(
            "Pixel-refined Outpaint requires video conditioning."
        )
    entry = video_conditioning[0]
    if not isinstance(entry, (tuple, list)) or not entry:
        raise ValueError("Invalid Outpaint video-conditioning entry.")
    source_input = entry[0]

    if torch.is_tensor(source_input):
        source = source_input.detach()
        if source.dim() == 5:
            if int(source.shape[0]) != 1:
                raise ValueError(
                    "Outpaint refinement supports one conditioning batch."
                )
            source = source[0]
        if source.dim() == 4:
            if int(source.shape[0]) in (1, 3, 4):
                source = source[:3]
            elif int(source.shape[-1]) in (1, 3, 4):
                source = source[..., :3].permute(3, 0, 1, 2)
            else:
                source = None
            if source is not None:
                source = _fit_refinement_frames_cthw(
                    source,
                    int(num_frames),
                )
                if generation_mask is not None:
                    # Extend real boundary pixels over the neutral/marker
                    # canvas before area downsampling. Otherwise an odd source
                    # coordinate averages padding into the half-res edge.
                    source_mask = _coerce_refinement_mask_cthw(
                        generation_mask,
                        height=int(source.shape[-2]),
                        width=int(source.shape[-1]),
                        num_frames=int(num_frames),
                    )
                    source = _edge_extend_ltx2_masked_control_video(
                        source,
                        source_mask,
                    )
                return _resize_refinement_video_cthw(
                    source,
                    height=int(height),
                    width=int(width),
                    mode="area",
                )

    # The Outpaint path normally supplies an already prepared CTHW tensor.
    # Retain support for file-backed conditioning and unexpected layouts so
    # the refinement path remains compatible with the generic LTX pipeline.
    loaded = load_video_conditioning(
        video_path=source_input,
        height=int(height),
        width=int(width),
        frame_cap=int(num_frames),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return _fit_refinement_frames_cthw(
        loaded[0, :3],
        int(num_frames),
    )


def _fit_refinement_frames_cthw(
    video: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """Trim or tail-pad a CTHW tensor to the requested frame count."""
    if video.dim() != 4 or int(video.shape[1]) <= 0:
        raise ValueError("Outpaint refinement video must contain frames.")
    num_frames = max(1, int(num_frames))
    if int(video.shape[1]) >= num_frames:
        return video[:, :num_frames]
    missing = num_frames - int(video.shape[1])
    return torch.cat(
        [video, video[:, -1:].expand(-1, missing, -1, -1)],
        dim=1,
    )


def _resize_refinement_video_cthw(
    video: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    """Resize CTHW pixels in bounded CPU chunks without latent upscaling."""
    height = int(height)
    width = int(width)
    if tuple(video.shape[-2:]) == (height, width):
        return video

    source = video.detach().cpu()
    output = torch.empty(
        (
            int(source.shape[0]),
            int(source.shape[1]),
            height,
            width,
        ),
        dtype=source.dtype,
        device="cpu",
    )
    pixels_per_frame = max(1, height * width)
    chunk_frames = max(1, min(16, 8_000_000 // pixels_per_frame))
    for start in range(0, int(source.shape[1]), chunk_frames):
        end = min(int(source.shape[1]), start + chunk_frames)
        frames = (
            source[:, start:end]
            .permute(1, 0, 2, 3)
            .to(dtype=torch.float32)
        )
        if mode == "lanczos":
            # Lightricks' published LTX-2.3 Outpaint graph uses Lanczos for
            # the decoded pass-one -> pass-two pixel handoff. PyTorch does
            # not expose Lanczos through F.interpolate, so use OpenCV's
            # float-preserving implementation in the same bounded chunks.
            import cv2

            resized_frames = []
            for frame in frames:
                frame_hwc = (
                    frame.permute(1, 2, 0)
                    .contiguous()
                    .numpy()
                )
                resized_hwc = cv2.resize(
                    frame_hwc,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                if resized_hwc.ndim == 2:
                    resized_hwc = resized_hwc[..., None]
                resized_frames.append(
                    torch.from_numpy(resized_hwc).permute(2, 0, 1)
                )
            resized = torch.stack(resized_frames)
        elif mode == "nearest":
            resized = F.interpolate(
                frames,
                size=(height, width),
                mode="nearest",
            )
        elif mode == "area":
            resized = F.interpolate(
                frames,
                size=(height, width),
                mode="area",
            )
        else:
            resized = F.interpolate(
                frames,
                size=(height, width),
                mode=mode,
                align_corners=False,
                antialias=True,
            )
        if source.dtype == torch.uint8:
            resized = resized.round().clamp(0.0, 255.0)
        elif source.is_floating_point():
            source_min = float(frames.amin())
            source_max = float(frames.amax())
            resized = resized.clamp(source_min, source_max)
        output[:, start:end] = (
            resized.to(dtype=source.dtype)
            .permute(1, 0, 2, 3)
        )
    return output


def _coerce_refinement_mask_cthw(
    generation_mask: torch.Tensor | None,
    *,
    height: int,
    width: int,
    num_frames: int,
) -> torch.Tensor:
    """Normalize an Outpaint generation mask to 1xTxHxW."""
    if generation_mask is None or not torch.is_tensor(generation_mask):
        raise ValueError(
            "Pixel-refined Outpaint requires a generation mask."
        )
    mask = generation_mask.detach()
    if mask.dim() == 5:
        if int(mask.shape[0]) != 1:
            raise ValueError("Outpaint refinement supports one mask batch.")
        mask = mask[0]
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4 and int(mask.shape[-1]) == 1:
        mask = mask.permute(3, 0, 1, 2)
    if mask.dim() != 4 or int(mask.shape[0]) != 1:
        raise ValueError(
            "Outpaint refinement mask must have shape 1xTxHxW."
        )
    mask = _fit_refinement_frames_cthw(mask, int(num_frames))
    mask = mask.to(device="cpu", dtype=torch.float32)
    if mask.numel() and float(mask.max()) > 1.0:
        mask = mask.div(255.0)
    mask = mask.clamp(0.0, 1.0)
    return _resize_refinement_video_cthw(
        mask,
        height=int(height),
        width=int(width),
        # The official graph uses area sampling. This matters when a source
        # rectangle begins on an odd full-resolution row: its half-resolution
        # boundary is 0.5, not a one-pixel nearest-neighbor shift.
        mode="area",
    )


def _decode_blend_reencode_outpaint(
    *,
    latent: torch.Tensor,
    source: torch.Tensor,
    generation_mask: torch.Tensor,
    video_decoder,
    video_encoder,
    tiling_config: TilingConfig | None,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    target_height: int | None = None,
    target_width: int | None = None,
    interrupt_check: Callable[[], bool] | None = None,
) -> torch.Tensor | None:
    """Build the official pixel-space handoff for Outpaint pass two.

    LTX's two-stage workflow decodes pass one, restores the protected source,
    resizes the cleaned pixels, then VAE-encodes them before adding pass-two
    noise. Passing the first latent through a generic learned spatial upscaler
    distorts Outpaint geometry; denoising pass one at target resolution causes
    the green missing-canvas sentinel to dominate the generated area.
    """
    decoded_fhwc = vae_decode_video_to_tensor(
        latent,
        video_decoder,
        tiling_config,
        expected_frames=int(num_frames),
        expected_height=int(height),
        expected_width=int(width),
        interrupt_check=interrupt_check,
    )
    if decoded_fhwc is None:
        return None
    if interrupt_check is not None and interrupt_check():
        return None

    decoded_cthw = decoded_fhwc.permute(3, 0, 1, 2)
    blended_cthw = _apply_ltx2_mask_blend(
        decoded_cthw,
        source,
        generation_mask,
        int(num_frames),
        int(height),
        int(width),
        # Lightricks uses dilation 5 and a full-frame Laplacian blend on the
        # intermediate handoff.
        mask_low_res_dilation=5,
        source_feather_pixels=24,
        match_generated_canvas=False,
        full_frame_laplacian=True,
    )
    del decoded_fhwc
    del decoded_cthw
    if interrupt_check is not None and interrupt_check():
        return None

    target_height = int(target_height or height)
    target_width = int(target_width or width)
    blended_cthw = _resize_refinement_video_cthw(
        blended_cthw,
        height=target_height,
        width=target_width,
        mode="lanczos",
    )
    if interrupt_check is not None and interrupt_check():
        return None

    refinement_video = blended_cthw.unsqueeze(0).to(
        device=device,
        dtype=dtype,
    )
    if blended_cthw.dtype == torch.uint8:
        refinement_video.div_(127.5).sub_(1.0)
    encoded = vae_encode_video(
        refinement_video,
        video_encoder,
        tiling_config,
    )
    return encoded


class _TransformerBenchWrapper:
    def __init__(self, module, enabled: bool = False) -> None:
        self._module = module
        self._enabled = bool(enabled)
        self._cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_total_ms = 0.0
        self._cpu_calls = 0

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __call__(self, *args, **kwargs):
        if not self._enabled:
            return self._module(*args, **kwargs)
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = self._module(*args, **kwargs)
            end.record()
            self._cuda_events.append((start, end))
            return out

        t0 = time.perf_counter()
        out = self._module(*args, **kwargs)
        self._cpu_total_ms += (time.perf_counter() - t0) * 1000.0
        self._cpu_calls += 1
        return out

    def consume(self) -> tuple[float, int]:
        if not self._enabled:
            return 0.0, 0
        if torch.cuda.is_available():
            if not self._cuda_events:
                return 0.0, 0
            torch.cuda.synchronize()
            total_ms = 0.0
            for start, end in self._cuda_events:
                total_ms += float(start.elapsed_time(end))
            calls = len(self._cuda_events)
            self._cuda_events.clear()
            return total_ms, calls

        total_ms = self._cpu_total_ms
        calls = self._cpu_calls
        self._cpu_total_ms = 0.0
        self._cpu_calls = 0
        return total_ms, calls


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
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
        self.pipeline_components._pipeline_name = 'distilled'
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
        negative_prompt: str = "",
        NAG_scale: float = 1.0,
        NAG_tau: float = 3.5,
        NAG_alpha: float = 0.5,
        end_images: list[tuple] | None = None,
        end_images_stage2: list[tuple] | None = None,
        inject_images: list[tuple] | None = None,
        inject_images_stage2: list[tuple] | None = None,
        alt_guidance_scale: float = 1.0,
        video_conditioning: list[tuple[str, float]] | None = None,
        video_conditioning_downscale_factor: int = 1,
        video_conditioning_generation_mask: torch.Tensor | None = None,
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
        single_stage: bool = False,
        full_resolution_refine: bool = False,
        keyframe_conditioning_mode: str = "replace",
        keyframe_inject_mode: str = "additive",
        use_ancestral_sampler: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        alt_guidance_scale = 1.0
        full_resolution_refine = bool(full_resolution_refine)
        attention_masked_outpaint = bool(
            video_conditioning_generation_mask is not None
            and not full_resolution_refine
        )
        # Single-stage mode: run the distilled denoise at FULL target
        # resolution and skip the stage-2 upscale+refine entirely. Trades
        # more VRAM/time (stage 1 at 4x the pixels) for no upscale artifacts
        # and no refine-time sigma schedule.
        if single_stage:
            stage_1_width = int(width)
            stage_1_height = int(height)
        else:
            stage_1_width = int(width) // 2
            stage_1_height = int(height) // 2

        generator = torch.Generator(device=self.device).manual_seed(seed)
        mask_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        # Match WanGP's native LTX-2.5 noise stream so an identical prompt,
        # seed, and LoRA produces the same ancestral base/refine trajectory.
        ancestral_generator = torch.Generator(device=self.device).manual_seed(
            int(seed) + 10000
        )
        noiser = GaussianNoiser(generator=generator)
        # Single-stage runs all denoising at full target res with no upscale
        # refine pass, so we use DPM-Solver++ 2M — a 2nd-order multistep
        # sampler that extracts more quality per step by blending the current
        # and previous denoised predictions. Matches ComfyUI's 'dpmpp_2m' in
        # spirit; closer in behavior to the ClownSampler 'res_2s' that the
        # reference single-stage workflow uses than plain euler would be.
        # Standard non-Outpaint 2-stage generation uses Euler for its first
        # pass. Maestro intentionally keeps Outpaint deterministic too: the
        # published graph's ancestral first pass can diffuse the #66FF00 mask
        # sentinel into the generated canvas with this MMGP/FP8 runtime. That
        # produces a broad green cast which final edge cleanup cannot remove.
        if full_resolution_refine:
            # Compatibility override for Maestro's quantized/streamed path.
            # Keep the official sigma schedule and pixel-space handoff, but
            # avoid injecting ancestral noise into marker-conditioned pixels.
            stepper = EulerDiffusionStep()
            print(
                "[LTX2] Outpaint sampling: marker-safe deterministic Euler "
                "first pass and full-resolution refinement; "
                "official 5/2 Laplacian blend schedule."
            )
        elif single_stage:
            stepper = DPMSolverPlusPlus2MDiffusionStep()
        elif use_ancestral_sampler:
            stepper = LTX25EulerAncestralDiffusionStep(
                generator=ancestral_generator,
            )
        else:
            stepper = EulerDiffusionStep()
        # Keep the native LTX-2.5 ancestral trajectory continuous across its
        # 8-step base and 3-step full-resolution refinement. LTX-2/2.3 retain
        # Maestro's established low-eta refinement behavior below.
        if full_resolution_refine:
            # Lightricks' published graph uses deterministic euler_cfg_pp for
            # its two-step full-resolution refinement pass.
            stepper_stage2 = EulerDiffusionStep()
        elif attention_masked_outpaint:
            # Keep the older source-attention compatibility path
            # deterministic; unlike the pixel-handoff workflow it was tuned
            # around a three-step refinement schedule.
            stepper_stage2 = EulerDiffusionStep()
        elif use_ancestral_sampler:
            stepper_stage2 = stepper
        else:
            stepper_stage2 = EulerAncestralDiffusionStep(
                generator=ancestral_generator,
                eta=0.25,
            )
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
        dtype = torch.bfloat16

        text_encoder = self._get_model("text_encoder")
        if enhance_prompt:
            prompt = generate_enhanced_prompt(text_encoder, prompt, images[0][0] if len(images) > 0 else None)
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(
            text_encoder, text_connectors
        )
        encode_fn = lambda prompts: postprocess_text_embeddings(
            encode_text(text_encoder, prompts=prompts),
            feature_extractor,
            video_connector,
            audio_connector,
        )
        enable_audio_text_nag = False
        video_NAG = None
        audio_NAG = None
        if float(NAG_scale) > 1.0 and negative_prompt:
            contexts = self.text_encoder_cache.encode(
                encode_fn,
                [prompt, negative_prompt],
                device=self.device,
                parallel=True,
            )
        else:
            contexts = self.text_encoder_cache.encode(encode_fn, [prompt], device=self.device, parallel=True)

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()
        if float(NAG_scale) > 1.0 and negative_prompt:
            (video_context, audio_context), (video_context_n, audio_context_nag) = contexts
            video_pos_len = video_context.shape[0] if video_context.dim() == 2 else video_context.shape[1]
            video_context_n = _align_seq_len(video_context_n, video_pos_len)
            video_cat_dim = 0 if video_context.dim() == 2 else 1
            video_context = torch.cat([video_context, video_context_n], dim=video_cat_dim)
            video_NAG = {
                "scale": float(NAG_scale),
                "tau": float(NAG_tau),
                "alpha": float(NAG_alpha),
                "cap_embed_len": int(video_pos_len),
                "enable_audio_text_nag": enable_audio_text_nag,
            }
            if enable_audio_text_nag:
                audio_pos_len = audio_context.shape[0] if audio_context.dim() == 2 else audio_context.shape[1]
                audio_context_nag = _align_seq_len(audio_context_nag, audio_pos_len)
                audio_cat_dim = 0 if audio_context.dim() == 2 else 1
                audio_context = torch.cat([audio_context, audio_context_nag], dim=audio_cat_dim)
                audio_NAG = {
                    "scale": float(NAG_scale),
                    "tau": float(NAG_tau),
                    "alpha": float(NAG_alpha),
                    "cap_embed_len": int(audio_pos_len),
                    "enable_audio_text_nag": enable_audio_text_nag,
                }
        else:
            video_context, audio_context = contexts[0]

        # Stage 1: Initial low resolution video generation.
        bench_transformer = _env_flag(_BENCH_TRANSFORMER_ENV, "0")
        video_encoder = self._get_model("video_encoder")
        transformer = _TransformerBenchWrapper(self._get_model("transformer"), enabled=bench_transformer)
        bind_interrupt_check(transformer, interrupt_check)
        # DISTILLED_SIGMA_VALUES = [0.421875, 0]
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        pass_no = 1
        if loras_slists is not None:
            stage_1_steps = len(stage_1_sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_1_steps,
                phase_switch_step=stage_1_steps,
                phase_switch_step2=stage_1_steps,
            )

        if callback is not None:
            # Pre-declare the FULL multi-pass total (stage 1 + stage 2) so the
            # progress bar tracks 1..N across the whole job instead of filling
            # stage 1 to 100%, then jumping to (stage1_steps)/(stage1+stage2)
            # the moment pass 2 kicks in. Stage 2's sigma length is determined
            # here (mirroring the logic at stage 2 setup below) so the total
            # is known at the very first callback.
            if single_stage and not full_resolution_refine:
                # No stage 2 — the progress bar should only account for stage 1.
                _stage_2_len = 0
            elif full_resolution_refine:
                _stage_2_len = (
                    len(OUTPAINT_FULL_RES_REFINE_SIGMA_VALUES) - 1
                )
            elif stage2_steps > 0:
                from .utils.constants import build_stage2_sigmas
                _stage_2_len = len(build_stage2_sigmas(stage2_steps)) - 1
            else:
                _stage_2_len = len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1
            _pipeline_total_steps = (len(stage_1_sigmas) - 1) + _stage_2_len
            callback(-1, None, True, override_num_inference_steps=len(stage_1_sigmas) - 1,
                     pass_no=pass_no, total_steps_hint=_pipeline_total_steps)

        def denoising_loop_stage1(
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
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                    video_nag=video_NAG,
                    audio_nag=audio_NAG,
                    alt_guidance_scale=alt_guidance_scale,
                    # ID-LoRA: read identity guidance scale from pipeline
                    # components (set by ltx2.py when voice_reference is
                    # provided). Default 0.0 = inactive. Distilled is
                    # single-phase so we activate on stage 1.
                    audio_identity_guidance_scale=float(getattr(self.pipeline_components, 'identity_guidance_scale', 0.0) or 0.0),
                ),
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
            width=stage_1_width,
            height=stage_1_height,
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
            if (
                full_resolution_refine
                or video_conditioning_generation_mask is not None
            ):
                reference_attention_generation_mask = (
                    _select_reference_attention_generation_mask(
                        full_resolution_refine,
                        video_conditioning_generation_mask,
                    )
                )
                if full_resolution_refine:
                    print(
                        "[LTX2] Applying full-reference IC-LoRA attention "
                        "for Lightricks' official Outpaint mask guide."
                    )
                else:
                    print(
                        "[LTX2] Applying source-region reference "
                        "attention (complete canvas retained)."
                    )
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
                    generation_mask=reference_attention_generation_mask,
                )
            elif int(video_conditioning_downscale_factor or 1) > 1:
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
            num_steps=len(stage_1_sigmas) - 1,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            audio_conditionings=audio_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop_stage1,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            mask_context=mask_context,
        )
        stage1_transformer_ms = 0.0
        stage1_transformer_calls = 0
        if bench_transformer:
            stage1_transformer_ms, stage1_transformer_calls = transformer.consume()
            print(
                "[WAN2GP][LTX2][bench] transformer stage1: "
                f"{stage1_transformer_ms / 1000.0:.3f}s ({stage1_transformer_calls} calls)"
            )
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        # Single-stage: decode the stage-1 latent directly and return — no
        # upscale, no stage-2 refine. stage_1_output_shape already has the
        # target resolution in this branch.
        if single_stage and not full_resolution_refine:
            torch.cuda.synchronize()
            del transformer
            del video_encoder
            cleanup_memory()

            latent_slice = None
            if return_latent_slice is not None:
                latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
            video_latent = [video_state.latent]
            video_state = None
            decoded_video = vae_decode_video_to_tensor(
                video_latent,
                self._get_model("video_decoder"),
                tiling_config,
                expected_frames=int(stage_1_output_shape.frames),
                expected_height=int(stage_1_output_shape.height),
                expected_width=int(stage_1_output_shape.width),
                interrupt_check=interrupt_check,
                generator=generator,
            )
            decoded_audio = vae_decode_audio(
                audio_state.latent, self._get_model("audio_decoder"), self._get_model("vocoder")
            )
            if latent_slice is not None:
                return decoded_video, decoded_audio, latent_slice
            return decoded_video, decoded_audio

        # Stage 1 conditionings are no longer needed and can retain a sizable
        # encoded reference video during the pixel-space handoff.
        del stage_1_conditionings
        mask_context = None

        # Standard stage 2 spatially upscales the latent. Official Outpaint
        # instead decodes pass one, Laplacian-blends the protected source,
        # resizes those pixels with Lanczos, and re-encodes for refinement.
        if full_resolution_refine:
            print(
                "[LTX2] Decoding and source-blending the half-resolution "
                "Outpaint pass before pixel resize."
            )
            refinement_source = _coerce_refinement_source_cthw(
                video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                num_frames=stage_1_output_shape.frames,
                generation_mask=video_conditioning_generation_mask,
            )
            refinement_mask = _coerce_refinement_mask_cthw(
                video_conditioning_generation_mask,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                num_frames=stage_1_output_shape.frames,
            )
            first_pass_shape = tuple(video_state.latent[:1].shape)
            upscaled_video_latent = _decode_blend_reencode_outpaint(
                latent=video_state.latent[:1],
                source=refinement_source,
                generation_mask=refinement_mask,
                video_decoder=self._get_model("video_decoder"),
                video_encoder=video_encoder,
                tiling_config=tiling_config,
                num_frames=stage_1_output_shape.frames,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                target_height=height,
                target_width=width,
                device=self.device,
                dtype=dtype,
                interrupt_check=interrupt_check,
            )
            if upscaled_video_latent is None:
                return None, None
            expected_refinement_shape = (
                first_pass_shape[0],
                first_pass_shape[1],
                first_pass_shape[2],
                int(height) // 32,
                int(width) // 32,
            )
            if (
                tuple(upscaled_video_latent.shape)
                != expected_refinement_shape
            ):
                raise RuntimeError(
                    "Outpaint pixel handoff produced latent shape "
                    f"{tuple(upscaled_video_latent.shape)}; expected "
                    f"{expected_refinement_shape} after resizing pass one "
                    f"from {first_pass_shape}."
                )
            del refinement_source
            del refinement_mask
            del video_state
            refinement_steps = (
                len(OUTPAINT_FULL_RES_REFINE_SIGMA_VALUES) - 1
            )
            print(
                "[LTX2] Decoded, source-blended with an area/Lanczos "
                "handoff, and re-encoded pass one at target resolution; "
                f"running {refinement_steps}-step pixel refinement."
            )
        else:
            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=self._get_model("spatial_upsampler"),
            )

        torch.cuda.synchronize()
        cleanup_memory()

        if full_resolution_refine:
            stage_2_sigmas = torch.Tensor(
                OUTPAINT_FULL_RES_REFINE_SIGMA_VALUES
            ).to(self.device)
            print(
                "[LTX2] Outpaint refinement sigmas: "
                f"{OUTPAINT_FULL_RES_REFINE_SIGMA_VALUES} "
                f"({len(OUTPAINT_FULL_RES_REFINE_SIGMA_VALUES) - 1} steps)."
            )
        elif attention_masked_outpaint:
            stage_2_sigmas = torch.Tensor(
                OUTPAINT_ATTENTION_STAGE_2_SIGMA_VALUES
            ).to(self.device)
            print(
                "[LTX2] Official Outpaint stage-2 sigmas: "
                f"{OUTPAINT_ATTENTION_STAGE_2_SIGMA_VALUES}."
            )
        elif stage2_steps > 0:
            from .utils.constants import build_stage2_sigmas
            stage_2_sigmas = torch.Tensor(build_stage2_sigmas(stage2_steps)).to(self.device)
        else:
            stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        pass_no = 2
        if loras_slists is not None:
            stage_2_steps = len(stage_2_sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_2_steps,
                phase_switch_step=0,
                phase_switch_step2=stage_2_steps,
            )
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(stage_2_sigmas) - 1, pass_no=pass_no)

        def denoising_loop_stage2(
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
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                    video_nag=video_NAG,
                    audio_nag=audio_NAG,
                    alt_guidance_scale=alt_guidance_scale,
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
        stage_2_conditionings = _kf_se_fn(
            images=images,
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
            num_steps=len(stage_2_sigmas) - 1,
        )
        # ID-LoRA: stage 2 must freeze the voice-cloned audio from stage 1.
        # Without this, stage 2 re-noises + re-denoises the audio from
        # scratch, destroying the voice cloning. WanGP v11.77 calls this
        # the `freeze_audio_stage2` path — see distilled.py:718 upstream.
        # Detection: any positive identity_guidance_scale on components
        # signals an ID-LoRA run.
        _id_active = float(getattr(self.pipeline_components, 'identity_guidance_scale', 0.0) or 0.0) > 0.0
        _freeze_stage2_audio = bool(
            _id_active or full_resolution_refine
        )
        # Stage 2 audio conditionings: when ID-LoRA active, drop the ref
        # token conditioning (AudioConditionByReferenceLatent) from stage
        # 1. Stage 2 is frozen-audio anyway, so re-prepending ref tokens
        # adds extra work for no gain — and matches WanGP's
        # `audio_conditionings_stage2 = []` convention (ltx2.py line 1254).
        # We use the AudioConditionByReferenceLatent type-check rather than
        # an indirect signal so this stays robust if other conditionings
        # ever get added to the list.
        if full_resolution_refine:
            # The initial audio latent already contains stage 1's source-aware
            # result. The official Outpaint refinement freezes it rather than
            # reapplying audio guides during the visual cleanup pass.
            stage_2_audio_conditionings = []
        elif _id_active:
            from ..ltx_core.conditioning import AudioConditionByReferenceLatent
            stage_2_audio_conditionings = [
                c for c in (audio_conditionings or [])
                if not isinstance(c, AudioConditionByReferenceLatent)
            ]
        else:
            stage_2_audio_conditionings = audio_conditionings
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            audio_conditionings=stage_2_audio_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper_stage2,
            denoising_loop_fn=denoising_loop_stage2,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            # When ID-LoRA active: skip noising audio, freeze it across
            # the loop. Initial audio latent (stage 1's voice-cloned
            # output) flows through unchanged.
            audio_noise_scale=0.0 if _freeze_stage2_audio else None,
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            mask_context=mask_context,
            freeze_audio=_freeze_stage2_audio,
        )
        if bench_transformer:
            stage2_transformer_ms, stage2_transformer_calls = transformer.consume()
            total_transformer_ms = stage1_transformer_ms + stage2_transformer_ms
            total_transformer_calls = stage1_transformer_calls + stage2_transformer_calls
            print(
                "[WAN2GP][LTX2][bench] transformer stage2: "
                f"{stage2_transformer_ms / 1000.0:.3f}s ({stage2_transformer_calls} calls)"
            )
            print(
                "[WAN2GP][LTX2][bench] transformer total: "
                f"{total_transformer_ms / 1000.0:.3f}s ({total_transformer_calls} calls)"
            )
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        latent_slice = None
        if return_latent_slice is not None:
            latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
        video_latent = [video_state.latent]
        video_state = None
        decoded_video = vae_decode_video_to_tensor(
            video_latent,
            self._get_model("video_decoder"),
            tiling_config,
            expected_frames=int(stage_2_output_shape.frames),
            expected_height=int(stage_2_output_shape.height),
            expected_width=int(stage_2_output_shape.width),
            interrupt_check=interrupt_check,
            generator=generator,
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self._get_model("audio_decoder"), self._get_model("vocoder")
        )
        if latent_slice is not None:
            return decoded_video, decoded_audio, latent_slice
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_distilled_arg_parser()
    args = parser.parse_args()
    pipeline = DistilledPipeline(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
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
