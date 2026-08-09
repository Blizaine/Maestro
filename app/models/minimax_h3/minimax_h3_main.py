"""Native MiniMax H3 Base (T2VA / FL2VA / Ref2VA) runtime for Maestro.

The sampling contract follows the official Diffusers implementation pinned in
``UPSTREAM.md``.  Model construction is checkpoint-shaped so MMGP can stream
Comfy-Org's compact consumer weights on machines that cannot hold the full
42.5 GB stack in VRAM at once.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm import tqdm

from mmgp import offload, quant_router
from shared.qtypes.int8_convrot import install_native_lora_forwards
from shared.utils import files_locator as fl

from .audio_vae import AutoencoderKLMiniMaxH3Audio
from .checkpoint import (
    preprocess_audio_vae_state_dict,
    preprocess_conditioner_state_dict,
    preprocess_video_vae_state_dict,
)
from .conditioner import MiniMaxH3Conditioner, MiniMaxH3Qwen3VL, build_h3_processor, load_h3_qwen_config
from .convrot_layout import restore_interleaved_h3_qkv
from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    prepare_keyframe_image,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .ref2va import (
    build_ref2va_packed_sequence,
    ensure_ref2va_prompt_relationships,
    prepare_references,
    trim_reference_num_frames,
)
from .scheduler import MiniMaxH3Scheduler
from .first_block_cache import MiniMaxH3FirstBlockCache
from .transformer import (
    MiniMaxH3Transformer,
    _activation_chunk_tokens,
    get_linear_split_map,
)
from .turbo import (
    MINIMAX_H3_TURBO_MIN_STEPS,
    find_minimax_h3_turbo_loras,
    h3_scheduler_grid_points,
)
from .video_vae import AutoencoderKLMiniMaxH3


VIDEO_LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)
VIDEO_LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)
AUDIO_LATENTS_MEAN = (
    -0.020211687488382354,
    0.3876466479950502,
    -0.04398279799186767,
    -0.28591514936373,
    0.08179686214561671,
    -0.35782641352446604,
    0.040623809960919084,
    -0.01552534501956604,
    -0.223362481667332,
    0.1821006842509091,
    0.2941778783780663,
    -0.07901167601970885,
    -0.056815072777201,
    -0.3699028221860095,
    -0.31616315591624855,
    0.5905951377425391,
    -0.052139568068853864,
    0.013673160263486295,
    -0.03691647864630577,
    0.09732660653298163,
    -0.3394662328788498,
    -0.30685677538541667,
    -0.24504598907458763,
    -0.034698524462007344,
    0.02868032184767538,
    -0.21217779266454084,
    -0.1678263169941987,
    0.3221287889040614,
    -0.1223055851554907,
    0.4356604928128464,
    -0.0502599202236253,
    0.3979258376211797,
)
AUDIO_LATENTS_STD = (
    1.6895524230479284,
    2.76263727217653,
    1.7945344281264435,
    1.6801681847309828,
    1.6390226546605453,
    2.7788298348882177,
    1.7659090095747236,
    1.6199757612137327,
    2.6336525640336896,
    1.8539356672817833,
    2.5056497896915633,
    1.811019237886178,
    1.9579657790720237,
    1.6685498243529284,
    1.4922469314453364,
    3.298670198067373,
    1.9491804496832168,
    1.8720003270431442,
    1.8334080103291832,
    1.6488070416529093,
    1.6176957696319716,
    1.9131449234774398,
    1.5695245398428617,
    1.6943659940415912,
    1.8318420762504692,
    1.5540637421583379,
    1.9344930328968526,
    1.599198216109855,
    1.718045989838149,
    1.6307219190837705,
    1.8661226051202384,
    1.5613768203168363,
)

MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000


def normalize_h3_overlap_frames(frame_count: int) -> int:
    """Round an overlap to H3's legal ``17 * n + 1`` lattice."""

    frame_count = int(frame_count or 0)
    if frame_count < 0:
        raise ValueError("MiniMax H3 overlap must be zero or a positive frame count.")
    if frame_count == 0:
        return 0
    return max(
        1,
        ((frame_count - 1 + MINIMAX_H3_FRAMES_PER_CHUNK // 2)
         // MINIMAX_H3_FRAMES_PER_CHUNK)
        * MINIMAX_H3_FRAMES_PER_CHUNK
        + 1,
    )


def floor_h3_overlap_frames(frame_count: int) -> int:
    """Floor a short continuation to the nearest usable H3 overlap."""

    frame_count = int(frame_count or 0)
    if frame_count <= 0:
        return 0
    return max(
        1,
        ((frame_count - 1) // MINIMAX_H3_FRAMES_PER_CHUNK)
        * MINIMAX_H3_FRAMES_PER_CHUNK
        + 1,
    )


def _keyframe_latent_stats_cpu() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the official FL2VA keyframe normalization tensors on CPU.

    H3 rounds encoded keyframes to float16, promotes them back to float32,
    and normalizes them on CPU before returning the packed rows to the GPU.
    Maestro sets a CUDA default device globally, so an omitted ``device``
    here would silently put these constants on CUDA and break that contract.
    """
    means = torch.tensor(
        VIDEO_LATENTS_MEAN,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).view(1, -1, 1, 1, 1)
    stds = torch.tensor(
        VIDEO_LATENTS_STD,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).view(1, -1, 1, 1, 1)
    return means, stds


def _first_path(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _tensor_to_pil(image) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not isinstance(image, torch.Tensor):
        return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")

    tensor = image.detach().to("cpu")
    if tensor.ndim == 4:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"MiniMax H3 keyframes must be CHW tensors, got {tuple(tensor.shape)}.")
    if tensor.dtype == torch.uint8:
        pixels = tensor.permute(1, 2, 0).numpy()
    else:
        pixels = tensor.float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
        pixels = pixels.permute(1, 2, 0).numpy()
    return Image.fromarray(pixels).convert("RGB")


def _as_video_tensor(input_video) -> torch.Tensor | None:
    """Normalize a continuation tensor to channel/time/height/width form."""

    if input_video is None or not isinstance(input_video, torch.Tensor):
        return None
    continuation = input_video
    if continuation.ndim == 3:
        continuation = continuation.unsqueeze(1)
    if continuation.ndim != 4 or continuation.shape[1] < 1:
        return None
    return continuation


def _split_continuation_video(
    input_video,
    prefix_frames_count: int,
    *,
    has_explicit_start: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Split a legal overlap into exact history and a regenerated boundary.

    H3's overlap lattice is ``17*n+1``. The first ``17*n`` frames become
    clean multi-frame history conditions; the final frame remains the normal
    FL2VA first-frame anchor. The outer window assembler later removes the
    complete overlap, so only newly generated frames enter the joined movie.
    """

    if has_explicit_start:
        return None, None, 0
    continuation = _as_video_tensor(input_video)
    if continuation is None:
        return None, None, 0
    try:
        requested_raw = int(prefix_frames_count or 0)
    except (TypeError, ValueError):
        requested_raw = 0
    if requested_raw <= 0:
        return None, None, 0
    requested = normalize_h3_overlap_frames(requested_raw)
    continuation_count = min(requested, int(continuation.shape[1]))
    if continuation_count < requested:
        continuation_count = floor_h3_overlap_frames(continuation_count)
    if continuation_count <= 0:
        return None, None, 0
    boundary = continuation[:, -1:]
    history = (
        continuation[:, -continuation_count:-1]
        if continuation_count > 1
        else None
    )
    return history, boundary, continuation_count


def _last_continuation_frame(input_video, prefix_frames_count: int):
    """Compatibility helper returning the regenerated overlap boundary."""

    _, boundary, _ = _split_continuation_video(
        input_video,
        prefix_frames_count,
    )
    return boundary


def _prepare_stereo_waveform(
    waveform,
    sample_rate: int | None,
    sample_count: int,
) -> torch.Tensor | None:
    """Convert sample-major or channel-major audio to padded 32 kHz stereo."""

    if waveform is None or sample_count <= 0:
        return None
    audio = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim == 2:
        if audio.shape[0] not in (1, MINIMAX_H3_AUDIO_CHANNELS):
            if audio.shape[1] in (1, MINIMAX_H3_AUDIO_CHANNELS):
                audio = audio.transpose(0, 1)
            else:
                raise ValueError(
                    "MiniMax H3 continuation audio must be mono or stereo; "
                    f"got {tuple(audio.shape)}."
                )
    else:
        raise ValueError(
            "MiniMax H3 continuation audio must be one- or two-dimensional; "
            f"got {tuple(audio.shape)}."
        )
    if audio.shape[0] == 1:
        audio = audio.expand(MINIMAX_H3_AUDIO_CHANNELS, -1).contiguous()
    elif audio.shape[0] != MINIMAX_H3_AUDIO_CHANNELS:
        audio = audio[:MINIMAX_H3_AUDIO_CHANNELS]

    sample_rate = int(sample_rate or MINIMAX_H3_AUDIO_SAMPLE_RATE)
    if sample_rate <= 0:
        raise ValueError("MiniMax H3 continuation audio needs a positive sample rate.")
    if sample_rate != MINIMAX_H3_AUDIO_SAMPLE_RATE:
        import torchaudio.functional as audio_functional

        audio = audio_functional.resample(
            audio,
            sample_rate,
            MINIMAX_H3_AUDIO_SAMPLE_RATE,
        )
    audio = audio[..., :sample_count]
    if audio.shape[-1] < sample_count:
        audio = F.pad(audio, (0, sample_count - audio.shape[-1]))
    return audio.contiguous()


def _strip_transformer_wrappers(
    state_dict,
    quantization_map=None,
    tied_weights_map=None,
):
    restore_interleaved_h3_qkv(state_dict)
    prefixes = ("model.diffusion_model.", "diffusion_model.")

    def strip(mapping):
        if mapping is None:
            return None
        normalized = {}
        for key, value in mapping.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            normalized[key] = value
        return normalized

    return strip(state_dict), strip(quantization_map), strip(tied_weights_map)


def _normalize_conditioner_checkpoint_namespaces(
    state_dict,
    quantization_map=None,
    tied_weights_map=None,
):
    """Map every supported Qwen checkpoint layout onto Maestro's wrapper.

    Comfy's NVFP4 export already uses ``model.*`` while the BF16, Quanto
    INT8, and GGUF files published for WanGP use ``language_model.*``.
    MMGP applies the same names to its quantization and tied-weight metadata,
    so all three mappings must be renamed together or the large checkpoint
    appears to load while every language-model parameter is reported missing.
    """

    prefixes = (
        ("text_encoder.language_model.", "model."),
        ("text_encoder.model.", "model."),
        ("text_encoder.visual.", "visual."),
        ("language_model.", "model."),
    )

    def normalize_name(name):
        if not isinstance(name, str):
            return name
        for source_prefix, target_prefix in prefixes:
            if name.startswith(source_prefix):
                return target_prefix + name[len(source_prefix) :]
        return name

    def normalize_tied_value(value):
        if isinstance(value, str):
            return normalize_name(value)
        if isinstance(value, list):
            return [normalize_tied_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(normalize_tied_value(item) for item in value)
        return value

    def normalize(mapping, *, tied=False):
        if mapping is None:
            return None
        normalized = {}
        for key, value in mapping.items():
            normalized[normalize_name(key)] = normalize_tied_value(value) if tied else value
        return normalized

    return (
        normalize(state_dict),
        normalize(quantization_map),
        normalize(tied_weights_map, tied=True),
    )


def probe_h3_checkpoint(filename: str) -> dict[str, int | bool | None]:
    """Inspect H3 tensor headers before allocating its 20B/33B network."""

    state_dict, _ = quant_router.load_metadata_state_dict(filename)
    table = None
    for key, tensor in state_dict.items():
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key == "adaln_t_table":
            table = tensor
            break
    if table is None:
        return {
            "compressed_modulation": False,
            "adaln_curve_grid": None,
            "time_embed_dim": 2688,
        }
    if len(table.shape) != 2 or int(table.shape[0]) < 2:
        raise ValueError(f"Invalid H3 AdaLN curve table shape: {tuple(table.shape)}")
    return {
        "compressed_modulation": True,
        "adaln_curve_grid": int(table.shape[0]),
        "time_embed_dim": int(table.shape[1]),
    }


def _load_transformer(
    filename: str,
    dtype: torch.dtype,
    *,
    qkv_layout: str = "contiguous",
) -> MiniMaxH3Transformer:
    checkpoint = probe_h3_checkpoint(filename)
    with init_empty_weights(include_buffers=True):
        transformer = MiniMaxH3Transformer(
            curve_grid=checkpoint["adaln_curve_grid"],
            curve_dim=int(checkpoint["time_embed_dim"]),
            dtype=dtype,
        )
    inner_size = 56 * 128
    # Comfy's scaled-FP8 pruned checkpoints already store grouped [Q, K, V]
    # weights in the exact fused layout used by this runtime.  MMGP 3.7.6's
    # scaled-FP8 fused splitter rebuilds the three tensors as shared-storage
    # views and later mistakes them for tied parameters, corrupting attention.
    # Keep that proven consumer path fused.  Full WanGP ConvRot checkpoints
    # use the official head-interleaved layout and still require an explicit
    # split/reorder before inference.
    split_map = (
        get_linear_split_map(inner_size, interleaved=True)
        if qkv_layout == "interleaved"
        else None
    )
    if split_map is not None:
        offload.split_linear_modules(transformer, split_map)
    offload.load_model_data(
        transformer,
        filename,
        writable_tensors=False,
        default_dtype=dtype,
        preprocess_sd=_strip_transformer_wrappers,
        fused_split_map=split_map,
    )
    transformer._model_dtype = dtype
    transformer.h3_checkpoint_info = checkpoint
    transformer.split_linear_modules_map = split_map
    transformer.h3_qkv_layout = qkv_layout
    print(
        "[MiniMax H3] Loaded "
        f"{'pruned 20B curve' if checkpoint['compressed_modulation'] else 'full 33B'} "
        f"transformer ({qkv_layout} QKV, "
        f"{'independent split projections' if split_map is not None else 'fused projection'})."
    )
    return transformer.eval().requires_grad_(False)


def _load_conditioner(
    filename: str,
    assets_root: str,
    dtype: torch.dtype,
    *,
    variant: str = "nvfp4_awq",
) -> MiniMaxH3Conditioner:
    config_path = fl.locate_file(os.path.join(assets_root, "text_encoder", "config.json"))
    processor_path = fl.locate_folder(os.path.join(assets_root, "processor"))
    config = load_h3_qwen_config(config_path)
    tokenizer, processor = build_h3_processor(processor_path)
    # Qwen keeps rotary-frequency tables as computed, non-persistent buffers,
    # so they are intentionally absent from the checkpoint.  Keep those small
    # buffers materialized while Accelerate places the 32B parameters on meta.
    with init_empty_weights(include_buffers=False):
        qwen = MiniMaxH3Qwen3VL(
            config,
            dtype=dtype,
            consumer_quantized=variant == "nvfp4_awq",
        )

    def preprocess_checkpoint(state_dict, quantization_map=None, tied_weights_map=None):
        state_dict, quantization_map, tied_weights_map = (
            _normalize_conditioner_checkpoint_namespaces(
                state_dict,
                quantization_map,
                tied_weights_map,
            )
        )
        if variant == "nvfp4_awq":
            state_dict = preprocess_conditioner_state_dict(state_dict)
        return state_dict, quantization_map, tied_weights_map

    offload.load_model_data(
        qwen,
        filename,
        writable_tensors=False,
        preprocess_sd=preprocess_checkpoint,
        default_dtype=dtype,
        ignore_unused_weights=True,
    )
    qwen._model_dtype = dtype
    # These towers are profiled independently to keep the vision encoder off
    # the GPU during text-only work. MMGP reads dtype metadata from each
    # top-level profiled module, not from its former Qwen parent; preserve the
    # override on both children so the NVFP4 checkpoint's intentional INT8
    # embedding/FP32 scale mixture cannot trip its uniform-dtype assertion.
    qwen.model._model_dtype = dtype
    qwen.visual._model_dtype = dtype
    qwen.eval().requires_grad_(False)
    conditioner = MiniMaxH3Conditioner(qwen, tokenizer, processor).eval().requires_grad_(False)
    conditioner._model_dtype = dtype
    return conditioner


def _load_video_vae(filename: str) -> AutoencoderKLMiniMaxH3:
    # Rotary tables are computed, non-persistent buffers and therefore are
    # not present in the compact checkpoint.
    with init_empty_weights(include_buffers=False):
        vae = AutoencoderKLMiniMaxH3(
            latents_mean=VIDEO_LATENTS_MEAN,
            latents_std=VIDEO_LATENTS_STD,
        )
    offload.load_model_data(
        vae,
        filename,
        writable_tensors=False,
        preprocess_sd=preprocess_video_vae_state_dict,
        default_dtype=torch.float16,
    )
    vae._model_dtype = torch.float16
    return vae.eval().requires_grad_(False)


def _load_audio_vae(filename: str) -> AutoencoderKLMiniMaxH3Audio:
    # Preserve any computed codec buffers while keeping all learned
    # parameters empty until MMGP streams the checkpoint.
    with init_empty_weights(include_buffers=False):
        vae = AutoencoderKLMiniMaxH3Audio(
            latents_mean=AUDIO_LATENTS_MEAN,
            latents_std=AUDIO_LATENTS_STD,
        )
    offload.load_model_data(
        vae,
        filename,
        writable_tensors=False,
        preprocess_sd=preprocess_audio_vae_state_dict,
        default_dtype=torch.float32,
    )
    vae._model_dtype = torch.float32
    return vae.eval().requires_grad_(False)


class MiniMaxH3Model:
    """Maestro generation wrapper for the H3 Base FL2VA/Ref2VA checkpoints."""

    def __init__(
        self,
        model_filename,
        model_def,
        text_encoder_filename,
        dtype: torch.dtype = torch.bfloat16,
        minimax_h3_text_encoder: str = "nvfp4_awq",
        **_kwargs,
    ):
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.model_def = model_def
        self.assets_root = model_def.get("minimax_h3_assets_root", "minimax_h3")
        self.omni_reference = bool(model_def.get("omni_reference", False))

        transformer_path = _first_path(model_filename)
        if not transformer_path:
            raise FileNotFoundError("MiniMax H3 transformer checkpoint is missing.")
        if not text_encoder_filename:
            raise FileNotFoundError("MiniMax H3 Qwen3-VL conditioner checkpoint is missing.")

        video_vae_path = fl.locate_file(
            os.path.join(self.assets_root, "vae", "minimax_h3_video_vae_fp16.safetensors")
        )
        audio_vae_path = fl.locate_file(
            os.path.join(self.assets_root, "vae", "minimax_h3_audio_vae_fp32.safetensors")
        )

        self.text_encoder_variant = str(minimax_h3_text_encoder or "nvfp4_awq")
        self.transformer = _load_transformer(
            transformer_path,
            dtype,
            qkv_layout=str(model_def.get("minimax_h3_qkv_layout") or "contiguous"),
        )
        self.conditioner = _load_conditioner(
            text_encoder_filename,
            self.assets_root,
            dtype,
            variant=self.text_encoder_variant,
        )
        self.vae = _load_video_vae(video_vae_path)
        self.audio_vae = _load_audio_vae(audio_vae_path)
        self.scheduler = MiniMaxH3Scheduler(shift=12.0)
        self.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
        self._turbo_lora_active = False
        self._turbo_lora_paths: tuple[str, ...] = ()
        self.__interrupt = False

    def validate_loras(self, loras_selected) -> None:
        """Validate special H3 adapter requirements before MMGP loads them."""

        turbo_paths = tuple(find_minimax_h3_turbo_loras(loras_selected))
        self._turbo_lora_paths = turbo_paths
        self._turbo_lora_active = bool(turbo_paths)

    def finalize_loras(self) -> None:
        """Preserve ConvRot math after MMGP attaches active LoRA hooks."""

        convrot_layers = [
            module
            for module in self.transformer.modules()
            if getattr(module, "_mm_requires_native_linear_forward", False)
        ]
        installed = install_native_lora_forwards(self.transformer)
        if convrot_layers and self._turbo_lora_active and installed == 0:
            raise RuntimeError(
                "MiniMax H3 Turbo could not attach its ConvRot-safe LoRA path."
            )
        if installed:
            print(
                "[MiniMax H3 LoRA] Preserved native ConvRot activation math for "
                f"{installed} adapter-targeted layer(s)."
            )

    @property
    def _interrupt(self) -> bool:
        return self.__interrupt

    @_interrupt.setter
    def _interrupt(self, value: bool) -> None:
        self.__interrupt = bool(value)
        if hasattr(self, "transformer"):
            self.transformer._interrupt = self.__interrupt
        if hasattr(self, "conditioner"):
            self.conditioner._interrupt = self.__interrupt

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return tuple(self.transformer.config.patch_size)

    def _condition_pixels(
        self,
        source,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Convert a PIL keyframe or CTHW window history to normalized pixels."""

        if isinstance(source, Image.Image):
            pixels = torch.from_numpy(np.array(source.convert("RGB"), dtype=np.uint8))
            video = pixels.permute(2, 0, 1)[:, None].to(self.device)
            video = video.float().div(255.0)
        else:
            video = _as_video_tensor(source)
            if video is None:
                raise ValueError("MiniMax H3 received an invalid visual condition.")
            video = video.to(self.device)
            if video.dtype == torch.uint8:
                video = video.float().div(255.0)
            else:
                video = video.float()
                if float(video.amin()) < -0.01:
                    video = video.add(1.0).mul(0.5)
                video = video.clamp(0.0, 1.0)

        if tuple(video.shape[-2:]) != (height, width):
            video = F.interpolate(
                video.permute(1, 0, 2, 3),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)
        pixel_mean = torch.tensor(
            MINIMAX_H3_PIXEL_MEAN,
            device=self.device,
        ).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(
            MINIMAX_H3_PIXEL_STD,
            device=self.device,
        ).view(1, -1, 1, 1, 1)
        return (video[None] - pixel_mean) / pixel_std

    def _encode_visual_conditions(
        self,
        conditions: list[dict],
        latent_height: int,
        latent_width: int,
        generator: torch.Generator,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor | None, tuple]:
        """Encode clean keyframes and multi-frame history in packed order."""

        if not conditions:
            return None, ()

        means, stds = _keyframe_latent_stats_cpu()
        rows: list[torch.Tensor] = []
        condition_shapes: list[tuple[int, int, int]] = []
        anchors: list[tuple] = []
        for condition in conditions:
            if self._interrupt:
                return None, ()
            pixels = self._condition_pixels(
                condition["source"],
                height,
                width,
            )
            posterior = self.vae.encode_condition(
                pixels,
                keep_all_latents=bool(condition.get("keep_all_latents", False)),
            )
            encoded = posterior.sample(
                generator=torch.Generator().manual_seed(
                    MINIMAX_H3_KEYFRAME_ENCODE_SEED
                )
            )
            encoded = encoded.to(torch.float16).float().cpu()
            latent_frames = int(encoded.shape[2])
            condition_shapes.append(
                (latent_frames, int(encoded.shape[3]), int(encoded.shape[4]))
            )
            rows.append(
                patchify_video_latents(
                    (encoded - means) / stds,
                    self.patch_size,
                )
            )
            anchor = str(condition["anchor"])
            if anchor == "frame":
                anchors.append(
                    (anchor, latent_frames, int(condition["frame_index"]))
                )
            else:
                anchors.append((anchor, latent_frames))

        clean_rows = torch.cat(rows).to(self.device)
        noise = keyframe_condition_noise(
            tuple(condition_shapes),
            self.patch_size,
            24,
            generator=generator,
            device=self.device,
        )
        return (
            self.scheduler.scale_noise(
                clean_rows,
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                noise,
            ),
            tuple(anchors),
        )

    def _encode_keyframes(
        self,
        images: list[Image.Image],
        latent_height: int,
        latent_width: int,
        generator: torch.Generator,
    ) -> torch.Tensor | None:
        """Backward-compatible one-frame wrapper used by focused tests."""

        rows, _ = self._encode_visual_conditions(
            [
                {"anchor": "first", "source": image}
                for image in images
            ],
            latent_height,
            latent_width,
            generator,
            height=latent_height * self.vae.spatial_compression_ratio,
            width=latent_width * self.vae.spatial_compression_ratio,
        )
        return rows

    def _encode_continuation_audio(
        self,
        waveform,
        sample_rate: int | None,
        continuation_count: int,
        history_count: int,
        fps: float,
    ) -> tuple[torch.Tensor | None, tuple, torch.Tensor | None]:
        """Encode the previous window's matching audio as history/boundary rows."""

        overlap_samples = int(
            round(continuation_count / fps * MINIMAX_H3_AUDIO_SAMPLE_RATE)
        )
        stereo = _prepare_stereo_waveform(
            waveform,
            sample_rate,
            overlap_samples,
        )
        if stereo is None:
            return None, (), None

        posterior = self.audio_vae.encode(
            stereo.to(self.device)[:, None],
            return_dict=False,
        )[0]
        latents = posterior.mode().float().cpu().transpose(1, 2)
        audio_mean = torch.tensor(
            AUDIO_LATENTS_MEAN,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).view(1, 1, -1)
        audio_std = torch.tensor(
            AUDIO_LATENTS_STD,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).view(1, 1, -1)
        normalized = (latents - audio_mean) / audio_std

        boundary_latents = (
            int(normalized.shape[1])
            if history_count <= 0
            else min(
                int(normalized.shape[1]),
                max(1, round(40 / fps)),
            )
        )
        history_latents = int(normalized.shape[1]) - boundary_latents
        blocks: list[torch.Tensor] = []
        anchors: list[tuple[str, int]] = []
        if history_latents > 0:
            blocks.append(normalized[:, :history_latents].reshape(-1, 32))
            anchors.append(("history", history_latents))
        if boundary_latents > 0:
            blocks.append(normalized[:, history_latents:].reshape(-1, 32))
            anchors.append(("first", boundary_latents))

        history_samples = int(
            round(history_count / fps * MINIMAX_H3_AUDIO_SAMPLE_RATE)
        )
        history_waveform = stereo[..., :history_samples] if history_samples else None
        return (
            torch.cat(blocks).to(self.device) if blocks else None,
            tuple(anchors),
            history_waveform,
        )

    def _encode_references(
        self,
        references: list,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Encode ordered Ref2VA visual and audio conditioning rows."""

        video_mean, video_std = _keyframe_latent_stats_cpu()
        audio_mean = torch.tensor(
            AUDIO_LATENTS_MEAN,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).view(1, 1, -1)
        audio_std = torch.tensor(
            AUDIO_LATENTS_STD,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).view(1, 1, -1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)

        video_rows: list[torch.Tensor] = []
        audio_rows: list[torch.Tensor] = []
        for reference in references:
            if self._interrupt:
                return None, None
            if reference.kind != "audio":
                if reference.kind == "image":
                    pixels = torch.from_numpy(np.array(reference.image, dtype=np.uint8))
                    pixels = pixels.to(self.device).permute(2, 0, 1)[None, :, None]
                else:
                    usable_frames = trim_reference_num_frames(reference.frames.shape[0])
                    frames = reference.frames[:usable_frames]
                    pixels = torch.from_numpy(frames.copy()).to(self.device).permute(3, 0, 1, 2)[None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                moments = (
                    self.vae._encode_clip(pixels)
                    if reference.kind == "image"
                    else self.vae._encode(pixels)
                )
                posterior = DiagonalGaussianDistribution(moments)
                latents = posterior.sample(
                    generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                )
                latents = latents.to(torch.float16).float().cpu()
                reference.num_latent_frames = int(latents.shape[2])
                reference.latent_height = int(latents.shape[3])
                reference.latent_width = int(latents.shape[4])
                video_rows.append(
                    patchify_video_latents((latents - video_mean) / video_std, self.patch_size)
                )
            if reference.has_audio:
                posterior = self.audio_vae.encode(
                    reference.waveform.to(self.device)[:, None],
                    return_dict=False,
                )[0]
                latents = posterior.mode().float().cpu().transpose(1, 2)
                reference.num_audio_latents = int(latents.shape[1])
                normalized = (latents - audio_mean) / audio_std
                audio_rows.append(normalized.reshape(-1, 32))

        visual_conditions = torch.cat(video_rows) if video_rows else None
        if visual_conditions is not None:
            noise = keyframe_condition_noise(
                tuple(
                    (reference.num_latent_frames, reference.latent_height, reference.latent_width)
                    for reference in references
                    if reference.kind != "audio"
                ),
                self.patch_size,
                24,
                generator=generator,
                device=self.device,
            )
            visual_conditions = self.scheduler.scale_noise(
                visual_conditions.to(self.device),
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                noise,
            )
        audio_conditions = torch.cat(audio_rows).to(self.device) if audio_rows else None
        return visual_conditions, audio_conditions

    @torch.inference_mode()
    def generate(
        self,
        input_prompt: str,
        image_start=None,
        image_end=None,
        input_video=None,
        input_waveform=None,
        input_waveform_sample_rate: int | None = None,
        prefix_frames_count: int = 0,
        frame_num: int = 124,
        height: int = 480,
        width: int = 864,
        fps: float = MINIMAX_H3_FPS,
        sampling_steps: int = 20,
        seed: int | None = None,
        callback=None,
        minimax_h3_references=None,
        minimax_h3_reference_detail: str = "match",
        **_kwargs,
    ):
        self._interrupt = False
        if not isinstance(input_prompt, str):
            raise ValueError("MiniMax H3 accepts one text prompt per generation.")
        if height % 32 or width % 32:
            raise ValueError(f"MiniMax H3 dimensions must be multiples of 32, got {width}x{height}.")

        fps = float(fps or MINIMAX_H3_FPS)
        if abs(fps - MINIMAX_H3_FPS) > 1e-6:
            raise ValueError(
                f"MiniMax H3 generates at {MINIMAX_H3_FPS} fps, got {fps:g}."
            )
        frame_num = align_num_frames(int(frame_num))
        duration = frame_num / fps
        if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax H3 supports {MINIMAX_H3_MIN_DURATION:g}-{MINIMAX_H3_MAX_DURATION:g}s at 24 fps; "
                f"the aligned request is {frame_num} frames ({duration:.3f}s)."
            )
        if int(sampling_steps) < 2:
            raise ValueError("MiniMax H3 needs at least two scheduler grid points.")
        if self._turbo_lora_active and int(sampling_steps) < MINIMAX_H3_TURBO_MIN_STEPS:
            raise ValueError(
                "MiniMax H3 Turbo LoRA needs at least "
                f"{MINIMAX_H3_TURBO_MIN_STEPS} denoising steps; "
                f"received {int(sampling_steps)}."
            )

        history_video = boundary_video = None
        continuation_count = history_count = 0
        history_waveform = None
        keyframes: list[Image.Image] = []
        visual_conditions: list[dict] = []
        if not self.omni_reference:
            history_video, boundary_video, continuation_count = (
                _split_continuation_video(
                    input_video,
                    prefix_frames_count,
                    has_explicit_start=image_start is not None,
                )
            )
            history_count = (
                int(history_video.shape[1])
                if history_video is not None
                else 0
            )
            if history_video is not None:
                visual_conditions.append(
                    {
                        "anchor": "history",
                        "source": history_video,
                        "keep_all_latents": True,
                    }
                )

            start_source = image_start if image_start is not None else boundary_video
            keyframe_sources = [
                ("first", start_source),
                ("last", image_end),
            ]
            for anchor, source in keyframe_sources:
                image = _tensor_to_pil(source)
                if image is None:
                    continue
                image = prepare_keyframe_image(
                    image,
                    height,
                    width,
                    stretch=len(keyframes) == 0,
                )
                keyframes.append(image)
                visual_conditions.append(
                    {"anchor": anchor, "source": image}
                )

        target_frame_num = frame_num - history_count
        if target_frame_num <= 0:
            raise ValueError(
                "MiniMax H3 sliding-window overlap leaves no frames to generate."
            )
        if align_num_frames(target_frame_num) != target_frame_num:
            raise ValueError(
                "MiniMax H3 overlap must leave a target on the 17*n+5 frame lattice; "
                f"{frame_num} total frames minus {history_count} history frames leaves "
                f"{target_frame_num}."
            )

        request_seed = int(torch.seed() if seed is None else seed)
        generator = torch.Generator(device=self.device).manual_seed(request_seed)
        num_latent_frames = video_latent_num_frames(target_frame_num)
        latent_height = height // self.vae.spatial_compression_ratio
        latent_width = width // self.vae.spatial_compression_ratio
        num_audio_latents = audio_latent_num_frames(target_frame_num)

        audio_condition_rows = None
        if self.omni_reference:
            conditioned_prompt = ensure_ref2va_prompt_relationships(
                input_prompt,
                minimax_h3_references,
                duration_seconds=frame_num / fps,
            )
            if conditioned_prompt != str(input_prompt or "").strip():
                print(
                    "[MiniMax H3 Ref2VA] Added explicit reference relationships "
                    "to an untagged prompt."
                )
            references = prepare_references(
                minimax_h3_references,
                num_frames=frame_num,
                target_height=height,
                target_width=width,
                audio_sample_rate=32000,
                detail=minimax_h3_reference_detail,
            )
            prompt_embeds, text_tags = self.conditioner.forward_ref2va(
                conditioned_prompt,
                self.device,
                references,
            )
            if prompt_embeds is None or self._interrupt:
                return None
            condition_rows, audio_condition_rows = self._encode_references(references, generator)
            if self._interrupt:
                return None
            layout = build_ref2va_packed_sequence(
                text_tags,
                references,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                self.patch_size,
            )
        else:
            prompt_embeds, text_tags = self.conditioner(input_prompt, self.device, keyframes or None)
            if prompt_embeds is None or self._interrupt:
                return None
            condition_rows, anchors = self._encode_visual_conditions(
                visual_conditions,
                latent_height,
                latent_width,
                generator,
                height=height,
                width=width,
            )
            audio_condition_rows, audio_anchors, history_waveform = (
                self._encode_continuation_audio(
                    input_waveform,
                    input_waveform_sample_rate,
                    continuation_count,
                    history_count,
                    fps,
                )
                if continuation_count > 0
                else (None, (), None)
            )
            layout = build_packed_sequence(
                text_tags,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                self.patch_size,
                anchors,
                audio_condition_anchors=audio_anchors,
            )
            if continuation_count > 1:
                print(
                    "[MiniMax H3] Continuing with "
                    f"{history_count} motion-history frames + one boundary frame"
                    + (
                        " and matching stereo audio."
                        if audio_condition_rows is not None
                        else "."
                    )
                )
        if self._interrupt:
            return None

        video_noise = randn_tensor(
            (1, 24, num_latent_frames, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        video_rows = patchify_video_latents(video_noise, self.patch_size)
        audio_rows = randn_tensor(
            (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, 32),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        if condition_rows is not None:
            video_rows = torch.cat([condition_rows, video_rows])
        if audio_condition_rows is not None:
            audio_rows = torch.cat([audio_condition_rows, audio_rows])

        scheduler_points = h3_scheduler_grid_points(
            int(sampling_steps),
            turbo_active=self._turbo_lora_active,
        )
        self.scheduler.set_timesteps(scheduler_points, device=self.device)
        self.audio_scheduler.set_timesteps(scheduler_points, device=self.device)
        timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps
        if self._turbo_lora_active:
            print(
                "[MiniMax H3 Turbo] Using "
                f"{len(timesteps)} denoising evaluations with independent "
                "video shift 12 / audio shift 3 schedules."
            )
        row_plan = [
            tuple(
                tensor.to(self.device)
                for tensor in build_row_timesteps(
                    layout,
                    float(video_timestep),
                    float(audio_timestep),
                    max(float(video_timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    1.0,
                )
            )
            for video_timestep, audio_timestep in zip(timesteps, audio_timesteps)
        ]
        token_tags = layout.token_tags.to(self.device)
        position_ids = layout.position_ids.to(self.device)
        video_indices = layout.video_indices.to(self.device)
        audio_indices = layout.audio_indices.to(self.device)
        text_indices = layout.text_indices.to(self.device)

        target_starts = []
        if audio_indices.numel() > layout.num_condition_audio_rows:
            target_starts.append(
                int(audio_indices[layout.num_condition_audio_rows].item())
            )
        if video_indices.numel() > layout.num_condition_video_rows:
            target_starts.append(
                int(video_indices[layout.num_condition_video_rows].item())
            )
        target_start_index = (
            min(target_starts) if target_starts else layout.sequence_length
        )

        cache_config = getattr(self.transformer, "cache", None)
        first_block_cache = (
            MiniMaxH3FirstBlockCache(cache_config)
            if cache_config is not None
            and getattr(cache_config, "cache_type", "") == "first_block"
            else None
        )
        if first_block_cache is not None:
            # WGP seeds num_steps with the per-window inference count before
            # generation. Replace that seed on the first H3 window, then
            # accumulate subsequent windows so the final skipped/total log is
            # accurate instead of double-counting window one.
            if not getattr(cache_config, "_h3_count_started", False):
                cache_config.num_steps = 0
                cache_config._h3_count_started = True
            cache_config.num_steps += len(timesteps)
            if not hasattr(cache_config, "skipped_steps"):
                cache_config.skipped_steps = 0

        # Emit one concise line with the actual path used. This makes slow
        # user reports actionable without restoring noisy per-request logs.
        first_block = self.transformer.blocks[0]
        qkv_mode = (
            "split"
            if hasattr(first_block.attn, "q_proj")
            else "fused"
        )
        qkv_width = (
            first_block.attn.heads * first_block.attn.head_dim
            if qkv_mode == "split"
            else first_block.attn.heads * first_block.attn.head_dim * 3
        )
        qkv_chunk = _activation_chunk_tokens(
            layout.sequence_length,
            self.transformer.config.hidden_size,
            qkv_width,
        )
        mlp_chunk = _activation_chunk_tokens(
            layout.sequence_length,
            first_block.mlp.fc1.in_features,
            first_block.mlp.fc1.out_features,
        )
        attention_backend = str(
            offload.shared_state.get("_attention", "sdpa")
        )
        cache_label = (
            f"first-block/{first_block_cache.threshold:g}"
            if first_block_cache is not None
            else "off"
        )
        print(
            "[MiniMax H3 Perf] "
            f"{width}x{height}, {frame_num} frames/{duration:.2f}s, "
            f"{layout.sequence_length:,} packed rows, {len(timesteps)} steps; "
            f"attention={attention_backend}, qkv={qkv_mode} "
            f"{(layout.sequence_length + qkv_chunk - 1) // qkv_chunk}x"
            f"{qkv_chunk:,}, mlp "
            f"{(layout.sequence_length + mlp_chunk - 1) // mlp_chunk}x"
            f"{mlp_chunk:,}, cache={cache_label}."
        )

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(timesteps))
        try:
            with tqdm(total=len(timesteps), desc="MiniMax H3 denoising") as progress:
                for index, (video_timestep, audio_timestep) in enumerate(zip(timesteps, audio_timesteps)):
                    if self._interrupt:
                        return None
                    if first_block_cache is not None:
                        first_block_cache.begin_step(index)
                    unique_timesteps, timestep_indices = row_plan[index]
                    prediction = self.transformer(
                        hidden_states=video_rows[None],
                        audio_hidden_states=audio_rows[None],
                        encoder_hidden_states=prompt_embeds,
                        timestep=unique_timesteps,
                        timestep_indices=timestep_indices,
                        token_tags=token_tags,
                        position_ids=position_ids,
                        video_indices=video_indices,
                        audio_indices=audio_indices,
                        text_indices=text_indices,
                        return_dict=False,
                        first_block_cache=first_block_cache,
                        target_start_index=target_start_index,
                    )
                    if prediction is None or self._interrupt:
                        return None
                    video_velocity, audio_velocity = prediction
                    video_rows[layout.num_condition_video_rows :] = self.scheduler.step(
                        video_velocity[0, layout.num_condition_video_rows :].float(),
                        video_timestep,
                        video_rows[layout.num_condition_video_rows :],
                        return_dict=False,
                    )[0]
                    audio_rows[layout.num_condition_audio_rows :] = self.audio_scheduler.step(
                        audio_velocity[0, layout.num_condition_audio_rows :].float(),
                        audio_timestep,
                        audio_rows[layout.num_condition_audio_rows :],
                        return_dict=False,
                    )[0]
                    if callback is not None:
                        callback(index, None)
                    progress.update()
        finally:
            if first_block_cache is not None:
                first_block_cache.reset()

        if self._interrupt:
            return None
        video_latents = unpatchify_video_tokens(
            video_rows[layout.num_condition_video_rows :],
            num_latent_frames,
            latent_height,
            latent_width,
            24,
            self.patch_size,
        )
        video_mean = torch.tensor(VIDEO_LATENTS_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        video_std = torch.tensor(VIDEO_LATENTS_STD, device=self.device).view(1, -1, 1, 1, 1)
        video_latents = video_latents * video_std + video_mean
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            video = self.vae.decode(video_latents, return_dict=False)[0]
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1).mul(2).sub(1)
        video = video[:, :, :target_frame_num]
        output_video = video[0]
        if history_video is not None:
            output_video = torch.cat(
                [history_video.to(output_video), output_video],
                dim=1,
            )

        audio_latents = unpack_audio_tokens(
            audio_rows[layout.num_condition_audio_rows :],
            num_audio_latents,
        )
        audio_mean = torch.tensor(AUDIO_LATENTS_MEAN, device=self.device).view(1, -1, 1)
        audio_std = torch.tensor(AUDIO_LATENTS_STD, device=self.device).view(1, -1, 1)
        audio_latents = audio_latents * audio_std + audio_mean
        audio = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        audio = audio.float()[:, 0]
        target_samples = int(
            round(target_frame_num / fps * MINIMAX_H3_AUDIO_SAMPLE_RATE)
        )
        audio = audio[..., :target_samples]
        if audio.shape[-1] < target_samples:
            audio = F.pad(audio, (0, target_samples - audio.shape[-1]))
        if history_count > 0:
            history_samples = int(
                round(history_count / fps * MINIMAX_H3_AUDIO_SAMPLE_RATE)
            )
            prefix_audio = (
                history_waveform.to(audio)
                if history_waveform is not None
                else torch.zeros(
                    (MINIMAX_H3_AUDIO_CHANNELS, history_samples),
                    dtype=audio.dtype,
                    device=audio.device,
                )
            )
            prefix_audio = prefix_audio[..., :history_samples]
            if prefix_audio.shape[-1] < history_samples:
                prefix_audio = F.pad(
                    prefix_audio,
                    (0, history_samples - prefix_audio.shape[-1]),
                )
            audio = torch.cat([prefix_audio, audio], dim=-1)
        total_samples = int(
            round(frame_num / fps * MINIMAX_H3_AUDIO_SAMPLE_RATE)
        )
        audio = audio[..., :total_samples].transpose(0, 1).cpu().numpy()
        return {
            "x": output_video,
            "audio": audio,
            "audio_sampling_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE,
        }
