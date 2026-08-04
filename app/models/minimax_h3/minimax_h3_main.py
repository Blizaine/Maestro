"""Native MiniMax H3 Base (T2VA / FL2VA) runtime for Maestro.

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
from accelerate import init_empty_weights
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm import tqdm

from mmgp import offload
from shared.utils import files_locator as fl

from .audio_vae import AutoencoderKLMiniMaxH3Audio
from .checkpoint import (
    preprocess_audio_vae_state_dict,
    preprocess_conditioner_state_dict,
    preprocess_video_vae_state_dict,
)
from .conditioner import MiniMaxH3Conditioner, MiniMaxH3Qwen3VL, build_h3_processor, load_h3_qwen_config
from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MiniMaxH3PreparedReference,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    prepare_keyframe_image,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .scheduler import MiniMaxH3Scheduler
from .transformer import MiniMaxH3Transformer
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


def _load_transformer(filename: str, dtype: torch.dtype) -> MiniMaxH3Transformer:
    with init_empty_weights(include_buffers=True):
        transformer = MiniMaxH3Transformer(dtype=dtype)
    offload.load_model_data(
        transformer,
        filename,
        writable_tensors=False,
        default_dtype=dtype,
    )
    transformer._model_dtype = dtype
    return transformer.eval().requires_grad_(False)


def _load_conditioner(filename: str, assets_root: str, dtype: torch.dtype) -> MiniMaxH3Conditioner:
    config_path = fl.locate_file(os.path.join(assets_root, "text_encoder", "config.json"))
    processor_path = fl.locate_folder(os.path.join(assets_root, "processor"))
    config = load_h3_qwen_config(config_path)
    tokenizer, processor = build_h3_processor(processor_path)
    # Qwen keeps rotary-frequency tables as computed, non-persistent buffers,
    # so they are intentionally absent from the checkpoint.  Keep those small
    # buffers materialized while Accelerate places the 32B parameters on meta.
    with init_empty_weights(include_buffers=False):
        qwen = MiniMaxH3Qwen3VL(config, dtype=dtype)
    offload.load_model_data(
        qwen,
        filename,
        writable_tensors=False,
        preprocess_sd=preprocess_conditioner_state_dict,
        default_dtype=dtype,
    )
    qwen._model_dtype = dtype
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
    """Maestro generation wrapper for the H3 Base FL2VA checkpoint."""

    def __init__(
        self,
        model_filename,
        model_def,
        text_encoder_filename,
        dtype: torch.dtype = torch.bfloat16,
        **_kwargs,
    ):
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.model_def = model_def
        self.assets_root = model_def.get("minimax_h3_assets_root", "minimax_h3")

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

        self.transformer = _load_transformer(transformer_path, dtype)
        self.conditioner = _load_conditioner(text_encoder_filename, self.assets_root, dtype)
        self.vae = _load_video_vae(video_vae_path)
        self.audio_vae = _load_audio_vae(audio_vae_path)
        self.scheduler = MiniMaxH3Scheduler(shift=12.0)
        self.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
        self.__interrupt = False

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

    def _condition_normalization(self):
        """The pixel and latent normalization constants every conditioning still is encoded through."""
        means, stds = _keyframe_latent_stats_cpu()
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)
        return means, stds, pixel_mean, pixel_std

    def _encode_condition_still(self, image: Image.Image, normalization) -> tuple[torch.Tensor, tuple[int, int]]:
        """Encode one still into normalized, patchified latent rows, and report the latent size it encoded to.

        The posterior is sampled under a fixed seed rather than off the request's generator, and the sample is
        rounded through float16 before it is normalized on CPU — both are part of the reference contract.
        """
        means, stds, pixel_mean, pixel_std = normalization
        pixels = torch.from_numpy(np.array(image, dtype=np.uint8)).to(self.device)
        pixels = pixels.permute(2, 0, 1)[None, :, None]
        pixels = (pixels.float().div(255.0) - pixel_mean) / pixel_std
        moments = self.vae._encode_clip(pixels)
        posterior = DiagonalGaussianDistribution(moments)
        encoded = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
        encoded = encoded.to(torch.float16).float().cpu()
        return patchify_video_latents((encoded - means) / stds, self.patch_size), tuple(encoded.shape[-2:])

    def _encode_keyframes(
        self,
        images: list[Image.Image],
        latent_height: int,
        latent_width: int,
        generator: torch.Generator,
    ) -> torch.Tensor | None:
        if not images:
            return None

        normalization = self._condition_normalization()
        rows = []
        for image in images:
            if self._interrupt:
                return None
            frame_rows, _ = self._encode_condition_still(image, normalization)
            rows.append(frame_rows)

        clean_rows = torch.cat(rows).to(self.device)
        noise = keyframe_condition_noise(
            ((1, latent_height, latent_width),) * len(images),
            self.patch_size,
            24,
            generator=generator,
            device=self.device,
        )
        return self.scheduler.scale_noise(clean_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)

    def _prepare_reference_image(self, image) -> Image.Image | None:
        """Fit one `ref2va` reference still onto a canvas of its own.

        A keyframe is stretched or cropped onto the target's canvas because it *is* a frame of the output. A
        reference is not on the timeline, so it keeps its own framing and its canvas is resolved from its own aspect
        ratio -- landing, as always, on the multiple of 32 that the VAE's 16x compression and the 2x2 patchify need.
        """
        image = _tensor_to_pil(image)
        if image is None:
            return None
        height, width = resolve_canvas_size(image.width, image.height)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image

    def _encode_reference_images(
        self,
        images: list[Image.Image],
        generator: torch.Generator,
    ) -> tuple[torch.Tensor | None, list[MiniMaxH3PreparedReference]]:
        """Encode `ref2va` reference stills into their conditioning rows.

        Same encode and the same noise augmentation as a keyframe -- what differs is that each still reports the
        latent size it landed on, because the layout can no longer assume the target's.
        """
        if not images:
            return None, []

        normalization = self._condition_normalization()
        rows, references = [], []
        for image in images:
            if self._interrupt:
                return None, []
            still_rows, (latent_height, latent_width) = self._encode_condition_still(image, normalization)
            rows.append(still_rows)
            references.append(
                MiniMaxH3PreparedReference(kind="image", latent_height=latent_height, latent_width=latent_width)
            )

        clean_rows = torch.cat(rows).to(self.device)
        noise = keyframe_condition_noise(
            tuple((1, reference.latent_height, reference.latent_width) for reference in references),
            self.patch_size,
            24,
            generator=generator,
            device=self.device,
        )
        return self.scheduler.scale_noise(clean_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise), references

    def _load_reference_waveform(self, source) -> torch.Tensor | None:
        """Load one audio reference as the stereo waveform the audio VAE expects, at the VAE's own sample rate."""
        if source is None:
            return None
        import librosa

        waveform, _ = librosa.load(source, sr=self.audio_vae.config.sampling_rate, mono=False)
        waveform = torch.as_tensor(np.asarray(waveform), dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform[None]
        if waveform.shape[0] < MINIMAX_H3_AUDIO_CHANNELS:
            waveform = waveform.repeat(MINIMAX_H3_AUDIO_CHANNELS, 1)
        return waveform[:MINIMAX_H3_AUDIO_CHANNELS]

    def _encode_reference_audio(
        self,
        waveforms: list[torch.Tensor],
    ) -> tuple[torch.Tensor | None, list[MiniMaxH3PreparedReference]]:
        """Encode `ref2va` audio references into their conditioning rows.

        Two differences from the visual conditions. The audio VAE takes the stereo channels as the batch, so one
        reference encodes as `[2, 1, samples]`. And the rows are packed clean: audio references carry no noise
        augmentation, they sit at a conditioning timestep of their own.
        """
        if not waveforms:
            return None, []

        rows, references = [], []
        for waveform in waveforms:
            if self._interrupt:
                return None, []
            sample = waveform.to(device=self.device, dtype=torch.float32).unsqueeze(1)
            latents = self.audio_vae.encode(sample, return_dict=False)[0].mode().float()
            # Inverse of `unpack_audio_tokens`: [C, D, T] -> [C, T, D] -> channel-major rows.
            rows.append(latents.permute(0, 2, 1).reshape(-1, latents.shape[1]))
            references.append(MiniMaxH3PreparedReference(kind="audio", num_audio_latents=latents.shape[-1]))

        return torch.cat(rows).to(self.device), references

    @torch.inference_mode()
    def generate(
        self,
        input_prompt: str,
        image_start=None,
        image_end=None,
        input_ref_images=None,
        audio_guide=None,
        audio_guide2=None,
        frame_num: int = 124,
        height: int = 480,
        width: int = 864,
        sampling_steps: int = 20,
        seed: int | None = None,
        callback=None,
        **_kwargs,
    ):
        self._interrupt = False
        if not isinstance(input_prompt, str):
            raise ValueError("MiniMax H3 accepts one text prompt per generation.")
        if height % 32 or width % 32:
            raise ValueError(f"MiniMax H3 dimensions must be multiples of 32, got {width}x{height}.")

        frame_num = align_num_frames(int(frame_num))
        duration = frame_num / MINIMAX_H3_FPS
        if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax H3 supports {MINIMAX_H3_MIN_DURATION:g}-{MINIMAX_H3_MAX_DURATION:g}s at 24 fps; "
                f"the aligned request is {frame_num} frames ({duration:.3f}s)."
            )
        if int(sampling_steps) < 2:
            raise ValueError("MiniMax H3 needs at least two scheduler grid points.")

        keyframes = [item for item in (_tensor_to_pil(image_start), _tensor_to_pil(image_end)) if item is not None]
        anchors = tuple(
            anchor
            for anchor, item in (("first", image_start), ("last", image_end))
            if item is not None
        )
        keyframes = [
            prepare_keyframe_image(image, height, width, stretch=index == 0)
            for index, image in enumerate(keyframes)
        ]

        # `ref2va` references, in the order MiniMax-H3 packs them: stills first, then audio.
        reference_images = [
            prepared
            for prepared in (self._prepare_reference_image(image) for image in input_ref_images or ())
            if prepared is not None
        ]
        reference_waveforms = [
            waveform
            for waveform in (self._load_reference_waveform(source) for source in (audio_guide, audio_guide2))
            if waveform is not None
        ]
        if (reference_images or reference_waveforms) and keyframes:
            raise ValueError(
                "MiniMax H3 conditions on either keyframes or references, not both: a keyframe is a frame of the "
                "output, a reference is material that precedes it, and they occupy the same conditioning rows."
            )

        request_seed = int(torch.seed() if seed is None else seed)
        generator = torch.Generator(device=self.device).manual_seed(request_seed)
        num_latent_frames = video_latent_num_frames(frame_num)
        latent_height = height // self.vae.spatial_compression_ratio
        latent_width = width // self.vae.spatial_compression_ratio
        num_audio_latents = audio_latent_num_frames(frame_num)

        prompt_embeds, text_tags = self.conditioner(
            input_prompt,
            self.device,
            (keyframes or reference_images) or None,
            len(reference_waveforms),
        )
        if prompt_embeds is None or self._interrupt:
            return None

        # The conditioning rows are encoded before the layout is built, because a reference reports the latent size
        # it landed on and the layout needs it. Their noise is drawn first either way, which is what the request's
        # generator reproduces.
        audio_condition_rows = None
        if reference_images or reference_waveforms:
            condition_rows, references = self._encode_reference_images(reference_images, generator)
            audio_condition_rows, audio_references = self._encode_reference_audio(reference_waveforms)
            if self._interrupt:
                return None
            layout = build_ref2va_packed_sequence(
                text_tags,
                tuple(references + audio_references),
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                self.patch_size,
            )
        else:
            condition_rows = self._encode_keyframes(
                keyframes,
                latent_height,
                latent_width,
                generator,
            )
            if self._interrupt:
                return None
            layout = build_packed_sequence(
                text_tags,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                self.patch_size,
                anchors,
            )

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

        self.scheduler.set_timesteps(int(sampling_steps), device=self.device)
        self.audio_scheduler.set_timesteps(int(sampling_steps), device=self.device)
        timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps
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

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(timesteps))
        with tqdm(total=len(timesteps), desc="MiniMax H3 denoising") as progress:
            for index, (video_timestep, audio_timestep) in enumerate(zip(timesteps, audio_timesteps)):
                if self._interrupt:
                    return None
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

        audio_latents = unpack_audio_tokens(
            audio_rows[layout.num_condition_audio_rows :],
            num_audio_latents,
        )
        audio_mean = torch.tensor(AUDIO_LATENTS_MEAN, device=self.device).view(1, -1, 1)
        audio_std = torch.tensor(AUDIO_LATENTS_STD, device=self.device).view(1, -1, 1)
        audio_latents = audio_latents * audio_std + audio_mean
        audio = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        audio = audio.float().permute(1, 0, 2)[0].transpose(0, 1).cpu().numpy()
        return {
            "x": video[0],
            "audio": audio,
            "audio_sampling_rate": 32000,
        }
