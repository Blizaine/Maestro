import copy
import json
import math
import os
import re
import types
from typing import Callable, Iterator

import torch
import torchaudio
from accelerate import init_empty_weights
from shared.qtypes.int8_convrot import install_native_lora_forwards
from shared.utils import files_locator as fl

from .ltx_core.conditioning import AudioConditionByLatent, AudioConditionByLatentPrefix, AudioConditionByReferenceLatent
from .ltx_core.model.audio_vae import (
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    AudioProcessor,
    VocoderConfigurator,
)
from .ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from .ltx_core.model.upsampler import LatentUpsamplerConfigurator
from .ltx_core.model.video_vae import VideoDecoderConfigurator, VideoEncoderConfigurator
from .ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
from .ltx_core.text_encoders.gemma import (
    AUDIO_EMBEDDINGS_CONNECTOR_KEY_OPS,
    GemmaTextEmbeddingsConnectorModelConfigurator,
    GemmaTextEmbeddingsConnectorModel,
    TEXT_EMBEDDING_PROJECTION_KEY_OPS,
    TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS,
    VIDEO_EMBEDDINGS_CONNECTOR_KEY_OPS,
    build_gemma_text_encoder,
)
from .ltx_core.text_encoders.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from .ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from .ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from .ltx_core.types import AudioLatentShape, VideoPixelShape
from .inpainting import (
    _apply_ltx2_mask_blend,
    _build_outpainting_mask_cthw,
    _edge_extend_ltx2_masked_control_video,
    _get_outpainting_inner_rect,
    _merge_ltx2_masks,
    _normalize_outpainting_dims,
    _pad_ltx2_masked_control_video_tail,
    _paint_ltx2_inpaint_control_video,
    _paint_ltx2_masked_control_video,
)
from .ltx2_runtime import (
    LTX2_OUTPAINTING_LAPLACIAN_BLEND,
    LTX2_OUTPAINTING_LAPLACIAN_MASK_LOW_RES_DILATION,
    LTX2_OUTPAINTING_MARKER_RESIDUE_CLEANUP,
    LTX2_OUTPAINTING_SOURCE_FEATHER_PIXELS,
)
from .ltx_pipelines.distilled import DistilledPipeline
from .ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from .ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DEFAULT_NEGATIVE_PROMPT


_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
_SPATIAL_UPSCALER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
LTX2_USE_FP32_ROPE_FREQS = True
LTX2_OUTPAINT_GAMMA = 2.0
LTX2_DISABLE_STAGE2_WITH_CONTROL_VIDEO = True
LTX2_ENABLE_EMBEDDING_LORAS = False
LTX2_EMBEDDING_LORA_PREFIXES = (
    "text_embedding_projection.",
    "feature_extractor_linear.",
    "text_embeddings_connector.",
    "embeddings_connector.",
    "video_embeddings_connector.",
    "audio_embeddings_connector.",
)
# Comfy's lora_unet convention flattens module paths with underscores, but
# several real LTX module names also contain underscores. Restore those
# names before translating the remaining separators to dots. This mirrors
# current WanGP and keeps older LTX-2/2.3 adapters usable with LTX-2.5.
LTX2_COMFY_LORA_UNDERSCORED_NAMES = (
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_video_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "av_ca_v2a_gate_adaln_single",
    "audio_prompt_adaln_single",
    "audio_patchify_proj",
    "audio_to_video_attn",
    "prompt_adaln_single",
    "video_to_audio_attn",
    "audio_adaln_single",
    "transformer_blocks",
    "timestep_embedder",
    "audio_proj_out",
    "to_gate_logits",
    "patchify_proj",
    "adaln_single",
    "audio_attn1",
    "audio_attn2",
    "audio_ff",
    "linear_1",
    "linear_2",
    "proj_out",
    "k_norm",
    "q_norm",
    "to_out",
)


def _decord_frame_to_numpy(frame):
    """Convert a decord VideoReader frame to numpy regardless of which
    bridge mode is currently active.

    decord exposes a GLOBAL `bridge` setting. With the default 'native'
    bridge, `reader[i]` returns a decord.NDArray that has `.asnumpy()`.
    With the 'torch' bridge (set by `_apply_film_grain_to_file` in
    launch.py — properly cleaned up since the related fix but other
    callers might also flip it), `reader[i]` returns a torch.Tensor
    that has `.numpy()` (after `.cpu()` if it's on GPU).

    Symptom this helper prevents:
        AttributeError: 'Tensor' object has no attribute 'asnumpy'

    Used by every retake-pipeline frame-extraction site below.
    """
    # decord.NDArray path — native bridge
    if hasattr(frame, "asnumpy"):
        return frame.asnumpy()
    # torch.Tensor path — torch bridge
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    if hasattr(frame, "numpy"):
        return frame.numpy()
    # Already a numpy array (paranoid fallback — shouldn't happen with
    # decord but cheap to support).
    return frame


def _prepare_ltx_audio_waveform(input_waveform, target_channels):
    """Return continuation audio as ``(batch, channels, samples)``.

    File-backed audio slices enter LTX channel-first, while decoded native
    audio leaves the model sample-first.  Sliding-window continuation feeds
    that decoded tail back into the next pass, so treating every 2-D array as
    channel-first turns ``(samples, 1)`` into ``(1, samples, 1)`` and leaves a
    one-sample signal for the mel transform.  Accept both layouts explicitly
    and keep the batch dimension unambiguous.
    """

    waveform = torch.as_tensor(input_waveform, dtype=torch.float32, device="cpu")
    target_channels = max(1, int(target_channels or 1))

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        first, second = (int(value) for value in waveform.shape)
        if first <= 8 and second > 8:
            pass  # channels, samples
        elif second <= 8 and first > 8:
            waveform = waveform.transpose(0, 1)  # samples, channels
        elif first in (1, target_channels):
            pass
        elif second in (1, target_channels):
            waveform = waveform.transpose(0, 1)
        else:
            raise ValueError(
                "LTX continuation audio must be mono or channel-oriented; "
                f"got {tuple(waveform.shape)}."
            )
    elif waveform.ndim == 3:
        if int(waveform.shape[0]) != 1:
            raise ValueError(
                "LTX continuation audio supports one batch at a time; "
                f"got {tuple(waveform.shape)}."
            )
        if int(waveform.shape[1]) <= 8 and int(waveform.shape[2]) > 8:
            return waveform.contiguous()
        if int(waveform.shape[2]) <= 8 and int(waveform.shape[1]) > 8:
            return waveform.transpose(1, 2).contiguous()
        if int(waveform.shape[1]) in (1, target_channels):
            return waveform.contiguous()
        if int(waveform.shape[2]) in (1, target_channels):
            return waveform.transpose(1, 2).contiguous()
        raise ValueError(
            "LTX continuation audio must be batch/channel/sample oriented; "
            f"got {tuple(waveform.shape)}."
        )
    else:
        raise ValueError(
            "LTX continuation audio must be one-, two-, or three-dimensional; "
            f"got {tuple(waveform.shape)}."
        )

    return waveform.unsqueeze(0).contiguous()


def _resolve_retake_pipeline_models(pipeline):
    """Return the model container used by the native Retake pipeline.

    DistilledPipeline exposes ``models`` directly. The LTX-2 Dev and
    reference workflows use TI2VidTwoStagesPipeline, whose equivalent
    container is ``stage_1_models``. Retake is a single-stage diffusion pass,
    so the first-stage container is the correct source for either workflow.
    """
    models = getattr(pipeline, "models", None)
    if models is None:
        models = getattr(pipeline, "stage_1_models", None)
    if models is None:
        raise RuntimeError(
            "The loaded LTX-2 pipeline does not expose models for Retake."
        )
    return models


def _uses_distilled_pipeline_dispatch(
    active_pipeline,
    active_official_outpaint=None,
) -> bool:
    """Recognize distilled dispatch, including an explicit Outpaint adapter.

    The canonical path uses the baked distilled pipeline. Retain explicit
    adapter support for compatible Dev checkpoints and runtime wrappers so
    the source-attention arguments always reach DistilledPipeline.
    """
    return bool(
        active_official_outpaint is not None
        or isinstance(active_pipeline, DistilledPipeline)
    )


def _uses_native_two_stage_dispatch(
    base_pipeline,
    active_official_outpaint=None,
) -> bool:
    """Return whether the original Dev two-stage pipeline should run."""
    return bool(
        active_official_outpaint is None
        and isinstance(base_pipeline, TI2VidTwoStagesPipeline)
    )


def _select_video_conditioning_generation_mask(
    official_outpaint_pipeline,
    input_frames,
    input_masks,
    input_masks2,
):
    """Select the source-preservation mask forwarded to DistilledPipeline."""
    if not official_outpaint_pipeline:
        return None
    return input_masks if input_frames is not None else input_masks2


def _normalize_config(config_value):
    if isinstance(config_value, dict):
        return config_value
    if isinstance(config_value, (bytes, bytearray, memoryview)):
        try:
            config_value = bytes(config_value).decode("utf-8")
        except Exception:
            return {}
    if isinstance(config_value, str):
        try:
            return json.loads(config_value)
        except json.JSONDecodeError:
            return {}
    return {}


def _load_config_from_checkpoint(path, fallback_config_path: str | None = None):
    from mmgp import quant_router

    if isinstance(path, (list, tuple)):
        if not path:
            return {}
        path = path[0]
    if not path:
        return {}

    def _read_config_metadata(one_path: str) -> dict:
        if not one_path:
            return {}
        _, metadata = quant_router.load_metadata_state_dict(one_path)
        if not metadata:
            return {}
        return _normalize_config(metadata.get("config"))

    config = _read_config_metadata(path)
    if config:
        return config
    if not fallback_config_path:
        return {}
    try:
        with open(fallback_config_path, "r", encoding="utf-8") as reader:
            return _normalize_config(json.load(reader))
    except Exception:
        return {}


def _strip_model_prefix(key: str) -> str:
    for prefix in ("model.", "velocity_model."):
        if key.startswith(prefix):
            return _strip_model_prefix(key[len(prefix) :])
    return key


def _apply_sd_ops(state_dict: dict, quantization_map: dict | None, sd_ops):
    if sd_ops is not None:
        has_match = False
        for key in state_dict.keys():
            key = _strip_model_prefix(key)
            if sd_ops.apply_to_key(key) is not None:
                has_match = True
                break
        if not has_match:
            new_sd = {_strip_model_prefix(k): v for k, v in state_dict.items()}
            new_qm = {}
            if quantization_map:
                new_qm = {_strip_model_prefix(k): v for k, v in quantization_map.items()}
            return new_sd, new_qm

    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        if sd_ops is None:
            new_sd[key] = value
            continue
        else:
            new_key = sd_ops.apply_to_key(key)
            if new_key is None:
                continue
            new_pairs = sd_ops.apply_to_key_value(new_key, value)
        for pair in new_pairs:
            new_sd[pair.new_key] = pair.new_value

    new_qm = {}
    if quantization_map:
        for key, value in quantization_map.items():
            key = _strip_model_prefix(key)
            if sd_ops is None:
                new_key = key
            else:
                new_key = sd_ops.apply_to_key(key)
                if new_key is None:
                    continue
            new_qm[new_key] = value
    return new_sd, new_qm


def _make_sd_postprocess(sd_ops):
    def postprocess(state_dict, quantization_map):
        return _apply_sd_ops(state_dict, quantization_map, sd_ops)

    return postprocess


def _split_vae_state_dict(state_dict: dict, prefix: str):
    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        if key.startswith(prefix):
            key = key[len(prefix) :]
        elif key.startswith(("encoder.", "decoder.", "per_channel_statistics.")):
            key = key
        else:
            continue
        if key.startswith("per_channel_statistics."):
            suffix = key[len("per_channel_statistics.") :]
            new_sd[f"encoder.per_channel_statistics.{suffix}"] = value.clone()
            new_sd[f"decoder.per_channel_statistics.{suffix}"] = value.clone()
        else:
            new_sd[key] = value

    return new_sd, {}


def _make_vae_postprocess(prefix: str):
    def postprocess(state_dict, quantization_map):
        return _split_vae_state_dict(state_dict, prefix)

    return postprocess


def _split_diffusion_vae_state_dict(state_dict: dict, prefix: str):
    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        if key.startswith(prefix):
            key = key[len(prefix):]
        elif not key.startswith(("encoder.", "decoder.", "per_channel_statistics.")):
            continue
        if key == "decoder.type_emb":
            continue
        if key.startswith("per_channel_statistics."):
            suffix = key[len("per_channel_statistics."):]
            new_sd[f"encoder.per_channel_statistics.{suffix}"] = value.clone()
            new_sd[f"decoder.per_channel_statistics.{suffix}"] = value.clone()
        elif key.startswith("decoder.t_embedder.mlp.0."):
            new_sd[key.replace(
                "decoder.t_embedder.mlp.0.",
                "decoder.t_embedder.timestep_embedder.linear_1.",
            )] = value
        elif key.startswith("decoder.t_embedder.mlp.2."):
            new_sd[key.replace(
                "decoder.t_embedder.mlp.2.",
                "decoder.t_embedder.timestep_embedder.linear_2.",
            )] = value
        elif ".attn.qkv." in key:
            prefix_key, suffix = key.split(".attn.qkv.", 1)
            q, k, v = value.chunk(3, dim=0)
            new_sd[f"{prefix_key}.attn.qkv.to_q.{suffix}"] = q
            new_sd[f"{prefix_key}.attn.qkv.to_k.{suffix}"] = k
            new_sd[f"{prefix_key}.attn.qkv.to_v.{suffix}"] = v
        else:
            new_sd[key] = value
    return new_sd, {}


def _make_diffusion_vae_postprocess(prefix: str):
    def postprocess(state_dict, quantization_map):
        return _split_diffusion_vae_state_dict(state_dict, prefix)

    return postprocess


def _diffusion_vae_encoder_config(config: dict) -> dict:
    encoder = config["vae"]["encoder"]
    return {
        "vae": {
            "dims": encoder["dims"],
            "in_channels": encoder["in_channels"],
            "latent_channels": encoder["out_channels"],
            "encoder_blocks": encoder["blocks"],
            "patch_size": encoder["patch_size"],
            "norm_layer": encoder["norm_layer"],
            "latent_log_var": encoder["latent_log_var"],
            "encoder_spatial_padding_mode": encoder["spatial_padding_mode"],
        }
    }


class _AudioVAEWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        per_stats = getattr(decoder, "per_channel_statistics", None)
        if per_stats is not None:
            self.per_channel_statistics = per_stats
        self.decoder = decoder


class _VAEContainer(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class _ExternalConnectorWrapper:
    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def __call__(self, *args, **kwargs):
        return self._module(*args, **kwargs)


class LTX2SuperModel(torch.nn.Module):
    def __init__(self, ltx2_model: "LTX2") -> None:
        super().__init__()
        object.__setattr__(self, "_ltx2", ltx2_model)

        transformer = ltx2_model.model
        velocity_model = getattr(transformer, "velocity_model", transformer)
        self.velocity_model = velocity_model
        split_map = getattr(transformer, "split_linear_modules_map", None)
        if split_map is not None:
            self.split_linear_modules_map = split_map

        self.text_embedding_projection = ltx2_model.text_embedding_projection
        self.video_embeddings_connector = ltx2_model.video_embeddings_connector
        self.audio_embeddings_connector = ltx2_model.audio_embeddings_connector

    @property
    def _interrupt(self) -> bool:
        return self._ltx2._interrupt

    @_interrupt.setter
    def _interrupt(self, value: bool) -> None:
        self._ltx2._interrupt = value

    def forward(self, *args, **kwargs):
        return self._ltx2.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._ltx2.generate(*args, **kwargs)

    def get_trans_lora(self):
        return self, None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._ltx2, name)


class _LTX2VAEHelper:
    def __init__(self, block_size: int = 64) -> None:
        self.block_size = block_size

    def get_VAE_tile_size(
        self,
        vae_config: int,
        device_mem_capacity: float,
        mixed_precision: bool,
        output_height: int | None = None,
        output_width: int | None = None,
    ) -> int | tuple[int, int]:
        if vae_config >= 4:
            vae_config = 0

        if vae_config == 0:
            if mixed_precision:
                device_mem_capacity = device_mem_capacity / 1.5
            if device_mem_capacity >= 24000:
                use_vae_config = 1
            elif device_mem_capacity >= 8000:
                use_vae_config = 2
            else:
                use_vae_config = 3
        else:
            use_vae_config = vae_config

        ref_size = output_height if output_height is not None else output_width
        if ref_size is not None and ref_size > 480:
            use_vae_config += 1

        spatial_tile_size = 128
        if use_vae_config <= 1:
            spatial_tile_size = 0
        elif use_vae_config == 2:
            spatial_tile_size = 512
        elif use_vae_config == 3:
            spatial_tile_size = 256

        return spatial_tile_size


def _attach_lora_preprocessor(transformer: torch.nn.Module) -> None:
    def preprocess_loras(self: torch.nn.Module, model_type: str, sd: dict) -> dict:
        if not sd:
            return sd
        module_names = getattr(self, "_lora_module_names", None)
        if module_names is None:
            module_names = {name for name, _ in self.named_modules()}
            self._lora_module_names = module_names

        def split_lora_key(lora_key: str) -> tuple[str | None, str]:
            if lora_key.endswith(".alpha"):
                return lora_key[: -len(".alpha")], ".alpha"
            if lora_key.endswith(".diff"):
                return lora_key[: -len(".diff")], ".diff"
            if lora_key.endswith(".diff_b"):
                return lora_key[: -len(".diff_b")], ".diff_b"
            if lora_key.endswith(".dora_scale"):
                return lora_key[: -len(".dora_scale")], ".dora_scale"
            pos = lora_key.rfind(".lora_")
            if pos > 0:
                return lora_key[:pos], lora_key[pos:]
            return None, ""

        new_sd = {}
        dropped_keys = []
        for key, value in sd.items():
            original_key = key
            if key.startswith("model."):
                key = key[len("model.") :]
            if key.startswith("diffusion_model."):
                key = key[len("diffusion_model.") :]
            if key.startswith("transformer."):
                key = key[len("transformer.") :]

            module_name, suffix = split_lora_key(key)
            if not module_name:
                dropped_keys.append(original_key)
                continue
            if module_name.startswith("lora_unet_"):
                module_name = module_name[len("lora_unet_") :]
                for name in LTX2_COMFY_LORA_UNDERSCORED_NAMES:
                    module_name = module_name.replace(
                        name,
                        name.replace("_", "\0"),
                    )
                module_name = module_name.replace("_", ".").replace(
                    "\0",
                    "_",
                )
            if (
                not LTX2_ENABLE_EMBEDDING_LORAS
                and module_name.startswith(LTX2_EMBEDDING_LORA_PREFIXES)
            ):
                continue
            if module_name.startswith("embeddings_connector."):
                module_name = (
                    "video_embeddings_connector."
                    f"{module_name[len('embeddings_connector.'):]}"
                )
            if module_name.startswith("feature_extractor_linear."):
                module_name = (
                    "text_embedding_projection."
                    f"{module_name[len('feature_extractor_linear.'):]}"
                )
            if module_name not in module_names:
                prefixed_name = f"velocity_model.{module_name}"
                if (
                    module_name.endswith(".to_out")
                    and f"{prefixed_name}.0" in module_names
                ):
                    module_name = f"{prefixed_name}.0"
                elif prefixed_name in module_names:
                    module_name = prefixed_name
                else:
                    dropped_keys.append(original_key)
                    continue
            elif (
                module_name.endswith(".to_out")
                and f"{module_name}.0" in module_names
            ):
                module_name = f"{module_name}.0"
            new_sd[f"{module_name}{suffix}"] = value
        if dropped_keys:
            sample = ", ".join(dropped_keys[:8])
            if len(dropped_keys) > 8:
                sample += ", ..."
            raise ValueError(
                f"LTX2 LoRA preprocessing dropped {len(dropped_keys)} unmatched keys for model '{model_type}': {sample}"
            )
        return new_sd

    transformer.preprocess_loras = types.MethodType(preprocess_loras, transformer)


def _coerce_image_list(image_value):
    if isinstance(image_value, list):
        return image_value[0] if image_value else None
    return image_value


def _to_latent_index(frame_idx: int, stride: int) -> int:
    frame_idx = int(frame_idx)
    stride = int(stride)
    if frame_idx <= 0:
        return 0
    # Causal LTX VAEs keep pixel frame 0 in its own latent slot.
    return (frame_idx - 1) // stride + 1


def _normalize_tiling_size(tile_size: int) -> int:
    tile_size = int(tile_size)
    if tile_size <= 0:
        return 0
    tile_size = max(64, tile_size)
    if tile_size % 32 != 0:
        tile_size = int(math.ceil(tile_size / 32) * 32)
    return tile_size


def _normalize_temporal_tiling_size(tile_frames: int) -> int:
    tile_frames = int(tile_frames)
    if tile_frames <= 0:
        return 0
    tile_frames = max(16, tile_frames)
    if tile_frames % 8 != 0:
        tile_frames = int(math.ceil(tile_frames / 8) * 8)
    return tile_frames


def _normalize_temporal_overlap(overlap_frames: int, tile_frames: int) -> int:
    overlap_frames = max(0, int(overlap_frames))
    if overlap_frames % 8 != 0:
        overlap_frames = int(round(overlap_frames / 8) * 8)
    overlap_frames = max(0, min(overlap_frames, max(0, tile_frames - 8)))
    return overlap_frames


def _build_tiling_config(tile_size: int | tuple | list | None, fps: float | None) -> TilingConfig | None:
    temporal_tiling_divisor = 1
    spatial_config = None
    if isinstance(tile_size, (tuple, list)):
        if len(tile_size) == 0:
            tile_size = None
        else:
            if len(tile_size) > 1:
                temporal_tiling_divisor = max(1, int(tile_size[0] or 1))
            tile_size = tile_size[-1]
    if tile_size is not None:
        tile_size = _normalize_tiling_size(tile_size)
        if tile_size > 0:
            overlap = max(0, tile_size // 4)
            overlap = int(math.floor(overlap / 32) * 32)
            if overlap >= tile_size:
                overlap = max(0, tile_size - 32)
            spatial_config = SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=overlap)

    temporal_config = None
    if fps is not None and fps > 0:
        temporal_tiling_divisor = max(1, temporal_tiling_divisor)
        tile_frames = _normalize_temporal_tiling_size(int(math.ceil(float(fps) * 5.0 / temporal_tiling_divisor)))
        if tile_frames > 0:
            overlap_frames = int(round(tile_frames * 3 / 8))
            overlap_frames = _normalize_temporal_overlap(overlap_frames, tile_frames)
            temporal_config = TemporalTilingConfig(
                tile_size_in_frames=tile_frames,
                tile_overlap_in_frames=overlap_frames,
            )

    if spatial_config is None and temporal_config is None:
        return None
    return TilingConfig(spatial_config=spatial_config, temporal_config=temporal_config)


def _infer_ic_lora_downscale_factor(loras_selected) -> int | None:
    factors = []
    for lora_path in loras_selected or []:
        name = os.path.basename(str(lora_path)).lower()
        if "ic-lora" not in name:
            continue
        match = re.search(r"-ref([0-9]+(?:\.[0-9]+)?)", name)
        if not match:
            factors.append(1)
            continue
        ref_ratio = float(match.group(1))
        if ref_ratio <= 0:
            factors.append(1)
            continue
        factors.append(max(1, int(round(1.0 / ref_ratio))))
    if not factors:
        return None
    unique_factors = sorted(set(factors))
    if len(unique_factors) > 1:
        raise ValueError(f"Conflicting IC-LoRA reference downscale factors in selected LoRAs: {unique_factors}")
    return unique_factors[0]


def _apply_gamma_to_media(media_tensor: torch.Tensor | None, gamma: float) -> bool:
    if media_tensor is None or not torch.is_tensor(media_tensor) or media_tensor.dim() < 2 or gamma <= 0 or media_tensor.numel() == 0:
        return False
    exponent = 1.0 / float(gamma)
    if media_tensor.dtype == torch.uint8:
        corrected = media_tensor.to(dtype=torch.float32).div_(255.0).clamp_(0.0, 1.0).pow_(exponent)
        media_tensor.copy_(corrected.mul_(255.0).round_().clamp_(0.0, 255.0).to(dtype=torch.uint8))
        return True
    corrected = media_tensor.to(dtype=torch.float32).add_(1.0).mul_(0.5).clamp_(0.0, 1.0).pow_(exponent)
    media_tensor.copy_(corrected.mul_(2.0).sub_(1.0).to(dtype=media_tensor.dtype))
    return True


def _apply_gamma_to_video_rect(video_tensor: torch.Tensor | None, rect: tuple[int, int, int, int] | None, gamma: float) -> bool:
    if video_tensor is None or not torch.is_tensor(video_tensor) or rect is None or video_tensor.dim() < 4:
        return False
    top, bottom, left, right = rect
    region = video_tensor[..., top:bottom, left:right]
    return _apply_gamma_to_media(region, gamma)


def _collect_video_chunks(
    video: Iterator[torch.Tensor] | torch.Tensor,
    interrupt_check: Callable[[], bool] | None = None,
    expected_frames: int | None = None,
    expected_height: int | None = None,
    expected_width: int | None = None,
) -> torch.Tensor | None:
    iterator = None
    if video is None:
        return None
    try:
        if torch.is_tensor(video):
            frames = video
            if expected_height is not None or expected_width is not None:
                frames = frames[:, :expected_height, :expected_width]
            return frames.permute(3, 0, 1, 2)
        else:
            iterator = iter(video)
            video_tensor = None
            write_pos = 0
            for chunk in iterator:
                if interrupt_check is not None and interrupt_check():
                    return None
                if chunk is None:
                    continue
                chunk = chunk if torch.is_tensor(chunk) else torch.tensor(chunk)
                if expected_height is not None or expected_width is not None:
                    chunk = chunk[:, :expected_height, :expected_width]
                if video_tensor is None:
                    channels = int(chunk.shape[-1])
                    frame_capacity = int(expected_frames) if expected_frames is not None and expected_frames > 0 else int(chunk.shape[0])
                    video_tensor = torch.empty(
                        (channels, frame_capacity, chunk.shape[1], chunk.shape[2]),
                        dtype=chunk.dtype,
                        device=chunk.device,
                    )
                frame_count = min(int(chunk.shape[0]), int(video_tensor.shape[1] - write_pos))
                if frame_count <= 0:
                    break
                video_tensor[:, write_pos : write_pos + frame_count].copy_(chunk[:frame_count].permute(3, 0, 1, 2))
                write_pos += frame_count
            if video_tensor is None:
                return None
            return video_tensor[:, :write_pos]
    finally:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
    # frames = frames.to(dtype=torch.float32).div_(127.5).sub_(1.0)
    # return frames.permute(3, 0, 1, 2).contiguous()


class LTX2:
    def __init__(
        self,
        model_filename,
        model_type: str,
        base_model_type: str,
        model_def: dict,
        dtype: torch.dtype = torch.bfloat16,
        VAE_dtype: torch.dtype = torch.float32,
        text_encoder_filename: str | None = None,
        text_encoder_filepath = None,
        checkpoint_paths: dict | None = None,
    ) -> None:
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.VAE_dtype = VAE_dtype
        self.base_model_type = base_model_type
        self.model_def = model_def
        self._interrupt = False
        self.vae = _LTX2VAEHelper()
        from .ltx_core.model.transformer import rope as rope_utils

        self.use_fp32_rope_freqs = bool(model_def.get("ltx2_rope_freqs_fp32", LTX2_USE_FP32_ROPE_FREQS))
        rope_utils.set_use_fp32_rope_freqs(self.use_fp32_rope_freqs)

        if isinstance(model_filename, (list, tuple)):
            if not model_filename:
                raise ValueError("Missing LTX-2 checkpoint path.")
            transformer_path = list(model_filename)
        else:
            transformer_path = model_filename
        component_paths = checkpoint_paths or {}
        if component_paths:
            transformer_path = component_paths.get("transformer")
            if not transformer_path:
                raise ValueError("Missing transformer path in checkpoint_paths.")

        gemma_root = text_encoder_filepath if text_encoder_filename is None else text_encoder_filename
        if not gemma_root:
            raise ValueError("Missing Gemma text encoder path.")
        if component_paths:
            spatial_upsampler_path = component_paths.get("spatial_upsampler")
        else:
            spatial_upsampler_path = None
        if not spatial_upsampler_path:
            spatial_upsampler_name = model_def.get("ltx2_spatial_upscaler_file", _SPATIAL_UPSCALER_FILENAME)
            spatial_upsampler_path = fl.locate_file(spatial_upsampler_name)

        # Internal FP8 handling is disabled; mmgp manages quantization/dtypes.
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")

        pipeline_models = self._init_models(
            transformer_path=transformer_path,
            component_paths=component_paths,
            gemma_root=gemma_root,
            gemma_side_files_root=text_encoder_filepath,
            spatial_upsampler_path=spatial_upsampler_path,
        )

        if pipeline_kind == "distilled":
            self.pipeline = DistilledPipeline(
                device=self.device,
                models=pipeline_models,
            )
        else:
            self.pipeline = TI2VidTwoStagesPipeline(
                device=self.device,
                stage_1_models=pipeline_models,
                stage_2_models=pipeline_models,
            )
        self._build_diffuser_model()

    def finalize_loras(self) -> None:
        """Keep INT8 ConvRot base math intact after MMGP attaches LoRAs."""

        lora_target = getattr(self, "diffuser_model", self.model)
        installed = install_native_lora_forwards(lora_target)
        if installed:
            print(
                "[LTX LoRA] Preserved native INT8 ConvRot activation math for "
                f"{installed} adapter-targeted layer(s)."
            )

    def _init_models(
        self,
        transformer_path,
        component_paths: dict,
        gemma_root: str,
        gemma_side_files_root: str | None,
        spatial_upsampler_path: str,
    ):
        from mmgp import offload as mmgp_offload

        fallback_config_path = component_paths.get("model_config") if component_paths else None
        base_config = _load_config_from_checkpoint(transformer_path, fallback_config_path=fallback_config_path)
        if not base_config:
            raise ValueError("Missing config in transformer checkpoint.")

        def _component_path(key: str):
            if component_paths:
                path = component_paths.get(key)
                if not path:
                    raise ValueError(f"Missing '{key}' path in checkpoint_paths.")
                return path
            return transformer_path

        def _component_config(path):
            config = _load_config_from_checkpoint(path, fallback_config_path=fallback_config_path)
            return config or base_config

        def _load_component(model, path, sd_ops=None, postprocess=None, ignore_unused_weights=False):
            if postprocess is None and sd_ops is not None:
                postprocess = _make_sd_postprocess(sd_ops)
            mmgp_offload.load_model_data(
                model,
                path,
                postprocess_sd=postprocess,
                default_dtype=self.dtype,
                writable_tensors=False,
                ignore_missing_keys=False,
                ignore_unused_weights=ignore_unused_weights,
            )
            model.eval().requires_grad_(False)
            return model

        transformer_sd_ops = LTXV_MODEL_COMFY_RENAMING_MAP
        with init_empty_weights():
            velocity_model = LTXModelConfigurator.from_config(base_config)
        velocity_model = _load_component(velocity_model, transformer_path, transformer_sd_ops, ignore_unused_weights=True)
        transformer = X0Model(velocity_model)
        transformer.eval().requires_grad_(False)
        VAE_URLs = self.model_def.get("VAE_URLs", None)
        video_vae_path =  fl.locate_file(VAE_URLs[0]) if VAE_URLs is not None and len(VAE_URLs) else _component_path("video_vae")
        video_config = copy.deepcopy(_component_config(video_vae_path))
        diffusion_vae = (
            video_config.get("vae", {}).get("_class_name")
            == "CausalDiffusionVAE"
        )
        if diffusion_vae:
            print("[Maestro][LTX2] Loading NAD Diffusion Decoder.")
        video_config_vae = video_config.setdefault("vae", {})
        video_config_vae["spatial_padding_mode"] = "reflect"
        video_config_vae["encoder_spatial_padding_mode"] = "reflect"
        video_config_vae["decoder_spatial_padding_mode"] = "reflect"
        # print("[LTX2 VAE Config] forcing encoder/decoder spatial_padding_mode=reflect")
        with init_empty_weights():
            video_encoder = VideoEncoderConfigurator.from_config(
                _diffusion_vae_encoder_config(video_config)
                if diffusion_vae
                else video_config
            )
            video_decoder = (
                DiffusionVideoDecoder.from_config(video_config)
                if diffusion_vae
                else VideoDecoderConfigurator.from_config(video_config)
            )
            video_vae = _VAEContainer(video_encoder, video_decoder)
        vae_postprocess = (
            _make_diffusion_vae_postprocess("vae.")
            if diffusion_vae
            else _make_vae_postprocess("vae.")
        )
        video_vae = _load_component(
            video_vae,
            video_vae_path,
            postprocess=vae_postprocess,
            ignore_unused_weights=True,
        )
        video_encoder = video_vae.encoder
        video_decoder = video_vae.decoder

        audio_vae_path = _component_path("audio_vae")
        audio_config = _component_config(audio_vae_path)
        with init_empty_weights():
            audio_encoder = AudioEncoderConfigurator.from_config(audio_config)
            audio_decoder = AudioDecoderConfigurator.from_config(audio_config)
            audio_vae = _VAEContainer(audio_encoder, audio_decoder)
        audio_vae = _load_component(audio_vae, audio_vae_path, postprocess=_make_vae_postprocess("audio_vae."))
        audio_encoder = audio_vae.encoder
        audio_decoder = audio_vae.decoder

        vocoder_path = _component_path("vocoder")
        vocoder_config = _component_config(vocoder_path)
        with init_empty_weights():
            vocoder = VocoderConfigurator.from_config(vocoder_config)
        vocoder = _load_component(vocoder, vocoder_path, VOCODER_COMFY_KEYS_FILTER)

        text_projection_path = _component_path("text_embedding_projection")
        text_projection_config = _component_config(text_projection_path)
        with init_empty_weights():
            text_embedding_projection = GemmaFeaturesExtractorProjLinear.from_config(text_projection_config)
        text_embedding_projection = _load_component( text_embedding_projection, text_projection_path, TEXT_EMBEDDING_PROJECTION_KEY_OPS )

        self.split_text_connectors = "video_embeddings_connector" in component_paths
        if self.split_text_connectors:
            video_connector_path = _component_path("video_embeddings_connector")
            video_connector_config = _component_config(video_connector_path)
            with init_empty_weights():
                video_embeddings_connector = (
                    Embeddings1DConnectorConfigurator.from_config(
                        video_connector_config
                    )
                )
            video_embeddings_connector = _load_component(
                video_embeddings_connector,
                video_connector_path,
                VIDEO_EMBEDDINGS_CONNECTOR_KEY_OPS,
            )

            audio_connector_path = _component_path("audio_embeddings_connector")
            audio_connector_config = _component_config(audio_connector_path)
            with init_empty_weights():
                audio_embeddings_connector = (
                    AudioEmbeddings1DConnectorConfigurator.from_config(
                        audio_connector_config
                    )
                )
            audio_embeddings_connector = _load_component(
                audio_embeddings_connector,
                audio_connector_path,
                AUDIO_EMBEDDINGS_CONNECTOR_KEY_OPS,
            )
            text_embeddings_connector = GemmaTextEmbeddingsConnectorModel(
                video_embeddings_connector,
                audio_embeddings_connector,
            )
        else:
            text_connector_path = _component_path("text_embeddings_connector")
            text_connector_config = _component_config(text_connector_path)
            with init_empty_weights():
                text_embeddings_connector = GemmaTextEmbeddingsConnectorModelConfigurator.from_config(text_connector_config)
            text_embeddings_connector = _load_component( text_embeddings_connector, text_connector_path, TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS )

        text_encoder = build_gemma_text_encoder(
            gemma_root,
            default_dtype=self.dtype,
            side_files_root=gemma_side_files_root,
        )
        text_encoder.eval().requires_grad_(False)

        upsampler_config = _load_config_from_checkpoint(spatial_upsampler_path)
        with init_empty_weights():
            spatial_upsampler = LatentUpsamplerConfigurator.from_config(upsampler_config)
        spatial_upsampler = _load_component(spatial_upsampler, spatial_upsampler_path, None)

        self.text_encoder = text_encoder
        self.text_embedding_projection = text_embedding_projection
        self.text_embeddings_connector = text_embeddings_connector
        self.video_embeddings_connector = text_embeddings_connector.video_embeddings_connector
        self.audio_embeddings_connector = text_embeddings_connector.audio_embeddings_connector
        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.audio_encoder = audio_encoder
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.spatial_upsampler = spatial_upsampler
        self.model = transformer
        self.model2 = None

        return types.SimpleNamespace(
            text_encoder=self.text_encoder,
            text_embedding_projection=self.text_embedding_projection,
            text_embeddings_connector=self.text_embeddings_connector,
            video_encoder=self.video_encoder,
            video_decoder=self.video_decoder,
            audio_encoder=self.audio_encoder,
            audio_decoder=self.audio_decoder,
            vocoder=self.vocoder,
            spatial_upsampler=self.spatial_upsampler,
            transformer=self.model,
        )

    def _detach_text_encoder_connectors(self) -> None:
        text_encoder = getattr(self, "text_encoder", None)
        if text_encoder is None:
            return
        connectors = {}
        feature_extractor = getattr(self, "text_embedding_projection", None)
        video_connector = getattr(self, "video_embeddings_connector", None)
        audio_connector = getattr(self, "audio_embeddings_connector", None)
        if feature_extractor is not None:
            connectors["feature_extractor_linear"] = feature_extractor
        if video_connector is not None:
            connectors["embeddings_connector"] = video_connector
        if audio_connector is not None:
            connectors["audio_embeddings_connector"] = audio_connector
        if not connectors:
            return
        for name, module in connectors.items():
            if name in text_encoder._modules:
                del text_encoder._modules[name]
            setattr(text_encoder, name, _ExternalConnectorWrapper(module))
        self._text_connectors = connectors

    def _build_diffuser_model(self) -> None:
        self._detach_text_encoder_connectors()
        self.diffuser_model = LTX2SuperModel(self)
        _attach_lora_preprocessor(self.diffuser_model)


    def get_trans_lora(self):
        trans = getattr(self, "diffuser_model", None)
        if trans is None:
            trans = self.model
        return trans, None

    def get_loras_transformer(self, get_model_recursive_prop, model_type, video_prompt_type, base_model_type=None, model_def = None, **kwargs):
        control_map = {
            "O": "pose_align",
            "P": "pose",
            "D": "depth",
            "T": "depth_temporal",
            "E": "canny",
        }
        loras = []
        loras_mult = []
        video_prompt_type = video_prompt_type or ""
        resolved_base_model_type = base_model_type or self.base_model_type
        official_outpaint_stack = bool(
            kwargs.get("outpaint_official_stack", False)
        )

        def _ensure_lora_dir_file(filename, repo_id):
            """Resolve or download a transformer LoRA in LTX's LoRA folder."""
            import sys

            wgp_module = sys.modules.get("wgp") or sys.modules.get("app.wgp")
            if (
                wgp_module is not None
                and hasattr(wgp_module, "resolve_lora_path")
            ):
                target_path = wgp_module.resolve_lora_path(
                    model_type,
                    filename,
                )
            else:
                target_path = os.path.join("loras", "ltx2", filename)
            if os.path.isfile(target_path):
                return target_path

            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                from huggingface_hub import hf_hub_download

                print(
                    f"[LTX2] Official Outpaint dependency not found - "
                    f"downloading {filename}..."
                )
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=os.path.dirname(target_path),
                )
            except Exception as error:
                print(
                    f"[LTX2] Failed to download {filename}: {error}"
                )
                return None

        if (
            official_outpaint_stack
            and model_def.get("ltx2_pipeline", "two_stage")
            != "distilled"
        ):
            distilled_filename = (
                "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
            )
            distilled_path = _ensure_lora_dir_file(
                distilled_filename,
                "Lightricks/LTX-2.3",
            )
            if distilled_path and os.path.isfile(distilled_path):
                loras.append(distilled_path)
                loras_mult.append("0.5;0.5")
                print(
                    "[LTX2] Official Outpaint stack: loaded Distilled "
                    f"1.1 LoRA {distilled_path} "
                    "(strength=0.5 in both passes)."
                )
            else:
                raise FileNotFoundError(
                    "The official LTX-2.3 Outpaint workflow requires "
                    f"{distilled_filename}."
                )

        # ID-LoRA: voice-identity LoRA — auto-load when the caller provided a
        # voice_reference. Lives BEFORE the distilled-only early return so it
        # works on dev models too (which return early below with no other
        # auto-loaded LoRAs). Per upstream WanGP v11.77, the CelebVHQ ID-LoRA
        # works on both dev (two-stage) and distilled pipelines despite being
        # trained primarily on dev.
        #
        # Multiplier convention matches upstream:
        #   - distilled (single-phase): scalar 1.0
        #   - dev (two-phase): "1;0" — phase 1 only. Phase 2 in two_stage
        #     doesn't process audio so the LoRA has nothing to amplify there.
        voice_reference = kwargs.get("voice_reference")
        if voice_reference:
            from . import ltx2_handler
            spec = ltx2_handler._get_arch_spec(resolved_base_model_type)
            id_lora_filename = spec.get("id_lora", "")
            if id_lora_filename:
                # IMPORTANT: the LoRA file MUST live in the lora_dir
                # (typically `loras/ltx2/`) — NOT in `ckpts/`. MMGP's
                # LoRA injection treats files in the lora_dir as
                # transformer-attached LoRAs with proper hook patching,
                # while files in `ckpts/` (the default preload location)
                # appear to be loaded but never get their weights
                # actually applied to inference. User testing on
                # 2026-05-26 confirmed: with the file in `ckpts/` we
                # got gibberish audio output even though every other
                # diagnostic (file size, multiplier, voice ref encoded,
                # ref tokens prepended) looked correct.
                #
                # Match the path resolution wgp.py uses for activated
                # LoRAs: get_lora_dir(model_type). We resolve it via
                # sys.modules to avoid a circular import on wgp.
                import sys
                wgp_module = sys.modules.get('wgp') or sys.modules.get('app.wgp')
                if wgp_module is not None and hasattr(wgp_module, 'get_lora_dir'):
                    lora_dir = wgp_module.get_lora_dir(model_type)
                else:
                    # Fallback: handler always returns <loras_root>/ltx2
                    lora_dir = os.path.join('loras', 'ltx2')
                target_path = os.path.join(lora_dir, id_lora_filename)

                id_lora_path = None
                if os.path.isfile(target_path):
                    id_lora_path = target_path
                else:
                    # Self-heal from older installs that downloaded
                    # the LoRA to ckpts/ via the preload_URLs path.
                    # Move it into lora_dir so MMGP picks it up
                    # properly.
                    legacy_path = fl.locate_file(id_lora_filename, error_if_none=False)
                    if legacy_path and os.path.isfile(legacy_path) and (
                        os.path.abspath(legacy_path) != os.path.abspath(target_path)
                    ):
                        try:
                            os.makedirs(lora_dir, exist_ok=True)
                            import shutil
                            shutil.move(legacy_path, target_path)
                            print(f"[LTX2] Migrated ID-LoRA: {legacy_path} → {target_path} "
                                  f"(LoRA must be in lora_dir for MMGP to apply weights)")
                            id_lora_path = target_path
                        except Exception as e:
                            print(f"[LTX2] Could not migrate ID-LoRA to lora_dir: {e} — falling back to legacy path")
                            id_lora_path = legacy_path
                    else:
                        # Fresh download into lora_dir
                        try:
                            os.makedirs(lora_dir, exist_ok=True)
                            from huggingface_hub import hf_hub_download
                            print(f"[LTX2] ID-LoRA not found locally — downloading {id_lora_filename} → {lora_dir}/")
                            id_lora_path = hf_hub_download(
                                repo_id="DeepBeepMeep/LTX-2",
                                filename=id_lora_filename,
                                local_dir=lora_dir,
                            )
                            print(f"[LTX2] Downloaded ID-LoRA: {os.path.basename(id_lora_path)}")
                        except Exception as e:
                            print(f"[LTX2] Failed to download ID-LoRA: {e}")
                            id_lora_path = None
                if id_lora_path and os.path.isfile(id_lora_path):
                    is_distilled = model_def.get("ltx2_pipeline", "two_stage") == "distilled"
                    id_lora_mult = 1.0 if is_distilled else "1;0"
                    loras.append(id_lora_path)
                    loras_mult.append(id_lora_mult)
                    print(f"[LTX2] Auto-loaded ID-LoRA: {id_lora_path} "
                          f"(multiplier={id_lora_mult}, pipeline={'distilled' if is_distilled else 'two_stage'})")
                else:
                    print(f"[LTX2] WARNING: ID-LoRA unavailable — voice-reference will produce noise without it")

        pending_outpaint = str(
            kwargs.get("video_guide_outpainting") or ""
        ).strip()
        dev_outpaint_active = False
        if pending_outpaint and not pending_outpaint.startswith("#"):
            try:
                pending_dims = [
                    float(value)
                    for value in pending_outpaint.split()
                ]
                dev_outpaint_active = (
                    len(pending_dims) == 4
                    and any(value > 0 for value in pending_dims)
                )
            except (TypeError, ValueError):
                dev_outpaint_active = False
        if (
            model_def.get("ltx2_pipeline", "two_stage") != "distilled"
            and not dev_outpaint_active
        ):
            # Keep Dev's existing auto-LoRA behavior, except that Outpaint
            # must load its official system LoRA on both LTX-2.3 pipelines.
            return loras, loras_mult
        preload_urls = get_model_recursive_prop(model_type, "preload_URLs") or []

        # Decide whether outpaint LoRA should load. Trigger: non-empty outpainting
        # dims (either passed explicitly or parsed from the video_guide_outpainting
        # "top bottom left right" percentage string). Replaces the old "O" letter
        # signal so outpainting works through the standard masked-gen pipeline.
        def _outpaint_dims_active():
            dims = kwargs.get("outpainting_dims")
            if _normalize_outpainting_dims(dims) is not None:
                return True
            vgo = str(kwargs.get("video_guide_outpainting") or "").strip()
            if not vgo or vgo.startswith("#"):
                return False
            try:
                parts = [float(v) for v in vgo.split()]
            except (TypeError, ValueError):
                return False
            return len(parts) == 4 and any(p > 0 for p in parts)

        mask_preserving_outpaint = bool(
            kwargs.get("outpaint_mask_preserve", False)
        )
        if resolved_base_model_type == "ltx2_22B":
            if any(letter in video_prompt_type for letter in control_map):
                for file_name in preload_urls:
                    if "union-control" in os.path.basename(file_name):
                        loras.append(fl.locate_file(os.path.basename(file_name)))
                        loras_mult.append(1.0)
                        break
            if _outpaint_dims_active():
                from . import ltx2_handler
                spec = ltx2_handler._get_arch_spec(resolved_base_model_type)
                if mask_preserving_outpaint:
                    outpaint_filename = spec.get("inpaint_ic_lora", "")
                    outpaint_repo = (
                        "Lightricks/"
                        "LTX-2.3-22b-IC-LoRA-In-Outpainting"
                    )
                    outpaint_label = "mask-preserving in/outpainting"
                    outpaint_multiplier = 1.0
                else:
                    outpaint_filename = spec.get("outpaint_ic_lora", "")
                    outpaint_repo = (
                        "oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint"
                    )
                    outpaint_label = "legacy outpaint"
                    try:
                        outpaint_multiplier = float(
                            kwargs.get("outpaint_lora_strength", 1.0)
                            or 1.0
                        )
                    except (TypeError, ValueError):
                        outpaint_multiplier = 1.0
                    outpaint_multiplier = max(
                        0.0,
                        min(2.0, outpaint_multiplier),
                    )
                if outpaint_filename:
                    outpaint_path = fl.locate_file(outpaint_filename, error_if_none=False)
                    if not outpaint_path:
                        dest_dir = fl.get_download_location()
                        print(
                            f"[LTX2] {outpaint_label.title()} IC-LoRA "
                            f"not found locally — downloading "
                            f"{outpaint_filename}..."
                        )
                        from huggingface_hub import hf_hub_download

                        download_repos = [outpaint_repo]
                        if mask_preserving_outpaint:
                            # Wan2GP mirrors the official file here. This
                            # keeps first-use installation resilient if the
                            # upstream model page requires a fresh consent.
                            download_repos.append("DeepBeepMeep/LTX-2")
                        for download_repo in download_repos:
                            try:
                                downloaded = hf_hub_download(
                                    repo_id=download_repo,
                                    filename=outpaint_filename,
                                    local_dir=dest_dir,
                                )
                                outpaint_path = downloaded
                                print(
                                    f"[LTX2] Downloaded "
                                    f"{outpaint_label} IC-LoRA: "
                                    f"{os.path.basename(downloaded)}"
                                )
                                break
                            except Exception as error:
                                print(
                                    f"[LTX2] Could not download "
                                    f"{outpaint_label} IC-LoRA from "
                                    f"{download_repo}: {error}"
                                )
                    if outpaint_path and os.path.isfile(outpaint_path):
                        _op_mult = (
                            # Lightricks' current two-stage graph keeps the
                            # In/Outpaint IC-LoRA on the transformer while
                            # refining the decoded-pixel handoff. Guide tokens
                            # are cropped before pass two; the adapter remains.
                            "1;1"
                            if official_outpaint_stack
                            else outpaint_multiplier
                        )
                        loras.append(outpaint_path)
                        loras_mult.append(_op_mult)
                        print(
                            f"[LTX2] Loaded {outpaint_label} IC-LoRA: "
                            f"{outpaint_path} (strength={_op_mult})"
                        )
                    else:
                        message = (
                            f"{outpaint_label.title()} IC-LoRA "
                            f"{outpaint_filename} is unavailable."
                        )
                        if official_outpaint_stack:
                            raise FileNotFoundError(message)
                        print(
                            f"[LTX2] WARNING: {message} "
                            "Generation will proceed without it."
                        )
        else:
            for letter, signature in control_map.items():
                if letter in video_prompt_type:
                    for file_name in preload_urls:
                        if signature in file_name:
                            loras.append(fl.locate_file(os.path.basename(file_name)))
                            loras_mult.append(1.0)
                            break
        return loras, loras_mult

    def generate(
        self,
        input_prompt: str,
        n_prompt: str | None = None,
        image_start=None,
        image_end=None,
        sampling_steps: int = 40,
        guide_scale: float = 4.0,
        alt_guide_scale: float = 1.0,
        input_video=None,
        prefix_frames_count: int = 0,
        # Motion suffix: symmetric counterpart to (input_video + prefix_frames_count).
        # Places the last `suffix_frames_count` frames of the output at positions
        # [frame_num - K .. frame_num - 1] using the same keyframe-conditioning
        # mechanism as the prefix. Used by the blend endpoint to carry motion
        # trajectory from clip B into the end of the generated transition.
        input_video_end=None,
        suffix_frames_count: int = 0,
        input_frames=None,
        input_frames2=None,
        input_ref_images=None,
        input_masks=None,
        input_masks2=None,
        frames_relative_positions_list=None,
        masking_strength: float | None = None,
        input_video_strength: float | None = None,
        return_latent_slice=None,
        video_prompt_type: str = "",
        denoising_strength: float | None = None,
        cfg_star_switch: int = 0,
        apg_switch: int = 0,
        perturbation_switch: int = 0,
        perturbation_layers: list[int] | None = None,
        perturbation_start: float = 0.0,
        perturbation_end: float = 1.0,
        audio_cfg_scale: float | None = None,
        alt_scale: float = 0.0,
        NAG_scale: float = 1.0,
        NAG_tau: float = 3.5,
        NAG_alpha: float = 0.5,
        self_refiner_setting: int = 0,
        self_refiner_plan: str = "",
        self_refiner_f_uncertainty: float = 0.1,
        self_refiner_certain_percentage: float = 0.999,
        stage2_steps: int = 0,
        loras_slists=None,
        loras_selected=None,
        text_connectors=None,
        input_waveform=None,
        input_waveform_sample_rate=None,
        audio_scale: float | None = None,
        injection_strength: float | None = None,
        masking_source: dict | None = None,
        outpainting_dims: list[float] | None = None,
        outpaint_mask_preserve: bool = False,
        outpaint_official_stack: bool = False,
        frame_num: int = 121,
        height: int = 1024,
        width: int = 1536,
        fps: float = 24.0,
        seed: int = 0,
        stg_scale: float = 1.0,
        cfg_rescale: float = 0.0,
        modality_scale: float = 1.0,
        use_gradient_estimation: bool = False,
        ge_gamma: float = 2.0,
        keyframe_conditioning_mode: str = "replace",
        keyframe_inject_mode: str = "additive",
        retake_video: str | None = None,
        retake_start_frame: int = 0,
        retake_end_frame: int = -1,
        retake_strength: float = 1.0,
        retake_masks_path: str | None = None,
        retake_engine: str = "native",
        regenerate_audio: bool = True,
        reference_pipeline: bool = False,
        progressive_pipeline: bool = False,
        progressive_stage2_steps: int = 5,
        progressive_stage3_steps: int = 3,
        progressive_stage2_sigma: float = 0.85,
        progressive_stage3_sigma: float = 0.85,
        progressive_stage1_image_weight: float = 0.7,
        progressive_stage3_image_weight: float = 0.7,
        sample_solver: str = "euler",
        stg_schedule: list[float] | None = None,
        text_attention_amplifier: dict | None = None,
        callback=None,
        VAE_tile_size=None,
        **kwargs,
    ):
        if self._interrupt:
            return None

        image_start = _coerce_image_list(image_start)
        image_end = _coerce_image_list(image_end)
        if input_ref_images is None:
            input_ref_images = []
        elif isinstance(input_ref_images, (list, tuple)):
            input_ref_images = list(input_ref_images)
        else:
            input_ref_images = [input_ref_images]
        if frames_relative_positions_list is None:
            frames_relative_positions_list = []
        elif isinstance(frames_relative_positions_list, (list, tuple)):
            frames_relative_positions_list = list(frames_relative_positions_list)
        else:
            frames_relative_positions_list = [frames_relative_positions_list]

        prefix_frames_count = int(prefix_frames_count or 0)
        video_prompt_type = video_prompt_type or ""
        outpainting_dims = _normalize_outpainting_dims(outpainting_dims)
        mask_preserving_outpaint = bool(
            outpaint_mask_preserve
            and self.base_model_type == "ltx2_22B"
            and outpainting_dims is not None
            and "V" in video_prompt_type
        )
        single_stage_masked_outpaint = bool(
            mask_preserving_outpaint
            and kwargs.get("single_stage_pipeline", False)
        )
        full_resolution_outpaint_refine = bool(
            mask_preserving_outpaint
            and kwargs.get("outpaint_full_resolution_refine", False)
        )
        official_outpaint_pipeline = bool(
            mask_preserving_outpaint
            and outpaint_official_stack
        )
        if mask_preserving_outpaint:
            input_masks = _merge_ltx2_masks(
                input_masks,
                _build_outpainting_mask_cthw(
                    input_frames,
                    outpainting_dims,
                ),
            )
            input_masks2 = _merge_ltx2_masks(
                input_masks2,
                _build_outpainting_mask_cthw(
                    input_frames2,
                    outpainting_dims,
                ),
            )
            if (
                single_stage_masked_outpaint
                and not full_resolution_outpaint_refine
            ):
                # Give the one-pass fallback clean boundary pixels; the
                # normal Diffusers path uses a neutral reference canvas and
                # a spatial source-attention mask.
                input_frames = _edge_extend_ltx2_masked_control_video(
                    input_frames,
                    input_masks,
                )
                input_frames2 = _edge_extend_ltx2_masked_control_video(
                    input_frames2,
                    input_masks2,
                )
            elif full_resolution_outpaint_refine:
                # Current Lightricks In/Outpainting graph: the IC-LoRA was
                # trained on this exact missing-region marker.
                input_frames = _paint_ltx2_inpaint_control_video(
                    input_frames,
                    input_masks,
                )
                input_frames2 = _paint_ltx2_inpaint_control_video(
                    input_frames2,
                    input_masks2,
                )
            else:
                input_frames = _paint_ltx2_masked_control_video(
                    input_frames,
                    input_masks,
                )
                input_frames2 = _paint_ltx2_masked_control_video(
                    input_frames2,
                    input_masks2,
                )
            masking_strength = 0.0
            print(
                "[LTX2] Using mask-preserving Outpaint conditioning "
                "(protected source + generated canvas)."
            )
            if official_outpaint_pipeline:
                if full_resolution_outpaint_refine:
                    print(
                        "[LTX2] Official Lightricks Outpaint: "
                        "#66FF00 mask guide + full reference attention; "
                        "IC-LoRA active for coarse generation and "
                        "refinement."
                    )
                    print(
                        "[LTX2] Using decoded-pixel Lanczos handoff instead "
                        "of learned latent upscaling."
                    )
                else:
                    print(
                        "[LTX2] Diffusers-reference Outpaint: neutral "
                        "canvas + source-region reference attention; "
                        "stage-one IC-LoRA + learned latent x2 upscale + "
                        "bare distilled refinement."
                    )
            elif single_stage_masked_outpaint:
                print(
                    "[LTX2] Outpaint is running at full target resolution "
                    "in one diffusion stage; stage-2 spatial upscaling is "
                    "disabled."
                )
                print(
                    "[LTX2] Generated-canvas reference tokens are "
                    "masked out (one-pass fallback)."
                )
        self_refiner_max_plans = int(self.model_def.get("self_refiner_max_plans", 1))
        # Per upstream WanGP commit 5da7f23 ("unlocked ltx2 dev"), the gamma
        # roundtrip preprocessing (used by the IC-LoRA Outpaint pipeline) is
        # no longer gated on the distilled pipeline. It applies whenever the
        # 22B family is asked to outpaint via the "G" video_prompt_type letter.
        requested_outpaint_gamma_roundtrip = bool(
            self.base_model_type == "ltx2_22B"
            and outpainting_dims is not None
            and "G" in video_prompt_type
            and not mask_preserving_outpaint
        )
        use_outpaint_gamma_roundtrip = False

        def _get_frame_dim(video_tensor: torch.Tensor) -> int | None:
            if video_tensor.dim() < 2:
                return None
            if video_tensor.dim() == 5:
                if video_tensor.shape[1] in (1, 3, 4):
                    return 2
                if video_tensor.shape[-1] in (1, 3, 4):
                    return 1
            if video_tensor.shape[0] in (1, 3, 4):
                return 1
            if video_tensor.shape[-1] in (1, 3, 4):
                return 0
            return 0

        def _frame_count(video_value) -> int | None:
            if not torch.is_tensor(video_value):
                return None
            frame_dim = _get_frame_dim(video_value)
            if frame_dim is None:
                return None
            return int(video_value.shape[frame_dim])

        def _slice_frames(video_value: torch.Tensor, start: int, end: int) -> torch.Tensor:
            frame_dim = _get_frame_dim(video_value)
            if frame_dim == 1:
                return video_value[:, start:end]
            if frame_dim == 2:
                return video_value[:, :, start:end]
            return video_value[start:end]

        def _maybe_trim_control(video_value, target_frames: int):
            if not torch.is_tensor(video_value) or target_frames <= 0:
                return video_value, None
            current_frames = _frame_count(video_value)
            if current_frames is None:
                return video_value, None
            if current_frames > target_frames:
                video_value = _slice_frames(video_value, 0, target_frames)
                current_frames = target_frames
            return video_value, current_frames

        try:
            masking_strength = float(masking_strength) if masking_strength is not None else 0.0
        except (TypeError, ValueError):
            masking_strength = 0.0
        try:
            input_video_strength = float(input_video_strength) if input_video_strength is not None else 1.0
        except (TypeError, ValueError):
            input_video_strength = 1.0
        input_video_strength = max(0.0, min(1.0, input_video_strength))
        # Injection strength: separate control for injected reference frames
        _injection_strength = float(injection_strength) if injection_strength is not None else input_video_strength
        _injection_strength = max(0.0, min(1.0, _injection_strength))
        if requested_outpaint_gamma_roundtrip:
            conditioning_gamma_applied = _apply_gamma_to_media(image_start, LTX2_OUTPAINT_GAMMA)
            conditioning_gamma_applied = _apply_gamma_to_media(image_end, LTX2_OUTPAINT_GAMMA) or conditioning_gamma_applied
            if torch.is_tensor(input_video) and prefix_frames_count > 0:
                conditioning_gamma_applied = _apply_gamma_to_media(input_video[:, :prefix_frames_count], LTX2_OUTPAINT_GAMMA) or conditioning_gamma_applied
            if input_ref_images is not None:
                for ref_image in input_ref_images:
                    conditioning_gamma_applied = _apply_gamma_to_media(ref_image, LTX2_OUTPAINT_GAMMA) or conditioning_gamma_applied
            if conditioning_gamma_applied:
                print("[LTX2] Applying full-frame gamma preprocessing for outpainting IC-LoRA conditioning images.")
                use_outpaint_gamma_roundtrip = True
        if "G" not in video_prompt_type:
            denoising_strength = 1.0
            masking_strength = 0.0
        # Per upstream WanGP commit 5da7f23 ("unlocked ltx2 dev"), IC-LoRA
        # downscale-factor inference applies to all pipelines (was previously
        # gated on distilled only). If a Dev user activates an IC-LoRA — e.g.
        # the union-control or outpaint LoRA — the downscale factor must be
        # honored so the IC-LoRA receives correctly-sized control conditioning.
        ic_lora_downscale_factor = _infer_ic_lora_downscale_factor(loras_selected)
        video_conditioning_downscale_factor = ic_lora_downscale_factor or 1

        video_conditioning = None
        masking_source = None
        if input_frames is not None or input_frames2 is not None:
            control_start_frame = int(prefix_frames_count)
            expected_guide_frames = max(1, int(frame_num) - control_start_frame + (1 if prefix_frames_count > 1 else 0))
            if prefix_frames_count > 1:
                control_start_frame = -control_start_frame
            input_frames, frames_len = _maybe_trim_control(input_frames, expected_guide_frames)
            input_frames2, frames_len2 = _maybe_trim_control(input_frames2, expected_guide_frames)
            input_masks, _ = _maybe_trim_control(input_masks, expected_guide_frames)
            input_masks2, _ = _maybe_trim_control(input_masks2, expected_guide_frames)
            if mask_preserving_outpaint:
                input_frames, input_masks = (
                    _pad_ltx2_masked_control_video_tail(
                        input_frames,
                        input_masks,
                        expected_guide_frames,
                        repeat_last_frame=(
                            single_stage_masked_outpaint
                            or official_outpaint_pipeline
                        ),
                        pad_rgb=None,
                    )
                )
                input_frames2, input_masks2 = (
                    _pad_ltx2_masked_control_video_tail(
                        input_frames2,
                        input_masks2,
                        expected_guide_frames,
                        repeat_last_frame=(
                            single_stage_masked_outpaint
                            or official_outpaint_pipeline
                        ),
                        pad_rgb=None,
                    )
                )
            if requested_outpaint_gamma_roundtrip:
                control_tensor = input_frames if input_frames is not None else input_frames2
                control_rect = None if control_tensor is None else _get_outpainting_inner_rect(control_tensor.shape[-2], control_tensor.shape[-1], outpainting_dims)
                if control_rect is not None and _apply_gamma_to_video_rect(control_tensor, control_rect, LTX2_OUTPAINT_GAMMA):
                    print("[LTX2] Applying preserved-area gamma preprocessing for outpainting IC-LoRA control video.")
                    use_outpaint_gamma_roundtrip = True

            control_strength = 1.0
            if (
                not mask_preserving_outpaint
                and denoising_strength is not None
                and "G" in video_prompt_type
            ):
                try:
                    control_strength = float(denoising_strength)
                except (TypeError, ValueError):
                    control_strength = 1.0
            control_strength = max(0.0, min(1.0, control_strength))

            conditioning_entries = []
            if input_frames is not None:
                conditioning_entries.append((input_frames, control_start_frame, control_strength))
            if input_frames2 is not None:
                conditioning_entries.append((input_frames2, control_start_frame, control_strength))
            if conditioning_entries:
                video_conditioning = conditioning_entries
            if masking_strength > 0.0:
                if input_masks is not None and input_frames is not None:
                    masking_source = {
                        "video": input_frames,
                        "mask": input_masks,
                        "start_frame": control_start_frame,
                    }
                elif input_masks2 is not None and input_frames2 is not None:
                    masking_source = {
                        "video": input_frames2,
                        "mask": input_masks2,
                        "start_frame": control_start_frame,
                    }

        interrupt_check = lambda: self._interrupt

        # ── Native Retake Pipeline ───────────────────────────────────────
        # Uses Lightricks' denoise_mask approach for proper retake quality.
        # Early return — bypasses the rest of generate() and uses RetakePipeline directly.
        if retake_video and os.path.isfile(retake_video) and retake_engine == "native":
            import decord
            from .ltx_pipelines.retake import RetakePipeline

            retake_models = _resolve_retake_pipeline_models(self.pipeline)
            vr = decord.VideoReader(retake_video)
            total_frames = len(vr)
            retake_fps = vr.get_avg_fps()
            src_h, src_w = vr[0].shape[:2]

            # Determine target resolution while preserving source aspect ratio
            user_h, user_w = int(height), int(width)
            if user_h > 0 and user_w > 0 and (user_h != src_h or user_w != src_w):
                # User chose a resolution preset (e.g. 720p) — scale to fit while
                # preserving the source aspect ratio, then align to 32
                max_dim = max(user_h, user_w)  # e.g. 1280 for 720p
                src_max = max(src_h, src_w)
                if src_max > max_dim:
                    # Downscale: fit the longer side to max_dim
                    scale = max_dim / src_max
                    aligned_h = (int(src_h * scale) // 32) * 32
                    aligned_w = (int(src_w * scale) // 32) * 32
                else:
                    # Source is already smaller than target — just align to 32
                    aligned_h = (src_h // 32) * 32
                    aligned_w = (src_w // 32) * 32
            else:
                # Auto: use source resolution aligned to 32
                aligned_h = (src_h // 32) * 32
                aligned_w = (src_w // 32) * 32
            needs_rescale = (aligned_h != src_h) or (aligned_w != src_w)

            # Re-encode FPS to 25 if source is significantly different (LTX optimized for 25fps)
            target_fps = retake_fps
            needs_fps_change = False
            if abs(retake_fps - 25.0) > 2.0:
                target_fps = 25.0
                needs_fps_change = True

            end_f = retake_end_frame if retake_end_frame > 0 else total_frames
            end_f = min(end_f, total_frames)
            start_f = max(0, retake_start_frame)
            clip_frames = end_f - start_f

            import tempfile
            import subprocess
            _retake_temp_dir = tempfile.mkdtemp(prefix="retake_")

            # Step 1: Pre-process entire source video (scale + fps) if needed
            if needs_rescale or needs_fps_change:
                scaled_source = os.path.join(_retake_temp_dir, "source_aligned.mp4")
                vf_filters = []
                if needs_rescale:
                    vf_filters.append(f"scale={aligned_w}:{aligned_h}:flags=lanczos")
                if needs_fps_change:
                    vf_filters.append(f"fps={target_fps}")
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", retake_video,
                    "-vf", ",".join(vf_filters),
                    "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                    "-an", scaled_source
                ]
                subprocess.run(ffmpeg_cmd, capture_output=True, timeout=300)
                source_for_stitch = scaled_source
                # Re-read frame count after fps conversion
                vr_aligned = decord.VideoReader(scaled_source)
                total_frames = len(vr_aligned)
                del vr_aligned
                # Recompute retake region for new frame count
                end_f = retake_end_frame if retake_end_frame > 0 else total_frames
                end_f = min(end_f, total_frames)
                start_f = max(0, retake_start_frame)
                clip_frames = end_f - start_f
                retake_fps = target_fps
                print(f"[Retake Native] Pre-processed source: {src_w}x{src_h} → {aligned_w}x{aligned_h}@{target_fps}fps ({total_frames} frames)")
            else:
                source_for_stitch = retake_video

            # Step 2: Extract retake clip from the (possibly rescaled) source
            retake_clip_path = os.path.join(_retake_temp_dir, "clip.mp4")
            import av
            vr_source = decord.VideoReader(source_for_stitch)
            out_container = av.open(retake_clip_path, mode='w')
            stream = out_container.add_stream('h264', rate=int(retake_fps))
            stream.width = aligned_w
            stream.height = aligned_h
            stream.pix_fmt = 'yuv420p'
            for fi in range(start_f, end_f):
                frame_data = _decord_frame_to_numpy(vr_source[fi])
                av_frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
                for packet in stream.encode(av_frame):
                    out_container.mux(packet)
            for packet in stream.encode():
                out_container.mux(packet)
            out_container.close()

            # Boundary-frame I2V anchors. These pin the first and last frames
            # of the retake clip to the original source content, which keeps
            # the regenerated segment seamlessly continuous with the unchanged
            # frames before/after it (important for TEMPORAL retake).
            #
            # HOWEVER, anchors are anti-features in two cases:
            #
            # (1) SAM-masked SPATIAL inpaint — both anchors pinned to the
            #     source (woman in dress) overwhelm the spatial mask + prompt
            #     telling the model "change the dress to armor." The masked
            #     subject can't be replaced. Skip ALL boundary anchors.
            #
            # (2) END-OF-CLIP retake — when the retake region extends to the
            #     natural end of the source video, there's nothing AFTER it
            #     to blend back into, so pinning the end frame defeats the
            #     entire purpose of the retake. User reports: "man and woman
            #     sitting in original clip; retake the end with prompt 'woman
            #     stands up' produces no change because the end frame is
            #     locked to her sitting." Skip the END anchor only — keep
            #     the START anchor for visual continuity with the preserved
            #     frames before the retake region.
            #
            # The symmetric "start-of-clip" case (start_f == 0) is NOT
            # carved out: even for a beginning-of-clip retake the start
            # anchor is usually what the user wants (it provides the I2V
            # starting visual context — without it the model would generate
            # the opening from noise + prompt only, producing wildly different
            # content from what they uploaded as the source).
            _is_spatial_inpaint = bool(retake_masks_path and os.path.isfile(retake_masks_path))
            # 2-frame tolerance handles slider-precision off-by-one from the UI.
            _is_end_of_clip = end_f >= total_frames - 2
            if _is_spatial_inpaint:
                retake_images = []
                print(f"[Retake Native] Spatial mask active — skipping boundary I2V anchors "
                      f"(they would prevent the masked subject from being replaced)")
            else:
                from PIL import Image as _PILImage
                start_frame_path = os.path.join(_retake_temp_dir, "start.png")
                _PILImage.fromarray(_decord_frame_to_numpy(vr_source[start_f])).save(start_frame_path)
                retake_images = [(start_frame_path, 0, input_video_strength, "lanczos")]
                if _is_end_of_clip:
                    print(f"[Retake Native] Retake reaches end-of-clip (end_f={end_f}, "
                          f"total={total_frames}) — skipping END anchor so the prompt can "
                          f"drive composition change in the final frames")
                else:
                    end_frame_path = os.path.join(_retake_temp_dir, "end.png")
                    _PILImage.fromarray(_decord_frame_to_numpy(vr_source[min(end_f - 1, total_frames - 1)])).save(end_frame_path)
                    retake_images.append((end_frame_path, clip_frames - 1, input_video_strength))
            del vr, vr_source

            # Extract audio clip from original source for audio conditioning
            audio_clip_path = None
            start_sec = start_f / retake_fps
            end_sec = end_f / retake_fps
            try:
                import subprocess as _sp
                audio_clip_path = os.path.join(_retake_temp_dir, "audio_clip.wav")
                _sp.run([
                    "ffmpeg", "-y", "-i", retake_video,
                    "-ss", str(start_sec), "-to", str(end_sec),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                    audio_clip_path
                ], capture_output=True, timeout=30)
                if not os.path.isfile(audio_clip_path) or os.path.getsize(audio_clip_path) < 100:
                    audio_clip_path = None
                else:
                    print(f"[Retake Native] Extracted audio clip: {start_sec:.1f}s-{end_sec:.1f}s")
            except Exception:
                audio_clip_path = None

            tiling_config = _build_tiling_config(VAE_tile_size, retake_fps)

            print(f"[Retake Native] Clip: frames {start_f}-{end_f} ({clip_frames} frames), "
                  f"res={aligned_w}x{aligned_h}, fps={retake_fps:.1f}, audio={'yes' if audio_clip_path else 'no'}")

            retake_pipeline = RetakePipeline(
                models=retake_models,
                device=self.device,
                dtype=torch.bfloat16,
            )

            video_chunks, audio_tensor = retake_pipeline(
                source_video_path=retake_clip_path,
                prompt=input_prompt,
                seed=int(seed),
                height=aligned_h,
                width=aligned_w,
                num_frames=clip_frames,
                frame_rate=float(retake_fps),
                start_frame=0,
                end_frame=clip_frames,
                tiling_config=tiling_config,
                callback=callback,
                interrupt_check=interrupt_check,
                text_connectors=text_connectors or getattr(self, "_text_connectors", None),
                images=retake_images,
                loras_slists=loras_slists,
                regenerate_audio=regenerate_audio,
                source_audio_path=audio_clip_path,
                spatial_mask_path=retake_masks_path,
                # CFG: turns on classifier-free guidance inside the retake
                # pipeline so the prompt actually pushes the masked region
                # toward it. Required for prompt-driven inpainting.
                negative_prompt=(n_prompt or ""),
                cfg_guidance_scale=float(guide_scale),
            )

            # Collect video tensor
            video_tensor = _collect_video_chunks(
                video_chunks,
                interrupt_check=interrupt_check,
                expected_frames=clip_frames,
                expected_height=aligned_h,
                expected_width=aligned_w,
            )
            if video_tensor is None:
                return None

            audio_np = None
            if audio_tensor is not None and audio_tensor.numel() > 1:
                audio_np = audio_tensor.detach().float().cpu().numpy()
                if audio_np.ndim == 2 and audio_np.shape[0] in (1, 2) and audio_np.shape[1] > audio_np.shape[0]:
                    audio_np = audio_np.T  # [channels, samples] → [samples, channels]
            result = {
                "x": video_tensor,
                "audio": audio_np,
                "audio_sampling_rate": int(getattr(self.vocoder, "output_sampling_rate", 44100)) if hasattr(self, 'vocoder') else 44100,
                "retake_stitch_info": {
                    "source_video": source_for_stitch,  # use pre-scaled source for stitching
                    "original_video": retake_video,  # original source (may have audio)
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "total_frames": total_frames,
                    "fps": float(retake_fps),
                    "temp_dir": _retake_temp_dir,
                    "src_w": aligned_w,
                    "src_h": aligned_h,
                    "regenerate_audio": regenerate_audio,
                },
            }

            # Cleanup temp clip (stitch info temp_dir cleaned after stitching in wgp.py)
            return result

        # ── Legacy Retake (MaskInjection approach) ───────────────────────
        # Retake mode: extract retake clip, process only the retake region
        # After generation, the caller stitches: original[0:start] + retake + original[end:]
        _retake_stitch_info = None
        if retake_video and os.path.isfile(retake_video):
            import decord
            try:
                vr = decord.VideoReader(retake_video)
                total_frames = len(vr)
                retake_fps = vr.get_avg_fps()
                src_h, src_w = vr[0].shape[:2]
                end_frame = retake_end_frame if retake_end_frame > 0 else total_frames
                end_frame = min(end_frame, total_frames)
                start_f = max(0, retake_start_frame)
                clip_frames = end_frame - start_f

                # Extract first and last frames of the RETAKE REGION as image conditioning
                frame_first = _decord_frame_to_numpy(vr[start_f])
                frame_last = _decord_frame_to_numpy(vr[min(end_frame - 1, total_frames - 1)])

                # Extract the retake clip to a temp video file
                import tempfile
                _retake_temp_dir = tempfile.mkdtemp(prefix="retake_")
                start_frame_path = os.path.join(_retake_temp_dir, "start.png")
                end_frame_path = os.path.join(_retake_temp_dir, "end.png")
                retake_clip_path = os.path.join(_retake_temp_dir, "clip.mp4")

                from PIL import Image as _PILImage
                _PILImage.fromarray(frame_first).save(start_frame_path)
                _PILImage.fromarray(frame_last).save(end_frame_path)

                # Write retake clip
                import av
                out_container = av.open(retake_clip_path, mode='w')
                stream = out_container.add_stream('h264', rate=int(retake_fps))
                stream.width = src_w
                stream.height = src_h
                stream.pix_fmt = 'yuv420p'
                for fi in range(start_f, end_frame):
                    frame_data = _decord_frame_to_numpy(vr[fi])
                    av_frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
                    for packet in stream.encode(av_frame):
                        out_container.mux(packet)
                for packet in stream.encode():
                    out_container.mux(packet)
                out_container.close()
                del vr

                # Set image conditioning from retake region boundary frames
                if image_start is None:
                    image_start = start_frame_path
                if image_end is None:
                    image_end = end_frame_path
                keyframe_conditioning_mode = "additive"

                # Build mask for the retake clip (entire clip = regenerate)
                if retake_masks_path and os.path.isfile(retake_masks_path):
                    import numpy as np
                    sam_masks = np.load(retake_masks_path)  # [T, H, W] bool
                    # Slice to retake region
                    if sam_masks.shape[0] > clip_frames:
                        sam_masks = sam_masks[start_f:end_frame]
                    mask_t = torch.from_numpy(sam_masks.astype(np.float32))
                    mask = mask_t.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
                    print(f"[Inpaint] SAM spatial mask for clip: {sam_masks.shape} "
                          f"({sam_masks.sum() / sam_masks.size * 100:.1f}% masked)")
                else:
                    # Full clip regeneration — mask = all 1s
                    mask = torch.ones(1, 1, clip_frames, 1, 1)

                masking_source = {
                    "video": retake_clip_path,
                    "mask": mask,
                    "start_frame": 0,
                }
                masking_strength = max(0.01, float(retake_strength))

                # Only process the retake region frames
                frame_num = clip_frames
                fps = float(retake_fps) if retake_fps > 0 else fps
                target_height = src_h
                target_width = src_w

                # Save stitch info for post-processing
                _retake_stitch_info = {
                    "source_video": retake_video,
                    "start_frame": start_f,
                    "end_frame": end_frame,
                    "total_frames": total_frames,
                    "fps": fps,
                    "temp_dir": _retake_temp_dir,
                    "src_w": src_w,
                    "src_h": src_h,
                }

                print(f"[Retake] Clip: frames {start_f}-{end_frame} ({clip_frames} frames) from {total_frames}, "
                      f"strength={masking_strength:.2f}, fps={fps:.1f}, res={src_w}x{src_h}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Retake] Failed to set up retake: {e}")

        latent_conditioning_stage2 = None

        latent_stride = 8
        if hasattr(self.pipeline, "pipeline_components"):
            scale_factors = getattr(self.pipeline.pipeline_components, "video_scale_factors", None)
            if scale_factors is not None:
                latent_stride = int(getattr(scale_factors, "time", scale_factors[0]))

        images = []          # start frames + prefix — controlled by keyframe_conditioning_mode
        end_images = []       # end frames — controlled by keyframe_conditioning_mode
        end_images_stage2 = []
        inject_images = []    # injected keyframes — controlled by keyframe_inject_mode
        inject_images_stage2 = []
        images_stage2 = []
        stage2_override = False
        has_prefix_frames = input_video is not None and torch.is_tensor(input_video) and prefix_frames_count > 0
        has_suffix_frames = input_video_end is not None and torch.is_tensor(input_video_end) and suffix_frames_count > 0
        is_start_image_only = image_start is not None and (not has_prefix_frames or prefix_frames_count <= 1)
        use_guiding_latent_for_start_image = bool(self.model_def.get("use_guiding_latent_for_start_image", False))
        use_guiding_start_image = use_guiding_latent_for_start_image and is_start_image_only

        def _append_prefix_entries(target_list, extra_list=None):
            if not has_prefix_frames or is_start_image_only:
                return
            frame_count = min(prefix_frames_count, input_video.shape[1])
            if frame_count <= 0:
                return
            # Preserve the overlap as one temporal conditioning sequence.
            # Sampling isolated stills here keeps appearance but discards the
            # direction and speed of motion at a sliding-window boundary.  The
            # native LTX pipeline accepts an F,H,W,C tensor and VAE-encodes the
            # complete prefix, allowing the next window to continue its motion.
            entry = (
                input_video[:, :frame_count].permute(1, 2, 3, 0),
                0,
                input_video_strength,
            )
            if frame_count > 1:
                print(
                    f"[LTX2] Continuing with {frame_count} motion-history "
                    "frames as one temporal prefix."
                )
            target_list.append(entry)
            if extra_list is not None:
                extra_list.append(entry)

        def _append_suffix_entries(target_list, extra_list=None):
            """Mirror of _append_prefix_entries but at the END of the output.
            Places suffix frames at output positions [frame_num - K .. frame_num - 1]
            using the same VideoConditionByKeyframeIndex mechanism as the prefix,
            except with a temporal offset. Because LTX-2's attention is global
            (bidirectional), a suffix is architecturally identical to a prefix
            except for where its tokens sit in the sequence.
            """
            if not has_suffix_frames:
                return
            frame_count = min(suffix_frames_count, input_video_end.shape[1])
            if frame_count <= 0:
                return
            # Pick source-video frames at latent_stride intervals
            src_indices = list(range(0, frame_count, latent_stride))
            last_src = frame_count - 1
            if src_indices[-1] != last_src:
                # Ensure the final suffix frame (= the "landing" anchor) dominates its latent slot.
                src_indices.append(last_src)
            # Map each source index to its output position. The final source frame
            # lands at output index (frame_num - 1).
            output_start = int(frame_num) - frame_count
            for src_idx in src_indices:
                out_pos = output_start + src_idx
                if out_pos < 0:
                    continue
                entry = (input_video_end[:, src_idx], _to_latent_index(out_pos, latent_stride), input_video_strength)
                target_list.append(entry)
                if extra_list is not None:
                    extra_list.append(entry)

        def _append_injected_ref_entries(target_list, extra_list=None):
            injected_ref_count = min(len(input_ref_images), len(frames_relative_positions_list))
            for ref_image, frame_idx in zip(input_ref_images[:injected_ref_count], frames_relative_positions_list[:injected_ref_count]):
                # Replace mode needs latent-space index; additive uses pixel-space
                idx = _to_latent_index(int(frame_idx), latent_stride) if keyframe_inject_mode == "replace" else int(frame_idx)
                entry = (ref_image, idx, _injection_strength, "lanczos")
                target_list.append(entry)
                if extra_list is not None:
                    extra_list.append(entry)

        # End frame index: replace mode needs latent-space index, additive needs pixel-space
        _end_frame_idx = _to_latent_index(int(frame_num - 1), latent_stride) if keyframe_conditioning_mode == "replace" else int(frame_num - 1)

        active_official_outpaint = None
        if official_outpaint_pipeline:
            if isinstance(self.pipeline, DistilledPipeline):
                active_official_outpaint = self.pipeline
            elif isinstance(self.pipeline, TI2VidTwoStagesPipeline):
                if not hasattr(self, "_official_outpaint_pipeline"):
                    self._official_outpaint_pipeline = DistilledPipeline(
                        device=self.device,
                        models=self.pipeline.stage_1_models,
                    )
                    self._official_outpaint_pipeline.text_encoder_cache = (
                        self.pipeline.text_encoder_cache
                    )
                    print(
                        "[LTX2] Initialized official Outpaint sampler over "
                        "the loaded Dev transformer."
                    )
                active_official_outpaint = (
                    self._official_outpaint_pipeline
                )
            else:
                raise RuntimeError(
                    "Official LTX-2.3 Outpaint requires a Dev or "
                    "distilled-compatible LTX pipeline."
                )

        if _uses_native_two_stage_dispatch(
            self.pipeline,
            active_official_outpaint,
        ):
            _append_prefix_entries(images, images_stage2)

            if has_suffix_frames:
                # Video_end supersedes image_end: the suffix's final frame is
                # already the landing anchor, with full motion leading into it.
                _append_suffix_entries(end_images, end_images_stage2)
            elif image_end is not None:
                entry = (image_end, _end_frame_idx, input_video_strength)
                end_images.append(entry)
                end_images_stage2.append(entry)

            if image_start is not None:
                entry = (image_start, _to_latent_index(0, latent_stride), input_video_strength, "lanczos")
                if use_guiding_start_image:
                    end_images.append(entry)
                    images_stage2.append(entry)
                    stage2_override = True
                else:
                    images.append(entry)
                    images_stage2.append(entry)
            _append_injected_ref_entries(inject_images, inject_images_stage2)
        else:
            _append_prefix_entries(images)
            if image_start is not None:
                images.append((image_start, _to_latent_index(0, latent_stride), input_video_strength, "lanczos"))
            if has_suffix_frames:
                _append_suffix_entries(end_images, end_images_stage2)
            elif image_end is not None:
                entry = (image_end, _end_frame_idx, input_video_strength)
                end_images.append(entry)
                end_images_stage2.append(entry)
            _append_injected_ref_entries(inject_images, inject_images_stage2)

        tiling_config = _build_tiling_config(VAE_tile_size, fps)
        text_connectors = text_connectors or getattr(self, "_text_connectors", None)

        audio_conditionings = None
        if input_waveform is not None:
            if audio_scale is None:
                audio_scale = 1.0
            # Strength 0-1 controls denoise mask (how much audio conditions the generation)
            # Values >1.0 keep mask at 0 (full conditioning) AND amplify the audio latent
            audio_strength = max(0.0, float(audio_scale))
            audio_latent_boost = max(1.0, audio_strength)  # amplification factor for values >1.0
            audio_strength = min(audio_strength, 1.0)  # clamp the denoise mask part to 0-1
            if audio_strength > 0.0:
                if self._interrupt:
                    return None
                target_channels = int(getattr(self.audio_encoder, "in_channels", 1))
                waveform = _prepare_ltx_audio_waveform(
                    input_waveform,
                    target_channels,
                )
                waveform_sample_rate = input_waveform_sample_rate
                if self._interrupt:
                    return None
                if target_channels <= 0:
                    target_channels = waveform.shape[1]
                if waveform.shape[1] != target_channels:
                    if waveform.shape[1] == 1 and target_channels > 1:
                        waveform = waveform.repeat(1, target_channels, 1)
                    elif target_channels == 1:
                        waveform = waveform.mean(dim=1, keepdim=True)
                    else:
                        waveform = waveform[:, :target_channels, :]
                        if waveform.shape[1] < target_channels:
                            pad_channels = target_channels - waveform.shape[1]
                            pad = torch.zeros(
                                (waveform.shape[0], pad_channels, waveform.shape[2]),
                                dtype=waveform.dtype,
                            )
                            waveform = torch.cat([waveform, pad], dim=1)

                audio_processor = AudioProcessor(
                    sample_rate=self.audio_encoder.sample_rate,
                    mel_bins=self.audio_encoder.mel_bins,
                    mel_hop_length=self.audio_encoder.mel_hop_length,
                    n_fft=self.audio_encoder.n_fft,
                )
                waveform = waveform.to(device="cpu", dtype=torch.float32)
                audio_processor = audio_processor.to(waveform.device)
                mel = audio_processor.waveform_to_mel(waveform, waveform_sample_rate)
                if self._interrupt:
                    return None
                audio_params = next(self.audio_encoder.parameters(), None)
                audio_device = audio_params.device if audio_params is not None else self.device
                audio_dtype = audio_params.dtype if audio_params is not None else self.dtype
                mel = mel.to(device=audio_device, dtype=audio_dtype)
                with torch.inference_mode():
                    audio_latent = self.audio_encoder(mel)
                if self._interrupt:
                    return None
                audio_downsample = getattr(
                    getattr(self.audio_encoder, "patchifier", None),
                    "audio_latent_downsample_factor",
                    4,
                )
                target_shape = AudioLatentShape.from_video_pixel_shape(
                    VideoPixelShape(
                        batch=audio_latent.shape[0],
                        frames=int(frame_num),
                        width=1,
                        height=1,
                        fps=float(fps),
                    ),
                    channels=audio_latent.shape[1],
                    mel_bins=audio_latent.shape[3],
                    sample_rate=self.audio_encoder.sample_rate,
                    hop_length=self.audio_encoder.mel_hop_length,
                    audio_latent_downsample_factor=audio_downsample,
                )
                target_frames = target_shape.frames
                audio_latent = audio_latent.to(device=self.device, dtype=self.dtype)
                # For audio_scale >1.0: amplify the latent signal to boost audio influence
                if audio_latent_boost > 1.0:
                    audio_latent = audio_latent * audio_latent_boost
                if audio_latent.shape[2] < target_frames:
                    # Sliding-window audio continuity: the provided latent is a prefix
                    # (previous window's trailing audio) shorter than the full target.
                    # Freeze it as a clean prefix and let the model generate the rest.
                    audio_conditionings = [AudioConditionByLatentPrefix(audio_latent)]
                else:
                    if audio_latent.shape[2] > target_frames:
                        audio_latent = audio_latent[:, :, :target_frames, :]
                    audio_conditionings = [AudioConditionByLatent(audio_latent, audio_strength)]

        # ID-LoRA: encode voice reference audio for identity preservation
        voice_ref_waveform = kwargs.get("voice_reference_waveform")
        voice_ref_sr = kwargs.get("voice_reference_sample_rate", 16000)
        identity_guidance_scale = kwargs.get("identity_guidance_scale", 0.0)
        # ID-LoRA voice-clone runs use the EXACT optimum.quanto int8 path (no
        # Maestro Triton-kernel injection), staying symbol-identical to upstream
        # Wan2GP — the configuration the CelebVHQ ID-LoRA was validated against.
        #
        # NOTE: the actual root cause of the earlier gibberish was an
        # un-normalized int16 voice reference (±32768 instead of [-1,1]) — fixed
        # in wgp.py's PyAV loader. This quanto revert is a belt-and-suspenders
        # measure: the identity-guidance extrapolation (out = pos + 3*(pos-id))
        # amplifies any per-layer numerical error, so we prefer the exact int8
        # forward during voice clones. Mirrors the HiDream / Scenema workaround
        # (hidream_handler.py, scenema_audio.py). The revert is process-global,
        # so the `else` branch re-enables the Triton kernels on the next NORMAL
        # run to keep regular video generation at full speed.
        if voice_ref_waveform is not None and float(identity_guidance_scale or 0.0) > 0.0:
            try:
                from shared.kernels.quanto_int8_inject import disable_quanto_int8_kernel
                from shared.kernels import quanto_int8_inject as _inj
                _triton_was_on = disable_quanto_int8_kernel(notify_disabled=True)
                if _inj._BASE_PATCH_STATE.enabled and _inj._BASE_PATCH_STATE.orig_forward is not None:
                    from optimum.quanto.tensor.weights import qbytes as _qbytes
                    _qbytes.WeightQBytesLinearFunction.forward = staticmethod(_inj._BASE_PATCH_STATE.orig_forward)
                    _inj._BASE_PATCH_STATE.enabled = False
                    _inj._BASE_PATCH_STATE.orig_forward = None
                print(
                    f"[ID-LoRA] Using exact optimum.quanto int8 path for this voice-clone run "
                    f"(Triton kernel was {'ON' if _triton_was_on else 'off'}) to match Wan2GP."
                )
            except Exception as e:
                print(f"[ID-LoRA] Could not revert quanto int8 patches (non-fatal): {e}")
        else:
            # Normal (non voice-clone) run: if a previous voice-clone run this
            # session disabled the Triton int8 kernels, restore them so regular
            # video keeps full speed. maybe_enable_* respects the user's
            # int8-kernel setting (via WAN2GP_QUANTO_INT8_KERNEL), so this is a
            # no-op when the user has them off or when they are already active.
            try:
                from shared.kernels import quanto_int8_inject as _inj
                if not _inj._PATCH_STATE.enabled:
                    from shared.kernels.quanto_int8_inject import maybe_enable_quanto_int8_kernel
                    maybe_enable_quanto_int8_kernel()
            except Exception:
                pass
        if voice_ref_waveform is not None and hasattr(self, 'audio_encoder') and self.audio_encoder is not None:
            try:
                vr_waveform = torch.from_numpy(voice_ref_waveform) if not torch.is_tensor(voice_ref_waveform) else voice_ref_waveform
                if vr_waveform.ndim == 1:
                    vr_waveform = vr_waveform.unsqueeze(0).unsqueeze(0)
                elif vr_waveform.ndim == 2:
                    vr_waveform = vr_waveform.unsqueeze(0)
                target_ch = int(getattr(self.audio_encoder, "in_channels", vr_waveform.shape[1]))
                if target_ch > 0 and vr_waveform.shape[1] != target_ch:
                    if vr_waveform.shape[1] == 1 and target_ch > 1:
                        vr_waveform = vr_waveform.repeat(1, target_ch, 1)
                    elif target_ch == 1:
                        vr_waveform = vr_waveform.mean(dim=1, keepdim=True)
                # Encode via same path as main audio
                from .ltx_core.model.audio_vae.ops import AudioProcessor as _VRAudioProcessor
                vr_processor = _VRAudioProcessor(
                    sample_rate=self.audio_encoder.sample_rate,
                    mel_bins=self.audio_encoder.mel_bins,
                    mel_hop_length=self.audio_encoder.mel_hop_length,
                    n_fft=self.audio_encoder.n_fft,
                )
                # Force the mel computation onto CPU. MelSpectrogram creates its
                # internal `window` tensor lazily inside waveform_to_mel; an active
                # torch.device("cuda") default context otherwise allocates that
                # window on the GPU, which OOMs when VRAM is already full from the
                # video model + LoRAs (the reported ID-LoRA voice-ref crash).
                with torch.device("cpu"):
                    vr_waveform = vr_waveform.to(device="cpu", dtype=torch.float32)
                    vr_processor = vr_processor.to(vr_waveform.device)
                    vr_mel = vr_processor.waveform_to_mel(vr_waveform, voice_ref_sr)
                audio_params = next(self.audio_encoder.parameters(), None)
                audio_device = audio_params.device if audio_params is not None else self.device
                audio_dtype = audio_params.dtype if audio_params is not None else self.dtype
                vr_mel = vr_mel.to(device=audio_device, dtype=audio_dtype)
                with torch.inference_mode():
                    vr_latent = self.audio_encoder(vr_mel)
                vr_latent = vr_latent.to(device=self.device, dtype=self.dtype)
                # Compute positions for the reference audio
                from .ltx_core.components.patchifiers import AudioPatchifier as _VRAudioPatchifier
                vr_patchifier = _VRAudioPatchifier(patch_size=1)
                vr_shape = AudioLatentShape(
                    batch=vr_latent.shape[0],
                    channels=vr_latent.shape[1],
                    frames=vr_latent.shape[2],
                    mel_bins=vr_latent.shape[3],
                )
                vr_positions = vr_patchifier.get_patch_grid_bounds(vr_shape, device=self.device)
                # Store on pipeline components — still needed so distilled.py
                # can detect ID-LoRA active and trigger freeze_audio on
                # stage 2. The ref_audio_latent / ref_audio_positions
                # fallback path in helpers.py.denoise_audio_video stays as
                # a safety net; primary path is now via audio_conditionings
                # below, matching WanGP v11.77.
                self.pipeline.pipeline_components.ref_audio_latent = vr_latent
                self.pipeline.pipeline_components.ref_audio_positions = vr_positions
                self.pipeline.pipeline_components.identity_guidance_scale = identity_guidance_scale
                # Primary integration: build AudioConditionByReferenceLatent and
                # append to audio_conditionings so the regular conditioning
                # system applies it INSIDE noise_audio_state (matches WanGP
                # ltx2.py line 1253 — `audio_conditionings = [AudioConditionByReferenceLatent(audio_latent)]`).
                # This ensures the prepended ref tokens are present BEFORE
                # the noiser runs and BEFORE _prewarm computes
                # prepared_audio_context, instead of being patched in
                # afterwards through the out-of-band pipeline_components
                # mechanism. The out-of-band path stays as a safety net
                # for older callers that don't go through this branch.
                if audio_conditionings is None:
                    audio_conditionings = []
                audio_conditionings = list(audio_conditionings) + [AudioConditionByReferenceLatent(vr_latent)]
                ref_dur = vr_latent.shape[2] * 160 * 4 / 16000
                print(f"[ID-LoRA] Voice reference encoded: {vr_latent.shape}, ~{ref_dur:.1f}s, "
                      f"identity_scale={identity_guidance_scale}, "
                      f"audio_conditionings now has {len(audio_conditionings)} item(s)")
            except Exception as e:
                print(f"[ID-LoRA] Failed to encode voice reference (non-fatal): {e}")
                import traceback
                traceback.print_exc()
        else:
            # Clear any previous reference
            if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'pipeline_components'):
                self.pipeline.pipeline_components.ref_audio_latent = None
                self.pipeline.pipeline_components.ref_audio_positions = None
                self.pipeline.pipeline_components.identity_guidance_scale = 0.0

        target_height = int(height)
        target_width = int(width)
        if target_height % 64 != 0:
            target_height = int(math.ceil(target_height / 64) * 64)
        if target_width % 64 != 0:
            target_width = int(math.ceil(target_width / 64) * 64)

        if latent_conditioning_stage2 is not None:
            expected_lat_h = target_height // 32
            expected_lat_w = target_width // 32
            if (
                latent_conditioning_stage2.shape[3] != expected_lat_h
                or latent_conditioning_stage2.shape[4] != expected_lat_w
            ):
                latent_conditioning_stage2 = None
            else:
                latent_conditioning_stage2 = latent_conditioning_stage2.to(device=self.device, dtype=self.dtype)

        if _uses_native_two_stage_dispatch(
            self.pipeline,
            active_official_outpaint,
        ):
            negative_prompt = n_prompt if n_prompt else DEFAULT_NEGATIVE_PROMPT
            # Reference-workflow variant (Advanced Settings toggle). A lazily
            # created sibling pipeline that shares the standard pipeline's
            # models, components, and text-encoder cache — the standard
            # two-stage pipeline object is never mutated. Mirrors the
            # progressive_pipeline lazy-swap below.
            active_two_stage = self.pipeline
            if reference_pipeline:
                if not hasattr(self, '_reference_pipeline'):
                    from .ltx_pipelines.ti2vid_two_stages_ref import TI2VidTwoStagesRefPipeline
                    self._reference_pipeline = TI2VidTwoStagesRefPipeline(
                        device=self.device,
                        stage_1_models=self.pipeline.stage_1_models,
                        stage_2_models=self.pipeline.stage_2_models,
                    )
                    # Share mutable state so ID-LoRA refs and cached text
                    # embeddings behave identically on both paths.
                    self._reference_pipeline.pipeline_components = self.pipeline.pipeline_components
                    self._reference_pipeline.text_encoder_cache = self.pipeline.text_encoder_cache
                    print("[LTX2] Reference two-stage pipeline initialized (lazy)")
                active_two_stage = self._reference_pipeline
            pipeline_output = active_two_stage(
                prompt=input_prompt,
                negative_prompt=negative_prompt,
                seed=int(seed),
                height=target_height,
                width=target_width,
                num_frames=int(frame_num),
                frame_rate=float(fps),
                num_inference_steps=int(sampling_steps),
                cfg_guidance_scale=float(guide_scale),
                audio_cfg_guidance_scale=float(guide_scale if audio_cfg_scale is None else audio_cfg_scale),
                cfg_star_switch=cfg_star_switch,
                apg_switch=apg_switch,
                perturbation_switch=perturbation_switch,
                perturbation_layers=perturbation_layers,
                perturbation_start=perturbation_start,
                perturbation_end=perturbation_end,
                alt_guidance_scale=float(alt_guide_scale),
                alt_scale=float(alt_scale),
                images=images,
                end_images=end_images or None,
                end_images_stage2=end_images_stage2 or None,
                inject_images=inject_images or None,
                inject_images_stage2=inject_images_stage2 or None,
                images_stage2=images_stage2 if stage2_override else None,
                video_conditioning=video_conditioning,
                video_conditioning_downscale_factor=video_conditioning_downscale_factor,
                latent_conditioning_stage2=latent_conditioning_stage2,
                tiling_config=tiling_config,
                enhance_prompt=False,
                audio_conditionings=audio_conditionings,
                callback=callback,
                interrupt_check=interrupt_check,
                loras_slists=loras_slists,
                text_connectors=text_connectors,
                masking_source=masking_source,
                masking_strength=masking_strength,
                return_latent_slice=return_latent_slice,
                self_refiner_setting=self_refiner_setting,
                self_refiner_plan=self_refiner_plan,
                self_refiner_f_uncertainty=self_refiner_f_uncertainty,
                self_refiner_certain_percentage=self_refiner_certain_percentage,
                self_refiner_max_plans=self_refiner_max_plans,
                stg_scale=stg_scale,
                cfg_rescale=cfg_rescale,
                modality_scale=modality_scale,
                use_gradient_estimation=use_gradient_estimation,
                ge_gamma=ge_gamma,
                keyframe_conditioning_mode=keyframe_conditioning_mode,
                keyframe_inject_mode=keyframe_inject_mode,
                sample_solver=sample_solver,
                stg_schedule=stg_schedule,
                text_attention_amplifier=text_attention_amplifier,
            )
        else:
            # Select pipeline: progressive 3-stage or standard distilled 2-stage
            active_pipeline = active_official_outpaint or self.pipeline
            _progressive_pad = None
            if progressive_pipeline and active_official_outpaint is None:
                if isinstance(self.pipeline, DistilledPipeline):
                    if not hasattr(self, '_progressive_pipeline'):
                        from .ltx_pipelines.progressive import ProgressivePipeline
                        self._progressive_pipeline = ProgressivePipeline(
                            device=self.device, models=self.pipeline.models,
                        )
                        print("[LTX2] Progressive 3-stage pipeline initialized (lazy)")
                    active_pipeline = self._progressive_pipeline
                    print("[LTX2] Using progressive 3-stage pipeline")

            _pipeline_kwargs = dict(
                prompt=input_prompt,
                seed=int(seed),
                height=target_height,
                width=target_width,
                num_frames=int(frame_num),
                frame_rate=float(fps),
                images=images,
                end_images=end_images or None,
                end_images_stage2=end_images_stage2 or None,
                inject_images=inject_images or None,
                inject_images_stage2=inject_images_stage2 or None,
                alt_guidance_scale=float(alt_guide_scale),
                video_conditioning=video_conditioning,
                video_conditioning_downscale_factor=video_conditioning_downscale_factor,
                latent_conditioning_stage2=latent_conditioning_stage2,
                tiling_config=tiling_config,
                enhance_prompt=False,
                audio_conditionings=audio_conditionings,
                callback=callback,
                interrupt_check=interrupt_check,
                loras_slists=loras_slists,
                text_connectors=text_connectors,
                masking_source=masking_source,
                masking_strength=masking_strength,
                return_latent_slice=return_latent_slice,
                self_refiner_setting=self_refiner_setting,
                self_refiner_plan=self_refiner_plan,
                self_refiner_f_uncertainty=self_refiner_f_uncertainty,
                self_refiner_certain_percentage=self_refiner_certain_percentage,
                self_refiner_max_plans=self_refiner_max_plans,
                stage2_steps=stage2_steps,
                keyframe_conditioning_mode=keyframe_conditioning_mode,
                keyframe_inject_mode=keyframe_inject_mode,
            )
            if _uses_distilled_pipeline_dispatch(
                active_pipeline,
                active_official_outpaint,
            ):
                # DistilledPipeline-only kwargs. ProgressivePipeline doesn't
                # accept NAG, negative_prompt, or single_stage — passing any
                # of these to it raises TypeError. Gate them here.
                video_conditioning_generation_mask = (
                    _select_video_conditioning_generation_mask(
                        official_outpaint_pipeline,
                        input_frames,
                        input_masks,
                        input_masks2,
                    )
                )
                if active_official_outpaint is not None:
                    print(
                        "[LTX2] Official Outpaint dispatch: "
                        f"{type(active_pipeline).__module__}."
                        f"{type(active_pipeline).__name__}; "
                        "production_pixel_handoff="
                        f"{full_resolution_outpaint_refine}; "
                        "generation_mask="
                        f"{video_conditioning_generation_mask is not None}; "
                        "reference_attention="
                        f"{'full' if full_resolution_outpaint_refine else 'source-masked'}; "
                        "learned_latent_upscale="
                        f"{not full_resolution_outpaint_refine}."
                    )
                _pipeline_kwargs.update(
                    negative_prompt=(
                        ""
                        if active_official_outpaint is not None
                        else (n_prompt if n_prompt else DEFAULT_NEGATIVE_PROMPT)
                    ),
                    NAG_scale=float(NAG_scale),
                    NAG_tau=float(NAG_tau),
                    NAG_alpha=float(NAG_alpha),
                    single_stage=bool(kwargs.get("single_stage_pipeline", False)),
                    full_resolution_refine=(
                        full_resolution_outpaint_refine
                    ),
                    use_ancestral_sampler=(
                        self.base_model_type == "ltx2_25_22B"
                    ),
                    video_conditioning_generation_mask=(
                        video_conditioning_generation_mask
                    ),
                )
            else:
                # Progressive params only for the progressive pipeline
                _pipeline_kwargs.update(
                    progressive_stage2_steps=progressive_stage2_steps,
                    progressive_stage3_steps=progressive_stage3_steps,
                    progressive_stage2_sigma=progressive_stage2_sigma,
                    progressive_stage3_sigma=progressive_stage3_sigma,
                    progressive_stage1_image_weight=progressive_stage1_image_weight,
                    progressive_stage3_image_weight=progressive_stage3_image_weight,
                )
            pipeline_output = active_pipeline(**_pipeline_kwargs)

        latent_slice = None
        if isinstance(pipeline_output, tuple) and len(pipeline_output) == 3:
            video, audio, latent_slice = pipeline_output
        else:
            video, audio = pipeline_output

        if video is None or audio is None:
            return None

        if self._interrupt:
            return None

        video_tensor = _collect_video_chunks(
            video,
            interrupt_check=interrupt_check,
            expected_frames=int(frame_num),
            expected_height=int(height),
            expected_width=int(width),
        )
        if video_tensor is None:
            return None

        video_tensor = video_tensor[:, :frame_num, :height, :width]
        if use_outpaint_gamma_roundtrip:
            # Defensive: in-place gamma mutation below fails if video_tensor is
            # still in inference_mode. video_vae.decode_video_to_tensor was
            # updated (per upstream WanGP commit 5da7f23) to allocate the
            # buffer outside inference_mode for exactly this reason.
            if torch.is_inference(video_tensor):
                raise RuntimeError(
                    "LTX2 decoded video output is still an inference tensor; "
                    "decode_video_to_tensor must allocate the output buffer "
                    "outside inference mode."
                )
            exponent = float(LTX2_OUTPAINT_GAMMA)
            if video_tensor.dtype == torch.uint8:
                corrected = video_tensor.to(dtype=torch.float32).div_(255.0).clamp_(0.0, 1.0).pow_(exponent)
                video_tensor.copy_(corrected.mul_(255.0).round_().clamp_(0.0, 255.0).to(dtype=torch.uint8))
            else:
                corrected = video_tensor.to(dtype=torch.float32).add_(1.0).mul_(0.5).clamp_(0.0, 1.0).pow_(exponent)
                video_tensor.copy_(corrected.mul_(2.0).sub_(1.0).to(dtype=video_tensor.dtype))
        if (
            mask_preserving_outpaint
            and LTX2_OUTPAINTING_LAPLACIAN_BLEND
        ):
            blend_source = (
                input_frames
                if input_frames is not None
                else input_frames2
            )
            blend_mask = (
                input_masks
                if input_masks is not None
                else input_masks2
            )
            production_outpaint_blend = bool(
                official_outpaint_pipeline
                and full_resolution_outpaint_refine
            )
            video_tensor = _apply_ltx2_mask_blend(
                video_tensor,
                blend_source,
                blend_mask,
                frame_num,
                height,
                width,
                mask_low_res_dilation=(
                    # The published LTX-2.3 two-stage graph uses dilation 5
                    # for the half-resolution handoff and 2 for the final
                    # full-resolution Laplacian blend.
                    2
                    if production_outpaint_blend
                    else LTX2_OUTPAINTING_LAPLACIAN_MASK_LOW_RES_DILATION
                ),
                source_feather_pixels=(
                    LTX2_OUTPAINTING_SOURCE_FEATHER_PIXELS
                ),
                # The official path owns the generated margins. Post color
                # correction is deliberately disabled; only source pixels in
                # the narrow boundary feather are restored below.
                match_generated_canvas=not bool(
                    kwargs.get("single_stage_pipeline", False)
                    or official_outpaint_pipeline
                ),
                blend_mode=(
                    "gaussian"
                    if (
                        official_outpaint_pipeline
                        and not production_outpaint_blend
                    )
                    else "laplacian"
                ),
                # Lightricks applies the complete Laplacian result after both
                # passes. With the official dilation 2 and Kornia-equivalent
                # pyramid geometry, low-frequency color crosses the seam
                # while source high-frequency detail remains anchored.
                full_frame_laplacian=production_outpaint_blend,
                correct_marker_residue=bool(
                    production_outpaint_blend
                    and LTX2_OUTPAINTING_MARKER_RESIDUE_CLEANUP
                ),
            )
            if production_outpaint_blend:
                seam_label = "the marker-safe official Laplacian blend "
            elif official_outpaint_pipeline:
                seam_label = "the Diffusers-reference Gaussian seam "
            else:
                seam_label = "a bounded Laplacian seam "
            print(
                f"[LTX2] Restored protected source with "
                f"{seam_label}"
                f"({LTX2_OUTPAINTING_SOURCE_FEATHER_PIXELS}px source "
                "feather; generated canvas left ungraded)."
            )
        audio_np = audio.detach().float().cpu().numpy() if audio is not None else None
        if audio_np is not None and audio_np.ndim == 2:
            if audio_np.shape[0] in (1, 2) and audio_np.shape[1] > audio_np.shape[0]:
                audio_np = audio_np.T
        output_audio_sampling_rate = int(getattr(self.vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE))
        result = {
            "x": video_tensor,
            "audio": audio_np,
            "audio_sampling_rate": output_audio_sampling_rate,
        }
        if latent_slice is not None:
            result["latent_slice"] = latent_slice
        if _retake_stitch_info is not None:
            result["retake_stitch_info"] = _retake_stitch_info
        return result
