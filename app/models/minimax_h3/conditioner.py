"""Qwen3-VL layer-50 conditioning for MiniMax H3."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2VLImageProcessorFast

from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig, register_qwen3_vl_config
from models.ideogram4.qwen3_vl_transformers import Qwen3VLModel, Qwen3VLTextModel, Qwen3VLVisionModel
from models.krea2.krea2_main import Krea2Qwen3VLProcessor


VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
TEXT_ENCODER_LAYERS = 50

# Qwen3-VL vision geometry. A visual block is always `temporal_patch_size` frames: an image is its own
# frame repeated, a video contributes consecutive pairs.
_PATCH_SIZE = 16
_TEMPORAL_PATCH_SIZE = 2
_MERGE_SIZE = 2
_IMAGE_PIXEL_BOUNDS = (65536, 16777216)
_VIDEO_PIXEL_BOUNDS = (4096, 25165824)


def _visual_patches(frames: torch.Tensor, video: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn `temporal_patch_size` THWC RGB frames into the flattened patches Qwen3-VL's vision tower takes.

    Returns the patches and the `(t, h, w)` grid describing them. Video blocks get a far wider pixel budget
    than stills, which is what lets a clip be sampled at a useful resolution without blowing up the sequence.
    """
    if frames.shape[0] == 1:
        frames = frames.repeat(_TEMPORAL_PATCH_SIZE, 1, 1, 1)
    if frames.shape[0] != _TEMPORAL_PATCH_SIZE:
        raise ValueError(f"A Qwen visual block needs {_TEMPORAL_PATCH_SIZE} frames, got {frames.shape[0]}.")
    _, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"Qwen visual input must be RGB, got {channels} channels.")

    factor = _PATCH_SIZE * _MERGE_SIZE
    min_pixels, max_pixels = _VIDEO_PIXEL_BOUNDS if video else _IMAGE_PIXEL_BOUNDS
    target_h = max(factor, round(height / factor) * factor)
    target_w = max(factor, round(width / factor) * factor)
    if target_h * target_w > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        target_h = max(factor, math.floor(height / scale / factor) * factor)
        target_w = max(factor, math.floor(width / scale / factor) * factor)
    elif target_h * target_w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        target_h = math.ceil(height * scale / factor) * factor
        target_w = math.ceil(width * scale / factor) * factor

    images = F.interpolate(
        frames.permute(0, 3, 1, 2), size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True
    )
    images = images.mul(2.0).sub_(1.0)
    grid_h, grid_w = target_h // _PATCH_SIZE, target_w // _PATCH_SIZE
    patches = images.reshape(
        1, _TEMPORAL_PATCH_SIZE, 3,
        grid_h // _MERGE_SIZE, _MERGE_SIZE, _PATCH_SIZE,
        grid_w // _MERGE_SIZE, _MERGE_SIZE, _PATCH_SIZE,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flattened = patches.reshape(grid_h * grid_w, 3 * _TEMPORAL_PATCH_SIZE * _PATCH_SIZE * _PATCH_SIZE)
    grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long, device=frames.device)
    return flattened, grid


class MiniMaxH3Int8Embedding(nn.Module):
    """Row-scaled INT8 embedding used by the Comfy MiniMax H3 checkpoint.

    The checkpoint keeps the Qwen vocabulary table quantized and stores one
    floating-point scale per vocabulary row.  Looking up the INT8 rows before
    dequantizing them avoids materializing the full 1.5 GB BF16 table.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        output_dtype: torch.dtype,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.output_dtype = output_dtype
        # MMGP normally requires every unquantized parameter in a model to
        # share its execution dtype.  This module deliberately keeps mixed
        # INT8 weights and FP32 row scales while producing BF16/FP16 output.
        # Locking the storage dtype prevents profiling and later dtype-change
        # passes from converting either checkpoint tensor.
        self._lock_dtype = output_dtype
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty((num_embeddings, 1), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        quantized_rows = F.embedding(input_ids, self.weight, self.padding_idx)
        row_scales = F.embedding(input_ids, self.weight_scale, self.padding_idx)
        return quantized_rows.to(self.output_dtype) * row_scales.to(self.output_dtype)


class MiniMaxH3PreScaledLinear(nn.Linear):
    """AWQ/NVFP4 linear with the checkpoint's input smoothing scale."""

    def __init__(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.register_buffer("pre_quant_scale", torch.empty(in_features, dtype=dtype), persistent=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.pre_quant_scale.to(device=input.device, dtype=input.dtype)
        return F.linear(input * scale, self.weight, self.bias)


class MiniMaxH3Qwen3VL(nn.Module):
    """Checkpoint-shaped Qwen3-VL wrapper.

    The consumer checkpoint uses the top-level prefixes ``model`` and
    ``visual`` and ends after decoder layer 50.  H3 consumes that layer's
    unnormalized output, so the absent final norm is intentionally replaced by
    an identity module.
    """

    def __init__(self, config: Qwen3VLConfig, dtype: torch.dtype | None = None):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.model = Qwen3VLTextModel(config.text_config)
        source_embedding = self.model.embed_tokens
        self.model.embed_tokens = MiniMaxH3Int8Embedding(
            source_embedding.num_embeddings,
            source_embedding.embedding_dim,
            source_embedding.padding_idx,
            output_dtype=dtype or source_embedding.weight.dtype,
        )
        self.model.norm = nn.Identity()
        for layer in self.model.layers:
            down = layer.mlp.down_proj
            layer.mlp.down_proj = MiniMaxH3PreScaledLinear(
                down.in_features,
                down.out_features,
                down.bias is not None,
                down.weight.dtype,
            )
            out = layer.self_attn.o_proj
            layer.self_attn.o_proj = MiniMaxH3PreScaledLinear(
                out.in_features,
                out.out_features,
                out.bias is not None,
                out.weight.dtype,
            )

    get_rope_index = Qwen3VLModel.get_rope_index


def load_h3_qwen_config(config_path: str) -> Qwen3VLConfig:
    register_qwen3_vl_config()
    config = Qwen3VLConfig.from_json_file(config_path)
    config.text_config.num_hidden_layers = TEXT_ENCODER_LAYERS
    return config


def build_h3_processor(config_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(config_dir, trust_remote_code=False)
    image_processor = Qwen2VLImageProcessorFast.from_pretrained(config_dir)
    return tokenizer, Krea2Qwen3VLProcessor(image_processor, tokenizer)


def _tag_vision_spans(input_ids: torch.Tensor) -> torch.Tensor:
    """Return H3 modality tags, including the vision boundary tokens."""

    ids = input_ids[0].tolist()
    tags = torch.ones(len(ids), dtype=torch.long)
    start = None
    for index, token in enumerate(ids):
        if token == VISION_START_TOKEN_ID:
            start = index
        if token == VISION_END_TOKEN_ID and start is not None:
            tags[start : index + 1] = 0
            start = None
    if start is not None:
        tags[start:] = 0
    return tags


class MiniMaxH3Conditioner(nn.Module):
    def __init__(self, qwen: MiniMaxH3Qwen3VL, tokenizer, processor, max_text_tokens: int = 512):
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_text_tokens = max_text_tokens
        self._interrupt = False

    @property
    def language_model(self):
        return self.qwen.model

    @property
    def visual(self):
        return self.qwen.visual

    def _plain_inputs(self, prompt: str, device: torch.device):
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        # Match the MiniMax/Diffusers presentation exactly: there is no chat
        # template or padding, but Qwen still receives the all-live tokenizer
        # mask and applies its native causal attention internally.
        attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.bool)
        return input_ids, attention_mask, None, encoded

    @staticmethod
    def _presentation(prompt: str, num_images: int, num_audio_references: int) -> str:
        """Announce the conditioning material ahead of the prompt, the way MiniMax-H3 was trained to read it.

        Every visual gets a numbered `<Picture N>` header followed by its vision block. An audio reference is
        announced by a bare `<Audio N>` header: the waveform itself never reaches the text encoder, it is conditioned
        on through the packed sequence, so the marker exists only to tell the prompt which references it may refer
        to. Headers are numbered per modality, and audio follows the pictures.
        """
        return (
            "".join(
                f"<Picture {index + 1}>: <|vision_start|><|image_pad|><|vision_end|>" for index in range(num_images)
            )
            + "".join(f"<Audio {index + 1}>: " for index in range(num_audio_references))
            + prompt
        )

    def _reference_entries(self, prompt: str, references: list):
        """Interleave the presentation's token ids with the visual blocks they announce.

        A video is not one block: Qwen3-VL reads a clip as a run of `<seconds>` markers each followed by a
        two-frame block, which is the layout `get_rope_index` expects when it splits a video grid per frame.
        Audio is announced by a bare header -- the waveform reaches the model through the packed sequence.
        """
        entries: list = []
        counters = {"image": 0, "video": 0, "audio": 0}

        def add_text(text: str) -> None:
            entries.extend(self.tokenizer(text, add_special_tokens=False)["input_ids"])

        def add_visual(frames: torch.Tensor, video: bool) -> None:
            entries.extend((VISION_START_TOKEN_ID, {"frames": frames, "video": video}, VISION_END_TOKEN_ID))

        for item in references:
            kind = item["type"]
            if kind not in counters:
                raise ValueError(f"A reference kind must be 'image', 'video' or 'audio', got {kind!r}.")
            counters[kind] += 1
            if kind == "image":
                add_text(f"<Picture {counters[kind]}>: ")
                add_visual(item["frames"], video=False)
            elif kind == "audio":
                add_text(f"<Audio {counters[kind]}>: ")
            else:
                add_text(f"<Video {counters[kind]}>: ")
                frames = item["frames"]
                timestamps = list(item.get("timestamps") or [index / 2.0 for index in range(frames.shape[0])])
                if frames.shape[0] % _TEMPORAL_PATCH_SIZE:
                    frames = torch.cat((frames, frames[-1:]), dim=0)
                    timestamps.append(timestamps[-1])
                for index in range(0, frames.shape[0], _TEMPORAL_PATCH_SIZE):
                    add_text(f"<{(timestamps[index] + timestamps[index + 1]) / 2.0:.1f} seconds>")
                    add_visual(frames[index : index + _TEMPORAL_PATCH_SIZE], video=True)
        add_text(prompt)
        return entries

    def _reference_inputs(self, prompt: str, references: list, device: torch.device):
        """Assemble a mixed image/video/audio reference prompt by hand.

        The processor can only expand `<|image_pad|>` for stills, so a clip has to be patched, embedded and
        expanded here: every block contributes as many pad tokens as the vision tower returns rows for it.
        """
        entries = self._reference_entries(prompt, references)

        input_ids: list[int] = []
        blocks = []
        for entry in entries:
            if isinstance(entry, int):
                input_ids.append(entry)
                continue
            if self._interrupt:
                return None
            frames = entry["frames"].to(device=device, dtype=torch.float32)
            patches, grid = _visual_patches(frames, video=entry["video"])
            merged, deepstack = self.qwen.visual(patches, grid_thw=grid)
            if merged is None:
                return None
            start = len(input_ids)
            input_ids.extend([VIDEO_TOKEN_ID if entry["video"] else IMAGE_TOKEN_ID] * merged.shape[0])
            blocks.append(
                {
                    "start": start,
                    "size": merged.shape[0],
                    "merged": merged,
                    "deepstack": deepstack,
                    "grid": grid,
                    "video": entry["video"],
                }
            )

        ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        inputs_embeds = self.qwen.model.embed_tokens(ids)
        visual_mask = torch.zeros(ids.shape, dtype=torch.bool, device=device)
        deepstack = None
        for block in blocks:
            end = block["start"] + block["size"]
            inputs_embeds[0, block["start"] : end] = block["merged"].to(inputs_embeds.dtype)
            visual_mask[0, block["start"] : end] = True
            if deepstack is None:
                deepstack = list(block["deepstack"])
            else:
                deepstack = [torch.cat((deepstack[i], block["deepstack"][i]), dim=0) for i in range(len(deepstack))]

        attention_mask = torch.ones(ids.shape, dtype=torch.bool, device=device)
        image_grids = [b["grid"] for b in blocks if not b["video"]]
        video_grids = [b["grid"] for b in blocks if b["video"]]
        position_ids, _ = self.qwen.get_rope_index(
            ids,
            image_grid_thw=torch.cat(image_grids) if image_grids else None,
            video_grid_thw=torch.cat(video_grids) if video_grids else None,
            attention_mask=attention_mask,
        )
        return ids, attention_mask, position_ids, inputs_embeds, visual_mask, deepstack

    def _vision_inputs(self, presentation: str, images: list, device: torch.device):
        encoded = self.processor(
            text=[presentation],
            images=images,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_text_tokens + 4096,
            return_tensors="pt",
        ).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        return input_ids, attention_mask, None, encoded

    @torch.inference_mode()
    def forward(
        self,
        prompt: str,
        device: torch.device,
        images: list | None = None,
        num_audio_references: int = 0,
        references: list | None = None,
    ):
        self.qwen.model._interrupt = self._interrupt
        self.qwen.visual._interrupt = self._interrupt
        if self._interrupt:
            return None, None
        presentation = self._presentation(prompt, len(images or ()), num_audio_references)
        if references is not None:
            # Mixed reference sets go through the hand-assembled path: the processor cannot expand a clip.
            assembled = self._reference_inputs(prompt, references, device)
            if assembled is None:
                return None, None
            input_ids, attention_mask, position_ids, inputs_embeds, visual_mask, deepstack = assembled
        elif images:
            input_ids, attention_mask, position_ids, processor_inputs = self._vision_inputs(
                presentation, images, device
            )
            grid = processor_inputs["image_grid_thw"]
            pixels = processor_inputs["pixel_values"].to(device=device, dtype=torch.float32)
            image_embeds, deepstack = self.qwen.visual(pixels, grid_thw=grid)
            if image_embeds is None or self._interrupt:
                return None, None
            inputs_embeds = self.qwen.model.embed_tokens(input_ids)
            visual_mask = input_ids == IMAGE_TOKEN_ID
            inputs_embeds = inputs_embeds.masked_scatter(
                visual_mask.unsqueeze(-1).expand_as(inputs_embeds),
                image_embeds.to(inputs_embeds.dtype),
            )
            position_ids, _ = self.qwen.get_rope_index(
                input_ids,
                image_grid_thw=grid,
                attention_mask=attention_mask,
            )
        else:
            input_ids, attention_mask, position_ids, _ = self._plain_inputs(presentation, device)
            inputs_embeds = visual_mask = deepstack = None

        outputs = self.qwen.model(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
            return_mid_results_layers=[TEXT_ENCODER_LAYERS - 1],
        )
        if outputs.last_hidden_state is None or not outputs.mid_results:
            return None, None
        # The layer snapshot is taken before the (absent) final norm.
        embeddings = outputs.mid_results[0]
        tags = _tag_vision_spans(input_ids)
        return embeddings, tags
