"""Qwen3-VL layer-50 conditioning for MiniMax H3."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2VLImageProcessorFast, Qwen3VLVideoProcessor

from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig, register_qwen3_vl_config
from models.ideogram4.qwen3_vl_transformers import Qwen3VLModel, Qwen3VLTextModel, Qwen3VLVisionModel
from models.krea2.krea2_main import Krea2Qwen3VLProcessor


VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
TEXT_ENCODER_LAYERS = 50


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

    def __init__(
        self,
        config: Qwen3VLConfig,
        dtype: torch.dtype | None = None,
        *,
        consumer_quantized: bool = True,
    ):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.model = Qwen3VLTextModel(config.text_config)
        if consumer_quantized:
            source_embedding = self.model.embed_tokens
            self.model.embed_tokens = MiniMaxH3Int8Embedding(
                source_embedding.num_embeddings,
                source_embedding.embedding_dim,
                source_embedding.padding_idx,
                output_dtype=dtype or source_embedding.weight.dtype,
            )
        self.model.norm = nn.Identity()
        if consumer_quantized:
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
    processor = Krea2Qwen3VLProcessor(image_processor, tokenizer)
    # Ref2VA uses Qwen's dedicated temporal processor. Keep it attached to
    # the existing lightweight processor wrapper so FL2VA remains unchanged.
    processor.video_processor = Qwen3VLVideoProcessor.from_pretrained(config_dir)
    return tokenizer, processor


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
    def __init__(
        self,
        qwen: MiniMaxH3Qwen3VL,
        tokenizer,
        processor,
        *,
        gguf_vision_autocast: bool = False,
    ):
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor
        # GGUF Qwen3-VL vision checkpoints deliberately mix FP16 projection
        # and quantized attention weights with FP32 LayerNorm parameters. A
        # single input cast cannot satisfy that tower: patch embedding forces
        # hidden states back to FP16, then an un-autocast FP32 LayerNorm raises
        # ``expected scalar type Half but found Float``. CUDA FP16 autocast is
        # the checkpoint's intended execution contract; keep it scoped to
        # GGUF so the known-good NVFP4/AWQ path remains unchanged.
        self.gguf_vision_autocast = bool(gguf_vision_autocast)
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
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        # Match the MiniMax/Diffusers presentation exactly: there is no chat
        # template or padding, but Qwen still receives the all-live tokenizer
        # mask and applies its native causal attention internally.
        attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.bool)
        return input_ids, attention_mask, None, encoded

    def _vision_inputs(self, prompt: str, images: list, device: torch.device):
        presentation = "".join(
            f"<Picture {index + 1}>: <|vision_start|><|image_pad|><|vision_end|>"
            for index in range(len(images))
        ) + prompt
        encoded = self.processor(
            text=[presentation],
            images=images,
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        ).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        return input_ids, attention_mask, None, encoded

    def _encode_visual(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        """Run the independently offloaded Qwen vision tower safely."""

        pixels = pixel_values.to(device=device, dtype=torch.float32)
        grid = grid_thw.to(device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.gguf_vision_autocast and device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            return self.qwen.visual(pixels, grid_thw=grid)

    @staticmethod
    def _merge_deepstack(
        image_mask: torch.Tensor | None,
        video_mask: torch.Tensor | None,
        image_deepstack: list[torch.Tensor] | None,
        video_deepstack: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        if image_mask is not None and video_mask is not None:
            visual_mask = image_mask | video_mask
            image_joint = image_mask[visual_mask]
            video_joint = video_mask[visual_mask]
            deepstack = []
            for image_embed, video_embed in zip(image_deepstack or [], video_deepstack or []):
                joint = image_embed.new_zeros(
                    (int(visual_mask.sum().item()), image_embed.shape[-1]),
                    device=image_embed.device,
                )
                joint[image_joint] = image_embed
                joint[video_joint] = video_embed
                deepstack.append(joint)
            return visual_mask, deepstack
        if image_mask is not None:
            return image_mask, image_deepstack
        if video_mask is not None:
            return video_mask, video_deepstack
        return None, None

    @torch.inference_mode()
    def forward_ref2va(self, prompt: str, device: torch.device, references: list):
        """Encode the official ordered Ref2VA media presentation."""

        from .ref2va import build_ref2va_presentation, sample_reference_video_frames

        self.qwen.model._interrupt = self._interrupt
        self.qwen.visual._interrupt = self._interrupt
        if self._interrupt:
            return None, None

        merge_size = self.processor.image_processor.merge_size**2
        pixel_values = image_grid_thw = None
        image_token_counts: list[int] = []
        images = [reference.image for reference in references if reference.kind == "image"]
        if images:
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
            image_token_counts = [int(grid.prod()) // merge_size for grid in image_grid_thw]

        pixel_values_videos = video_grid_thw = None
        video_block_token_counts: list[int] = []
        videos = [reference for reference in references if reference.kind == "video"]
        if videos:
            sampled = [sample_reference_video_frames(reference.frames) for reference in videos]
            for reference, (_, timestamps) in zip(videos, sampled):
                reference.block_timestamps = timestamps
            vision = self.processor.video_processor(
                videos=[np.stack(frames) for frames, _ in sampled],
                do_sample_frames=False,
                return_tensors="pt",
            )
            pixel_values_videos = vision["pixel_values_videos"]
            video_grid_thw = vision["video_grid_thw"]
            video_block_token_counts = [int(grid[1]) * int(grid[2]) // merge_size for grid in video_grid_thw]
            for reference, grid in zip(videos, video_grid_thw):
                if int(grid[0]) != len(reference.block_timestamps):
                    raise ValueError(
                        f"The Qwen processor created {int(grid[0])} video blocks, but MiniMax H3 labels "
                        f"{len(reference.block_timestamps)} blocks for that reference."
                    )

        token_ids, token_tags = build_ref2va_presentation(
            self.tokenizer,
            prompt,
            references,
            image_token_counts,
            video_block_token_counts,
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        inputs_embeds = self.qwen.model.embed_tokens(input_ids)

        image_mask = video_mask = None
        image_deepstack = video_deepstack = None
        if pixel_values is not None:
            image_embeds, image_deepstack = self._encode_visual(
                pixel_values,
                image_grid_thw,
                device,
            )
            if image_embeds is None or self._interrupt:
                return None, None
            image_mask = input_ids == IMAGE_TOKEN_ID
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                image_embeds.to(inputs_embeds.dtype),
            )
        if pixel_values_videos is not None:
            video_embeds, video_deepstack = self._encode_visual(
                pixel_values_videos,
                video_grid_thw,
                device,
            )
            if video_embeds is None or self._interrupt:
                return None, None
            video_mask = input_ids == VIDEO_TOKEN_ID
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.unsqueeze(-1).expand_as(inputs_embeds),
                video_embeds.to(inputs_embeds.dtype),
            )

        visual_mask, deepstack = self._merge_deepstack(
            image_mask,
            video_mask,
            image_deepstack,
            video_deepstack,
        )
        position_ids, _ = self.qwen.get_rope_index(
            input_ids,
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
            video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(device),
            attention_mask=attention_mask,
        )
        outputs = self.qwen.model(
            input_ids=None,
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
        return outputs.mid_results[0], torch.tensor(token_tags, dtype=torch.long)

    @torch.inference_mode()
    def forward(self, prompt: str, device: torch.device, images: list | None = None):
        self.qwen.model._interrupt = self._interrupt
        self.qwen.visual._interrupt = self._interrupt
        if self._interrupt:
            return None, None
        if images:
            input_ids, attention_mask, position_ids, processor_inputs = self._vision_inputs(prompt, images, device)
            grid = processor_inputs["image_grid_thw"]
            image_embeds, deepstack = self._encode_visual(
                processor_inputs["pixel_values"],
                grid,
                device,
            )
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
            input_ids, attention_mask, position_ids, _ = self._plain_inputs(prompt, device)
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
