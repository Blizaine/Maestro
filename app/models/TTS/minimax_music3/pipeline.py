"""Single-GPU MiniMax-Music3 inference for Maestro.

MiniMax's reference service places the autoregressive and flow-matching
stages on separate GPUs. Maestro keeps the reference generation math while
letting MMGP stream those stages through one GPU. The global language model
and the small RVQ depth decoder are co-tenants because both are touched for
every 25 Hz semantic frame; the flow transformer and vocoder run only after
that stage has been released.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache, CacheLayerMixin

from services.optional_acceleration import optional_flash_attention_available

from .condition_encoder import MiniMaxMusic3ConditionEncoder
from .rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
from .transformer import MiniMaxMusic3Transformer1DModel
from .vocoder import MiniMaxMusic3Vocoder


_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END_TOKEN_ID = 151670
_AUDIO_CFG_TOKEN_ID = 151654
_AUDIO_CODE_OFFSET = 151675
_SEMANTIC_VOCAB_SIZE = 16384
_MAX_PROMPT_TOKENS = 5000
_MAX_AUDIO_FRAMES = 7500  # Five minutes at 25 Hz, matching the public model card.

_AR_CFG_SCALE = 1.5
_AR_CFG_TOP_K = 50
_AR_SAMPLING_TOP_K = 50
_FLOW_CFG_SCALE = 1.7

_CHUNK_FRAMES = 200
_CHUNK_HOP = 100
_OVERLAP_LATENT_LENGTH = 172
_CROP_LEFT_LATENT = 86
_CROP_RIGHT_LATENT = 344 - _CROP_LEFT_LATENT

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def estimate_music3_kv_cache_bytes(
    config,
    *,
    prompt_tokens: int,
    duration_seconds: float,
    batch_size: int = 2,
) -> int:
    """Estimate the bf16 Qwen KV cache retained during semantic planning."""

    total_tokens = max(1, int(prompt_tokens)) + max(
        1, int(float(duration_seconds) * 25)
    )
    return int(
        2  # key and value
        * int(getattr(config, "num_hidden_layers", 0) or 0)
        * int(getattr(config, "num_key_value_heads", 0) or 0)
        * int(getattr(config, "head_dim", 0) or 0)
        * 2  # bf16 bytes
        * max(1, int(batch_size))
        * total_tokens
    )


class Music3PreallocatedCacheLayer(CacheLayerMixin):
    """Append-only cache without DynamicCache's per-token reallocations.

    Music3 emits 25 semantic tokens per second. Repeatedly concatenating 36
    Qwen cache tensors thousands of times leaves incompatible freed blocks in
    CUDA's allocator and eventually fills a 24 GB card even though the live
    cache is only about 1.4 GB for a 150-second song. This layer allocates its
    final storage once, mutates it in place, and returns only the populated
    prefix so attention cost still grows naturally instead of attending the
    entire maximum song length from the first token.
    """

    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max(1, int(max_cache_len))
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        batch_size, num_heads, _, head_dim = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device
        shape = (batch_size, num_heads, self.max_cache_len, head_dim)
        self.keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.values = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        next_length = self.cumulative_length + key_states.shape[-2]
        if next_length > self.max_cache_len:
            raise RuntimeError(
                "MiniMax-Music3 exceeded its preallocated language-model "
                f"cache ({next_length} > {self.max_cache_len} tokens)."
            )
        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs else None
        )
        if cache_position is None:
            cache_position = torch.arange(
                self.cumulative_length,
                next_length,
                device=self.device,
            )
        self.keys.index_copy_(2, cache_position, key_states)
        self.values.index_copy_(2, cache_position, value_states)
        self.cumulative_length = next_length
        return (
            self.keys[..., :next_length, :],
            self.values[..., :next_length, :],
        )

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return min(
            self.max_cache_len,
            self.cumulative_length + cache_position.shape[0],
        ), 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


class Music3PreallocatedCache(Cache):
    def __init__(self, config, max_cache_len: int):
        super().__init__(
            layers=[
                Music3PreallocatedCacheLayer(max_cache_len)
                for _ in range(int(config.num_hidden_layers))
            ]
        )


def _music3_attention_backend() -> str:
    # A Windows wheel can import successfully while containing no cubin for
    # the active GPU. The shared startup guard validates both ABI loading and
    # Maestro's known wheel architecture manifest before Music3 opts into FA2.
    return (
        "flash_attention_2"
        if optional_flash_attention_available()
        else "sdpa"
    )


def normalize_music3_qwen_config(config):
    """Bridge the Music3 checkpoint's Transformers 5.x RoPE config to 4.x.

    The converted checkpoint serializes Qwen's rotary base inside the newer
    ``rope_parameters`` mapping. Maestro currently ships Transformers 4.x,
    whose Qwen3 implementation reads ``config.rope_theta`` instead. Without
    this bridge it silently falls back to 10,000 rather than Music3's trained
    value of 1,000,000, corrupting the autoregressive music-token trajectory.
    """

    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        return config
    rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        return config
    config.rope_theta = float(rope_theta)
    return config


def clean_music_caption(caption: str) -> str:
    """Normalize accepted Markdown without changing the checkpoint template."""

    def _rewrite_special_tag(match: re.Match) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, str(caption or ""))
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text).strip()


def normalize_music3_lyrics(lyrics: str) -> str:
    """Put Music3 section tags on their checkpoint-required own lines."""

    output = []
    for line in str(lyrics or "").split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text.strip()}"


def build_music3_prompt(caption: str, lyrics: str) -> str:
    """Assemble the exact special-token contract used by Music3."""

    return (
        f"{_IM_START}{_CAPTION_START}{clean_music_caption(caption)}{_CAPTION_END}"
        f"{_LYRICS_START}{normalize_music3_lyrics(lyrics)}{_LYRICS_END}"
        f"{_IM_END}{_AUDIO_START}"
    )


def music3_chunk_starts(num_frames: int) -> list[int]:
    num_frames = max(1, int(num_frames))
    if num_frames <= _CHUNK_FRAMES:
        return [0]
    return list(range(0, num_frames - _CHUNK_HOP, _CHUNK_HOP))


def _sample_top_k(
    logits: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(_AR_SAMPLING_TOP_K, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sample_device = generator.device if generator is not None else probs.device
    return torch.multinomial(
        probs.to(sample_device), 1, generator=generator
    ).squeeze(-1).to(probs.device)


class MiniMaxMusic3LanguageModelRunner(nn.Module):
    """Top-level MMGP hook around Qwen's base model.

    Calling ``Qwen3ForCausalLM.model`` directly avoids materializing logits
    for every token in a long caption, but MMGP hooks top-level forwards. This
    wrapper gives us both: one hookable module and only the final hidden state.
    """

    def __init__(self, language_model: Qwen3ForCausalLM):
        super().__init__()
        self.language_model = language_model
        self._compile_me = False

    @property
    def config(self):
        return self.language_model.config

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.embed_tokens(token_ids)

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.lm_head(hidden_states)

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Music3 language generation requires input ids or embeddings.")
            inputs_embeds = self.embed_tokens(input_ids)
        output = self.language_model.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        return output.last_hidden_state[:, -1], output.past_key_values


class MiniMaxMusic3DepthRunner(nn.Module):
    """Hookable local-LM pass that generates all seven residual codebooks."""

    def __init__(self, decoder: MiniMaxMusic3RVQDepthDecoder):
        super().__init__()
        self.decoder = decoder
        self._compile_me = False

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_residual_codes(self, codes: torch.Tensor) -> torch.Tensor:
        offsets = (
            torch.arange(
                self.decoder.config.num_codebooks - 1,
                device=codes.device,
            )
            * self.decoder.config.audio_vocab_size
        ).unsqueeze(0)
        return self.decoder.audio_embeddings(codes + offsets)

    def forward(
        self,
        last_hidden: torch.Tensor,
        semantic_code: torch.Tensor,
        semantic_embed: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = [self.decoder.projection(last_hidden).unsqueeze(1)]
        sequence.append(self.decoder.projection(semantic_embed).unsqueeze(1))
        codes = [semantic_code]
        hidden_parts = []
        for index in range(1, self.decoder.config.num_codebooks):
            hidden = self.decoder(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = self.decoder.audio_heads[index - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            logits = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
            code = _sample_top_k(logits, generator).repeat(2)
            codes.append(code)
            if index < self.decoder.config.num_codebooks - 1:
                embed = self.decoder.audio_embeddings(
                    code + (index - 1) * self.decoder.config.audio_vocab_size
                )
                sequence.append(self.decoder.projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)


class MiniMaxMusic3Pipeline:
    frame_rate = 25
    sampling_rate = 44100
    latent_hop_length = 512

    def __init__(
        self,
        *,
        tokenizer: Qwen2TokenizerFast,
        language_model: MiniMaxMusic3LanguageModelRunner,
        rvq_depth_decoder: MiniMaxMusic3DepthRunner,
        condition_encoder: MiniMaxMusic3ConditionEncoder,
        transformer: MiniMaxMusic3Transformer1DModel,
        vocoder: MiniMaxMusic3Vocoder,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.rvq_depth_decoder = rvq_depth_decoder
        self.condition_encoder = condition_encoder
        self.transformer = transformer
        self.vocoder = vocoder
        self.scheduler = scheduler
        self._interrupt = False
        self._early_stop = False

    @classmethod
    def from_pretrained(cls, model_root: str | Path, dtype=torch.bfloat16):
        root = Path(model_root)
        missing = [
            name
            for name in (
                "tokenizer",
                "language_model",
                "rvq_depth_decoder",
                "condition_encoder",
                "transformer",
                "vocoder",
            )
            if not (root / name).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"MiniMax-Music3 is missing component folders: {', '.join(missing)}. "
                "Run the generation again to resume the official model download."
            )

        # The official converted checkpoint publishes a consolidated
        # ``tokenizer.json`` (plus its config), not the separate
        # ``vocab.json``/``merges.txt`` files required by the slow
        # Qwen2Tokenizer in Maestro's Transformers 4.x runtime.
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            root / "tokenizer", local_files_only=True
        )
        attention_backend = _music3_attention_backend()
        print(f"[MiniMax Music3] Qwen attention backend: {attention_backend}")
        qwen_config = normalize_music3_qwen_config(
            Qwen3Config.from_pretrained(
                root / "language_model",
                local_files_only=True,
            )
        )
        print(
            "[MiniMax Music3] Qwen rotary base: "
            f"{qwen_config.rope_theta:,.0f} (checkpoint value)."
        )
        qwen = Qwen3ForCausalLM.from_pretrained(
            root / "language_model",
            config=qwen_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=attention_backend,
        ).eval()
        depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
            root,
            subfolder="rvq_depth_decoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        condition_encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(
            root,
            subfolder="condition_encoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        transformer = MiniMaxMusic3Transformer1DModel.from_pretrained(
            root,
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        vocoder = MiniMaxMusic3Vocoder.from_pretrained(
            root,
            subfolder="vocoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1,
            shift=1.0,
            invert_sigmas=True,
        )
        return cls(
            tokenizer=tokenizer,
            language_model=MiniMaxMusic3LanguageModelRunner(qwen),
            rvq_depth_decoder=MiniMaxMusic3DepthRunner(depth),
            condition_encoder=condition_encoder,
            transformer=transformer,
            vocoder=vocoder,
            scheduler=scheduler,
        )

    def request_early_stop(self):
        self._early_stop = True

    def _stopped(self) -> bool:
        return bool(self._interrupt)

    @staticmethod
    def _execution_device() -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _release_stage(offloadobj=None):
        if offloadobj is not None and hasattr(offloadobj, "unload_all"):
            offloadobj.unload_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _set_status(set_progress_status, text: str):
        if callable(set_progress_status):
            set_progress_status(text)

    def _embed_audio_frame(self, frame_codes: torch.Tensor) -> torch.Tensor:
        embeds = self.language_model.embed_tokens(
            frame_codes[:, :1] + _AUDIO_CODE_OFFSET
        )
        extra = self.rvq_depth_decoder.embed_residual_codes(
            frame_codes[:, 1:]
        ).sum(dim=1, keepdim=True)
        return (embeds + extra.to(embeds.dtype)) * math.pow(
            self.rvq_depth_decoder.decoder.config.num_codebooks, -0.5
        )

    @torch.no_grad()
    def _generate_semantic_hiddens(
        self,
        *,
        caption: str,
        lyrics: str,
        duration_seconds: float,
        generator: torch.Generator,
        set_progress_status=None,
    ) -> Optional[torch.Tensor]:
        text = build_music3_prompt(caption, lyrics)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > _MAX_PROMPT_TOKENS:
            raise ValueError(
                f"The assembled MiniMax-Music3 prompt has {input_ids.shape[1]} tokens; "
                f"the maximum is {_MAX_PROMPT_TOKENS}."
            )
        unconditional_ids = input_ids.clone()
        unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
        text_ids = torch.cat((input_ids, unconditional_ids), dim=0).to(
            self._execution_device()
        )

        max_frames = min(
            max(1, int(duration_seconds * self.frame_rate)),
            _MAX_AUDIO_FRAMES,
        )
        max_cache_len = input_ids.shape[1] + max_frames + 1
        estimated_cache_gib = estimate_music3_kv_cache_bytes(
            self.language_model.config,
            prompt_tokens=input_ids.shape[1],
            duration_seconds=duration_seconds,
        ) / (1024**3)
        past_key_values = Music3PreallocatedCache(
            self.language_model.config,
            max_cache_len=max_cache_len,
        )
        print(
            "[MiniMax Music3 Memory] Preallocated non-growing Qwen KV cache "
            f"({estimated_cache_gib:.2f} GB for {duration_seconds:g}s)."
        )
        last_hidden, past_key_values = self.language_model(
            input_ids=text_ids,
            past_key_values=past_key_values,
        )
        # The long prompt prefill can leave several GB of now-unused, oddly
        # sized SDPA/FlashAttention workspaces in CUDA's caching allocator.
        # Return those blocks before the 25 Hz autoregressive loop begins.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_gib = torch.cuda.memory_allocated() / (1024**3)
            reserved_gib = torch.cuda.memory_reserved() / (1024**3)
            print(
                "[MiniMax Music3 Memory] Semantic stage after prompt prefill: "
                f"{allocated_gib:.2f} GB allocated / "
                f"{reserved_gib:.2f} GB reserved."
            )
        vocab_mask = torch.ones(
            self.language_model.config.vocab_size,
            dtype=torch.bool,
            device=last_hidden.device,
        )
        vocab_mask[
            _AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE
        ] = False
        vocab_mask[_AUDIO_END_TOKEN_ID] = False

        frame_hiddens = []
        last_reported_second = -1
        started_at = time.monotonic()
        for frame_index in range(max_frames + 1):
            if self._stopped():
                return None
            logits = self.language_model.project_logits(last_hidden).float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
            threshold = torch.topk(
                conditional, _AR_CFG_TOP_K, dim=-1
            ).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            sampled = _sample_top_k(guided, generator)
            if int(sampled.item()) == _AUDIO_END_TOKEN_ID:
                break

            semantic_code = sampled - _AUDIO_CODE_OFFSET
            semantic_pair = semantic_code.repeat(2)
            semantic_embed = self.language_model.embed_tokens(
                semantic_pair + _AUDIO_CODE_OFFSET
            )
            frame_codes, depth_hidden = self.rvq_depth_decoder(
                last_hidden,
                semantic_pair,
                semantic_embed,
                generator,
            )
            if frame_index > 0:
                frame_hiddens.append(
                    torch.cat((last_hidden[:1], depth_hidden), dim=-1).cpu()
                )
                current_second = len(frame_hiddens) // self.frame_rate
                if current_second != last_reported_second:
                    elapsed = max(0.1, time.monotonic() - started_at)
                    speed = len(frame_hiddens) / elapsed
                    self._set_status(
                        set_progress_status,
                        "Planning song structure "
                        f"({current_second}s/{duration_seconds:g}s, {speed:.1f} frames/s)",
                    )
                    if torch.cuda.is_available() and current_second % 30 == 0:
                        print(
                            "[MiniMax Music3 Memory] Semantic stage at "
                            f"{current_second}s: "
                            f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB "
                            "allocated / "
                            f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB "
                            "reserved."
                        )
                    last_reported_second = current_second
                if len(frame_hiddens) >= max_frames or self._early_stop:
                    break
            feedback = self._embed_audio_frame(frame_codes)
            last_hidden, past_key_values = self.language_model(
                inputs_embeds=feedback,
                past_key_values=past_key_values,
            )

        del past_key_values, last_hidden
        if not frame_hiddens:
            raise ValueError(
                "MiniMax-Music3 generated zero audio frames; try a more detailed "
                "music caption or a different seed."
            )
        return torch.stack(frame_hiddens, dim=1)

    @torch.no_grad()
    def _denoise_chunks(
        self,
        *,
        frame_hiddens: torch.Tensor,
        num_inference_steps: int,
        generator: torch.Generator,
        callback=None,
        set_progress_status=None,
        offloadobj=None,
    ) -> Optional[list[torch.Tensor]]:
        starts = music3_chunk_starts(frame_hiddens.shape[1])
        total_steps = len(starts) * num_inference_steps
        latent_chunks = []
        previous_latent = None
        previous_condition = None
        device = self._execution_device()
        completed_steps = 0

        for chunk_index, chunk_start in enumerate(starts):
            if self._stopped():
                return None
            chunk_end = min(chunk_start + _CHUNK_FRAMES, frame_hiddens.shape[1])
            self._set_status(
                set_progress_status,
                f"Encoding music chunk {chunk_index + 1}/{len(starts)}",
            )
            condition = self.condition_encoder(
                frame_hiddens[:, chunk_start:chunk_end].to(device)
            ).to(self.transformer.dtype)
            overlap = 0
            if previous_latent is not None:
                overlap = min(previous_latent.shape[-1], condition.shape[1])
                condition[:, :overlap] = previous_condition[:, :overlap].to(
                    condition.device, condition.dtype
                )

            latents = torch.randn(
                (1, self.transformer.config.in_channels, condition.shape[1]),
                generator=generator,
                device=device,
                dtype=condition.dtype,
            )
            noise_prompt = (
                latents[..., :overlap].clone() if overlap > 0 else None
            )
            sigmas = np.linspace(
                1.0,
                1.0 / num_inference_steps,
                num_inference_steps,
            ).tolist()
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
            timesteps = self.scheduler.timesteps

            for step_index, timestep_value in enumerate(timesteps):
                if self._stopped():
                    return None
                if overlap > 0:
                    time_value = timestep_value.to(latents.dtype)
                    latents[..., :overlap] = (
                        1.0 - (1.0 - 1e-6) * time_value
                    ) * noise_prompt + time_value * previous_latent[..., :overlap].to(
                        latents.device, latents.dtype
                    )

                timestep = timestep_value.expand(2).to(latents.dtype)
                model_latents = torch.cat((latents, latents), dim=0)
                model_condition = torch.cat(
                    (condition, torch.zeros_like(condition)), dim=0
                )
                prediction = self.transformer(
                    hidden_states=model_latents,
                    timestep=timestep,
                    encoder_hidden_states=model_condition,
                    return_dict=False,
                )[0]
                conditional, unconditional = prediction.chunk(2, dim=0)
                velocity = unconditional + (
                    conditional - unconditional
                ) * _FLOW_CFG_SCALE
                latents = self.scheduler.step(
                    velocity,
                    timestep_value,
                    latents,
                    return_dict=False,
                )[0]

                completed_steps += 1
                if callable(callback):
                    callback(
                        step_idx=completed_steps - 1,
                        override_num_inference_steps=total_steps,
                        total_steps_hint=total_steps,
                        denoising_extra=(
                            f"Music chunk {chunk_index + 1}/{len(starts)}"
                        ),
                    )

            if overlap > 0:
                latents[..., :overlap] = previous_latent[..., :overlap].to(
                    latents.device, latents.dtype
                )
            overlap_start = max(
                0, latents.shape[-1] - 2 * _OVERLAP_LATENT_LENGTH
            )
            overlap_end = max(
                overlap_start,
                latents.shape[-1] - _OVERLAP_LATENT_LENGTH,
            )
            previous_latent = latents[
                ..., overlap_start:overlap_end
            ].detach().cpu()
            previous_condition = condition[
                :, overlap_start:overlap_end
            ].detach().cpu()
            latent_chunks.append(latents.detach().cpu())
            del condition, latents, model_condition, model_latents, prediction

        return latent_chunks

    @torch.no_grad()
    def _decode_chunks(
        self,
        latent_chunks: list[torch.Tensor],
        *,
        duration_seconds: float,
        set_progress_status=None,
    ) -> torch.Tensor:
        waveform_chunks = []
        device = self._execution_device()
        for chunk_index, latents in enumerate(latent_chunks):
            self._set_status(
                set_progress_status,
                f"Decoding stereo audio {chunk_index + 1}/{len(latent_chunks)}",
            )
            waveform = self.vocoder(
                latents.to(device=device, dtype=self.vocoder.dtype)
            ).float()
            left = 0 if chunk_index == 0 else _CROP_LEFT_LATENT * self.latent_hop_length
            right = (
                0
                if chunk_index == len(latent_chunks) - 1
                else _CROP_RIGHT_LATENT * self.latent_hop_length
            )
            stop = waveform.shape[-1] - right if right > 0 else waveform.shape[-1]
            waveform_chunks.append(waveform[..., left:stop].cpu())
        audio = torch.cat(waveform_chunks, dim=-1).clamp(-1.0, 1.0)
        max_samples = int(round(duration_seconds * self.sampling_rate))
        return audio[..., :max_samples]

    @torch.no_grad()
    def generate(
        self,
        input_prompt: str,
        model_mode=None,
        audio_guide=None,
        *,
        alt_prompt: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        sampling_steps: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        callback=None,
        set_progress_status=None,
        offloadobj=None,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False
        lyrics = str(input_prompt or "").strip()
        caption = str(alt_prompt or "").strip()
        if not lyrics:
            raise ValueError(
                "Lyrics cannot be empty for MiniMax-Music3. Use [Instrumental] "
                "for an instrumental track."
            )
        if not caption:
            raise ValueError(
                "Music Caption cannot be empty for MiniMax-Music3. Describe the "
                "genre, vocals, instrumentation, arrangement, and production."
            )
        if audio_guide is not None:
            raise ValueError("MiniMax-Music3 does not support reference audio.")

        try:
            duration = float(duration_seconds or 60.0)
        except (TypeError, ValueError):
            duration = 60.0
        duration = min(300.0, max(1.0, duration))
        steps = sampling_steps if sampling_steps is not None else num_inference_steps
        try:
            steps = int(steps or 30)
        except (TypeError, ValueError):
            steps = 30
        steps = min(100, max(1, steps))
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            seed = int(torch.seed() % (2**31 - 1))
        generator = torch.Generator(device=self._execution_device()).manual_seed(seed)

        self._set_status(set_progress_status, "Encoding MiniMax-Music3 prompt")
        frame_hiddens = self._generate_semantic_hiddens(
            caption=caption,
            lyrics=lyrics,
            duration_seconds=duration,
            generator=generator,
            set_progress_status=set_progress_status,
        )
        if frame_hiddens is None or self._stopped():
            return None
        self._release_stage(offloadobj)

        latent_chunks = self._denoise_chunks(
            frame_hiddens=frame_hiddens,
            num_inference_steps=steps,
            generator=generator,
            callback=callback,
            set_progress_status=set_progress_status,
            offloadobj=offloadobj,
        )
        if latent_chunks is None or self._stopped():
            return None
        self._release_stage(offloadobj)

        audio = self._decode_chunks(
            latent_chunks,
            duration_seconds=duration,
            set_progress_status=set_progress_status,
        )
        return {
            "x": audio,
            "audio_sampling_rate": self.sampling_rate,
            "overridden_inputs": {
                "duration_seconds": duration,
                "num_inference_steps": steps,
            },
        }
