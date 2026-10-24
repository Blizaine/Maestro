# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
import collections
import gc
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from safetensors.torch import load_file
from shared.utils.text_encoder_cache import TextEncoderCache
from tokenizers import Tokenizer
from tqdm import tqdm

from .backbone import GenerationCancelled
from .duration import LONGFORM_CHUNK_MAX_SECONDS, duration_frame_budget
from .fish_audio_handler import DIALOGUE_CUT_LINES, DIALOGUE_HISTORY_SECONDS
from .loading import load_codec, load_transformers
from .progress import PhaseProgress, generation_progress, set_phase_status
from .prompt import (
    DRAMA3_MAX_PROMPT_ROWS,
    build_drama3_prompt,
    check_paired_tags,
    drama3_text_chunks,
    fit_history_to_prompt_rows,
    reference_transcripts,
    speaker_lines_to_tags,
    validate_drama3_text,
)
from .runtime_support import (
    can_place_gpu_experts,
    cuda_graphs,
    fish_vllm_availability,
    runtime,
)

# Prompt rows processed together so each routed expert is read once per chunk,
# rather than reread for every prompt token.
PREFILL_ROWS = 512
SILENCE_SECONDS, SILENCE_SHARE, SILENCE_LOGIT = 8.0, 0.8, 2.0
CODEC_CONTEXT_FRAMES = 1152
CODEC_DECODE_TILE_FRAMES = 128


class FishAudioPipeline:
    sample_rate = 44100
    frame_length = 2048

    def __init__(self, backbone_path, codebooks_path, codec_path, tokenizer_path, lm_decoder_engine="legacy", profile=0):
        if lm_decoder_engine not in ("legacy", "vllm"):
            raise ValueError("Fish Audio Drama 3 supports the Legacy engine and a capability-gated vLLM engine")
        if lm_decoder_engine == "vllm":
            supported, reason = fish_vllm_availability()
            if not supported:
                raise RuntimeError(f"Fish Audio vLLM engine is unavailable: {reason}")
        self.engine = lm_decoder_engine
        self.profile = int(profile)
        self._gpu_experts_checked = False
        self._gpu_experts_resident = False
        self.config, self.transformer, self.transformer2 = load_transformers(backbone_path, codebooks_path, self.engine)
        self.vae = load_codec(codec_path)
        self.silence_tables, self.silence_offsets, self.silence_limits, self.silence_probe = self._silence_classifier()
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._interrupt = self._early_stop = False
        self.transformer.abort_requested = self.transformer2.abort_requested = self._abort_requested
        self.vae.get_VAE_tile_size = self.get_vae_tile_size
        self.last_codes = None
        self.last_finish_reason = None
        # What a voice reference gives, by file (path, size, modification time): its encoded audio codes, and the Whisper transcript
        # of a reference given without one.
        self.reference_cache = TextEncoderCache()
        print(f"[Fish Audio] Drama 3 - {self.engine}; CPU expert kernels remain available as a fallback")
        self.last_duration_budget_frames = None
        self.last_duration_is_auto = False
        self.last_duration_context_limited = False
        self.last_duration_context_limit_frames = None
        self.last_longform_report = None
        self._frame_graph = None
        self._tower_graph = None
        self._decode_device = None
        self.last_decode_backend = "eager"
        self.last_graph_capture_seconds = 0.0
        self.last_expert_cache_report = None

    def _silence_classifier(self):
        """The first layer of the silent-frame classifier as tables per codebook entry: the codec latent of a frame is the sum of its codes'
        projected codebook entries, so the layer applied to a frame (for its own slot and as one of the 2 previous frames) is a sum of rows."""
        probe_path = Path(__file__).with_name("silence_probe.safetensors")
        if not probe_path.is_file():
            # This auxiliary classifier is not part of the published Drama 3
            # asset set. EOS and the user's duration limit remain authoritative.
            print("[Fish Audio] Optional silence probe is unavailable; generation will stop at EOS or its duration limit.")
            return None, None, None, None
        probe = load_file(str(probe_path))
        hidden = probe["b1"].numel()
        weights = probe["w1"].view(hidden, 3, -1).transpose(0, 1).reshape(3 * hidden, -1)  # rows: the frame, the frame before, the one before
        quantizers = [*self.vae.quantizer.semantic_quantizer.quantizers, *self.vae.quantizer.quantizer.quantizers]
        with torch.no_grad():
            tables = [(vq.codebook.weight.float() @ vq.out_proj.weight[:, :, 0].float().T + vq.out_proj.bias.float()) @ weights.T for vq in quantizers]
        sizes = [len(table) for table in tables]
        return torch.cat(tables).numpy(), np.cumsum([0] + sizes[:-1]), np.array(sizes) - 1, (probe["b1"].numpy(), probe["w2"].numpy(), float(probe["b2"]))

    @staticmethod
    def get_vae_tile_size(vae_config, device_mem_capacity, mixed_precision):
        if vae_config == 1:
            return 0
        return 128 if vae_config == 3 or (vae_config == 0 and device_mem_capacity < 12000) else 512

    def _abort_requested(self):
        return self._interrupt

    def _check_abort(self, *args):
        if self._interrupt:
            raise GenerationCancelled()

    def request_early_stop(self):
        self._early_stop = True

    def _release_decoder_caches(self):
        device = getattr(self, "_decode_device", None)
        if torch.cuda.is_initialized() and device is not None and device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()  # a decoding step queued ahead may still run (and its CPU expert jobs)
        frame_graph = getattr(self, "_frame_graph", None)
        if frame_graph is not None:
            frame_graph.close()
            self._frame_graph = None
        tower_graph = getattr(self, "_tower_graph", None)
        if tower_graph is not None:
            tower_graph.close()
            self._tower_graph = None
        cuda_graphs.invalidate()
        self.transformer.release_cache()
        self.transformer2.release_cache()
        # Small-GPU slots and staging are decode-only. Release them before the
        # waveform codec uses its own weights/activation reserve.
        for bank in self._expert_banks():
            bank.release_decode_cache()

    def _expert_banks(self):
        from .expert_formats import FishSwiGLUExperts
        if not isinstance(self.transformer, torch.nn.Module):
            return []
        return [module for module in self.transformer.modules() if isinstance(module, FishSwiGLUExperts)]

    def _prepare_partial_expert_cache(self, device):
        self.last_expert_cache_report = None
        if (getattr(self, "engine", "legacy") != "vllm"
                or getattr(self, "_gpu_experts_resident", False)
                or os.environ.get("MAESTRO_FISH_EXPERT_CACHE", "1") == "0"):
            return
        from .expert_cache import cache_slot_count
        banks = self._expert_banks()
        if not banks or torch.device(device).type != "cuda":
            return
        if not all(bank.supports_gpu_device(device) for bank in banks):
            print("[Fish Audio] Packed expert format is unsupported by this GPU; retaining CPU experts")
            return
        topk = self.config.text.num_experts_per_tok
        free, total = torch.cuda.mem_get_info(device)
        slots = cache_slot_count([bank.expert_nbytes for bank in banks],
                                 [bank.num_experts for bank in banks], free, total, topk,
                                 host_free_bytes=psutil.virtual_memory().available)
        if not slots:
            print("[Fish Audio] Expert cache has insufficient GPU or staging-RAM headroom; retaining CPU experts")
            return
        try:
            for bank in banks:
                bank.place_cached_experts(slots, topk, device)
        except (RuntimeError, ValueError) as error:
            for bank in banks:
                bank.release_decode_cache()
            torch.cuda.empty_cache()
            print(f"[Fish Audio] GPU expert cache unavailable; retaining CPU experts: {str(error)[:160]}")
            return
        print(f"[Fish Audio] Packed expert GPU cache: {slots}/{banks[0].num_experts} slots per bank; CPU routing outside capture")

    def _reference_transcripts(self, guides, alt_prompt):
        """The transcript of each voice reference: given in alt_prompt, else made by the configured Whisper model, loaded once for
        all the references without one and released right after them."""
        texts = reference_transcripts(alt_prompt, len(guides))
        try:
            return [text or self._transcribe(guide, index) for index, (guide, text) in enumerate(zip(guides, texts), 1)]
        finally:
            try:
                from services.audio_analysis import unload_whisper

                unload_whisper()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _file_key(path):
        stat = os.stat(path)
        return os.path.abspath(path), stat.st_size, stat.st_mtime_ns

    def _transcribe(self, path, index):
        """Transcribe one reference through Maestro's optional faster-whisper service."""
        def transcribe(paths):
            set_phase_status(f"Transcribing Voice Reference {index}")
            self._check_abort()
            from services.audio_analysis import _transcribe

            segments = _transcribe(paths[0])
            self._check_abort()
            return [" ".join(segment.text for segment in segments).strip()]
        text = self.reference_cache.encode(transcribe, [path], cache_keys=[("transcript", "faster-whisper-small", *self._file_key(path))])[0]
        print(f"[Fish Audio] Whisper Transcript of Voice Reference {index}: {text}")
        validate_drama3_text(text, f"Whisper transcript of voice reference {index} (type its transcript instead)")
        return text

    def _reference_codes(self, path, label):
        def encode(paths):
            set_phase_status(f"Encoding Reference Audio{label}")
            audio, sample_rate = sf.read(paths[0], dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio.mean(axis=1).copy())[None, None]
            if sample_rate != self.sample_rate:
                waveform = audio_functional.resample(waveform, sample_rate, self.sample_rate)
            waveform = waveform.to(device=runtime.execution_device(self.vae), dtype=self.vae._model_dtype)
            return [self.vae(waveform, encode=True)[0].cpu()]
        return self.reference_cache.encode(encode, [path], cache_keys=[("codes", *self._file_key(path))])[0]

    def _place_vllm_gpu_experts(self, device):
        """Stage all expert banks only when the selected profile leaves room for caches."""

        if self._gpu_experts_checked:
            return self._gpu_experts_resident
        self._gpu_experts_checked = True
        if self.engine != "vllm":
            return False
        from .expert_formats import FishSwiGLUExperts

        banks = [module for module in self.transformer.modules() if isinstance(module, FishSwiGLUExperts)]
        if not banks:
            print("[Fish Audio] vLLM expert staging unavailable; no routed expert banks were found")
            return False
        if torch.device(device).type != "cuda":
            print("[Fish Audio] vLLM layer kernels selected; experts remain on CPU because the model is not on CUDA")
            return False

        if not all(bank.supports_gpu_device(device) for bank in banks):
            print("[Fish Audio] vLLM layer kernels selected; expert format is unsupported by this GPU, retaining CPU experts")
            return False

        required = sum(bank.expert_nbytes for bank in banks)
        free, total = torch.cuda.mem_get_info(torch.device(device))
        reserve = max(4 * 1024**3, int(total * 0.15))
        if not can_place_gpu_experts(required, free, total, self.profile):
            print(
                "[Fish Audio] vLLM layer kernels selected; experts remain on CPU "
                f"(full bank {required / 1024**3:.1f} GiB, free {free / 1024**3:.1f} GiB, "
                f"reserve {reserve / 1024**3:.1f} GiB, profile {self.profile})"
            )
            return False

        try:
            for bank in banks:
                bank.place_gpu_experts(torch.device(device))
            torch.cuda.synchronize(torch.device(device))
        except torch.cuda.OutOfMemoryError:
            for bank in banks:
                bank.release_gpu_experts()
            gc.collect()
            torch.cuda.empty_cache()
            print("[Fish Audio] GPU expert staging hit the VRAM limit; keeping the banks on CPU")
            return False

        self._gpu_experts_resident = True
        print(f"[Fish Audio] vLLM Triton experts resident on GPU ({required / 1024**3:.1f} GiB; {reserve / 1024**3:.1f} GiB reserved)")
        return True

    def _generate_codes(self, prompt, max_frames, temperature, top_k, top_p, callback, offset=0, total=None, part=""):
        """Up to max_frames frames for one prompt; progress counts from offset frames out of total (the whole generation)."""
        backbone, tower = self.transformer, self.transformer2
        total = max_frames if total is None else total
        prompt_rows = len(prompt)
        capacity = prompt_rows + max_frames
        if capacity > self.config.text.max_position_embeddings:
            raise ValueError("The reference, text and maximum duration exceed Drama 3's context capacity")
        device = runtime.execution_device(backbone)
        self._place_vllm_gpu_experts(device)
        backbone.setup_caches(capacity)
        tower.setup_caches(capacity)
        frames = torch.zeros((capacity, prompt.shape[1]), dtype=torch.long, device=device)
        self._decode_device = frames.device
        frames[:prompt_rows] = prompt.to(device)
        positions = torch.arange(capacity, device=device)
        chunk_size = PREFILL_ROWS
        chunks = math.ceil(prompt_rows / chunk_size)
        total_layers = chunks * len(backbone.model.layers) + math.ceil((prompt_rows - 1) / chunk_size) * (len(tower.shared_layers) + len(tower.residual_layers))
        encoding = f"Encoding Text Prompt{part.replace(' - Part', '')}"
        set_phase_status(encoding)
        with cuda_graphs.scope(enabled=False), ExitStack() as stack:
            progress = stack.enter_context(PhaseProgress(total_layers, encoding, "layers", next_status="Generating Speech" + part))
            bar = stack.enter_context(tqdm(total=total_layers, desc=encoding, unit="layer", mininterval=1 / 3, leave=False))
            def completed(*args):
                self._check_abort()
                bar.update()
                progress.advance()
            for layer in (*backbone.model.layers, *tower.shared_layers, *tower.residual_layers):
                stack.callback(layer.register_forward_hook(completed).remove)
            for start in range(0, prompt_rows, chunk_size):
                end = min(start + chunk_size, prompt_rows)
                chunk = frames[start:end]
                embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
                hidden = backbone(embeddings[None], positions[start:end], start)[0]
                paired_end = min(end, prompt_rows - 1)
                if paired_end > start:
                    tower.prefill(hidden[:paired_end - start], positions[start:paired_end], frames[start:paired_end], frames[start + 1:paired_end + 1], paired_end)
        hidden = hidden[-1:]
        logits = backbone.compute_logits(hidden)
        self._prepare_partial_expert_cache(device)
        from .decode_graph import WholeFrameDecoder

        self.last_decode_backend = "eager"
        self.last_graph_capture_seconds = 0.0
        set_phase_status("Preparing Speech Decoder" + part)
        self._frame_graph, graph_reason = WholeFrameDecoder.try_create(
            backbone, tower, hidden, frames[prompt_rows - 1:prompt_rows],
            frames[:prompt_rows, 0], prompt_rows - 1, temperature, top_k, top_p,
            engine=getattr(self, "engine", "legacy"), profile=getattr(self, "profile", 0),
            check_abort=self._check_abort,
        )
        if self._frame_graph is not None:
            self._frame_graph.validate()
            self.last_decode_backend = "cuda_graph"
            self.last_graph_capture_seconds = self._frame_graph.capture_seconds
            print(f"[Fish Audio] Whole-frame CUDA graph captured in {self.last_graph_capture_seconds:.2f}s (capacity {capacity} rows){part}")
        elif graph_reason != getattr(self, "_graph_fallback_reason", None):
            print(f"[Fish Audio] Eager speech decoder: {graph_reason}")
            self._graph_fallback_reason = graph_reason
        self._tower_graph = None
        if self._frame_graph is None:
            from .tower_decode_graph import TowerFrameDecoder
            self._tower_graph, tower_reason = TowerFrameDecoder.try_create(
                backbone, tower, hidden, logits, frames[prompt_rows - 1:prompt_rows],
                frames[:prompt_rows, 0], prompt_rows - 1, temperature, top_k, top_p,
                engine=getattr(self, "engine", "legacy"), check_abort=self._check_abort,
            )
            if self._tower_graph is not None:
                self._tower_graph.validate()
                self.last_decode_backend = "tower_cuda_graph"
                self.last_graph_capture_seconds = self._tower_graph.capture_seconds
                print(f"[Fish Audio] Tower-only CUDA graph captured in {self.last_graph_capture_seconds:.2f}s{part}")
            elif tower_reason != getattr(self, "_tower_fallback_reason", None):
                print(f"[Fish Audio] Eager audio tower: {tower_reason}")
                self._tower_fallback_reason = tower_reason
        set_phase_status("Generating Speech" + part)
        produced, finish_reason = 0, "length"
        if callback is not None:
            callback(step_idx=offset - 1, override_num_inference_steps=total, denoising_extra="Generating Speech" + part, progress_unit="frames")
        updated = 0.0
        silence_window, quiet, silence_layers = round(SILENCE_SECONDS * self.sample_rate / self.frame_length), [], collections.deque(maxlen=3)
        if self.silence_probe is not None:
            silence_bias, silence_weights, silence_offset = self.silence_probe
            silence_width = len(silence_bias)
        frame_done = torch.cuda.Event(blocking=True) if device.type == "cuda" else None
        status = torch.empty(
            frames.shape[1] + 1,
            dtype=torch.long,
            device="cpu",
            pin_memory=device.type == "cuda" and torch.cuda.is_available(),
        )  # the sampled frame and its finiteness
        started = time.perf_counter()
        with tqdm(total=max_frames, desc="Generating Speech" + part, unit="frame", mininterval=1 / 3) as progress, cuda_graphs.scope(enabled=False):
            for step in range(max_frames):
                self._check_abort()
                if self._early_stop:
                    finish_reason = "early_stop"
                    break
                position = prompt_rows - 1 + step
                if self._frame_graph is not None:
                    sampled, finite = self._frame_graph.step(position)
                elif self._tower_graph is not None:
                    sampled, finite, extra, semantic = self._tower_graph.step(
                        position, hidden, logits, frames[position:position + 1],
                        frames[:position + 1, 0])
                    del logits, hidden
                else:
                    sampled, finite, extra, semantic = tower.decode_step(logits, hidden, positions[position:position + 1], frames[position:position + 1], frames[:position + 1, 0],
                        end=position + 1, temperature=temperature, top_k=top_k, top_p=top_p)
                    del logits, hidden
                status.copy_(torch.cat((sampled[0], finite.to(sampled.dtype).view(1))), non_blocking=True)  # read after frame_done
                frames[position + 1] = sampled[0]  # kept only if the frame is produced (counted below)
                if frame_done is not None:
                    frame_done.record(torch.cuda.current_stream(device))
                if self._frame_graph is None and step + 1 < max_frames:
                    hidden, logits = backbone.decode(sampled[:, 0], extra, semantic, positions[position + 1:position + 2], position + 1)
                if self._frame_graph is None:
                    del extra, semantic
                del sampled, finite
                # The frame's one host sync: a blocking event, whose wait sleeps instead of spinning beside the CPU experts.
                if frame_done is not None:
                    frame_done.synchronize()
                values = status.tolist()
                main, finite = values[0], values[-1]
                if not finite:
                    raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
                if main == self.config.eos_token_id:
                    finish_reason = "eos"
                    break
                produced += 1
                # The classifier's first layer for this frame, then the frame and the 2 before it (the first frame repeated at the start)
                if self.silence_probe is not None:
                    silence_layers.append(self.silence_tables[np.minimum(values[1:-1], self.silence_limits) + self.silence_offsets].sum(0))
                    frame, previous, before = (silence_layers[max(-index, -len(silence_layers))] for index in (1, 2, 3))
                    silence_hidden = silence_bias + frame[:silence_width] + previous[silence_width:2 * silence_width] + before[2 * silence_width:]
                    quiet.append(float(np.maximum(silence_hidden, 0) @ silence_weights) + silence_offset > SILENCE_LOGIT)
                if self.silence_probe is not None and len(quiet) >= silence_window and sum(quiet[-silence_window:]) >= SILENCE_SHARE * silence_window:
                    # Cut after the last frames of speech (mostly not quiet), with a short margin.
                    produced = min(produced, max((index for index in range(10, len(quiet)) if sum(quiet[index - 10:index]) < 5), default=0) + 6)
                    finish_reason = "silence"
                    print(f"[Fish Audio] The speech stopped{part} and stayed silent for {SILENCE_SECONDS:.0f} s: generation stopped and the silence was cut. "
                          "If words are missing, generate it again with another seed.")
                    break
                progress.update()
                now = time.monotonic()
                if callback is not None and now - updated >= 1 / 3:
                    callback(step_idx=offset + produced - 1, override_num_inference_steps=total, progress_unit="frames",
                        denoising_extra=f"Generating Speech{part} - {(offset + produced) * self.frame_length / self.sample_rate:.1f}s")
                    updated = now
        elapsed = time.perf_counter() - started
        cache_rows = [bank._fish_gpu_cache.stats() for bank in self._expert_banks()
                      if bank._fish_gpu_cache is not None]
        if cache_rows:
            self.last_expert_cache_report = {
                "banks": len(cache_rows), "slots_per_bank": cache_rows[0]["slots"],
                **{key: sum(row[key] for row in cache_rows)
                   for key in ("hits", "misses", "transferred_bytes", "gpu_bytes")},
            }
            print(f"[Fish Audio] Expert cache hits {self.last_expert_cache_report['hits']}, misses {self.last_expert_cache_report['misses']}")
        print(f"[Fish Audio] {produced} Frames in {elapsed:.2f}s ({produced / elapsed:.2f} Frames/s); Finish: {finish_reason}")
        return frames[prompt_rows:prompt_rows + produced, 1:].T.contiguous().cpu(), finish_reason

    def _decode(self, codes, tile_size):
        set_phase_status("Decoding Speech")
        frames = codes.shape[-1]
        requested_tile = int(tile_size) if tile_size else CODEC_DECODE_TILE_FRAMES
        tile = max(1, min(requested_tile, CODEC_DECODE_TILE_FRAMES))
        total = math.ceil(frames / tile)
        parts = []
        with PhaseProgress(total, "Decoding Speech", "tiles") as progress, tqdm(total=total, desc="Decoding Speech", unit="tile", mininterval=1 / 3) as bar:
            for start in range(0, frames, tile):
                self._check_abort()
                end = min(start + tile, frames)
                # Preserve the codec's causal attention and convolution context.
                left = max(0, start - CODEC_CONTEXT_FRAMES)
                segment = codes[None, :, left:end].to(runtime.execution_device(self.vae)).clone()
                decoded = self.vae(segment)
                parts.append(decoded[0, :, (start - left) * self.frame_length:].float().cpu())
                del decoded, segment
                bar.update()
                progress.advance()
        return torch.cat(parts, dim=-1)

    def _history(self, turns, seconds):
        """Keep recent whole turns within `seconds`, preserving transcript/audio alignment."""
        frame_budget = max(0, int(seconds * self.sample_rate / self.frame_length))
        kept = []
        for text, codes in reversed([turn for turn in turns if turn[1].shape[-1]]):
            turn_frames = codes.shape[-1]
            if frame_budget <= 0 or turn_frames > frame_budget:
                break
            kept.insert(0, (text, codes))
            frame_budget -= turn_frames
        return kept

    def _build_bounded_prompt(self, text, transcripts, references, turns, history_seconds, anchor_turn=None):
        """Fit bounded, aligned history after preserving references and current text."""
        base = build_drama3_prompt(
            text, self.tokenizer, self.config, transcripts, references, ()
        )
        recent_history = []
        if history_seconds <= 0:
            self._last_prompt_history_context = {
                "recent_turn_count": 0,
                "recent_audio_seconds": 0.0,
                "anchor_included": False,
                "anchor_audio_seconds": 0.0,
                "total_turn_count": 0,
                "total_audio_seconds": 0.0,
                "history_prompt_rows": 0,
            }
            return base
        recent_history = self._history(turns, history_seconds)
        history = list(recent_history)
        if anchor_turn is not None:
            if not any(codes is anchor_turn[1] for _, codes in recent_history):
                history.insert(0, anchor_turn)
        if not history:
            self._last_prompt_history_context = {
                "recent_turn_count": 0,
                "recent_audio_seconds": 0.0,
                "anchor_included": False,
                "anchor_audio_seconds": 0.0,
                "total_turn_count": 0,
                "total_audio_seconds": 0.0,
                "history_prompt_rows": 0,
            }
            return base
        available = DRAMA3_MAX_PROMPT_ROWS - len(base)
        bounded_history = fit_history_to_prompt_rows(history, self.tokenizer, available)
        anchor_included = bool(
            anchor_turn is not None
            and any(codes is anchor_turn[1] for _, codes in bounded_history)
        )
        recent_kept = [
            (turn_text, codes)
            for turn_text, codes in bounded_history
            if any(codes is recent_codes for _, recent_codes in recent_history)
        ]
        def audio_seconds(turns_to_measure):
            frames = sum(int(codes.shape[-1]) for _, codes in turns_to_measure)
            return frames * self.frame_length / self.sample_rate

        anchor_seconds = (
            int(anchor_turn[1].shape[-1]) * self.frame_length / self.sample_rate
            if anchor_included
            else 0.0
        )
        recent_seconds = audio_seconds(recent_kept)
        total_seconds = audio_seconds(bounded_history)
        bounded_prompt = (
            build_drama3_prompt(
                text, self.tokenizer, self.config, transcripts, references, bounded_history
            )
            if bounded_history
            else base
        )
        self._last_prompt_history_context = {
            "recent_turn_count": len(recent_kept),
            "recent_audio_seconds": round(recent_seconds, 3),
            "anchor_included": anchor_included,
            "anchor_audio_seconds": round(anchor_seconds, 3),
            "total_turn_count": len(bounded_history),
            "total_audio_seconds": round(total_seconds, 3),
            "history_prompt_rows": len(bounded_prompt) - len(base),
        }
        return bounded_prompt

    @torch.inference_mode()
    @generation_progress
    def generate(self, input_prompt, *, alt_prompt="", audio_guide=None, audio_guide2=None, audio_guide3=None, audio_prompt_type="", duration_seconds=0,
                 temperature=0.8, top_k=0, top_p=0.95, seed=-1, VAE_tile_size=0, custom_settings=None, callback=None, set_progress_status=None, **kwargs):
        self._interrupt = self._early_stop = False
        self.last_codes = self.last_finish_reason = None
        self.last_duration_budget_frames = None
        self.last_duration_is_auto = False
        self.last_duration_context_limited = False
        self.last_duration_context_limit_frames = None
        self.last_longform_report = None
        if seed >= 0:
            torch.manual_seed(seed)
        settings = custom_settings or {}  # saved settings may predate these options
        report = None
        try:
            duration_budget = duration_frame_budget(
                duration_seconds, input_prompt, self.sample_rate, self.frame_length
            )
            max_frames = duration_budget.frames
            self.last_duration_budget_frames = max_frames
            self.last_duration_is_auto = duration_budget.automatic
            if duration_budget.automatic:
                print(
                    f"[Fish Audio] Auto speech budget: {max_frames} frames "
                    f"({duration_budget.seconds}s from {duration_budget.script_units} "
                    "printable script units; 8s + ceil(units / 4), capped at 3600s)."
                )
            validate_drama3_text(speaker_lines_to_tags(input_prompt), "Speech Script")
            check_paired_tags(input_prompt, "Speech Script", tokenizer=self.tokenizer)
            lines_per_part = int(settings.get(DIALOGUE_CUT_LINES["id"], DIALOGUE_CUT_LINES["default"]))
            if not 0 <= lines_per_part <= DIALOGUE_CUT_LINES["max"]:
                raise ValueError(f"{DIALOGUE_CUT_LINES['id']} must be between 0 and {DIALOGUE_CUT_LINES['max']}")
            chunks = drama3_text_chunks(input_prompt, lines_per_part)
            if not chunks:
                raise ValueError("Speech Script: text must not be empty")
            try:
                history_seconds = float(
                    settings.get(DIALOGUE_HISTORY_SECONDS["id"], DIALOGUE_HISTORY_SECONDS["default"])
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("dialogue_history_seconds must be finite and between 0 and 120") from error
            if not math.isfinite(history_seconds) or not 0 <= history_seconds <= DIALOGUE_HISTORY_SECONDS["max"]:
                raise ValueError("dialogue_history_seconds must be finite and between 0 and 120")

            chunk_reports = [
                {
                    "index": chunk.index,
                    "source_span": {"start": chunk.source_start, "end": chunk.source_end},
                    "source_units": chunk.speech_units,
                    "prompt_rows": None,
                    "context_limit_frames": None,
                    "budget_frames": None,
                    "generated_frames": 0,
                    "finish_reason": None,
                    "cap_reasons": [],
                    "status": "pending",
                    "eos_completed": False,
                    "history_context": {
                        "recent_turn_count": 0,
                        "recent_audio_seconds": 0.0,
                        "anchor_included": False,
                        "anchor_audio_seconds": 0.0,
                        "total_turn_count": 0,
                        "total_audio_seconds": 0.0,
                        "history_prompt_rows": 0,
                    },
                }
                for chunk in chunks
            ]
            report = {
                "version": 1,
                "status": "running",
                "mode": "auto" if duration_budget.automatic else "maximum",
                "duration_limit_seconds": duration_budget.seconds,
                "duration_budget_frames": max_frames,
                "generated_frames": 0,
                "total_chunks": len(chunks),
                "completed_chunks": 0,
                "incomplete": False,
                "covered_source_span": {"start": 0, "end": 0},
                "history_recent_max_seconds": history_seconds,
                "history_anchor_max_seconds": LONGFORM_CHUNK_MAX_SECONDS,
                "history_policy": "complete_turns_fitted_to_prompt_rows",
                "rng": {
                    "strategy": "torch_manual_seed_once_per_job" if seed >= 0 else "ambient_torch_rng_state",
                    "seed": int(seed) if seed >= 0 else None,
                    "reseed_per_chunk": False,
                },
                "completion_basis": "Every completed chunk ended with EOS; generated speech is not verified word-by-word.",
                "warnings": [],
                "chunks": chunk_reports,
            }
            self.last_longform_report = report

            def add_warning(message):
                if message not in report["warnings"]:
                    report["warnings"].append(message)

            with ExitStack() as stack:
                stack.enter_context(runtime.execution_scope(self._check_abort))
                from .codec import DecoderBlock, EncoderBlock, TransformerBlock
                from .codec_rvq import ConvNeXtBlock
                for module in self.vae.modules():
                    if isinstance(module, (TransformerBlock, EncoderBlock, DecoderBlock, ConvNeXtBlock)):
                        stack.callback(module.register_forward_hook(self._check_abort).remove)
                # Voice references for Speakers 1, 2 and 3 (A, B, D), each bound to its speaker tag by its transcript.
                guides = [guide for letter, guide in zip("ABD", (audio_guide, audio_guide2, audio_guide3)) if letter in audio_prompt_type]
                transcripts = [f"<|speaker:{index}|> {text}" for index, text in enumerate(self._reference_transcripts(guides, alt_prompt))]
                references = [self._reference_codes(guide, f" {index}/{len(guides)}" if len(guides) > 1 else "") for index, guide in enumerate(guides, 1)]
                history_turns, history_anchor = [], None
                code_parts, audio_parts = [], []
                codec_context = None
                produced, finish_reason = 0, "length"
                segment_frames = max(1, math.floor(LONGFORM_CHUNK_MAX_SECONDS * self.sample_rate / self.frame_length))
                decode_tile = min(int(VAE_tile_size) if VAE_tile_size else CODEC_DECODE_TILE_FRAMES, CODEC_DECODE_TILE_FRAMES)

                for index, (chunk, chunk_report) in enumerate(zip(chunks, chunk_reports), 1):
                    chunk_report["status"] = "running"
                    if produced >= max_frames:
                        chunk_report["status"] = "not_started"
                        chunk_report["cap_reasons"].append("whole_job_limit")
                        add_warning(
                            "The selected whole-output maximum was reached. "
                            "Later parts were not generated; this file is partial."
                        )
                        break

                    try:
                        self._last_prompt_history_context = {
                            "recent_turn_count": 0,
                            "recent_audio_seconds": 0.0,
                            "anchor_included": False,
                            "anchor_audio_seconds": 0.0,
                            "total_turn_count": 0,
                            "total_audio_seconds": 0.0,
                            "history_prompt_rows": 0,
                        }
                        prompt = self._build_bounded_prompt(
                            chunk.prompt_text,
                            transcripts,
                            references,
                            history_turns,
                            history_seconds,
                            history_anchor,
                        )
                        chunk_report["history_context"] = dict(
                            self._last_prompt_history_context
                        )
                    except ValueError:
                        chunk_report["status"] = "context_incomplete"
                        chunk_report["cap_reasons"].append("reference_or_prompt_context")
                        add_warning(
                            f"Part {index} could not fit with the selected references and script. "
                            "Later parts were not generated; this file is partial."
                        )
                        if not code_parts:
                            raise
                        break

                    prompt_rows = len(prompt)
                    context_frames = self.config.text.max_position_embeddings - prompt_rows
                    chunk_report["prompt_rows"] = prompt_rows
                    chunk_report["context_limit_frames"] = max(0, context_frames)
                    if context_frames <= 0:
                        chunk_report["status"] = "context_incomplete"
                        chunk_report["cap_reasons"].append("model_context")
                        add_warning(
                            f"Part {index} reached the model context limit before generation. "
                            "Later parts were not generated; this file is partial."
                        )
                        if not code_parts:
                            raise ValueError("The references and text leave no Drama 3 context capacity for speech")
                        break

                    remaining_job_frames = max_frames - produced
                    if duration_budget.automatic:
                        chunk_budget = duration_frame_budget(
                            0, chunk.source_text, self.sample_rate, self.frame_length
                        ).frames
                        desired_frames = min(chunk_budget, segment_frames)
                        if chunk_budget > segment_frames:
                            chunk_report["cap_reasons"].append("segment_guard_120_seconds")
                    else:
                        desired_frames = segment_frames
                        chunk_report["cap_reasons"].append("segment_guard_120_seconds")

                    part_max_frames = desired_frames
                    if remaining_job_frames <= part_max_frames:
                        part_max_frames = remaining_job_frames
                        chunk_report["cap_reasons"].append("whole_job_remaining_budget")
                    if context_frames < part_max_frames:
                        part_max_frames = context_frames
                        chunk_report["cap_reasons"].append("model_context")
                        self.last_duration_context_limited = True
                        if self.last_duration_context_limit_frames is None:
                            self.last_duration_context_limit_frames = context_frames
                        else:
                            self.last_duration_context_limit_frames = min(
                                self.last_duration_context_limit_frames,
                                context_frames,
                            )
                    if part_max_frames <= 0:
                        chunk_report["status"] = "context_incomplete"
                        add_warning(
                            f"Part {index} has no generation budget after context limits. "
                            "Later parts were not generated; this file is partial."
                        )
                        break

                    chunk_report["budget_frames"] = part_max_frames
                    part = f" - Part {index}/{len(chunks)}" if len(chunks) > 1 else ""
                    codes, finish_reason = self._generate_codes(
                        prompt,
                        part_max_frames,
                        temperature,
                        top_k,
                        top_p,
                        callback,
                        produced,
                        max_frames,
                        part,
                    )
                    self._release_decoder_caches()
                    chunk_report["decode_backend"] = getattr(self, "last_decode_backend", "eager")
                    chunk_report["graph_capture_seconds"] = getattr(self, "last_graph_capture_seconds", 0.0)
                    chunk_report["expert_cache"] = getattr(self, "last_expert_cache_report", None)
                    if codes.ndim != 2 or codes.shape[-1] > part_max_frames:
                        raise RuntimeError("Drama 3 returned an invalid semantic-code chunk")
                    chunk_frames = int(codes.shape[-1])
                    chunk_report["generated_frames"] = chunk_frames
                    chunk_report["finish_reason"] = finish_reason
                    produced += chunk_frames
                    report["generated_frames"] = produced
                    self.last_finish_reason = finish_reason

                    if chunk_frames:
                        code_parts.append(codes)
                        decode_codes = codes if codec_context is None else torch.cat((codec_context, codes), dim=-1)
                        prefix_frames = 0 if codec_context is None else codec_context.shape[-1]
                        decoded = self._decode(decode_codes, decode_tile)
                        if prefix_frames:
                            decoded = decoded[..., prefix_frames * self.frame_length:]
                        audio_parts.append(decoded)
                        codec_context = decode_codes[..., -CODEC_CONTEXT_FRAMES:].contiguous()

                    if finish_reason == "eos" and chunk_frames:
                        chunk_report["status"] = "complete"
                        chunk_report["eos_completed"] = True
                        report["completed_chunks"] += 1
                        report["covered_source_span"]["end"] = chunk.source_end
                        completed_turn = (chunk.prompt_text, codes)
                        if history_seconds > 0:
                            if history_anchor is None:
                                history_anchor = completed_turn
                            else:
                                history_turns = self._history(history_turns + [completed_turn], history_seconds)
                        if index < len(chunks) and produced < max_frames:
                            continue
                        if index < len(chunks) and produced >= max_frames:
                            add_warning(
                                "The selected whole-output maximum was reached. "
                                "Later parts were not generated; this file is partial."
                            )
                            break
                        continue

                    chunk_report["status"] = "incomplete"
                    if finish_reason == "length":
                        cap_reasons = chunk_report["cap_reasons"]
                        if "model_context" in cap_reasons:
                            cap = "the model context limit"
                        elif "whole_job_remaining_budget" in cap_reasons:
                            cap = "the selected whole-output maximum"
                        elif "segment_guard_120_seconds" in cap_reasons:
                            cap = "the two-minute part safety limit"
                        else:
                            cap = "its Auto maximum"
                        add_warning(
                            f"Part {index} reached {cap} before finishing. "
                            "Later parts were not generated; this file is partial."
                        )
                    elif finish_reason == "early_stop":
                        add_warning(
                            f"Generation stopped early in part {index}. "
                            "Later parts were not generated; this file is partial."
                        )
                    elif finish_reason == "eos":
                        add_warning(
                            f"Part {index} ended without producing audio. "
                            "Later parts were not generated; this file is partial."
                        )
                    else:
                        add_warning(
                            f"Part {index} stopped before finishing (reason: {finish_reason}). "
                            "Later parts were not generated; this file is partial."
                        )
                    break

                for chunk_report in chunk_reports:
                    if chunk_report["status"] == "pending":
                        chunk_report["status"] = "not_started"
                        reason = "whole_job_limit" if produced >= max_frames else "previous_chunk_incomplete"
                        chunk_report["cap_reasons"].append(reason)

                report["incomplete"] = report["completed_chunks"] != len(chunks)
                report["status"] = "incomplete" if report["incomplete"] else "complete"
                self.last_longform_report = report
                self.last_codes = torch.cat(code_parts, dim=-1) if code_parts else torch.empty((0, 0), dtype=torch.long)
                self.last_finish_reason = finish_reason
                if duration_budget.automatic:
                    print(
                        f"[Fish Audio] Auto speech finished: {finish_reason}; "
                        f"generated {produced}/{max_frames} whole-job budget frames across "
                        f"{report['completed_chunks']}/{len(chunks)} EOS-completed chunks."
                    )
                if not audio_parts:
                    if self._early_stop:
                        return None
                    raise RuntimeError("Drama 3 finished without producing speech")
                audio = torch.cat(audio_parts, dim=-1)
                if not torch.isfinite(audio).all():
                    raise FloatingPointError("Drama 3 decoded non-finite audio")
                result = {
                    "x": audio,
                    "audio_sampling_rate": self.sample_rate,
                    "speech_generation": report,
                    "generation_warnings": list(report["warnings"]),
                }
                return result
        except GenerationCancelled:
            if report is not None:
                report["status"] = "cancelled"
                report["incomplete"] = True
                report["generated_frames"] = sum(row["generated_frames"] for row in report["chunks"])
                report["warnings"].append(
                    "Generation was cancelled. Only audio completed before cancellation is available."
                )
                for row in report["chunks"]:
                    if row["status"] == "running":
                        row["status"] = "cancelled"
                    elif row["status"] == "pending":
                        row["status"] = "not_started"
                        row["cap_reasons"].append("cancelled")
                self.last_longform_report = report
            return None
        except Exception as error:
            if report is not None:
                report["status"] = "error"
                report["incomplete"] = True
                if report["chunks"]:
                    active = next((row for row in report["chunks"] if row["status"] == "running"), None)
                    if active is not None:
                        active["status"] = "error"
                message = f"Long-form speech generation failed: {error}"
                if message not in report["warnings"]:
                    report["warnings"].append(message)
                for row in report["chunks"]:
                    if row["status"] == "pending":
                        row["status"] = "not_started"
                        row["cap_reasons"].append("previous_chunk_error")
                self.last_longform_report = report
            raise
        finally:
            self._release_decoder_caches()

    def release(self):
        self._release_decoder_caches()
        self.last_codes = None
        if self.transformer is not None:
            for module in self.transformer.modules():
                release = getattr(module, "release_gpu_experts", None)
                if callable(release):
                    release()
        self.transformer = self.transformer2 = self.vae = None
