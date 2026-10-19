# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
import collections
import gc
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tqdm import tqdm

from .runtime_support import can_place_gpu_experts, cuda_graphs, fish_vllm_availability, runtime
from .progress import generation_progress, PhaseProgress, set_phase_status
from shared.utils.text_encoder_cache import TextEncoderCache
from .backbone import GenerationCancelled
from .loading import load_transformers, load_codec
from .fish_audio_handler import DIALOGUE_CUT_LINES, DIALOGUE_HISTORY_SECONDS
from .prompt import build_drama3_prompt, dialogue_parts, reference_transcripts, speaker_lines_to_tags, validate_drama3_text

# Prompt rows processed together so each routed expert is read once per chunk,
# rather than reread for every prompt token.
PREFILL_ROWS = 512
SILENCE_SECONDS, SILENCE_SHARE, SILENCE_LOGIT = 8.0, 0.8, 2.0


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
        if torch.cuda.is_initialized():
            torch.cuda.current_stream().synchronize()  # a decoding step queued ahead may still run (and its CPU expert jobs)
        cuda_graphs.invalidate()
        self.transformer.release_cache()
        self.transformer2.release_cache()

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
                sampled, finite, extra, semantic = tower.decode_step(logits, hidden, positions[position:position + 1], frames[position:position + 1], frames[:position + 1, 0],
                    end=position + 1, temperature=temperature, top_k=top_k, top_p=top_p)
                # Graph outputs are not held when their stage replays again (MMGP would capture it again to protect them).
                del logits, hidden
                status.copy_(torch.cat((sampled[0], finite.to(sampled.dtype).view(1))), non_blocking=True)  # read after frame_done
                frames[position + 1] = sampled[0]  # kept only if the frame is produced (counted below)
                if frame_done is not None:
                    frame_done.record()
                if step + 1 < max_frames:
                    hidden, logits = backbone.decode(sampled[:, 0], extra, semantic, positions[position + 1:position + 2], position + 1)
                del sampled, finite, extra, semantic
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
        print(f"[Fish Audio] {produced} Frames in {elapsed:.2f}s ({produced / elapsed:.2f} Frames/s); Finish: {finish_reason}")
        return frames[prompt_rows:prompt_rows + produced, 1:].T.contiguous().cpu(), finish_reason

    def _decode(self, codes, tile_size):
        set_phase_status("Decoding Speech")
        frames = codes.shape[-1]
        tile = tile_size or frames
        total = math.ceil(frames / tile)
        parts = []
        with PhaseProgress(total, "Decoding Speech", "tiles") as progress, tqdm(total=total, desc="Decoding Speech", unit="tile", mininterval=1 / 3) as bar:
            for start in range(0, frames, tile):
                self._check_abort()
                end = min(start + tile, frames)
                # The causal codec's attention sees 8 * 127 preceding frames. Include
                # its convolution context too; retain the prefix for short utterances.
                left = max(0, start - 1152)
                segment = codes[None, :, left:end].to(runtime.execution_device(self.vae)).clone()
                decoded = self.vae(segment)
                parts.append(decoded[0, :, (start - left) * self.frame_length:].float().cpu())
                del decoded, segment
                bar.update()
                progress.advance()
        return torch.cat(parts, dim=-1)

    def _history(self, turns, seconds):
        """The most recent whole parts whose audio fits in `seconds`, in order (a part that produced no audio is left out)."""
        frames, kept = seconds * self.sample_rate / self.frame_length, []
        for text, codes in reversed([turn for turn in turns if turn[1].shape[-1]]):
            frames -= codes.shape[-1]
            if frames < 0:
                break
            kept.insert(0, (text, codes))
        return kept

    @torch.inference_mode()
    @generation_progress
    def generate(self, input_prompt, *, alt_prompt="", audio_guide=None, audio_guide2=None, audio_guide3=None, audio_prompt_type="", duration_seconds=30,
                 temperature=0.8, top_k=30, top_p=0.95, seed=-1, VAE_tile_size=0, custom_settings=None, callback=None, set_progress_status=None, **kwargs):
        self._interrupt = self._early_stop = False
        self.last_codes = self.last_finish_reason = None
        if seed >= 0:
            torch.manual_seed(seed)
        settings = custom_settings or {}  # saved settings may predate these options
        try:
            with ExitStack() as stack:
                stack.enter_context(runtime.execution_scope(self._check_abort))
                from .codec import TransformerBlock, EncoderBlock, DecoderBlock
                from .codec_rvq import ConvNeXtBlock
                for module in self.vae.modules():
                    if isinstance(module, (TransformerBlock, EncoderBlock, DecoderBlock, ConvNeXtBlock)):
                        stack.callback(module.register_forward_hook(self._check_abort).remove)
                # Voice references for Speakers 1, 2 and 3 (A, B, D), each bound to its speaker tag by its transcript.
                guides = [guide for letter, guide in zip("ABD", (audio_guide, audio_guide2, audio_guide3)) if letter in audio_prompt_type]
                transcripts = [f"<|speaker:{index}|> {text}" for index, text in enumerate(self._reference_transcripts(guides, alt_prompt))]
                references = [self._reference_codes(guide, f" {index}/{len(guides)}" if len(guides) > 1 else "") for index, guide in enumerate(guides, 1)]
                parts = dialogue_parts(speaker_lines_to_tags(input_prompt), int(settings.get(DIALOGUE_CUT_LINES["id"], DIALOGUE_CUT_LINES["default"])))
                history_seconds = float(settings.get(DIALOGUE_HISTORY_SECONDS["id"], DIALOGUE_HISTORY_SECONDS["default"]))
                max_frames = max(1, math.floor(duration_seconds * self.sample_rate / self.frame_length))
                turns, produced, finish_reason = [], 0, "length"
                for index, text in enumerate(parts, 1):
                    # Each part is a new prompt: references, the last parts as history (text and their generated audio), its lines.
                    prompt = build_drama3_prompt(text, self.tokenizer, self.config, transcripts, references, self._history(turns, history_seconds))
                    part = f" - Part {index}/{len(parts)}" if len(parts) > 1 else ""
                    codes, finish_reason = self._generate_codes(prompt, max_frames - produced, temperature, top_k, top_p, callback, produced, max_frames, part)
                    turns.append((text, codes))
                    produced += codes.shape[-1]
                    if finish_reason not in ("eos", "silence") or produced >= max_frames:
                        break  # the maximum duration, or an early stop (a part that fell silent lets the next one go on)
                    if index < len(parts):
                        self._release_decoder_caches()
                codes = torch.cat([codes for _, codes in turns], dim=-1)
                self.last_codes, self.last_finish_reason = codes, finish_reason
                self._release_decoder_caches()
                if codes.shape[-1] == 0:
                    if self._early_stop:
                        return None
                    raise RuntimeError("Drama 3 finished without producing speech")
                audio = self._decode(codes, VAE_tile_size)
                if not torch.isfinite(audio).all():
                    raise FloatingPointError("Drama 3 decoded non-finite audio")
                return {"x": audio, "audio_sampling_rate": self.sample_rate}
        except GenerationCancelled:
            return None
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
