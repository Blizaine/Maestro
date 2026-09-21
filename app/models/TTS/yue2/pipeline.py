"""Lyrics/style/score to stereo music using WanGP's shared AR engine and MMGP."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from mmgp import offload
from tqdm import tqdm
from transformers import Qwen3Config

from models.TTS.yue2._engine.token_generation import TokenGenerationEngine

from .modules import YuE2Config
from .protocol import ABC_END, MUSIC_END, CONTEXT, GenerationConfig, SongRequest, token_prefixes, negative_prefix, chunk_ranges
from .sampling import YuE2LogitsProcessor
from .tokenization_yue2 import YuE2TextTokenizer
from .transformer import YuE2AR, YuE2Acoustic
from .vae import YuE2VAE, YuE2VAEConfig


class YuE2Pipeline:
    sample_rate = 48000
    frame_rate = 25

    def __init__(self, ar_weights, acoustic_weights, tokenizer_path, vae_weights, vae_config, dtype, vae_dtype, lm_decoder_engine, scoring_checkpoint=None):
        directory = Path(__file__).parent
        self._interrupt = False
        self._early_stop = False
        self.lm_decoder_engine = lm_decoder_engine
        self.scoring_checkpoint = scoring_checkpoint

        dtype = vae_dtype = torch.bfloat16
        ar_config = Qwen3Config(**json.loads((directory / "yue2_ar.json").read_text()))
        nar_config = YuE2Config(**json.loads((directory / "yue2.json").read_text()))
        with torch.device("meta"):
            self.text_encoder = YuE2AR(ar_config)
            self.transformer = YuE2Acoustic(nar_config)
            self.vae = YuE2VAE(YuE2VAEConfig(**json.loads(Path(vae_config).read_text())))
        for model, filename, precision in ((self.text_encoder, ar_weights, dtype), (self.transformer, acoustic_weights, dtype), (self.vae, vae_weights, vae_dtype)):
            offload.load_model_data(model, filename, default_dtype=precision, writable_tensors=False)
            model.eval().requires_grad_(False)
        self.text_encoder.configure_engine(lm_decoder_engine, self._abort_requested)
        self.transformer.engine = lm_decoder_engine
        self.transformer.abort_fn = self._abort_requested
        self.vae._model_dtype = vae_dtype
        self.vae._offload_hooks = ["decode", "decode_tiled"]
        self.vae.get_VAE_tile_size = self.get_vae_tile_size
        self._vae_abort_handles = [block.register_forward_hook(self._abort_after_block) for block in self.vae.decoder.layers]
        self.tokenizer = YuE2TextTokenizer(tokenizer_path)
        self.engine = TokenGenerationEngine(self.text_encoder, ar_weights, self.tokenizer, enforce_eager=lm_decoder_engine == "legacy")
        print(f"[YuE2] AR LM engine: {lm_decoder_engine} (CUDA graphs: {'off' if lm_decoder_engine == 'legacy' else 'on'}; Triton decoder kernels: {'on' if lm_decoder_engine == 'vllm' else 'off'}; attention: {'FlashAttention 2' if lm_decoder_engine == 'vllm' else 'PyTorch SDPA'}).")
        print(f"[YuE2] Acoustic flow: PyTorch midpoint solver; attention: {'FlashAttention 2' if lm_decoder_engine == 'vllm' else 'PyTorch SDPA'}; CUDA graphs: off.")
        self.generation_config = GenerationConfig()
        self.last_plan = None
        self.last_latents = None
        self.last_truncated = {}

    @staticmethod
    def get_vae_tile_size(vae_config, device_mem_capacity, mixed_precision):
        if vae_config == 1:
            return 0
        if vae_config == 3 or (vae_config == 0 and device_mem_capacity < 12000):
            return 256
        return 1024

    def _abort_requested(self):
        return self._interrupt

    def request_early_stop(self):
        self._early_stop = True

    def _early_stop_requested(self):
        return self._early_stop

    def _abort_after_block(self, module, inputs, output):
        if self._interrupt:
            raise InterruptedError("YuE2 audio decoding interrupted")

    def _tokens(self, prefix, sampling, seed, phase, callback, negative=None, cfg_scale=1.0, direct=False):
        available = CONTEXT - max(len(prefix), len(negative) if negative is not None else 0)
        if phase == "semantic":
            if available < 5:
                raise ValueError("YuE2 lyrics/score leave no room for audio synthesis. Shorten the lyrics or score.")
            if sampling.max_tokens > available:
                print(f"[YuE2] Maximum audio duration limited by context: {sampling.max_tokens / self.frame_rate:.2f}s requested, up to {available / self.frame_rate:.2f}s available after lyrics/score. Generation can end earlier.")
                sampling = replace(sampling, max_tokens=available, min_tokens=min(sampling.min_tokens, available - 1))
        if sampling.max_tokens > available:
            raise ValueError(f"YuE2 {phase} prompt and generation budget exceed its {CONTEXT}-token context. Shorten the lyrics, score or duration.")
        # Trigger MMGP before the engine allocates persistent GPU state.
        self.text_encoder.model.embed_tokens(torch.tensor([prefix[0]], device="cuda"))
        tokens, truncated = self.engine.generate_tokens(prefix, end_token=ABC_END if phase == "abc" else MUSIC_END, max_tokens=sampling.max_tokens, seed=seed, logits_processor=YuE2LogitsProcessor(sampling, phase, direct), negative=negative, cfg_scale=cfg_scale, callback=callback, abort_fn=self._abort_requested, stop_fn=self._early_stop_requested if phase == "semantic" else None, stop_min_tokens=sampling.min_tokens if self._early_stop else 0, progress_label="YuE2 score" if phase == "abc" else "YuE2 semantic audio", initial_cache_tokens=60 * self.frame_rate if phase == "semantic" else 0)
        self.last_truncated[phase] = truncated
        if self._early_stop and phase == "semantic":
            print(f"[YuE2] Early stop during {phase}: {len(tokens)} tokens retained.")
        elif truncated:
            print(f"YuE2 {phase} reached its token limit; output may be incomplete.")
        else:
            print(f"[YuE2] {phase} reached its end token after {len(tokens)} tokens" + (f" ({len(tokens) / self.frame_rate:.2f}s of audio)." if phase == "semantic" else "."))
        return tokens

    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        from .artist_adapter import active_artists, artist_style_prompt
        from .instrumental import is_instrumental, section_plan, instrumental_settings, active_instrumental
        prompt = args[0] if args else kwargs.get('input_prompt')
        if is_instrumental(kwargs, prompt):
            self._interrupt = self._early_stop = False
            plan = section_plan(prompt)
            if args:
                args = (plan, *args[1:])
            else:
                kwargs['input_prompt'] = plan
            kwargs.update(model_mode=0, audio_prompt_type='', audio_guide=None,
                          custom_settings=instrumental_settings(kwargs.get('custom_settings')))
            with active_instrumental(self) as instrumental:
                result = self._generate(*args, **kwargs)
                if isinstance(result, dict):
                    result.setdefault('artifact_metadata', {})['instrumental'] = instrumental
                    result.setdefault('overridden_inputs', {}).update(
                        prompt=plan, model_mode=0, audio_prompt_type='', audio_guide=None,
                        custom_settings=kwargs['custom_settings'], _music_instrumental=True)
                return result
        with active_artists(self, kwargs.get("custom_settings")) as artists:
            if artists:
                kwargs['alt_prompt'] = artist_style_prompt(kwargs.get('alt_prompt'), artists)
            result = self._generate(*args, **kwargs)
            if artists and isinstance(result, dict):
                metadata = result.setdefault('artifact_metadata', {})
                metadata['artists'] = artists
                if len(artists) == 1:
                    metadata['artist'] = artists[0]  # Existing single-song readers.
                else:
                    metadata['artist_mix'] = 'additive-ar-joint-nar-weighted-companions-v1'
            return result

    @torch.inference_mode()
    def _generate(self, input_prompt, alt_prompt, seed, duration_seconds, sampling_steps, guide_scale, temperature, top_k, top_p, model_mode=2, custom_settings=None, VAE_tile_size=1024, callback=None, audio_prompt_type="", audio_guide=None, offloadobj=None, **kwargs):
        self._interrupt = self._early_stop = False
        self.last_plan = self.last_latents = None
        self.last_truncated = {}
        mode = ("full", "melody", "off")[model_mode]
        abc = custom_settings["abc"].strip() if "A" not in audio_prompt_type and custom_settings is not None and "abc" in custom_settings else ""
        maximum = int(duration_seconds * self.frame_rate)
        sampling = replace(self.generation_config.semantic, max_tokens=maximum, min_tokens=min(200, maximum - 1), temperature=temperature, top_k=top_k, top_p=top_p)
        try:
            if "A" in audio_prompt_type:
                from .score_dependencies import ensure_score_dependencies
                ensure_score_dependencies(self._abort_requested)
                from .sheetsage2.scoring import score_audio
                if not self.scoring_checkpoint:
                    from .yue2_handler import REPO_ID, REVISION, SCORING_CHECKPOINT
                    from shared.utils import files_locator as fl
                    from wgp import process_files_def
                    self.scoring_checkpoint = fl.locate_file("sheetsage2/" + SCORING_CHECKPOINT, error_if_none=False)
                    if not self.scoring_checkpoint:
                        process_files_def(repoId=REPO_ID, revision=REVISION, sourceFolderList=["sheetsage2"], fileList=[[SCORING_CHECKPOINT]])
                        self.scoring_checkpoint = fl.locate_file("sheetsage2/" + SCORING_CHECKPOINT)
                if self._interrupt:
                    raise InterruptedError("YuE2 scoring cancelled")
                self.engine.release_runtime_allocations()
                offloadobj.unload_all()
                abc = score_audio(audio_guide, self.scoring_checkpoint, mode == "melody", callback, self._abort_requested)
            request = SongRequest(style=alt_prompt, lyrics=input_prompt, cot=mode, abc=abc or None, cfg_scale=guide_scale, seed=seed)
            if mode == "off":
                abc_ids = []
            elif abc:
                abc_ids = self.tokenizer.encode(abc)
            else:
                abc_ids = self._tokens(token_prefixes(request, self.tokenizer), self.generation_config.abc, seed, "abc", callback)
                abc = self.tokenizer.decode(abc_ids)
            self.last_plan = {"abc": abc, "abc_ids": abc_ids, "request": request.to_dict()}
            prefix = token_prefixes(request, self.tokenizer, abc_ids)
            negative = negative_prefix(request, self.tokenizer, abc_ids) if guide_scale != 1 else None
            codec = self._tokens(prefix, sampling, seed, "semantic", callback, negative, guide_scale, mode == "off")
            self.engine.release_runtime_allocations()
            if not codec:
                if self._early_stop:
                    return None
                raise RuntimeError("YuE2 generated no audio tokens.")
            audio = self.decode_codec(prefix, codec, seed, sampling_steps, VAE_tile_size, callback)
            if self._interrupt:
                return None
            if not torch.isfinite(audio).all():
                raise FloatingPointError("YuE2 decoded non-finite audio.")
            return {"x": audio[0].float().clamp_(-1, 1), "audio_sampling_rate": self.sample_rate,
                    "artifact_metadata": {"architecture": "yue2", "plan": self.last_plan,
                        "truncated": dict(self.last_truncated),
                        "acoustic_inputs": {
                            "version": 1,
                            "codec_tokens": len(codec),
                            "codec_sha256": hashlib.sha256(json.dumps(codec, separators=(",", ":")).encode("utf-8")).hexdigest(),
                            "prefix_sha256": hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode("utf-8")).hexdigest(),
                            "noise_seed": seed, "sampling_steps": sampling_steps,
                            "vae_tile_size": VAE_tile_size, "engine": self.lm_decoder_engine,
                        },
                        "weights_revision": "864a479cbf3e810e1b2c1993b438510750e383b2",
                        "sample_rate": self.sample_rate, "channels": 2},
                    "overridden_inputs": {"_yue2_plan": self.last_plan,
                        "_yue2_truncated": dict(self.last_truncated),
                        "_yue2_weights_revision": "864a479cbf3e810e1b2c1993b438510750e383b2",
                        "_yue2_sample_rate": self.sample_rate}}
        except InterruptedError:
            if not self._interrupt:
                raise
            print("[YuE2] Generation aborted.")
            return None
        finally:
            self.engine.release_runtime_allocations()
            self.last_latents = None

    @torch.inference_mode()
    def decode_codec(self, prefix, codec, seed, sampling_steps=32, VAE_tile_size=1024, callback=None):
        """Render fixed native codec IDs through the production acoustic/VAE path.

        Reconstruction diagnostics use this same path while bypassing semantic
        sampling. The AR model still supplies the decoder's conditioning cache.
        """
        chunks = chunk_ranges(len(codec), len(prefix))
        noise = torch.randn((len(codec), 64), device="cpu", generator=torch.Generator(device="cpu").manual_seed(seed))
        latent_parts = []
        total = len(chunks) * sampling_steps
        with tqdm(total=total, desc="YuE2 acoustic synthesis", unit="step") as progress:
            for chunk_index, (start, end) in enumerate(chunks):
                ar_ids = prefix + codec[start:end] + [MUSIC_END]
                cache = [self.text_encoder.condition(ar_ids)]
                def report(step):
                    progress.update()
                    if callback is not None:
                        callback(step_idx=chunk_index * sampling_steps + step, override_num_inference_steps=total, denoising_extra="YuE2 acoustic synthesis")
                latent_parts.append(self.transformer.synthesize(noise[start:end], cache, len(ar_ids), sampling_steps, report))
        latents = torch.cat(latent_parts)
        self.last_latents = latents
        latent = latents.T[None]
        tile = VAE_tile_size
        total = (len(latents) + tile - 1) // tile if tile else 1
        with tqdm(total=total, desc="YuE2 audio decoding", unit="tile") as progress:
            def report_decode(completed, total):
                progress.update()
                if callback is not None:
                    callback(step_idx=completed - 1, override_num_inference_steps=total, denoising_extra="YuE2 audio decoding", progress_unit="tiles")
            if tile:
                audio = self.vae.decode_tiled(latent, core_frames=tile, halo_frames=16, on_progress=report_decode)
            else:
                audio = self.vae.decode(latent).float().cpu()
                report_decode(1, 1)
        return audio

    def release(self):
        self.engine.close()
        self.last_latents = self.last_plan = None
