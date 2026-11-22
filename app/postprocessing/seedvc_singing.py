"""Local SeedVC v1 pitch-conditioned singing runtime."""

from __future__ import annotations

import contextlib
import importlib
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

_REPO = "DeepBeepMeep/LTX-2"
_REVISION = "1798f80b3340d2f87cffae42f9fa5367d0963d0c"
_ROOT = "seed-vc"
_CONFIG = "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
_CHECKPOINT = "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
_VOCODER = "bigvgan_v2_44khz_128band_512x"
_WHISPER = "whisper-small"
_LOCK = threading.Lock()


class SingingConverter:
    def __init__(self, app):
        self.app = app

    def convert_tensor(self, source_audio, source_rate, reference_audio, reference_rate,
                       *, output_rate, diffusion_steps, cfg_rate):
        import numpy as np
        import soundfile as sf
        import torch
        from torchaudio.functional import resample

        with tempfile.TemporaryDirectory(prefix="maestro-singing-") as work:
            source_path, reference_path = Path(work, "source.wav"), Path(work, "voice.wav")
            sf.write(source_path, source_audio.float().cpu().mean(dim=0).numpy(), source_rate, subtype="FLOAT")
            sf.write(reference_path, reference_audio.float().cpu().mean(dim=0).numpy(), reference_rate, subtype="FLOAT")
            audio = None
            for _, candidate in self.app.voice_conversion(
                str(source_path), str(reference_path), int(diffusion_steps),
                1.0, float(cfg_rate), False, 0,
            ):
                if candidate is not None:
                    audio = candidate
        if audio is None:
            raise RuntimeError("SeedVC produced no converted singing.")
        rate, samples = audio
        samples = np.asarray(samples)
        scale = 32768.0 if samples.dtype == np.int16 else 1.0
        waveform = torch.from_numpy(samples.astype(np.float32) / scale).reshape(1, -1)
        if rate != output_rate:
            waveform = resample(waveform, rate, output_rate)
        return waveform


@contextlib.contextmanager
def singing_converter():
    import torch
    import yaml
    from shared.utils import files_locator as fl
    from shared.utils.download import process_download_defs

    process_download_defs([{
        "repoId": _REPO, "revision": _REVISION,
        "sourceFolderList": [_ROOT, _VOCODER, _WHISPER],
        "fileList": [
            [_CHECKPOINT, _CONFIG, "campplus_cn_common.bin", "rmvpe.pt"],
            ["config.json", "bigvgan_generator.pt"],
            ["config.json", "generation_config.json", "preprocessor_config.json",
             "model.safetensors", "added_tokens.json", "merges.txt",
             "normalizer.json", "special_tokens_map.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.json"],
        ],
    }])
    model_fields = (
        "model_f0", "semantic_fn", "vocoder_fn", "campplus_model",
        "to_mel_f0", "mel_fn_args", "f0_fn",
    )
    with _LOCK, tempfile.TemporaryDirectory(prefix="maestro-singing-config-") as work:
        cache = os.environ.get("HF_HUB_CACHE")
        try:
            app = importlib.import_module("postprocessing.seedvc.app_svc")
        finally:
            if cache is None:
                os.environ.pop("HF_HUB_CACHE", None)
            else:
                os.environ["HF_HUB_CACHE"] = cache
        saved = {key: getattr(app, key, None) for key in (
            *model_fields, "device", "sr", "hop_length", "max_context_window",
            "overlap_wave_len", "fp16", "load_custom_model_from_hf",
        )}
        config = yaml.safe_load(Path(fl.locate_file(f"{_ROOT}/{_CONFIG}")).read_text(encoding="utf-8"))
        config["model_params"]["vocoder"]["name"] = fl.locate_folder(_VOCODER)
        config["model_params"]["speech_tokenizer"]["name"] = fl.locate_folder(_WHISPER)
        config_path = Path(work, "model.yml")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        def load_asset(repo, filename, config_filename=None):
            if filename not in {"campplus_cn_common.bin", "rmvpe.pt"} or config_filename:
                raise FileNotFoundError(f"Unexpected SeedVC asset {repo}/{filename}")
            return fl.locate_file(f"{_ROOT}/{filename}")

        app.load_custom_model_from_hf = load_asset
        app.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            models = app.load_models(SimpleNamespace(
                checkpoint=fl.locate_file(f"{_ROOT}/{_CHECKPOINT}"),
                config=str(config_path), fp16=app.device.type == "cuda",
            ))
            for key, value in zip(model_fields, models, strict=True):
                setattr(app, key, value)
            app.max_context_window = app.sr // app.hop_length * 30
            app.overlap_wave_len = app.overlap_frame_len * app.hop_length
            del models
            yield SingingConverter(app)
        finally:
            for key, value in saved.items():
                setattr(app, key, value)
