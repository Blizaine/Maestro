"""Frozen 48 kHz stereo encoder and isolated 25 Hz training target cache.

Oobleck encoder adapted from m-a-p/YuE2-Vae (pinned in music_assets).
See vae.py and THIRD_PARTY_NOTICES.md for the upstream MIT notices.
"""
import gc
import json
from math import ceil, gcd
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from scipy.signal import resample_poly
import soundfile as sf

from .vae import ResidualUnit, WNConv1d, get_activation
from .music_assets import ensure_asset, VAE_REVISION
from services.music_contracts import tokenizer_pair
from services.music_styles import file_digest
from services.music_training import project_directory, update_project


class EncoderBlock(nn.Module):
    def __init__(self, channels, output, stride):
        super().__init__()
        self.layers = nn.Sequential(*(ResidualUnit(channels, channels, d, 'snake') for d in (1, 3, 9)),
            get_activation('snake', channels), WNConv1d(channels, output, 2 * stride, stride=stride, padding=ceil(stride / 2)))

    def forward(self, x):
        return self.layers(x)


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [WNConv1d(2, 64, 7, padding=3)]
        channels = 64
        for multiplier, stride in zip((1, 2, 4, 8, 16, 32), (2, 2, 4, 4, 5, 6)):
            output = 64 * multiplier
            layers.append(EncoderBlock(channels, output, stride))
            channels = output
        layers.extend((get_activation('snake', channels), WNConv1d(channels, 128, 3, padding=1)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Deterministic posterior mean, matching upstream encode(sample=False).
        return self.layers(x).chunk(2, dim=1)[0]

    @torch.inference_mode()
    def encode_tiled(self, waveform, *, core=512, halo=32, cancelled=lambda: False):
        count = len(waveform) // 1920
        parts = []
        for start in range(0, count, core):
            if cancelled():
                raise InterruptedError('Audio preparation cancelled')
            left, end = max(0, start - halo), min(count, start + core)
            right = min(len(waveform), (end + halo) * 1920)
            x = torch.as_tensor(waveform[left * 1920:right].T[None], device='cuda')
            with torch.backends.cudnn.flags(allow_tf32=False):
                z = self(x)[0, :, start - left:end - left].T.float().cpu().numpy()
            if len(z) != end - start or not np.isfinite(z).all():
                raise ValueError('The audio encoder produced invalid frame-aligned targets')
            parts.append(z)
        if not parts:
            raise ValueError('The recording is too short to encode')
        return np.concatenate(parts)


def read_stereo(path, sample_rate=48000):
    # ffmpeg supports the same upload formats as preparation (including m4a).
    import shutil
    import subprocess
    try:
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
    except (RuntimeError, sf.LibsndfileError):
        executable = shutil.which('ffmpeg')
        if not executable:
            raise ValueError('ffmpeg is required to read this recording')
        result = subprocess.run([executable, '-v', 'error', '-i', str(path), '-f', 'f32le',
                                 '-ar', str(sample_rate), '-ac', '2', 'pipe:1'], capture_output=True, check=True)
        audio, rate = np.frombuffer(result.stdout, dtype='<f4').reshape(-1, 2), sample_rate
    if len(audio) / rate > 20 * 60:
        raise ValueError('Training recordings must be at most 20 minutes each')
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] != 2:
        raise ValueError('Use mono or stereo source recordings')
    if rate != sample_rate:
        factor = gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // factor, rate // factor).astype(np.float32)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('The source recording is empty or contains invalid samples')
    return audio


def target_identity(project, track):
    return {'version': 1, 'source_sha256': track['audio_sha256'], 'dataset_digest': project['dataset_digest'],
            'vae_revision': VAE_REVISION, 'tokenizer_revision': tokenizer_pair(project)['revision'],
            'sample_rate': 48000, 'channels': 2, 'frame_rate': 25, 'posterior': 'mean', 'halo': 32, 'precision': 'fp32-no-tf32'}


def load_encoder(path, device='cpu'):
    with torch.device('cpu'):
        model = AudioEncoder().float()
    with safe_open(str(path), framework='pt', device='cpu') as source:
        weights = {key.removeprefix('encoder.'): source.get_tensor(key) for key in source.keys() if key.startswith('encoder.')}
    # The official full VAE retains weight normalization; the inference port's
    # convolutions store the equivalent unwrapped weight.
    for key in list(weights):
        if key.endswith('.weight_g'):
            base = key.removesuffix('_g')
            weights[base] = torch._weight_norm(weights.pop(base + '_v').to(device), weights.pop(key).to(device), 0).cpu()
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval().requires_grad_(False)


def prepare_audio(project, *, report, cancelled):
    if not project.get('prepared'):
        raise ValueError('Prepare music tokens before preparing audio targets')
    directory = project_directory(project['id']) / 'audio_targets'
    directory.mkdir(exist_ok=True)
    model = None
    try:
        path = ensure_asset('vae_training', report=report, cancelled=cancelled)
        model = load_encoder(path, 'cuda')
        for i, track in enumerate(project['tracks']):
            if cancelled():
                raise InterruptedError('Audio preparation cancelled')
            identity = target_identity(project, track)
            marker, output = directory / (track['id'] + '.json'), directory / (track['id'] + '.npy')
            if file_digest(Path(track['audio_path'])) != track['audio_sha256']:
                raise ValueError('A source recording changed. Create a new project')
            if marker.is_file() and output.is_file() and json.loads(marker.read_text()) == identity:
                continue
            report(f"Preparing source sound: {track['name']}", 100 * i / len(project['tracks']))
            latents = model.encode_tiled(read_stereo(track['audio_path']), cancelled=cancelled)
            with output.with_suffix('.tmp').open('wb') as stream:
                np.save(stream, latents.astype(np.float32), allow_pickle=False)
            output.with_suffix('.tmp').replace(output)
            marker.write_text(json.dumps(identity), encoding='utf-8')
        update_project(project['id'], audio_prepared={'version': 1, 'vae_revision': VAE_REVISION,
                                                     'dataset_digest': project['dataset_digest']})
    finally:
        model = None
        gc.collect()
        torch.cuda.empty_cache()
