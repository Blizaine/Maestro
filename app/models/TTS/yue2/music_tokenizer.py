"""Mothersuperior v4 real-audio tokenization, using the unmodified MERT backbone.

Head architecture and overlap stitching adapted from ar_prep.py at the pinned
tokenizer revision. Recordings are read only; preparation is cached per dataset.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from accelerate import init_empty_weights

from .music_assets import ensure_asset, MERT_REVISION
from services.music_contracts import tokenizer_pair, pair_asset
from .sheetsage2.configuration_mert2 import MERT2Config
from .sheetsage2.modeling_mert2 import MERT2Model
from services.music_training import project_directory, update_project
from services.music_styles import file_digest


class RealAudioHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(1024, 512)
        self.pos = nn.Parameter(torch.zeros(1, 512, 512))
        layer = nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.1, batch_first=True,
                                          norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 8)
        self.norm = nn.LayerNorm(512)
        self.head = nn.Linear(512, 32768)

    def forward(self, values):
        return self.head(self.norm(self.enc(self.inp(values) + self.pos[:, :values.shape[1]])))


@torch.inference_mode()
def predict_codes(head, features, *, cancelled=lambda: False):
    features = features.astype(np.float32)
    features = (features - features.mean(0)) / (features.std(0) + 1e-5)
    length, window = len(features), 512
    if length < 1:
        raise ValueError("The recording did not contain enough audio to tokenize")
    output = np.zeros(length, dtype=np.int32)
    starts = list(range(0, max(1, length - window + 1), window // 2))
    if starts[-1] + window < length:
        starts.append(max(0, length - window))
    for start in starts:
        if cancelled():
            raise InterruptedError("Music tokenization cancelled")
        sample = features[start:start + window]
        count = len(sample)
        if count < window:
            sample = np.pad(sample, ((0, window - count), (0, 0)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            codes = head(torch.as_tensor(sample[None], device="cuda"))[0, :count].float().argmax(-1).cpu().numpy()
        low = start + (0 if start == 0 else window // 4)
        high = start + count - (0 if start + count >= length else window // 4)
        output[low:high] = codes[low - start:high - start]
    return output


def _waveform(path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "json", str(path)], capture_output=True, text=True, check=True)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if not 1 <= duration <= 600:
        raise ValueError("Use recordings between 1 second and 10 minutes for music training")
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                             "-ar", "24000", "-f", "f32le", "pipe:1"], capture_output=True, check=True)
    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if not np.isfinite(waveform).all():
        raise ValueError("The recording contains invalid audio samples")
    return waveform


def prepare_project(project, *, report, cancelled):
    pair = tokenizer_pair(project)
    paths = {name: ensure_asset(name, cancelled=cancelled, report=report)
             for name in ("mert", "mert_config", "mert_processor")}
    paths['tokenizer'] = pair_asset(project, 'head', report=report, cancelled=cancelled)
    directory = project_directory(project["id"]) / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    cache_identity = {"dataset_digest": project["dataset_digest"], "mert_revision": MERT_REVISION,
                      "tokenizer_revision": pair['revision'], "feature_layer": 20, "frame_rate": 25}
    processor = json.loads(paths["mert_processor"].read_text())
    if processor.get("do_normalize") is not False or processor.get("sampling_rate") != 24000:
        raise ValueError("The MERT processor does not match the real-audio tokenizer")
    config = MERT2Config(**json.loads(paths["mert_config"].read_text()))
    config._attn_implementation = "sdpa"
    model = head = None
    try:
        with init_empty_weights():
            model = MERT2Model(config)
        model.load_state_dict(load_file(str(paths["mert"])), strict=True, assign=True)
        model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
        head = RealAudioHead()
        weights = (load_file(str(paths['tokenizer'])) if paths['tokenizer'].suffix == '.safetensors'
                   else torch.load(paths['tokenizer'], map_location='cpu', weights_only=True)['model'])
        head.load_state_dict(weights, strict=True)
        del weights
        head = head.to("cuda").eval().requires_grad_(False)
        tracks = project["tracks"]
        for index, track in enumerate(tracks):
            if cancelled():
                raise InterruptedError("Music preparation cancelled")
            path = Path(track["audio_path"])
            if file_digest(path) != track["audio_sha256"]:
                raise ValueError(f"{track['name']} changed after this dataset was created; create a new project")
            marker = directory / f"{track['id']}.json"
            codes_path = directory / f"{track['id']}.npy"
            if marker.is_file() and codes_path.is_file() and json.loads(marker.read_text()) == cache_identity:
                report(f"Reusing prepared tokens: {track['name']}", (index + 1) / len(tracks) * 100)
                continue
            report(f"Extracting music features: {track['name']}", index / len(tracks) * 100)
            waveform = _waveform(path)
            features = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for start in range(0, len(waveform), 24000 * 30):
                    if cancelled():
                        raise InterruptedError("Music preparation cancelled")
                    chunk = waveform[start:start + 24000 * 30]
                    # The upstream head was trained with sub-second tails omitted.
                    if len(chunk) >= 24000:
                        encoded = model(torch.as_tensor(chunk[None], device="cuda"), output_hidden_states=True)
                        features.append(encoded.hidden_states[20][0].float().cpu())
                        del encoded
            hidden = torch.cat(features)
            features = F.interpolate(hidden.T[None], size=round(len(waveform) / 24000 * 25),
                                     mode="linear", align_corners=False)[0].T.numpy()
            codes = predict_codes(head, features, cancelled=cancelled)
            with codes_path.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, codes, allow_pickle=False)
            codes_path.with_suffix(".tmp").replace(codes_path)
            marker.write_text(json.dumps(cache_identity), encoding="utf-8")
            report(f"Prepared {track['name']}: {len(codes) / 25:.1f}s", (index + 1) / len(tracks) * 100)
        update_project(project["id"], prepared=cache_identity)
    finally:
        model = head = None
        gc.collect()
        torch.cuda.empty_cache()
