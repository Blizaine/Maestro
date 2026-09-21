"""Real MERT features and complete minted anchors for the author adaptation."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file

from .audio_training_data import prepare_audio, target_identity, read_stereo
from .music_assets import ROOT, MERT_REVISION, ensure_asset
from .music_tokenizer import _waveform, prepare_project
from .protocol import SongRequest, token_prefixes
from .sheetsage2.configuration_mert2 import MERT2Config
from .sheetsage2.modeling_mert2 import MERT2Model
from services.music_styles import file_digest
from services.music_training import project_directory, update_project


def norm_features(features):
    value = np.asarray(features, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 1024 or not np.isfinite(value).all():
        raise ValueError('Invalid real-audio MERT features')
    return (value - value.mean(0)) / (value.std(0) + 1e-5)


def preparation_identity(project):
    manifest = json.loads(Path(__file__).with_name('pair_regularizer.json').read_text())
    return {'version': 1, 'dataset_digest': project['dataset_digest'], 'mert_revision': MERT_REVISION,
            'regularizer_digest': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}


def prepare_features(entries, *, report, cancelled):
    """Entries contain trusted audio paths, hashes and feature cache paths."""
    pending = []
    for entry in entries:
        marker = entry['features'].with_suffix('.json')
        identity = {'source_sha256': entry['sha256'], 'mert_revision': MERT_REVISION, 'layer': 20, 'frame_rate': 25}
        if file_digest(entry['audio']) != entry['sha256']:
            raise ValueError('An adaptation recording changed')
        try:
            if entry['features'].is_file() and json.loads(marker.read_text()) == identity:
                continue
        except (OSError, ValueError):
            pass
        pending.append((entry, marker, identity))
    if not pending:
        return
    paths = {k: ensure_asset(k, report=report, cancelled=cancelled) for k in ('mert', 'mert_config', 'mert_processor')}
    processor = json.loads(paths['mert_processor'].read_text())
    if processor.get('do_normalize') is not False or processor.get('sampling_rate') != 24000:
        raise ValueError('Unexpected MERT input normalization')
    config = MERT2Config(**json.loads(paths['mert_config'].read_text()))
    config._attn_implementation = 'sdpa'
    model = None
    try:
        with init_empty_weights():
            model = MERT2Model(config)
        model.load_state_dict(load_file(str(paths['mert'])), strict=True, assign=True)
        model.cuda().bfloat16().eval().requires_grad_(False)
        for i, (entry, marker, identity) in enumerate(pending):
            if cancelled():
                raise InterruptedError('Sound adaptation preparation stopped')
            report(f"Preparing sound features {i + 1}/{len(pending)}: {entry['audio'].name}", 100 * i / len(pending))
            audio = _waveform(entry['audio'])
            parts = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for start in range(0, len(audio), 24000 * 30):
                    if cancelled():
                        raise InterruptedError('Sound feature preparation stopped')
                    chunk = audio[start:start + 24000 * 30]
                    if len(chunk) >= 24000:
                        output = model(torch.as_tensor(chunk[None], device='cuda'), output_hidden_states=True)
                        parts.append(output.hidden_states[20][0].float().cpu())
                        del output
            if not parts:
                raise ValueError('This recording is too short to prepare')
            features = F.interpolate(torch.cat(parts).T[None], size=round(len(audio) / 24000 * 25),
                                     mode='linear', align_corners=False)[0].T.numpy()
            entry['features'].parent.mkdir(parents=True, exist_ok=True)
            temporary = entry['features'].with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, features, allow_pickle=False)
            temporary.replace(entry['features'])
            marker.write_text(json.dumps(identity), encoding='utf-8')
    finally:
        model = None
        gc.collect(); torch.cuda.empty_cache()


def prepare_pair(project, *, report, cancelled):
    # Existing token/target preparation is reused; its head will only be replaced
    # when a saved adaptation pair is selected into a new project.
    if not project.get('prepared'):
        prepare_project(project, report=report, cancelled=cancelled)
        from services.music_training import get_project
        project = get_project(project['id'])
    prepare_audio(project, report=report, cancelled=cancelled)
    manifest = json.loads(Path(__file__).with_name('pair_regularizer.json').read_text())
    acoustic = json.loads(Path(__file__).with_name('acoustic_regularizer.json').read_text())
    for entry in acoustic['files']:
        ensure_asset('acoustic:' + entry['path'], report=report, cancelled=cancelled)
    entries = []
    for track in project['tracks']:
        entries.append({'audio': Path(track['audio_path']), 'sha256': track['audio_sha256'],
                        'features': project_directory(project['id']) / 'pair_features' / (track['id'] + '.npy')})
    for entry in manifest['files']:
        path = ensure_asset('pair:' + entry['path'], report=report, cancelled=cancelled)
        entries.append({'audio': path, 'sha256': entry['sha256'], 'features': path.with_name('mert.npy')})
    prepare_features(entries, report=report, cancelled=cancelled)
    update_project(project['id'], pair_prepared=preparation_identity(project))


def load_pair_data(project, tokenizer):
    if project.get('pair_prepared') != preparation_identity(project):
        raise ValueError('Prepare sound-adaptation data first')
    directory = project_directory(project['id'])
    groups = {'artist': [], 'heldout': [], 'minted': [], 'minted_val': []}
    skipped = []
    def item(features, latents, audio, prefix, codes=None):
        x, z = norm_features(np.load(features, allow_pickle=False)), np.load(latents, mmap_mode='r', allow_pickle=False)
        if z.ndim != 2 or z.shape[1] != 64 or not np.isfinite(z).all() or abs(len(x) - len(z)) > 2:
            raise ValueError('The sound features and latent targets must share a 25 Hz timeline')
        n = min(len(x), len(z))
        if n < 128:
            return None
        if len(prefix) + 513 > 12288:
            raise ValueError('This recording has too much lyric text for sound adaptation')
        result = {'features': x[:n], 'latent': z[:n], 'audio_path': audio, 'prefix': prefix}
        if codes is not None:
            y = np.load(codes, allow_pickle=False)
            if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or len(y) < n or y.min() < 0 or y.max() >= 32768:
                raise ValueError('Invalid minted semantic tokens')
            result['codec'] = y[:n]
        return result
    for track in project['tracks']:
        if file_digest(Path(track['audio_path'])) != track['audio_sha256']:
            raise ValueError('A source recording changed after preparation')
        if json.loads((directory / 'audio_targets' / (track['id'] + '.json')).read_text()) != target_identity(project, track):
            raise ValueError('Reprepare the source sound targets')
        prefix = token_prefixes(SongRequest(style=project['trigger'] + ', ' + track['style'], lyrics=track['lyrics'], cot='off'), tokenizer)
        row = item(directory / 'pair_features' / (track['id'] + '.npy'), directory / 'audio_targets' / (track['id'] + '.npy'), Path(track['audio_path']), prefix)
        if row is None or not track['holdout'] and len(row['latent']) < 512:
            skipped.append(track['name']); continue
        groups['heldout' if track['holdout'] else 'artist'].append(row)
    manifest = json.loads(Path(__file__).with_name('pair_regularizer.json').read_text())
    for track in manifest['tracks']:
        path = ROOT / 'acoustic_regularizer' / track['id']
        request = json.loads((path / 'request.json').read_text())
        prefix = token_prefixes(SongRequest(style=request['style'], lyrics=request['lyrics'], cot='off'), tokenizer)
        row = item(path / 'mert.npy', path / 'latent.npy', path / 'audio.flac', prefix, path / 'semantic.npy')
        if row is None:
            raise ValueError('A regularizer recording is too short')
        groups['minted_val' if track['holdout'] else 'minted'].append(row)
    if any(not rows for rows in groups.values()):
        raise ValueError('Sound adaptation needs a training excerpt of at least 20.5 seconds and a separate check excerpt of at least 5.2 seconds')
    return groups, skipped
