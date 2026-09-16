"""Optional, confidence-masked lyric timing and AR cursor supervision.

Uses torchaudio's released Hybrid Demucs and MMS_FA weights. Source text and
recordings are read-only; cache identity includes both and the pinned assets.
"""
import gc
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F

from .audio_training_data import read_stereo
from .music_assets import ASSETS, ensure_asset
from .protocol import INSTRUCTIONS
from services.music_training import project_directory, update_project
from services.music_styles import file_digest

ALIGNMENT_VERSION = 'mms-fa-hdemucs-v1'
MIN_SCORE = .12


def lyric_words(lyrics):
    tags = [match.span() for match in re.finditer(r'\[[^\]]*\]', lyrics)]
    words, unsupported = [], 0
    for match in re.finditer(r"[\w’']+", lyrics):
        if any(start <= match.start() < end for start, end in tags):
            continue
        normalized = unicodedata.normalize('NFKD', match.group().lower().replace('’', "'"))
        value = re.sub("[^a-z']", '', normalized)
        if not re.search('[a-z]', value) or any(unicodedata.category(char).startswith('L') and ord(char) > 127 for char in normalized):
            unsupported += 1
            continue
        words.append((match.start(), match.end(), value))
    if unsupported > max(2, len(words) * .05):
        raise ValueError('Automatic lyric alignment currently supports Latin-script lyrics; spell out numbers and review unsupported words')
    return words


def identity(project, track):
    return {'version': ALIGNMENT_VERSION, 'dataset_digest': project['dataset_digest'],
            'audio_sha256': track['audio_sha256'], 'lyrics_sha256': hashlib.sha256(track['lyrics'].encode()).hexdigest(),
            'aligner_sha256': ASSETS['lyric_aligner']['sha256'], 'separator_sha256': ASSETS['vocal_separator']['sha256'],
            'emission_core_seconds': 20, 'emission_halo_seconds': 1}


def validate_words(rows, lyrics, duration):
    previous = 0.0
    for start, end, score, c0, c1 in rows:
        if (not all(np.isfinite(v) for v in (start, end, score)) or start < previous or end < start
                or end > duration + .04 or not 0 <= score <= 1 or not 0 <= c0 < c1 <= len(lyrics)):
            raise ValueError('Lyric alignment contains invalid or non-monotonic word positions')
        previous = start
    accepted = [r for r in rows if r[2] >= MIN_SCORE and .02 <= r[1] - r[0] <= 2.5]
    return len(accepted) / max(1, len(rows))


@torch.inference_mode()
def vocal_stem(model, audio, cancelled):
    # Fixed overlapping crops avoid full-song separator activation peaks.
    count = len(audio)
    output, weights = np.zeros(count, np.float32), np.zeros(count, np.float32)
    core, overlap = 8 * 44100, 44100
    for start in range(0, count, core - overlap):
        if cancelled():
            raise InterruptedError('Lyric preparation cancelled')
        part = torch.tensor(audio[start:start+core].T[None], device='cuda')
        mean, std = part.mean(), part.std().clamp_min(1e-6)
        # Match the public separation recipe's input normalization.
        separated = model((part - mean) / std)[0, 3] * std + mean
        vocals = separated.mean(0).float().cpu().numpy()
        ramp = np.ones(len(vocals), np.float32)
        fade = min(overlap, len(vocals))
        if start:
            ramp[:fade] *= np.linspace(0, 1, fade)
        if start + len(vocals) < count:
            ramp[-fade:] *= np.linspace(1, 0, fade)
        output[start:start+len(vocals)] += vocals * ramp
        weights[start:start+len(vocals)] += ramp
        if start + len(vocals) == count:
            break
    return output / weights.clip(1e-6)


@torch.inference_mode()
def emissions(model, audio, normalize, cancelled):
    signal = torch.tensor(audio, device='cpu')[None]
    if normalize:
        signal = F.layer_norm(signal, signal.shape)
    frames = (signal.shape[1] - 400) // 320 + 1
    result = []
    for start in range(0, frames, 1000):
        if cancelled():
            raise InterruptedError('Lyric alignment cancelled')
        left, end = max(0, start - 50), min(frames, start + 1000)
        right = min(signal.shape[1], (end + 50) * 320 + 80)
        logits, _ = model(signal[:, left * 320:right].cuda())
        result.append(logits[:, start-left:end-left].float().log_softmax(-1).cpu())
    return torch.cat(result, 1)


def align_project(project, *, report, cancelled):
    import torchaudio
    import soundfile as sf
    from scipy.signal import resample_poly
    directory = project_directory(project['id']) / 'alignment'
    directory.mkdir(exist_ok=True)
    separator = aligner = None
    summaries = []
    try:
        paths = {name: ensure_asset(name, report=report, cancelled=cancelled) for name in ('vocal_separator', 'lyric_aligner')}
        for index, track in enumerate(project['tracks']):
            expected = identity(project, track)
            marker = directory / (track['id'] + '.json')
            if file_digest(Path(track['audio_path'])) != track['audio_sha256']:
                raise ValueError('A source recording changed; create a new project')
            if marker.exists():
                cached = json.loads(marker.read_text())
                if cached.get('identity') == expected:
                    summaries.append(cached['summary']); continue
            words = lyric_words(track['lyrics'])
            if not words:
                summary = {'track_id': track['id'], 'status': 'instrumental', 'coverage': 0, 'words': 0}
                rows = []
            else:
                report(f"Separating vocals: {track['name']}", 100 * index / len(project['tracks']))
                with torch.device('cpu'):
                    separator = torchaudio.models.hdemucs_high(sources=['drums', 'bass', 'other', 'vocals'])
                separator.load_state_dict(torch.load(paths['vocal_separator'], map_location='cpu', weights_only=True))
                separator.cuda().eval()
                stem = vocal_stem(separator, read_stereo(track['audio_path'], 44100), cancelled)
                separator = None; gc.collect(); torch.cuda.empty_cache()
                sf.write(directory / (track['id'] + '-vocals.wav'), stem, 44100, subtype='FLOAT')
                audio = resample_poly(stem, 160, 441).astype(np.float32)
                bundle = torchaudio.pipelines.MMS_FA
                with torch.device('cpu'):
                    aligner = torchaudio.models.wav2vec2_model(**bundle._params)
                state = torch.load(paths['lyric_aligner'], map_location='cpu', weights_only=True)
                axes = bundle._remove_aux_axis
                if axes:
                    for key in ('aux.weight', 'aux.bias'):
                        state[key] = torch.stack([value for i, value in enumerate(state[key]) if i not in axes])
                aligner.load_state_dict(state); del state
                aligner.cuda().eval()
                report(f"Aligning lyrics: {track['name']}", 100 * (index + .5) / len(project['tracks']))
                emission = emissions(aligner, audio, bundle._normalize_waveform, cancelled)
                aligner = None; gc.collect(); torch.cuda.empty_cache()
                labels = bundle.get_dict(star=None)
                tokens = [[labels[c] for c in word] for _, _, word in words]
                flat = [token for word in tokens for token in word]
                if len(flat) >= emission.shape[1]:
                    raise ValueError('The supplied lyrics are too long for this recording')
                # CPU CTC alignment keeps full-song dynamic-programming memory off the GPU.
                aligned, scores = torchaudio.functional.forced_align(emission, torch.tensor([flat], device='cpu'), blank=0)
                spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
                if len(spans) != len(flat):
                    raise ValueError('Could not align every lyric character; review the supplied lyrics')
                rows, offset = [], 0
                for (c0, c1, _), token_ids in zip(words, tokens):
                    chunk = spans[offset:offset+len(token_ids)]; offset += len(token_ids)
                    rows.append([chunk[0].start * .02, chunk[-1].end * .02,
                                 float(np.mean([float(s.score) for s in chunk])), c0, c1])
                coverage = validate_words(rows, track['lyrics'], len(audio) / 16000)
                summary = {'track_id': track['id'], 'coverage': coverage, 'words': len(words),
                           'status': 'ready' if coverage >= .6 else 'needs_review'}
            temporary = marker.with_suffix('.tmp')
            temporary.write_text(json.dumps({'identity': expected, 'summary': summary, 'words': rows}), encoding='utf-8')
            temporary.replace(marker)
            summaries.append(summary)
        training_ids = {t['id'] for t in project['tracks'] if not t['holdout']}
        ready = any(s['status'] == 'ready' and s['track_id'] in training_ids for s in summaries)
        ready = ready and all(s['status'] != 'needs_review' for s in summaries if s['track_id'] in training_ids)
        update_project(project['id'], alignment={'version': ALIGNMENT_VERSION, 'ready': ready, 'tracks': summaries})
    finally:
        separator = aligner = None
        gc.collect(); torch.cuda.empty_cache()


def cursor_targets(project, track, prefix, tokenizer, frames):
    data = json.loads((project_directory(project['id']) / 'alignment' / (track['id'] + '.json')).read_text())
    if data['identity'] != identity(project, track):
        raise ValueError('Lyric alignment is stale; prepare it again')
    if data['summary']['status'] == 'instrumental':
        return None
    if data['summary']['status'] != 'ready':
        if track['holdout']:
            return None
        raise ValueError('Review the lyric alignment before training')
    text = track['lyrics']
    header = f"{INSTRUCTIONS['off']}\n[Tags]\n{project['trigger']}, {track['style']}\n[Lyrics]\n"
    head_ids, full_ids = tokenizer.encode(header), tokenizer.encode(header + text + '\n')
    if full_ids[:len(head_ids)] != head_ids or prefix[1:len(full_ids)+1] != full_ids:
        raise ValueError('The lyric tokenizer prefix does not match alignment; refusing incorrect supervision')
    lyric_ids = full_ids[len(head_ids):]
    decoded, starts = tokenizer._enc.decode_with_offsets(lyric_ids)
    if decoded != unicodedata.normalize('NFC', text + '\n'):
        raise ValueError('The decoded lyrics do not match the alignment text')
    ends = starts[1:] + [len(decoded)]
    rows, columns, weights = [], [], []
    for start, end, score, c0, c1 in data['words']:
        if score < MIN_SCORE or not .02 <= end - start <= 2.5:
            continue
        c0 = len(unicodedata.normalize('NFC', text[:c0]))
        c1 = len(unicodedata.normalize('NFC', text[:c1]))
        tokens = [i for i, (a, b) in enumerate(zip(starts, ends)) if a < c1 and b > c0]
        if not tokens:
            continue
        # No carry-forward across long intros or gaps: only confident word spans.
        for frame in range(max(0, int(np.ceil(start * 25))), min(frames, int(np.ceil(end * 25)))):
            rows.extend([frame] * len(tokens)); columns.extend(tokens); weights.extend([1 / len(tokens)] * len(tokens))
    return {'start': 1 + len(head_ids), 'end': 1 + len(full_ids), 'rows': rows, 'columns': columns, 'weights': weights}


def cursor_loss(hidden, prefix_length, targets, head):
    frames = len(hidden) - prefix_length
    rows = torch.tensor(targets['rows'], device=hidden.device, dtype=torch.long)
    valid = rows < frames
    if not valid.any():
        return hidden.new_zeros((), dtype=torch.float32)
    rows = rows[valid]
    columns = torch.tensor(targets['columns'], device=hidden.device, dtype=torch.long)[valid]
    weights = torch.tensor(targets['weights'], device=hidden.device, dtype=torch.float32)[valid]
    unique, inverse = torch.unique(rows, return_inverse=True)
    q = head(hidden[prefix_length - 1 + unique].float())
    k = hidden[targets['start']:targets['end']].float()
    scores = (q @ k.T) * (q.shape[-1] ** -.5)
    logp = scores.log_softmax(-1)
    return -(logp[inverse, columns] * weights).sum() / len(unique)
