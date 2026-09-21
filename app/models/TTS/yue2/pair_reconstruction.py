"""Listen to the complete adapted head/decoder, without any AR style adapter."""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from services.music_contracts import pair_asset
from services.music_pair_adaptation import checkpoint_pair
from services.music_styles import file_digest
from services.music_training import project_directory
from .artist_adapter import active_adapters
from .audio_training_data import read_stereo
from .music_tokenizer import RealAudioHead, predict_codes
from .pair_training import head_weights
from .protocol import CODEC_OFFSET, SongRequest, token_prefixes


def reconstruct_pair(project, options, output_dir, job_id, *, report, cancelled, publish):
    import wgp
    row, after = checkpoint_pair(project, options['checkpoint'])
    if any(row[key] != options[key] for key in ('head_sha256', 'nar_sha256')):
        raise ValueError('The sound pair changed while comparison was queued')
    before = {key: pair_asset(project, key, report=report, cancelled=cancelled) for key in ('head', 'nar')}
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    stem = f'music-sound-pair-{job_id}'
    files, records = [], []
    head = pipeline = None

    def check():
        if cancelled():
            raise InterruptedError('Sound comparison cancelled')

    def record(path, details):
        records.append({'file': path.name, **details}); files.append(path.name)
        path.with_suffix('.meta.json').write_text(json.dumps({
            'params': {'model_type': 'yue2', 'generation_mode': 'audio', 'seed': options['seed'],
                       'prompt': f"Sound comparison: {details['track']} — {details['variant']}"},
            'generation_mode': 'audio', 'job_id': job_id, 'created_at': time.time(),
            'output_filename': path.name, 'model_details': {'reconstruction': details}}, indent=2), encoding='utf-8')
        publish(files)

    report_path = directory / f'{stem}.json'
    try:
        tracks = [next(t for t in project['tracks'] if t['id'] == id) for id in options['track_ids']]
        # Tokenize with each head before loading the music generator. The full
        # excerpt features retain the author's per-recording normalization.
        codes = {}
        for variant, pair in (('before', before), ('after', after)):
            check(); report(f'Tokenizing sound comparison: {variant}', 0)
            head = RealAudioHead().cuda().eval().requires_grad_(False)
            head.load_state_dict(head_weights(pair['head']), strict=True)
            for track in tracks:
                check()
                if file_digest(Path(track['audio_path'])) != track['audio_sha256']:
                    raise ValueError('A source recording changed')
                features = np.load(project_directory(project['id']) / 'pair_features' / (track['id'] + '.npy'), allow_pickle=False)
                values = predict_codes(head, features, cancelled=cancelled)[:options['seconds'] * 25]
                codes[variant, track['id']] = values
            head = None; gc.collect(); torch.cuda.empty_cache()
        check(); report('Loading YuE2 for original / before / after comparison', 5)
        wgp.wan_model, wgp.offloadobj = wgp.load_models('yue2', output_type='audio')
        pipeline = wgp.wan_model
        pipeline.text_encoder.abort_fn = pipeline.transformer.abort_fn = cancelled
        for i, track in enumerate(tracks):
            count = min(len(codes[v, track['id']]) for v in ('before', 'after'))
            common = {'track': track['name'], 'track_id': track['id'], 'heldout': track['holdout'],
                      'project_id': project['id'], 'checkpoint': row['step'], 'seconds': count / 25,
                      'source_sha256': track['audio_sha256'], 'sample_rate': 48000, 'channels': 2}
            original = directory / f'{stem}-{i + 1}-original.wav'
            sf.write(original, read_stereo(Path(track['audio_path']))[:count * 1920], 48000, subtype='PCM_24')
            record(original, {**common, 'variant': 'original'})
            request = SongRequest(style=project['trigger'] + ', ' + track['style'], lyrics=track['lyrics'], cot='off', seed=options['seed'], cfg_scale=1)
            prefix = token_prefixes(request, pipeline.tokenizer)
            for j, (variant, pair) in enumerate((('before', before), ('after', after))):
                check(); report(f"Sound comparison {i + 1}/{len(tracks)}: {variant}", 10 + (i * 2 + j) / (2 * len(tracks)) * 85)
                values = codes[variant, track['id']][:count]
                with active_adapters(pipeline, {'nar': pair['nar']}, 1.0):
                    audio = pipeline.decode_codec(prefix, (values + CODEC_OFFSET).tolist(), options['seed'], options['steps'], 1024, lambda **kwargs: check())
                    if not torch.isfinite(audio).all():
                        raise ValueError('The sound comparison produced invalid audio')
                    path = directory / f'{stem}-{i + 1}-{variant}.wav'
                    sf.write(path, audio[0].float().cpu().clamp(-1, 1).T.numpy(), 48000, subtype='PCM_24')
                pipeline.last_latents = None
                record(path, {**common, 'variant': variant, 'seed': options['seed'], 'steps': options['steps'],
                              'head_sha256': file_digest(pair['head']), 'nar_sha256': file_digest(pair['nar']),
                              'codec_sha256': hashlib.sha256(values.astype('<i4').tobytes()).hexdigest()})
        return {'files': files, 'report': report_path.name}
    finally:
        report_path.write_text(json.dumps({'project_id': project['id'], 'options': options,
            'comparison': 'Each tokenizer with its matching decoder; no AR style adapter; fixed seed and recordings',
            'records': records}, indent=2), encoding='utf-8')
        head = None
        if pipeline is not None:
            pipeline.release()
        wgp.release_model(); gc.collect(); torch.cuda.empty_cache()
