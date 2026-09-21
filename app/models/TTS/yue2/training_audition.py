"""Render a saved checkpoint through the production YuE2 generator, serially."""
from __future__ import annotations

import gc
import json

from services.music_styles import file_digest
from services.music_training import project_directory
from services.music_contracts import tokenizer_pair, pair_asset


def checkpoint_adapter_mode(branch, row):
    mode = row.get('adapter_mode', 'joint') if branch == 'joint' else 'separate'
    if mode not in {'separate', 'joint'}:
        raise ValueError('Unsupported checkpoint adapter mode')
    return mode


def checkpoint_sources(project, branch, row, *, cancelled, report):
    from .music_assets import ensure_asset
    directory = project_directory(project['id'])
    if branch == 'joint':
        sources = {key: directory / 'joint_checkpoints' / row['file' if key == 'ar' else 'nar_file'] for key in ('ar', 'nar')}
        for key, path in sources.items():
            if path.resolve().parent != directory / 'joint_checkpoints' or file_digest(path) != row[key + '_sha256']:
                raise ValueError('The joint checkpoint pair changed; cannot audition mismatched adapters')
        return sources
    if branch == 'style':
        return {'ar': directory / 'checkpoints' / row['file'],
                'nar': pair_asset(project, 'nar', cancelled=cancelled, report=report)}
    sources = {'nar': directory / 'audio_checkpoints' / row['file']}
    conditioning = row.get('conditioning_checkpoint', '')
    if conditioning:
        if conditioning not in {item['file'] for item in project.get('checkpoints', [])}:
            raise ValueError('The audio checkpoint is missing its paired style checkpoint')
        sources['ar'] = directory / 'checkpoints' / conditioning
        expected = project.get('audio_training_options', {}).get('conditioning_sha256')
        if file_digest(sources['ar']) != expected:
            raise ValueError('The style checkpoint used during sound adaptation changed; cannot audition a mismatched pair')
    return sources


def render_audition(project, branch, row, settings, identity, *, report, cancelled):
    import soundfile as sf
    import torch
    import wgp
    from .artist_adapter import active_adapters

    def check():
        if cancelled():
            raise InterruptedError('Checkpoint audition cancelled; training checkpoint is saved')

    sources = checkpoint_sources(project, branch, row, cancelled=cancelled, report=report)
    check()
    pipeline = None
    directory = project_directory(project['id']) / 'auditions'
    directory.mkdir(exist_ok=True)
    path = directory / f'{identity}.flac'
    temporary = path.with_suffix('.tmp')
    try:
        report(f'Loading YuE2 to audition {branch} step {row["step"]}')
        wgp.wan_model, wgp.offloadobj = wgp.load_models('yue2', output_type='audio')
        pipeline = wgp.wan_model
        # Include queue cancellation in every token/flow/VAE cancellation hook.
        original_abort = pipeline._abort_requested
        pipeline._abort_requested = lambda: cancelled() or original_abort()
        pipeline.text_encoder.abort_fn = pipeline.transformer.abort_fn = pipeline._abort_requested

        def progress(**values):
            check()
            phase = values.get('denoising_extra', 'Generating audio')
            report(f'Audition step {row["step"]}: {phase}')

        style = settings['style']
        if project['trigger'].casefold() not in style.casefold():
            style = f"{project['trigger']}, {style}"
        with active_adapters(pipeline, sources, settings['strength'], mode=checkpoint_adapter_mode(branch, row)):
            result = pipeline.generate(input_prompt=settings['lyrics'], alt_prompt=style,
                seed=settings['seed'], duration_seconds=settings['seconds'],
                sampling_steps=settings['sampling_steps'], guide_scale=settings['guide_scale'],
                temperature=settings['temperature'], top_k=settings['top_k'], top_p=settings['top_p'],
                model_mode=settings['model_mode'], VAE_tile_size=settings['vae_tile_size'],
                callback=progress, offloadobj=wgp.offloadobj)
            check()
            if not result or result.get('x') is None:
                raise ValueError('YuE2 did not return an audition')
            audio = result['x'].float().cpu()
            if not torch.isfinite(audio).all():
                raise ValueError('The audition contains non-finite audio')
            sf.write(str(temporary), audio.clamp(-1, 1).T.numpy(), result['audio_sampling_rate'], format='FLAC', subtype='PCM_24')
        temporary.replace(path)
        details = {'file': path.name, 'seconds': audio.shape[-1] / result['audio_sampling_rate'],
                   'engine': pipeline.lm_decoder_engine, 'sample_rate': result['audio_sampling_rate'],
                   'sources': {key: file_digest(value) for key, value in sources.items()},
                   'metadata': result.get('artifact_metadata', {}), 'dataset_digest': project['dataset_digest']}
        path.with_suffix('.json').write_text(json.dumps({'settings': settings, **details}, indent=2), encoding='utf-8')
        return details
    finally:
        temporary.unlink(missing_ok=True)
        try:
            if pipeline is not None:
                pipeline.release()
        finally:
            wgp.release_model()
            pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
