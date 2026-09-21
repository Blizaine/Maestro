"""Fixed, bounded listening requests and checkpoint-by-checkpoint orchestration."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import time

from . import music_training as projects


def audition_options(raw):
    if raw is None:
        return {"enabled": False}
    if not isinstance(raw, dict) or not isinstance(raw.get('enabled', False), bool):
        raise ValueError('Expected checkpoint audition settings')
    if not raw.get('enabled'):
        return {'enabled': False}
    style, lyrics = str(raw.get('style') or '').strip(), str(raw.get('lyrics') or '').strip()
    if not style or len(style) > 2000 or not lyrics or len(lyrics) > 4000:
        raise ValueError('Give the audition a style caption and short new lyrics (or [Instrumental])')
    for key, low, high, default in (('seconds', 8, 60, 30), ('seed', 0, 2**32 - 1, 22005)):
        value = raw.get(key, default)
        if isinstance(value, bool) or not str(value).isdigit() or not low <= int(value) <= high:
            raise ValueError(f'Audition {key} must be a whole number between {low} and {high}')
    try:
        strength = float(raw.get('strength', 1))
    except (TypeError, ValueError) as error:
        raise ValueError('Audition strength must be between 0 and 1.5') from error
    if not math.isfinite(strength) or not 0 <= strength <= 1.5:
        raise ValueError('Audition strength must be between 0 and 1.5')
    # These are explicit rather than inherited from mutable Studio controls.
    return {'enabled': True, 'style': style, 'lyrics': lyrics,
            'seconds': int(raw.get('seconds', 30)), 'seed': int(raw.get('seed', 22005)), 'strength': strength,
            'sampling_steps': 32, 'temperature': .8, 'top_k': 64, 'top_p': .95,
            'guide_scale': 1, 'model_mode': 2, 'vae_tile_size': 512}


def request_fingerprint(project, settings):
    return hashlib.sha256(json.dumps({'trigger': project['trigger'], 'settings': settings},
                                    sort_keys=True).encode()).hexdigest()[:16]


def save_audition(project_id, record):
    # All writers run under the project's worker ownership. Preserve samples
    # from other checkpoints and requests when one sample is retried.
    project = projects.get_project(project_id)
    records = [row for row in project.get('auditions', []) if row['id'] != record['id']]
    return projects.update_project(project_id, auditions=records + [record])


def render_checkpoint(project, branch, checkpoint, settings, *, report, cancelled):
    from models.TTS.yue2.training_audition import render_audition
    from .music_styles import file_digest
    settings = audition_options(settings)
    if not settings['enabled']:
        raise ValueError('Set up a fixed audition request first')
    field = {'audio': 'audio_checkpoints', 'joint': 'joint_checkpoints'}.get(branch, 'checkpoints')
    if branch not in {'style', 'audio', 'joint'}:
        raise ValueError('Choose a style or audio checkpoint')
    row = next((item for item in project.get(field, []) if item['file'] == checkpoint), None)
    if row is None:
        raise ValueError('Choose a checkpoint saved by this project')
    directory = projects.project_directory(project['id'])
    path = (directory / field / checkpoint).resolve()
    if path.parent != directory / field:
        raise ValueError('Invalid checkpoint path')
    digest = file_digest(path)
    if branch == 'joint':
        from models.TTS.yue2.training_audition import checkpoint_sources
        sources = checkpoint_sources(project, branch, row, report=report, cancelled=cancelled)
        digest = hashlib.sha256((digest + file_digest(sources['nar'])).encode()).hexdigest()
    fingerprint = request_fingerprint(project, settings)
    identity = f'{branch}-{row["step"]}-{fingerprint}-{digest[:12]}'
    record = {'id': identity, 'branch': branch, 'step': row['step'], 'checkpoint': checkpoint,
              'checkpoint_sha256': digest, 'request_id': fingerprint, 'settings': settings,
              'created_at': time.time(), 'status': 'running'}
    save_audition(project['id'], record)
    try:
        if cancelled():
            raise InterruptedError('Checkpoint audition cancelled; training checkpoint is saved')
        result = render_audition(project, branch, row, settings, identity, report=report, cancelled=cancelled)
        if cancelled():
            raise InterruptedError('Checkpoint audition cancelled; training checkpoint is saved')
        actual = file_digest(path)
        if branch == 'joint':
            actual = hashlib.sha256((actual + file_digest(sources['nar'])).encode()).hexdigest()
        if actual != digest:
            raise ValueError('The checkpoint changed while its audition was rendering')
        record.update(result, status='completed')
        return record
    except Exception as error:
        record.update(status='cancelled' if isinstance(error, InterruptedError) else 'failed', error=str(error)[:2000])
        raise
    finally:
        save_audition(project['id'], record)


def train_with_auditions(project_id, options, trainer, branch, *, report, cancelled):
    settings = audition_options(options.get('audition'))
    if not settings['enabled']:
        trainer(projects.get_project(project_id), options, report=report, cancelled=cancelled)
        return 0
    field = {'audio': 'audio_completed_steps', 'joint': 'joint_completed_steps'}.get(branch, 'completed_steps')
    history_field = {'audio': 'audio_checkpoints', 'joint': 'joint_checkpoints'}.get(branch, 'checkpoints')
    resume, failures = options['resume'], 0
    while True:
        project = projects.get_project(project_id)
        before = project.get(field, 0) if resume else 0
        if before >= options['steps']:
            return failures
        if cancelled():
            raise InterruptedError('Music training stopped; saved checkpoints are available to resume')
        projects.update_project(project_id, status='training')
        trainer(project, {**options, 'resume': resume}, report=report, cancelled=cancelled, pause_at_checkpoint=True)
        # The trainer has returned and its model/optimizer frame is gone. Clear
        # cycles before inference loads anything onto the same GPU.
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        project = projects.get_project(project_id)
        completed = project.get(field, 0)
        if completed <= before:
            raise RuntimeError('Training did not save a new checkpoint; resume the saved state to continue')
        if cancelled():
            raise InterruptedError('Music training stopped; checkpoint saved')
        row = next(item for item in project[history_field] if item['step'] == completed)
        projects.update_project(project_id, status='auditioning')
        report(f'Rendering fixed audition for {branch} step {completed}', completed / options['steps'] * 100)
        try:
            render_checkpoint(project, branch, row['file'], settings, report=lambda message, percent=None: report(message), cancelled=cancelled)
        except InterruptedError:
            raise
        except Exception as error:
            failures += 1
            report(f'Step {completed} saved; audition failed: {error}. Continuing training.')
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        resume = True
        if completed >= options['steps']:
            return failures
