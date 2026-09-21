"""Immutable head/decoder selection and validated options for author adaptation."""
from pathlib import Path
import shutil

from . import music_training as projects
from .music_contracts import tokenizer_pair
from .music_styles import file_digest

OBJECTIVE = 'head-nar-waveform-v1'
AUTHOR_REVISION = 'e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5'


def adaptation_options(raw):
    if not isinstance(raw, dict):
        raise ValueError('Expected sound adaptation settings')
    base = projects.training_options({'steps': 100, 'rank': 32, **raw})
    return {key: base[key] for key in ('steps', 'seed', 'resume')} | {
        'checkpoint_every': min(25, base['steps']), 'window_frames': 512,
        'head_learning_rate': 1e-4, 'nar_learning_rate': 5e-5, 'io_learning_rate': 2e-5,
        'audio_weight': 4.0, 'audio_tmax': .4, 'audio_frames': 150, 'audio_margin': 25,
        'anchor_batch': 16, 'anchor_microbatch': 2, 'schedule_steps': 3000,
    }


def checkpoint_pair(source, checkpoint):
    row = next((r for r in source.get('pair_checkpoints', []) if r['file'] == checkpoint), None)
    if not row:
        raise ValueError('Choose a saved sound-adaptation checkpoint')
    directory = projects.project_directory(source['id']) / 'pair_checkpoints'
    paths = {}
    for branch in ('head', 'nar'):
        name = row['file' if branch == 'head' else 'nar_file']
        path = directory / name
        if Path(name).name != name or path.resolve().parent != directory or file_digest(path) != row[branch + '_sha256']:
            raise ValueError('The saved tokenizer/decoder pair has changed')
        paths[branch] = path
    return row, paths


def select_pair(project_id, checkpoint, *, auto_root=None):
    """Publish a separate project; existing tokens and AR styles never change."""
    source = projects.get_project(project_id)
    if source.get('status') in {'queued', 'preparing', 'training', 'auditioning'} and not (
        auto_root and source.get('auto_training_root') == auto_root
    ):
        raise ValueError('Wait for sound adaptation to finish before choosing its checkpoint')
    row, paths = checkpoint_pair(source, checkpoint)
    child = projects.create_project(source['name'][:65] + f" · adapted {row['step']}", source['trigger'],
                                    source['tracks'], pair=source.get('tokenizer_pair', 'v4'))
    target = projects.project_directory(child['id']) / 'adapted_pair'
    target.mkdir()
    for branch, path in paths.items():
        shutil.copyfile(path, target / (branch + '.safetensors'))
    return projects.update_project(child['id'],
        adapted_pair={key: row[key] for key in ('head_sha256', 'nar_sha256')} | {
            'source_project': project_id, 'step': row['step'], 'objective': OBJECTIVE},
        reviews=source.get('reviews', {}), sequence_coverage=source.get('sequence_coverage', []),
        **({'auto_training_root': auto_root} if auto_root else {}),
        message='Sound pair selected. Prepare fresh music tokens, then train song style with this adapted sound.')


def comparison_options(raw, project):
    row, _ = checkpoint_pair(project, raw.get('checkpoint'))
    if not project.get('pair_prepared'):
        raise ValueError('Prepare sound features before comparing a pair')
    # Prefer one genuinely unseen excerpt; also compare one practice excerpt.
    chosen = [next((t['id'] for t in project['tracks'] if t['holdout'] == heldout), None)
              for heldout in (True, False)]
    return {'checkpoint': row['file'], 'head_sha256': row['head_sha256'], 'nar_sha256': row['nar_sha256'],
            'track_ids': [i for i in chosen if i], 'seconds': 30, 'seed': 22005, 'steps': 32}
