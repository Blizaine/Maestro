"""Durable automatic training; stage execution belongs to the shared GPU queue."""
from . import music_training as projects

STAGES = (
    ('analyze-songs', 'Analyze songs and find the main voice'),
    ('build-dataset', 'Prepare the training dataset'),
    ('prepare-pair', 'Prepare voice & sound training'),
    ('adapt-pair', 'Learn voice & sound'),
    ('select-pair', 'Carry the learned sound into song style'),
    ('prepare', 'Prepare song-style training'),
    ('train', 'Learn song style'),
    ('publish', 'Save the finished LoRA'),
)


def auto_options(raw, project):
    if not isinstance(raw, dict):
        raise ValueError('Expected Auto training options')
    saved = project.get('auto_training') or {}
    if saved.get('status') == 'completed':
        raise ValueError('Auto training is complete. Open its voice or song-style project to train further.')
    if project.get('adapted_pair'):
        raise ValueError('This project already has a learned voice. Open its original voice project to use Auto training.')
    result = {}
    for key, default in (('voice_steps', 100), ('style_steps', 200)):
        value = raw.get(key, saved.get(key, default))
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1600:
            raise ValueError('Choose whole-number voice and song-style targets from 1 to 1600 steps')
        if saved and value != saved.get(key):
            raise ValueError('Resume uses the saved Auto targets. Train further in the resulting projects after it finishes.')
        result[key] = value
    return result


def update_auto(project_id, **changes):
    with projects._lock:
        plan = {**projects.get_project(project_id).get('auto_training', {}), **changes}
        projects.update_project(project_id, auto_training=plan)
        return plan


def linked_projects(project):
    plan = project.get('auto_training') or {}
    return {project['id'], *(plan[key] for key in ('sound_project_id', 'style_project_id') if plan.get(key))}


def execute_stage(operation, project_id, options, *, report, cancelled):
    """Import GPU code only when a queued stage actually runs."""
    project = projects.get_project(project_id)
    if operation == 'analyze-songs':
        from .music_dataset_analysis import analyze_songs
        return analyze_songs(project, report=report, cancelled=cancelled)
    if operation == 'build-dataset':
        from .music_dataset import build_dataset, dataset_selection
        selected = dataset_selection(project, automatic=True)
        if not any(not row['holdout'] and row['clip']['end'] - row['clip']['start'] >= 20.5 for row in selected):
            raise ValueError('Auto could not find a training excerpt of at least 20.5 seconds. Open prepared clips to adjust voice selection or add a longer excerpt, then resume Auto.')
        if not any(row['holdout'] and row['clip']['end'] - row['clip']['start'] >= 5.2 for row in selected):
            raise ValueError('Auto needs a check-only excerpt of at least 5.2 seconds. Open prepared clips to adjust the check-only song, then resume Auto.')
        result = build_dataset(project, automatic=True, report=report, cancelled=cancelled)
        update_auto(project_id, selection_summary={
            'training_clips': sum(not row['holdout'] for row in selected),
            'check_clips': sum(row['holdout'] for row in selected),
            'unreviewed_clips': sum(not row['clip']['reviewed'] for row in selected),
            'warnings': [f"{track['name']}: {warning}" for track in project['tracks']
                         for warning in project['preparation']['songs'][track['id']].get('warnings', [])],
        })
        return result
    if operation == 'prepare-pair':
        from models.TTS.yue2.pair_training_data import prepare_pair
        return prepare_pair(project, report=report, cancelled=cancelled)
    if operation == 'adapt-pair':
        from models.TTS.yue2.pair_training import train_pair
        return train_pair(project, options, report=report, cancelled=cancelled)
    if operation == 'select-pair':
        from .music_pair_adaptation import checkpoint_pair, select_pair
        row, _ = checkpoint_pair(project, options['checkpoint'])
        # Recover a fully copied child even if the app stopped before recording
        # its ID on the Auto run. Never adopt unrelated manual experiments.
        for child in projects.list_projects():
            pair = child.get('adapted_pair') or {}
            if (child.get('auto_training_root') == options['auto_root']
                    and pair.get('source_project') == project_id
                    and all(pair.get(key) == row[key] for key in ('step', 'head_sha256', 'nar_sha256'))):
                return child
        return select_pair(project_id, row['file'], auto_root=options['auto_root'])
    if operation == 'prepare':
        from models.TTS.yue2.music_tokenizer import prepare_project
        from .music_data_review import sequence_coverage
        prepare_project(project, report=report, cancelled=cancelled)
        return sequence_coverage(projects.get_project(project_id), report=report, cancelled=cancelled)
    if operation == 'train':
        from models.TTS.yue2.artist_training import train_project
        return train_project(project, options, report=report, cancelled=cancelled)
    if operation == 'publish':
        from .music_styles import list_styles, load_style, save_style
        from models.TTS.yue2.artist_adapter import read_upstream_adapter
        from .music_contracts import pair_asset
        for style in list_styles(include_archived=True):
            if (style.get('training') or {}).get('auto_root') == options['auto_root']:
                return load_style(style['id'], verify=True)
        row = _checkpoint(project, 'checkpoints', options['steps'])
        from pathlib import Path
        if Path(row['file']).name != row['file']:
            raise ValueError('Invalid saved song-style checkpoint')
        ar = read_upstream_adapter(projects.project_directory(project_id) / 'checkpoints' / row['file'], 'ar')
        nar = read_upstream_adapter(pair_asset(project, 'nar', report=report, cancelled=cancelled), 'nar')
        _check_cancel(cancelled)
        return save_style(options['name'][:65] + f" · voice-{project['adapted_pair']['step']} · style-{row['step']}",
            project['trigger'], ar, nar, pair=project.get('tokenizer_pair', 'v4'), adapted_pair=project['adapted_pair'],
            training={'auto_root': options['auto_root'], 'project_id': project_id,
                      'checkpoint': row['file'], 'dataset_digest': project['dataset_digest']})
    raise ValueError('Unknown automatic music stage')


def _checkpoint(project, field, target):
    rows = sorted((row for row in project.get(field, []) if row['step'] >= target), key=lambda row: row['step'])
    if not rows:
        raise ValueError('Training did not save the requested checkpoint. Resume Auto to continue from its saved progress.')
    return rows[0]


def _check_cancel(cancelled):
    if cancelled():
        raise InterruptedError('Auto training stopped. Completed preparation and checkpoints are saved; resume when ready.')


def run_auto(project_id, options, *, report, cancelled, claim, cleanup, execute=None):
    """Run within one queue slot. Never require a browser to advance the stages."""
    from .music_pair_adaptation import adaptation_options
    execute = execute or execute_stage

    def plan():
        return projects.get_project(project_id)['auto_training']

    def step(operation, target, body=None, *, skip=False, link=None):
        _check_cancel(cancelled)
        if operation in plan().get('done', []):
            return
        claim(target)
        index, label = next((i, label) for i, (key, label) in enumerate(STAGES) if key == operation)
        update_auto(project_id, stage=operation, status='running')
        if target != project_id:
            projects.update_project(target, status='training' if operation in {'adapt-pair', 'train'} else 'preparing')

        def progress(message, percent=None):
            # Do not raise from the trainer's progress callback: cancellation
            # must give the optimizer a chance to save its current step.
            if not cancelled():
                report(f'{label}: {message}', (index + max(0, min(100, percent or 0)) / 100) / len(STAGES) * 100)
        progress('using saved work' if skip else 'starting', 0)
        try:
            result = None if skip else execute(operation, target, body or {}, report=progress, cancelled=cancelled)
            _check_cancel(cancelled)
            links = {link: result['id']} if link else {}
            update_auto(project_id, done=[*plan().get('done', []), operation], **links)
        finally:
            cleanup()

    root = projects.get_project(project_id)
    claim(project_id)
    if not plan().get('sound_project_id'):
        if root.get('preparation_draft'):
            step('analyze-songs', project_id)
            step('build-dataset', project_id, link='sound_project_id')
        else:
            update_auto(project_id, sound_project_id=project_id, done=['analyze-songs', 'build-dataset'])
    sound_id = plan()['sound_project_id']
    claim(sound_id)
    projects.update_project(sound_id, training_workflow='author')
    sound = projects.get_project(sound_id)
    step('prepare-pair', sound_id, skip=bool(sound.get('pair_prepared')))
    sound = projects.get_project(sound_id)
    voice_options = adaptation_options({'steps': options['voice_steps'],
        'resume': bool(sound.get('pair_resume_available')), 'seed': sound.get('pair_training_options', {}).get('seed', 22005)})
    step('adapt-pair', sound_id, voice_options, skip=sound.get('pair_completed_steps', 0) >= options['voice_steps'])
    if not plan().get('style_project_id'):
        checkpoint = _checkpoint(projects.get_project(sound_id), 'pair_checkpoints', options['voice_steps'])
        step('select-pair', sound_id, {'checkpoint': checkpoint['file'], 'auto_root': project_id}, link='style_project_id')
    style_id = plan()['style_project_id']
    claim(style_id)
    projects.update_project(style_id, training_workflow='style')
    style = projects.get_project(style_id)
    step('prepare', style_id, skip=bool(style.get('prepared')))
    style = projects.get_project(style_id)
    stored = style.get('training_options') or {}
    train_options = projects.training_options({'steps': options['style_steps'], 'rank': stored.get('rank', 64),
        'seed': stored.get('seed', 22005), 'learning_rate': stored.get('learning_rate', .0001),
        'lyric_alignment': stored.get('lyric_alignment', False), 'resume': bool(style.get('resume_available'))})
    step('train', style_id, train_options, skip=style.get('completed_steps', 0) >= options['style_steps'])
    step('publish', style_id, {'auto_root': project_id, 'name': root['name'], 'steps': options['style_steps']}, link='style_id')
    _check_cancel(cancelled)
    return plan()
