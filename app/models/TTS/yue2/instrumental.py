"""Instrumental request rules and the optional, pinned AR adapter."""
from contextlib import contextmanager
import re


def is_instrumental(inputs, lyrics=None):
    custom = inputs.get('custom_settings') or {}
    return (inputs.get('_music_instrumental') is True
            or (isinstance(custom, dict) and custom.get('instrumental') is True)
            or str(lyrics if lyrics is not None else inputs.get('prompt', '')).strip().lower() == '[instrumental]')


def section_plan(lyrics):
    """Accept the author's three caption forms; never feed sung prose to this LoRA."""
    lines = str(lyrics or '').strip().splitlines()
    result = []
    previous_end = 0
    for line in lines:
        if not line.strip():
            continue
        match = re.fullmatch(
            r'\[(intro|verse|pre-chorus|chorus|bridge|outro)(?: (\d+):([0-5]\d)-(\d+):([0-5]\d))?\]',
            line.strip(), re.IGNORECASE)
        if not match:
            return '[instrumental]'
        tag, start_min, start_sec, end_min, end_sec = match.groups()
        if start_min is not None:
            start, end = int(start_min) * 60 + int(start_sec), int(end_min) * 60 + int(end_sec)
            if start < previous_end or end <= start:
                return '[instrumental]'
            previous_end = end
        result.append(line.strip().lower())
    return '\n'.join(result) or '[instrumental]'


def instrumental_settings(custom=None):
    # Keep the user's artist choices in saved settings, but they are inactive
    # for instrumental jobs. Automatic score planning starts from a clean score.
    return {**(custom or {}), 'instrumental': True, 'abc': ''}


@contextmanager
def active_instrumental(pipeline):
    from .music_assets import ASSETS, ensure_asset
    from .artist_adapter import active_adapters
    key = 'instrumental_ar'
    path = ensure_asset(key, cancelled=pipeline._abort_requested, report=print)
    print('[YuE2] Instrumental AR LoRA at strength 1; full score planning; stock acoustic decoder.')
    with active_adapters(pipeline, {'ar': path}, 1.0):
        yield {'repo_id': ASSETS[key]['repo_id'], 'revision': ASSETS[key]['revision'],
               'file': ASSETS[key]['remote_path'], 'sha256': ASSETS[key]['sha256'],
               'strength': 1.0, 'cot': 'full', 'decoder': 'stock'}
