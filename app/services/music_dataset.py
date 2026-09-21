"""Reviewed song preparation, kept separate from immutable training datasets."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid

from . import music_training as projects
from .music_styles import file_digest
from .music_contracts import tokenizer_pair

VERSION = 1
ACTIVE = {'queued', 'preparing', 'training', 'auditioning'}
DELIVERY = {'unspecified', 'rap', 'sung', 'mixed'}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def check_cancel(cancelled):
    if cancelled():
        raise InterruptedError('Song preparation stopped. Completed analysis and clips are saved; resume when ready.')


def run_media(arguments, cancelled=lambda: False):
    """Bounded, cancellable media conversion without shell parsing or pipe deadlocks."""
    executable = shutil.which('ffmpeg')
    if not executable:
        raise ValueError('ffmpeg is required to prepare songs')
    check_cancel(cancelled)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([executable, '-hide_banner', '-nostdin', '-loglevel', 'error', '-y', *arguments],
                                   stdout=subprocess.DEVNULL, stderr=errors,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            started = time.monotonic()
            while True:
                check_cancel(cancelled)
                if time.monotonic() - started > 1800:
                    raise ValueError('Audio conversion timed out')
                try:
                    code = process.wait(timeout=.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if code:
                errors.seek(0)
                raise ValueError('Could not read this audio recording: ' + errors.read(1500).decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def audio_duration(path):
    executable = shutil.which('ffprobe')
    if not executable:
        raise ValueError('ffprobe is required to inspect songs')
    try:
        result = subprocess.run([executable, '-v', 'error', '-show_entries', 'format=duration',
                                 '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
                                capture_output=True, text=True, timeout=30, check=True,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        seconds = float(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError) as error:
        raise ValueError('Could not read the duration of this song') from error
    if not math.isfinite(seconds) or not 1 <= seconds <= 1200:
        raise ValueError('Use songs between 1 second and 20 minutes long')
    return seconds


def create_draft(name, trigger, tracks, *, pair='v9'):
    tokenizer_pair({'tokenizer_pair': pair})
    name, trigger = str(name or '').strip(), str(trigger or '').strip()
    if not name or len(name) > 100 or not trigger or len(trigger) > 200:
        raise ValueError('Enter a name and a short, distinctive style trigger')
    if not isinstance(tracks, list) or not 2 <= len(tracks) <= 50:
        raise ValueError('Choose 2–50 different songs, including a held-out song')
    normalized, seen = [], set()
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError('Each song needs an audio file')
        path = Path(str(track.get('audio_path') or '')).resolve()
        if not path.is_file() or path.suffix.lower() not in projects.AUDIO_EXTENSIONS or path.stat().st_size > 1024**3:
            raise ValueError('Choose supported audio recordings smaller than 1 GB each')
        digest = file_digest(path)
        if digest in seen:
            raise ValueError('Use different original songs for training and held-out evaluation')
        seen.add(digest)
        style = str(track.get('style') or '').strip()
        if len(style) > 1800:
            raise ValueError('Keep song descriptions under 1800 characters')
        normalized.append({'id': digest[:16], 'audio_path': str(path), 'audio_sha256': digest,
                           'name': str(track.get('name') or path.stem)[:200], 'duration': audio_duration(path),
                           'lyrics': '', 'style': style, 'holdout': track.get('holdout') is True})
    if not any(t['holdout'] for t in normalized) or all(t['holdout'] for t in normalized):
        raise ValueError('Keep at least one training song and one separate held-out song')
    project = {'version': 1, 'id': uuid.uuid4().hex[:16], 'name': name, 'trigger': trigger,
               'tokenizer_pair': pair, 'tracks': normalized, 'dataset_digest': signature(normalized),
               'created_at': time.time(), 'status': 'draft', 'progress': 0, 'checkpoints': [],
               'preparation_draft': True, 'preparation': {'version': VERSION, 'revision': 0, 'songs': {}},
               'message': 'Ready to analyze voices and lyrics'}
    with projects._lock:
        projects._write(project)
    return project


def require_draft(project, *, editable=False):
    if not project.get('preparation_draft'):
        raise ValueError('Choose a song preparation draft')
    if editable and project.get('status') in ACTIVE:
        raise ValueError('Wait for song preparation to finish stopping before editing this draft')


def song_directory(project, track_id):
    if not any(track['id'] == track_id for track in project['tracks']):
        raise ValueError('Choose a song in this draft')
    return projects.project_directory(project['id']) / 'song_preparation' / track_id


def assign_speakers(words, turns):
    """Tag timed words conservatively; never infer an artist's identity."""
    tagged = []
    for source in words:
        word = dict(source)
        start, end = float(word['start']), float(word['end'])
        totals = {}
        for turn in turns:
            overlap = max(0., min(end, turn['end']) - max(start, turn['start']))
            if overlap:
                totals[turn['speaker']] = totals.get(turn['speaker'], 0.) + overlap
        ranked = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
        length = max(.02, end - start)
        word['speaker'] = ranked[0][0] if ranked and ranked[0][1] >= length * .35 else 'unknown'
        # Adjacent turns inside a word are ambiguous too, even without simultaneous speech.
        word['overlap'] = len(ranked) > 1 and ranked[1][1] >= length * .25
        word['uncertain'] = bool(word.get('uncertain')) or word.get('probability', 0.) < .5
        tagged.append(word)
    return tagged


def voice_summary(words):
    voices = {}
    for word in words:
        key = word['speaker']
        row = voices.setdefault(key, {'id': key, 'word_count': 0, 'confident_words': 0, 'seconds': 0.,
                                      'start': word['start'], 'end': word['end']})
        row['word_count'] += 1
        row['confident_words'] += int(not word['uncertain'] and not word['overlap'])
        row['seconds'] += max(0., word['end'] - word['start'])
    rows = sorted(voices.values(), key=lambda row: (-row['confident_words'], -row['word_count'], row['id']))
    for index, row in enumerate(rows):
        row['label'] = 'Unassigned voice' if row['id'] == 'unknown' else f'Voice {index + 1}'
        # Find a clean phrase for auditioning this anonymous cluster.
        candidates = [word for word in words if word['speaker'] == row['id'] and not word['overlap']]
        if candidates:
            first = next((word for word in candidates if not word['uncertain']), candidates[0])
            row['start'] = max(0., first['start'] - .15)
            row['end'] = first['end']
            for word in words:
                if word['start'] < first['start']:
                    continue
                if word['speaker'] != row['id'] or word['overlap'] or word['end'] - row['start'] > 8:
                    break
                row['end'] = word['end'] + .15
    known = [row for row in rows if row['id'] != 'unknown']
    selected = [known[0]['id']] if known else ([rows[0]['id']] if rows else [])
    return rows, selected


def suggest_clips(words, selected, duration, style):
    """Target 20–30s, split on phrases, and never bridge an excluded voice."""
    runs, run = [], []
    for word in words:
        keep = word['speaker'] in selected and not word['overlap']
        if run and (not keep or word['start'] - run[-1]['end'] > 3):
            runs.append(run)
            run = []
        if keep:
            run.append(word)
    if run:
        runs.append(run)
    clips = []
    for run in runs:
        while run:
            count = 0
            for word in run:
                if word['end'] - run[0]['start'] > 29.6:
                    break
                count += 1
            count = max(1, count)
            if count < len(run):
                boundaries = [i + 1 for i, word in enumerate(run[:count])
                              if word.get('phrase_end') and word['end'] - run[0]['start'] >= 16]
                if boundaries:
                    count = min(boundaries, key=lambda n: abs(run[n-1]['end'] - run[0]['start'] - 24))
            chunk, run = run[:count], run[count:]
            start, end = max(0., chunk[0]['start'] - .15), min(duration, chunk[-1]['end'] + .15)
            # Trim padding at any other speaker, including very short interjections.
            for word in words:
                if word['speaker'] not in selected or word['overlap']:
                    if word['end'] <= chunk[0]['start']:
                        start = max(start, word['end'])
                    if word['start'] >= chunk[-1]['end']:
                        end = min(end, word['start'])
            warnings = []
            if end - start < 10:
                warnings.append('Short excerpt; consider excluding or adjusting its boundaries.')
            if any(word['uncertain'] for word in chunk):
                warnings.append('Some lyrics are uncertain. Listen and correct the transcript.')
            if not chunk[-1].get('phrase_end'):
                warnings.append('Check the phrase ending before using this clip.')
            text = ''.join(word['text'] + ('\n' if word.get('phrase_end') else ' ') for word in chunk).strip()
            clips.append({'id': signature([chunk[0]['start'], chunk[-1]['end'], text])[:16],
                          'start': round(start, 3), 'end': round(end, 3), 'lyrics': text,
                          'style': style or 'Music with lead vocals and instrumental accompaniment',
                          'delivery': 'unspecified', 'included': end - start >= 10,
                          'reviewed': False, 'warnings': warnings})
    for left, right in zip(clips, clips[1:]):
        if left['end'] > right['start']:
            boundary = round((left['end'] + right['start']) / 2, 3)
            left['end'] = right['start'] = boundary
    return clips


def _save_preparation(project, preparation):
    preparation['revision'] = project['preparation']['revision'] + 1
    # Changing a draft never mutates any dataset already compiled from it.
    preparation.pop('dataset_project_id', None)
    preparation.pop('build_hash', None)
    return projects.update_project(project['id'], preparation=preparation)


def edit_song(project_id, body, *, regenerate=False):
    with projects._lock:
        project = projects.get_project(project_id)
        require_draft(project, editable=True)
        prep = copy.deepcopy(project['preparation'])
        if body.get('revision') != prep['revision']:
            raise ValueError('This draft changed in another window. Reload it before saving your edits.')
        track_id = body.get('track_id')
        directory = song_directory(project, track_id)
        song = prep['songs'].get(track_id)
        if not song:
            raise ValueError('Analyze this song before reviewing clips')
        if regenerate:
            selected = body.get('selected_speakers')
            if not isinstance(selected, list) or not selected or not all(isinstance(x, str) for x in selected) or not set(selected) <= {v['id'] for v in song['speakers']}:
                raise ValueError('Select at least one detected voice')
            words = json.loads((directory / 'words.json').read_text(encoding='utf-8'))['words']
            song['selected_speakers'] = list(dict.fromkeys(selected))
            track = next(track for track in project['tracks'] if track['id'] == track_id)
            song['clips'] = suggest_clips(words, selected, song['duration'], track['style'])
        else:
            rows = body.get('clips')
            if not isinstance(rows, list) or len(rows) > 500:
                raise ValueError('Save the complete list of suggested clips')
            original = {clip['id']: clip for clip in song['clips']}
            seen, checked = set(), []
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', row['id']) or row['id'] in seen:
                    raise ValueError('Choose clips from this song')
                seen.add(row['id'])
                try:
                    start, end = float(row['start']), float(row['end'])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError('Clip boundaries must be numbers') from error
                if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= song['duration'] or end - start > 60:
                    raise ValueError('Each excerpt must stay inside its song and be at most 60 seconds long')
                lyrics, style = str(row.get('lyrics') or '').strip(), str(row.get('style') or '').strip()
                if len(lyrics) > 40000 or len(style) > 1800 or row.get('included') is True and (not lyrics or not style):
                    raise ValueError('Each included clip needs lyrics and a short music description')
                if row.get('delivery') not in DELIVERY:
                    raise ValueError('Choose rap, sung, mixed or unspecified vocal delivery')
                checked.append({**original.get(row['id'], {'id': row['id'], 'warnings': ['Manually added excerpt.']}), 'start': start, 'end': end, 'lyrics': lyrics, 'style': style,
                                'delivery': row['delivery'], 'included': row.get('included') is True,
                                'reviewed': row.get('reviewed') is True})
            included = sorted((row for row in checked if row['included']), key=lambda row: row['start'])
            if any(b['start'] < a['end'] - .05 for a, b in zip(included, included[1:])):
                raise ValueError('Included clips must not overlap; adjust boundaries or exclude one')
            song['clips'] = checked
            song['holdout'] = body.get('holdout') is True
        return _save_preparation(project, prep)


def dataset_selection(project, *, automatic=False):
    require_draft(project)
    selection = []
    for track in project['tracks']:
        song = project['preparation']['songs'].get(track['id'])
        if not song:
            raise ValueError('Finish analyzing every song before building the dataset')
        for clip in song['clips']:
            if clip['included']:
                if not clip['reviewed'] and not automatic:
                    raise ValueError('Listen to and review every included clip before creating the dataset')
                selection.append({'track': track, 'clip': clip, 'holdout': song['holdout']})
    if not 2 <= len(selection) <= 500:
        raise ValueError('Include 2–500 excerpts' if automatic else 'Include 2–500 reviewed excerpts')
    if not any(row['holdout'] for row in selection) or all(row['holdout'] for row in selection):
        raise ValueError('Include clips from at least one training song and one different held-out song')
    return selection


def build_dataset(project, *, report, cancelled, automatic=False):
    selection = dataset_selection(project, automatic=automatic)
    identity = {'selection': selection, 'pair': project['tokenizer_pair'], 'version': VERSION}
    if automatic:
        identity['selection_mode'] = 'auto'
    digest = signature(identity)
    for existing in projects.list_projects():
        if existing.get('source_preparation') == {'project_id': project['id'], 'build_hash': digest}:
            projects.update_project(project['id'], preparation={**project['preparation'], 'dataset_project_id': existing['id'], 'build_hash': digest})
            return existing
    directory = projects.project_directory(project['id']) / 'excerpts' / digest
    directory.mkdir(parents=True, exist_ok=True)
    tracks, verified = [], set()
    for index, row in enumerate(selection):
        check_cancel(cancelled)
        source, clip = row['track'], row['clip']
        if source['id'] not in verified:
            if file_digest(Path(source['audio_path'])) != source['audio_sha256']:
                raise ValueError('An original song changed. Create a new preparation draft with that recording.')
            verified.add(source['id'])
        path = directory / f"{source['id']}-{clip['id']}.wav"
        stamp = path.with_suffix('.json')
        valid = False
        if path.is_file() and stamp.is_file():
            try:
                valid = json.loads(stamp.read_text())['sha256'] == file_digest(path)
            except (ValueError, KeyError):
                pass
        if not valid:
            temporary = path.with_suffix('.partial.wav')
            try:
                run_media(['-ss', str(clip['start']), '-i', source['audio_path'], '-t', str(clip['end']-clip['start']),
                           '-map', '0:a:0', '-vn', '-ac', '2', '-ar', '48000', '-c:a', 'pcm_s16le', str(temporary)], cancelled)
                check_cancel(cancelled)
                temporary.replace(path)
                write_json(stamp, {'sha256': file_digest(path)})
            finally:
                temporary.unlink(missing_ok=True)
        delivery = {'rap': 'Rapped lead vocals.', 'sung': 'Sung lead vocals.', 'mixed': 'Rapped and sung lead vocals.'}.get(clip['delivery'], '')
        tracks.append({'audio_path': str(path.resolve()), 'name': f"{source['name']} · {clip['start']:.1f}–{clip['end']:.1f}s",
                       'lyrics': clip['lyrics'] if clip['lyrics'].startswith('[') else '[verse]\n' + clip['lyrics'],
                       'style': (clip['style'] + ' ' + delivery).strip(), 'holdout': row['holdout'],
                       'source_song': source['audio_sha256'], 'reviewed': clip['reviewed']})
        report(f"Saving {'automatic' if automatic else 'reviewed'} excerpt {index + 1}/{len(selection)}", (index + 1) / len(selection) * 95)
    check_cancel(cancelled)
    created = projects.create_project(project['name'], project['trigger'], tracks, pair=project['tokenizer_pair'])
    created = projects.update_project(created['id'], source_preparation={'project_id': project['id'], 'build_hash': digest},
        source_excerpts=[{'track_id': row['track']['id'], 'name': row['track']['name'], 'start': row['clip']['start'],
                          'end': row['clip']['end'], 'delivery': row['clip']['delivery']} for row in selection],
        selection_mode='auto' if automatic else 'reviewed')
    projects.update_project(project['id'], preparation={**project['preparation'], 'dataset_project_id': created['id'], 'build_hash': digest})
    return created
