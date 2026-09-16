"""Durable, local music datasets and bounded YuE2 training projects."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid

from .music_styles import style_directory, file_digest

PROJECT_ROOT = Path("settings/music_training")
_lock = threading.RLock()
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}


def project_directory(project_id, root=None):
    return style_directory(project_id, root or PROJECT_ROOT)


def _write(project, root=None):
    directory = project_directory(project["id"], root)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "project.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(project, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def get_project(project_id, *, root=None):
    with _lock:
        try:
            return json.loads((project_directory(project_id, root) / "project.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("Music training project was not found") from error


def list_projects(*, root=None):
    projects = []
    for path in Path(root or PROJECT_ROOT).glob("*/project.json"):
        try:
            projects.append(get_project(path.parent.name, root=root))
        except ValueError:
            continue
    return sorted(projects, key=lambda project: project["created_at"], reverse=True)


def update_project(project_id, *, root=None, **changes):
    with _lock:
        project = get_project(project_id, root=root)
        for key, value in changes.items():
            if key not in {"id", "tracks", "dataset_digest", "created_at"}:
                project[key] = value
        _write(project, root)
        return project


def review_fingerprint(track):
    """Bind a review to this recording and the exact labels used for training."""
    return hashlib.sha256(json.dumps({key: track.get(key) for key in
        ("audio_sha256", "lyrics", "style", "holdout", "source_song")}, sort_keys=True).encode()).hexdigest()


def review_track(project_id, track_id, reviewed):
    with _lock:
        project = get_project(project_id)
        track = next((item for item in project['tracks'] if item['id'] == track_id), None)
        if track is None:
            raise ValueError('Choose a recording in this project')
        if file_digest(Path(track['audio_path'])) != track['audio_sha256']:
            raise ValueError('This recording changed. Create a new dataset before reviewing it')
        reviews = dict(project.get('reviews') or {})
        if reviewed:
            reviews[track_id] = review_fingerprint(track)
        else:
            reviews.pop(track_id, None)
        return update_project(project_id, reviews=reviews)


def create_project(name, trigger, tracks, *, root=None):
    name, trigger = str(name or "").strip(), str(trigger or "").strip()
    if not name or len(name) > 100 or not trigger or len(trigger) > 200:
        raise ValueError("Enter a name and a short, distinctive style trigger")
    if not isinstance(tracks, list) or not 2 <= len(tracks) <= 50:
        raise ValueError("Choose 2–50 songs, including at least one held-out song")
    normalized, seen, song_splits, reviews = [], set(), {}, {}
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError("Each recording must include its audio file, lyrics and style caption")
        audio = Path(str(track.get("audio_path") or "")).resolve()
        if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError("Each song needs an existing audio file")
        if audio.stat().st_size > 1024**3:
            raise ValueError("Each training recording must be smaller than 1 GB")
        lyrics, style = str(track.get("lyrics") or "").strip(), str(track.get("style") or "").strip()
        if not lyrics or not style or len(lyrics) > 40000 or len(style) > 2000:
            raise ValueError("Supply full section-labelled lyrics and a short style caption for each song")
        digest = file_digest(audio)
        if digest in seen:
            raise ValueError("Use different recordings for training and held-out evaluation")
        seen.add(digest)
        normalized.append({"id": digest[:16], "audio_path": str(audio), "audio_sha256": digest,
                           "name": audio.stem, "lyrics": lyrics, "style": style,
                           "holdout": track.get("holdout") is True})
        song = str(track.get('source_song') or '').strip()[:200]
        if song:
            normalized[-1]['source_song'] = song
            key = song.casefold()
            if key in song_splits and song_splits[key] != normalized[-1]['holdout']:
                raise ValueError('Excerpts from the same original song must stay together: all training or all held out')
            song_splits[key] = normalized[-1]['holdout']
        if track.get('reviewed') is True:
            reviews[digest[:16]] = review_fingerprint(normalized[-1])
    if not any(track["holdout"] for track in normalized) or all(track["holdout"] for track in normalized):
        raise ValueError("Keep at least one training song and one separate held-out song")
    project = {"version": 1, "id": uuid.uuid4().hex[:16], "name": name, "trigger": trigger,
               "tracks": normalized, "created_at": time.time(), "status": "draft", "progress": 0,
               "dataset_digest": hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest(),
               "checkpoints": [], "reviews": reviews, "message": "Ready to prepare audio tokens"}
    with _lock:
        _write(project, root)
    return project


def training_options(raw):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("Training settings must be an object")
    for key in ("steps", "rank", "seed"):
        value = raw.get(key)
        if value is not None and (isinstance(value, bool) or not str(value).lstrip("-").isdigit()):
            raise ValueError("Training steps, rank and seed must be whole numbers")
    try:
        steps, rank = int(raw.get("steps", 800)), int(raw.get("rank", 64))
        seed, learning_rate = int(raw.get("seed", 22005)), float(raw.get("learning_rate", 1e-4))
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Training steps, rank, seed and learning rate must be numbers") from error
    if not 1 <= steps <= 1600 or rank not in (4, 8, 16, 32, 64):
        raise ValueError("Use 1–1600 training steps and rank 4, 8, 16, 32 or 64")
    if not math.isfinite(learning_rate) or not 1e-6 <= learning_rate <= 1e-3:
        raise ValueError("Learning rate must be between 0.000001 and 0.001")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Training seed must be between 0 and 4294967295")
    from .music_auditions import audition_options
    return {"steps": steps, "rank": rank, "seed": seed, "learning_rate": learning_rate,
            "audition": audition_options(raw.get('audition')),
            "lyric_alignment": raw.get("lyric_alignment") is True,
            "resume": raw.get("resume") is True, "checkpoint_every": min(200, steps),
            "artist_fraction": 0.5, "max_tokens": 12288, "accumulation_steps": 2}


def audio_training_options(raw):
    if not isinstance(raw, dict):
        raise ValueError('Expected audio adaptation options')
    base = training_options({**raw, 'rank': 32, 'learning_rate': raw.get('learning_rate', 5e-5),
                             'steps': raw.get('steps', 200)})
    conditioning = raw.get('conditioning_checkpoint', '')
    if not isinstance(conditioning, str) or Path(conditioning).name != conditioning and conditioning:
        raise ValueError('Choose a saved conditioning checkpoint')
    return {key: base[key] for key in ('steps', 'rank', 'seed', 'learning_rate', 'resume', 'audition')} | {
        'io_learning_rate': 2e-5, 'checkpoint_every': min(100, base['steps']), 'window_frames': 512,
        'conditioning_checkpoint': conditioning}


def fork_project(project_id):
    """New experiment with immutable source metadata; copy only verified preparation."""
    import shutil
    source = get_project(project_id)
    created = create_project(source['name'][:80] + ' · experiment', source['trigger'], source['tracks'])
    if created['dataset_digest'] != source['dataset_digest']:
        raise ValueError('Source recordings changed; prepare a new dataset')
    original, destination = project_directory(project_id), project_directory(created['id'])
    changes = {'reviews': source.get('reviews', {}), 'sequence_coverage': source.get('sequence_coverage', [])}
    for folder, field in (('prepared', 'prepared'), ('audio_targets', 'audio_prepared'), ('alignment', 'alignment')):
        if source.get(field) and (original / folder).is_dir():
            shutil.copytree(original / folder, destination / folder)
            changes[field] = source[field]
    return update_project(created['id'], **changes)
