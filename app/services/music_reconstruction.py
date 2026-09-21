"""Read-only reconstruction requests for prepared music-training recordings."""
import math

from .music_styles import load_style, validate_strength
from .music_contracts import adapter_contract


def reconstruction_options(raw, project):
    if not isinstance(raw, dict):
        raise ValueError("Reconstruction settings must be an object")
    if not project.get("prepared"):
        raise ValueError("Prepare this project's audio before reconstructing it")
    style_id = str(raw.get("style_id") or "")
    style = load_style(style_id)
    training = style.get('training') or {}
    if (not isinstance(training, dict) or training.get("project_id") != project["id"]
            or training.get("dataset_digest") != project["dataset_digest"]):
        raise ValueError("Choose a saved checkpoint style from this training project")
    tracks = raw.get("track_ids", [track["id"] for track in project["tracks"]])
    known = {track["id"] for track in project["tracks"]}
    if (not isinstance(tracks, list) or not 1 <= len(tracks) <= 4
            or any(not isinstance(track, str) or track not in known for track in tracks)
            or len(set(tracks)) != len(tracks)):
        raise ValueError("Choose 1–4 different recordings from this project")
    options = {"style_id": style_id, "track_ids": list(tracks)}
    joint = adapter_contract(style) == 'joint'
    comparison = raw.get('comparison', 'joint' if joint else 'audio' if training.get('audio_checkpoint') else 'ar')
    if not isinstance(comparison, str) or comparison not in {'ar', 'audio', 'joint'}:
        raise ValueError('Choose an AR, audio adaptation or joint comparison')
    if joint != (comparison == 'joint'):
        raise ValueError('Joint checkpoints compare both adapters together; separate checkpoints use AR or audio comparisons')
    options['comparison'] = comparison
    if comparison == 'audio' and style.get('adapted_pair'):
        raise ValueError('Use the sound-pair comparison in Voice & sound; generic audio adapters do not match the adapted tokenizer')
    for key, default, low, high in (("seconds", 60, 10, 60), ("seed", 22005, 0, 2**32 - 1),
                                   ("steps", 32, 1, 64)):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value) or not low <= value <= high:
            raise ValueError(f"{key} must be a whole number between {low} and {high}")
        options[key] = int(value)
    strength = raw.get("artist_strength", 1.0)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError("artist_strength must be a number between 0 and 1.5")
    options["artist_strength"] = validate_strength(strength)
    # Freeze the checkpoint identity when queued, then verify it again at run time.
    options["ar_sha256"] = style["ar"]["sha256"]
    options["nar_sha256"] = style["nar"]["sha256"]
    return options
