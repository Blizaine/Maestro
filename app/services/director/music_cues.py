"""Lightweight, CPU-only percussion timing estimates for music direction.

HPSS separates transient energy, not instrument identities. These cues are
explicitly estimates: piano attacks or other broadband transients can resemble
drums. Never describe this as stem separation or exact drum transcription.
"""
def _percussion_features(y, sr):
    import numpy as np
    import librosa

    duration = len(y) / sr
    rate = 10
    energy = np.zeros(max(1, int(np.ceil(duration * rate))))
    total_energy = np.zeros_like(energy)
    # Bound STFT/median-filter memory even for hour-long soundtracks. Context
    # on both sides avoids manufacturing an onset at each chunk boundary.
    for start in range(0, len(y), sr * 20):
        stop = min(len(y), start + sr * 20)
        left, right = max(0, start-sr), min(len(y), stop+sr)
        spectrum = np.abs(librosa.stft(y[left:right], n_fft=1024, hop_length=256))
        _, percussive = librosa.decompose.hpss(spectrum, margin=(2.0, 2.0))
        power = np.sum(percussive ** 2, axis=0)
        full = np.sum(spectrum ** 2, axis=0)
        # Exclude narrow tonal attacks where HPSS alone is ambiguous.
        broad = np.mean(percussive > np.maximum(percussive.max(axis=0) * .12, 1e-8), axis=0)
        power = np.where(broad >= .08, power, 0)
        positions = left/sr + np.arange(len(power)) * 256/sr
        selected = (positions >= start/sr) & (positions < stop/sr)
        bins = np.minimum((positions[selected] * rate).astype(int), len(energy)-1)
        np.maximum.at(energy, bins, power[selected])
        np.maximum.at(total_energy, bins, full[selected])
    return energy, total_energy


def detect_percussion(y, sr):
    import numpy as np
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    duration, rate = len(y) / sr, 10
    energy, total_energy = _percussion_features(y, sr)
    if energy.max() < 1e-8:
        return ([] if total_energy.max() < 1e-8 else None), []
    # A single loud crash must not suppress quieter repeated hits elsewhere
    # in the mix. Use a robust level, with a small floor relative to the peak.
    threshold = max(np.quantile(energy[energy > 1e-8], .95) * .12, energy.max() * .003)
    peaks = ((energy >= maximum_filter1d(energy, size=3))
             & (energy >= threshold)
             & (energy / np.maximum(total_energy, 1e-8) >= .025))
    # Require a sustained rhythmic passage; an isolated impact is not proof
    # that a drummer has entered. Bridge the spaces between individual hits.
    density = uniform_filter1d(peaks.astype(float), size=21, mode='constant') * 21
    active = maximum_filter1d((density >= 3).astype(float), size=7) > 0
    intervals, cues = [], []
    indices = np.flatnonzero(peaks & active)
    if not len(indices):
        return None, cues
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 25) + 1)
    for group in groups:
        if len(group) < 3:
            continue
        start = float(group[0] / rate)
        end = min(duration, float(group[-1] / rate + .5))
        intervals.append({'start': start, 'end': end})
        cues.append({'type': 'percussion_entry', 'time': start,
                     'confidence': 'estimated', 'evidence': 'broadband_percussive_transients'})
    return intervals or None, cues


def format_music_cues(clip):
    """Give the writer both absolute song time and local shot time."""
    if 'percussion_activity' not in clip:
        return ''
    status = clip.get('percussion_activity', 'unknown')
    cues = [c for c in clip.get('music_cues') or [] if c.get('type') == 'percussion_entry']
    parts = []
    for cue in clip.get('music_cues') or []:
        if cue.get('type') == 'section_change':
            song_time = float(cue['time'])
            local_time = max(0, song_time - float(clip.get('start', 0)))
            parts.append(f'Music section changes to {cue.get("label", "the next section")} '
                         f'at song {song_time:.2f}s, shot +{local_time:.2f}s. '
                         'Reflect this within the clip through performance, framing or a camera change '
                         'where appropriate; preserve the full planned clip duration.')
    for cue in cues:
        song_time = float(cue['time'])
        local_time = max(0, song_time - float(clip.get('start', 0)))
        parts.append(f'Likely percussion entrance at song {song_time:.2f}s, shot +{local_time:.2f}s. '
                     'If the referenced band includes a drummer, favor revealing that same drummer '
                     'and their first visible strike at this entrance, following the supplied audio.')
    if status == 'quiet':
        parts.append('No sustained percussion was detected here. Favor the ensemble, listening '
                     'reactions, or environment; avoid an invented drum solo or drummer entrance. '
                     'This percussion cue is not evidence of singing.')
    elif status == 'active' and not cues:
        parts.append('Percussion is already active; a drummer insert may follow the audible rhythm, '
                     'but do not describe this as a new drum entrance.')
    elif status == 'unknown':
        parts.append('Instrument timing is unknown; do not invent an instrument entrance from the beat grid.')
    parts.append('Percussion timing is an audio estimate, not a verified instrument identity. '
                 'Do not infer guitar, bass, or other instrument entrances from it. '
                 'Keep instrument emphasis grounded in these cues and the user’s requested direction.')
    return ' '.join(parts)
