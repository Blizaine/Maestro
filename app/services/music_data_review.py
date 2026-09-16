"""Local listening/transcription aids. Drafts never replace training labels."""
from pathlib import Path
import json
import subprocess
import time

from . import music_training as projects
from .music_styles import file_digest


def sequence_coverage(project, *, report, cancelled):
    """Show the exact amount of each recording fitting AR's sequence allowance."""
    import numpy as np
    from models.TTS.yue2.music_assets import ensure_asset
    from models.TTS.yue2.protocol import SongRequest, token_prefixes
    from models.TTS.yue2.tokenization_yue2 import YuE2TextTokenizer
    tokenizer = YuE2TextTokenizer(str(ensure_asset('text_tokenizer', report=report, cancelled=cancelled)))
    coverage = []
    for track in project['tracks']:
        if cancelled():
            raise InterruptedError('Music preparation stopped')
        prefix = token_prefixes(SongRequest(style=f"{project['trigger']}, {track['style']}", lyrics=track['lyrics'], cot='off'), tokenizer)
        count = len(np.load(projects.project_directory(project['id']) / 'prepared' / (track['id'] + '.npy'), mmap_mode='r', allow_pickle=False))
        used = min(count, max(0, 12288 - len(prefix) - 1))
        coverage.append({'track_id': track['id'], 'source_seconds': count / 25,
                         'training_seconds': used / 25, 'truncated': used < count})
    projects.update_project(project['id'], sequence_coverage=coverage)
    return coverage


def draft_lyrics(project, *, report, cancelled):
    from . import audio_analysis
    drafts = dict(project.get('review_drafts') or {})
    try:
        for index, track in enumerate(project['tracks']):
            if cancelled():
                raise InterruptedError('Lyric review stopped; completed drafts are saved')
            if file_digest(Path(track['audio_path'])) != track['audio_sha256']:
                raise ValueError('A recording changed. Create a new dataset before transcribing it')
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', track['audio_path']],
                                   capture_output=True, text=True, check=True, timeout=30)
            duration = float(json.loads(probe.stdout)['format']['duration'])
            if not 1 <= duration <= 600:
                raise ValueError('Use recordings between 1 second and 10 minutes for lyric review')
            report(f"Listening to {track['name']} ({index + 1}/{len(project['tracks'])})", index / len(project['tracks']) * 100)
            # Do not seed the decoder with the labels under review: this is an
            # independent draft, not a confirmation of whatever text was pasted.
            model = audio_analysis._get_whisper_model()
            def transcribe(use_vad):
                segments, info = model.transcribe(track['audio_path'], beam_size=5, word_timestamps=False,
                                                  language=None, vad_filter=use_vad)
                rows = []
                for segment in segments:
                    if cancelled():
                        raise InterruptedError('Lyric review stopped; completed drafts are saved')
                    if segment.text.strip():
                        rows.append({'start': round(segment.start, 2), 'end': round(segment.end, 2),
                                     'text': segment.text.strip(),
                                     'uncertain': segment.avg_logprob < -.8 or segment.no_speech_prob > .6})
                return rows, info
            rows, info = transcribe(True)
            fallback = not rows
            if fallback:
                if cancelled():
                    raise InterruptedError('Lyric review stopped; completed drafts are saved')
                report(f"No words detected in {track['name']}; retrying without the speech filter")
                rows, info = transcribe(False)
            drafts[track['id']] = {'fingerprint': projects.review_fingerprint(track), 'created_at': time.time(),
                'segments': rows, 'text': '\n'.join(row['text'] for row in rows), 'language': info.language,
                'seconds': info.duration, 'review_required': True, 'speech_filter_retry': fallback,
                'note': ('No words recognized; keep or enter lyrics manually. ' if not rows else '') +
                        'Local Whisper draft. Singing, repeated sections and backing vocals may be missed or mistaken. Listen and correct before applying.'}
            projects.update_project(project['id'], review_drafts=drafts)
    finally:
        audio_analysis.unload_whisper()
