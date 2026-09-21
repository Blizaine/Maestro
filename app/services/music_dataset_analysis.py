"""Resumable song analysis using Maestro's existing local audio models."""
import gc
import json
from pathlib import Path

from . import music_training as projects
from .music_styles import file_digest
from .music_dataset import (VERSION, assign_speakers, check_cancel, require_draft,
                            run_media, signature, song_directory, suggest_clips,
                            voice_summary, write_json)


def separate_vocals(track, destination, *, report, cancelled):
    import torch
    import torchaudio
    import soundfile as sf
    from models.TTS.yue2.music_assets import ensure_asset
    from models.TTS.yue2.audio_training_data import read_stereo
    from models.TTS.yue2.lyric_alignment import vocal_stem
    model = None
    temporary = destination.with_suffix('.partial.wav')
    try:
        weights = ensure_asset('vocal_separator', report=report, cancelled=cancelled)
        check_cancel(cancelled)
        with torch.device('cpu'):
            model = torchaudio.models.hdemucs_high(sources=['drums', 'bass', 'other', 'vocals'])
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        model.cuda().eval()
        audio = vocal_stem(model, read_stereo(track['audio_path'], 44100), cancelled)
        sf.write(temporary, audio, 44100, subtype='FLOAT')
        check_cancel(cancelled)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def transcribe_words(path, *, cancelled, language=None):
    from . import audio_analysis
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    from collections import Counter
    try:
        model = audio_analysis._get_whisper_model()
        # Music has sustained vowels, quiet rap and instrumental gaps; do not let
        # a speech-only VAD silently remove them. Hallucinations remain reviewable.
        words, languages = [], Counter()
        # Re-detect each minute, so an unusual spoken intro cannot choose the
        # language for an entire long song. Context overlaps; ownership does not.
        with sf.SoundFile(path) as source:
            rate, duration = source.samplerate, len(source) / source.samplerate
            for start in range(0, int(duration) + 1, 60):
                if start >= duration:
                    break
                check_cancel(cancelled)
                left, end = max(0, start - 1.5), min(duration, start + 60)
                right = min(duration, end + 1.5)
                source.seek(round(left * rate))
                audio = source.read(round((right - left) * rate), dtype='float32', always_2d=True).mean(1)
                divisor = gcd(rate, 16000)
                audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype('float32')
                segments, info = model.transcribe(audio, language=language or None, beam_size=5,
                    word_timestamps=True, vad_filter=False, condition_on_previous_text=False)
                count = 0
                for segment in segments:
                    check_cancel(cancelled)
                    uncertain = segment.avg_logprob < -.8 or segment.no_speech_prob > .6
                    chunk = []
                    for word in segment.words or []:
                        begin, finish = left + float(word.start), left + float(word.end)
                        if word.word.strip() and finish > begin and start <= (begin + finish) / 2 < end:
                            chunk.append({'start': begin, 'end': min(duration, finish), 'text': word.word.strip(),
                                'probability': float(word.probability), 'uncertain': uncertain, 'phrase_end': False})
                    if chunk:
                        chunk[-1]['phrase_end'] = True
                    words.extend(chunk); count += len(chunk)
                languages[info.language] += count
        return {'words': words, 'language': language or (languages.most_common(1)[0][0] if languages else ''),
                'languages': dict(languages), 'transcription_version': 2}
    finally:
        audio_analysis.unload_whisper()


def diarize_voices(path, *, cancelled):
    from . import audio_analysis
    import torch
    import soundfile as sf
    from scipy.signal import resample_poly
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    pipeline = None
    try:
        check_cancel(cancelled)
        pipeline = audio_analysis.get_diarizer_pipeline(profile='music')
        if pipeline is None:
            raise ValueError('The speaker model is unavailable. Check the audio analysis model settings.')
        audio, rate = sf.read(path, dtype='float32')
        if rate != 16000:
            from math import gcd
            common = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // common, rate // common).astype('float32')
        with torch.inference_mode():
            result = pipeline({'waveform': torch.from_numpy(audio[None]), 'sample_rate': 16000}, min_speakers=1)
        check_cancel(cancelled)
        annotation = getattr(result, 'speaker_diarization', result)
        return [{'start': float(span.start), 'end': float(span.end), 'speaker': str(speaker)}
                for span, _, speaker in annotation.itertracks(yield_label=True)]
    finally:
        # Pyannote disables TF32 for reproducibility; don't leave the next video
        # or music generation running with a different global precision policy.
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        pipeline = None
        audio_analysis.unload_diarizer()


def analyze_songs(project, *, report, cancelled, retry_voices=False, track_id=None, language=None):
    require_draft(project)
    for index, track in enumerate(project['tracks']):
        if track_id and track['id'] != track_id:
            continue
        check_cancel(cancelled)
        if file_digest(Path(track['audio_path'])) != track['audio_sha256']:
            raise ValueError('An original song changed. Create a new preparation draft.')
        # Complete song summaries include user review; resuming never replaces them.
        current = projects.get_project(project['id'])
        previous = current['preparation']['songs'].get(track['id'])
        failed_voices = previous and (previous.get('diarization_failed') or any(
            warning.startswith('Voices could not be separated automatically.') for warning in previous.get('warnings', [])))
        if previous and not track_id and not (retry_voices and failed_voices):
            continue
        directory = song_directory(project, track['id'])
        directory.mkdir(parents=True, exist_ok=True)
        identity = signature({'version': VERSION, 'audio': track['audio_sha256']})

        def stage(message, fraction):
            report(f"{track['name']}: {message} ({index + 1}/{len(project['tracks'])})",
                   (index + fraction) / len(project['tracks']) * 100)

        def cached(name):
            try:
                value = json.loads((directory / name).read_text(encoding='utf-8'))
                return value if value.get('identity') == identity else None
            except (OSError, ValueError):
                return None

        stage('preparing audio preview', 0)
        preview = directory / 'preview.mp3'
        if not preview.is_file():
            temporary = directory / 'preview.partial.mp3'
            try:
                run_media(['-i', track['audio_path'], '-map', '0:a:0', '-vn', '-ac', '2', '-ar', '44100',
                           '-c:a', 'libmp3lame', '-b:a', '160k', str(temporary)], cancelled)
                check_cancel(cancelled)
                temporary.replace(preview)
            finally:
                temporary.unlink(missing_ok=True)
        vocals = directory / 'vocals.wav'
        stem = cached('stem.json')
        if not stem or not vocals.is_file() or stem.get('sha256') != file_digest(vocals):
            stage('separating vocals', .1)
            separate_vocals(track, vocals, report=report, cancelled=cancelled)
            write_json(directory / 'stem.json', {'identity': identity, 'sha256': file_digest(vocals)})
        transcription = cached('transcription.json')
        if not transcription or track_id:
            stage('transcribing words', .4)
            transcription = {'identity': identity, **transcribe_words(vocals, cancelled=cancelled, language=language)}
            check_cancel(cancelled)
            write_json(directory / 'transcription.json', transcription)
        diarization = cached('voices.json')
        if retry_voices and diarization and diarization.get('warning'):
            diarization = None
        if not diarization:
            stage('finding voices', .7)
            try:
                diarization = {'identity': identity, 'turns': diarize_voices(vocals, cancelled=cancelled), 'warning': ''}
            except InterruptedError:
                raise
            except Exception as error:
                # A reviewable transcript is useful even without a speaker model.
                # Do not pretend the anonymous fallback identifies the artist.
                diarization = {'identity': identity, 'turns': [],
                               'warning': 'Voices could not be separated automatically. Review each excerpt for other singers. ' + str(error)[:300]}
            check_cancel(cancelled)
            write_json(directory / 'voices.json', diarization)
        stage('suggesting excerpts', .95)
        words = assign_speakers(transcription['words'], diarization['turns'])
        # Ignore decoder timestamps outside the recording rather than producing
        # unplayable clips from a hallucinated trailing segment.
        words = [word for word in words if 0 <= word['start'] < word['end'] <= track['duration']]
        words.sort(key=lambda word: (word['start'], word['end']))
        write_json(directory / 'words.json', {'identity': identity, 'words': words})
        voices, selected = voice_summary(words)
        warnings = [diarization['warning']] if diarization.get('warning') else []
        if not words:
            warnings.append('No lyrics were recognized. Add excerpts and lyrics manually, or exclude this song from the dataset.')
        elif len(words) < track['duration'] * .3:
            warnings.append('Only a small part of this song was transcribed. Choose the lyrics language and rescan, or add excerpts manually. Instrumental sections can also cause low coverage.')
        if any(word['overlap'] for word in words):
            warnings.append('Overlapping or ambiguous voices were left out of suggested clips.')
        warnings.append('Voice labels are local to this song. One singer can appear in more than one voice; listen before selecting.')
        proposed = suggest_clips(words, selected, track['duration'], track['style'])
        if track_id and previous:
            kept = [clip for clip in previous['clips'] if clip.get('reviewed') or clip['id'].startswith('manual-')]
            proposed = kept + [clip for clip in proposed if not any(clip['start'] < old['end'] and clip['end'] > old['start'] for old in kept)]
            proposed.sort(key=lambda clip: clip['start'])
            warnings.append(f"Kept {len(kept)} reviewed or manual excerpts. New suggestions need review.")
        summary = {'duration': track['duration'], 'language': transcription.get('language', ''),
                   'language_override': language or '', 'recognized_words': len(words),
                   'speakers': voices, 'selected_speakers': selected,
                   'diarization_failed': bool(diarization.get('warning')),
                   'holdout': previous['holdout'] if previous else track['holdout'],
                   'clips': proposed, 'warnings': warnings}
        check_cancel(cancelled)
        with projects._lock:
            current = projects.get_project(project['id'])
            prep = current['preparation']
            prep.pop('dataset_project_id', None)
            prep.pop('build_hash', None)
            prep['songs'][track['id']] = summary
            prep['revision'] += 1
            projects.update_project(project['id'], preparation=prep)
