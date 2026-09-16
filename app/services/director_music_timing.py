"""Model-native clip caps for music videos, separate from song duration."""

import copy
import math


def resolve_clip_limits(model_def, requested_seconds=None, execution_profile=None):
    profile = execution_profile or {}
    fps = float(model_def.get('fps') or profile.get('fps') or 24)
    minimum = max(1, int(model_def.get('frames_minimum') or 1))
    step = max(1, int(model_def.get('frames_steps') or 1))
    defaults = model_def.get('sliding_window_defaults') or {}
    hard = int(model_def.get('frames_maximum') or defaults.get('window_max') or 26 * fps)
    hard = minimum + max(0, (hard - minimum) // step) * step
    recommended = int(profile.get('recommended_max_frames') or profile.get('effective_max_frames') or defaults.get('window_default') or 22 * fps)
    recommended = min(hard, max(minimum, recommended))
    recommended = minimum + (recommended - minimum) // step * step
    if requested_seconds is None:
        effective = min(hard, int(profile.get('effective_max_frames') or recommended))
    else:
        if isinstance(requested_seconds, bool):
            raise ValueError('Clip length must be a number of seconds or Auto.')
        seconds = float(requested_seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Clip length must be a positive finite number of seconds.')
        # Round down: the requested maximum is never exceeded by lattice rounding.
        frames = min(hard, int(profile.get('manual_max_frames') or hard), max(minimum, math.floor(seconds * fps + 1e-6)))
        effective = minimum + (frames - minimum) // step * step
    return {'fps': fps, 'frames_minimum': minimum, 'frame_step': step,
            'hard_max_frames': hard, 'recommended_frames': recommended,
            'max_frames': effective, 'max_seconds': effective / fps,
            'recommended_seconds': recommended / fps, 'auto': requested_seconds is None}


def plan_capped_music_clips(analysis, *, maximum_seconds, fps, frames_steps,
                            frames_minimum, total_duration=None, energy_bias=0):
    """Balance editorial pace with musical cues and the generation ceiling.

    Normal/faster cuts retain musical passages. Slower cuts can span their
    boundaries; -2 uses the fewest clips that fit, snapping near balanced cuts.
    Very short editorial cuts can render a native minimum and trim it.
    """
    duration = float(total_duration or analysis.get('duration') or 0)
    if not math.isfinite(duration) or duration <= 0:
        return []
    cap = float(maximum_seconds)
    minimum = frames_minimum / fps
    if not math.isfinite(cap) or cap < minimum - 1e-8:
        raise ValueError('Clip length is below this model’s minimum duration.')
    editorial_min = min(minimum, 2.0, cap / 2)
    bias = max(-2, min(2, int(energy_bias or 0)))
    bpm = max(1, float(analysis.get('bpm') or 120))
    beat_duration = 60 / bpm
    beats = sorted(float(b['time'] if isinstance(b, dict) else b.time) for b in analysis.get('beats') or [])
    if not beats:
        beats = [i * beat_duration for i in range(int(duration / beat_duration) + 1)]
    sections = analysis.get('sections') or []
    lyrics = sorted(analysis.get('lyrics') or [], key=lambda line: line.get('start', 0))
    accents = {float(b['time']): .5 + float(b.get('strength', 0)) for b in analysis.get('beats') or [] if isinstance(b, dict)}
    accents.update({float(t): 1.8 for t in analysis.get('downbeats') or []})
    section_times = {float(s['start']) for s in sections}
    anchor_times = {float(c['time']) for c in analysis.get('music_cues') or [] if c.get('type') == 'percussion_entry'}
    for index, line in enumerate(lyrics):
        for t in (line.get('start', 0), line.get('end', 0)):
            accents[float(t)] = max(2.5, accents.get(float(t), 0))
        if index and line.get('speaker') and lyrics[index-1].get('speaker') and line['speaker'] != lyrics[index-1]['speaker']:
            anchor_times.add(float(line['start']))
    anchors = [0.0]
    if bias >= 0:
        for time in sorted(section_times):
            if time - anchors[-1] >= editorial_min and duration - time >= editorial_min:
                anchors.append(time)
    anchors.append(duration)
    # A nearby estimated entrance must not displace a known section change.
    # It can still guide performance inside that section's first shot.
    if bias >= 0:
        for time in sorted(anchor_times):
            if 0 < time < duration and all(abs(time-other) >= editorial_min for other in anchors):
                anchors.append(time)
    else:
        # Long clips can contain section/performer changes and drum entrances.
        # Prefer nearby cues, but don't force an extra generated clip for each.
        for time in section_times | anchor_times:
            accents[time] = max(3.5, accents.get(time, 0))
    anchors.sort()
    cuts = [0.0]
    for passage_start, passage_end in zip(anchors, anchors[1:]):
        energy = next((float(s.get('energy', .5)) for s in sections if s['start'] <= passage_start < s['end']), .5)
        if bias < 0:
            preferred = cap * (1 if bias == -2 else .92)
        else:
            preferred = cap * (.92 - .2 * energy - bias * .12)
        preferred = max(editorial_min, preferred)
        span = passage_end - passage_start
        count = 1 if bias == 0 and span <= cap + 1e-8 else max(1, math.ceil(span / preferred - 1e-9))
        count = min(count, max(1, math.floor(span / editorial_min)))
        for index in range(1, count):
            remaining = count - index
            low = max(cuts[-1] + editorial_min, passage_end - remaining * cap)
            high = min(cuts[-1] + cap, passage_end - remaining * editorial_min)
            target = cuts[-1] + (passage_end - cuts[-1]) / (remaining + 1)
            if bias < 0:
                # Keep enough whole output frames for the remaining clips,
                # including a fractional final frame in the source audio.
                low = math.ceil(low * fps - 1e-7) / fps
                high = math.floor(high * fps + 1e-7) / fps
                target = min(high, max(low, target))
            candidates = [t for t in accents if low - 1e-8 <= t <= high + 1e-8]
            if bias < 0:
                snap_radius = min(cap * .12, max(.5, 2 * beat_duration))
                candidates = [t for t in candidates if abs(t-target) <= snap_radius]
            # Prefer phrase endings over a nearby ordinary beat, within the
            # clip cap and the selected pace's musical-boundary policy.
            def score(time):
                mid_phrase = any(l.get('start', 0) + .12 < time < l.get('end', 0) - .12 for l in lyrics)
                return accents[time] - 1.5 * abs(time-target) / max(1, preferred/3) - (1 if mid_phrase else 0)
            cut = max(candidates, key=score) if candidates else min(high, max(low, target))
            if bias < 0:
                cut = min(high, max(low, round(cut * fps) / fps))
            cuts.append(cut)
        cuts.append(passage_end)
    result = []
    for start, end in zip(cuts, cuts[1:]):
        section = next((s for s in analysis.get('sections') or []
                        if s['start'] <= (start + end) / 2 < s['end']), {})
        label, energy = section.get('label', 'verse'), float(section.get('energy', .5))
        speakers = {}
        for line in analysis.get('lyrics') or []:
            speaker = line.get('speaker')
            if speaker:
                speakers[speaker] = speakers.get(speaker, 0) + max(0, min(end, line['end']) - max(start, line['start']))
        max_frames = frames_minimum + max(0, (math.floor(cap * fps + 1e-6) - frames_minimum) // frames_steps) * frames_steps
        frames = frames_minimum + max(0, math.ceil(((end-start)*fps - frames_minimum - 1e-6) / frames_steps)) * frames_steps
        result.append({'start': start, 'end': end, 'beat_count': max(1, round((end-start)/beat_duration)),
                       'duration_frames': min(max_frames, frames), 'section_label': label, 'energy': energy,
                       'label': label, 'music_timing_version': 1,
                       **music_context(analysis, start, end),
                       'suggested_prompt_hint': f'{label}, {"high" if energy > .6 else "low" if energy < .3 else "moderate"} energy',
                       'dominant_speaker': max(speakers, key=speakers.get) if speakers and max(speakers.values()) > 0 else None})
    return result


def music_context(analysis, start, end):
    cues = [dict(c) for c in analysis.get('music_cues') or [] if start <= float(c.get('time', -1)) < end]
    for section in analysis.get('sections') or []:
        if start < float(section['start']) < end:
            cues.append({'type': 'section_change', 'time': float(section['start']),
                         'label': section.get('label', 'section'),
                         'confidence': 'analysis', 'evidence': 'music_structure'})
    cues.sort(key=lambda cue: float(cue['time']))
    intervals = analysis.get('percussion_activity')
    activity = 'unknown' if intervals is None else 'quiet'
    if any(float(i['start']) < end and float(i['end']) > start for i in intervals or []):
        activity = 'active'
    return {'music_cues': cues, 'percussion_activity': activity}


def prepare_music_timeline(plans, clips, *, fps, minimum_frames, maximum_frames, frame_step):
    """Keep editorial times on the output frame grid, pad only generation.

    Independently rounding every shot to a latent lattice moves musical cuts.
    Instead quantize absolute cut positions to video frames (at most one frame
    of error), render upward to the model lattice, and discard the extra tail.
    """
    result_plans, result_clips = [], []
    for index, clip in enumerate(clips):
        first = round(float(clip.get('start', 0)) * fps)
        end = float(clip.get('end', 0))
        last = math.ceil(end * fps - 1e-7) if index == len(clips)-1 else round(end * fps)
        if last <= first:
            continue
        count = max(1, math.ceil((last-first) / maximum_frames))
        boundaries = [first + round((last-first) * n/count) for n in range(count+1)]
        for part, (start_frame, end_frame) in enumerate(zip(boundaries, boundaries[1:])):
            output_frames = end_frame - start_frame
            generation_frames = minimum_frames + max(0, math.ceil((output_frames-minimum_frames)/frame_step)) * frame_step
            plan = copy.deepcopy(plans[index]) if index < len(plans) else {}
            planned = copy.deepcopy(clip)
            planned.update(start=start_frame/fps, end=end_frame/fps, duration_sec=output_frames/fps,
                           duration_frames=generation_frames, output_frames=output_frames,
                           music_timing_version=1, _director_source_clip_indices=[index],
                           _director_segment_index=part, _director_segment_count=count)
            planned['music_cues'] = [c for c in clip.get('music_cues') or []
                                    if start_frame/fps - .5/fps <= c['time'] < end_frame/fps - .5/fps]
            plan.update(_director_generation_frames=generation_frames, _director_duration_sec=output_frames/fps,
                        _director_source_clip_indices=[index])
            result_plans.append(plan)
            result_clips.append(planned)
    return result_plans, result_clips


def music_output_trim(generation_frames, output_frames):
    """Validate an explicit editorial duration before queueing or rerunning."""
    if isinstance(output_frames, bool) or not isinstance(output_frames, int) or not 1 <= output_frames <= generation_frames:
        raise ValueError('Music output duration must fit inside its generated clip.')
    return generation_frames - output_frames
