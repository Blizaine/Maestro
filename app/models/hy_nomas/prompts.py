# Adapted from Hy-Nomas (https://github.com/Tencent-Hunyuan/Hy-Nomas, hymm/inference.py and hymm/pipeline/base.py), Copyright (C) 2026 Tencent,
# Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""World prompts of Hy-Nomas: a JSON world description (style, perspective, subjects and timed segments with their actions) shaped
into one text-encoder prompt per segment with the training-side rules, the segments' frame boundaries, and the token positions of
each segment's actions in its prompt (the attention masks between actions and the text later segments may see)."""
import copy
import json
import re

from tqdm import tqdm

FPS = 24
TEMPORAL_RATIO = 4
# the fields the model was trained on (keep_keywords of the inference preset), and the per-segment shaping rules
KEEP_KEYWORDS = ("Style", "perspective_view", "subject_list", "subject_list.subject_id", "subject_list.display_name", "subject_list.subject_attribute",
                 "segments", "segments.visual_description_long", "segments.description_change", "segments.contained_actions",
                 "segments.contained_actions.start_time", "segments.contained_actions.end_time", "segments.contained_actions.action_description")
_CAMERA_TEXT_BLOCK = re.compile(r"\s*<Camera(\d+)>.*?</Camera\1>\s*", flags=re.DOTALL)


def parse_timestamp(value):
    """Seconds of a float or of 'SS.mmm', 'MM:SS.mmm' or 'HH:MM:SS.mmm'."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    parts = value.strip().split(":")
    return sum(float(part) * 60 ** index for index, part in enumerate(reversed(parts)))


def _relative_timestamp(seconds):
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:06.3f}"


def _keep_keys(obj, keys):
    """Deep copy of obj keeping only keys (dot paths; a single name is kept at every depth; a listed parent with nested rules keeps only
    those children)."""
    tree = {}
    for key in keys:
        parts, node = key.split("."), tree
        for part in parts[:-1]:
            node.setdefault(None, set()).add(part)
            node = node.setdefault(part, {})
        node.setdefault(None, set()).add(parts[-1])
    inherited = tree.get(None, set())

    def apply(item, rules):
        keep = (rules.get(None, set()) if rules else set()) | inherited
        result = {}
        for key, value in item.items():
            if key not in keep:
                continue
            sub = rules.get(key) if rules else None
            if sub:
                value = apply(value, sub) if isinstance(value, dict) else [apply(v, sub) if isinstance(v, dict) else v for v in value] if isinstance(value, list) else value
            result[key] = value
        return result

    return apply(copy.deepcopy(obj), tree)


def _strip_camera_blocks(segment):
    """Remove the <CameraN>...</CameraN> text blocks of a camera-controlled segment (a "camera" trajectory, or the source's
    segment_control "camera"): its camera comes from the trajectory."""
    if segment.get("segment_control") != "camera" and segment.get("camera") is None:
        return segment

    def clean(value):
        if isinstance(value, str):
            return re.sub(r"\s+", " ", _CAMERA_TEXT_BLOCK.sub(" ", value)).strip() if "<Camera" in value else value
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return [clean(item) for item in value] if isinstance(value, list) else value

    return clean(segment)


def parse_world(prompt):
    """The world description of a prompt: a JSON object with 'segments' (legacy 'events' / 'subjects_lifecycle' renamed), or None for a
    plain text prompt."""
    try:
        item = json.loads(prompt)
    except (ValueError, TypeError):
        return None
    if not isinstance(item, dict):
        return None
    item = dict(item)
    if "segments" not in item and "events" in item:
        item["segments"] = item.pop("events")
    if "subject_list" not in item and "subjects_lifecycle" in item:
        item["subject_list"] = item.pop("subjects_lifecycle")
    return item if isinstance(item.get("segments"), list) and item["segments"] else None


def plain_world(prompt):
    """A plain text prompt as a world of one segment (the format the model was trained on)."""
    return {"segments": [{"visual_description_long": prompt.strip(), "contained_actions": []}]}


def segment_prompts(world):
    """One JSON prompt per segment: the global fields in the first one only, the long visual description in the first segment only, the
    description change from the second one, actions last with timestamps relative to their segment, only the trained fields kept."""
    context = {key: value for key, value in _keep_keys(world, KEEP_KEYWORDS).items() if key != "segments"}
    prompts = []
    for index, segment in enumerate(world["segments"]):
        segment = dict(_strip_camera_blocks(segment))
        if index > 0:
            segment.pop("visual_description_long", None)
        else:
            segment.pop("description_change", None)
        if "contained_actions" in segment:
            actions = segment.pop("contained_actions")
            start = parse_timestamp(segment.get("start_time", 0.0))
            segment["contained_actions"] = [({**action, **{key: _relative_timestamp(parse_timestamp(action[key]) - start)
                                                            for key in ("start_time", "end_time") if key in action}})
                                            if isinstance(action, dict) else action for action in actions]
        prompts.append(json.dumps(_keep_keys({**(context if index == 0 else {}), "segments": [segment]}, KEEP_KEYWORDS), ensure_ascii=False))
    return prompts


def segment_boundaries(world):
    """Pixel frame boundaries of the segments from their timestamps (len(segments) + 1 values), or None if a timestamp is missing."""
    boundaries = []
    for segment in world["segments"]:
        if segment.get("start_time") is None or segment.get("end_time") is None:
            return None
        if not boundaries:
            boundaries.append(int(parse_timestamp(segment["start_time"]) * FPS))
        boundaries.append(int(parse_timestamp(segment["end_time"]) * FPS))
    return boundaries


def latent_boundaries(pixel_boundaries, latent_frames):
    """Latent frame boundaries (pixel frame // 4, strictly increasing, the last one the clip's last latent frame)."""
    latent = [0]
    for boundary in pixel_boundaries[1:]:
        latent.append(max(latent[-1] + 1, min(boundary // TEMPORAL_RATIO, latent_frames)))
    latent[-1] = latent_frames
    return latent


def _segment_actions(prompt):
    segments = json.loads(prompt).get("segments")
    actions = segments[0].get("contained_actions") if segments and isinstance(segments[0], dict) else None
    return segments[0], actions if isinstance(actions, list) and actions else None


def action_text_offsets(prompt, tokenize):
    """Token offsets [o_0, ..., o_A] of the segment's actions in its encoded text (o_0: end of the head before the actions, action k:
    [o_k, o_k+1), o_A: the valid text length), from the common token prefix of the prompt and its variants keeping actions 0..k-1;
    None without actions. tokenize(text) returns the token ids of the encoded (cropped) text."""
    segment, actions = _segment_actions(prompt)
    if actions is None:
        return None
    world = json.loads(prompt)
    full = tokenize(prompt)

    def variant(count):
        shortened = dict(segment)
        if count:
            shortened["contained_actions"] = actions[:count]
        else:
            shortened.pop("contained_actions")
        ids = tokenize(json.dumps({key: ([shortened] if key == "segments" else value) for key, value in world.items()}, ensure_ascii=False))
        common = 0
        while common < min(len(ids), len(full)) and ids[common] == full[common]:
            common += 1
        return common

    offsets = []
    for count in range(len(actions)):
        offsets.append(max(variant(count), offsets[-1] if offsets else 0))
    return offsets + [len(full)]


def action_video_windows(prompt, segment_frames):
    """Latent frame windows [start, stop, action] tiling the segment (action -1 for gaps), from the actions' segment-relative timestamps;
    None without actions, when a single action covers the whole segment or when actions overlap (the segment then attends to its whole
    text, as the source does)."""
    _, actions = _segment_actions(prompt)
    if actions is None:
        return None
    def to_latent(seconds):
        return int(max(seconds, 0.0) * FPS) // TEMPORAL_RATIO
    spans = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or "start_time" not in action or "end_time" not in action:
            return None
        start, stop = (max(0, min(to_latent(parse_timestamp(action[key])), segment_frames)) for key in ("start_time", "end_time"))
        if stop > start:
            spans.append([start, stop, index])
    if not spans:
        return None
    spans.sort(key=lambda span: (span[0], span[1]))
    if any(spans[i][0] < spans[i - 1][1] for i in range(1, len(spans))):
        tqdm.write("[Hy-Nomas] Overlapping actions in a segment: their per-action attention is disabled for that segment.")
        return None
    windows, cursor = [], 0
    for start, stop, index in spans:
        if start > cursor:
            windows.append([cursor, start, -1])
        windows.append([start, stop, index])
        cursor = stop
    if cursor < segment_frames:
        windows.append([cursor, segment_frames, -1])
    return None if len(windows) == 1 and windows[0][2] == 0 and len(actions) == 1 else windows
