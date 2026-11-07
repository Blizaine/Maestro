"""Build an editable, agent-authored cut using the native Editor document.

Inputs are explicit ordered media, timings and gains. No writer or render is
invoked here; the caller authorizes paths and submits export separately.
"""
from __future__ import annotations

import math
import uuid
from typing import Any, Callable

from services.editor_projects import create_editor_project, normalize_editor_project


def number(value, label, low=0, high=86400):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be a finite number between {low} and {high}.")
    return value


def build_finish_project(body: dict, resolve: Callable[[Any], dict]) -> dict:
    """Resolve each asset into native probe metadata plus a trusted local path."""
    if not isinstance(body, dict) or set(body) - {"name", "workspace", "canvas", "clips", "narration", "music", "sfx", "speech_ranges", "duck_music", "export"}:
        raise ValueError("Use name, workspace, canvas, clips, narration, music, sfx and export only.")
    clips = body.get("clips")
    if not isinstance(clips, list) or not 1 <= len(clips) <= 200:
        raise ValueError("clips must contain 1 to 200 ordered takes or title-card assets.")
    canvas = body.get("canvas") or {}
    if not isinstance(canvas, dict) or set(canvas) - {"width", "height", "fps", "background"}:
        raise ValueError("canvas accepts width, height, fps and background.")
    width = number(canvas.get("width", 1280), "width", 64, 4096)
    height = number(canvas.get("height", 720), "height", 64, 4096)
    if int(width) != width or int(height) != height or width % 2 or height % 2:
        raise ValueError("Canvas dimensions must be even integers.")
    fps = number(canvas.get("fps", 24), "fps", 1, 120)
    project = create_editor_project(name=body.get("name") or "Agent finished film", workspace=body.get("workspace") or "default", width=int(width), height=int(height), fps=fps)
    project["export"]["encoder"] = "software"
    if "background" in canvas:
        import re
        if not isinstance(canvas["background"], str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", canvas["background"]):
            raise ValueError("Canvas background must be #RRGGBB.")
        project["canvas"]["background"] = canvas["background"]
    assets = project["assets"]
    main = project["tracks"][0]
    project["tracks"] = [main]

    def add_asset(value):
        asset = resolve(value)
        if not isinstance(asset, dict) or asset.get("type") not in {"video", "audio", "image"} or not asset.get("path"):
            raise ValueError("The selected media could not be resolved.")
        identifier = uuid.uuid4().hex[:16]
        assets[identifier] = {**asset, "id": identifier, "origin": asset.get("origin", "output")}
        return identifier, asset

    def make_item(row, asset_id, asset, start, default_duration):
        source_in = number(row.get("source_in", 0), "source_in")
        duration = number(row.get("duration", default_duration), "duration", 1 / fps)
        available = float(asset.get("duration") or 0)
        if asset["type"] != "image" and (available <= 0 or source_in + duration > available + 1 / fps):
            raise ValueError("An authored trim exceeds the available source media; revise it explicitly.")
        gain = number(row.get("gain_db", 0), "gain_db", -60, 12)
        return {"id": uuid.uuid4().hex[:16], "asset_id": asset_id, "start": start, "duration": duration, "source_in": source_in, "speed": 1,
                "volume": 10 ** (gain / 20), "muted": row.get("include_audio", True) is False,
                "fade_in": number(row.get("fade_in", 0), "fade_in", 0, duration),
                "fade_out": number(row.get("fade_out", 0), "fade_out", 0, duration)}

    position = 0.0
    for row in clips:
        if not isinstance(row, dict) or set(row) - {"asset", "duration", "source_in", "include_audio", "gain_db", "fade_in", "fade_out"} or "asset" not in row:
            raise ValueError("Each clip needs asset and optional explicit trim, gain, audio and fades.")
        if "include_audio" in row and type(row["include_audio"]) is not bool:
            raise ValueError("include_audio must be a boolean.")
        identifier, asset = add_asset(row["asset"])
        if asset["type"] == "audio":
            raise ValueError("Put audio in narration, music or sfx, rather than the video sequence.")
        default = max(0, float(asset.get("duration") or 0) - float(row.get("source_in", 0))) if asset["type"] == "video" else 3
        item = make_item(row, identifier, asset, position, default)
        main["items"].append(item)
        position += item["duration"]
    if position > 86400:
        raise ValueError("Finished films may be at most 24 hours.")

    speech = []
    raw_ranges = body.get("speech_ranges") or []
    if not isinstance(raw_ranges, list) or len(raw_ranges) > 100:
        raise ValueError("speech_ranges accepts at most 100 reviewed [start, end] pairs.")
    for pair in raw_ranges:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each speech range must be [start, end].")
        start = number(pair[0], "speech start", 0, position)
        end = number(pair[1], "speech end", start, position)
        if end <= start:
            raise ValueError("A speech range must have positive duration.")
        speech.append((start, end))
    if type(body.get("duck_music", True)) is not bool:
        raise ValueError("duck_music must be a boolean.")

    for category in ("narration", "music", "sfx"):
        rows = body.get(category) or []
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or len(rows) > 100:
            raise ValueError(f"{category} must contain at most 100 explicitly timed audio assets.")
        track = {"id": uuid.uuid4().hex[:16], "name": category.title(), "type": "audio", "z_index": 0, "muted": False, "locked": False, "items": []}
        for row in rows:
            if not isinstance(row, dict) or set(row) - {"asset", "start", "duration", "source_in", "gain_db", "fade_in", "fade_out"} or "asset" not in row:
                raise ValueError(f"Each {category} item needs asset and explicit optional timing/gain/fades.")
            identifier, asset = add_asset(row["asset"])
            if not asset.get("has_audio"):
                raise ValueError(f"The selected {category} asset has no audio stream.")
            start = number(row.get("start", 0), "audio start", 0, position)
            remaining = max(0, float(asset.get("duration") or 0) - float(row.get("source_in", 0)))
            adjusted = {**row, "gain_db": row.get("gain_db", -18 if category == "music" else -9 if category == "sfx" else 0)}
            item = make_item(adjusted, identifier, asset, start, min(remaining, position - start))
            if start + item["duration"] > position + 1 / fps:
                raise ValueError("Audio extends beyond the authored film; revise the duration explicitly.")
            track["items"].append(item)
            if category == "narration":
                speech.append((start, start + item["duration"]))
            elif category == "music" and body.get("duck_music", True) and speech:
                intervals = [(max(0, a - start), min(item["duration"], b - start)) for a, b in speech if a < start + item["duration"] and b > start]
                points = {0.0, item["duration"]}
                for a, b in intervals:
                    points.update((max(0, a - .1), a, b, min(item["duration"], b + .2)))
                item["keyframes"] = {"volume": [
                    {"time": t, "value": item["volume"] * (10 ** (-12 / 20) if any(a <= t <= b for a, b in intervals) else 1), "easing": "linear"}
                    for t in sorted(points)
                ]}
        if track["items"]:
            project["tracks"].append(track)
    # Duck only declared speech intervals and actual narration placements.
    # Never invent transcript/voice timing from the written screenplay.
    return normalize_editor_project(project, workspace=project["workspace"])
