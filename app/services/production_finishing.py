"""Build an editable, agent-authored cut using the native Editor document.

Inputs are explicit ordered media, timings, graphics and gains. No writer or
render is invoked here; the caller authorizes paths and submits export separately.
"""
from __future__ import annotations

import copy
import math
import uuid
from typing import Any, Callable

from services.editor_animation import KEYFRAME_EASINGS, KEYFRAME_VALUE_RANGES, MAX_KEYFRAMES_PER_PROPERTY
from services.editor_projects import create_editor_project, normalize_editor_project


def number(value, label, low=0, high=86400):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be a finite number between {low} and {high}.")
    return value


def _named(value: Any, label: str, *, limit: int = 80) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{label} must be nonempty text up to {limit} characters.")
    return value.strip()


def _overlay_transform(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) - {"x", "y", "scale", "rotation"}:
        raise ValueError("Overlay transform accepts x, y, scale and zero rotation only.")
    result = {}
    for axis in ("x", "y"):
        if axis in value:
            result[axis] = number(value[axis], f"overlay transform {axis}", -32768, 32768)
    if "scale" in value:
        result["scale"] = number(value["scale"], "overlay transform scale", 0.05, 4)
    if "rotation" in value:
        result["rotation"] = number(value["rotation"], "overlay transform rotation", 0, 0)
    return result


def _overlay_keyframes(value: Any, duration: float) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise ValueError("Overlay keyframes must be an object.")
    supported = {"x", "y", "scale", "opacity"}
    if set(value) - supported:
        raise ValueError("Overlay keyframes support x, y, scale and opacity only.")
    checked: dict[str, list[dict[str, Any]]] = {}
    for property_name, points in value.items():
        if not isinstance(points, list) or len(points) > MAX_KEYFRAMES_PER_PROPERTY:
            raise ValueError(f"Overlay keyframes.{property_name} must contain at most {MAX_KEYFRAMES_PER_PROPERTY} points.")
        low, high = KEYFRAME_VALUE_RANGES[property_name]
        checked_points = []
        for index, point in enumerate(points, start=1):
            label = f"overlay keyframes.{property_name} point {index}"
            if not isinstance(point, dict) or set(point) - {"time", "value", "easing"} or not {"time", "value"} <= set(point):
                raise ValueError(f"{label} needs time/value and optional easing only.")
            at = number(point["time"], f"{label} time", 0, duration)
            value_number = number(point["value"], f"{label} value", low, high)
            easing = point.get("easing", "linear")
            if not isinstance(easing, str) or easing not in KEYFRAME_EASINGS:
                raise ValueError(f"{label} easing must be one of {', '.join(sorted(KEYFRAME_EASINGS))}.")
            checked_points.append({"time": at, "value": value_number, "easing": easing})
        checked[property_name] = checked_points
    return checked


def build_finish_project(body: dict, resolve: Callable[[Any], dict]) -> dict:
    """Resolve each asset into native probe metadata plus a trusted local path."""
    allowed_body = {"name", "workspace", "canvas", "clips", "narration", "music", "sfx", "overlays", "speech_ranges", "duck_music", "export"}
    if not isinstance(body, dict) or set(body) - allowed_body:
        raise ValueError("Use name, workspace, canvas, clips, overlays, narration, music, sfx and export only.")
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
        resolved_duration = asset.get("duration", 0)
        if type(resolved_duration) not in (int, float) or not math.isfinite(resolved_duration) or resolved_duration < 0:
            raise ValueError("Resolved media duration must be a finite nonnegative number.")
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

    overlays = body.get("overlays", [])
    if overlays is None:
        overlays = []
    if not isinstance(overlays, list) or len(overlays) > 96:
        raise ValueError("overlays must contain at most 96 editable graphic compositions.")
    for index, overlay in enumerate(overlays, start=1):
        allowed = {"name", "start", "duration", "graphic", "transform", "keyframes", "fade_in", "fade_out", "opacity"}
        if not isinstance(overlay, dict) or set(overlay) - allowed or "graphic" not in overlay:
            raise ValueError("Each overlay needs graphic and only accepts name, timing, transform, keyframes, fades and opacity.")
        start = number(overlay.get("start", 0), "overlay start", 0, position)
        duration = number(overlay.get("duration"), "overlay duration", 1 / fps, position)
        if start + duration > position:
            raise ValueError("An overlay must end within the finished film.")
        fade_in = number(overlay.get("fade_in", 0), "overlay fade_in", 0, duration)
        fade_out = number(overlay.get("fade_out", 0), "overlay fade_out", 0, duration)
        opacity = number(overlay.get("opacity", 1), "overlay opacity", 0, 1)
        transform = _overlay_transform(overlay["transform"]) if "transform" in overlay else {}
        keyframes = _overlay_keyframes(overlay["keyframes"], duration) if "keyframes" in overlay else {}

        # MCP image elements carry authorized media references. Convert each to
        # a project-owned image asset before the shared native graphic validator.
        graphic = copy.deepcopy(overlay["graphic"])
        if isinstance(graphic, dict) and isinstance(graphic.get("elements"), list):
            converted = []
            for element in graphic["elements"]:
                if isinstance(element, dict) and element.get("type") == "image":
                    if "asset_id" in element:
                        raise ValueError("Image elements use an authorized asset reference, not a project asset_id.")
                    if "asset" not in element:
                        raise ValueError("Each image graphic element needs an authorized asset reference.")
                    asset_id, asset = add_asset(element["asset"])
                    if asset["type"] != "image":
                        raise ValueError("Graphic image elements need an image asset.")
                    element.pop("asset")
                    element["asset_id"] = asset_id
                converted.append(element)
            graphic["elements"] = converted

        track_name = _named(overlay.get("name", f"Overlay {index}"), "overlay name")
        item = {
            "id": uuid.uuid4().hex[:16], "start": start, "duration": duration,
            "source_in": 0, "speed": 1, "volume": 1, "opacity": opacity,
            "fade_in": fade_in, "fade_out": fade_out, "transform": transform,
            "graphic": graphic,
        }
        if keyframes:
            item["keyframes"] = keyframes
        project["tracks"].append({
            "id": uuid.uuid4().hex[:16], "name": track_name, "type": "text",
            "z_index": 100 + index, "muted": False, "locked": False,
            "items": [item],
        })

    speech = []
    raw_ranges = body.get("speech_ranges", [])
    if raw_ranges is None:
        raw_ranges = []
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

    audio_track_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for category in ("narration", "music", "sfx"):
        rows = body.get(category, [])
        if rows is None:
            rows = []
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or len(rows) > 100:
            raise ValueError(f"{category} must contain at most 100 explicitly timed audio assets.")
        for row in rows:
            allowed_row = {"asset", "start", "duration", "source_in", "gain_db", "fade_in", "fade_out", "track", "name"}
            if not isinstance(row, dict) or set(row) - allowed_row or "asset" not in row:
                raise ValueError(f"Each {category} item needs asset and explicit optional timing/gain/fades, track and name.")
            identifier, asset = add_asset(row["asset"])
            if not asset.get("has_audio"):
                raise ValueError(f"The selected {category} asset has no audio stream.")
            start = number(row.get("start", 0), "audio start", 0, position)
            remaining = max(0, float(asset.get("duration") or 0) - float(row.get("source_in", 0)))
            adjusted = {**row, "gain_db": row.get("gain_db", -18 if category == "music" else -9 if category == "sfx" else 0)}
            item = make_item(adjusted, identifier, asset, start, min(remaining, position - start))
            if start + item["duration"] > position:
                raise ValueError("Audio must end within the authored film; revise the duration explicitly.")
            if "name" in row:
                item["name"] = _named(row["name"], f"{category} item name", limit=120)
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

            requested_track = _named(row.get("track", category.title()), f"{category} track")
            # Category remains part of the grouping identity so a track label
            # reused in narration and music cannot merge different track types.
            key = (category, requested_track)
            audio_track_groups.setdefault(key, []).append(item)

    for (_category, requested_track), items in audio_track_groups.items():
        # Editor tracks are single-lane timelines and normalize overlaps by
        # moving later items. Partition rows before normalization so explicit
        # concurrent VO, music or SFX remains at its authored start time.
        lanes: list[dict[str, Any]] = []
        for _order, item in sorted(enumerate(items), key=lambda pair: (pair[1]["start"], pair[0])):
            start = item["start"]
            end = start + item["duration"]
            lane = next((candidate for candidate in lanes if start >= candidate["end"] - 1e-9), None)
            if lane is None:
                lane = {"end": end, "items": []}
                lanes.append(lane)
            lane["items"].append(item)
            lane["end"] = end
        for lane_index, lane in enumerate(lanes):
            name = requested_track if lane_index == 0 else f"{requested_track} · {lane_index + 1}"
            project["tracks"].append({
                "id": uuid.uuid4().hex[:16], "name": name, "type": "audio",
                "z_index": 0, "muted": False, "locked": False,
                "items": lane["items"],
            })
    if len(project["tracks"]) > 100:
        raise ValueError("The finished project would exceed the Editor track limit of 100.")
    # Duck only declared speech intervals and actual narration placements.
    # Never invent transcript/voice timing from the written screenplay.
    return normalize_editor_project(project, workspace=project["workspace"])
