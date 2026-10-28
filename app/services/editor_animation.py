"""Pure helpers for Editor keyframe and color normalization/render expressions."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class EditorAnimationError(ValueError):
    """Invalid animation or color data in an Editor project."""


KEYFRAME_VALUE_RANGES: dict[str, tuple[float, float]] = {
    "x": (-32768.0, 32768.0),
    "y": (-32768.0, 32768.0),
    "scale": (0.05, 4.0),
    "rotation": (-360.0, 360.0),
    "opacity": (0.0, 1.0),
    "volume": (0.0, 4.0),
}
KEYFRAME_EASINGS = {"linear", "hold", "ease-in", "ease-out", "ease-in-out"}
EDITOR_COLOR_RANGES: dict[str, tuple[float, float, float]] = {
    "exposure": (0.0, -2.0, 2.0),
    "contrast": (1.0, 0.0, 2.0),
    "saturation": (1.0, 0.0, 2.0),
    "temperature": (0.0, -1.0, 1.0),
}
MAX_KEYFRAMES_PER_PROPERTY = 1000


def _strict_finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise EditorAnimationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise EditorAnimationError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise EditorAnimationError(f"{label} must be a finite number")
    return number


def normalize_editor_keyframes(
    raw_keyframes: Any,
    *,
    duration: float,
) -> dict[str, list[dict[str, Any]]]:
    """Validate and sort a clip's keyframes without accepting expression text."""
    if raw_keyframes is None:
        return {}
    if not isinstance(raw_keyframes, Mapping):
        raise EditorAnimationError("keyframes must be an object")
    duration = _strict_finite_number(duration, label="clip duration")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for property_name, raw_points in raw_keyframes.items():
        name = str(property_name)
        if name not in KEYFRAME_VALUE_RANGES:
            raise EditorAnimationError(f"unsupported keyframe property '{name}'")
        if not isinstance(raw_points, list):
            raise EditorAnimationError(f"keyframes.{name} must be an array")
        if len(raw_points) > MAX_KEYFRAMES_PER_PROPERTY:
            raise EditorAnimationError(
                f"keyframes.{name} has more than {MAX_KEYFRAMES_PER_PROPERTY} points"
            )
        points: list[dict[str, Any]] = []
        low, high = KEYFRAME_VALUE_RANGES[name]
        for point_index, raw_point in enumerate(raw_points):
            point_label = f"keyframes.{name} point {point_index + 1}"
            if not isinstance(raw_point, Mapping):
                raise EditorAnimationError(f"{point_label} must be an object")
            point_time = _strict_finite_number(
                raw_point.get("time"), label=f"{point_label} time"
            )
            if point_time < 0 or point_time > duration:
                raise EditorAnimationError(
                    f"{point_label} time must be between 0 and clip duration"
                )
            point_value = _strict_finite_number(
                raw_point.get("value"), label=f"{point_label} value"
            )
            if point_value < low or point_value > high:
                raise EditorAnimationError(
                    f"{point_label} value must be between {low:g} and {high:g}"
                )
            easing = str(raw_point.get("easing") or "linear").strip().lower()
            if easing not in KEYFRAME_EASINGS:
                raise EditorAnimationError(
                    f"{point_label} easing must be one of {', '.join(sorted(KEYFRAME_EASINGS))}"
                )
            points.append({"time": point_time, "value": point_value, "easing": easing})
        points.sort(key=lambda point: point["time"])
        for previous, current in zip(points, points[1:]):
            if current["time"] == previous["time"]:
                raise EditorAnimationError(
                    f"keyframes.{name} cannot contain duplicate times"
                )
        if points:
            normalized[name] = points
    return normalized


def normalize_editor_color(raw_color: Any) -> dict[str, float] | None:
    """Fill defaults and clamp the bounded, static Editor color controls."""
    if raw_color is None:
        return None
    if not isinstance(raw_color, Mapping):
        raise EditorAnimationError("color must be an object")
    unknown = set(raw_color) - set(EDITOR_COLOR_RANGES)
    if unknown:
        raise EditorAnimationError(
            f"unsupported color property '{sorted(str(value) for value in unknown)[0]}'"
        )
    normalized: dict[str, float] = {}
    for name, (default, low, high) in EDITOR_COLOR_RANGES.items():
        value = _strict_finite_number(raw_color.get(name, default), label=f"color.{name}")
        normalized[name] = min(high, max(low, value))
    return normalized


def format_ffmpeg_number(value: float) -> str:
    """Render only already-validated finite numeric constants into expressions."""
    if not math.isfinite(value):
        raise EditorAnimationError("FFmpeg expression values must be finite")
    if abs(value) < 0.000000005:
        value = 0.0
    return f"{value:.8f}"


def editor_keyframe_expression(
    keyframes: Mapping[str, list[Mapping[str, Any]]],
    property_name: str,
    *,
    fallback: float,
    time_expression: str,
) -> str:
    """Build a whitelist-only FFmpeg expression matching Editor interpolation."""
    points = keyframes.get(property_name) or []
    if not points:
        return format_ffmpeg_number(fallback)

    def segment_expression(index: int) -> str:
        point = points[index]
        next_point = points[index + 1]
        start = float(point["time"])
        end = float(next_point["time"])
        span = end - start
        value = float(point["value"])
        delta = float(next_point["value"]) - value
        u = f"(({time_expression}-{format_ffmpeg_number(start)})/{format_ffmpeg_number(span)})"
        easing = str(point["easing"])
        if easing == "hold":
            curve = "0"
        elif easing == "ease-in":
            curve = f"({u})*({u})"
        elif easing == "ease-out":
            curve = f"(1-(1-({u}))*(1-({u})))"
        elif easing == "ease-in-out":
            curve = f"(3*({u})*({u})-2*({u})*({u})*({u}))"
        else:
            curve = u
        return (
            f"({format_ffmpeg_number(value)}+({format_ffmpeg_number(delta)})*({curve}))"
        )

    def choose_segment(first: int, last: int) -> str:
        if first == last:
            return segment_expression(first)
        middle = (first + last) // 2
        pivot = format_ffmpeg_number(float(points[middle + 1]["time"]))
        return (
            f"if(lt({time_expression},{pivot}),"
            f"{choose_segment(first, middle)},{choose_segment(middle + 1, last)})"
        )

    first_time = format_ffmpeg_number(float(points[0]["time"]))
    first_value = format_ffmpeg_number(float(points[0]["value"]))
    if len(points) == 1:
        return first_value
    last_time = format_ffmpeg_number(float(points[-1]["time"]))
    last_value = format_ffmpeg_number(float(points[-1]["value"]))
    segment_tree = choose_segment(0, len(points) - 2)
    return (
        f"if(lt({time_expression},{first_time}),{first_value},"
        f"if(gte({time_expression},{last_time}),{last_value},{segment_tree}))"
    )


def evaluate_editor_keyframes(
    points: list[Mapping[str, Any]],
    time_seconds: float,
    *,
    fallback: float,
) -> float:
    """Reference evaluator used by tests to check generated FFmpeg expressions."""
    if not points:
        return fallback
    if time_seconds <= float(points[0]["time"]):
        return float(points[0]["value"])
    for point, next_point in zip(points, points[1:]):
        start = float(point["time"])
        end = float(next_point["time"])
        if time_seconds < end:
            if point["easing"] == "hold":
                ratio = 0.0
            else:
                ratio = min(1.0, max(0.0, (time_seconds - start) / (end - start)))
                if point["easing"] == "ease-in":
                    ratio *= ratio
                elif point["easing"] == "ease-out":
                    ratio = 1.0 - (1.0 - ratio) ** 2
                elif point["easing"] == "ease-in-out":
                    ratio = 3.0 * ratio**2 - 2.0 * ratio**3
            return float(point["value"]) + (
                float(next_point["value"]) - float(point["value"])
            ) * ratio
    return float(points[-1]["value"])


def editor_color_filters(color: Mapping[str, Any] | None) -> list[str]:
    """Emit bounded FFmpeg filters in the documented Editor color order."""
    if not color:
        return []
    exposure = float(color["exposure"])
    contrast = float(color["contrast"])
    saturation = float(color["saturation"])
    temperature = float(color["temperature"])
    filters: list[str] = []
    if abs(exposure) > 1e-9:
        filters.append(f"exposure=exposure={format_ffmpeg_number(exposure)}:black=0")
    if abs(contrast - 1.0) > 1e-9:
        filters.append(
            f"eq=contrast={format_ffmpeg_number(contrast)}:saturation=1:brightness=0"
        )
    if abs(saturation - 1.0) > 1e-9:
        filters.append(
            f"eq=contrast=1:saturation={format_ffmpeg_number(saturation)}:brightness=0"
        )
    if abs(temperature) > 1e-9:
        red_gain = 1.0 + 0.2 * temperature
        blue_gain = 1.0 - 0.2 * temperature
        filters.append(
            "colorchannelmixer="
            f"rr={format_ffmpeg_number(red_gain)}:gg=1:bb={format_ffmpeg_number(blue_gain)}"
        )
    return filters
