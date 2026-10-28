"""Chroma key settings and native-resolution FFmpeg filters for Editor clips."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping


EDITOR_KEYING_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "color": "#00ff00",
    "similarity": 0.15,
    "softness": 0.10,
    "spill": 0.5,
}
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


class EditorKeyingError(ValueError):
    """A chroma key setting is malformed or outside its supported range."""


def normalize_editor_keying(value: Any) -> dict[str, Any] | None:
    """Validate and normalize an authored keying object.

    Missing keying stays missing so legacy projects continue through the
    renderer's exact pre-keying graph. A partial object gets UI-compatible
    defaults, while explicit malformed values are rejected rather than
    silently clamped.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EditorKeyingError("keying must be an object")

    enabled = value.get("enabled", EDITOR_KEYING_DEFAULTS["enabled"])
    if not isinstance(enabled, bool):
        raise EditorKeyingError("keying.enabled must be a boolean")

    color = value.get("color", EDITOR_KEYING_DEFAULTS["color"])
    if not isinstance(color, str) or not _HEX_COLOR_RE.fullmatch(color):
        raise EditorKeyingError("keying.color must be a six-digit hex color")

    normalized: dict[str, Any] = {
        "enabled": enabled,
        "color": color.lower(),
    }
    for name, minimum, maximum in (
        ("similarity", 0.00001, 1.0),
        ("softness", 0.0, 1.0),
        ("spill", 0.0, 1.0),
    ):
        raw = value.get(name, EDITOR_KEYING_DEFAULTS[name])
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EditorKeyingError(f"keying.{name} must be a finite number")
        try:
            number = float(raw)
        except (OverflowError, TypeError, ValueError) as error:
            raise EditorKeyingError(
                f"keying.{name} must be a finite number"
            ) from error
        if not math.isfinite(number):
            raise EditorKeyingError(f"keying.{name} must be a finite number")
        if number < minimum or number > maximum:
            raise EditorKeyingError(
                f"keying.{name} must be between {minimum:g} and {maximum:g}"
            )
        normalized[name] = number
    return normalized


def editor_keying_filters(
    keying: Mapping[str, Any],
    *,
    input_label: str,
    output_label: str,
    suffix: str,
) -> list[str]:
    """Build key, despill, and source-alpha-preserving FFmpeg graph steps.

    The input must be the trimmed clip at its native source resolution. The
    returned graph keeps the keyed RGB at that resolution and multiplies the
    newly calculated matte by the source alpha before handing off to later
    resize, transform, color, opacity, and fade processing.
    """
    if not keying.get("enabled"):
        return [f"[{input_label}]null[{output_label}]"]

    color = str(keying["color"])[1:]
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    similarity = float(keying["similarity"])
    softness = float(keying["softness"])
    spill = float(keying["spill"])
    blend = softness if softness > 0.0001 else 0.0

    source_rgb = f"editor_key_source_rgb_{suffix}"
    source_alpha_input = f"editor_key_source_alpha_input_{suffix}"
    source_alpha = f"editor_key_source_alpha_{suffix}"
    keyed_color_input = f"editor_key_color_input_{suffix}"
    keyed_alpha_input = f"editor_key_alpha_input_{suffix}"
    keyed_alpha = f"editor_key_alpha_{suffix}"
    combined_alpha = f"editor_key_combined_alpha_{suffix}"
    keyed_color = f"editor_key_color_{suffix}"

    filters = [
        f"[{input_label}]format=rgba,split=2"
        f"[{source_rgb}][{source_alpha_input}]",
    ]
    key_chain = (
        f"colorkey=0x{color}:similarity={similarity:.8f}:blend={blend:.8f}"
    )
    if spill > 0.0 and green > red and green > blue:
        key_chain += (
            f",despill=type=green:mix=0.5:expand=0:red=0:"
            f"green={-spill:.8f}:blue=0:brightness=0:alpha=0"
        )
    elif spill > 0.0 and blue > red and blue > green:
        key_chain += (
            f",despill=type=blue:mix=0.5:expand=0:red=0:green=0:"
            f"blue={-spill:.8f}:brightness=0:alpha=0"
        )
    filters.extend([
        f"[{source_rgb}]{key_chain},split=2"
        f"[{keyed_color_input}][{keyed_alpha_input}]",
        f"[{source_alpha_input}]format=yuva444p,alphaextract,format=gray"
        f"[{source_alpha}]",
        f"[{keyed_alpha_input}]format=yuva444p,alphaextract,format=gray"
        f"[{keyed_alpha}]",
        f"[{source_alpha}][{keyed_alpha}]"
        f"blend=all_mode=multiply:shortest=1[{combined_alpha}]",
        f"[{keyed_color_input}]format=rgba[{keyed_color}]",
        f"[{keyed_color}][{combined_alpha}]alphamerge[{output_label}]",
    ])
    return filters
