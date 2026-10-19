"""Compile readable absolute-pixel MOSS Image layout drafts.

The public layout vocabulary and conversion semantics follow the OpenMOSS
MOSS-Image prompt-enhancer references:
https://github.com/OpenMOSS/MOSS-Image/tree/95559c93ce9f85967de70435deefc1153a1a2fab/skills/moss-image-prompt-enhancer

This small in-process helper intentionally handles new layout drafts only; it
does not interpret legacy, already-centered prompts.
"""

from __future__ import annotations

import json
import math
import re


_MAX_PIXELS = 4_194_304
_ROOT_FIELDS = frozenset(
    {"width", "height", "bbox_coord_space", "bbox_format", "summary", "items"}
)
_SUMMARY_FIELDS = frozenset({"caption", "meta", "canvas", "palette"})
_ITEM_FIELDS = frozenset({"label", "caption", "bbox", "text", "tags", "palette"})
_CANVAS_KINDS = frozenset({"single", "multi_panel", "collage"})
_LABELS = frozenset({"object", "character", "text"})
_RESOLUTION = re.compile(r"(\d+)\s*[xX×]\s*(\d+)")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fields(value, allowed: frozenset[str], required: frozenset[str], location: str) -> None:
    _require(isinstance(value, dict), f"{location} must be an object")
    unknown = [key for key in value if key not in allowed]
    missing = required.difference(value)
    _require(
        not unknown,
        f"{location}: unknown fields {sorted((repr(key) for key in unknown))}",
    )
    _require(not missing, f"{location}: missing fields {sorted(missing)}")


def _nonempty_text(value, location: str) -> None:
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{location} must be nonempty text",
    )


def _string_array(value, location: str) -> None:
    _require(isinstance(value, list), f"{location} must be an array")
    for index, item in enumerate(value):
        _nonempty_text(item, f"{location}[{index}]")


def _canvas_dimensions(width, height, location: str = "Layout") -> None:
    for name, value in (("width", width), ("height", height)):
        _require(
            type(value) is int and value > 0 and value % 16 == 0,
            f"{location} {name} must be a positive integer multiple of 16",
        )
    _require(
        width * height <= _MAX_PIXELS,
        f"{location} canvas exceeds {_MAX_PIXELS} pixels",
    )


def _selected_canvas(resolution: str) -> tuple[int, int]:
    match = _RESOLUTION.fullmatch(resolution)
    _require(match is not None, "MOSS Image resolution must be WIDTHxHEIGHT")
    width, height = map(int, match.groups())
    _canvas_dimensions(width, height, "Selected canvas")
    return width, height


def _centered_box(box, width: int, height: int, location: str) -> list[int]:
    _require(isinstance(box, list) and len(box) == 4, f"{location} must be [x1,y1,x2,y2]")
    valid_numbers = all(
        type(value) in (int, float)
        and (type(value) is int or math.isfinite(value))
        for value in box
    )
    _require(valid_numbers, f"{location} must contain four finite nonboolean numbers")

    x1, y1, x2, y2 = box
    _require(x1 < x2 and y1 < y2, f"{location} must have positive width and height")
    _require(
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
        f"{location} exceeds the {width}x{height} canvas",
    )

    # Match the upstream export: round noise first, enclose fractional edges in
    # pixels, then center and reorder from xyxy to yxyx.
    x1, y1, x2, y2 = (round(value, 9) for value in (x1, y1, x2, y2))
    result = [
        math.floor(y1) - height // 2,
        math.floor(x1) - width // 2,
        math.ceil(y2) - height // 2,
        math.ceil(x2) - width // 2,
    ]
    _require(
        result[0] < result[2] and result[1] < result[3],
        f"{location} is too small at export precision",
    )
    return result


def compile_layout_draft(draft: dict, resolution: str | None = None) -> str:
    """Validate and compact a readable layout draft for the MOSS Image prompt.

    Draft boxes use absolute top-left pixel ``[x1, y1, x2, y2]`` coordinates.
    When *resolution* is supplied, the draft canvas must match it exactly.
    The returned prompt contains centered ``[y1, x1, y2, x2]`` integer boxes
    and omits width and height.
    """

    _fields(draft, _ROOT_FIELDS, _ROOT_FIELDS, "Layout draft")
    width, height = draft["width"], draft["height"]
    _canvas_dimensions(width, height)
    _require(draft["bbox_coord_space"] == "abs", "Draft bbox_coord_space must be abs")
    _require(draft["bbox_format"] == "xyxy", "Draft bbox_format must be xyxy")

    if resolution is not None:
        _require(isinstance(resolution, str), "MOSS Image resolution must be text")
        selected_width, selected_height = _selected_canvas(resolution)
        _require(
            (width, height) == (selected_width, selected_height),
            f"Layout canvas {width}x{height} does not match selected canvas "
            f"{selected_width}x{selected_height}",
        )

    summary = draft["summary"]
    _fields(summary, _SUMMARY_FIELDS, frozenset({"caption"}), "summary")
    _nonempty_text(summary["caption"], "summary.caption")
    if "canvas" in summary:
        _require(
            type(summary["canvas"]) is str and summary["canvas"] in _CANVAS_KINDS,
            "summary.canvas must be single, multi_panel or collage",
        )
    for key in ("meta", "palette"):
        if key in summary:
            _string_array(summary[key], f"summary.{key}")

    items = draft["items"]
    _require(isinstance(items, list) and bool(items), "items must be a nonempty flat array")
    compiled_items = []
    for index, item in enumerate(items):
        location = f"items[{index}]"
        _fields(item, _ITEM_FIELDS, frozenset({"label", "caption", "bbox"}), location)
        _require(
            type(item["label"]) is str and item["label"] in _LABELS,
            f"{location}.label must be object, character or text",
        )
        _nonempty_text(item["caption"], f"{location}.caption")
        if item["label"] == "text":
            _require("text" in item, f"{location}: text items require literal text")
        if "text" in item:
            _nonempty_text(item["text"], f"{location}.text")
        for key in ("tags", "palette"):
            if key in item:
                _string_array(item[key], f"{location}.{key}")

        compiled_item = dict(item)
        compiled_item["bbox"] = _centered_box(item["bbox"], width, height, f"{location}.bbox")
        compiled_items.append(compiled_item)

    result = {
        "summary": summary,
        "items": compiled_items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    _require(math.isfinite(value), f"Invalid JSON number: {token}")
    return value


def _reject_constant(token: str):
    raise ValueError(f"Invalid JSON number: {token}")


def strict_loads(text: str) -> dict:
    """Load one JSON object, rejecting duplicate keys and non-finite numbers."""

    _require(isinstance(text, str), "JSON input must be text")
    result = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_float=_finite_float,
        parse_constant=_reject_constant,
    )
    _require(isinstance(result, dict), "JSON document must be an object")
    return result
