"""Safe, editable graphic overlays for Maestro Editor text tracks.

Graphics stay as small JSON layer descriptions in the project. This module is
the single Pillow renderer used by preview and export, so both paths share the
same wrapping, font selection, image fitting, and alpha behavior.
"""
from __future__ import annotations

import io
import math
import os
import re
import unicodedata
from functools import lru_cache
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps


GRAPHIC_VERSION = 1
GRAPHIC_MAX_ELEMENTS = 40
GRAPHIC_MAX_DESIGN_DIMENSION = 8192
GRAPHIC_MAX_DESIGN_AREA = 33_554_432
GRAPHIC_MAX_OUTPUT_DIMENSION = 8192
GRAPHIC_MAX_OUTPUT_AREA = 33_554_432
GRAPHIC_FONT_FAMILIES = frozenset({
    "Arial", "Arial Black", "Georgia", "Times New Roman", "Verdana",
    "Trebuchet MS", "Courier New", "Impact",
})
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_ROOT_FIELDS = {"version", "width", "height", "elements"}
_COMMON_FIELDS = {"id", "type", "x", "y", "width", "height", "opacity"}
_ELEMENT_FIELDS = {
    "rect": _COMMON_FIELDS | {"fill", "border_color", "border_width", "radius"},
    "text": _COMMON_FIELDS | {
        "text", "font_family", "font_size", "color", "align", "bold",
    },
    "image": _COMMON_FIELDS | {"asset_id", "fit"},
}
_ELEMENT_REQUIRED = {
    "rect": {"id", "type", "x", "y", "width", "height", "fill"},
    "text": {
        "id", "type", "x", "y", "width", "height", "text",
        "font_family", "font_size", "color",
    },
    "image": {"id", "type", "x", "y", "width", "height", "asset_id"},
}


class EditorGraphicError(ValueError):
    """A graphic layer description cannot be safely normalized or rendered."""


def _number(value: Any, label: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditorGraphicError(f"Graphic {label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < low or parsed > high:
        raise EditorGraphicError(
            f"Graphic {label} must be between {low:g} and {high:g}"
        )
    return parsed


def _positive_integer(value: Any, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        raise EditorGraphicError(
            f"Graphic {label} must be a whole number between 1 and {maximum}"
        )
    return value


def _asset_record(assets: Mapping[str, Any], asset_id: str) -> Mapping[str, Any] | None:
    record = assets.get(asset_id)
    if isinstance(record, Mapping):
        return record
    for key, candidate in assets.items():
        if str(key) == asset_id and isinstance(candidate, Mapping):
            return candidate
        if isinstance(candidate, Mapping) and str(candidate.get("id") or "") == asset_id:
            return candidate
    return None


def normalize_editor_graphic(
    raw: Any,
    assets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and copy an editable graphic layer document.

    `assets`, when provided, is the owning Editor project's asset table. Image
    references must point at an existing image asset; path authorization is
    deliberately left to the caller's asset resolver.
    """
    if not isinstance(raw, Mapping):
        raise EditorGraphicError("Graphic must be an object")
    unknown_root = set(raw) - _ROOT_FIELDS
    if unknown_root:
        raise EditorGraphicError(
            f"Graphic has unsupported field '{sorted(unknown_root)[0]}'"
        )
    if set(raw) != _ROOT_FIELDS:
        missing = sorted(_ROOT_FIELDS - set(raw))
        raise EditorGraphicError(f"Graphic is missing required field '{missing[0]}'")
    version = raw.get("version")
    if type(version) is not int or version != GRAPHIC_VERSION:
        raise EditorGraphicError(f"Graphic version must be {GRAPHIC_VERSION}")
    design_width = _positive_integer(
        raw.get("width"), "width", maximum=GRAPHIC_MAX_DESIGN_DIMENSION
    )
    design_height = _positive_integer(
        raw.get("height"), "height", maximum=GRAPHIC_MAX_DESIGN_DIMENSION
    )
    if design_width * design_height > GRAPHIC_MAX_DESIGN_AREA:
        raise EditorGraphicError("Graphic design dimensions are too large")
    elements = raw.get("elements")
    if not isinstance(elements, list) or not elements:
        raise EditorGraphicError("Graphic must contain at least one element")
    if len(elements) > GRAPHIC_MAX_ELEMENTS:
        raise EditorGraphicError(
            f"Graphic supports at most {GRAPHIC_MAX_ELEMENTS} elements"
        )
    if assets is not None and not isinstance(assets, Mapping):
        raise EditorGraphicError("Graphic image assets must be an object")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, element in enumerate(elements, start=1):
        label = f"element {index}"
        if not isinstance(element, Mapping):
            raise EditorGraphicError(f"Graphic {label} must be an object")
        kind = element.get("type")
        if not isinstance(kind, str) or kind not in _ELEMENT_FIELDS:
            raise EditorGraphicError(f"Graphic {label} has an unsupported type")
        unknown = set(element) - _ELEMENT_FIELDS[kind]
        if unknown:
            raise EditorGraphicError(
                f"Graphic {label} has unsupported field '{sorted(unknown)[0]}'"
            )
        missing = _ELEMENT_REQUIRED[kind] - set(element)
        if missing:
            raise EditorGraphicError(
                f"Graphic {label} is missing required field '{sorted(missing)[0]}'"
            )
        element_id = element.get("id")
        if not isinstance(element_id, str) or not _ID_RE.fullmatch(element_id):
            raise EditorGraphicError(f"Graphic {label} id must use letters, numbers, _ or -")
        if element_id in seen_ids:
            raise EditorGraphicError(f"Graphic element id '{element_id}' is repeated")
        seen_ids.add(element_id)
        item: dict[str, Any] = {
            "id": element_id,
            "type": kind,
            "x": _number(element.get("x"), f"{label} x", low=-16_384, high=16_384),
            "y": _number(element.get("y"), f"{label} y", low=-16_384, high=16_384),
            "width": _number(
                element.get("width"), f"{label} width", low=0.01, high=16_384
            ),
            "height": _number(
                element.get("height"), f"{label} height", low=0.01, high=16_384
            ),
            "opacity": _number(
                element.get("opacity", 1.0), f"{label} opacity", low=0.0, high=1.0
            ),
        }
        if kind == "rect":
            fill = element.get("fill")
            if not isinstance(fill, str) or not _COLOR_RE.fullmatch(fill):
                raise EditorGraphicError(f"Graphic {label} fill must be a #RRGGBB color")
            item["fill"] = fill.lower()
            border_color = element.get("border_color")
            if border_color is not None:
                if not isinstance(border_color, str) or not _COLOR_RE.fullmatch(border_color):
                    raise EditorGraphicError(
                        f"Graphic {label} border_color must be a #RRGGBB color"
                    )
                item["border_color"] = border_color.lower()
            item["border_width"] = _number(
                element.get("border_width", 0),
                f"{label} border_width",
                low=0,
                high=2048,
            )
            item["radius"] = _number(
                element.get("radius", 0), f"{label} radius", low=0, high=4096
            )
        elif kind == "text":
            text = element.get("text")
            if not isinstance(text, str):
                raise EditorGraphicError(f"Graphic {label} text must be a string")
            if len(text) > 10_000:
                raise EditorGraphicError(f"Graphic {label} text is longer than 10000 characters")
            if any(
                unicodedata.category(character) == "Cc"
                and character not in "\n\r\t"
                for character in text
            ):
                raise EditorGraphicError(f"Graphic {label} text contains a control character")
            font_family = element.get("font_family")
            if not isinstance(font_family, str) or font_family not in GRAPHIC_FONT_FAMILIES:
                raise EditorGraphicError(
                    f"Graphic {label} font_family must be one of the supported Editor fonts"
                )
            color = element.get("color")
            if not isinstance(color, str) or not _COLOR_RE.fullmatch(color):
                raise EditorGraphicError(f"Graphic {label} color must be a #RRGGBB color")
            align = element.get("align", "left")
            if not isinstance(align, str) or align not in {"left", "center", "right"}:
                raise EditorGraphicError(
                    f"Graphic {label} align must be left, center, or right"
                )
            bold = element.get("bold", False)
            if not isinstance(bold, bool):
                raise EditorGraphicError(f"Graphic {label} bold must be true or false")
            item.update({
                "text": text,
                "font_family": font_family,
                "font_size": _positive_integer(
                    element.get("font_size"), "text font_size", maximum=1024
                ),
                "color": color.lower(),
                "align": align,
                "bold": bold,
            })
        else:
            asset_id = element.get("asset_id")
            if not isinstance(asset_id, str) or not _ID_RE.fullmatch(asset_id):
                raise EditorGraphicError(f"Graphic {label} asset_id is invalid")
            fit = element.get("fit", "contain")
            if not isinstance(fit, str) or fit not in {"contain", "cover"}:
                raise EditorGraphicError(f"Graphic {label} fit must be contain or cover")
            if assets is not None:
                asset = _asset_record(assets, asset_id)
                if asset is None:
                    raise EditorGraphicError(
                        f"Graphic {label} references missing image asset '{asset_id}'"
                    )
                if str(asset.get("type") or "").lower() != "image":
                    raise EditorGraphicError(
                        f"Graphic {label} asset '{asset_id}' must be an image"
                    )
            item.update({"asset_id": asset_id, "fit": fit})
        normalized.append(item)

    return {
        "version": GRAPHIC_VERSION,
        "width": design_width,
        "height": design_height,
        "elements": normalized,
    }


def _font_candidates(family: str, bold: bool) -> list[str]:
    windows_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    font_names = {
        "Arial": ("arial.ttf", "arialbd.ttf"),
        "Arial Black": ("ariblk.ttf", "ariblk.ttf"),
        "Georgia": ("georgia.ttf", "georgiab.ttf"),
        "Times New Roman": ("times.ttf", "timesbd.ttf"),
        "Verdana": ("verdana.ttf", "verdanab.ttf"),
        "Trebuchet MS": ("trebuc.ttf", "trebucbd.ttf"),
        "Courier New": ("cour.ttf", "courbd.ttf"),
        "Impact": ("impact.ttf", "impact.ttf"),
    }
    linux_regular = {
        "Arial": "LiberationSans-Regular.ttf",
        "Arial Black": "LiberationSans-Bold.ttf",
        "Georgia": "LiberationSerif-Regular.ttf",
        "Times New Roman": "LiberationSerif-Regular.ttf",
        "Verdana": "LiberationSans-Regular.ttf",
        "Trebuchet MS": "LiberationSans-Regular.ttf",
        "Courier New": "LiberationMono-Regular.ttf",
        "Impact": "LiberationSans-Bold.ttf",
    }
    linux_bold = {
        "Arial": "LiberationSans-Bold.ttf",
        "Arial Black": "LiberationSans-Bold.ttf",
        "Georgia": "LiberationSerif-Bold.ttf",
        "Times New Roman": "LiberationSerif-Bold.ttf",
        "Verdana": "LiberationSans-Bold.ttf",
        "Trebuchet MS": "LiberationSans-Bold.ttf",
        "Courier New": "LiberationMono-Bold.ttf",
        "Impact": "LiberationSans-Bold.ttf",
    }
    unix_name = linux_bold[family] if bold else linux_regular[family]
    system_supplemental = f"/System/Library/Fonts/Supplemental/{family}.ttf"
    names = font_names[family]
    chosen_windows = names[1 if bold else 0]
    candidates = [
        os.environ.get("MAESTRO_EDITOR_FONT", "").strip(),
        os.path.join(windows_fonts, chosen_windows),
        system_supplemental,
        f"/usr/share/fonts/truetype/liberation2/{unix_name}",
        f"/usr/share/fonts/truetype/liberation/{unix_name}",
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
    ]
    return [path for path in candidates if path and os.path.isfile(path)]


@lru_cache(maxsize=64)
def _load_font(family: str, bold: bool, pixel_size: int) -> ImageFont.FreeTypeFont:
    candidates = _font_candidates(family, bold)
    if not candidates:
        raise EditorGraphicError(
            f"No font file is installed for {family}; install a supported font to render this graphic"
        )
    try:
        return ImageFont.truetype(candidates[0], pixel_size)
    except OSError as error:
        raise EditorGraphicError(
            f"Could not load the installed {family} font for this graphic"
        ) from error


def _color_with_opacity(color: str, opacity: float) -> tuple[int, int, int, int]:
    return (
        int(color[1:3], 16),
        int(color[3:5], 16),
        int(color[5:7], 16),
        max(0, min(255, int(round(opacity * 255)))),
    )


def _assert_glyphs_supported(text: str, font: ImageFont.FreeTypeFont, element_id: str) -> None:
    try:
        missing_mask = font.getmask("\U0010FFFF")
        missing_size = font.getlength("\U0010FFFF")
    except Exception:
        missing_mask = None
        missing_size = None
    for character in dict.fromkeys(text):
        if character in "\r\n\t" or unicodedata.category(character) in {"Cf", "Cc"}:
            continue
        try:
            char_mask = font.getmask(character)
            char_size = font.getlength(character)
        except (UnicodeEncodeError, OSError, ValueError) as error:
            codepoint = f"U+{ord(character):04X}"
            raise EditorGraphicError(
                f"Graphic text layer '{element_id}' contains unsupported glyph {codepoint}; "
                "choose another font or replace that character"
            ) from error
        if (
            missing_mask is not None
            and char_size == missing_size
            and char_mask.size == missing_mask.size
            and bytes(char_mask) == bytes(missing_mask)
        ):
            codepoint = f"U+{ord(character):04X}"
            raise EditorGraphicError(
                f"Graphic text layer '{element_id}' contains unsupported glyph {codepoint}; "
                "choose another font or replace that character"
            )


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float, element_id: str) -> list[str]:
    lines: list[str] = []
    for explicit_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if explicit_line == "":
            lines.append("")
            continue
        line = ""
        for token in re.findall(r"\s+|\S+", explicit_line):
            candidate = line + token
            if font.getlength(candidate) <= max_width:
                line = candidate
                continue
            if line:
                lines.append(line)
                line = ""
            if font.getlength(token) <= max_width:
                line = token
                continue
            # Keep long words, whitespace, and unspaced scripts visible by
            # wrapping at character boundaries. Combining marks stay with
            # their base.
            fragments: list[str] = []
            fragment = ""
            for character in token:
                candidate_fragment = fragment + character
                if fragment and unicodedata.combining(character) == 0 and font.getlength(candidate_fragment) > max_width:
                    fragments.append(fragment)
                    fragment = character
                else:
                    fragment = candidate_fragment
                if font.getlength(fragment) > max_width:
                    raise EditorGraphicError(
                        f"Graphic text layer '{element_id}' has a character wider than its text box; "
                        "increase the box width or reduce the font size"
                    )
            if fragments:
                lines.extend(fragments)
            line = fragment
        if line:
            lines.append(line)
    return lines


def _draw_text_element(
    canvas: Image.Image,
    element: Mapping[str, Any],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> None:
    font_size = max(1, int(round(float(element["font_size"]) * (scale_x + scale_y) / 2)))
    if font_size > 2048:
        raise EditorGraphicError(
            f"Graphic text layer '{element['id']}' scales above 2048 pixels; reduce font_size or use a larger design canvas"
        )
    font = _load_font(str(element["font_family"]), bool(element["bold"]), font_size)
    _assert_glyphs_supported(str(element["text"]), font, str(element["id"]))
    x = float(element["x"]) * scale_x + offset_x
    y = float(element["y"]) * scale_y + offset_y
    box_width = float(element["width"]) * scale_x
    box_height = float(element["height"]) * scale_y
    lines = _wrap_text(str(element["text"]), font, box_width, str(element["id"]))
    ascent, descent = font.getmetrics()
    line_height = max(1, ascent + descent)
    if len(lines) * line_height > box_height + 1e-6:
        raise EditorGraphicError(
            f"Graphic text layer '{element['id']}' does not fit its text box; "
            "increase the box height or reduce the font size"
        )
    draw = ImageDraw.Draw(canvas)
    color = _color_with_opacity(str(element["color"]), float(element["opacity"]))
    align = str(element["align"])
    for row, line in enumerate(lines):
        line_width = font.getlength(line)
        if line_width > box_width + 1e-6:
            raise EditorGraphicError(
                f"Graphic text layer '{element['id']}' does not fit its text box; "
                "increase the box width or reduce the font size"
            )
        if align == "center":
            text_x = x + max(0.0, (box_width - line_width) / 2)
        elif align == "right":
            text_x = x + max(0.0, box_width - line_width)
        else:
            text_x = x
        text_y = y + row * line_height
        draw.text((text_x, text_y), line, font=font, fill=color, anchor="lt")


def _resolved_image(value: Any, element_id: str) -> Image.Image:
    if isinstance(value, Image.Image):
        if value.width <= 0 or value.height <= 0 or value.width * value.height > GRAPHIC_MAX_OUTPUT_AREA:
            raise EditorGraphicError(
                f"Graphic image layer '{element_id}' exceeds the supported image size"
            )
        image = value.copy()
    elif isinstance(value, (str, os.PathLike)):
        try:
            with Image.open(value) as opened:
                # Inspect the header dimensions before decoding the full image
                # into memory. Pillow's own decompression-bomb ceiling is much
                # higher than the Editor limit below.
                if (
                    opened.width <= 0
                    or opened.height <= 0
                    or opened.width * opened.height > GRAPHIC_MAX_OUTPUT_AREA
                ):
                    raise EditorGraphicError(
                        f"Graphic image layer '{element_id}' exceeds the supported image size"
                    )
                image = ImageOps.exif_transpose(opened).convert("RGBA")
        except EditorGraphicError:
            raise
        except (OSError, ValueError) as error:
            raise EditorGraphicError(
                f"Graphic image layer '{element_id}' could not be opened"
            ) from error
    else:
        raise EditorGraphicError(
            f"Graphic image layer '{element_id}' resolver must return an image or authorized path"
        )
    if image.width <= 0 or image.height <= 0 or image.width * image.height > GRAPHIC_MAX_OUTPUT_AREA:
        raise EditorGraphicError(
            f"Graphic image layer '{element_id}' exceeds the supported image size"
        )
    return image.convert("RGBA")


def _draw_image_element(
    canvas: Image.Image,
    element: Mapping[str, Any],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
    resolve_image: Any,
) -> None:
    asset_id = str(element["asset_id"])
    try:
        resolved = resolve_image(asset_id)
    except EditorGraphicError:
        raise
    except Exception as error:
        raise EditorGraphicError(
            f"Graphic image layer '{element['id']}' could not resolve image asset '{asset_id}'"
        ) from error
    source = _resolved_image(resolved, str(element["id"]))
    x = float(element["x"]) * scale_x + offset_x
    y = float(element["y"]) * scale_y + offset_y
    box_width = max(1.0, float(element["width"]) * scale_x)
    box_height = max(1.0, float(element["height"]) * scale_y)
    if element["fit"] == "contain":
        ratio = min(box_width / source.width, box_height / source.height)
        drawn_width = source.width * ratio
        drawn_height = source.height * ratio
        drawn_x = x + (box_width - drawn_width) / 2
        drawn_y = y + (box_height - drawn_height) / 2
        crop_x0, crop_y0, crop_x1, crop_y1 = 0.0, 0.0, float(source.width), float(source.height)
    else:
        target_ratio = box_width / box_height
        source_ratio = source.width / source.height
        if source_ratio > target_ratio:
            crop_width = source.height * target_ratio
            crop_x0 = (source.width - crop_width) / 2
            crop_y0, crop_x1, crop_y1 = 0.0, crop_x0 + crop_width, float(source.height)
        else:
            crop_height = source.width / target_ratio
            crop_y0 = (source.height - crop_height) / 2
            crop_x0, crop_x1, crop_y1 = 0.0, float(source.width), crop_y0 + crop_height
        drawn_x, drawn_y = x, y
        drawn_width, drawn_height = box_width, box_height

    left = max(0, int(math.floor(drawn_x)))
    top = max(0, int(math.floor(drawn_y)))
    right = min(canvas.width, int(math.ceil(drawn_x + drawn_width)))
    bottom = min(canvas.height, int(math.ceil(drawn_y + drawn_height)))
    if right <= left or bottom <= top:
        return
    fraction_x0 = max(0.0, min(1.0, (left - drawn_x) / drawn_width))
    fraction_y0 = max(0.0, min(1.0, (top - drawn_y) / drawn_height))
    fraction_x1 = max(0.0, min(1.0, (right - drawn_x) / drawn_width))
    fraction_y1 = max(0.0, min(1.0, (bottom - drawn_y) / drawn_height))
    source_rect = (
        crop_x0 + (crop_x1 - crop_x0) * fraction_x0,
        crop_y0 + (crop_y1 - crop_y0) * fraction_y0,
        crop_x0 + (crop_x1 - crop_x0) * fraction_x1,
        crop_y0 + (crop_y1 - crop_y0) * fraction_y1,
    )
    crop = source.crop(tuple(int(round(value)) for value in source_rect))
    if crop.width <= 0 or crop.height <= 0:
        return
    crop = crop.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    opacity = float(element["opacity"])
    if opacity < 1.0:
        alpha = crop.getchannel("A").point(lambda value: int(round(value * opacity)))
        crop.putalpha(alpha)
    canvas.alpha_composite(crop, (left, top))


def render_editor_graphic(
    graphic: Mapping[str, Any],
    *,
    width: int,
    height: int,
    resolve_image: Any = None,
) -> bytes:
    """Render one graphic into transparent RGBA PNG bytes.

    Design coordinates use a uniform contain scale and are centered in the
    requested output, leaving transparent letterbox space when aspect ratios
    differ. This preserves proportions and typography across project canvases.
    """
    output_width = _positive_integer(
        width, "render width", maximum=GRAPHIC_MAX_OUTPUT_DIMENSION
    )
    output_height = _positive_integer(
        height, "render height", maximum=GRAPHIC_MAX_OUTPUT_DIMENSION
    )
    if output_width * output_height > GRAPHIC_MAX_OUTPUT_AREA:
        raise EditorGraphicError("Graphic render dimensions are too large")
    normalized = normalize_editor_graphic(graphic)
    if any(element["type"] == "image" for element in normalized["elements"]) and not callable(resolve_image):
        raise EditorGraphicError("Graphic image layers require an authorized image resolver")
    canvas = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
    scale = min(
        output_width / normalized["width"],
        output_height / normalized["height"],
    )
    scale_x = scale_y = scale
    offset_x = (output_width - normalized["width"] * scale) / 2
    offset_y = (output_height - normalized["height"] * scale) / 2
    for element in normalized["elements"]:
        element_layer = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
        if element["type"] == "rect":
            x0 = float(element["x"]) * scale_x + offset_x
            y0 = float(element["y"]) * scale_y + offset_y
            x1 = x0 + float(element["width"]) * scale_x
            y1 = y0 + float(element["height"]) * scale_y
            radius = float(element["radius"]) * (scale_x + scale_y) / 2
            draw = ImageDraw.Draw(element_layer)
            rectangle = (x0, y0, x1, y1)
            fill = _color_with_opacity(str(element["fill"]), float(element["opacity"]))
            if radius > 0:
                draw.rounded_rectangle(rectangle, radius=radius, fill=fill)
            else:
                draw.rectangle(rectangle, fill=fill)
            border_color = element.get("border_color")
            border_width = int(round(float(element["border_width"]) * (scale_x + scale_y) / 2))
            if border_color and border_width > 0:
                outline = _color_with_opacity(str(border_color), float(element["opacity"]))
                if radius > 0:
                    draw.rounded_rectangle(
                        rectangle,
                        radius=radius,
                        outline=outline,
                        width=max(1, border_width),
                    )
                else:
                    draw.rectangle(
                        rectangle,
                        outline=outline,
                        width=max(1, border_width),
                    )
        elif element["type"] == "text":
            _draw_text_element(
                element_layer,
                element,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        else:
            _draw_image_element(
                element_layer,
                element,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=offset_x,
                offset_y=offset_y,
                resolve_image=resolve_image,
            )
        canvas.alpha_composite(element_layer)
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False)
    return output.getvalue()


__all__ = [
    "EditorGraphicError",
    "normalize_editor_graphic",
    "render_editor_graphic",
]
