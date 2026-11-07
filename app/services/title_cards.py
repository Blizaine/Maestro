"""Deterministic, CPU-only title-card validation and rendering."""
from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping
import unicodedata

from PIL import Image, ImageDraw, ImageFont, ImageOps


_ALLOWED_FIELDS = {
    "text", "subtitle", "role", "style", "background_color", "text_color",
    "duration_seconds",
}
_ROLES = {"opening", "intertitle", "end"}
_STYLES = {"minimal", "classic", "bold"}
_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{6}\Z")
_MIN_DURATION_SECONDS = 0.5
_MAX_DURATION_SECONDS = 15.0
_MIN_DIMENSION = 64
_MAX_DIMENSION = 4096
_MAX_PIXEL_COUNT = 16_777_216
_MAX_FPS = 120.0
_FFMPEG_TIMEOUT_SECONDS = 180


class TitleCardRenderError(RuntimeError):
    """Raised when a title card cannot be rendered without losing content."""


def _text(value: Any, field: str, maximum: int, *, required: bool) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    if len(value) > maximum:
        raise ValueError(f"{field} must be at most {maximum} characters.")
    if required and not value.strip():
        raise ValueError(f"{field} must contain visible text.")
    if any(
        unicodedata.category(character) == "Cc"
        and character not in "\n\r\t"
        for character in value
    ):
        raise ValueError(f"{field} contains an unsupported control character.")
    return value


def _color(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _COLOR_RE.fullmatch(value):
        raise ValueError(f"{field} must be a #RRGGBB color.")
    return value.upper()


def normalize_title_card(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a bounded title-card description without rewriting its copy."""
    if not isinstance(value, Mapping):
        raise ValueError("Title card must be an object.")
    unknown = set(value) - _ALLOWED_FIELDS
    if unknown:
        field = sorted(str(item) for item in unknown)[0]
        raise ValueError(f"Unsupported title-card field: {field}.")
    if "text" not in value:
        raise ValueError("text is required.")

    title = _text(value["text"], "text", 500, required=True)
    subtitle = _text(value.get("subtitle", ""), "subtitle", 1000, required=False)
    role = value.get("role", "intertitle")
    if not isinstance(role, str) or role not in _ROLES:
        raise ValueError("role must be opening, intertitle, or end.")
    style = value.get("style", "classic")
    if not isinstance(style, str) or style not in _STYLES:
        raise ValueError("style must be minimal, classic, or bold.")

    duration = value.get("duration_seconds", 3.0)
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or not _MIN_DURATION_SECONDS <= float(duration) <= _MAX_DURATION_SECONDS
    ):
        raise ValueError("duration_seconds must be finite and between 0.5 and 15 seconds.")

    return {
        "text": title,
        "subtitle": subtitle,
        "role": role,
        "style": style,
        "background_color": _color(value.get("background_color", "#000000"), "background_color"),
        "text_color": _color(value.get("text_color", "#FFFFFF"), "text_color"),
        "duration_seconds": float(duration),
    }


def _rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))


def _font_candidates(style: str) -> list[str]:
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if style == "classic":
        names = ("georgia.ttf", "Georgia.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "Times New Roman.ttf")
        system_paths = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        )
    elif style == "bold":
        names = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
        system_paths = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        )
    else:
        names = ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
        system_paths = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        )

    paths = [str(windows_fonts / name) for name in names[:2]]
    paths.extend(system_paths)
    paths.extend(str(windows_fonts / name) for name in names[2:])
    paths.extend((
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ))
    result: list[str] = []
    for path in paths:
        if path not in result:
            result.append(path)
    return result


@lru_cache(maxsize=96)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def _font_supports(font: ImageFont.FreeTypeFont, text: str) -> bool:
    try:
        missing = font.getmask("\ufffe")
        for character in text:
            if character.isspace() or unicodedata.category(character) in {"Cf", "Mn", "Mc", "Me"}:
                continue
            if bytes(font.getmask(character)) == bytes(missing):
                return False
        return True
    except Exception:
        return False


def _font_for_text(style: str, text: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _font_candidates(style):
        try:
            font = _load_font(path, size)
        except OSError:
            continue
        if _font_supports(font, text):
            return font
    raise TitleCardRenderError(
        "No installed Segoe, DejaVu, or compatible real font can render every title character. "
        "Install a font with the required language glyphs; no text was omitted."
    )


def _graphemes(text: str) -> list[str]:
    """Keep common combining-mark and emoji join sequences intact on wrap."""
    result: list[str] = []
    current = ""
    for character in text:
        category = unicodedata.category(character)
        is_modifier = 0x1F3FB <= ord(character) <= 0x1F3FF
        if current and (
            category in {"Mn", "Mc", "Me"}
            or character == "\u200d"
            or current.endswith("\u200d")
            or is_modifier
        ):
            current += character
        else:
            if current:
                result.append(current)
            current = character
    if current:
        result.append(current)
    return result


def _wrap_paragraph(text: str, font: ImageFont.FreeTypeFont, maximum_width: int) -> list[str]:
    """Prefer word boundaries, then wrap long words without discarding text."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[str] = []
    line = ""

    def split_long_token(token: str) -> str:
        chunk = ""
        for unit in _graphemes(token):
            candidate = chunk + unit
            if draw.textlength(candidate, font=font) <= maximum_width:
                chunk = candidate
                continue
            if not chunk:
                raise TitleCardRenderError("A title character is wider than the safe card area.")
            lines.append(chunk)
            chunk = unit
            if draw.textlength(chunk, font=font) > maximum_width:
                raise TitleCardRenderError("A title character is wider than the safe card area.")
        return chunk

    for token in re.findall(r"\s+|\S+", text):
        if draw.textlength(line + token, font=font) <= maximum_width:
            line += token
            continue
        if line:
            lines.append(line)
            line = ""
        if draw.textlength(token, font=font) <= maximum_width:
            line = token
        else:
            line = split_long_token(token)
    lines.append(line)
    return lines


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, maximum_width: int) -> list[str]:
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = normalized_newlines.expandtabs(4).split("\n")
    lines: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        lines.extend(_wrap_paragraph(paragraph, font, maximum_width))
        if index < len(paragraphs) - 1:
            # Blank entries retain explicit newlines in the rendered layout.
            continue
    return lines


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return max(1, ascent + descent)


def _render_text_image(
    card: dict[str, Any],
    width: int,
    height: int,
    image_path: str | None,
) -> tuple[Image.Image, list[str]]:
    warnings: list[str] = []
    background_color = _rgb(card["background_color"])
    if image_path is None:
        image = Image.new("RGB", (width, height), background_color)
    else:
        source = Path(image_path)
        if not source.is_file():
            raise TitleCardRenderError("The title-card artwork file was not found.")
        try:
            with Image.open(source) as opened:
                if opened.width <= 0 or opened.height <= 0 or opened.width * opened.height > _MAX_PIXEL_COUNT * 4:
                    raise TitleCardRenderError("The title-card artwork dimensions exceed the supported limit.")
                source_image = opened.convert("RGBA")
        except (OSError, Image.DecompressionBombError) as error:
            raise TitleCardRenderError(f"The title-card artwork could not be decoded: {error}") from error

        source_ratio = source_image.width / source_image.height
        target_ratio = width / height
        if abs(source_ratio - target_ratio) > 0.005:
            warnings.append("Artwork was center-cropped to fill the title-card canvas.")
        backdrop = Image.new("RGBA", (width, height), (*background_color, 255))
        fitted = ImageOps.fit(
            source_image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        backdrop.alpha_composite(fitted)
        image = backdrop.convert("RGB")

    text = card["text"].expandtabs(4)
    subtitle = card["subtitle"].expandtabs(4)

    minimum_size = max(12, int(min(width, height) * 0.025))
    style_factor = {"minimal": 0.085, "classic": 0.105, "bold": 0.12}[card["style"]]
    maximum_size = max(minimum_size, int(min(width, height) * style_factor))
    safe_width = int(width * 0.84)
    safe_height = int(height * 0.78)
    selected: tuple[ImageFont.FreeTypeFont, list[str], ImageFont.FreeTypeFont | None, list[str], int] | None = None

    for title_size in range(maximum_size, minimum_size - 1, -2):
        title_font = _font_for_text(card["style"], text, title_size)
        title_lines = _wrap_text(text, title_font, safe_width)
        title_height = len(title_lines) * _line_height(title_font)
        title_height += max(0, len(title_lines) - 1) * max(2, title_size // 5)

        subtitle_font: ImageFont.FreeTypeFont | None = None
        subtitle_lines: list[str] = []
        subtitle_height = 0
        if subtitle.strip():
            subtitle_size = max(10, int(title_size * 0.44))
            subtitle_font = _font_for_text(card["style"], subtitle, subtitle_size)
            subtitle_lines = _wrap_text(subtitle, subtitle_font, safe_width)
            subtitle_height = len(subtitle_lines) * _line_height(subtitle_font)
            subtitle_height += max(0, len(subtitle_lines) - 1) * max(2, subtitle_size // 5)

        gap = max(12, title_size // 2) if subtitle_lines else 0
        total_height = title_height + gap + subtitle_height
        if total_height <= safe_height:
            selected = (title_font, title_lines, subtitle_font, subtitle_lines, gap)
            break

    if selected is None:
        raise TitleCardRenderError(
            "The complete title and subtitle do not fit the safe card area at a readable font size."
        )

    title_font, title_lines, subtitle_font, subtitle_lines, gap = selected
    draw = ImageDraw.Draw(image)
    fill = _rgb(card["text_color"])
    luminance = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
    stroke_fill = (0, 0, 0) if luminance > 140 else (255, 255, 255)
    stroke_width = 0 if card["style"] == "minimal" else max(1, title_font.size // (30 if card["style"] == "bold" else 48))

    title_height = len(title_lines) * _line_height(title_font)
    title_height += max(0, len(title_lines) - 1) * max(2, title_font.size // 5)
    subtitle_height = 0
    if subtitle_font is not None:
        subtitle_height = len(subtitle_lines) * _line_height(subtitle_font)
        subtitle_height += max(0, len(subtitle_lines) - 1) * max(2, subtitle_font.size // 5)
    current_y = max(0, int((height - title_height - gap - subtitle_height) / 2))

    def draw_lines(lines: list[str], font: ImageFont.FreeTypeFont, y: int, line_gap: int) -> int:
        line_height = _line_height(font)
        for line in lines:
            line_width = float(draw.textlength(line, font=font))
            x = (width - line_width) / 2
            baseline = y + font.getmetrics()[0]
            draw.text(
                (x, baseline),
                line,
                fill=fill,
                font=font,
                anchor="ls",
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            y += line_height + line_gap
        return y

    current_y = draw_lines(title_lines, title_font, current_y, max(2, title_font.size // 5))
    current_y += gap
    if subtitle_font is not None:
        draw_lines(subtitle_lines, subtitle_font, current_y, max(2, subtitle_font.size // 5))
    return image, warnings


def _resolve_ffmpeg(ffmpeg_path: str | os.PathLike[str] | None) -> str:
    if ffmpeg_path is not None:
        requested = os.fspath(ffmpeg_path)
        if not requested:
            raise TitleCardRenderError("ffmpeg_path must not be empty.")
        resolved = requested if os.path.isfile(requested) else shutil.which(requested)
        if resolved:
            return resolved
        raise TitleCardRenderError("The requested FFmpeg executable was not found.")

    # Mirror Maestro's shared resolver without importing video_decode, which
    # imports Torch and NumPy in a path that should remain lightweight.
    configured = os.environ.get("FFMPEG_BINARY", "")
    if configured and os.path.isfile(configured):
        return configured
    app_root = Path(__file__).resolve().parents[1]
    bundled = app_root / "ffmpeg_bins" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    resolved = str(bundled) if bundled.is_file() else shutil.which("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not resolved:
        resolved = shutil.which("ffmpeg")
    if not resolved:
        raise TitleCardRenderError("FFmpeg was not found; supply ffmpeg_path or install the app's media tool.")
    return str(resolved)


def _resolve_ffprobe(ffmpeg_path: str) -> str | None:
    ffmpeg_file = Path(ffmpeg_path)
    executable = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    sibling = ffmpeg_file.with_name(executable)
    if sibling.is_file():
        return str(sibling)
    configured = os.environ.get("FFPROBE_BINARY")
    if configured and os.path.isfile(configured):
        return configured
    return shutil.which(executable) or shutil.which("ffprobe")


def _verify_video(
    ffprobe_path: str | None,
    output_path: str,
    *,
    width: int,
    height: int,
    frame_count: int,
) -> str | None:
    if not ffprobe_path:
        return "Frame count and encoded dimensions were not independently checked because ffprobe was unavailable."
    command = [
        ffprobe_path,
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,nb_frames",
        "-of", "json",
        output_path,
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return "Frame count and encoded dimensions were not independently checked because ffprobe could not run."
    if result.returncode != 0:
        return "Frame count and encoded dimensions were not independently checked because ffprobe rejected the output."
    try:
        streams = json.loads(result.stdout or "{}").get("streams") or []
        video = streams[0]
        observed_width = int(video.get("width") or 0)
        observed_height = int(video.get("height") or 0)
        observed_frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    except (ValueError, TypeError, IndexError, json.JSONDecodeError):
        return "Frame count and encoded dimensions could not be read from ffprobe output."
    if (observed_width, observed_height, observed_frames) != (width, height, frame_count):
        raise TitleCardRenderError(
            "FFmpeg produced a title card with unexpected dimensions or frame count "
            f"({observed_width}x{observed_height}, {observed_frames} frames)."
        )
    return None


def render_title_card(
    card: Mapping[str, Any],
    output_path: str | os.PathLike[str],
    *,
    width: int,
    height: int,
    fps: int | float = 24,
    image_path: str | os.PathLike[str] | None = None,
    ffmpeg_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Render one still title card as a silent H.264/AAC MP4 using the CPU.

    The output is assembled in temporary files beside the destination and is
    atomically replaced only after FFmpeg completes and any available ffprobe
    check passes. The returned duration reflects the encoded frame lattice.
    """
    normalized = normalize_title_card(card)
    if isinstance(width, bool) or not isinstance(width, int) or not _MIN_DIMENSION <= width <= _MAX_DIMENSION or width % 2:
        raise ValueError(f"width must be an even integer from {_MIN_DIMENSION} to {_MAX_DIMENSION}.")
    if isinstance(height, bool) or not isinstance(height, int) or not _MIN_DIMENSION <= height <= _MAX_DIMENSION or height % 2:
        raise ValueError(f"height must be an even integer from {_MIN_DIMENSION} to {_MAX_DIMENSION}.")
    if width * height > _MAX_PIXEL_COUNT:
        raise ValueError("Title-card dimensions exceed the supported pixel limit.")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or not 1 <= float(fps) <= _MAX_FPS
    ):
        raise ValueError(f"fps must be finite and between 1 and {_MAX_FPS:g}.")

    destination = Path(output_path).expanduser().absolute()
    if destination.suffix.casefold() != ".mp4":
        raise ValueError("output_path must use the .mp4 extension.")
    if image_path is not None:
        artwork = Path(image_path).expanduser().absolute()
        if artwork == destination:
            raise ValueError("The output video cannot overwrite its artwork source.")
        image_source = str(artwork)
    else:
        image_source = None
    destination.parent.mkdir(parents=True, exist_ok=True)

    duration = normalized["duration_seconds"]
    frame_count = max(1, int(math.floor(duration * float(fps) + 0.5)))
    encoded_duration = frame_count / float(fps)
    ffmpeg = _resolve_ffmpeg(ffmpeg_path)
    png_fd, png_path = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".part.png", dir=destination.parent)
    video_fd, temporary_video = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".part.mp4", dir=destination.parent)
    os.close(png_fd)
    os.close(video_fd)
    warnings: list[str] = []
    try:
        rendered, warnings = _render_text_image(normalized, width, height, image_source)
        rendered.save(png_path, format="PNG")
        command = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-loop", "1", "-framerate", format(float(fps), ".12g"), "-i", png_path,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-map", "0:v:0", "-map", "1:a:0",
            "-frames:v", str(frame_count), "-t", format(encoded_duration, ".9f"),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "1",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "2",
            "-shortest", "-movflags", "+faststart", temporary_video,
        ]
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_FFMPEG_TIMEOUT_SECONDS,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as error:
            raise TitleCardRenderError("FFmpeg timed out while rendering the title card.") from error
        except OSError as error:
            raise TitleCardRenderError(f"FFmpeg could not be started: {error}") from error
        if result.returncode != 0 or not os.path.isfile(temporary_video) or os.path.getsize(temporary_video) == 0:
            detail = (result.stderr or "FFmpeg returned an error.").strip().splitlines()
            raise TitleCardRenderError(f"Could not render title card: {(detail[-1] if detail else 'FFmpeg failed')[:400]}")

        warning = _verify_video(
            _resolve_ffprobe(ffmpeg),
            temporary_video,
            width=width,
            height=height,
            frame_count=frame_count,
        )
        if warning:
            warnings.append(warning)
        os.replace(temporary_video, destination)
    finally:
        for temporary in (png_path, temporary_video):
            try:
                os.unlink(temporary)
            except OSError:
                pass

    return {
        "path": str(destination),
        "duration": encoded_duration,
        "width": width,
        "height": height,
        "fps": float(fps),
        "frame_count": frame_count,
        "card": normalized,
        "warnings": warnings,
    }


__all__ = ["TitleCardRenderError", "normalize_title_card", "render_title_card"]
