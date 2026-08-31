"""Prompt-format helpers for MiniMax-Music3.

Music3 receives the music description and lyrics as separate conditioning
inputs.  Bracketed text in the lyrics is therefore deliberately restricted to
known section markers; production notes belong in the structured caption.
"""

from __future__ import annotations

import re


_CANONICAL_TAGS = {
    "intro": "Intro",
    "verse": "Verse",
    "pre-chorus": "Pre-Chorus",
    "pre chorus": "Pre-Chorus",
    "chorus": "Chorus",
    "post-chorus": "Post-Chorus",
    "post chorus": "Post-Chorus",
    "bridge": "Bridge",
    "instrumental": "Instrumental",
    "solo": "Solo",
    "guitar solo": "Guitar Solo",
    "outro": "Outro",
}
_DISPLAY_TAGS = (
    "[Intro], [Verse], [Pre-Chorus], [Chorus], [Post-Chorus], "
    "[Bridge], [Instrumental], [Solo], [Guitar Solo], or [Outro]"
)
_TAG_WITH_TAIL_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(.*?)\s*$")
_TAG_ONLY_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*$")
_DESCRIPTOR_SPLIT_RE = re.compile(r"\s*(?:[-\u2013\u2014:|,;])\s*", re.UNICODE)
_NUMBERED_SECTION_RE = re.compile(
    r"^(intro|verse|pre[- ]chorus|chorus|post[- ]chorus|bridge|"
    r"instrumental|solo|guitar solo|outro)\s+(?:\d+|one|two|three|"
    r"four|five|six)\s*$",
    re.IGNORECASE,
)
_PAREN_INSTRUMENTAL_RE = re.compile(
    r"^\s*\(\s*instrumental(?:\s+(?:break|interlude|solo))?\s*\)\s*$",
    re.IGNORECASE,
)
_PAREN_ONLY_RE = re.compile(r"^\s*\(([^()]+)\)\s*$")
_PRODUCTION_CUE_RE = re.compile(
    r"\b(?:whisper(?:ed|ing)?|spoken|softly|loudly|breathy|raspy|falsetto|"
    r"belt(?:ed|ing)?|guitar|piano|drums?|bass|synth(?:esizer)?|strings?|"
    r"brass|choir|vocals?|harmon(?:y|ies)|solo|instrumental|fade|build|drop|"
    r"breakdown|tempo|bpm|key|reverb|delay|distortion|enters?|exits?)\b",
    re.IGNORECASE,
)


def _tag_key(value: str) -> str:
    text = str(value or "").strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.casefold()


def _canonical_bare_tag(value: str) -> str | None:
    key = _tag_key(value)
    direct = _CANONICAL_TAGS.get(key)
    if direct:
        return direct
    numbered = _NUMBERED_SECTION_RE.fullmatch(key)
    if numbered:
        return _CANONICAL_TAGS.get(_tag_key(numbered.group(1)))
    return None


def _canonical_tag_and_direction(value: str) -> tuple[str | None, str]:
    """Return a canonical tag and any production direction embedded in it."""

    raw = str(value or "").strip()
    direct = _canonical_bare_tag(raw)
    if direct:
        return direct, ""

    parts = _DESCRIPTOR_SPLIT_RE.split(raw, maxsplit=1)
    if len(parts) == 2:
        canonical = _canonical_bare_tag(parts[0])
        if canonical:
            return canonical, parts[1].strip()

    # Recover common LLM output such as ``[Intro heartbeat pulse]`` while
    # preferring the longest valid marker (``Guitar Solo`` before ``Solo``).
    normalized = _tag_key(raw)
    for key in sorted(_CANONICAL_TAGS, key=len, reverse=True):
        if normalized.startswith(f"{key} "):
            direction = raw[len(key) :].strip(" -\u2013\u2014:|,;")
            direction = re.sub(
                r"^(?:\d+|one|two|three|four|five|six)\s*[,;:\-]?\s*",
                "",
                direction,
                flags=re.IGNORECASE,
            )
            return _CANONICAL_TAGS[key], direction
    return None, ""


def _parenthetical_production_cue(value: str) -> str | None:
    """Return an obvious stage direction, but preserve sung parentheticals."""

    match = _PAREN_ONLY_RE.fullmatch(str(value or ""))
    if not match:
        return None
    cue = match.group(1).strip()
    return cue if _PRODUCTION_CUE_RE.search(cue) else None


def normalize_generated_music3_song(
    style: str,
    lyrics: str,
) -> tuple[str, str]:
    """Repair predictable LLM formatting without discarding lyric words.

    Descriptions accidentally embedded in a section marker are moved into the
    Arrangement caption.  A literal ``(instrumental)`` stage direction becomes
    the actual ``[Instrumental]`` control tag so it cannot be sung aloud.
    """

    output: list[str] = []
    local_directions: list[tuple[str, str]] = []
    current_section = "Arrangement"
    for source_line in str(lyrics or "").replace("\r\n", "\n").split("\n"):
        stripped = source_line.strip()
        if _PAREN_INSTRUMENTAL_RE.fullmatch(stripped):
            if not output or output[-1].strip().casefold() != "[instrumental]":
                output.append("[Instrumental]")
            current_section = "Instrumental"
            continue

        production_cue = _parenthetical_production_cue(stripped)
        if production_cue:
            local_directions.append((current_section, production_cue))
            continue

        match = _TAG_WITH_TAIL_RE.fullmatch(source_line)
        if not match:
            output.append(source_line.rstrip())
            continue

        canonical, direction = _canonical_tag_and_direction(match.group(1))
        if canonical is None:
            output.append(source_line.rstrip())
            continue

        output.append(f"[{canonical}]")
        current_section = canonical
        if direction:
            local_directions.append((canonical, direction))

        trailing = match.group(2).strip()
        if trailing:
            # Text after a valid bare marker is almost always an intended lyric
            # line.  Preserve it instead of silently dropping it as the raw
            # checkpoint normalizer historically did.
            output.append(trailing)

    normalized_lyrics = "\n".join(output).strip()
    normalized_lyrics = re.sub(r"\n{3,}", "\n\n", normalized_lyrics)

    normalized_style = str(style or "").strip()
    if local_directions:
        directions = "; ".join(
            f"{tag}: {direction}" for tag, direction in local_directions
        )
        note = (
            "Section-local production directions moved out of the lyric tags: "
            f"{directions}."
        )
        if re.search(r"^###\s+Arrangement\s*$", normalized_style, re.MULTILINE):
            normalized_style = f"{normalized_style.rstrip()}\n{note}"
        elif normalized_style:
            normalized_style = f"{normalized_style}\n\n### Arrangement\n{note}"
        else:
            normalized_style = f"### Arrangement\n{note}"

    return normalized_style, normalized_lyrics


def validate_music3_lyrics(lyrics: str) -> str | None:
    """Return an actionable error for lyrics Music3 may narrate or truncate."""

    for line_number, source_line in enumerate(
        str(lyrics or "").replace("\r\n", "\n").split("\n"),
        start=1,
    ):
        stripped = source_line.strip()
        if not stripped:
            continue
        if _PAREN_INSTRUMENTAL_RE.fullmatch(stripped):
            return (
                f"MiniMax-Music3 lyric line {line_number} uses {stripped}. "
                "Use [Instrumental] on its own line instead; parenthetical "
                "stage directions may be sung aloud."
            )
        production_cue = _parenthetical_production_cue(stripped)
        if production_cue:
            return (
                f"MiniMax-Music3 lyric line {line_number} contains the stage "
                f"direction ({production_cue}). Move it to the Music Caption; "
                "parenthetical production cues may be sung aloud."
            )
        if "[" not in stripped and "]" not in stripped:
            continue

        match = _TAG_ONLY_RE.fullmatch(stripped)
        if not match:
            return (
                f"MiniMax-Music3 lyric line {line_number} must put its section "
                "tag alone on the line. Move lyric words and production notes "
                "to their own fields."
            )

        raw_tag = match.group(1).strip()
        if _CANONICAL_TAGS.get(_tag_key(raw_tag)):
            continue

        canonical, direction = _canonical_tag_and_direction(raw_tag)
        if canonical:
            suffix = (
                f" and move '{direction}' to the Music Caption"
                if direction
                else ""
            )
            return (
                f"MiniMax-Music3 section tags must be bare and canonical. "
                f"Change [{raw_tag}] to [{canonical}]{suffix}."
            )
        return (
            f"MiniMax-Music3 does not recognize the section tag [{raw_tag}]. "
            f"Use {_DISPLAY_TAGS}."
        )
    return None
