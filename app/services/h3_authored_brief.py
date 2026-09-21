"""Read explicit structure in imported video briefs without rewriting their prose.

Timestamped shot blocks are authored story units, not a bag of sentences. Keep
their camera, choreography and timing together; production notes are context.
This module uses only explicit labels, never guesses characters from capitals.
"""

from __future__ import annotations

from functools import lru_cache
import re

from models.minimax_h3.speakers import is_h3_production_label


_CLOCK = r"(?:\d{1,2}:)?\d{1,3}(?:\.\d+)?"
_TIMED_HEADING = re.compile(
    rf"(?:【|\[|\()\s*(?P<start>{_CLOCK})\s*(?:s|sec(?:onds?)?)?\s*"
    rf"[-–—]\s*(?P<end>{_CLOCK})\s*(?:s|sec(?:onds?)?)?\s*"
    r"(?:[|｜:：]\s*(?P<title>[^】\]\)\r\n]{1,200}))?\s*(?:】|\]|\))",
    re.IGNORECASE,
)
_NOTES_LABEL = (
    r"(?:(?:vfx|cinematography|negative(?:\s+prompt)?|constraints|requirements)"
    r"(?:\s+(?:requirements|direction|notes|design))?|"
    r"(?:camera|visuals?|sound|audio|style|lighting)\s+(?:requirements|direction|notes|design))"
)
_NOTES_HEADING = re.compile(
    rf"(?:【|\[)\s*{_NOTES_LABEL}\s*(?:】|\])|"
    rf"(?:^|\n)\s*(?:#{{1,4}}\s*)?{_NOTES_LABEL}\s*:",
    re.IGNORECASE,
)
_PROFILE = re.compile(
    r"\b(?P<name>(?:Character|Subject|Actor|Role)\s+(?:[A-Z](?![a-z])|\d+))"
    r"\s*(?:\((?P<details>[^)\r\n]{1,160})\))?\s*:\s*",
)
_SECTION_HEADING = re.compile(r"【([^】\r\n]{1,200})】|\[([^\]\r\n]{1,200})\]")
_LINE_HEADING = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]+(?P<hash>[^\r\n]{1,200})|"
    r"\*\*(?P<bold>[^*\r\n]{1,200})\*\*[ \t]*:?|"
    r"__(?P<underline>[^_\r\n]{1,200})__[ \t]*:?|"
    r"(?P<plain>[\w][\w /&-]{0,100}):)[ \t]*\r?$"
)
_LINE_TIMED_HEADING = re.compile(
    rf"(?m)^[ \t]*(?:[-*][ \t]+|\d+[.)][ \t]+)?"
    rf"(?P<start>{_CLOCK})\s*(?:s|sec(?:onds?)?)?\s*[-–—]\s*"
    rf"(?P<end>{_CLOCK})\s*(?:s|sec(?:onds?)?)"
    # Accept "0-3s: [ARRIVAL]" without a space before the colon. The title
    # labels the block; it must not become an action or a character name.
    r"(?:[ \t]*[:：|｜–—-][ \t]*|[ \t]+|(?=\r?$))"
    r"(?:[\[【](?P<title>[^\]】\r\n]{1,200})[\]】][ \t]*)?",
    re.IGNORECASE,
)
_NUMBERED_SHOT_HEADING = re.compile(
    rf"\bShot\s+\d+\s*[:.\-–—]?\s*[【\[(]\s*t?\s*"
    rf"(?P<start>{_CLOCK})\s*(?:s|sec(?:onds?)?)?\s*[-–—]\s*t?\s*"
    rf"(?P<end>{_CLOCK})\s*(?:s|sec(?:onds?)?)?\s*[】\])]",
    re.IGNORECASE,
)
_LINE_NUMBERED_SHOT_HEADING = re.compile(
    rf"(?m)^[ \t]*(?:#{{1,6}}[ \t]+|[-*][ \t]+)?(?:\*\*|__)?"
    rf"Shot[ \t]+\d+[ \t]*(?:[:.\-–—|][ \t]*)?"
    rf"(?P<start>{_CLOCK})[ \t]*(?:s|sec(?:onds?)?)?[ \t]*"
    rf"(?:[-–—]|\bto\b)[ \t]*(?P<end>{_CLOCK})[ \t]*(?:s|sec(?:onds?)?)?"
    r"(?:[ \t]*[|｜:：–—-][ \t]*(?P<title>[^*\r\n]{1,200}?))?"
    r"[ \t]*(?:\*\*|__)?[ \t]*\r?$",
    re.IGNORECASE,
)


def _timed_headings(source: str) -> list[re.Match]:
    # A numbered shot list is the detailed clock. Chapter/skill ranges and a
    # prose synopsis may repeat it; they must not invalidate or duplicate it.
    # Pasted storyboards also use standalone "SHOT 1 — 0:00–0:02" headings.
    # Recognize that explicit structure before sentence/cast extraction; its
    # clock must not become an action, and adjectives in its body are not names.
    line_shots = list(_LINE_NUMBERED_SHOT_HEADING.finditer(source))
    shots = sorted([*_NUMBERED_SHOT_HEADING.finditer(source), *line_shots],
                   key=lambda match: match.start())
    return shots if len(shots) >= 2 else sorted(
        [*_TIMED_HEADING.finditer(source), *_LINE_TIMED_HEADING.finditer(source), *line_shots],
        key=lambda match: match.start(),
    )


@lru_cache(maxsize=8)
def _source_headings(source: str) -> tuple[tuple[int, int, str, int], ...]:
    """Read heading boundaries in common pasted-brief formats, without flattening."""
    headings = [(m.start(), m.end(), m.group(1) or m.group(2), 1)
                for m in _SECTION_HEADING.finditer(source)]
    headings += [(m.start(), m.end(), next(v for v in m.groupdict().values() if v),
                  len(m.group(0).lstrip()) - len(m.group(0).lstrip().lstrip("#")) or 1)
                 for m in _LINE_HEADING.finditer(source)
                 if not any(start <= m.start() < end for start, end, _label, _depth in headings)]
    headings += [(m.start(), m.end(), "timed shot", 0)
                 for m in _timed_headings(source)]
    return tuple(sorted(headings))


def _section_kind(label: str) -> str:
    words = {
        word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
        for word in re.sub(r"[^a-z0-9]+", " ", label.casefold()).split()
    }
    if words & {"requirement", "constraint", "rule", "negative", "specification",
                "system", "density", "design", "note"}:
        return "notes"
    if words & {"cast", "character", "profile", "characterization", "subject", "role", "actor"}:
        return "profiles"
    if words & {"dialogue", "script", "screenplay", "story", "action",
                "scene", "shot", "beat", "timeline", "lyric", "voiceover"}:
        return "story"
    if is_h3_production_label(label) or any(
        word in {"render", "effect", "lighting", "wardrobe"} for word in words
    ):
        return "notes"
    return ""


@lru_cache(maxsize=8)
def _descriptive_sections(source: str) -> tuple[tuple[int, int, str], ...]:
    headings = _source_headings(source)
    sections = []
    for index, (start, _end, label, depth) in enumerate(headings):
        kind = _section_kind(label)
        if kind not in {"profiles", "notes"}:
            continue
        stop = len(source)
        for next_start, _next_end, next_label, next_depth in headings[index + 1:]:
            next_kind = _section_kind(next_label)
            if next_kind or next_depth == 0 or next_depth <= depth:
                stop = next_start
                break
        sections.append((start, stop, kind))
    return tuple(sections)


@lru_cache(maxsize=8)
def production_note_spans(source: str) -> tuple[tuple[int, int], ...]:
    """Identify explicitly headed production sections, ending at the next heading.

    Labels within an effects/lighting/cinematography section explain that
    section's settings; their subheadings do not introduce speaking characters.
    A timed shot or a dialogue section ends this context immediately.
    """
    return tuple((start, end) for start, end, _kind in _descriptive_sections(source))


def character_profile_spans(source: str) -> tuple[tuple[int, int], ...]:
    """Distinguish cast descriptions from other headed production notes."""
    return tuple(
        (start, end) for start, end, kind in _descriptive_sections(source)
        if kind == "profiles"
    )


def negative_prompt_spans(source: str) -> tuple[tuple[int, int], ...]:
    """Locate explicitly negative lists without turning their terms into asks.

    The original text stays available to the writer as restrictions. These
    spans are for positive-intent detection only; 'excessive dialogue' in a
    negative list neither requests conversation nor bans a supplied line.
    """
    headings = _source_headings(source)
    spans = []
    for index, (start, _end, label, depth) in enumerate(headings):
        if not re.search(r"\b(?:negatives?|prohibitions?|exclusions?)\b", label, re.I):
            continue
        end = next((a for a, _b, _label, level in headings[index + 1:] if level <= depth), len(source))
        spans.append((start, end))
    # Inline lists also occur without a section header. Bound the list at a
    # sentence/newline, next heading or explicit positive contrast, so later
    # requested speech still counts.
    for match in re.finditer(
        r"(?:^|(?<=[.!?;\n\r】\]]))\s*(?:(?:strictly\s+)?"
        r"(?:prohibited|forbidden|disallowed)\s*:|negative\s+prompt\s*:|"
        r"(?:avoid|exclude|no|without)\s+|(?:do\s+not|don't)\s+|"
        r"never\s+(?!ending\b|stopping\b|pausing\b))", source, re.I,
    ):
        tail = source[match.end():]
        stop = re.search(
            r"[.!?](?=\s|$)|[;\r\n]|\b(?:but|however|instead|then|until|while|as|before|after|during)\b|"
            r",\s*and\s+(?=[^,;.!?]{0,50}\b(?:begins?|starts?|gives?|delivers?|speaks?|asks?)\b)",
            tail, re.I,
        )
        end = match.end() + stop.start() if stop else len(source)
        end = min(end, next((a for a, _b, _label, _level in headings if a >= match.end()), end))
        spans.append((match.start(), end))
    return tuple(sorted(spans))


def positive_instruction_text(source: str) -> str:
    """Mask negative instructions for intent checks, preserving source offsets."""
    for start, end in reversed(negative_prompt_spans(source)):
        source = source[:start] + " " * (end - start) + source[end:]
    return source


def _seconds(value: str) -> float:
    parts = value.split(":")
    return float(parts[-1]) + (60 * float(parts[-2]) if len(parts) > 1 else 0)


def explicit_character_profiles(source: str) -> list[dict[str, str]]:
    """Recognize inline or multiline Character A / Subject 1 / Role A definitions."""
    result = []
    seen = set()
    for match in _PROFILE.finditer(source):
        name = match.group("name")
        if name in seen:
            continue
        seen.add(name)
        result.append({"name": name, "details": (match.group("details") or "").strip()})
    # Names in a cast section are definitions even when they do not use the
    # special Character A / Subject 1 notation. Do not infer this from a name
    # followed by a colon elsewhere: that can be an actual screenplay turn.
    for start, end, kind in _descriptive_sections(source):
        if kind != "profiles":
            continue
        for match in re.finditer(
            r"(?m)^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?"
            r"(?P<name>[^\W\d][\w'’.-]*(?:[ \t]+[\w'’.-]+){0,3})"
            r"(?:[ \t]*\((?P<details>[^)\r\n]{1,160})\))?"
            r"(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?(?P<body>[^\r\n]+)", source[start:end]
        ):
            name = match.group("name")
            if name in seen or is_h3_production_label(name):
                continue
            seen.add(name)
            result.append({"name": name, "details": (match.group("details") or "").strip()})
        # Flattened imported briefs can put several definitions on one line.
        # Require the explicit parenthesized profile form here, rather than
        # treating every colon ("Opening: 50", "Effects density:") as a role.
        for match in re.finditer(
            r"(?:^|(?<=[.!?;】\]]))\s*"
            r"(?P<name>[^\W\d][\w'’.-]*(?:[ \t]+[\w'’.-]+){0,3})"
            r"[ \t]*\((?P<details>[^)\r\n]{1,160})\)[ \t]*:",
            source[start:end], re.M,
        ):
            name = match.group("name")
            if name not in seen and not is_h3_production_label(name):
                seen.add(name)
                result.append({"name": name, "details": match.group("details").strip()})
    return result


@lru_cache(maxsize=8)
def _timed_brief(source: str) -> tuple[str, tuple[tuple[float, float, str, int, int, str], ...]]:
    matches = _timed_headings(source)
    if len(matches) < 2:
        return "", ()
    ranges = [(_seconds(item.group("start")), _seconds(item.group("end"))) for item in matches]
    if any(end <= start for start, end in ranges) or any(
        start < ranges[index - 1][1] for index, (start, _end) in enumerate(ranges) if index
    ):
        return "", ()
    # A time range quoted in conversation is not an imported storyboard.
    if ranges[0][0] != 0 or any(
        start - ranges[index - 1][1] > 0.1 for index, (start, _end) in enumerate(ranges) if index
    ):
        return "", ()
    context = [source[:matches[0].start()].strip()]
    blocks = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        raw_body = source[match.end():stop]
        chapter = re.search(
            rf"\b(?:skill|phase|chapter|act|part)\s+(?:[A-Z]|\d+)\s*[·:–—-]"
            rf"[^\r\n]{{0,180}}?\(\s*t?{_CLOCK}\s*[-–—]\s*t?{_CLOCK}s?"
            r"[^)\r\n]*\)\s*[·*]?\s*$", raw_body, re.I,
        )
        if chapter:
            context.append(chapter.group().strip(" ·*\r\n"))
            stop = match.end() + chapter.start()
            raw_body = source[match.end():stop]
        body_offset = match.end() + len(raw_body) - len(raw_body.lstrip(" \\\r\n"))
        body = raw_body.strip(" ·*\\\r\n")
        body_end = stop
        # Notes inside an earlier timed block belong to that block. Only a
        # trailing production-notes section is shared across the sequence.
        if index == len(matches) - 1:
            note = _NOTES_HEADING.search(body)
            starts = [start for start, _end in production_note_spans(body)]
            if note:
                starts.append(note.start())
            if starts:
                note_start = min(starts)
                context.append(body[note_start:].strip())
                body_end = body_offset + note_start
                body = body[:note_start].strip()
        if not body:
            return "", ()
        blocks.append((*ranges[index], body, body_offset, body_end, (match.groupdict().get("title") or "").strip()))
    return "\n\n".join(item for item in context if item), tuple(blocks)


def authored_timed_brief(source: str) -> dict:
    """Return new containers so callers cannot mutate the cached source parse."""
    context, blocks = _timed_brief(str(source or ""))
    return {
        "context": context,
        "events": [
            {"text": text, "source_start_seconds": start, "source_end_seconds": end,
             "source_offset": offset, "source_end": stop,
             **({"source_title": title} if title else {})}
            for start, end, text, offset, stop, title in blocks
        ],
    }


_OPTICAL_MODIFIER = (
    r"(?:strong|subtle|slight|soft|hard|shallow|deep|volumetric|cinematic|filmic|"
    r"photorealistic|realistic|natural|warm|cool|dramatic|high|low|fine|coarse|"
    r"minimal|heavy|gentle|bright|dark|anamorphic|atmospheric|epic|professional|"
    r"very|extremely|no|without)"
)
_OPTICAL_SETTING = re.compile(
    rf"(?:{_OPTICAL_MODIFIER}[ -]+)*(?:motion blur|depth of field|light rays|"
    r"lighting|film grain|lens flare|bokeh|dynamic range|contrast|"
    r"cinematic perspective|filmic perspective|cinematic style|filmic style)|"
    r"(?:creating|making) (?:a |an )?(?:dramatic|cinematic|smooth) transition|"
    r"\d{1,3}(?:\.\d+)?\s*mm\s+(?:anamorphic\s+)?lens",
    re.IGNORECASE,
)


def authored_optical_settings(source: str) -> list[str]:
    """Copy standalone optical settings, never actions or quoted speech.

    These settings do not need creative rewriting or extra timeline space.
    A whole-clause match deliberately excludes camera moves, reveals, actions
    involving light and focus changes tied to a particular movement.
    """
    text = re.sub(r'<d>.*?</d>|"[^"\r\n]*"|“[^”\r\n]*”', " ", str(source or ""), flags=re.S)
    clauses = re.split(r"(?<=[.!?])\s+|[,;\r\n]+", text)
    result: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        cue = clause.strip(" \t.!?:")
        if _OPTICAL_SETTING.fullmatch(cue) and cue.casefold() not in seen:
            seen.add(cue.casefold())
            result.append(cue)
    return result


_SOUND_CUE = re.compile(
    r"(?:(?:sfx|sound(?:\s+effects?)?|audio)\s*:\s*)?"
    r"(?:(?:a|an|the)\s+)?"
    r"(?:(?:hard|soft|loud|quiet|faint|distant|close|nearby|sharp|deep|low|high|"
    r"heavy|light|short|brief|long|sudden|steady|sustained|rising|fading|gentle|"
    r"metallic|mechanical|electronic|rhythmic|muffled|echoing|rapid|continuous)\s+)*"
    r"(?:fizz(?:ing)?|hiss(?:ing)?|buzz(?:ing)?|hum(?:ming)?|roar(?:ing)?|rumble|rumbling|thunder|"
    r"clang|clank|clatter|rattle|thud|thump|click|crackle|whoosh|whistle|beep|"
    r"ringing|chime|ticking|rustle|footsteps)(?:s|es)?(?:\s+sounds?)?"
    r"(?:\s+(?:(?:is\s+)?(?:still\s+)?(?:going|continuing|fading(?:\s+(?:out|away))?)|"
    r"(?:still\s+)?(?:continues?|persists?|lingers?|fades?(?:\s+(?:out|away))?)|"
    r"keeps?\s+(?:going|ringing|echoing)))?",
    re.IGNORECASE,
)


def is_standalone_sound_cue(value: str) -> bool:
    return bool(_SOUND_CUE.fullmatch(str(value or "").strip(" \t.!?")))


def authored_sound_cues(source: str) -> list[str]:
    """Copy standalone nonverbal sound cues, never physical action or speech.

    A brief may write "Hard fizz, then a foam column erupts." The sound belongs
    in the event's audio field, not in its visible-action checklist. A whole
    clause match excludes "the courier clicks a latch" and "glass shatters";
    those still need visible staging and must not be satisfied by an SFX label.
    Explicit before/after dependencies are not split or moved by this helper.
    """
    text = re.sub(r'<d>.*?</d>|"[^"\r\n]*"|“[^”\r\n]*”|'
                  r"(?<!\w)['‘][^'’\r\n]*['’](?!\w)", " ", str(source or ""), flags=re.S)
    clauses = re.split(r"(?<=[.!?])\s+|[,;\r\n]+", text)
    result = []
    seen = set()
    for clause in clauses:
        cue = re.sub(r"^then\s+", "", clause.strip(" \t.!?"), flags=re.I)
        if is_standalone_sound_cue(cue) and cue.casefold() not in seen:
            seen.add(cue.casefold())
            result.append(cue)
    return result


def video_direction_source(source: str) -> str:
    """Scope an explicitly separate still-image brief to the starting frame.

    Keep the complete source for vision and event extraction. Only persistent
    video directions use this slice: a closed door in the requested still is
    not a command to keep that door closed throughout the subsequent video.
    Require both an image-generation request and an explicit prompt boundary;
    an ordinary first-frame instruction must never discard global constraints.
    """
    boundary = re.search(
        r"(?:^|\n)[ \t]*(?:#{1,6}[ \t]*)?(?:"
        r"(?:video|animation)\s+prompt\s*:|"
        r"(?:once|when)\s+[^\n]{0,60}?ready\s*,\s*use\s+it\s+as\s+"
        r"(?:a\s+)?reference[^\n]{0,120}?(?:this|following)\s+(?:video\s+)?prompt\s*:)",
        source, re.I,
    )
    if boundary and re.search(
        r"\b(?:produce|create|generate|draw)\s+(?:(?:one|a|an|the|single)\s+)?"
        r"(?:still|image|picture)\b|(?:^|\n)\s*(?:image|still)\s+prompt\s*:",
        source[:boundary.start()], re.I,
    ):
        return source[boundary.end():]
    return source


def explicit_negative_constraints(source: str) -> str:
    """Keep authored prohibition clauses as global directions, never as events."""
    source = video_direction_source(source)
    # Only sentence-start prohibitions or a clear Throughout clause qualify;
    # incidental narration such as "he finds no key" is not a global rule.
    # A flattened paste can put a notes heading directly after a prohibition.
    # Give explicit section labels a boundary so their prose is not mistaken
    # for part of that prohibition.
    source = _NOTES_HEADING.sub("\n", source)
    source = _SECTION_HEADING.sub(
        lambda match: "\n" if is_h3_production_label(match.group(1) or match.group(2)) else match.group(),
        source,
    )
    clauses = re.split(r"(?<=[.!?])\s+|[\r\n]+", source)
    return " ".join(dict.fromkeys(
        item.strip() for item in clauses
        if re.match(
            r"\s*(?:Throughout\s*,\s*)?(?:(?:no|never|without)\b|"
            r"(?:strictly\s+)?(?:prohibited|forbidden|disallowed)\s*:)",
            item, re.IGNORECASE,
        )
    ))
