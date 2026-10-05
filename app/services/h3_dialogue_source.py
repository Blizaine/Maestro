"""Conservative source checks for explicitly cued H3 dialogue."""

from __future__ import annotations

import re


_TAGGED_DIALOGUE = re.compile(r"<d\b[^>]*>.*?</d\s*>", re.IGNORECASE | re.DOTALL)
_NAME_TOKEN = r"(?:[A-Z][a-z][A-Za-z0-9_'’.-]*|[A-Z]{2,}[A-Z0-9_'’.-]*)"
_PERSON_NAME = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}}"
_HUMAN_ROLE = r"(?i:(?:(?:a|the)\s+)?(?:man|woman|person|speaker|host|guest|boy|girl|child)|he|she|they)"
_SPEECH_VERB = (
    r"(?i:says?|said|asks?|asked|replies?|replied|answers?|answered|"
    r"exclaims?|exclaimed|shouts?|shouted|whispers?|whispered|mutters?|"
    r"muttered|murmurs?|murmured|yells?|yelled|tells?|told|calls?|called|"
    r"continues?|continued|announces?|announced|screams?|screamed)"
)
_DELIVERY_TAIL = (
    r"(?:\s+(?:in|with)\s+(?:a\s+|an\s+|the\s+)?"
    r"[\w'’.-]+(?:\s+[\w'’.-]+){0,5})?"
)
_ATTRIBUTION = re.compile(
    rf"(?P<speaker>{_PERSON_NAME}|{_HUMAN_ROLE})\s+{_SPEECH_VERB}"
    rf"{_DELIVERY_TAIL}\s*[,;:]?\s*$"
)
_SCREENPLAY_LABEL = re.compile(
    rf"^\s*(?P<name>{_PERSON_NAME}|CHARACTER\s+\d+)"
    r"(?:\s*\([^()\r\n]{1,60}\))?\s*:\s*$"
)
_NON_SPEAKER_LABELS = {
    "action", "banner", "billboard", "board", "book", "camera", "caption",
    "card", "cast", "characters", "constraints", "dialogue", "display",
    "image", "inscription", "label", "letter", "lighting", "monitor",
    "movie", "negative", "newspaper", "notes", "novel", "painting", "photo",
    "picture", "plaque", "poster", "prompt", "scene", "screen", "script",
    "sign", "slogan", "song", "sound", "style", "summary", "text", "title",
    "vfx", "visual", "visuals", "window",
}


def _is_human_speech_cue(source: str, position: int) -> bool:
    prefix = source[:position]
    attribution = _ATTRIBUTION.search(prefix)
    if attribution:
        speaker_words = attribution.group("speaker").strip().casefold().split()
        if speaker_words and speaker_words[-1] not in _NON_SPEAKER_LABELS:
            return True

    line = prefix.rsplit("\n", 1)[-1]
    match = _SCREENPLAY_LABEL.fullmatch(line)
    if not match:
        return False
    name = match.group("name").strip()
    return name.casefold() not in _NON_SPEAKER_LABELS


def unclosed_h3_spoken_quote_error(source: str) -> str | None:
    """Return an admission error for a clearly cued, unclosed dialogue quote.

    Only straight and typographic double quotes are considered. Complete H3
    ``<d>`` blocks are already explicit dialogue and are skipped so quoted
    phrases inside a spoken line do not become source-boundary errors.
    """

    text = str(source or "")
    tagged = iter(_TAGGED_DIALOGUE.finditer(text))
    tag = next(tagged, None)
    stack: list[tuple[str, int]] = []
    index = 0
    while index < len(text):
        if tag is not None and index == tag.start():
            index = tag.end()
            tag = next(tagged, None)
            continue

        char = text[index]
        if char not in {'"', "“", "”"}:
            index += 1
            continue

        backslashes = 0
        previous = index - 1
        while previous >= 0 and text[previous] == "\\":
            backslashes += 1
            previous -= 1
        if backslashes % 2:
            index += 1
            continue

        if char == '"':
            if stack and stack[-1][0] == '"':
                stack.pop()
            else:
                stack.append(('"', index))
        elif char == "“":
            stack.append(("”", index))
        elif stack and stack[-1][0] == "”":
            stack.pop()
        index += 1

    if any(_is_human_speech_cue(text, opening) for _closing, opening in stack):
        return (
            "MiniMax H3 found an unclosed quoted dialogue line after a speaker cue. "
            "Close the dialogue quote and place stage or sound directions, such as "
            "an audience reaction, outside the quotation marks before enhancing."
        )
    return None
