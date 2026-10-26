"""Fish Drama 3 script-format recovery for generated speech fragments."""
from __future__ import annotations

from dataclasses import dataclass
import re


_SUPPORTED_TAGS = frozenset({
    "whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful",
})
_DELIVERY_VERBS = (
    "scream", "screams", "whisper", "whispers", "laugh", "laughs",
    "chuckle", "chuckles", "sigh", "sighs", "gasp", "gasps",
    "shout", "shouts", "yell", "yells", "moan", "moans",
    "groan", "groans", "sob", "sobs", "cry", "cries", "pause",
    "pauses", "breathe", "breathes", "murmur", "murmurs", "mutter",
    "mutters", "hiss", "hisses", "pant", "pants", "exhale", "exhales",
    "inhale", "inhales", "stammer", "stammers", "cough", "coughs",
    "hum", "hums", "sing", "sings", "snort", "snorts", "wheeze",
    "wheezes",
)
_DELIVERY_CUE_RE = re.compile(
    rf"^(?P<cue>(?:{'|'.join(_DELIVERY_VERBS)})(?:\b[^<>]*)?)$",
    re.IGNORECASE,
)
_SAFE_CUE_TEXT_RE = re.compile(r"^[\w\s,'’.-]{1,120}$", re.UNICODE)
_ANGLE_TOKEN_RE = re.compile(r"<[^<>\r\n]{1,160}>")
_UNMATCHED_CUE_RE = re.compile(r"\[(?P<cue>[^\[\]\r\n]*)$")
_LEADING_CUE_RE = re.compile(r"^\[(?P<cue>[^\[\]\r\n]{1,160})\]")
_LEADING_TAG_RE = re.compile(r"^<(?P<name>[A-Za-z][\w:.-]*)>")
_TAG_TOKEN_RE = re.compile(r"<\s*(/?)\s*([A-Za-z][\w:.-]*)\b([^<>]*?)>")


@dataclass(frozen=True)
class FishScriptFormat:
    """Normalize clear model-authored cue typos and recover safe cue seams.

    `source_text` protects exact angle tokens and cue prefixes supplied by the
    user or by the model's native-format instructions.
    """

    source_text: str = ""

    def normalize_fragment(self, text: str) -> tuple[str, tuple[str, ...]]:
        """Convert only unmistakable one-off delivery cues to Fish brackets."""
        value = str(text or "")
        protected_tokens = set(_ANGLE_TOKEN_RE.findall(str(self.source_text or "")))
        changed: list[str] = []

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if token in protected_tokens:
                return token
            content = token[1:-1].strip()
            if content.casefold() in _SUPPORTED_TAGS:
                return token
            cue = _DELIVERY_CUE_RE.fullmatch(content)
            if cue is None or not _SAFE_CUE_TEXT_RE.fullmatch(content):
                return token

            leading = re.match(r"[A-Za-z]+", content)
            if leading is None or leading.group(0).casefold() not in _DELIVERY_VERBS:
                return token

            changed.append(token)
            return f"[{content}]"

        return _ANGLE_TOKEN_RE.sub(replace, value), tuple(changed)

    def recover_continuation(
        self, previous: str, addition: str
    ) -> tuple[str, str, bool]:
        """Close an inherited open cue only before a separate Fish directive."""
        left = str(previous or "")
        right = str(addition or "")
        match = _UNMATCHED_CUE_RE.search(left)
        if match is None or self._source_protects_open_cue(match.group(0)):
            return left, right, False
        if not _starts_with_independent_directive(right):
            return left, right, False
        if not _has_safe_lexical_seam(left, right):
            return left, right, False
        return left + "]", right, True

    def _source_protects_open_cue(self, fragment: str) -> bool:
        source = str(self.source_text or "")
        start = 0
        while True:
            opening = source.find("[", start)
            if opening < 0:
                return False
            closing = source.find("]", opening + 1)
            end = len(source) if closing < 0 else closing + 1
            source_cue = source[opening:end]
            if source_cue.startswith(fragment):
                return True
            start = opening + 1


def _starts_with_independent_directive(text: str) -> bool:
    """Recognize a complete cue or balanced supported paired tag at the seam."""
    value = str(text or "").lstrip()
    cue = _LEADING_CUE_RE.match(value)
    if cue and _SAFE_CUE_TEXT_RE.fullmatch(cue.group("cue")):
        return True

    opening = _LEADING_TAG_RE.match(value)
    if opening is None:
        return False
    first_name = opening.group("name").casefold()
    if first_name not in _SUPPORTED_TAGS:
        return False

    stack: list[str] = []
    saw_first_close = False
    for match in _TAG_TOKEN_RE.finditer(value):
        closing, raw_name, attributes = match.groups()
        name = raw_name.casefold()
        if name not in _SUPPORTED_TAGS or attributes.strip():
            return False
        if closing:
            if not stack or stack[-1] != name:
                return False
            stack.pop()
            if not stack and name == first_name:
                saw_first_close = True
        else:
            stack.append(name)
    return saw_first_close and not stack


def _has_safe_lexical_seam(previous: str, addition: str) -> bool:
    """Reject a directive boundary that could split one spoken word."""
    if not previous or not addition or addition[0].isspace():
        return True
    if not previous[-1].isalpha():
        return True

    visible = addition.lstrip()
    while True:
        cue = _LEADING_CUE_RE.match(visible)
        if cue is None or not _SAFE_CUE_TEXT_RE.fullmatch(cue.group("cue")):
            break
        visible = visible[cue.end():].lstrip()

    # Native paired tags add markup around audible text. Ignore them when
    # checking whether the actual characters at the seam touch as one word.
    visible = _TAG_TOKEN_RE.sub("", visible).lstrip()
    if not visible:
        return True
    first = visible[0]
    return not (first.isalpha() and first.islower())
