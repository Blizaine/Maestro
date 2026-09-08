"""Mode-independent speech intent and duration budgets for Creative writing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from services.dialogue_timing import (
    DIALOGUE_DEFAULT_WORDS_PER_SECOND,
    DIALOGUE_MAX_WORDS_PER_SECOND,
)


def spoken_word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", str(text or "")))


def only_supplied_dialogue_requested(prompt: str) -> bool:
    return bool(re.search(
        r"\b(?:only\s+(?:use|speak|say)?\s*(?:these|the supplied|the quoted|the following|the provided)?\s*"
        r"(?:exact\s+)?(?:lines?|dialogue)|no\s+(?:extra|additional|other|added|new)\s+(?:lines?|dialogue|speech)|"
        r"do\s+not\s+(?:add|invent|write)\s+(?:any\s+)?(?:(?:extra|additional|other|new)\s+)?"
        r"(?:lines?|dialogue|speech))\b",
        str(prompt or ""), re.IGNORECASE,
    ))


def dialogue_forbidden(prompt: str) -> bool:
    """Recognize a scene-wide instruction, not a locally silent reaction/shot."""
    source = str(prompt or "")
    matches = re.finditer(
        r"\b(?:(?:no|without)\s+(?:spoken\s+)?(?:dialogue|speech|talking|voices?)|"
        r"(?:do not|don't|never)\s+(?:speak|talk|add dialogue)|"
        r"(?:silent|nonverbal)\s+(?:film|movie|video|sequence)|"
        r"(?:entire|whole)\s+(?:scene|clip|video|sequence)\s+(?:is\s+|stays\s+|remains\s+)?silent|"
        r"(?:music|instrumental)[ -]only)\b",
        source, re.IGNORECASE,
    )
    for match in matches:
        after = source[match.end():]
        if re.match(
            r"\s+(?:until|before|after|between|outside|during|from|"
            r"(?:in|for)\s+(?:the\s+)?(?:first|last|opening|final|initial|next|\d))\b",
            after, re.IGNORECASE,
        ):
            continue
        return True
    return False


def _speech_context(prompt: str) -> str:
    # Visible words are not a request for a narrator to read them aloud.
    return re.sub(
        r"\b(?:sign|banner|label|subtitle|caption|marquee|poster|billboard|screen|"
        r"monitor|display|placard|headline|logo|on-screen\s+text)\b[^.!?\r\n]{0,35}"
        r"\b(?:says?|reads?|shows?|displays?|bears?)\b\s*(?:\"[^\"]*\"|“[^”]*”)?",
        "visible text", str(prompt or ""), flags=re.IGNORECASE,
    )


def conversation_brief(prompt: str) -> bool:
    """Dialogue-led scenes need developed speech, not a token reaction line."""
    if dialogue_forbidden(prompt):
        return False
    return bool(re.search(
        r"\b(?:talk(?:s|ed|ing)?|convers(?:ation|e|es|ed|ing)|chat(?:s|ted|ting)?|"
        r"discuss(?:ion|es|ed|ing)?|debat(?:e|es|ed|ing)|argu(?:e|es|ed|ing)|"
        r"banter(?:s|ed|ing)?|interview(?:s|ed|ing)?|tell(?:s|ing)?|told|"
        r"explain(?:s|ed|ing)?|present(?:s|ed|ing|ation)?|announc(?:e|es|ed|ing)|"
        r"tutorial|walk[ -]?through|podcast|monologue|dialogue|speech)\b",
        _speech_context(prompt), re.IGNORECASE,
    ))


def creative_dialogue_expected(prompt: str) -> bool:
    if dialogue_forbidden(prompt):
        return False
    return conversation_brief(prompt) or bool(re.search(
        r"\b(?:say(?:s|ing)?|said|speak(?:s|ing)?|spoke|ask(?:s|ed|ing)?|"
        r"answer(?:s|ed|ing)?|repl(?:y|ies|ied|ying)|respond(?:s|ed|ing)?|"
        r"confront(?:s|ed|ing)?|greet(?:s|ed|ing)?|warn(?:s|ed|ing)?|"
        r"jok(?:e|es|ed|ing)|reun(?:ite|ites|ited|ion)|"
        r"(?:characters?|friends?|coworkers?|colleagues?|siblings?|couple)\s+"
        r"(?:meet|interact|reconcile)|(?:two|three|four|five|\d+)\s+"
        r"(?:characters|friends|coworkers|colleagues|siblings))\b",
        _speech_context(prompt), re.IGNORECASE,
    ))


@dataclass(frozen=True)
class DialogueBudget:
    minimum: int
    target: int
    maximum: int

    def instruction(self) -> str:
        return (
            f"Aim for {self.target} spoken words total across all speakers "
            f"(at least {self.minimum} for a developed exchange; hard maximum {self.maximum}). "
            f"Schedule speech at {DIALOGUE_DEFAULT_WORDS_PER_SECOND:g} words per second by default, "
            f"never above {DIALOGUE_MAX_WORDS_PER_SECOND:g}. "
            "Use the allocated speech time for specific ideas, questions, answers, and character reactions, "
            "not repeated greetings or filler. Include listener responses when appropriate. "
            "Exact supplied lines count toward this total and must remain verbatim. "
            "Reserve the remaining time for requested action and pauses."
        )


def creative_dialogue_budget(prompt: str, duration_seconds: float | None) -> DialogueBudget | None:
    if only_supplied_dialogue_requested(prompt) or not creative_dialogue_expected(prompt):
        return None
    duration = float(duration_seconds or 8.0)
    if not math.isfinite(duration) or duration <= 0:
        return None
    # Dialogue-led scenes allocate most of the clip to speech. A verbal beat
    # in an action scene uses half, leaving room for its physical progression.
    brief_speech = bool(re.search(
        r"\b(?:brief|short|sparse|minimal|occasional)\s+(?:tactical\s+)?"
        r"(?:dialogue|speech|exchange|spoken\s+reaction)|\b(?:one|a single)\s+(?:short\s+)?line\b",
        str(prompt or ""), re.IGNORECASE,
    ))
    speech_fraction = 0.25 if brief_speech else (0.85 if conversation_brief(prompt) else 0.5)
    maximum = max(1, math.floor(duration * DIALOGUE_MAX_WORDS_PER_SECOND))
    target = min(maximum, max(1, round(duration * speech_fraction * DIALOGUE_DEFAULT_WORDS_PER_SECOND)))
    return DialogueBudget(1 if brief_speech else max(1, math.ceil(target * 0.75)), target, maximum)
