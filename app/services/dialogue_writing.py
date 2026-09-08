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
    # A conversation draft at 75% of the target was routinely accepted as a
    # single short reaction per window. Keep sustained speech close to its
    # allocation while leaving mixed action scenes their wider breathing room.
    minimum_fraction = 0.9 if conversation_brief(prompt) else 0.75
    return DialogueBudget(1 if brief_speech else max(1, math.ceil(target * minimum_fraction)), target, maximum)


def requested_dialogue_topics(prompt: str) -> list[str]:
    """Read explicit talking-point lists, excluding visual/style checklists.

    These are coverage hints for authored speech, never extra dialogue for a
    silent scene or an exact-only script. Free prose remains the writer's job.
    """
    if not conversation_brief(prompt) or only_supplied_dialogue_requested(prompt):
        return []
    topics: list[str] = []
    in_topics = False
    for line in str(prompt or "").splitlines():
        text = line.strip()
        if re.match(
            r"^(?:features?|topics?|talking points?|discussion points?|"
            r"things to (?:discuss|explain|cover)|(?:discuss|explain|cover)(?: the following)?)"
            r"(?:\s+(?:include|includes|are))?\s*:\s*$", text, re.IGNORECASE,
        ):
            in_topics = True
            continue
        bullet = re.match(r"^(?:[•*\-–]|\d+[.)])\s+(.+)$", text)
        if in_topics and bullet:
            topic = bullet.group(1).strip(" *.!;:")
            if topic and topic not in topics:
                topics.append(topic)
        elif text:
            in_topics = False
    return topics


def _dialogue_topic_terms(text: str) -> set[str]:
    # Expand common spoken equivalents before comparing named talking points.
    # Only dialogue text enters this check; camera prose cannot satisfy it.
    text = re.sub(r"\bUI\b", "user interface", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSFX\b", "sound effects", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:saving|saved|saves)\b", "save", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:sharing|shared|shares)\b", "share", text, flags=re.IGNORECASE)
    ignored = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "the", "to", "with", "w",
        "feature", "features", "support", "supports", "include", "includes",
        "generate", "generates", "generation",
    }
    return {
        word[:5] if len(word) > 5 else word
        for word in re.findall(r"[^\W_]+", text.casefold())
        if word not in ignored
    }


def dialogue_topic_covered(topic: str, spoken_text: str) -> bool:
    """Conservative lexical check for an explicit named discussion topic.

    Allow ordinary paraphrasing and inflections, but require distinctive
    acronyms (including their spoken expansions) such as SFX or RefMod.
    This is a writing aid, not a general semantic/factual judge.
    """
    terms = _dialogue_topic_terms(topic)
    if not terms:
        return True
    spoken = _dialogue_topic_terms(spoken_text)
    named_terms = re.findall(r"\b(?:[A-Z]{2,}[a-z]*|[A-Z][a-z]+[A-Z][A-Za-z]*)\b", topic)
    if any(not _dialogue_topic_terms(name) <= spoken for name in named_terms):
        return False
    return len(terms & spoken) >= max(1, math.ceil(len(terms) * 0.65))
