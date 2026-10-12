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


def dialogue_minimum_threshold(
    minimum_words: int,
    *,
    automatic: bool,
    exchange_complete: bool,
) -> int:
    minimum = max(0, int(minimum_words))
    # A duration-derived minimum guides the writer toward a developed exchange,
    # but cannot justify padding a complete natural exchange. Explicit minimums
    # and incomplete exchanges remain strict; the hard maximum is checked
    # separately and always remains strict.
    if automatic and exchange_complete:
        return 0
    return minimum


def explicit_dialogue_minimum_words(prompt: str) -> int | None:
    """Read a clearly stated numeric minimum for the spoken exchange only."""

    source = re.sub(
        r'"[^"\r\n]*"|“[^”\r\n]*”|‘[^’\r\n]*’|`[^`\r\n]*`',
        " ",
        str(prompt or ""),
    )
    minimum = r"(?:at\s+least|no\s+fewer\s+than|not\s+less\s+than|minimum(?:\s+of)?)"
    speech_noun = r"(?:dialogue|exchange|conversation|speech)"
    minima = [
        int(match.group(1))
        for match in re.finditer(
            rf"\b{minimum}\s+(\d{{1,3}})\s+"
            rf"(?:spoken\s+words?|dialogue\s+words?|words?\s+of\s+(?:spoken\s+)?{speech_noun})\b",
            source,
            flags=re.IGNORECASE,
        )
    ]
    # Also accept a minimum attached to the speech noun by an explicit
    # requirement verb, or phrased as a minimum ``in/for`` the exchange. Mere
    # proximity to “dialogue” is insufficient: visual copy may share a clause.
    clause_pattern = re.compile(r"[^.!?;\r\n]+", flags=re.DOTALL)
    attached_minimum_patterns = (
        re.compile(
            rf"\b{speech_noun}\b[^.!?;\r\n]{{0,40}}?"
            rf"\b(?:must|should|needs?\s+to|has\s+to|requires?|contain(?:s)?|include(?:s)?|use(?:s)?|be)\b"
            rf"[^.!?;\r\n]{{0,25}}?\b{minimum}\s+(?:of\s+)?(\d{{1,3}})\s+words?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\b{minimum}\s+(?:of\s+)?(\d{{1,3}})\s+words?\b"
            rf"[^.!?;\r\n]{{0,25}}?\b(?:in|for|across)\s+(?:the\s+)?{speech_noun}\b",
            flags=re.IGNORECASE,
        ),
    )
    for clause in clause_pattern.findall(source):
        for pattern in attached_minimum_patterns:
            minima.extend(int(match.group(1)) for match in pattern.finditer(clause))
    return max(minima) if minima else None


def dialogue_minimum_satisfied(
    actual_words: int,
    minimum_words: int,
    *,
    automatic: bool = False,
    exchange_complete: bool = True,
) -> bool:
    return int(actual_words) >= dialogue_minimum_threshold(
        minimum_words,
        automatic=automatic,
        exchange_complete=exchange_complete,
    )


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
    # Natural restriction lists often begin with a scene-specific noun before
    # reaching the speech item ("No baby, dialogue, collision, or cuts").
    # Collapse only a bare list item: positive clauses such as "dialogue begins
    # later" do not match, and the quote/temporal checks below still apply.
    source = re.sub(
        r"\b(?:no|without)\s+[^,;.!?\r\n]{1,80},\s*"
        r"(?:[^,;.!?\r\n]{1,80},\s*)*"
        r"(?:and\s+|or\s+)?(?:spoken\s+)?"
        r"(?:dialogue|speech|talking|voices?)\b"
        r"(?=\s*(?:[,.;!?]|$))",
        "no dialogue", source, flags=re.IGNORECASE,
    )
    # A shared "no" can govern comma- or conjunction-separated restrictions.
    source = re.sub(
        r"\b(?:no|without)\s+(?:(?:cuts?|slow motion|subtitles?|captions?|music|"
        r"narration|logos?|watermarks?|magic)(?:\s*,\s*(?:(?:and|or)\s+)?|"
        r"\s+(?:and|or)\s+))+(dialogue|speech|talking|voices?)\b"
        r"(?=\s*(?:[,.;!?]|$|until\b|before\b|after\b|during\b|for\b|in\b))",
        r"no \1", source, flags=re.IGNORECASE,
    )
    matches = re.finditer(
        r"\b(?:(?:no|without)\s+(?:spoken\s+)?(?:dialogue|speech|talking|voices?)|"
        r"(?:do not|don't|never)\s+(?:speak|talk|add dialogue)|"
        r"(?:no[ -]?one|nobody)\s+(?:ever\s+)?(?:speaks?|talks?|"
        r"mouths?\s+(?:any\s+)?words?)|"
        r"(?:silent|nonverbal)\s+"
        r"(?:(?:\d+(?:\.\d+)?(?:[- ](?:second|minute)s?|s)|\d+:\d+|"
        r"live[- ]action|animated|cinematic|dramatic|short|feature|"
        r"martial[- ]arts|vertical|horizontal|portrait|landscape|"
        r"photorealistic|studio|product|commercial)\s+){0,10}"
        r"(?:scene|film|movie|video|sequence)|"
        r"(?:entire|whole)\s+(?:scene|clip|film|movie|video|sequence)\s+(?:is\s+|stays\s+|remains\s+)?silent|"
        r"(?:music|instrumental)[ -]only)\b",
        source, re.IGNORECASE,
    )
    for match in matches:
        # A character can say "No one speaks here". Only quotes introduced by
        # a speech attribution are dialogue; a quote around a pasted prompt is
        # still allowed to contain scene-wide production restrictions.
        before = source[:match.start()]
        straight_start = before.rfind('"') if before.count('"') % 2 else -1
        curly_start = before.rfind("“") if before.rfind("“") > before.rfind("”") else -1
        quote_start = max(straight_start, curly_start)
        quote_lead = before[:quote_start] if quote_start >= 0 else ""
        if quote_start >= 0 and re.search(
            r"\b(?:says?|said|speaks?|spoke|asks?|asked|replies?|replied)\s*,?\s*$",
            quote_lead, re.IGNORECASE,
        ):
            continue

        # Preserve local silence before later requested speech. Explicit
        # entire/whole/full-scene scope remains global even with "during".
        clause_start = max(
            before.rfind(mark) for mark in (".", "!", "?", ";", "\n", "\r")
        ) + 1
        clause_prefix = before[clause_start:]
        prefix_is_temporal = re.match(
            r"\s*(?:(?:until|before|after|between|outside|during|from)\b|"
            r"(?:in|for)\s+(?:the\s+)?(?:first|last|opening|final|initial|next|"
            r"\d+(?:\.\d+)?(?:\s+(?:seconds?|minutes?|beats?|shots?))?))\b",
            clause_prefix, re.IGNORECASE,
        )
        prefix_is_global = re.match(
            r"\s*(?:during|for|in|throughout)\s+(?:the\s+)?"
            r"(?:entire|whole|full)\s+(?:scene|clip|film|movie|video|sequence)\b",
            clause_prefix, re.IGNORECASE,
        )
        if prefix_is_temporal and not prefix_is_global:
            continue
        after = source[match.end():]
        suffix_is_temporal = re.match(
            r"\s+(?:plays?\s+)?(?:until|before|after|between|outside|during|from|"
            r"(?:in|for)\s+(?:the\s+)?(?:first|last|opening|final|initial|next|\d))\b",
            after, re.IGNORECASE,
        )
        suffix_is_global = re.match(
            r"\s+(?:plays?\s+)?(?:during|for|in|throughout)\s+(?:the\s+)?"
            r"(?:entire|whole|full)\s+(?:scene|clip|film|movie|video|sequence)\b",
            after, re.IGNORECASE,
        )
        if suffix_is_temporal and not suffix_is_global:
            continue
        return True
    return False


def _speech_context(prompt: str) -> str:
    from services.h3_authored_brief import positive_instruction_text

    # A negative mention of banter or dialogue cannot create a speech quota.
    source = positive_instruction_text(str(prompt or ""))
    # Visible words are not a request for a narrator to read them aloud.
    return re.sub(
        r"\b(?:sign|banner|label|subtitle|caption|marquee|poster|billboard|screen|"
        r"monitor|display|placard|headline|logo|on-screen\s+text)\b[^.!?\r\n]{0,35}"
        r"\b(?:says?|reads?|shows?|displays?|bears?)\b\s*(?:\"[^\"]*\"|“[^”]*”)?",
        "visible text", source, flags=re.IGNORECASE,
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


def requested_dialogue_turns(prompt: str) -> int | None:
    """Return an explicit bounded turn count such as ``four-turn conversation``."""

    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    }
    match = re.search(
        r"\b(one|two|three|four|five|six|[1-6])(?:[ -])turn\s+"
        r"(?:natural\s+)?(?:conversation|dialogue|exchange)\b",
        _speech_context(prompt), re.IGNORECASE,
    )
    if not match:
        return None
    token = match.group(1).casefold()
    return words.get(token, int(token) if token.isdigit() else None)


def explicit_multi_speaker_dialogue(prompt: str) -> bool:
    """Recognize clear reciprocal/group intent without treating any speech as a conversation."""

    if dialogue_forbidden(prompt) or requested_dialogue_turns(prompt) == 1:
        return False
    source = _speech_context(prompt)
    if re.search(
        r"\b(?:write|create|deliver|perform|give)\s+(?:(?:a|an|short|brief|solo|single-speaker)\s+){0,3}"
        r"monologue\b|\bmonologue\s+(?:from|by)\b",
        source,
        re.IGNORECASE,
    ):
        return False

    conversation_noun = re.compile(
        r"\b(?:conversation|dialogue|exchange|discussion|debate|chat|banter)\b",
        re.IGNORECASE,
    )
    explicit_link = re.compile(r"\b(?:between|among)\b", re.IGNORECASE)
    named_group = re.compile(
        r"\b(?:the\s+)?(?:two|three|four|five|both|a\s+pair\s+of|the\s+pair\s+of)\s+"
        r"(?:[\w'-]+\s+){0,3}(?:people|persons|strangers|friends|characters|speakers|voices|"
        r"partners|coworkers|colleagues|siblings|mechanics|workers|soldiers|guards|students|"
        r"neighbors|neighbours|passengers|children|adults|teammates|classmates|actors|couples)\b",
        re.IGNORECASE,
    )
    dialogue_action = re.compile(
        r"\b(?:talk(?:s|ed|ing)?|speak(?:s|ing)?|spoke|convers(?:e|es|ed|ing)|"
        r"chat(?:s|ted|ting)?|discuss(?:es|ed|ing)?|debate(?:s|d|ing)?|"
        r"argu(?:e|es|ed|ing)|exchange(?:s|d|ing)?|banter(?:s|ed|ing)?)\b",
        re.IGNORECASE,
    )
    clauses = re.split(r"(?<=[.!?;])\s+|[\r\n]+", source)
    for clause in clauses:
        if conversation_noun.search(clause) and (
            explicit_link.search(clause) or named_group.search(clause)
        ):
            return True
        if re.search(r"\b(?:each other|one another)\b", clause, re.IGNORECASE) and dialogue_action.search(clause):
            return True
        group_match = named_group.search(clause)
        if group_match:
            tail = clause[group_match.end():]
            action = dialogue_action.search(tail)
            if action and not re.search(
                r"\b(?:listen|listening|silent|quiet|narrator|voiceover|audience)\b",
                tail[:action.start()],
                re.IGNORECASE,
            ):
                return True
    return False


def compact_dialogue_brief(prompt: str) -> bool:
    """Return whether the user explicitly asks for a brief exchange."""

    return bool(re.search(
        r"\b(?:brief|short|concise)\b[^.!?\r\n]{0,40}"
        r"\b(?:conversation|dialogue|exchange)\b",
        str(prompt or ""), re.IGNORECASE,
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
    minimum_is_automatic: bool = True

    def instruction(self) -> str:
        minimum_guidance = (
            f"duration-derived minimum {self.minimum} is a writing target once the exchange is complete"
            if self.minimum_is_automatic
            else f"explicit minimum {self.minimum} is strict"
        )
        return (
            f"Aim for {self.target} spoken words total across all speakers "
            f"({minimum_guidance}; hard maximum {self.maximum}). "
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
    sparse_speech = bool(re.search(
        r"\b(?:brief|short|sparse|minimal|occasional)\s+(?:tactical\s+)?"
        r"(?:dialogue|speech|exchange|spoken\s+reaction|line)|"
        r"\b(?:one|a single)\s+(?:short\s+)?line\b",
        str(prompt or ""), re.IGNORECASE,
    ))
    compact_exchange = compact_dialogue_brief(prompt)
    if sparse_speech:
        speech_fraction = 0.25
    elif compact_exchange:
        # A concise exchange must not be padded until it crowds out the action.
        # Four complete technical turns in a 14.4s action scene naturally land
        # around 11-16 words, while an open dialogue-led scene remains denser.
        speech_fraction = 0.35
    else:
        speech_fraction = 0.85 if conversation_brief(prompt) else 0.5
    maximum = max(1, math.floor(duration * DIALOGUE_MAX_WORDS_PER_SECOND))
    target = min(maximum, max(1, round(duration * speech_fraction * DIALOGUE_DEFAULT_WORDS_PER_SECOND)))
    # A conversation draft at 75% of the target was routinely accepted as a
    # single short reaction per window. Keep sustained speech close to its
    # allocation while leaving mixed action scenes their wider breathing room.
    minimum_fraction = 0.9 if conversation_brief(prompt) and not compact_exchange else 0.75
    automatic_minimum = 1 if sparse_speech else max(1, math.ceil(target * minimum_fraction))
    explicit_minimum = explicit_dialogue_minimum_words(prompt)
    minimum_is_automatic = explicit_minimum is None
    minimum = automatic_minimum if minimum_is_automatic else explicit_minimum
    if explicit_minimum is not None:
        target = min(maximum, max(target, explicit_minimum))
    return DialogueBudget(minimum, target, maximum, minimum_is_automatic)


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


_DIALOGUE_ACTION_RE = re.compile(
    r"\b(?:talk(?:s|ed|ing)?|speak(?:s|ing)?|spoke|convers(?:e|es|ed|ing)|"
    r"chat(?:s|ted|ting)?|discuss(?:es|ed|ing)?|debate(?:s|d|ing)?|"
    r"argu(?:e|es|ed|ing)|exchange(?:s|d|ing)?|banter(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_DIALOGUE_GROUP_RE = re.compile(
    r"\b(?:each\s+other|one\s+another|all\s+(?:of\s+them\s+)?)\b|"
    r"\b(?:everyone|everybody|the\s+(?:group|trio|pair))\b[^.!?;\r\n]{0,80}"
    r"\b(?:talk|speak|convers|chat|discuss|debate|argu|exchange|banter)\w*\b|"
    r"\b(?:talk|speak|convers|chat|discuss|debate|argu|exchange|banter)\w*"
    r"[^.!?;\r\n]{0,80}\b(?:all|everyone|everybody|the\s+(?:group|trio|pair))\b",
    re.IGNORECASE,
)
_DIALOGUE_LISTENER_RE = re.compile(
    r"\b(?:listen(?:s|ed|ing)?|(?:is|are|was|were)\s+(?:deliberately\s+)?"
    r"(?:silent|quiet|listening)|remain(?:s|ed)?\s+(?:silent|quiet)|"
    r"stay(?:s|ed)?\s+(?:silent|quiet)|keep(?:s|ing)?\s+(?:silent|quiet)|"
    r"sit(?:s|ting)?\s+(?:silently|quietly)|(?:do|does|did|will|won't|doesn't|"
    r"don't|didn't)\s+not\s+(?:speak|talk|join\s+the\s+conversation)|"
    r"(?:doesn't|don't|didn't|won't)\s+(?:speak|talk|join\s+the\s+conversation)|"
    r"(?:never|refuses?\s+to)\s+(?:speak|talk|join\s+the\s+conversation)|"
    r"mouths?\s+(?:stay\s+)?closed|holds?\s+(?:their|his|her)\s+tongue|"
    r"silent\s+listener)\b",
    re.IGNORECASE,
)


def requested_dialogue_speakers(prompt: str) -> list[str]:
    """Return named speakers explicitly included in a mutual group exchange.

    A list of characters is not a speaking contract by itself. Require an
    explicit reciprocal or collective conversation cue, then remove anyone
    the brief deliberately leaves listening or silent in that same exchange
    clause. Nearby scene beats do not change the original exchange's cast.
    """

    if not conversation_brief(prompt) or dialogue_forbidden(prompt):
        return []
    try:
        from services.h3_story_ledger import extract_h3_source_intent

        cast_names = extract_h3_source_intent(prompt).get("cast_names") or []
    except (ImportError, KeyError, TypeError, ValueError):
        return []
    if len(cast_names) < 2:
        return []

    source = _speech_context(prompt)
    source = re.sub(r"<d\b[^>]*>[\s\S]*?</d>", " ", source, flags=re.IGNORECASE)
    clauses = [
        clause.strip()
        for clause in re.split(r"(?<=[.!?;])\s+|[\r\n]+", source)
        if clause.strip()
    ]

    first_names = {}
    for name in cast_names:
        first_names.setdefault(name.split()[0].casefold(), []).append(name)

    def name_in(text: str, name: str) -> bool:
        full_name = re.search(
            rf"(?<![\w]){re.escape(name)}(?![\w])", text, flags=re.IGNORECASE,
        )
        if full_name:
            return True
        first_name = name.split()[0]
        return len(first_names.get(first_name.casefold(), [])) == 1 and bool(re.search(
            rf"(?<![\w]){re.escape(first_name)}(?![\w])", text, flags=re.IGNORECASE,
        ))

    def remove_named_source_references(text: str) -> str:
        # Franchise and source labels such as "Ron Swanson from Parks and
        # Rec" are present in cast extraction as proper nouns. They describe
        # a character; they are not additional people in the exchange.
        return re.sub(
            r"\bfrom\s+(?:the\s+)?[A-Z][\w'’-]*"
            r"(?:\s+(?:and|&)\s+[A-Z][\w'’-]*)*"
            r"(?:\s+[A-Z][\w'’-]*){0,2}",
            " ", text,
        )

    def listener_subjects(text: str) -> set[str]:
        """Find cast names that are the local grammatical subject of a cue."""
        occurrences: list[tuple[int, int, str]] = []
        for name in cast_names:
            aliases = [name]
            first = name.split()[0]
            if len(first_names.get(first.casefold(), [])) == 1:
                aliases.append(first)
            for alias in aliases:
                occurrences.extend(
                    (match.start(), match.end(), name)
                    for match in re.finditer(
                        rf"(?<![\w]){re.escape(alias)}(?![\w])", text,
                        flags=re.IGNORECASE,
                    )
                )
        # Prefer a full name over its overlapping first-name alias.
        occurrences.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        distinct: list[tuple[int, int, str]] = []
        for occurrence in occurrences:
            if any(
                occurrence[0] < prior[1] and prior[0] < occurrence[1]
                for prior in distinct
            ):
                continue
            distinct.append(occurrence)

        explicit: set[str] = set()
        for cue in _DIALOGUE_LISTENER_RE.finditer(text):
            before = [item for item in distinct if item[1] <= cue.start()]
            direct_subjects = [
                item for item in before
                if re.fullmatch(
                    r"\s*(?:,\s*)?(?:(?:who|that)\s+)?"
                    r"(?:(?:still|quietly|silently|closely|intently|patiently|"
                    r"calmly|deliberately|attentively)\s+)?",
                    text[item[1]:cue.start()],
                    flags=re.IGNORECASE,
                )
            ]
            if not direct_subjects:
                continue
            nearest = direct_subjects[-1]
            subjects = [nearest]
            cursor = nearest[0]
            for previous in reversed([item for item in before if item[1] <= nearest[0]]):
                if re.fullmatch(
                    r"\s*(?:,\s*(?:(?:and|&)\s*)?|(?:and|&)\s+)\s*",
                    text[previous[1]:cursor],
                    flags=re.IGNORECASE,
                ):
                    subjects.append(previous)
                    cursor = previous[0]
                else:
                    break
            explicit.update(item[2] for item in subjects)
        return explicit

    for clause_index, clause in enumerate(clauses):
        if not _DIALOGUE_GROUP_RE.search(clause) or not _DIALOGUE_ACTION_RE.search(clause):
            continue
        speaker_clause = remove_named_source_references(clause)
        participants = [name for name in cast_names if name_in(speaker_clause, name)]
        explicit_listeners = listener_subjects(clause)
        participants = [
            name for name in participants
            if name not in explicit_listeners
        ]
        if len(participants) >= 2:
            return participants
    return []


def canonical_requested_dialogue_speaker(
    speaker: str,
    expected_speakers: list[str],
) -> str:
    """Expand a uniquely resolvable whole-token shortening to its cast name."""

    normalized = " ".join(str(speaker or "").split())
    if not normalized:
        return normalized
    key = normalized.casefold()
    exact = [name for name in expected_speakers if name.casefold() == key]
    if exact:
        return exact[0]
    prefixes = [
        name for name in expected_speakers
        if name.casefold().startswith(key + " ")
    ]
    if len(prefixes) == 1:
        return prefixes[0]
    return normalized


def creative_dialogue_violations(
    prompt: str,
    turns: list[dict],
    *,
    minimum_words: int | None = None,
    maximum_words: int | None = None,
    automatic_minimum: bool = False,
) -> list[str]:
    """Audit a structured exchange against its speaker, word, and topic contracts."""

    valid_turns = [
        turn for turn in turns
        if isinstance(turn, dict)
        and isinstance(turn.get("speaker"), str)
        and turn["speaker"].strip()
        and isinstance(turn.get("text"), str)
        and turn["text"].strip()
    ]
    if len(valid_turns) != len(turns):
        return ["every dialogue turn needs a nonempty speaker and spoken text"]

    from services.h3_story_ledger import extract_locked_dialogue

    violations: list[str] = []
    actual_words = sum(spoken_word_count(turn["text"]) for turn in valid_turns)
    if maximum_words is not None and actual_words > int(maximum_words):
        violations.append(f"{actual_words} spoken words; at most {int(maximum_words)} allowed")

    expected_speakers = requested_dialogue_speakers(prompt)
    actual_speakers = [" ".join(turn["speaker"].split()) for turn in valid_turns]

    def speaker_matches(expected: str, actual: str) -> bool:
        expected_key = expected.casefold()
        actual_key = actual.casefold()
        if expected_key == actual_key:
            return True
        # A first-name shortening is safe only when it identifies one requested
        # speaker uniquely. Do not turn arbitrary cast mentions into new owners.
        candidates = {
            name.casefold() for name in expected_speakers
            if name.casefold() == actual_key or name.casefold().startswith(actual_key + " ")
        }
        return candidates == {expected_key}

    missing_speakers = [
        name for name in expected_speakers
        if not any(speaker_matches(name, actual) for actual in actual_speakers)
    ]
    if missing_speakers:
        violations.append("missing requested speaking participants: " + ", ".join(missing_speakers))
    unexpected_speakers = [
        actual for actual in actual_speakers
        if expected_speakers and not any(speaker_matches(expected, actual) for expected in expected_speakers)
    ]
    if unexpected_speakers:
        violations.append("unrequested dialogue speakers: " + ", ".join(unexpected_speakers))

    if (
        len(expected_speakers) < 2
        and explicit_multi_speaker_dialogue(prompt)
        and len({speaker.casefold() for speaker in actual_speakers}) < 2
    ):
        violations.append("the requested conversation needs at least two distinct speakers")

    exact_turns = requested_dialogue_turns(prompt)
    if exact_turns is not None and len(valid_turns) != exact_turns:
        violations.append(f"the user requested exactly {exact_turns} turns; found {len(valid_turns)}")

    spoken = " ".join(turn["text"] for turn in valid_turns)
    missing_topics = [
        topic for topic in requested_dialogue_topics(prompt)
        if not dialogue_topic_covered(topic, spoken)
    ]
    if missing_topics:
        violations.append("missing spoken talking points: " + "; ".join(missing_topics))

    locked_lines = [item["text"] for item in extract_locked_dialogue(prompt)]
    missing_quotes = [line for line in locked_lines if line not in [turn["text"] for turn in valid_turns]]
    if missing_quotes:
        violations.append("supplied dialogue must remain verbatim: " + "; ".join(missing_quotes))

    if minimum_words is not None and not dialogue_minimum_satisfied(
        actual_words,
        int(minimum_words),
        automatic=automatic_minimum,
        exchange_complete=not violations,
    ):
        violations.append(f"{actual_words} spoken words; at least {int(minimum_words)} required")
    return violations
