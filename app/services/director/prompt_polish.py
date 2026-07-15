"""
Director Prompt Polish — applies model-specific prompt optimization to Director output.

Three modes:
  - full_guide:   Inject the full enhance guide into the planner's system prompt
  - light_guide:  Inject a lightweight dialect cheat sheet into the planner's system prompt
  - third_pass:   Run each prompt through the enhance pipeline after planning

The setting is stored as `director_prompt_polish` in services config.
"""

import os
import re
from functools import lru_cache
from typing import Optional

_ENHANCE_DIR = os.path.join(os.path.dirname(__file__), "..", "llm_guides", "enhance")
_DIALECT_DIR = os.path.join(os.path.dirname(__file__), "..", "llm_guides", "dialect")


# ── Image Prompt Sanitization (Layer 1) ──────────────────────────────────
# Deterministic post-process for Pass 2 image_prompts and Pass 3 image
# polish output. Catches violations of rules in image_prompt_rules_*.md
# that the LLM consistently drops on NSFW content. Three passes:
#
#   1. Garment stripper: enforces the GARMENT BAN (#1 rule). Replaces
#      "[color] [garment]" with just "[color]" so the image model
#      preserves whatever the reference shows. Skips modifiers like
#      "removed"/"torn"/"unbuttoned" because those signal the garment
#      is being acted on (which the rule says IS allowed).
#
#   2. Narrative-filler stripper: removes phrases like "showing the
#      heat of the moment" that are emotion/narrative, not physical
#      mechanics. The image model can't render these and they waste
#      the prompt's attention budget.
#
#   3. Duplicate-descriptor dedupe (Pass 3 only — needs name_to_desc
#      map): when the same descriptor appears 2+ times, the 2nd+
#      occurrences become pronouns. Pass 3's LLM commonly does this
#      when it re-emphasizes character identity and ends up saying
#      "the woman in the white sweater on the left with red hair"
#      twice in 30 words.
#
# NOTE: deliberately does NOT strip embellishment trigger phrases
# (LoRA-specific descriptors). Those are sometimes legitimate
# LoRA invocations — Pass 3 sees LoRA hints intentionally and may
# invoke them on purpose. Banning would break valid LoRA usage.

_GARMENT_NOUNS = (
    "shirt", "t-shirt", "tshirt", "polo", "blouse", "sweater", "sweatshirt",
    "hoodie", "jacket", "coat", "blazer", "vest", "cardigan", "dress", "gown",
    "skirt", "pants", "slacks", "trousers", "jeans", "shorts", "leggings",
    "tights", "stockings", "lab coat", "scrubs", "uniform", "robe", "bodice",
    "corset", "bikini", "bra", "lingerie", "suit", "tie", "tuxedo",
)

# Match "[modifier] [garment]" — the modifier is usually a color or material.
# Multi-word garments like "lab coat" need the longest-match-first ordering,
# so we sort by length descending.
_GARMENT_RE = re.compile(
    r"\b([a-z][a-z\-]*)\s+(" +
    "|".join(re.escape(g) for g in sorted(_GARMENT_NOUNS, key=len, reverse=True)) +
    r")\b",
    re.IGNORECASE,
)

# Modifiers that signal the garment is being acted on (clothing change),
# which the rules say IS allowed to keep the garment word.
_GARMENT_KEEP_MODIFIERS = frozenset({
    "removed", "torn", "ripped", "unbuttoned", "open", "loose", "new",
    "fresh", "clean", "wet", "damp", "soaked", "stained",
})

# Verbs that signal a clothing-change scene — when one of these appears
# in the ~30 chars BEFORE a "[modifier] [garment]" match, the rules say
# we MUST keep the garment word so the model knows what's changing.
# Match up to ~5 words back for "she removes her white sweater" patterns.
_GARMENT_CHANGE_VERB_RE = re.compile(
    r"\b(remov\w*|tear\w*|tore|torn|rip\w*|pull\w*|strip\w*|undress\w*|"
    r"unbutton\w*|unzip\w*|put on|puts on|putting on|wear\w*|wore|worn|"
    r"slip\w* (on|off|out of|into)|change\w* into|dress\w* in)\b",
    re.IGNORECASE,
)

# Narrative-filler phrases the polish layer strips from image prompts —
# emotion/atmosphere language the image model can't render (observed Gemma 4B
# failure cases on mature content). Clinical cinematographic filler, so it's
# tracked in-repo (clean-by-construction).
_FILLER_PATTERNS_RAW = [
    r",?\s*showing the (heat|intensity|peak|throes) of (the )?moment[,.]?",
    r",?\s*at the peak of (her|his|their) (climax|passion|ecstasy)[,.]?",
    r",?\s*locked (intimately|passionately) (together|in embrace)[,.]?",
    r",?\s*in the throes of (passion|ecstasy)[,.]?",
    r",?\s*lost in (the|their) (moment|passion)[,.]?",
    r",?\s*in a moment of pure (passion|ecstasy|bliss)[,.]?",
    r",?\s*overwhelmed (by|with) (passion|desire|ecstasy)[,.]?",
]


@lru_cache(maxsize=1)
def _get_filler_patterns() -> list:
    """Compile the narrative-filler regex patterns (cached once per process)."""
    return [re.compile(p, re.IGNORECASE) for p in _FILLER_PATTERNS_RAW]


def _strip_garments(text: str) -> tuple[str, int]:
    """Replace '[modifier] [garment]' → '[modifier]'. Returns (cleaned, count).

    Two skip rules preserve garment words when the rules allow them:
      - Modifier in KEEP_MODIFIERS list ("removed", "torn", etc.).
      - Clothing-change verb in the ~30 chars before the match
        ("removes her white sweater" → keep "sweater").
    """
    counter = [0]

    def _sub(m: "re.Match[str]") -> str:
        modifier = m.group(1)
        if modifier.lower() in _GARMENT_KEEP_MODIFIERS:
            return m.group(0)
        # Look back ~30 chars for a clothing-change verb. If present,
        # this is a scene where clothing IS changing — the rules say
        # to keep the garment word so the model knows what's different.
        lookback_start = max(0, m.start() - 30)
        lookback = text[lookback_start:m.start()]
        if _GARMENT_CHANGE_VERB_RE.search(lookback):
            return m.group(0)
        counter[0] += 1
        return modifier

    cleaned = _GARMENT_RE.sub(_sub, text)
    return cleaned, counter[0]


def _strip_narrative_filler(text: str) -> tuple[str, int]:
    """Remove emotion/narrative phrases the image model can't render.

    Inserts a sentence boundary (". ") when the next non-whitespace
    character after the removed phrase is an uppercase letter — that
    signals a new sentence started, and the trailing period was eaten
    by the regex's "[,.]?" terminator. Otherwise plain removal.
    """
    count = 0
    for pat in _get_filler_patterns():
        def _resolve(m: "re.Match[str]", t: str = text) -> str:
            tail = t[m.end():].lstrip()
            if tail and tail[0].isalpha() and tail[0].isupper():
                return ". "
            return ""
        new_text, n = pat.subn(_resolve, text)
        if n:
            count += n
            text = new_text
    # Cleanup: collapse double-spaces, orphaned commas, double periods,
    # and orphan-period-then-space-then-period sequences.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"\s+([.,])", r"\1", text)
    text = re.sub(r"\.\s*\.", ".", text)
    return text.strip(), count


# Object-pronoun targets: after these prepositions, "she/he/they" becomes
# "her/him/them". List covers the common cases image-prompt narration uses.
_PREPOSITION_TAIL_RE = re.compile(
    r"\b(with|to|for|from|by|at|on|of|in|into|onto|toward|towards|"
    r"between|behind|beside|near|next to|alongside|beneath|under|"
    r"underneath|over|above|across|against|around|past)\s*$",
    re.IGNORECASE,
)


def _descriptor_pronoun(descriptor: str, *, object_form: bool = False) -> str:
    """Pick a pronoun based on gender keywords in the descriptor.

    object_form=True returns "her"/"him"/"them" for use after prepositions
    or as a verb's direct object.

    Uses word-boundary matching so the gender word is recognized whether
    it sits at the start of the descriptor ("woman in white") or in the
    middle ("the woman in white"). The earlier leading-space-only check
    silently fell through to "they"/"them" for descriptors that started
    with the gender word — the most common shape — defeating the entire
    point of gender-aware substitution.
    """
    if not descriptor:
        return "them" if object_form else "they"
    d = descriptor.lower()
    fem = (r"\bwoman\b", r"\bwomen\b", r"\bgirl\b", r"\bmother\b",
           r"\bsister\b", r"\bwife\b", r"\blady\b", r"\bqueen\b",
           r"\bdaughter\b", r"\bgirlfriend\b", r"\bbride\b", r"\bnun\b",
           r"\bfemale\b")
    masc = (r"\bman\b", r"\bmen\b", r"\bboy\b", r"\bfather\b",
            r"\bbrother\b", r"\bhusband\b", r"\bking\b", r"\bguy\b",
            r"\bson\b", r"\bboyfriend\b", r"\bgroom\b", r"\bmonk\b",
            r"\bmale\b", r"\bgentleman\b")
    for pat in fem:
        if re.search(pat, d):
            return "her" if object_form else "she"
    for pat in masc:
        if re.search(pat, d):
            return "him" if object_form else "he"
    return "them" if object_form else "they"


def _dedupe_descriptors(text: str, name_to_descriptor: dict) -> tuple[str, int]:
    """Replace 2nd+ occurrences of each descriptor with a pronoun.

    The first occurrence is kept (image model needs it to identify the
    character). Subsequent identical descriptors become "she"/"he"/"they"
    (or the object form "her"/"him"/"them" when following a preposition).

    The match optionally consumes a leading article ("the"/"a"/"an" + space)
    so that "the {descriptor}" in the source becomes the bare pronoun
    rather than "the he" / "the she". Article-pronoun pairs are wrong
    English regardless of source — there is no "the he" construction —
    so consuming the article is always the right behavior. The
    `_collapse_article_pronoun` safety net below catches any stragglers
    that come from other code paths.
    """
    if not name_to_descriptor:
        return text, 0
    count = 0
    for _name, desc in name_to_descriptor.items():
        if not desc or len(desc) < 12:  # Don't dedupe short / generic descriptors
            continue
        # Pattern: optional article + descriptor. Article goes into a
        # named group so we can detect it inside the loop without losing
        # the lazy-binding semantics.
        pat = re.compile(
            r"(?P<art>\b(?:the|a|an)\s+)?" + re.escape(desc),
            re.IGNORECASE,
        )
        matches = list(pat.finditer(text))
        if len(matches) < 2:
            continue
        # Replace from the back so earlier indices stay valid.
        for m in reversed(matches[1:]):
            # Look at the chars immediately before this match to decide
            # whether we need an object pronoun ("with her") or subject
            # pronoun ("she leans over"). Use m.start() (which points to
            # the article when present, or the descriptor when not) so
            # the preposition lookback sees the right preceding context.
            preceding = text[:m.start()]
            object_form = bool(_PREPOSITION_TAIL_RE.search(preceding))
            pronoun = _descriptor_pronoun(desc, object_form=object_form)
            text = text[:m.start()] + pronoun + text[m.end():]
            count += 1
    # Safety net: a different code path (name-substitution chain, polish
    # LLM output, etc.) may still produce "the he" / "a she" patterns
    # that this function's article-aware match didn't catch. Collapse
    # them unconditionally — there is no valid English "the he".
    text, _collapsed = _collapse_article_pronoun(text)
    return text, count


# Article + pronoun = always wrong English. "The he reaches" / "a she
# turns" are produced when name → descriptor → pronoun substitution
# chains run across different functions and one of them missed the
# leading article. Cheap regex pass to clean up.
_ARTICLE_PRONOUN_RE = re.compile(
    r"\b(the|a|an)\s+(he|she|they|him|her|them)\b",
    re.IGNORECASE,
)


# ── Speaker-name harvest (defense for Pass 2 dropping speaker_name) ─
# Pass 2's schema marks `subjects_on_screen[].speaker_name` as REQUIRED
# when the screenplay names a character, but in practice the LLM
# sometimes omits it — leaving names like "Blaine" baked into
# video_prompt narrative ("Blaine reaches for the gun...") with no
# entry in our name → descriptor map. Without that entry the polish
# layer can't substitute the name out, and the video model receives
# "Blaine" as if it were a known proper noun.
#
# These regexes scan the prompt text directly to recover any names
# Pass 2 dropped. Two patterns:
#   1. Speech tags: "Name says/said/replied/whispered/etc."
#   2. Vocatives inside quoted dialogue: "..., Name," / "..., Name."
# Both produce false positives if a regular noun gets capitalized at
# sentence start; the harvest step filters those by requiring the
# match be inside narrative prose (not a sentence start) and not in
# a small stop-list of common capitalized English words.
_SPEAKER_TAG_RE = re.compile(
    r"\b([A-Z][a-z]{2,})\s+"
    r"(?:says?|said|replies|replied|reply|asks?|asked|"
    r"shouts?|shouted|whispers?|whispered|murmurs?|murmured|"
    r"continues?|continued|adds?|added|exclaims?|exclaimed|"
    r"yells?|yelled|growls?|growled|snaps?|snapped|"
    r"cries|cried|laughs?|laughed|sighs?|sighed|"
    r"calls?|called|responds?|responded|answers?|answered|"
    r"mutters?|muttered|declares?|declared)\b"
)
_VOCATIVE_IN_QUOTE_RE = re.compile(
    r"['\"][^'\"]{2,}?,\s*([A-Z][a-z]{2,})\s*[,!?.]"
)
# Common capitalized English words / honorifics / sentence-start tokens
# we DO NOT want to harvest as a character name. The list is
# conservative — false positives here mean the harvest skips a
# legitimate name and the bug returns; false negatives just leave
# behind a junk entry that gets filtered out at substitution time
# (regex won't find these as standalone words in narrative).
_NAME_HARVEST_STOPLIST = frozenset({
    "The", "A", "An", "He", "She", "They", "Him", "Her", "Them",
    "His", "Hers", "Their", "Theirs", "It", "Its", "I", "We", "Us", "Our",
    "You", "Your", "Yours", "This", "That", "These", "Those",
    "And", "But", "Or", "So", "Yet", "For", "Nor",
    "Mr", "Mrs", "Ms", "Dr", "Sir", "Madam", "Lord", "Lady",
    "Yes", "No", "Maybe", "Okay", "Ok", "Sure", "Right", "Well",
    "Today", "Tomorrow", "Yesterday", "Now", "Then", "Here", "There",
    "Yeah", "Hey", "Oh", "Ah", "Um",
})


def _harvest_speaker_names(text: str) -> set[str]:
    """Return proper-noun candidates that look like character names.

    Scans speech-tag patterns ("X says", "X replied") and vocatives
    inside quoted dialogue (",X,"). Filters out common capitalized
    English words via _NAME_HARVEST_STOPLIST. Returns an empty set
    on empty input.

    The returned names are RAW — caller must still match each to a
    known character via context (typically via a sole unnamed subject
    on screen) before adding to the substitution map.
    """
    if not text or not text.strip():
        return set()
    names: set[str] = set()
    for m in _SPEAKER_TAG_RE.finditer(text):
        cand = m.group(1)
        if cand not in _NAME_HARVEST_STOPLIST:
            names.add(cand)
    for m in _VOCATIVE_IN_QUOTE_RE.finditer(text):
        cand = m.group(1)
        if cand not in _NAME_HARVEST_STOPLIST:
            names.add(cand)
    return names


def _collapse_article_pronoun(text: str) -> tuple[str, int]:
    """Collapse 'the he' / 'a she' / 'an they' → just the pronoun.

    Runs as the final pass of `_dedupe_descriptors` and is also safe to
    call from anywhere that does name/descriptor substitution. Keeps
    the pronoun and drops the leading article — the article was wrong
    by definition (English doesn't put articles before pronouns).

    Returns (cleaned_text, num_collapsed).
    """
    if not text:
        return text, 0
    count = 0

    def _sub(m):
        nonlocal count
        count += 1
        return m.group(2)  # keep the pronoun

    return _ARTICLE_PRONOUN_RE.sub(_sub, text), count


# Match any capitalized proper-noun-looking token (length >= 4 to avoid
# common 3-letter capitalized words like "And", "But", "For").
_PROPER_NOUN_TOKEN_RE = re.compile(r"\b([A-Z][a-z]{3,})\b")


# ── Storyboard format detector ────────────────────────────────────
# When Multi-Shot LoRA Mode is on, Pass 2 emits Format B storyboard
# content inside video_prompts and window_prompts, structured as a
# series of "Shot N (CameraType, Xs): description" blocks. The
# IC-LoRA was TRAINED on this exact literal structure — internal
# camera cuts only render when this structure is preserved verbatim.
#
# Pass 3's polish LLM, given a generic "refine this prompt" task,
# sees the structured blocks and "fixes" them into flowing prose
# ("A close-up focuses on..." "The camera shifts to..."), destroying
# the structure the LoRA needs. The IC-LoRA then produces zero
# internal cuts because the trained trigger pattern is gone.
#
# Detection: look for the literal `Shot N (...,Xs):` pattern. At
# least one match strongly indicates Format B. Detection is robust
# to the `multishot` flag being missing or wrong — we go by what's
# in the prompt text itself.
_STORYBOARD_SHOT_RE = re.compile(
    r"\bShot\s+\d+\s*\([^)]*?,\s*\d+\s*s\)\s*:",
    re.IGNORECASE,
)


def _is_storyboard_format(text: str) -> bool:
    """Return True when `text` contains at least one storyboard
    `Shot N (Camera, Xs):` block. Used by Pass 3 polish to skip
    storyboard-formatted prompts (the polish LLM would otherwise
    rewrite them as flowing prose and break the IC-LoRA trigger).
    """
    if not text:
        return False
    return bool(_STORYBOARD_SHOT_RE.search(text))


# Match "Shot N (CameraType on Name, Xs):" / "of Name" / "from Name" /
# "with Name" patterns and capture the relational preposition + name
# portion so we can strip it. The camera-type tokens the IC-LoRA was
# trained on don't contain character references; "Close-up on Henry"
# breaks the trained pattern.
_STORYBOARD_CAMERA_NAME_LEAK_RE = re.compile(
    r"(Shot\s+\d+\s*\(\s*[A-Za-z][A-Za-z\s\-]+?)\s+(?:on|of|from|with|over)\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*(,\s*\d+\s*s\s*\))",
    re.IGNORECASE,
)


# Known sex-act leet trigger tokens from popular NSFW video LoRAs.
# These appear in `trainedWords` sidecar fields and the LoRA-hint
# system surfaces them in the LLM system prompt. Smaller LLMs and
# even Qwen-class models sometimes pepper these into non-sex-act
# scenes for "energy" — observed production bug: a SFW music video
# about football had "bl0wj0b" in a keyframe_prompt because the
# LLM treated leet triggers as generic energy boosters.
#
# Detection is by EXACT TOKEN match (case-insensitive), not a regex
# pattern like "letter+digit" — that would false-positive on legit
# tokens like RTX4090, model names, scientific terms, etc. The list
# below is intentionally complete-as-known and easy to extend when
# new sex-act LoRAs ship.
_SEX_ACT_LEET_TOKENS = (
    "bl0wj0b", "bl0w_j0b", "bj0b",
    "m15510n4ry", "m1ss10n4ry", "m1ssi0nary",
    "c0wg1rl", "c0wgrl", "c0w_g1rl",
    "r3v3rs3_c0wg1rl", "rc0wg1rl",
    "d0gg1e", "d0gg1e_style", "d0ggy",
    "0r4l", "0r4l_s3x",
    "h4ndj0b", "hj0b",
    "f1ng3r1ng", "f1nger1ng",
    "r1m_j0b", "r1mj0b",
    "p3n3tr4t10n", "p3n3tr4te",
    "h4rdc0r3", "s0ftc0r3",
    "n5fw", "3xpl1c1t",
    "th1cc",
    "0n_t0p", "fr0m_b3h1nd",
)


def strip_sex_act_leet_tokens(text: str) -> tuple[str, int]:
    """Strip sex-act leet-coded LoRA triggers from non-sex-act prompts.

    Catches the literal token in multiple wrapping contexts:
      - "(c0wg1rl)" in parens (storyboard-mimic format)
      - "bl0wj0b — description" with em-dash prefix
      - "bl0wj0b. The man..." starting a sentence
      - "bl0wj0b The man..." standalone

    Used as a defense-in-depth safety net. The leet warning in the
    LoRA-hint system prompt tells the LLM not to do this, but smaller
    LLMs ignore it and drop the tokens into keyframes / image prompts
    for "energy." Stripping after the LLM removes the embarrassing
    leak before it reaches downstream models or the user.

    Returns (cleaned_text, num_stripped).
    """
    if not text:
        return text, 0
    count = 0
    for token in _SEX_ACT_LEET_TOKENS:
        esc = re.escape(token)
        # Try in order of most-specific-first so we strip the trailing
        # punctuation along with the token.
        for pat_src in (
            r"\(\s*" + esc + r"\s*\)\s*[—–-]*\s*",  # (c0wg1rl) —
            r"\(\s*" + esc + r"\s*\)\s*",                       # (c0wg1rl)
            r"\b" + esc + r"\s+[—–-]+\s*",            # c0wg1rl —
            r"\b" + esc + r"\s*\.\s+",                          # c0wg1rl. (start of next sentence)
            r"\b" + esc + r"\s*\.",                             # c0wg1rl.
            r"\b" + esc + r"\b\s*",                             # c0wg1rl (catch-all)
        ):
            pat = re.compile(pat_src, re.IGNORECASE)
            new_text, n = pat.subn("", text)
            if n:
                text = new_text
                count += n
                break  # don't double-strip the same token
    # Collapse any orphaned multi-space the strip created
    if count:
        text = re.sub(r"\s{2,}", " ", text).strip()
    return text, count


def strip_storyboard_camera_name_leaks(text: str) -> tuple[str, int]:
    """Strip character-name leaks from storyboard camera-type parens.

    Catches LLM outputs like "Shot 2 (Close-up on Henry, 7s):" and
    rewrites them as "Shot 2 (Close-up, 7s):". The IC-LoRA was trained
    on clean camera-type tokens; names in the parens break the trained
    pattern and the LoRA doesn't render the cut.

    Cases caught: "on Henry", "of Mary", "from Mary", "with Henry",
    "over Mary's shoulder" → all stripped.

    Returns (cleaned, count_stripped).
    """
    if not text or "Shot" not in text:
        return text, 0
    count = 0

    def _sub(m):
        nonlocal count
        count += 1
        # m.group(1) = "Shot N (CameraType"
        # m.group(2) = ", Xs)"
        return f"{m.group(1).rstrip()}{m.group(2)}"

    cleaned = _STORYBOARD_CAMERA_NAME_LEAK_RE.sub(_sub, text)
    return cleaned, count


def _strip_hallucinated_names(
    input_text: str,
    output_text: str,
    name_to_descriptor: dict,
    fallback_descriptors: list,
) -> tuple[str, list[str]]:
    """Remove polish-LLM-invented character names from output_text.

    A "hallucinated name" is any proper-noun token in output_text that
    is NOT in input_text AND NOT in name_to_descriptor AND NOT a
    common capitalized English word. The polish LLM occasionally
    invents these even when the system prompt forbids it — typically
    because it saw a similar character in training and "filled in" a
    name for a generic descriptor.

    Substitution strategy (in order of preference):
      - "the {Name}'s {noun}" → "their {noun}" (gender-neutral)
      - "{Name}'s {noun}"     → "their {noun}"
      - "the {Name} ..."      → first matching fallback descriptor
      - bare "{Name} ..."     → first matching fallback descriptor

    Conservative: returns the input unchanged when no fallback
    descriptor is available (cleaner to leave the name than substitute
    it for nothing). Logs each hallucination for debugging.

    Args:
        input_text: The polish input prompt (pre-LLM).
        output_text: The polish output prompt (post-LLM).
        name_to_descriptor: Known character name → descriptor map.
        fallback_descriptors: List of descriptor strings to use when
            substituting a hallucinated name. Typically the on-screen
            subjects' visual_descriptions for this shot.

    Returns:
        (cleaned_text, list_of_hallucinated_names_found).
    """
    if not output_text:
        return output_text, []

    in_input_lower = {t.lower() for t in _PROPER_NOUN_TOKEN_RE.findall(input_text or "")}
    known_names_lower = {n.lower() for n in name_to_descriptor.keys() if n}
    stoplist_lower = {s.lower() for s in _NAME_HARVEST_STOPLIST}

    out_tokens = _PROPER_NOUN_TOKEN_RE.findall(output_text)
    hallucinated: list[str] = []
    seen: set[str] = set()
    for tok in out_tokens:
        tl = tok.lower()
        if tl in seen:
            continue
        seen.add(tl)
        if tl in in_input_lower:
            continue
        if tl in known_names_lower:
            continue
        if tl in stoplist_lower:
            continue
        hallucinated.append(tok)

    if not hallucinated:
        return output_text, []

    text = output_text
    fallback = (fallback_descriptors[0] if fallback_descriptors else None)

    for name in hallucinated:
        # Possessive forms first (most specific) → gender-neutral pronoun
        text = re.sub(
            r"\bthe\s+" + re.escape(name) + r"'s\b",
            "their",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b" + re.escape(name) + r"'s\b",
            "their",
            text,
            flags=re.IGNORECASE,
        )
        if fallback:
            # "the {Name}" → fallback descriptor (descriptor already has
            # its own article semantics; no double-the because we strip
            # the leading "the ")
            text = re.sub(
                r"\bthe\s+" + re.escape(name) + r"\b",
                fallback,
                text,
                flags=re.IGNORECASE,
            )
            # Bare "{Name}" → "the {fallback}" (in narrative position
            # the descriptor needs an article).
            text = re.sub(
                r"\b" + re.escape(name) + r"\b",
                f"the {fallback}",
                text,
                flags=re.IGNORECASE,
            )
        # else: leave the name as-is. Better than substituting with
        # nothing (which would corrupt the sentence) or with a guess
        # we can't justify.

    return text, hallucinated


# Speech-verb dictionary used by _strip_dialogue. Covers the common
# constructions a polish LLM hallucinates into image_prompts when it
# pulls dialogue from screenplay context. Smaller models (Gemma 4 4B
# observed) are most prone to this; bigger ones (Qwen 9B Claude Opus)
# generally don't, but the strip runs unconditionally as a safety net.
_SPEECH_VERBS_RE = re.compile(
    r"\b(?:says?|said|replies|replied|reply|shouts?|shouted|whispers?|whispered|"
    r"asks?|asked|continues?|continued|adds?|added|murmurs?|murmured|exclaims?|"
    r"exclaimed|yells?|yelled|declares?|declared|mutters?|muttered|calls?|called|"
    r"cries|cried|growls?|growled|snaps?|snapped|barks?|barked|responds?|responded|"
    r"answers?|answered|laughs?|laughed|pleads?|pleaded|interjects?|interjected|"
    r"insists?|insisted|begs?|begged|sighs?|sighed|chuckles?|chuckled)\b",
    re.IGNORECASE,
)
# Matches any double- OR single-quoted span (lazy).
_QUOTED_SPAN_RE = re.compile(r"['\"][^'\"]*['\"]")


def _strip_dialogue(text: str) -> tuple[str, int]:
    """Drop any sentence that contains BOTH a speech verb AND a quoted span.

    image_prompt is for a SINGLE STILL FRAME — there's no audio, no time,
    no spoken words. But polish LLMs (especially smaller models like
    Gemma 4 4B that have seen screenplay context for the surrounding
    shots) sometimes "embellish" by adding dialogue:

      Input:   "A close-up of Ember braced against the brambles."
      Polished: "A close-up of the orange dragon braced against the
                brambles. Ember says 'Variables are fun, Milo.' Milo
                replies, 'They are, but this bramble is decidedly not.'"

    The dialogue is uniquely identifiable: it sits in a sentence that
    has a speech verb (says/replies/etc) and quoted text. Drop the
    entire sentence — leaving fragments of "Ember says" without the
    quote behind would make the prompt grammatically broken.

    Trade-off (acknowledged): a legitimate prompt like 'A storefront
    sign says "Welcome"' would also get dropped. Risk is low because
    the Director's planner doesn't write that kind of prose, and the
    "no dialogue" rule in the polish system prompt means the LLM
    shouldn't be adding text-on-objects either. If false positives
    show up in practice, we can tighten the subject to require a
    capitalized name / pronoun / character descriptor.

    Returns (cleaned_text, num_sentences_dropped).
    """
    if not text or not text.strip():
        return text, 0
    # Split into sentences. The boundary is sentence-ending punctuation
    # `[.!?]`, optionally followed by a closing quote `["']`, then
    # whitespace. Critically, the closing-quote case is the only way
    # to correctly split dialogue like `"Milo." they replies "..."` —
    # the period is INSIDE the quote there, so a naive split on
    # `[.!?]\s` would never trigger and the dialogue collapses into
    # one super-sentence with the next legitimate prose. Python doesn't
    # support variable-width lookbehind so we use finditer on the full
    # boundary pattern and slice manually.
    sentences = []
    boundary_re = re.compile(r"[.!?][\"']?\s+")
    cursor = 0
    for m in boundary_re.finditer(text):
        seg = text[cursor:m.end()].strip()
        if seg:
            sentences.append(seg)
        cursor = m.end()
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            sentences.append(tail)
    kept = []
    dropped = 0
    for s in sentences:
        if _SPEECH_VERBS_RE.search(s) and _QUOTED_SPAN_RE.search(s):
            dropped += 1
            continue
        kept.append(s)
    if dropped == 0:
        return text, 0
    cleaned = " ".join(kept).strip()
    # Collapse any double-spaces left behind by the strip
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned, dropped


# ── B: motion-verb sanitizer for image prompts ──────────────────────
# Image prompts are FROZEN frames. Sentences containing motion-implying
# progressive verbs ("rising", "thrusting", "bobbing", etc.) describe
# action over time and have no business in a still-frame prompt. Pass 2
# rules forbid this but Pass 2 emits motion verbs anyway. Drop offending
# sentences entirely. Static -ing words ("standing", "leaning",
# "wearing", "holding") describe poses/states and are fine — they are
# explicitly excluded from the regex below.
_MOTION_VERBS_RE = re.compile(
    r"\b("
    # Locomotion
    r"walking|running|jogging|sprinting|stepping|striding|"
    # Body movement / oscillation
    r"rising|falling|bobbing|swaying|rocking|bouncing|jumping|leaping|"
    r"spinning|twisting|rotating|turning|tilting|nodding|shaking|trembling|"
    # Sex acts (motion-form)
    r"thrusting|grinding|pumping|stroking|riding|"
    # Gestures (durational)
    r"reaching|grabbing|gesturing|pointing|waving|swinging|"
    # Locomotion variants
    r"approaching|departing|arriving|fleeing|chasing|retreating|advancing|"
    # Body fluid / breath progressives that imply motion
    r"breathing|panting|gasping|moaning|crying|laughing|"
    # Generic
    r"moving|shifting|coming|going"
    r")\b",
    re.IGNORECASE,
)


def _strip_motion_sentences(text: str) -> tuple[str, int]:
    """Drop any sentence containing a motion-implying verb.

    Image prompts must describe a single frozen instant. Motion verbs
    describe what happens OVER TIME, which the image model can't render.
    The strip is sentence-level (drops the entire sentence) rather than
    surgical because a half-rewritten sentence ("the woman was" with the
    verb removed) reads worse than no sentence at all.

    Returns (cleaned_text, num_sentences_dropped).
    """
    if not text or not text.strip():
        return text, 0
    boundary_re = re.compile(r"[.!?][\"']?\s+")
    sentences = []
    cursor = 0
    for m in boundary_re.finditer(text):
        seg = text[cursor:m.end()].strip()
        if seg:
            sentences.append(seg)
        cursor = m.end()
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            sentences.append(tail)
    kept = []
    dropped = 0
    for s in sentences:
        if _MOTION_VERBS_RE.search(s):
            dropped += 1
            continue
        kept.append(s)
    if dropped == 0:
        return text, 0
    cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned, dropped


# ── C: orphan-fragment cleanup before the boilerplate suffix ────────
# Image prompts always end with the boilerplate "Preserve character
# identity, attire, and body attributes from the reference image." If
# polish leaves a sentence-fragment immediately before that suffix
# (e.g. "...torso. large pendulous breasts Preserve character..."),
# the fragment is orphaned — not part of any complete sentence. Drop
# everything between the last sentence-ending punctuation and the
# boilerplate suffix.
_PRESERVE_BOILERPLATE_RE = re.compile(
    r"(.*?)(\s*Preserve character identity[^.]*\.?)\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _strip_orphan_before_boilerplate(text: str) -> tuple[str, int]:
    """If the boilerplate suffix is preceded by a fragment (no sentence-
    ending punctuation), drop the fragment.

    Returns (cleaned_text, num_fragments_dropped).
    """
    if not text or not text.strip():
        return text, 0
    m = _PRESERVE_BOILERPLATE_RE.search(text)
    if not m:
        return text, 0
    before = m.group(1).rstrip()
    boilerplate = m.group(2).strip()
    if not before:
        return text, 0
    if before[-1] in ".!?":
        return text, 0  # already properly terminated, nothing orphaned
    # Find the last sentence-ending punctuation in `before`
    last_punct = max(before.rfind('.'), before.rfind('!'), before.rfind('?'))
    if last_punct < 0:
        # No punctuation at all — entire text before boilerplate is a
        # fragment. Conservative: leave it (might be a single legitimate
        # noun phrase) and let the boilerplate get appended cleanly.
        return text, 0
    kept = before[:last_punct + 1]
    return f"{kept} {boilerplate}", 1


# ── D: rewrite long-descriptor possessives ──────────────────────────
# A 6-word descriptor used possessively reads catastrophically ("the
# woman in middle with massive breasts' chest"). Convert to "the
# {noun} of the {descriptor}" form when the descriptor before "'s" is
# 4+ words. Handles BOTH "the X's Y" and bare "X's Y" forms — the
# original regex required a leading "the", missing cases like
# "woman in white with huge breasts's lower stomach" where the polish
# LLM dropped the article.
#
# Group 1: the descriptor (4+ words after substitution check).
# Group 2: the head noun.
# Group 3: optional second word of a 2-word noun phrase (e.g. "lower
#   stomach", "right hand", "white sweater"). The callback decides
#   whether to fold it into the noun phrase based on a verb blocklist —
#   "X's hand reaches" must NOT capture "reaches" into the noun.
_LONG_POSSESSIVE_RE = re.compile(
    r"(?:\bthe\s+)?([\w][\w\s\-]{8,80}?)'s\s+(\w+)(\s+\w+)?",
    re.IGNORECASE,
)

# Words that, if captured as the second word of a 2-word noun phrase,
# are actually the next clause's verb. Trim them out of the noun phrase
# and append after the rewrite. Conservative — list only common verbs;
# missing entries fall back to keeping a 2-word noun phrase, which
# usually still reads fine ("the lower stomach of the X" is grammatical
# even if "lower" was actually a stray adverb).
_POSSESSIVE_TRIM_VERBS = frozenset({
    # Body action / state verbs
    "reaches", "lifts", "presses", "twitches", "tightens", "loosens",
    "moves", "shifts", "drops", "rises", "falls", "turns", "tilts",
    "trembles", "shakes", "shudders", "quivers", "clenches", "relaxes",
    "opens", "closes", "looks", "stares", "glances", "grips", "releases",
    "rests", "lays", "sits", "stands", "leans", "bends", "stretches",
    "appears", "becomes", "remains", "stays", "feels", "seems",
    "grabs", "pulls", "pushes", "slides", "rolls", "spins",
    # Linking verbs
    "is", "was", "are", "were", "be", "been", "being",
    # Common -ing forms (could appear in present-progressive narrative)
    "reaching", "lifting", "pressing", "twitching", "tightening",
    "shifting", "rising", "falling", "turning", "tilting",
    "trembling", "clenching", "leaning", "bending", "stretching",
    "moving", "looking", "staring", "resting", "standing",
})


def _rewrite_long_possessives(text: str) -> tuple[str, int]:
    """Rewrite "(the )?{long descriptor}'s {noun}" → "the {noun} of the {descriptor}".

    Only fires when the descriptor before "'s" is 4+ words — short
    possessives ("the man's hand") read fine. Longer ones ("the
    woman in middle with massive breasts' chest") become unparseable.

    Both leading-article and bare-possessive forms are handled. The
    head-noun capture greedily takes a second word when one is present
    and unambiguously part of the noun phrase (not a verb in the
    blocklist) — this fixes cases like "X's lower stomach" that the
    single-word-noun regex would have left dangling.

    Returns (cleaned_text, num_rewrites).
    """
    if not text or not text.strip():
        return text, 0
    rewrites = 0

    def _sub(m):
        nonlocal rewrites
        descriptor = m.group(1).strip()
        noun = m.group(2)
        extra_raw = m.group(3) or ""  # includes leading whitespace
        extra = extra_raw.strip()
        if len(descriptor.split()) < 4:
            return m.group(0)
        # Decide whether the trailing word belongs to the noun phrase or
        # is the next clause's verb. Verbs get pulled out and appended
        # AFTER the rewritten possessive so the sentence still parses.
        if extra and extra.lower() not in _POSSESSIVE_TRIM_VERBS:
            noun_phrase = f"{noun} {extra}"
            tail = ""
        else:
            noun_phrase = noun
            tail = extra_raw if extra else ""  # preserves the leading space
        rewrites += 1
        return f"the {noun_phrase} of the {descriptor}{tail}"

    cleaned = _LONG_POSSESSIVE_RE.sub(_sub, text)
    return cleaned, rewrites


# ── E: collapse hallucinated double articles ────────────────────────
# When a name → descriptor substitution runs and the prompt contained
# "the Blaine" (polish LLM occasionally hallucinates an article before
# a name), the substitution produces "the the man in cowboy hat".
# Detect and collapse double articles into one.
_DOUBLE_ARTICLE_RE = re.compile(
    r"\b(the|a|an)\s+(the|a|an)\b",
    re.IGNORECASE,
)


def _collapse_double_articles(text: str) -> tuple[str, int]:
    """Collapse "the the X" / "a the X" / "the a X" → "the X" / "a X" / "the X".

    Cheap regex pass that runs after name → descriptor substitution to
    fix the article-doubling hallucination. The second article wins
    because if the descriptor is "the man in cowboy hat", the leading
    "the" is intrinsic to the substitution and should be kept.

    Returns (cleaned_text, num_collapsed).
    """
    if not text or not text.strip():
        return text, 0
    count = 0

    def _sub(m):
        nonlocal count
        count += 1
        return m.group(2)  # keep the second article

    cleaned = _DOUBLE_ARTICLE_RE.sub(_sub, text)
    return cleaned, count


def sanitize_image_prompt(
    text: str,
    name_to_descriptor: Optional[dict] = None,
    *,
    log_prefix: str = "[ImagePromptSanitize]",
) -> str:
    """Apply Layer 1 deterministic cleanup to an image_prompt.

    Run after Pass 2 produces the image_prompt and again after Pass 3
    polish. Dedupe pass is skipped if name_to_descriptor is None.

    Args:
        text: The image_prompt to sanitize.
        name_to_descriptor: Map of character name → canonical descriptor,
            used for the dedupe pass. Pass None to skip dedupe (typical
            for Pass 2 sanitization where the map isn't readily available).
        log_prefix: Prefix for the diagnostic print emitted when changes
            are made. Defaults to a generic tag — Pass 2 and Pass 3 call
            sites override with stage-specific tags.

    Returns:
        The cleaned image_prompt. Always returns a string (returns input
        unchanged if it's empty or no changes are needed).
    """
    if not text or not text.strip():
        return text
    # Strip any hallucinated dialogue FIRST — before the other cleanup
    # passes run, so they don't waste cycles deduping descriptors inside
    # speech tags we're about to delete.
    cleaned, dialogue_dropped = _strip_dialogue(text)
    cleaned, garments = _strip_garments(cleaned)
    cleaned, fillers = _strip_narrative_filler(cleaned)
    # Motion-verb sanitizer (B): drop sentences containing motion verbs
    # like "rising", "thrusting", "bobbing". Image prompts must be
    # frozen frames; motion verbs are nonsense in a still.
    cleaned, motion_dropped = _strip_motion_sentences(cleaned)
    # Long-possessive rewrite (D): "the {6-word descriptor}'s chest" →
    # "the chest of the {descriptor}". Reads less awkwardly when the
    # descriptor is too long for natural English possession.
    cleaned, long_poss = _rewrite_long_possessives(cleaned)
    deduped = 0
    if name_to_descriptor:
        cleaned, deduped = _dedupe_descriptors(cleaned, name_to_descriptor)
    # Double-article collapse (E): runs AFTER dedupe in case dedupe
    # introduced a new article-substitution pattern.
    cleaned, double_articles = _collapse_double_articles(cleaned)
    # Orphan-fragment cleanup (C) goes LAST so it sees the final shape
    # of the prompt — any other passes might have removed sentences and
    # changed where the boilerplate suffix sits.
    cleaned, orphans = _strip_orphan_before_boilerplate(cleaned)
    if (dialogue_dropped or garments or fillers or deduped
            or motion_dropped or long_poss or double_articles or orphans):
        notes = []
        if dialogue_dropped:
            notes.append(f"{dialogue_dropped} dialogue sentence(s) stripped")
        if garments:
            notes.append(f"{garments} garment(s) stripped")
        if fillers:
            notes.append(f"{fillers} filler(s) removed")
        if motion_dropped:
            notes.append(f"{motion_dropped} motion sentence(s) stripped")
        if long_poss:
            notes.append(f"{long_poss} long possessive(s) rewritten")
        if deduped:
            notes.append(f"{deduped} descriptor(s) deduped")
        if double_articles:
            notes.append(f"{double_articles} double-article(s) collapsed")
        if orphans:
            notes.append(f"{orphans} orphan fragment(s) dropped")
        print(f"{log_prefix} {', '.join(notes)}")
    return cleaned



# Map architecture prefixes to guide filenames (mirrors enhance_guides.py)
_VIDEO_ARCH_MAP = {
    "ltx2": "ltx2_video",
    "ltxv": "ltx2_video",
    "t2v": "wan_video",
    "i2v": "wan_video",
    "ti2v": "wan_video",
    "animate": "wan_video",
    "wanmove": "wan_video",
    "ovi": "wan_video",
    "lucy": "wan_video",
    "multitalk": "wan_video",
    "phantom": "wan_video",
    "fun_inp": "wan_video",
    "alpha": "wan_video",
    "fantasy": "wan_video",
    "chrono": "wan_video",
    "flf2v": "wan_video",
    "hunyuan": "wan_video",
    "heartmula": "wan_video",
}

_IMAGE_ARCH_MAP = {
    "qwen_image_edit": "qwen_image_edit",
    "qwen_image_layered": "qwen_image_edit",
    "qwen_image": "qwen_image_edit",
    "flux": "flux_image",
    "z_image": "flux_image",
}


def _match_arch(model_type: str, arch_map: dict) -> str:
    """Find the longest matching architecture prefix."""
    model_lower = model_type.lower()
    best_key = ""
    for prefix in arch_map:
        if model_lower.startswith(prefix) and len(prefix) > len(best_key):
            best_key = prefix
    return arch_map.get(best_key, "")


def _load_file(filepath: str) -> Optional[str]:
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def get_video_guide(video_model: str, mode: str = "full") -> str:
    """Get the video prompt guide for a model.

    Args:
        video_model: Model type string (e.g., "ltx2_22B_distilled")
        mode: "full" for complete enhance guide, "light" for dialect cheat sheet

    Returns:
        Guide text or empty string.
    """
    base = _match_arch(video_model, _VIDEO_ARCH_MAP) or "generic_video"
    if mode == "full":
        guide = _load_file(os.path.join(_ENHANCE_DIR, f"{base}.md"))
    else:
        guide = _load_file(os.path.join(_DIALECT_DIR, f"{base}.md"))
    if guide:
        print(f"[PromptPolish] Loaded {mode} video guide: {base}.md")
    return guide or ""


def get_image_guide(image_model: str, mode: str = "full") -> str:
    """Get the image prompt guide for a model.

    Args:
        image_model: Model type string (e.g., "qwen_image_edit_plus_20B")
        mode: "full" for complete enhance guide, "light" for dialect cheat sheet

    Returns:
        Guide text or empty string.
    """
    base = _match_arch(image_model, _IMAGE_ARCH_MAP) or "generic_image"
    if mode == "full":
        guide = _load_file(os.path.join(_ENHANCE_DIR, f"{base}.md"))
    else:
        guide = _load_file(os.path.join(_DIALECT_DIR, f"{base}.md"))
    if guide:
        print(f"[PromptPolish] Loaded {mode} image guide: {base}.md")
    return guide or ""


def load_lora_guides(video_loras: list[str] = None, image_loras: list[str] = None,
                     video_model: str = "", image_model: str = "") -> str:
    """Load trigger words for activated LoRAs (pass 1-2 injection).

    ONLY includes LoRAs that have real trainedWords in their .civitai.json
    sidecar. LoRAs without sidecar triggers are silently skipped — their
    .guide.md prose was causing the LLM to insert literal "trigger word
    for LoRA" placeholder text into prompts.
    """
    import json as _json
    trigger_lines = []

    for loras, model_type, label in [
        (video_loras, video_model, "Video"),
        (image_loras, image_model, "Image"),
    ]:
        if not loras or not model_type:
            continue
        try:
            import wgp
            lora_dir = wgp.get_lora_dir(model_type)
        except Exception:
            continue
        for lora_name in loras:
            # Only use trainedWords from the CivitAI sidecar
            sidecar_path = os.path.join(lora_dir, os.path.splitext(lora_name)[0] + ".civitai.json")
            trigger_words = []
            if os.path.isfile(sidecar_path):
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as sf:
                        sidecar = _json.loads(sf.read())
                    trigger_words = sidecar.get("trainedWords", []) or []
                except Exception:
                    pass
            if trigger_words:
                trigger_lines.append(f"- {label}: {', '.join(trigger_words[:5])}")

    if not trigger_lines:
        return ""

    any_leet = any(any(c.isdigit() for c in ln) for ln in trigger_lines)
    # Stronger leet block. The previous version ("Copy them EXACTLY as
    # written") only explained the encoding, not the appropriate-use
    # rule. Smaller LLMs dropped leet triggers into non-sex-act scenes
    # for "energy" — observed: a SFW football music video had
    # "bl0wj0b" in a keyframe prompt. The new block forbids leet
    # triggers in any context that isn't an explicit sex-act, with
    # specific examples of what they each mean.
    leet_block = (
        "\n\nLEET-CODED TRIGGER WORDS (contain digits like 4/0/3/5/1) — "
        "STRICT USAGE RULES:\n"
        "These trigger words name SPECIFIC SEX ACTS or POSITIONS. They are "
        "NSFW-only. Common ones include:\n"
        "  bl0wj0b   = oral sex (giving/receiving)\n"
        "  m15510n4ry = missionary position intercourse\n"
        "  c0wg1rl    = woman-on-top straddle position\n"
        "  r3v3rs3_c0wg1rl = reverse cowgirl\n"
        "  d0gg1e     = doggy-style intercourse\n"
        "\n"
        "INCLUDE a leet trigger ONLY when the video_prompt's scene LITERALLY "
        "DEPICTS that specific act. Use it ONCE per video_prompt — never "
        "stack multiple leet triggers in one prompt.\n\n"
        "NEVER use leet triggers in:\n"
        "  - image_prompt or keyframe_prompts (these are still images; "
        "    leet triggers are VIDEO LoRA triggers and don't apply)\n"
        "  - SFW content of any kind (music video, dialogue, action, "
        "    cooking, drama — anywhere that isn't a sex act)\n"
        "  - Non-act NSFW content (foreplay before intercourse, "
        "    undressing, kissing, post-coital aftermath — these are NOT "
        "    sex acts in themselves)\n"
        "  - As decoration / energy boosters / 'flavor' tokens\n\n"
        "If you find yourself wanting to add a leet trigger to a "
        "scene that doesn't literally depict the named act, the answer "
        "is to OMIT IT. No leet trigger is better than the wrong one — "
        "wrong placement breaks the LoRA's trained pattern and produces "
        "weaker results in BOTH the misplaced scene AND the actual sex "
        "scenes that needed the trigger."
    ) if any_leet else ""

    print(f"[PromptPolish] Loaded {len(trigger_lines)} LoRA trigger(s) for pass 1-2")
    return (
        "\n\nACTIVE LORA TRIGGER WORDS — include the most relevant trigger "
        "naturally in video_prompt when it matches the scene content. "
        "Do NOT force triggers into image_prompt or keyframe_prompts. "
        "Do NOT include a trigger that does not match the scene. "
        "Do NOT invent new trigger words or write placeholder text like "
        "'trigger word for LoRA'." + leet_block + "\n"
        + "\n".join(trigger_lines)
    )


def build_polish_block(video_model: str, image_model: str, mode: str,
                       video_loras: list[str] = None, image_loras: list[str] = None) -> str:
    """Build a combined guide block to inject into a planner system prompt.

    Args:
        video_model: Video model type
        image_model: Image model type
        mode: "full" or "light"
        video_loras: Activated video LoRA filenames (for guide loading)
        image_loras: Activated image LoRA filenames (for guide loading)

    Returns:
        Combined guide block string, or empty string if no guides found.
    """
    parts = []
    vg = get_video_guide(video_model, mode)
    if vg:
        parts.append(vg)
    ig = get_image_guide(image_model, mode)
    if ig:
        parts.append(ig)

    # Add LoRA guides
    lora_block = load_lora_guides(video_loras, image_loras, video_model, image_model)
    if lora_block:
        parts.append(lora_block)

    if not parts:
        return ""

    label = "MODEL-SPECIFIC PROMPTING GUIDES" if mode == "full" else "MODEL DIALECT RULES"
    return f"\n{label} — Apply these rules to the prompts you write:\n\n" + "\n\n".join(parts)


def polish_prompts_third_pass(
    clip_plans: list[dict],
    video_model: str,
    image_model: str,
    nsfw: bool = False,
    video_loras: list[str] = None,
    image_loras: list[str] = None,
    image_paths: list[str] = None,
    characters: list = None,
) -> list[dict]:
    """Post-process clip plans through the enhance LLM pipeline (third pass).

    Runs each video_prompt and image_prompt through the Studio enhance flow
    with model-specific guides, preserving Director structure.

    Args:
        clip_plans: List of clip plan dicts with video_prompt/image_prompt
        video_model: Video model type for guide selection
        image_model: Image model type for guide selection
        nsfw: Whether NSFW mode is enabled
        video_loras: Activated video LoRA filenames (for guide context)
        image_loras: Activated image LoRA filenames (for guide context)
        image_paths: Reference image paths. Currently UNUSED — was previously
            forwarded to image polish calls but caused the polish LLM to
            "correct" future-scene image_prompts toward content visible in
            the original reference (e.g. adding alley details when the
            shot called for a new kitchen scene). Kept in the signature
            for caller compatibility; pass any value (it will be ignored).
            Identity preservation is handled by the text-based character
            mapping (see `characters` arg below) plus the actual image-gen
            model receiving the reference at generation time.
        characters: List of CharacterProfile objects (or dicts with display_name
            + physical_description) used to build a character reference block
            for the polish system prompt. Without this, the polish LLM
            substitutes generic descriptors like "the woman" for unrecognized
            names — catastrophic when the named character is e.g. a unicorn.

    Returns:
        clip_plans with polished prompts (modified in-place).
    """
    from services import llm_service

    total = sum(1 for p in clip_plans if p.get("video_prompt") or p.get("image_prompt"))
    polished = 0

    # ── Per-shot character context ─────────────────────────────────────
    # The character mapping is built PER SHOT, not globally, because the
    # global character profile only captures the original setup state
    # (e.g. "Cathy: woman on left, Ava: woman on right"). When a keyframe
    # repositions characters mid-scene — Cathy walks across the room and
    # now stands on the right — the global descriptor is stale. Pass 2
    # already produces accurate per-shot subjects_on_screen with
    # contextual visual_descriptions ("the brunette woman now on the right
    # after crossing the room"); we override the global descriptor with
    # the per-shot value when both are present.
    #
    # The per-shot context is computed fresh inside the iteration loop
    # below. The helper here just builds the global fallback map (used
    # for characters who appear in dialogue but aren't in the current
    # shot's subjects_on_screen — e.g. someone mentioned by name but
    # off-camera).

    def _make_char_map_from_global(chars: list) -> tuple[dict, dict]:
        """Build (id→name, name→global_descriptor) from CharacterProfile list."""
        char_id_to_name: dict[str, str] = {}
        name_to_desc: dict[str, str] = {}
        for c in chars or []:
            if isinstance(c, dict):
                cid = c.get("id")
                name = c.get("display_name") or c.get("name")
                desc = c.get("physical_description") or c.get("description")
            else:
                cid = getattr(c, "id", None)
                name = getattr(c, "display_name", None)
                desc = getattr(c, "physical_description", None)
            if cid and name:
                char_id_to_name[cid] = name
            if name and desc:
                name_to_desc[name] = desc
        return char_id_to_name, name_to_desc

    _global_char_id_to_name, _global_name_to_desc = _make_char_map_from_global(characters or [])

    def _build_per_shot_char_context(plan: dict) -> tuple[dict, dict, str]:
        """Build per-shot (descriptor_to_name, name_to_descriptor, ref_block_text).

        Starts from the global character mapping, then overrides each
        character's descriptor with their per-shot visual_description from
        Pass 2's subjects_on_screen — when present. This gives the polish
        LLM (and the post-process normalizer) the up-to-date positional /
        state description for THIS shot's framing, not the original setup
        descriptor that may be stale by the time this scene plays out.
        """
        # Start with global descriptors; per-shot subjects override.
        name_to_desc = dict(_global_name_to_desc)
        # Per-shot overrides
        for subj in (plan.get("subjects_on_screen") or []):
            if not isinstance(subj, dict):
                continue
            cid = subj.get("character_id")
            vis_desc = subj.get("visual_description")
            if not cid or not vis_desc:
                continue
            name = _global_char_id_to_name.get(cid)
            if name:
                name_to_desc[name] = vis_desc
            # Also pick up any screenplay-invented name (Pass 2's
            # speaker_name field). The screenplay LLM often picks a
            # personal name like "Blaine" for a character whose user-
            # supplied profile is generic ("char_1: Man in black").
            # Without this, downstream prompts use "Blaine" but the
            # polish layer doesn't know to substitute it. Adding the
            # speaker_name to the map alongside vis_desc lets
            # _normalize_names_descriptors find and replace it.
            speaker = subj.get("speaker_name")
            if speaker and isinstance(speaker, str) and speaker.strip():
                name_to_desc[speaker.strip()] = vis_desc

        # ── Defensive name harvest (Bug 3 backstop) ──────────────────
        # When Pass 2 drops speaker_name despite the schema marking it
        # REQUIRED, scan the prompt text directly for proper-noun names
        # in speech-tag / vocative positions. For each harvested name
        # that isn't already mapped, link it to the sole unnamed subject
        # on screen — heuristic but typically correct in two-character
        # shots where one speaker is named and the other is "the woman".
        # Multi-unnamed-subject cases skip with a log so we can see when
        # the heuristic isn't safe to apply.
        harvested = (
            _harvest_speaker_names(plan.get("video_prompt", ""))
            | _harvest_speaker_names(plan.get("image_prompt", ""))
        )
        if harvested:
            unnamed_subjects = [
                s for s in (plan.get("subjects_on_screen") or [])
                if isinstance(s, dict)
                and s.get("visual_description")
                and not (s.get("speaker_name") or "").strip()
            ]
            for harv_name in harvested:
                if harv_name in name_to_desc:
                    continue
                if len(unnamed_subjects) == 1:
                    name_to_desc[harv_name] = unnamed_subjects[0]["visual_description"]
                    print(
                        f"[PromptPolish] Harvested screenplay name "
                        f"'{harv_name}' → linked to sole unnamed subject "
                        f"({name_to_desc[harv_name][:40]}...)"
                    )
                elif len(unnamed_subjects) > 1:
                    print(
                        f"[PromptPolish] Harvested name '{harv_name}' "
                        f"but {len(unnamed_subjects)} subjects on screen "
                        f"have no speaker_name — cannot link unambiguously, "
                        f"name will leak to output. Pass 2 should have set "
                        f"speaker_name on the right subject."
                    )

        # Inverse: descriptor (lowercased for case-insensitive matching) → name
        desc_to_name = {desc.lower(): name for name, desc in name_to_desc.items() if desc}
        # NOTE: the system-prompt character_reference_block is no longer
        # built here — see _build_filtered_char_block() below. The full
        # name_to_desc map is still returned so the post-process passes
        # (_normalize_names_descriptors, hallucination stripper) have
        # the complete map to work from. Only the LLM-facing system
        # prompt gets the filtered subset.
        return desc_to_name, name_to_desc, ""

    def _build_filtered_char_block(input_text: str, name_to_desc: dict) -> str:
        """Build a character-reference block containing ONLY names that
        actually appear in `input_text`.

        Why: the previous static block listed every known character even
        when none appeared in the current prompt. The polish LLM saw
        entries like "Blaine → strong man in black", interpreted the
        arrow as bidirectional, and "promoted" descriptors back to
        names the input never had ("the strong man in black" became
        "The Blaine"). Filtering eliminates the inciting condition —
        if the polish LLM has no name to insert, it can't insert one.

        Wording is also strengthened: explicit ONE-WAY arrow, explicit
        DO-NOT-REVERSE rule, replace verb in imperative form. The
        previous wording ("when the prompt contains any of these names
        ... replace the name with the matching descriptor") was
        ambiguous enough that smaller LLMs treated it as bidirectional.

        Returns empty string when no names appear in input or the map
        is empty — in that case the system prompt has zero character
        guidance and the polish LLM cannot invent names.
        """
        if not input_text or not name_to_desc:
            return ""
        relevant: dict[str, str] = {}
        for name, desc in name_to_desc.items():
            if not name or not desc:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", input_text):
                relevant[name] = desc
        if not relevant:
            return ""
        char_lines = [
            f"  - REPLACE every standalone occurrence of '{name}' with '{desc}' "
            f"(only outside quoted dialogue)."
            for name, desc in relevant.items()
        ]
        return (
            "\n\nCHARACTER NAME REPLACEMENT — STRICT ONE-WAY RULE:\n"
            "The arrows below go ONE WAY ONLY. Replace the NAME with the "
            "DESCRIPTOR. Never the reverse.\n"
            "- DO replace 'Blaine' (the name) with the descriptor.\n"
            "- DO NOT replace 'the tall man in a red jacket' (the descriptor) "
            "with 'Blaine' or any other name.\n"
            "- DO NOT invent character names that are not in this list.\n"
            "- Apply ONLY in narrative prose OUTSIDE of quoted dialogue. "
            "Inside quotes, characters address each other by name and "
            "names must be preserved verbatim.\n"
            "- Use the EXACT descriptor — never substitute a generic "
            "'the man' / 'the woman' for non-human characters.\n\n"
            + "\n".join(char_lines)
        )

    # Build LoRA hint text per mode.
    #
    # ONLY inject trigger words that come from the CivitAI sidecar's
    # `trainedWords` field. These are the actual tokens the LoRA was
    # trained on. If the sidecar has no trainedWords, the LoRA is
    # silently skipped — it still loads and affects generation, but the
    # LLM won't be told anything about it.
    #
    # We deliberately do NOT extract triggers from .guide.md files.
    # Guide prose like "include the trigger phrase 'Unchained'" looks
    # like a trigger but is actually a description of what the user
    # should do — injecting it into the LLM system prompt causes it
    # to be treated as a mandatory insertion token, producing broken
    # output like "the doctor, Unchained, in white..."
    import re as _re

    def _build_lora_hints(loras: list[str], model_type: str) -> str:
        if not loras or not model_type:
            return ""
        try:
            import wgp
            import json as _json
            lora_dir = wgp.get_lora_dir(model_type)
            trigger_lines: list[str] = []
            for lora_name in loras:
                sidecar_path = os.path.join(lora_dir, os.path.splitext(lora_name)[0] + ".civitai.json")
                trigger_words: list[str] = []
                if os.path.isfile(sidecar_path):
                    try:
                        with open(sidecar_path, "r", encoding="utf-8") as sf:
                            sidecar = _json.loads(sf.read())
                        trigger_words = sidecar.get("trainedWords", []) or []
                    except Exception:
                        pass

                if trigger_words:
                    trigger_lines.append(f"- {', '.join(trigger_words[:5])}")
                # else: silently skip — do NOT inject guide prose as hints

            if not trigger_lines:
                return ""

            any_leet = any(any(c.isdigit() for c in ln) for ln in trigger_lines)
            # Leet-coded triggers (tokens containing digits like 4/0/3/5/1)
            # name a SPECIFIC sex act / position. They're useful only when
            # the scene actually depicts that act. Earlier guidance made
            # them mandatory-always — that pushed the polish LLM to prepend
            # 'm15510n4ry.' or 'bl0wj0b.' to dialogue scenes where no sex
            # was happening, diluting the LoRA's trained association without
            # any benefit. Inclusion is now conditional on scene content.
            leet_block = (
                "\n\nLEET-CODED TRIGGERS (contain digits like 4/0/3/5/1) — "
                "INCLUDE ONLY WHEN THE SCENE DEPICTS THE TRIGGER'S ACT:\n"
                "Triggers like 'bl0wj0b', 'm15510n4ry', 'd0gg1e', 'c0wg1rl' "
                "name a SPECIFIC position or sex act. They are intentionally "
                "non-natural-language tokens — the LoRA was trained to "
                "associate them with the matching act. They are useful "
                "ONLY when the prompt describes that act:\n"
                "- 'bl0wj0b'   → use only if the scene shows oral sex\n"
                "- 'm15510n4ry' → use only if the scene shows missionary position\n"
                "- 'd0gg1e'    → use only if the scene shows doggy-style\n"
                "- 'c0wg1rl'   → use only if the scene shows woman-on-top riding\n"
                "- 'r3v3rs3_c0wg1rl' → use only for reverse-cowgirl specifically\n\n"
                "If the scene is non-sexual (dialogue only, walking, looking, "
                "talking, leaning, kissing, undressing, foreplay without "
                "intercourse, or any other non-act content), OMIT all leet "
                "triggers. Do not force one in for 'completeness' — the "
                "non-act prompt won't activate the LoRA's trained pattern "
                "and may dilute the LoRA's effect on the actual sex scenes "
                "later.\n\n"
                "ALSO OMIT leet triggers from CLIMAX, ORGASM, and AFTERMATH "
                "scenes. These are reaction / resolution shots — the camera "
                "is on faces flushing, eyes closing, breath hitching, "
                "characters collapsing into each other, post-coital embrace, "
                "lying entwined in sheets, breathing slowing, sweat cooling, "
                "characters separating, fade-out moments. The position-"
                "specific trigger no longer reflects what the camera is "
                "showing in these shots — a 'm15510n4ry' tag belongs in "
                "shots where missionary is ACTIVELY being performed (visible "
                "thrusting, visible position), not in the orgasm-reaction "
                "shot that follows it or the lying-tangled aftermath shot. "
                "The LoRA was trained on shots OF the act, not shots of the "
                "moments around it. Tagging a non-act shot with the position "
                "trigger trains the model toward weaker associations.\n\n"
                "QUICK TEST — ask: 'is the position itself the literal subject "
                "of this frame?' If yes (visible thrusting / riding / "
                "penetration in the described pose), include the trigger. "
                "If no (faces, embraces, expressions, transitions, "
                "aftermath), OMIT it.\n\n"
                "When the scene DOES match a trigger, place it verbatim "
                "using ONE of these forms (do NOT decode it into English — "
                "'bl0wj0b' stays 'bl0wj0b'):\n"
                "- First word + period:    'bl0wj0b. The woman dips her head down onto him...'\n"
                "- Opening parenthetical:  '(m15510n4ry) The man thrusts steadily...'\n"
                "- Em-dash action prefix:  'c0wg1rl - the woman straddles him and grinds her hips...'\n"
                "Pick whichever fits the prompt's existing structure. The "
                "FORBIDDEN PATTERNS below DO NOT apply to leet triggers — "
                "those are for plain-English triggers only.\n"
            ) if any_leet else ""
            return (
                "\n\n[LORA TRIGGER WORDS — these are exact tokens the model was "
                "trained on. Pick the ONE most relevant trigger per prompt "
                "and include it. Plain-English triggers (e.g. 'Unchained', "
                "'BEEG', 'LTXNUDES') follow the natural-prose rule below; "
                "leet-coded triggers are MANDATORY (see leet block at the end "
                "of this directive).\n\n"
                "FOR PLAIN-ENGLISH TRIGGERS — include IF AND ONLY IF it forms a "
                "natural, grammatical part of a sentence. If you cannot weave "
                "it in naturally, OMIT IT ENTIRELY.\n\n"
                "FORBIDDEN INSERTION PATTERNS for plain-English triggers "
                "(any of these ruins the prompt):\n"
                "- At the start as a standalone tag:  'Unchained, the doctor...'\n"
                "- As a comma-offset appositive:      'the doctor, Unchained, in white...'\n"
                "- As a parenthetical:                'the doctor (Unchained) in white...'\n"
                "- As a standalone label anywhere:    '...in the exam room. Unchained. She...'\n"
                "- Attached to an unrelated character: 'the doctor, Mystic XXX, leans...'\n\n"
                "ACCEPTABLE INSERTIONS for plain-English triggers — only if grammatically natural:\n"
                "- Body/appearance descriptor trigger ('detailed muscle definition'): "
                "scoped to the right character inside a sentence — "
                "'the man with detailed muscle definition lifts the crate...'\n"
                "- Style tag trigger ('Mystic XXX', 'Unchained'): use only when the "
                "trigger names a genre or action the scene actually depicts. If it "
                "does not fit grammatically, OMIT IT. Do not force it in.\n\n"
                "Do NOT invent variants. Do NOT include a plain-English trigger that does not "
                "match the scene." + leet_block + "]\n"
            ) + "\n".join(trigger_lines)
        except Exception:
            pass
        return ""

    video_lora_hints = _build_lora_hints(video_loras or [], video_model)
    image_lora_hints = _build_lora_hints(image_loras or [], image_model)

    # Director-specific system overrides — the prompts are already detailed from passes 1-2.
    # The 3rd pass should REFINE (align structure, weave LoRA triggers) not EXPAND (add details).
    # Only mention LoRA triggers in the system prompt when we actually have
    # trigger words to inject. When hints are empty, any mention of "LoRA"
    # or "trigger" causes the LLM to hallucinate fake trigger syntax.
    _video_lora_line = (
        "- Weave in the provided LoRA trigger words naturally if they fit the content\n"
        if video_lora_hints else ""
    )
    _image_lora_line = (
        "- Weave in the provided LoRA trigger words naturally if they fit the content\n"
        if image_lora_hints else ""
    )
    _no_lora_line = (
        "- Do NOT mention LoRAs, trigger words, or anything related to model adapters — "
        "no text like 'triggering the X_lora', 'trigger word for LoRA', or invented LoRA names\n"
    )

    # CRITICAL DIALOGUE PROTECTION RULE — placed at the very top of the
    # system prompt because Pass 3 polish was observed to apply name→
    # descriptor substitution globally, including INSIDE quoted dialogue:
    #   Pass 2:  Ember says, "Variables are fun, Milo, they're like sparks."
    #   Polish:  The dragon says, "Variables are fun, boy in scruffy old
    #             clothes and glasses, they are like sparks."
    # The video model then tries to lip-sync the absurd "boy in scruffy
    # old clothes and glasses" as actual speech. Catastrophic.
    _dialogue_protection_rule = (
        "ABSOLUTE RULE — DIALOGUE IS SACROSANCT:\n"
        "Quoted dialogue (text inside single or double quotes) MUST be "
        "preserved EXACTLY as written. Do NOT modify the text inside quotes "
        "in ANY way. Specifically, do NOT replace character names inside "
        "quotes with descriptors — characters address each other by name "
        "when speaking, that is natural human/character behavior.\n\n"
        "Name → descriptor substitution applies ONLY to narrative prose "
        "OUTSIDE of quotes (action lines, camera movement, who-speaks "
        "tags). The words BETWEEN the quote marks are dialogue and stay "
        "untouched.\n\n"
        "BAD (polish broke the dialogue):\n"
        "  Pass 2:  Maya says \"Variables are fun, Leo.\"\n"
        "  Polish:  The woman in the green scarf says \"Variables are fun, boy in scruffy clothes.\"\n"
        "                                                                    ^^^^^^^^^^^^^^^^^^^^^^^ WRONG: name was inside quotes\n\n"
        "GOOD (polish only touched narrative prose):\n"
        "  Pass 2:  Maya says \"Variables are fun, Leo.\"\n"
        "  Polish:  The woman in the green scarf says \"Variables are fun, Leo.\"\n"
        "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^ name in narrative replaced; dialogue preserved\n\n"
    )

    # Base templates — character_reference_block is appended PER SHOT
    # inside the iteration loop below, using subjects_on_screen for that
    # specific shot's contextual descriptions.
    # ── Anti-hallucination rule ────────────────────────────────────
    # The character_reference_block (built per-call below) lists ONLY
    # names that appear in the input prompt — but smaller polish LLMs
    # still try to infer/insert names they've seen elsewhere. Belt-and-
    # suspenders: explicit DO-NOT-INVENT rules in the base system
    # prompt, plus the input-filtered ref_block plus the post-process
    # _strip_hallucinated_names safety net.
    # NOTE: example descriptors here are deliberately mundane AND
    # distinctive ("yellow raincoat") — a community report caught the
    # OLD descriptors ("the strong man in black", "the woman in white")
    # copied VERBATIM into generated prompts by the 4B polish model,
    # alongside a dragon lifted from the old dialogue example. If these
    # new ones ever show up in a user's output, it's the same bleed.
    _no_invent_rule = (
        "ABSOLUTE RULE — NEVER INVENT CHARACTER NAMES:\n"
        "If a character is referred to in the input as 'the tall man in "
        "a red jacket' or 'the woman in a yellow raincoat', use that "
        "descriptor. Do NOT invent a personal name for them (no 'Blaine', "
        "no 'Sarah', no 'Mark') — even if names exist elsewhere in your "
        "training data. The only personal names allowed in the output "
        "are: (a) names already present in the input prompt, or (b) "
        "names listed in the CHARACTER NAME REPLACEMENT block below "
        "(which only appears when the input already mentions those "
        "names).\n\n"
        "ABSOLUTE RULE — NEVER REPLACE PRONOUNS:\n"
        "Pronouns (he/she/they/his/her/their/him/her/them) MUST stay as "
        "pronouns. Do NOT replace 'his logic' with 'the tall man in a "
        "red jacket's logic' or 'Blaine's logic'. Pronouns are already correct.\n\n"
        "ABSOLUTE RULE — NEVER REVERSE NAME REPLACEMENT:\n"
        "If you see 'the tall man in a red jacket' in the input, leave it "
        "as 'the tall man in a red jacket'. Do NOT promote it to 'Blaine' "
        "or any other name. The character mapping below — when present — "
        "runs name → descriptor, NEVER descriptor → name.\n\n"
    )
    # ── Anti-bleed rule ─────────────────────────────────────────────
    # Small polish LLMs (Gemma 4 4B) copy CONTENT out of instruction
    # examples into their output. Observed in the wild: a music video
    # whose every clip gained an "immense iridescent dragon" plus
    # characters renamed to this file's old example descriptors. The
    # examples teach format/grammar only — say so explicitly, first,
    # in both system prompts.
    _format_only_rule = (
        "ABSOLUTE RULE — EXAMPLES ARE FORMAT ONLY:\n"
        "The examples in these instructions exist to show FORMAT and "
        "grammar. NEVER copy their subjects, names, animals, creatures, "
        "objects, clothing, or settings into your output. Everything in "
        "your output must come from the INPUT prompt — if something is "
        "not in the input, it does not appear in the output.\n\n"
    )
    _video_system_base = (
        "You are refining an already-detailed video prompt for the generation model. "
        "The prompt was written by a Director AI and is already complete.\n\n"
        + _format_only_rule +
        _dialogue_protection_rule +
        _no_invent_rule +
        "YOUR JOB:\n"
        "- Align the prompt structure to the model's expected format (present tense, flowing paragraph)\n"
        + _video_lora_line +
        "- Fix any awkward phrasing or redundancy in NARRATIVE PROSE ONLY (never inside quotes)\n"
        "- In narrative prose ONLY: when a character name listed in the CHARACTER NAME REPLACEMENT block below appears in the input, replace it with the matching descriptor; never substitute a generic 'the woman' / 'the man' for a non-human character\n\n"
        "DO NOT:\n"
        "- Modify any text inside single or double quotes — dialogue is preserved verbatim\n"
        "- Replace character names INSIDE quoted dialogue with descriptors\n"
        "- Invent character names that aren't in the input or in the mapping block\n"
        "- Replace a descriptor like 'the tall man in a red jacket' with a name like 'Blaine' (the mapping arrow is one-way: name → descriptor, NEVER reverse)\n"
        "- Replace pronouns (he/she/they/his/her/their/him/her/them) with names or descriptors — pronouns stay as pronouns\n"
        "- Add scene details (lighting, colors, atmosphere, clothing) that aren't in the original\n"
        "- Re-describe the setting — a start image already shows the scene to the video model\n"
        "- Make the prompt longer — keep it the same length or shorter\n"
        "- Invent actions, dialogue, or camera movements not in the original\n"
        "- Add style tags, emphasis phrases, or mood descriptions the original didn't have (no 'ultra-detailed', 'cinematic intensity', 'dramatic close-up', 'extreme realism')\n"
        "- Replace a character name in narrative prose with a generic 'the woman' / 'the man' when the character mapping below provides a specific descriptor (e.g. Rex → the German Shepherd, NOT the man)\n"
        + (("" if video_lora_hints else _no_lora_line)) +
        "- Use markdown formatting of any kind. NO **bold**, NO *italics*, NO headers, NO code blocks. Plain text only.\n\n"
        "Output ONLY the refined prompt. No explanation, no labels."
    )

    _image_system_base = (
        "You are refining an already-detailed image prompt for the generation model. "
        "The prompt was written by a Director AI and is already complete. "
        "The output is a SINGLE STILL FRAME — no audio, no time, no speech.\n\n"
        + _format_only_rule +
        _dialogue_protection_rule +
        _no_invent_rule +
        "YOUR JOB:\n"
        + _image_lora_line +
        "- Fix any awkward phrasing in NARRATIVE PROSE ONLY (never inside quotes)\n"
        "- Ensure the prompt ends with: 'Preserve character identity, attire, body attributes, and the art style of the reference image.'\n"
        "- When a character name listed in the CHARACTER NAME REPLACEMENT block below appears in the input, replace it with the matching descriptor IN NARRATIVE PROSE ONLY; never default to generic 'the woman' / 'the man' for non-human characters\n\n"
        "DO NOT:\n"
        "- Invent character names that aren't in the input or in the mapping block\n"
        "- Replace a descriptor with a name (the mapping arrow is one-way: name → descriptor, NEVER reverse)\n"
        "- Replace pronouns (he/she/they/his/her/their/him/her/them) with names or descriptors — pronouns stay as pronouns\n"
        # NEW: explicit no-dialogue-in-image rule. Smaller / more eager
        # models (e.g. Gemma 4 4B) will otherwise see screenplay context
        # and "embellish" the image prompt with dialogue from the script
        # ("Ember says 'Variables are fun, Milo.' Milo replies, ...").
        # Image gen models can't render speech and the dialogue often
        # comes out as broken pronouns ("they says ..."). Dialogue
        # belongs ONLY in video_prompt fields, never in image_prompt.
        # If the input prompt happens to already contain quoted speech,
        # leave it alone (handled by the dialogue-protection rule above);
        # but never CREATE new quoted speech.\n"
        "- ADD ANY dialogue, quoted speech, or 'X says \"...\"' constructions. The image is a SINGLE STILL FRAME — there is no audio, no time, no spoken words. If the input prompt has no quoted dialogue, the output must also have no quoted dialogue. Dialogue belongs ONLY in video_prompt fields, never in image_prompt.\n"
        "- Modify any text inside single or double quotes — dialogue is preserved verbatim\n"
        "- Replace character names INSIDE quoted dialogue with descriptors\n"
        "- Add lighting, color temperature, or atmosphere details not in the original\n"
        "- Change the scene setting or mood (no inventing 'soft morning glow' etc.)\n"
        "- Make the prompt longer — keep it the same length or shorter\n"
        "- Remove structural elements like 'create new scene' prefix or 'from the reference image' phrases\n"
        "- Add style tags or emphasis phrases the original didn't have (no 'ultra-detailed', 'cinematic intensity', 'dramatic close-up', 'extreme realism')\n"
        "- Add character appearance descriptions or [placeholder] brackets — the prompt already has what it needs\n"
        "- Replace 'the woman from the reference image' with appearance descriptions — keep it as-is\n"
        "- Substitute a character name in narrative prose with a generic 'the woman' / 'the man' when the character mapping below shows a non-human descriptor\n"
        + (("" if image_lora_hints else _no_lora_line)) +
        "- Use markdown formatting of any kind. NO **bold**, NO *italics*, NO headers, NO code blocks. Plain text only.\n\n"
        "Output ONLY the refined prompt. No explanation, no labels."
    )

    # Strip markdown formatting the LLM may have added (**bold**, *italics*,
    # `code`, headers). Image/video models don't parse markdown and the raw
    # asterisks become visible artifacts in the generated scene.
    _md_bold = _re.compile(r"\*\*(.+?)\*\*")
    _md_italic = _re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
    _md_code = _re.compile(r"`([^`]+)`")
    _md_header = _re.compile(r"^#{1,6}\s+", _re.MULTILINE)

    def _strip_markdown(text: str) -> str:
        if not text:
            return text
        text = _md_bold.sub(r"\1", text)
        text = _md_italic.sub(r"\1", text)
        text = _md_code.sub(r"\1", text)
        text = _md_header.sub("", text)
        return text

    # ── Apostrophe-aware quoted-span iteration ─────────────────────
    # The previous regex `(['\"])([^'\"]*?)\1` was BROKEN by single-
    # quoted dialogue containing contractions: text like
    #   'You know, it's ridiculous how much you've been on my mind.'
    # got split at every contraction apostrophe, producing 3 spurious
    # spans instead of 1 real one. When the polish LLM converted
    # surrounding quotes from single to double, the count mismatch
    # (BEFORE: 3 spurious spans, AFTER: 1 real span) caused
    # _revert_modified_dialogue to skip the revert entirely and the
    # mangled dialogue ("Cathy" → "woman in white with massive breasts")
    # passed through untouched.
    #
    # Fix: separate single- and double-quote regexes. Single-quote spans
    # require word-boundary lookbehind/lookahead so contraction
    # apostrophes (which are flanked by letters on BOTH sides) are
    # never mistaken for span boundaries. Double-quote spans are
    # unambiguous since contractions never use double-quotes.
    _DOUBLE_QUOTED_SPAN_RE = _re.compile(r'"([^"]*?)"', _re.DOTALL)
    _SINGLE_QUOTED_SPAN_RE = _re.compile(
        r"(?<![A-Za-z])'(.*?)'(?![A-Za-z])",
        _re.DOTALL,
    )

    def _iter_quoted_spans(text: str):
        """Yield (quote_char, content, start, end) for every quoted span
        in document order.

        Single-quote spans are matched with contraction-aware
        lookarounds; double-quote spans match normally. Spans are
        returned sorted by start position so callers can walk them
        sequentially. In the rare malformed case of overlapping spans
        (e.g. polish output mixing quote types incoherently), the
        consumer can detect overlap by tracking position.
        """
        if not text:
            return
        spans: list[tuple[str, str, int, int]] = []
        for m in _DOUBLE_QUOTED_SPAN_RE.finditer(text):
            spans.append(('"', m.group(1), m.start(), m.end()))
        for m in _SINGLE_QUOTED_SPAN_RE.finditer(text):
            spans.append(("'", m.group(1), m.start(), m.end()))
        spans.sort(key=lambda s: s[2])
        # Drop overlapping spans — keep the earliest in document order.
        last_end = -1
        for span in spans:
            if span[2] >= last_end:
                yield span
                last_end = span[3]

    def _normalize_names_descriptors(
        text: str,
        descriptor_to_name: dict[str, str],
        name_to_descriptor: dict[str, str],
    ) -> tuple[str, int, int]:
        """Apply bidirectional name/descriptor normalization:

        - INSIDE quotes (dialogue): replace any character descriptor with
          the original name. Polish LLMs sometimes substitute name →
          descriptor everywhere including dialogue; characters addressing
          each other by full physical descriptions sounds insane.

        - OUTSIDE quotes (narrative prose): replace any character name
          with the canonical descriptor. The image-gen model has no idea
          who "Lumi" is — it needs "the white unicorn". Polish LLMs that
          have been told "don't touch dialogue" sometimes become
          risk-averse and stop replacing names in narrative too.

        Both directions use word-boundary regex to avoid false positives
        on common words (a character named "Star" doesn't replace "star"
        in "look at the star"). Maps are PER SHOT — descriptors reflect
        the current shot's positioning/state from Pass 2's
        subjects_on_screen, not the original setup descriptors.

        Returns (cleaned_text, reverts_in_dialogue, replaces_in_narrative).
        """
        if not text or (not descriptor_to_name and not name_to_descriptor):
            return text, 0, 0

        reverts = 0
        replaces = 0

        def _replace_inside_quotes(seg: str) -> str:
            """Inside quotes: descriptor → name (revert wrong substitutions)."""
            nonlocal reverts
            for desc, name in descriptor_to_name.items():
                pattern = _re.compile(_re.escape(desc), _re.IGNORECASE)
                if pattern.search(seg):
                    seg = pattern.sub(name, seg)
                    reverts += 1
            return seg

        def _replace_outside_quotes(seg: str) -> str:
            """Outside quotes: name → descriptor (apply rule polish missed)."""
            nonlocal replaces
            for name, desc in name_to_descriptor.items():
                pattern = _re.compile(r"\b" + _re.escape(name) + r"\b")
                if pattern.search(seg):
                    seg = pattern.sub(desc, seg)
                    replaces += 1
            return seg

        parts: list[str] = []
        pos = 0
        for quote_char, inside, start, end in _iter_quoted_spans(text):
            outside = text[pos:start]
            outside = _replace_outside_quotes(outside)
            inside = _replace_inside_quotes(inside)
            parts.append(outside)
            parts.append(f"{quote_char}{inside}{quote_char}")
            pos = end
        parts.append(_replace_outside_quotes(text[pos:]))
        return "".join(parts), reverts, replaces

    def _revert_modified_dialogue(before: str, after: str) -> tuple[str, int]:
        """Position-pair quoted spans between input and polish output.
        Any quoted span whose contents differ from the input gets reverted
        to the input's original text.

        Defensive complement to _normalize_names_descriptors. That function
        only reverts known descriptor→name substitutions inside quotes; it
        can't catch when the polish LLM hallucinates a NEW descriptor
        ("woman in middle with massive breasts") that isn't in our map at
        all. This function reverts ANY modification to quoted text — the
        polish layer has no business changing dialogue, period.

        Conservative on edge cases:
        - If quote counts differ between before and after (polish added or
          removed a quoted span), returns the polished text unchanged
          rather than risk a misalignment that mangles output. The
          mismatch is logged so we can investigate.
        - Quote characters from the input are preserved (straight " vs '
          stay the way the screenplay author wrote them).
        - Empty quotes are skipped to avoid replacing intentional speech
          beats with empty strings.

        Uses _iter_quoted_spans which is contraction-aware — the previous
        regex split single-quoted dialogue containing apostrophes
        ("it's", "you've") into multiple fake spans, causing this revert
        to silently bail on count mismatch. Real-world impact: dialogue
        like "Cathy" being mangled to descriptor inside quotes was
        passing through unfixed.

        Returns (cleaned_text, reverts_count).
        """
        if not before or not after:
            return after, 0
        before_quotes = list(_iter_quoted_spans(before))
        after_quotes = list(_iter_quoted_spans(after))
        if not after_quotes:
            return after, 0
        if len(before_quotes) != len(after_quotes):
            # Polish added/removed quoted spans. Possible legitimate cases
            # (rare): polish split a long line into two. But more often
            # this means the polish output is malformed in some other way.
            # Don't try to repair — that risks worse output.
            print(
                f"[PromptPolish] Dialogue revert skipped — quote count "
                f"mismatch (input had {len(before_quotes)}, output has "
                f"{len(after_quotes)})"
            )
            return after, 0

        # Build the reverted output by walking after-quotes in order and
        # substituting each one's content with the matching before-quote.
        out_parts: list[str] = []
        pos = 0
        reverts = 0
        for (b_quote, b_content, _b_start, _b_end), (a_quote, a_content, a_start, a_end) in zip(before_quotes, after_quotes):
            out_parts.append(after[pos:a_start])
            if a_content == b_content:
                # No modification — keep as is.
                out_parts.append(after[a_start:a_end])
            else:
                # Polish modified this dialogue. Revert to input's content
                # while preserving the quote-character style polish chose
                # (sometimes polish converts ' to " or vice versa, which
                # is harmless and we don't want to undo).
                out_parts.append(f"{a_quote}{b_content}{a_quote}")
                reverts += 1
            pos = a_end
        out_parts.append(after[pos:])
        return "".join(out_parts), reverts

    def _enhance_video(prompt: str, system_prompt: str, desc_to_name: dict, name_to_desc: dict) -> str:
        # Storyboard format guard: when the input prompt is Format B
        # (contains "Shot N (Camera, Xs):" blocks for the IC-LoRA),
        # SKIP polish entirely. The polish LLM otherwise rewrites the
        # structured blocks into flowing prose, destroying the literal
        # trigger pattern the IC-LoRA was trained on. The IC-LoRA then
        # produces no internal cuts because the structure is gone.
        # User-reported regression: Pass 2 produced perfect storyboard
        # output, Pass 3 turned every `Shot 1 (Medium, 8s):` into
        # "The woman in dark brown dress slowly wipes..." flowing prose.
        # Trade-off: storyboard prompts don't get name → descriptor
        # substitution, dialogue protection, or markdown stripping.
        # The Pass 2 LLM already follows those rules when writing
        # storyboard content; the additional polish layer was net-
        # negative for this format.
        if _is_storyboard_format(prompt):
            print(
                "[PromptPolish] Video polish SKIPPED — input is in Multi-Shot "
                "LoRA storyboard format. Polishing would convert structured "
                "'Shot N (Camera, Xs):' blocks to flowing prose and break the "
                "IC-LoRA trigger pattern."
            )
            return prompt
        out = llm_service.enhance_prompt(
            prompt=prompt, mode="video", max_new_tokens=512, temperature=0.3,
            nsfw=nsfw, model_type=video_model,
            system_override=system_prompt,
            lora_system_hint=video_lora_hints,
        )
        if out:
            cleaned = _strip_markdown(out)
            cleaned, reverts, replaces = _normalize_names_descriptors(cleaned, desc_to_name, name_to_desc)
            if reverts:
                print(f"[PromptPolish] Reverted {reverts} dialogue substitution(s) — polish put known descriptors inside quoted dialogue")
            if replaces:
                print(f"[PromptPolish] Replaced {replaces} name(s) in narrative prose with descriptors — polish left them as bare names")
            # Position-paired catch-all: revert ANY modification to quoted
            # dialogue, including hallucinated descriptors that aren't in
            # our character map. Runs after the descriptor-aware revert
            # above so that any remaining quote modifications get caught.
            cleaned, dlg_reverts = _revert_modified_dialogue(prompt, cleaned)
            if dlg_reverts:
                print(f"[PromptPolish] Reverted {dlg_reverts} quoted-dialogue modification(s) by position-pair")
            # Hallucinated-name strip: any proper noun in output not in
            # input AND not in our map is a polish-LLM invention (typical
            # symptom: 'the strong man in black' → 'The Blaine'). Strip
            # by substituting with the on-screen descriptor; possessives
            # ('Blaine's logic') become 'their logic'.
            cleaned, hallucinations = _strip_hallucinated_names(
                prompt, cleaned, name_to_desc,
                fallback_descriptors=list(name_to_desc.values()),
            )
            if hallucinations:
                print(
                    f"[PromptPolish] Stripped {len(hallucinations)} hallucinated "
                    f"name(s) from video prompt: {', '.join(hallucinations)}"
                )
            # Double-article collapse — when name → descriptor substitution
            # runs and the prompt had "the {name}" (polish LLM occasionally
            # hallucinates an article before a name), we get "the the
            # {descriptor}". Cheap regex pass to collapse.
            cleaned, doubles = _collapse_double_articles(cleaned)
            if doubles:
                print(f"[PromptPolish] Collapsed {doubles} double-article(s) introduced by name substitution")
            # Article-pronoun collapse: name → pronoun substitutions
            # earlier in the chain may have left "the he" / "the she"
            # behind. Always wrong English; collapse unconditionally.
            cleaned, art_pron = _collapse_article_pronoun(cleaned)
            if art_pron:
                print(f"[PromptPolish] Collapsed {art_pron} article+pronoun pair(s)")
            return cleaned
        return out

    def _enhance_image(prompt: str, system_prompt: str, desc_to_name: dict, name_to_desc: dict) -> str:
        # Storyboard format guard — same rationale as _enhance_video.
        # Image prompts shouldn't normally be in storyboard format
        # (they describe a single frame), but if Pass 2 emits one
        # anyway, the polish LLM would mangle it the same way.
        if _is_storyboard_format(prompt):
            print(
                "[PromptPolish] Image polish SKIPPED — input is in storyboard format."
            )
            return prompt
        # IMPORTANT: do NOT pass image_paths to Pass 3 image polish.
        # Pass 2 generates image_prompts that describe FUTURE start
        # frames for each shot — often a different location, new pose,
        # added/removed objects, or other state changes that don't match
        # the original reference image. The reference would bias the
        # polish toward describing what's in the reference instead of
        # the new scene state. Identity preservation is handled by the
        # text-based character_reference_block (per-shot, includes
        # current positioning) and by the actual image-gen model at
        # generation time.
        out = llm_service.enhance_prompt(
            prompt=prompt, mode="image", max_new_tokens=256, temperature=0.3,
            nsfw=nsfw, model_type=image_model,
            system_override=system_prompt,
            lora_system_hint=image_lora_hints,
        )
        if out:
            cleaned = _strip_markdown(out)
            cleaned, reverts, replaces = _normalize_names_descriptors(cleaned, desc_to_name, name_to_desc)
            if reverts:
                print(f"[PromptPolish] Reverted {reverts} image-prompt dialogue substitution(s)")
            if replaces:
                print(f"[PromptPolish] Replaced {replaces} name(s) in image-prompt narrative with descriptors")
            # Image prompts shouldn't normally contain dialogue at all,
            # but when they do (Pass 2 sometimes leaves quoted text in
            # an image_prompt), apply the same position-paired revert as
            # video prompts so polish can't mangle quoted speech.
            cleaned, dlg_reverts = _revert_modified_dialogue(prompt, cleaned)
            if dlg_reverts:
                print(f"[PromptPolish] Reverted {dlg_reverts} image-prompt quoted-dialogue modification(s) by position-pair")
            # Hallucinated-name strip (same defense-in-depth as video).
            cleaned, hallucinations = _strip_hallucinated_names(
                prompt, cleaned, name_to_desc,
                fallback_descriptors=list(name_to_desc.values()),
            )
            if hallucinations:
                print(
                    f"[PromptPolish] Stripped {len(hallucinations)} hallucinated "
                    f"name(s) from image prompt: {', '.join(hallucinations)}"
                )
            # Layer 1 sanitization (garments / filler / descriptor dedupe).
            # Runs AFTER name normalization so the dedupe pass sees the
            # same descriptor strings the normalizer just inserted.
            cleaned = sanitize_image_prompt(
                cleaned, name_to_desc, log_prefix="[PromptPolish Pass3 image sanitize]"
            )
            return cleaned
        return out

    # Counters track actual changes vs. unchanged passthroughs separately.
    # A "polished" call only counts when the LLM produced output that's
    # genuinely different from the input — `enhance_prompt` returns the
    # original prompt unchanged when the model gave us empty content
    # (the thinking-eats-budget failure mode), so we mustn't count those
    # as successful polishes.
    total_calls = 0
    actually_changed = 0
    unchanged_passthrough = 0
    failed = 0
    # Early-abort safety: if the first 3 polish calls all return identical
    # passthroughs (signals a thinking-stubborn Qwen template that's
    # ignoring all three suppression mechanisms in enhance_prompt), stop
    # firing additional polish calls. The user's screenplay-driven Pass 2
    # output is usable as-is and we save many wasted CivitAI/llama-server
    # round trips. The threshold is 3 to allow occasional one-off model
    # hiccups before giving up entirely.
    EARLY_ABORT_THRESHOLD = 3
    polish_aborted = False

    def _count_outcome(original: str, enhanced: str) -> None:
        nonlocal total_calls, actually_changed, unchanged_passthrough, polish_aborted
        total_calls += 1
        is_passthrough = (
            not enhanced
            or not enhanced.strip()
            or enhanced.strip() == (original or "").strip()
        )
        if is_passthrough:
            unchanged_passthrough += 1
        else:
            actually_changed += 1
        # If we've made several calls and ALL of them came back as
        # passthroughs, the LLM is silently failing every time. Abort.
        if (
            total_calls >= EARLY_ABORT_THRESHOLD
            and actually_changed == 0
            and not polish_aborted
        ):
            polish_aborted = True
            print(
                f"[PromptPolish] Early-abort: {total_calls} consecutive polish "
                "calls returned empty/unchanged content. The LLM is likely a "
                "thinking-stubborn build (Qwen3.5/3.6 chat templates often "
                "ignore enable_thinking=False). Skipping remaining polish "
                "calls and using Pass 2 output as-is."
            )

    # ── Diagnostic: log the global character map once per polish run ─
    # Helps debug "why didn't 'Blaine' get substituted?" — when a
    # user-defined character name leaks to output, the first thing to
    # check is whether the name is even in our map. If this log shows
    # an empty global map, the character profiles aren't reaching
    # polish_prompts_third_pass at all. If the global map has the
    # name but per-shot logs show it missing, _build_per_shot_char_context
    # is dropping it. Goes to stdout so it shows in the API log file
    # alongside the existing "Loaded video guide" / "Loaded LoRA
    # trigger" lines.
    if _global_name_to_desc:
        _gnames = ", ".join(f"{n!r}->{d[:30]}" for n, d in _global_name_to_desc.items())
        print(f"[PromptPolish] Global character map: {_gnames}")
    else:
        print(
            "[PromptPolish] Global character map is EMPTY — "
            "either no characters were defined in Director setup, "
            "or character profiles failed to propagate to "
            "polish_prompts_third_pass. Name-based substitutions "
            "in narrative prose will not fire."
        )

    for plan_idx, plan in enumerate(clip_plans):
        # Build per-shot character context. The descriptors here reflect
        # THIS shot's positioning and state (per Pass 2's
        # subjects_on_screen), overriding the global setup descriptors
        # when both exist. Falls back to global for characters mentioned
        # in dialogue but not currently on-camera.
        shot_desc_to_name, shot_name_to_desc, _unused_ref_block = _build_per_shot_char_context(plan)
        # Per-shot diagnostic. Shows which names are in the substitution
        # map for THIS shot — the map _normalize_names_descriptors will
        # use to convert names → descriptors in narrative AND descriptors
        # → names in dialogue. If a name appears in the polish output
        # but isn't listed here, _normalize_names_descriptors had no
        # entry to substitute and the post-process can't fix it.
        if shot_name_to_desc:
            _snames = ", ".join(f"{n!r}->{d[:30]}" for n, d in shot_name_to_desc.items())
            print(f"[PromptPolish] Shot {plan_idx+1} name map: {_snames}")
        else:
            print(f"[PromptPolish] Shot {plan_idx+1} name map: EMPTY")

        # Per-call system-prompt builder. The character_reference_block
        # is now filtered to ONLY names that actually appear in `text` —
        # eliminates the reverse-substitution incitement bug where the
        # polish LLM saw "Blaine → strong man in black" in the system
        # prompt and inserted "Blaine" into output that had no name.
        def _build_video_system_for(text: str) -> str:
            return _video_system_base + _build_filtered_char_block(text, shot_name_to_desc)

        def _build_image_system_for(text: str) -> str:
            return _image_system_base + _build_filtered_char_block(text, shot_name_to_desc)

        wps = plan.get("window_prompts", [])
        has_windows = bool(wps and len(wps) > 1)

        # Polish video prompt (skip if window_prompts exist — video_prompt is unused)
        if not has_windows and not polish_aborted:
            vp = plan.get("video_prompt", "")
            if vp and vp.strip():
                try:
                    enhanced = _enhance_video(vp, _build_video_system_for(vp), shot_desc_to_name, shot_name_to_desc)
                    _count_outcome(vp, enhanced)
                    if enhanced and enhanced.strip() and enhanced.strip() != vp.strip():
                        plan["video_prompt"] = enhanced.strip()
                        polished += 1
                except Exception as e:
                    failed += 1
                    print(f"[PromptPolish] Video polish failed: {e}")

        # Polish window prompts (each with context of the other windows)
        if has_windows and not polish_aborted:
            polished_wps = []
            for wi, wp in enumerate(wps):
                if polish_aborted:
                    polished_wps.append(wp)
                    continue
                if wp and wp.strip():
                    # Storyboard format guard at the CALLER level (not
                    # just inside _enhance_video). User-reported bug:
                    # the inner guard correctly skipped the LLM call,
                    # but _enhance_video had already been handed a
                    # `full_text` containing the "[Window N of N in a
                    # continuous scene. Continue from...]" context
                    # prefix that the caller adds before each polish
                    # call. The skip returned `full_text` unchanged
                    # and the caller stored it as the new window
                    # content — permanently embedding the meta-prefix
                    # into the prompt that goes to the video model.
                    # Result: every storyboard window had garbage
                    # like "[Window 1 of 2 in a continuous scene.]
                    # Shot 1 (Two-Shot, 12s)...".
                    #
                    # Fix: detect storyboard at the CALLER and skip
                    # both the context-adding AND the polish call.
                    # The original wp goes straight through, clean.
                    if _is_storyboard_format(wp):
                        print(
                            f"[PromptPolish] Window {wi+1}/{len(wps)} "
                            f"SKIPPED — storyboard format detected, "
                            f"context prefix omitted to preserve "
                            f"IC-LoRA trigger pattern."
                        )
                        polished_wps.append(wp)
                        _count_outcome(wp, wp)  # count as passthrough
                        continue
                    try:
                        # Add context so enhancer continues naturally from previous window
                        context = f"[Window {wi+1} of {len(wps)} in a continuous scene. "
                        if wi > 0:
                            context += f"Continue from the previous window: '{wps[wi-1][-100:]}'. "
                            context += "Re-establish any ongoing state (crowd still cheering, rain still falling) before new action — the video model has no memory beyond the last few frames."
                        context += "]\n\n"
                        full_text = context + wp
                        enhanced = _enhance_video(full_text, _build_video_system_for(full_text), shot_desc_to_name, shot_name_to_desc)
                        _count_outcome(wp, enhanced)
                        if enhanced and enhanced.strip() and enhanced.strip() != wp.strip():
                            polished_wps.append(enhanced.strip())
                            polished += 1
                        else:
                            polished_wps.append(wp)
                    except Exception:
                        failed += 1
                        polished_wps.append(wp)
                else:
                    polished_wps.append(wp)
            plan["window_prompts"] = polished_wps

        # Polish image prompt
        if not polish_aborted:
            ip = plan.get("image_prompt", "")
            if ip and ip.strip():
                try:
                    enhanced = _enhance_image(ip, _build_image_system_for(ip), shot_desc_to_name, shot_name_to_desc)
                    _count_outcome(ip, enhanced)
                    if enhanced and enhanced.strip() and enhanced.strip() != ip.strip():
                        plan["image_prompt"] = enhanced.strip()
                        polished += 1
                except Exception as e:
                    failed += 1
                    print(f"[PromptPolish] Image polish failed: {e}")

        # Polish keyframe prompts (each one is an image prompt for an
        # additional reference frame the video model uses mid-clip).
        # These follow the same image-prompt rules as the start frame
        # so we run them through _enhance_image with the same per-shot
        # system prompt. Previously skipped — surfaced as Issue #1 in
        # the Director Dashboard ("no diff for keyframes").
        if not polish_aborted:
            kfs = plan.get("keyframe_prompts", []) or []
            if isinstance(kfs, list) and kfs:
                polished_kfs = []
                for ki, kf in enumerate(kfs):
                    if polish_aborted:
                        polished_kfs.append(kf)
                        continue
                    if isinstance(kf, str) and kf.strip():
                        try:
                            enhanced = _enhance_image(kf, _build_image_system_for(kf), shot_desc_to_name, shot_name_to_desc)
                            _count_outcome(kf, enhanced)
                            if enhanced and enhanced.strip() and enhanced.strip() != kf.strip():
                                polished_kfs.append(enhanced.strip())
                                polished += 1
                            else:
                                polished_kfs.append(kf)
                        except Exception as e:
                            failed += 1
                            polished_kfs.append(kf)
                            print(f"[PromptPolish] Keyframe[{ki}] polish failed: {e}")
                    else:
                        polished_kfs.append(kf)
                plan["keyframe_prompts"] = polished_kfs


    # Report all four outcomes so silent-failure modes are visible.
    # The most useful number for the user is actually_changed: that's how
    # many prompts the LLM successfully rewrote. unchanged_passthrough is
    # the failure count we used to silently absorb (thinking-eats-budget
    # → enhance_prompt returns original → counted as "polished").
    print(
        f"[PromptPolish] Third-pass complete: "
        f"{actually_changed} prompts rewritten, "
        f"{unchanged_passthrough} unchanged (LLM returned empty/identical), "
        f"{failed} errored, "
        f"{total_calls} total polish calls across {total} clip(s)"
    )
    return clip_plans
