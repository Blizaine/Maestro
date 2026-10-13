"""Small source-action ledger for music-driven Director plans.

This deliberately recognizes a bounded set of visible actions. Performance,
style, and static identity descriptions do not become story milestones.
"""

from __future__ import annotations

import re
from typing import Any


_REVISION = 1
_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"[^.!?;\n]+(?:[.!?;]+|$)")
_SUBJECT_TAG_RE = re.compile(r"<\s*subject\s+\d+\s*>", re.IGNORECASE)
_NATIVE_SECTION_RE = re.compile(
    r"(?im)^\s*(subject_definitions|summary|retention_analysis|"
    r"detailed_description|overall_soundscape|non_diegetic_music)\s*:\s*"
)

# Keep the inventory finite and action-focused. In particular, singing,
# dancing, holding, wearing, and riding are not transitions by themselves.
_ACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "boarding",
        re.compile(
            r"\b(?:board(?:s|ed|ing)?|mount(?:s|ed|ing)?|"
            r"climb(?:s|ed|ing)?\s+(?:up\s+)?(?:on|onto|atop|aboard)|"
            r"hop(?:s|ped|ping)?\s+(?:on|onto|aboard)|"
            r"step(?:s|ped|ping)?\s+(?:on|onto|aboard)|"
            r"clamber(?:s|ed|ing)?\s+(?:on|onto|aboard)|"
            r"(?:get|gets|got|getting)\s+(?:on|onto|in|into|aboard))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "flight",
        re.compile(
            r"\b(?:fly|flies|flew|flying|soar|soars|soared|soaring|"
            r"glide|glides|glided|gliding|swoop|swoops|swooped|swooping|"
            r"dive|dives|dived|diving|descend|descends|descended|descending|"
            r"take\s+off|takes\s+off|taking\s+off|lift\s+off|lifts\s+off|"
            r"lifting\s+off|launch|launches|launched|launching)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "landing",
        re.compile(
            r"\b(?:land|lands|landed|landing|touch\s+down|touches\s+down|"
            r"touched\s+down|dismount|dismounts|dismounted|dismounting|"
            r"(?:get|gets|got|getting)\s+off|alight|alights|alighted)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "attack",
        re.compile(
            r"\b(?:breathes|breaths|breathe|breathing)\s+fire\b|"
            r"\b(?:attack|attacks|attacked|attacking|strike|strikes|struck|"
            r"striking|hit|hits|hit|hitting|fire|fires|fired|firing|"
            r"shoot|shoots|shot|shooting|blast|blasts|blasted|blasting|"
            r"charge|charges|charged|charging|bite|bites|bit|biting|"
            r"claw|claws|clawed|clawing|swipe\s+at|swipes\s+at|"
            r"slash|slashes|slashed|slashing|stab|stabs|stabbed|stabbing|"
            r"punch|punches|punched|punching|kick|kicks|kicked|kicking|"
            r"smash|smashes|smashed|smashing|destroy|destroys|destroyed|"
            r"burn|burns|burned|burning)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "action",
        re.compile(
            r"\b(?:open|opens|opened|opening|"
            r"close|closes|closed|closing|grab|grabs|grabbed|grabbing|"
            r"pick\s+up|picks\s+up|picked\s+up|picking\s+up|"
            r"drop|drops|dropped|dropping|throw|throws|threw|thrown|throwing|"
            r"stop|stops|stopped|stopping|"
            r"hand|hands|handed|handing|give|gives|gave|given|giving|"
            r"draw|draws|drew|drawn|drawing|raise|raises|raised|raising|"
            r"lower|lowers|lowered|lowering|pull|pulls|pulled|pulling|"
            r"push|pushes|pushed|pushing|enter|enters|entered|entering|"
            r"exit|exits|exited|exiting|walk|walks|walked|walking|"
            r"run|runs|ran|running|jump|jumps|jumped|jumping|"
            r"fall|falls|fell|fallen|falling|leap|leaps|leaped|leapt|leaping|"
            r"swim|swims|swam|swimming|row|rows|rowed|rowing|"
            r"paddle|paddles|paddled|paddling|sail|sails|sailed|sailing|"
            r"drive|drives|drove|driven|driving|transform|transforms|"
            r"transformed|transforming|turns\s+around|turned\s+around|"
            r"spins|spun|spinning|waves|waved|waving)\b",
            re.IGNORECASE,
        ),
    ),
)

_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your",
    "his", "her", "their", "our", "one", "another",
}
_ENTITY_STOP_WORDS = {
    "aboard", "after", "and", "are", "as", "at", "atop", "before", "by",
    "can", "could", "down", "from", "had", "has", "have", "he", "her",
    "into", "in", "is", "it", "its", "may", "might", "must", "of", "off",
    "on", "onto", "or", "over", "she", "should", "that", "the", "their",
    "them", "then", "they", "through", "to", "up", "was", "were", "while",
    "will", "with", "would", "under", "when", "where", "who", "which",
}
_GENERIC_ENTITY_HEADS = {
    "angle", "camera", "film", "frame", "image", "music", "prompt", "scene",
    "shot", "shots", "song", "video", "view",
}
_STATIC_MODIFIER_HEADS = {
    "arms", "background", "camera", "clothes", "door", "doors", "field",
    "frame", "hair", "landscape", "lights", "lips", "mouth", "pose",
    "scene", "shot", "shots", "sky", "sunset", "view", "water", "waters",
}
_BODY_PART_NOUN_PREFIX_RE = re.compile(
    r"\b(?:a|an|the|this|that|these|those|my|your|his|her|its|our|their|"
    r"one|two|both|each|left|right|front|back|upper|lower|dominant|off)\s+"
    r"(?:(?:left|right|front|back|upper|lower|dominant|off|bare|raised|open|closed)\s+)?$",
    re.I,
)
_DIRECTORIAL_SENTENCE_PREFIX_RE = re.compile(
    r"^\s*(?:show|include|depict|feature|capture|follow|frame|"
    r"focus(?:\s+on)?|illustrate|portray)\b",
    re.I,
)
_ACTION_REFERENCE_HEAD_RE = re.compile(
    r"^\s+(?:action|event|beat|sequence|moment|scene|shot|move|movement|"
    r"step|direction|instruction|schedule|plan)\b",
    re.I,
)
_ACTION_REFERENCE_VERB_PREFIX_RE = re.compile(
    r"\b(?:replay|repeat|revisit|restate|mention|reference|describe|show|"
    r"include|depict|preserve|avoid|omit|skip)(?:s|ed|ing)?\s+"
    r"(?:(?:the|a|an|requested|previous|prior|above|earlier|completed)\s+)?$",
    re.I,
)
_DIRECTIVE_ACTION_REFERENCE_RE = re.compile(
    r"\b(?:show|include|depict|feature|capture|reference|describe|preserve)\b"
    r"[^.!?;]{0,96}\b(?:action|event|beat|sequence|moment|scene|shot|move|"
    r"movement|step)\b",
    re.I,
)
_FILMMAKING_CUE_RE = re.compile(
    r"\b(?:camera|cameras|lens|lenses|cinematograph\w*|filmmak\w*|"
    r"wide|medium|close[- ]?up|establishing|tracking|pov|locked|static|fixed|"
    r"frame|framing|shot|shots|footage|filming|film|video|scene|sequence|"
    r"composition|focus|pan|tilt|zoom|dolly|cut|take)\b",
    re.I,
)
_PHYSICAL_ATTACK_CUE_RE = re.compile(
    r"\b(?:gun|guns|rifle|rifles|pistol|pistols|firearm|firearms|weapon|"
    r"weapons|bow|arrow|arrows|bullet|bullets|missile|missiles|cannon|"
    r"shotgun|sniper|target|targets|enemy|enemies|opponent|opponents|"
    r"victim|victims)\b|\bin the (?:arm|leg|chest|head|back)\b",
    re.I,
)
_FILMING_OBJECT_RE = re.compile(
    r"\b(?:a|an|the|this|that)?\s*(?:wide\s+|medium\s+|close[- ]?up\s+|"
    r"establishing\s+|tracking\s+|pov\s+)?(?:shot|scene|sequence|take|"
    r"footage|film|video|close[- ]?up)\b",
    re.I,
)
_NON_MILESTONE_CLAUSE_RE = re.compile(
    r"^\s*(?:(?:and\s+then|then|and|but|while|as|after|before|when)\s+)?"
    r"(?:(?:the\s+)?(?:singer|vocalist|performer|musician|artist|band|camera|subject)\s+)?"
    r"(?:watch(?:es|ed|ing)?|look(?:s|ed|ing)?|listen(?:s|ed|ing)?|"
    r"sing(?:s|ing)?|perform(?:s|ed|ing)?|dance(?:s|d|ing)?|"
    r"play(?:s|ed|ing)?|wait(?:s|ed|ing)?|remain(?:s|ed|ing)?|"
    r"stand(?:s|ing)?|stood|sit(?:s|ting)?|sat)\b",
    re.I,
)
_CLAUSE_BREAK_RE = re.compile(
    r"[,;.!?]|\b(?:and\s+then|then|while|as|after|before|when|but|and)\b",
    re.I,
)


def _action_clause(value: str, start: int, end: int) -> str:
    left = value[max(0, start - 100):start]
    right = value[end:min(len(value), end + 110)]
    prior_breaks = list(_CLAUSE_BREAK_RE.finditer(left))
    next_break = _CLAUSE_BREAK_RE.search(right)
    return (left[prior_breaks[-1].end():] if prior_breaks else left) + value[start:end] + (
        right[:next_break.start()] if next_break else right
    )


def _direct_film_object(tail: str) -> bool:
    remainder = re.sub(r"^\s*(?:with|on|into|to)\s+", "", tail, flags=re.I)
    return bool(_FILMING_OBJECT_RE.match(remainder))


def _is_filmmaking_action(
    value: str,
    start: int,
    end: int,
    verb: str,
) -> bool:
    """Ignore camera-direction vocabulary without hiding concrete attacks.

    The action inventory includes ``shot`` and ``shoot`` for literal violence,
    so only discard them when nearby wording makes the filmmaking meaning
    clear. Explicit weapons, targets, or injury locations keep the physical
    reading even when camera language occurs elsewhere in the sentence.
    """

    normalized = _normal_text(verb)
    camera_action = normalized in {
        "shot", "shoot", "shoots", "shooting", "fire", "fires", "fired",
        "firing", "open", "opens", "opened", "opening", "close", "closes",
        "closed", "closing", "stop", "stops", "stopped", "stopping",
    }
    if not camera_action:
        return False

    context = value[max(0, start - 88):min(len(value), end + 100)]
    clause = _action_clause(value, start, end)
    if _PHYSICAL_ATTACK_CUE_RE.search(clause):
        return False

    cue = bool(_FILMMAKING_CUE_RE.search(context))
    if not cue:
        return False

    if normalized in {"stop", "stops", "stopped", "stopping"}:
        return bool(re.search(
            r"\b(?:camera|cameraman|cinematographer|dolly|tracking\s+shot)\s+"
            r"(?:is\s+)?stop(?:s|ped|ping)?\b",
            context,
            re.I,
        ))

    # "Open/close with a shot" and "opening/closing shot" describe an edit or
    # composition. Opening/closing doors, scenes of action, etc. stay intact
    # unless a clear camera/shot object is present nearby.
    if normalized in {"open", "opens", "opened", "opening", "close", "closes", "closed", "closing"}:
        tail = value[end:min(len(value), end + 80)]
        if normalized == "close" and re.match(r"\s*-\s*up\b", tail, re.I):
            return True
        direct_film_object = _direct_film_object(tail)
        camera_subject = bool(re.search(
            r"\b(?:camera|video|film|scene|sequence)\s+(?:opens?|closes?|opening|closing)\b",
            context,
            re.I,
        ))
        return direct_film_object or camera_subject

    # ``shot`` as a visual framing noun is commonly modified by lens/shot
    # sizes or followed by duration/composition language. A literal ``shot in
    # the leg`` is covered by the physical cue above.
    if normalized == "shot":
        return bool(
            re.search(
                r"\b(?:wide|medium|close[- ]?up|establishing|tracking|pov|locked|static|fixed)\s+shot\b|"
                r"\bshot\s+(?:throughout|of\s+(?:the\s+)?(?:scene|singer|performer)|from\s+)",
                context,
                re.I,
            )
        )

    # ``shoot``/``fire`` are filmmaking only when aimed at a film object or
    # when a camera is the grammatical subject; the bare verbs remain actions.
    tail = value[end:min(len(value), end + 80)]
    camera_subject = bool(re.search(
        r"\b(?:camera|cameraman|cinematographer|director|filmmaker)\s+"
        r"(?:is\s+)?(?:shoots?|shooting|fires?|firing)\b",
        context,
        re.I,
    ))
    return camera_subject or _direct_film_object(tail)


_PERFORMANCE_VERB_WORDS = {
    "sing", "sings", "singing", "sang", "sung", "dance", "dances",
    "dancing", "danced", "perform", "performs", "performing", "performed",
    "hold", "holds", "holding", "held", "wear", "wears", "wearing", "wore",
    "ride", "rides", "riding", "rode",
}
_SEMANTIC_VERB_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("climb", re.compile(r"\bclimb(?:s|ed|ing)?\b", re.I)),
    ("fly", re.compile(r"\b(?:fly|flies|flew|flying|flown|soar|soars|soared|soaring)\b", re.I)),
    ("swoop", re.compile(r"\bswoop(?:s|ed|ing)?\b", re.I)),
    ("breath_fire", re.compile(r"\b(?:breaths?|breathe|breathes|breathing|breathed)\s+fire\b", re.I)),
)
_GENERIC_SEMANTIC_VERB_FAMILIES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\bwalk(?:s|ed|ing)?\b", r"\brun(?:s|ning)?\b|\bran\b",
        r"\bjump(?:s|ed|ing)?\b", r"\bfall(?:s|ing)?\b|\bfell\b|\bfallen\b",
        r"\bleap(?:s|ed|ing)?\b|\bleapt\b", r"\bswim(?:s|ming)?\b|\bswam\b",
        r"\brow(?:s|ed|ing)?\b", r"\bpaddl(?:e|es|ed|ing)\b",
        r"\bsail(?:s|ed|ing)?\b", r"\bdriv(?:e|es|en|ing)\b|\bdrove\b",
        r"\bstop(?:s|ped|ping)?\b", r"\b(?:land|lands|landed|landing)\b",
        r"\b(?:touch(?:es|ed|ing)?\s+down)\b", r"\b(?:glide|glides|glided|gliding)\b",
        r"\b(?:launch|launches|launched|launching)\b",
        r"\b(?:open|opens|opened|opening)\b", r"\bclos(?:e|es|ed|ing)\b",
        r"\bgrab(?:s|bed|bing)?\b", r"\bpick(?:s|ed|ing)?\s+up\b",
        r"\bdrop(?:s|ped|ping)?\b", r"\bthrow(?:s|ing)?\b|\bthrew\b|\bthrown\b",
        r"\bhand(?:s|ed|ing)?\b", r"\bgiv(?:e|es|en|ing)\b|\bgave\b",
        r"\bdraw(?:s|ing)?\b|\bdrew\b|\bdrawn\b",
        r"\brais(?:e|es|ed|ing)\b", r"\blower(?:s|ed|ing)?\b",
        r"\bpull(?:s|ed|ing)?\b", r"\bpush(?:es|ed|ing)?\b",
        r"\benter(?:s|ed|ing)?\b", r"\bexit(?:s|ed|ing)?\b",
        r"\btransform(?:s|ed|ing)?\b", r"\bturn(?:s|ed|ing)\s+around\b",
        r"\bspin(?:s|ning)?\b|\bspun\b", r"\bwave(?:s|d|ing)?\b",
        r"\battack(?:s|ed|ing)?\b", r"\bstrik(?:e|es|ing)\b|\bstruck\b",
        r"\bhit(?:s|ting)?\b", r"\bfire(?:s|d|ing)?\b",
        r"\bshoot(?:s|ing)?\b|\bshot\b", r"\bblast(?:s|ed|ing)?\b",
        r"\bcharg(?:e|es|ed|ing)\b", r"\bbit(?:e|es|ing)\b|\bbite\b",
        r"\bclaw(?:s|ed|ing)?\b", r"\bswipe(?:s|d|ing)?\s+at\b",
        r"\bslash(?:es|ed|ing)?\b", r"\bstab(?:s|bed|bing)?\b",
        r"\bpunch(?:es|ed|ing)?\b", r"\bkick(?:s|ed|ing)?\b",
        r"\bsmash(?:es|ed|ing)?\b", r"\bdestroy(?:s|ed|ing)?\b",
        r"\bburn(?:s|ed|ing)?\b",
    )
)
_ACTION_CLAUSE_BREAK_RE = re.compile(
    r"[,;.!?]|\b(?:and\s+then|then|and|but|while|as|after|before|when|because)\b",
    re.I,
)
_ACTION_RELATION_RE = re.compile(
    r"\b(away\s+from|in\s+front\s+of|next\s+to|up\s+to|out\s+of|"
    r"towards?|across|along|around|behind|beneath|beside|between|beyond|"
    r"down|from|in|inside|into|near|onto|on|over|past|through|to|under|up)\b",
    re.I,
)
_BROAD_TRAVERSAL_RELATIONS = {"across", "along", "through"}
_TRAVERSAL_PLACE_RELATION_RE = re.compile(
    r"\b(away\s+from|in\s+front\s+of|next\s+to|up\s+to|out\s+of|"
    r"across|along|around|behind|beneath|beside|between|down|in|inside|"
    r"into|near|off|onto|on|over|past|through|towards?|under|up|within|at|to)\b",
    re.I,
)
_TRAVERSAL_CLAUSE_END_RE = re.compile(
    r"[;.!?]|\b(?:and\s+then|then|and|but|while|as|after|before|when|because)\b",
    re.I,
)
_CAMERA_AIM_TERMS = {
    "camera", "frame", "lens", "screen", "view", "viewfinder", "viewer", "vision", "shot",
}
_PLACE_NOT_SHOWN_RE = re.compile(
    r"\b(?:not|never|no\s+longer)\b.{0,32}\b(?:visible|shown|depicted|seen|present|"
    r"in\s+view|on\s+screen|in\s+frame|included)\b|"
    r"\b(?:absent|off\s*screen|excluded|omitted)\b",
    re.I,
)
_TRAVERSAL_PLACE_WORDS = {
    "arena", "avenue", "battlefield", "beach", "bridge", "building", "canyon",
    "castle", "city", "cliff", "coast", "coastline", "corridor", "desert",
    "earth", "field", "floor", "forest", "garden", "ground", "hall", "highway",
    "hill", "house", "island", "jungle", "lake", "lighthouse", "meadow", "mountain", "ocean",
    "park", "path", "platform", "plain", "prairie", "road", "room", "shore",
    "shoreline", "stage", "street", "stream", "terrain", "town", "trail", "tunnel",
    "tower", "valley", "village", "woods",
}
_TRAVERSAL_MEDIA_WORDS = {
    "cloud", "clouds", "debris", "dust", "fire", "fog", "haze", "leaves", "mist",
    "rubble", "smoke", "wind",
}
_ACTION_DESCRIPTION_WORDS = {
    "calm", "calmly", "careful", "carefully", "contemplative", "continuous",
    "deliberate", "deliberately", "determined", "gentle", "gently", "gradual",
    "gradually", "measured", "physical", "purposeful", "purposefully", "quiet",
    "reflective", "small", "slow", "slowly", "steady", "steadily", "thoughtful",
    "thoughtfully", "unhurried",
}
_PRONOUNS = {"he", "she", "it", "they", "them", "him", "her"}
_CONNECTOR_RE = re.compile(r"\b(?:and\s+then|then|and|but|while|as|after|before|when)\b", re.I)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|without|avoid|avoids|avoided|cannot|can['’]t|"
    r"don['’]t|doesn['’]t|didn['’]t|won['’]t|shouldn['’]t|no\s+longer)\b",
    re.IGNORECASE,
)
_STATE_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (re.compile(r"\b(?:riding|rides|rode)\b", re.I), False),
    (re.compile(
        r"\b(?:mounted|perched|seated|sitting|standing)\s+"
        r"(?:(?:already|still|securely|safely|firmly|comfortably)\s+){0,2}"
        r"(?:on|onto|atop|astride)\b", re.I,
    ), False),
    (re.compile(r"\b(?:perches|sits|stands)\s+(?:on|atop|astride)\b", re.I), False),
    (re.compile(r"\b(?:astride|aboard|atop)\b", re.I), False),
    (re.compile(r"\b(?:is|are|was|were|remains|remain|stays|stay)\s+(?:already\s+|still\s+)?(?:on|aboard|atop)\b", re.I), False),
    (re.compile(r"\b(?:carries|carrying|carried)\b", re.I), True),
    (re.compile(r"\b(?:carried\s+by)\b", re.I), False),
)


def _normal_text(value: Any) -> str:
    text = _SUBJECT_TAG_RE.sub(" ", str(value or ""))
    return " ".join(_WORD_RE.findall(text.casefold()))


def _mask_quoted_text(value: str) -> str:
    """Blank quoted dialogue without shifting source offsets."""

    patterns = (
        r'"[^"\n]*"',
        r"“[^”\n]*”",
        r"‘[^’\n]*’",
        r"(?<!\w)'[^'\n]+'(?!\w)",
    )
    masked = value
    for pattern in patterns:
        masked = re.sub(pattern, lambda match: " " * len(match.group(0)), masked)
    return masked


def _sentence_ranges(value: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _SENTENCE_RE.finditer(value)]


def _action_matches(value: str) -> list[tuple[int, int, str, str]]:
    found: list[tuple[int, int, str, str]] = []
    for kind, pattern in _ACTION_PATTERNS:
        for match in pattern.finditer(value):
            if not _is_filmmaking_action(value, match.start(), match.end(), match.group(0)):
                found.append((match.start(), match.end(), kind, match.group(0)))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, str, str]] = []
    for item in found:
        if any(item[0] < prior[1] and item[1] > prior[0] for prior in selected):
            continue
        selected.append(item)
    return selected


def _is_negated_before(value: str, start: int) -> bool:
    prefix = value[:start]
    tail = prefix[-96:]
    matches = list(_NEGATION_RE.finditer(tail))
    if not matches:
        return False
    # Use the closest preceding negation. Earlier unrelated negation should
    # not hide a later construction such as "never climbs ... and does not attack".
    match = matches[-1]
    # "not only" is an emphasis construction, not a request to omit the act.
    if re.search(r"\bnot\s+only\b", tail, re.I) and not re.search(
        r"\bnot\s+only\b.*\b(?:never|without|avoid|cannot|can['’]t|don['’]t)\b",
        tail,
        re.I,
    ):
        return False
    words_after = re.findall(r"[a-z0-9]+", tail[match.end():], re.I)
    return len(words_after) <= 6


def _is_nominal_action_reference(
    value: str,
    start: int,
    end: int,
    verb: str,
) -> bool:
    """Ignore gerunds used as references or timeline labels, not performed acts."""

    if not _normal_text(verb).endswith("ing"):
        return False
    prefix = value[max(0, start - 96):start]
    tail = value[end:min(len(value), end + 80)]
    if _ACTION_REFERENCE_HEAD_RE.match(tail):
        return True
    if _ACTION_REFERENCE_VERB_PREFIX_RE.search(prefix):
        return True
    if re.search(r"\b(?:before|after|during)\s*$", prefix, re.I):
        return bool(_DIRECTIVE_ACTION_REFERENCE_RE.search(prefix))
    return False


def _is_static_modifier_match(value: str, start: int, end: int, verb: str) -> bool:
    """Skip common adjective/noun phrases that resemble an action verb."""

    normalized_verb = _normal_text(verb)
    prefix = value[max(0, start - 64):start]
    tail = value[end:min(len(value), end + 72)]
    if normalized_verb in {"hand", "hands", "handed", "handing"}:
        if _BODY_PART_NOUN_PREFIX_RE.search(prefix) or re.search(
            r"\b[A-Za-z][\w’'-]*['’]s\s+"
            r"(?:(?:left|right|front|back|upper|lower)\s+)?$",
            prefix,
            re.I,
        ):
            return True
        if re.match(
            r"\s+(?:gesture|gestures|grip|grips|signal|signals|shape|shapes)\b",
            tail,
            re.I,
        ):
            return True
    if normalized_verb == "close" and re.match(r"\s*-\s*ups?\b", tail, re.I):
        return True
    if normalized_verb not in {"open", "closed", "raised", "burning", "landing", "close", "opening"}:
        return False
    tail_words = re.findall(r"[A-Za-z][\w'-]*", value[end:end + 72])
    if not tail_words:
        return False
    if normalized_verb == "raised" and tail_words[0].casefold() == "arms":
        return True
    # Look through a short adjective chain for unmistakable static scene or
    # body-part heads ("open grassy field", "raised arms", "landing lights").
    if tail_words[0].casefold() in {"a", "an", "the"}:
        return False  # "opens the gate" / "closed a door" are verb phrases.
    if normalized_verb == "open" and tail_words[0].casefold() in {"water", "waters"}:
        # "at golden-hour open water" is a setting description, while a
        # concrete opening action normally has an explicit object ("opens a
        # door") or a subject immediately before the verb.
        if re.search(
            r"\b(?:at|in|across|on|over|toward|towards|into|out\s+of|near|"
            r"by|around|through|off|along|from)\s+"
            r"(?:[A-Za-z][\w'-]*\s+){0,3}$",
            prefix,
            re.I,
        ):
            return True
    if not any(word.casefold() in _STATIC_MODIFIER_HEADS for word in tail_words[:5]):
        return False
    # Also recognize adjectival phrases introduced by an article or a common
    # scene preposition, without treating an actual finite subject+verb as one.
    return bool(re.search(
        r"\b(?:a|an|the)\s+$|\b(?:in|with|under|near|against|beside)\s+(?:(?:a|an|the)\s+)?$",
        prefix,
        re.IGNORECASE,
    ) or not prefix.strip())


def _entity_candidates(value: str) -> list[tuple[int, int, list[str]]]:
    tokens = list(re.finditer(r"[A-Za-z][\w’'-]*", value))
    action_starts = [start for start, _end, _kind, _verb in _action_matches(value)]
    candidates: list[tuple[int, int, list[str]]] = []
    for index, token in enumerate(tokens):
        if token.group(0).casefold() not in _DETERMINERS:
            continue
        words: list[str] = []
        end = token.end()
        for later in tokens[index + 1:index + 7]:
            word = later.group(0)
            if (
                word.casefold() in _ENTITY_STOP_WORDS
                or word.casefold() in _PERFORMANCE_VERB_WORDS
                or any(later.start() <= action_start < later.end() for action_start in action_starts)
            ):
                break
            words.append(word)
            end = later.end()
        aliases = _aliases_from_words(words)
        if aliases:
            candidates.append((token.start(), end, aliases))

    # Proper names have no article: keep short capitalized phrases as possible
    # cast aliases, while excluding sentence-initial style/camera labels above.
    for match in re.finditer(r"\b[A-Z][a-z0-9’'-]*(?:\s+[A-Z][a-z0-9’'-]*){0,2}\b", value):
        aliases = _aliases_from_words(match.group(0).split())
        if aliases:
            candidates.append((match.start(), match.end(), aliases))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates


def _aliases_from_words(words: list[str]) -> list[str]:
    cleaned = [word.strip("’'-").casefold() for word in words if word.strip("’'-")]
    while cleaned and (cleaned[0] in _DETERMINERS or cleaned[0] in _ENTITY_STOP_WORDS):
        cleaned.pop(0)
    while cleaned and cleaned[-1] in _ENTITY_STOP_WORDS:
        cleaned.pop()
    if not cleaned or cleaned[-1] in _GENERIC_ENTITY_HEADS or cleaned[-1] in _PRONOUNS:
        return []
    values = [" ".join(cleaned)]
    for count in (3, 2, 1):
        if len(cleaned) >= count:
            value = " ".join(cleaned[-count:])
            if value not in values:
                values.append(value)
    return values


def _select_entity(candidates: list[tuple[int, int, list[str]]]) -> list[str]:
    usable = [item for item in candidates if item[2]]
    return list(usable[-1][2]) if usable else []


def _actor_aliases(
    sentence: str,
    action_start: int,
    prior_actor: list[str],
    action_number: int,
) -> list[str]:
    prefix = sentence[:action_start]
    pronoun = re.search(r"\b(?:as|while|when|after|before|and\s+then|then)?\s*(it|he|she|they|them)\s*$", prefix, re.I)
    if pronoun:
        antecedents = _entity_candidates(prefix[:pronoun.start()])
        if antecedents:
            if pronoun.group(1).casefold() in {"they", "them"} and len(antecedents) > 1:
                aliases: list[str] = []
                for candidate in antecedents[-2:]:
                    aliases.extend(candidate[2])
                return _dedupe(aliases)
            return _select_entity(antecedents)

    if action_number == 0:
        directive = _DIRECTORIAL_SENTENCE_PREFIX_RE.match(sentence)
        if directive and directive.end() <= action_start:
            directed_subjects = _entity_candidates(prefix[directive.end():])
            for candidate in directed_subjects:
                if candidate[2]:
                    return candidate[2]

    candidates = _entity_candidates(prefix)
    if action_number > 0 and prior_actor:
        connectors = list(_CONNECTOR_RE.finditer(prefix))
        if connectors:
            after_connector = [item for item in candidates if item[0] >= connectors[-1].end()]
            if after_connector:
                return _select_entity(after_connector)
        return prior_actor

    # The first concrete action in a sentence is usually owned by its first
    # meaningful noun phrase. This avoids mistaking a direct object for the
    # actor in clauses such as "the kitten sees the dragon, then climbs on it".
    for candidate in candidates:
        if candidate[2]:
            return candidate[2]
    return []


def _target_aliases(sentence_tail: str) -> list[str]:
    tail = sentence_tail.lstrip()
    if not tail:
        return []
    stop = re.search(r"[,;.!?]|\b(?:and|then|while|as|after|before|when|because)\b", tail, re.I)
    candidate_text = tail[:stop.start()] if stop else tail
    entities = _entity_candidates(candidate_text)
    if entities:
        return _select_entity(entities)
    words = re.findall(r"[A-Za-z][\w’'-]*", candidate_text)
    while words and words[0].casefold() in _DETERMINERS:
        words.pop(0)
    for index, word in enumerate(words):
        if word.casefold() in _ENTITY_STOP_WORDS:
            words = words[:index]
            break
    return _aliases_from_words(words[:5])


def _event_text_end(
    sentence: str,
    action_end: int,
    next_action_start: int | None,
) -> int:
    if next_action_start is None:
        tail = sentence[action_end:]
        for separator in re.finditer(
            r"[,;]|\b(?:and\s+then|then|and|but|while|as|after|before|when)\b",
            tail,
            re.IGNORECASE,
        ):
            if _NON_MILESTONE_CLAUSE_RE.match(tail[separator.end():]):
                return action_end + separator.start()
        return len(sentence)
    between = sentence[action_end:next_action_start]
    separator = re.search(
        r"[,;]|\b(?:and\s+then|then|and|but|while|as|after|before|when)\b",
        between,
        re.IGNORECASE,
    )
    if separator:
        return action_end + separator.start()
    return next_action_start


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normal_text(value)
        if normalized and normalized not in seen and normalized not in _PRONOUNS:
            seen.add(normalized)
            result.append(value)
    return result


def build_music_narrative_contract(source: Any) -> dict[str, Any]:
    """Extract ordered, concrete music-video actions from the original brief."""

    original = str(source or "")
    # Work from the original source so creation-prefix clauses such as
    # "Also include ... as it swoops down and breathes fire" remain visible.
    # This small parser is self-contained instead of importing H3's event
    # splitter, which can discard action clauses attached to directives.
    visible_source = _mask_quoted_text(original)
    events: list[dict[str, Any]] = []
    for sentence_start, sentence_end in _sentence_ranges(original):
        sentence = original[sentence_start:sentence_end]
        visible_sentence = visible_source[sentence_start:sentence_end]
        matches = [
            match for match in _action_matches(visible_sentence)
            if not _is_static_modifier_match(
                visible_sentence, match[0], match[1], match[3],
            )
            and not _is_nominal_action_reference(
                visible_sentence, match[0], match[1], match[3],
            )
        ]
        prior_actor: list[str] = []
        for action_number, (action_start, action_end, kind, verb_text) in enumerate(matches):
            if _is_negated_before(visible_sentence, action_start):
                continue
            actor_aliases = _actor_aliases(
                visible_sentence, action_start, prior_actor, action_number,
            )
            prior_actor = actor_aliases or prior_actor
            tail = sentence[action_end:]
            following = [item[0] for item in matches if item[0] > action_start]
            next_action_start = following[0] if following else None
            text_end = _event_text_end(sentence, action_end, next_action_start)
            action_text = sentence[action_start:text_end].strip(" \t\r\n.,;:!?-")
            if not action_text:
                continue

            event: dict[str, Any] = {
                "event_id": f"M{len(events) + 1}",
                "text": action_text,
                "kind": kind,
            }
            if actor_aliases:
                event["actor_aliases"] = _dedupe(actor_aliases)
            target_aliases = _target_aliases(tail) if kind == "boarding" else []
            if target_aliases:
                event["target_aliases"] = _dedupe(target_aliases)
            events.append(event)

    return {"revision": _REVISION, "source": original, "events": events}


def _contract_events(contract: Any) -> list[dict[str, Any]]:
    if not isinstance(contract, dict) or not isinstance(contract.get("events"), list):
        return []
    return [event for event in contract["events"] if isinstance(event, dict)]


def validate_music_narrative_allocation(
    contract: Any,
    allocation: Any,
    clip_count: Any,
) -> list[str]:
    """Check exact-once, source-ordered allocation to valid zero-based clips."""

    issues: list[str] = []
    events = _contract_events(contract)
    expected_ids = [str(event.get("event_id") or "") for event in events]
    if not isinstance(clip_count, int) or isinstance(clip_count, bool) or clip_count < 0:
        return ["music narrative clip_count must be a nonnegative integer"]
    if not isinstance(allocation, list):
        return ["music narrative allocation must be a list"]

    assigned_ids: list[str] = []
    indexes: list[int] = []
    seen: set[str] = set()
    valid_ids = set(expected_ids)
    for offset, item in enumerate(allocation):
        if not isinstance(item, dict):
            issues.append(f"allocation item {offset + 1} must be an object")
            continue
        event_id = item.get("event_id")
        clip_index = item.get("clip_index")
        if not isinstance(event_id, str):
            issues.append(f"allocation item {offset + 1} needs a string event_id")
            continue
        assigned_ids.append(event_id)
        if event_id not in valid_ids:
            issues.append(f"allocation contains unknown music narrative event {event_id!r}")
        if event_id in seen:
            issues.append(f"music narrative event {event_id!r} is allocated more than once")
        seen.add(event_id)
        if not isinstance(clip_index, int) or isinstance(clip_index, bool):
            issues.append(f"music narrative event {event_id!r} clip_index must be an integer")
            continue
        indexes.append(clip_index)
        if clip_index < 0 or clip_index >= clip_count:
            issues.append(
                f"music narrative event {event_id!r} clip_index {clip_index} "
                f"is outside 0..{max(clip_count - 1, 0)}"
            )

    if assigned_ids != expected_ids:
        issues.append(
            "music narrative allocation must include every event exactly once "
            "in source order"
        )
    if len(indexes) == len(allocation) and any(
        later < earlier for earlier, later in zip(indexes, indexes[1:])
    ):
        issues.append("music narrative clip assignments must be nondecreasing in source order")
    return issues


def _aliases(event: dict[str, Any], key: str) -> list[str]:
    raw = event.get(key)
    values = raw if isinstance(raw, list) else []
    return _dedupe([str(value) for value in values if str(value).strip()])


def _allocation_map(allocation: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(allocation, list):
        return result
    for item in allocation:
        if not isinstance(item, dict):
            continue
        event_id = item.get("event_id")
        clip_index = item.get("clip_index")
        if isinstance(event_id, str) and isinstance(clip_index, int) and not isinstance(clip_index, bool):
            result[event_id] = clip_index
    return result


def _native_sections(value: Any) -> dict[str, str]:
    text = str(value or "")
    matches = list(_NATIVE_SECTION_RE.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).casefold()] = text[match.end():stop].strip()
    return sections


def _row_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("video_prompt") or "")
    native_values = [prompt]
    native_values.extend(
        str(row.get(key))
        for key in ("native_ir", "video_ir", "prompt_ir", "_director_native_ir")
        if isinstance(row.get(key), str) and row.get(key).strip()
    )
    native_markers = {"subject_definitions", "summary", "retention_analysis", "detailed_description"}
    for value in native_values:
        sections = _native_sections(value)
        if not sections:
            continue
        if native_markers.intersection(sections):
            detail = row.get("detailed_description")
            if isinstance(detail, str):
                return detail
            return sections.get("detailed_description", "")

        # Some legacy prompts end with only audio metadata. They are not a
        # full Context-IR document: retain the preceding visual prompt so the
        # narrative check still examines what the writer actually described.
        if value == prompt:
            audio_matches = [
                match for match in _NATIVE_SECTION_RE.finditer(value)
                if match.group(1).casefold() in {"overall_soundscape", "non_diegetic_music"}
            ]
            if audio_matches:
                return value[:audio_matches[0].start()].strip()

    explicit_native_ir = any(
        row.get(key) is not None
        for key in ("native_ir", "video_ir", "prompt_ir", "_director_native_ir")
    )
    if explicit_native_ir:
        detail = row.get("detailed_description")
        if isinstance(detail, str):
            return detail
        return ""
    return prompt


def _identity_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    subjects = row.get("subjects_on_screen")
    identity_fields = (
        "character_id", "speaker_name", "visual_description", "position_or_relation",
        "identity", "identity_description", "description",
    )
    if isinstance(subjects, list):
        for subject in subjects:
            if isinstance(subject, dict):
                texts.extend(
                    str(subject.get(field))
                    for field in identity_fields
                    if isinstance(subject.get(field), str) and subject[field].strip()
                )
    for key in ("subject_definitions", "subject_defs", "compiled_subject_definitions"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)
        elif isinstance(value, dict):
            texts.extend(str(item) for item in value.values() if isinstance(item, str) and item.strip())
        elif isinstance(value, list):
            texts.extend(str(item) for item in value if isinstance(item, str) and item.strip())
            texts.extend(
                str(item[field])
                for item in value if isinstance(item, dict)
                for field in identity_fields
                if isinstance(item.get(field), str) and item[field].strip()
            )
    for value in (
        row.get("video_prompt"), row.get("native_ir"), row.get("video_ir"),
        row.get("prompt_ir"), row.get("_director_native_ir"),
    ):
        definition = _native_sections(value).get("subject_definitions")
        if definition:
            texts.append(definition)
    return texts


def _contains_alias(text: str, aliases: list[str]) -> bool:
    padded = f" {_normal_text(text)} "
    return any(f" {_normal_text(alias)} " in padded for alias in aliases if _normal_text(alias))


def _state_relation_starts(
    text: str,
    actor_aliases: list[str],
    target_aliases: list[str],
) -> list[int]:
    """Find mounted/riding states only when both requested entities match."""

    if not actor_aliases:
        return []
    starts: list[int] = []
    offset = 0
    for start, end in _sentence_ranges(text):
        sentence = text[start:end]
        normalized = _normal_text(sentence)
        for pattern, reversed_relation in _STATE_PATTERNS:
            for match in pattern.finditer(normalized):
                if _is_negated_before(normalized, match.start()):
                    continue
                before = normalized[max(0, match.start() - 100):match.start()]
                after = normalized[match.end():min(len(normalized), match.end() + 100)]
                if reversed_relation:
                    actors_match = _contains_alias(after, actor_aliases)
                    targets_match = not target_aliases or _contains_alias(before, target_aliases)
                else:
                    actors_match = _contains_alias(before, actor_aliases)
                    # Bind the requested target to the object of this physical
                    # relation. A nearby later mention after "beside/near/with"
                    # must not turn standing on the ground into riding the dragon.
                    target_clause = re.split(
                        r"\b(?:beside|near|with|without|while|whereas|but|and\s+then|"
                        r"instead\s+of|away\s+from|in\s+front\s+of|behind|under|"
                        r"beneath|above|across)\b",
                        after,
                        maxsplit=1,
                    )[0]
                    targets_match = not target_aliases or _contains_alias(target_clause, target_aliases)
                if actors_match and targets_match:
                    starts.append(offset + match.start())
        offset += len(normalized) + 1
    return sorted(set(starts))


def _sentence_for_position(text: str, position: int) -> tuple[str, int] | None:
    offset = 0
    for start, end in _sentence_ranges(text):
        sentence = text[start:end]
        normalized = _normal_text(sentence)
        if offset <= position <= offset + len(normalized):
            return normalized, position - offset
        offset += len(normalized) + 1
    return None


def _action_positions(
    prompt: str,
    event: dict[str, Any],
    *,
    require_actor: bool = True,
) -> list[int]:
    action_text = _normal_text(event.get("text"))
    analysis_prompt = _mask_quoted_text(prompt)
    if not action_text:
        return []
    positions: list[int] = []
    ranges = _sentence_ranges(analysis_prompt)
    sentences = [_normal_text(analysis_prompt[start:end]) for start, end in ranges]
    actor_aliases = _aliases(event, "actor_aliases")
    offset = 0
    for sentence_index, sentence in enumerate(sentences):
        cursor = 0
        while True:
            position = sentence.find(action_text, cursor)
            if position < 0:
                break
            if not _is_negated_before(sentence, position) and (
                not require_actor or not actor_aliases or _semantic_actor_matches(
                    sentences, sentence_index, sentence, position, actor_aliases,
                )
            ):
                positions.append(offset + position)
            cursor = position + max(1, len(action_text))
        offset += len(sentence) + 1
    return positions


def _event_semantic_family(event: dict[str, Any]) -> re.Pattern[str] | None:
    source_text = _normal_text(event.get("text"))
    for _family, pattern in _SEMANTIC_VERB_FAMILIES:
        if pattern.search(source_text):
            return pattern
    return None


def _last_clause_prefix(sentence: str, end: int) -> str:
    prefix = sentence[:end]
    separators = list(re.finditer(
        r"[,;:]|\b(?:and\s+then|then|while|as|after|before|when)\b",
        prefix,
        re.I,
    ))
    return prefix[separators[-1].end():] if separators else prefix


def _preceding_clause_context(sentence: str, end: int) -> str:
    prefix = sentence[:end]
    separators = list(re.finditer(
        r"[,;:]|\b(?:and\s+then|then|while|as|after|before|when)\b",
        prefix,
        re.I,
    ))
    return prefix[:separators[-1].start()] if separators else ""


def _semantic_actor_matches(
    sentences: list[str],
    sentence_index: int,
    sentence: str,
    action_start: int,
    aliases: list[str],
) -> bool:
    if not aliases:
        return False
    clause_prefix = _last_clause_prefix(sentence, action_start)
    if _pronoun_subject_matches(
        sentences, sentence_index, sentence, action_start, aliases,
    ):
        return True
    candidates = _entity_candidates(clause_prefix)
    if candidates:
        # The last explicit noun phrase before the verb is the safest small
        # heuristic for distinguishing "the kitten climbs" from "the puppy
        # climbs while the kitten watches".
        return any(
            _contains_alias(" ".join(candidate[2]), aliases)
            for candidate in candidates[-1:]
        )

    pronoun = re.search(r"\b(?:it|he|she|they|them)\s*$", clause_prefix, re.I)
    if pronoun:
        # Retain the existing simple-pronoun continuation path. The helper
        # above handles compound clauses whose object phrases previously
        # hid their leading pronoun; it does not tighten unrelated prose.
        prior = " ".join((
            *sentences[max(0, sentence_index - 2):sentence_index],
            _preceding_clause_context(sentence, action_start),
        ))
        return _contains_alias(prior, aliases)

    # A verb may inherit the current sentence's subject after "then". Permit
    # that only when the requested actor is explicit earlier in this sentence.
    return _contains_alias(sentence[:action_start], aliases)


def _direct_wrong_boarding_target(
    sentence: str,
    action_end: int,
    target_aliases: list[str],
) -> bool:
    if not target_aliases:
        return False
    tail = sentence[action_end:]
    object_clause = re.split(
        r"[;.!?]|\b(?:and\s+then|then|and|but|while|as|after|before|when|"
        r"beside|near|with|under|behind|beneath|above|across)\b",
        tail,
        maxsplit=1,
        flags=re.I,
    )[0]
    relation = re.search(r"\b(?:on|onto|atop|aboard|into|in)\s+", object_clause, re.I)
    if not relation:
        return False
    object_phrase = object_clause[relation.end():].strip()
    first_word = _normal_text(object_phrase).split(" ", 1)[0] if _normal_text(object_phrase) else ""
    if first_word in {"it", "him", "her", "them"}:
        return False
    return not _contains_alias(object_phrase, target_aliases)


_CLIMB_CONTACT_RE = re.compile(
    r"\b(?:plant(?:s|ed|ing)?|grip(?:s|ped|ping)?|grasp(?:s|ed|ing)?|"
    r"brace(?:s|d|ing)?|clutch(?:es|ed|ing)?|cling(?:s|ing)?|"
    r"hook(?:s|ed|ing)?|press(?:es|ed|ing)?)\b",
    re.I,
)
_CLIMB_CONTACT_STOP_RE = re.compile(
    r"[,;.!?]|\b(?:and\s+then|then|and|but|while|as|after|before|when|"
    r"beside|near|instead\s+of|away\s+from|behind|beneath|above|across)\b",
    re.I,
)
_CLIMB_SELF_MOVE_RE = re.compile(
    r"\b(?:pull(?:s|ed|ing)?|haul(?:s|ed|ing)?|hoist(?:s|ed|ing)?)\s+"
    r"(?:(?:itself|himself|herself|themselves)|"
    r"(?:its|his|her|their)\s+(?:body|torso|paws?|legs?|feet|limbs))\b"
    r".{0,28}\b(?:up|upward|upwards|higher|onto|on\s+top\s+of)\b",
    re.I,
)
_CLIMB_BODY_UP_RE = re.compile(
    r"\b(?:(?:its|his|her|their)\s+)?(?:body|torso|paws?|legs?|feet|limbs)\s+"
    r"(?:twist(?:s|ed|ing)?|mov(?:e|es|ed|ing)|shift(?:s|ed|ing)|"
    r"ris(?:e|es|en|ing)|reach(?:es|ed|ing)|push(?:es|ed|ing)|"
    r"lift(?:s|ed|ing))\b.{0,20}\b(?:up|upward|upwards|higher)\b|"
    r"\b(?:twist(?:s|ed|ing)?|mov(?:e|es|ed|ing)|shift(?:s|ed|ing)|"
    r"ris(?:e|es|en|ing)|reach(?:es|ed|ing)|push(?:es|ed|ing)|"
    r"lift(?:s|ed|ing))\s+(?:its|his|her|their)\s+"
    r"(?:body|torso|paws?|legs?|feet|limbs)\b.{0,20}\b(?:up|upward|upwards|higher)\b",
    re.I,
)
_CLIMB_NON_ACTOR_HEADS = {
    "armor", "armour", "body", "feet", "forearm", "foreleg", "hindleg",
    "joint", "leg", "limb", "neck", "paw", "plating", "plate", "paws",
    "saddle", "scale", "scales", "shoulder", "snout", "stance", "torso",
}
_PRONOUN_NON_ACTOR_HEADS = _GENERIC_ENTITY_HEADS | _CLIMB_NON_ACTOR_HEADS


def _pronoun_subject_matches(
    sentences: list[str],
    sentence_index: int,
    sentence: str,
    action_start: int,
    aliases: list[str],
) -> bool:
    """Bind a leading clause pronoun only when its local antecedent is clear.

    Object phrases such as ``the dragon's scaled leg`` can follow an actor
    without replacing that actor as the subject of the next ``it`` clause.
    Keep the binding local and reject it when another plausible entity could
    own the pronoun.
    """

    clause_prefix = _last_clause_prefix(sentence, action_start)
    if not re.match(r"\s*(?:it|he|she|they|them)\b", clause_prefix, re.I):
        return False

    prior = _preceding_clause_context(sentence, action_start)
    if not prior and sentence_index > 0:
        prior = sentences[sentence_index - 1]
    if not prior:
        return False

    candidates = _entity_candidates(_normal_text(prior))
    plausible: list[str] = []
    for _start, _end, candidate_aliases in candidates:
        candidate = " ".join(candidate_aliases)
        words = _normal_text(candidate_aliases[0]).split()
        if not words:
            continue
        if (
            words[-1] in _PRONOUN_NON_ACTOR_HEADS
            and not _contains_alias(candidate, aliases)
        ):
            continue
        plausible.append(candidate)

    return bool(plausible) and all(
        _contains_alias(candidate, aliases) for candidate in plausible
    )


def _latest_non_target_actor_matches(
    sentence: str,
    action_start: int,
    actor_aliases: list[str],
    target_aliases: list[str],
) -> bool:
    """Resolve an implied subject such as ``pulling its body upward`` conservatively."""

    prefix = _preceding_clause_context(sentence, action_start)
    candidates = _entity_candidates(prefix)
    meaningful: list[list[str]] = []
    for _start, _end, aliases in candidates:
        phrase = " ".join(aliases)
        if target_aliases and _contains_alias(phrase, target_aliases):
            continue
        words = _normal_text(phrase).split()
        if not words or words[-1] in _CLIMB_NON_ACTOR_HEADS:
            continue
        meaningful.append(aliases)
    if not meaningful:
        return False
    return _contains_alias(" ".join(meaningful[-1]), actor_aliases)


def _climb_compound_actor_matches(
    sentences: list[str],
    sentence_index: int,
    sentence: str,
    action_start: int,
    actor_aliases: list[str],
    target_aliases: list[str],
) -> bool:
    if _semantic_actor_matches(
        sentences, sentence_index, sentence, action_start, actor_aliases,
    ):
        return True
    clause = _last_clause_prefix(sentence, action_start)
    if re.search(r"\b(?:its|his|her|their)\s+(?:body|torso|paws?|legs?|feet|limbs)\b", clause, re.I):
        return _latest_non_target_actor_matches(
            sentence, action_start, actor_aliases, target_aliases,
        )
    return False


def _climb_equivalent_positions(prompt: str, event: dict[str, Any]) -> list[int]:
    """Accept only contact with the named target plus upward self/body movement."""

    family = _event_semantic_family(event)
    if (
        str(event.get("kind") or "").casefold() != "boarding"
        or family is None
        or not family.search(_normal_text(event.get("text")))
    ):
        return []
    actor_aliases = _aliases(event, "actor_aliases")
    target_aliases = _aliases(event, "target_aliases")
    if not actor_aliases or not target_aliases:
        return []

    analysis_prompt = _mask_quoted_text(prompt)
    ranges = _sentence_ranges(analysis_prompt)
    sentences = [analysis_prompt[start:end] for start, end in ranges]
    positions: list[int] = []
    for sentence_index, (sentence_start, sentence_end) in enumerate(ranges):
        sentence = analysis_prompt[sentence_start:sentence_end]
        contacts: list[int] = []
        for contact in _CLIMB_CONTACT_RE.finditer(sentence):
            if _is_negated_before(sentence, contact.start()):
                continue
            if not _climb_compound_actor_matches(
                sentences, sentence_index, sentence, contact.start(),
                actor_aliases, target_aliases,
            ):
                continue
            tail = sentence[contact.end():]
            stop = _CLIMB_CONTACT_STOP_RE.search(tail)
            direct_object = tail[:stop.start()] if stop else tail
            if _contains_alias(direct_object, target_aliases):
                contacts.append(contact.start())
        if not contacts:
            continue

        for movement_pattern in (_CLIMB_SELF_MOVE_RE, _CLIMB_BODY_UP_RE):
            for movement in movement_pattern.finditer(sentence):
                if _is_negated_before(sentence, movement.start()):
                    continue
                if not _climb_compound_actor_matches(
                    sentences, sentence_index, sentence, movement.start(),
                    actor_aliases, target_aliases,
                ):
                    continue
                if _direct_wrong_boarding_target(
                    sentence, movement.end(), target_aliases,
                ):
                    continue
                # A contact clause must precede or accompany the upward move.
                if any(contact <= movement.start() for contact in contacts):
                    normalized_start = len(_normal_text(analysis_prompt[:sentence_start] + " " + sentence[:movement.start()]))
                    positions.append(normalized_start)
                    break
    return sorted(set(positions))


def _semantic_action_positions(prompt: str, event: dict[str, Any]) -> list[int]:
    """Accept bounded inflection changes while keeping actor/target grounded."""

    pattern = _event_semantic_family(event)
    if pattern is None:
        return []
    analysis_prompt = _mask_quoted_text(prompt)
    ranges = _sentence_ranges(analysis_prompt)
    sentences = [analysis_prompt[start:end] for start, end in ranges]
    actor_aliases = _aliases(event, "actor_aliases")
    target_aliases = _aliases(event, "target_aliases")
    positions: list[int] = []
    for sentence_index, (start, end) in enumerate(ranges):
        sentence = analysis_prompt[start:end]
        for match in pattern.finditer(sentence):
            if _is_negated_before(sentence, match.start()):
                continue
            if not _semantic_actor_matches(
                sentences, sentence_index, sentence, match.start(), actor_aliases,
            ):
                continue
            if target_aliases:
                if _direct_wrong_boarding_target(sentence, match.end(), target_aliases):
                    continue
                neighborhood = " ".join(sentences[sentence_index:sentence_index + 2])
                if not _contains_alias(neighborhood, target_aliases):
                    continue
            normalized_start = len(_normal_text(analysis_prompt[:start] + " " + sentence[:match.start()]))
            positions.append(normalized_start)
    return positions


def _generic_semantic_actor_matches(
    sentences: list[str],
    sentence_index: int,
    sentence: str,
    action_start: int,
    actor_aliases: list[str],
) -> bool:
    if _semantic_actor_matches(
        sentences, sentence_index, sentence, action_start, actor_aliases,
    ):
        return True

    # A source verb can appear as a noun in a generated phrase such as
    # "the singer begins a slow, measured walk". In that construction the
    # parser's last article-led candidate is an adjective stack, not a new
    # actor. Ignore only these known descriptive modifiers and retain the
    # nearest plausible noun phrase for wrong-actor rejection.
    clause_prefix = _last_clause_prefix(sentence, action_start)
    candidates = _entity_candidates(clause_prefix)
    plausible = []
    for candidate in candidates:
        candidate_prefix = clause_prefix[:candidate[0]]
        if re.search(
            r"\b(?:from|in|on|onto|to|near|beside|by|with|under|over|through|"
            r"at|into|behind|beneath|between|off|around|across|along)\s+$",
            candidate_prefix,
            re.I,
        ):
            continue
        words = _normal_text(" ".join(candidate[2])).split()
        if words and all(word in _ACTION_DESCRIPTION_WORDS for word in words):
            continue
        plausible.append(candidate)
    if plausible:
        return _contains_alias(" ".join(plausible[-1][2]), actor_aliases)
    return False


def _relation_before_action_anchor(text: str, anchors: list[str]) -> str:
    normalized = _normal_text(text)
    anchor_positions = [normalized.find(_normal_text(anchor)) for anchor in anchors]
    anchor_positions = [position for position in anchor_positions if position >= 0]
    if not anchor_positions:
        return ""
    prefix = normalized[:min(anchor_positions)]
    matches = list(_ACTION_RELATION_RE.finditer(prefix))
    return _normal_text(matches[-1].group(0)) if matches else ""


def _semantic_action_clause_tail(tail: str) -> str:
    for separator in _ACTION_CLAUSE_BREAK_RE.finditer(tail):
        following = tail[separator.end():].lstrip()
        if separator.group(0).casefold() == "and":
            starts_another_clause = _NON_MILESTONE_CLAUSE_RE.match(following) or any(
                pattern.match(following) for _kind, pattern in _ACTION_PATTERNS
            )
            if not starts_another_clause:
                continue
        return tail[:separator.start()]
    return tail


def _traversal_clause_tail(tail: str) -> str:
    """Keep comma-separated place modifiers while dropping later clauses."""

    separator = _TRAVERSAL_CLAUSE_END_RE.search(tail)
    return tail[:separator.start()] if separator else tail


def _place_is_visually_established(prompt: str, anchors: list[str]) -> bool:
    """Require a positive, on-screen mention of a broad traversal setting."""

    for start, end in _sentence_ranges(prompt):
        sentence = _normal_text(prompt[start:end])
        for anchor in anchors:
            normalized_anchor = _normal_text(anchor)
            if not normalized_anchor:
                continue
            pattern = re.compile(rf"\b{re.escape(normalized_anchor)}\b")
            for match in pattern.finditer(sentence):
                local = sentence[max(0, match.start() - 44):min(len(sentence), match.end() + 58)]
                if _is_negated_before(sentence, match.start()):
                    continue
                if _PLACE_NOT_SHOWN_RE.search(local):
                    continue
                if re.search(r"\bno\s+(?:[a-z0-9]+\s+){0,2}$", sentence[:match.start()]):
                    continue
                return True
    return False


def _has_conflicting_traversal_route(
    action_tail: str,
    source_relation: str,
    anchors: list[str],
) -> bool:
    """Reject a different named route while ignoring camera aim and atmosphere."""

    clause = _traversal_clause_tail(action_tail)
    relations = list(_TRAVERSAL_PLACE_RELATION_RE.finditer(clause))
    for index, relation_match in enumerate(relations):
        relation = _normal_text(relation_match.group(0))
        next_start = relations[index + 1].start() if index + 1 < len(relations) else len(clause)
        target = clause[relation_match.end():next_start].strip(" \t\r\n,.:—–-")
        if not target:
            continue
        if _contains_alias(target, list(_CAMERA_AIM_TERMS)):
            continue

        target_words = set(_WORD_RE.findall(target.casefold()))
        target_has_source_place = _contains_alias(target, anchors)
        if target_has_source_place:
            if relation != source_relation:
                return True
            continue

        has_named_place = bool(target_words & _TRAVERSAL_PLACE_WORDS)
        has_atmospheric_medium = bool(target_words & _TRAVERSAL_MEDIA_WORDS)
        if relation in _BROAD_TRAVERSAL_RELATIONS:
            if has_named_place and not has_atmospheric_medium:
                return True
        elif relation in {"away from", "off", "out of", "into", "onto", "toward", "towards"}:
            if has_named_place:
                return True
    return False


def _generic_semantic_action_positions(prompt: str, event: dict[str, Any]) -> list[int]:
    """Allow same-verb inflections/adverbs while grounding actor and object.

    This fallback deliberately stays within a finite inventory of source
    action verbs. It requires the requested actor, preserves a meaningful
    source object or place when present, and never handles boarding events,
    which retain their stricter target/contact checks.
    """

    if str(event.get("kind") or "").casefold() == "boarding":
        return []
    source_text = _normal_text(event.get("text"))
    source_pattern = next(
        (pattern for pattern in _GENERIC_SEMANTIC_VERB_FAMILIES if pattern.match(source_text)),
        None,
    )
    actor_aliases = _aliases(event, "actor_aliases")
    if source_pattern is None or not actor_aliases:
        return []
    source_match = source_pattern.match(source_text)
    if source_match is None:
        return []
    anchors = _target_aliases(source_text[source_match.end():])

    analysis_prompt = _mask_quoted_text(prompt)
    ranges = _sentence_ranges(analysis_prompt)
    sentences = [_normal_text(analysis_prompt[start:end]) for start, end in ranges]
    positions: list[int] = []
    for sentence_index, (start, end) in enumerate(ranges):
        sentence = sentences[sentence_index]
        for match in source_pattern.finditer(sentence):
            if _is_negated_before(sentence, match.start()):
                continue
            if not _generic_semantic_actor_matches(
                sentences, sentence_index, sentence, match.start(), actor_aliases,
            ):
                continue
            tail = sentence[match.end():]
            action_tail = _semantic_action_clause_tail(tail)
            source_relation = _relation_before_action_anchor(
                source_text[source_match.end():], anchors,
            )
            if anchors:
                if _contains_alias(action_tail, anchors):
                    if source_relation and _relation_before_action_anchor(
                        action_tail, anchors,
                    ) != source_relation:
                        continue
                    if (
                        source_relation in _BROAD_TRAVERSAL_RELATIONS
                        and _has_conflicting_traversal_route(
                            tail, source_relation, anchors,
                        )
                    ):
                        continue
                elif (
                    source_relation in _BROAD_TRAVERSAL_RELATIONS
                    and _contains_alias(_traversal_clause_tail(tail), anchors)
                ):
                    if _relation_before_action_anchor(
                        _traversal_clause_tail(tail), anchors,
                    ) != source_relation:
                        continue
                elif (
                    source_relation in _BROAD_TRAVERSAL_RELATIONS
                    and _place_is_visually_established(analysis_prompt, anchors)
                    and not _has_conflicting_traversal_route(
                        tail, source_relation, anchors,
                    )
                ):
                    pass
                else:
                    continue
            normalized_start = len(_normal_text(analysis_prompt[:start] + " " + sentence[:match.start()]))
            positions.append(normalized_start)
    return positions


def _event_coverage_positions(prompt: str, event: dict[str, Any]) -> list[int]:
    literal_positions = _action_positions(prompt, event)
    semantic_positions = _semantic_action_positions(prompt, event)
    generic_semantic_positions = _generic_semantic_action_positions(prompt, event)
    if literal_positions or semantic_positions or generic_semantic_positions:
        return literal_positions or semantic_positions or generic_semantic_positions
    return _climb_equivalent_positions(prompt, event)


_SINGER_ROLE_ALIASES = {"singer", "vocalist", "lead singer", "lead vocalist"}


def _event_with_row_actor_identity(
    row: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    """Bridge a requested singer role to one explicitly declared vocalist ID.

    Director rows use stable semantic IDs (for example ``lead_performer``),
    while source actions name the person by role (for example ``the singer``).
    Only an unambiguous, explicitly vocalist-role subject can connect those
    forms. A merely single visible subject, or one of several vocalists, is
    not enough to infer that identity.
    """

    source_aliases = _aliases(event, "actor_aliases")
    if not source_aliases or not {
        _normal_text(alias) for alias in source_aliases
    }.issubset(_SINGER_ROLE_ALIASES):
        return event

    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        return event
    vocalist_ids = {
        subject.get("character_id", "").strip()
        for subject in subjects
        if isinstance(subject, dict)
        and _normal_text(subject.get("performance_role") or subject.get("role")) == "vocalist"
        and isinstance(subject.get("character_id"), str)
        and subject["character_id"].strip()
    }
    if len(vocalist_ids) != 1:
        return event

    actor_aliases = _dedupe(source_aliases + [next(iter(vocalist_ids))])
    return {**event, "actor_aliases": actor_aliases}


def _is_pending_boarding_relation(
    text: str,
    event: dict[str, Any],
) -> list[int]:
    if str(event.get("kind") or "").casefold() != "boarding":
        return []
    return _state_relation_starts(
        text,
        _aliases(event, "actor_aliases"),
        _aliases(event, "target_aliases"),
    )


def music_narrative_row_issue(
    row: Any,
    *,
    contract: Any,
    allocation: Any,
    index: Any,
) -> str | None:
    """Validate this clip's assignment, literal action, actor, and board state."""

    events = _contract_events(contract)
    if not events:
        return None
    if not isinstance(row, dict):
        return "music narrative row must be an object"
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        return "music narrative row index must be a nonnegative integer"

    assignments = _allocation_map(allocation)
    expected = [
        str(event.get("event_id") or "")
        for event in events
        if assignments.get(str(event.get("event_id") or "")) == index
    ]
    source_ids = row.get("source_event_ids")
    if not isinstance(source_ids, list) or source_ids != expected:
        return (
            f"clip {index + 1} source_event_ids must exactly match its assigned "
            f"music narrative events {expected!r}"
        )

    prompt = _row_prompt(row)
    identity_texts = _identity_texts(row)
    event_by_id = {str(event.get("event_id") or ""): event for event in events}

    # A future milestone may be described in global context, but it must not
    # already be performed in this clip's actual visual prompt.
    for event in events:
        event_id = str(event.get("event_id") or "")
        assigned_index = assignments.get(event_id)
        row_event = _event_with_row_actor_identity(row, event)
        if assigned_index is not None and index < assigned_index and _event_coverage_positions(prompt, row_event):
            return (
                f"clip {index + 1} performs future music narrative event {event_id} "
                f"before its assigned clip {assigned_index + 1}"
            )

    # A future independent cut cannot begin from a physical result that has
    # not yet been assigned. On the boarding clip, the same relation is allowed
    # only after its literal climbing/boarding action appears.
    for event in events:
        event_id = str(event.get("event_id") or "")
        board_index = assignments.get(event_id)
        if str(event.get("kind") or "").casefold() != "boarding" or board_index is None or index > board_index:
            continue
        if any(_is_pending_boarding_relation(text, event) for text in identity_texts):
            return (
                f"clip {index + 1} identity definitions start the requested actor "
                f"already riding or mounted before boarding event {event_id}"
            )
        relations = _is_pending_boarding_relation(prompt, event)
        if not relations:
            continue
        if index < board_index:
            return (
                f"clip {index + 1} starts the requested actor already riding or "
                f"mounted before pending boarding event {event_id}; an independent "
                "cut must preserve the unboarded state"
            )
        action_positions = _event_coverage_positions(prompt, event)
        if not action_positions or any(
            relation_start < action_positions[0] for relation_start in relations
        ):
            return (
                f"clip {index + 1} opens with the requested actor already riding "
                f"or mounted before visibly performing boarding event {event_id}"
            )

    assigned_positions: list[tuple[str, int]] = []
    for event_id in expected:
        event = event_by_id.get(event_id)
        if event is None:
            continue
        row_event = _event_with_row_actor_identity(row, event)
        positions = _event_coverage_positions(prompt, row_event)
        if not positions:
            literal_positions = _action_positions(prompt, event, require_actor=False)
            actor_aliases = _aliases(event, "actor_aliases")
            if literal_positions and actor_aliases:
                return (
                    f"clip {index + 1} must show the requested actor "
                    f"{actor_aliases[0]!r} before assigned source event {event_id}"
                )
            return (
                f"clip {index + 1} must visibly perform assigned source event "
                f"{event_id} using its source verb family with the requested actor"
                + (f" and target {_aliases(event, 'target_aliases')[0]!r}" if _aliases(event, "target_aliases") else "")
                + f": {event.get('text', '')}"
            )
        assigned_positions.append((event_id, positions[0]))
    if any(
        later_position < earlier_position
        for (_, earlier_position), (_, later_position)
        in zip(assigned_positions, assigned_positions[1:])
    ):
        return f"clip {index + 1} assigned music narrative actions appear out of source order"
    return None


def format_music_narrative_context(
    contract: Any,
    allocation: Any,
    start: Any,
    end: Any,
) -> str:
    """Format completed, assigned, and pending milestones for one clip batch.

    ``start`` and ``end`` use a zero-based half-open clip range.
    """

    events = _contract_events(contract)
    if not events:
        return ""
    mapping = _allocation_map(allocation)
    try:
        first = max(0, int(start))
        stop = max(first, int(end))
    except (TypeError, ValueError, OverflowError):
        first, stop = 0, 0

    completed: list[str] = []
    assigned: list[str] = []
    pending: list[str] = []
    per_clip: list[str] = []
    event_by_id = {str(event.get("event_id") or ""): event for event in events}
    for event in events:
        event_id = str(event.get("event_id") or "")
        clip_index = mapping.get(event_id)
        line = f"{event_id} ({event.get('kind', 'action')}): {event.get('text', '')}"
        if clip_index is not None and clip_index < first:
            completed.append(line)
        elif clip_index is not None and first <= clip_index < stop:
            assigned.append(f"clip {clip_index + 1}: {line}")
        else:
            pending.append(line)
    for clip_index in range(first, stop):
        clip_events = [
            event for event in events
            if mapping.get(str(event.get("event_id") or "")) == clip_index
        ]
        clip_ids = [str(event.get("event_id") or "") for event in clip_events]
        line = f"clip {clip_index + 1}: source_event_ids must be exactly {clip_ids!r}"
        if clip_events:
            line += "; detailed_description must put the requested actor before each unchanged literal action clause: "
            line += "; ".join(
                f"{_aliases(event, 'actor_aliases')[0] if _aliases(event, 'actor_aliases') else 'the requested actor'} "
                f"{event.get('text', '')}"
                for event in clip_events
            )
        per_clip.append(line)

    def section(label: str, values: list[str]) -> str:
        body = "\n".join(f"- {value}" for value in values) if values else "- None"
        return f"{label}:\n{body}"

    return "\n\n".join((
        "MUSIC NARRATIVE CONTRACT — keep the source actions in order across all clips, including independent cuts.",
        section("Completed before this batch", completed),
        section("Assigned within this batch", assigned),
        section("Exact requirements for every global clip in this batch", per_clip),
        section("Pending after this batch", pending),
        "Each clip must copy its listed source_event_ids exactly, including an empty list. Visibly perform every assigned action in that clip's detailed_description before showing its result; include the correct actor before the literal source clause without paraphrasing it. Preserve each actor's identity when shown, but allow motivated cutaways, solo performance, reaction shots, POV and environmental inserts. Persistent physical facts remain true offscreen; they do not require every actor in every frame or replaying the action that established them. A cut cannot skip a pending climb, boarding, flight, landing, or attack. Keep the requested music style and ongoing vocals/performance alive while showing the assigned story action. Global scene context and identity descriptions do not count as action coverage.",
    ))
