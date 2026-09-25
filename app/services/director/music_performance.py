"""Keep audible vocals separate from the people shown in a music-video shot."""

from collections.abc import Mapping
import re


PERFORMANCE_ROLES = ("vocalist", "instrumentalist", "non_vocal", "non_performer")

MUSIC_PERFORMANCE_RULES = """MUSIC PERFORMANCE ROLES:
- Describe only people and objects assigned to the current shot. A soundtrack does not
  require a singer, band member, instrument, or any other person to appear on screen.
- Give each visible person a role based on the user's concept, explicit performer mapping,
  or reference: vocalist (sings in this shot), instrumentalist (plays without singing),
  non_vocal (dancer, listener, or other non-singing person), or non_performer.
  Do not infer a role from camera focus, musical energy, or an unknown voice.
- A visible assigned vocalist lip-syncs only their own audible part. Keep their lips relaxed
  and closed during instrumental intros, gaps, and pauses; detected vocals do not require
  that vocalist to be visible.
- A visible assigned instrumentalist plays with relaxed closed lips and active hands/body,
  without singing or mouthing lyrics. Wind and brass players use their natural embouchure.
- In a group shot, only explicitly assigned vocalists sing. Backing vocals, duets, and a
  singing instrumentalist are valid when assigned; do not invent extra singers.
- Preserve explicitly requested cheers, shouts, and other non-song expressions without
  transferring the soundtrack's vocals to that person. Preserve dance and other actions.
- Keep each visible person's role consistent across cuts. Match a singer's visible mouth
  movement only to that person's own audible part; leave the supplied soundtrack unchanged.
- Remove conflicting mouth actions from action_beats, ending_beat, image_prompt,
  video_prompt, and window_prompts while preserving the rest of the requested performance.
"""


_LEGACY_GLOBAL_DIRECTION = (
    "Vocal ownership stays with the assigned singer across camera cuts. "
    "Only an explicitly assigned vocalist lip-syncs, and only to their own "
    "audible vocal part. Non-singing guitarists, bassists and drummers keep "
    "their lips closed without mouthing lyrics, except for a non-singing "
    "expression explicitly requested by the user; their hands and bodies "
    "continue the instrumental performance. During an instrument-only "
    "cutaway with audible vocals, the singer continues off screen; do not transfer "
    "the vocal to the person in view or insert a singer into the shot."
)
_LEGACY_ACTIVITY_DIRECTIONS = (
    "This interval has no detected vocals: even the lead singer keeps relaxed "
    "closed lips throughout, moving or listening to the instrumental music. "
    "No singing, lip-sync, bellowing or invented vocal breath.",
    "Vocal activity in this interval is unconfirmed. Default to relaxed closed "
    "lips, including the lead singer; allow lyric-shaped mouth movement only "
    "when a voice is actually audible in the supplied audio. Guitar riffs and "
    "drum hits never drive the mouth. Do not invent shouts or vocal breaths.",
    "Vocals occur within this interval, not necessarily throughout it. The "
    "assigned singer closes their lips during instrumental gaps and starts "
    "lip-sync only when their actual vocal part enters.",
)


def strip_legacy_music_performance_direction(
    prompt,
    subjects=(),
    vocal_activity=None,
    *,
    project_context="",
):
    """Remove exact old role boilerplate while preserving surrounding prose."""
    text = str(prompt or "")
    for emitted in (_LEGACY_GLOBAL_DIRECTION, *_LEGACY_ACTIVITY_DIRECTIONS):
        text = text.replace(emitted, "")

    # Older compilation appended one canonical line for each role. Remove it
    # only when the current subject metadata lets us reconstruct that exact
    # emitted line; never consume an arbitrary sentence or its camera/action
    # prefix while trying to recognize saved boilerplate.
    for subject in subjects or ():
        role = _field(subject, "performance_role")
        description = re.sub(
            r"\s+", " ", str(_field(subject, "visual_description") or "")
        ).strip(" .")
        if not description:
            continue
        emitted = None
        if role == "vocalist":
            if _requested_expression(subject, project_context):
                emitted = (
                    f"{description} may perform the user's explicitly requested "
                    "expression; this never transfers the song's vocal to them."
                )
            else:
                emitted = (
                    f"Assigned visible vocalist: {description}; mouth movement "
                    "follows only their vocal part when audible."
                )
        elif role == "instrumentalist":
            if _requested_expression(subject, project_context):
                emitted = (
                    f"{description} may perform the user's explicitly requested "
                    "expression; this never transfers the song's vocal to them."
                )
            elif _WIND_PLAYER.search(description):
                emitted = (
                    f"{description} uses the instrument's embouchure without "
                    "singing or mouthing lyrics."
                )
            else:
                emitted = (
                    f"{description} keeps their mouth closed and does not sing, "
                    "mouth lyrics, or lip-sync; natural body movement continues."
                )
        elif role == "non_vocal":
            emitted = (
                f"{description} does not sing or mouth the lyrics; preserve any "
                "explicitly described non-singing expression or cheering."
            )
        if emitted:
            text = text.replace(emitted, "")

    # Collapse only horizontal whitespace created by removal. Preserve blank
    # lines that separate H3 prompt fields and references.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def music_performance_direction(subjects=(), vocal_activity=None, *, project_context=""):
    """Compile concise directions only for people explicitly visible in this shot."""
    parts = []
    for subject in subjects or ():
        def field(key):
            return subject.get(key) if isinstance(subject, Mapping) else getattr(subject, key, None)

        role = field("performance_role")
        description = re.sub(r"\s+", " ", str(field("visual_description") or "")).strip(" .")
        if not description or role not in PERFORMANCE_ROLES or role == "non_performer":
            continue
        requested_expression = _requested_expression(subject, project_context)
        if role == "vocalist":
            if vocal_activity in {"silent", "unknown"}:
                if requested_expression:
                    direction = (
                        f"{description} keeps relaxed closed lips except during the "
                        "explicitly requested non-song expression; do not lip-sync "
                        "unless their own vocal part is audible."
                    )
                else:
                    direction = (
                        f"{description} keeps relaxed closed lips through this interval; "
                        "no singing or lyric mouthing."
                    )
            else:
                direction = (
                    f"{description} lip-syncs only their own audible vocal part, "
                    "with relaxed closed lips during instrumental gaps."
                )
            if requested_expression:
                direction += " Keep that expression separate from their assigned song vocal."
            parts.append(direction)
        elif role == "instrumentalist" and _WIND_PLAYER.search(description):
            direction = (
                f"{description} uses the instrument's natural embouchure without "
                "singing or mouthing lyrics."
            )
            if requested_expression:
                direction += " Preserve the explicitly requested non-song expression separately from playing."
            parts.append(direction)
        elif role == "instrumentalist":
            if requested_expression:
                direction = (
                    f"{description} keeps relaxed closed lips except during the "
                    "explicitly requested non-song expression; never lip-sync to "
                    "source vocals, and keep hands and body active."
                )
            else:
                direction = (
                    f"{description} plays with relaxed closed lips, without singing or "
                    "mouthing lyrics; keep hands and body active."
                )
            parts.append(direction)
        else:
            direction = (
                f"{description} remains a non-singing presence; preserve their "
                "described action without lip-syncing to the soundtrack."
            )
            if requested_expression:
                direction += " Preserve the explicitly requested non-song expression."
            parts.append(direction)
    return " ".join(parts)


_INSTRUMENT_ROLE = re.compile(r"\b(?:drummer|guitarist|bassist|keyboardist|pianist|percussionist)\b", re.I)
_WIND_PLAYER = re.compile(r"\b(?:flut\w*|saxophon\w*|trumpet\w*|trombon\w*|clarinet\w*|oboe\w*|bassoon\w*|tuba|harmonica|bagpipe|brass|wind instrument)\b", re.I)
_AUDIENCE = re.compile(r"\b(?:crowd|audience|fans|spectators)\b", re.I)
_EXPRESSION = re.compile(r"\b(?:cheer\w*|shout\w*|bellow\w*|scream\w*)\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|never|without|avoid|stop|doesn't|don't)\b", re.I)
_OPEN_MOUTH = re.compile(
    r"\b(?P<owner>his|her|their) (?:mouth|lips) (?:is|are|stays?|remains?) "
    r"(?:wide |slightly )?open(?: in (?:a |an )?[^,.;]+)?"
    r"|\bopens? (?P<possessive>his|her|their) mouth(?: wide)?(?: mid[- ]vocali[sz]ation)?"
    r"|\b(?P<pose>mouth wide open|open[- ]mouth(?:ed)? (?:vocalization|shout|singing))",
    re.I,
)
_VOCAL_ACTION = re.compile(
    r"\b(?:sings?|singing|raps?|rapping|bellows?|bellowing|shouts?|shouting|"
    r"screams?|screaming|lip[- ]sync(?:s|ing)?|vocali[sz](?:es|ing)|"
    r"mouths? (?:the )?lyrics)\b(?:(?!\band\b)[^,.;])*", re.I,
)
_VOCAL_BREATH = re.compile(r"\b(?:deep,? audible breath|breathing heavily|panting)\b", re.I)


def _field(subject, key):
    return subject.get(key) if isinstance(subject, Mapping) else getattr(subject, key, None)


def _aliases(subject):
    description = str(_field(subject, "visual_description") or "")
    aliases = [str(_field(subject, key) or "").strip() for key in ("speaker_name", "character_id")]
    aliases += _INSTRUMENT_ROLE.findall(description)
    if _field(subject, "performance_role") == "vocalist":
        aliases += re.findall(r"\b(?:lead singer|lead vocalist|singer|vocalist)\b", description, re.I)
    # The identifying noun phrase, not generic clothing/age/color words shared
    # by several band members. Unknown descriptions are deliberately conservative.
    opening = re.split(r"\s+(?:in|with|wearing)\s+|[,(]", description, maxsplit=1, flags=re.I)[0].strip()
    if opening and opening.casefold() not in {"the man", "the woman", "a man", "a woman", "person", "performer"}:
        aliases.append(opening)
    return [re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.I) for alias in aliases if alias]


def _requested_expression(subject, project_context):
    return any(
        any(name.search(sentence) for name in _aliases(subject))
        and _EXPRESSION.search(sentence) and not _NEGATION.search(sentence)
        and not _AUDIENCE.search(sentence)
        for sentence in re.split(r"[.!?;]\s*", project_context or "")
    )


def constrain_music_performance(prompt, subjects=(), vocal_activity=None, *, project_context=""):
    """Resolve contradictory musician mouth cues, without replacing choreography.

    Only a known instrumentalist or a singer without positive vocal evidence is
    constrained. Unknown actors, ambiguous ensemble sentences, wind embouchure,
    quoted text and user-requested expressions are left intact. This is scoped
    to source-song planning; narrative dialogue never passes through it.
    """
    text = strip_legacy_music_performance_direction(
        prompt,
        subjects,
        vocal_activity,
        project_context=project_context,
    )
    people = [s for s in subjects or () if _field(s, "performance_role") in {"vocalist", "instrumentalist"}]
    if not people:
        return text
    aliases = [_aliases(s) for s in people]
    blocked = []
    for person in people:
        explicitly_requested = _requested_expression(person, project_context)
        blocked.append(
            not explicitly_requested
            and not _WIND_PLAYER.search(str(_field(person, "visual_description") or ""))
            and (_field(person, "performance_role") == "instrumentalist" or vocal_activity in {"silent", "unknown"})
        )
    # Resolve simple sentence-local subjects, then their following pronouns.
    # Never transfer a drummer's restriction to a singer in a mixed shot.
    current = 0 if len(people) == 1 else None
    pieces = re.split(r"(?<=[.!?;])(?=\s)|(?=\[Shot\s+\d+\])", text)
    for index, sentence in enumerate(pieces):
        matches = [i for i, names in enumerate(aliases) if any(name.search(sentence) for name in names)]
        if len(matches) == 1:
            current = matches[0]
            # Writers often introduce a performer by name plus (S1), then use
            # only S1 after a camera cut or crowd insert. Bind that observed ID
            # to the same person; a solo drummer may still be S3, not S1.
            for stable_id in re.findall(r"\((S\d+)\)", sentence, re.I):
                aliases[current].append(re.compile(r"(?<!\w)" + stable_id + r"(?!\w)", re.I))
        elif len(matches) > 1 or _AUDIENCE.search(sentence):
            current = None
        if current is None or not blocked[current] or _AUDIENCE.search(sentence):
            continue
        if '"' in sentence or '<d>' in sentence or sentence.lstrip().startswith("Project context:"):
            continue
        # A negative instruction already requests the desired behavior. Do not
        # turn "does not sing" into a contradictory affirmative sentence.
        if _NEGATION.search(sentence):
            continue
        def close_mouth(match):
            owner = match.group("owner")
            if owner:
                return f"{owner} lips stay relaxed and closed"
            possessive = match.group("possessive")
            if possessive:
                verb = "keeps" if match[0].lower().startswith("opens ") else "keep"
                return f"{verb} {possessive} lips relaxed and closed"
            return "lips relaxed and closed"
        sentence = _OPEN_MOUTH.sub(close_mouth, sentence)
        def replace_vocal_action(match):
            prefix = sentence[:match.start()]
            if re.search(r"\b(?:a|an|the|his|her|their)\s+$", prefix, re.I):
                return "closed-mouth expression"
            word = match.group(0).split()[0].lower()
            verb = "moving" if word.endswith("ing") else "move" if re.search(r"\bto\s+$", prefix) else "moves"
            return f"{verb} with the music with relaxed closed lips"
        sentence = _VOCAL_ACTION.sub(replace_vocal_action, sentence)
        sentence = _VOCAL_BREATH.sub(
            lambda m: "quiet breath through the nose" if "breath" in m[0] and "breathing" not in m[0]
            else "breathing quietly through the nose", sentence,
        )
        pieces[index] = sentence
    return "".join(pieces)
