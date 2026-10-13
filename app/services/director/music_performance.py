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
- An explicitly assigned singer stays a vocalist when shown in narrative or movement
  shots, including riding; vocal activity controls mouth movement during each interval.
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
    # The current compiler can be called at planning, preflight and final
    # reference assembly. Remove only exact compiler-owned per-subject lines
    # for every activity variant, then emit the current role once. This also
    # cleans saved plans whose current activity differs from their old one.
    for activity in (vocal_activity, None, "active", "silent", "unknown"):
        for subject in subjects or ():
            for context in (project_context, ""):
                description = _cue_description(subject)
                labels = list(dict.fromkeys(filter(None, (_cue_label(subject), description))))
                for role in PERFORMANCE_ROLES:
                    for label in labels:
                        emitted = _compile_music_performance_direction(
                            [subject], activity, project_context=context,
                            role_overrides=[role], labels=[label],
                        )
                        if emitted:
                            text = text.replace(emitted, "")
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
    return _compile_music_performance_direction(
        subjects, vocal_activity, project_context=project_context,
    )


def _compile_music_performance_direction(
    subjects=(), vocal_activity=None, *, project_context="",
    role_overrides=None, labels=None,
):
    parts = []
    for index, subject in enumerate(subjects or ()):
        role = (
            role_overrides[index]
            if role_overrides is not None and index < len(role_overrides)
            else resolved_music_performance_role(subject, project_context)
        )
        description = _cue_description(subject)
        label = (
            labels[index]
            if labels is not None and index < len(labels)
            else _cue_label(subject)
        )
        if not label or role not in PERFORMANCE_ROLES or role == "non_performer":
            continue
        requested_expression = _requested_expression(subject, project_context)
        if role == "vocalist":
            if vocal_activity in {"silent", "unknown"}:
                if requested_expression:
                    direction = (
                        f"{label} keeps relaxed closed lips except during the "
                        "explicitly requested non-song expression; do not lip-sync "
                        "unless their own vocal part is audible."
                    )
                else:
                    direction = (
                        f"{label} keeps relaxed closed lips through this interval; "
                        "no singing or lyric mouthing."
                    )
            else:
                direction = (
                    f"{label} lip-syncs only their own audible vocal part, "
                    "with relaxed closed lips during instrumental gaps."
                )
            if requested_expression:
                direction += " Keep that expression separate from their assigned song vocal."
            parts.append(direction)
        elif role == "instrumentalist" and _WIND_PLAYER.search(description):
            direction = (
                f"{label} uses the instrument's natural embouchure without "
                "singing or mouthing lyrics."
            )
            if requested_expression:
                direction += " Preserve the explicitly requested non-song expression separately from playing."
            parts.append(direction)
        elif role == "instrumentalist":
            if requested_expression:
                direction = (
                    f"{label} keeps relaxed closed lips except during the "
                    "explicitly requested non-song expression; never lip-sync to "
                    "source vocals, and keep hands and body active."
                )
            else:
                direction = (
                    f"{label} plays with relaxed closed lips, without singing or "
                    "mouthing lyrics; keep hands and body active."
                )
            parts.append(direction)
        else:
            direction = (
                f"{label} remains a non-singing presence; preserve their "
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
_COLLECTIVE_CAST_SCOPE = re.compile(
    r"\b(?:all\s+(?:(?P<count>two|three|four|five|six|seven|eight|nine|ten|"
    r"\d+)\s+)?(?:(?:of\s+)?(?:the\s+)?(?:visible\s+|on-screen\s+)?)?"
    r"(?:subjects?|people|performers|characters|members|musicians|band|cast|"
    r"group|ensemble)|both\s+(?:visible\s+)?(?:subjects?|people|performers|"
    r"characters|members|musicians|band|cast|group|ensemble))\b",
    re.I,
)
_COLLECTIVE_CLOSED_MOUTH_ADJUNCT = re.compile(
    r"(?P<connector>,\s*|\s+with\s+)"
    r"(?:(?:their|all\s+their|everyone['’]s|the\s+subjects?['’]?)\s+)?"
    r"(?:mouths?|lips)\s+"
    r"(?:(?:stay|remain|are|is|keep|keeps|held|hold)\s+)?"
    r"(?:relaxed\s+)?closed\b"
    r"(?:\s+(?:and|but)\s+(?:attentive|focused|still|silent|quiet|calm))?",
    re.I,
)
_SOURCE_CLOSED_MOUTH_REQUEST = re.compile(
    r"\b(?:(?:keep|keeps|maintain|maintains|stay|stays|remain|remains|"
    r"are|is|hold|holds)\s+(?:(?:their|his|her|the\s+performer['’]s)\s+)?"
    r"(?:mouths?|lips)\s+(?:relaxed\s+)?(?:closed|shut)|"
    r"(?:mouths?|lips)\s+(?:stay|remain|are|is|keep|keeps)\s+"
    r"(?:relaxed\s+)?(?:closed|shut)|(?:closed|shut)\s+(?:mouths?|lips))\b",
    re.I,
)
_SOURCE_NO_SONG_REQUEST = re.compile(
    r"\b(?:no\s+(?:one|singing|lip[- ]?sync(?:ing)?|lyric\s+mouthing)|"
    r"(?:do\s+not|does\s+not|did\s+not|don't|doesn't|didn't|never|"
    r"without|should\s+not|must\s+not)\s+(?:sing|singing|lip[- ]?sync(?:ing)?|"
    r"mouth(?:ing)?\s+(?:the\s+)?lyrics))\b",
    re.I,
)
_SOURCE_GLOBAL_NO_SONG_REQUEST = re.compile(
    r"\b(?:no\s+(?:singing|lip[- ]?sync(?:ing)?|lyric\s+mouthing)|"
    r"no\s+(?:one|one should|anyone)\s+(?:sings?|lip[- ]?syncs?|"
    r"mouths?\s+lyrics)|nobody\s+(?:sings?|lip[- ]?syncs?))\b",
    re.I,
)


def _field(subject, key):
    return subject.get(key) if isinstance(subject, Mapping) else getattr(subject, key, None)


def _cue_description(subject):
    return re.sub(
        r"\s+", " ", str(_field(subject, "visual_description") or "")
    ).strip(" .")


def _cue_label(subject):
    """Prefer the explicit cast name; keep appearance prose for unnamed subjects."""
    return re.sub(
        r"\s+", " ", str(_field(subject, "speaker_name") or "")
    ).strip(" .") or _cue_description(subject)


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


_GENERIC_IDENTITY_LABELS = {
    "a man", "the man", "a woman", "the woman", "a person", "the person",
    "person", "performer", "a performer", "the performer", "unidentified person",
}


def _assignment_aliases(subject):
    """Use mapped identities and distinctive descriptions, not generic roles."""
    aliases = []
    speaker_name = str(_field(subject, "speaker_name") or "").strip()
    if speaker_name and speaker_name.casefold() not in _GENERIC_IDENTITY_LABELS:
        aliases.append(speaker_name)
    character_id = str(_field(subject, "character_id") or "").strip()
    if character_id:
        aliases.append(character_id)
    description = str(_field(subject, "visual_description") or "")
    opening = re.split(
        r"\s+(?:in|with|wearing)\s+|[,(]", description,
        maxsplit=1, flags=re.I,
    )[0].strip()
    if opening and opening.casefold() not in _GENERIC_IDENTITY_LABELS:
        aliases.append(opening)
    return [
        re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.I)
        for alias in dict.fromkeys(aliases)
    ]


_VOCAL_ASSIGNMENT = re.compile(
    r"\b(?:sing(?:s|ing)?|sang|sung|rap(?:s|ping|ped)?|"
    r"(?:lead\s+|backing\s+)?(?:singer|vocalist)|"
    r"vocali[sz](?:es|ing|ed)|lip[- ]sync(?:s|ing)?|"
    r"perform(?:s|ing)?\s+(?:the\s+)?(?:song|vocals?)|"
    r"provides?\s+(?:the\s+)?vocals?)\b",
    re.I,
)
_NEGATED_VOCAL_PREFIX = re.compile(
    r"(?:\b(?:is|are|was|were|do|does|did|will|would|should|must|can|could|"
    r"may|might|has|have|had)\s+)?"
    r"(?:not|never|without|no|isn't|aren't|wasn't|weren't|doesn't|don't|"
    r"didn't|won't|wouldn't|shouldn't|mustn't|can't|couldn't|hasn't|haven't)\s+"
    r"(?:(?:to|be|being|a|an|the|lead|backing|really|currently|ever)\s*)*$",
    re.I,
)
_DIRECT_ASSIGNMENT_MODIFIERS = re.compile(
    r"^(?:(?:is|are|was|were|be|been|being|will|would|should|must|can|could|"
    r"may|might|do|does|did|has|have|had|to|the|a|an|lead|backing|assigned|"
    r"now|also|still|currently|together|both|happily|proudly|loudly|softly|"
    r"enthusiastically)\b\s*|\w+ly\b\s*)*$",
    re.I,
)
_COORDINATED_ASSIGNMENT_GAP = re.compile(
    r"^(?:(?:,\s*)?(?:and|&)\s+|,\s+)"
    r"(?:the\s+)?[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*){0,2}"
    r"(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)"
    r"(?:the\s+)?[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*){0,2})*"
    r"(?:\s+(?:is|are|was|were|will|would|should|can|could|must|may|might|"
    r"also|both|together|now))*$"
)


def _direct_assignment_gap(gap):
    gap = re.sub(r"\s+", " ", gap or "").strip()
    if not gap or _DIRECT_ASSIGNMENT_MODIFIERS.fullmatch(gap):
        return True
    # Shared predicates such as "Mara and Kitten sing" are explicit for both
    # cast identities, while a different person watching the singer is not.
    return bool(_COORDINATED_ASSIGNMENT_GAP.fullmatch(gap))


def _has_explicit_vocal_assignment(subject, project_context):
    aliases = _assignment_aliases(subject)
    if not aliases:
        return False
    for sentence in re.split(r"[.!?;]\s*", project_context or ""):
        for predicate in _VOCAL_ASSIGNMENT.finditer(sentence):
            prefix = sentence[:predicate.start()]
            if _NEGATED_VOCAL_PREFIX.search(prefix[-96:]):
                continue
            for alias in aliases:
                for mention in alias.finditer(sentence, 0, predicate.start()):
                    if _direct_assignment_gap(sentence[mention.end():predicate.start()]):
                        return True
    return False


def resolved_music_performance_role(subject, project_context=""):
    """Resolve a shot role from the explicit project assignment for this identity."""
    role = _field(subject, "performance_role")
    if role == "vocalist":
        return role
    if _has_explicit_vocal_assignment(subject, project_context):
        return "vocalist"
    return role


def _requested_expression(subject, project_context):
    return any(
        any(name.search(sentence) for name in _aliases(subject))
        and _EXPRESSION.search(sentence) and not _NEGATION.search(sentence)
        and not _AUDIENCE.search(sentence)
        for sentence in re.split(r"[.!?;]\s*", project_context or "")
    )


_CAST_COUNTS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _collective_scope_covers_cast(sentence, subject_count):
    if subject_count < 2:
        return False
    for match in _COLLECTIVE_CAST_SCOPE.finditer(sentence or ""):
        raw_count = match.group("count")
        if raw_count:
            count = _CAST_COUNTS.get(raw_count.casefold())
            if count is None:
                try:
                    count = int(raw_count)
                except ValueError:
                    continue
            if count != subject_count:
                continue
        elif match.group(0).lstrip().casefold().startswith("both "):
            # "Both" is explicitly a two-person scope; it cannot safely
            # describe a larger visible cast that includes an active singer.
            if subject_count != 2:
                continue
        return True
    return False


def _source_requests_vocalist_silence(project_context, vocalists, subject_count):
    """Do not override an explicit source request to keep the singer silent."""

    for sentence in re.split(r"[.!?;]\s*", project_context or ""):
        if _SOURCE_GLOBAL_NO_SONG_REQUEST.search(sentence):
            return True
        if not (
            _SOURCE_CLOSED_MOUTH_REQUEST.search(sentence)
            or _SOURCE_NO_SONG_REQUEST.search(sentence)
        ):
            continue
        if _collective_scope_covers_cast(sentence, subject_count):
            return True
        if any(
            alias.search(sentence)
            for subject in vocalists
            for alias in _aliases(subject)
        ):
            return True
    return False


def _remove_collective_active_vocal_closure(
    prompt,
    subjects,
    vocal_activity,
    *,
    project_context="",
):
    """Drop only a cast-wide closed-mouth adjunct that contradicts active vocals."""

    vocalists = [
        subject for subject in subjects or ()
        if resolved_music_performance_role(subject, project_context) == "vocalist"
    ]
    if (
        vocal_activity != "active"
        or not vocalists
        or _source_requests_vocalist_silence(
            project_context, vocalists, len(subjects or ()),
        )
    ):
        return str(prompt or "")

    pieces = re.split(r"(?<=[.!?;])(?=\s)|(?=\[Shot\s+\d+\])", str(prompt or ""))
    for index, sentence in enumerate(pieces):
        if not _collective_scope_covers_cast(sentence, len(subjects or ())):
            continue
        match = _COLLECTIVE_CLOSED_MOUTH_ADJUNCT.search(sentence)
        if not match:
            continue
        # Keep action/focus prose and only remove a sentence-final adjunct. A
        # main predicate such as "Everyone keeps their mouths closed" is not
        # safe to rewrite because removing it would leave a broken sentence.
        if sentence[match.end():].strip(" .!?;,"):
            continue
        collective = _COLLECTIVE_CAST_SCOPE.search(sentence)
        if not collective or not sentence[collective.end():match.start()].strip(" ,;:"):
            continue
        pieces[index] = sentence[:match.start()] + sentence[match.end():]
    return "".join(pieces)


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
    text = _remove_collective_active_vocal_closure(
        text,
        subjects,
        vocal_activity,
        project_context=project_context,
    )
    subject_list = list(subjects or ())
    roles = [
        resolved_music_performance_role(subject, project_context)
        for subject in subject_list
    ]
    people = [
        subject for subject, role in zip(subject_list, roles)
        if role in {"vocalist", "instrumentalist"}
    ]
    if not people:
        return text
    aliases = [_aliases(s) for s in people]
    blocked = []
    for person in people:
        role = resolved_music_performance_role(person, project_context)
        explicitly_requested = _requested_expression(person, project_context)
        blocked.append(
            not explicitly_requested
            and not _WIND_PLAYER.search(str(_field(person, "visual_description") or ""))
            and (role == "instrumentalist" or vocal_activity in {"silent", "unknown"})
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
