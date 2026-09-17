"""Keep audible vocals separate from the people shown in a music-video shot."""

from collections.abc import Mapping
import re


PERFORMANCE_ROLES = ("vocalist", "instrumentalist", "non_vocal", "non_performer")

MUSIC_PERFORMANCE_RULES = """MUSIC PERFORMANCE ROLES:
- Establish each person's role from the user's concept, performer mappings and references,
  then keep that role across cuts. Camera focus never turns an instrumentalist into a singer.
- For each subjects_on_screen entry, set performance_role: vocalist (assigned lead/backing
  vocals in this shot), instrumentalist (plays without singing), non_vocal (dancer, listener,
  or other person not singing), or non_performer (instrument, scenery, object).
- Vocals in the soundtrack do NOT require the vocalist to be on screen. A guitar/drum
  cutaway can happen during a vocal phrase: the established singer continues OFF SCREEN.
  State that explicitly in video_prompt and every window_prompt. Do not insert the singer
  into that cutaway or transfer their voice/lip movements to the musician being shown.
- For non-singing guitarists, bassists, drummers and listeners, explicitly describe relaxed,
  CLOSED lips, with no singing, mouthing lyrics, or lip-sync. Keep hands, body, expression
  and instrument playing lively; closing the mouth must not freeze their performance.
- In a wide band shot, only the assigned visible vocalist mouths the audible vocal;
  the other musicians keep their mouths closed. Backing vocals, duets and a guitarist
  who also sings are allowed when the user assigns them. Do not invent backing singers.
- A wind/brass player's mouth may form the required embouchure; that is instrument
  playing, not lyric-shaped mouth movement. Match first-frame mouth poses to these roles.
- Preserve explicitly requested cheers, shouts or other non-singing expressions;
  those do not turn an audience member into the source track's vocalist.
- Keep one role per person consistent throughout a shot. A camera cut changes framing,
  not vocal ownership. Preserve the correct performer for each voice in a duet.
- The person's vocalist ROLE is not evidence that vocals occur in this INTERVAL.
  During instrumental intros, breaks and pauses, even the lead singer keeps relaxed
  closed lips while listening, moving or interacting with the band. Never animate
  guitar riffs as syllables. When vocal evidence is unknown, use this listening pose
  by default and allow mouth movement only with an actual voice in the source audio.
- Describe intensity with eyes, posture, hands and instrument playing, not invented
  bellows, open-mouth shouts, panting or vocal breaths. A drummer does not shout just
  because a chorus is energetic. A cheer explicitly requested for the audience does
  not authorize a musician to shout. Remove conflicting mouth actions from action_beats,
  ending_beat, image_prompt, video_prompt and window_prompts before returning the plan.
"""


_VOCAL_OWNERSHIP = (
    "Vocal ownership stays with the assigned singer across camera cuts. "
    "Only an explicitly assigned vocalist lip-syncs, and only to their own "
    "audible vocal part. Non-singing guitarists, bassists and drummers keep "
    "their lips closed without mouthing lyrics, except for a non-singing "
    "expression explicitly requested by the user; their hands and bodies "
    "continue the instrumental performance. During an instrument-only "
    "cutaway with audible vocals, the singer continues off screen; do not transfer the vocal "
    "to the person in view or insert a singer into the shot."
)


def music_performance_direction(subjects=(), vocal_activity=None, *, project_context=""):
    """Compile assigned roles, without guessing vocalists from camera focus.

    Older plans have no role metadata and receive the conditional direction.
    Missing roles never imply that an unidentified person must remain silent.
    """
    parts = [_VOCAL_OWNERSHIP]
    if vocal_activity == "silent":
        parts.append(
            "This interval has no detected vocals: even the lead singer keeps relaxed "
            "closed lips throughout, moving or listening to the instrumental music. "
            "No singing, lip-sync, bellowing or invented vocal breath."
        )
    elif vocal_activity == "unknown":
        parts.append(
            "Vocal activity in this interval is unconfirmed. Default to relaxed closed "
            "lips, including the lead singer; allow lyric-shaped mouth movement only "
            "when a voice is actually audible in the supplied audio. Guitar riffs and "
            "drum hits never drive the mouth. Do not invent shouts or vocal breaths."
        )
    elif vocal_activity == "active":
        parts.append(
            "Vocals occur within this interval, not necessarily throughout it. "
            "The assigned singer closes their lips during instrumental gaps and "
            "starts lip-sync only when their actual vocal part enters."
        )
    for subject in subjects or ():
        def field(key):
            return subject.get(key) if isinstance(subject, Mapping) else getattr(subject, key, None)

        role = field("performance_role")
        description = re.sub(r"\s+", " ", str(field("visual_description") or "")).strip(" .")
        if not description or role not in PERFORMANCE_ROLES or role == "non_performer":
            continue
        if role in {"vocalist", "instrumentalist"} and _requested_expression(subject, project_context):
            parts.append(f"{description} may perform the user's explicitly requested expression; this never transfers the song's vocal to them.")
            continue
        if role == "vocalist":
            parts.append(f"Assigned visible vocalist: {description}; mouth movement follows only their vocal part when audible.")
        elif role == "instrumentalist" and re.search(
            r"\b(?:flute|flutist|flautist|saxophone|saxophonist|trumpet|trombone|clarinet|oboe|bassoon|tuba|harmonica|bagpipe|brass|wind instrument)\w*\b",
            description, re.IGNORECASE,
        ):
            parts.append(f"{description} uses the instrument's embouchure without singing or mouthing lyrics.")
        elif role == "instrumentalist":
            parts.append(f"{description} keeps their mouth closed and does not sing, mouth lyrics, or lip-sync; natural body movement continues.")
        else:
            parts.append(f"{description} does not sing or mouth the lyrics; preserve any explicitly described non-singing expression or cheering.")
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
    people = [s for s in subjects or () if _field(s, "performance_role") in {"vocalist", "instrumentalist"}]
    if not people:
        return str(prompt or "")
    text = str(prompt or "")
    direction = music_performance_direction(subjects, vocal_activity, project_context=project_context)
    if direction in text:
        return direction.join(
            constrain_music_performance(part, subjects, vocal_activity, project_context=project_context)
            for part in text.split(direction)
        )
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
    pieces = re.split(r"(?<=[.!?;])(?=\s)|(?=\[Shot\s+\d+\])", str(prompt or ""))
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
