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
"""


_VOCAL_OWNERSHIP = (
    "Vocal ownership stays with the assigned singer across camera cuts. "
    "Only an explicitly assigned vocalist lip-syncs, and only to their own "
    "audible vocal part. Non-singing guitarists, bassists and drummers keep "
    "their lips closed without mouthing lyrics; their hands and bodies "
    "continue the instrumental performance. During an instrument-only "
    "cutaway the singer continues off screen; do not transfer the vocal "
    "to the person in view or insert a singer into the shot."
)


def music_performance_direction(subjects=()):
    """Compile assigned roles, without guessing vocalists from camera focus.

    Older plans have no role metadata and receive the conditional direction.
    Missing roles never imply that an unidentified person must remain silent.
    """
    parts = [_VOCAL_OWNERSHIP]
    for subject in subjects or ():
        def field(key):
            return subject.get(key) if isinstance(subject, Mapping) else getattr(subject, key, None)

        role = field("performance_role")
        description = re.sub(r"\s+", " ", str(field("visual_description") or "")).strip(" .")
        if not description or role not in PERFORMANCE_ROLES or role == "non_performer":
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
