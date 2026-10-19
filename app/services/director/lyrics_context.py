"""Keep optional lyric meaning separate from mandatory source-audio timing."""
from __future__ import annotations


def lyrics_context_enabled(value: object) -> bool:
    # Requests and saved projects use a JSON boolean, never truthy strings.
    return value is True


def lyric_visual_policy(enabled: bool) -> str:
    if enabled:
        return (
            "LYRIC VISUAL CONTEXT: Lyrics are optional thematic inspiration for "
            "imagery and emotional progression, subordinate to the user's visual "
            "concept and references. Do not illustrate every line literally or "
            "add characters, events, dialogue, captions or settings that conflict "
            "with the concept. The recording owns all words and vocal timing."
        )
    return (
        "LYRIC VISUAL CONTEXT DISABLED: Develop visuals only from the user's "
        "concept and references. Audio analysis supplies tempo, sections, energy, "
        "vocal activity and performer timing, not story content. Do not infer "
        "characters, objects or events from the song's words. Preserve the "
        "soundtrack and assigned lip-sync."
    )
