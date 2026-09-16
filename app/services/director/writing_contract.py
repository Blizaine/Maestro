"""Shared Studio craft guidance, scoped to Director's existing production plan."""
from services.adaptive_enhancement import adaptive_writing_guide
from services.guide_loader import load_guide

DIRECTOR_WRITING_VERSION = 1


def director_writing_contract(
    brief: str, *, source_song: bool = False, source_audio: bool = False, nsfw: bool = False,
) -> str:
    parts = [adaptive_writing_guide(brief), (
        "DIRECTOR PRODUCTION SCOPE\n"
        "Apply this writing guidance inside the supplied production plan and output schema. "
        "The cast/reference map, exact shot interval, assigned events, user constraints and "
        "reviewed edits are authoritative. Develop only open details. A shot's ending state "
        "becomes the next continuous shot's opening state: posture, facing, occupied hands, "
        "props, damage and door positions cannot reset at a cut. A camera cut can reveal a "
        "different view of the same event; it must not replay its impact or advance an actor "
        "through an unseen physical transition. Preserve the supplied first image's opening "
        "state; identity-only references do not impose its pose. Keep each audible line owned "
        "by its named speaker and stage who is visibly speaking or listening. Write the final "
        "image/video prompts for their respective model formats. Do not paste this contract "
        "or invent extra schema fields in the output."
    )]
    if source_song:
        parts.append(
            "SOURCE SONG AUTHORITY\n"
            "The supplied recording owns all lyrics, voice identities and vocal timing. "
            "Conversation-expansion guidance does not apply to this music video. Do not "
            "invent dialogue, new lyrics or competing speech. Stage singing only within "
            "the analyzed vocal intervals for the mapped performer; use closed-mouth "
            "action or listening during instrumental gaps. Develop the visual story "
            "around the song, keeping its waveform and timeline unchanged."
        )
    elif source_audio:
        parts.append(
            "SOURCE RECORDING AUTHORITY\n"
            "The supplied recording and timestamped transcript own the spoken words, "
            "speaker identities and timing. Do not expand, paraphrase, duplicate or "
            "invent speech. Plan speaking faces, listening reactions and visual action "
            "around those exact intervals without changing the source waveform."
        )
    if nsfw:
        shared = load_guide("enhance", "nsfw_shared")
        if shared:
            parts.append(shared)
    return "\n\n".join(part for part in parts if part)
