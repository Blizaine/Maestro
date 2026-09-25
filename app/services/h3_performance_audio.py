"""Shared enhancement policy for H3's exact target soundtrack input."""

import math
import re
from typing import TypedDict


PERFORMANCE_AUDIO_DIRECTION = (
    "Preserve the supplied target soundtrack unchanged. For requested singing or "
    "speaking, synchronize that performer's lip movement to its audible vocals. "
    "Match physical performance to its rhythm; during instrumental passages, continue the visible action without "
    "inventing singing. Bystanders do not mouth the performer's vocals."
)

PERFORMANCE_AUDIO_GUIDANCE = (
    "EXACT SOUNDTRACK PERFORMANCE: The uploaded audio already supplies all vocals, "
    "words, music and timing. Write visual performance and camera direction only. "
    "Do not invent, transcribe, rewrite or allocate dialogue/lyrics, <d> blocks, "
    "speech word budgets, extra sound effects or a replacement score. Quoted lyrics "
    "in the brief describe the supplied performance, not a new generated script. "
    "No transcript or musical analysis is supplied: do not guess lyric timings or "
    "invent instrumental/vocal sections. Follow the actual conditioning audio. "
    "A sustained performance can continue across windows with evolving choreography "
    "and coverage; it does not restart at each boundary. Do not impose blanket "
    "silence or closed mouths on a requested singing performance. "
    + PERFORMANCE_AUDIO_DIRECTION
)

_PERFORMANCE_CLOCK_LINE_RE = re.compile(
    r"^\s*H3_PERFORMANCE_AUDIO_CLOCK\s+duration_seconds="
    r"(?P<duration>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
    re.I | re.M,
)
_STATIC_HOLD_SOURCE_RE = re.compile(
    r"^\s*(?:(?:the|a|an)\s+)?(?P<video>(?:\d+(?:\.\d+)?|\.\d+))\s*-\s*second\s+"
    r"(?:video|film|clip)\s+(?:then\s+)?holds?\s+(?P<target>.+?)\s+"
    r"in\s+silence\s+for\s+(?:the\s+)?remaining\s+"
    r"(?P<tail>(?:\d+(?:\.\d+)?|\.\d+))\s+seconds?\s*\.?\s*$",
    re.I,
)
_STATIC_HOLD_AMBIGUITY_RE = re.compile(
    r"(?:[\"“”]|\b(?:if|unless|whether|perhaps|maybe|may|might|could|"
    r"would|should|not|never|no|without|but|however|instead)\b)",
    re.I,
)
_STATIC_HOLD_ACTION_RE = re.compile(
    r"\b(?:move|moves|moving|continue|continues|continuing|walk|walks|walking|"
    r"run|runs|running|dance|dances|dancing|perform|performs|performing|"
    r"resume|resumes|resuming|begin|begins|beginning|start|starts|starting)\b",
    re.I,
)


class H3StaticHoldRequest(TypedDict):
    """A source sentence that explicitly requests a timed silent final hold."""

    source_event_id: str
    source_event_text: str
    video_duration_seconds: float
    requested_tail_seconds: float
    static_hold_explicit: bool


class H3StaticHoldWindow(TypedDict):
    """The part of a global final hold that intersects one visual window."""

    window: int
    global_start_seconds: float
    global_end_seconds: float
    local_start_seconds: float
    local_end_seconds: float
    hold_seconds: float


class H3StaticHoldProjection(TypedDict):
    """Source-backed global hold and its per-window time contributions."""

    source_event_id: str
    source_event_text: str
    global_start_seconds: float
    global_end_seconds: float
    total_hold_seconds: float
    windows: list[H3StaticHoldWindow]
    audio_duration_seconds: float
    video_duration_seconds: float
    requested_tail_seconds: float


def _finite_positive_number(value) -> float | None:
    """Accept JSON numeric metadata only; never infer duration from prose."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def performance_audio_item_duration_seconds(item) -> float | None:
    """Read an existing numeric duration field from a drive-audio manifest row."""
    if not isinstance(item, dict):
        return None
    value = _finite_positive_number(item.get("duration_seconds"))
    if value is not None:
        return value
    return _finite_positive_number(item.get("duration"))


def performance_audio_duration_seconds(reference_context: str | None) -> float | None:
    """Read the machine-generated clock line for an exact performance driver.

    The surrounding reference description is intentionally not parsed: role
    prose, filenames, and natural-language durations are not timing evidence.
    """
    context = str(reference_context or "")
    if not has_h3_performance_audio(context):
        return None
    match = _PERFORMANCE_CLOCK_LINE_RE.search(context)
    if not match:
        return None
    try:
        return _finite_positive_number(float(match.group("duration")))
    except (TypeError, ValueError, OverflowError):
        return None


def performance_audio_window_clock(
    reference_context: str | None,
    window_durations,
) -> list[dict[str, float | int]]:
    """Project a verified soundtrack endpoint into visual window intervals.

    Returns an empty list when the exact-drive duration or any visual window
    duration is unknown/invalid.  Values are numeric context only; they do not
    allocate frames or promise exact soundtrack cutting by the compiler.
    """
    audio_duration = performance_audio_duration_seconds(reference_context)
    if audio_duration is None or not isinstance(window_durations, (list, tuple)):
        return []
    durations = [_finite_positive_number(value) for value in window_durations]
    if not durations or any(value is None for value in durations):
        return []

    windows: list[dict[str, float | int]] = []
    start = 0.0
    for index, duration in enumerate(durations, start=1):
        assert duration is not None
        end = start + duration
        audio_end_local = min(duration, max(0.0, audio_duration - start))
        silent_tail = max(0.0, end - max(start, audio_duration))
        windows.append({
            "window": index,
            "global_start_seconds": start,
            "global_end_seconds": end,
            "audio_end_local_seconds": audio_end_local,
            "silent_tail_seconds": silent_tail,
        })
        start = end
    return windows


def discover_h3_requested_static_holds(source_events) -> list[H3StaticHoldRequest]:
    """Find explicit, affirmative timed final-static-hold requirements.

    Authored numbers here describe the requested video and tail only. They are
    never treated as a measured soundtrack duration; that must come from the
    machine-generated performance-audio clock.
    """
    if not isinstance(source_events, (list, tuple)):
        return []

    holds: list[H3StaticHoldRequest] = []
    for event in source_events:
        if not isinstance(event, dict):
            continue
        event_id = event.get("event_id")
        text = event.get("text")
        if not isinstance(event_id, str) or not re.fullmatch(r"E\d+", event_id):
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        if _STATIC_HOLD_AMBIGUITY_RE.search(text):
            continue
        match = _STATIC_HOLD_SOURCE_RE.fullmatch(text.strip())
        if not match:
            continue

        target = match.group("target").strip(" .;:")
        # Require a final visual state, rather than a generic silence or a
        # duration sentence that could describe continuing action.
        if not re.search(r"\b(?:final|last|ending)\b", target, re.I):
            continue
        if not re.search(
            r"\b(?:composition|frame|shot|image|tableau|view|picture)\b",
            target, re.I,
        ):
            continue
        if _STATIC_HOLD_ACTION_RE.search(target):
            continue
        if re.search(r"\b(?:as|while|where|that|which|whose)\b", target, re.I):
            # A held framing can still contain a moving performer. Do not
            # promote a subordinate action clause to a static-scene contract.
            continue

        video = _finite_positive_number(float(match.group("video")))
        tail = _finite_positive_number(float(match.group("tail")))
        if video is None or tail is None or tail >= video:
            continue
        holds.append({
            "source_event_id": event_id,
            "source_event_text": text,
            "video_duration_seconds": video,
            "requested_tail_seconds": tail,
            "static_hold_explicit": True,
        })
    return holds


def project_h3_requested_static_tail(
    reference_context: str | None,
    window_durations,
    hold_request,
    *,
    tolerance_seconds: float = 0.05,
) -> H3StaticHoldProjection | None:
    """Project an explicit final hold over windows using a verified audio end.

    The source event authenticates that a static hold was requested. The
    audio endpoint is read only from the exact-drive reference clock, and the
    visual windows must cover the authored video duration. Any ambiguity or
    material mismatch returns ``None`` so a caller cannot waive visual proof.
    """
    if not isinstance(hold_request, dict) or hold_request.get("static_hold_explicit") is not True:
        return None
    source_event_id = hold_request.get("source_event_id")
    source_event_text = hold_request.get("source_event_text")
    if (
        not isinstance(source_event_id, str)
        or not re.fullmatch(r"E\d+", source_event_id)
        or not isinstance(source_event_text, str)
        or not source_event_text.strip()
    ):
        return None

    video_duration = _finite_positive_number(hold_request.get("video_duration_seconds"))
    requested_tail = _finite_positive_number(hold_request.get("requested_tail_seconds"))
    tolerance = _finite_positive_number(tolerance_seconds)
    audio_duration = performance_audio_duration_seconds(reference_context)
    if any(value is None for value in (video_duration, requested_tail, tolerance, audio_duration)):
        return None
    assert video_duration is not None
    assert requested_tail is not None
    assert tolerance is not None
    assert audio_duration is not None

    if not isinstance(window_durations, (list, tuple)) or not window_durations:
        return None
    durations = [_finite_positive_number(value) for value in window_durations]
    if any(value is None for value in durations):
        return None
    normalized_durations = [float(value) for value in durations if value is not None]
    total_visual_duration = sum(normalized_durations)
    actual_tail = video_duration - audio_duration
    if (
        abs(total_visual_duration - video_duration) > tolerance
        or actual_tail <= 0
        or abs(actual_tail - requested_tail) > tolerance
    ):
        return None

    tail_start = audio_duration
    tail_end = video_duration
    projected_windows: list[H3StaticHoldWindow] = []
    window_start = 0.0
    for index, duration in enumerate(normalized_durations, start=1):
        window_end = window_start + duration
        overlap_start = max(window_start, tail_start)
        overlap_end = min(window_end, tail_end)
        if overlap_end > overlap_start:
            projected_windows.append({
                "window": index,
                "global_start_seconds": overlap_start,
                "global_end_seconds": overlap_end,
                "local_start_seconds": overlap_start - window_start,
                "local_end_seconds": overlap_end - window_start,
                "hold_seconds": overlap_end - overlap_start,
            })
        window_start = window_end

    covered_hold = sum(item["hold_seconds"] for item in projected_windows)
    if abs(covered_hold - actual_tail) > tolerance:
        return None
    return {
        "source_event_id": source_event_id,
        "source_event_text": source_event_text,
        "global_start_seconds": tail_start,
        "global_end_seconds": tail_end,
        "total_hold_seconds": actual_tail,
        "windows": projected_windows,
        "audio_duration_seconds": audio_duration,
        "video_duration_seconds": video_duration,
        "requested_tail_seconds": requested_tail,
    }


def performance_audio_window_guidance(
    reference_context: str | None,
    window_durations,
    *,
    static_hold_requested: bool = False,
) -> list[str]:
    """Format local visual guidance from the verified machine clock only."""
    clocks = performance_audio_window_clock(reference_context, window_durations)
    if not clocks:
        return []
    audio_end = performance_audio_duration_seconds(reference_context)
    assert audio_end is not None
    guidance = []
    for item in clocks:
        start = float(item["global_start_seconds"])
        end = float(item["global_end_seconds"])
        local_end = float(item["audio_end_local_seconds"])
        tail = float(item["silent_tail_seconds"])
        line = (
            f"Verified supplied-audio clock for window {item['window']}: global "
            f"{start:.3f}–{end:.3f}s; soundtrack ends at global {audio_end:.3f}s."
        )
        if audio_end > end:
            line += " The soundtrack covers this entire window."
            if int(item["window"]) < len(clocks):
                line += " It continues into later windows."
            else:
                line += " Its endpoint lies beyond the selected visual duration."
        elif audio_end <= start:
            line += " The soundtrack has already ended before this window."
        else:
            line += f" In this window it ends at local {local_end:.3f}s."
        if tail > 0:
            if static_hold_requested:
                line += (
                    f" The source requests a static visual hold for the {tail:.3f}s "
                    "after the soundtrack ends; do not add audio."
                )
            else:
                line += (
                    f" The remaining {tail:.3f}s are silent; continue the requested "
                    "visual progression, honoring any explicitly requested final hold at the audio endpoint, "
                    "without inventing audio or adding an unrequested static hold."
                )
        guidance.append(line)
    return guidance


def has_h3_performance_audio(reference_context: str | None) -> bool:
    """Recognize the exact driver in native and interactive reference inventories.

    A voice/style sample, or a numbered video's paired reference audio, does
    not replace the target soundtrack and must retain normal speech checks.
    """
    return bool(re.search(
        r"^\s*(?:The exact target soundtrack supplies\b|"
        r"Exact target soundtrack:[^\r\n]*\bintent=AUDIO REUSE / PERFORMANCE DRIVER\b)",
        str(reference_context or ""), re.I | re.M,
    ))


def enforce_h3_performance_audio(result: str) -> str:
    """Keep the structured visual draft while the uploaded waveform owns audio."""
    def visual_direction(match):
        detail = match[2].rstrip()
        if PERFORMANCE_AUDIO_DIRECTION not in detail:
            detail += " " + PERFORMANCE_AUDIO_DIRECTION
        return match[1] + detail + "\n\n"

    result = re.sub(r"<d>.*?</d>", "", result, flags=re.I | re.S)
    result = re.sub(
        r"(?ms)(^\s*detailed_description\s*:)(.*?)(?=^\s*overall_soundscape\s*:)",
        visual_direction,
        result, count=1,
    )
    result = re.sub(
        r"(?ms)^\s*overall_soundscape\s*:.*?(?=^\s*non_diegetic_music\s*:)",
        "overall_soundscape: The supplied target soundtrack, preserved exactly with its original vocals, music and timing.\n\n",
        result, count=1,
    )
    return re.sub(r"(?ms)^\s*non_diegetic_music\s*:.*\Z",
                  "non_diegetic_music: N/A — music is already in the supplied target soundtrack.", result)
