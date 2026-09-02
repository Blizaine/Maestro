"""Structured MiniMax H3 sliding-window prompt planning.

The generic rolling-window scheduler can select a different prompt for every
pass, but historically H3 received the same complete timeline each time.  This
module plans the timeline once, then deterministically compiles one compact
Context-IR prompt per continuation window.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable

from services.h3_prompt_budget import (
    H3_PROMPT_QUALITY_TARGET,
    fit_h3_base_prompt,
)
from services.h3_story_ledger import (
    H3DialogueTimingError,
    H3_STORY_LEDGER_VERSION,
    UNREQUESTED_SPECTACLE_PATTERNS,
    extract_h3_source_intent,
    normalize_h3_planning_style,
    plan_h3_story_segments,
    recover_h3_plain_story,
    sanitize_h3_nonverbal_audio,
    sanitize_h3_prompt_text,
)

_H3_WINDOW_PLANNER_VERSION = 4 + H3_STORY_LEDGER_VERSION
_CAMERA_COVERAGE_VALUES = {"auto", "continuous", "multi_shot"}
_H3_INJECTED_POSITION_RE = re.compile(r"^[Ww](\d+):(\d{1,3})$")


def normalize_h3_camera_coverage(value: Any) -> str:
    """Return the stable camera-coverage value used by UI and API callers."""

    normalized = str(value or "auto").strip().casefold().replace("-", "_")
    return normalized if normalized in _CAMERA_COVERAGE_VALUES else "auto"


def compute_h3_window_boundaries(
    total_frames: int,
    window_frames: int,
    *,
    fps: float = 24.0,
    overlap_frames: int = 1,
    discard_frames: int = 0,
) -> list[dict[str, Any]]:
    """Return the exact committed-output span owned by every H3 pass."""

    total = max(1, int(total_frames))
    window = max(1, int(window_frames))
    fps_value = max(1.0, float(fps))
    overlap = max(0, min(window - 1, int(overlap_frames)))
    discard = max(0, min(window - overlap - 1, int(discard_frames)))
    stride = max(1, window - overlap - discard)

    if total <= window:
        count = 1
    else:
        left_after_first = total - window + discard
        count = 1 + int(math.ceil(left_after_first / float(stride)))

    boundaries: list[dict[str, Any]] = []
    committed_start = 0
    for index in range(count):
        committed_length = window if index == 0 else stride
        committed_end = min(total, committed_start + committed_length)
        boundaries.append(
            {
                "index": index + 1,
                "start_frame": committed_start,
                "end_frame": committed_end,
                "start_seconds": round(committed_start / fps_value, 3),
                "end_seconds": round(committed_end / fps_value, 3),
            }
        )
        committed_start = committed_end
    return boundaries


def parse_h3_manual_window_prompts(
    prompt: str,
    *,
    expected_count: int,
    window_prompts: Iterable[Any] | None = None,
) -> list[str]:
    """Return exactly one non-empty user-authored prompt per FL2VA pass.

    Newer Studio clients submit the already-split ``h3_window_prompts``
    array. Older saved settings and API clients may only carry newline-
    separated ``prompt`` text, so retain that as a durable fallback instead
    of letting the complete multi-line story reach every continuation pass.
    """

    if isinstance(window_prompts, (list, tuple)):
        prompts = [
            item.strip()
            for item in window_prompts
            if isinstance(item, str) and item.strip()
        ]
    else:
        prompts = [
            line.strip()
            for line in str(prompt or "").replace("\r\n", "\n").replace(
                "\r", "\n"
            ).split("\n")
            if line.strip()
        ]

    expected = max(1, int(expected_count))
    if len(prompts) != expected:
        raise ValueError(
            "Manual MiniMax H3 First / Last sequence needs exactly "
            f"{expected} non-empty prompt "
            f"{'line' if expected == 1 else 'lines'} "
            f"(window 1 through window {expected}); received {len(prompts)}. "
            "Add or remove prompt lines, or adjust Duration / Window Length."
        )
    return prompts


def normalize_h3_injected_keyframes(
    injected_keyframes: Iterable[dict[str, Any]] | None,
    boundaries: Iterable[dict[str, Any]],
    *,
    fps: float = 24.0,
) -> list[dict[str, Any]]:
    """Resolve UI frame tokens onto the exact committed H3 timeline.

    The Studio UI stores symbolic ``W<n>:<percent>`` positions so the backend,
    rather than browser-side seconds math, owns the final quantized placement.
    Numeric positions remain the legacy one-based absolute-frame form and
    ``L`` advances across successive window ends.
    """

    spans = list(boundaries)
    if not spans:
        return []
    fps_value = max(1.0, float(fps))
    total_frames = max(1, int(spans[-1]["end_frame"]))
    normalized: list[dict[str, Any]] = []
    legacy_end_window = 0

    for source_index, item in enumerate(injected_keyframes or []):
        if not isinstance(item, dict):
            raise ValueError("Every H3 injected keyframe must be an object.")
        path = str(item.get("path") or "").strip()
        token = str(item.get("position") or "").strip()
        if not path:
            raise ValueError("Every H3 injected keyframe needs an image path.")
        if not token:
            raise ValueError("Every H3 injected keyframe needs a timeline position.")

        match = _H3_INJECTED_POSITION_RE.fullmatch(token)
        if match:
            window_index = min(
                len(spans) - 1,
                max(0, int(match.group(1)) - 1),
            )
            percentage = min(100, max(0, int(match.group(2))))
            span = spans[window_index]
            first = int(span["start_frame"])
            last = max(first, int(span["end_frame"]) - 1)
            absolute_frame = first + int(round((last - first) * percentage / 100.0))
        elif token.casefold() == "l":
            window_index = min(legacy_end_window, len(spans) - 1)
            legacy_end_window += 1
            span = spans[window_index]
            absolute_frame = max(
                int(span["start_frame"]),
                int(span["end_frame"]) - 1,
            )
        else:
            try:
                absolute_frame = int(token) - 1
            except ValueError as error:
                raise ValueError(
                    f"Invalid H3 injected-frame position {token!r}."
                ) from error
            if not 0 <= absolute_frame < total_frames:
                raise ValueError(
                    f"H3 injected-frame position {token!r} is outside the "
                    f"{total_frames}-frame output."
                )
            window_index = next(
                (
                    index
                    for index, span in enumerate(spans)
                    if int(span["start_frame"])
                    <= absolute_frame
                    < int(span["end_frame"])
                ),
                len(spans) - 1,
            )
            span = spans[window_index]

        local_frame = absolute_frame - int(span["start_frame"])
        normalized.append(
            {
                "source_index": source_index,
                "path": path,
                "position": token,
                "absolute_frame": absolute_frame,
                "global_seconds": round(absolute_frame / fps_value, 3),
                "window": window_index + 1,
                "local_frame": local_frame,
                "local_seconds": round(local_frame / fps_value, 3),
            }
        )
    return normalized


def h3_window_plan_signature(
    prompt: str,
    *,
    model_type: str,
    resolution: str,
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
    discard_frames: int,
    fps: float,
    has_start_image: bool,
    has_end_image: bool,
    camera_coverage: str = "auto",
    planning_style: str = "faithful",
    injected_keyframes: list[dict[str, Any]] | None = None,
) -> str:
    """Fingerprint every input that can change a window plan."""

    payload = {
        "planner_version": _H3_WINDOW_PLANNER_VERSION,
        "prompt": str(prompt or "").strip(),
        "model_type": str(model_type or ""),
        "resolution": str(resolution or ""),
        "total_frames": int(total_frames),
        "window_frames": int(window_frames),
        "overlap_frames": int(overlap_frames),
        "discard_frames": int(discard_frames),
        "fps": round(float(fps), 6),
        "has_start_image": bool(has_start_image),
        "has_end_image": bool(has_end_image),
        "camera_coverage": normalize_h3_camera_coverage(camera_coverage),
        "planning_style": normalize_h3_planning_style(planning_style),
        "injected_keyframes": [
            {
                "path": str(item.get("path") or ""),
                "position": str(item.get("position") or ""),
            }
            for item in (injected_keyframes or [])
            if isinstance(item, dict)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def reviewed_h3_window_plan_matches(
    plan: Any,
    window_prompts: Any,
    *,
    source_prompt: str,
    model_type: str,
    resolution: str,
    window_frames: int,
    boundaries: Iterable[dict[str, Any]],
    camera_coverage: str = "auto",
    planning_style: str = "faithful",
) -> bool:
    """Validate a user-reviewed H3 plan without trusting a stale signature.

    Studio keeps the short story concept separate from the compiled prompt for
    every native H3 pass.  Once those prompts are visible (and potentially
    edited), Generate or Add to Queue must preserve that exact snapshot.  A
    planner-version bump or a harmless client/server signature disagreement
    must not silently invoke the LLM again and replace the reviewed prompts.

    Geometry is compared window by window, so duration, overlap, discard, and
    window-length changes still invalidate the plan.  The UI also clears its
    reviewed plan whenever media anchors, model, resolution, or the source
    concept changes.
    """

    if not isinstance(plan, dict) or not isinstance(window_prompts, (list, tuple)):
        return False
    prompts = [
        item.strip()
        for item in window_prompts
        if isinstance(item, str) and item.strip()
    ]
    expected = [item for item in boundaries if isinstance(item, dict)]
    windows = plan.get("windows")
    if not prompts or not isinstance(windows, list):
        return False
    if len(prompts) != len(expected) or len(windows) != len(expected):
        return False
    if str(plan.get("plan_kind") or "sliding_window") != "sliding_window":
        return False
    if str(plan.get("source_prompt") or "").strip() != str(source_prompt or "").strip():
        return False
    if str(plan.get("model_type") or "") != str(model_type or ""):
        return False
    if str(plan.get("resolution") or "") != str(resolution or ""):
        return False
    if int(plan.get("window_frames") or 0) != int(window_frames):
        return False
    if normalize_h3_camera_coverage(plan.get("camera_coverage")) != normalize_h3_camera_coverage(camera_coverage):
        return False
    if normalize_h3_planning_style(plan.get("planning_style")) != normalize_h3_planning_style(planning_style):
        return False

    for index, (window, boundary, prompt) in enumerate(
        zip(windows, expected, prompts),
        start=1,
    ):
        if not isinstance(window, dict):
            return False
        try:
            geometry_matches = (
                int(window.get("index") or index) == int(boundary.get("index") or index)
                and int(window.get("start_frame") or 0) == int(boundary.get("start_frame") or 0)
                and int(window.get("end_frame") or 0) == int(boundary.get("end_frame") or 0)
            )
        except (TypeError, ValueError):
            return False
        if not geometry_matches or str(window.get("prompt") or "").strip() != prompt:
            return False
    return True


def _compact(value: Any, limit: int) -> str:
    text = sanitize_h3_prompt_text(value).strip(" \t\r\n-.,;:!?")
    if len(text) <= limit:
        return text
    prefix = text[: limit + 1]
    # Prefer a complete sentence, then a complete clause. The old hard word
    # cut produced fragments such as "The road continues toward the.." and
    # "His posture is..", which are actively harmful inside an H3 prompt.
    sentence_ends = [
        match.end()
        for match in re.finditer(r"[.!?](?=\s|$)", prefix)
        if match.end() >= max(36, int(limit * 0.35))
    ]
    if sentence_ends:
        shortened = prefix[: sentence_ends[-1]]
    else:
        clause_ends = [
            match.start()
            for match in re.finditer(r"[,;:](?=\s|$)", prefix)
            if match.start() >= max(36, int(limit * 0.45))
        ]
        if clause_ends:
            shortened = prefix[: clause_ends[-1]]
        else:
            shortened = prefix[:limit].rsplit(" ", 1)[0]
    return shortened.rstrip(" \t\r\n-.,;:!?")


def _h3_shot_timestamp(seconds: float) -> str:
    """Format official H3 cut markers as ``MM:SS.mmm``."""

    total_ms = max(0, int(round(float(seconds or 0.0) * 1000.0)))
    minutes, remainder = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _dialogue_sentence(item: Any, speaker_ids: dict[str, str]) -> str:
    if not isinstance(item, dict):
        return ""
    text = " ".join(str(item.get("text") or "").split()).strip()
    if not text:
        return ""
    speaker = _compact(item.get("speaker") or "Speaker", 80)
    key = speaker.casefold()
    requested_id = re.sub(
        r"\s*,\s*", ",", str(item.get("speaker_id") or "").upper().strip("() ")
    )
    if not re.fullmatch(r"S\d+(?:,S\d+)*", requested_id):
        assigned_numbers = [
            int(number)
            for value in speaker_ids.values()
            for number in re.findall(r"S(\d+)", value)
        ]
        requested_id = speaker_ids.get(key) or f"S{max(assigned_numbers, default=0) + 1}"
    speaker_ids.setdefault(key, requested_id)
    stable_id = speaker_ids[key]
    language = _compact(item.get("language") or "English", 30)
    delivery = _compact(item.get("delivery") or "speaks naturally", 100)
    action = _compact(item.get("action") or "", 120)
    is_voiceover = bool(re.search(
        r"\b(?:off[- ]camera|off[- ]screen|voice[- ]?over|unseen first-person)\b",
        f"{delivery} {action}",
        re.IGNORECASE,
    ))
    if is_voiceover:
        delivery = "says in an off-screen voiceover"
    if not re.search(
        r"\b(?:speaks?|says?|asks?|replies?|responds?|whispers?|shouts?|"
        r"yells?|declares?|states?|tells?|calls?\s+out)\b",
        delivery,
        flags=re.IGNORECASE,
    ):
        delivery = (
            f"speaks {delivery}"
            if re.match(r"^(?:in|with)\b", delivery, flags=re.IGNORECASE)
            else f"speaks with {delivery} delivery"
        )
    lead = f"{speaker} ({stable_id}) {delivery}"
    if action:
        lead += f" while {action}"
    if is_voiceover:
        return (
            f"{lead}: <d>[{language}] {text}</d> while the corresponding "
            "on-screen character's lips remain completely closed."
        )
    return (
        f"{lead}: <d>[{language}] {text}</d>. Immediately after the line, "
        "the speaker closes their mouth."
    )


def _window_dialogue_items(item: dict[str, Any]) -> list[Any]:
    """Read dialogue from the v2 shot list or a v1 saved planner object."""

    nested: list[Any] = []
    for shot in item.get("shots") or []:
        if isinstance(shot, dict):
            nested.extend(shot.get("dialogue") or [])
    if nested:
        return nested
    return list(item.get("dialogue") or [])


def _normalized_window_shots(
    item: dict[str, Any],
    duration: float,
) -> list[dict[str, Any]]:
    """Normalize planner shots and retain compatibility with v1 plans."""

    raw_shots = [
        shot for shot in (item.get("shots") or [])
        if isinstance(shot, dict)
    ]
    if not raw_shots:
        action = _compact(item.get("action"), 430)
        if not action:
            return []
        raw_shots = [{
            "shot": 1,
            "start_seconds": 0.0,
            "end_seconds": duration,
            "transition": "opening composition",
            "framing": "the established cinematic composition",
            "camera": "the camera follows the visible action naturally",
            "action": action,
            "dialogue": item.get("dialogue") or [],
            "sound_effects": item.get("sound_effects") or "N/A",
        }]

    normalized: list[dict[str, Any]] = []
    cursor = 0.0
    count = len(raw_shots)
    minimum_shot = min(0.75, max(0.1, duration / max(1, count * 2)))
    for index, raw in enumerate(raw_shots):
        # The planner proposes cut points, but the compiler owns a complete
        # local timeline. Snap every shot to the preceding end so malformed
        # LLM times cannot leave dead gaps or overlapping instructions.
        start = 0.0 if index == 0 else cursor
        start = min(max(0.0, start), max(0.0, duration - minimum_shot))
        default_end = duration * (index + 1) / max(1, count)
        try:
            requested_end = float(raw.get("end_seconds", default_end))
        except (TypeError, ValueError):
            requested_end = default_end
        end = duration if index + 1 == count else requested_end
        remaining_shots = count - index - 1
        latest_end = duration - minimum_shot * remaining_shots
        end = min(latest_end, max(start + minimum_shot, end))
        normalized.append({
            "shot": index + 1,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "transition": _compact(
                raw.get("transition") or ("opening composition" if index == 0 else "hard cut"),
                70,
            ),
            "framing": _compact(raw.get("framing") or "cinematic medium shot", 130),
            "camera": _compact(raw.get("camera") or "the camera follows the action", 170),
            "action": _compact(raw.get("action"), 330),
            "dialogue": list(raw.get("dialogue") or []),
            "sound_effects": _compact(raw.get("sound_effects") or "N/A", 130),
        })
        cursor = end
    normalized[-1]["end_seconds"] = round(duration, 3)
    return normalized


def _shot_prompt_sentence(
    shot: dict[str, Any],
    *,
    speaker_ids: dict[str, str],
    preamble: str = "",
) -> str:
    number = int(shot["shot"])
    start = float(shot["start_seconds"])
    end = float(shot["end_seconds"])
    transition = _compact(shot.get("transition"), 70)
    framing = _compact(shot.get("framing"), 130)
    camera = _compact(shot.get("camera"), 170)
    action = _compact(shot.get("action"), 330)

    if number == 1:
        lead = f"[Shot 1] {preamble} From {start:.2f} to {end:.2f} seconds, {framing}".strip()
    elif transition.casefold().startswith(("continuous", "without a cut", "reframe")):
        lead = (
            f"[Shot {number}] At {_h3_shot_timestamp(start)}, without a cut, "
            f"reframe to {framing}; continue through {_h3_shot_timestamp(end)}"
        )
    else:
        lead = (
            f"[Shot {number}] At {_h3_shot_timestamp(start)}, "
            f"{transition or 'hard cut'} to {framing}; continue through "
            f"{_h3_shot_timestamp(end)}"
        )
    details = "; ".join(part for part in (camera, action) if part)
    sentence = f"{lead}; {details}." if details else f"{lead}."
    dialogue = " ".join(
        value
        for value in (
            _dialogue_sentence(line, speaker_ids)
            for line in (shot.get("dialogue") or [])
        )
        if value
    )
    return f"{sentence} {dialogue}".strip()


def compile_h3_window_prompts(
    plan: dict[str, Any],
    boundaries: Iterable[dict[str, Any]],
    *,
    has_start_image: bool = False,
    has_end_image: bool = False,
    injected_keyframes: Iterable[dict[str, Any]] | None = None,
    source_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Compile a planner JSON object into complete, window-local H3 prompts."""

    spans = list(boundaries)
    windows = plan.get("windows") if isinstance(plan, dict) else None
    if not isinstance(windows, list) or len(windows) != len(spans):
        raise ValueError(
            f"H3 window planner returned {len(windows or [])} windows; expected {len(spans)}."
        )

    subjects = _compact(plan.get("subject_continuity"), 360)
    setting = _compact(plan.get("setting_continuity"), 260)
    visual = _compact(plan.get("visual_continuity"), 360)
    initial_state = _compact(plan.get("initial_state"), 320)
    ambient = _compact(
        sanitize_h3_nonverbal_audio(
            plan.get("ambient_audio") or "Natural location ambience"
        ),
        260,
    )
    music = _compact(plan.get("music") or "N/A", 180)
    source_intent = (
        plan.get("source_intent")
        if isinstance(plan.get("source_intent"), dict)
        else {}
    )
    requested_nonverbal = _compact(
        plan.get("requested_nonverbal_vocals")
        or source_intent.get("requested_nonverbal_vocals"),
        180,
    )
    sequence_shape = str(
        plan.get("sequence_shape")
        or ("ongoing" if source_intent.get("ongoing_motion") else "resolved")
    ).casefold()
    shared_visual = ". ".join(item for item in (subjects, setting, visual) if item)
    speaker_ids: dict[str, str] = {}
    compiled: list[dict[str, Any]] = []
    timed_keyframes = list(injected_keyframes or [])
    previous_closing = _compact(
        initial_state or "The requested scene is established in its opening composition",
        320,
    )

    for position, (span, item) in enumerate(zip(spans, windows)):
        if not isinstance(item, dict):
            raise ValueError(f"H3 window {position + 1} is not an object.")
        closing = _compact(item.get("closing_state"), 320)
        if not closing:
            closing = "The action holds in a concrete state ready to continue" if position + 1 < len(spans) else "The requested final beat settles naturally"

        duration = max(0.1, float(span["end_seconds"]) - float(span["start_seconds"]))
        shots = _normalized_window_shots(item, duration)
        if not shots or not any(shot.get("action") for shot in shots):
            raise ValueError(f"H3 window {position + 1} has no visible action.")
        unique_effects: list[str] = []
        seen_effects: set[str] = set()
        for shot in shots:
            for raw_effect in re.split(r"\s*;\s*", str(shot.get("sound_effects") or "")):
                effect = _compact(raw_effect, 130)
                key = effect.casefold()
                if key in {"", "n/a", "none", "no one-time effect"} or key in seen_effects:
                    continue
                seen_effects.add(key)
                unique_effects.append(effect)
        if len(unique_effects) > 1:
            unique_effects = [
                effect for effect in unique_effects
                if not effect.casefold().startswith("natural synchronized effects")
            ]
        effects = "; ".join(unique_effects)
        if not effects:
            effects = _compact(
                item.get("sound_effects") or "No one-time effect",
                150,
            )
        continuity_instruction = (
            "This is the opening continuation segment."
            if position == 0
            else "Begin by matching the supplied previous frame exactly; do not restart, recap, or repeat earlier action."
        )
        cast_handoff_instruction = (
            ""
            if position == 0
            else (
                "Every principal still present at this boundary is already in "
                "the scene in their last established position, whether visible "
                "or briefly off camera. Preserve the same identity instances and "
                "spatial relationships. A cut or reframe may reveal someone "
                "already there but must not turn that reveal into a new entrance. "
                "Do not replay completed entrances or blocking; anyone already "
                "gone remains absent."
            )
        )
        if sequence_shape == "ongoing":
            outcome_instruction = (
                "Carry the requested motion cleanly across this boundary without settling or restarting."
                if position + 1 < len(spans)
                else "Keep the requested motion active through the final frame; do not invent a stop or resolution."
            )
        else:
            outcome_instruction = (
                "Perform only this segment's assigned events; later events remain unperformed."
                if position + 1 < len(spans)
                else "Complete only the final events assigned to this segment."
            )
        dialogue_items = _window_dialogue_items(item)
        dialogue = any(
            isinstance(dialogue_item, dict)
            and str(dialogue_item.get("text") or "").strip()
            for dialogue_item in dialogue_items
        )
        nonverbal_value = (
            requested_nonverbal.partition(":")[2]
            if ":" in requested_nonverbal else requested_nonverbal
        )
        nonverbal_terms = [
            term.strip().casefold()
            for term in nonverbal_value.split(",")
            if term.strip()
        ]
        window_performance = " ".join(
            str(shot.get(field) or "")
            for shot in shots
            for field in ("action", "sound_effects")
        ).casefold()
        local_nonverbal = (
            requested_nonverbal
            if any(term in window_performance for term in nonverbal_terms)
            else ""
        )
        nonverbal_clause = f" {local_nonverbal}." if local_nonverbal else ""
        silence = (
            "Only the tagged words are spoken once in order. Outside those lines, "
            "there are no additional spoken words, muttering, or gibberish."
            + nonverbal_clause
            if dialogue
            else "No spoken words, muttering, or gibberish occur in this segment."
            + nonverbal_clause
        )

        picture_instructions: list[str] = []
        local_start_picture = position > 0 or has_start_image
        local_end_picture = has_end_image and position + 1 == len(spans)
        if local_start_picture and local_end_picture:
            last_shot = max(1, len(shots))
            picture_instructions.append(
                "How the reference pictures align with the target video - "
                "<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of "
                f"the target video; <Picture 2> (from [Shot {last_shot}]) aligns "
                f"with the {duration:.2f}-second mark of the target video."
            )
            next_picture_index = 3
        elif local_start_picture:
            picture_instructions.append(
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced."
            )
            next_picture_index = 2
        elif local_end_picture:
            last_shot = max(1, len(shots))
            picture_instructions.append(
                "How the reference pictures align with the target video - "
                f"<Picture 1> (from [Shot {last_shot}]) aligns with the "
                f"{duration:.2f}-second mark of the target video."
            )
            next_picture_index = 2
        else:
            next_picture_index = 1
        window_keyframes: list[dict[str, Any]] = []
        for keyframe in timed_keyframes:
            if not isinstance(keyframe, dict) or int(keyframe.get("window") or 0) != position + 1:
                continue
            local_seconds = max(0.0, float(keyframe.get("local_seconds") or 0.0))
            mapped = {
                **keyframe,
                "picture_index": next_picture_index,
            }
            window_keyframes.append(mapped)
            picture_instructions.append(
                f"At {local_seconds:.2f} seconds into this segment, "
                f"<Picture {next_picture_index}> is fully referenced as an exact "
                "injected frame; the visible action must arrive at it naturally "
                "and continue from it without an unrequested cut."
            )
            next_picture_index += 1

        coverage = _compact(item.get("coverage") or "auto cinematic coverage", 80)
        pacing = _compact(item.get("pacing") or "natural real-time pacing", 180)
        pacing_sentence = f"Coverage is {coverage}; pacing is {pacing}."
        if "slow motion" not in pacing.casefold():
            pacing_sentence += " Slow motion occurs only when explicitly requested."
        preamble = " ".join(
            part
            for part in (
                f"{shared_visual or 'A coherent cinematic scene'}.",
                continuity_instruction,
                (
                    f"At 0.00 seconds, continue from this exact previous-scene "
                    f"state: {previous_closing}."
                    if position > 0 else
                    f"At 0.00 seconds, {previous_closing}."
                ),
                cast_handoff_instruction,
                pacing_sentence,
            )
            if part
        )
        visual_parts = [
            _shot_prompt_sentence(
                shot,
                speaker_ids=speaker_ids,
                preamble=preamble if shot_index == 0 else "",
            )
            for shot_index, shot in enumerate(shots)
        ]
        visual_parts.extend(
            [
                silence,
                outcome_instruction,
                f"The segment ends with {closing}.",
            ]
        )
        soundscape = ambient
        if position > 0:
            soundscape += "; the same ambience continues seamlessly without restarting"
        if effects and effects.casefold() not in {"n/a", "none", "no one-time effect"}:
            soundscape += f". Synchronized effects in this segment: {effects}"
        music_value = music
        if music.casefold() != "n/a" and position > 0:
            music_value = f"{music}; continue seamlessly from the preceding segment without restarting"

        prompt_parts = picture_instructions + [
            f"integrated_multimodal_description: {' '.join(visual_parts)}",
            f"overall_soundscape: {soundscape}.",
            f"non_diegetic_music: {music_value}",
        ]
        prompt = "\n\n".join(prompt_parts)
        if (
            source_prompt is not None
            and not re.search(r"\bwindows?\b", str(source_prompt), flags=re.IGNORECASE)
            and re.search(r"\bwindows?\b", prompt, flags=re.IGNORECASE)
        ):
            raise ValueError(
                "H3 camera plan introduced the internal term 'window' into visible scene content."
            )
        budgeted_prompt = fit_h3_base_prompt(prompt)
        compiled.append(
            {
                **span,
                "title": _compact(item.get("title") or f"Window {position + 1}", 80),
                "opening_state": previous_closing,
                "closing_state": closing,
                "coverage": coverage,
                "pacing": pacing,
                "shot_count": len(shots),
                "shots": shots,
                "injected_keyframes": window_keyframes,
                "prompt": budgeted_prompt.prompt,
                "prompt_tokens": budgeted_prompt.token_count,
                "prompt_quality_target": H3_PROMPT_QUALITY_TARGET,
                "prompt_compacted": budgeted_prompt.compacted,
            }
        )
        previous_closing = closing
    return compiled


def _schema(window_count: int) -> dict[str, Any]:
    dialogue_item = {
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "speaker_id": {"type": "string"},
            "language": {"type": "string"},
            "delivery": {"type": "string"},
            "action": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["speaker", "speaker_id", "language", "delivery", "action", "text"],
        "additionalProperties": False,
    }
    shot_item = {
        "type": "object",
        "properties": {
            "shot": {"type": "integer"},
            "start_seconds": {"type": "number"},
            "end_seconds": {"type": "number"},
            "transition": {"type": "string"},
            "framing": {"type": "string"},
            "camera": {"type": "string"},
            "action": {"type": "string"},
            "dialogue": {"type": "array", "items": dialogue_item},
            "sound_effects": {"type": "string"},
        },
        "required": [
            "shot",
            "start_seconds",
            "end_seconds",
            "transition",
            "framing",
            "camera",
            "action",
            "dialogue",
            "sound_effects",
        ],
        "additionalProperties": False,
    }
    window_item = {
        "type": "object",
        "properties": {
            "window": {"type": "integer"},
            "title": {"type": "string"},
            "coverage": {"type": "string"},
            "pacing": {"type": "string"},
            "shots": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": shot_item,
            },
            "closing_state": {"type": "string"},
        },
        "required": [
            "window",
            "title",
            "coverage",
            "pacing",
            "shots",
            "closing_state",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "subject_continuity": {"type": "string"},
            "setting_continuity": {"type": "string"},
            "visual_continuity": {"type": "string"},
            "editing_style": {"type": "string"},
            "initial_state": {"type": "string"},
            "ambient_audio": {"type": "string"},
            "music": {"type": "string"},
            "windows": {
                "type": "array",
                "minItems": window_count,
                "maxItems": window_count,
                "items": window_item,
            },
        },
        "required": [
            "subject_continuity",
            "setting_continuity",
            "visual_continuity",
            "editing_style",
            "initial_state",
            "ambient_audio",
            "music",
            "windows",
        ],
        "additionalProperties": False,
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    try:
        import json_repair

        value = json_repair.loads(cleaned)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


_UNREQUESTED_SPECTACLE_PATTERNS = UNREQUESTED_SPECTACLE_PATTERNS


def _narrative_dialogue_expected(prompt: str, window_count: int) -> bool:
    """Return whether a long character interaction should not be all-mute."""

    source = " ".join(str(prompt or "").split())
    lowered = source.casefold()
    if int(window_count) < 2:
        return False
    if re.search(
        r"\b(?:silent|silently|no dialogue|without dialogue|nonverbal|"
        r"music video|montage|instrumental|landscape|establishing shot)\b",
        lowered,
    ):
        return False
    if re.search(r"\".+?\"", source):
        return True
    # An explicit verbal action is sufficient on its own. Requiring two
    # multi-word proper names missed ordinary briefs such as ``George
    # Costanza starts telling Joey ...`` and caused Creative planning to
    # compile the entire exchange as silent visual action.
    if re.search(
        r"\b(?:talk(?:s|ed|ing)?(?:\s+(?:to|with))?|convers(?:e|es|ed|ing)|"
        r"chat(?:s|ted|ting)?|tell(?:s|ing)?|told|explain(?:s|ed|ing)?|"
        r"inform(?:s|ed|ing)?|announce(?:s|d|ing)?|describe(?:s|d|ing)?|"
        r"present(?:s|ed|ing)?|say(?:s|ing)?|said|speak(?:s|ing)?|spoke|"
        r"ask(?:s|ed|ing)?|answer(?:s|ed|ing)?|repl(?:y|ies|ied|ying)|"
        r"respond(?:s|ed|ing)?|discuss(?:es|ed|ing)?|debate(?:s|d|ing)?|"
        r"argu(?:e|es|ed|ing)|jok(?:e|es|ed|ing)|banter(?:s|ed|ing)?)\b",
        lowered,
    ):
        return True
    interaction = re.search(
        r"\b(?:confront|attack|"
        r"threaten|rescue|save|protect|warn|interview|can't believe|"
        r"cannot believe|disbelie|astonish|surpris)\w*\b",
        lowered,
    )
    # Two distinct multi-token proper names are a conservative signal that
    # this is a character scene rather than a generic action montage.
    names = {
        match.group(0).casefold()
        for match in re.finditer(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
            source,
        )
    }
    return bool(interaction and len(names) >= 2)


def _creative_dialogue_expected(prompt: str, window_count: int) -> bool:
    """Let Creative write dialogue for character scenes, never silent briefs."""

    source = " ".join(str(prompt or "").split())
    lowered = source.casefold()
    if int(window_count) < 2 or re.search(
        r"\b(?:silent|silently|no dialogue|without dialogue|nonverbal|"
        r"music video|montage|instrumental|landscape|establishing shot)\b",
        lowered,
    ):
        return False
    if _narrative_dialogue_expected(source, window_count):
        return True
    return bool(re.search(
        r"\b(?:these|the)\s+(?:two|three|four|five|\d+)\s+characters?\b|"
        r"\bcharacters?\s+(?:meet|interact|debate|joke|banter|converse)\b",
        lowered,
    ))


def _plan_contract_violations(
    source_prompt: str,
    plan: dict[str, Any] | None,
    *,
    expect_dialogue: bool,
) -> list[str]:
    """Find high-impact errors that a small planner LLM may ignore."""

    if not isinstance(plan, dict):
        return ["invalid plan"]
    rendered = json.dumps(plan, ensure_ascii=False)
    lowered_source = str(source_prompt or "").casefold()
    lowered_plan = rendered.casefold()
    violations: list[str] = []

    invented = []
    for pattern in _UNREQUESTED_SPECTACLE_PATTERNS:
        generated_match = re.search(pattern, lowered_plan, flags=re.IGNORECASE)
        if (
            generated_match
            and not re.search(pattern, lowered_source, flags=re.IGNORECASE)
        ):
            invented.append(generated_match.group(0).strip())
    if invented:
        violations.append(
            "invented unrequested power/effect: "
            + ", ".join(sorted(set(invented)))
        )

    windows = plan.get("windows") or []
    timeline_text = json.dumps(
        [
            {
                "action": item.get("action"),
                "closing_state": item.get("closing_state"),
                "sound_effects": item.get("sound_effects"),
                "dialogue": item.get("dialogue"),
                "shots": item.get("shots"),
            }
            for item in windows
            if isinstance(item, dict)
        ],
        ensure_ascii=False,
    ).casefold()
    if re.search(
        r"(?:\bglobal\s+\d+(?:\.\d+)?\s*s|\bwindow\s+[1-9]\b)",
        timeline_text,
        flags=re.IGNORECASE,
    ):
        violations.append("used a global window label/timestamp inside a local shot")

    for index, item in enumerate(windows):
        if not isinstance(item, dict):
            continue
        shots = item.get("shots")
        if shots is not None and not any(
            isinstance(shot, dict) and str(shot.get("action") or "").strip()
            for shot in shots
        ):
            violations.append(f"window {index + 1} has no visible shot action")

    if expect_dialogue:
        has_dialogue = any(
            isinstance(item, dict)
            and any(
                isinstance(line, dict)
                and str(line.get("text") or "").strip()
                for line in _window_dialogue_items(item)
            )
            for item in windows
        )
        if not has_dialogue:
            violations.append(
                "left a long named-character interaction entirely mute"
            )

    return violations


def _infer_camera_coverage(prompt: str, requested: str = "auto") -> str:
    requested = normalize_h3_camera_coverage(requested)
    if requested != "auto":
        return requested
    lowered = str(prompt or "").casefold()
    # A user-authored POV is a perspective lock, not merely a suggested shot
    # size. Keep one continuous viewpoint unless the request explicitly asks
    # for edits/cuts; generic action heuristics must not cut outside the POV.
    if re.search(r"\b(?:pov|first[- ]person)\b", lowered) and not re.search(
        r"\b(?:multi[- ]shot|camera cuts?|hard cuts?|cut to|montage)\b",
        lowered,
    ):
        return "continuous"
    if re.search(
        r"\b(?:single[- ]take|one[- ]take|unbroken|continuous tracking|no cuts?)\b",
        lowered,
    ):
        return "continuous"
    if re.search(
        r"\b(?:fight|battle|chase|attack|trailer|montage|high[- ]speed|"
        r"high rate of speed|fast[- ]paced|rapid|plummet|free[- ]?fall|"
        r"breakneck|dynamic action|argument|interview|conversation|"
        r"talk|dialogue|multi[- ]shot|camera cuts?|shot[- /]reverse[- ]shot)\b",
        lowered,
    ):
        return "multi_shot"
    return "continuous"


def _fallback_plan(
    prompt: str,
    window_count: int,
    *,
    window_durations: list[float] | None = None,
    camera_coverage: str = "auto",
) -> dict[str, Any]:
    """A no-LLM fallback that never exposes the full plot to every window."""

    source = " ".join(str(prompt or "").split()).strip()
    source = re.sub(
        r"^(?:integrated_multimodal_description|overall_soundscape|non_diegetic_music):\s*",
        "",
        source,
        flags=re.IGNORECASE,
    )
    # Split explicit temporal transitions and common "setup + obligation"
    # concepts.  The latter matters for prompts such as "walks through town
    # and has to save a truck": leaving that whole sentence in pass one lets
    # H3 see (and finish) the rescue before a continuation begins.
    beat_separator = re.compile(
        r"(?<=[.!?])\s+|\s*;\s*|"
        r"\s+(?:and\s+then|then|after\s+that|next)\s+|"
        r"\s+and\s+(?:has|needs|must|tries|attempts)\s+to\s+",
        flags=re.IGNORECASE,
    )
    beats = [
        part.strip(" ,;:-")
        for part in beat_separator.split(source)
        if part.strip(" ,;:-")
    ]
    if not beats:
        beats = ["Establish and begin the requested action"]
    beat_buckets: list[list[str]] = [[] for _ in range(window_count)]
    if len(beats) == 1:
        beat_buckets[0].append(beats[0])
    else:
        for beat_index, beat in enumerate(beats):
            target = int(round(beat_index * (window_count - 1) / (len(beats) - 1)))
            beat_buckets[min(window_count - 1, max(0, target))].append(beat)
    resolved_coverage = _infer_camera_coverage(prompt, camera_coverage)
    lowered_source = source.casefold()
    fast_action = bool(re.search(
        r"\b(?:fight|battle|chase|attack|high[- ]speed|fast[- ]paced|rapid)\b",
        lowered_source,
    ))
    windows = []
    for index in range(window_count):
        assigned = beat_buckets[index]
        if index == 0:
            opening_beat = assigned or beats[:1]
            action = f"Establish the scene and begin: {' '.join(opening_beat)}. Do not complete the central outcome"
        elif index + 1 == window_count:
            remaining = " ".join(assigned).strip()
            action = "Continue from the prior physical state and complete the requested central action and outcome"
            if remaining:
                action += f": {remaining}"
        else:
            middle = " ".join(assigned).strip()
            action = "Continue from the prior physical state and visibly advance toward the central outcome"
            if middle:
                action += f": {middle}"
            action += ". Keep the outcome unresolved"
        duration = (
            float(window_durations[index])
            if window_durations and index < len(window_durations)
            else 10.0
        )
        shot_count = 1
        if resolved_coverage == "multi_shot":
            shot_count = 3 if fast_action else 2
        shots = []
        for shot_index in range(shot_count):
            start = duration * shot_index / shot_count
            end = duration * (shot_index + 1) / shot_count
            if shot_count == 1:
                framing = "a coherent cinematic composition"
                camera = "a motivated tracking or locked camera follows the action"
                shot_action = action
            elif shot_index == 0:
                framing = "a readable wide or medium-wide view of the geography"
                camera = "a fast tracking move establishes positions and direction"
                shot_action = f"Begin this segment's assigned beat: {action}"
            elif shot_index + 1 == shot_count:
                framing = "a tighter impact or reaction angle"
                camera = "a responsive handheld or low-angle move follows through, then settles clearly"
                shot_action = "Complete only this segment's assigned beat without repeating its opening"
            else:
                framing = "a dynamic medium action angle"
                camera = "a hard cut and rapid lateral tracking move maintain real-time speed"
                shot_action = "Continue the same beat with new physical choreography and no slow motion"
            shots.append({
                "shot": shot_index + 1,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "transition": "opening composition" if shot_index == 0 else "hard cut",
                "framing": framing,
                "camera": camera,
                "action": shot_action,
                "dialogue": [],
                "sound_effects": "Natural synchronized effects for the visible action",
            })
        windows.append(
            {
                "window": index + 1,
                "title": f"Continuation {index + 1}",
                "coverage": resolved_coverage,
                "pacing": (
                    "fast real-time action with sharp impacts and no slow motion"
                    if fast_action
                    else "natural real-time pacing"
                ),
                # Keep the legacy summary for saved plans and older callers;
                # inference uses the structured shot list below.
                "action": action,
                "shots": shots,
                "closing_state": (
                    "The visible action pauses in a concrete position ready for the next continuation"
                    if index + 1 < window_count
                    else "The requested outcome settles into a clear final beat"
                ),
            }
        )
    return {
        "subject_continuity": "Keep every established subject's identity, appearance, wardrobe, and carried objects unchanged",
        "setting_continuity": "Keep the established location, geography, time of day, and background elements unchanged",
        "visual_continuity": "Keep lighting, color, screen direction, and established geography coherent across coverage",
        "editing_style": (
            "Motivated cinematic cuts and dynamic H3 camera coverage"
            if resolved_coverage == "multi_shot"
            else "One continuous motivated camera move"
        ),
        "initial_state": "The requested scene is established and the subjects begin in natural positions",
        "ambient_audio": "Continuous natural ambience appropriate to the established location",
        "music": "N/A",
        "windows": windows,
    }


def plan_h3_sliding_windows(
    prompt: str,
    *,
    model_type: str,
    resolution: str,
    total_frames: int,
    window_frames: int,
    overlap_frames: int = 1,
    discard_frames: int = 0,
    fps: float = 24.0,
    has_start_image: bool = False,
    has_end_image: bool = False,
    image_paths: list[str] | None = None,
    injected_keyframes: list[dict[str, Any]] | None = None,
    nsfw: bool = False,
    camera_coverage: str = "auto",
    planning_style: str = "faithful",
) -> dict[str, Any]:
    """Use Maestro's configured LLM to create and compile an H3 window plan."""

    raw_prompt = str(prompt or "").strip()
    prompt = recover_h3_plain_story(raw_prompt)
    source_intent = extract_h3_source_intent(prompt)
    if prompt != raw_prompt and "subject_definitions:" in raw_prompt.casefold():
        print(
            "[MiniMax H3] Recovered the original story from an existing "
            "single-clip Context-IR prompt before sliding-window planning."
        )

    boundaries = compute_h3_window_boundaries(
        total_frames,
        window_frames,
        fps=fps,
        overlap_frames=overlap_frames,
        discard_frames=discard_frames,
    )
    camera_coverage = normalize_h3_camera_coverage(camera_coverage)
    planning_style = normalize_h3_planning_style(planning_style)
    normalized_keyframes = normalize_h3_injected_keyframes(
        injected_keyframes,
        boundaries,
        fps=fps,
    )
    signature = h3_window_plan_signature(
        prompt,
        model_type=model_type,
        resolution=resolution,
        total_frames=total_frames,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        discard_frames=discard_frames,
        fps=fps,
        has_start_image=has_start_image,
        has_end_image=has_end_image,
        camera_coverage=camera_coverage,
        planning_style=planning_style,
        injected_keyframes=injected_keyframes,
    )
    if len(boundaries) <= 1:
        return {
            "source_prompt": str(prompt or ""),
            "signature": signature,
            "planned_by": "not_needed",
            "total_frames": int(total_frames),
            "window_frames": int(window_frames),
            "window_count": 1,
            "plan_kind": "sliding_window",
            "camera_coverage": camera_coverage,
            "planning_style": planning_style,
            "injected_keyframes": normalized_keyframes,
            "windows": [],
            "window_prompts": [],
        }

    media_context = []
    if has_start_image:
        media_context.append("<Picture 1> is the exact first frame and visual identity/scene anchor.")
    if has_end_image:
        picture_index = 2 if has_start_image else 1
        media_context.append(f"<Picture {picture_index}> is the required final frame destination.")
    attachment_picture_index = int(has_start_image) + int(has_end_image)
    for keyframe in normalized_keyframes:
        attachment_picture_index += 1
        media_context.append(
            f"<Picture {attachment_picture_index}> is an exact injected visual "
            f"anchor at {keyframe['global_seconds']:.3f}s on the full timeline "
            f"(segment {keyframe['window']}, {keyframe['local_seconds']:.3f}s local)."
        )
    expect_dialogue = (
        _creative_dialogue_expected(prompt, len(boundaries))
        if planning_style == "creative"
        else bool(re.search(r"[\"“][^\"”]+[\"”]", prompt))
    )
    resolved_coverage = _infer_camera_coverage(prompt, camera_coverage)
    story_ledger: dict[str, Any] | None = None
    planning_warnings: list[str] = []
    planning_diagnostics: list[str] = []
    planning_notes: list[str] = []
    dialogue_fragments: list[dict[str, Any]] = []
    try:
        staged = plan_h3_story_segments(
            prompt,
            segment_durations=[
                float(span["end_seconds"]) - float(span["start_seconds"])
                for span in boundaries
            ],
            mode="sliding_window",
            camera_coverage=resolved_coverage,
            reference_context=" ".join(media_context),
            expect_dialogue=expect_dialogue,
            planning_style=planning_style,
            image_paths=image_paths,
            nsfw=nsfw,
        )
        planned_by = staged["planned_by"]
        planning_warnings = list(staged.get("planning_warnings") or [])
        planning_diagnostics = list(staged.get("planning_diagnostics") or [])
        planning_notes = list(staged.get("planning_notes") or [])
        dialogue_fragments = list(staged.get("dialogue_fragments") or [])
        source_intent = staged.get("source_intent") or source_intent
        story_ledger = staged["ledger"]
        windows = []
        for index, segment in enumerate(staged["segments"]):
            windows.append({
                **segment,
                "window": index + 1,
            })
        plan = {
            "subject_continuity": story_ledger.get("subject_continuity", ""),
            "setting_continuity": story_ledger.get("setting_continuity", ""),
            "visual_continuity": story_ledger.get("visual_continuity", ""),
            "editing_style": story_ledger.get("editing_style", ""),
            "initial_state": story_ledger.get("initial_state", ""),
            "ambient_audio": story_ledger.get("ambient_audio", ""),
            "music": story_ledger.get("music", "N/A"),
            "source_intent": source_intent,
            "requested_nonverbal_vocals": story_ledger.get(
                "requested_nonverbal_vocals",
                source_intent.get("requested_nonverbal_vocals", ""),
            ),
            "sequence_shape": story_ledger.get(
                "sequence_shape",
                "ongoing" if source_intent.get("ongoing_motion") else "resolved",
            ),
            "windows": windows,
        }
        compiled = compile_h3_window_prompts(
            plan,
            boundaries,
            has_start_image=has_start_image,
            has_end_image=has_end_image,
            injected_keyframes=normalized_keyframes,
            source_prompt=prompt,
        )
    except H3DialogueTimingError:
        raise
    except Exception as error:
        print(f"[MiniMax H3] Window planner fallback: {error}")
        planned_by = "deterministic_fallback"
        planning_warnings.append(
            "The H3 window compiler could not use the AI camera plan, so "
            "Maestro generated a source-faithful deterministic plan."
        )
        plan = _fallback_plan(
            prompt,
            len(boundaries),
            window_durations=[
                float(span["end_seconds"]) - float(span["start_seconds"])
                for span in boundaries
            ],
            camera_coverage=camera_coverage,
        )
        plan["source_intent"] = source_intent
        plan["requested_nonverbal_vocals"] = source_intent.get(
            "requested_nonverbal_vocals", ""
        )
        plan["sequence_shape"] = (
            "ongoing" if source_intent.get("ongoing_motion") else "resolved"
        )
        compiled = compile_h3_window_prompts(
            plan,
            boundaries,
            has_start_image=has_start_image,
            has_end_image=has_end_image,
            injected_keyframes=normalized_keyframes,
            source_prompt=prompt,
        )

    return {
        "source_prompt": str(prompt or ""),
        "signature": signature,
        "planned_by": planned_by,
        "planning_warnings": list(dict.fromkeys(planning_warnings)),
        "planning_diagnostics": list(dict.fromkeys(planning_diagnostics)),
        "planning_notes": list(dict.fromkeys(planning_notes)),
        "total_frames": int(total_frames),
        "window_frames": int(window_frames),
        "window_count": len(compiled),
        "plan_kind": "sliding_window",
        "camera_coverage": camera_coverage,
        "planning_style": planning_style,
        "resolution": str(resolution or ""),
        "model_type": str(model_type or ""),
        "injected_keyframes": normalized_keyframes,
        "subject_continuity": plan.get("subject_continuity", ""),
        "setting_continuity": plan.get("setting_continuity", ""),
        "editing_style": plan.get("editing_style", ""),
        "story_ledger": story_ledger,
        "dialogue_fragments": dialogue_fragments,
        "source_intent": source_intent,
        "windows": compiled,
        "window_prompts": [item["prompt"] for item in compiled],
    }
