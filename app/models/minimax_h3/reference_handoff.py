"""Pure diagnostics for the Ref2VA prompt/reference handoff.

This module deliberately shares the same prompt compiler and voice-window
selector as the render path. It does not inspect or decode media files.
"""

from __future__ import annotations

import hashlib
import math
import re
import copy
from typing import Any

from services.h3_reference_audio import video_audio_contributes_reference_ordinal

from .ref2va import (
    _DIALOGUE_TAG_RE,
    _build_ref2va_character_bindings,
    _ref2va_event_speaker_marker,
    _ref2va_voice_audio_by_subject,
    add_ref2va_continuation_context,
    align_ref2va_voice_reference_order,
    ensure_ref2va_prompt_relationships,
    select_ref2va_window_voice_references,
)
from .speakers import (
    _ref2va_declared_marker_subjects,
    _ref2va_direct_subject_speaker_cue,
    _resolve_ref2va_dialogue_owners,
    is_h3_spoken_quote,
)


_SPOKEN_QUOTE_RE = re.compile(r'"[^"\r\n]{1,500}"|“[^”\r\n]{1,500}”')
_SUBJECT_DEFINITIONS_RE = re.compile(
    r"(?ms)^\s*subject_definitions\s*:(.*?)(?=^\s*summary\s*:)"
)
_ABSOLUTE_PATH_RE = re.compile(
    r"^(?:[A-Za-z]:[\\/]|\\\\|/)|^[A-Za-z][A-Za-z0-9+.-]*://"
)


def _json_safe_alignment(value: dict[str, dict[int, int]]) -> dict[str, dict[str, int]]:
    return {
        str(kind): {str(old): int(new) for old, new in mapping.items()}
        for kind, mapping in (value or {}).items()
    }


def _public_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or _ABSOLUTE_PATH_RE.match(text):
        return None
    return text


def _public_id(item: dict[str, Any]) -> str | None:
    value = item.get("id") or item.get("reference_id") or item.get("ref_id")
    text = _public_text(value)
    # IDs can be caller supplied, so avoid accidentally publishing a path
    # stored in an ID field as well as paths stored in media fields.
    if text and ("/" in text or "\\" in text):
        return None
    return text


def _reference_signature(item: dict[str, Any]) -> tuple[Any, ...]:
    """Fields stable across the selector's validate/copy pass, excluding paths."""

    return tuple(
        item.get(key)
        for key in (
            "id", "reference_id", "ref_id", "type", "kind", "library_character_id",
            "character_name", "role", "image_intent", "video_intent", "audio_intent",
            "has_audio", "include_audio", "video_audio_role", "audio_path",
            "_h3_video_audio_source_id",
        )
    )


def _selected_reference_indexes(
    aligned: list[dict[str, Any]],
    scoped: list[dict[str, Any]],
) -> list[int | None]:
    """Match the selector's stable subsequence without path-based guessing."""

    output: list[int | None] = []
    cursor = 0
    for item in scoped:
        signature = _reference_signature(item)
        match = next(
            (index for index in range(cursor, len(aligned)) if _reference_signature(aligned[index]) == signature),
            None,
        )
        output.append(match)
        if match is not None:
            cursor = match + 1
    return output


def _reference_bindings(
    references: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    bindings: list[dict[str, Any]] = []
    visual_by_subject: dict[str, dict[str, Any]] = {}
    picture_index = 0
    video_index = 0
    audio_index = 0

    for reference_index, item in enumerate(references):
        kind = str(item.get("type") or item.get("kind") or "").strip().lower()
        character_id = _public_text(item.get("library_character_id") or item.get("character_id"))
        name = _public_text(item.get("character_name") or item.get("name") or item.get("role"))
        row: dict[str, Any] = {
            "reference_index": reference_index,
            "type": kind,
            "reference_id": _public_id(item),
            "character_id": character_id,
            "name": name,
            "embedded_audio_enabled": False,
        }

        if kind == "image":
            picture_index += 1
            row.update(label=f"Picture {picture_index}", type_index=picture_index)
        elif kind == "video":
            video_index += 1
            row.update(label=f"Video {video_index}", type_index=video_index)
            role = str(item.get("video_audio_role") or "").strip().lower()
            include_audio = item.get("include_audio", True) is not False
            has_audio = item.get("has_audio")
            if role == "ignore" or not include_audio or has_audio is False:
                embedded_enabled: bool | None = False
            elif item.get("audio_path"):
                # An attached soundtrack is an external audio source, not an
                # embedded stream in the video row.
                embedded_enabled = False
            elif role and role != "legacy":
                # Typed video audio roles are routed to standalone audio rows.
                embedded_enabled = False
            elif has_audio is True:
                embedded_enabled = True
            else:
                embedded_enabled = None
            row["embedded_audio_enabled"] = embedded_enabled
        elif kind == "audio":
            audio_index += 1
            row.update(label=f"Audio {audio_index}", type_index=audio_index)
        else:
            row.update(label=f"{kind or 'reference'} {reference_index + 1}", type_index=reference_index + 1)

        has_audio_ordinal = kind == "audio" or (
            kind == "video" and video_audio_contributes_reference_ordinal(item)
        )
        if kind == "video" and has_audio_ordinal:
            audio_index += 1
            row["audio_label"] = f"Audio {audio_index}"
        elif kind == "audio":
            # The regular audio branch already allocated this label above.
            row["audio_label"] = row["label"]

        bindings.append(row)

    (
        reference_subjects,
        _character_subjects,
        _role_subjects,
        _speaker_aliases,
        _character_subject_count,
    ) = _build_ref2va_character_bindings(references)
    for item_index, subject in reference_subjects.items():
        if 0 <= item_index < len(bindings):
            visual_by_subject.setdefault(str(subject), bindings[item_index])
    return bindings, visual_by_subject


def _dialogue_events(prompt: str) -> list[tuple[int, int, str]]:
    events = [
        (match.start(), match.end(), match.group(1).strip())
        for match in _DIALOGUE_TAG_RE.finditer(prompt)
    ]
    for match in _SPOKEN_QUOTE_RE.finditer(prompt):
        if not is_h3_spoken_quote(prompt, match):
            continue
        if any(start < match.end() and match.start() < end for start, end, _text in events):
            continue
        events.append((match.start(), match.end(), match.group(0).strip()))
    return sorted(events, key=lambda event: event[0])


def _plain_subject_speaker_cue(
    source: str,
    dialogue_start: int,
    valid_subjects: set[int],
) -> int | None:
    """Mirror the resolver's plain ``<Subject N>`` cue for diagnostics.

    The shared owner resolver accepts the last valid Subject tag in the
    current clause. Keep this check in step with that branch so a one-character
    fallback is not mistaken for authored ownership when the tag was explicit.
    """

    before_start = max(0, int(dialogue_start) - 360)
    before = str(source or "")[before_start:int(dialogue_start)]
    clause_start = max(
        before.rfind("."),
        before.rfind("!"),
        before.rfind("?"),
        before.rfind(";"),
        before.rfind("\n"),
    ) + 1
    explicit_subjects = re.findall(
        r"<Subject\s+(\d+)>", before[clause_start:], flags=re.IGNORECASE,
    )
    if not explicit_subjects:
        return None
    subject = int(explicit_subjects[-1])
    return subject if subject in valid_subjects else None


def _speaker_diagnostics(
    prompt: str,
    references: list[dict[str, Any]],
    reference_bindings: list[dict[str, Any]],
    visual_by_subject: dict[str, dict[str, Any]],
    *,
    ownership_prompt: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    (
        _reference_subjects,
        character_subjects,
        _role_subjects,
        speaker_aliases,
        character_subject_count,
    ) = _build_ref2va_character_bindings(references)
    valid_subjects = set(range(1, character_subject_count + 1))
    events = _dialogue_events(prompt)
    event_spans = [(start, end, text) for start, end, text in events]
    source_prompt = prompt if ownership_prompt is None else str(ownership_prompt)
    source_events = _dialogue_events(source_prompt)
    resolution_events = source_events if len(source_events) == len(events) else events
    resolution_prompt = source_prompt if len(source_events) == len(events) else prompt
    resolution_spans = [(start, end, text) for start, end, text in resolution_events]
    resolved = _resolve_ref2va_dialogue_owners(
        resolution_prompt,
        resolution_spans,
        speaker_aliases,
        valid_subjects,
        strict=False,
    ) if events else []
    compiled_resolved = _resolve_ref2va_dialogue_owners(
        prompt,
        event_spans,
        speaker_aliases,
        valid_subjects,
        strict=False,
    ) if events else []
    audio_by_subject = _ref2va_voice_audio_by_subject(
        references,
        character_subjects,
        _role_subjects,
    )
    reference_by_audio_label = {
        row["audio_label"]: row
        for row in reference_bindings
        if row.get("audio_label")
    }
    definitions = _SUBJECT_DEFINITIONS_RE.search(prompt)
    definitions_span = (definitions.start(1), definitions.end(1)) if definitions else None
    resolution_definitions = _SUBJECT_DEFINITIONS_RE.search(resolution_prompt)
    resolution_definitions_span = (
        (resolution_definitions.start(1), resolution_definitions.end(1))
        if resolution_definitions else None
    )

    # Match Ref2VA's per-prompt event order so a known later speaker can
    # correctly remain S2 when an earlier line has an unknown owner.
    local_speakers: dict[tuple[str, Any], int] = {}
    speaker_bindings: dict[tuple[Any, ...], dict[str, Any]] = {}
    dialogue_bindings: list[dict[str, Any]] = []
    previous_end = 0
    source_previous_end = 0
    for event_index, ((start, end, text), (subject, owner_name), (source_start, source_end, _source_text)) in enumerate(
        zip(events, resolved, resolution_events),
        start=1,
    ):
        marker = _ref2va_event_speaker_marker(
            prompt,
            start,
            previous_end,
            definitions_span,
        )
        compiled_subject = compiled_resolved[event_index - 1][0]
        if subject is not None and character_subject_count == 1 and not owner_name:
            explicit_subject = _ref2va_direct_subject_speaker_cue(
                resolution_prompt,
                source_start,
                valid_subjects,
            )
            plain_subject = _plain_subject_speaker_cue(
                resolution_prompt,
                source_start,
                valid_subjects,
            )
            source_marker = _ref2va_event_speaker_marker(
                resolution_prompt,
                source_start,
                source_previous_end,
                resolution_definitions_span,
            )
            declared_subjects = _ref2va_declared_marker_subjects(
                resolution_prompt,
                valid_subjects,
            )
            has_explicit_evidence = bool(
                explicit_subject
                or plain_subject == subject
                or (
                    source_marker
                    and declared_subjects.get(int(source_marker.group(1))) == subject
                )
            )
            if not has_explicit_evidence:
                # The owner resolver has a deliberate one-character fallback.
                # Keep its outcome visible as compiled_subject, but don't
                # present that fallback as authored ownership evidence.
                subject = None
        speaker_key: tuple[str, Any] = (
            ("subject", subject)
            if subject is not None
            else ("name", owner_name.casefold())
            if owner_name
            else ("event", event_index)
        )
        expected_speaker = local_speakers.setdefault(speaker_key, len(local_speakers) + 1)
        speaker_id = int(marker.group(1)) if marker is not None else expected_speaker
        visual = visual_by_subject.get(str(subject)) if subject is not None else None
        audio_ordinal = audio_by_subject.get(subject) if subject is not None else None
        voice_label = f"Audio {audio_ordinal}" if audio_ordinal is not None else None
        voice_reference = reference_by_audio_label.get(voice_label) if voice_label else None
        clean_owner = _public_text(owner_name)
        display_name = (visual or {}).get("name") or clean_owner
        binding_status = "bound" if subject is not None and visual is not None else "unbound" if clean_owner else "unknown"
        dialogue_row = {
            "dialogue_index": event_index,
            "text": text,
            "local_speaker": f"S{speaker_id}" if speaker_id is not None else None,
            "subject": f"Subject {subject}" if subject is not None else None,
            "compiled_subject": f"Subject {compiled_subject}" if compiled_subject is not None else None,
            "character_id": (visual or {}).get("character_id"),
            "name": display_name,
            "reference_id": (visual or {}).get("reference_id"),
            "audio_label": voice_label,
            "voice_reference_id": (voice_reference or {}).get("reference_id"),
            "binding_status": binding_status,
        }
        dialogue_bindings.append(dialogue_row)

        group_key = (speaker_id, subject, clean_owner)
        speaker_bindings.setdefault(group_key, {
            "local_speaker": f"S{speaker_id}" if speaker_id is not None else None,
            "subject": f"Subject {subject}" if subject is not None else None,
            "compiled_subject": f"Subject {compiled_subject}" if compiled_subject is not None else None,
            "character_id": (visual or {}).get("character_id"),
            "name": display_name,
            "reference_id": (visual or {}).get("reference_id"),
            "audio_label": voice_label,
            "voice_reference_id": (voice_reference or {}).get("reference_id"),
            "binding_status": binding_status,
        })
        previous_end = end
        source_previous_end = source_end

    return list(speaker_bindings.values()), dialogue_bindings


def _voice_scope(
    diagnostics: dict[str, Any],
    aligned: list[dict[str, Any]],
    selected_indexes: list[int | None],
) -> dict[str, Any]:
    kept = {index for index in selected_indexes if index is not None}
    voice_rows: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(aligned):
        if item.get("type") == "audio" and item.get("audio_intent", "voice") == "voice":
            voice_rows.append((index, item))
    result = {
        "applied": True,
        "speaking_subjects": [int(value) for value in diagnostics.get("speaking_subjects", [])],
        "voice_total": int(diagnostics.get("voice_total", len(voice_rows))),
        "voice_kept": int(diagnostics.get("voice_kept", 0)),
        "voice_omitted": int(diagnostics.get("voice_omitted", 0)),
        "max_audio_references": int(diagnostics.get("max_audio_references", 2)),
        "non_voice_audio_references": int(diagnostics.get("non_voice_audio_references", 0)),
        "kept_roles": [_public_text(value) for value in diagnostics.get("kept_roles", []) if _public_text(value)],
        "omitted_roles": [_public_text(value) for value in diagnostics.get("omitted_roles", []) if _public_text(value)],
        "kept_reference_ids": [],
        "omitted_reference_ids": [],
    }
    for index, item in voice_rows:
        identifier = _public_id(item)
        if identifier is None:
            continue
        target = "kept_reference_ids" if index in kept else "omitted_reference_ids"
        result[target].append(identifier)
    return result


def _audio_only_voice_scope(references: list[dict[str, Any]]) -> dict[str, Any]:
    voice_references = [
        item for item in references
        if item.get("type") == "audio" and item.get("audio_intent", "voice") == "voice"
    ]
    return {
        "applied": False,
        "reason": "audio_only",
        "voice_total": len(voice_references),
        "voice_kept": len(voice_references),
        "voice_omitted": 0,
        "speaking_subjects": [],
        "max_audio_references": 0,
        "non_voice_audio_references": sum(
            1 for item in references
            if item.get("type") == "audio" and item.get("audio_intent", "voice") != "voice"
        ),
        "kept_roles": [
            _public_text(item.get("character_name") or item.get("role"))
            for item in voice_references
            if _public_text(item.get("character_name") or item.get("role"))
        ],
        "omitted_roles": [],
        "kept_reference_ids": [
            identifier for item in voice_references
            if (identifier := _public_id(item)) is not None
        ],
        "omitted_reference_ids": [],
    }


def _finite_duration(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def compile_ref2va_prompt_handoff(
    prompt: str,
    references,
    *,
    duration_seconds: float | None = None,
    continuation: bool = False,
    audio_only: bool = False,
    window_index: int = 1,
) -> dict[str, Any]:
    """Compile the exact pure prompt/reference stages used at the H3 boundary.

    The helper intentionally delegates all matching, ordering, and prompt
    relationship semantics to the existing Ref2VA functions. It only adds a
    JSON-safe, path-free account of the resulting conditioning handoff.
    """

    source_prompt = str(prompt or "")
    aligned_prompt, aligned_references, alignment_remap = align_ref2va_voice_reference_order(
        source_prompt,
        references,
    )
    voice_scope: dict[str, Any]
    if audio_only:
        runtime_prompt = aligned_prompt
        runtime_references = aligned_references
        voice_scope = _audio_only_voice_scope(runtime_references)
    else:
        runtime_prompt, runtime_references, selection = select_ref2va_window_voice_references(
            aligned_prompt,
            aligned_references,
        )
        selected_indexes = _selected_reference_indexes(aligned_references, runtime_references)
        voice_scope = _voice_scope(selection, aligned_references, selected_indexes)
    scoped_prompt = runtime_prompt
    if not audio_only:
        runtime_prompt = ensure_ref2va_prompt_relationships(
            runtime_prompt,
            runtime_references,
            duration_seconds=duration_seconds,
        )

    if continuation:
        runtime_prompt = add_ref2va_continuation_context(runtime_prompt)

    bindings, visual_by_subject = _reference_bindings(runtime_references)
    speaker_bindings, dialogue_bindings = _speaker_diagnostics(
        runtime_prompt,
        runtime_references,
        bindings,
        visual_by_subject,
        ownership_prompt=scoped_prompt,
    )
    encoded_prompt = runtime_prompt.encode("utf-8")
    try:
        normalized_window_index = max(1, int(window_index))
    except (TypeError, ValueError, OverflowError):
        normalized_window_index = 1
    trace: dict[str, Any] = {
        "schema_version": 1,
        "stage": "prepared",
        "window_index": normalized_window_index,
        "compiled_prompt": runtime_prompt,
        "prompt_sha256": hashlib.sha256(encoded_prompt).hexdigest(),
        "continuation": bool(continuation),
        "reference_bindings": bindings,
        "speaker_bindings": speaker_bindings,
        "dialogue_bindings": dialogue_bindings,
        "voice_scope": voice_scope,
        "alignment_remap": _json_safe_alignment(alignment_remap),
        "conditioning_note": (
            "This trace describes diagnostic reference conditioning; it is not a guarantee of generated voice quality."
        ),
    }
    normalized_duration = _finite_duration(duration_seconds)
    if normalized_duration is not None:
        trace["duration_seconds"] = normalized_duration
    return {
        "scoped_prompt": scoped_prompt,
        "compiled_prompt": runtime_prompt,
        "runtime_references": runtime_references,
        "alignment_remap": _json_safe_alignment(alignment_remap),
        "voice_scope": voice_scope,
        "trace": trace,
    }


def refresh_ref2va_handoff_traces(
    traces: list[dict[str, Any]],
    prompts: list[str],
    references,
) -> list[dict[str, Any]]:
    """Recompile prepared traces against the final, normalized manifest.

    Native prepare can expand typed video soundtracks into standalone audio
    rows. Call this after that normalization so MCP diagnostics describe the
    exact manifest the render path will receive, while preserving the server's
    previously computed timeline geometry.
    """

    if not isinstance(traces, list) or not isinstance(prompts, list) or len(traces) != len(prompts):
        return copy.deepcopy(traces) if isinstance(traces, list) else []
    refreshed: list[dict[str, Any]] = []
    geometry_fields = {
        "timeline_start_frame",
        "timeline_end_frame",
        "timeline_contribution_frames",
        "timeline_contribution_duration_seconds",
        "rendered_pass_frames",
        "rendered_pass_duration_seconds",
        "duration_basis",
    }
    for trace, prompt in zip(traces, prompts):
        if not isinstance(trace, dict) or not isinstance(prompt, str):
            refreshed.append(copy.deepcopy(trace))
            continue
        if trace.get("stage") != "prepared":
            refreshed.append(copy.deepcopy(trace))
            continue
        voice_scope = trace.get("voice_scope") if isinstance(trace.get("voice_scope"), dict) else {}
        try:
            handoff = compile_ref2va_prompt_handoff(
                prompt,
                references,
                duration_seconds=trace.get("duration_seconds"),
                continuation=bool(trace.get("continuation")),
                audio_only=voice_scope.get("reason") == "audio_only",
                window_index=trace.get("window_index", 1),
            )
            current = handoff["trace"]
        except Exception as error:
            current = copy.deepcopy(trace)
            current.update({
                "stage": "diagnostic_unavailable",
                "compiled_prompt": None,
                "prompt_sha256": None,
                "authored_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "reference_bindings": [],
                "speaker_bindings": [],
                "dialogue_bindings": [],
                "voice_scope": {
                    "applied": False,
                    "reason": "diagnostic_unavailable",
                    "voice_total": 0,
                    "voice_kept": 0,
                    "voice_omitted": 0,
                    "kept_roles": [],
                    "omitted_roles": [],
                },
                "alignment_remap": {},
                "diagnostic_error": type(error).__name__,
                "conditioning_note": (
                    "No compiled handoff trace is available; this diagnostic did not alter the authored prompt."
                ),
            })
            refreshed.append(current)
            continue
        current.update({
            key: copy.deepcopy(value)
            for key, value in trace.items()
            if key in geometry_fields
        })
        refreshed.append(current)
    return refreshed
