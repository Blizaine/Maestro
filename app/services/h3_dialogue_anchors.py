"""Resolve source references to exact dialogue lines and verify their timing.

This module deliberately accepts only a small, explicit line-anchor grammar.
It does not infer that a line happened from a camera action mentioning speech:
the caller must pass the accepted native windows, and an exact immutable line
must be present in a shot's structured ``dialogue`` list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import unicodedata
from typing import Any


_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
}
_ANCHOR_HEAD_RE = re.compile(
    r"(?P<relation>only\s+after|after|before)\s+"
    r"(?:(?P<pronoun>his|her|their)\s+|"
    r"(?P<speaker>[\w][\w .'-]*?)\s*['’]s\s+|"
    r"(?P<article>the)\s+)?"
    r"(?P<ordinal>first|second|third|last|final|"
    r"1st|2nd|3rd|\d+(?:st|nd|rd|th)?)\s+line\b",
    re.IGNORECASE,
)
_NEGATED_RELATION_RE = re.compile(
    r"\b(?:not|never|without|fails?\s+to|tries?\s+to|attempts?\s+to|"
    r"would|could|may|might|unless)\s*$",
    re.IGNORECASE,
)
_NON_NAME_CAPITALIZED = {
    "a", "after", "an", "at", "before", "but", "each", "every", "he",
    "her", "his", "if", "in", "it", "keep", "only", "once", "one",
    "she", "the", "their", "then", "they", "this", "those", "to", "use",
    "we", "when", "while", "you",
}


def resolve_h3_dialogue_timing_anchors(
    source_events: Sequence[Mapping[str, Any]],
    locked_dialogue: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Resolve explicit source line ordinals against immutable dialogue.

    ``locked_dialogue`` must be the unfragmented, user-authored catalog (for
    example, ``extract_locked_dialogue(prompt)``), not generated dialogue or a
    model-authored paraphrase. A pronoun-scoped ordinal is resolved only when
    exactly one catalog speaker is named in that immutable source event.
    Ambiguous references are returned as unresolved instead of guessing.
    """

    catalog, catalog_errors = _normalize_locked_dialogue(locked_dialogue)
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = [
        {"reason": error, "source_phrase": "", "event_id": ""}
        for error in catalog_errors
    ]

    for event in source_events:
        if not isinstance(event, Mapping):
            continue
        event_id = _clean(event.get("event_id")).upper()
        text = _clean(event.get("text"))
        if not event_id or not text:
            continue
        for anchor_index, match in enumerate(_ANCHOR_HEAD_RE.finditer(text), start=1):
            phrase = match.group(0)
            prefix = text[max(0, match.start() - 48):match.start()]
            if _NEGATED_RELATION_RE.search(prefix):
                unresolved.append({
                    "anchor_id": f"{event_id}:L{anchor_index}",
                    "event_id": event_id,
                    "source_phrase": phrase,
                    "source_text": text,
                    "reason": "the line relation is negated, hypothetical, or attempted",
                })
                continue

            scope_speaker: str | None = None
            pronoun = _clean(match.group("pronoun")).casefold()
            explicit_speaker = _clean(match.group("speaker"))
            if explicit_speaker:
                match_speakers = list(dict.fromkeys(
                    item["speaker"] for item in catalog
                    if _same_name(item["speaker"], explicit_speaker)
                ))
                if len(match_speakers) != 1:
                    unresolved.append({
                        "anchor_id": f"{event_id}:L{anchor_index}",
                        "event_id": event_id,
                        "source_phrase": phrase,
                        "source_text": text,
                        "reason": "the named speaker is absent or ambiguous in immutable dialogue",
                    })
                    continue
                scope_speaker = match_speakers[0]
            elif pronoun:
                mentioned = _mentioned_actor_names(text, catalog)
                if len(mentioned) != 1 or not any(
                    _same_name(mentioned[0], item["speaker"]) for item in catalog
                ):
                    unresolved.append({
                        "anchor_id": f"{event_id}:L{anchor_index}",
                        "event_id": event_id,
                        "source_phrase": phrase,
                        "source_text": text,
                        "reason": (
                            f"{pronoun} does not have one uniquely named dialogue speaker "
                            "in this source event"
                        ),
                    })
                    continue
                scope_speaker = mentioned[0]
            elif not match.group("article") and match.group("ordinal").casefold() not in {"last", "final"}:
                unresolved.append({
                    "anchor_id": f"{event_id}:L{anchor_index}",
                    "event_id": event_id,
                    "source_phrase": phrase,
                    "source_text": text,
                    "reason": "an unpossessed ordinal line reference has no explicit global scope",
                })
                continue

            ordinal_text = match.group("ordinal").casefold()
            if scope_speaker:
                candidates = [
                    item for item in catalog
                    if _same_name(item["speaker"], scope_speaker)
                ]
                ordinal_scope = "speaker"
            else:
                candidates = list(catalog)
                ordinal_scope = "global"

            if ordinal_text in {"last", "final"}:
                ordinal_index = len(candidates) - 1
                ordinal_label = ordinal_text
            else:
                ordinal_index = _ORDINALS.get(ordinal_text)
                if ordinal_index is None:
                    ordinal_index = int(re.sub(r"\D", "", ordinal_text) or "0")
                ordinal_index -= 1
                ordinal_label = ordinal_text

            if ordinal_index < 0 or ordinal_index >= len(candidates):
                unresolved.append({
                    "anchor_id": f"{event_id}:L{anchor_index}",
                    "event_id": event_id,
                    "source_phrase": phrase,
                    "source_text": text,
                    "reason": "the ordinal does not resolve to an immutable dialogue line",
                })
                continue

            line = candidates[ordinal_index]
            relation = "before" if match.group("relation").casefold() == "before" else "after"
            resolved.append({
                "anchor_id": f"{event_id}:L{anchor_index}",
                "event_id": event_id,
                "relation": relation,
                "relation_phrase": match.group("relation"),
                "ordinal": ordinal_label,
                "ordinal_scope": ordinal_scope,
                "speaker": line["speaker"],
                "dialogue_id": line["dialogue_id"],
                "dialogue_text": line["text"],
                "source_phrase": phrase,
                "source_text": text,
            })

    return {"requirements": resolved, "unresolved": unresolved}


def validate_h3_dialogue_timing_anchors(
    source_events: Sequence[Mapping[str, Any]],
    locked_dialogue: Sequence[Mapping[str, Any]],
    accepted_segments: Sequence[Mapping[str, Any]],
    *,
    source_event_positions: Mapping[str, Sequence[Mapping[str, Any]]],
    render_dialogue_catalog: Sequence[Mapping[str, Any]] = (),
) -> dict[str, list[dict[str, Any]]]:
    """Verify line anchors against accepted exact dialogue and shot ordering.

    Accepted segment records use the native shape: ``{"index": 2,
    "shots": [{"start_seconds": ..., "end_seconds": ..., "dialogue":
    [{"dialogue_id", "speaker", "text"}]}]}``. The caller supplies
    ``source_event_positions`` from its accepted shot-to-beat mapping, keyed by
    immutable E-id; each position is ``{"segment": 2, "shot": 1}``. Position
    numbers are one-based. Only delivered dialogue records count. Wording in
    the source, camera action, or planned beat is never delivery evidence.
    """

    resolution = resolve_h3_dialogue_timing_anchors(source_events, locked_dialogue)
    catalog, _catalog_errors = _normalize_locked_dialogue(locked_dialogue)
    delivered, delivery_errors, shot_lookup = _accepted_deliveries(
        accepted_segments, catalog, render_dialogue_catalog=render_dialogue_catalog,
    )
    unresolved = list(resolution["unresolved"])
    violations: list[dict[str, Any]] = []
    unresolved_delivery_ids: set[str] = set()
    invalid_delivery_ids: set[str] = set()
    anchored_dialogue_ids = {
        item["dialogue_id"] for item in resolution["requirements"]
    }
    for error in delivery_errors:
        item = dict(error)
        dialogue_id = _clean(item.get("dialogue_id")).upper()
        if dialogue_id not in anchored_dialogue_ids:
            continue
        if item.pop("unresolved", False):
            unresolved_delivery_ids.add(dialogue_id)
            unresolved.append(item)
        else:
            invalid_delivery_ids.add(dialogue_id)
            violations.append(item)
    verified: list[dict[str, Any]] = []

    for requirement in resolution["requirements"]:
        dialogue_id = requirement["dialogue_id"]
        line_positions = delivered.get(dialogue_id, [])
        if not line_positions:
            if dialogue_id in unresolved_delivery_ids or dialogue_id in invalid_delivery_ids:
                continue
            violations.append({
                **requirement,
                "reason": "the exact anchored dialogue ID, speaker, and text are absent from accepted shots",
            })
            continue
        if len(line_positions) != 1:
            violations.append({
                **requirement,
                "reason": "the exact anchored dialogue line appears more than once in accepted shots",
            })
            continue

        action_positions, position_error = _event_positions_for(
            requirement["event_id"], source_event_positions, shot_lookup,
        )
        if position_error:
            unresolved.append({
                **requirement,
                "reason": position_error,
            })
            continue
        if not action_positions:
            unresolved.append({
                **requirement,
                "reason": "no accepted shot position is mapped to the anchored source event",
            })
            continue

        line_position = line_positions[0]
        first_action = min(action_positions, key=_start_key)
        last_action = max(action_positions, key=_end_key)
        if any(
            int(position["segment"]) == int(line_position["segment"])
            and int(position["shot"]) == int(line_position["shot"])
            for position in action_positions
        ):
            unresolved.append({
                **requirement,
                "line_position": _public_position(line_position),
                "action_start": _public_position(first_action),
                "action_end": _public_position(last_action),
                "reason": (
                    "the exact line and anchored action share a shot, so shot-level "
                    "timing cannot establish their order"
                ),
            })
            continue
        if requirement["relation"] == "after":
            preserved = _end_key(line_position) <= _start_key(first_action)
            comparison = "the action starts after the exact line ends"
        else:
            preserved = _end_key(last_action) <= _start_key(line_position)
            comparison = "the action ends before the exact line starts"

        if not preserved:
            violations.append({
                **requirement,
                "line_position": _public_position(line_position),
                "action_start": _public_position(first_action),
                "action_end": _public_position(last_action),
                "reason": (
                    f"the accepted shot order does not prove that {comparison}"
                ),
            })
            continue
        verified.append({
            **requirement,
            "line_position": _public_position(line_position),
            "action_start": _public_position(first_action),
            "action_end": _public_position(last_action),
            "evidence": "exact immutable dialogue was delivered in accepted shot order",
        })

    return {
        "requirements": resolution["requirements"],
        "verified": verified,
        "unresolved": unresolved,
        "violations": violations,
    }


def _normalize_locked_dialogue(
    dialogue: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[str]]:
    normalized: list[dict[str, str]] = []
    errors: list[str] = []
    seen: set[str] = set()
    indexed = list(enumerate(dialogue))
    if all(isinstance(item, Mapping) and _number(item.get("source_offset")) is not None
           for _index, item in indexed):
        indexed.sort(key=lambda pair: _number(pair[1].get("source_offset")) or 0.0)
    for _index, item in indexed:
        if not isinstance(item, Mapping):
            errors.append("immutable dialogue catalog contains a non-object item")
            continue
        dialogue_id = _clean(item.get("dialogue_id")).upper()
        speaker = _clean(item.get("speaker"))
        text = _clean(item.get("text"))
        if not dialogue_id or not speaker or not text:
            errors.append("immutable dialogue item lacks an ID, speaker, or exact text")
            continue
        if dialogue_id in seen:
            errors.append(f"immutable dialogue ID {dialogue_id} is duplicated")
            continue
        seen.add(dialogue_id)
        normalized.append({"dialogue_id": dialogue_id, "speaker": speaker, "text": text})
    return normalized, errors


def _mentioned_actor_names(text: str, catalog: Sequence[Mapping[str, str]]) -> list[str]:
    """Find named actors conservatively, including names absent from the catalog.

    An unlisted second person still makes a possessive pronoun ambiguous; it
    must not disappear merely because that person has no line in the catalog.
    """

    found: list[str] = []
    for item in catalog:
        speaker = item["speaker"]
        if _contains_name(text, speaker) and not any(_same_name(speaker, name) for name in found):
            found.append(speaker)
    for match in re.finditer(r"\b[A-Z][a-z]+(?:[’'-][A-Z]?[a-z]+)*(?:\s+[A-Z][a-z]+(?:[’'-][A-Z]?[a-z]+)*)*\b", text):
        name = _clean(match.group(0))
        if name.casefold() in _NON_NAME_CAPITALIZED:
            continue
        if not any(_same_name(name, existing) for existing in found):
            found.append(name)
    return found


def _contains_name(text: str, name: str) -> bool:
    return bool(re.search(
        rf"(?<![\w]){re.escape(name)}(?![\w])",
        text,
        flags=re.IGNORECASE,
    ))


def _same_name(left: str, right: str) -> bool:
    return _normalize_space(left).casefold() == _normalize_space(right).casefold()


def _accepted_deliveries(
    accepted_segments: Sequence[Mapping[str, Any]],
    catalog: Sequence[Mapping[str, str]],
    *,
    render_dialogue_catalog: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    catalog_by_id = {item["dialogue_id"]: item for item in catalog}
    fragment_by_id: dict[str, dict[str, Any]] = {}
    for item in render_dialogue_catalog:
        if not isinstance(item, Mapping):
            continue
        fragment_id = _clean(item.get("dialogue_id")).upper()
        parent_id = _clean(item.get("source_dialogue_id")).upper()
        if fragment_id and parent_id in catalog_by_id:
            fragment_by_id[fragment_id] = dict(item)
    accepted: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    shot_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    fragments: dict[str, list[dict[str, Any]]] = {}
    ordered_dialogue_occurrences: list[tuple[tuple[int, int], str, int | None]] = []
    for segment_offset, segment in enumerate(accepted_segments, start=1):
        if not isinstance(segment, Mapping):
            continue
        segment_number = _positive_int(segment.get("index", segment.get("segment"))) or segment_offset
        shots = segment.get("shots")
        if not isinstance(shots, Sequence) or isinstance(shots, (str, bytes)):
            continue
        for shot_offset, shot in enumerate(shots, start=1):
            if not isinstance(shot, Mapping):
                continue
            shot_number = _positive_int(shot.get("shot")) or shot_offset
            position = {
                "segment": segment_number,
                "shot": shot_number,
                "start_seconds": _number(shot.get("start_seconds")),
                "end_seconds": _number(shot.get("end_seconds")),
            }
            shot_lookup[(segment_number, shot_number)] = position
            lines = shot.get("dialogue")
            if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)):
                continue
            for line in lines:
                if not isinstance(line, Mapping):
                    continue
                dialogue_id = _clean(line.get("dialogue_id")).upper()
                fragment_meta = fragment_by_id.get(dialogue_id, {})
                source_id = (
                    _clean(line.get("source_dialogue_id")).upper()
                    or _clean(fragment_meta.get("source_dialogue_id")).upper()
                )
                fragment_match = re.fullmatch(r"(D\d+)F(\d+)", dialogue_id)
                parent_id = source_id or (fragment_match.group(1) if fragment_match else "")
                if dialogue_id not in catalog_by_id and parent_id not in catalog_by_id:
                    # Unknown dialogue IDs cannot prove an immutable line
                    # anchor; ordinary dialogue validation reports them.
                    continue
                expected_id = dialogue_id if dialogue_id in catalog_by_id else parent_id
                expected = catalog_by_id[expected_id]
                speaker = _clean(line.get("speaker"))
                text = _clean(line.get("text"))
                dialogue_position = (segment_number, shot_number)
                ordered_dialogue_occurrences.append((
                    dialogue_position,
                    expected_id,
                    _positive_int(line.get("fragment_index"))
                    or (_positive_int(fragment_match.group(2)) if fragment_match else None),
                ))
                is_fragment = expected_id != dialogue_id or bool(line.get("fragment_count"))
                if is_fragment:
                    if not _same_name(speaker, expected["speaker"]):
                        errors.append({
                            "dialogue_id": expected_id,
                            "segment": segment_number,
                            "shot": shot_number,
                            "reason": "accepted dialogue fragment has the wrong immutable speaker",
                        })
                        continue
                    fragment_index = (
                        _positive_int(line.get("fragment_index"))
                        or _positive_int(fragment_meta.get("fragment_index"))
                    )
                    if fragment_index is None and fragment_match:
                        fragment_index = _positive_int(fragment_match.group(2))
                    fragment_count = (
                        _positive_int(line.get("fragment_count"))
                        or _positive_int(fragment_meta.get("fragment_count"))
                    )
                    if fragment_index is None or fragment_count is None:
                        errors.append({
                            "dialogue_id": expected_id,
                            "segment": segment_number,
                            "shot": shot_number,
                            "reason": "accepted dialogue fragment lacks complete source/index/count metadata",
                            "unresolved": True,
                        })
                        continue
                    expected_fragment_text = _clean(fragment_meta.get("text"))
                    if expected_fragment_text and not _same_text(text, expected_fragment_text):
                        errors.append({
                            "dialogue_id": expected_id,
                            "segment": segment_number,
                            "shot": shot_number,
                            "reason": "accepted dialogue fragment differs from its render-catalog text",
                        })
                        continue
                    fragments.setdefault(expected_id, []).append({
                        "index": fragment_index,
                        "count": fragment_count,
                        "position": position,
                        "speaker": speaker,
                        "text": text,
                    })
                    continue
                if not _same_name(speaker, expected["speaker"]) or not _same_text(text, expected["text"]):
                    errors.append({
                        "dialogue_id": expected_id,
                        "segment": segment_number,
                        "shot": shot_number,
                        "reason": "accepted dialogue ID has the wrong immutable speaker or text",
                    })
                    continue
                accepted.setdefault(expected_id, []).append(position)

    for dialogue_id, parts in fragments.items():
        counts = {part["count"] for part in parts}
        indexes = [part["index"] for part in parts]
        if len(counts) != 1:
            errors.append({
                "dialogue_id": dialogue_id,
                "reason": "accepted dialogue fragments disagree about the immutable line length",
            })
            continue
        fragment_count = next(iter(counts))
        if sorted(indexes) != list(range(1, fragment_count + 1)):
            errors.append({
                "dialogue_id": dialogue_id,
                "reason": "accepted dialogue fragments are missing, duplicated, or out of order",
            })
            continue
        by_index = sorted(parts, key=lambda part: part["index"])
        expected = catalog_by_id[dialogue_id]
        reconstructed = " ".join(part["text"] for part in by_index)
        if not all(_same_name(part["speaker"], expected["speaker"]) for part in by_index):
            errors.append({
                "dialogue_id": dialogue_id,
                "reason": "accepted dialogue fragments change the immutable speaker",
            })
            continue
        if not _same_text(reconstructed, expected["text"]):
            errors.append({
                "dialogue_id": dialogue_id,
                "reason": "accepted dialogue fragments do not reconstruct the exact immutable text",
            })
            continue
        # Fragments of one exact line may cross a window boundary, but another
        # spoken line cannot be interleaved before this line has completed.
        ordered_parts = sorted(by_index, key=lambda part: _start_key(part["position"]))
        if [part["index"] for part in ordered_parts] != list(range(1, fragment_count + 1)):
            errors.append({
                "dialogue_id": dialogue_id,
                "reason": "accepted dialogue fragments are delivered out of temporal order",
            })
            continue
        first_position = ordered_parts[0]["position"]
        last_position = ordered_parts[-1]["position"]
        first_key = (first_position["segment"], first_position["shot"])
        last_key = (last_position["segment"], last_position["shot"])
        intervening = [
            item for item in ordered_dialogue_occurrences
            if first_key <= item[0] <= last_key
            and item[1] != dialogue_id
        ]
        if intervening:
            errors.append({
                "dialogue_id": dialogue_id,
                "reason": "another dialogue line interrupts the immutable line fragments",
            })
            continue
        combined_position = {
            "segment": first_position["segment"],
            "shot": first_position["shot"],
            "start_seconds": first_position["start_seconds"],
            "end_segment": last_position["segment"],
            "end_shot": last_position["shot"],
            "end_seconds": last_position["end_seconds"],
        }
        accepted.setdefault(dialogue_id, []).append(combined_position)
    return accepted, errors, shot_lookup


def _event_positions_for(
    event_id: str,
    event_positions: Mapping[str, Sequence[Mapping[str, Any]]],
    shot_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    raw_positions = next((
        values for key, values in event_positions.items()
        if _clean(key).upper() == event_id.upper()
    ), None)
    if raw_positions is None:
        return [], None
    valid: list[dict[str, Any]] = []
    for raw in raw_positions:
        if not isinstance(raw, Mapping):
            continue
        segment = _positive_int(raw.get("segment"))
        shot = _positive_int(raw.get("shot"))
        if segment is None or shot is None:
            continue
        accepted_position = shot_lookup.get((segment, shot))
        if accepted_position is None:
            return [], "source event position points outside the accepted shot list"
        valid.append(dict(accepted_position))
    return valid, None


def _start_key(position: Mapping[str, Any]) -> tuple[int, float, int]:
    shot = int(position["shot"])
    segment = int(position["segment"])
    start = position.get("start_seconds")
    return (segment, float(start) if start is not None else float(shot), shot)


def _end_key(position: Mapping[str, Any]) -> tuple[int, float, int]:
    shot = int(position.get("end_shot") or position["shot"])
    segment = int(position.get("end_segment") or position["segment"])
    end = position.get("end_seconds")
    return (segment, float(end) if end is not None else float(shot), shot)


def _public_position(position: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in position.items()
        if key in {
            "segment", "shot", "start_seconds", "end_seconds",
            "end_segment", "end_shot",
        }
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_space(value: str) -> str:
    return _clean(value)


def _same_text(left: str, right: str) -> bool:
    return unicodedata.normalize("NFC", _normalize_space(left)) == unicodedata.normalize(
        "NFC", _normalize_space(right),
    )
