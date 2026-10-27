"""Compact, whole-song creative arcs for music-video planning.

The planner can use rows as timeline groups (up to 48), then expand those
groups into detailed clips. Source-event IDs remain owned by the narrative
contract and are assigned exactly once, in source order.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any, Optional


MAX_MUSIC_ARC_ROWS = 48
MAX_MUSIC_ARC_CLIPS = 2048
MAX_MUSIC_ARC_EVENTS = 16
_MAX_BEAT_CHARS = 360
_MAX_FIELD_CHARS = 240
_MAX_REPRISE_REASON_CHARS = 200
_MAX_ISSUES = 12
_MAX_REPETITION_FINDINGS = 12
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_COMMON_WORDS = {
    "a", "about", "after", "again", "along", "and", "as", "at", "away",
    "before", "behind", "beneath", "beside", "between", "by", "down",
    "for", "from", "in", "into", "is", "it", "its", "near", "of", "off",
    "on", "onto", "or", "over", "past", "the", "then", "through", "to",
    "toward", "under", "up", "upon", "while", "with", "within",
}
_GENERIC_WORDS = {
    "beautiful", "cinematic", "compelling", "dynamic", "engaging", "image",
    "imagery", "match", "matches", "matching", "mood", "music", "scene",
    "shot", "shots", "song", "support", "supports", "supporting", "video",
    "visual", "visuals",
}
_GENERIC_FILLER_RE = re.compile(
    r"\b(?:various|different|generic|random)\s+(?:cinematic\s+)?"
    r"(?:shots?|scenes?|visuals?|imagery|montages?)\b|"
    r"\b(?:visuals?|imagery|shots?|scenes?|montages?)\s+"
    r"(?:that\s+)?(?:match|matches|matching|reflect|reflects|support|supports|"
    r"complement|complements)\s+(?:the\s+)?(?:song|music|lyrics|mood)\b|"
    r"\b(?:cinematic|beautiful|dramatic|engaging|dynamic|compelling)\s+"
    r"(?:visuals?|imagery|shots?|scenes?|montages?)\b",
    re.IGNORECASE,
)

# These details can vary while the underlying visual beat repeats. Keeping
# this list modest makes the comparison conservative across music-video types.
_SETTING_CAMERA_WORDS = {
    "aerial", "amber", "angle", "angles", "arena", "beach", "blue", "bright",
    "camera", "close", "closeup", "coast", "coastal", "dawn", "day", "desert",
    "dusk", "environment", "field", "film", "forest", "frame", "frames", "gold",
    "golden", "hall", "highway", "indoor", "landscape", "lens", "lighting", "location",
    "mountain", "night", "ocean", "outdoor", "overhead", "palette", "pan", "panoramic",
    "plaza", "red", "road", "rooftop", "room", "scene", "setting", "shore", "shot",
    "side", "sky", "stage", "street", "studio", "sunset", "tilt", "track", "tracking",
    "train", "tunnel", "view", "wide", "window", "yellow",
}


def creative_plan_schema() -> dict[str, Any]:
    """Return the strict JSON schema for a compact whole-song arc."""

    return {
        "type": "array",
        "minItems": 0,
        "maxItems": MAX_MUSIC_ARC_ROWS,
        "items": {
            "type": "object",
            "properties": {
                "clip_index": {"type": "integer", "minimum": 0, "maximum": MAX_MUSIC_ARC_ROWS - 1},
                "visual_beat": {"type": "string", "minLength": 8, "maxLength": _MAX_BEAT_CHARS},
                "purpose": {"type": "string", "minLength": 2, "maxLength": _MAX_FIELD_CHARS},
                "environment": {"type": "string", "minLength": 2, "maxLength": _MAX_FIELD_CHARS},
                "camera": {"type": "string", "minLength": 2, "maxLength": _MAX_FIELD_CHARS},
                "source_event_ids": {
                    "type": "array",
                    "maxItems": MAX_MUSIC_ARC_EVENTS,
                    "items": {"type": "string", "minLength": 1, "maxLength": 80},
                    "uniqueItems": True,
                },
                "intentional_reprise": {"type": "boolean"},
                "reprise_reason": {"type": "string", "maxLength": _MAX_REPRISE_REASON_CHARS},
            },
            "required": [
                "clip_index", "visual_beat", "purpose", "environment", "camera",
                "source_event_ids", "intentional_reprise", "reprise_reason",
            ],
            "additionalProperties": False,
        },
    }


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


_ARC_ROW_CONTAINERS = (
    "beats", "rows", "arc", "music_arc", "outline", "timeline_groups", "groups",
)
_ARC_VISUAL_BEAT_ALIASES = ("beat", "prompt")
_ARC_ZERO_BASED_INDEX_FIELDS = ("clip_index", "timeline_index", "group_index", "index")
_ARC_ONE_BASED_INDEX_FIELDS = ("clip_number", "timeline_number", "group_number", "number")
_ARC_ZERO_BASED_RANGE_FIELDS = ("clip_index_range", "timeline_index_range", "group_index_range", "index_range")
_ARC_ONE_BASED_RANGE_FIELDS = ("clip_number_range", "timeline_number_range", "group_number_range", "number_range")
_ARC_RANGE_FIELDS = frozenset((
    *_ARC_ZERO_BASED_RANGE_FIELDS, *_ARC_ONE_BASED_RANGE_FIELDS,
    "clip_range", "timeline_range", "group_range", "range",
))
_ARC_INDEX_FIELDS = frozenset((*_ARC_ZERO_BASED_INDEX_FIELDS, *_ARC_ONE_BASED_INDEX_FIELDS))
_ARC_RANGE_KEY_RE = re.compile(r"^(\d+)\s*[-–]\s*(\d+)$")


def _arc_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"music arc {label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        return int(value)
    raise ValueError(f"music arc {label} must be an integer")


def _arc_source_rows(value: Any) -> tuple[list[Any], Optional[list[int]], Optional[list[tuple[int, int]]]]:
    """Unwrap known row containers and complete numeric/range maps."""

    if isinstance(value, Mapping):
        containers = [
            key for key in _ARC_ROW_CONTAINERS
            if key in value and isinstance(value[key], (Mapping, Sequence))
            and not isinstance(value[key], (str, bytes, bytearray))
        ]
        if containers:
            if len(containers) != 1 or set(value) != {containers[0]}:
                raise ValueError("music arc wrapper has unrelated fields")
            return _arc_source_rows(value[containers[0]])

        numeric: list[tuple[Mapping[str, Any], int]] = []
        ranges: list[tuple[Mapping[str, Any], tuple[int, int]]] = []
        for key, row in value.items():
            if not isinstance(key, str) or not isinstance(row, Mapping):
                raise ValueError("music arc object must contain a recognized row container or numeric group keys")
            if re.fullmatch(r"\s*\d+\s*", key):
                numeric.append((row, _arc_integer(key, "group key")))
            elif (match := _ARC_RANGE_KEY_RE.fullmatch(key)):
                ranges.append((row, (int(match.group(1)), int(match.group(2)))))
            else:
                raise ValueError(f"music arc object has unrelated key {key[:60]!r}")
        if numeric and ranges:
            raise ValueError("music arc cannot mix numeric keys and range keys")
        if numeric:
            return [row for row, _index in numeric], [index for _row, index in numeric], None
        if ranges:
            return [row for row, _bounds in ranges], None, [bounds for _row, bounds in ranges]
        raise ValueError("music arc object has no timeline rows")

    if not _is_sequence(value):
        raise ValueError("music arc rows must be a list or recognized object container")
    rows = list(value)
    # BasePlanner wraps a root JSON object in a one-item list. Unwrap only
    # recognized containers/maps, rather than treating an arbitrary row as one.
    if len(rows) == 1 and isinstance(rows[0], Mapping):
        nested = rows[0]
        if any(key in nested for key in _ARC_ROW_CONTAINERS):
            return _arc_source_rows(nested)
        if nested and all(
            isinstance(key, str)
            and (re.fullmatch(r"\s*\d+\s*", key) or _ARC_RANGE_KEY_RE.fullmatch(key))
            and isinstance(row, Mapping)
            for key, row in nested.items()
        ):
            return _arc_source_rows(nested)
    return rows, None, None


def _arc_explicit_index(row: Mapping[str, Any]) -> Optional[int]:
    values = [_arc_integer(row[field], field) for field in _ARC_ZERO_BASED_INDEX_FIELDS if field in row]
    for field in _ARC_ONE_BASED_INDEX_FIELDS:
        if field in row:
            number = _arc_integer(row[field], field)
            if number < 1:
                raise ValueError(f"music arc {field} must start at 1")
            values.append(number - 1)
    if len(set(values)) > 1:
        raise ValueError("music arc row has conflicting timeline index fields")
    return values[0] if values else None


def _normalize_arc_visual_beat_aliases(row: dict[str, Any], row_number: int) -> None:
    """Map row-local beat aliases to visual_beat without losing conflicts.

    When multiple spellings are present, text is equal after collapsing
    whitespace with ``str.split``/``str.join``. Keep the canonical value when
    supplied; otherwise prefer ``beat`` to ``prompt``. The final text is still
    validated and whitespace-collapsed by ``_collapsed_text``.
    """

    fields = [
        field for field in ("visual_beat", *_ARC_VISUAL_BEAT_ALIASES)
        if field in row
    ]
    if not fields:
        return

    if len(fields) > 1:
        values = [row[field] for field in fields]
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"music arc row {row_number} visual_beat aliases must be text")
        collapsed = [" ".join(value.split()) for value in values]
        if len(set(collapsed)) > 1:
            raise ValueError(f"music arc row {row_number} has conflicting visual_beat alias values")

    canonical = next((field for field in ("visual_beat", *_ARC_VISUAL_BEAT_ALIASES) if field in row), None)
    if canonical != "visual_beat":
        row["visual_beat"] = row[canonical]
    for field in _ARC_VISUAL_BEAT_ALIASES:
        row.pop(field, None)


def _arc_singleton_range(value: Any, field: str) -> tuple[int, Optional[int]]:
    """Read an explicit range only when it names one authored timeline row."""

    if isinstance(value, Mapping):
        if set(value) == {"start_index", "end_index"}:
            start, end, base = _arc_integer(value["start_index"], "range start"), _arc_integer(value["end_index"], "range end"), 0
        elif set(value) == {"start_number", "end_number"}:
            start, end, base = _arc_integer(value["start_number"], "range start"), _arc_integer(value["end_number"], "range end"), 1
        elif set(value) == {"start", "end"}:
            start, end, base = _arc_integer(value["start"], "range start"), _arc_integer(value["end"], "range end"), None
        else:
            raise ValueError("music arc range object must contain only start and end")
    elif isinstance(value, str):
        match = _ARC_RANGE_KEY_RE.fullmatch(value)
        if not match:
            raise ValueError("music arc range must use an inclusive start-end form")
        start, end, base = int(match.group(1)), int(match.group(2)), None
    elif _is_sequence(value) and len(value) == 2:
        start, end, base = _arc_integer(value[0], "range start"), _arc_integer(value[1], "range end"), None
    else:
        raise ValueError("music arc range must contain exactly two integer endpoints")
    if end != start:
        raise ValueError("music arc ranges must identify one timeline group per authored row")
    if base is None:
        base = 0 if field in _ARC_ZERO_BASED_RANGE_FIELDS else 1 if field in _ARC_ONE_BASED_RANGE_FIELDS else None
    return start, base


def coerce_music_arc_rows(rows: Any, timeline_group_count: int) -> list[dict[str, Any]]:
    """Normalize explicit remote JSON layouts without inventing timeline rows."""

    if (
        not isinstance(timeline_group_count, int)
        or isinstance(timeline_group_count, bool)
        or timeline_group_count < 0
    ):
        raise ValueError("music arc timeline group count must be nonnegative")
    source_rows, keyed_indexes, keyed_ranges = _arc_source_rows(rows)
    if len(source_rows) != timeline_group_count:
        raise ValueError(
            f"music arc must cover exactly {timeline_group_count} timeline groups "
            f"(received {len(source_rows)})"
        )

    prepared: list[dict[str, Any]] = []
    row_indexes: list[Optional[int]] = []
    row_ranges: list[Optional[tuple[int, Optional[int]]]] = []
    for position, raw in enumerate(source_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"music arc row {position + 1} must be an object")
        row = dict(raw)
        index = _arc_explicit_index(row)
        range_fields = [field for field in _ARC_RANGE_FIELDS if field in row]
        if len(range_fields) > 1 or (range_fields and index is not None):
            raise ValueError(f"music arc row {position + 1} has conflicting index fields")
        row_range = _arc_singleton_range(row[range_fields[0]], range_fields[0]) if range_fields else None
        for field in (*_ARC_INDEX_FIELDS, *range_fields):
            row.pop(field, None)
        _normalize_arc_visual_beat_aliases(row, position + 1)
        prepared.append(row)
        row_indexes.append(index)
        row_ranges.append(row_range)

    if keyed_indexes is not None:
        if keyed_ranges is not None:
            raise ValueError("music arc cannot mix numeric keys and range keys")
        mapped = _resolve_music_arc_key_positions(keyed_indexes, timeline_group_count)
        if any(row_ranges):
            raise ValueError("music arc cannot combine group keys and row ranges")
        for row_index, key_index in zip(row_indexes, mapped):
            if row_index is not None and row_index != key_index:
                raise ValueError("music arc row conflicts with its numeric group key")
        positions = mapped
    elif keyed_ranges is not None:
        if any(index is not None for index in row_indexes) or any(row_ranges):
            raise ValueError("music arc cannot combine range keys with row indexes or ranges")
        if any(start != end for start, end in keyed_ranges):
            raise ValueError("music arc ranges must identify one timeline group per authored row")
        positions = _resolve_music_arc_key_positions([start for start, _end in keyed_ranges], timeline_group_count)
    elif any(row_ranges):
        if any(index is not None for index in row_indexes) or any(value is None for value in row_ranges):
            raise ValueError("music arc cannot mix indexed, ranged, and positional rows")
        positions = _resolve_music_arc_row_ranges(row_ranges, timeline_group_count)
    elif any(index is not None for index in row_indexes):
        if any(index is None for index in row_indexes):
            raise ValueError("music arc cannot mix indexed and unindexed rows")
        positions = [int(index) for index in row_indexes]
        _require_complete_music_arc_positions(positions, timeline_group_count)
    else:
        # Positional rows are safe only for a complete array whose ordering is
        # the fixed timeline ordering supplied in the prompt.
        positions = list(range(timeline_group_count))

    return sorted(
        [dict(row, clip_index=index) for row, index in zip(prepared, positions)],
        key=lambda row: row["clip_index"],
    )


def _require_complete_music_arc_positions(positions: list[int], count: int) -> None:
    if len(positions) != count or set(positions) != set(range(count)):
        raise ValueError("music arc indexes must cover the complete timeline exactly once")


def _resolve_music_arc_key_positions(raw_indexes: list[int], count: int) -> list[int]:
    candidates = [
        [index - base for index in raw_indexes]
        for base in (0, 1)
        if set(index - base for index in raw_indexes) == set(range(count))
        and len(set(raw_indexes)) == count
    ]
    if len(candidates) != 1:
        raise ValueError("music arc keys must cover the complete timeline with unambiguous numbering")
    return candidates[0]


def _resolve_music_arc_row_ranges(
    ranges: list[Optional[tuple[int, Optional[int]]]], count: int,
) -> list[int]:
    values = [value for value in ranges if value is not None]
    candidates: set[tuple[int, ...]] = set()
    for base in (0, 1):
        positions = [start - (hint if hint is not None else base) for start, hint in values]
        if set(positions) == set(range(count)) and len(set(positions)) == count:
            candidates.add(tuple(positions))
    if len(candidates) != 1:
        raise ValueError("music arc ranges must cover the complete timeline with unambiguous numbering")
    return list(next(iter(candidates)))


def _contract_event_ids(contract: Any) -> list[str]:
    if not isinstance(contract, Mapping) or not _is_sequence(contract.get("events")):
        raise ValueError("music arc contract must contain an events list")
    events = contract["events"]
    if len(events) > MAX_MUSIC_ARC_EVENTS:
        raise ValueError(f"music arc contract exceeds {MAX_MUSIC_ARC_EVENTS} source events")
    event_ids: list[str] = []
    seen: set[str] = set()
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"music arc contract event {index + 1} must be an object")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id != event_id.strip():
            raise ValueError(f"music arc contract event {index + 1} needs an immutable event_id")
        if len(event_id) > 80 or event_id in seen:
            raise ValueError(f"music arc contract event_id {event_id[:80]!r} is invalid or duplicated")
        event_ids.append(event_id)
        seen.add(event_id)
    return event_ids


def _collapsed_text(value: Any, field: str, limit: int, row_number: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"music arc row {row_number} {field} must be text")
    result = " ".join(value.split())
    if not result:
        raise ValueError(f"music arc row {row_number} {field} must not be empty")
    if len(result) > limit:
        raise ValueError(f"music arc row {row_number} {field} exceeds {limit} characters")
    return result


def _tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.casefold())


def _content_tokens(value: str) -> list[str]:
    return [token for token in _tokens(value) if token not in _COMMON_WORDS | _GENERIC_WORDS]


def _normalized_arc_text(value: str) -> str:
    """Normalize spelling variants while retaining visible treatment details."""
    aliases = {"singer": "performer", "vocalist": "performer", "frontperson": "performer"}
    actions = {"singing": "sing", "sings": "sing", "performed": "perform", "performing": "perform", "performs": "perform"}
    return " ".join(actions.get(aliases.get(token, token), aliases.get(token, token)) for token in _tokens(value))


def _very_close_arc_text(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    left_norm = _normalized_arc_text(left)
    right_norm = _normalized_arc_text(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    smaller = min(len(left_tokens), len(right_tokens))
    if smaller < 2:
        return False
    shared = len(left_tokens & right_tokens)
    return (
        shared / smaller >= 0.90
        and SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.90
    )


def music_arc_repetition_findings(rows: Any) -> list[dict[str, Any]]:
    """Return conservative, zero-based pairs of repeated beat and treatment.

    Similar performance roles or a continuing state do not establish a
    repetition. A finding requires a very close visual beat plus matching
    purpose, environment, and camera treatment. Findings are advisory records,
    separate from structural and source-assignment validation.
    """
    if not _is_sequence(rows) or len(rows) > MAX_MUSIC_ARC_ROWS:
        return []
    ordered: list[tuple[int, Mapping[str, Any]]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index = row.get("clip_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in seen:
            continue
        if not all(isinstance(row.get(field), str) and row[field].strip()
                   for field in ("visual_beat", "purpose", "environment", "camera")):
            continue
        seen.add(index)
        ordered.append((index, row))
    ordered.sort(key=lambda item: item[0])

    findings: list[dict[str, Any]] = []
    for current_index, current in ordered:
        current_reason = current.get("reprise_reason")
        if (
            current.get("intentional_reprise") is True
            and isinstance(current_reason, str)
            and _reason_is_meaningful(" ".join(current_reason.split()))
        ):
            continue
        match = None
        for prior_index, prior in reversed(ordered):
            if prior_index >= current_index:
                continue
            if not _very_close_arc_text(current["visual_beat"], prior["visual_beat"]):
                continue
            if not all(_very_close_arc_text(current[field], prior[field])
                       for field in ("purpose", "environment", "camera")):
                continue
            match = (prior_index, prior)
            break
        if match is None:
            continue
        prior_index, prior = match
        findings.append({
            "clip_index": current_index,
            "prior_clip_index": prior_index,
            "evidence": {
                "visual_beat": {
                    "current": current["visual_beat"],
                    "prior": prior["visual_beat"],
                },
                "purpose": {"current": current["purpose"], "prior": prior["purpose"]},
                "environment": {
                    "current": current["environment"], "prior": prior["environment"],
                },
                "camera": {"current": current["camera"], "prior": prior["camera"]},
                "current_intentional_reprise": current.get("intentional_reprise") is True,
                "current_reprise_reason": str(current.get("reprise_reason") or "")[:_MAX_REPRISE_REASON_CHARS],
            },
        })
        if len(findings) >= _MAX_REPETITION_FINDINGS:
            break
    return findings


def music_arc_reprise_reason_findings(rows: Any) -> list[dict[str, Any]]:
    """Return advisory records for reprise flags without a concise reason."""
    if not _is_sequence(rows) or len(rows) > MAX_MUSIC_ARC_ROWS:
        return []
    findings: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("intentional_reprise") is not True:
            continue
        index = row.get("clip_index")
        reason = row.get("reprise_reason")
        if (
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            or isinstance(reason, str) and _reason_is_meaningful(" ".join(reason.split()))
        ):
            continue
        findings.append({
            "clip_index": index,
            "prior_clip_index": None,
            "evidence": {
                "visual_beat": str(row.get("visual_beat") or "")[:_MAX_BEAT_CHARS],
                "reprise_reason": str(reason or "")[:_MAX_REPRISE_REASON_CHARS],
            },
        })
        if len(findings) >= _MAX_REPETITION_FINDINGS:
            break
    return findings


def _whole_brief_copy(beat: str, contract: Any) -> bool:
    if not isinstance(contract, Mapping) or not isinstance(contract.get("source"), str):
        return False
    source = " ".join(_tokens(contract["source"]))
    visual = " ".join(_tokens(beat))
    if not source or not visual:
        return False
    source_words = source.split()
    if len(source_words) < 8:
        return False
    ratio = SequenceMatcher(None, source, visual).ratio()
    if ratio >= 0.92:
        return True
    # Catch a copied brief with a short clip-specific prefix or suffix.
    source_set, visual_set = set(source_words), set(visual.split())
    return len(source_set) >= 8 and len(source_set & visual_set) / len(source_set) >= 0.94


def _reason_is_meaningful(reason: str) -> bool:
    tokens = _content_tokens(reason)
    return len(reason) >= 8 and len(tokens) >= 2


def _safe_issue(issues: list[str], value: str) -> None:
    if len(issues) < _MAX_ISSUES and value not in issues:
        issues.append(value[:360])


def music_arc_quality_issues(
    rows: Any,
    contract: Any,
    allow_early_events: bool = False,
) -> list[str]:
    """Return bounded structural and creative diagnostics for a compact arc.

    Early event clustering is repairable feedback. It is reported only when
    three or more source actions all land in roughly the opening fifth of a
    plan with at least twelve timeline groups.
    """

    issues: list[str] = []
    if not _is_sequence(rows):
        return ["music arc must be an array of timeline-group rows"]
    row_count = len(rows)
    if row_count > MAX_MUSIC_ARC_CLIPS:
        return [f"music arc exceeds the {MAX_MUSIC_ARC_CLIPS}-clip validation limit"]
    check_creative_quality = row_count <= MAX_MUSIC_ARC_ROWS
    try:
        expected_ids = _contract_event_ids(contract)
    except ValueError as exc:
        _safe_issue(issues, str(exc))
        expected_ids = []

    indexed: dict[int, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _safe_issue(issues, f"music arc row {position + 1} must be an object")
            continue
        index = row.get("clip_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            _safe_issue(issues, f"music arc row {position + 1} needs a nonnegative integer clip_index")
            continue
        if index >= row_count:
            _safe_issue(issues, f"music arc clip_index {index} is outside the {row_count}-row plan")
            continue
        if index in indexed:
            _safe_issue(issues, f"music arc clip_index {index} is repeated")
            continue
        indexed[index] = row
    if len(indexed) != row_count or set(indexed) != set(range(row_count)):
        _safe_issue(issues, "music arc must cover each row exactly once in clip_index order")

    source_assignments: list[str] = []
    assigned_positions: dict[str, list[int]] = {}
    for index in range(row_count):
        row = indexed.get(index)
        if row is None:
            continue
        for field, limit in (
            ("visual_beat", _MAX_BEAT_CHARS),
            ("purpose", _MAX_FIELD_CHARS),
            ("environment", _MAX_FIELD_CHARS),
            ("camera", _MAX_FIELD_CHARS),
        ):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                _safe_issue(issues, f"clip {index} needs nonempty {field} text")
            elif len(value) > limit:
                _safe_issue(issues, f"clip {index} {field} exceeds {limit} characters")

        beat = row.get("visual_beat")
        if isinstance(beat, str) and beat.strip():
            beat = " ".join(beat.split())
            if check_creative_quality:
                content = _content_tokens(beat)
                # Camera wording can accompany a concrete beat ("cinematic
                # shot of the cook folding herbs"). Judge the remaining
                # content instead of rejecting that phrase anywhere in it.
                concrete_content = content
                if _GENERIC_FILLER_RE.search(beat):
                    inert = _SETTING_CAMERA_WORDS | {
                        "atmospheric", "change", "changes", "complement", "complements",
                        "move", "movement", "reflect", "reflects", "sweeping", "that",
                    }
                    concrete_content = [
                        token for token in _content_tokens(_GENERIC_FILLER_RE.sub(" ", beat))
                        if token not in inert
                    ]
                if len(content) < 2 or len(concrete_content) < 2:
                    _safe_issue(issues, f"clip {index} visual_beat needs a specific visible action, change, or image")
                if _whole_brief_copy(beat, contract):
                    # A brief consisting only of one fight can equal the exact
                    # named action recovered into its sole assigned slot. That
                    # is source coverage, not global-brief recycling elsewhere.
                    from .music_narrative import combat_source_action
                    source_events = contract.get("events", [])
                    event = source_events[0] if len(source_events) == 1 else {}
                    named_action = combat_source_action(event)
                    recovered_anchor = (
                        named_action
                        and row.get("source_event_ids") == [event.get("event_id")]
                        and _normalized_arc_text(beat) == _normalized_arc_text(named_action)
                    )
                    if not recovered_anchor:
                        _safe_issue(issues, f"clip {index} visual_beat repeats the full source brief instead of one beat")

        ids = row.get("source_event_ids")
        if not _is_sequence(ids):
            _safe_issue(issues, f"clip {index} source_event_ids must be a list")
        else:
            if len(ids) > MAX_MUSIC_ARC_EVENTS:
                _safe_issue(issues, f"clip {index} has more than {MAX_MUSIC_ARC_EVENTS} source_event_ids")
            local_seen: set[str] = set()
            for event_id in ids:
                if not isinstance(event_id, str) or not event_id:
                    _safe_issue(issues, f"clip {index} source_event_ids must contain nonempty strings")
                    continue
                source_assignments.append(event_id)
                assigned_positions.setdefault(event_id, []).append(index)
                if event_id in local_seen:
                    _safe_issue(issues, f"clip {index} repeats source event {event_id!r}")
                local_seen.add(event_id)

        reprise = row.get("intentional_reprise")
        reason = row.get("reprise_reason")
        if not isinstance(reprise, bool):
            _safe_issue(issues, f"clip {index} intentional_reprise must be a boolean")
        if not isinstance(reason, str) or len(reason) > _MAX_REPRISE_REASON_CHARS:
            _safe_issue(issues, f"clip {index} reprise_reason must be text of at most {_MAX_REPRISE_REASON_CHARS} characters")

    if source_assignments != expected_ids:
        _safe_issue(
            issues,
            "music arc must assign every contract source_event_id exactly once and in source order",
        )

    if check_creative_quality and not allow_early_events and row_count >= 12 and len(expected_ids) >= 3:
        positions = [assigned_positions.get(event_id, []) for event_id in expected_ids]
        if all(len(value) == 1 for value in positions):
            opening_groups = max(1, math.ceil(row_count * 0.20))
            if max(value[0] for value in positions) < opening_groups:
                _safe_issue(
                    issues,
                    "Repairable planning issue: all source events land in roughly the opening 20% of the song; spread their payoffs across the arc unless the brief explicitly asks for an early sequence",
                )

    return issues


def music_arc_action_issues(rows: Any, contract: Any) -> list[str]:
    """Check that assigned milestones appear in their outline beats.

    IDs alone cannot assign a run to a standing beat or a boarding action to
    the shot before it happens. Reuse the detailed planner's action matcher;
    outline prose does not need a physical-state ledger or native H3 fields.
    Non-narrative concepts have no source milestones and pass through.
    """
    from .music_narrative import music_narrative_row_issue

    if not isinstance(contract, Mapping) or not contract.get("events"):
        return []
    if not _is_sequence(rows):
        return ["music arc must be an array of timeline-group rows"]
    allocation = [
        {"event_id": event_id, "clip_index": row.get("clip_index")}
        for row in rows if isinstance(row, Mapping)
        for event_id in row.get("source_event_ids", [])
    ]
    issues: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("source_event_ids"):
            continue
        issue = music_narrative_row_issue(
            {"video_prompt": row.get("visual_beat", ""),
             "source_event_ids": row["source_event_ids"]},
            contract=contract, allocation=allocation, index=row.get("clip_index"),
        )
        if issue:
            _safe_issue(issues, "Outline action mismatch: " + issue)
    return issues


def ground_music_combat_arc(rows: Any, contract: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover an omitted combat anchor from its source without resetting the arc.

    The event's assigned slot is already grammar constrained. A solo performance
    cannot stand in for that fight; keep the other beats, camera and environment
    intact and restore only the named source action at its existing slot.
    """
    from .music_narrative import _event_coverage_positions, combat_source_action

    if not _is_sequence(rows):
        return [], []
    grounded = copy.deepcopy(list(rows))
    notes: list[dict[str, Any]] = []
    events = contract.get("events", []) if isinstance(contract, Mapping) else []
    for event in events:
        if not isinstance(event, dict) or event.get("kind") != "combat":
            continue
        event_id = event.get("event_id")
        anchors = [row for row in grounded if isinstance(row, dict)
                   and event_id in row.get("source_event_ids", [])]
        if len(anchors) != 1:
            continue  # Structural/source assignment checks own this failure.
        row = anchors[0]
        if _event_coverage_positions(str(row.get("visual_beat") or ""), event):
            continue
        action = combat_source_action(event)
        if not action or len(action) > _MAX_BEAT_CHARS:
            continue  # Never truncate an actor or action to manufacture coverage.
        if not _event_coverage_positions(action, event):
            continue
        row["visual_beat"] = action
        # The rejected solo-performance purpose must not steer detail writing
        # away from the recovered source action again.
        row["purpose"] = "Develop the requested physical conflict between its named participants."
        notes.append({
            "kind": "source_combat_recovery",
            "clip_index": row["clip_index"],
            "clip_end_index": row["clip_index"],
            "message": "The outline omitted its assigned fight; restored the named source action while preserving the other beats.",
        })
    return grounded, notes


def filter_music_arc_context(
    plan: Any,
    contract: Any,
    allocation: Any,
    *,
    suppressed_indices: Any = (),
) -> list[dict[str, Any]]:
    """Return a nonmutating arc view without explicit future-event conflicts.

    Detailed writers see the full arc overview, so a row that performs an
    event before its canonical assignment can keep steering repairs toward
    that early action. Use the narrative contract's allocation as the source
    of truth; the arc's own event IDs are only hints and may be inconsistent.
    Inconclusive or merely incomplete outline prose stays available to the
    writer. A rejected row can also be suppressed explicitly during repair.
    """

    if not _is_sequence(plan):
        return []

    suppressed: set[int] = set()
    if isinstance(suppressed_indices, int) and not isinstance(suppressed_indices, bool):
        if suppressed_indices >= 0:
            suppressed.add(suppressed_indices)
    elif _is_sequence(suppressed_indices) or isinstance(suppressed_indices, (set, frozenset)):
        for index in suppressed_indices:
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
                suppressed.add(index)

    allocation_map: dict[str, int] = {}
    events: list[Mapping[str, Any]] = []
    contract_valid = isinstance(contract, Mapping) and _is_sequence(contract.get("events"))
    if contract_valid:
        events = [event for event in contract["events"] if isinstance(event, Mapping)]
        if len(events) == len(contract["events"]) and isinstance(allocation, list):
            from .music_narrative import validate_music_narrative_allocation

            if validate_music_narrative_allocation(contract, allocation, len(plan)):
                allocation = None
        if len(events) == len(contract["events"]) and isinstance(allocation, list):
            for item in allocation:
                if not isinstance(item, Mapping):
                    allocation_map.clear()
                    break
                event_id = item.get("event_id")
                clip_index = item.get("clip_index")
                if (
                    not isinstance(event_id, str)
                    or not event_id
                    or not isinstance(clip_index, int)
                    or isinstance(clip_index, bool)
                    or clip_index < 0
                ):
                    allocation_map.clear()
                    break
                allocation_map[event_id] = clip_index
            expected_ids = [str(event.get("event_id") or "") for event in events]
            if (
                len(allocation_map) != len(allocation)
                or len(allocation_map) != len(expected_ids)
                or any(not event_id for event_id in expected_ids)
                or set(allocation_map) != set(expected_ids)
            ):
                allocation_map.clear()

    filtered: list[dict[str, Any]] = []
    for raw in plan:
        if not isinstance(raw, Mapping):
            continue
        row = copy.deepcopy(dict(raw))
        index = row.get("clip_index")
        if isinstance(index, int) and not isinstance(index, bool) and index in suppressed:
            continue

        if (
            allocation_map
            and isinstance(index, int)
            and not isinstance(index, bool)
            and index >= 0
        ):
            assigned_ids = [
                str(event.get("event_id") or "")
                for event in events
                if allocation_map.get(str(event.get("event_id") or "")) == index
            ]
            row["source_event_ids"] = assigned_ids
            from .music_narrative import music_narrative_row_issue

            issue = music_narrative_row_issue(
                {
                    "source_event_ids": assigned_ids,
                    "video_prompt": str(row.get("visual_beat") or ""),
                },
                contract=contract,
                allocation=allocation,
                index=index,
            )
            if issue and "performs future music narrative event" in issue:
                continue

        filtered.append(row)
    return filtered


def normalize_music_arc(rows: Any, clips: Any, contract: Any) -> list[dict[str, Any]]:
    """Normalize an outline or expanded arc against its clip rows.

    The LLM schema is capped at 48 outline groups. The normalized form may be
    expanded to 2,048 actual clips. Rows are sorted by zero-based
    ``clip_index`` and source events must match the contract exactly once, in
    order. Expanded rows may intentionally inherit a group's visual beat, so
    semantic variety checks run on compact outlines only.
    """

    if not _is_sequence(clips):
        raise ValueError("music arc clips must be a list of timeline groups")
    clip_count = len(clips)
    if clip_count > MAX_MUSIC_ARC_CLIPS:
        raise ValueError(f"music arc supports at most {MAX_MUSIC_ARC_CLIPS} clip rows")
    rows = coerce_music_arc_rows(rows, clip_count)
    if len(rows) != clip_count:
        raise ValueError(
            f"music arc must cover exactly {clip_count} timeline groups (received {len(rows)})"
        )
    expected_ids = _contract_event_ids(contract)

    normalized: list[dict[str, Any]] = []
    expected_fields = {
        "clip_index", "visual_beat", "purpose", "environment", "camera",
        "source_event_ids", "intentional_reprise", "reprise_reason",
    }
    seen_indexes: set[int] = set()
    for position, raw in enumerate(rows):
        row_number = position + 1
        if not isinstance(raw, Mapping):
            raise ValueError(f"music arc row {row_number} must be an object")
        extra = set(raw) - expected_fields
        missing = expected_fields - set(raw)
        if extra:
            raise ValueError(f"music arc row {row_number} has unexpected fields: {sorted(map(str, extra))[:3]!r}")
        if missing:
            raise ValueError(f"music arc row {row_number} is missing required fields: {sorted(missing)!r}")

        index = raw.get("clip_index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < clip_count:
            raise ValueError(f"music arc row {row_number} clip_index must be from 0 to {max(clip_count - 1, 0)}")
        if index in seen_indexes:
            raise ValueError(f"music arc clip_index {index} is repeated")
        seen_indexes.add(index)

        ids = raw.get("source_event_ids")
        if not isinstance(ids, list) or len(ids) > MAX_MUSIC_ARC_EVENTS:
            raise ValueError(f"music arc row {row_number} source_event_ids must be a list of at most {MAX_MUSIC_ARC_EVENTS} IDs")
        if any(not isinstance(event_id, str) or not event_id or event_id != event_id.strip() for event_id in ids):
            raise ValueError(f"music arc row {row_number} source_event_ids must use exact nonempty contract IDs")
        if len(set(ids)) != len(ids):
            raise ValueError(f"music arc row {row_number} repeats a source_event_id")

        reprise = raw.get("intentional_reprise")
        if not isinstance(reprise, bool):
            raise ValueError(f"music arc row {row_number} intentional_reprise must be a boolean")
        reason_value = raw.get("reprise_reason")
        if not isinstance(reason_value, str):
            raise ValueError(f"music arc row {row_number} reprise_reason must be text")
        reason = " ".join(reason_value.split())
        if len(reason) > _MAX_REPRISE_REASON_CHARS:
            raise ValueError(f"music arc row {row_number} reprise_reason exceeds {_MAX_REPRISE_REASON_CHARS} characters")
        normalized.append({
            "clip_index": index,
            "visual_beat": _collapsed_text(raw.get("visual_beat"), "visual_beat", _MAX_BEAT_CHARS, row_number),
            "purpose": _collapsed_text(raw.get("purpose"), "purpose", _MAX_FIELD_CHARS, row_number),
            "environment": _collapsed_text(raw.get("environment"), "environment", _MAX_FIELD_CHARS, row_number),
            "camera": _collapsed_text(raw.get("camera"), "camera", _MAX_FIELD_CHARS, row_number),
            "source_event_ids": list(ids),
            "intentional_reprise": reprise,
            "reprise_reason": reason,
        })

    normalized.sort(key=lambda row: row["clip_index"])
    if seen_indexes != set(range(clip_count)):
        raise ValueError("music arc must cover each timeline group exactly once")
    # Check ownership before creative quality, which also reports allocation
    # errors but would otherwise hide the actionable event diagnostics.
    if [event_id for row in normalized for event_id in row["source_event_ids"]] != expected_ids:
        from .music_arc_repair import allocation_report, allocation_error
        report = allocation_report(arc_allocation(normalized), contract, clip_count)
        raise ValueError("music arc must assign every contract source_event_id exactly once and in source order; "
                         + allocation_error(report))
    if clip_count <= MAX_MUSIC_ARC_ROWS:
        issues = music_arc_quality_issues(normalized, contract)
        hard_issues = [issue for issue in issues if not issue.startswith("Repairable planning issue:")]
        if hard_issues:
            raise ValueError(hard_issues[0])
    return normalized


def arc_allocation(plan: Any) -> list[dict[str, Any]]:
    """Convert a normalized arc to the existing event_id/clip_index format."""

    if not _is_sequence(plan):
        raise ValueError("music arc plan must be a list")
    ordered: list[Mapping[str, Any]] = []
    for row in plan:
        if not isinstance(row, Mapping):
            raise ValueError("music arc plan rows must be objects")
        index = row.get("clip_index")
        ids = row.get("source_event_ids")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("music arc plan needs nonnegative integer clip_index values")
        if not isinstance(ids, list) or any(not isinstance(event_id, str) or not event_id for event_id in ids):
            raise ValueError(f"music arc clip {index} source_event_ids must be a list of IDs")
        ordered.append(row)
    ordered.sort(key=lambda row: row["clip_index"])
    allocation: list[dict[str, Any]] = []
    for row in ordered:
        index = row.get("clip_index")
        ids = row.get("source_event_ids")
        allocation.extend({"event_id": event_id, "clip_index": index} for event_id in ids)
    return allocation


def format_music_arc_context(
    plan: Any,
    start: Any,
    end: Any,
    *,
    one_based: bool = False,
    quality_notes: Any = None,
) -> str:
    """Format the full compact arc and detailed assignments in ``[start,end)``.

    ``start`` and ``end`` are zero-based, half-open timeline-group indexes.
    Detailed writers use ``one_based`` to match the source-action schedule;
    reviewers that return clip_index values retain zero-based labels.
    The overview collapses inherited group beats to at most 48 entries, and per-clip text is shortened to
    keep detailed three-clip prompts bounded.
    """

    if not _is_sequence(plan) or not plan:
        return ""
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in plan:
        if isinstance(row, Mapping):
            index = row.get("clip_index")
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
                by_index[index] = row
    if not by_index:
        return ""
    try:
        first = int(start) if not isinstance(start, bool) else 0
        stop = int(end) if not isinstance(end, bool) else first
    except (TypeError, ValueError, OverflowError):
        first, stop = 0, 0
    first = max(0, min(first, MAX_MUSIC_ARC_CLIPS))
    stop = max(first, min(stop, MAX_MUSIC_ARC_CLIPS))
    offset = 1 if one_based else 0

    def text(row: Mapping[str, Any], key: str, limit: int) -> str:
        value = " ".join(str(row.get(key) or "").split())
        return value if len(value) <= limit else value[:limit - 1].rstrip() + "…"

    overview_groups: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, row in sorted(by_index.items()):
        beat_key = " ".join(_tokens(str(row.get("visual_beat") or "")))
        if (
            overview_groups
            and row.get("intentional_reprise") is not True
            and beat_key
            and beat_key == " ".join(_tokens(str(overview_groups[-1][2].get("visual_beat") or "")))
            and index == overview_groups[-1][1] + 1
            and row.get("intentional_reprise") is overview_groups[-1][2].get("intentional_reprise")
        ):
            group_start, _old_end, group_row = overview_groups[-1]
            overview_groups[-1] = (group_start, index, group_row)
        else:
            overview_groups.append((index, index, row))
    overview_groups = overview_groups[:MAX_MUSIC_ARC_ROWS]
    overview = [
        f"{group_start + offset}-{group_end + offset}: {text(row, 'visual_beat', 112)}"
        + (f" [{text(row, 'purpose', 44)}]" if row.get("purpose") else "")
        for group_start, group_end, row in overview_groups
    ]
    if len(by_index) > MAX_MUSIC_ARC_ROWS and len(overview_groups) == MAX_MUSIC_ARC_ROWS:
        shown_end = overview_groups[-1][1]
        omitted = sum(1 for index in by_index if index > shown_end)
        if omitted:
            overview.append(f"Further clips through {max(by_index) + offset} continue the established progression ({omitted} rows summarized).")
    assigned: list[str] = []
    for index in range(first, stop):
        row = by_index.get(index)
        if row is None:
            continue
        ids = row.get("source_event_ids")
        id_text = repr(ids) if isinstance(ids, list) else "[]"
        line = (
            f"clip {index + offset}: visual_beat={text(row, 'visual_beat', 180)}; "
            f"purpose={text(row, 'purpose', 88)}; "
            f"environment={text(row, 'environment', 96)}; "
            f"camera={text(row, 'camera', 96)}; source_event_ids={id_text}"
        )
        if row.get("intentional_reprise") is True:
            line += f"; intentional_reprise: {text(row, 'reprise_reason', 120)}"
        assigned.append(line)

    overview_text = "\n".join(f"- {line}" for line in overview) if overview else "- None"
    assigned_text = "\n".join(f"- {line}" for line in assigned) if assigned else "- No groups in this batch"
    note_lines: list[str] = []
    if _is_sequence(quality_notes):
        for note in quality_notes[:_MAX_REPETITION_FINDINGS]:
            if not isinstance(note, Mapping):
                continue
            index = note.get("clip_index")
            end_index = note.get("clip_end_index", index)
            if (
                not isinstance(index, int) or isinstance(index, bool)
                or not isinstance(end_index, int) or isinstance(end_index, bool)
                or index < 0 or end_index < index
                or end_index >= MAX_MUSIC_ARC_CLIPS
                or end_index < first or index >= stop
            ):
                continue
            # Batch filtering can remove a range's actual arc row (for example,
            # a future-action row). Do not reintroduce its note on its own.
            overlap_start = max(index, first)
            overlap_stop = min(end_index + 1, stop, MAX_MUSIC_ARC_CLIPS)
            if not any(row_index in by_index for row_index in range(overlap_start, overlap_stop)):
                continue
            message = note.get("message")
            if isinstance(message, str) and message.strip():
                note_lines.append("- " + text({"note": message}, "note", 360))
    sections = [
        ("WHOLE-SONG MUSIC VIDEO ARC (global clip numbers, starting at 1)"
         if one_based else "WHOLE-SONG MUSIC VIDEO ARC (timeline groups in clip_index order)"),
        overview_text,
        (f"ASSIGNED CLIPS FOR THIS BATCH [global clips {first + 1}-{stop}]"
         if one_based else f"ASSIGNED GROUPS FOR THIS BATCH [zero-based {first}, {stop})"),
        assigned_text,
        "Follow the full-song progression while preserving the requested genre, concept, and continuity. "
        "Keep each assigned visual beat distinct and develop it through the group's detailed clips. "
        "Use source_event_ids exactly as assigned; keep source actions in source order and place them "
        "at their assigned group's opening. Camera or environment may stay fixed when the brief calls "
        "for it. Treat repeated outline notes as advisory guidance for distinct visible development.",
        ("Complete each assigned source action within its clip; following clips begin from its completed result."
         if one_based else ""),
    ]
    if note_lines:
        sections.append("WHOLE-SONG ARC QUALITY NOTES (advisory)\n" + "\n".join(note_lines))
    return "\n\n".join(sections)
