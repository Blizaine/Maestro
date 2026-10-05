"""Compiler-owned camera cards for silent, source-bound H3 beats.

The camera writer chooses how an assigned action is photographed. It never
rewrites the action itself: action text is compiled from immutable source
events, then attached after the camera-only draft has been validated.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


_WRITER_FIELDS = ("framing", "camera", "transition", "sound_effects")
_OPENING_FRAMING = "The supplied frame's exact opening composition"
_OPENING_TRANSITION = "continue supplied frame"
_DEFAULT_FRAMING = "cinematic medium shot"
_DEFAULT_CAMERA = "a motivated camera follows the action"
_DEFAULT_TRANSITION = "hard cut"
_CONTINUOUS_TRANSITION = "continuous reframe"
_DEFAULT_SOUND = "Natural synchronized effects"
_MAX_PHASES = 4
_TARGET_PHASE_WORDS = 90
_PROOF_SENTINEL = object()

_CLAUSE_BOUNDARY = re.compile(
    r"(?<=[.!?;])\s+|"
    r"(?<=,)\s+(?=(?:and|but|then|finally|afterward|afterwards|"
    r"meanwhile|while|as|when|before|after|who)\b)|"
    r"\s+(?=(?:then|finally|afterward|afterwards|meanwhile)\b)",
    re.IGNORECASE,
)
_WORD = re.compile(r"\S+")
_ACTION_WORD = re.compile(r"[\w’'-]+", re.UNICODE)
_LINE_LABEL = re.compile(
    r"^\s*(?:[A-Z][A-Z0-9 _'’-]{0,36})\s*:\s*(?=\S)",
)
_AMBIGUOUS_OPTICAL_REFERENCE = re.compile(
    r"\b(?:man|woman|person|people|fighter|actor|character|body)\b|"
    r"\b(?:show(?:s|ed|ing)?|watch(?:es|ed|ing)?|see(?:s|ing)?|"
    r"reveal(?:s|ed|ing)?|shatter(?:s|ed|ing)?|rip(?:s|ped|ping)?|"
    r"smash(?:es|ed|ing)?|slam(?:s|med|ming)?|strike|strikes|struck|"
    r"kick|kicks|kicked|punch|punches|punching|spin|spins|spinning|"
    r"crash|crashes|crashing|fall|falls|falling|fly|flies|flying|"
    r"launch|launches|launching|streak|streaks|streaking)\b",
    re.IGNORECASE,
)


def _ledger_helpers():
    """Load existing ledger normalizers lazily to avoid a module cycle."""

    from services.h3_story_ledger import (
        _filmable_source_event,
        _find_spoken_verb,
        _h3_strip_inline_context_labels,
        sanitize_h3_prompt_text,
        strip_h3_source_clock_cues,
    )

    return (
        _filmable_source_event,
        _find_spoken_verb,
        _h3_strip_inline_context_labels,
        sanitize_h3_prompt_text,
        strip_h3_source_clock_cues,
    )


def _optical_helpers():
    from services.h3_optical_actions import optical_only_clauses

    return optical_only_clauses


def _cast_pattern(cast_names: Any) -> re.Pattern[str] | None:
    if not isinstance(cast_names, (list, tuple, set)):
        return None
    names = sorted(
        {str(value).strip() for value in cast_names if isinstance(value, str) and value.strip()},
        key=lambda value: (-len(value), value.casefold()),
    )
    if not names:
        return None
    alternatives = "|".join(re.escape(value) for value in names)
    return re.compile(rf"(?<![\w])(?:{alternatives})(?![\w])", re.IGNORECASE)


def _safe_optical_cues(
    unit: str,
    optical_only_clauses,
    cast_pattern: re.Pattern[str] | None,
) -> tuple[str, ...]:
    """Keep camera metadata only for unambiguous camera-owned wording.

    The optical helper is useful for detecting candidate lens clauses, but its
    comparison text is never used as source prose. A candidate that names a
    cast member, person/body, or physical predicate stays only in the immutable
    action string.
    """

    safe: list[str] = []
    for value in optical_only_clauses(unit, cast_pattern=cast_pattern):
        cue = str(value or "").strip()
        if not cue or _AMBIGUOUS_OPTICAL_REFERENCE.search(cue):
            continue
        if cast_pattern is not None and cast_pattern.search(cue):
            continue
        safe.append(cue)
    return tuple(safe)


def _normalize_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _has_metadata_key(record: Mapping[str, Any], terms: tuple[str, ...]) -> bool:
    return any(any(term in str(key).casefold() for term in terms) for key in record)


def _has_unsupported_occurrence_or_relation(record: Mapping[str, Any]) -> bool:
    if _has_metadata_key(record, ("recurr", "occurrence", "enabler")):
        return True
    for key, value in record.items():
        normalized_key = str(key).casefold()
        if normalized_key in {"text", "description", "summary"}:
            continue
        if isinstance(value, Mapping) and _has_unsupported_occurrence_or_relation(value):
            return True
        if isinstance(value, list) and any(
            isinstance(item, Mapping) and _has_unsupported_occurrence_or_relation(item)
            for item in value
        ):
            return True
        if isinstance(value, str) and any(
            term in value.casefold() for term in ("same_passage_enabler", "spaced_recurrence")
        ):
            return True
    return False


def _metadata_value_is_set(value: Any) -> bool:
    if value is None or value is False or value == "" or value == [] or value == {}:
        return False
    return True


def _has_generated_marker(record: Mapping[str, Any]) -> bool:
    for key, value in record.items():
        normalized_key = str(key).casefold()
        if "generated" in normalized_key and _metadata_value_is_set(value):
            return True
        if normalized_key in {"text", "description", "summary"}:
            continue
        if isinstance(value, Mapping) and _has_generated_marker(value):
            return True
        if isinstance(value, list) and any(
            isinstance(item, Mapping) and _has_generated_marker(item)
            for item in value
        ):
            return True
    return False


def _source_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _source_map_texts(source_event_map: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(source_event_map, Mapping):
        items = source_event_map.items()
    elif isinstance(source_event_map, list):
        items = (
            (item.get("event_id"), item)
            for item in source_event_map
            if isinstance(item, Mapping)
        )
    else:
        return result
    for key, value in items:
        event_id = _normalize_id(key or (value.get("event_id") if isinstance(value, Mapping) else ""))
        if not event_id:
            continue
        if isinstance(value, Mapping):
            text = value.get("text")
        else:
            text = value
        if isinstance(text, str):
            result[event_id] = text
    return result


def _canonical_key(value: Any) -> str:
    _, _, _, sanitize, _ = _ledger_helpers()
    return " ".join(sanitize(value).casefold().split())


def _canonical_action(value: Any) -> str:
    _, _, _, sanitize, _ = _ledger_helpers()
    return sanitize(value).strip()


def _split_long_clause(value: str, target_words: int = _TARGET_PHASE_WORDS) -> list[str]:
    # Actor/contact and cause/effect clauses stay indivisible. Ninety words is
    # a readability target for grouping, never a reason to cut a source clause.
    return [value.strip()] if value.strip() else []


def _split_source_units(value: str) -> list[str]:
    units = [part.strip() for part in _CLAUSE_BOUNDARY.split(value) if part.strip()]
    chunks = [chunk for unit in units for chunk in _split_long_clause(unit)]
    return chunks or ([value.strip()] if value.strip() else [])


def _word_count(value: str) -> int:
    return len(_WORD.findall(value))


def _phase_groups(fragments: list[dict[str, Any]], limit: int) -> list[list[dict[str, Any]]]:
    """Pack contiguous source fragments into at most ``limit`` readable phases."""

    if not fragments:
        return []
    if len(fragments) <= limit:
        return [[fragment] for fragment in fragments]

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_words = 0
    target_words = max(
        1,
        (sum(max(1, _word_count(item["action"])) for item in fragments) + limit - 1) // limit,
    )
    for index, fragment in enumerate(fragments):
        words = max(1, _word_count(fragment["action"]))
        remaining_fragments = len(fragments) - index
        remaining_groups = limit - len(groups)
        should_split = bool(
            current
            and current_words + words > target_words
            and len(groups) < limit - 1
            and remaining_fragments >= remaining_groups
        )
        if should_split:
            groups.append(current)
            current = []
            current_words = 0
        current.append(fragment)
        current_words += words
    if current:
        groups.append(current)
    if len(groups) <= limit:
        return groups

    # Very clause-dense prose can exceed the target phase count. Repack in
    # source order; phases may exceed 90 words, but no text is cut.
    packed: list[list[dict[str, Any]]] = [[] for _ in range(limit)]
    total_words = sum(max(1, _word_count(item["action"])) for item in fragments)
    target = max(1, (total_words + limit - 1) // limit)
    group_index = 0
    group_words = 0
    for index, fragment in enumerate(fragments):
        remaining_items = len(fragments) - index
        remaining_groups = limit - group_index
        if (
            group_index < limit - 1
            and packed[group_index]
            and group_words + max(1, _word_count(fragment["action"])) > target
            and remaining_items >= remaining_groups
        ):
            group_index += 1
            group_words = 0
        packed[group_index].append(fragment)
        group_words += max(1, _word_count(fragment["action"]))
    return [group for group in packed if group]


def _merge_text(parts: list[str], *, separator: str = " ") -> str:
    return separator.join(part.strip() for part in parts if part and part.strip()).strip()


@dataclass(frozen=True)
class _PhasePlan:
    action: str
    source_event_ids: tuple[str, ...]
    optical_clauses: tuple[str, ...]


@dataclass(frozen=True)
class _BeatPlan:
    beat_id: str
    source_event_ids: tuple[str, ...]
    phases: tuple[_PhasePlan, ...]
    opening: bool
    canonical_action: str


class SourceCameraProof:
    """Opaque evidence that a bound camera draft retained its source actions."""

    __slots__ = ("_nonce", "_plans", "_source_hashes")

    def __init__(self, plans, source_hashes, *, _sentinel=None):
        if _sentinel is not _PROOF_SENTINEL:
            raise TypeError("SourceCameraProof values are created by the compiler")
        self._nonce = object()
        self._plans = tuple(plans)
        self._source_hashes = tuple(sorted(source_hashes.items()))

    def __deepcopy__(self, memo):
        # The proof is immutable and should keep its authority through the
        # ledger's ordinary defensive copies.
        return self

    def __repr__(self) -> str:
        return "<SourceCameraProof opaque>"

    @property
    def expected_actions_by_beat(self) -> dict[str, tuple[str, ...]]:
        return {plan.beat_id: tuple(phase.action for phase in plan.phases) for plan in self._plans}

    @property
    def source_hashes(self) -> dict[str, str]:
        return dict(self._source_hashes)

    def covers_beat(
        self,
        segment: Any,
        beat: Any,
        source_event_map: Any,
    ) -> bool:
        if not isinstance(segment, Mapping) or not isinstance(beat, Mapping):
            return False
        if segment.get("event_assignment_error") or segment.get("dialogue_assignment_error"):
            return False
        attached = segment.get("_source_camera_proof")
        if not isinstance(attached, SourceCameraProof) or attached._nonce is not self._nonce:
            return False

        beat_id = _normalize_id(beat.get("beat_id"))
        plan = next((item for item in self._plans if item.beat_id == beat_id), None)
        if plan is None:
            return False
        if tuple(_normalize_id(value) for value in (beat.get("source_event_ids") or [])) != plan.source_event_ids:
            return False
        if bool(beat.get("_start_frame_continuation")) != plan.opening:
            return False
        if beat.get("dialogue_ids"):
            return False
        if _has_unsupported_occurrence_or_relation(beat):
            return False
        if _has_generated_marker(beat) or any(
            _metadata_value_is_set(beat.get(key))
            for key in ("_audio_driven", "_final_state_only", "_final_state_source_event_ids")
        ):
            return False
        beat_canonical = beat.get("_canonical_action")
        if beat_canonical and _canonical_action(beat_canonical) != _canonical_action(plan.canonical_action):
            return False

        source_texts = _source_map_texts(source_event_map)
        expected_hashes = dict(self._source_hashes)
        source_records: dict[str, Mapping[str, Any]] = {}
        if isinstance(source_event_map, Mapping):
            source_records = {
                _normalize_id(key): value
                for key, value in source_event_map.items()
                if isinstance(value, Mapping)
            }
        elif isinstance(source_event_map, list):
            source_records = {
                _normalize_id(value.get("event_id")): value
                for value in source_event_map
                if isinstance(value, Mapping)
            }
        if any(
            source_id not in source_texts
            or _source_hash(source_texts[source_id]) != expected_hashes.get(source_id)
            or (
                source_id in source_records
                and (
                    _has_unsupported_occurrence_or_relation(source_records[source_id])
                    or _has_generated_marker(source_records[source_id])
                    or source_records[source_id].get("requirement_kind") == "final_state"
                    or bool(source_records[source_id].get("dialogue_ids"))
                )
            )
            for source_id in plan.source_event_ids
        ):
            return False

        expected_actions = [phase.action for phase in plan.phases]
        expected_optics = [phase.optical_clauses for phase in plan.phases]
        if isinstance(segment.get("event_cards"), Mapping):
            event_index = next(
                (index for index, item in enumerate(self._plans, start=1) if item.beat_id == beat_id),
                None,
            )
            if event_index is None:
                return False
            event = segment["event_cards"].get(f"event_{event_index}")
            if not isinstance(event, Mapping):
                return False
            actions: list[str] = []
            cameras: list[str] = []
            if plan.opening:
                opening = event.get("opening")
                if not isinstance(opening, Mapping):
                    return False
                if (
                    opening.get("framing") != _OPENING_FRAMING
                    or opening.get("transition") != _OPENING_TRANSITION
                    or opening.get("recovery") != ""
                ):
                    return False
                actions.append(str(opening.get("action") or ""))
                cameras.append(str(opening.get("camera") or "") + " " + str(opening.get("framing") or ""))
            phases = event.get("phases")
            if not isinstance(phases, list):
                return False
            actions.extend(str(item.get("action") or "") for item in phases if isinstance(item, Mapping))
            cameras.extend(
                str(item.get("camera") or "") + " " + str(item.get("framing") or "")
                for item in phases if isinstance(item, Mapping)
            )
        else:
            shots = segment.get("shots")
            if not isinstance(shots, list):
                return False
            owned = [
                shot for shot in shots
                if isinstance(shot, Mapping)
                and beat_id in [_normalize_id(value) for value in (shot.get("beat_ids") or [])]
            ]
            if any([_normalize_id(value) for value in (shot.get("beat_ids") or [])] != [beat_id] for shot in owned):
                return False
            if plan.opening:
                first_shot = next((shot for shot in shots if isinstance(shot, Mapping)), None)
                opening_shot = owned[0] if owned else None
                if (
                    first_shot is not opening_shot
                    or opening_shot.get("framing") != _OPENING_FRAMING
                    or opening_shot.get("transition") != _OPENING_TRANSITION
                    or "recovery" in opening_shot
                ):
                    return False
            actions = [str(shot.get("action") or "") for shot in owned]
            cameras = [
                str(shot.get("camera") or "") + " " + str(shot.get("framing") or "")
                for shot in owned
            ]

        if len(actions) != len(expected_actions):
            return False
        if [_canonical_action(action) for action in actions] != [
            _canonical_action(action) for action in expected_actions
        ]:
            return False
        for camera, optical in zip(cameras, expected_optics):
            camera_key = _canonical_key(camera)
            if any(_canonical_key(cue) not in camera_key for cue in optical):
                return False
        return True


class SourceCameraContract:
    """Strict camera-only schema plus compiler-owned source action phases."""

    def __init__(
        self,
        plans: list[_BeatPlan],
        source_hashes: dict[str, str],
        camera_coverage: str,
    ):
        self._plans = tuple(plans)
        self._source_hashes = dict(source_hashes)
        self._camera_coverage = camera_coverage
        self._proof = SourceCameraProof(
            self._plans, self._source_hashes, _sentinel=_PROOF_SENTINEL,
        )

    def _phase_keys(self, plan: _BeatPlan) -> list[str]:
        return [f"phase_{index}" for index in range(1, len(plan.phases) - int(plan.opening) + 1)]

    def _phase_draft_fields(self, *, opening: bool = False) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "framing": {"type": "string"},
            "camera": {"type": "string"},
            "transition": {"type": "string"},
            "sound_effects": {"type": "string"},
        }
        if opening:
            properties["framing"] = {"type": "string", "const": _OPENING_FRAMING}
            properties["transition"] = {"type": "string", "const": _OPENING_TRANSITION}
        elif self._camera_coverage == "continuous":
            properties["transition"] = {"type": "string", "const": _CONTINUOUS_TRANSITION}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def schema(self, segment_number: int) -> dict[str, Any]:
        if type(segment_number) is not int or segment_number < 1:
            raise ValueError("segment_number must be a positive integer")
        events: dict[str, Any] = {}
        for index, plan in enumerate(self._plans, start=1):
            properties: dict[str, Any] = {}
            if plan.opening:
                properties["opening"] = self._phase_draft_fields(opening=True)
            for phase_key in self._phase_keys(plan):
                properties[phase_key] = self._phase_draft_fields()
            events[f"event_{index}"] = {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        return {
            "type": "object",
            "properties": {
                "segment": {"type": "integer", "minimum": segment_number, "maximum": segment_number},
                "title": {"type": "string"},
                "coverage": {"type": "string"},
                "pacing": {"type": "string"},
                "event_cards": {
                    "type": "object",
                    "properties": events,
                    "required": list(events),
                    "additionalProperties": False,
                },
            },
            "required": ["segment", "title", "coverage", "pacing", "event_cards"],
            "additionalProperties": False,
        }

    def prompt_events(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index, plan in enumerate(self._plans, start=1):
            event: dict[str, Any] = {
                "event_card": f"event_{index}",
                "beat_id": plan.beat_id,
                "source_event_ids": list(plan.source_event_ids),
            }
            source_phase_index = 0
            if plan.opening:
                phase = plan.phases[0]
                event["opening"] = {
                    "source_action": phase.action,
                    "source_event_ids": list(phase.source_event_ids),
                    "authored_optics": list(phase.optical_clauses),
                }
                source_phase_index = 1
            for output_index, phase in enumerate(plan.phases[source_phase_index:], start=1):
                event[f"phase_{output_index}"] = {
                    "source_action": phase.action,
                    "source_event_ids": list(phase.source_event_ids),
                    "authored_optics": list(phase.optical_clauses),
                }
            result.append(event)
        return result

    @staticmethod
    def _merge_optics(camera: str, clauses: tuple[str, ...]) -> str:
        value = camera.strip()
        key = " ".join(value.casefold().split())
        additions: list[str] = []
        for clause in clauses:
            clause_key = " ".join(clause.casefold().split())
            if clause_key and clause_key not in key:
                additions.append(clause)
                key = " ".join(f"{key} {clause_key}".split())
        if not additions:
            return value
        return ". ".join(
            part.rstrip(" .") for part in [*additions, value] if part.strip()
        )

    def bind(self, draft: Any) -> dict[str, Any]:
        """Validate camera-only JSON and insert exact source actions."""

        top_fields = {"segment", "title", "coverage", "pacing", "event_cards"}
        if not isinstance(draft, Mapping) or set(draft) != top_fields:
            raise ValueError("Camera draft needs exactly the segment metadata and event_cards fields.")
        if type(draft.get("segment")) is not int or draft["segment"] < 1:
            raise ValueError("Camera draft needs a positive integer segment.")
        for field in ("title", "coverage", "pacing"):
            if not isinstance(draft.get(field), str):
                raise ValueError(f"Camera draft field {field} must be a string.")
        raw_events = draft.get("event_cards")
        expected_events = [f"event_{index}" for index in range(1, len(self._plans) + 1)]
        if not isinstance(raw_events, Mapping) or set(raw_events) != set(expected_events):
            raise ValueError(
                "event_cards is missing, foreign, or contains duplicate event cards."
            )

        bound_events: dict[str, Any] = {}
        for index, plan in enumerate(self._plans, start=1):
            event_key = f"event_{index}"
            raw_event = raw_events[event_key]
            expected_keys = (["opening"] if plan.opening else []) + self._phase_keys(plan)
            if not isinstance(raw_event, Mapping) or set(raw_event) != set(expected_keys):
                raise ValueError(
                    f"event_cards.{event_key}: needs exactly its fixed camera phase keys."
                )

            source_index = 0
            old_event: dict[str, Any] = {}
            if plan.opening:
                opening_draft = raw_event["opening"]
                self._validate_card(
                    opening_draft, opening=True,
                    location=f"event_cards.{event_key}: opening",
                )
                first_phase = plan.phases[0]
                old_event["opening"] = {
                    "action": first_phase.action,
                    "framing": _OPENING_FRAMING,
                    "camera": self._merge_optics(str(opening_draft["camera"]), first_phase.optical_clauses),
                    "transition": _OPENING_TRANSITION,
                    "sound_effects": str(opening_draft["sound_effects"]),
                    "recovery": "",
                }
                source_index = 1

            old_phases: list[dict[str, str]] = []
            for output_index, phase in enumerate(plan.phases[source_index:], start=1):
                phase_key = f"phase_{output_index}"
                phase_draft = raw_event[phase_key]
                self._validate_card(
                    phase_draft,
                    location=f"event_cards.{event_key}: {phase_key}",
                    required_transition=(
                        _CONTINUOUS_TRANSITION
                        if self._camera_coverage == "continuous"
                        else None
                    ),
                )
                old_phases.append({
                    "action": phase.action,
                    "framing": str(phase_draft["framing"]),
                    "camera": self._merge_optics(str(phase_draft["camera"]), phase.optical_clauses),
                    "transition": str(phase_draft["transition"]),
                    "sound_effects": str(phase_draft["sound_effects"]),
                })
            old_event["phases"] = old_phases
            bound_events[event_key] = old_event

        result = {field: draft[field] for field in ("segment", "title", "coverage", "pacing")}
        result["event_cards"] = bound_events
        result["_source_camera_proof"] = self._proof
        return result

    @staticmethod
    def _validate_card(
        value: Any,
        *,
        opening: bool = False,
        location: str = "event_cards",
        required_transition: str | None = None,
    ) -> None:
        if not isinstance(value, Mapping) or set(value) != set(_WRITER_FIELDS):
            raise ValueError(
                f"{location} may contain only framing, camera, transition, and sound_effects."
            )
        if any(not isinstance(value.get(field), str) for field in _WRITER_FIELDS):
            raise ValueError(f"Every field in {location} must be a string.")
        if opening and (
            value.get("framing") != _OPENING_FRAMING
            or value.get("transition") != _OPENING_TRANSITION
        ):
            raise ValueError(
                f"{location} must keep the supplied-frame opening composition and transition."
            )
        if required_transition and value.get("transition") != required_transition:
            raise ValueError(
                f"{location} must use the {required_transition!r} transition."
            )

    def fallback_event(self, event_key: str) -> dict[str, Any] | None:
        """Return camera-only defaults for one known event card."""

        match = re.fullmatch(r"event_(\d+)", str(event_key or ""))
        if not match:
            return None
        index = int(match.group(1))
        if index < 1 or index > len(self._plans):
            return None
        plan = self._plans[index - 1]
        event: dict[str, Any] = {}
        source_index = 0
        if plan.opening:
            phase = plan.phases[0]
            event["opening"] = {
                "framing": _OPENING_FRAMING,
                "camera": self._merge_optics(_DEFAULT_CAMERA, phase.optical_clauses),
                "transition": _OPENING_TRANSITION,
                "sound_effects": _DEFAULT_SOUND,
            }
            source_index = 1
        for output_index, phase in enumerate(plan.phases[source_index:], start=1):
            event[f"phase_{output_index}"] = {
                "framing": _DEFAULT_FRAMING,
                "camera": self._merge_optics(_DEFAULT_CAMERA, phase.optical_clauses),
                "transition": (
                    _CONTINUOUS_TRANSITION
                    if self._camera_coverage == "continuous"
                    else _DEFAULT_TRANSITION
                ),
                "sound_effects": _DEFAULT_SOUND,
            }
        return event


def _build_phase_plan(
    beat: Mapping[str, Any],
    source_ids: list[str],
    source_texts: Mapping[str, str],
    *,
    opening: bool,
    cast_pattern: re.Pattern[str] | None,
) -> _BeatPlan | None:
    filmable, find_spoken, strip_labels, sanitize, strip_clocks = _ledger_helpers()
    optical_only_clauses = _optical_helpers()
    fragments: list[dict[str, Any]] = []
    canonical_parts: list[str] = []

    for source_id in source_ids:
        raw = source_texts[source_id]
        source_text = sanitize(strip_labels(raw))
        # Remove a source line label only after the ledger's context-label
        # normalizer has run; never reject or rewrite the action that follows.
        source_text = _LINE_LABEL.sub("", source_text, count=1).strip()
        if not source_text or find_spoken(source_text):
            return None
        canonical = strip_clocks(filmable(source_text)).strip()
        if not canonical:
            return None
        canonical_parts.append(canonical)
        for unit in _split_source_units(canonical):
            # Preserve complete canonical source units verbatim in action.
            # `physical_action_text` is comparison-only and must never replace
            # this compiler-owned source string: mixed lens/action clauses can
            # otherwise lose the very physical facts this contract protects.
            optics = _safe_optical_cues(unit, optical_only_clauses, cast_pattern)
            candidate_optics = optical_only_clauses(unit, cast_pattern=cast_pattern)
            pure_optical = bool(candidate_optics) and not unit.casefold().startswith(("he ", "she ", "they "))
            fragments.append({
                "action": unit,
                "optical": list(optics),
                "pure_optical": pure_optical,
                "source_event_ids": [source_id],
            })

    # A unit authored solely as a lens instruction remains verbatim in action,
    # but is packed into an adjacent physical phase so it cannot manufacture a
    # separate temporal hold. Mixed lens/action units are never split.
    if fragments and all(item["pure_optical"] for item in fragments):
        return None
    folded_fragments: list[dict[str, Any]] = []
    pending_optics: list[dict[str, Any]] = []
    for item in fragments:
        if item["pure_optical"]:
            pending_optics.append(item)
            continue
        if pending_optics:
            item["action"] = _merge_text([
                *(pending["action"] for pending in pending_optics), item["action"],
            ])
            item["optical"] = [
                *(cue for pending in pending_optics for cue in pending["optical"]),
                *item["optical"],
            ]
            item["source_event_ids"] = [
                *(source_id for pending in pending_optics for source_id in pending["source_event_ids"]),
                *item["source_event_ids"],
            ]
            pending_optics = []
        folded_fragments.append(item)
    if pending_optics:
        if not folded_fragments:
            return None
        last = folded_fragments[-1]
        last["action"] = _merge_text([
            last["action"], *(pending["action"] for pending in pending_optics),
        ])
        last["optical"].extend(
            cue for pending in pending_optics for cue in pending["optical"]
        )
        last["source_event_ids"].extend(
            source_id for pending in pending_optics for source_id in pending["source_event_ids"]
        )
    fragments = folded_fragments

    # The ledger's grouped canonical action is a useful provenance check when
    # present. It is never used as the action source; source_events remain so.
    proposed_canonical = beat.get("_canonical_action")
    if proposed_canonical:
        joined = " Then ".join(canonical_parts)
        if _canonical_key(proposed_canonical) != _canonical_key(joined):
            return None

    phase_limit = _MAX_PHASES - int(opening)
    if opening:
        phase_limit = _MAX_PHASES
    groups = _phase_groups(fragments, phase_limit)
    if not groups or len(groups) > phase_limit:
        return None

    phases: list[_PhasePlan] = []
    compiled_source_words: list[str] = []
    for group in groups:
        action_parts: list[tuple[str, bool]] = []
        optical: list[str] = []
        event_ids: list[str] = []
        previous_source_id = ""
        for item in group:
            for source_id in item["source_event_ids"]:
                if source_id != previous_source_id and action_parts and source_id not in event_ids:
                    action_parts.append(("Then", True))
                if source_id not in event_ids:
                    event_ids.append(source_id)
                previous_source_id = source_id
            action_parts.append((item["action"], False))
            optical.extend(item["optical"])
        action = _merge_text([part for part, _compiler_joiner in action_parts])
        if not action:
            return None
        if [_token.casefold() for _token in _ACTION_WORD.findall(action)] != [
            _token.casefold()
            for part, _compiler_joiner in action_parts
            for _token in _ACTION_WORD.findall(part)
        ]:
            return None
        compiled_source_words.extend(
            token.casefold()
            for part, is_compiler_joiner in action_parts
            if not is_compiler_joiner
            for token in _ACTION_WORD.findall(part)
        )
        phases.append(_PhasePlan(
            action=action,
            source_event_ids=tuple(event_ids),
            optical_clauses=tuple(dict.fromkeys(optical)),
        ))
    expected_source_words = [
        token.casefold()
        for canonical in canonical_parts
        for token in _ACTION_WORD.findall(canonical)
    ]
    if compiled_source_words != expected_source_words:
        # Fail closed if phase packing/folding ever drops or reorders authored
        # source words. Synthetic cross-event ``Then`` joiners are excluded.
        return None
    if opening and len(phases) > _MAX_PHASES:
        return None
    if not opening and not 1 <= len(phases) <= _MAX_PHASES:
        return None
    return _BeatPlan(
        beat_id=_normalize_id(beat.get("beat_id")),
        source_event_ids=tuple(source_ids),
        phases=tuple(phases),
        opening=opening,
        canonical_action=" Then ".join(canonical_parts),
    )


def build_source_camera_contract(
    assigned_beats: Any,
    *,
    source_events: Any,
    camera_coverage: str = "multi_shot",
    cast_names: Any = None,
) -> SourceCameraContract | None:
    """Build a bounded contract for fully source-bound silent camera beats.

    Ineligible or ambiguous input returns ``None`` so the caller can keep its
    existing camera-writing path.
    """

    if camera_coverage not in {"multi_shot", "continuous"}:
        return None
    if not isinstance(assigned_beats, list) or not assigned_beats:
        return None
    if not isinstance(source_events, list) or not source_events:
        return None

    source_map: dict[str, Mapping[str, Any]] = {}
    source_order: dict[str, int] = {}
    seen_source_ids: set[str] = set()
    for index, event in enumerate(source_events):
        if not isinstance(event, Mapping):
            return None
        event_id = _normalize_id(event.get("event_id"))
        if (
            not event_id
            or event_id in seen_source_ids
            or not isinstance(event.get("text"), str)
        ):
            return None
        seen_source_ids.add(event_id)
        if event.get("requirement_kind") == "final_state":
            # Final-state requirements need their dedicated endpoint guards.
            continue
        if _has_unsupported_occurrence_or_relation(event):
            return None
        if _has_generated_marker(event) or event.get("dialogue_ids"):
            return None
        source_map[event_id] = event
        source_order[event_id] = index

    assigned_ids: list[str] = []
    seen_beat_ids: set[str] = set()
    normalized_beats: list[tuple[Mapping[str, Any], list[str]]] = []
    first_frame_indexes: list[int] = []
    for index, beat in enumerate(assigned_beats):
        if not isinstance(beat, Mapping):
            return None
        beat_id = _normalize_id(beat.get("beat_id"))
        if not beat_id or beat_id in seen_beat_ids:
            return None
        seen_beat_ids.add(beat_id)
        if beat.get("dialogue_ids"):
            return None
        if _has_unsupported_occurrence_or_relation(beat):
            return None
        if _has_generated_marker(beat):
            return None
        if any(
            _metadata_value_is_set(beat.get(key))
            for key in ("_audio_driven", "_final_state_only", "_final_state_source_event_ids")
        ):
            return None
        if beat.get("_start_frame_continuation"):
            first_frame_indexes.append(index)
        raw_ids = beat.get("source_event_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            return None
        source_ids = [_normalize_id(value) for value in raw_ids]
        if any(not source_id or source_id not in source_map for source_id in source_ids):
            return None
        if len(source_ids) != len(set(source_ids)):
            return None
        assigned_ids.extend(source_ids)
        normalized_beats.append((beat, source_ids))

    if first_frame_indexes and first_frame_indexes != [0]:
        return None
    if len(assigned_ids) != len(set(assigned_ids)):
        return None
    positions = [source_order[source_id] for source_id in assigned_ids]
    if positions != sorted(positions):
        return None

    try:
        from services.h3_recurrence import recurring_source_event_ids

        recurring_ids = {
            _normalize_id(value) for value in recurring_source_event_ids(source_events)
        }
    except Exception:
        # An inability to prove the recurrence metadata is absence of proof.
        return None
    if set(assigned_ids) & recurring_ids:
        return None

    source_texts = {
        source_id: str(event.get("text") or "")
        for source_id, event in source_map.items()
    }
    plans: list[_BeatPlan] = []
    cast_names_pattern = _cast_pattern(cast_names)
    for index, (beat, source_ids) in enumerate(normalized_beats):
        opening = bool(index == 0 and beat.get("_start_frame_continuation"))
        plan = _build_phase_plan(
            beat, source_ids, source_texts, opening=opening,
            cast_pattern=cast_names_pattern,
        )
        if plan is None:
            return None
        plans.append(plan)

    hashes = {source_id: _source_hash(source_texts[source_id]) for source_id in assigned_ids}
    return SourceCameraContract(plans, hashes, camera_coverage)
