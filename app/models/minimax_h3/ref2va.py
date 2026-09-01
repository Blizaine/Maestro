"""Official MiniMax H3 Ref2VA media preparation and packed layout.

This is a local-file-oriented port of the Hugging Face Diffusers Ref2VA
blocks pinned in UPSTREAM.md. Maestro keeps request validation separate from
media decoding so malformed jobs fail before they enter the generation queue.
"""

from __future__ import annotations

import math
import os
import re
import threading
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    _fill_audio_condition_positions,
    _spatial_position_grid,
    _temporal_position_grid,
    _temporal_position_span,
    _unpack_condition_anchor,
    resolve_canvas_size,
)
from .reference_manifest import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    validate_reference_manifest,
)


MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2
_REFERENCE_TAG_RE = re.compile(r"<(?:Picture|Video|Audio)\s+\d+>", re.IGNORECASE)
_PICTURE_TAG_RE = re.compile(r"<Picture\s+(\d+)>", re.IGNORECASE)
_DIALOGUE_TAG_RE = re.compile(r"<d(?:\s+[^>]*)?>(.*?)</d>", re.IGNORECASE | re.DOTALL)
_SPEAKER_MARKER_RE = re.compile(r"\(S(\d+)\)", re.IGNORECASE)
_AUDIO_LABEL_RE = re.compile(
    r"(?P<tag><Audio\s+(?P<tag_index>\d+)>)|"
    r"(?P<plain>\bAudio\s+(?P<plain_index>\d+)\b)",
    re.IGNORECASE,
)
_SPEECH_VERB_RE = re.compile(
    r"\b(?:say|says|said|ask|asks|asked|reply|replies|replied|respond|responds|"
    r"responded|answer|answers|answered|speak|speaks|spoke|shout|shouts|shouted|"
    r"yell|yells|yelled|whisper|whispers|whispered|exclaim|exclaims|exclaimed|"
    r"murmur|murmurs|murmured|call|calls|called|cry|cries|cried|add|adds|added|"
    r"remark|remarks|remarked|state|states|stated|declare|declares|declared|"
    r"warn|warns|warned|demand|demands|demanded|tell|tells|told)\b",
    re.IGNORECASE,
)
_DIALOGUE_OWNER_NAME_RE = re.compile(
    r"(?<!\w)([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})(?!\w)"
)
_DIALOGUE_OWNER_LEADING_WORDS = {
    "a", "after", "an", "and", "as", "at", "both", "during", "from",
    "he", "her", "his", "immediately", "in", "inside", "it", "later",
    "next", "only", "outside", "she", "the", "their", "then", "they",
    "this", "while", "with",
}

_REF2VA_DIALOGUE_OWNERSHIP_CONTRACT = (
    "Each tagged line is performed once by its adjacent <Subject N> (Sx) speaker."
)
_REF2VA_VOICE_ACOUSTIC_CONTRACT = (
    "Voice references define vocal identity, timbre, emotion, and delivery; each new performance "
    "uses the distance, reflections, and ambience of the target environment."
)
_REF2VA_IDENTITY_ISOLATION_CONTRACT = (
    "Identity references define subject appearance; the target uses the newly described setting, "
    "composition, pose, camera view, and natural motion."
)

_REFERENCE_BACKGROUND_SESSION = None
_REFERENCE_BACKGROUND_SESSION_LOCK = threading.Lock()
_REFERENCE_BACKGROUND_RUN_LOCK = threading.Lock()

_REF2VA_CONTEXT_IR_HEADERS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


def _normalize_ref2va_context_ir_sections(text: str) -> str:
    """Keep MiniMax's six Context-IR fields as distinct prompt sections.

    Window-scoped reference filtering edits an already compiled prompt.  An
    omitted reference clause can be the final sentence before the next field;
    if its terminating newline is removed as part of that clause, MiniMax sees
    ``summary`` as more subject prose (and ``detailed_description`` as
    retention prose).  Besides confusing narration with dialogue, that can
    promote an identity reference into target footage.  Normalize recognized
    field boundaries after every such edit, including saved prompts produced
    by older Maestro versions.
    """

    source = str(text or "").strip()
    if not source or not re.search(
        r"(?i)\bsubject_definitions\s*:",
        source,
    ):
        return source
    for header in _REF2VA_CONTEXT_IR_HEADERS[1:]:
        source = re.sub(
            rf"(?:[ \t]+|\r?\n[ \t]*)({re.escape(header)}\s*:)",
            lambda match: f"\n\n{match.group(1)}",
            source,
            flags=re.IGNORECASE,
        )
    source = re.sub(r"\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)+", "\n\n", source)
    return source.strip()


def _validate_ref2va_context_ir_sections(text: str) -> str:
    """Reject a complete Context-IR prompt if its field layout is corrupted."""

    source = _normalize_ref2va_context_ir_sections(text)
    present = [
        bool(re.search(rf"(?mi)^\s*{re.escape(header)}\s*:", source))
        for header in _REF2VA_CONTEXT_IR_HEADERS
    ]
    # Advanced manual prompts may intentionally provide only a subset.  A
    # generated six-field prompt, however, must remain complete and ordered.
    if not all(present):
        return source
    offsets = [
        re.search(rf"(?mi)^\s*{re.escape(header)}\s*:", source).start()
        for header in _REF2VA_CONTEXT_IR_HEADERS
    ]
    if offsets != sorted(offsets):
        raise ValueError(
            "MiniMax H3 Omni Context-IR sections are out of order after "
            "reference preparation; no generation was started."
        )
    return source


def _sanitize_legacy_ref2va_context_ir(text: str) -> str:
    """Remove old Maestro control prose that H3 can mistake for speech.

    Before the native Subject/Speaker/Audio compiler, saved sequence prompts
    used repeated negative instructions such as ``never spoken narration`` and
    ``only X's mouth moves``.  Ref2VA sometimes performs that prose instead of
    treating it as metadata.  Loaded gallery settings can still contain those
    prompts, so sanitize them at the final model boundary as well as
    invalidating their planner signature.
    """

    source = _normalize_ref2va_context_ir_sections(text)
    if not (
        re.search(r"(?mi)^\s*subject_definitions\s*:", source)
        and re.search(r"(?mi)^\s*detailed_description\s*:", source)
    ):
        return source

    definitions = re.search(
        r"(?ms)^\s*subject_definitions\s*:(.*?)(?=^\s*summary\s*:)",
        source,
    )
    if definitions:
        body = definitions.group(1)
        canonical_start = re.search(
            r"<Subject\s+\d+>\s+is\b",
            body,
            flags=re.IGNORECASE,
        )
        # Old LLM output was sometimes concatenated directly before Maestro's
        # canonical reference map (``black shirt <Subject 1> is ...``).  Drop
        # only a prefix that contains no reference label of its own.
        if (
            canonical_start
            and not _REFERENCE_TAG_RE.search(body[:canonical_start.start()])
        ):
            body = body[canonical_start.start():]
            source = (
                f"{source[:definitions.start(1)]} {body.strip()}\n\n"
                f"{source[definitions.end(1):].lstrip()}"
            )

    substitutions = (
        (r"\bVisual direction only,\s*never spoken narration:\s*", ""),
        (r"\bSilent visual action,\s*never spoken narration:\s*", ""),
        (
            r",?\s*only\s+[^,.;:\r\n]{1,100}['’]s\s+mouth\s+moves\s+while\s+"
            r"every\s+other\s+visible\s+mouth\s+stays\s+closed",
            "",
        ),
        (
            r"\s*Immediately after the line,\s*the speaker closes their mouth\.",
            "",
        ),
        (
            r"\s*Only the tagged words are spoken once in order\.\s*Outside those "
            r"lines, there are no additional spoken words, muttering, or gibberish\.",
            "",
        ),
        (
            r"\s*No words are spoken or mouthed in this shot;\s*only explicitly "
            r"requested nonverbal reactions may be heard\.",
            "",
        ),
        (
            r"\b[A-Z][A-Za-z0-9 _'’.-]{0,80}\s+visibly delivers the assigned "
            r"dialogue line(?:\.\s*Then\s+|\.?)",
            "",
        ),
    )
    for pattern, replacement in substitutions:
        source = re.sub(pattern, replacement, source, flags=re.IGNORECASE)
    source = re.sub(r"[ \t]{2,}", " ", source)
    source = re.sub(r"(?m)^\s*;\s*", "", source)
    return _normalize_ref2va_context_ir_sections(source)


def _normalize_ref2va_speaker_alias(value: Any) -> str:
    alias = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    alias = re.sub(
        r"\s+(?:voice(?:\s+reference)?|audio(?:\s+reference)?|character\s+reference)\s*$",
        "",
        alias,
    )
    return alias.strip(" \t\r\n.,:;_-–—")


def _ref2va_character_visual(item: dict) -> bool:
    kind = item.get("type")
    return (
        kind == "image" and item.get("image_intent", "identity") == "identity"
    ) or (
        kind == "video" and item.get("video_intent", "motion") == "character"
    )


def _ref2va_alias_values(item: dict, fallback_role: str = "") -> list[str]:
    values: list[str] = []
    for raw in (item.get("character_name"), item.get("role"), fallback_role):
        alias = _normalize_ref2va_speaker_alias(raw)
        if alias and alias not in values:
            values.append(alias)
    return values


def _build_ref2va_character_bindings(items: list[dict]):
    """Assign character Subjects before scene/style Subjects are described.

    Saved-character media may be interleaved (or contain both an image and a
    video), but one library character must always remain one Subject.  The
    returned alias table contains only unambiguous names.
    """

    key_subjects: dict[str, int] = {}
    reference_subjects: dict[int, int] = {}
    character_subjects: dict[str, int] = {}
    alias_candidates: dict[str, set[int]] = {}

    for reference_index, item in enumerate(items):
        if not _ref2va_character_visual(item):
            continue
        library_id = str(item.get("library_character_id") or "").strip()
        aliases = _ref2va_alias_values(item)
        identity_key = (
            f"library:{library_id}"
            if library_id
            else f"name:{aliases[0]}"
            if aliases
            else f"reference:{reference_index}"
        )
        subject = key_subjects.get(identity_key)
        if subject is None:
            subject = len(key_subjects) + 1
            key_subjects[identity_key] = subject
        reference_subjects[reference_index] = subject
        if library_id:
            character_subjects[library_id] = subject
        for alias in aliases:
            alias_candidates.setdefault(alias, set()).add(subject)

    # Audio references often appear before their paired visual reference.
    # Register their names after all visual subjects have been allocated.
    for item in items:
        if item.get("type") != "audio" or item.get("audio_intent", "voice") != "voice":
            continue
        library_id = str(item.get("library_character_id") or "").strip()
        mapped_subject = character_subjects.get(library_id) if library_id else None
        if mapped_subject is None:
            for alias in _ref2va_alias_values(item):
                candidates = alias_candidates.get(alias, set())
                if len(candidates) == 1:
                    mapped_subject = next(iter(candidates))
                    break
        if mapped_subject is not None:
            for alias in _ref2va_alias_values(item):
                alias_candidates.setdefault(alias, set()).add(mapped_subject)

    speaker_aliases = {
        alias: next(iter(subjects))
        for alias, subjects in alias_candidates.items()
        if len(subjects) == 1
    }
    role_subjects = dict(speaker_aliases)
    return (
        reference_subjects,
        character_subjects,
        role_subjects,
        speaker_aliases,
        len(key_subjects),
    )


def _ref2va_audio_item_subject(
    item: dict,
    character_subjects: dict[str, int],
    role_subjects: dict[str, int],
) -> int | None:
    """Return the immutable visual Subject paired with one voice reference."""

    library_id = str(item.get("library_character_id") or "").strip()
    if library_id and library_id in character_subjects:
        return character_subjects[library_id]
    for alias in _ref2va_alias_values(item):
        subject = role_subjects.get(alias)
        if subject is not None:
            return subject
    return None


def _ref2va_voice_audio_by_subject(
    items: list[dict],
    character_subjects: dict[str, int],
    role_subjects: dict[str, int],
) -> dict[int, int]:
    """Map each stable visual Subject to its independent Audio ordinal."""

    result: dict[int, int] = {}
    audio_ordinal = 0
    for item in items:
        kind = item.get("type")
        carries_audio = kind == "audio" or (
            kind == "video"
            and (item.get("has_audio") or item.get("audio_path"))
            and item.get("include_audio", True)
        )
        if not carries_audio:
            continue
        audio_ordinal += 1
        if kind != "audio" or item.get("audio_intent", "voice") != "voice":
            continue
        subject = _ref2va_audio_item_subject(
            item,
            character_subjects,
            role_subjects,
        )
        if subject is not None:
            result.setdefault(subject, audio_ordinal)
    return result


def _ref2va_audio_ordinals_by_item(items: list[dict]) -> dict[int, int]:
    """Return the independent ``<Audio N>`` ordinal owned by each item."""

    labels: dict[int, int] = {}
    ordinal = 0
    for index, item in enumerate(items):
        kind = item.get("type")
        carries_audio = kind == "audio" or (
            kind == "video"
            and (item.get("has_audio") or item.get("audio_path"))
            and item.get("include_audio", True)
        )
        if carries_audio:
            ordinal += 1
            labels[index] = ordinal
    return labels


def canonicalize_ref2va_reference_order(
    prompt: str,
    references,
) -> tuple[str, list[dict], dict[str, dict[int, int]]]:
    """Present visual references before standalone audio references.

    MiniMax treats the packed reference order as semantic.  ComfyUI/WanGP's
    native Ref2VA input path presents every visual reference first, followed
    by standalone audio references.  Saved Maestro characters arrive as
    ``image, audio, image, audio`` pairs, which previously interleaved audio
    rows between character portraits on the shared rotary timeline.  Keep the
    each modality's user order intact, group pictures before videos and
    standalone audio, and atomically remap Audio labels when an audio-bearing
    video makes that necessary.
    """

    items = validate_reference_manifest(
        references,
        require_files=False,
        require_visual=False,
        allow_empty=True,
    )
    indexed = list(enumerate(items))
    ordered_entries = [
        entry for kind in ("image", "video", "audio")
        for entry in indexed
        if entry[1].get("type") == kind
    ]
    if [index for index, _item in ordered_entries] == list(range(len(items))):
        return str(prompt or ""), items, {}

    old_audio = _ref2va_audio_ordinals_by_item(items)
    ordered_items = [item for _index, item in ordered_entries]
    new_audio_by_original: dict[int, int] = {}
    ordinal = 0
    for original_index, item in ordered_entries:
        carries_audio = item.get("type") == "audio" or (
            item.get("type") == "video"
            and (item.get("has_audio") or item.get("audio_path"))
            and item.get("include_audio", True)
        )
        if carries_audio:
            ordinal += 1
            new_audio_by_original[original_index] = ordinal
    audio_map = {
        old_audio[original_index]: new_audio_by_original[original_index]
        for original_index in old_audio
        if old_audio[original_index] != new_audio_by_original[original_index]
    }
    ordinal_maps = {"Audio": audio_map} if audio_map else {}
    return (
        _remap_ref2va_audio_labels(prompt, audio_map),
        ordered_items,
        ordinal_maps,
    )


def _ref2va_dialogue_subject_order(
    text: str,
    speaker_aliases: dict[str, int],
    character_subject_count: int,
) -> list[int]:
    """Resolve target speakers in first-vocal-event order.

    MiniMax's documented ``Sx`` namespace is event ordered. This helper also
    accepts an already structured prompt, where ``<Subject N> (Sx)`` may be the
    only ownership cue beside a dialogue tag.
    """

    source = str(text or "")
    valid_subjects = set(range(1, character_subject_count + 1))
    events = list(_DIALOGUE_TAG_RE.finditer(source))
    if not events:
        events.extend(
            re.finditer(r'"([^"\r\n]{1,500})"|“([^”\r\n]{1,500})”', source)
        )
    events.sort(key=lambda match: match.start())

    order: list[int] = []
    for match in events:
        subject = _resolve_ref2va_dialogue_speaker(
            source,
            match.start(),
            match.end(),
            speaker_aliases,
            valid_subjects,
        )
        if subject in valid_subjects and subject not in order:
            order.append(subject)
    return order


def _remap_ref2va_audio_labels(text: str, ordinal_map: dict[int, int]) -> str:
    if not ordinal_map:
        return str(text or "")

    def replace(match: re.Match) -> str:
        raw = match.group("tag_index") or match.group("plain_index")
        old = int(raw)
        new = int(ordinal_map.get(old, old))
        return f"<Audio {new}>" if match.group("tag") else f"Audio {new}"

    return _AUDIO_LABEL_RE.sub(replace, str(text or ""))


def _strip_ref2va_audio_clauses(text: str, ordinals: set[int]) -> str:
    """Remove prompt clauses that describe audio references omitted for a window."""

    if not ordinals:
        return str(text or "")
    values = "|".join(str(value) for value in sorted(ordinals))
    label_re = re.compile(
        rf"(?:<Audio\s+(?:{values})>|\bAudio\s+(?:{values})\b)",
        re.IGNORECASE,
    )
    # Context-IR reference relationships are sentence/semicolon clauses. Work
    # at that boundary so removing an unused voice never deletes adjacent
    # character identity, action, camera, or dialogue instructions.
    parts = re.split(r"([.;](?:\s+|$)|\n+)", str(text or ""))
    output: list[str] = []
    index = 0
    while index < len(parts):
        clause = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if not label_re.search(clause):
            output.extend((clause, separator))
        elif "\n" in separator:
            # The omitted clause may terminate a Context-IR section.  Retain
            # its structural newline even though its punctuation and prose are
            # removed, otherwise the following field becomes part of the
            # preceding field's value.
            output.append("\n\n")
        index += 2
    stripped = re.sub(r"[ \t]{2,}", " ", "".join(output)).strip()
    return _normalize_ref2va_context_ir_sections(stripped)


def select_ref2va_window_voice_references(
    prompt: str,
    references,
    *,
    max_audio_references: int = 2,
) -> tuple[str, list[dict], dict[str, Any]]:
    """Limit each native Ref2VA pass to voices that speak in that window.

    A project may contain more characters than H3 can accept audio references
    for in one pass. Passing every saved voice to every continuation window
    made H3 replay source recordings and promote their paired portraits into
    target footage. Visual identity references remain stable across the whole
    sequence; voice recordings are selected in first-vocal-event order for the
    current prompt and their Audio ordinals are rebuilt atomically.
    """

    source_prompt, items, _order_remap = canonicalize_ref2va_reference_order(
        prompt,
        references,
    )
    (
        _reference_subjects,
        character_subjects,
        role_subjects,
        speaker_aliases,
        character_subject_count,
    ) = _build_ref2va_character_bindings(items)
    speaker_order = _ref2va_dialogue_subject_order(
        source_prompt,
        speaker_aliases,
        character_subject_count,
    )
    has_dialogue = bool(
        _DIALOGUE_TAG_RE.search(source_prompt)
        or re.search(r'"[^"\r\n]{1,500}"|“[^”\r\n]{1,500}”', source_prompt)
    )

    audio_entries: list[tuple[int, int, bool, int | None]] = []
    audio_ordinal = 0
    non_voice_audio_count = 0
    for item_index, item in enumerate(items):
        kind = item.get("type")
        if kind == "video":
            if (
                (item.get("has_audio") or item.get("audio_path"))
                and item.get("include_audio", True)
            ):
                audio_ordinal += 1
                non_voice_audio_count += 1
                audio_entries.append((item_index, audio_ordinal, False, None))
            continue
        if kind != "audio":
            continue
        audio_ordinal += 1
        is_voice = item.get("audio_intent", "voice") == "voice"
        mapped_subject = (
            _ref2va_audio_item_subject(item, character_subjects, role_subjects)
            if is_voice
            else None
        )
        audio_entries.append((item_index, audio_ordinal, is_voice, mapped_subject))
        if not is_voice:
            non_voice_audio_count += 1

    available_voice_slots = max(0, int(max_audio_references) - non_voice_audio_count)
    voice_entries = [entry for entry in audio_entries if entry[2]]
    selected_voice_indices: list[int] = []
    if has_dialogue and available_voice_slots:
        for subject in speaker_order:
            match = next(
                (entry for entry in voice_entries if entry[3] == subject),
                None,
            )
            if match is not None and match[0] not in selected_voice_indices:
                selected_voice_indices.append(match[0])
            if len(selected_voice_indices) >= available_voice_slots:
                break
        # Preserve the common one-character/generic-voice workflow when the
        # audio was not explicitly paired by character metadata.
        for entry in voice_entries:
            if len(selected_voice_indices) >= available_voice_slots:
                break
            if entry[3] is None and entry[0] not in selected_voice_indices:
                selected_voice_indices.append(entry[0])

    selected_voice_set = set(selected_voice_indices)
    omitted_entries = [entry for entry in voice_entries if entry[0] not in selected_voice_set]
    omitted_ordinals = {entry[1] for entry in omitted_entries}
    retained_indices = {
        index
        for index, item in enumerate(items)
        if item.get("type") != "audio"
        or item.get("audio_intent", "voice") != "voice"
        or index in selected_voice_set
    }
    scoped_items = [item for index, item in enumerate(items) if index in retained_indices]

    old_to_new: dict[int, int] = {}
    new_audio_ordinal = 0
    for item_index, old_ordinal, _is_voice, _subject in audio_entries:
        if item_index not in retained_indices:
            continue
        new_audio_ordinal += 1
        old_to_new[old_ordinal] = new_audio_ordinal

    scoped_prompt = _strip_ref2va_audio_clauses(source_prompt, omitted_ordinals)
    scoped_prompt = _remap_ref2va_audio_labels(scoped_prompt, old_to_new)
    omitted_roles = [
        str(items[index].get("character_name") or items[index].get("role") or f"Audio {ordinal}")
        for index, ordinal, _is_voice, _subject in omitted_entries
    ]
    kept_roles = [
        str(items[index].get("character_name") or items[index].get("role") or f"Audio {ordinal}")
        for index, ordinal, _is_voice, _subject in voice_entries
        if index in selected_voice_set
    ]
    diagnostics: dict[str, Any] = {
        "speaking_subjects": speaker_order,
        "voice_total": len(voice_entries),
        "voice_kept": len(selected_voice_set),
        "voice_omitted": len(omitted_entries),
        "kept_roles": kept_roles,
        "omitted_roles": omitted_roles,
        "max_audio_references": int(max_audio_references),
        "non_voice_audio_references": non_voice_audio_count,
    }
    return scoped_prompt, scoped_items, diagnostics


def _remap_ref2va_reference_labels(
    text: str,
    ordinal_maps: dict[str, dict[int, int]],
) -> str:
    """Atomically remap Subject/Picture/Video/Audio labels in one prompt."""

    source = str(text or "")
    for kind in ("Subject", "Picture", "Video"):
        mapping = ordinal_maps.get(kind, {})
        if not mapping:
            continue

        def replace(match: re.Match, *, current_kind=kind, current_map=mapping) -> str:
            old = int(match.group(1))
            return f"<{current_kind} {int(current_map.get(old, old))}>"

        source = re.sub(
            rf"<{kind}\s+(\d+)>",
            replace,
            source,
            flags=re.IGNORECASE,
        )
    return _remap_ref2va_audio_labels(source, ordinal_maps.get("Audio", {}))


def align_ref2va_voice_reference_order(
    prompt: str,
    references,
) -> tuple[str, list[dict], dict[str, dict[int, int]]]:
    """Keep immutable Subject identities while normalizing media presentation.

    ``<Subject N>`` identifies reusable reference content; ``(Sx)`` identifies
    first-vocal-event order inside one target clip.  Reordering complete
    character groups to make those two namespaces numerically match was not
    part of MiniMax's contract and made identity labels depend on dialogue
    order.  Keep Subject/Picture labels stable and only normalize the physical
    presentation to visual references followed by standalone audio.
    """

    return canonicalize_ref2va_reference_order(prompt, references)


def _ref2va_alias_occurrences(source: str, aliases: dict[str, int]):
    occurrences: list[tuple[int, int, str, int]] = []
    for alias, subject in aliases.items():
        alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
        for match in re.finditer(
            rf"(?<!\w){alias_pattern}(?!\w)",
            source,
            re.IGNORECASE,
        ):
            occurrences.append((match.start(), match.end(), alias, subject))
    return sorted(occurrences, key=lambda row: (row[0], -(row[1] - row[0])))


def _clean_ref2va_dialogue_owner_name(value: str) -> str:
    """Return a human speaker label without sentence-leading glue words."""

    words = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n,;:-")
    parts = words.split()
    while len(parts) > 1 and parts[0].casefold() in _DIALOGUE_OWNER_LEADING_WORDS:
        parts.pop(0)
    candidate = " ".join(parts).strip()
    if not candidate or candidate.casefold() in _DIALOGUE_OWNER_LEADING_WORDS:
        return ""
    return candidate


def _resolve_ref2va_dialogue_owner_name(
    source: str,
    dialogue_start: int,
    dialogue_end: int,
) -> str:
    """Resolve an explicitly named speaker, including non-reference actors.

    Ref2VA's ``(Sx)`` marker identifies first-vocal-event order. It does not
    identify ``<Subject x>``.  Detecting the grammatical speaker separately is
    what lets a saved Blaine character share a scene with unreferenced Rachel
    and Ross without either guest inheriting Blaine's portrait or voice.
    """

    before = source[max(0, dialogue_start - 420):dialogue_start]
    after = source[dialogue_end:dialogue_end + 180]
    clause_start = max(
        before.rfind("."),
        before.rfind("!"),
        before.rfind("?"),
        before.rfind(";"),
        before.rfind("\n"),
    ) + 1
    clause = before[clause_start:]

    def occurrences(value: str):
        return [
            (match.start(), match.end(), _clean_ref2va_dialogue_owner_name(match.group(1)))
            for match in _DIALOGUE_OWNER_NAME_RE.finditer(value)
            if _clean_ref2va_dialogue_owner_name(match.group(1))
        ]

    # Direct screenplay syntax: ``Rachel: <d>...</d>``.
    for start, end, name in reversed(occurrences(clause)):
        tail = clause[end:]
        leading = clause[:start]
        if (
            re.fullmatch(r"\s*(?:'s\s+voice\s*)?[:,\-–—]\s*", tail, re.IGNORECASE)
            and not _SPEECH_VERB_RE.search(leading)
        ):
            return name

    # Postposed attribution: ``<d>...</d>, replies Rachel``.
    after_verb = re.match(
        rf"\s*[,;:\-–—]?\s*({_SPEECH_VERB_RE.pattern})",
        after,
        re.IGNORECASE,
    )
    if after_verb:
        candidates = [
            (start - after_verb.end(), -(end - start), name)
            for start, end, name in occurrences(after)
            if start >= after_verb.end()
        ]
        if candidates:
            return min(candidates)[-1]

    # Natural prose: ``Blaine turns to Yoda and says, ...``. The last name
    # before the speech verb is not necessarily the speaker, so reject names
    # introduced by object prepositions such as ``to`` or ``at``.
    verbs = list(_SPEECH_VERB_RE.finditer(clause))
    if verbs:
        verb = verbs[-1]
        candidates = []
        for start, end, name in occurrences(clause):
            if end > verb.start():
                continue
            leading = clause[max(0, start - 32):start]
            is_object = bool(re.search(
                r"(?:\bto|\bat|\btoward|\btowards|\bwith|\bbeside|\bnear|\bbehind)\s+$",
                leading,
                re.IGNORECASE,
            ))
            candidates.append((is_object, verb.start() - end, -(end - start), name))
        non_objects = [candidate for candidate in candidates if not candidate[0]]
        if non_objects:
            return min(non_objects)[-1]
        if candidates:
            return min(candidates)[-1]

    # Finished Context-IR often writes ``Rachel (S1) ...`` beside the tag
    # without repeating a speech verb. This marker can confirm the nearby
    # name, but it still never maps the name to Subject 1.
    marked = list(re.finditer(
        r"([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})"
        r"\s*\(S\d+\)",
        clause,
    ))
    if marked:
        return _clean_ref2va_dialogue_owner_name(marked[-1].group(1))
    return ""


def _resolve_ref2va_dialogue_speaker(
    source: str,
    dialogue_start: int,
    dialogue_end: int,
    speaker_aliases: dict[str, int],
    valid_subjects: set[int],
) -> int | None:
    """Resolve a dialogue speaker from natural-language cues around one line."""

    before_start = max(0, dialogue_start - 360)
    before = source[before_start:dialogue_start]
    after = source[dialogue_end:dialogue_end + 180]
    clause_start = max(
        before.rfind("."),
        before.rfind("!"),
        before.rfind("?"),
        before.rfind(";"),
        before.rfind("\n"),
    ) + 1
    clause = before[clause_start:]
    explicit_subjects = re.findall(
        r"<Subject\s+(\d+)>", clause, flags=re.IGNORECASE
    )
    if explicit_subjects:
        explicit_subject = int(explicit_subjects[-1])
        if explicit_subject in valid_subjects:
            return explicit_subject
    occurrences = _ref2va_alias_occurrences(clause, speaker_aliases)

    # Direct screenplay syntax: ``Yoda: "..."`` or ``Yoda's voice: "..."``.
    # Do not mistake the addressed character in natural prose such as
    # ``Thanos says to Yoda, <d>...</d>`` for a screenplay speaker label.  A
    # preceding speech verb means the grammatical-speaker pass below owns the
    # decision, where ``to Yoda`` is correctly treated as the object.
    for start, end, _alias, subject in reversed(occurrences):
        tail = clause[end:]
        leading = clause[:start]
        if (
            re.fullmatch(r"\s*(?:'s\s+voice\s*)?[:,\-–—]\s*", tail, re.IGNORECASE)
            and not _SPEECH_VERB_RE.search(leading)
        ):
            return subject

    # Postposed attribution is highly specific and takes precedence over an
    # unrelated speech verb in the preceding sentence.
    after_verb = re.match(
        rf"\s*[,;:\-–—]?\s*({_SPEECH_VERB_RE.pattern})",
        after,
        re.IGNORECASE,
    )
    if after_verb:
        after_occurrences = _ref2va_alias_occurrences(after, speaker_aliases)
        candidates = [
            (start - after_verb.end(), -len(alias), subject)
            for start, _end, alias, subject in after_occurrences
            if start >= after_verb.end()
        ]
        if candidates:
            return min(candidates)[-1]

    # Natural syntax before the line: ``Blaine turns to Yoda and says, ...``.
    # Select the last non-object character before the final speech verb.
    verbs = list(_SPEECH_VERB_RE.finditer(clause))
    if verbs:
        verb = verbs[-1]
        candidates = []
        for start, end, alias, subject in occurrences:
            if end > verb.start():
                continue
            leading = clause[max(0, start - 28):start]
            is_object = bool(re.search(
                r"(?:\bto|\bat|\btoward|\btowards|\bwith|\bbeside|\bnear|\bbehind)\s+$",
                leading,
                re.IGNORECASE,
            ))
            candidates.append((is_object, verb.start() - end, -len(alias), subject))
        non_objects = [candidate for candidate in candidates if not candidate[0]]
        if non_objects:
            return min(non_objects)[-1]
        if candidates:
            return min(candidates)[-1]

    # A manually authored Context-IR prompt may put the <d> tag in the sentence
    # after the named performance cue, for example: ``Yoda nods. He answers.
    # <d>...</d>``. Follow that short discourse chain, but only when the latest
    # named sentence has one unambiguous grammatical subject. This is purposely
    # conservative: ``Yoda and Blaine react. They answer.`` remains ambiguous.
    preceding_dialogue_end = 0
    for previous_match in _DIALOGUE_TAG_RE.finditer(source, 0, dialogue_start):
        preceding_dialogue_end = previous_match.end()
    discourse_start = max(preceding_dialogue_end, dialogue_start - 720)
    # Context-IR fields are independent contracts. Never reach backward from
    # detailed_description into subject_definitions/retention_analysis and use
    # a saved character named there as the grammatical speaker of a guest's
    # line. The latest field header is the hard discourse boundary.
    field_headers = list(re.finditer(
        r"(?im)^\s*(?:subject_definitions|summary|retention_analysis|"
        r"detailed_description|overall_soundscape|non_diegetic_music)\s*:\s*",
        source[discourse_start:dialogue_start],
    ))
    if field_headers:
        discourse_start += field_headers[-1].end()
    discourse = source[discourse_start:dialogue_start]
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?;])\s+|[\r\n]+", discourse)
        if segment.strip()
    ]
    for segment in reversed(segments):
        segment_occurrences = _ref2va_alias_occurrences(segment, speaker_aliases)
        if not segment_occurrences:
            continue
        candidate_subjects: set[int] = set()
        for start, _end, _alias, candidate_subject in segment_occurrences:
            leading = segment[max(0, start - 32):start]
            is_object = bool(re.search(
                r"(?:\bto|\bat|\btoward|\btowards|\bwith|\bbeside|\bnear|"
                r"\bbehind|\bfrom|\bfor|\bof|\bby)\s+$",
                leading,
                re.IGNORECASE,
            ))
            if not is_object:
                candidate_subjects.add(candidate_subject)
        if len(candidate_subjects) == 1:
            return next(iter(candidate_subjects))
        # Do not reach past a more recent sentence that names multiple possible
        # speakers. Guessing here would recreate the original voice-swap bug.
        return None
    return None


def _ambiguous_ref2va_dialogue_error(words: str) -> ValueError:
    excerpt = re.sub(r"\s+", " ", words).strip()[:80]
    return ValueError(
        "MiniMax H3 Omni could not determine which referenced character speaks "
        f"{excerpt!r}. Name the speaker beside the line (for example, Yoda says, "
        '"Do or do not.") or place that character\'s explicit <Subject N> tag '
        "beside the line. (Sx) labels only identify vocal-event order."
    )


def _canonicalize_ref2va_tagged_dialogue(
    text: str,
    speaker_aliases: dict[str, int],
    character_subject_count: int,
    audio_by_subject: dict[int, int] | None = None,
) -> str:
    """Repair named H3 dialogue using official event-ordered speaker IDs.

    ``<Subject N>`` numbers identify reusable reference content.  MiniMax's
    ``(Sx)`` numbers are a separate namespace assigned in order of the first
    actual vocal event.  Treating Subject 1 as Speaker 1 caused a later
    character who spoke first to inherit or repeat another character's line.
    """

    valid_subjects = set(range(1, character_subject_count + 1))
    audio_by_subject = dict(audio_by_subject or {})
    matches = list(_DIALOGUE_TAG_RE.finditer(text))
    if not matches:
        return text
    edits: list[tuple[int, int, str]] = []
    vocal_speakers: dict[tuple[str, Any], int] = {}
    previous_dialogue_end = 0
    for match in matches:
        words = match.group(1).strip()
        subject = _resolve_ref2va_dialogue_speaker(
            text,
            match.start(),
            match.end(),
            speaker_aliases,
            valid_subjects,
        )
        explicit_owner = _resolve_ref2va_dialogue_owner_name(
            text,
            match.start(),
            match.end(),
        )
        context_start = max(previous_dialogue_end, match.start() - 240)
        markers = list(_SPEAKER_MARKER_RE.finditer(text, context_start, match.start()))
        marker = markers[-1] if markers else None
        # ``(Sx)`` is vocal-event order, not a Subject identifier.  It can
        # validate a resolved speaker but must never select a face by itself.
        # The old fallback silently mapped S1 to Subject 1, which is exactly
        # how a correct voice could be lip-synced by the wrong character.
        if subject is None and character_subject_count == 1 and not explicit_owner:
            subject = 1
        if subject is None and character_subject_count > 1 and not explicit_owner:
            raise _ambiguous_ref2va_dialogue_error(words)
        speaker_key: tuple[str, Any] = (
            ("subject", subject)
            if subject is not None else
            ("name", explicit_owner.casefold())
            if explicit_owner else
            ("event", len(vocal_speakers) + 1)
        )
        speaker = vocal_speakers.setdefault(speaker_key, len(vocal_speakers) + 1)
        if subject is not None or explicit_owner:
            owner_context = text[max(previous_dialogue_end, match.start() - 180):match.start()]
            owner_clause_start = max(
                owner_context.rfind("."),
                owner_context.rfind("!"),
                owner_context.rfind("?"),
                owner_context.rfind(";"),
                owner_context.rfind("\n"),
            ) + 1
            nearby_subject = bool(
                subject is not None
                and re.search(
                    rf"<Subject\s+{subject}>",
                    owner_context[owner_clause_start:],
                    flags=re.IGNORECASE,
                )
            )
            owner = (
                "" if subject is None or nearby_subject else f"<Subject {subject}> "
            )
            nearby_audio = bool(
                subject is not None
                and re.search(
                    rf"<Audio\s+{audio_by_subject.get(subject, -1)}>",
                    owner_context[owner_clause_start:],
                    flags=re.IGNORECASE,
                )
            )
            voice = (
                f" in the voice referenced from <Audio {audio_by_subject[subject]}>,"
                if subject is not None and subject in audio_by_subject and not nearby_audio
                else ""
            )
            if marker is not None:
                if int(marker.group(1)) != speaker:
                    edits.append((marker.start(), marker.end(), f"{owner}(S{speaker}){voice}"))
                elif owner:
                    edits.append((marker.start(), marker.end(), f"{owner}(S{speaker}){voice}"))
                elif voice:
                    edits.append((marker.start(), marker.end(), f"(S{speaker}){voice}"))
            else:
                direct = f"{owner}(S{speaker})"
                if voice:
                    direct += voice
                edits.append((match.start(), match.start(), f"{direct} "))
        previous_dialogue_end = match.end()

    for start, end, replacement in reversed(edits):
        text = f"{text[:start]}{replacement}{text[end:]}"

    # Subject definitions describe reusable visual identity only.  Speaker
    # IDs belong beside actual dialogue events and may reset per window.
    definitions = re.search(
        r"(?ms)^\s*subject_definitions\s*:(.*?)(?=^\s*summary\s*:)",
        text,
    )
    if definitions:
        body = re.sub(
            r"(<Subject\s+\d+>)\s*\(S\d+\)(?=\s+is\b)",
            r"\1",
            definitions.group(1),
            flags=re.IGNORECASE,
        )
        for (speaker_kind, subject), speaker in vocal_speakers.items():
            if speaker_kind != "subject":
                continue
            body = re.sub(
                rf"(<Audio\s+\d+>[^.\r\n]{{0,260}}?"
                rf"(?:for|to|of)\s+<Subject\s+{subject}>)"
                r"(?:\s*\(S\d+\))?",
                rf"\1 (S{speaker})",
                body,
                flags=re.IGNORECASE,
            )
        text = (
            f"{text[:definitions.start(1)]}{body}{text[definitions.end(1):]}"
        )
    return text


def _ensure_ref2va_voice_acoustic_contract(text: str) -> str:
    """Keep voice identity while explicitly rejecting source-room acoustics."""

    if _REF2VA_VOICE_ACOUSTIC_CONTRACT.casefold() in str(text or "").casefold():
        return text
    source = str(text or "").rstrip()
    music = re.search(r"(?mi)^\s*non_diegetic_music\s*:", source)
    if music:
        insert_at = music.start()
        return (
            f"{source[:insert_at].rstrip()} {_REF2VA_VOICE_ACOUSTIC_CONTRACT}\n\n"
            f"{source[insert_at:].lstrip()}"
        ).strip()
    return f"{source} {_REF2VA_VOICE_ACOUSTIC_CONTRACT}".strip()


def _ensure_ref2va_identity_isolation_contract(text: str, items: list[dict]) -> str:
    """Prevent identity media from becoming a keyframe, insert, or cutaway."""

    if not any(_ref2va_character_visual(item) for item in items):
        return str(text or "")
    if _REF2VA_IDENTITY_ISOLATION_CONTRACT.casefold() in str(text or "").casefold():
        return str(text or "")
    source = str(text or "").rstrip()
    soundscape = re.search(r"(?mi)^\s*overall_soundscape\s*:", source)
    if soundscape:
        insert_at = soundscape.start()
        return (
            f"{source[:insert_at].rstrip()} {_REF2VA_IDENTITY_ISOLATION_CONTRACT}\n\n"
            f"{source[insert_at:].lstrip()}"
        ).strip()
    return f"{source} {_REF2VA_IDENTITY_ISOLATION_CONTRACT}".strip()


def _apply_ref2va_media_contracts(text: str, items: list[dict]) -> str:
    compiled = _ensure_ref2va_identity_isolation_contract(text, items)
    if any(
        item.get("type") == "audio"
        and item.get("audio_intent", "voice") == "voice"
        for item in items
    ):
        compiled = _ensure_ref2va_voice_acoustic_contract(compiled)
    return compiled


@dataclass
class MiniMaxH3PreparedReference:
    """One prepared Ref2VA reference, kept in request order."""

    kind: str
    has_audio: bool = False
    image: Any = None
    frames: Any = None
    waveform: torch.Tensor | None = None
    block_timestamps: list[float] = field(default_factory=list)
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0
    role: str = ""
    audio_intent: str = ""
    image_intent: str = ""

    @property
    def num_video_rows(self) -> int:
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS


def ensure_ref2va_prompt_relationships(
    prompt: str,
    references,
    *,
    duration_seconds: float | None = None,
) -> str:
    """Compile a raw Ref2VA request into explicit six-field Context-IR.

    MiniMax H3 uses natural-language Context-IR to decide whether audio is
    copied/reused or merely referenced. Media tensors alone cannot communicate
    that distinction, so a raw Studio prompt receives a complete relationship
    map and literal dialogue tags. An already enhanced/tagged prompt keeps its
    creative content, while its dialogue speaker markers are checked against
    the same immutable character manifest used for raw prompts.
    """

    ordered_prompt, items, _order_remap = canonicalize_ref2va_reference_order(
        prompt,
        references,
    )
    text = _normalize_ref2va_context_ir_sections(ordered_prompt)
    (
        character_reference_subjects,
        character_subjects,
        role_subjects,
        speaker_aliases,
        character_subject_count,
    ) = _build_ref2va_character_bindings(items)
    voice_audio_by_subject = _ref2va_voice_audio_by_subject(
        items,
        character_subjects,
        role_subjects,
    )
    if _REFERENCE_TAG_RE.search(text):
        text = _sanitize_legacy_ref2va_context_ir(text)
        compiled = _canonicalize_ref2va_tagged_dialogue(
            text,
            speaker_aliases,
            character_subject_count,
            voice_audio_by_subject,
        )
        return _validate_ref2va_context_ir_sections(
            _apply_ref2va_media_contracts(compiled, items)
        )

    picture_index = 0
    video_index = 0
    audio_index = 0
    subject_index = character_subject_count
    relationships: list[str] = []
    retention: list[str] = []

    for reference_index, item in enumerate(items):
        kind = item["type"]
        role = item.get("role") or f"the supplied {kind} reference"
        if kind == "image":
            picture_index += 1
            intent = item.get("image_intent", "identity")
            if intent == "composition":
                relationships.append(
                    f"<Picture {picture_index}> is a soft composition and cast-layout reference for {role} "
                    "that preserves the intended subjects, wardrobe, setting, and spatial arrangement "
                    "while generating a naturally moving opening rather than copying the picture as a "
                    "frozen first frame."
                )
                retention.append(
                    f"<Picture {picture_index}> ([Shot 1] composition anchor): weak_reference - "
                    "retain broad subject placement, setting, and spatial relationships."
                )
            elif intent == "scene":
                subject_index += 1
                relationships.append(
                    f"<Subject {subject_index}> is the environment and location for {role}, defined by "
                    f"<Picture {picture_index}>; its architecture, materials, lighting context, and scene "
                    "identity apply without treating incidental people as target character identities."
                )
                retention.append(
                    f"<Subject {subject_index}> (appears in [Shot 1]): fully_preserved - preserve the "
                    "referenced architecture, materials, lighting context, and location identity."
                )
            elif intent == "style":
                subject_index += 1
                relationships.append(
                    f"<Subject {subject_index}> is the visual treatment for {role}, guided by "
                    f"<Picture {picture_index}>; use its medium, palette, lighting language, and texture "
                    "without copying its people, pose, framing, or exact composition."
                )
                retention.append(
                    f"<Subject {subject_index}>: weak_reference - retain broad similarity in medium, "
                    "palette, lighting language, and texture."
                )
            else:
                character_subject = character_reference_subjects[reference_index]
                relationships.append(
                    f"<Subject {character_subject}> is {role} from <Picture {picture_index}>, preserving "
                    "visual identity and appearance; the source background, framing, composition, and "
                    "pose do not define the target scene."
                )
                retention.append(
                    f"<Subject {character_subject}> (appears in [Shot 1]): fully_preserved - preserve the "
                    f"identity and appearance defined by <Picture {picture_index}>."
                )
            continue

        if kind == "video":
            next_video_index = video_index + 1
            has_soundtrack = bool(item.get("has_audio") or item.get("audio_path"))
            if has_soundtrack and item.get("include_audio", True):
                audio_index += 1
                relationships.append(
                    f"<Audio {audio_index}> is the synchronized soundtrack paired with "
                    f"<Video {next_video_index}>; reuse its audible timeline and synchronize "
                    "visible action and lip movement to it."
                )
                retention.append(
                    f"<Audio {audio_index}>: partially_copy - reuse the enabled soundtrack timeline "
                    "while allowing synchronized scene effects."
                )
            video_index = next_video_index
            video_intent = item.get("video_intent", "motion")
            if video_intent == "character":
                character_subject = character_reference_subjects[reference_index]
                relationships.append(
                    f"<Subject {character_subject}> is {role} from <Video {video_index}>, preserving "
                    "identity, appearance, and characteristic motion while using the newly described "
                    "target scene and action."
                )
                retention.append(
                    f"<Subject {character_subject}> (appears in [Shot 1]): fully_preserved - preserve the "
                    f"identity and appearance defined by <Video {video_index}> while generating the "
                    "requested target action and setting."
                )
            elif video_intent == "scene":
                relationships.append(
                    f"<Video {video_index}> provides environment, lighting, and scene continuity for {role}; "
                    "do not copy incidental people as target character identities."
                )
                retention.append(
                    f"<Video {video_index}>: partially_preserved - retain the requested environment, "
                    "lighting, and scene continuity."
                )
            else:
                relationships.append(
                    f"<Video {video_index}> provides motion, camera, scene, and temporal reference for {role}."
                )
                retention.append(
                    f"<Video {video_index}>: partially_preserved - retain the requested motion, camera, "
                    "scene, and temporal structure."
                )
            continue

        audio_index += 1
        intent = item.get("audio_intent", "voice")
        if intent == "drive":
            relationships.append(
                f"<Audio {audio_index}> is the performance-driving audio timeline for {role} "
                "and supplies the audible content synchronized to visible action and lip movement."
            )
            retention.append(
                f"<Audio {audio_index}>: partially_copy - reuse its audible content and timeline "
                "while allowing synchronized scene effects."
            )
        elif intent == "style":
            relationships.append(
                f"<Audio {audio_index}> is an audio style, rhythm, and texture reference for {role} "
                "without copying its waveform, source words, or exact timing."
            )
            retention.append(
                f"<Audio {audio_index}>: weak_reference - retain broad similarity in sound, rhythm, "
                "texture, or music style."
            )
        else:
            character_key = str(item.get("library_character_id") or "").strip()
            mapped_subject = character_subjects.get(character_key) if character_key else None
            if mapped_subject is None:
                for alias in _ref2va_alias_values(item, role):
                    mapped_subject = role_subjects.get(alias)
                    if mapped_subject is not None:
                        break
            target = (
                f"<Subject {mapped_subject}>"
                if mapped_subject else str(role)
            )
            relationships.append(
                f"<Audio {audio_index}> is the voice-timbre reference for {target}, guiding vocal "
                "identity, emotion, and delivery for newly scripted dialogue without reusing the "
                "recording's words or timing."
            )
            retention.append(
                f"<Audio {audio_index}>: reference - use its voice timbre, emotion, and delivery "
                "without copying the source signal, words, timing, or recording-room acoustics."
            )

    dialogue_counter = 0
    dialogue_word_count = 0
    def is_visible_text_quote(match) -> bool:
        before = text[max(0, match.start() - 150):match.start()]
        after = text[match.end():match.end() + 100]
        if re.search(
            r"(?i)\b(?:titled|entitled|called|named|captioned)\s*[:,-]?\s*$",
            before,
        ):
            return True
        visible_noun = re.search(
            r"(?i)\b(?:sign|banner|label|subtitle|caption|marquee|poster|billboard|"
            r"screen|monitor|display|neon|placard|headline|logo|shirt|door|wall)\b",
            before,
        )
        visible_cue = re.search(
            r"(?i)\b(?:reads?|reading|shows?|showing|displays?|displaying|bears?|"
            r"bearing|marked|printed|written|spells?|saying|with(?:\s+the)?\s+"
            r"(?:text|words?|lettering))\s*[:,-]?\s*$",
            before,
        )
        if visible_noun and visible_cue:
            return True
        return bool(re.match(
            r"(?i)^\s*(?:appears?|is\s+(?:visible|written|printed|displayed)|glows?)"
            r"\b[^.!?\r\n]{0,70}\b(?:on|across|above|below|behind|over)\b",
            after,
        ))

    def compile_dialogue(match):
        nonlocal dialogue_counter, dialogue_word_count
        if is_visible_text_quote(match):
            return match.group(0)
        dialogue_counter += 1
        words = (match.group(1) or match.group(2) or "").strip()
        dialogue_word_count += len(words.split())
        # Speaker/Subject ownership is compiled in one place below. Inserting
        # ``(S{subject})`` here used the immutable Subject number as if it were
        # event order and could silently bind an unreferenced actor to the only
        # saved character.
        return f"<d>[English] {words}</d>"

    compiled_target = re.sub(
        r'"([^"\r\n]{1,500})"|“([^”\r\n]{1,500})”',
        compile_dialogue,
        text,
    )
    compiled_target = _canonicalize_ref2va_tagged_dialogue(
        compiled_target,
        speaker_aliases,
        character_subject_count,
        voice_audio_by_subject,
    )
    tagged_dialogue = list(_DIALOGUE_TAG_RE.finditer(compiled_target))
    dialogue_counter = len(tagged_dialogue)
    dialogue_word_count = sum(
        len(re.sub(r"^\s*\[[^]]+\]\s*", "", match.group(1)).split())
        for match in tagged_dialogue
    )
    relationship_block = " ".join(relationships)
    retention_block = " ".join(retention)
    if dialogue_counter:
        duration = max(2.0, float(duration_seconds or 8.0))
        dialogue_rule = (
            f"The tagged lines are performed once in source order and fit naturally within the "
            f"{duration:.2f}-second scene."
        )
    else:
        dialogue_rule = "The described performance remains nonverbal."
    has_mapped_music = any(item.get("audio_intent") in {"drive", "style"} for item in items)
    requests_music = bool(re.search(r"\b(?:music|song|score|soundtrack)\b", text, re.IGNORECASE))
    music_direction = (
        "Use only the mapped audio reference according to its assigned retention role."
        if has_mapped_music
        else "Follow only the music explicitly requested in the target description."
        if requests_music
        else "N/A"
    )
    task_types = ["reference generation"]
    if any(
        item.get("audio_intent") == "drive"
        or (
            item.get("type") == "video"
            and (item.get("has_audio") or item.get("audio_path"))
            and item.get("include_audio", True)
        )
        for item in items
    ):
        task_types.append("audio reuse")
    if any(
        item.get("type") == "audio"
        and item.get("audio_intent", "voice") in {"voice", "style"}
        for item in items
    ):
        task_types.append("audio reference")
    compiled_prompt = (
        f"subject_definitions: {relationship_block}\n\n"
        f"summary: [{' + '.join(task_types)}] A finished video matching the requested action, "
        "identity, setting, reference roles, and explicitly tagged dialogue.\n\n"
        f"retention_analysis: {retention_block}\n\n"
        "detailed_description: The target video maintains the requested visual style, lighting, "
        "color, and cinematic texture. "
        f"[Shot 1] "
        f"{compiled_target} {dialogue_rule}\n\n"
        "overall_soundscape: Scene-appropriate stereo ambience and synchronized practical effects "
        "accompany the visible action and scripted dialogue.\n\n"
        f"non_diegetic_music: {music_direction}"
    )
    compiled_prompt = _canonicalize_ref2va_tagged_dialogue(
        compiled_prompt,
        speaker_aliases,
        character_subject_count,
        voice_audio_by_subject,
    )
    return _apply_ref2va_media_contracts(compiled_prompt, items)


def add_ref2va_continuation_context(
    prompt: str,
    *,
    picture_offset: int = 1,
) -> str:
    """Reserve leading Picture labels for native Ref2VA continuation frames.

    Ref2VA presents a carried boundary frame to Qwen before the user's
    canonical references, exactly as upstream does for start/end conditions.
    The boundary therefore becomes ``<Picture 1>`` and only the user's
    Picture labels shift; Video and Audio numbering is unaffected. The
    transformer still receives the boundary as an exact keyframe condition,
    not as a general identity reference.
    """

    offset = max(0, int(picture_offset or 0))
    normalized = str(prompt or "").strip()
    if not normalized or offset <= 0:
        return normalized

    shifted = _PICTURE_TAG_RE.sub(
        lambda match: f"<Picture {int(match.group(1)) + offset}>",
        normalized,
    )
    continuity = (
        "<Picture 1> is the exact final frame carried from the preceding "
        "window. Use it only as the opening composition and motion boundary; "
        "continue forward without restaging it or treating it as a new "
        "identity reference."
    )
    retention = "<Picture 1>: exact opening-boundary continuity"

    subject_pattern = re.compile(
        r"(^\s*subject_definitions\s*:\s*)",
        re.IGNORECASE | re.MULTILINE,
    )
    if subject_pattern.search(shifted):
        shifted = subject_pattern.sub(
            lambda match: f"{match.group(1)}{continuity} ",
            shifted,
            count=1,
        )
    else:
        shifted = f"subject_definitions: {continuity}\n\n{shifted}"

    retention_pattern = re.compile(
        r"(^\s*retention_analysis\s*:\s*)",
        re.IGNORECASE | re.MULTILINE,
    )
    if retention_pattern.search(shifted):
        shifted = retention_pattern.sub(
            lambda match: f"{match.group(1)}{retention}; ",
            shifted,
            count=1,
        )
    else:
        shifted = f"{shifted}\n\nretention_analysis: {retention}"
    return shifted


def _decode_audio_stream(av, container, stream) -> tuple[torch.Tensor, int]:
    sample_rate = int(stream.codec_context.sample_rate)
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        chunks.extend(torch.from_numpy(item.to_ndarray()) for item in resampler.resample(frame))
    chunks.extend(torch.from_numpy(item.to_ndarray()) for item in resampler.resample(None))
    if not chunks:
        raise ValueError("The selected audio stream contains no decodable samples.")
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def decode_reference_video(path: str, *, decode_audio: bool = True):
    import av

    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream was found in {path}.")
            stream = container.streams.video[0]
            frames = []
            rotation = 0.0
            for frame in container.decode(stream):
                rotation = float(getattr(frame, "rotation", 0.0) or 0.0)
                frames.append(frame.to_ndarray(format="rgb24"))
            frame_rate = float(stream.average_rate or stream.guessed_rate or 0)
            soundtrack = None
            if decode_audio and container.streams.audio:
                container.seek(0)
                soundtrack = _decode_audio_stream(av, container, container.streams.audio[0])
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not decode reference video {os.path.basename(path)}: {error}") from error
    if not frames:
        raise ValueError(f"No video frames were found in {path}.")
    if frame_rate <= 0:
        raise ValueError(f"Reference video {os.path.basename(path)} has no valid frame rate.")
    pixels = np.stack(frames)
    turns = round(rotation / 90.0) % 4
    if turns:
        pixels = np.ascontiguousarray(np.rot90(pixels, k=-turns, axes=(1, 2)))
    return pixels, frame_rate, soundtrack


def decode_reference_audio(path: str) -> tuple[torch.Tensor, int]:
    import av

    try:
        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError(f"No audio stream was found in {path}.")
            return _decode_audio_stream(av, container, container.streams.audio[0])
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not decode reference audio {os.path.basename(path)}: {error}") from error


def reference_media_to_uint8(media) -> np.ndarray:
    if isinstance(media, list):
        return np.stack([reference_media_to_uint8(item) for item in media])
    if isinstance(media, Image.Image):
        return np.asarray(media.convert("RGB"))
    if isinstance(media, torch.Tensor):
        media = media.movedim(-3, -1).cpu().numpy()
    media = np.asarray(media)
    if media.dtype != np.uint8:
        media = (media * 255.0).round().clip(0, 255).astype(np.uint8)
    return media


def resolve_reference_image_size(
    width: int,
    height: int,
    *,
    detail: str = "match",
    target_height: int | None = None,
    target_width: int | None = None,
) -> tuple[int, int]:
    """Resolve official maximum detail or Maestro's consumer-friendly match size."""

    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    if detail == "max":
        scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    elif detail == "match":
        if not target_height or not target_width:
            raise ValueError("Matched reference detail needs the target height and width.")
        scale = min(1.0, math.sqrt((target_height * target_width) / float(height * width)))
    else:
        raise ValueError("Reference detail must be 'match' or 'max'.")
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def prepare_reference_image(image: Image.Image, height: int, width: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _get_reference_background_session():
    """Load Maestro's existing U2Net remover once, explicitly on CPU.

    H3 references are prepared after the diffusion model has been loaded.  A
    default ONNX Runtime CUDA provider would therefore consume generation
    VRAM and can also be binary-incompatible with PyTorch's CUDA runtime.  The
    short device override mirrors Recast's proven CPU-only session setup.
    """

    global _REFERENCE_BACKGROUND_SESSION
    if _REFERENCE_BACKGROUND_SESSION is not None:
        return _REFERENCE_BACKGROUND_SESSION
    with _REFERENCE_BACKGROUND_SESSION_LOCK:
        if _REFERENCE_BACKGROUND_SESSION is None:
            model_home = os.path.abspath(os.path.join("ckpts", "rembg"))
            os.makedirs(model_home, exist_ok=True)
            os.environ.setdefault("U2NET_HOME", model_home)
            import onnxruntime as ort
            from rembg import new_session

            original_get_device = ort.get_device
            try:
                ort.get_device = lambda: "CPU"
                _REFERENCE_BACKGROUND_SESSION = new_session("u2net")
            finally:
                ort.get_device = original_get_device
    return _REFERENCE_BACKGROUND_SESSION


def _composite_authored_alpha_on_white(image: Image.Image) -> Image.Image | None:
    """Preserve a user's authored cutout instead of re-segmenting it."""

    if image.mode not in {"RGBA", "LA"} and "transparency" not in image.info:
        return None
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    minimum, _maximum = alpha.getextrema()
    if minimum >= 255:
        return None
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


@lru_cache(maxsize=32)
def _isolate_reference_image_background_cached(
    normalized_path: str,
    modified_ns: int,
    file_size: int,
) -> Image.Image:
    """Return a cached white-background identity portrait.

    The stat values intentionally participate in the cache key so replacing a
    file at the same path cannot reuse an obsolete cutout.
    """

    del modified_ns, file_size
    with Image.open(normalized_path) as source:
        oriented = ImageOps.exif_transpose(source)
        authored_cutout = _composite_authored_alpha_on_white(oriented)
        if authored_cutout is not None:
            return authored_cutout
        source_rgb = oriented.convert("RGB")

    from rembg import remove

    with _REFERENCE_BACKGROUND_RUN_LOCK:
        cutout = remove(
            source_rgb,
            session=_get_reference_background_session(),
            alpha_matting=True,
            alpha_matting_erode_size=1,
            bgcolor=(255, 255, 255, 0),
        )
    if not isinstance(cutout, Image.Image):
        raise ValueError("U2Net returned an unsupported character cutout.")
    print(
        "[MiniMax H3 Ref2VA] Isolated character background: "
        f"{os.path.basename(normalized_path)}."
    )
    return cutout.convert("RGB")


def isolate_reference_image_background(path: str) -> Image.Image:
    """Remove one identity portrait's source background without changing it."""

    normalized_path = os.path.normcase(os.path.realpath(path))
    stat = os.stat(normalized_path)
    return _isolate_reference_image_background_cached(
        normalized_path,
        int(stat.st_mtime_ns),
        int(stat.st_size),
    ).copy()


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    if fps == MINIMAX_H3_FPS:
        return frames
    scale = MINIMAX_H3_FPS / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5))
    return np.repeat(frames, repeats, axis=0)


def resolve_reference_video_size(
    width: int,
    height: int,
    *,
    detail: str = "match",
    target_height: int | None = None,
    target_width: int | None = None,
) -> tuple[int, int]:
    """Resolve Ref2VA video detail without silently exceeding the output area.

    The official high-detail path keeps MiniMax's 768px-short-edge canvas.
    Maestro's default ``match`` path instead preserves the reference aspect
    ratio while bounding its pixel area to the requested output.  Reference
    video rows share the transformer's attention sequence with the generated
    clip, so decoding a 480/544p job's reference at 768p can more than double
    the denoising working set and exhaust a 24 GB card.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"A reference video must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference video must be within 1:4 and 4:1, got {width}x{height}.")
    if detail == "max":
        return resolve_canvas_size(width, height)
    if detail != "match":
        raise ValueError("Reference detail must be 'match' or 'max'.")
    if not target_height or not target_width:
        raise ValueError("Matched reference detail needs the target height and width.")

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    scale = min(1.0, math.sqrt((target_height * target_width) / float(height * width)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def prepare_reference_frames(
    frames: np.ndarray,
    num_frames: int,
    *,
    detail: str = "max",
    target_height: int | None = None,
    target_width: int | None = None,
) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"A reference video must contain RGB frames, got {tuple(frames.shape)}.")
    frames = frames[:num_frames]
    height, width = resolve_reference_video_size(
        frames.shape[2],
        frames.shape[1],
        detail=detail,
        target_height=target_height,
        target_width=target_width,
    )
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def prepare_reference_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
    max_duration: float,
    *,
    start_time: float = 0.0,
    pad_to_duration: bool = False,
) -> torch.Tensor:
    waveform = torch.as_tensor(waveform, device=torch.device("cpu"))
    if waveform.ndim != 2 or waveform.shape[0] not in (1, MINIMAX_H3_AUDIO_CHANNELS):
        raise ValueError(
            "A reference soundtrack must be a mono or stereo (channels, samples) waveform, "
            f"got {tuple(waveform.shape)}."
        )
    if sample_rate <= 0:
        raise ValueError(f"A reference soundtrack must have a positive sample rate, got {sample_rate}.")
    duration_samples = max(1, int(round(max_duration * sample_rate)))
    start_sample = max(0, int(round(max(0.0, float(start_time)) * sample_rate)))
    waveform = waveform.to(torch.float32)[:, start_sample : start_sample + duration_samples]
    if pad_to_duration and waveform.shape[-1] < duration_samples:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, duration_samples - waveform.shape[-1]),
        )
    if waveform.shape[0] == 1:
        waveform = waveform.expand(MINIMAX_H3_AUDIO_CHANNELS, -1).contiguous()
    if sample_rate != target_sample_rate:
        import torchaudio

        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
    if pad_to_duration:
        target_samples = max(1, int(round(max_duration * target_sample_rate)))
        waveform = waveform[:, :target_samples]
        if waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, target_samples - waveform.shape[-1]),
            )
    return waveform.contiguous()


def prepare_references(
    manifest,
    *,
    num_frames: int,
    target_height: int,
    target_width: int,
    audio_sample_rate: int = 32000,
    detail: str = "match",
    timeline_start_frame: int = 0,
) -> list[MiniMaxH3PreparedReference]:
    """Decode and prepare every reference without changing target geometry."""

    _prompt, ordered_items, _order_remap = canonicalize_ref2va_reference_order(
        "",
        manifest,
    )
    items = validate_reference_manifest(ordered_items, require_files=True)
    max_duration = num_frames / MINIMAX_H3_FPS
    timeline_start_frame = max(0, int(timeline_start_frame or 0))
    timeline_start_time = timeline_start_frame / MINIMAX_H3_FPS
    prepared: list[MiniMaxH3PreparedReference] = []

    for item in items:
        kind = item["type"]
        reference = MiniMaxH3PreparedReference(
            kind=kind,
            role=item.get("role", ""),
            audio_intent=item.get("audio_intent", ""),
            image_intent=item.get("image_intent", ""),
        )

        if kind == "image":
            if item.get("remove_background"):
                try:
                    image = isolate_reference_image_background(item["path"])
                except Exception as error:
                    # Background isolation improves identity cleanliness but
                    # must never turn a valid reference into a failed render.
                    print(
                        "[MiniMax H3 Ref2VA] Character background isolation "
                        f"failed for {os.path.basename(item['path'])}; using the "
                        f"original image ({error})."
                    )
                    with Image.open(item["path"]) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
            else:
                with Image.open(item["path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
            height, width = resolve_reference_image_size(
                *image.size,
                detail=detail,
                target_height=target_height,
                target_width=target_width,
            )
            reference.image = prepare_reference_image(image, height, width).copy()
        elif kind == "video":
            wants_embedded_audio = bool(item.get("include_audio", True)) and not item.get("audio_path")
            if item.get("has_audio") is False:
                wants_embedded_audio = False
            frames, fps, soundtrack = decode_reference_video(item["path"], decode_audio=wants_embedded_audio)
            frames = resample_reference_frames(reference_media_to_uint8(frames), fps)
            source_height, source_width = frames.shape[1:3]
            reference.frames = prepare_reference_frames(
                frames,
                num_frames,
                detail=detail,
                target_height=target_height,
                target_width=target_width,
            )
            prepared_height, prepared_width = reference.frames.shape[1:3]
            print(
                "[MiniMax H3 Ref2VA] Prepared reference video "
                f"{source_width}x{source_height} -> {prepared_width}x{prepared_height} "
                f"({reference.frames.shape[0]} frames, detail={detail})."
            )
            if item.get("include_audio", True):
                if item.get("audio_path"):
                    soundtrack = decode_reference_audio(item["audio_path"])
                if soundtrack is not None:
                    waveform, sample_rate = soundtrack
                    reference.waveform = prepare_reference_waveform(
                        waveform, sample_rate, audio_sample_rate, max_duration
                    )
                    reference.has_audio = reference.waveform.shape[-1] > 0
        else:
            waveform, sample_rate = decode_reference_audio(item["path"])
            intent = item.get("audio_intent", "voice")
            follows_sequence_timeline = intent in {"drive", "style"}
            segment_start_time = timeline_start_time if follows_sequence_timeline else 0.0
            reference.waveform = prepare_reference_waveform(
                waveform,
                sample_rate,
                audio_sample_rate,
                max_duration,
                start_time=segment_start_time,
                pad_to_duration=follows_sequence_timeline,
            )
            reference.has_audio = True
            if follows_sequence_timeline:
                print(
                    "[MiniMax H3 Ref2VA] Prepared "
                    f"{intent} audio timeline segment "
                    f"{segment_start_time:.2f}-{segment_start_time + max_duration:.2f}s "
                    f"from {os.path.basename(item['path'])}."
                )

        prepared.append(reference)
    return prepared


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    stride = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    timestamps = [index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for index in range(0, len(timestamps), MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    ]
    return [frames[index] for index in indices], block_timestamps


def trim_reference_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"A reference video must have at least one frame, got {num_frames}.")
    return (
        max(1, (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK)
        * MINIMAX_H3_FRAMES_PER_CHUNK
        + MINIMAX_H3_LATENTS_PER_CHUNK
    )


def build_ref2va_presentation(
    tokenizer,
    prompt: str,
    references: list[MiniMaxH3PreparedReference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
) -> tuple[list[int], list[int]]:
    def text(value: str):
        token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return token_ids, [MINIMAX_H3_TEXT_TAG] * len(token_ids)

    def vision(pad_token: str, num_tokens: int):
        token_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return token_ids, [MINIMAX_H3_VIDEO_TAG] * len(token_ids)

    token_ids: list[int] = []
    token_tags: list[int] = []

    def emit(segment):
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if reference.kind == "image":
            counts["image"] += 1
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
        elif reference.kind == "video":
            counts["video"] += 1
            emit(text(f"<Video {counts['video']}>: "))
            for timestamp in reference.block_timestamps:
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
    emit(text(prompt))
    return token_ids, token_tags


def _reference_temporal_span(num_latent_frames: int) -> float:
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        for index in range(num_latent_frames)
    )


def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(position_ids, rows: slice, num_audio_latents: int, rotary_time: float, width_grid):
    if num_audio_latents == 0:
        return
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: list[MiniMaxH3PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors=(),
    audio_condition_anchors=(),
    target_condition_audio_latents: int = 0,
    target_condition_video_frames: int = 0,
) -> MiniMaxH3PackedSequence:
    """Build Ref2VA references plus optional native continuation history.

    The packed order mirrors WanGP 12.44: keyframe video, keyframe audio,
    canonical ordered references, target audio, then target video. Rotary
    time still places canonical references before the carried history and the
    newly generated target, so the references remain authoritative while the
    history supplies local motion and sound continuity.
    """

    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    target_frame_grid, target_width_grid = _frame_position_grid(
        latent_height,
        latent_width,
        patch_h,
        patch_w,
    )
    rows_per_target_frame = target_frame_grid.shape[0]
    num_target_video_rows = num_latent_frames * rows_per_target_frame
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    keyframe_frames = sum(
        _unpack_condition_anchor(anchor)[1]
        for anchor in keyframe_anchors
    )
    keyframe_video_rows = keyframe_frames * rows_per_target_frame
    keyframe_audio_latents = sum(
        int(anchor[1]) if isinstance(anchor, tuple) else 1
        for anchor in audio_condition_anchors
    )
    keyframe_audio_rows = (
        keyframe_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    )
    num_reference_video_rows = sum(reference.num_video_rows for reference in references if reference.kind != "audio")
    num_reference_audio_rows = sum(reference.num_audio_rows for reference in references)
    sequence_length = (
        num_text_tokens
        + keyframe_video_rows
        + keyframe_audio_rows
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    keyframe_start = num_text_tokens
    keyframe_audio_start = keyframe_start + keyframe_video_rows
    reference_start = keyframe_audio_start + keyframe_audio_rows
    video_indices: list[torch.Tensor] = (
        [torch.arange(keyframe_start, keyframe_audio_start)]
        if keyframe_video_rows
        else []
    )
    audio_indices: list[torch.Tensor] = (
        [torch.arange(keyframe_audio_start, reference_start)]
        if keyframe_audio_rows
        else []
    )
    cursor = reference_start
    rotary_time = float(num_text_tokens)
    for reference in references:
        if reference.kind == "image":
            rows = slice(cursor, cursor + reference.num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(reference.latent_height, reference.latent_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif reference.kind == "audio":
            rows = slice(cursor, cursor + reference.num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, reference.num_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(reference.num_audio_latents)
        elif reference.kind == "video":
            audio_rows = slice(cursor, cursor + reference.num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + reference.num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))
            frame_grid, width_grid = _frame_position_grid(
                reference.latent_height, reference.latent_width, patch_h, patch_w
            )
            _fill_audio_positions(position_ids, audio_rows, reference.num_audio_latents, rotary_time, width_grid)
            frame_time = _temporal_position_grid(reference.num_latent_frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.num_latent_frames, 1)
            rotary_time += max(float(reference.num_audio_latents), _reference_temporal_span(reference.num_latent_frames))
        else:
            raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}.")

    history_frames = sum(
        _unpack_condition_anchor(anchor)[1]
        for anchor in keyframe_anchors
        if _unpack_condition_anchor(anchor)[0] == "history"
    )
    target_origin = rotary_time + _temporal_position_span(history_frames)
    target_times = _temporal_position_grid(
        num_latent_frames,
        target_origin,
    )
    condition_cursor = keyframe_start
    history_time = rotary_time
    for entry in keyframe_anchors:
        anchor, condition_frames, frame_index = _unpack_condition_anchor(entry)
        if condition_frames <= 0:
            raise ValueError(
                "MiniMax H3 Ref2VA condition anchors must contain at least "
                f"one latent frame, got {entry!r}."
            )
        rows = slice(
            condition_cursor,
            condition_cursor + condition_frames * rows_per_target_frame,
        )
        condition = position_ids[rows].view(
            condition_frames,
            rows_per_target_frame,
            3,
        )
        if anchor == "history":
            condition[:, :, 0] = _temporal_position_grid(
                condition_frames,
                history_time,
            )[:, None]
            history_time += _temporal_position_span(condition_frames)
        elif anchor == "first":
            condition[:, :, 0] = target_times[:condition_frames, None]
        elif anchor == "last":
            condition[:, :, 0] = (
                target_origin
                + _temporal_position_span(num_latent_frames)
                - _ROPE_FRAME_RESCALE
            )
        elif anchor == "frame":
            if frame_index is None:
                raise ValueError(
                    "A MiniMax H3 Ref2VA 'frame' condition needs a target "
                    "frame index."
                )
            condition[:, :, 0] = (
                target_origin + frame_index * _ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(
                f"Unknown MiniMax H3 Ref2VA keyframe anchor {anchor!r}."
            )
        condition[:, :, 1:] = target_frame_grid[None]
        condition_cursor = rows.stop

    _fill_audio_condition_positions(
        position_ids,
        keyframe_audio_start,
        audio_condition_anchors,
        rotary_time,
        target_origin,
        target_width_grid,
    )
    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        target_origin,
        target_width_grid,
    )
    position_ids[video_start:, 0] = target_times.repeat_interleave(
        target_frame_grid.shape[0]
    )
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=(
            keyframe_video_rows + num_reference_video_rows
        ),
        num_condition_audio_rows=(
            keyframe_audio_rows + num_reference_audio_rows
        ),
        num_target_condition_audio_latents=max(
            0,
            int(target_condition_audio_latents),
        ),
        num_target_condition_video_rows=(
            max(0, int(target_condition_video_frames))
            * rows_per_target_frame
        ),
    )
