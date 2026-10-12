"""Source-owned story anchors that survive screenplay shortening.

These are performed actions, not extra camera instructions. The shot writer
allocates their IDs; compilation copies their original text instead of relying
on a second paraphrase to preserve a small but important detail.
"""

from __future__ import annotations

import re
from typing import Any

_DIALOGUE_BODY_BLOCK_RE = re.compile(r"<d>.*?</d>", re.IGNORECASE | re.DOTALL)
_DIALOGUE_FINAL_MARKER_RE = re.compile(
    r"\b(?:then|finally|eventually|now|remains?|stays?|ends?\s+with|"
    r"finishes?\s+with|at\s+(?:the\s+)?end|at\s+the\s+final\s+"
    r"(?:frame|image|moment)|in\s+the\s+final\s+(?:frame|image|moment))\b",
    re.IGNORECASE,
)
_DIALOGUE_TERMINAL_CLAUSE_RE = re.compile(
    r"\b(?:then|ends?\s+with|finishes?\s+with)\b", re.IGNORECASE,
)
_DIALOGUE_FINAL_GROUND_RE = re.compile(
    r"\b(?:(?:stands?|standing|sits?|sitting|seated|kneels?|kneeling|"
    r"lies?|lying|rests?|resting)\s+(?:firmly\s+)?(?:on|upon)\s+"
    r"(?:the\s+)?(?:ground|floor|earth|soil|pavement)|"
    r"(?:both|two)\s+(?:feet|paws)\s+(?:(?:are|remain|stay)\s+)?"
    r"(?:firmly\s+)?(?:on|upon)\s+(?:the\s+)?(?:ground|floor)|"
    r"(?:is|are|remains?|stays?)\s+grounded)\b",
    re.IGNORECASE,
)
_DIALOGUE_FINAL_MOUNTED_RE = re.compile(
    r"\b(?:astride|aboard|riding|mounted|mount(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_DIALOGUE_FINAL_AIRBORNE_RE = re.compile(
    r"\b(?:airborne|in\s+flight|taking\s+flight|takes?\s+flight|"
    r"lifts?\s+off)\b",
    re.IGNORECASE,
)
_CLOSING_MOUNTED_SUPPORT_RE = re.compile(
    r"\b(?:astride|aboard|riding|mounted)\b", re.IGNORECASE,
)
_CLOSING_GROUND_SUPPORT_RE = re.compile(
    r"\b(?:ground|floor|earth|soil|pavement)\b", re.IGNORECASE,
)


_SOURCE_PROP_OPERATION_FAMILIES = (
    (
        "stow",
        re.compile(
            r"\b(?:stow\w*|stash\w*|tuck\w*\s+away|tuck\w*|pocket(?:s|ed|ing)|"
            r"pocket(?=\s+(?!(?:and|then|with|of|in|into|on|at|under|over|by|from|to|for|while|where|when|that|which)\b)[A-Za-z])|"
            r"slip\w*|slide\w*|"
            r"slid|put\w*\s+away|put|puts|putting|place\w*|secure\w*|sheathe\w*|"
            r"holster\w*|strap\w*|fasten\w*|clip\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "draw",
        re.compile(
            r"\b(?:unsheathe\w*|unholster\w*|draw\w*|pull\w*\s+out|pull\w*|"
            r"remove\w*|unfasten\w*|unclip\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "transfer",
        re.compile(
            r"\b(?:give\w*|gave|hand\w*|pass\w*|transfer\w*|"
            r"deliver\w*|return\w*|offer\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "retrieve",
        re.compile(
            r"\b(?:pick\w*\s+up|lift\w*|grab\w*|collect\w*|"
            r"retrieve\w*|take\w*\s+out|take\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "open",
        re.compile(r"\b(?:open\w*|unlock\w*|unseal\w*)\b", re.IGNORECASE),
    ),
    (
        "close",
        re.compile(r"\b(?:close\w*|shut\w*|lock\w*)\b", re.IGNORECASE),
    ),
)
_SOURCE_PROP_TARGET_BOUNDARY_RE = re.compile(
    r"\b(?:in|into|onto|on|at|under|over|through|from|to|away|back|"
    r"down|up|out|aside|across|along|toward|towards|beside|behind|between|past|"
    r"then|while|before|after|and|with|around|where|when|which|because)\b|[,.;!?\r\n]",
    re.IGNORECASE,
)
_SOURCE_PROP_ANCHOR_STOPWORDS = {
    "a", "an", "the", "his", "her", "their", "its", "my", "your",
    "our", "this", "that", "these", "those", "it", "them", "him",
    "me", "you", "us", "are", "is", "was", "were", "be", "been",
    "being", "and", "as", "by", "for", "of", "with", "around",
    "quietly", "carefully", "securely", "gently", "quickly", "firmly",
    "slowly", "softly", "patiently", "deliberately", "neatly",
    "completely", "fully", "tightly",
}
_SOURCE_NONPROP_ANCHORS = {
    "attention", "breath", "eyes", "face", "flight", "foot", "feet",
    "hand", "hands", "head", "heart", "leg", "legs", "look", "mood",
    "shoulder", "shoulders", "smile", "step", "steps", "voice", "wing",
    "wings", "say", "says", "said", "speak", "speaks", "spoke",
    "ask", "asks", "asked", "reply", "replies", "replied", "answer",
    "answers", "answered", "call", "calls", "called", "whisper",
    "whispers", "whispered", "shout", "shouts", "shouted", "arm",
    "arms", "mouth", "lips", "fist", "fingers", "finger", "moment", "time", "break", "seat", "chance", "risk", "responsibility",
    "walk", "bow", "turn", "nap", "advice", "help", "guidance",
    "encouragement", "warning", "conversation", "discussion", "crowd",
    "conclusion", "close", "near", "closer", "nearer", "distance", "gap", "guard", "person",
    "man", "woman", "child", "traveler", "stranger", "passerby",
    "threshold", "doorway", "entrance", "exit", "hallway", "road", "street",
    "idea", "thought", "suggestion", "gaze", "apology", "comfort", "resistance",
    "comparison",
}
_SOURCE_SPOKEN_QUOTE_RE = re.compile(
    r"(?:\"[^\"\r\n]*\"|“[^”\r\n]*”|‘[^’\r\n]*’|'[^'\r\n]*')"
)
_SOURCE_SPEECH_CUE_RE = re.compile(
    r"\b(?:says?|said|speaks?|spoke|asks?|asked|replies?|replied|"
    r"answers?|answered|whispers?|whispered|shouts?|shouted|yells?|yelled|"
    r"calls?\s+out|utters?|recites?)\b",
    re.IGNORECASE,
)
_SOURCE_STATIC_STATE_PREFIXES = {
    "the", "a", "an", "his", "her", "their", "its", "my", "your",
    "some", "any", "is", "are", "was", "were", "remains", "remain",
    "remained", "stays", "stay", "stayed", "appears", "appeared",
    "seems", "seemed", "looks", "looked", "wide", "slightly", "partly",
    "partially", "half", "fully", "already", "still", "freshly", "newly",
    "barely", "almost", "nearly", "ajar", "large", "small", "huge",
    "heavy", "old", "wooden", "iron", "red", "blue", "bright", "dark",
    "stands", "stand", "stood", "sits", "sit", "sat", "rests", "rest",
    "rested", "hangs", "hang", "hung", "left", "kept",
}
_SOURCE_BARE_VERB_PREFIXES = {
    "to", "must", "should", "will", "can", "could", "would", "please",
    "and", "then", "finally", "now",
}


def _is_static_open_close_word(text: str, family: str, match: re.Match[str]) -> bool:
    """Avoid treating common adjective/state uses as performed operations."""

    if family not in {"open", "close"}:
        return False
    tokens = re.findall(
        r"[A-Za-z][A-Za-z’'-]*",
        text[max(0, match.start() - 80):match.start()],
    )
    previous = tokens[-1].casefold() if tokens else ""
    word = match.group(0).casefold()
    if family == "open" and word == "open":
        if not previous or previous in _SOURCE_BARE_VERB_PREFIXES:
            return False
        return previous in _SOURCE_STATIC_STATE_PREFIXES
    if family == "open" and word == "opening":
        if previous in {"is", "are", "was", "were"}:
            prior = tokens[-2].casefold() if len(tokens) > 1 else ""
            return prior in _SOURCE_STATIC_STATE_PREFIXES
        return previous in _SOURCE_STATIC_STATE_PREFIXES
    if family == "close" and word == "close":
        return previous in _SOURCE_STATIC_STATE_PREFIXES
    if word in {"closed", "locked", "unlocked", "unsealed", "shut"}:
        return previous in _SOURCE_STATIC_STATE_PREFIXES
    if family == "close" and word == "lock":
        return previous in _SOURCE_STATIC_STATE_PREFIXES
    return False


def _strip_spoken_content_for_prop_check(value: Any) -> str:
    """Exclude tagged speech and attributed quotes from action/object checks."""

    text = re.sub(r"<\s*d\b[^>]*>.*?</\s*d\s*>", " ", str(value or ""), flags=re.IGNORECASE | re.DOTALL)
    spans = list(_SOURCE_SPOKEN_QUOTE_RE.finditer(text))
    for match in reversed(spans):
        before = text[max(0, match.start() - 100):match.start()]
        after = text[match.end():match.end() + 100]
        if _SOURCE_SPEECH_CUE_RE.search(before) or re.match(
            r"\s*,?\s*(?:[A-Z][\w’'-]*(?:\s+[A-Z][\w’'-]*){0,2}\s+)?"
            r"(?:says?|said|speaks?|spoke|asks?|asked|replies?|replied|"
            r"answers?|answered|whispers?|whispered|shouts?|shouted|"
            r"calls?\s+out|utters?|recites?)\b",
            after,
            flags=re.IGNORECASE,
        ):
            text = text[:match.start()] + " " + text[match.end():]
    return text


def _source_prop_action_anchors(text: str) -> list[tuple[str, str]]:
    """Find explicit prop operations paired with a short object-name anchor."""

    source = _strip_spoken_content_for_prop_check(text)
    requirements: list[tuple[str, str]] = []
    for family, pattern in _SOURCE_PROP_OPERATION_FAMILIES:
        for match in pattern.finditer(source):
            if _is_static_open_close_word(source, family, match):
                continue
            operation = match.group(0).casefold()
            tail = source[match.end():match.end() + 100]
            if family == "transfer" and operation.startswith("pass"):
                transfer_boundary = _SOURCE_PROP_TARGET_BOUNDARY_RE.search(tail)
                marker = (
                    transfer_boundary.group(0).strip().casefold()
                    if transfer_boundary else ""
                )
                if marker not in {"to", "along", "around"} or not tail[:transfer_boundary.start()].strip():
                    # ``passes the gate`` or ``passes the guard`` usually
                    # describes travel. Require a stated handoff/distribution.
                    continue
            if family == "transfer" and operation in {"hand", "hands"}:
                prefix = source[max(0, match.start() - 24):match.start()]
                if re.search(r"\b(?:my|your|his|her|their|its|the|a|an)\s+$", prefix, re.IGNORECASE) or re.match(
                    r"\s+(?:am|is|are|was|were|remain\w*|stay\w*|clasp\w*|trembl\w*|"
                    r"shak\w*|reach\w*|curl\w*|rest\w*|hang\w*)\b",
                    tail,
                    flags=re.IGNORECASE,
                ):
                    # Do not mistake ``her hands clasp`` or ``hands tremble``
                    # for a handoff verb.
                    continue
            target = source[match.end():match.end() + 100]
            boundary = _SOURCE_PROP_TARGET_BOUNDARY_RE.search(target)
            if boundary:
                target = target[:boundary.start()]
            words = [
                word.casefold()
                for word in re.findall(r"[A-Za-z][A-Za-z’'-]*", target)
                if word.casefold() not in _SOURCE_PROP_ANCHOR_STOPWORDS
            ]
            if not words:
                continue
            anchor = words[-1].strip("’'-")
            if len(anchor) < 3 or anchor in _SOURCE_NONPROP_ANCHORS:
                continue
            requirement = (family, anchor)
            if requirement not in requirements:
                requirements.append(requirement)

    # Preserve authored attached/stowed states such as ``with a knife sheathed
    # at the left hip`` even when the source event does not phrase them as a
    # transitive action.
    for match in re.finditer(
        r"\bwith\s+(?:a|an|the|his|her|their|its)?\s*"
        r"(?P<object>[A-Za-z][\w’'-]*(?:\s+[A-Za-z][\w’'-]*){0,5})\s+"
        r"(?:sheathed|holstered|strapped|fastened|clipped|tucked|secured)\b",
        source,
        flags=re.IGNORECASE,
    ):
        words = [
            word.casefold()
            for word in re.findall(r"[A-Za-z][A-Za-z’'-]*", match.group("object"))
            if word.casefold() not in _SOURCE_PROP_ANCHOR_STOPWORDS
        ]
        if words:
            anchor = words[-1].strip("’'-")
            if len(anchor) >= 3 and anchor not in _SOURCE_NONPROP_ANCHORS:
                requirement = ("stow", anchor)
                if requirement not in requirements:
                    requirements.append(requirement)
    return requirements


def _prompt_has_prop_operation(body: str, family: str, anchor: str) -> bool:
    family_pattern = dict(_SOURCE_PROP_OPERATION_FAMILIES).get(family)
    if family_pattern is None:
        return False
    clean_body = _strip_spoken_content_for_prop_check(body)
    anchor_forms = {anchor, anchor + "s", anchor + "es"}
    if anchor.endswith("y") and len(anchor) > 2:
        anchor_forms.add(anchor[:-1] + "ies")
    if anchor.endswith("fe"):
        anchor_forms.add(anchor[:-2] + "ves")
    elif anchor.endswith("f"):
        anchor_forms.add(anchor[:-1] + "ves")
    anchor_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(value) for value in sorted(anchor_forms, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    if family == "stow" and re.search(
        rf"{anchor_pattern.pattern}(?:\s+[A-Za-z][\w’'-]*){{0,3}}\s+"
        r"(?:sheathed|holstered|strapped|fastened|clipped|tucked|secured)\b",
        clean_body,
        flags=re.IGNORECASE,
    ):
        return True
    for clause in re.split(r"(?<=[.!?;])\s+|\r?\n|,\s+then\s+", clean_body, flags=re.IGNORECASE):
        for operation in family_pattern.finditer(clause):
            if _is_static_open_close_word(clause, family, operation):
                continue
            target = clause[operation.end():operation.end() + 100]
            boundary = _SOURCE_PROP_TARGET_BOUNDARY_RE.search(target)
            if boundary:
                target = target[:boundary.start()]
            if anchor_pattern.search(target):
                return True
    return False


def director_audio_policy(source: str) -> dict[str, bool]:
    """Resolve explicit prohibitions before any writer rewrites the brief."""
    from services.dialogue_writing import dialogue_forbidden
    from services.h3_story_ledger import extract_locked_dialogue

    source = str(source or "")
    # A character saying "no music" is dialogue content, not an instruction
    # about the production's soundtrack.
    for line in reversed(extract_locked_dialogue(source)):
        start, end = line.get("source_offset"), line.get("source_end")
        if isinstance(start, int) and isinstance(end, int):
            source = source[:start] + " " + source[end:]
    text = re.sub(r"\s+", " ", source).casefold()
    speech_text = re.sub(r"\bsilent (?:\w+\s+){0,3}(?:film|short|sequence|video)\b", "silent film", text)
    speech_forbidden = dialogue_forbidden(speech_text)
    music_forbidden = bool(re.search(
        r"\b(?:no (?:background |non[- ]diegetic )?music|without (?:background )?music|"
        r"do not (?:add|generate|include) (?:any )?music)\b", text,
    )) or bool(re.search(r"\bno\s+(?:speech|dialogue)(?:\s*[,/]?\s*(?:or|and))?\s+(?:background\s+)?music\b", text))
    if not music_forbidden:
        # A shared negation can govern an enumerated list: "no floating,
        # battle, speech or music". Match a complete music list item rather
        # than treating an unrelated later "music plays" clause as forbidden.
        for prohibition in re.finditer(
            r"\b(?:no|without|do not (?:add|generate|include))\s+([^.;\n]{1,200})",
            text,
        ):
            items = re.split(r",|\b(?:and|or)\b", prohibition.group(1))
            if len(items) > 1 and any(re.fullmatch(
                r"\s*(?:any\s+)?(?:(?:background|non[- ]diegetic)\s+)?music\s*",
                item,
            ) for item in items):
                music_forbidden = True
                break
    return {
        "generated_speech_allowed": not speech_forbidden,
        "generated_music_allowed": not music_forbidden,
    }


def build_source_contract(source: str) -> dict:
    from services.h3_story_ledger import extract_source_events

    policy = director_audio_policy(source)
    events = []
    for event in extract_source_events(source):
        text = str(event.get("text") or "").strip()
        if policy["generated_speech_allowed"] is False:
            # The source-event extractor removes locked speech contents; also
            # remove a dangling speech attribution after its quote was removed.
            before_attribution = text
            text = re.sub(r"\s*(?:and\s+)?(?:says?|speaks?|whispers?|shouts?|replies?|asks?)\s*[,.:;]?\s*$", "", text, flags=re.IGNORECASE)
            if text != before_attribution and re.fullmatch(
                r"(?:(?:[A-Z][\w'-]*\s+){1,3}|(?:The|A|An)\s+[\w'-]+\s+)"
                r"(?:says?|speaks?|whispers?|shouts?|replies?|asks?)\s*[,.:;]?",
                before_attribution,
            ):
                continue
        # Keep reference-role and preservation instructions in the source brief,
        # rather than treating them as things an actor must perform once.
        if not text or re.match(
            r"(?:preserve|maintain|keep|avoid|never|do not|no\b|"
            r"(?:picture|image|reference)\s*\d*\s+(?:supplies|provides|defines|is for))\b",
            text, re.IGNORECASE,
        ):
            continue
        events.append({"event_id": f"E{len(events) + 1}", "text": text})
    return {"revision": 1, "events": events, "audio_policy": policy}


def format_source_contract(contract: dict) -> str:
    events = "\n".join(f"{e['event_id']}: {e['text']}" for e in contract.get("events", []))
    policy = contract.get("audio_policy") or {}
    return (
        "IMMUTABLE USER ACTIONS (independent of the shortened screenplay):\n"
        + (events or "No concrete source actions were specified; develop the concept creatively.")
        + "\nAssign each event_id exactly once in chronological order using source_event_ids. "
        "An event may contain several ordered actions: perform all of them in order. "
        "Do not put its result in the opening frame before performing it. "
        "Carry forward explicit wardrobe and prop facts until an authored action changes them. "
        "Use an empty source_event_ids array for a connecting shot."
        + ("\nGENERATED SPEECH IS FORBIDDEN. dialogue_beats must be []." if policy.get("generated_speech_allowed") is False else "")
        + ("\nGENERATED MUSIC IS FORBIDDEN. Use ambience and practical effects only." if policy.get("generated_music_allowed") is False else "")
    )


def source_contract_issues(rows: list[dict], contract: dict, *, require_ids: bool = True) -> list[str]:
    """Check ownership/order. IDs alone are never used as proof of motion."""
    expected = [event["event_id"] for event in contract.get("events", [])]
    if not require_ids and not any("source_event_ids" in row for row in rows):
        return []  # Plans saved before the source contract existed.
    actual = []
    issues = []
    for index, row in enumerate(rows):
        ids = row.get("source_event_ids")
        if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
            issues.append(f"Shot {index + 1}: source_event_ids must be an array of event IDs")
            continue
        actual.extend(ids)
    if actual != expected:
        issues.append(
            "Source actions must be allocated exactly once in source order; "
            f"expected {expected}, received {actual}"
        )
    if any("source_event_ids" in row for row in rows):
        from .physical_continuity import source_physical_state_issues
        lookup = {
            event["event_id"]: {"event_id": event["event_id"], "text": event["text"]}
            for event in contract.get("events", [])
        }
        assigned = [[lookup[event_id] for event_id in row.get("source_event_ids", [])
                     if event_id in lookup] for row in rows]
        issues.extend(source_physical_state_issues(rows, assigned))
        issues.extend(_dialogue_source_motion_issues(rows, assigned))
    return issues


def _dialogue_subject_context(
    row: dict[str, Any], subject_id: str, retained_body: str,
) -> tuple[set[str], set[str], bool] | None:
    """Resolve only explicit stable-ID, cast-name, and dialogue-speaker aliases."""

    subject_key = re.sub(r"\s+", " ", subject_id).strip().casefold()
    owner_aliases = {subject_id.strip()}
    owner_keys = {subject_key}
    other_aliases: set[str] = set()
    subjects = row.get("subjects_on_screen")
    if isinstance(subjects, list):
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            names = {
                str(subject.get(field) or "").strip()
                for field in ("character_id", "speaker_name")
                if str(subject.get(field) or "").strip()
            }
            keys = {re.sub(r"\s+", " ", name).casefold() for name in names}
            if subject_key in keys:
                owner_aliases.update(names)
                owner_keys.update(keys)
            else:
                other_aliases.update(names)

    speaker_matches = False
    beats = row.get("dialogue_beats")
    if isinstance(beats, list):
        for beat in beats:
            if not isinstance(beat, dict):
                continue
            for field in ("speaker_id", "speaker_name"):
                speaker = str(beat.get(field) or "").strip()
                if speaker and re.sub(r"\s+", " ", speaker).casefold() in owner_keys:
                    speaker_matches = True

    def mentioned(text: str, aliases: set[str]) -> bool:
        return any(
            re.search(
                rf"(?<![\w]){re.escape(alias)}(?![\w])", text, re.IGNORECASE,
            )
            for alias in aliases if alias
        )

    body_mentions_owner = mentioned(retained_body, owner_aliases)
    if not speaker_matches and not body_mentions_owner:
        return None
    return owner_aliases, other_aliases, speaker_matches or body_mentions_owner


def _latest_alias_position(text: str, aliases: set[str]) -> int:
    latest = -1
    for alias in aliases:
        if not alias:
            continue
        for match in re.finditer(
            rf"(?<![\w]){re.escape(alias)}(?![\w])", text, re.IGNORECASE,
        ):
            latest = max(latest, match.end())
    return latest


def _terminal_state_cue_belongs_to_subject(
    terminal: str,
    cue: re.Match[str],
    subject_context: tuple[set[str], set[str], bool],
) -> bool:
    owner_aliases, other_aliases, implicit_owner = subject_context
    prefix = terminal[max(0, cue.start() - 100):cue.start()]
    owner_position = _latest_alias_position(prefix, owner_aliases)
    other_position = _latest_alias_position(prefix, other_aliases)
    if owner_position >= 0:
        return owner_position > other_position
    if other_position >= 0:
        return False
    return implicit_owner


def _dialogue_terminal_state_issues(
    row: dict[str, Any], index: int, retained_prompt: str,
) -> list[str]:
    """Catch explicit terminal body states that oppose typed closing facts.

    The check is limited to a final/remaining clause after the last dialogue
    block and only compares recognized ground, mounted, and airborne states.
    """

    dialogue_blocks = list(_DIALOGUE_BODY_BLOCK_RE.finditer(retained_prompt))
    if not dialogue_blocks:
        return []
    after_dialogue = retained_prompt[dialogue_blocks[-1].end():]
    markers = list(_DIALOGUE_FINAL_MARKER_RE.finditer(after_dialogue))
    if not markers:
        return []
    clause_markers = list(_DIALOGUE_TERMINAL_CLAUSE_RE.finditer(after_dialogue))
    terminal_start = clause_markers[-1].end() if clause_markers else 0
    terminal = after_dialogue[terminal_start:].strip(" \t\r\n,;:-")
    if not terminal:
        return []

    from .physical_continuity import normalize_physical_state, physical_state_from_row

    try:
        state = normalize_physical_state(physical_state_from_row(row))
    except ValueError:
        return []
    if not state:
        return []
    retained_body = _DIALOGUE_BODY_BLOCK_RE.sub("", retained_prompt)

    cues: list[tuple[int, str, re.Match[str]]] = []
    for family, pattern in (
        ("ground", _DIALOGUE_FINAL_GROUND_RE),
        ("mounted", _DIALOGUE_FINAL_MOUNTED_RE),
        ("airborne", _DIALOGUE_FINAL_AIRBORNE_RE),
    ):
        for match in pattern.finditer(terminal):
            cues.append((match.start(), family, match))

    issues: list[str] = []
    for fact in state["closing"]:
        predicate = fact["predicate"]
        if predicate not in {"support", "grounded"}:
            continue
        context = _dialogue_subject_context(
            row, str(fact["subject_id"]), retained_body,
        )
        if context is None:
            continue
        subject_cues = [
            (position, family)
            for position, family, match in cues
            if _terminal_state_cue_belongs_to_subject(terminal, match, context)
        ]
        if not subject_cues:
            continue
        body_state = max(subject_cues, key=lambda cue: cue[0])[1]
        value = str(fact["value"])
        conflict = None
        if predicate == "support":
            expected = (
                "mounted"
                if _CLOSING_MOUNTED_SUPPORT_RE.search(value)
                else "ground"
                if _CLOSING_GROUND_SUPPORT_RE.search(value)
                else None
            )
            if expected in {"ground", "mounted"} and body_state in {"ground", "mounted"} and body_state != expected:
                conflict = (
                    f"explicit final {body_state} support conflicts with closing support={value!r}"
                )
        else:
            expected = value.casefold()
            if (expected == "airborne" and body_state == "ground") or (
                expected == "grounded" and body_state == "airborne"
            ):
                conflict = (
                    f"explicit final {body_state} body state conflicts with closing grounded={value!r}"
                )
        if conflict:
            issues.append(
                f"Shot {index}: dialogue-bearing video_prompt {conflict} for "
                f"{fact['subject_id']!r}; repair the retained terminal body or typed closing state"
            )
    return issues


def _dialogue_source_motion_issues(rows: list[dict], assigned: list[list[Any]]) -> list[str]:
    """Reject clear loss of source motion in retained dialogue-bearing prose.

    Silent shots are compiled from source actions. Spoken shots retain authored
    speech placement, so do not silently replace their body or append actions
    after speech. This bounded vocabulary catches missing physical operations;
    it is a repair guard, not a general semantic or rendering proof.
    """
    operations = {
        "boarding": r"\b(?:climb\w*|mounts?|mounted|mounting|boards?|boarded|boarding|ascend\w*)\b",
        "takeoff": r"\b(?:takes? flight|tak(?:e|es|ing) off|launch\w*|lifts? off)\b",
        "landing": r"\b(?:lands?|landed|landing|alights?|touch(?:es)? down)\b",
        "dismount": r"\b(?:dismount\w*|slides? down|sliding down|climbs? down|climbing down)\b",
        "wing folding": r"\b(?:fold\w*|tuck\w*)\s+(?:its\s+|the\s+|their\s+)?(?:massive\s+)?wings\b",
    }
    issues = []
    for index, (row, actions) in enumerate(zip(rows, assigned), start=1):
        if not any(isinstance(beat, dict) and str(beat.get("spoken_text") or "").strip()
                   for beat in row.get("dialogue_beats") or []):
            continue
        source = " ".join(
            str(action.get("text") or "") if isinstance(action, dict) else str(action)
            for action in actions
        )
        retained_prompt = re.split(
            r"\boverall_soundscape\s*:",
            str(row.get("video_prompt") or ""),
            flags=re.IGNORECASE,
        )[0]
        body = _DIALOGUE_BODY_BLOCK_RE.sub("", retained_prompt)
        for label, pattern in operations.items():
            if re.search(pattern, source, re.IGNORECASE) and not re.search(pattern, body, re.IGNORECASE):
                issues.append(f"Shot {index}: dialogue-bearing video_prompt omits assigned source motion ({label}); repair its authored action/speech order")
        for family, anchor in _source_prop_action_anchors(source):
            if not _prompt_has_prop_operation(body, family, anchor):
                issues.append(
                    f"Shot {index}: dialogue-bearing video_prompt omits assigned source prop operation "
                    f"({family}: {anchor}); repair its authored action/speech order"
                )
        issues.extend(_dialogue_terminal_state_issues(row, index, retained_prompt))
    return issues


def _typed_closing_description(value: Any) -> str:
    """Render only declared closing facts for source-owned silent rows."""

    from .physical_continuity import normalize_physical_state

    try:
        state = normalize_physical_state(value)
    except ValueError:
        # The planner's physical-state validator reports malformed data. Keep
        # this compiler from promoting an invalid model string into blocking.
        return ""
    if not state or not state["closing"]:
        return ""
    facts = "; ".join(
        f"{fact['subject_id']} {fact['predicate']}: {fact['value']}"
        for fact in state["closing"]
    )
    return f"At the final frame: {facts}."


def apply_source_contract(rows: list[dict], contract: dict) -> list[dict]:
    """Compile assigned source actions, keeping one authority for action prose."""
    lookup = {event["event_id"]: event["text"] for event in contract.get("events", [])}
    # Legacy saved fixtures without IDs keep their original action prose.
    active = any("source_event_ids" in row for row in rows)
    carried_details: list[str] = []
    for row in rows:
        row["_director_audio_policy"] = dict(contract.get("audio_policy") or {})
        row["_director_source_contract"] = contract
        if not active:
            continue
        ids = row.get("source_event_ids") or []
        actions = [lookup[event_id] for event_id in ids if event_id in lookup]
        if actions:
            # Required actions replace their model paraphrases; free-form camera,
            # environment, lighting and pacing remain authored by the director.
            row["action_beats"] = actions
            has_spoken_dialogue = any(
                isinstance(beat, dict) and str(beat.get("spoken_text") or "").strip()
                for beat in row.get("dialogue_beats") or []
            )
            if not has_spoken_dialogue:
                # Model-derived endings can leap ahead to a later source event
                # even after action_beats have been replaced by exact assigned
                # source text. Silent rows use that source text plus typed
                # endpoint facts as their only temporal authorities.
                row["ending_beat"] = ""
                row["closing_blocking"] = _typed_closing_description(
                    row.get("physical_state")
                )
            # Explicit carried-prop clauses must survive the compression boundary.
            # Limit this to attached/stowed details; posture/contact are temporal
            # facts owned by the typed physical-state ledger instead.
            for action in actions:
                if re.search(r"\b(?:unsheath\w*|unholster\w*|unfasten\w*|unclip\w*|draws?|removes?)\b", action, re.IGNORECASE):
                    # A later explicit release changes stowed-prop state. Clear
                    # conservatively rather than assert an obsolete source pose.
                    carried_details.clear()
                for match in re.finditer(
                    r"\bwith\s+[^.;,]{1,90}\b(?:sheathed|holstered|strapped|fastened|"
                    r"tucked|clipped)\b[^.;,]*", action, re.IGNORECASE,
                ):
                    detail = match.group(0).strip()
                    if detail not in carried_details:
                        carried_details.append(detail)
        # Source-owned action prose is complete; a state proposal cannot append
        # an extra performed action. Connecting shots retain authored motion.
        if row.get("physical_state") and not ids:
            from .physical_continuity import (
                physical_transition_actions_missing_from_prompt,
            )
            actions = list(row.get("action_beats") or [])
            actions.extend(physical_transition_actions_missing_from_prompt(
                row["physical_state"], " ".join(actions),
            ))
            row["action_beats"] = actions
        row["_director_carried_source_details"] = list(carried_details)
    return rows
