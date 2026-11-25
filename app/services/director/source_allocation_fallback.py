"""Deterministic source-event allocation after bounded LLM repairs fail.

LOCAL PATCH (short-film story mode). Music-video planning already compiles a
clip from its assigned source actions when writer repairs fail; short-film
story planning instead aborted the whole production when a small writer model
left source_event_ids out of order, duplicated, or invented (e.g. E7/E8 for a
four-event brief), even though every shot was otherwise usable.

This module repairs only event *ownership*:

1. drop unknown and duplicate IDs;
2. assign every contract event to one shot with a monotonic (story-order)
   dynamic program that maximises word overlap between the event text and the
   shot's authored fields, with a bonus for the writer's own assignment;
3. for dialogue-bearing shots whose retained prose omits an assigned physical
   operation, insert the exact source sentence *before* the first spoken line
   (never inside a <d> block and never after the speech).

Callers must re-run the unchanged validators (source_contract_issues and
validate_physical_state_sequence) and keep failing when anything remains, so
no invalid plan is accepted silently.
"""

from __future__ import annotations

import copy
import re
from typing import Any

_STOP = {
    "the", "and", "with", "while", "then", "that", "this", "from", "into", "onto",
    "over", "under", "above", "below", "back", "says", "said", "say", "into",
    "their", "there", "they", "them", "his", "her", "its", "a", "an", "of", "to",
    "in", "on", "at", "for", "by", "as", "is", "are", "be", "finally", "small",
    "alien",  # the recurring host appears in most shots; it does not localise an event
}

_SHOT_TEXT_FIELDS = (
    "title", "scene_goal", "causal_handoff", "spatial_setup", "environment",
    "video_prompt", "ending_beat", "closing_blocking", "action_beats",
    "camera_plan", "audio_plan",
)


def _words(value: Any) -> set[str]:
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    elif isinstance(value, dict):
        value = " ".join(str(v) for v in value.values())
    words = set()
    for token in re.findall(r"[a-z][a-z0-9'-]+", str(value or "").lower()):
        token = token.strip("'-")
        if len(token) < 3 or token in _STOP:
            continue
        # crude stem so "lifts"/"lifting"/"lift" and "orbits"/"orbiting" meet
        for suffix in ("ing", "es", "s", "ed"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                break
        words.add(token)
    return words


def _shot_words(row: dict) -> set[str]:
    bag: set[str] = set()
    for field in _SHOT_TEXT_FIELDS:
        value = row.get(field)
        if field == "video_prompt":
            # spoken words describe the news, not what is on screen
            value = re.sub(r"<d>.*?</d>", " ", str(value or ""), flags=re.I | re.S)
        bag |= _words(value)
    return bag


def _allocate(rows: list[dict], events: list[dict]) -> list[list[str]]:
    n_shots, n_events = len(rows), len(events)
    original_owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        for event_id in row.get("source_event_ids") or []:
            original_owner.setdefault(event_id, index)
    shot_bags = [_shot_words(row) for row in rows]
    event_bags = [_words(event["text"]) for event in events]

    def score(e: int, s: int) -> float:
        overlap = len(event_bags[e] & shot_bags[s])
        bonus = 0.5 if original_owner.get(events[e]["event_id"]) == s else 0.0
        return overlap + bonus

    # dp[e][s]: best total for events[:e+1] with event e in shot s (monotonic).
    neg = float("-inf")
    dp = [[neg] * n_shots for _ in range(n_events)]
    back = [[0] * n_shots for _ in range(n_events)]
    for s in range(n_shots):
        dp[0][s] = score(0, s)
    for e in range(1, n_events):
        best_prev, best_idx = neg, 0
        for s in range(n_shots):
            if dp[e - 1][s] > best_prev:
                best_prev, best_idx = dp[e - 1][s], s
            # small spread bonus: prefer moving to a new shot when evidence ties
            dp[e][s] = best_prev + score(e, s) + (0.01 if s > best_idx else 0.0)
            back[e][s] = best_idx
    s = max(range(n_shots), key=lambda k: dp[n_events - 1][k])
    owners = [0] * n_events
    for e in range(n_events - 1, -1, -1):
        owners[e] = s
        s = back[e][s]
    allocation: list[list[str]] = [[] for _ in rows]
    for e, shot in enumerate(owners):
        allocation[shot].append(events[e]["event_id"])
    return allocation


def _action_sentence(text: str) -> str:
    """The source clause as a standalone visible action (speech attribution removed)."""

    sentence = str(text or "").strip().rstrip(".;,")
    sentence = re.sub(
        r"(?:\s*,)?\s+(?:while|as|and|then)\s+(?:the\s+)?[\w'’ -]{1,40}?\s+(?:says?|speaks?|adds?|continues?)\s*$",
        "", sentence, flags=re.I,
    )
    sentence = re.sub(r"(?:\s*,)?\s+(?:and\s+)?(?:says?|speaks?)\s*$", "", sentence, flags=re.I)
    sentence = sentence.strip(" ,;")
    return (sentence[:1].upper() + sentence[1:] + ".") if sentence else ""


def _insert_before_speech(prompt: str, source_text: str) -> str:
    sentence = _action_sentence(source_text)
    if not sentence:
        return prompt
    split = re.search(r"\boverall_soundscape\s*:", prompt, re.I)
    head, tail = (prompt[: split.start()], prompt[split.start():]) if split else (prompt, "")
    match = re.search(r"<d>", head, re.I)
    if match:
        # place it at the start of the sentence that introduces the speech
        cut = max(head.rfind(". ", 0, match.start()), head.rfind("] ", 0, match.start()))
        cut = cut + 2 if cut >= 0 else 0
        head = head[:cut] + sentence + " " + head[cut:]
    else:
        head = head.rstrip() + " " + sentence + "\n"
    return head + tail


def reallocate_source_events(rows: list[dict], contract: dict) -> list[dict]:
    """Return a copy of rows with a valid, story-ordered source allocation."""

    from .source_contract import _dialogue_source_motion_issues

    events = list(contract.get("events") or [])
    rows = copy.deepcopy(rows)
    if not rows:
        return rows
    if not events:
        for row in rows:
            row["source_event_ids"] = []
        return rows
    allocation = _allocate(rows, events)
    for row, ids in zip(rows, allocation):
        row["source_event_ids"] = ids
        row["_source_allocation_fallback"] = True

    lookup = {event["event_id"]: event for event in events}
    assigned = [[lookup[i] for i in row["source_event_ids"]] for row in rows]
    issues = _dialogue_source_motion_issues(rows, assigned)
    flagged = {
        int(m.group(1)) - 1
        for issue in issues
        for m in [re.match(r"Shot (\d+): dialogue-bearing video_prompt omits", issue)]
        if m
    }
    for index in sorted(flagged):
        row = rows[index]
        prompt = str(row.get("video_prompt") or "")
        for event in assigned[index]:
            prompt = _insert_before_speech(prompt, event["text"])
        row["video_prompt"] = prompt
    return rows
