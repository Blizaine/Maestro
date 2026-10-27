"""Focused, transactional ownership corrections for a fixed music outline."""
from collections import Counter
from collections.abc import Mapping
import copy
import json


def allocation_report(assignments, contract, clip_count):
    expected = [event["event_id"] for event in contract.get("events", [])]
    pairs = (list(assignments.items()) if isinstance(assignments, Mapping) else
             [(item.get("event_id"), item.get("clip_index"))
              for item in assignments if isinstance(item, dict)]
             if isinstance(assignments, list) else [])
    pairs = [(event, index) for event, index in pairs if isinstance(event, str)]
    counts = Counter(event for event, _ in pairs)
    valid = lambda index: isinstance(index, int) and not isinstance(index, bool) and 0 <= index < clip_count
    rank = {event: index for index, event in enumerate(expected)}
    ordered = sorted(pairs, key=lambda pair: (pair[1] if valid(pair[1]) else clip_count,
                                            rank.get(pair[0], len(expected))))
    by_event = {event: index for event, index in pairs if counts[event] == 1 and valid(index)}
    inversions = [{"earlier_id": first, "later_id": second,
                   "earlier_clip": by_event[first], "later_clip": by_event[second]}
                  for offset, first in enumerate(expected) for second in expected[offset + 1:]
                  if first in by_event and second in by_event and by_event[first] > by_event[second]]
    report = {
        "expected": expected, "received": [event for event, _ in ordered],
        "missing": [event for event in expected if not counts[event]],
        "duplicate": list(dict.fromkeys(event for event, _ in pairs if counts[event] > 1)),
        "unknown": list(dict.fromkeys(event for event, _ in pairs if event not in rank)),
        "invalid_indices": [event for event, index in pairs if not valid(index)],
        "out_of_order": inversions,
    }
    implicated = {event for item in inversions for event in (item["earlier_id"], item["later_id"])}
    report["preserved_assignments"] = {event: by_event[event] for event in expected
        if event in by_event and counts[event] == 1 and event not in implicated}
    return report


def allocation_error(report):
    return "; ".join(f"{key}={report[key]!r}" for key in
                     ("expected", "received", "missing", "duplicate", "unknown",
                      "invalid_indices", "out_of_order"))


def _envelope(response):
    if isinstance(response, list) and len(response) == 1:
        response = response[0]
    if not isinstance(response, dict) or "beats" not in response:
        raise ValueError("allocation-only correction requires an existing beats outline")
    if set(response) - {"beats", "source_event_clips"}:
        raise ValueError("outline response has unrelated fields; repair its structure first")
    return response


def build_allocation_repair_request(response, groups, contract):
    """Only structurally usable beats can be preserved for focused correction."""
    from .music_arc import coerce_music_arc_rows, normalize_music_arc
    original = _envelope(response)
    beats = coerce_music_arc_rows(copy.deepcopy(original["beats"]), len(groups))
    # Check creative row structure separately; no source plan is accepted here.
    # Full ownership and action validation run on the candidate before commit.
    structural = [{**row, "source_event_ids": []} for row in beats]
    normalize_music_arc(structural, groups, {"events": []})
    if any("source_event_ids" in row for row in beats):
        raise ValueError("beats must omit source_event_ids; use source_event_clips")
    report = allocation_report(original.get("source_event_clips"), contract, len(groups))
    anchors = report["preserved_assignments"]
    expected = report["expected"]
    properties = {}
    for position, event in enumerate(expected):
        lower = max([0] + [anchors[key] for key in expected[:position + 1] if key in anchors])
        upper = min([len(groups) - 1] + [anchors[key] for key in expected[position:] if key in anchors])
        properties[event] = {"type": "integer", "minimum": lower, "maximum": upper}
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"source_event_clips": {
                  "type": "object", "properties": properties, "required": expected,
                  "additionalProperties": False}}, "required": ["source_event_clips"]}
    prompt = (
        "Correct only source-event ownership for this fixed music outline. "
        "Return source_event_clips only, not beats. Every contract ID must occur once; "
        "its ZERO-BASED clip index must be nondecreasing in contract source order. "
        "Keep preserved_assignments exactly unchanged. For missing/invalid/conflicting "
        "assignments, choose the supported beat within the schema bounds. Do not invent "
        "events, renumber IDs, reorder beats, or rewrite creative fields. Unknown IDs "
        "are not contract actions and must not appear in the corrected mapping. "
        "If no supported correction exists, do not fabricate a source action.\n"
        + json.dumps({"contract": contract, "fixed_beats": beats,
                      "rejected_assignments": original.get("source_event_clips"),
                      "diagnostics": report, "required_response_schema": schema}, ensure_ascii=False)
    )
    return prompt, schema, report


def apply_allocation_patch(response, proposal, groups, contract):
    """Validate a working copy; a rejected patch cannot mutate its baseline."""
    from .music_arc import coerce_music_arc_rows, normalize_music_arc
    original = _envelope(response)
    _prompt, _schema, report = build_allocation_repair_request(original, groups, contract)
    if isinstance(proposal, list) and len(proposal) == 1:
        proposal = proposal[0]
    if not isinstance(proposal, dict) or set(proposal) - {"source_event_clips", "beats"}:
        raise ValueError("allocation patch must contain source_event_clips only")
    if "beats" in proposal and proposal["beats"] != original["beats"]:
        raise ValueError("allocation patch tried to rewrite fixed creative beats")
    mapping = proposal.get("source_event_clips")
    expected = report["expected"]
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        raise ValueError("allocation patch must contain exactly the contract IDs; "
                         + allocation_error(allocation_report(mapping, contract, len(groups))))
    if any(not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(groups)
           for index in mapping.values()):
        raise ValueError("allocation patch contains an invalid zero-based clip index")
    if any(mapping[event] != index for event, index in report["preserved_assignments"].items()):
        raise ValueError("allocation patch moved a preserved source assignment")
    working = copy.deepcopy(original)
    working["source_event_clips"] = copy.deepcopy(mapping)
    rows = coerce_music_arc_rows(copy.deepcopy(working["beats"]), len(groups))
    by_index = {row["clip_index"]: row for row in rows}
    for row in rows:
        row["source_event_ids"] = []
    for event in expected:
        by_index[mapping[event]]["source_event_ids"].append(event)
    normalize_music_arc(rows, groups, contract)
    return working
