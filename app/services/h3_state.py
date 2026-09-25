"""Conservative source-event completion and camera handoff helpers for H3."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,79}$")
_MAX_ACCEPTED_ACTIONS = 4
_MAX_ACCEPTED_ACTION_CHARS = 600
_MAX_AUTHORITATIVE_HANDOFF_ACTION_CHARS = 40_000
_MAX_SOURCE_PROP_IDENTITIES = 4
_MAX_SOURCE_PROP_IDENTITY_CHARS = 96
_MAX_COMPLETED_SOURCE_QUOTES = 3
_MAX_COMPLETED_SOURCE_QUOTE_CHARS = 420
_MAX_WINDOW_INDEX = 99_999
_MAX_HISTORY_CONTEXT_CHARS = 4_800


def _event_key(value: Any) -> str:
    text = str(value or "").strip()
    return text.upper() if _IDENTIFIER_RE.fullmatch(text) else ""


def _source_event_index(
    source_events: Any,
) -> tuple[list[str], dict[str, Mapping[str, Any]], list[str]]:
    diagnostics: list[str] = []
    if not isinstance(source_events, Sequence) or isinstance(source_events, (str, bytes)):
        return [], {}, ["source events input is malformed"]

    raw_ids: list[tuple[int, str, Mapping[str, Any]]] = []
    counts: dict[str, int] = {}
    for index, event in enumerate(source_events):
        if not isinstance(event, Mapping):
            diagnostics.append(f"source event at position {index + 1} is malformed")
            continue
        event_id = _event_key(event.get("event_id"))
        if not event_id:
            diagnostics.append(f"source event at position {index + 1} has no valid event ID")
            continue
        counts[event_id] = counts.get(event_id, 0) + 1
        raw_ids.append((index, event_id, event))

    duplicate_ids = {event_id for event_id, count in counts.items() if count > 1}
    if duplicate_ids:
        diagnostics.append(
            "duplicate source event IDs ignored: " + ", ".join(sorted(duplicate_ids))
        )

    order: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for _index, event_id, event in raw_ids:
        if event_id in duplicate_ids:
            continue
        order.append(event_id)
        by_id[event_id] = event
    return order, by_id, diagnostics


def _validated_overview_details(
    raw_relationships: Any,
    source_order: list[str],
    source_by_id: Mapping[str, Mapping[str, Any]],
    diagnostics: list[str],
) -> dict[str, list[str]]:
    if raw_relationships is None:
        return {}
    if not isinstance(raw_relationships, Mapping):
        diagnostics.append("validated overview/detail relationships are malformed")
        return {}

    positions = {event_id: index for index, event_id in enumerate(source_order)}
    candidates: dict[str, list[str]] = {}
    duplicate_parents: set[str] = set()
    for raw_parent, raw_children in raw_relationships.items():
        parent = _event_key(raw_parent)
        if not parent or parent not in source_by_id:
            diagnostics.append("overview relationship references an unknown parent event")
            continue
        if parent in candidates or parent in duplicate_parents:
            candidates.pop(parent, None)
            duplicate_parents.add(parent)
            diagnostics.append(f"duplicate overview relationship for {parent}")
            continue
        if not isinstance(raw_children, Sequence) or isinstance(raw_children, (str, bytes)):
            diagnostics.append(f"overview {parent} has malformed detail IDs")
            continue
        children = [_event_key(value) for value in raw_children]
        if (
            not children
            or any(not child or child not in source_by_id for child in children)
            or len(set(children)) != len(children)
            or parent in children
        ):
            diagnostics.append(f"overview {parent} has invalid detail IDs")
            continue
        child_positions = [positions[child] for child in children]
        if child_positions != sorted(child_positions):
            diagnostics.append(f"overview {parent} detail IDs are out of source order")
            continue
        candidates[parent] = children

    # Cyclic overview links cannot prove that any parent result is covered.
    visit_state: dict[str, int] = {}
    visit_stack: list[str] = []
    cyclic: set[str] = set()

    def visit(parent: str) -> None:
        visit_state[parent] = 1
        visit_stack.append(parent)
        for child in candidates.get(parent, []):
            if child not in candidates:
                continue
            state = visit_state.get(child, 0)
            if state == 0:
                visit(child)
            elif state == 1:
                try:
                    cycle_start = visit_stack.index(child)
                except ValueError:
                    cycle_start = 0
                cyclic.update(visit_stack[cycle_start:])
        visit_stack.pop()
        visit_state[parent] = 2

    for parent in list(candidates):
        if visit_state.get(parent, 0) == 0:
            visit(parent)
    if cyclic:
        diagnostics.append(
            "cyclic overview relationships ignored: " + ", ".join(sorted(cyclic))
        )
        for parent in cyclic:
            candidates.pop(parent, None)

    return candidates


def _validated_camera_action_history(
    raw_actions: Any,
    source_by_id: Mapping[str, Mapping[str, Any]],
    current_window_index: Any,
    diagnostics: list[str],
) -> list[dict[str, Any]]:
    """Keep a short, provenance-bound history of accepted earlier-window actions."""

    if raw_actions is None:
        return []
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        diagnostics.append("accepted camera action history is malformed")
        return []
    if (
        not isinstance(current_window_index, int)
        or isinstance(current_window_index, bool)
        or current_window_index < 1
        or current_window_index > _MAX_WINDOW_INDEX
    ):
        diagnostics.append(
            "accepted camera action history omitted because current window index is invalid"
        )
        return []

    entries: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[tuple[int, tuple[str, ...], str]] = set()
    for position, item in enumerate(raw_actions):
        if not isinstance(item, Mapping):
            diagnostics.append(f"accepted camera action {position + 1} is malformed")
            continue
        if item.get("accepted") is not True:
            diagnostics.append(f"unaccepted camera action {position + 1} omitted")
            continue

        window_index = item.get("window_index")
        if (
            not isinstance(window_index, int)
            or isinstance(window_index, bool)
            or window_index < 1
            or window_index > _MAX_WINDOW_INDEX
        ):
            diagnostics.append(f"accepted camera action {position + 1} has an invalid window index")
            continue
        if window_index >= current_window_index:
            diagnostics.append(
                f"camera action from window {window_index} is not prior to current window "
                f"{current_window_index}; omitted"
            )
            continue

        raw_event_ids = item.get("source_event_ids")
        if raw_event_ids is None and item.get("source_event_id") is not None:
            raw_event_ids = [item.get("source_event_id")]
        if not isinstance(raw_event_ids, Sequence) or isinstance(raw_event_ids, (str, bytes)):
            diagnostics.append(f"accepted camera action {position + 1} has malformed source IDs")
            continue
        event_ids = [_event_key(value) for value in raw_event_ids]
        if any(not event_id for event_id in event_ids):
            diagnostics.append(
                f"accepted camera action {position + 1} has malformed source IDs"
            )
            continue
        event_ids = list(dict.fromkeys(event_ids))
        if any(event_id not in source_by_id for event_id in event_ids):
            diagnostics.append(f"accepted camera action {position + 1} has unknown source IDs")
            continue

        action = item.get("action")
        if not isinstance(action, str) or not action.strip():
            diagnostics.append(f"accepted camera action {position + 1} has no visible action text")
            continue
        action = action.strip()
        if len(action) > _MAX_ACCEPTED_ACTION_CHARS:
            diagnostics.append(f"accepted camera action {position + 1} exceeds the handoff limit")
            continue

        key = (window_index, tuple(event_ids), action)
        if key in seen:
            diagnostics.append(f"duplicate accepted camera action {position + 1} omitted")
            continue
        seen.add(key)
        entries.append((window_index, position, {
            "accepted": True,
            "window_index": window_index,
            "source_event_ids": event_ids,
            "action": action,
        }))

    entries.sort(key=lambda entry: (entry[0], entry[1]))
    history = [entry for _window, _position, entry in entries]
    if len(history) > _MAX_ACCEPTED_ACTIONS:
        diagnostics.append(
            f"accepted camera action history limited to the latest {_MAX_ACCEPTED_ACTIONS} actions"
        )
        history = history[-_MAX_ACCEPTED_ACTIONS:]
    return history


def _validated_source_prop_identities(
    raw_identities: Any,
    source_by_id: Mapping[str, Mapping[str, Any]],
    completed_event_ids: set[str],
    camera_history: Sequence[Mapping[str, Any]],
    diagnostics: list[str],
) -> list[dict[str, str]]:
    """Keep source-quoted prop identity only after its source event is accepted."""

    if raw_identities is None:
        return []
    if not isinstance(raw_identities, Sequence) or isinstance(raw_identities, (str, bytes)):
        diagnostics.append("source prop identity input is malformed")
        return []

    visible_event_ids = set(completed_event_ids)
    for action in camera_history:
        visible_event_ids.update(action.get("source_event_ids", []))

    identities: list[dict[str, str]] = []
    seen_prop_ids: set[str] = set()
    for position, item in enumerate(raw_identities):
        if not isinstance(item, Mapping):
            diagnostics.append(f"source prop identity {position + 1} is malformed")
            continue
        prop_id = _event_key(item.get("prop_id"))
        source_event_id = _event_key(item.get("source_event_id"))
        identity = item.get("identity")
        quote = item.get("source_quote")
        if (
            not prop_id
            or prop_id in seen_prop_ids
            or not source_event_id
            or source_event_id not in source_by_id
            or not isinstance(identity, str)
            or not identity.strip()
            or len(identity.strip()) > _MAX_SOURCE_PROP_IDENTITY_CHARS
            or not isinstance(quote, str)
            or not quote.strip()
        ):
            diagnostics.append(f"source prop identity {position + 1} has invalid fields")
            continue

        source_text = source_by_id[source_event_id].get("text")
        identity = identity.strip()
        quote = quote.strip()
        if (
            not isinstance(source_text, str)
            or quote not in source_text
            or identity.casefold() not in quote.casefold()
        ):
            diagnostics.append(
                f"source prop identity {prop_id} lacks an exact source-quoted identity"
            )
            continue
        if source_event_id not in visible_event_ids:
            diagnostics.append(
                f"source prop identity {prop_id} withheld until its source event is accepted"
            )
            continue

        seen_prop_ids.add(prop_id)
        identities.append({
            "prop_id": prop_id,
            "identity": identity,
            "source_event_id": source_event_id,
            "source_quote": quote,
        })
        if len(identities) >= _MAX_SOURCE_PROP_IDENTITIES:
            if position + 1 < len(raw_identities):
                diagnostics.append(
                    f"source prop identities limited to {_MAX_SOURCE_PROP_IDENTITIES} entries"
                )
            break
    return identities


def _completed_source_event_quotes(
    completed_event_ids: Sequence[str],
    source_by_id: Mapping[str, Mapping[str, Any]],
    diagnostics: list[str],
) -> list[dict[str, str]]:
    """Expose a few exact, completed source-event texts without state inference."""

    quotes: list[dict[str, str]] = []
    for event_id in completed_event_ids[-_MAX_COMPLETED_SOURCE_QUOTES:]:
        event = source_by_id.get(event_id)
        text = event.get("text") if isinstance(event, Mapping) else None
        if not isinstance(text, str) or not text.strip():
            diagnostics.append(f"completed source event {event_id} has no usable exact text")
            continue
        text = text.strip()
        if len(text) > _MAX_COMPLETED_SOURCE_QUOTE_CHARS:
            diagnostics.append(
                f"completed source event {event_id} omitted from compact handoff "
                "because its text is long"
            )
            continue
        quotes.append({"source_event_id": event_id, "text": text})
    return quotes


def project_h3_achieved_state(
    source_events: Sequence[Mapping[str, Any]],
    accepted_completed_event_ids: Sequence[str],
    *,
    validated_overview_details: Mapping[str, Sequence[str]] | None = None,
    accepted_visible_camera_actions: Sequence[Mapping[str, Any]] | None = None,
    current_window_index: int | None = None,
    source_prop_identities: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return an ordered completion audit without inferring entity properties.

    ``accepted_completed_event_ids`` must contain only events whose visible
    camera action passed the existing coverage check or was source-compiled by
    the deterministic fallback. Assignments and planner claims are not
    completion evidence. A caller-provided overview is non-executable only
    when its ordered detail IDs are structurally valid; it is covered after
    all details fall inside the completed executable prefix. Optional camera
    history must contain accepted visible actions and is filtered to earlier
    native windows. Optional prop identities require an exact source quote and
    are exposed as identity only, never as position or possession.
    """

    source_order, source_by_id, diagnostics = _source_event_index(source_events)
    overview_details = _validated_overview_details(
        validated_overview_details, source_order, source_by_id, diagnostics,
    )
    overview_ids = set(overview_details)
    obligation_order = [event_id for event_id in source_order if event_id not in overview_ids]

    if not isinstance(accepted_completed_event_ids, Sequence) or isinstance(
        accepted_completed_event_ids, (str, bytes)
    ):
        diagnostics.append("accepted completion input is malformed")
        accepted_completed_event_ids = []
    accepted = {
        event_id
        for value in accepted_completed_event_ids
        if (event_id := _event_key(value))
    }
    unknown = sorted(accepted - set(source_by_id))
    if unknown:
        diagnostics.append("unknown completed source event IDs: " + ", ".join(unknown))
    falsely_completed_overviews = sorted(accepted & overview_ids)
    if falsely_completed_overviews:
        diagnostics.append(
            "overview IDs are non-executable; completion must come from detail obligations: "
            + ", ".join(falsely_completed_overviews)
        )
    accepted.intersection_update(obligation_order)

    completed_prefix: list[str] = []
    gap_seen = False
    out_of_order: list[str] = []
    for event_id in obligation_order:
        if event_id in accepted and not gap_seen:
            completed_prefix.append(event_id)
        elif event_id not in accepted:
            gap_seen = True
        else:
            out_of_order.append(event_id)
    if out_of_order:
        diagnostics.append(
            "later source events are marked complete before an earlier obligation: "
            + ", ".join(out_of_order)
        )

    completed = set(completed_prefix)
    covered_overviews: set[str] = set()
    changed = True
    while changed:
        changed = False
        for parent, children in overview_details.items():
            if parent in covered_overviews:
                continue
            if all(child in completed or child in covered_overviews for child in children):
                covered_overviews.add(parent)
                changed = True

    camera_history = _validated_camera_action_history(
        accepted_visible_camera_actions,
        source_by_id,
        current_window_index,
        diagnostics,
    )
    prop_identities = _validated_source_prop_identities(
        source_prop_identities,
        source_by_id,
        completed | covered_overviews,
        camera_history,
        diagnostics,
    )
    completed_quotes = _completed_source_event_quotes(
        completed_prefix, source_by_id, diagnostics,
    )

    return {
        "completed_source_event_ids": completed_prefix,
        "covered_overview_ids": [
            event_id for event_id in source_order if event_id in covered_overviews
        ],
        "completed_source_event_quotes": completed_quotes,
        "accepted_visible_camera_actions": camera_history,
        "source_prop_identities": prop_identities,
        "current_window_index": current_window_index,
        "diagnostics": diagnostics,
    }


def format_h3_authoritative_handoff(
    snapshot: Mapping[str, Any],
    *,
    latest_accepted_visible_camera_action: str = "",
) -> str:
    """Render the concise latest-action handoff used by native prompts."""

    if not isinstance(snapshot, Mapping) or not isinstance(
        latest_accepted_visible_camera_action, str
    ):
        return ""
    action = latest_accepted_visible_camera_action.strip()
    if not action or len(action) > _MAX_AUTHORITATIVE_HANDOFF_ACTION_CHARS:
        return ""
    return f"Continue from the visible result of: {action} Do not repeat it."


def format_h3_accepted_history_context(snapshot: Mapping[str, Any]) -> str:
    """Format bounded, provenance-labeled context for a camera writer.

    This is separate from the concise native handoff. It includes only
    completed source text, accepted visible actions from earlier windows,
    and exact source-quoted prop identity. No current position or possession
    is inferred.
    """

    if not isinstance(snapshot, Mapping):
        return ""
    current_index = snapshot.get("current_window_index")
    if (
        not isinstance(current_index, int)
        or isinstance(current_index, bool)
        or current_index < 1
        or current_index > _MAX_WINDOW_INDEX
    ):
        return ""
    raw_completed_ids = snapshot.get("completed_source_event_ids", [])
    completed_event_ids: set[str] = set()
    if isinstance(raw_completed_ids, Sequence) and not isinstance(
        raw_completed_ids, (str, bytes)
    ):
        completed_event_ids = {
            event_id for value in raw_completed_ids if (event_id := _event_key(value))
        }

    raw_history = snapshot.get("accepted_visible_camera_actions", [])
    history: list[dict[str, Any]] = []
    if isinstance(raw_history, Sequence) and not isinstance(raw_history, (str, bytes)):
        for item in raw_history:
            if not isinstance(item, Mapping) or item.get("accepted") is not True:
                continue
            window_index = item.get("window_index")
            event_ids = item.get("source_event_ids")
            visible_action = item.get("action")
            if (
                not isinstance(window_index, int)
                or isinstance(window_index, bool)
                or window_index < 1
                or window_index > _MAX_WINDOW_INDEX
                or window_index >= current_index
                or not isinstance(event_ids, Sequence)
                or isinstance(event_ids, (str, bytes))
                or any(not _event_key(value) for value in event_ids)
                or not isinstance(visible_action, str)
                or not visible_action.strip()
                or len(visible_action.strip()) > _MAX_ACCEPTED_ACTION_CHARS
            ):
                continue
            history.append({
                "window_index": window_index,
                "source_event_ids": [_event_key(value) for value in event_ids],
                "action": visible_action.strip(),
            })
    history.sort(key=lambda item: item["window_index"])
    history = history[-_MAX_ACCEPTED_ACTIONS:]
    visible_event_ids = set(completed_event_ids)
    for item in history:
        visible_event_ids.update(item["source_event_ids"])

    raw_quotes = snapshot.get("completed_source_event_quotes", [])
    source_quotes: list[dict[str, str]] = []
    if isinstance(raw_quotes, Sequence) and not isinstance(raw_quotes, (str, bytes)):
        for item in raw_quotes[-_MAX_COMPLETED_SOURCE_QUOTES:]:
            if not isinstance(item, Mapping):
                continue
            event_id = _event_key(item.get("source_event_id"))
            text = item.get("text")
            if (
                not event_id
                or event_id not in completed_event_ids
                or not isinstance(text, str)
                or not text.strip()
                or len(text.strip()) > _MAX_COMPLETED_SOURCE_QUOTE_CHARS
            ):
                continue
            source_quotes.append({"source_event_id": event_id, "text": text.strip()})

    raw_props = snapshot.get("source_prop_identities", [])
    props: list[str] = []
    if isinstance(raw_props, Sequence) and not isinstance(raw_props, (str, bytes)):
        for item in raw_props[:_MAX_SOURCE_PROP_IDENTITIES]:
            if not isinstance(item, Mapping):
                continue
            prop_id = _event_key(item.get("prop_id"))
            source_event_id = _event_key(item.get("source_event_id"))
            identity = item.get("identity")
            quote = item.get("source_quote")
            if (
                not prop_id
                or not source_event_id
                or source_event_id not in visible_event_ids
                or not isinstance(identity, str)
                or not identity.strip()
                or len(identity.strip()) > _MAX_SOURCE_PROP_IDENTITY_CHARS
                or not isinstance(quote, str)
                or not quote.strip()
                or identity.strip().casefold() not in quote.casefold()
            ):
                continue
            props.append(identity.strip())

    parts: list[str] = []
    if props:
        parts.append(
            "Source-quoted prop identity only; no position or holder is implied: "
            + "; ".join(dict.fromkeys(props))
            + "."
        )
    if source_quotes:
        parts.append(
            "Exact completed source-event text for identity and chronology context only; "
            "it does not request replay: "
            + " | ".join(item["text"] for item in source_quotes)
            + "."
        )
    if history:
        parts.append(
            "Accepted visible camera actions from earlier windows, in chronological order "
            "(history only): "
            + " | ".join(
                f"window {item['window_index']}: {item['action']}" for item in history
            )
            + ". The latest accepted visible action takes precedence; continue from it "
            "without repeating completed actions."
        )
    context = " ".join(parts)
    return context if len(context) <= _MAX_HISTORY_CONTEXT_CHARS else ""
