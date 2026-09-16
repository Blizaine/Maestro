"""Action-first candidate. Text-only experiment, not the production default."""
from copy import deepcopy
import json
import math


def schedule_request(schema, *, prompt, durations, canonical, events, references):
    schema = deepcopy(schema)
    schema["properties"]["generated_dialogue"].update(minItems=0, maxItems=0)
    beat = schema["properties"]["beats"]["items"]
    beat["properties"]["action_seconds"] = {"type": "number", "minimum": 0, "maximum": max(durations)}
    beat["required"].append("action_seconds")
    system = (
        "Plan a complete film scene before writing any spoken lines. Return the requested JSON. "
        "Keep the user's identities, event order, facts, tone and outcome. Develop unspecified staging naturally. "
        "Group the ordered source events into one to three beats per window; every E-id appears once, in order. "
        "Put the last supplied event in the final window. Plan backward from its visible outcome so the ending happens on camera. "
        "Use description for the complete physical progression, including movements required between conversations. "
        "state_after records concrete positions and object states. An invitation is not an entrance: "
        "show the requested movement before dialogue at the destination. Avoid unnecessary establishing or reaction beats. "
        "For each beat action_seconds reserves ONLY time that cannot also contain speech: arrivals, travel, "
        "handoffs, necessary silent reactions and the visible ending. Speaking while seated or gesturing needs no extra silent time. "
        "Keep that reserved time economical and filmable. The sum must leave room for all requested exchanges in each window. "
        "Do not write dialogue yet: generated_dialogue and dialogue_ids are empty. A second writer gets only the "
        "remaining speech time. Describe the intent and outcome of each requested exchange, without inventing its words. "
        "Shared continuity describes stable facts, never future actions that should replay in each shot. "
        "No additional characters, unrequested powers, effects, or changes to the ending."
    )
    request = (
        f"User brief:\n{prompt}\n\nWindow durations: {json.dumps(durations)} seconds.\n"
        f"Ordered source events: {json.dumps(events, ensure_ascii=False)}\n"
        f"Cast and blocking: {canonical.get('subject_continuity')}\n"
        f"Opening: {canonical.get('initial_state')}\n"
        f"Reference inventory: {references}\n"
        "Describe the final visible outcome first in required_final_outcome, then schedule every event to reach it. "
        "Do not use up the first window on an entrance and greeting if later action and conversation would be crowded out."
    )
    return request, system, schema


def retain_reservations(ledger, candidate):
    """Preserve the same clock through dialogue-only recanonicalization."""
    if not isinstance(candidate, dict):
        return
    if candidate.get("_story_time"):
        ledger["_story_time"] = deepcopy(candidate["_story_time"])
    for target, source in zip(ledger.get("beats", []), candidate.get("beats", [])):
        value = source.get("_action_seconds", source.get("action_seconds"))
        if value is not None:
            target["_action_seconds"] = value


def reserve_time(ledger, durations):
    reservations = []
    for index, duration in enumerate(durations, 1):
        beats = [b for b in ledger["beats"] if b["segment"] == index]
        values = [b.get("_action_seconds", 0) for b in beats]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Action reservations must be finite nonnegative seconds.")
        reserved = sum(values)
        if reserved >= duration - 1:
            raise ValueError(f"Window {index} reserves {reserved:g}s of {duration:g}s before its requested speech.")
        reservations.append({"segment": index, "action_seconds": reserved,
                             "speech_seconds": duration - reserved,
                             "required_state": beats[-1]["state_after"]})
    ledger["_story_time"] = reservations


def speech_budget(ledger, index, original):
    from services.dialogue_writing import DialogueBudget
    from services.dialogue_timing import DIALOGUE_DEFAULT_WORDS_PER_SECOND, DIALOGUE_MAX_WORDS_PER_SECOND

    clock = next((x for x in ledger.get("_story_time", []) if x["segment"] == index), None)
    if not clock:
        return original
    # Leave turn changes and natural pauses inside the speaking allocation.
    available = max(0, clock["speech_seconds"] - 0.8)
    maximum = min(original.maximum, max(1, math.floor(available * DIALOGUE_MAX_WORDS_PER_SECOND)))
    target = min(maximum, original.target, max(1, math.floor(available * DIALOGUE_DEFAULT_WORDS_PER_SECOND)))
    return DialogueBudget(min(original.minimum, max(1, math.ceil(target * 0.8))), target, maximum)


def reserve_camera_actions(shots, beats):
    """The same action allowance survives expansion into multiple camera cards."""
    for index, beat in enumerate(beats, 1):
        reserved = beat.get("_action_seconds", 0)
        if reserved <= 0:
            continue
        actions = [s for s in shots if s.get("event_indices") == [index] and not s.get("dialogue_ids")]
        if not actions:
            raise ValueError(f"Event {index} needs its reserved physical action before the spoken performance.")
        for shot in actions:
            shot["_action_floor"] = reserved / len(actions)


def action_first_clock(shots, weights, total, floors):
    """Do not borrow reserved action time to fit an overlong AI exchange."""
    from services.h3_story_ledger import _h3_filmable_shot_durations

    reserved = [max(0.0, s.get("_action_floor", 0)) for s in shots]
    if not any(reserved):
        return None
    remaining = total - sum(reserved)
    speech = [i for i, seconds in enumerate(reserved) if not seconds]
    if remaining <= 0 or not speech:
        return None
    fitted = _h3_filmable_shot_durations([weights[i] for i in speech], remaining,
                                       minimums=[floors[i] for i in speech])
    result = list(reserved)
    for i, seconds in zip(speech, fitted):
        result[i] = seconds
    return result
