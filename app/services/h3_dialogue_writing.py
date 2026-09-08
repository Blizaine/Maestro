"""Focused Creative speech repair against an immutable H3 story schedule."""

from copy import deepcopy
import json
from typing import Any, Callable

from services.dialogue_writing import (
    conversation_brief, creative_dialogue_budget, dialogue_topic_covered,
    requested_dialogue_topics,
)
from services.h3_story_ledger import (
    _canonicalize_story_ledger, _dialogue_catalog, _dialogue_word_count,
    _ledger_schema, _merge_h3_cast_names, _normalize_key, _resolve_h3_cast_name,
    extract_source_events, ledger_violations,
)


def creative_dialogue_windows(
    prompt: str, ledger: dict[str, Any], locked_dialogue: list[dict[str, Any]], durations: list[float],
) -> list[dict[str, Any]]:
    """Audit spoken text; camera descriptions cannot satisfy talking points."""
    catalog = {item["dialogue_id"]: item for item in _dialogue_catalog(ledger, locked_dialogue)}
    all_spoken = " ".join(item["text"] for item in catalog.values())
    source_events = extract_source_events(prompt)
    topic_owners: dict[int, list[str]] = {}
    topics = requested_dialogue_topics(prompt)
    for position, topic in enumerate(topics):
        event_ids = {item["event_id"] for item in source_events if _normalize_key(topic) in _normalize_key(item["text"])}
        owners = [int(beat["segment"]) for beat in ledger.get("beats", []) if event_ids & set(beat.get("source_event_ids", []))]
        # Derived beats may not retain a literal bullet. Allocate unmatched
        # talking points chronologically instead of dropping them.
        owner = min(owners) if owners else min(len(durations), 1 + position * len(durations) // max(1, len(topics)))
        topic_owners.setdefault(owner, []).append(topic)
    windows = []
    locked_ids = {item["dialogue_id"] for item in locked_dialogue}
    for index, duration in enumerate(durations, 1):
        budget = creative_dialogue_budget(prompt, duration)
        if not budget:
            continue
        beats = [item for item in ledger.get("beats", []) if item.get("segment") == index]
        ids = list(dict.fromkeys(did for beat in beats for did in beat.get("dialogue_ids", []) if did in catalog))
        lines = [catalog[did] for did in ids]
        if not conversation_brief(prompt) and not lines and len(durations) > 1:
            continue
        local_locked = [catalog[did] for did in ids if did in locked_ids]
        locked_words = sum(_dialogue_word_count(item["text"]) for item in local_locked)
        if locked_words > budget.maximum:
            continue  # The render scheduler splits long immutable quotes.
        words = sum(_dialogue_word_count(item["text"]) for item in lines)
        required_topics = topic_owners.get(index, [])
        missing_topics = [topic for topic in required_topics if not dialogue_topic_covered(topic, all_spoken)]
        problems = []
        if words < budget.minimum or words > budget.maximum:
            problems.append(f"{words} spoken words; aim for {budget.target} ({budget.minimum}–{budget.maximum} allowed)")
        if missing_topics:
            problems.append("missing spoken talking points: " + "; ".join(missing_topics))
        windows.append({
            "segment": index, "duration_seconds": duration, "spoken_words": words,
            "minimum_words": budget.minimum, "target_words": budget.target, "maximum_words": budget.maximum,
            "writing_budget": budget.instruction(), "locked_lines": local_locked,
            "generated_word_maximum": max(0, budget.maximum - locked_words),
            "story_beats": [item["description"] for item in beats],
            "current_dialogue": [catalog[did] for did in ids if did not in locked_ids],
            "required_spoken_topics": required_topics, "missing_topics": missing_topics, "problems": problems,
        })
    return windows


def complete_creative_dialogue(
    prompt: str, ledger: dict[str, Any], *, canonical_ledger: dict[str, Any],
    locked_dialogue: list[dict[str, Any]], durations: list[float],
    generate: Callable[..., str], system_prompt: str,
) -> tuple[dict[str, Any], list[str]]:
    """Repair windows independently, retrying once with precise feedback."""
    from services.h3_window_planner import _parse_json_object

    cast_names = _merge_h3_cast_names(
        list((canonical_ledger.get("source_intent") or {}).get("cast_names") or []),
        [item["speaker"] for item in locked_dialogue if item.get("speaker")], prompt=prompt,
    )
    # An invented speaker in a failed draft must not become an allowed cast member.
    dialogue_schema = deepcopy(_ledger_schema(
        len(durations), source_event_count=1, locked_dialogue_count=len(locked_dialogue),
        allow_generated_dialogue=True,
    )["properties"]["generated_dialogue"])
    dialogue_schema.update(minItems=1, maxItems=6)
    obligations = [item for item in creative_dialogue_windows(prompt, ledger, locked_dialogue, durations) if item["problems"]]
    for obligation in obligations:
        number = obligation["segment"]
        schema = deepcopy(dialogue_schema)
        schema["items"]["properties"]["segment"] = {"type": "integer", "enum": [number]}
        if cast_names:
            schema["items"]["properties"]["speaker"] = {"type": "string", "enum": cast_names}
        feedback = "; ".join(obligation["problems"])
        previous_attempt = ""
        for attempt in range(2):
            try:
                adjacent = [
                    {"segment": item["segment"], "speaker": item["speaker"], "text": item["text"]}
                    for item in ledger.get("generated_dialogue", []) if abs(int(item["segment"]) - number) == 1
                ]
                raw = generate(
                    prompt=(
                        f"COMPLETE SPARSE CREATIVE DIALOGUE: write ONLY window {number} of {len(durations)}. "
                        "Replace its AI-authored speech with a developed exchange that advances the discussion. "
                        "Use specific ideas and natural responses, not generic praise or filler. Speak the required "
                        "talking points, including their requested details; visual descriptions do not count. "
                        "Do not invent product capabilities, performance claims, or facts beyond the brief. "
                        "Keep cast, language, scene facts and exact locked lines. Do not repeat locked lines in your output. "
                        "The word budget is the total across ALL speakers, including locked lines. "
                        "Return only {\"generated_dialogue\": [...]} with speaker, language, delivery, text, segment.\n"
                        f"User brief:\n{prompt}\nCast: {json.dumps(cast_names, ensure_ascii=False)}\n"
                        f"Adjacent dialogue (do not repeat): {json.dumps(adjacent, ensure_ascii=False)}\n"
                        f"Window to complete:\n{json.dumps(obligation, ensure_ascii=False)}\n"
                        f"Validation feedback: {feedback}\n"
                        + (f"Rejected attempt (rewrite to fix the feedback):\n{previous_attempt}\n" if previous_attempt else "")
                    ),
                    system_prompt=system_prompt, max_new_tokens=1200,
                    temperature=0.45 if attempt == 0 else 0.3, top_p=0.88, enable_thinking=False,
                    frequency_penalty=0.2, presence_penalty=0.1,
                    json_schema={
                        "type": "object", "properties": {"generated_dialogue": schema},
                        "required": ["generated_dialogue"], "additionalProperties": False,
                    },
                )
                previous_attempt = str(raw or "")[:8000]
                candidate = _parse_json_object(raw)
                additions = candidate.get("generated_dialogue") if isinstance(candidate, dict) else None
                if not isinstance(additions, list) or not 1 <= len(additions) <= 6:
                    raise ValueError("return one to six authored speaking turns for this window")
                authored_texts = {_normalize_key(item["text"]) for item in locked_dialogue + ledger.get("generated_dialogue", [])
                                  if item in locked_dialogue or item.get("segment") != number}
                for item in additions:
                    if not isinstance(item, dict) or not all(str(item.get(key) or "").strip() for key in ("speaker", "language", "delivery", "text")):
                        raise ValueError("each turn needs speaker, language, delivery, and text")
                    if item.get("segment") != number:
                        raise ValueError(f"write only segment {number}")
                    speaker = _resolve_h3_cast_name(item["speaker"], cast_names)
                    if cast_names and speaker not in cast_names:
                        raise ValueError("dialogue introduced a speaker outside the requested cast")
                    item["speaker"] = speaker
                    text_key = _normalize_key(item["text"])
                    if text_key in authored_texts:
                        raise ValueError("dialogue repeated a locked line or a line from another window")
                    authored_texts.add(text_key)
                replacement = deepcopy(ledger)
                replacement["generated_dialogue"] = sorted(
                    [item for item in ledger.get("generated_dialogue", []) if item["segment"] != number] + additions,
                    key=lambda item: item["segment"],
                )
                compiled = _canonicalize_story_ledger(
                    prompt, canonical_ledger, replacement, locked_dialogue=locked_dialogue,
                    segment_count=len(durations), allow_generated_dialogue=True,
                )
                checked = next(item for item in creative_dialogue_windows(prompt, compiled, locked_dialogue, durations) if item["segment"] == number)
                if checked["problems"]:
                    feedback = "; ".join(checked["problems"])
                    words = checked["spoken_words"]
                    if words < checked["minimum_words"]:
                        feedback += (
                            f". Your rejected script needs at least {checked['minimum_words'] - words} MORE spoken words; "
                            f"aim to add {checked['target_words'] - words}. Develop an idea or a listener's response "
                            "instead of returning another reply of the same length"
                        )
                    elif words > checked["maximum_words"]:
                        feedback += (
                            f". Remove at least {words - checked['maximum_words']} spoken words; "
                            f"aim to remove {words - checked['target_words']}. Keep locked lines verbatim"
                        )
                    raise ValueError(feedback)
                # Check immutable events and exact lines without requiring the
                # other windows to have completed their writing repairs yet.
                violations = ledger_violations(
                    prompt, compiled, segment_count=len(durations), locked_dialogue=locked_dialogue,
                    expect_dialogue=False, allow_generated_dialogue=True,
                    require_dialogue_per_segment=False,
                )
                if violations:
                    raise ValueError("; ".join(violations))
                ledger = {**ledger, "beats": compiled["beats"], "generated_dialogue": compiled["generated_dialogue"]}
                break
            except Exception as error:
                feedback = str(error)
                print(f"[MiniMax H3] Creative dialogue window {number}, attempt {attempt + 1}: {feedback}")
    warnings = []
    for item in creative_dialogue_windows(prompt, ledger, locked_dialogue, durations):
        if item["problems"]:
            warnings.append(
                f"AI dialogue needs review in window {item['segment']}: " + "; ".join(item["problems"])
                + ". Automatic writing repair was unsuccessful; review or enhance again before generating."
            )
    return ledger, warnings
