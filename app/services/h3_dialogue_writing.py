"""Focused Creative speech repair against an immutable H3 story schedule."""

from copy import deepcopy
import json
import math
from typing import Any, Callable

from services.dialogue_writing import (
    conversation_brief, creative_dialogue_budget, dialogue_topic_covered,
    requested_dialogue_topics,
)
from services.dialogue_timing import (
    DIALOGUE_DEFAULT_WORDS_PER_SECOND, DIALOGUE_MAX_WORDS_PER_SECOND,
)
from services.h3_story_ledger import (
    _canonicalize_story_ledger, _dialogue_catalog, _dialogue_word_count,
    _ledger_schema, _merge_h3_cast_names, _normalize_key, _resolve_h3_cast_name,
    extract_source_events, ledger_violations,
)


def _shorten_generated_dialogue(
    prompt: str, lines: list[dict[str, Any]], *, target_words: int,
    generate: Callable[..., str], system_prompt: str,
    per_line_targets: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Copyedit only AI-written words; speaker/event ownership stays local."""
    from services.h3_window_planner import _parse_json_object

    if not lines:
        return []
    target = max(len(lines), int(target_words))
    weights = [max(1, _dialogue_word_count(line["text"])) for line in lines]
    total_weight = sum(weights)
    extra = target - len(lines)
    targets = [1 + extra * weight // total_weight for weight in weights]
    remainder_order = sorted(range(len(lines)), key=lambda i: extra * weights[i] % total_weight, reverse=True)
    for i in remainder_order[:target - sum(targets)]:
        targets[i] += 1
    if per_line_targets is not None:
        targets = per_line_targets
        target = sum(targets)
    turns = {
        f"L{i + 1}": {"speaker": line["speaker"], "language": line["language"],
                       "text": line["text"], "target_words": targets[i],
                       **({"maximum_words": targets[i]} if per_line_targets is not None else {})}
        for i, line in enumerate(lines)
    }
    request = (
            "SHORTEN AI-WRITTEN LINES. These are editable draft lines, not user quotations. "
            "Keep each speaker's intent, requested information, character voice, and the exchange's progression. "
            "Rewrite concisely; remove redundant qualifiers, introductions and repeated explanations. "
            "Do not merely change punctuation or keep almost all the original wording. "
            f"Aim for {target} spoken words TOTAL. Each turn has its share of that target below. "
            "Return only an object mapping each L-key to revised spoken text, in the same language. "
            "Write a new, complete short utterance in the same language; do not cut an unfinished sentence. "
            "Do not return speaker names, delivery notes, event IDs or other metadata. "
            "Every listed turn must still have meaningful spoken text; do not join or drop turns.\n"
            f"User brief (context only; do not write more dialogue):\n{prompt}\n"
            f"AI draft to shorten:\n{json.dumps(turns, ensure_ascii=False)}"
        )
    if per_line_targets is not None:
        request += (
            "\nEach maximum_words is a HARD per-line limit from the actual camera clock. "
            "Prefer a complete, natural sentence a word or two below that limit. "
            "Shorten hesitation and polite filler before cutting the requested information."
        )
    def write(text: str) -> str:
        return generate(
            prompt=text,
            system_prompt=system_prompt, max_new_tokens=768,
            temperature=0.3, top_p=0.88, enable_thinking=True,
            thinking_budget=512, reasoning_effort="low",
            frequency_penalty=0.0, presence_penalty=0.0,
            # llama-server's JSON grammar disables reasoning. A short editing
            # pass needs room to choose a complete concise phrase and count it;
            # grammar-capping words instead can cut the sentence mid-thought.
            json_schema=None,
        )
    raw = write(request)
    candidate = _parse_json_object(raw)
    if not isinstance(candidate, dict) or set(candidate) != set(turns):
        raise ValueError("return revised text for every L-key")
    if per_line_targets is not None:
        over = {
            key: {"language": turn["language"], "text": candidate.get(key),
                  "target_words": max(1, int(turn["maximum_words"] * 0.65)),
                  "maximum_words": turn["maximum_words"]}
            for key, turn in turns.items()
            if _dialogue_word_count(candidate.get(key)) > turn["maximum_words"]
        }
        if over:
            try:
                # Retry only the overlong turns together. One request per
                # line multiplied writer latency for dialogue-heavy windows.
                repaired = _parse_json_object(write(
                    "SHORTEN OVERLONG SPOKEN TURNS. Rewrite each listed line toward its target_words; "
                    "never exceed its maximum_words. Express its core question or answer as a NEW "
                    "complete natural utterance. Omit greetings, repetition and decorative phrasing. "
                    "Keep essential information and each turn's language. Do not truncate sentences, "
                    "join turns, or return acting directions. Return only JSON mapping the listed "
                    "L-keys to their revised spoken text.\n"
                    f"Editable turns:\n{json.dumps(over, ensure_ascii=False)}"
                ))
            except Exception as error:
                print(f"[MiniMax H3] Camera dialogue copyedit retry: {error}")
                repaired = None  # Keep a useful first edit if the retry fails.
            if isinstance(repaired, dict):
                for key in over:
                    if isinstance(repaired.get(key), str) and repaired[key].strip() and (
                        _dialogue_word_count(repaired[key]) < _dialogue_word_count(candidate.get(key))
                    ):
                        candidate[key] = repaired[key]
    result = deepcopy(lines)
    for i, line in enumerate(result, 1):
        text = candidate[f"L{i}"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each revised turn needs spoken text")
        line["text"] = text.strip()
    return result


def fit_camera_dialogue(
    prompt: str, segment: dict[str, Any], catalog: list[dict[str, Any]],
    generated_dialogue: list[dict[str, Any]], *,
    generate: Callable[..., str], system_prompt: str,
) -> list[str]:
    """Fit AI text to its actual shot, retaining every speaker/event and quote.

    A whole-window word budget is insufficient when arrival and travel shots
    consume part of it. One focused copyedit handles all squeezed AI turns;
    neither immutable quotations nor the writer's physical staging are cut.
    """
    from services.h3_story_ledger import _apply_h3_filmable_shot_clock

    by_id = {item["dialogue_id"]: item for item in catalog}
    editable = {item["dialogue_id"]: item for item in generated_dialogue}
    shots = segment.get("shots") or []
    if not any(shot.get("dialogue") for shot in shots):
        # Silent action already has an authored/filmable clock. A dialogue
        # copyedit must not redistribute its deliberate wind-ups and impacts.
        return []

    def reflow() -> None:
        if not shots:
            return
        if all(
            sum(_dialogue_word_count(by_id[line["dialogue_id"]]["text"])
                for line in shot.get("dialogue", [])) / DIALOGUE_MAX_WORDS_PER_SECOND
            + len(shot.get("dialogue", [])) * 0.2
            <= float(shot["end_seconds"]) - float(shot["start_seconds"]) + 0.01
            for shot in shots
        ):
            return  # Preserve an already-valid clock, including opening speech.
        assignments = [[{
            "description": shot.get("action", ""),
            "dialogue_ids": [line["dialogue_id"] for line in shot.get("dialogue", [])],
        }] for shot in shots]
        _apply_h3_filmable_shot_clock(
            shots, assignments, by_id, duration=float(shots[-1]["end_seconds"]), only_when_squeezed=True,
        )

    # A shorter turn may have freed enough time elsewhere. Rebalance before
    # requesting more writing, and again after the edit; stale per-shot caps
    # must not reject words that now fit the complete physical/speech clock.
    reflow()
    lines, limits = [], []
    for shot in segment.get("shots") or []:
        local = [by_id[line["dialogue_id"]] for line in shot.get("dialogue", [])]
        seconds = float(shot["end_seconds"]) - float(shot["start_seconds"])
        maximum = max(0, math.floor((seconds - len(local) * 0.2 + 0.002) * DIALOGUE_MAX_WORDS_PER_SECOND))
        if sum(_dialogue_word_count(line["text"]) for line in local) <= maximum:
            continue
        writable = [line for line in local if line["dialogue_id"] in editable]
        locked_words = sum(_dialogue_word_count(line["text"]) for line in local if line["dialogue_id"] not in editable)
        available = maximum - locked_words
        if not writable or available < len(writable):
            continue  # Preserve exact supplied text; report the remaining squeeze below.
        weights = [max(1, _dialogue_word_count(line["text"])) for line in writable]
        # Round downward so each edited turn has its own safe share, including
        # when a legacy camera shot is split at a speaker change later.
        targets = [max(1, math.floor(available * weight / sum(weights))) for weight in weights]
        if sum(targets) > available:
            continue
        lines.extend(writable)
        limits.extend(targets)
    if lines:
        try:
            revised = _shorten_generated_dialogue(
                prompt, lines, target_words=sum(limits), per_line_targets=limits,
                generate=generate, system_prompt=system_prompt,
            )
            for source, candidate in zip(lines, revised):
                if _dialogue_word_count(candidate["text"]) < _dialogue_word_count(source["text"]):
                    # A shorter line may fit after time freed by another line
                    # is redistributed; the old per-shot caps are not final.
                    source["text"] = candidate["text"]
                    editable[source["dialogue_id"]]["text"] = candidate["text"]
        except Exception as error:
            print(f"[MiniMax H3] Camera dialogue copyedit: {error}")
        reflow()
    squeezed = []
    for shot in segment.get("shots") or []:
        local = [by_id[line["dialogue_id"]] for line in shot.get("dialogue", [])]
        words = sum(_dialogue_word_count(line["text"]) for line in local)
        seconds = float(shot["end_seconds"]) - float(shot["start_seconds"])
        if words and words / DIALOGUE_MAX_WORDS_PER_SECOND + len(local) * 0.2 > seconds + 0.01:
            squeezed.append(str(shot["shot"]))
    if not squeezed:
        return []
    return [
        f"Window {segment['segment']}, shot(s) {', '.join(squeezed)} still need more speaking time. "
        "The complete draft is saved for review; shorten the lines or adjust the shot timing."
    ]


def creative_dialogue_windows(
    prompt: str, ledger: dict[str, Any], locked_dialogue: list[dict[str, Any]], durations: list[float],
    *, camera_segments: list[dict[str, Any]] | None = None,
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
        if ledger.get("_story_time"):
            from promptbench.story_time import speech_budget
            budget = speech_budget(ledger, index, budget)
        if camera_segments is not None and index <= len(camera_segments):
            shots = camera_segments[index - 1].get("shots") or []
            speaking = [shot for shot in shots if shot.get("dialogue")]
            if speaking:
                speech_seconds = sum(max(0.0, float(shot["end_seconds"]) - float(shot["start_seconds"])) for shot in speaking)
                turns = sum(len(shot["dialogue"]) for shot in speaking)
                maximum = max(1, math.floor((speech_seconds - turns * 0.2 + 0.002) * DIALOGUE_MAX_WORDS_PER_SECOND))
                target = min(budget.target, maximum, max(1, round(speech_seconds * DIALOGUE_DEFAULT_WORDS_PER_SECOND)))
                # Copyediting requests natural phrasing below hard per-line
                # maxima. A one/two-word miss of a preferred density target
                # must not reject that successful edit. This small allowance
                # never raises the hard speech ceiling.
                minimum = min(budget.minimum, max(1, math.ceil(target * 0.9) - 2))
                budget = type(budget)(minimum, target, min(budget.maximum, maximum))
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
        writing_notes = []
        if words > budget.maximum:
            problems.append(f"{words} spoken words; aim for {budget.target} ({budget.minimum}–{budget.maximum} allowed)")
        elif not words:
            problems.append("the requested spoken exchange has no dialogue")
        elif words < budget.minimum:
            writing_notes.append(f"{words} spoken words; preferred target {budget.target}")
        if missing_topics:
            problems.append("missing spoken talking points: " + "; ".join(missing_topics))
        windows.append({
            "segment": index, "duration_seconds": duration, "spoken_words": words,
            "minimum_words": budget.minimum, "target_words": budget.target, "maximum_words": budget.maximum,
            "writing_budget": budget.instruction(), "locked_lines": local_locked,
            "generated_word_maximum": max(0, budget.maximum - locked_words),
            "story_beats": [item["description"] for item in beats],
            **({"reserved_action_clock": ledger["_story_time"][index - 1],
                "required_outcome": ledger.get("required_final_outcome"),
                "instruction": "Write for the remaining speech time. Keep the planned movements and final outcome; do not write extra turns that crowd them out."}
               if ledger.get("_story_time") else {}),
            "source_events": [event for event in source_events
                              if any(event["event_id"] in beat.get("source_event_ids", []) for beat in beats)],
            "current_dialogue": [catalog[did] for did in ids if did not in locked_ids],
            "required_spoken_topics": required_topics, "missing_topics": missing_topics,
            "problems": problems, "writing_notes": writing_notes,
        })
    return windows


def complete_creative_dialogue(
    prompt: str, ledger: dict[str, Any], *, canonical_ledger: dict[str, Any],
    locked_dialogue: list[dict[str, Any]], durations: list[float],
    generate: Callable[..., str], system_prompt: str,
    copyedit_system_prompt: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Develop each window; reserve retries for actual speech or story problems."""
    from services.h3_window_planner import _parse_json_object

    cast_names = _merge_h3_cast_names(
        list((canonical_ledger.get("source_intent") or {}).get("cast_names") or []),
        [item["speaker"] for item in locked_dialogue if item.get("speaker")], prompt=prompt,
    )
    # An invented speaker in a failed draft must not become an allowed cast member.
    dialogue_schema = deepcopy(_ledger_schema(
        len(durations), source_event_count=len(extract_source_events(prompt)), locked_dialogue_count=len(locked_dialogue),
        allow_generated_dialogue=True,
    )["properties"]["generated_dialogue"])
    dialogue_schema.update(minItems=1, maxItems=6)
    obligations = [item for item in creative_dialogue_windows(prompt, ledger, locked_dialogue, durations)
                   if item["problems"] or item["writing_notes"]]
    for obligation in obligations:
        number = obligation["segment"]
        schema = deepcopy(dialogue_schema)
        schema["items"]["properties"]["segment"] = {"type": "integer", "enum": [number]}
        if cast_names:
            schema["items"]["properties"]["speaker"] = {"type": "string", "enum": cast_names}
        feedback = "; ".join(obligation["problems"] + obligation["writing_notes"])
        previous_attempt = ""
        def fit_score(item):
            return (int(not item["spoken_words"]), max(0, item["spoken_words"] - item["maximum_words"]),
                    len(item["missing_topics"]), max(0, item["minimum_words"] - item["spoken_words"]),
                    abs(item["spoken_words"] - item["target_words"]))

        best_draft = None
        best_score = fit_score(obligation)
        copyedited = False
        overlong_lines = obligation["current_dialogue"] if obligation["spoken_words"] > obligation["maximum_words"] else []
        # A preferred word count is writing guidance, not a reason to keep
        # regenerating an otherwise valid exchange. Give sparse writing one
        # development pass; retain repairs for malformed output, missing
        # topics or invalid timing introduced by that pass.
        for attempt in range(3):
            try:
                adjacent = [
                    {"segment": item["segment"], "speaker": item["speaker"], "text": item["text"]}
                    for item in ledger.get("generated_dialogue", []) if abs(int(item["segment"]) - number) == 1
                ]
                if overlong_lines and not copyedited and (attempt or not obligation["missing_topics"]):
                    copyedited = True
                    edited = _shorten_generated_dialogue(
                        prompt, overlong_lines,
                        target_words=max(1, obligation["target_words"] - sum(
                            _dialogue_word_count(item["text"]) for item in obligation["locked_lines"])),
                        generate=generate, system_prompt=copyedit_system_prompt or system_prompt,
                    )
                    raw = json.dumps({"generated_dialogue": edited}, ensure_ascii=False)
                elif attempt < 2:
                    raw = generate(
                    prompt=(
                        f"FIT THE SPOKEN SCRIPT: write ONLY window {number} of {len(durations)}. "
                        "Revise its AI-authored speech to fit the supplied total word budget: shorten an overlong "
                        "exchange, develop an incomplete one, and keep a complete response that is already in range. "
                        "Preserve the order, speaker roles, meaning and outcome of the local discussion. "
                        "Use specific ideas and natural responses, not generic praise or filler. Speak the required "
                        "talking points, including their requested details; visual descriptions do not count. "
                        "Do not invent product capabilities, performance claims, or facts beyond the brief. "
                        "Keep cast, language, scene facts and exact locked lines. Do not repeat locked lines in your output. "
                        "The word budget is the total across ALL speakers, including locked lines. "
                        "Each line's source_event_id identifies the supplied local event where it is spoken; "
                        "use an empty string only for a connective response without a specific source event. "
                        "Return only {\"generated_dialogue\": [...]} with speaker, language, delivery, text, segment, source_event_id.\n"
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
                else:
                    break
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
                    preserve_adaptation=True,
                )
                checked = next(item for item in creative_dialogue_windows(prompt, compiled, locked_dialogue, durations) if item["segment"] == number)
                # Validate hard story/speaker constraints before considering a
                # draft's preferred dialogue density. A shorter safe exchange
                # is still reviewable if both attempts miss the writing target;
                # retaining an overlong AI draft instead makes the exact-quote
                # scheduler fail on words the user never supplied.
                violations = ledger_violations(
                    prompt, compiled, segment_count=len(durations), locked_dialogue=locked_dialogue,
                    expect_dialogue=False, allow_generated_dialogue=True,
                    require_dialogue_per_segment=False,
                )
                if violations:
                    raise ValueError("; ".join(violations))
                overlong_lines = (
                    checked["current_dialogue"]
                    if checked["spoken_words"] > checked["maximum_words"] else []
                )
                score = fit_score(checked)
                if score < best_score:
                    best_draft, best_score = compiled, score
                if checked["problems"]:
                    feedback = "; ".join(checked["problems"] + checked["writing_notes"])
                    words = checked["spoken_words"]
                    if words < checked["minimum_words"]:
                        feedback += (
                            f". Try adding {checked['target_words'] - words} spoken words if useful. Develop an idea or a listener's response "
                            "instead of returning another reply of the same length"
                        )
                    elif words > checked["maximum_words"]:
                        feedback += (
                            f". Remove at least {words - checked['maximum_words']} spoken words; "
                            f"aim to remove {words - checked['target_words']}. Keep locked lines verbatim"
                        )
                    raise ValueError(feedback)
                if not checked["writing_notes"]:
                    best_draft = compiled
                break
            except Exception as error:
                feedback = str(error)
                print(f"[MiniMax H3] Creative dialogue window {number}, attempt {attempt + 1}: {feedback}")
        if best_draft is not None:
            ledger = {**ledger, "beats": best_draft["beats"],
                      "generated_dialogue": best_draft["generated_dialogue"]}
    warnings = []
    for item in creative_dialogue_windows(prompt, ledger, locked_dialogue, durations):
        if item["problems"]:
            warnings.append(
                f"AI dialogue needs review in window {item['segment']}: " + "; ".join(item["problems"])
                + ". Automatic writing repair was unsuccessful; review or enhance again before generating."
            )
    return ledger, warnings
