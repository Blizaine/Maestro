"""Shared writing guidance for Studio's unified Enhance action.

Legacy Faithful/Creative requests keep their contracts. Adaptive requests use
one writing policy throughout; scene guides add technique, never a new mode.
"""
from __future__ import annotations

import re
import json

from services.guide_loader import load_guide


def normalize_writing_style(value: object) -> str:
    style = str(value or "").strip().casefold()
    return style if style in {"faithful", "creative", "adaptive"} else "faithful"


def adaptive_dialogue_expected(prompt: str) -> bool:
    """Require a verbal request, not merely two named people or an attack."""
    from services.dialogue_writing import _speech_context, conversation_brief, dialogue_forbidden
    from services.h3_story_ledger import extract_locked_dialogue
    if dialogue_forbidden(prompt):
        return False
    if extract_locked_dialogue(prompt):
        return True
    context = _speech_context(prompt)
    context = re.sub(
        r"\b(?:camera|lens|framing|cinematography|prompt|description)\b[^.!?\n]*"
        r"\b(?:explain\w*|tell\w*|describe\w*)\b[^.!?\n]*", "", context, flags=re.I)
    return conversation_brief(context) or bool(re.search(
        r"\b(?:says?|said|saying|speaks?|speaking|spoke|asks?|asked|asking|"
        r"answers?|answered|answering|replies|replied|replying)\b", context, re.I))


def adaptive_dialogue_expansion_requested(prompt: str) -> bool:
    """Distinguish an authored line from a request to invent a spoken exchange.

    A supplied taunt, greeting or closing line does not impose a minimum word
    count on an action scene. Unscripted responses and explicit conversation
    requests still need writing, even when another turn is already supplied.
    """
    from services.dialogue_writing import conversation_brief, dialogue_forbidden, only_supplied_dialogue_requested
    from services.h3_story_ledger import extract_locked_dialogue, normalize_h3_dialogue_tags, _PLANNER_SPEECH_VERB
    if dialogue_forbidden(prompt) or only_supplied_dialogue_requested(prompt):
        return False
    source = normalize_h3_dialogue_tags(prompt)
    locked = extract_locked_dialogue(source)
    if not locked:
        return adaptive_dialogue_expected(source)
    marker = "__supplied_line__"
    for line in reversed(locked):
        source = source[:line['source_offset']] + marker + source[line['source_end']:]
    # Never infer a writing request from the words spoken by a character.
    for clause in re.split(r"(?<=[.!?;])\s+|[\r\n]+", source):
        # Performance restrictions do not request additional words. In an
        # already scripted conversation, "No overlapping speech" controls
        # turn-taking rather than imposing a fresh dialogue-writing quota.
        if re.fullmatch(
            r"\s*(?:no|without|avoid)\s+(?:overlapping|simultaneous)\s+"
            r"(?:speech|dialogue|talking)(?:\s+or\s+narration)?[.!?;]?\s*",
            clause, flags=re.I,
        ):
            continue
        if marker not in clause and conversation_brief(clause):
            return True
        cues = list(_PLANNER_SPEECH_VERB.finditer(clause))
        for index, cue in enumerate(cues):
            end = cues[index + 1].start() if index + 1 < len(cues) else len(clause)
            if marker in clause[cue.end():end]:
                continue  # The supplied words fulfill this speech request.
            if index == 0 and marker in clause[:cue.start()]:
                continue  # Postposed attribution: "Ready", Nora replies.
            return True
    return False


def adaptive_writing_guide(prompt: str) -> str:
    guides = [load_guide("enhance", "adaptive_writing")]
    # Select craft advice only. Explicit requirements in the brief always win.
    if re.search(r"\b(?:fight|fighting|duel|combat|kung[ -]?fu|wuxia|martial arts)\b", prompt, re.I):
        from promptbench.experiments import fight_choreography_enabled
        guides.append(load_guide("enhance", "craft_action_choreography"
                                 if fight_choreography_enabled() else "craft_action"))
    from services.dialogue_writing import conversation_brief
    if conversation_brief(prompt):
        guides.append(load_guide("enhance", "craft_conversation"))
    return "\n\n".join(guide for guide in guides if guide)


def is_writing_instruction(text: str) -> bool:
    """Recognize explicit requests to the writer, never quoted character speech."""
    instruction = re.match(
        r"^(?:then\s+|please\s+)?(?:write|rewrite|expand|enhance|develop)\s+"
        r"(?:the\s+|a\s+|this\s+)?(?:shot[ -]for[ -]shot|prompt|choreography|"
        r"screenplay|script|scene description)\b", str(text or "").strip(), re.I,
    )
    spoken_direction = re.match(
        r"^(?:please\s+)?(?:write|create|develop|include)\s+"
        r"(?:[\w-]+\s+){0,5}(?:dialogue|conversation|spoken exchange)\b",
        str(text or "").strip(), re.I,
    )
    # A list of production restrictions is shared guidance, not an action
    # to schedule as its own beat. Do not match physical events like "No one
    # catches the falling glass" or a character who refuses an action.
    term = r'(?:cuts?|slow motion|dialogue|speech|talking|voices?|magic|music|subtitles?|captions?|logos?|watermarks?|game UI|eye lasers?|weapons?|(?:new|extra) powers)'
    separator = r'(?:\s*[,;]\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)'
    restriction = re.fullmatch(rf'(?:no|without)\s+{term}(?:{separator}{term})*',
                              str(text or '').strip(' .!?;\t\r\n'), re.I)
    return bool(instruction or spoken_direction or restriction)


def draft_spoken_exchange(prompt: str, duration_seconds: float | None, generator,
                          *, language: str, reference_context: str = "", nsfw: bool = False) -> str:
    """Author unspecified speech before asking for H3's camera/markup prose.

    Exact user scripts and silent scenes bypass this stage. A small structured
    exchange keeps speaker ownership independent of an LLM's XML formatting.
    The returned internal brief locks these lines for the visual-writing stage;
    callers keep the original user prompt in their review/queue snapshot.
    """
    from services.dialogue_writing import (
        compact_dialogue_brief, creative_dialogue_budget,
        requested_dialogue_turns, spoken_word_count,
    )
    from services.h3_story_ledger import extract_locked_dialogue

    if not adaptive_dialogue_expected(prompt) or extract_locked_dialogue(prompt):
        return prompt
    budget = creative_dialogue_budget(prompt, duration_seconds)
    if budget is None:
        return prompt
    requested_turns = requested_dialogue_turns(prompt)
    suggested_turns = requested_turns or max(1, round(budget.target / 12))
    # An explicit turn count already supplies a useful structural floor. Do
    # not pad a complete four-turn exchange merely to clear a dense open-
    # conversation quota; require roughly two words per requested turn while
    # retaining the duration-derived hard maximum.
    minimum_words = (
        min(budget.minimum, requested_turns * 2)
        if requested_turns and compact_dialogue_brief(prompt) else budget.minimum
    )
    schema = {"type": "object", "additionalProperties": False, "required": ["turns"],
              "properties": {"turns": {"type": "array", "minItems": 1,
                  "items": {"type": "object", "additionalProperties": False,
                      "required": ["speaker", "text"], "properties": {
                          "speaker": {"type": "string"}, "text": {"type": "string"}}}}}}
    system = (
        "Write the actual spoken exchange for a short film. Return JSON turns with speaker and text. "
        "Use the user's named speakers; otherwise use stable descriptive names from the brief or inventory. "
        "Each text contains only spoken words, with no tags, names, or directions. "
        "Develop the requested topic, motive and emotional change. Preserve who asks, answers, acts, "
        "and makes the final request; do not exchange their roles. Avoid filler greetings. "
        "Make the last line fulfill the requested ending. This stage writes speech only; "
        "the camera, sound and action will be written separately."
    )
    nsfw_guidance = load_guide("enhance", "nsfw_shared") if nsfw else ""
    if nsfw_guidance:
        system += "\n\n" + nsfw_guidance
    copyedit_system = (
        "Copyedit the supplied spoken exchange. Return only a JSON object mapping each supplied "
        "L-key to revised spoken text. Each key stays with its original speaker and turn position; "
        "preserve speaker roles, intent, requested information, the exchange's progression and ending. "
        "Write complete, natural utterances in the same language. Do not return speaker names, turns, "
        "metadata, tags, or directions. Keep every line meaningful and concise enough to meet its word target. "
        f"The whole exchange must contain {minimum_words}–{budget.maximum} spoken words, "
        f"aiming for {budget.target}, including any unchanged lines."
    )
    if nsfw_guidance:
        copyedit_system += "\n\n" + nsfw_guidance
    feedback = ""
    last_error = None
    last_attempt_over_budget = False
    repair_turns: list[dict[str, str]] | None = None
    for attempt in range(1, 3):
        attempt_over_budget = repair_turns is not None
        raw = ""
        turns = None
        copyedit_error = None
        if repair_turns is not None:
            from services.h3_dialogue_writing import _shorten_generated_dialogue
            from services.llm_sampling import caller_sampling

            writer_errors = []

            def copyedit_generator(**kwargs):
                try:
                    return generator(**kwargs)
                except BaseException as error:
                    writer_errors.append(error)
                    raise

            editable_lines = [
                {"speaker": turn["speaker"], "language": language, "text": turn["text"]}
                for turn in repair_turns
            ]
            # The copyeditor receives only fixed L-keys and text fields. Its
            # result is mapped back onto the original speakers in order.
            try:
                with caller_sampling(True):
                    edited_lines = _shorten_generated_dialogue(
                        prompt, editable_lines, target_words=budget.target,
                        generate=copyedit_generator, system_prompt=copyedit_system,
                        maximum_words=budget.maximum,
                    )
                if len(edited_lines) != len(repair_turns):
                    raise ValueError("return revised text for every original turn")
                turns = [
                    {"speaker": original["speaker"], "text": edited["text"]}
                    for original, edited in zip(repair_turns, edited_lines)
                ]
                raw = json.dumps({"turns": turns}, ensure_ascii=False)
            except (ValueError, KeyError, TypeError) as error:
                if any(error is writer_error for writer_error in writer_errors):
                    raise
                copyedit_error = error
        else:
            raw = generator(
                prompt=(f"User brief: {prompt}\n\n"
                        f"Duration: {duration_seconds or 8:g} seconds. Language: {language}. "
                        f"{'Use exactly' if requested_turns else 'Write about'} {suggested_turns} concise turns "
                        f"and {budget.target} spoken words TOTAL "
                        f"across all speakers (allowed {minimum_words}–{budget.maximum} words, not per turn). "
                        "Leave time for listening reactions.\n"
                        + (f"Reference inventory: {reference_context}\n" if re.search(r'<(?:Subject|Picture|Video|Audio) \d+>', reference_context) else "")
                        + feedback),
                system_prompt=system, json_schema=schema, max_new_tokens=max(768, budget.maximum * 6),
                temperature=0.6, enable_thinking=False, frequency_penalty=0.0, presence_penalty=0.0,
            )

        try:
            if copyedit_error is not None:
                raise copyedit_error
            if turns is None:
                turns = json.loads(raw)["turns"]
            if isinstance(turns, list):
                # A model may encode an explicitly silent nod as an empty
                # dialogue row beside the one requested spoken reaction.
                # Empty speech is not a vocal event; preserve the original
                # action brief and validate the remaining audible turns.
                turns = [
                    turn for turn in turns
                    if not (
                        isinstance(turn, dict)
                        and isinstance(turn.get("text"), str)
                        and not turn["text"].strip()
                    )
                ]
            if not isinstance(turns, list) or not turns or not all(
                isinstance(turn, dict) and isinstance(turn.get("speaker"), str)
                and turn["speaker"].strip() and isinstance(turn.get("text"), str)
                and turn["text"].strip() and not re.search(r"</?d>|[\r\n]", turn["text"])
                for turn in turns
            ):
                raise ValueError("Return nonempty speaker/text pairs, without speech tags or newlines.")
            count = sum(spoken_word_count(turn["text"]) for turn in turns)
            if requested_turns and len(turns) != requested_turns:
                raise ValueError(
                    f"The user requested exactly {requested_turns} turns; your exchange had {len(turns)}."
                )
            attempt_over_budget = count > budget.maximum
            if not minimum_words <= count <= budget.maximum:
                error = ValueError(f"Your exchange had {count} words; use {minimum_words}–{budget.maximum} total, aiming for {budget.target}.")
                if attempt == 1 and attempt_over_budget:
                    repair_turns = [
                        {"speaker": " ".join(turn["speaker"].split()), "text": turn["text"].strip()}
                        for turn in turns
                    ]
                raise error
            speakers: dict[str, int] = {}
            lines = []
            for turn in turns:
                speaker = " ".join(turn["speaker"].split())
                speaker_id = speakers.setdefault(speaker.casefold(), len(speakers) + 1)
                lines.append(f'{speaker} (S{speaker_id}) says, '
                             f'<d>[{language}] {turn["text"].strip()}</d>')
            return (prompt + "\n\nSpoken script for this adaptation. Only use these lines, in this order, "
                    "beside these speakers. The story and camera scheduler owns their timing: place each line after "
                    "its requested prerequisite action and before any requested consequence. Do not invent an absolute "
                    "speech timestamp here or move a reaction ahead of the action that causes it. Named characters speak "
                    "in the scene unless "
                    "the brief explicitly requests voiceover; do not turn visible speakers into off-screen narrators. "
                    "Keep names as plain text, without angle brackets. Preserve every spoken word.\n"
                    + "\n".join(lines))
        except (ValueError, KeyError, TypeError) as error:
            last_error = error
            last_attempt_over_budget = attempt_over_budget
            reason = str(error)
            print(f"[Enhance dialogue] attempt {attempt}/2 rejected: {reason}")
            feedback = f"Revise this draft; do not start over: {raw}\nCorrection: {error}"
    selected_duration = duration_seconds or 8
    last_reason = str(last_error or "").rstrip(" .")
    diagnostic = (
        f"Last validation reason: {last_reason}. "
        f"Selected duration: {selected_duration:g} seconds."
    )
    if last_attempt_over_budget:
        diagnostic += " The draft exceeded the word limit; increase the selected duration or shorten the dialogue, then retry."
    raise ValueError(
        "The AI could not complete the requested dialogue. Retry enhancement. " + diagnostic
    ) from last_error
