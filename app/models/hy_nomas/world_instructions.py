# Derived from Hy-Nomas prompt instructions (https://github.com/Tencent-Hunyuan/Hy-Nomas), Copyright (C) 2026 Tencent, Apache-2.0.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0.
"""Pure parser for the model's bracketed move instructions."""

from __future__ import annotations

import math
import re

_COMMAND_KEYS = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "look_left": "left",
    "look_right": "right",
    "look_up": "up",
    "look_down": "down",
    "wait": None,
}
_TAG = re.compile(r"\[([^\]]+)\]")
_DURATION = re.compile(r"^(.*?)(?:\s+([0-9]+(?:\.[0-9]+)?)\s*s?)?$", re.IGNORECASE)


def _parse_instruction_line(line):
    tags = list(_TAG.finditer(line))
    if not tags:
        return None
    recognized = False
    keys, event_text, seconds = [], "", None
    for index, match in enumerate(tags):
        body = match.group(1).strip()
        parts = [part.strip() for part in body.split("+") if part.strip()]
        line_duration = None
        parsed_parts = []
        for part in parts:
            duration_match = _DURATION.fullmatch(part)
            command, duration = duration_match.group(1).strip(), duration_match.group(2)
            if duration is not None:
                value = float(duration)
                if not math.isfinite(value) or value <= 0 or value > 60:
                    raise ValueError("Hy-Nomas move duration must be greater than zero and at most 60 seconds.")
                line_duration = value
            parsed_parts.append(command.lower())
        if parsed_parts == ["event"]:
            recognized = True
            event_text = line[match.end():].strip()
            if not event_text:
                raise ValueError("An [Event] instruction needs text after its tag.")
            if index != len(tags) - 1:
                raise ValueError("[Event] must be the last bracketed command on its line.")
            break
        if all(part in _COMMAND_KEYS for part in parsed_parts):
            recognized = True
            keys.extend(_COMMAND_KEYS[part] for part in parsed_parts if _COMMAND_KEYS[part] is not None)
            if line_duration is not None:
                seconds = line_duration
        elif not recognized:
            if line.strip() == f"[{body}]":
                raise ValueError(f"Unknown Hy-Nomas move command: {body!r}.")
            return None
        else:
            raise ValueError(f"Unknown Hy-Nomas move command: {body!r}.")
    if not recognized:
        return None
    if not event_text:
        # Multiple bracket groups on a line are one combined move. The event
        # marker, when present, is deliberately last by the prompt contract.
        remainder = _TAG.sub("", line).strip()
        if remainder:
            raise ValueError("Text after a move command must follow an [Event] tag.")
    return {"text": line.strip(), "keys": keys, "event": event_text, "seconds": seconds}


def split_instructions(prompt):
    """Split a description and subsequent bracketed move lines.

    Lines without recognized move tags before the first instruction remain part
    of the description. After instructions begin, untagged lines are rejected
    instead of silently disappearing from the user's prompt.
    """
    if not isinstance(prompt, str):
        raise ValueError("Hy-Nomas prompt must be text.")
    lines = prompt.splitlines()
    description, instructions = [], []
    started = False
    for line in lines:
        parsed = _parse_instruction_line(line)
        if parsed is None:
            if started and line.strip():
                raise ValueError("Every line after Hy-Nomas move instructions begin must contain a bracketed command.")
            description.append(line)
            continue
        started = True
        instructions.append(parsed)
    return "\n".join(description).strip(), instructions
