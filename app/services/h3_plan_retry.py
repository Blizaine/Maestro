"""Resumable H3 camera planning without rewriting accepted windows."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any


class H3PlanRetryError(ValueError):
    """A saved plan cannot safely be repaired with the current inputs."""


def window_warning_number(message: str) -> int | None:
    match = re.match(r"(?:AI dialogue needs review in window |Window )(\d+)\b", str(message), re.I)
    return int(match.group(1)) if match else None


def retry_fingerprint(signature: str, image_paths: list[str] | None, nsfw: bool) -> str:
    payload = [signature, list(image_paths or []), bool(nsfw)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def retryable_windows(plan: Any) -> list[int]:
    if not isinstance(plan, dict) or not isinstance(plan.get("camera_checkpoint"), dict):
        return []
    count = len(plan.get("windows") or [])
    flagged = plan["camera_checkpoint"].get("review_windows")
    if isinstance(flagged, list):
        return sorted({number for number in flagged if type(number) is int and 1 <= number <= count})
    # Compatibility with checkpoints written before structured review status.
    return sorted({number for warning in plan.get("planning_warnings") or []
                   if (number := window_warning_number(warning)) is not None and 1 <= number <= count})


def validate_retry_plan(plan: dict | None, *, fingerprint: str, count: int) -> dict | None:
    if plan is None:
        return None
    checkpoint = plan.get("camera_checkpoint") if isinstance(plan, dict) else None
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
        raise H3PlanRetryError("This saved draft does not support window repair. Create a new draft to enable it.")
    if plan.get("retry_fingerprint") != fingerprint:
        raise H3PlanRetryError("The prompt, references or timing changed. Create a new draft for these settings.")
    context = checkpoint.get("context")
    if (not isinstance(context, dict)
            or len(plan.get("windows") or []) != count
            or len(checkpoint.get("segments") or []) != count
            or len(context.get("durations") or []) != count):
        raise H3PlanRetryError("The saved window schedule is incomplete. Create a new draft.")
    targets = retryable_windows(plan)
    if not targets:
        raise H3PlanRetryError("This draft has no flagged windows to retry. Generate it or create a new draft.")
    return {**deepcopy(checkpoint), "retry_windows": targets}


def finish_retry_plan(result: dict, previous: dict | None, fingerprint: str) -> dict:
    result["retry_fingerprint"] = fingerprint
    if previous is not None:
        if len(result["windows"]) != len(previous["windows"]):
            raise H3PlanRetryError("The repair returned an incomplete schedule. Your saved draft is unchanged.")
        targets = set(retryable_windows(previous))
        # Compiling one repaired window must not normalize or otherwise change
        # any accepted prompt, including its timing and presentation metadata.
        result["windows"] = [window if index + 1 in targets else deepcopy(previous["windows"][index])
                             for index, window in enumerate(result["windows"])]
        result["window_prompts"] = [window["prompt"] for window in result["windows"]]
        result["retried_windows"] = sorted(targets)
    result["retryable_windows"] = retryable_windows(result)
    return result
