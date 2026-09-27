"""Validated, lightweight settings for Krea 2 Identity Edit."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


IDENTITY_SETTINGS_DEFAULTS = {
    "krea2_ref_boost": 1.0,
    "krea2_ref_boost_a": 1.0,
    "krea2_grounding_px": 768,
}

_INTEGER_RE = re.compile(r"^[+-]?\d+$")


def _float_setting(values: Mapping[str, Any], key: str) -> float:
    raw_value = values.get(key, IDENTITY_SETTINGS_DEFAULTS[key])
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, str)):
        raise ValueError(f"{key} must be a finite number between 0 and 10.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{key} must be a finite number between 0 and 10.") from error
    if not math.isfinite(value) or not 0.0 <= value <= 10.0:
        raise ValueError(f"{key} must be a finite number between 0 and 10.")
    return value


def _integer_setting(values: Mapping[str, Any], key: str) -> int:
    raw_value = values.get(key, IDENTITY_SETTINGS_DEFAULTS[key])
    if isinstance(raw_value, bool):
        raise ValueError(f"{key} must be an integer between 384 and 1536.")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, float):
        if not math.isfinite(raw_value) or not raw_value.is_integer():
            raise ValueError(f"{key} must be an integer between 384 and 1536.")
        value = int(raw_value)
    elif isinstance(raw_value, str) and _INTEGER_RE.fullmatch(raw_value.strip()):
        value = int(raw_value.strip())
    else:
        raise ValueError(f"{key} must be an integer between 384 and 1536.")
    if not 384 <= value <= 1536:
        raise ValueError(f"{key} must be an integer between 384 and 1536.")
    return value


def normalize_identity_settings(values: Mapping[str, Any] | None = None) -> dict[str, float | int]:
    """Return the three canonical values, filling defaults and rejecting invalid input."""
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ValueError("Krea 2 Identity Edit custom_settings must be an object.")
    return {
        "krea2_ref_boost": _float_setting(values, "krea2_ref_boost"),
        "krea2_ref_boost_a": _float_setting(values, "krea2_ref_boost_a"),
        "krea2_grounding_px": _integer_setting(values, "krea2_grounding_px"),
    }
