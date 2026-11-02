"""Lightweight labels for acceleration LoRAs; never a compatibility verdict.

Use explicit structured metadata when available, then technical filename
tokens. Do not scan descriptive prose or load weight tensors to build a list.
"""
from __future__ import annotations

from collections.abc import Mapping
import re


_ACCELERATION_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])(?:turbo|lightning|lcm|tcd|pdd|dmd|dmad|hyper[-_ ]?sd|lightx2v)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_STEP_TOKEN = re.compile(r"(?:^|[^a-z0-9])(?:[1-9]|1[0-6])[-_ ]?steps?(?:$|[^a-z0-9])", re.IGNORECASE)


def is_acceleration_lora(filename: str, metadata: Mapping | None = None) -> bool:
    """Unknown adapters remain ordinary/unclassified in the list filter."""
    if isinstance(metadata, Mapping):
        explicit = metadata.get("is_acceleration")
        if isinstance(explicit, bool):
            return explicit
    name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    return bool(_ACCELERATION_TOKEN.search(name) or _STEP_TOKEN.search(name))
