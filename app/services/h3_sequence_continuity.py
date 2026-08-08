"""Small, bounded continuity pack for queued MiniMax H3 Omni sequences."""

from __future__ import annotations

import math
import os
import re
from typing import Any

import numpy as np
from PIL import Image


def reference_capacity(references: list[dict[str, Any]], needed_images: int = 1) -> bool:
    images = sum(
        1 for item in references
        if isinstance(item, dict) and item.get("type") == "image"
    )
    return len(references) + needed_images <= 12 and images + needed_images <= 9


def append_generated_reference(
    references: list[dict[str, Any]],
    path: str,
    *,
    role: str,
) -> tuple[list[dict[str, Any]], int | None]:
    """Append a weak composition image and return its Picture label number."""

    if not os.path.isfile(path) or not reference_capacity(references, 1):
        return references, None
    next_references = [dict(item) for item in references]
    picture_number = 1 + sum(
        1 for item in next_references if item.get("type") == "image"
    )
    next_references.append({
        "id": f"maestro-continuity-{picture_number}-{os.path.basename(path)}",
        "type": "image",
        "path": path,
        "filename": os.path.basename(path),
        "role": role,
        "image_intent": "composition",
        "_maestro_generated_continuity": True,
    })
    return next_references, picture_number


def augment_prompt_with_continuity(
    prompt: str,
    *,
    picture_number: int,
    kind: str,
) -> str:
    """Bind an appended Picture label without changing canonical identities."""

    label = f"<Picture {int(picture_number)}>"
    if label in str(prompt or ""):
        return str(prompt or "")
    if kind == "look":
        definition = (
            f"{label} is a weak project-look reference for the established "
            "wardrobe, location, lighting, and color only; it is not an identity source"
        )
        detail = (
            f"Use {label} only to maintain the established scene look and wardrobe; "
            "do not reproduce its framing as an opening still. "
        )
    else:
        definition = (
            f"{label} is a weak continuity reference for the preceding clip's "
            "blocking, environment state, lighting, and screen direction only; "
            "it is not an identity source"
        )
        detail = (
            f"Use {label} only to preserve prior blocking, environment state, "
            "lighting, and screen direction; begin with a newly generated moving "
            "composition rather than an opening freeze-frame. "
        )

    text = str(prompt or "")
    text = re.sub(
        r"(?m)^(subject_definitions:\s*)(.*)$",
        lambda match: f"{match.group(1)}{match.group(2).rstrip()}; {definition}.",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(retention_analysis:\s*)(.*)$",
        lambda match: f"{match.group(1)}{match.group(2).rstrip()}; {label}: weak_reference",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(detailed_description:\s*)",
        lambda match: f"{match.group(1)}{detail}",
        text,
        count=1,
    )
    return text


def _sharpness(frame: np.ndarray) -> float:
    gray = frame.astype(np.float32).mean(axis=2)
    try:
        import cv2

        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:
        horizontal = np.diff(gray, axis=1)
        vertical = np.diff(gray, axis=0)
        return float(horizontal.var() + vertical.var())


def select_reference_frame(
    video_path: str,
    *,
    kind: str,
    fps: float = 24.0,
) -> Image.Image:
    """Select a sharp look frame or a recent, readable continuity frame."""

    import decord

    reader = decord.VideoReader(video_path)
    frame_count = len(reader)
    if frame_count < 1:
        del reader
        raise ValueError("Generated clip contains no frames.")
    if kind == "look":
        low = max(0, int(frame_count * 0.12))
        high = max(low, int(frame_count * 0.82))
    else:
        tail = max(8, int(max(1.0, float(fps)) * 1.75))
        low = max(0, frame_count - tail)
        high = frame_count - 1
    sample_count = min(14, max(1, high - low + 1))
    indices = sorted({
        int(round(value))
        for value in np.linspace(low, high, sample_count)
    })
    best_frame: np.ndarray | None = None
    best_score = -math.inf
    span = max(1, high - low)
    for index in indices:
        frame = reader[index].asnumpy()
        mean_luma = float(frame.mean())
        exposure_penalty = abs(mean_luma - 127.5) / 127.5
        score = math.log1p(max(0.0, _sharpness(frame))) - exposure_penalty
        if kind != "look":
            score += 0.7 * ((index - low) / span)
        if score > best_score:
            best_score = score
            best_frame = frame
    del reader
    if best_frame is None:
        raise ValueError("Could not select a continuity frame.")
    return Image.fromarray(best_frame).convert("RGB")
