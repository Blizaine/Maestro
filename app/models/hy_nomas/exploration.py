# Adapted from Hy-Nomas (https://github.com/Tencent-Hunyuan/Hy-Nomas, models/hy_nomas/exploration.py), Copyright (C) 2026 Tencent, Apache-2.0.
# Modified for Maestro. Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0.
"""Model-side exploration request contract; the application supplies its UI adapter."""

from __future__ import annotations

import queue
from dataclasses import dataclass, field

from .prompts import FPS, TEMPORAL_RATIO

MOVE_SECONDS = 1.0
EVENT_SECONDS = 2.5
MOVE_STRIDE = 5
EXPLORE_PROMPT = "__hy_nomas_explore_world__"
CAMERA_CONTROLS = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "look_up": "up",
    "look_down": "down",
    "look_left": "left",
    "look_right": "right",
}


def move_frames(seconds, event=False):
    """Latent frames for a move, rounded to the decoder's five-frame stride."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("World move duration must be a finite number of seconds.") from error
    if not 0 < seconds <= 60:
        raise ValueError("World move duration must be greater than zero and at most 60 seconds.")
    seconds = max(seconds, EVENT_SECONDS if event else 0.0)
    return max(MOVE_STRIDE, round(seconds * FPS / TEMPORAL_RATIO / MOVE_STRIDE) * MOVE_STRIDE)


@dataclass
class WorldExplorationAdapter:
    """Thread-safe request/response bridge expected by ``model_factory._explore``.

    The model creates no UI or gallery entry. The primary app integration may
    construct this adapter and feed requests using ``submit``; model results
    are returned through ``events``.
    """

    world_path: str
    requests: queue.Queue = field(default_factory=queue.Queue)
    events: queue.Queue = field(default_factory=queue.Queue)

    def next_request(self, timeout=0.25):
        try:
            return self.requests.get(timeout=timeout)
        except queue.Empty:
            raise

    def publish(self, event, **payload):
        self.events.put({"event": event, **payload})

    def submit(self, request):
        if request is not None and not isinstance(request, dict):
            raise ValueError("World exploration requests must be dictionaries or None to finish.")
        self.requests.put(request)


def exploration_for(value):
    return value if isinstance(value, WorldExplorationAdapter) else None


def explorer_menus(manifest, memory):
    """Pure menu metadata for an app adapter; it does not expose UI by itself."""
    from .history import memory_choices
    from .transformer import DEPTH, HEAD_DIM, HEADS

    if not isinstance(manifest, dict) or not isinstance(memory, dict):
        raise ValueError("Hy-Nomas explorer menus need a world manifest and memory profile.")
    width, height = int(manifest["width"]), int(manifest["height"])
    fps = float(manifest.get("fps", FPS))
    return {
        "controls": CAMERA_CONTROLS,
        "seconds": [(value, f"{value} s", move_frames(value)) for value in (1, 2, 3)],
        "steps": [(8, "Best"), (6, "Balanced"), (4, "Fast")],
        "memory": memory_choices((width // 32) * (height // 32), DEPTH, HEADS, HEAD_DIM, TEMPORAL_RATIO / fps, bool(memory.get("preload"))),
        "defaults": {"seconds": 1, "steps": 4, "memory": 18},
        "task": {"num_inference_steps": manifest["steps"], "video_length": 5},
    }
