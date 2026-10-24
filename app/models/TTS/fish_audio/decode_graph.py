# Copyright Fish Audio contributors; see LICENSE_FISH_AUDIO.
# Native Maestro integration informed by fishaudio/fish-speech
# 541566932abbaebcb0a40e562a1f846738383f38, fish_omni/generate.py.
"""A part-local whole-frame CUDA graph, with transactional capture warmup.

The graph owns fixed inputs and captures sampling, the codebook tower and the
next backbone step together. No captured allocation outlives the part's caches.
Only the accelerated, fully GPU-resident path is eligible; failed capture falls
back before generating any real frames, after restoring caches and RNG.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import time

import torch

from .backbone import GenerationCancelled
from .graph_residency import (
    graph_eligibility,
    resident_capture_scope,
    weight_signature,
)
from .tower import RAS_WINDOW_FRAMES


class GraphStateRestoreError(Exception):
    """Capture could not safely restore the real request; do not retry it."""


@contextmanager
def preserve_decode_state(backbone, tower, device):
    """Restore cache contents in place and RNG, including when capture fails."""
    originals = (backbone.kv, backbone.conv, backbone.state, tower.kv)
    snapshots = [value.clone() for value in originals]
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            # A poisoned CUDA context must propagate; continuing eagerly would
            # claim an unsafe recovery. Ordinary unsupported-capture/OOM errors
            # leave the stream usable and allow the transaction to roll back.
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                for original, snapshot in zip(originals, snapshots):
                    original.copy_(snapshot)
            except RuntimeError as error:
                raise GraphStateRestoreError("Fish graph capture could not restore decode state") from error


class WholeFrameDecoder:
    """Fixed-address frame transition; constructed only after real prefill."""

    def __init__(self, backbone, tower, hidden, frame, history, position,
                 temperature, top_k, top_p):
        self.backbone, self.tower = backbone, tower
        self.device = hidden.device
        self.capacity = min(backbone.cache_capacity, tower.cache_capacity)
        self.hidden = hidden.clone()
        self.frame = frame.clone()
        self.position = torch.tensor([position], dtype=torch.long, device=self.device)
        self.history = torch.full((RAS_WINDOW_FRAMES,), -1, dtype=torch.long, device=self.device)
        window = history[-RAS_WINDOW_FRAMES:]
        if window.numel():
            self.history[-window.numel():].copy_(window)
        self.temperature = torch.tensor(temperature, dtype=torch.float32, device=self.device)
        self.top_k = torch.tensor(top_k, dtype=torch.long, device=self.device)
        self.top_p = torch.tensor(top_p, dtype=torch.float32, device=self.device)
        self.sampled = torch.empty_like(frame)
        self.finite = torch.ones((), dtype=torch.bool, device=self.device)
        self.graph = None
        self.capture_seconds = 0.0
        self._cache_objects = (backbone.kv, backbone.conv, backbone.state, tower.kv)
        self._next_position = position
        self._signature = None

    def _run_frame(self):
        """One GPU-only transition; also usable eagerly for parity tests."""
        backbone, tower = self.backbone, self.tower
        logits = backbone.compute_logits(self.hidden)
        sampled, finite, extra, semantic = tower.sample_frame(
            logits, self.hidden, self.position, self.frame, self.history, tower.kv,
            tower.main_candidates, tower.candidate_rows, self.capacity,
            self.temperature, self.top_k, self.top_p,
        )
        self.sampled.copy_(sampled)
        self.finite.copy_(finite)
        hidden, _ = backbone.decode_token(
            sampled[:, 0], extra, semantic, self.position + 1, self.capacity, True,
            backbone.kv, backbone.conv, backbone.state, return_logits=False,
        )
        self.hidden.copy_(hidden)
        self.frame.copy_(sampled)
        self.history.copy_(torch.cat((self.history[1:], sampled[0, :1])))
        self.position.add_(1)

    def _reset_inputs(self, hidden, frame, history, position):
        self.hidden.copy_(hidden)
        self.frame.copy_(frame)
        self.position.fill_(position)
        self.history.fill_(-1)
        window = history[-RAS_WINDOW_FRAMES:]
        if window.numel():
            self.history[-window.numel():].copy_(window)
        self._next_position = position

    @classmethod
    def try_create(cls, backbone, tower, hidden, frame, history, position,
                   temperature, top_k, top_p, *, engine, profile, check_abort=None):
        if os.environ.get("MAESTRO_FISH_CUDA_GRAPH", "1") == "0":
            return None, "disabled by MAESTRO_FISH_CUDA_GRAPH"
        eligible, reason = graph_eligibility(backbone, tower, engine, profile)
        if not eligible:
            return None, reason
        decoder = None
        try:
            if check_abort is not None:
                check_abort()
            device = hidden.device
            cache_bytes = sum(value.numel() * value.element_size() for value in
                              (backbone.kv, backbone.conv, backbone.state, tower.kv))
            free, _ = torch.cuda.mem_get_info(device)
            # The rollback snapshot and a minimum working reserve are additional
            # to the already allocated weights/caches. Capture may need more;
            # CUDA OOM then rolls back to eager without repeating the prompt.
            if free < cache_bytes + 256 * 1024**2:
                return None, "insufficient free VRAM for capture rollback and working space"
            tower.prepare_sampling()
            decoder = cls(backbone, tower, hidden, frame, history, position,
                          temperature, top_k, top_p)
            signature = weight_signature(backbone, tower)
            started = time.perf_counter()
            with preserve_decode_state(backbone, tower, device), resident_capture_scope(backbone, tower):
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        # Warm the same valid cache position; the state contents
                        # are transactional, and no real output is retained.
                        decoder._reset_inputs(hidden, frame, history, position)
                        decoder._run_frame()
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                if check_abort is not None:
                    check_abort()
                eligible, reason = graph_eligibility(backbone, tower, engine, profile)
                if not eligible:
                    raise RuntimeError(reason)
                if weight_signature(backbone, tower) != signature:
                    raise RuntimeError("decode weights changed during warmup")
                decoder._reset_inputs(hidden, frame, history, position)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    decoder._run_frame()
                decoder.graph = graph
            decoder._reset_inputs(hidden, frame, history, position)
            decoder._signature = signature
            decoder.capture_seconds = time.perf_counter() - started
            return decoder, None
        except GenerationCancelled:
            if decoder is not None:
                decoder.close()
            raise
        except (RuntimeError, ValueError) as error:
            if decoder is not None:
                decoder.close()
            return None, f"capture unavailable: {str(error)[:300]}"

    def validate(self):
        """Validate once after capture, never tree-scan weights on each frame."""
        if self.graph is None or weight_signature(self.backbone, self.tower) != self._signature:
            raise RuntimeError("Fish CUDA graph weights changed; discard the capture before decoding")

    def step(self, position):
        if self.graph is None:
            raise RuntimeError("Fish CUDA graph has been released")
        if position != self._next_position or position + 1 >= self.capacity:
            raise RuntimeError("Fish CUDA graph position is outside its captured cache")
        current = (self.backbone.kv, self.backbone.conv, self.backbone.state, self.tower.kv)
        if any(actual is not captured for actual, captured in zip(current, self._cache_objects)):
            raise RuntimeError("Fish CUDA graph caches changed; discard the capture before decoding")
        self.graph.replay()
        self._next_position += 1
        # The caller copies both results before the next replay. They are fixed
        # graph-owned buffers, not independently retained frame tensors.
        return self.sampled, self.finite

    def close(self):
        self.graph = None
        self._signature = None
