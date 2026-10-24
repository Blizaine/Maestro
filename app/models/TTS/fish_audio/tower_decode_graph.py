# Copyright Fish Audio contributors; see LICENSE_FISH_AUDIO.
"""A sampling-only CUDA Graph for Fish's audio codebook tower."""
from __future__ import annotations

import os
import time

import torch

from . import graph_residency
from .backbone import GenerationCancelled
from .decode_graph import GraphStateRestoreError, preserve_decode_state
from .graph_residency import resident_capture_scope
from .tower import RAS_WINDOW_FRAMES

_CAPTURE_RESERVE_BYTES = 256 * 1024**2


def _device_key(device):
    if device.type != "cuda":
        raise RuntimeError("Fish tower CUDA Graph requires CUDA inputs")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return "cuda", int(index)


def _tower_weight_signature(tower, device):
    """Sign only physical parameter/buffer storage used by the tower."""
    expected = _device_key(device)
    entries = []
    for kind, method_name in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
        for name, value in graph_residency._named_items(tower, method_name):
            physical = list(graph_residency._physical_tensors(value, name or "<root>"))
            if not physical:
                raise RuntimeError(f"tower {kind} {name!r} has unknown physical storage")
            for physical_name, tensor in physical:
                if graph_residency._device_info(tensor) != expected:
                    raise RuntimeError(f"tower dense weight {physical_name!r} is not on the decode CUDA device")
                entries.append((kind, name, physical_name, graph_residency._storage_signature(tensor)))
    if not entries:
        raise RuntimeError("tower has no registered physical weights to verify")
    return "fish-tower-physical-weights-v1", tuple(entries)


def _capture_eligibility(backbone, tower, device, engine):
    """Check tower placement and, when present, stable shared MMGP residency."""
    if engine != "vllm":
        return False, "Fish tower CUDA Graph requires the vllm engine", None
    try:
        expected = _device_key(device)
        trees = graph_residency._root_trees(backbone, tower)
        states = graph_residency._manager_states(trees)
        # A manager shared by both roots has already been checked for active
        # models and complete/stable block state by _manager_states. Separate
        # managers cannot prove that the roots remain resident together.
        if states and (
            len(states) != 1
            or set(states[0].get("roots", {})) != {"backbone", "tower"}
        ):
            raise RuntimeError("MMGP does not prove the backbone and tower are resident together")
        cache = getattr(tower, "kv", None)
        if cache is None or graph_residency._device_info(cache) != expected:
            raise RuntimeError("tower KV cache is not on the decode CUDA device")
        for name in ("main_candidates", "candidate_rows"):
            table = getattr(tower, name, None)
            if table is not None and graph_residency._device_info(table) != expected:
                raise RuntimeError(f"tower {name} is not on the decode CUDA device")
        signature = _tower_weight_signature(tower, device)
        return True, None, signature
    except Exception as error:  # noqa: BLE001 - unknown residency must fail closed.
        detail = str(error) or type(error).__name__
        return False, f"Fish tower CUDA Graph residency is not verified: {detail}", None


class TowerFrameDecoder:
    """Fixed-address capture of sampling, codebooks, and their embedding sum."""

    def __init__(self, backbone, tower, hidden, logits, frame, history, position,
                 temperature, top_k, top_p):
        self.backbone = backbone
        self.tower = tower
        self.device = hidden.device
        self.capacity = int(tower.cache_capacity)
        self.hidden = hidden.clone()
        self.logits = logits.clone()
        self.frame = frame.clone()
        self.position = torch.tensor([position], dtype=torch.long, device=self.device)
        self.history = torch.full((RAS_WINDOW_FRAMES,), -1, dtype=torch.long, device=self.device)
        self._copy_history(history)
        self.temperature = torch.as_tensor(temperature, dtype=torch.float32, device=self.device).reshape(())
        self.top_k = torch.as_tensor(top_k, dtype=torch.long, device=self.device).reshape(())
        self.top_p = torch.as_tensor(top_p, dtype=torch.float32, device=self.device).reshape(())
        self.sampled = torch.empty_like(frame)
        self.finite = torch.empty((), dtype=torch.bool, device=self.device)
        self.extra = torch.empty_like(hidden)
        self.semantic = torch.empty((frame.shape[0],), dtype=torch.bool, device=self.device)
        self.graph = None
        self.capture_seconds = 0.0
        self._next_position = int(position)
        self._signature = None
        self._tower_cache = tower.kv
        self._candidate_tables = (tower.main_candidates, tower.candidate_rows)

    def _copy_history(self, history):
        values = torch.as_tensor(history, dtype=torch.long, device=self.device).reshape(-1)
        window = values[-RAS_WINDOW_FRAMES:]
        self.history.fill_(-1)
        if window.numel():
            self.history[-window.numel():].copy_(window)

    def _set_inputs(self, hidden, logits, frame, history, position):
        self.hidden.copy_(hidden)
        self.logits.copy_(logits)
        self.frame.copy_(frame)
        self.position.fill_(position)
        self._copy_history(history)

    def _sample_frame(self):
        sampled, finite, extra, semantic = self.tower.sample_frame(
            self.logits,
            self.hidden,
            self.position,
            self.frame,
            self.history,
            self.tower.kv,
            self.tower.main_candidates,
            self.tower.candidate_rows,
            self.capacity,
            self.temperature,
            self.top_k,
            self.top_p,
        )
        self.sampled.copy_(sampled)
        self.finite.copy_(finite)
        self.extra.copy_(extra)
        self.semantic.copy_(semantic)

    def _check_capture_objects(self):
        if self.tower.kv is not self._tower_cache:
            raise RuntimeError("Fish tower CUDA graph cache changed; discard the capture before decoding")
        if any(actual is not captured for actual, captured in zip(
            (self.tower.main_candidates, self.tower.candidate_rows), self._candidate_tables
        )):
            raise RuntimeError("Fish tower CUDA graph sampling tables changed; discard the capture before decoding")

    @classmethod
    def try_create(cls, backbone, tower, hidden, logits, frame, history, position,
                   temperature, top_k, top_p, *, engine, check_abort=None):
        if os.environ.get("MAESTRO_FISH_CUDA_GRAPH", "1") == "0":
            return None, "disabled by MAESTRO_FISH_CUDA_GRAPH"
        if check_abort is not None:
            check_abort()
        if hidden.device != logits.device or hidden.device != frame.device:
            return None, "Fish tower graph inputs must share one CUDA device"
        eligible, reason, signature = _capture_eligibility(backbone, tower, hidden.device, engine)
        if not eligible:
            return None, reason

        decoder = None
        try:
            # Prefill has completed before this API is called. Candidate tables
            # are immutable capture inputs and must be ready outside the graph.
            tower.prepare_sampling()
            eligible, reason, signature = _capture_eligibility(backbone, tower, hidden.device, engine)
            if not eligible:
                return None, reason

            cache_values = (backbone.kv, backbone.conv, backbone.state, tower.kv)
            cache_bytes = sum(value.numel() * value.element_size() for value in cache_values)
            free, _ = torch.cuda.mem_get_info(hidden.device)
            if free < cache_bytes + _CAPTURE_RESERVE_BYTES:
                return None, "insufficient free VRAM for cache rollback and capture reserve"

            decoder = cls(backbone, tower, hidden, logits, frame, history, position,
                          temperature, top_k, top_p)
            started = time.perf_counter()
            with preserve_decode_state(backbone, tower, hidden.device), resident_capture_scope(backbone, tower):
                stream = torch.cuda.Stream(device=hidden.device)
                stream.wait_stream(torch.cuda.current_stream(hidden.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        decoder._set_inputs(hidden, logits, frame, history, position)
                        decoder._sample_frame()
                torch.cuda.current_stream(hidden.device).wait_stream(stream)
                torch.cuda.synchronize(hidden.device)
                if check_abort is not None:
                    check_abort()
                eligible, reason, current_signature = _capture_eligibility(
                    backbone, tower, hidden.device, engine
                )
                if not eligible:
                    raise RuntimeError(reason)
                if current_signature != signature:
                    raise RuntimeError("tower weights changed during graph warmup")

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    decoder._set_inputs(hidden, logits, frame, history, position)
                    with torch.cuda.graph(graph, stream=stream):
                        decoder._sample_frame()
                torch.cuda.current_stream(hidden.device).wait_stream(stream)
                torch.cuda.synchronize(hidden.device)
                if check_abort is not None:
                    check_abort()
                decoder.graph = graph

            decoder._signature = signature
            decoder.capture_seconds = time.perf_counter() - started
            decoder.validate()
            return decoder, None
        except (GenerationCancelled, GraphStateRestoreError):
            if decoder is not None:
                decoder.close()
            raise
        except (RuntimeError, ValueError) as error:
            if decoder is not None:
                decoder.close()
            return None, f"capture unavailable: {str(error)[:300]}"

    def validate(self):
        """Check the captured tower weights and fixed cache/table objects."""
        if self.graph is None:
            raise RuntimeError("Fish tower CUDA graph has been released")
        self._check_capture_objects()
        eligible, reason, signature = _capture_eligibility(
            self.backbone, self.tower, self.device, "vllm"
        )
        if not eligible or signature != self._signature:
            raise RuntimeError(reason or "Fish tower CUDA graph weights changed; discard the capture")

    def step(self, position, hidden, logits, frame, history):
        if self.graph is None:
            raise RuntimeError("Fish tower CUDA graph has been released")
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("Fish tower CUDA graph position must be an integer")
        if position != self._next_position or position < 0 or position >= self.capacity:
            raise RuntimeError("Fish tower CUDA graph position is outside its captured cache")
        self._check_capture_objects()
        self._set_inputs(hidden, logits, frame, history, position)
        self.graph.replay()
        self._next_position += 1
        return self.sampled, self.finite, self.extra, self.semantic

    def close(self):
        self.graph = None
        self._signature = None
