"""Windows 11 DLSS-NR frame-size adapter for the native 64-pixel output rule."""

from __future__ import annotations

import logging
import math
from numbers import Integral
from typing import Callable

import numpy as np


_OUTPUT_ALIGNMENT = 64
_MAX_DIMENSION = 7680
_MAX_SHORT_DIMENSION = 4320
_SUPPORTED_SCALES = (1.0, 1.5, 1.724, 2.0, 3.0)
_LOGGER = logging.getLogger(__name__)


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"DLSS {name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"DLSS {name} must be a positive integer")
    return value


def _fits_resolution(width: int, height: int) -> bool:
    return max(width, height) <= _MAX_DIMENSION and min(width, height) <= _MAX_SHORT_DIMENSION


def _even_output_dimension(source: int, scale: float) -> int:
    # Keep this exactly in sync with NeuralRenderingSession's protocol dimensions.
    return max(2, math.floor(source * scale / 2 + 0.5) * 2)


def _next_aligned_source(source: int, scale: float, name: str) -> tuple[int, int]:
    """Keep aligned content unchanged; otherwise find its first even padded size."""
    original_output = _even_output_dimension(source, scale)
    if original_output % _OUTPUT_ALIGNMENT == 0:
        return source, original_output

    candidate = source + (source & 1)
    while candidate <= _MAX_DIMENSION:
        output = _even_output_dimension(candidate, scale)
        if output > _MAX_DIMENSION:
            break
        if output % _OUTPUT_ALIGNMENT == 0:
            return candidate, output
        candidate += 2
    raise ValueError(
        f"DLSS {scale:g}x cannot align the {name} dimension from {source}px "
        f"within the {_MAX_DIMENSION}x{_MAX_SHORT_DIMENSION} limit"
    )


def _pad_edges(array: np.ndarray, pad_height: int, pad_width: int) -> np.ndarray:
    if not pad_height and not pad_width:
        return array
    padding = ((0, pad_height), (0, pad_width)) + ((0, 0),) * (array.ndim - 2)
    return np.ascontiguousarray(np.pad(array, padding, mode="edge"))


class AlignedNeuralSession:
    """Pad guides at the right/bottom edges for native DLSS, then crop its output.

    The public dimensions describe the caller's original content. The worker is
    constructed against a source canvas whose even-rounded scaled output is
    divisible by 64, so the worker never has to stretch the original picture.
    """

    def __init__(
        self,
        width: int,
        height: int,
        frames: int,
        scale: float,
        intensity: float = 1.0,
        abort_callback=None,
        *,
        session_factory: Callable,
    ):
        self.render_width = _positive_dimension(width, "width")
        self.render_height = _positive_dimension(height, "height")
        self.frames = _positive_dimension(frames, "frame count")
        if self.frames > 0xFFFFFFFF:
            raise ValueError("DLSS frame count exceeds the worker protocol limit")

        if not _fits_resolution(self.render_width, self.render_height):
            raise ValueError(
                f"DLSS render {self.render_width}x{self.render_height} exceeds "
                f"the {_MAX_DIMENSION}x{_MAX_SHORT_DIMENSION} limit"
            )

        try:
            scale = float(scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("DLSS scale must be finite, positive, and supported") from error
        if not math.isfinite(scale) or scale <= 0 or scale not in _SUPPORTED_SCALES:
            raise ValueError(
                f"DLSS scale must be one of {', '.join(map(str, _SUPPORTED_SCALES))}"
            )
        self.scale = scale

        try:
            intensity = float(intensity)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("DLSS intensity must be finite") from error
        if not math.isfinite(intensity):
            raise ValueError("DLSS intensity must be finite")

        self.output_width = _even_output_dimension(self.render_width, scale)
        self.output_height = _even_output_dimension(self.render_height, scale)
        if not _fits_resolution(self.output_width, self.output_height):
            raise ValueError(
                f"DLSS output {self.output_width}x{self.output_height} exceeds "
                f"the {_MAX_DIMENSION}x{_MAX_SHORT_DIMENSION} limit"
            )

        self.native_render_width, native_output_width = _next_aligned_source(
            self.render_width, scale, "width"
        )
        self.native_render_height, native_output_height = _next_aligned_source(
            self.render_height, scale, "height"
        )
        if not _fits_resolution(self.native_render_width, self.native_render_height):
            raise ValueError(
                f"DLSS padded render {self.native_render_width}x{self.native_render_height} "
                f"exceeds the {_MAX_DIMENSION}x{_MAX_SHORT_DIMENSION} limit"
            )
        if not _fits_resolution(native_output_width, native_output_height):
            raise ValueError(
                f"DLSS padded output {native_output_width}x{native_output_height} "
                f"exceeds the {_MAX_DIMENSION}x{_MAX_SHORT_DIMENSION} limit"
            )

        if not callable(session_factory):
            raise TypeError("session_factory must be callable")

        self._pad_width = self.native_render_width - self.render_width
        self._pad_height = self.native_render_height - self.render_height
        self._native_output_width = native_output_width
        self._native_output_height = native_output_height
        self._abort_callback = abort_callback
        self._closed = False

        # All geometry checks happen before this call so invalid dimensions can
        # never start a worker or allocate a padded frame.
        session = session_factory(
            self.native_render_width,
            self.native_render_height,
            self.frames,
            self.scale,
            intensity,
            abort_callback,
        )
        self._session = session

        try:
            negotiated_render = (
                getattr(session, "render_width", None),
                getattr(session, "render_height", None),
            )
            if negotiated_render != (self.native_render_width, self.native_render_height):
                raise RuntimeError(
                    "DLSS worker negotiated unexpected render dimensions "
                    f"{negotiated_render[0]}x{negotiated_render[1]} (expected "
                    f"{self.native_render_width}x{self.native_render_height})"
                )

            negotiated_output = (
                getattr(session, "output_width", None),
                getattr(session, "output_height", None),
            )
            if negotiated_output != (native_output_width, native_output_height):
                raise RuntimeError(
                    "DLSS worker negotiated unexpected output dimensions "
                    f"{negotiated_output[0]}x{negotiated_output[1]} (expected "
                    f"{native_output_width}x{native_output_height})"
                )
        except Exception:
            self._close_after_error()
            raise

        if self._pad_width or self._pad_height:
            _LOGGER.info(
                "[DLSS Win11 alignment] %dx%d -> requested %dx%d; "
                "internal %dx%d -> native %dx%d (scale=%s)",
                self.render_width,
                self.render_height,
                self.output_width,
                self.output_height,
                self.native_render_width,
                self.native_render_height,
                native_output_width,
                native_output_height,
                f"{scale:g}",
            )

    @property
    def needs_guides(self) -> bool:
        return getattr(self._session, "needs_guides", True)

    def _close_after_error(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._session.close(abort=True)
        except Exception:
            # Preserve the protocol or frame error that triggered cleanup.
            pass

    def _validate_frame(self, array, name: str, channels: int | None) -> np.ndarray:
        array = np.asarray(array)
        shape = (self.render_height, self.render_width)
        if array.ndim < 2 or array.shape[:2] != shape:
            raise ValueError(
                f"DLSS {name} must start with dimensions {shape[1]}x{shape[0]}"
            )
        if channels is not None and (array.ndim != 3 or array.shape[2] != channels):
            raise ValueError(f"DLSS {name} must have {channels} channels")
        if channels is None and array.ndim != 2:
            raise ValueError("DLSS depth must be a two-dimensional array")
        return array

    def _validate_output(self, output: np.ndarray | None) -> np.ndarray | None:
        if output is None:
            return None
        if not isinstance(output, np.ndarray):
            raise ValueError("DLSS output buffer must be a NumPy array")
        if output.shape != (self.output_height, self.output_width, 4):
            raise ValueError(
                f"DLSS output buffer must have shape "
                f"({self.output_height}, {self.output_width}, 4)"
            )
        if output.dtype != np.uint8:
            raise ValueError("DLSS output buffer must use uint8 pixels")
        if not output.flags.writeable:
            raise ValueError("DLSS output buffer must be writable")
        return output

    def process_frame(
        self,
        index: int,
        rgba: np.ndarray,
        motion: np.ndarray,
        reset: bool,
        output: np.ndarray | None = None,
        *,
        depth: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("DLSS session is closed")
        try:
            if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < self.frames:
                raise ValueError("DLSS frame index must be within the session frame range")
            rgba = self._validate_frame(rgba, "RGBA frame", 4)
            motion = self._validate_frame(motion, "motion guide", 2)
            if depth is not None:
                depth = self._validate_frame(depth, "depth guide", None)
            output = self._validate_output(output)

            padded_rgba = _pad_edges(rgba, self._pad_height, self._pad_width)
            padded_motion = _pad_edges(motion, self._pad_height, self._pad_width)
            padded_depth = (
                None if depth is None else _pad_edges(depth, self._pad_height, self._pad_width)
            )

            has_padding = bool(self._pad_width or self._pad_height)
            if has_padding:
                native = self._session.process_frame(
                    index,
                    padded_rgba,
                    padded_motion,
                    reset,
                    depth=padded_depth,
                )
                native = np.asarray(native)
                expected_native = (self._native_output_height, self._native_output_width, 4)
                if native.shape != expected_native or native.dtype != np.uint8:
                    raise RuntimeError(
                        f"DLSS worker returned {native.shape} {native.dtype}; expected "
                        f"{expected_native} uint8"
                    )
                cropped = native[: self.output_height, : self.output_width]
                if output is None:
                    return np.ascontiguousarray(cropped)
                np.copyto(output, cropped)
                return output

            worker_output = output if output is not None and output.flags.c_contiguous else None
            result = self._session.process_frame(
                index,
                padded_rgba,
                padded_motion,
                reset,
                worker_output,
                depth=padded_depth,
            )
            result = np.asarray(result)
            expected = (self.output_height, self.output_width, 4)
            if result.shape != expected or result.dtype != np.uint8:
                raise RuntimeError(
                    f"DLSS worker returned {result.shape} {result.dtype}; expected {expected} uint8"
                )
            if output is not None and result is not output:
                np.copyto(output, result)
                return output
            return result
        except Exception:
            self._close_after_error()
            raise

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self._session.close(abort=abort)
