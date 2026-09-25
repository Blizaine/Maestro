"""Opt-in Win10 NR, with actual DLSS SR in a separate unhooked worker.

The direct bridge preserves resolution. Enlargement must come from DLSS SR,
never from an ordinary resize disguised as neural super-resolution.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path

SR_OUTPUT_ROW_ALIGNMENT = 64
MAX_SR_INPUT_PADDING = 128


def _even_scaled_dimension(value, scale):
    """Match NeuralRenderingSession's even-rounded target dimensions."""
    return max(2, math.floor(value * scale / 2 + 0.5) * 2)


def _check_native_dimensions(width, height, label):
    if max(width, height) > 7680 or min(width, height) > 4320:
        raise ValueError(f"Experimental Windows 10 DLSS {label} {width}x{height} exceeds 7680x4320")


HASHES = {
    "native/bin/dlss5nr_bridge.dll": "a60b0a8783daab45c5d001bb954459acf3e116f331b9218030bfc6378e4515dd",
    "runtime/caller/nvngx.dll_comfy.dll": "4b2913f2c90ef7721eda508c5b5962f49934f4ff654e7f4c1c245695cefd7591",
    "runtime/nvngx_dlssnr.dll": "6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927",
    "sr/dlss/nvngx_dlss.dll": "c85f971ce023c9f3492fc7455f0b01a24ba18ea39636407a846902c4360b0b7e",
    "sr/host/nr-depth-worker.exe": "f8e2967912e5d596e8e36049370487b83620b0cb5845937b681cf835bafc6d0b",
}


@lru_cache(maxsize=20)
def _verified(path: str, mtime_ns: int, size: int, expected: str) -> bool:
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest() == expected


def installation_error(root: Path) -> str:
    for filename, expected in HASHES.items():
        path = root / filename
        try:
            stat = path.stat()
            if not _verified(str(path), stat.st_mtime_ns, stat.st_size, expected):
                return f"experimental DLSS checksum mismatch: {filename}"
        except OSError:
            return f"missing experimental DLSS file: {filename}"
    if any((root / "sr/host" / name).exists() for name in ("dxgi.dll", "renodx-dlss5.addon64")):
        return "experimental DLSS SR host contains an incompatible ReShade hook"
    return ""


class ExperimentalNeuralSession:
    def __init__(self, width, height, frames, scale, intensity, abort_callback, *, still_image, runtime_dir):
        from . import runtime
        from .direct_session import DirectNRSession
        if scale not in runtime.NR_MODES:
            supported = ", ".join(f"x{value:g}" for value in runtime.NR_MODES)
            raise ValueError(f"Experimental Windows 10 DLSS supports {supported}")
        if scale == 1.0:
            output_width, output_height = width, height
        else:
            output_width = _even_scaled_dimension(width, scale)
            output_height = _even_scaled_dimension(height, scale)
        _check_native_dimensions(width, height, "render dimensions")
        _check_native_dimensions(output_width, output_height, "output dimensions")

        # The pinned experimental SR worker only succeeds for output rows whose
        # width is a multiple of 64 RGBA pixels. Pad source pixels on the right
        # to satisfy that observed worker constraint, then crop before DirectNR.
        # Keep public dimensions at the requested even-rounded sizes.
        self._right_padding = 0
        self._sr_input_width = width
        self._sr_output_width = output_width
        self._sr_output_height = output_height
        if scale > 1.0 and output_width % SR_OUTPUT_ROW_ALIGNMENT:
            for padding in range(1, MAX_SR_INPUT_PADDING + 1):
                candidate_input_width = width + padding
                candidate_output_width = _even_scaled_dimension(candidate_input_width, scale)
                if candidate_output_width % SR_OUTPUT_ROW_ALIGNMENT == 0:
                    self._right_padding = padding
                    self._sr_input_width = candidate_input_width
                    self._sr_output_width = candidate_output_width
                    break
            else:
                raise ValueError(
                    "Experimental Windows 10 DLSS could not align the SR output row "
                    f"within {MAX_SR_INPUT_PADDING} source pixels"
                )
        _check_native_dimensions(self._sr_input_width, height, "padded render dimensions")
        _check_native_dimensions(self._sr_output_width, output_height, "padded SR output dimensions")

        error = installation_error(runtime_dir)
        if error:
            raise RuntimeError(error)
        self.render_width, self.render_height = width, height
        self.output_width, self.output_height = output_width, output_height
        self.needs_guides = scale > 1
        self.sr = self.nr = None
        try:
            if self.needs_guides:
                self.sr = runtime.NeuralRenderingSession(self._sr_input_width, height, frames, scale, intensity, abort_callback,
                                                         host_dir=runtime_dir / "sr/host")
                negotiated_render = self.sr.render_width, self.sr.render_height
                negotiated_output = self.sr.output_width, self.sr.output_height
                expected_render = self._sr_input_width, height
                expected_output = self._sr_output_width, output_height
                if negotiated_render != expected_render or negotiated_output != expected_output:
                    raise RuntimeError(
                        "DLSS SR negotiated unexpected dimensions "
                        f"{negotiated_render[0]}x{negotiated_render[1]} -> "
                        f"{negotiated_output[0]}x{negotiated_output[1]}; expected "
                        f"{expected_render[0]}x{expected_render[1]} -> "
                        f"{expected_output[0]}x{expected_output[1]}"
                    )
            self.nr = DirectNRSession(self.output_width, self.output_height, frames,
                                      temporal=not still_image, intensity=intensity,
                                      runtime_dir=runtime_dir, abort_callback=abort_callback)
            print(f"[DLSS Win10 experimental] {width}x{height} -> "
                  f"{self.output_width}x{self.output_height}; "
                  f"SR right padding={self._right_padding}; "
                  f"{'DLSS SR + direct NR' if self.needs_guides else 'direct NR'}; "
                  f"{'still image' if still_image else 'NVIDIA optical flow'}; isolated worker.", flush=True)
        except BaseException:
            self.close(abort=True)
            raise

    def process_frame(self, index, rgba, motion, reset, output=None, *, depth=None):
        if self.sr is not None:
            if self._right_padding:
                from . import runtime
                def pad_right(array, name):
                    if array is None or len(array.shape) < 2 or tuple(array.shape[:2]) != (self.render_height, self.render_width):
                        raise ValueError(
                            f"{name} must have render dimensions "
                            f"{self.render_width}x{self.render_height} before SR padding"
                        )
                    pad_width = ((0, 0), (0, self._right_padding)) + ((0, 0),) * (array.ndim - 2)
                    return runtime.np.pad(array, pad_width, mode="edge")

                rgba = pad_right(rgba, "RGBA frame")
                motion = pad_right(motion, "Motion guide")
                depth = pad_right(depth, "Depth guide")
            rgba = self.sr.process_frame(index, rgba, motion, reset, depth=depth)
            expected_shape = (self._sr_output_height, self._sr_output_width, 4)
            if rgba.shape != expected_shape:
                raise RuntimeError(f"DLSS SR returned {rgba.shape}, expected {expected_shape}")
            if self._right_padding:
                rgba = rgba[:, :self.output_width, :]
        return self.nr.process_frame(index, rgba, reset, output)

    def close(self, *, abort=False):
        try:
            if self.nr is not None:
                self.nr.close(abort=abort)
        finally:
            if self.sr is not None:
                self.sr.close(abort=abort)
