"""Assemble completed native windows without retaining their decoded frames."""

from __future__ import annotations

from contextlib import suppress
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


class WindowVideoWriter:
    def __init__(self, *, fps: float, directory: str, ffmpeg: str = "ffmpeg"):
        self._directory = tempfile.TemporaryDirectory(
            prefix=".maestro-windows-", dir=directory, ignore_cleanup_errors=True,
        )
        self.root = Path(self._directory.name)
        self.fps = fps
        self.ffmpeg = ffmpeg
        self.segments: list[tuple[Path, int]] = []

    def append(self, frames, *, encode) -> None:
        frame_count = int(frames.shape[1])
        if frame_count < 1:
            raise ValueError("Cannot append an empty video window.")
        segment = self.root / f"window-{len(self.segments) + 1:04d}.mp4"
        encode(frames, str(segment))
        if not segment.is_file() or not segment.stat().st_size:
            raise RuntimeError("The completed video window could not be encoded.")
        self.segments.append((segment, frame_count))

    def write_video(self, path: str) -> None:
        if not self.segments:
            raise ValueError("There are no completed video windows to assemble.")
        assembled = self.root / "assembled.mp4"
        try:
            if len(self.segments) == 1:
                shutil.copyfile(self.segments[0][0], assembled)
            else:
                concat = self.root / "windows.ffconcat"
                lines = ["ffconcat version 1.0"]
                for segment, frames in self.segments:
                    lines.extend((f"file {segment.name}", f"duration {frames / self.fps:.12f}"))
                concat.write_text("\n".join(lines) + "\n", encoding="ascii")
                result = subprocess.run(
                    [self.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "concat",
                     "-safe", "1", "-i", str(concat.resolve()), "-map", "0:v:0",
                     "-an", "-c:v", "copy", "-movflags", "+faststart", str(assembled)],
                    capture_output=True, text=True, check=False,
                )
                if result.returncode:
                    raise RuntimeError(f"Could not assemble completed video windows: {result.stderr.strip()}")
            os.replace(assembled, path)
        finally:
            with suppress(OSError):
                assembled.unlink(missing_ok=True)

    def close(self) -> None:
        self._directory.cleanup()
