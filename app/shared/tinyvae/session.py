"""Bound optional TinyVAE preview work so it cannot disrupt generation."""
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from .media import encode_video


class PreviewSession:
    """Decode sparse previews and encode video without queuing stale work."""

    def __init__(self, decoder, send_cmd, gen, image, video=False, duration=None):
        self.decoder = decoder
        self.send_cmd = send_cmd
        self.gen = gen
        self.image = bool(image)
        self.duration = self._valid_duration(duration)
        self.video = bool(video and not self.image and self.duration is not None)
        self.context = None
        self.last_step = -1
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Preview Encoder") if self.video else None
        self.future = None
        self.closed = False
        self.failed = False

        self._closing = False
        self._close_complete = False
        self._notice_sent = False
        self._lock = threading.RLock()
        self._capture_lock = threading.Lock()

    @staticmethod
    def _valid_duration(duration):
        try:
            value = float(duration)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    def _aborted(self):
        try:
            return bool(self.gen.get("abort", False))
        except Exception:
            return False

    def _inactive_locked(self):
        return self.closed or self._closing or self.failed

    def wants_capture(self, step, total, pass_no):
        """Return whether this step is due, without changing session state."""
        try:
            step, total = int(step), int(total)
            if step < 0 or total <= 0 or self._aborted():
                return False
            context = (pass_no, total)
            final = step >= total - 1
            with self._lock:
                if self._inactive_locked():
                    return False
                last_step = self.last_step
                if context != self.context or step < last_step:
                    last_step = -1
                interval = max(1, (total + 4) // 6)
                due = last_step < 0 or final or step - last_step >= interval
                if not due:
                    return False
                future = self.future
                if self.video and future is not None:
                    try:
                        busy = not future.done()
                    except Exception:
                        busy = True
                    if busy and not final:
                        return False
                return True
        except Exception:
            return False

    def _cancelled(self, context):
        with self._lock:
            stale = context != self.context
            stopped = self.closed or self.failed
        return stopped or stale or self._aborted()

    def _decode_cancelled(self, context):
        with self._lock:
            closing = self._closing
        return closing or self._cancelled(context)

    def _fail(self, message):
        with self._lock:
            if self.failed:
                return
            self.failed = True
            self.closed = True
            self._closing = True
            notify = not self._notice_sent
            self._notice_sent = True
            executor = self.executor
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        if notify:
            try:
                self.send_cmd("preview_notice", message)
            except Exception:
                pass

    def _encode(self, frames, fps, context):
        try:
            if self._cancelled(context):
                return
            preview = encode_video(frames, fps, lambda: self._cancelled(context))
            if preview is not None and not self._cancelled(context):
                self.send_cmd("preview", preview)
        except Exception:
            self._fail("Live video preview was disabled because encoding failed.")

    def capture(self, latent, step, total, pass_no):
        if not self.wants_capture(step, total, pass_no):
            return
        try:
            with self._capture_lock:
                if not self.wants_capture(step, total, pass_no):
                    return
                step, total = int(step), int(total)
                context = (pass_no, total)
                final = step >= total - 1
                with self._lock:
                    if self._inactive_locked() or self._aborted():
                        return
                    if context != self.context or step < self.last_step:
                        self.context, self.last_step = context, -1
                    future = self.future

                if future is not None:
                    try:
                        if not future.done() and not final:
                            return
                        future.result()
                    except Exception:
                        self._fail("Live preview was disabled because video encoding failed.")
                        return
                    with self._lock:
                        if self.future is future:
                            self.future = None

                if self._decode_cancelled(context):
                    return
                with self._lock:
                    if self._inactive_locked() or context != self.context:
                        return
                    self.last_step = step

                try:
                    with torch.inference_mode():
                        preview = self.decoder(
                            latent,
                            image=self.image,
                            abort_check=lambda: self._decode_cancelled(context),
                            video=self.video,
                            duration=self.duration if self.video else None,
                        )
                except Exception:
                    self._fail("Live preview was disabled because decoding failed.")
                    return

                if preview is None or self._decode_cancelled(context):
                    return
                if self.video:
                    try:
                        frames, fps = preview
                        if not frames or self.executor is None:
                            raise ValueError("video preview has no frames or encoder")
                        with self._lock:
                            if self._inactive_locked() or context != self.context:
                                return
                            self.future = self.executor.submit(self._encode, frames, fps, context)
                    except Exception:
                        self._fail("Live video preview was disabled because encoding failed.")
                else:
                    try:
                        if not self._decode_cancelled(context):
                            self.send_cmd("preview", preview)
                    except Exception:
                        self._fail("Live preview was disabled because it could not be sent.")
        except Exception:
            # Preview work is optional; no error from it should reach generation.
            self._fail("Live preview was disabled because preview processing failed.")

    def close(self, cancel=False):
        try:
            with self._lock:
                if self._close_complete:
                    return
                aborted = bool(cancel or self._aborted() or self.failed)
                self._closing = True
                if aborted:
                    self.closed = True
                future = self.future
                executor = self.executor

            if aborted:
                if future is not None:
                    try:
                        future.cancel()
                    except Exception:
                        pass
            elif future is not None:
                try:
                    future.result()
                except Exception:
                    self._fail("Live preview was disabled because video encoding failed.")

            if executor is not None:
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    self._fail("Live preview was disabled because the preview worker could not shut down.")
            with self._lock:
                self.closed = True
                self._close_complete = True
                self.future = None
        except Exception:
            self._fail("Live preview was disabled because the preview worker could not shut down.")
            with self._lock:
                self.closed = True
                self._close_complete = True
