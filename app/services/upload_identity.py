"""Atomic, future-only identity reuse for opted-in uploads."""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from typing import Callable


_publish_lock = threading.RLock()
_COMPRESSED_AUDIO = {".mp3", ".m4a", ".aac"}


def publish_identical_upload(
    content: bytes,
    extension: str,
    upload_dir: str,
    *,
    transcode_to_wav: Callable[[str, str], None] | None = None,
) -> tuple[str, str]:
    """Publish exact bytes once and return ``(filename, path)``.

    Compressed audio is identified by its source bytes but published as the
    same downstream PCM WAV used by the ordinary upload route.
    """

    extension = str(extension or ".bin").casefold()
    if not extension.startswith(".") or any(char in extension for char in "/\\"):
        raise ValueError("Invalid upload extension")
    identity = hashlib.sha256()
    identity.update(extension.encode("utf-8"))
    identity.update(b"\0")
    identity.update(content)
    digest = identity.hexdigest()
    final_extension = ".wav" if extension in _COMPRESSED_AUDIO else extension
    filename = f"upload_{digest}{final_extension}"
    final_path = os.path.join(upload_dir, filename)
    os.makedirs(upload_dir, exist_ok=True)

    with _publish_lock:
        if os.path.isfile(final_path):
            return filename, final_path
        token = uuid.uuid4().hex
        source_temp = os.path.join(upload_dir, f".upload_{token}{extension}")
        publish_temp = source_temp
        wav_temp = ""
        try:
            with open(source_temp, "xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if extension in _COMPRESSED_AUDIO:
                if transcode_to_wav is None:
                    raise ValueError("Compressed audio requires a WAV transcoder")
                wav_temp = os.path.join(upload_dir, f".upload_{token}.wav")
                transcode_to_wav(source_temp, wav_temp)
                if not os.path.isfile(wav_temp):
                    raise OSError("Audio transcoder did not create a WAV file")
                publish_temp = wav_temp
            os.replace(publish_temp, final_path)
            return filename, final_path
        finally:
            for temporary in (source_temp, wav_temp):
                if temporary and os.path.isfile(temporary):
                    try:
                        os.remove(temporary)
                    except OSError:
                        pass
