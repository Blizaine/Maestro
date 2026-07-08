"""
Stall-resilient download configuration + UI hook.

Problem this solves
-------------------
Fresh-install users hit a brutal first impression when HuggingFace
CDN drops a connection mid-download. The TCP socket stays alive but
no bytes flow, and the default `huggingface_hub` library doesn't
notice — the download client sits indefinitely on a stalled read
while the user watches a frozen progress bar in console.

Two-layer fix:

1. Set HF_HUB_DOWNLOAD_TIMEOUT (newer hf-hub versions read this).
2. Monkey-patch the requests library to inject a default read
   timeout on every HTTP call. The read timeout fires when no
   bytes flow for N seconds; `requests.iter_content` raises
   `requests.exceptions.ReadTimeout` which `huggingface_hub`'s
   retry layer catches and resumes from the partial file.

Together these make stalled connections self-heal in ~30s instead
of waiting forever. The 30s threshold is conservative — even slow
internet delivers SOME bytes every few seconds; 30s of silence is
unambiguous "TCP died but didn't tell us."

UI hook
-------
Active downloads are tracked in a module-level dict so the API
layer can surface the state to the UI:

  {
      "<file_id>": {
          "filename": str,            # display name
          "started_at": float,        # epoch
          "last_active_at": float,    # epoch (for stall detection)
          "downloaded_bytes": int,
          "total_bytes": int | None,
          "status": "downloading" | "stalled" | "retrying" | "done",
      },
      ...
  }

The tracking is best-effort — it depends on `tqdm`-monkey-patching
the progress bars HuggingFace Hub uses. If a code path bypasses
tqdm, the dict may be empty even during active downloads. Stalls
are still recovered automatically by the timeout layer regardless.

Import this module BEFORE huggingface_hub or any module that uses
requests for downloads. Effects apply globally on import.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

# ── Layer 1: timeouts ──────────────────────────────────────────────

# 30s without bytes flowing is a near-universal indicator that the
# TCP connection is effectively dead even if the socket hasn't
# closed. Even rural broadband delivers some bytes every few seconds
# during an active transfer.
_READ_TIMEOUT_SECONDS = 30

# 10s connect timeout matches Maestro's other network defaults and
# is generous enough for normal HF CDN handshakes worldwide.
_CONNECT_TIMEOUT_SECONDS = 10


def _install_request_timeouts() -> None:
    """Inject default timeout into requests calls that don't specify one.

    Long-polling endpoints inside Maestro that legitimately want to
    wait >30s should pass an explicit `timeout=` keyword (e.g.
    `timeout=300`) — the patch only fills in the default, doesn't
    override explicit values.
    """
    # 1. Tell hf-hub directly via the env var it reads. Affects newer
    #    huggingface_hub versions where this is honored at the
    #    request_wrapper layer. Do this even if `requests` somehow
    #    isn't importable, since the env var is consumed by hf-hub
    #    independently.
    os.environ.setdefault(
        "HF_HUB_DOWNLOAD_TIMEOUT", str(_READ_TIMEOUT_SECONDS)
    )

    try:
        import requests
    except ImportError:
        # `requests` not installed — should never happen in the
        # Maestro venv (it's a transitive dep) but bail gracefully
        # if it does. Env var still applies.
        return

    # 2. Patch the Session class. Most HTTP libraries route through
    #    Session.request, including hf-hub's _request_wrapper.
    if not getattr(requests.Session.request, "_maestro_timeout_patched", False):
        _original_session_request = requests.Session.request

        def _session_request(self, method, url, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (
                    _CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS
                )
            return _original_session_request(self, method, url, **kwargs)

        _session_request._maestro_timeout_patched = True
        requests.Session.request = _session_request

    # 3. Patch the module-level requests.request too, for direct callers.
    if not getattr(requests.request, "_maestro_timeout_patched", False):
        _original_module_request = requests.request

        def _module_request(method, url, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (
                    _CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS
                )
            return _original_module_request(method, url, **kwargs)

        _module_request._maestro_timeout_patched = True
        requests.request = _module_request


# ── Layer 2: UI download tracking via tqdm patch ───────────────────

_active_downloads: dict = {}
_active_downloads_lock = threading.Lock()


def get_active_downloads() -> list:
    """Return a snapshot of currently-active downloads for the UI.

    Each entry includes a derived `seconds_since_progress` so the UI
    can render "stalled" states without doing math itself.
    """
    now = time.time()
    with _active_downloads_lock:
        # Expire terminal (incomplete) markers after a minute so the banner
        # shows an interrupted download briefly, then clears on its own.
        for fid in [
            fid for fid, st in _active_downloads.items()
            if st.get("status") == "incomplete" and now - st.get("ended_at", now) > 60
        ]:
            _active_downloads.pop(fid, None)
        results = []
        for file_id, state in _active_downloads.items():
            results.append({
                **state,
                "file_id": file_id,
                "seconds_since_progress": (
                    round(now - state.get("last_active_at", now), 1)
                ),
            })
        return results


def _record_download_progress(
    file_id: str,
    filename: str,
    downloaded: int,
    total: Optional[int],
    status: str = "downloading",
) -> None:
    now = time.time()
    with _active_downloads_lock:
        existing = _active_downloads.get(file_id, {})
        _active_downloads[file_id] = {
            "filename": filename,
            "started_at": existing.get("started_at", now),
            "last_active_at": now if downloaded > existing.get("downloaded_bytes", 0) else existing.get("last_active_at", now),
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "status": status,
        }


def _record_download_done(
    file_id: str,
    downloaded: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    """Mark a download finished.

    A tqdm bar's close fires whether the transfer completed OR was
    interrupted (crash, OOM, network drop). When we know the expected
    size and the byte count fell short, don't silently drop the entry as
    "done" — that's what made a truncated download look complete and the
    next model load fail mysteriously. Instead log it and leave a
    short-lived "incomplete" marker the UI can show. A clean finish (or
    an unknown-size bar) is popped as before.
    """
    if total and downloaded is not None and downloaded < total:
        short = total - downloaded
        print(
            f"[safe_download] WARNING: '{file_id}' ended at "
            f"{downloaded}/{total} bytes ({short} short) — download did not "
            f"complete. The file may be truncated; re-run the download / "
            f"generation to fetch the rest."
        )
        now = time.time()
        with _active_downloads_lock:
            existing = _active_downloads.get(file_id, {})
            _active_downloads[file_id] = {
                **existing,
                "filename": existing.get("filename", file_id),
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "status": "incomplete",
                "ended_at": now,
                "last_active_at": existing.get("last_active_at", now),
                "started_at": existing.get("started_at", now),
            }
        return
    with _active_downloads_lock:
        _active_downloads.pop(file_id, None)


def _is_download_tqdm(self) -> bool:
    """Heuristic: does this tqdm instance look like a file download?

    Download progress bars (HuggingFace, urllib3, requests-toolbelt,
    etc.) consistently use `unit="B"` and `unit_scale=True` so the
    bar formats sizes as KB/MB/GB instead of raw counts. Other
    libraries use tqdm for non-download things (training step
    counters, dataset iteration, etc.) where filtering by `B`/scale
    cleanly excludes them.

    Total-size > a few KB is also required to filter out trivial
    progress bars from misc utilities. Genuine model downloads are
    always millions of bytes minimum.
    """
    try:
        unit = getattr(self, "unit", "")
        unit_scale = getattr(self, "unit_scale", False)
        total = getattr(self, "total", None)
        if unit not in ("B", "iB"):
            return False
        if not unit_scale:
            return False
        if total is None or total < 16 * 1024:  # under 16 KB → not a real download
            return False
        return True
    except Exception:
        return False


def _install_tqdm_hook() -> None:
    """Patch the base `tqdm.tqdm` class to observe download progress.

    Why the base class instead of huggingface_hub's tqdm subclass:
    different libraries (and even different hf-hub versions) use
    different subclasses of tqdm. HuggingFace, urllib3, and Wan2GP's
    misc URL fetches all create progress bars via subclasses or
    wrappers. Patching the base class catches all of them.

    To avoid noise from non-download tqdm bars (training loops,
    dataset iteration, etc.), each instance is filtered by
    `_is_download_tqdm` heuristics — only `unit="B"` + `unit_scale`
    + non-trivial total qualifies. Other tqdm uses pass through
    unaffected.

    Failure-tolerant: if `tqdm` isn't importable or patching
    raises, this silently skips. The timeout layer above is the
    critical fix; UI progress is nice-to-have polish.
    """
    try:
        import tqdm as tqdm_module
    except ImportError:
        return

    Tqdm = getattr(tqdm_module, "tqdm", None)
    if Tqdm is None or not isinstance(Tqdm, type):
        return
    if getattr(Tqdm, "_maestro_progress_patched", False):
        return

    try:
        _original_init = Tqdm.__init__
        _original_update = Tqdm.update
        _original_close = Tqdm.close

        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            try:
                if not _is_download_tqdm(self):
                    self._maestro_file_id = None
                    return
                desc = kwargs.get("desc", "") or getattr(self, "desc", "") or ""
                # tqdm sometimes prefixes desc with "(...)" timing or
                # writes "Fetching N files: ..." for HF group bars.
                # Strip the meaningless padding/whitespace for display.
                desc = str(desc).strip(": ()") or f"download-{id(self)}"
                self._maestro_file_id = desc
                self._maestro_filename = desc
                _record_download_progress(
                    self._maestro_file_id,
                    self._maestro_filename,
                    downloaded=int(getattr(self, "n", 0) or 0),
                    total=int(self.total) if getattr(self, "total", None) else None,
                )
            except Exception:
                self._maestro_file_id = None

        def _patched_update(self, n=1):
            result = _original_update(self, n)
            try:
                file_id = getattr(self, "_maestro_file_id", None)
                if file_id:
                    _record_download_progress(
                        file_id,
                        getattr(self, "_maestro_filename", file_id),
                        downloaded=int(getattr(self, "n", 0) or 0),
                        total=int(self.total) if getattr(self, "total", None) else None,
                    )
            except Exception:
                pass
            return result

        def _patched_close(self, *args, **kwargs):
            try:
                file_id = getattr(self, "_maestro_file_id", None)
                if file_id:
                    _record_download_done(
                        file_id,
                        downloaded=int(getattr(self, "n", 0) or 0),
                        total=int(self.total) if getattr(self, "total", None) else None,
                    )
            except Exception:
                pass
            return _original_close(self, *args, **kwargs)

        Tqdm.__init__ = _patched_init
        Tqdm.update = _patched_update
        Tqdm.close = _patched_close
        Tqdm._maestro_progress_patched = True
        print("[safe_download] tqdm download hook installed")
    except Exception as e:
        print(f"[safe_download] tqdm hook install skipped: {e}")


# ── Module init ────────────────────────────────────────────────────


def install() -> None:
    """Apply all patches. Idempotent — safe to call multiple times.

    Each layer is wrapped in its own try/except so a failure in one
    doesn't take out the others. Specifically: a broken tqdm hook
    (UI progress tracking) must never prevent the timeout patches
    (the actual stall-protection) from installing.
    """
    try:
        _install_request_timeouts()
    except Exception as e:
        print(f"[safe_download] timeout patches install failed: {e}")
    try:
        _install_tqdm_hook()
    except Exception as e:
        print(f"[safe_download] tqdm hook install failed: {e}")


# Auto-install on import — this is the whole point of the module.
install()
