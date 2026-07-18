"""
Windows-safe file serving and deletion.

THE PROBLEM (Windows only):
Windows uses mandatory file locking. When a process opens a file, other
processes can't delete or rename it unless the opener passed
FILE_SHARE_DELETE to CreateFileW. Python's built-in open() does NOT
pass FILE_SHARE_DELETE — it uses FILE_SHARE_READ only.

Concrete consequence in this app:
- The browser plays a video clip via a <video> element. FastAPI's
  FileResponse opens the file with default share mode and streams it.
  HTTP keep-alive keeps the connection (and file handle) open for
  many seconds, sometimes minutes.
- The user clicks "delete" in the gallery. Backend calls os.remove
  → PermissionError. Falls back to os.rename → ALSO PermissionError
  (rename has the same share requirement as delete).
- The user is stuck with a "deleted" file that can't actually be
  removed without closing the entire app.

THE FIX (this module):
Open all served files with FILE_SHARE_READ | FILE_SHARE_WRITE |
FILE_SHARE_DELETE via CreateFileW. With share-delete set, Windows
behaves like POSIX — deletes and renames succeed immediately, the
inode lives until the last handle closes, and disk space is reclaimed
naturally.

USAGE:
- Replace `FileResponse(path)` with `share_delete_file_response(path)`
- Replace `os.remove(path)` in the gallery delete path with `safe_delete(path)`

Linux/macOS: these helpers fall back to standard FileResponse / os.remove,
since POSIX file semantics already give us this behavior for free.
"""
from __future__ import annotations

import os
import sys
import time
import threading
from typing import Optional

from starlette.responses import FileResponse, Response, StreamingResponse
from starlette.requests import Request
from starlette.types import Send, Receive, Scope


_IS_WINDOWS = sys.platform == "win32"


# ── Windows-only: low-level file open with FILE_SHARE_DELETE ────────────
if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.restype = wintypes.HANDLE
    _CreateFileW.argtypes = [
        wintypes.LPCWSTR,   # lpFileName
        wintypes.DWORD,     # dwDesiredAccess
        wintypes.DWORD,     # dwShareMode
        ctypes.c_void_p,    # lpSecurityAttributes
        wintypes.DWORD,     # dwCreationDisposition
        wintypes.DWORD,     # dwFlagsAndAttributes
        wintypes.HANDLE,    # hTemplateFile
    ]

    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    _FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000  # hint for OS prefetcher
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    def _open_share_delete(path: str):
        """Open a file for binary read with FILE_SHARE_DELETE.

        Returns a Python file object. The OS will allow other processes
        to delete or rename this file even while we hold it open — the
        actual disk space is freed when the last handle closes.
        """
        handle = _CreateFileW(
            path,
            _GENERIC_READ,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_SEQUENTIAL_SCAN,
            None,
        )
        # CreateFileW returns INVALID_HANDLE_VALUE (-1 cast to HANDLE)
        # on failure. ctypes returns a small int; compare against the
        # 32-bit two's-complement representation we computed above.
        if handle is None or (handle & 0xFFFFFFFF) == 0xFFFFFFFF:
            err = ctypes.get_last_error()
            raise OSError(err, f"CreateFileW failed for {path}")
        # Bridge the Win32 HANDLE to a Python file descriptor, then to
        # a buffered file object. msvcrt.open_osfhandle takes ownership
        # of the handle — closing the file object closes the handle.
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except Exception:
            _kernel32.CloseHandle(handle)
            raise
        # buffering=-1 picks default block size (typically 8KB); good
        # for streaming. We pass binary read mode.
        return os.fdopen(fd, "rb", buffering=-1)

else:
    def _open_share_delete(path: str):  # type: ignore[misc]
        # POSIX: regular open already allows concurrent delete.
        return open(path, "rb")


# ── Custom Starlette response that opens files with share-delete ─────────
# The default Starlette FileResponse uses anyio.open_file which delegates
# to Python's open(). We need our share-delete open instead.
#
# Implementation note: we keep this minimal. We support:
#   - Full file streaming
#   - HTTP Range requests (essential for video <video> seeking)
#   - Content-Type, Content-Length, Last-Modified, ETag headers
#
# We deliberately do NOT support: aiofiles async I/O. The Win32 file
# handle is sync. For typical gallery loads (a few thumbnails + one
# active video) this is fine — the chunks are small and the OS does
# the actual I/O concurrently. If this ever becomes a bottleneck we
# can use a thread pool.

_CHUNK_SIZE = 64 * 1024  # 64 KB per chunk


class ShareDeleteFileResponse(Response):
    """File response that opens with FILE_SHARE_DELETE on Windows so the
    file can be deleted/renamed even while we're streaming it.

    On non-Windows platforms this is functionally identical to
    Starlette's FileResponse since POSIX already allows concurrent
    deletes — we still use the share-delete codepath (which falls back
    to plain open) to keep behavior consistent across platforms.
    """

    chunk_size = _CHUNK_SIZE

    def __init__(
        self,
        path: str,
        media_type: Optional[str] = None,
        filename: Optional[str] = None,
        method: Optional[str] = None,
        stat_result: Optional[os.stat_result] = None,
    ):
        self.path = path
        self.filename = filename
        self.send_header_only = method is not None and method.upper() == "HEAD"

        if media_type is None:
            # Sniff from extension. Mirrors Starlette's FileResponse default.
            import mimetypes
            media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.media_type = media_type

        if stat_result is None:
            try:
                stat_result = os.stat(path)
            except FileNotFoundError:
                stat_result = None
        self.stat_result = stat_result

        # Header-only init; body is streamed in __call__.
        super().__init__(status_code=200, media_type=self.media_type)
        self._set_headers()

    def _set_headers(self) -> None:
        if self.stat_result is None:
            return
        from email.utils import formatdate
        size = self.stat_result.st_size
        mtime = self.stat_result.st_mtime
        # Weak ETag from inode + size + mtime (similar to Starlette's).
        etag_base = f"{self.stat_result.st_mtime_ns}-{size}-{self.stat_result.st_ino}"
        import hashlib
        etag = hashlib.md5(etag_base.encode()).hexdigest()
        self.headers.setdefault("content-length", str(size))
        self.headers.setdefault("last-modified", formatdate(mtime, usegmt=True))
        self.headers.setdefault("etag", f'"{etag}"')
        self.headers.setdefault("accept-ranges", "bytes")
        if self.filename:
            from urllib.parse import quote
            content_disposition = f'attachment; filename*=utf-8\'\'{quote(self.filename)}'
            self.headers.setdefault("content-disposition", content_disposition)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Look for Range header — required for video seeking in <video>.
        range_header: Optional[str] = None
        for name, value in scope.get("headers", []):
            if name == b"range":
                range_header = value.decode("latin-1")
                break

        if not os.path.isfile(self.path):
            await self._send_error(send, 404)
            return

        # Open the file FIRST and use os.fstat on the open handle to get the
        # CURRENT size. Avoids the TOCTOU race where self.stat_result was
        # captured at __init__ time but the file changed between then and now
        # (rapid regeneration overwriting the same output path, antivirus
        # touching, file still being flushed, etc.). Mismatched
        # Content-Length → h11 "Too little data for declared Content-Length"
        # error in the connection writer.
        try:
            f = _open_share_delete(self.path)
        except FileNotFoundError:
            await self._send_error(send, 404)
            return
        except OSError:
            await self._send_error(send, 404)
            return

        try:
            try:
                size = os.fstat(f.fileno()).st_size
            except OSError:
                # Fall back to stat_result if fstat fails (shouldn't happen)
                size = self.stat_result.st_size if self.stat_result else 0

            start, end = 0, size - 1
            status = 200

            if range_header:
                try:
                    start, end = _parse_range(range_header, size)
                    status = 206
                except _RangeError:
                    # Range Not Satisfiable
                    self.headers["content-range"] = f"bytes */{size}"
                    await self._send_error(send, 416)
                    return

            length = end - start + 1
            self.headers["content-length"] = str(length)
            if status == 206:
                self.headers["content-range"] = f"bytes {start}-{end}/{size}"

            # Send headers
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": self.raw_headers,
            })

            if self.send_header_only:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return

            # Stream the file body
            if start:
                f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(self.chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                await send({
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": remaining > 0,
                })
            # Defense-in-depth: if we hit EOF before remaining was satisfied
            # (file was truncated AFTER we opened it — possible on Windows
            # with the FILE_SHARE_WRITE flag used by _open_share_delete),
            # pad with zeros to satisfy the declared Content-Length. The
            # alternative — sending an empty body marker with bytes still
            # owed — triggers `h11._util.LocalProtocolError: Too little data
            # for declared Content-Length` which spams the console even
            # though the client already got the meaningful bytes.
            if remaining > 0:
                _PAD = b"\x00" * min(self.chunk_size, remaining)
                while remaining > 0:
                    pad = _PAD if remaining >= self.chunk_size else b"\x00" * remaining
                    remaining -= len(pad)
                    await send({
                        "type": "http.response.body",
                        "body": pad,
                        "more_body": remaining > 0,
                    })
        finally:
            f.close()

    async def _send_error(self, send: Send, status: int) -> None:
        msg = {404: b"Not Found", 416: b"Range Not Satisfiable"}.get(status, b"Error")
        body = msg
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body, "more_body": False})


class _RangeError(Exception):
    pass


def _parse_range(header: str, size: int) -> tuple[int, int]:
    """Parse a single-range Range header. Returns (start, end_inclusive).

    Multi-range (`bytes=0-100,200-300`) is not supported — browsers
    rarely use it for media playback. Falls back to first range.
    """
    if not header.startswith("bytes="):
        raise _RangeError("invalid units")
    spec = header[6:].split(",", 1)[0].strip()
    if "-" not in spec:
        raise _RangeError("missing dash")
    s, e = spec.split("-", 1)
    if s and e:
        start, end = int(s), int(e)
    elif s and not e:
        start, end = int(s), size - 1
    elif not s and e:
        # Suffix range: "bytes=-500" = last 500 bytes
        suffix = int(e)
        start, end = max(0, size - suffix), size - 1
    else:
        raise _RangeError("empty range")
    if start < 0 or end >= size or start > end:
        raise _RangeError("out of bounds")
    return start, end


def share_delete_file_response(
    path: str,
    media_type: Optional[str] = None,
    filename: Optional[str] = None,
    method: Optional[str] = None,
) -> Response:
    """Drop-in replacement for FastAPI's FileResponse that uses
    FILE_SHARE_DELETE on Windows.

    Use this for any output-file endpoint where the gallery needs to
    delete the file later: gallery files, output files, generated
    thumbnails, etc.
    """
    return ShareDeleteFileResponse(
        path=path, media_type=media_type, filename=filename, method=method
    )


# ── Robust delete helper ────────────────────────────────────────────────
# Even with share-delete on the serve side, edge cases remain:
#   - The file might be open by some OTHER process not under our control
#     (Windows Explorer thumbnail cache, antivirus scanner, the user's
#     own file viewer, etc.)
#   - The user might have an out-of-band Python script (or older sessions)
#     holding the file
#
# safe_delete() handles these by:
#   1. Trying os.remove with a short retry (covers transient AV scans)
#   2. Falling back to renaming the file to a hidden ".trash_<ts>_<name>"
#      sibling — this works whenever os.remove works (same share rules)
#      AND additionally works when the file is held by anything that
#      DID open with share-delete
#   3. Adding the renamed file to a deferred-delete queue that retries
#      every few seconds for several minutes
#   4. Returning success to the caller IMMEDIATELY — the gallery removes
#      the file from view; the on-disk cleanup happens in background


_DEFERRED_DELETE_QUEUE: list[tuple[str, float]] = []
_DEFERRED_LOCK = threading.Lock()
_DEFERRED_THREAD_STARTED = False


def _deferred_delete_worker():
    """Periodically retry deletes from the queue. Never exits.

    Entries may be files OR directories (trash-renamed folders from
    safe_delete_dir) — directories are removed with rmtree.
    """
    import shutil
    while True:
        time.sleep(5)
        with _DEFERRED_LOCK:
            queue_snapshot = list(_DEFERRED_DELETE_QUEUE)
        if not queue_snapshot:
            continue
        still_pending: list[tuple[str, float]] = []
        for path, queued_at in queue_snapshot:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"[safe_delete] Deferred cleanup succeeded: {os.path.basename(path)}")
            except FileNotFoundError:
                pass  # Already gone — drop from queue
            except OSError:
                # Still locked. Keep in queue if not too old.
                if time.time() - queued_at < 30 * 60:  # 30 min max
                    still_pending.append((path, queued_at))
                else:
                    print(f"[safe_delete] Giving up on {os.path.basename(path)} after 30 min")
        with _DEFERRED_LOCK:
            # Replace contents — preserving items we kept and any added
            # while we were processing.
            new_items = [item for item in _DEFERRED_DELETE_QUEUE if item not in queue_snapshot]
            _DEFERRED_DELETE_QUEUE.clear()
            _DEFERRED_DELETE_QUEUE.extend(still_pending + new_items)


def _ensure_deferred_thread() -> None:
    global _DEFERRED_THREAD_STARTED
    if _DEFERRED_THREAD_STARTED:
        return
    _DEFERRED_THREAD_STARTED = True
    t = threading.Thread(target=_deferred_delete_worker, daemon=True, name="safe_delete_worker")
    t.start()


def safe_delete(path: str, *, retries: int = 3, retry_delay: float = 0.2) -> dict:
    """Delete a file with Windows-friendly fallbacks.

    Returns a dict describing the outcome:
        {"deleted": True}                              — fully removed
        {"deleted": True, "deferred": True,
         "trash_path": "..."}                          — renamed, will clean up later
        {"deleted": False, "reason": "not_found"}     — wasn't there
        {"deleted": False, "reason": "locked"}        — couldn't even rename
    """
    if not os.path.isfile(path):
        return {"deleted": False, "reason": "not_found"}

    # Stage 1: Direct delete with retries
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            os.remove(path)
            return {"deleted": True}
        except FileNotFoundError:
            return {"deleted": False, "reason": "not_found"}
        except PermissionError as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(retry_delay)

    # Stage 2: Rename to hidden trash sibling. With share-delete on the
    # serve path, this almost always succeeds even when the browser is
    # actively streaming. The renamed file disappears from gallery
    # listings (which filter dotfiles) so the user sees it as deleted.
    try:
        directory = os.path.dirname(path) or "."
        basename = os.path.basename(path)
        trash_path = os.path.join(directory, f".trash_{int(time.time() * 1000)}_{basename}")
        os.rename(path, trash_path)
        # Queue for background cleanup
        with _DEFERRED_LOCK:
            _DEFERRED_DELETE_QUEUE.append((trash_path, time.time()))
        _ensure_deferred_thread()
        return {"deleted": True, "deferred": True, "trash_path": trash_path}
    except Exception as rename_err:
        # Stage 3: Even rename failed. This means SOMETHING is holding
        # the file open without FILE_SHARE_DELETE. Report failure so
        # the caller can decide what to do (typically still report
        # success to the UI and try again later).
        print(f"[safe_delete] Failed: {os.path.basename(path)} (remove: {last_err}, rename: {rename_err})")
        return {"deleted": False, "reason": "locked"}


def is_trash_name(name: str) -> bool:
    """Return True if `name` looks like a deferred-delete trash file.

    Use this to filter trash entries out of gallery listings so the
    user doesn't see the in-flight cleanup files.
    """
    return name.startswith(".trash_")


# Serializes .favorites.json read-modify-write cycles. Two writers exist
# (launch.py's favorites endpoints and director_pipeline's delete sweep),
# both on FastAPI's threadpool — without the lock an interleave can write
# a stale favorites set back, resurrecting entries for deleted files.
# RLock: RMW call sites hold it across load+modify+save, and the load/save
# helpers re-acquire it internally.
favorites_lock = threading.RLock()


def safe_delete_dir(path: str) -> dict:
    """Delete a directory tree with the same Windows-friendly fallbacks
    as safe_delete.

    Files are deleted with a single direct attempt (bulk callers should
    not sleep through per-file retry backoff); anything locked gets the
    trash-rename treatment. If the directory itself still can't be
    removed, the WHOLE folder is renamed to a hidden .trash_* sibling
    and queued for the deferred worker (which handles directories) —
    per-file queue entries inside it are dropped at that point, since
    renaming the folder invalidates their paths and the folder-level
    rmtree covers them. Listings never see the renamed folder (all
    filter dot-names) and sweep_trash reclaims it after a restart.

    Returns {"removed": bool, "files_deleted": int, "files_deferred": int,
    "errors": [...]} — "removed" means gone from the original path
    (possibly hidden-pending-cleanup), matching what a UI should show.
    """
    import shutil
    deleted = 0
    deferred = 0
    errors: list[str] = []
    for root, dirs, files in os.walk(path, topdown=False):
        for fname in files:
            result = safe_delete(os.path.join(root, fname), retries=1)
            if result.get("deferred"):
                deferred += 1
            elif result.get("deleted"):
                deleted += 1
            elif result.get("reason") != "not_found":
                errors.append(fname)
        for dname in dirs:
            try:
                os.rmdir(os.path.join(root, dname))
            except OSError:
                pass
    try:
        os.rmdir(path)
        return {"removed": True, "files_deleted": deleted, "files_deferred": deferred, "errors": errors}
    except FileNotFoundError:
        return {"removed": True, "files_deleted": deleted, "files_deferred": deferred, "errors": errors}
    except OSError:
        pass
    parent = os.path.dirname(path) or "."
    hidden = os.path.join(parent, f".trash_{int(time.time() * 1000)}_{os.path.basename(path)}")
    try:
        os.rename(path, hidden)
    except OSError as exc:
        errors.append(f"folder: {exc}")
        return {"removed": False, "files_deleted": deleted, "files_deferred": deferred, "errors": errors}
    with _DEFERRED_LOCK:
        stale_prefix = os.path.normcase(os.path.abspath(path)) + os.sep
        _DEFERRED_DELETE_QUEUE[:] = [
            (p, t) for (p, t) in _DEFERRED_DELETE_QUEUE
            if not os.path.normcase(os.path.abspath(p)).startswith(stale_prefix)
        ]
        _DEFERRED_DELETE_QUEUE.append((hidden, time.time()))
    _ensure_deferred_thread()
    return {"removed": True, "files_deleted": deleted, "files_deferred": deferred, "errors": errors}


def sweep_trash(base: str) -> int:
    """Startup sweep: reclaim leftover .trash_* entries one level under
    `base` — renamed by earlier runs whose deferred worker never finished
    (server restarted while files were locked). Best-effort; anything
    still locked goes back on the deferred queue."""
    import shutil
    count = 0
    try:
        entries = list(os.scandir(base))
    except OSError:
        return 0
    for entry in entries:
        if not is_trash_name(entry.name):
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
            count += 1
        except OSError:
            with _DEFERRED_LOCK:
                _DEFERRED_DELETE_QUEUE.append((entry.path, time.time()))
            _ensure_deferred_thread()
    if count:
        print(f"[safe_delete] Startup sweep reclaimed {count} leftover trash entries in {base}")
    return count


def recycle_file(path: str) -> bool:
    """Send a file to the Windows Recycle Bin (SHFileOperationW with
    FOF_ALLOWUNDO). Returns True only when the file is actually gone from
    its original location. Used for deletions in OTHER installs' folders,
    where an undo path matters more than reclaiming space instantly.
    Falls back to False (caller decides) on non-Windows or API failure —
    including oversized files the Bin silently refuses."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", ctypes.c_uint16),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        FO_DELETE = 3
        FOF_ALLOWUNDO = 0x40
        FOF_NOCONFIRMATION = 0x10
        FOF_SILENT = 0x4
        FOF_NOERRORUI = 0x400
        op = SHFILEOPSTRUCTW()
        op.wFunc = FO_DELETE
        op.pFrom = os.path.abspath(path) + "\x00"  # double-null via LPCWSTR terminator
        op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        return result == 0 and not op.fAnyOperationsAborted and not os.path.exists(path)
    except Exception:
        return False


def safe_join_under(base: str, *parts: str):
    """Join `parts` under `base`; return the absolute path only if it stays
    inside `base` after resolving symlinks, else None. Shared with services
    that cannot import launch.py's _safe_join (circular import)."""
    try:
        base_real = os.path.realpath(base)
        joined = os.path.realpath(os.path.join(base_real, *parts))
        if os.name == "nt":
            if os.path.normcase(joined) != os.path.normcase(base_real) and \
               not os.path.normcase(joined).startswith(os.path.normcase(base_real) + os.sep):
                return None
        else:
            if joined != base_real and not joined.startswith(base_real + os.sep):
                return None
        return joined
    except (ValueError, OSError):
        return None
