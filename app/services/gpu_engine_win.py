"""Windows-only: read the GPU 3D-engine utilization the way Task Manager does.

NVML's ``utilization.gpu`` (what nvidia-smi and our ``live_stats`` report) is
a coarse, *sampled* "was a kernel running" counter. For diffusion workloads
(many short kernels + frequent sync / mmgp-offload gaps) it under-reports —
landing around 40% while the GPU is actually pinned. Task Manager instead
reads the Windows performance counter ``\\GPU Engine(*)\\Utilization
Percentage`` and shows, per engine, how long that engine node was occupied —
which tracks much closer to "the engine is busy".

This module sums the ``engtype_3D`` instances of that counter — exactly what
Task Manager's "3D" engine number is — using the PDH API through ``ctypes``
(no pywin32 dependency). It keeps one persistent PDH query open and collects
on each call, so each reading is the average over the interval since the last
call (~the 2s poll cadence).

``get_gpu_3d_utilization()`` returns the summed 3D-engine percentage (0..100)
or ``None`` on any non-Windows platform / PDH failure, so callers fall back to
the NVML value.

Single-GPU assumption: instances of multiple physical GPUs are summed. Fine
for one GPU; a multi-GPU host would need ``phys_N`` filtering.
"""

import sys
import threading

_IS_WIN = sys.platform == "win32"

_lock = threading.Lock()
_initialized = False
_init_failed = False
_hquery = None
_hcounter = None

if _IS_WIN:
    import ctypes
    from ctypes import wintypes

    _pdh = ctypes.WinDLL("pdh.dll")

    PDH_FMT_DOUBLE = 0x00000200
    PDH_MORE_DATA = 0x800007D2  # unsigned; restypes are DWORD so this matches
    ERROR_SUCCESS = 0
    _GPU_3D_PATH = r"\GPU Engine(*engtype_3D)\Utilization Percentage"

    class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
        class _U(ctypes.Union):
            _fields_ = [
                ("longValue", ctypes.c_long),
                ("doubleValue", ctypes.c_double),
                ("largeValue", ctypes.c_longlong),
                ("AnsiStringValue", ctypes.c_char_p),
                ("WideStringValue", ctypes.c_wchar_p),
            ]

        _anonymous_ = ("u",)
        _fields_ = [("CStatus", wintypes.DWORD), ("u", _U)]

    class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
        _fields_ = [
            ("szName", ctypes.c_wchar_p),
            ("FmtValue", _PDH_FMT_COUNTERVALUE),
        ]

    # PDH returns PDH_STATUS (a LONG). The error codes have the high bit set,
    # so read them as unsigned DWORD to compare cleanly against the constants.
    _pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    _pdh.PdhOpenQueryW.restype = wintypes.DWORD
    _pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    _pdh.PdhAddEnglishCounterW.restype = wintypes.DWORD
    _pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
    _pdh.PdhCollectQueryData.restype = wintypes.DWORD
    _pdh.PdhGetFormattedCounterArrayW.argtypes = [
        ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    _pdh.PdhGetFormattedCounterArrayW.restype = wintypes.DWORD


def _ensure_init() -> bool:
    """Open the PDH query + counter once. Returns True if ready."""
    global _initialized, _init_failed, _hquery, _hcounter
    if _initialized:
        return True
    if _init_failed:
        return False
    try:
        hquery = ctypes.c_void_p()
        if _pdh.PdhOpenQueryW(None, None, ctypes.byref(hquery)) != ERROR_SUCCESS:
            _init_failed = True
            return False
        hcounter = ctypes.c_void_p()
        if _pdh.PdhAddEnglishCounterW(hquery, _GPU_3D_PATH, None, ctypes.byref(hcounter)) != ERROR_SUCCESS:
            _init_failed = True
            return False
        # Prime the query so the first real read has a baseline to diff against
        # (Utilization Percentage is a time-based counter; needs two samples).
        _pdh.PdhCollectQueryData(hquery)
        _hquery = hquery
        _hcounter = hcounter
        _initialized = True
        return True
    except Exception:
        _init_failed = True
        return False


def get_gpu_3d_utilization():
    """Summed GPU 3D-engine utilization (0..100), matching Task Manager.

    Returns None on non-Windows or any PDH failure so the caller falls back
    to the NVML value.
    """
    if not _IS_WIN:
        return None
    with _lock:
        if not _ensure_init():
            return None
        try:
            if _pdh.PdhCollectQueryData(_hquery) != ERROR_SUCCESS:
                return None

            buf_size = wintypes.DWORD(0)
            item_count = wintypes.DWORD(0)
            # Sizing call: NULL buffer -> PDH_MORE_DATA + required size/count.
            status = _pdh.PdhGetFormattedCounterArrayW(
                _hcounter, PDH_FMT_DOUBLE,
                ctypes.byref(buf_size), ctypes.byref(item_count), None,
            )
            if status != PDH_MORE_DATA or buf_size.value == 0:
                return None

            buffer = (ctypes.c_byte * buf_size.value)()
            status = _pdh.PdhGetFormattedCounterArrayW(
                _hcounter, PDH_FMT_DOUBLE,
                ctypes.byref(buf_size), ctypes.byref(item_count),
                ctypes.cast(buffer, ctypes.c_void_p),
            )
            if status != ERROR_SUCCESS:
                return None

            items = ctypes.cast(
                buffer, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W * item_count.value)
            ).contents
            total = 0.0
            for it in items:
                v = it.FmtValue.doubleValue
                if v == v and 0.0 <= v < 1.0e6:  # finite (not NaN) + sane range
                    total += v
            return min(total, 100.0)
        except Exception:
            return None
