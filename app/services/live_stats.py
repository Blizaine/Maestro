"""Lightweight live system telemetry for the hardware-status indicators.

Returns plain JSON numbers (CPU %, RAM, GPU utilization + VRAM) cheaply
enough to be polled every ~2 seconds by the frontend. This is the JSON
counterpart to the Gradio-era ``shared/utils/stats.py`` (which returns
HTML and blocks 1s on CPU sampling) — kept separate so the polled
endpoint stays dependency-light and non-blocking.

Design notes:
  - CPU uses ``psutil.cpu_percent(interval=None)`` (non-blocking). It
    reports utilization since the *previous* call, so at a ~2s poll
    cadence each reading is a smooth trailing average over that window.
    We prime it once at import so the first real reading is meaningful
    rather than 0.0.
  - GPU goes through NVML (``pynvml``) — the same path stats.py uses —
    which reports *device-wide* VRAM (all processes), i.e. what a
    hardware monitor should show. NVIDIA only; on any other vendor /
    when NVML is unavailable we return ``available: False`` and the UI
    simply hides the GPU gauges.
  - Every metric is wrapped in try/except so a transient driver hiccup
    degrades one number to 0 instead of failing the whole poll.
"""

from pathlib import Path

import psutil

try:
    import pynvml
    try:
        pynvml.nvmlInit()
        _nvml_ok = True
    except Exception:
        _nvml_ok = False
except Exception:  # pynvml not importable at all
    pynvml = None
    _nvml_ok = False

# Prime the non-blocking CPU sampler so the first poll returns a real
# value instead of 0.0 (psutil keeps per-process state between calls).
try:
    psutil.cpu_percent(interval=None)
except Exception:
    pass


def get_live_stats() -> dict:
    """Return a snapshot of live CPU / RAM / GPU usage as JSON-able numbers."""

    # ---- CPU (non-blocking; since last call) -------------------------
    try:
        cpu_percent = float(psutil.cpu_percent(interval=None))
    except Exception:
        cpu_percent = 0.0

    # ---- RAM ---------------------------------------------------------
    try:
        vm = psutil.virtual_memory()
        ram_percent = float(vm.percent)
        ram_used_gb = vm.used / (1024 ** 3)
        ram_total_gb = vm.total / (1024 ** 3)
    except Exception:
        ram_percent = ram_used_gb = ram_total_gb = 0.0

    # ---- Storage -----------------------------------------------------
    # Report the filesystem containing Maestro rather than assuming the
    # application lives on the system/root volume.
    try:
        disk = psutil.disk_usage(str(Path(__file__).resolve().parent))
        disk_percent = float(disk.percent)
        disk_used_gb = disk.used / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)
        disk_free_gb = disk.free / (1024 ** 3)
    except Exception:
        disk_percent = disk_used_gb = disk_total_gb = disk_free_gb = 0.0

    # ---- GPU (NVIDIA / NVML) -----------------------------------------
    gpu_available = False
    gpu_name = None
    gpu_percent = vram_used_gb = vram_total_gb = vram_percent = 0.0
    gpu_temp_c = 0.0
    vram_available = False
    temperature_available = False
    if _nvml_ok and pynvml is not None:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
            # A valid NVML device handle means the GPU exists.  Some unified-
            # memory devices (for example NVIDIA GB10) do not implement the
            # conventional NVML device-memory query.
            gpu_available = True
            try:
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8", errors="replace")
            except Exception:
                gpu_name = None

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = float(util.gpu)
            except Exception:
                gpu_percent = 0.0

            try:
                gpu_temp_c = float(
                    pynvml.nvmlDeviceGetTemperature(
                        handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                )
                temperature_available = True
            except Exception:
                # Temperature reporting is optional and must not invalidate
                # otherwise valid GPU telemetry.
                pass

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used_gb = mem.used / (1024 ** 3)
                vram_total_gb = mem.total / (1024 ** 3)
                vram_percent = (mem.used / mem.total) * 100.0 if mem.total else 0.0
                vram_available = True
            except Exception:
                # Unified-memory GPUs may not expose dedicated VRAM via NVML.
                pass
        except Exception:
            # No usable NVML device handle this tick.
            gpu_available = False

    # NVML util.gpu is a coarse, sampled compute number that under-reports for
    # diffusion (matches nvidia-smi, ~40% under load). On Windows prefer the
    # 3D-engine performance counter Task Manager reads, which tracks real
    # engine-busy time; keep the NVML value as compute_percent for the tooltip.
    gpu_compute_percent = gpu_percent
    if gpu_available:
        try:
            from services.gpu_engine_win import get_gpu_3d_utilization
            win3d = get_gpu_3d_utilization()
            if win3d is not None:
                gpu_percent = float(win3d)
        except Exception:
            pass

    return {
        "cpu": {
            "percent": round(cpu_percent, 1),
        },
        "ram": {
            "percent": round(ram_percent, 1),
            "used_gb": round(ram_used_gb, 2),
            "total_gb": round(ram_total_gb, 2),
        },
        "disk": {
            "percent": round(disk_percent, 1),
            "used_gb": round(disk_used_gb, 2),
            "total_gb": round(disk_total_gb, 2),
            "free_gb": round(disk_free_gb, 2),
        },
        "gpu": {
            "available": gpu_available,
            "name": gpu_name,
            "vram_available": vram_available,
            "percent": round(gpu_percent, 1),
            "compute_percent": round(gpu_compute_percent, 1),
            "temperature_available": temperature_available,
            "temperature_c": round(gpu_temp_c, 1),
            "vram_used_gb": round(vram_used_gb, 2),
            "vram_total_gb": round(vram_total_gb, 2),
            "vram_percent": round(vram_percent, 1),
        },
    }
