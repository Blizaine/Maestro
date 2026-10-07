"""Early allocator selection adapted from WanGP 17.01, before CUDA allocations.

Upstream: 0e58385fbde7ff102d276e4a9e490845de76b4ea, WanGP Community License 2.0.
Maestro adaptations (2026-10-05): preserve allocator defaults, idempotent launch,
report active/fallback state to settings, and keep unsupported systems bootable.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time

VRAM_ALLOCATOR_CHOICES = ("default", "vmm", "vmm_spill")
CUDA_STACK_BYTES = 256
_startup_applied = False
_vram_debug = False
_startup_state = {"requested": "default", "active": "default", "fallback_reason": None}
_allocator_cli_override = None


def _argv_value(argv, option):
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return None


def requested_vram_allocator(argv, config_filename):
    value = _argv_value(argv, "--vram-allocator")
    if value is None:
        config_dir = _argv_value(argv, "--config")
        candidates = ([os.path.join(os.path.abspath(config_dir), os.path.basename(config_filename))] if config_dir else []) + [config_filename]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as reader:
                        value = json.load(reader).get("vram_allocator")
                except (OSError, ValueError, AttributeError) as error:
                    print(f"[VRAM] Could not read allocator preference: {error}")
                break
    return value or "default"


def apply_startup_settings(argv, config_filename):
    global _startup_applied, _vram_debug, _allocator_cli_override
    if _startup_applied:
        return dict(_startup_state)
    import torch
    _allocator_cli_override = _argv_value(argv, "--vram-allocator")
    requested = requested_vram_allocator(argv, config_filename)
    _startup_state["requested"] = requested
    _startup_applied = True
    if requested not in VRAM_ALLOCATOR_CHOICES:
        _startup_state["fallback_reason"] = f"Unknown allocator {requested!r}"
        print(f"[VRAM] {_startup_state['fallback_reason']}; using PyTorch's allocator")
        requested = "default"
    if torch.version.hip is not None or not torch.cuda.is_available():
        if requested != "default":
            _startup_state["fallback_reason"] = "MMGP's allocator requires an NVIDIA CUDA GPU"
        return dict(_startup_state)
    if requested in ("vmm", "vmm_spill"):
        try:
            from mmgp import allocator
            allocator.install(spill=requested == "vmm_spill")
        except (RuntimeError, OSError, ImportError, AttributeError) as error:
            _startup_state["fallback_reason"] = str(error)
            print(f"[VRAM] MMGP Optimized allocator unavailable ({error}); using PyTorch's allocator")
        else:
            _startup_state["active"] = requested
            print("[VRAM] MMGP Optimized allocator" + (" with RAM spilling" if requested == "vmm_spill" else ""))
    debug_mb = float(_argv_value(argv, "--vram-debug") or 0)
    if debug_mb > 0:
        if _startup_state["active"] not in ("vmm", "vmm_spill"):
            print("[VRAM] --vram-debug needs the MMGP allocator; debug reporting is disabled")
        else:
            from mmgp.allocator import debug
            debug.start(min_mb=debug_mb)
            _vram_debug = True
    # A context has to exist before this driver call. No model or kernel has
    # loaded yet; the allocator above is already installed if it was requested.
    try:
        stack_bytes = int(os.environ.get("WANGP_CUDA_STACK_BYTES", CUDA_STACK_BYTES))
        if stack_bytes > 0:
            target = _argv_value(argv, "--gpu") or "cuda"
            torch.zeros(1, device=target)
            driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
            result = driver.cuCtxSetLimit(0, ctypes.c_size_t(stack_bytes))
            if result != 0:
                raise RuntimeError(f"CUDA driver returned {result}")
    except (RuntimeError, OSError, ValueError, AttributeError) as error:
        print(f"[VRAM] CUDA stack reserve adjustment skipped: {error}")
    return dict(_startup_state)


def allocator_status(config):
    requested = config.get("vram_allocator", "default")
    return {
        "vram_allocator_active": _startup_state["active"],
        "vram_allocator_fallback_reason": _startup_state["fallback_reason"],
        "vram_allocator_restart_required": _allocator_cli_override is None and requested != _startup_state["requested"],
        "vram_allocator_cli_override": _allocator_cli_override,
    }


def write_vram_debug_report(output_dir, label):
    if not _vram_debug:
        return
    from mmgp.allocator import debug
    path = os.path.join(output_dir, "vram_debug", f"{time.strftime('%Y-%m-%d-%Hh%Mm%Ss')}_{label or 'generation'}.json")
    debug.report(path)
    debug.reset()
    print(f"[VRAM debug] Report: {path}")
