"""
Install the pre-built llama.cpp CUDA kernels for GGUF models if a wheel
matches the current Python / PyTorch / CUDA env.

Without this, mmgp prints

    [GGUF][llama.cpp CUDA] kernels unavailable, using fallback

at every Maestro startup, and GGUF model variants (the ones with
"_gguf_" in the model_type) load with a slow CPU dequant path. With
it, GGUF dequant runs on-GPU and the warning disappears.

The wheels are published as a GitHub release by deepbeepmeep at
https://github.com/deepbeepmeep/kernels/releases. They're built per
(Python minor, PyTorch major.minor, CUDA major.minor, OS) combination,
so we detect the runtime env and pick the matching wheel. If no entry
matches -an unreleased combo -this is a soft no-op. Maestro keeps working with the slower
fallback path; only GGUF-variant generation is affected, and the
default INT8 / BF16 model variants don't use these kernels at all.

Idempotent: if `llamacpp_gguf_cuda` is already importable, exits
immediately. Re-runs cheaply on every update.

Designed to be called from install.js / update.js after torch is
installed, like:

    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "python scripts/install_gguf_kernels.py"
      }
    }

Always exits 0 -never blocks install/update on this optional kernel.
"""
from __future__ import annotations

import importlib
import re
import subprocess
import sys


# Known pre-built wheels, keyed by (python_minor, torch_major_minor, cuda).
# Update this map when deepbeepmeep publishes new combos. Keep the
# CUDA values as they appear in `torch.version.cuda` (e.g. "12.8", "13.0").
_WHEELS_WINDOWS: dict[tuple[int, str, str], str] = {
    # Python 3.10 + PyTorch 2.7.x + CUDA 12.8 -the current default
    # Maestro venv on Windows.
    (10, "2.7", "12.8"): (
        "https://github.com/deepbeepmeep/kernels/releases/download/"
        "GGUF_Kernels/llamacpp_gguf_cuda-1.0.2+torch271cu128py310-"
        "cp310-cp310-win_amd64.whl"
    ),
    # Python 3.11 + PyTorch 2.10.x + CUDA 13 -newer combo for users
    # who've upgraded their venv.
    (11, "2.10", "13.0"): (
        "https://github.com/deepbeepmeep/kernels/releases/download/"
        "GGUF_Kernels/llamacpp_gguf_cuda-1.0.2+torch210cu13py311-"
        "cp311-cp311-win_amd64.whl"
    ),
}

_WHEELS_LINUX: dict[tuple[int, str, str], str] = {
    (11, "2.10", "13.0"): (
        "https://github.com/deepbeepmeep/kernels/releases/download/"
        "GGUF_Kernels/llamacpp_gguf_cuda-1.0.2+torch210cu13py311-"
        "cp311-cp311-linux_x86_64.whl"
    ),
}


def _already_installed() -> bool:
    try:
        importlib.import_module("llamacpp_gguf_cuda")
        return True
    except ImportError:
        return False


def _detect_env() -> tuple[int, str, str, str]:
    """Return (python_minor, torch_major_minor, cuda_major_minor, os_name)."""
    py_minor = sys.version_info.minor
    try:
        import torch  # type: ignore
        # torch.__version__ looks like "2.7.1+cu128"; strip the local
        # version, then keep major.minor only.
        torch_core = torch.__version__.split("+")[0]
        m = re.match(r"^(\d+\.\d+)", torch_core)
        torch_short = m.group(1) if m else torch_core
        # torch.version.cuda is "12.8", "13.0", etc -already major.minor.
        cuda = (torch.version.cuda or "").strip()
    except Exception:
        torch_short = ""
        cuda = ""
    if sys.platform == "win32":
        os_name = "windows"
    elif sys.platform.startswith("linux"):
        os_name = "linux"
    else:
        os_name = sys.platform
    return py_minor, torch_short, cuda, os_name


def _pick_wheel_url(py_minor: int, torch_short: str, cuda: str, os_name: str) -> str | None:
    key = (py_minor, torch_short, cuda)
    if os_name == "windows":
        return _WHEELS_WINDOWS.get(key)
    if os_name == "linux":
        return _WHEELS_LINUX.get(key)
    return None


def main() -> int:
    if _already_installed():
        print("[GGUF kernels] llamacpp_gguf_cuda already installed - skipping.")
        return 0

    py_minor, torch_short, cuda, os_name = _detect_env()
    env_label = (
        f"Python 3.{py_minor}, PyTorch {torch_short or '?'}, "
        f"CUDA {cuda or '?'}, {os_name}"
    )
    print(f"[GGUF kernels] Detected env: {env_label}")

    url = _pick_wheel_url(py_minor, torch_short, cuda, os_name)
    if not url:
        print(
            "[GGUF kernels] No pre-built wheel for this env combination. "
            "GGUF model variants will use the slower CPU dequant fallback at "
            "load time. INT8 / BF16 / FP8 variants are unaffected -they "
            "don't use these kernels."
        )
        return 0

    print(f"[GGUF kernels] Installing: {url}")
    cmd = [sys.executable, "-m", "pip", "install", url]
    try:
        subprocess.run(cmd, check=True)
        print(
            "[GGUF kernels] Install OK. The "
            "'kernels unavailable, using fallback' warning should be gone "
            "on next Maestro startup."
        )
        return 0
    except subprocess.CalledProcessError as e:
        print(
            f"[GGUF kernels] Install failed (exit {e.returncode}). GGUF "
            "models will use the slower fallback; this is non-fatal."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
