"""Install optional pinned WGPKernels matching the active NVIDIA runtime.

Called after torch setup by Pinokio Install/Update. Failure leaves the existing
Kitchen/Triton/PyTorch fallback available and never blocks a Maestro update.
"""
from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys

VERSION = "1.1.0"
BASE = "https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.1.0/"


def wheel_url(cuda: str, system: str, machine: str, capability: tuple[int, int]) -> str | None:
    if capability < (8, 0) or machine.lower() not in ("amd64", "x86_64"):
        return None
    tag = {"12.8": "cu128", "13.0": "cu130"}.get(cuda)
    target = {"Windows": "win_amd64", "Linux": "linux_x86_64"}.get(system)
    return f"{BASE}wgpkernels-{VERSION}%2B{tag}-py3-none-{target}.whl" if tag and target else None


def main() -> int:
    try:
        import torch
        if not torch.cuda.is_available():
            print("[WGPKernels] No CUDA GPU; existing kernels retained.")
            return 0
        url = wheel_url(torch.version.cuda or "", platform.system(), platform.machine(), torch.cuda.get_device_capability())
        if not url:
            print("[WGPKernels] No matching optional wheel; existing kernels retained.")
            return 0
        desired = VERSION + "+" + ("cu128" if torch.version.cuda == "12.8" else "cu130")
        try:
            if importlib.metadata.version("wgpkernels") == desired:
                print(f"[WGPKernels] {desired} already installed.")
                return 0
        except importlib.metadata.PackageNotFoundError:
            pass
        subprocess.run([shutil.which("uv") or "uv", "pip", "install", "--python", sys.executable, "--no-deps", url], check=True, timeout=300)
        print(f"[WGPKernels] Installed {desired}; Auto verifies it before use.")
    except Exception as error:
        print(f"[WGPKernels] Optional installation unavailable: {error}. Existing kernels retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
