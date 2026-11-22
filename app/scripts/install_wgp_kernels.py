"""Install optional, pinned WGPKernels matching a supported NVIDIA runtime.

Called after torch setup by Pinokio Install/Update. Unsupported environments or
failed upgrades leave the existing Kitchen/Triton/PyTorch fallbacks available.
"""
from __future__ import annotations

import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys

VERSION = "1.5.0"
BASE = f"https://github.com/deepbeepmeep/WGPKernels/releases/download/v{VERSION}/"

# Exact stack combinations documented and validated by the v1.5.0 release.
# The wheels are `py3-none`, but only these app-supported Python/Torch stacks
# have release validation: Python 3.10 + Torch 2.7.1/CUDA 12.8 and Python 3.11
# + Torch 2.10.0/CUDA 13.0.
_STACK_TAGS = {
    (10, "2.7.1", "12.8"): "cu128",
    (11, "2.10.0", "13.0"): "cu130",
}
_WHEEL_SHA256 = {
    ("cu128", "win_amd64"): "3c471e4de8c6dc89653b4bcfaa0ee39cfa1b538e0303f733f5dc02cdb2078335",
    ("cu128", "linux_x86_64"): "dff268568701dd2485971d5ffcdc29d61577689092f89a558ba557dd69dc1a17",
    ("cu130", "win_amd64"): "013a406e47cceca81282e73027c602e2f8db9c6d4bfebba61afb39f7b7551893",
    ("cu130", "linux_x86_64"): "5206cd03c1f9881c4e257f59b0f6d40f131b4cdd960d400e8869db6a70c41406",
}


def _linux_glibc_supported(system: str, libc: tuple[str, str] | None) -> bool:
    """WGPKernels Linux CUDA assets require glibc 2.34 or newer."""
    if system != "Linux":
        return True
    if not libc or libc[0].lower() != "glibc":
        return False
    match = re.match(r"^(\d+)\.(\d+)", libc[1])
    return bool(match and tuple(map(int, match.groups())) >= (2, 34))


def wheel_url(
    cuda: str,
    system: str,
    machine: str,
    capability: tuple[int, int],
    python_minor: int,
    torch_version: str,
    libc: tuple[str, str] | None = None,
) -> str | None:
    """Return a hash-pinned wheel URL only for a supported runtime stack."""
    if capability < (8, 0) or machine.lower() not in ("amd64", "x86_64"):
        return None
    if system == "Windows":
        wheel_platform = "win_amd64"
    elif system == "Linux":
        if not _linux_glibc_supported(system, libc):
            return None
        wheel_platform = "linux_x86_64"
    else:
        return None

    tag = _STACK_TAGS.get((python_minor, torch_version, cuda))
    if not tag:
        return None
    digest = _WHEEL_SHA256.get((tag, wheel_platform))
    if not digest:
        return None
    filename = f"wgpkernels-{VERSION}%2B{tag}-py3-none-{wheel_platform}.whl"
    return f"{BASE}{filename}#sha256={digest}"


def _newer_public_release_is_installed(installed: str, desired: str) -> bool:
    """Keep a manually installed newer release, while correcting stack labels."""
    try:
        from packaging.version import Version

        installed_version = Version(installed)
        desired_version = Version(desired)
        same_variant = installed_version.local == desired_version.local
        return same_variant and Version(installed_version.public) > Version(desired_version.public)
    except Exception:
        return False


def main() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            print("[WGPKernels] No CUDA GPU; existing kernels retained.")
            return 0
        torch_version = torch.__version__.split("+", 1)[0]
        cuda = (torch.version.cuda or "").strip()
        url = wheel_url(
            cuda,
            platform.system(),
            platform.machine(),
            torch.cuda.get_device_capability(),
            sys.version_info.minor,
            torch_version,
            platform.libc_ver(),
        )
        if not url:
            print("[WGPKernels] No matching optional wheel; existing kernels retained.")
            return 0

        tag = _STACK_TAGS[(sys.version_info.minor, torch_version, cuda)]
        desired = f"{VERSION}+{tag}"
        try:
            installed = importlib.metadata.version("wgpkernels")
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed == desired:
            print(f"[WGPKernels] {desired} already installed.")
            return 0
        if installed and _newer_public_release_is_installed(installed, desired):
            print(f"[WGPKernels] {installed} is newer than the optional {desired} pin; keeping it.")
            return 0

        subprocess.run(
            [shutil.which("uv") or "uv", "pip", "install", "--python", sys.executable, "--no-deps", url],
            check=True,
            timeout=300,
        )
        print(f"[WGPKernels] Installed {desired}; Auto verifies it before use.")
    except Exception as error:
        print(f"[WGPKernels] Optional installation unavailable: {error}. Existing kernels retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
