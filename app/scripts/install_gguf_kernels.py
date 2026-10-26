"""Install optional, pre-built llama.cpp CUDA kernels for GGUF models.

The pin is selected only for the Python/PyTorch/CUDA combinations published in
the GGUF 1.0.29 release. Unsupported environments and install failures keep the
existing slower GGUF fallback; PyTorch dependencies are never changed.
"""
from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
import sys

VERSION = "1.0.29"
RELEASE_URL = "https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.29/"

# key = (Python minor, full PyTorch version, CUDA runtime)
_BUILD_TAGS: dict[tuple[int, str, str], str] = {
    (10, "2.7.1", "12.8"): "torch271cu128py310",
    (11, "2.10.0", "13.0"): "torch210cu130py311",
}
_WHEEL_SHA256 = {
    ("torch271cu128py310", "windows"): "42ed4fd96ee9756b89b1c27cb85836bbe1eec09501da6df3410c7db9df82cf4d",
    ("torch271cu128py310", "linux"): "72da2794ff79bef5e7d2db41959382a1dfb5ad6a3797c58b5abeb5b6247b50f2",
    ("torch210cu130py311", "windows"): "6ea255647d2b9ee6143b975af6b1f1abc6d7256622fd8e212000d46bb4ca641b",
    ("torch210cu130py311", "linux"): "9509d38b737e9b825dd8d523e27a74d534eec2d30394b0c2c3fe7381011ea014",
}


def _glibc_supported(libc: tuple[str, str] | None) -> bool:
    """GGUF Linux CUDA assets require glibc 2.28 or newer."""
    if not libc or libc[0].lower() != "glibc":
        return False
    match = re.match(r"^(\d+)\.(\d+)", libc[1])
    return bool(match and tuple(map(int, match.groups())) >= (2, 28))


def _wheel_spec(
    py_minor: int,
    torch_version: str,
    cuda: str,
    os_name: str,
    machine: str,
    implementation: str,
    libc: tuple[str, str] | None,
) -> tuple[str, str] | None:
    """Return (hash-pinned URL, expected distribution version), if supported."""
    if implementation.lower() != "cpython" or machine.lower() not in ("amd64", "x86_64"):
        return None
    build = _BUILD_TAGS.get((py_minor, torch_version, cuda))
    if not build:
        return None

    if os_name == "windows":
        wheel_platform = "win_amd64"
        platform_key = "windows"
        cp_tag = f"cp3{py_minor}"
    elif os_name == "linux":
        if not _glibc_supported(libc):
            return None
        wheel_platform = "linux_x86_64"
        platform_key = "linux"
        cp_tag = f"cp3{py_minor}"
    else:
        return None

    digest = _WHEEL_SHA256[(build, platform_key)]
    filename = f"llamacpp_gguf_cuda-{VERSION}%2B{build}-{cp_tag}-{cp_tag}-{wheel_platform}.whl"
    url = f"{RELEASE_URL}{filename}#sha256={digest}"
    distribution_version = f"{VERSION}+{build}"
    return url, distribution_version


def _detect_env() -> tuple[int, str, str, str, str, str, tuple[str, str]]:
    """Return Python minor, full Torch version, CUDA, OS, machine, impl and libc."""
    py_minor = sys.version_info.minor
    try:
        import torch  # type: ignore

        torch_version = torch.__version__.split("+", 1)[0]
        cuda = (torch.version.cuda or "").strip()
    except Exception:
        torch_version = ""
        cuda = ""

    if sys.platform == "win32":
        os_name = "windows"
    elif sys.platform.startswith("linux"):
        os_name = "linux"
    else:
        os_name = sys.platform
    return (
        py_minor,
        torch_version,
        cuda,
        os_name,
        platform.machine(),
        sys.implementation.name,
        platform.libc_ver(),
    )


def _newer_public_release_is_installed(installed: str, desired: str) -> bool:
    """Keep a user's newer manual pin, but replace stale or wrong-build wheels."""
    if installed == desired:
        return False
    try:
        from packaging.version import Version

        installed_version = Version(installed)
        desired_version = Version(desired)
    except Exception:
        return False
    # Different local build labels for the same release need a repair for this
    # environment. A newer public release is left alone to avoid downgrading a
    # package the user installed manually.
    same_build = installed_version.local == desired_version.local
    return same_build and Version(installed_version.public) > Version(desired_version.public)


def _installed_version() -> str | None:
    try:
        return importlib.metadata.version("llamacpp_gguf_cuda")
    except importlib.metadata.PackageNotFoundError:
        # An importable stale module is not enough to skip the pin update.
        return None
    except Exception:
        # Bad metadata should not turn this optional install into a hard failure.
        return None


def main() -> int:
    detected = _detect_env()
    py_minor, torch_version, cuda, os_name, machine, implementation, libc = detected
    env_label = (
        f"Python 3.{py_minor}, PyTorch {torch_version or '?'}, "
        f"CUDA {cuda or '?'}, {os_name}"
    )
    print(f"[GGUF kernels] Detected env: {env_label}")

    spec = _wheel_spec(py_minor, torch_version, cuda, os_name, machine, implementation, libc)
    if not spec:
        print(
            "[GGUF kernels] No pre-built wheel for this env combination. "
            "GGUF variants keep using the existing fallback."
        )
        return 0
    url, desired = spec

    installed = _installed_version()
    if installed and _newer_public_release_is_installed(installed, desired):
        print(f"[GGUF kernels] {installed} is newer than the optional {desired} pin; keeping it.")
        return 0
    if installed == desired:
        print(f"[GGUF kernels] {desired} already installed.")
        return 0

    print(f"[GGUF kernels] Installing: {url}")
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", url]
    try:
        subprocess.run(cmd, check=True, timeout=300)
        print("[GGUF kernels] Install OK. The fallback warning should be gone on next Maestro startup.")
    except Exception as error:
        print(f"[GGUF kernels] Optional install failed: {error}. GGUF models keep their fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
