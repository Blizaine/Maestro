"""Verify the mandatory CUDA 13 runtime before Pinokio publishes its marker."""
from __future__ import annotations

import re
import sys
from collections.abc import Sequence


SUPPORTED_CAPABILITIES = {(8, 9), (9, 0), (10, 0), (12, 0)}


def _version_tuple(value: object, parts: int = 2) -> tuple[int, ...]:
    numbers = [int(token) for token in re.findall(r"\d+", str(value or ""))[:parts]]
    return tuple(numbers + [0] * (parts - len(numbers)))


def validate_required_runtime(
    *,
    python_version: object,
    torch_version: object,
    cuda_version: object,
    triton_version: object,
    cuda_available: bool,
    capability: Sequence[int] | None,
) -> list[str]:
    problems: list[str] = []
    if _version_tuple(python_version) < (3, 11):
        problems.append("Python 3.11 or newer is required")
    if _version_tuple(torch_version) < (2, 10):
        problems.append("PyTorch 2.10 or newer is required")
    if _version_tuple(cuda_version) < (13, 0):
        problems.append("the PyTorch CUDA 13 build is required")
    if _version_tuple(triton_version) < (3, 6):
        problems.append("Triton 3.6 or newer is required")
    if not cuda_available:
        problems.append("PyTorch cannot access the NVIDIA GPU")
    normalized_capability = None
    if capability is not None:
        try:
            normalized_capability = tuple(int(value) for value in capability[:2])
        except (TypeError, ValueError):
            normalized_capability = None
    if normalized_capability not in SUPPORTED_CAPABILITIES:
        problems.append("the GPU does not expose an H3 Sol-compatible CUDA capability")
    return problems


def main() -> int:
    try:
        import torch
        import triton
    except Exception as exc:
        print(f"[Sol Runtime] Error: a required module could not load ({type(exc).__name__}).")
        return 1

    cuda_available = bool(torch.cuda.is_available())
    capability = None
    if cuda_available:
        try:
            capability = torch.cuda.get_device_capability(0)
        except Exception:
            capability = None

    problems = validate_required_runtime(
        python_version=sys.version.split()[0],
        torch_version=getattr(torch, "__version__", None),
        cuda_version=getattr(torch.version, "cuda", None),
        triton_version=getattr(triton, "__version__", None),
        cuda_available=cuda_available,
        capability=capability,
    )
    if problems:
        print(f"[Sol Runtime] Error: required runtime verification failed: {'; '.join(problems)}.")
        return 1

    capability_label = f"sm_{capability[0]}{capability[1]}"
    print(
        "[Sol Runtime] Required CUDA 13 runtime verified "
        f"(PyTorch {torch.__version__}, Triton {triton.__version__}, {capability_label})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
