"""Optional SageGP dispatch for the dense SM120 attention paths."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from subprocess import CalledProcessError
import sys

import torch


PRECISE = "sagegp_precise"
FAST = "sagegp_fast"
ADAPTIVE = "sagegp_adaptive"
MODES = (PRECISE, FAST, ADAPTIVE)
FIRST_STEP_PRECISE_DEFAULT = True
OVERLAY_NAME = "maestro-sagegp"

# None outside a sampling job. Track the current step separately for every worker
# so preview requests or an interrupted job cannot affect the next generation.
_sampling = ContextVar("sagegp_sampling", default=None)
_failed_kernels = set()


def activate_overlay():
    """Prefer the optional wheel without overwriting an existing Sage install."""
    overlay = Path(sys.prefix) / OVERLAY_NAME
    if (overlay / "sagegp" / "sm120" / "dense.py").is_file() and str(overlay) not in sys.path:
        sys.path.insert(0, str(overlay))


def sageattention_version():
    """SageGP ships the SageAttention 2 namespace under a different wheel name."""
    try:
        version("sagegp")
    except PackageNotFoundError:
        return version("sageattention")
    return "2.2.0"


@contextmanager
def sampling_scope():
    token = _sampling.set((FIRST_STEP_PRECISE_DEFAULT, 0))
    try:
        yield
    finally:
        _sampling.reset(token)


def configure_sampling(first_step_precise):
    if _sampling.get() is not None:
        _sampling.set((bool(first_step_precise), 0))


def set_step(step):
    state = _sampling.get()
    if state is not None:
        _sampling.set((state[0], step))


def effective_mode(mode):
    if mode in (FAST, ADAPTIVE) and _sampling.get() == (True, 0):
        return PRECISE
    return mode


def _has_sm120(device):
    if torch.version.hip or not torch.cuda.is_available():
        return False
    try:
        return tuple(torch.cuda.get_device_capability(device)) == (12, 0)
    except (AssertionError, RuntimeError, ValueError):
        return False


@lru_cache(maxsize=1)
def _load_cores():
    try:
        import triton
        from sagegp.sm120 import dense
    except (ImportError, OSError, RuntimeError):
        return None

    fast = routed = wide = None
    try:
        triton_version = tuple(int(part) for part in triton.__version__.split(".")[:2])
        if triton_version >= (3, 6) and hasattr(triton.language, "dot_scaled"):
            from sagegp.sm120 import fast, routed, wide
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    return dense, fast, routed, wide


def available_modes(device=None):
    if not _has_sm120(device):
        return ()
    cores = _load_cores()
    if cores is None:
        return ()
    _, fast, routed, _ = cores
    return (PRECISE,) + ((FAST,) if fast is not None else ()) + ((ADAPTIVE,) if routed is not None else ())


def _run_kernel(kernel, q, k, v, **settings):
    from triton.compiler.errors import CompilationError

    key = (q.device, kernel.__name__)
    if key in _failed_kernels:
        return None
    try:
        # Keep the original Q usable if an optional kernel fails after
        # preparing or quantizing it.
        return kernel.attention_owned([q, k, v], recycle_q=False, **settings)
    except torch.OutOfMemoryError:
        raise
    except (CompilationError, ImportError, OSError, RuntimeError, CalledProcessError) as error:
        _failed_kernels.add(key)
        print(f"[SageGP] {kernel.__name__} unavailable on {q.device}; using standard attention: {error}")
        return None


def attend_dense(mode, q, k, v):
    """Return None for unsupported inputs, leaving them intact for fallback."""
    if q.device.type != "cuda" or not _has_sm120(q.device):
        return None
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        return None
    if q.device != k.device or k.device != v.device or q.dtype != k.dtype or k.dtype != v.dtype:
        return None
    if not all(tensor.stride(-1) == 1 for tensor in (q, k, v)):
        return None
    cores = _load_cores()
    if cores is None:
        return None
    dense, fast, routed, wide = cores
    mode = effective_mode(mode)
    batch, queries, heads, width = q.shape
    keys, kv_heads = k.shape[1:3]
    geometry = (batch, queries, keys, heads, kv_heads, width)
    if mode == FAST and fast is not None and fast.eligible(q.device, q.dtype, *geometry):
        return _run_kernel(fast, q, k, v)
    if mode in (FAST, ADAPTIVE):
        if routed is not None and routed.eligible(q.device, q.dtype, *geometry):
            return _run_kernel(routed, q, k, v)
        if wide is not None and q.shape == k.shape and wide.preferred(q.device, q.dtype, *geometry):
            return _run_kernel(wide, q, k, v)
    if queries > 8192 and dense.eligible(*geometry, q.dtype):
        return _run_kernel(dense, q, k, v, int8_scores=True)
    return None
