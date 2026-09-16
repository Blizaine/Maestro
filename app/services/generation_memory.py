"""Diagnose memory failures and release resources after model frames unwind."""
from functools import wraps
import gc
import sys
import traceback


def classify_memory_error(error):
    """Only identify RAM/VRAM when the allocator message identifies it.

    A generic CUDA driver OOM can surface asynchronously, and does not tell
    us whether device memory, pinned host memory, or another allocation failed.
    """
    message = str(error).lower()
    if "cuda error: out of memory" in message:
        return "CUDA"
    if "cuda out of memory" in message:
        return "VRAM"
    if isinstance(error, MemoryError) or (
        "defaultcpuallocator" in message and "allocate" in message
    ):
        return "RAM"
    return ""


def log_generation_memory(cuda):
    """Best-effort readings before teardown; never initialize CUDA or mask OOM."""
    parts = []
    gib = 1024 ** 3
    try:
        import psutil

        ram = psutil.virtual_memory()
        process = psutil.Process().memory_info()
        parts.extend([
            f"system RAM available {ram.available / gib:.2f}/{ram.total / gib:.2f} GiB",
            f"process RSS {process.rss / gib:.2f} GiB",
        ])
        # psutil exposes Windows private committed bytes separately from RSS.
        if hasattr(process, "private"):
            parts.append(f"process private commit {process.private / gib:.2f} GiB")
    except Exception as error:
        parts.append(f"RAM readings unavailable ({type(error).__name__})")
    try:
        if cuda.is_initialized():
            parts.extend([
                f"PyTorch VRAM allocated {cuda.memory_allocated() / gib:.2f} GiB",
                f"reserved {cuda.memory_reserved() / gib:.2f} GiB",
            ])
            free, total = cuda.mem_get_info()
            parts.append(f"CUDA free {free / gib:.2f}/{total / gib:.2f} GiB")
    except Exception as error:
        parts.append(f"CUDA readings incomplete ({type(error).__name__})")
    print("[Memory at failure, before cleanup] " + "; ".join(parts))


def cleanup_failed_generation(cleanup):
    """Keep successful model caching, but tear down after a failed call.

    GPU activations held by a traceback are live allocations: empty_cache()
    inside the model's except block cannot release them.
    """
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            failed = False
            try:
                result = function(*args, **kwargs)
                failed = result is False
                return result
            except BaseException as error:
                failed = True
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                if failed:
                    try:
                        cleanup()
                    except Exception as cleanup_error:
                        print(f"[Memory] Failed-generation cleanup: {cleanup_error}")
        return wrapped
    return decorate


def release_auxiliary_models():
    """Release loaded post-processors without importing unused model stacks."""
    from shared.utils import offload_registry

    released = []
    for name in offload_registry.registered_names():
        try:
            released.extend(offload_registry.release_all([name]))
        except Exception as error:
            print(f"[Memory] Could not release {name}: {error}")
    # FlashVSR also needs teardown after a load failure before offload.profile
    # exists. Older in-process instances were not registered at all.
    module = sys.modules.get("postprocessing.flashvsr.runtime")
    runtime = getattr(module, "_RUNTIME", None)
    if runtime is not None and any(getattr(runtime, field, None) is not None for field in ("dit", "lq_proj", "tcdecoder", "vae", "offloadobj")):
        try:
            module.release_models()
            released.append("FlashVSR")
        except Exception as error:
            print(f"[Memory] Could not release FlashVSR: {error}")
    gc.collect()
    return released
