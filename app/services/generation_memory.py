"""Release failed-job resources after model frames have unwound."""
from functools import wraps
import gc
import sys
import traceback


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
