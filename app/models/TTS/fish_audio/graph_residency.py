"""Fail-closed residency checks for whole-frame Fish CUDA Graph capture."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping


_MISSING = object()
_ALLOWED_PROFILES = (1, 3)
_FISH_EXPERT_MODULE_SUFFIX = ".fish_audio.expert_formats"
_MMGP_MODULE = "mmgp.offload"
_QUANT_STORAGE_ATTRS = ("_data", "_scale", "_scale_shift", "_shift")


class _ResidencyError(RuntimeError):
    pass


def _device_info(value):
    """Return a canonical (device type, index) pair without querying CUDA."""

    device = getattr(value, "device", _MISSING)
    if device is _MISSING:
        raise _ResidencyError("tensor has no device metadata")
    if isinstance(device, str):
        parts = device.split(":", 1)
        device_type = parts[0]
        index = None if len(parts) == 1 else parts[1]
        if index is not None:
            try:
                index = int(index)
            except (TypeError, ValueError):
                raise _ResidencyError(f"unrecognized tensor device {device!r}") from None
    else:
        device_type = getattr(device, "type", _MISSING)
        index = getattr(device, "index", _MISSING)
    if device_type is _MISSING or index is _MISSING:
        raise _ResidencyError("tensor device metadata is incomplete")
    if device_type == "cpu":
        return "cpu", index
    if device_type != "cuda" or not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise _ResidencyError(f"tensor is not on a supported device ({device!r})")
    return "cuda", index


def _is_tensor_like(value):
    """Recognize real tensors and metadata-only tensor fakes used by CPU tests."""

    return (
        value is not None
        and getattr(value, "device", _MISSING) is not _MISSING
        and getattr(value, "shape", _MISSING) is not _MISSING
        and callable(getattr(value, "data_ptr", None))
    )


def _physical_tensors(value, path, seen=None):
    """Walk a tensor and MMGP/Quanto physical storage such as _data and _scale."""

    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)

    child_entries = []
    for attr in _QUANT_STORAGE_ATTRS:
        try:
            child = getattr(value, attr, _MISSING)
        except Exception as error:
            raise _ResidencyError(f"could not inspect {path}.{attr}: {error}") from error
        if child is not _MISSING and child is not None and child is not value:
            child_entries.extend(_physical_tensors(child, f"{path}.{attr}", seen))

    # Quantized wrappers can expose a logical tensor interface without owning
    # storage. Treat their physical leaves as authoritative when present.
    if child_entries:
        yield from child_entries
    elif _is_tensor_like(value):
        yield path, value


def _named_items(module, method_name):
    method = getattr(module, method_name, None)
    if not callable(method):
        raise _ResidencyError(f"module does not expose {method_name}()")
    try:
        # MMGP's LoRA hooks store a registered child -> model reference even
        # when no adapters are active. These module graphs contain cycles.
        # Inspect each physical module/tensor once, just as PyTorch's default
        # traversal does; expanding every alias would recurse indefinitely.
        items = method(remove_duplicate=True)
    except TypeError:
        items = method()
    try:
        result = list(items)
    except Exception as error:
        raise _ResidencyError(f"could not enumerate {method_name}(): {error}") from error
    for item in result:
        if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[0], str):
            raise _ResidencyError(f"{method_name}() returned an unknown entry")
    return result


def _named_modules(root):
    modules = _named_items(root, "named_modules")
    if not modules or modules[0][1] is not root:
        raise _ResidencyError("named_modules() did not include the model root")
    return modules


def _root_trees(backbone, tower):
    if backbone is None or tower is None:
        raise _ResidencyError("backbone and tower are required")
    return [
        ("backbone", backbone, _named_modules(backbone)),
        ("tower", tower, _named_modules(tower)),
    ]


def _expert_kind(module):
    cls = type(module)
    is_fish = getattr(cls, "__name__", "") == "FishSwiGLUExperts"
    has_marker = any(
        hasattr(module, name)
        for name in ("_fish_gpu_tensors", "_fish_gpu_pointers", "expert_tensor_names")
    )
    if not is_fish and not has_marker:
        return False
    module_name = getattr(cls, "__module__", "")
    if not is_fish or not module_name.endswith(_FISH_EXPERT_MODULE_SUFFIX):
        raise _ResidencyError("found an unknown Fish expert-bank implementation")
    return True


def _expert_banks(trees):
    banks = []
    for root_name, _, modules in trees:
        for module_name, module in modules:
            if _expert_kind(module):
                banks.append((root_name, module_name, module))
    if not banks:
        raise _ResidencyError("no Fish expert banks were found")
    return banks


def _under_prefix(name, prefix):
    return not prefix or name == prefix or name.startswith(prefix + ".")


def _is_expert_checkpoint_entry(name, module_name, bank):
    if not _under_prefix(name, module_name):
        return False
    names = getattr(bank, "expert_tensor_names", None)
    if not isinstance(names, (tuple, list)) or not all(isinstance(item, str) for item in names):
        return False
    relative_name = name[len(module_name) + 1:] if module_name else name
    return relative_name in names


def _all_physical_entries(trees, banks):
    entries = []
    for root_name, root, _ in trees:
        root_banks = [(name, module) for bank_root, name, module in banks if bank_root == root_name]
        for kind, method_name in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
            for name, value in _named_items(root, method_name):
                tensors = list(_physical_tensors(value, name or "<root>"))
                if not tensors:
                    raise _ResidencyError(f"could not identify physical tensor storage for {root_name}.{name}")
                for storage_name, tensor in tensors:
                    device = _device_info(tensor)
                    expert_cpu = any(
                        device[0] == "cpu" and _is_expert_checkpoint_entry(name, bank_name, bank)
                        for bank_name, bank in root_banks
                    )
                    entries.append((root_name, kind, name, storage_name, tensor, device, expert_cpu))
    return entries


def _check_expert_bank(root_name, module_name, bank):
    names = getattr(bank, "expert_tensor_names", None)
    copies = getattr(bank, "_fish_gpu_tensors", None)
    pointers = getattr(bank, "_fish_gpu_pointers", None)
    expert_device = getattr(bank, "_fish_gpu_device", None)
    count = getattr(bank, "num_experts", None)

    if not isinstance(names, (tuple, list)) or not names or not all(isinstance(name, str) for name in names):
        raise _ResidencyError(f"{root_name}.{module_name}: expert tensor names are unavailable")
    if not isinstance(copies, (tuple, list)) or len(copies) != len(names) or any(copy is None for copy in copies):
        raise _ResidencyError(f"{root_name}.{module_name}: GPU expert bank is incomplete")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise _ResidencyError(f"{root_name}.{module_name}: expert count is unknown")
    if getattr(bank, "use_triton", None) is not True:
        raise _ResidencyError(f"{root_name}.{module_name}: GPU expert execution is not enabled")

    expected_device = _device_info(type("DeviceHolder", (), {"device": expert_device})())
    if expected_device[0] != "cuda":
        raise _ResidencyError(f"{root_name}.{module_name}: GPU expert device is not CUDA")
    for index, copy in enumerate(copies):
        if not _is_tensor_like(copy):
            raise _ResidencyError(f"{root_name}.{module_name}: GPU expert tensor {index} is unknown")
        if _device_info(copy) != expected_device:
            raise _ResidencyError(f"{root_name}.{module_name}: GPU expert tensors span multiple devices")

    if not _is_tensor_like(pointers):
        raise _ResidencyError(f"{root_name}.{module_name}: GPU expert pointer table is missing")
    if _device_info(pointers) != expected_device:
        raise _ResidencyError(f"{root_name}.{module_name}: GPU expert pointer table is on another device")
    shape = tuple(getattr(pointers, "shape", ()))
    if shape != (count, len(names)):
        raise _ResidencyError(f"{root_name}.{module_name}: GPU expert pointer table has the wrong shape")
    dtype = str(getattr(pointers, "dtype", "")).split(".")[-1]
    if dtype != "int64":
        raise _ResidencyError(f"{root_name}.{module_name}: GPU expert pointer table is not int64")
    return expected_device


def _ids(value):
    if value is _MISSING or value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {item for item in value if item is not None}
    return {value}


def _verify_manager_class(manager):
    cls = type(manager)
    if getattr(cls, "__module__", None) != _MMGP_MODULE or getattr(cls, "__name__", None) != "offload":
        raise _ResidencyError("an unknown MMGP manager is attached")
    required = (
        "models",
        "active_models_ids",
        "blocks_of_modules",
        "preloaded_blocks_per_model",
        "loaded_blocks",
        "auto_preload",
        "auto_preload_requested",
        "ready_to_check_mem",
        "ensure_model_loaded",
    )
    for name in required:
        if not hasattr(manager, name):
            raise _ResidencyError(f"MMGP manager is missing {name}")
    if not isinstance(manager.models, Mapping) or not isinstance(manager.blocks_of_modules, Mapping):
        raise _ResidencyError("MMGP manager has unknown model/block state")
    if not isinstance(manager.preloaded_blocks_per_model, Mapping) or not isinstance(manager.loaded_blocks, Mapping):
        raise _ResidencyError("MMGP manager has unknown residency state")
    if not isinstance(manager.active_models_ids, (list, tuple, set, frozenset)):
        raise _ResidencyError("MMGP manager active-model state is unknown")
    if not callable(manager.ready_to_check_mem) or not callable(manager.ensure_model_loaded):
        raise _ResidencyError("MMGP manager hooks are unknown")
    if manager.auto_preload is not None or manager.auto_preload_requested:
        raise _ResidencyError("MMGP auto-preload is active")


def _manager_states(trees):
    attachments = {}
    for root_name, root, modules in trees:
        for module_name, module in modules:
            try:
                manager = getattr(module, "_mm_manager", None)
            except Exception as error:
                raise _ResidencyError(f"could not inspect MMGP state on {root_name}.{module_name}: {error}") from error
            if manager is None:
                continue
            key = id(manager)
            state = attachments.setdefault(key, {"manager": manager, "roots": {}, "annotated_ids": set()})
            state["roots"][root_name] = root
            for attr in ("_mm_model_id", "_mm_id"):
                try:
                    state["annotated_ids"].update(_ids(getattr(module, attr, _MISSING)))
                except Exception as error:
                    raise _ResidencyError(f"could not inspect {root_name}.{module_name}.{attr}: {error}") from error

    states = list(attachments.values())
    for state in states:
        manager = state["manager"]
        _verify_manager_class(manager)
        model_roots = state["roots"]
        roots_by_id = {
            model_id: root_name
            for model_id, model in manager.models.items()
            for root_name, root in model_roots.items()
            if model is root
        }
        model_ids = set(roots_by_id)
        if state["annotated_ids"]:
            for model_id in state["annotated_ids"]:
                if model_id not in manager.models or model_id not in roots_by_id:
                    raise _ResidencyError("MMGP manager references a model outside the captured roots")
            model_ids.update(state["annotated_ids"])
        if not model_ids:
            raise _ResidencyError("MMGP manager is not tied to either captured model root")

        for model_id in model_ids:
            if model_id not in manager.active_models_ids:
                raise _ResidencyError(f"MMGP model {model_id!r} is not active")
            if model_id not in manager.preloaded_blocks_per_model or model_id not in manager.loaded_blocks:
                raise _ResidencyError(f"MMGP model {model_id!r} has incomplete block residency state")
            preloaded = manager.preloaded_blocks_per_model[model_id]
            if not isinstance(preloaded, Mapping):
                raise _ResidencyError(f"MMGP model {model_id!r} has unknown preloaded-block state")

            prefix = f"{model_id}/"
            registered_entries = [
                entry for entry in manager.blocks_of_modules
                if entry == model_id or (isinstance(entry, str) and entry.startswith(prefix))
            ]
            block_names = {
                entry[len(prefix):]
                for entry in manager.blocks_of_modules
                if isinstance(entry, str) and entry.startswith(prefix) and entry[len(prefix):]
            }
            if not registered_entries:
                raise _ResidencyError(f"MMGP model {model_id!r} has no registered block inventory")
            for entry in manager.blocks_of_modules:
                if entry == model_id or (isinstance(entry, str) and entry.startswith(prefix)):
                    records = manager.blocks_of_modules[entry]
                    if not isinstance(records, (list, tuple)):
                        raise _ResidencyError(f"MMGP block inventory for {model_id!r} is unknown")

            unpreloaded = block_names.difference(preloaded)
            loaded = manager.loaded_blocks[model_id]
            if unpreloaded and (loaded is None or unpreloaded != {loaded}):
                raise _ResidencyError(f"MMGP model {model_id!r} has blocks that may load during capture")
            state.setdefault("model_ids", set()).add(model_id)
    return states


def graph_eligibility(backbone, tower, engine, profile):
    """Return whether the entire Fish decoder is resident on one stable CUDA device."""

    if engine != "vllm":
        return False, "whole-frame CUDA Graphs require the vllm engine"
    if isinstance(profile, bool) or not isinstance(profile, int) or profile not in _ALLOWED_PROFILES:
        return False, "whole-frame CUDA Graphs require profile 1 or 3"
    try:
        trees = _root_trees(backbone, tower)
        banks = _expert_banks(trees)
        devices = set()
        for root_name, module_name, bank in banks:
            devices.add(_check_expert_bank(root_name, module_name, bank))
        for _, _, _, _, _, device, expert_cpu in _all_physical_entries(trees, banks):
            if not expert_cpu:
                if device[0] != "cuda":
                    raise _ResidencyError("a dense parameter or buffer is not on CUDA")
                devices.add(device)
        if len(devices) != 1:
            raise _ResidencyError("dense weights and GPU expert copies do not share one CUDA device")
        _manager_states(trees)
    except Exception as error:
        detail = str(error) or type(error).__name__
        return False, f"Fish CUDA Graph residency is not verified: {detail}"
    return True, "Fish dense weights and complete GPU expert banks are resident on one CUDA device"


def _storage_signature(tensor):
    if not _is_tensor_like(tensor):
        raise _ResidencyError("physical storage is not tensor-like")
    try:
        data_ptr = int(tensor.data_ptr())
    except Exception as error:
        raise _ResidencyError(f"could not read tensor data pointer: {error}") from error
    storage_ptr = data_ptr
    for accessor in ("untyped_storage", "storage"):
        method = getattr(tensor, accessor, None)
        if not callable(method):
            continue
        try:
            storage = method()
            pointer = getattr(storage, "data_ptr", None)
            if callable(pointer):
                storage_ptr = int(pointer())
                break
        except Exception:
            continue
    shape = tuple(getattr(tensor, "shape", ()))
    stride_method = getattr(tensor, "stride", None)
    try:
        stride = tuple(stride_method()) if callable(stride_method) else None
    except Exception:
        stride = None
    offset_method = getattr(tensor, "storage_offset", None)
    try:
        storage_offset = int(offset_method()) if callable(offset_method) else None
    except Exception:
        storage_offset = None
    try:
        version = getattr(tensor, "_version", None)
    except Exception:
        # Inference-mode tensors intentionally have no version counter.
        version = None
    return (
        _device_info(tensor),
        str(getattr(tensor, "dtype", "unknown")),
        shape,
        stride,
        storage_offset,
        storage_ptr,
        data_ptr,
        version,
    )


def weight_signature(backbone, tower):
    """Return comparable physical-storage identities for dense and GPU expert weights."""

    trees = _root_trees(backbone, tower)
    banks = _expert_banks(trees)
    dense = []
    for root_name, kind, name, storage_name, tensor, _, expert_cpu in _all_physical_entries(trees, banks):
        if not expert_cpu:
            dense.append((root_name, kind, name, storage_name, _storage_signature(tensor)))

    expert = []
    for root_name, module_name, bank in banks:
        names = tuple(getattr(bank, "expert_tensor_names", ()) or ())
        copies = getattr(bank, "_fish_gpu_tensors", None)
        copy_signatures = ()
        if isinstance(copies, (tuple, list)):
            copy_signatures = tuple(
                None if copy is None else _storage_signature(copy)
                for copy in copies
            )
        pointers = getattr(bank, "_fish_gpu_pointers", None)
        pointer_signature = None if pointers is None else _storage_signature(pointers)
        expert.append((
            root_name,
            module_name,
            type(bank).__module__,
            type(bank).__qualname__,
            getattr(getattr(bank, "format", None), "name", None),
            names,
            getattr(bank, "num_experts", None),
            getattr(bank, "hidden", None),
            getattr(bank, "intermediate", None),
            getattr(bank, "rotation", None),
            _device_info(type("DeviceHolder", (), {"device": getattr(bank, "_fish_gpu_device", None)})())
            if getattr(bank, "_fish_gpu_device", None) is not None else None,
            copy_signatures,
            pointer_signature,
        ))
    return "fish-physical-weights-v1", tuple(dense), tuple(expert)


@contextmanager
def resident_capture_scope(backbone, tower):
    """Suppress stable MMGP memory checks for warmup/capture, restoring exact state."""

    trees = _root_trees(backbone, tower)
    states = _manager_states(trees)
    changed = []
    try:
        for state in states:
            manager = state["manager"]
            instance_dict = getattr(manager, "__dict__", None)
            if not isinstance(instance_dict, dict):
                raise _ResidencyError("MMGP manager instance state cannot be restored exactly")
            prior = instance_dict.get("ready_to_check_mem", _MISSING)
            changed.append((manager, prior))
            manager.ready_to_check_mem = lambda: False
        yield
    finally:
        for manager, prior in reversed(changed):
            if prior is _MISSING:
                try:
                    delattr(manager, "ready_to_check_mem")
                except AttributeError:
                    pass
            else:
                manager.ready_to_check_mem = prior
