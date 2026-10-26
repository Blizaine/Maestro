"""CPU-only checks for Fish CUDA Graph residency gates."""

from __future__ import annotations

import unittest
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

# Avoid importing the full TTS model registry in this metadata-only test.
import models

tts_package = sys.modules.get("models.TTS", getattr(models, "TTS", None))
if tts_package is None:
    tts_package = types.ModuleType("models.TTS")
    tts_package.__path__ = [str(APP / "models" / "TTS")]
sys.modules["models.TTS"] = tts_package
models.TTS = tts_package

from models.TTS.fish_audio.graph_residency import (
    _named_items,
    graph_eligibility,
    resident_capture_scope,
    weight_signature,
)


class FakeDevice:
    def __init__(self, device_type="cuda", index=0):
        self.type = device_type
        self.index = index

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and (self.type, self.index) == (other.type, other.index)

    def __repr__(self):
        suffix = "" if self.index is None else f":{self.index}"
        return f"{self.type}{suffix}"


class FakeStorage:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class FakeTensor:
    def __init__(self, device="cuda", index=0, shape=(2, 3), dtype="bfloat16", pointer=1000, version=0):
        self.device = FakeDevice(device, index)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.pointer = pointer
        self._version = version

    def data_ptr(self):
        return self.pointer

    def untyped_storage(self):
        return FakeStorage(self.pointer - 16)

    def stride(self):
        stride = 1
        values = []
        for size in reversed(self.shape):
            values.append(stride)
            stride *= size
        return tuple(reversed(values))

    def storage_offset(self):
        return 0


class FakeQuantizedTensor(FakeTensor):
    def __init__(self, *, scale_device="cuda", scale_index=0, scale_pointer=2500):
        super().__init__(shape=(8, 8), dtype="quanto", pointer=2400)
        self._data = FakeTensor(shape=(8, 8), dtype="uint8", pointer=2410)
        self._scale = FakeTensor(scale_device, scale_index, shape=(8,), dtype="float16", pointer=scale_pointer)

    def data_ptr(self):
        raise RuntimeError("quantized wrapper has no physical storage")


class FakeModule:
    def __init__(self, *, parameters=None, buffers=None, modules=None):
        self._parameters = dict(parameters or {})
        self._buffers = dict(buffers or {})
        self._modules = dict(modules or {})

    def named_modules(self, prefix="", recurse=True, remove_duplicate=True):
        yield prefix, self
        if recurse:
            for name, module in self._modules.items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_modules(child_prefix, recurse=True, remove_duplicate=remove_duplicate)

    def _named_values(self, field, prefix="", recurse=True):
        for name, value in getattr(self, field).items():
            if value is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, value
        if recurse:
            for name, module in self._modules.items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from module._named_values(field, child_prefix, recurse=True)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        yield from self._named_values("_parameters", prefix, recurse)

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        yield from self._named_values("_buffers", prefix, recurse)


class FishSwiGLUExperts(FakeModule):
    __module__ = "models.TTS.fish_audio.expert_formats"

    def __init__(self):
        names = ("w13_weight", "w2_weight", "w13_scale", "w2_scale")
        super().__init__(buffers={name: FakeTensor("cpu", None, shape=(2, 4), dtype="uint8", pointer=3000 + i)
                                  for i, name in enumerate(names)})
        self.expert_tensor_names = names
        self.num_experts = 2
        self.hidden = 8
        self.intermediate = 16
        self.rotation = 0
        self.use_triton = True
        self.format = SimpleNamespace(name="fake-fp8")
        self._fish_gpu_device = FakeDevice("cuda", 0)
        self._fish_gpu_tensors = tuple(
            FakeTensor(shape=(2, 4), dtype="uint8", pointer=4000 + i * 100)
            for i in range(len(names))
        )
        self._fish_gpu_pointers = FakeTensor(shape=(2, len(names)), dtype="int64", pointer=5000)


def make_models():
    backbone = FakeModule(
        parameters={
            "dense": FakeTensor(shape=(8, 8), pointer=1000),
            "quantized": FakeQuantizedTensor(),
        },
        modules={"experts": FishSwiGLUExperts()},
    )
    tower = FakeModule(parameters={"dense": FakeTensor(shape=(8, 8), pointer=2000)})
    return backbone, tower


def make_torch_cycle_models():
    names = ("w13_weight", "w2_weight", "w13_scale", "w2_scale")

    def init_expert_bank(self):
        torch.nn.Module.__init__(self)
        for index, name in enumerate(names):
            self.register_buffer(name, torch.empty((2, 4), dtype=torch.uint8))
        self.expert_tensor_names = names
        self.num_experts = 2
        self.hidden = 8
        self.intermediate = 16
        self.rotation = 0
        self.use_triton = True
        self.format = SimpleNamespace(name="fake-fp8")
        self._fish_gpu_device = FakeDevice("cuda", 0)
        self._fish_gpu_tensors = tuple(
            FakeTensor(shape=(2, 4), dtype="uint8", pointer=8000 + i * 100)
            for i in range(len(names))
        )
        self._fish_gpu_pointers = FakeTensor(shape=(2, len(names)), dtype="int64", pointer=9000)

    TorchFishSwiGLUExperts = type(
        "FishSwiGLUExperts",
        (torch.nn.Module,),
        {
            "__module__": "models.TTS.fish_audio.expert_formats",
            "__init__": init_expert_bank,
        },
    )

    backbone = torch.nn.Module()
    shared_parameter = torch.nn.Parameter(torch.ones(2))
    backbone.register_parameter("dense", shared_parameter)
    backbone.register_parameter("dense_alias", shared_parameter)
    backbone.register_buffer("hidden_cpu_buffer", torch.ones(1))
    backbone.add_module("experts", TorchFishSwiGLUExperts())

    shared = torch.nn.Module()
    shared.register_parameter("tied_dense", shared_parameter)
    shared.register_buffer("tied_cpu_buffer", backbone.hidden_cpu_buffer)
    backbone.add_module("layer", shared)
    backbone.add_module("layer_alias", shared)
    # MMGP attaches its model object to a hooked submodule this way. nn.Module
    # registers it as a child, producing a real child-to-root cycle.
    shared._mm_lora_model = backbone

    tower = torch.nn.Module()
    tower.register_parameter("tower_dense", torch.nn.Parameter(torch.ones(1)))
    return backbone, tower


def make_manager(root, *, blocks=None, preloaded=None, loaded=None, active=True, auto_preload=None,
                 auto_preload_requested=False):
    def ready_to_check_mem(self):
        return True

    def ensure_model_loaded(self, model_id):
        return model_id in self.active_models_ids

    Manager = type(
        "offload",
        (),
        {
            "__module__": "mmgp.offload",
            "ready_to_check_mem": ready_to_check_mem,
            "ensure_model_loaded": ensure_model_loaded,
        },
    )
    manager = Manager()
    manager.models = {"backbone": root}
    manager.active_models_ids = ["backbone"] if active else []
    manager.blocks_of_modules = blocks or {"backbone": []}
    manager.preloaded_blocks_per_model = {"backbone": preloaded or {}}
    manager.loaded_blocks = {"backbone": loaded}
    manager.auto_preload = auto_preload
    manager.auto_preload_requested = auto_preload_requested
    root._mm_manager = manager
    return manager


class FishGraphResidencyTests(unittest.TestCase):
    def test_resident_vllm_profile_one_and_three_are_eligible(self):
        backbone, tower = make_models()
        for profile in (1, 3):
            with self.subTest(profile=profile):
                eligible, reason = graph_eligibility(backbone, tower, "vllm", profile)
                self.assertTrue(eligible, reason)

    def test_engine_and_profile_are_gated(self):
        backbone, tower = make_models()
        for engine, profile in (("legacy", 1), ("vllm", 2), ("vllm", 1.0), ("vllm", True)):
            with self.subTest(engine=engine, profile=profile):
                eligible, reason = graph_eligibility(backbone, tower, engine, profile)
                self.assertFalse(eligible)
                self.assertTrue(reason)

    def test_dense_and_quant_physical_storage_must_share_the_cuda_device(self):
        backbone, tower = make_models()
        backbone._parameters["quantized"]._scale = FakeTensor("cpu", None, shape=(8,), dtype="float16")
        eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
        self.assertFalse(eligible)
        self.assertIn("not on CUDA", reason)

        backbone._parameters["quantized"]._scale = FakeTensor(shape=(8,), dtype="float16")
        tower._parameters["dense"] = FakeTensor(index=1)
        eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
        self.assertFalse(eligible)
        self.assertIn("one CUDA device", reason)

    def test_quantized_wrapper_signature_uses_physical_children_only(self):
        backbone, tower = make_models()
        wrapper = backbone._parameters["quantized"]
        before = weight_signature(backbone, tower)

        wrapper._data = FakeTensor(shape=(8, 8), dtype="uint8", pointer=7001)
        data_changed = weight_signature(backbone, tower)
        self.assertNotEqual(before, data_changed)

        wrapper._scale = FakeTensor("cpu", None, shape=(8,), dtype="float16", pointer=7002)
        eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
        self.assertFalse(eligible)
        self.assertIn("not on CUDA", reason)

    def test_inference_mode_torch_tensor_without_version_counter_can_be_signed(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable in the test environment")

        with torch.inference_mode():
            inference_tensor = torch.empty((2, 4), dtype=torch.uint8)
        self.assertTrue(torch.is_inference(inference_tensor))
        with self.assertRaises(RuntimeError):
            _ = inference_tensor._version

        backbone, tower = make_models()
        bank = backbone._modules["experts"]
        bank._fish_gpu_tensors = (inference_tensor, *bank._fish_gpu_tensors[1:])
        signature = weight_signature(backbone, tower)
        self.assertEqual(signature[0], "fish-physical-weights-v1")

    def test_each_expert_bank_needs_complete_cuda_copies_and_pointer_table(self):
        cases = (
            ("short bank", lambda bank: setattr(bank, "_fish_gpu_tensors", bank._fish_gpu_tensors[:-1])),
            ("missing pointer table", lambda bank: setattr(bank, "_fish_gpu_pointers", None)),
            ("wrong pointer shape", lambda bank: setattr(bank, "_fish_gpu_pointers", FakeTensor(shape=(1, 4), dtype="int64"))),
            ("wrong pointer dtype", lambda bank: setattr(bank, "_fish_gpu_pointers", FakeTensor(shape=(2, 4), dtype="int32"))),
            ("CPU copy", lambda bank: setattr(bank, "_fish_gpu_tensors", (FakeTensor("cpu", None), *bank._fish_gpu_tensors[1:]))),
            ("Triton disabled", lambda bank: setattr(bank, "use_triton", False)),
        )
        for name, change in cases:
            with self.subTest(name=name):
                backbone, tower = make_models()
                change(backbone._modules["experts"])
                eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
                self.assertFalse(eligible)
                self.assertTrue(reason)

    def test_unknown_mmgp_and_dynamic_residency_are_rejected(self):
        backbone, tower = make_models()
        make_manager(backbone, blocks={"backbone/layer.0": []})
        eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
        self.assertFalse(eligible)
        self.assertIn("may load", reason)

        backbone, tower = make_models()
        make_manager(backbone, auto_preload=object())
        eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
        self.assertFalse(eligible)
        self.assertIn("auto-preload", reason)
        with self.assertRaisesRegex(RuntimeError, "auto-preload"):
            with resident_capture_scope(backbone, tower):
                self.fail("unstable managers must not enter capture scope")

        backbone, tower = make_models()
        backbone._mm_manager = SimpleNamespace()
        eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
        self.assertFalse(eligible)
        self.assertIn("unknown MMGP manager", reason)

    def test_single_current_block_is_stable_when_other_blocks_are_preloaded(self):
        backbone, tower = make_models()
        make_manager(
            backbone,
            blocks={"backbone/layer.0": [], "backbone/layer.1": []},
            preloaded={"layer.0": 100},
            loaded="layer.1",
        )
        eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
        self.assertTrue(eligible, reason)

    def test_cpu_checkpoint_expert_banks_are_excluded_from_signature(self):
        backbone, tower = make_models()
        bank = backbone._modules["experts"]
        before = weight_signature(backbone, tower)
        bank._buffers["w13_weight"].pointer += 50000
        after = weight_signature(backbone, tower)
        self.assertEqual(before, after)

        bank._buffers["unrecognized_cpu_state"] = FakeTensor("cpu", None, dtype="int64")
        eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
        self.assertFalse(eligible)
        self.assertIn("not on CUDA", reason)

    def test_torch_module_cycles_and_tied_aliases_are_enumerated_once(self):
        backbone, tower = make_torch_cycle_models()

        modules = _named_items(backbone, "named_modules")
        self.assertEqual(len({id(module) for _, module in modules}), len(modules))
        module_names = {name for name, _ in modules}
        self.assertIn("layer", module_names)
        self.assertNotIn("layer_alias", module_names)

        parameters = _named_items(backbone, "named_parameters")
        parameter_names = [name for name, _ in parameters]
        self.assertEqual(len({id(parameter) for _, parameter in parameters}), len(parameters))
        self.assertIn("dense", parameter_names)
        self.assertNotIn("dense_alias", parameter_names)
        self.assertNotIn("layer.tied_dense", parameter_names)

        buffers = _named_items(backbone, "named_buffers")
        buffer_names = [name for name, _ in buffers]
        self.assertEqual(len({id(buffer) for _, buffer in buffers}), len(buffers))
        self.assertIn("hidden_cpu_buffer", buffer_names)
        self.assertIn("experts.w13_weight", buffer_names)
        self.assertNotIn("layer.tied_cpu_buffer", buffer_names)

        signature = weight_signature(backbone, tower)
        dense_entries = signature[1]
        self.assertIn("dense", [entry[2] for entry in dense_entries])
        self.assertIn("hidden_cpu_buffer", [entry[2] for entry in dense_entries])
        self.assertNotIn("experts.w13_weight", [entry[2] for entry in dense_entries])

        eligible, reason = graph_eligibility(backbone, tower, "vllm", 1)
        self.assertFalse(eligible)
        self.assertIn("not on CUDA", reason)
        self.assertNotIn("recursion", reason.lower())

    def test_quant_storage_and_gpu_expert_copy_replacements_change_signature(self):
        backbone, tower = make_models()
        before = weight_signature(backbone, tower)
        backbone._parameters["quantized"]._scale = FakeTensor(shape=(8,), dtype="float16", pointer=9090)
        quant_changed = weight_signature(backbone, tower)
        self.assertNotEqual(before, quant_changed)

        backbone, tower = make_models()
        before = weight_signature(backbone, tower)
        bank = backbone._modules["experts"]
        bank._fish_gpu_tensors = (FakeTensor(shape=(2, 4), dtype="uint8", pointer=12345), *bank._fish_gpu_tensors[1:])
        copy_changed = weight_signature(backbone, tower)
        self.assertNotEqual(before, copy_changed)

        bank._fish_gpu_pointers = FakeTensor(shape=(2, 4), dtype="int64", pointer=54321)
        pointer_changed = weight_signature(backbone, tower)
        self.assertNotEqual(copy_changed, pointer_changed)

    def test_capture_scope_restores_class_method_after_normal_and_exceptional_exit(self):
        backbone, tower = make_models()
        manager = make_manager(backbone)
        self.assertNotIn("ready_to_check_mem", manager.__dict__)
        with resident_capture_scope(backbone, tower):
            self.assertFalse(manager.ready_to_check_mem())
        self.assertNotIn("ready_to_check_mem", manager.__dict__)
        self.assertTrue(manager.ready_to_check_mem())

        prior = lambda: "prior instance value"
        manager.ready_to_check_mem = prior
        with self.assertRaisesRegex(ValueError, "capture failed"):
            with resident_capture_scope(backbone, tower):
                self.assertFalse(manager.ready_to_check_mem())
                raise ValueError("capture failed")
        self.assertIs(manager.ready_to_check_mem, prior)
        self.assertEqual(manager.ready_to_check_mem(), "prior instance value")

    def test_capture_scope_rejects_unknown_manager_before_yield(self):
        backbone, tower = make_models()
        backbone._mm_manager = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "unknown MMGP manager"):
            with resident_capture_scope(backbone, tower):
                self.fail("unknown managers must not enter capture scope")


if __name__ == "__main__":
    unittest.main()
