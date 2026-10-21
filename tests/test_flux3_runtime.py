"""CPU-only runtime contracts for native FLUX 3 Klein components."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

# These unit tests assert the runtime's MMGP loader contract without importing
# MMGP's optional GPU quantization stack or loading model weights.
mmgp_stub = ModuleType("mmgp")
mmgp_stub.offload = SimpleNamespace(load_model_data=Mock())

from models.flux3.autoencoder import AutoEncoder, latent_ids  # noqa: E402
from models.flux3.dit import Packed  # noqa: E402
from models.flux3.sampling import DEFAULT_ALPHA, DEFAULT_STEPS, denoise_flow_map, timesteps  # noqa: E402
from models.flux3.text_encoder import FEATURE_WIDTH, OUTPUT_LAYERS, encode_prompt  # noqa: E402

_runtime_name = "models.flux3._runtime_contract_test"
_runtime_spec = importlib.util.spec_from_file_location(
    _runtime_name, ROOT / "app/models/flux3/runtime.py"
)
runtime_module = importlib.util.module_from_spec(_runtime_spec)
sys.modules[_runtime_name] = runtime_module
_missing = object()
_original_mmgp = sys.modules.get("mmgp", _missing)
sys.modules["mmgp"] = mmgp_stub
try:
    # Restore only the MMGP entry; replacing all of sys.modules would evict
    # legitimate torch imports and can trigger duplicate TORCH_LIBRARY errors.
    _runtime_spec.loader.exec_module(runtime_module)
finally:
    sys.modules.pop(_runtime_name, None)
    if _original_mmgp is _missing:
        sys.modules.pop("mmgp", None)
    else:
        sys.modules["mmgp"] = _original_mmgp


class _ZeroDiT(torch.nn.Module):
    def forward(self, packed, timesteps_ref=None):
        return torch.zeros_like(packed.data["image"])


class _ShapeVAE(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.decoder = torch.nn.Identity()
        self.decoded_ids = []
        self.nonfinite = nonfinite

    def decode(self, tokens, ids):
        self.decoded_ids.append(ids.clone())
        height = int(ids[:, 1].max()) + 1
        width = int(ids[:, 2].max()) + 1
        fill = float("nan") if self.nonfinite else 0.0
        return torch.full((3, height * 16, width * 16), fill, device=tokens.device)


class _ConstantDiT(torch.nn.Module):
    def __init__(self, value=2.0):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, packed, timesteps_ref=None):
        self.calls += 1
        return torch.full_like(packed.data["image"], self.value)


class _FakeTokenizer:
    padding_side = "right"

    def __init__(self, real_length=81):
        self.real_length = real_length
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        length = kwargs["max_length"] if kwargs.get("padding") == "max_length" else self.real_length
        input_ids = torch.ones((1, length), dtype=torch.long)
        attention_mask = torch.zeros((1, length), dtype=torch.long)
        attention_mask[:, :min(length, self.real_length)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeTextProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, *args, **kwargs):
        return "encoded chat template"


class _DistinctiveHiddenStateModel:
    def __init__(self):
        self.call_kwargs = None
        self.lm_head = Mock(side_effect=AssertionError("the Qwen vocabulary head must not run"))

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        length = kwargs["input_ids"].shape[1]
        hidden_states = tuple(
            torch.full((1, length, 2560), layer, dtype=torch.bfloat16)
            for layer in range(33)
        )
        return SimpleNamespace(hidden_states=hidden_states)


def _runtime_for_cpu_test():
    runtime = object.__new__(runtime_module.model_factory)
    runtime._abort = False
    runtime.device = torch.device("cpu")
    runtime.transformer = _ZeroDiT()
    runtime.vae = _ShapeVAE()
    runtime.text_encoder = SimpleNamespace(
        visual=SimpleNamespace(blocks=[]),
        language_model=SimpleNamespace(layers=[]),
    )
    runtime.processor = None
    return runtime


class Flux3RuntimeTests(unittest.TestCase):
    def test_quantized_checkpoint_validation_preserves_native_vision_qkv(self):
        module = torch.nn.Module()
        module.visual = torch.nn.Module()
        module.visual.qkv = torch.nn.Linear(4, 12, bias=False)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "encoder.safetensors")
            save_file({"visual.qkv.weight": torch.zeros((12, 4), dtype=torch.int8)}, filename)
            runtime_module._validate_public_checkpoint(module, [filename], "Qwen vision")

    def test_quantized_checkpoint_validation_splits_only_separate_destination_qkv(self):
        module = torch.nn.Module()
        module.attn = torch.nn.Module()
        for name in ("q", "k", "v"):
            setattr(module.attn, name + "_proj", torch.nn.Linear(4, 2, bias=False))
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "dit.safetensors")
            save_file({"attn.qkv.weight": torch.zeros((6, 4), dtype=torch.int8)}, filename)
            runtime_module._validate_public_checkpoint(module, [filename], "DiT")

    def test_public_transformers5_rope_schema_preserves_released_values_on_4x(self):
        parameters = {"rope_type": "default", "rope_theta": 5000000, "mrope_interleaved": True, "mrope_section": [24, 20, 20]}
        config = SimpleNamespace(text_config=SimpleNamespace(rope_scaling=None, rope_parameters=parameters, rope_theta=10000))
        runtime_module._normalize_qwen_rope_config(config)
        self.assertEqual(config.text_config.rope_theta, 5000000)
        self.assertEqual(config.text_config.rope_scaling["mrope_section"], [24, 20, 20])
        self.assertNotIn("rope_theta", config.text_config.rope_scaling)
        original = dict(config.text_config.rope_scaling)
        runtime_module._normalize_qwen_rope_config(config)
        self.assertEqual(config.text_config.rope_scaling, original)

    def test_base_sampler_uses_cfg_without_a_flow_map_target(self):
        packed = Packed(data={"image": torch.zeros((1, 2))}, ids={}, timesteps={"image": torch.zeros(1)}, mode_indices={})
        negative = Packed(data={"image": packed.data["image"]}, ids={}, timesteps={"image": torch.zeros(1)}, mode_indices={})
        class BaseDiT:
            flow_map = False
            def __call__(self, value, timesteps_ref=None):
                self_test.assertIsNone(timesteps_ref)
                return torch.full_like(value.data["image"], 1 if value is negative else 3)
        self_test = self
        result = denoise_flow_map(BaseDiT(), packed, torch.tensor([1.0, 0.0]), negative_packed=negative, guidance_scale=3.5)
        torch.testing.assert_close(result, torch.full((1, 2), -8.0))

    def test_mmgp_test_shim_does_not_leak_into_import_state(self):
        self.assertIsNot(sys.modules.get("mmgp"), mmgp_stub)
        self.assertNotIn("models.flux3.runtime", sys.modules)

    def test_flow_map_sampler_keeps_fp32_euler_state_and_cancels_after_one_step(self):
        packed = Packed(
            data={"image": torch.ones((2, 128), dtype=torch.float32)},
            ids={"image": torch.zeros((2, 4))},
            timesteps={"image": torch.zeros((2,), dtype=torch.bfloat16)},
            mode_indices={"image": [(0, 2)], "image_cond": [(0, 0)], "text": [(0, 0)]},
        )
        dit = _ConstantDiT(2.0)
        result = denoise_flow_map(dit, packed, torch.tensor([1.0, 0.5, 0.0]))
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.full((2, 128), -1.0))

        cancelled = False
        callback_steps = []
        cancelling_dit = _ConstantDiT(2.0)

        def check_cancel():
            if cancelled:
                raise RuntimeError("cancel requested")

        def cancel_after_first(step):
            nonlocal cancelled
            callback_steps.append(step)
            cancelled = True

        with self.assertRaisesRegex(RuntimeError, "cancel requested"):
            denoise_flow_map(
                cancelling_dit,
                packed,
                timesteps(steps=5, alpha=DEFAULT_ALPHA),
                on_step=cancel_after_first,
                check_cancel=check_cancel,
            )
        self.assertEqual(callback_steps, [0])
        self.assertEqual(cancelling_dit.calls, 1)

    def test_flow_map_sampler_expands_lora_multipliers_and_selects_each_step(self):
        from shared.utils.loras_mutipliers import parse_loras_multipliers

        packed = Packed(
            data={"image": torch.zeros((1, 128))},
            ids={"image": torch.zeros((1, 4))},
            timesteps={"image": torch.zeros((1,), dtype=torch.bfloat16)},
            mode_indices={"image": [(0, 1)], "image_cond": [(0, 0)], "text": [(0, 0)]},
        )
        _, loras_slists, error = parse_loras_multipliers(["0.25,0.75"], 1, 2, nb_phases=1)
        self.assertEqual(error, "")
        activate_loras = Mock()
        set_step = Mock()
        with patch.dict(sys.modules, {"mmgp": mmgp_stub}):
            with patch.object(mmgp_stub.offload, "activate_loras", activate_loras, create=True), \
                    patch.object(mmgp_stub.offload, "set_step_no_for_lora", set_step, create=True):
                denoise_flow_map(
                    _ConstantDiT(), packed, torch.tensor([1.0, 0.5, 0.0]), loras_slists=loras_slists,
                )

        self.assertEqual(activate_loras.call_args.args[1:], (["0"], [[0.25, 0.75]]))
        self.assertEqual([call.args[1] for call in set_step.call_args_list], [0, 1])

    def test_default_flow_map_schedule_has_sixteen_steps(self):
        schedule = timesteps()
        self.assertEqual(len(schedule) - 1, DEFAULT_STEPS)
        self.assertAlmostEqual(schedule[0].item(), 1.0, places=5)
        self.assertEqual(schedule[-1].item(), 0.0)

    def test_prompt_features_pad_to_eighty_and_concatenate_layers_four_through_thirty_two(self):
        processor = _FakeTextProcessor()
        model = _DistinctiveHiddenStateModel()
        tokens, ids = encode_prompt(model, processor, "A short prompt", "cpu")
        self.assertEqual(tuple(tokens.shape), (160, FEATURE_WIDTH))
        self.assertEqual(tokens.dtype, torch.bfloat16)
        self.assertEqual(tuple(ids.shape), (160, 4))
        self.assertEqual(ids[:, 3].tolist(), list(range(160)))
        self.assertEqual(model.call_kwargs["position_ids"].shape, (4, 1, 160))
        self.assertTrue(model.call_kwargs["output_hidden_states"])
        for output_index, layer in enumerate(OUTPUT_LAYERS):
            start = output_index * 2560
            self.assertTrue(torch.all(tokens[0, start:start + 2560] == layer))
        model.lm_head.assert_not_called()

    def test_native_autoencoder_decodes_unscaled_grid_ids_to_requested_canvas(self):
        height, width = 32, 48
        latent_h, latent_w = height // 16, width // 16
        ids = latent_ids(latent_h, latent_w)
        with torch.device("meta"):
            vae = AutoEncoder()
            tokens = torch.empty((latent_h * latent_w, 128), device="meta")
        decoded = vae.decode(tokens, ids)
        self.assertEqual(tuple(decoded.shape), (3, height, width))

    def test_native_vae_locks_parameters_root_and_running_stat_buffers_for_mmgp(self):
        with torch.device("meta"):
            vae = AutoEncoder()
        runtime_module._lock_native_vae_fp32(vae)
        runtime_module._configure_native_vae_offload(vae)
        self.assertEqual(vae._lock_dtype, torch.float32)
        self.assertEqual(vae.encoder.conv_in._lock_dtype, torch.float32)
        self.assertEqual(vae.bn._lock_dtype, torch.float32)
        self.assertEqual(vae._convertWeightsFloatTo, torch.float32)
        self.assertEqual(vae._model_dtype, torch.float32)
        self.assertEqual(vae._offload_hooks, ["encode", "decode"])

    def test_numbered_encoder_shards_are_passed_to_mmgp_once(self):
        first = "C:/models/text/model-00001-of-00002.safetensors"
        second = "C:/models/text/model-00002-of-00002.safetensors"
        with patch.object(runtime_module.offload, "load_model_data") as loader:
            runtime_module._load_mmgp_checkpoint(
                torch.nn.Linear(2, 2), [first, second], default_dtype=torch.bfloat16
            )
        self.assertEqual(loader.call_args.args[1], first)
        self.assertEqual(loader.call_args.kwargs["default_dtype"], torch.bfloat16)

    def test_qwen_checkpoint_prefix_uses_mmgp_prefix_filter_for_quantization_metadata(self):
        checkpoint = "C:/models/text/qwen3vl_4b_int8_convrot.safetensors"
        with patch.object(runtime_module.offload, "load_model_data") as loader:
            runtime_module._load_mmgp_checkpoint(
                torch.nn.Linear(2, 2), [checkpoint],
                modelPrefix="model", writable_tensors=False,
                default_dtype=torch.bfloat16,
            )

        # MMGP applies modelPrefix to state_dict, quantization_map, and
        # tied_weights_map after reading safetensors metadata. The old
        # state-only preprocess_sd path left the quantization map prefixed,
        # so INT8 descriptors were silently dropped for the bare Qwen model.
        self.assertEqual(loader.call_args.args[1], checkpoint)
        self.assertEqual(loader.call_args.kwargs["modelPrefix"], "model")
        self.assertNotIn("preprocess_sd", loader.call_args.kwargs)

    def test_checkpoint_validation_requires_exact_names_and_shapes(self):
        module = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.safetensors"
            save_file({"weight": torch.zeros((2, 3)), "bias": torch.zeros(2)}, str(valid))
            runtime_module._validate_checkpoint_shapes(module, [str(valid)], "test")

            wrong = Path(directory) / "wrong.safetensors"
            save_file({"weight": torch.zeros((3, 2)), "bias": torch.zeros(2)}, str(wrong))
            with self.assertRaisesRegex(ValueError, "shape mismatches"):
                runtime_module._validate_checkpoint_shapes(module, [str(wrong)], "test")

    def test_generic_zero_outpaint_values_are_inactive_but_positive_margins_are_rejected(self):
        self.assertFalse(runtime_module._active_outpainting([0, 0, 0, 0]))
        self.assertFalse(runtime_module._active_outpainting("0 0 0 0"))
        self.assertFalse(runtime_module._active_outpainting("0, 0, 0, 0"))
        self.assertTrue(runtime_module._active_outpainting([0, 1, 0, 0]))
        self.assertTrue(runtime_module._active_outpainting("0 1 0 0"))

    def test_reference_inputs_are_used_only_when_reference_selector_is_active(self):
        self.assertEqual(runtime_module._active_references("", [object()], None), [])
        with self.assertRaisesRegex(ValueError, "reference-image editing only"):
            runtime_module._active_references("V", None, None)

    def test_ten_references_keep_order_and_distinct_native_conditioning_ids(self):
        runtime = _runtime_for_cpu_test()
        references = [Image.new("RGB", (32, 32), (index * 20, 0, 0)) for index in range(10)]
        captured = []

        def encode(pixels):
            marker = round(float((pixels[0, 0, 0] + 1) * 127.5))
            return torch.full((4, 128), marker), latent_ids(2, 2)

        class CapturingDiT(_ZeroDiT):
            def forward(self, packed, timesteps_ref=None):
                captured.append((packed.data["image_cond"].clone(), packed.ids["image_cond"].clone()))
                return super().forward(packed, timesteps_ref)

        runtime.vae.encode = Mock(side_effect=encode)
        runtime.transformer = CapturingDiT()
        with patch.object(runtime_module, "encode_prompt",
                          return_value=(torch.zeros((1, 20480)), torch.zeros((1, 4)))):
            output = runtime_module.model_factory.generate(
                runtime, input_prompt="Combine all ten references", sampling_steps=1,
                width=64, height=64, video_prompt_type="I", input_ref_images=references,
            )
        self.assertEqual(tuple(output.shape), (3, 1, 64, 64))
        self.assertEqual(runtime.vae.encode.call_count, 10)
        tokens, ids = captured[0]
        self.assertEqual(tuple(tokens.shape), (40, 128))
        for index in range(10):
            section = slice(index * 4, (index + 1) * 4)
            self.assertTrue(torch.all(tokens[section] == index * 20))
            self.assertTrue(torch.all(ids[section, 3] == -(index + 1)))
            torch.testing.assert_close(ids[section, 1:3], latent_ids(2, 2)[:, 1:3] * 0.5)

    def test_eleven_active_references_are_rejected_without_silently_dropping_one(self):
        references = [Image.new("RGB", (32, 32)) for _ in range(11)]
        with self.assertRaisesRegex(ValueError, "up to 10 reference images"):
            runtime_module._active_references("I", references, None)

    def test_generation_accepts_custom_positive_steps_and_reports_batch_progress(self):
        runtime = _runtime_for_cpu_test()
        callback_calls = []

        def callback(*args, **kwargs):
            callback_calls.append((args, kwargs))

        with patch.object(runtime_module, "encode_prompt",
                          return_value=(torch.zeros((1, 20480)), torch.zeros((1, 4)))):
            output = runtime_module.model_factory.generate(
                runtime,
                input_prompt="A red kite",
                seed=42,
                sampling_steps=3,
                width=64,
                height=64,
                batch_size=2,
                video_prompt_type="",
                input_ref_images=[object()],
                input_frames=object(),
                image_start=object(),
                outpainting_dims="0 0 0 0",
                callback=callback,
            )

        self.assertEqual(tuple(output.shape), (3, 2, 64, 64))
        self.assertEqual(tuple(runtime.vae.decoded_ids[0][:, 1:3].max(dim=0).values.tolist()), (3, 3))
        self.assertEqual(callback_calls[0][0], (-1, None, True))
        self.assertEqual(callback_calls[0][1]["override_num_inference_steps"], 6)
        self.assertEqual([call[0][0] for call in callback_calls[1:]], list(range(6)))

    def test_nonpositive_steps_are_rejected_before_runtime_work(self):
        runtime = _runtime_for_cpu_test()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            runtime_module.model_factory.generate(runtime, input_prompt="test", sampling_steps=0)

    def test_generation_rejects_nonfinite_native_vae_pixels(self):
        runtime = _runtime_for_cpu_test()
        runtime.vae = _ShapeVAE(nonfinite=True)
        with patch.object(runtime_module, "encode_prompt",
                          return_value=(torch.zeros((1, 20480)), torch.zeros((1, 4)))):
            with self.assertRaisesRegex(RuntimeError, "non-finite pixels"):
                runtime_module.model_factory.generate(
                    runtime, input_prompt="test", sampling_steps=1, width=64, height=64
                )


if __name__ == "__main__":
    unittest.main()
