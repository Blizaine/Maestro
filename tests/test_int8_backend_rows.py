"""CPU contract tests for bounded Triton ConvRot row projections."""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
import weakref
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

_APP = Path(__file__).resolve().parents[1] / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))


def _import_project_module(name):
    return __import__(name, fromlist=["*"])


class _FakeTensor:
    """Small metadata-only tensor for CPU routing tests."""

    def __init__(self, shape, dtype=torch.bfloat16, device=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = device or torch.device("cuda", 0)
        self.is_cuda = True
        self.requires_grad = False

    def is_contiguous(self):
        return True

    def __getitem__(self, key):
        start, stop, step = key.indices(self.shape[0])
        rows = len(range(start, stop, step))
        return _FakeTensor((rows, *self.shape[1:]), self.dtype, self.device)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _FakeTensor(shape, self.dtype, self.device)

    def numel(self):
        size = 1
        for dim in self.shape:
            size *= dim
        return size


class TestTritonConvRotRows(unittest.TestCase):
    def setUp(self):
        self.backend = _import_project_module("shared.kernels.int8_backend")
        self.inject = _import_project_module("shared.kernels.quanto_int8_inject")
        self.convrot = _import_project_module("shared.qtypes.int8_convrot")
        self.provider = self.backend._TRITON_ROW_PROJECTION
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)

    def _fake_module(self, rows=71_680, features=5_376, outputs=7_168):
        device = torch.device("cuda", 0)
        data = _FakeTensor((outputs, features), torch.int8, device)
        scales = _FakeTensor((outputs, 1), torch.float32, device)
        weight = SimpleNamespace(
            _data=data,
            _scale=scales,
            dtype=torch.bfloat16,
            shape=(outputs, features),
        )
        return SimpleNamespace(
            weight_qtype=SimpleNamespace(name="qint8_convrot"),
            _convrot_group_size=256,
            qweight=weight,
            bias=None,
            training=False,
            _mm_lora_data=None,
        ), _FakeTensor((rows, features), torch.bfloat16, device)

    def _patch_metadata_only_cuda(self, *, native_results=None, lock=True, block_k=64):
        self.patches.close()
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        original_is_tensor = torch.is_tensor
        is_tensor = lambda value: isinstance(value, _FakeTensor) or original_is_tensor(value)
        triton_module = SimpleNamespace(
            _env_flag=lambda _name, default="1": lock,
            _select_static_triton_int8_config=lambda _m, _k, _n: (64, 128, block_k, 8, 4),
        )
        self.patches.enter_context(mock.patch.object(self.backend, "_backend", "triton"))
        self.patches.enter_context(mock.patch.object(self.inject, "_TRITON_MODULE", triton_module))
        self.patches.enter_context(mock.patch.object(self.inject, "_RUNTIME_DISABLED", False))
        self.patches.enter_context(mock.patch.object(self.inject, "_NATIVE_FALLBACK_MAX_M", 0))
        self.patches.enter_context(mock.patch.object(self.inject, "_CONVROT_BACKENDS", {}))
        self.patches.enter_context(mock.patch.object(torch, "is_tensor", side_effect=is_tensor))
        self.patches.enter_context(mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        self.patches.enter_context(mock.patch.object(torch.compiler, "is_compiling", return_value=False))
        if native_results is None:
            native_results = []
        native_calls = []
        native_iter = iter(native_results)
        self.patches.enter_context(mock.patch.object(
            self.inject,
            "_prefer_native_convrot_path",
            side_effect=lambda probe, _weight, _bias: (
                native_calls.append(tuple(probe.shape)) or next(native_iter, False)
            ),
        ))
        self.patches.enter_context(mock.patch.object(
            self.convrot,
            "_rotate_activation",
            side_effect=lambda x, _group: x,
        ))
        return native_calls

    def test_configure_registers_only_the_active_backend_provider(self):
        offload = _import_project_module("mmgp.offload")
        from optimum.quanto.tensor.weights import qbytes

        names = (
            "_backend", "_kitchen", "_kitchen_hip", "_direct_cutlass",
            "_sm120_cutlass", "_original_forward", "_wide_convrot_triton",
            "_blockwise_triton", "_blockwise_required", "_kitchen_dlpack_export",
            "revision",
        )
        saved = {name: getattr(self.backend, name) for name in names}
        saved_forward = qbytes.WeightQBytesLinearFunction.forward
        saved_env = __import__("os").environ.get("WAN2GP_QUANTO_INT8_KERNEL")
        kitchen = SimpleNamespace(
            _DISABLE_CUTLASS_INT8=True,
            _wrap_for_dlpack=lambda tensor: tensor,
        )
        try:
            with mock.patch.object(self.backend.triton, "disable_quanto_int8_kernel"), \
                 mock.patch.object(self.backend.triton, "maybe_enable_quanto_int8_kernel", return_value=True), \
                 mock.patch.object(self.backend, "_register_ops"), \
                 mock.patch.object(offload, "register_row_projection") as register, \
                 mock.patch.object(offload, "unregister_row_projection") as unregister, \
                 mock.patch.object(self.backend, "_backend", "pytorch"), \
                 mock.patch.object(self.backend, "_original_forward", None), \
                 mock.patch.object(self.backend, "_kitchen_dlpack_export", None), \
                 mock.patch.object(self.backend, "_blockwise_required", False):
                self.backend.configure("triton", resolved=("triton", object()))
                register.assert_called_with(self.backend._TRITON_ROW_PROJECTION)
                unregister.assert_called_with(self.backend._KITCHEN_ROW_PROJECTION)

                self.backend.configure("kitchen", resolved=("kitchen", kitchen))
                register.assert_called_with(self.backend._KITCHEN_ROW_PROJECTION)
                unregister.assert_called_with(self.backend._TRITON_ROW_PROJECTION)

                self.backend.configure("disabled", resolved=("pytorch", None))
                unregister.assert_any_call(self.backend._KITCHEN_ROW_PROJECTION)
                unregister.assert_any_call(self.backend._TRITON_ROW_PROJECTION)
        finally:
            qbytes.WeightQBytesLinearFunction.forward = staticmethod(saved_forward)
            for name, value in saved.items():
                setattr(self.backend, name, value)
            if saved_env is None:
                __import__("os").environ.pop("WAN2GP_QUANTO_INT8_KERNEL", None)
            else:
                __import__("os").environ["WAN2GP_QUANTO_INT8_KERNEL"] = saved_env
            kitchen._wrap_for_dlpack = lambda tensor: tensor

    @unittest.skipUnless(
        os.environ.get("MAESTRO_TEST_INT8_ROWS_NATIVE_SLA") == "1",
        "Explicit native-shape CUDA H3 SLA component test",
    )
    def test_cuda_native_h3_sla_with_quantized_rows_and_owned_input_release(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        from mmgp import offload, quant_router

        transformer = _import_project_module("models.minimax_h3.transformer")
        attention_kit = _import_project_module("shared.attention_kit")
        sla_map_module = _import_project_module("models.minimax_h3.sla_block_map")
        sla_class = _import_project_module("models.minimax_h3.sla_attention").MiniMaxH3SLAAttention
        device = torch.device("cuda", 0)
        dtype = torch.bfloat16
        hidden_size, heads, head_dim = 5_376, 56, 128
        outputs = heads * head_dim
        tokens = 107 * 34 * 60 + 1_206 + 400
        protected_prefix = 1_206 + 400
        total_memory = torch.cuda.get_device_properties(device).total_memory
        cap_bytes = min(12 * 1024**3, total_memory)
        torch.cuda.set_per_process_memory_fraction(cap_bytes / total_memory, device)
        quant_router.register_handler("shared.qtypes.int8_convrot")

        def make_projection(seed):
            generator = torch.Generator().manual_seed(seed)
            model = torch.nn.Module()
            model.proj = torch.nn.Linear(hidden_size, outputs, bias=False, dtype=dtype)
            weight = torch.randint(
                -8, 9, (outputs, hidden_size), generator=generator, dtype=torch.int8
            )
            scale = torch.full((outputs, 1), 0.003, dtype=torch.float32)
            metadata = json.dumps({
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            }).encode("utf-8")
            state = {
                "proj.weight": weight,
                "proj.weight_scale": scale,
                "proj.comfy_quant": torch.tensor(list(metadata), dtype=torch.uint8),
            }
            offload.load_model_data(
                model, (state, None), default_dtype=dtype, verboseLevel=0
            )
            return model.proj.to(device).eval()

        previous_split = attention_kit.head_split
        previous_kernel_check = os.environ.get("MAESTRO_CONVROT_KERNEL_CHECK")
        attention_kit.configure(2)
        previous_mode = offload.shared_state.get("_attention")
        had_mode = "_attention" in offload.shared_state
        previous_runtime_backend = self.backend._backend
        started = time.perf_counter()
        try:
            triton_mod, reason = self.inject._probe_triton_backend()
            if triton_mod is None:
                self.skipTest(f"Triton is unavailable: {reason}")
            with mock.patch.dict(os.environ, {"MAESTRO_CONVROT_KERNEL_CHECK": "0"}):
                self.backend.configure("triton", resolved=("triton", triton_mod))
            # Keep this setting active during provider eligibility and forwards.
            env_context = mock.patch.dict(os.environ, {"MAESTRO_CONVROT_KERNEL_CHECK": "0"})
            env_context.start()
            try:
                sla = sla_class({
                    "sparsity_ratio": 0.90,
                    "block_size": 64,
                    "min_seq_len": 8_192,
                    "dense_last_steps": 0,
                    "protect_audio": True,
                })
                attention = transformer.MiniMaxH3Attention(
                    hidden_size, heads, head_dim, 1e-6, dtype, sla_attention=sla
                )
                del attention.qkv_proj
                attention.qkv_proj = None
                attention.q_proj = make_projection(101)
                attention.k_proj = make_projection(102)
                attention.v_proj = make_projection(103)
                attention.eval().to(device)

                offload.shared_state["_attention"] = "sla"
                sla.begin_forward(protected_prefix, device, dtype)
                self.assertTrue(sla.enabled, "real CUDA SLA runtime did not enable")
                self.assertTrue(sla.use_for_layer(tokens, None))

                positions = torch.arange(tokens, device=device, dtype=torch.long)
                position_ids = torch.stack(
                    (positions, positions // 34, positions % 60), dim=-1
                )
                rotary = transformer.MiniMaxH3RotaryEmbedding(freq_dim=16).to(device)(
                    position_ids
                )

                with torch.inference_mode():
                    residual = torch.randn(
                        (1, tokens, hidden_size), device=device, dtype=dtype
                    )
                    norm = torch.nn.RMSNorm(
                        hidden_size, eps=1e-6, dtype=dtype
                    ).to(device)
                    normalized = transformer._rms_norm_in_chunks(norm, residual)
                    source_ref = weakref.ref(normalized)
                    provider_supported = offload.linear_rows_supported(
                        (attention.q_proj, attention.k_proj, attention.v_proj),
                        normalized,
                    )
                    self.assertTrue(provider_supported)
                    owned_input = [normalized]
                    del normalized

                    memory_trace = []

                    def record_memory(stage, **extra):
                        memory_trace.append({
                            "stage": stage,
                            "allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
                            "reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
                            "source_alive": source_ref() is not None,
                            **extra,
                        })

                    protected_calls = []
                    real_map = sla_map_module.get_block_map

                    def record_real_map(*args, **kwargs):
                        protected_calls.append({
                            "prefix": int(kwargs.get("protect_upto", -1)),
                            "source_released": source_ref() is None,
                        })
                        return real_map(*args, **kwargs)

                    row_calls = []
                    real_rows = offload.linear_rows
                    real_prepare = self.provider.prepare
                    real_norm = transformer._rms_norm_inplace

                    def record_prepare(module, rows):
                        prepared = real_prepare(module, rows)
                        record_memory(
                            "prepare_return",
                            output_shape=list(prepared.shape),
                            output_dtype=str(prepared.dtype),
                        )
                        return prepared

                    def record_rows(module, linear_input, start, stop):
                        output_rows = real_rows(module, linear_input, start, stop)
                        row_calls.append((module, int(start), int(stop)))
                        projection = next(
                            name for name in ("q", "k", "v")
                            if module is getattr(attention, name + "_proj")
                        )
                        record_memory(
                            "row_return",
                            projection=projection,
                            start=int(start),
                            stop=int(stop),
                            output_shape=list(output_rows.shape),
                            output_dtype=str(output_rows.dtype),
                        )
                        return output_rows

                    def record_norm(norm_module, tensor, *args, **kwargs):
                        record_memory(
                            "rms_norm_entry",
                            input_shape=list(tensor.shape),
                            input_dtype=str(tensor.dtype),
                            weight_dtype=str(norm_module.weight.dtype),
                        )
                        return real_norm(norm_module, tensor, *args, **kwargs)

                    torch.cuda.reset_peak_memory_stats(device)
                    record_memory("attention_entry")
                    with (
                        # Avoid MagicMock call_args retaining full-sequence CUDA Q/K tensors.
                        mock.patch.object(sla_map_module, "get_block_map", new=record_real_map),
                        mock.patch.object(offload, "linear_rows", new=record_rows),
                        mock.patch.object(self.provider, "prepare", new=record_prepare),
                        mock.patch.object(transformer, "_rms_norm_inplace", new=record_norm),
                    ):
                        try:
                            output = attention(owned_input, rotary=rotary)
                        except torch.cuda.OutOfMemoryError:
                            failure = {
                                "status": "oom_in_real_h3_attention",
                                "tokens": tokens,
                                "hidden_size": hidden_size,
                                "heads": heads,
                                "head_dim": head_dim,
                                "dtype": str(dtype),
                                "backend": self.backend._backend,
                                "provider_supported": bool(provider_supported),
                                "row_projection_calls": len(row_calls),
                                "sla_sparse_calls": sla._calls,
                                "block_map_calls": len(protected_calls),
                                "source_released_at_oom": source_ref() is None,
                                "owned_input_empty_at_oom": len(owned_input) == 0,
                                "residual_retained_at_oom": residual is not None,
                                "allocator_cap_gib": cap_bytes / 1024**3,
                                "current_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
                                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                                "memory_trace": memory_trace,
                            }
                            print("INT8_CONVROT_H3_SLA_RESULT=" + json.dumps(failure, sort_keys=True), flush=True)
                            raise
                    torch.cuda.synchronize(device)

                group_count = 8
                group_width = outputs // group_count
                self.assertEqual(len(owned_input), 0)
                self.assertIsNone(source_ref(), "original normalized source remained live")
                self.assertIsNotNone(residual, "the block residual must remain live")
                self.assertEqual(tuple(residual.shape), (1, tokens, hidden_size))
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertEqual(tuple(output.shape), (1, tokens, hidden_size))
                self.assertEqual(sla._calls, group_count)
                self.assertFalse(sla._runtime_failed, "SLA fell back from sparse attention")
                self.assertEqual(len(protected_calls), group_count)
                self.assertEqual(
                    [call["prefix"] for call in protected_calls],
                    [protected_prefix] * group_count,
                )
                self.assertTrue(all(call["source_released"] for call in protected_calls))
                self.assertEqual(len(row_calls), 3 * group_count)
                for group in range(group_count):
                    start, stop = group * group_width, (group + 1) * group_width
                    self.assertEqual(
                        row_calls[group * 3 : group * 3 + 3],
                        [
                            (attention.q_proj, start, stop),
                            (attention.k_proj, start, stop),
                            (attention.v_proj, start, stop),
                        ],
                    )
                result = {
                    "status": "ok",
                    "tokens": tokens,
                    "hidden_size": hidden_size,
                    "heads": heads,
                    "head_dim": head_dim,
                    "dtype": str(dtype),
                    "backend": self.backend._backend,
                    "provider_supported": bool(provider_supported),
                    "sla_sparse_calls": sla._calls,
                    "protected_prefix_rows": protected_prefix,
                    "source_released_by_first_sparse_call": protected_calls[0]["source_released"],
                    "source_released_after_forward": source_ref() is None,
                    "residual_retained": residual is not None,
                    "full_attention_output_shape": list(output.shape),
                    "row_projection_calls": len(row_calls),
                    "allocator_cap_gib": cap_bytes / 1024**3,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                    "elapsed_seconds": time.perf_counter() - started,
                    "memory_trace": memory_trace,
                }
                print("INT8_CONVROT_H3_SLA_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
            finally:
                env_context.stop()
        finally:
            attention_kit.configure(previous_split)
            if had_mode:
                offload.shared_state["_attention"] = previous_mode
            else:
                offload.shared_state.pop("_attention", None)
            # This opt-in test runs as an isolated CUDA subprocess. Return the
            # backend to its safe default before that process exits.
            if self.backend._backend != previous_runtime_backend:
                self.backend.configure("disabled", resolved=("pytorch", None))
            if previous_kernel_check is None:
                os.environ.pop("MAESTRO_CONVROT_KERNEL_CHECK", None)
            else:
                os.environ["MAESTRO_CONVROT_KERNEL_CHECK"] = previous_kernel_check

    def test_support_probes_h3_chunk_and_tail_then_accepts_only_triton_choice(self):
        module, x = self._fake_module(rows=50_849)
        native_calls = self._patch_metadata_only_cuda()
        with torch.no_grad():
            self.assertTrue(self.provider.supports(module, x))
        self.assertEqual(native_calls, [(1, 8192, 5_376), (1, 1_697, 5_376)])

        module, x = self._fake_module(rows=57_345)
        native_calls = self._patch_metadata_only_cuda()
        with torch.no_grad():
            self.assertTrue(self.provider.supports(module, x))
        self.assertEqual(native_calls, [(1, 8192, 5_376), (1, 1, 5_376)])

        module, x = self._fake_module(rows=50_849)
        native_calls = self._patch_metadata_only_cuda(native_results=[False, True])
        with torch.no_grad():
            self.assertFalse(self.provider.supports(module, x))
        self.assertEqual(native_calls, [(1, 8192, 5_376), (1, 1_697, 5_376)])

    def test_cached_triton_dispatch_skips_repeated_rotation_probe(self):
        module, x = self._fake_module(rows=50_849)
        weight = module.qweight
        rows = (8192, 1_697)
        cached = {
            (x.device, x.dtype, (1, row_count, x.shape[1]),
             (row_count * x.shape[1], x.shape[1], 1), tuple(weight.shape)): False
            for row_count in rows
        }
        with mock.patch.object(self.inject, "_CONVROT_BACKENDS", cached):
            with mock.patch.object(
                self.convrot, "_rotate_activation", side_effect=AssertionError("cache missed")
            ):
                self.assertFalse(self.provider._native_dispatch_selected(module, x))

    def test_support_rejects_unlocked_or_incompatible_kernel_configuration(self):
        module, x = self._fake_module()
        self._patch_metadata_only_cuda(lock=False)
        with torch.no_grad():
            self.assertFalse(self.provider.supports(module, x))

        module, x = self._fake_module()
        self._patch_metadata_only_cuda()
        triton_module = self.inject._TRITON_MODULE
        triton_module._select_static_triton_int8_config = lambda _m, _k, n: (
            64, 128, 128 if n == 7_168 else 64, 8, 4
        )
        with torch.no_grad():
            self.assertFalse(self.provider.supports(module, x))

        module, x = self._fake_module(rows=57_345)
        self._patch_metadata_only_cuda()
        triton_module = self.inject._TRITON_MODULE
        triton_module._select_static_triton_int8_config = lambda m, _k, _n: (
            64, 128, 128 if m < 256 else 64, 8, 4
        )
        with torch.no_grad():
            self.assertFalse(self.provider.supports(module, x))

    def test_support_rejects_small_training_lora_and_native_threshold_inputs(self):
        for rows, training, lora, native_max in (
            (49_999, False, None, 0),
            (71_680, True, None, 0),
            (71_680, False, {"adapter": object()}, 0),
            (71_680, False, None, 7_000),
        ):
            with self.subTest(rows=rows, training=training, lora=bool(lora), native_max=native_max):
                module, x = self._fake_module(rows=rows)
                module.training = training
                module._mm_lora_data = lora
                self._patch_metadata_only_cuda()
                self.inject._NATIVE_FALLBACK_MAX_M = native_max
                with torch.no_grad():
                    self.assertFalse(self.provider.supports(module, x))

    def test_prepare_rotates_owned_destination_with_scratch_bounded_bf16_tiles(self):
        module = SimpleNamespace(_convrot_group_size=256)
        torch.manual_seed(17)
        x = torch.randn((10, 512), dtype=torch.bfloat16)
        expected = self.convrot._rotate_activation(x.clone(), 256)
        source_before = x.clone()
        data_ptr = x.data_ptr()
        tile_rows = []
        rotate = self.convrot._rotate_activation

        def record_tile(tile, group):
            tile_rows.append(tile.shape[0])
            return rotate(tile, group)

        with mock.patch.object(self.backend, "_SCRATCH_BYTES", 4 * 512 * 2):
            with mock.patch.object(self.convrot, "_rotate_activation", side_effect=record_tile):
                prepared = self.provider.prepare(module, x)
        self.assertIsNot(prepared, x)
        self.assertNotEqual(prepared.data_ptr(), data_ptr)
        self.assertEqual(max(tile_rows), 4)
        self.assertEqual(sum(tile_rows), 10)
        torch.testing.assert_close(prepared, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, source_before, rtol=0, atol=0)

    def test_mixed_provider_preparation_keeps_plain_projection_source_unchanged(self):
        offload = _import_project_module("mmgp.offload")
        torch.manual_seed(23)
        x = torch.randn((1, 10, 512), dtype=torch.bfloat16)
        source_before = x.clone()
        class Int8Module:
            _convrot_group_size = 256

        int8_module = Int8Module()
        plain_module = torch.nn.Linear(512, 12, bias=True, dtype=torch.bfloat16)

        def choose_provider(module, flattened):
            if module is int8_module:
                return self.provider
            return offload._PLAIN_ROW_PROJECTION

        with mock.patch.object(self.backend, "_SCRATCH_BYTES", 4 * 512 * 2):
            with mock.patch.object(offload, "_row_projection", side_effect=choose_provider):
                linear_input = offload.prepare_linear_input(
                    [x], [int8_module, plain_module]
                )
                actual = offload.linear_rows(plain_module, linear_input, 0, 12)
        self.assertIsNotNone(linear_input.x)
        torch.testing.assert_close(linear_input.x, source_before.reshape(-1, 512), rtol=0, atol=0)
        torch.testing.assert_close(x, source_before, rtol=0, atol=0)
        torch.testing.assert_close(
            actual,
            F.linear(source_before, plain_module.weight, plain_module.bias),
            rtol=0,
            atol=0,
        )

    def test_row_ranges_match_full_triton_projection_and_bias(self):
        torch.manual_seed(19)
        rows, features, outputs = 23, 256, 512
        x = torch.randn((rows, features), dtype=torch.bfloat16)
        original = x.clone()
        data = torch.randint(-8, 9, (outputs, features), dtype=torch.int8)
        scales = torch.rand((outputs, 1), dtype=torch.float32).add_(0.01)
        bias = torch.randn((outputs,), dtype=torch.bfloat16)
        weight = SimpleNamespace(_data=data, _scale=scales, dtype=torch.bfloat16)
        module = SimpleNamespace(qweight=weight, bias=bias, _convrot_group_size=256)
        prepared = self.provider.prepare(module, x)
        rotated = self.convrot._rotate_activation(original, 256)

        def fake_fused(activation, qweight, qscale, out_dtype):
            result = activation.float() @ qweight.float().T
            result.mul_(qscale.reshape(1, -1))
            return result.to(out_dtype)

        with mock.patch.object(self.inject, "_prepare_weight_scale", side_effect=lambda scale, _n, _device: scale.reshape(-1)):
            with mock.patch.object(self.inject, "_fused_quant_scaled_mm_call", side_effect=fake_fused):
                actual = torch.cat([
                    self.provider.rows(module, prepared, 0, 128),
                    self.provider.rows(module, prepared, 128, outputs),
                ], dim=-1)
        expected = fake_fused(rotated, data, scales.reshape(-1), torch.bfloat16)
        expected.add_(bias)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
