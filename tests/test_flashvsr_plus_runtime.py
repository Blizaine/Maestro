"""CPU-only FlashVSR v1.1 and Plus Turbo scheduling, attention, and asset contracts."""
from __future__ import annotations

import sys
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from postprocessing.flashvsr import runtime
from postprocessing.flashvsr.wan_video_dit import (
    generate_draft_block_mask,
    generate_draft_block_mask_sage,
    structural_block_candidates,
    structural_block_mask,
)
from postprocessing.flashvsr.wgp_bridge import FlashVSRBridge


class _FilesLocator:
    def locate_file(self, name, error_if_none=True, **_kwargs):
        if error_if_none:
            raise FileNotFoundError(name)
        return None


class FlashVSRModelSelectionTests(unittest.TestCase):
    def test_v11_remains_default_and_settings_tuple_is_unchanged(self):
        bridge = FlashVSRBridge({"flashvsr_mode": FlashVSRBridge.MODE_TINY}, _FilesLocator())
        self.assertEqual(bridge.settings(), (True, "tiny", FlashVSRBridge.PERSIST_UNLOAD))
        self.assertEqual(bridge.model(), FlashVSRBridge.MODEL_V11)
        self.assertEqual(bridge.model_for_upsampling("flashvsr2"), FlashVSRBridge.MODEL_V11)
        self.assertEqual(bridge.model_for_upsampling("flashvsr2pass2"), FlashVSRBridge.MODEL_V11)

    def test_versioned_ids_select_distinct_models_and_keep_scale_contract(self):
        bridge = FlashVSRBridge({"flashvsr_mode": FlashVSRBridge.MODE_TINY}, _FilesLocator())
        cases = (
            ("flashvsr2", FlashVSRBridge.MODEL_V11, 2.0, False),
            ("flashvsr2pass2", FlashVSRBridge.MODEL_V11, 2.0, True),
            ("flashvsrplus2", FlashVSRBridge.MODEL_PLUS_TURBO, 2.0, False),
            ("flashvsrplus2pass2", FlashVSRBridge.MODEL_PLUS_TURBO, 2.0, True),
        )
        for method, expected_model, expected_scale, expected_two_pass in cases:
            with self.subTest(method=method):
                self.assertEqual(bridge.model_for_upsampling(method), expected_model)
                self.assertEqual(bridge.scale_for_upsampling(method), expected_scale)
                self.assertEqual(bridge.is_two_pass_upsampling(method), expected_two_pass)

    def test_invalid_persisted_model_normalizes_to_legacy_v11(self):
        bridge = FlashVSRBridge(
            {"flashvsr_mode": FlashVSRBridge.MODE_TINY, "flashvsr_model": "future_model"},
            _FilesLocator(),
        )
        self.assertEqual(bridge.model(), FlashVSRBridge.MODEL_V11)

    def test_plus_download_definition_is_pinned_and_isolated_from_v11_files(self):
        bridge = FlashVSRBridge({"flashvsr_mode": FlashVSRBridge.MODE_TINY}, _FilesLocator())
        definition = bridge.query_download_def(
            enabled_only=False,
            model=FlashVSRBridge.MODEL_PLUS_TURBO,
            variant="tiny-long",
        )
        self.assertEqual(definition["repoId"], "ruchangyao/FlashVSR_plus")
        self.assertEqual(definition["revision"], bridge.PLUS_REPO_REVISION)
        self.assertEqual(definition["sourceFolderList"], ["turbo", ""])
        self.assertEqual(definition["targetFolderList"], ["FlashVSR", "FlashVSR"])
        self.assertEqual(
            definition["fileNameMap"][bridge.PLUS_TRANSFORMER_FILENAME],
            "dit.safetensors",
        )
        self.assertEqual(
            definition["fileNameMap"][bridge.PLUS_LQ_PROJ_FILENAME],
            "lq_proj_in.safetensors",
        )
        self.assertEqual(definition["fileNameMap"][bridge.PLUS_TCDECODER_FILENAME], "TCDecoder.ckpt")
        self.assertNotIn(bridge.TRANSFORMER_FILENAME, definition["fileNameMap"])
        for metadata in bridge.PLUS_ASSET_METADATA.values():
            self.assertEqual(len(metadata["sha256"]), 64)
            self.assertGreater(metadata["size"], 0)
        self.assertEqual(bridge.PLUS_GITHUB_PROMPT_SIZE, 4_195_504)
        self.assertEqual(len(bridge.PLUS_GITHUB_PROMPT_BLOB_SHA1), 40)

    def test_full_plus_uses_vae_and_not_tcdecoder(self):
        bridge = FlashVSRBridge({"flashvsr_mode": FlashVSRBridge.MODE_FULL}, _FilesLocator())
        definition = bridge.query_download_def(
            enabled_only=False,
            model=FlashVSRBridge.MODEL_PLUS_TURBO,
            variant="full",
        )
        self.assertEqual(definition["fileNameMap"][bridge.PLUS_VAE_FILENAME], "Wan2.1_VAE.pth")
        self.assertNotIn(bridge.PLUS_TCDECODER_FILENAME, definition["fileNameMap"])

    def test_plus_resolved_paths_match_target_folder_and_turbo_subfolder(self):
        class RecordingLocator:
            def __init__(self, root):
                self.root = Path(root)
                self.lookups = []

            def locate_file(self, name, error_if_none=True, **_kwargs):
                self.lookups.append(name)
                return str(self.root / name)

        with tempfile.TemporaryDirectory() as temp_dir:
            locator = RecordingLocator(temp_dir)
            bridge = FlashVSRBridge({"flashvsr_mode": FlashVSRBridge.MODE_TINY}, locator)
            paths = bridge.paths("tiny-long", model=FlashVSRBridge.MODEL_PLUS_TURBO)

        expected = [
            os.path.join("FlashVSR", "turbo", bridge.PLUS_TRANSFORMER_FILENAME),
            os.path.join("FlashVSR", "turbo", bridge.PLUS_LQ_PROJ_FILENAME),
            os.path.join("FlashVSR", "turbo", bridge.PLUS_POSI_PROMPT_FILENAME),
            os.path.join("FlashVSR", bridge.PLUS_TCDECODER_FILENAME),
        ]
        self.assertEqual(locator.lookups, expected)
        self.assertTrue(paths.transformer.endswith(expected[0]))
        self.assertTrue(paths.lq_proj.endswith(expected[1]))
        self.assertTrue(paths.posi_prompt.endswith(expected[2]))
        self.assertTrue(paths.tcdecoder.endswith(expected[3]))

    def test_cached_plus_files_are_checked_against_sha256(self):
        bridge = FlashVSRBridge({"flashvsr_mode": FlashVSRBridge.MODE_TINY}, _FilesLocator())
        original_metadata = FlashVSRBridge.PLUS_ASSET_METADATA
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "asset.ckpt"
            path.write_bytes(b"abc")
            FlashVSRBridge.PLUS_ASSET_METADATA = {
                **original_metadata,
                "test.ckpt": {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()},
            }
            try:
                bridge._verify_plus_hf_asset(str(path), "test.ckpt")
                path.write_bytes(b"abd")
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    bridge._verify_plus_hf_asset(str(path), "test.ckpt")
            finally:
                FlashVSRBridge.PLUS_ASSET_METADATA = original_metadata


class FlashVSRRuntimeContractTests(unittest.TestCase):
    def test_v11_cache_and_startup_contract_is_preserved(self):
        self.assertEqual(runtime.continue_cache_frames(runtime.FLASHVSR_MODEL_V11), 11)
        self.assertEqual(runtime._next_conditioning_frame_count(1), 25)
        self.assertEqual(runtime.FLASHVSR_KV_CACHE_WINDOWS, 1)

    def test_plus_schedule_chunks_and_repeated_tail_conditioning(self):
        still = runtime._plus_schedule(1)
        self.assertEqual(still.latent_frames, 4)
        self.assertEqual(still.still_frames, 13)
        self.assertEqual(still.steps[0].latents, (0, 4))
        self.assertEqual(still.steps[0].lq_clips, [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17)])
        self.assertEqual(still.steps[0].lq_end, 13)

        video = runtime._plus_schedule(17)
        self.assertEqual(video.latent_frames, 8)
        self.assertEqual([step.latents for step in video.steps], [(0, 4), (4, 8)])
        self.assertEqual(video.steps[0].lq_end, 13)
        self.assertEqual(video.steps[1].lq_clips, [(17, 21), (21, 25), (25, 29), (29, 33)])
        self.assertEqual(video.steps[1].lq_end, 29)
        self.assertEqual([step.rope_start for step in video.steps], [0, 4])
        self.assertEqual(runtime.continue_cache_frames(runtime.FLASHVSR_MODEL_PLUS_TURBO), 19)

    def test_turbo_uses_exact_25_local_and_five_draft_layers(self):
        layers = runtime.FLASHVSR_PLUS_LOCAL_LAYERS[runtime.FLASHVSR_MODEL_PLUS_TURBO]
        self.assertEqual(len(layers), 25)
        self.assertEqual(layers, frozenset(i for i in range(30) if i % 6))

        dit = SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=SimpleNamespace(num_heads=2)) for _ in range(30)]
        )
        configs = runtime._plus_layer_configs(
            dit, runtime.FLASHVSR_MODEL_PLUS_TURBO, 4, 4, 2, 0, 3, 100_000, torch.device("cpu")
        )
        self.assertEqual(len(configs), 30)
        self.assertEqual(configs[0]["local_range"], runtime.FLASHVSR_PLUS_LOCAL_RANGE)
        self.assertEqual(configs[0]["kv_len"], 3)
        self.assertEqual(configs[0]["topk_groups"], 1)
        self.assertIsNone(configs[0]["block_mask"])
        self.assertEqual(configs[1]["local_range"], 3)
        self.assertEqual(configs[1]["kv_len"], 1)
        self.assertEqual(tuple(configs[1]["block_mask"].shape), (1, 2, 32, 64))

    def test_plus_fixed_topk_ignores_v11_slider_preference(self):
        self.assertEqual(
            runtime._select_topk_ratio(runtime.FLASHVSR_MODEL_PLUS_TURBO, 1.25, "tiny-long", 128, 128),
            runtime.FLASHVSR_PLUS_TOPK_RATIO,
        )
        self.assertEqual(
            runtime._select_topk_ratio(runtime.FLASHVSR_MODEL_V11, 1.25, "tiny-long", 128, 128),
            1.25,
        )

    def test_v11_auto_topk_ratio_covers_tiny_variants_and_preserves_full_floor(self):
        for variant in (
            runtime.FLASHVSR_VARIANT_TINY,
            runtime.FLASHVSR_VARIANT_TINY_LONG,
            runtime.FLASHVSR_VARIANT_FULL,
        ):
            with self.subTest(variant=variant):
                self.assertEqual(
                    runtime._select_topk_ratio(runtime.FLASHVSR_MODEL_V11, 0.0, variant, 512, 512),
                    2.0,
                )
                self.assertEqual(
                    runtime._select_topk_ratio(runtime.FLASHVSR_MODEL_V11, 0.0, variant, 1536, 2560),
                    runtime.FLASHVSR_FULL_MIN_AUTO_TOPK_RATIO,
                )

    def test_tiny_long_default_ratio_produces_nonempty_draft_attention_masks(self):
        ratio = runtime._select_topk_ratio(
            runtime.FLASHVSR_MODEL_V11, 0.0, runtime.FLASHVSR_VARIANT_TINY_LONG, 512, 512
        )
        window_size = 16
        topk = int(window_size * window_size * ratio) - 1
        self.assertGreater(topk, 0)

        torch.manual_seed(11)
        local = torch.ones((2, 2), dtype=torch.bool)
        q = torch.randn(4, 128, 8)
        k = torch.randn(8, 128, 8)
        dense_mask = generate_draft_block_mask(
            1, 2, 2, [q.clone(), k.clone()], topk=topk, local_attn_mask=local, groups=1
        )
        sage_mask = generate_draft_block_mask_sage(
            1, 2, 2, [q.clone(), k.clone()], topk=topk, local_attn_mask=local, groups=1
        )
        self.assertTrue(torch.count_nonzero(dense_mask).item() > 0)
        self.assertTrue(torch.count_nonzero(sage_mask).item() > 0)

    def test_v11_auto_topk_ratio_still_honors_explicit_value(self):
        self.assertEqual(
            runtime._select_topk_ratio(
                runtime.FLASHVSR_MODEL_V11, 1.25, runtime.FLASHVSR_VARIANT_TINY_LONG, 1536, 2560
            ),
            1.25,
        )

    def test_plus_official_head_projection_aliases_are_normalized_without_copying(self):
        weight = torch.randn(64, 1536)
        bias = torch.randn(64)
        normalized = runtime._preprocess_transformer_state_dict(
            {"head.proj.weight": weight, "head.proj.bias": bias}
        )
        self.assertEqual(tuple(normalized["head.head.weight"].shape), (64, 1536))
        self.assertEqual(tuple(normalized["head.head.bias"].shape), (64,))
        self.assertIs(normalized["head.head.weight"], weight)
        self.assertIs(normalized["head.head.bias"], bias)
        self.assertTrue(torch.equal(normalized["head.head.weight"], weight))
        self.assertTrue(torch.equal(normalized["head.head.bias"], bias))
        self.assertNotIn("head.proj.weight", normalized)
        self.assertNotIn("head.proj.bias", normalized)

    def test_transformer_head_aliases_are_noop_for_legacy_names_and_ignore_modulation(self):
        head = torch.randn(64, 1536)
        bias = torch.randn(64)
        modulation = torch.randn(1, 2, 1536)
        unrelated = torch.randn(64, 1536)
        normalized = runtime._preprocess_transformer_state_dict(
            {
                "head.head.weight": head,
                "head.head.bias": bias,
                "head.modulation": modulation,
                "model.head.proj.weight": unrelated,
            }
        )
        self.assertIs(normalized["head.head.weight"], head)
        self.assertIs(normalized["head.head.bias"], bias)
        self.assertIs(normalized["head.modulation"], modulation)
        self.assertIs(normalized["model.head.proj.weight"], unrelated)

    def test_transformer_head_alias_conflicts_are_rejected(self):
        aliases = (
            ("head.proj.weight", "head.head.weight", torch.randn(64, 1536)),
            ("head.proj.bias", "head.head.bias", torch.randn(64)),
        )
        for source_name, target_name, value in aliases:
            with self.subTest(source=source_name), self.assertRaisesRegex(
                ValueError, f"both {source_name} and {target_name}"
            ):
                runtime._preprocess_transformer_state_dict(
                    {source_name: value, target_name: torch.randn_like(value)}
                )

    def test_structural_mask_has_two_half_windows_per_key_window(self):
        candidates, count, entries = structural_block_candidates(2, 2, 2, 1, 3)
        self.assertEqual(tuple(candidates.shape), (8, 12))
        self.assertEqual(count, 96)
        self.assertEqual(entries, 96)

        mask = structural_block_mask(2, 2, 2, 1, 3, 2, torch.device("cpu"))
        self.assertEqual(tuple(mask.shape), (1, 2, 8, 24))
        self.assertEqual(mask.dtype, torch.int8)
        self.assertTrue(torch.equal(mask[:, :, :, 0::2], mask[:, :, :, 1::2]))

    def test_sage_draft_mask_supports_grouped_and_legacy_v11_contracts(self):
        torch.manual_seed(7)
        q = torch.randn(4, 128, 8)
        k = torch.randn(8, 128, 8)
        local = torch.ones((2, 2), dtype=torch.bool)
        plus = generate_draft_block_mask_sage(
            1, 2, 2, [q.clone(), k.clone()], topk=3, local_attn_mask=local, groups=1
        )
        v11 = generate_draft_block_mask_sage(
            1, 2, 2, [q.clone(), k.clone()], topk=3, local_attn_mask=local
        )
        dense = generate_draft_block_mask(
            1, 2, 2, [q.clone(), k.clone()], topk=3, local_attn_mask=local, groups=1
        )
        self.assertEqual(tuple(plus.shape), (1, 2, 4, 16))
        self.assertEqual(tuple(v11.shape), (1, 2, 4, 16))
        self.assertEqual(tuple(dense.shape), (1, 2, 4, 8))
        self.assertEqual(plus.dtype, torch.int8)
        self.assertEqual(v11.dtype, torch.int8)

    def test_prompt_loading_keeps_v11_safetensors_and_safely_loads_plus_tensor(self):
        from safetensors.torch import save_file

        context = torch.ones((1, 512, 4096), dtype=torch.float16)
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = Path(temp_dir) / "legacy.safetensors"
            plus_path = Path(temp_dir) / "plus.pth"
            save_file({"context": context}, str(legacy_path))
            torch.save(context, plus_path)

            legacy = runtime._load_prompt_context(str(legacy_path), runtime.FLASHVSR_MODEL_V11, torch.bfloat16)
            plus = runtime._load_prompt_context(str(plus_path), runtime.FLASHVSR_MODEL_PLUS_TURBO, torch.bfloat16)

        self.assertEqual(legacy.dtype, torch.bfloat16)
        self.assertEqual(plus.dtype, torch.bfloat16)
        self.assertEqual(tuple(plus.shape), (1, 512, 4096))

    def test_pickle_checkpoint_loader_is_weights_only_and_tensor_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights.ckpt"
            torch.save({"weight": torch.ones(2, 3)}, path)
            state_dict = runtime._load_safe_checkpoint_state_dict(str(path), "test checkpoint")
            self.assertEqual(tuple(state_dict["weight"].shape), (2, 3))

            torch.save({"step": 1}, path)
            with self.assertRaisesRegex(ValueError, "non-tensor"):
                runtime._load_safe_checkpoint_state_dict(str(path), "test checkpoint")

    def test_two_chunk_plus_upscale_trims_output_reuses_cache_and_cancels(self):
        class FakeProjection:
            def __init__(self):
                self.stream_calls = 0

            def clear_cache(self):
                pass

            def stream_forward(self, _chunk):
                self.stream_calls += 1
                return torch.tensor([self.stream_calls])

        class FakeDecoder:
            def clean_mem(self):
                pass

        def make_runtime():
            instance = runtime.FlashVSRRuntime()
            instance.device = torch.device("cpu")
            instance.dtype = torch.float32
            instance.model = runtime.FLASHVSR_MODEL_PLUS_TURBO
            instance.variant = runtime.FLASHVSR_VARIANT_TINY_LONG
            instance.dit = SimpleNamespace(blocks=[object() for _ in range(30)], clear_cross_kv=lambda: None)
            instance.lq_proj = FakeProjection()
            instance.tcdecoder = FakeDecoder()
            instance._prepare_run_state = lambda: None
            decode_calls = []

            def decode(_self, latents, _sample, lq_start, lq_end, _out_h, _out_w, padded_h, padded_w, _tile, _mems, **_kwargs):
                chunk_number = len(decode_calls)
                frame_count = 13 if chunk_number % 2 == 0 else 16
                value = -0.5 if chunk_number % 2 == 0 else 0.5
                decode_calls.append((int(latents.shape[2]), lq_start, lq_end, frame_count))
                frames = torch.full((1, 3, frame_count, padded_h, padded_w), value, dtype=torch.float32)
                return frames, None

            instance._decode_tcdecoder = decode.__get__(instance, runtime.FlashVSRRuntime)
            return instance, decode_calls

        def fake_denoise(_dit, latents, _context, _lq, cache_k, cache_v, _index, *_args, **kwargs):
            denoise_ratios.append(kwargs["topk_ratio"])
            return torch.zeros_like(latents), cache_k, cache_v

        sample = torch.zeros((3, 17, 3, 5), dtype=torch.float32)
        instance, decode_calls = make_runtime()
        denoise_ratios = []
        with patch.object(runtime, "_denoise_stream_chunk", fake_denoise), \
             patch.object(runtime, "_maybe_empty_cache"), \
             patch.object(runtime, "_wavelet_color_fix_from_sample"):
            frames, cache = instance.upscale(
                sample, 2.0, persistent_models=True, return_continue_cache=True, topk_ratio=1.25
            )
            continued, continued_cache = instance.upscale(
                sample, 2.0, continue_cache=cache, persistent_models=True,
                return_continue_cache=True, topk_ratio=4.0,
            )

            self.assertEqual(tuple(frames.shape), (3, 17, 6, 10))
            self.assertTrue(torch.all(frames[:, :13] == 64))
            self.assertTrue(torch.all(frames[:, 13:] == 191))
            self.assertEqual(cache["tail_frames"].shape[1], 17)
            self.assertTrue(torch.equal(continued, frames))
            self.assertTrue(torch.equal(continued_cache["tail_frames"], cache["tail_frames"]))
            self.assertEqual(decode_calls[:2], [(4, 0, 13, 13), (4, 13, 29, 16)])
            self.assertEqual(denoise_ratios, [2.5, 2.5, 2.5, 2.5])

            cancelled, cancelled_cache = None, None
            cancellation_runtime, cancellation_decodes = make_runtime()
            abort_checks = [0]

            def abort_after_two_projection_inputs():
                abort_checks[0] += 1
                return abort_checks[0] >= 4

            cancelled, cancelled_cache = cancellation_runtime.upscale(
                sample, 2.0, persistent_models=False,
                abort_callback=abort_after_two_projection_inputs,
            )
            self.assertIsNone(cancelled)
            self.assertIsNone(cancelled_cache)
            self.assertEqual(cancellation_runtime.lq_proj, None)
            self.assertEqual(cancellation_decodes, [])


if __name__ == "__main__":
    unittest.main()
