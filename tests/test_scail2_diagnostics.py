"""CPU-only tests for privacy-safe SCAIL-2 performance diagnostics."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.scail2_diagnostics import (  # noqa: E402
    LOW_AVAILABLE_HOST_RAM_WARNING,
    _checkpoint_quantization_markers,
    build_scail2_performance_diagnostic,
    collect_scail2_performance_diagnostic,
    format_scail2_performance_summary,
)


def diagnostic(**overrides):
    values = {
        "architecture": "scail2_14B",
        "model_type": "scail2_14B_recast_fast",
        "params": {
            "resolution": "832x480",
            "video_length": 125,
            "sliding_window_size": 81,
            "num_inference_steps": 8,
            "guidance_scale": 1,
        },
        "transformer_checkpoint": "C:\\private\\models\\scail2_14B_quanto_mbf16_int8.safetensors",
        "text_encoder_checkpoint": "C:\\private\\models\\umt5-xxl-quanto_int8.safetensors",
        "transformer_quantization": "int8",
        "text_encoder_quantization": "int8",
        "total_ram_gb": 31.8,
        "available_ram_gb": 20,
        "own_pinned_gb": 4,
        "profile": 4,
        "transformer_budget_mb": 1200,
        "attention_head_split": 0,
        "attention_head_split_auto_override": False,
        "reserved_ram_fraction": 0.4,
        "gpu_name": "NVIDIA GeForce RTX 4080",
        "gpu_vram_gb": 16,
        "smart_memory_pinning": True,
        "ram_allocator_mode": "default",
        "ram_allocator_active": "mmgp",
    }
    values.update(overrides)
    return build_scail2_performance_diagnostic(**values)


class Scail2DiagnosticTests(unittest.TestCase):
    def test_reports_effective_workload_and_only_checkpoint_basenames(self):
        result = diagnostic()
        self.assertEqual(result["summary"]["steps"], 8)
        self.assertEqual(result["summary"]["guidance_scale"], 1)
        self.assertEqual(result["summary"]["sliding_window_frames"], 81)
        self.assertEqual(
            result["summary"]["selected_transformer_checkpoint"],
            "scail2_14B_quanto_mbf16_int8.safetensors",
        )
        self.assertNotIn("private", repr(result))
        self.assertNotIn("C:\\", repr(result))

    def test_user_supplied_model_and_resolution_strings_cannot_leak_paths(self):
        result = diagnostic(
            model_type="C:\\private\\clip.mp4",
            params={"resolution": "C:\\private\\media.mp4"},
        )
        self.assertEqual(result["summary"]["model_type"], "custom")
        self.assertEqual(result["summary"]["resolution"], "unknown")
        self.assertNotIn("private", repr(result))

    def test_warns_for_unquantized_bf16_checkpoint_on_32gb_host(self):
        result = diagnostic(
            transformer_checkpoint="scail2_14B_mbf16.safetensors",
            text_encoder_checkpoint="umt5-xxl-enc-bf16.safetensors",
        )
        self.assertEqual(
            result["summary"]["transformer_unquantized_bf16"], True,
        )
        self.assertEqual(
            result["summary"]["text_encoder_unquantized_bf16"], True,
        )
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("32GB-class RAM", result["warnings"][0])
        self.assertIn("supported INT8", result["warnings"][0])

    def test_warns_for_unquantized_bf16_checkpoint_on_lower_ram_host(self):
        result = diagnostic(
            transformer_checkpoint="scail2_14B_mbf16.safetensors",
            total_ram_gb=31.0,
        )
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("limited-RAM system (31 GiB)", result["warnings"][0])

    def test_bf16_capacity_warning_stops_at_40_gib_but_pressure_still_warns(self):
        result = diagnostic(
            transformer_checkpoint="scail2_14B_mbf16.safetensors",
            total_ram_gb=40,
        )
        self.assertEqual(result["warnings"], [])
        pressured = diagnostic(
            transformer_checkpoint="scail2_14B_mbf16.safetensors",
            total_ram_gb=40,
            available_ram_gb=6,
            own_pinned_gb=0,
        )
        self.assertIn(LOW_AVAILABLE_HOST_RAM_WARNING, pressured["warnings"])

    def test_does_not_misread_quantized_bf16_storage_as_bf16_checkpoint(self):
        result = diagnostic(
            transformer_checkpoint="scail2_14B_quanto_mbf16_int8.safetensors",
            text_encoder_checkpoint="umt5-xxl-enc-quanto_int8.safetensors",
        )
        self.assertEqual(
            result["summary"]["transformer_unquantized_bf16"], False,
        )
        self.assertEqual(result["warnings"], [])

    def test_quantized_filename_markers_cover_repository_checkpoint_formats(self):
        # These basenames are declared by app/defaults/scail2_14B.json,
        # app/models/wan/wan_handler.py, app/defaults/minimax_h3.json, and
        # app/models/ltx25/ltx25_handler.py respectively.
        self.assertEqual(
            _checkpoint_quantization_markers(
                "scail2_14B_quanto_mbf16_int8.safetensors",
            ),
            ["int8", "quanto"],
        )
        self.assertEqual(
            _checkpoint_quantization_markers(
                "models_t5_umt5-xxl-enc-quanto_int8.safetensors",
            ),
            ["int8", "quanto"],
        )
        self.assertEqual(
            _checkpoint_quantization_markers(
                "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
            ),
            ["fp8"],
        )
        self.assertEqual(
            _checkpoint_quantization_markers(
                "ltx-2.5-22b_video_embeddings_connector_nvfp4_bf16.safetensors",
            ),
            ["fp4", "nvfp4"],
        )

    def test_bf16_warning_is_limited_to_32gb_class(self):
        result = diagnostic(
            transformer_checkpoint="scail2_14B_mbf16.safetensors",
            total_ram_gb=63.7,
        )
        self.assertEqual(result["warnings"], [])

    def test_reports_measured_low_host_ram_spare(self):
        result = diagnostic(available_ram_gb=6, own_pinned_gb=0)
        self.assertLess(result["summary"]["estimated_host_spare_gb"], 8)
        self.assertEqual(result["warnings"], [LOW_AVAILABLE_HOST_RAM_WARNING])

    def test_does_not_report_low_ram_when_available_and_reclaimable_are_sufficient(self):
        result = diagnostic(available_ram_gb=20, own_pinned_gb=4)
        self.assertNotIn(LOW_AVAILABLE_HOST_RAM_WARNING, result["warnings"])

    def test_ignores_non_scail_architectures(self):
        self.assertIsNone(diagnostic(architecture="wan_14B"))

    def test_log_line_includes_placement_but_no_user_paths(self):
        result = diagnostic(
            transformer_checkpoint="C:\\private\\scail2_14B_mbf16.safetensors",
            text_encoder_checkpoint="C:\\private\\umt5_bf16.safetensors",
        )
        line = format_scail2_performance_summary(result["summary"])
        self.assertIn("steps=8, CFG=1", line)
        self.assertIn("planned_profile=4", line)
        self.assertIn("transformer_budget_setting=1200 MB", line)
        self.assertIn("head_split=0 (saved)", line)
        self.assertIn("smart_pinning=True", line)
        self.assertIn("NVIDIA GeForce RTX 4080 (16.0 GiB)", line)
        self.assertIn("scail2_14B_mbf16.safetensors", line)
        self.assertIn("MMGP_pinned_before_load=4.0 GiB", line)
        self.assertNotIn("private", line)
        self.assertNotIn("C:\\", line)

    def test_zero_configured_transformer_budget_is_labeled_as_profile_default(self):
        result = diagnostic(transformer_budget_mb=0)
        summary = result["summary"]
        self.assertTrue(summary["transformer_budget_uses_profile_default"])
        self.assertEqual(summary["configured_transformer_budget_mb"], 0)
        self.assertIn("MMGP profile default", format_scail2_performance_summary(summary))

    def test_manual_preload_is_distinguished_from_transformer_budget(self):
        result = diagnostic(
            transformer_budget_mb=8500,
            preload_mode="manual",
            manual_preload_mb=8500,
        )
        self.assertEqual(
            result["summary"]["transformer_budget_setting_kind"],
            "manual preload",
        )
        line = format_scail2_performance_summary(result["summary"])
        self.assertIn("8500 MB (manual preload)", line)
        self.assertIn("preload_mode=manual", line)

    def test_manual_preload_is_labeled_manual_when_auto_tune_owns_profile(self):
        result = diagnostic(
            transformer_budget_mb=0,
            preload_mode="manual",
            manual_preload_mb=2816,
            plan_applied=True,
        )
        self.assertEqual(result['summary']['transformer_budget_setting_kind'], 'manual preload')
        self.assertEqual(result['summary']['configured_manual_preload_mb'], 2816)

    def test_collector_selects_registry_filenames_without_filesystem_or_load_calls(self):
        class FakeWgp:
            transformer_quantization = "int8"
            text_encoder_quantization = "int8"
            transformer_dtype_policy = ""

            def __init__(self):
                self.calls = []

            def get_model_filename(
                self, model_type, quantization, dtype_policy="", URLs=None,
            ):
                self.calls.append(("get_model_filename", quantization, URLs))
                if URLs is None:
                    if quantization == "int8":
                        return "https://models/scail2_14B_quanto_mbf16_int8.safetensors"
                    return "https://models/scail2_14B_mbf16.safetensors"
                if quantization == "int8":
                    return "https://models/models_t5_umt5-xxl-enc-quanto_int8.safetensors"
                return "https://models/models_t5_umt5-xxl-enc-bf16.safetensors"

            def get_model_recursive_prop(
                self, model_type, prop, *, return_list=False,
            ):
                self.calls.append(("get_model_recursive_prop", prop))
                return ["encoder-bf16-url", "encoder-quanto-int8-url"]

            def download_models(self, *args, **kwargs):
                raise AssertionError("diagnostics must not download checkpoints")

            def get_local_model_filename(self, *args, **kwargs):
                raise AssertionError("diagnostics must not locate local files")

            def load_model(self, *args, **kwargs):
                raise AssertionError("diagnostics must not load a model")

        fake_wgp = FakeWgp()
        result = collect_scail2_performance_diagnostic(
            wgp=fake_wgp,
            model_definition={"architecture": "scail2_14B"},
            params={
                "model_type": "scail2_14B_recast_fast",
                "num_inference_steps": 8,
                "guidance_scale": 1,
                "resolution": "832x480",
                "video_length": 125,
                "sliding_window_size": 81,
            },
            plan={
                "profile": 4,
                "transformer_budget_mb": 0,
                "attention_head_split": 0,
                "override_attention_head_split": False,
                "reserved_ram_fraction": 0.4,
            },
            hardware={
                "ram_gb": 31.8,
                "ram_available_gb": 18,
                "ram_allocator_active": "mmgp",
                "gpu_name": "RTX 4080",
                "gpu_vram_gb": 16,
            },
            live_stats={"ram": {"total_gb": 31.8, "available_gb": 18}},
            own_pinned_gb=2,
            settings={"smart_memory_pinning": True, "ram_allocator": "default"},
        )
        self.assertEqual(
            result["summary"]["selected_transformer_checkpoint"],
            "scail2_14B_quanto_mbf16_int8.safetensors",
        )
        self.assertEqual(
            result["summary"]["selected_text_encoder_checkpoint"],
            "models_t5_umt5-xxl-enc-quanto_int8.safetensors",
        )
        self.assertEqual(
            [call[0] for call in fake_wgp.calls],
            [
                "get_model_filename",
                "get_model_recursive_prop",
                "get_model_filename",
            ],
        )

    def test_collector_returns_none_for_non_scail_model_without_resolving_files(self):
        class NoCalls:
            def get_model_filename(self, *args, **kwargs):
                raise AssertionError("non-SCAIL jobs do not resolve files")

        self.assertIsNone(collect_scail2_performance_diagnostic(
            wgp=NoCalls(),
            model_definition={"architecture": "wan_14B"},
            params={"model_type": "t2v"},
            plan={},
            hardware={},
            live_stats={},
            own_pinned_gb=0,
            settings={},
        ))


if __name__ == "__main__":
    unittest.main()
