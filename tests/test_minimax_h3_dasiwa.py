"""DaSiWa model identity, conditioning, and baked acceleration boundaries."""

from __future__ import annotations

import ast
import copy
from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.minimax_h3.dasiwa import (  # noqa: E402
    normalize_dasiwa_turbo_request, normalize_dasiwa_turbo_steps,
    validate_dasiwa_checkpoint,
)
from models.minimax_h3.minimax_h3_handler import family_handler  # noqa: E402
from models.minimax_h3.turbo import (  # noqa: E402
    h3_scheduler_grid_points, minimax_h3_adapter_workflow,
    normalize_minimax_h3_turbo_request,
)


def preset(refs=False, turbo=False):
    name = f"minimax_h3_{'ref2va_' if refs else ''}dasiwa{'_turbo' if turbo else ''}"
    return json.loads((ROOT / "app/defaults" / f"{name}.json").read_text())


def definition(refs=False, turbo=False):
    model = preset(refs, turbo)["model"]
    return {**family_handler.query_model_def(model["architecture"], model), **model}


def production_function(filename, name, namespace=None):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {"__package__": "models.minimax_h3", **(namespace or {})}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), scope)
    return scope[name]


class DasiwaPresetTests(unittest.TestCase):
    def test_pairs_share_the_exact_creator_file_without_stock_aliases(self):
        for turbo in (False, True):
            frames, refs = preset(turbo=turbo), preset(refs=True, turbo=turbo)
            for key in ("URLs", "download_sources", "source_revision", "source_sha256"):
                self.assertEqual(frames["model"][key], refs["model"][key])
            model = frames["model"]
            filename = model["URLs"][0]
            self.assertNotIn("?", filename)
            self.assertEqual(model["download_sources"][filename]["sha256"], model["source_sha256"])
            self.assertIn("fileId=3263048" if turbo else "fileId=3263052",
                          model["download_sources"][filename]["url"])
            self.assertEqual(model["source_size_bytes"], 20967668632 if turbo else 20967668624)
            if turbo:
                self.assertEqual(frames["override_attention"], "")
                self.assertEqual(refs["override_attention"], "")
            self.assertEqual(frames["num_inference_steps"], 8 if turbo else 25)
            for reference in (False, True):
                md = definition(reference, turbo)
                self.assertEqual(md.get("omni_reference", False), reference)
                self.assertEqual(md["custom_frames_injection"], not reference)
                self.assertEqual(md["compatible_model_paths"], {})
                self.assertEqual(md["minimax_h3_qkv_layout"], "grouped")
                self.assertEqual(minimax_h3_adapter_workflow(md), "ref2va")
                self.assertTrue(md["returns_audio"])

    def test_baked_recipe_has_no_managed_turbo_and_opt_in_sol_attention(self):
        option = production_function("app/launch.py", "_minimax_h3_turbo_option")
        for refs in (False, True):
            md = definition(refs, True)
            self.assertIsNone(option(md))
            self.assertTrue(md["sol_attention"])
            self.assertFalse(md["sla_attention"])
            self.assertFalse(md["sla_attention_default"])
            self.assertFalse(md["first_block_cache"])
            self.assertEqual((md["inference_steps_min"], md["inference_steps_max"]), (4, 8))
            self.assertNotIn("audio_refinement", [item["id"] for item in md["custom_settings"]])
            standard = option(definition(refs, False))
            self.assertTrue(all(item["workflow"] in ("all", "ref2va") for item in standard["presets"]))

    def test_full_or_other_h3_recipes_cannot_masquerade_as_dasiwa(self):
        for architecture in ("minimax_h3_full", "minimax_h3_ref2va_full", "minimax_h3_voice_audio"):
            with self.assertRaises(ValueError):
                family_handler.query_model_def(architecture, preset()["model"])
        with self.assertRaises(ValueError):
            family_handler.query_model_def("minimax_h3", {**preset()["model"], "minimax_h3_fused_turbo": True})


class DasiwaRuntimeTests(unittest.TestCase):
    checkpoint = {"compressed_modulation": True, "adaln_curve_grid": 1025,
        "time_embed_dim": 8, "convrot": True, "quantization_format": "int8_tensorwise",
        "convrot_group_size": 256}

    def test_quantization_and_curve_mismatches_fail_before_load(self):
        validate_dasiwa_checkpoint(self.checkpoint, "grouped")
        for key, wrong in (("convrot", False), ("convrot_group_size", 128),
                           ("time_embed_dim", 2688), ("adaln_curve_grid", 1000),
                           ("quantization_format", "int4")):
            with self.assertRaises(ValueError):
                validate_dasiwa_checkpoint({**self.checkpoint, key: wrong}, "grouped")
        with self.assertRaises(ValueError):
            validate_dasiwa_checkpoint(self.checkpoint, "interleaved")

    def test_real_transformer_loader_reads_descriptors_and_uses_ref2va_basis(self):
        transformer = Mock()
        transformer.eval.return_value.requires_grad_.return_value = transformer
        offload = Mock()
        descriptors = Mock(return_value=self.checkpoint)
        loader = production_function("app/models/minimax_h3/minimax_h3_main.py", "_load_transformer", {
            "probe_h3_checkpoint": lambda _: {**self.checkpoint, "convrot_group_size": None},
            "convrot_quantization_info_from_file": descriptors,
            "_first_path": lambda value: value,
            "init_empty_weights": lambda **_: nullcontext(),
            "MiniMaxH3Transformer": lambda **_: transformer,
            "get_linear_split_map": lambda *_, **__: {"qkv": "split"},
            "offload": offload, "partial": partial, "_strip_transformer_wrappers": lambda *_, **__: None,
        })
        result = loader("dasiwa.safetensors", "bf16", qkv_layout="grouped", dasiwa=True)
        descriptors.assert_called_once_with("dasiwa.safetensors")
        self.assertEqual(result.h3_lora_model_type, "minimax_h3_ref2va")
        offload.load_model_data.assert_called_once()
        offload.split_linear_modules.assert_called_once()

    def test_four_and_eight_ui_steps_equal_model_evaluations(self):
        for count in (4, 8):
            self.assertEqual(h3_scheduler_grid_points(count, turbo_active=False) - 1, count)
            self.assertEqual(normalize_dasiwa_turbo_steps(count), count)
        self.assertEqual(normalize_dasiwa_turbo_steps(None), 8)
        for count in (True, 3, 9, 4.5, "invalid", float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                normalize_dasiwa_turbo_steps(count)

    def test_request_preserves_ordinary_loras_and_clears_hidden_recipes(self):
        body = {"num_inference_steps": 4, "activated_loras": ["character.safetensors"],
                "loras_multipliers": "0.7", "minimax_h3_turbo_mode": True,
                "override_attention": "sla", "skip_steps_cache_type": "first_block_cache"}
        applied = normalize_minimax_h3_turbo_request(body, full_checkpoint=False,
            model_def=definition(turbo=True), workflow="ref2va")
        self.assertFalse(applied)
        self.assertEqual(body["num_inference_steps"], 4)
        self.assertEqual(body["activated_loras"], ["character.safetensors"])
        self.assertEqual(body["loras_multipliers"], "0.7")
        self.assertFalse(body["minimax_h3_turbo_mode"])
        self.assertEqual(body["override_attention"], "")
        self.assertEqual(body["skip_steps_cache_type"], "")
        self.assertEqual((body["flow_shift"], body["audio_flow_shift"]), (9, 4))
        for lora in ("new_turbo.safetensors", "PDD_accelerator.safetensors", "my_vdn.safetensors"):
            with self.assertRaisesRegex(ValueError, "cannot use"):
                normalize_dasiwa_turbo_request({"activated_loras": [lora]})
        with self.assertRaisesRegex(ValueError, "Audio refinement"):
            normalize_dasiwa_turbo_request({"custom_settings": {"audio_refinement": "enabled"}})

    def test_request_preserves_dense_auto_and_opt_in_sol_attention(self):
        for attention in ("", "auto", "sdpa", "sage", "sage2", "flash", "sol"):
            with self.subTest(attention=attention):
                body = {"num_inference_steps": 6, "override_attention": attention,
                        "skip_steps_cache_type": "first_block"}
                normalize_dasiwa_turbo_request(body)
                self.assertEqual(body["override_attention"], attention)
                self.assertEqual(body["num_inference_steps"], 6)
                self.assertEqual(body["skip_steps_cache_type"], "")
                self.assertFalse(body["minimax_h3_turbo_mode"])
                self.assertEqual(body["minimax_h3_turbo_preset"], "")

    def test_settings_migration_preserves_backend_and_clears_stale_sla(self):
        for attention, expected in (("", ""), ("auto", "auto"), ("sdpa", "sdpa"),
                                    ("sage", "sage"), ("sage2", "sage2"),
                                    ("flash", "flash"), ("sol", "sol"), ("sla", "")):
            with self.subTest(attention=attention):
                settings = copy.deepcopy(preset(turbo=True))
                settings["num_inference_steps"] = 6
                settings["override_attention"] = attention
                settings["skip_steps_cache_type"] = "first_block"
                family_handler.fix_settings("minimax_h3", 2.58, definition(turbo=True), settings)
                self.assertEqual(settings["override_attention"], expected)
                self.assertEqual(settings["num_inference_steps"], 6)
                self.assertEqual(settings["skip_steps_cache_type"], "")

        stale = copy.deepcopy(preset(turbo=True))
        stale["num_inference_steps"] = 9
        stale["override_attention"] = "sol"
        family_handler.fix_settings("minimax_h3", 2.58, definition(turbo=True), stale)
        self.assertEqual(stale["num_inference_steps"], 8)
        self.assertEqual(stale["override_attention"], "sol")

    def test_generation_validation_keeps_opt_in_attention_without_extra_acceleration(self):
        for attention, expected in (("", ""), ("auto", "auto"), ("sdpa", "sdpa"),
                                    ("sol", "sol"), ("sla", "")):
            with self.subTest(attention=attention):
                request = {
                    "num_inference_steps": 6,
                    "override_attention": attention,
                    "minimax_h3_turbo_mode": True,
                    "minimax_h3_turbo_preset": "legacy",
                    "skip_steps_cache_type": "first_block",
                    "video_length": 65,
                    "sliding_window_size": 65,
                    "resolution": "1280x704",
                }
                error = family_handler.validate_generative_settings(
                    "minimax_h3", definition(turbo=True), request
                )
                self.assertIsNone(error)
                self.assertEqual(request["override_attention"], expected)
                self.assertEqual(request["num_inference_steps"], 6)
                self.assertFalse(request["minimax_h3_turbo_mode"])
                self.assertEqual(request["minimax_h3_turbo_preset"], "")
                self.assertEqual(request["skip_steps_cache_type"], "")


if __name__ == "__main__":
    unittest.main()
