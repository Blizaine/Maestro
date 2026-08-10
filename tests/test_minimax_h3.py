"""Model-free and optional runtime regressions for MiniMax H3 support."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import typing
import unittest
from unittest import mock


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
_HANDLER_PATH = _APP / "models" / "minimax_h3" / "minimax_h3_handler.py"
_MAIN_PATH = _APP / "models" / "minimax_h3" / "minimax_h3_main.py"
_PACKING_PATH = _APP / "models" / "minimax_h3" / "packing.py"
_VIDEO_VAE_PATH = _APP / "models" / "minimax_h3" / "video_vae.py"
_REF2VA_PATH = _APP / "models" / "minimax_h3" / "ref2va.py"
_TRANSFORMER_PATH = _APP / "models" / "minimax_h3" / "transformer.py"
_CONDITIONER_PATH = _APP / "models" / "minimax_h3" / "conditioner.py"
_CHECKPOINT_PATH = _APP / "models" / "minimax_h3" / "checkpoint.py"
_TURBO_PATH = _APP / "models" / "minimax_h3" / "turbo.py"
_NVFP4_PATH = _APP / "shared" / "qtypes" / "nvfp4.py"
_INT8_CONVROT_PATH = _APP / "shared" / "qtypes" / "int8_convrot.py"
_WGP_PATH = _APP / "wgp.py"
_LAUNCH_PATH = _APP / "launch.py"
_LLM_SERVICE_PATH = _APP / "services" / "llm_service.py"
_DEFAULT_PATH = _APP / "defaults" / "minimax_h3.json"
_REF2VA_DEFAULT_PATH = _APP / "defaults" / "minimax_h3_ref2va.json"
_FULL_DEFAULT_PATH = _APP / "defaults" / "minimax_h3_full.json"
_REF2VA_FULL_DEFAULT_PATH = _APP / "defaults" / "minimax_h3_ref2va_full.json"
_STORE_PATH = _ROOT / "ui" / "src" / "stores" / "useStore.ts"
_PROMPT_INPUT_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "PromptInput.tsx"
_DURATION_SLIDER_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "DurationSlider.tsx"
_ADVANCED_SETTINGS_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "AdvancedSettings.tsx"
_INPUTS_PANEL_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "InputsPanel.tsx"
_TURBO_TOGGLE_PATH = (
    _ROOT / "ui" / "src" / "components" / "Sidebar" / "MiniMaxH3TurboToggle.tsx"
)
_SIDEBAR_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "Sidebar.tsx"
_TYPES_PATH = _ROOT / "ui" / "src" / "types" / "index.ts"
_OMNI_REFERENCE_SECTION_PATH = (
    _ROOT / "ui" / "src" / "components" / "Sidebar" / "OmniReferenceSection.tsx"
)
_H3_MULTI_WINDOW_CONTROLS_PATH = (
    _ROOT / "ui" / "src" / "components" / "Sidebar" / "H3MultiWindowControls.tsx"
)
_GENERATE_BUTTON_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "GenerateButton.tsx"
_RESOLUTION_PRESETS_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "ResolutionPresets.tsx"
_ASPECT_RATIO_GRID_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "AspectRatioGrid.tsx"
_MODEL_SELECTOR_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "ModelSelector.tsx"
_LORA_SELECTOR_PATH = _ROOT / "ui" / "src" / "components" / "SettingsDrawer" / "LoraSelector.tsx"
_ENHANCE_GUIDES_PATH = _APP / "services" / "enhance_guides.py"
_PROMPT_POLISH_PATH = _APP / "services" / "director" / "prompt_polish.py"
_H3_ENHANCE_GUIDE_PATH = _APP / "services" / "llm_guides" / "enhance" / "minimax_h3_video.md"
_H3_REF2VA_GUIDE_PATH = (
    _APP / "services" / "llm_guides" / "enhance" / "minimax_h3_ref2va_video.md"
)
_H3_DIALECT_GUIDE_PATH = _APP / "services" / "llm_guides" / "dialect" / "minimax_h3_video.md"
_H3_REF2VA_DIALECT_GUIDE_PATH = (
    _APP / "services" / "llm_guides" / "dialect" / "minimax_h3_ref2va_video.md"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_handler_class():
    tree = ast.parse(_read(_HANDLER_PATH), filename=str(_HANDLER_PATH))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(name.startswith("_") for name in names):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_hf_url",
            "_text_encoder_variants",
            "_recommend_text_encoder",
            "normalize_h3_overlap_frames",
            "pace_h3_sliding_window_prompt",
        }:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "family_handler":
            selected.append(node)
    namespace = {
        "os": os,
        "torch": types.SimpleNamespace(bfloat16="bfloat16"),
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_HANDLER_PATH), "exec"), namespace)
    return namespace["family_handler"]


def _load_source_function(path: Path, name: str, *, include_private_assignments: bool = False):
    tree = ast.parse(_read(path), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    body = []
    if include_private_assignments:
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(item.startswith("_") for item in names):
                body.append(node)
    body.append(function)
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _load_h3_memory_helpers():
    names = {
        "_normalize_h3_resolution",
        "_h3_resolution_pixels",
        "recommended_h3_window_profile",
        "recommended_h3_window_frames",
        "recommended_h3_omni_sequence_profile",
        "h3_runtime_preflight",
        "apply_h3_window_memory_policy",
        "apply_h3_omni_sequence_memory_policy",
        "apply_h3_native_omni_memory_policy",
        "normalize_h3_overlap_frames",
        "pace_h3_sliding_window_prompt",
    }
    tree = ast.parse(_read(_HANDLER_PATH), filename=str(_HANDLER_PATH))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assigned = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
            if any(name.startswith("_") for name in assigned):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
    namespace = {}
    module = ast.Module(body=selected, type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(_HANDLER_PATH), "exec"),
        namespace,
    )
    return namespace


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(_read(path), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Could not find literal assignment {name}")


def _load_minimax_h3_lora_routing_helpers():
    tree = ast.parse(_read(_LAUNCH_PATH), filename=str(_LAUNCH_PATH))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CIVIT_TO_LOCAL_ARCH"
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_is_minimax_h3_identity",
            "_civitai_lora_arch",
        }:
            selected.append(node)
    namespace = {}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_LAUNCH_PATH), "exec"), namespace)
    return namespace


def _load_frame_aligner():
    tree = ast.parse(_read(_WGP_PATH), filename=str(_WGP_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "align_model_frame_count"
    )
    namespace = {}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_WGP_PATH), "exec"), namespace)
    return namespace["align_model_frame_count"]


def _load_turbo_helpers():
    spec = importlib.util.spec_from_file_location("maestro_minimax_h3_turbo_test", _TURBO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_llm_enhance_helpers():
    tree = ast.parse(_read(_LLM_SERVICE_PATH), filename=str(_LLM_SERVICE_PATH))
    helper_names = {
        "_extract_h3_quoted_dialogue",
        "_h3_requests_speech",
        "_extract_h3_dialogue_blocks",
        "_h3_dialogue_schedule",
        "_build_h3_timed_silence_clause",
        "_build_h3_dialogue_requirement",
        "_h3_dialogue_contract_satisfied",
        "_h3_timed_silence_contract_satisfied",
        "_h3_voice_binding_contract_satisfied",
        "_has_complete_h3_ref2va_structure",
        "_has_complete_h3_context_structure",
        "_compile_h3_explicit_dialogue",
        "_inject_missing_h3_dialogue",
        "_inject_h3_generated_dialogue",
        "_strip_h3_untagged_dialogue_duplicates",
        "_enforce_h3_soundscape_silence",
        "_enforce_h3_music_request",
        "_build_h3_ref2va_tagged_fallback",
        "_build_h3_context_fallback",
        "_clean_enhance_output",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "_H3_REF2VA_FIELDS" in names or "_H3_CONTEXT_FIELDS" in names:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in helper_names:
            selected.append(node)
    namespace = {"Optional": typing.Optional}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_LLM_SERVICE_PATH), "exec"), namespace)
    return namespace


class TestMiniMaxH3Definition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = _load_handler_class()

    def test_default_model_is_pinned_and_consumer_friendly(self):
        defaults = json.loads(_DEFAULT_PATH.read_text(encoding="utf-8"))
        model = defaults["model"]
        self.assertEqual(model["architecture"], "minimax_h3")
        self.assertEqual(defaults["num_inference_steps"], 20)
        self.assertEqual(defaults["video_length"], 124)
        self.assertEqual(defaults["resolution"], "864x480")
        self.assertIn("minimax_h3_fl2va_pruned_fp8_scaled.safetensors", model["URLs"][0])
        self.assertIn("0543966fbdce5ba05709a8f2031c94bdba629b4a", model["URLs"][0])
        self.assertNotIn("minimax_h3_text_encoder", defaults)

    def test_handler_exposes_base_fl2va_contract(self):
        model_def = self.handler.query_model_def("minimax_h3", {})
        self.assertEqual(
            self.handler.query_supported_types(),
            [
                "minimax_h3",
                "minimax_h3_full",
                "minimax_h3_ref2va",
                "minimax_h3_ref2va_full",
            ],
        )
        self.assertEqual((model_def["fps"], model_def["frames_minimum"]), (24, 124))
        self.assertEqual((model_def["frames_steps"], model_def["frames_maximum"]), (17, 345))
        self.assertEqual(
            (model_def["frame_alignment_modulus"], model_def["frame_alignment_remainder"]),
            (17, 5),
        )
        self.assertEqual(model_def["image_prompt_types_allowed"], "TSE")
        self.assertTrue(model_def["end_frames_always_enabled"])
        self.assertTrue(model_def["t2v_class"])
        self.assertTrue(model_def["i2v_class"])
        self.assertTrue(model_def["custom_frames_injection"])
        self.assertTrue(model_def["returns_audio"])
        self.assertFalse(model_def["supports_reference_audio"])
        self.assertTrue(model_def["no_negative_prompt"])
        self.assertTrue(model_def["sliding_window"])
        self.assertTrue(model_def["video_continuation"])
        self.assertTrue(model_def["first_block_cache"])
        self.assertEqual(
            tuple(model_def["first_block_cache_thresholds"]),
            (0.06, 0.08, 0.10, 0.12, 0.14),
        )
        self.assertEqual(
            model_def["skip_steps_multiplier_label"],
            "First Block Cache Threshold",
        )
        self.assertTrue(model_def["sliding_window_exact_total_frames"])
        self.assertTrue(model_def["sliding_window_auto_prompt_pacing"])
        self.assertTrue(
            model_def["sliding_window_memory_policy"]["manual_override"]
        )
        self.assertEqual(
            model_def["sliding_window_memory_policy"]["checkpoint"],
            "pruned",
        )
        self.assertEqual(
            model_def["sliding_window_defaults"],
            {
                "window_min": 124,
                "window_max": 345,
                "window_step": 17,
                "window_default": 345,
                "overlap_min": 1,
                "overlap_max": 103,
                "overlap_step": 17,
                "overlap_offset": 1,
                "overlap_default": 18,
                "discard_last_frames": 0,
            },
        )
        self.assertTrue(model_def["audio_guide_window_slicing"])
        self.assertTrue(model_def["sliding_window_audio_history"])
        self.assertTrue(model_def["any_audio_prompt"])
        self.assertTrue(model_def["audio_prompt_choices"])
        self.assertTrue(model_def["output_audio_is_input_audio"])
        self.assertTrue(model_def["minimax_h3_media_sources"])
        self.assertTrue(model_def["video_to_video_inpaint"])
        self.assertEqual(
            model_def["mask_preprocessing"]["selection"],
            ["", "A", "NA"],
        )
        self.assertEqual(
            model_def["audio_prompt_type_sources"]["selection"],
            ["", "A", "K", "2"],
        )
        self.assertIn(
            ("Use Control Video", "GV"),
            model_def["guide_custom_choices"]["choices"],
        )
        self.assertEqual(model_def["director_video_strategy"], "bounded_start_end")
        self.assertEqual(model_def["director_shot_image_support"], "optional")
        self.assertEqual(model_def["director_audio_input_mode"], "none")
        self.assertIn("FIRST / LAST", model_def["selector_help"])
        self.assertIn("source-media workflows", model_def["selector_help"])
        self.assertIn("Control Video's audio", model_def["selector_help"])
        self.assertIn("converts Full adapters", model_def["lora_compatibility_note"])
        self.assertTrue(model_def["director_endpoint_continuity"])
        self.assertFalse(model_def["director_trim_end_frames"])
        self.assertIn("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", model_def["text_encoder_URLs"][0])
        self.assertEqual(model_def["minimax_h3_text_encoder_default"], "nvfp4_awq")
        self.assertEqual(
            set(model_def["minimax_h3_text_encoder_variants"]),
            {"nvfp4_awq", "gguf_q2_k", "gguf_q4_k_m", "int8", "bf16"},
        )

    def test_fl2va_media_source_validation_fails_before_generation(self):
        model_def = self.handler.query_model_def("minimax_h3", {})
        base = {
            "video_length": 124,
            "sliding_window_size": 124,
            "sliding_window_overlap": 18,
            "video_prompt_type": "",
            "audio_prompt_type": "",
            "video_guide": None,
            "audio_guide": None,
        }
        missing_control = {**base, "audio_prompt_type": "2"}
        self.assertIn(
            "requires a Control Video",
            self.handler.validate_generative_settings(
                "minimax_h3", model_def, missing_control
            ),
        )
        conflicting = {
            **base,
            "audio_prompt_type": "A2",
            "video_prompt_type": "GV",
            "video_guide": "control.mp4",
            "audio_guide": "soundtrack.wav",
        }
        self.assertIn(
            "cannot also use a source soundtrack",
            self.handler.validate_generative_settings(
                "minimax_h3", model_def, conflicting
            ),
        )
        missing_soundtrack = {**base, "audio_prompt_type": "A"}
        self.assertIn(
            "requires an uploaded Soundtrack",
            self.handler.validate_generative_settings(
                "minimax_h3", model_def, missing_soundtrack
            ),
        )

        missing_mask = {
            **base,
            "video_prompt_type": "GVA",
            "video_guide": "control.mp4",
        }
        self.assertIn(
            "requires a mask video",
            self.handler.validate_generative_settings(
                "minimax_h3", model_def, missing_mask
            ),
        )
        bad_strength = {
            **base,
            "video_prompt_type": "GV",
            "video_guide": "control.mp4",
            "denoising_strength": 1.1,
        }
        self.assertIn(
            "between 0 and 1",
            self.handler.validate_generative_settings(
                "minimax_h3", model_def, bad_strength
            ),
        )

    def test_text_encoder_recommendation_is_hardware_aware(self):
        model_def = self.handler.query_model_def("minimax_h3", {})
        recommend = self.handler.recommend_text_encoder
        self.assertEqual(
            recommend({"supports_nvfp4": True, "ram_gb": 32}, model_def),
            "nvfp4_awq",
        )
        self.assertEqual(
            recommend({"supports_nvfp4": False, "ram_gb": 64}, model_def),
            "nvfp4_awq",
        )
        self.assertEqual(
            recommend(
                {
                    "supports_nvfp4": False,
                    "ram_gb": 64,
                    "gpu_vram_gb": 16,
                },
                model_def,
            ),
            "gguf_q2_k",
        )
        self.assertEqual(
            recommend({"supports_nvfp4": False, "ram_gb": 32}, model_def),
            "nvfp4_awq",
        )
        self.assertEqual(
            recommend({"supports_nvfp4": False, "ram_gb": 16}, model_def),
            "gguf_q2_k",
        )

    def test_all_h3_variants_expose_native_portrait_and_auto_aspect(self):
        for model_type in self.handler.query_supported_types():
            model_def = self.handler.query_model_def(model_type, {})
            self.assertTrue(model_def["supports_auto_aspect"])
            self.assertEqual(
                model_def["resolution_preset_order"],
                ["480p", "540p", "720p", "1080p"],
            )
            presets = model_def["resolution_presets"]
            self.assertEqual(presets["480p"]["values"]["9:16"], "480x864")
            self.assertEqual(presets["540p"]["values"]["9:16"], "544x960")
            self.assertEqual(presets["720p"]["label"], "720p")
            self.assertEqual(presets["720p"]["values"]["16:9"], "1280x704")
            self.assertEqual(presets["720p"]["values"]["9:16"], "704x1280")
            self.assertEqual(presets["720p"]["values"]["auto"], "auto_720p")
            self.assertEqual(presets["768p"]["label"], "768p High")
            self.assertEqual(presets["768p"]["values"]["16:9"], "1344x768")
            self.assertEqual(presets["768p"]["values"]["auto"], "auto_768p")
            self.assertNotIn("768p", model_def["resolution_preset_order"])
            self.assertTrue(presets["1080p"]["experimental"])
            self.assertEqual(presets["1080p"]["label"], "1080p")
            self.assertEqual(presets["1080p"]["values"]["16:9"], "1920x1088")
            self.assertEqual(presets["1080p"]["values"]["9:16"], "1088x1920")
            self.assertEqual(presets["1080p"]["values"]["4:3"], "1440x1088")
            self.assertEqual(presets["1080p"]["values"]["3:4"], "1088x1440")

    def test_old_generic_resolutions_snap_without_changing_orientation(self):
        normalize = _load_source_function(
            _HANDLER_PATH,
            "_normalize_h3_resolution",
            include_private_assignments=True,
        )
        self.assertEqual(normalize("1280x720"), "1280x704")
        self.assertEqual(normalize("1280x704"), "1280x704")
        self.assertEqual(normalize("720x1280"), "704x1280")
        self.assertEqual(normalize("704x1280"), "704x1280")
        self.assertEqual(normalize("1104x832"), "1024x768")
        self.assertEqual(normalize("832x1104"), "768x1024")
        self.assertEqual(normalize("1024x1024"), "768x768")
        self.assertEqual(normalize("1920x1088"), "1920x1088")
        self.assertEqual(normalize("1088x1920"), "1088x1920")
        self.assertEqual(normalize("auto_1080p"), "auto_1080p")
        self.assertEqual(normalize("auto_768p"), "auto_768p")
        self.assertEqual(normalize("auto_540p"), "auto_540p")
        self.assertEqual(normalize("900x1600"), "768x1344")
        self.assertEqual(normalize("not-a-size"), "864x480")

    def test_h3_window_recommendations_are_checkpoint_aware(self):
        helpers = _load_h3_memory_helpers()
        recommend = helpers["recommended_h3_window_frames"]
        profile = helpers["recommended_h3_window_profile"]
        apply_policy = helpers["apply_h3_window_memory_policy"]
        normalize_overlap = helpers["normalize_h3_overlap_frames"]
        pruned = {
            "architecture": "minimax_h3",
            "omni_reference": False,
            "minimax_h3_full_checkpoint": False,
        }
        full = {
            "architecture": "minimax_h3_full",
            "omni_reference": False,
            "minimax_h3_full_checkpoint": True,
        }
        self.assertEqual(normalize_overlap(1), 1)
        self.assertEqual(normalize_overlap(20), 18)
        self.assertEqual(normalize_overlap(35), 35)
        self.assertEqual(normalize_overlap(120, window_frames=124), 103)

        # Each checkpoint has its own measured peak-memory curve. Pruned
        # streams fewer weights, while Full's split projections can reduce
        # peak activation pressure at some resolutions.
        self.assertEqual(recommend(12, "1344x768", pruned), 124)
        self.assertEqual(recommend(16, "1344x768", pruned), 124)
        self.assertEqual(recommend(24, "1344x768", pruned), 243)
        self.assertEqual(recommend(32, "1344x768", pruned), 345)
        self.assertEqual(recommend(16, "1344x768", full), 124)
        self.assertEqual(recommend(24, "1344x768", full), 243)
        # The aligned 720p tier is deliberately allowed longer windows as
        # VRAM increases.  It must not collapse every consumer GPU onto the
        # same conservative five-second recommendation.
        self.assertEqual(recommend(8, "1280x704", pruned), 124)
        self.assertEqual(recommend(12, "1280x704", pruned), 124)
        self.assertEqual(recommend(16, "1280x704", pruned), 243)
        self.assertEqual(recommend(23, "1280x704", pruned), 243)
        self.assertEqual(recommend(24, "1280x704", pruned), 345)
        self.assertEqual(recommend(32, "1280x704", pruned), 345)
        # The H3-specific residency cap now leaves enough packed-sequence
        # workspace for both checkpoints to complete 345 frames on 24 GB.
        self.assertEqual(recommend(16, "1280x704", full), 243)
        self.assertEqual(recommend(24, "1280x704", full), 345)
        self.assertEqual(
            profile(8, "1344x768", pruned)["fallback_resolution"],
            "480p",
        )
        self.assertEqual(recommend(12, "1920x1088", pruned), 0)
        self.assertEqual(recommend(16, "1920x1088", pruned), 124)
        self.assertEqual(recommend(24, "1920x1088", pruned), 158)
        self.assertEqual(recommend(32, "1920x1088", pruned), 243)
        self.assertEqual(recommend(40, "1920x1088", pruned), 345)
        self.assertEqual(recommend(24, "1920x1088", full), 192)
        self.assertEqual(recommend(32, "auto_1080p", full), 192)
        self.assertEqual(recommend(12, "960x544", pruned), 243)
        self.assertEqual(recommend(16, "960x544", pruned), 345)
        self.assertEqual(recommend(8, "960x544", pruned), 124)
        self.assertEqual(recommend(8, "864x480", pruned), 243)
        self.assertEqual(recommend(12, "864x480", pruned), 345)
        self.assertEqual(recommend(8, "960x544", full), 0)
        self.assertEqual(recommend(8, "864x480", full), 124)
        self.assertEqual(recommend(12, "864x480", full), 243)
        # Recommendations use actual pixel load, so lower-pixel square/4:3
        # variants within a preset can safely use a longer window than its
        # 16:9 or 9:16 variant.
        self.assertEqual(recommend(24, "1024x768", full), 345)
        self.assertEqual(recommend(24, "1440x1088", full), 243)
        self.assertEqual(recommend(24, "1088x1920", full), 192)
        self.assertEqual(recommend(32, "1344x768", full), 345)

        # Within one checkpoint, more VRAM never shortens the recommendation,
        # and more pixels never lengthen it. Checkpoints are deliberately not
        # compared because fused versus split projections change peak shape.
        vram_samples = [8, 12, 16, 24, 32, 40, 48]
        resolutions = [
            "864x480",
            "960x544",
            "1280x704",
            "1344x768",
            "1920x1088",
        ]
        for model in (pruned, full):
            for resolution in resolutions:
                values = [recommend(vram, resolution, model) for vram in vram_samples]
                self.assertEqual(values, sorted(values))
            for vram in vram_samples:
                values = [recommend(vram, resolution, model) for resolution in resolutions]
                self.assertEqual(values, sorted(values, reverse=True))

        params = {
            "resolution": "1344x768",
            "video_length": 345,
            "sliding_window_size": 345,
        }
        adjustment = apply_policy(
            params,
            pruned,
            {"gpu_vram_gb": 12},
        )
        self.assertEqual(params["video_length"], 345)
        self.assertEqual(params["sliding_window_size"], 124)
        self.assertEqual(adjustment["effective_window_frames"], 124)
        self.assertEqual(adjustment["checkpoint"], "pruned")

        unsupported = {
            "resolution": "1920x1088",
            "video_length": 345,
            "sliding_window_size": 345,
        }
        rejection = apply_policy(
            unsupported,
            pruned,
            {"gpu_vram_gb": 12},
        )
        self.assertTrue(rejection["unsupported"])
        self.assertEqual(unsupported["sliding_window_size"], 345)
        self.assertIn("720p or lower", rejection["message"])
        self.assertIn("124-frame", rejection["message"])

        manual = dict(params, sliding_window_size=345)
        manual["sliding_window_memory_override"] = True
        self.assertIsNone(
            apply_policy(
                manual,
                pruned,
                {"gpu_vram_gb": 16},
            )
        )
        self.assertEqual(manual["sliding_window_size"], 345)

        omni = dict(params, sliding_window_size=345)
        self.assertIsNone(
            apply_policy(
                omni,
                {"omni_reference": True},
                {"gpu_vram_gb": 16},
            )
        )
        self.assertEqual(omni["sliding_window_size"], 345)

    def test_h3_omni_sequence_recomputes_native_clip_budget(self):
        helpers = _load_h3_memory_helpers()
        recommend = helpers["recommended_h3_omni_sequence_profile"]
        apply_policy = helpers["apply_h3_omni_sequence_memory_policy"]
        omni = {
            "architecture": "minimax_h3_ref2va",
            "omni_reference": True,
            "minimax_h3_full_checkpoint": False,
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
            "omni_sequence_memory_policy": {
                "reference_margin_steps": 1,
            },
        }

        # Auto tracks canvas pressure and reserves one legal Ref2VA step.
        self.assertEqual(recommend(24, "960x544", omni)["frames"], 328)
        self.assertEqual(recommend(24, "1280x704", omni)["frames"], 328)
        self.assertEqual(recommend(24, "1920x1088", omni)["frames"], 141)

        full_omni = dict(
            omni,
            architecture="minimax_h3_ref2va_full",
            minimax_h3_full_checkpoint=True,
        )
        self.assertEqual(
            recommend(24, "1920x1088", full_omni)["frames"],
            175,
        )

        params = {
            "resolution": "1280x704",
            "video_length": 960,
            "minimax_h3_sequence_clip_frames": 345,
        }
        adjustment = apply_policy(params, omni, {"gpu_vram_gb": 24})
        self.assertEqual(params["video_length"], 960)
        self.assertEqual(params["minimax_h3_sequence_clip_frames"], 328)
        self.assertEqual(adjustment["effective_clip_frames"], 328)
        self.assertEqual(adjustment["reference_margin_frames"], 17)

        manual = dict(
            params,
            minimax_h3_sequence_clip_frames=345,
            minimax_h3_sequence_memory_override=True,
        )
        manual_adjustment = apply_policy(
            manual,
            omni,
            {"gpu_vram_gb": 24},
        )
        self.assertEqual(manual["minimax_h3_sequence_clip_frames"], 345)
        self.assertTrue(manual_adjustment["manual_override"])

        unsupported = {
            "resolution": "1920x1088",
            "video_length": 960,
        }
        rejection = apply_policy(
            unsupported,
            omni,
            {"gpu_vram_gb": 12},
        )
        self.assertTrue(rejection["unsupported"])
        self.assertIn("Sequence Window Length", rejection["message"])

    def test_h3_native_omni_one_shot_uses_the_base_pass_budget(self):
        helpers = _load_h3_memory_helpers()
        apply_policy = helpers["apply_h3_native_omni_memory_policy"]
        pruned = {
            "architecture": "minimax_h3_ref2va",
            "omni_reference": True,
            "minimax_h3_full_checkpoint": False,
        }
        full = {
            "architecture": "minimax_h3_ref2va_full",
            "omni_reference": True,
            "minimax_h3_full_checkpoint": True,
        }

        params = {
            "resolution": "1920x1088",
            "video_length": 345,
            "sliding_window_size": 345,
        }
        adjustment = apply_policy(params, pruned, {"gpu_vram_gb": 24})
        self.assertEqual(params["video_length"], 158)
        self.assertEqual(params["sliding_window_size"], 158)
        self.assertEqual(adjustment["effective_frames"], 158)

        safe = {
            "resolution": "1920x1088",
            "video_length": 141,
        }
        self.assertIsNone(apply_policy(safe, pruned, {"gpu_vram_gb": 24}))
        self.assertEqual(safe["video_length"], 141)

        sequence = {
            "resolution": "1920x1088",
            "video_length": 960,
            "minimax_h3_reference_sequence": True,
        }
        self.assertIsNone(
            apply_policy(sequence, pruned, {"gpu_vram_gb": 24})
        )
        self.assertEqual(sequence["video_length"], 960)

        unsupported = {
            "resolution": "1920x1088",
            "video_length": 124,
        }
        rejection = apply_policy(unsupported, full, {"gpu_vram_gb": 16})
        self.assertTrue(rejection["unsupported"])
        self.assertIn("Multi-window sequence", rejection["message"])

        launch = _read(_LAUNCH_PATH)
        duration_slider = _read(_DURATION_SLIDER_PATH)
        self.assertIn("apply_h3_native_omni_memory_policy", launch)
        self.assertIn("VRAM-aware default", duration_slider)
        self.assertIn("manually raise the native pass", duration_slider)

    def test_full_h3_preflight_recommends_pruned_turbo_without_blocking(self):
        preflight = _load_h3_memory_helpers()["h3_runtime_preflight"]
        full_first_last = {
            "architecture": "minimax_h3_full",
            "omni_reference": False,
            "minimax_h3_full_checkpoint": True,
        }
        full_omni = {
            "architecture": "minimax_h3_ref2va_full",
            "omni_reference": True,
            "minimax_h3_full_checkpoint": True,
        }
        pruned = {
            "architecture": "minimax_h3",
            "omni_reference": False,
            "minimax_h3_full_checkpoint": False,
        }

        warning = preflight(
            full_first_last,
            {"supports_triton": False, "ram_gb": 32},
        )
        self.assertEqual(warning["level"], "warning")
        self.assertFalse(warning["blocking"])
        self.assertTrue(warning["recommended_turbo"])
        self.assertEqual(warning["recommended_model_type"], "minimax_h3")
        self.assertEqual(
            {reason["code"] for reason in warning["reasons"]},
            {"triton_unavailable", "system_ram_low"},
        )
        self.assertIn("54 GB", warning["message"])
        self.assertEqual(
            preflight(
                full_omni,
                {"supports_triton": False, "ram_gb": 128},
            )["recommended_model_type"],
            "minimax_h3_ref2va",
        )
        self.assertIsNone(
            preflight(
                full_first_last,
                {"supports_triton": True, "ram_gb": 128},
            )
        )
        self.assertIsNone(
            preflight(pruned, {"supports_triton": False, "ram_gb": 16})
        )

    def test_h3_continuation_windows_never_expand_past_the_safe_cap(self):
        tree = ast.parse(_read(_WGP_PATH), filename=str(_WGP_PATH))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "compute_next_sliding_window_length"
        )
        namespace = {
            "align_model_frame_count": lambda frames, _model_def, **_kwargs: max(
                124,
                ((int(frames) - 5 + 16) // 17) * 17 + 5,
            )
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(
            compile(ast.fix_missing_locations(module), str(_WGP_PATH), "exec"),
            namespace,
        )
        size_next = namespace["compute_next_sliding_window_length"]
        model_def = {"sliding_window_exact_total_frames": True}
        # After a 124-frame first pass, H3 has 221 output frames left plus
        # its 18-frame continuation overlap. The next pass must still remain
        # at the selected 124-frame safe cap.
        self.assertEqual(size_next(239, 124, 17, model_def), 124)
        self.assertEqual(size_next(99, 124, 17, model_def), 124)

    def test_h3_single_prompt_is_paced_and_dialogue_partitioned_per_window(self):
        pace = _load_h3_memory_helpers()["pace_h3_sliding_window_prompt"]
        prompt = (
            "A three-part action unfolds. "
            "<d>[English] Opening line.</d> "
            "<d>[English] Middle line.</d> "
            "<d>[English] Final line.</d>"
        )
        first = pace(
            prompt,
            1,
            3,
            fps=24,
            current_video_length=124,
            requested_frames_to_generate=345,
            num_frames_generated=0,
            reuse_frames=18,
        )
        middle = pace(
            prompt,
            2,
            3,
            fps=24,
            current_video_length=124,
            requested_frames_to_generate=345,
            num_frames_generated=124,
            reuse_frames=18,
        )
        final = pace(
            prompt,
            3,
            3,
            fps=24,
            current_video_length=124,
            requested_frames_to_generate=345,
            num_frames_generated=230,
            reuse_frames=18,
        )
        self.assertIn("continuation window 1 of 3", first)
        self.assertIn("Opening line", first)
        self.assertNotIn("Middle line", first)
        self.assertIn("Middle line", middle)
        self.assertNotIn("Opening line", middle)
        self.assertIn("Final line", final)
        self.assertNotIn("Middle line", final)

        self.assertEqual(
            self.handler.custom_prompt_preprocess(
                prompt,
                window_no=1,
                total_windows=2,
                prompts=["explicit first", "explicit second"],
                model_def={"omni_reference": False},
            ),
            prompt,
        )

    def test_conditioner_namespaces_cover_wangp_int8_bf16_and_gguf(self):
        normalize = _load_source_function(
            _MAIN_PATH,
            "_normalize_conditioner_checkpoint_namespaces",
        )
        state, quantization, tied = normalize(
            {
                "language_model.embed_tokens.weight": "embedding",
                "language_model.layers.0.self_attn.q_proj.weight._data": "q",
                "visual.patch_embed.proj.weight": "vision",
                "model.layers.1.mlp.up_proj.weight": "already-normalized",
            },
            {"language_model.layers.0.self_attn.q_proj": "int8"},
            {"language_model.embed_tokens.weight": "language_model.lm_head.weight"},
        )
        self.assertEqual(state["model.embed_tokens.weight"], "embedding")
        self.assertEqual(
            state["model.layers.0.self_attn.q_proj.weight._data"],
            "q",
        )
        self.assertEqual(state["visual.patch_embed.proj.weight"], "vision")
        self.assertEqual(
            state["model.layers.1.mlp.up_proj.weight"],
            "already-normalized",
        )
        self.assertIn("model.layers.0.self_attn.q_proj", quantization)
        self.assertEqual(
            tied["model.embed_tokens.weight"],
            "model.lm_head.weight",
        )

    def test_h3_resolution_and_encoder_capabilities_reach_the_ui_and_backend(self):
        launch = _read(_LAUNCH_PATH)
        wgp = _read(_WGP_PATH)
        store = _read(_STORE_PATH)
        presets = _read(_RESOLUTION_PRESETS_PATH)
        aspects = _read(_ASPECT_RATIO_GRID_PATH)
        self.assertIn("_recommended_minimax_h3_encoder", launch)
        self.assertIn('"recommended": key == _h3_encoder_default', launch)
        self.assertIn('"resolution_presets": md.get("resolution_presets")', launch)
        self.assertIn("elif minimax_h3_references:", wgp)
        self.assertIn("_h3_auto_budgets", wgp)
        self.assertIn("_h3_auto_fallbacks", wgp)
        self.assertIn("resolveResolution(get().modelOptions, preset, ratio)", store)
        self.assertIn("findResolutionSelection(res, get().modelOptions)", store)
        self.assertIn("modelOptions?.resolution_preset_order", presets)
        self.assertIn("modelOptions?.supports_auto_aspect", aspects)

    def test_full_33b_defaults_are_pinned_and_keep_existing_ids_as_pruned_aliases(self):
        fl2va = json.loads(_FULL_DEFAULT_PATH.read_text(encoding="utf-8"))
        ref2va = json.loads(_REF2VA_FULL_DEFAULT_PATH.read_text(encoding="utf-8"))
        for defaults, architecture, filename in (
            (fl2va, "minimax_h3_full", "MiniMax-H3-FL2VA_int8_convrot.safetensors"),
            (ref2va, "minimax_h3_ref2va_full", "MiniMax-H3-Ref2VA_int8_convrot.safetensors"),
        ):
            model = defaults["model"]
            self.assertEqual(model["architecture"], architecture)
            self.assertEqual(model["minimax_h3_qkv_layout"], "interleaved")
            self.assertTrue(any(filename in url for url in model["URLs"]))
            self.assertTrue(
                all("fec7846aef352e58a1cfb699455e3d104281e68b" in url for url in model["URLs"])
            )
        self.assertEqual(
            json.loads(_DEFAULT_PATH.read_text(encoding="utf-8"))["model"]["architecture"],
            "minimax_h3",
        )
        self.assertEqual(
            json.loads(_REF2VA_DEFAULT_PATH.read_text(encoding="utf-8"))["model"]["architecture"],
            "minimax_h3_ref2va",
        )

    def test_ref2va_default_and_handler_contract_are_separate_from_fl2va(self):
        defaults = json.loads(_REF2VA_DEFAULT_PATH.read_text(encoding="utf-8"))
        model = defaults["model"]
        self.assertEqual(model["architecture"], "minimax_h3_ref2va")
        self.assertIn("minimax_h3_ref2va_pruned_fp8_scaled.safetensors", model["URLs"][0])
        self.assertIn("0543966fbdce5ba05709a8f2031c94bdba629b4a", model["URLs"][0])
        self.assertEqual(defaults["minimax_h3_references"], [])
        self.assertEqual(defaults["minimax_h3_reference_detail"], "match")

        model_def = self.handler.query_model_def("minimax_h3_ref2va", {})
        self.assertTrue(model_def["omni_reference"])
        self.assertTrue(model_def["supports_reference_audio"])
        self.assertTrue(model_def["t2v_class"])
        self.assertFalse(model_def["i2v_class"])
        self.assertFalse(model_def["end_frames_always_enabled"])
        self.assertTrue(model_def["sliding_window"])
        self.assertTrue(model_def["video_continuation"])
        self.assertTrue(model_def["sliding_window_exact_total_frames"])
        self.assertTrue(model_def["sliding_window_audio_history"])
        self.assertEqual(model_def["image_prompt_types_allowed"], "")
        self.assertEqual(
            model_def["omni_reference_limits"],
            {"image": 9, "video": 3, "audio": 3, "total": 12},
        )
        self.assertEqual(model_def["omni_reference_detail_default"], "match")
        self.assertEqual(model_def["director_video_strategy"], "omni_reference")
        self.assertEqual(
            model_def["director_shot_image_support"],
            "direct_references",
        )
        self.assertEqual(model_def["director_audio_input_mode"], "reference_manifest")
        self.assertFalse(model_def["director_endpoint_continuity"])
        self.assertEqual(model_def["director_memory_policy"]["checkpoint"], "pruned")
        self.assertNotIn("sliding_window_memory_policy", model_def)
        self.assertEqual(
            model_def["omni_sequence_memory_policy"]["checkpoint"],
            "pruned",
        )
        self.assertEqual(
            model_def["omni_sequence_memory_policy"]["reference_margin_steps"],
            1,
        )
        self.assertEqual(
            model_def["sliding_window_defaults"]["overlap_default"],
            18,
        )
        self.assertIn("OMNI REFERENCES", model_def["selector_help"])
        self.assertIn("audio references", model_def["selector_help"])

    def test_ref2va_long_duration_requires_reference_sequence(self):
        inserted = False
        if str(_APP) not in sys.path:
            sys.path.insert(0, str(_APP))
            inserted = True
        try:
            from models.minimax_h3.minimax_h3_handler import (
                family_handler as runtime_handler,
            )
            model_def = runtime_handler.query_model_def("minimax_h3_ref2va", {})
        finally:
            if inserted:
                sys.path.remove(str(_APP))
        ordinary = {
            "video_length": 720,
            "sliding_window_size": 345,
            "sliding_window_overlap": 18,
            "resolution": "864x480",
        }
        self.assertIsNone(
            runtime_handler.validate_generative_settings(
                "minimax_h3_ref2va",
                model_def,
                ordinary,
            )
        )
        self.assertEqual(ordinary["video_length"], 345)
        self.assertEqual(ordinary["sliding_window_size"], 345)

        sequence = {
            **ordinary,
            "video_length": 720,
            "minimax_h3_reference_sequence": True,
        }
        self.assertIsNone(
            runtime_handler.validate_generative_settings(
                "minimax_h3_ref2va",
                model_def,
                sequence,
            )
        )
        self.assertEqual(sequence["video_length"], 720)
        self.assertEqual(sequence["sliding_window_size"], 345)
        self.assertEqual(sequence["sliding_window_overlap"], 18)

    def test_first_last_multi_window_toggle_enforces_one_native_pass(self):
        inserted = False
        if str(_APP) not in sys.path:
            sys.path.insert(0, str(_APP))
            inserted = True
        try:
            from models.minimax_h3.minimax_h3_handler import (
                family_handler as runtime_handler,
            )
            model_def = runtime_handler.query_model_def("minimax_h3", {})
        finally:
            if inserted:
                sys.path.remove(str(_APP))
        request = {
            "video_length": 720,
            "sliding_window_size": 243,
            "sliding_window_overlap": 18,
            "resolution": "864x480",
            "minimax_h3_multi_window": False,
        }
        self.assertIsNone(
            runtime_handler.validate_generative_settings(
                "minimax_h3",
                model_def,
                request,
            )
        )
        self.assertEqual(request["video_length"], 243)
        self.assertEqual(request["sliding_window_size"], 243)

        # First / Last now opens as one native pass. An older preset without
        # the new checkbox field must therefore behave exactly like False;
        # only an explicit True opts into a long sliding-window timeline.
        legacy_request = {
            "video_length": 720,
            "sliding_window_size": 243,
            "sliding_window_overlap": 18,
            "resolution": "864x480",
        }
        self.assertIsNone(
            runtime_handler.validate_generative_settings(
                "minimax_h3",
                model_def,
                legacy_request,
            )
        )
        self.assertEqual(legacy_request["video_length"], 243)

    def test_native_omni_manual_window_override_is_not_silently_capped(self):
        helpers = _load_h3_memory_helpers()
        apply_policy = helpers["apply_h3_native_omni_memory_policy"]
        model_def = self.handler.query_model_def("minimax_h3_ref2va", {})
        request = {
            "resolution": "1920x1088",
            "video_length": 345,
            "sliding_window_size": 345,
            "sliding_window_memory_override": True,
        }
        self.assertIsNone(apply_policy(request, model_def, {"gpu_vram_gb": 24}))
        self.assertEqual(request["video_length"], 345)
        self.assertEqual(request["sliding_window_size"], 345)

    def test_ref2va_manual_sequence_bypasses_the_llm_planner(self):
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)
        prompt_input = _read(_PROMPT_INPUT_PATH)
        multi_window_controls = _read(_H3_MULTI_WINDOW_CONTROLS_PATH)
        wgp = _read(_WGP_PATH)

        self.assertIn("minimax_h3_sequence_prompt_mode", multi_window_controls)
        self.assertIn("Manual - one per line", multi_window_controls)
        self.assertIn("h3ManualSequencePrompts", store)
        self.assertIn("h3OmniSequenceWindowCount", store)
        self.assertIn("usesH3ManualSequence", prompt_input)
        self.assertIn("build_manual_h3_reference_sequence_plan", launch)
        self.assertIn("the sequence planner LLM is bypassed", launch)
        self.assertIn('minimax_h3_sequence_prompt_mode="auto"', wgp)

    def test_shared_h3_window_ui_and_durable_overrides_are_wired(self):
        controls = _read(_H3_MULTI_WINDOW_CONTROLS_PATH)
        sidebar = _read(_SIDEBAR_PATH)
        advanced = _read(_ADVANCED_SETTINGS_PATH)
        duration = _read(_DURATION_SLIDER_PATH)
        store = _read(_STORE_PATH)
        launch = _read(_LAUNCH_PATH)
        client = _read(_ROOT / "ui" / "src" / "api" / "client.ts")

        self.assertIn("Multi-window sequence", controls)
        self.assertIn("minimax_h3_multi_window", controls)
        self.assertIn("minimax_h3_reference_sequence", controls)
        self.assertIn("Window prompts", controls)
        self.assertIn("<H3MultiWindowControls />", sidebar)
        self.assertNotIn("Plan Prompt Across Windows", advanced)
        self.assertIn("Window Length", duration)
        self.assertIn("Recommended", duration)
        self.assertIn("saveH3WindowOverride", duration)
        self.assertIn("modelOptions?.frames_maximum ?? 345", duration)
        self.assertIn("h3WindowOverrideKey", store)
        self.assertIn("api.fetchH3WindowOverrides()", store)
        self.assertIn('@api.get("/api/v1/h3-window-overrides")', launch)
        self.assertIn('@api.put("/api/v1/h3-window-overrides")', launch)
        self.assertIn("updateH3WindowOverrides", client)

    def test_h3_selector_names_and_audio_badges_are_user_facing(self):
        expected_names = {
            _DEFAULT_PATH: "H3 First / Last — Pruned",
            _FULL_DEFAULT_PATH: "H3 First / Last — Full",
            _REF2VA_DEFAULT_PATH: "H3 Omni — Pruned",
            _REF2VA_FULL_DEFAULT_PATH: "H3 Omni — Full",
        }
        for path, expected_name in expected_names.items():
            defaults = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(defaults["model"]["name"], expected_name)

        full_fl2va = self.handler.query_model_def("minimax_h3_full", {})
        full_omni = self.handler.query_model_def("minimax_h3_ref2va_full", {})
        self.assertIn("FULL 33B", full_fl2va["selector_help"])
        self.assertEqual(
            full_fl2va["sliding_window_memory_policy"]["checkpoint"],
            "full",
        )
        self.assertIn("converts Pruned adapters", full_omni["lora_compatibility_note"])

        selector = _read(_MODEL_SELECTOR_PATH)
        self.assertIn("Audio Out", selector)
        self.assertIn("Audio In", selector)
        self.assertNotIn("badges.push('Audio')", selector)
        self.assertIn("lora_compatibility_note", _read(_LORA_SELECTOR_PATH))
        launch = _read(_LAUNCH_PATH)
        self.assertIn('"selector_help": md.get("selector_help", "")', launch)
        self.assertIn('"lora_compatibility_note": md.get("lora_compatibility_note", "")', launch)
        self.assertIn('"director_memory_policy": md.get', launch)

    def test_h3_reserves_transformer_activation_workspace(self):
        source = _read(_HANDLER_PATH)
        self.assertIn("_TRANSFORMER_WORKING_VRAM_MB = 10 * 1024", source)
        self.assertIn('"workingVRAM": {', source)
        self.assertIn('"transformer": _TRANSFORMER_WORKING_VRAM_MB', source)

    def test_h3_video_references_get_a_dedicated_memory_profile(self):
        launch = _read(_LAUNCH_PATH)
        wgp = _read(_WGP_PATH)
        transformer = _read(_TRANSFORMER_PATH)
        self.assertIn("_h3_video_reference_count", launch)
        self.assertIn("h3_reference_activation_gb", launch)
        self.assertIn("compute_h3_weight_budget", launch)
        self.assertIn("h3_weight_budget_gb", launch)
        self.assertIn("resident H3 profile will reload with packed-sequence headroom", launch)
        self.assertIn("_maestro_profile_vram_coefficient", wgp)
        self.assertIn("get_linear_split_map", transformer)
        self.assertIn("MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", transformer)
        self.assertIn("from shared.attention import pay_attention", transformer)
        self.assertIn("[MiniMax H3 Perf]", _read(_MAIN_PATH))
        self.assertIn('"first_block_cache": md.get', launch)

    def test_all_auxiliary_downloads_are_revision_pinned(self):
        downloads = self.handler.query_model_files(lambda item: [item], "minimax_h3")
        self.assertEqual(len(downloads), 2)
        self.assertEqual(downloads[0]["repoId"], "Comfy-Org/MiniMax-H3")
        self.assertEqual(downloads[0]["revision"], "0543966fbdce5ba05709a8f2031c94bdba629b4a")
        self.assertEqual(downloads[0]["sourceFolderList"], ["vae"])
        self.assertIn("minimax_h3_video_vae_fp16.safetensors", downloads[0]["fileList"][0])
        self.assertIn("minimax_h3_audio_vae_fp32.safetensors", downloads[0]["fileList"][0])
        self.assertEqual(downloads[1]["repoId"], "MiniMaxAI/MiniMax-H3")
        self.assertEqual(downloads[1]["revision"], "5d9b308a59ab12e67147f191e184baf704185bd1")

    def test_maestro_registers_the_family_and_uses_its_native_frame_grid(self):
        source = _read(_WGP_PATH)
        self.assertIn('"models.minimax_h3.minimax_h3_handler"', source)
        self.assertIn("video_length = normalize_model_total_frame_count(video_length, model_def)", source)
        self.assertIn(
            "frame_num=align_model_frame_count(current_video_length, model_def, for_generation=True)",
            source,
        )
        self.assertIn('model_def.get("frames_maximum", None)', source)

    def test_studio_h3_duration_and_window_controls_are_model_aware(self):
        duration = _read(_DURATION_SLIDER_PATH)
        advanced = _read(_ADVANCED_SETTINGS_PATH)
        store = _read(_STORE_PATH)
        launch = _read(_LAUNCH_PATH)

        self.assertIn(
            "supportsSlidingWindows = modelOptions?.sliding_window === true",
            duration,
        )
        self.assertIn("modelOptions?.frames_maximum", duration)
        self.assertIn("const directOmni = isOmniReference && !omniReferenceSequence", duration)
        self.assertIn("const h3MultiWindowEnabled = isOmniReference", duration)
        self.assertIn(
            "const h3NativeDurationSlider = isH3 && !h3MultiWindowEnabled",
            duration,
        )
        self.assertIn(
            "s.params.minimax_h3_multi_window === true",
            duration,
        )
        self.assertIn(
            "const durationSliderMax = h3NativeDurationSlider ? maximumFrames : maxDuration",
            duration,
        )
        self.assertIn(
            "const durationSliderStep = h3NativeDurationSlider ? frameStep : durationStep",
            duration,
        )
        self.assertIn("setDuration(preferredSeconds)", duration)
        self.assertIn("max={isH3 ? maximumFrames : windowMaxSeconds}", duration)
        self.assertIn("value={isH3 ? currentWindowFrames : windowSize}", duration)
        self.assertIn("sliderValue / fps", duration)
        self.assertNotIn("Math.max(minDuration, windowSize)", duration)
        self.assertIn("modelOptions?.sliding_window", advanced)
        self.assertIn("if (!supportsSlidingWindows && maximumFrames != null)", store)
        self.assertIn("const h3SingleNativePass = (", store)
        self.assertIn("nativeMaximum ?? Number.POSITIVE_INFINITY", store)
        self.assertIn("expandNativeWindow", store)
        self.assertIn("delete params.sliding_window_size", store)
        self.assertIn('"frames_maximum": md.get("frames_maximum")', launch)
        self.assertIn("sliding_window_memory_policy", duration)
        self.assertIn("s.params.resolution", duration)
        self.assertIn("safeWindowFrames", duration)
        self.assertIn("unsupportedAutoResolution", duration)
        self.assertIn("fallbackResolution", duration)
        self.assertIn("sliding_window_memory_override", store)
        self.assertIn("full prompt auto-paced", duration)
        self.assertIn('"sliding_window_memory_policy": md.get(', launch)
        self.assertIn('"omni_sequence_memory_policy": md.get(', launch)
        self.assertIn('h3_window_adjustment.get("unsupported")', launch)
        self.assertIn(
            'body.get("minimax_h3_multi_window", False) is True',
            launch,
        )

    def test_h3_is_enabled_for_existing_and_fresh_installs(self):
        store = _read(_STORE_PATH)
        default_block = store.split("const DEFAULT_ENABLED_MODELS = new Set([", 1)[1].split("])\n", 1)[0]
        self.assertIn("'minimax_h3'", default_block)
        self.assertIn("'minimax_h3_full'", default_block)
        self.assertIn("'minimax_h3_ref2va'", default_block)
        self.assertIn("'minimax_h3_ref2va_full'", default_block)
        self.assertIn("const DEFAULTS_VERSION = 8", store)
        self.assertIn("6: ['minimax_h3']", store)
        self.assertIn("7: ['minimax_h3_ref2va']", store)
        self.assertIn("8: ['minimax_h3_full', 'minimax_h3_ref2va_full']", store)
        self.assertIn('md.get("returns_audio", False)', _read(_LAUNCH_PATH))

    def test_h3_prompt_guides_cover_native_audio_and_director(self):
        self.assertIn('"minimax_h3": "minimax_h3_video.md"', _read(_ENHANCE_GUIDES_PATH))
        self.assertIn('"minimax_h3": "minimax_h3_video"', _read(_PROMPT_POLISH_PATH))
        self.assertIn(
            '"minimax_h3_ref2va": "minimax_h3_ref2va_video"',
            _read(_PROMPT_POLISH_PATH),
        )
        enhance_guide = _read(_H3_ENHANCE_GUIDE_PATH)
        dialect_guide = _read(_H3_DIALECT_GUIDE_PATH)
        ref2va_dialect_guide = _read(_H3_REF2VA_DIALECT_GUIDE_PATH)
        self.assertIn("joint video-and-audio", enhance_guide)
        for required in (
            "integrated_multimodal_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
            "(S1)",
            "<d>[English]",
            "remain silent with their mouths closed",
        ):
            self.assertIn(required, enhance_guide)
        self.assertIn("but supplies no script", enhance_guide)
        self.assertIn("TIMED SILENCE AROUND DIALOGUE", enhance_guide)
        self.assertIn("idle staring", enhance_guide)
        self.assertIn("<d>[English] Exact words.</d>", dialect_guide)
        self.assertIn("Never invent extra speech", dialect_guide)
        self.assertIn("proper names", dialect_guide)
        self.assertIn("Maestro maps the exact per-shot", ref2va_dialect_guide)
        self.assertIn("Do not guess reference numbers", ref2va_dialect_guide)
        self.assertIn("subject_definitions, summary, retention_analysis", ref2va_dialect_guide)
        self.assertIn("proper names", ref2va_dialect_guide)

    def test_h3_enhance_path_preserves_context_ir_contract(self):
        launch = _read(_LAUNCH_PATH)
        llm_service = _read(_LLM_SERVICE_PATH)
        self.assertIn("needs_h3_context_ir", launch)
        self.assertIn("enhancer_enabled > 0 and not needs_h3_context_ir", launch)
        self.assertIn("is_h3_context_ir", llm_service)
        self.assertIn("is_h3_ref2va", llm_service)
        self.assertIn("is_h3_structured = is_h3_context_ir or is_h3_ref2va", llm_service)
        self.assertIn('mode in ("video", "avatar") and not is_h3_structured', llm_service)
        self.assertIn("CRITICAL MINIMAX H3 OUTPUT CONTRACT", llm_service)
        self.assertIn("effective_max_tokens = max(effective_max_tokens, 768)", llm_service)
        self.assertIn("effective_max_tokens = max(effective_max_tokens, 1200)", llm_service)

    def test_ref2va_prompt_guide_uses_official_labels_and_six_sections(self):
        self.assertIn(
            '"minimax_h3_ref2va": "minimax_h3_ref2va_video.md"',
            _read(_ENHANCE_GUIDES_PATH),
        )
        guide = _read(_H3_REF2VA_GUIDE_PATH)
        for required in (
            "<Picture 1>",
            "<Video 1>",
            "<Audio 1>",
            "subject_definitions:",
            "summary:",
            "retention_analysis:",
            "detailed_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
            "TIMED SILENCE AROUND DIALOGUE",
            "do not authorize",
        ):
            self.assertIn(required, guide)

    def test_omni_reference_request_and_ui_are_wired_end_to_end(self):
        launch = _read(_LAUNCH_PATH)
        main = _read(_MAIN_PATH)
        wgp = _read(_WGP_PATH)
        store = _read(_STORE_PATH)
        section = _read(_OMNI_REFERENCE_SECTION_PATH)
        generate_button = _read(_GENERATE_BUTTON_PATH)
        self.assertIn('if _generation_model_def.get("omni_reference"):', launch)
        self.assertIn("validate_reference_manifest", launch)
        self.assertIn("per_clip_minimax_h3_references", launch)
        self.assertIn("director_trim_end_frames", launch)
        self.assertIn('"minimax_h3_references": minimax_h3_references', wgp)
        self.assertIn('multi_clip_info.get("concat_audio_path")', wgp)
        self.assertIn("build_ref2va_packed_sequence", main)
        self.assertIn("duration_seconds=target_frame_num / fps", main)
        self.assertIn("num_condition_video_rows", main)
        self.assertIn("const omniReferences = state.params.minimax_h3_references ?? []", store)
        self.assertIn("delete params.minimax_h3_references", store)
        self.assertIn("reference_context: referenceContext", store)
        self.assertIn("intent=AUDIO REUSE / PERFORMANCE DRIVER", store)
        self.assertIn("intent=VOICE REFERENCE", store)
        self.assertIn('draggable', section)
        self.assertIn("Include soundtrack", section)
        self.assertIn("Attach audio", section)
        self.assertIn("audio_path", section)
        self.assertIn("Maximum detail", section)
        self.assertIn("Voice reference", section)
        self.assertIn("Music / performance timeline", section)
        self.assertIn("Music / sound style only", section)
        self.assertIn("advances through the recording across sequence clips", section)
        self.assertIn("timeline_start_frame=window_start_frame_no", main)
        self.assertNotIn('accept="image/*,video/*,audio/*', section)
        self.assertNotIn('accept="audio/*', section)
        self.assertIn("iOS/WebKit can", section)
        self.assertIn("type !== 'audio' && type !== 'video'", section)
        self.assertIn("const hasOmniVisualReference = useStore(s =>", generate_button)
        self.assertNotIn(
            "useStore(s => s.params.minimax_h3_references ?? [])",
            generate_button,
        )

    def test_non_sliding_h3_enhance_request_stays_one_timeline(self):
        store = _read(_STORE_PATH)
        prompt_input = _read(_PROMPT_INPUT_PATH)
        expected = "supportsSlidingWindows = state.modelOptions?.sliding_window === true"
        self.assertIn(expected, store)
        self.assertIn("(!isH3FirstLast || params.minimax_h3_multi_window === true)", store)
        self.assertIn("&& stride > 0", store)
        self.assertIn("supportsSlidingWindows = modelOptions?.sliding_window === true", prompt_input)
        self.assertIn("(!isH3FirstLast || h3FirstLastMultiWindow)", prompt_input)
        self.assertIn("&& stride > 0", prompt_input)

    def test_ref2va_enhance_cleanup_preserves_structured_reference_reuse(self):
        helpers = _load_llm_enhance_helpers()
        structured = "\n".join(
            (
                "subject_definitions: <Subject 1> comes from <Picture 1> and uses <Audio 1>.",
                "summary: <Subject 1> speaks.",
                "retention_analysis: <Picture 1> reference; <Audio 1> reference.",
                "detailed_description: <Subject 1> (S1) says <d>[English] Hello.</d>.",
                "overall_soundscape: <Audio 1> guides the voice over room ambience.",
                "non_diegetic_music: N/A",
            )
        )
        cleaned = helpers["_clean_enhance_output"](structured, preserve_structure=True)
        self.assertTrue(helpers["_has_complete_h3_ref2va_structure"](cleaned))
        self.assertGreaterEqual(cleaned.count("<Audio 1>"), 3)

        fallback = helpers["_build_h3_ref2va_tagged_fallback"](
            'A man says, "Hello."',
            "<Picture 1>: identity\n<Audio 1>: intent=VOICE REFERENCE",
        )
        self.assertTrue(helpers["_has_complete_h3_ref2va_structure"](fallback))
        self.assertTrue(
            helpers["_h3_dialogue_contract_satisfied"]('A man says, "Hello."', fallback)
        )
        self.assertIn("<d>[English] Hello.</d>", fallback)
        self.assertIn("begin at the first frame", fallback)
        self.assertTrue(
            helpers["_h3_voice_binding_contract_satisfied"](
                fallback,
                "<Picture 1>: identity\n<Audio 1>: intent=VOICE REFERENCE",
            )
        )

        omitted = structured.replace("<d>[English] Hello.</d>", "the requested line")
        self.assertFalse(
            helpers["_h3_dialogue_contract_satisfied"]('A man says, "Hello."', omitted)
        )
        repaired = helpers["_inject_missing_h3_dialogue"](
            omitted,
            'A man says, "Hello."',
            ref2va=True,
        )
        self.assertTrue(
            helpers["_h3_dialogue_contract_satisfied"]('A man says, "Hello."', repaired)
        )

        base = helpers["_build_h3_context_fallback"](
            'Jim says, "Run it locally."',
            has_start_image=False,
        )
        self.assertTrue(helpers["_has_complete_h3_context_structure"](base))
        self.assertIn("<d>[English] Run it locally.</d>", base)
        requirement = helpers["_build_h3_dialogue_requirement"](
            'Jim says, "Run it locally."',
            10,
        )
        self.assertIn("REQUIRED VERBATIM", requirement)
        self.assertIn("Run it locally.", requirement)
        self.assertIn("From 0.00 to 2.00 seconds", requirement)
        self.assertIn("no human voice", requirement)

        timed = helpers["_build_h3_ref2va_tagged_fallback"](
            'Blaine says, "Snap this, bitch" before punching.',
            "<Picture 1>: Blaine identity\n<Audio 1>: Blaine voice",
            duration_seconds=10,
        )
        self.assertTrue(
            helpers["_h3_timed_silence_contract_satisfied"](
                'Blaine says, "Snap this, bitch" before punching.',
                timed,
                10,
            )
        )
        duplicated = timed.replace(
            "summary: A finished video matching the requested action, identity, setting, and explicitly tagged dialogue.",
            'summary: Blaine declares, "Snap this, bitch."',
        )
        deduplicated = helpers["_strip_h3_untagged_dialogue_duplicates"](
            duplicated,
            'Blaine says, "Snap this, bitch" before punching.',
        )
        self.assertIn("summary: Blaine declares, the scripted line", deduplicated)
        self.assertIn("<d>[English] Snap this, bitch</d>", deduplicated)

        noisy = timed.replace(
            "overall_soundscape: Continuous",
            "overall_soundscape: Blaine grunts loudly. Continuous",
        ).replace(
            "non_diegetic_music: N/A",
            "non_diegetic_music: Epic orchestral score.",
        )
        cleaned_sound = helpers["_enforce_h3_soundscape_silence"](
            noisy,
            'Blaine says, "Snap this, bitch" before punching.',
        )
        self.assertNotIn("grunts loudly", cleaned_sound)
        self.assertIn("no human voices", cleaned_sound)
        no_invented_music = helpers["_enforce_h3_music_request"](
            cleaned_sound,
            'A cinematic fight. Blaine says, "Snap this, bitch".',
            "<Audio 1>: intent=VOICE REFERENCE",
        )
        self.assertTrue(no_invented_music.endswith("non_diegetic_music: N/A"))

        silent_discussion = helpers["_build_h3_context_fallback"](
            "Jim and Dwight discuss local AI.",
            has_start_image=False,
        )
        generated = helpers["_inject_h3_generated_dialogue"](
            silent_discussion,
            "Jim (S1): <d>[English] It runs locally.</d>\n"
            "Dwight (S2): <d>[English] Good. More secure.</d>\nIgnore this narration.",
            ref2va=False,
        )
        self.assertTrue(
            helpers["_h3_dialogue_contract_satisfied"](
                "Jim and Dwight discuss local AI.",
                generated,
            )
        )
        self.assertNotIn("Ignore this narration", generated)

    def test_frame_aligner_preserves_h3_and_legacy_grids(self):
        align = _load_frame_aligner()
        h3 = {
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frame_alignment_modulus": 17,
            "frame_alignment_remainder": 5,
            "frame_alignment_mode": "ceil",
            "latent_size": 17,
        }
        self.assertEqual([align(value, h3) for value in (1, 120, 124, 125, 345, 999)], [124, 124, 124, 141, 345, 345])
        self.assertEqual(align(346, h3, clamp_maximum=False), 362)
        legacy = {"latent_size": 4, "frames_steps": 4}
        self.assertEqual(align(120, legacy), 117)
        self.assertEqual(align(120, legacy, for_generation=True), 121)


class TestMiniMaxH3RuntimeSource(unittest.TestCase):
    def test_runtime_uses_the_official_dual_scheduler_and_audio_output(self):
        main = _read(_MAIN_PATH)
        self.assertIn("MiniMaxH3Scheduler(shift=12.0)", main)
        self.assertIn("MiniMaxH3Scheduler(shift=3.0)", main)
        self.assertIn('"audio_sampling_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE', main)
        self.assertIn("MINIMAX_H3_KEYFRAME_ENCODE_SEED", main)
        self.assertIn("prepare_keyframe_image", main)

    def test_h3_runtime_uses_previous_window_motion_and_audio(self):
        main = _read(_MAIN_PATH)
        wgp = _read(_WGP_PATH)
        store = _read(_STORE_PATH)
        self.assertIn("def _split_continuation_video", main)
        self.assertIn("input_video=None", main)
        self.assertIn("input_waveform=None", main)
        self.assertIn("prefix_frames_count: int = 0", main)
        self.assertIn('"anchor": "history"', main)
        self.assertIn('"keep_all_latents": True', main)
        self.assertIn("_encode_continuation_audio", main)
        self.assertIn("continuation_picture", main)
        self.assertIn("add_ref2va_continuation_context", main)
        self.assertIn("keyframe_anchors=anchors", main)
        self.assertIn("audio_condition_anchors=audio_anchors", main)
        self.assertIn("pre_audio_guide_sample_rate", wgp)
        self.assertIn('model_def.get("audio_guide_window_slicing", False)', wgp)
        self.assertIn("_normalizeSlidingWindowOverlap", store)
        self.assertIn('"sliding_window_audio_history": md.get(', _read(_LAUNCH_PATH))
        self.assertNotIn(
            "params.sliding_window_overlap = swDefaults?.overlap_default",
            store,
        )
        self.assertIn('"sliding_window_trim_to_requested"', wgp)
        self.assertIn('"sliding_window_end_image_at_final"', wgp)

    def test_h3_runtime_exposes_native_media_source_conditioning(self):
        main = _read(_MAIN_PATH)
        handler = _read(_HANDLER_PATH)
        launch = _read(_LAUNCH_PATH)
        inputs = _read(
            _ROOT / "ui" / "src" / "components" / "Sidebar" / "InputsPanel.tsx"
        )
        utils = _read(_APP / "shared" / "utils" / "utils.py")
        self.assertIn("def _build_frozen_control_video", main)
        self.assertIn("def _encode_target_audio_condition", main)
        self.assertIn("target_condition_audio_latents=", main)
        self.assertIn("target_condition_video_frames=", main)
        self.assertIn("generated_audio_local_indices", main)
        self.assertIn('"audio_prompt_type_sources"', handler)
        self.assertIn('"output_audio_is_input_audio": True', handler)
        self.assertIn('"minimax_h3_media_sources"', launch)
        self.assertIn("h3MediaSources", inputs)
        self.assertIn("The source pictures remain unchanged", inputs)
        self.assertIn("frame_offset = 1", utils)
        self.assertIn('frame_alignment_remainder", 1', _read(_WGP_PATH))

    def test_h3_runtime_exposes_native_video_to_video_editing(self):
        main = _read(_MAIN_PATH)
        handler = _read(_HANDLER_PATH)
        launch = _read(_LAUNCH_PATH)
        inputs = _read(_INPUTS_PANEL_PATH)
        advanced = _read(_ADVANCED_SETTINGS_PATH)
        types = _read(_TYPES_PATH)
        store = _read(_STORE_PATH)
        provenance = _read(_APP / "models" / "minimax_h3" / "UPSTREAM.md")

        self.assertIn("def _resize_video_mask", main)
        self.assertIn("def _reinject_video_source", main)
        self.assertIn("video_to_video_mode", main)
        self.assertIn("source_posterior.mode()", main)
        self.assertIn('"video_to_video_inpaint": True', handler)
        self.assertIn('"mask_preprocessing"', handler)
        self.assertIn('"video_to_video_inpaint": md.get(', launch)
        self.assertIn('"mask_preprocessing": extract_choice', launch)
        self.assertIn("h3ControlVisualMode", inputs)
        self.assertIn("Edit inside a white mask", inputs)
        self.assertIn("Mask protection duration", inputs)
        self.assertIn("!modelOptions?.minimax_h3_media_sources", advanced)
        self.assertIn("minimax_h3_control_visual_mode", types)
        self.assertIn("'video_mask'", store)
        self.assertIn("Phase 6 adapts WanGP v12.44", provenance)

    def test_h3_timed_frame_injection_reaches_the_native_runtime(self):
        resolve = _load_source_function(
            _MAIN_PATH,
            "_resolve_h3_injected_frame_conditions",
        )
        images = ["overlap", "first-fresh", "last"]
        self.assertEqual(
            resolve(
                images,
                [0, 18, 344],
                history_count=17,
                target_frame_num=328,
            ),
            [("first-fresh", 1), ("last", 327)],
        )
        with self.assertRaisesRegex(ValueError, "one injected-frame position"):
            resolve(
                images,
                [18],
                history_count=17,
                target_frame_num=328,
            )

        main = _read(_MAIN_PATH)
        wgp = _read(_WGP_PATH)
        launch = _read(_LAUNCH_PATH)
        inputs = _read(_ROOT / "ui" / "src" / "components" / "Sidebar" / "InputsPanel.tsx")
        guide = _read(_H3_ENHANCE_GUIDE_PATH)
        self.assertIn('"anchor": "frame"', main)
        self.assertIn("frames_to_inject = frames_to_inject_for_model", wgp)
        self.assertIn('"custom_frames_injection": md.get', launch)
        self.assertIn("modelOptions?.custom_frames_injection === true", inputs)
        self.assertIn("injected-frame picture is an exact visual destination", guide)

    def test_continuation_uses_streaming_vae_tiles_and_migrates_legacy_overlap(self):
        handler = _read(_HANDLER_PATH)
        video_vae = _read(_VIDEO_VAE_PATH)
        self.assertIn("settings_version < 2.58", handler)
        self.assertIn("normalize_h3_overlap_frames", handler)
        self.assertIn("Decode one tile at a time", video_vae)
        self.assertIn("new_tails.append", video_vae)
        self.assertIn("keep_all_latents", video_vae)

    def test_turbo_lora_uses_h3_specific_validation_and_step_contract(self):
        main = _read(_MAIN_PATH)
        transformer = _read(_TRANSFORMER_PATH)
        wgp = _read(_WGP_PATH)
        self.assertIn("def validate_loras", main)
        self.assertIn("h3_scheduler_grid_points", main)
        self.assertIn("video shift 12 / audio shift 3 schedules", main)
        self.assertIn("def preprocess_loras", transformer)
        self.assertIn("Adapt AdaLN width", transformer)
        self.assertIn("convert_adaln_loras", transformer)
        self.assertIn("def finalize_loras", main)
        self.assertIn("install_native_lora_forwards", main)
        self.assertIn('hasattr(wan_model, "validate_loras")', wgp)
        self.assertIn('hasattr(wan_model, "finalize_loras")', wgp)
        launch = _read(_LAUNCH_PATH)
        self.assertIn("def _lora_is_compatible_with_model", launch)
        self.assertIn("minimax_h3_full_checkpoint", launch)

    def test_turbo_metadata_detection_and_evaluation_count(self):
        turbo = _load_turbo_helpers()
        self.assertTrue(
            turbo.is_minimax_h3_turbo_lora(
                "minimax_h3_turbo_4step_ckpt500.safetensors"
            )
        )
        self.assertEqual(turbo.h3_scheduler_grid_points(8, turbo_active=False), 8)
        self.assertEqual(turbo.h3_scheduler_grid_points(8, turbo_active=True), 9)

        metadata = {
            "__metadata__": {
                "application": "W_eff = W + lora_B @ lora_A",
                "base_model": "MiniMax-H3",
                "sampler_steps": "4",
            }
        }
        raw_header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            renamed = Path(temp_dir) / "renamed_adapter.safetensors"
            renamed.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header)
            self.assertTrue(turbo.is_minimax_h3_turbo_lora(str(renamed)))

            ordinary = Path(temp_dir) / "ordinary.safetensors"
            ordinary_header = json.dumps(
                {"__metadata__": {"base_model": "MiniMax-H3"}}
            ).encode("utf-8")
            ordinary.write_bytes(
                struct.pack("<Q", len(ordinary_header)) + ordinary_header
            )
            self.assertFalse(turbo.is_minimax_h3_turbo_lora(str(ordinary)))

    def test_managed_turbo_mode_uses_the_pinned_six_step_recipe(self):
        turbo = _load_turbo_helpers()
        body = {
            "minimax_h3_turbo_mode": True,
            "num_inference_steps": 20,
            "activated_loras": [
                "cinematic_style.safetensors",
                "minimax_h3_turbo_4step_ema_ckpt500.safetensors",
                r"loras\minimax_h3\minimax_h3_turbo_4step_ckpt500.safetensors",
            ],
            "loras_multipliers": "1.15 1.05 0.65",
        }

        self.assertTrue(
            turbo.normalize_minimax_h3_turbo_request(
                body,
                full_checkpoint=True,
            )
        )
        self.assertEqual(body["num_inference_steps"], 6)
        self.assertEqual(
            body["activated_loras"],
            [
                "cinematic_style.safetensors",
                turbo.MINIMAX_H3_TURBO_LORA_FILENAME,
            ],
        )
        self.assertEqual(body["loras_multipliers"], "1.15 0.65")
        self.assertEqual(
            turbo.MINIMAX_H3_TURBO_LORA_SHA256,
            "82d0acff583b04ad9a4238a7440b584b56094bfb7c4fdb2981f67c7a4784b62d",
        )

        disabled = {"minimax_h3_turbo_mode": False, "num_inference_steps": 20}
        self.assertFalse(
            turbo.normalize_minimax_h3_turbo_request(
                disabled,
                full_checkpoint=False,
            )
        )
        self.assertEqual(disabled["num_inference_steps"], 20)

        missing_selection = {
            "minimax_h3_turbo_mode": True,
            "activated_loras": [],
            "loras_multipliers": "",
        }
        self.assertTrue(
            turbo.normalize_minimax_h3_turbo_request(
                missing_selection,
                full_checkpoint=True,
            )
        )
        self.assertEqual(missing_selection["loras_multipliers"], "0.50")

        pruned = {"minimax_h3_turbo_mode": True}
        self.assertTrue(
            turbo.normalize_minimax_h3_turbo_request(
                pruned,
                full_checkpoint=False,
            )
        )
        self.assertEqual(pruned["num_inference_steps"], 6)
        self.assertEqual(pruned["loras_multipliers"], "0.50")

    def test_managed_turbo_choice_is_discoverable_for_full_and_pruned(self):
        launch = _read(_LAUNCH_PATH)
        toggle = _read(_TURBO_TOGGLE_PATH)
        sidebar = _read(_SIDEBAR_PATH)
        advanced = _read(_ADVANCED_SETTINGS_PATH)
        types_source = _read(_TYPES_PATH)

        self.assertIn("def _minimax_h3_turbo_option", launch)
        self.assertIn('names.add(turbo_option["filename"])', launch)
        self.assertIn("MINIMAX_H3_TURBO_LORA_FILENAME: {", launch)
        self.assertIn('"minimax_h3_turbo": _minimax_h3_turbo_option(md)', launch)
        self.assertIn('"minimax_h3_runtime_advisory":', launch)
        self.assertIn("_minimax_h3_runtime_advisory", launch)
        self.assertIn("normalize_minimax_h3_turbo_request", launch)
        self.assertIn("<MiniMaxH3TurboToggle />", sidebar)
        self.assertIn("Experimental", toggle)
        self.assertIn("setParam('num_inference_steps', option.steps)", toggle)
        self.assertIn("toggleLora(option.filename)", toggle)
        self.assertIn("setLoraWeight(option.filename, 0, option.weight)", toggle)
        self.assertIn("Use Pruned Turbo", toggle)
        self.assertIn("recommended_model_type", toggle)
        self.assertIn("disabled={h3TurboMode}", advanced)
        self.assertIn("minimax_h3_turbo_mode?: boolean", types_source)
        self.assertIn("minimax_h3_runtime_advisory?:", types_source)

        option = _load_source_function(_LAUNCH_PATH, "_minimax_h3_turbo_option")
        sys.path.insert(0, str(_APP))
        try:
            pruned = option({
                "architecture": "minimax_h3",
                "minimax_h3_full_checkpoint": False,
            })
            full = option({
                "architecture": "minimax_h3_full",
                "minimax_h3_full_checkpoint": True,
            })
        finally:
            if sys.path and sys.path[0] == str(_APP):
                sys.path.pop(0)
        self.assertEqual(pruned["steps"], 6)
        self.assertEqual(pruned["weight"], 0.50)
        self.assertEqual(full["steps"], 6)
        self.assertEqual(full["weight"], 0.50)
        self.assertTrue(full["experimental"])

    def test_consumer_checkpoint_shapes_are_kept_native(self):
        transformer = _read(_TRANSFORMER_PATH)
        conditioner = _read(_CONDITIONER_PATH)
        main = _read(_MAIN_PATH)
        self.assertIn("self.qkv_proj", transformer)
        self.assertIn("self.fc1", transformer)
        self.assertIn("adaln_t_table", transformer)
        self.assertIn("curve_dim: int = 8", transformer)
        self.assertIn("TEXT_ENCODER_LAYERS = 50", conditioner)
        self.assertIn("class MiniMaxH3Int8Embedding", conditioner)
        self.assertIn("pre_quant_scale", conditioner)
        self.assertIn("self.model.norm = nn.Identity()", conditioner)
        self.assertIn("attention_mask=attention_mask,", conditioner)
        self.assertIn("native causal attention", conditioner)
        self.assertIn("dtype=torch.float32", transformer)
        self.assertIn('if qkv_layout == "interleaved"', main)
        self.assertIn("else 'fused projection'", main)

    def test_compact_vae_adapters_and_nvfp4_awq_scale_are_present(self):
        checkpoint = _read(_CHECKPOINT_PATH)
        nvfp4 = _read(_NVFP4_PATH)
        self.assertIn("_reorder_interleaved_qkv", checkpoint)
        self.assertIn("weight_g", checkpoint)
        self.assertIn("weight_v", checkpoint)
        self.assertIn('qmodule.register_buffer(\n                "pre_quant_scale"', nvfp4)
        self.assertIn("input = input * pre_quant_scale.to", nvfp4)

    def test_full_h3_convrot_quantization_handler_is_registered(self):
        wgp = _read(_WGP_PATH)
        convrot = _read(_INT8_CONVROT_PATH)
        self.assertIn('"shared.qtypes.int8_convrot"', wgp)
        self.assertIn('HANDLER_NAME = "int8_convrot"', convrot)
        self.assertIn("class QLinearInt8ConvRot", convrot)
        self.assertIn('split_handlers={"weight._scale": _split_scale}', convrot)
        self.assertNotIn('"weight._data": _split_weight_data', convrot)

    def test_conditioner_loader_preserves_mixed_quantization_contract(self):
        main = _read(_MAIN_PATH)
        checkpoint = _read(_CHECKPOINT_PATH)
        self.assertIn("_normalize_conditioner_checkpoint_namespaces", main)
        self.assertIn('if variant == "nvfp4_awq":', main)
        self.assertIn("state_dict = preprocess_conditioner_state_dict(state_dict)", main)
        self.assertIn("preprocess_sd=preprocess_checkpoint", main)
        self.assertIn("consumer_quantized=variant == \"nvfp4_awq\"", main)
        self.assertIn("qwen.model._model_dtype = dtype", main)
        self.assertIn("qwen.visual._model_dtype = dtype", main)
        self.assertIn("with init_empty_weights(include_buffers=False):", main)
        self.assertIn('descriptor.get("format") != "int8_tensorwise"', checkpoint)
        self.assertIn('state_dict.pop(f"{prefix}.comfy_quant", None)', checkpoint)

    def test_h3_loaders_materialize_nonpersistent_runtime_buffers(self):
        main = _read(_MAIN_PATH)
        video_loader = main.split("def _load_video_vae", 1)[1].split("def _load_audio_vae", 1)[0]
        audio_loader = main.split("def _load_audio_vae", 1)[1].split("class MiniMaxH3Model", 1)[0]
        self.assertIn("init_empty_weights(include_buffers=False)", video_loader)
        self.assertIn("init_empty_weights(include_buffers=False)", audio_loader)

    def test_upstream_provenance_is_recorded(self):
        provenance = _read(_APP / "models" / "minimax_h3" / "UPSTREAM.md")
        self.assertIn("abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc", provenance)
        self.assertIn("5d9b308a59ab12e67147f191e184baf704185bd1", provenance)
        self.assertIn("0543966fbdce5ba05709a8f2031c94bdba629b4a", provenance)
        self.assertIn("fec7846aef352e58a1cfb699455e3d104281e68b", provenance)
        self.assertIn("4ed4c744a396e43294f851f35cab769e11a89f2d", provenance)
        self.assertIn("b382d0940cdbab29cff5d33301b34b337ad5517e", provenance)
        self.assertIn("5c8b4ac3c5e15135b6510d9b6d4d57002e4bb5e4", provenance)
        self.assertIn("639ee1351e5b57c5992903690199719607c3700e", provenance)
        self.assertIn("Apache-2.0", provenance)


class TestMiniMaxH3LoraBrowserRouting(unittest.TestCase):
    def test_civitai_filter_and_base_mapping_target_shared_h3_directory(self):
        civit_map = _literal_assignment(_LAUNCH_PATH, "CIVIT_TO_LOCAL_ARCH")
        hf_map = _literal_assignment(_LAUNCH_PATH, "HF_BASE_TO_LOCAL_DIR")
        filters = _literal_assignment(_LAUNCH_PATH, "CIVITAI_MODEL_FILTERS")

        self.assertEqual(civit_map["MiniMax H3"], "minimax_h3")
        self.assertEqual(hf_map["MiniMaxAI/MiniMax-H3"], "minimax_h3")
        self.assertEqual(hf_map["Comfy-Org/MiniMax-H3"], "minimax_h3")
        self.assertIn(
            {
                "label": "MiniMax H3",
                "civitai_base": "MiniMax H3",
                "default_dir": "minimax_h3",
            },
            filters,
        )

    def test_h3_identity_detection_accepts_current_metadata_variants_only(self):
        helpers = _load_minimax_h3_lora_routing_helpers()
        is_h3 = helpers["_is_minimax_h3_identity"]
        civit_arch = helpers["_civitai_lora_arch"]

        for value in (
            "MiniMax H3",
            "MiniMaxAI/MiniMax-H3",
            "base_model:adapter:Comfy-Org/MiniMax-H3",
            "minimax-h3",
            "minimax_h3",
        ):
            with self.subTest(value=value):
                self.assertTrue(is_h3(value))
                self.assertEqual(civit_arch(value), "minimax_h3")

        for value in ("H3", "MiniMax M3", "Hunyuan Video", "LTX-2.3"):
            with self.subTest(value=value):
                self.assertFalse(is_h3(value))
        self.assertEqual(civit_arch("LTXV 2.3"), "ltx2")

    def test_browser_and_pasted_url_flows_use_canonical_h3_routing(self):
        launch = _read(_LAUNCH_PATH)
        self.assertIn("arch = _civitai_lora_arch(base)", launch)
        self.assertIn("inferred_target_arch = _civitai_lora_arch(base_model)", launch)
        self.assertIn("target_arch = _civitai_lora_arch(base_model)", launch)
        self.assertIn("if _is_minimax_h3_identity(_identity_blob):", launch)
        self.assertIn('hf_base_label = "MiniMax H3 (detected from repo name/tags)"', launch)


_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "diffusers", "transformers")
)


@unittest.skipUnless(_RUNTIME_AVAILABLE, "MiniMax H3 runtime dependencies are not installed")
class TestMiniMaxH3RuntimeMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_APP))
        import torch

        cls.torch = torch

    @classmethod
    def tearDownClass(cls):
        if sys.path and sys.path[0] == str(_APP):
            sys.path.pop(0)

    def test_fl2va_overlap_splits_motion_history_and_boundary_frame(self):
        from models.minimax_h3.minimax_h3_main import (
            _build_frozen_control_video,
            _prepare_stereo_waveform,
            _split_continuation_video,
        )

        video = self.torch.arange(3 * 35 * 2 * 2).reshape(3, 35, 2, 2)
        history, boundary, count = _split_continuation_video(video, 20)
        self.assertEqual(count, 18)
        self.assertEqual(tuple(history.shape), (3, 17, 2, 2))
        self.assertTrue(self.torch.equal(history, video[:, -18:-1]))
        self.assertTrue(self.torch.equal(boundary, video[:, -1:]))

        no_history = _split_continuation_video(video, 0)
        self.assertEqual(no_history, (None, None, 0))
        explicit = _split_continuation_video(
            video,
            18,
            has_explicit_start=True,
        )
        self.assertEqual(explicit, (None, None, 0))

        sample_major = self.torch.stack(
            [self.torch.arange(100), self.torch.arange(100) + 100],
            dim=1,
        ).float()
        stereo = _prepare_stereo_waveform(sample_major, 32000, 120)
        self.assertEqual(tuple(stereo.shape), (2, 120))
        self.assertTrue(self.torch.equal(stereo[0, :100], sample_major[:, 0]))
        self.assertTrue(self.torch.equal(stereo[1, :100], sample_major[:, 1]))
        self.assertEqual(float(stereo[:, 100:].abs().sum()), 0.0)
        unpadded = _prepare_stereo_waveform(
            sample_major,
            32000,
            120,
            pad=False,
        )
        self.assertEqual(tuple(unpadded.shape), (2, 100))

        prefix = self.torch.full((3, 3, 2, 2), -0.5)
        control = self.torch.arange(3 * 5 * 2 * 2).reshape(3, 5, 2, 2)
        control = control.float().div(control.max()).mul(2).sub(1)
        frozen = _build_frozen_control_video(
            control,
            prefix,
            frame_num=9,
            prefix_frames_count=2,
            height=2,
            width=2,
        )
        self.assertEqual(tuple(frozen.shape), (3, 9, 2, 2))
        self.assertTrue(self.torch.equal(frozen[:, :2], prefix[:, -2:]))
        self.assertTrue(self.torch.equal(frozen[:, 2:7], control))
        self.assertTrue(
            self.torch.equal(
                frozen[:, 7:],
                control[:, -1:].repeat(1, 2, 1, 1),
            )
        )
        combined_guide = self.torch.cat([prefix[:, -2:], control], dim=1)
        deduplicated = _build_frozen_control_video(
            combined_guide,
            prefix,
            frame_num=7,
            prefix_frames_count=2,
            height=2,
            width=2,
        )
        self.assertTrue(self.torch.equal(deduplicated, combined_guide))

    def test_video_to_video_mask_mapping_and_source_reinjection(self):
        from models.minimax_h3.minimax_h3_main import (
            _reinject_video_source,
            _resize_video_mask,
        )

        mask = self.torch.zeros((1, 5, 4, 4), dtype=self.torch.float32)
        mask[:, :2] = 1.0
        latent_mask = _resize_video_mask(
            mask,
            latent_shape=(3, 2, 2),
            clip_length=17,
            temporal_ratio=4,
        )
        self.assertEqual(tuple(latent_mask.shape), (1, 1, 3, 2, 2))
        self.assertTrue(self.torch.all(latent_mask[:, :, 0] == 1))
        self.assertTrue(self.torch.all(latent_mask[:, :, 1] == 1))
        self.assertTrue(self.torch.all(latent_mask[:, :, 2] == 0))

        video_rows = self.torch.zeros((2, 1), dtype=self.torch.float32)
        source_rows = self.torch.zeros_like(video_rows)
        source_noise = self.torch.ones_like(video_rows)
        editable = self.torch.tensor([[1.0], [0.0]])
        buffer = self.torch.empty_like(video_rows)
        _reinject_video_source(
            video_rows,
            source_rows,
            source_noise,
            editable,
            0.5,
            buffer,
        )
        self.assertEqual(float(video_rows[0]), 0.0)
        self.assertEqual(float(video_rows[1]), 0.5)

    def test_packed_sequence_places_visual_and_audio_history_before_target(self):
        from models.minimax_h3.packing import (
            MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG,
            build_packed_sequence,
            build_row_timesteps,
        )

        layout = build_packed_sequence(
            self.torch.tensor([1, 1]),
            num_latent_frames=7,
            latent_height=4,
            latent_width=4,
            num_audio_latents=4,
            patch_size=(1, 2, 2),
            keyframe_anchors=(("history", 5), ("first", 1)),
            audio_condition_anchors=(("history", 27), ("first", 2)),
        )
        self.assertEqual(layout.sequence_length, 120)
        self.assertEqual(layout.num_condition_video_rows, 24)
        self.assertEqual(layout.num_condition_audio_rows, 58)
        self.assertEqual(int(layout.video_indices[0]), 2)
        self.assertEqual(int(layout.video_indices[24]), 92)
        self.assertEqual(int(layout.audio_indices[0]), 26)
        self.assertEqual(int(layout.audio_indices[58]), 84)
        self.assertTrue(
            self.torch.all(
                layout.token_tags[layout.video_indices] == MINIMAX_H3_VIDEO_TAG
            )
        )
        self.assertTrue(
            self.torch.all(
                layout.token_tags[layout.audio_indices] == MINIMAX_H3_AUDIO_TAG
            )
        )
        target_origin = layout.position_ids[92, 0]
        self.assertAlmostEqual(
            float(layout.position_ids[84, 0]),
            float(target_origin),
        )
        self.assertGreater(float(target_origin), 2.0)

        unique, inverse = build_row_timesteps(
            layout,
            video_timestep=0.5,
            audio_timestep=0.25,
            condition_video_timestep=0.999,
            condition_audio_timestep=1.0,
        )
        self.assertTrue(
            self.torch.allclose(
                unique,
                self.torch.tensor([0.25, 0.5, 0.999, 1.0]),
            )
        )
        self.assertEqual(inverse.shape[0], layout.sequence_length)

        frozen_layout = build_packed_sequence(
            self.torch.tensor([1, 1]),
            num_latent_frames=7,
            latent_height=4,
            latent_width=4,
            num_audio_latents=4,
            patch_size=(1, 2, 2),
            target_condition_audio_latents=2,
            target_condition_video_frames=7,
        )
        frozen_unique, frozen_inverse = build_row_timesteps(
            frozen_layout,
            video_timestep=0.5,
            audio_timestep=0.25,
            condition_video_timestep=0.999,
            condition_audio_timestep=1.0,
        )
        frozen_rows = frozen_unique[frozen_inverse]
        target_audio = frozen_layout.audio_indices
        self.assertTrue(self.torch.all(frozen_rows[target_audio[:2]] == 1.0))
        self.assertTrue(self.torch.all(frozen_rows[target_audio[4:6]] == 1.0))
        self.assertTrue(
            self.torch.all(frozen_rows[frozen_layout.video_indices] == 1.0)
        )

    def test_ref2va_manifest_limits_and_visual_reference_requirement(self):
        from models.minimax_h3.ref2va import validate_reference_manifest

        manifest = validate_reference_manifest(
            [
                {"type": "image", "path": "portrait.png", "role": "Lead actor"},
                {"type": "audio", "path": "voice.wav", "role": "Lead voice"},
                {
                    "type": "video",
                    "path": "movement.mp4",
                    "audio_path": "replacement-voice.wav",
                    "include_audio": True,
                    "role": "Movement reference",
                },
            ],
            require_files=False,
        )
        self.assertEqual([item["type"] for item in manifest], ["image", "audio", "video"])
        self.assertEqual(manifest[0]["role"], "Lead actor")
        self.assertEqual(manifest[0]["image_intent"], "identity")
        self.assertEqual(manifest[1]["audio_intent"], "voice")
        self.assertTrue(manifest[2]["include_audio"])
        self.assertEqual(manifest[2]["audio_path"], "replacement-voice.wav")

        with self.assertRaisesRegex(ValueError, "cannot be used alone"):
            validate_reference_manifest(
                [{"type": "audio", "path": "voice.wav"}],
                require_files=False,
            )
        with self.assertRaisesRegex(ValueError, "at most 9 image"):
            validate_reference_manifest(
                [{"type": "image", "path": f"portrait-{index}.png"} for index in range(10)],
                require_files=False,
            )
        with self.assertRaisesRegex(ValueError, "invalid audio intent"):
            validate_reference_manifest(
                [
                    {"type": "image", "path": "portrait.png"},
                    {"type": "audio", "path": "voice.wav", "audio_intent": "mystery"},
                ],
                require_files=False,
            )
        with self.assertRaisesRegex(ValueError, "invalid image intent"):
            validate_reference_manifest(
                [
                    {
                        "type": "image",
                        "path": "portrait.png",
                        "image_intent": "mystery",
                    },
                ],
                require_files=False,
            )

    def test_ref2va_music_references_advance_while_voice_reuses_its_start(self):
        from PIL import Image

        from models.minimax_h3.ref2va import prepare_references

        waveform = self.torch.arange(60, dtype=self.torch.float32).reshape(1, -1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "person.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            audio_paths = []
            for name in ("timeline.wav", "style.wav", "voice.wav"):
                path = root / name
                path.write_bytes(b"reference")
                audio_paths.append(path)

            with mock.patch(
                "models.minimax_h3.ref2va.decode_reference_audio",
                return_value=(waveform, 10),
            ):
                references = prepare_references(
                    [
                        {"type": "image", "path": str(image_path)},
                        {
                            "type": "audio",
                            "path": str(audio_paths[0]),
                            "audio_intent": "drive",
                        },
                        {
                            "type": "audio",
                            "path": str(audio_paths[1]),
                            "audio_intent": "style",
                        },
                        {
                            "type": "audio",
                            "path": str(audio_paths[2]),
                            "audio_intent": "voice",
                        },
                    ],
                    num_frames=24,
                    target_height=32,
                    target_width=32,
                    audio_sample_rate=10,
                    timeline_start_frame=48,
                )

        expected_timeline = self.torch.arange(20, 30, dtype=self.torch.float32)
        expected_voice = self.torch.arange(0, 10, dtype=self.torch.float32)
        self.assertTrue(self.torch.equal(references[1].waveform[0], expected_timeline))
        self.assertTrue(self.torch.equal(references[2].waveform[0], expected_timeline))
        self.assertTrue(self.torch.equal(references[3].waveform[0], expected_voice))
        self.assertTrue(self.torch.equal(references[1].waveform[0], references[1].waveform[1]))

    def test_ref2va_music_timeline_does_not_restart_after_reference_ends(self):
        from models.minimax_h3.ref2va import prepare_reference_waveform

        waveform = self.torch.arange(25, dtype=self.torch.float32).reshape(1, -1)
        segment = prepare_reference_waveform(
            waveform,
            sample_rate=10,
            target_sample_rate=10,
            max_duration=1.0,
            start_time=2.0,
            pad_to_duration=True,
        )
        self.assertEqual(tuple(segment.shape), (2, 10))
        self.assertTrue(
            self.torch.equal(
                segment[0],
                self.torch.tensor([20, 21, 22, 23, 24, 0, 0, 0, 0, 0], dtype=self.torch.float32),
            )
        )

    def test_ref2va_raw_prompt_gets_explicit_audio_semantics(self):
        from models.minimax_h3.ref2va import ensure_ref2va_prompt_relationships

        prompt = 'Blaine says, "Snap this." while fighting a villain.'
        voice = ensure_ref2va_prompt_relationships(
            prompt,
            [
                {"type": "image", "path": "blaine.png", "role": "Blaine"},
                {
                    "type": "audio",
                    "path": "blaine.wav",
                    "role": "Blaine",
                    "audio_intent": "voice",
                },
            ],
            duration_seconds=10,
        )
        self.assertIn("<Picture 1>", voice)
        self.assertIn("<Audio 1>", voice)
        self.assertIn("voice-timbre", voice)
        self.assertIn("do not copy its source words", voice)
        self.assertIn("begin at the first frame", voice)
        self.assertIn("subject_definitions:", voice)
        self.assertIn("detailed_description:", voice)
        self.assertIn("<d>[English] Snap this.</d>", voice)
        self.assertNotIn('"Snap this."', voice)
        self.assertIn("source location, background, composition, framing, or pose", voice)
        self.assertIn("only spoken words", voice)
        self.assertIn("From 0.00 to 2.00 seconds", voice)
        self.assertIn("From 3.00 to 10.00 seconds", voice)

        drive = ensure_ref2va_prompt_relationships(
            "A singer performs.",
            [
                {"type": "image", "path": "singer.png"},
                {"type": "audio", "path": "song.wav", "audio_intent": "drive"},
            ],
        )
        self.assertIn("performance-driving audio timeline", drive)
        self.assertIn("reuse its audible content", drive)

        director_images = ensure_ref2va_prompt_relationships(
            "Two characters cross the room.",
            [
                {
                    "type": "image",
                    "path": "shot.png",
                    "role": "the planned shot",
                    "image_intent": "composition",
                },
                {
                    "type": "image",
                    "path": "room.png",
                    "role": "the dojo",
                    "image_intent": "scene",
                },
            ],
        )
        self.assertIn("soft composition and cast-layout reference", director_images)
        self.assertIn("environment and location", director_images)
        self.assertIn("rather than copying the picture as a frozen first frame", director_images)

        tagged = "<Picture 1> defines <Subject 1>."
        self.assertEqual(
            ensure_ref2va_prompt_relationships(
                tagged,
                [{"type": "image", "path": "singer.png"}],
            ),
            tagged,
        )

    def test_ref2va_reference_detail_policy_is_bounded_and_grid_aligned(self):
        from models.minimax_h3.ref2va import (
            resolve_reference_image_size,
            resolve_reference_video_size,
        )

        matched = resolve_reference_image_size(
            900,
            1600,
            detail="match",
            target_height=480,
            target_width=864,
        )
        maximum = resolve_reference_image_size(
            900,
            1600,
            detail="max",
            target_height=480,
            target_width=864,
        )
        self.assertEqual(matched, (864, 480))
        self.assertEqual(maximum, (3648, 2048))
        self.assertTrue(all(size % 32 == 0 for size in matched + maximum))

        matched_video = resolve_reference_video_size(
            1272,
            720,
            detail="match",
            target_height=544,
            target_width=960,
        )
        maximum_video = resolve_reference_video_size(
            1272,
            720,
            detail="max",
            target_height=544,
            target_width=960,
        )
        self.assertEqual(matched_video, (544, 960))
        self.assertEqual(maximum_video, (768, 1344))
        self.assertLessEqual(
            matched_video[0] * matched_video[1],
            544 * 960,
        )

    def test_transformer_supports_full_and_pruned_modulation_and_split_qkv(self):
        from models.minimax_h3.transformer import (
            MiniMaxH3Transformer,
            get_linear_split_map,
        )

        common = {
            "hidden_size": 32,
            "num_layers": 1,
            "token_refiner_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "ffn_dim": 64,
            "video_channels": 4,
            "audio_channels": 4,
            "text_dim": 16,
            "timestep_input_dim": 8,
            "time_embed_hidden_size": 32,
            "rope_freq_dim": 2,
            "dtype": self.torch.float32,
        }
        pruned = MiniMaxH3Transformer(curve_grid=17, curve_dim=8, **common)
        full = MiniMaxH3Transformer(curve_grid=None, curve_dim=32, **common)
        self.assertTrue(pruned.use_adaln_curves)
        self.assertTrue(hasattr(pruned, "adaln_t_table"))
        self.assertFalse(hasattr(pruned, "time_embedder"))
        self.assertFalse(full.use_adaln_curves)
        self.assertTrue(hasattr(full, "time_embedder"))
        self.assertFalse(hasattr(full, "adaln_t_table"))

        contiguous = get_linear_split_map(4)
        contiguous_handler = contiguous["qkv_proj"]["split_handlers"]["weight"]
        contiguous_source = self.torch.arange(12, dtype=self.torch.float32).view(12, 1)
        contiguous_parts = contiguous_handler(
            contiguous_source,
            0,
            [4, 4, 4],
            {"info": contiguous["qkv_proj"]},
        )
        self.assertEqual(
            [part.flatten().tolist() for part in contiguous_parts],
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
        )
        self.assertEqual(
            len({part.untyped_storage().data_ptr() for part in contiguous_parts}),
            3,
        )
        self.assertNotEqual(
            contiguous_parts[0].untyped_storage().data_ptr(),
            contiguous_source.untyped_storage().data_ptr(),
        )
        interleaved = get_linear_split_map(
            4,
            interleaved=True,
            num_attention_heads=2,
            attention_head_dim=2,
        )
        handler = interleaved["qkv_proj"]["split_handlers"]["weight"]
        source = self.torch.arange(12, dtype=self.torch.float32).view(12, 1)
        query, key, value = handler(
            source,
            0,
            [4, 4, 4],
            {"info": interleaved["qkv_proj"]},
        )
        self.assertEqual(query.flatten().tolist(), [0.0, 1.0, 6.0, 7.0])
        self.assertEqual(key.flatten().tolist(), [2.0, 3.0, 8.0, 9.0])
        self.assertEqual(value.flatten().tolist(), [4.0, 5.0, 10.0, 11.0])
        self.assertEqual(
            len({part.untyped_storage().data_ptr() for part in (query, key, value)}),
            3,
        )

    def test_ref2va_presentation_labels_follow_manifest_order(self):
        from models.minimax_h3.ref2va import (
            MiniMaxH3PreparedReference,
            build_ref2va_presentation,
        )

        class RecordingTokenizer:
            def __init__(self):
                self.segments = []

            def __call__(self, value, add_special_tokens=False):
                self.segments.append(value)
                return {"input_ids": [1000 + len(self.segments)]}

            @staticmethod
            def convert_tokens_to_ids(value):
                return {
                    "<|vision_start|>": 10,
                    "<|vision_end|>": 11,
                    "<|image_pad|>": 12,
                    "<|video_pad|>": 13,
                }[value]

        tokenizer = RecordingTokenizer()
        references = [
            MiniMaxH3PreparedReference(kind="image"),
            MiniMaxH3PreparedReference(
                kind="video",
                has_audio=True,
                block_timestamps=[0.25, 0.75],
            ),
            MiniMaxH3PreparedReference(kind="audio", has_audio=True),
        ]
        token_ids, token_tags = build_ref2va_presentation(
            tokenizer,
            "A finished scene.",
            references,
            image_token_counts=[2],
            video_block_token_counts=[3],
        )
        self.assertEqual(
            tokenizer.segments,
            [
                "<Picture 1>: ",
                "<Audio 1>: ",
                "<Video 1>: ",
                "<0.2 seconds>",
                "<0.8 seconds>",
                "<Audio 2>: ",
                "A finished scene.",
            ],
        )
        self.assertEqual(len(token_ids), len(token_tags))
        self.assertGreater(token_tags.count(0), 0)

    def test_ref2va_layout_keeps_ordered_condition_rows_before_targets(self):
        from models.minimax_h3.ref2va import (
            MiniMaxH3PreparedReference,
            build_ref2va_packed_sequence,
        )

        references = [
            MiniMaxH3PreparedReference(
                kind="image",
                num_latent_frames=1,
                latent_height=4,
                latent_width=4,
            ),
            MiniMaxH3PreparedReference(kind="audio", has_audio=True, num_audio_latents=3),
            MiniMaxH3PreparedReference(
                kind="video",
                has_audio=True,
                num_latent_frames=2,
                latent_height=4,
                latent_width=4,
                num_audio_latents=2,
            ),
        ]
        packed = build_ref2va_packed_sequence(
            self.torch.tensor([1, 1]),
            references,
            num_latent_frames=2,
            latent_height=4,
            latent_width=4,
            num_audio_latents=3,
            patch_size=(1, 2, 2),
        )
        self.assertEqual(packed.sequence_length, 38)
        self.assertEqual(packed.num_condition_video_rows, 12)
        self.assertEqual(packed.num_condition_audio_rows, 10)
        self.assertEqual(packed.text_indices.tolist(), [0, 1])
        self.assertEqual(packed.video_indices.tolist(), list(range(2, 6)) + list(range(16, 24)) + list(range(30, 38)))
        self.assertEqual(packed.audio_indices.tolist(), list(range(6, 16)) + list(range(24, 30)))
        self.assertEqual(packed.token_tags[packed.video_indices].unique().tolist(), [0])
        self.assertEqual(packed.token_tags[packed.audio_indices].unique().tolist(), [2])

    def test_ref2va_layout_packs_continuation_before_canonical_references(self):
        from models.minimax_h3.ref2va import (
            MiniMaxH3PreparedReference,
            build_ref2va_packed_sequence,
        )

        references = [
            MiniMaxH3PreparedReference(
                kind="image",
                num_latent_frames=1,
                latent_height=4,
                latent_width=4,
            ),
        ]
        packed = build_ref2va_packed_sequence(
            self.torch.tensor([1, 1]),
            references,
            num_latent_frames=7,
            latent_height=4,
            latent_width=4,
            num_audio_latents=4,
            patch_size=(1, 2, 2),
            keyframe_anchors=(("history", 5), ("first", 1)),
            audio_condition_anchors=(("history", 27), ("first", 2)),
        )
        self.assertEqual(packed.sequence_length, 124)
        self.assertEqual(packed.num_condition_video_rows, 28)
        self.assertEqual(packed.num_condition_audio_rows, 58)
        self.assertEqual(int(packed.video_indices[0]), 2)
        self.assertEqual(int(packed.video_indices[24]), 84)
        self.assertEqual(int(packed.video_indices[28]), 96)
        self.assertEqual(int(packed.audio_indices[0]), 26)
        self.assertEqual(int(packed.audio_indices[58]), 88)
        # Canonical-reference rotary time comes first; carried history then
        # advances the target origin while remaining physically packed ahead
        # of those reference rows for the model's video/audio token arrays.
        self.assertLess(
            float(packed.position_ids[84, 0]),
            float(packed.position_ids[2, 0]),
        )
        self.assertLess(
            float(packed.position_ids[2, 0]),
            float(packed.position_ids[96, 0]),
        )

    def test_ref2va_continuation_reserves_picture_one_without_renumbering_audio(self):
        from models.minimax_h3.ref2va import add_ref2va_continuation_context

        prompt = (
            "subject_definitions: <Picture 1> defines Alex and <Audio 1> defines the voice.\n"
            "summary: Alex continues running.\n"
            "retention_analysis: <Picture 1>: identity; <Audio 1>: reference\n"
            "detailed_description: <Picture 1> remains Alex.\n"
            "overall_soundscape: Street ambience.\n"
            "non_diegetic_music: N/A"
        )
        shifted = add_ref2va_continuation_context(prompt)
        self.assertIn("<Picture 1> is the exact final frame", shifted)
        self.assertIn("<Picture 2> defines Alex", shifted)
        self.assertIn("<Picture 2>: identity", shifted)
        self.assertEqual(shifted.count("<Audio 1>"), 2)
        self.assertNotIn("<Audio 2>", shifted)

    def test_video_patch_round_trip_and_scheduler_length(self):
        from models.minimax_h3.packing import patchify_video_latents, unpatchify_video_tokens
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler

        source = self.torch.arange(1 * 2 * 3 * 4 * 6, dtype=self.torch.float32).reshape(1, 2, 3, 4, 6)
        rows = patchify_video_latents(source, (1, 2, 2))
        restored = unpatchify_video_tokens(rows, 3, 4, 6, 2, (1, 2, 2))
        self.assertTrue(self.torch.equal(source, restored))

        scheduler = MiniMaxH3Scheduler(shift=12.0)
        scheduler.set_timesteps(20, device="cpu")
        self.assertEqual(len(scheduler.sigmas), 20)
        self.assertEqual(len(scheduler.timesteps), 19)
        self.assertEqual(float(scheduler.timesteps[0]), 0.0)
        self.assertEqual(float(scheduler.sigmas[-1]), 0.0)

    def test_keyframe_normalization_stays_on_cpu_with_non_cpu_default_device(self):
        from models.minimax_h3.minimax_h3_main import _keyframe_latent_stats_cpu

        previous_device = self.torch.get_default_device()
        try:
            # Maestro runs with a CUDA default device. ``meta`` reproduces the
            # constructor-routing behavior without requiring a GPU in CI.
            self.torch.set_default_device("meta")
            means, stds = _keyframe_latent_stats_cpu()
        finally:
            self.torch.set_default_device(previous_device)

        self.assertEqual(means.device.type, "cpu")
        self.assertEqual(stds.device.type, "cpu")
        self.assertEqual(tuple(means.shape), (1, 24, 1, 1, 1))
        self.assertEqual(tuple(stds.shape), (1, 24, 1, 1, 1))
        self.assertEqual(means.dtype, self.torch.float32)
        self.assertEqual(stds.dtype, self.torch.float32)

    def test_tiny_joint_transformer_forward(self):
        from models.minimax_h3.transformer import MiniMaxH3Transformer

        model = MiniMaxH3Transformer(
            hidden_size=8,
            num_layers=1,
            token_refiner_layers=1,
            num_attention_heads=1,
            attention_head_dim=8,
            ffn_dim=12,
            video_channels=2,
            audio_channels=3,
            patch_size=(1, 1, 1),
            text_dim=6,
            curve_grid=4,
            curve_dim=2,
            rope_freq_dim=1,
            dtype=self.torch.float32,
        ).eval()
        # Production weights replace this table from the checkpoint.  The
        # tiny model has no checkpoint, so initialize its empty placeholder
        # to keep the numerical smoke test deterministic.
        model.adaln_t_table.data.zero_()
        self.assertEqual(model.video_patch_proj._lock_dtype, self.torch.float32)
        self.assertEqual(model.audio_patch_proj._lock_dtype, self.torch.float32)
        self.assertEqual(model.blocks[0].adaln_proj.linear._lock_dtype, self.torch.float16)
        self.assertEqual(model.final_layer.adaln_proj.linear._lock_dtype, self.torch.float16)
        self.assertEqual(model.final_layer.video_out._lock_dtype, self.torch.float32)
        self.assertEqual(model.final_layer.audio_out._lock_dtype, self.torch.float32)
        video_rows = self.torch.randn(1, 3, 2)
        audio_rows = self.torch.randn(1, 4, 3)
        text_rows = self.torch.randn(1, 2, 6)
        position_ids = self.torch.zeros(9, 3, dtype=self.torch.float64)
        token_tags = self.torch.tensor([1, 1, 2, 2, 2, 2, 0, 0, 0])
        timestep_indices = self.torch.tensor([0, 0, 1, 1, 1, 1, 0, 0, 0])
        video, audio = model(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=text_rows,
            timestep=self.torch.tensor([0.1, 0.4]),
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=self.torch.tensor([6, 7, 8]),
            audio_indices=self.torch.tensor([2, 3, 4, 5]),
            text_indices=self.torch.tensor([0, 1]),
            return_dict=False,
        )
        self.assertEqual(tuple(video.shape), (1, 3, 2))
        self.assertEqual(tuple(audio.shape), (1, 4, 3))
        self.assertTrue(self.torch.isfinite(video).all())
        self.assertTrue(self.torch.isfinite(audio).all())

    def test_chunked_h3_projections_match_unchunked_math(self):
        import models.minimax_h3.transformer as h3_transformer

        attention = h3_transformer.MiniMaxH3Attention(8, 1, 8, 1e-5, self.torch.float32).eval()
        mlp = h3_transformer.MiniMaxH3MLP(8, 12, self.torch.float32).eval()
        hidden = self.torch.randn(1, 7, 8)
        positions = self.torch.zeros(7, 3)
        rotary = h3_transformer.MiniMaxH3RotaryEmbedding(1)(positions)

        previous = h3_transformer.MINIMAX_H3_ACTIVATION_CHUNK_TOKENS
        try:
            with self.torch.inference_mode():
                h3_transformer.MINIMAX_H3_ACTIVATION_CHUNK_TOKENS = 64
                expected_attention = attention(hidden, rotary)
                expected_mlp = mlp(hidden.clone())
                h3_transformer.MINIMAX_H3_ACTIVATION_CHUNK_TOKENS = 2
                actual_attention = attention(hidden, rotary)
                chunked_mlp_input = hidden.clone()
                actual_mlp = mlp(chunked_mlp_input)
        finally:
            h3_transformer.MINIMAX_H3_ACTIVATION_CHUNK_TOKENS = previous

        self.assertTrue(self.torch.allclose(actual_attention, expected_attention, atol=1e-5, rtol=1e-5))
        self.assertTrue(self.torch.allclose(actual_mlp, expected_mlp, atol=1e-5, rtol=1e-5))
        self.assertEqual(actual_mlp.data_ptr(), chunked_mlp_input.data_ptr())

    def test_h3_projection_chunks_expand_only_below_large_sequence_guard(self):
        from models.minimax_h3.transformer import _activation_chunk_tokens

        qkv_chunk = _activation_chunk_tokens(60_000, 5_376, 21_504)
        mlp_chunk = _activation_chunk_tokens(60_000, 5_376, 28_672)
        self.assertGreater(qkv_chunk, 8_192)
        self.assertGreater(mlp_chunk, 8_192)
        self.assertLessEqual(qkv_chunk, 32_768)
        self.assertLessEqual(mlp_chunk, 32_768)
        self.assertEqual(qkv_chunk % 256, 0)
        self.assertEqual(mlp_chunk % 256, 0)
        self.assertEqual(
            _activation_chunk_tokens(91_278, 5_376, 21_504),
            8_192,
        )
        self.assertEqual(
            _activation_chunk_tokens(91_278, 5_376, 28_672),
            8_192,
        )

    def test_h3_first_block_cache_reuses_only_after_warmup(self):
        from models.minimax_h3.first_block_cache import MiniMaxH3FirstBlockCache

        config = types.SimpleNamespace(
            threshold=0.08,
            start_step=1,
            skipped_steps=0,
        )
        cache = MiniMaxH3FirstBlockCache(config)
        signature = self.torch.tensor([1.0, 2.0])
        cache.begin_step(0)
        self.assertTrue(cache.should_compute(signature.clone()))
        head = self.torch.tensor([[1.0, 2.0]])
        captured = cache.capture_head_output(head)
        cache.store_tail_residual(
            self.torch.tensor([[4.0, 6.0]]),
            captured,
        )

        cache.begin_step(1)
        self.assertFalse(cache.should_compute(signature.clone()))
        reused = self.torch.tensor([[10.0, 10.0]])
        cache.apply_tail_residual(reused)
        self.assertTrue(
            self.torch.equal(reused, self.torch.tensor([[13.0, 14.0]]))
        )
        self.assertEqual(config.skipped_steps, 1)

    def test_h3_attention_accepts_owned_input_and_releases_the_holder(self):
        from models.minimax_h3.transformer import MiniMaxH3Attention

        attention = MiniMaxH3Attention(8, 1, 8, 1e-5, self.torch.float32).eval()
        owned = [self.torch.randn(1, 4, 8)]
        with self.torch.inference_mode():
            output = attention(owned)
        self.assertEqual(owned, [])
        self.assertEqual(tuple(output.shape), (1, 4, 8))

    def test_curve_adaln_uses_fp32_math_with_compact_fp16_storage(self):
        from models.minimax_h3.transformer import MiniMaxH3AdaLNProjection

        projection = MiniMaxH3AdaLNProjection(2, 2, 2, 1, self.torch.float16).eval()
        projection.linear.weight.data.copy_(
            self.torch.tensor(
                [[0.3333, -1.777], [2.125, 0.03125], [-0.8125, 1.333], [3.141, -2.718]],
                dtype=self.torch.float16,
            )
        )
        projection.linear.bias.data.copy_(
            self.torch.tensor([0.125, -0.25, 0.375, -0.5], dtype=self.torch.float16)
        )
        curve = self.torch.tensor([[0.12345, -0.98765]], dtype=self.torch.float32)
        chunks = projection(curve)
        actual = self.torch.cat(chunks, dim=-1)
        expected = self.torch.nn.functional.linear(
            curve,
            projection.linear.weight.float(),
            projection.linear.bias.float(),
        )

        self.assertEqual(projection.linear.weight.dtype, self.torch.float16)
        self.assertEqual(actual.dtype, self.torch.float32)
        self.assertTrue(self.torch.equal(actual, expected))

    def test_full_adaln_uses_linear_module_forward_for_convrot_contract(self):
        from models.minimax_h3.transformer import MiniMaxH3AdaLNProjection

        torch = self.torch

        class ProbeLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                        dtype=torch.float32,
                    )
                )
                self.bias = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
                self.calls = 0

            def forward(self, rows):
                self.calls += 1
                # A visible sentinel makes this fail if AdaLN bypasses the
                # module and invokes F.linear on its raw weight instead.
                return torch.nn.functional.linear(rows, self.weight, self.bias) + 17.0

        projection = MiniMaxH3AdaLNProjection(
            2,
            2,
            2,
            1,
            torch.float32,
            apply_silu=True,
        ).eval()
        probe = ProbeLinear()
        projection.linear = probe
        curve = torch.tensor([[0.25, -0.5]], dtype=torch.float32)

        actual = torch.cat(projection(curve), dim=-1)
        expected = torch.nn.functional.linear(
            torch.nn.functional.silu(curve),
            probe.weight,
            probe.bias,
        ) + 17.0

        self.assertEqual(probe.calls, 1)
        self.assertTrue(torch.equal(actual, expected))

    def test_nvfp4_pre_quant_scale_loads_and_affects_forward(self):
        from models.minimax_h3.conditioner import MiniMaxH3PreScaledLinear
        from shared.qtypes.nvfp4 import QLinearNVFP4, _NVFP4_QTYPE

        source = MiniMaxH3PreScaledLinear(3, 2, bias=True, dtype=self.torch.float32)
        qmodule = QLinearNVFP4.qcreate(source, _NVFP4_QTYPE, device="cpu")
        qmodule.weight = self.torch.nn.Parameter(
            self.torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        )
        qmodule.bias = self.torch.nn.Parameter(self.torch.tensor([0.25, -0.5]))

        scale = self.torch.tensor([2.0, 3.0, 4.0])
        missing_keys, unexpected_keys, error_messages = [], [], []
        state_dict = {"pre_quant_scale": scale.clone()}
        qmodule._load_from_state_dict(
            state_dict,
            "",
            {},
            False,
            missing_keys,
            unexpected_keys,
            error_messages,
        )
        self.assertTrue(self.torch.equal(qmodule.pre_quant_scale, scale))
        self.assertNotIn("pre_quant_scale", state_dict)

        input_rows = self.torch.tensor([[1.0, 1.0, 1.0]])
        expected = self.torch.nn.functional.linear(
            input_rows * scale,
            qmodule.weight,
            qmodule.bias,
        )
        self.assertTrue(self.torch.equal(qmodule(input_rows), expected))

        # MMGP's quant router transfers ordinary handler attributes but omits
        # registered buffers. Simulate that transfer and prove the mirrored
        # scale still governs the routed forward path.
        self.assertTrue(self.torch.equal(qmodule._nvfp4_pre_quant_scale, scale))
        del qmodule._buffers["pre_quant_scale"]
        self.assertFalse(hasattr(qmodule, "pre_quant_scale"))
        self.assertTrue(self.torch.equal(qmodule(input_rows), expected))

    def test_nvfp4_fallback_matches_official_combined_scale_order(self):
        from shared.qtypes.nvfp4 import (
            _NVFP4_LAYOUT_TENSORCORE,
            _dequantize_nvfp4_weight,
        )

        # TensorCore scale tiles require 128 output rows and 64 input
        # channels at minimum.  0xFF decodes to two -6.0 FP4 values.
        packed_weight = self.torch.full((128, 32), 0xFF, dtype=self.torch.uint8)
        block_scale = self.torch.full(
            (128, 4),
            0.00099945068359375,
            dtype=self.torch.bfloat16,
        )
        tensor_scale = self.torch.tensor(0.0030059814453125, dtype=self.torch.float32)
        actual = _dequantize_nvfp4_weight(
            packed_weight,
            block_scale,
            self.torch.ones((), dtype=self.torch.float32),
            tensor_scale,
            self.torch.bfloat16,
            self.torch.device("cpu"),
            layout=_NVFP4_LAYOUT_TENSORCORE,
        )
        expected_value = self.torch.tensor(-6.0, dtype=self.torch.bfloat16) * (
            block_scale[0, 0] * tensor_scale.to(self.torch.bfloat16)
        )
        old_order_value = (
            self.torch.tensor(-6.0, dtype=self.torch.bfloat16) * block_scale[0, 0]
        ) * tensor_scale.to(self.torch.bfloat16)

        self.assertTrue(self.torch.equal(actual, self.torch.full_like(actual, expected_value)))
        self.assertNotEqual(expected_value.item(), old_order_value.item())

    def test_full_h3_convrot_checkpoint_loads_as_executable_quantized_linear(self):
        from mmgp import offload, quant_router
        from shared.qtypes import int8_convrot

        quant_router.register_handler("shared.qtypes.int8_convrot")
        descriptor = self.torch.tensor(
            list(
                json.dumps(
                    {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": 4,
                    }
                ).encode("utf-8")
            ),
            dtype=self.torch.uint8,
        )
        weight = self.torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=self.torch.int8,
        )
        scales = self.torch.tensor([1.0, 1.0], dtype=self.torch.float32)
        state_dict = {
            "linear.weight": weight,
            "linear.weight_scale": scales,
            "linear.comfy_quant": descriptor,
        }
        model = self.torch.nn.Module()
        model.linear = self.torch.nn.Linear(4, 2, bias=False, dtype=self.torch.float32)

        offload.load_model_data(
            model,
            (state_dict, None),
            default_dtype=self.torch.float32,
            verboseLevel=0,
        )

        input_rows = self.torch.tensor([[1.0, 2.0, 4.0, 8.0]])
        dequantized = weight.float() * scales[:, None]
        expected = self.torch.nn.functional.linear(
            int8_convrot._rotate_activation(input_rows, 4),
            dequantized,
        )
        actual = model.linear(input_rows)
        self.assertEqual(type(model.linear.weight).__name__, "Int8ConvRotWeightTensor")
        self.assertEqual(model.linear._convrot_group_size, 4)
        self.assertTrue(self.torch.allclose(actual, expected))

    def test_full_h3_convrot_lora_keeps_native_rotation_and_uses_raw_input(self):
        from mmgp import offload, quant_router
        from shared.qtypes import int8_convrot

        quant_router.register_handler("shared.qtypes.int8_convrot")
        descriptor = self.torch.tensor(
            list(
                json.dumps(
                    {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": 4,
                    }
                ).encode("utf-8")
            ),
            dtype=self.torch.uint8,
        )
        weight = self.torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=self.torch.int8,
        )
        scales = self.torch.tensor([1.0, 1.0], dtype=self.torch.float32)
        model = self.torch.nn.Module()
        model.linear = self.torch.nn.Linear(4, 2, bias=False, dtype=self.torch.float32)
        offload.load_model_data(
            model,
            (
                {
                    "linear.weight": weight,
                    "linear.weight_scale": scales,
                    "linear.comfy_quant": descriptor,
                },
                None,
            ),
            default_dtype=self.torch.float32,
            verboseLevel=0,
        )

        class Manager:
            @staticmethod
            def _get_lora_scaling(_scalings, _model, _adapter):
                return 1.0

        module = model.linear
        lora_a = self.torch.tensor([[1.0, 0.0, -1.0, 0.5]])
        lora_b = self.torch.tensor([[0.25], [-0.75]])
        module._mm_lora_old_forward = module.forward
        module._mm_lora_model = model
        module._mm_lora_data = {
            "turbo_GPU": [lora_a, lora_b, None, None, 1.0, {"type": "lora"}]
        }
        module._mm_manager = Manager()
        model._loras_active_adapters = ["turbo"]
        model._loras_scaling = {}
        self.assertEqual(int8_convrot.install_native_lora_forwards(model), 1)

        input_rows = self.torch.tensor([[1.0, 2.0, 4.0, 8.0]])
        dequantized = weight.float() * scales[:, None]
        base = self.torch.nn.functional.linear(
            int8_convrot._rotate_activation(input_rows, 4),
            dequantized,
        )
        update = self.torch.nn.functional.linear(
            self.torch.nn.functional.linear(input_rows, lora_a),
            lora_b,
        )
        actual = model.linear(input_rows)
        wrong_unrotated_base = self.torch.nn.functional.linear(
            input_rows,
            dequantized,
        ) + update
        self.assertTrue(self.torch.allclose(actual, base + update))
        self.assertFalse(self.torch.allclose(actual, wrong_unrotated_base))

    def test_full_h3_convrot_grouped_qkv_rows_and_scales_split_contiguously(self):
        from mmgp import offload, quant_router
        from models.minimax_h3.transformer import get_linear_split_map

        quant_router.register_handler("shared.qtypes.int8_convrot")
        split_map = get_linear_split_map(
            4,
            interleaved=True,
            num_attention_heads=2,
            attention_head_dim=2,
        )
        model = self.torch.nn.Module()
        model.attn = self.torch.nn.Module()
        model.attn.qkv_proj = self.torch.nn.Linear(
            4,
            12,
            bias=False,
            dtype=self.torch.float32,
        )
        offload.split_linear_modules(model, split_map)

        weight = (
            self.torch.arange(48, dtype=self.torch.int16)
            .reshape(12, 4)
            .sub(24)
            .to(self.torch.int8)
        )
        scales = self.torch.arange(1, 13, dtype=self.torch.float32)
        descriptor = self.torch.tensor(
            list(
                json.dumps(
                    {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": 4,
                    }
                ).encode("utf-8")
            ),
            dtype=self.torch.uint8,
        )
        offload.load_model_data(
            model,
            (
                {
                    "attn.qkv_proj.weight": weight,
                    "attn.qkv_proj.weight_scale": scales,
                    "attn.qkv_proj.comfy_quant": descriptor,
                },
                None,
            ),
            default_dtype=self.torch.float32,
            fused_split_map=split_map,
            verboseLevel=0,
        )

        expected_rows = {
            "q_proj": [0, 1, 2, 3],
            "k_proj": [4, 5, 6, 7],
            "v_proj": [8, 9, 10, 11],
        }
        for name, rows in expected_rows.items():
            module = getattr(model.attn, name)
            self.assertTrue(self.torch.equal(module.weight._data, weight[rows]))
            self.assertTrue(
                self.torch.equal(module.weight._scale[:, 0], scales[rows])
            )

    def test_full_h3_lora_qkv_rows_stay_logically_grouped_for_mmgp_split(self):
        from models.minimax_h3.transformer import MiniMaxH3Transformer

        grouped = self.torch.arange(24, dtype=self.torch.float32).reshape(12, 2)
        stub = types.SimpleNamespace(
            h3_qkv_layout="interleaved",
            use_adaln_curves=False,
            config=types.SimpleNamespace(
                num_attention_heads=2,
                attention_head_dim=2,
            ),
        )
        lora_a = self.torch.ones(2, 4)
        state_dict = {
            "blocks.0.attn.qkv_proj.lora_A.weight": lora_a,
            "blocks.0.attn.qkv_proj.lora_B.weight": grouped,
        }
        processed = MiniMaxH3Transformer.preprocess_loras(
            stub,
            "minimax_h3_full",
            state_dict,
        )
        self.assertIs(
            processed["blocks.0.attn.qkv_proj.lora_B.weight"],
            grouped,
        )
        self.assertIs(
            processed["blocks.0.attn.qkv_proj.lora_A.weight"],
            lora_a,
        )

    def test_full_h3_adaln_lora_is_converted_for_pruned_checkpoint(self):
        from models.minimax_h3 import lora_affine

        canonical_table = self.torch.randn(32, 8)
        canonical_affine = self.torch.zeros(9, 2688)
        canonical_affine[:8, :8] = self.torch.eye(8)
        down = self.torch.randn(2, 2688)
        up = self.torch.randn(4, 2)
        prefix = "blocks.0.adaln_proj.linear"
        state_dict = {
            f"{prefix}.lora_A.weight": down,
            f"{prefix}.lora_B.weight": up,
        }

        with mock.patch.object(
            lora_affine,
            "_load_affine_package",
            return_value=(canonical_table, canonical_affine),
        ):
            count, architecture, source_width, target_width = (
                lora_affine.convert_adaln_loras(
                    "minimax_h3",
                    state_dict,
                    canonical_table.clone(),
                )
            )

        self.assertEqual((count, architecture), (1, "fl2va"))
        self.assertEqual((source_width, target_width), (2688, 8))
        self.assertEqual(
            state_dict[f"{prefix}.lora_A.weight"].shape,
            (2, 8),
        )
        self.assertEqual(state_dict[f"{prefix}.diff_b"].shape, (4,))

    def test_turbo_lora_is_validated_for_full_and_pruned_before_load(self):
        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        turbo_path = "minimax_h3_turbo_4step.safetensors"
        full = types.SimpleNamespace(
            transformer=types.SimpleNamespace(use_adaln_curves=False)
        )
        MiniMaxH3Model.validate_loras(full, [turbo_path])
        self.assertTrue(full._turbo_lora_active)

        MiniMaxH3Model.validate_loras(full, [])
        self.assertFalse(full._turbo_lora_active)

        pruned = types.SimpleNamespace(
            transformer=types.SimpleNamespace(use_adaln_curves=True)
        )
        MiniMaxH3Model.validate_loras(pruned, [turbo_path])
        self.assertTrue(pruned._turbo_lora_active)

    def test_row_scaled_int8_embedding_loads_and_dequantizes_selected_rows(self):
        from models.minimax_h3.checkpoint import preprocess_conditioner_state_dict
        from models.minimax_h3.conditioner import MiniMaxH3Int8Embedding

        weight = self.torch.tensor(
            [[1, -2, 3], [4, 5, -6], [-7, 8, 9], [10, -11, 12]],
            dtype=self.torch.int8,
        )
        scales = self.torch.tensor([0.5, 0.25, 2.0, 0.125], dtype=self.torch.float32)
        marker = self.torch.tensor(
            list(b'{"format":"int8_tensorwise"}'),
            dtype=self.torch.uint8,
        )
        state_dict = {
            "model.embed_tokens.comfy_quant": marker,
            "model.embed_tokens.weight": weight.clone(),
            "model.embed_tokens.weight_scale": scales.clone(),
        }
        processed = preprocess_conditioner_state_dict(state_dict)
        self.assertNotIn("model.embed_tokens.comfy_quant", processed)
        self.assertEqual(tuple(processed["model.embed_tokens.weight_scale"].shape), (4, 1))

        embedding = MiniMaxH3Int8Embedding(4, 3, None, self.torch.float32)
        embedding.load_state_dict(
            {
                "weight": processed["model.embed_tokens.weight"],
                "weight_scale": processed["model.embed_tokens.weight_scale"],
            },
            assign=True,
        )
        input_ids = self.torch.tensor([[3, 0, 2, 3]])
        expected = weight[input_ids].float() * scales[input_ids].unsqueeze(-1)
        self.assertTrue(self.torch.equal(embedding(input_ids), expected))
        self.assertFalse(embedding.weight.requires_grad)
        self.assertEqual(embedding._lock_dtype, self.torch.float32)


if __name__ == "__main__":
    unittest.main()
