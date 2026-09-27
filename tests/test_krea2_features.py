"""Model-free regressions for Krea 2 editing and LoRA discovery."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
import types
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
_HANDLER_PATH = _APP / "models" / "krea2" / "krea2_handler.py"
_IDENTITY_CONTROLS_PATH = _APP / "models" / "krea2" / "identity_controls.py"
_MAIN_PATH = _APP / "models" / "krea2" / "krea2_main.py"
_MMDIT_PATH = _APP / "models" / "krea2" / "krea2_mmdit.py"
_LAUNCH_PATH = _APP / "launch.py"
_GGUF_PATH = _APP / "shared" / "utils" / "gguf_mapping.py"
_TYPES_PATH = _ROOT / "ui" / "src" / "types" / "index.ts"
_REF_UI_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "ImageRefSection.tsx"
_INPUTS_UI_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "InputsPanel.tsx"
_STORE_PATH = _ROOT / "ui" / "src" / "stores" / "useStore.ts"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_handler_class():
    source = _read(_HANDLER_PATH)
    tree = ast.parse(source, filename=str(_HANDLER_PATH))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(name.startswith("_") for name in names):
                selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "family_handler":
            selected.append(node)
    controls_spec = importlib.util.spec_from_file_location("_krea2_identity_controls_test", _IDENTITY_CONTROLS_PATH)
    controls = importlib.util.module_from_spec(controls_spec)
    controls_spec.loader.exec_module(controls)
    namespace = {
        "os": os,
        "torch": types.SimpleNamespace(bfloat16="bfloat16", float32="float32"),
        "gr": types.SimpleNamespace(Info=lambda *_args, **_kwargs: None),
        "build_hf_url": lambda repo, *parts: "https://huggingface.co/" + repo + "/resolve/main/" + "/".join(parts),
        "IDENTITY_SETTINGS_DEFAULTS": controls.IDENTITY_SETTINGS_DEFAULTS,
        "normalize_identity_settings": controls.normalize_identity_settings,
        "Mapping": Mapping,
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_HANDLER_PATH), "exec"), namespace)
    return namespace["family_handler"]


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(_read(path), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Could not find literal assignment {name}")


def _load_lora_preprocessor():
    tree = ast.parse(_read(_MMDIT_PATH), filename=str(_MMDIT_PATH))
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SingleStreamDiT":
            method = next(
                child for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name == "preprocess_loras"
            )
            break
    if method is None:
        raise AssertionError("SingleStreamDiT.preprocess_loras is missing")
    namespace = {}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_MMDIT_PATH), "exec"), namespace)
    return namespace["preprocess_loras"]


def _load_model_download_check(*, vision_exists: bool):
    tree = ast.parse(_read(_LAUNCH_PATH), filename=str(_LAUNCH_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_check_model_downloaded"
    )
    model_def = {"vision_encoder_filename": "vision.safetensors"}
    fake_wgp = types.SimpleNamespace(
        get_model_def=lambda _model_type: model_def,
        get_model_recursive_prop=lambda *_args, **_kwargs: [],
        resolve_lora_path=lambda *_args, **_kwargs: "",
        fl=types.SimpleNamespace(
            locate_file=lambda *_args, **_kwargs: "vision.safetensors" if vision_exists else None,
        ),
    )
    namespace = {
        "os": os,
        "wgp": fake_wgp,
        "_model_weight_groups": lambda _model_type: [["transformer.safetensors"]],
        "_variant_group_downloaded": lambda _group, model_type=None: True,
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_LAUNCH_PATH), "exec"), namespace)
    return namespace["_check_model_downloaded"]("krea2_raw_edit")


class TestKrea2EditDefinitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = _load_handler_class()

    def test_edit_defaults_share_base_models_and_identity_lora(self):
        cases = {
            "krea2_raw_edit.json": ("krea2_raw_edit", "krea2_raw", 20, 2),
            "krea2_turbo_edit.json": ("krea2_turbo_edit", "krea2_turbo", 8, 0),
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                payload = json.loads((_APP / "defaults" / filename).read_text(encoding="utf-8"))
                architecture, base, steps, guidance = expected
                self.assertEqual(payload["model"]["architecture"], architecture)
                self.assertEqual(payload["model"]["URLs"], base)
                self.assertIn("krea2_identity_edit_v1_2.safetensors", payload["model"]["loras"][0])
                self.assertEqual(payload["video_prompt_type"], "KI")
                self.assertEqual(payload["num_inference_steps"], steps)
                self.assertEqual(payload["guidance_scale"], guidance)
                self.assertEqual(payload["custom_settings"], {
                    "krea2_ref_boost": 1.0,
                    "krea2_ref_boost_a": 1.0,
                    "krea2_grounding_px": 768,
                })

    def test_handler_registers_edit_models_and_reference_capabilities(self):
        supported = self.handler.query_supported_types()
        self.assertIn("krea2_raw_edit", supported)
        self.assertIn("krea2_turbo_edit", supported)
        for model_type in ("krea2_raw_edit", "krea2_turbo_edit"):
            with self.subTest(model_type=model_type):
                model_def = self.handler.query_model_def(model_type, {})
                self.assertEqual(model_def["image_ref_choices"]["default"], "KI")
                self.assertEqual(model_def["max_image_refs"], 2)
                self.assertEqual(model_def["inpaint_video_prompt_type"], "VAG")
                self.assertTrue(model_def["preserve_image_ref_native_size"])
                self.assertEqual(model_def["lock_guidance_scale"], model_type == "krea2_turbo_edit")
                self.assertEqual(model_def["vision_encoder_filename"], "Qwen3-VL-4B-Instruct_vision_bf16.safetensors")
                controls = {item["id"]: item for item in model_def["custom_settings"]}
                self.assertEqual(set(controls), {
                    "krea2_ref_boost", "krea2_ref_boost_a", "krea2_grounding_px",
                })
                self.assertEqual(controls["krea2_ref_boost"]["default"], 1.0)
                self.assertEqual(controls["krea2_ref_boost_a"]["default"], 1.0)
                self.assertEqual((controls["krea2_grounding_px"]["min"], controls["krea2_grounding_px"]["max"], controls["krea2_grounding_px"]["step"]), (384, 1536, 64))
                self.assertEqual(len(model_def["preload_URLs"]), 1)
                self.assertNotIn("|", model_def["preload_URLs"][0])
                self.assertFalse(model_def["no_background_removal"])
        self.assertNotIn("image_ref_choices", self.handler.query_model_def("krea2_raw", {}))
        self.assertFalse(self.handler.query_model_def("krea2_raw", {})["lock_guidance_scale"])
        self.assertTrue(self.handler.query_model_def("krea2_turbo", {})["lock_guidance_scale"])

    def test_turbo_guidance_is_canonical_zero_and_raw_guidance_remains_available(self):
        for model_type in ("krea2_turbo", "krea2_turbo_edit"):
            with self.subTest(model_type=model_type):
                settings = {"guidance_scale": 5.0}
                self.assertIsNone(self.handler.validate_generative_settings(model_type, {}, settings))
                self.assertEqual(settings["guidance_scale"], 0)
                defaults = {"guidance_scale": 5.0}
                self.handler.fix_settings(model_type, 0, {}, defaults)
                self.assertEqual(defaults["guidance_scale"], 0)
        raw = {"guidance_scale": 2.5}
        self.assertIsNone(self.handler.validate_generative_settings("krea2_raw_edit", {}, raw))
        self.assertEqual(raw["guidance_scale"], 2.5)

    def test_edit_defaults_and_reference_limit_are_enforced(self):
        raw_defaults = {}
        turbo_defaults = {}
        self.handler.update_default_settings("krea2_raw_edit", {}, raw_defaults)
        self.handler.update_default_settings("krea2_turbo_edit", {}, turbo_defaults)
        self.assertEqual((raw_defaults["num_inference_steps"], raw_defaults["guidance_scale"]), (20, 2))
        self.assertEqual((turbo_defaults["num_inference_steps"], turbo_defaults["guidance_scale"]), (8, 0))
        self.assertEqual(raw_defaults["video_prompt_type"], "KI")
        self.assertEqual(raw_defaults["custom_settings"], {
            "krea2_ref_boost": 1.0,
            "krea2_ref_boost_a": 1.0,
            "krea2_grounding_px": 768,
        })
        existing_defaults = {"custom_settings": {"dlss_intensity": 0.4, "krea2_ref_boost": 2.5}}
        self.handler.update_default_settings("krea2_raw_edit", {}, existing_defaults)
        self.assertEqual(existing_defaults["custom_settings"], {
            "dlss_intensity": 0.4,
            "krea2_ref_boost": 2.5,
            "krea2_ref_boost_a": 1.0,
            "krea2_grounding_px": 768,
        })
        self.assertIsNone(self.handler.validate_generative_settings(
            "krea2_raw_edit", {}, {"image_mode": 1, "image_refs": [1, 2], "denoising_strength": 1},
        ))
        self.assertIn("at most two reference images", self.handler.validate_generative_settings(
            "krea2_raw_edit", {}, {"image_mode": 1, "image_refs": [1, 2, 3], "denoising_strength": 1},
        ))
        self.assertIn("at most one additional reference image", self.handler.validate_generative_settings(
            "krea2_raw_edit", {}, {"image_mode": 2, "image_refs": [1, 2], "denoising_strength": 1},
        ))
        valid = {"image_mode": 1, "image_refs": [], "custom_settings": {"dlss_intensity": 0.4, "krea2_ref_boost": 2.5}}
        self.assertIsNone(self.handler.validate_generative_settings("krea2_raw_edit", {}, valid))
        self.assertEqual(valid["custom_settings"], {
            "dlss_intensity": 0.4,
            "krea2_ref_boost": 2.5,
            "krea2_ref_boost_a": 1.0,
            "krea2_grounding_px": 768,
        })
        for settings in (
            {"krea2_ref_boost": True},
            {"krea2_ref_boost": float("nan")},
            {"krea2_ref_boost_a": float("inf")},
            {"krea2_ref_boost": 10.1},
            {"krea2_grounding_px": 383},
            {"krea2_grounding_px": 768.5},
        ):
            with self.subTest(settings=settings):
                invalid = {"image_mode": 1, "image_refs": [], "custom_settings": settings}
                self.assertIsNotNone(self.handler.validate_generative_settings("krea2_raw_edit", {}, invalid))

    def test_identity_settings_normalizer_is_strict_and_canonical(self):
        spec = importlib.util.spec_from_file_location("_krea2_identity_controls_test", _IDENTITY_CONTROLS_PATH)
        controls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controls)
        normalize = controls.normalize_identity_settings
        self.assertEqual(normalize({}), {
            "krea2_ref_boost": 1.0,
            "krea2_ref_boost_a": 1.0,
            "krea2_grounding_px": 768,
        })
        self.assertEqual(normalize({"krea2_ref_boost": "0", "krea2_grounding_px": 1536, "ignored": 7}), {
            "krea2_ref_boost": 0.0,
            "krea2_ref_boost_a": 1.0,
            "krea2_grounding_px": 1536,
        })
        for value in (True, float("nan"), float("inf"), -0.1, 10.1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize({"krea2_ref_boost": value})
        for value in (True, 383, 1537, 768.5, float("inf"), "768.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize({"krea2_grounding_px": value})

    def test_current_upstream_identity_and_multi_reference_path_is_present(self):
        main = _read(_MAIN_PATH)
        self.assertIn("Qwen3VLVisionModel", main)
        self.assertIn('base_model_type in ("krea2_raw_edit", "krea2_turbo_edit")', main)
        self.assertIn("images=images * len(text)", main)
        self.assertIn("reference_images=reference_images", main)
        self.assertIn("target_len=latents.shape[1]", main)
        self.assertIn('identity_settings["krea2_grounding_px"]', main)
        self.assertIn("_build_krea2_text_encoder_preprocessor", main)
        self.assertIn("fit_all_references=_fit_all_reference_images(identity_edit, video_prompt_type)", main)
        self.assertIn('if fit_crop and not model_def.get("preserve_image_ref_native_size", False):', _read(_ROOT / "app" / "wgp.py"))

    def test_cfg_forwards_identity_bias_to_both_conditioning_streams(self):
        tree = ast.parse(_read(_MMDIT_PATH), filename=str(_MMDIT_PATH))
        dit = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SingleStreamDiT")
        forward_cfg = next(node for node in dit.body if isinstance(node, ast.FunctionDef) and node.name == "forward_cfg")
        block_calls = [
            node for node in ast.walk(forward_cfg)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "block"
        ]
        self.assertEqual(len(block_calls), 2)
        for call in block_calls:
            forwarded = {item.arg for item in call.keywords}
            self.assertTrue({"reference_token_lengths", "ref_boost", "ref_boost_a", "target_len"}.issubset(forwarded))

    def test_edit_model_readiness_requires_the_vision_encoder(self):
        self.assertFalse(_load_model_download_check(vision_exists=False))
        self.assertTrue(_load_model_download_check(vision_exists=True))


class TestKrea2LoraCompatibility(unittest.TestCase):
    def test_diffusers_and_kohya_lora_names_are_mapped(self):
        preprocess = _load_lora_preprocessor()
        mapped = preprocess(None, "krea2_raw", {
            "transformer_blocks.0.attn.to_q.lora_A.weight": 1,
            "final_layer.linear.lora_B.weight": 2,
            "lora_unet_transformer_blocks_1_attn_to_out_0.lora_down.weight": 3,
        })
        self.assertIn("blocks.0.attn.wq.lora_A.weight", mapped)
        self.assertIn("last.linear.lora_B.weight", mapped)
        self.assertIn("blocks.1.attn.wo.lora_down.weight", mapped)

    def test_gguf_mapping_keeps_triplets_consistent(self):
        namespace = {}
        exec(compile(_read(_GGUF_PATH), str(_GGUF_PATH), "exec"), namespace)
        state, quant, tied = namespace["remap_state_dict_triplet"](
            {"token_embd.weight": 1},
            {"token_embd.weight": "q"},
            {"token_embd.weight": ["token_embd.weight"]},
            {"token_embd.weight": "embed_tokens.weight"},
        )
        self.assertEqual(state, {"embed_tokens.weight": 1})
        self.assertEqual(quant, {"embed_tokens.weight": "q"})
        self.assertEqual(tied, {"embed_tokens.weight": ["embed_tokens.weight"]})


class TestKrea2LoraBrowser(unittest.TestCase):
    def test_civitai_and_huggingface_routes_target_krea_directory(self):
        civit_map = _literal_assignment(_LAUNCH_PATH, "CIVIT_TO_LOCAL_ARCH")
        hf_map = _literal_assignment(_LAUNCH_PATH, "HF_BASE_TO_LOCAL_DIR")
        filters = _literal_assignment(_LAUNCH_PATH, "CIVITAI_MODEL_FILTERS")
        self.assertEqual(civit_map["Krea 2"], "krea2")
        self.assertEqual(hf_map["krea/Krea-2-Raw"], "krea2")
        self.assertEqual(hf_map["krea/Krea-2-Turbo"], "krea2")
        self.assertIn(
            {"label": "Krea 2", "civitai_base": "Krea 2", "default_dir": "krea2"},
            filters,
        )

    def test_owned_lora_filter_and_reference_limit_are_wired_to_ui(self):
        launch = _read(_LAUNCH_PATH)
        ref_ui = _read(_REF_UI_PATH)
        inputs_ui = _read(_INPUTS_UI_PATH)
        types_source = _read(_TYPES_PATH)
        self.assertIn('"max_image_refs": md.get("max_image_refs")', launch)
        self.assertIn("max_image_refs?: number | null", types_source)
        self.assertIn("const configuredMaxRefs = modelOptions?.max_image_refs ?? null", ref_ui)
        self.assertIn("const sourceSlots = imageMode === 2 || qwenHasSource ? 1 : 0", ref_ui)
        self.assertIn("configuredMaxRefs - sourceSlots", ref_ui)
        self.assertIn("files.slice(0, room).forEach(addImageRef)", ref_ui)
        self.assertIn("canAddRef", inputs_ui)
        self.assertIn("lora.directory === activeFilter.default_dir", _read(
            _ROOT / "ui" / "src" / "components" / "LoraBrowser" / "LoraBrowser.tsx"
        ))

    def test_all_four_krea_models_are_curated_image_defaults(self):
        store = _read(_STORE_PATH)
        default_block = store.split("const DEFAULT_ENABLED_MODELS = new Set([", 1)[1].split("])\n", 1)[0]
        for model_type in (
            "krea2_raw",
            "krea2_turbo",
            "krea2_raw_edit",
            "krea2_turbo_edit",
        ):
            with self.subTest(model_type=model_type):
                self.assertIn(f"'{model_type}'", default_block)
        defaults_version = int(store.split("const DEFAULTS_VERSION = ", 1)[1].splitlines()[0])
        self.assertGreaterEqual(defaults_version, 5)
        self.assertIn(
            "5: ['krea2_raw', 'krea2_turbo', 'krea2_raw_edit', 'krea2_turbo_edit']",
            store,
        )


if __name__ == "__main__":
    unittest.main()
