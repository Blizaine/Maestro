"""Structural regressions for the official DaSiWa Krea/Wan/LTX presets."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ROOT / "app" / "defaults"
KREA_HANDLER = ROOT / "app" / "models" / "krea2" / "krea2_handler.py"
WAN_HANDLER = ROOT / "app" / "models" / "wan" / "wan_handler.py"
LTX_HANDLER = ROOT / "app" / "models" / "ltx2" / "ltx2_handler.py"


def _load_default(filename: str) -> dict:
    return json.loads((DEFAULTS / filename).read_text(encoding="utf-8"))


def _literal_assignments(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _load_family_handler(path: Path, *, extra_globals: dict | None = None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "family_handler"
    )
    namespace = _literal_assignments(path)
    namespace["torch"] = SimpleNamespace(bfloat16="bfloat16", float32="float32")
    if extra_globals:
        namespace.update(extra_globals)
    module = ast.Module(body=[class_node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["family_handler"]


def _load_functions(path: Path, function_names: set[str]) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    if {node.name for node in functions} != function_names:
        raise AssertionError(f"Missing handler functions: {function_names}")
    namespace = {}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def _build_hf_url(repo: str, *parts: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/" + "/".join(parts)


class DaSiWaFamilyPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.krea_handler = _load_family_handler(
            KREA_HANDLER,
            extra_globals={
                "build_hf_url": _build_hf_url,
                "torch": SimpleNamespace(bfloat16="bfloat16", float32="float32"),
            },
        )
        cls.wan_handler = _load_family_handler(WAN_HANDLER)
        cls.wan_classifiers = _load_functions(
            WAN_HANDLER,
            {"test_class_i2v", "test_class_t2v"},
        )
        cls.ltx_handler = _load_family_handler(
            LTX_HANDLER,
            extra_globals={
                "_get_arch_spec": lambda _model_type: {
                    "profiles_dir": "ltx2_22B",
                    "spatial_upscaler": "ltx-spatial.safetensors",
                },
                "_GEMMA_FILENAME": "gemma-3-12b-it-qat-q4_0-unquantized.safetensors",
                "_GEMMA_QUANTO_FILENAME": "gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors",
                "build_hf_url": _build_hf_url,
            },
        )

    def test_architectures_resolve_to_the_expected_family_capabilities(self):
        turbo = _load_default("dasiwa_krea2_turbo.json")
        raw = _load_default("dasiwa_krea2_raw.json")
        self.assertIn(turbo["model"]["architecture"], self.krea_handler.query_supported_types())
        self.assertIn(raw["model"]["architecture"], self.krea_handler.query_supported_types())
        turbo_caps = self.krea_handler.query_model_def(turbo["model"]["architecture"], {})
        raw_caps = self.krea_handler.query_model_def(raw["model"]["architecture"], {})
        self.assertTrue(turbo_caps["image_outputs"])
        self.assertTrue(turbo_caps["lock_guidance_scale"])
        self.assertTrue(turbo["model"]["nsfw_only"])
        self.assertIn("Mature", turbo["model"]["name"])
        self.assertFalse(raw_caps["lock_guidance_scale"])
        self.assertNotIn("nsfw_only", raw["model"])

        wan = _load_default("dasiwa_wan2_2_i2v_lightspeed_v9.json")
        wan_arch = wan["model"]["architecture"]
        self.assertIn(wan_arch, self.wan_handler.query_supported_types())
        self.assertTrue(self.wan_classifiers["test_class_i2v"](wan_arch))
        self.assertFalse(self.wan_classifiers["test_class_t2v"](wan_arch))

        ltx = _load_default("dasiwa_ltx2_3_dragonleap_v4.json")
        ltx_arch = ltx["model"]["architecture"]
        self.assertIn(ltx_arch, self.ltx_handler.query_supported_types())
        ltx_caps = self.ltx_handler.query_model_def(
            ltx_arch,
            {"ltx2_pipeline": ltx["model"]["ltx2_pipeline"]},
        )
        self.assertTrue(ltx_caps["returns_audio"])
        self.assertTrue(ltx_caps["lock_inference_steps"])
        self.assertEqual(ltx_caps["visible_phases"], 0)

    def test_family_settings_match_distillation_modes(self):
        turbo = _load_default("dasiwa_krea2_turbo.json")
        raw = _load_default("dasiwa_krea2_raw.json")
        self.assertEqual((turbo["num_inference_steps"], turbo["guidance_scale"]), (8, 0))
        self.assertEqual((raw["num_inference_steps"], raw["guidance_scale"]), (52, 3.5))
        self.assertGreaterEqual(raw["model"]["inference_steps_max"], raw["num_inference_steps"])

        wan = _load_default("dasiwa_wan2_2_i2v_lightspeed_v9.json")
        self.assertEqual(wan["num_inference_steps"], 4)
        self.assertEqual(wan["guidance_phases"], 2)
        self.assertEqual((wan["guidance_scale"], wan["guidance2_scale"]), (1.0, 1.0))
        self.assertEqual(wan["sample_solver"], "euler")
        self.assertTrue(wan["model"]["nsfw_only"])
        self.assertIn("Mature", wan["model"]["name"])

        ltx = _load_default("dasiwa_ltx2_3_dragonleap_v4.json")
        self.assertEqual(ltx["model"]["ltx2_pipeline"], "distilled")
        self.assertEqual(ltx["num_inference_steps"], 8)
        self.assertEqual((ltx["video_length"], ltx["resolution"]), (241, "1280x720"))
        self.assertNotIn("nsfw_only", ltx["model"])
        self.assertIn("Experimental", ltx["model"]["name"])

    def test_all_checkpoint_sources_are_pinned_and_keyed_by_clean_filename(self):
        filenames = (
            "dasiwa_krea2_turbo.json",
            "dasiwa_krea2_raw.json",
            "dasiwa_wan2_2_i2v_lightspeed_v9.json",
            "dasiwa_ltx2_3_dragonleap_v4.json",
        )
        for preset_filename in filenames:
            with self.subTest(preset=preset_filename):
                model = _load_default(preset_filename)["model"]
                expected_filenames = set(model["URLs"])
                expected_filenames.update(model.get("URLs2", []))
                sources = model["download_sources"]
                self.assertEqual(set(sources), expected_filenames)

                for filename, source in sources.items():
                    self.assertNotIn("?", filename)
                    self.assertNotIn("/", filename)
                    self.assertTrue(filename.endswith(".safetensors"))
                    self.assertRegex(source["sha256"], re.compile(r"^[0-9a-f]{64}$"))
                    self.assertGreater(source["size_bytes"], 0)
                    parsed = urlparse(source["url"])
                    self.assertEqual(parsed.scheme, "https")
                    self.assertIn(parsed.hostname, {"civitai.com", "civitai.red"})
                    file_id = parse_qs(parsed.query).get("fileId", [""])[0]
                    self.assertTrue(file_id)
                    version_id = parsed.path.rstrip("/").split("/")[-1]
                    self.assertIn(f"{version_id}:{file_id}", model["source_revision"])
                    top_hashes = model["source_sha256"]
                    if isinstance(top_hashes, dict):
                        expected_hash = "high" if "High" in filename else "low"
                        self.assertEqual(source["sha256"], top_hashes[expected_hash])
                    else:
                        self.assertEqual(source["sha256"], top_hashes)
                self.assertIn("Civitai API key", model["selector_help"])

    def test_wan_lightspeed_pair_is_high_with_high_and_low_with_low_v9(self):
        model = _load_default("dasiwa_wan2_2_i2v_lightspeed_v9.json")["model"]
        high, = model["URLs"]
        low, = model["URLs2"]
        self.assertIn("HighV9.safetensors", high)
        self.assertIn("LowV9.safetensors", low)
        self.assertNotIn(".gguf", high.lower())
        self.assertNotIn(".gguf", low.lower())

        high_id = parse_qs(urlparse(model["download_sources"][high]["url"]).query)["fileId"][0]
        low_id = parse_qs(urlparse(model["download_sources"][low]["url"]).query)["fileId"][0]
        self.assertEqual(high_id, "2455463")
        self.assertEqual(low_id, "2455626")
        self.assertEqual(model["download_sources"][high]["size_bytes"], 14528641504)
        self.assertEqual(model["download_sources"][low]["size_bytes"], 14528641504)
        self.assertIn("source image used for the first frame", model["selector_help"])
        self.assertIn("both matching SynthSeduction v9 High and Low checkpoints are required", model["selector_help"])

    def test_ltx_is_full_checkpoint_and_clearly_marks_unrendered_status(self):
        model = _load_default("dasiwa_ltx2_3_dragonleap_v4.json")["model"]
        filename, = model["URLs"]
        self.assertIn("dragonleapV4", filename)
        self.assertIn("Experimental", model["name"])
        self.assertIn("not yet been render-tested", model["description"])
        self.assertNotIn("lora", filename.lower())
        self.assertEqual(
            model["download_sources"][filename]["sha256"],
            model["source_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
