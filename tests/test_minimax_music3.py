"""Model-free regressions for the native MiniMax-Music3 integration."""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import re
import types
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
_HANDLER = _APP / "models" / "TTS" / "minimax_music3_handler.py"
_PIPELINE = _APP / "models" / "TTS" / "minimax_music3" / "pipeline.py"
_DEFAULT = _APP / "defaults" / "minimax_music3.json"
_WGP = _APP / "wgp.py"
_LAUNCH = _APP / "launch.py"
_STORE = _ROOT / "ui" / "src" / "stores" / "useStore.ts"
_MUSIC_UI = _ROOT / "ui" / "src" / "components" / "Sidebar" / "MusicControls.tsx"
_DIRECTOR_MUSIC_UI = (
    _ROOT / "ui" / "src" / "components" / "Sidebar" / "DirectorSongSetup.tsx"
)
_CLIENT = _ROOT / "ui" / "src" / "api" / "client.ts"
_GUIDE = _APP / "services" / "llm_guides" / "music" / "song_writer_minimax_music3.md"
_INSTRUMENTAL_GUIDE = (
    _APP
    / "services"
    / "llm_guides"
    / "music"
    / "song_writer_minimax_music3_instrumental.md"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_handler_namespace():
    tree = ast.parse(_read(_HANDLER), filename=str(_HANDLER))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef)):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "family_handler":
            selected.append(node)
    namespace = {
        "os": os,
        "torch": types.SimpleNamespace(bfloat16="bf16"),
        "fl": types.SimpleNamespace(),
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_HANDLER), "exec"), namespace)
    return namespace


def _load_prompt_helpers():
    wanted_constants = {
        "_IM_START",
        "_IM_END",
        "_CAPTION_START",
        "_CAPTION_END",
        "_LYRICS_START",
        "_LYRICS_END",
        "_AUDIO_START",
        "_CHUNK_FRAMES",
        "_CHUNK_HOP",
        "_SPECIAL_TAG_RE",
        "_LEADING_TAGS_RE",
    }
    wanted_functions = {
        "clean_music_caption",
        "normalize_music3_lyrics",
        "build_music3_prompt",
        "music3_chunk_starts",
        "estimate_music3_kv_cache_bytes",
        "normalize_music3_qwen_config",
    }
    tree = ast.parse(_read(_PIPELINE), filename=str(_PIPELINE))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = set()
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.update(
                        item.id for item in target.elts if isinstance(item, ast.Name)
                    )
            if names & wanted_constants:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            selected.append(node)
    namespace = {"re": re}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_PIPELINE), "exec"), namespace)
    return namespace


def _load_launch_music_helpers():
    tree = ast.parse(_read(_LAUNCH), filename=str(_LAUNCH))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_music3_writer_duration_instruction"
    ]
    namespace = {"math": math}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_LAUNCH), "exec"), namespace)
    return namespace


class MiniMaxMusic3Tests(unittest.TestCase):
    def test_default_definition_is_visible_and_license_aware(self):
        default = json.loads(_read(_DEFAULT))
        model = default["model"]
        self.assertEqual(model["name"], "MiniMax-Music3")
        self.assertEqual(model["architecture"], "minimax_music3")
        self.assertIn("MiniMax-Music3 Community License", model["license_name"])
        self.assertGreaterEqual(model["model_size_gb"], 28)
        self.assertGreaterEqual(len(model["required_model_assets"]), 7)
        self.assertEqual(default["num_inference_steps"], 30)
        self.assertEqual(default["guidance_scale"], 1.7)
        self.assertEqual(default["duration_seconds"], 120)

    def test_handler_contract_and_download_manifest(self):
        namespace = _load_handler_namespace()
        handler = namespace["family_handler"]
        self.assertEqual(handler.query_supported_types(), ["minimax_music3"])
        model_def = handler.query_model_def("minimax_music3", {})
        self.assertTrue(model_def["audio_only"])
        self.assertTrue(model_def["inference_steps"])
        self.assertTrue(model_def["music3_structured_caption"])
        self.assertEqual(model_def["duration_slider"]["max"], 300)
        self.assertEqual(model_def["duration_slider"]["default"], 120)
        manifest = handler.query_model_files([], "minimax_music3")
        self.assertEqual(manifest["repoId"], "MiniMaxAI/MiniMax-Music3")
        self.assertEqual(len(manifest["sourceFolderList"]), 8)
        self.assertEqual(len(manifest["sourceFolderList"]), len(manifest["fileList"]))
        flattened = [item for group in manifest["fileList"] for item in group]
        self.assertIn("LICENSE", flattened)
        self.assertIn("model-00004-of-00004.safetensors", flattened)
        self.assertIn("diffusion_pytorch_model-00002-of-00002.safetensors", flattened)

    def test_settings_validation(self):
        handler = _load_handler_namespace()["family_handler"]
        model_def = handler.query_model_def("minimax_music3", {})
        valid = {
            "alt_prompt": "### Global Metadata\nPop",
            "duration_seconds": 30,
            "num_inference_steps": 30,
        }
        self.assertIsNone(
            handler.validate_generative_prompt(
                "minimax_music3", model_def, valid, "[Verse]\nHello"
            )
        )
        self.assertIsNone(
            handler.validate_generative_settings(
                "minimax_music3", model_def, valid
            )
        )
        self.assertIn(
            "Music Caption",
            handler.validate_generative_prompt(
                "minimax_music3", model_def, {**valid, "alt_prompt": ""}, "lyrics"
            ),
        )
        self.assertIn(
            "between 5 and 300",
            handler.validate_generative_settings(
                "minimax_music3", model_def, {**valid, "duration_seconds": 301}
            ),
        )

    def test_prompt_assembly_preserves_checkpoint_contract(self):
        helpers = _load_prompt_helpers()
        normalize = helpers["normalize_music3_lyrics"]
        prompt = helpers["build_music3_prompt"](
            "### Global Metadata\n- Warm pop\n\n### Arrangement\n**Wide chorus**",
            "[Verse - intimate] words that must be dropped\nLine one\n[Chorus]\nLine two",
        )
        self.assertTrue(prompt.startswith("<|im_start|><|caption_start|>"))
        self.assertTrue(prompt.endswith("<|im_end|><|audio_start|>"))
        self.assertNotIn("###", prompt)
        self.assertNotIn("**", prompt)
        lyrics = normalize("[Verse] words on tag line\nActual line\n[CHORUS]\nHook")
        self.assertEqual(lyrics.splitlines()[0], "[start]")
        self.assertEqual(lyrics.splitlines()[1], "[verse]")
        self.assertNotIn("words on tag line", lyrics)
        self.assertIn("[chorus]", lyrics)

    def test_chunk_schedule_matches_official_overlap_geometry(self):
        starts = _load_prompt_helpers()["music3_chunk_starts"]
        self.assertEqual(starts(200), [0])
        self.assertEqual(starts(201), [0, 100])
        self.assertEqual(starts(750), [0, 100, 200, 300, 400, 500, 600])

    def test_long_song_cache_estimate_matches_qwen_geometry(self):
        helpers = _load_prompt_helpers()
        config = types.SimpleNamespace(
            num_hidden_layers=36,
            num_key_value_heads=8,
            head_dim=128,
        )
        estimate = helpers["estimate_music3_kv_cache_bytes"](
            config,
            prompt_tokens=500,
            duration_seconds=150,
        )
        self.assertGreater(estimate, 1024**3)

    def test_transformers_4_rope_config_uses_checkpoint_theta(self):
        normalize = _load_prompt_helpers()["normalize_music3_qwen_config"]
        config = types.SimpleNamespace(
            rope_theta=10_000.0,
            rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        )
        self.assertIs(normalize(config), config)
        self.assertEqual(config.rope_theta, 1_000_000.0)

        legacy = types.SimpleNamespace(rope_theta=1_000_000.0)
        self.assertIs(normalize(legacy), legacy)
        self.assertEqual(legacy.rope_theta, 1_000_000.0)

    def test_runtime_registers_handler_and_single_gpu_stages(self):
        wgp = _read(_WGP)
        pipeline = _read(_PIPELINE)
        launch = _read(_LAUNCH)
        self.assertIn('"models.TTS.minimax_music3_handler"', wgp)
        self.assertIn("Qwen2TokenizerFast.from_pretrained", pipeline)
        self.assertIn("Qwen3Config.from_pretrained", pipeline)
        self.assertIn("normalize_music3_qwen_config", pipeline)
        self.assertNotIn("Qwen2Tokenizer.from_pretrained", pipeline)
        self.assertIn("MiniMaxMusic3LanguageModelRunner", pipeline)
        self.assertIn("MiniMaxMusic3DepthRunner", pipeline)
        self.assertIn("Music3PreallocatedCache", pipeline)
        self.assertNotIn("DynamicCache(", pipeline)
        self.assertIn("optional_flash_attention_available", pipeline)
        self.assertIn('"flash_attention_2"', pipeline)
        self.assertIn('else "sdpa"', pipeline)
        self.assertIn("self._release_stage(offloadobj)", pipeline)
        self.assertIn('"audio_sampling_rate": self.sampling_rate', pipeline)
        self.assertIn("compute_music3_weight_budget", launch)
        self.assertIn("resident Music3 profile will reload", launch)
        self.assertIn("music3_kv_cache_gb", launch)
        self.assertIn("MiniMax-Music3 ran out of VRAM while planning the song", wgp)

    def test_music_ui_and_song_writer_are_model_aware(self):
        store = _read(_STORE)
        ui = _read(_MUSIC_UI)
        client = _read(_CLIENT)
        launch = _read(_LAUNCH)
        self.assertIn("'minimax_music3'", store)
        self.assertIn("const DEFAULTS_VERSION = 10", store)
        self.assertIn("music3_structured_caption", ui)
        self.assertIn("model_type: params.model_type", ui)
        self.assertIn("duration_seconds: durationSeconds", ui)
        self.assertIn("model_type?: string", client)
        self.assertIn("duration_seconds?: number", client)
        self.assertIn('load_guide("music", "song_writer_minimax_music3")', launch)
        self.assertIn("_music3_writer_duration_instruction", launch)

    def test_music3_writer_receives_a_bounded_runtime_contract(self):
        runtime = _load_launch_music_helpers()["_music3_writer_duration_instruction"]
        self.assertIn("30 seconds long", runtime(30))
        self.assertIn("ending near 30 seconds", runtime(30))
        self.assertIn("120 seconds long", runtime(None))
        self.assertIn("300 seconds long", runtime(999))

    def test_director_can_select_and_submit_music3(self):
        store = _read(_STORE)
        setup = _read(_DIRECTOR_MUSIC_UI)
        launch = _read(_LAUNCH)
        self.assertIn("directorMusicModel: string", store)
        self.assertIn("directorMusicModel: 'ace_step_v1_5_xl_sft_lm_4b'", store)
        self.assertGreaterEqual(
            store.count("model_type: s.directorMusicModel"),
            2,
        )
        self.assertIn("'minimax_music3'", setup)
        self.assertIn("Music model", setup)
        self.assertIn("maximumDuration = isMusic3 ? 300 : 360", setup)
        self.assertIn("generation_timeout_s", launch)
        self.assertIn("1536 if is_minimax_music3 else 1024", launch)

    def test_music3_guides_follow_official_structure(self):
        guide = _read(_GUIDE)
        instrumental = _read(_INSTRUMENTAL_GUIDE)
        for text in (guide, instrumental):
            self.assertIn("### Global Metadata", text)
            self.assertIn("### Vocal Details", text)
            self.assertIn("### Arrangement", text)
            self.assertIn("[STYLE]", text)
            self.assertIn("[LYRICS]", text)
            self.assertIn("TARGET RUNTIME CONTRACT", text)
            self.assertIn("5-20 seconds", text)
        self.assertIn("[Instrumental]", instrumental)

    def test_vendored_diffusers_components_keep_license_headers(self):
        component_dir = _PIPELINE.parent
        for name in (
            "condition_encoder.py",
            "rvq_depth_decoder.py",
            "transformer.py",
            "vocoder.py",
        ):
            text = _read(component_dir / name)
            self.assertIn("Copyright 2026 The MiniMax Team", text)
            self.assertIn("Apache License", text)
        notice = _read(component_dir / "NOTICE.md")
        self.assertIn("MiniMax-Music3 Community License", notice)


if __name__ == "__main__":
    unittest.main()
