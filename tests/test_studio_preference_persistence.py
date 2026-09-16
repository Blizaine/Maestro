"""Regression coverage for durable, non-project Studio preferences."""

import ast
import json
import os
import unittest


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAUNCH_PATH = os.path.join(_ROOT, "app", "launch.py")
_STORE_PATH = os.path.join(_ROOT, "ui", "src", "stores", "useStore.ts")
_CLIENT_PATH = os.path.join(_ROOT, "ui", "src", "api", "client.ts")


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_normalizers():
    source = _read(_LAUNCH_PATH)
    tree = ast.parse(source)
    wanted = {
        "_normalize_studio_model_map",
        "_normalize_studio_preferences",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), _LAUNCH_PATH, "exec"), namespace)
    return namespace


class TestStudioPreferencePersistence(unittest.TestCase):
    def test_director_gpu_limit_round_trip_auto_and_other_preference_saves(self):
        normalize = _load_normalizers()["_normalize_studio_preferences"]
        saved = normalize({"director_max_shot_frames_per_model": {"minimax_h3_ref2va_fused_turbo": 345}})
        saved = normalize({"audio_sub_mode": "music"}, current=json.loads(json.dumps(saved)))
        self.assertEqual(saved["director_max_shot_frames_per_model"], {"minimax_h3_ref2va_fused_turbo": 345})
        self.assertEqual(normalize({"director_max_shot_frames_per_model": {}}, current=saved)["director_max_shot_frames_per_model"], {})
        for value in (True, -1, 0, 1.5, float('nan'), float('inf'), 60001):
            with self.assertRaises(ValueError):
                normalize({"director_max_shot_frames_per_model": {"model": value}})

    def test_music_migration_stamp_and_subsequent_choice_survive_other_saves(self):
        normalize = _load_normalizers()["_normalize_studio_preferences"]
        saved = normalize({"music_defaults_version": 1, "director_music_model": "yue2"})
        saved = normalize({"director_music_model": "minimax_music3"}, current=saved)
        restored = normalize({"audio_sub_mode": "music"}, current=json.loads(json.dumps(saved)))
        self.assertEqual(restored["music_defaults_version"], 1)
        self.assertEqual(restored["director_music_model"], "minimax_music3")
        for value in (True, -1, 1.5, "1", None):
            with self.assertRaises(ValueError):
                normalize({"music_defaults_version": value})

    def test_music_clip_maximum_round_trip_and_legacy_saves(self):
        normalize = _load_normalizers()["_normalize_studio_preferences"]
        for value in (None, 2.0, 124 / 24):
            saved = normalize({"director_music_clip_seconds": value})
            restored = normalize(json.loads(json.dumps(saved)))
            self.assertEqual(restored["director_music_clip_seconds"], value)
            self.assertEqual(normalize({"generation_mode": "audio"}, current=restored)["director_music_clip_seconds"], value)
        for bad in (True, 0, -1, 301, "6", float("nan"), float("inf")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                normalize({"director_music_clip_seconds": bad})

    def test_backend_normalizes_safe_preferences_only(self):
        normalize = _load_normalizers()["_normalize_studio_preferences"]
        result = normalize({
            "generation_mode": "audio",
            "studio_video_workflow": "references",
            "audio_sub_mode": "music",
            "selected_model_per_mode": {"audio": "minimax_music3"},
            "selected_model_per_audio_sub_mode": {"music": "minimax_music3"},
            "h3_optimizations": {
                "override_attention": "sol",
                "skip_steps_cache_type": "first_block",
                "skip_steps_multiplier": 0.08,
                "skip_steps_start_step_perc": 25,
            },
        })
        self.assertEqual(result["studio_video_workflow"], "references")
        self.assertEqual(result["audio_sub_mode"], "music")
        self.assertEqual(result["selected_model_per_audio_sub_mode"]["music"], "minimax_music3")
        self.assertEqual(result["h3_optimizations"]["override_attention"], "sol")
        self.assertNotIn("prompt", result)

    def test_backend_rejects_invalid_preference_values(self):
        normalize = _load_normalizers()["_normalize_studio_preferences"]
        with self.assertRaises(ValueError):
            normalize({"audio_sub_mode": "video"})
        with self.assertRaises(ValueError):
            normalize({"h3_optimizations": {"override_attention": "unknown"}})

    def test_step_counts_round_trip_independently_without_project_inputs(self):
        normalize = _load_normalizers()["_normalize_studio_preferences"]
        steps = {"minimax_h3_fused_turbo": 12, "minimax_h3_ref2va_fused_turbo": 9}
        saved = normalize({"inference_steps_per_model": steps, "prompt": "Temporary scene", "seed": 42})
        restored = normalize(json.loads(json.dumps(saved)))
        self.assertEqual(restored, {"inference_steps_per_model": steps})
        # Older clients that save navigation alone must not erase step choices.
        self.assertEqual(normalize({"generation_mode": "image"}, current=restored)["inference_steps_per_model"], steps)

    def test_invalid_step_preferences_are_rejected(self):
        normalize = _load_normalizers()["_normalize_studio_preferences"]
        for bad in (None, [], {"": 8}, {"model": True}, {"model": 5.5}, {"model": 0}, {"model": 1001}, {"model": "12"}, {"model": float("nan")}, {"model": float("inf")}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                normalize({"inference_steps_per_model": bad})

    def test_enhancement_default_is_explicit_and_survives_unrelated_saves(self):
        normalize = _load_normalizers()["_normalize_studio_preferences"]
        self.assertNotIn("enhance_on_generation_default", normalize({}))
        for enabled in (True, False):
            saved = normalize({"enhance_on_generation_default": enabled, "enhanceOnGeneration": not enabled})
            restored = normalize(json.loads(json.dumps(saved)))
            self.assertEqual(restored, {"enhance_on_generation_default": enabled})
            self.assertEqual(normalize({"generation_mode": "image"}, current=restored)["enhance_on_generation_default"], enabled)

    def test_enhancement_default_rejects_non_boolean_opt_in(self):
        normalize = _load_normalizers()["_normalize_studio_preferences"]
        for bad in (None, 0, 1, "true", "false", [], {}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                normalize({"enhance_on_generation_default": bad})

    def test_api_and_store_restore_requested_choices(self):
        launch = _read(_LAUNCH_PATH)
        client = _read(_CLIENT_PATH)
        store = _read(_STORE_PATH)

        self.assertIn('@api.get("/api/v1/studio-preferences")', launch)
        self.assertIn('@api.put("/api/v1/studio-preferences")', launch)
        self.assertIn("fetchStudioPreferences", client)
        self.assertIn("updateStudioPreferences", client)
        self.assertIn("studioVideoWorkflow: restoredVideoWorkflow", store)
        self.assertIn("audioSubMode: restoredAudioSubMode", store)
        self.assertIn("selectedModelPerAudioSubMode", store)
        self.assertIn("const restoredH3Attention: '' | 'sol' | 'sla' | 'sdpa'", store)
        self.assertIn("override_attention: restoredH3Attention", store)
        self.assertIn("skip_steps_cache_type: h3Preferences?.skip_steps_cache_type", store)
        self.assertIn("_persistStickyStudioPreferences(get())", store)

    def test_persistence_excludes_working_project_inputs(self):
        store = _read(_STORE_PATH)
        block_start = store.index("function _persistStickyStudioPreferences")
        block_end = store.index("export const useStore", block_start)
        block = store[block_start:block_end]
        self.assertNotIn("image_start", block)
        self.assertNotIn("audio_guide", block)
        self.assertNotIn("params.prompt", block)
        self.assertNotIn("activated_loras", block)


if __name__ == "__main__":
    unittest.main()
