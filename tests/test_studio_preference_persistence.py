"""Regression coverage for durable, non-project Studio preferences."""

import ast
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
