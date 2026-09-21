import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest


PATH = Path(__file__).resolve().parents[1] / "app/models/TTS/yue2/prompting.py"
sys.path.insert(0, str(PATH.parents[3]))
spec = importlib.util.spec_from_file_location("yue2_prompting", PATH)
prompting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompting)


class Yue2InputsTests(unittest.TestCase):
    def test_old_minimal_job_uses_safe_defaults(self):
        for value in ({}, {"model_mode": None}):
            inputs = {"alt_prompt": "Acoustic pop", **value}
            with self.subTest(value=value):
                self.assertIsNone(prompting.validate_song_inputs(inputs, "[Verse]\nHello"))
                self.assertEqual(inputs["model_mode"], 2)

    def test_explicit_composition_choices_are_preserved(self):
        for mode in (0, 1, 2):
            inputs = {"alt_prompt": "Pop", "model_mode": mode}
            with self.subTest(mode=mode):
                self.assertIsNone(prompting.validate_song_inputs(inputs, "Hello"))
                self.assertEqual(inputs["model_mode"], mode)
        for mode in (False, True, "0", "", 0.0, -1, 3):
            with self.subTest(invalid=mode):
                self.assertIn("Choose", prompting.validate_song_inputs(
                    {"alt_prompt": "Pop", "model_mode": mode}, "Hello"))

    def test_cached_model_defaults_refresh_without_changing_other_settings(self):
        # Run the real defaults loader against a temporary pre-update cache.
        # This is separate from live job validation, which must preserve mode 0.
        app = PATH.parents[3]
        source = ast.parse((app / "wgp.py").read_text(encoding="utf-8"))
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == "get_default_settings")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "settings.json"
            cache.write_text(json.dumps({"model_mode": 0, "num_inference_steps": 40}), encoding="utf-8")
            namespace = {"Path": Path, "json": json,
                         "test_class_i2v": lambda model: False,
                         "get_settings_file_name": lambda model: str(cache),
                         "fix_settings": lambda model, settings: None,
                         "get_model_def": lambda model: {"yue2_composition": model == "yue2"},
                         "args": SimpleNamespace(seed=-1, frames=0, steps=0)}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "wgp.py", "exec"), namespace)
            load = namespace["get_default_settings"]
            self.assertEqual(load("yue2"), {"model_mode": 2, "num_inference_steps": 40})
            self.assertEqual(load("another_model")["model_mode"], 0)

    def test_cover_score_and_direct_modes_have_distinct_requirements(self):
        base = {"alt_prompt": "Acoustic pop", "audio_prompt_type": "A", "model_mode": 0}
        self.assertIn("Upload", prompting.validate_song_inputs(base, "Hello"))
        base["audio_guide"] = "source.wav"
        self.assertIsNone(prompting.validate_song_inputs(base, "Hello"))
        base["model_mode"] = 2
        self.assertIn("requires", prompting.validate_song_inputs(base, "Hello"))
        base["audio_prompt_type"] = ""
        base["custom_settings"] = {"abc": "V:Vocal\nC D E F"}
        self.assertIn("planning mode", prompting.validate_song_inputs(base, "Hello"))

    def test_bad_numeric_settings_fail_before_model_load(self):
        for value in (float("nan"), float("inf"), 0, 601):
            with self.subTest(value=value):
                self.assertIn("duration", prompting.validate_song_inputs({"alt_prompt": "Pop", "duration_seconds": value}, "Hi"))

    def test_writer_receives_the_actual_short_song_limit(self):
        instruction = prompting.writer_duration_instruction(8)
        self.assertIn("8 seconds", instruction)
        self.assertIn("fewer sections", instruction)
        self.assertIn("Preserve supplied", instruction)


if __name__ == "__main__":
    unittest.main()
