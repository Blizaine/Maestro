"""Duration, script integrity and assembly regressions for H3 Voice Audio."""
from pathlib import Path
import ast
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import torch

from models.minimax_h3.voice_audio import (
    audio_duration, audio_request, normalized_words, normalize_audio_settings, plan_audio_request, segment_prompt,
    SEGMENT_PAUSE_SECONDS,
)
from models.minimax_h3.dialogue import generate_audio, trim_dialogue_surplus, _voice_references


class LongAudioPlanningTests(unittest.TestCase):
    def test_api_and_engine_keep_the_complete_multiline_script(self):
        script = "Speaker 1: Hello.\nSpeaker 2: Welcome.\nSpeaker 1: Thank you."
        body = normalize_audio_settings({"prompt": script, "duration_seconds": 300, "multi_prompts_gen_type": 0})
        self.assertEqual(body["prompt"], script)
        self.assertEqual(body["multi_prompts_gen_type"], 2)
        app = Path(__file__).resolve().parents[1] / "app"
        source = ast.parse((app / "wgp.py").read_text(encoding="utf-8"))
        guard = next(node for node in ast.walk(source) if isinstance(node, ast.If)
                     and "_h3_omni_context_ir" in ast.unparse(node.test)
                     and "minimax_h3_audio_only" in ast.unparse(node.test))
        namespace = {"multi_prompts_gen_type": 0, "_h3_omni_context_ir": False,
                     "model_def": {"minimax_h3_audio_only": True}, "prompt": script}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "wgp.py", "exec"), namespace)
        self.assertEqual(namespace["prompts"], [script])
        launch = ast.parse((app / "launch.py").read_text(encoding="utf-8"))
        api_guard = next(node for node in ast.walk(launch) if isinstance(node, ast.If)
                         and ast.unparse(node.test) == "_generation_model_def.get('minimax_h3_audio_only')")
        api_body = {"prompt": script, "duration_seconds": 300, "multi_prompts_gen_type": 0}
        namespace = {"body": api_body, "_generation_model_def": {"minimax_h3_audio_only": True}, "HTTPException": ValueError}
        exec(compile(ast.Module(body=[api_guard], type_ignores=[]), "launch.py", "exec"), namespace)
        self.assertEqual(api_body["multi_prompts_gen_type"], 2)
        self.assertEqual(api_body["prompt"], script)
        api_body["duration_seconds"] = 301
        with self.assertRaises(ValueError):
            exec(compile(ast.Module(body=[api_guard], type_ignores=[]), "launch.py", "exec"), namespace)

    @unittest.skipUnless(shutil.which("node"), "Node is required to exercise the Studio duration code")
    def test_studio_preserves_custom_h3_audio_durations(self):
        ui = Path(__file__).resolve().parents[1] / "ui"
        if not (ui / "node_modules/typescript").exists():
            self.skipTest("UI dependencies are not installed")
        script = r'''
const fs = require('fs'), ts = require('typescript'), vm = require('vm'), assert = require('assert/strict');
const source = ts.createSourceFile('useStore.ts', fs.readFileSync('src/stores/useStore.ts', 'utf8'), ts.ScriptTarget.Latest, true);
let setter, submission;
function visit(node) {
  if (ts.isPropertyAssignment(node) && node.name.getText(source) === 'setDurationSeconds') setter = node.initializer;
  if (ts.isBinaryExpression(node) && node.left.getText(source) === 'params.duration_seconds' && node.right.getText(source).includes('audio_segment_max_seconds')) submission = node;
  ts.forEachChild(node, visit);
}
visit(source); assert.ok(setter); assert.ok(submission);
const ds = { min: 5, max: 300, default: 15 };
let state = { modelOptions: { audio_only: true, audio_segment_max_seconds: 45, duration_slider: ds }, params: {} };
const context = { get: () => state, set: update => Object.assign(state, typeof update === 'function' ? update(state) : update) };
const compiled = ts.transpile('const setDuration = ' + setter.getText(source) + '; setDuration;', { target: ts.ScriptTarget.ES2022 });
const setDuration = vm.runInNewContext(compiled, context);
for (const seconds of [5, 14.4, 20, 45, 60, 300]) {
  setDuration(seconds); assert.equal(state.durationSeconds, seconds);
  const params = {}; vm.runInNewContext(submission.getText(source), { params, state, ds, sliderDefault: 15 });
  assert.equal(params.duration_seconds, seconds);
  assert.equal(state.params.minimax_h3_multi_window, false);
}
setDuration(301); assert.equal(state.durationSeconds, 300);
setDuration(NaN); assert.equal(state.durationSeconds, 15);
'''
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ui, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_sound_has_45_second_passes_and_five_minute_total(self):
        for duration in (5, 15, 44.9, 45, 45.1, 90, 299.9, 300):
            with self.subTest(duration=duration):
                plan = plan_audio_request("Sound: Steady rain on a roof.", duration, 71)
                self.assertTrue(all(5 <= part.duration_s <= 45 for part in plan))
                self.assertAlmostEqual(sum(part.duration_s for part in plan), duration)
                self.assertEqual([part.seed for part in plan], [71 + i * 1000 for i in range(len(plan))])
        self.assertEqual(audio_duration(None), 15)
        for invalid in (0, -1, 4.9, 300.1, float("nan"), float("inf"), "unknown"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                plan_audio_request("Hello.", invalid)
        with self.assertRaises(ValueError):
            audio_request("Hello.", 45.1)

    def test_long_monologue_keeps_every_word_once_in_order(self):
        script = " ".join(f"word{i}" for i in range(790)) + "."
        plan = plan_audio_request(script, 300)
        self.assertGreater(len(plan), 1)
        self.assertEqual(" ".join(part.text for part in plan), script)
        self.assertTrue(all(part.duration_s <= 45 for part in plan))
        self.assertLessEqual(sum(p.duration_s for p in plan) + SEGMENT_PAUSE_SECONDS * (len(plan) - 1), 300)
        self.assertTrue(all(len(normalized_words(p.text)) / p.duration_s <= 3 + 1e-6 for p in plan))
        with self.assertRaisesRegex(ValueError, "complete H3 script needs"):
            plan_audio_request(script, 45)
        with self.assertRaisesRegex(ValueError, "shorten the script"):
            plan_audio_request(" ".join(["word"] * 1000), 300)

    def test_sentence_boundaries_and_default_pace(self):
        sentence = "This is a sentence with exactly ten words here."
        plan = plan_audio_request(" ".join([sentence] * 20), 100)
        self.assertTrue(all(part.text.endswith(".") for part in plan))
        self.assertEqual(" ".join(part.text for part in plan), " ".join([sentence] * 20))
        normal = plan_audio_request(" ".join(["word"] * 56), 45)[0]
        self.assertAlmostEqual(normal.duration_s, 56 / 2.8 + 1.4)

    def test_speaker_and_language_directions_are_not_spoken(self):
        plan = plan_audio_request("Speaker 1: [French, calm] Bonjour, mon ami.\n"
                                  "Speaker 2: [English, excited] Hello there.\n"
                                  "Speaker 1: Je suis ici.", 45)
        self.assertEqual([part.speaker for part in plan], [1, 2, 1])
        self.assertEqual([part.language_code for part in plan], ["fr", "en", "fr"])
        self.assertEqual(plan[0].text, "Bonjour, mon ami.")
        self.assertIn("calm", segment_prompt(plan[0], True))
        self.assertIn("<d>[French] Bonjour, mon ami.</d>", segment_prompt(plan[0], True))
        for prompt in ("Speaker 3: Hello.", "Speaker 1: [English]", "Sound:", ""):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                plan_audio_request(prompt, 45)

    def test_short_script_does_not_fill_five_minutes(self):
        plan = plan_audio_request("Hello, welcome to Maestro.", 300)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].duration_s, 5)

    def test_non_latin_text_is_counted_and_preserved(self):
        for script in ("你好世界。" * 75, "Привет, добро пожаловать. " * 75):
            with self.subTest(script=script[:12]):
                plan = plan_audio_request(script, 300)
                self.assertGreater(len(plan), 1)
                self.assertEqual(normalized_words(" ".join(part.text for part in plan)), normalized_words(script))

    def test_native_dialogue_tags_split_without_speaking_the_visual_directions(self):
        native = ("subject_definitions: Speaker 1 (S1) and Speaker 2 (S2). "
                  "detailed_description: A studio shot. Speaker (S1) says <d>[English] Hello.</d> "
                  "Speaker (S2) says <d>[French] Bonjour.</d> overall_soundscape: Quiet.")
        self.assertEqual(plan_audio_request(native, 45)[0].native_prompt, native)
        long_plan = plan_audio_request(native, 300)
        self.assertEqual([part.text for part in long_plan], ["Hello.", "Bonjour."])
        self.assertEqual([part.speaker for part in long_plan], [1, 2])
        self.assertTrue(all("studio shot" not in part.text for part in long_plan))

    def test_handler_keeps_total_duration_separate_from_hidden_canvas(self):
        from models.minimax_h3.minimax_h3_handler import family_handler
        definition = family_handler.query_model_def("minimax_h3_voice_audio", {})
        self.assertEqual(definition["duration_slider"]["max"], 300)
        self.assertEqual(definition["duration_slider"]["default"], 15)
        self.assertEqual(definition["audio_segment_max_seconds"], 45)
        settings = {"prompt": "Hello.", "duration_seconds": 300, "num_inference_steps": 20}
        family_handler.fix_settings("minimax_h3_voice_audio", 2.58, definition, settings)
        self.assertIsNone(family_handler.validate_generative_settings("minimax_h3_voice_audio", definition, settings))
        self.assertEqual(settings["duration_seconds"], 300)
        self.assertEqual(settings["resolution"], "32x32")
        self.assertLess(settings["video_length"], 1100)
        settings["duration_seconds"] = 301
        self.assertIsNotNone(family_handler.validate_generative_settings("minimax_h3_voice_audio", definition, settings))


class AudioAssemblyTests(unittest.TestCase):
    def pipeline(self):
        return SimpleNamespace(_interrupt=False, dialogue_whisper=Mock(), generate=Mock())

    def test_joined_output_uses_trimmed_duration_and_reuses_each_voice(self):
        pipeline = self.pipeline()
        paths, calls = [], []
        def generate(**kwargs):
            calls.append(kwargs)
            reference = kwargs.get("audio_guide")
            if reference:
                self.assertTrue(Path(reference).is_file())
                paths.append(reference)
            self.assertTrue(kwargs["_audio_segment"])
            return {"x": torch.ones(2, 160000), "audio_sampling_rate": 32000}
        pipeline.generate.side_effect = generate
        with patch("models.minimax_h3.dialogue.trim_dialogue_surplus", side_effect=lambda model, audio, *a, **k: audio[:, :32000]) as trim:
            result = generate_audio(pipeline, "Speaker 1: One.\nSpeaker 2: Two.\nSpeaker 1: Three.\nSpeaker 2: Four.", duration_seconds=45)
        self.assertEqual(trim.call_count, 4)
        self.assertEqual([call["_audio_speaker"] for call in calls], [1, 2, 1, 2])
        self.assertIsNone(calls[0]["audio_guide"])
        self.assertIsNone(calls[1]["audio_guide"])
        self.assertNotEqual(calls[2]["audio_guide"], calls[3]["audio_guide"])
        self.assertTrue(all(not Path(path).exists() for path in paths))
        self.assertEqual(result["x"].shape, (2, round((4 + 3 * .18) * 32000)))
        self.assertEqual(result["overridden_inputs"]["duration_seconds"], 4.54)
        self.assertEqual(result["overridden_inputs"]["audio_segment_count"], 4)

    def test_cancellation_between_segments_returns_no_partial_output(self):
        pipeline = self.pipeline()
        def generate(**kwargs):
            pipeline._interrupt = True
            return {"x": torch.zeros(2, 160000), "audio_sampling_rate": 32000}
        pipeline.generate.side_effect = generate
        self.assertIsNone(generate_audio(pipeline, "Speaker 1: First.\nSpeaker 1: Second.", duration_seconds=45))
        self.assertEqual(pipeline.generate.call_count, 1)

    def test_cancellation_during_alignment_returns_no_output(self):
        pipeline = self.pipeline()
        pipeline.generate.return_value = {"x": torch.zeros(2, 160000), "audio_sampling_rate": 32000}
        def trim(model, audio, *args, **kwargs):
            pipeline._interrupt = True
            return audio
        with patch("models.minimax_h3.dialogue.trim_dialogue_surplus", side_effect=trim):
            self.assertIsNone(generate_audio(pipeline, "Hello.", duration_seconds=45))

    def test_failed_generation_cleans_temporary_voice_references(self):
        from models.minimax_h3 import dialogue
        pipeline = self.pipeline()
        pipeline.generate.side_effect = [
            {"x": torch.zeros(2, 160000), "audio_sampling_rate": 32000},
            RuntimeError("synthetic generation failure"),
        ]
        original, paths = dialogue._write_reference, []
        def write(path, audio, rate):
            paths.append(path)
            return original(path, audio, rate)
        with patch.object(dialogue, "_write_reference", side_effect=write), self.assertRaisesRegex(RuntimeError, "synthetic"):
            generate_audio(pipeline, "Speaker 1: Hello.\nSpeaker 1: Welcome.", duration_seconds=45)
        self.assertTrue(paths)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_sound_assembly_reaches_300_without_a_larger_native_pass(self):
        pipeline = self.pipeline()
        pipeline.generate.side_effect = lambda **kw: {"x": torch.zeros(2, round(kw["duration_seconds"] * 32000)), "audio_sampling_rate": 32000}
        with patch("models.minimax_h3.dialogue.trim_dialogue_surplus") as trim:
            result = generate_audio(pipeline, "Sound: Rain.", duration_seconds=300)
        self.assertEqual(pipeline.generate.call_count, 7)
        self.assertTrue(all(call.kwargs["duration_seconds"] <= 45 for call in pipeline.generate.call_args_list))
        self.assertLessEqual(result["x"].shape[-1], 300 * 32000)
        self.assertAlmostEqual(result["x"].shape[-1] / 32000, 300, places=3)
        trim.assert_not_called()

    def test_assembly_stays_on_cpu_when_torch_default_device_is_not_cpu(self):
        pipeline = self.pipeline()
        pipeline.generate.side_effect = lambda **kw: {
            "x": torch.zeros(2, round(kw["duration_seconds"] * 32000), device="cpu"),
            "audio_sampling_rate": 32000,
        }
        with torch.device("meta"):
            result = generate_audio(pipeline, "Sound: Rain.", duration_seconds=46)
        self.assertEqual(result["x"].device.type, "cpu")
        self.assertEqual(result["x"].shape[-1], 46 * 32000)

    def test_reference_budget_is_fifteen_seconds_total(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder, patch("models.minimax_h3.ref2va.decode_reference_audio", return_value=(torch.ones(1, 20 * 32000), 32000)):
            with patch("models.minimax_h3.dialogue._write_reference", side_effect=lambda path, audio, rate: (audio.shape[-1] / rate, audio.shape[0])):
                refs = _voice_references(folder, {1: "one.wav", 2: "two.wav"})
        self.assertEqual(refs, {1: (7.5, 2), 2: (7.5, 2)})

    def test_whisper_trims_only_confident_script_boundaries(self):
        words = [dict(word=word, start=start, end=start+.3, probability=.99)
                 for word, start in (("extra", .2), ("hello", 1), ("there", 1.5), ("extra", 3))]
        audio = torch.zeros(2, 160000)
        with patch("models.minimax_h3.dialogue._transcribe_words", return_value=words):
            trimmed = trim_dialogue_surplus(Mock(), audio, 32000, "Hello there.", "en")
            self.assertLess(trimmed.shape[-1], audio.shape[-1])
            untouched = trim_dialogue_surplus(Mock(), audio, 32000, "Entirely different words and sentence.", "en")
            self.assertIs(untouched, audio)

    def test_uncertain_expected_words_are_never_removed_to_find_a_boundary(self):
        audio = torch.zeros(2, 160000)
        audio[:, 0] = .7
        words = [dict(word="yes", start=0., end=1.4, probability=.1),
                 dict(word="maestro", start=1.5, end=2., probability=.99)]
        with patch("models.minimax_h3.dialogue._transcribe_words", return_value=words):
            result = trim_dialogue_surplus(Mock(), audio, 32000, "Yes, Maestro.", "en")
        torch.testing.assert_close(result[:, 0], audio[:, 0])
        words[0].update(probability=.99, end=.3)
        words[-1]["probability"] = .1
        with patch("models.minimax_h3.dialogue._transcribe_words", return_value=words):
            result = trim_dialogue_surplus(Mock(), audio, 32000, "Yes, Maestro.", "en")
        self.assertEqual(result.shape, audio.shape)


if __name__ == "__main__":
    unittest.main()
