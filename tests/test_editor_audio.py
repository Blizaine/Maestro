import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.editor_audio import (  # noqa: E402
    analyze_editor_audio, editor_audio_capabilities, transcribe_editor_audio,
)
from services.editor_projects import EditorProjectError  # noqa: E402


class EditorAudioTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.source = self.outputs / "source.wav"
        self.source.write_bytes(b"synthetic source")
        self.project = {"workspace": "default", "assets": {"a": {
            "id": "a", "name": "source.wav", "path": str(self.source),
            "type": "audio", "origin": "output", "has_audio": True,
        }}, "tracks": [{"items": [{"id": "clip", "asset_id": "a", "start": 10,
                                   "duration": 4, "source_in": 2, "speed": 1.5}]}]}
        self.kwargs = {"save_root": str(self.outputs), "uploads_root": str(self.root / "uploads")}

    def tearDown(self):
        self.folder.cleanup()

    def cached_model(self):
        model = self.root / "ckpts/whisper/models--Systran--faster-whisper-small/snapshots/test"
        model.mkdir(parents=True)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (model / name).write_text("{}")
        return model

    def test_readiness_never_loads_or_downloads_a_model(self):
        with patch("services.editor_audio.importlib.util.find_spec", return_value=object()):
            self.assertFalse(editor_audio_capabilities(str(self.root))["transcription_available"])
            self.cached_model()
            ready = editor_audio_capabilities(str(self.root))
            self.assertTrue(ready["transcription_available"])
            self.assertFalse(ready["downloads"])
            self.assertEqual(ready["device"], "cpu")

    def test_loudness_uses_selected_range_and_peak_protection(self):
        report = {"input_i": "-24", "input_tp": "-4"}
        completed = subprocess.CompletedProcess([], 0, b"", json.dumps(report).encode())
        with patch("services.editor_audio.subprocess.run", return_value=completed) as run:
            result = analyze_editor_audio(self.project, "clip", **self.kwargs)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "2.0")
        self.assertEqual(command[command.index("-t") + 1], "6.0")
        self.assertEqual(result["gain_db"], 2.5)
        self.assertAlmostEqual(result["gain"], 10 ** (2.5 / 20))

    def test_silent_audio_and_invalid_ranges_leave_project_unchanged(self):
        before = json.dumps(self.project, sort_keys=True)
        completed = subprocess.CompletedProcess([], 0, b"", b'{"input_i":"-inf","input_tp":"-inf"}')
        with patch("services.editor_audio.subprocess.run", return_value=completed):
            with self.assertRaises(EditorProjectError):
                analyze_editor_audio(self.project, "clip", **self.kwargs)
        self.assertEqual(json.dumps(self.project, sort_keys=True), before)
        self.project["tracks"][0]["items"][0]["duration"] = 2000
        with self.assertRaisesRegex(EditorProjectError, "30 minutes"):
            analyze_editor_audio(self.project, "clip", **self.kwargs)

    def test_transcript_uses_cached_cpu_weights_and_maps_source_speed(self):
        model_path = self.cached_model()
        word = SimpleNamespace(start=1, end=2, word=" hello")
        segment = SimpleNamespace(start=1, end=4, text=" Hello there.", words=[word])
        model = MagicMock()
        model.transcribe.return_value = ([segment], SimpleNamespace(language="en"))
        constructor = MagicMock(return_value=model)
        module = SimpleNamespace(WhisperModel=constructor)
        with patch("services.editor_audio.importlib.util.find_spec", return_value=object()), \
                patch.dict(sys.modules, {"faster_whisper": module}), \
                patch("services.editor_audio.subprocess.run", return_value=subprocess.CompletedProcess([], 0, b"", b"")):
            result = transcribe_editor_audio(self.project, "clip", app_root=str(self.root), **self.kwargs)
        self.assertEqual(constructor.call_args.args[0], str(model_path))
        self.assertTrue(constructor.call_args.kwargs["local_files_only"])
        self.assertEqual(constructor.call_args.kwargs["device"], "cpu")
        self.assertEqual(result["offset"], 10)
        self.assertEqual(result["cues"][0]["start"], round(1 / 1.5, 3))
        self.assertEqual(result["cues"][0]["end"], round(4 / 1.5, 3))


if __name__ == "__main__":
    unittest.main()
