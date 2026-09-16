"""Missing CUDA libraries must not silently discard Director transcription."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import audio_analysis as analysis


class TranscriptionRuntimeTests(unittest.TestCase):
    @staticmethod
    def model(segments):
        return SimpleNamespace(transcribe=mock.Mock(return_value=(iter(segments), None)))

    def test_missing_library_during_lazy_iteration_retries_without_duplicate_lines(self):
        first = SimpleNamespace(start=0.0, end=1.0, text="First line.")
        second = SimpleNamespace(start=1.0, end=2.0, text="Second line.")
        def broken_segments():
            yield first
            raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
        gpu, cpu = self.model(broken_segments()), self.model([first, second])
        with mock.patch.object(analysis, "_get_whisper_model", side_effect=[gpu, cpu]) as load, mock.patch.object(analysis, "unload_whisper") as unload:
            lyrics = analysis._transcribe("speech.wav", "[Verse] First line. Second line.")
        self.assertEqual([line.text for line in lyrics], [first.text, second.text])
        self.assertEqual(load.call_args_list, [mock.call(device="cuda"), mock.call(device="cpu")])
        self.assertEqual(gpu.transcribe.call_args, cpu.transcribe.call_args)
        unload.assert_called_once()

    def test_missing_library_at_model_load_also_retries(self):
        with mock.patch.object(analysis, "_get_whisper_model", side_effect=[OSError("libcudnn.so.9: cannot open shared object file"), self.model([])]), mock.patch.object(analysis, "unload_whisper"):
            self.assertEqual(analysis._transcribe("speech.wav"), [])

    def test_other_errors_do_not_retry_on_cpu(self):
        for error in (RuntimeError("CUDA out of memory"), OSError("audio file not found"), RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED")):
            with self.subTest(error=error), mock.patch.object(analysis, "_get_whisper_model", side_effect=error) as load, mock.patch.object(analysis, "unload_whisper") as unload:
                with self.assertRaises(type(error)):
                    analysis._transcribe("speech.wav")
            self.assertEqual(load.call_count, 1)
            unload.assert_not_called()

    def test_cpu_failure_is_propagated_once(self):
        with mock.patch.object(analysis, "_get_whisper_model", side_effect=[RuntimeError("libcublas.so.12 not found"), RuntimeError("CPU failed")]) as load, mock.patch.object(analysis, "unload_whisper"):
            with self.assertRaisesRegex(RuntimeError, "CPU failed"):
                analysis._transcribe("speech.wav")
        self.assertEqual(load.call_count, 2)

    def test_cpu_whisper_uses_int8_and_is_cached_by_device(self):
        factory = mock.Mock(return_value=object())
        with mock.patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=factory)}), mock.patch.object(analysis, "_whisper_model", None), mock.patch.object(analysis, "_whisper_device", ""), mock.patch.object(analysis.os, "makedirs"):
            first = analysis._get_whisper_model(device="cpu")
            self.assertIs(first, analysis._get_whisper_model(device="cpu"))
            factory.assert_called_once()
            self.assertEqual(factory.call_args.kwargs["compute_type"], "int8")
            self.assertEqual(factory.call_args.kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
