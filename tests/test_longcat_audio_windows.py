"""CPU regression for continuation audio-window timestamp coverage."""
from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch


_MAIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "models"
    / "longcat"
    / "longcat_main.py"
)


def _load_audio_window_builder(audio_loader):
    tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"), filename=str(_MAIN_PATH))
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LongCatModel"
    )
    method = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_audio_windows"
    )
    namespace = {
        "librosa": SimpleNamespace(load=audio_loader),
        "np": np,
        "torch": torch,
    }
    exec(compile(ast.unparse(method), str(_MAIN_PATH), "exec"), namespace)
    return namespace["_build_audio_windows"]


class LongCatAudioWindowTests(unittest.TestCase):
    def _build_window(self, loaded_audio):
        def load_audio(path, sr):
            return loaded_audio, sr

        method = _load_audio_window_builder(load_audio)
        encoded_lengths = []
        model = SimpleNamespace(transformer=SimpleNamespace(audio_window=5))

        def encode_audio(self, samples, fps, device, sample_rate):
            encoded_lengths.append(len(samples))
            sequence_length = int(len(samples) / sample_rate * fps)
            return torch.arange(sequence_length, dtype=torch.float32).unsqueeze(-1)

        model._get_audio_embedding = MethodType(encode_audio, model)
        windows = method(model, "guide.wav", 16, 16, 60, 2)
        return windows, encoded_lengths

    def test_short_audio_is_padded_through_later_window_timestamp(self):
        loaded_audio = np.zeros(16_000, dtype=np.float32)
        windows, encoded_lengths = self._build_window(loaded_audio)

        self.assertEqual(encoded_lengths, [76_000])
        self.assertEqual(tuple(windows.shape), (1, 16, 5, 1))
        torch.testing.assert_close(
            windows[0, :, 2, 0],
            torch.arange(120, 152, 2, dtype=torch.float32),
        )

    def test_long_guide_is_not_trimmed_for_later_window(self):
        loaded_audio = np.zeros(160_000, dtype=np.float32)

        windows, encoded_lengths = self._build_window(loaded_audio)

        self.assertEqual(encoded_lengths, [160_000])
        self.assertEqual(tuple(windows.shape), (1, 16, 5, 1))
        torch.testing.assert_close(
            windows[0, :, 2, 0],
            torch.arange(120, 152, 2, dtype=torch.float32),
        )


if __name__ == "__main__":
    unittest.main()
