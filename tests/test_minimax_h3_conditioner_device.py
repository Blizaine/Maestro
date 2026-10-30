"""CPU preprocessing and explicit-device handoff for MiniMax H3 Ref2VA."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


class MiniMaxH3ConditionerDeviceTests(unittest.TestCase):
    def test_ref2va_preprocessors_stay_cpu_and_encoder_handoff_is_explicit(self):
        from models.minimax_h3.conditioner import (
            IMAGE_TOKEN_ID,
            VIDEO_TOKEN_ID,
            MiniMaxH3Conditioner,
        )

        class RecordingImageProcessor:
            merge_size = 1

            def __init__(self):
                self.output = None

            def __call__(self, *, images, return_tensors):
                self.assert_request(images, return_tensors)
                self.output = {
                    "pixel_values": torch.ones((1, 3, 2, 2)),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                }
                return self.output

            @staticmethod
            def assert_request(images, return_tensors):
                assert len(images) == 1
                assert return_tensors == "pt"

        class RecordingVideoProcessor:
            def __init__(self):
                self.output = None

            def __call__(self, *, videos, do_sample_frames, return_tensors):
                assert len(videos) == 1
                assert do_sample_frames is False
                assert return_tensors == "pt"
                self.output = {
                    "pixel_values_videos": torch.ones((1, 1, 3, 2, 2)),
                    "video_grid_thw": torch.tensor([[1, 1, 1]]),
                }
                return self.output

        class RecordingVisual:
            def __init__(self):
                self.calls = []
                self._interrupt = False

            def __call__(self, pixels, *, grid_thw):
                self.calls.append((pixels.device, pixels.dtype, grid_thw.device))
                return torch.ones((1, 1, 4), device=pixels.device), None

        class RecordingModel:
            def __init__(self):
                self._interrupt = False
                self.embed_devices = []

            def embed_tokens(self, input_ids):
                self.embed_devices.append(input_ids.device)
                return torch.zeros((1, input_ids.shape[-1], 4), device=input_ids.device)

            def __call__(self, *, inputs_embeds, **_kwargs):
                return SimpleNamespace(
                    last_hidden_state=inputs_embeds,
                    mid_results=[inputs_embeds],
                )

        class RecordingQwen:
            def __init__(self):
                self.model = RecordingModel()
                self.visual = RecordingVisual()
                self._interrupt = False

            @staticmethod
            def get_rope_index(input_ids, **_kwargs):
                return torch.zeros((3, 1, input_ids.shape[-1]), device=input_ids.device), None

        image_processor = RecordingImageProcessor()
        video_processor = RecordingVideoProcessor()
        qwen = RecordingQwen()
        conditioner = MiniMaxH3Conditioner(
            qwen,
            tokenizer=None,
            processor=SimpleNamespace(
                image_processor=image_processor,
                video_processor=video_processor,
            ),
        )
        image_reference = SimpleNamespace(kind="image", image=object())
        video_reference = SimpleNamespace(
            kind="video",
            frames=np.zeros((1, 2, 2, 3), dtype=np.uint8),
        )
        requested_device = torch.device("meta")
        process_default = torch.empty(0).device

        def build_presentation(_tokenizer, _prompt, references, *_counts):
            token_id = IMAGE_TOKEN_ID if references[0].kind == "image" else VIDEO_TOKEN_ID
            return [token_id], [0]

        with patch(
            "models.minimax_h3.ref2va.build_ref2va_presentation",
            side_effect=build_presentation,
        ), patch(
            "models.minimax_h3.ref2va.sample_reference_video_frames",
            return_value=([np.zeros((2, 2, 3), dtype=np.uint8)], [0.0]),
        ):
            with torch.device("meta"):
                image_embeddings, image_tags = conditioner.forward_ref2va(
                    "test prompt",
                    requested_device,
                    [image_reference],
                )
                video_embeddings, video_tags = conditioner.forward_ref2va(
                    "test prompt",
                    requested_device,
                    [video_reference],
                )
                # The scoped CPU processor contexts must restore the outer
                # non-CPU factory default for work after preprocessing.
                self.assertEqual(torch.empty(0).device, requested_device)

        self.assertEqual(torch.empty(0).device, process_default)
        self.assertEqual(image_embeddings.device, requested_device)
        self.assertEqual(video_embeddings.device, requested_device)
        self.assertEqual(image_tags.device, requested_device)
        self.assertEqual(video_tags.device, requested_device)
        for output in (image_processor.output, video_processor.output):
            self.assertIsNotNone(output)
            self.assertTrue(all(tensor.device.type == "cpu" for tensor in output.values()))
        self.assertEqual(
            qwen.visual.calls,
            [
                (requested_device, torch.float32, requested_device),
                (requested_device, torch.float32, requested_device),
            ],
        )
        self.assertEqual(qwen.model.embed_devices, [requested_device, requested_device])


if __name__ == "__main__":
    unittest.main()
