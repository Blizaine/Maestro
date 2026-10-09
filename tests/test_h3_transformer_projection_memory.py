"""Bounded native H3 input/output projections preserve inference behavior."""

from pathlib import Path
import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import torch
from models.minimax_h3 import transformer as h3


class H3TransformerProjectionMemoryTests(unittest.TestCase):
    def _model_and_inputs(self):
        torch.manual_seed(173)
        model = h3.MiniMaxH3Transformer(
            hidden_size=8,
            num_layers=1,
            token_refiner_layers=1,
            num_attention_heads=1,
            attention_head_dim=8,
            ffn_dim=12,
            video_channels=2,
            audio_channels=3,
            patch_size=(1, 1, 1),
            text_dim=6,
            curve_grid=4,
            curve_dim=2,
            rope_freq_dim=1,
            dtype=torch.float32,
        ).eval()
        model.adaln_t_table.data.copy_(torch.tensor([
            [0.0, 0.0],
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]))
        inputs = {
            "hidden_states": torch.randn(1, 3, 2),
            "audio_hidden_states": torch.randn(1, 4, 3),
            "encoder_hidden_states": torch.randn(1, 2, 6),
            "timestep": torch.tensor([0.1, 0.4]),
            "timestep_indices": torch.tensor([0, 0, 1, 1, 1, 1, 0, 0, 0]),
            "token_tags": torch.tensor([1, 1, 2, 2, 2, 2, 0, 0, 0]),
            "position_ids": torch.zeros(9, 3),
            "video_indices": torch.tensor([6, 7, 8]),
            "audio_indices": torch.tensor([2, 3, 4, 5]),
            "text_indices": torch.tensor([0, 1]),
            "return_dict": False,
        }
        return model, inputs

    def test_tiny_forward_is_numerically_stable_and_releases_projection_chunks(self):
        model, inputs = self._model_and_inputs()
        with torch.inference_mode(), patch.object(
            h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 64
        ):
            expected_video, expected_audio = model(**inputs)

        video_input_sizes = []
        video_output_sizes = []
        audio_input_sizes = []
        audio_output_sizes = []
        video_projection_refs = []
        video_output_refs = []
        input_outputs_alive_at_block = []
        output_chunks_alive_at_audio_head = []

        def track_video_projection(_module, args, output):
            video_input_sizes.append(args[0].shape[1])
            video_projection_refs.append(weakref.ref(output))

        def check_projection_lifetime(_module, _args):
            input_outputs_alive_at_block.append(
                any(reference() is not None for reference in video_projection_refs)
            )

        def track_video_output_input(_module, args):
            video_output_sizes.append(args[0].shape[1])

        def track_video_output(_module, _args, output):
            video_output_refs.append(weakref.ref(output))

        def track_audio_input(_module, args):
            audio_output_sizes.append(args[0].shape[1])
            output_chunks_alive_at_audio_head.append(
                any(reference() is not None for reference in video_output_refs)
            )

        def track_audio_projection_input(_module, args):
            audio_input_sizes.append(args[0].shape[1])

        hooks = [
            model.video_patch_proj.register_forward_hook(track_video_projection),
            model.blocks[0].register_forward_pre_hook(check_projection_lifetime),
            model.final_layer.video_out.register_forward_pre_hook(track_video_output_input),
            model.final_layer.video_out.register_forward_hook(track_video_output),
            model.final_layer.audio_out.register_forward_pre_hook(track_audio_input),
            model.audio_patch_proj.register_forward_pre_hook(track_audio_projection_input),
        ]
        try:
            with torch.inference_mode(), patch.object(
                h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 2
            ):
                actual_video, actual_audio = model(**inputs)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(video_input_sizes, [2, 1])
        self.assertEqual(audio_input_sizes, [2, 2])
        self.assertEqual(video_output_sizes, [2, 1])
        self.assertEqual(audio_output_sizes, [2, 2])
        self.assertEqual(input_outputs_alive_at_block, [False])
        self.assertEqual(output_chunks_alive_at_audio_head, [False, False])
        torch.testing.assert_close(actual_video, expected_video, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual_audio, expected_audio, rtol=2e-5, atol=2e-5)

    def test_input_projection_falls_back_to_fp32_for_missing_or_uint8_weight_dtype(self):
        class ProjectionProbe(torch.nn.Module):
            def __init__(self, reported_dtype):
                super().__init__()
                self.projection = torch.nn.Linear(3, 4)
                self.weight = (
                    None
                    if reported_dtype is None
                    else SimpleNamespace(dtype=reported_dtype)
                )
                self.out_features = 4
                self.input_dtypes = []

            def forward(self, rows):
                self.input_dtypes.append(rows.dtype)
                return self.projection(rows)

        indices = torch.arange(5)
        input_rows = torch.randn(1, 5, 3, dtype=torch.bfloat16)
        for reported_dtype in (None, torch.uint8):
            with self.subTest(reported_dtype=reported_dtype):
                projection = ProjectionProbe(reported_dtype)
                packed = torch.zeros(1, 5, 4, dtype=torch.bfloat16)
                with torch.inference_mode(), patch.object(
                    h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 2
                ):
                    h3._project_token_rows_into_packed(
                        projection, input_rows, indices, packed
                    )
                    expected = projection.projection(input_rows.float()).to(
                        packed.dtype
                    )
                self.assertEqual(
                    projection.input_dtypes,
                    [torch.float32, torch.float32, torch.float32],
                )
                torch.testing.assert_close(packed, expected)

    def test_chunked_output_projection_preserves_autocast_result_dtype(self):
        torch.manual_seed(211)
        projection = torch.nn.Linear(4, 2)
        packed = torch.randn(1, 5, 4, dtype=torch.bfloat16)
        indices = torch.arange(5)
        with torch.inference_mode(), torch.autocast(
            "cpu", dtype=torch.bfloat16
        ):
            expected = projection(packed.index_select(1, indices).float())
            with patch.object(h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 2):
                actual = h3._project_selected_rows(projection, packed, indices)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_grad_enabled_projection_paths_keep_full_calls_and_gradients(self):
        rows = torch.randn(1, 5, 3, requires_grad=True)
        input_projection = torch.nn.Linear(3, 4)
        packed = torch.zeros(1, 5, 4)
        indices = torch.arange(5)
        input_sizes = []
        input_hook = input_projection.register_forward_pre_hook(
            lambda _module, args: input_sizes.append(args[0].shape[1])
        )
        try:
            with patch.object(h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 2):
                h3._project_token_rows_into_packed(
                    input_projection, rows, indices, packed
                )
        finally:
            input_hook.remove()
        packed.sum().backward()
        self.assertEqual(input_sizes, [5])
        self.assertIsNotNone(rows.grad)
        self.assertIsNotNone(input_projection.weight.grad)

        packed_input = torch.randn(1, 5, 4, requires_grad=True)
        output_projection = torch.nn.Linear(4, 2)
        selected_sizes = []
        output_hook = output_projection.register_forward_pre_hook(
            lambda _module, args: selected_sizes.append(args[0].shape[1])
        )
        try:
            with patch.object(h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 2):
                output = h3._project_selected_rows(
                    output_projection, packed_input, indices
                )
        finally:
            output_hook.remove()
        output.sum().backward()
        self.assertEqual(selected_sizes, [5])
        self.assertIsNotNone(packed_input.grad)
        self.assertIsNotNone(output_projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
