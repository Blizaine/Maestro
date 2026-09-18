"""Large packed reference sequences need bounded, numerically equivalent norms."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import torch
from models.minimax_h3 import transformer as h3


class H3NormMemoryTests(unittest.TestCase):
    def test_chunked_norm_matches_native_dtype_and_preserves_input(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for weight_dtype in (dtype, torch.float32):
                with self.subTest(dtype=dtype, weight_dtype=weight_dtype):
                    norm = torch.nn.RMSNorm(8, eps=1e-5, dtype=weight_dtype)
                    hidden = torch.randn(2, 19, 16, dtype=dtype)[..., ::2]
                    original = hidden.clone()
                    with torch.inference_mode():
                        expected = norm(hidden)
                        sizes = []
                        hook = norm.register_forward_pre_hook(lambda _, args: sizes.append(args[0].shape[-2]))
                        try:
                            with patch.object(h3, 'MINIMAX_H3_ACTIVATION_CHUNK_TOKENS', 4):
                                actual = h3._rms_norm_in_chunks(norm, hidden)
                        finally:
                            hook.remove()
                    self.assertEqual(sizes, [4, 4, 4, 4, 3])
                    self.assertEqual(actual.dtype, expected.dtype)
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(hidden, original, rtol=0, atol=0)

    def test_gradient_path_keeps_native_norm_and_gradients(self):
        norm = torch.nn.RMSNorm(8)
        hidden = torch.randn(1, 17, 8, requires_grad=True)
        expected = norm(hidden)
        expected_grads = torch.autograd.grad(expected.square().sum(), (hidden, norm.weight))
        sizes = []
        hook = norm.register_forward_pre_hook(lambda _, args: sizes.append(args[0].shape[-2]))
        try:
            with patch.object(h3, 'MINIMAX_H3_ACTIVATION_CHUNK_TOKENS', 4):
                actual = h3._rms_norm_in_chunks(norm, hidden)
        finally:
            hook.remove()
        self.assertEqual(sizes, [17])
        actual_grads = torch.autograd.grad(actual.square().sum(), (hidden, norm.weight))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_full_block_and_output_norm_keep_inference_math(self):
        torch.manual_seed(139)
        block = h3.MiniMaxH3Block(8, 1, 8, 12, 2, 1e-5, torch.float32,
                                 compressed_modulation=False).eval()
        final = h3.MiniMaxH3FinalLayer(8, 2, 2, 3, 1e-5, torch.float32,
                                      compressed_modulation=False).eval()
        hidden = torch.randn(1, 17, 8)
        curve = torch.randn(2, 2)
        runs = ((0, 9, 0), (9, 17, 1))
        rotary = h3.MiniMaxH3RotaryEmbedding(1)(torch.zeros(17, 3))
        with torch.inference_mode():
            with patch.object(h3, 'MINIMAX_H3_ACTIVATION_CHUNK_TOKENS', 64):
                expected = final(block(hidden.clone(), curve, runs, rotary, None), curve, runs)
            with patch.object(h3, 'MINIMAX_H3_ACTIVATION_CHUNK_TOKENS', 4):
                actual = final(block(hidden.clone(), curve, runs, rotary, None), curve, runs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
