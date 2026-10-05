"""Exercise the real rotary consumer without loading transformer weights."""
import ast
from pathlib import Path
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app/models/minimax_h3/transformer.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
rotary = next(node for node in tree.body
              if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3RotaryEmbedding")
scope = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=[rotary], type_ignores=[]), str(SOURCE), "exec"), scope)
RotaryEmbedding = scope["MiniMaxH3RotaryEmbedding"]


class H3CheckpointRotaryPrecisionTests(unittest.TestCase):
    def test_half_precision_checkpoint_frequencies_produce_fp32_rotary_values(self):
        for stored_dtype in (torch.bfloat16, torch.float16):
            for input_dtype in (torch.bfloat16, torch.float16):
                with self.subTest(stored_dtype=stored_dtype, input_dtype=input_dtype):
                    frequencies = torch.linspace(0.001, 0.997, 16).to(stored_dtype)
                    module = RotaryEmbedding()
                    module.load_state_dict({"inv_freq": frequencies}, assign=True)
                    positions = torch.tensor(
                        [[0, 0, 0], [7, 13, 19], [48, 128, 256]], dtype=input_dtype,
                    )
                    cosine, sine = module(positions)
                    self.assertEqual(cosine.dtype, torch.float32)
                    self.assertEqual(sine.dtype, torch.float32)
                    self.assertEqual(cosine.shape, (3, 96))
                    # Independent double-precision reference preserves the
                    # checkpoint's stored frequencies and modality ordering.
                    angles = torch.cat([
                        positions[:, axis:axis + 1].double() * frequencies.double()
                        for axis in range(3)
                    ], dim=1)
                    angles = torch.cat([angles, angles], dim=1)
                    torch.testing.assert_close(cosine, angles.cos().float(), rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(sine, angles.sin().float(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
