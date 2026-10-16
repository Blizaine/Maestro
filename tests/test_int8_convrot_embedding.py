"""CPU regressions for INT8 ConvRot embedding loads and row-wise gather."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

try:
    import torch

    from mmgp import offload, quant_router
    from shared.qtypes import int8_convrot

    # MMGP's registered embedding implementation is needed for quantized
    # embedding weights; the ConvRot post-load hook then installs its forward.
    from shared.qtypes import gguf as _gguf  # noqa: F401

    MMGP_AVAILABLE = True
except ImportError:
    torch = None
    offload = None
    quant_router = None
    int8_convrot = None
    MMGP_AVAILABLE = False


def _make_convrot_state(source_weight: torch.Tensor, group_size: int) -> dict[str, torch.Tensor]:
    hadamard = int8_convrot._regular_hadamard(group_size, "cpu", torch.float32)
    rotated = torch.matmul(
        source_weight.float().reshape(source_weight.shape[0], -1, group_size),
        hadamard.T,
    ).reshape_as(source_weight)
    scale = rotated.abs().amax(dim=-1, keepdim=True).div(127.0).clamp_min_(1e-30)
    quantized = (rotated / scale).round().clamp_(-128, 127).to(torch.int8)
    marker = json.dumps({
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": group_size,
    }).encode("utf-8")
    return {
        "embedding.weight": quantized,
        "embedding.weight_scale": scale,
        "embedding.comfy_quant": torch.tensor(list(marker), dtype=torch.uint8),
    }


@unittest.skipUnless(MMGP_AVAILABLE, "vendored MMGP / Quanto runtime is unavailable")
class Int8ConvRotEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.handler_was_registered = "shared.qtypes.int8_convrot" in quant_router._HANDLER_MODULES
        quant_router.register_handler("shared.qtypes.int8_convrot")

    def tearDown(self):
        if not self.handler_was_registered:
            quant_router.unregister_handler("shared.qtypes.int8_convrot")

    def test_mmgp_load_restores_selected_rotated_rows_with_scale_and_embedding_rules(self):
        torch.manual_seed(7162)
        vocab, width, group_size = 37, 64, 64
        source_weight = torch.randn(vocab, width, dtype=torch.float32).mul_(0.17).to(torch.bfloat16)
        plain_weight = torch.randn(vocab, width, dtype=torch.bfloat16).mul_(0.08)

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(
                    vocab, width, max_norm=0.72, norm_type=2.0, dtype=torch.bfloat16
                )
                self.embedding.embed_scale = torch.tensor(0.625, dtype=torch.float32)
                self.plain_embedding = torch.nn.Embedding(vocab, width, dtype=torch.bfloat16)

        model = TinyEncoder()
        quantized_state = _make_convrot_state(source_weight, group_size)
        quantized_state["plain_embedding.weight"] = plain_weight.clone()

        offload.load_model_data(
            model,
            (quantized_state, None),
            default_dtype=torch.bfloat16,
            verboseLevel=0,
        )

        self.assertEqual(model.embedding._convrot_group_size, group_size)
        self.assertIsNot(model.embedding.forward.__func__, torch.nn.Embedding.forward)
        self.assertEqual(model.plain_embedding.forward.__func__, torch.nn.Embedding.forward)
        self.assertFalse(hasattr(model.plain_embedding, "_convrot_group_size"))
        self.assertEqual(model.plain_embedding.weight.dtype, torch.bfloat16)
        torch.testing.assert_close(model.plain_embedding.weight, plain_weight)

        ids = torch.tensor([[1, 5, 8], [13, 21, 34]], dtype=torch.long)
        qweight = model.embedding.qweight
        self.assertEqual(tuple(qweight._data.shape), (vocab, width))
        self.assertEqual(tuple(qweight._scale.shape), (vocab, 1))
        self.assertEqual(qweight._data.dtype, torch.int8)
        self.assertEqual(qweight._scale.dtype, torch.float32)

        embedding_calls = []
        functional_embedding = torch.nn.functional.embedding

        def observe_gather(input_ids, weight, *args, **kwargs):
            output = functional_embedding(input_ids, weight, *args, **kwargs)
            embedding_calls.append((tuple(input_ids.shape), tuple(weight.shape), tuple(output.shape)))
            return output

        with patch("torch.nn.functional.embedding", side_effect=observe_gather):
            actual = model.embedding(ids)

        # Both the int8 rows and their FP32 scales are looked up by token ID;
        # the implementation never expands/dequantizes the entire vocab table.
        self.assertEqual(len(embedding_calls), 2)
        self.assertEqual(embedding_calls[0], ((2, 3), (vocab, width), (2, 3, width)))
        self.assertEqual(embedding_calls[1], ((2, 3), (vocab, 1), (2, 3, 1)))

        reference = torch.nn.Embedding(
            vocab, width, max_norm=0.72, norm_type=2.0, dtype=torch.bfloat16
        )
        reference.weight = torch.nn.Parameter(source_weight.clone())
        expected = reference(ids) * torch.tensor(0.625, dtype=torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.01)

        plain_actual = model.plain_embedding(ids)
        plain_expected = torch.nn.functional.embedding(ids, plain_weight)
        torch.testing.assert_close(plain_actual, plain_expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
