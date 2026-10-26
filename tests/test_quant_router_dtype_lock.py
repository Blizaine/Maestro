"""A quantized router must honor dtype locks before any default-dtype setter."""
from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tests"))


class QuantRouterDtypeLockTests(unittest.TestCase):
    def test_locked_router_load_ignores_initial_and_cached_bf16_dtype(self):
        import torch
        from mmgp import quant_router
        from shared.qtypes import int8_convrot
        from test_h3_header_int8_runtime import explicit_hadamard

        original = torch.nn.Linear(256, 16, dtype=torch.bfloat16)
        original._lock_dtype = torch.float32
        router = quant_router.QLinearQuantoRouter.qcreate(original, int8_convrot._QINT8_CONVROT_QTYPE)
        self.assertEqual(router.weight.dtype, torch.float32)
        self.assertEqual(router._lock_dtype, torch.float32)
        # State loading can precede the setter, or encounter an older cached
        # dtype. The explicit precision lock takes precedence in either case.
        router._router_default_dtype = torch.bfloat16
        weight = torch.arange(16 * 256, dtype=torch.int32).remainder(63).sub(31).to(torch.int8).reshape(16, 256)
        scale = torch.linspace(0.001, 0.02, 16).reshape(16, 1)
        bias = torch.linspace(-0.1, 0.2, 16).to(torch.bfloat16)
        marker = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
        state = int8_convrot.convert_to_quanto({
            "proj.weight": weight, "proj.weight_scale": scale, "proj.bias": bias,
            "proj.comfy_quant": torch.tensor(list(marker.encode()), dtype=torch.uint8),
        }, torch.bfloat16)["state_dict"]
        state.pop("proj.qweight", None)
        missing, unexpected, errors = [], [], []
        quant_router._load_with_qmodule(router, int8_convrot.QLinearInt8ConvRot, state,
                                       "proj.", {}, True, missing, unexpected, errors)
        self.assertEqual((missing, unexpected, errors), ([], [], []))
        self.assertEqual(router.weight.dtype, torch.float32)
        self.assertEqual(router.bias.dtype, torch.float32)
        self.assertEqual(router._convrot_default_dtype, torch.float32)
        self.assertEqual(router._lock_dtype, torch.float32)
        sample = torch.linspace(-0.6, 0.7, 512).reshape(2, 256)
        expected = torch.nn.functional.linear(sample @ explicit_hadamard(torch, 256),
                                               weight.float() * scale, bias.float())
        torch.testing.assert_close(router(sample), expected, rtol=1e-5, atol=1e-5)

    def test_unlocked_router_keeps_original_dtype(self):
        import torch
        from mmgp import quant_router
        from shared.qtypes import int8_convrot

        original = torch.nn.Linear(256, 16, dtype=torch.bfloat16)
        router = quant_router.QLinearQuantoRouter.qcreate(original, int8_convrot._QINT8_CONVROT_QTYPE)
        self.assertEqual(router.weight.dtype, torch.bfloat16)
        router.set_default_dtype(torch.float16)
        self.assertEqual(router._router_default_dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
