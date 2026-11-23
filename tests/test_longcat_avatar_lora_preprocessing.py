"""CPU regression checks for the upstream LongCat Avatar v1.6 LoRA format."""

from pathlib import Path
import sys

import pytest
import torch


APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.longcat.lora_preprocessing import (  # noqa: E402
    make_longcat_avatar_v1_6_lora_preprocessor,
    preprocess_longcat_avatar_v1_6_lora,
)


_ESCAPED_DOT = "___lorahyphen___"


def _prefix(module_name):
    return "lora___lorahyphen___" + module_name.replace(".", _ESCAPED_DOT)


def _upstream_v1_6_state():
    projection = _prefix("blocks.0.attn.proj")
    qkv = _prefix("blocks.0.attn.qkv")
    kv = _prefix("blocks.0.cross_attn.kv_linear")
    return {
        projection + ".alpha_scale": torch.tensor(0.5),
        projection + ".lora_down.weight": torch.ones((4, 3)),
        projection + ".lora_up.weight": torch.full((5, 4), 2.0),
        qkv + ".alpha_scale": torch.tensor(0.25),
        qkv + ".lora_down.weight": torch.ones((6, 3)),
        qkv + ".lora_up.blocks.0.weight": torch.full((4, 2), 3.0),
        qkv + ".lora_up.blocks.1.weight": torch.full((4, 2), 4.0),
        qkv + ".lora_up.blocks.2.weight": torch.full((4, 2), 5.0),
        kv + ".alpha_scale": torch.tensor(1.25),
        kv + ".lora_down.weight": torch.ones((4, 3)),
        kv + ".lora_up.blocks.0.weight": torch.full((3, 2), 6.0),
        kv + ".lora_up.blocks.1.weight": torch.full((3, 2), 7.0),
    }


def test_v1_6_alpha_and_parallel_up_blocks_match_native_scaling():
    modules = {
        "blocks.0.attn.proj",
        "blocks.0.attn.qkv",
        "blocks.0.cross_attn.kv_linear",
    }

    class TinyTransformer:
        def named_modules(self):
            return {name: object() for name in modules}.items()

    preprocessor = make_longcat_avatar_v1_6_lora_preprocessor(TinyTransformer())
    converted = preprocessor("longcat_avatar", _upstream_v1_6_state())

    projection_alpha = converted["diffusion_model.blocks.0.attn.proj.alpha"]
    projection_down = converted[
        "diffusion_model.blocks.0.attn.proj.lora_A.weight"
    ]
    assert float(projection_alpha / projection_down.shape[0]) == pytest.approx(0.5)
    assert torch.equal(
        converted["diffusion_model.blocks.0.attn.proj.lora_B.weight"],
        torch.full((5, 4), 2.0),
    )

    qkv_alpha = converted["diffusion_model.blocks.0.attn.qkv.alpha"]
    qkv_down = converted["diffusion_model.blocks.0.attn.qkv.lora_A.weight"]
    qkv_up = converted["diffusion_model.blocks.0.attn.qkv.lora_B.weight"]
    assert float(qkv_alpha / qkv_down.shape[0]) == pytest.approx(0.25)
    assert qkv_up.shape == (12, 6)
    assert torch.equal(qkv_up[:4, :2], torch.full((4, 2), 3.0))
    assert torch.equal(qkv_up[4:8, 2:4], torch.full((4, 2), 4.0))
    assert torch.equal(qkv_up[8:12, 4:6], torch.full((4, 2), 5.0))
    assert torch.count_nonzero(qkv_up[:4, 2:]) == 0
    assert torch.count_nonzero(qkv_up[4:8, :2]) == 0
    assert torch.count_nonzero(qkv_up[8:12, :4]) == 0

    kv_alpha = converted["diffusion_model.blocks.0.cross_attn.kv_linear.alpha"]
    kv_down = converted[
        "diffusion_model.blocks.0.cross_attn.kv_linear.lora_A.weight"
    ]
    kv_up = converted[
        "diffusion_model.blocks.0.cross_attn.kv_linear.lora_B.weight"
    ]
    assert float(kv_alpha / kv_down.shape[0]) == pytest.approx(1.25)
    assert kv_up.shape == (6, 4)
    assert torch.equal(kv_up[:3, :2], torch.full((3, 2), 6.0))
    assert torch.equal(kv_up[3:, 2:], torch.full((3, 2), 7.0))


def test_legacy_longcat_lora_state_dict_is_unchanged():
    state = {
        "blocks.0.attn.proj.lora_down.weight": torch.ones((2, 3)),
        "blocks.0.attn.proj.lora_up.weight": torch.ones((5, 2)),
        "blocks.0.attn.proj.alpha": torch.tensor(1.0),
    }
    result = preprocess_longcat_avatar_v1_6_lora(
        state,
        {"blocks.0.attn.proj"},
    )
    assert result is state
