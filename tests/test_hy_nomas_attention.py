import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_APP = str(Path(__file__).resolve().parents[1] / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from models.hy_nomas.attention import attend_projected, joint_qkv_attention, q8_prefix, quantize_heads


def test_cross_attention_handles_cfg_batch_gqa_and_padding_mask():
    torch.manual_seed(13)
    query = torch.randn((2, 3, 4, 8), dtype=torch.float32)
    key = torch.randn((1, 5, 2, 8), dtype=torch.float32)
    value = torch.randn_like(key)
    # The current Maestro API expects [batch, query, heads, key] before its
    # internal conversion to PyTorch SDPA's [batch, heads, query, key].
    mask = torch.tensor([
        [[[True, True, False, False, False]], [[True, False, True, False, False]], [[False, True, False, True, False]]],
        [[[True, False, False, True, False]], [[False, False, True, True, False]], [[True, True, True, True, False]]],
    ], dtype=torch.bool)
    actual = attend_projected(query, key, value, attention_mask=mask, force_attention="sdpa")

    expanded_key = key.expand(2, -1, -1, -1).repeat_interleave(2, dim=2)
    expanded_value = value.expand(2, -1, -1, -1).repeat_interleave(2, dim=2)
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2), expanded_key.transpose(1, 2), expanded_value.transpose(1, 2),
        attn_mask=mask.transpose(1, 2),
    ).transpose(1, 2)
    assert actual.shape == (2, 3, 4, 8)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_joint_attention_prepends_q8_prefix_to_image_and_text_streams():
    torch.manual_seed(17)
    heads, head_dim, hidden = 2, 64, 128
    projections = tuple(nn.Linear(hidden, hidden, bias=False) for _ in range(3))
    image = torch.randn((3, hidden), dtype=torch.float32)
    text = torch.randn((2, hidden), dtype=torch.float32)
    raw_key = torch.randn((1, 2, heads, head_dim), dtype=torch.float32)
    raw_value = torch.randn_like(raw_key)
    key_data, key_scales = quantize_heads(raw_key)
    value_data, value_scales = quantize_heads(raw_value)
    prefix = q8_prefix(key_data, key_scales, value_data, value_scales)

    result = joint_qkv_attention(
        [([image], projections), ([text], projections)],
        heads,
        head_dim,
        norm_rope=lambda query, key, group: None,
        kv_prefix=prefix,
    )
    assert result.shape == (1, 5, heads, head_dim)
    assert torch.isfinite(result).all()
