import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
_APP = str(_ROOT / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from models.hy_nomas.attention import attend_projected, broadcast_singleton_values
from models.hy_nomas.camera import PINNED_CAMERA_ATTENTION, validate_camera_attention_config
from models.hy_nomas.transformer import NomasCameraModule


def test_singleton_camera_softmax_equals_projected_v_for_non_square_qk():
    """The global Q/K lengths differ; each configured frame-group still has one K."""
    torch.manual_seed(121)
    batch, frames, heads, head_dim = 1, 3, 2, 8
    query = torch.randn(batch, 7, heads, head_dim)
    key = torch.randn(batch, frames, heads, head_dim)
    value = torch.randn_like(key)
    out_projection = nn.Linear(heads * head_dim, 11)
    group_sizes = (2, 4, 1)

    expected_parts, reduced_parts = [], []
    offset = 0
    for frame, query_count in enumerate(group_sizes):
        query_group = query[:, offset:offset + query_count]
        offset += query_count
        # Tencent's pinned chunk-size-one profile routes this frame's spatial
        # queries to only the corresponding keyboard token.
        exact = attend_projected(
            query_group, key[:, frame:frame + 1], value[:, frame:frame + 1],
            force_attention="sdpa",
        )
        expected_parts.append(out_projection(exact.flatten(-2)))

        projected_v = out_projection(value[:, frame].flatten(1))
        reduced_parts.append(broadcast_singleton_values(projected_v, query_count))

    exact_output = torch.cat(expected_parts, dim=1)
    reduced_output = torch.cat(reduced_parts, dim=1)
    assert exact_output.shape == (batch, 7, 11)
    assert torch.allclose(exact_output, reduced_output, atol=1e-6, rtol=1e-6)

    # The same singleton identity holds for any Q/K values, including after
    # learned normalizations or RoPE, since there is no competing key.
    changed_query = query * 7.0 + 3.0
    changed_key = key * 0.125 - 2.0
    frame = 1
    changed = attend_projected(
        changed_query[:, 2:6], changed_key[:, frame:frame + 1], value[:, frame:frame + 1],
        force_attention="sdpa",
    )
    assert torch.equal(changed, value[:, frame:frame + 1].expand(-1, 4, -1, -1))


def test_singleton_camera_broadcast_is_the_runtime_camera_module_path():
    torch.manual_seed(122)
    module = NomasCameraModule().eval()
    camera = torch.randn(3, 64)
    one_per_frame = module(camera, query_tokens_per_frame=1)
    spatial = module(camera, query_tokens_per_frame=5)
    assert spatial.shape == (3, 5, 2048)
    assert torch.equal(spatial, one_per_frame.expand(-1, 5, -1))


def test_only_pinned_singleton_camera_attention_configuration_is_accepted():
    assert validate_camera_attention_config() == PINNED_CAMERA_ATTENTION
    assert validate_camera_attention_config(dict(PINNED_CAMERA_ATTENTION)) == PINNED_CAMERA_ATTENTION

    for unsupported in (
        {"per_chunk_camera_attn": False},
        {"camera_chunk_size": 2},
        {"per_segment_camera_attn": True},
    ):
        with pytest.raises(ValueError, match="multi-key camera attention is not implemented"):
            validate_camera_attention_config(unsupported)


def test_camera_checkpoint_audit_expands_all_162_conditional_extras():
    audit_path = _ROOT / "app" / "models" / "hy_nomas" / "camera_checkpoint_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    assert audit["tencent_source"]["revision"] == "18974d41ada7af4aa3a026480197ab0ac0702682"
    assert audit["tencent_source"]["config"]["settings"] == PINNED_CAMERA_ATTENTION

    expected = {}
    for block in range(54):
        expected[f"action_modules.{block}.img_attn_q.weight"] = [1024, 2048]
        expected[f"action_modules.{block}.key_attn_q_norm.weight"] = [64]
        expected[f"action_modules.{block}.key_attn_k_norm.weight"] = [64]
    observed = {entry["key"]: entry["shape"] for entry in audit["header_parity"]["ignored_entries"]}
    assert observed == expected
    parity = audit["header_parity"]
    assert parity["header_tensor_entries"] == 3043
    assert parity["quantized_layer_count"] == 775
    assert parity["local_meta_state_entries_after_canonicalizing_quantized_weights"] == 2106
    assert parity["local_entries_missing_or_shape_mismatched"] == 0
    assert parity["additional_source_camera_entries"] == parity["ignored_entry_count"] == 162


def test_local_camera_module_loads_used_camera_weights_and_omits_proven_singleton_extras():
    module = NomasCameraModule()
    state = module.state_dict()
    assert set(state) == {
        "keyboard_embed.0.weight", "keyboard_embed.0.bias",
        "keyboard_embed.2.weight", "keyboard_embed.2.bias",
        "keyboard_attn_kv.weight", "proj_keyboard.weight",
    }
    assert state["keyboard_attn_kv.weight"].shape == (2048, 128)
    assert state["proj_keyboard.weight"].shape == (2048, 1024)
    assert not any("img_attn_q" in name or "key_attn_" in name for name in state)
