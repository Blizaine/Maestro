import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.fibo2.attention import attend_projected
from models.fibo2.pipeline import _pil_image, pack, unpack, validate_generation_inputs
from models.fibo2.transformer import BriaFibo2Attention


def _sdpa_reference(query, key, value, mask):
    batch, _, query_heads, _ = query.shape
    kv_heads = key.shape[2]
    if kv_heads != query_heads:
        groups = query_heads // kv_heads
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)
    if mask.ndim == 2:
        mask = mask[:, None, None, :].bool()
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        attn_mask=mask,
    )
    return output.transpose(1, 2)


def test_projected_attention_non_square_padding_and_gqa_matches_sdpa():
    torch.manual_seed(19)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 5, 2, 8)
    value = torch.randn(2, 5, 2, 8)
    # Integer tokenizer masks are common; zeros must remain blocked after the
    # adapter reaches Maestro's SDPA wrapper.
    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 1, 0]])

    result = attend_projected(query, key, value, attention_mask=mask)
    expected = _sdpa_reference(query, key, value, mask)

    assert result.shape == (2, 3, 4, 8)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_projected_attention_additive_mask_and_broadcast_key_batch():
    torch.manual_seed(29)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(1, 5, 2, 8)
    value = torch.randn(1, 5, 2, 8)
    mask = torch.zeros(2, 4, 3, 5)
    mask[:, :, :, -1] = -10000

    result = attend_projected(query, key, value, attention_mask=mask)
    expected = _sdpa_reference(query, key.expand(2, -1, -1, -1),
                               value.expand(2, -1, -1, -1), mask)

    assert result.shape == (2, 3, 4, 8)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_projected_attention_preserves_qwen_maestro_mask_layout():
    torch.manual_seed(31)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 5, 4, 8)
    value = torch.randn(2, 5, 4, 8)
    # Qwen3-VL's explicit text mask is already [batch, query, heads, key].
    # Keep query and key lengths different to catch accidental transposes.
    mask = torch.zeros(2, 3, 1, 5, dtype=torch.bool)
    mask[:, :, :, -1] = True

    result = attend_projected(query, key, value, attention_mask=mask,
                              attention_mask_layout="maestro")
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        attn_mask=mask.transpose(1, 2),
    ).transpose(1, 2)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_cross_attention_cfg_batch_preserves_per_item_results():
    torch.manual_seed(41)
    attention = BriaFibo2Attention(query_dim=16, heads=4, dim_head=4, kv_dim=24).eval()
    # Four rows exercise a batched conditional/unconditional pair plus a
    # second prompt pair, each with distinct key padding.
    query = torch.randn(4, 3, 16)
    context = torch.randn(4, 5, 24)
    mask = torch.tensor([
        [1, 1, 1, 0, 0], [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0], [1, 0, 0, 0, 0],
    ])

    batched = attention(query, encoder_hidden_states=context, attention_mask=mask)
    individual = torch.cat([
        attention(query[index:index + 1], encoder_hidden_states=context[index:index + 1],
                  attention_mask=mask[index:index + 1])
        for index in range(query.shape[0])
    ])

    assert batched.shape == (4, 3, 16)
    torch.testing.assert_close(batched, individual, rtol=1e-5, atol=1e-6)


def test_invalid_attention_mask_is_rejected_instead_of_dropped():
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 3, 2, 4)
    value = torch.randn_like(key)
    with pytest.raises(ValueError, match="key length"):
        attend_projected(query, key, value, attention_mask=torch.ones(1, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="Q/K/V head dimensions"):
        attend_projected(query, key, torch.randn(1, 3, 3, 4))


def test_fibo_latent_pack_round_trip():
    source = torch.arange(2 * 32 * 8 * 10).reshape(2, 32, 8, 10)
    packed = pack(source)
    assert packed.shape == (2, 128, 4, 5)
    assert torch.equal(unpack(packed), source)


@pytest.mark.parametrize("width,height", [(1024, 1024), (1152, 1024), (16, 32)])
def test_valid_fibo_geometry_and_cfg_batch(width, height):
    assert validate_generation_inputs(width, height, 4, 2) == (width, height, 4, 2)


@pytest.mark.parametrize("args, message", [
    ((1025, 1024, 50, 1), "multiples of 16"),
    ((16, 8193, 50, 1), "up to 8192"),
    ((16, 16, 0, 1), "sampling steps"),
])
def test_invalid_fibo_geometry_is_rejected(args, message):
    with pytest.raises(ValueError, match=message):
        validate_generation_inputs(*args)


def test_masked_and_turbo_edit_require_their_sources():
    with pytest.raises(ValueError, match="exactly one source image"):
        validate_generation_inputs(1024, 1024, 4, 1, references=[1, 2], has_mask=True)
    with pytest.raises(ValueError, match="requires a source image"):
        validate_generation_inputs(1024, 1024, 4, 1, edit_only=True)


def test_turbo_masked_control_image_path_does_not_require_reference_images():
    from models.fibo2.fibo2_handler import family_handler

    model_def = family_handler.query_model_def("fibo2_edit_turbo", {})
    inputs = {"image_mode": 2, "image_refs": [], "video_prompt_type": "VA", "guidance_scale": 1.0}
    assert model_def["at_least_one_image_ref_needed"] is False
    assert family_handler.validate_generative_settings("fibo2_edit_turbo", model_def, inputs) is None

    # WGP supplies its inpaint Control Image as input_frames shaped CxTxHxW.
    source = torch.zeros(3, 1, 16, 16)
    source[0] = 1
    converted = _pil_image(source)
    assert converted.size == (16, 16)
    assert converted.getpixel((0, 0)) == (255, 127, 127)


def test_fibo_accepts_wgp_raw_and_structured_prompt_forms():
    from models.fibo2.caption import Caption, VISION_MARKER

    raw, instruction = Caption._parse_prompt(
        "A bright poster with text #4, blue ink.\n # ignore this note\n"
    )
    assert raw == "A bright poster with text #4, blue ink."
    assert instruction is None

    prompt = json.dumps({
        "structured_caption": {
            "short_description": "A cat with a tag reading MILO",
            "text_render": [{"text": "MILO"}],
        },
        "edit_instruction": "Add the tag to <image_1> and preserve the cat.",
    })
    parsed, edit_instruction = Caption._parse_prompt(prompt)
    assert parsed["text_render"] == [{"text": "MILO"}]
    assert edit_instruction == "Add the tag to <image_1> and preserve the cat."
    edit_prompt = json.loads(Caption._get_edit_prompt(prompt, 1))
    assert edit_prompt["image"] == VISION_MARKER
    assert edit_prompt["structured_caption"]["text_render"][0]["text"] == "MILO"
    assert edit_prompt["edit_instruction"] == "Add the tag to <image_1> and preserve the cat."


def test_defaults_and_handler_pin_companion_asset_contract():
    dev = json.loads((ROOT / "app/defaults/fibo2_dev.json").read_text(encoding="utf-8"))
    turbo = json.loads((ROOT / "app/defaults/fibo2_edit_turbo.json").read_text(encoding="utf-8"))
    assert dev["num_inference_steps"] == 50 and dev["guidance_scale"] == 5.0
    assert turbo["num_inference_steps"] == 4 and turbo["guidance_scale"] == 1.0
    assert all("/resolve/72bf0a84dd97688a4e9427a8c872114b1162c4b8/" in url
               for url in dev["model"]["URLs"] + turbo["model"]["URLs"])
    assert all("CC BY-NC 4.0" in item["model"]["description"] for item in (dev, turbo))
    for item in (dev, turbo):
        assert item["model"]["license_name"] == "CC BY-NC 4.0 (Bria Fibo weights)"
        assert "briaai/Fibo-2.0" in item["model"]["license_url"]
        assert "non-commercial use only" in item["model"]["selector_help"]
        assert "separate Bria agreement" in item["model"]["selector_help"]

    from models.fibo2.fibo2_handler import REPO_REVISION, family_handler

    assert REPO_REVISION == "72bf0a84dd97688a4e9427a8c872114b1162c4b8"
    assert family_handler.get_lora_dir("fibo2_dev", None, "loras") == str(Path("loras") / "fibo2")
    model_def = family_handler.query_model_def("fibo2_dev", {})
    assert model_def["instruction_image_edit"] is True
    assert model_def["at_least_one_image_ref_needed"] is False
    assert all("/resolve/72bf0a84dd97688a4e9427a8c872114b1162c4b8/" in url
               for url in model_def["text_encoder_URLs"])
    assert model_def["max_image_refs"] == 4
    assert "Qwen3-VL-4B-Instruct/tokenizer.json" in model_def["required_model_assets"]
    assert "fibo2/flux2_vae_fp16.safetensors" in model_def["required_model_assets"]
    assert model_def["required_model_asset_groups"] == [[
        "Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_bf16.safetensors",
        "Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_int8_convrot.safetensors",
    ]]
    assets = family_handler.query_model_files([], "fibo2_dev")[0]
    assert assets["revision"] == REPO_REVISION
    assert "fibo2_encoder_config.json" in assets["fileList"][0]
    assert "tokenizer.json" in assets["fileList"][0]
    assert "flux2_vae_fp16.safetensors" in assets["fileList"][1]


def test_required_asset_readiness_names_encoder_tokenizer_and_vae(monkeypatch):
    from models.fibo2 import pipeline

    monkeypatch.setattr(pipeline.fl, "locate_file", lambda *_args, **_kwargs: None)
    with pytest.raises(FileNotFoundError) as error:
        pipeline._required_assets("Qwen3-VL-4B-Instruct")
    message = str(error.value)
    assert "fibo2_encoder_config.json" in message
    assert "tokenizer.json" in message
    assert "flux2_vae_fp16.safetensors" in message

    with pytest.raises(FileNotFoundError, match="Qwen3-VL-4B encoder weights"):
        pipeline._resolve_text_encoder_checkpoint(None, "Qwen3-VL-4B-Instruct")


def test_component_provenance_logs_local_paths_and_model_root(monkeypatch, capsys):
    from models.fibo2 import pipeline

    monkeypatch.setattr(pipeline.fl, "describe_file_source", lambda _path: {
        "kind": "linked", "root": "D:\\models\\Maestro\\ckpts",
    })
    pipeline._log_component_paths("Qwen3-VL encoder", [r"D:\\models\\Maestro\\ckpts\\Qwen\\model.safetensors"])
    output = capsys.readouterr().out
    assert "Qwen3-VL encoder" in output
    assert "D:\\models\\Maestro\\ckpts\\Qwen\\model.safetensors" in output
    assert "linked model root D:\\models\\Maestro\\ckpts" in output
    assert "https://" not in output


def test_fibo_notice_records_source_revision_and_weight_terms():
    notice = (ROOT / "app/models/fibo2/NOTICE.txt").read_text(encoding="utf-8")
    assert "ab96e2b58c89adb9df418af74d81cc8b1e1be074" in notice
    assert "72bf0a84dd97688a4e9427a8c872114b1162c4b8" in notice
    assert "CC BY-NC 4.0" in notice
    assert "commercial use requires a" in notice.lower()
    assert "7281a4704a5716e40eea6d247ef4e1df5dfae7a5d266f4589738d484104d2331" in notice
