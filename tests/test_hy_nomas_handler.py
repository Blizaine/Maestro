import os
import sys
from pathlib import Path

import pytest

_APP = str(Path(__file__).resolve().parents[1] / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from models.hy_nomas.hy_nomas_handler import (
    ASSET_REVISION,
    DECODER_FILE,
    DIT_FILE,
    DIT_LFS_SHA256,
    MODEL_TYPE,
    QWEN_ASSET_REVISION,
    TEXT_ENCODER_FILES,
    TEXT_FOLDER,
    family_handler,
)
from models.hy_nomas.hy_nomas_main import PixelUMMLicenseAcknowledgementRequired


def test_handler_declares_job_based_world_contract_and_required_companions():
    model_def = family_handler.query_model_def(MODEL_TYPE, {})
    assert os.path.join(TEXT_FOLDER, "config.json") in model_def["required_model_assets"]
    assert DECODER_FILE in model_def["required_model_assets"]
    assert len(model_def["text_encoder_URLs"]) == len(TEXT_ENCODER_FILES)
    assert all(QWEN_ASSET_REVISION in url for url in model_def["text_encoder_URLs"])
    downloads = family_handler.query_model_files(None, MODEL_TYPE, model_def)
    assert downloads[0]["revision"] == QWEN_ASSET_REVISION
    assert downloads[1]["revision"] == ASSET_REVISION
    assert model_def["hy_nomas_asset_provenance"]["dit"]["path"] == DIT_FILE
    assert model_def["hy_nomas_asset_provenance"]["dit"]["sha256"] == DIT_LFS_SHA256
    assert model_def["required_model_asset_groups"] == [[
        os.path.join(TEXT_FOLDER, name) for name in TEXT_ENCODER_FILES
    ]]
    assert model_def["world_model"] is True
    assert model_def["custom_guide"] == {"label": "Saved world", "required": False, "file_types": [".wgpworld"]}
    assert model_def["runtime_custom_settings"] == ["world_output", "acknowledge_pixelumm_noncommercial"]
    assert model_def["supports_latent_preview"] is False
    assert "world_outputs" not in model_def
    assert not hasattr(family_handler, "query_world_explorer")
    assert not hasattr(family_handler, "get_rgb_factors")
    notice = model_def["runtime_notice"].lower()
    assert "queued jobs" in notice
    assert "not real-time" in notice
    assert "qwen text encoder and tokenizer" in notice


def test_model_loading_is_blocked_without_pixelumm_noncommercial_acknowledgement():
    with pytest.raises(PixelUMMLicenseAcknowledgementRequired, match="non-commercial research or evaluation"):
        family_handler.load_model("not-read.safetensors", MODEL_TYPE, MODEL_TYPE, {}, text_encoder_filename=None)
