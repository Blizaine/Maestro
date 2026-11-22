from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


def test_zonos3_metadata_has_hindi_and_explicit_eager_settings():
    pytest.importorskip("optimum.quanto")
    from models.TTS.zonos3.zonos3_handler import family_handler
    from models.TTS.zonos3.loading import CONFIG

    languages = CONFIG["languages"]
    # The table keeps separate US/UK English presets, so 44 entries map to 43 languages.
    assert len({entry["nemo"] for entry in languages}) == 43
    assert any(entry["code"] == "hi" and entry["label"] == "Hindi" for entry in languages)

    model_def = family_handler.query_model_def("zonos3_mini", {})
    assert model_def["experimental"] is True
    assert model_def["cuda_graphs"] is False
    assert model_def["audio_only"] is True
    assert len(model_def["custom_settings"][0]["choices"]) == 27
    language_setting = next(setting for setting in model_def["custom_settings"] if setting["id"] == "language")
    assert ("Hindi", "hi") in language_setting["choices"]
    download = family_handler.query_model_files([], "zonos3_mini")[0]
    assert download["repoId"] == "DeepBeepMeep/TTS"
    assert download["revision"] == "87b4ea98d666a4f97eaaef4bfeb4e34975d7a109"
    assert download["sourceFolderList"] == ["zonos3"]
    assert download["fileList"][0] == [
        "ZAC1_keep2_decoder.safetensors", "MECT_ownmect_s2.safetensors", "voices.safetensors",
        "superbpe8192.json", "steering.safetensors",
    ]
    assert json.loads((APP / "defaults" / "zonos3_mini.json").read_text(encoding="utf-8"))["model"]["name"].endswith("(Experimental)")
    assert json.loads((APP / "defaults" / "zonos3_mini.json").read_text(encoding="utf-8"))["model"]["URLs"][0].endswith(
        "/resolve/87b4ea98d666a4f97eaaef4bfeb4e34975d7a109/ZONOS3_Mini_bf16.safetensors"
    )
    reading_heads = json.loads((APP / "models" / "TTS" / "zonos3" / "reading_heads.json").read_text(encoding="utf-8"))
    assert reading_heads["hi"]


def test_zonos3_lm_prefill_and_decode_use_eager_torch_path():
    pytest.importorskip("optimum.quanto")
    import torch
    from models.TTS.zonos3.lm import Zonos3LM

    config = {
        "n_layers": 2, "dim": 8, "head_dim": 4, "n_heads": 2, "n_kv_heads": 1,
        "norm_eps": 1e-5, "rope_theta": 10000.0, "max_seqlen": 64, "loss_softcap": 15.0,
        "global_period": 2, "gdn_heads": 1, "gdn_head_dim": 8, "gdn_kernel": 2,
        "ffn_hidden": 12, "moe_hidden": 8, "moe_routed_dim": 4, "n_experts": 4,
        "moe_topk": 2, "n_codebooks": 2, "codebook_size": 8, "eoa_id": 8,
        "audio_pad_id": 9, "text_vocab": 8, "text_rows": 16, "speaker_embedding_dim": 4,
        "condition_names": ["speaking_rate"], "condition_bucket_counts": [2],
    }
    torch.manual_seed(7)
    model = Zonos3LM(config, features=None).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.03)
        model.setup_caches(16)
        prompt_rows = torch.tensor([[1, 2, 1], [2, 3, 2], [3, 4, 3]], dtype=torch.long)
        logits = model.prefill(prompt_rows, torch.arange(3), 0)
        decoded = model.decode(torch.tensor([[4, 5, 4]]), torch.tensor([3]), 3)

    assert logits.shape == (2, config["codebook_size"] + 2)
    assert decoded.shape == logits.shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(decoded).all()


def test_zonos3_notices_record_tokenizer_and_component_licenses():
    package = APP / "models" / "TTS" / "zonos3"
    notice = (package / "NOTICE.md").read_text(encoding="utf-8")
    assert "superbpe8192.json" in notice
    assert "Hindi (`hi`)" in notice
    assert "until a real end-to-end render is verified" not in notice
    assert (package / "LICENSE_WANPG.txt").is_file()
    assert (package / "LICENSE_ZONOS3").is_file()
    assert (package / "mect" / "LICENSE_APACHE_2.0").is_file()


def test_zonos3_prompt_validation_uses_the_bundled_dialogue_parser():
    pytest.importorskip("optimum.quanto")
    from models.TTS.zonos3.zonos3_handler import family_handler

    prompt = "Speaker 1: नमस्ते।\nSpeaker 2: नमस्कार।"
    result = family_handler.validate_generative_prompt(
        "zonos3_mini", {}, {"custom_settings": {"voice": "Imogen_en-British_F_1"}}, prompt
    )
    assert result is None


def test_zonos3_optional_normalizer_preserves_unicode_and_folds_lines(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    import builtins
    from models.TTS.zonos3 import prompt as zonos_prompt

    real_import = builtins.__import__

    def import_without_textnorm(name, *args, **kwargs):
        if name == "zyphra_textnorm":
            raise ModuleNotFoundError("optional dependency missing", name="zyphra_textnorm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_textnorm)
    monkeypatch.setattr(zonos_prompt, "_textnorm_module", zonos_prompt._TEXTNORM_UNSET)
    monkeypatch.setattr(zonos_prompt, "_textnorm_warning_issued", False)
    encoder = object.__new__(zonos_prompt.TextEncoder)
    encoder._normalizer = None
    encoder.nemo = {"hi": "hi"}
    encoder.pause_markers = ["\uE003", "\uE004", "\uE005", "\uE006"]

    with pytest.warns(RuntimeWarning, match="text will be passed through"):
        spoken, cues = encoder.spoken("नमस्ते,\nदुनिया [happy]अच्छा $5.32", "hi")
    assert spoken == "नमस्ते, दुनिया \uE00Bअच्छा $5.32"
    assert cues == ["happy"]
    assert encoder.normalize("नमस्ते\n\uE004 $5.32", "hi") == "नमस्ते \uE004 $5.32"


def test_zonos3_device_resolution_prefers_mmgp_execution_hook():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from models.TTS.zonos3.device import execution_device

    module = torch.nn.Embedding(4, 4)
    module._hf_hook = SimpleNamespace(execution_device="cuda:0")
    assert execution_device(module) == torch.device("cuda:0")


def test_zonos3_lora_dir_uses_wgp_three_argument_handler_contract():
    from types import SimpleNamespace
    from models.TTS.zonos3.zonos3_handler import family_handler

    assert family_handler.get_lora_dir("zonos3_mini", SimpleNamespace(), "loras") == str(
        Path("loras") / "zonos3_mini"
    )
