"""Preview compatibility follows runtime registrations, without loading models."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from shared.preview_runtime import model_preview_support, rgb_preview_supported
from shared.tinyvae import decoder as tiny_decoder


RGB_HANDLER = SimpleNamespace(get_rgb_factors=lambda architecture: ([], []))
CUSTOM_HANDLER = SimpleNamespace(preview_latents=lambda *args: None)


@pytest.mark.parametrize(
    "architecture,definition,handler,expected",
    [
        ("minimax_h3", {}, None, (False, True)),
        ("minimax_h3_full", {}, None, (False, True)),
        ("minimax_h3_ref2va_full", {}, None, (False, True)),
        ("t2v", {}, RGB_HANDLER, (True, True)),
        ("ltx2", {}, RGB_HANDLER, (True, True)),
        ("ltx2_25", {}, RGB_HANDLER, (True, True)),
        ("flux", {}, RGB_HANDLER, (True, True)),
        ("flux_schnell", {}, RGB_HANDLER, (True, False)),
        ("flux2_dev", {}, RGB_HANDLER, (True, False)),
        ("flux2_klein_4b", {}, RGB_HANDLER, (True, True)),
        ("flux2_klein_9b", {}, RGB_HANDLER, (True, True)),
        ("z_image", {}, RGB_HANDLER, (True, True)),
        ("z_image_base", {}, RGB_HANDLER, (True, False)),
        ("qwen_image_20B", {}, RGB_HANDLER, (True, True)),
        ("qwen_image_edit_plus2_20B", {}, RGB_HANDLER, (True, True)),
        ("qwen_image_21_7B", {}, RGB_HANDLER, (True, False)),
        ("longcat", {}, RGB_HANDLER, (True, False)),
        ("hidream_o1", {}, CUSTOM_HANDLER, (True, False)),
        ("unknown", {}, None, (False, False)),
        ("flux", {"external_runtime": True}, RGB_HANDLER, (False, False)),
        ("minimax_h3", {"audio_only": True}, RGB_HANDLER, (False, False)),
        ("ltx2", {"ltx2_msr": True}, RGB_HANDLER, (True, False)),
        ("ltx2", {"joyai_echo": True}, RGB_HANDLER, (True, False)),
        ("ltx2", {"ltx2_edit_anything": True}, RGB_HANDLER, (True, False)),
    ],
)
def test_native_support_tracks_exact_architecture_and_contract(architecture, definition, handler, expected):
    original = dict(definition)
    rgb, tiny = expected
    assert model_preview_support(architecture, definition, handler) == {
        "rgb": rgb, "tiny_vae_frames": tiny, "tiny_vae_video": tiny,
    }
    assert definition == original


def test_catalog_support_never_downloads_loads_or_consumes_render_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Compatibility queries must not prepare a decoder or call its renderer")

    monkeypatch.setattr(tiny_decoder, "prepare_decoder", forbidden)
    monkeypatch.setattr(tiny_decoder, "load_decoder", forbidden)
    handler = SimpleNamespace(get_rgb_factors=forbidden, preview_latents=forbidden)
    rng = torch.random.get_rng_state().clone()
    support = model_preview_support("flux", {}, handler)
    assert support == {"rgb": True, "tiny_vae_frames": True, "tiny_vae_video": True}
    assert torch.equal(rng, torch.random.get_rng_state())


def test_custom_rgb_hook_and_absent_handlers_follow_runtime_gate():
    assert rgb_preview_supported(CUSTOM_HANDLER)
    assert rgb_preview_supported(RGB_HANDLER)
    assert not rgb_preview_supported(None)
    assert not rgb_preview_supported(SimpleNamespace())


def test_model_catalog_publishes_exact_variant_support(monkeypatch):
    director = ModuleType("services.director_model_compat")
    director.assess_director_model = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, director.__name__, director)
    definitions = {
        "fused-h3": {"architecture": "minimax_h3", "name": "H3 Fused"},
        "custom-flux-dev": {"architecture": "flux", "name": "Custom checkpoint"},
        "custom-flux-schnell": {"architecture": "flux_schnell", "name": "Custom checkpoint"},
        "audio-h3": {"architecture": "minimax_h3", "audio_only": True},
    }
    runtime = SimpleNamespace(
        families_infos={"unknown": (100, "Unknown"), "test": (1, "Test")},
        displayed_model_types=list(definitions),
        get_model_def=definitions.get,
        get_model_family=lambda *args, **kwargs: "test",
        get_base_model_type=lambda model: definitions[model]["architecture"],
        model_types_handlers={"flux": RGB_HANDLER, "flux_schnell": RGB_HANDLER},
        test_class_i2v=lambda model: False,
        test_class_t2v=lambda model: True,
    )
    tree = ast.parse((APP / "launch.py").read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "list_models")
    node.decorator_list = []
    namespace = {"wgp": runtime, "_check_model_downloaded": lambda model: False}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "launch.py", "exec"), namespace)
    models = {model["model_type"]: model for model in namespace["list_models"]()["models"]}
    assert models["fused-h3"]["preview_support"] == {
        "rgb": False, "tiny_vae_frames": True, "tiny_vae_video": True,
    }
    assert models["custom-flux-dev"]["preview_support"]["tiny_vae_frames"] is True
    assert models["custom-flux-schnell"]["preview_support"] == {
        "rgb": True, "tiny_vae_frames": False, "tiny_vae_video": False,
    }
    assert not any(models["audio-h3"]["preview_support"].values())
