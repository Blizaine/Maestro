"""The experimental recipe must survive normal catalogue/settings handling."""
# ruff: noqa: E402 -- app imports follow the repository path bootstrap.
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.minimax_h3.minimax_h3_handler import family_handler
from models.minimax_h3.duration import h3_duration_model_def
from services.h3_duration_recommendations import h3_duration_recommendations_enabled


def definition(reference=False):
    name = "minimax_h3_ref2va_pika_flex" if reference else "minimax_h3_pika_flex"
    data = json.loads((ROOT / "app/defaults" / (name + ".json")).read_text())
    base = data["model"]["architecture"]
    model = {**data["model"], **family_handler.query_model_def(base, data["model"])}
    return data, base, model


@pytest.mark.parametrize("reference", [False, True])
def test_recipe_and_conditioning_survive_settings_migration(reference):
    data, base, model = definition(reference)
    settings = {key: value for key, value in data.items() if key != "model"}
    settings.update(video_length=243, sliding_window_size=243, resolution="1344x768")
    settings.update(num_inference_steps=20, flow_shift=5, audio_flow_shift=5)
    if reference:
        settings["minimax_h3_references"] = [{"type": "image", "path": "identity.png"}]
    else:
        settings["image_prompt_type"] = "SE"
        settings["image_start"] = "start.png"
        settings["image_end"] = "end.png"
    family_handler.fix_settings(base, 3, model, settings)
    assert settings["video_length"] == 243
    assert settings["resolution"] == "1344x768"
    assert settings["num_inference_steps"] == 6
    assert settings["flow_shift"] == 12
    assert settings["audio_flow_shift"] == 3
    if reference:
        assert settings["minimax_h3_references"][0]["path"] == "identity.png"
    else:
        assert settings["image_prompt_type"] == "SE"
        assert settings["image_start"] == "start.png"
    assert family_handler.validate_generative_settings(base, model, settings) is None


@pytest.mark.parametrize("reference", [False, True])
def test_published_default_canvas_is_not_snapped_to_ordinary_h3(reference):
    data, base, model = definition(reference)
    settings = {key: value for key, value in data.items() if key != "model"}
    family_handler.fix_settings(base, 3, model, settings)
    assert settings["resolution"] == "832x480"
    assert family_handler.validate_generative_settings(base, model, settings) is None


@pytest.mark.parametrize("reference", [False, True])
def test_flex_has_no_borrowed_duration_evidence_or_extended_override(reference):
    _, _, model = definition(reference)
    assert not h3_duration_recommendations_enabled(model)
    assert h3_duration_model_def(model, {"minimax_h3_extended_duration": True}) is model
    assert model["inference_steps_min"] == model["inference_steps_max"] == 6
    assert not model["sliding_window"]
    assert not model["minimax_h3_two_pass_compatible"]


@pytest.mark.parametrize("change", [
    {"video_length": 123}, {"video_length": 363}, {"video_length": 125},
    {"resolution": "864x480"}, {"fps": 30}, {"num_inference_steps": 4},
    {"minimax_h3_extended_duration": True}, {"activated_loras": ["turbo.safetensors"]},
])
def test_incompatible_requests_fail_before_model_allocation(change):
    data, base, model = definition()
    settings = {key: value for key, value in data.items() if key != "model"}
    settings.update(change)
    assert family_handler.validate_generative_settings(base, model, settings)
