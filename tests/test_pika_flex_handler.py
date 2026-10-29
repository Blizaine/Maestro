"""The experimental recipe must survive normal catalogue/settings handling."""
# ruff: noqa: E402 -- app imports follow the repository path bootstrap.
import ast
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.minimax_h3.minimax_h3_handler import (
    apply_h3_native_omni_memory_policy,
    apply_h3_window_memory_policy,
    family_handler,
    normalize_h3_window_geometry,
    validate_pika_flex_single_pass_request,
)
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
@pytest.mark.parametrize("resolution", ["832x480", "1344x768"])
def test_pika_flex_duration_restore_preserves_exact_resolution_bucket(reference, resolution):
    data, base, model = definition(reference)
    settings = {key: value for key, value in data.items() if key != "model"}
    settings.update(video_length=362, sliding_window_size=243, resolution=resolution)

    family_handler.fix_settings(base, 3, model, settings)

    assert settings["video_length"] == 362
    assert settings["sliding_window_size"] == 362
    assert settings["resolution"] == resolution
    assert family_handler.validate_generative_settings(base, model, settings) is None


@pytest.mark.parametrize("reference", [False, True])
def test_pika_flex_362_frame_duration_ignores_legacy_243_window(reference):
    data, base, model = definition(reference)
    settings = {key: value for key, value in data.items() if key != "model"}
    settings.update(
        video_length=362,
        sliding_window_size=243,
        sliding_window_overlap=18,
        minimax_h3_multi_window=True,
        minimax_h3_reference_sequence=True,
        minimax_h3_extended_duration=True,
    )

    family_handler.fix_settings(base, 3, model, settings)

    assert settings["video_length"] == 362
    assert settings["sliding_window_size"] == 362
    assert settings["sliding_window_overlap"] == 0
    assert settings["sliding_window_discard_last_frames"] == 0
    assert settings["minimax_h3_multi_window"] is False
    assert settings["minimax_h3_reference_sequence"] is False
    assert settings["minimax_h3_extended_duration"] is False
    assert family_handler.validate_generative_settings(base, model, settings) is None


@pytest.mark.parametrize("reference", [False, True])
def test_pika_flex_api_geometry_and_memory_policy_preserve_one_pass_duration(reference):
    _, _, model = definition(reference)
    request = {
        "video_length": 362,
        "sliding_window_size": 243,
        "sliding_window_overlap": 18,
        "minimax_h3_multi_window": False,
        "minimax_h3_reference_sequence": False,
    }

    validate_pika_flex_single_pass_request(request, model)
    assert apply_h3_window_memory_policy(
        request, model, {"gpu_vram_gb": 8, "ram_gb": 16},
    ) is None
    assert apply_h3_native_omni_memory_policy(
        request, model, {"gpu_vram_gb": 8, "ram_gb": 16},
    ) is None
    normalize_h3_window_geometry(request, model)

    assert request["video_length"] == 362
    assert request["sliding_window_size"] == 362
    assert request["sliding_window_overlap"] == 0
    assert request["minimax_h3_multi_window"] is False
    assert request["minimax_h3_reference_sequence"] is False


@pytest.mark.parametrize("payload", [
    {"video_length": 362, "minimax_h3_multi_window": True},
    {"video_length": 362, "minimax_h3_reference_sequence": True},
    {"video_length": 362, "multi_prompts_gen_type": 3},
    {"video_length": 362, "per_clip_frames": [243, 243]},
])
def test_pika_api_rejects_explicit_continuation_requests(payload):
    _, _, model = definition()

    with pytest.raises(ValueError, match="one native pass"):
        validate_pika_flex_single_pass_request(payload, model)


def test_pika_api_rejects_duration_above_native_limit_instead_of_shortening():
    _, _, model = definition()

    with pytest.raises(ValueError, match="at most 362 frames"):
        validate_pika_flex_single_pass_request(
            {"video_length": 363, "sliding_window_size": 243},
            model,
        )


@pytest.mark.parametrize("reference", [False, True])
def test_pika_api_rejects_explicit_extended_duration_before_geometry_normalization(reference):
    _, _, model = definition(reference)

    with pytest.raises(ValueError, match="does not support extended duration"):
        validate_pika_flex_single_pass_request(
            {"video_length": 362, "minimax_h3_extended_duration": True},
            model,
        )


def test_api_validates_pika_single_pass_before_generic_clip_schedule_repair():
    source = (ROOT / "app/launch.py").read_text(encoding="utf-8")
    prepare = source[source.index("async def _prepare_generation_submission") :]

    assert prepare.index("validate_pika_flex_single_pass_request") < prepare.index(
        "raw_h3_clip_frames = body.get(\"per_clip_frames\")"
    )


def test_api_exposes_pika_workflow_capability_to_studio():
    tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
    endpoint = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "get_model_options")
    values = [value for node in ast.walk(endpoint) if isinstance(node, ast.Dict)
              for key, value in zip(node.keys, node.values)
              if isinstance(key, ast.Constant) and key.value == "minimax_h3_pika_flex"]
    assert len(values) == 1, "Studio needs the Flex capability in the actual API response"
    expression = compile(ast.Expression(values[0]), "model-options", "eval")
    assert eval(expression, {"md": {"minimax_h3_pika_flex": True}}) is True
    assert eval(expression, {"md": {}}) is False


def test_ordinary_h3_keeps_legacy_window_duration_clamp():
    model = family_handler.query_model_def("minimax_h3", {})
    request = {
        "video_length": 362,
        "sliding_window_size": 243,
        "minimax_h3_multi_window": False,
    }

    normalize_h3_window_geometry(request, model)

    assert request["video_length"] == 243
    assert request["sliding_window_size"] == 243

    validate_pika_flex_single_pass_request(
        {"video_length": 362, "minimax_h3_multi_window": True},
        model,
    )


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
