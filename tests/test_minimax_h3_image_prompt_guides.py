"""MiniMax H3 Image guide routing stays separate from H3 video prompting."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.enhance_guides import get_enhance_guide
from services.authoring_guides import authoring_guide_bundle


@pytest.fixture
def fake_wgp(monkeypatch):
    """Keep these enhancer tests independent of WGP's CLI-parsing module."""
    monkeypatch.setitem(sys.modules, "wgp", SimpleNamespace(get_model_def=lambda _model_type: {}))


def _guide(model_type, *, has_images=False, model_def=None):
    fake_wgp = SimpleNamespace(get_model_def=lambda _model_type: model_def or {})
    with patch.dict(sys.modules, {"wgp": fake_wgp}):
        return get_enhance_guide(model_type, "image", has_images=has_images, model_def=model_def)


def test_generation_and_edit_route_to_distinct_h3_image_guides():
    generation = _guide("h3_image", has_images=False)
    edit = _guide("h3_image", has_images=True)
    video = _guide("minimax_h3_fused_turbo", has_images=False)

    assert "MINIMAX H3 IMAGE TEXT-TO-IMAGE GUIDE" in generation
    assert "MINIMAX H3 IMAGE REFERENCE AND EDIT GUIDE" in edit
    assert "MINIMAX H3 IMAGE" not in video
    assert "minimax h3" in video.lower()
    assert "do not default to photography" in generation.lower()
    assert "<Picture 1>" in edit and "up to 16" in edit
    assert "exact requested lettering" in edit.lower()


def test_public_id_prefers_image_guide_over_broader_h3_prefix():
    guide = _guide("h3_image", has_images=False)

    assert "MINIMAX H3 IMAGE TEXT-TO-IMAGE GUIDE" in guide
    assert "MINIMAX H3 CONTEXT" not in guide


def test_h3_image_architecture_routes_even_when_public_id_has_another_prefix():
    guide = _guide(
        "registered_image_model",
        model_def={"architecture": "h3_image"},
    )

    assert "MINIMAX H3 IMAGE TEXT-TO-IMAGE GUIDE" in guide


def test_mcp_guide_bundle_routes_h3_image_generation_and_reference_edit():
    definition = {"architecture": "h3_image"}
    generation = authoring_guide_bundle(
        "h3_image", "image.generate", {}, definition,
    )
    edit = authoring_guide_bundle(
        "h3_image", "image.edit",
        {
            "references": [{"id": "portrait", "type": "image", "role": "identity"}],
            "params": {"image_refs": ["private-path.png"]},
        },
        definition,
    )

    assert "MINIMAX H3 IMAGE TEXT-TO-IMAGE GUIDE" in generation["format_guide"]
    assert "MINIMAX H3 IMAGE REFERENCE AND EDIT GUIDE" in edit["format_guide"]
    assert "up to 16" in edit["format_guide"]
    assert "IMAGE MEDIUM AND STYLE" in edit["shared_guidance"][0]


def test_studio_image_enhancer_maps_sixteen_ordered_pictures_and_keeps_exact_text(fake_wgp):
    from services import llm_service

    references = [f"h3-image-ref-{index + 1}.png" for index in range(16)]
    source = 'Combine the references as a poster. The sign reads "KEEP LEFT".'
    result_text = 'The poster combines the requested references. A sign reads "KEEP LEFT".'
    with patch.object(llm_service, "generate", return_value=result_text) as generate:
        result = llm_service.enhance_prompt(
            source,
            mode="image",
            model_type="h3_image",
            image_paths=references,
            raw_enhancer_mode=True,
        )

    assert result == result_text
    request = generate.call_args.kwargs
    assert request["image_paths"] == references
    assert source in request["prompt"]
    positions = []
    for number in range(1, 17):
        label = f"<Picture {number}> = the attached image in position {number}."
        assert label in request["prompt"]
        positions.append(request["prompt"].index(label))
    assert positions == sorted(positions)
    assert "<Picture 17>" not in request["prompt"]
    assert "MINIMAX H3 IMAGE REFERENCE AND EDIT GUIDE" in request["system_prompt"]
    assert "CRITICAL MINIMAX H3 OUTPUT CONTRACT" not in request["system_prompt"]


def test_studio_image_enhancer_without_references_uses_generation_guide(fake_wgp):
    from services import llm_service

    with patch.object(llm_service, "generate", return_value="A small blue paper crane.") as generate:
        llm_service.enhance_prompt(
            "A small blue paper crane.",
            mode="image",
            model_type="h3_image",
        )

    request = generate.call_args.kwargs
    assert "MINIMAX H3 IMAGE TEXT-TO-IMAGE GUIDE" in request["system_prompt"]
    assert "Ordered reference map" not in request["prompt"]
    assert "<Picture 1>" not in request["prompt"]
