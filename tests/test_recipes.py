from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services import recipes  # noqa: E402
from services.agent_access.guides import get_guide  # noqa: E402


@pytest.fixture
def recipe_dirs(tmp_path, monkeypatch):
    user_dir = tmp_path / "user"
    bundled_dir = tmp_path / "bundled"
    user_dir.mkdir()
    bundled_dir.mkdir()
    monkeypatch.setattr(recipes, "USER_DIR", str(user_dir))
    monkeypatch.setattr(recipes, "BUNDLED_DIR", str(bundled_dir))
    return user_dir, bundled_dir


def legacy_recipe(**overrides):
    value = {
        "recipe_version": 1,
        "name": "My Look",
        "description": "A saved look",
        "mode": "video",
        "model_type": "test_video_model",
        "loras": [{"filename": "look.safetensors", "multiplier": "0.8"}],
        "params": {
            "resolution": "960x544",
            "num_inference_steps": 8,
            "seed": 42,
        },
        "prompt_example": "A test subject in a blue studio",
        "nsfw": False,
    }
    value.update(overrides)
    return value


def test_v1_import_keeps_apply_fields_and_adds_safe_guidance(recipe_dirs):
    card = recipes.import_recipe(legacy_recipe())
    loaded = recipes.get_recipe(card["id"])

    assert loaded["recipe_version"] == 1
    assert loaded["model_type"] == "test_video_model"
    assert loaded["mode"] == "video"
    assert loaded["params"] == {"resolution": "960x544", "num_inference_steps": 8}
    assert loaded["prompt_example"] == "A test subject in a blue studio"
    assert loaded["loras"] == [{"filename": "look.safetensors", "multiplier": "0.8"}]
    assert loaded["requirements"] == ["Model: test_video_model", "LoRA available: look.safetensors"]
    assert loaded["workflow_steps"]
    assert card["source"] == "user"


def test_import_accepts_bounded_declarative_steps_and_discards_unknown_fields(recipe_dirs):
    card = recipes.import_recipe(legacy_recipe(
        recipe_version=2,
        requirements=["Model is installed"],
        model_guidance="Use the listed model.",
        prompt_guidance="Keep the style token.",
        workflow_guidance="Review each result before iterating.",
        workflow_steps=[{
            "type": "configure",
            "title": "Set the resolution",
            "instruction": "Use the suggested output size.",
            "setting": "resolution",
            "value": "960x544",
            "script": "this field is discarded",
        }],
        arbitrary_code="not stored",
    ))
    loaded = recipes.get_recipe(card["id"])

    assert loaded["requirements"] == ["Model is installed"]
    assert loaded["workflow_steps"] == [{
        "type": "configure",
        "title": "Set the resolution",
        "instruction": "Use the suggested output size.",
        "setting": "resolution",
        "value": "960x544",
    }]
    assert "arbitrary_code" not in loaded


@pytest.mark.parametrize("steps", [
    [{"type": "run_script", "title": "Run", "instruction": "execute code"}],
    [{"type": "configure", "title": "Set it", "instruction": "Change it", "setting": "seed"}],
    [{"type": "generate", "title": "x" * 101, "instruction": "Render"}],
    [{"type": "generate", "title": "Too much", "instruction": "x" * 601}],
    [{"type": "generate", "title": f"Step {i}", "instruction": "Render"} for i in range(13)],
    [{"type": "configure", "title": "Bad value", "instruction": "Set it", "setting": "resolution", "value": {"code": "x"}}],
])
def test_import_rejects_unsupported_or_unbounded_steps(recipe_dirs, steps):
    with pytest.raises(ValueError):
        recipes.import_recipe(legacy_recipe(recipe_version=2, workflow_steps=steps))


@pytest.mark.parametrize("fields", [
    {"requirements": ["ready"] * 13},
    {"requirements": ["x" * 241]},
    {"prompt_guidance": "x" * 2401},
])
def test_import_rejects_unbounded_recipe_guidance(recipe_dirs, fields):
    with pytest.raises(ValueError):
        recipes.import_recipe(legacy_recipe(recipe_version=2, **fields))


def test_video_save_writes_thumbnail_and_short_preview_then_delete_cleans_both(recipe_dirs, tmp_path, monkeypatch):
    user_dir, _ = recipe_dirs
    source = tmp_path / "render.mp4"
    source.write_bytes(b"source")
    calls = []

    def fake_ffmpeg(command, **_kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"media")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(recipes.subprocess, "run", fake_ffmpeg)
    card = recipes.save_recipe_from_params(
        "Sample recipe", "A description", {"model_type": "model-a", "resolution": "640x360"},
        "video", [], "A prompt", source_media=str(source),
    )

    json_path = user_dir / f"{card['id']}.json"
    thumb_path = user_dir / f"{card['id']}.jpg"
    preview_path = user_dir / f"{card['id']}.preview.mp4"
    assert card["preview_url"].endswith("?preview=true")
    assert recipes.get_recipe_preview_path(card["id"]) == str(preview_path)
    assert thumb_path.is_file() and preview_path.is_file()
    assert "-t" in calls[1] and "4" in calls[1]
    assert "-an" in calls[1] and "libx264" in calls[1]
    assert recipes.delete_recipe(card["id"])
    assert not json_path.exists() and not thumb_path.exists() and not preview_path.exists()


def test_image_recipe_has_no_video_preview(recipe_dirs, tmp_path, monkeypatch):
    image = tmp_path / "render.png"
    image.write_bytes(b"image")

    def fake_ffmpeg(command, **_kwargs):
        Path(command[-1]).write_bytes(b"thumbnail")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(recipes.subprocess, "run", fake_ffmpeg)
    card = recipes.save_recipe_from_params(
        "Image recipe", "", {"model_type": "image-model"}, "image", [], "A prompt", source_media=str(image),
    )
    assert card["thumbnail_url"]
    assert card["preview_url"] is None
    assert recipes.get_recipe_preview_path(card["id"]) is None


def test_mcp_recipe_guide_describes_existing_discovery_and_review_flow():
    guide = get_guide("recipe")
    assert "GET /api/v1/recipes/{rid}" in guide
    assert "preview=true" in guide
    assert "never scripts" in guide
    assert "prepared-generation flow" in guide
    from services.agent_access.catalog import operation_domain, required_scope

    assert operation_domain("/api/v1/recipes") == "assets"
    assert operation_domain("/api/v1/recipes/example") == "assets"
    assert required_scope("GET", "/api/v1/recipes/example") == "observe"
