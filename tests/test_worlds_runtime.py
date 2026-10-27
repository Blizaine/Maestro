import ast
from pathlib import Path

import pytest

from app.services.worlds_runtime import (
    persist_world_sidecar, world_generation_context, world_runtime_context,
)

WORLD = {"world_model": True}
SETTINGS = {"acknowledge_pixelumm_noncommercial": True, "world_output": "world_video"}


def test_acknowledgement_required_before_model_load():
    assert world_runtime_context({}, None) == {}
    assert world_runtime_context(WORLD, SETTINGS) == {"acknowledge_pixelumm_noncommercial": True}
    assert world_generation_context(WORLD, SETTINGS) == {"world_output": "world_video"}
    for settings in (None, {}, {**SETTINGS, "acknowledge_pixelumm_noncommercial": "true"}, {**SETTINGS, "world_output": "video"}):
        with pytest.raises(ValueError):
            world_runtime_context(WORLD, settings)


def test_sidecar_is_saved_only_beside_completed_owned_media(tmp_path):
    media = tmp_path / "clip.mp4"
    with pytest.raises(ValueError):
        persist_world_sidecar(media, {".wgpworld": b"world"}, WORLD)
    media.write_bytes(b"video")
    result = persist_world_sidecar(media, {".wgpworld": b"world"}, WORLD)
    assert Path(result) == media.with_suffix(".wgpworld")
    assert Path(result).read_bytes() == b"world"
    assert persist_world_sidecar(media, {".wgpworld": b"world"}, WORLD) == result
    with pytest.raises(FileExistsError):
        persist_world_sidecar(media, {".wgpworld": b"other"}, WORLD)
    for files in ({"../state": b"bad"}, {".wgpworld": b""}, {".wgpworld": "not bytes"}, {}):
        with pytest.raises(ValueError):
            persist_world_sidecar(media, files, WORLD)
    assert not list(tmp_path.glob(".world-*.tmp"))


def test_worlds_runtime_has_no_startup_gpu_import():
    source = Path(__file__).parents[1] / "app/services/worlds_runtime.py"
    tree = ast.parse(source.read_text())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("torch" in ast.unparse(node) or "models." in ast.unparse(node) for node in imports)


def test_unknown_world_latent_layout_never_prepares_rgb_or_tiny_decoder():
    from app.shared.preview_runtime import model_preview_support, prepare_preview_decoder
    definition = {"world_model": True, "supports_latent_preview": False}
    assert not any(model_preview_support("hy_nomas", definition, object()).values())
    assert prepare_preview_decoder("tiny_vae_video", "hy_nomas", definition) == (None, None, None)
