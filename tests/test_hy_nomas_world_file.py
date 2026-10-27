import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save

_APP = str(Path(__file__).resolve().parents[1] / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from models.hy_nomas import world_file


def _jpeg(size, color):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def _world(camera=True):
    manifest = {
        "model_type": "hy_nomas",
        "width": 64,
        "height": 32,
        "fps": 24,
        "steps": 8,
        "world": {"perspective_view": "first-person", "segments": [{}, {}]},
        "generations": [{"seed": 23, "latent_frames": 3}],
        "segments": [
            {"latent_start": 0, "latent_frames": 2, "video_frames": 8, "text_offsets": [2, 5],
             "action_windows": [[0, 1, -1], [1, 2, 0]], "camera": camera, "generation": 0},
            {"latent_start": 2, "latent_frames": 1, "video_frames": 4, "text_offsets": None,
             "action_windows": None, "camera": False, "generation": 0},
        ],
        "history_frames": 18,
        "world_id": "test-world-23",
    }
    latents = torch.randn((1, 32, 3, 1, 2), dtype=torch.bfloat16)
    text_states = [torch.randn((5, 3584), dtype=torch.bfloat16), torch.randn((3, 3584), dtype=torch.bfloat16)]
    camera_tensor = torch.eye(4, dtype=torch.float32).repeat(12, 1, 1) if camera else None
    return manifest, latents, text_states, camera_tensor


def _payload(**kwargs):
    manifest, latents, text_states, camera = _world()
    manifest.update(kwargs)
    return world_file.build(manifest, latents, text_states, camera, _jpeg((32, 16), "navy"), _jpeg((64, 32), "navy"))


def test_world_file_roundtrips_bytes_path_tensors_and_thumbnail(tmp_path):
    manifest, latents, text_states, camera = _world()
    encoded = world_file.build(manifest, latents, text_states, camera, _jpeg((32, 16), "navy"), _jpeg((64, 32), "navy"))

    loaded_manifest, tensors = world_file.read(encoded)
    assert loaded_manifest["format"] == "wgpworld"
    assert loaded_manifest["version"] == 1
    assert loaded_manifest["world_id"] == "test-world-23"
    assert torch.equal(tensors["latents"], latents)
    assert torch.equal(tensors["text_0"], text_states[0])
    assert torch.equal(tensors["camera"], camera)

    path = tmp_path / "sample.wgpworld"
    path.write_bytes(encoded)
    path_manifest, path_tensors = world_file.read(path)
    assert path_manifest["world_id"] == loaded_manifest["world_id"]
    assert torch.equal(path_tensors["latents"], latents)
    assert world_file.read_thumbnail(path) == world_file.read_thumbnail(encoded)


def test_world_file_requires_wgpworld_extension(tmp_path):
    path = tmp_path / "world.zip"
    path.write_bytes(_payload())
    with pytest.raises(ValueError, match=r"must end in \.wgpworld"):
        world_file.read(path)


def test_world_file_rejects_total_size_over_limit(monkeypatch):
    monkeypatch.setattr(world_file, "MAX_WORLD_BYTES", 8)
    with pytest.raises(ValueError, match="total size limit"):
        world_file.read(b"x" * 9)


def test_world_file_rejects_unknown_duplicate_and_compressed_entries():
    payload = _payload()
    with zipfile.ZipFile(io.BytesIO(payload), "r") as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]

    extra = io.BytesIO()
    with zipfile.ZipFile(extra, "w", compression=zipfile.ZIP_STORED) as target:
        for name, data in entries:
            target.writestr(name, data)
        target.writestr("../escape", b"x")
    with pytest.raises(ValueError, match="exactly once"):
        world_file.read(extra.getvalue())

    compressed = io.BytesIO()
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in entries:
            target.writestr(name, data)
    with pytest.raises(ValueError, match="compressed"):
        world_file.read(compressed.getvalue())


def test_world_file_rejects_unsupported_manifest_version():
    payload = _payload()
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload), "r") as source, zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_STORED) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["version"] = 2
                data = json.dumps(manifest).encode("utf-8")
            target.writestr(info.filename, data)
    with pytest.raises(ValueError, match="version 1"):
        world_file.read(rebuilt.getvalue())


def test_world_file_rejects_mismatched_tensor_schema():
    manifest, latents, text_states, camera = _world()
    text_states[0] = torch.zeros((5, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="text_0"):
        world_file.build(manifest, latents, text_states, camera, _jpeg((32, 16), "navy"), _jpeg((64, 32), "navy"))


def test_world_file_rejects_wrong_safetensors_shape_before_loading():
    manifest, latents, text_states, camera = _world()
    manifest = {**manifest, "format": "wgpworld", "version": 1}
    wrong_tensors = {"latents": latents[:, :, :, :, :1].contiguous(), "text_0": text_states[0], "text_1": text_states[1], "camera": camera}
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
        archive.writestr("tensors.safetensors", save(wrong_tensors))
        archive.writestr("thumbnail.jpg", _jpeg((32, 16), "navy"))
        archive.writestr("poster.jpg", _jpeg((64, 32), "navy"))
    with pytest.raises(ValueError, match="latents"):
        world_file.read(rebuilt.getvalue())


def test_world_file_requires_camera_tensor_when_segment_is_camera_driven():
    manifest, latents, text_states, _camera = _world(camera=True)
    with pytest.raises(ValueError, match="camera tensor presence"):
        world_file.build(manifest, latents, text_states, None, _jpeg((32, 16), "navy"), _jpeg((64, 32), "navy"))


def test_world_file_rejects_nonfinite_camera_poses():
    manifest, latents, text_states, camera = _world()
    camera[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite FP32"):
        world_file.build(manifest, latents, text_states, camera, _jpeg((32, 16), "navy"), _jpeg((64, 32), "navy"))
