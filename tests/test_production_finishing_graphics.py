from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.agent_access.production_routes import ProductionService
from services.editor_projects import load_editor_project, save_editor_project
from services.production_finishing import build_finish_project


def media_resolver(reference):
    if reference in {"video", "interview.mp4"}:
        return {"name": "interview.mp4", "path": "C:/media/interview.mp4", "type": "video", "duration": 30, "width": 1920, "height": 1080, "fps": 24, "has_audio": True}
    if reference in {"logo", "logo.png"}:
        return {"name": "logo.png", "path": "C:/media/logo.png", "type": "image", "duration": 0, "width": 512, "height": 256, "fps": 0, "has_audio": False}
    if reference in {"voice-a", "voice-b", "music"}:
        return {"name": f"{reference}.wav", "path": f"C:/media/{reference}.wav", "type": "audio", "duration": 30, "width": 0, "height": 0, "fps": 0, "has_audio": True}
    raise ValueError("Media reference is not authorized.")


def graphic(elements=None):
    return {
        "version": 1,
        "width": 1344,
        "height": 768,
        "elements": elements or [{
            "id": "lower-third", "type": "rect", "x": 24, "y": 600,
            "width": 500, "height": 100, "fill": "#15202B", "opacity": 0.9,
            "border_color": "#FFFFFF", "border_width": 1, "radius": 10,
        }, {
            "id": "speaker-name", "type": "text", "x": 48, "y": 620,
            "width": 440, "height": 60, "text": "ALEX RIVERA",
            "font_family": "Arial", "font_size": 36, "color": "#FFFFFF",
            "opacity": 1, "align": "left", "bold": True,
        }],
    }


def body_with_overlay(**overlay_overrides):
    overlay = {"name": "Alex lower third", "start": 0.5, "duration": 7, "graphic": graphic()}
    overlay.update(overlay_overrides)
    return {"name": "Interview", "workspace": "default", "clips": [{"asset": "video", "duration": 10}], "overlays": [overlay]}


def test_graphic_image_references_become_project_assets_and_survive_edit_reload(tmp_path):
    body = body_with_overlay(graphic=graphic([{
        "id": "brand-mark", "type": "image", "x": 20, "y": 20,
        "width": 200, "height": 90, "asset": "logo", "opacity": 1, "fit": "contain",
    }]))
    original = copy.deepcopy(body)
    project = build_finish_project(body, media_resolver)
    assert body == original
    overlay_track = next(track for track in project["tracks"] if track["type"] == "text")
    layer = overlay_track["items"][0]["graphic"]["elements"][0]
    assert "asset" not in layer
    assert project["assets"][layer["asset_id"]]["type"] == "image"

    saved = save_editor_project(str(tmp_path), "default", project, expected_revision=0)
    loaded = load_editor_project(str(tmp_path), "default", saved["id"])
    assert loaded["tracks"][1]["items"][0]["graphic"] == overlay_track["items"][0]["graphic"]
    loaded["tracks"][1]["items"][0]["graphic"]["elements"][0]["opacity"] = 0.4
    edited = save_editor_project(str(tmp_path), "default", loaded, expected_revision=loaded["revision"])
    reloaded = load_editor_project(str(tmp_path), "default", edited["id"])
    assert reloaded["tracks"][1]["items"][0]["graphic"]["elements"][0]["opacity"] == 0.4
    json.dumps(reloaded)


def test_overlapping_graphic_overlays_are_kept_on_separate_editable_tracks():
    body = body_with_overlay()
    body["overlays"].append({
        "name": "Guest label", "start": 0.5, "duration": 7,
        "graphic": graphic([{
            "id": "guest", "type": "text", "x": 700, "y": 600,
            "width": 500, "height": 80, "text": "JAMIE LEE",
            "font_family": "Arial", "font_size": 32, "color": "#FFFFFF",
        }]),
    })
    project = build_finish_project(body, media_resolver)
    graphics = [track for track in project["tracks"] if track["type"] == "text"]
    assert [track["name"] for track in graphics] == ["Alex lower third", "Guest label"]
    assert [track["items"][0]["start"] for track in graphics] == [0.5, 0.5]
    assert graphics[0]["z_index"] < graphics[1]["z_index"]
    assert graphics[0]["items"][0]["id"] != graphics[1]["items"][0]["id"]


def test_audio_rows_keep_named_independent_speaker_tracks_and_music():
    body = {
        "clips": [{"asset": "video", "duration": 10}],
        "narration": [
            {"asset": "voice-a", "track": "Voice — Alex", "name": "Opening answer", "start": 0, "duration": 2},
            {"asset": "voice-b", "track": "Voice — Jamie", "name": "Reply", "start": 3, "duration": 2},
        ],
        "music": [{"asset": "music", "track": "Music bed", "name": "Piano", "start": 0, "duration": 10}],
    }
    project = build_finish_project(body, media_resolver)
    audio = [track for track in project["tracks"] if track["type"] == "audio"]
    assert [track["name"] for track in audio] == ["Voice — Alex", "Voice — Jamie", "Music bed"]
    assert [track["items"][0]["name"] for track in audio] == ["Opening answer", "Reply", "Piano"]
    assert min(point["value"] for point in audio[2]["items"][0]["keyframes"]["volume"]) < audio[2]["items"][0]["volume"]


def test_same_named_nonoverlapping_audio_rows_share_one_track_with_authored_starts():
    project = build_finish_project({
        "clips": [{"asset": "video", "duration": 10}],
        "narration": [
            {"asset": "voice-a", "track": "Interview VO", "start": 2, "duration": 2},
            {"asset": "voice-b", "track": "Interview VO", "start": 0, "duration": 2},
        ],
    }, media_resolver)
    audio = [track for track in project["tracks"] if track["type"] == "audio"]
    assert [track["name"] for track in audio] == ["Interview VO"]
    assert [item["start"] for item in audio[0]["items"]] == [0, 2]


def test_same_named_overlapping_audio_rows_split_into_lanes_without_shifting():
    project = build_finish_project({
        "clips": [{"asset": "video", "duration": 10}],
        "narration": [
            {"asset": "voice-a", "track": "Interview VO", "start": 0, "duration": 4},
            {"asset": "voice-b", "track": "Interview VO", "start": 2, "duration": 4},
        ],
    }, media_resolver)
    audio = [track for track in project["tracks"] if track["type"] == "audio"]
    assert [track["name"] for track in audio] == ["Interview VO", "Interview VO · 2"]
    assert [track["items"][0]["start"] for track in audio] == [0, 2]
    assert all(track["items"][0]["start"] + track["items"][0]["duration"] <= 6 for track in audio)


@pytest.mark.parametrize("overlay, message", [
    ({"name": "A", "start": 0, "duration": 2, "graphic": graphic(), "wipe": True}, "overlay"),
    ({"name": "A", "start": 9, "duration": 2, "graphic": graphic()}, "within the finished film"),
    ({"name": "A", "start": 0, "duration": 2, "graphic": graphic(), "opacity": float("nan")}, "finite number"),
    ({"name": "A", "start": 0, "duration": 2, "graphic": {**graphic(), "future": True}}, "unsupported field"),
    ({"name": "A", "start": 0, "duration": 2, "graphic": graphic(), "transform": {"x": 1, "future": 2}}, "transform"),
    ({"name": "A", "start": 0, "duration": 2, "graphic": graphic(), "keyframes": {"rotation": [{"time": 0, "value": 0}]}}, "keyframes"),
    ({"name": "A", "start": 0, "duration": 2, "graphic": graphic(), "keyframes": {"opacity": [{"time": 3, "value": 1}]}}, "finite number"),
])
def test_overlay_settings_and_graphics_reject_unknown_or_out_of_bounds_values(overlay, message):
    with pytest.raises(ValueError, match=message):
        build_finish_project({"clips": [{"asset": "video", "duration": 10}], "overlays": [overlay]}, media_resolver)


def test_graphic_image_element_cannot_supply_unowned_project_asset_id():
    body = body_with_overlay(graphic=graphic([{
        "id": "brand-mark", "type": "image", "x": 20, "y": 20,
        "width": 200, "height": 90, "asset_id": "attacker-controlled", "opacity": 1,
    }]))
    with pytest.raises(ValueError, match="authorized asset reference"):
        build_finish_project(body, media_resolver)


def test_resolved_media_metadata_must_have_finite_duration():
    def invalid_resolver(_reference):
        return {"name": "bad.mp4", "path": "C:/media/bad.mp4", "type": "video", "duration": float("nan")}

    with pytest.raises(ValueError, match="finite nonnegative number"):
        build_finish_project({"clips": [{"asset": "bad", "duration": 2}]}, invalid_resolver)


def test_finish_response_reports_editable_project_and_media_image_authorization(tmp_path, monkeypatch):
    app_root = tmp_path / "app"
    outputs = app_root / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "interview.mp4").write_bytes(b"video")
    (outputs / "logo.png").write_bytes(b"image")
    private_image = tmp_path / "private.png"
    private_image.write_bytes(b"private")
    backend = SimpleNamespace(
        app_root=app_root,
        runtime=SimpleNamespace(directory=app_root / "settings"),
        jobs={},
        additional_media_roots=lambda: (),
        snapshot_job=lambda _job: None,
    )
    service = ProductionService(
        backend, output_root=lambda: outputs,
        export_editor=lambda _body: {"job_id": "unused"},
        rerun_director=lambda *_args: {}, director_snapshot=lambda _pid: {"clips": []},
    )

    def probe(path):
        is_image = Path(path).suffix.lower() == ".png"
        return {"name": Path(path).name, "path": path, "type": "image" if is_image else "video",
                "duration": 0 if is_image else 10, "width": 64, "height": 64, "fps": 24,
                "has_audio": False}

    monkeypatch.setattr("services.editor_projects.probe_media", probe)
    body = {
        "workspace": "default", "clips": [{"asset": {"filename": "interview.mp4"}, "duration": 10}],
        "overlays": [{"name": "Brand", "start": 0, "duration": 3,
                      "graphic": graphic([{"id": "logo", "type": "image", "x": 0, "y": 0,
                          "width": 100, "height": 50, "asset": {"filename": "logo.png"}, "opacity": 1}])}],
    }
    result = asyncio.run(service.finish(body, SimpleNamespace(id="agent", scopes=("manage",))))
    assert result["editable"] is True
    assert result["project_id"] and result["revision"] == 1
    assert any(track["name"] == "Brand" and track["type"] == "text" for track in result["tracks"])
    body["overlays"][0]["graphic"]["elements"][0]["asset"] = str(private_image)
    with pytest.raises(ValueError, match="outside approved Maestro storage"):
        asyncio.run(service.finish(body, SimpleNamespace(id="agent", scopes=("manage",))))
