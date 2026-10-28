"""Pure regressions for the path-free, review-only Editor auto-edit service."""

from __future__ import annotations

import json
import os
import sys

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.editor_auto_edit import (  # noqa: E402
    EditorAutoEditError,
    create_editor_auto_edit_proposal,
    get_editor_auto_edit_proposal,
    list_editor_auto_edit_proposals,
    save_editor_auto_edit_proposal,
)


def _project() -> dict:
    return {
        "schema_version": 6,
        "revision": 12,
        "id": "project-editor-1",
        "name": "Path-free review",
        "workspace": "workspace-a",
        "canvas": {"width": 1920, "height": 1080, "fps": 30},
        "assets": {
            "asset-video-a": {
                "id": "asset-video-a", "type": "video", "duration": 10,
                "name": r"C:\private\source-a.mp4", "path": r"C:\private\source-a.mp4",
                "url": "file:///private/source-a.mp4", "api_key": "must-not-leak",
            },
            "asset-video-b": {"id": "asset-video-b", "type": "video", "duration": 8, "name": "shot-b.mp4"},
            "asset-audio": {"id": "asset-audio", "type": "audio", "duration": 40, "name": "music.wav"},
        },
        "tracks": [
            {
                "id": "video-track-1", "type": "video", "locked": False,
                "items": [
                    {"id": "clip-a", "asset_id": "asset-video-a", "name": "A", "start": 0,
                     "duration": 4, "source_in": 0, "speed": 1, "volume": 1,
                     "director": {"pipeline_id": "director-pipeline-a", "clip_index": 1, "video_prompt": "A wide view of the field"}},
                    {"id": "clip-b", "asset_id": "asset-video-b", "name": "B", "start": 4,
                     "duration": 3, "source_in": 1, "speed": 1, "volume": 1,
                     "director": {"pipeline_id": "director-pipeline-a", "clip_index": 0, "video_prompt": r"C:\private\prompt.txt secret=hello Bearer supersecret"}},
                    {"id": "clip-linked", "asset_id": "asset-video-a", "start": 10,
                     "duration": 2, "source_in": 4, "speed": 1, "volume": 1,
                     "link_group_id": "linked-group-1",
                     "director": {"pipeline_id": "director-pipeline-a", "clip_index": 2, "video_prompt": "Linked shot"}},
                    {"id": "clip-pinned", "asset_id": "asset-video-a", "start": 14,
                     "duration": 1, "source_in": 6, "speed": 1, "volume": 1, "pinned": True,
                     "director": {"pipeline_id": "director-pipeline-a", "clip_index": 3, "video_prompt": "Pinned shot"}},
                ],
            },
            {
                "id": "audio-track-1", "type": "audio", "locked": False,
                "items": [{"id": "music-clip", "asset_id": "asset-audio", "start": 0,
                           "duration": 20, "source_in": 0, "speed": 1, "volume": 0.8}],
            },
        ],
    }


def _row(action_type: str, **values) -> dict:
    row = {
        "id": None,
        "type": action_type,
        "item_id": None,
        "asset_id": None,
        "track_id": None,
        "start": None,
        "duration": None,
        "source_in": None,
        "volume": None,
        "seconds": None,
        "reason": "A concise edit idea",
    }
    row.update(values)
    return row


def _plan(*actions: dict, summary: str = "A tighter opening", missing_media: list[str] | None = None) -> dict:
    return {"summary": summary, "actions": list(actions), "missing_media": missing_media or []}


def _writer_once(result):
    calls = []

    def writer(*, ledger, request, schema):
        calls.append((ledger, request, schema))
        return result

    return writer, calls


def test_complete_string_draft_and_malformed_response_are_retained(tmp_path):
    raw = json.dumps(_plan(_row("set-volume", item_id="music-clip", volume=.5), summary="Creative review " * 500))
    writer, _ = _writer_once(raw)
    result = create_editor_auto_edit_proposal(tmp_path, _project(), mode="polish", workspace="workspace-a",
        client_intent_id="intent-full-draft-0001", writer=writer)
    assert result["drafts"][0]["raw"] == raw
    assert result["drafts"][0]["truncated"] is False
    writer, calls = _writer_once('{"summary":"cut off')
    result = create_editor_auto_edit_proposal(tmp_path, _project(), mode="polish", workspace="workspace-a",
        client_intent_id="intent-bad-draft-0001", writer=writer)
    assert len(calls) == 2 and len(result["drafts"]) == 2
    assert all('cut off' in draft["raw"] for draft in result["drafts"])
    assert result["status"] == "needs_review" and result["errors"]


def test_draft_truncation_is_explicit_and_failed_repair_cannot_strand_intent(tmp_path):
    writer, _ = _writer_once('x' * 70000)
    result = create_editor_auto_edit_proposal(tmp_path, _project(), mode="polish", workspace="workspace-a",
        client_intent_id="intent-clipped-draft-0001", writer=writer)
    assert result["drafts"][0]["truncated"] is True
    assert len(result["drafts"][0]["raw"]) == 65536
    count = 0
    def bad_rows(**_):
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("repair unavailable")
        return _plan(*[_row("move", item_id="not-present", start=0) for _ in range(64)])
    result = create_editor_auto_edit_proposal(tmp_path, _project(), mode="polish", workspace="workspace-a",
        client_intent_id="intent-many-bad-rows-0001", writer=bad_rows)
    assert count == 2 and result["status"] == "needs_review"
    assert len(result["warnings"]) <= 64


def test_ledger_is_bounded_path_free_and_keeps_only_concise_director_prompt(tmp_path):
    writer, calls = _writer_once(_plan(_row("set-volume", item_id="music-clip", volume=0.4)))
    result = create_editor_auto_edit_proposal(
        tmp_path, _project(), mode="polish", workspace="workspace-a",
        client_intent_id="intent-path-redact-0001", client_signature="sig-client-7", writer=writer,
    )

    assert result["status"] == "ready"
    assert result["client_signature"] == "sig-client-7"
    assert result["base_revision"] == 12
    assert len(calls) == 1
    ledger = calls[0][0]
    serialized = json.dumps(ledger)
    assert "C:\\private" not in serialized
    assert "file:///" not in serialized
    assert "api_key" not in serialized
    assert "secret=hello" not in serialized
    assert "supersecret" not in serialized
    assert "[redacted path]" in serialized
    assert set(ledger["assets"][0]) == {"id", "type", "duration", "name"}
    assert calls[0][2]["additionalProperties"] is False
    assert calls[0][2]["required"] == ["summary", "actions", "missing_media"]
    assert len(calls[0][2]["properties"]["actions"]["items"]["required"]) == 11


def test_partial_valid_rows_survive_stale_ids_source_bounds_pins_and_linked_geometry(tmp_path):
    project = _project()
    project["tracks"].append({
        "id": "locked-track-1", "type": "video", "locked": True,
        "items": [{"id": "locked-clip", "asset_id": "asset-video-a", "start": 25,
                   "duration": 1, "source_in": 0, "speed": 1, "volume": 1}],
    })
    writer, calls = _writer_once(_plan(
        _row("set-volume", item_id="music-clip", volume=0.3),
        _row("move", item_id="stale-clip", start=5),
        _row("trim", item_id="clip-a", duration=9, source_in=2),
        _row("remove", item_id="clip-pinned"),
        _row("move", item_id="clip-linked", start=12),
        _row("set-volume", item_id="clip-a", volume=1.2),
        _row("move", item_id="locked-clip", start=26),
    ))
    result = create_editor_auto_edit_proposal(
        tmp_path, project, mode="revise", workspace="workspace-a",
        client_intent_id="intent-partial-0001", writer=writer,
    )

    assert result["status"] == "ready"
    assert [action["item_id"] for action in result["actions"]] == ["music-clip", "clip-a"]
    assert len(result["warnings"]) == 5
    assert any("unknown clip" in warning for warning in result["warnings"])
    assert any("source asset duration" in warning for warning in result["warnings"])
    assert any("pinned" in warning for warning in result["warnings"])
    assert any("Linked clips" in warning for warning in result["warnings"])
    assert any("locked or pinned track" in warning for warning in result["warnings"])
    assert len(calls) == 2  # one bounded repair attempt after invalid rows
    assert calls[1][1]["repair"]["invalid_action_indexes"] == [2, 3, 4, 5, 7]
    assert len(result["actions"]) == 2
    assert len({action["id"] for action in result["actions"]}) == 2


def test_one_focused_repair_adds_only_new_valid_rows_and_retains_both_drafts(tmp_path):
    first = _plan(
        _row("set-volume", item_id="music-clip", volume=0.6),
        _row("remove", item_id="does-not-exist"),
        missing_media=["rain ambience"],
    )
    repaired = _plan(
        _row("set-volume", item_id="music-clip", volume=0.6),
        _row("move", item_id="clip-b", start=9),
        missing_media=["rain ambience", "wind texture"],
    )
    calls = []

    def writer(*, ledger, request, schema):
        calls.append(request)
        return first if len(calls) == 1 else repaired

    result = create_editor_auto_edit_proposal(
        tmp_path, _project(), mode="assemble", workspace="workspace-a",
        client_intent_id="intent-repair-0001", writer=writer,
    )

    assert len(calls) == 2
    assert calls[1]["repair"]["valid_actions"][0]["volume"] == 0.6
    assert len(result["actions"]) == 2  # valid initial row is not duplicated
    assert [action["item_id"] for action in result["actions"]] == ["music-clip", "clip-b"]
    assert result["missing_media"] == ["rain ambience", "wind texture"]
    assert [draft["phase"] for draft in result["drafts"]] == ["initial", "repair"]
    assert len(result["drafts"]) == 2
    assert "does-not-exist" in result["drafts"][0]["raw"]


def test_intent_idempotence_reuses_running_or_completed_record_and_checks_signature(tmp_path):
    writer, calls = _writer_once(_plan(_row("set-volume", item_id="music-clip", volume=0.2)))
    kwargs = dict(
        mode="polish", workspace="workspace-a", client_intent_id="intent-idempotent-0001",
        client_signature="expected-signature", brief="Make the opening calmer", target_seconds=18,
    )
    first = create_editor_auto_edit_proposal(tmp_path, _project(), writer=writer, **kwargs)
    second = create_editor_auto_edit_proposal(tmp_path, _project(), writer=None, **kwargs)

    assert first == second
    assert len(calls) == 1
    assert second["brief"] == "Make the opening calmer"
    assert second["target_seconds"] == 18
    assert get_editor_auto_edit_proposal(tmp_path, first["proposal_id"]) == first
    assert len(list_editor_auto_edit_proposals(tmp_path, project_id="project-editor-1", workspace="workspace-a")) == 1
    with pytest.raises(EditorAutoEditError, match="another proposal request"):
        create_editor_auto_edit_proposal(
            tmp_path, _project(), writer=writer,
            **{**kwargs, "client_signature": "different-signature"},
        )


def test_director_order_is_deterministic_preserves_audio_and_skips_locked_linked_or_pinned(tmp_path):
    writer, calls = _writer_once(_plan())
    result = create_editor_auto_edit_proposal(
        tmp_path, _project(), mode="director", workspace="workspace-a",
        client_intent_id="intent-director-order-0001", writer=writer,
    )

    assert calls == []
    assert result["status"] == "ready"
    assert {action["item_id"]: action["start"] for action in result["actions"]} == {
        "clip-b": 0.0, "clip-a": 3.0,
    }
    assert all(action["track_id"] == "video-track-1" for action in result["actions"])
    assert "review timing" in result["warnings"][0].lower()
    assert any("left fixed" in warning for warning in result["warnings"])
    assert "music-clip" not in {action["item_id"] for action in result["actions"]}


def test_director_skips_reorder_that_overlaps_a_non_director_clip(tmp_path):
    project = _project()
    project["tracks"][0]["items"].append({
        "id": "overlay-clip", "asset_id": "asset-video-a", "start": 2.5,
        "duration": 1, "source_in": 0, "speed": 1, "volume": 1,
    })
    result = create_editor_auto_edit_proposal(
        tmp_path, project, mode="director", workspace="workspace-a",
        client_intent_id="intent-director-overlap-0001",
    )

    assert result["status"] == "needs_review"
    assert result["actions"] == []
    assert any("overlaps another clip" in warning for warning in result["warnings"])


def test_director_mixed_pipeline_track_is_preserved_with_warning(tmp_path):
    project = _project()
    project["tracks"][0]["items"][1]["director"]["pipeline_id"] = "director-pipeline-b"
    result = create_editor_auto_edit_proposal(
        tmp_path, project, mode="director", workspace="workspace-a",
        client_intent_id="intent-director-mixed-pipeline-0001",
    )

    assert result["status"] == "needs_review"
    assert result["actions"] == []
    assert any("different or unidentified Director pipelines" in warning for warning in result["warnings"])


def test_director_plan_over_64_actions_fails_before_persisting_intent(tmp_path):
    project = {
        "id": "project-large-director", "revision": 1, "workspace": "workspace-a",
        "assets": {"asset-director": {"id": "asset-director", "type": "video", "duration": 70}},
        "tracks": [{
            "id": "video-track-large", "type": "video", "locked": False,
            "items": [
                {"id": f"director-shot-{index:03d}", "asset_id": "asset-director",
                 "start": index, "duration": 1, "source_in": 0, "speed": 1, "volume": 1,
                 "director": {"pipeline_id": "director-pipeline-large", "clip_index": 69 - index,
                              "video_prompt": f"Shot {index}"}}
                for index in range(70)
            ],
        }],
    }
    with pytest.raises(EditorAutoEditError, match="more than 64 actions"):
        create_editor_auto_edit_proposal(
            tmp_path, project, mode="director", workspace="workspace-a",
            client_intent_id="intent-director-too-many-actions-0001",
        )

    assert list_editor_auto_edit_proposals(tmp_path) == []


def test_source_unknown_duration_audio_crossfade_and_shifted_pins_are_rejected(tmp_path):
    project = _project()
    project["assets"]["asset-unknown-duration"] = {
        "id": "asset-unknown-duration", "type": "video", "duration": 0, "name": "unprobed.mp4",
    }
    project["tracks"][0]["items"].append({
        "id": "clip-unprobed", "asset_id": "asset-unknown-duration", "start": 20,
        "duration": 1, "source_in": 0, "speed": 1, "volume": 1,
    })
    writer, _ = _writer_once(_plan(
        _row("set-volume", item_id="music-clip", volume=0.4),
        _row("crossfade", item_id="music-clip", seconds=0.5),
        _row("crossfade", item_id="clip-b", seconds=1),
        _row("trim", item_id="clip-unprobed", duration=0.5, source_in=0),
        _row("insert", asset_id="asset-unknown-duration", track_id="video-track-1",
             start=24, duration=1, source_in=0),
    ))

    result = create_editor_auto_edit_proposal(
        tmp_path, project, mode="revise", workspace="workspace-a",
        client_intent_id="intent-geometry-safety-0001", writer=writer,
    )

    assert [action["item_id"] for action in result["actions"]] == ["music-clip"]
    assert any("visual track" in warning for warning in result["warnings"])
    assert any("following clips" in warning for warning in result["warnings"])
    assert sum("source asset duration" in warning for warning in result["warnings"]) == 2


def test_strict_root_structure_gets_one_repair_without_losing_bounded_draft(tmp_path):
    calls = []

    def writer(*, ledger, request, schema):
        calls.append(request)
        if len(calls) == 1:
            return {
                **_plan(_row("set-volume", item_id="music-clip", volume=0.5)),
                "unapproved_tool_call": {"path": r"C:\private\secret"},
            }
        return {"actions": "malformed repair"}

    result = create_editor_auto_edit_proposal(
        tmp_path, _project(), mode="polish", workspace="workspace-a",
        client_intent_id="intent-root-repair-0001", writer=writer,
    )

    assert len(calls) == 2
    assert calls[1]["repair"]["errors"]
    assert result["status"] == "ready"
    assert len(result["actions"]) == 1
    assert result["actions"][0]["volume"] == 0.5
    stored = json.dumps(result["drafts"])
    assert "C:\\private" not in stored
    assert "unapproved_tool_call" not in stored


def test_storage_save_rejects_identity_changes_and_asset_clip_limits_are_clear(tmp_path):
    writer, _ = _writer_once(_plan(_row("set-volume", item_id="music-clip", volume=1)))
    record = create_editor_auto_edit_proposal(
        tmp_path, _project(), mode="polish", workspace="workspace-a",
        client_intent_id="intent-save-guard-0001", writer=writer,
    )
    tampered = dict(record, project_id="project-other")
    with pytest.raises(EditorAutoEditError, match="immutable"):
        save_editor_auto_edit_proposal(tmp_path, tampered)

    too_many_assets = _project()
    too_many_assets["assets"] = {
        f"asset-{index:03d}": {"id": f"asset-{index:03d}", "type": "video", "duration": 1}
        for index in range(129)
    }
    with pytest.raises(EditorAutoEditError, match="at most 128"):
        create_editor_auto_edit_proposal(
            tmp_path, too_many_assets, mode="director", workspace="workspace-a",
            client_intent_id="intent-too-many-assets-0001",
        )
    too_many_clips = _project()
    too_many_clips["tracks"][0]["items"].extend({} for _ in range(197))
    with pytest.raises(EditorAutoEditError, match="at most 200"):
        create_editor_auto_edit_proposal(
            tmp_path, too_many_clips, mode="director", workspace="workspace-a",
            client_intent_id="intent-too-many-clips-0001",
        )


def test_brief_and_target_duration_bounds_are_enforced_before_writer(tmp_path):
    writer, calls = _writer_once(_plan())
    with pytest.raises(EditorAutoEditError, match="4000 characters"):
        create_editor_auto_edit_proposal(
            tmp_path, _project(), mode="assemble", workspace="workspace-a",
            client_intent_id="intent-long-brief-0001", brief="x" * 4001, writer=writer,
        )
    with pytest.raises(EditorAutoEditError, match="86400"):
        create_editor_auto_edit_proposal(
            tmp_path, _project(), mode="assemble", workspace="workspace-a",
            client_intent_id="intent-long-target-0001", target_seconds=86401, writer=writer,
        )
    assert calls == []


def test_missing_writer_does_not_leave_a_new_running_proposal_but_existing_intent_is_recoverable(tmp_path):
    intent = "intent-writer-lifecycle-0001"
    with pytest.raises(EditorAutoEditError, match="writer is available"):
        create_editor_auto_edit_proposal(
            tmp_path, _project(), mode="assemble", workspace="workspace-a",
            client_intent_id=intent, writer=None,
        )
    assert list_editor_auto_edit_proposals(tmp_path) == []

    writer, _ = _writer_once(_plan(_row("set-volume", item_id="music-clip", volume=1)))
    created = create_editor_auto_edit_proposal(
        tmp_path, _project(), mode="assemble", workspace="workspace-a",
        client_intent_id=intent, writer=writer,
    )
    assert create_editor_auto_edit_proposal(
        tmp_path, _project(), mode="assemble", workspace="workspace-a",
        client_intent_id=intent, writer=None,
    ) == created
